"""Five-second torque-control diagnostic at a freshly solved grasp pose.

Run from the complete lab project: python check_hold.py
Uses existing scene/contact settings. Does not attach, lift, or reposition cube.
Joint state is reset once for initialization, never inside the control loop.
Output is diagnostic evidence, not a grasp or coursework pass certificate.
"""

import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pinocchio as pin

from config import CUBE_PLACEMENT, DT
from controller import compute_torque
from inverse_geometry import computeqgrasppose
from tools import setupwithpybullet, collision, jointlimitsviolated

DURATION = 5.0
# Diagnostic stop thresholds; these are not robot hardware specifications.
MAX_POSITION_ERROR = 0.35
MAX_JOINT_SPEED = 5.0


def main():
    if not np.isfinite(DT) or DT <= 0:
        raise ValueError("DT must be finite and positive")
    robot, sim, cube = setupwithpybullet()
    print("[HOLD] PyBullet loaded; Meshcat server is not needed.")
    print("[HOLD] Solving initial grasp only; path.py is not used.")
    q0, success = computeqgrasppose(
        robot, robot.q0.copy(), cube, CUBE_PLACEMENT, None
    )
    q0 = np.asarray(q0, dtype=float).copy()
    if q0.shape != (robot.model.nq,) or not np.all(np.isfinite(q0)):
        raise RuntimeError("Solver returned invalid q0")
    if not success or jointlimitsviolated(robot, q0) or collision(robot, q0):
        raise RuntimeError("Initial grasp is invalid; no simulation steps executed")

    names = list(robot.model.names[1:])
    joints = sim.bulletCtrlJointsInPinOrder
    if robot.model.nq != robot.model.nv or len(joints) != robot.model.nv:
        raise RuntimeError("This diagnostic expects the fixed-base revolute-joint model")
    actual_names = [sim.p.getJointInfo(sim.robot, j)[1].decode() for j in joints]
    if actual_names != names:
        raise RuntimeError("PyBullet and Pinocchio joint orders disagree")
    # Uploaded setqsim uses bodyUniqueId=1; refuse an incompatible scene.
    if sim.robot != 1:
        raise RuntimeError("Provided setqsim hard-codes body ID 1; actual robot ID differs")

    # Respect positive finite URDF torque limits. Missing limits are disclosed.
    raw_limits = np.array([sim.p.getJointInfo(sim.robot, j)[10] for j in joints])
    known_limits = np.isfinite(raw_limits) & (raw_limits > 0)
    torque_limits = np.where(known_limits, raw_limits, np.inf)
    print(f"[HOLD] Verified joint order: {len(joints)} joints")
    print(f"[HOLD] URDF torque limits available: {known_limits.sum()}/{len(joints)}")
    if not np.all(known_limits):
        print("[HOLD] Missing torque limits: " + ", ".join(
            n for n, valid in zip(names, known_limits) if not valid
        ))
        print("[HOLD] Those joints have no added torque cap; divergence stops remain active.")

    sim.setqsim(q0)  # One initialization only; normal dynamics from here onward.
    q, v = sim.getpybulletstate()
    np.testing.assert_allclose(q, q0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(v, 0, atol=1e-9, rtol=0)
    base_start, base_orientation = sim.p.getBasePositionAndOrientation(sim.robot)
    base_start = np.asarray(base_start)
    base_mass = sim.p.getDynamicsInfo(sim.robot, -1)[0]
    print(f"[HOLD] Bullet base mass: {base_mass:g} kg; monitoring base drift")
    zero = np.zeros(robot.model.nv)
    cube_z_start = sim.p.getBasePositionAndOrientation(sim.cubeId)[0][2]

    output_dir = Path(__file__).resolve().parent / "part_b_logs"
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / ("hold_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")
    print(f"[HOLD] CSV: {filename}")
    steps = int(np.ceil(DURATION / DT))
    print_every = max(1, round(0.5 / DT))
    tail_errors, tail_speeds = [], []
    clipped_steps = 0
    peak_error = 0.0
    completed = False

    with filename.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "max_error_rad", "max_speed_rad_s", "max_torque_Nm",
             "torque_clipped", "base_drift_m", "base_rotation_rad", "cube_z_m"]
            + ["q_" + n for n in names]
            + ["v_" + n for n in names]
            + ["tau_command_" + n for n in names]
            + ["q_des_" + n for n in names]
        )
        try:
            for k in range(steps + 1):
                tick_start = time.perf_counter()
                t = k * DT
                q, v = sim.getpybulletstate()
                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                    raise RuntimeError("Non-finite simulator state")
                error = float(np.max(np.abs(pin.difference(robot.model, q, q0))))
                speed = float(np.max(np.abs(v)))
                base_pos, base_rot = sim.p.getBasePositionAndOrientation(sim.robot)
                base_drift = float(np.linalg.norm(np.asarray(base_pos) - base_start))
                rotation = 2 * np.arccos(np.clip(abs(np.dot(base_rot, base_orientation)), 0, 1))
                cube_z = sim.p.getBasePositionAndOrientation(sim.cubeId)[0][2]
                tau_raw = compute_torque(robot, q, v, q0, zero, zero)
                tau = np.clip(tau_raw, -torque_limits, torque_limits)
                clipped = bool(np.any(np.abs(tau_raw) > torque_limits))
                writer.writerow(
                    [t, error, speed, float(np.max(np.abs(tau))), int(clipped),
                     base_drift, rotation, cube_z]
                    + list(q) + list(v) + list(tau) + list(q0)
                )
                peak_error = max(peak_error, error)
                if t >= DURATION - 1.0:
                    tail_errors.append(error)
                    tail_speeds.append(speed)
                if k % print_every == 0 or k == steps:
                    print(f"t={t:.2f}s  max_error={error:.6f}rad  "
                          f"max_speed={speed:.6f}rad/s  "
                          f"base_drift={base_drift:.5f}m  clipped={clipped}")
                    stream.flush()
                if error > MAX_POSITION_ERROR or speed > MAX_JOINT_SPEED:
                    raise RuntimeError("Joint error/speed exceeded diagnostic stop threshold")
                if base_drift > 0.02 or rotation > 0.05:
                    raise RuntimeError("Base moved: fixed-base dynamics assumption needs review")
                if k == steps:
                    completed = True
                    break
                sim.step(tau.tolist())
                if clipped:
                    clipped_steps += 1
                # Pacing affects display only; simulation time always advances by DT.
                time.sleep(max(0.0, DT - (time.perf_counter() - tick_start)))
        except BaseException:
            print(f"[HOLD STOPPED] Partial log saved: {filename}")
            raise

    print(f"[HOLD] completed={completed}; simulated_time={steps * DT:.3f}s")
    print(f"[HOLD] peak_joint_error={peak_error:.6f} rad")
    print(f"[HOLD] last_1s_max_error={max(tail_errors):.6f} rad")
    print(f"[HOLD] last_1s_max_speed={max(tail_speeds):.6f} rad/s")
    print(f"[HOLD] torque_clipped_steps={clipped_steps}/{steps}")
    print(f"[HOLD] cube_z_change={cube_z - cube_z_start:.6f} m (not a grasp test)")
    print("[HOLD] Diagnostic finished; inspect errors and CSV before judging stability.")
    print("[HOLD] Simulation is paused. This does not certify grasping or transport.")
    try:
        input("Press Enter to close the PyBullet window...")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
