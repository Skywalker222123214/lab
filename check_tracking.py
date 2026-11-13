"""CW2 empty-hand motion diagnostic; run in the full lab project.

python check_tracking.py

Virtual cube poses are used only to solve coordinated hand configurations.
The physical cube stays in the supplied scene. This is not a lift/grasp test.
Existing trajectory.py, controller.py and repaired setup_pybullet.py are used.
"""

import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pinocchio as pin

from config import CUBE_PLACEMENT, DT, LEFT_HAND, RIGHT_HAND
from controller import compute_torque
from inverse_geometry import computeqgrasppose
from tools import (setupwithpybullet, setcubeplacement, collision,
                   jointlimitsviolated)
from trajectory import PathTrajectory

INITIAL_HOLD = 1.0
FINAL_HOLD = 1.0
HOVER_HEIGHT = 0.08
UPWARD_DISTANCE = 0.02
# Experimental stop limits, not robot hardware specifications.
MAX_ERROR = 0.15
MAX_SPEED = 5.0


def build_reference(robot, cube):
    """Solve two nearby, unloaded hand poses and check sampled segments."""
    hover = CUBE_PLACEMENT.copy()
    hover.translation = CUBE_PLACEMENT.translation + np.array([0., 0., HOVER_HEIGHT])
    upper = hover.copy()
    upper.translation = hover.translation + np.array([0., 0., UPWARD_DISTANCE])
    print("[TRACK] Solving virtual grasp 8 cm above the physical cube")
    try:
        q_start, ok_start = computeqgrasppose(
            robot, robot.q0.copy(), cube, hover, None
        )
        if not ok_start:
            raise RuntimeError("Hover pose solver failed; simulation has not stepped")
        q_start = np.asarray(q_start, dtype=float).copy()
        print("[TRACK] Solving a second virtual grasp 2 cm higher")
        q_upper, ok_upper = computeqgrasppose(
            robot, q_start.copy(), cube, upper, None
        )
        if not ok_upper:
            raise RuntimeError("Upper pose solver failed; simulation has not stepped")
        q_upper = np.asarray(q_upper, dtype=float).copy()
    finally:
        # Restore the Pinocchio cube geometry to the real scene before checking.
        # No PyBullet cube pose is reset or moved by this program.
        setcubeplacement(robot, cube, CUBE_PLACEMENT)

    if (q_start.shape != (robot.model.nq,) or q_upper.shape != q_start.shape
            or not np.all(np.isfinite(q_start)) or not np.all(np.isfinite(q_upper))):
        raise RuntimeError("Invalid configuration returned by grasp solver")
    if np.max(np.abs(q_upper - q_start)) > 0.25:
        raise RuntimeError("Large IK branch change for small motion; no steps executed")
    trajectory = PathTrajectory(
        [q_start, q_upper, q_start], max_velocity=0.15,
        max_acceleration=0.3, min_segment_time=2.5
    )
    samples = 0
    for i, delta in enumerate(trajectory.deltas):
        intervals = max(10, int(np.ceil(np.max(np.abs(delta)) / 0.002)))
        for fraction in np.linspace(0., 1., intervals + 1):
            q = trajectory.points[i] + fraction * delta
            if jointlimitsviolated(robot, q) or collision(robot, q):
                raise RuntimeError(f"Reference invalid on segment {i}; no steps executed")
            samples += 1
    print(f"[TRACK] {samples} reference configurations passed limits/collision checks")
    print("[TRACK] Discrete checks do not certify continuous-time collision avoidance")
    return trajectory


def contact_summary(sim):
    labels = {sim.robot: "robot", sim.tableId: "table", sim.cubeId: "cube",
              sim.obstacleId: "obstacle", sim.floor: "floor"}
    for c in sim.p.getContactPoints(bodyA=sim.robot):
        if c[9] <= 1e-4:
            continue
        link = "base" if c[3] == -1 else sim.p.getJointInfo(sim.robot, c[3])[12].decode()
        print(f"[CONTACT] {link} -> {labels.get(c[2], str(c[2]))}: {c[9]:.4f} N")


def main():
    if not np.isfinite(DT) or DT <= 0:
        raise ValueError("DT must be finite and positive")
    robot, sim, cube = setupwithpybullet()
    if not hasattr(sim, "dynamics_sync_marker_masses"):
        raise RuntimeError("Use the repaired setup_pybullet.py with dynamics synchronization")
    # Keep mouse dragging from applying an unintended external force.
    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_MOUSE_PICKING, 0)
    names = list(robot.model.names[1:])
    joints = sim.bulletCtrlJointsInPinOrder
    actual_names = [sim.p.getJointInfo(sim.robot, j)[1].decode() for j in joints]
    if (robot.model.nq != robot.model.nv or actual_names != names
            or len(joints) != robot.model.nv or sim.robot != 1):
        raise RuntimeError("Unexpected robot model/order or hard-coded reset body ID")

    trajectory = build_reference(robot, cube)
    hand_frames = [robot.model.getFrameId(n) for n in (LEFT_HAND, RIGHT_HAND)]
    hand_links = []
    for name in (LEFT_HAND, RIGHT_HAND):
        matches = [i for i in range(sim.p.getNumJoints(sim.robot))
                   if sim.p.getJointInfo(sim.robot, i)[1].decode() == name]
        if len(matches) != 1:
            raise RuntimeError("Cannot find unique Bullet hand frame: " + name)
        hand_links.append(matches[0])
    if any(i >= robot.model.nframes for i in hand_frames):
        raise RuntimeError("Missing Pinocchio hand frame")

    for label, q in [("start", trajectory.points[0]), ("upper", trajectory.points[1])]:
        pin.framesForwardKinematics(robot.model, robot.data, q)
        print(f"[TRACK] {label} desired hand heights: " + ", ".join(
            f"{robot.data.oMf[i].translation[2]:.4f} m" for i in hand_frames
        ))

    limits = np.asarray([sim.p.getJointInfo(sim.robot, j)[10] for j in joints])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise RuntimeError("Missing positive finite URDF torque limits")
    sim.setqsim(trajectory.points[0])  # Only state reset in this experiment.
    q, v = sim.getpybulletstate()
    np.testing.assert_allclose(q, trajectory.points[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(v, 0, atol=1e-9, rtol=0)
    base0, orientation0 = sim.p.getBasePositionAndOrientation(sim.robot)
    duration = INITIAL_HOLD + trajectory.duration + FINAL_HOLD
    steps = int(np.ceil(duration / DT))
    interval = max(1, round(0.5 / DT))
    directory = Path(__file__).resolve().parent / "part_b_logs"
    directory.mkdir(exist_ok=True)
    filename = directory / ("tracking_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")
    print(f"[TRACK] Duration {steps * DT:.3f} simulated seconds; CSV: {filename}")
    print("[TRACK] Empty hands: hold, move up, return, hold. Mouse picking disabled.")
    peak_error, peak_hand_error = 0., 0.
    tail_errors = []
    clipped_steps = 0
    previous_tau = np.zeros(robot.model.nv)

    with filename.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "phase", "max_error_rad", "max_speed_rad_s", "max_tau_next_Nm",
             "torque_clipped", "step_allowed", "base_drift_m", "base_rotation_rad",
             "left_hand_error_m", "right_hand_error_m"]
            + [prefix + n for prefix in ("q_", "v_", "q_des_", "v_des_", "a_des_",
                                         "tau_next_", "tau_applied_previous_") for n in names]
        )
        try:
            for k in range(steps + 1):
                tick = time.perf_counter()
                t = k * DT
                q_des, v_des, a_des = trajectory.sample(t - INITIAL_HOLD)
                q, v = sim.getpybulletstate()
                tau_raw = compute_torque(robot, q, v, q_des, v_des, a_des)
                tau = np.clip(tau_raw, -limits, limits)
                clipped = bool(np.any(np.abs(tau_raw) > limits))
                error = float(np.max(np.abs(pin.difference(robot.model, q, q_des))))
                speed = float(np.max(np.abs(v)))
                base, orientation = sim.p.getBasePositionAndOrientation(sim.robot)
                drift = float(np.linalg.norm(np.asarray(base) - base0))
                rotation = float(2*np.arccos(np.clip(abs(np.dot(orientation, orientation0)), 0, 1)))
                pin.framesForwardKinematics(robot.model, robot.data, q_des)
                hand_errors = [float(np.linalg.norm(
                    np.asarray(sim.p.getLinkState(sim.robot, link, computeForwardKinematics=True)[4])
                    - robot.data.oMf[frame].translation
                )) for frame, link in zip(hand_frames, hand_links)]
                invalid = (error > MAX_ERROR or speed > MAX_SPEED
                           or drift > 0.02 or rotation > 0.05)
                step_allowed = not invalid and k < steps
                phase = ("initial_hold" if t < INITIAL_HOLD else
                         "tracking" if t < INITIAL_HOLD + trajectory.duration else "final_hold")
                writer.writerow(
                    [t, phase, error, speed, float(np.max(np.abs(tau))), int(clipped),
                     int(step_allowed), drift, rotation] + hand_errors
                    + list(q) + list(v) + list(q_des) + list(v_des) + list(a_des)
                    + list(tau) + list(previous_tau)
                )
                peak_error = max(peak_error, error)
                peak_hand_error = max(peak_hand_error, max(hand_errors))
                if t >= INITIAL_HOLD + trajectory.duration:
                    tail_errors.append(error)
                if k % interval == 0 or invalid or k == steps:
                    print(f"t={t:.3f}s {phase} error={error:.6f}rad "
                          f"speed={speed:.6f}rad/s hand_error={max(hand_errors):.6f}m "
                          f"clipped={clipped}")
                    stream.flush()
                if invalid:
                    contact_summary(sim)
                    raise RuntimeError("Tracking stop threshold exceeded; inspect saved CSV")
                if k == steps:
                    break
                sim.step(tau.tolist())
                previous_tau = tau.copy()
                clipped_steps += int(clipped)
                time.sleep(max(0., DT - (time.perf_counter() - tick)))
        except BaseException:
            print(f"[TRACK STOPPED] Partial CSV saved: {filename}")
            raise

    print(f"[TRACK] completed=True; simulated_time={steps * DT:.3f}s")
    print(f"[TRACK] peak_joint_error={peak_error:.6f} rad")
    print(f"[TRACK] peak_hand_position_error={peak_hand_error:.6f} m")
    print(f"[TRACK] final_hold_max_joint_error={max(tail_errors):.6f} rad")
    print(f"[TRACK] torque_clipped_steps={clipped_steps}/{steps}")
    contact_summary(sim)
    print("[TRACK] Simulation paused. Empty-hand tracking only; grasping not tested.")
    try:
        input("Press Enter to close PyBullet...")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
