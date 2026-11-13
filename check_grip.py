"""CW2 on-table bilateral-contact experiment. No lifting or attachments.

Run from the complete school lab: python check_grip.py
Existing controller and repaired setup_pybullet are used unchanged.
Commanded inward forces are mapped through hand Jacobians; contact readings
are diagnostics, not a closed-loop force regulator or proof of lifting.
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
from tools import setupwithpybullet, collision, jointlimitsviolated

SETTLE_TIME = 1.0
RAMP_TIME = 3.0
KEEP_TIME = 2.0
MAX_ERROR = 0.15
MAX_SPEED = 5.0


def ramp_force(t, target):
    u = np.clip((t - SETTLE_TIME) / RAMP_TIME, 0., 1.)
    return float(target * u**3 * (10. - 15.*u + 6.*u**2))


def squeeze_torque(robot, q, v, q_ref, frame_ids, axis, force):
    """Add equal/opposite inward force commands at the two hand origins.

    Force means desired hand-on-cube force, so its contribution is +J.T F.
    LOCAL_WORLD_ALIGNED expresses linear velocity at the hand origin in world
    axes. Pinocchio spatial vectors put the three linear entries first.
    """
    zero = np.zeros(robot.model.nv)
    tau = compute_torque(robot, q, v, q_ref, zero, zero)
    for frame, direction in zip(frame_ids, (-axis, axis)):
        jacobian = pin.computeFrameJacobian(
            robot.model, robot.data, q, frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).copy()
        tau += jacobian[:3].T @ (force * direction)
    if not np.all(np.isfinite(tau)):
        raise ValueError("Non-finite squeeze torque")
    return tau


def hand_link_groups(sim):
    """Physical wrist links and fixed descendants; exclude forearm contacts."""
    infos = [sim.p.getJointInfo(sim.robot, i)
             for i in range(sim.p.getNumJoints(sim.robot))]
    groups, roots = [], []
    for name in ("LARM_JOINT5", "RARM_JOINT5"):
        matches = [i for i, info in enumerate(infos) if info[1].decode() == name]
        if len(matches) != 1:
            raise RuntimeError("Cannot identify physical wrist joint: " + name)
        root = matches[0]
        roots.append(root)
        group = set()
        for i in range(len(infos)):
            ancestor = i
            while ancestor != -1 and ancestor != root:
                ancestor = infos[ancestor][16]
            if ancestor == root:
                group.add(i)
        groups.append(group)
    return groups, roots


def measure_contact(sim, groups, axis):
    normal = np.zeros(2)
    inward = np.zeros(2)
    counts = np.zeros(2, dtype=int)
    # normalOnB points from cube toward hand. Negating gives hand-on-cube force.
    for c in sim.p.getContactPoints(bodyA=sim.robot, bodyB=sim.cubeId):
        for side, (group, direction) in enumerate(zip(groups, (-axis, axis))):
            if c[3] in group:
                normal[side] += c[9]
                inward[side] += max(0., float(np.dot(-np.asarray(c[7])*c[9], direction)))
                counts[side] += int(c[9] > 1e-4)
    table_load = 0.
    for c in sim.p.getContactPoints(bodyA=sim.robot, bodyB=sim.tableId):
        if c[3] in groups[0] or c[3] in groups[1]:
            table_load += c[9]
    return inward, normal, counts, float(table_load)


def main():
    if not np.isfinite(DT) or DT <= 0:
        raise ValueError("DT must be finite and positive")
    robot, sim, cube = setupwithpybullet()
    if not hasattr(sim, "dynamics_sync_marker_masses"):
        raise RuntimeError("Use the repaired setup_pybullet.py")
    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_MOUSE_PICKING, 0)
    names = list(robot.model.names[1:])
    joints = sim.bulletCtrlJointsInPinOrder
    if (robot.model.nq != robot.model.nv or len(joints) != robot.model.nv
            or [sim.p.getJointInfo(sim.robot, j)[1].decode() for j in joints] != names
            or sim.robot != 1):
        raise RuntimeError("Robot model/order differs from the validated setup")

    q_ref, success = computeqgrasppose(
        robot, robot.q0.copy(), cube, CUBE_PLACEMENT, None
    )
    q_ref = np.asarray(q_ref, dtype=float).copy()
    if (q_ref.shape != (robot.model.nq,) or not np.all(np.isfinite(q_ref))
            or not success or jointlimitsviolated(robot, q_ref) or collision(robot, q_ref)):
        raise RuntimeError("Invalid initial grasp; no simulation steps executed")
    frames = [robot.model.getFrameId(n) for n in (LEFT_HAND, RIGHT_HAND)]
    if any(i >= robot.model.nframes for i in frames):
        raise RuntimeError("Missing hand frame")
    groups, roots = hand_link_groups(sim)
    cube_links = [-1] + list(range(sim.p.getNumJoints(sim.cubeId)))
    cube_mass = sum(sim.p.getDynamicsInfo(sim.cubeId, i)[0] for i in cube_links)
    cube_mu = sim.p.getDynamicsInfo(sim.cubeId, -1)[1]
    # Product is a preload-sizing estimate; real grip is judged by contacts.
    friction = np.asarray([cube_mu * sim.p.getDynamicsInfo(sim.robot, i)[1] for i in roots])
    if cube_mass <= 0 or not np.isfinite(cube_mass) or np.any(friction <= 0) or not np.all(np.isfinite(friction)):
        raise RuntimeError("Invalid loaded cube mass or contact friction")
    weight = cube_mass * abs(sim.pybulletConfigs["gravity"])
    force_target = float(max(5., 1.25 * weight / (2. * np.min(friction))))
    if force_target > 40.:
        raise RuntimeError("Estimated preload exceeds this diagnostic's 40 N/hand limit")
    limits = np.asarray([sim.p.getJointInfo(sim.robot, j)[10] for j in joints])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise RuntimeError("Missing torque limits")
    print(f"[GRIP] Loaded cube total mass={cube_mass:.6f} kg; weight={weight:.3f} N")
    print(f"[GRIP] Friction estimates L/R={friction}; target per hand={force_target:.3f} N")
    print("[GRIP] 1 s settle, 3 s force ramp, 2 s contact hold. No lifting.")

    sim.setqsim(q_ref)  # Initialization only; neither cube nor joints reset again.
    q, v = sim.getpybulletstate()
    np.testing.assert_allclose(q, q_ref, atol=1e-9, rtol=0)
    np.testing.assert_allclose(v, 0, atol=1e-9, rtol=0)
    base0, base_rot0 = sim.p.getBasePositionAndOrientation(sim.robot)
    cube0, cube_rot0 = sim.p.getBasePositionAndOrientation(sim.cubeId)
    steps = int(np.ceil((SETTLE_TIME + RAMP_TIME + KEEP_TIME) / DT))
    report_every = max(1, round(.5 / DT))
    directory = Path(__file__).resolve().parent / "part_b_logs"
    directory.mkdir(exist_ok=True)
    filename = directory / ("grip_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")
    print(f"[GRIP] CSV: {filename}")
    previous_tau = np.zeros(robot.model.nv)
    tail_forces, tail_tables = [], []
    clipped_steps, peak_error = 0, 0.

    with filename.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "force_reference_per_hand_N", "left_inward_N", "right_inward_N",
             "left_total_normal_N", "right_total_normal_N", "left_loaded_contacts",
             "right_loaded_contacts", "hand_table_load_N", "max_joint_error_rad",
             "max_speed_rad_s", "max_tau_next_Nm", "torque_clipped", "step_allowed",
             "cube_x_m", "cube_y_m", "cube_z_m", "cube_rotation_rad", "base_drift_m"]
            + [prefix + n for prefix in ("q_", "v_", "q_des_", "tau_next_", "tau_applied_previous_")
               for n in names]
        )
        try:
            for k in range(steps + 1):
                tick = time.perf_counter()
                t = k * DT
                q, v = sim.getpybulletstate()
                cube_pos, cube_rot = sim.p.getBasePositionAndOrientation(sim.cubeId)
                axis = np.asarray(sim.p.getMatrixFromQuaternion(cube_rot)).reshape(3, 3)[:, 1]
                force = ramp_force(t, force_target)
                tau_raw = squeeze_torque(robot, q, v, q_ref, frames, axis, force)
                tau = np.clip(tau_raw, -limits, limits)
                clipped = bool(np.any(np.abs(tau_raw) > limits))
                inward, normal, counts, table_load = measure_contact(sim, groups, axis)
                error = float(np.max(np.abs(pin.difference(robot.model, q, q_ref))))
                speed = float(np.max(np.abs(v)))
                base_pos, base_rot = sim.p.getBasePositionAndOrientation(sim.robot)
                base_drift = float(np.linalg.norm(np.asarray(base_pos) - base0))
                base_angle = float(2*np.arccos(np.clip(abs(np.dot(base_rot, base_rot0)), 0, 1)))
                cube_angle = float(2*np.arccos(np.clip(abs(np.dot(cube_rot, cube_rot0)), 0, 1)))
                cube_shift = float(np.linalg.norm(np.asarray(cube_pos) - cube0))
                invalid = (error > MAX_ERROR or speed > MAX_SPEED or base_drift > .02
                           or base_angle > .05 or cube_shift > .03 or cube_angle > .15
                           or table_load > 5.)
                writer.writerow(
                    [t, force] + list(inward) + list(normal) + list(counts)
                    + [table_load, error, speed, float(np.max(np.abs(tau))), int(clipped),
                       int(not invalid and k < steps)] + list(cube_pos) + [cube_angle, base_drift]
                    + list(q) + list(v) + list(q_ref) + list(tau) + list(previous_tau)
                )
                peak_error = max(peak_error, error)
                if t >= SETTLE_TIME + RAMP_TIME + KEEP_TIME - 1.:
                    tail_forces.append(inward.copy())
                    tail_tables.append(table_load)
                if k % report_every == 0 or invalid or k == steps:
                    print(f"t={t:.3f}s request={force:.2f}N/hand "
                          f"contact_L/R={inward[0]:.2f}/{inward[1]:.2f}N "
                          f"table={table_load:.3f}N error={error:.6f}rad clipped={clipped}")
                    stream.flush()
                if invalid:
                    raise RuntimeError(
                        f"Grip diagnostic stopped: error={error:.4f}rad speed={speed:.3f}rad/s "
                        f"cube_shift={cube_shift:.4f}m cube_angle={cube_angle:.4f}rad "
                        f"table_load={table_load:.3f}N base_drift={base_drift:.4f}m"
                    )
                if k == steps:
                    break
                sim.step(tau.tolist())
                previous_tau = tau.copy()
                clipped_steps += int(clipped)
                time.sleep(max(0., DT - (time.perf_counter() - tick)))
        except BaseException:
            print(f"[GRIP STOPPED] Partial CSV saved: {filename}")
            raise

    tail = np.asarray(tail_forces)
    bilateral = float(np.mean(np.all(tail > 1., axis=1)))
    print(f"[GRIP] completed=True; simulated_time={steps*DT:.3f}s")
    print(f"[GRIP] last_1s_mean_inward_L/R={tail.mean(axis=0)} N")
    print(f"[GRIP] last_1s_min_inward_L/R={tail.min(axis=0)} N")
    print(f"[GRIP] last_1s_both_contacts_above_1N_fraction={bilateral:.3f}")
    print(f"[GRIP] last_1s_max_hand_table_load={max(tail_tables):.6f} N")
    print(f"[GRIP] peak_joint_error={peak_error:.6f} rad; clipped_steps={clipped_steps}/{steps}")
    print("[GRIP] On-table contact test finished. No lifting or friction-support claim yet.")
    try:
        input("Press Enter to close PyBullet...")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
