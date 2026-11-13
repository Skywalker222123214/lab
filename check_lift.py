"""Local CW2 lift experiment: preload, support, lift 2 cm, hold 2 s.

Run inside the full lab: python check_lift.py
Requires existing check_grip.py, trajectory.py, controller.py and repaired
setup_pybullet.py. No cube resets, attachments or contact-filter changes.
This local diagnostic does not replace Part A's complete obstacle planner.
"""

import csv
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import pinocchio as pin

from config import CUBE_PLACEMENT, DT, LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from controller import compute_torque
from check_grip import hand_link_groups, measure_contact, ramp_force
from inverse_geometry import computeqgrasppose
from tools import setupwithpybullet, setcubeplacement, collision, jointlimitsviolated
from trajectory import PathTrajectory

HEIGHT = 0.02
PRELOAD_END = 4.0
LIFT_START = 5.0
HOLD_TIME = 2.0


def smooth_profile(t, duration):
    """Normalized displacement and physical-time second derivative."""
    if t <= 0:
        return 0., 0.
    if t >= duration:
        return 1., 0.
    u = t / duration
    return u**3*(10.-15.*u+6.*u*u), 60.*u*(1.-u)*(1.-2.*u)/duration**2


def build_lift_reference(robot, cube):
    upper = CUBE_PLACEMENT.copy()
    upper.translation = CUBE_PLACEMENT.translation + np.array([0., 0., HEIGHT])
    try:
        q0, ok0 = computeqgrasppose(robot, robot.q0.copy(), cube, CUBE_PLACEMENT, None)
        if not ok0:
            raise RuntimeError("Initial grasp solve failed; no dynamics steps executed")
        q1, ok1 = computeqgrasppose(robot, q0.copy(), cube, upper, None)
        if not ok1:
            raise RuntimeError("Raised grasp solve failed; no dynamics steps executed")
        trajectory = PathTrajectory([q0, q1], max_velocity=.12,
                                    max_acceleration=.25, min_segment_time=3.)
        if np.max(np.abs(trajectory.deltas[0])) > .25:
            raise RuntimeError("IK changed branch during small lift; no dynamics steps executed")
        pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
        hooks = [cube.data.oMf[cube.model.getFrameId(n)].copy()
                 for n in (LEFT_HOOK, RIGHT_HOOK)]
        frames = [robot.model.getFrameId(n) for n in (LEFT_HAND, RIGHT_HAND)]
        max_position, max_angle = 0., 0.
        for s in np.linspace(0., 1., 101):
            q = (1.-s)*trajectory.points[0] + s*trajectory.points[1]
            pose = CUBE_PLACEMENT.copy()
            pose.translation = CUBE_PLACEMENT.translation + np.array([0., 0., HEIGHT*s])
            setcubeplacement(robot, cube, pose)
            if jointlimitsviolated(robot, q) or collision(robot, q):
                raise RuntimeError("Local lift reference has a collision/limit violation")
            pin.framesForwardKinematics(robot.model, robot.data, q)
            for frame, hook in zip(frames, hooks):
                target = pose * hook
                current = robot.data.oMf[frame]
                ep = float(np.linalg.norm(current.translation-target.translation))
                er = float(np.linalg.norm(pin.log(current.inverse()*target).vector[3:]))
                max_position, max_angle = max(max_position, ep), max(max_angle, er)
                if ep > .003 or er > .015:
                    raise RuntimeError("Interpolated lift reference loses grasp geometry")
        print(f"[LIFT] 101 sampled references checked: grasp error "
              f"<={max_position:.6f} m, {max_angle:.6f} rad")
        print("[LIFT] These are discrete checks, not a continuous collision guarantee.")
        return trajectory, frames
    finally:
        setcubeplacement(robot, cube, CUBE_PLACEMENT)


def lift_torque(robot, q, v, reference, frames, axis, normal_force, up_per_hand):
    qd, vd, ad = reference
    tau = compute_torque(robot, q, v, qd, vd, ad)
    for frame, direction in zip(frames, (-axis, axis)):
        jacobian = pin.computeFrameJacobian(
            robot.model, robot.data, q, frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).copy()
        force = normal_force*direction + np.array([0., 0., up_per_hand])
        tau += jacobian[:3].T @ force
    if not np.all(np.isfinite(tau)):
        raise ValueError("Non-finite lift torque")
    return tau


def has_lift_evidence(height, box_table, robot_table, inward, slip):
    """Local experiment criteria, not the lecturer's grading thresholds."""
    return bool(height >= .015 and box_table < .5 and robot_table < .5
                and np.all(np.asarray(inward) > 1.) and slip < .005)


def main():
    if not np.isfinite(DT) or DT <= 0:
        raise ValueError("Invalid DT")
    robot, sim, cube = setupwithpybullet()
    if not hasattr(sim, "dynamics_sync_marker_masses"):
        raise RuntimeError("Use the validated dynamics-synchronized setup_pybullet.py")
    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_MOUSE_PICKING, 0)
    joints = sim.bulletCtrlJointsInPinOrder
    names = list(robot.model.names[1:])
    if (robot.model.nq != robot.model.nv or len(joints) != robot.model.nv
            or [sim.p.getJointInfo(sim.robot, j)[1].decode() for j in joints] != names
            or sim.robot != 1):
        raise RuntimeError("Unexpected robot configuration/order/reset body ID")
    trajectory, frames = build_lift_reference(robot, cube)
    groups, wrists = hand_link_groups(sim)
    hands = []
    for name in (LEFT_HAND, RIGHT_HAND):
        matches = [i for i in range(sim.p.getNumJoints(sim.robot))
                   if sim.p.getJointInfo(sim.robot, i)[1].decode() == name]
        if len(matches) != 1:
            raise RuntimeError("Missing unique Bullet hand frame")
        hands.append(matches[0])
    mass = sum(sim.p.getDynamicsInfo(sim.cubeId, i)[0]
               for i in [-1]+list(range(sim.p.getNumJoints(sim.cubeId))))
    mu_box = sim.p.getDynamicsInfo(sim.cubeId, -1)[1]
    mus = np.array([mu_box*sim.p.getDynamicsInfo(sim.robot, j)[1] for j in wrists])
    gravity = abs(sim.pybulletConfigs["gravity"])
    if not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(mus)) or np.any(mus <= 0):
        raise RuntimeError("Invalid actual cube mass/friction")
    weight = mass*gravity
    normal_target = float(max(5., 1.25*weight/(2.*np.min(mus))))
    limits = np.asarray([sim.p.getJointInfo(sim.robot, j)[10] for j in joints])
    if normal_target > 40. or np.any(limits <= 0) or not np.all(np.isfinite(limits)):
        raise RuntimeError("Preload/torque settings outside this diagnostic's range")
    print(f"[LIFT] Actual cube mass={mass:.6f} kg; squeeze={normal_target:.3f} N/hand")
    print(f"[LIFT] Gravity support={weight/2:.3f} N upward/hand, ramped before motion")
    sim.setqsim(trajectory.points[0])  # Only state reset; physical cube is never reset.
    q, v = sim.getpybulletstate()
    np.testing.assert_allclose(q, trajectory.points[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(v, 0, atol=1e-9, rtol=0)
    base0, base_rot0 = sim.p.getBasePositionAndOrientation(sim.robot)
    cube0, cube_rot0 = sim.p.getBasePositionAndOrientation(sim.cubeId)
    end_motion = LIFT_START + trajectory.duration
    steps = int(np.ceil((end_motion + HOLD_TIME)/DT))
    interval = max(1, round(.5/DT))
    out = Path(__file__).resolve().parent / 'part_b_logs'
    out.mkdir(exist_ok=True)
    filename = out / ('lift_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
    print(f"[LIFT] Duration={steps*DT:.3f} s; CSV: {filename}")
    baseline_cube, baseline_relative = None, None
    ready_samples, tail = [], []
    previous_tau = np.zeros(robot.model.nv)
    clipped_steps, off_table_hold_time = 0, 0.
    peak_height, peak_error = 0., 0.
    with filename.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ['time_s','phase','squeeze_reference_N','up_reference_per_hand_N',
             'left_inward_N','right_inward_N','cube_table_support_N',
             'robot_table_load_N','obstacle_load_N','actual_lift_m','relative_slip_m',
             'max_joint_error_rad','max_speed_rad_s','torque_clipped','step_allowed',
             'cube_x_m','cube_y_m','cube_z_m','cube_rotation_rad','base_drift_m']
            + [prefix+n for prefix in ('q_','v_','q_des_','tau_next_','tau_applied_previous_') for n in names]
        )
        try:
            for k in range(steps+1):
                tick = time.perf_counter()
                t = k*DT
                q, v = sim.getpybulletstate()
                cube_pos, cube_rot = sim.p.getBasePositionAndOrientation(sim.cubeId)
                cube_pos = np.asarray(cube_pos)
                axis = np.asarray(sim.p.getMatrixFromQuaternion(cube_rot)).reshape(3,3)[:,1]
                inward, _, _, _ = measure_contact(sim, groups, axis)
                box_table = float(sum(c[9] for c in sim.p.getContactPoints(bodyA=sim.cubeId, bodyB=sim.tableId)))
                robot_table = float(sum(c[9] for c in sim.p.getContactPoints(bodyA=sim.robot, bodyB=sim.tableId)))
                obstacle = float(sum(c[9] for body in (sim.robot, sim.cubeId)
                                     for c in sim.p.getContactPoints(bodyA=body, bodyB=sim.obstacleId)))
                midpoint = np.mean([sim.p.getLinkState(sim.robot, i, computeForwardKinematics=True)[4]
                                    for i in hands], axis=0)
                # Record the settled, pre-lift cube height, not the nominal URDF height.
                if PRELOAD_END-.25 <= t < PRELOAD_END:
                    ready_samples.append(bool(np.all(inward > max(1., .25*normal_target)) and robot_table < .5))
                if baseline_cube is None and t >= PRELOAD_END:
                    if (not ready_samples or not all(ready_samples)
                            or not np.all(inward > max(1., .25*normal_target)) or robot_table >= .5):
                        raise RuntimeError("Bilateral preload not established; lift cancelled")
                    baseline_cube = cube_pos.copy()
                    baseline_relative = cube_pos-midpoint
                    print(f"[LIFT] Preload established; settled cube height={cube_pos[2]:.6f} m")
                support, _ = smooth_profile(t-PRELOAD_END, LIFT_START-PRELOAD_END)
                _, z_acceleration = smooth_profile(t-LIFT_START, trajectory.duration)
                upward = .5*mass*(gravity*support + HEIGHT*z_acceleration)
                squeeze = ramp_force(t, normal_target)
                reference = trajectory.sample(t-LIFT_START)
                tau_raw = lift_torque(robot, q, v, reference, frames, axis, squeeze, upward)
                tau = np.clip(tau_raw, -limits, limits)
                clipped = bool(np.any(np.abs(tau_raw) > limits))
                error = float(np.max(np.abs(pin.difference(robot.model, q, reference[0]))))
                speed = float(np.max(np.abs(v)))
                base, base_rot = sim.p.getBasePositionAndOrientation(sim.robot)
                drift = float(np.linalg.norm(np.asarray(base)-base0))
                base_angle = float(2*np.arccos(np.clip(abs(np.dot(base_rot,base_rot0)),0,1)))
                cube_angle = float(2*np.arccos(np.clip(abs(np.dot(cube_rot,cube_rot0)),0,1)))
                height = 0. if baseline_cube is None else float(cube_pos[2]-baseline_cube[2])
                slip = 0. if baseline_relative is None else float(np.linalg.norm(cube_pos-midpoint-baseline_relative))
                lateral = float(np.linalg.norm((cube_pos-np.asarray(cube0))[:2]))
                invalid = (error > .15 or speed > 5. or drift > .02 or base_angle > .05
                           or lateral > .03 or cube_angle > .15 or robot_table > 5.
                           or obstacle > 1. or height < -.015 or height > .06 or slip > .015)
                phase = ('preload' if t < PRELOAD_END else 'support_ramp' if t < LIFT_START
                         else 'lifting' if t < end_motion else 'hold')
                writer.writerow(
                    [t,phase,squeeze,upward]+list(inward)
                    + [box_table,robot_table,obstacle,height,slip,error,speed,int(clipped),int(not invalid and k<steps)]
                    + list(cube_pos)+[cube_angle,drift]+list(q)+list(v)+list(reference[0])+list(tau)+list(previous_tau)
                )
                peak_height, peak_error = max(peak_height,height), max(peak_error,error)
                evidence = has_lift_evidence(height, box_table, robot_table, inward, slip)
                if phase == 'hold' and k < steps and evidence:
                    off_table_hold_time += DT
                if t >= end_motion+HOLD_TIME-1.:
                    tail.append([height,box_table,slip,inward[0],inward[1],float(evidence)])
                if k % interval == 0 or invalid or k == steps:
                    print(f"t={t:.3f}s {phase} lift={height:.5f}m table={box_table:.3f}N "
                          f"grip_L/R={inward[0]:.2f}/{inward[1]:.2f}N slip={slip:.5f}m "
                          f"error={error:.6f}rad clipped={clipped}")
                    stream.flush()
                if invalid:
                    raise RuntimeError(f"Lift diagnostic stopped: speed={speed:.3f}rad/s "
                                       f"robot_table={robot_table:.3f}N obstacle={obstacle:.3f}N "
                                       f"slip={slip:.4f}m error={error:.4f}rad")
                if k == steps:
                    break
                sim.step(tau.tolist())
                previous_tau = tau.copy()
                clipped_steps += int(clipped)
                time.sleep(max(0.,DT-(time.perf_counter()-tick)))
        except BaseException:
            print(f"[LIFT STOPPED] Partial CSV saved: {filename}")
            raise

    tail = np.asarray(tail)
    print(f"[LIFT] completed=True; simulated_time={steps*DT:.3f}s")
    print(f"[LIFT] peak_actual_lift={peak_height:.6f} m; requested={HEIGHT:.6f} m")
    print(f"[LIFT] last_1s_min_lift={tail[:,0].min():.6f} m")
    print(f"[LIFT] last_1s_max_cube_table_support={tail[:,1].max():.6f} N")
    print(f"[LIFT] last_1s_max_relative_slip={tail[:,2].max():.6f} m")
    print(f"[LIFT] off_table_bilateral_contact_time_during_hold={off_table_hold_time:.3f} s")
    print(f"[LIFT] last_1s_lift_evidence_fraction={tail[:,5].mean():.3f}")
    print(f"[LIFT] peak_joint_error={peak_error:.6f} rad; clipped_steps={clipped_steps}/{steps}")
    print("[LIFT] Inspect measured evidence; completion alone is not successful lifting.")
    print("[LIFT] Simulation now paused. No lowering or full transport is implemented here.")
    try:
        input('Press Enter to close PyBullet...')
    except EOFError:
        pass


if __name__ == '__main__':
    main()
