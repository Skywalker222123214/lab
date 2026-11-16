#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin

from trajectory import PathTrajectory
from grasp_cycle import check_setup, check_initial_state, pause_simulation
from transport import (PlannedTrajectory, TransportCycle, validate_path_geometry,
                       trajectory_metadata, save_plan, transport_log,
                       cube_pose, pose_errors)

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)


def tracking_torque(robot, q, vq, reference, frames, axis, squeeze, upward):
    q_des, v_des, a_des = reference
    if not np.all(np.isfinite(np.r_[q, vq, q_des, v_des, a_des])):
        raise ValueError('Non-finite state or reference')
    error = pin.difference(robot.model, q, q_des)
    acceleration = a_des + Kp*error + Kv*(v_des-vq)
    torques = np.asarray(pin.rnea(robot.model, robot.data, q, vq, acceleration)).copy()

    # Add the squeeze and object support forces at the two hands.
    for frame, direction in zip(frames, (-axis, axis)):
        J = pin.computeFrameJacobian(robot.model, robot.data, q, frame,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
        support = np.array([0., 0., upward]) if np.ndim(upward)==0 else np.asarray(upward)
        force = squeeze*direction + support
        torques += J[:3].T @ force
    if not np.all(np.isfinite(torques)):
        raise ValueError('Non-finite torque')
    return torques


def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    torques = trajs.update(sim, robot, tcurrent, state=(q, vq))
    sim.step(torques)
    trajs.record_step(torques)


def settle_cube(robot,sim,cube,nominal_start,timeout=3.,
                linear_speed=1e-3,angular_speed=1e-3,stable_time=.3):
    """Let the real cube settle while holding a verified noncontact posture.

    Only robot joints are explicitly initialized. The cube is never reset,
    constrained, frozen, or moved to a planned pose in the physics engine.
    """
    from tools import collision, jointlimitsviolated, setcubeplacement, rununtil
    from inverse_geometry import computeqgrasppose
    from config import DT
    settings = np.array([timeout,linear_speed,angular_speed,stable_time])
    if not np.all(np.isfinite(settings)) or np.min(settings)<=0:
        raise ValueError('Settling time and velocity thresholds must be positive')
    limits = np.array([sim.p.getJointInfo(sim.robot,j)[10]
                       for j in sim.bulletCtrlJointsInPinOrder])
    park = None
    try:
        # The URDF zero posture intersects the table/cube. Solve only a parking
        # pose here; the actual initial grasp is solved AFTER physical settling.
        for dz in (.15,.20,.10):
            above = nominal_start.copy()
            above.translation = above.translation+np.array([0.,0.,dz])
            candidate,ok = computeqgrasppose(robot,robot.q0.copy(),cube,above,None)
            setcubeplacement(robot,cube,nominal_start)
            if not ok or jointlimitsviolated(robot,candidate) or collision(robot,candidate):
                continue
            sim.setqsim(candidate)
            sim.p.performCollisionDetection()
            gaps = {name:min((point[8] for point in sim.p.getClosestPoints(sim.robot,body,.1)),
                            default=.1)
                    for name,body in [('cube',sim.cubeId),('table',sim.tableId),
                                      ('obstacle',sim.obstacleId)]}
            if min(gaps.values())>.005:
                park = candidate.copy()
                break
    finally:
        setcubeplacement(robot,cube,nominal_start)
    if park is None:
        raise RuntimeError('No noncontact parking posture; settling not started')
    check_initial_state(sim,park)
    base_position,base_quaternion = sim.p.getBasePositionAndOrientation(sim.robot)
    base_pose = pin.XYZQUATToSE3(np.r_[base_position,base_quaternion])
    mass = sum(sim.p.getDynamicsInfo(sim.cubeId,i)[0]
               for i in [-1]+list(range(sim.p.getNumJoints(sim.cubeId))))
    weight = mass*abs(sim.pybulletConfigs['gravity'])
    needed = int(np.ceil(stable_time/DT))
    stable_steps = 0
    peak_error = 0.
    sim.settling_report = dict(settled=False,robot_parking_initialization='resetJointState',
        parking_cube_offset_z_m=dz,parking_clearances_m=gaps,
        linear_speed_threshold_m_s=linear_speed,angular_speed_threshold_rad_s=angular_speed,
        required_stable_time_s=stable_time,timeout_s=timeout,robot_cube_contact_steps=0,
        cube_reset=False)

    def hold():
        nonlocal peak_error
        q,v = sim.getpybulletstate()
        error = pin.difference(robot.model,q,park)
        peak_error = max(peak_error,float(np.max(np.abs(error))))
        if (jointlimitsviolated(robot,q) or peak_error>.03
                or not np.all(np.isfinite(np.r_[q,v]))):
            raise RuntimeError('Robot did not remain stable during cube settling')
        a = Kp*error-Kv*v
        tau = np.asarray(pin.rnea(robot.model,robot.data,q,v,a)).copy()
        if not np.all(np.isfinite(tau)) or np.any(np.abs(tau)>limits):
            raise RuntimeError('Invalid/excessive parking torque; no step applied')
        sim.step(tau.tolist())

    for step in range(int(np.ceil(timeout/DT))):
        rununtil(hold,DT)
        actual = cube_pose(sim)
        linear,angular = sim.p.getBaseVelocity(sim.cubeId)
        speed,spin = float(np.linalg.norm(linear)),float(np.linalg.norm(angular))
        if not np.all(np.isfinite(np.r_[actual.homogeneous.ravel(),linear,angular])):
            raise RuntimeError('Non-finite cube state during settling')
        # No hand, arm or torso may contact the cube/table/obstacle in this phase.
        for name,body in [('cube',sim.cubeId),('table',sim.tableId),('obstacle',sim.obstacleId)]:
            if sim.p.getContactPoints(sim.robot,body):
                if name=='cube':
                    sim.settling_report['robot_cube_contact_steps'] += 1
                raise RuntimeError('Robot touched '+name+' before grasp initialization')
        now_position,now_quaternion = sim.p.getBasePositionAndOrientation(sim.robot)
        base_error = pose_errors(pin.XYZQUATToSE3(np.r_[now_position,now_quaternion]),base_pose)
        if base_error[0]>.002 or base_error[1]>.01:
            raise RuntimeError('Robot base drifted during cube settling')
        table_load = float(sum(point[9] for point in sim.p.getContactPoints(sim.cubeId,sim.tableId)))
        if speed<=linear_speed and spin<=angular_speed and table_load>=.8*weight:
            stable_steps += 1
        else:
            stable_steps = 0
        sim.settling_report.update(physical_steps=step+1,elapsed_s=(step+1)*DT,
            stable_time_s=stable_steps*DT,linear_speed_m_s=speed,angular_speed_rad_s=spin,
            table_support_N=table_load,peak_parking_joint_error_rad=peak_error)
        if stable_steps>=needed:
            translation = actual.translation-nominal_start.translation
            angle = pose_errors(actual,nominal_start)[1]
            sim.settling_report.update(settled=True,
                nominal_pose_xyzquat=pin.SE3ToXYZQUAT(nominal_start).tolist(),
                actual_pose_xyzquat=pin.SE3ToXYZQUAT(actual).tolist(),
                settling_translation_m=translation.tolist(),settling_orientation_rad=angle,
                parking_q=park.tolist())
            print(f'[SETTLE] settled=True; elapsed={(step+1)*DT:.3f}s; '
                  f'stable={stable_steps*DT:.3f}s; speed={speed:.3g}m/s; spin={spin:.3g}rad/s; '
                  f'delta_xyz_m={translation}; delta_angle={angle:.6g}rad',flush=True)
            return actual,sim.settling_report
    raise RuntimeError(f'Cube did not settle within {timeout:.3f}s; '
                       f'speed={speed:.6g}m/s, spin={spin:.6g}rad/s, '
                       f'stable={stable_steps*DT:.3f}s')


def initialize_grasp(robot,sim,cube,q_initial,cube_start_actual):
    """Explicit simulation-only joint initialization, not a planned arm approach."""
    from config import LEFT_HAND,RIGHT_HAND,LEFT_HOOK,RIGHT_HOOK
    before = cube_pose(sim)
    if max(pose_errors(before,cube_start_actual))>1e-9:
        raise RuntimeError('Cube moved while planning; fresh measured-start planning required')
    sim.setqsim(q_initial)
    check_initial_state(sim,q_initial)
    sim.p.performCollisionDetection()
    after = cube_pose(sim)
    if max(pose_errors(after,before))>1e-12:
        raise RuntimeError('Robot initialization unexpectedly changed the cube pose')
    pin.framesForwardKinematics(cube.model,cube.data,cube.q0)
    position_error = angle_error = 0.
    for hand_name,hook_name in zip((LEFT_HAND,RIGHT_HAND),(LEFT_HOOK,RIGHT_HOOK)):
        ids = [i for i in range(sim.p.getNumJoints(sim.robot))
               if sim.p.getJointInfo(sim.robot,i)[1].decode()==hand_name]
        if len(ids)!=1:
            raise RuntimeError('Missing Bullet hand frame')
        link = sim.p.getLinkState(sim.robot,ids[0],computeForwardKinematics=True)
        hand = pin.XYZQUATToSE3(np.r_[link[4],link[5]])
        hook = cube.data.oMf[cube.model.getFrameId(hook_name)]
        ep,er = pose_errors(hand,after*hook)
        position_error,angle_error = max(position_error,ep),max(angle_error,er)
    # Part 1/geometry preflight use strict 1e-3. Bullet FK also includes the
    # measured free-base drift and float32 link-frame output; record both.
    if position_error>1.1e-3 or angle_error>1.1e-3:
        raise RuntimeError('Actual Bullet hands do not match the verified initial grasp')
    print(f'[INIT] robot explicitly initialized to q_initial; cube unchanged; '
          f'hand error={position_error:.6g}m/{angle_error:.6g}rad',flush=True)
    return dict(robot_grasp_initialization='resetJointState',cube_reset=False,
                robot_q_initial=q_initial.tolist(),actual_cube_start_error_m=pose_errors(after,cube_start_actual)[0],
                actual_cube_start_orientation_error_rad=pose_errors(after,cube_start_actual)[1],
                actual_hand_position_error_m=position_error,actual_hand_orientation_error_rad=angle_error)


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import hashlib
    import json
    from pathlib import Path
    from tools import setupwithpybullet,rununtil,setcubeplacement
    from config import DT,CUBE_PLACEMENT,CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import plan_grasp_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only',action='store_true',
                        help='Physically settle and replan; do not initialize/execute the grasp')
    parser.add_argument('--plan',help='Deprecated: measured-start runs always require fresh planning')
    parser.add_argument('--trajectory',choices=['qp','quintic'],default='qp')
    parser.add_argument('--max-iterations',type=int,default=4000)
    parser.add_argument('--settle-timeout',type=float,default=3.)
    parser.add_argument('--seed',type=int,help='Optional reproducible RRT seed')
    parser.add_argument('--headless',action='store_true',help='Use Bullet DIRECT display mode')
    args = parser.parse_args()
    if args.plan is not None:
        parser.error('--plan cannot be used with measured-start initialization; fresh planning is required')
    if args.max_iterations<=0 or not np.isfinite(args.settle_timeout) or args.settle_timeout<=0:
        parser.error('Iterations and settling timeout must be positive')
    if args.seed is not None:
        if not 0<=args.seed<=2**32-1:
            parser.error('Seed must be in [0, 2**32-1]')
        np.random.seed(args.seed)
    if args.headless:
        from setup_pybullet import pybulletConfigs
        pybulletConfigs['gui'] = False

    root = Path(__file__).resolve().parent
    directory = root/'part_b_logs'
    directory.mkdir(exist_ok=True)
    report_path = directory/('settled_transport_run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    run_report = dict(completed=False,fresh_planning=True,cached_plan_used=False,
        reference_offset_applied=False,seed=args.seed,plan_only=args.plan_only,
        nominal_start_xyzquat=pin.SE3ToXYZQUAT(CUBE_PLACEMENT).tolist(),
        original_goal_xyzquat=pin.SE3ToXYZQUAT(CUBE_PLACEMENT_TARGET).tolist(),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ['control.py','transport.py','path.py','inverse_geometry.py',
                                    'qp_trajectory.py','setup_pybullet.py','config.py']})
    robot,sim,cube = setupwithpybullet()
    try:
        check_setup(robot,sim)
        cube_start_actual,settling = settle_cube(robot,sim,cube,CUBE_PLACEMENT,
                                                timeout=args.settle_timeout)
        run_report['settling'] = settling
        q0,successinit = computeqgrasppose(robot,robot.q0.copy(),cube,cube_start_actual,None)
        qe,successend = computeqgrasppose(robot,robot.q0.copy(),cube,CUBE_PLACEMENT_TARGET,None)
        if not (successinit and successend):
            raise RuntimeError('Measured-start or original-goal grasp solve failed')
        # q0 and qe are the inputs used by planning, timing and execution below.
        run_report.update(q_initial=q0.tolist(),q_goal=qe.tolist(),
                          actual_start_xyzquat=pin.SE3ToXYZQUAT(cube_start_actual).tolist())
        cube_path = []
        try:
            path = plan_grasp_path(q0,qe,cube_start_actual,CUBE_PLACEMENT_TARGET,
                               robot=robot,cube=cube,cube_path_out=cube_path,
                               max_iterations=args.max_iterations)
            if len(path)<2:
                raise RuntimeError('Planner did not return a measured-start path')
            # Use the current path, with one matching cube pose per waypoint.
            # Edits must pass geometry before new timing/Bezier fitting.
            frames,geometry_stats = validate_path_geometry(robot,cube,path,cube_path,
                q0,qe,cube_start_actual,CUBE_PLACEMENT_TARGET)
            trajectory = PlannedTrajectory(path,cube_path,q0,qe,
                cube_start_actual,CUBE_PLACEMENT_TARGET,method=args.trajectory)
            stats = trajectory_metadata(trajectory,geometry_stats)
            stats.update(actual_start_xyzquat=run_report['actual_start_xyzquat'],
                         original_goal_xyzquat=run_report['original_goal_xyzquat'],
                         settling=settling,reference_offset_applied=False)
            filename = save_plan(__file__,trajectory,stats)
            run_report.update(plan=str(filename),preflight=stats)
        finally:
            # This updates only Pinocchio geometry; it never moves the Bullet cube.
            setcubeplacement(robot,cube,cube_start_actual)
        print(f'[PLAN] fresh measured-start path: {len(path)} configurations; '
              f'motion={trajectory.duration:.3f}s; original goal retained',flush=True)
        if args.plan_only:
            run_report.update(plan_only_passed=True,transport_physical_steps=0)
            print(f'[PLAN] Settling executed {settling["physical_steps"]} physical steps; '
                  'no grasp initialization or transport steps executed.',flush=True)
        else:
            grasp_init = initialize_grasp(robot,sim,cube,q0,cube_start_actual)
            run_report['grasp_initialization'] = grasp_init
            with transport_log(__file__,robot) as (writer,stream):
                run_report['csv'] = str(stream.name)
                trajs = TransportCycle(robot,sim,trajectory,frames,writer,tracking_torque,
                                       initialization=settling)
                steps = 0
                # A bounded table-handoff wait may extend the cycle duration.
                while steps*DT<trajs.duration:
                    rununtil(controllaw,DT,sim,robot,trajs,steps*DT,cube)
                    if steps%1000==0:
                        stream.flush()
                    steps += 1
                trajs.update(sim,robot,steps*DT,apply=False)
            result = trajs.report()
            run_report.update(result,transport_physical_steps=trajs.applied_steps,
                total_physical_steps=settling['physical_steps']+trajs.applied_steps)
    except Exception as exc:
        run_report['error'] = type(exc).__name__+': '+str(exc)
        print('[CONTROL STOPPED]',run_report['error'],flush=True)
        raise
    finally:
        if 'settling' not in run_report and hasattr(sim,'settling_report'):
            run_report['settling'] = sim.settling_report
        report_path.write_text(json.dumps(run_report,indent=2,allow_nan=False)+'\n')
        print('[RUN] report:',report_path,flush=True)
        pause_simulation()
