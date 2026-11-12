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
from transport import (PlannedTrajectory, TransportCycle, validate_reference,
                       save_plan, load_plan, transport_log)

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


if __name__ == "__main__":

    import argparse
    from tools import setupwithpybullet, rununtil, setcubeplacement
    from config import DT, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true',
                        help='Check and save a path without advancing the simulation')
    parser.add_argument('--plan', help='Use a saved .npz path; all checks are repeated')
    parser.add_argument('--max-iterations', type=int, default=2000,
                        help='RRT search budget when not loading a saved plan (default: 2000)')
    args = parser.parse_args()

    robot, sim, cube = setupwithpybullet()

    try:
        check_setup(robot, sim)

        q0, successinit = computeqgrasppose(robot, robot.q0.copy(), cube, CUBE_PLACEMENT, None)
        qe, successend = computeqgrasppose(robot, robot.q0.copy(), cube, CUBE_PLACEMENT_TARGET, None)
        if not (successinit and successend):
            raise RuntimeError('Initial or goal grasp solve failed')

        cube_path = []
        try:
            if args.plan:
                path, cube_path = load_plan(args.plan)
            else:
                path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET,
                                   robot=robot, cube=cube, cube_path_out=cube_path,
                                   max_iterations=args.max_iterations)
            if len(path) < 2:
                raise RuntimeError('Planner did not return a usable path')

            #0 init and end velocities
            def maketraj(path, cube_path):
                return PlannedTrajectory(path, cube_path, q0, qe,
                                         CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

            trajectory = maketraj(path, cube_path)
            try:
                frames, stats = validate_reference(robot, cube, trajectory)
            except Exception as exc:
                save_plan(__file__, trajectory, dict(preflight_passed=False,
                          error=str(exc), configurations=len(path)))
                raise
            stats['preflight_passed'] = True
            filename = save_plan(__file__, trajectory, stats)
        finally:
            setcubeplacement(robot, cube, CUBE_PLACEMENT)

        if args.plan_only:
            print('[PLAN] Checks passed. No simulation steps executed.')
            print('[PLAN] Inspect the saved plan/report before running physical transport.')
        else:
            #setting initial configuration
            sim.setqsim(q0)
            check_initial_state(sim, q0)

            with transport_log(__file__, robot) as (writer, stream):
                trajs = TransportCycle(robot, sim, trajectory, frames, writer, tracking_torque)
                steps = int(np.ceil(trajs.duration/DT))
                total_time = steps*DT
                tcur = 0.
                step = 0

                while step < steps:
                    rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
                    if step % max(1, round(.5/DT)) == 0:
                        stream.flush()
                    step += 1
                    tcur = step*DT

                trajs.update(sim, robot, tcur, apply=False)

            trajs.report()
    except Exception as exc:
        print(f'[CONTROL STOPPED] {type(exc).__name__}: {exc}')
        raise
    finally:
        pause_simulation()
