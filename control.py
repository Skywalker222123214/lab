#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin

from trajectory import PathTrajectory
from grasp_cycle import (ControlCycle, build_lift_reference, check_setup,
                         check_initial_state, cycle_log, print_result,
                         pause_simulation)

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

    # Add the squeeze and vertical support forces at the two hands.
    for frame, direction in zip(frames, (-axis, axis)):
        J = pin.computeFrameJacobian(robot.model, robot.data, q, frame,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
        force = squeeze*direction + np.array([0., 0., upward])
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

    from tools import setupwithpybullet, rununtil
    from config import DT

    robot, sim, cube = setupwithpybullet()

    try:
        check_setup(robot, sim)

        # Local lift-and-return check; qe is 2 cm above the starting grasp.
        # Replace this local path with computepath for the full transport task.
        lift, frames = build_lift_reference(robot, cube)
        q0, qe = lift.points[0].copy(), lift.points[-1].copy()
        path = [q0, qe]

        #setting initial configuration
        sim.setqsim(q0)
        check_initial_state(sim, q0)

        #0 init and end velocities
        def maketraj(path, T):
            return PathTrajectory(path, max_velocity=.12,
                                  max_acceleration=.25, min_segment_time=T)

        outward = maketraj(path, 3.)
        with cycle_log(__file__, robot) as (writer, stream):
            trajs = ControlCycle(robot, sim, outward, frames, writer, tracking_torque)
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

        print_result(trajs, total_time)
    except Exception as exc:
        print(f'[CONTROL STOPPED] {type(exc).__name__}: {exc}')
        raise
    finally:
        pause_simulation()
