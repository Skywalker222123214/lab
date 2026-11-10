#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement
DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    """Find a grasp configuration using iterative inverse kinematics."""

    setcubeplacement(robot, cube, cubetarget)

    # Start within the joint limits.
    q = projecttojointlimits(robot, qcurrent.copy())

    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)
    left_hook_id = cube.model.getFrameId(LEFT_HOOK)
    right_hook_id = cube.model.getFrameId(RIGHT_HOOK)

    # The target poses.
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    target_left = cubetarget * cube.data.oMf[left_hook_id]
    target_right = cubetarget * cube.data.oMf[right_hook_id]

    max_iterations = 500
    step_size = 0.1
    

    position_tolerance = 1e-3 # metres
    orientation_tolerance = 1e-3 # radians

    for iteration in range(max_iterations + 1):

        # 1. Compute the current hand poses.
        pin.framesForwardKinematics(robot.model, robot.data, q)
        current_left = robot.data.oMf[left_hand_id].copy()
        current_right = robot.data.oMf[right_hand_id].copy()

        # 2. Compute the pose errors.
        left_relative = current_left.inverse() * target_left
        right_relative = current_right.inverse() * target_right

        error_left = pin.log(left_relative).vector
        error_right = pin.log(right_relative).vector
        error = np.concatenate([error_left, error_right])

        # Check position and orientation separately for both hands.
        position_error = max(
            norm(left_relative.translation),
            norm(right_relative.translation)
            )

        orientation_error = max(
            norm(error_left[3:]),
            norm(error_right[3:])
            )

        if iteration % 50 == 0:
            debug_print(
                f"Iteration {iteration}: "
                f"position = {position_error:.6f} m, "
                f"orientation = {orientation_error:.6f} rad"
            )

        # 3. Stop if both hands are aligned.
        if (position_error < position_tolerance
            and orientation_error < orientation_tolerance):

            in_collision = collision(robot, q)

            debug_print(f"Hands aligned after {iteration} iterations.")
            debug_print("Collision:", in_collision)

            return q, not in_collision

        # Stop before making any further adjustment.
        if iteration == max_iterations:
            break

        # 4. Compute the hand Jacobians.
        J_hand_left = pin.computeFrameJacobian(
            robot.model,
            robot.data,
            q,
            left_hand_id,
            pin.LOCAL
        ).copy()

        J_hand_right = pin.computeFrameJacobian(
            robot.model,
            robot.data,
            q,
            right_hand_id,
            pin.LOCAL
        ).copy()

       
        J_hand = np.vstack([J_hand_left, J_hand_right])

        #request hand motion towards the targets.(from tutorial3)
        dq = pinv(J_hand) @ error

        q = pin.integrate(robot.model, q, step_size * dq)
        q = projecttojointlimits(robot, q)

    debug_print("Reached the iteration limit without aligning both hands.")
    return q, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    print("Initial grasp success:", successinit)
    print("Goal grasp success:", successend)

    # Display the initial grasp with the cube at its initial pose.
    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    pin.framesForwardKinematics(robot.model, robot.data, q0)
    updatevisuals(viz, robot, cube, q0)

    
    
    
