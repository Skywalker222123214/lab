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

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    """Find a grasp configuration using iterative inverse kinematics."""

    setcubeplacement(robot, cube, cubetarget)

    # Start within the joint limits.
    q = projecttojointlimits(robot, qcurrent.copy())

    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)
    left_hook_id = cube.model.getFrameId(LEFT_HOOK)
    right_hook_id = cube.model.getFrameId(RIGHT_HOOK)

    # The target poses remain fixed throughout the iterations.
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    target_left = cubetarget * cube.data.oMf[left_hook_id]
    target_right = cubetarget * cube.data.oMf[right_hook_id]

    max_iterations = 500
    step_size = 0.1
    damping = 1e-4

    position_tolerance = 1e-3       # metres
    orientation_tolerance = 1e-3    # radians

    for iteration in range(max_iterations + 1):

        # 1. Compute the current hand poses.
        pin.framesForwardKinematics(robot.model, robot.data, q)
        current_left = robot.data.oMf[left_hand_id].copy()
        current_right = robot.data.oMf[right_hand_id].copy()

        # 2. Compute the pose errors.
        left_relative = current_left.inverse() * target_left
        right_relative = current_right.inverse() * target_right

        error_left = pin.log6(left_relative).vector
        error_right = pin.log6(right_relative).vector
        error = np.concatenate([error_left, error_right])

        # Check position and orientation separately for both hands.
        position_error = max(
            np.linalg.norm(current_left.translation - target_left.translation),
            np.linalg.norm(current_right.translation - target_right.translation)
        )

        orientation_error = max(
            np.linalg.norm(error_left[3:]),
            np.linalg.norm(error_right[3:])
        )

        if iteration % 50 == 0:
            print(
                f"Iteration {iteration}: "
                f"position = {position_error:.6f} m, "
                f"orientation = {orientation_error:.6f} rad"
            )

        # 3. Stop if both hands are aligned.
        if (position_error < position_tolerance
                and orientation_error < orientation_tolerance):

            in_collision = collision(robot, q)

            print(f"Hands aligned after {iteration} iterations.")
            print("Collision:", in_collision)

            return q, not in_collision

        # Stop before making any further adjustment.
        if iteration == max_iterations:
            break

        # 4. Compute the hand Jacobians.
        pin.computeJointJacobians(robot.model, robot.data, q)

        J_hand_left = pin.getFrameJacobian(
            robot.model, robot.data,
            left_hand_id, pin.ReferenceFrame.LOCAL
        ).copy()

        J_hand_right = pin.getFrameJacobian(
            robot.model, robot.data,
            right_hand_id, pin.ReferenceFrame.LOCAL
        ).copy()

        # Error Jacobians and damped update adapted from:
        # https://gepetto.github.io/doc/pinocchio/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        J_error_left = -pin.Jlog6(left_relative.inverse()) @ J_hand_left
        J_error_right = -pin.Jlog6(right_relative.inverse()) @ J_hand_right

        J = np.vstack([J_error_left, J_error_right])

        # 5. Compute and apply a small adjustment.
        dq = -J.T @ np.linalg.solve(
            J @ J.T + damping * np.eye(12),
            error
        )

        q = pin.integrate(robot.model, q, step_size * dq)
        q = projecttojointlimits(robot, q)

    print("Reached the iteration limit without aligning both hands.")
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
    updatevisuals(viz, robot, cube, q0)

    
    
    
