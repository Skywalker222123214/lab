#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""
import time

import pinocchio as pin
import numpy as np
from numpy.linalg import norm

from inverse_geometry import computeqgrasppose
from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from tools import collision, jointlimitsviolated, setcubeplacement

def samplegrasp(
    robot, cube, qseed,
    cube_start, cube_goal,
    max_attempts=20
):
    """Sample a cube pose and find a valid grasp configuration."""

    # Sample around the start and goal positions.
    lower = np.minimum(
        cube_start.translation, cube_goal.translation
    ) + np.array([-0.05, -0.05, 0.0])

    upper = np.maximum(
        cube_start.translation, cube_goal.translation
    ) + np.array([0.05, 0.05, 0.15])

    for _ in range(max_attempts):

        # First version: vary position and keep the start orientation.
        cube_sample = cube_start.copy()
        cube_sample.translation = np.random.uniform(lower, upper)

        q_sample, success = computeqgrasppose(
            robot, qseed, cube, cube_sample
        )

        if success:
            return q_sample, cube_sample, True

    return None, None, False

def distance(q1,q2):
    """Return the Euclidean distance between two configurations"""
    return norm(q2-q1)

def NEAREST_VERTEX(G, q_rand):
    """Return the index of the tree node closest to q_rand."""

    if not G:
        raise ValueError("The tree must contain at least one node.")

    nearest_index = 0
    nearest_distance = distance(G[0][1], q_rand)

    for i, node in enumerate(G[1:], start=1):
        candidate_distance = distance(node[1], q_rand)

        if candidate_distance < nearest_distance:
            nearest_distance = candidate_distance
            nearest_index = i

    return nearest_index

def grasp_step(
    robot, cube,
    q_near, cube_near, cube_sample,
    alpha=0.05,
    max_joint_step=0.15
):
    """Try one small step towards a sampled cube pose."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    # Interpolate the cube pose towards the sample.
    cube_next = pin.SE3.Interpolate(
        cube_near, cube_sample, alpha
    )

    # Use the previous grasp configuration as the initial guess.
    q_next, success = computeqgrasppose(
        robot, q_near, cube, cube_next
    )

    if not success:
        return None, None, False

    # Reject a large jump in the robot configuration.
    joint_step = distance(q_near, q_next)

    if joint_step > max_joint_step:
        return None, None, False

    return q_next, cube_next, True

def check_grasp_edge(
    robot, cube,
    q_start, q_end,
    cube_start, cube_end,
    joint_resolution=0.02,
    translation_resolution=0.005,
    rotation_resolution=0.05,
    position_tolerance=1e-3,
    orientation_tolerance=1e-3
):
    """Check a discretised edge and return its sampled states."""

    if min(
        joint_resolution,
        translation_resolution,
        rotation_resolution
    ) <= 0:
        raise ValueError("Resolutions must be positive.")

    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)

    left_hook_id = cube.model.getFrameId(LEFT_HOOK)
    right_hook_id = cube.model.getFrameId(RIGHT_HOOK)

    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    cube_to_left = cube.data.oMf[left_hook_id].copy()
    cube_to_right = cube.data.oMf[right_hook_id].copy()

    # Choose the number of intervals from the size of the motion.
    cube_motion = pin.log(cube_start.inverse() * cube_end).vector

    n_steps = max(
        4,
        int(np.ceil(distance(q_start, q_end) / joint_resolution)),
        int(np.ceil(norm(cube_motion[:3]) / translation_resolution)),
        int(np.ceil(norm(cube_motion[3:]) / rotation_resolution))
    )

    edge_qs = []
    edge_cubes = []

    # Include BOTH endpoints.
    for i in range(n_steps + 1):
        t = i / n_steps

        # Linear joint interpolation, following Tutorial 4.
        # Additional grasp checks are specific to CW2.
        q = (1.0 - t) * q_start + t * q_end
        cube_pose = pin.SE3.Interpolate(cube_start, cube_end, t)

        if jointlimitsviolated(robot, q):
            return [], [], False

        setcubeplacement(robot, cube, cube_pose)
        pin.framesForwardKinematics(robot.model, robot.data, q)

        current_left = robot.data.oMf[left_hand_id].copy()
        current_right = robot.data.oMf[right_hand_id].copy()

        target_left = cube_pose * cube_to_left
        target_right = cube_pose * cube_to_right

        left_relative = current_left.inverse() * target_left
        right_relative = current_right.inverse() * target_right

        position_error = max(
            norm(left_relative.translation),
            norm(right_relative.translation)
        )

        orientation_error = max(
            norm(pin.log(left_relative).vector[3:]),
            norm(pin.log(right_relative).vector[3:])
        )

        if (
            position_error >= position_tolerance
            or orientation_error >= orientation_tolerance
        ):
            return [], [], False

        if collision(robot, q):
            return [], [], False

        edge_qs.append(q.copy())
        edge_cubes.append(cube_pose.copy())

    return edge_qs, edge_cubes, True
def connect_grasp_projected(
    robot, cube,
    q_start, q_goal,
    cube_start, cube_goal,
    joint_resolution=0.03,
    translation_resolution=0.003,
    rotation_resolution=0.03,
    max_joint_step=0.15
):
    """Try a connection using intermediate grasp configurations."""

    cube_motion = pin.log(
        cube_start.inverse() * cube_goal
    ).vector

    n_steps = max(
        1,
        int(np.ceil(distance(q_start, q_goal) / joint_resolution)),
        int(np.ceil(norm(cube_motion[:3]) / translation_resolution)),
        int(np.ceil(norm(cube_motion[3:]) / rotation_resolution))
    )

    path_qs = [q_start.copy()]
    path_cubes = [cube_start.copy()]

    q_previous = q_start.copy()
    cube_previous = cube_start.copy()

    for i in range(1, n_steps + 1):
        t = i / n_steps

        if i == n_steps:
            # Finish at the exact requested goal configuration.
            q_next = q_goal.copy()
            cube_next = cube_goal.copy()
        else:
            cube_next = pin.SE3.Interpolate(
                cube_start, cube_goal, t
            )

            # Predict towards q_goal, then correct the grasp using IK.
            fraction = 1.0 / (n_steps - i + 1)
            q_guess = (
                (1.0 - fraction) * q_previous
                + fraction * q_goal
            )

            q_next, success = computeqgrasppose(
                robot, q_guess, cube, cube_next
            )

            if not success:
                return [], [], False

        # Reject jumps between different IK solutions.
        if distance(q_previous, q_next) > max_joint_step:
            return [], [], False

        # Check the actual small connection, including both endpoints.
        edge_qs, edge_cubes, success = check_grasp_edge(
            robot, cube,
            q_previous, q_next,
            cube_previous, cube_next
        )

        if not success:
            return [], [], False

        # The first point is already stored.
        path_qs.extend(edge_qs[1:])
        path_cubes.extend(edge_cubes[1:])

        q_previous = q_next.copy()
        cube_previous = cube_next.copy()

    return path_qs, path_cubes, True

def rrt(
    robot, cube,
    q_init, q_goal,
    cube_start, cube_goal,
    max_iterations=300,
    goal_bias=0.2
):
    """Build a grasp-constrained tree and try to reach q_goal."""

    # Adapted from ARO Tutorial 4 (4_rrt.ipynb).
    # Added cube poses, grasp constraints, and stored edge samples.
    G = [(None, q_init.copy(), cube_start.copy())]
    edges = {}

    for iteration in range(max_iterations):
        if iteration % 200 == 0:
            print(
                f"[RRT] {iteration}/{max_iterations}, nodes={len(G)}"
            )

        # 1. Sometimes aim directly towards the goal.
        if np.random.rand() < goal_bias:
            q_rand = q_goal.copy()
            cube_rand = cube_goal.copy()
        else:
            q_rand, cube_rand, sample_success = samplegrasp(
                robot, cube, q_init,
                cube_start, cube_goal
            )

            if not sample_success:
                continue

        # 2. Find the nearest existing node.
        near_index = NEAREST_VERTEX(G, q_rand)
        q_near = G[near_index][1]
        cube_near = G[near_index][2]

        # 3. Try a small extension.
        q_new, cube_new, step_success = grasp_step(
            robot, cube,
            q_near, cube_near, cube_rand
        )

        if not step_success:
            continue

        # Avoid adding a node that makes essentially no progress.
        if distance(q_near, q_new) < 1e-6:
            continue

        # 4. Check the connection before adding the node.
        edge_qs, edge_cubes, edge_success = check_grasp_edge(
            robot, cube,
            q_near, q_new,
            cube_near, cube_new
        )

        if not edge_success:
            continue

        # 5. Save the new node and its checked incoming edge.
        new_index = len(G)

        G.append((
            near_index,
            q_new.copy(),
            cube_new.copy()
        ))

        edges[new_index] = (edge_qs, edge_cubes)

        # 6. Try to connect this node to the EXACT goal configuration.
        goal_qs, goal_cubes, goal_success = connect_grasp_projected(
            robot, cube,
            q_new, q_goal,
            cube_new, cube_goal
        )

        if goal_success:
            goal_index = len(G)

            G.append((
                new_index,
                q_goal.copy(),
                cube_goal.copy()
            ))

            edges[goal_index] = (goal_qs, goal_cubes)

            return G, edges, True

    return G, edges, False


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(
    qinit, qgoal, cubeplacementq0, cubeplacementqgoal,
    robot=None, cube=None,
    max_iterations=2000, goal_bias=0.2,
    cube_path_out=None
):
    """Return a list of checked configurations, or [] on planning failure.

    Optional cube_path_out is filled with the corresponding stored cube poses
    for visualization. It does not change the return type of this function.
    """
    if robot is None or cube is None:
        raise ValueError("Pass the initialized robot and cube to computepath.")
    if cube_path_out is not None:
        cube_path_out.clear()

    G, edges, found = rrt(
        robot, cube, qinit, qgoal,
        cubeplacementq0, cubeplacementqgoal,
        max_iterations=max_iterations,
        goal_bias=goal_bias
    )

    if not found:
        return []

    path, checked_cube_path = getpath(G, edges)
    if not np.allclose(path[0], qinit, rtol=0, atol=1e-9):
        raise RuntimeError("Recovered path does not start at qinit.")
    if not np.allclose(path[-1], qgoal, rtol=0, atol=1e-9):
        raise RuntimeError("Recovered path does not end at qgoal.")

    if cube_path_out is not None:
        cube_path_out.extend(pose.copy() for pose in checked_cube_path)

    return path


def getpath(G, edges):
    """Recover the checked root-to-goal path; call only after RRT succeeds."""
    if not G:
        raise ValueError("The tree is empty.")

    branch = []
    index = len(G) - 1

    while G[index][0] is not None:
        parent = G[index][0]
        if not 0 <= parent < index:
            raise ValueError("Invalid parent index in the tree.")
        branch.append(index)
        index = parent

    if index != 0:
        raise ValueError("The goal branch does not reach the root.")

    branch.reverse()
    path = [G[0][1].copy()]
    cube_path = [G[0][2].copy()]

    for child_index in branch:
        edge_qs, edge_cubes = edges[child_index]
        if len(edge_qs) < 2 or len(edge_qs) != len(edge_cubes):
            raise ValueError("Invalid stored edge.")

        # The start of each edge is already in the path.
        path.extend(q.copy() for q in edge_qs[1:])
        cube_path.extend(pose.copy() for pose in edge_cubes[1:])

    return path, cube_path


def displaypath(robot, path, dt, viz, cube=None, cube_path=None):
    """Display stored configurations and optionally their matching cube poses."""
    from setup_meshcat import updatevisuals

    if dt < 0:
        raise ValueError("Playback delay must be nonnegative.")
    if (cube is None) != (cube_path is None):
        raise ValueError("Provide both cube and cube_path, or neither.")
    if cube_path is not None and len(path) != len(cube_path):
        raise ValueError("Robot and cube paths must have the same length.")

    for i, q in enumerate(path):
        if cube is None:
            viz.display(q)
        else:
            setcubeplacement(robot, cube, cube_path[i])
            pin.framesForwardKinematics(robot.model, robot.data, q)
            updatevisuals(viz, robot, cube, q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    # 1. Load the robot, cube, and visualizer.
    robot, cube, viz = setupwithmeshcat()

    q_seed = robot.q0.copy()

    # 2. Compute the initial grasp configuration.
    q0, successinit = computeqgrasppose(
        robot, q_seed, cube, CUBE_PLACEMENT, viz
    )

    # 3. Compute the goal grasp configuration.
    qe, successend = computeqgrasppose(
        robot, q_seed, cube, CUBE_PLACEMENT_TARGET, viz
    )

    print("Initial grasp success:", successinit)
    print("Goal grasp success:", successend)

    if not (successinit and successend):
        raise RuntimeError(
            "Cannot start RRT: initial or goal grasp is invalid."
        )

    # 4. Display the initial state before planning.
    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    pin.framesForwardKinematics(robot.model, robot.data, q0)
    updatevisuals(viz, robot, cube, q0)

    # 5. Plan through the coursework entry point.
    print("\n--- Computing path ---")
    cube_path = []
    path = computepath(
        q0, qe,
        CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET,
        robot=robot, cube=cube,
        max_iterations=2000,
        goal_bias=0.2,
        cube_path_out=cube_path
    )

    # 6. Play the returned path through the display entry point.
    if path:
        print(f"Path found: {len(path)} configurations.")
        displaypath(
            robot, path, dt=0.03, viz=viz,
            cube=cube, cube_path=cube_path
        )
    else:
        setcubeplacement(robot, cube, CUBE_PLACEMENT)
        pin.framesForwardKinematics(robot.model, robot.data, q0)
        updatevisuals(viz, robot, cube, q0)
        print("No path found; showing the initial pose.")
