#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv
from inverse_geometry import computeqgrasppose
from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from numpy.linalg import norm 
import time
from tools import collision, jointlimitsviolated, setcubeplacement
DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Diagnostic additions: the planning equations and acceptance thresholds below
# are unchanged. Only printing and failure bookkeeping have been changed.
import io
from collections import Counter
from contextlib import redirect_stdout

DIAG = Counter()
COLLISION_COUNTS = Counter()
LAST_FAILURE = {}


def record_collisions(robot, stage):
    """Read results immediately after collision(robot, q) returned True."""
    for pair, result in zip(
        robot.collision_model.collisionPairs,
        robot.collision_data.collisionResults
    ):
        if result.isCollision():
            first = robot.collision_model.geometryObjects[pair.first].name
            second = robot.collision_model.geometryObjects[pair.second].name
            names = tuple(sorted((str(first), str(second))))
            COLLISION_COUNTS[(stage, names)] += 1


def grasp_errors(robot, cube, q, cube_pose):
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    positions, angles = [], []
    for hand, hook in ((LEFT_HAND, LEFT_HOOK), (RIGHT_HAND, RIGHT_HOOK)):
        actual = robot.data.oMf[robot.model.getFrameId(hand)].copy()
        desired = cube_pose * cube.data.oMf[cube.model.getFrameId(hook)]
        relative = actual.inverse() * desired
        positions.append(float(norm(relative.translation)))
        angles.append(float(norm(pin.log(relative).vector[3:])))
    return max(positions), max(angles)


def solve_grasp_quiet(robot, qcurrent, cube, cubetarget, viz=None, *, stage='endpoint'):
    """Keep the existing IK unchanged; silence only its stdout, not exceptions."""
    DIAG[f'{stage}.ik_calls'] += 1
    if DEBUG:
        q, success = computeqgrasppose(robot, qcurrent, cube, cubetarget, viz)
    else:
        with redirect_stdout(io.StringIO()):
            q, success = computeqgrasppose(robot, qcurrent, cube, cubetarget, viz)

    if success:
        return q, success

    if q is None or not np.all(np.isfinite(q)):
        DIAG[f'{stage}.ik_invalid_result'] += 1
        return q, success

    position, angle = grasp_errors(robot, cube, q, cubetarget)
    LAST_FAILURE[f'{stage}.ik'] = f'position={position:.6g}m, angle={angle:.6g}rad'
    # These diagnostic thresholds match the user's current IK/edge tolerances.
    if position >= 1e-3 or angle >= 1e-3:
        DIAG[f'{stage}.ik_not_aligned'] += 1
    else:
        setcubeplacement(robot, cube, cubetarget)
        if collision(robot, q):
            DIAG[f'{stage}.ik_collision'] += 1
            record_collisions(robot, f'{stage}.ik')
        else:
            DIAG[f'{stage}.ik_other_failure'] += 1
    return q, success


def report_progress(G, q_goal, cube_goal, iteration, maximum):
    best_q = min(distance(node[1], q_goal) for node in G)
    best_cube = min(norm(node[2].translation - cube_goal.translation) for node in G)
    print(
        f'[RRT] {iteration}/{maximum} nodes={len(G)} '
        f'best_q={best_q:.4f} best_cube={best_cube:.4f}m '
        f'step_failed={DIAG["step_failed"]} '
        f'growth_edge_failed={DIAG["growth_edge_failed"]} '
        f'goal_attempts={DIAG["goal_attempts"]}'
    )


def report_final(G, q_goal, cube_goal, iteration, maximum):
    report_progress(G, q_goal, cube_goal, iteration, maximum)
    print('[Diagnostics: counts overlap; one attempt can contribute to several counters]')
    for name, count in sorted(DIAG.items()):
        if count:
            print(f'  {name}: {count}')
    print('[Last failure details by stage]')
    for name, detail in sorted(LAST_FAILURE.items()):
        print(f'  {name}: {detail}')
    if COLLISION_COUNTS:
        print('[Most frequent collision pairs, separated by stage]')
        for stage in sorted({key[0] for key in COLLISION_COUNTS}):
            pairs = [(names, count) for (label, names), count in COLLISION_COUNTS.items() if label == stage]
            for names, count in sorted(pairs, key=lambda item: item[1], reverse=True)[:3]:
                print(f'  {stage}: {names[0]} <-> {names[1]} ({count})')
    positions = np.array([node[2].translation for node in G])
    print('[Tree cube-position extent]')
    print('  min xyz:', positions.min(axis=0))
    print('  max xyz:', positions.max(axis=0))


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

    for attempt in range(max_attempts):

        # First version: vary position and keep the start orientation.
        cube_sample = cube_start.copy()
        cube_sample.translation = np.random.uniform(lower, upper)

        q_sample, success = solve_grasp_quiet(
            robot, qseed, cube, cube_sample, stage="sample"
        )

        if success:
            DIAG["sample_valid"] += 1
            debug_print(f"Valid sample found on attempt {attempt + 1}.")
            return q_sample, cube_sample, True

    DIAG["sample_exhausted"] += 1
    debug_print("No valid sample found within the attempt limit.")
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
    q_next, success = solve_grasp_quiet(
        robot, q_near, cube, cube_next, stage="step"
    )

    if not success:
        debug_print("Step rejected: no valid grasp found.")
        return None, None, False

    # Reject a large jump in the robot configuration.
    joint_step = distance(q_near, q_next)

    if joint_step > max_joint_step:
        DIAG["step_joint_jump"] += 1
        LAST_FAILURE["step_joint_jump"] = f"distance={joint_step:.6g}"
        debug_print(f"Step rejected: joint-space jump = {joint_step:.6f}")
        return None, None, False

    debug_print(f"Step accepted: joint-space change = {joint_step:.6f}")
    return q_next, cube_next, True

def check_grasp_edge(
    robot, cube,
    q_start, q_end,
    cube_start, cube_end,
    joint_resolution=0.02,
    translation_resolution=0.005,
    rotation_resolution=0.05,
    position_tolerance=1e-3,
    orientation_tolerance=1e-3,
    context="growth"
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
        1,
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
            DIAG[f"{context}.edge_joint_limits"] += 1
            debug_print(f"Edge rejected at t={t:.3f}: joint limits.")
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
            DIAG[f"{context}.edge_grasp"] += 1
            LAST_FAILURE[f"{context}.edge_grasp"] = (
                f"t={t:.6g}, position={position_error:.6g}m, "
                f"angle={orientation_error:.6g}rad"
            )
            debug_print(
                f"Edge rejected at t={t:.3f}: grasp error, "
                f"position={position_error:.6f} m, "
                f"orientation={orientation_error:.6f} rad"
            )
            return [], [], False

        if collision(robot, q):
            DIAG[f"{context}.edge_collision"] += 1
            record_collisions(robot, f"{context}.edge")
            debug_print(f"Edge rejected at t={t:.3f}: collision.")
            return [], [], False

        edge_qs.append(q.copy())
        edge_cubes.append(cube_pose.copy())

    debug_print(f"Edge accepted at all {n_steps + 1} checked states.")
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

            q_next, success = solve_grasp_quiet(
                robot, q_guess, cube, cube_next, stage="goal"
            )

            if not success:
                debug_print("Goal connection failed: intermediate IK.")
                return [], [], False

        # Reject jumps between different IK solutions.
        if distance(q_previous, q_next) > max_joint_step:
            DIAG["goal_joint_jump"] += 1
            LAST_FAILURE["goal_joint_jump"] = f"distance={distance(q_previous, q_next):.6g}, t={t:.6g}"
            debug_print("Goal connection failed: joint-space jump.")
            return [], [], False

        # Check the actual small connection, including both endpoints.
        edge_qs, edge_cubes, success = check_grasp_edge(
            robot, cube,
            q_previous, q_next,
            cube_previous, cube_next,
            context="goal"
        )

        if not success:
            debug_print("Goal connection failed: invalid small edge.")
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
    DIAG.clear()
    COLLISION_COUNTS.clear()
    LAST_FAILURE.clear()

    for iteration in range(max_iterations):
        if iteration % 200 == 0:
            report_progress(G, q_goal, cube_goal, iteration, max_iterations)

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
            DIAG["step_failed"] += 1
            continue

        # Avoid adding a node that makes essentially no progress.
        if distance(q_near, q_new) < 1e-6:
            DIAG["no_progress"] += 1
            continue

        # 4. Check the connection before adding the node.
        edge_qs, edge_cubes, edge_success = check_grasp_edge(
            robot, cube,
            q_near, q_new,
            cube_near, cube_new
        )

        if not edge_success:
            DIAG["growth_edge_failed"] += 1
            continue

        # 5. Save the new node and its checked incoming edge.
        new_index = len(G)

        G.append((
            near_index,
            q_new.copy(),
            cube_new.copy()
        ))

        edges[new_index] = (edge_qs, edge_cubes)

        debug_print(
            f"RRT iteration {iteration + 1}: "
            f"added node {new_index}, "
            f"tree size = {len(G)}"
        )

        # 6. Try to connect this node to the EXACT goal configuration.
        DIAG["goal_attempts"] += 1
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

            report_final(G, q_goal, cube_goal, iteration + 1, max_iterations)
            print(f"Path found! Tree contains {len(G)} nodes.")
            return G, edges, True

    report_final(G, q_goal, cube_goal, max_iterations, max_iterations)
    print(f"No path found within {max_iterations} RRT iterations.")
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
    from tools import setupwithmeshcat, setcubeplacement
    from setup_meshcat import updatevisuals
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    # 1. Load the robot, cube, and visualizer.
    robot, cube, viz = setupwithmeshcat()

    q_seed = robot.q0.copy()

    # 2. Compute the initial grasp configuration.
    print("\n--- Computing initial grasp ---")

    q0, successinit = solve_grasp_quiet(
        robot,
        q_seed,
        cube,
        CUBE_PLACEMENT,
        viz
    )

    # 3. Compute the goal grasp configuration.
    print("\n--- Computing goal grasp ---")

    qe, successend = solve_grasp_quiet(
        robot,
        q_seed,
        cube,
        CUBE_PLACEMENT_TARGET,
        viz
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

    print("Path found:", bool(path))
    print("Path configurations:", len(path))

    # 6. Play the returned path through the display entry point.
    if path:
        displaypath(
            robot, path, dt=0.03, viz=viz,
            cube=cube, cube_path=cube_path
        )
    else:
        setcubeplacement(robot, cube, CUBE_PLACEMENT)
        pin.framesForwardKinematics(robot.model, robot.data, q0)
        updatevisuals(viz, robot, cube, q0)
        print("No path to display; showing the initial pose.")