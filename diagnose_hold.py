"""Collect model/contact evidence while running the unchanged check_hold test.

Run in the full lab directory: python diagnose_hold.py
Needs the existing check_hold.py and controller.py. Does not change gains,
collision filters, masses or simulation timing. Adds start/end diagnostics.
"""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

import check_hold


def bullet_static_gravity(sim):
    """Joint gravity compensation from loaded masses, COMs and joint axes.

    For each revolute joint, sum gravitational moments of all descendant
    link COMs about its world-space axis; negate to get holding torque.
    This is a static diagnostic, not a replacement dynamics controller.
    LinkState index 4 supplies the joint/URDF link origin; index 1 supplies
    the world inertial-frame orientation for Bullet's reported joint axis.
    Axis convention checked against Bullet's BodyJointInfoUtility.h and
    examples/Importers/ImportURDFDemo/URDF2Bullet.cpp in bulletphysics/bullet3.
    """
    p = sim.p
    count = p.getNumJoints(sim.robot)
    infos = [p.getJointInfo(sim.robot, i) for i in range(count)]
    states = [p.getLinkState(sim.robot, i, computeForwardKinematics=True)
              for i in range(count)]
    masses = [p.getDynamicsInfo(sim.robot, i)[0] for i in range(count)]
    gravity = np.array([0.0, 0.0, sim.pybulletConfigs["gravity"]])
    output = []
    for joint in sim.bulletCtrlJointsInPinOrder:
        if infos[joint][2] != p.JOINT_REVOLUTE:
            raise ValueError("Static moment diagnostic expects revolute joints")
        origin = np.asarray(states[joint][4])
        rotation = np.asarray(p.getMatrixFromQuaternion(states[joint][1])).reshape(3, 3)
        axis = rotation @ np.asarray(infos[joint][13])
        moment = 0.0
        for link in range(count):
            ancestor = link
            while ancestor != -1 and ancestor != joint:
                ancestor = infos[ancestor][16]
            if ancestor == joint:
                arm = np.asarray(states[link][0]) - origin
                moment += np.dot(axis, np.cross(arm, masses[link] * gravity))
        output.append(-moment)
    return np.asarray(output)


def snapshot(robot, sim, label, torques):
    q, v = sim.getpybulletstate()
    names = list(robot.model.names[1:])
    g_pin = np.asarray(pin.computeGeneralizedGravity(robot.model, robot.data, q)).copy()
    result = {
        "label": label,
        "pin_gravity_linear": np.asarray(robot.model.gravity.linear).tolist(),
        "base_pose": sim.p.getBasePositionAndOrientation(sim.robot),
        "base_mass_kg": sim.p.getDynamicsInfo(sim.robot, -1)[0],
        "joints": [], "contacts": [],
    }
    try:
        g_bullet = bullet_static_gravity(sim)
        result["max_static_gravity_difference_Nm"] = float(np.max(np.abs(g_bullet - g_pin)))
        print(f"[DIAG {label}] max static gravity difference = "
              f"{result['max_static_gravity_difference_Nm']:.6f} Nm")
    except Exception as error:
        result["bullet_static_gravity_error"] = str(error)
        g_bullet = None
        print(f"[DIAG {label}] static comparison unavailable: {error}")

    for i, (name, joint) in enumerate(zip(names, sim.bulletCtrlJointsInPinOrder)):
        info = sim.p.getJointInfo(sim.robot, joint)
        record = {
            "name": name, "q_rad": float(q[i]), "v_rad_s": float(v[i]),
            "pin_gravity_Nm": float(g_pin[i]),
            "commanded_torque_Nm": float(torques[i]),
            "bullet_lower_rad": info[8], "bullet_upper_rad": info[9],
            "pin_lower_rad": float(robot.model.lowerPositionLimit[i]),
            "pin_upper_rad": float(robot.model.upperPositionLimit[i]),
            "bullet_damping": info[6], "bullet_friction": info[7],
        }
        if g_bullet is not None:
            record["bullet_gravity_Nm"] = float(g_bullet[i])
            record["gravity_difference_Nm"] = float(g_bullet[i] - g_pin[i])
        result["joints"].append(record)

    labels = {sim.robot: "robot", sim.cubeId: "cube", sim.tableId: "table",
              sim.obstacleId: "obstacle"}
    if sim.floor is not None:
        labels[sim.floor] = "floor"
    for contact in sim.p.getContactPoints(bodyA=sim.robot):
        body_a, body_b, link_a, link_b = contact[1:5]
        def link_name(body, link):
            if link == -1:
                return "base"
            return sim.p.getJointInfo(body, link)[12].decode()
        result["contacts"].append({
            "body_a": labels.get(body_a, str(body_a)),
            "link_a": link_name(body_a, link_a),
            "body_b": labels.get(body_b, str(body_b)),
            "link_b": link_name(body_b, link_b),
            "distance_m": contact[8], "normal_force_N": contact[9],
            "normal_on_b": contact[7],
        })
    loaded = [c for c in result["contacts"] if c["normal_force_N"] > 1e-4]
    print(f"[DIAG {label}] robot contact points: {len(result['contacts'])}; "
          f"loaded points: {len(loaded)}")
    for contact in sorted(loaded, key=lambda c: c["normal_force_N"], reverse=True)[:8]:
        print(f"  {contact['link_a']} -> {contact['body_b']}/{contact['link_b']}: "
              f"{contact['normal_force_N']:.4f} N")
    return result


def main():
    directory = Path(__file__).resolve().parent / "part_b_logs"
    directory.mkdir(exist_ok=True)
    filename = directory / ("diagnose_hold_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    records = []
    expected_steps = int(np.ceil(check_hold.DURATION / check_hold.DT))
    original_setup = check_hold.setupwithpybullet

    def save():
        filename.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def setup():
        robot, sim, cube = original_setup()
        original_step = sim.step
        state = {"steps": 0}

        def step(torques=None):
            if state["steps"] == 0:
                records.append(snapshot(robot, sim, "initial", torques))
                save()
            original_step(torques)
            state["steps"] += 1
            if state["steps"] == expected_steps:
                records.append(snapshot(robot, sim, "final", torques))
                save()

        sim.step = step
        return robot, sim, cube

    check_hold.setupwithpybullet = setup
    print(f"[DIAG] Output: {filename}")
    print("[DIAG] Running the same controller; no physics settings are changed.")
    try:
        check_hold.main()
    finally:
        check_hold.setupwithpybullet = original_setup
        save()
        print(f"[DIAG] Saved: {filename}")


if __name__ == "__main__":
    main()
