"""Repeat the validated lift, then inspect contacts in the paused scene.

Run in the full lab: python view_grasp_contacts.py
No changes to controller, dynamics, collisions, or simulation step count.
Yellow wires are nominal URDF wrist collision boxes/cylinders; not Bullet's
internal expanded margins. Red/green marks are actual reported contact points.
An optional second view hides wrist visual meshes only (no physics changes).
"""

import builtins
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

import check_lift
from check_grip import hand_link_groups
from config import NEXTAGE_URDF


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def shape_segments(geometry):
    """Return local wireframe segments for the supplied primitive geometry."""
    box = geometry.find('box')
    cylinder = geometry.find('cylinder')
    if box is not None:
        half = np.array([float(x) for x in box.get('size').split()])/2
        vertices = [half*np.array([1 if i & (1<<j) else -1 for j in range(3)])
                    for i in range(8)]
        return [(vertices[i], vertices[i^(1<<j)]) for i in range(8)
                for j in range(3) if not i & (1<<j)]
    if cylinder is not None:
        radius, half = float(cylinder.get('radius')), float(cylinder.get('length'))/2
        angles = np.linspace(0, 2*np.pi, 33)
        segments = []
        for z in (-half, half):
            circle = [np.array([radius*np.cos(a), radius*np.sin(a), z]) for a in angles]
            segments += list(zip(circle[:-1], circle[1:]))
        for a in angles[:-1:8]:
            segments.append((np.array([radius*np.cos(a),radius*np.sin(a),-half]),
                             np.array([radius*np.cos(a),radius*np.sin(a),half])))
        return segments
    return []


def draw_line(p, a, b, color, width=2.):
    return p.addUserDebugLine(np.asarray(a).tolist(), np.asarray(b).tolist(),
                              lineColorRGB=color, lineWidth=width, lifeTime=0)


def draw_cross(p, point, color):
    point = np.asarray(point)
    for axis in np.eye(3):
        draw_line(p, point-.0015*axis, point+.0015*axis, color, 4.)


def show_contacts(sim):
    p = sim.p
    groups, _ = hand_link_groups(sim)
    urdf = ET.parse(NEXTAGE_URDF).getroot()
    drawn = 0
    for index in sorted(groups[0] | groups[1]):
        name = p.getJointInfo(sim.robot, index)[12].decode()
        link = urdf.find("link[@name='"+name+"']")
        if link is None:
            continue
        state = p.getLinkState(sim.robot, index, computeForwardKinematics=True)
        world_r = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3,3)
        world_t = np.asarray(state[4])
        for collision in link.findall('collision'):
            origin = collision.find('origin')
            xyz = np.zeros(3) if origin is None else np.array([float(x) for x in origin.get('xyz','0 0 0').split()])
            rpy = np.zeros(3) if origin is None else np.array([float(x) for x in origin.get('rpy','0 0 0').split()])
            rotation = world_r @ rpy_matrix(rpy)
            translation = world_t + world_r @ xyz
            for a, b in shape_segments(collision.find('geometry')):
                draw_line(p, rotation@a+translation, rotation@b+translation, [1.,.8,0.])
                drawn += 1

    report = {"urdf": str(NEXTAGE_URDF), "nominal_collision_wire_segments": drawn,
              "contacts": [], "normal_force_sum_N": {"left":0., "right":0.}}
    for contact in p.getContactPoints(bodyA=sim.robot, bodyB=sim.cubeId):
        if contact[9] <= 1e-4:
            continue
        side = 'left' if contact[3] in groups[0] else 'right' if contact[3] in groups[1] else 'other'
        if side == 'other':
            continue
        pa, pb = np.asarray(contact[5]), np.asarray(contact[6])
        draw_cross(p, pa, [1.,0.,0.])
        draw_cross(p, pb, [0.,1.,0.])
        draw_line(p, pb, pb+.015*np.asarray(contact[7]), [.1,.5,1.], 2.)
        properties = p.getDynamicsInfo(sim.robot, contact[3])
        record = {
            "side": side,
            "hand_link": p.getJointInfo(sim.robot, contact[3])[12].decode(),
            "hand_point_world_m": pa.tolist(), "cube_point_world_m": pb.tolist(),
            "signed_contact_distance_m": contact[8],
            "point_separation_m": float(np.linalg.norm(pa-pb)),
            "normal_force_N": contact[9],
            "hand_collision_margin_m": properties[11] if len(properties)>11 else None,
        }
        report['contacts'].append(record)
        report['normal_force_sum_N'][side] += contact[9]
        print(f"[CONTACT VIEW] {side}: {contact[9]:.4f} N, "
              f"signed gap={1000*contact[8]:.4f} mm, "
              f"point separation={1000*np.linalg.norm(pa-pb):.4f} mm")
    directory = Path(__file__).resolve().parent/'part_b_logs'
    directory.mkdir(exist_ok=True)
    filename = directory/('contact_view_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    filename.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"[CONTACT VIEW] L/R normal force sums: {report['normal_force_sum_N']}")
    print(f"[CONTACT VIEW] Contact report: {filename}")
    print('[CONTACT VIEW] YELLOW: nominal wrist collision shapes. RED: hand contact points.')
    print('[CONTACT VIEW] GREEN: cube contact points. BLUE: contact normals (fixed display length).')
    print('[CONTACT VIEW] Physics remains paused; view changes do not advance simulation.')
    return groups


def main():
    original_setup = check_lift.setupwithpybullet
    missing = object()
    original_input = check_lift.__dict__.get('input', missing)
    context = {}

    def setup():
        robot, sim, cube = original_setup()
        context['sim'] = sim
        return robot, sim, cube

    def inspect_pause(prompt):
        sim = context['sim']
        groups = show_contacts(sim)
        builtins.input('\nVIEW 1: inspect or capture screenshot. Press Enter for collision-only wrist view...')
        for index in sorted(groups[0] | groups[1]):
            sim.p.changeVisualShape(sim.robot, index, rgbaColor=[.7,.7,.7,0.])
        builtins.input('\nVIEW 2: wrist visual meshes hidden; physics unchanged. Inspect/screenshot, then Enter to close...')
        return ''

    check_lift.setupwithpybullet = setup
    check_lift.input = inspect_pause
    try:
        check_lift.main()
    finally:
        check_lift.setupwithpybullet = original_setup
        if original_input is missing:
            del check_lift.input
        else:
            check_lift.input = original_input


if __name__ == '__main__':
    main()
