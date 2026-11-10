"""Contact checks, local motion schedule and logging used by control.py.

The local cycle lifts 2 cm and returns to its starting position. Full target
transport will use the path planner once the planned path is available.
"""
from contextlib import contextmanager
import csv


from datetime import datetime


from pathlib import Path



import numpy as np


import pinocchio as pin


from config import CUBE_PLACEMENT, DT, LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK


from trajectory import PathTrajectory


from inverse_geometry import computeqgrasppose


from tools import setcubeplacement, collision, jointlimitsviolated


HEIGHT = .02


PRELOAD_END = 4.


LIFT_START = 5.


HOLD_TIME = 2.


TRANSFER_TIME = 2.


UNLOAD_TIME = 3.


FINAL_HOLD = 1.


def hand_link_groups(sim):
    """Physical wrist links and fixed descendants; exclude forearm contacts."""
    infos = [sim.p.getJointInfo(sim.robot, i)
             for i in range(sim.p.getNumJoints(sim.robot))]
    groups, roots = [], []
    for name in ("LARM_JOINT5", "RARM_JOINT5"):
        matches = [i for i, info in enumerate(infos) if info[1].decode() == name]
        if len(matches) != 1:
            raise RuntimeError("Cannot identify physical wrist joint: " + name)
        root = matches[0]
        roots.append(root)
        group = set()
        for i in range(len(infos)):
            ancestor = i
            while ancestor != -1 and ancestor != root:
                ancestor = infos[ancestor][16]
            if ancestor == root:
                group.add(i)
        groups.append(group)
    return groups, roots


def measure_contact(sim, groups, axis):
    normal = np.zeros(2)
    inward = np.zeros(2)
    counts = np.zeros(2, dtype=int)
    # normalOnB points from cube toward hand. Negating gives hand-on-cube force.
    for c in sim.p.getContactPoints(bodyA=sim.robot, bodyB=sim.cubeId):
        for side, (group, direction) in enumerate(zip(groups, (-axis, axis))):
            if c[3] in group:
                normal[side] += c[9]
                inward[side] += max(0., float(np.dot(-np.asarray(c[7])*c[9], direction)))
                counts[side] += int(c[9] > 1e-4)
    table_load = 0.
    for c in sim.p.getContactPoints(bodyA=sim.robot, bodyB=sim.tableId):
        if c[3] in groups[0] or c[3] in groups[1]:
            table_load += c[9]
    return inward, normal, counts, float(table_load)


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


def has_lift_evidence(height, box_table, robot_table, inward, slip):
    """Local experiment criteria, not the lecturer's grading thresholds."""
    return bool(height >= .015 and box_table < .5 and robot_table < .5
                and np.all(np.asarray(inward) > 1.) and slip < .005)


def cycle_reference(t, outward, returning, mass, gravity, squeeze_target):
    """Pure schedule. Return phase, (q,v,a), squeeze and upward force/hand."""
    top = LIFT_START + outward.duration
    down = top + HOLD_TIME
    landed = down + returning.duration
    unload = landed + TRANSFER_TIME
    end_unload = unload + UNLOAD_TIME
    squeeze = squeeze_target * smooth_profile(t-1., 3.)[0]
    support = smooth_profile(t-PRELOAD_END, LIFT_START-PRELOAD_END)[0]
    acceleration = 0.
    if t < LIFT_START:
        reference = outward.sample(0.)
        phase = 'preload' if t < PRELOAD_END else 'support_ramp'
    elif t < top:
        reference = outward.sample(t-LIFT_START)
        acceleration = HEIGHT*smooth_profile(t-LIFT_START, outward.duration)[1]
        phase = 'lifting'
    elif t < down:
        reference = outward.sample(outward.duration)
        phase = 'top_hold'
    elif t < landed:
        reference = returning.sample(t-down)
        acceleration = -HEIGHT*smooth_profile(t-down, returning.duration)[1]
        phase = 'lowering'
    else:
        reference = returning.sample(returning.duration)
        support = 1.-smooth_profile(t-landed, TRANSFER_TIME)[0]
        phase = 'load_transfer'
        if t >= unload:
            squeeze *= 1.-smooth_profile(t-unload, UNLOAD_TIME)[0]
            phase = 'unload' if t < end_unload else 'final_hold'
    return phase, reference, float(squeeze), .5*mass*(gravity*support+acceleration)


class ControlCycle:
    """Runtime state passed through the template's trajs argument."""
    def __init__(self, robot, sim, outward, frames, writer, torque_function):
        self.torque_function = torque_function
        self.outward = outward
        self.returning = PathTrajectory(outward.points[::-1], max_velocity=.12,
                                       max_acceleration=.25, min_segment_time=3.)
        # This milestone has one straight lift segment; its reverse was checked
        # against the same geometry. A general planner needs its own validation.
        if len(outward.points) != 2:
            raise ValueError('Local cycle expects two checked lift endpoints')
        self.frames, self.writer = frames, writer
        self.groups, wrists = hand_link_groups(sim)
        self.mass = sum(sim.p.getDynamicsInfo(sim.cubeId, i)[0]
                        for i in [-1]+list(range(sim.p.getNumJoints(sim.cubeId))))
        mu = sim.p.getDynamicsInfo(sim.cubeId, -1)[1]
        friction = np.array([mu*sim.p.getDynamicsInfo(sim.robot, i)[1] for i in wrists])
        self.gravity = abs(sim.pybulletConfigs['gravity'])
        if (not np.isfinite(self.mass) or self.mass <= 0 or
                not np.isfinite(self.gravity) or self.gravity <= 0 or
                not np.all(np.isfinite(friction)) or np.any(friction <= 0)):
            raise RuntimeError('Invalid physical mass/friction/gravity')
        self.weight = self.mass*self.gravity
        self.squeeze = float(max(5., 1.25*self.weight/(2.*friction.min())))
        self.limits = np.array([sim.p.getJointInfo(sim.robot,j)[10]
                               for j in sim.bulletCtrlJointsInPinOrder])
        if self.squeeze > 40 or np.any(self.limits <= 0) or not np.all(np.isfinite(self.limits)):
            raise RuntimeError('Invalid torque limits or excessive preload')
        self.hands = []
        for name in (LEFT_HAND, RIGHT_HAND):
            matches = [i for i in range(sim.p.getNumJoints(sim.robot))
                       if sim.p.getJointInfo(sim.robot,i)[1].decode() == name]
            if len(matches) != 1:
                raise RuntimeError('Missing Bullet hand frame: '+name)
            self.hands.append(matches[0])
        self.top = LIFT_START+outward.duration
        self.down = self.top+HOLD_TIME
        self.landed = self.down+self.returning.duration
        self.unload = self.landed+TRANSFER_TIME
        self.duration = self.unload+UNLOAD_TIME+FINAL_HOLD
        self.base0, self.base_rot0 = sim.p.getBasePositionAndOrientation(sim.robot)
        self.cube0, self.cube_rot0 = sim.p.getBasePositionAndOrientation(sim.cubeId)
        self.baseline = self.relative = None
        self.preload_ready = self.landing_ready = 0.
        self.preload_checked = self.landing_checked = False
        self.peak_height = self.peak_error = self.top_evidence_time = 0.
        self.clipped_steps = self.applied_steps = 0
        self.final_samples = []
        self.previous_tau = np.zeros(robot.model.nv)
        self.last_phase = None
        self.next_report = 0.
        print(f'[CONTROL] mass={self.mass:.6f} kg; squeeze={self.squeeze:.3f} N/hand')
        print(f'[CONTROL] planned duration={self.duration:.3f} s; local return to start')

    def update(self, sim, robot, t, apply=True, state=None):
        q, v = sim.getpybulletstate() if state is None else state
        pos, rotation = sim.p.getBasePositionAndOrientation(sim.cubeId)
        pos = np.asarray(pos)
        axis = np.asarray(sim.p.getMatrixFromQuaternion(rotation)).reshape(3,3)[:,1]
        inward, _, _, _ = measure_contact(sim,self.groups,axis)
        def load(a,b):
            return float(sum(c[9] for c in sim.p.getContactPoints(bodyA=a,bodyB=b)))
        box_table = load(sim.cubeId,sim.tableId)
        robot_table = load(sim.robot,sim.tableId)
        obstacle = sum(load(a,sim.obstacleId) for a in (sim.robot,sim.cubeId))
        midpoint = np.mean([sim.p.getLinkState(sim.robot,i,computeForwardKinematics=True)[4]
                            for i in self.hands],axis=0)
        phase, reference, squeeze, upward = cycle_reference(
            t,self.outward,self.returning,self.mass,self.gravity,self.squeeze)
        if PRELOAD_END-.25 <= t < PRELOAD_END:
            self.preload_ready = (self.preload_ready+DT if
                np.all(inward > max(1.,.25*self.squeeze)) and robot_table < .5 else 0.)
        if t >= PRELOAD_END and not self.preload_checked:
            if self.preload_ready < .25-1.5*DT or not np.all(inward > max(1.,.25*self.squeeze)):
                raise RuntimeError('Bilateral preload not established; lift cancelled')
            self.preload_checked = True
            self.baseline, self.relative = pos.copy(), pos-midpoint
        height = 0. if self.baseline is None else float(pos[2]-self.baseline[2])
        slip = 0. if self.relative is None else float(np.linalg.norm(pos-midpoint-self.relative))
        # Keep squeeze until table has carried >=80% of the weight continuously
        # for 0.25 s. No automatic retry or object teleportation on failure.
        if self.unload-.5 <= t < self.unload:
            ready = (box_table >= .8*self.weight and robot_table < .5 and abs(height) < .005)
            self.landing_ready = self.landing_ready+DT if ready else 0.
        if t >= self.unload and not self.landing_checked:
            if (self.landing_ready < .25-1.5*DT or box_table < .8*self.weight
                    or abs(height) >= .005 or robot_table >= .5):
                raise RuntimeError('Table support not established; squeeze unloading cancelled')
            self.landing_checked = True
        tau_raw = self.torque_function(robot,q,v,reference,self.frames,axis,squeeze,upward)
        tau = np.clip(tau_raw,-self.limits,self.limits)
        clipped = bool(np.any(np.abs(tau_raw)>self.limits))
        error = float(np.max(np.abs(pin.difference(robot.model,q,reference[0]))))
        speed = float(np.max(np.abs(v)))
        base, base_rot = sim.p.getBasePositionAndOrientation(sim.robot)
        drift = float(np.linalg.norm(np.asarray(base)-self.base0))
        angle = lambda a,b: float(2*np.arccos(np.clip(abs(np.dot(a,b)),0,1)))
        lateral = float(np.linalg.norm((pos-np.asarray(self.cube0))[:2]))
        finite = np.all(np.isfinite(np.r_[q,v,pos,inward,box_table,robot_table,obstacle,height,slip]))
        invalid = (not finite or error>.15 or speed>5 or drift>.02
                   or angle(base_rot,self.base_rot0)>.05 or lateral>.03
                   or angle(rotation,self.cube_rot0)>.15 or robot_table>5
                   or obstacle>1 or height<-.015 or height>.06 or slip>.015)
        self.writer.writerow([t,phase,squeeze,upward,*inward,box_table,robot_table,obstacle,
            height,slip,error,speed,int(clipped),int(apply and not invalid),*pos,
            *q,*v,*reference[0],*tau,*self.previous_tau])
        if t+1e-9 >= self.next_report or phase != self.last_phase or invalid:
            print(f't={t:.3f}s {phase} lift={height:.5f}m table={box_table:.3f}N '
                  f'grip={inward[0]:.2f}/{inward[1]:.2f}N error={error:.6f}rad clipped={clipped}')
            self.next_report = t+.5
            self.last_phase = phase
        if invalid:
            raise RuntimeError('Control stopped: inspect last CSV row; proposed torque was not applied')
        self.peak_height, self.peak_error = max(self.peak_height,height), max(self.peak_error,error)
        if phase == 'top_hold' and apply and has_lift_evidence(height,box_table,robot_table,inward,slip):
            self.top_evidence_time += DT
        if phase == 'final_hold':
            self.final_samples.append([height,box_table,robot_table,speed,error])
        self.pending_clipped = clipped
        return tau.tolist()

    def record_step(self, torques):
        self.previous_tau = np.asarray(torques).copy()
        self.clipped_steps += int(self.pending_clipped)
        self.applied_steps += 1


def check_setup(robot, sim):
    if not np.isfinite(DT) or DT <= 0:
        raise ValueError('Invalid DT')
    if not hasattr(sim, 'dynamics_sync_marker_masses'):
        raise RuntimeError('Keep the validated dynamics-synchronized setup_pybullet.py')
    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_MOUSE_PICKING, 0)
    joints = sim.bulletCtrlJointsInPinOrder
    names = list(robot.model.names[1:])
    if (robot.model.nq != robot.model.nv or len(joints) != robot.model.nv
            or [sim.p.getJointInfo(sim.robot,j)[1].decode() for j in joints] != names
            or sim.robot != 1):
        raise RuntimeError('Unexpected robot/order/reset body ID')


def check_initial_state(sim, q0):
    q, v = sim.getpybulletstate()
    np.testing.assert_allclose(q, q0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(v, 0, atol=1e-9, rtol=0)


@contextmanager
def cycle_log(entry_file, robot):
    directory = Path(entry_file).resolve().parent/'part_b_logs'
    directory.mkdir(exist_ok=True)
    filename = directory/('control_cycle_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
    print('[CONTROL] CSV:', filename)
    with filename.open('w', newline='') as stream:
        writer = csv.writer(stream)
        names = list(robot.model.names[1:])
        writer.writerow(['time_s','phase','squeeze_reference_N','up_reference_per_hand_N',
            'left_inward_N','right_inward_N','cube_table_support_N','robot_table_load_N',
            'obstacle_load_N','actual_lift_m','relative_slip_m','max_joint_error_rad',
            'max_speed_rad_s','torque_clipped','step_allowed','cube_x_m','cube_y_m','cube_z_m']
            +[prefix+n for prefix in ('q_','v_','q_des_','tau_next_','tau_applied_previous_') for n in names])
        try:
            yield writer, stream
        except BaseException:
            print('[CONTROL] Partial CSV:', filename)
            raise


def print_result(trajs, total_time):
    final = np.asarray(trajs.final_samples)
    placed = (len(final)>0 and np.all(np.abs(final[:,0])<.005)
              and np.all(final[:,1]>=.8*trajs.weight) and np.all(final[:,2]<.5)
              and np.max(final[:,3])<.01)
    lift_verified = trajs.top_evidence_time >= HOLD_TIME-.05
    print(f'[CONTROL] completed=True; simulated_time={total_time:.3f}s')
    print(f'[CONTROL] peak_lift={trajs.peak_height:.6f}m; top_hold_evidence={trajs.top_evidence_time:.3f}s')
    print(f'[CONTROL] peak_joint_error={trajs.peak_error:.6f}rad; clipped_steps={trajs.clipped_steps}/{trajs.applied_steps}')
    print(f'[CONTROL] local_lift_verified={lift_verified}; table_return_verified={placed}')
    print('[CONTROL] Local cycle only. Full target transport remains pending.')
    if not lift_verified or not placed:
        raise RuntimeError('Cycle finished, but measured lift/placement criteria were not met')


def pause_simulation():
    try:
        input('Simulation paused. Press Enter to close PyBullet...')
    except (EOFError, KeyboardInterrupt):
        pass
    # Simulation_base.__del__ closes its connection.
