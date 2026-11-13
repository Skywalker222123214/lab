"""Timing and contact checks for the path returned by computepath.

All returned waypoints are retained. Robot and cube references share the same
quintic progress on each segment. Cube orientation must remain constant in this
version; rotational object dynamics are not included in the support feedforward.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pinocchio as pin

from config import DT, LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from trajectory import PathTrajectory
from tools import setcubeplacement, collision, jointlimitsviolated
from grasp_cycle import hand_link_groups, measure_contact, smooth_profile
from path import obstacle_clearance, OBSTACLE_CLEARANCE_REQUIREMENTS

PRELOAD_END = 4.
MOTION_START = 5.
ARRIVAL_HOLD = 1.
TRANSFER_TIME = 2.
UNLOAD_TIME = 3.
FINAL_SETTLE = 1.
FINAL_HOLD = 1.
INITIAL_SETTLING_END = 1.  # Squeeze starts after this settling window.
MAX_PRELOAD_DISPLACEMENT = .015
CUBE_SPEED_LIMIT = .2


def cube_motion_violations(t, position, speed, start_position, gravity):
    """Bound initial settling separately from subsequent controlled motion."""
    violations = []
    if t < PRELOAD_END:
        displacement = float(np.linalg.norm(np.asarray(position)-start_position))
        if displacement > MAX_PRELOAD_DISPLACEMENT:
            violations.append(f'preload_displacement_m={displacement:.6g}>'
                              f'{MAX_PRELOAD_DISPLACEMENT:.6g}')
    # The supplied cube starts about 4 mm above its settled support height.
    # Limit that motion by the allowed displacement and free-fall energy,
    # including 0.05 m/s contact allowance; do not apply transport speed yet.
    limit = (np.sqrt(2.*gravity*MAX_PRELOAD_DISPLACEMENT)+.05
             if t < INITIAL_SETTLING_END else CUBE_SPEED_LIMIT)
    if speed > limit:
        violations.append(f'cube_speed_m_s={speed:.6g}>{limit:.6g}')
    return violations


def check_pose(pose):
    R, p = np.asarray(pose.rotation), np.asarray(pose.translation)
    if (R.shape != (3,3) or p.shape != (3,) or not np.all(np.isfinite(np.r_[R.ravel(),p]))
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-7, rtol=0)):
        raise ValueError('Invalid cube pose in planned path')


class PlannedTrajectory:
    def __init__(self, path, cube_path, q0, qe, start, goal):
        points = np.asarray(path, dtype=float)
        if (points.ndim != 2 or len(points)<2 or not np.all(np.isfinite(points))
                or len(cube_path) != len(points)):
            raise ValueError('Need at least two finite configurations and matching cube poses')
        if (points.shape[1:] != np.asarray(q0).shape
                or not np.allclose(points[0],q0,atol=1e-9,rtol=0)
                or not np.allclose(points[-1],qe,atol=1e-9,rtol=0)):
            raise ValueError('Path endpoints do not match q0 and qe')
        self.cube_path = [pose.copy() for pose in cube_path]
        for pose in [start,goal]+self.cube_path:
            check_pose(pose)
        for actual, desired in [(self.cube_path[0],start),(self.cube_path[-1],goal)]:
            if (not np.allclose(actual.translation,desired.translation,atol=1e-9,rtol=0)
                    or not np.allclose(actual.rotation,desired.rotation,atol=1e-9,rtol=0)):
                raise ValueError('Cube path endpoints do not match coursework placements')
        if any(not np.allclose(pose.rotation,start.rotation,atol=1e-7,rtol=0)
               for pose in self.cube_path):
            raise ValueError('This controller requires constant cube orientation')
        if any(pose.translation[2] < min(start.translation[2],goal.translation[2])-1e-6
               for pose in self.cube_path):
            raise ValueError('Cube path goes below the start/goal support height')
        self.joints = PathTrajectory(points,max_velocity=.12,max_acceleration=.25,
                                     min_segment_time=.1)
        self.cube_deltas = np.diff([p.translation for p in self.cube_path],axis=0)
        lengths = np.linalg.norm(self.cube_deltas,axis=1)
        # Also bound planned cube speed to 3 cm/s and acceleration to 8 cm/s^2.
        times = np.maximum(self.joints.segment_times,
                           np.maximum((15./8.)*lengths/.03,
                                      np.sqrt((10./np.sqrt(3.))*lengths/.08)))
        self.joints.segment_times = times
        self.joints.knot_times = np.r_[0.,np.cumsum(times)]
        self.joints.duration = float(self.joints.knot_times[-1])
        self.duration = self.joints.duration
        self.points = self.joints.points
        self.start, self.goal = start.copy(), goal.copy()

    def sample(self,t):
        t = float(t)
        reference = self.joints.sample(t)
        if t <= 0:
            return reference,self.cube_path[0].copy(),np.zeros(3),np.zeros(3)
        if t >= self.duration:
            return reference,self.cube_path[-1].copy(),np.zeros(3),np.zeros(3)
        i = int(np.searchsorted(self.joints.knot_times,t,side='right')-1)
        duration = self.joints.segment_times[i]
        u = (t-self.joints.knot_times[i])/duration
        h = u**3*(10.-15.*u+6.*u*u)
        dh = 30.*u*u*(1.-u)**2/duration
        ddh = 60.*u*(1.-u)*(1.-2.*u)/duration**2
        pose = self.cube_path[i].copy()
        pose.translation = pose.translation + h*self.cube_deltas[i]
        return reference,pose,dh*self.cube_deltas[i],ddh*self.cube_deltas[i]


def collision_pair_names(robot):
    """Read the results of the most recent collision(robot, q) call."""
    names = []
    for pair, result in zip(robot.collision_model.collisionPairs,
                            robot.collision_data.collisionResults):
        if result.isCollision():
            objects = robot.collision_model.geometryObjects
            names.append(f'{objects[pair.first].name} <-> {objects[pair.second].name}')
    return names


def limit_joint_names(robot, q):
    violated = ((q < robot.model.lowerPositionLimit)
                | (q > robot.model.upperPositionLimit))
    return [str(robot.model.names[i+1]) for i in np.flatnonzero(violated)]


def diagnose_segment(robot, cube, trajectory, segment):
    """Print extra samples only after rejection; never accept or alter a path."""
    print('[CHECK DETAIL] Rejected segment samples (zero-based indices):', flush=True)
    for fraction in (0., .25, .5, .75, 1.):
        q = ((1.-fraction)*trajectory.points[segment]
             + fraction*trajectory.points[segment+1])
        pose = trajectory.cube_path[segment].copy()
        pose.translation = pose.translation + fraction*trajectory.cube_deltas[segment]
        setcubeplacement(robot, cube, pose)
        limits = limit_joint_names(robot, q)
        colliding = bool(collision(robot, q))
        pairs = collision_pair_names(robot) if colliding else []
        clearance = (None if colliding else
                     {component: result[0] for component, result in obstacle_clearance(robot).items()})
        xyz = np.array2string(pose.translation, precision=9, separator=',')
        print(f'[CHECK DETAIL] fraction={fraction:.6f}; cube_xyz={xyz}; '
              f'limit_joints={limits}; collision={colliding}; pairs={pairs}; '
              f'obstacle_clearance_m={clearance}', flush=True)


def validate_reference(robot,cube,trajectory):
    """Denser independent samples, not a continuous collision certificate."""
    frames = [robot.model.getFrameId(n) for n in (LEFT_HAND,RIGHT_HAND)]
    hooks = [cube.model.getFrameId(n) for n in (LEFT_HOOK,RIGHT_HOOK)]
    if any(i>=robot.model.nframes for i in frames) or any(i>=cube.model.nframes for i in hooks):
        raise ValueError('Missing grasp frame')
    pin.framesForwardKinematics(cube.model,cube.data,cube.q0)
    local_hooks = [cube.data.oMf[i].copy() for i in hooks]
    max_position = max_angle = 0.
    minimum_clearance = {component: np.inf for component in OBSTACLE_CLEARANCE_REQUIREMENTS}
    checked = 0
    try:
        for i, duration in enumerate(trajectory.joints.segment_times):
            intervals = max(4,
                int(np.ceil(np.linalg.norm(trajectory.points[i+1]-trajectory.points[i])/.005)),
                int(np.ceil(np.linalg.norm(trajectory.cube_deltas[i])/.001)))
            for u in np.linspace(0.,1.,intervals+1):
                # h(u) covers [0,1]; check uniform geometric progress more densely
                # than the planner, independently of playback timing.
                q = (1.-u)*trajectory.points[i]+u*trajectory.points[i+1]
                pose = trajectory.cube_path[i].copy()
                pose.translation = pose.translation+u*trajectory.cube_deltas[i]
                setcubeplacement(robot,cube,pose)
                limit_violation = bool(jointlimitsviolated(robot,q))
                in_collision = bool(collision(robot,q)) if not limit_violation else False
                if limit_violation or in_collision:
                    # Capture these results before the diagnostic samples update them.
                    limits = limit_joint_names(robot,q) if limit_violation else []
                    pairs = collision_pair_names(robot) if in_collision else []
                    message = (f'Preflight collision/limit violation: segment={i}, fraction={u:.6f}; '
                               f'limit_joints={limits}; collision={in_collision}; pairs={pairs}')
                    print('[CHECK FAILURE]', message, flush=True)
                    try:
                        diagnose_segment(robot,cube,trajectory,i)
                    except Exception as exc:
                        # Optional reporting must not replace the actual rejection.
                        print(f'[CHECK DETAIL] Additional samples unavailable: {exc}', flush=True)
                    raise RuntimeError(message)
                for component, (clearance, closest_pair) in obstacle_clearance(robot).items():
                    minimum_clearance[component] = min(minimum_clearance[component], clearance)
                    required = OBSTACLE_CLEARANCE_REQUIREMENTS[component]
                    if clearance < required:
                        message = (f'Preflight {component} obstacle clearance: '
                                   f'segment={i}, fraction={u:.6f}; distance={clearance:.9f}m; '
                                   f'required={required:.9f}m; pair={closest_pair}')
                        print('[CHECK FAILURE]', message, flush=True)
                        try:
                            diagnose_segment(robot,cube,trajectory,i)
                        except Exception as exc:
                            print(f'[CHECK DETAIL] Additional samples unavailable: {exc}', flush=True)
                        raise RuntimeError(message)
                pin.framesForwardKinematics(robot.model,robot.data,q)
                for frame, hook in zip(frames,local_hooks):
                    relative = robot.data.oMf[frame].inverse()*(pose*hook)
                    ep = float(np.linalg.norm(relative.translation))
                    er = float(np.linalg.norm(pin.log(relative).vector[3:]))
                    max_position,max_angle = max(max_position,ep),max(max_angle,er)
                    if ep>=1e-3 or er>=1e-3:
                        raise RuntimeError(f'Preflight grasp error: segment={i}, fraction={u:.6f}, '
                                           f'position={ep:.9f}m, angle={er:.9f}rad')
                checked += 1
            if i%50 == 0 or i==len(trajectory.joints.segment_times)-1:
                print(f'[CHECK] {i+1}/{len(trajectory.joints.segment_times)} segments',flush=True)
    finally:
        setcubeplacement(robot,cube,trajectory.start)
    stats = dict(configurations=len(trajectory.points),checked_states=checked,
        motion_duration_s=trajectory.duration,max_grasp_position_error_m=max_position,
        max_grasp_orientation_error_rad=max_angle,
        min_robot_obstacle_clearance_m=minimum_clearance['robot'],
        min_cube_obstacle_clearance_m=minimum_clearance['cube'],
        required_robot_obstacle_clearance_m=OBSTACLE_CLEARANCE_REQUIREMENTS['robot'],
        required_cube_obstacle_clearance_m=OBSTACLE_CLEARANCE_REQUIREMENTS['cube'],
        continuous_collision_guarantee=False)
    print(f'[CHECK] passed: {checked} samples; grasp <= {max_position:.9f}m / {max_angle:.9f}rad')
    for component, clearance in minimum_clearance.items():
        print(f'[CHECK] {component} obstacle clearance >= {clearance:.6f}m '
              f'(required {OBSTACLE_CLEARANCE_REQUIREMENTS[component]:.6f}m)')
    print(f'[CHECK] motion duration={trajectory.duration:.3f}s; all waypoints retained')
    return frames,stats


def save_plan(entry_file,trajectory,stats):
    directory = Path(entry_file).resolve().parent/'part_b_logs'
    directory.mkdir(exist_ok=True)
    filename = directory/('transport_plan_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.npz')
    np.savez_compressed(filename,path=trajectory.points,
        cube_positions=np.array([p.translation for p in trajectory.cube_path]),
        cube_rotations=np.array([p.rotation for p in trajectory.cube_path]))
    filename.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('[PLAN] Saved:',filename)
    return filename


def load_plan(filename):
    # Never unpickle a supplied archive. Geometry/endpoints are rechecked later.
    with np.load(filename,allow_pickle=False) as data:
        points = np.asarray(data['path'],dtype=float).copy()
        positions = np.asarray(data['cube_positions'],dtype=float).copy()
        rotations = np.asarray(data['cube_rotations'],dtype=float).copy()
    if (points.ndim!=2 or positions.shape!=(len(points),3)
            or rotations.shape!=(len(points),3,3)
            or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations))):
        raise ValueError('Malformed saved plan')
    return points,[pin.SE3(R,p) for R,p in zip(rotations,positions)]


TRANSPORT_COLUMNS = ['time_s','phase','squeeze_reference_N',
    'support_x_per_hand_N','support_y_per_hand_N','support_z_per_hand_N',
    'left_inward_N','right_inward_N','cube_table_support_N','robot_table_load_N',
    'obstacle_load_N','cube_tracking_error_m','relative_slip_m','max_joint_error_rad',
    'max_speed_rad_s','cube_speed_m_s','torque_clipped','step_allowed',
    'cube_x_m','cube_y_m','cube_z_m','cube_des_x_m','cube_des_y_m','cube_des_z_m',
    'base_drift_m','cube_rotation_change_rad','grip_loss_time_s']


class TransportCycle:
    def __init__(self,robot,sim,trajectory,frames,writer,torque_function):
        self.trajectory,self.frames,self.writer = trajectory,frames,writer
        self.torque_function = torque_function
        self.groups,wrists = hand_link_groups(sim)
        self.mass = sum(sim.p.getDynamicsInfo(sim.cubeId,i)[0]
                        for i in [-1]+list(range(sim.p.getNumJoints(sim.cubeId))))
        mu = sim.p.getDynamicsInfo(sim.cubeId,-1)[1]
        friction = np.array([mu*sim.p.getDynamicsInfo(sim.robot,i)[1] for i in wrists])
        self.gravity = abs(sim.pybulletConfigs['gravity'])
        if (not np.isfinite(self.mass) or self.mass<=0 or not np.isfinite(self.gravity)
                or self.gravity<=0 or not np.all(np.isfinite(friction)) or np.any(friction<=0)):
            raise ValueError('Invalid physical mass/friction/gravity')
        self.weight = self.mass*self.gravity
        self.squeeze = float(max(5.,1.25*self.weight/(2.*friction.min())))
        self.limits = np.array([sim.p.getJointInfo(sim.robot,j)[10]
                               for j in sim.bulletCtrlJointsInPinOrder])
        if self.squeeze>40 or np.any(self.limits<=0) or not np.all(np.isfinite(self.limits)):
            raise ValueError('Invalid torque limits or excessive preload')
        self.hands = []
        for name in (LEFT_HAND,RIGHT_HAND):
            ids=[j for j in range(sim.p.getNumJoints(sim.robot))
                 if sim.p.getJointInfo(sim.robot,j)[1].decode()==name]
            if len(ids)!=1:
                raise ValueError('Missing Bullet hand frame: '+name)
            self.hands.append(ids[0])
        self.motion_end = MOTION_START+trajectory.duration
        self.transfer = self.motion_end+ARRIVAL_HOLD
        self.unload = self.transfer+TRANSFER_TIME
        self.duration = self.unload+UNLOAD_TIME+FINAL_SETTLE+FINAL_HOLD
        self.base0,self.base_rot0 = sim.p.getBasePositionAndOrientation(sim.robot)
        _,self.cube_rot0 = sim.p.getBasePositionAndOrientation(sim.cubeId)
        self.offset = self.relative = None
        self.preload_ready = self.landing_ready = self.grip_loss = 0.
        self.landing_checked = False
        self.previous_tau = np.zeros(robot.model.nv)
        self.pending_clipped = False
        self.applied_steps = self.clipped_steps = 0
        self.peak_joint_error = self.peak_cube_error = 0.
        self.off_table_time = 0.
        self.final_samples = []
        self.last_phase = None
        self.next_report = 0.
        print(f'[TRANSPORT] mass={self.mass:.6f}kg; squeeze={self.squeeze:.3f}N/hand')
        print(f'[TRANSPORT] duration={self.duration:.3f}s; no automatic return to start')

    def commands(self,t):
        reference,pose,velocity,acceleration = self.trajectory.sample(t-MOTION_START)
        squeeze = self.squeeze*smooth_profile(t-INITIAL_SETTLING_END,3.)[0]
        support = smooth_profile(t-PRELOAD_END,1.)[0]
        if t<PRELOAD_END:
            phase='preload'
        elif t<MOTION_START:
            phase='support_ramp'
        elif t<self.motion_end:
            phase='transport'
        elif t<self.transfer:
            phase='arrival_hold'
        elif t<self.unload:
            phase='load_transfer'
            support = 1.-smooth_profile(t-self.transfer,TRANSFER_TIME)[0]
        else:
            if t < self.unload+UNLOAD_TIME:
                phase = 'unload'
            elif t < self.unload+UNLOAD_TIME+FINAL_SETTLE:
                phase = 'final_settle'
            else:
                phase = 'final_hold'
            support=0.
            squeeze *= 1.-smooth_profile(t-self.unload,UNLOAD_TIME)[0]
        support_force = .5*self.mass*(np.array([0.,0.,self.gravity*support])+acceleration)
        return phase,reference,pose,squeeze,support_force

    def update(self,sim,robot,t,apply=True,state=None):
        q,v = sim.getpybulletstate() if state is None else state
        p=sim.p
        pos,rot = p.getBasePositionAndOrientation(sim.cubeId)
        pos = np.asarray(pos)
        cube_velocity,_ = p.getBaseVelocity(sim.cubeId)
        cube_speed = float(np.linalg.norm(cube_velocity))
        axis=np.asarray(p.getMatrixFromQuaternion(rot)).reshape(3,3)[:,1]
        inward,_,_,_ = measure_contact(sim,self.groups,axis)
        def load(a,b):
            return float(sum(c[9] for c in p.getContactPoints(bodyA=a,bodyB=b)))
        table,robot_table = load(sim.cubeId,sim.tableId),load(sim.robot,sim.tableId)
        obstacle=sum(load(a,sim.obstacleId) for a in (sim.robot,sim.cubeId))
        midpoint=np.mean([p.getLinkState(sim.robot,i,computeForwardKinematics=True)[4]
                          for i in self.hands],axis=0)
        phase,reference,pose,squeeze,support = self.commands(t)
        if PRELOAD_END-.25<=t<PRELOAD_END:
            ready=np.all(inward>max(1.,.25*self.squeeze)) and robot_table<.5
            self.preload_ready=self.preload_ready+DT if ready else 0.
        if t>=PRELOAD_END and self.offset is None:
            if (self.preload_ready<.25-1.5*DT
                    or not np.all(inward>max(1.,.25*self.squeeze)) or robot_table>=.5):
                raise RuntimeError('Bilateral preload not established; transport cancelled')
            # Measured initial table settling is reported, not applied as a state reset.
            self.offset=pos-self.trajectory.start.translation
            self.relative=pos-midpoint
            print('[TRANSPORT] measured initial settling offset (m):',self.offset)
        desired=pose.translation+(np.zeros(3) if self.offset is None else self.offset)
        cube_error=0. if self.offset is None else float(np.linalg.norm(pos-desired))
        slip=0. if self.relative is None else float(np.linalg.norm(pos-midpoint-self.relative))
        if self.unload-.5<=t<self.unload:
            ready=table>=.8*self.weight and robot_table<.5 and cube_error<.005 and cube_speed<.01
            self.landing_ready=self.landing_ready+DT if ready else 0.
        if t>=self.unload and not self.landing_checked:
            if (self.landing_ready<.25-1.5*DT or table<.8*self.weight
                    or robot_table>=.5 or cube_error>=.005 or cube_speed>=.01):
                raise RuntimeError('Target table support not established; squeeze unloading cancelled')
            self.landing_checked=True
        if MOTION_START<=t<self.unload:
            self.grip_loss=self.grip_loss+DT if np.any(inward<1.) else 0.
        # After unloading, the cube may remain still while the wrists relax.
        # Relative slip is a grasp check only while active grip is required.
        tau_raw=self.torque_function(robot,q,v,reference,self.frames,axis,squeeze,support)
        tau=np.clip(tau_raw,-self.limits,self.limits)
        clipped=bool(np.any(np.abs(tau_raw)>self.limits))
        error=float(np.max(np.abs(pin.difference(robot.model,q,reference[0]))))
        speed=float(np.max(np.abs(v)))
        base,base_rot=p.getBasePositionAndOrientation(sim.robot)
        drift=float(np.linalg.norm(np.asarray(base)-self.base0))
        angle=lambda a,b:float(2*np.arccos(np.clip(abs(np.dot(a,b)),0,1)))
        cube_angle=angle(rot,self.cube_rot0)
        finite=np.all(np.isfinite(np.r_[q,v,pos,rot,inward,table,robot_table,obstacle,
                                       cube_error,slip,cube_speed,drift,base_rot,support,tau]))
        violations = cube_motion_violations(t,pos,cube_speed,
                                           self.trajectory.start.translation,self.gravity)
        if not finite:
            violations.append('non_finite_observation_or_torque')
        limits = [('joint_error_rad',error,.15),('joint_speed_rad_s',speed,5.),
                  ('base_drift_m',drift,.02),
                  ('base_rotation_rad',angle(base_rot,self.base_rot0),.05),
                  ('cube_rotation_rad',cube_angle,.15),('robot_table_load_N',robot_table,5.),
                  ('obstacle_load_N',obstacle,1.),('grip_loss_time_s',self.grip_loss,.1)]
        if self.offset is not None:
            limits.append(('cube_tracking_error_m',cube_error,.02))
        if PRELOAD_END<=t<self.unload:
            limits.append(('relative_slip_m',slip,.015))
        violations.extend(f'{name}={value:.6g}>{limit:.6g}'
                          for name,value,limit in limits if value>limit)
        invalid = bool(violations)
        self.writer.writerow([t,phase,squeeze,*support,*inward,table,robot_table,obstacle,
            cube_error,slip,error,speed,cube_speed,int(clipped),int(apply and not invalid),
            *pos,*desired,drift,cube_angle,self.grip_loss,
            *q,*v,*reference[0],*tau,*self.previous_tau])
        if t+1e-9>=self.next_report or phase!=self.last_phase or invalid:
            print(f't={t:.3f}s {phase} cube_error={cube_error:.5f}m table={table:.3f}N '
                  f'grip={inward[0]:.2f}/{inward[1]:.2f}N error={error:.6f}rad clipped={clipped}',flush=True)
            self.next_report=t+.5
            self.last_phase=phase
        if invalid:
            raise RuntimeError('Transport stopped: '+'; '.join(violations)+
                               '. Proposed torque not applied; inspect final CSV row.')
        self.peak_joint_error=max(self.peak_joint_error,error)
        self.peak_cube_error=max(self.peak_cube_error,cube_error)
        if phase=='transport' and apply and table<.5 and np.all(inward>1.):
            self.off_table_time+=DT
        if phase=='final_hold':
            raw_error=float(np.linalg.norm(pos-self.trajectory.goal.translation))
            horizontal=float(np.linalg.norm((pos-self.trajectory.goal.translation)[:2]))
            self.final_samples.append([cube_error,table,robot_table,cube_speed,error,raw_error,horizontal])
        self.pending_clipped=clipped
        return tau.tolist()

    def record_step(self,torques):
        self.previous_tau=np.asarray(torques).copy()
        self.applied_steps+=1
        self.clipped_steps+=int(self.pending_clipped)

    def report(self):
        final=np.asarray(self.final_samples)
        # Experimental evidence criteria, not lecturer grading tolerances.
        placed=(len(final)>0 and self.landing_checked and np.all(final[:,0]<.005)
                and np.all(final[:,1]>=.8*self.weight) and np.all(final[:,2]<.5)
                and np.all(final[:,3]<.01) and np.all(final[:,4]<.01)
                and np.all(final[:,5]<.01) and np.all(final[:,6]<.005))
        print(f'[TRANSPORT] completed=True; simulated_time={self.applied_steps*DT:.3f}s')
        print(f'[TRANSPORT] peak_joint_error={self.peak_joint_error:.6f}rad; '
              f'peak_cube_tracking_error={self.peak_cube_error:.6f}m')
        print(f'[TRANSPORT] clipped_steps={self.clipped_steps}/{self.applied_steps}; '
              f'off_table_bilateral_contact_time={self.off_table_time:.3f}s')
        if len(final):
            print(f'[TRANSPORT] final_raw_goal_position_error={final[-1,5]:.6f}m; '
                  f'final_horizontal_goal_error={final[-1,6]:.6f}m')
        print(f'[TRANSPORT] target_placement_evidence={placed}')
        if not placed:
            raise RuntimeError('Run finished, but target placement evidence criteria were not met')


def transport_log(entry_file,robot):
    """Return a context manager for one transport run."""
    from contextlib import contextmanager
    import csv

    @contextmanager
    def opened():
        directory=Path(entry_file).resolve().parent/'part_b_logs'
        directory.mkdir(exist_ok=True)
        filename=directory/('transport_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
        print('[TRANSPORT] CSV:',filename)
        with filename.open('w',newline='') as stream:
            writer=csv.writer(stream)
            names=list(robot.model.names[1:])
            writer.writerow(TRANSPORT_COLUMNS+[prefix+n for prefix in
                ('q_','v_','q_des_','tau_next_','tau_applied_previous_') for n in names])
            try:
                yield writer,stream
            except BaseException:
                print('[TRANSPORT] Partial CSV:',filename)
                raise
    return opened()
