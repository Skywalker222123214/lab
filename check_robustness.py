"""Run the real control.py with reproducible position changes, then audit CSV.

Configuration overrides happen before importing any model/control modules.
The config.py file, controller, physics timestep and safety limits are untouched.
Run each case in a fresh Python process. No saved path is loaded.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import runpy
import sys
import time
import traceback
import xml.etree.ElementTree as ET


# Offsets in metres, in the world frame. Orientations and heights stay fixed.
# Include the difficult negative-y goal, even if the current IK rejects it.
CASES = {
    'baseline': ((0., 0., 0.), (0., 0., 0.)),
    'start_x_minus_10mm': ((-.01, 0., 0.), (0., 0., 0.)),
    'start_x_plus_10mm': ((.01, 0., 0.), (0., 0., 0.)),
    'start_y_minus_10mm': ((0., -.01, 0.), (0., 0., 0.)),
    'start_y_plus_10mm': ((0., .01, 0.), (0., 0., 0.)),
    'goal_x_minus_10mm': ((0., 0., 0.), (-.01, 0., 0.)),
    'goal_x_plus_10mm': ((0., 0., 0.), (.01, 0., 0.)),
    'goal_y_minus_10mm': ((0., 0., 0.), (0., -.01, 0.)),
    'goal_y_plus_10mm': ((0., 0., 0.), (0., .01, 0.)),
    'both_shifted': ((-.01, -.01, 0.), (-.01, .01, 0.)),
}


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def audit_csv(filename, goal, model_file, mass, gravity, dt):
    """Independently recompute errors, limits and final evidence at every step."""
    limits = {
        joint.attrib['name']: {k: float(v) for k, v in joint.find('limit').attrib.items()}
        for joint in ET.parse(model_file).getroot().findall('joint')
        if joint.find('limit') is not None
    }
    peaks = {name: 0. for name in (
        'obstacle_load_N', 'robot_table_load_N', 'cube_tracking_error_m',
        'relative_slip_m', 'max_joint_error_rad', 'cube_rotation_change_rad')}
    count = applied = clipped = final_count = 0
    previous_time = previous_torque = None
    final_raw = final_horizontal = None
    phases = {}

    def require(condition, message):
        if not condition:
            raise ValueError(f'CSV row {count}: {message}')

    with Path(filename).open(newline='') as stream:
        reader = csv.DictReader(stream)
        joints = [key[2:] for key in reader.fieldnames
                  if key.startswith('q_') and not key.startswith('q_des_')]
        require(len(joints) == 15, 'unexpected joint columns')
        for row in reader:
            count += 1
            value = {k: float(v) for k, v in row.items() if k != 'phase'}
            require(all(math.isfinite(v) for v in value.values()), 'non-finite data')
            t = value['time_s']
            require(abs(t if previous_time is None else t-previous_time-dt) < 1e-10,
                    'incorrect timestamp or missing step')
            previous_time = t
            phases[row['phase']] = phases.get(row['phase'], 0) + 1
            applied += int(value['step_allowed'])
            clipped += int(value['torque_clipped'])
            error = max(abs(value['q_'+j]-value['q_des_'+j]) for j in joints)
            speed = max(abs(value['v_'+j]) for j in joints)
            require(abs(error-value['max_joint_error_rad']) < 1e-12,
                    'joint error does not match raw states')
            require(abs(speed-value['max_speed_rad_s']) < 1e-12,
                    'joint speed does not match raw states')
            old_torque = [value['tau_applied_previous_'+j] for j in joints]
            require(old_torque == (previous_torque if previous_torque is not None
                                   else [0.]*len(joints)), 'torque history mismatch')
            previous_torque = [value['tau_next_'+j] for j in joints]
            for joint in joints:
                bound = limits[joint]
                require(bound['lower'] <= value['q_'+joint] <= bound['upper'],
                        joint+' position limit')
                require(abs(value['v_'+joint]) <= bound['velocity'], joint+' speed limit')
                for prefix in ('tau_next_', 'tau_applied_previous_'):
                    require(abs(value[prefix+joint]) <= bound['effort'], joint+' torque limit')
            for key in peaks:
                peaks[key] = max(peaks[key], value[key])
            position = [value['cube_'+axis+'_m'] for axis in 'xyz']
            raw = math.sqrt(sum((p-g)**2 for p, g in zip(position, goal)))
            horizontal = math.hypot(position[0]-goal[0], position[1]-goal[1])
            tracking = math.sqrt(sum((value['cube_'+a+'_m']-value['cube_des_'+a+'_m'])**2
                                     for a in 'xyz'))
            if t >= 4.:
                require(abs(tracking-value['cube_tracking_error_m']) < 1e-12,
                        'cube error does not match raw states')
            if row['phase'] == 'final_hold':
                final_count += 1
                require(tracking < .005 and raw < .01 and horizontal < .005,
                        'final position criteria')
                require(value['cube_table_support_N'] >= .8*mass*gravity,
                        'final table support')
                require(value['robot_table_load_N'] < .5
                        and value['cube_speed_m_s'] < .01 and error < .01,
                        'final stability criteria')
                require(all(value[k] == 0. for k in (
                    'squeeze_reference_N', 'support_x_per_hand_N',
                    'support_y_per_hand_N', 'support_z_per_hand_N')), 'final force commands')
                final_raw, final_horizontal = raw, horizontal
            if value['step_allowed'] == 0:
                require(row['phase'] == 'final_hold', 'unexpected rejected physical step')
    require(count == applied+1, 'missing final observation or rejected step')
    require(clipped == 0, 'torque clipping occurred')
    require(final_count == round(1./dt)+1, 'incomplete final hold')
    require(peaks['obstacle_load_N'] == peaks['robot_table_load_N'] == 0.,
            'nonzero obstacle or robot/table contact load')
    return dict(independent_audit_passed=True, rows=count, applied_steps=applied,
                clipped_steps=clipped, final_hold_samples=final_count,
                phases=phases, peaks=peaks, final_raw_goal_error_m=final_raw,
                final_horizontal_goal_error_m=final_horizontal,
                all_numeric_values_finite=True, all_joint_limits_checked=True,
                errors_and_torque_history_recomputed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List cases without loading models')
    parser.add_argument('--case', choices=CASES, default='baseline')
    parser.add_argument('--seed', type=int, default=0, help='NumPy RRT seed, default 0')
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--headless', action='store_true', help='Bullet DIRECT; physics unchanged')
    args = parser.parse_args()
    if args.list:
        for name, offsets in CASES.items():
            print(name, 'start_offset_m=', offsets[0], 'goal_offset_m=', offsets[1])
        return 0
    if not 0 <= args.seed <= 2**32-1 or args.max_iterations <= 0:
        parser.error('seed must be in [0, 2**32-1] and iterations must be positive')

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import numpy as np
    import config
    start_offset, goal_offset = CASES[args.case]
    config.CUBE_PLACEMENT = config.CUBE_PLACEMENT.copy()
    config.CUBE_PLACEMENT.translation = config.CUBE_PLACEMENT.translation + np.array(start_offset)
    config.CUBE_PLACEMENT_TARGET = config.CUBE_PLACEMENT_TARGET.copy()
    config.CUBE_PLACEMENT_TARGET.translation = (config.CUBE_PLACEMENT_TARGET.translation
                                              + np.array(goal_offset))
    if args.headless:
        config.USE_PYBULLET_GUI = False
    np.random.seed(args.seed)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.case+'_seed'+str(args.seed)
    directory = root/'robustness_logs'
    directory.mkdir(exist_ok=True)
    log_path, summary_path = directory/(run_id+'.txt'), directory/(run_id+'.json')
    result = dict(case=args.case, seed=args.seed, plan_only=args.plan_only,
                  headless=args.headless, max_iterations=args.max_iterations,
                  start_m=config.CUBE_PLACEMENT.translation.tolist(),
                  goal_m=config.CUBE_PLACEMENT_TARGET.translation.tolist(),
                  fresh_planning=True, cached_plan_used=False,
                  original_control_main=True, original_wall_clock_timing=True,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.glob('*.py'))},
                  terminal_log=str(log_path), passed=False)
    print('[ROBUSTNESS] case:', args.case, 'seed:', args.seed, flush=True)
    print('[ROBUSTNESS] start (m):', result['start_m'], 'goal (m):', result['goal_m'], flush=True)
    print('[ROBUSTNESS] terminal:', log_path, flush=True)
    argv_before, stdin_before = sys.argv, sys.stdin
    started = time.perf_counter()
    try:
        sys.argv = [str(root/'control.py'), '--trajectory', 'qp',
                    '--max-iterations', str(args.max_iterations)]
        if args.plan_only:
            sys.argv.append('--plan-only')
        if args.headless:
            sys.stdin = io.StringIO('\n')
        with log_path.open('w') as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                try:
                    namespace = runpy.run_path(str(root/'control.py'), run_name='__main__')
                    result.update(main_returned_normally=True,
                                  configurations=len(namespace['path']),
                                  preflight=namespace['stats'],
                                  qp=namespace['trajectory'].joints.report)
                    if not args.plan_only:
                        cycle = namespace['trajs']
                        result.update(simulated_time_s=cycle.applied_steps*config.DT,
                                      clipped_steps=cycle.clipped_steps,
                                      peak_cube_tracking_error_m=cycle.peak_cube_error,
                                      peak_joint_error_rad=cycle.peak_joint_error)
                    result['passed'] = True
                except (Exception, KeyboardInterrupt) as exc:
                    result['error'] = type(exc).__name__+': '+str(exc)
                    traceback.print_exc()
                finally:
                    stream.flush()
        console = log_path.read_text()
        match = re.search(r'^\[TRANSPORT\] CSV: (.+)$', console, re.MULTILINE)
        if match:
            result['csv'] = match[1]
        if result['passed'] and not args.plan_only:
            try:
                result['csv_audit'] = audit_csv(result['csv'], result['goal_m'],
                    config.NEXTAGE_URDF, cycle.mass, cycle.gravity, config.DT)
            except Exception as exc:
                result.update(passed=False, csv_audit_error=type(exc).__name__+': '+str(exc))
        result['wall_seconds'] = time.perf_counter()-started
    finally:
        sys.argv, sys.stdin = argv_before, stdin_before
        summary_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('[ROBUSTNESS] summary:', summary_path, flush=True)
    print('[ROBUSTNESS] passed:', result['passed'], flush=True)
    if 'csv_audit_error' in result:
        print('[ROBUSTNESS] CSV audit:', result['csv_audit_error'], flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
