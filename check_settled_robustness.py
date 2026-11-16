"""Fresh measured-start robustness tests; each run uses the actual control.py.

No cached plans, altered physics, controller patches or weakened safety gates.
Default display is Bullet DIRECT. Use --gui for a visible DICE demonstration.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime


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
SUITES = {
    'screen': list(CASES),
    'smoke': ['goal_y_plus_10mm', 'both_shifted'],
    'full': list(CASES),
}
SOURCE_FILES = ['control.py', 'transport.py', 'path.py', 'inverse_geometry.py',
                'qp_trajectory.py', 'setup_pybullet.py', 'config.py']


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def hashes(root):
    names = list(json.loads((root/'expected_runtime_sha256.json').read_text()))
    names += ['check_settled_robustness.py', 'audit_settled_run.py', 'expected_runtime_sha256.json']
    names += sorted(str(p.relative_to(root)) for p in (root/'models').rglob('*') if p.is_file())
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(set(names))}


def verify_runtime(root, environment=False):
    expected = json.loads((root/'expected_runtime_sha256.json').read_text())
    changed = [name for name, digest in expected.items()
               if not (root/name).is_file()
               or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
    if changed:
        raise ValueError('Files differ from the reviewed runtime/test bundle: '+', '.join(changed))
    if environment:
        import importlib
        from importlib.metadata import version
        print('[VERIFY] Python:', sys.version, flush=True)
        for module, package in [('numpy','numpy'), ('pinocchio','pin'), ('scipy','scipy'),
                                ('pybullet','pybullet'), ('quadprog','quadprog')]:
            importlib.import_module(module)
            print('[VERIFY]', package, version(package), flush=True)
        print('[VERIFY]', len(expected), 'runtime/tool/model hashes match the reviewed bundle', flush=True)


def worker(args, root):
    # Override task inputs before any model/control module imports config.
    import numpy as np
    import pinocchio as pin
    import config
    start_offset, goal_offset = CASES[args.case]
    config.CUBE_PLACEMENT = config.CUBE_PLACEMENT.copy()
    config.CUBE_PLACEMENT_TARGET = config.CUBE_PLACEMENT_TARGET.copy()
    config.CUBE_PLACEMENT.translation += np.asarray(start_offset)
    config.CUBE_PLACEMENT_TARGET.translation += np.asarray(goal_offset)
    context = dict(case=args.case, seed=args.seed, plan_only=args.plan_only,
                   nominal_start_xyzquat=pin.SE3ToXYZQUAT(config.CUBE_PLACEMENT).tolist(),
                   original_goal_xyzquat=pin.SE3ToXYZQUAT(config.CUBE_PLACEMENT_TARGET).tolist(),
                   source_sha256=hashes(root), python=sys.version,
                   original_control_main=True, original_wall_clock_pacing=True)
    write_json(args.worker_context, context)
    sys.argv = [str(root/'control.py'), '--trajectory', 'qp', '--seed', str(args.seed),
                '--max-iterations', str(args.max_iterations)]
    if args.plan_only:
        sys.argv.append('--plan-only')
    if not args.gui:
        sys.argv.append('--headless')
    try:
        runpy.run_path(str(root/'control.py'), run_name='__main__')
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def classify(text, error, timeout):
    if timeout:
        return 'wall_clock_timeout'
    if 'audit' in error.lower():
        return 'audit'
    if '[SETTLE] settled=True' not in text:
        return 'startup_or_settling'
    if 'grasp solve failed' in text:
        return 'endpoint_IK'
    if '[CHECK] geometry passed:' not in text:
        return 'planning_or_geometry'
    if '[PLAN] fresh measured-start path:' not in text:
        return 'time_parameterization_or_plan_save'
    if '[INIT] robot explicitly initialized' not in text:
        return 'grasp_initialization'
    if any(x in text for x in ['Target table support not established', '[TRANSFER]']):
        return 'dynamics_or_table_handoff'
    return 'dynamics_or_final_placement'


def verify_report(report, context, root, plan_only):
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(report['seed'] == context['seed'], 'reported RRT seed differs')
    require(report['plan_only'] == plan_only, 'reported execution mode differs')
    for key in ['nominal_start_xyzquat', 'original_goal_xyzquat']:
        require(report[key] == context[key], key + ' differs from requested input')
    require(report['fresh_planning'] and not report['cached_plan_used'], 'not fresh planning')
    require(not report['reference_offset_applied'], 'reference offset was applied')
    require(report['settling']['settled'], 'settling did not complete')
    require(not report['settling']['cube_reset'], 'cube was reset')
    require(report['settling']['robot_cube_contact_steps'] == 0, 'robot contact while settling')
    require(report['preflight']['geometry_checked_before_timing'], 'preflight order')
    require(report['preflight']['preflight_passed'], 'geometric preflight failed')
    for name in SOURCE_FILES:
        require(report['source_sha256'][name] == context['source_sha256'][name], name+' changed')
    require(context['source_sha256'] == hashes(root), 'source/model changed during test')
    require(report['preflight']['actual_start_xyzquat'] == report['actual_start_xyzquat'],
            'preflight did not use measured start')
    require(report['preflight']['original_goal_xyzquat'] == report['original_goal_xyzquat'],
            'preflight goal changed')
    if plan_only:
        require(report['plan_only_passed'] and not report['completed'], 'plan-only outcome')
        require(report['transport_physical_steps'] == 0, 'plan-only executed transport')
        require('grasp_initialization' not in report, 'plan-only initialized grasp')
        return None
    require(report['completed'] and report['target_placement_evidence'], 'transport incomplete')
    from audit_settled_run import audit
    result = audit(context['run_report'])
    require(result.get('reported_peaks_independently_verified') is True,
            'update audit_settled_run.py from the reviewed bundle')
    return result


def run_case(args, root, directory, case, seed, plan_only):
    tag = case + '_seed' + str(seed) + ('_plan' if plan_only else '_physics')
    log, context_path = directory/(tag+'.txt'), directory/(tag+'_context.json')
    result_path = directory/(tag+'.json')
    result = dict(case=case, seed=seed, plan_only=plan_only, passed=False,
                  terminal=str(log), context=str(context_path), physical_completion=False)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
               '--case', case, '--seed', str(seed), '--max-iterations', str(args.max_iterations),
               '--worker-context', str(context_path)]
    if plan_only:
        command.append('--plan-only')
    if args.gui:
        command.append('--gui')
    print('[TEST] START', tag, 'log:', log, flush=True)
    started, timeout = time.perf_counter(), False
    try:
        with log.open('w') as stream:
            with subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT) as process:
                try:
                    result['exit_code'] = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timeout = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    result['exit_code'] = process.returncode
                    result['error'] = 'Wall-clock limit exceeded; incomplete case'
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    result.update(interrupted=True, error='Interrupted by user')
                    raise
        text = log.read_text(errors='replace')
        match = re.search(r'^\[RUN\] report: (.+)$', text, re.MULTILINE)
        if match:
            result['run_report'] = match[1]
        if result['exit_code'] != 0 or timeout:
            errors = re.findall(r'^\[CONTROL STOPPED\] (.+)$', text, re.MULTILINE)
            result.setdefault('error', errors[-1] if errors else 'Worker failed; see terminal log')
            result['failure_stage'] = classify(text, result['error'], timeout)
        else:
            try:
                if not match:
                    raise ValueError('actual control.py did not save a run report')
                context = json.loads(context_path.read_text())
                context['run_report'] = match[1]
                report = json.loads(Path(match[1]).read_text())
                audited = verify_report(report, context, root, plan_only)
                result.update(passed=True, run=report, physical_completion=not plan_only)
                if audited is not None:
                    result['audit'] = audited
            except Exception as exc:
                result.update(error='Audit: '+type(exc).__name__+': '+str(exc), failure_stage='audit')
    except KeyboardInterrupt:
        result.update(passed=False, interrupted=True, error='Interrupted by user',
                      failure_stage='user_interrupted')
        result.setdefault('exit_code',130)
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        write_json(result_path, result)
    print('[TEST]', 'PASS' if result['passed'] else 'FAIL', tag,
          f'{result["wall_seconds"]:.1f}s', result.get('error',''), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--case', choices=CASES)
    select.add_argument('--suite', choices=SUITES)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Check reviewed source/models and imports; no simulation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', help='Repeat selected cases for each seed')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--gui', action='store_true', help='Show Bullet; default is DIRECT')
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=1200., help='Wall-clock seconds per case')
    parser.add_argument('--stop-on-failure', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-context', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list:
        for name, offsets in CASES.items():
            print(name, 'start offset=', offsets[0], 'goal offset=', offsets[1])
        return 0
    import math
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if (args.max_iterations<=0 or not math.isfinite(args.timeout) or args.timeout<=0
            or any(not 0<=seed<=2**32-1 for seed in seeds) or len(set(seeds))!=len(seeds)):
        parser.error('Positive iterations/finite timeout and distinct seeds in [0, 2**32-1] required')
    root = Path(__file__).resolve().parent
    try:
        verify_runtime(root, environment=args.verify)
    except Exception as exc:
        parser.exit(2, '[VERIFY] '+type(exc).__name__+': '+str(exc)+'\n')
    if args.verify:
        return 0
    if args.worker:
        if not args.case or not args.worker_context:
            parser.error('Internal worker arguments missing')
        return worker(args, root)
    directory = root/'settled_robustness_logs'
    directory.mkdir(exist_ok=True)
    lock = (directory/'.running.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another suite is running in this lab; wait before starting another')
    cases = SUITES[args.suite] if args.suite else [args.case or 'both_shifted']
    plan_only = args.plan_only or args.suite=='screen'
    directory = directory/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir()
    suite = dict(passed=False, complete=False, plan_only=plan_only,
                 scope='geometry_and_timing' if plan_only else 'full_physics_and_audit',
                 source_sha256=hashes(root), cases=cases, seeds=seeds, results=[])
    summary_path = directory/'suite_summary.json'
    print('[SUITE] summary:', summary_path, flush=True)
    write_json(summary_path, suite)
    try:
        for seed in seeds:
            for case in cases:
                result = run_case(args, root, directory, case, seed, plan_only)
                suite['results'].append(result)
                write_json(summary_path, suite)
                if result.get('interrupted'):
                    suite['interrupted'] = True
                    return 130
                if not result['passed'] and args.stop_on_failure:
                    return 1
        suite.update(complete=True, passed=all(r['passed'] for r in suite['results']))
    except KeyboardInterrupt:
        suite['interrupted'] = True
        return 130
    finally:
        suite['passed_count'] = sum(r['passed'] for r in suite['results'])
        suite['attempted_count'] = len(suite['results'])
        write_json(summary_path, suite)
        print('[SUITE] complete:', suite['complete'], 'passed:', suite['passed_count'],
              '/', len(cases)*len(seeds), 'summary:', summary_path, flush=True)
        lock.close()
    return 0 if suite['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
