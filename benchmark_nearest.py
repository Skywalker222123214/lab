"""Compare the actual linear and buffered KD-tree functions in a supplied path.py.

Nearest mode uses identical growing synthetic data and needs only NumPy/SciPy.
RRT mode runs the geometric planner and needs the full ARO environment.
Neither mode modifies path.py. All measured results include KD-tree rebuilds.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import scipy
from scipy.spatial import KDTree


def load_nearest_functions(source):
    """Load the real query functions without importing the simulation modules."""
    tree = ast.parse(source.read_text())
    names = {'distance', 'NEAREST_VERTEX', 'NEAREST_VERTEX_KD'}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError('path.py must define distance and both nearest functions.')
    namespace = {'np': np, 'norm': np.linalg.norm, 'KDTree': KDTree}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'),
         namespace)
    return namespace['NEAREST_VERTEX'], namespace['NEAREST_VERTEX_KD']


def run_queries(function, configurations, queries, use_kd):
    nodes = [(None, q, None) for q in configurations]
    result = np.empty(queries.shape[:2], dtype=np.int64)
    G, nearest_data = [], {}
    start = time.perf_counter()
    for i, node in enumerate(nodes):
        G.append(node)
        for j, query in enumerate(queries[i]):
            if use_kd:
                result[i, j] = function(G, query, nearest_data)
            else:
                result[i, j] = function(G, query)
    elapsed = time.perf_counter() - start
    return elapsed, result


def verify_results(configurations, queries, linear, kd):
    # At query row i, only the first i+1 nodes exist.
    upper = np.arange(1, len(configurations) + 1)[:, None]
    if np.any(kd < 0) or np.any(kd >= upper):
        raise AssertionError('KD-tree returned an unavailable node.')
    dl = np.linalg.norm(configurations[linear] - queries, axis=-1)
    dk = np.linalg.norm(configurations[kd] - queries, axis=-1)
    if not np.allclose(dl, dk, rtol=1e-12, atol=1e-12):
        raise AssertionError('Nearest distances differ between the methods.')
    return int(np.count_nonzero(linear != kd)), float(np.max(np.abs(dl - dk)))


def benchmark_nearest(args, source):
    linear, kd = load_nearest_functions(source)
    warm = np.zeros((1, args.dimensions))
    warm_queries = warm[:, None, :]
    run_queries(linear, warm, warm_queries, False)
    run_queries(kd, warm, warm_queries, True)
    rows, summaries = [], []
    for count in args.nodes:
        group = []
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            rng = np.random.default_rng(seed)
            configurations = rng.uniform(-np.pi, np.pi, (count, args.dimensions))
            queries = rng.uniform(-np.pi, np.pi,
                                  (count, args.queries_per_node, args.dimensions))
            results = {}
            # Alternate timing order to reduce systematic first-run effects.
            order = ['linear', 'kd'] if repeat % 2 == 0 else ['kd', 'linear']
            for method in order:
                function = linear if method == 'linear' else kd
                results[method] = run_queries(function, configurations, queries,
                                              method == 'kd')
            ties, maximum_error = verify_results(configurations, queries,
                                                 results['linear'][1], results['kd'][1])
            row = dict(nodes=count, queries=count * args.queries_per_node,
                       seed=seed, linear_seconds=results['linear'][0],
                       kd_seconds=results['kd'][0], differing_indices=ties,
                       max_distance_error=maximum_error)
            rows.append(row)
            group.append(row)
        linear_median = statistics.median(row['linear_seconds'] for row in group)
        kd_median = statistics.median(row['kd_seconds'] for row in group)
        summary = dict(nodes=count, queries=count * args.queries_per_node,
                       linear_median_seconds=linear_median,
                       kd_median_seconds=kd_median,
                       speedup=linear_median / kd_median)
        summaries.append(summary)
        print(f"nodes={count:5d} queries={summary['queries']:5d} "
              f"linear={linear_median:.6f}s KD={kd_median:.6f}s "
              f"speedup={summary['speedup']:.2f}x", flush=True)
    return dict(dataset='synthetic uniform joint configurations; not a robot path',
                dimensions=args.dimensions, queries_per_node=args.queries_per_node,
                rebuild_every='default from source (currently 64)',
                includes='query calls, pending-node scans, initial build and rebuilds',
                rows=rows, summaries=summaries)


def benchmark_rrt(args, source):
    # Import the supplied project only in full-planner mode.
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('aro_path_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from setup_pinocchio import setuppinocchio
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    original_kd, original_rrt = module.NEAREST_VERTEX_KD, module.rrt
    rows = []
    try:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            order = ['linear', 'kd'] if repeat % 2 == 0 else ['kd', 'linear']
            for method in order:
                # Independent scene objects avoid state left by another run.
                np.random.seed(seed)
                setup_output = io.StringIO()
                with contextlib.redirect_stdout(setup_output):
                    robot, _, _, cube = setuppinocchio()
                    q0, start_ok = module.computeqgrasppose(
                        robot, robot.q0.copy(), cube, CUBE_PLACEMENT)
                    qe, goal_ok = module.computeqgrasppose(
                        robot, robot.q0.copy(), cube, CUBE_PLACEMENT_TARGET)
                if not (start_ok and goal_ok):
                    raise RuntimeError('Endpoint IK failed before the timed comparison.')
                stats = dict(nearest_seconds=0.0, nearest_calls=0,
                             nodes=None, found=None)

                def timed_nearest(G, query, state):
                    start = time.perf_counter()
                    if method == 'linear':
                        result = module.NEAREST_VERTEX(G, query)
                    else:
                        result = original_kd(G, query, state)
                    stats['nearest_seconds'] += time.perf_counter() - start
                    stats['nearest_calls'] += 1
                    return result

                def observed_rrt(*positional, **keywords):
                    G, edges, found = original_rrt(*positional, **keywords)
                    stats.update(nodes=len(G), found=bool(found))
                    return G, edges, found

                module.NEAREST_VERTEX_KD, module.rrt = timed_nearest, observed_rrt
                cube_path, output = [], io.StringIO()
                # Give both methods the same random stream at planner entry.
                np.random.seed(seed)
                start = time.perf_counter()
                with contextlib.redirect_stdout(output):
                    path = module.plan_grasp_path(
                        q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET,
                        robot=robot, cube=cube,
                        max_iterations=args.max_iterations,
                        cube_path_out=cube_path)
                elapsed = time.perf_counter() - start
                if len(path) != len(cube_path):
                    raise AssertionError('Robot and cube paths have different lengths.')
                path_length = (float(np.linalg.norm(np.diff(np.asarray(path), axis=0),
                                                   axis=1).sum()) if path else None)
                row = dict(seed=seed, method=method, success=bool(path),
                           planning_seconds=elapsed, configurations=len(path),
                           joint_path_length=path_length, planner_output=output.getvalue(),
                           endpoint_output=setup_output.getvalue(), **stats)
                rows.append(row)
                print(f"seed={seed} method={method:6s} success={bool(path)} "
                      f"total={elapsed:.3f}s nearest={stats['nearest_seconds']:.3f}s "
                      f"calls={stats['nearest_calls']} nodes={stats['nodes']} "
                      f"configurations={len(path)}", flush=True)
    finally:
        module.NEAREST_VERTEX_KD, module.rrt = original_kd, original_rrt
    return dict(dataset='nominal geometric RRT; no viewer or dynamics execution',
                max_iterations=args.max_iterations, endpoint_IK_timed=False, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=Path('path.py'))
    parser.add_argument('--mode', choices=['nearest', 'rrt'], default='nearest')
    parser.add_argument('--nodes', type=int, nargs='+', default=[100, 500, 2000])
    parser.add_argument('--dimensions', type=int, default=15)
    parser.add_argument('--queries-per-node', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.nodes + [args.dimensions, args.queries_per_node,
                         args.repeats, args.max_iterations]) < 1:
        parser.error('Counts must be positive.')
    if args.seed < 0 or args.seed + args.repeats - 1 > 2**32 - 1:
        parser.error('Seeds must remain in [0, 2**32-1].')
    source = args.path.resolve()
    source_text = source.read_text()
    compile(source_text, str(source), 'exec')
    metadata = dict(mode=args.mode, path=str(source),
                    source_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                    python=sys.version, platform=platform.platform(),
                    processor=platform.processor(), numpy=np.__version__,
                    scipy=scipy.__version__, repeats=args.repeats,
                    first_seed=args.seed)
    print(f"Source: {source}\nSHA256: {metadata['source_sha256']}", flush=True)
    data = (benchmark_nearest(args, source) if args.mode == 'nearest'
            else benchmark_rrt(args, source))
    destination = args.output or Path(f'benchmark_{args.mode}_results.json')
    destination.write_text(json.dumps(dict(metadata=metadata, results=data),
                                      indent=2, allow_nan=False) + '\n')
    print(f'Results saved to {destination.resolve()}', flush=True)


if __name__ == '__main__':
    main()
