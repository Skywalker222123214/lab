"""Numerical checks only. Synthetic data here are NOT valid robot grasp poses."""

import numpy as np
from trajectory import PathTrajectory


def main():
    # A deliberate detour: a trajectory using only the endpoints would miss it.
    path = np.zeros((3, 15))
    path[1, :2] = [0.2, -0.1]
    path[2, :2] = [0.1, 0.15]
    traj = PathTrajectory(path)

    for point, t in zip(path, traj.knot_times):
        q, v, a = traj.sample(t)
        np.testing.assert_allclose(q, point, atol=1e-12)
        np.testing.assert_allclose(v, 0, atol=1e-12)
        np.testing.assert_allclose(a, 0, atol=1e-12)
    print("PASS: every waypoint is reached with zero velocity and acceleration")

    for t, expected in [(-1.0, path[0]), (traj.duration + 1, path[-1])]:
        q, v, a = traj.sample(t)
        np.testing.assert_allclose(q, expected)
        np.testing.assert_allclose(v, 0)
        np.testing.assert_allclose(a, 0)

    for i, dt in enumerate(traj.segment_times):
        for u in [0.2, 0.5, 0.8]:
            t = traj.knot_times[i] + u * dt
            eps = dt * 1e-5
            qm, vm, _ = traj.sample(t - eps)
            qp, vp, _ = traj.sample(t + eps)
            _, v, a = traj.sample(t)
            np.testing.assert_allclose((qp - qm) / (2 * eps), v, atol=1e-8)
            np.testing.assert_allclose((vp - vm) / (2 * eps), a, atol=1e-8)
        # Check the exact extrema of h' and |h''|, not only a time grid.
        for u in [0.5, (3-np.sqrt(3))/6, (3+np.sqrt(3))/6]:
            _, v, a = traj.sample(traj.knot_times[i] + u * dt)
            assert np.all(np.abs(v) <= traj.velocity_limits + 1e-12)
            assert np.all(np.abs(a) <= traj.acceleration_limits + 1e-12)
    print("PASS: derivatives agree with finite differences and respect configured limits")

    # A repeated configuration must not cause division by zero.
    stationary = PathTrajectory(np.zeros((2, 15)))
    for vector in stationary.sample(stationary.duration / 2):
        np.testing.assert_allclose(vector, 0)
    for bad_path in [[], [[0]], [[0], [np.nan]]]:
        try:
            PathTrajectory(bad_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid path accepted")
    print("PASS: endpoint holds, repeated points and invalid input checks")
    print(f"Example duration: {traj.duration:.3f} s")
    print("STEP 1 OK -- numerical trajectory checks only; no robot simulation performed")


if __name__ == "__main__":
    main()
