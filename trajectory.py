"""Baseline CW2 joint trajectory; requires only NumPy.

Input: configurations in the robot's unchanged joint order (radians).
Output: desired configuration, velocity and acceleration at time t (seconds).
This interpolates straight joint-space segments, stopping at every waypoint.
It does not certify collision avoidance or the two-hand grasp constraint.
The default derivative limits are tuning values, not robot specifications.
"""

import numpy as np


class PathTrajectory:
    def __init__(self, path, max_velocity=0.3, max_acceleration=0.6,
                 min_segment_time=0.1):
        self.points = np.array(path, dtype=float, copy=True)
        if (self.points.ndim != 2 or self.points.shape[0] < 2
                or self.points.shape[1] < 1
                or not np.all(np.isfinite(self.points))):
            raise ValueError("path must contain at least two finite configuration vectors")

        dofs = self.points.shape[1]
        self.velocity_limits = self._limits(max_velocity, dofs, "max_velocity")
        self.acceleration_limits = self._limits(max_acceleration, dofs, "max_acceleration")
        if not np.isfinite(min_segment_time) or min_segment_time <= 0:
            raise ValueError("min_segment_time must be finite and positive")

        self.deltas = np.diff(self.points, axis=0)
        # For h(u)=10u^3-15u^4+6u^5 on [0,1]:
        # max h'(u)=15/8; max |h''(u)|=10/sqrt(3).
        velocity_times = np.max(
            (15.0 / 8.0) * np.abs(self.deltas) / self.velocity_limits, axis=1
        )
        acceleration_times = np.max(np.sqrt(
            (10.0 / np.sqrt(3.0)) * np.abs(self.deltas)
            / self.acceleration_limits
        ), axis=1)
        self.segment_times = np.maximum(
            np.maximum(velocity_times, acceleration_times), min_segment_time
        )
        self.knot_times = np.r_[0.0, np.cumsum(self.segment_times)]
        self.duration = float(self.knot_times[-1])

    @staticmethod
    def _limits(value, dofs, name):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(dofs, float(array))
        if array.shape != (dofs,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(name + " must be positive: a scalar or one value per joint")
        return array.copy()

    def sample(self, t):
        """Return fresh (q_des, v_des, a_des) vectors; hold endpoints outside time range."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        zero = np.zeros(self.points.shape[1])
        if t <= 0:
            return self.points[0].copy(), zero.copy(), zero.copy()
        if t >= self.duration:
            return self.points[-1].copy(), zero.copy(), zero.copy()

        i = int(np.searchsorted(self.knot_times, t, side="right") - 1)
        dt = self.segment_times[i]
        u = (t - self.knot_times[i]) / dt
        h = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        dh = 30.0 * u**2 * (1.0 - u)**2
        ddh = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)
        return (
            self.points[i] + h * self.deltas[i],
            (dh / dt) * self.deltas[i],
            (ddh / dt**2) * self.deltas[i],
        )
