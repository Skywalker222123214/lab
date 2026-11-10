"""CW2 computed-torque tracking core, without cube grasp-force control.

Uses the model's dynamics through Pinocchio RNEA, with acceleration-level PD.
No simulation stepping, state resets, or torque clipping happens here.
"""

import numpy as np
import pinocchio as pin

KP = 300.0
KD = 2.0 * np.sqrt(KP)


def compute_torque(robot, q, v, q_des, v_des, a_des):
    """Return joint torque (N m) in Pinocchio velocity/joint order.

    tau = M(q) [a_des + KP * error + KD * (v_des - v)] + nle(q,v).
    The cube's mass/contact forces are not included as an attached load.
    """
    model = robot.model
    vectors = []
    for name, value, size in [
        ("q", q, model.nq), ("v", v, model.nv),
        ("q_des", q_des, model.nq),
        ("v_des", v_des, model.nv), ("a_des", a_des, model.nv),
    ]:
        array = np.asarray(value, dtype=float)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector of shape ({size},)")
        vectors.append(array)
    q, v, q_des, v_des, a_des = vectors
    error = pin.difference(model, q, q_des)
    acceleration = a_des + KP * error + KD * (v_des - v)
    tau = np.asarray(pin.rnea(model, robot.data, q, v, acceleration)).copy()
    if not np.all(np.isfinite(tau)):
        raise ValueError("Inverse dynamics produced non-finite torque")
    return tau
