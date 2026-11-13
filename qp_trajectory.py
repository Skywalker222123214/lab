"""Level-1 constrained least-squares Bezier trajectory parametrisation.

The geometric segments are retained exactly. Their degree-9 Bezier control
points are q_i + c_k*(q_{i+1}-q_i), using the same scalar c_k for the cube.
The four interior c_k are QP variables; the first/last three are fixed to 0/1.
This gives exact waypoint interpolation and C2 joins, with a stop at each knot.

The objective fits the validated timed segment samples and regularises squared
normalised acceleration. Monotone control points, speed and acceleration bounds
are linear constraints. Derivative-control-point bounds certify the limits for
the entire polynomial, rather than only a sampled grid. Durations are allocated
before solving; this does not optimise total time or implement level 2.

QP sign conventions follow ARO Tutorial 6 (6_convex_optimisation.ipynb),
quadprog_solve_qp. The least-squares
expansion is written explicitly here: H=C.T@C, q=-C.T@target for 0.5*||Cx-y||^2.
"""
from math import comb
import numpy as np
import quadprog

DEGREE = 9
FREE = np.arange(3, DEGREE-2)
FIXED = np.r_[np.zeros(DEGREE-2), np.ones(3)]
ACCELERATION_WEIGHT = .0002
BOUND_SUBDIVISIONS = 16


def bernstein(degree, u):
    u = np.asarray(u, dtype=float)
    indices = np.arange(degree+1)
    return (np.array([comb(degree, int(k)) for k in indices])
            * u[..., None]**indices * (1.-u[..., None])**(degree-indices))


def derivative_matrix(order):
    matrix = np.eye(DEGREE+1)
    for k in range(order):
        matrix = (DEGREE-k)*np.diff(matrix, axis=0)
    return matrix


def bernstein_gram(degree):
    return np.array([[comb(degree,i)*comb(degree,j)
                      / ((2*degree+1)*comb(2*degree,i+j))
                      for j in range(degree+1)] for i in range(degree+1)])


def split_polygon(points,u):
    """de Casteljau subdivision, also valid for matrices of linear coefficients."""
    current = np.array(points,dtype=float,copy=True)
    left,right = [current[0]],[current[-1]]
    while len(current)>1:
        current = (1.-u)*current[:-1]+u*current[1:]
        left.append(current[0]);right.append(current[-1])
    return np.array(left),np.array(right[::-1])


def subdivision_operator(degree):
    rows=[]
    for a,b in zip(np.linspace(0,1,BOUND_SUBDIVISIONS+1)[:-1],
                   np.linspace(0,1,BOUND_SUBDIVISIONS+1)[1:]):
        _,right=split_polygon(np.eye(degree+1),a)
        restricted,_=split_polygon(right,(b-a)/(1.-a))
        rows.extend(restricted)
    return np.array(rows)


def elevated_quintic():
    # Exact degree elevation of [0,0,0,1,1,1], the timed fitting reference.
    return np.array([sum(comb(5,j)*comb(DEGREE-5,k-j)/comb(DEGREE,k)
                         for j in range(3,6) if 0 <= k-j <= DEGREE-5)
                     for k in range(DEGREE+1)])


D1, D2 = derivative_matrix(1), derivative_matrix(2)
# Every restricted derivative polynomial lies within its own control polygon.
# These remain linear inequalities, but are much tighter than a single global
# derivative polygon. Thus we avoid unnecessarily slowing a proven motion.
V_BOUND = subdivision_operator(DEGREE-1)@D1
A_BOUND = subdivision_operator(DEGREE-2)@D2
GRID = np.linspace(0.,1.,41)
B = bernstein(DEGREE,GRID)
TARGET = B@elevated_quintic()
M2 = bernstein_gram(DEGREE-2)
# integral (d2/du2 quintic smoothstep)^2 = 120/7.
REGULARISATION = ACCELERATION_WEIGHT/(120./7.)
H_FULL = B.T@B/len(GRID) + REGULARISATION*D2.T@M2@D2
H = H_FULL[np.ix_(FREE,FREE)]
LINEAR = (H_FULL@FIXED - B.T@TARGET/len(GRID))[FREE]


def profile_cost(coefficients):
    residual = B@coefficients-TARGET
    acceleration = D2@coefficients
    return float(.5*np.mean(residual**2)
                 + .5*REGULARISATION*acceleration@M2@acceleration)


def solve_profile(speed_bound, acceleration_bound):
    """Solve the four-variable QP, and check primal/dual optimality residuals."""
    matrices = [D1]
    values = [np.zeros(DEGREE)]
    if np.isfinite(speed_bound):
        matrices.append(-V_BOUND)
        values.append(np.full(len(V_BOUND),-speed_bound))
    if np.isfinite(acceleration_bound):
        matrices.extend([A_BOUND,-A_BOUND])
        values.extend([np.full(len(A_BOUND),-acceleration_bound)]*2)
    # quadprog expects A.T*x >= b. Fixed endpoints are eliminated exactly.
    full_A = np.vstack(matrices)
    A = full_A[:,FREE]
    b = np.concatenate(values)-full_A@FIXED
    keep = np.max(np.abs(A),axis=1)>0
    if np.any(b[~keep]>1e-12):
        raise ValueError('Infeasible fixed Bezier endpoint constraints')
    A,b = A[keep],b[keep]
    result = quadprog.solve_qp(H,-LINEAR,A.T,b)
    x,dual = result[0],result[4]
    coefficients = FIXED.copy()
    coefficients[FREE] = x
    slack = A@x-b
    primal = float(max(0.,-np.min(slack)))
    stationarity = float(np.max(np.abs(H@x+LINEAR-A.T@dual)))
    complementarity = float(np.max(np.abs(dual*slack)))
    if (primal>1e-8 or stationarity>1e-8 or complementarity>1e-8
            or np.min(dual)<-1e-8 or not np.all(np.isfinite(coefficients))):
        raise RuntimeError('QP solution failed independent feasibility/KKT checks')
    return coefficients, dict(primal_residual=primal,stationarity_residual=stationarity,
                              complementarity_residual=complementarity,
                              objective=profile_cost(coefficients))


class QPSegmentTrajectory:
    def __init__(self, points, cube_deltas, minimum_times,
                 joint_velocity=.12, joint_acceleration=.25,
                 cube_velocity=.03, cube_acceleration=.08):
        self.points = np.array(points,dtype=float,copy=True)
        cube_deltas = np.asarray(cube_deltas,dtype=float)
        minimum_times = np.asarray(minimum_times,dtype=float)
        if (self.points.ndim!=2 or len(self.points)<2
                or cube_deltas.shape!=(len(self.points)-1,3)
                or minimum_times.shape!=(len(self.points)-1,)
                or not np.all(np.isfinite(np.r_[self.points.ravel(),cube_deltas.ravel(),minimum_times]))
                or np.any(minimum_times<=0)):
            raise ValueError('Invalid QP trajectory inputs')
        limits = np.array([joint_velocity,joint_acceleration,cube_velocity,cube_acceleration])
        if not np.all(np.isfinite(limits)) or np.min(limits)<=0:
            raise ValueError('QP derivative limits must be positive')
        self.deltas = np.diff(self.points,axis=0)
        joint_steps = np.max(np.abs(self.deltas),axis=1)
        cube_steps = np.linalg.norm(cube_deltas,axis=1)
        witness = elevated_quintic()
        # Retain the validated clock exactly. The QP changes the polynomial
        # coefficients, not the knot times. No time variable or retiming fallback.
        self.segment_times = minimum_times.copy()
        self.knot_times = np.r_[0.,np.cumsum(self.segment_times)]
        self.duration = float(self.knot_times[-1])
        self.coefficients = []
        details = []
        for dq,dp,T in zip(joint_steps,cube_steps,self.segment_times):
            speed_bound = min(joint_velocity*T/dq if dq>0 else np.inf,
                              cube_velocity*T/dp if dp>0 else np.inf)
            acceleration_bound = min(joint_acceleration*T*T/dq if dq>0 else np.inf,
                                     cube_acceleration*T*T/dp if dp>0 else np.inf)
            c,detail = solve_profile(speed_bound,acceleration_bound)
            # Compare to the analytic quintic reference at the same duration.
            if detail['objective']>profile_cost(witness)+1e-10:
                raise RuntimeError('QP objective exceeds the analytic quintic reference')
            self.coefficients.append(c)
            details.append(detail)
        self.coefficients = np.array(self.coefficients)
        self.velocity_coefficients = self.coefficients@D1.T
        self.acceleration_coefficients = self.coefficients@D2.T
        speed = np.max(self.coefficients@V_BOUND.T,axis=1)/self.segment_times
        acceleration = np.max(np.abs(self.coefficients@A_BOUND.T),axis=1)/self.segment_times**2
        self.report = dict(trajectory_method='qp_bezier_level1',bezier_degree=DEGREE,
            qp_segments=len(self.segment_times),qp_variables_per_segment=len(FREE),
            derivative_bound_subdivisions=BOUND_SUBDIVISIONS,
            qp_primal_residual=max(d['primal_residual'] for d in details),
            qp_stationarity_residual=max(d['stationarity_residual'] for d in details),
            qp_complementarity_residual=max(d['complementarity_residual'] for d in details),
            qp_mean_objective=float(np.mean([d['objective'] for d in details])),
            analytic_quintic_reference_objective=profile_cost(witness),
            mean_relative_objective_improvement=float(np.mean([
                (profile_cost(witness)-d['objective'])/profile_cost(witness) for d in details])),
            max_control_point_change_from_quintic=float(np.max(np.abs(self.coefficients-witness))),
            max_joint_velocity_bound_rad_s=float(np.max(speed*joint_steps)),
            max_joint_acceleration_bound_rad_s2=float(np.max(acceleration*joint_steps)),
            max_cube_velocity_bound_m_s=float(np.max(speed*cube_steps)),
            max_cube_acceleration_bound_m_s2=float(np.max(acceleration*cube_steps)),
            objective_acceleration_weight=ACCELERATION_WEIGHT,
            knot_times_preserved=True,
            geometric_path_preserved=True,stops_at_all_waypoints=True,
            collision_constraints_in_QP=False,total_time_optimised=False)
        if (np.min(self.velocity_coefficients)<-1e-8
                or self.report['max_joint_velocity_bound_rad_s']>joint_velocity+1e-8
                or self.report['max_joint_acceleration_bound_rad_s2']>joint_acceleration+1e-8
                or self.report['max_cube_velocity_bound_m_s']>cube_velocity+1e-8
                or self.report['max_cube_acceleration_bound_m_s2']>cube_acceleration+1e-8):
            raise RuntimeError('QP polynomial derivative certificate failed')

    def progress(self, segment, u):
        T = self.segment_times[segment]
        return (float(bernstein(DEGREE,u)@self.coefficients[segment]),
                float(bernstein(DEGREE-1,u)@self.velocity_coefficients[segment])/T,
                float(bernstein(DEGREE-2,u)@self.acceleration_coefficients[segment])/T**2)

    def sample(self,t):
        t = float(t)
        if not np.isfinite(t):
            raise ValueError('Trajectory time must be finite')
        zero = np.zeros(self.points.shape[1])
        if t<=0:
            return self.points[0].copy(),zero.copy(),zero.copy()
        if t>=self.duration:
            return self.points[-1].copy(),zero.copy(),zero.copy()
        i = int(np.searchsorted(self.knot_times,t,side='right')-1)
        h,dh,ddh = self.progress(i,(t-self.knot_times[i])/self.segment_times[i])
        return self.points[i]+h*self.deltas[i],dh*self.deltas[i],ddh*self.deltas[i]
