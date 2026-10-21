"""LQG = LQR + Kalman filter, for a robot that follows a path.

The state x has two errors of the robot with respect to the path:
    e_y      lateral error: how far the robot is to the left (+) or to the right (-) of the path   [m]
    e_theta  heading error: the yaw of the robot minus the direction of the path                   [rad]
The control u is the extra turn rate:  u = omega - v * kappa
(kappa = curvature of the path; v * kappa is the turn that the path itself needs: the "feedforward").

Linear model for small errors at the speed v:     e_y' = v * e_theta,     e_theta' = u
Discrete model (step dt, u constant in the step):  x[k+1] = A x[k] + B u[k]
    A = [[1, v*dt],          B = [[v*dt*dt/2],
         [0, 1   ]]               [dt       ]]

LQR:     u = -K x.  K makes the sum of  x'Qx + u'Ru  as small as possible
         (Q: how bad an error is, R: how expensive an extra turn is).
         K comes from the discrete algebraic Riccati equation (DARE), solved here by simple iteration.
Kalman:  the measured errors have noise. The filter predicts x with the same model (A, B) and corrects the
         prediction with each measurement z = x + noise. W (noise of the model) and V (noise of the measurement)
         say how much to trust each one.
LQR + Kalman filter = LQG. Separation principle: the two parts are designed one by one.

Plain Python on purpose (2x2 matrices): every step can be read and checked by hand.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Matrix = List[List[float]]


# ------------------------------------------------------------------ small matrix helpers (lists of rows)
def eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def diag(*values: float) -> Matrix:
    return [[float(values[i]) if i == j else 0.0 for j in range(len(values))] for i in range(len(values))]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def mul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def sub(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def inv(a: Matrix) -> Matrix:
    """Inverse of a 1x1 or 2x2 matrix (all this controller needs)."""
    if len(a) == 1:
        return [[1.0 / a[0][0]]]
    (p, q), (r, s) = a
    det = p * s - q * r
    if abs(det) < 1e-15:
        raise ValueError("matrix cannot be inverted")
    return [[s / det, -q / det], [-r / det, p / det]]


def wrap(angle: float) -> float:
    """Angle in [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------- the model
def model(v: float, dt: float) -> Tuple[Matrix, Matrix]:
    """A and B of the error model at the speed v (exact for a constant u during dt)."""
    return [[1.0, v * dt], [0.0, 1.0]], [[0.5 * v * dt * dt], [dt]]


# --------------------------------------------------------------------------------------------- LQR
def dare(A: Matrix, B: Matrix, Q: Matrix, R: Matrix, iterations: int = 100000, tol: float = 1e-10) -> Matrix:
    """Solve  P = Q + A'PA - A'PB (R + B'PB)^-1 B'PA  by iteration. P is the 'cost to go' of a state."""
    P = [row[:] for row in Q]
    At, Bt = transpose(A), transpose(B)
    for _ in range(iterations):
        BtP = mul(Bt, P)
        K = mul(inv(add(R, mul(BtP, B))), mul(BtP, A))          # (R + B'PB)^-1 B'PA
        AtP = mul(At, P)
        P_next = add(Q, sub(mul(AtP, A), mul(mul(AtP, B), K)))
        change = max(abs(P_next[i][j] - P[i][j]) for i in range(len(P)) for j in range(len(P)))
        P = P_next
        if change < tol * (1.0 + max(abs(x) for row in P for x in row)):
            return P
    raise ValueError("the Riccati iteration did not converge (is the speed > 0?)")


def lqr_gain(A: Matrix, B: Matrix, Q: Matrix, R: Matrix) -> Matrix:
    """K = (R + B'PB)^-1 B'PA, with P from the Riccati equation."""
    P = dare(A, B, Q, R)
    BtP = mul(transpose(B), P)
    return mul(inv(add(R, mul(BtP, B))), mul(BtP, A))


class LqrSchedule:
    """The gain K depends on the speed (v is inside A and B). It is computed once for every speed step of
    1 cm/s and kept ("gain scheduling"). Below v_min the model is not controllable (a robot that does not move
    cannot change its lateral error), so v_min is used there."""

    def __init__(self, q_y: float, q_theta: float, r: float, dt: float, v_min: float = 0.05):
        self.Q = diag(q_y, q_theta)
        self.R = [[float(r)]]
        self.dt = dt
        self.v_min = v_min
        self._cache: Dict[float, Tuple[float, float]] = {}

    def gain(self, v: float) -> Tuple[float, float]:
        key = round(max(abs(v), self.v_min), 2)
        if key not in self._cache:
            A, B = model(key, self.dt)
            K = lqr_gain(A, B, self.Q, self.R)
            self._cache[key] = (K[0][0], K[0][1])
        return self._cache[key]


# ---------------------------------------------------------------------------------------------- Kalman
class KalmanFilter:
    """Kalman filter for x = (e_y, e_theta). Both errors are measured: z = x + noise (C = identity)."""

    def __init__(self, w_y: float, w_theta: float, v_y: float, v_theta: float):
        self.W = diag(w_y ** 2, w_theta ** 2)     # noise of the model in one step (slip, delays ...)
        self.V = diag(v_y ** 2, v_theta ** 2)     # noise of the measured pose
        self.x: Matrix = [[0.0], [0.0]]
        self.P: Matrix = [row[:] for row in self.V]
        self.ready = False
        self.last_gain: Matrix = eye(2)

    def reset(self, z: Sequence[float]) -> None:
        """Start from a measurement (it is as uncertain as a measurement)."""
        self.x = [[float(z[0])], [float(z[1])]]
        self.P = [row[:] for row in self.V]
        self.ready = True

    def predict(self, A: Matrix, B: Matrix, u: float) -> None:
        """x = A x + B u,   P = A P A' + W"""
        self.x = add(mul(A, self.x), mul(B, [[u]]))
        self.P = add(mul(mul(A, self.P), transpose(A)), self.W)

    def update(self, z: Sequence[float]) -> None:
        """K = P (P + V)^-1,   x = x + K (z - x),   P = (I - K) P"""
        K = mul(self.P, inv(add(self.P, self.V)))
        innovation = [[z[0] - self.x[0][0]], [wrap(z[1] - self.x[1][0])]]
        self.x = add(self.x, mul(K, innovation))
        self.P = mul(sub(eye(2), K), self.P)
        self.last_gain = K

    def estimate(self) -> Tuple[float, float]:
        return self.x[0][0], self.x[1][0]
