"""Numerical linearization of the nonlinear aircraft model about a trim point.

Central differences of ``state_derivative`` give the Jacobians

    A = d f / d x,   B = d f / d u     at (x0, u0),   f(x0, u0) = 0

of the twelve-state model. Perturbations are scaled to each variable (a velocity in m/s, an
angle in rad, thrust in N), and the central formula has an O(h^2) truncation error, so the
Jacobians agree with the analytic small-perturbation matrices to about 1e-8 relative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..aircraft import Aircraft
from ..equations_of_motion import CONTROL_NAMES, STATE_NAMES, state_derivative
from ..equations_of_motion.rigid_body import PHI, THETA, P, Q, R, U, V, W
from ..trim_solver import TrimResult

LONGITUDINAL = (U, W, Q, THETA)  # u, w, q, theta
LATERAL = (V, P, R, PHI)  # v, p, r, phi  (v becomes beta = v / V in the lateral model)


@dataclass(frozen=True)
class LinearModel:
    """x_dot = A x + B u in perturbation variables about (x0, u0)."""

    A: np.ndarray
    B: np.ndarray
    states: tuple[str, ...]
    controls: tuple[str, ...]
    x0: np.ndarray | None = None
    u0: np.ndarray | None = None

    def submodel(self, state_idx, control_idx, states=None, controls=None) -> LinearModel:
        A = self.A[np.ix_(state_idx, state_idx)]
        B = self.B[np.ix_(state_idx, control_idx)]
        return LinearModel(
            A=A,
            B=B,
            states=tuple(states or (self.states[i] for i in state_idx)),
            controls=tuple(controls or (self.controls[j] for j in control_idx)),
        )


def jacobians(f, x0, u0, dx, du):
    """Central-difference Jacobians of f(x, u) with per-variable steps."""
    n, m = len(x0), len(u0)
    A, B = np.zeros((n, n)), np.zeros((n, m))
    for j in range(n):
        e = np.zeros(n)
        e[j] = dx[j]
        A[:, j] = (f(x0 + e, u0) - f(x0 - e, u0)) / (2 * dx[j])
    for j in range(m):
        e = np.zeros(m)
        e[j] = du[j]
        B[:, j] = (f(x0, u0 + e) - f(x0, u0 - e)) / (2 * du[j])
    return A, B


def linearize(ac: Aircraft, t: TrimResult, rel_step: float = 1e-5) -> LinearModel:
    """Twelve-state Jacobians of the nonlinear model at the trim point ``t``."""
    x0, u0 = t.x, t.controls
    Vt = t.V
    dx = np.full(12, rel_step)
    dx[[U, V, W]] = rel_step * Vt  # velocities: a fraction of airspeed
    dx[[9, 10, 11]] = rel_step * max(Vt, 1.0) * 100.0  # positions: only density depends on h
    du = np.array([rel_step, rel_step, rel_step, rel_step * ac.weight])
    A, B = jacobians(lambda x, u: state_derivative(ac, x, u), x0, u0, dx, du)
    return LinearModel(A=A, B=B, states=STATE_NAMES, controls=CONTROL_NAMES, x0=x0, u0=u0)


def longitudinal(model: LinearModel) -> LinearModel:
    """Longitudinal block: states (u, w, q, theta), controls (de, thrust)."""
    return model.submodel(LONGITUDINAL, (0, 3))


def lateral(model: LinearModel, airspeed: float) -> LinearModel:
    """Lateral block in (beta, p, r, phi) with beta = v / V; controls (da, dr)."""
    sub = model.submodel(LATERAL, (1, 2))
    T = np.diag([1.0 / airspeed, 1.0, 1.0, 1.0])  # beta = T x
    Ti = np.linalg.inv(T)
    return LinearModel(
        A=T @ sub.A @ Ti,
        B=T @ sub.B,
        states=("beta", "p", "r", "phi"),
        controls=sub.controls,
    )


def coupling(model: LinearModel) -> float:
    """Largest longitudinal-lateral cross term of A, relative to the largest entry of A
    restricted to the eight dynamic states (zero for symmetric flight)."""
    lon, lat = list(LONGITUDINAL), list(LATERAL)
    A = model.A
    cross = max(np.abs(A[np.ix_(lon, lat)]).max(), np.abs(A[np.ix_(lat, lon)]).max())
    return float(cross / np.abs(A[np.ix_(lon + lat, lon + lat)]).max())
