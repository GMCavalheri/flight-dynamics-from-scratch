"""Time histories of the nonlinear model and of its linearization under the same inputs.

Integration is piecewise between input breakpoints with an explicit 8th-order Runge-Kutta
(DOP853), so control jumps are resolved exactly and the tolerance governs accuracy. Output
times are read from the integrator's dense output (7th order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..aircraft import Aircraft
from ..equations_of_motion import state_derivative
from ..linearization import LinearModel
from .inputs import breakpoints, control_increment


@dataclass(frozen=True)
class SimResult:
    t: np.ndarray  # (n,)
    x: np.ndarray  # (n, 12) or (n, n_states) for a linear model, absolute or perturbation
    u: np.ndarray  # (n, 4) controls applied

    def __getitem__(self, i):
        return self.x[:, i]


def _piecewise(f, x0, t_eval, inputs, rtol, atol):
    t_end = float(t_eval[-1])
    pts = breakpoints(inputs, t_end)
    xs, ts = [], []
    x = np.asarray(x0, dtype=float)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        du = control_increment(inputs, 0.5 * (a + b))  # constant on (a, b)
        last = b == t_end
        te = t_eval[(t_eval >= a) & ((t_eval < b) | (last & (t_eval <= b)))]
        sol = solve_ivp(
            lambda _, y, du=du: f(y, du),
            (a, b),
            x,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x = sol.y[:, -1]
        ts.append(te)
        xs.append(sol.sol(te).T if len(te) else np.empty((0, len(x))))
    return np.concatenate(ts), np.vstack(xs)


def simulate(
    ac: Aircraft,
    x0,
    u0,
    inputs=(),
    t_end: float = 60.0,
    dt: float = 0.05,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> SimResult:
    """Nonlinear response from state x0 with trim controls u0 plus ``inputs``."""
    u0 = np.asarray(u0, dtype=float)
    t_eval = np.round(np.arange(0.0, t_end + 0.5 * dt, dt), 12)

    def f(x, du):
        return state_derivative(ac, x, u0 + du)

    t, x = _piecewise(f, x0, t_eval, inputs, rtol, atol)
    u = np.array([u0 + control_increment(inputs, tk) for tk in t])
    return SimResult(t=t, x=x, u=u)


def simulate_linear(
    model: LinearModel,
    inputs=(),
    t_end: float = 60.0,
    dt: float = 0.05,
    dx0=None,
    control_index=(0, 1, 2, 3),
) -> SimResult:
    """Perturbation response of x_dot = A x + B u. ``control_index`` maps the model's
    control columns to the (de, da, dr, thrust) channels of the inputs."""
    A, B = model.A, model.B
    n = A.shape[0]
    x0 = np.zeros(n) if dx0 is None else np.asarray(dx0, dtype=float)
    idx = list(control_index)[: B.shape[1]]
    t_eval = np.round(np.arange(0.0, t_end + 0.5 * dt, dt), 12)

    def f(x, du):
        return A @ x + B @ du[idx]

    t, x = _piecewise(f, x0, t_eval, inputs, 1e-11, 1e-13)
    u = np.array([control_increment(inputs, tk) for tk in t])
    return SimResult(t=t, x=x, u=u)
