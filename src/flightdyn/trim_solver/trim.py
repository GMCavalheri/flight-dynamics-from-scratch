"""Trim: the controls and attitude that make every body acceleration zero.

A trim point is a steady flight condition: airspeed V, altitude h, flight-path angle gamma and
turn rate psi_dot are specified; the unknowns are

    z = (alpha, beta, phi, de, da, dr, T / W)

and the equations are the six body-axis accelerations u_dot ... r_dot = 0 plus one constraint
that picks the solution: zero sideslip (coordinated flight, the default) or wings level. The
pitch attitude follows from the flight-path angle, and the body rates from the turn rate
(Stevens, Lewis & Johnson, *Aircraft Control and Simulation*, ch. 3):

    a = cos(alpha) cos(beta),   b = sin(phi) sin(beta) + cos(phi) sin(alpha) cos(beta)
    tan(theta) = (a b + sin(gamma) sqrt(a^2 - sin^2(gamma) + b^2)) / (a^2 - sin^2(gamma))
    (p, q, r) = psi_dot (-sin(theta), sin(phi) cos(theta), cos(phi) cos(theta))

The first relation makes the velocity vector climb at exactly gamma; the second is the body
view of a constant rotation about the vertical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from ..aircraft import Aircraft
from ..equations_of_motion import state_derivative
from ..equations_of_motion.rigid_body import PHI, THETA, ZD, P, R, U, W
from ..rotations import body_velocity
from ..units import G0


@dataclass(frozen=True)
class TrimResult:
    """A trimmed flight condition: the 12-state vector and the control vector."""

    x: np.ndarray
    controls: np.ndarray
    V: float
    h: float
    gamma: float
    turn_rate: float
    alpha: float
    beta: float
    residual: float  # largest |acceleration| left [m/s^2 or rad/s^2]

    @property
    def phi(self) -> float:
        return float(self.x[PHI])

    @property
    def theta(self) -> float:
        return float(self.x[THETA])

    @property
    def de(self) -> float:
        return float(self.controls[0])

    @property
    def thrust(self) -> float:
        return float(self.controls[3])


def pitch_attitude(alpha, beta, phi, gamma) -> float:
    """Pitch angle giving flight-path angle gamma (Stevens, Lewis & Johnson, ch. 3)."""
    a = np.cos(alpha) * np.cos(beta)
    b = np.sin(phi) * np.sin(beta) + np.cos(phi) * np.sin(alpha) * np.cos(beta)
    sg = np.sin(gamma)
    return float(np.arctan2(a * b + sg * np.sqrt(a * a - sg * sg + b * b), a * a - sg * sg))


def trim_state(V, h, gamma, turn_rate, alpha, beta, phi) -> np.ndarray:
    x = np.zeros(12)
    x[U : W + 1] = body_velocity(V, alpha, beta)
    theta = pitch_attitude(alpha, beta, phi, gamma)
    x[PHI], x[THETA] = phi, theta
    x[P : R + 1] = turn_rate * np.array(
        [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]
    )
    x[ZD] = -h
    return x


def trim(
    ac: Aircraft,
    V: float,
    h: float,
    gamma: float = 0.0,
    turn_rate: float = 0.0,
    constraint: str = "zero_sideslip",
    guess: np.ndarray | None = None,
    g: float = G0,
    tol: float = 1e-10,
) -> TrimResult:
    """Trim ``ac`` at airspeed V [m/s], altitude h [m], flight path gamma and turn rate [rad/s].

    ``constraint`` closes the system: "zero_sideslip" (beta = 0, coordinated) or
    "wings_level" (phi = 0; only meaningful without turning).
    """
    if constraint not in ("zero_sideslip", "wings_level"):
        raise ValueError("constraint must be 'zero_sideslip' or 'wings_level'")
    W_ = ac.mass.mass * g

    def unpack(z):
        alpha, beta, phi, de, da, dr, t = z
        return alpha, beta, phi, np.array([de, da, dr, t * W_])

    def residual(z):
        alpha, beta, phi, u = unpack(z)
        x = trim_state(V, h, gamma, turn_rate, alpha, beta, phi)
        xd = state_derivative(ac, x, u, g)
        extra = beta if constraint == "zero_sideslip" else phi
        return np.concatenate([xd[U : R + 1], [extra]])

    z0 = np.array([ac.aero.alpha0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]) if guess is None else guess
    sol = root(residual, z0, method="hybr", options=dict(xtol=1e-13))
    res = float(np.max(np.abs(residual(sol.x)[:6])))
    if not sol.success or res > tol:
        raise RuntimeError(f"trim did not converge (residual {res:.2e}): {sol.message}")
    alpha, beta, phi, u = unpack(sol.x)
    x = trim_state(V, h, gamma, turn_rate, alpha, beta, phi)
    return TrimResult(
        x=x,
        controls=u,
        V=V,
        h=h,
        gamma=gamma,
        turn_rate=turn_rate,
        alpha=float(alpha),
        beta=float(beta),
        residual=res,
    )


def trim_guess(t: TrimResult, W: float) -> np.ndarray:
    """Unknown vector of a previous trim, to start a neighbouring one (continuation)."""
    de, da, dr, T = t.controls
    return np.array([t.alpha, t.beta, t.phi, de, da, dr, T / W])
