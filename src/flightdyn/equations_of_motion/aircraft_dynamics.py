"""The nonlinear aircraft model: rigid body + standard atmosphere + derivative aerodynamics.

Controls are the vector ``(de, da, dr, thrust)`` in rad, rad, rad, N, with deflections measured
from the reference (trim) setting of the derivative model. Altitude is ``h = -z_d``, taken as
geopotential altitude: with a constant gravity g0 the two are the same thing.

The alpha-dot derivatives make the model implicit: the force depends on alpha_dot, which
depends on the accelerations. Because the force is affine in alpha_dot,

    x_dot = f0 + alpha_dot (f1 - f0),     alpha_dot = (u w_dot - w u_dot) / (u^2 + w^2)

with f0, f1 the derivatives evaluated at alpha_dot = 0 and 1, the fixed point is solved exactly
in one step instead of by iteration or by a one-step-lagged alpha_dot.
"""

from __future__ import annotations

import numpy as np

from ..aero_derivatives import body_forces_moments, coefficients
from ..aircraft import Aircraft
from ..atmosphere import isa
from ..rotations import wind_angles
from ..units import G0
from .rigid_body import THETA, ZD, U, W, rigid_body_derivative

CONTROL_NAMES = ("de", "da", "dr", "thrust")


def air_data(x) -> dict:
    """Airspeed, angles, altitude and atmosphere of a 12-state vector."""
    V, alpha, beta = wind_angles(x[U], x[1], x[W])
    h = -x[ZD]
    atm = isa(h, geopotential=True)
    rho, a = float(atm.rho), float(atm.a)
    return dict(V=V, alpha=alpha, beta=beta, h=h, rho=rho, mach=V / a, qbar=0.5 * rho * V * V)


def forces_and_moments(ac: Aircraft, x, controls, alpha_dot: float = 0.0, air: dict | None = None):
    """Body-axis aerodynamic + thrust force [N] and moment [N m], gravity excluded."""
    air = air_data(x) if air is None else air
    de, da, dr, T = controls
    c = coefficients(
        ac.aero,
        ac.geometry,
        air["V"],
        air["alpha"],
        air["beta"],
        x[3],
        x[4],
        x[5],
        alpha_dot,
        air["mach"],
        de,
        da,
        dr,
    )
    F, M = body_forces_moments(c, ac.geometry, air["qbar"], air["alpha"])
    Ft, Mt = ac.thrust.force_moment(T)
    return F + Ft, M + Mt


def _alpha_dot(x, xdot) -> float:
    u, w = x[U], x[W]
    return (u * xdot[W] - w * xdot[U]) / (u * u + w * w)


def state_derivative(ac: Aircraft, x, controls, g: float = G0) -> np.ndarray:
    """Time derivative of the 12-state vector under the given controls."""
    x = np.asarray(x, dtype=float)
    air = air_data(x)
    F, M = forces_and_moments(ac, x, controls, 0.0, air)
    f0 = rigid_body_derivative(x, F, M, ac.mass, g)
    if ac.aero.CL_adot == 0.0 and ac.aero.Cm_adot == 0.0:
        return f0
    F, M = forces_and_moments(ac, x, controls, 1.0, air)
    df = rigid_body_derivative(x, F, M, ac.mass, g) - f0
    ad = _alpha_dot(x, f0) / (1.0 - _alpha_dot(x, df))
    return f0 + ad * df


def level_state(V: float, alpha: float, h: float, gamma: float = 0.0) -> np.ndarray:
    """12-state vector for wings-level flight at airspeed V, angle of attack alpha, flight-path
    angle gamma and altitude h (heading north, at the origin)."""
    x = np.zeros(12)
    x[U], x[W] = V * np.cos(alpha), V * np.sin(alpha)
    x[THETA] = alpha + gamma
    x[ZD] = -h
    return x
