"""Analytic small-perturbation equations from dimensional derivatives.

The classical decoupled linear models (NASA CR-2144 Appendix C; Etkin & Reid ch. 6), with the
reference state U0 = V cos(alpha0), W0 = V sin(alpha0) in body axes and theta0 = alpha0 + gamma0.

Longitudinal, states (u, w, q, theta), controls (de, thrust), written as E x_dot = F x + G u:

    (1 - Xudot) u' - Xwdot w'        = Xu u + Xw w + (Xq - W0) q - g cos(theta0) theta + ...
    -Zudot u' + (1 - Zwdot) w'       = Zu u + Zw w + (Zq + U0) q - g sin(theta0) theta + ...
    -Mudot u' - Mwdot w' + q'        = Mu u + Mw w + Mq q + ...
    theta' = q

Lateral, states (beta, p, r, phi), controls (da, dr), with primed rolling/yawing derivatives:

    beta' = Yv beta + (Yp + W0)/V p + (Yr - U0)/V r + g cos(theta0)/V phi + Y*_d d
    p'    = L'b beta + L'p p + L'r r + L'_d d
    r'    = N'b beta + N'p p + N'r r + N'_d d
    phi'  = p + tan(theta0) r

CR-2144 neglects Xudot, Zudot, Mudot and Xwdot (its Appendix C note) and Xq; ``neglect_udot``
reproduces that, otherwise the full alpha-dot terms of the force model are kept.
"""

from __future__ import annotations

import numpy as np

from ..aero_derivatives import (
    DimensionalDerivatives,
    body_axis_coefficients,
    coefficients,
    dimensional_derivatives,
)
from ..aircraft import Aircraft
from ..equations_of_motion import air_data
from ..trim_solver import TrimResult
from ..units import G0
from .numerical import LinearModel


def longitudinal_model(
    dd: DimensionalDerivatives,
    theta0: float,
    thrust_derivatives: tuple[float, float, float] = (0.0, 0.0, 0.0),
    g: float = G0,
    neglect_udot: bool = False,
) -> LinearModel:
    """(u, w, q, theta) model; ``thrust_derivatives`` = (X_T, Z_T, M_T) per unit thrust."""
    Xud, Zud, Mud, Xwd = (0.0,) * 4 if neglect_udot else (dd.Xudot, dd.Zudot, dd.Mudot, dd.Xwdot)
    E = np.array(
        [
            [1.0 - Xud, -Xwd, 0.0, 0.0],
            [-Zud, 1.0 - dd.Zwdot, 0.0, 0.0],
            [-Mud, -dd.Mwdot, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    F = np.array(
        [
            [dd.Xu, dd.Xw, dd.Xq - dd.W0, -g * np.cos(theta0)],
            [dd.Zu, dd.Zw, dd.Zq + dd.U0, -g * np.sin(theta0)],
            [dd.Mu, dd.Mw, dd.Mq, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    XT, ZT, MT = thrust_derivatives
    G = np.array([[dd.Xde, XT], [dd.Zde, ZT], [dd.Mde, MT], [0.0, 0.0]])
    Ei = np.linalg.inv(E)
    return LinearModel(
        A=Ei @ F, B=Ei @ G, states=("u", "w", "q", "theta"), controls=("de", "thrust")
    )


def lateral_model(dd: DimensionalDerivatives, theta0: float, g: float = G0) -> LinearModel:
    """(beta, p, r, phi) model with primed derivatives."""
    V = dd.V
    A = np.array(
        [
            [dd.Yv, (dd.Yp + dd.W0) / V, (dd.Yr - dd.U0) / V, g * np.cos(theta0) / V],
            [dd.Lb_p, dd.Lp_p, dd.Lr_p, 0.0],
            [dd.Nb_p, dd.Np_p, dd.Nr_p, 0.0],
            [0.0, 1.0, np.tan(theta0), 0.0],
        ]
    )
    B = np.array(
        [
            [dd.Ystar_da, dd.Ystar_dr],
            [dd.Lda_p, dd.Ldr_p],
            [dd.Nda_p, dd.Ndr_p],
            [0.0, 0.0],
        ]
    )
    return LinearModel(A=A, B=B, states=("beta", "p", "r", "phi"), controls=("da", "dr"))


def rereference(ac: Aircraft, t: TrimResult) -> Aircraft:
    """The same aircraft with its derivative model re-expanded about the trim point ``t``.

    The model is linear in every variable, so moving its reference is exact: CL0, CD0, Cm0
    become the coefficients at the trim (elevator included), alpha0 and mach0 the trim values,
    and the lateral derivatives are converted to body axes so that they no longer depend on
    the old stability axes. Controls of the new model are perturbations from the trim setting.
    Only for straight, wings-level trims (no rates, no sideslip).
    """
    if t.turn_rate != 0.0 or abs(t.beta) > 1e-12 or abs(t.phi) > 1e-12:
        raise ValueError("rereference needs a straight, wings-level trim")
    d = ac.aero
    air = air_data(t.x)
    de = t.controls[0]
    c = coefficients(
        d, ac.geometry, air["V"], air["alpha"], 0.0, 0.0, 0.0, 0.0, 0.0, air["mach"], de, 0, 0
    )
    bc = body_axis_coefficients(d)
    Yp, Yr = d.CY_p, d.CY_r
    if d.lateral_axes == "stability":
        ca, sa = np.cos(d.alpha0), np.sin(d.alpha0)
        Yp, Yr = d.CY_p * ca - d.CY_r * sa, d.CY_p * sa + d.CY_r * ca
    new = d.with_(
        alpha0=air["alpha"],
        mach0=air["mach"],
        CL0=c.CL,
        CD0=c.CD,
        Cm0=c.Cm,
        Cl_b=bc.Cl_b,
        Cl_p=bc.Cl_p,
        Cl_r=bc.Cl_r,
        Cl_da=bc.Cl_da,
        Cl_dr=bc.Cl_dr,
        Cn_b=bc.Cn_b,
        Cn_p=bc.Cn_p,
        Cn_r=bc.Cn_r,
        Cn_da=bc.Cn_da,
        Cn_dr=bc.Cn_dr,
        CY_p=Yp,
        CY_r=Yr,
        lateral_axes="body",
    )
    return ac.with_(aero=new)


def analytic_model(ac: Aircraft, t: TrimResult, g: float = G0) -> tuple[LinearModel, LinearModel]:
    """Longitudinal and lateral small-perturbation models at a straight, wings-level trim,
    from the Appendix A dimensional derivatives of the re-referenced aircraft."""
    re = rereference(ac, t)
    air = air_data(t.x)
    dd = dimensional_derivatives(re.aero, re.geometry, re.mass, air["rho"], air["V"])
    m, Iy = ac.mass.mass, ac.mass.Iyy
    ci, si = np.cos(ac.thrust.incidence), np.sin(ac.thrust.incidence)
    thrust = (ci / m, -si / m, ac.thrust.offset / Iy)
    theta0 = t.theta
    return longitudinal_model(dd, theta0, thrust, g), lateral_model(dd, theta0, g)
