"""Linear stability-derivative model of the aerodynamic forces and moments.

The coefficients are first-order Taylor series about a reference flight condition, in the form
tabulated by NASA CR-2144 (Heffley & Jewell 1972, Appendix A) and Etkin & Reid (ch. 5):

    CL = CL0 + CL_a (alpha - alpha0) + CL_adot a_hat + CL_q q_hat + CL_M (M - M0) + CL_de de
    CD = CD0 + CD_a (alpha - alpha0) + CD_q q_hat + CD_M (M - M0) + CD_de de
    Cm = Cm0 + Cm_a (alpha - alpha0) + Cm_adot a_hat + Cm_q q_hat + Cm_M (M - M0) + Cm_de de
    CY = CY_b beta + CY_p p_hat + CY_r r_hat + CY_da da + CY_dr dr
    Cl = Cl_b beta + Cl_p p_hat + Cl_r r_hat + Cl_da da + Cl_dr dr      (likewise Cn)

with the rate normalisation q_hat = q cbar/(2V), a_hat = alpha_dot cbar/(2V),
p_hat = p b/(2V), r_hat = r b/(2V). Control deflections are measured from their reference
(trim) setting, so they are zero at the reference condition.

Lift and drag act perpendicular and parallel to the free stream in the plane of symmetry, so
they are rotated into body axes through the *instantaneous* angle of attack. The rolling and
yawing moment derivatives are defined either in body axes or in stability axes, which are fixed
in the aircraft at the reference angle alpha0 (CR-2144 Appendix A, axis systems); stability-axis
moments and rates are rotated through alpha0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """Reference wing area S [m^2], span b [m] and mean aerodynamic chord cbar [m]."""

    S: float
    b: float
    cbar: float


@dataclass(frozen=True)
class StabilityDerivatives:
    """Nondimensional derivatives about a reference condition (per radian unless noted).

    ``alpha0`` [rad] and ``mach0`` define the reference. Longitudinal force derivatives are
    stability-axis lift and drag derivatives; ``lateral_axes`` says whether Cl and Cn
    derivatives are in "stability" or "body" axes. Mach derivatives are per unit Mach number.
    """

    alpha0: float
    mach0: float
    CL0: float
    CD0: float
    CL_a: float
    CD_a: float
    Cm_a: float
    CL_de: float
    Cm_de: float
    CY_b: float
    Cl_b: float
    Cn_b: float
    Cl_p: float
    Cn_p: float
    Cl_r: float
    Cn_r: float
    Cl_da: float
    Cn_da: float
    CY_dr: float
    Cl_dr: float
    Cn_dr: float
    Cm0: float = 0.0
    CL_adot: float = 0.0
    Cm_adot: float = 0.0
    CL_q: float = 0.0
    CD_q: float = 0.0
    Cm_q: float = 0.0
    CL_M: float = 0.0
    CD_M: float = 0.0
    Cm_M: float = 0.0
    CD_de: float = 0.0
    CY_p: float = 0.0
    CY_r: float = 0.0
    CY_da: float = 0.0
    lateral_axes: str = "stability"

    def __post_init__(self):
        if self.lateral_axes not in ("stability", "body"):
            raise ValueError("lateral_axes must be 'stability' or 'body'")

    def with_(self, **changes) -> StabilityDerivatives:
        return replace(self, **changes)


@dataclass(frozen=True)
class AeroCoefficients:
    """Coefficients at one instant: lift, drag, side force, and body-axis moments."""

    CL: float
    CD: float
    CY: float
    Cl: float  # rolling moment, body axes
    Cm: float
    Cn: float  # yawing moment, body axes


def stability_to_body_moments(Cl_s, Cn_s, alpha0):
    """Rotate rolling/yawing moments from stability axes (at alpha0) to body axes."""
    ca, sa = np.cos(alpha0), np.sin(alpha0)
    return Cl_s * ca - Cn_s * sa, Cl_s * sa + Cn_s * ca


def body_to_stability_rates(p, r, alpha0):
    """Roll and yaw rate about the stability axes fixed at alpha0."""
    ca, sa = np.cos(alpha0), np.sin(alpha0)
    return p * ca + r * sa, -p * sa + r * ca


def coefficients(
    d: StabilityDerivatives,
    geom: Geometry,
    V: float,
    alpha: float,
    beta: float,
    p: float,
    q: float,
    r: float,
    alpha_dot: float,
    mach: float,
    de: float,
    da: float,
    dr: float,
) -> AeroCoefficients:
    """Evaluate the derivative model at an instantaneous state (angles in rad, rates in rad/s)."""
    da_ = alpha - d.alpha0
    dm = mach - d.mach0
    c2v, b2v = geom.cbar / (2 * V), geom.b / (2 * V)
    q_hat, a_hat = q * c2v, alpha_dot * c2v
    CL = d.CL0 + d.CL_a * da_ + d.CL_adot * a_hat + d.CL_q * q_hat + d.CL_M * dm + d.CL_de * de
    CD = d.CD0 + d.CD_a * da_ + d.CD_q * q_hat + d.CD_M * dm + d.CD_de * de
    Cm = d.Cm0 + d.Cm_a * da_ + d.Cm_adot * a_hat + d.Cm_q * q_hat + d.Cm_M * dm + d.Cm_de * de
    stab = d.lateral_axes == "stability"
    pl, rl = body_to_stability_rates(p, r, d.alpha0) if stab else (p, r)
    p_hat, r_hat = pl * b2v, rl * b2v
    CY = d.CY_b * beta + d.CY_p * p_hat + d.CY_r * r_hat + d.CY_da * da + d.CY_dr * dr
    Cl = d.Cl_b * beta + d.Cl_p * p_hat + d.Cl_r * r_hat + d.Cl_da * da + d.Cl_dr * dr
    Cn = d.Cn_b * beta + d.Cn_p * p_hat + d.Cn_r * r_hat + d.Cn_da * da + d.Cn_dr * dr
    if stab:
        Cl, Cn = stability_to_body_moments(Cl, Cn, d.alpha0)
    return AeroCoefficients(CL=CL, CD=CD, CY=CY, Cl=Cl, Cm=Cm, Cn=Cn)


def body_forces_moments(c: AeroCoefficients, geom: Geometry, qbar: float, alpha: float):
    """Body-axis aerodynamic force [N] and moment [N m] from the coefficients.

    Lift and drag are rotated through the instantaneous angle of attack:
    X = qS (CL sin a - CD cos a), Z = -qS (CL cos a + CD sin a).
    """
    qS = qbar * geom.S
    ca, sa = np.cos(alpha), np.sin(alpha)
    force = qS * np.array([c.CL * sa - c.CD * ca, c.CY, -(c.CL * ca + c.CD * sa)])
    moment = qS * np.array([geom.b * c.Cl, geom.cbar * c.Cm, geom.b * c.Cn])
    return force, moment


def to_stability_axes(ac):
    """The same aircraft described in stability axes (body axes rotated by alpha0 about y).

    The derivative model is re-referenced to alpha0 = 0 (lift and drag do not change; lateral
    derivatives, if in body axes, are rotated), the inertia tensor is rotated, and the thrust
    line's incidence grows by alpha0. The physics is unchanged — every eigenvalue is the same —
    but W0 = 0, which is the form the classical approximations assume.
    """
    from .dimensional import body_axis_coefficients  # local: dimensional imports this module

    d = ac.aero
    a = d.alpha0
    ca, sa = np.cos(a), np.sin(a)
    C = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])  # rows: stability axes
    I_s = C @ ac.mass.inertia @ C.T
    lat = {}
    if d.lateral_axes == "body":
        # body-axis derivatives -> stability axes: the inverse rotation of Appendix B
        bc = body_axis_coefficients(d.with_(alpha0=-a))
        lat = {k: getattr(bc, k) for k in ("Cl_b", "Cl_p", "Cl_r", "Cl_da", "Cl_dr",
                                           "Cn_b", "Cn_p", "Cn_r", "Cn_da", "Cn_dr")}  # fmt: skip
        Yp, Yr = d.CY_p * ca + d.CY_r * sa, -d.CY_p * sa + d.CY_r * ca
        lat.update(CY_p=Yp, CY_r=Yr)
    aero = d.with_(alpha0=0.0, lateral_axes="stability", **lat)
    mass = replace(ac.mass, Ixx=I_s[0, 0], Iyy=I_s[1, 1], Izz=I_s[2, 2], Ixz=-I_s[0, 2])
    thrust = replace(ac.thrust, incidence=ac.thrust.incidence + a)
    return ac.with_(aero=aero, mass=mass, thrust=thrust)
