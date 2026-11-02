"""Body-axis nondimensional and dimensional stability derivatives.

Implements NASA CR-2144 (Heffley & Jewell 1972) Appendix B (stability-axis to body-axis
transformation of the nondimensional derivatives) and Appendix A, section 5 (dimensional
derivative definitions). Dimensional derivatives are per unit mass or inertia, in SI: ``Xu``
[1/s] is (dX/du)/m, ``Mw`` [1/(m s)] is (dM/dw)/Iyy, ``Lb`` [1/s^2] is (dL/dbeta)/Ixx, and so
on. Primed lateral derivatives absorb the Ixz inertia coupling:

    L' = (L + Ixz N / Ixx) G,   N' = (N + Ixz L / Izz) G,   G = 1 / (1 - Ixz^2 / (Ixx Izz))

Two additions to the printed definitions, both derived from the force model in ``model.py``:
``Xq`` and ``Zq`` (absent from Appendix A) are -rho S cbar V CX_q / (4 m) and likewise for
CN_q, by analogy with the printed ``Mq``; and the alpha-dot derivatives also give the u_dot
terms (``Xudot``, ``Zudot``, ``Mudot``) that the report neglects, because
alpha_dot = (u w_dot - w u_dot) / V^2 depends on both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..aircraft import MassProperties
from .model import Geometry, StabilityDerivatives


@dataclass(frozen=True)
class BodyAxisCoefficients:
    """CR-2144 Appendix B body-axis coefficients at the reference condition.

    CN is the normal force coefficient (positive up), CX the axial force coefficient (positive
    aft). Rate derivatives use the same normalisation as the stability-axis ones.
    """

    CN: float
    CX: float
    CN_a: float
    CN_adot: float
    CN_q: float
    CN_M: float
    CN_de: float
    CX_a: float
    CX_adot: float
    CX_q: float
    CX_M: float
    CX_de: float
    Cl_b: float
    Cl_p: float
    Cl_r: float
    Cl_da: float
    Cl_dr: float
    Cn_b: float
    Cn_p: float
    Cn_r: float
    Cn_da: float
    Cn_dr: float


def body_axis_coefficients(d: StabilityDerivatives) -> BodyAxisCoefficients:
    """Transform the derivatives to body axes (CR-2144 Appendix B, p. B-1)."""
    ca, sa = np.cos(d.alpha0), np.sin(d.alpha0)
    lon = dict(
        CN=d.CL0 * ca + d.CD0 * sa,
        CX=d.CD0 * ca - d.CL0 * sa,
        CN_a=d.CL_a * ca - d.CL0 * sa + d.CD_a * sa + d.CD0 * ca,
        CN_adot=d.CL_adot * ca,
        CN_q=d.CL_q * ca + d.CD_q * sa,
        CN_M=d.CL_M * ca + d.CD_M * sa,
        CN_de=d.CL_de * ca + d.CD_de * sa,
        CX_a=d.CD_a * ca - d.CD0 * sa - d.CL_a * sa - d.CL0 * ca,
        CX_adot=-d.CL_adot * sa,
        CX_q=d.CD_q * ca - d.CL_q * sa,
        CX_M=d.CD_M * ca - d.CL_M * sa,
        CX_de=d.CD_de * ca - d.CL_de * sa,
    )
    if d.lateral_axes == "body":
        lat = dict(
            Cl_b=d.Cl_b,
            Cl_p=d.Cl_p,
            Cl_r=d.Cl_r,
            Cl_da=d.Cl_da,
            Cl_dr=d.Cl_dr,
            Cn_b=d.Cn_b,
            Cn_p=d.Cn_p,
            Cn_r=d.Cn_r,
            Cn_da=d.Cn_da,
            Cn_dr=d.Cn_dr,
        )
    else:
        c2, s2, cs = ca * ca, sa * sa, sa * ca
        lat = dict(
            Cl_b=d.Cl_b * ca - d.Cn_b * sa,
            Cl_p=d.Cl_p * c2 - (d.Cl_r + d.Cn_p) * cs + d.Cn_r * s2,
            Cl_r=d.Cl_r * c2 - (d.Cn_r - d.Cl_p) * cs - d.Cn_p * s2,
            Cl_da=d.Cl_da * ca - d.Cn_da * sa,
            Cl_dr=d.Cl_dr * ca - d.Cn_dr * sa,
            Cn_b=d.Cn_b * ca + d.Cl_b * sa,
            Cn_p=d.Cn_p * c2 - (d.Cn_r - d.Cl_p) * cs - d.Cl_r * s2,
            Cn_r=d.Cn_r * c2 + (d.Cl_r + d.Cn_p) * cs + d.Cl_p * s2,
            Cn_da=d.Cn_da * ca + d.Cl_da * sa,
            Cn_dr=d.Cn_dr * ca + d.Cl_dr * sa,
        )
    return BodyAxisCoefficients(**lon, **lat)


@dataclass(frozen=True)
class DimensionalDerivatives:
    """Body-axis dimensional derivatives (SI) at a reference condition U0, W0."""

    U0: float
    W0: float
    Xu: float
    Xw: float
    Xq: float
    Xwdot: float
    Xudot: float
    Xde: float
    Zu: float
    Zw: float
    Zq: float
    Zwdot: float
    Zudot: float
    Zde: float
    Mu: float
    Mw: float
    Mq: float
    Mwdot: float
    Mudot: float
    Mde: float
    Yv: float
    Yp: float
    Yr: float
    Ystar_da: float
    Ystar_dr: float
    Lb: float
    Lp: float
    Lr: float
    Lda: float
    Ldr: float
    Nb: float
    Np: float
    Nr: float
    Nda: float
    Ndr: float
    Lb_p: float
    Lp_p: float
    Lr_p: float
    Lda_p: float
    Ldr_p: float
    Nb_p: float
    Np_p: float
    Nr_p: float
    Nda_p: float
    Ndr_p: float

    @property
    def V(self) -> float:
        return float(np.hypot(self.U0, self.W0))

    @property
    def Yb(self) -> float:
        """Side-force derivative per unit sideslip [m/s^2] (CR-2144: Y_beta = V Y_v)."""
        return self.V * self.Yv


def dimensional_derivatives(
    d: StabilityDerivatives,
    geom: Geometry,
    mp: MassProperties,
    rho: float,
    V: float,
) -> DimensionalDerivatives:
    """CR-2144 Appendix A body-axis dimensional derivatives at the reference condition."""
    c = body_axis_coefficients(d)
    S, b, cb, m = geom.S, geom.b, geom.cbar, mp.mass
    U0, W0, M = V * np.cos(d.alpha0), V * np.sin(d.alpha0), d.mach0
    Cm = d.Cm0
    k = rho * S * U0 / m
    lon = dict(
        Xu=k * (-M / 2 * c.CX_M - c.CX + W0 / (2 * U0) * c.CX_a),
        Xw=k / 2 * (-c.CX_a - 2 * W0 / U0 * (c.CX + M / 2 * c.CX_M)),
        Xq=-rho * S * cb * V / (4 * m) * c.CX_q,
        Xwdot=-rho * S * cb / (4 * m) * U0 / V * c.CX_adot,
        Xudot=rho * S * cb / (4 * m) * W0 / V * c.CX_adot,
        Xde=-rho * S * V**2 / (2 * m) * c.CX_de,
        Zu=k * (-M / 2 * c.CN_M - c.CN + W0 / (2 * U0) * c.CN_a),
        Zw=k / 2 * (-c.CN_a - 2 * W0 / U0 * (c.CN + M / 2 * c.CN_M)),
        Zq=-rho * S * cb * V / (4 * m) * c.CN_q,
        Zwdot=-rho * S * cb / (4 * m) * U0 / V * c.CN_adot,
        Zudot=rho * S * cb / (4 * m) * W0 / V * c.CN_adot,
        Zde=-rho * S * V**2 / (2 * m) * c.CN_de,
    )
    ky = rho * S * cb * U0 / mp.Iyy
    lon.update(
        Mu=ky * (M / 2 * d.Cm_M + Cm - W0 / (2 * U0) * d.Cm_a),
        Mw=ky / 2 * (d.Cm_a + 2 * W0 / U0 * (Cm + M / 2 * d.Cm_M)),
        Mq=rho * S * cb**2 * V / (4 * mp.Iyy) * d.Cm_q,
        Mwdot=rho * S * cb**2 / (4 * mp.Iyy) * U0 / V * d.Cm_adot,
        Mudot=-rho * S * cb**2 / (4 * mp.Iyy) * W0 / V * d.Cm_adot,
        Mde=rho * S * cb * V**2 / (2 * mp.Iyy) * d.Cm_de,
    )
    ky = rho * S * V / (2 * m)
    qb = rho * S * V**2 * b / 2  # dynamic pressure times S b
    rb = rho * S * V * b**2 / 4
    unprimed = dict(
        Lb=qb / mp.Ixx * c.Cl_b,
        Lp=rb / mp.Ixx * c.Cl_p,
        Lr=rb / mp.Ixx * c.Cl_r,
        Lda=qb / mp.Ixx * c.Cl_da,
        Ldr=qb / mp.Ixx * c.Cl_dr,
        Nb=qb / mp.Izz * c.Cn_b,
        Np=rb / mp.Izz * c.Cn_p,
        Nr=rb / mp.Izz * c.Cn_r,
        Nda=qb / mp.Izz * c.Cn_da,
        Ndr=qb / mp.Izz * c.Cn_dr,
    )
    G = 1.0 / (1.0 - mp.Ixz**2 / (mp.Ixx * mp.Izz))
    primed = {}
    for s in ("b", "p", "r", "da", "dr"):
        L, N = unprimed["L" + s], unprimed["N" + s]
        primed["L" + s + "_p"] = (L + mp.Ixz * N / mp.Ixx) * G
        primed["N" + s + "_p"] = (N + mp.Ixz * L / mp.Izz) * G
    # side-force rate derivatives: CY_p, CY_r are per p_hat, r_hat in the lateral axes
    Yp, Yr = d.CY_p, d.CY_r
    if d.lateral_axes == "stability":
        ca, sa = np.cos(d.alpha0), np.sin(d.alpha0)
        Yp, Yr = d.CY_p * ca - d.CY_r * sa, d.CY_p * sa + d.CY_r * ca
    lat = dict(
        Yv=ky * d.CY_b,
        Yp=rho * S * V * b / (4 * m) * Yp,
        Yr=rho * S * V * b / (4 * m) * Yr,
        Ystar_da=ky * d.CY_da,
        Ystar_dr=ky * d.CY_dr,
    )
    return DimensionalDerivatives(U0=U0, W0=W0, **lon, **lat, **unprimed, **primed)
