"""Moving the centre of gravity: exact transfer of the derivative model to a new reference.

Moving the CG aft by dx = dh cbar along body x (the aerodynamic forces unchanged), two things
change (Etkin & Reid ch. 5; Nelson ch. 2):

* moments about the new CG gain r x F with r = (dx, 0, 0) from the CG to the old reference:
  Cm' = Cm + dh CN and Cn' = Cn + (dx/b) CY (CN the body normal-force coefficient);
* the old reference point moves at v + w x r. A yaw rate gives it sideslip beta + r dx/V. A
  pitch rate changes its angle of attack by -cos(alpha0) q dx/V and, because the reference
  velocity has a normal component W0 = V sin(alpha0) in body axes, its speed by
  -sin(alpha0) q dx: the dynamic pressure and Mach number change at first order too. So the
  q and r derivatives pick up alpha, beta, Mach and reference-coefficient terms; and since
  lift and drag are defined relative to the local velocity, which turns between the two
  points, CL_q and CD_q exchange CD0 and CL0 terms. In stability axes (alpha0 = 0) this
  reduces to the textbook CL_q' = CL_q - 2 dh CL_a, Cm_q' = Cm_q - 2 dh Cm_a + dh CN_q'.

Linearised about the reference, the transfer is exact for the linear derivative model. The
inertias are kept (the mass that moves is assumed small, a labelled simplification), and the
alpha_dot derivatives are transferred but no q_dot derivatives created (the acceleration of
the old reference point, -q_dot dx, is neglected, as in the textbooks).
"""

from __future__ import annotations

import numpy as np

from ..aero_derivatives import StabilityDerivatives, body_axis_coefficients
from ..aircraft import Aircraft


def lateral_in_body_axes(d: StabilityDerivatives) -> StabilityDerivatives:
    """The same derivatives with the rolling/yawing ones expressed in body axes."""
    if d.lateral_axes == "body":
        return d
    bc = body_axis_coefficients(d)
    ca, sa = np.cos(d.alpha0), np.sin(d.alpha0)
    names = ("Cl_b", "Cl_p", "Cl_r", "Cl_da", "Cl_dr", "Cn_b", "Cn_p", "Cn_r", "Cn_da", "Cn_dr")
    return d.with_(
        lateral_axes="body",
        CY_p=d.CY_p * ca - d.CY_r * sa,
        CY_r=d.CY_p * sa + d.CY_r * ca,
        **{k: getattr(bc, k) for k in names},
    )


def move_cg(ac: Aircraft, dh: float) -> Aircraft:
    """The aircraft with its CG moved aft by ``dh`` mean chords (negative: forward)."""
    d = lateral_in_body_axes(ac.aero)
    c = body_axis_coefficients(d)
    dx = dh * ac.geometry.cbar
    k = 2 * dx / ac.geometry.b  # beta_old = beta + k r_hat
    e = dx / ac.geometry.b  # yaw arm of the side force, in spans
    ca, sa, M0 = np.cos(d.alpha0), np.sin(d.alpha0), d.mach0
    # a pitch rate makes the old reference point see, per unit q_hat:
    da = -2 * dh * ca  # angle of attack change
    dv = -2 * dh * sa  # relative speed change: Mach changes by M0 dv, dynamic pressure 2 dv

    def q_term(C0, C_a, C_M):
        return da * C_a + dv * (2 * C0 + M0 * C_M)

    CN_q_old = c.CN_q + q_term(c.CN, c.CN_a, c.CN_M)  # normal force at the old point, per q_hat
    lon = dict(
        Cm0=d.Cm0 + dh * c.CN,
        Cm_a=d.Cm_a + dh * c.CN_a,
        Cm_adot=d.Cm_adot + dh * c.CN_adot,
        Cm_M=d.Cm_M + dh * c.CN_M,
        Cm_de=d.Cm_de + dh * c.CN_de,
        Cm_q=d.Cm_q + q_term(d.Cm0, d.Cm_a, d.Cm_M) + dh * CN_q_old,
        # lift and drag are normal and parallel to the local velocity, which at the old
        # reference is turned by -da per q_hat: L' = L + D da, D' = D - L da
        CL_q=d.CL_q + q_term(d.CL0, d.CL_a, d.CL_M) + da * d.CD0,
        CD_q=d.CD_q + q_term(d.CD0, d.CD_a, d.CD_M) - da * d.CL0,
    )
    lat = dict(
        CY_r=d.CY_r + k * d.CY_b,
        Cl_r=d.Cl_r + k * d.Cl_b,
        Cn_b=d.Cn_b + e * d.CY_b,
        Cn_p=d.Cn_p + e * d.CY_p,
        Cn_r=d.Cn_r + k * d.Cn_b + e * (d.CY_r + k * d.CY_b),
        Cn_da=d.Cn_da + e * d.CY_da,
        Cn_dr=d.Cn_dr + e * d.CY_dr,
    )
    # the thrust line passes dx sin(i_th) further below the new CG
    thrust = ac.thrust.__class__(
        incidence=ac.thrust.incidence,
        offset=ac.thrust.offset + dx * np.sin(ac.thrust.incidence),
    )
    return ac.with_(aero=d.with_(**lon, **lat), thrust=thrust)


def neutral_point_shift(ac: Aircraft) -> float:
    """Stick-fixed neutral point, in chords aft of the current CG: Cm_a + dh CN_a = 0."""
    c = body_axis_coefficients(ac.aero)
    return -ac.aero.Cm_a / c.CN_a
