"""Reference aircraft built from the cited data in ``reference_data`` (converted to SI).

* :func:`b747` — Boeing 747 at CR-2144 flight condition 1 (landing) or 2 (power approach), the
  two conditions with tabulated nondimensional derivatives.
* :func:`navion` — the Navion of Teper (1969), one sea-level cruise condition.

What is *not* in the sources, and is assumed here (labelled where used):

* 747: CD_M = 0 and CY_p = CY_r = 0 (not tabulated); thrust independent of speed (T_u = 0),
  so the model's X_u omits the propulsive part of the report's X_u*.
* 747: the aerodynamic pitching moment at the reference, Cm0, is not printed. It is set so
  that the reference condition is in pitch equilibrium with the published thrust line,
  Cm0 = -T l_th / (q S cbar), with T from the body-axis axial force balance there
  (:func:`b747_reference_thrust`). This is a deduction, not data: with Cm0 = 0 the report's
  own M_u* and M_w are not reproduced, with this Cm0 they are (validation report).
* Navion: Teper gives every derivative in body-fixed stability axes, so the model's body axes
  *are* those stability axes (alpha0 = 0). Cm_de is not printed in Table X-B; it is derived
  from M_de of Table X-D. The propeller thrust is taken along the x axis through the CG.
"""

from __future__ import annotations

import numpy as np

from flightdyn.aero_derivatives import DimensionalDerivatives, Geometry, StabilityDerivatives
from flightdyn.aircraft import Aircraft, MassProperties, ThrustLine
from flightdyn.units import DEG, FT, LB, LBF, SLUG_FT2

from reference import b747 as b747_data
from reference import b747_nondimensional
from reference import navion as navion_data

B747_GEOMETRY = Geometry(S=5500.0 * FT**2, b=195.68 * FT, cbar=27.31 * FT)  # Table IX-3


def b747_mass(fc: int) -> MassProperties:
    d = b747_data(fc)
    return MassProperties(
        mass=d["W"] * LB,  # weight in lbf = mass in lbm
        Ixx=d["IX"] * SLUG_FT2,
        Iyy=d["IY"] * SLUG_FT2,
        Izz=d["IZ"] * SLUG_FT2,
        Ixz=d["IXZ"] * SLUG_FT2,
    )


def b747_reference_thrust(fc: int, CL: float | None = None, CD: float | None = None) -> float:
    """Thrust [N] that balances the body-axis axial force at the published reference
    (Table IX-3: alpha0, gamma0 = 0, thrust inclined i_th to body x):
    T cos(i_th) + q S (CL sin a0 - CD cos a0) - W sin(a0 + gamma0) = 0."""
    d, c = b747_data(fc), b747_nondimensional(fc)
    CL, CD = (c["CL"] if CL is None else CL), (c["CD"] if CD is None else CD)
    a0, th0, ith = d["ALPHA"] * DEG, (d["ALPHA"] + d["GAMMA"]) * DEG, d["ITH"] * DEG
    qS = d["Q"] * 5500.0  # lbf, Table IX-3
    T_lbf = (d["W"] * np.sin(th0) - qS * (CL * np.sin(a0) - CD * np.cos(a0))) / np.cos(ith)
    return T_lbf * LBF


def b747_reference_cm0(fc: int) -> float:
    """Aerodynamic Cm at the reference that balances the thrust moment, -T l_th / (q S cbar)."""
    d = b747_data(fc)
    T_lbf = b747_reference_thrust(fc) / LBF
    return -T_lbf * d["LTH"] / (d["Q"] * 5500.0 * 27.31)


def b747_derivatives(fc: int, lateral_axes: str = "stability") -> StabilityDerivatives:
    """CR-2144 Table IX-1 (fc 1) / IX-2 (fc 2) derivatives about the Table IX-3 condition."""
    d, c = b747_data(fc), b747_nondimensional(fc)
    lat = {k: c[k] for k in c if k[:2] in ("CY", "Cl", "Cn")}
    lat["CY_da"] = 0.0  # Table IX-8 prints Y*_da = 0
    return StabilityDerivatives(
        alpha0=d["ALPHA"] * DEG,
        mach0=d["M"],
        CL0=c["CL"],
        CD0=c["CD"],
        Cm0=b747_reference_cm0(fc),  # balances the thrust moment (module docstring)
        CL_a=c["CL_a"],
        CD_a=c["CD_a"],
        Cm_a=c["Cm_a"],
        CL_adot=c["CL_adot"],
        Cm_adot=c["Cm_adot"],
        CL_q=c["CL_q"],
        Cm_q=c["Cm_q"],
        CL_M=c["CL_M"],
        Cm_M=c["Cm_M"],
        CL_de=c["CL_de"],
        Cm_de=c["Cm_de"],
        lateral_axes=lateral_axes,
        **lat,
    )


def b747(fc: int = 2, lateral_axes: str = "stability") -> Aircraft:
    d = b747_data(fc)
    return Aircraft(
        name=f"Boeing 747, CR-2144 flight condition {fc}",
        mass=b747_mass(fc),
        geometry=B747_GEOMETRY,
        aero=b747_derivatives(fc, lateral_axes),
        thrust=ThrustLine(incidence=d["ITH"] * DEG, offset=d["LTH"] * FT),
    )


def b747_condition(fc: int) -> dict[str, float]:
    """Reference flight condition in SI: altitude, airspeed, angle of attack, flight path."""
    d = b747_data(fc)
    return dict(h=d["H"] * FT, V=d["VTO"] * FT, alpha=d["ALPHA"] * DEG, gamma=d["GAMMA"] * DEG)


def navion() -> Aircraft:
    n = navion_data()
    qSc_over_Iy = n["qbar"] * n["S"] * n["c"] / n["Iy"]
    return Aircraft(
        name="Navion (Teper 1969)",
        mass=MassProperties(
            mass=n["W"] * LB,
            Ixx=n["Ix"] * SLUG_FT2,
            Iyy=n["Iy"] * SLUG_FT2,
            Izz=n["Iz"] * SLUG_FT2,
            Ixz=n["Ixz"] * SLUG_FT2,
        ),
        geometry=Geometry(S=n["S"] * FT**2, b=n["b"] * FT, cbar=n["c"] * FT),
        aero=StabilityDerivatives(
            alpha0=0.0,
            mach0=n["M"],
            CL0=n["CL"],
            CD0=n["CD"],
            CL_a=n["CL_a"],
            CD_a=n["CD_a"],
            Cm_a=n["Cm_a"],
            CL_adot=n["CL_adot"],
            Cm_adot=n["Cm_adot"],
            Cm_q=n["Cm_q"],
            CL_M=n["CL_M"],
            CD_M=n["CD_M"],
            Cm_M=n["Cm_M"],
            CL_de=n["CL_de"],
            CD_de=n["CD_de"],
            Cm_de=n["Mde"] / qSc_over_Iy,  # derived: Table X-B prints no Cm_de
            CY_b=n["CY_b"],
            CY_da=n["CY_da"],
            CY_dr=n["CY_dr"],
            Cl_b=n["Cl_b"],
            Cl_p=n["Cl_p"],
            Cl_r=n["Cl_r"],
            Cl_da=n["Cl_da"],
            Cl_dr=n["Cl_dr"],
            Cn_b=n["Cn_b"],
            Cn_p=n["Cn_p"],
            Cn_r=n["Cn_r"],
            Cn_da=n["Cn_da"],
            Cn_dr=n["Cn_dr"],
            lateral_axes="stability",
        ),
    )


def navion_condition() -> dict[str, float]:
    n = navion_data()
    return dict(h=n["h"] * FT, V=n["VTO"] * FT, alpha=0.0, gamma=n["gamma0"] * DEG)


_NAN = float("nan")
_UNPRIMED = {k: _NAN for k in ("Lb", "Lp", "Lr", "Lda", "Ldr", "Nb", "Np", "Nr", "Nda", "Ndr")}


def b747_table_derivatives(
    fc: int, implied_terms: bool = True
) -> tuple[DimensionalDerivatives, tuple[float, float, float]]:
    """CR-2144 Tables IX-4 and IX-8 (report units: ft, s, rad, lbf), with the thrust
    derivatives (X, Z, M per lbf). Unprimed lateral derivatives are not printed (NaN); Y_p, Y_r
    and X_wdot are zero.

    ``implied_terms`` adds the three terms the force model implies but the tables do not print,
    each from printed values: Z_udot = -(W0/U0) Z_wdot and M_udot = -(W0/U0) M_wdot, because
    alpha_dot = (U0 w_dot - W0 u_dot) / V^2; and X_q = -tan(alpha0) Z_q / cos(alpha0), because
    CX_q / CN_q = -tan(alpha0) and the table's Z_q carries U0 = V cos(alpha0). With them the
    published phugoid damping is reproduced; without them (Appendix C's stated neglect) it is
    not (validation report)."""
    d = b747_data(fc)
    V, a = d["VTO"], d["ALPHA"] * DEG
    r = np.tan(a) if implied_terms else 0.0
    dd = DimensionalDerivatives(
        U0=V * np.cos(a),
        W0=V * np.sin(a),
        Xu=d["XU*"],
        Xw=d["XW"],
        Xq=-r * d["ZQ"] / np.cos(a),
        Xwdot=0.0,
        Xudot=0.0,
        Xde=d["XDE"],
        Zu=d["ZU*"],
        Zw=d["ZW"],
        Zq=d["ZQ"],
        Zwdot=d["ZWD"],
        Zudot=-r * d["ZWD"],
        Zde=d["ZDE"],
        Mu=d["MU*"],
        Mw=d["MW"],
        Mq=d["MQ"],
        Mwdot=d["MWD"],
        Mudot=-r * d["MWD"],
        Mde=d["MDE"],
        Yv=d["YV"],
        Yp=0.0,
        Yr=0.0,
        Ystar_da=d["Y*DA"],
        Ystar_dr=d["Y*DR"],
        Lb_p=d["LB'"],
        Lp_p=d["LP'"],
        Lr_p=d["LR'"],
        Lda_p=d["L'DA"],
        Ldr_p=d["L'DR"],
        Nb_p=d["NB'"],
        Np_p=d["NP'"],
        Nr_p=d["NR'"],
        Nda_p=d["N'DA"],
        Ndr_p=d["N'DR"],
        **_UNPRIMED,
    )
    return dd, (d["XDTH"], d["ZDTH"], d["MDTH"])


def navion_table_derivatives() -> DimensionalDerivatives:
    """Teper Tables X-D and X-E as printed (ft units, stability axes: W0 = 0). Z_q and Z_wdot
    are not printed (CL_q = CL_adot = 0 in Table X-B) and are zero."""
    n = navion_data()
    return DimensionalDerivatives(
        U0=n["VTO"],
        W0=0.0,
        Xu=n["Xu"],
        Xw=n["Xw"],
        Xq=0.0,
        Xwdot=0.0,
        Xudot=0.0,
        Xde=n["Xde"],
        Zu=n["Zu"],
        Zw=n["Zw"],
        Zq=0.0,
        Zwdot=0.0,
        Zudot=0.0,
        Zde=n["Zde"],
        Mu=n["Mu"],
        Mw=n["Mw"],
        Mq=n["Mq"],
        Mwdot=n["Mwdot"],
        Mudot=0.0,
        Mde=n["Mde"],
        Yv=n["Yv"],
        Yp=0.0,
        Yr=0.0,
        Ystar_da=n["Y*da"],
        Ystar_dr=n["Y*dr"],
        Lb_p=n["L'b"],
        Lp_p=n["L'p"],
        Lr_p=n["L'r"],
        Lda_p=n["L'da"],
        Ldr_p=n["L'dr"],
        Nb_p=n["N'b"],
        Np_p=n["N'p"],
        Nr_p=n["N'r"],
        Nda_p=n["N'da"],
        Ndr_p=n["N'dr"],
        **_UNPRIMED,
    )
