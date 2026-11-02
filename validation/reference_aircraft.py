"""Reference aircraft built from the cited data in ``reference_data`` (converted to SI).

* :func:`b747` — Boeing 747 at CR-2144 flight condition 1 (landing) or 2 (power approach), the
  two conditions with tabulated nondimensional derivatives.
* :func:`navion` — the Navion of Teper (1969), one sea-level cruise condition.

What is *not* in the sources, and is assumed here (labelled where used):

* 747: CD_M = 0 and CY_p = CY_r = 0 (not tabulated); thrust independent of speed (T_u = 0),
  so the model's X_u omits the propulsive part of the report's X_u*.
* Navion: Teper gives every derivative in body-fixed stability axes, so the model's body axes
  *are* those stability axes (alpha0 = 0). Cm_de is not printed in Table X-B; it is derived
  from M_de of Table X-D. The propeller thrust is taken along the x axis through the CG.
"""

from __future__ import annotations

from flightdyn.aero_derivatives import Geometry, StabilityDerivatives
from flightdyn.aircraft import Aircraft, MassProperties, ThrustLine
from flightdyn.units import DEG, FT, LB, SLUG_FT2

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
        Cm0=0.0,  # trimmed reference
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
