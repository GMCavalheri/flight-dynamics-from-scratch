"""International Standard Atmosphere, built from its published definition.

Reused unchanged from the sibling project propulsion-from-scratch (AERO-03), where it is
verified against the US Standard Atmosphere 1976 tables (tests/test_atmosphere.py here).

Definition: ICAO Doc 7488/3, *Manual of the ICAO Standard Atmosphere* (1993), identical to the
US Standard Atmosphere 1976 below 32 km. The atmosphere is a perfect gas in hydrostatic
equilibrium whose temperature is piecewise linear in *geopotential* altitude H:

    layer        H_base [m]   T_base [K]   lapse dT/dH [K/m]
    troposphere        0        288.15        -0.0065
    tropopause     11000        216.65         0
    stratosphere   20000        216.65        +0.001     (valid to 32 km)

Integrating dP/dH = -rho g0 with P = rho R T gives the closed forms used below. A uniform
temperature offset ``dT`` (an "ISA+dT day") changes T and rho but, by convention, not the
pressure-altitude relation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_SL = 288.15  # K
P_SL = 101325.0  # Pa
RHO_SL = 1.225  # kg/m^3 (= P_SL / (R_AIR T_SL), to 4 figures)
G0 = 9.80665  # m/s^2
# J/(kg K): US 1976 gas constant R* = 8314.32 J/(kmol K) over sea-level molar mass
# M0 = 28.9644 kg/kmol. (ICAO Doc 7488 prints 287.05287; the 7e-7 relative difference is why
# the two documents' tabulated pressures can differ in the 7th significant figure.)
R_AIR = 8314.32 / 28.9644
GAMMA_AIR = 1.4
R_EARTH = 6356766.0  # m, radius used to define geopotential altitude

# (base geopotential altitude [m], base temperature [K], lapse rate [K/m])
LAYERS = ((0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.001))
H_MAX = 32000.0


def _base_pressures() -> tuple[float, ...]:
    """Pressure at each layer base, from integrating hydrostatics up from sea level."""
    p = [P_SL]
    for (h0, t0, lam), (h1, _, _) in zip(LAYERS[:-1], LAYERS[1:], strict=True):
        p.append(_layer_pressure(h1 - h0, t0, lam, p[-1]))
    return tuple(p)


def _layer_pressure(dh, t0, lam, p0):
    if lam == 0.0:
        return p0 * np.exp(-G0 * dh / (R_AIR * t0))
    return p0 * (1.0 + lam * dh / t0) ** (-G0 / (R_AIR * lam))


P_BASE = _base_pressures()


def geometric_to_geopotential(z):
    """Geopotential altitude H = R z / (R + z) for geometric altitude z [m]."""
    z = np.asarray(z, dtype=float)
    return R_EARTH * z / (R_EARTH + z)


@dataclass(frozen=True)
class Atmosphere:
    """Atmospheric state. Arrays broadcast like the altitude input."""

    T: np.ndarray  # static temperature [K]
    P: np.ndarray  # static pressure [Pa]
    rho: np.ndarray  # density [kg/m^3]

    @property
    def a(self):
        """Speed of sound [m/s]."""
        return np.sqrt(GAMMA_AIR * R_AIR * self.T)

    @property
    def theta(self):
        """Temperature ratio T / T_SL (used to correct engine parameters)."""
        return self.T / T_SL

    @property
    def delta(self):
        """Pressure ratio P / P_SL."""
        return self.P / P_SL

    @property
    def sigma(self):
        """Density ratio rho / rho_SL."""
        return self.rho / RHO_SL


def isa(h, dT: float = 0.0, geopotential: bool = False) -> Atmosphere:
    """ISA state at altitude ``h`` [m] (geometric unless ``geopotential``), 0 <= h <= 32 km.

    ``dT`` [K] shifts the temperature of the whole column (ISA+dT day) at fixed pressure.
    """
    h = np.asarray(h, dtype=float)
    H = h if geopotential else geometric_to_geopotential(h)
    if np.any(H < -1e-9) or np.any(H > H_MAX + 1e-9):
        raise ValueError("ISA model is defined here for 0 <= H <= 32 km")
    T = np.empty_like(H)
    P = np.empty_like(H)
    for i, (h0, t0, lam) in enumerate(LAYERS):
        top = LAYERS[i + 1][0] if i + 1 < len(LAYERS) else np.inf
        m = (H >= h0) & (H < top) if i else (H < top)
        dh = H[m] - h0
        T[m] = t0 + lam * dh
        P[m] = _layer_pressure(dh, t0, lam, P_BASE[i])
    T = T + dT
    return Atmosphere(T=T[()], P=P[()], rho=(P / (R_AIR * T))[()])


def pressure_altitude(P, geopotential: bool = False):
    """Altitude [m] at which the standard atmosphere has pressure ``P`` [Pa] (inverse of isa)."""
    P = np.asarray(P, dtype=float)
    H = np.empty_like(P)
    for i, (h0, t0, lam) in enumerate(LAYERS):
        lo = P_BASE[i + 1] if i + 1 < len(LAYERS) else 0.0
        m = (P <= P_BASE[i] if i else np.ones_like(P, bool)) & (P > lo)
        if lam == 0.0:
            H[m] = h0 - R_AIR * t0 / G0 * np.log(P[m] / P_BASE[i])
        else:
            H[m] = h0 + t0 / lam * ((P[m] / P_BASE[i]) ** (-R_AIR * lam / G0) - 1.0)
    if geopotential:
        return H[()]
    return (R_EARTH * H / (R_EARTH - H))[()]
