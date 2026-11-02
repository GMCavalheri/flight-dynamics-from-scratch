"""Sweeps of CG position, tail size and lateral derivative balance.

Every point is a full trim + numerical linearization of the nonlinear model (longitudinal),
or the analytic stability-axis model at the reference condition (lateral map), so the trade
studies use exactly the machinery validated in Lessons 3-5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..aero_derivatives import dimensional_derivatives, to_stability_axes
from ..aircraft import Aircraft
from ..atmosphere import isa
from ..linearization import lateral_model, linearize, longitudinal
from ..modal_analysis import lateral_modes, longitudinal_modes
from ..trim_solver import trim
from .cg import move_cg, neutral_point_shift
from .tail import HorizontalTail, scale_tail


@dataclass(frozen=True)
class LongitudinalPoint:
    dh: float  # CG shift from the reference [mean chords, aft positive]
    static_margin: float  # neutral point minus CG [mean chords]
    de_trim: float  # [rad]
    eigenvalues: np.ndarray
    modes: dict | None  # phugoid / short period, when both are oscillatory

    @property
    def max_real(self) -> float:
        return float(self.eigenvalues.real.max())

    @property
    def det(self) -> float:
        """Product of the roots = constant term of the characteristic polynomial. It changes
        sign when a real root crosses zero: aperiodic divergence (Etkin & Reid ch. 6)."""
        return float(np.prod(self.eigenvalues).real)


def longitudinal_point(ac: Aircraft, V: float, h: float, dh: float = 0.0) -> LongitudinalPoint:
    a = move_cg(ac, dh)
    t = trim(a, V, h)
    A = longitudinal(linearize(a, t)).A
    ev = np.linalg.eigvals(A)
    oscillatory = sum(abs(z.imag) > 1e-9 for z in ev) == 4
    return LongitudinalPoint(
        dh=dh,
        static_margin=neutral_point_shift(a),
        de_trim=t.de,
        eigenvalues=ev,
        modes=longitudinal_modes(A) if oscillatory else None,
    )


def divergence_limit(ac: Aircraft, V: float, h: float, bracket=(0.0, 0.4)) -> float:
    """Aft CG shift [chords] at which a real longitudinal root crosses zero (det A = 0).

    The weakly damped phugoid may cross the imaginary axis first, as a very slow oscillatory
    divergence; that is a handling-qualities matter, not a CG limit, and is not searched for.
    """
    return brentq(lambda dh: longitudinal_point(ac, V, h, dh).det, *bracket, xtol=1e-6)


def trim_limit(ac: Aircraft, V: float, h: float, de_limit: float, bracket=(-0.6, 0.2)) -> float:
    """CG shift [chords] at which trimming at (V, h) needs elevator ``de_limit``."""
    return brentq(lambda dh: trim(move_cg(ac, dh), V, h).de - de_limit, *bracket, xtol=1e-6)


def scissors(
    ac_aft: Aircraft,
    cond_aft: dict,
    ac_fwd: Aircraft,
    cond_fwd: dict,
    tail_aft: HorizontalTail,
    tail_fwd: HorizontalTail,
    scales,
    de_limit: float,
    margin: float = 0.05,
):
    """CG limits against tail size, in chords from the reference CG.

    Aft: the neutral point, the static-margin limit (neutral point minus ``margin``) and the
    aperiodic divergence limit, in the ``cond_aft`` condition. Forward: the CG at which trim in
    ``cond_fwd`` needs ``de_limit`` of elevator.
    """
    rows = []
    for k in scales:
        aft = scale_tail(ac_aft, tail_aft, k)
        fwd = scale_tail(ac_fwd, tail_fwd, k)
        hn = neutral_point_shift(aft)
        rows.append(
            dict(
                k=k,
                neutral=hn,
                static_limit=hn - margin,
                divergence=divergence_limit(aft, cond_aft["V"], cond_aft["h"],
                                            bracket=(hn - 0.15, hn + 0.05)),
                forward_limit=trim_limit(fwd, cond_fwd["V"], cond_fwd["h"], de_limit),
            )
        )  # fmt: skip
    return rows


def lateral_dimensional(ac: Aircraft, V: float, h: float, k_lb: float = 1.0, k_nb: float = 1.0):
    """Stability-axis dimensional derivatives at the reference, with Cl_b and Cn_b scaled."""
    st = to_stability_axes(ac)
    d = st.aero.with_(Cl_b=k_lb * st.aero.Cl_b, Cn_b=k_nb * st.aero.Cn_b)
    rho = float(isa(h, geopotential=True).rho)
    return dimensional_derivatives(d, st.geometry, st.mass, rho, V)


def spiral_criterion(dd) -> float:
    """L'b N'r - N'b L'r: positive for a stable spiral (Etkin & Reid ch. 8; theta0 = 0)."""
    return dd.Lb_p * dd.Nr_p - dd.Nb_p * dd.Lr_p


def lateral_map(ac: Aircraft, V: float, h: float, k_lb, k_nb):
    """Spiral root and dutch-roll damping on a grid of Cl_b and Cn_b scale factors."""
    spiral = np.full((len(k_nb), len(k_lb)), np.nan)
    zeta = np.full_like(spiral, np.nan)
    crit = np.full_like(spiral, np.nan)
    for i, kn in enumerate(k_nb):
        for j, kl in enumerate(k_lb):
            dd = lateral_dimensional(ac, V, h, kl, kn)
            crit[i, j] = spiral_criterion(dd)
            try:
                m = lateral_modes(lateral_model(dd, 0.0).A)
            except ValueError:  # dutch roll split into real roots
                continue
            spiral[i, j] = m["spiral"].eigenvalues[0].real
            zeta[i, j] = m["dutch_roll"].zeta
    return spiral, zeta, crit
