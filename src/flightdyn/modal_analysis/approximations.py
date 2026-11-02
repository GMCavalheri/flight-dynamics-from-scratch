"""Classical low-order approximations of the flight modes.

Each one keeps the few derivatives that dominate a mode and neglects the coupling to the
others (Etkin & Reid ch. 6 and 8; Nelson ch. 4 and 5; McRuer, Ashkenas & Graham ch. 6-7).
They are tested against the exact eigenvalues of the 4-state models, not as substitutes for
them.
"""

from __future__ import annotations

import numpy as np

from ..aero_derivatives import DimensionalDerivatives
from ..units import G0
from .modes import Mode


def _quadratic(b: float, c: float, name: str) -> Mode:
    """Roots of s^2 + b s + c."""
    roots = np.roots([1.0, b, c]).astype(complex)
    return Mode(name, tuple(sorted(roots, key=lambda z: z.imag)))


def short_period(dd: DimensionalDerivatives) -> Mode:
    """Constant-speed pitching: (w, q) with u fixed (Etkin & Reid, ch. 6).

    s^2 - (Zw + Mq + Mwdot (U0 + Zq)) s + Zw Mq - (U0 + Zq) Mw = 0  (Zwdot neglected).
    """
    U = dd.U0 + dd.Zq
    return _quadratic(-(dd.Zw + dd.Mq + dd.Mwdot * U), dd.Zw * dd.Mq - U * dd.Mw, "short_period")


def phugoid_lanchester(V: float, CL: float, CD: float, g: float = G0) -> Mode:
    """Lanchester's phugoid: wn = sqrt(2) g / V, zeta = CD / (sqrt(2) CL)."""
    wn, zeta = np.sqrt(2.0) * g / V, CD / (np.sqrt(2.0) * CL)
    return _quadratic(2 * zeta * wn, wn * wn, "phugoid")


def phugoid(dd: DimensionalDerivatives, g: float = G0) -> Mode:
    """Two-state phugoid (u, theta), with w and q at their quasi-static values (Nelson, ch. 4):
    s^2 - Xu s - g Zu / U0 = 0."""
    return _quadratic(-dd.Xu, -g * dd.Zu / dd.U0, "phugoid")


def roll_subsidence(dd: DimensionalDerivatives) -> Mode:
    """Single-degree roll: p' = L'p p, lambda = L'p."""
    return Mode("roll", (complex(dd.Lp_p),))


def spiral(dd: DimensionalDerivatives, g: float = G0) -> Mode:
    """Spiral root as minus the constant over the linear coefficient of the lateral
    characteristic polynomial, keeping the dominant term of the latter (Etkin & Reid ch. 8):

    lambda = - g (L'b N'r - N'b L'r) / (V (L'b N'p - N'b L'p))   (theta0 = 0, Y_p = Y_r = 0).
    """
    num = dd.Lb_p * dd.Nr_p - dd.Nb_p * dd.Lr_p
    den = dd.Lb_p * dd.Np_p - dd.Nb_p * dd.Lp_p
    return Mode("spiral", (complex(-g * num / (dd.V * den)),))


def dutch_roll(dd: DimensionalDerivatives) -> Mode:
    """Two-state (beta, r) dutch roll with roll motion suppressed (Nelson, ch. 5), from
    beta' = Yv beta - r and r' = N'b beta + N'r r:

    s^2 - (Yv + N'r) s + (N'b + Yv N'r) = 0.
    """
    return _quadratic(-(dd.Yv + dd.Nr_p), dd.Nb_p + dd.Yv * dd.Nr_p, "dutch_roll")
