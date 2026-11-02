"""Eigenvalues of the linear models, sorted into the classical flight modes.

A complex pair lambda = -zeta wn +/- i wn sqrt(1 - zeta^2) is an oscillation with natural
frequency wn = |lambda| and damping ratio zeta = -Re(lambda) / |lambda|; a real root
lambda = -1/T is a subsidence (T > 0) or divergence (T < 0). Two real roots that belong to
one second-order factor (an overdamped short period) are described by the same wn, zeta
through wn^2 = lambda1 lambda2 and 2 zeta wn = -(lambda1 + lambda2).

Classification (Etkin & Reid ch. 6; Nelson ch. 4-5):

* longitudinal (4 roots): the two roots of smallest magnitude are the phugoid, the two of
  largest magnitude the short period;
* lateral (4 roots): the complex pair is the dutch roll; of the two real roots, the larger in
  magnitude is roll subsidence, the smaller the spiral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LN2 = np.log(2.0)


@dataclass(frozen=True)
class Mode:
    """One mode: its eigenvalue(s) and the derived characteristics."""

    name: str
    eigenvalues: tuple[complex, ...]

    @property
    def is_oscillatory(self) -> bool:
        return any(abs(ev.imag) > 0 for ev in self.eigenvalues)

    @property
    def wn(self) -> float:
        """Natural frequency [rad/s] (for a first-order mode, |lambda|)."""
        if len(self.eigenvalues) == 1:
            return float(abs(self.eigenvalues[0]))
        return float(np.sqrt(abs(np.prod(self.eigenvalues).real)))

    @property
    def zeta(self) -> float:
        if len(self.eigenvalues) == 1:
            return float(np.sign(-self.eigenvalues[0].real))
        return float(-np.sum(self.eigenvalues).real / (2.0 * self.wn))

    @property
    def wd(self) -> float:
        """Damped frequency [rad/s]."""
        return float(max(abs(ev.imag) for ev in self.eigenvalues))

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.wd if self.wd > 0 else np.inf

    @property
    def time_constant(self) -> float:
        """T = -1/lambda for a first-order mode (negative: divergence)."""
        return float(-1.0 / self.eigenvalues[0].real)

    @property
    def time_to_half(self) -> float:
        """Time to half amplitude [s]; negative values are times to double."""
        sigma = max(ev.real for ev in self.eigenvalues)
        return float(-LN2 / sigma) if sigma != 0 else np.inf


def eigen(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors, sorted by increasing magnitude."""
    w, v = np.linalg.eig(A)
    order = np.argsort(np.abs(w))
    return w[order], v[:, order]


def _pair(roots) -> tuple[complex, ...]:
    return tuple(sorted((complex(r) for r in roots), key=lambda z: z.imag))


def longitudinal_modes(A: np.ndarray) -> dict[str, Mode]:
    """Phugoid and short period of a 4-state longitudinal model."""
    w, _ = eigen(A)
    if len(w) != 4:
        raise ValueError("expected a 4-state longitudinal model")
    return {
        "phugoid": Mode("phugoid", _pair(w[:2])),
        "short_period": Mode("short_period", _pair(w[2:])),
    }


def lateral_modes(A: np.ndarray) -> dict[str, Mode]:
    """Dutch roll, roll subsidence and spiral of a 4-state lateral model."""
    w, _ = eigen(A)
    if len(w) != 4:
        raise ValueError("expected a 4-state lateral model")
    cplx = [z for z in w if abs(z.imag) > 1e-12 * max(1.0, abs(z))]
    real = sorted((z for z in w if abs(z.imag) <= 1e-12 * max(1.0, abs(z))), key=abs)
    if len(cplx) != 2:
        raise ValueError("no oscillatory dutch roll: lateral roots are all real")
    return {
        "spiral": Mode("spiral", (complex(real[0].real),)),
        "roll": Mode("roll", (complex(real[1].real),)),
        "dutch_roll": Mode("dutch_roll", _pair(cplx)),
    }


def mode_shape(A: np.ndarray, eigenvalue: complex, normalise: int | None = None) -> np.ndarray:
    """Eigenvector of ``eigenvalue``, scaled so component ``normalise`` is 1 (real, positive),
    or to unit length with the largest component real."""
    w, v = np.linalg.eig(A)
    vec = v[:, np.argmin(np.abs(w - eigenvalue))]
    k = int(np.argmax(np.abs(vec))) if normalise is None else normalise
    vec = vec / vec[k]
    return vec if normalise is not None else vec / np.linalg.norm(vec)
