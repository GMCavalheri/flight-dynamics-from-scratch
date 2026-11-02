"""Modes from time histories: the matrix pencil method.

A free response sampled every dt is a sum of damped exponentials, y_k = sum_i c_i z_i^k with
z_i = exp(lambda_i dt). Shifted Hankel matrices of the data then satisfy a generalized
eigenvalue problem whose eigenvalues are the z_i (Hua & Sarkar, IEEE Trans. ASSP 38(5), 1990).
With several output channels the Hankel matrices are stacked, so every channel constrains the
same poles. The singular value decomposition filters the data down to ``order`` poles.

No model is used: applied to the nonlinear simulation, the method recovers the frequencies and
damping the *nonlinear* aircraft actually shows, to compare with the linear eigenvalues.
"""

from __future__ import annotations

import numpy as np


def hankel(y: np.ndarray, L: int) -> np.ndarray:
    """(N - L) x (L + 1) Hankel matrix of one channel."""
    N = len(y)
    return np.array([y[i : i + L + 1] for i in range(N - L)])


def matrix_pencil(y, dt: float, order: int, L: int | None = None) -> np.ndarray:
    """Continuous-time poles lambda of the samples ``y`` (shape (N,) or (N, channels))."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    N = y.shape[0]
    L = N // 3 if L is None else L
    # scale channels so each counts equally
    scale = np.sqrt(np.mean(y**2, axis=0))
    scale[scale == 0] = 1.0
    Y = np.vstack([hankel(y[:, j] / scale[j], L) for j in range(y.shape[1])])
    _, _, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh[:order].conj().T  # (L + 1) x order
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    return np.log(z.astype(complex)) / dt


def damped_pairs(poles, min_imag: float = 1e-9):
    """Upper-half-plane oscillatory poles and real poles, sorted by magnitude."""
    poles = np.asarray(poles)
    osc = sorted((p for p in poles if p.imag > min_imag), key=abs)
    real = sorted((p.real for p in poles if abs(p.imag) <= min_imag), key=abs)
    return osc, real
