"""Attitude kinematics: direction cosine matrices, Euler angles and quaternions.

Frames: Earth axes North-East-Down (NED) and body axes (x forward, y right wing, z down).
Euler angles are the aerospace 3-2-1 sequence: yaw ``psi`` about z, then pitch ``theta`` about
the new y, then roll ``phi`` about the new x (Stevens, Lewis & Johnson, *Aircraft Control and
Simulation*, ch. 1; Etkin & Reid, *Dynamics of Flight*, ch. 4). Quaternions are scalar-first,
``q = (q0, q1, q2, q3)``, and represent the same body-from-NED rotation.
"""

from __future__ import annotations

import numpy as np


def _r1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _r2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _r3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def dcm_body_from_ned(phi: float, theta: float, psi: float) -> np.ndarray:
    """Matrix C with v_body = C @ v_ned for the 3-2-1 Euler angles (rad)."""
    return _r1(phi) @ _r2(theta) @ _r3(psi)


def euler_from_dcm(C: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`dcm_body_from_ned`: (phi, theta, psi), theta in [-pi/2, pi/2]."""
    theta = -np.arcsin(np.clip(C[0, 2], -1.0, 1.0))
    phi = np.arctan2(C[1, 2], C[2, 2])
    psi = np.arctan2(C[0, 1], C[0, 0])
    return float(phi), float(theta), float(psi)


def euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> np.ndarray:
    """(phi_dot, theta_dot, psi_dot) from body rates; singular at theta = +/-90 deg."""
    sp, cp = np.sin(phi), np.cos(phi)
    tt, ct = np.tan(theta), np.cos(theta)
    return np.array(
        [
            p + (q * sp + r * cp) * tt,
            q * cp - r * sp,
            (q * sp + r * cp) / ct,
        ]
    )


def quat_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """Unit quaternion of the 3-2-1 rotation (scalar first)."""
    cf, sf = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cs, ss = np.cos(psi / 2), np.sin(psi / 2)
    return np.array(
        [
            cf * ct * cs + sf * st * ss,
            sf * ct * cs - cf * st * ss,
            cf * st * cs + sf * ct * ss,
            cf * ct * ss - sf * st * cs,
        ]
    )


def dcm_from_quat(q: np.ndarray) -> np.ndarray:
    """Body-from-NED direction cosine matrix of a (not necessarily unit) quaternion."""
    q0, q1, q2, q3 = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ]
    )


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    return euler_from_dcm(dcm_from_quat(q))


def quat_rates(q: np.ndarray, p: float, qr: float, r: float, k: float = 1.0) -> np.ndarray:
    """Quaternion derivative for body rates (p, q, r).

    The term ``k (1 - |q|^2) q`` pulls the quaternion back to unit norm (Stevens, Lewis &
    Johnson, ch. 1), so integration error in the norm decays instead of accumulating; it is
    zero on the unit sphere and leaves the rotation itself unchanged.
    """
    q = np.asarray(q, dtype=float)
    omega = np.array(
        [
            [0.0, -p, -qr, -r],
            [p, 0.0, r, -qr],
            [qr, -r, 0.0, p],
            [r, qr, -p, 0.0],
        ]
    )
    return 0.5 * omega @ q + k * (1.0 - q @ q) * q


def wind_angles(u: float, v: float, w: float) -> tuple[float, float, float]:
    """(V, alpha, beta) from body velocity components; beta = asin(v / V)."""
    V = float(np.sqrt(u * u + v * v + w * w))
    return V, float(np.arctan2(w, u)), float(np.arcsin(v / V))


def body_velocity(V: float, alpha: float, beta: float) -> np.ndarray:
    """Body-axis velocity (u, v, w) of airspeed V at angles alpha, beta (inverse of wind_angles)."""
    return V * np.array([np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)])
