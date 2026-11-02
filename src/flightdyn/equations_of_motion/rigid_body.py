"""Nonlinear 6-DOF equations of motion of a rigid aircraft over a flat, non-rotating Earth.

Body axes (x forward, y right, z down) with origin at the centre of gravity; Earth axes NED.
With body velocity v = (u, v, w), body rates w = (p, q, r), applied force F and moment M
(aerodynamics and propulsion, gravity excluded), Newton and Euler in the rotating body frame
give (Etkin & Reid ch. 4; Stevens, Lewis & Johnson ch. 1):

    dv/dt   = F/m + C g_ned - w x v
    I dw/dt = M - w x (I w)
    attitude: 3-2-1 Euler-angle rates, or the quaternion rate
    dr_ned/dt = C^T v,          C = body-from-NED direction cosine matrix

The 12-state Euler-angle vector is ``u v w p q r phi theta psi x_n y_e z_d``; the 13-state
quaternion vector replaces the three angles by ``q0 q1 q2 q3``.
"""

from __future__ import annotations

import numpy as np

from .. import rotations
from ..aircraft import MassProperties
from ..units import G0

U, V, W, P, Q, R, PHI, THETA, PSI, XN, YE, ZD = range(12)
STATE_NAMES = ("u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "x_n", "y_e", "z_d")
# quaternion state: velocity, rates, quaternion (6..9), NED position (10..12)
Q0, XN_Q, YE_Q, ZD_Q = 6, 10, 11, 12


def _dynamics(vel, omega, C, force, moment, mp: MassProperties, g: float):
    g_body = C @ np.array([0.0, 0.0, g])
    vel_dot = np.asarray(force) / mp.mass + g_body - np.cross(omega, vel)
    I = mp.inertia
    omega_dot = mp.inertia_inv @ (np.asarray(moment) - np.cross(omega, I @ omega))
    pos_dot = C.T @ vel
    return vel_dot, omega_dot, pos_dot


def rigid_body_derivative(x, force, moment, mp: MassProperties, g: float = G0) -> np.ndarray:
    """Time derivative of the 12-state Euler-angle vector ``x``.

    ``force`` and ``moment`` are body-axis vectors [N, N m] about the centre of gravity,
    excluding gravity, which is added here.
    """
    x = np.asarray(x, dtype=float)
    vel, omega = x[U : W + 1], x[P : R + 1]
    phi, theta, psi = x[PHI], x[THETA], x[PSI]
    C = rotations.dcm_body_from_ned(phi, theta, psi)
    vel_dot, omega_dot, pos_dot = _dynamics(vel, omega, C, force, moment, mp, g)
    att_dot = rotations.euler_rates(phi, theta, *omega)
    return np.concatenate([vel_dot, omega_dot, att_dot, pos_dot])


def rigid_body_derivative_quat(xq, force, moment, mp: MassProperties, g: float = G0):
    """Time derivative of the 13-state quaternion vector (no gimbal-lock singularity)."""
    xq = np.asarray(xq, dtype=float)
    vel, omega, quat = xq[0:3], xq[3:6], xq[Q0 : Q0 + 4]
    C = rotations.dcm_from_quat(quat)
    vel_dot, omega_dot, pos_dot = _dynamics(vel, omega, C, force, moment, mp, g)
    quat_dot = rotations.quat_rates(quat, *omega)
    return np.concatenate([vel_dot, omega_dot, quat_dot, pos_dot])


def euler_to_quat_state(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.concatenate([x[:6], rotations.quat_from_euler(*x[PHI : PSI + 1]), x[XN:]])


def quat_to_euler_state(xq) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    return np.concatenate([xq[:6], rotations.euler_from_quat(xq[Q0 : Q0 + 4]), xq[XN_Q:]])


def angular_momentum_ned(x, mp: MassProperties) -> np.ndarray:
    """Angular momentum about the CG in Earth axes, C^T I w (constant without moments)."""
    x = np.asarray(x, dtype=float)
    C = rotations.dcm_body_from_ned(*x[PHI : PSI + 1])
    return C.T @ (mp.inertia @ x[P : R + 1])


def kinetic_energy(x, mp: MassProperties) -> float:
    """Translational plus rotational kinetic energy [J]."""
    x = np.asarray(x, dtype=float)
    vel, omega = x[U : W + 1], x[P : R + 1]
    return float(0.5 * mp.mass * vel @ vel + 0.5 * omega @ mp.inertia @ omega)
