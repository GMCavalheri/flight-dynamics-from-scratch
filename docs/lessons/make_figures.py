"""Teaching figures for the lesson series (docs/figures/lessons/).

Run with ``uv run python docs/lessons/make_figures.py``. Validation figures live in
``validation/``; these illustrate ideas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from flightdyn import equations_of_motion as eom
from flightdyn import plotting as P
from flightdyn.aircraft import MassProperties
from flightdyn.rotations import euler_rates

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))

OUT = P.FIGURES / "lessons"
TIGHT = dict(method="DOP853", rtol=1e-10, atol=1e-10)


def save(fig, name):
    return P.save(fig, name, OUT)


def l1_precession():
    """Torque-free precession of an axisymmetric body vs the closed form."""
    I, I3, r0, w0 = 2.0, 5.0, 1.5, 0.4
    mp = MassProperties(mass=1.0, Ixx=I, Iyy=I, Izz=I3)
    x0 = np.zeros(12)
    x0[[eom.P, eom.R]] = (w0, r0)
    t = np.linspace(0.0, 8.0, 400)
    sol = solve_ivp(
        lambda _, x: eom.rigid_body_derivative(x, (0, 0, 0), (0, 0, 0), mp, 0.0),
        (0, t[-1]),
        x0,
        t_eval=t,
        **TIGHT,
    )
    om = (I3 - I) * r0 / I
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, sol.y[eom.P], color=P.SERIES[0], label="p, 6-DOF integration")
    ax.plot(t, sol.y[eom.Q], color=P.SERIES[1], label="q, 6-DOF integration")
    ax.plot(t, w0 * np.cos(om * t), "--", color=P.REFERENCE, lw=1, label="closed form")
    ax.plot(t, w0 * np.sin(om * t), "--", color=P.REFERENCE, lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("body rate [rad/s]")
    ax.set_title("Torque-free precession: (p, q) turns at Ω = (I₃ − I) r / I")
    ax.legend(loc="upper right", ncol=3)
    save(fig, "l1_precession.png")


def l1_gimbal_lock():
    """Euler-angle rates blow up near theta = 90 deg; the quaternion loop does not care."""
    q = 0.5
    mp = MassProperties(mass=1.0, Ixx=1.0, Iyy=1.0, Izz=1.0)
    x0 = np.zeros(12)
    x0[eom.Q] = q
    x0[eom.R] = 0.01  # a little yaw rate, so the Euler-angle rates see the singularity
    t = np.linspace(0.0, 2 * np.pi / q, 600)
    sol = solve_ivp(
        lambda _, x: eom.rigid_body_derivative_quat(x, (0, 0, 0), (0, 0, 0), mp, 0.0),
        (0, t[-1]),
        eom.euler_to_quat_state(x0),
        t_eval=t,
        **TIGHT,
    )
    th = np.array([eom.quat_to_euler_state(sol.y[:, k])[eom.THETA] for k in range(len(t))])
    phi = np.array([eom.quat_to_euler_state(sol.y[:, k])[6] for k in range(len(t))])
    psidot = np.array([euler_rates(phi[k], th[k], *sol.y[3:6, k])[2] for k in range(len(t))])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(t, np.degrees(th), color=P.SERIES[0])
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("pitch angle θ [deg]")
    axes[0].set_title("A loop, integrated with quaternions")
    axes[1].semilogy(t, np.abs(psidot), color=P.SERIES[1])
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("|dψ/dt| [rad/s]")
    axes[1].set_title("Euler-angle rates near θ = ±90°")
    save(fig, "l1_gimbal_lock.png")


FIGURES = [l1_precession, l1_gimbal_lock]

if __name__ == "__main__":
    P.use_style()
    for make in FIGURES:
        make()
        print(make.__name__)
