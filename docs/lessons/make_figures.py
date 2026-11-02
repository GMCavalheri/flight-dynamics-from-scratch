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
from flightdyn.aircraft import MassProperties, ThrustLine
from flightdyn.rotations import euler_rates
from flightdyn.trim_solver import trim, trim_guess
from flightdyn.units import G0, KT, LBF

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
from reference_aircraft import b747, b747_condition  # noqa: E402

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


def _trim_sweep(ac, speeds, h, **kw):
    out, guess = [], None
    W = ac.weight
    for V in speeds:
        t = trim(ac, V, h, guess=guess, **kw)
        guess = trim_guess(t, W)
        out.append(t)
    return out


def l3_trim_curves():
    """747 power-approach model trimmed across speed: alpha, elevator, thrust."""
    ac, c = b747(2), b747_condition(2)
    kt = np.linspace(140.0, 200.0, 25)
    no_line = ac.with_(thrust=ThrustLine(0.0, 0.0))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for model, label, color in [
        (ac, "published thrust line", P.SERIES[0]),
        (no_line, "thrust through the CG, along x", P.SERIES[1]),
    ]:
        ts = _trim_sweep(model, kt * KT, c["h"])
        axes[0].plot(kt, [np.degrees(t.alpha) for t in ts], color=color, label=label)
        axes[1].plot(kt, [np.degrees(t.de) for t in ts], color=color)
        axes[2].plot(kt, [t.thrust / LBF / 1e3 for t in ts], color=color)
    ref_kt = c["V"] / KT
    axes[0].plot(ref_kt, np.degrees(c["alpha"]), "o", color=P.REFERENCE, label="CR-2144 fc 2")
    axes[1].plot(ref_kt, 0.0, "o", color=P.REFERENCE)
    for ax, lab in zip(
        axes,
        ["angle of attack α [deg]", "elevator δe from reference [deg]", "thrust [klbf]"],
        strict=True,
    ):
        ax.set_xlabel("true airspeed [kt]")
        ax.set_ylabel(lab)
    axes[0].set_title("Trim of the 747 power-approach model, sea level")
    axes[0].legend(loc="upper right", fontsize=8)
    save(fig, "l3_trim_curves.png")


def l3_turns():
    """Coordinated level turns: bank, load factor and elevator per g."""
    ac, c = b747(2), b747_condition(2)
    rates = np.radians(np.linspace(0.0, 3.0, 13))
    ts, guess = [], None
    for w in rates:
        t = trim(ac, c["V"], c["h"], turn_rate=w, guess=guess)
        guess = trim_guess(t, ac.weight)
        ts.append(t)
    phi = np.array([t.phi for t in ts])
    n = 1.0 / np.cos(phi)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(np.degrees(rates), np.degrees(phi), color=P.SERIES[0], label="trim")
    axes[0].plot(
        np.degrees(rates),
        np.degrees(np.arctan(c["V"] * rates / G0)),
        "--",
        color=P.REFERENCE,
        lw=1,
        label="tan φ = Vψ̇/g",
    )
    axes[0].set_xlabel("turn rate ψ̇ [deg/s]")
    axes[0].set_ylabel("bank angle φ [deg]")
    axes[0].set_title("Coordinated level turns, 165 kt")
    axes[0].legend()
    axes[1].plot(n, [np.degrees(t.de) for t in ts], color=P.SERIES[0])
    axes[1].set_xlabel("load factor n = 1/cos φ")
    axes[1].set_ylabel("elevator δe [deg]")
    axes[1].set_title("Elevator per g")
    save(fig, "l3_turns.png")


FIGURES = [l1_precession, l1_gimbal_lock, l3_trim_curves, l3_turns]

if __name__ == "__main__":
    P.use_style()
    for make in FIGURES:
        make()
        print(make.__name__)
