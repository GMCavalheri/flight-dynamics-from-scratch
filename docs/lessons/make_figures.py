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
from flightdyn.linearization import lateral, linearize, longitudinal
from flightdyn.modal_analysis import lateral_modes, longitudinal_modes, mode_shape
from flightdyn.nonlinear_sim import Pulse, simulate, simulate_linear
from flightdyn.rotations import euler_rates
from flightdyn.trade_studies import move_cg, neutral_point_shift
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
    untrimmed = ac.with_(aero=ac.aero.with_(Cm0=0.0))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for model, label, color in [
        (ac, "Cm0 balances the thrust moment", P.SERIES[0]),
        (untrimmed, "Cm0 = 0", P.SERIES[1]),
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


def _b747_fc2_linear():
    ac, c = b747(2), b747_condition(2)
    t = trim(ac, c["V"], c["h"])
    lm = linearize(ac, t)
    return longitudinal(lm), lateral(lm, c["V"]), c["V"]


def l4_s_plane():
    """All eight roots of the 747 power-approach model, with a zoom on the slow ones."""
    lon, lat, _ = _b747_fc2_linear()
    modes = longitudinal_modes(lon.A) | lateral_modes(lat.A)
    colors = dict(
        short_period=P.SERIES[0],
        phugoid=P.SERIES[1],
        dutch_roll=P.SERIES[2],
        roll=P.SERIES[3],
        spiral=P.SERIES[4],
    )
    labels = dict(
        short_period="short period",
        phugoid="phugoid",
        dutch_roll="dutch roll",
        roll="roll subsidence",
        spiral="spiral",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax in axes:
        for name, m in modes.items():
            ev = np.array(m.eigenvalues)
            ax.plot(ev.real, ev.imag, "x", ms=9, mew=2, color=colors[name], label=labels[name])
        ax.axhline(0, color=P.TEXT_MUTED, lw=0.8)
        ax.axvline(0, color=P.TEXT_MUTED, lw=0.8)
        ax.set_xlabel("real part σ [1/s]")
        ax.set_ylabel("imaginary part ω [rad/s]")
    for zeta in (0.1, 0.3, 0.6):
        r = np.linspace(0, 1.2, 50)
        th = np.arccos(zeta)
        axes[0].plot(-r * np.cos(th), r * np.sin(th), ":", color=P.GRID, lw=1)
        axes[0].annotate(f"ζ = {zeta}", (-1.15 * np.cos(th), 1.15 * np.sin(th)), fontsize=8,
                         color=P.TEXT_MUTED)  # fmt: skip
    axes[0].set_title("747, power approach: the eight roots")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].set_xlim(-0.08, 0.02)
    axes[1].set_ylim(-0.2, 0.2)
    axes[1].set_title("Zoom: phugoid and spiral")
    save(fig, "l4_s_plane.png")


def _phasor(ax, vec, names, title):
    for k, (z, name) in enumerate(zip(vec, names, strict=True)):
        ax.annotate(
            "", xy=(np.angle(z), abs(z)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=P.SERIES[k], lw=2),
        )  # fmt: skip
        ax.plot([], [], color=P.SERIES[k], lw=2, label=f"{name}  |{abs(z):.2f}|")
    ax.set_rmax(1.15 * max(abs(vec)))
    ax.set_title(title, pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.85, 1.12), fontsize=8)
    ax.set_yticklabels([])


def l4_mode_shapes():
    """Phasor diagrams: relative amplitude and phase of each state in a mode."""
    lon, lat, V = _b747_fc2_linear()
    ml = longitudinal_modes(lon.A)
    md = lateral_modes(lat.A)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), subplot_kw=dict(projection="polar"))
    # longitudinal states as angles: u/V, alpha = w/V, theta; normalised to theta = 1
    S = np.diag([1 / V, 1 / V, 1.0, 1.0])
    for ax, name, title in (
        (axes[0], "short_period", "Short period (θ = 1)"),
        (axes[1], "phugoid", "Phugoid (θ = 1)"),
    ):
        lam = ml[name].eigenvalues[-1]
        v = S @ mode_shape(lon.A, lam, normalise=3)
        _phasor(ax, v[[0, 1, 3]], ["u/V", "α", "θ"], title)
    lam = md["dutch_roll"].eigenvalues[-1]
    v = mode_shape(lat.A, lam, normalise=0)
    psi = v[2] / lam  # heading: psi_dot = r (theta0 small)
    _phasor(axes[2], np.array([v[0], v[3], psi]), ["β", "φ", "ψ"], "Dutch roll (β = 1)")
    save(fig, "l4_mode_shapes.png")


def l5_large_bank():
    """A large aileron input: where the linear model stops describing the aircraft."""
    ac, c = b747(2), b747_condition(2)
    t = trim(ac, c["V"], 3000.0)
    lm = linearize(ac, t)
    inp = [Pulse("da", np.radians(20.0), 1.0, 20.0)]
    nl = simulate(ac, t.x, t.controls, inp, t_end=60.0, dt=0.05)
    # lateral linear model with heading: (v, p, r, phi, psi)
    lat_idx = [1, 3, 5, 6, 8]
    lin = simulate_linear(lm.submodel(lat_idx, [1, 2]), inp, t_end=60.0, dt=0.05,
                          control_index=(1, 2))  # fmt: skip
    d = nl.x - t.x
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    rows = [
        (axes[0, 0], "bank angle φ [deg]", np.degrees(d[:, 6]), np.degrees(lin.x[:, 3])),
        (axes[0, 1], "heading ψ [deg]", np.degrees(d[:, 8]), np.degrees(lin.x[:, 4])),
        (axes[1, 0], "sideslip β [deg]",
         np.degrees(np.arcsin(nl.x[:, 1] / np.linalg.norm(nl.x[:, :3], axis=1))),
         np.degrees(lin.x[:, 0] / c["V"])),
        (axes[1, 1], "altitude change Δh [m]", -d[:, 11], np.zeros_like(lin.t)),
    ]  # fmt: skip
    for ax, lab, a, b in rows:
        ax.plot(nl.t, a, color=P.SERIES[0], label="nonlinear 6-DOF")
        ax.plot(lin.t, b, "--", color=P.REFERENCE, lw=1, label="linear lateral model")
        ax.set_ylabel(lab)
    for ax in axes[1]:
        ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("747, 165 kt: 20° of aileron for 20 s, controls then free", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "l5_large_bank.png")


def l6_cm_alpha():
    """Pitching moment against angle of attack for several CG positions (747, fc 2)."""
    ac = b747(2)
    hn = neutral_point_shift(ac)
    a = np.radians(np.linspace(0.0, 12.0, 50))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, h in enumerate((0.10, 0.25, 0.40, 0.25 + hn, 0.55)):
        d = move_cg(ac, h - 0.25).aero
        cm = d.Cm0 + d.Cm_a * (a - d.alpha0)
        label = f"h = {h:.2f}" + (" (neutral point)" if abs(h - 0.25 - hn) < 1e-9 else "")
        ls = "--" if "neutral" in label else "-"
        ax.plot(np.degrees(a), cm, ls, color=P.SERIES[k], label=label)
    ax.axhline(0, color=P.TEXT_MUTED, lw=0.8)
    ax.axvline(np.degrees(ac.aero.alpha0), color=P.GRID, lw=3, zorder=0)
    ax.set_xlabel("angle of attack α [deg]")
    ax.set_ylabel("aerodynamic C_m about the CG (δe = 0)")
    ax.set_title("747 power approach: the slope of C_m(α) is the static stability")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "l6_cm_alpha.png")


FIGURES = [
    l1_precession,
    l1_gimbal_lock,
    l3_trim_curves,
    l3_turns,
    l4_s_plane,
    l4_mode_shapes,
    l5_large_bank,
    l6_cm_alpha,
]

if __name__ == "__main__":
    P.use_style()
    for make in FIGURES:
        make()
        print(make.__name__)
