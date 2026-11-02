"""Animated mode shapes of the Boeing 747 (power approach), from its linear eigenvectors.

Run with ``uv run python validation/mode_animations.py``. Writes
``docs/figures/anim_short_period.gif``, ``anim_phugoid.gif`` and ``anim_dutch_roll.gif``.

Each animation is the free response x(t) = Re(c v exp(lambda t)) of one mode of the trimmed,
numerically linearized model: one eigenvalue lambda, its eigenvector v, and a scale c chosen so
the motion is visible. The amplitude is stated in each title; the relative amplitudes and
phases of the states are the model's. The phugoid runs faster than real time (see its title).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flightdyn import plotting as P  # noqa: E402
from flightdyn.linearization import linearize  # noqa: E402
from flightdyn.trim_solver import trim  # noqa: E402

from reference_aircraft import b747, b747_condition  # noqa: E402

LON = [0, 2, 4, 7, 11]  # u, w, q, theta, z_d
LAT = [1, 3, 5, 6, 8]  # v, p, r, phi, psi
AC, COND = b747(2), b747_condition(2)
TRIM = trim(AC, COND["V"], COND["h"])
LM = linearize(AC, TRIM)
V = COND["V"]
FRAMES = 90
FPS = 15


def mode_response(idx, pick, component, amplitude, t):
    """Free response of the eigenvalue chosen by ``pick`` (a function of the eigenvalues),
    scaled so that state ``component`` peaks at ``amplitude``."""
    A = LM.A[np.ix_(idx, idx)]
    w, vecs = np.linalg.eig(A)
    k = pick(w)
    lam, v = w[k], vecs[:, k]
    v = v / v[component] * amplitude  # component real, = amplitude at t = 0
    return lam, (v[:, None] * np.exp(lam * t)[None, :]).real


def _upper(ws, slow: bool):
    upper = [i for i, z in enumerate(ws) if z.imag > 1e-6]
    return min(upper, key=lambda i: abs(ws[i])) if slow else max(upper, key=lambda i: abs(ws[i]))


def _glyph(ax, xy, color=P.SERIES[0]):
    patch = Polygon(xy, closed=True, fc=color, ec=P.TEXT, lw=0.8)
    ax.add_patch(patch)
    return patch


def short_period():
    t = np.linspace(0.0, 12.0, FRAMES)
    lam, x = mode_response(LON, lambda w: _upper(w, slow=False), 3, np.radians(6.0), t)
    U0, W0 = TRIM.x[0], TRIM.x[2]
    alpha = (U0 * x[1] - W0 * x[0]) / V**2  # perturbation angle of attack
    theta = x[3]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xlim(-1.0, 1.3)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.axhline(0, color=P.GRID, lw=1)
    body = _glyph(ax, P.aircraft_side(0, 0, theta[0]))
    (vel,) = ax.plot([], [], color=P.SERIES[1], lw=2)
    label = ax.text(-0.95, -0.55, "", fontsize=9, color=P.TEXT_MUTED)
    ax.text(-0.95, 0.5, f"Short period: ωn = {abs(lam):.2f} rad/s, ζ = {-lam.real / abs(lam):.2f}"
            "  (θ amplitude 6°)", fontsize=9, color=P.TEXT)  # fmt: skip
    ax.text(0.75, 0.08, "flight path", fontsize=8, color=P.SERIES[1])

    def frame(k):
        body.set_xy(P.aircraft_side(0, 0, theta[k]))
        gamma = theta[k] - alpha[k]
        vel.set_data([0.55, 0.55 + 0.6 * np.cos(gamma)], [0.0, 0.6 * np.sin(gamma)])
        label.set_text(f"t = {t[k]:4.1f} s   Δα = {np.degrees(alpha[k]):+5.1f}°   "
                       f"Δθ = {np.degrees(theta[k]):+5.1f}°")  # fmt: skip
        return body, vel, label

    return FuncAnimation(fig, frame, frames=FRAMES, blit=False)


def phugoid():
    t = np.linspace(0.0, 100.0, FRAMES)
    lam, x = mode_response(LON, lambda w: _upper(w, slow=True), 0, 10.0, t)
    h = -x[4]
    theta = x[3]
    dist = (V * t) / 1000.0
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(width_ratios=[2, 1]))
    ax.set_xlim(0, dist[-1])
    ax.set_ylim(1.3 * h.min(), 1.3 * h.max())
    ax.set_xlabel("distance flown [km]")
    ax.set_ylabel("altitude change [m]")
    speedup = t[-1] / (FRAMES / FPS)
    ax.set_title(f"Phugoid: period {2 * np.pi / lam.imag:.0f} s, ζ = {-lam.real / abs(lam):.3f} "
                 f"(10 m/s amplitude, {speedup:.0f}× speed)")  # fmt: skip
    (trace,) = ax.plot([], [], color=P.SERIES[1], lw=1.5)
    (dot,) = ax.plot([], [], "o", color=P.SERIES[0])
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.axhline(0, color=P.GRID, lw=1)
    body = _glyph(ax2, P.aircraft_side(0, 0, theta[0]))
    label = ax2.text(-0.75, -0.55, "", fontsize=9, color=P.TEXT_MUTED)

    def frame(k):
        trace.set_data(dist[: k + 1], h[: k + 1])
        dot.set_data([dist[k]], [h[k]])
        body.set_xy(P.aircraft_side(0, 0, theta[k]))
        U0, W0 = TRIM.x[0], TRIM.x[2]
        dv = (U0 * x[0, k] + W0 * x[1, k]) / V
        label.set_text(f"ΔV = {dv:+5.1f} m/s\nΔθ = {np.degrees(theta[k]):+5.1f}°")
        return trace, dot, body, label

    fig.tight_layout()
    return FuncAnimation(fig, frame, frames=FRAMES, blit=False)


def dutch_roll():
    t = np.linspace(0.0, 17.0, FRAMES)
    lam, x = mode_response(LAT, lambda w: _upper(w, slow=False), 0, np.radians(6.0) * V, t)
    beta, phi, psi = x[0] / V, x[3], x[4]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8))
    for a in (ax, ax2):
        a.set_aspect("equal")
        a.axis("off")
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.75, 0.75)
    ax.set_title("Top view: heading and sideslip", fontsize=10)
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_title("Rear view: bank", fontsize=10)
    ax2.axhline(-0.4, color=P.GRID, lw=1)
    top = _glyph(ax, P.aircraft_top(0, 0, np.pi / 2))
    rear = _glyph(ax2, P.aircraft_rear(0.0))
    (vel,) = ax.plot([], [], color=P.SERIES[1], lw=2)
    ax.text(0.12, 0.62, "flight path", fontsize=8, color=P.SERIES[1])
    label = fig.text(0.02, 0.03, "", fontsize=9, color=P.TEXT_MUTED)
    fig.suptitle(f"Dutch roll: ωn = {abs(lam):.2f} rad/s, ζ = {-lam.real / abs(lam):.2f} "
                 "(β amplitude 6°)", fontsize=11, x=0.02, ha="left")  # fmt: skip

    def frame(k):
        # page up = initial heading; heading psi is clockwise seen from above
        top.set_xy(P.aircraft_top(0, 0, np.pi / 2 - psi[k]))
        chi = psi[k] + beta[k]  # direction of the velocity: sideslip is wind from the right
        vel.set_data([0, 0.7 * np.sin(chi)], [0, 0.7 * np.cos(chi)])
        rear.set_xy(P.aircraft_rear(phi[k]))
        b, f, p = (np.degrees(v[k]) for v in (beta, phi, psi))
        label.set_text(f"t = {t[k]:4.1f} s   β = {b:+5.1f}°   φ = {f:+5.1f}°   Δψ = {p:+5.1f}°")
        return top, vel, rear, label

    return FuncAnimation(fig, frame, frames=FRAMES, blit=False)


if __name__ == "__main__":
    P.use_style()
    for name, make in (
        ("anim_short_period.gif", short_period),
        ("anim_phugoid.gif", phugoid),
        ("anim_dutch_roll.gif", dutch_roll),
    ):
        path = P.save_gif(make(), name, fps=FPS)
        print(path.name, f"{path.stat().st_size / 1024:.0f} kB")
