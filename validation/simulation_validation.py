"""Nonlinear simulation against the linear predictions (Boeing 747, power approach).

Run with ``uv run python validation/simulation_validation.py``. Writes
``validation/results/simulation_validation.md`` and ``docs/figures/sim_elevator_doublet.png``,
``sim_rudder_doublet.png``, ``sim_amplitude.png``.

Checks:
* doublet responses of the nonlinear model and its linearization side by side;
* the difference scales as amplitude^2 (elevator) and amplitude^3 (rudder: symmetry);
* the modes of the nonlinear time histories, identified without a model (matrix pencil),
  against the eigenvalues of the linear models with and without altitude.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flightdyn import plotting as P  # noqa: E402
from flightdyn.linearization import linearize  # noqa: E402
from flightdyn.modal_analysis import Mode  # noqa: E402
from flightdyn.nonlinear_sim import (  # noqa: E402
    Doublet,
    damped_pairs,
    matrix_pencil,
    simulate,
    simulate_linear,
)
from flightdyn.trim_solver import trim  # noqa: E402

from reference_aircraft import b747, b747_condition  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LON4, LON_H, LAT = [0, 2, 4, 7], [0, 2, 4, 7, 11], [1, 3, 5, 6]
AC, COND = b747(2), b747_condition(2)
TRIM = trim(AC, COND["V"], COND["h"])
LM = linearize(AC, TRIM)
V = COND["V"]


def responses(inputs, model_idx, ctrl, t_end, dt):
    nl = simulate(AC, TRIM.x, TRIM.controls, inputs, t_end=t_end, dt=dt)
    lin = simulate_linear(LM.submodel(model_idx, list(ctrl)), inputs, t_end=t_end, dt=dt,
                          control_index=ctrl)  # fmt: skip
    return nl, lin


def elevator_figure():
    inp = [Doublet("de", np.radians(2.0), 1.0, 2.0)]
    nl, lin = responses(inp, LON_H, (0, 3), 120.0, 0.05)
    d = nl.x - TRIM.x
    alpha_nl = np.degrees(np.arctan2(nl.x[:, 2], nl.x[:, 0]) - TRIM.alpha)
    speed_nl = np.hypot(nl.x[:, 0], nl.x[:, 2]) - V
    # linear perturbations of airspeed and angle of attack: project (du, dw) on the trim
    # velocity direction and normal to it
    U0, W0 = TRIM.x[0], TRIM.x[2]
    du, dw = lin.x[:, 0], lin.x[:, 1]
    speed_lin = (U0 * du + W0 * dw) / V
    alpha_lin = np.degrees((U0 * dw - W0 * du) / V**2)
    rows = [
        ("angle of attack Δα [deg]", alpha_nl, alpha_lin),
        ("pitch rate q [deg/s]", np.degrees(d[:, 4]), np.degrees(lin.x[:, 2])),
        ("airspeed ΔV [m/s]", speed_nl, speed_lin),
        ("altitude Δh [m]", -d[:, 11], -lin.x[:, 4]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, (lab, a, b) in zip(axes.flat, rows, strict=True):
        ax.plot(nl.t, a, color=P.SERIES[0], label="nonlinear 6-DOF")
        ax.plot(lin.t, b, "--", color=P.REFERENCE, lw=1, label="linear (u, w, q, θ, h)")
        ax.set_ylabel(lab)
    for ax in axes[1]:
        ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("747 power approach: 2° elevator doublet (2 s + 2 s)", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    P.save(fig, "sim_elevator_doublet.png")


def rudder_figure():
    inp = [Doublet("dr", np.radians(3.0), 1.0, 2.0)]
    nl, lin = responses(inp, LAT, (1, 2), 40.0, 0.05)
    d = nl.x - TRIM.x
    beta_nl = np.degrees(np.arcsin(nl.x[:, 1] / np.linalg.norm(nl.x[:, :3], axis=1)))
    rows = [
        ("sideslip β [deg]", beta_nl, np.degrees(lin.x[:, 0] / V)),
        ("roll rate p [deg/s]", np.degrees(d[:, 3]), np.degrees(lin.x[:, 1])),
        ("yaw rate r [deg/s]", np.degrees(d[:, 5]), np.degrees(lin.x[:, 2])),
        ("bank angle φ [deg]", np.degrees(d[:, 6]), np.degrees(lin.x[:, 3])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, (lab, a, b) in zip(axes.flat, rows, strict=True):
        ax.plot(nl.t, a, color=P.SERIES[0], label="nonlinear 6-DOF")
        ax.plot(lin.t, b, "--", color=P.REFERENCE, lw=1, label="linear (β, p, r, φ)")
        ax.set_ylabel(lab)
    for ax in axes[1]:
        ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("747 power approach: 3° rudder doublet — the dutch roll", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    P.save(fig, "sim_rudder_doublet.png")


def amplitude_study():
    amps = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    out = {"de": [], "dr": [], "sp": [], "dr_pole": []}
    eig_lon = damped_pairs(np.linalg.eigvals(LM.A[np.ix_(LON_H, LON_H)]))[0]
    eig_lat = damped_pairs(np.linalg.eigvals(LM.A[np.ix_(LAT, LAT)]))[0]
    for a in amps:
        for ch, idx, model_idx, ctrl in (("de", [4, 7], LON_H, (0, 3)),
                                         ("dr", [5, 6], LAT, (1, 2))):  # fmt: skip
            inp = [Doublet(ch, np.radians(a), 1.0, 2.0)]
            nl, lin = responses(inp, model_idx, ctrl, 30.0, 0.05)
            k = [model_idx.index(i) for i in idx]
            lin_peak = np.abs(lin.x[:, k]).max()
            out[ch].append(np.abs((nl.x[:, idx] - TRIM.x[idx]) - lin.x[:, k]).max() / lin_peak)
            m = nl.t >= 5.0
            Y = nl.x[m][:, model_idx] - TRIM.x[model_idx]
            if ch == "de":
                sp = damped_pairs(matrix_pencil(Y[:, :4], 0.05, 4))[0][-1]
                out["sp"].append(sp)
            else:
                drp = damped_pairs(matrix_pencil(Y, 0.05, 4))[0][-1]
                out["dr_pole"].append(drp)
    return amps, out, eig_lon[-1], eig_lat[-1]


def amplitude_figure(amps, out, sp_eig, dr_eig):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(amps, out["de"], "o-", color=P.SERIES[0], label="elevator doublet (q, θ)")
    ax.loglog(amps, out["dr"], "o-", color=P.SERIES[2], label="rudder doublet (r, φ)")
    ax.loglog(amps, out["de"][2] * (amps / amps[2]) ** 1, ":", color=P.TEXT_MUTED, lw=1,
              label="slope 1 (error/peak ∝ a)")  # fmt: skip
    ax.loglog(amps, out["dr"][2] * (amps / amps[2]) ** 2, "--", color=P.TEXT_MUTED, lw=1,
              label="slope 2 (error/peak ∝ a²)")  # fmt: skip
    ax.set_xticks(amps, [f"{a:g}" for a in amps])
    ax.minorticks_off()
    ax.set_xlabel("doublet amplitude a [deg]")
    ax.set_ylabel("max |nonlinear − linear| / linear peak")
    ax.set_title("Linear model error grows with amplitude")
    ax.legend(fontsize=8)
    ax = axes[1]
    sp_shift = [100 * (abs(p) / abs(sp_eig) - 1) for p in out["sp"]]
    dr_shift = [100 * (abs(p) / abs(dr_eig) - 1) for p in out["dr_pole"]]
    ax.plot(amps, sp_shift, "o-", color=P.SERIES[0], label="short period ωn")
    ax.plot(amps, dr_shift, "o-", color=P.SERIES[2], label="dutch roll ωn")
    ax.axhline(0, color=P.TEXT_MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(amps, [f"{a:g}" for a in amps])
    ax.minorticks_off()
    ax.set_xlabel("doublet amplitude a [deg]")
    ax.set_ylabel("identified − linear [%]")
    ax.set_title("Frequency of the nonlinear response (matrix pencil)")
    ax.legend()
    P.save(fig, "sim_amplitude.png")


def pole_table():
    rows = []
    cases = (("de", LON_H, 300.0, 0.1, 5), ("dr", LAT, 60.0, 0.05, 4))
    for ch, model_idx, t_end, dt, order in cases:
        inp = [Doublet(ch, np.radians(0.25), 1.0, 2.0)]
        nl = simulate(AC, TRIM.x, TRIM.controls, inp, t_end=t_end, dt=dt)
        m = nl.t >= 5.0
        found = matrix_pencil(nl.x[m][:, model_idx] - TRIM.x[model_idx], dt, order)
        if ch == "de":
            full = damped_pairs(np.linalg.eigvals(LM.A[np.ix_(LON_H, LON_H)]))[0]
            four = damped_pairs(np.linalg.eigvals(LM.A[np.ix_(LON4, LON4)]))[0]
            for name, f, e5, e4 in zip(("phugoid", "short period"), damped_pairs(found)[0],
                                       full, four, strict=True):  # fmt: skip
                rows.append((name, Mode(name, (f, f.conjugate())), Mode(name, (e5, e5.conjugate())),
                             Mode(name, (e4, e4.conjugate()))))  # fmt: skip
        else:
            osc, real = damped_pairs(found)
            eig = np.linalg.eigvals(LM.A[np.ix_(LAT, LAT)])
            eo, er = damped_pairs(eig)
            rows.append(("dutch roll", Mode("dr", (osc[0], osc[0].conjugate())),
                         Mode("dr", (eo[0], eo[0].conjugate())), None))  # fmt: skip
            for name, f, e in zip(("spiral", "roll"), real, er, strict=True):
                rows.append((name, Mode(name, (complex(f),)), Mode(name, (complex(e),)), None))
    return rows


def write_results(rows, amps, out):
    L = [
        "# Nonlinear simulation against the linear model",
        "",
        "Generated by `validation/simulation_validation.py`. Boeing 747, CR-2144 flight",
        "condition 2 (power approach, sea level), trimmed; doublets of 2 s + 2 s.",
        "",
        "## Modes of the nonlinear response (0.25° doublets, matrix pencil, t ≥ 5 s)",
        "",
        "| Mode | identified ωn | ζ | linear with h: ωn | ζ | linear 4-state: ωn | ζ |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, f, e5, e4 in rows:
        if len(f.eigenvalues) == 1:
            lam_f, lam_e = f.eigenvalues[0].real, e5.eigenvalues[0].real
            L.append(f"| {name} | λ = {lam_f:.5f} | | λ = {lam_e:.5f} | | | |")
            continue
        four = f"{e4.wn:.5f} | {e4.zeta:.4f}" if e4 else " | "
        L.append(f"| {name} | {f.wn:.5f} | {f.zeta:.4f} | {e5.wn:.5f} | {e5.zeta:.4f} | {four} |")
    _, f, e5, e4 = rows[0]
    dw, dz = 100 * (e4.wn / f.wn - 1), 100 * (e4.zeta / f.zeta - 1)
    L += [
        "",
        "The nonlinear phugoid matches the linear model only once altitude is a state: as the",
        "aircraft climbs and dives, the density it flies in changes, which the 4-state model",
        f"ignores ({dw:+.1f} % in frequency, {dz:+.1f} % in damping here).",
        "",
        "## Error of the linear model against amplitude",
        "",
        "| doublet [deg] | elevator: max error / peak | rudder: max error / peak |",
        "|---|---|---|",
    ]
    for a, e, r in zip(amps, out["de"], out["dr"], strict=True):
        L.append(f"| {a:g} | {e:.2e} | {r:.2e} |")
    L += [
        "",
        "Relative error ∝ a for the elevator (absolute error ∝ a²) and ∝ a² for the rudder",
        "(absolute ∝ a³): mirror symmetry makes lateral states odd in a lateral input, so the",
        "quadratic term vanishes.",
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "simulation_validation.md").write_text("\n".join(L))


if __name__ == "__main__":
    P.use_style()
    elevator_figure()
    rudder_figure()
    amps, out, sp_eig, dr_eig = amplitude_study()
    amplitude_figure(amps, out, sp_eig, dr_eig)
    write_results(pole_table(), amps, out)
    print((RESULTS / "simulation_validation.md").read_text())
