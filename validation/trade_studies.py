"""Stability and control trade studies on the Boeing 747 model.

Run with ``uv run python validation/trade_studies.py``. Writes
``validation/results/trade_studies.md`` and ``docs/figures/trade_cg_sweep.png``,
``trade_scissors.png``, ``trade_lateral_map.png``.

Assumptions (labelled, not 747 data): the horizontal tail arm l_t = 3.3 cbar (the tail is
calibrated from the published Cm_q, Cm_adot and Cm_de, HorizontalTail.calibrate); an elevator
limit of -20 deg for the forward-CG trim limit (the 747 trims with a movable stabilizer, so
this limit is illustrative); a minimum static margin of 5 % MAC; inertias unchanged by CG
moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flightdyn import plotting as P  # noqa: E402
from flightdyn.trade_studies import (  # noqa: E402
    HorizontalTail,
    divergence_limit,
    lateral_map,
    longitudinal_point,
    neutral_point_shift,
    scissors,
)

from reference_aircraft import b747, b747_condition  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LT_C = 3.3  # assumed tail arm [mean chords]
DE_LIMIT = np.radians(-20.0)  # assumed elevator limit for the forward CG limit
MARGIN = 0.05  # minimum static margin [MAC]
H_REF = 0.25  # CR-2144 Table IX-3 CG [MAC]


def cg_sweep():
    ac, c = b747(2), b747_condition(2)
    hn = neutral_point_shift(ac)
    dhs = np.linspace(-0.2, hn - 0.005, 40)
    pts = [longitudinal_point(ac, c["V"], c["h"], dh) for dh in dhs]
    return ac, c, pts, hn


def cg_figure(pts, hn, dyn):
    h = np.array([H_REF + p.dh for p in pts])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax = axes[0, 0]
    ax.plot(h, [100 * p.static_margin for p in pts], color=P.SERIES[0])
    ax.set_ylabel("static margin [% MAC]")
    ax.set_title("Static margin and trim")
    ax2 = ax.twinx()
    ax2.plot(h, [np.degrees(p.de_trim) for p in pts], color=P.SERIES[1])
    ax2.set_ylabel("trim elevator δe [deg]", color=P.SERIES[1])
    ax2.grid(False)
    ax = axes[0, 1]
    sp = [(H_REF + p.dh, p.modes["short_period"]) for p in pts if p.modes]
    ax.plot([a for a, _ in sp], [m.wn for _, m in sp], color=P.SERIES[0], label="ωn [rad/s]")
    ax.plot([a for a, _ in sp], [m.zeta for _, m in sp], color=P.SERIES[2], label="ζ")
    ax.set_title("Short period")
    ax.legend()
    ax = axes[1, 0]
    ph = [(H_REF + p.dh, p.modes["phugoid"]) for p in pts if p.modes]
    ax.plot([a for a, _ in ph], [m.zeta for _, m in ph], color=P.SERIES[1])
    ax.set_ylabel("phugoid ζ")
    ax.set_title("Phugoid damping")
    ax = axes[1, 1]
    ev = np.array([np.sort(p.eigenvalues.real) for p in pts])
    for k in range(4):
        ax.plot(h, ev[:, k], ".", ms=3, color=P.SERIES[k % 4])
    ax.axhline(0, color=P.TEXT_MUTED, lw=0.8)
    ax.set_ylim(-1.0, 0.1)
    ax.set_ylabel("real parts of the 4 roots [1/s]")
    ax.set_title("Longitudinal roots")
    for ax in axes.flat:
        ax.axvline(H_REF + hn, color=P.REFERENCE, ls="--", lw=1)
        ax.axvline(H_REF + dyn, color=P.SERIES[3], ls=":", lw=1.5)
        ax.axvline(H_REF, color=P.GRID, lw=3, zorder=0)
    for ax in axes[1]:
        ax.set_xlabel("CG position h [fraction of MAC]")
    axes[0, 0].annotate(
        "neutral point",
        (H_REF + hn, 0.9),
        xycoords=("data", "axes fraction"),
        ha="right",
        fontsize=8,
        xytext=(-4, 0),
        textcoords="offset points",
    )
    axes[0, 0].annotate(
        "aperiodic divergence",
        (H_REF + dyn, 0.75),
        xycoords=("data", "axes fraction"),
        ha="right",
        fontsize=8,
        color=P.SERIES[3],
        xytext=(-4, 0),
        textcoords="offset points",
    )
    fig.suptitle("747 power approach: moving the centre of gravity (CR-2144 reference: 0.25 MAC)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    P.save(fig, "trade_cg_sweep.png")


def scissors_rows():
    ac1, c1, ac2, c2 = b747(1), b747_condition(1), b747(2), b747_condition(2)
    t1, t2 = HorizontalTail.calibrate(ac1, LT_C), HorizontalTail.calibrate(ac2, LT_C)
    ks = np.linspace(0.6, 1.5, 10)
    return t1, t2, scissors(ac2, c2, ac1, c1, t2, t1, ks, DE_LIMIT, MARGIN)


def scissors_figure(rows):
    k = np.array([r["k"] for r in rows])
    g = {name: H_REF + np.array([r[name] for r in rows])
         for name in ("neutral", "static_limit", "divergence", "forward_limit")}  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_betweenx(k, g["forward_limit"], np.minimum(g["static_limit"], g["divergence"]),
                     where=g["forward_limit"] < np.minimum(g["static_limit"], g["divergence"]),
                     color=P.SERIES[0], alpha=0.12, lw=0, label="usable CG range")  # fmt: skip
    ax.plot(g["forward_limit"], k, color=P.SERIES[1], label="forward: trim at landing, δe = −20°")
    ax.plot(g["static_limit"], k, color=P.SERIES[0], label="aft: static margin 5 %")
    ax.plot(g["neutral"], k, "--", color=P.REFERENCE, lw=1, label="neutral point")
    ax.plot(g["divergence"], k, ":", color=P.SERIES[3], lw=2, label="aperiodic divergence")
    ax.plot(H_REF, 1.0, "o", color=P.REFERENCE)
    ax.annotate("747 data: tail as built, CG 0.25", (H_REF, 1.0), xytext=(6, -12),
                textcoords="offset points", fontsize=8)  # fmt: skip
    ax.set_xlabel("CG position h [fraction of MAC]")
    ax.set_ylabel("horizontal tail area / 747 tail area")
    ax.set_title("Tail sizing: the scissors diagram")
    ax.legend(loc="lower right", fontsize=8)
    P.save(fig, "trade_scissors.png")


def lateral_study():
    ac, c = b747(2), b747_condition(2)
    k_lb = np.linspace(0.1, 2.5, 49)
    k_nb = np.linspace(0.2, 3.0, 57)
    spiral, zeta, crit = lateral_map(ac, c["V"], c["h"], k_lb, k_nb)
    return k_lb, k_nb, spiral, zeta, crit


def lateral_figure(k_lb, k_nb, spiral, zeta, crit):
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(k_lb, k_nb, zeta, levels=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
                    colors=[P.SERIES[2]], linewidths=1)  # fmt: skip
    ax.clabel(cs, fmt="ζ = %.2f", fontsize=8)
    ax.contourf(k_lb, k_nb, crit, levels=[crit.min() - 1, 0.0], colors=[P.SERIES[1]], alpha=0.15)
    ax.contourf(k_lb, k_nb, zeta, levels=[np.nanmin(zeta) - 1, 0.0], colors=[P.SERIES[3]],
                alpha=0.15)  # fmt: skip
    ax.contour(k_lb, k_nb, crit, levels=[0.0], colors=[P.SERIES[1]], linewidths=2)
    ax.contour(k_lb, k_nb, spiral, levels=[0.0], colors=[P.REFERENCE], linewidths=1,
               linestyles="--")  # fmt: skip
    ax.plot(1, 1, "o", color=P.REFERENCE)
    ax.annotate("747 power approach", (1, 1), xytext=(6, 6), textcoords="offset points",
                fontsize=8)  # fmt: skip
    ax.text(0.25, 2.7, "spiral divergent", color=P.SERIES[1], fontsize=9)
    ax.text(1.55, 2.3, "spiral stable,\ndutch roll stable", color=P.TEXT_MUTED, fontsize=9)
    ax.text(1.95, 0.3, "dutch roll\nunstable", color=P.SERIES[3], fontsize=9)
    ax.plot([], [], color=P.SERIES[1], lw=2, label="spiral criterion L'β N'r = N'β L'r")
    ax.plot([], [], "--", color=P.REFERENCE, lw=1, label="spiral root = 0 (eigenvalues)")
    ax.plot([], [], color=P.SERIES[2], lw=1, label="dutch-roll damping ζ")
    ax.set_xlabel("dihedral effect: C_lβ / C_lβ (747)")
    ax.set_ylabel("directional stiffness: C_nβ / C_nβ (747)")
    ax.set_title("The lateral design trade: spiral against dutch roll")
    ax.legend(loc="upper right", fontsize=8)
    P.save(fig, "trade_lateral_map.png")


def write_results(ac, pts, hn, dyn, t1, t2, rows, lat):
    k_lb, k_nb, spiral, zeta, crit = lat
    ref = next(p for p in pts if abs(p.dh) == min(abs(q.dh) for q in pts))
    L = [
        "# Stability and control trade studies (Boeing 747 model)",
        "",
        "Generated by `validation/trade_studies.py`. Assumptions are listed in its docstring:",
        f"tail arm l_t = {LT_C} c̄, elevator limit {np.degrees(DE_LIMIT):.0f}° (illustrative),",
        f"minimum static margin {100 * MARGIN:.0f} % MAC, inertias unchanged by CG moves.",
        "",
        "## CG sweep (power approach, CR-2144 fc 2)",
        "",
        f"* Stick-fixed neutral point: {H_REF + hn:.3f} MAC; static margin at the published CG",
        f"  ({H_REF} MAC): {100 * hn:.1f} % MAC.",
        f"* Aperiodic divergence (a real root crosses zero) at {H_REF + dyn:.3f} MAC, "
        f"{100 * (hn - dyn):.1f} % MAC ahead of the neutral point: the positive M_u of the",
        "  thrust-moment trim makes speed stability fail before static stability does.",
        "* The phugoid is weakly damped everywhere (ζ ≈ 0.02 at the published CG) and crosses",
        "  ζ = 0 at some CG positions for small tails, as a very slow oscillatory divergence;",
        "  that is a handling-qualities question, not used as a CG limit here.",
        f"* Short period at the published CG: ωn = {ref.modes['short_period'].wn:.3f} rad/s, "
        f"ζ = {ref.modes['short_period'].zeta:.3f}.",
        "",
        "## Horizontal tail calibrated from the published derivatives",
        "",
        "| | fc 1 (landing) | fc 2 (power approach) |",
        "|---|---|---|",
        f"| η a_t V_H | {t1.eta_at_VH:.3f} | {t2.eta_at_VH:.3f} |",
        f"| downwash gradient ε_α = Cm_α̇ / Cm_q | {t1.eps_a:.3f} | {t2.eps_a:.3f} |",
        f"| elevator effectiveness τ | {t1.tau:.3f} | {t2.tau:.3f} |",
        f"| tail share of C_Lα | {t1.contributions()['CL_a']:.3f} | "
        f"{t2.contributions()['CL_a']:.3f} |",
        f"| tail share of C_mα | {t1.contributions()['Cm_a']:.3f} | "
        f"{t2.contributions()['Cm_a']:.3f} |",
        "",
        "## Scissors diagram: CG limits [MAC] against tail size",
        "",
        "| tail / 747 tail | forward limit | aft limit (5 % margin) | neutral point "
        "| aperiodic divergence | usable range |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        aft = min(r["static_limit"], r["divergence"])
        span = aft - r["forward_limit"]
        L.append(
            f"| {r['k']:.1f} | {H_REF + r['forward_limit']:.3f} | {H_REF + r['static_limit']:.3f}"
            f" | {H_REF + r['neutral']:.3f} | {H_REF + r['divergence']:.3f} | "
            f"{max(span, 0) * 100:.1f} % |"
        )
    i1 = np.argmin(np.abs(k_nb - 1.0))
    j_cross = np.where(np.diff(np.sign(crit[i1])))[0]
    kl_cross = k_lb[j_cross[0]] if len(j_cross) else float("nan")
    agree = np.nanmax(np.abs(np.sign(spiral) + np.sign(crit))[np.isfinite(spiral)]) == 0
    L += [
        "",
        "## Lateral map (power approach, stability axes)",
        "",
        f"* The spiral root changes sign exactly where L'β N'r − N'β L'r does: "
        f"{'every' if agree else 'not every'} grid point agrees.",
        f"* At the 747's directional stiffness, the spiral becomes divergent below "
        f"{kl_cross:.2f} × its dihedral effect.",
        f"* Dutch-roll damping at the nominal point: ζ = "
        f"{zeta[i1, np.argmin(np.abs(k_lb - 1.0))]:.3f}.",
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "trade_studies.md").write_text("\n".join(L))


if __name__ == "__main__":
    P.use_style()
    ac, c, pts, hn = cg_sweep()
    dyn = divergence_limit(ac, c["V"], c["h"], bracket=(hn - 0.15, hn))
    cg_figure(pts, hn, dyn)
    t1, t2, rows = scissors_rows()
    scissors_figure(rows)
    lat = lateral_study()
    lateral_figure(*lat)
    write_results(ac, pts, hn, dyn, t1, t2, rows, lat)
    print((RESULTS / "trade_studies.md").read_text())
