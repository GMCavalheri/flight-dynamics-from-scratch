"""Classical modes of the Boeing 747 and the Navion against the published roots.

Run with ``uv run python validation/mode_validation.py``. Writes
``validation/results/mode_validation.md`` and ``docs/figures/modes_b747.png``.

Three routes to the modes, each compared with the published transfer-function denominators
(CR-2144 Tables IX-5 and IX-9; Teper Tables X-F and X-G):

1. *table*: the reports' printed dimensional derivatives -> analytic 4x4 models. Checks the
   linear models and the classification. A Monte Carlo over the 3-figure rounding of the
   printed derivatives gives the band within which agreement is all one can ask.
2. *reference*: the nondimensional derivatives -> Appendix A -> analytic models at the
   published reference condition.
3. *trim*: the full nonlinear model trimmed at the reference speed and altitude, then
   linearized numerically.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flightdyn import plotting as P  # noqa: E402
from flightdyn.aero_derivatives import dimensional_derivatives, to_stability_axes  # noqa: E402
from flightdyn.atmosphere import isa  # noqa: E402
from flightdyn.linearization import (  # noqa: E402
    lateral,
    lateral_model,
    linearize,
    longitudinal,
    longitudinal_model,
)
from flightdyn.modal_analysis import approximations, lateral_modes, longitudinal_modes  # noqa: E402
from flightdyn.trim_solver import trim  # noqa: E402
from flightdyn.units import G0_FT  # noqa: E402

from reference import B747_FLIGHT_CONDITIONS, b747, navion  # noqa: E402
from reference_aircraft import b747 as b747_aircraft  # noqa: E402
from reference_aircraft import (  # noqa: E402
    b747_condition,
    b747_table_derivatives,
    navion_condition,
    navion_table_derivatives,
)
from reference_aircraft import navion as navion_aircraft  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
KEYS = (
    "ZETA_PH",
    "OMEGA_PH",
    "ZETA_SP",
    "OMEGA_SP",
    "INV_T_SPIRAL",
    "INV_T_ROLL",
    "ZETA_DR",
    "OMEGA_DR",
)
LABELS = ("ζ ph", "ω ph", "ζ sp", "ω sp", "1/T s", "1/T r", "ζ dr", "ω dr")
NAVION_KEYS = dict(
    zip(
        KEYS,
        (
            "zeta_ph",
            "omega_ph",
            "zeta_sp",
            "omega_sp",
            "inv_T_spiral",
            "inv_T_roll",
            "zeta_dr",
            "omega_dr",
        ),
        strict=True,
    )
)
ROUNDED = ("Xu", "Xw", "Zu", "Zw", "Zq", "Zwdot", "Mu", "Mw", "Mq", "Mwdot", "Yv", "Lb_p", "Lp_p",
           "Lr_p", "Nb_p", "Np_p", "Nr_p")  # fmt: skip


def summary(A_lon, A_lat) -> dict[str, float]:
    lo, la = longitudinal_modes(A_lon), lateral_modes(A_lat)
    return dict(
        ZETA_PH=lo["phugoid"].zeta,
        OMEGA_PH=lo["phugoid"].wn,
        ZETA_SP=lo["short_period"].zeta,
        OMEGA_SP=lo["short_period"].wn,
        INV_T_SPIRAL=-la["spiral"].eigenvalues[0].real,
        INV_T_ROLL=-la["roll"].eigenvalues[0].real,
        ZETA_DR=la["dutch_roll"].zeta,
        OMEGA_DR=la["dutch_roll"].wn,
    )


def table_modes(dd, theta0, thrust=(0.0, 0.0, 0.0)):
    return summary(
        longitudinal_model(dd, theta0, thrust, g=G0_FT).A, lateral_model(dd, theta0, g=G0_FT).A
    )


def _half_unit(x):
    if x == 0:
        return 0.0
    return 0.5 * 10 ** (np.floor(np.log10(abs(x))) - 2)


def rounding_band(dd, theta0, n=400, seed=0):
    """2.5-97.5 % range of each mode quantity when the printed 3-figure derivatives vary
    uniformly within their rounding interval."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in KEYS}
    for _ in range(n):
        ch = {f: getattr(dd, f) + rng.uniform(-1, 1) * _half_unit(getattr(dd, f)) for f in ROUNDED}
        for k, v in table_modes(dataclasses.replace(dd, **ch), theta0).items():
            out[k].append(v)
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def b747_table_rows():
    rows = []
    for fc in B747_FLIGHT_CONDITIONS:
        d = b747(fc)
        th0 = np.radians(d["ALPHA"] + d["GAMMA"])
        dd, thrust = b747_table_derivatives(fc)
        bare, _ = b747_table_derivatives(fc, implied_terms=False)
        rows.append(
            dict(
                fc=fc,
                published={k: d[k] for k in KEYS},
                model=table_modes(dd, th0, thrust),
                bare=table_modes(bare, th0, thrust),
                band=rounding_band(dd, th0),
            )
        )
    return rows


def model_routes(ac, cond):
    rho = float(isa(cond["h"], geopotential=True).rho)
    dd = dimensional_derivatives(ac.aero, ac.geometry, ac.mass, rho, cond["V"])
    th0 = cond["alpha"] + cond["gamma"]
    ref = summary(longitudinal_model(dd, th0).A, lateral_model(dd, th0).A)
    # the classical approximations assume stability axes (W0 = 0)
    st = to_stability_axes(ac)
    dd = dimensional_derivatives(st.aero, st.geometry, st.mass, rho, cond["V"])
    t = trim(ac, cond["V"], cond["h"])
    lm = linearize(ac, t)
    trm = summary(longitudinal(lm).A, lateral(lm, cond["V"]).A)
    return dd, ref, trm, t


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def approximation_rows(dd, CL, CD, exact):
    ph_l = approximations.phugoid_lanchester(dd.V, CL, CD)
    return [
        ("short period ωn", approximations.short_period(dd).wn, exact["OMEGA_SP"]),
        ("short period ζ", approximations.short_period(dd).zeta, exact["ZETA_SP"]),
        ("phugoid ωn, Lanchester √2 g/V", ph_l.wn, exact["OMEGA_PH"]),
        ("phugoid ζ, Lanchester C_D/(√2 C_L)", ph_l.zeta, exact["ZETA_PH"]),
        ("phugoid ωn, 2-state (u, θ)", approximations.phugoid(dd).wn, exact["OMEGA_PH"]),
        ("phugoid ζ, 2-state (u, θ)", approximations.phugoid(dd).zeta, exact["ZETA_PH"]),
        ("roll 1/T = −L'p", -approximations.roll_subsidence(dd).eigenvalues[0].real,
         exact["INV_T_ROLL"]),
        ("spiral 1/T", -approximations.spiral(dd).eigenvalues[0].real, exact["INV_T_SPIRAL"]),
        ("dutch roll ωn, 2-state (β, r)", approximations.dutch_roll(dd).wn, exact["OMEGA_DR"]),
        ("dutch roll ζ, 2-state (β, r)", approximations.dutch_roll(dd).zeta, exact["ZETA_DR"]),
    ]  # fmt: skip


def write_results(rows, nav, routes, approx):
    L = [
        "# Classical modes against the published roots",
        "",
        "Generated by `validation/mode_validation.py`. ζ damping ratio, ω natural frequency",
        "[rad/s], 1/T inverse time constant [1/s] (negative: divergent). ph phugoid, sp short",
        "period, s spiral, r roll subsidence, dr dutch roll.",
        "",
        "## 1. Table route: printed dimensional derivatives → modes",
        "",
        "### Navion (Teper 1969, Tables X-D/X-E → X-F/X-G)",
        "",
        "| | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
        "| Teper | " + " | ".join(f"{nav['published'][k]:.4g}" for k in KEYS) + " |",
        "| model | " + " | ".join(f"{nav['model'][k]:.4g}" for k in KEYS) + " |",
        "| difference | "
        + " | ".join(f"{pct(nav['model'][k], nav['published'][k]):+.2f} %" for k in KEYS)
        + " |",
        "",
        "### Boeing 747 (CR-2144, Tables IX-4/IX-8 → IX-5/IX-9)",
        "",
        "Difference of the model from the published root, in %. In brackets: the band (2.5–97.5 %)",
        "spanned when the printed 3-figure derivatives vary within their rounding.",
        "",
        "| fc | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]  # fmt: skip
    for r in rows:
        cells = []
        for k in KEYS:
            p, m = r["published"][k], r["model"][k]
            lo, hi = (pct(v, p) for v in r["band"][k])
            flag = "" if lo <= 0 <= hi or abs(pct(m, p)) < 0.6 else " †"
            cells.append(f"{pct(m, p):+.1f} [{min(lo, hi):+.1f}, {max(lo, hi):+.1f}]{flag}")
        L.append(f"| {r['fc']} | " + " | ".join(cells) + " |")
    worst = max(
        abs(pct(r["model"][k], r["published"][k]))
        for r in rows
        for k in KEYS
        if not (r["fc"] == 7 and k in ("ZETA_PH", "OMEGA_PH"))
    )
    L += [
        "",
        "† the published value lies outside the rounding band and the difference exceeds 0.6 %.",
        "",
        f"Largest difference outside fc 7's phugoid: {worst:.2f} %. The fc 7 phugoid is nearly",
        "critically damped and its roots move by several percent within the rounding band.",
        "",
        "The model includes three terms the tables do not print, each derived from printed",
        "values (`b747_table_derivatives`): Z_u̇ = −(W0/U0) Z_ẇ, M_u̇ = −(W0/U0) M_ẇ and",
        "X_q = −tan α0 Z_q / cos α0. Without them (CR-2144 Appendix C says the u̇ terms are",
        "neglected) the phugoid damping is low by: "
        + ", ".join(
            f"fc {r['fc']} {pct(r['bare']['ZETA_PH'], r['published']['ZETA_PH']):+.1f} %"
            for r in rows
        )
        + ". The report's numbers were evidently computed with them.",
        "",
        "## 2–3. Reference and trim routes: nondimensional data and the nonlinear model",
        "",
        "*reference*: Appendix A at the published condition. *trim*: the nonlinear model trimmed",
        "at the published speed and altitude, linearized numerically.",
        "",
        "| aircraft | route | " + " | ".join(LABELS) + " |",
        "|---|---|" + "---|" * len(LABELS),
    ]  # fmt: skip
    for name, pub, ref, trm, t, a0 in routes:
        for route, m in (("reference", ref), ("trim", trm)):
            cells = " | ".join(f"{pct(m[k], pub[k]):+.1f} %" for k in KEYS)
            L.append(f"| {name} | {route} | {cells} |")
        L.append(f"| | trim α − α0 = {np.degrees(t.alpha - a0):+.2f}° | |" + " |" * (len(KEYS) - 1))
    L += [
        "",
        "The reference route reproduces the 747 to the accuracy of the table route: the",
        "nondimensional data, the Appendix A formulas and the model's α̇ and q terms are",
        "consistent. The trim route differs in the phugoid and dutch-roll damping because the",
        "model trims below α0: the tabulated C_L is W/(qS) and omits the lift of the inclined",
        "thrust, so the model needs less wing lift. The Navion's ~1–2 % offsets in both routes are",
        "Teper's S = 184 ft² in the dimensional table (derivative validation).",
        "",
        "## Approximations against the exact 4-state roots (reference route, stability axes)",
        "",
        "| Approximation | 747 fc 2 | exact | difference | Navion | exact | difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for (lab, a1, e1), (_, a2, e2) in zip(*approx, strict=True):
        L.append(
            f"| {lab} | {a1:.4g} | {e1:.4g} | {pct(a1, e1):+.1f} % | {a2:.4g} | {e2:.4g} | "
            f"{pct(a2, e2):+.1f} % |"
        )
    L.append("")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mode_validation.md").write_text("\n".join(L))


def roots(zeta, wn):
    return complex(-zeta * wn, wn * np.sqrt(max(1 - zeta * zeta, 0.0)))


def figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    panels = [
        (axes[0, 0], "ZETA_SP", "OMEGA_SP", "Short period"),
        (axes[0, 1], "ZETA_PH", "OMEGA_PH", "Phugoid"),
        (axes[1, 0], "ZETA_DR", "OMEGA_DR", "Dutch roll"),
    ]
    for ax, kz, kw, title in panels:
        for r in rows:
            p = roots(r["published"][kz], r["published"][kw])
            m = roots(r["model"][kz], r["model"][kw])
            ax.plot(p.real, p.imag, "o", ms=11, mfc="none", color=P.REFERENCE, lw=1)
            ax.plot(m.real, m.imag, "o", ms=4, color=P.SERIES[0])
            ax.annotate(str(r["fc"]), (p.real, p.imag), xytext=(7, 4), textcoords="offset points",
                        fontsize=8, color=P.TEXT_MUTED)  # fmt: skip
        ax.set_xlabel("real part [1/s]")
        ax.set_ylabel("imaginary part [rad/s]")
        ax.set_title(f"{title}, upper root")
    ax = axes[1, 1]
    fcs = [r["fc"] for r in rows]
    for k, lab, color in (("INV_T_ROLL", "roll subsidence", P.SERIES[1]),
                          ("INV_T_SPIRAL", "spiral", P.SERIES[2])):  # fmt: skip
        ax.plot(fcs, [-r["published"][k] for r in rows], "o", ms=11, mfc="none", color=P.REFERENCE)
        ax.plot(fcs, [-r["model"][k] for r in rows], "o", ms=4, color=color, label=lab)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(0, color=P.TEXT_MUTED, lw=0.8)
    ax.set_xlabel("flight condition")
    ax.set_ylabel("real root −1/T [1/s] (symlog)")
    ax.set_title("Roll and spiral")
    ax.set_xticks(fcs)
    ax.legend(loc="upper left")
    axes[0, 0].plot([], [], "o", ms=11, mfc="none", color=P.REFERENCE, label="CR-2144")
    axes[0, 0].plot([], [], "o", ms=4, color=P.SERIES[0], label="model")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle(
        "Boeing 747, ten flight conditions: model roots from the printed derivatives vs CR-2144",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )  # fmt: skip
    fig.tight_layout()
    P.save(fig, "modes_b747.png")


def navion_table():
    n = navion()
    return dict(
        published={k: n[v] for k, v in NAVION_KEYS.items()},
        model=table_modes(navion_table_derivatives(), 0.0),
    )


if __name__ == "__main__":
    P.use_style()
    rows, nav = b747_table_rows(), navion_table()
    routes, approx = [], []
    for name, ac, cond, pub in (
        ("747 fc 1", b747_aircraft(1), b747_condition(1), {k: b747(1)[k] for k in KEYS}),
        ("747 fc 2", b747_aircraft(2), b747_condition(2), {k: b747(2)[k] for k in KEYS}),
        ("Navion", navion_aircraft(), navion_condition(), nav["published"]),
    ):
        dd, ref, trm, t = model_routes(ac, cond)
        routes.append((name, pub, ref, trm, t, cond["alpha"]))
        if name != "747 fc 1":
            approx.append(approximation_rows(dd, ac.aero.CL0, ac.aero.CD0, ref))
    write_results(rows, nav, routes, approx)
    figure(rows)
    print((RESULTS / "mode_validation.md").read_text())
