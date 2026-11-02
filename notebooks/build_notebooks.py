"""Generate and execute the demo notebooks from the cell sources below.

Run with ``uv run python notebooks/build_notebooks.py``. Keeping the sources here (instead of
hand-editing .ipynb JSON) makes the notebooks reviewable in diffs and reproducible: rebuilds are
byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """\
import sys
import numpy as np
import matplotlib.pyplot as plt
from flightdyn import plotting as P

P.use_style()
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, "../validation")  # cited reference data and reference aircraft
from reference_aircraft import b747, b747_condition, navion, navion_condition"""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_equations_of_motion.ipynb": [
        (
            "md",
            """\
# 1 · The equations of motion of a rigid aircraft

The twelve-state rigid-body model, checked against motions with closed-form answers.
Lesson: [01](../docs/lessons/01-equations-of-motion.md) ·
Theory: [equations of motion](../docs/equations_of_motion.md).""",
        ),
        (
            "code",
            SETUP
            + """
from scipy.integrate import solve_ivp
from flightdyn.aircraft import MassProperties
from flightdyn import equations_of_motion as eom
from flightdyn.rotations import dcm_body_from_ned""",
        ),
        (
            "md",
            """\
## Torque-free precession

An axisymmetric body ($I_{xx} = I_{yy} = I$, $I_{zz} = I_3$) spinning at r keeps r constant while
(p, q) rotates at $\\Omega = (I_3 - I)r/I$.""",
        ),
        (
            "code",
            """\
mp = MassProperties(mass=1.0, Ixx=2.0, Iyy=2.0, Izz=5.0)
x0 = np.zeros(12)
x0[eom.P], x0[eom.R] = 0.4, 1.5
t = np.linspace(0, 8, 200)
sol = solve_ivp(lambda _, x: eom.rigid_body_derivative(x, (0, 0, 0), (0, 0, 0), mp, 0.0),
                (0, 8), x0, t_eval=t, method="DOP853", rtol=1e-10, atol=1e-10)
omega = (5.0 - 2.0) * 1.5 / 2.0
print(f"precession rate {omega} rad/s; max error in p: "
      f"{np.abs(sol.y[eom.P] - 0.4 * np.cos(omega * t)).max():.1e}")
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t, sol.y[eom.P], label="p")
ax.plot(t, sol.y[eom.Q], label="q")
ax.set_xlabel("time [s]")
ax.legend();""",
        ),
        (
            "md",
            """\
## A tumbling body still falls along a parabola

Gravity alone acts on the centre of gravity, whatever the body does about it.""",
        ),
        (
            "code",
            """\
mp = MassProperties(mass=1200.0, Ixx=1400.0, Iyy=4000.0, Izz=4800.0, Ixz=250.0)
x0 = np.zeros(12)
x0[[0, 1, 2]] = (30.0, -4.0, 6.0)
x0[[3, 4, 5]] = (1.2, -0.7, 0.4)
x0[[6, 7, 8]] = (0.3, 0.2, -0.5)
v_ned = dcm_body_from_ned(0.3, 0.2, -0.5).T @ x0[:3]
t = np.linspace(0, 3, 13)
sol = solve_ivp(lambda _, x: eom.rigid_body_derivative(x, (0, 0, 0), (0, 0, 0), mp),
                (0, 3), x0, t_eval=t, method="DOP853", rtol=1e-11, atol=1e-11)
parabola = v_ned[2] * t + 0.5 * 9.80665 * t**2
print(f"largest departure from the parabola: {np.abs(sol.y[eom.ZD] - parabola).max():.1e} m")""",
        ),
    ],
    "02_aero_derivatives.ipynb": [
        (
            "md",
            """\
# 2 · Aerodynamic derivatives

The 747's nondimensional derivatives through the CR-2144 formulas, against the report's own
dimensional derivatives. Lesson: [02](../docs/lessons/02-aerodynamic-derivatives.md) ·
Theory: [aerodynamic derivatives](../docs/aero_derivatives.md).""",
        ),
        (
            "code",
            SETUP
            + """
from flightdyn.aero_derivatives import body_axis_coefficients, dimensional_derivatives
from flightdyn.atmosphere import isa
from derivative_validation import b747_comparison, navion_comparison""",
        ),
        (
            "code",
            """\
ac, c = b747(2), b747_condition(2)
rho = float(isa(c["h"], geopotential=True).rho)
dd = dimensional_derivatives(ac.aero, ac.geometry, ac.mass, rho, c["V"])
print(f"power approach: Zw = {dd.Zw:.3f} 1/s, Mq = {dd.Mq:.3f} 1/s, "
      f"L'b = {dd.Lb_p:.3f} 1/s^2, N'b = {dd.Nb_p:.3f} 1/s^2")
print(f"Cm0 balancing the thrust moment: {ac.aero.Cm0:.4f}")""",
        ),
        (
            "md",
            "## Model against CR-2144, both readings of the lateral axis system",
        ),
        (
            "code",
            """\
print(f"{'':6s}{'fc':>3s}{'CR-2144':>11s}{'model':>11s}{'diff %':>8s}{'body-axis %':>13s}")
for r in b747_comparison():
    body = f"{100 * (r['body'] / r['ref'] - 1):+13.1f}" if r["body"] is not None else ""
    print(f"{r['key']:6s}{r['fc']:3d}{r['ref']:11.4g}{r['model']:11.4g}"
          f"{100 * (r['model'] / r['ref'] - 1):+8.2f}{body}")""",
        ),
        ("md", "## The Navion's dimensional table used S = 184 ft²"),
        (
            "code",
            """\
ratios = [r["ref"] / r["model"] for r in navion_comparison()]
print(f"Teper / model: {min(ratios):.4f} to {max(ratios):.4f};  184/180 = {184 / 180:.4f}")""",
        ),
    ],
    "03_trim.ipynb": [
        (
            "md",
            """\
# 3 · Trim

Level flight, climbs and coordinated turns of the 747 model.
Lesson: [03](../docs/lessons/03-trim.md) · Theory: [trim](../docs/trim.md).""",
        ),
        (
            "code",
            SETUP
            + """
from flightdyn.trim_solver import trim, trim_guess
from flightdyn.units import KT, LBF""",
        ),
        (
            "code",
            """\
ac, c = b747(2), b747_condition(2)
t = trim(ac, c["V"], c["h"])
print(f"alpha {np.degrees(t.alpha):.2f} deg (published {np.degrees(c['alpha']):.2f}), "
      f"elevator {np.degrees(t.de):.2f} deg, thrust {t.thrust / LBF:.0f} lbf, "
      f"residual {t.residual:.1e}")
t = trim(ac, c["V"], c["h"], gamma=np.radians(3.0))
print(f"3 deg climb: alpha {np.degrees(t.alpha):.2f} deg, thrust {t.thrust / LBF:.0f} lbf")""",
        ),
        ("md", "## Trim curve: elevator against speed (speed stability)"),
        (
            "code",
            """\
kt = np.linspace(140, 200, 13)
guess, rows = None, []
for v in kt:
    t = trim(ac, v * KT, 0.0, guess=guess)
    guess = trim_guess(t, ac.weight)
    rows.append((np.degrees(t.alpha), np.degrees(t.de)))
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(kt, rows[:, 1])
ax.set_xlabel("true airspeed [kt]")
ax.set_ylabel("trim elevator [deg]");""",
        ),
        ("md", "## Coordinated level turns"),
        (
            "code",
            """\
for rate in (1.0, 2.0, 3.0):
    t = trim(ac, c["V"], 0.0, turn_rate=np.radians(rate))
    print(f"{rate:.0f} deg/s: bank {np.degrees(t.phi):5.1f} deg, n = {1 / np.cos(t.phi):.3f}, "
          f"elevator {np.degrees(t.de):+.2f} deg, rudder {np.degrees(t.controls[2]):+.2f} deg")""",
        ),
    ],
    "04_linearization_and_modes.ipynb": [
        (
            "md",
            """\
# 4 · Linearization and the classical modes

Numerical and analytic linear models, the five modes, and the ten published 747 conditions.
Lesson: [04](../docs/lessons/04-linearization-and-modes.md) ·
Theory: [linearization and modes](../docs/linearization_and_modes.md).""",
        ),
        (
            "code",
            SETUP
            + """
from flightdyn.linearization import analytic_model, lateral, linearize, longitudinal
from flightdyn.modal_analysis import lateral_modes, longitudinal_modes
from flightdyn.trim_solver import trim
from mode_validation import b747_table_rows, KEYS""",
        ),
        (
            "code",
            """\
ac, c = b747(2), b747_condition(2)
t = trim(ac, c["V"], c["h"])
lm = linearize(ac, t)
lon, lat = analytic_model(ac, t)
print("numerical vs analytic, largest difference:",
      f"{np.abs(longitudinal(lm).A - lon.A).max():.1e},",
      f"{np.abs(lateral(lm, c['V']).A - lat.A).max():.1e}")
modes = longitudinal_modes(lon.A) | lateral_modes(lat.A)
for name, m in modes.items():
    if m.is_oscillatory:
        print(f"{name:13s} wn {m.wn:.3f} rad/s  zeta {m.zeta:.3f}  period {m.period:.1f} s")
    else:
        print(f"{name:13s} T {m.time_constant:.2f} s")""",
        ),
        ("md", "## The 747 at ten flight conditions, from the report's own derivatives"),
        (
            "code",
            """\
rows = b747_table_rows()
print("fc  " + "  ".join(f"{k:>12s}" for k in KEYS))
for r in rows:
    print(f"{r['fc']:2d}  " + "  ".join(
        f"{100 * (r['model'][k] / r['published'][k] - 1):+11.1f}%" for k in KEYS))""",
        ),
    ],
    "05_nonlinear_simulation.ipynb": [
        (
            "md",
            """\
# 5 · Nonlinear simulation

Doublets on the nonlinear 747, the linear prediction, and the modes read back from the time
history. Lesson: [05](../docs/lessons/05-nonlinear-simulation.md) ·
Theory: [nonlinear simulation](../docs/nonlinear_simulation.md).""",
        ),
        (
            "code",
            SETUP
            + """
from flightdyn.linearization import linearize
from flightdyn.nonlinear_sim import Doublet, damped_pairs, matrix_pencil, simulate, simulate_linear
from flightdyn.trim_solver import trim""",
        ),
        (
            "code",
            """\
ac, c = b747(2), b747_condition(2)
t = trim(ac, c["V"], c["h"])
lm = linearize(ac, t)
LAT = [1, 3, 5, 6]
doublet = [Doublet("dr", np.radians(3.0), 1.0, 2.0)]
nl = simulate(ac, t.x, t.controls, doublet, t_end=40.0, dt=0.05)
lin = simulate_linear(lm.submodel(LAT, [1, 2]), doublet, t_end=40.0, dt=0.05,
                      control_index=(1, 2))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(nl.t, np.degrees(nl.x[:, 6] - t.x[6]), label="nonlinear φ")
ax.plot(lin.t, np.degrees(lin.x[:, 3]), "--", color="k", lw=1, label="linear φ")
ax.set_xlabel("time [s]")
ax.set_ylabel("bank angle [deg]")
ax.legend();""",
        ),
        (
            "md",
            """\
## Modes of the nonlinear response (matrix pencil)

A small doublet keeps the response linear, so the identified poles should be the linear
eigenvalues; at 3° the roll root already moves by a few percent (Lesson 5, section 4).""",
        ),
        (
            "code",
            """\
small = simulate(ac, t.x, t.controls, [Doublet("dr", np.radians(0.25), 1.0, 2.0)],
                 t_end=60.0, dt=0.05)
free = small.t >= 5.0
found = matrix_pencil(small.x[free][:, LAT] - t.x[LAT], 0.05, 4)
eig = np.linalg.eigvals(lm.A[np.ix_(LAT, LAT)])
print("identified:", np.round(np.sort_complex(found), 4))
print("linear:    ", np.round(np.sort_complex(eig), 4))""",
        ),
    ],
    "06_trade_studies.ipynb": [
        (
            "md",
            """\
# 6 · Stability trade studies

Neutral point, the CG's effect on the short period, and the lateral spiral/dutch-roll trade.
Lesson: [06](../docs/lessons/06-stability-trade-studies.md) ·
Theory: [trade studies](../docs/trade_studies.md).""",
        ),
        (
            "code",
            SETUP
            + """
from flightdyn.trade_studies import lateral_map, longitudinal_point, neutral_point_shift""",
        ),
        (
            "code",
            """\
ac, c = b747(2), b747_condition(2)
hn = neutral_point_shift(ac)
print(f"neutral point {0.25 + hn:.3f} MAC, static margin {100 * hn:.1f} % MAC")
for h in (0.10, 0.20, 0.30, 0.40):
    p = longitudinal_point(ac, c["V"], c["h"], h - 0.25)
    sp = p.modes["short_period"]
    print(f"CG {h:.2f}: margin {100 * p.static_margin:5.1f} %, short period "
          f"wn {sp.wn:.3f} rad/s, zeta {sp.zeta:.3f}, trim elevator {np.degrees(p.de_trim):+.1f} deg")""",
        ),
        ("md", "## Spiral and dutch roll against dihedral effect and directional stiffness"),
        (
            "code",
            """\
k_lb, k_nb = np.linspace(0.1, 2.5, 25), np.linspace(0.2, 3.0, 29)
spiral, zeta, crit = lateral_map(ac, c["V"], c["h"], k_lb, k_nb)
fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contour(k_lb, k_nb, zeta, levels=[0.0, 0.1, 0.2, 0.3])
ax.clabel(cs, fmt="ζ=%.1f")
ax.contour(k_lb, k_nb, crit, levels=[0.0], colors="k")
ax.plot(1, 1, "ko")
ax.set_xlabel("C_lβ scale")
ax.set_ylabel("C_nβ scale");""",
        ),
    ],
}


def _merge_streams(outputs):
    """Join consecutive stdout/stderr chunks; how the kernel splits them depends on timing."""
    merged = []
    for out in outputs:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and out.get("output_type") == "stream"
            and prev.get("output_type") == "stream"
            and prev.get("name") == out.get("name")
        ):
            prev["text"] += out["text"]
        else:
            merged.append(out)
    return merged


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:02d}"  # stable ids keep rebuilds diff-free
    NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    for cell in nb.cells:
        cell.get("outputs", [])[:] = _merge_streams(cell.get("outputs", []))
    path = HERE / name
    nbformat.write(nb, path)
    return path


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        print("built", build(name, cells).name)


if __name__ == "__main__":
    main()
