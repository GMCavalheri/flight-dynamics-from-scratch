# CLAUDE.md

From-scratch aircraft flight dynamics (6-DOF rigid-body equations of motion → stability-derivative
aerodynamics → trim → linearization and modal analysis → nonlinear simulation → trade studies).
Plan: `AERO-04-flight-dynamics-from-scratch-PLAN.md` (tick the phase checkboxes as phases land).

## Commands
- `uv sync` — install deps (numpy, scipy, matplotlib; dev: pytest, ruff, nbclient)
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format --check .` — lint/format (CI runs both). Ruff also
  formats the ```python blocks inside Markdown, so run it after editing docs too
- `uv run python validation/run_all.py` — regenerate every validation figure (`docs/figures/`)
  and results table (`validation/results/`); single scripts also run on their own
- `uv run python notebooks/build_notebooks.py` — regenerate and execute the notebooks (edit cell
  sources there, never the .ipynb JSON); rebuilds are byte-identical
- `uv run python docs/lessons/make_figures.py` — regenerate the lesson figures

## Layout
- `src/flightdyn/` — single package; subpackages mirror the plan: `equations_of_motion`,
  `aero_derivatives`, `trim_solver`, `linearization`, `modal_analysis`, `nonlinear_sim`,
  `trade_studies`; shared `atmosphere.py`, `units.py`, `rotations.py`, `aircraft.py`,
  `plotting.py`.
- `tests/` — pytest, one file per module. `validation/` — scripts + cited reference data.
- `docs/` — theory note per module, validation report, figures; `docs/lessons/` is the lesson
  course, one lesson per phase (keep its numbers in sync with the code). `notebooks/` — demos.

## Conventions
- **SI units internally** (m, s, kg, N, rad). Helpers that take or return other units say so
  in the name (`*_ft`, `*_kt`, `*_deg`, `*_slug_ft2`). Reference data in imperial units is
  converted once, on load, with the exact factors in `units.py`.
- Axes: body axes x forward, y right wing, z down; Earth axes North-East-Down on a flat,
  non-rotating Earth. Euler angles are the 3-2-1 sequence (ψ, θ, φ). Stability axes are body
  axes rotated by the trim angle of attack about y.
- State vector order (12 states): `u v w p q r phi theta psi x_n y_e z_d`; controls
  `de da dr thrust` (rad, rad, rad, N), deflections measured from the reference (trim) setting
  of the derivative model. Signs follow the reference data (CR-2144, Teper): positive elevator
  gives a nose-down moment (Cm_de < 0), positive aileron a right-wing-down rolling moment
  (Cl_da > 0), positive rudder a nose-left yawing moment (Cn_dr < 0).
- Altitude `h = -z_d` is geopotential (gravity is the constant g0). Reference data altitudes
  are pressure altitudes, i.e. geopotential in the standard atmosphere.
- Derivative names follow NASA CR-2144 / Etkin: nondimensional `CL_a`, `Cm_q`, `Cn_b`
  (stability-axis lift/drag; lateral moments in stability or body axes, `lateral_axes`);
  dimensional (per unit mass or inertia, SI) `Xu`, `Zw`, `Mq`, `Lb`, `Nr`; primed lateral
  derivatives `Lp_p` include the Ixz coupling.
- `validation/reference_aircraft.py` builds the Boeing 747 (CR-2144 fc 1 and 2) and the Navion
  (Teper 1969) from the cited CSVs; tests import it through `tests/conftest.py`.
- Pure NumPy/SciPy — no flight-dynamics libraries (no JSBSim, python-control, etc.).

## Docs (GitHub-rendered math)
- Display equations go in ```` ```math ```` fences (no Markdown processing inside).
- Inline `$...$` is processed as Markdown first: no `\,` or `\;` (they become `,` `;`), no `\\`
  row breaks (put matrices in a display block), and no `_` right after `}` `)` `]` (it starts
  emphasis; write `C_{m_{\alpha}}`, not `{C_m}_\alpha`). `tests/test_docs.py` guards this.

## Testing rules
- Every test asserts against a closed-form result or a cited published value; name the source
  in a comment. Never loosen a tolerance to make a failing model pass without explaining why.
- Cite textbooks at chapter level unless the section/table number has been checked in the book.
- Reference data in `validation/reference_data/` must carry its citation (document, table,
  page). Do not invent data: leave a value out if it cannot be checked against its source.
- Assumed inputs (thrust line, engine model, CG shifts in trade studies) are labelled as
  assumptions, never presented as aircraft data.
- No employer references in the repo; name aircraft types only.

## Workflow
- Commit small logical changes directly to `main`, push after
  `uv run ruff check . && uv run ruff format --check . && uv run pytest` pass (no pipes: a pipe
  hides the exit code). Stage explicit paths rather than `git add -A`.
- Each phase lands as code + tests, then its lesson in `docs/lessons/` as a separate commit.
