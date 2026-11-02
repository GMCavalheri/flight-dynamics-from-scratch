# Lesson 7 · Verification and validation

> *A flight-dynamics model produces plausible-looking oscillations whether it is right or
> wrong. This lesson looks back over the project and asks how we know the numbers in the other
> six lessons can be trusted — and what checking them taught us about the model, and about the
> fifty-year-old reports it was checked against.*

**Code:** [`tests/`](../../tests), [`validation/`](../../validation) ·
**Report:** [validation report](../validation-report.md) ·
**Previous:** [Lesson 6 — Stability trade studies](06-stability-trade-studies.md)

## Learning objectives

By the end of this lesson you should be able to

1. distinguish verification (solving the equations right) from validation (solving the right
   equations);
2. build a hierarchy of verification checks for a dynamics code, from closed forms to
   convergence orders;
3. separate data, assumptions, deductions and calibrations, and label each;
4. audit published data against itself, and judge agreement against rounding;
5. state where a model stops being trustworthy.

---

## 1. Two different questions

**Verification** asks whether the code solves the equations it claims to solve. Its references
are mathematical: closed-form solutions, identities, a second independent derivation.
**Validation** asks whether those equations describe the aircraft. Its references are physical:
measured or published data. A verified model can still be invalid (right answer to the wrong
equations), and a model that matches data can still be unverified (two errors cancelling).
This project keeps the two apart: `tests/` holds both, every test names its reference, and
[`validation/`](../../validation) holds the comparisons with real aircraft.

## 2. A hierarchy of verification checks

From cheapest to most searching:

| Level | Idea | Examples here |
|---|---|---|
| closed form | the code must reproduce a known solution | torque-free precession, constant-moment pitching, a ballistic parabola |
| conservation | invariants must stay invariant | angular momentum and energy with $I_{xz}$ |
| second route | two independent derivations must agree | numerical vs analytic Jacobians (1e-8); 6-DOF trim vs a scalar wind-axis solution (1e-10); Euler vs quaternion attitude |
| invariance | the physics must not depend on a convention | eigenvalues in body and stability axes (1e-10); re-referenced vs original model |
| symmetry | what symmetry forbids must not happen | longitudinal–lateral coupling 1e-31 in symmetric flight |
| order | errors must shrink at the predicted rate | nonlinear − linear ∝ a² (elevator) and a³ (rudder); CG-transfer residual ∝ rate² |

The last level is the most searching. A test that only checks "small error" passes a model
whose error is small *by luck at that point*. A test that checks how the error *scales* fails
as soon as a term is missing. That is how the CG transfer of Lesson 6 was caught: its residual
was small (1e-4 of the weight) but halved, instead of quartering, when the rates halved. The
missing terms turned out to be first order.

The rudder case shows the other side. Its error ratio was 8, not the 4 expected, and the test
failed. Nothing was wrong with the model. The prediction was: mirror symmetry makes lateral
states odd functions of lateral inputs, so the quadratic term vanishes. Updating the test to
expect a³ turned a failure into a stronger check.

## 3. Validation: data, assumption, deduction, calibration

Every input of the model falls into one of four classes, and the code says which:

- **Data**: printed in a cited table and transcribed as printed ([`validation/reference_data/`](../../validation/reference_data)).
- **Assumption**: chosen, not measured, and labelled. Examples: $C_{D_M}$ = 0 for the 747,
  thrust independent of speed, the tail arm of 3.3 c̄ in the trade studies, the −20° elevator
  limit.
- **Deduction**: not printed, but forced by printed numbers. The 747's aerodynamic $C_{m_0}$ is
  the one that balances the published thrust line; the Navion's $C_{m_{\delta_e}}$ comes from its
  printed $M_{\delta_e}$.
- **Calibration**: a parameter tuned to match data, after which that data no longer counts as
  validation. The horizontal tail of Lesson 6 is calibrated to $C_{m_q}$, $C_{m_{\dot\alpha}}$ and
  $C_{m_{\delta_e}}$, so it cannot be validated against them.

What is left for validation is everything the inputs do not already fix: the dimensional
derivatives from the nondimensional ones, the roots from the derivatives, trim from the
derivatives and the thrust line.

## 4. How good is "agreement"?

Both reports print three or four significant figures. A model fed rounded inputs cannot be
expected to match rounded outputs better than the rounding allows, and for a sensitive
quantity that allowance can be large. The [mode validation](../../validation/mode_validation.py)
quantifies it with a Monte Carlo: vary every printed derivative uniformly within its rounding
interval, recompute the roots, and record the spread. For most modes the band is under 1 %.
For the 747's fc 7 phugoid, nearly critically damped, it spans −11 % to +0.5 % in damping
ratio. A 6 % "disagreement" there is no disagreement; a 9 % one at fc 2, where the band is
2 % wide, is a real difference that needs explaining.

The rule this project follows: never loosen a tolerance to make a test pass without writing
down why, in the test, and the why must be a number (a rounding band, a printed inconsistency),
not a feeling.

## 5. Case study: what the 747's data was hiding

Running the 747's nondimensional derivatives through CR-2144's own formulas reproduced its
dimensional table to 0.65 %, with three exceptions. The way they were resolved is a good
picture of validation in practice.

1. **$M_w$ was 1.7 % off.** A first explanation fitted it perfectly: the table matched to
   0.35 % if a small Mach term entered with the wrong sign. That was recorded as a sign slip in
   the report.
2. **$M_u^{\ast}$ was off by a factor of 3.** Unexplained, and set aside.
3. **Lesson 4's phugoid**, from the nondimensional route, was 30 % off in damping, traced to
   $M_u$.

Looking at all three together, both $M_u$ and $M_w$ contain the reference $C_m$, which had been
set to zero because the data is "trimmed". But the report also publishes a thrust line, 10 ft
below the CG. If the airframe is trimmed *including* the thrust moment, the aerodynamic $C_m$
at the reference is $-Tl_{th}/(qS\bar c)$ = −0.096, not zero. With that one deduction, $M_u^{\ast}$
agreed to 0.1 %, $M_w$ to 0.2 %, the phugoid to 1 %, and the trim of Lesson 3 moved to within
0.5° of the published elevator. The "sign slip" was wrong. It fitted one number, and only the
second number exposed it.

The lesson generalises: one anomaly can be explained many ways; two anomalies with a common
cause usually admit one. Resolve discrepancies together where you can, not one at a time.

## 6. What validation found

In the **reference data**, kept as printed and documented by the tests:

- the 747's lateral derivatives are stability-axis (the tables do not say);
- its data is trimmed with the thrust moment;
- its published roots include $\dot u$ terms and $X_q$ that its Appendix C says are neglected
  (without them the phugoid damping is 7–9 % low, outside the rounding band);
- its $Z_q$ uses $U_0$ where the derivation gives V;
- the Navion's dimensional derivatives use S = 184 ft², against 180 ft² printed;
- the 747's dynamic pressure at fc 9 and 10 is printed 1.1 % above $\rho V^2/2$.

In **this model**, all fixed and tested: the incomplete CG transfer; the ISA's lower limit at
0 m, which made sea-level linearization fail; body-axis Δu plotted as airspeed; classical
approximations fed body-axis derivatives; a CG limit that picked up a marginal phugoid; and a
wrong claim about the doublet spectrum in a lesson. The [validation report](../validation-report.md)
describes each.

## 7. Reproducibility

Every figure, table, GIF and notebook in the repository is generated by a script and rebuilt
byte for byte:

```bash
uv run python validation/run_all.py          # validation figures and results tables
uv run python notebooks/build_notebooks.py   # executed notebooks
uv run python docs/lessons/make_figures.py   # lesson figures
```

Byte-identical rebuilds turn "did anything change?" into a `git status`. That caught one stale
figure in this project: a validation plot regenerated after the $C_{m_0}$ deduction but not
committed.

## 8. Where this model ends

The model is validated for small motions about the published flight conditions. Outside that:

- **Linear aerodynamics.** Derivatives are constant: no stall, no Mach-dependent tables, no
  large-angle effects. Lesson 5's large-bank case shows where the linear lateral model
  departs; the aerodynamic model itself is linear in every case.
- **Rigid body.** CR-2144's plots of the 747 derivatives are labelled "flexible": the data
  includes aeroelastic effects at its conditions, which a rigid model cannot extrapolate.
- **No engine dynamics** and thrust independent of speed; no ground effect; flat, non-rotating
  Earth.
- **No stability augmentation.** The real 747 flies with a yaw damper (CR-2144 documents it);
  the model is the bare airframe.
- **Trade studies** rest on stated assumptions (tail arm, elevator limit, fixed inertias): use
  them for trends.

## 9. Exercises

1. Classify each as verification or validation: (a) the Navion's roots match Teper;
   (b) numerical and analytic Jacobians agree; (c) the nonlinear error scales as a².
2. Why does an error-order test catch a missing first-order term that a tolerance test
   might miss?
3. A colleague proposes adjusting the 747's $C_{m_q}$ until the phugoid damping matches
   CR-2144. What is wrong with then calling the phugoid "validated"?
4. The Monte Carlo band for a root is [−2.1 %, −0.1 %] and the model is −1.1 % off. Is that
   agreement? What if the band were [−0.4 %, +0.3 %]?
5. Plan the verification of a yaw damper added to the model: list one check at each level of
   the table in section 2.

<details>
<summary>Answers</summary>

1. (a) Validation (published data). (b) Verification (two derivations of the same equations).
   (c) Verification (a property of the Taylor series that the linearization must satisfy).
2. A tolerance test checks one point; a missing term may happen to be small there. An order
   test checks how the error changes as the input shrinks: a first-order error halves when
   the input halves, a second-order one quarters, so the ratio exposes the missing term
   whatever its size.
3. The phugoid would then be calibrated, not validated: the data used to tune the model cannot
   also confirm it. And $C_{m_q}$ is printed data; tuning it would hide whatever the real cause
   is (here, the $\dot u$ terms).
4. Yes: −1.1 % lies inside what rounding alone produces. With [−0.4 %, +0.3 %] it is outside
   the band, so something beyond rounding differs and needs an explanation.
5. Closed form: with yaw-rate feedback gain k, the dutch-roll damping term changes by
   $N'_{\delta_r} k$ in the two-state approximation. Second route: closed-loop matrix
   $A + BK$ built analytically vs by linearizing the nonlinear model with the damper in the loop.
   Invariance: the damper must not change the longitudinal roots. Symmetry: a pure elevator
   input must not move the rudder. Order: closed-loop nonlinear − linear error ∝ a³ for rudder
   inputs.

</details>

## Key takeaways

- Verification checks the equations are solved; validation checks they are the right ones.
  Keep the evidence for each separate.
- The strongest verification compares two independent routes, or checks how errors scale.
- Label every input as data, assumption, deduction or calibration. Calibrated quantities
  cannot validate.
- Judge agreement against the rounding of the source, and explain discrepancies together.
- Here, checking two reports against themselves found an axis convention, a trim convention,
  undocumented terms and a wrong wing area — and six errors in this model.

## Further reading

- W. L. Oberkampf & C. J. Roy, *Verification and Validation in Scientific Computing*,
  Cambridge, 2010.
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*,
  Hermosa, 1998.
- R. V. Jategaonkar, *Flight Vehicle System Identification*, AIAA, Ch. 1 (model validation
  against flight data).
