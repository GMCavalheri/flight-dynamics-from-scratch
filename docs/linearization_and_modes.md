# Linearization and modal analysis

Theory note for [`src/flightdyn/linearization/`](../src/flightdyn/linearization) and
[`src/flightdyn/modal_analysis/`](../src/flightdyn/modal_analysis).
Lesson: [04](lessons/04-linearization-and-modes.md).

## Two routes to the linear model

- **Numerical** (`linearize`): central differences of `state_derivative` about a trim, steps
  scaled per variable; twelve-state A (12×12) and B (12×4).
- **Analytic** (`longitudinal_model`, `lateral_model`, `analytic_model`): the small-perturbation
  equations of CR-2144 Appendix C from the dimensional derivatives. Longitudinal in descriptor
  form, with the $\dot\alpha$ terms on the left:

```math
\begin{pmatrix}1 - X_{\dot u} & -X_{\dot w} & 0 & 0\\ -Z_{\dot u} & 1 - Z_{\dot w} & 0 & 0\\ -M_{\dot u} & -M_{\dot w} & 1 & 0\\ 0 & 0 & 0 & 1\end{pmatrix}\dot{\mathbf x}
= \begin{pmatrix}X_u & X_w & X_q - W_0 & -g\cos\theta_0\\ Z_u & Z_w & Z_q + U_0 & -g\sin\theta_0\\ M_u & M_w & M_q & 0\\ 0 & 0 & 1 & 0\end{pmatrix}\mathbf x + \mathbf G\mathbf u
```

  Lateral in (β, p, r, φ) with primed derivatives. `rereference` re-expands the derivative
  model about any wings-level trim, so the two routes describe the same point.

## Modes

`longitudinal_modes` pairs the four roots by magnitude (phugoid smallest, short period
largest); `lateral_modes` takes the complex pair as dutch roll and the two real roots as
spiral (smaller) and roll subsidence. A `Mode` gives ωn, ζ, damped frequency, period, time
constant and time to half (negative: to double). `mode_shape` returns normalised eigenvectors.
Approximations (`approximations`): short period (w, q), Lanchester and two-state phugoid, roll
$\lambda = L'_{p}$, spiral $-g(L'_\beta N'_r - N'_\beta L'_r)/(V(L'_\beta N'_p - N'_\beta L'_p))$,
two-state dutch roll; they assume stability axes.

## Verification and validation

- Numerical and analytic Jacobians agree to $10^{-8}$; symmetric flight decouples to $10^{-31}$;
  eigenvalues are the same in body and stability axes to $10^{-10}$.
- Navion (Teper X-D/E → X-F/G): all eight mode values within 0.11 %.
- 747 (CR-2144 IX-4/8 → IX-5/9), ten conditions: within 1.2 %, except fc 5's spiral (1.8 %)
  and fc 7's near-critical phugoid, both inside their rounding bands. This needs the
  $\dot u$ terms and $X_q$, which CR-2144's Appendix C says are neglected; without them the
  phugoid damping is 7–9 % low at fc 1–2.

See [`validation/results/mode_validation.md`](../validation/results/mode_validation.md).
