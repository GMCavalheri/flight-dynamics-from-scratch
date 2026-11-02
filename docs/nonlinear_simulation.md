# Nonlinear simulation and mode identification

Theory note for [`src/flightdyn/nonlinear_sim/`](../src/flightdyn/nonlinear_sim).
Lesson: [05](lessons/05-nonlinear-simulation.md).

## Simulation

Inputs (`Step`, `Pulse`, `Doublet`) are increments on the trim controls, piecewise constant
with known breakpoints. `simulate` integrates the nonlinear model with DOP853 (rtol = atol =
$10^{-10}$) segment by segment between breakpoints, sampling the dense output. `simulate_linear`
does the same for any `LinearModel`. A doublet of half-width T has spectrum
$4a\sin^2(\omega T/2)/\omega$, peaking at $\omega \approx 2.33/T$.

## Identification: the matrix pencil

For a free response sampled every Δt, $y_k = \sum_i c_iz_i^k$ with $z_i = e^{\lambda_i\Delta t}$. With
the Hankel matrix $\mathbf Y$ of the samples, its truncated SVD $\mathbf Y \approx \mathbf U\boldsymbol\Sigma\mathbf V^{H}$
(M terms), and $\mathbf V_1$, $\mathbf V_2$ the right singular vectors without their last and first row,
the $z_i$ are the eigenvalues of $\mathbf V_1^{+}\mathbf V_2$ (Hua & Sarkar 1990). Several output channels
are stacked so they share the poles. No model is involved.

## Verification and validation

- Linear simulation equals the matrix-exponential step response to $10^{-10}$; the pencil
  recovers known poles to $10^{-8}$.
- Nonlinear minus linear response scales as $a^2$ for elevator doublets and $a^3$ for rudder
  doublets: mirror symmetry makes lateral states odd in lateral inputs.
- Poles identified from the nonlinear 747's time histories match the linear eigenvalues to
  0.1 %, provided altitude is a state of the linear model (the density gradient moves the
  phugoid by 1.6 %).

See [`validation/results/simulation_validation.md`](../validation/results/simulation_validation.md).
