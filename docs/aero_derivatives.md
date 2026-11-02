# Aerodynamic derivatives

Theory note for [`src/flightdyn/aero_derivatives/`](../src/flightdyn/aero_derivatives).
Lesson: [02](lessons/02-aerodynamic-derivatives.md).

## Model

First-order expansion about a reference (α0, M0), in NASA CR-2144's form (Appendix A):
stability-axis lift and drag, pitching moment, side force, and rolling/yawing moments in
stability or body axes (`lateral_axes`):

```math
C_L = C_{L_0} + C_{L_\alpha}\Delta\alpha + C_{L_{\dot\alpha}}\frac{\dot\alpha\bar c}{2V} + C_{L_q}\frac{q\bar c}{2V} + C_{L_M}\Delta M + C_{L_{\delta_e}}\delta_e
```

and likewise for $C_D$ (with $C_{D_q}$), $C_m$, $C_Y$, $C_l$, $C_n$. Lift and drag are rotated into
body axes through the **instantaneous** α. Stability-axis moments and rates are rotated through
the fixed α0.

## Transformations and dimensional derivatives

- [`body_axis_coefficients`](../src/flightdyn/aero_derivatives/dimensional.py): CR-2144
  Appendix B (stability → body), e.g.
  $C_{N_\alpha} = C_{L_\alpha}\cos\alpha_0 - C_L\sin\alpha_0 + C_{D_\alpha}\sin\alpha_0 + C_D\cos\alpha_0$.
- `dimensional_derivatives`: CR-2144 Appendix A section 5 (body axes, SI), plus three terms the
  force model implies and the appendix does not print: $X_q$ and $Z_q$ (with V, by analogy
  with $M_q$) and the $\dot u$ terms of $\dot\alpha$ ($X_{\dot u}$, $Z_{\dot u}$, $M_{\dot u}$).
- Primed lateral derivatives: $L' = (L + I_{xz}N/I_{xx})G$, $N' = (N + I_{xz}L/I_{zz})G$,
  $G = 1/(1 - I_{xz}^2/(I_{xx}I_{zz}))$.
- [`to_stability_axes`](../src/flightdyn/aero_derivatives/model.py): the same aircraft with
  α0 = 0 (inertia tensor and thrust line rotated), for the classical approximations.

## Verification and validation

- Every Appendix A and B formula equals a finite difference of the force model
  ([`tests/test_aero_derivatives.py`](../tests/test_aero_derivatives.py)).
- 747 (CR-2144 Tables IX-1/2 → IX-4/8): all derivatives within 0.65 %, except $Z_q$ (the
  table uses $U_0$ for V, 1.1 %). The comparison shows the lateral derivatives are
  stability-axis, and that the data is trimmed with the thrust moment ($C_{m_0}$ =
  $-Tl_{th}/(qS\bar c)$ reproduces $M_u^{\ast}$ and $M_w$).
- Navion (Teper Tables X-B/C → X-D/E): uniformly 2.22 % (moments) larger than the tables imply,
  i.e. S = 184 ft² in the dimensional table against 180 ft² printed.

See [`validation/results/derivative_validation.md`](../validation/results/derivative_validation.md).
