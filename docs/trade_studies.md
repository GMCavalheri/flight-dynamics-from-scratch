# Stability and control trade studies

Theory note for [`src/flightdyn/trade_studies/`](../src/flightdyn/trade_studies).
Lesson: [06](lessons/06-stability-trade-studies.md).

## CG transfer

`move_cg` moves the CG aft by Δh mean chords (Δx = Δh c̄). Moments gain $\mathbf r \times \mathbf F$:
$\Delta C_m = \Delta h C_N$, $\Delta C_n = (\Delta x/b)C_Y$. The old reference point moves at
$\mathbf v + \boldsymbol\omega \times \mathbf r$: per unit $\hat q$ its angle of attack changes by $-2\Delta h\cos\alpha_0$ and
its speed by $-2\Delta h\sin\alpha_0$ (relative), so dynamic pressure and Mach shift at first order
in body axes; per unit $\hat r$ its sideslip changes by $2\Delta x/b$. Lift and drag, defined
relative to the local velocity, exchange $C_{D_0}$ and $C_{L_0}$ terms. Together these give the
rigid-body transfer to first order, which the test checks with a residual quadratic in the
rates. Inertias are kept (a labelled simplification).

## Neutral point, divergence, tail

- Neutral point: $C_{m_\alpha} + \Delta h C_{N_\alpha} = 0$; static margin $K_n = -C_{m_\alpha}/C_{N_\alpha}$.
- Aperiodic divergence: the CG where the product of the longitudinal roots (the constant term
  of the characteristic polynomial) changes sign.
- Horizontal tail (`HorizontalTail.calibrate`): from $C_{m_q} = -2\eta a_tV_H l_t/\bar c$ with an
  assumed tail arm, then $\epsilon_\alpha = C_{m_{\dot\alpha}}/C_{m_q}$ and $\tau = -C_{m_{\delta_e}}/(\eta a_tV_H)$;
  `scale_tail` adds (k − 1) times each tail contribution.
- Lateral map: spiral root and dutch-roll damping over scaled $C_{l_\beta}$ and $C_{n_\beta}$; the spiral
  boundary is $L'_\beta N'_r = N'_\beta L'_r$ (θ0 = 0, stability axes).

## Results (747, assumptions in `validation/trade_studies.py`)

Neutral point 0.470 MAC (static margin 22 % at the published 0.25 MAC); aperiodic divergence
at 0.448 MAC; usable CG range 44 % MAC with the 747's tail, 15 % with 60 % of it. See
[`validation/results/trade_studies.md`](../validation/results/trade_studies.md) and the optional
AERO-01 cross-check [`validation/results/vlm_navion.md`](../validation/results/vlm_navion.md).
