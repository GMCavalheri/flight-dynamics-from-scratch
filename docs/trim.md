# Trim

Theory note for [`src/flightdyn/trim_solver/`](../src/flightdyn/trim_solver).
Lesson: [03](lessons/03-trim.md).

## Formulation

Specified: airspeed V, altitude h, flight-path angle γ, turn rate ψ̇. Unknowns and equations:

```math
\mathbf z = (\alpha, \beta, \phi, \delta_e, \delta_a, \delta_r, T/W), \qquad
(\dot u, \dot v, \dot w, \dot p, \dot q, \dot r) = 0, \quad \beta = 0 \ \text{(or } \phi = 0\text{)}
```

solved by `scipy.optimize.root` (hybrid Powell) to a residual below $10^{-10}$. The state follows
from the unknowns through the flight-path constraint and the turn kinematics (Stevens, Lewis &
Johnson, ch. 3):

```math
\tan\theta = \frac{ab + \sin\gamma\sqrt{a^2 - \sin^2\gamma + b^2}}{a^2 - \sin^2\gamma}, \qquad
(p, q, r) = \dot\psi\left(-\sin\theta,\ \sin\phi\cos\theta,\ \cos\phi\cos\theta\right)
```

with $a = \cos\alpha\cos\beta$ and $b = \sin\phi\sin\beta + \cos\phi\sin\alpha\cos\beta$.
`trim_guess` restarts a neighbouring trim from a previous one (continuation).

## Verification ([`tests/test_trim.py`](../tests/test_trim.py))

- Level flight, 3° climb, 4° descent: equal to an independent wind-axis solution of
  $T\cos\alpha = D + W\sin\gamma$, $L + T\sin\alpha = W\cos\gamma$, $C_m = 0$ (bisection), to $10^{-10}$ rad.
- Coordinated turn: the inertial acceleration $\dot\psi\hat z \times \mathbf v_{NED}$ and bank of the
  total force, $\tan\mu = V\dot\psi/g$, to $10^{-12}$.
- 20 s of nonlinear flight from a trim with fixed controls: nothing moves but ψ and position.

## The 747's reference condition

With the aerodynamic $C_{m_0}$ balancing the published thrust line (2.5°, 10 ft below the CG),
the model trims within 0.45° of α0 with under 0.5° of elevator. The remaining α offset is the
lift of the inclined thrust, which the tabulated $C_L = W/(qS)$ leaves out.
