# Equations of motion

Theory note for [`src/flightdyn/equations_of_motion/`](../src/flightdyn/equations_of_motion)
and [`rotations.py`](../src/flightdyn/rotations.py). Lesson: [01](lessons/01-equations-of-motion.md).

## Assumptions

Rigid aircraft, constant mass, flat non-rotating Earth with constant gravity $g_0$ (so
geometric and geopotential altitude coincide), plane of symmetry x-z (so $I_{xy} = I_{yz} = 0$).

## Frames and states

Earth axes North-East-Down; body axes x forward, y right, z down, origin at the CG. Attitude
by the 3-2-1 Euler angles (ψ, θ, φ), or by a unit quaternion. State vector:

```math
\mathbf x = (u, v, w, p, q, r, \phi, \theta, \psi, x_n, y_e, z_d)
```

## Equations

```math
\begin{aligned}
\dot{\mathbf v} &= \mathbf F/m + \mathbf C\,(0, 0, g_0)^{T} - \boldsymbol\omega \times \mathbf v \\
\mathbf I\dot{\boldsymbol\omega} &= \mathbf M - \boldsymbol\omega \times \mathbf I\boldsymbol\omega,
\qquad \mathbf I = \begin{pmatrix}I_{xx} & 0 & -I_{xz}\\ 0 & I_{yy} & 0\\ -I_{xz} & 0 & I_{zz}\end{pmatrix} \\
(\dot\phi, \dot\theta, \dot\psi) &= \left(p + (q\sin\phi + r\cos\phi)\tan\theta,\ q\cos\phi - r\sin\phi,\ (q\sin\phi + r\cos\phi)/\cos\theta\right) \\
\dot{\mathbf q} &= \tfrac12\boldsymbol\Omega(\boldsymbol\omega)\,\mathbf q + k\left(1 - \lVert\mathbf q\rVert^2\right)\mathbf q \\
\dot{\mathbf r}_{NED} &= \mathbf C^{T}\mathbf v
\end{aligned}
```

with $\mathbf C = \mathbf R_1(\phi)\mathbf R_2(\theta)\mathbf R_3(\psi)$ the body-from-NED direction cosine
matrix. **F** and **M** are aerodynamic plus propulsive (gravity is added inside).

## The aircraft model

[`state_derivative`](../src/flightdyn/equations_of_motion/aircraft_dynamics.py) closes the loop
with the ISA (altitude $-z_{d}$, geopotential), the derivative model and the thrust line. The
$\dot\alpha$ derivatives make it implicit. Because the force is affine in $\dot\alpha$, two
evaluations give $\dot{\mathbf x} = \mathbf f_0 + \dot\alpha(\mathbf f_1 - \mathbf f_0)$ and the fixed point
$\dot\alpha = (u\dot w - w\dot u)/(u^2 + w^2)$ is solved exactly, with no lag.

## Verification ([`tests/test_equations_of_motion.py`](../tests/test_equations_of_motion.py))

| Case | Reference |
|---|---|
| constant pitching moment | $q = Mt/I_{yy}$, $\theta = Mt^2/(2I_{yy})$ |
| ballistic flight while tumbling | parabola in Earth axes |
| torque-free axisymmetric body | precession at $(I_3 - I)r/I$ |
| torque-free body with $I_{xz}$ | constant $\mathbf h_{NED}$ and kinetic energy |
| gravity only | constant total energy |
| Euler vs quaternion, loop through θ = 90° | same trajectory; quaternions pass the singularity |
| Poisson's equation, constant roll | $\dot{\mathbf C} = -[\boldsymbol\omega\times]\mathbf C$; $\mathbf q(t)$ exact |

References: Etkin & Reid, *Dynamics of Flight*, ch. 4; Stevens, Lewis & Johnson, *Aircraft
Control and Simulation*, ch. 1.
