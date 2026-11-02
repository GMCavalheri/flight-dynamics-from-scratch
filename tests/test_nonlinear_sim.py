import numpy as np
import pytest
from scipy.linalg import expm

from flightdyn.linearization import LinearModel, lateral, linearize, longitudinal
from flightdyn.nonlinear_sim import (
    Doublet,
    Pulse,
    Step,
    breakpoints,
    control_increment,
    damped_pairs,
    matrix_pencil,
    simulate,
    simulate_linear,
)
from flightdyn.trim_solver import trim

from reference_aircraft import b747, b747_condition

AC, COND = b747(2), b747_condition(2)
TRIM = trim(AC, COND["V"], COND["h"])
LM = linearize(AC, TRIM)
LON_H = [0, 2, 4, 7, 11]  # u, w, q, theta, z_d: the longitudinal set with altitude
LAT = [1, 3, 5, 6]  # v, p, r, phi


def test_input_shapes():
    d = Doublet("de", 0.1, start=1.0, width=2.0)
    assert [d.value(t) for t in (0.5, 1.0, 2.9, 3.0, 4.9, 5.0)] == [0, 0.1, 0.1, -0.1, -0.1, 0]
    assert d.breakpoints == (1.0, 3.0, 5.0)
    assert Step("thrust", 5.0, 2.0).value(1.9) == 0.0 and Step("thrust", 5.0, 2.0).value(2.0) == 5
    assert (
        Pulse("da", 1.0, 1.0, 0.5).value(1.4) == 1.0 and Pulse("da", 1.0, 1.0, 0.5).value(1.5) == 0
    )
    assert breakpoints([d, Step("dr", 1.0, 10.0)], 8.0) == [0.0, 1.0, 3.0, 5.0, 8.0]
    assert np.allclose(control_increment([d, Step("dr", 0.2)], 4.0), [-0.1, 0, 0.2, 0])


def test_matrix_pencil_recovers_known_poles():
    # y = sum of damped exponentials with known poles, two channels: exact recovery
    poles = np.array([-0.5 + 0.7j, -0.5 - 0.7j, -0.003 + 0.13j, -0.003 - 0.13j])
    t = np.arange(0.0, 150.0, 0.1)
    c1 = np.array([1.0 + 0.5j, 1.0 - 0.5j, 0.3j, -0.3j])
    c2 = np.array([0.2, 0.2, 2.0 - 1j, 2.0 + 1j])
    Y = np.column_stack([(np.exp(np.outer(t, poles)) @ c).real for c in (c1, c2)])
    found = matrix_pencil(Y, 0.1, 4)
    assert np.allclose(np.sort_complex(found), np.sort_complex(poles), atol=1e-8)
    osc, real = damped_pairs(found)
    assert len(osc) == 2 and not real and osc[0] == pytest.approx(poles[2], abs=1e-8)


def test_linear_simulation_matches_matrix_exponential():
    # step response of x' = A x + B u from rest: x(t) = A^-1 (e^{At} - I) B u
    A = np.array([[-0.4, 1.0], [-2.0, -0.3]])
    B = np.array([[0.0], [1.0]])
    model = LinearModel(A=A, B=B, states=("a", "b"), controls=("de",))
    r = simulate_linear(model, [Step("de", 0.5, 0.0)], t_end=6.0, dt=0.5, control_index=(0,))
    for tk, xk in zip(r.t, r.x, strict=True):
        exact = np.linalg.solve(A, (expm(A * tk) - np.eye(2)) @ B[:, 0] * 0.5)
        assert np.allclose(xk, exact, atol=1e-10)


@pytest.mark.parametrize(
    "channel, idx, ctrl, order",
    [("de", [4, 7], (0, 3), 2), ("dr", [5, 6], (1, 2), 3)],
)
def test_nonlinear_minus_linear_error_order(channel, idx, ctrl, order):
    # the linearization is the first-order term of the nonlinear response, so their
    # difference scales with the square of the input amplitude. For a lateral input it scales
    # with the cube: by mirror symmetry the lateral states are odd functions of a lateral
    # input, so the quadratic term vanishes
    model_idx = LON_H if channel == "de" else LAT
    k = [model_idx.index(i) for i in idx]

    def err(a):
        inp = [Doublet(channel, np.radians(a), 1.0, 2.0)]
        nl = simulate(AC, TRIM.x, TRIM.controls, inp, t_end=20.0, dt=0.1)
        sub = LM.submodel(model_idx, list(ctrl))
        lin = simulate_linear(sub, inp, t_end=20.0, dt=0.1, control_index=ctrl)
        return np.abs((nl.x[:, idx] - TRIM.x[idx]) - lin.x[:, k]).max()

    ratio = err(1.0) / err(0.5)
    assert ratio == pytest.approx(2.0**order, rel=0.05)


def test_identified_poles_of_the_nonlinear_aircraft_match_its_eigenvalues():
    # free response after a small elevator doublet, analysed without any model: the poles
    # of the nonlinear time history are the eigenvalues of the linear model with altitude
    inp = [Doublet("de", np.radians(0.25), 1.0, 2.0)]
    r = simulate(AC, TRIM.x, TRIM.controls, inp, t_end=300.0, dt=0.1)
    m = r.t >= 5.0
    Y = r.x[m][:, LON_H] - TRIM.x[LON_H]
    found, _ = damped_pairs(matrix_pencil(Y, 0.1, 5))
    eig, _ = damped_pairs(np.linalg.eigvals(LM.A[np.ix_(LON_H, LON_H)]))
    assert len(found) == 2
    for f, e in zip(found, eig, strict=True):
        assert abs(f - e) / abs(e) < 2e-3


def test_identified_lateral_poles_match_eigenvalues():
    inp = [Doublet("dr", np.radians(0.25), 1.0, 2.0)]
    r = simulate(AC, TRIM.x, TRIM.controls, inp, t_end=60.0, dt=0.05)
    m = r.t >= 5.0
    Y = r.x[m][:, LAT] - TRIM.x[LAT]
    found = matrix_pencil(Y, 0.05, 4)
    eig = np.linalg.eigvals(lateral(LM, COND["V"]).A)
    for e in eig:
        assert np.min(np.abs(found - e)) / abs(e) < 5e-3


def test_longitudinal_and_lateral_submodels_are_consistent():
    # the convenience blocks used elsewhere are the same rows and columns of the Jacobian
    assert np.allclose(longitudinal(LM).A, LM.A[np.ix_([0, 2, 4, 7], [0, 2, 4, 7])])
