import numpy as np
import pytest

from flightdyn.equations_of_motion import state_derivative
from flightdyn.linearization import (
    analytic_model,
    coupling,
    lateral,
    linearize,
    longitudinal,
    rereference,
)
from flightdyn.trim_solver import trim

from reference_aircraft import b747, b747_condition, navion, navion_condition

CASES = [
    ("747 fc1", b747(1), b747_condition(1)),
    ("747 fc2", b747(2), b747_condition(2)),
    ("Navion", navion(), navion_condition()),
]


@pytest.mark.parametrize("name, ac, c", CASES, ids=[c[0] for c in CASES])
def test_numerical_jacobian_matches_analytic_small_perturbation_model(name, ac, c):
    # the central-difference Jacobian of the nonlinear model and the Appendix A/C analytic
    # matrices (re-expanded about the trim, all alpha-dot terms kept) describe one linear model
    t = trim(ac, c["V"], c["h"])
    lm = linearize(ac, t)
    lon, lat = analytic_model(ac, t)
    for num, ana in ((longitudinal(lm), lon), (lateral(lm, c["V"]), lat)):
        scale = np.abs(ana.A).max()
        assert np.allclose(num.A, ana.A, atol=2e-8 * scale, rtol=1e-6)
        assert np.allclose(num.B, ana.B, atol=2e-8 * np.abs(ana.B).max(), rtol=1e-6)


@pytest.mark.parametrize("name, ac, c", CASES, ids=[c[0] for c in CASES])
def test_symmetric_flight_decouples(name, ac, c):
    # Etkin & Reid ch. 6: about symmetric flight the longitudinal and lateral equations
    # separate exactly
    lm = linearize(ac, trim(ac, c["V"], c["h"]))
    assert coupling(lm) < 1e-12


def test_rereferenced_model_is_the_same_aircraft():
    # re-expanding a linear model about another point must not change it anywhere
    ac, c = b747(1), b747_condition(1)
    t = trim(ac, c["V"], c["h"])
    re = rereference(ac, t)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = t.x + rng.normal(scale=[3, 2, 2, 0.05, 0.05, 0.05, 0.1, 0.05, 0, 0, 0, 10])
        u = rng.normal(scale=[0.05, 0.05, 0.05, 1e4])
        u_re = u.copy()
        u_re[0] -= t.controls[0]  # the new model measures elevator from the trim setting
        assert np.allclose(state_derivative(ac, x, u), state_derivative(re, x, u_re), atol=1e-11)


def test_jacobian_of_linear_function_is_exact():
    from flightdyn.linearization import jacobians

    A0, B0 = np.array([[1.0, 2.0], [-3.0, 0.5]]), np.array([[0.0], [4.0]])
    A, B = jacobians(lambda x, u: A0 @ x + B0 @ u, np.ones(2), np.zeros(1), [1e-3] * 2, [1e-3])
    assert np.allclose(A, A0) and np.allclose(B, B0)
