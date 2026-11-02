import numpy as np
import pytest

from flightdyn import rotations as R

ANGLES = [(0.3, -0.2, 1.1), (-1.2, 0.7, -2.9), (0.0, 1.4, 0.5), (2.5, -1.3, 3.0)]


@pytest.mark.parametrize("phi, theta, psi", ANGLES)
def test_dcm_is_a_rotation(phi, theta, psi):
    C = R.dcm_body_from_ned(phi, theta, psi)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(C) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("phi, theta, psi", ANGLES)
def test_euler_round_trip(phi, theta, psi):
    assert R.euler_from_dcm(R.dcm_body_from_ned(phi, theta, psi)) == pytest.approx(
        (phi, theta, psi), abs=1e-12
    )


@pytest.mark.parametrize("phi, theta, psi", ANGLES)
def test_quaternion_matches_euler(phi, theta, psi):
    q = R.quat_from_euler(phi, theta, psi)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(R.dcm_from_quat(q), R.dcm_body_from_ned(phi, theta, psi), atol=1e-14)
    assert R.euler_from_quat(q) == pytest.approx((phi, theta, psi), abs=1e-12)


def test_gravity_in_body_axes():
    # Etkin & Reid ch. 4: gravity in body axes is g (-sin theta, cos theta sin phi,
    # cos theta cos phi), independent of heading
    phi, theta, psi = 0.4, -0.3, 2.0
    gb = R.dcm_body_from_ned(phi, theta, psi) @ np.array([0.0, 0.0, 1.0])
    expected = [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]
    assert np.allclose(gb, expected, atol=1e-15)


@pytest.mark.parametrize("phi, theta, psi", ANGLES[:3])
def test_euler_rates_match_dcm_kinematics(phi, theta, psi):
    # Poisson's equation dC/dt = -[w x] C must hold for the Euler-angle rates
    p, q, r = 0.3, -0.5, 0.8
    rates = R.euler_rates(phi, theta, p, q, r)
    h = 1e-6
    a = np.array([phi, theta, psi])
    dC = (R.dcm_body_from_ned(*(a + h * rates)) - R.dcm_body_from_ned(*(a - h * rates))) / (2 * h)
    W = np.array([[0.0, -r, q], [r, 0.0, -p], [-q, p, 0.0]])
    assert np.allclose(dC, -W @ R.dcm_body_from_ned(*a), atol=1e-9)


def test_quaternion_rates_constant_roll():
    # a constant roll rate p for time t rotates the body by phi = p t about x:
    # q(t) = (cos(pt/2), sin(pt/2), 0, 0) exactly
    p, t = 0.7, 2.0
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    from scipy.integrate import solve_ivp

    sol = solve_ivp(lambda _, q: R.quat_rates(q, p, 0.0, 0.0), (0, t), q0, rtol=1e-12, atol=1e-12)
    assert np.allclose(sol.y[:, -1], [np.cos(p * t / 2), np.sin(p * t / 2), 0, 0], atol=1e-10)


def test_quaternion_norm_correction_is_restoring():
    # off the unit sphere, the correction term moves |q|^2 back toward 1
    q = 1.1 * R.quat_from_euler(0.2, 0.1, 0.3)
    dq = R.quat_rates(q, 0.0, 0.0, 0.0)
    assert q @ dq < 0.0


def test_wind_angles_round_trip():
    V, a, b = 120.0, 0.15, -0.08
    assert R.wind_angles(*R.body_velocity(V, a, b)) == pytest.approx((V, a, b), abs=1e-13)
