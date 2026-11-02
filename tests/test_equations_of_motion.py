import numpy as np
import pytest
from scipy.integrate import solve_ivp

from flightdyn.aircraft import MassProperties
from flightdyn.equations_of_motion import (
    PHI,
    PSI,
    THETA,
    XN,
    ZD,
    P,
    Q,
    R,
    U,
    W,
    angular_momentum_ned,
    euler_to_quat_state,
    kinetic_energy,
    quat_to_euler_state,
    rigid_body_derivative,
    rigid_body_derivative_quat,
)

TIGHT = dict(method="DOP853", rtol=1e-11, atol=1e-11)
BODY = MassProperties(mass=1200.0, Ixx=1400.0, Iyy=4000.0, Izz=4800.0, Ixz=250.0)


def integrate(x0, t, mp, force=(0, 0, 0), moment=(0, 0, 0), g=9.80665, quat=False, **kw):
    f = rigid_body_derivative_quat if quat else rigid_body_derivative
    sol = solve_ivp(
        lambda _, x: f(x, force, moment, mp, g), (0.0, t[-1]), x0, t_eval=t, **TIGHT, **kw
    )
    assert sol.success
    return sol.y


def test_rest_in_equilibrium():
    # level attitude, lift balancing weight: every derivative but position is zero
    x = np.zeros(12)
    x[U] = 60.0
    dx = rigid_body_derivative(x, (0, 0, -BODY.mass * 9.80665), (0, 0, 0), BODY)
    assert np.allclose(dx, [0, 0, 0, 0, 0, 0, 0, 0, 0, 60.0, 0, 0], atol=1e-12)


def test_pure_pitching_under_constant_moment():
    # no forces, no gravity, constant pitching moment M about the principal y axis:
    # q = M t / Iyy, theta = M t^2 / (2 Iyy) (Newton's second law for rotation)
    mp = MassProperties(mass=1.0, Ixx=2.0, Iyy=3.0, Izz=4.0)
    M, t = 0.6, np.linspace(0.0, 2.0, 9)
    y = integrate(np.zeros(12), t, mp, moment=(0, M, 0), g=0.0)
    assert np.allclose(y[Q], M * t / mp.Iyy, atol=1e-10)
    assert np.allclose(y[THETA], M * t**2 / (2 * mp.Iyy), atol=1e-10)
    assert np.allclose(y[[P, R, PHI, PSI]], 0.0, atol=1e-12)


def test_ballistic_trajectory_while_tumbling():
    # with gravity alone the CG follows a parabola in Earth axes whatever the body does:
    # x_n = x0 + Vn t, z_d = z0 + Vd t + g t^2 / 2 (here the body also spins and nutates)
    x0 = np.zeros(12)
    x0[[U, 1, W]] = (30.0, -4.0, 6.0)
    x0[[P, Q, R]] = (1.2, -0.7, 0.4)
    x0[[PHI, THETA, PSI]] = (0.3, 0.2, -0.5)
    from flightdyn.rotations import dcm_body_from_ned

    v_ned = dcm_body_from_ned(0.3, 0.2, -0.5).T @ x0[[U, 1, W]]
    t = np.linspace(0.0, 3.0, 13)
    y = integrate(x0, t, BODY)
    g = 9.80665
    assert np.allclose(y[XN], v_ned[0] * t, atol=1e-8)
    assert np.allclose(y[XN + 1], v_ned[1] * t, atol=1e-8)
    assert np.allclose(y[ZD], v_ned[2] * t + 0.5 * g * t**2, atol=1e-8)


def test_torque_free_axisymmetric_precession():
    # body symmetric about z (Ixx = Iyy = I, Izz = I3): r stays constant and (p, q) rotates
    # at Omega = (I3 - I) r / I (Goldstein, Classical Mechanics, ch. 5, Euler's equations)
    I, I3, r0 = 2.0, 5.0, 1.5
    mp = MassProperties(mass=1.0, Ixx=I, Iyy=I, Izz=I3)
    x0 = np.zeros(12)
    x0[[P, Q, R]] = (0.4, 0.0, r0)
    t = np.linspace(0.0, 4.0, 17)
    y = integrate(x0, t, mp, g=0.0)
    om = (I3 - I) * r0 / I
    assert np.allclose(y[R], r0, atol=1e-10)
    assert np.allclose(y[P], 0.4 * np.cos(om * t), atol=1e-9)
    assert np.allclose(y[Q], 0.4 * np.sin(om * t), atol=1e-9)


def test_torque_free_conservation_with_product_of_inertia():
    # without applied moments, angular momentum in inertial axes and kinetic energy are
    # constants of the motion, including the Ixz coupling of an aircraft
    x0 = np.zeros(12)
    x0[[P, Q, R]] = (0.8, 0.3, -0.5)
    x0[[PHI, THETA]] = (0.2, 0.1)
    t = np.linspace(0.0, 10.0, 21)
    y = integrate(x0, t, BODY, g=0.0)
    H0, E0 = angular_momentum_ned(x0, BODY), kinetic_energy(x0, BODY)
    for k in range(len(t)):
        assert np.allclose(angular_momentum_ned(y[:, k], BODY), H0, atol=1e-8)
        assert kinetic_energy(y[:, k], BODY) == pytest.approx(E0, rel=1e-9)


def test_energy_exchange_under_gravity():
    # with gravity only, kinetic energy changes by m g (z_d - z_d0): total energy conserved
    x0 = np.zeros(12)
    x0[[U, W, P, R]] = (50.0, 5.0, 0.3, 0.2)
    x0[THETA] = 0.4
    t = np.linspace(0.0, 5.0, 11)
    y = integrate(x0, t, BODY)
    E = [kinetic_energy(y[:, k], BODY) - BODY.mass * 9.80665 * y[ZD, k] for k in range(len(t))]
    assert np.allclose(E, E[0], rtol=1e-10)


def test_euler_and_quaternion_forms_agree():
    # the same motion integrated in both attitude parameterisations
    x0 = np.zeros(12)
    x0[[U, W, P, Q, R]] = (40.0, 2.0, 0.6, -0.4, 0.3)
    x0[[PHI, THETA, PSI]] = (0.1, -0.2, 0.3)
    t = np.linspace(0.0, 4.0, 9)
    force, moment = (500.0, -200.0, -9000.0), (300.0, -800.0, 150.0)
    ye = integrate(x0, t, BODY, force, moment)
    yq = integrate(euler_to_quat_state(x0), t, BODY, force, moment, quat=True)
    for k in range(len(t)):
        d = quat_to_euler_state(yq[:, k]) - ye[:, k]
        d[PHI : PSI + 1] = np.angle(np.exp(1j * d[PHI : PSI + 1]))  # same angle modulo 2 pi
        assert np.allclose(d, 0.0, atol=1e-7)


def test_quaternion_form_passes_through_vertical():
    # theta = 90 deg is singular for Euler angles but not for quaternions: a loop at constant
    # pitch rate q returns the body to its initial attitude after 2 pi / q
    q = 0.5
    x0 = np.zeros(12)
    x0[Q] = q
    t = np.array([0.0, 2 * np.pi / q])
    yq = integrate(euler_to_quat_state(x0), t, BODY, g=0.0, quat=True)
    assert np.allclose(np.abs(yq[6:10, -1]), [1, 0, 0, 0], atol=1e-9)


def test_gyroscopic_coupling_from_product_of_inertia():
    # with Ixz != 0 a pure rolling moment also produces yaw acceleration:
    # [Ixx -Ixz; -Ixz Izz] [p_dot; r_dot] = [L; 0]  =>  r_dot = Ixz L / (Ixx Izz - Ixz^2)
    L = 1000.0
    dx = rigid_body_derivative(np.zeros(12), (0, 0, 0), (L, 0, 0), BODY, g=0.0)
    det = BODY.Ixx * BODY.Izz - BODY.Ixz**2
    assert dx[P] == pytest.approx(BODY.Izz * L / det)
    assert dx[R] == pytest.approx(BODY.Ixz * L / det)
