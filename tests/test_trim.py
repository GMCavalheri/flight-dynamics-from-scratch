import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from flightdyn.aircraft import ThrustLine
from flightdyn.atmosphere import isa
from flightdyn.equations_of_motion import (
    PHI,
    PSI,
    THETA,
    ZD,
    forces_and_moments,
    state_derivative,
)
from flightdyn.rotations import dcm_body_from_ned
from flightdyn.trim_solver import pitch_attitude, trim
from flightdyn.units import G0

from reference_aircraft import b747, b747_condition, navion, navion_condition

# a simple aircraft: Navion derivatives, thrust along the body x axis through the CG
SIMPLE = navion().with_(thrust=ThrustLine(0.0, 0.0))
V, H = 60.0, 1000.0


def closed_form_level_trim(ac, V, h, gamma):
    """Wind-axis balance solved as one scalar equation in alpha (independent of the 6-DOF):
    T cos a = D + W sin g, L + T sin a = W cos g, Cm = 0 (thrust through the CG)."""
    d, geo = ac.aero, ac.geometry
    atm = isa(h, geopotential=True)
    qS = 0.5 * float(atm.rho) * V * V * geo.S
    dM = V / float(atm.a) - d.mach0
    W = ac.mass.mass * G0

    def parts(a):
        de = -(d.Cm0 + d.Cm_a * (a - d.alpha0) + d.Cm_M * dM) / d.Cm_de
        CL = d.CL0 + d.CL_a * (a - d.alpha0) + d.CL_M * dM + d.CL_de * de
        CD = d.CD0 + d.CD_a * (a - d.alpha0) + d.CD_M * dM + d.CD_de * de
        T = (qS * CD + W * np.sin(gamma)) / np.cos(a)
        return de, qS * CL + T * np.sin(a) - W * np.cos(gamma), T

    a = brentq(lambda a: parts(a)[1], -0.2, 0.4, xtol=1e-15)
    de, _, T = parts(a)
    return a, de, T


@pytest.mark.parametrize("gamma_deg", (0.0, 3.0, -4.0))
def test_level_and_climbing_trim_matches_closed_form(gamma_deg):
    g = np.radians(gamma_deg)
    t = trim(SIMPLE, V, H, gamma=g)
    a, de, T = closed_form_level_trim(SIMPLE, V, H, g)
    assert t.alpha == pytest.approx(a, abs=1e-10)
    assert t.de == pytest.approx(de, abs=1e-10)
    assert t.thrust == pytest.approx(T, rel=1e-9)
    assert t.theta == pytest.approx(a + g, abs=1e-12)  # wings level: theta = alpha + gamma
    assert t.residual < 1e-10


def test_climb_angle_of_velocity_vector():
    g = np.radians(5.0)
    t = trim(SIMPLE, V, H, gamma=g)
    h_dot = -state_derivative(SIMPLE, t.x, t.controls)[ZD]
    assert h_dot == pytest.approx(V * np.sin(g), rel=1e-10)


def test_pitch_attitude_reduces_to_alpha_plus_gamma():
    assert pitch_attitude(0.1, 0.0, 0.0, 0.05) == pytest.approx(0.15, abs=1e-15)


def test_coordinated_turn_balances_centripetal_force():
    # In a level turn at rate psi_dot the CG accelerates at psi_dot z x v_ned (z down), and the
    # non-gravitational force must supply that plus support the weight: an inertial-frame
    # statement, independent of the body-frame equations the trim solves. Coordinated: beta = 0.
    psi_dot = np.radians(6.0)
    t = trim(SIMPLE, V, H, turn_rate=psi_dot)
    assert t.beta == pytest.approx(0.0, abs=1e-12)
    F, _ = forces_and_moments(SIMPLE, t.x, t.controls)
    C = dcm_body_from_ned(t.x[PHI], t.x[THETA], t.x[PSI])
    v_ned = C.T @ t.x[:3]
    F_ned = C.T @ F / SIMPLE.mass.mass
    accel = F_ned + np.array([0.0, 0.0, G0])
    assert np.allclose(accel, np.cross([0.0, 0.0, psi_dot], v_ned), atol=1e-12)
    assert v_ned[2] == pytest.approx(0.0, abs=1e-12)  # level
    # bank angle of the non-gravitational force: tan(mu) = V psi_dot / g
    horizontal = np.hypot(F_ned[0], F_ned[1])
    assert np.arctan2(horizontal, -F_ned[2]) == pytest.approx(np.arctan(V * psi_dot / G0))


@pytest.mark.parametrize("kw", [dict(), dict(turn_rate=np.radians(4.0))])
def test_trim_is_an_equilibrium_of_the_nonlinear_model(kw):
    # holding the trim controls, 20 s of nonlinear flight keep every state but heading and
    # position constant (a climb is steady only instantaneously: density falls as it climbs)
    ac = b747(2)
    c = b747_condition(2)
    t = trim(ac, c["V"], 3000.0, **kw)
    sol = solve_ivp(
        lambda _, x: state_derivative(ac, x, t.controls),
        (0.0, 20.0),
        t.x,
        method="DOP853",
        rtol=1e-11,
        atol=1e-11,
    )
    xf = sol.y[:, -1]
    assert np.allclose(xf[:6], t.x[:6], atol=1e-7)
    assert np.allclose(xf[PHI : THETA + 1], t.x[PHI : THETA + 1], atol=1e-8)
    assert xf[PSI] == pytest.approx(kw.get("turn_rate", 0.0) * 20.0, abs=1e-7)


def test_wings_level_and_zero_sideslip_agree_for_a_symmetric_aircraft():
    # in straight flight a symmetric aircraft needs no sideslip, bank or lateral controls
    ac, c = b747(2), b747_condition(2)
    a = trim(ac, c["V"], 0.0, constraint="zero_sideslip")
    b = trim(ac, c["V"], 0.0, constraint="wings_level")
    assert np.allclose(a.controls, b.controls, atol=1e-9)
    assert abs(b.phi) < 1e-12 and abs(b.beta) < 1e-12


@pytest.mark.parametrize("fc", (1, 2))
def test_b747_trims_near_its_published_reference(fc):
    # The model's Cm0 balances the published thrust line's moment at the reference
    # (reference_aircraft.py), so the elevator stays near its reference setting. The tabulated
    # CL is W/(q S) and leaves out the lift of the thrust, which is inclined alpha + i_th to the
    # flight path: the trim angle of attack is lower by about T sin(alpha + i_th) / (q S CL_a)
    ac, c = b747(fc), b747_condition(fc)
    t = trim(ac, c["V"], c["h"])
    d = ac.aero
    atm = isa(c["h"], geopotential=True)
    qS = 0.5 * float(atm.rho) * c["V"] ** 2 * ac.geometry.S
    thrust_lift = t.thrust * np.sin(t.alpha + ac.thrust.incidence) / (qS * d.CL_a)
    assert abs(np.degrees(t.de)) < 0.5
    assert t.alpha - c["alpha"] == pytest.approx(-thrust_lift, abs=np.radians(0.1))
    # pitch balance: aerodynamic moment plus thrust moment vanish at the trim
    dM = c["V"] / float(atm.a) - d.mach0
    cm = d.Cm0 + d.Cm_a * (t.alpha - d.alpha0) + d.Cm_M * dM + d.Cm_de * t.de
    assert cm + t.thrust * ac.thrust.offset / (qS * ac.geometry.cbar) == pytest.approx(0, abs=1e-12)
    # with Cm0 = 0 (taking "trimmed" to exclude the thrust moment) the elevator must hold it
    t0 = trim(ac.with_(aero=d.with_(Cm0=0.0)), c["V"], c["h"])
    assert abs(t0.de) > 4 * abs(t.de)


def test_navion_trims_at_its_reference():
    # Teper's Navion: stability axes are the body axes (alpha0 = 0), thrust through the CG
    ac, c = navion(), navion_condition()
    t = trim(ac, c["V"], c["h"])
    assert abs(np.degrees(t.alpha)) < 0.1 and abs(np.degrees(t.de)) < 0.1
