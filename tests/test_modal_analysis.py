import dataclasses

import numpy as np
import pytest

from flightdyn.aero_derivatives import dimensional_derivatives, to_stability_axes
from flightdyn.atmosphere import isa
from flightdyn.linearization import lateral_model, longitudinal_model
from flightdyn.modal_analysis import (
    Mode,
    approximations,
    lateral_modes,
    longitudinal_modes,
    mode_shape,
)
from flightdyn.units import G0_FT

from reference import B747_FLIGHT_CONDITIONS, b747
from reference import navion as navion_data
from reference_aircraft import b747 as b747_aircraft
from reference_aircraft import (
    b747_condition,
    b747_table_derivatives,
    navion_table_derivatives,
)


def second_order_block(wn, zeta):
    return np.array([[0.0, 1.0], [-wn * wn, -2 * zeta * wn]])


def test_mode_characteristics_of_a_known_pair():
    # s^2 + 2 zeta wn s + wn^2: period 2 pi / (wn sqrt(1 - zeta^2)),
    # time to half amplitude ln 2 / (zeta wn)
    wn, zeta = 2.0, 0.3
    m = Mode("x", tuple(np.linalg.eigvals(second_order_block(wn, zeta))))
    assert (m.wn, m.zeta) == pytest.approx((wn, zeta))
    assert m.period == pytest.approx(2 * np.pi / (wn * np.sqrt(1 - zeta**2)))
    assert m.time_to_half == pytest.approx(np.log(2) / (zeta * wn))


def test_overdamped_pair_and_first_order_mode():
    m = Mode("x", (complex(-1.0), complex(-4.0)))
    assert (m.wn, m.zeta) == pytest.approx((2.0, 1.25))  # wn^2 = 4, 2 zeta wn = 5
    s = Mode("spiral", (complex(0.02),))
    assert s.time_constant == pytest.approx(-50.0)  # divergent
    assert s.time_to_half == pytest.approx(-np.log(2) / 0.02)  # negative: time to double


def test_classification_of_synthetic_systems():
    A = np.zeros((4, 4))
    A[:2, :2], A[2:, 2:] = second_order_block(1.5, 0.6), second_order_block(0.1, 0.05)
    m = longitudinal_modes(A)
    assert (m["short_period"].wn, m["short_period"].zeta) == pytest.approx((1.5, 0.6))
    assert (m["phugoid"].wn, m["phugoid"].zeta) == pytest.approx((0.1, 0.05))
    B = np.zeros((4, 4))
    B[:2, :2], B[2, 2], B[3, 3] = second_order_block(1.2, 0.1), -1.5, 0.01
    m = lateral_modes(B)
    assert m["roll"].eigenvalues[0].real == pytest.approx(-1.5)
    assert m["spiral"].eigenvalues[0].real == pytest.approx(0.01)
    assert (m["dutch_roll"].wn, m["dutch_roll"].zeta) == pytest.approx((1.2, 0.1))


def test_mode_shape_is_an_eigenvector():
    A = np.array([[-0.5, 1.0, 0.0], [-2.0, -0.3, 0.1], [0.0, 0.2, -1.0]])
    lam = np.linalg.eigvals(A)[0]
    v = mode_shape(A, lam, normalise=1)
    assert np.allclose(A @ v, lam * v) and v[1] == pytest.approx(1.0)


# --- approximations: exact in the limit where their neglected couplings vanish -------------
ac, c = b747_aircraft(2), b747_condition(2)
DD = dimensional_derivatives(
    ac.aero, ac.geometry, ac.mass, float(isa(c["h"], geopotential=True).rho), c["V"]
)


def test_short_period_approximation_exact_when_speed_decouples():
    dd = dataclasses.replace(DD, Zu=0.0, Mu=0.0, Mudot=0.0, Zudot=0.0, Zwdot=0.0)
    A = longitudinal_model(dd, 0.0, g=0.0).A
    exact = sorted(np.linalg.eigvals(A[1:3, 1:3]), key=lambda z: z.imag)
    approx = approximations.short_period(dd)
    assert np.allclose(sorted(approx.eigenvalues, key=lambda z: z.imag), exact)


def test_roll_and_dutch_roll_approximations_exact_when_roll_decouples():
    # with no roll coupling (L'b = L'r = N'p = 0), no gravity and stability axes (U0 = V)
    dd = dataclasses.replace(DD, Lb_p=0.0, Lr_p=0.0, Np_p=0.0, Yp=0.0, Yr=0.0, U0=DD.V, W0=0.0)
    m = lateral_modes(lateral_model(dd, 0.0, g=0.0).A)
    exact_dr = m["dutch_roll"]
    assert approximations.dutch_roll(dd).wn == pytest.approx(exact_dr.wn)
    assert approximations.dutch_roll(dd).zeta == pytest.approx(exact_dr.zeta)
    assert approximations.roll_subsidence(dd).eigenvalues[0] == pytest.approx(dd.Lp_p)


def test_lanchester_phugoid_closed_form():
    m = approximations.phugoid_lanchester(V=100.0, CL=0.5, CD=0.05, g=9.81)
    assert m.wn == pytest.approx(np.sqrt(2) * 9.81 / 100.0)
    assert m.zeta == pytest.approx(0.05 / (np.sqrt(2) * 0.5))


# --- validation: the reports' dimensional derivatives -> their published roots --------------
def _modes(dd, theta0, thrust=(0.0, 0.0, 0.0)):
    lon = longitudinal_modes(longitudinal_model(dd, theta0, thrust, g=G0_FT).A)
    lat = lateral_modes(lateral_model(dd, theta0, g=G0_FT).A)
    return dict(
        ZETA_PH=lon["phugoid"].zeta,
        OMEGA_PH=lon["phugoid"].wn,
        ZETA_SP=lon["short_period"].zeta,
        OMEGA_SP=lon["short_period"].wn,
        INV_T_SPIRAL=-lat["spiral"].eigenvalues[0].real,
        INV_T_ROLL=-lat["roll"].eigenvalues[0].real,
        ZETA_DR=lat["dutch_roll"].zeta,
        OMEGA_DR=lat["dutch_roll"].wn,
    )


def test_navion_modes_reproduce_teper_to_printed_digits():
    # Teper (1969) Tables X-D/X-E -> Tables X-F/X-G: every printed digit (4-5 figures)
    n = navion_data()
    m = _modes(navion_table_derivatives(), 0.0)
    keys = dict(
        ZETA_PH="zeta_ph",
        OMEGA_PH="omega_ph",
        ZETA_SP="zeta_sp",
        OMEGA_SP="omega_sp",
        INV_T_SPIRAL="inv_T_spiral",
        INV_T_ROLL="inv_T_roll",
        ZETA_DR="zeta_dr",
        OMEGA_DR="omega_dr",
    )
    for k, ref in keys.items():
        assert m[k] == pytest.approx(n[ref], rel=1.5e-3), k


@pytest.mark.parametrize("fc", B747_FLIGHT_CONDITIONS)
def test_b747_modes_reproduce_cr2144(fc):
    # CR-2144 Tables IX-4/IX-8 (3 figures) -> Tables IX-5/IX-9. Within 1.2 %, except fc 5's
    # spiral (1.8 %) and fc 7's phugoid, a nearly critically damped pair (zeta 0.32,
    # wn 0.0098 rad/s); both lie inside the band their roots span when the printed
    # derivatives vary within their rounding (fc 7: -11 % to +0.5 % in zeta, -2.6 % to
    # +9.7 % in wn; validation/mode_validation.py)
    d = b747(fc)
    dd, thrust = b747_table_derivatives(fc)
    m = _modes(dd, np.radians(d["ALPHA"] + d["GAMMA"]), thrust)
    for k, value in m.items():
        tol = 1.2e-2
        if fc == 7 and k in ("ZETA_PH", "OMEGA_PH"):
            tol = 8e-2
        if fc == 5 and k == "INV_T_SPIRAL":
            tol = 2e-2
        assert value == pytest.approx(d[k], rel=tol), k


def test_b747_phugoid_needs_the_implied_alpha_dot_terms():
    # without Z_udot, M_udot and X_q (CR-2144 Appendix C says they are neglected) the
    # published phugoid damping is missed by 7-9 % at fc 1 and 2; with them it is not
    for fc in (1, 2):
        d = b747(fc)
        th0 = np.radians(d["ALPHA"])
        bare = _modes(b747_table_derivatives(fc, implied_terms=False)[0], th0)["ZETA_PH"]
        full = _modes(b747_table_derivatives(fc)[0], th0)["ZETA_PH"]
        assert abs(bare / d["ZETA_PH"] - 1) > 0.07
        assert abs(full / d["ZETA_PH"] - 1) < 0.012


def test_modes_do_not_depend_on_the_axis_system():
    # the same aircraft described in body or stability axes has the same eigenvalues
    rho = float(isa(c["h"], geopotential=True).rho)
    st = to_stability_axes(ac)
    dd_s = dimensional_derivatives(st.aero, st.geometry, st.mass, rho, c["V"])
    assert dd_s.W0 == 0.0
    for A_b, A_s in (
        (longitudinal_model(DD, c["alpha"]).A, longitudinal_model(dd_s, 0.0).A),
        (lateral_model(DD, c["alpha"]).A, lateral_model(dd_s, 0.0).A),
    ):
        eb = np.sort_complex(np.linalg.eigvals(A_b))
        es = np.sort_complex(np.linalg.eigvals(A_s))
        assert np.allclose(eb, es, rtol=1e-10, atol=1e-12)
