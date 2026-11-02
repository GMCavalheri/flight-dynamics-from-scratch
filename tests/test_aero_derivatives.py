import numpy as np
import pytest

from flightdyn.aero_derivatives import (
    body_axis_coefficients,
    coefficients,
    dimensional_derivatives,
    stability_to_body_moments,
)
from flightdyn.atmosphere import isa
from flightdyn.equations_of_motion import air_data, forces_and_moments, level_state
from flightdyn.units import FT, SLUG_FT3

from reference import b747 as b747_data
from reference import navion as navion_data
from reference_aircraft import b747, b747_condition, navion, navion_condition

AC = b747(1)  # landing: the largest alpha0 (8.5 deg) makes axis effects visible
COND = b747_condition(1)


def coeffs(d, **kw):
    s = dict(V=70.0, alpha=d.alpha0, beta=0.0, p=0.0, q=0.0, r=0.0, alpha_dot=0.0)
    s.update(mach=d.mach0, de=0.0, da=0.0, dr=0.0)
    s.update(kw)
    return coefficients(d, AC.geometry, **s)


def test_reference_condition_returns_reference_coefficients():
    c = coeffs(AC.aero)
    assert (c.CL, c.CD, c.Cm) == (AC.aero.CL0, AC.aero.CD0, AC.aero.Cm0)
    assert (c.CY, c.Cl, c.Cn) == (0.0, 0.0, 0.0)


def test_stability_to_body_rotation_is_orthogonal():
    a = 0.3
    Cl, Cn = stability_to_body_moments(0.7, -0.2, a)
    assert np.hypot(Cl, Cn) == pytest.approx(np.hypot(0.7, -0.2))
    assert stability_to_body_moments(Cl, Cn, -a) == pytest.approx((0.7, -0.2))


def _diff(f, h=1e-6):
    return (f(h) - f(-h)) / (2 * h)


def test_appendix_b_matches_rotated_model():
    # CR-2144 Appendix B transformation vs numerical derivatives of the rotated model:
    # CN = CL cos a + CD sin a, CX = CD cos a - CL sin a, both taken at the instantaneous alpha
    d, bc = AC.aero, body_axis_coefficients(AC.aero)
    V = 70.0
    b2v = AC.geometry.b / (2 * V)

    def CN(c, a):
        return c.CL * np.cos(a) + c.CD * np.sin(a)

    def CX(c, a):
        return c.CD * np.cos(a) - c.CL * np.sin(a)

    a0 = d.alpha0
    assert _diff(lambda h: CN(coeffs(d, alpha=a0 + h), a0 + h)) == pytest.approx(bc.CN_a)
    assert _diff(lambda h: CX(coeffs(d, alpha=a0 + h), a0 + h)) == pytest.approx(bc.CX_a)
    assert _diff(lambda h: coeffs(d, beta=h).Cl) == pytest.approx(bc.Cl_b)
    assert _diff(lambda h: coeffs(d, beta=h).Cn) == pytest.approx(bc.Cn_b)
    assert _diff(lambda h: coeffs(d, p=h / b2v).Cl) == pytest.approx(bc.Cl_p)
    assert _diff(lambda h: coeffs(d, r=h / b2v).Cl) == pytest.approx(bc.Cl_r)
    assert _diff(lambda h: coeffs(d, p=h / b2v).Cn) == pytest.approx(bc.Cn_p)
    assert _diff(lambda h: coeffs(d, r=h / b2v).Cn) == pytest.approx(bc.Cn_r)
    assert _diff(lambda h: coeffs(d, da=h).Cn) == pytest.approx(bc.Cn_da)


def test_body_axes_derivatives_pass_through_unchanged():
    d = AC.aero.with_(lateral_axes="body")
    bc = body_axis_coefficients(d)
    assert (bc.Cl_p, bc.Cn_r, bc.Cl_b) == (d.Cl_p, d.Cn_r, d.Cl_b)


def test_dimensional_derivatives_are_force_derivatives():
    # Appendix A dimensional derivatives must equal finite differences of the model's
    # body-axis force and moment (per unit mass or inertia) at the reference condition
    atm = isa(COND["h"], geopotential=True)
    rho = float(atm.rho)
    # reference Mach exactly V/a (the report prints 0.198 for 0.19795), so the model state and
    # the Appendix A formulas describe the same point
    ac = AC.with_(aero=AC.aero.with_(mach0=COND["V"] / float(atm.a)))
    dd = dimensional_derivatives(ac.aero, ac.geometry, ac.mass, rho, COND["V"])
    x0 = level_state(COND["V"], COND["alpha"], COND["h"])
    u0 = np.zeros(4)
    m, Iy = AC.mass.mass, AC.mass.Iyy

    def fm(dx=None, du=None, ad=0.0):
        x, u = x0.copy(), u0.copy()
        if dx:
            x[dx[0]] += dx[1]
        if du:
            u[du[0]] += du[1]
        return forces_and_moments(ac, x, u, ad)

    def d_dx(i, comp, out, h=1e-4):
        return (fm(dx=(i, h))[out][comp] - fm(dx=(i, -h))[out][comp]) / (2 * h)

    assert d_dx(0, 0, 0) / m == pytest.approx(dd.Xu, rel=1e-6)
    assert d_dx(2, 0, 0) / m == pytest.approx(dd.Xw, rel=1e-6)
    assert d_dx(0, 2, 0) / m == pytest.approx(dd.Zu, rel=1e-6)
    assert d_dx(2, 2, 0) / m == pytest.approx(dd.Zw, rel=1e-6)
    assert d_dx(4, 2, 0) / m == pytest.approx(dd.Zq, rel=1e-6)
    assert d_dx(4, 0, 0) / m == pytest.approx(dd.Xq, rel=1e-6)
    assert d_dx(0, 1, 1) / Iy == pytest.approx(dd.Mu, rel=1e-6)
    assert d_dx(2, 1, 1) / Iy == pytest.approx(dd.Mw, rel=1e-6)
    assert d_dx(4, 1, 1) / Iy == pytest.approx(dd.Mq, rel=1e-6)
    # alpha_dot = w_dot U0 / V^2 at the reference
    k = dd.U0 / COND["V"] ** 2
    dZ = (fm(ad=1e-3)[0][2] - fm(ad=-1e-3)[0][2]) / 2e-3
    dM = (fm(ad=1e-3)[1][1] - fm(ad=-1e-3)[1][1]) / 2e-3
    assert dZ * k / m == pytest.approx(dd.Zwdot, rel=1e-6)
    assert dM * k / Iy == pytest.approx(dd.Mwdot, rel=1e-6)
    dMde = (fm(du=(0, 1e-4))[1][1] - fm(du=(0, -1e-4))[1][1]) / 2e-4
    assert dMde / Iy == pytest.approx(dd.Mde, rel=1e-6)
    # lateral: unprimed derivatives are body-axis moments per unit inertia
    V = COND["V"]
    assert d_dx(1, 1, 0) / m * V == pytest.approx(dd.Yb, rel=1e-6)
    assert d_dx(1, 0, 1) * V / AC.mass.Ixx == pytest.approx(dd.Lb, rel=1e-5)
    assert d_dx(3, 0, 1) / AC.mass.Ixx == pytest.approx(dd.Lp, rel=1e-6)
    assert d_dx(5, 2, 1) / AC.mass.Izz == pytest.approx(dd.Nr, rel=1e-6)


def test_air_data_of_level_state():
    x = level_state(COND["V"], COND["alpha"], COND["h"])
    a = air_data(x)
    assert (a["V"], a["alpha"], a["beta"]) == pytest.approx((COND["V"], COND["alpha"], 0.0))


# --- validation against the report's own dimensional derivatives ---------------------------
# CR-2144 Tables IX-1/IX-2 (nondimensional, stability axes) + Table IX-3 (mass, flight
# condition) -> Appendix A formulas -> compare with Tables IX-4 and IX-8 (printed to 3 figures).
# Model units are SI; the scale converts to the report's ft-based units.
LON = [
    ("XW", "Xw", 1),
    ("ZW", "Zw", 1),
    ("MW", "Mw", FT),
    ("ZWD", "Zwdot", 1),
    ("ZQ", "Zq", 1 / FT),
    ("MWD", "Mwdot", FT),
    ("MQ", "Mq", 1),
    ("XDE", "Xde", 1 / FT),
    ("ZDE", "Zde", 1 / FT),
    ("MDE", "Mde", 1),
    ("XU*", "Xu", 1),
    ("ZU*", "Zu", 1),
    ("MU*", "Mu", FT),
]
LAT = [
    ("YV", "Yv"),
    ("Y*DR", "Ystar_dr"),
    ("LB'", "Lb_p"),
    ("NB'", "Nb_p"),
    ("LP'", "Lp_p"),
    ("NP'", "Np_p"),
    ("LR'", "Lr_p"),
    ("NR'", "Nr_p"),
    ("L'DA", "Lda_p"),
    ("N'DA", "Nda_p"),
    ("L'DR", "Ldr_p"),
    ("N'DR", "Ndr_p"),
]


def _b747_dimensional(fc, axes="stability"):
    ac, cond = b747(fc, axes), b747_condition(fc)
    rho = float(isa(cond["h"], geopotential=True).rho)
    return dimensional_derivatives(ac.aero, ac.geometry, ac.mass, rho, cond["V"])


@pytest.mark.parametrize("fc", (1, 2))
def test_b747_dimensional_derivatives_match_cr2144(fc):
    # agreement within 1 %: inputs and outputs are both printed to 3 figures, so rounding alone
    # reaches 0.7 % (L'_b at fc 2). M_u* and M_w agree only because the model's Cm0 balances
    # the published thrust moment at the reference (reference_aircraft.py); with Cm0 = 0 they
    # are off by a factor 1.6-3 and by 1.7 %. X_u* and Z_u* also contain the report's thrust-
    # speed term T_u, which the model lacks; they agree without it. One exception: the table's
    # Z_q equals -rho S cbar U0 CN_q / (4 m), cos(alpha0) below the V form the force model
    # gives (1.1 % at fc 1).
    dd, ref = _b747_dimensional(fc), b747_data(fc)
    for key, attr, scale in LON:
        tol = 1.2e-2 if key == "ZQ" else 1e-2
        assert getattr(dd, attr) * scale == pytest.approx(ref[key], rel=tol), key
    for key, attr in LAT:
        assert getattr(dd, attr) == pytest.approx(ref[key], rel=1e-2), key


def test_b747_lateral_derivatives_are_stability_axis():
    # Tables IX-1/IX-2 do not name their axis system. Read as stability-axis derivatives they
    # reproduce Table IX-8; read as body-axis ones they miss by up to 115 % (L'_dr)
    dd = _b747_dimensional(1, "body")
    ref = b747_data(1)
    errors = [abs(getattr(dd, attr) / ref[key] - 1) for key, attr in LAT if ref[key]]
    assert max(errors) > 0.5


def test_navion_dimensional_derivatives_scale_with_wing_area():
    # Teper's Table X-D/X-E dimensional derivatives are uniformly 2.22 % larger (moments) and
    # 2.31 % larger (forces) than Table X-B/X-C nondimensional data with Table X-A's
    # rho, S = 180 ft^2 and V give. 1.0222 = 184/180, i.e. Table X-D was computed with
    # S = 184 ft^2; the extra 0.08 % on forces is the printed mass 85.4 slug vs W/g = 85.47.
    n, ac, c = navion_data(), navion(), navion_condition()
    dd = dimensional_derivatives(ac.aero, ac.geometry, ac.mass, n["rho"] * SLUG_FT3, c["V"])
    moments = [("Mw", dd.Mw * FT), ("Mq", dd.Mq), ("L'p", dd.Lp_p), ("N'b", dd.Nb_p)]
    forces = [("Zw", dd.Zw), ("Xu", dd.Xu), ("Yv", dd.Yv), ("Zde", dd.Zde / FT)]
    for key, value in moments:
        assert n[key] / value == pytest.approx(184 / 180, abs=3e-4), key
    for key, value in forces:
        assert n[key] / value == pytest.approx(184 / 180 * 85.47 / 85.4, abs=5e-4), key


def test_b747_cm0_is_what_reproduces_m_u():
    # with the aerodynamic Cm0 = 0 at the reference, M_u* misses the table by 1.6-3x
    for fc in (1, 2):
        ac, cond = b747(fc), b747_condition(fc)
        ac = ac.with_(aero=ac.aero.with_(Cm0=0.0))
        rho = float(isa(cond["h"], geopotential=True).rho)
        dd = dimensional_derivatives(ac.aero, ac.geometry, ac.mass, rho, cond["V"])
        assert dd.Mu * FT / b747_data(fc)["MU*"] > 1.5
