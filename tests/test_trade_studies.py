import numpy as np
import pytest
from scipy.optimize import brentq

from flightdyn.aero_derivatives import body_axis_coefficients
from flightdyn.equations_of_motion import forces_and_moments, level_state
from flightdyn.linearization import lateral_model
from flightdyn.modal_analysis import lateral_modes
from flightdyn.trade_studies import (
    HorizontalTail,
    lateral_dimensional,
    lateral_in_body_axes,
    move_cg,
    neutral_point_shift,
    scale_tail,
    spiral_criterion,
)

from reference_aircraft import b747, b747_condition, navion

AC, COND = b747(1), b747_condition(1)  # alpha0 = 8.5 deg: axis effects are largest


def _fm_about(ac, x, u, r_p):
    """Force and moment about a point offset r_p (body axes) from the model's reference,
    evaluated with that point's velocity v + w x r_p."""
    xp = x.copy()
    xp[:3] = x[:3] + np.cross(x[3:6], r_p)
    F, M = forces_and_moments(ac, xp, u)
    return F, M


def test_move_cg_is_the_rigid_body_transfer():
    # the moved model evaluated at the new CG must give the old model's force at the old
    # reference point (which moves at v + w x r) and its moment plus r x F, to first order
    # in the rates: the error must fall as the rate squared
    dh = 0.12
    moved = move_cg(AC, dh)
    r_p = np.array([dh * AC.geometry.cbar, 0.0, 0.0])  # old reference, from the new CG
    x0 = level_state(COND["V"], COND["alpha"], COND["h"])
    u = np.array([0.02, 0.01, -0.01, 0.0])  # thrust off: its line is treated separately

    def err(s):
        x = x0.copy()
        x[1], x[3:6] = 2.0 * s, np.array([0.05, 0.04, 0.03]) * s
        F_old, M_old = _fm_about(AC.with_(aero=lateral_in_body_axes(AC.aero)), x, u, r_p)
        F_new, M_new = forces_and_moments(moved, x, u)
        target = M_old + np.cross(r_p, F_old)
        return max(np.abs(F_new - F_old).max() / AC.weight,
                   np.abs(M_new - target).max() / (AC.weight * AC.geometry.cbar))  # fmt: skip

    assert err(1.0) < 1e-4
    assert err(1.0) / err(0.5) == pytest.approx(4.0, rel=0.05)


def test_move_cg_composes_and_inverts():
    a = move_cg(move_cg(AC, 0.1), -0.1)
    b = lateral_in_body_axes(AC.aero)
    for k in ("Cm_a", "Cm_q", "CL_q", "CD_q", "Cm0", "Cn_b", "Cn_r", "Cl_r", "CY_r"):
        assert getattr(a.aero, k) == pytest.approx(getattr(b, k), abs=1e-12), k
    assert a.thrust.offset == pytest.approx(AC.thrust.offset, abs=1e-12)


def test_neutral_point_has_zero_static_stability():
    # Cm_a is linear in the CG position with slope CN_a; it vanishes at the neutral point
    hn = neutral_point_shift(AC)
    assert move_cg(AC, hn).aero.Cm_a == pytest.approx(0.0, abs=1e-12)
    cn_a = body_axis_coefficients(AC.aero).CN_a
    assert move_cg(AC, 0.1).aero.Cm_a == pytest.approx(AC.aero.Cm_a + 0.1 * cn_a)
    # static margin falls one-for-one as the CG moves aft
    assert neutral_point_shift(move_cg(AC, 0.1)) == pytest.approx(hn - 0.1)


def test_tail_calibration_and_scaling():
    tail = HorizontalTail.calibrate(AC, lt_c=3.3)
    c = tail.contributions()
    # by construction the tail carries all of Cm_q, the published Cm_adot and Cm_de
    assert (c["Cm_q"], c["Cm_adot"], c["Cm_de"]) == pytest.approx(
        (AC.aero.Cm_q, AC.aero.Cm_adot, AC.aero.Cm_de)
    )
    assert scale_tail(AC, tail, 1.0).aero == AC.aero
    # derivatives are linear in the tail area
    d0, d1, d2 = (scale_tail(AC, tail, k).aero for k in (0.0, 1.0, 2.0))
    for name in c:
        assert getattr(d2, name) - getattr(d1, name) == pytest.approx(
            getattr(d1, name) - getattr(d0, name)
        )
    # a bigger tail moves the neutral point aft
    assert neutral_point_shift(scale_tail(AC, tail, 1.3)) > neutral_point_shift(AC)


def test_spiral_boundary_is_the_spiral_criterion():
    # in stability axes with theta0 = 0 the lateral characteristic polynomial's constant term
    # is (g/V)(L'b N'r - N'b L'r): the spiral root crosses zero exactly where that vanishes
    # (Etkin & Reid ch. 8). Scan dihedral effect at fixed directional stiffness.
    ac, V = navion(), 53.6
    k_nb = 2.5

    def spiral(kl):
        dd = lateral_dimensional(ac, V, 0.0, kl, k_nb)
        return lateral_modes(lateral_model(dd, 0.0).A)["spiral"].eigenvalues[0].real

    def crit(kl):
        return spiral_criterion(lateral_dimensional(ac, V, 0.0, kl, k_nb))

    k_eig = brentq(spiral, 0.05, 3.0, xtol=1e-12)
    k_crit = brentq(crit, 0.05, 3.0, xtol=1e-12)
    assert k_eig == pytest.approx(k_crit, rel=1e-8)
    assert spiral(k_eig * 1.2) < 0 < spiral(k_eig * 0.8)  # more dihedral: stable
