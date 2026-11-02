"""Internal-consistency checks on the transcribed reference data.

Each check relates two printed numbers through a definition, so a transcription slip in either
shows up. Tolerances follow the three significant figures the reports print.
"""

import numpy as np
import pytest

from flightdyn.atmosphere import isa
from flightdyn.units import FT, G0_FT, KT, SLUG_FT3

from reference import B747_FLIGHT_CONDITIONS, b747, b747_nondimensional, navion


@pytest.mark.parametrize("fc", B747_FLIGHT_CONDITIONS)
def test_b747_dynamic_pressure_and_mach(fc):
    # CR-2144 Table IX-3: Q = rho V^2 / 2 and M = V / a in the standard atmosphere at H
    d = b747(fc)
    atm = isa(d["H"] * FT, geopotential=True)
    rho = atm.rho / SLUG_FT3
    # FC 9 and 10 print Q about 1.1 % above rho V^2 / 2 (175.3 -> 177, 221.9 -> 224) while
    # FC 8 at the same altitude agrees to 0.4 %: an inconsistency in the report, kept as printed
    tol = 1.2e-2 if fc in (9, 10) else 6e-3
    assert 0.5 * rho * d["VTO"] ** 2 == pytest.approx(d["Q"], rel=tol)
    assert d["VTO"] * FT / atm.a == pytest.approx(d["M"], rel=6e-3)
    assert d["VTO"] * FT / KT == pytest.approx(d["VTO_KTAS"], rel=4e-3)


@pytest.mark.parametrize("fc", B747_FLIGHT_CONDITIONS)
def test_b747_thrust_derivatives_are_one_over_mass(fc):
    # Appendix A: a unit thrust inclined i_th to the body x axis, 10 ft below the CG, gives
    # X = cos(i_th)/m, Z = -sin(i_th)/m, M = l_th/Iy per pound (Table IX-3 ITH, LTH)
    d = b747(fc)
    m = d["W"] / G0_FT
    ith = np.radians(d["ITH"])
    assert d["XDTH"] == pytest.approx(np.cos(ith) / m, rel=5e-3)
    assert d["ZDTH"] == pytest.approx(-np.sin(ith) / m, rel=5e-3)
    assert d["MDTH"] == pytest.approx(d["LTH"] / d["IY"], rel=5e-3)


@pytest.mark.parametrize("fc", B747_FLIGHT_CONDITIONS)
def test_b747_sideforce_derivative(fc):
    # Appendix A, p. A-17: Y_beta = V Y_v
    d = b747(fc)
    assert d["YB"] == pytest.approx(d["VTO"] * d["YV"], rel=1e-2)


@pytest.mark.parametrize("fc", (1, 2))
def test_b747_lift_coefficient_is_weight_over_qs(fc):
    # level flight: CL ~ W / (q S); thrust tilted by alpha + i_th carries ~1 % of the weight
    d, c = b747(fc), b747_nondimensional(fc)
    assert c["CL"] == pytest.approx(d["W"] / (d["Q"] * 5500.0), rel=1.5e-2)


def test_navion_consistency():
    # Teper Table X-A: m = W/g, qbar = rho V^2 / 2, M = V/a, CL = W/(qbar S) (Table X-B)
    n = navion()
    assert n["m"] == pytest.approx(n["W"] / G0_FT, rel=2e-3)
    assert n["qbar"] == pytest.approx(0.5 * n["rho"] * n["VTO"] ** 2, rel=2e-3)
    assert n["M"] == pytest.approx(n["VTO"] / n["a"], rel=3e-3)
    # CL is printed to two figures (0.41 for W/(qbar S) = 0.415)
    assert n["CL"] == pytest.approx(n["W"] / (n["qbar"] * n["S"]), abs=0.006)
    assert n["rho"] == pytest.approx(isa(0.0).rho / SLUG_FT3, rel=1e-3)
