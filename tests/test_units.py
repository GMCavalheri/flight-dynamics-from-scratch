import pytest

from flightdyn import units


def test_exact_definitions():
    # NIST SP 811 (2008), Appendix B.9: 1 slug = 1.459 390 E+01 kg, 1 lbf = 4.448 222 E+00 N,
    # 1 lbf/ft^2 = 4.788 026 E+01 Pa, 1 slug/ft^3 = 5.153 788 E+02 kg/m^3
    assert units.SLUG == pytest.approx(14.59390, rel=1e-6)
    assert units.LBF == pytest.approx(4.448222, rel=1e-6)
    assert units.PSF == pytest.approx(47.88026, rel=1e-6)
    assert units.SLUG_FT3 == pytest.approx(515.3788, rel=1e-6)
    assert units.G0_FT == pytest.approx(32.17405, rel=1e-6)


def test_weight_to_mass():
    # a pound-mass weighs one pound-force at standard gravity: W [lbf] / g [ft/s^2] -> slug
    assert 1.0 * units.LBF / units.G0 == pytest.approx(units.LB)
    assert units.LB / units.SLUG == pytest.approx(1.0 / units.G0_FT)
