import numpy as np
import pytest

from flightdyn.atmosphere import P_SL, RHO_SL, T_SL, isa, pressure_altitude


def test_sea_level():
    # ICAO Doc 7488 / US Standard Atmosphere 1976: defining sea-level values, a0 = 340.294 m/s
    a = isa(0.0)
    assert a.T == T_SL and a.P == P_SL
    assert a.rho == pytest.approx(RHO_SL, rel=1e-5)
    assert a.a == pytest.approx(340.294, abs=1e-3)


# US Standard Atmosphere 1976 (NOAA-S/T 76-1562), Table 4: base values at geopotential altitudes
@pytest.mark.parametrize(
    "H, T, P",
    [(11000.0, 216.65, 22632.06), (20000.0, 216.65, 5474.889), (32000.0, 228.65, 868.0187)],
)
def test_layer_bases_geopotential(H, T, P):
    a = isa(H, geopotential=True)
    assert a.T == pytest.approx(T, abs=1e-9)
    assert a.P == pytest.approx(P, rel=1e-6)


def test_geometric_table_value():
    # US Standard Atmosphere 1976, Table I at Z = 10 000 m (geometric):
    # T = 223.252 K, P = 2.6500E+04 Pa, rho = 4.1351E-01 kg/m^3
    a = isa(10000.0)
    assert a.T == pytest.approx(223.252, abs=1e-3)
    assert a.P == pytest.approx(2.6500e4, rel=5e-5)
    assert a.rho == pytest.approx(0.41351, rel=5e-5)


def test_hydrostatic_balance():
    # dP/dH = -rho g0 must hold everywhere (central difference, off the layer kinks)
    H = np.linspace(250.0, 31750.0, 64)
    dh = 0.5
    dPdH = (isa(H + dh, geopotential=True).P - isa(H - dh, geopotential=True).P) / (2 * dh)
    assert np.allclose(dPdH, -isa(H, geopotential=True).rho * 9.80665, rtol=1e-6)


def test_pressure_continuous_at_layer_boundaries():
    for H in (11000.0, 20000.0):
        lo, hi = isa(H - 1e-6, geopotential=True).P, isa(H + 1e-6, geopotential=True).P
        assert lo == pytest.approx(hi, rel=1e-9)


def test_pressure_altitude_inverts_isa():
    z = np.array([0.0, 1500.0, 10668.0, 11500.0, 18000.0, 25000.0])
    assert np.allclose(pressure_altitude(isa(z).P), z, atol=1e-6)


def test_hot_day_offset():
    # an ISA+dT day keeps the standard pressure and changes density as 1/T
    std, hot = isa(0.0), isa(0.0, dT=15.0)
    assert hot.P == std.P
    assert hot.rho == pytest.approx(std.rho * T_SL / (T_SL + 15.0))


def test_out_of_range():
    with pytest.raises(ValueError):
        isa(40000.0)


def test_below_sea_level():
    # US Standard Atmosphere 1976, Table I at Z = -1000 m (geometric):
    # T = 294.651 K, P = 1.1393E+05 Pa, rho = 1.3470 kg/m^3
    a = isa(-1000.0)
    assert a.T == pytest.approx(294.651, abs=1e-3)
    assert a.P == pytest.approx(1.1393e5, rel=5e-5)
    assert a.rho == pytest.approx(1.3470, rel=5e-5)
