from datetime import datetime, timezone

import numpy as np
import pytest

from solarstrat import Vehicle
from solarstrat.solar import array_power, clear_sky_ghi, sun_cos_zenith


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc).timestamp()


def test_sun_near_zenith_at_solar_noon_in_darwin():
    # Darwin (12.46 S, 130.84 E) on 22 August: solar noon is about 03:18 UTC and the
    # sun's declination is about +11.7 deg, so the zenith angle is about 24 deg.
    cz = sun_cos_zenith(-12.46, 130.84, utc(2022, 8, 22, 3, 18))
    assert np.degrees(np.arccos(cz)) == pytest.approx(24.2, abs=0.7)


def test_sun_below_horizon_at_local_midnight():
    assert sun_cos_zenith(-12.46, 130.84, utc(2022, 8, 22, 15, 18)) < 0


def test_clear_sky_irradiance_is_zero_at_night_and_about_1kw_overhead():
    assert clear_sky_ghi(-0.2) == 0.0
    assert clear_sky_ghi(1.0) == pytest.approx(1098 * np.exp(-0.057), rel=1e-6)


def test_hot_array_produces_less_power():
    veh = Vehicle()
    cool = array_power(1000.0, 0.0, veh)
    hot = array_power(1000.0, 40.0, veh)
    assert hot < cool
    # 40 degC hotter air -> 40 * 0.29 % = 11.6 % less power
    assert hot / cool == pytest.approx(1 + veh.array_temp_coeff * 40 / (1 + veh.array_temp_coeff * 5), rel=1e-6)
