import json

import numpy as np
import pytest

from solarstrat import load_era5
from solarstrat.weather import WIND_HEIGHT_FACTOR, headwind, weather_points


@pytest.mark.parametrize("year", range(2016, 2026))
def test_bundled_era5_files_load(route, year):
    w = load_era5(year)
    assert w.ghi.shape == (len(weather_points(route)), len(w.time_s))
    assert np.all(np.diff(w.time_s) == 3600.0)
    assert 700 < w.ghi.max() < 1200          # outback August sun, W/m2
    assert not np.isnan(w.ghi).any()


def test_wind_from_ahead_is_headwind_and_from_behind_is_tailwind():
    # car heading south (180 deg)
    assert headwind(10.0, 180.0, 180.0) == pytest.approx(10.0 * WIND_HEIGHT_FACTOR)
    assert headwind(10.0, 0.0, 180.0) == pytest.approx(-10.0 * WIND_HEIGHT_FACTOR)
    assert headwind(10.0, 90.0, 180.0) == pytest.approx(0.0, abs=1e-9)


def test_missing_values_are_filled(tmp_path):
    data = {"point_km": [0.0], "time": ["2022-08-22T00:00", "2022-08-22T01:00", "2022-08-22T02:00"]}
    for v in ("shortwave_radiation", "temperature_2m", "wind_speed_10m", "wind_direction_10m"):
        data[v] = [[0.0, None, 100.0]]
    p = tmp_path / "w.json"
    p.write_text(json.dumps(data))
    w = load_era5(0, path=p)
    assert w.ghi[0, 1] == pytest.approx(50.0)
