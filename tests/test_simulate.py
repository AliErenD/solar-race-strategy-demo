from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from solarstrat import Race, Simulator, Vehicle
from solarstrat.config import ACST, ControlStop
from solarstrat.simulate import G, air_density, traction_power
from solarstrat.weather import Weather

from conftest import flat_segments


def dark_still_weather(start, days=8, temp=25.0):
    t = start.timestamp() - 86400 + 3600.0 * np.arange(24 * (days + 2))
    z = np.zeros((2, len(t)))
    return Weather(point_km=np.array([0.0, 100.0]), time_s=t, ghi=z, temp_c=z + temp,
                   wind_ms=z, wind_from_deg=z)


def test_flat_road_power_matches_hand_calculation():
    veh = Vehicle()
    v = 80 / 3.6
    rho = air_density(25.0)
    expected = (veh.mass_kg * G * veh.crr + 0.5 * rho * veh.cda_m2 * v ** 2) * v / veh.drive_eff
    assert traction_power(v, 0.0, 0.0, rho, veh) == pytest.approx(expected)


def test_headwind_costs_power_and_downhill_regenerates():
    veh = Vehicle()
    v, rho = 20.0, 1.2
    assert traction_power(v, 0.0, 5.0, rho, veh) > traction_power(v, 0.0, 0.0, rho, veh)
    assert traction_power(v, -0.03, 0.0, rho, veh) < 0


def test_battery_energy_balance_without_sun():
    race = Race(start=datetime(2022, 8, 22, 9, 0, tzinfo=ACST), stops=())
    segs = flat_segments(n=50)              # 50 km, well inside one driving day
    sim = Simulator(segs, dark_still_weather(race.start), race)
    veh = Vehicle()
    res = sim.run(np.full(50, 72.0), veh)
    hours = 50 / 72.0
    load = traction_power(72 / 3.6, 0.0, 0.0, air_density(25.0), veh) + veh.aux_w
    used_wh = veh.battery_wh * (1 - float(res.soc_end[-1]))
    assert used_wh == pytest.approx(load * hours, rel=1e-6)
    assert res.elapsed_h == pytest.approx(hours, rel=1e-9)


def test_car_only_drives_between_8_and_17_local(coarse_sim, veh):
    res = coarse_sim.run(np.full(len(coarse_sim.segs), 65.0), veh)
    assert res.nights >= 4
    local_h = ((res.seg_end_s + coarse_sim.offset_s) % 86400) / 3600
    assert np.all((local_h >= 8.0 - 1e-6) & (local_h <= 17.0 + 1e-6))


def test_control_stop_at_17_is_finished_next_morning(veh):
    # 30 km at 60 km/h from 16:15 -> arrive at the stop 16:45, 15 min left today
    race = Race(start=datetime(2022, 8, 22, 16, 15, tzinfo=ACST),
                stops=(ControlStop("test", 30.0),))
    segs = flat_segments(n=40)
    sim = Simulator(segs, dark_still_weather(race.start), race)
    res = sim.run(np.full(40, 60.0), veh)
    # driving 40 min + stop 30 min, no time lost across the night
    assert res.nights == 1
    assert res.race_time_h == pytest.approx(40 / 60 + 0.5, abs=1e-6)
    local_h = ((res.seg_end_s[29] + sim.offset_s) % 86400) / 3600
    assert local_h == pytest.approx(8.25, abs=1e-6)     # 15 min served after 08:00


def test_every_control_stop_is_visited_once(coarse_sim):
    assert coarse_sim.stops_after.sum() == len(coarse_sim.race.stops) == 9


def test_slower_plan_arrives_later_with_more_charge(coarse_sim, veh):
    n = len(coarse_sim.segs)
    fast = coarse_sim.run(np.full(n, 75.0), veh)
    slow = coarse_sim.run(np.full(n, 65.0), veh)
    assert slow.arrival_s > fast.arrival_s
    assert slow.min_soc > fast.min_soc


def test_vectorised_worlds_match_single_runs(coarse_sim, veh):
    n = len(coarse_sim.segs)
    speeds = np.full(n, 70.0)
    crr = np.array([0.0025, 0.003, 0.0035])
    batch = coarse_sim.run(speeds, replace(veh, crr=crr))
    for k, c in enumerate(crr):
        single = coarse_sim.run(speeds, replace(veh, crr=float(c)))
        assert batch.min_soc[k] == pytest.approx(float(single.min_soc), abs=1e-12)


def test_wrong_number_of_speeds_is_rejected(coarse_sim, veh):
    with pytest.raises(ValueError):
        coarse_sim.run(np.full(3, 70.0), veh)
