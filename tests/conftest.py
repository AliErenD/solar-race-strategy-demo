import numpy as np
import pytest

from solarstrat import Race, Simulator, Vehicle, clear_sky_weather, load_route, make_segments
from solarstrat.route import Segments


@pytest.fixture(scope="session")
def route():
    return load_route()


@pytest.fixture(scope="session")
def race():
    return Race().for_year(2022)


@pytest.fixture(scope="session")
def coarse_sim(route, race):
    """Whole route in 10 km segments under a clear sky: fast and deterministic."""
    return Simulator(make_segments(route, 10.0), clear_sky_weather(route, race.start), race)


@pytest.fixture
def veh():
    return Vehicle()


def flat_segments(n=10, length_m=1000.0, lat=-20.0, lon=134.0, heading=180.0):
    return Segments(start_km=np.arange(n) * length_m / 1000.0, length_m=np.full(n, length_m),
                    sin_grade=np.zeros(n), lat=np.full(n, lat), lon=np.full(n, lon),
                    heading_deg=np.full(n, heading))
