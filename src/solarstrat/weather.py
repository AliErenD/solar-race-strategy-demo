"""Weather along the route: ERA5 reanalysis from the Open-Meteo archive API.

Weather is stored at a few points along the route (every 250 km) as hourly
series. A lookup uses the nearest point in distance and linear interpolation
in time.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import DATA_DIR
from .route import Route
from .solar import clear_sky_ghi, sun_cos_zenith

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ("shortwave_radiation", "temperature_2m", "wind_speed_10m", "wind_direction_10m")
WIND_HEIGHT_FACTOR = 0.75   # 10 m wind -> wind at car height (log profile, open terrain)


@dataclass(frozen=True)
class Weather:
    point_km: np.ndarray        # (P,) route distance of each weather point
    time_s: np.ndarray          # (T,) UTC seconds since epoch, hourly
    ghi: np.ndarray             # (P, T) W/m2
    temp_c: np.ndarray          # (P, T)
    wind_ms: np.ndarray         # (P, T) at 10 m
    wind_from_deg: np.ndarray   # (P, T) meteorological direction

    def point_index(self, km) -> np.ndarray:
        mids = (self.point_km[1:] + self.point_km[:-1]) / 2
        return np.searchsorted(mids, km)

    def at(self, point: int, t_s: float) -> tuple[float, float, float, float]:
        """(ghi, temp, wind speed, wind-from direction) at one point and time."""
        t = self.time_s
        return (float(np.interp(t_s, t, self.ghi[point])),
                float(np.interp(t_s, t, self.temp_c[point])),
                float(np.interp(t_s, t, self.wind_ms[point])),
                float(np.interp(t_s, t, self.wind_from_deg[point])))


def headwind(wind_ms, wind_from_deg, heading_deg):
    """Wind component against the car at car height, m/s (+ = headwind)."""
    return WIND_HEIGHT_FACTOR * wind_ms * np.cos(np.radians(wind_from_deg - heading_deg))


def weather_points(route: Route, spacing_km: float = 250.0) -> np.ndarray:
    pts = np.arange(0.0, route.length_km, spacing_km)
    return np.append(pts, route.length_km)


def fetch_era5(route: Route, year: int, out: Path | None = None,
               first_day: str = "08-21", last_day: str = "08-31") -> Path:
    """Download hourly ERA5 data for the weather points and save it as JSON."""
    km = weather_points(route)
    lat = np.interp(km, route.km, route.lat)
    lon = np.interp(km, route.km, route.lon)
    query = urllib.parse.urlencode({
        "latitude": ",".join(f"{x:.4f}" for x in lat),
        "longitude": ",".join(f"{x:.4f}" for x in lon),
        "start_date": f"{year}-{first_day}",
        "end_date": f"{year}-{last_day}",
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "models": "era5",
    })
    with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=60) as resp:
        payload = json.load(resp)
    if isinstance(payload, dict):
        payload = [payload]
    if len(payload) != len(km):
        raise RuntimeError(f"expected {len(km)} locations, got {len(payload)}")
    data = {
        "source": "ERA5 reanalysis via Open-Meteo Historical Weather API (CC BY 4.0)",
        "point_km": km.tolist(),
        "time": payload[0]["hourly"]["time"],
    }
    for var in HOURLY_VARS:
        data[var] = [[round(v, 1) if v is not None else None for v in p["hourly"][var]]
                     for p in payload]
    out = out or DATA_DIR / "weather" / f"era5_{year}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    return out


def _fill_gaps(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    for row in a:
        bad = np.isnan(row)
        if bad.all():
            raise ValueError("weather series has no data")
        if bad.any():
            idx = np.arange(len(row))
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
    return a


def load_era5(year: int, path: Path | None = None) -> Weather:
    path = path or DATA_DIR / "weather" / f"era5_{year}.json"
    data = json.loads(Path(path).read_text())
    time_s = np.array([datetime.fromisoformat(t).replace(tzinfo=timezone.utc).timestamp()
                       for t in data["time"]])
    arr = {v: _fill_gaps(np.array(data[v], dtype=float)) for v in HOURLY_VARS}
    # ERA5 radiation is the mean over the hour ending at the time stamp;
    # move it to the middle of that hour.
    return Weather(point_km=np.array(data["point_km"]), time_s=time_s - 1800.0,
                   ghi=arr["shortwave_radiation"], temp_c=arr["temperature_2m"],
                   wind_ms=arr["wind_speed_10m"], wind_from_deg=arr["wind_direction_10m"])


def clear_sky_weather(route: Route, start: datetime, days: int = 8,
                      temp_c: float = 25.0) -> Weather:
    """Synthetic weather: clear sky everywhere, no wind. Used offline and in tests."""
    km = weather_points(route)
    lat = np.interp(km, route.km, route.lat)
    lon = np.interp(km, route.km, route.lon)
    t0 = (start - timedelta(days=1)).timestamp()
    time_s = t0 + 3600.0 * np.arange(24 * (days + 2))
    ghi = clear_sky_ghi(sun_cos_zenith(lat[:, None], lon[:, None], time_s[None, :]))
    shape = ghi.shape
    return Weather(point_km=km, time_s=time_s, ghi=ghi, temp_c=np.full(shape, temp_c),
                   wind_ms=np.zeros(shape), wind_from_deg=np.zeros(shape))
