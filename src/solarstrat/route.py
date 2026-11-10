"""Route loading and splitting into fixed-length segments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_DIR


@dataclass(frozen=True)
class Route:
    km: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elev_m: np.ndarray

    @property
    def length_km(self) -> float:
        return float(self.km[-1])


@dataclass(frozen=True)
class Segments:
    start_km: np.ndarray
    length_m: np.ndarray
    sin_grade: np.ndarray       # sine of the road slope, + is uphill
    lat: np.ndarray             # segment midpoint
    lon: np.ndarray
    heading_deg: np.ndarray     # direction of travel, 0 = north, 90 = east

    @property
    def mid_km(self) -> np.ndarray:
        return self.start_km + self.length_m / 2000.0

    @property
    def end_km(self) -> np.ndarray:
        return self.start_km + self.length_m / 1000.0

    def __len__(self) -> int:
        return len(self.start_km)


def load_route(path: Path | str = DATA_DIR / "route.csv") -> Route:
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=4)
    return Route(km=data[:, 0], lat=data[:, 1], lon=data[:, 2], elev_m=data[:, 3])


def _heading(lat1, lon1, lat2, lon2) -> np.ndarray:
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    return np.degrees(np.arctan2(dlon, dlat)) % 360.0


def make_segments(route: Route, seg_km: float = 5.0) -> Segments:
    edges = np.arange(0.0, route.length_km, seg_km)
    edges = np.append(edges, route.length_km)
    lat = np.interp(edges, route.km, route.lat)
    lon = np.interp(edges, route.km, route.lon)
    elev = np.interp(edges, route.km, route.elev_m)
    length_m = np.diff(edges) * 1000.0
    sin_grade = np.diff(elev) / np.hypot(length_m, np.diff(elev))
    mid = edges[:-1] + np.diff(edges) / 2
    return Segments(
        start_km=edges[:-1],
        length_m=length_m,
        sin_grade=sin_grade,
        lat=np.interp(mid, route.km, route.lat),
        lon=np.interp(mid, route.km, route.lon),
        heading_deg=_heading(lat[:-1], lon[:-1], lat[1:], lon[1:]),
    )
