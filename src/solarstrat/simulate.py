"""Race simulator: drive a speed plan along the route and track the battery.

Physics per segment: rolling resistance, gravity on slopes, aerodynamic drag
with the wind component, drivetrain efficiency and regenerative braking.
Race rules: driving only 08:00-17:00 local time (the car stops wherever it is
at 17:00), array charging in camp until 19:00 and from 06:30, and a 30 minute
charging halt at every control stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Race, Vehicle
from .route import Segments
from .solar import array_power
from .weather import Weather, headwind

G = 9.81
R_AIR = 287.05
CAMP_STEP_S = 600.0


def air_density(temp_c):
    return 101325.0 / (R_AIR * (np.asarray(temp_c) + 273.15))


def traction_power(v_ms, sin_grade, headwind_ms, rho, veh: Vehicle):
    """Battery power needed to hold speed v (negative = regen into battery), W."""
    cos_grade = np.sqrt(1.0 - sin_grade ** 2)
    v_air = v_ms + headwind_ms
    force = (veh.mass_kg * G * (veh.crr * cos_grade + sin_grade)
             + 0.5 * rho * veh.cda_m2 * v_air * abs(v_air))
    p_wheel = force * v_ms
    return np.where(p_wheel >= 0, p_wheel / veh.drive_eff, p_wheel * veh.regen_eff)


@dataclass
class SimResult:
    arrival_s: float            # UTC seconds since epoch
    start_s: float
    seg_end_s: np.ndarray       # (n,) time at the end of each segment
    soc_end: np.ndarray         # (n,) or (n, N) state of charge after each segment
    seg_min_soc: np.ndarray     # lowest state of charge reached in each segment
    nights: int
    overnight_h: float = 15.0   # 17:00 -> 08:00

    @property
    def elapsed_h(self) -> float:
        return (self.arrival_s - self.start_s) / 3600.0

    @property
    def race_time_h(self) -> float:
        """Elapsed time without the overnight halts: the number the race ranks on."""
        return self.elapsed_h - self.nights * self.overnight_h

    @property
    def min_soc(self):
        return self.seg_min_soc.min(axis=0)


class Simulator:
    """Holds everything that does not depend on the speed plan."""

    def __init__(self, segs: Segments, weather: Weather, race: Race):
        self.segs = segs
        self.race = race
        self.weather = weather
        self.point = weather.point_index(segs.mid_km)
        self.offset_s = race.utc_offset_h * 3600.0
        # weather as one (4, P, T) array on a uniform hourly grid -> fast lookup
        self._wx = np.stack([weather.ghi, weather.temp_c, weather.wind_ms, weather.wind_from_deg])
        self._t0 = float(weather.time_s[0])
        self._nt = len(weather.time_s)
        stop_km = np.array([s.km for s in race.stops])
        self.stops_after = (np.searchsorted(stop_km, segs.end_km, side="right")
                            - np.searchsorted(stop_km, segs.start_km, side="right"))

    # -- helpers -----------------------------------------------------------
    def _wx_at(self, p: int, t: float):
        k = (t - self._t0) / 3600.0
        k = min(max(k, 0.0), self._nt - 1.000001)
        i = int(k)
        f = k - i
        return self._wx[:, p, i] * (1.0 - f) + self._wx[:, p, i + 1] * f

    def _local_midnight(self, t: float) -> float:
        return t - ((t + self.offset_s) % 86400.0)

    def _charge(self, e, p_net, dt, veh):
        de = p_net * dt / 3600.0
        e = e + np.where(de > 0, de * veh.charge_eff, de)
        return np.minimum(e, veh.battery_wh)

    def _stand(self, e, p: int, t: float, t_end: float, veh, load_w):
        """Stand still from t to t_end, charging from the array."""
        while t < t_end - 1e-6:
            dt = min(CAMP_STEP_S, t_end - t)
            ghi, temp, _, _ = self._wx_at(p, t + dt / 2)
            e = self._charge(e, array_power(ghi, temp, veh) - load_w, dt, veh)
            t += dt
        return e

    def _camp(self, e, p: int, t: float, veh):
        """Overnight stop. Returns (time driving resumes, energy)."""
        r = self.race
        midnight = self._local_midnight(t)
        e = self._stand(e, p, t, max(t, midnight + r.charge_end_h * 3600), veh, 0.0)
        next_midnight = midnight + 86400.0
        e = self._stand(e, p, next_midnight + r.charge_start_h * 3600,
                        next_midnight + r.drive_start_h * 3600, veh, 0.0)
        return next_midnight + r.drive_start_h * 3600, e

    # -- main loop -----------------------------------------------------------
    def run(self, speeds_kmh, veh: Vehicle) -> SimResult:
        segs, r = self.segs, self.race
        speeds = np.asarray(speeds_kmh, dtype=float)
        if speeds.shape != (len(segs),):
            raise ValueError(f"need one speed per segment ({len(segs)}), got {speeds.shape}")
        cap = veh.battery_wh
        # number of Monte Carlo worlds = broadcast shape of all vehicle fields
        worlds = np.broadcast_shapes(*(np.shape(v) for v in vars(veh).values()))
        e = np.full(worlds, 1.0) * cap * r.soc_start
        t = r.start.timestamp()
        start = t
        nights = 0
        n = len(segs)
        seg_end_s = np.empty(n)
        soc_end = np.empty((n,) + e.shape)
        seg_min = np.empty((n,) + e.shape)

        for i in range(n):
            p = int(self.point[i])
            v = speeds[i] / 3.6
            remaining = segs.length_m[i]
            lowest = e / cap
            while remaining > 1e-6:
                end_today = self._local_midnight(t) + r.drive_end_h * 3600.0
                if t >= end_today - 1e-6:
                    t, e = self._camp(e, p, t, veh)
                    nights += 1
                    continue
                dt = min(remaining / v, end_today - t)
                ghi, temp, wind, wind_from = self._wx_at(p, t + dt / 2)
                hw = headwind(wind, wind_from, segs.heading_deg[i])
                p_load = traction_power(v, segs.sin_grade[i], hw, air_density(temp), veh) + veh.aux_w
                e = self._charge(e, array_power(ghi, temp, veh) - p_load, dt, veh)
                lowest = np.minimum(lowest, e / cap)
                t += dt
                remaining -= v * dt
            # a control stop that reaches 17:00 is finished the next morning
            halt = r.stop_minutes * 60.0 * int(self.stops_after[i])
            while halt > 1e-6:
                end_today = self._local_midnight(t) + r.drive_end_h * 3600.0
                if t >= end_today - 1e-6:
                    t, e = self._camp(e, p, t, veh)
                    nights += 1
                    continue
                dt = min(halt, end_today - t)
                e = self._stand(e, p, t, t + dt, veh, veh.aux_w)
                t += dt
                halt -= dt
            seg_end_s[i] = t
            soc_end[i] = e / cap
            seg_min[i] = lowest

        return SimResult(arrival_s=t, start_s=start, seg_end_s=seg_end_s,
                         soc_end=soc_end, seg_min_soc=seg_min, nights=nights,
                         overnight_h=24.0 - (r.drive_end_h - r.drive_start_h))


def simulate(segs: Segments, speeds_kmh, weather: Weather, veh: Vehicle, race: Race) -> SimResult:
    return Simulator(segs, weather, race).run(speeds_kmh, veh)
