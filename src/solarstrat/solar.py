"""Sun position, clear-sky irradiance and solar array output."""
from __future__ import annotations

import numpy as np

SOLAR_CONSTANT = 1361.0  # W/m2


def sun_cos_zenith(lat_deg, lon_deg, utc_s):
    """Cosine of the solar zenith angle (NOAA low-precision formulas).

    utc_s is seconds since the Unix epoch. Returns values <= 0 at night.
    Accuracy is about 0.5 degree, which is enough for energy planning.
    """
    utc_s = np.asarray(utc_s, dtype=float)
    days = utc_s / 86400.0
    gamma = 2 * np.pi * ((days % 365.25) / 365.25)          # fractional year
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    minutes_utc = (utc_s % 86400.0) / 60.0
    true_solar_min = minutes_utc + eqtime + 4.0 * np.asarray(lon_deg)
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)
    lat = np.radians(lat_deg)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)


def clear_sky_ghi(cos_z):
    """Global horizontal irradiance under a clear sky (Haurwitz model), W/m2."""
    cz = np.clip(cos_z, 0.0, None)
    with np.errstate(divide="ignore", over="ignore"):
        ghi = 1098.0 * cz * np.exp(-0.057 / np.where(cz > 0, cz, 1.0))
    return np.where(cz > 0.01, ghi, 0.0)


def array_power(ghi, air_temp_c, vehicle):
    """Electrical power from a flat, car-mounted array, W.

    Cell temperature rises about 30 degC above air at 1000 W/m2.
    """
    cell_temp = air_temp_c + 0.03 * ghi
    temp_factor = 1.0 + vehicle.array_temp_coeff * (cell_temp - 25.0)
    return np.maximum(ghi, 0.0) * vehicle.array_area_m2 * vehicle.array_eff * temp_factor
