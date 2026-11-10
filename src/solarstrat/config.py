"""Vehicle and race parameters.

The vehicle numbers are illustrative values for a single-seat Challenger-class
solar car. They are not the parameters of any real team's car.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Vehicle:
    # Any field may also be a numpy array; the simulator then runs one
    # "world" per element (used by the Monte Carlo risk analysis).
    mass_kg: float = 270.0          # car + driver + ballast
    cda_m2: float = 0.07            # drag coefficient x frontal area
    crr: float = 0.0030             # rolling resistance coefficient
    drive_eff: float = 0.95         # battery -> wheel (motor + controller)
    regen_eff: float = 0.60         # wheel -> battery when braking downhill
    aux_w: float = 40.0             # telemetry, BMS, fans, lights
    array_area_m2: float = 4.0
    array_eff: float = 0.23         # cell efficiency x MPPT/wiring losses
    array_temp_coeff: float = -0.0029   # power change per degC above 25 degC
    battery_wh: float = 5000.0
    charge_eff: float = 0.96        # energy stored / energy put in
    soc_min: float = 0.10           # never plan below this state of charge


@dataclass(frozen=True)
class ControlStop:
    name: str
    km: float


# Route Notes distances (Darwin = 0 km). Every stop is a 30 minute halt.
WSC_CONTROL_STOPS = (
    ControlStop("Katherine", 322.0),
    ControlStop("Dunmarra", 631.0),
    ControlStop("Tennant Creek", 988.0),
    ControlStop("Barrow Creek", 1212.3),
    ControlStop("Alice Springs", 1496.0),
    ControlStop("Erldunda", 1694.0),
    ControlStop("Coober Pedy", 2181.0),
    ControlStop("Glendambo", 2434.0),
    ControlStop("Port Augusta", 2720.5),
)

ACST = timezone(timedelta(hours=9.5))


@dataclass(frozen=True)
class Race:
    start: datetime = datetime(2027, 8, 22, 8, 0, tzinfo=ACST)
    drive_start_h: float = 8.0      # local clock, driving allowed 08:00-17:00
    drive_end_h: float = 17.0
    charge_end_h: float = 19.0      # array charging allowed until 19:00 ...
    charge_start_h: float = 6.5     # ... and again from 06:30
    stop_minutes: float = 30.0
    min_speed_kmh: float = 60.0     # slower than this on open road is not allowed
    max_speed_kmh: float = 110.0    # South Australian open-road limit
    soc_start: float = 1.0
    stops: tuple[ControlStop, ...] = field(default=WSC_CONTROL_STOPS)

    @property
    def utc_offset_h(self) -> float:
        return self.start.utcoffset().total_seconds() / 3600.0

    def for_year(self, year: int) -> "Race":
        """Same race calendar, moved to another year (for historical weather)."""
        from dataclasses import replace
        return replace(self, start=self.start.replace(year=year))
