"""Monte Carlo risk analysis of a speed plan.

The weather is taken as known; the car's parameters are not. Each Monte Carlo
"world" draws its own drag, rolling resistance, array output, battery
capacity, drivetrain efficiency and auxiliary load. The plan is driven as is
in every world, and we count how often the battery falls below its minimum.
Scaling all speeds down by a factor alpha buys safety margin at the cost of
arrival time; we search for the largest alpha that meets a target risk.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Vehicle
from .simulate import Simulator


@dataclass(frozen=True)
class Uncertainty:
    # relative standard deviations (0.05 = 5 %)
    cda: float = 0.05
    crr: float = 0.10
    array: float = 0.05     # array efficiency, soiling, irradiance error
    battery: float = 0.03   # usable capacity
    drive_eff: float = 0.01
    aux: float = 0.20


def sample_worlds(veh: Vehicle, unc: Uncertainty, n: int, seed: int = 0) -> Vehicle:
    rng = np.random.default_rng(seed)

    def draw(value, rel):
        return value * np.clip(rng.normal(1.0, rel, n), 1 - 3 * rel, 1 + 3 * rel)

    return replace(
        veh,
        cda_m2=draw(veh.cda_m2, unc.cda),
        crr=draw(veh.crr, unc.crr),
        array_eff=draw(veh.array_eff, unc.array),
        battery_wh=draw(veh.battery_wh, unc.battery),
        drive_eff=np.minimum(draw(veh.drive_eff, unc.drive_eff), 0.99),
        aux_w=draw(veh.aux_w, unc.aux),
    )


@dataclass
class RiskResult:
    violation_probability: float
    min_soc: np.ndarray         # lowest state of charge in each world


def assess(sim: Simulator, speeds_kmh, worlds: Vehicle, soc_min: float) -> RiskResult:
    res = sim.run(speeds_kmh, worlds)
    lowest = np.asarray(res.min_soc)
    return RiskResult(float(np.mean(lowest < soc_min)), lowest)


@dataclass
class ScaledPlan:
    alpha: float
    speeds_kmh: np.ndarray
    risk: RiskResult


def risk_targeted_scaling(sim: Simulator, speeds_kmh, worlds: Vehicle, soc_min: float,
                          target: float = 0.05, tol: float = 1e-3) -> ScaledPlan:
    """Largest alpha in (0, 1] whose scaled plan meets the target risk."""
    v_min = sim.race.min_speed_kmh
    base = np.asarray(speeds_kmh, dtype=float)

    def scaled(a):
        return np.maximum(base * a, v_min)

    risk = assess(sim, scaled(1.0), worlds, soc_min)
    if risk.violation_probability <= target:
        return ScaledPlan(1.0, scaled(1.0), risk)
    lo, hi = 0.5, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if assess(sim, scaled(mid), worlds, soc_min).violation_probability <= target:
            lo = mid
        else:
            hi = mid
    return ScaledPlan(lo, scaled(lo), assess(sim, scaled(lo), worlds, soc_min))
