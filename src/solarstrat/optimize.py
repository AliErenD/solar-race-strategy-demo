"""Speed planning.

Two plans are compared:
* constant speed: the fastest single speed that keeps the battery above its
  minimum state of charge for the whole race (found by bisection);
* optimized: one speed per block of road (100 km by default), chosen with
  SLSQP to minimise driving time subject to the battery limit at every block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import Vehicle
from .simulate import SimResult, Simulator


@dataclass
class Plan:
    speeds_kmh: np.ndarray      # one speed per segment
    result: SimResult
    label: str


def constant_speed_plan(sim: Simulator, veh: Vehicle, tol_kmh: float = 0.05) -> Plan:
    r = sim.race
    n = len(sim.segs)

    def feasible(v):
        return sim.run(np.full(n, v), veh).min_soc >= veh.soc_min

    lo, hi = r.min_speed_kmh, r.max_speed_kmh
    if not feasible(lo):
        raise ValueError("battery limit is broken even at the minimum legal speed")
    if feasible(hi):
        lo = hi
    while hi - lo > tol_kmh:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    speeds = np.full(n, lo)
    return Plan(speeds, sim.run(speeds, veh), f"constant {lo:.1f} km/h")


def _blocks(sim: Simulator, block_km: float) -> np.ndarray:
    """Index of the block that each segment belongs to."""
    return (sim.segs.start_km // block_km).astype(int)


def optimize_plan(sim: Simulator, veh: Vehicle, block_km: float = 100.0,
                  start: Plan | None = None, maxiter: int = 100) -> Plan:
    r = sim.race
    blk = _blocks(sim, block_km)
    nb = blk.max() + 1
    block_m = np.bincount(blk, weights=sim.segs.length_m, minlength=nb)
    start = start or constant_speed_plan(sim, veh)
    x0 = np.array([start.speeds_kmh[blk == b].mean() for b in range(nb)])

    def expand(x):
        return x[blk]

    def driving_hours(x):
        return float(np.sum(block_m / 1000.0 / x))

    def driving_hours_grad(x):
        return -block_m / 1000.0 / x ** 2

    cache: dict[bytes, np.ndarray] = {}

    def block_min_soc(x):
        key = x.tobytes()
        if key not in cache:
            res = sim.run(expand(x), veh)
            cache[key] = np.array([res.seg_min_soc[blk == b].min() for b in range(nb)])
        return cache[key]

    sol = minimize(
        driving_hours, x0, jac=driving_hours_grad, method="SLSQP",
        bounds=[(r.min_speed_kmh, r.max_speed_kmh)] * nb,
        constraints=[{"type": "ineq", "fun": lambda x: block_min_soc(x) - veh.soc_min}],
        options={"maxiter": maxiter, "ftol": 1e-6, "eps": 1e-3},
    )
    speeds = expand(np.clip(sol.x, r.min_speed_kmh, r.max_speed_kmh))
    result = sim.run(speeds, veh)
    # SLSQP may stop slightly outside the constraint; never return a worse plan
    if result.min_soc < veh.soc_min - 1e-3 or result.race_time_h > start.result.race_time_h:
        return Plan(start.speeds_kmh, start.result, start.label)
    return Plan(speeds, result, f"optimized ({nb} blocks of {block_km:.0f} km)")
