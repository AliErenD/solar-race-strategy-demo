import numpy as np

from solarstrat import constant_speed_plan, optimize_plan


def test_constant_plan_is_the_fastest_feasible_constant_speed(coarse_sim, veh):
    plan = constant_speed_plan(coarse_sim, veh)
    v = plan.speeds_kmh[0]
    assert plan.result.min_soc >= veh.soc_min
    faster = coarse_sim.run(np.full(len(coarse_sim.segs), v + 0.5), veh)
    assert faster.min_soc < veh.soc_min


def test_optimized_plan_is_feasible_legal_and_not_slower(coarse_sim, veh):
    const = constant_speed_plan(coarse_sim, veh)
    opt = optimize_plan(coarse_sim, veh, block_km=500.0, start=const, maxiter=30)
    race = coarse_sim.race
    assert opt.result.min_soc >= veh.soc_min - 1e-3
    assert opt.speeds_kmh.min() >= race.min_speed_kmh
    assert opt.speeds_kmh.max() <= race.max_speed_kmh
    assert opt.result.arrival_s <= const.result.arrival_s
