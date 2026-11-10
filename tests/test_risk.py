import numpy as np
import pytest

from solarstrat import Uncertainty, assess, constant_speed_plan, risk_targeted_scaling, sample_worlds


def test_worlds_are_centred_on_the_nominal_car(veh):
    w = sample_worlds(veh, Uncertainty(), 4000, seed=3)
    assert w.crr.shape == (4000,)
    assert np.mean(w.crr) == pytest.approx(veh.crr, rel=0.01)
    assert np.std(w.cda_m2) / veh.cda_m2 == pytest.approx(0.05, rel=0.1)
    assert np.all(w.drive_eff < 1.0)


def test_same_seed_gives_same_worlds(veh):
    a = sample_worlds(veh, Uncertainty(), 50, seed=7)
    b = sample_worlds(veh, Uncertainty(), 50, seed=7)
    assert np.array_equal(a.array_eff, b.array_eff)


def test_plan_on_the_limit_is_risky_and_scaling_fixes_it(coarse_sim, veh):
    plan = constant_speed_plan(coarse_sim, veh)
    worlds = sample_worlds(veh, Uncertainty(), 200, seed=1)
    risk = assess(coarse_sim, plan.speeds_kmh, worlds, veh.soc_min)
    assert risk.violation_probability > 0.2       # no margin left in the nominal plan

    scaled = risk_targeted_scaling(coarse_sim, plan.speeds_kmh, worlds, veh.soc_min, target=0.05)
    assert scaled.alpha < 1.0
    assert scaled.risk.violation_probability <= 0.05
    later = coarse_sim.run(scaled.speeds_kmh, veh)
    assert later.arrival_s > plan.result.arrival_s
