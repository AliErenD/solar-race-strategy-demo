"""Solar car race strategy: speed planning and risk analysis for the
Darwin-Adelaide route."""
from .config import Race, Vehicle
from .optimize import Plan, constant_speed_plan, optimize_plan
from .risk import Uncertainty, assess, risk_targeted_scaling, sample_worlds
from .route import load_route, make_segments
from .simulate import Simulator, simulate
from .weather import clear_sky_weather, fetch_era5, load_era5

__all__ = [
    "Race", "Vehicle", "Plan", "constant_speed_plan", "optimize_plan", "Uncertainty",
    "assess", "risk_targeted_scaling", "sample_worlds", "load_route", "make_segments",
    "Simulator", "simulate", "clear_sky_weather", "fetch_era5", "load_era5",
]
