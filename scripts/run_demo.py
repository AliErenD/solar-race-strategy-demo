"""Plan the race for one weather year, analyse the risk, back-test all years.

Usage:
    python scripts/run_demo.py                 # 2022 weather, figures + 10-year table
    python scripts/run_demo.py --year 2019 --no-backtest

Writes docs/img/*.png and docs/results.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solarstrat import (Race, Simulator, Uncertainty, Vehicle, assess, constant_speed_plan,
                        load_era5, load_route, make_segments, optimize_plan,
                        risk_targeted_scaling, sample_worlds)
from solarstrat.config import ACST, ROOT

DOCS = ROOT / "docs"
IMG = DOCS / "img"
# categorical slots 1-3 of the validated default palette, light surface
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a65", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TARGET_RISK = 0.05


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    })


def run_year(segs, year, veh, worlds, block_km=100.0):
    race = Race().for_year(year)
    sim = Simulator(segs, load_era5(year), race)
    const = constant_speed_plan(sim, veh)
    opt = optimize_plan(sim, veh, block_km=block_km, start=const)
    risk_opt = assess(sim, opt.speeds_kmh, worlds, veh.soc_min)
    scaled = risk_targeted_scaling(sim, opt.speeds_kmh, worlds, veh.soc_min, TARGET_RISK)
    safe = sim.run(scaled.speeds_kmh, veh)
    return dict(year=year, sim=sim, const=const, opt=opt, risk_opt=risk_opt,
                scaled=scaled, safe=safe)


def hours_label(h):
    return f"{int(h)} h {round((h % 1) * 60):02d} min"


def arrival_label(ts):
    return datetime.fromtimestamp(ts, ACST).strftime("%d %b, %H:%M")


def plot_plan(r, route, path):
    segs = r["sim"].segs
    x = np.append(segs.start_km, segs.end_km[-1])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    plans = [(r["const"].speeds_kmh, BLUE, "constant speed"),
             (r["opt"].speeds_kmh, ORANGE, "optimized"),
             (r["scaled"].speeds_kmh, AQUA, f"risk-targeted (alpha = {r['scaled'].alpha:.3f})")]
    for v, c, lab in plans:
        ax.stairs(v, x, color=c, linewidth=2, label=lab, baseline=None)
    ax.set_ylabel("speed (km/h)")
    ax.set_title(f"Speed plan, Darwin to Adelaide, {r['year']} weather")
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    ax2.fill_between(route.km, route.elev_m, color=GRID, linewidth=0)
    ax2.plot(route.km, route.elev_m, color=MUTED, linewidth=1)
    ax2.set_ylabel("elevation (m)")
    ax2.set_xlabel("distance from Darwin (km)")
    ax2.set_xlim(0, route.length_km)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_soc(r, veh, path):
    segs = r["sim"].segs
    fig, ax = plt.subplots(figsize=(9, 4))
    lines = [(r["const"].result, BLUE, "constant speed"),
             (r["opt"].result, ORANGE, "optimized"),
             (r["safe"], AQUA, "risk-targeted")]
    for res, c, lab in lines:
        ax.plot(segs.end_km, 100 * np.asarray(res.soc_end), color=c, label=lab)
    ax.axhline(100 * veh.soc_min, color=INK, linewidth=1, linestyle="--")
    ax.text(segs.end_km[-1], 100 * veh.soc_min + 1.5, "minimum 10 %", ha="right",
            va="bottom", color=INK, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, segs.end_km[-1])
    ax.set_ylabel("battery state of charge (%)")
    ax.set_xlabel("distance from Darwin (km)")
    ax.set_title(f"Battery along the route (nominal car), {r['year']} weather")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk(r, worlds_n, veh, path):
    sim = r["sim"]
    fig, ax = plt.subplots(figsize=(9, 4))
    both = 100 * np.concatenate([r["risk_opt"].min_soc, r["scaled"].risk.min_soc])
    bins = np.arange(np.floor(both.min() / 2) * 2, np.ceil(both.max() / 2) * 2 + 2, 2.0)
    p_opt = r["risk_opt"].violation_probability
    p_safe = r["scaled"].risk.violation_probability
    ax.hist(100 * r["risk_opt"].min_soc, bins=bins, histtype="step", color=ORANGE, linewidth=2,
            label=f"optimized: {100 * p_opt:.1f} % of runs below the limit")
    ax.hist(100 * r["scaled"].risk.min_soc, bins=bins, histtype="step", color=AQUA, linewidth=2,
            label=f"risk-targeted: {100 * p_safe:.1f} %")
    ax.axvline(100 * veh.soc_min, color=INK, linewidth=1, linestyle="--")
    ax.text(100 * veh.soc_min - 0.5, ax.get_ylim()[1] * 0.95, "limit", ha="right", va="top",
            color=INK, fontsize=9)
    ax.set_xlabel("lowest state of charge during the race (%); below 0 % = energy shortfall")
    ax.set_ylabel("Monte Carlo runs")
    ax.set_title(f"{worlds_n} Monte Carlo runs with uncertain car parameters, {r['year']} weather")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--worlds", type=int, default=300)
    ap.add_argument("--seg-km", type=float, default=5.0)
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()

    style()
    IMG.mkdir(parents=True, exist_ok=True)
    route = load_route()
    segs = make_segments(route, args.seg_km)
    veh = Vehicle()
    worlds = sample_worlds(veh, Uncertainty(), args.worlds, seed=1)

    r = run_year(segs, args.year, veh, worlds)
    plot_plan(r, route, IMG / "speed_plan.png")
    plot_soc(r, veh, IMG / "battery.png")
    plot_risk(r, args.worlds, veh, IMG / "risk.png")

    out = [f"# Results ({args.worlds} Monte Carlo runs, {args.seg_km:g} km segments)\n",
           f"## {args.year} weather\n",
           "| plan | race time (without nights) | arrival (local) | lowest SoC, nominal car | runs below 10 % SoC |",
           "|---|---|---|---|---|"]
    rows = [("constant speed " + r["const"].label.split()[1] + " km/h", r["const"].result,
             assess(r["sim"], r["const"].speeds_kmh, worlds, veh.soc_min).violation_probability),
            ("optimized", r["opt"].result, r["risk_opt"].violation_probability),
            (f"risk-targeted, alpha = {r['scaled'].alpha:.3f}", r["safe"],
             r["scaled"].risk.violation_probability)]
    for name, res, p in rows:
        out.append(f"| {name} | {hours_label(res.race_time_h)} | {arrival_label(res.arrival_s)} | "
                   f"{100 * float(res.min_soc):.1f} % | {100 * p:.1f} % |")
    print("\n".join(out))

    if not args.no_backtest:
        out += ["", "## Back-test over 10 weather years\n",
                "| year | constant speed | optimized gain | optimized risk | alpha | risk after scaling | time cost |",
                "|---|---|---|---|---|---|---|"]
        for year in range(2016, 2026):
            y = r if year == args.year else run_year(segs, year, veh, worlds)
            gain = (y["const"].result.race_time_h - y["opt"].result.race_time_h) * 60
            cost = (y["safe"].race_time_h - y["opt"].result.race_time_h) * 60
            line = (f"| {year} | {y['const'].speeds_kmh[0]:.1f} km/h | {gain:.0f} min | "
                    f"{100 * y['risk_opt'].violation_probability:.1f} % | {y['scaled'].alpha:.3f} | "
                    f"{100 * y['scaled'].risk.violation_probability:.1f} % | +{cost:.0f} min |")
            out.append(line)
            print(line, flush=True)

    (DOCS / "results.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
