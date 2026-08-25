#!/usr/bin/env python3
"""Schedule comparison demo: TWAP / depth-proportional / KKT / Almgren-Chriss.

Parameters mirror the notebook's calibration style (tail exponent p=0.45,
flat cost c from a half-spread, S=10,000 shares over 390 minutes) with a
synthetic U-shaped intraday depth curve, since the proprietary quote data
is not in the repo. Regenerates figs/schedule_comparison.png and
figs/schedule_comparison.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from impact_model import allocate_schedule, allocate_schedule_risk_averse, compare_schedules  # noqa: E402

T = 390
minutes = np.arange(T)
# U-shaped intraday depth: deep at the open/close, thinner midday
Dt = 220.0 + 180.0 * ((minutes - T / 2) / (T / 2)) ** 2
c = 0.02        # $/share flat cost (half-spread scale)
p = 0.45        # calibrated tail exponent from the modeling note
S = 10_000.0
sigma_per_minute = 0.02  # ~$0.40 daily vol / sqrt(390)

# small order: fits inside the flat region, so impact cost is c*S for every
# schedule and risk aversion is FREE (front-load hard); large order: the
# concave tail engages and a genuine cost/risk tradeoff appears
tables = []
for label, order in (("small (flat region)", S), ("large (tail engaged)", 150_000.0)):
    t = compare_schedules(Dt, c, p, order, sigma_per_minute, risk_aversions=(1e-6, 1e-5))
    t.insert(0, "order", label)
    tables.append(t)
    print(f"\n== {label}: S={order:,.0f}")
    print(t.drop(columns="order").to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
import pandas as pd
table = pd.concat(tables, ignore_index=True)

figs = Path(__file__).resolve().parent.parent / "figs"
figs.mkdir(exist_ok=True)
table.to_csv(figs / "schedule_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
schedules = {
    "TWAP": np.full(T, S / T),
    "depth-proportional": Dt * (S / Dt.sum()),
    "KKT risk-neutral": allocate_schedule(Dt, c, p, S),
    "AC lam=1e-5": allocate_schedule_risk_averse(Dt, c, p, S, sigma_per_minute, 1e-5),
}
for name, x in schedules.items():
    axes[0].plot(minutes, x, label=name, linewidth=1.4)
    axes[1].plot(minutes, S - np.cumsum(x), label=name, linewidth=1.4)
axes[0].set_title("Shares per minute")
axes[0].set_xlabel("Minute of session")
axes[1].set_title("Remaining inventory")
axes[1].set_xlabel("Minute of session")
axes[0].legend(fontsize=8)
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.suptitle("Execution schedules under the piecewise impact model")
plt.tight_layout()
plt.savefig(figs / "schedule_comparison.png", dpi=150, bbox_inches="tight")
print(f"saved -> {figs}/schedule_comparison.png and .csv")
