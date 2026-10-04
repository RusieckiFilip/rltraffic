"""P8.2 gate G3: two interpretive checks on the timing run's records, imports nothing from the project.

1. Warm-up: is the registered window (decisions 20-359 of each episode) past the machine's own start-of-episode
   transient? Per cell: median(decisions 20-39) / median(decisions 120-359), worst episode; and the median and the
   nearest-rank p95 over decisions 120-359 against the registered 20-359 figures.
2. Variability: per cell, the three episodes' medians; and, within groups of rows that run the SAME computation (the
   same network at the same context length on the same device), the spread of their medians -- the empirical floor
   below which a difference between two rows of the table is not interpretable.

usage: python g3_analysis.py <run_dir> <out_json>
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
GROUPS = {
    "hz1x1 DT K=20": ["hz1x1.dt_k20", "hz1x1.dt_nortg", "hz1x1.h4.k20", "hz1x1.c3.anchor_k200"],
    "hz1x1 DT K=1": ["hz1x1.h4.k1", "hz1x1.h4.k1_b1280"],
    "hz1x1 DT K=2": ["hz1x1.h4.k2", "hz1x1.h4.k2_b640"],
    "hz1x1 BC MLP": ["hz1x1.bc", "hz1x1.bc_top10", "hz1x1.bc_best2_20", "hz1x1.bc_any_20", "hz1x1.bc_worst2_20",
                     "hz1x1.bc_best2_all"],
    "hz1x1 MAPPO": ["hz1x1.mappo1000", "hz1x1.mappo500", "hz1x1.mappo060"],
    "grid4x4 spatial DT, 1 head": ["grid4x4.dt_spatial", "grid4x4.dt_nomix"],
    "grid4x4 spatial DT, 4 heads": ["grid4x4.dt_spatial_h4", "grid4x4.dt_nomix_h4"]
    + [f"grid4x4.c3.{s}" for s in ("ft_k5", "ft_k20", "ft_k100", "ft_k100_b1000", "ft_k100_b16000", "scratch_k100")],
    "grid4x4 BC MLP": ["grid4x4.bc", "grid4x4.bc_top10", "grid4x4.bc_top10_perix"],
    "grid4x4 MAPPO": ["grid4x4.mappo1000", "grid4x4.mappo060"],
}


def p95(values):
    n, cum = len(values), 0
    for v, c in sorted(Counter(values).items()):
        cum += c
        if cum * 100 >= 95 * n:
            return v


run = json.loads((RUN / "run.json").read_text())
cells = {}
for label in run["row_order"]:
    r = json.loads((RUN / f"{label}.json").read_text())
    eps = [e["decision_ns"] for e in r["episodes"]]
    reg = [x for e in eps for x in e[20:]]
    late = [x for e in eps for x in e[120:]]
    cells[label] = {
        "start_transient_ratio_worst_episode": max(st.median(e[20:40]) / st.median(e[120:]) for e in eps),
        "median_ms_20_359": st.median(reg) / 1e6, "p95_ms_20_359": p95(reg) / 1e6,
        "median_ms_120_359": st.median(late) / 1e6, "p95_ms_120_359": p95(late) / 1e6,
        "episode_medians_ms": [st.median(e[20:]) / 1e6 for e in eps],
    }
for c in cells.values():
    c["median_change_if_120"] = c["median_ms_120_359"] / c["median_ms_20_359"] - 1
    c["p95_change_if_120"] = c["p95_ms_120_359"] / c["p95_ms_20_359"] - 1
    c["episode_spread"] = max(c["episode_medians_ms"]) / min(c["episode_medians_ms"]) - 1
groups = {}
for name, rows in GROUPS.items():
    for dev in ("cpu", "cuda"):
        meds = [cells[f"{r}_{dev}"]["median_ms_20_359"] for r in rows]
        groups[f"{name} / {dev}"] = {"rows": rows, "min_ms": min(meds), "max_ms": max(meds), "spread": max(meds) / min(meds) - 1}
summary = {
    "cells": len(cells),
    "cells_with_start_transient_above_1.2": sum(c["start_transient_ratio_worst_episode"] > 1.2 for c in cells.values()),
    "median_change_if_120_max_abs": max(abs(c["median_change_if_120"]) for c in cells.values()),
    "p95_change_if_120_max_abs": max(abs(c["p95_change_if_120"]) for c in cells.values()),
    "cells_p95_change_above_10pct": sum(abs(c["p95_change_if_120"]) > 0.10 for c in cells.values()),
    "episode_spread_max": max(c["episode_spread"] for c in cells.values()),
    "same_computation_spread_max_hz1x1": max(g["spread"] for k, g in groups.items() if k.startswith("hz1x1")),
    "same_computation_spread_max_grid4x4": max(g["spread"] for k, g in groups.items() if k.startswith("grid4x4")),
}
OUT.write_text(json.dumps({"summary": summary, "groups": groups, "cells": cells}, indent=1))
print(json.dumps(summary, indent=1))
