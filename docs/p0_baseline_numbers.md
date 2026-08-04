# P0.2 baseline numbers (the §3.1 sanity anchors)

**Run date:** 2026-07-09 · **Config:** `experiments/configs/p0_baselines.json` · **Seeds:** 101, 202, 303
(3 seeds) · **Train episodes:** 60 · **Eval episodes:** 3 · **max_steps:** 360 · **delta_time:** 10 ·
**control:** acyclic · **reward:** `queue_length` · **Backend:** CityFlow.

This file exists because `docs/PROJECT_PLAN.md` §3.1 has cited it since 2026-07-09 while it never
existed (0 commits touched the path), and its backing data lived only in `output/experiments/`, which is
gitignored — so §7's sanity anchor could not be reproduced from a fresh clone (finding folded into P0.6).

**Backing data, now tracked:** `docs/data/p0_baselines/results.json` (+ `summary.csv`), a byte-identical
copy of the 2026-07-09 run output. `output/` stays gitignored; only this copy is committed.
**Reproduction test:** `tests/test_p0_baseline_anchors.py` reads the committed `results.json`, recomputes
each mean independently from the per-seed cells, and asserts it reproduces the six §3.1 numbers below.
Numbers here are rounded to 2 decimal places, matching plan §3.1; the test carries the full precision.

## §3.1 anchor metric — average travel time (seconds, lower is better)

The primary sanity anchor. Any later phase producing numbers compares against these before results are
accepted (plan §7).

| Environment | MaxPressure | Random | MAPPO |
|---|---|---|---|
| cf_hz1x1   | **160.56 ± 0.00** | 307.53 ± 0.60 | 197.91 ± 1.78 |
| cf_grid4x4 | **141.65 ± 0.00** | 207.26 ± 1.93 | 632.95 ± 51.63 (gridlock, 1040 veh stuck) |

## Full metrics (mean ± std over 3 seeds)

`episode_reward` higher is better; the other three lower is better.

### cf_hz1x1 (CityFlow, 1×1)

| Policy | episode_reward | average_travel_time | final_vehicle_count | average_waiting_queue |
|---|---|---|---|---|
| MaxPressure | -32648.00 ± 0.00 | 160.56 ± 0.00 | 208.00 ± 0.00 | 90.69 ± 0.00 |
| Random | -40332.78 ± 149.62 | 307.53 ± 0.60 | 157.00 ± 0.72 | 112.04 ± 0.42 |
| MAPPO | -37218.67 ± 4132.26 | 197.91 ± 1.78 | 167.00 ± 17.15 | 103.39 ± 11.48 |

### cf_grid4x4 (CityFlow, 4×4)

| Policy | episode_reward | average_travel_time | final_vehicle_count | average_waiting_queue |
|---|---|---|---|---|
| MaxPressure | -4564.00 ± 0.00 | 141.65 ± 0.00 | 20.00 ± 0.00 | 12.67 ± 0.00 |
| Random | -17347.78 ± 507.07 | 207.26 ± 1.93 | 40.67 ± 1.91 | 48.19 ± 1.41 |
| MAPPO | -192925.00 ± 13030.27 | 632.95 ± 51.63 | 1040.00 ± 40.11 | 535.78 ± 36.36 |

**On cf_grid4x4 MAPPO:** the high travel time and ~1040 vehicles stuck at horizon are gridlock, not a
recording error — the run is preserved as-is so the collapse is on the record (it is one of the outcomes
`PREREGISTRATION.md` §10 registers publishing under).
