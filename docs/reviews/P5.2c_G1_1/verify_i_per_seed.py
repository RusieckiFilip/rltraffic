"""Independent check (json + statistics only, nothing from offline/) of I's per-seed values on the real data:
I_s = mean_draws(s4 - n4) - mean_draws(s1 - n1) for each training seed, under att_ours (the eval files) and att_engine
(P8.4b's cells), and which seeds reverse against the pooled sign (the mean of the per-seed values)."""
from __future__ import annotations

import json
import os
import statistics

OUT = "/home/filip/rltraffic/output"
P84B = os.path.join(OUT, "p8_4b_rederivation")
SEEDS = (101, 202, 303, 404, 505)
DRAWS = tuple(range(1000, 1100))


def src(method: str) -> tuple[str, str]:
    if method in ("dt_spatial", "dt_nomix"):
        return os.path.join(OUT, "p5_1", f"eval_{method}.json"), f"{method}_at_grid4x4_mappo1000"
    return os.path.join(OUT, "p5_2", f"eval_mappo1000_{method}.json"), f"{method}_at_mappo1000"


def values(method: str, definition: str) -> dict[tuple[int, int], float]:
    path, prefix = src(method)
    if definition == "att_ours":
        d = json.load(open(path))
        out = {(int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"]) for e in d["episodes"]}
    else:
        out = {}
        for s in SEEDS:
            for dr in DRAWS:
                c = json.load(open(os.path.join(P84B, f"cell_grid4x4_{prefix}_seed{s}_draw{dr}.json")))
                assert c["reproduces_committed"] is True
                out[(s, dr)] = float(c["att_engine"])
    assert sorted(out) == sorted((s, dr) for s in SEEDS for dr in DRAWS), (method, definition)
    return out


for definition in ("att_ours", "att_engine"):
    s4, n4, s1, n1 = (values(m, definition) for m in ("dt_spatial_h4", "dt_nomix_h4", "dt_spatial", "dt_nomix"))
    per_seed = {s: statistics.fmean([s4[(s, d)] - n4[(s, d)] for d in DRAWS])
                - statistics.fmean([s1[(s, d)] - n1[(s, d)] for d in DRAWS]) for s in SEEDS}
    pooled = statistics.fmean(per_seed.values())
    against = [s for s, v in per_seed.items() if (v < 0) != (pooled < 0) and v != 0.0]
    print(definition, {s: round(v, 4) for s, v in per_seed.items()}, "pooled", round(pooled, 4),
          "favouring d4 (negative):", sum(v < 0 for v in per_seed.values()), "reverse against pooled:", against)
