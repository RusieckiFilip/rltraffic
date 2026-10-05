"""The README's C1 lines checked under Rule R (A11: att_engine primary on hz1x1 and grid4x4), from P8.4b's committed campaign.

Reads every cell file of output/p8_4b_rederivation/ (38,500; each carries att_engine and att_ours and reproduces_committed),
imports nothing from the project. Per scenario and tier: each arm's mean over its five seeds x 100 held-out draws under BOTH
definitions, the ranking, the leader against the runner-up as a paired contrast (per-draw means over the seeds, n = 100,
normal 95 % CI -- the method that reproduces P8.4b's committed V2 / V4 contrasts), and on grid4x4 d = dt_spatial - dt_nomix.
usage: python c1_rule_r.py <p8_4b_cells_dir> <out_json>
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

CELLS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TIERS = {"hz1x1": ["mappo1000", "mappo500", "maxpressure", "fixedtime", "random", "mix33", "mix50", "mix67"],
         "grid4x4": ["mappo1000", "maxpressure", "fixedtime", "random"]}
ARMS = {"hz1x1": ["dt", "madt", "bc", "bc_top10", "iql"],
        "grid4x4": ["dt_spatial", "dt_nomix", "bc", "bc_top10", "bc_top10_perix", "iql"]}
vals = defaultdict(dict)  # (scen, tier, method, defn) -> {(seed, draw): value}
n_files = n_bad = 0
for f in CELLS.glob("cell_*.json"):
    c = json.loads(f.read_text())
    n_files += 1
    if c.get("reproduces_committed") is not True:
        n_bad += 1
    scen, tier, method = c["scenario"], c["tier"], c["method"]
    if tier in TIERS.get(scen, []) and method in ARMS[scen] and "@" in c["arm"] and c["seed"] is not None:
        for defn in ("att_engine", "att_ours"):
            vals[(scen, tier, method, defn)][(c["seed"], c["draw_id"])] = c[defn]
print(f"cell files read: {n_files}; not reproducing committed: {n_bad}")


def per_draw(d):
    by = defaultdict(list)
    for (seed, draw), v in d.items():
        by[draw].append(v)
    return {draw: st.mean(v) for draw, v in by.items()}


out = {"cells_read": n_files, "not_reproducing": n_bad, "tiers": {}}
for scen, tiers in TIERS.items():
    for tier in tiers:
        entry = {}
        for defn in ("att_engine", "att_ours"):
            means = {m: st.mean(vals[(scen, tier, m, defn)].values()) for m in ARMS[scen] if vals.get((scen, tier, m, defn))}
            counts = {m: len(vals[(scen, tier, m, defn)]) for m in means}
            order = sorted(means, key=means.get)
            lead, second = order[0], order[1]
            a, b = per_draw(vals[(scen, tier, lead, defn)]), per_draw(vals[(scen, tier, second, defn)])
            diffs = [a[k] - b[k] for k in sorted(a) if k in b]
            mu, sd = st.mean(diffs), st.stdev(diffs)
            half = 1.959963984540054 * sd / math.sqrt(len(diffs))
            e = {"means": means, "cells_per_arm": counts, "order": order,
                 "leader_minus_runner_up": {"leader": lead, "runner_up": second, "mean": mu, "ci95": [mu - half, mu + half],
                                            "n_draws": len(diffs)}}
            if scen == "grid4x4" and "dt_spatial" in means and "dt_nomix" in means:
                s, n = per_draw(vals[(scen, tier, "dt_spatial", defn)]), per_draw(vals[(scen, tier, "dt_nomix", defn)])
                dd = [s[k] - n[k] for k in sorted(s) if k in n]
                m2, h2 = st.mean(dd), 1.959963984540054 * st.stdev(dd) / math.sqrt(len(dd))
                e["d_spatial_minus_nomix"] = {"mean": m2, "ci95": [m2 - h2, m2 + h2]}
            entry[defn] = e
        out["tiers"][f"{scen}/{tier}"] = entry
        en = entry["att_engine"]
        lr = en["leader_minus_runner_up"]
        print(f"{scen:8s} {tier:12s} engine leader {lr['leader']:14s} {en['means'][lr['leader']]:9.3f} vs {lr['runner_up']:14s} "
              f"{lr['mean']:+8.4f} [{lr['ci95'][0]:+.4f}, {lr['ci95'][1]:+.4f}] | ours leader {entry['att_ours']['order'][0]}"
              + (f" | d(eng) {en['d_spatial_minus_nomix']['mean']:+.4f} [{en['d_spatial_minus_nomix']['ci95'][0]:+.4f}, "
                 f"{en['d_spatial_minus_nomix']['ci95'][1]:+.4f}]" if "d_spatial_minus_nomix" in en else ""))
OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
