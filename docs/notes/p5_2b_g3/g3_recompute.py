"""P5.2b gate G3: the corrected cell and every random-tier statement recomputed by a THIRD route, importing nothing from
the project. Reads P5.2's committed eval files (att_ours), P8.4b's cells (att_engine), the correction run's (i) file and
(ii) cells, and the artifact; recomputes each arm's level (mean over 500 episodes), the corrected cell's mean / sd / CI,
the ranking, Q2a, Q2b (15 pairs against the artifact's predicted order, with the exact-tie rule), Q3a, Q3c (per-draw
means over the five seeds, n = 100, mean +/- 1.96 sd / sqrt(n)), and the per-seed firsts; compares with the artifact.
usage: python g3_recompute.py <main_tree> <out_json>
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

MAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
METHODS = ["dt_spatial", "dt_nomix", "bc", "bc_top10", "bc_top10_perix", "iql"]
NON_DT = ["bc", "bc_top10", "bc_top10_perix", "iql"]
# The project's dt_gate.mean_ci95 (P5.2's own route) uses z = 1.96; with the exact quantile 1.959963984540054 the CI
# endpoints differ in the fifth significant figure and nothing else does (checked at G3).
Z = 1.96
art = json.loads((MAIN / "output/p5_2b/artifacts/p5_2b_correction.json").read_text())


def registered_predicted_order():
    """P5.2's registered predicted levels at the random tier, read as a LITERAL from tier_sweep.py's source with ast
    (the table P5.2 registered; nothing of the project is executed), sorted ascending as tier_sweep.predicted_order does."""
    import ast
    tree = ast.parse((MAIN / "offline/tier_sweep.py").read_text())
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if targets and any(getattr(t, "id", None) == "PREDICTED_LEVELS" for t in targets) and node.value is not None:
            table = ast.literal_eval(node.value)
            return sorted(METHODS, key=lambda m: float(table[m]["random"]))
    raise ValueError("PREDICTED_LEVELS not found")


PREDICTED = registered_predicted_order()
problems, counts = [], {}


def note(k, n=1):
    counts[k] = counts.get(k, 0) + n


def ours_episodes(method, corrected=False):
    path = MAIN / ("output/p5_2b/eval_random_iql.json" if corrected else f"output/p5_2/eval_random_{method}.json")
    eps = json.loads(path.read_text())["episodes"]
    return {(int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"]) for e in eps}


def engine_episodes(method, corrected=False):
    work = MAIN / ("output/p5_2b/rederivation" if corrected else "output/p8_4b_rederivation")
    out = {}
    for seed in (101, 202, 303, 404, 505):
        for draw in range(1000, 1100):
            row = json.loads((work / f"cell_grid4x4_{method}_at_random_seed{seed}_draw{draw}.json").read_text())
            out[(seed, draw)] = (float(row["att_engine"]), float(row["att_ours"]), row)
    return out


def ci(values):
    m = st.mean(values); s = st.stdev(values); h = Z * s / math.sqrt(len(values))
    return m, s, m - h, m + h


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# the corrected cell: (i) against (ii), 500 episodes
ci_ours = ours_episodes("iql", corrected=True)
cii = engine_episodes("iql", corrected=True)
assert len(ci_ours) == 500 and len(cii) == 500
mismatch = sum(ci_ours[k] != cii[k][1] for k in ci_ours)
note("ii_vs_i_episodes", 500)
if mismatch:
    problems.append(f"(ii) att_ours differs from (i) on {mismatch} episodes")
record = json.loads((MAIN / "output/p5_2b/training_random_iql.json").read_text())
paths = {int(r["seed"]): r["checkpoint_path"] for r in record["runs"]}
bad_src = sum(1 for (seed, _), (_, _, row) in cii.items() if Path(row["policy_source"]["checkpoint"]).resolve() != Path(paths[seed]).resolve() or row.get("reproduces_committed") is not True)
if bad_src:
    problems.append(f"{bad_src} (ii) cells name another checkpoint or do not reproduce (i)")
prov = json.loads((MAIN / "output/p5_2b/eval_random_iql.json").read_text()).get("model_provenance") or {}
sd = {int(r["seed"]): r["state_dict_sha256"] for r in record["runs"]}
if {int(k): v.get("state_dict_sha256") for k, v in prov.items()} != sd:
    problems.append("(i)'s model_provenance digests are not the training record's")
note("provenance_seeds", len(sd))

# levels under both definitions, before and after
levels = {}
cells = {}
for defn in ("att_ours", "att_engine"):
    for when in ("before", "after"):
        L, C = {}, {}
        for m in METHODS:
            if defn == "att_ours":
                e = ours_episodes("iql", corrected=True) if (m == "iql" and when == "after") else ours_episodes(m)
            else:
                src = engine_episodes("iql", corrected=True) if (m == "iql" and when == "after") else engine_episodes(m)
                e = {k: v[0] for k, v in src.items()}
            assert len(e) == 500
            C[m] = e; L[m] = st.mean(e.values())
        levels[(defn, when)] = L; cells[(defn, when)] = C
note("levels", 24)

out = {"problems": problems, "statements": {}}
for (defn, when), L in levels.items():
    b = art["statements"][defn][when]; C = cells[(defn, when)]
    order = sorted(METHODS, key=lambda m: L[m])
    predicted = PREDICTED
    conc = 0
    pairs = [(a, c) for i, a in enumerate(sorted(METHODS)) for c in sorted(METHODS)[i + 1:]]
    for a, c in pairs:
        if L[a] == L[c]:
            continue
        if (predicted.index(a) < predicted.index(c)) == (order.index(a) < order.index(c)):
            conc += 1
    best = min(NON_DT, key=lambda m: L[m])
    # per-draw means over the five seeds, then the difference
    draws = sorted({d for _, d in C["dt_nomix"]})
    diffs = [st.mean([C["dt_nomix"][(s, d)] for s in (101, 202, 303, 404, 505)]) - st.mean([C[best][(s, d)] for s in (101, 202, 303, 404, 505)]) for d in draws]
    q3c = ci(diffs)
    firsts = {}
    for seed in (101, 202, 303, 404, 505):
        sl = {m: st.mean([v for (s, _), v in C[m].items() if s == seed]) for m in METHODS}
        firsts[str(seed)] = min(METHODS, key=lambda m: sl[m])
    rev = [int(s) for s, f in firsts.items() if f != order[0]]
    iql_cell = ci(list(C["iql"].values()))
    mine = {"order": order, "levels": L, "q2b": conc, "q2a_first": order[0], "q3a_rank": 1 + sum(L[m] < L["dt_nomix"] for m in METHODS if m != "dt_nomix"),
            "q3c": {"best": best, "mean": q3c[0], "low": q3c[2], "high": q3c[3], "n": len(diffs)}, "per_seed_firsts": firsts, "reverses": rev,
            "iql_cell": {"mean": iql_cell[0], "std": iql_cell[1], "low": iql_cell[2], "high": iql_cell[3]}}
    out["statements"][f"{defn}/{when}"] = mine
    # compare
    checks = [("order", order == b["ranking"]["order"]), ("q2b", conc == b["q2b"]["n_concordant"]), ("q2a", order[0] == b["q2a"]["measured_first"]),
              ("q3a", mine["q3a_rank"] == b["q3a"]["rank"]), ("q3c_best", best == b["q3c"]["best_non_dt"]),
              ("q3c_mean", close(q3c[0], b["q3c"]["mean"])), ("q3c_low", close(q3c[2], b["q3c"]["ci95_low"])), ("q3c_high", close(q3c[3], b["q3c"]["ci95_high"])),
              ("reverses", rev == list(b["q2a"].get("reverses_on_seeds", []))),
              ("firsts", firsts == {s: v["first"] for s, v in b["per_seed"]["seeds"].items()})]
    checks += [(f"level_{m}", close(L[m], b["ranking"]["levels"][m])) for m in METHODS]
    note("checks", len(checks))
    for name, ok in checks:
        if not ok:
            problems.append(f"{defn}/{when}: {name} disagrees with the artifact")
    if when == "after":
        cell = art["cells"]["corrected"][defn]
        for k, v in (("mean", iql_cell[0]), ("std", iql_cell[1]), ("ci95_low", iql_cell[2]), ("ci95_high", iql_cell[3])):
            note("cell_checks")
            if not close(v, cell[k]):
                problems.append(f"corrected cell {defn} {k}: mine {v} artifact {cell[k]}")
OUT.write_text(json.dumps(out, indent=1))
print("predicted order (registered):", PREDICTED)
print("counts:", counts)
print("problems:", problems if problems else "none")
for k, v in out["statements"].items():
    print(f"{k:18s} iql {v['iql_cell']['mean']:.4f} [{v['iql_cell']['low']:.4f}, {v['iql_cell']['high']:.4f}] | order {v['order'][:3]}.. | Q2b {v['q2b']} | Q3c {v['q3c']['mean']:+.4f} [{v['q3c']['low']:+.4f}, {v['q3c']['high']:+.4f}] | reverses {v['reverses']}")
