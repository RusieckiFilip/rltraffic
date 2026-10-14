"""P5.2c gate G3 (BRIEF_45 section 6): every statement of docs/data/p5_2_statements.json recomputed by the coordinator by an
independent route from the raw files, importing NOTHING from offline/ (json, statistics, math, itertools only).

Rules are taken from the registered text, docs/plans/p5.2.md section 4 (cited per constant below), and BRIEF_45 section 3.3,
not from the module or tier_sweep.  Every comparison is printed as MATCH or DIFF; floats compare at 1e-9 relative (1e-12
absolute near zero), everything else exactly.

Inputs: output/p5_1/eval_<m>.json and output/p5_2/eval_<tier>_<m>.json (att_ours = att_horizon); output/p5_2b/eval_random_iql.json
(the corrected IQL cell); att_engine and A11(b)'s companions from output/p8_4b_rederivation/cell_grid4x4_<m>_at_<tier>_seed<s>_draw<d>.json
and output/p5_2b/rederivation/ (the corrected IQL cell), each row refused unless reproduces_committed is true and its att_ours
equals the eval file's att_horizon for its (seed, draw) under ==.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from pathlib import Path

O = Path("/home/filip/rltraffic/output")
ART_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/filip/rltraffic-p52c/docs/data/p5_2_statements.json")
ART = json.loads(ART_PATH.read_text(encoding="utf-8"))
SEEDS = (101, 202, 303, 404, 505)
DRAWS = tuple(range(1000, 1100))
DEFS = ("att_ours", "att_engine")
Z = 1.96  # docs/plans/p5.2.md:409, "1.96 sigma / sqrt(n)"
METHODS = ("dt_spatial", "dt_nomix", "bc", "bc_top10", "bc_top10_perix", "iql")
OOS = ("maxpressure", "fixedtime", "random")
TIERS = ("mappo1000",) + OOS

# docs/plans/p5.2.md:489-497, the registered level table (predicted att_horizon)
PRED = {
    "maxpressure": {"dt_spatial": 208.2667, "dt_nomix": 166.5265, "bc": 175.7248, "bc_top10": 751.0633,
                    "bc_top10_perix": 166.0938, "iql": 264.0804},
    "fixedtime": {"dt_spatial": 256.3101, "dt_nomix": 204.9412, "bc": 222.1050, "bc_top10": 1042.7531,
                  "bc_top10_perix": 230.5996, "iql": 360.8493},
    "random": {"dt_spatial": 312.9006, "dt_nomix": 250.1900, "bc": 272.1254, "bc_top10": 1242.6948,
               "bc_top10_perix": 274.8156, "iql": 414.9930},
}
PRED_MAPPO_PERIX = 165.7657                  # :495
Q1B = {"maxpressure": 167.4920, "fixedtime": 206.9318}   # :497, :576-578
LEVEL_BAND, ANCHOR_BAND = 0.30, 0.10         # :560, :576-577
K_THRESHOLD = math.ceil(9 / 13 * 19)         # :561-563 -> 14
HARD = ("dt_nomix", "bc_top10_perix", "bc", "dt_spatial")   # :622
PRED_FIRST = {t: min(PRED[t], key=PRED[t].get) for t in OOS}  # :593-598
Q2B_THRESHOLD = 12                           # :619

results: list[tuple[str, str, object, object]] = []   # (where, status, mine, theirs)
maxdiff = {"value": 0.0}


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    if a is None or b is None:
        return False
    d = abs(a - b)
    maxdiff["value"] = max(maxdiff["value"], d / max(1.0, abs(a), abs(b)))
    return d <= 1e-12 or d <= 1e-9 * max(abs(a), abs(b))


def check(where: str, mine: object, theirs: object, *, approx: bool = False) -> None:
    if approx:
        ok = close(float(mine), float(theirs))
    else:
        ok = mine == theirs
    results.append((where, "MATCH" if ok else "DIFF", mine, theirs))


# ---------------------------------------------------------------- cells
def eval_path(cid: str) -> Path:
    src, rest = cid.split(":")
    method, tier = rest.split("@")
    if src == "p5_1":
        return O / "p5_1" / f"eval_{method}.json"
    if src == "p5_2":
        return O / "p5_2" / f"eval_{tier}_{method}.json"
    assert cid == "p5_2b:iql@random", cid
    return O / "p5_2b" / "eval_random_iql.json"


def rederivation(cid: str, seed: int, draw: int) -> Path:
    src, rest = cid.split(":")
    method, tier = rest.split("@")
    if src == "p5_2b":
        return O / "p5_2b" / "rederivation" / f"cell_grid4x4_iql_at_random_seed{seed}_draw{draw}.json"
    return O / "p8_4b_rederivation" / f"cell_grid4x4_{method}_at_{tier}_seed{seed}_draw{draw}.json"


CELLS: dict[str, dict] = {}


def load(cid: str) -> dict:
    if cid in CELLS:
        return CELLS[cid]
    payload = json.loads(eval_path(cid).read_text(encoding="utf-8"))
    ours = {(int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"]) for e in payload["episodes"]}
    hvc = {(int(e["seed"]), int(e["draw_id"])): float(e["horizon_vehicle_count"]) for e in payload["episodes"]}
    rew = {(int(e["seed"]), int(e["draw_id"])): float(e["episode_reward"]) for e in payload["episodes"]}
    assert sorted(ours) == sorted((s, d) for s in SEEDS for d in DRAWS), cid
    engine, comp = {}, {}
    for s in SEEDS:
        for d in DRAWS:
            row = json.loads(rederivation(cid, s, d).read_text(encoding="utf-8"))
            assert row["reproduces_committed"] is True, (cid, s, d)
            assert float(row["att_ours"]) == ours[(s, d)], (cid, s, d)
            assert (int(row["seed"]), int(row["draw_id"])) == (s, d)
            engine[(s, d)] = float(row["att_engine"])
            comp[(s, d)] = {k: row[k] for k in ("created", "entered", "entered_fraction", "never_entered")}
    CELLS[cid] = {"att_ours": ours, "att_engine": engine, "comp": comp, "hvc": hvc, "rew": rew, "cell": payload["cell"]}
    return CELLS[cid]


def book(tier: str, version: str) -> dict[str, str]:
    """method -> cell id, as BRIEF_45 section 3.2 registers the inputs."""
    if tier == "mappo1000":
        b = {m: f"p5_1:{m}@grid4x4_mappo1000" for m in ("dt_spatial", "dt_nomix", "bc", "bc_top10", "iql", "behaviour")}
        b["bc_top10_perix"] = "p5_2:bc_top10_perix@mappo1000"
        b["dt_spatial_h4"] = "p5_2:dt_spatial_h4@mappo1000"
        b["dt_nomix_h4"] = "p5_2:dt_nomix_h4@mappo1000"
        return b
    b = {m: f"p5_2:{m}@{tier}" for m in METHODS}
    if tier == "random":
        b["behaviour"] = ANCHOR    # D12: the shared random anchor is the behaviour policy at random
        if version == "corrected":
            b["iql"] = "p5_2b:iql@random"
    else:
        b["behaviour"] = f"p5_2:behaviour@{tier}"
    return b


ANCHOR = "p5_1:random@grid4x4_mappo1000"


def pdm(cid: str, definition: str) -> dict[int, float]:
    v = load(cid)[definition]
    return {d: statistics.fmean([v[(s, d)] for s in SEEDS]) for d in DRAWS}


def seed_level(cid: str, definition: str, seed: int) -> float:
    v = load(cid)[definition]
    return statistics.fmean([v[(seed, d)] for d in DRAWS])


def level(cid: str, definition: str) -> float:
    return statistics.fmean(load(cid)[definition].values())


def contrast(left: str, right: str, definition: str) -> dict:
    L, R = pdm(left, definition), pdm(right, definition)
    diffs = [L[d] - R[d] for d in DRAWS]
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    half = Z * sd / math.sqrt(len(diffs))
    lv, rv = load(left)[definition], load(right)[definition]
    by_seed = {str(s): statistics.fmean([lv[(s, d)] - rv[(s, d)] for d in DRAWS]) for s in SEEDS}
    return {"mean": mean, "low": mean - half, "high": mean + half, "exactly_zero": all(x == 0.0 for x in diffs),
            "by_seed": by_seed, "identical": all(lv[k] == rv[k] for k in lv), "diffs": diffs}


def composite(l4: tuple[str, str], l1: tuple[str, str], definition: str) -> dict:
    a, b = pdm(l4[0], definition), pdm(l4[1], definition)
    c, d_ = pdm(l1[0], definition), pdm(l1[1], definition)
    diffs = [(a[d] - b[d]) - (c[d] - d_[d]) for d in DRAWS]
    mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    half = Z * sd / math.sqrt(len(diffs))
    va, vb, vc, vd = (load(x)[definition] for x in (*l4, *l1))
    by_seed = {str(s): statistics.fmean([va[(s, d)] - vb[(s, d)] for d in DRAWS])
               - statistics.fmean([vc[(s, d)] - vd[(s, d)] for d in DRAWS]) for s in SEEDS}
    return {"mean": mean, "low": mean - half, "high": mean + half, "exactly_zero": all(x == 0.0 for x in diffs),
            "by_seed": by_seed, "neg": sum(x < 0 for x in diffs), "pos": sum(x > 0 for x in diffs)}


def interval(low: float, high: float, predicted: str) -> str:
    if predicted == "positive":
        return "HELD" if low > 0 else ("FAILED" if high < 0 else "NOT RESOLVED")
    return "HELD" if high < 0 else ("FAILED" if low > 0 else "NOT RESOLVED")


def conj(verdicts: list[str]) -> str:
    if all(v == "HELD" for v in verdicts):
        return "HELD"
    if any(v == "FAILED" for v in verdicts):
        return "FAILED"
    return "NOT RESOLVED"


def linear_extensions(levels: dict[str, float]) -> list[tuple[str, ...]]:
    arms = sorted(levels)
    out = []
    for perm in itertools.permutations(arms):
        pos = {a: i for i, a in enumerate(perm)}
        if all(pos[a] < pos[b] for a in arms for b in arms if levels[a] < levels[b]):
            out.append(perm)
    return out


def concordance(levels: dict[str, float], predicted_order: list[str]) -> tuple[int, int, int]:
    """(count under the tie rule, min, max over tie-break orders); docs/plans/p5.2.md:606-609, 618."""
    pairs = list(itertools.combinations(predicted_order, 2))   # (a, b): a predicted lower
    rule = sum(1 for a, b in pairs if levels[a] < levels[b])
    counts = []
    for perm in linear_extensions(levels):
        pos = {x: i for i, x in enumerate(perm)}
        counts.append(sum(1 for a, b in pairs if pos[a] < pos[b]))
    return rule, min(counts), max(counts)


def rank_of(levels: dict[str, float], arm: str) -> tuple[int, int, int]:
    rule = 1 + sum(1 for a, v in levels.items() if a != arm and v < levels[arm])
    ranks = [perm.index(arm) + 1 for perm in linear_extensions(levels)]
    return rule, min(ranks), max(ranks)


def firsts(levels: dict[str, float]) -> list[str]:
    m = min(levels.values())
    return sorted(a for a, v in levels.items() if v == m)


# ---------------------------------------------------------------- the artifact's statements, keyed
STMT: dict[tuple, dict] = {}
for s in ART["statements"]:
    key = (s["id"], s["tier"], s["definition"], s["version"], s.get("arm"))
    assert key not in STMT, key
    STMT[key] = s


def stmt(sid: str, tier, definition: str, version: str, arm=None) -> dict:
    for v in (version, "both"):
        if (sid, tier, definition, v, arm) in STMT:
            return STMT[(sid, tier, definition, v, arm)]
    raise KeyError((sid, tier, definition, version, arm))


def check_contrast(where: str, mine: dict, numbers: dict) -> None:
    check(f"{where}.mean", mine["mean"], numbers["mean_difference"], approx=True)
    check(f"{where}.low", mine["low"], numbers["ci95_low"], approx=True)
    check(f"{where}.high", mine["high"], numbers["ci95_high"], approx=True)
    check(f"{where}.exactly_zero", mine["exactly_zero"], numbers["exactly_zero"])
    for s, v in mine["by_seed"].items():
        check(f"{where}.by_seed[{s}]", v, numbers["per_seed"]["by_seed"][s], approx=True)
    if "discriminability" in numbers:
        check(f"{where}.identical", mine["identical"], numbers["discriminability"]["identical"])


VERSIONS = ("corrected", "as_committed")
SEEN = set()

# ---------------------------------------------------------------- Q0
for D in DEFS:
    b = book("mappo1000", "corrected")
    d1 = contrast(b["dt_spatial"], b["dt_nomix"], D)
    d4 = contrast(b["dt_spatial_h4"], b["dt_nomix_h4"], D)
    I = composite((b["dt_spatial_h4"], b["dt_nomix_h4"]), (b["dt_spatial"], b["dt_nomix"]), D)
    s = stmt("Q0.d1", "mappo1000", D, "both"); SEEN.add(id(s)); check_contrast(f"Q0.d1/{D}", d1, s["numbers"])
    s = stmt("Q0a", "mappo1000", D, "both"); SEEN.add(id(s)); check_contrast(f"Q0a/{D}", d4, s["numbers"])
    check(f"Q0a/{D}.verdict", interval(d4["low"], d4["high"], "positive"), s["verdict"])
    s = stmt("Q0b", "mappo1000", D, "both"); SEEN.add(id(s))
    n = s["numbers"]
    check(f"Q0b/{D}.mean", I["mean"], n["mean_difference"], approx=True)
    check(f"Q0b/{D}.low", I["low"], n["ci95_low"], approx=True)
    check(f"Q0b/{D}.high", I["high"], n["ci95_high"], approx=True)
    check(f"Q0b/{D}.draws_negative", I["neg"], n["draws_negative"])
    check(f"Q0b/{D}.draws_positive", I["pos"], n["draws_positive"])
    for sd_, v in I["by_seed"].items():
        check(f"Q0b/{D}.by_seed[{sd_}]", v, n["per_seed"]["by_seed"][sd_], approx=True)
    pooled = statistics.fmean(I["by_seed"].values())
    rev = sorted(int(k) for k, v in I["by_seed"].items() if v != 0 and (v < 0) != (pooled < 0))
    check(f"Q0b/{D}.reverses_against_pooled_sign", len(rev), n["per_seed"]["reverses_against_pooled_sign"])
    q = n["seed_reversal_qualifier"]
    want = f"d4 vs d1 reverses on {len(rev)} of 5 training seeds ({', '.join(map(str, rev))})" if rev else None
    check(f"Q0b/{D}.qualifier", (want in q) if (q and want) else (q is None and want is None), True)
    v0b = "HELD" if I["low"] <= 0.0 <= I["high"] else "FAILED"      # :523-524
    check(f"Q0b/{D}.verdict", v0b, s["verdict"])
    s = stmt("Q0.stop_rule", "mappo1000", D, "both"); SEEN.add(id(s))
    check(f"stop/{D}.verdict", "STOP" if d4["high"] < 0 else "CONTINUE", s["verdict"])     # :526-530

# ---------------------------------------------------------------- Q1, Q1b
for D in DEFS:
    for V in VERSIONS:
        cells = []
        for t in OOS:
            b = book(t, V)
            for m in METHODS:
                cells.append(((m, t), level(b[m], D), PRED[t][m]))
        cells.append((("bc_top10_perix", "mappo1000"), level(book("mappo1000", V)["bc_top10_perix"], D),
                      PRED_MAPPO_PERIX))
        held = {c: abs(m_ - p) / p <= LEVEL_BAND for c, m_, p in cells}
        s = stmt("Q1", None, D, V); SEEN.add(id(s))
        theirs = {tuple(c["cell"]): c for c in s["numbers"]["cells"]}
        check(f"Q1/{D}/{V}.cell_set", sorted(held), sorted(theirs))
        for c, m_, p in cells:
            check(f"Q1/{D}/{V}.{c}.measured", m_, theirs[c]["measured"], approx=True)
            check(f"Q1/{D}/{V}.{c}.held", held[c], theirs[c]["held"])
        check(f"Q1/{D}/{V}.n_held", sum(held.values()), s["numbers"]["n_held"])
        check(f"Q1/{D}/{V}.threshold", K_THRESHOLD, s["numbers"]["threshold"])
        check(f"Q1/{D}/{V}.verdict", "HELD" if sum(held.values()) >= K_THRESHOLD else "FAILED", s["verdict"])
    q1b = {}
    for t in ("maxpressure", "fixedtime"):
        m_ = level(book(t, "corrected")["behaviour"], D)
        ok = abs(m_ - Q1B[t]) / Q1B[t] <= ANCHOR_BAND
        q1b[t] = "HELD" if ok else "FAILED"
        s = stmt("Q1b", t, D, "both"); SEEN.add(id(s))
        check(f"Q1b/{t}/{D}.measured", m_, s["numbers"]["measured"], approx=True)
        check(f"Q1b/{t}/{D}.verdict", q1b[t], s["verdict"])
    s = stmt("Q1b.both", None, D, "both"); SEEN.add(id(s))
    check(f"Q1b.both/{D}.verdict", "HELD" if all(v == "HELD" for v in q1b.values()) else "FAILED", s["verdict"])

# ---------------------------------------------------------------- ranking, Q2a, Q2b, Q2b-hard, D9, Q3a
for D in DEFS:
    q2b_tiers = {V: {} for V in VERSIONS}
    for t in OOS:
        for V in (VERSIONS if t == "random" else ("both",)):
            vb = "corrected" if V == "both" else V
            b = book(t, vb)
            levels = {m: level(b[m], D) for m in METHODS}
            order = sorted(METHODS, key=lambda m: (levels[m], m))
            s = stmt("ranking", t, D, V); SEEN.add(id(s))
            for m in METHODS:
                check(f"ranking/{t}/{D}/{V}.level[{m}]", levels[m], s["numbers"]["levels"][m], approx=True)
            groups = [sorted(g) for _, g in itertools.groupby(order, key=lambda m: levels[m])]
            check(f"ranking/{t}/{D}/{V}.groups", [sorted(g) for g in groups], [sorted(g) for g in s["numbers"]["groups"]])
            # Q2a
            f = firsts(levels)
            s = stmt("Q2a", t, D, V); SEEN.add(id(s))
            check(f"Q2a/{t}/{D}/{V}.possible_firsts", f, sorted(s["numbers"]["possible_firsts"]))
            v2a = "HELD" if f == [PRED_FIRST[t]] else "FAILED"
            check(f"Q2a/{t}/{D}/{V}.verdict", v2a, s["verdict"])
            check(f"Q2a/{t}/{D}/{V}.predicted_first", PRED_FIRST[t], s["numbers"]["predicted_first"])
            # Q2b
            porder = sorted(METHODS, key=lambda m: PRED[t][m])
            rule, lo, hi = concordance(levels, porder)
            s = stmt("Q2b", t, D, V); SEEN.add(id(s))
            check(f"Q2b/{t}/{D}/{V}.n_concordant", rule, s["numbers"]["n_concordant"])
            check(f"Q2b/{t}/{D}/{V}.range", [lo, hi], list(s["numbers"]["range"]))
            v2b = "HELD" if rule >= Q2B_THRESHOLD else "FAILED"
            check(f"Q2b/{t}/{D}/{V}.verdict", v2b, s["verdict"])
            for VV in (VERSIONS if V == "both" else (V,)):
                q2b_tiers[VV][t] = (v2b, rule, lo, hi)
            # per seed (D9 and Q2b's per-seed blocks)
            seed_levels = {sd: {m: seed_level(b[m], D, sd) for m in METHODS} for sd in SEEDS}
            s9 = stmt("D9", t, D, V); SEEN.add(id(s9))
            for sd in SEEDS:
                sl = seed_levels[sd]
                r_, lo_, hi_ = concordance(sl, porder)
                hr, hlo, hhi = concordance({m: sl[m] for m in HARD}, sorted(HARD, key=lambda m: PRED[t][m]))
                rk, rklo, rkhi = rank_of(sl, "dt_nomix")
                blk = s9["numbers"]["seeds"][str(sd)]
                check(f"D9/{t}/{D}/{V}/{sd}.first", firsts(sl), sorted(blk["first_place_possible"]))
                check(f"D9/{t}/{D}/{V}/{sd}.n_concordant", [r_, lo_, hi_], [blk["n_concordant"], *blk["n_concordant_range"]])
                check(f"D9/{t}/{D}/{V}/{sd}.hard", [hr, hlo, hhi], [blk["hard_n_concordant"], *blk["hard_n_concordant_range"]])
                check(f"D9/{t}/{D}/{V}/{sd}.dt_nomix_rank", [rk, rklo, rkhi], [blk["dt_nomix_rank"], *blk["dt_nomix_rank_range"]])
                for m in METHODS:
                    check(f"D9/{t}/{D}/{V}/{sd}.level[{m}]", sl[m], blk["levels"][m], approx=True)
                check(f"Q2b/{t}/{D}/{V}/{sd}.per_seed", [r_, lo_, hi_],
                      [stmt("Q2b", t, D, V)["numbers"]["per_seed_n_concordant"][str(sd)],
                       *stmt("Q2b", t, D, V)["numbers"]["per_seed_range"][str(sd)]])
            # Q2b-hard (corrected iql never enters the hard subset; one statement per tier)
            if V in ("both", "corrected"):
                hl = {m: levels[m] for m in HARD}
                hr, hlo, hhi = concordance(hl, sorted(HARD, key=lambda m: PRED[t][m]))
                s = stmt("Q2b_hard", t, D, "both"); SEEN.add(id(s))
                check(f"Q2b_hard/{t}/{D}", [hr, hlo, hhi], [s["numbers"]["n_concordant"], *s["numbers"]["range"]])
            # Q3a
            rk, rklo, rkhi = rank_of(levels, "dt_nomix")
            s = stmt("Q3a", t, D, V); SEEN.add(id(s))
            check(f"Q3a/{t}/{D}/{V}.rank", [rk, rklo, rkhi], [s["numbers"]["rank"], *s["numbers"]["rank_range"]])
            v3a = "HELD" if f == ["dt_nomix"] else "FAILED"
            check(f"Q3a/{t}/{D}/{V}.verdict", v3a, s["verdict"])
    for V in VERSIONS:
        s = stmt("Q2b.aggregate", None, D, V); SEEN.add(id(s))
        check(f"Q2b.aggregate/{D}/{V}.verdict", conj([q2b_tiers[V][t][0] for t in OOS]), s["verdict"])
        check(f"Q2b.aggregate/{D}/{V}.overall", [sum(q2b_tiers[V][t][1] for t in OOS),
                                                 sum(q2b_tiers[V][t][2] for t in OOS), sum(q2b_tiers[V][t][3] for t in OOS)],
              [s["numbers"]["overall"]["n_concordant"], *s["numbers"]["overall"]["range"]])
    # Q3a at mappo1000, context
    b = book("mappo1000", "corrected")
    levels = {m: level(b[m], D) for m in METHODS}
    s = stmt("Q3a", "mappo1000", D, "both"); SEEN.add(id(s))
    check(f"Q3a/mappo1000/{D}.rank", list(rank_of(levels, "dt_nomix")), [s["numbers"]["rank"], *s["numbers"]["rank_range"]])
    check(f"Q3a/mappo1000/{D}.verdict", None, s["verdict"])

# ---------------------------------------------------------------- Q3b, Q3c
for D in DEFS:
    q3b = {}
    for t in TIERS:
        b = book(t, "corrected")
        c = contrast(b["dt_nomix"], b["behaviour"], D)
        s = stmt("Q3b", t, D, "both"); SEEN.add(id(s))
        check_contrast(f"Q3b/{t}/{D}", c, s["numbers"])
        if t == "mappo1000":
            check(f"Q3b/{t}/{D}.verdict", None, s["verdict"])
            continue
        q3b[t] = interval(c["low"], c["high"], "negative")
        check(f"Q3b/{t}/{D}.verdict", q3b[t], s["verdict"])
        resolved = c["low"] > 0 or c["high"] < 0
        sub = s["numbers"]["sub_prediction"]
        check(f"Q3b/{t}/{D}.sub.resolved", resolved, sub["resolved"])
        check(f"Q3b/{t}/{D}.sub.matched", resolved == (t == "random"), sub["matched"])
    s = stmt("Q3b.aggregate", None, D, "both"); SEEN.add(id(s))
    check(f"Q3b.aggregate/{D}.verdict", conj([q3b[t] for t in OOS]), s["verdict"])
    for t in TIERS:
        for V in (VERSIONS if t == "random" else ("both",)):
            b = book(t, "corrected" if V == "both" else V)
            nondt = {m: level(b[m], D) for m in ("bc", "bc_top10", "bc_top10_perix", "iql")}
            best = firsts(nondt)
            assert len(best) == 1, (t, D, V, best)
            c = contrast(b["dt_nomix"], b[best[0]], D)
            s = stmt("Q3c", t, D, V); SEEN.add(id(s))
            check(f"Q3c/{t}/{D}/{V}.best", best[0], s["numbers"]["best_non_dt"])
            check_contrast(f"Q3c/{t}/{D}/{V}", c, s["numbers"]["contrast"])
            reading = ("resolves for the DT" if c["high"] < 0 else "resolves against the DT" if c["low"] > 0
                       else "NOT RESOLVED")
            check(f"Q3c/{t}/{D}/{V}.reading", reading, s["reading"])
            check(f"Q3c/{t}/{D}/{V}.verdict", None, s["verdict"])

# ---------------------------------------------------------------- Q4
for D in DEFS:
    d = {}
    for t in TIERS:
        b = book(t, "corrected")
        c = contrast(b["dt_spatial"], b["dt_nomix"], D)
        d[t] = c
        s = stmt("Q4.d", t, D, "both"); SEEN.add(id(s))
        check_contrast(f"Q4.d/{t}/{D}", c, s["numbers"])
    q4a = {}
    for t in OOS:
        s = stmt("Q4a", t, D, "both"); SEEN.add(id(s))
        check_contrast(f"Q4a/{t}/{D}", d[t], s["numbers"])
        q4a[t] = interval(d[t]["low"], d[t]["high"], "positive")
        check(f"Q4a/{t}/{D}.verdict", q4a[t], s["verdict"])
    s = stmt("Q4a.aggregate", None, D, "both"); SEEN.add(id(s))
    check(f"Q4a.aggregate/{D}.verdict", conj([q4a[t] for t in OOS]), s["verdict"])
    chain = [d[t]["mean"] for t in TIERS]
    inc = all(a < b_ for a, b_ in zip(chain, chain[1:]))
    s = stmt("Q4b", None, D, "both"); SEEN.add(id(s))
    check(f"Q4b/{D}.verdict", "HELD" if inc else "FAILED", s["verdict"])
    check(f"Q4b/{D}.strictly_decreasing", all(a > b_ for a, b_ in zip(chain, chain[1:])),
          s["numbers"]["strictly_decreasing"])
    three = [d[t]["mean"] for t in ("mappo1000", "maxpressure", "random")]
    s = stmt("Q4b.three_rung", None, D, "both"); SEEN.add(id(s))
    check(f"Q4b.three_rung/{D}.verdict", "HELD" if all(a < b_ for a, b_ in zip(three, three[1:])) else "FAILED",
          s["verdict"])

# ---------------------------------------------------------------- Q5
Q5 = {"Q5a": (("bc_top10", "bc"), OOS, {t: "positive" for t in OOS}),
      "Q5b": (("bc_top10_perix", "bc_top10"), TIERS, {t: "negative" for t in TIERS}),
      "Q5c": (("bc_top10_perix", "bc"), TIERS, {"mappo1000": "negative", "maxpressure": "negative",
                                                 "fixedtime": "positive", "random": "positive"})}   # :683-689
for D in DEFS:
    for sid, ((l, r), tiers, pred) in Q5.items():
        vs = {}
        for t in tiers:
            b = book(t, "corrected")
            c = contrast(b[l], b[r], D)
            s = stmt(sid, t, D, "both"); SEEN.add(id(s))
            check_contrast(f"{sid}/{t}/{D}", c, s["numbers"])
            vs[t] = interval(c["low"], c["high"], pred[t])
            check(f"{sid}/{t}/{D}.verdict", vs[t], s["verdict"])
        s = stmt(f"{sid}.aggregate", None, D, "both"); SEEN.add(id(s))
        check(f"{sid}.aggregate/{D}.verdict", conj(list(vs.values())), s["verdict"])

# ---------------------------------------------------------------- Q6 and the collapse audit
for D in DEFS:
    for t in TIERS:
        b = book(t, "corrected")
        for arm in ("dt_spatial", "dt_nomix"):
            c = contrast(b[arm], ANCHOR, D)
            s = stmt("Q6", t, D, "both", arm); SEEN.add(id(s))
            check_contrast(f"Q6/{t}/{D}/{arm}", c, s["numbers"])
            check(f"Q6/{t}/{D}/{arm}.verdict", "COLLAPSED" if c["low"] > 0 else "NOT COLLAPSED", s["verdict"])  # :699-701
audit = ART["descriptive_collapse_audit"]
for row in audit["rows"] + audit["dt_rows"]:
    V = row["version"]
    b = book(row["tier"], "corrected" if V == "both" else V)
    c = contrast(b[row["arm"]], ANCHOR, row["definition"])
    check_contrast(f"audit/{row['arm']}/{row['tier']}/{row['definition']}/{V}", c, row["contrast"])
    check(f"audit/{row['arm']}/{row['tier']}/{row['definition']}/{V}.verdict",
          "COLLAPSED" if c["low"] > 0 else "NOT COLLAPSED", row["verdict"])
mine_rows = sorted((a, t, D, V) for D in DEFS for t in TIERS for a in ("bc", "bc_top10", "bc_top10_perix", "iql")
                   for V in ((VERSIONS) if (t == "random" and a == "iql") else ("both",)))
check("audit.row_set", mine_rows, sorted((r["arm"], r["tier"], r["definition"], r["version"]) for r in audit["rows"]))
for sh in audit["sentence_holds"]:
    D, V = sh["definition"], sh["version"]
    collapse_on = {}
    for a in ("bc", "bc_top10", "bc_top10_perix", "iql"):
        collapse_on[a] = [t for t in TIERS if contrast(book(t, V)[a], ANCHOR, D)["low"] > 0]
    dt_never = all(contrast(book(t, V)[a], ANCHOR, D)["low"] <= 0 for t in TIERS for a in ("dt_spatial", "dt_nomix"))
    check(f"audit.sentence/{D}/{V}.collapse_on", collapse_on, sh["non_dt_arms_collapse_on"])
    check(f"audit.sentence/{D}/{V}.holds", dt_never and all(collapse_on[a] for a in collapse_on), sh["holds"])

# ---------------------------------------------------------------- leaders
for L in ART["leaders"]:
    t, D, V = L["tier"], L["definition"], L["version"]
    b = book(t, "corrected" if V == "both" else V)
    levels = {m: level(b[m], D) for m in METHODS}
    lead = firsts(levels)
    check(f"leaders/{t}/{D}/{V}.leaders", lead, sorted(L["leaders"]))
    rest = {m: v for m, v in levels.items() if m not in lead}
    runners = firsts(rest)
    check(f"leaders/{t}/{D}/{V}.runners_up", runners, sorted(L["runners_up"]))
    theirs = {(c["left_arm"].split("@")[0], c["right_arm"].split("@")[0]): c for c in L["leader_minus_runner_up"]}
    check(f"leaders/{t}/{D}/{V}.pairs", sorted((a, r) for a in lead for r in runners), sorted(theirs))
    for a in lead:
        for r in runners:
            check_contrast(f"leaders/{t}/{D}/{V}/{a}-{r}", contrast(b[a], b[r], D), theirs[(a, r)])
    psf = {str(sd): firsts({m: seed_level(b[m], D, sd) for m in METHODS}) for sd in SEEDS}
    check(f"leaders/{t}/{D}/{V}.per_seed_first_places", psf, {k: sorted(v) for k, v in L["per_seed_first_places"].items()})

# ---------------------------------------------------------------- cells
def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


for cid, blk in ART["cells"].items():
    cell = load(cid)
    for D in DEFS:
        xs = list(cell[D].values())
        mine = {"mean": statistics.fmean(xs), "std": statistics.stdev(xs), "median": pct(xs, 0.5),
                "q1": pct(xs, 0.25), "q3": pct(xs, 0.75)}
        mine["ci95"] = Z * mine["std"] / math.sqrt(len(xs))
        th = blk[D]
        check(f"cell/{cid}/{D}.n", len(xs), th["n"])
        for k in ("mean", "std", "median", "ci95"):
            check(f"cell/{cid}/{D}.{k}", mine[k], th[k], approx=True)
        check(f"cell/{cid}/{D}.iqr0", mine["q1"], th["iqr"][0], approx=True)
        check(f"cell/{cid}/{D}.iqr1", mine["q3"], th["iqr"][1], approx=True)
        for sd in SEEDS:
            check(f"cell/{cid}/{D}.seed{sd}", seed_level(cid, D, sd), th["per_seed_means"][str(sd)], approx=True)
    comp = list(cell["comp"].values())
    th = blk["companions"]
    for k in ("created", "entered", "entered_fraction", "never_entered"):
        check(f"cell/{cid}.{k}_mean", statistics.fmean(float(r[k]) for r in comp), th[f"{k}_mean"], approx=True)
    check(f"cell/{cid}.episodes_with_never_entered", sum(1 for r in comp if r["never_entered"] > 0),
          th["episodes_with_never_entered"])
    check(f"cell/{cid}.hvc_mean", statistics.fmean(cell["hvc"].values()), blk["horizon_vehicle_count_mean"], approx=True)

# the shared anchor's two copies (BRIEF_45 section 3.2, Q6): episode-identical under both definitions
for D in DEFS:
    check(f"anchor_copies_identical/{D}", load("p5_2:random@mappo1000")[D] == load(ANCHOR)[D], True)

# ---------------------------------------------------------------- ties: identical-episode and equal-level pairs
for T in ART["ties"]:
    t, D = T["tier"], T["definition"]
    V = T["version"]
    cells = T["cells"]
    arms = sorted(cells)
    eq_level = sorted([a, b_] for a, b_ in itertools.combinations(arms, 2) if level(cells[a], D) == level(cells[b_], D))
    ident = sorted([a, b_] for a, b_ in itertools.combinations(arms, 2) if load(cells[a])[D] == load(cells[b_])[D])
    check(f"ties/{t}/{D}/{V}.equal_level_pairs", eq_level, sorted(sorted(p) for p in T["equal_level_pairs"]))
    check(f"ties/{t}/{D}/{V}.identical_episode_pairs", ident, sorted(sorted(p["pair"]) for p in T["identical_episode_pairs"]))
    for p in T["identical_episode_pairs"]:
        ca, cb = load(cells[p["pair"][0]]), load(cells[p["pair"][1]])
        fields_ok = (ca["att_ours"] == cb["att_ours"] and ca["att_engine"] == cb["att_engine"] and ca["comp"] == cb["comp"]
                     and ca["hvc"] == cb["hvc"] and ca["rew"] == cb["rew"])
        check(f"ties/{t}/{D}/{V}.{p['pair']}.all_eight_fields_identical", fields_ok, True)
        check(f"ties/{t}/{D}/{V}.{p['pair']}.n_episodes", len(ca["att_ours"]), p["n_episodes"])
    some = {}
    for a, b_ in itertools.combinations(arms, 2):
        va, vb = load(cells[a])[D], load(cells[b_])[D]
        n_eq = sum(1 for k in va if va[k] == vb[k])
        if n_eq:
            some[tuple(sorted((a, b_)))] = n_eq
    theirs = {tuple(sorted(p["pair"])): p["n_equal"] for p in T["pairs_with_equal_episodes"]}
    check(f"ties/{t}/{D}/{V}.pairs_with_equal_episodes", some, theirs)

# ---------------------------------------------------------------- coverage and definition_dependent
unseen = [(s["id"], s["tier"], s["definition"], s["version"], s.get("arm")) for s in ART["statements"] if id(s) not in SEEN]
check("coverage: statements not recomputed", unseen, [])
by_key: dict[tuple, dict] = {}
for s in ART["statements"]:
    by_key.setdefault((s["id"], s["tier"], s["version"], s.get("arm")), {})[s["definition"]] = s
for key, pair in by_key.items():
    if set(pair) != set(DEFS):
        check(f"definitions present {key}", sorted(pair), list(DEFS)); continue
    a, b_ = pair["att_ours"], pair["att_engine"]
    on = a["definition_dependent_on"]
    mine = (a["verdict"] != b_["verdict"]) if on == "verdict" else (a["reading"] != b_["reading"])
    check(f"definition_dependent {key}", [mine, mine], [a["definition_dependent"], b_["definition_dependent"]])
    check(f"primary {key}", [False, True], [a["primary"], b_["primary"]])

n_diff = sum(1 for r in results if r[1] == "DIFF")
if len(sys.argv) <= 1:
    Path(__file__).with_name("g3_recompute_results.json").write_text(json.dumps(
        [{"where": w, "status": st, "mine": repr(m), "theirs": repr(t)} for w, st, m, t in results], indent=1),
        encoding="utf-8")
print(f"comparisons: {len(results)}  MATCH: {len(results) - n_diff}  DIFF: {n_diff}  "
      f"max relative float difference: {maxdiff['value']:.3e}  cells loaded: {len(CELLS)}")
for where, status, mine, theirs in results:
    if status == "DIFF":
        print(f"DIFF {where}: mine={mine!r} theirs={theirs!r}")
