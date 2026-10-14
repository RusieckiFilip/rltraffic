"""P5.2c G5 fix round (BRIEF_45 Amendment D): the coordinator's checks of the rebuilt artifact's NEW and CHANGED fields,
by routes that import nothing from offline/.  Key sets are asserted before values are compared (docs/reviews/P5.2c_G3.md
section 7).  The contrasts against the random anchor are recomputed from the raw eval files and P8.4b rows.

D1.1 q5 descriptive reading: the two tier sets from the coordinator's own collapse rule, and the reading's wording.
D1.2 never_scored and changes: every row's point value and CI equal the statement's (Q3c's from its contrast), key sets
     asserted, null only where the statement has no point estimate.
D1.3 literal_records: every row carries att_engine / att_engine_equal (or a null with a note); the three named rows;
     att_engine_equal consistent with the strings; a sample of numeric att_engine values against the statements.
D1.4 p8_4b labels.  D1.5 definitions keys.  D1.6 per-seed notes on exactly the blocks with a behaviour/random arm.
D1.7 Q2a tiers_note, the audit rule, Q4b's citation, Q6 registered_expectation_met, predictions_registered_on.
D1.8 the small-sample note on exactly the blocks with 0 < n_used < 20.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ART = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else
                      "/home/filip/rltraffic-p52c/docs/data/p5_2_statements.json").read_text(encoding="utf-8"))
O = Path("/home/filip/rltraffic/output")
SEEDS, DRAWS = (101, 202, 303, 404, 505), range(1000, 1100)
DEFS = ("att_ours", "att_engine")
out: list[tuple[str, bool, object, object]] = []


def check(where, mine, theirs):
    out.append((where, mine == theirs, mine, theirs))


S = {(s["id"], s["tier"], s["definition"], s["version"], s.get("arm")): s for s in ART["statements"]}


def stmt(sid, tier, d, version="both", arm=None):
    for v in (version, "both"):
        if (sid, tier, d, v, arm) in S:
            return S[(sid, tier, d, v, arm)]
    raise KeyError((sid, tier, d, version, arm))


def point(s):
    """The statement's point estimate and CI by the coordinator's own rule."""
    n = s["numbers"]
    if "mean_difference" in n:
        return n["mean_difference"], n.get("ci95_low"), n.get("ci95_high")
    if "contrast" in n:
        c = n["contrast"]
        if "mean" in n:
            assert n["mean"] == c["mean_difference"]
        return c["mean_difference"], c["ci95_low"], c["ci95_high"]
    return None, None, None


# ---- raw values for the collapse rule (D1.1)
def ours(path):
    d = json.loads(Path(path).read_text())
    return {(int(e["seed"]), int(e["draw_id"])): float(e["att_horizon"]) for e in d["episodes"]}


def engine(prefix):
    v = {}
    for s in SEEDS:
        for dr in DRAWS:
            r = json.loads((O / "p8_4b_rederivation" / f"cell_grid4x4_{prefix}_seed{s}_draw{dr}.json").read_text())
            assert r["reproduces_committed"] is True
            v[(s, dr)] = float(r["att_engine"])
    return v


def collapsed(left, right):
    diffs = [statistics.fmean([left[(s, d)] for s in SEEDS]) - statistics.fmean([right[(s, d)] for s in SEEDS])
             for d in DRAWS]
    m, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    return m - 1.96 * sd / 10 > 0


cells = {"ours": {}, "engine": {}}
anchor = {"att_ours": ours(O / "p5_1/eval_random.json"), "att_engine": engine("random_at_grid4x4_mappo1000")}
for f in ("bc_top10", "bc_top10_perix"):
    for t in ("mappo1000", "maxpressure", "fixedtime", "random"):
        if t == "mappo1000" and f == "bc_top10":
            src_o, src_e = O / "p5_1/eval_bc_top10.json", "bc_top10_at_grid4x4_mappo1000"
        elif t == "mappo1000":
            src_o, src_e = O / "p5_2/eval_mappo1000_bc_top10_perix.json", "bc_top10_perix_at_mappo1000"
        else:
            src_o, src_e = O / f"p5_2/eval_{t}_{f}.json", f"{f}_at_{t}"
        cells["ours"][(f, t)] = ours(src_o)
        cells["engine"][(f, t)] = engine(src_e)
for q in ART["comparison_with_the_record"]["q5"]:
    for d in DEFS:
        key = "ours" if d == "att_ours" else "engine"
        col = {(f, t): collapsed(cells[key][(f, t)], anchor[d]) for (f, t) in cells[key]}
        alone = [t for t in ("mappo1000", "maxpressure", "fixedtime", "random")
                 if col[("bc_top10", t)] and not col[("bc_top10_perix", t)]]
        both = [t for t in ("mappo1000", "maxpressure", "fixedtime", "random")
                if col[("bc_top10", t)] and col[("bc_top10_perix", t)]]
        dr = q["descriptive_reading"][d]
        check(f"D1.1 q5/{d}.alone", alone, dr["tiers_global_filter_collapses_alone"])
        check(f"D1.1 q5/{d}.both", both, dr["tiers_both_filters_collapse"])
        first_clause = dr["reading"].split(";")[0]
        check(f"D1.1 q5/{d}.reading names no both-collapse tier in its first clause",
              [t for t in both if t in first_clause], [])
        check(f"D1.1 q5/{d}.reading names every alone tier", all(t in first_clause for t in alone), True)
        check(f"D1.1 q5/{d}.reading says both collapse on {both}",
              all(t in dr["reading"].split(";", 1)[1] for t in both) if both else True, True)

# ---- D1.2 never_scored and changes: key sets first
REQUIRED = {"mean_difference", "ci95_low", "ci95_high"}
for entry in ART["comparison_with_the_record"]["never_scored"]:
    for row in entry["this_artifact"]:
        s = stmt(row["id"], row["tier"], row["definition"], row["version"], row.get("arm"))
        check(f"D1.2 never_scored/{row['id']}/{row['tier']}/{row['definition']} key set", REQUIRED <= set(row), True)
        check(f"D1.2 never_scored/{row['id']}/{row['tier']}/{row['definition']} point",
              point(s), (row.get("mean_difference"), row.get("ci95_low"), row.get("ci95_high")))
for c in ART["changes"]:
    for step, (d, v) in (("as_committed_att_ours", ("att_ours", "as_committed")),
                         ("corrected_att_ours", ("att_ours", "corrected")),
                         ("corrected_att_engine", ("att_engine", "corrected"))):
        s = stmt(c["id"], c["tier"], d, v, c.get("arm"))
        check(f"D1.2 changes/{c['id']}/{c['tier']}/{step} key set", REQUIRED <= set(c[step]), True)
        check(f"D1.2 changes/{c['id']}/{c['tier']}/{step} point",
              point(s), (c[step].get("mean_difference"), c[step].get("ci95_low"), c[step].get("ci95_high")))
        check(f"D1.2 changes/{c['id']}/{c['tier']}/{step} verdict+reading", (s["verdict"], s["reading"]),
              (c[step]["verdict"], c[step]["reading"]))
    o = [(c[k]["verdict"], c[k]["reading"]) for k in ("as_committed_att_ours", "corrected_att_ours", "corrected_att_engine")]
    check(f"D1.2 changes/{c['id']}/{c['tier']} flags", (o[0] != o[1], o[1] != o[2]),
          (c["changed_by_the_correction"], c["changed_by_rule_r"]))
q3c_random = next(c for c in ART["changes"] if (c["id"], c["tier"]) == ("Q3c", "random"))
check("D1.2 Q3c random moves under the correction (value)",
      (round(q3c_random["as_committed_att_ours"]["mean_difference"], 4),
       round(q3c_random["corrected_att_ours"]["mean_difference"], 4)), (64.3603, 56.8432))

# ---- D1.3 literal_records' att_engine column
lit = ART["comparison_with_the_record"]["literal_records"]


def norm(x):
    return re.sub(r"\s+", " ", str(x).replace("−", "-").replace("**", "")).strip()


for r in lit:
    check(f"D1.3 literal {r['value']} key set", {"att_engine", "att_engine_equal"} <= set(r), True)
    if r.get("att_engine") is None:
        check(f"D1.3 literal {r['value']} null carries a note", bool(r.get("att_engine_note")), True)
        check(f"D1.3 literal {r['value']} null has att_engine_equal null", r.get("att_engine_equal"), None)
    else:
        check(f"D1.3 literal {r['value']} att_engine_equal consistent",
              norm(r["att_engine"]) == norm(r["recorded"]), r["att_engine_equal"])
row = next(r for r in lit if r["value"] == "q3c.all_against")
check("D1.3 q3c.all_against does not hold under att_engine", row["att_engine_equal"], False)
print("q3c.all_against att_engine:", row["att_engine"])
for r in lit:
    if "31.5" in str(r["recorded"]):
        check(f"D1.3 {r['value']} carries 31.4 under att_engine", "31.4" in str(r["att_engine"]), True)
nulls = [r["value"] for r in lit if r.get("att_engine") is None]
print("rows with no att_engine counterpart:", len(nulls), nulls)
# a numeric sample against the corrected att_engine statements, at 4 decimals
sample = {"q3b.maxpressure.mean": round(stmt("Q3b", "maxpressure", "att_engine")["numbers"]["mean_difference"], 4),
          "q3b.random.mean": round(stmt("Q3b", "random", "att_engine")["numbers"]["mean_difference"], 4),
          "q0.i.mean": round(stmt("Q0b", "mappo1000", "att_engine")["numbers"]["mean_difference"], 4),
          "q4.random.mean": round(stmt("Q4.d", "random", "att_engine")["numbers"]["mean_difference"], 4)}
for value, mine in sample.items():
    r = next((r for r in lit if r["value"] == value), None)
    if r is None:
        print("  (no literal row", value, ")")
        continue
    check(f"D1.3 sample {value}", f"{mine:.4f}".replace("-", ""), norm(r["att_engine"]).replace("-", "").replace("+", ""))

# ---- D1.4 p8_4b labels
p8 = {(r["contrast_id"], r["definition"]): r for r in ART["comparison_with_the_record"]["p8_4b"]}
for d in DEFS:
    check(f"D1.4 V4/{d} statement", p8[("V4", d)]["statement"], "Q0.d1")
    check(f"D1.4 V5/{d} statement", p8[("V5", d)]["statement"], "Q0.stop_rule")
    check(f"D1.4 V5/{d} outcome is the stop rule's", p8[("V5", d)]["this_artifact"]["outcome"],
          stmt("Q0.stop_rule", "mappo1000", d)["verdict"])
    for k in ("mean_difference", "ci95_low", "ci95_high"):
        check(f"D1.4 V5/{d}.{k} is d4's", p8[("V5", d)]["this_artifact"][k], stmt("Q0a", "mappo1000", d)["numbers"][k])
        check(f"D1.4 V4/{d}.{k} is d1's", p8[("V4", d)]["this_artifact"][k], stmt("Q0.d1", "mappo1000", d)["numbers"][k])

# ---- D1.5 definitions
for k in ("changes_flags", "reverses_on_n_seeds", "aggregates"):
    check(f"D1.5 definitions.{k} present", k in ART["definitions"], True)


# ---- D1.6 and D1.8: walk every contrast block
def blocks(node, path=""):
    if isinstance(node, dict):
        if isinstance(node.get("paired_comparison"), dict) and "left_arm" in node:
            yield path, node
        for k, v in node.items():
            yield from blocks(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from blocks(v, f"{path}[{i}]")


n_label = n_small = 0
for path, b in blocks(ART):
    arms = [b["left_arm"].split("@")[0], b["right_arm"].split("@")[0]]
    want = [a for a in arms if a in ("behaviour", "random")]
    note = b["per_seed"]["note"]
    has = "evaluation labels, not training seeds" in note
    check(f"D1.6 {path} label iff a behaviour/random arm", has, bool(want))
    n_label += has
    nu = b["paired_comparison"]["wilcoxon"]["n_used"]
    small = "n >= 20" in b["paired_comparison"]["p_value_note"]
    check(f"D1.8 {path} small-sample note iff 0 < n_used < 20", small, 0 < nu < 20)
    n_small += small
print("D1.6 blocks labelled:", n_label, "| D1.8 blocks with the small-sample note:", n_small)

# ---- D1.7
for s in ART["statements"]:
    if s["id"] == "Q2a":
        check(f"D1.7 Q2a/{s['tier']}/{s['definition']} tiers_note", "fixedtime" in s["numbers"].get("tiers_note", ""), True)
    if s["id"] == "Q6":
        n = s["numbers"]
        check(f"D1.7 Q6/{s['tier']}/{s['arm']}/{s['definition']} expectation_met",
              n["registered_expectation_met"], s["verdict"] == n["registered_expectation"])
    if s["id"] == "Q4b":
        check(f"D1.7 Q4b/{s['definition']} cites 673-675", "673-675" in s["numbers"]["form"], True)


def carries(node):
    if isinstance(node, dict):
        return any(k in node for k in ("registered_value", "predicted", "predicted_first")) or any(
            carries(v) for v in node.values())
    if isinstance(node, list):
        return any(carries(v) for v in node)
    return False


for s in ART["statements"]:
    if carries(s["numbers"]):
        pro = s["numbers"].get("predictions_registered_on", "")
        check(f"D1.7 {s['id']}/{s['tier']}/{s['definition']}/{s['version']} predictions_registered_on", bool(pro), True)
        if s["definition"] == "att_engine":
            check(f"D1.7 {s['id']}/{s['tier']}/att_engine says not att_engine", "not att_engine" in pro, True)
check("D1.7 audit rule no longer 'verbatim'", "verbatim" in ART["descriptive_collapse_audit"]["rule"], False)
check("D1.7 what_this_does_not_say[0] covers Q3-Q5", all(x in ART["what_this_does_not_say"][0] for x in ("Q3b", "Q4", "Q5")),
      True)

bad = [o for o in out if not o[1]]
print(f"checks: {len(out)}; failures: {len(bad)}")
for where, _, mine, theirs in bad:
    print(f"FAIL {where}: mine={mine!r} artifact={theirs!r}")
