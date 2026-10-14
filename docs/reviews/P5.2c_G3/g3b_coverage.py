"""P5.2c G3 addendum (2026-10-14, after G4): the blocks Amendment C's recompute did not compare.

g3_recompute.py verified every statement, leaders block, audit row, cell, ties block and label (3,518 comparisons, 0
differences).  It did not compare the 'this_artifact' side of comparison_with_the_record's summary sub-blocks, which
restate statement values: never_scored, p8_4b, p5_1_grid, p5_2_stop_rule_record and q5.  Here each restated value is
compared with the statement it restates (verified by g3_recompute.py), and q5's registered values with the registered
text (docs/plans/p5.2.md:683-689).  Whether each 'recorded' value quotes its source record is G5 mandate M2a's item 5.
Nothing from offline/ is imported."""
from __future__ import annotations

import json
import sys
from pathlib import Path

A = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/home/filip/rltraffic-p52c/docs/data/p5_2_statements.json").read_text(encoding="utf-8"))
C = A["comparison_with_the_record"]
S = {(s["id"], s["tier"], s["definition"], s["version"], s.get("arm")): s for s in A["statements"]}
out = []


def stmt(sid, tier, definition, version="both", arm=None):
    for v in (version, "both"):
        if (sid, tier, definition, v, arm) in S:
            return S[(sid, tier, definition, v, arm)]
    raise KeyError((sid, tier, definition, version, arm))


def point(s):
    """The statement's point estimate under the name its kind uses (None for an aggregate, which has none)."""
    n = s["numbers"]
    if "mean_difference" in n:
        return n["mean_difference"]
    if "mean" in n:          # Q3c stores its point estimate as numbers.mean (= numbers.contrast.mean_difference)
        assert n["mean"] == n["contrast"]["mean_difference"]
        return n["mean"]
    return None


def check(where, mine, theirs):
    out.append((where, mine == theirs, mine, theirs))


# never_scored
n_rows = 0
for entry in C["never_scored"]:
    for row in entry["this_artifact"]:
        n_rows += 1
        s = stmt(row["id"], row["tier"], row["definition"], row["version"], row.get("arm"))
        w = f"never_scored/{row['id']}/{row['tier']}/{row['definition']}/{row.get('arm')}"
        check(f"{w}.mean_difference", point(s), row["mean_difference"])
        check(f"{w}.ci95_low", s["numbers"].get("ci95_low"), row["ci95_low"])
        check(f"{w}.ci95_high", s["numbers"].get("ci95_high"), row["ci95_high"])
        check(f"{w}.verdict", s["verdict"], row["verdict"])
        check(f"{w}.reading", s["reading"], row["reading"])
# p8_4b: V4 is Q0.d1 (P5.1's P2a), V5 is Q0a's d4 (P5.2's stop rule)
for row in C["p8_4b"]:
    s = stmt(row["statement"], "mappo1000", row["definition"])
    n = s["numbers"]
    for k in ("mean_difference", "ci95_low", "ci95_high"):
        check(f"p8_4b/{row['contrast_id']}/{row['definition']}.{k}", n[k], row["this_artifact"][k])
    check(f"p8_4b/{row['contrast_id']}/{row['definition']}.n_paired", n["n_draws"], row["this_artifact"]["n_paired"])
# p5_1_grid: P1 is Q0.d1 under att_ours; P3 is Q6 at mappo1000 under att_ours, per DT arm
s = stmt("Q0.d1", "mappo1000", "att_ours")
for k, v in C["p5_1_grid"]["P1"].items():
    mine = s["numbers"]["paired_comparison"]["wilcoxon"]["p_value"] if k == "p_value" else (
        s["numbers"][k] if k in s["numbers"] else s["numbers"]["paired_comparison"][k])
    check(f"p5_1_grid/P1.{k}", mine, v["this_artifact"])
for arm in ("dt_spatial", "dt_nomix"):
    s = stmt("Q6", "mappo1000", "att_ours", arm=arm)
    for k, v in C["p5_1_grid"]["P3"][arm].items():
        if k == "outcome":
            mine = s["numbers"]["p5_1_outcome"]
        elif k == "p_value":
            mine = s["numbers"]["paired_comparison"]["wilcoxon"]["p_value"]
        else:
            mine = s["numbers"][k] if k in s["numbers"] else s["numbers"]["paired_comparison"][k]
        check(f"p5_1_grid/P3/{arm}.{k}", mine, v["this_artifact"])
# the stop-rule record: Q0a's d4 under att_ours
s = stmt("Q0a", "mappo1000", "att_ours")
for k in ("mean_difference", "ci95_low", "ci95_high"):
    check(f"stop_rule.{k}", s["numbers"][k], C["p5_2_stop_rule_record"]["this_artifact"][k])
check("stop_rule.verdict", stmt("Q0.stop_rule", "mappo1000", "att_ours")["verdict"],
      C["p5_2_stop_rule_record"]["this_artifact"]["verdict"])
# q5: the fixed-time measured values against the statements; the registered values against the plan text
REGISTERED_FIXEDTIME = {"Q5a": 820.65, "Q5b": -812.15, "Q5c": 8.49}     # docs/plans/p5.2.md:684, :686, :689
for row in C["q5"]:
    for sid, blk in row["fixedtime_registered_beside_measured"].items():
        check(f"q5/{row['record']}/{sid}.registered", REGISTERED_FIXEDTIME[sid], blk["registered"])
        for definition, m in blk["measured"].items():
            s = stmt(sid, "fixedtime", definition)
            for k in ("mean_difference", "ci95_low", "ci95_high", "exactly_zero"):
                check(f"q5/{row['record']}/{sid}/{definition}.{k}", s["numbers"][k], m[k])
            check(f"q5/{row['record']}/{sid}/{definition}.verdict", s["verdict"], m["verdict"])

bad = [o for o in out if not o[1]]
print(f"never_scored rows: {n_rows}; comparisons: {len(out)}; differences: {len(bad)}")
for where, _, mine, theirs in bad:
    print(f"DIFF {where}: statement={mine!r} block={theirs!r}")
