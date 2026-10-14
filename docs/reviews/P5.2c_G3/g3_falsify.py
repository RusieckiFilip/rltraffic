"""Falsify the coordinator's G3 recompute: each case corrupts ONE value in a scratch copy of the committed artifact, and
g3_recompute.py must report at least one DIFF (or refuse) on it.  A case it passes silently means the recompute does not
check that value, so its 'all MATCH' would certify nothing there."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

SP = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g3")
SRC = Path("/home/filip/rltraffic-p52c/docs/data/p5_2_statements.json")
ART = json.loads(SRC.read_text(encoding="utf-8"))


def find(a, sid, tier, definition, version="both", arm=None):
    for s in a["statements"]:
        if (s["id"], s["tier"], s["definition"], s["version"], s.get("arm")) == (sid, tier, definition, version, arm):
            return s
    raise KeyError((sid, tier, definition, version, arm))


def nudge(x: float) -> float:
    return x * (1 + 1e-7) if x else 1e-7


CASES = {
    "Q5c.aggregate verdict FAILED -> NOT RESOLVED (att_engine)":
        lambda a: find(a, "Q5c.aggregate", None, "att_engine").__setitem__("verdict", "NOT RESOLVED"),
    "Q0b mean_difference nudged 1e-7 relative":
        lambda a: find(a, "Q0b", "mappo1000", "att_engine")["numbers"].__setitem__(
            "mean_difference", nudge(find(a, "Q0b", "mappo1000", "att_engine")["numbers"]["mean_difference"])),
    "Q0b by_seed[101] nudged":
        lambda a: find(a, "Q0b", "mappo1000", "att_ours")["numbers"]["per_seed"]["by_seed"].__setitem__(
            "101", nudge(find(a, "Q0b", "mappo1000", "att_ours")["numbers"]["per_seed"]["by_seed"]["101"])),
    "Q0b qualifier names d1 vs d4":
        lambda a: find(a, "Q0b", "mappo1000", "att_ours")["numbers"].__setitem__(
            "seed_reversal_qualifier",
            find(a, "Q0b", "mappo1000", "att_ours")["numbers"]["seed_reversal_qualifier"].replace("d4 vs d1", "d1 vs d4")),
    "Q1 corrected n_held 15 -> 14":
        lambda a: find(a, "Q1", None, "att_ours", "corrected")["numbers"].__setitem__("n_held", 14),
    "Q1 one cell's held flag flipped":
        lambda a: find(a, "Q1", None, "att_engine", "as_committed")["numbers"]["cells"][0].__setitem__(
            "held", not find(a, "Q1", None, "att_engine", "as_committed")["numbers"]["cells"][0]["held"]),
    "Q2b fixedtime range widened":
        lambda a: find(a, "Q2b", "fixedtime", "att_ours")["numbers"].__setitem__(
            "range", [find(a, "Q2b", "fixedtime", "att_ours")["numbers"]["range"][0],
                      find(a, "Q2b", "fixedtime", "att_ours")["numbers"]["range"][1] + 1]),
    "Q2b_hard random n_concordant +1":
        lambda a: find(a, "Q2b_hard", "random", "att_engine")["numbers"].__setitem__(
            "n_concordant", find(a, "Q2b_hard", "random", "att_engine")["numbers"]["n_concordant"] + 1),
    "D9 fixedtime seed 303 hard range changed":
        lambda a: find(a, "D9", "fixedtime", "att_ours")["numbers"]["seeds"]["303"].__setitem__(
            "hard_n_concordant_range", [0, 6]),
    "Q3a fixedtime rank range collapsed":
        lambda a: find(a, "Q3a", "fixedtime", "att_ours")["numbers"].__setitem__("rank_range", [2, 2]),
    "Q3c random corrected best_non_dt -> bc":
        lambda a: find(a, "Q3c", "random", "att_engine", "corrected")["numbers"].__setitem__("best_non_dt", "bc"),
    "Q3b fixedtime sub_prediction matched flipped":
        lambda a: find(a, "Q3b", "fixedtime", "att_ours")["numbers"]["sub_prediction"].__setitem__(
            "matched", not find(a, "Q3b", "fixedtime", "att_ours")["numbers"]["sub_prediction"]["matched"]),
    "Q4b strictly_decreasing flipped":
        lambda a: find(a, "Q4b", None, "att_ours")["numbers"].__setitem__("strictly_decreasing", False),
    "Q6 random dt_spatial ci95_low nudged":
        lambda a: find(a, "Q6", "random", "att_engine", "both", "dt_spatial")["numbers"].__setitem__(
            "ci95_low", nudge(find(a, "Q6", "random", "att_engine", "both", "dt_spatial")["numbers"]["ci95_low"])),
    "audit row verdict flipped (bc random)":
        lambda a: next(r for r in a["descriptive_collapse_audit"]["rows"]
                       if (r["arm"], r["tier"], r["definition"]) == ("bc", "random", "att_engine")).__setitem__(
            "verdict", "NOT COLLAPSED"),
    "audit sentence collapse_on edited":
        lambda a: a["descriptive_collapse_audit"]["sentence_holds"][0]["non_dt_arms_collapse_on"].__setitem__("bc", []),
    "leaders fixedtime per-seed first place edited":
        lambda a: next(L for L in a["leaders"] if (L["tier"], L["definition"]) == ("fixedtime", "att_engine"))[
            "per_seed_first_places"].__setitem__("101", ["bc"]),
    "leaders random runner-up contrast high nudged":
        lambda a: next(L for L in a["leaders"] if (L["tier"], L["definition"], L["version"]) ==
                       ("random", "att_ours", "as_committed"))["leader_minus_runner_up"][0].__setitem__(
            "ci95_high", nudge(next(L for L in a["leaders"] if (L["tier"], L["definition"], L["version"]) ==
                                    ("random", "att_ours", "as_committed"))["leader_minus_runner_up"][0]["ci95_high"])),
    "cell median nudged (p5_2b iql att_engine)":
        lambda a: a["cells"]["p5_2b:iql@random"]["att_engine"].__setitem__(
            "median", nudge(a["cells"]["p5_2b:iql@random"]["att_engine"]["median"])),
    "cell companions never_entered count +1":
        lambda a: a["cells"]["p5_1:bc@grid4x4_mappo1000"]["companions"].__setitem__(
            "episodes_with_never_entered", a["cells"]["p5_1:bc@grid4x4_mappo1000"]["companions"]["episodes_with_never_entered"] + 1),
    "ties: one identical-episode pair dropped (fixedtime att_ours)":
        lambda a: next(T for T in a["ties"] if (T["tier"], T["definition"]) == ("fixedtime", "att_ours"))[
            "identical_episode_pairs"].pop(),
    "definition_dependent flipped (Q3c maxpressure att_ours)":
        lambda a: find(a, "Q3c", "maxpressure", "att_ours").__setitem__(
            "definition_dependent", not find(a, "Q3c", "maxpressure", "att_ours")["definition_dependent"]),
    "primary flipped (Q2a random corrected att_engine)":
        lambda a: find(a, "Q2a", "random", "att_engine", "corrected").__setitem__("primary", False),
    "a statement removed (Q5b random att_ours)":
        lambda a: a["statements"].remove(find(a, "Q5b", "random", "att_ours")),
}


def main() -> int:
    caught = 0
    for name, mutate in CASES.items():
        a = copy.deepcopy(ART)
        mutate(a)
        path = SP / "falsify_artifact.json"
        path.write_text(json.dumps(a), encoding="utf-8")
        proc = subprocess.run([sys.executable, "-P", str(SP / "g3_recompute.py"), str(path)], capture_output=True,
                              text=True, cwd="/tmp")
        first = proc.stdout.splitlines()[0] if proc.stdout else ""
        diff_lines = [line for line in proc.stdout.splitlines() if line.startswith("DIFF")]
        ok = proc.returncode != 0 or bool(diff_lines)
        caught += ok
        why = (f"exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1][:110]}" if proc.returncode
               else (diff_lines[0][:150] if diff_lines else first))
        print(f"{'CAUGHT' if ok else 'MISSED'}  {name}  |  {why}", flush=True)
    path.unlink(missing_ok=True)
    print(f"{caught} of {len(CASES)} caught")
    return 0 if caught == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
