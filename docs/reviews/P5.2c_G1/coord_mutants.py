"""Coordinator mutants for P5.2c gate G1 (BRIEF_45 section 6), independent of the implementer's mutate.py.

Each mutant is an exact string replacement in a work copy of the 719d753 export, its count asserted; the whole
tests/test_tier_statements.py runs with -x against the work copy with the real output root; the file is restored from
the pristine export and its bytes verified.  Exit 0 = SURVIVED, exit 1 = KILLED (a test failed), anything else = ERROR
(collection/import failure: not a valid kill).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SP = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g1")
BASE = SP / "base"
WORK = SP / "work"
TARGET = "offline/tier_statements.py"
PY = "/home/filip/rltraffic/.venv/bin/pytest"

MUTANTS = [
    ("CM1", "Q0b: an exactly-zero I scored HELD instead of NOT RESOLVED (F-14: untested branch)",
     'q0b, q0b_reading = "NOT RESOLVED", "exactly zero (identical episodes)',
     'q0b, q0b_reading = "HELD", "exactly zero (identical episodes)', 1),
    ("CM2", "composite I: CI half-width doubled, mean unchanged",
     "    stats = mean_ci95(values)\n    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95\n    seeds = sorted(",
     "    stats = mean_ci95(values)\n    low, high = stats.mean - 2 * stats.ci95, stats.mean + 2 * stats.ci95\n"
     "    seeds = sorted(", 1),
    ("CM3", "Q3b sub-prediction: random no longer predicted to resolve",
     'predicted_to_resolve = {"maxpressure": False, "fixedtime": False, "random": True}',
     'predicted_to_resolve = {"maxpressure": False, "fixedtime": False, "random": False}', 1),
    ("CM4", "Q1b scored with LEVEL_BAND (0.30) instead of ANCHOR_BAND (0.10)",
     "cells=(cell,), band=ts.ANCHOR_BAND, threshold=1)",
     "cells=(cell,), band=ts.LEVEL_BAND, threshold=1)", 1),
    ("CM5", "Q4a scored on four tiers (mappo1000 added)",
     '    outcomes = {}\n    for tier in OUT_OF_SAMPLE_TIERS:\n        outcomes[tier] = interval_outcome(blocks[tier]',
     '    outcomes = {}\n    for tier in TIERS:\n        outcomes[tier] = interval_outcome(blocks[tier]', 1),
    ("CM6", "Q5a scored on four tiers (mappo1000 added)",
     '("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in OUT_OF_SAMPLE_TIERS}),',
     '("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in TIERS}),', 1),
    ("CM7", "leaders: per-seed first places read att_ours whatever the definition",
     "seed_levels = _seed_levels(book, tier, definition, ts.METHODS)",
     'seed_levels = _seed_levels(book, tier, "att_ours", ts.METHODS)', 1),
    ("CM8", "collapse audit: the sentence holds if ANY non-DT arm collapses somewhere",
     '"holds": dt_never and all(collapsing.values())})',
     '"holds": dt_never and any(collapsing.values())})', 1),
    ("CM9", "contrast: D9 per-seed block always on att_ours",
     "left_name=left.spec.arm, right_name=right.spec.arm,\n                                 metric=definition)",
     'left_name=left.spec.arm, right_name=right.spec.arm,\n                                 metric="att_ours")', 1),
    ("CM10", "Q6: COLLAPSED iff the CI's HIGH end is above zero",
     'verdict = "COLLAPSED" if block["ci95_low"] > 0.0 else "NOT COLLAPSED"\n            entries.append(_entry("Q6"',
     'verdict = "COLLAPSED" if block["ci95_high"] > 0.0 else "NOT COLLAPSED"\n            entries.append(_entry("Q6"',
     1),
    ("CM11", "Q5c: random's predicted sign flipped to negative",
     '"fixedtime": "positive", "random": "positive"}),',
     '"fixedtime": "positive", "random": "negative"}),', 1),
    ("CM12", "Q2a: a tie for first counted as the predicted unique first place",
     '"outcome": "HELD" if firsts == [predicted[0]] else "FAILED",',
     '"outcome": "HELD" if predicted[0] in firsts else "FAILED",', 1),
    ("CM13", "contrast: exactly_zero if ANY per-draw difference is zero",
     "    exactly_zero = all(value == 0.0 for value in differences)",
     "    exactly_zero = any(value == 0.0 for value in differences)", 1),
    ("CM14", "Q4b: the chain scored non-strictly (<=)",
     "increasing = all(a < b for a, b in zip(values, values[1:]))",
     "increasing = all(a <= b for a, b in zip(values, values[1:]))", 1),
    ("CM15", "composite I: per-seed values computed as d1 - d4 (sign flipped)",
     "    by_seed = {str(seed): (seed_mean(episodes[0], seed) - seed_mean(episodes[1], seed))\n"
     "               - (seed_mean(episodes[2], seed) - seed_mean(episodes[3], seed)) for seed in seeds}",
     "    by_seed = {str(seed): (seed_mean(episodes[2], seed) - seed_mean(episodes[3], seed))\n"
     "               - (seed_mean(episodes[0], seed) - seed_mean(episodes[1], seed)) for seed in seeds}", 1),
    ("CM16", "leaders: the lead 'resolves' when the CI's LOW end is below zero",
     'resolved = [block["ci95_high"] < 0.0 for block in contrasts]',
     'resolved = [block["ci95_low"] < 0.0 for block in contrasts]', 1),
    ("CM17", "Q3b: the aggregate ignores fixedtime (two tiers only)",
     "                          verdict=conjunction(list(outcomes.values())),\n"
     '                          numbers={"tiers": outcomes, "rule": "sign negative on all three',
     "                          verdict=conjunction([o for t, o in outcomes.items() if t != \"fixedtime\"]),\n"
     '                          numbers={"tiers": outcomes, "rule": "sign negative on all three', 1),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tests() -> tuple[int, str]:
    env = dict(os.environ, RLTRAFFIC_OUTPUT_ROOT="/home/filip/rltraffic/output", PYTHONPATH=str(WORK))
    proc = subprocess.run([PY, "-q", "-x", "-p", "no:cacheprovider", "tests/test_tier_statements.py"],
                          cwd=WORK, env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    first = next((line for line in out.splitlines() if line.startswith(("FAILED", "ERROR"))), "")
    tail = [line for line in out.splitlines() if line.strip()][-1:]
    return proc.returncode, (first or (tail[0] if tail else ""))


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(BASE, WORK, symlinks=True)
    pristine = (BASE / TARGET).read_text(encoding="utf-8")
    pristine_sha = sha(BASE / TARGET)
    results = []
    code, line = run_tests()
    results.append({"id": "CONTROL", "exit": code, "first": line})
    print(f"CONTROL exit={code} {line}", flush=True)
    if code != 0:
        print("control failed: abort")
        return 2
    for mid, title, old, new, count in MUTANTS:
        found = pristine.count(old)
        if found != count:
            results.append({"id": mid, "title": title, "status": f"ERROR: pattern found {found} times, expected {count}"})
            print(f"{mid} PATTERN COUNT {found} != {count}", flush=True)
            continue
        (WORK / TARGET).write_text(pristine.replace(old, new), encoding="utf-8")
        code, line = run_tests()
        status = {0: "SURVIVED", 1: "KILLED"}.get(code, f"ERROR(exit {code})")
        (WORK / TARGET).write_text(pristine, encoding="utf-8")
        assert sha(WORK / TARGET) == pristine_sha, "restore failed"
        results.append({"id": mid, "title": title, "status": status, "exit": code, "first": line})
        print(f"{mid} {status}: {title} | {line}", flush=True)
    (SP / "coord_mutants_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert sha(WORK / TARGET) == pristine_sha
    return 0


if __name__ == "__main__":
    sys.exit(main())
