"""Coordinator mutants for P5.2c's G5 fix round (BRIEF_45 Amendment D): the G3 harness's 44 mutants on C8's code commit
1d6519e, plus 14 new ones (CF1-CF14) aimed at D1 and D2, written by the coordinator from the module diff at 1a23dd4
without reading mutate.py's strings.

The 44 are imported verbatim from the committed docs/reviews/P5.2c_G3/coord_mutants_g3.py.  Each mutant is an exact string
replacement in a work copy of the 1d6519e export (module and tests byte-identical to 1a23dd4's; 1d6519e changes only
mutate.py), its count asserted.  The committed artifact is REMOVED from the export, so T-regress skips and every kill is
the tests' own.  The whole tests/test_tier_statements.py runs with -x and the real output root; the file is restored and
its bytes verified.  Exit 0 = SURVIVED, 1 = KILLED, anything else = ERROR.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SP = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g5fix")
BASE = SP / "base"
WORK = SP / "work"
TARGET = "offline/tier_statements.py"
PY = "/home/filip/rltraffic/.venv/bin/pytest"
G3_HARNESS = Path("/home/filip/rltraffic/docs/reviews/P5.2c_G3/coord_mutants_g3.py")


def g3_mutants() -> list[tuple[str, str, str, str, int]]:
    spec = importlib.util.spec_from_file_location("coord_mutants_g3", G3_HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.MUTANTS)


MUTANTS: list[tuple[str, str, str, str, int]] = g3_mutants()
assert len(MUTANTS) == 44, len(MUTANTS)

MUTANTS += [
    ("CF1", "D1.1: the 'collapses alone' tiers taken from Q5b's resolving tiers (the old reading)",
     "alone = [tier for tier in tiers if collapsed[(global_filter, tier)] and not collapsed[(per_intersection, tier)]]",
     "alone = [tier for tier in tiers if tier in resolves]", 1),
    ("CF2", "D1.1: the both-collapse tiers never reported",
     "both = [tier for tier in tiers if collapsed[(global_filter, tier)] and collapsed[(per_intersection, tier)]]",
     "both = []", 1),
    ("CF3", "D1.2: the summary point read from numbers only (Q3c null again)",
     'source = numbers if "mean_difference" in numbers else numbers.get("contrast", {})',
     "source = numbers", 1),
    ("CF4", "D1.3: the att_engine column computed under att_ours",
     'definition=PRIMARY_DEFINITION, version="corrected")',
     'definition="att_ours", version="corrected")', 1),
    ("CF5", "D1.3: att_engine_equal always True",
     '"att_engine_equal": normalise(engine) == normalise(row.recorded)}',
     '"att_engine_equal": True}', 1),
    ("CF6", "D1.4: V5 reported as Q0a again",
     '("V5", H4_CELLS, "Q0a", ts.stop_rule_verdict, "Q0.stop_rule",',
     '("V5", H4_CELLS, "Q0a", ts.stop_rule_verdict, "Q0a",', 1),
    ("CF7", "D1.6: the random anchor's seed labels not marked as evaluation labels",
     'EVALUATION_LABEL_METHODS: tuple[str, ...] = ("behaviour", "random")',
     'EVALUATION_LABEL_METHODS: tuple[str, ...] = ("behaviour",)', 1),
    ("CF8", "D1.7: registered_expectation_met always True",
     '"registered_expectation_met": verdict == expectation,',
     '"registered_expectation_met": True,', 1),
    ("CF9", "D1.8: the small-sample bound lowered to 2 (n_used = 5 no longer flagged)",
     "SMALL_SAMPLE_PAIRS = 20",
     "SMALL_SAMPLE_PAIRS = 2", 1),
    ("CF10", "D1.9(a): the temporary unlinked whether or not this call created it",
     "        if created:\n",
     "        if True:\n", 1),
    ("CF11", "D1.9(b): the git toplevel check disabled",
     'if toplevel is not None and Path(toplevel).resolve() != Path(block["code_root"]).resolve():',
     'if False and toplevel is not None and Path(toplevel).resolve() != Path(block["code_root"]).resolve():', 1),
    ("CF12", "D2.1 (the reviewer's R1): changed_by_rule_r read from the verdict only",
     '"changed_by_rule_r": outcome[1] != outcome[2]',
     '"changed_by_rule_r": steps[1]["verdict"] != steps[2]["verdict"]', 1),
    ("CF13", "D2.2 (the reviewer's R2): Q4 random's registered value transposed in REGISTERED_VALUES only",
     '"Q4": {"mappo1000": 39.5649, "maxpressure": 41.7402, "fixedtime": 51.3689, "random": 62.7106},',
     '"Q4": {"mappo1000": 39.5649, "maxpressure": 41.7402, "fixedtime": 51.3689, "random": 62.7016},', 1),
    ("CF14", "D1.7: predictions_registered_on never added",
     'if "predictions_registered_on" not in numbers and _carries_registered_value(numbers):',
     'if False:', 1),
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
    dry = "--dry" in sys.argv
    pristine = (BASE / TARGET).read_text(encoding="utf-8")
    pristine_sha = sha(BASE / TARGET)
    bad = [(mid, pristine.count(old), count) for mid, _, old, _, count in MUTANTS if pristine.count(old) != count]
    print(f"{len(MUTANTS)} mutants; pattern-count mismatches: {bad}", flush=True)
    if dry:
        return 0 if not bad else 3
    assert not (BASE / "docs" / "data" / "p5_2_statements.json").exists(), "the export must not hold the artifact"
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(BASE, WORK, symlinks=True)
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
    (SP / "coord_mutants_g5fix_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert sha(WORK / TARGET) == pristine_sha
    assert sha(BASE / TARGET) == pristine_sha
    print("pristine sha", pristine_sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
