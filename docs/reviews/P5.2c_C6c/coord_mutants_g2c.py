"""Coordinator mutants for P5.2c, start of gate G3 (BRIEF_45 Amendment B.1, B.1.3): the G1.1 harness re-run on C6c.

The 32 G1.1 mutants are imported verbatim from the committed harness docs/reviews/P5.2c_G1_1/coord_mutants_g11.py (on
main).  Six new mutants aimed at B.1.2(1)'s label ruling (I's identity), written by the coordinator from the module diff
at 12251e6 without reading the implementer's mutate.py strings.

Each mutant is an exact string replacement in a work copy of the 3e5de9c export (its module and tests equal 12251e6's;
3e5de9c changes only mutate.py), its count asserted; the whole tests/test_tier_statements.py runs with -x against the
work copy with the real output root; the file is restored and its bytes verified.  Exit 0 = SURVIVED, 1 = KILLED,
anything else = ERROR (not a valid kill).
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

SP = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g2c")
BASE = SP / "base"
WORK = SP / "work"
TARGET = "offline/tier_statements.py"
PY = "/home/filip/rltraffic/.venv/bin/pytest"
G11_HARNESS = Path("/home/filip/rltraffic/docs/reviews/P5.2c_G1_1/coord_mutants_g11.py")


def g11_mutants() -> list[tuple[str, str, str, str, int]]:
    spec = importlib.util.spec_from_file_location("coord_mutants_g11", G11_HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.MUTANTS)


MUTANTS: list[tuple[str, str, str, str, int]] = g11_mutants()
assert len(MUTANTS) == 32, len(MUTANTS)

MUTANTS += [
    ("CL1", "B.1.2(1): I's identity forced True",
     '    identical = identical_pairs["d4"] and identical_pairs["d1"]\n',
     "    identical = True\n", 1),
    ("CL2", "B.1.2(1): I's identity forced False",
     '    identical = identical_pairs["d4"] and identical_pairs["d1"]\n',
     "    identical = False\n", 1),
    ("CL3", "B.1.2(1): I identical when EITHER pair is",
     '    identical = identical_pairs["d4"] and identical_pairs["d1"]\n',
     '    identical = identical_pairs["d4"] or identical_pairs["d1"]\n', 1),
    ("CL4", "B.1.2(1): the not-identical exact zero scored NOT RESOLVED",
     'q0b, q0b_reading = "HELD", ("exactly zero on every draw, from episodes that are not identical',
     'q0b, q0b_reading = "NOT RESOLVED", ("exactly zero on every draw, from episodes that are not identical', 1),
    ("CL5", "B.1.2(1): d1's identity read from d4's cells",
     'for name, first in (("d4", 0), ("d1", 2))}',
     'for name, first in (("d4", 0), ("d1", 0))}', 1),
    ("CL6", "B.1.2(1): each pair's identity compares a cell with itself",
     "            {(episode.seed, episode.draw_id): episode.att_horizon for episode in episodes[first + 1]},\n",
     "            {(episode.seed, episode.draw_id): episode.att_horizon for episode in episodes[first]},\n", 1),
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
    (SP / "coord_mutants_g2c_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert sha(WORK / TARGET) == pristine_sha
    assert sha(BASE / TARGET) == pristine_sha
    print("pristine sha", pristine_sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
