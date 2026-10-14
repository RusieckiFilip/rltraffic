"""Coordinator mutants for P5.2c gate G3 (BRIEF_45 Amendment B.2, B.2.4): the harness re-run on C6d's code commit.

The 38 mutants of the C6c check are imported verbatim from the committed harness docs/reviews/P5.2c_C6c/coord_mutants_g2c.py
(on main), which itself imports the 32 of G1.1.  Six new mutants (CD1-CD6) aim at B.2.3's labels, written by the
coordinator from the module diff at de4e94f without reading the implementer's mutate.py strings.  CD6 is a boundary
mutant (n_used = 1), included to measure, not prescribed by B.2.3.

Each mutant is an exact string replacement in a work copy of the de4e94f export (its module and test file are
byte-identical to 6267f68's, b5b8a98's, 9fa93e9's and 870e71a's), its count asserted; the whole
tests/test_tier_statements.py runs with -x against the work copy with the real output root; the file is restored and
its bytes verified.  The export holds no docs/data/p5_2_statements.json, so T-regress skips and every kill is the tests'
own.  Exit 0 = SURVIVED, 1 = KILLED, anything else = ERROR (not a valid kill).
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

SP = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g3")
BASE = SP / "base"
WORK = SP / "work"
TARGET = "offline/tier_statements.py"
PY = "/home/filip/rltraffic/.venv/bin/pytest"
G2C_HARNESS = Path("/home/filip/rltraffic/docs/reviews/P5.2c_C6c/coord_mutants_g2c.py")


def g2c_mutants() -> list[tuple[str, str, str, str, int]]:
    spec = importlib.util.spec_from_file_location("coord_mutants_g2c", G2C_HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.MUTANTS)


MUTANTS: list[tuple[str, str, str, str, int]] = g2c_mutants()
assert len(MUTANTS) == 38, len(MUTANTS)

MUTANTS += [
    ("CD1", "B.2.3: p_value_defined forced True",
     '            "p_value_defined": defined,\n',
     '            "p_value_defined": True,\n', 1),
    ("CD2", "B.2.3: p_value_defined forced False",
     '            "p_value_defined": defined,\n',
     '            "p_value_defined": False,\n', 1),
    ("CD3", "B.2.3: p_value_defined read from n_zero == 0 instead of n_used > 0",
     '            "p_value_defined": defined,\n',
     '            "p_value_defined": int(wilcoxon.n_zero) == 0,\n', 1),
    ("CD4", "B.2.3: the two p-value notes swapped",
     '"(BRIEF_45 Amendment B.2, B.2.3)" if defined else\n',
     '"(BRIEF_45 Amendment B.2, B.2.3)" if not defined else\n', 1),
    ("CD5", "B.2.3: the defined note no longer says z's sign is not the direction",
     '"never read by a verdict, and its z is computed from min(W+, W-), so the sign of z is not the "\n'
     '                "direction of the difference',
     '"never read by a verdict, and its z is computed from min(W+, W-), so the sign of z is the "\n'
     '                "direction of the difference', 1),
    ("CD6", "boundary (measured, not prescribed): both labels defined only from n_used > 1",
     "    defined = int(wilcoxon.n_used) > 0\n",
     "    defined = int(wilcoxon.n_used) > 1\n", 1),
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
    (SP / "coord_mutants_g3_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert sha(WORK / TARGET) == pristine_sha
    assert sha(BASE / TARGET) == pristine_sha
    print("pristine sha", pristine_sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
