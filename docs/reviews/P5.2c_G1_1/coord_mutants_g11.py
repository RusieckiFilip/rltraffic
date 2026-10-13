"""Coordinator mutants for P5.2c gate G1.1 (BRIEF_45 Amendment B, B3), independent of the implementer's mutate.py.

The seventeen G1 mutants are imported verbatim from the committed harness docs/reviews/P5.2c_G1/coord_mutants.py (on
main), with two changes Amendment B makes necessary: CM1 is reversed per B2.1 (the registered inclusion rule now gives
HELD on an exactly-zero I, so the mutant scores it NOT RESOLVED), and CM15's code is gone (B2.2 replaced the restated
per-seed loop with the imported worker), so it is re-aimed twice, by the coordinator, at the new composite rows.  Then
fourteen new mutants aimed at the B2 rulings, written by the coordinator without reading the implementer's strings.

Each mutant is an exact string replacement in a work copy of the 5bbb540 export (the code commit; ca27b48 differs from it
only in evidence files), its count asserted; the whole tests/test_tier_statements.py runs with -x against the work copy
with the real output root; the file is restored and its bytes verified.  Exit 0 = SURVIVED, 1 = KILLED, anything else =
ERROR (not a valid kill).
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

SP = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g11")
BASE = SP / "base"
WORK = SP / "work"
TARGET = "offline/tier_statements.py"
PY = "/home/filip/rltraffic/.venv/bin/pytest"
G1_HARNESS = Path("/home/filip/rltraffic/docs/reviews/P5.2c_G1/coord_mutants.py")


def g1_mutants() -> list[tuple[str, str, str, str, int]]:
    spec = importlib.util.spec_from_file_location("coord_mutants_g1", G1_HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.MUTANTS)


G1 = {m[0]: m for m in g1_mutants()}
assert sorted(G1) == sorted(f"CM{i}" for i in range(1, 18)), sorted(G1)

MUTANTS: list[tuple[str, str, str, str, int]] = [
    ("CM1r", "Q0b: an exactly-zero I scored NOT RESOLVED (CM1 reversed per B2.1)",
     'q0b, q0b_reading = "HELD", "exactly zero (identical episodes)',
     'q0b, q0b_reading = "NOT RESOLVED", "exactly zero (identical episodes)', 1),
]
MUTANTS += [G1[f"CM{i}"] for i in range(2, 15)]
MUTANTS += [
    ("CM15a", "composite I per-seed: d4's inner difference sign-flipped (n4 - s4) in the worker's left rows",
     "definition: by_key[0][(seed, draw)] - by_key[1][(seed, draw)]}",
     "definition: by_key[1][(seed, draw)] - by_key[0][(seed, draw)]}", 1),
    ("CM15b", "composite I per-seed: the last (seed, draw) dropped from the worker's rows",
     "    keys = sorted(by_key[0])\n",
     "    keys = sorted(by_key[0])[:-1]\n", 1),
]
MUTANTS += [G1["CM16"], G1["CM17"]]
MUTANTS += [
    ("CB1a", "Q3b sub-prediction: the exact-zero source always says 'identical episodes'",
     '"identical episodes" if block["discriminability"]["identical"] else',
     '"identical episodes" if True else', 1),
    ("CB2", "composite I: the worker's names swapped (d1 named left, d4 right) -- the qualifier's text names the wrong pair",
     'left_name="d4", right_name="d1", metric=definition)',
     'left_name="d1", right_name="d4", metric=definition)', 1),
    ("CB3a", "B2.3: measured_first_under_rule not null under a pooled tie",
     'statements["q2a"]["measured_first_under_rule"] = order[0] if len(firsts) == 1 else None',
     'statements["q2a"]["measured_first_under_rule"] = order[0] if len(firsts) >= 1 else None', 1),
    ("CB3b", "B2.3: the reversal list compares only the first element, not the set of first places",
     'if set(block["firsts"]) != set(firsts)]',
     'if block["firsts"][0] != firsts[0]]', 1),
    ("CB4a", "B2.4: the running-file check reads only the parent directory's NAME",
     "        if package not in path.parents:\n",
     '        if path.parent.name != "offline":\n', 1),
    ("CB4b", "B2.4: provenance hashes the module found by import, not the running file",
     'path = running["this module\'s own file"] if name == "offline.tier_statements" else located(name)[0]',
     "path = located(name)[0]", 1),
    ("CB5a", "B2.5: the ulp bound made absolute (4 ulps of 1.0) instead of relative to the magnitude",
     "if abs(mine - recorded) > ULP_BOUND * math.ulp(max(abs(mine), abs(recorded))):",
     "if abs(mine - recorded) > ULP_BOUND * math.ulp(1.0):", 1),
    ("CB5b", "B2.5: the stop-rule record's ulp check skipped",
     "    for k in keys[:3]:\n        assert_within_ulps(d4[k]",
     "    for k in keys[:0]:\n        assert_within_ulps(d4[k]", 1),
    ("CB5c", "B2.5: P8.4b's ulp check on the first key only (the CI ends unchecked)",
     "            for k in keys:\n                assert_within_ulps(mine[k], recorded[k]",
     "            for k in keys[:1]:\n                assert_within_ulps(mine[k], recorded[k]", 1),
    ("CB6", "B2.6: GATE 2's quote cited one line short (1274-1280)",
     "Quote(_BRIEF_27, (1274, 1281),",
     "Quote(_BRIEF_27, (1274, 1280),", 1),
    ("CB9a", "B2.9: the out-path existence check on the RESOLVED path (a dangling symlink passes)",
     "    if unresolved.is_symlink() or unresolved.exists():\n",
     "    if resolved.exists():\n", 1),
    ("CB14a", "B2.14: CAMPAIGN_COMPLETE not listed (the manifest only)",
     "for name in (ar.MANIFEST_NAME, ar.COMPLETE_MARKER):  # BRIEF_45 Amendment B, B2.14",
     "for name in (ar.MANIFEST_NAME,):  # BRIEF_45 Amendment B, B2.14", 1),
    ("CB14b", "B2.14: both campaign files listed with the MANIFEST's sha256",
     'files.append({"file": f"output/{relative}", "sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest(),',
     'files.append({"file": f"output/{relative}", "sha256": hashlib.sha256((output / P8_4B_DIR / ar.MANIFEST_NAME)'
     '.read_bytes()).hexdigest(),', 1),
    ("CB0", "Q0b's reading on a resolving I: the inclusion rule written with strict inequalities (a CI touching zero "
            "excluded)",
     'elif interaction["ci95_low"] <= 0.0 <= interaction["ci95_high"]:',
     'elif interaction["ci95_low"] < 0.0 < interaction["ci95_high"]:', 1),
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
    (SP / "coord_mutants_g11_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert sha(WORK / TARGET) == pristine_sha
    assert sha(BASE / TARGET) == pristine_sha
    print("pristine sha", pristine_sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
