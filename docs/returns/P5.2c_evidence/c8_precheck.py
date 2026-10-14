"""C8's scratch pre-check (BRIEF_45 Amendment D, D5 step 2.1): the tests that are green on arrival -- D2's, over code
that is correct today -- each against its mutant on the UNCHANGED module.  A git archive export of the commit holding
that module, with the new test file copied in; each mutant an exact replacement counted once; only the named test run;
the module restored byte for byte after each.  Reuses mutate.py's export and run machinery.

    python precheck.py <repo> <commit> <test file> <work dir> <output root>
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

REPO, COMMIT, TEST_FILE, WORK, OUTPUT_ROOT = (Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]),
                                              Path(sys.argv[5]))
MODULE = "offline/tier_statements.py"
TESTS = "tests/test_tier_statements.py"

spec = importlib.util.spec_from_file_location("mutate", REPO / "docs/returns/P5.2c_evidence/mutate.py")
assert spec is not None and spec.loader is not None
mutate = importlib.util.module_from_spec(spec)
sys.modules["mutate"] = mutate
spec.loader.exec_module(mutate)

tree = WORK / "tree"
commit = mutate.export(REPO, COMMIT, tree)
shutil.copyfile(TEST_FILE, tree / TESTS)
pristine = (tree / MODULE).read_bytes()
print(f"# C8 pre-check on a git archive export of {commit}; module sha256 {hashlib.sha256(pristine).hexdigest()}")
print(f"# the test file copied in: sha256 {hashlib.sha256((tree / TESTS).read_bytes()).hexdigest()}")

CASES = (
    ("R1", "M1b's R1: changes_block's changed_by_rule_r read from the verdict alone",
     '"changed_by_rule_r": outcome[1] != outcome[2]', '"changed_by_rule_r": steps[1]["verdict"] != steps[2]["verdict"]',
     "test_d2_1_changes_flags_by_this_files_rule"),
    ("R2", "M1b's R2: REGISTERED_VALUES Q4 random 62.7106 -> 62.7016, in the table alone",
     '    "Q4": {"mappo1000": 39.5649, "maxpressure": 41.7402, "fixedtime": 51.3689, "random": 62.7106},',
     '    "Q4": {"mappo1000": 39.5649, "maxpressure": 41.7402, "fixedtime": 51.3689, "random": 62.7016},',
     "test_d2_2_every_registered_value_is_the_plans"),
    ("D2.4a", "every statement stored per version, an IQL-free one twice instead of once as both",
     '        if CORRECTED_IQL in corrected["inputs"] or ORIGINAL_IQL in original["inputs"]:',
     '        if True:',
     "test_t_versions_the_iql_free_statements_are_stored_once_and_the_iql_ones_twice"),
    ("D2.4b", "the P8.4b comparison emitted empty",
     '"p5_1_grid": p5_1, "p8_4b": p8,', '"p5_1_grid": p5_1, "p8_4b": [],',
     "test_t_inputs_the_synthetic_build_passes_every_comparison_of_the_record"),
)

code, output = mutate.run_tests(tree, tuple(f"{TESTS}::{case[4]}" for case in CASES), OUTPUT_ROOT)
print(f"\n## control (no mutation): pytest exit {code} -- {output.strip().splitlines()[-1]}")
if code != 0:
    print(output)
    raise SystemExit("CONTROL FAILED")
verdicts = {}
for mutant_id, description, old, new, test in CASES:
    text = pristine.decode("utf-8")
    assert text.count(old) == 1, (mutant_id, text.count(old))
    (tree / MODULE).write_text(text.replace(old, new), encoding="utf-8")
    try:
        code, output = mutate.run_tests(tree, (f"{TESTS}::{test}",), OUTPUT_ROOT)
    finally:
        (tree / MODULE).write_bytes(pristine)
    verdict = "KILLED" if code == 1 else ("SURVIVED" if code == 0 else f"ERROR (pytest exit {code})")
    verdicts[mutant_id] = verdict
    print(f"\n## {mutant_id} -- {description}: {verdict}")
    print(f"   test: {test}")
    for line in [line for line in output.splitlines() if line.startswith("E ")][:3]:
        print(f"   {line[:220]}")
    print(f"   {output.strip().splitlines()[-1]}")
assert (tree / MODULE).read_bytes() == pristine
print(f"\n## the module restored byte for byte; {sum(v == 'KILLED' for v in verdicts.values())} of {len(CASES)} KILLED: "
      f"{verdicts}")
raise SystemExit(0 if all(v == "KILLED" for v in verdicts.values()) else 1)
