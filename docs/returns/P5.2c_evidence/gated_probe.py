"""B.2.3 (4)'s gated test alone against each of B.2.3's four mutants, reusing mutate.py's own export, replacement and
run machinery.  The run of record lists each mutant with a synthetic killing test, and ``pytest -x`` stops at the
first failure, so one run cannot evidence both; this is the other half.

    python gated_probe.py <repo> <commit> <work dir> <output root>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO, COMMIT, WORK, OUTPUT_ROOT = (Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
GATED = "tests/test_tier_statements.py::test_b23_real_every_p_value_defined_follows_n_used_and_exactly_eight_are_undefined"

spec = importlib.util.spec_from_file_location("mutate", REPO / "docs/returns/P5.2c_evidence/mutate.py")
assert spec is not None and spec.loader is not None
mutate = importlib.util.module_from_spec(spec)
sys.modules["mutate"] = mutate            # the dataclasses in it look their module up by name
spec.loader.exec_module(mutate)

tree = WORK / "tree"
commit = mutate.export(REPO, COMMIT, tree)
print(f"# B.2.3 (4)'s gated test alone, on an exported copy of {commit}")
print(f"# the only test run: {GATED.split('::')[1]}")
chosen = [m for m in mutate.MUTANTS if m.mutant_id.startswith("B.2.3")]
assert len(chosen) == 4, [m.mutant_id for m in chosen]

code, output = mutate.run_tests(tree, (GATED,), OUTPUT_ROOT)
print(f"\n## control (no mutation): pytest exit {code} -- {output.strip().splitlines()[-1]}")
if code != 0:
    print(output)
    raise SystemExit("CONTROL FAILED")

verdicts = {}
for mutant in chosen:
    originals = {}
    try:
        for relative, old, new, count in mutant.replacements:
            path = tree / relative
            originals[relative] = path.read_text(encoding="utf-8")
            found = originals[relative].count(old)
            assert found == count, f"{mutant.mutant_id}: {found} occurrence(s), not {count}"
            path.write_text(originals[relative].replace(old, new), encoding="utf-8")
        code, output = mutate.run_tests(tree, (GATED,), OUTPUT_ROOT)
    finally:
        for relative, original in originals.items():
            (tree / relative).write_text(original, encoding="utf-8")
    verdict = "KILLED" if code == 1 else ("SURVIVED" if code == 0 else f"ERROR (pytest exit {code})")
    verdicts[mutant.mutant_id] = verdict
    print(f"\n## {mutant.mutant_id} -- {mutant.description}: {verdict}")
    for line in [line for line in output.splitlines() if line.startswith("E ")][:3]:
        print(f"   {line[:200]}")
    print(f"   {output.strip().splitlines()[-1]}")

print(f"\n## {sum(v == 'KILLED' for v in verdicts.values())} of {len(chosen)} KILLED by the gated test alone: {verdicts}")
raise SystemExit(0 if all(v == "KILLED" for v in verdicts.values()) else 1)
