"""B.2.2(3)'s four falsifications of the erfc guard, each in a clean ``git archive`` export of the artifact commit and
never in the worktree (``BRIEF_45`` Amendment B.2).  Reuses ``mutate.py``'s export and run machinery.

    python falsify.py <repo> <commit> <work dir> <output root>

Exit 0 iff the control passes and all four falsifications fail for their stated reason.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO, COMMIT, WORK, OUTPUT_ROOT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
GUARD = "tests/test_erfc_determinism.py::test_the_committed_p_values_all_still_reproduce_exactly"
ARTIFACT = "docs/data/p5_2_statements.json"
TEST = "tests/test_erfc_determinism.py"

spec = importlib.util.spec_from_file_location("mutate", REPO / "docs/returns/P5.2c_evidence/mutate.py")
assert spec is not None and spec.loader is not None
mutate = importlib.util.module_from_spec(spec)
sys.modules["mutate"] = mutate
spec.loader.exec_module(mutate)

tree = WORK / "tree"
commit = mutate.export(REPO, COMMIT, tree)
print(f"# B.2.2(3): the four falsifications, on a git archive export of {commit}")
print(f"# the only test run: {GUARD.split('::')[1]}")
print(f"# the export's artifact: {ARTIFACT}")


def wilcoxon_paths(node: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every dict carrying a numeric ``z`` and a ``p_value``: the guard's own rule, by this script's own walk."""
    found: list[tuple[Any, ...]] = []
    if isinstance(node, dict):
        if "z" in node and "p_value" in node and isinstance(node.get("z"), (int, float)):
            found.append(path)
        for key, value in node.items():
            found.extend(wilcoxon_paths(value, path + (key,)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(wilcoxon_paths(value, path + (index,)))
    return found


def at(payload: Any, path: tuple[Any, ...]) -> Any:
    for step in path:
        payload = payload[step]
    return payload


original_artifact = (tree / ARTIFACT).read_bytes()
original_test = (tree / TEST).read_text(encoding="utf-8")
payload = json.loads(original_artifact)
paths = sorted(wilcoxon_paths(payload), key=lambda p: [str(x) for x in p])
print(f"# this script's own walk of the artifact finds {len(paths)} (z, p_value) pairs")
assert len(paths) == 136, len(paths)
chosen = paths[0]
rendered = "".join(f"[{step!r}]" for step in chosen)
print(f"# the pair it edits: {rendered}")
print(f"#   {json.dumps(at(payload, chosen), sort_keys=True)}")


def restore() -> None:
    (tree / ARTIFACT).write_bytes(original_artifact)
    (tree / TEST).write_text(original_test, encoding="utf-8")


def run(label: str, expect: str) -> bool:
    code, output = mutate.run_tests(tree, (GUARD,), OUTPUT_ROOT)
    lines = [line for line in output.splitlines() if line.startswith("E ") and expect in line]
    if not lines:
        lines = [line for line in output.splitlines() if line.startswith("E ")][:2]
    print(f"\n## {label}: pytest exit {code} -- {'FAILS as required' if code == 1 else 'DID NOT FAIL'}")
    for line in lines[:2]:
        print(f"   {line[:420]}")
    print(f"   {output.strip().splitlines()[-1]}")
    matched = code == 1 and any(expect in line for line in output.splitlines())
    print(f"   expected reason {expect!r} present: {matched}")
    return matched


results = {}
code, output = mutate.run_tests(tree, (GUARD,), OUTPUT_ROOT)
print(f"\n## control (nothing falsified): pytest exit {code} -- {output.strip().splitlines()[-1]}")
if code != 0:
    print(output)
    raise SystemExit("CONTROL FAILED: the guard must pass at the artifact commit before it can be falsified")

# (1) the literal set back to 351.
text = original_test
assert text.count("assert len(pairs) == 487, (") == 1
(tree / TEST).write_text(text.replace("assert len(pairs) == 487, (", "assert len(pairs) == 351, ("), encoding="utf-8")
results["1 the literal back to 351"] = run("(1) the pinned literal set back to 351", "found 487")
restore()

# (2) one of the 136 p_values moved by one ulp.
payload = json.loads(original_artifact)
block = at(payload, chosen)
was = block["p_value"]
block["p_value"] = math.nextafter(was, math.inf)
assert block["p_value"] != was
(tree / ARTIFACT).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"\n#   (2) moves {rendered}.p_value from {was!r} to {block['p_value']!r} (one ulp up)")
results["2 one p_value moved by one ulp"] = run("(2) one of the 136 p_values moved by one ulp", "would change")
restore()

# (3) one of the 136 given an alternative key.
payload = json.loads(original_artifact)
at(payload, chosen)["alternative"] = "less"
(tree / ARTIFACT).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
results["3 an alternative key added"] = run("(3) one of the 136 given \"alternative\": \"less\"",
                                            "not exactly P5.3c's twelve")
restore()

# (4) one of the 136 Wilcoxon blocks deleted.
payload = json.loads(original_artifact)
parent, key = at(payload, chosen[:-1]), chosen[-1]
del parent[key]
(tree / ARTIFACT).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
results["4 a Wilcoxon block deleted"] = run(f"(4) the Wilcoxon block at {rendered} deleted", "found 486")
restore()

assert (tree / ARTIFACT).read_bytes() == original_artifact, "the export's artifact was not restored"
assert (tree / TEST).read_text(encoding="utf-8") == original_test, "the export's test was not restored"
print("\n## the export's artifact and test restored byte for byte")
print(f"\n## {sum(results.values())} of {len(results)} falsifications failed for their stated reason: {results}")
raise SystemExit(0 if all(results.values()) else 1)
