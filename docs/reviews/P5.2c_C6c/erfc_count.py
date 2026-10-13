"""Coordinator's independent check of the implementer's G2 blocker (BRIEF_45 Amendment B.1, B.1.3 step 3).

1. Walk docs/data/**/*.json of a 3e5de9c export WITHOUT the artifact, by this script's own walker (dicts carrying a
   numeric "z" and a "p_value"), and count pairs per file.
2. Build the artifact in memory from the export (git and code blocks are placeholders; the pairs do not depend on them),
   write it ONLY into the export's docs/data, and walk again.
3. For the pairs the artifact adds: count them by path class, check each p against min(1, 2*Phi(z)) two ways (the
   guard's own dt_gate._normal_cdf under ==, and math.erfc within 4 ulps, an independent route), and print every pair
   with z == 0 together with the dict it sits in (n_used etc.) and the statement's reading.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

EXPORT = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g2c/erfc")
sys.path.insert(0, str(EXPORT))

import offline.tier_statements as tst  # noqa: E402
from offline.dt_gate import _normal_cdf  # noqa: E402

assert Path(tst.__file__).resolve().is_relative_to(EXPORT.resolve()), tst.__file__
DATA = EXPORT / "docs" / "data"
ART = DATA / "p5_2_statements.json"


def walk(node, path, out):
    if isinstance(node, dict):
        if "z" in node and "p_value" in node and isinstance(node.get("z"), (int, float)) and not isinstance(
                node.get("z"), bool):
            out.append((path, node))
        for key, value in node.items():
            walk(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk(value, f"{path}[{index}]", out)


def sweep():
    out = []
    for artifact in sorted(DATA.rglob("*.json")):
        walk(json.loads(artifact.read_text(encoding="utf-8")), artifact.name, out)
    return out


assert not ART.exists(), "the export already holds an artifact"
before = sweep()
print("BEFORE: pairs", len(before))
print("  by file", dict(sorted(collections.Counter(p.split(".json")[0] for p, _ in before).items())))

roots = tst.Roots(repo_root=EXPORT, output_root=Path("/home/filip/rltraffic/output"))
inputs = tst.load_inputs(roots)
artifact = tst.build_artifact(inputs, git={"commit": "x", "dirty": False, "code_root": "x"}, code=[])
data = tst.serialize(artifact)
ART.write_bytes(data)
print("artifact bytes", len(data))

after = sweep()
added = [(p, n) for p, n in after if p.startswith("p5_2_statements.json")]
print("AFTER: pairs", len(after), "added", len(added), "other files unchanged:",
      [p for p, _ in after if not p.startswith("p5_2_statements.json")] == [p for p, _ in before])
print("  added by path class:")
for cls, n in sorted(collections.Counter(re.sub(r"\[\d+\]", "", p) for p, _ in added).items(), key=lambda kv: -kv[1]):
    print(f"   {n:4d}  {cls}")
alt = [p for p, n in added if "alternative" in n]
print("  added carrying 'alternative':", alt)
eq = sum(1 for _, n in added if min(1.0, 2.0 * _normal_cdf(float(n["z"]))) == float(n["p_value"]))
print("  reproduce under == via dt_gate._normal_cdf:", eq, "of", len(added))
worst = 0.0
for _, n in added:
    z, p = float(n["z"]), float(n["p_value"])
    ref = min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))
    worst = max(worst, abs(ref - p) / math.ulp(max(ref, p)))
print("  max |p - erfc(|z|/sqrt2)| in ulps (independent route):", worst)
signs = collections.Counter("neg" if float(n["z"]) < 0 else ("zero" if float(n["z"]) == 0 else "pos") for _, n in added)
print("  z signs:", dict(signs))

statements = artifact["statements"]
print("ZERO-z pairs:")
for p, n in added:
    if float(n["z"]) == 0.0:
        m = re.match(r"p5_2_statements\.json\.statements\[(\d+)\]", p)
        st = statements[int(m.group(1))] if m else None
        keys = {k: n[k] for k in sorted(n) if k not in ("z", "p_value")}
        print(" ", p, "z", n["z"], "p", n["p_value"], keys)
        if st is not None:
            print("     statement", st.get("id"), st.get("tier"), st.get("definition"), "|", st.get("verdict"), "|",
                  str(st.get("reading"))[:220])
