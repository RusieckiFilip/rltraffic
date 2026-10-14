"""Build the P5.2c artifact IN MEMORY from one tree and write its leaves (path -> value) as JSON.  Writes only the
file named on the command line; nothing under the tree and nothing under output/.

    python dump.py <tree> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TREE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(TREE))

import offline.tier_statements as tst  # noqa: E402

assert Path(tst.__file__).resolve().is_relative_to(TREE), tst.__file__
GIT = {"commit": "0" * 40, "dirty": False, "code_root": "synthetic"}
CODE = [{"module": "offline.tier_statements", "path": "offline/tier_statements.py", "sha256": "0" * 64}]
roots = tst.Roots(repo_root=TREE, output_root=Path("/home/filip/rltraffic/output"))
artifact = json.loads(tst.serialize(tst.build_artifact(tst.load_inputs(roots), git=GIT, code=CODE)))


def leaves(node: Any, path: str, out: dict[str, Any]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            leaves(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            leaves(value, f"{path}[{index}]", out)
    else:
        out[path] = node


flat: dict[str, Any] = {}
leaves(artifact, "", flat)
OUT.write_text(json.dumps(flat, indent=0, sort_keys=True) + "\n", encoding="utf-8")
print(f"{len(flat)} leaves from {TREE}")
