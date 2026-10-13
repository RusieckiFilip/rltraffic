"""Build the P5.2c statements IN MEMORY from the 5bbb540 export and print Q0b's per-seed block and qualifier, and
Q3b's sub-prediction at fixedtime, under both definitions.  Writes nothing."""
from __future__ import annotations

import sys
from pathlib import Path

BASE = "/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g11/base"
sys.path.insert(0, BASE)

import offline.tier_statements as tst  # noqa: E402

assert Path(tst.__file__).resolve().is_relative_to(Path(BASE).resolve()), tst.__file__
roots = tst.Roots(repo_root=Path(BASE), output_root=Path("/home/filip/rltraffic/output"))
inputs = tst.load_inputs(roots)
books = tst.make_books(inputs.cells)
for definition in tst.DEFINITIONS:
    entries = {e["id"]: e for e in tst.score_q0(books["corrected"], definition=definition)}
    q0b = entries["Q0b"]
    block = q0b["numbers"]["per_seed"]
    print(definition, "Q0b", q0b["verdict"], "|", q0b["reading"])
    print("   by_seed", {k: round(v, 4) for k, v in block["by_seed"].items()},
          "favouring_left", block["n_seeds_favouring_left"], "against", block["reverses_against_pooled_sign"])
    print("   qualifier:", q0b["numbers"]["seed_reversal_qualifier"])
    q3b = {(e["id"], e["tier"]): e for e in tst.score_q3b(books["corrected"], definition=definition)}
    print("   Q3b fixedtime sub:", q3b[("Q3b", "fixedtime")]["numbers"]["sub_prediction"])
