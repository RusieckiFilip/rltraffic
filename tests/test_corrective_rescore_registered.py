"""P9.1's re-scoring on the REGISTERED grid book (``offline/corrective_verdicts.py``; ``BRIEF_46`` 4.5: "``rescore``
reproduces P5.2c's verdict for one unchanged statement -- a tautology guard").

Format: P5.2c's ``p5.2-statements/1.0`` (``docs/data/p5_2_statements.json``).  Alignment: a statement is keyed by
``(id, tier, arm, definition)``; the registered book is P5.2c's ``corrected`` one (P5.2b's random-tier IQL cell).
With NOTHING substituted, P9.1's route -- P5.2c's pinned loader, then ``rescore_p5_2`` -- must give every committed
statement's verdict and reading, read here from the committed artifact.  Needs the main tree's pinned ``output/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import offline.corrective_verdicts as cv
import offline.tier_statements as tst

REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", "/nonexistent"))


@pytest.mark.skipif(not (OUTPUT / "p5_2").is_dir(), reason="needs the main tree's pinned output/ (RLTRAFFIC_OUTPUT_ROOT)")
def test_the_registered_book_rescored_with_nothing_substituted_is_every_committed_p5_2c_decision() -> None:
    inputs = tst.load_inputs(tst.Roots(repo_root=REPO, output_root=OUTPUT))
    books = tst.make_books(inputs.cells)
    rescored = cv.rescore_p5_2(books, {})
    committed = json.loads((REPO / "docs" / "data" / "p5_2_statements.json").read_text())["statements"]
    registered = {(e["id"], e["tier"], e["arm"], e["definition"]): e for e in committed
                  if e["version"] in ("both", "corrected")}
    assert {(e["id"], e["tier"], e["arm"], e["definition"]) for e in rescored} == set(registered)
    for entry in rescored:
        mine = registered[(entry["id"], entry["tier"], entry["arm"], entry["definition"])]
        assert (entry["verdict"], entry["reading"]) == (mine["verdict"], mine["reading"]), entry["id"]
    block = cv.moved_block(tst.all_statements(books), rescored)
    assert not any(e["moved"] or e["reading_changed"] for e in block)
    assert sum(e["kind"] == "verdict" for e in block) > 0
