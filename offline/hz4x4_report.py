"""P9.3 Phase B's report (A27 point 8): hz4x4's ladder, beside the grid's -- ``old-cells`` and ``report``.

Format versions: ``p9.3-old-cells/1.0`` (``docs/data/p9_3_old_cells.json``) and ``p9.3-hz4x4/1.0``
(``docs/data/p9_3_hz4x4.json``).  Alignment convention: none of its own -- every value is read from a record another
module wrote (``offline.hz4x4_phase_b``'s evaluation and training records; ``offline.tier_statements.load_inputs`` for the
grid's committed cells), and every row carries ``att_ours`` and ``att_engine`` of ONE held-out episode (seed, draw).

WHAT THIS IS -- AND WHAT IT IS NOT
----------------------------------
``PREREGISTRATION`` A27 (i), label 8: a conditional generalizability extension registered after the grid results.  hz4x4
has NO registered statement (A27 (i) registers none) and NO primary definition (A11's gate failed there by construction,
criterion 4).  Its leaders are the grid's DESCRIPTIVE rule (``BRIEF_48`` Amendment G, G.2, A-B2), under both definitions,
``registered: false``, ``definition_dependent`` flagged; the references' levels stand beside the leaders and never compete
for the lead.  The grid's corresponding cells come by reference and per-episode copy from their committed sources, with
path and digest, and nothing is re-rolled on the grid (Amendment A, A.5 Q4).

WHY A NEW MODULE (``BRIEF_48`` Amendment H, H.2-4)
--------------------------------------------------
``offline/hz4x4_phase_b.py`` and ``offline/hz4x4_ladder.py`` are frozen at the campaign's commit, so that the code which
wrote the records is provably the code the branch carries; the report reads their records and imports their constants,
and changes neither.

WHY A MIRROR OF THE LEADERS RULE
--------------------------------
``tier_statements.leaders_block`` loops over the grid's tiers, methods, versions and primary definition -- module
constants.  :func:`hz4x4_leaders_block` is its body with those four made parameters; every function it calls
(``level_groups``, ``_level``, ``contrast``, ``_seed_levels``) is ``tier_statements``' own, unchanged.  A test holds it to
the original on the grid's committed books and on a synthetic grid book.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import offline.hz4x4_ladder as hz
import offline.hz4x4_phase_b as pb
import offline.tier_statements as st

ARTIFACT_FORMAT_REPORT = "p9.3-hz4x4/1.0"
ARTIFACT_FORMAT_OLD_CELLS = "p9.3-old-cells/1.0"

#: The source tag of hz4x4's cells (``CellSpec.source``; cell ids are ``p9_3:<method>@<tier>``).
SOURCE = "p9_3"

GATE_ARTIFACT = Path("docs/data/p9_3_engine_att_gate.json")
DECLARATION_ARTIFACT = Path("docs/data/p9_3_declaration.json")

#: The five references (``hz4x4_phase_b.REFERENCE_CELLS``) as arm labels.
REFERENCE_ARMS: tuple[str, ...] = tuple(f"{method}@{tier}" for method, tier in pb.REFERENCE_CELLS)

#: hz4x4 cell -> the grid's committed counterpart (Q4).  ``random@mappo1000`` is P5.2c's shared anchor
#: (``tier_statements.ANCHOR_CELL``, which its books file under ``tier_statements.ANCHOR``).
GRID_COUNTERPARTS: Mapping[tuple[str, str], str] = {
    ("dt_nomix", "mappo1000"): "p5_1:dt_nomix@grid4x4_mappo1000",
    ("dt_nomix", "maxpressure"): "p5_2:dt_nomix@maxpressure",
    ("dt_nomix", "fixedtime"): "p5_2:dt_nomix@fixedtime",
    ("behaviour", "mappo1000"): "p5_1:behaviour@grid4x4_mappo1000",
    ("behaviour", "maxpressure"): "p5_2:behaviour@maxpressure",
    ("behaviour", "fixedtime"): "p5_2:behaviour@fixedtime",
    ("random", "mappo1000"): st.ANCHOR_CELL,
}

#: Every hz4x4 cell whose grid counterpart is P9.1's (A27 (b), (d)) -- the matched arms on every tier, ``dt_nomix`` and
#: ``behaviour`` on ``random_fresh`` -- and so is not committed until P9.1's campaign is.
P9_1_COUNTERPARTS: tuple[tuple[str, str], ...] = tuple(
    (cell.method, cell.tier) for cell in pb.evaluation_cells() if (cell.method, cell.tier) not in GRID_COUNTERPARTS
)

#: What the artifact is not, in the order the paper's limitations section reads them.
WHAT_THIS_IS_NOT: tuple[str, ...] = (
    "not a new topology: the grid's 4x4 lattice with real demand and real road geometry (A27 (i))",
    "not a new claim: a replication of the grid's ladder, reported beside it (A27 (i))",
    "not a C3 target: hz4x4 is audited and is not one (A15)",
    "not registered: A27 (i) registers no hz4x4 statement, so its leaders are descriptive (BRIEF_48 Amendment G, G.2)",
    "not single-definition: A11's gate failed on hz4x4 (criterion 4), so no claim rests on att_ours or att_engine alone",
    "not a timing claim: every training second is an uncontrolled clock (A27 (a))",
)


def load_cells(
    output_root: str | Path, *, seeds: Sequence[int] = hz.SEEDS, draws: Sequence[int] | None = None
) -> dict[str, Any]:
    """Every evaluation cell as a ``tier_statements.Cell``, its rows read from its records in (seed, draw) order --
    refused unless every planned record exists and says it reproduced, and no refusal is on disk."""
    raise NotImplementedError


def book_of(cells: Mapping[str, Any]) -> dict[tuple[str, str], Any]:
    """``(method, tier)`` -> cell, for the leaders rule."""
    raise NotImplementedError


def hz4x4_leaders_block(
    books: Mapping[str, Mapping[tuple[str, str], Any]],
    *,
    tiers: Sequence[str],
    methods: Sequence[str],
    primary_definition: str | None,
    versions_of: Callable[[str], Sequence[str]],
) -> list[dict[str, Any]]:
    """``tier_statements.leaders_block``'s body with its four grid constants made parameters."""
    raise NotImplementedError


def leaders(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    """hz4x4's leaders: the mirror over the five arms on each tier, no primary, the references' levels beside."""
    raise NotImplementedError


def cell_summaries(cells: Mapping[str, Any]) -> dict[str, Any]:
    """``tier_statements.cells_block``'s per-cell computation, with hz4x4's sources."""
    raise NotImplementedError


def training_block(output_root: str | Path) -> list[dict[str, Any]]:
    """The 100 training records, each checkpoint's sha256 verified against its record."""
    raise NotImplementedError


def old_cells(grid_cells: Mapping[str, Any], *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The grid's corresponding cells, copied per episode with source path and digest; P9.1's named as pending."""
    raise NotImplementedError


def report_artifact(
    *,
    cells: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    old: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """``docs/data/p9_3_hz4x4.json``'s payload."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    """``old-cells`` and ``report``.  Run from the main tree with this branch's code on ``PYTHONPATH``."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - the CLI is exercised through main() by the tests
    raise SystemExit(main())
