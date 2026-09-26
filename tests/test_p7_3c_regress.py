"""P7.3c's regressions against REAL on-disk campaign data (``BRIEF_41`` C5; Amendment A, Q2; ``DEFERRED`` 95).

* **``DEFERRED`` 95 -- T-regress (b) for grid4x4.**  ``docs/data/p7_3d_grid4x4.json`` regenerates BYTE-IDENTICALLY through
  ``report`` over ``output/p7_3d/cells`` into ``tmp_path``, with Amendment A3's two substitutions and nothing else
  (``tests/test_t_regress_artifacts.py``'s ``_regenerates``, imported rather than copied).  It lands in C5 because C5
  extracts ``report``'s row code into the helper the stage-1 gate shares -- the first commit that touches the grid4x4
  report path (Amendment A, Q2's addition) -- and it is written BEFORE that extraction, so the pin cannot merely agree
  with it.
* **The stage-1 gate on the REAL chunks.**  P7.3d's own 700 chunks, re-labelled into P7.3c's stage 1 and with their
  bookkeeping fields moved, reproduce the committed artifact 700/700: the gate's row, built from a chunk as ``report``
  builds it, IS the committed record.
* **The corpus pin.**  G3's verified corpus ``SHA256SUMS`` hashes to ``P7_3C_CORPUS_SUMS_SHA256`` (Amendment C, C2).

GATES, each naming what it consumes: ``RLTRAFFIC_OUTPUT_ROOT`` holding ``p7_3d/cells/canary.json`` (gitignored), with
the main tree's draws and A20(a)'s checkpoints beside it (``report`` re-derives both digests); ``RLTRAFFIC_SUMO_CORPORA``
holding ``grid4x4_sumo_maxpressure/SHA256SUMS``.  Nothing here executes a campaign driver or starts a process.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import offline.transfer_curve as tcv
from tests.test_t_regress_artifacts import _regenerates, _work_dir_or_skip

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"


def test_p7_3d_grid4x4_regenerates_byte_identically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``DEFERRED`` 95.  *Mutation:* one aggregate of ``_grid4x4_estimates`` perturbed -> the bytes move -> this dies."""
    _regenerates(
        monkeypatch, tmp_path, work_name="p7_3d/cells", artifact_name="p7_3d_grid4x4.json",
        stage=tcv.STAGE_GRID4X4,
    )


def test_the_stage1_gate_reproduces_p7_3ds_own_700_chunks_relabelled(tmp_path: Path) -> None:
    source = _work_dir_or_skip("p7_3d/cells", marker="canary.json")
    names = sorted(path.name for path in source.glob("cell_*.json"))
    assert len(names) == 700
    work = tmp_path / "cells"
    work.mkdir()
    for name in names:
        chunk = json.loads((source / name).read_bytes())
        chunk.update(
            stage=tcv.STAGE_P7_3C_REPRODUCE, git_commit="f" * 40, seconds=float(chunk["seconds"]) + 1.0,
            canary_seconds=0.5,
        )
        (work / name).write_text(json.dumps(chunk), encoding="utf-8")
    result = tcv.stage1_reproduction_check(work, data_dir=DATA)
    assert result["line"] == "stage1_check REPRODUCED 700/700"
    assert (result["verdict"], result["n_checked"], result["cells"]) == ("REPRODUCED", 700, {})


def test_the_verified_grid4x4_corpus_hashes_to_the_pin() -> None:
    root = os.environ.get("RLTRAFFIC_SUMO_CORPORA")
    if not root:
        pytest.skip("RLTRAFFIC_SUMO_CORPORA is unset: G3's grid4x4 SUMO corpus is gitignored, main tree only")
    sums = Path(root) / "grid4x4_sumo_maxpressure" / "SHA256SUMS"
    if not sums.is_file():
        pytest.skip(f"{sums} is absent: G3's grid4x4 SUMO corpus is gitignored, main tree only")
    assert hashlib.sha256(sums.read_bytes()).hexdigest() == tcv.P7_3C_CORPUS_SUMS_SHA256
