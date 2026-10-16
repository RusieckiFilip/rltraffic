"""Tests for ``BRIEF_48`` Amendment B: B.5.1's committed ``collect verify`` record and B.3's stall remedy.

Written before the code they test (red first).

* **B.5.1** -- ``collect verify --out`` writes one record of every run it verified, so the verification is a committed
  artifact rather than a terminal's scrollback.
* **B.3** -- every process that rolls CityFlow episodes for the gate (G3) and for Phase B's evaluation (G3b) is cut by
  ``faulthandler.dump_traceback_later(timeout, exit=True)`` re-armed per episode, and its driver restarts it from the
  records on disk, at most 20 times, each restart logged; any other non-zero exit stops.  The four registered cases:
  a blocked fake episode is cut and the run completes on restart; a raising one is not restarted; a healthy run longer
  than the timeout in total is never cut; the cap stops a run that hangs every time.

The fake episodes run in real child processes (``sys.executable``), with a timeout of one or two seconds in place of
the registered 600, so the faulthandler cut is the real one.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import offline.hz4x4_ladder as hz

REPO = Path(__file__).resolve().parents[1]


# ======================================================================
# B.5.1: the collect verify record
# ======================================================================


def _verified(name: str, episodes: int, git_hash: str = "5918dadc", checkpoint: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "episodes": episodes, "flow_digests_checked": episodes, "git_hash": git_hash}
    if checkpoint is not None:
        record["checkpoint_sha256"] = checkpoint
    return record


def test_the_verify_record_carries_every_run_and_the_totals() -> None:
    records = [_verified("cf_hz4x4__maxpressure", 200)] + [
        _verified(f"cf_hz4x4__mappo1000__seed{s}", 40, checkpoint=f"{s:064d}") for s in hz.SEEDS
    ]
    artifact = hz.corpus_verify_artifact(records, tiers=["maxpressure", "mappo1000"])
    assert artifact["format_version"] == hz.ARTIFACT_FORMAT_CORPUS_VERIFY
    assert artifact["runs"] == records
    assert (artifact["n_runs"], artifact["n_episodes"], artifact["n_flow_digests_checked"]) == (6, 400, 400)
    assert artifact["tiers"] == ["maxpressure", "mappo1000"]
    assert artifact["collected_by_git_hashes"] == ["5918dadc"]
    assert artifact["all_verified"] is True


def test_the_verify_record_refuses_nothing_verified_and_names_every_code_tree() -> None:
    with pytest.raises(ValueError, match="no run"):
        hz.corpus_verify_artifact([], tiers=["maxpressure"])
    mixed = [_verified("cf_hz4x4__maxpressure", 200), _verified("cf_hz4x4__mappo1000__seed101", 40, git_hash="0abc")]
    assert hz.corpus_verify_artifact(mixed, tiers=["maxpressure", "mappo1000"])["collected_by_git_hashes"] == [
        "0abc", "5918dadc"]


def test_collect_verify_takes_an_out_path() -> None:
    namespace = hz.build_parser().parse_args(["collect", "verify", "--sweep", "s.json", "--out", "v.json"])
    assert namespace.out == "v.json"
    assert hz.build_parser().parse_args(["collect", "plan", "--sweep", "s.json"]).out is None
