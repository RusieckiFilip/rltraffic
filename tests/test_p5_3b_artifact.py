"""The SHIPPED artifact: ``docs/data/p5_3b_nortg.json``, read as bytes and pinned.

``docs/reviews/P5.3a.md`` **MJ-4**: *"No test reads the committed artifact ... So nothing pins the
40 cells, R6's null control, A8's registered ``fixedtime`` prediction, the crosscheck's figures, row
B's ordering, or verdict-freedom."*  Every comparable artifact in this repo is read by tests; this
file is P5.3b's.

⚠️ **These tests need no corpus, no checkpoint and no simulator** -- the artifact is committed, so
they run on a CI runner where the campaign itself cannot.  That is deliberate: the claims a referee
reads are pinned by tests that always execute.

The load-bearing one is :func:`test_the_reported_paired_difference_recomputes_from_the_episodes`,
which rebuilds the headline from the artifact's own ``dt_nortg`` records and the **committed
``p4_6``/``p4_7`` grids** -- not from the artifact's copy of the ``dt`` column -- and asserts exact
equality.  ``==``, never ``allclose`` (CLAUDE.md section 2).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
from offline.method_tier_grid import assert_no_verdicts
from offline.nortg_campaign import (
    ARTIFACT_FORMAT_VERSION,
    NORTG_METHOD,
    NORTG_TIERS,
    TIER_GRID_ARTIFACT,
    nortg_arm_key,
)
from offline.rtg_ablation import INTERVENTION_KEYS

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "data"
ARTIFACT = DATA / "p5_3b_nortg.json"

#: The committed ``dt`` cell means these comparisons are anchored to, copied digit for digit from
#: ``p4_6_grid.json`` / ``p4_7_grid.json`` and asserted against them below.
COMMITTED_DT_ATT = {
    "mappo1000": 104.95575898180847,
    "mix50": 107.70262931184996,
    "random": 420.37638648227966,
}


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _committed_dt_episodes(tier: str) -> list[dict[str, Any]]:
    grid = json.loads((DATA / TIER_GRID_ARTIFACT[tier]).read_text(encoding="utf-8"))
    return [entry for entry in grid["episodes"] if entry["arm"] == f"dt@{tier}"]


def _per_draw_means(records: list[dict[str, Any]]) -> dict[int, float]:
    """An independent reimplementation of ``dt_gate._per_draw_means``, written here.

    The campaign sorts both arms by ``(seed, draw_id)`` before pairing, so this bucketing sees the
    same order and exact equality is the right bar rather than a tolerance.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for record in sorted(records, key=lambda r: (int(r["seed"]), int(r["draw_id"]))):
        buckets[int(record["draw_id"])].append(float(record["att_horizon"]))
    return {draw: float(np.mean(values)) for draw, values in buckets.items()}


# ----------------------------------------------------------------------
# Shape
# ----------------------------------------------------------------------


def test_the_artifact_declares_its_format_and_the_registered_tier_set(artifact: dict[str, Any]) -> None:
    assert artifact["format_version"] == ARTIFACT_FORMAT_VERSION
    assert tuple(artifact["tiers"]) == NORTG_TIERS == ("mappo1000", "mix50", "random")
    assert artifact["method"] == NORTG_METHOD
    assert artifact["declared_gradient_steps"] == 40_000
    assert tuple(artifact["seeds"]) == TRAINING_SEEDS


def test_there_are_exactly_fifteen_cells_and_they_are_the_declared_ones(artifact: dict[str, Any]) -> None:
    cells = artifact["cells"]
    assert len(cells) == 15
    got = {(cell["tier"], int(cell["seed"])) for cell in cells}
    assert got == {(tier, seed) for tier in NORTG_TIERS for seed in TRAINING_SEEDS}


def test_every_cell_covers_the_whole_held_out_pool(artifact: dict[str, Any]) -> None:
    for cell in artifact["cells"]:
        assert cell["n_episodes"] == len(HELD_OUT_DRAWS) == 100, cell["arm"]
        assert [int(d) for d in cell["draw_ids"]] == list(HELD_OUT_DRAWS), cell["arm"]
        assert cell["arm"] == nortg_arm_key(cell["tier"])


def test_the_episode_records_are_complete_at_three_tiers_by_five_seeds_by_one_hundred_draws(
    artifact: dict[str, Any],
) -> None:
    episodes = artifact["episodes"]
    assert len(episodes) == len(NORTG_TIERS) * len(TRAINING_SEEDS) * len(HELD_OUT_DRAWS) == 1500
    keys = {(e["arm"], int(e["seed"]), int(e["draw_id"])) for e in episodes}
    assert len(keys) == len(episodes), "duplicate (arm, seed, draw) records"
    assert keys == {
        (nortg_arm_key(tier), seed, draw)
        for tier in NORTG_TIERS
        for seed in TRAINING_SEEDS
        for draw in HELD_OUT_DRAWS
    }


# ----------------------------------------------------------------------
# Gate 3 as a committed fact
# ----------------------------------------------------------------------


def test_gate_three_every_nortg_checkpoint_ignores_the_return_token_exactly(
    artifact: dict[str, Any],
) -> None:
    """Q3, the acceptance criterion, pinned in the shipped bytes: 180 exact zeros."""
    cells = artifact["arm_validity"]["cells"]
    assert len(cells) == 15
    checked = 0
    for cell in cells:
        assert cell["rtg_mode"] == "zero", cell["checkpoint"]
        assert sorted(cell["interventions"]) == sorted(INTERVENTION_KEYS)
        for key, values in cell["interventions"].items():
            assert float(values["flip_rate"]) == 0.0, f"{cell['tier']}@{cell['seed']} {key}"
            checked += 1
    assert checked == 15 * 12 == 180


def test_the_arm_validity_summary_agrees_with_the_cells_it_summarises(artifact: dict[str, Any]) -> None:
    summary = artifact["arm_validity"]
    assert summary["n_cells"] == 15
    assert summary["n_values_checked"] == 180
    assert summary["max_flip_rate"] == 0.0


# ----------------------------------------------------------------------
# The headline, recomputed by an independent route
# ----------------------------------------------------------------------


def test_the_reported_paired_difference_recomputes_from_the_episodes(artifact: dict[str, Any]) -> None:
    """⭐ CLAUDE.md section 2: the critical quantity computed twice, by a different route.

    The ``dt`` side is read from the **committed grid artifacts**, not from this artifact's copy of
    them, so a transcription error in the copy cannot hide here.
    """
    nortg_by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in artifact["episodes"]:
        nortg_by_tier[str(entry["arm"]).split("@", 1)[1]].append(entry)

    for tier in NORTG_TIERS:
        left = _per_draw_means(_committed_dt_episodes(tier))
        right = _per_draw_means(nortg_by_tier[tier])
        shared = sorted(set(left) & set(right))
        assert len(shared) == 100, tier
        differences = [left[draw] - right[draw] for draw in shared]
        recomputed = float(np.asarray(differences, dtype=np.float64).mean())

        reported = artifact["comparisons"][tier]["paired"]
        assert reported["n_shared_draws"] == 100, tier
        assert reported["mean_difference"] == recomputed, tier
        assert artifact["comparisons"][tier]["abs_mean_difference"] == abs(recomputed), tier


def test_the_dt_reference_cells_are_the_committed_ones(artifact: dict[str, Any]) -> None:
    """The reused column is named, sourced, and equal to the merged artifact digit for digit."""
    for tier in NORTG_TIERS:
        reference = artifact["reference_dt_cells"][tier]
        assert reference["source"] == TIER_GRID_ARTIFACT[tier]
        assert reference["att_horizon_mean"] == COMMITTED_DT_ATT[tier], tier

        grid = json.loads((DATA / TIER_GRID_ARTIFACT[tier]).read_text(encoding="utf-8"))
        assert grid["cells"][f"dt@{tier}"]["att_horizon_mean"] == COMMITTED_DT_ATT[tier], tier
        assert artifact["comparisons"][tier]["att_dt_mean"] == COMMITTED_DT_ATT[tier], tier


def test_the_scored_predictions_agree_with_the_comparisons_they_are_scored_from(
    artifact: dict[str, Any],
) -> None:
    magnitudes = {
        tier: artifact["comparisons"][tier]["abs_mean_difference"] for tier in NORTG_TIERS
    }
    q1 = artifact["predictions"]["Q1"]
    assert q1["largest"] == max(magnitudes, key=lambda tier: magnitudes[tier])
    assert q1["smallest"] == min(magnitudes, key=lambda tier: magnitudes[tier])
    assert q1["holds"] == (q1["largest"] == "mix50" and q1["smallest"] == "random")
    assert q1["scale"] == "raw ATT"

    paired = artifact["comparisons"]["random"]["paired"]
    q2 = artifact["predictions"]["Q2"]
    assert q2["ci_contains_zero"] == (paired["ci95_low"] <= 0.0 <= paired["ci95_high"])
    assert q2["holds"] == q2["ci_contains_zero"]
    assert "failure to reject" in q2["reading"].lower()


def test_every_tier_reports_its_per_seed_reversals(artifact: dict[str, Any]) -> None:
    for tier in NORTG_TIERS:
        record = artifact["comparisons"][tier]["per_seed"]
        assert record["n_seeds"] == 5
        assert sorted(int(s) for s in record["per_seed"]) == sorted(TRAINING_SEEDS)
        assert 0 <= record["seeds_reversed"] <= 5


# ----------------------------------------------------------------------
# What the artifact must NOT say
# ----------------------------------------------------------------------


def test_the_shipped_bytes_carry_no_verdict_and_no_equivalence_threshold() -> None:
    """``PREREGISTRATION`` A7 withdrew the per-tier delta rule; ``BRIEF_30`` section 5 forbids
    reinventing it.  A scan of the bytes, not of an in-memory payload built by the same code."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert_no_verdicts(payload)
    text = ARTIFACT.read_text(encoding="utf-8").lower()
    for token in (
        "equivalent", "within_delta", "equivalence margin", "delta_att",
        "genuinely better", "inconclusive at this power", "the token is inert",
    ):
        assert token not in text, token


def test_the_artifact_states_the_three_things_a_reader_must_not_conclude(artifact: dict[str, Any]) -> None:
    """``BRIEF_30`` section 9, carried in the artifact rather than only in the packet."""
    limitations = " ".join(artifact["limitations"]).lower()
    assert "failure to reject" in limitations
    assert "row b" in limitations
    assert "216000" in " ".join(artifact["limitations"])
    assert "420" in " ".join(artifact["limitations"])


def test_the_artifact_records_which_commits_produced_its_inputs(artifact: dict[str, Any]) -> None:
    """``DEFERRED`` 39: a single write-time commit describes when a report was assembled and
    nothing about what produced its numbers."""
    runtime = artifact["runtime"]
    assert runtime["measurement_git_commits"], "no measurement commits recorded"
    assert runtime["unreachable_measurement_commits"] == []
    assert runtime["written_at_git_commit"]
