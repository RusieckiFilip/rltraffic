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

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
from offline.method_tier_grid import assert_no_verdicts
from offline.nortg_campaign import (
    ADMISSION_FIELDS,
    ARTIFACT_FORMAT_VERSION,
    ATT_DEFINITIONS,
    MECHANISM_FIELDS,
    NORTG_METHOD,
    NORTG_TIERS,
    PRIMARY_ATT,
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


def _per_draw_means(records: list[dict[str, Any]], key: str = "att_horizon") -> dict[int, float]:
    """An independent reimplementation of ``dt_gate._per_draw_means``, written here.

    The campaign sorts both arms by ``(seed, draw_id)`` before pairing, so this bucketing sees the
    same order and exact equality is the right bar rather than a tolerance.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for record in sorted(records, key=lambda r: (int(r["seed"]), int(r["draw_id"]))):
        buckets[int(record["draw_id"])].append(float(record[key]))
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

    Under ``att_ours`` the ``dt`` side is read from the **committed grid artifacts**, not from this
    artifact's copy, so a transcription error in the copy cannot hide here.  ``att_ours`` is the
    definition the committed grids carry; the primary (``att_engine``) half is recomputed against
    the artifact's own reference rows in
    :func:`test_the_primary_definition_recomputes_from_the_reference_column`.
    """
    nortg_by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in artifact["episodes"]:
        nortg_by_tier[str(entry["arm"]).split("@", 1)[1]].append(entry)

    for tier in NORTG_TIERS:
        left = _per_draw_means(_committed_dt_episodes(tier))
        right = _per_draw_means(nortg_by_tier[tier], "att_ours")
        shared = sorted(set(left) & set(right))
        assert len(shared) == 100, tier
        differences = [left[draw] - right[draw] for draw in shared]
        recomputed = float(np.asarray(differences, dtype=np.float64).mean())

        reported = artifact["comparisons"][tier]["by_definition"]["att_ours"]["paired"]
        assert reported["n_shared_draws"] == 100, tier
        assert reported["mean_difference"] == recomputed, tier


def test_every_episode_carries_the_five_a11b_quantities(artifact: dict[str, Any]) -> None:
    """AMENDMENT D1 / A11(b), pinned in the shipped bytes: unconditional, on every cell."""
    assert tuple(ADMISSION_FIELDS) == (
        "att_ours", "att_engine", "entered", "created", "never_entered"
    )
    for entry in artifact["episodes"]:
        for field in ADMISSION_FIELDS:
            assert field in entry, f"{entry.get('arm')} {entry.get('draw_id')} {field}"
        assert entry["entered"] + entry["never_entered"] == entry["created"]


def test_every_contrast_reports_both_definitions_with_att_engine_primary(
    artifact: dict[str, Any],
) -> None:
    """AMENDMENT D2 / Rule R: ``att_engine`` primary on hz1x1, ``att_ours`` beside it, every table."""
    assert artifact["primary_att_definition"] == PRIMARY_ATT == "att_engine"
    for tier in NORTG_TIERS:
        entry = artifact["comparisons"][tier]
        assert sorted(entry["by_definition"]) == sorted(ATT_DEFINITIONS), tier
        assert entry["primary_definition"] == PRIMARY_ATT, tier
        # the top level IS the primary, so no reader can mistake which one it is
        assert entry["paired"] == entry["by_definition"][PRIMARY_ATT]["paired"], tier


def test_every_tier_states_whether_its_two_arms_differ_at_all(artifact: dict[str, Any]) -> None:
    """AMENDMENT D3.1 / E4: *a contrast over identical inputs is not a null result.*"""
    for tier in NORTG_TIERS:
        record = artifact["discriminability"][tier]
        assert set(record) >= {"distinct", "n_identical", "n_compared", "definition"}, tier
        assert record["n_compared"] == 500, tier
        assert isinstance(record["distinct"], bool), tier


def test_a_non_distinct_null_control_is_never_reported_as_a_null(artifact: dict[str, Any]) -> None:
    """🚨 E4, the binding half: registered before any P5.3b number existed."""
    q2 = artifact["predictions"]["Q2"]
    distinct = artifact["discriminability"]["random"]["distinct"]
    assert q2["arms_distinct"] == distinct
    if not distinct:
        assert q2["holds"] is None, "a non-distinct contrast scores neither pass nor fail"
        assert q2["artefact_of_non_discrimination"] is True
        assert "cannot discriminate" in q2["reading"].lower()
    else:
        assert q2["artefact_of_non_discrimination"] is False


def test_the_dt_reference_cells_are_the_committed_ones(artifact: dict[str, Any]) -> None:
    """The reused column is named, sourced, and equal to the merged artifact digit for digit."""
    for tier in NORTG_TIERS:
        reference = artifact["reference_dt_cells"][tier]
        assert reference["source"] == TIER_GRID_ARTIFACT[tier]
        # ⚠️ Named explicitly: the committed grids' ``att_horizon`` IS ``att_ours``, and this
        # task's primary is ``att_engine``.  A bare ``att_horizon_mean`` here would carry the
        # engine mean under the committed grid's own field name -- the BEHAVIOUR_ATT hazard.
        assert "att_horizon_mean" not in reference, (
            "reference_dt_cells must not reuse the committed grid's field name for a different "
            "definition"
        )
        assert reference["att_ours_mean"] == COMMITTED_DT_ATT[tier], tier
        assert reference["att_engine_mean"] != reference["att_ours_mean"], tier

        grid = json.loads((DATA / TIER_GRID_ARTIFACT[tier]).read_text(encoding="utf-8"))
        assert grid["cells"][f"dt@{tier}"]["att_horizon_mean"] == COMMITTED_DT_ATT[tier], tier
        assert artifact["comparisons"][tier]["committed_att_ours_mean"] == COMMITTED_DT_ATT[tier], tier
        assert (
            artifact["comparisons"][tier]["by_definition"]["att_ours"]["att_dt_mean"]
            == COMMITTED_DT_ATT[tier]
        ), tier


def test_the_scored_predictions_agree_with_the_comparisons_they_are_scored_from(
    artifact: dict[str, Any],
) -> None:
    magnitudes = {
        tier: artifact["comparisons"][tier]["abs_mean_difference"] for tier in NORTG_TIERS
    }
    assert artifact["predictions"]["Q1"]["definition"] == PRIMARY_ATT
    q1 = artifact["predictions"]["Q1"]
    assert q1["largest"] == max(magnitudes, key=lambda tier: magnitudes[tier])
    assert q1["smallest"] == min(magnitudes, key=lambda tier: magnitudes[tier])
    # ⚠️ AUTHORISED EDIT, BRIEF_33 section 2.2 (2026-09-10). This line was
    #     assert q1["holds"] == (q1["largest"] == "mix50" and q1["smallest"] == "random")
    # which pinned the unqualified `true` MJ-3 removed: `random`'s two arms are IDENTICAL on 500 of
    # 500 cells, so its limb could not have been falsified by any outcome and the prediction was
    # never tested there. The rule is three-valued and the artifact ships it as `holds_rule`.
    assert artifact["discriminability"]["random"]["distinct"] is False
    assert q1["holds"] is None
    assert "holds_rule" in q1
    for which in ("largest", "smallest"):
        limb = q1[f"{which}_limb"]
        assert limb["registered_tier"] == {"largest": "mix50", "smallest": "random"}[which]
        assert limb["as_registered"] is (limb["tier"] == limb["registered_tier"])
    # and the rule, applied here by a second route, agrees with the shipped value
    limbs = (q1["largest_limb"], q1["smallest_limb"])
    if any(limb["is_evidence"] and not limb["as_registered"] for limb in limbs):
        expected: bool | None = False
    elif all(limb["is_evidence"] and limb["as_registered"] for limb in limbs):
        expected = True
    else:
        expected = None
    assert q1["holds"] is expected
    assert q1["scale"] == "raw ATT"

    paired = artifact["comparisons"]["random"]["paired"]
    q2 = artifact["predictions"]["Q2"]
    assert q2["ci_contains_zero"] == (paired["ci95_low"] <= 0.0 <= paired["ci95_high"])


def test_every_tier_reports_its_per_seed_reversals(artifact: dict[str, Any]) -> None:
    for tier in NORTG_TIERS:
        record = artifact["comparisons"][tier]["by_definition"][PRIMARY_ATT]["per_seed"]
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


# ----------------------------------------------------------------------
# BRIEF_33: the decomposition A13(b) requires, and the mechanism block
# ----------------------------------------------------------------------

DECOMPOSITION = DATA / "p5_3b_decomposition.json"
TERMS = ("population", "clock_origin", "cadence")


def test_every_contrast_carries_a13bs_three_component_decomposition(
    artifact: dict[str, Any],
) -> None:
    """🔒 ``PREREGISTRATION`` A13(b) makes it REQUIRED for every report of the difference.

    P5.3b reported that difference in three places and the decomposition nowhere; that is
    ``docs/reviews/P5.3b.md`` BL-2(b), and this test is what stops it recurring silently. The
    ``source.sha256`` is checked against the file on disk, so the block cannot drift away from the
    artifact it claims to summarise.
    """
    digest = hashlib.sha256(DECOMPOSITION.read_bytes()).hexdigest()
    for tier in NORTG_TIERS:
        block = artifact["comparisons"][tier]["definition_difference_decomposition"]
        assert block["n_episodes_per_arm"] == 500, tier
        assert block["residual_max"] == 0.0, tier
        assert block["source"]["sha256"] == digest, tier
        for label in ("dt", "dt_nortg"):
            record = block["per_arm"][label]
            assert abs(sum(record[t] for t in TERMS) - record["total"]) < 1e-9, (tier, label)
        contrast = block["contrast"]
        assert abs(sum(contrast[t] for t in TERMS) - contrast["total"]) < 1e-9, tier
        assert abs(block["identity_gap"]) < 1e-9, tier


def test_the_decomposition_explains_the_difference_the_artifact_itself_reports(
    artifact: dict[str, Any],
) -> None:
    """⭐ The critical quantity, recomputed here by a route that does not touch the block.

    ``delta_ours - delta_engine`` is read from ``by_definition`` -- the campaign's own paired
    statistics -- and compared against the three measured terms. If the decomposition described some
    other quantity, this is what would notice.
    """
    for tier in NORTG_TIERS:
        by_definition = artifact["comparisons"][tier]["by_definition"]
        gap = (
            by_definition["att_ours"]["paired"]["mean_difference"]
            - by_definition["att_engine"]["paired"]["mean_difference"]
        )
        contrast = artifact["comparisons"][tier]["definition_difference_decomposition"]["contrast"]
        assert abs(sum(contrast[t] for t in TERMS) - gap) < 1e-9, tier


def test_the_mechanism_block_is_labelled_exploratory_and_unregistered(
    artifact: dict[str, Any],
) -> None:
    """``PREREGISTRATION`` section 2 fixes confirmatory versus exploratory, and no-RTG is named in
    the exploratory list. This block was found AFTER the numbers existed, by a post-merge review."""
    mechanism = artifact["mechanism"]
    assert mechanism["registered"] is False
    assert mechanism["exploratory"] is True
    assert "post-merge review" in mechanism["found_by"]


def test_the_outcome_identity_counts_recompute_from_the_artifacts_own_episodes(
    artifact: dict[str, Any],
) -> None:
    """🔒 The independent route, written here rather than imported.

    Eight fields, per ``mix50`` seed, against ``dt_nortg@random`` seed 101 -- rebuilt from the
    artifact's own ``episodes`` with a plain dict comparison, then checked against the shipped
    counts. ``CLAUDE.md`` section 2: the critical quantity computed twice, by a different route.
    """
    fields = ("att_engine", "att_ours", "entered", "created", "never_entered",
              "horizon_vehicle_count", "episode_reward", "completed_at_horizon")
    keyed: dict[tuple[str, int], dict[int, tuple[Any, ...]]] = defaultdict(dict)
    for entry in artifact["episodes"]:
        keyed[(entry["tier"], int(entry["seed"]))][int(entry["draw_id"])] = tuple(
            entry[field] for field in fields
        )
    reference = keyed[("random", 101)]

    for seed in TRAINING_SEEDS:
        rows = keyed[("mix50", seed)]
        shared = sorted(set(rows) & set(reference))
        identical = sum(1 for draw in shared if rows[draw] == reference[draw])
        shipped = artifact["mechanism"]["outcome_identity"]["mix50_against_reference"][str(seed)]
        assert shipped["n_compared"] == len(shared), seed
        assert shipped["n_identical"] == identical, seed

    for seed in TRAINING_SEEDS:
        rows = keyed[("random", seed)]
        identical = sum(1 for draw in sorted(set(rows) & set(reference))
                        if rows[draw] == reference[draw])
        shipped = artifact["mechanism"]["outcome_identity"]["random_seed_invariance"][str(seed)]
        assert shipped["n_identical"] == identical, seed


def test_the_action_identity_counts_recompute_from_the_decomposition_artifact(
    artifact: dict[str, Any],
) -> None:
    """The same shape as the outcome counts, on ``action_sequence_sha256``, by an independent route.

    ⭐ This is what makes *"the same attractor"* a statement about the POLICY rather than about the
    outcome: two runs can land on the same eight numbers without taking the same decisions.
    """
    decomposition = json.loads(DECOMPOSITION.read_text(encoding="utf-8"))
    keyed: dict[tuple[str, str, int], dict[int, str]] = defaultdict(dict)
    for row in decomposition["episodes"]:
        keyed[(row["method"], row["tier"], int(row["seed"]))][int(row["draw_id"])] = (
            row["action_sequence_sha256"]
        )
    reference = keyed[("dt_nortg", "random", 101)]

    for seed in TRAINING_SEEDS:
        rows = keyed[("dt_nortg", "mix50", seed)]
        identical = sum(1 for draw in sorted(set(rows) & set(reference))
                        if rows[draw] == reference[draw])
        shipped = artifact["mechanism"]["action_identity"]["mix50_against_reference"][str(seed)]
        assert shipped["n_identical"] == identical, seed


def test_the_per_cell_sequence_count_says_whether_a_cell_is_open_loop(
    artifact: dict[str, Any],
) -> None:
    """🔒 ``BRIEF_33`` AMENDMENT D1, recomputed from the decomposition artifact's own rows.

    One distinct sequence over 100 draws means the cell took the same 360 decisions whatever it
    observed. This is the field that distinguishes *"the same outcome"* from *"the same fixed
    sequence, independent of input"*, and the distinction is the whole of the claim.
    """
    decomposition = json.loads(DECOMPOSITION.read_text(encoding="utf-8"))
    counts: dict[str, set[str]] = defaultdict(set)
    draws: dict[str, int] = defaultdict(int)
    for row in decomposition["episodes"]:
        key = f"{row['method']}@{row['tier']} seed {int(row['seed'])}"
        counts[key].add(row["action_sequence_sha256"])
        draws[key] += 1

    shipped = artifact["mechanism"]["action_identity"]["n_distinct_action_sequences"]
    assert set(shipped) == set(counts)
    for key, sequences in counts.items():
        assert draws[key] == 100, key
        assert shipped[key] == len(sequences), key


def test_the_attractor_is_identified_from_the_null_control_and_counted_on_mix50(
    artifact: dict[str, Any],
) -> None:
    """The attractor digest is whatever ``dt_nortg@random`` emits, recomputed here from its rows."""
    decomposition = json.loads(DECOMPOSITION.read_text(encoding="utf-8"))
    random_digests = {
        row["action_sequence_sha256"]
        for row in decomposition["episodes"]
        if row["method"] == "dt_nortg" and row["tier"] == "random"
    }
    record = artifact["mechanism"]["action_identity"]["attractor"]["mix50"]
    if len(random_digests) == 1:
        assert record["attractor_sequence_sha256"] == random_digests.pop()
        carrying = sum(
            1
            for row in decomposition["episodes"]
            if row["method"] == "dt_nortg"
            and row["tier"] == "mix50"
            and row["action_sequence_sha256"] == record["attractor_sequence_sha256"]
        )
        assert record["n_draws_carrying_it"] == carrying
    else:
        assert record["attractor_sequence_sha256"] is None


def test_the_embedded_per_arm_block_equals_the_decomposition_artifacts_own_summary(
    artifact: dict[str, Any],
) -> None:
    """🔒 MJ-1: the per-arm split is a number the paper quotes, and nothing read it.

    A swap of ``per_arm.dt`` and ``per_arm.dt_nortg`` in the shipped bytes survived 178 tests, as did
    shifting one arm's ``clock_origin`` and ``total`` together. The summing check next door cannot
    see either, because a swapped or shifted block still sums to its own total.

    The tie is to the decomposition artifact **read from disk**, arm by arm, under ``==``.
    """
    decomposition = json.loads(DECOMPOSITION.read_text(encoding="utf-8"))
    for tier in NORTG_TIERS:
        per_arm = artifact["comparisons"][tier]["definition_difference_decomposition"]["per_arm"]
        for method in ("dt", "dt_nortg"):
            source = decomposition["summary"]["per_arm"][f"{method}@{tier}"]
            for term in (*TERMS, "total"):
                assert per_arm[method][term] == source[term]["mean"], (tier, method, term)


def test_the_outcome_identity_is_defined_on_exactly_the_eight_registered_fields(
    artifact: dict[str, Any],
) -> None:
    """🔒 MJ-3: the eight-field definition was unprotected in code AND in the artifact.

    Dropping ``completed_at_horizon`` from ``MECHANISM_FIELDS`` survived 176 tests, and popping a
    name from the shipped list survived 178 -- on this data the counts coincide on seven fields,
    which is exactly why nothing noticed. The paper sentence *"identical on eight fields"* would
    then rest on a list no test reads.

    The eight names are written out literally here rather than imported into the expectation, so the
    constant and the artifact are both checked against the brief's text and not against each other.
    """
    registered = (
        "att_engine",
        "att_ours",
        "entered",
        "created",
        "never_entered",
        "horizon_vehicle_count",
        "episode_reward",
        "completed_at_horizon",
    )
    assert tuple(MECHANISM_FIELDS) == registered
    assert tuple(artifact["mechanism"]["outcome_identity"]["fields"]) == registered


def test_the_attractor_reading_never_reads_as_a_claim_on_a_tier_that_does_not_carry_it(
    artifact: dict[str, Any],
) -> None:
    """mn-3: ``mappo1000`` carries the attractor on 0 draws and its reading must say so.

    The old template produced *"the same fixed [0, 0, 0, 0, 0] of 100 draws on mappo1000 emit the
    byte-identical action sequence ... regardless of observation or demand"* -- a sentence that reads
    as a positive claim about a tier where the measurement is the opposite.
    """
    attractor = artifact["mechanism"]["action_identity"]["attractor"]
    assert attractor["mappo1000"]["n_draws_carrying_it"] == 0
    reading = attractor["mappo1000"]["reading"].lower()
    assert "no draw" in reading or "does not" in reading, reading
    assert "regardless of observation" not in reading
    # and the tier that DOES carry it still says so
    assert attractor["mix50"]["n_draws_carrying_it"] > 0
    assert "regardless of observation" in attractor["mix50"]["reading"]


def test_the_mechanism_ships_a_reading_of_the_registered_shape(artifact: dict[str, Any]) -> None:
    """``BRIEF_33`` section 3.3(3) fixes the shape of the sentence, and forbids any *why*."""
    reading = artifact["mechanism"]["reading"]
    assert "3 of 5" in reading
    for number in ("99", "97", "88", "105.59"):
        assert number in reading, number
    for forbidden in ("because", "cause", "due to", "explains why", "the reason"):
        assert forbidden not in reading.lower(), forbidden


def test_the_action_histograms_cover_every_collapsed_mix50_cell(artifact: dict[str, Any]) -> None:
    """mn-4: section 3.3(2) asks for *the collapsed cells* -- 202, 404 and 505, not 404 alone."""
    histograms = artifact["mechanism"]["action_identity"]["action_counts"]
    for seed in (101, 202, 404, 505):
        assert f"dt_nortg@mix50 seed {seed}" in histograms, seed
