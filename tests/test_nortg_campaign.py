"""``offline.nortg_campaign`` -- the P5.3b ``dt_nortg`` campaign, without a simulator or a GPU.

What this file is defending, in order of how much it would cost to get wrong
----------------------------------------------------------------------------
1. **The tier set is a RULE evaluated on committed data, not three names.**  ``BRIEF_28`` section 9
   registered the rule on 2026-08-24, before any spread number existed; P5.3a's row B resolved it.
   The test re-evaluates it from ``docs/data/p5_3a_rtg_probe.json`` rather than comparing constants
   to their own literals -- the class ``docs/reviews/P5.3a.md`` filed as theatre #8.
2. **The paired statistics are the repo's, imported and CALLED.**  ``docs/reviews/P5.2.md`` **MJ-4**:
   that packet's docstring claimed the protocol was reused from ``dt_gate._paired``,
   ``wilcoxon_signed_rank`` and ``offline_baselines.paired_comparison`` -- *"none of which was
   imported or called"*.  Three tests here fail if any of the three calls is removed.
3. **The probe is P5.3a's, one layer down.**  ``BRIEF_30`` section 4.4 asks for
   ``offline/rtg_ablation.py probe``; its CLI resolves checkpoints from ``_CHECKPOINT_LAYOUT`` and
   cannot address a ``dt_nortg`` file, so the campaign calls ``probe_cell`` directly with an
   explicit path (plan section 8, F2 -- confirmed by AMENDMENT A4).  A test pins that it really is
   that function and not a reimplementation.
4. **No equivalence threshold and no verdict, anywhere** (``PREREGISTRATION`` A7, ``BRIEF_30``
   section 5).  A CI containing 0 is a failure to reject, never a demonstration of equivalence.
5. **The output fence is code-enforced**, including the trap that ``output/p5_3a`` is fenced while
   ``output/p5_3b`` is not -- a string-prefix implementation would refuse this task's own directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from offline import nortg_campaign
from offline.dt_gate import EpisodeResult, _paired, wilcoxon_signed_rank
from offline.method_tier_grid import METHODS, TIERS, arm_key, assert_no_verdicts
from offline.offline_baselines import paired_comparison
from offline.nortg_campaign import (
    COMPARED_PAYLOAD_KEYS,
    EXCLUDED_PAYLOAD_KEYS,
    GATE_1B_CELLS,
    NORTG_METHOD,
    NORTG_RTG_MODE,
    NORTG_TIERS,
    assert_arm_validity,
    assert_payload_matches_committed,
    assert_writable,
    nortg_arm_key,
    paired_stats,
    per_seed_differences,
    probe_nortg_cell,
    report_artifact,
    row_b_pooled_scaled,
    score_q1,
    score_q2,
    select_tiers,
)
from offline.rtg_ablation import INTERVENTION_KEYS, PROBE_SEEDS, PROBE_TIERS

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "data"

SEEDS = (101, 202, 303, 404, 505)
DRAWS = (1000, 1001, 1002, 1003)


def _episodes(arm: str, per_seed_offset: dict[int, float], base: float = 100.0) -> list[EpisodeResult]:
    """One arm over ``SEEDS`` x ``DRAWS``; each seed sits ``per_seed_offset[seed]`` above *base*."""
    return [
        EpisodeResult(
            arm=arm,
            seed=seed,
            draw_id=draw,
            att_horizon=base + float(draw - DRAWS[0]) + per_seed_offset[seed],
            horizon_vehicle_count=40.0,
            episode_reward=-1.0,
        )
        for seed in SEEDS
        for draw in DRAWS
    ]


@pytest.fixture()
def paired_arms() -> tuple[list[EpisodeResult], list[EpisodeResult]]:
    """``dt`` minus ``dt_nortg`` is ``+13, -2, -2, -2, -2`` by seed: pooled ``+1.0``, 4 reversals.

    4 is deliberately none of ``ceil(5/2) = 3``, ``floor(5/2) = 2``, ``5/2 = 2.5``, ``0`` or ``5``,
    so no rounding rule agrees with it by accident (P5.2 **MN-5**).  The pooled mean is carried by a
    single seed, which is exactly the shape a per-seed qualifier exists to expose.
    """
    deltas = {101: 13.0, 202: -2.0, 303: -2.0, 404: -2.0, 505: -2.0}
    dt = _episodes("dt@mix50", deltas)
    nortg = _episodes("dt_nortg@mix50", dict.fromkeys(SEEDS, 0.0))
    return dt, nortg


# ----------------------------------------------------------------------
# The declared design
# ----------------------------------------------------------------------


def test_the_declared_arm_and_mode_are_the_registered_ones() -> None:
    assert NORTG_METHOD == "dt_nortg"
    assert NORTG_RTG_MODE == "zero"
    assert nortg_arm_key("mix50") == "dt_nortg@mix50"


def test_the_campaign_keys_its_own_arm_because_METHODS_may_not_grow() -> None:
    """``BRIEF_30`` section 6.5 forbids editing ``METHODS``; ``arm_key`` validates against it.

    ``offline/method_tier_grid.py:1701`` records that ``grid_comparisons`` emits pairs in
    ``METHODS`` order, so an entry added there would change the comparison enumeration of two
    merged artifacts.  This test pins both halves of the constraint: the tuple is untouched, and
    the campaign therefore cannot use ``arm_key`` for its own arm.
    """
    assert METHODS == ("bc", "bc_top10", "iql", "dt")
    assert NORTG_METHOD not in METHODS
    with pytest.raises(ValueError, match="unknown method 'dt_nortg'"):
        arm_key(NORTG_METHOD, "mix50")


def test_the_tier_rule_is_re_evaluated_from_committed_row_b_and_returns_the_three() -> None:
    """⭐ The rule, not the answer.  ``BRIEF_28`` section 9 evaluated on P5.3a's row B.

    Row B is the **between-episode** sd of the scaled RTG.  It is the axis because the marginal
    ``RtgSummary.std`` is 65-93 % within-episode ramp on the single-policy tiers and is fitted on
    the wrong population entirely for the mixtures (``docs/plans/p5.3a.md`` section 2).
    """
    probe = json.loads((DATA / "p5_3a_rtg_probe.json").read_text(encoding="utf-8"))
    pooled = row_b_pooled_scaled(probe)
    assert sorted(pooled) == sorted(PROBE_TIERS)
    assert len(pooled) == 8

    widest = max(pooled, key=lambda tier: pooled[tier])
    narrowest = min(pooled, key=lambda tier: pooled[tier])
    assert widest == "mix50"
    assert narrowest == "random"
    assert pooled[widest] / pooled[narrowest] > 50.0, pooled

    selection = select_tiers(probe)
    assert selection["widest"] == "mix50"
    assert selection["narrowest"] == "random"
    assert selection["declared"] == "mappo1000"
    assert tuple(selection["tiers"]) == NORTG_TIERS == ("mappo1000", "mix50", "random")
    assert selection["fallback_fired"] is False, (
        "the rule's 'if (ii) or (iii) is mappo1000, take the next one' clause did not fire, and "
        "recording that is what makes the selection auditable"
    )
    assert all(tier in TIERS for tier in NORTG_TIERS)


def test_gate_1b_reroll_cells_cover_all_three_provenances() -> None:
    """AMENDMENT A1: one re-roll per tier, because the three ``dt`` columns come from three places.

    ``mappo1000`` from ``output/p4_dt/`` (P4's reused column, in **no** integrity manifest --
    ``DEFERRED`` 56), ``mix50`` from ``output/p4_7/checkpoints/``, ``random`` from
    ``output/p4_6/checkpoints/``.
    """
    assert tuple(tier for tier, _ in GATE_1B_CELLS) == NORTG_TIERS
    assert {seed for _, seed in GATE_1B_CELLS} == {101}
    assert all(seed in PROBE_SEEDS for _, seed in GATE_1B_CELLS)


# ----------------------------------------------------------------------
# The output fence, and the filesystem-mutation barrier
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    ["p4_6/checkpoints/x.pt", "p4_7/checkpoints/x.pt", "p4_dt/dt_seed101.pt", "p5_3a/probe.json"],
)
def test_writing_into_another_campaigns_directory_is_refused(relative: str, tmp_path: Path) -> None:
    root = tmp_path / "output"
    target = root / relative
    with pytest.raises(ValueError, match="belongs to another campaign and is read-only here"):
        assert_writable(target)


def test_this_tasks_own_directory_is_writable_even_though_p5_3a_is_fenced(tmp_path: Path) -> None:
    """⭐ The prefix trap: ``output/p5_3a`` is fenced and ``output/p5_3b`` is not.

    A string-prefix implementation of the fence would refuse this task's own work directory, or --
    worse, if the comparison ran the other way -- would let ``output/p5_3a`` through.
    """
    root = tmp_path / "output"
    assert assert_writable(root / "p5_3b" / "checkpoints" / "x.pt") == root / "p5_3b" / "checkpoints" / "x.pt"
    with pytest.raises(ValueError, match="belongs to another campaign and is read-only here"):
        assert_writable(root / "p5_3a" / "x.json")


def test_the_fenced_set_names_every_merged_campaign_directory() -> None:
    from offline.nortg_campaign import FENCED_OUTPUT_DIRS

    assert set(FENCED_OUTPUT_DIRS) >= {
        "p4_3", "p4_4", "p4_5", "p4_6", "p4_7", "p4_dt", "p4_probe",
        "p5_1", "p5_2", "p5_3a", "p7_0", "p8_3",
    }
    assert "p5_3b" not in FENCED_OUTPUT_DIRS


def test_a_report_whose_inputs_are_missing_writes_nothing_and_creates_no_directory(
    tmp_path: Path,
) -> None:
    """The filesystem-mutation barrier: validate, then write.  Never the other way round."""
    work = tmp_path / "work"
    work.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="run `train` and `evaluate` for it first"):
        nortg_campaign.main(
            [
                "--corpus-root", str(tmp_path / "corpus"),
                "--work-dir", str(work),
                "--out-dir", str(out_dir),
                "report",
            ]
        )
    assert list(out_dir.iterdir()) == []
    assert list(work.iterdir()) == []


# ----------------------------------------------------------------------
# The paired statistics: imported, called, and each call independently detectable
# ----------------------------------------------------------------------


def test_paired_stats_returns_exactly_what_paired_comparison_returns(
    paired_arms: tuple[list[EpisodeResult], list[EpisodeResult]],
) -> None:
    """Field for field against a direct call, so the protocol is reused rather than described."""
    dt, nortg = paired_arms
    direct = paired_comparison(dt, nortg).to_json_obj()
    ours = paired_stats(dt, nortg)
    assert ours["paired"] == direct
    assert ours["paired"]["n_shared_draws"] == len(DRAWS)
    assert ours["paired"]["left_arm"] == "dt@mix50"
    assert ours["paired"]["right_arm"] == "dt_nortg@mix50"
    assert ours["abs_mean_difference"] == abs(direct["mean_difference"])


def test_removing_the_paired_comparison_call_is_detectable(
    monkeypatch: pytest.MonkeyPatch,
    paired_arms: tuple[list[EpisodeResult], list[EpisodeResult]],
) -> None:
    """MJ-4's fix, mechanically: the docstring's claim is a call the test can see."""
    def _sentinel(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("sentinel: paired_comparison really was called")

    monkeypatch.setattr(nortg_campaign, "paired_comparison", _sentinel)
    with pytest.raises(RuntimeError, match="sentinel: paired_comparison really was called"):
        paired_stats(*paired_arms)


def test_removing_the_shared_draw_pairing_call_is_detectable(
    monkeypatch: pytest.MonkeyPatch,
    paired_arms: tuple[list[EpisodeResult], list[EpisodeResult]],
) -> None:
    def _sentinel(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("sentinel: _paired really was called")

    monkeypatch.setattr(nortg_campaign, "_paired", _sentinel)
    with pytest.raises(RuntimeError, match="sentinel: _paired really was called"):
        paired_stats(*paired_arms)


def test_the_wilcoxon_double_compute_is_live_and_refuses_a_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    paired_arms: tuple[list[EpisodeResult], list[EpisodeResult]],
) -> None:
    """``wilcoxon_signed_rank`` is called on the SAME vectors and compared to the one inside
    ``PairedComparison``.  Two routes to one quantity; a disagreement is a refusal, not a warning."""
    dt, nortg = paired_arms
    left, right, shared = _paired(dt, nortg)
    assert len(shared) == len(DRAWS)
    reference = wilcoxon_signed_rank(left, right)
    assert paired_stats(dt, nortg)["paired"]["wilcoxon"]["p_value"] == reference.p_value

    def _wrong(*args: Any, **kwargs: Any) -> Any:
        return reference.__class__(
            w_plus=-1.0, w_minus=-1.0, statistic=-1.0, n_used=0, n_zero=0, z=0.0, p_value=0.5
        )

    monkeypatch.setattr(nortg_campaign, "wilcoxon_signed_rank", _wrong)
    with pytest.raises(ValueError, match="the two Wilcoxon routes disagree"):
        paired_stats(dt, nortg)


def test_a_comparison_without_shared_draws_is_void_and_raises(
    paired_arms: tuple[list[EpisodeResult], list[EpisodeResult]],
) -> None:
    """``PREREGISTRATION`` A5 point 3: not approximate -- void."""
    dt, _ = paired_arms
    elsewhere = [
        EpisodeResult(
            arm="dt_nortg@mix50", seed=101, draw_id=draw, att_horizon=1.0,
            horizon_vehicle_count=1.0, episode_reward=-1.0,
        )
        for draw in (2000, 2001)
    ]
    with pytest.raises(ValueError, match="makes this comparison"):
        paired_stats(dt, elsewhere)


def test_per_seed_reversals_counts_four_of_five_on_a_fixture_no_rounding_rule_matches(
    paired_arms: tuple[list[EpisodeResult], list[EpisodeResult]],
) -> None:
    dt, nortg = paired_arms
    pooled = paired_stats(dt, nortg)["paired"]["mean_difference"]
    assert pooled > 0.0
    record = per_seed_differences(dt, nortg, pooled)
    assert record["seeds_reversed"] == 4
    assert record["n_seeds"] == 5
    assert record["per_seed"]["101"] == 13.0
    assert record["per_seed"]["505"] == -2.0
    assert 4 not in {3, 2, 0, 5}, "guard against a rounding rule agreeing by accident (MN-5)"


def test_a_seed_whose_difference_is_exactly_zero_counts_as_a_reversal() -> None:
    """The conservative direction, registered in ``docs/plans/p5.3b.md`` section 3.4."""
    deltas = {101: 5.0, 202: 5.0, 303: 5.0, 404: 5.0, 505: 0.0}
    dt = _episodes("dt@mix50", deltas)
    nortg = _episodes("dt_nortg@mix50", dict.fromkeys(SEEDS, 0.0))
    pooled = paired_stats(dt, nortg)["paired"]["mean_difference"]
    record = per_seed_differences(dt, nortg, pooled)
    assert record["per_seed"]["505"] == 0.0
    assert record["seeds_reversed"] == 1


# ----------------------------------------------------------------------
# Q1 and Q2: scored, and no verdict anywhere
# ----------------------------------------------------------------------


def _comparisons(values: dict[str, float], ci: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    bounds = ci or {tier: (value - 1.0, value + 1.0) for tier, value in values.items()}
    return {
        tier: {
            "paired": {
                "mean_difference": value,
                "ci95_low": bounds[tier][0],
                "ci95_high": bounds[tier][1],
            },
            "abs_mean_difference": abs(value),
            "mean_absolute_difference": abs(value),
            "att_dt_mean": 100.0,
        }
        for tier, value in values.items()
    }


def test_q1_holds_when_both_endpoints_are_in_place() -> None:
    scored = score_q1(_comparisons({"mix50": -3.0, "mappo1000": -1.0, "random": 0.2}))
    assert scored["largest"] == "mix50"
    assert scored["smallest"] == "random"
    assert scored["holds"] is True
    assert scored["scale"] == "raw ATT"


@pytest.mark.parametrize(
    ("values", "why"),
    [
        ({"mix50": -1.0, "mappo1000": -3.0, "random": 0.2}, "mix50 displaced from largest"),
        ({"mix50": -3.0, "mappo1000": -0.1, "random": -1.0}, "random displaced from smallest"),
    ],
)
def test_q1_fails_when_either_endpoint_is_displaced(values: dict[str, float], why: str) -> None:
    """Both directions.  **Endpoints, never a trend** -- section 1b's R3 was falsified on
    exactly a monotonicity claim, and the standing instruction is to register endpoints."""
    scored = score_q1(_comparisons(values))
    assert scored["holds"] is False, why


#: Phrases that would CLAIM equivalence.  ⚠️ A blanket ban on the substring ``equival`` would
#: forbid the disclaimer ``BRIEF_30`` section 5 **requires** -- *"a CI containing 0 is a failure to
#: reject, never a demonstration of equivalence"* -- so the ban is on the claim forms and the
#: disclaimer is asserted PRESENT instead.  (This replaced a wrong assertion of mine; the Return
#: Packet discloses it.)
EQUIVALENCE_CLAIMS = (
    "is equivalent",
    "are equivalent",
    "equivalence margin",
    "within_delta",
    "equivalence threshold",
    "delta_att",
)


def test_q2_reports_a_ci_containing_zero_as_a_failure_to_reject_and_never_as_equivalence() -> None:
    """⚠️ A6's own words.  ``PREREGISTRATION`` A7 withdrew the per-tier delta rule; this task
    issues no equivalence verdict and defines no threshold."""
    scored = score_q2(_comparisons({"random": 0.1}, ci={"random": (-0.4, 0.6)}))
    assert scored["ci_contains_zero"] is True
    assert scored["holds"] is True
    assert "failure to reject" in scored["reading"].lower()
    assert "demonstration of equivalence" in scored["reading"].lower(), (
        "the disclaimer BRIEF_30 section 5 mandates must be present, not merely not-contradicted"
    )
    text = json.dumps(scored).lower()
    for claim in EQUIVALENCE_CLAIMS:
        assert claim not in text, claim
    assert_no_verdicts(scored)


def test_q2_reports_a_ci_excluding_zero_without_issuing_a_verdict() -> None:
    scored = score_q2(_comparisons({"random": 2.0}, ci={"random": (1.5, 2.5)}))
    assert scored["ci_contains_zero"] is False
    assert scored["holds"] is False
    text = json.dumps(scored).lower()
    for claim in EQUIVALENCE_CLAIMS:
        assert claim not in text, claim
    assert_no_verdicts(scored)


# ----------------------------------------------------------------------
# Q3: arm validity, the gate
# ----------------------------------------------------------------------


def _probe_cell(tier: str, seed: int, *, flip: float = 0.0, mode: str = "zero") -> dict[str, Any]:
    return {
        "tier": tier,
        "seed": seed,
        "checkpoint": f"output/p5_3b/checkpoints/{tier}_dt_nortg_seed{seed}.pt",
        "rtg_mode": mode,
        "n_steps": 7200,
        "n_streams": 20,
        "interventions": {
            key: {
                "flip_rate": flip if key == "grid_g8" else 0.0,
                "tvd": 0.0,
                "mean_abs_logit_delta": 0.0,
                "n_steps_compared": 7200,
            }
            for key in INTERVENTION_KEYS
        },
    }


def _all_cells(**kwargs: Any) -> list[dict[str, Any]]:
    return [_probe_cell(tier, seed, **kwargs) for tier in NORTG_TIERS for seed in PROBE_SEEDS]


def test_arm_validity_passes_when_every_flip_rate_is_exactly_zero() -> None:
    record = assert_arm_validity(_all_cells())
    assert record["n_cells"] == 15
    assert record["n_values_checked"] == 15 * len(INTERVENTION_KEYS)
    assert record["max_flip_rate"] == 0.0


def test_arm_validity_raises_naming_the_cell_and_the_intervention_on_any_non_zero_flip() -> None:
    cells = _all_cells()
    cells[7]["interventions"]["grid_g8"]["flip_rate"] = 1.0 / 7200.0
    with pytest.raises(ValueError, match="did not ignore the return token") as excinfo:
        assert_arm_validity(cells)
    message = str(excinfo.value)
    assert "grid_g8" in message and cells[7]["tier"] in message


def test_arm_validity_raises_when_a_checkpoint_is_not_in_zero_mode() -> None:
    cells = _all_cells()
    cells[3]["rtg_mode"] = "conditioned"
    with pytest.raises(ValueError, match="rtg_mode did not reach the training path"):
        assert_arm_validity(cells)


def test_arm_validity_refuses_a_cell_set_that_is_not_the_declared_fifteen() -> None:
    with pytest.raises(ValueError, match="the declared cell set is 3 tiers x 5 seeds"):
        assert_arm_validity(_all_cells()[:14])


def test_arm_validity_refuses_an_undeclared_intervention_key() -> None:
    cells = _all_cells()
    cells[0]["interventions"]["grid_g9"] = {"flip_rate": 0.0, "tvd": 0.0,
                                            "mean_abs_logit_delta": 0.0, "n_steps_compared": 7200}
    with pytest.raises(ValueError, match="the twelve declared interventions"):
        assert_arm_validity(cells)


def test_the_probe_is_p5_3as_and_not_a_reimplementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BRIEF_30`` section 4.4 asks for P5.3a's probe; plan F2 records why the CLI cannot be used.

    Calling ``rtg_ablation.probe_cell`` with an explicit ``checkpoint_path`` is the same instrument
    one layer down, and this test is what makes that a fact rather than a claim.
    """
    def _sentinel(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"sentinel: probe_cell called with checkpoint {kwargs['checkpoint_path']}")

    monkeypatch.setattr(nortg_campaign, "probe_cell", _sentinel)
    with pytest.raises(RuntimeError, match="sentinel: probe_cell called with checkpoint /tmp/x.pt"):
        probe_nortg_cell("mix50", 101, checkpoint_path="/tmp/x.pt", corpus_root="/tmp/corpus")


# ----------------------------------------------------------------------
# AMENDMENT A5: the payload comparison
# ----------------------------------------------------------------------


def test_the_a5_key_split_is_the_registered_one() -> None:
    assert EXCLUDED_PAYLOAD_KEYS == ("model", "provenance")
    assert COMPARED_PAYLOAD_KEYS == (
        "config", "format_version", "intersection_ids", "normalise",
        "rtg_scale", "scenario_id", "stats", "target_rtg",
    )
    assert set(COMPARED_PAYLOAD_KEYS).isdisjoint(EXCLUDED_PAYLOAD_KEYS)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "format_version": "dt-checkpoint/1.0",
        "config": {"state_dim": 25, "n_actions": 8, "rtg_mode": "conditioned"},
        "model": {"w": [1.0, 2.0]},
        "target_rtg": -6362.0,
        "rtg_scale": 11043.0,
        "normalise": True,
        "scenario_id": "cityflow1x1",
        "stats": {"rtg": {"count": 72000}},
        "intersection_ids": [],
        "provenance": {"seed": 101, "seconds": 213.3},
    }
    payload.update(overrides)
    return payload


def test_the_a5_comparison_accepts_payloads_that_differ_only_in_provenance(tmp_path: Path) -> None:
    import torch

    left, right = tmp_path / "l.pt", tmp_path / "r.pt"
    torch.save(_payload(), left)
    torch.save(_payload(provenance={"seed": 101, "seconds": 999.9, "device": "cuda"}), right)
    record = assert_payload_matches_committed(left, right)
    assert record["differing_keys"] == []
    assert tuple(record["compared_keys"]) == COMPARED_PAYLOAD_KEYS


@pytest.mark.parametrize("key", ["target_rtg", "rtg_scale", "config", "stats", "scenario_id"])
def test_the_a5_comparison_refuses_a_difference_in_any_compared_key(key: str, tmp_path: Path) -> None:
    import torch

    changed = {
        "target_rtg": -1.0, "rtg_scale": 1.0, "config": {"state_dim": 1},
        "stats": {"rtg": {"count": 1}}, "scenario_id": "elsewhere",
    }[key]
    left, right = tmp_path / "l.pt", tmp_path / "r.pt"
    torch.save(_payload(), left)
    torch.save(_payload(**{key: changed}), right)
    with pytest.raises(ValueError, match="payload keys differ outside model and provenance") as e:
        assert_payload_matches_committed(left, right)
    assert key in str(e.value)


def test_the_a5_comparison_refuses_a_payload_that_gained_or_lost_a_key(tmp_path: Path) -> None:
    """A later payload growing a key must not slip through uncompared."""
    import torch

    left, right = tmp_path / "l.pt", tmp_path / "r.pt"
    grown = _payload()
    grown["new_field"] = 1
    torch.save(grown, left)
    torch.save(_payload(), right)
    with pytest.raises(ValueError, match="the two payloads do not carry the same keys"):
        assert_payload_matches_committed(left, right)


# ----------------------------------------------------------------------
# The artifact: shape, and what it refuses
# ----------------------------------------------------------------------


def _minimal_report_inputs() -> dict[str, Any]:
    cells = [
        {
            "arm": nortg_arm_key(tier), "method": NORTG_METHOD, "tier": tier, "seed": seed,
            "n_episodes": 100, "att_horizon_mean": 100.0, "att_horizon_std": 1.0,
            "att_horizon_ci95": 0.2, "horizon_vehicle_count_mean": 40.0,
            "horizon_vehicle_count_std": 1.0, "draw_ids": list(DRAWS), "seeds": [seed],
        }
        for tier in NORTG_TIERS
        for seed in PROBE_SEEDS
    ]
    comparisons = _comparisons({tier: -1.0 for tier in NORTG_TIERS})
    for tier in comparisons:
        comparisons[tier]["per_seed"] = {"seeds_reversed": 0, "n_seeds": 5, "per_seed": {}}
    return {
        "cells": cells,
        "episodes": [],
        "comparisons": comparisons,
        "probe_cells": _all_cells(),
        "gates": {"gate_1": {}, "gate_1b": {}, "gate_2": {}},
        "selection": {"tiers": list(NORTG_TIERS)},
        "timings": {},
    }


def test_the_assembled_artifact_carries_no_verdict_and_no_threshold() -> None:
    payload = report_artifact(**_minimal_report_inputs())
    assert payload["format_version"] == "p5.3b-nortg/1.0"
    assert_no_verdicts(payload)
    text = json.dumps(payload).lower()
    for token in ("equivalent", "within_delta", "equivalence margin", "delta_att", "inert"):
        assert token not in text, token


def test_the_artifact_refuses_a_tier_outside_the_registered_three() -> None:
    inputs = _minimal_report_inputs()
    inputs["comparisons"] = dict(inputs["comparisons"])
    inputs["comparisons"]["mix33"] = inputs["comparisons"]["mix50"]
    with pytest.raises(ValueError, match="the registered tier set is"):
        report_artifact(**inputs)


def test_the_artifact_refuses_a_cell_set_that_is_not_three_tiers_by_five_seeds() -> None:
    inputs = _minimal_report_inputs()
    inputs["cells"] = inputs["cells"][:14]
    with pytest.raises(ValueError, match="the declared cell set is 3 tiers x 5 seeds"):
        report_artifact(**inputs)
