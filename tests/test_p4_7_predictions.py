"""Gate P1 and the registered predictions Q1-Q3 (``docs/plans/p4.7.md`` sections 4 and 6.3).

**The scoring rules were fixed in the plan before any P4.7 number existed** and are transcribed here
as executable form.  Each is tested on fixtures with a hand-computed outcome, **including its
boundary**: a rule whose threshold is never exercised at the edge is a rule nobody has checked.

⚠️ **These tests never read the real artifacts.**  The outcomes go in the Return Packet whichever way
they fall, and a test that asserted the real outcome would be a test written after seeing it -- which
is the exact failure ``PREREGISTRATION`` A8(a) exists to prevent.  What is tested is that the SCORER
implements the declared rule.
"""

from __future__ import annotations

from typing import Any

import pytest

from offline.method_tier_grid import MIXTURE_TIER_ORDER, TRAINING_STREAM_COUNT
from offline.mixture_tiers import (
    Q1_EXPERT_FRACTION_FLOOR,
    assert_phase1_reproduces,
    component_return_overlap,
    expert_dir_names,
    kept_expert_counts,
    score_q1,
    score_q2,
    score_q3,
)

EXPERT_DIR = "cf_hz1x1__mappo1000__seed101"
RANDOM_DIR = "cf_hz1x1__random"


def declaration_with(kept: dict[str, int], streams: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """A declaration carrying a chosen kept-composition per mixture tier."""
    tiers: dict[str, Any] = {}
    for tier in MIXTURE_TIER_ORDER:
        expert = kept[tier]
        tiers[tier] = {
            "top_decile_streams": 20,
            "training_streams": TRAINING_STREAM_COUNT,
            "top_decile_composition": {
                "by_dataset_dir": {EXPERT_DIR: expert, RANDOM_DIR: 20 - expert},
                "by_behaviour_seed": {"101": expert},
                "without_a_behaviour_seed": 20 - expert,
                "total": 20,
            },
            "training_composition": {
                "by_dataset_dir": {EXPERT_DIR: 100, RANDOM_DIR: 100},
                "total": TRAINING_STREAM_COUNT,
            },
        }
        if streams is not None:
            tiers[tier]["streams"] = [
                {"dataset_dir": f"/corpus/{EXPERT_DIR}", "total_return": value}
                for value in streams["expert"]
            ] + [
                {"dataset_dir": f"/corpus/{RANDOM_DIR}", "total_return": value}
                for value in streams["random"]
            ]
    return {"tiers": tiers}


# ----------------------------------------------------------------------
# Q1
# ----------------------------------------------------------------------


def test_the_expert_directories_come_from_the_declared_spec() -> None:
    """Five ``mappo1000`` seed directories, read from the spec rather than written out again."""
    names = expert_dir_names()
    assert len(names) == 5
    assert all(name.startswith("cf_hz1x1__mappo1000__seed") for name in names)
    assert RANDOM_DIR not in names


def test_q1_counts_the_kept_expert_streams_from_the_declaration() -> None:
    """The count is read off the kept set's own composition, not off the fraction that was asked for."""
    declaration = declaration_with({"mix33": 18, "mix50": 19, "mix67": 20})
    counts = kept_expert_counts(declaration)
    assert {tier: entry["expert"] for tier, entry in counts.items()} == {
        "mix33": 18,
        "mix50": 19,
        "mix67": 20,
    }
    assert all(entry["kept"] == 20 for entry in counts.values())


def test_q1_holds_at_the_declared_floor_and_fails_one_below_it() -> None:
    """18 of 20 is 90 %: HELD.  17 is 85 %: FAILED.  The boundary is exercised, not described."""
    assert Q1_EXPERT_FRACTION_FLOOR == 0.90
    held = score_q1(declaration_with({"mix33": 18, "mix50": 19, "mix67": 20}))
    assert held["outcome"] == "HELD"
    assert held["by_tier"]["mix33"]["fraction"] == pytest.approx(0.90)

    failed = score_q1(declaration_with({"mix33": 17, "mix50": 20, "mix67": 20}))
    assert failed["outcome"] == "FAILED"
    assert failed["failing_tiers"] == ["mix33"]


def test_q1_fails_when_any_single_tier_is_below_the_floor() -> None:
    """"All three" is the rule, so one tier below the floor is enough to falsify it."""
    for below in MIXTURE_TIER_ORDER:
        kept = {tier: 20 for tier in MIXTURE_TIER_ORDER}
        kept[below] = 17
        result = score_q1(declaration_with(kept))
        assert result["outcome"] == "FAILED", below
        assert result["failing_tiers"] == [below], below


def test_q1_carries_its_exact_hypergeometric_companion_and_the_seed_histogram() -> None:
    """The companion carries no threshold; the histogram is A5's second axis."""
    result = score_q1(declaration_with({"mix33": 18, "mix50": 19, "mix67": 20}))
    entry = result["by_tier"]["mix33"]
    assert 0.0 <= entry["hypergeometric_p_value"] <= 1.0
    assert entry["by_behaviour_seed"] == {"101": 18}
    assert "threshold" not in entry["companion_note"].lower() or "no threshold" in entry[
        "companion_note"
    ].lower()


def test_the_component_overlap_disclosure_is_measured_not_assumed() -> None:
    """Q1 is near-tautological if the components' returns do not overlap, so the overlap is measured.

    Two extremes with a hand-computable answer: disjoint distributions give a common-language effect
    size of exactly 1.0 and no random stream above the expert minimum; identical ones give 0.5.
    """
    disjoint = declaration_with(
        {t: 20 for t in MIXTURE_TIER_ORDER},
        streams={"expert": [-1.0, -2.0, -3.0], "random": [-10.0, -11.0, -12.0]},
    )
    result = component_return_overlap(disjoint, "mix50")
    assert result["common_language_effect_size"] == 1.0
    assert result["random_above_expert_minimum"] == 0
    assert result["separates_completely"] is True

    identical = declaration_with(
        {t: 20 for t in MIXTURE_TIER_ORDER},
        streams={"expert": [-1.0, -2.0], "random": [-1.0, -2.0]},
    )
    overlapping = component_return_overlap(identical, "mix50")
    assert overlapping["common_language_effect_size"] == 0.5
    assert overlapping["separates_completely"] is False


# ----------------------------------------------------------------------
# Q2
# ----------------------------------------------------------------------


def cells(advantages: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Cells whose bc/bc_top10 means realise the given advantage (bc minus bc_top10)."""
    return {
        tier: {
            "bc": {"att_horizon_mean": 200.0},
            "bc_top10": {"att_horizon_mean": 200.0 - value},
        }
        for tier, value in advantages.items()
    }


def test_q2_holds_only_when_all_three_are_positive_and_strictly_decreasing() -> None:
    """The rule is conjunctive and the ordering is strict, exactly as registered."""
    held = score_q2(cells({"mix33": 9.0, "mix50": 5.0, "mix67": 1.0}), [])
    assert held["outcome"] == "HELD"
    assert held["advantage_by_tier"]["mix33"] == 9.0

    wrong_order = score_q2(cells({"mix33": 1.0, "mix50": 5.0, "mix67": 9.0}), [])
    assert wrong_order["outcome"] == "FAILED"
    assert wrong_order["ordering_holds"] is False
    assert wrong_order["all_positive"] is True

    negative = score_q2(cells({"mix33": 9.0, "mix50": 5.0, "mix67": -1.0}), [])
    assert negative["outcome"] == "FAILED"
    assert negative["all_positive"] is False


def test_q2_is_not_resolved_when_two_advantages_are_exactly_equal() -> None:
    """A tie makes "decreases" undefined; the registered rule says NOT RESOLVED, not HELD.

    ``ordering_holds`` is asserted too, and that is not redundant: a mutation relaxing the strict
    ``>`` to ``>=`` survived when only the outcome was checked, because the tie branch is evaluated
    first.  The artifact would then have reported ``ordering_holds: True`` beside
    ``outcome: NOT RESOLVED`` -- internally inconsistent, and P4.6's finding M4 is exactly that
    class of defect.
    """
    tied = score_q2(cells({"mix33": 5.0, "mix50": 5.0, "mix67": 1.0}), [])
    assert tied["outcome"] == "NOT RESOLVED"
    assert tied["ordering_holds"] is False


def test_q2_is_not_scorable_without_all_three_mixtures() -> None:
    """A missing tier is unscorable, never a pass on the two that are present."""
    partial = score_q2(cells({"mix33": 9.0, "mix50": 5.0}), [])
    assert partial["outcome"] == "NOT SCORABLE"


def test_q2_carries_the_paired_interval_for_each_mixture() -> None:
    """The companion: a paired difference with its CI, its WIDTH and its effect size, no verdict."""

    class FakeComparison:
        def __init__(self, tier: str) -> None:
            self.left_arm = f"bc@{tier}"
            self.right_arm = f"bc_top10@{tier}"
            self.mean_difference = 9.0
            self.ci95_low = 7.0
            self.ci95_high = 11.0
            self.ci95_width = 4.0
            self.rank_biserial = 0.5

    result = score_q2(
        cells({"mix33": 9.0, "mix50": 5.0, "mix67": 1.0}),
        [FakeComparison(t) for t in MIXTURE_TIER_ORDER],
    )
    interval = result["advantage_intervals"]["mix33"]
    assert interval["ci95_width"] == 4.0
    assert interval["mean_difference"] == 9.0
    # ⚠️ The repo's own checker, not a substring search.  A first draft of this line asserted
    # `"verdict" not in str(result).lower()`, which fails on the companion note that says "no
    # equivalence verdict anywhere" -- i.e. it forbade the word in the sentence forbidding the
    # thing.  `assert_no_verdicts` is what BRIEF_17 section 4 is actually enforced by: it rejects
    # any KEY containing "verdict" and any VALUE that is one of the declared verdict strings.
    from offline.method_tier_grid import assert_no_verdicts

    assert_no_verdicts(result)


def test_q2_refuses_a_reversed_pair_rather_than_flipping_its_sign() -> None:
    """``bc - bc_top10`` fixes the sign; a reversed pair would silently invert every advantage."""

    class Reversed:
        left_arm = "bc_top10@mix33"
        right_arm = "bc@mix33"
        mean_difference = 9.0
        ci95_low = 7.0
        ci95_high = 11.0
        ci95_width = 4.0
        rank_biserial = 0.5

    with pytest.raises(ValueError, match="sign"):
        score_q2(cells({"mix33": 9.0, "mix50": 5.0, "mix67": 1.0}), [Reversed()])


# ----------------------------------------------------------------------
# Q3
# ----------------------------------------------------------------------


def diagnostics_with(volume_excludes: bool, difficulty_p: float) -> dict[str, Any]:
    return {
        "tiers": {
            tier: {
                "volume": {
                    "difference": -3.0,
                    "ci95_low": 1.0 if volume_excludes else -9.0,
                    "ci95_high": 5.0 if volume_excludes else 3.0,
                    "excludes_zero": volume_excludes,
                },
                "difficulty": {
                    "overlap": 2,
                    "kept_count": 20,
                    "expected_overlap": 2.0,
                    "p_value": difficulty_p,
                    "withdrawn": False,
                },
                "return_versus_difficulty_rho": -0.05,
            }
            for tier in MIXTURE_TIER_ORDER
        }
    }


def test_q3_holds_when_composition_separates_and_demand_does_not() -> None:
    """The registered conjunction: composition significant AND demand absent, on all three."""
    result = score_q3(
        declaration_with({"mix33": 20, "mix50": 20, "mix67": 20}),
        diagnostics_with(volume_excludes=False, difficulty_p=0.62),
    )
    assert result["outcome"] == "HELD"
    entry = result["by_tier"]["mix33"]
    assert entry["composition_signature"] is True
    assert entry["demand_signature"] is False


def test_q3_fails_when_the_demand_signature_is_present() -> None:
    """A demand signature falsifies "selects MODE, not DIFFICULTY" even with composition perfect."""
    by_volume = score_q3(
        declaration_with({"mix33": 20, "mix50": 20, "mix67": 20}),
        diagnostics_with(volume_excludes=True, difficulty_p=0.62),
    )
    assert by_volume["outcome"] == "FAILED"
    assert by_volume["by_tier"]["mix33"]["demand_signature"] is True

    by_difficulty = score_q3(
        declaration_with({"mix33": 20, "mix50": 20, "mix67": 20}),
        diagnostics_with(volume_excludes=False, difficulty_p=0.001),
    )
    assert by_difficulty["outcome"] == "FAILED"


def test_q3_fails_when_the_composition_signature_is_absent() -> None:
    """⚠️ The most informative outcome available: a null composition falsifies R2 outright.

    Six expert streams of twenty on ``mix33`` is close to the 33 % the corpus holds, so the
    hypergeometric tail is nowhere near 0.05 and the check reports no signature.
    """
    result = score_q3(
        declaration_with({"mix33": 6, "mix50": 10, "mix67": 13}),
        diagnostics_with(volume_excludes=False, difficulty_p=0.62),
    )
    assert result["outcome"] == "FAILED"
    assert result["by_tier"]["mix33"]["composition_signature"] is False
    assert result["falsifies_r2"] is True


def test_q3_reports_every_sub_check_whichever_way_it_falls() -> None:
    """A failed prediction still reports its numbers: withdrawing is not the same as hiding."""
    result = score_q3(
        declaration_with({"mix33": 6, "mix50": 10, "mix67": 13}),
        diagnostics_with(volume_excludes=True, difficulty_p=0.001),
    )
    for tier in MIXTURE_TIER_ORDER:
        entry = result["by_tier"][tier]
        for field in (
            "composition_p_value",
            "kept_expert",
            "expected_expert",
            "volume_ci95",
            "difficulty_p_value",
            "return_versus_difficulty_rho",
        ):
            assert field in entry, (tier, field)


def test_a_withdrawn_difficulty_check_cannot_carry_a_demand_signature() -> None:
    """A withdrawn check contributes nothing, whatever its p-value says (``BRIEF_18`` finding F3).

    Dormant on P4.7's own data -- only ``maxpressure`` withdraws, and no mixture is that tier -- but
    it is a guard, and a guard whose removal is invisible is ``DEFERRED`` 44's family.  A mutation
    that let a withdrawn check count survived until this test existed.
    """
    diagnostics = diagnostics_with(volume_excludes=False, difficulty_p=0.0001)
    for entry in diagnostics["tiers"].values():
        entry["difficulty"]["withdrawn"] = True
        entry["difficulty"]["withdrawn_reason"] = "circular on this tier by construction"
    result = score_q3(declaration_with({"mix33": 20, "mix50": 20, "mix67": 20}), diagnostics)
    for tier in MIXTURE_TIER_ORDER:
        entry = result["by_tier"][tier]
        assert entry["difficulty_withdrawn"] is True, tier
        assert entry["difficulty_p_value"] == 0.0001, tier
        assert entry["demand_signature"] is False, tier
    assert result["outcome"] == "HELD"


def test_q3_records_the_multiplicity_of_draws_shared_by_both_components() -> None:
    """The volume check works on SETS of draw ids, so a mixture's repeats must be disclosed."""
    result = score_q3(
        declaration_with({"mix33": 20, "mix50": 20, "mix67": 20}),
        diagnostics_with(volume_excludes=False, difficulty_p=0.62),
    )
    assert "multiplicity" in result
    assert "set" in result["multiplicity"].lower()


# ----------------------------------------------------------------------
# Gate P1
# ----------------------------------------------------------------------


def phase1_payload(att: float = 105.5) -> dict[str, Any]:
    return {
        "cells_by_tier": {
            "mappo1000": {"bc": {"att_horizon_mean": att, "n_episodes": 500}},
            "mix33": {"bc": {"att_horizon_mean": 300.0, "n_episodes": 500}},
        },
        "behaviour_cells": {"mappo1000": {"att_horizon_mean": 105.58203462874322}},
        "comparisons": [
            {"left_arm": "bc@mappo1000", "right_arm": "iql@mappo1000", "mean_difference": 1.6871},
            {"left_arm": "bc@mix33", "right_arm": "iql@mix33", "mean_difference": 9.0},
        ],
        "behaviour_comparisons": [
            {"left_arm": "bc@mappo1000", "right_arm": "behaviour@mappo1000",
             "mean_difference": -0.4186},
        ],
    }


def test_gate_p1_passes_when_every_phase1_leaf_is_identical() -> None:
    """The re-used column must regenerate bit-identically, and the gate says what it compared."""
    record = assert_phase1_reproduces(phase1_payload(), phase1_payload())
    assert record["status"] == "PASS"
    assert record["cells_compared"] == 1
    assert record["behaviour_cells_compared"] == 1
    assert record["comparisons_compared"] == 1
    assert record["behaviour_comparisons_compared"] == 1


def test_gate_p1_refuses_a_difference_of_one_part_in_a_trillion() -> None:
    """⚠️ Exact equality, never a tolerance.

    The perturbation is 1e-12 of an ATT unit -- far below anything that could matter scientifically,
    and exactly the size at which a tolerance would let a real regression through unnoticed.  The
    gate's job is to prove the SAME numbers, not similar ones.
    """
    with pytest.raises(ValueError, match="mappo1000"):
        assert_phase1_reproduces(phase1_payload(105.5 + 1e-12), phase1_payload(105.5))


def test_gate_p1_ignores_the_mixture_tiers_it_is_not_gating() -> None:
    """``mix33`` differs between the two payloads by construction and must not be compared."""
    candidate = phase1_payload()
    candidate["cells_by_tier"]["mix33"]["bc"]["att_horizon_mean"] = 999.0
    record = assert_phase1_reproduces(candidate, phase1_payload())
    assert record["status"] == "PASS"


def test_gate_p1_refuses_a_missing_phase1_cell() -> None:
    """A cell that vanished is a refusal, not a smaller comparison that still passes."""
    candidate = phase1_payload()
    del candidate["cells_by_tier"]["mappo1000"]["bc"]
    with pytest.raises(ValueError, match="bc@mappo1000|missing"):
        assert_phase1_reproduces(candidate, phase1_payload())


def test_an_unlabelled_constructed_reference_is_refused() -> None:
    """A constructed cell that does not say so is indistinguishable from a rolled-out one.

    ``method_tier_grid.grid_artifact`` labels behaviour cells from ``BEHAVIOUR_REFERENCE_BY_TIER``,
    which declares nothing for a mixture -- deliberately, since a mixture has two behaviour
    policies.  P4.7 supplies the label, and its ABSENCE is an error rather than a ``None`` a reader
    has to notice (``BRIEF_19`` section 3, constraint 1).
    """
    from offline.mixture_tiers import label_constructed_references

    payload = {
        "behaviour_cells": {
            "mappo1000": {"att_horizon_mean": 105.0, "reference": {"source": "committed"}},
            "mix33": {"att_horizon_mean": 322.0, "reference": None},
        }
    }
    with pytest.raises(ValueError, match="behaviour@mix33"):
        label_constructed_references(payload, {})

    label_constructed_references(payload, {"mix33": {"source": "constructed", "rng_seed": 1}})
    assert payload["behaviour_cells"]["mix33"]["reference"]["source"] == "constructed"
    assert payload["behaviour_cells"]["mappo1000"]["reference"]["source"] == "committed"

    payload["behaviour_cells"]["mappo1000"]["reference"] = None
    with pytest.raises(ValueError, match="no reference record at all"):
        label_constructed_references(payload, {"mix33": {"source": "constructed"}})


def test_gate_p1_refuses_a_changed_behaviour_comparison() -> None:
    """The behaviour comparisons are C1's per-tier sentence and are gated with the cells."""
    candidate = phase1_payload()
    candidate["behaviour_comparisons"][0]["mean_difference"] = -0.4187
    with pytest.raises(ValueError, match="behaviour@mappo1000"):
        assert_phase1_reproduces(candidate, phase1_payload())
