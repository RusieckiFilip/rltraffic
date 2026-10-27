"""A27(b)'s exposure table, from the code constants, against the registration's eight numbers (``BRIEF_46`` 4.3).

No format and no alignment convention: nothing is read from or written to disk.

Exposure is ``row visits = steps x decisions per step / decisions in the training set`` (A27(b)).  The module computes
it from the constants (``tier_sweep.JOINT_BATCH_SIZE``, ``CONTEXT_LENGTH``, ``EXPECTED_NODES``,
``offline_baselines.BC_BATCH_WINDOWS``, ``IQL_BATCH_TRANSITIONS``, ``DECISION_ROWS_PER_STREAM``,
``TOP_RETURN_FRACTION``, the 200-episode tiers); this file recomputes every number from A27's OWN literals -- 200
episodes x 360 decisions x 16 intersections, 64 joint windows of K = 20, 1,280 transitions -- with no code constant in
the arithmetic, and requires exact equality of the fractions as well as A27's one-decimal figures.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import offline.a27_arms as arms

STEPS = 40_000
GRID_SET = 200 * 360 * 16
GRID_DECILE = GRID_SET // 10
HZ_SET = 200 * 360
HZ_DECILE = HZ_SET // 10


def _visits(arm: str, scenario: str) -> Fraction:
    return arms.exposure_table()[scenario]["arms"][arm]["row_visits_at_40000"]


def test_the_training_set_sizes_are_a27bs() -> None:
    table = arms.exposure_table()
    assert (table["grid4x4"]["decisions_in_training_set"], table["grid4x4"]["decisions_in_top_decile"]) == (
        1_152_000, 115_200)
    assert (table["hz1x1"]["decisions_in_training_set"], table["hz1x1"]["decisions_in_top_decile"]) == (72_000, 7_200)


@pytest.mark.parametrize(
    ("arm", "decisions_per_step", "in_set", "a27"),
    [
        ("dt_spatial", 64 * 20 * 16, GRID_SET, 711.1),
        ("dt_nomix", 64 * 20 * 16, GRID_SET, 711.1),
        ("bc", 64 * 20, GRID_SET, 44.4),
        ("iql", 1_280, GRID_SET, 44.4),
        ("bc_top10", 64 * 20, GRID_DECILE, 444.4),
        ("bc_top10_perix", 64 * 20, GRID_DECILE, 444.4),
    ],
)
def test_the_registered_grid_exposure_is_a27bs(arm: str, decisions_per_step: int, in_set: int, a27: float) -> None:
    """``DT 711.1, BC and IQL 44.4, %BC (global and per intersection) 444.4``."""
    exact = Fraction(STEPS * decisions_per_step, in_set)
    assert _visits(arm, "grid4x4") == exact
    assert round(float(exact), 1) == a27


@pytest.mark.parametrize(
    ("arm", "decisions_per_step", "in_set", "a27"),
    [
        ("bc_m", 1_024 * 20, GRID_SET, 711.1),
        ("iql_m", 20_480, GRID_SET, 711.1),
        ("bc_top10_m", 102 * 20, GRID_DECILE, 708.3),
        ("bc_top10_perix_m", 102 * 20, GRID_DECILE, 708.3),
    ],
)
def test_the_matched_grid_exposure_is_a27bs(arm: str, decisions_per_step: int, in_set: int, a27: float) -> None:
    """``bc_m`` and ``iql_m`` at 711.1; the two top-decile arms at 708.3 (102 windows, rounded down, declared)."""
    exact = Fraction(STEPS * decisions_per_step, in_set)
    assert _visits(arm, "grid4x4") == exact
    assert round(float(exact), 1) == a27


def test_the_declared_rounding_is_down_and_no_baseline_is_over_exposed() -> None:
    """A27(b): 102.4 windows is not an integer; 102 gives 708.3 (-0.4 %), 103 would give 715.3 (+0.6 %)."""
    dt = _visits("dt_nomix", "grid4x4")
    assert arms.row_visits(STEPS, 103 * 20, GRID_DECILE) == Fraction(STEPS * 2_060, 115_200)
    assert round(float(arms.row_visits(STEPS, 103 * 20, GRID_DECILE)), 1) == 715.3
    for arm in ("bc_m", "iql_m", "bc_top10_m", "bc_top10_perix_m"):
        assert _visits(arm, "grid4x4") <= dt, arm
    assert _visits("bc_m", "grid4x4") == dt and _visits("iql_m", "grid4x4") == dt


@pytest.mark.parametrize(
    ("arm", "decisions_per_step", "in_set", "a27"),
    [
        ("dt", 64 * 20, HZ_SET, 711.1),
        ("bc", 64 * 20, HZ_SET, 711.1),
        ("iql", 1_280, HZ_SET, 711.1),
        ("bc_top10", 64 * 20, HZ_DECILE, 7_111.1),
    ],
)
def test_the_hz1x1_exposure_is_a27bs(arm: str, decisions_per_step: int, in_set: int, a27: float) -> None:
    """hz1x1's only asymmetry is the top-decile arms', and it runs the other way (stated, not re-run)."""
    exact = Fraction(STEPS * decisions_per_step, in_set)
    assert _visits(arm, "hz1x1") == exact
    assert round(float(exact), 1) == a27


def test_the_target_flag_does_not_change_exposure() -> None:
    assert _visits("iql_t", "grid4x4") == _visits("iql", "grid4x4")
    assert _visits("iql_t", "hz1x1") == _visits("iql", "hz1x1")
