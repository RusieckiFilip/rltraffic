"""P7.0 commit 3 -- the gate's statistics, its lane alignment and its branch rule.

Three things here are load-bearing and each is guarded by an executed mutation.

1. **Both statistics reduce to an integer numerator over ``n * m``**, so the
   double-computation asserts ``==`` on Python ints rather than on floats.  That is
   deliberate: a tolerance on a load-bearing quantity is a tolerance on the answer.
2. **Lane features are aligned by LANE ID, never by position.**  Measured from disk
   2026-08-16: ``utils/cityflow_utils.py`` appends ``incoming_lanes`` in roadLinks
   discovery order (``road_0_1_0_1`` first) while ``utils/sumo_utils.py`` stores
   ``sorted(...)``.  A positional comparison would silently compare different lanes in
   the two backends, which is `DEFERRED` 23's failure class one level down.
3. **The branch rule is total and mutually exclusive**, every criterion reports its
   distance to the boundary, and ``signed_distance`` is negative exactly when the
   criterion fires.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from offline import parity, transfer_gate as tg

# ----------------------------------------------------------------------
# Independent recomputation routes, written to share no code with the module
# ----------------------------------------------------------------------


def _brute_force_ks_numerator(x: np.ndarray, y: np.ndarray) -> int:
    """max_v abs(count_x(<=v) * m - count_y(<=v) * n), by explicit counting."""
    n, m = int(x.size), int(y.size)
    best = 0
    for v in sorted(set(int(t) for t in x) | set(int(t) for t in y)):
        cx = sum(1 for t in x if int(t) <= v)
        cy = sum(1 for t in y if int(t) <= v)
        best = max(best, abs(cx * m - cy * n))
    return best


def _brute_force_overlap_numerator(x: np.ndarray, y: np.ndarray) -> int:
    """sum_v min(count_x(v) * m, count_y(v) * n), by explicit counting."""
    from collections import Counter

    n, m = int(x.size), int(y.size)
    cx, cy = Counter(int(t) for t in x), Counter(int(t) for t in y)
    return sum(min(cx[v] * m, cy[v] * n) for v in set(cx) | set(cy))


# ----------------------------------------------------------------------
# T8 -- the KS statistic
# ----------------------------------------------------------------------


def test_ks_statistic_double_computed_exactly() -> None:
    rng = np.random.default_rng(20260816)
    for _ in range(20):
        x = rng.integers(0, 9, size=int(rng.integers(5, 60))).astype(np.int32)
        y = rng.integers(0, 12, size=int(rng.integers(5, 60))).astype(np.int32)
        numerator, denominator = tg.ks_statistic_exact(x, y)
        assert denominator == x.size * y.size
        assert numerator == _brute_force_ks_numerator(x, y)
        assert tg.ks_statistic(x, y) == numerator / denominator


def test_ks_statistic_is_zero_against_itself_and_one_for_disjoint_supports() -> None:
    x = np.array([0, 1, 2, 3, 3], dtype=np.int32)
    assert tg.ks_statistic(x, x) == 0.0
    assert tg.ks_statistic(np.array([0, 0], dtype=np.int32), np.array([9, 9], dtype=np.int32)) == 1.0


def test_ks_statistic_refuses_a_non_integer_sample() -> None:
    x = np.array([0.5, 1.5], dtype=np.float32)
    with pytest.raises(ValueError, match="integer counts"):
        tg.ks_statistic_exact(x, np.array([1, 2], dtype=np.int32))


def test_ks_statistic_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        tg.ks_statistic_exact(np.array([], dtype=np.int32), np.array([1], dtype=np.int32))


# ----------------------------------------------------------------------
# T9 -- the overlap coefficient
# ----------------------------------------------------------------------


def test_overlap_coefficient_double_computed_exactly() -> None:
    rng = np.random.default_rng(20260817)
    for _ in range(20):
        x = rng.integers(0, 9, size=int(rng.integers(5, 60))).astype(np.int32)
        y = rng.integers(0, 12, size=int(rng.integers(5, 60))).astype(np.int32)
        numerator, denominator = tg.overlap_coefficient_exact(x, y)
        assert denominator == x.size * y.size
        assert numerator == _brute_force_overlap_numerator(x, y)
        assert tg.overlap_coefficient(x, y) == numerator / denominator


def test_overlap_coefficient_endpoints() -> None:
    x = np.array([0, 1, 1, 2], dtype=np.int32)
    assert tg.overlap_coefficient(x, x) == 1.0
    assert tg.overlap_coefficient(x, np.array([7, 8], dtype=np.int32)) == 0.0


def test_overlap_coefficient_is_one_minus_total_variation() -> None:
    """A third, independent identity: OVL == 1 - TV, on the exact integer support."""
    from collections import Counter

    x = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
    y = np.array([0, 1, 1, 1, 3], dtype=np.int32)
    cx, cy = Counter(int(t) for t in x), Counter(int(t) for t in y)
    tv = 0.5 * sum(
        abs(cx[v] / x.size - cy[v] / y.size) for v in set(cx) | set(cy)
    )
    assert tg.overlap_coefficient(x, y) == pytest.approx(1.0 - tv, abs=1e-12)


# ----------------------------------------------------------------------
# T10 -- rho
# ----------------------------------------------------------------------


def test_rho_anchors_are_exact() -> None:
    fixed, maxp = 250.0, 175.0
    assert tg.rho(fixed, fixed, maxp) == 0.0
    assert tg.rho(fixed, maxp, maxp) == 1.0


def test_rho_double_computed_from_raw_arrays() -> None:
    """The independent route averages the raw horizon values itself."""
    att_fixed = np.array([250.0, 252.0, 248.0], dtype=np.float64)
    att_maxp = np.array([175.0, 176.0, 174.0], dtype=np.float64)
    att_rand = np.array([410.0, 400.0, 420.0], dtype=np.float64)

    f, m, r = float(att_fixed.mean()), float(att_maxp.mean()), float(att_rand.mean())
    expected = (f - r) / (f - m)
    assert tg.rho(f, r, m) == expected
    assert expected < 0.0  # random is worse than the fixed-time anchor


def test_rho_refuses_a_degenerate_denominator() -> None:
    with pytest.raises(ValueError, match="anchor span is zero"):
        tg.rho(200.0, 150.0, 200.0)


# ----------------------------------------------------------------------
# T11 -- lane alignment is BY ID, never positional
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeEpisode:
    lane_ids: tuple[str, ...]
    lane_vehicle_count: np.ndarray
    lane_waiting_vehicle_count: np.ndarray


def _permuted_episode(base: _FakeEpisode, order: list[int]) -> _FakeEpisode:
    return _FakeEpisode(
        lane_ids=tuple(base.lane_ids[i] for i in order),
        lane_vehicle_count=base.lane_vehicle_count[:, order],
        lane_waiting_vehicle_count=base.lane_waiting_vehicle_count[:, order],
    )


def _fixture_pair() -> tuple[_FakeEpisode, _FakeEpisode, list[str]]:
    lane_ids = ("road_a_0", "road_a_1", "road_b_0")
    counts = np.array([[1, 20, 300], [2, 30, 400], [3, 40, 500]], dtype=np.int32)
    waiting = counts * 2
    base = _FakeEpisode(lane_ids, counts, waiting)
    # The SUMO side sorts its lanes; here the permutation is deliberately non-trivial
    # so a positional read cannot accidentally agree.
    permuted = _permuted_episode(base, [2, 0, 1])
    return base, permuted, list(lane_ids)


def test_lane_samples_are_aligned_by_id_across_a_permutation() -> None:
    base, permuted, lane_ids = _fixture_pair()
    left = tg.lane_feature_samples([base], "lane_vehicle_count", lane_ids)
    right = tg.lane_feature_samples([permuted], "lane_vehicle_count", lane_ids)
    assert set(left) == set(right) == set(lane_ids)
    for lane_id in lane_ids:
        assert np.array_equal(left[lane_id], right[lane_id]), lane_id


def test_the_permutation_fixture_is_not_vacuous() -> None:
    """If the two column orders agreed, the alignment test above would prove nothing."""
    base, permuted, _ = _fixture_pair()
    assert base.lane_ids != permuted.lane_ids
    assert not np.array_equal(base.lane_vehicle_count, permuted.lane_vehicle_count)


def test_compare_lane_features_over_a_permutation_is_a_perfect_match() -> None:
    base, permuted, lane_ids = _fixture_pair()
    rows = tg.compare_lane_features([base], [permuted], lane_ids)
    assert len(rows) == len(lane_ids) * len(tg.LANE_ARRAYS)
    assert {row.feature for row in rows} == {
        f"{array}@{lane_id}" for array in tg.LANE_ARRAYS for lane_id in lane_ids
    }
    for row in rows:
        assert row.ks_statistic == 0.0, row.feature
        assert row.overlap_coefficient == 1.0, row.feature


def test_lane_samples_refuse_a_lane_absent_from_one_backend() -> None:
    base, _, lane_ids = _fixture_pair()
    with pytest.raises(KeyError, match="not present in this episode"):
        tg.lane_feature_samples([base], "lane_vehicle_count", lane_ids + ["road_z_9"])


def test_lane_samples_refuse_a_mid_run_lane_set_change() -> None:
    base, permuted, lane_ids = _fixture_pair()
    shrunk = _FakeEpisode(
        lane_ids=base.lane_ids[:2],
        lane_vehicle_count=base.lane_vehicle_count[:, :2],
        lane_waiting_vehicle_count=base.lane_waiting_vehicle_count[:, :2],
    )
    with pytest.raises(KeyError, match="not present in this episode"):
        tg.lane_feature_samples([base, shrunk], "lane_vehicle_count", lane_ids)


# ----------------------------------------------------------------------
# T12 / T15 -- the branch rule and its margins
# ----------------------------------------------------------------------


def _features(ks: list[float], ovl: list[float]) -> list[tg.FeatureComparison]:
    assert len(ks) == len(ovl)
    return [
        tg.FeatureComparison(
            feature=f"lane_vehicle_count@lane_{i}",
            array="lane_vehicle_count",
            lane_id=f"lane_{i}",
            n_cityflow=100,
            n_sumo=100,
            ks_statistic=k,
            overlap_coefficient=o,
            mean_cityflow=1.0,
            mean_sumo=1.0,
        )
        for i, (k, o) in enumerate(zip(ks, ovl))
    ]


_HEALTHY = dict(
    features=_features([0.10] * 16, [0.90] * 16),
    delta_cityflow=70.0,
    delta_sumo=60.0,
    rho_cityflow_random=-2.0,
    rho_sumo_random=-2.2,
)


def test_all_a_criteria_holding_gives_branch_a() -> None:
    verdict = tg.evaluate_branch(**_HEALTHY)
    assert verdict.branch == "A"
    assert verdict.firing_criteria == ()
    assert verdict.failed_a_criteria == ()


@pytest.mark.parametrize(
    "criterion, override",
    [
        ("B1", dict(delta_sumo=-1.0)),
        ("B2", dict(features=_features([0.10] * 16, [0.29] + [0.90] * 15))),
        ("B3", dict(features=_features([0.60] * 9 + [0.10] * 7, [0.90] * 16))),
        ("B4", dict(rho_sumo_random=+1.0)),
    ],
)
def test_each_b_criterion_fires_alone(criterion: str, override: dict) -> None:
    verdict = tg.evaluate_branch(**{**_HEALTHY, **override})
    assert verdict.branch == "B"
    assert verdict.firing_criteria == (criterion,), verdict.firing_criteria


def test_b4_fires_on_magnitude_as_well_as_on_sign() -> None:
    verdict = tg.evaluate_branch(**{**_HEALTHY, "rho_sumo_random": -9.0})
    assert verdict.branch == "B"
    assert verdict.firing_criteria == ("B4",)


@pytest.mark.parametrize(
    "override",
    [
        dict(features=_features([0.10] * 16, [0.40] + [0.90] * 15)),  # A2 gap
        dict(features=_features([0.60] * 6 + [0.10] * 10, [0.90] * 16)),  # A3 count gap
        dict(features=_features([0.80] + [0.10] * 15, [0.90] * 16)),  # A3 max-D gap
        dict(rho_sumo_random=-3.5),  # A4 gap
    ],
)
def test_the_gaps_between_a_and_b_give_branch_c(override: dict) -> None:
    verdict = tg.evaluate_branch(**{**_HEALTHY, **override})
    assert verdict.branch == "C", verdict.firing_criteria


def test_every_criterion_reports_a_row_and_the_rule_is_total() -> None:
    verdict = tg.evaluate_branch(**_HEALTHY)
    assert tuple(row.criterion for row in verdict.rows) == (
        "B1", "B2", "B3", "B4", "A1", "A2", "A3max", "A3count", "A4",
    )
    assert verdict.branch in {"A", "B", "C"}


_SWEEP = [
    _HEALTHY,
    {**_HEALTHY, "delta_sumo": -1.0},
    {**_HEALTHY, "features": _features([0.10] * 16, [0.05] + [0.90] * 15)},
    {**_HEALTHY, "features": _features([0.95] * 16, [0.90] * 16)},
    {**_HEALTHY, "rho_sumo_random": -9.0},
    {**_HEALTHY, "rho_sumo_random": -3.5},
    {**_HEALTHY, "features": _features([0.60] * 6 + [0.10] * 10, [0.45] * 16)},
]


def test_fired_is_recomputable_from_statistic_threshold_and_comparison() -> None:
    """Independent recomputation of ``fired`` from the row's own declared parts.

    Stronger than a sign invariant and free of the boundary caveat below: ``fired``
    must equal the negation of the row's declared comparison, evaluated here by
    ``operator`` rather than by the module.
    """
    import operator

    ops = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
    checked = 0
    for case in _SWEEP:
        for row in tg.evaluate_branch(**case).rows:
            if row.statistic is None or row.threshold is None:
                continue
            satisfied = ops[row.comparison](row.statistic, row.threshold)
            assert row.fired is (not satisfied), (
                f"{row.criterion}: statistic={row.statistic} "
                f"{row.comparison} threshold={row.threshold} but fired={row.fired}"
            )
            checked += 1
    assert checked >= 50, f"the sweep only exercised {checked} margin rows"


def test_signed_distance_sign_agrees_with_fired_away_from_the_boundary() -> None:
    """The margin's orientation.

    ⚠️ Stated as an implication rather than an equivalence, and that is a real
    limitation rather than a hedge: two of the declared criteria (B1 ``delta <= 0``
    and B3 ``count >= 9``) fire ON their boundary, so at a statistic exactly equal to
    its threshold ``signed_distance`` is 0.0 while ``fired`` is True.  The equivalence
    is therefore false at one point, and the test above -- which recomputes ``fired``
    from the declared comparison -- is what covers that point.
    """
    checked = 0
    for case in _SWEEP:
        for row in tg.evaluate_branch(**case).rows:
            if row.signed_distance is None:
                continue
            if row.signed_distance < 0.0:
                assert row.fired is True, row.criterion
                checked += 1
            elif row.signed_distance > 0.0:
                assert row.fired is False, row.criterion
                checked += 1
    assert checked >= 50, f"the sweep only exercised {checked} margin rows"


def test_the_boundary_firing_criteria_are_exactly_b1_b3_and_a1() -> None:
    """Pins the caveat above so the exempt set cannot silently grow.

    ``A1`` is the complement of ``B1`` (``delta > 0`` against ``delta <= 0``), so it
    inherits the same boundary; three criteria, not two.
    """
    verdict = tg.evaluate_branch(**_HEALTHY)
    boundary_firing = {
        row.criterion for row in verdict.rows if row.comparison in {"<", ">"}
    }
    assert boundary_firing == {"B1", "B3", "A1"}


def test_branch_c_names_the_a_criteria_that_failed() -> None:
    verdict = tg.evaluate_branch(
        **{**_HEALTHY, "features": _features([0.10] * 16, [0.40] + [0.90] * 15)}
    )
    assert verdict.branch == "C"
    assert verdict.firing_criteria == ()
    assert verdict.failed_a_criteria == ("A2",)


def test_b4_sign_flip_reports_no_continuous_margin() -> None:
    """Inventing a distance for a boolean would be worse than reporting none."""
    verdict = tg.evaluate_branch(**{**_HEALTHY, "rho_sumo_random": +1.0})
    row = next(r for r in verdict.rows if r.criterion == "B4")
    assert row.fired is True
    assert row.signed_distance is None
    assert "sign" in row.detail


def test_nearest_non_firing_is_the_smallest_relative_distance() -> None:
    verdict = tg.evaluate_branch(**_HEALTHY)
    assert verdict.firing_criteria == ()
    candidates = [
        row for row in verdict.rows
        if not row.fired and row.relative_distance is not None
    ]
    assert candidates
    expected = min(candidates, key=lambda row: abs(float(row.relative_distance)))
    assert verdict.nearest_non_firing == expected.criterion


def test_margins_are_reported_even_for_criteria_that_did_not_fire() -> None:
    verdict = tg.evaluate_branch(**{**_HEALTHY, "delta_sumo": -1.0})
    assert verdict.branch == "B"
    b2 = next(r for r in verdict.rows if r.criterion == "B2")
    assert b2.fired is False
    assert b2.signed_distance == pytest.approx(0.90 - tg.OVL_PATHOLOGICAL, abs=1e-12)


def test_evaluate_branch_refuses_an_empty_feature_table() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        tg.evaluate_branch(**{**_HEALTHY, "features": []})


# ----------------------------------------------------------------------
# T13 / T14 -- DEFERRED 18 on SUMO
# ----------------------------------------------------------------------


def _sumo_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def test_metric_set_independence_refuses_identical_metric_sets() -> None:
    """The vacuity control: comparing an env with itself proves nothing."""
    with pytest.raises(ValueError, match="metric sets are identical"):
        tg.metric_set_independence(
            parity.DECLARED_PARITY_SUMOCFG,
            steps=2,
            seed=1,
            metrics_a=["average_travel_time"],
            metrics_b=["average_travel_time"],
        )


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
def test_deferred_18_metric_set_independence_on_sumo() -> None:
    """`DEFERRED` 18, with both of its controls.

    Short rollout here; the reported measurement is the full 360-step one in the
    committed artifact.
    """
    record = tg.metric_set_independence(
        parity.DECLARED_PARITY_SUMOCFG, steps=60, seed=1000
    )

    # Vacuity control: the two envs really do request different metric sets.
    assert record["metrics_a"] != record["metrics_b"]
    assert len(record["metrics_a"]) == 1
    assert len(record["metrics_b"]) == 3

    # The structural half: on SUMO the metric can never be absent.
    assert record["att_always_requested"] is True

    # The empirical half.
    assert record["identical"] is True
    assert record["n_rows"] == 61

    # Positive control: the comparison is capable of returning False.
    assert record["control_identical"] is False


# ----------------------------------------------------------------------
# Structural facts the gate's validity rests on, pinned so they cannot drift
# ----------------------------------------------------------------------


def test_criterion_scales_are_declared_for_every_criterion() -> None:
    verdict = tg.evaluate_branch(**_HEALTHY)
    for row in verdict.rows:
        assert row.criterion in tg.CRITERION_SCALES, row.criterion
        assert len(tg.CRITERION_SCALES[row.criterion]) >= 3


def test_declared_thresholds_match_the_registered_plan() -> None:
    """A drift here would silently move a registered decision rule."""
    assert (tg.OVL_PATHOLOGICAL, tg.OVL_COMPARABLE) == (0.30, 0.50)
    assert (tg.KS_LARGE, tg.KS_MAX_COMPARABLE) == (0.50, 0.70)
    assert (tg.KS_LARGE_COUNT_PATHOLOGICAL, tg.KS_LARGE_COUNT_COMPARABLE) == (9, 4)
    assert (tg.RHO_COMPARABLE_FACTOR, tg.RHO_PATHOLOGICAL_FACTOR) == (0.5, 2.0)
    plan = Path(parity.REPO_ROOT / "docs" / "plans" / "p7.0.md").read_text(
        encoding="utf-8"
    )
    for token in ("0.30", "0.50", "0.70", "2.0 · M", "0.5 · M"):
        assert token in plan, token


# ----------------------------------------------------------------------
# The lane-convention diagnostic -- added after the registered run, selects nothing
# ----------------------------------------------------------------------


def test_cityflow_and_sumo_disagree_on_which_lane_index_turns_left() -> None:
    """The structural fact the diagnostic rests on, read from the scenario files.

    Neither side of this is an outcome: both are static attributes shipped with the
    scenario.  It is asserted here so that if a future scenario ships a consistent
    numbering, this test goes red and the diagnostic stops being needed rather than
    silently continuing to "correct" something that is already correct.
    """
    cf = tg.cityflow_lane_turns(
        parity.DECLARED_SCENARIO_DIR / "roadnet.json", "intersection_1_1"
    )
    su = tg.sumo_lane_turns(parity.DECLARED_SOURCE_NET)
    for road in ("road_0_1_0", "road_1_0_1", "road_2_1_2", "road_1_2_3"):
        assert cf[f"{road}_0"] == frozenset({"l"}), road
        assert cf[f"{road}_1"] == frozenset({"s"}), road
        assert su[f"{road}_0"] == frozenset({"s"}), road
        assert su[f"{road}_1"] == frozenset({"l"}), road


def test_semantic_correspondence_reverses_every_incoming_lane_on_this_scenario() -> None:
    cf = tg.cityflow_lane_turns(
        parity.DECLARED_SCENARIO_DIR / "roadnet.json", "intersection_1_1"
    )
    su = tg.sumo_lane_turns(parity.DECLARED_SOURCE_NET)
    lane_ids = sorted(cf)
    mapping = tg.lane_semantic_correspondence(cf, su, lane_ids)
    assert len(mapping) == 8
    assert tg.lane_semantic_correspondence(cf, su, lane_ids) == {
        lane: f"{lane.rsplit('_', 1)[0]}_{1 - int(lane.rsplit('_', 1)[1])}"
        for lane in lane_ids
    }
    assert not any(k == v for k, v in mapping.items())


def test_semantic_correspondence_is_the_identity_when_conventions_agree() -> None:
    """The diagnostic must not invent a permutation where none is needed."""
    cf = {"r_0": frozenset({"s"}), "r_1": frozenset({"l"})}
    su = {"r_0": frozenset({"s"}), "r_1": frozenset({"l"})}
    assert tg.lane_semantic_correspondence(cf, su, ["r_0", "r_1"]) == {
        "r_0": "r_0",
        "r_1": "r_1",
    }


def test_semantic_correspondence_refuses_a_genuine_topology_difference() -> None:
    cf = {"r_0": frozenset({"s"}), "r_1": frozenset({"l"})}
    su = {"r_0": frozenset({"s"}), "r_1": frozenset({"s"})}
    with pytest.raises(ValueError, match="disagree on topology"):
        tg.lane_semantic_correspondence(cf, su, ["r_0", "r_1"])


def test_road_level_samples_are_invariant_to_the_lane_permutation() -> None:
    base, permuted, lane_ids = _fixture_pair()
    left = tg.road_level_samples([base], "lane_vehicle_count", lane_ids)
    right = tg.road_level_samples([permuted], "lane_vehicle_count", lane_ids)
    assert set(left) == {"road_a", "road_b"}
    for road in left:
        assert np.array_equal(left[road], right[road]), road


def test_road_level_summation_is_double_computed() -> None:
    base, _, lane_ids = _fixture_pair()
    rows = tg.road_level_samples([base], "lane_vehicle_count", lane_ids)
    per_lane = tg.lane_feature_samples([base], "lane_vehicle_count", lane_ids)
    assert np.array_equal(rows["road_a"], per_lane["road_a_0"] + per_lane["road_a_1"])
    assert np.array_equal(rows["road_b"], per_lane["road_b_0"])


# ----------------------------------------------------------------------
# The 2026-08-16 re-registration: movement-paired alignment, and the VOID label
# ----------------------------------------------------------------------


def test_compare_lane_features_honours_a_non_identity_correspondence() -> None:
    """The registered alignment must actually re-pair the columns.

    ``base`` and ``permuted`` hold the same three columns in different orders; here
    the correspondence deliberately points each CityFlow lane at a DIFFERENT lane, so
    a comparison that quietly ignored it would return the perfect match that the
    identity produces and this test would not fire.
    """
    base, permuted, lane_ids = _fixture_pair()
    swap = {"road_a_0": "road_a_1", "road_a_1": "road_a_0", "road_b_0": "road_b_0"}
    rows = {r.feature: r for r in tg.compare_lane_features([base], [permuted], lane_ids, swap)}

    identity = {
        r.feature: r for r in tg.compare_lane_features([base], [permuted], lane_ids)
    }
    assert identity["lane_vehicle_count@road_a_0"].ks_statistic == 0.0
    assert rows["lane_vehicle_count@road_a_0"].ks_statistic == 1.0
    assert rows["lane_vehicle_count@road_a_0"].sumo_lane == "road_a_1"
    assert rows["lane_vehicle_count@road_b_0"].ks_statistic == 0.0


def test_compare_lane_features_refuses_an_incomplete_correspondence() -> None:
    base, permuted, lane_ids = _fixture_pair()
    with pytest.raises(KeyError, match="does not cover"):
        tg.compare_lane_features([base], [permuted], lane_ids, {"road_a_0": "road_a_0"})


def test_the_void_alignment_is_labelled_void_and_yields_no_branch() -> None:
    standing = tg.ALIGNMENT_STANDING[tg.ALIGNMENT_BY_LANE_ID]
    assert "VOID" in standing
    assert "not B, and not C either" in standing
    assert "REGISTERED" in tg.ALIGNMENT_STANDING[tg.ALIGNMENT_BY_MOVEMENT]
    assert "DIAGNOSTIC" in tg.ALIGNMENT_STANDING[tg.ALIGNMENT_ROAD_LEVEL]


def test_non_firing_ranked_is_ascending_and_covers_every_non_firing_row() -> None:
    verdict = tg.evaluate_branch(**_HEALTHY)
    names = [name for name, _ in verdict.non_firing_ranked]
    values = [value for _, value in verdict.non_firing_ranked]
    assert values == sorted(values)
    expected = {
        r.criterion for r in verdict.rows
        if not r.fired and r.relative_distance is not None
    }
    assert set(names) == expected
    assert verdict.nearest_non_firing == names[0]


def test_a_near_tie_between_two_criteria_is_reported_rather_than_broken() -> None:
    """B2 and A3max coincide when one feature is extremal in both statistics.

    Constructed here rather than taken from the run: a single feature with
    ``OVL == 1 - KS`` makes B2's margin ``ovl - 0.30`` and A3max's ``0.70 - ks`` the
    same number, because the thresholds also sum to 1.0.  A lone
    ``nearest_non_firing`` would hide one of them behind a float tie-break.
    """
    verdict = tg.evaluate_branch(
        **{**_HEALTHY, "features": _features([0.66] + [0.10] * 15, [0.34] + [0.90] * 15)}
    )
    ranked = dict(verdict.non_firing_ranked)
    assert "B2" in ranked and "A3max" in ranked
    assert ranked["B2"] == pytest.approx(ranked["A3max"], abs=1e-12)
    assert ranked["B2"] == pytest.approx(0.04, abs=1e-12)


# ----------------------------------------------------------------------
# Section 5.4 -- the green-phase lane-set correspondence table
# ----------------------------------------------------------------------


class _FakeIx:
    def __init__(self, num_phases, durations, mapping, roadlink_lanes):
        self.num_phases = num_phases
        self.phase_durations = durations
        self.phase_roadlink_mapping = mapping
        self.roadlink_lanes = roadlink_lanes


def test_green_action_lane_sets_skips_clearance_phases() -> None:
    """Greens are the phases longer than the clearance bound, in ascending order."""
    ix = _FakeIx(
        num_phases=4,
        durations=[5.0, 30.0, 5.0, 30.0],
        mapping=[[], [0], [], [1]],
        roadlink_lanes=[(["r_1"], ["out"]), (["r_0"], ["out"])],
    )
    rows = tg.green_action_lane_sets(ix)
    assert [r["action"] for r in rows] == [0, 1]
    assert [r["file_phase"] for r in rows] == [1, 3]
    assert rows[0]["released_incoming_lanes"] == ["r_1"]
    assert rows[1]["released_incoming_lanes"] == ["r_0"]


def test_green_action_semantics_agree_once_the_lanes_are_translated() -> None:
    """The whole point: raw names disagree, movement-translated names agree."""
    cf = [{"action": 0, "file_phase": 1, "released_incoming_lanes": ["r_1"]}]
    su = [{"action": 0, "file_phase": 0, "released_incoming_lanes": ["r_0"]}]
    correspondence = {"r_0": "r_1", "r_1": "r_0"}

    translated = tg.compare_green_action_semantics(cf, su, correspondence)
    assert translated["all_actions_agree"] is True
    assert translated["rows"][0]["sumo_released_lanes_in_cityflow_names"] == ["r_1"]

    raw = tg.compare_green_action_semantics(cf, su, {"r_0": "r_0", "r_1": "r_1"})
    assert raw["all_actions_agree"] is False


def test_green_action_semantics_flags_a_genuine_disagreement() -> None:
    cf = [{"action": 0, "file_phase": 1, "released_incoming_lanes": ["r_1", "s_1"]}]
    su = [{"action": 0, "file_phase": 0, "released_incoming_lanes": ["r_0"]}]
    out = tg.compare_green_action_semantics(cf, su, {"r_0": "r_1", "r_1": "r_0"})
    assert out["all_actions_agree"] is False
    assert out["n_actions_agreeing"] == 0
