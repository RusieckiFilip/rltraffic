"""Tests for ``offline/admission_probe.py`` -- P8.4a's vehicle-admission measurement.

Three layers, and the split is deliberate:

* **pure arithmetic and refusal** -- always runs, no simulator, no corpus.  The measurement identity
  R1-R5, the registered scoring rules, the exact-reproduction comparison and the
  filesystem-mutation barrier all live here.
* **fake-engine** -- exercises the live-env read path against a stand-in that mimics only the four
  calls the probe makes, so the reconciliation is tested without CityFlow.
* **simulator-backed** -- opt-in, skipping with a reason that names ``RLTRAFFIC_CORPUS_V11``.  These
  are the ones that prove the analytic vehicle count matches a real engine and that a replayed
  episode reproduces its committed ``att_horizon`` exactly.

The alignment convention, the identity and the scoring rules under test are stated in
``offline/admission_probe.py``'s module docstring and registered in ``docs/plans/p8.4a.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from offline.admission_probe import (
    ARTIFACT_FORMAT_VERSION,
    BEHAVIOUR_METHOD,
    E2_ADMISSION_FLOOR,
    E3_SCORED_SCENARIO,
    ESCALATION_DRAWS,
    PROBE_DRAWS,
    PROBE_SCENARIOS,
    PROBE_SEEDS,
    AdmissionEpisode,
    ProbeRoots,
    ReferenceCheck,
    admission_spread,
    assert_no_science_verdict,
    cell_admission_ratio,
    check_against_reference,
    created_from_flow,
    default_protected_roots,
    paired_admission_difference,
    per_seed_admission_ratios,
    probe_episode,
    read_admission_at_horizon,
    reconcile_admission,
    score_e1,
    score_e2,
    score_e3,
    seeds_for,
    summarise_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Fixtures and builders
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected datasets_v11/ directory to run the corpus-backed P8.4a tests"
        )
    return candidate


@pytest.fixture(scope="module")
def draws_root() -> Path:
    """The materialised draw tree; skip if it is absent (it is gitignored)."""
    env_value = os.environ.get("RLTRAFFIC_DRAWS")
    candidate = Path(env_value) if env_value else REPO_ROOT / "scenarios/draws"
    if not candidate.is_dir():
        pytest.skip(
            f"materialised draws not found at {candidate}: set RLTRAFFIC_DRAWS to a "
            "scenarios/draws directory to run the draw-backed P8.4a tests"
        )
    return candidate


@pytest.fixture(scope="module")
def cityflow_available() -> bool:
    """Skip when the CityFlow engine is not importable."""
    try:
        import cityflow  # noqa: F401
    except ImportError:
        pytest.skip("the CityFlow engine is not installed, so no episode can be rolled")
    return True


def flow_entry(start: int, *, end: int | None = None, interval: float = 5.0) -> dict[str, Any]:
    """One CityFlow flow entry with the shape every materialised draw uses."""
    return {
        "vehicle": {"length": 5.0, "maxSpeed": 11.11},
        "route": ["road_a", "road_b"],
        "interval": interval,
        "startTime": start,
        "endTime": start if end is None else end,
    }


def write_flow(tmp_path: Path, entries: list[dict[str, Any]], name: str = "flow.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def episode(
    *,
    scenario: str = "hz1x1",
    tier: str = "random",
    method: str = "bc",
    seed: int | None = 101,
    draw_id: int = 1000,
    created: int = 1000,
    entered: int = 1000,
    att_ours: float = 100.0,
    att_engine: float = 110.0,
    seconds: float = 0.7,
) -> AdmissionEpisode:
    """One episode record, with the counts consistent by construction."""
    never = created - entered
    return AdmissionEpisode(
        scenario=scenario,
        tier=tier,
        method=method,
        arm=f"{method}@{tier}",
        seed=seed,
        draw_id=draw_id,
        created=created,
        entered=entered,
        never_entered=never,
        entered_fraction=entered / created,
        completed_at_horizon=entered - 10,
        running_at_horizon=10,
        waiting_at_horizon=never,
        att_ours=att_ours,
        att_engine=att_engine,
        horizon_vehicle_count=10.0,
        episode_reward=-1.0,
        seconds=seconds,
    )


def cell(
    *,
    scenario: str = "hz1x1",
    tier: str = "random",
    method: str = "bc",
    per_seed_entered: dict[int | None, int],
    created: int = 1000,
    draws: tuple[int, ...] = (1000, 1001),
) -> Any:
    """A :class:`CellSummary` built from episodes, so the summary path is exercised too."""
    episodes = [
        episode(
            scenario=scenario,
            tier=tier,
            method=method,
            seed=seed,
            draw_id=draw,
            created=created,
            entered=entered,
        )
        for seed, entered in per_seed_entered.items()
        for draw in draws
    ]
    return summarise_cell(episodes)


class FakeMetrics:
    """Stands in for ``CityFlowMetrics``: only ``_episode`` is read by the probe."""

    def __init__(self, depart_time: int, completed: int) -> None:
        self._episode = {
            "depart_time": {f"v{i}": 0.0 for i in range(depart_time)},
            "completed": [{"vid": f"v{i}"} for i in range(completed)],
        }


class FakeEngine:
    """Stands in for ``cityflow.Engine``: only the three calls the probe makes exist."""

    def __init__(self, running: int, waiting: int, att_engine: float) -> None:
        self._running = running
        self._waiting = waiting
        self._att = att_engine
        self.calls: list[bool] = []

    def get_vehicles(self, include_waiting: bool = False) -> list[str]:
        self.calls.append(include_waiting)
        count = self._running + (self._waiting if include_waiting else 0)
        return [f"v{i}" for i in range(count)]

    def get_average_travel_time(self) -> float:
        return self._att


class FakeEnv:
    """An env exposing exactly the two private attributes the probe reads."""

    def __init__(self, engine: FakeEngine, metrics: FakeMetrics) -> None:
        self._eng = engine
        self._metrics = metrics


# ----------------------------------------------------------------------
# R5 -- created, from the flow file
# ----------------------------------------------------------------------


def test_created_from_flow_counts_only_entries_that_fire(tmp_path: Path) -> None:
    """``Flow::nextStep`` sees ``currentTime`` in ``0 .. horizon-1``, so ``startTime == horizon`` never fires.

    ``CityFlow/src/flow/flow.cpp:6-22``: the engine calls ``nextStep`` once per simulation step, and
    an entry emits when ``currentTime >= startTime``.  Over a 20-step episode ``currentTime`` takes
    the values 0..19, so entries at 0, 5 and 19 fire and entries at 20 and 21 do not.
    """
    path = write_flow(
        tmp_path,
        [flow_entry(0), flow_entry(5), flow_entry(19), flow_entry(20), flow_entry(21)],
    )
    assert created_from_flow(path, horizon_seconds=20) == 3


def test_created_from_flow_is_the_real_draws_count_and_not_the_entry_count(tmp_path: Path) -> None:
    """The eight entries past the horizon are exactly the trap ``provenance.json`` would spring.

    ``scenarios/draws/cityflow1x1/draw_1000`` records ``n_vehicles: 1821`` while its largest
    ``startTime`` is 3658; a 3600 s episode creates 1813.  Reconstructed here at small scale so the
    off-by-N is pinned without needing the draw tree.
    """
    entries = [flow_entry(t) for t in range(0, 3600, 100)] + [flow_entry(3600), flow_entry(3658)]
    path = write_flow(tmp_path, entries)
    assert created_from_flow(path, horizon_seconds=3600) == 36


def test_created_from_flow_refuses_a_multi_vehicle_entry(tmp_path: Path) -> None:
    """R5 is exact only because every entry is a single vehicle; that precondition is asserted.

    An entry with ``startTime < endTime`` emits ``floor((end-start)/interval)+1`` vehicles, which
    this count does not model.  Silently counting it as one is the failure mode.
    """
    path = write_flow(tmp_path, [flow_entry(0, end=100, interval=5.0)])
    with pytest.raises(ValueError, match="startTime"):
        created_from_flow(path, horizon_seconds=3600)


def test_created_from_flow_refuses_a_non_positive_horizon(tmp_path: Path) -> None:
    path = write_flow(tmp_path, [flow_entry(0)])
    with pytest.raises(ValueError, match="horizon_seconds"):
        created_from_flow(path, horizon_seconds=0)


# ----------------------------------------------------------------------
# R1-R4 -- the reconciliation
# ----------------------------------------------------------------------


def test_reconcile_admission_agrees_on_a_fully_admitted_episode() -> None:
    """The hz1x1 maxpressure episode measured during exploration, reconstructed from its counts."""
    counts = reconcile_admission(
        n_running=103,
        n_with_waiting=103,
        n_depart_time=1813,
        n_completed=1710,
        created_from_flow=1813,
    )
    assert counts.created == 1813
    assert counts.entered == 1813
    assert counts.never_entered == 0
    assert counts.entered_fraction == 1.0


def test_reconcile_admission_agrees_on_a_censored_episode() -> None:
    """The hz1x1 random episode: 615 vehicles created and never admitted."""
    counts = reconcile_admission(
        n_running=153,
        n_with_waiting=768,
        n_depart_time=1198,
        n_completed=1045,
        created_from_flow=1813,
    )
    assert counts.created == 1813
    assert counts.entered == 1198
    assert counts.never_entered == 615
    assert counts.entered_fraction == 1198 / 1813


def test_reconcile_admission_refuses_when_the_two_entered_routes_disagree() -> None:
    """R2 against R3: the metric's ``depart_time`` set must equal ``completed + running``."""
    with pytest.raises(ValueError, match="entered"):
        reconcile_admission(
            n_running=103,
            n_with_waiting=103,
            n_depart_time=1813,
            n_completed=1709,
            created_from_flow=1813,
        )


def test_reconcile_admission_refuses_when_the_two_created_routes_disagree() -> None:
    """R4 against R5: the engine's pool arithmetic must equal the flow file's count."""
    with pytest.raises(ValueError, match="created"):
        reconcile_admission(
            n_running=103,
            n_with_waiting=103,
            n_depart_time=1813,
            n_completed=1710,
            created_from_flow=1821,
        )


def test_reconcile_admission_refuses_a_negative_never_entered() -> None:
    """``include_waiting=True`` is a superset of ``include_waiting=False``; a deficit is a defect."""
    with pytest.raises(ValueError, match="never_entered"):
        reconcile_admission(
            n_running=103,
            n_with_waiting=100,
            n_depart_time=1813,
            n_completed=1710,
            created_from_flow=1810,
        )


def test_read_admission_at_horizon_makes_both_engine_calls_and_reconciles() -> None:
    """The live read path: both ``include_waiting`` values are asked for, and R1-R5 are checked."""
    engine = FakeEngine(running=153, waiting=615, att_engine=781.88)
    env = FakeEnv(engine, FakeMetrics(depart_time=1198, completed=1045))
    counts = read_admission_at_horizon(env, created=1813)
    assert counts.never_entered == 615
    assert counts.entered == 1198
    assert sorted(engine.calls) == [False, True]


def test_read_admission_at_horizon_refuses_an_env_without_an_engine() -> None:
    """A non-CityFlow env cannot be probed, and must say so rather than return zeros."""
    class Bare:
        pass

    with pytest.raises(TypeError, match="CityFlow"):
        read_admission_at_horizon(Bare(), created=10)


# ----------------------------------------------------------------------
# The registered scoring rules
# ----------------------------------------------------------------------


def test_cell_admission_ratio_is_a_population_ratio_not_a_mean_of_ratios() -> None:
    """The two aggregations differ whenever draws differ in size, and the registered one is pooled."""
    episodes = [
        episode(created=100, entered=100, draw_id=1000),
        episode(created=1000, entered=500, draw_id=1001),
    ]
    assert cell_admission_ratio(episodes) == 600 / 1100
    mean_of_ratios = (1.0 + 0.5) / 2
    assert cell_admission_ratio(episodes) != mean_of_ratios


def test_per_seed_admission_ratios_are_keyed_by_seed_and_sorted() -> None:
    episodes = [
        episode(seed=202, created=100, entered=90, draw_id=1000),
        episode(seed=101, created=100, entered=80, draw_id=1000),
        episode(seed=101, created=100, entered=100, draw_id=1001),
    ]
    ratios = per_seed_admission_ratios(episodes)
    assert list(ratios) == ["101", "202"]
    assert ratios["101"] == 180 / 200
    assert ratios["202"] == 0.9


def test_admission_spread_is_zero_for_a_single_seeded_arm() -> None:
    assert admission_spread({"None": 0.91}) == 0.0
    assert admission_spread({"101": 0.90, "202": 0.95}) == pytest.approx(0.05)


def test_score_e1_holds_when_the_arm_admits_at_least_as_many() -> None:
    """``deficit <= 0`` holds and does not escalate: exact equality is a result, not closeness."""
    cells = {
        "hz1x1": {
            "behaviour@random": cell(method=BEHAVIOUR_METHOD, per_seed_entered={101: 600, 202: 600}),
            "bc@random": cell(method="bc", per_seed_entered={101: 700, 202: 700}),
        }
    }
    scored = score_e1(cells)
    entry = next(e for e in scored["arms"] if e["arm"] == "bc@random")
    assert entry["status"] == "holds"
    assert entry["escalate"] is False
    assert scored["n_falsified"] == 0


def test_score_e1_escalates_a_deficit_inside_delta_and_calls_it_close() -> None:
    """``0 < deficit <= Delta`` is close, and closeness escalates -- it is not a result at n=10."""
    cells = {
        "hz1x1": {
            "behaviour@random": cell(method=BEHAVIOUR_METHOD, per_seed_entered={101: 600, 202: 700}),
            "bc@random": cell(method="bc", per_seed_entered={101: 640, 202: 640}),
        }
    }
    scored = score_e1(cells)
    entry = next(e for e in scored["arms"] if e["arm"] == "bc@random")
    assert entry["deficit"] == pytest.approx(0.02)
    assert entry["delta"] == pytest.approx(0.10)
    assert entry["status"] == "close"
    assert entry["escalate"] is True


def test_score_e1_falsifies_a_deficit_beyond_delta() -> None:
    cells = {
        "hz1x1": {
            "behaviour@random": cell(method=BEHAVIOUR_METHOD, per_seed_entered={101: 900, 202: 900}),
            "bc@random": cell(method="bc", per_seed_entered={101: 500, 202: 500}),
        }
    }
    scored = score_e1(cells)
    entry = next(e for e in scored["arms"] if e["arm"] == "bc@random")
    assert entry["deficit"] == pytest.approx(0.40)
    assert entry["delta"] == 0.0
    assert entry["status"] == "falsified"
    assert entry["escalate"] is True
    assert scored["n_falsified"] == 1


def test_score_e1_does_not_escalate_an_exactly_equal_pair() -> None:
    """Both at full admission: ``deficit == 0`` is not closeness and must not cost 100 draws."""
    cells = {
        "grid4x4": {
            "behaviour@random": cell(
                scenario="grid4x4", method=BEHAVIOUR_METHOD, per_seed_entered={101: 1000, 202: 1000}
            ),
            "iql@random": cell(
                scenario="grid4x4", method="iql", per_seed_entered={101: 1000, 202: 1000}
            ),
        }
    }
    scored = score_e1(cells)
    entry = next(e for e in scored["arms"] if e["arm"] == "iql@random")
    assert entry["deficit"] == 0.0
    assert entry["status"] == "holds"
    assert entry["escalate"] is False


def test_score_e1_refuses_a_tier_with_no_behaviour_anchor() -> None:
    """An arm with nothing to compare against is a wiring defect, not a passing cell."""
    cells = {"hz1x1": {"bc@random": cell(method="bc", per_seed_entered={101: 700})}}
    with pytest.raises(ValueError, match="behaviour"):
        score_e1(cells)


def test_score_e2_uses_the_registered_floor_and_names_every_arm_below_it() -> None:
    cells = {
        "hz1x1": {
            "behaviour@mappo1000": cell(
                tier="mappo1000", method=BEHAVIOUR_METHOD, per_seed_entered={101: 1000}
            ),
            "bc@mappo1000": cell(tier="mappo1000", method="bc", per_seed_entered={101: 985}),
            "bc@random": cell(tier="random", method="bc", per_seed_entered={101: 500}),
        }
    }
    scored = score_e2(cells)
    assert scored["floor"] == E2_ADMISSION_FLOOR
    below = [e["arm"] for e in scored["arms"] if not e["passes"]]
    assert below == ["bc@mappo1000"]
    assert all(e["tier"] == "mappo1000" for e in scored["arms"])


def test_score_e3_is_scored_on_hz1x1_and_reports_grid4x4_as_its_own_row() -> None:
    """Amendment A4: grid4x4's profile is a MEASUREMENT with its own row, never an exclusion."""
    hz = {
        f"behaviour@{tier}": cell(tier=tier, method=BEHAVIOUR_METHOD, per_seed_entered={101: n})
        for tier, n in (
            ("mappo1000", 1000),
            ("mappo500", 990),
            ("maxpressure", 950),
            ("fixedtime", 900),
            ("random", 650),
        )
    }
    grid = {
        f"behaviour@{tier}": cell(
            scenario="grid4x4", tier=tier, method=BEHAVIOUR_METHOD, per_seed_entered={101: 1000}
        )
        for tier in ("mappo1000", "random")
    }
    scored = score_e3({"hz1x1": hz, "grid4x4": grid})
    assert scored["scored_scenario"] == E3_SCORED_SCENARIO
    assert scored["holds"] is True
    assert scored["monotone"] is True
    assert [row["tier"] for row in scored["profile"]] == [
        "mappo1000",
        "mappo500",
        "maxpressure",
        "fixedtime",
        "random",
    ]
    assert [row["tier"] for row in scored["grid4x4_profile"]] == ["mappo1000", "random"]
    assert all(row["admission_ratio"] == 1.0 for row in scored["grid4x4_profile"])


def test_score_e3_reports_a_non_monotone_hz1x1_profile_as_non_monotone() -> None:
    """The registered pair can hold while the profile is not monotone; both are reported."""
    hz = {
        f"behaviour@{tier}": cell(tier=tier, method=BEHAVIOUR_METHOD, per_seed_entered={101: n})
        for tier, n in (
            ("mappo1000", 1000),
            ("mappo500", 900),
            ("maxpressure", 950),
            ("fixedtime", 940),
            ("random", 650),
        )
    }
    scored = score_e3({"hz1x1": hz, "grid4x4": {}})
    assert scored["holds"] is True
    assert scored["monotone"] is False


def test_paired_admission_difference_pairs_a_single_seeded_anchor_against_every_arm_seed() -> None:
    """``behaviour@maxpressure`` carries ``seed=None``; its one episode per draw pairs with all five."""
    arm = [
        episode(seed=s, draw_id=d, created=1000, entered=900)
        for s in (101, 202)
        for d in (1000, 1001)
    ]
    anchor = [
        episode(method=BEHAVIOUR_METHOD, seed=None, draw_id=d, created=1000, entered=800)
        for d in (1000, 1001)
    ]
    paired = paired_admission_difference(arm, anchor)
    assert paired["n_pairs"] == 4
    assert paired["anchor_is_single_seeded"] is True
    assert paired["mean"] == pytest.approx(0.1)
    assert paired["ci95"] == 0.0


def test_paired_admission_difference_refuses_an_unpairable_episode() -> None:
    """A draw the anchor never ran is not silently dropped from the denominator."""
    arm = [episode(seed=101, draw_id=1002)]
    anchor = [episode(method=BEHAVIOUR_METHOD, seed=101, draw_id=1000)]
    with pytest.raises(ValueError, match="1002"):
        paired_admission_difference(arm, anchor)


# ----------------------------------------------------------------------
# The exact-reproduction check
# ----------------------------------------------------------------------


def test_check_against_reference_accepts_an_exact_match() -> None:
    episodes = [episode(seed=101, draw_id=1000, att_ours=163.43353557639273)]
    check = check_against_reference(
        episodes, {(101, 1000): 163.43353557639273}, "docs/data/p4_6_grid.json"
    )
    assert check.exact is True
    assert check.n_compared == 1
    assert check.mismatches == ()


def test_check_against_reference_rejects_a_one_ulp_difference() -> None:
    """``==`` and never ``isclose``: a faithful replay reproduces the float exactly."""
    import math

    committed = 163.43353557639273
    episodes = [episode(seed=101, draw_id=1000, att_ours=math.nextafter(committed, math.inf))]
    check = check_against_reference(episodes, {(101, 1000): committed}, "reference")
    assert check.exact is False
    assert len(check.mismatches) == 1
    assert check.mismatches[0]["draw_id"] == 1000


def test_check_against_reference_reports_a_missing_reference_rather_than_passing() -> None:
    """An episode with nothing to compare against is counted, never treated as agreement."""
    episodes = [episode(seed=101, draw_id=1000), episode(seed=101, draw_id=1001)]
    check = check_against_reference(episodes, {(101, 1000): 100.0}, "reference")
    assert check.n_compared == 1
    assert check.n_missing == 1
    assert check.exact is False


def test_reference_check_is_not_exact_when_nothing_was_compared() -> None:
    """An empty comparison is not a pass; a cell with no reference must be visible as such."""
    check = ReferenceCheck(source="none", n_compared=0, n_missing=0, mismatches=())
    assert check.exact is False


# ----------------------------------------------------------------------
# The declared inventory
# ----------------------------------------------------------------------


def test_the_probe_arm_lists_are_the_ones_the_campaign_modules_declare() -> None:
    """``BRIEF_31`` wrote both arm lists from memory and got both wrong (Amendment A6).

    So the inventory is checked against the modules that produced the committed cells, not against
    the brief.
    """
    from offline import method_tier_grid, tier_sweep

    assert PROBE_SCENARIOS["hz1x1"].methods == method_tier_grid.METHODS
    assert PROBE_SCENARIOS["grid4x4"].methods == tier_sweep.METHODS
    assert "dt" not in PROBE_SCENARIOS["grid4x4"].methods
    assert set(PROBE_SCENARIOS["hz1x1"].tiers) == {
        t for t in method_tier_grid.TIERS if method_tier_grid.TIERS[t].phase == 1
    }
    assert set(PROBE_SCENARIOS["grid4x4"].tiers) == {"mappo1000", "random"}


def test_the_probe_draws_are_the_first_ten_of_the_registered_held_out_pool() -> None:
    from offline.dataset import DRAW_SPLITS

    low, high = DRAW_SPLITS["heldout"]
    assert PROBE_DRAWS == tuple(range(low, low + 10))
    assert ESCALATION_DRAWS == tuple(range(low, high + 1))


def test_the_probe_seeds_are_the_registered_training_seeds() -> None:
    from offline.dt_gate import TRAINING_SEEDS

    assert PROBE_SEEDS == TRAINING_SEEDS


def test_seeds_for_is_single_slotted_only_for_the_deterministic_anchors() -> None:
    """``behaviour@maxpressure`` and ``behaviour@fixedtime`` carry ``seed=None`` in their cells."""
    assert seeds_for("hz1x1", "maxpressure", BEHAVIOUR_METHOD) == (None,)
    assert seeds_for("hz1x1", "fixedtime", BEHAVIOUR_METHOD) == (None,)
    assert seeds_for("hz1x1", "random", BEHAVIOUR_METHOD) == PROBE_SEEDS
    assert seeds_for("hz1x1", "mappo1000", BEHAVIOUR_METHOD) == PROBE_SEEDS
    assert seeds_for("hz1x1", "maxpressure", "bc") == PROBE_SEEDS
    assert seeds_for("grid4x4", "random", BEHAVIOUR_METHOD) == PROBE_SEEDS


# ----------------------------------------------------------------------
# The artifact
# ----------------------------------------------------------------------


def test_assert_no_science_verdict_rejects_a_conclusion_about_the_headline() -> None:
    """``BRIEF_31`` section 6: this artifact measures admission and concludes nothing about P5.2."""
    with pytest.raises(ValueError, match="artefact"):
        assert_no_science_verdict({"e1": {"reading": "artefact"}})


def test_assert_no_science_verdict_rejects_a_nested_equivalence_verdict() -> None:
    """The repo's existing ban still applies, and a nested one is the one a shallow check misses."""
    with pytest.raises(ValueError, match="verdict"):
        assert_no_science_verdict({"cells": [{"a": {"verdict": "anything"}}]})


def test_assert_no_science_verdict_accepts_the_registered_status_words() -> None:
    """``holds`` / ``close`` / ``falsified`` are E1's registered statuses and must survive the check."""
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "e1": {"arms": [{"status": "falsified"}, {"status": "close"}, {"status": "holds"}]},
    }
    assert_no_science_verdict(payload)


def test_summarise_cell_refuses_a_mixed_arm_input() -> None:
    """One summary describes one cell; mixing two arms is how a tier ends up wearing two labels."""
    episodes = [episode(method="bc"), episode(method="iql")]
    with pytest.raises(ValueError, match="arm"):
        summarise_cell(episodes)


def test_summarise_cell_carries_per_seed_admission_never_only_the_pooled_number() -> None:
    summary = cell(per_seed_entered={101: 900, 202: 800})
    assert summary.per_seed_admission == {"101": 0.9, "202": 0.8}
    assert summary.admission_ratio == pytest.approx(0.85)
    assert summary.admission_spread == pytest.approx(0.1)
    record = summary.as_record()
    assert record["per_seed_admission"] == {"101": 0.9, "202": 0.8}
    assert record["att_difference_mean"] == pytest.approx(-10.0)


# ----------------------------------------------------------------------
# The filesystem-mutation barrier
# ----------------------------------------------------------------------


def test_default_protected_roots_cover_every_sibling_output_directory(tmp_path: Path) -> None:
    """``output/`` holds the only copy of nine manifests' checkpoints; siblings are read-only."""
    output = tmp_path / "output"
    for name in ("p4_6", "p5_1", "p5_2", "p5_3b", "p8_4a"):
        (output / name).mkdir(parents=True)
    corpus = tmp_path / "datasets_v11"
    corpus.mkdir()
    roots = ProbeRoots(
        repo_root=tmp_path,
        corpus_root=corpus,
        draws_root=tmp_path / "draws",
        output_root=output,
        work_dir=output / "p8_4a",
    )
    protected = default_protected_roots(roots)
    assert (output / "p5_3b").resolve() in protected
    assert (output / "p5_2").resolve() in protected
    assert corpus.resolve() in protected
    assert (output / "p8_4a").resolve() not in protected


def test_a_write_under_a_protected_root_is_refused_and_creates_nothing(tmp_path: Path) -> None:
    """A refused write leaves the tree byte-for-byte as it was: no file, no directory."""
    from offline.tier_sweep import assert_writable

    output = tmp_path / "output"
    (output / "p5_3b").mkdir(parents=True)
    (output / "p8_4a").mkdir(parents=True)
    roots = ProbeRoots(
        repo_root=tmp_path,
        corpus_root=tmp_path / "datasets_v11",
        draws_root=tmp_path / "draws",
        output_root=output,
        work_dir=output / "p8_4a",
    )
    protected = default_protected_roots(roots)
    target = output / "p5_3b" / "nested" / "admission.json"
    with pytest.raises(PermissionError, match="read-only"):
        assert_writable(target, protected)
    assert not (output / "p5_3b" / "nested").exists()
    assert sorted(p.name for p in (output / "p5_3b").iterdir()) == []


# ----------------------------------------------------------------------
# Simulator-backed
# ----------------------------------------------------------------------


def test_analytic_created_matches_a_raw_engine(
    cityflow_available: bool, tmp_path: Path
) -> None:
    """R5 against the engine itself, on a synthetic flow whose boundary entries are the point.

    A raw ``cityflow.Engine`` is stepped ``horizon`` times on a real roadnet with a flow file holding
    entries at 0, 5, ``horizon-1``, ``horizon`` and ``horizon+1``.  Nothing can finish in 20 s on
    this network, so the whole pool is still present and
    ``len(get_vehicles(include_waiting=True))`` IS the created count.
    """
    import cityflow

    scenario = REPO_ROOT / "scenarios/hangzhou_1x1_bc-tyc_18041610_1h"
    if not (scenario / "roadnet.json").is_file():
        pytest.skip(f"the hz1x1 roadnet is not present at {scenario}, so no engine can be built")

    horizon = 20
    route = json.loads((scenario / "flow.json").read_bytes())[0]["route"]
    entries = [
        {**flow_entry(t), "route": route}
        for t in (0, 5, horizon - 1, horizon, horizon + 1)
    ]
    flow_path = write_flow(tmp_path, entries)
    config = {
        "interval": 1.0,
        "seed": 0,
        "dir": f"{scenario}/",
        "roadnetFile": "roadnet.json",
        "flowFile": str(flow_path),
        "rlTrafficLight": False,
        "saveReplay": False,
        "laneChange": False,
    }
    config_path = tmp_path / "cityflow.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    engine = cityflow.Engine(str(config_path), 1)
    for _ in range(horizon):
        engine.next_step()
    from_engine = len(engine.get_vehicles(include_waiting=True))

    assert created_from_flow(flow_path, horizon_seconds=horizon) == from_engine
    assert from_engine == 3


def test_probe_reproduces_a_committed_att_horizon(
    cityflow_available: bool, corpus_v11_root: Path, draws_root: Path
) -> None:
    """The fidelity gate: one replayed episode must equal its committed ``att_horizon`` exactly.

    ``maxpressure`` on ``cityflow1x1`` draw 1000, against
    ``docs/data/p4_heldout_thresholds.json``.  If this drifts, the replay is not faithful and no
    admission number from it is interpretable.
    """
    from offline.dt_gate import _maxpressure_factory, env_settings_from_manifest

    manifest = corpus_v11_root / "cf_hz1x1__maxpressure" / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"the maxpressure tier is not in this corpus at {manifest}")
    config = draws_root / "cityflow1x1" / "draw_1000" / "cityflow.json"
    if not config.is_file():
        pytest.skip(f"held-out draw 1000 is not materialised at {config}")

    committed = json.loads(
        (REPO_ROOT / "docs/data/p4_heldout_thresholds.json").read_text(encoding="utf-8")
    )
    expected = next(
        e
        for e in committed["episodes"]
        if e["arm"] == "maxpressure" and e["draw_id"] == 1000
    )

    settings = env_settings_from_manifest(manifest)
    result = probe_episode(
        scenario="hz1x1",
        tier="maxpressure",
        method=BEHAVIOUR_METHOD,
        arm="behaviour@maxpressure",
        seed=None,
        draw_id=1000,
        config_path=config,
        env_settings=settings,
        scenario_id="cityflow1x1",
        choose_action_factory=_maxpressure_factory,
        engine_seed=1000,
        created=created_from_flow(
            draws_root / "cityflow1x1" / "draw_1000" / "flow.json",
            horizon_seconds=int(settings["max_steps"]) * int(settings["delta_time"]),
        ),
    )

    assert result.att_ours == expected["att_horizon"]
    assert result.horizon_vehicle_count == expected["horizon_vehicle_count"]
    assert result.created == 1813
    assert result.entered == 1813
    assert result.never_entered == 0
