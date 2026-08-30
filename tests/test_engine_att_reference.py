"""Tests for ``offline/engine_att_reference.py`` -- P8.4b Gate 0's engine-ATT reference.

Three layers, and the split is deliberate:

* **pure arithmetic and refusal** -- always runs, no simulator, no corpus.  Equation (C), the
  metric-cadence replay, the admission-latency identity, the criteria scoring, the import fence and
  the filesystem-mutation barrier all live here.
* **raw-engine** -- opt-in, needs the built ``cityflow`` module but no corpus.  This is where the
  LOAD-BEARING premise is tested against the engine's own serialised ``Archive``.
* **corpus-backed** -- opt-in, skipping with a reason that names ``RLTRAFFIC_CORPUS_V11`` and
  ``RLTRAFFIC_DRAWS``.  These roll real episodes through the observer env.

The alignment convention under test is stated in ``offline/engine_att_reference.py``'s module
docstring, derived in ``docs/plans/p8.4b-g0.md`` section 3, and registered as ``PREREGISTRATION``
A11 as amended by A12.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

from offline.engine_att_reference import (
    ARTIFACT_FORMAT_VERSION,
    C1_TOLERANCE,
    C3C_TOLERANCE,
    C4_MIN_DRAWS,
    C4_MIN_TIERS,
    EXTREME_EPISODES,
    GATE_DRAWS,
    GATE_SCENARIOS,
    GATE_TIERS,
    RECONSTRUCTION_SURFACE,
    EngineObservationRecorder,
    GateCell,
    GateEpisode,
    ScenarioCriteria,
    VehicleWindows,
    admission_latency,
    build_parser,
    cell_file_name,
    evaluate_scenario,
    gate_artifact,
    gate_cells,
    main,
    make_observer_env,
    metric_cadence_att,
    observer_env_class,
    reconstruct_att,
    reconstruct_episode,
    seeds_for,
    thread_regime,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "offline" / "engine_att_reference.py"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cityflow_available() -> bool:
    """Skip when the CityFlow engine is not installed, so no engine step can be taken."""
    try:
        import cityflow  # noqa: F401
    except ImportError:
        pytest.skip("the CityFlow engine is not installed, so no episode can be rolled")
    return True


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected datasets_v11/ directory to run the corpus-backed Gate 0 tests"
        )
    return candidate


@pytest.fixture(scope="module")
def draws_root() -> Path:
    """The materialised draw tree; skip if it is absent (it is gitignored, so per-worktree)."""
    env_value = os.environ.get("RLTRAFFIC_DRAWS")
    candidate = Path(env_value) if env_value else REPO_ROOT / "scenarios/draws"
    if not candidate.is_dir():
        pytest.skip(
            f"materialised draws not found at {candidate}: set RLTRAFFIC_DRAWS to a "
            "scenarios/draws directory to run the draw-backed Gate 0 tests"
        )
    return candidate


def windows(pairs: dict[str, tuple[float, float]]) -> VehicleWindows:
    """A :class:`VehicleWindows` from ``{id: (first, last)}``."""
    return VehicleWindows(
        first_seen={k: v[0] for k, v in pairs.items()},
        last_seen={k: v[1] for k, v in pairs.items()},
    )


def episode(**overrides: Any) -> GateEpisode:
    """A :class:`GateEpisode` whose defaults satisfy every criterion, for targeted mutation."""
    fields: dict[str, Any] = {
        "scenario": "hz1x1",
        "tier": "maxpressure",
        "method": "behaviour",
        "arm": "behaviour@maxpressure",
        "seed": None,
        "draw_id": 1000,
        "role": "tier",
        "engine_seed": 1000,
        "att_reference_engine_population": 160.0,
        "att_reference_entered_running": 160.0,
        "att_reference_entered_population": 160.0,
        "att_reference_metric_cadence": 165.0,
        "att_engine_call": 160.0,
        "att_ours": 165.0,
        "n_reference_ids": 1813,
        "n_entered_ids": 1813,
        "created_from_flow": 1813,
        "entered": 1813,
        "never_entered": 0,
        "admission_latency_mean": 0.0,
        "admission_latency_max": 0.0,
        "n_admission_delayed": 0,
        "interval": 1.0,
        "n_observations": 3600,
        "seconds": 1.0,
        "seconds_rollout": 0.9,
    }
    fields.update(overrides)
    return GateEpisode(**fields)


def covering_episodes(scenario: str = "hz1x1") -> list[GateEpisode]:
    """Seven tiers x three draws plus both extremes, all of them passing every criterion."""
    out: list[GateEpisode] = []
    for tier in GATE_TIERS:
        for draw_id in GATE_DRAWS:
            out.append(
                episode(
                    scenario=scenario,
                    tier=tier,
                    method="behaviour",
                    arm=f"behaviour@{tier}",
                    draw_id=draw_id,
                    role="tier",
                )
            )
    for extreme in EXTREME_EPISODES[scenario]:
        out.append(
            episode(
                scenario=scenario,
                tier=extreme.tier,
                method=extreme.method,
                arm=f"{extreme.method}@{extreme.tier}",
                seed=extreme.seed,
                draw_id=extreme.draw_id,
                role=f"extreme_{extreme.which}",
                never_entered=0 if extreme.which == "max" else 688,
                entered=1813 if extreme.which == "max" else 1125,
                att_reference_entered_running=160.0 if extreme.which == "max" else 120.0,
                att_reference_entered_population=160.0 if extreme.which == "max" else 120.0,
                n_entered_ids=1813 if extreme.which == "max" else 1125,
            )
        )
    return out


# ----------------------------------------------------------------------
# The registered declarations
# ----------------------------------------------------------------------


def test_the_registered_declarations_are_what_a11_and_a12_registered() -> None:
    """The constants are the registered ones, so a silent edit is a failing test."""
    assert ARTIFACT_FORMAT_VERSION == "p8.4b-g0-reference/1.0"
    assert C1_TOLERANCE == 1e-4
    assert C3C_TOLERANCE == 1e-4
    assert C4_MIN_TIERS == 7
    assert C4_MIN_DRAWS == 3
    assert GATE_TIERS == (
        "fixedtime",
        "mappo060",
        "mappo200",
        "mappo500",
        "mappo1000",
        "maxpressure",
        "random",
    )
    assert len(GATE_TIERS) == C4_MIN_TIERS
    assert GATE_DRAWS == (1000, 1001, 1002)
    assert len(GATE_DRAWS) == C4_MIN_DRAWS
    assert sorted(GATE_SCENARIOS) == ["grid4x4", "hz1x1"]
    assert "cologne3" not in " ".join(GATE_SCENARIOS)


def test_the_extreme_episodes_are_the_ones_the_committed_artifact_names() -> None:
    """Amendment A2's two extremes per scenario, recomputed from the artifact, not quoted.

    ``docs/data/p8_4a_admission.json`` is merged and unchanged; the minimum is read off it here by an
    independent route (a fresh scan of all rows) and compared with the declared constant.  The
    maximum is 1.0 with many ties, so the registered tie-break -- lowest ``(arm, seed, draw_id)`` --
    is applied here too rather than trusted.
    """
    payload = json.loads((REPO_ROOT / "docs/data/p8_4a_admission.json").read_bytes())
    rows = payload["episodes"]
    assert rows, "p8_4a_admission.json carries no episodes -- wrong artifact?"

    for scenario, declared in EXTREME_EPISODES.items():
        subset = [r for r in rows if r["scenario"] == scenario]
        assert subset, f"{scenario} has no rows in p8_4a_admission.json"
        by_which = {e.which: e for e in declared}
        assert sorted(by_which) == ["max", "min"]

        low = min(subset, key=lambda r: (r["entered_fraction"], r["arm"], str(r["seed"]), r["draw_id"]))
        want = by_which["min"]
        assert low["arm"] == f"{want.method}@{want.tier}"
        assert low["seed"] == want.seed
        assert low["draw_id"] == want.draw_id
        assert low["entered_fraction"] == want.entered_fraction

        top_value = max(r["entered_fraction"] for r in subset)
        ties = sorted(
            (r for r in subset if r["entered_fraction"] == top_value),
            key=lambda r: (r["arm"], str(r["seed"]), r["draw_id"]),
        )
        high = ties[0]
        want = by_which["max"]
        assert high["arm"] == f"{want.method}@{want.tier}"
        assert high["seed"] == want.seed
        assert high["draw_id"] == want.draw_id
        assert high["entered_fraction"] == want.entered_fraction


def test_lane_change_is_false_in_every_cityflow_config() -> None:
    """The premise that makes the reconstruction possible, re-asserted rather than inherited.

    With ``laneChange`` false there are no non-``isReal()`` shadow vehicles, so
    ``get_vehicles(include_waiting=True)`` IS the whole ``vehiclePool`` -- and ``LaneChange::finished``
    is never set, so every vehicle erased from the pool is credited to ``finishedVehicleCnt``.
    The enumeration is asserted non-empty AND asserted at its known size, so a deleted config is
    visible instead of quietly shrinking the evidence (a check must assert its input was non-empty,
    or *found nothing* masquerades as *found nothing wrong*).
    """
    configs = sorted((REPO_ROOT / "configs" / "sim").glob("*.json"))
    assert configs, "no CityFlow sim configs found -- wrong path?"
    assert len(configs) == 13, f"expected 13 sim configs, found {len(configs)}: {configs}"

    explicit = 0
    for path in configs:
        payload = json.loads(path.read_bytes())
        assert payload.get("laneChange", False) is False, f"{path} enables laneChange"
        assert float(payload["interval"]) == 1.0, f"{path} does not run at interval 1.0"
        explicit += "laneChange" in payload
    assert explicit == 11, f"expected 11 configs to set laneChange explicitly, got {explicit}"


# ----------------------------------------------------------------------
# A11's independence clause, mechanically enforced (Amendment A3)
# ----------------------------------------------------------------------


def _forbidden_imports(source: str, *, surface: tuple[str, ...]) -> list[str]:
    """Every reference to ``metrics`` or ``offline.admission_probe`` inside the named definitions."""
    tree = ast.parse(source, filename="engine_att_reference.py")
    banned = ("metrics", "offline.admission_probe", "admission_probe")
    found: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in surface:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import):
                for alias in inner.names:
                    if any(alias.name == b or alias.name.startswith(b + ".") for b in banned):
                        found.append(f"{node.name}:{inner.lineno}: import {alias.name}")
            elif isinstance(inner, ast.ImportFrom):
                module = inner.module or ""
                if any(module == b or module.startswith(b + ".") for b in banned):
                    found.append(f"{node.name}:{inner.lineno}: from {module} import ...")
            elif isinstance(inner, ast.Attribute):
                parts: list[str] = []
                cursor: Any = inner
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    parts.append(cursor.id)
                dotted = ".".join(reversed(parts))
                if any(dotted == b or dotted.startswith(b + ".") for b in banned):
                    found.append(f"{node.name}:{inner.lineno}: attribute {dotted}")
    return found


def test_the_import_fence_scanner_catches_a_planted_violation() -> None:
    """Positive control: the fence is a check only if it can fail.

    Three shapes are planted -- a plain import, a from-import and an attribute chain -- and the
    allowlist is asserted non-empty, because a fence over an empty surface passes vacuously.
    """
    assert RECONSTRUCTION_SURFACE, "the reconstruction allowlist is empty, so the fence is vacuous"

    planted = (
        "def reconstruct_att():\n"
        "    import metrics\n"
        "    from offline.admission_probe import created_from_flow\n"
        "    return metrics.CityFlowMetrics\n"
        "def unlisted():\n"
        "    import metrics\n"
    )
    flagged = _forbidden_imports(planted, surface=("reconstruct_att",))
    assert len(flagged) == 3, f"expected 3 violations inside the surface, got {flagged}"
    assert all("unlisted" not in f for f in flagged), flagged

    clean = "def reconstruct_att():\n    import numpy\n    from envs.cityflow_env import CityFlowEnv\n"
    assert _forbidden_imports(clean, surface=("reconstruct_att",)) == []


def test_the_reconstruction_imports_neither_metrics_nor_the_admission_probe() -> None:
    """A11: the reconstruction's evidential value IS its independence from both."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    names = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = [n for n in RECONSTRUCTION_SURFACE if n not in names]
    assert missing == [], f"RECONSTRUCTION_SURFACE names definitions that do not exist: {missing}"

    violations = _forbidden_imports(source, surface=RECONSTRUCTION_SURFACE)
    assert violations == [], (
        "A11 requires the reconstruction to import nothing from metrics/ and nothing from "
        "offline/admission_probe.py:\n" + "\n".join(violations)
    )


# ----------------------------------------------------------------------
# Equation (C)
# ----------------------------------------------------------------------


def test_contribution_is_last_seen_minus_first_seen_plus_interval() -> None:
    """Equation (C), recomputed by an independent numpy route rather than by calling it twice."""
    import numpy as np

    pairs = {"a": (1.0, 10.0), "b": (5.0, 5.0), "c": (2.0, 3600.0), "d": (3600.0, 3600.0)}
    got = reconstruct_att(windows(pairs), interval=1.0)

    first = np.array([1.0, 5.0, 2.0, 3600.0], dtype=np.float64)
    last = np.array([10.0, 5.0, 3600.0, 3600.0], dtype=np.float64)
    expected_total = float(np.sum(last - first + 1.0))
    assert got.n_ids == 4
    assert got.total == expected_total
    assert got.value == expected_total / 4

    # A vehicle seen on exactly one snapshot contributes one interval, never zero: it was in the
    # pool for the whole of that step.
    single = reconstruct_att(windows({"b": (5.0, 5.0)}), interval=1.0)
    assert single.value == 1.0


def test_reconstruct_att_refuses_a_broken_observation_stream() -> None:
    """Each refusal names a different defect, so a passing episode cannot hide one."""
    with pytest.raises(ValueError, match="interval must be positive"):
        reconstruct_att(windows({"a": (1.0, 2.0)}), interval=0.0)

    mismatched = VehicleWindows(first_seen={"a": 1.0}, last_seen={"a": 2.0, "b": 3.0})
    with pytest.raises(ValueError, match="observed different vehicle ids"):
        reconstruct_att(mismatched, interval=1.0)

    with pytest.raises(ValueError, match="was last seen before it was first seen"):
        reconstruct_att(windows({"a": (5.0, 4.0)}), interval=1.0)


def test_an_empty_stream_returns_zero_like_the_engine_does() -> None:
    """``engine.cpp:690`` returns 0 when the denominator is 0; the reference must not divide."""
    empty = reconstruct_att(windows({}), interval=1.0)
    assert empty.n_ids == 0
    assert empty.value == 0.0
    assert empty.total == 0.0


# ----------------------------------------------------------------------
# The recorder
# ----------------------------------------------------------------------


def test_the_recorder_keeps_only_decision_grid_snapshots_for_the_cadence_replay() -> None:
    """Per-second windows for (C); decision-grid SETS only where ``t % delta_time == 0``."""
    recorder = EngineObservationRecorder(delta_time=10.0)
    for step in range(1, 21):
        t = float(step)
        pooled = ["a"] if step <= 15 else []
        running = ["a"] if 3 <= step <= 15 else []
        recorder.observe(engine_time=t, pooled_ids=pooled, running_ids=running)

    assert recorder.observation_times == tuple(float(s) for s in range(1, 21))
    assert recorder.observed_interval == 1.0
    assert recorder.pool_windows.first_seen == {"a": 1.0}
    assert recorder.pool_windows.last_seen == {"a": 15.0}
    assert recorder.running_windows.first_seen == {"a": 3.0}
    assert recorder.running_windows.last_seen == {"a": 15.0}
    assert recorder.decision_grid_snapshots == ((10.0, frozenset({"a"})), (20.0, frozenset()))


def test_the_recorder_refuses_an_inconsistent_or_out_of_order_read() -> None:
    """Two defects that would silently corrupt every downstream number."""
    recorder = EngineObservationRecorder(delta_time=10.0)
    recorder.observe(engine_time=1.0, pooled_ids=["a"], running_ids=["a"])
    with pytest.raises(ValueError, match="engine time did not advance"):
        recorder.observe(engine_time=1.0, pooled_ids=["a"], running_ids=["a"])
    with pytest.raises(ValueError, match="running but not pooled"):
        recorder.observe(engine_time=2.0, pooled_ids=["a"], running_ids=["a", "b"])


def test_clearing_the_recorder_drops_every_observation() -> None:
    """One recorder must be able to serve consecutive episodes without leaking the first."""
    recorder = EngineObservationRecorder(delta_time=10.0)
    recorder.observe(engine_time=1.0, pooled_ids=["a"], running_ids=["a"])
    recorder.clear()
    assert recorder.observation_times == ()
    assert recorder.pool_windows.first_seen == {}
    assert recorder.decision_grid_snapshots == ()


def test_the_observed_interval_is_measured_and_refuses_a_ragged_grid() -> None:
    """The interval comes from the observations, never from the config."""
    ragged = EngineObservationRecorder(delta_time=10.0)
    for t in (1.0, 2.0, 4.0):
        ragged.observe(engine_time=t, pooled_ids=[], running_ids=[])
    with pytest.raises(ValueError, match="observation grid is not uniform"):
        _ = ragged.observed_interval

    short = EngineObservationRecorder(delta_time=10.0)
    short.observe(engine_time=1.0, pooled_ids=[], running_ids=[])
    with pytest.raises(ValueError, match="at least two observations"):
        _ = short.observed_interval


# ----------------------------------------------------------------------
# The admission-latency term (docs/plans/p8.4b-g0.md section 8.3)
# ----------------------------------------------------------------------


def test_admission_latency_is_the_gap_between_pool_entry_and_admission() -> None:
    """The term A12 (3a)'s bit-identity argument does not account for."""
    pool = windows({"a": (1.0, 100.0), "b": (2.0, 100.0), "c": (3.0, 100.0)})
    running = windows({"a": (1.0, 100.0), "b": (5.0, 100.0), "c": (14.0, 100.0)})
    got = admission_latency(pool, running)
    assert got.n_ids == 3
    assert got.n_delayed == 2
    assert got.maximum == 11.0
    assert got.mean == (0.0 + 3.0 + 11.0) / 3


def test_the_two_entered_only_variants_differ_by_exactly_the_admission_latency() -> None:
    """The identity measured on a real episode, asserted as an identity on constructed data.

    With every vehicle admitted (so the populations coincide), the entered-only variant on the
    ADMISSION clock is the pool variant minus the mean admission latency.  That identity is the root
    cause of A12 (3a)'s failure under ``BRIEF_32`` section 4's definition, and it is asserted here so
    the number reported in the artifact is not the only place it is checked.
    """
    pool = windows({"a": (1.0, 100.0), "b": (2.0, 100.0), "c": (3.0, 100.0)})
    running = windows({"a": (1.0, 100.0), "b": (5.0, 100.0), "c": (14.0, 100.0)})
    pooled = reconstruct_att(pool, interval=1.0)
    admitted = reconstruct_att(running, interval=1.0)
    latency = admission_latency(pool, running)
    assert admitted.value == pooled.value - latency.mean

    population_only = reconstruct_att(pool.restricted_to(running.ids()), interval=1.0)
    assert population_only.value == pooled.value


def test_restricting_a_stream_refuses_an_id_it_never_saw() -> None:
    """A restriction that silently invented an id would fabricate a population."""
    pool = windows({"a": (1.0, 10.0)})
    with pytest.raises(ValueError, match="never observed"):
        pool.restricted_to({"a", "zz"})


# ----------------------------------------------------------------------
# A12 (3c): the metric-cadence replay
# ----------------------------------------------------------------------


def test_the_metric_cadence_replay_uses_the_midpoint_and_end_of_window_estimators() -> None:
    """The published algorithm, computed by hand on a three-window example.

    Two vehicles.  ``a`` is active at t=10 and t=20 and gone at t=30; ``b`` appears at t=20 and is
    still active at t=30.  ``prev_active`` starts empty, so both are "new" when first seen.

    * ``a``: depart = 10 - 5 = 5; completed at t=30, real_tt = 30 - 5 = 25.
    * ``b``: depart = 20 - 5 = 15; still active at t=30, credit = 30 - 15 = 15.
    """
    snapshots = [
        (10.0, frozenset({"a"})),
        (20.0, frozenset({"a", "b"})),
        (30.0, frozenset({"b"})),
    ]
    got = metric_cadence_att(snapshots, delta_time=10.0)
    assert got.n_ids == 2
    assert got.total == 40.0
    assert got.value == 20.0


def test_the_metric_cadence_replay_never_counts_a_vehicle_it_never_saw_active() -> None:
    """A vehicle that appears and vanishes inside one window is invisible to the 10 s grid.

    That invisibility is a PROPERTY of the metric being replayed, not a defect in the replay, and it
    is pinned here so a future change to the replay cannot quietly "fix" it into a different metric.
    """
    snapshots = [(10.0, frozenset()), (20.0, frozenset()), (30.0, frozenset())]
    got = metric_cadence_att(snapshots, delta_time=10.0)
    assert got.n_ids == 0
    assert got.value == 0.0


def test_the_metric_cadence_replay_refuses_a_malformed_grid() -> None:
    with pytest.raises(ValueError, match="delta_time must be positive"):
        metric_cadence_att([(10.0, frozenset())], delta_time=0.0)
    with pytest.raises(ValueError, match="decision grid is not increasing"):
        metric_cadence_att([(20.0, frozenset()), (10.0, frozenset())], delta_time=10.0)
    with pytest.raises(ValueError, match="no decision-grid snapshots"):
        metric_cadence_att([], delta_time=10.0)


# ----------------------------------------------------------------------
# Criteria scoring
# ----------------------------------------------------------------------


def test_the_registered_cell_list_covers_seven_tiers_three_draws_and_both_extremes() -> None:
    """Criterion 4's coverage is a property of the DECLARATION, checkable before anything runs."""
    for scenario in GATE_SCENARIOS:
        cells = gate_cells(scenario)
        tier_cells = [c for c in cells if c.role == "tier"]
        assert {c.tier for c in tier_cells} == set(GATE_TIERS)
        for tier in GATE_TIERS:
            assert {c.draw_id for c in tier_cells if c.tier == tier} == set(GATE_DRAWS)
        assert len(tier_cells) == len(GATE_TIERS) * len(GATE_DRAWS)

        extremes = [c for c in cells if c.role.startswith("extreme")]
        assert len(extremes) == len(EXTREME_EPISODES[scenario])
        for declared in EXTREME_EPISODES[scenario]:
            assert any(
                c.tier == declared.tier
                and c.method == declared.method
                and c.seed == declared.seed
                and c.draw_id == declared.draw_id
                for c in extremes
            ), f"{scenario} is missing its {declared.which} extreme"
        assert len({(c.scenario, c.tier, c.method, c.seed, c.draw_id) for c in cells}) == len(cells)


def test_the_deterministic_anchors_take_a_single_null_seed_slot() -> None:
    """Mirrors ``admission_probe.seeds_for``: maxpressure and fixedtime are seedless anchors."""
    assert seeds_for("maxpressure", "behaviour") == (None,)
    assert seeds_for("fixedtime", "behaviour") == (None,)
    assert seeds_for("random", "behaviour") == (101,)
    assert seeds_for("mappo060", "behaviour") == (101,)


def test_a_clean_scenario_passes_every_criterion() -> None:
    """The baseline the mutations below are measured against."""
    got = evaluate_scenario(covering_episodes())
    assert got.c1_passed
    assert got.c1_max_deviation == 0.0
    assert got.c2_exact
    assert got.c4_passed
    assert got.c4_n_tiers == 7
    assert got.c4_min_draws_per_tier == 3
    assert got.c4_extremes_missing == ()
    assert got.c3b_passed
    assert got.c3c_agrees
    assert got.passed


def test_criterion_one_fails_when_the_reconstruction_and_the_engine_disagree() -> None:
    """A deviation above 1e-4 is a FAIL, and the observed maximum is reported whatever it is."""
    rows = covering_episodes()
    rows[0] = episode(
        scenario=rows[0].scenario,
        tier=rows[0].tier,
        arm=rows[0].arm,
        draw_id=rows[0].draw_id,
        att_reference_engine_population=160.5,
        att_engine_call=160.0,
    )
    got = evaluate_scenario(rows)
    assert not got.c1_passed
    assert got.c1_max_deviation == pytest.approx(0.5)
    assert not got.passed


def test_criterion_one_reports_a_deviation_that_is_below_the_bar_without_hiding_it() -> None:
    """1e-5 passes; the observed value is still reported, because the headroom is the evidence."""
    rows = covering_episodes()
    rows[0] = episode(
        scenario=rows[0].scenario,
        tier=rows[0].tier,
        arm=rows[0].arm,
        draw_id=rows[0].draw_id,
        att_reference_engine_population=160.0 + 1e-5,
    )
    got = evaluate_scenario(rows)
    assert got.c1_passed
    assert got.c1_max_deviation == pytest.approx(1e-5)


def test_criterion_two_is_exact_integer_equality_and_names_the_episode_that_broke_it() -> None:
    """One vehicle out is a defect, not a tolerance."""
    rows = covering_episodes()
    rows[2] = episode(
        scenario=rows[2].scenario,
        tier=rows[2].tier,
        arm=rows[2].arm,
        draw_id=rows[2].draw_id,
        n_reference_ids=1812,
    )
    got = evaluate_scenario(rows)
    assert not got.c2_exact
    assert len(got.c2_mismatches) == 1
    assert got.c2_mismatches[0]["n_reference_ids"] == 1812
    assert got.c2_mismatches[0]["created_from_flow"] == 1813
    assert not got.passed


def test_criterion_three_a_is_scored_as_registered_and_the_alternative_is_reported_beside_it() -> None:
    """A12 (3a): bit-identity where ``never_entered == 0``, ZERO tolerance.

    Scored against ``entered_running`` -- ``BRIEF_32`` section 4's entered-only variant, which A12
    did not amend.  The population-only reading raised as Q7 is reported as its own field and is
    never substituted for the registered one: reinterpreting a registered criterion is an amendment.
    """
    rows = covering_episodes()
    rows[0] = episode(
        scenario=rows[0].scenario,
        tier=rows[0].tier,
        arm=rows[0].arm,
        draw_id=rows[0].draw_id,
        never_entered=0,
        att_reference_entered_running=160.0 - 0.670160,
        att_reference_entered_population=160.0,
    )
    got = evaluate_scenario(rows)
    assert not got.c3a_passed
    assert got.c3a_max_difference == pytest.approx(0.670160)
    assert got.c3a_passed_population_reading
    assert got.c3a_max_difference_population_reading == 0.0
    assert not got.passed


def test_criterion_three_a_needs_a_qualifying_episode_to_mean_anything() -> None:
    """A control with no qualifying episode reports nothing, and must not pass vacuously."""
    rows = [
        episode(
            scenario="hz1x1",
            tier=tier,
            arm=f"behaviour@{tier}",
            draw_id=draw,
            never_entered=5,
            att_reference_entered_running=150.0,
            att_reference_entered_population=150.0,
            n_entered_ids=1808,
            entered=1808,
        )
        for tier in GATE_TIERS
        for draw in GATE_DRAWS
    ]
    got = evaluate_scenario(rows)
    assert got.c3a_n_qualifying == 0
    assert not got.c3a_passed
    assert not got.passed


def test_criterion_three_b_requires_a_difference_where_vehicles_were_censored() -> None:
    """The positive counterpart: where ``never_entered > 0`` the two MUST differ."""
    rows = covering_episodes()
    rows[1] = episode(
        scenario=rows[1].scenario,
        tier=rows[1].tier,
        arm=rows[1].arm,
        draw_id=rows[1].draw_id,
        never_entered=615,
        entered=1198,
        n_entered_ids=1198,
        att_reference_entered_running=160.0,
        att_reference_entered_population=160.0,
    )
    got = evaluate_scenario(rows)
    assert got.c3b_n_qualifying == 1
    assert got.c3b_min_difference == 0.0
    assert not got.c3b_passed
    assert not got.passed


def test_criterion_three_c_is_reported_and_never_folded_into_the_outcome() -> None:
    """A12: a surprise about ``att_ours`` is escalated as a new finding, never a gate FAIL."""
    rows = covering_episodes()
    rows[0] = episode(
        scenario=rows[0].scenario,
        tier=rows[0].tier,
        arm=rows[0].arm,
        draw_id=rows[0].draw_id,
        att_reference_metric_cadence=165.0 + 3.0,
    )
    got = evaluate_scenario(rows)
    assert not got.c3c_agrees
    assert got.c3c_max_deviation == pytest.approx(3.0)
    assert got.passed, "(3c) is REQUIRED and REPORTED but explicitly NOT GATING (A12)"


def test_criterion_four_fails_when_a_tier_or_an_extreme_is_missing() -> None:
    """Coverage is counted, not assumed, and the missing extreme is named."""
    rows = [e for e in covering_episodes() if e.tier != "mappo200"]
    got = evaluate_scenario(rows)
    assert got.c4_n_tiers == 6
    assert not got.c4_passed
    assert not got.passed

    rows = [e for e in covering_episodes() if not e.role.startswith("extreme")]
    got = evaluate_scenario(rows)
    assert got.c4_n_tiers == 7
    assert set(got.c4_extremes_missing) == {"min", "max"}
    assert not got.c4_passed


def test_evaluating_an_empty_or_mixed_scenario_raises_rather_than_scoring_it() -> None:
    """Two scenarios under one score would put two networks behind one criterion outcome."""
    with pytest.raises(ValueError, match="no episodes"):
        evaluate_scenario([])
    with pytest.raises(ValueError, match="episodes from more than one scenario"):
        evaluate_scenario([episode(scenario="hz1x1"), episode(scenario="grid4x4")])


# ----------------------------------------------------------------------
# The artifact and the barrier
# ----------------------------------------------------------------------


def test_the_artifact_carries_every_episode_the_criteria_and_no_verdict_on_the_metric() -> None:
    """``BRIEF_32`` section 6 reserves the choice of primary metric to ``Rule R``."""
    rows = covering_episodes()
    payload = gate_artifact(
        episodes=rows,
        criteria={"hz1x1": evaluate_scenario(rows)},
        timing={"seconds_total": 1.0},
        provenance={"runtime": {"git_commit": "deadbeef"}},
    )
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION
    assert len(payload["episodes"]) == len(rows)
    assert set(payload["criteria"]) == {"hz1x1"}
    assert payload["registered"]["tiers"] == list(GATE_TIERS)
    assert payload["registered"]["draws"] == list(GATE_DRAWS)
    assert "what_this_does_not_say" in payload
    assert "thread_regime" in payload["provenance"]

    text = json.dumps(payload)
    assert "average_travel_time" not in text.replace('"role"', ""), (
        "the artifact must not carry the bare per-step metric name as an episode-level key"
    )


def test_the_artifact_refuses_a_science_verdict_planted_in_it() -> None:
    """Proves ``assert_no_science_verdict`` is actually reached, not merely imported."""
    rows = covering_episodes()
    with pytest.raises(ValueError, match="verdict"):
        gate_artifact(
            episodes=rows,
            criteria={"hz1x1": evaluate_scenario(rows)},
            timing={"note": "headline_safe"},
            provenance={},
        )


def test_the_thread_regime_is_read_at_run_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch.set_num_threads()`` is a different knob from ``OMP_NUM_THREADS``.

    A recorded ``torch_num_threads = 1`` does not establish which regime produced a timing, so the
    environment variables are read here -- at run time, from the environment, never assumed.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("MKL_NUM_THREADS", "5")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    got = thread_regime()
    assert got["OMP_NUM_THREADS"] == "7"
    assert got["MKL_NUM_THREADS"] == "5"
    assert got["OPENBLAS_NUM_THREADS"] is None
    assert isinstance(got["torch_num_threads"], int)


def test_a_refused_destination_writes_nothing_and_creates_no_directory(tmp_path: Path) -> None:
    """The filesystem-mutation barrier: validation completes before the first byte is written."""
    from offline.tier_sweep import protected_roots_from, write_json_guarded

    protected = tmp_path / "checkpoints"
    protected.mkdir()
    target = protected / "nested" / "p8_4b_g0_reference.json"
    roots = protected_roots_from([protected])

    rows = covering_episodes()
    payload = gate_artifact(
        episodes=rows,
        criteria={"hz1x1": evaluate_scenario(rows)},
        timing={},
        provenance={},
    )
    with pytest.raises(PermissionError, match="read-only to this task"):
        write_json_guarded(payload, target, roots)
    assert not target.exists()
    assert not target.parent.exists()
    assert sorted(protected.iterdir()) == []


def test_the_cell_file_name_separates_every_episode() -> None:
    """One file per episode, so a job that dies takes one episode with it."""
    a = cell_file_name("hz1x1", "random", "behaviour", 101, 1000)
    b = cell_file_name("hz1x1", "random", "behaviour", 101, 1001)
    c = cell_file_name("hz1x1", "maxpressure", "behaviour", None, 1000)
    assert a != b
    assert a.endswith(".json") and c.endswith(".json")
    assert "none" in c
    assert len({a, b, c}) == 3


# ----------------------------------------------------------------------
# The CLI: a check that reports by printing is not a check
# ----------------------------------------------------------------------


def test_the_cli_ships_the_plural_tiers_flag_ruled_by_amendment_a4() -> None:
    """``admission_probe`` has ``--tier``; criterion 4 is a property of a SET, so this has ``--tiers``."""
    parser = build_parser()
    args = parser.parse_args(
        ["gate", "--scenario", "hz1x1", "--tiers", "random", "maxpressure", "--draws", "1000"]
    )
    assert args.tiers == ["random", "maxpressure"]
    assert args.draws == [1000]
    # argparse exits 2 on an unrecognised flag; the singular --tier is deliberately NOT accepted.
    with pytest.raises(SystemExit, match=r"^2$"):
        parser.parse_args(["gate", "--scenario", "hz1x1", "--tier", "random"])


def test_the_gate_exits_non_zero_on_a_deliberately_wrong_reference(tmp_path: Path) -> None:
    """The positive control that the gate can FAIL.

    A cell file whose reconstruction disagrees with the engine by 5 s is written into the work
    directory and ``report`` is run over it.  The process must exit NON-ZERO: a check that reports
    by printing is not a check.  The clean control beside it must exit zero, so the test proves the
    exit code tracks the criteria rather than always being non-zero.
    """
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "docs" / "data"
    out.mkdir(parents=True)

    def write(rows: list[GateEpisode]) -> None:
        for path in work.glob("*.json"):
            path.unlink()
        for row in rows:
            name = cell_file_name(row.scenario, row.tier, row.method, row.seed, row.draw_id)
            (work / name).write_text(json.dumps(row.as_record()), encoding="utf-8")

    clean = covering_episodes()
    write(clean)
    argv = [
        "--repo-root", str(tmp_path),
        "--work-dir", str(work),
        "--output-root", str(tmp_path / "output"),
        "report",
        "--out", "docs/data/p8_4b_g0_reference.json",
    ]
    assert main(argv) == 0
    assert (out / "p8_4b_g0_reference.json").is_file()

    broken = list(clean)
    broken[0] = episode(
        scenario=clean[0].scenario,
        tier=clean[0].tier,
        arm=clean[0].arm,
        draw_id=clean[0].draw_id,
        att_reference_engine_population=165.0,
        att_engine_call=160.0,
    )
    write(broken)
    assert main(argv) != 0


# ----------------------------------------------------------------------
# Raw-engine tests: the LOAD-BEARING premise, against the engine's own Archive
# ----------------------------------------------------------------------


def _raw_engine(config_path: Path, tmp_path: Path, seed: int = 12345) -> Any:
    """A bare CityFlow engine on *config_path*, with logging paths left untouched."""
    import cityflow

    payload = json.loads(config_path.read_bytes())
    local = tmp_path / "config.json"
    local.write_text(json.dumps(payload), encoding="utf-8")
    engine = cityflow.Engine(str(local), 1)
    engine.set_random_seed(int(seed))
    return engine


def test_first_seen_is_the_engines_enter_time(
    cityflow_available: bool, draws_root: Path, tmp_path: Path
) -> None:
    """THE LOAD-BEARING TEST.  ``first_seen - interval == enterTime``, against the engine's own state.

    ``Archive::dumpVehicle`` (``CityFlow/src/engine/archive.cpp:179-187``) serialises ``id`` and
    ``enterTime`` for every vehicle in ``vehiclePool``, and ``Archive::dump`` (``:153-176``) also
    writes ``step``, ``finishedVehicleCnt`` and ``cumulativeTravelTime``.  Both are bound to Python
    (``CityFlow/src/cityflow.cpp:36-42``).  That turns the premise from an inference about set
    membership into a DIRECT comparison against the engine's own serialised state.

    Three routes, all in this one test:

    1. ``first_seen - interval == enterTime`` for every pooled vehicle, at three interior steps and
       at the horizon.  Exact ``==``: these are integral multiples of ``interval``.
    2. the pooled id set equals ``get_vehicles(include_waiting=True)`` at the same instant.
    3. ``get_average_travel_time()`` recomputed from the archive ALONE -- the engine's own formula,
       ``(cumulativeTravelTime + SUM(step * interval - enterTime)) / (finishedVehicleCnt + n)`` --
       must equal the engine's own call.  This validates the archive route before it is used as
       evidence, and it is CLAUDE.md section 2's double computation by a genuinely different route.
    """
    config = draws_root / "cityflow_grid4x4" / "draw_1000" / "cityflow.json"
    if not config.is_file():
        pytest.skip(f"grid4x4 draw 1000 is not materialised at {config}")
    engine = _raw_engine(config, tmp_path)

    recorder = EngineObservationRecorder(delta_time=10.0)
    checkpoints = (120, 240, 360)
    compared = 0
    for step in range(1, max(checkpoints) + 1):
        engine.next_step()
        now = float(engine.get_current_time())
        recorder.observe(
            engine_time=now,
            pooled_ids=engine.get_vehicles(include_waiting=True),
            running_ids=engine.get_vehicles(include_waiting=False),
        )
        if step not in checkpoints:
            continue

        archive_path = tmp_path / f"archive_{step}.json"
        engine.snapshot().dump(str(archive_path))
        archive = json.loads(archive_path.read_bytes())
        enter_time = {v["id"]: float(v["enterTime"]) for v in archive["vehicles"]}
        assert enter_time, f"the archive at step {step} holds no vehicles -- nothing was compared"

        # Route 1: the premise itself, exactly.
        interval = recorder.observed_interval
        first_seen = recorder.pool_windows.first_seen
        for vid, entered_at in enter_time.items():
            assert vid in first_seen, f"{vid} is in the engine's pool but was never observed"
            assert first_seen[vid] - interval == entered_at, (
                f"step {step}: {vid} was first observed at {first_seen[vid]} but the engine "
                f"records enterTime {entered_at} with interval {interval}"
            )
            compared += 1

        # Route 2: the archive's pool is exactly what get_vehicles(include_waiting=True) returns.
        assert set(enter_time) == set(engine.get_vehicles(include_waiting=True))

        # Route 3: the engine's own formula, evaluated on the archive alone.
        pooled_time = sum(now - t for t in enter_time.values())
        denominator = int(archive["finishedVehicleCnt"]) + len(enter_time)
        from_archive = (float(archive["cumulativeTravelTime"]) + pooled_time) / denominator
        assert from_archive == float(engine.get_average_travel_time()), (
            f"step {step}: the archive route gives {from_archive!r} and the engine's own call gives "
            f"{engine.get_average_travel_time()!r}"
        )
        assert float(archive["step"]) == now / interval

    assert compared > 0, "no vehicle was compared, so the premise was not tested"


def test_first_seen_is_start_time_plus_one_interval_on_a_synthetic_flow(
    cityflow_available: bool, tmp_path: Path
) -> None:
    """The second, independent route to the premise: a flow whose departure times we chose.

    No archive and no real draw -- a hand-written flow file with three known ``startTime``s on the
    grid4x4 roadnet.  ``Flow::nextStep`` emits one vehicle at ``currentTime == startTime``, and the
    snapshot taken after that ``next_step()`` is labelled ``startTime + interval``.
    """
    scenario_dir = REPO_ROOT / "scenarios" / "grid4x4"
    roadnet = scenario_dir / "grid4x4_roadnet_red.json"
    if not roadnet.is_file():
        pytest.skip(f"the grid4x4 roadnet is not present at {roadnet}")

    source_flow = json.loads((scenario_dir / "grid4x4_flow.json").read_bytes())
    assert source_flow, "the grid4x4 flow file is empty -- nothing to build a synthetic flow from"
    template = source_flow[0]
    start_times = (3, 17, 41)
    flow = []
    for start in start_times:
        entry = json.loads(json.dumps(template))
        entry["startTime"] = start
        entry["endTime"] = start
        flow.append(entry)
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(json.dumps(flow), encoding="utf-8")

    config_path = tmp_path / "cityflow.json"
    config_path.write_text(
        json.dumps(
            {
                "interval": 1.0,
                "seed": 0,
                "dir": str(scenario_dir) + "/",
                "roadnetFile": roadnet.name,
                "flowFile": str(flow_path),
                "rlTrafficLight": False,
                "saveReplay": False,
                "laneChange": False,
            }
        ),
        encoding="utf-8",
    )
    import cityflow

    engine = cityflow.Engine(str(config_path), 1)
    recorder = EngineObservationRecorder(delta_time=10.0)
    for _ in range(60):
        engine.next_step()
        recorder.observe(
            engine_time=float(engine.get_current_time()),
            pooled_ids=engine.get_vehicles(include_waiting=True),
            running_ids=engine.get_vehicles(include_waiting=False),
        )

    first_seen = recorder.pool_windows.first_seen
    assert len(first_seen) == len(start_times), (
        f"expected {len(start_times)} vehicles, observed {sorted(first_seen)}"
    )
    assert sorted(first_seen.values()) == [float(s) + 1.0 for s in start_times]


# ----------------------------------------------------------------------
# Corpus-backed: the observer env on a real episode
# ----------------------------------------------------------------------


def _settings(corpus_v11_root: Path, directory: str) -> dict[str, Any]:
    from offline.dt_gate import env_settings_from_manifest

    return env_settings_from_manifest(corpus_v11_root / directory / "manifest.json")


def test_the_observer_env_is_configured_exactly_as_make_env_would(
    cityflow_available: bool, corpus_v11_root: Path, draws_root: Path
) -> None:
    """``make_env`` cannot return a subclass, so the mirror of its CityFlow branch is checked here."""
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    settings = _settings(corpus_v11_root, "cf_hz1x1__maxpressure")
    config = draws_root / "cityflow1x1" / "draw_1000" / "cityflow.json"
    if not config.is_file():
        pytest.skip(f"hz1x1 draw 1000 is not materialised at {config}")

    reference = make_env(
        EnvSpec(
            id="cityflow1x1",
            backend="cityflow",
            paths={"config": str(config)},
            settings=dict(settings),
        )
    )
    observer = make_observer_env(config, settings)
    try:
        assert isinstance(observer, observer_env_class())
        assert isinstance(observer, type(reference))
        assert observer.max_steps == reference.max_steps
        assert observer.delta_time == reference.delta_time
        assert type(observer._phase_controls[0]) is type(reference._phase_controls[0])
        assert observer.observation_space.shape == reference.observation_space.shape
        assert observer.action_space == reference.action_space
        assert observer._metric_names == reference._metric_names
        assert observer._thread_num == reference._thread_num
    finally:
        observer.close()
        reference.close()


def test_the_observer_snapshots_once_per_engine_step_not_once_per_decision_step(
    cityflow_available: bool, corpus_v11_root: Path, draws_root: Path
) -> None:
    """The grain criterion 1 depends on: ``delta_time`` observations per env step, not one.

    A 10 s observation grid is M1's quantisation defect -- exactly the error the reference exists to
    be free of -- so the count is asserted rather than assumed, and the recorded times are asserted
    to be the engine's own clock rather than a Python counter.
    """
    settings = _settings(corpus_v11_root, "cf_hz1x1__maxpressure")
    config = draws_root / "cityflow1x1" / "draw_1000" / "cityflow.json"
    if not config.is_file():
        pytest.skip(f"hz1x1 draw 1000 is not materialised at {config}")

    env = make_observer_env(config, settings)
    try:
        info = env.reset(seed=1000)
        assert env.recorder.observation_times == ()
        action = env.action_space.sample() * 0
        env.step(action)
        times = env.recorder.observation_times
        assert len(times) == int(env.delta_time)
        assert times == tuple(float(t) for t in range(1, int(env.delta_time) + 1))
        assert times[-1] == float(env._eng.get_current_time())
        assert info["sim_time"] == 0.0

        env.step(action)
        assert len(env.recorder.observation_times) == 2 * int(env.delta_time)
        assert env.recorder.observed_interval == 1.0
        assert env.recorder.decision_grid_snapshots[0][0] == float(env.delta_time)

        env.reset(seed=1000)
        assert env.recorder.observation_times == ()
    finally:
        env.close()


def test_the_reconstruction_reproduces_the_engine_on_a_real_episode(
    cityflow_available: bool, corpus_v11_root: Path, draws_root: Path
) -> None:
    """Criterion 1 and criterion 2 on one real episode, plus (3c), each by an independent route."""
    from offline.admission_probe import created_from_flow, read_admission_at_horizon
    from offline.dt_gate import _maxpressure_factory
    from offline.horizon_metric import horizon_rollout

    settings = _settings(corpus_v11_root, "cf_hz1x1__maxpressure")
    config = draws_root / "cityflow1x1" / "draw_1000" / "cityflow.json"
    if not config.is_file():
        pytest.skip(f"hz1x1 draw 1000 is not materialised at {config}")

    env = make_observer_env(config, settings)
    try:
        rollout = horizon_rollout(env, _maxpressure_factory(env), episodes=1, seed=1000)
        horizon = int(settings["max_steps"]) * int(settings["delta_time"])
        created = created_from_flow(config.parent / "flow.json", horizon_seconds=horizon)
        counts = read_admission_at_horizon(env, created=created)
        att_engine_call = float(env._eng.get_average_travel_time())
        built = reconstruct_episode(env.recorder)
    finally:
        env.close()

    assert built.interval == 1.0
    assert built.n_observations == horizon

    # Criterion 1, and the deviation is reported rather than merely thresholded.
    deviation = abs(built.engine_population.value - att_engine_call)
    assert deviation < C1_TOLERANCE, f"criterion 1 deviation {deviation!r}"

    # Criterion 2 by three routes: the observer's count, the flow file, and entered + never_entered.
    assert built.engine_population.n_ids == created
    assert built.engine_population.n_ids == counts.entered + counts.never_entered
    assert isinstance(built.engine_population.n_ids, int)

    # A12 (3c): the metric-cadence replay against our own metric at the horizon.
    att_ours = float(rollout.per_episode_horizon[0])
    assert abs(built.metric_cadence.value - att_ours) < C3C_TOLERANCE

    # The entered-only population is the metric's, and the latency term is the reason the two
    # entered-only variants differ (docs/plans/p8.4b-g0.md section 8.3).
    assert built.entered_population.n_ids == built.entered_running.n_ids
    assert built.latency.n_ids == built.entered_running.n_ids
    assert built.latency.maximum >= 0.0


def test_a_ten_second_observation_grid_would_not_reproduce_the_engine(
    cityflow_available: bool, corpus_v11_root: Path, draws_root: Path
) -> None:
    """M6 as a standing test: the 10 s grid is a DIFFERENT quantity, and by more than 1e-4.

    This is the defect ``BRIEF_32`` section 3 warns about, pinned so nobody can "simplify" the
    observer back onto the env-step boundary without the suite saying so.
    """
    from offline.dt_gate import _maxpressure_factory
    from offline.horizon_metric import horizon_rollout

    settings = _settings(corpus_v11_root, "cf_hz1x1__maxpressure")
    config = draws_root / "cityflow1x1" / "draw_1000" / "cityflow.json"
    if not config.is_file():
        pytest.skip(f"hz1x1 draw 1000 is not materialised at {config}")

    env = make_observer_env(config, settings)
    try:
        horizon_rollout(env, _maxpressure_factory(env), episodes=1, seed=1000)
        att_engine_call = float(env._eng.get_average_travel_time())
        recorder = env.recorder
        delta = float(env.delta_time)
        pool_first = recorder.pool_windows.first_seen
        pool_last = recorder.pool_windows.last_seen
        # Rebuild the same episode as it would look on a 10 s grid: an id is first "seen" at the
        # next grid point after it appeared and last "seen" at the last grid point before it left.
        coarse_first = {v: -(-t // delta) * delta for v, t in pool_first.items()}
        coarse_last = {v: (t // delta) * delta for v, t in pool_last.items()}
        keep = [v for v in coarse_first if coarse_last[v] >= coarse_first[v]]
        assert keep, "the coarse grid dropped every vehicle -- nothing was compared"
        coarse_windows = VehicleWindows(
            first_seen={v: coarse_first[v] for v in keep},
            last_seen={v: coarse_last[v] for v in keep},
        )
        built = reconstruct_att(coarse_windows, interval=delta)
    finally:
        env.close()

    assert abs(built.value - att_engine_call) > C1_TOLERANCE, (
        "a 10 s observation grid reproduced the engine to within criterion 1's tolerance, which "
        "would mean the per-second grain this gate rests on is not load-bearing after all"
    )
