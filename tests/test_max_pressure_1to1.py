"""Tests for ``offline/policies/max_pressure_1to1.py`` -- A27(f)'s set-semantics MaxPressure (``BRIEF_47`` §3.1, §4 test 1).

Three layers:

* **synthetic** -- always runs.  A hand-built intersection where one incoming lane is released by two links and one
  outgoing lane is fed by two, so the frozen per-link formula and the set formula differ by EXACTLY the duplicated counts,
  computed here by hand; a 1:1 stub on which the two agents must agree on every draw of a seeded count vector.
* **live structure** -- builds the real envs (no episode) and measures the per-phase multiplicities A27(f) states: SUMO
  hz1x1 2 / 1, SUMO grid4x4 3 / 1, CityFlow grid4x4 1 / <= 2, CityFlow hz1x1 1 / 1, on the selectable greens and on the
  file-phase basis the registration counted.
* **load-bearing** -- on CityFlow hz1x1 the variant's action equals ``MaxPressureAgent``'s on every step of a held-out
  episode, and its 100 held-out cells reproduce ``docs/data/p4_heldout_thresholds.json``'s MaxPressure cells under ``==``.

GATES: ``RLTRAFFIC_DRAWS`` (default: this tree's ``scenarios/draws``) holding the held-out draws 1000-1099 of
``cityflow1x1`` and ``cityflow_grid4x4`` with their parity subdirectories; CityFlow importable; ``sumo`` on the PATH and
``traci`` importable for the SUMO tests.  Nothing here writes outside ``tmp_path``.

**The registered mutant (packet):** make the variant sum per link.  The synthetic tests and the SUMO measurement die; the
CityFlow hz1x1 identity SURVIVES, by design -- on a 1:1 intersection the two formulas are the same number, which is the
whole point of the load-bearing test.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from algorithms.max_pressure import MaxPressureAgent
from offline.policies.max_pressure_1to1 import (
    ARM,
    PROVENANCE_FORMAT_VERSION,
    MaxPressure1to1Agent,
    make_maxpressure_1to1,
    maxpressure_1to1_factory,
    phase_lane_visits,
    phase_multiplicities,
    roadlink_structure_digest,
    set_pressure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------


class _Ix:
    """The four attributes the frozen agent reads, and nothing else."""

    def __init__(
        self,
        ix_id: str,
        roadlink_lanes: list[tuple[list[str], list[str]]],
        phase_roadlink_mapping: list[list[int]],
        phase_durations: list[float],
    ) -> None:
        self.id = ix_id
        self.roadlink_lanes = roadlink_lanes
        self.phase_roadlink_mapping = phase_roadlink_mapping
        self.phase_durations = phase_durations
        self.num_phases = len(phase_roadlink_mapping)


def _duplicated_ix() -> _Ix:
    """Incoming ``a`` released by L0 and L1; outgoing ``y`` fed by L1 and L2.  Phase 1 is a 5 s transition."""
    return _Ix(
        "J",
        roadlink_lanes=[(["a"], ["x"]), (["a"], ["y"]), (["b"], ["y"]), (["c"], ["z"])],
        phase_roadlink_mapping=[[0, 1, 2], [], [3]],
        phase_durations=[30.0, 5.0, 30.0],
    )


def _one_to_one_ix() -> _Ix:
    """Every lane appears in exactly one link of every phase that releases it (hz1x1's CityFlow shape)."""
    return _Ix(
        "K",
        roadlink_lanes=[(["a"], ["w", "x"]), (["b"], ["y", "z"]), (["c"], ["w", "y"]), (["d"], ["x", "z"])],
        phase_roadlink_mapping=[[], [0, 1], [2, 3], [0], [3]],
        phase_durations=[5.0, 30.0, 30.0, 30.0, 30.0],
    )


def _env(ix: _Ix, *, control_mode: str = "acyclic") -> SimpleNamespace:
    return SimpleNamespace(intersections=[ix], control_mode=control_mode)


def _info(ix: _Ix, counts: dict[str, int], avail: list[int] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"current_phase": 0}
    if avail is not None:
        payload["avail_actions"] = list(avail)
    return {"lane_vehicle_count": dict(counts), "intersections": {ix.id: payload}, "step": 0}


# ----------------------------------------------------------------------
# Synthetic: the two formulas, by hand
# ----------------------------------------------------------------------


def test_the_two_formulas_differ_by_exactly_the_duplicated_counts_on_a_hand_built_phase() -> None:
    """Phase 0 releases L0, L1, L2 with a=7, b=11, x=2, y=3.  Per link: (7-2)+(7-3)+(11-3) = 17.  Set: (7+11)-(2+3) = 13.

    The difference, 4, is ``(2-1)*a - (2-1)*y`` -- ``a`` visited twice, ``y`` fed twice.  Literals, not a re-run formula.
    """
    ix = _duplicated_ix()
    counts = {"a": 7, "b": 11, "c": 13, "x": 2, "y": 3, "z": 5}
    assert MaxPressureAgent._phase_pressure(ix, [0, 1, 2], counts) == 17.0
    assert set_pressure(ix, [0, 1, 2], counts) == 13.0
    assert MaxPressure1to1Agent._phase_pressure(ix, [0, 1, 2], counts) == 13.0
    assert MaxPressureAgent._phase_pressure(ix, [0, 1, 2], counts) - set_pressure(ix, [0, 1, 2], counts) == 7 - 3


def test_the_difference_is_the_duplicated_counts_on_every_seeded_count_vector() -> None:
    """``per_link - set == sum_in (visits-1)*count - sum_out (visits-1)*count``; the visits are THIS test's literals."""
    ix = _duplicated_ix()
    rng = np.random.default_rng(20261015)
    for _ in range(200):
        counts = {lane: int(value) for lane, value in zip("abcxyz", rng.integers(0, 40, size=6))}
        per_link = MaxPressureAgent._phase_pressure(ix, [0, 1, 2], counts)
        duplicated = (2 - 1) * counts["a"] - (2 - 1) * counts["y"]
        assert per_link - set_pressure(ix, [0, 1, 2], counts) == duplicated
        # Phase 2 is one link and 1:1, so the two formulas are the same number there.
        assert MaxPressureAgent._phase_pressure(ix, [3], counts) == set_pressure(ix, [3], counts)


def test_the_duplication_changes_the_chosen_green_where_the_variant_should_differ() -> None:
    """a=7, b=11, x=2, y=3 and c=16, z=2: per link phase 0 = 17 > phase 2 = 14; set phase 0 = 13 < 14."""
    ix = _duplicated_ix()
    counts = {"a": 7, "b": 11, "c": 16, "x": 2, "y": 3, "z": 2}
    assert int(MaxPressureAgent(_env(ix)).act(_info(ix, counts))[0]) == 0
    assert int(MaxPressure1to1Agent(_env(ix)).act(_info(ix, counts))[0]) == 1


def test_phase_lane_visits_counts_every_release_of_every_lane() -> None:
    incoming, outgoing = phase_lane_visits(_duplicated_ix(), [0, 1, 2])
    assert dict(incoming) == {"a": 2, "b": 1}
    assert dict(outgoing) == {"x": 1, "y": 2}


def test_on_a_one_to_one_intersection_the_variant_is_the_frozen_agent_on_every_seeded_draw() -> None:
    """Small counts make ties frequent, so the inherited tie-breaking is exercised as well as the formula."""
    ix = _one_to_one_ix()
    frozen, variant = MaxPressureAgent(_env(ix)), MaxPressure1to1Agent(_env(ix))
    rng = np.random.default_rng(7)
    for _ in range(500):
        counts = {lane: int(value) for lane, value in zip("abcdwxyz", rng.integers(0, 4, size=8))}
        info = _info(ix, counts)
        assert np.array_equal(variant.act(info), frozen.act(info))
        assert variant.act(info).dtype == np.int64


def test_avail_actions_restrict_the_variant_exactly_as_they_restrict_the_frozen_agent() -> None:
    ix = _duplicated_ix()
    counts = {"a": 7, "b": 11, "c": 16, "x": 2, "y": 3, "z": 2}
    for avail in ([0], [1], [0, 1]):
        info = _info(ix, counts, avail)
        assert int(MaxPressure1to1Agent(_env(ix)).act(info)[0]) in avail
    assert int(MaxPressure1to1Agent(_env(ix)).act(_info(ix, counts, [0]))[0]) == 0


def test_a_cyclic_control_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="acyclic control only"):
        MaxPressure1to1Agent(_env(_duplicated_ix(), control_mode="cyclic"))


def test_the_registry_factory_and_the_evaluation_factory_both_return_the_variant() -> None:
    ix = _duplicated_ix()
    counts = {"a": 7, "b": 11, "c": 16, "x": 2, "y": 3, "z": 2}
    policy = make_maxpressure_1to1(_env(ix), SimpleNamespace(), np.random.default_rng(0))
    assert int(policy(_info(ix, counts))[0]) == 1
    choose = maxpressure_1to1_factory(_env(ix))
    assert int(choose(None, _info(ix, counts))[0]) == 1
    assert ARM == "maxpressure_1to1"


def test_the_roadlink_digest_moves_when_one_lane_of_one_link_moves() -> None:
    first = roadlink_structure_digest([_duplicated_ix()])
    assert first == roadlink_structure_digest([_duplicated_ix()])
    assert len(first) == 64
    moved = _duplicated_ix()
    moved.roadlink_lanes[1] = (["a"], ["z"])
    assert roadlink_structure_digest([moved]) != first
    retimed = _duplicated_ix()
    retimed.phase_durations[1] = 6.0
    assert roadlink_structure_digest([retimed]) != first


def test_the_multiplicity_record_on_the_hand_built_intersection() -> None:
    record = phase_multiplicities([_duplicated_ix()])
    assert record["format_version"] == PROVENANCE_FORMAT_VERSION
    phases = record["intersections"]["J"]
    assert [p["file_phase"] for p in phases] == [0, 1, 2]
    assert [p["action"] for p in phases] == [0, None, 1]
    assert [p["selectable"] for p in phases] == [True, False, True]
    assert phases[0] == {
        "file_phase": 0, "action": 0, "selectable": True, "n_links": 3,
        "incoming_visits": 3, "incoming_distinct": 2, "incoming_max": 2, "incoming_min": 1,
        "outgoing_visits": 3, "outgoing_distinct": 2, "outgoing_max": 2, "outgoing_min": 1,
    }
    assert phases[1]["n_links"] == 0 and phases[1]["incoming_max"] is None
    greens = record["summary"]["selectable_greens"]
    assert greens == {
        "n_phases": 2, "n_incoming_repeated": 1, "n_outgoing_repeated": 1,
        "incoming_multiplicities": [1, 2], "outgoing_multiplicities": [1, 2],
    }
    files = record["summary"]["file_phases_with_links"]
    assert files["n_phases"] == 2 and files["n_file_phases"] == 3


# ----------------------------------------------------------------------
# Live structure (gated)
# ----------------------------------------------------------------------


def _draws_root() -> Path:
    value = os.environ.get("RLTRAFFIC_DRAWS")
    root = Path(value) if value else REPO_ROOT / "scenarios" / "draws"
    probes = [
        root / "cityflow1x1" / "draw_1000" / "parity" / "noteleport.sumocfg",
        root / "cityflow_grid4x4" / "draw_1000" / "parity" / "noteleport.sumocfg",
        root / "cityflow1x1" / "draw_1099" / "cityflow.json",
    ]
    missing = [str(path) for path in probes if not path.is_file()]
    if missing:
        pytest.skip(
            f"the held-out draws and their parity files are not at {root} ({missing[:1]}): set RLTRAFFIC_DRAWS to the "
            "main tree's scenarios/draws to run the live multiplicity and identity tests"
        )
    return root


def _require_cityflow() -> None:
    try:
        import cityflow  # noqa: F401
    except ImportError:
        pytest.skip("the CityFlow engine is not installed, so no CityFlow env can be built")


def _require_sumo() -> None:
    if shutil.which("sumo") is None:
        pytest.skip("the sumo binary is not on the PATH, so no SUMO env can be built")
    try:
        import traci  # noqa: F401
    except ImportError:
        pytest.skip("traci is not importable, so no SUMO env can be built")


def _thresholds() -> dict[str, Any]:
    return json.loads((DATA / "p4_heldout_thresholds.json").read_bytes())


def _cityflow_env(config: Path) -> Any:
    from experiments.config import EnvSpec
    from experiments.envs import make_env

    return make_env(
        EnvSpec(id="cityflow1x1", backend="cityflow", paths={"config": str(config)},
                settings=dict(_thresholds()["env_settings"]))
    )


def _sumo_env(scenario_key: str, root: Path) -> Any:
    from offline.aligned_env import observer_env_for_draw

    return observer_env_for_draw(scenario_key, 1000, out_root=root, arm="maxpressure")


def _independent_visits(ix: Any, links: list[int]) -> tuple[dict[str, int], dict[str, int]]:
    """A second route to the visit counts: a plain loop over ``roadlink_lanes``, no ``Counter``, no shared helper."""
    incoming: dict[str, int] = {}
    outgoing: dict[str, int] = {}
    for link in links:
        lanes_in, lanes_out = ix.roadlink_lanes[link]
        for lane in lanes_in:
            incoming[lane] = incoming.get(lane, 0) + 1
        for lane in lanes_out:
            outgoing[lane] = outgoing.get(lane, 0) + 1
    return incoming, outgoing


def _assert_difference_is_the_duplicated_counts(intersections: list[Any]) -> int:
    """On the real roadlink structure, with a synthetic count per lane, for every selectable green."""
    lanes = sorted(
        {lane for ix in intersections for pair in ix.roadlink_lanes for side in pair for lane in side}
    )
    counts = {lane: 3 * index + 1 for index, lane in enumerate(lanes)}
    checked = 0
    for ix in intersections:
        for green in MaxPressureAgent._green_phases(ix):
            links = list(ix.phase_roadlink_mapping[green])
            incoming, outgoing = _independent_visits(ix, links)
            duplicated = sum((visits - 1) * counts[lane] for lane, visits in incoming.items()) - sum(
                (visits - 1) * counts[lane] for lane, visits in outgoing.items()
            )
            per_link = MaxPressureAgent._phase_pressure(ix, links, counts)
            assert per_link - MaxPressure1to1Agent._phase_pressure(ix, links, counts) == duplicated
            checked += 1
    return checked


def test_sumo_hz1x1_every_green_visits_two_incoming_lanes_twice_and_four_outgoing_lanes_once() -> None:
    root = _draws_root()
    _require_sumo()
    env = _sumo_env("cityflow1x1", root)
    try:
        intersections = list(env.intersections)
        record = phase_multiplicities(intersections)
        assert record["summary"]["selectable_greens"] == {
            "n_phases": 8, "n_incoming_repeated": 8, "n_outgoing_repeated": 0,
            "incoming_multiplicities": [2], "outgoing_multiplicities": [1],
        }
        for phase in record["intersections"][str(intersections[0].id)]:
            if phase["selectable"]:
                assert (phase["n_links"], phase["incoming_visits"], phase["incoming_distinct"]) == (4, 4, 2)
                assert (phase["outgoing_visits"], phase["outgoing_distinct"]) == (4, 4)
        assert _assert_difference_is_the_duplicated_counts(intersections) == 8
    finally:
        env.close()


def test_sumo_grid4x4_every_green_visits_each_incoming_lane_three_times() -> None:
    root = _draws_root()
    _require_sumo()
    env = _sumo_env("cityflow_grid4x4", root)
    try:
        intersections = list(env.intersections)
        record = phase_multiplicities(intersections)
        assert record["summary"]["selectable_greens"] == {
            "n_phases": 128, "n_incoming_repeated": 128, "n_outgoing_repeated": 0,
            "incoming_multiplicities": [3], "outgoing_multiplicities": [1],
        }
        files = record["summary"]["file_phases_with_links"]
        # The registration's "160 green phases" is this basis: 128 greens and 32 transition phases that carry links.
        assert (files["n_phases"], files["n_incoming_repeated"], files["n_file_phases"]) == (160, 160, 256)
        assert _assert_difference_is_the_duplicated_counts(intersections) == 128
    finally:
        env.close()


def test_cityflow_grid4x4_feeds_an_outgoing_lane_twice_on_every_selectable_green() -> None:
    root = _draws_root()
    _require_cityflow()
    env = _cityflow_env(root / "cityflow_grid4x4" / "draw_1000" / "cityflow.json")
    try:
        intersections = list(env.intersections)
        record = phase_multiplicities(intersections)
        assert record["summary"]["selectable_greens"] == {
            "n_phases": 128, "n_incoming_repeated": 0, "n_outgoing_repeated": 128,
            "incoming_multiplicities": [1], "outgoing_multiplicities": [1, 2],
        }
        files = record["summary"]["file_phases_with_links"]
        # The registration's "160 of the 256 green phases" is this basis (transition phases included).
        assert (files["n_phases"], files["n_outgoing_repeated"], files["n_file_phases"]) == (256, 160, 256)
        assert _assert_difference_is_the_duplicated_counts(intersections) == 128
    finally:
        env.close()


def test_cityflow_hz1x1_is_one_to_one_on_every_green() -> None:
    root = _draws_root()
    _require_cityflow()
    env = _cityflow_env(root / "cityflow1x1" / "draw_1000" / "cityflow.json")
    try:
        record = phase_multiplicities(list(env.intersections))
        assert record["summary"]["selectable_greens"] == {
            "n_phases": 8, "n_incoming_repeated": 0, "n_outgoing_repeated": 0,
            "incoming_multiplicities": [1], "outgoing_multiplicities": [1],
        }
    finally:
        env.close()


# ----------------------------------------------------------------------
# Load-bearing: CityFlow hz1x1
# ----------------------------------------------------------------------


def test_load_bearing_the_variants_action_is_the_frozen_agents_on_every_step_of_a_held_out_episode() -> None:
    """Draw 1000, ``reset(seed=1000)``, the thresholds' own env settings; both agents read the SAME info each step, the
    episode advances on the variant's action, and every green's two pressures are compared too."""
    root = _draws_root()
    _require_cityflow()
    env = _cityflow_env(root / "cityflow1x1" / "draw_1000" / "cityflow.json")
    try:
        frozen, variant = MaxPressureAgent(env), MaxPressure1to1Agent(env)
        info = env.reset(seed=int(_thresholds()["engine_seed"]))
        steps = 0
        for _ in range(int(env.max_steps)):
            counts = info["lane_vehicle_count"]
            for ix in env.intersections:
                for green in MaxPressureAgent._green_phases(ix):
                    links = ix.phase_roadlink_mapping[green]
                    assert MaxPressureAgent._phase_pressure(ix, links, counts) == set_pressure(ix, links, counts)
            chosen = variant.act(info)
            assert np.array_equal(chosen, frozen.act(info))
            _reward, terminated, truncated, info = env.step(chosen)
            steps += 1
            if terminated or truncated:
                break
        assert steps == 360
    finally:
        env.close()


def test_load_bearing_the_variants_100_held_out_cells_reproduce_the_registered_maxpressure_cells() -> None:
    """``dt_gate.evaluate_arm`` -- the route the thresholds were measured by -- with the variant's factory, the artifact's
    own ``env_settings`` and ``engine_seed``; every field the artifact records, under ``==``, 100 of 100."""
    from offline.dt_gate import evaluate_arm
    from offline.materialise_draws import draw_config_path

    root = _draws_root()
    _require_cityflow()
    thresholds = _thresholds()
    registered = {int(e["draw_id"]): e for e in thresholds["episodes"] if e["arm"] == "maxpressure"}
    assert sorted(registered) == list(range(1000, 1100))
    results = evaluate_arm(
        arm=ARM,
        seed=None,
        draw_ids=sorted(registered),
        config_for_draw=lambda draw: draw_config_path("cityflow1x1", draw, out_root=root),
        env_settings=dict(thresholds["env_settings"]),
        scenario_id="cityflow1x1",
        choose_action_factory=maxpressure_1to1_factory,
        engine_seed=int(thresholds["engine_seed"]),
    )
    differing = [
        result.draw_id
        for result in results
        if (result.att_horizon, result.episode_reward, result.horizon_vehicle_count)
        != (
            registered[result.draw_id]["att_horizon"],
            registered[result.draw_id]["episode_reward"],
            registered[result.draw_id]["horizon_vehicle_count"],
        )
    ]
    assert len(results) == 100
    assert differing == []
