"""Tests for ``offline/backend_alignment.py`` -- P7.1 half B, the movement-keyed adapter.

``BRIEF_34`` §3.3's six load-bearing tests, plus purity:

1. 🔒 P7.0's registered 16-row table recomputed FROM RAW NPZ through the adapter's permutation, with
   the identity permutation reproducing the VOID table as the negative control.  This is the
   independent recomputation P7.0's numbers were fenced behind.
2. A synthetic intersection with a hand-derived inversion.
3. Action semantics: 8/8 under the correspondence, 4/8 under the identity.
4. Live inertness on a real SUMO episode, and the idempotence choice.
5. Refusals: an unknown lane, a wrong state width, a bad metric key.
6. hz4x4 gudang resolves 16/16, structure only.
7. Purity.

⚠️ **The statistics must be computed by the EXACT INTEGER route** (``ks_statistic_exact`` /
``overlap_coefficient_exact``).  A float-normalised histogram gives ``0.34072022160664817`` where the
artifact holds ``0.3407202216066482`` -- adjacent doubles, one ulp apart -- and the tempting response
to that failure is to loosen ``==``, which would retire the whole check.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline import backend_alignment as ba

REPO = Path(__file__).resolve().parents[1]


def _sumo_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


@pytest.fixture(scope="module")
def output_root() -> Path:
    """``RLTRAFFIC_OUTPUT_ROOT``, else ``<repo>/output``; skip unless P7.0's episodes are there."""
    env_value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    candidate = Path(env_value) if env_value else REPO / "output"
    if not (candidate / "p7_0" / "sumo__maxpressure").is_dir():
        pytest.skip(
            f"P7.0's episodes are not present at {candidate}/p7_0: set RLTRAFFIC_OUTPUT_ROOT to the "
            "tree that holds them to run the recomputation of P7.0's table"
        )
    return candidate


@pytest.fixture(scope="module")
def hz1x1() -> ba.ScenarioAlignment:
    """The hz1x1 alignment, built from the two network files only."""
    return _hz1x1_alignment()


def _hz1x1_alignment(**kwargs: Any) -> ba.ScenarioAlignment:
    """Built from the two FILES only: the frozen CityFlow parser and the SUMO ``.net.xml``."""
    from offline import parity

    return ba.alignment_for_scenario(
        "hangzhou_1x1_bc-tyc",
        cityflow_roadnet=parity.DECLARED_SCENARIO_DIR / "roadnet.json",
        sumo_net=parity.DECLARED_SOURCE_NET,
        **kwargs,
    )


def test_the_two_lane_orders_are_the_ones_the_backends_own_parsers_produce(
    hz1x1: ba.ScenarioAlignment,
) -> None:
    """The canonical order is the corpus's discovery order and SUMO's is sorted; they differ."""
    ix = hz1x1.intersections["intersection_1_1"]

    assert list(ix.canonical_lanes) == [
        "road_0_1_0_1", "road_0_1_0_0", "road_1_0_1_1", "road_1_0_1_0",
        "road_2_1_2_1", "road_2_1_2_0", "road_1_2_3_0", "road_1_2_3_1",
    ]
    assert list(ix.sumo_lanes) == sorted(ix.sumo_lanes)
    assert list(ix.canonical_lanes) != list(ix.sumo_lanes)
    assert set(ix.canonical_lanes) == set(ix.sumo_lanes)
    assert (ix.cityflow_num_phases, ix.sumo_num_phases, ix.n_actions) == (9, 16, 8)
    assert (ix.canonical_state_width(), ix.sumo_state_width()) == (25, 32)


def _sumo_info(alignment: ba.ScenarioAlignment, *, phase: int = 4) -> dict[str, Any]:
    """A well-formed SUMO ``info`` (C2) whose lane values encode their own lane index."""
    ix = alignment.intersections["intersection_1_1"]
    counts = {lane: 10 + index for index, lane in enumerate(ix.sumo_lanes)}
    waiting = {lane: 100 + index for index, lane in enumerate(ix.sumo_lanes)}
    outgoing = {"road_1_1_0_0": 7, "road_1_1_1_1": 8}
    state = (
        [float(counts[lane]) for lane in ix.sumo_lanes]
        + [float(waiting[lane]) for lane in ix.sumo_lanes]
        + [1.0 if i == phase else 0.0 for i in range(ix.sumo_num_phases)]
    )
    return {
        "sim_time": 1200.0,
        "vehicle_count": 231,
        "step": 120,
        "average_travel_time": 355.7984322508399,
        "lane_vehicle_count": {**counts, **outgoing},
        "lane_waiting_vehicle_count": {**waiting, **{k: 0 for k in outgoing}},
        "metrics": {"average_travel_time": 355.79, "queue": 12.0},
        "intersections": {
            "intersection_1_1": {
                "state": state,
                "avail_actions": [0, 1, 2, 3, 4, 5, 6, 7],
                "current_phase": phase,
                "time_in_phase": 4,
                "action_applied": True,
                "metrics": {},
                "reward": -12.5,
            }
        },
    }


# ----------------------------------------------------------------------
# The phase map
# ----------------------------------------------------------------------


def test_the_phase_map_is_the_one_measured_from_the_two_network_files() -> None:
    """Green action k -> SUMO 2k -> CityFlow k+1; every odd SUMO phase is the clearance slot."""
    for action in range(8):
        assert ba.sumo_phase_for_action(action) == 2 * action
        assert ba.cityflow_phase_for_sumo_phase(2 * action) == action + 1
    for odd in range(1, 16, 2):
        assert ba.cityflow_phase_for_sumo_phase(odd) == ba.CITYFLOW_CLEARANCE_PHASE


def test_the_phase_map_refuses_a_phase_outside_the_sumo_program() -> None:
    """16 phases: a 17th would silently become a 9th green."""
    with pytest.raises(ValueError, match="phase"):
        ba.cityflow_phase_for_sumo_phase(-1)


# ----------------------------------------------------------------------
# §3.3 test 2 -- the hand-derived inversion
# ----------------------------------------------------------------------


def test_the_permutation_is_the_hand_derived_inversion(hz1x1: ba.ScenarioAlignment) -> None:
    """🔒 §3.3 test 2. Every lane pairs with the OTHER index on its own road, on all four roads.

    Hand-derived from the two files: CityFlow gives ``_0`` the ``turn_left`` roadLink and ``_1``
    ``go_straight``; SUMO gives ``_0`` ``dir="s"`` and ``_1`` ``dir="l"``.
    """
    ix = hz1x1.intersections["intersection_1_1"]

    assert dict(ix.correspondence) == {
        "road_0_1_0_1": "road_0_1_0_0",
        "road_0_1_0_0": "road_0_1_0_1",
        "road_1_0_1_1": "road_1_0_1_0",
        "road_1_0_1_0": "road_1_0_1_1",
        "road_2_1_2_1": "road_2_1_2_0",
        "road_2_1_2_0": "road_2_1_2_1",
        "road_1_2_3_0": "road_1_2_3_1",
        "road_1_2_3_1": "road_1_2_3_0",
    }
    # 8 of 8 ids denote a different physical lane across the two files.
    assert all(k != v for k, v in ix.correspondence.items())
    # The permutation reads the SUMO state in canonical order: position i of the aligned block is
    # SUMO position permutation[i].
    assert sorted(ix.permutation) == list(range(8))
    for canonical_index, sumo_index in enumerate(ix.permutation):
        assert ix.sumo_lanes[sumo_index] == ix.correspondence[ix.canonical_lanes[canonical_index]]


def test_a_correspondence_forced_to_the_identity_gives_a_different_permutation() -> None:
    """The mutation §3.3 test 2 names: identity is a valid permutation and the WRONG one."""
    alignment = _hz1x1_alignment()
    ix = alignment.intersections["intersection_1_1"]
    identity = tuple(ix.sumo_lanes.index(lane) for lane in ix.canonical_lanes)

    assert identity != ix.permutation


# ----------------------------------------------------------------------
# §3.3 test 1 -- P7.0's registered table, recomputed from raw npz through the adapter
# ----------------------------------------------------------------------


def _lane_column(episodes: list[Any], array_name: str, lane: str) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(getattr(e, array_name))[
                :, {str(l): i for i, l in enumerate(e.lane_ids)}[lane]
            ]
            for e in episodes
        ]
    )


def _p7_0_cells(output_root: Path, name: str) -> list[Any]:
    from offline.trajectory_logger import MANIFEST_NAME, load_episode

    directory = output_root / "p7_0" / name
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    return [load_episode(directory / e["filename"]) for e in manifest["episodes"]]


def test_p7_0s_registered_table_recomputes_through_the_adapter(
    output_root: Path, hz1x1: ba.ScenarioAlignment
) -> None:
    """🔒 §3.3 test 1, the independent recomputation P7.0's numbers are fenced behind.

    The permutation comes from the ADAPTER; the statistics are recomputed from the raw ``.npz`` by
    the exact-integer route; ``compare_lane_features`` is never called. All 16 rows must equal
    ``docs/data/p7_0_gate.json:per_feature`` under ``==``.
    """
    from offline.transfer_gate import (
        LANE_ARRAYS,
        ks_statistic_exact,
        overlap_coefficient_exact,
    )

    gate = json.loads((REPO / "docs" / "data" / "p7_0_gate.json").read_text(encoding="utf-8"))
    registered = {row["feature"]: row for row in gate["per_feature"]}
    cityflow = _p7_0_cells(output_root, "cityflow__maxpressure")
    sumo = _p7_0_cells(output_root, "sumo__maxpressure")
    ix = hz1x1.intersections["intersection_1_1"]
    # The compared set is P7.0's: the CityFlow incoming lanes present on both sides, in CityFlow's
    # own order. The adapter supplies the SUMO partner of each.
    compared = list(gate["lane_alignment"]["compared_lanes"])
    assert compared == list(ix.canonical_lanes)

    checked = 0
    for array_name in LANE_ARRAYS:
        for lane in compared:
            partner = ix.correspondence[lane]
            x = _lane_column(cityflow, array_name, lane)
            y = _lane_column(sumo, array_name, partner)
            ks_num, ks_den = ks_statistic_exact(x, y)
            ovl_num, ovl_den = overlap_coefficient_exact(x, y)
            row = registered[f"{array_name}@{lane}"]

            assert row["sumo_lane"] == partner
            assert row["n_cityflow"] == int(x.size)
            assert row["n_sumo"] == int(y.size)
            assert ks_num / ks_den == row["ks_statistic"]
            assert ovl_num / ovl_den == row["overlap_coefficient"]
            checked += 1

    assert checked == 16
    assert min(row["overlap_coefficient"] for row in registered.values()) == 0.3407202216066482


def test_the_identity_permutation_reproduces_p7_0s_void_table(output_root: Path) -> None:
    """🔒 §3.3 test 1's negative control: by-lane-id gives the VOID numbers, min OVL 0.0554.

    Without this the test above is satisfied by any permutation that happens to reproduce 16 rows;
    with it, the two halves are each other's control.
    """
    from offline.transfer_gate import (
        LANE_ARRAYS,
        ks_statistic_exact,
        overlap_coefficient_exact,
    )

    gate = json.loads((REPO / "docs" / "data" / "p7_0_gate.json").read_text(encoding="utf-8"))
    void = {row["feature"]: row for row in gate["voided_per_feature"]["rows"]}
    cityflow = _p7_0_cells(output_root, "cityflow__maxpressure")
    sumo = _p7_0_cells(output_root, "sumo__maxpressure")

    checked = 0
    for array_name in LANE_ARRAYS:
        for lane in gate["lane_alignment"]["compared_lanes"]:
            x = _lane_column(cityflow, array_name, lane)
            y = _lane_column(sumo, array_name, lane)  # the identity: the VOID reading
            ks_num, ks_den = ks_statistic_exact(x, y)
            ovl_num, ovl_den = overlap_coefficient_exact(x, y)
            row = void[f"{array_name}@{lane}"]

            assert row["sumo_lane"] == lane
            assert ks_num / ks_den == row["ks_statistic"]
            assert ovl_num / ovl_den == row["overlap_coefficient"]
            checked += 1

    assert checked == 16
    assert min(row["overlap_coefficient"] for row in void.values()) == 0.055401662049861494


# ----------------------------------------------------------------------
# §3.3 test 3 -- action semantics
# ----------------------------------------------------------------------


def test_the_green_actions_release_the_same_lanes_under_the_correspondence(
    hz1x1: ba.ScenarioAlignment,
) -> None:
    """🔒 §3.3 test 3: 8/8 agree under the correspondence, 4/8 under the identity.

    Reproduced from ``docs/data/p7_0_gate.json``'s released-lane sets rather than quoted, and the
    identity control is what gives the 8/8 its discriminating power: the four that agree anyway are
    exactly the actions releasing both lanes of one road, which are symmetric under a within-road
    swap.
    """
    gate = json.loads((REPO / "docs" / "data" / "p7_0_gate.json").read_text(encoding="utf-8"))
    ix = hz1x1.intersections["intersection_1_1"]
    inverse = {sumo: cityflow for cityflow, sumo in ix.correspondence.items()}

    agree_translated = 0
    agree_identity = 0
    for row in gate["green_action_semantics"]["rows"]:
        cityflow_lanes = sorted(row["cityflow_released_lanes"])
        raw = sorted(row["sumo_released_lanes_raw"])
        assert row["sumo_file_phase"] == ba.sumo_phase_for_action(row["action"])
        assert row["cityflow_file_phase"] == ba.cityflow_phase_for_sumo_phase(row["sumo_file_phase"])
        agree_translated += sorted(inverse[lane] for lane in raw) == cityflow_lanes
        agree_identity += raw == cityflow_lanes

    assert agree_translated == 8
    assert agree_identity == 4


# ----------------------------------------------------------------------
# §3.3 test 5 -- refusals, and the idempotence choice
# ----------------------------------------------------------------------


def test_align_info_rewrites_the_lane_blocks_into_the_canonical_order(
    hz1x1: ba.ScenarioAlignment,
) -> None:
    """The positive control: every refusal below is otherwise met by `raise` on line one."""
    ix = hz1x1.intersections["intersection_1_1"]
    info = _sumo_info(hz1x1)
    aligned = ba.align_info(info, hz1x1)
    payload = aligned["intersections"]["intersection_1_1"]

    assert len(payload["state"]) == ix.canonical_state_width()
    counts = payload["state"][:8]
    waiting = payload["state"][8:16]
    onehot = payload["state"][16:]
    # Position i of the aligned block must be the SUMO value of the lane the correspondence pairs
    # with canonical lane i -- checked against the source dict, not against the permutation.
    for index, lane in enumerate(ix.canonical_lanes):
        partner = ix.correspondence[lane]
        assert counts[index] == info["lane_vehicle_count"][partner]
        assert waiting[index] == info["lane_waiting_vehicle_count"][partner]
    assert len(onehot) == 9
    assert onehot.index(1.0) == ba.cityflow_phase_for_sumo_phase(4)
    assert payload["current_phase"] == ba.cityflow_phase_for_sumo_phase(4)


def test_align_info_re_keys_the_lane_dicts_and_keeps_the_sumo_id(
    hz1x1: ba.ScenarioAlignment,
) -> None:
    """The key translation is declared, and the SUMO id survives in a sidecar rather than dying."""
    ix = hz1x1.intersections["intersection_1_1"]
    info = _sumo_info(hz1x1)
    aligned = ba.align_info(info, hz1x1)

    for lane in ix.canonical_lanes:
        partner = ix.correspondence[lane]
        assert aligned["lane_vehicle_count"][lane] == info["lane_vehicle_count"][partner]
        assert aligned["lane_waiting_vehicle_count"][lane] == (
            info["lane_waiting_vehicle_count"][partner]
        )
    assert aligned["lane_id_translation"][ix.canonical_lanes[0]] == (
        ix.correspondence[ix.canonical_lanes[0]]
    )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda i: i["lane_vehicle_count"].__setitem__("road_9_9_9_9", 3), "road_9_9_9_9"),
        (lambda i: i["intersections"]["intersection_1_1"].__setitem__("state", [0.0] * 31), "width"),
        # ⚠️ NOT a short list: C2 makes avail_actions the legal actions RIGHT NOW, so [0, 1] is
        # normal and must pass. The defect is an INDEX the canonical frame has no green phase for.
        (lambda i: i["intersections"]["intersection_1_1"].__setitem__("avail_actions", [0, 99]), "avail"),
        (lambda i: i["intersections"].__setitem__("intersection_9_9", {}), "intersection_9_9"),
        (lambda i: i["lane_vehicle_count"].pop("road_0_1_0_0"), "road_0_1_0_0"),
    ],
    ids=["unknown-lane", "wrong-state-width", "wrong-action-count", "unknown-intersection", "missing-lane"],
)
def test_align_info_refuses_rather_than_falling_back(
    hz1x1: ba.ScenarioAlignment, mutate: Any, match: str
) -> None:
    """🔒 §3.3 test 5. A positional fallback IS the defect this module exists to prevent."""
    info = _sumo_info(hz1x1)
    mutate(info)

    with pytest.raises((KeyError, ValueError), match=match):
        ba.align_info(info, hz1x1)


def test_align_info_accepts_a_short_avail_actions_list(hz1x1: ba.ScenarioAlignment) -> None:
    """The control for the refusal above: a SUBSET is what C2 says avail_actions is."""
    info = _sumo_info(hz1x1)
    info["intersections"]["intersection_1_1"]["avail_actions"] = [0, 3]

    aligned = ba.align_info(info, hz1x1)

    assert aligned["intersections"]["intersection_1_1"]["avail_actions"] == [0, 3]


def test_align_info_refuses_a_metric_key_that_is_not_there(hz1x1: ba.ScenarioAlignment) -> None:
    """C8: a metric set that silently shrinks is how a MAPPO checkpoint reads different semantics."""
    demanding = _hz1x1_alignment(metric_keys=("a_metric_no_backend_reports",))

    with pytest.raises(ValueError, match="a_metric_no_backend_reports"):
        ba.align_info(_sumo_info(hz1x1), demanding)


def test_align_info_refuses_a_second_application(hz1x1: ba.ScenarioAlignment) -> None:
    """🔒 §3.3 test 4's idempotence CHOICE, documented in the module docstring.

    Detecting and no-opping would hide a double application, which means a caller's model of the
    pipeline is wrong. The precondition it fails on is the 9-wide phase block a canonical info has.
    """
    once = ba.align_info(_sumo_info(hz1x1), hz1x1)

    with pytest.raises((KeyError, ValueError), match="width|already|canonical"):
        ba.align_info(once, hz1x1)


def test_align_info_does_not_mutate_its_input(hz1x1: ba.ScenarioAlignment) -> None:
    """Purity, by deep-compare before and after."""
    info = _sumo_info(hz1x1)
    before = copy.deepcopy(info)

    ba.align_info(info, hz1x1)

    assert info == before


def test_align_info_passes_the_untouched_fields_through_byte_identically(
    hz1x1: ba.ScenarioAlignment,
) -> None:
    """§3.3 test 4's inertness, on a synthetic info; test 4 proper runs it on a live episode."""
    info = _sumo_info(hz1x1)
    aligned = ba.align_info(info, hz1x1)

    for key in ("sim_time", "vehicle_count", "step", "average_travel_time"):
        assert aligned[key] == info[key]
    payload_in = info["intersections"]["intersection_1_1"]
    payload_out = aligned["intersections"]["intersection_1_1"]
    for key in ("time_in_phase", "action_applied", "reward"):
        assert payload_out[key] == payload_in[key]
    assert payload_out["avail_actions"] == payload_in["avail_actions"]


# ----------------------------------------------------------------------
# §3.3 test 4 -- live inertness on a real SUMO episode
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
def test_align_info_is_inert_on_every_decision_step_of_a_real_episode(
    hz1x1: ba.ScenarioAlignment,
) -> None:
    """🔒 §3.3 test 4, on the real env rather than a synthetic dict.

    At every decision step the aligned info must leave the global scalars and the per-intersection
    reward and timing BYTE-IDENTICAL, change only the lane blocks' order and the phase encoding, and
    preserve the multiset of lane values -- a permutation moves values, it does not invent them.
    Short (40 decision steps) because it is an inertness check, not a measurement.
    """
    import numpy as np

    from experiments.envs import make_env
    from offline import parity
    from offline.collect import POLICIES, _build_env_spec
    from offline.sumo_att_reference import collect_style_args

    args = collect_style_args(
        "sumo", "maxpressure", parity.DECLARED_PARITY_SUMOCFG, sentinel_out_dir="/nonexistent"
    )
    env = make_env(_build_env_spec(args))
    try:
        policy = POLICIES["maxpressure"](env, args, np.random.default_rng(1000))
        info = env.reset(seed=1000)
        for _ in range(40):
            aligned = ba.align_info(info, hz1x1)
            ix = hz1x1.intersections["intersection_1_1"]
            source = info["intersections"]["intersection_1_1"]
            target = aligned["intersections"]["intersection_1_1"]

            for key in ("sim_time", "vehicle_count", "step", "average_travel_time"):
                assert aligned[key] == info[key]
            for key in ("time_in_phase", "action_applied", "reward"):
                assert target[key] == source[key]
            assert len(target["state"]) == 25
            # The permutation moves values; it does not create or lose them.
            assert sorted(target["state"][:8]) == sorted(source["state"][:8])
            assert sorted(target["state"][8:16]) == sorted(source["state"][8:16])
            assert sum(target["state"][16:]) == 1.0
            assert target["current_phase"] == ba.cityflow_phase_for_sumo_phase(
                source["current_phase"]
            )
            # And the values land where the correspondence says, checked against the lane dict.
            for index, lane in enumerate(ix.canonical_lanes):
                partner = ix.correspondence[lane]
                assert target["state"][index] == info["lane_vehicle_count"][partner]

            action = policy(info)
            _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
def test_the_derived_sumo_lane_order_equals_the_live_envs() -> None:
    """⭐ The derivation is only trustworthy because it is checked against the env once.

    ``sumo_incoming_lanes_from_net`` reads the ``.net.xml``; ``utils/sumo_utils.py`` builds the same
    list from ``getControlledLinks`` at run time. If they ever diverge, every alignment built without
    a simulator is wrong.
    """
    from offline import parity
    from offline.transfer_gate import intersection_enumeration

    derived = ba.sumo_incoming_lanes_from_net(parity.DECLARED_SOURCE_NET)
    live = intersection_enumeration("sumo", parity.DECLARED_PARITY_SUMOCFG)["incoming_lane_order"]

    assert {k: list(v) for k, v in derived.items()} == {k: list(v) for k, v in live.items()}


# ----------------------------------------------------------------------
# §3.3 test 6 -- hz4x4 gudang, structure only
# ----------------------------------------------------------------------


def test_hz4x4_gudang_resolves_every_intersection() -> None:
    """🔒 §3.3 test 6: 16/16 resolve and each permutation is a proper permutation. No simulation."""
    from offline import parity

    scenario = parity.REPO_ROOT / "scenarios" / "hangzhou_4x4_gudang_18041610_1h"
    alignment = ba.alignment_for_scenario(
        "hangzhou_4x4_gudang",
        cityflow_roadnet=scenario / "roadnet_4X4.json",
        sumo_net=scenario / "hangzhou_4x4_gudang_18041610_1h.net.xml",
    )

    assert len(alignment.intersections) == 16
    for ix in alignment.intersections.values():
        assert len(ix.canonical_lanes) == 12
        assert sorted(ix.permutation) == list(range(12))
        # Every lane pairs with a lane of its OWN road: the correspondence is per road by
        # construction, and this is what makes a 12-lane permutation checkable by eye.
        for lane, partner in ix.correspondence.items():
            assert lane.rsplit("_", 1)[0] == partner.rsplit("_", 1)[0]
    # 128 of the 192 incoming lanes across the 16 intersections denote a DIFFERENT lane in the two
    # files -- the same inversion as hz1x1, at scale.
    inverted = sum(
        1
        for ix in alignment.intersections.values()
        for lane, partner in ix.correspondence.items()
        if lane != partner
    )
    assert inverted == 128


def test_cologne3_is_recorded_as_unalignable_and_not_repaired() -> None:
    """The scope sentence the freeze needs: the correspondence RAISES on all three intersections.

    Recorded with the message, not fixed: SUMO's `t`/`L`/`R` connection directions have no CityFlow
    counterpart and the movement assignments are reversed per lane. `DEFERRED` 74 owns the design
    work; this test pins that it is still open, so the freeze's scope sentence cannot go stale.
    """
    from offline import parity
    from offline.transfer_gate import (
        cityflow_lane_turns,
        lane_semantic_correspondence,
        sumo_lane_turns,
    )

    from utils.cityflow_utils import parse_roadnet

    scenario = parity.REPO_ROOT / "scenarios" / "cologne3"
    roadnet_path = scenario / "cologne3_roadnet_red.json"
    sumo_turns = sumo_lane_turns(scenario / "cologne3.net.xml")
    # ⚠️ The FROZEN parser's filter, not a hand-rolled one: an s2c-converted roadnet marks virtual
    # intersections `gt_virtual`, so a `virtual`-only filter sees 29 intersections here and 3 is
    # the truth (`utils/cityflow_utils.py:80`).
    non_virtual = parse_roadnet(roadnet_path).intersections

    assert len(non_virtual) == 3
    messages: list[str] = []
    for entry in non_virtual:
        turns = cityflow_lane_turns(roadnet_path, entry.id)
        with pytest.raises((ValueError, KeyError), match="matches 0 SUMO lanes|serves no"):
            lane_semantic_correspondence(turns, sumo_turns, sorted(turns))
        try:
            lane_semantic_correspondence(turns, sumo_turns, sorted(turns))
        except (ValueError, KeyError) as exc:
            messages.append(str(exc))

    assert len(messages) == 3
    # The reason, recorded rather than repaired: SUMO's connection directions on this network
    # include `t`, `L` and `R`, which `_CITYFLOW_TO_SUMO_TURN` has no counterpart for, so a lane
    # serving {l, r} in CityFlow matches nothing on the SUMO side.
    assert all("matches 0 SUMO lanes" in message for message in messages)
    assert {"t", "L", "R"} & {
        direction for directions in sumo_turns.values() for direction in directions
    }
