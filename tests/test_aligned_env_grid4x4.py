"""P7.3d C3a (first half) and T-obs: the aligned SUMO env takes a SCENARIO, and the first grid4x4
SUMO episode this project has ever run is a test (``BRIEF_39`` §3 C3a/C3b, §4 T-obs, Amendment A9).

Amendment A9 moves T-obs to the front: *"Assumption 4 (state width 40) is what T-obs exists to
test -- run it before anything is built on it."*  Everything P7.3d builds after this -- the probe,
the calibration, the cell -- stands on three things nobody had run: RESCO's grid4x4 network under
the env's 8-of-16 action rule, the movement-keyed alignment on 16 intersections with the IDENTITY
phase map, and the vehicle- and lane-level observer on 240 lanes (``BRIEF_39`` §0.9: *"that is a
grep, not a proof"*).

GATES, each naming the artifact it consumes
-------------------------------------------
* ``RLTRAFFIC_GRID4X4_RESCO`` -- the read-only candidates ROOT (no default; unset means skip).
* SUMO: ``traci`` importable and the ``sumo`` binary on ``PATH``.
* ``RLTRAFFIC_DRAWS`` (default: this tree's ``scenarios/draws``) holding P7.3d C1's parity
  configuration for grid4x4 **draw 1000** -- the draw T-obs runs, named at the decorator.

This file rolls **one** full grid4x4 SUMO episode (T-obs), of the six the whole suite may hold.
No travel time, reward or return is printed or compared with anything: ``e_sumo`` is asserted
finite and re-derivable, and its value belongs to C4's reference cells, not to a test log.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline import parity
from offline.aligned_env import (
    DECLARED_SCENARIO,
    AlignedEnv,
    aligned_observer_env_for_draw,
    alignment_for_scenario_key,
    declared_alignment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID4X4_KEY = "cityflow_grid4x4"
HZ1X1_KEY = "cityflow1x1"
T_OBS_DRAW = 1000

#: grid4x4's shape, typed here and never imported: 16 intersections, 80 roads x 3 lanes = 240 lanes,
#: 12 incoming lanes and 16 file phases per intersection (A15(g)), so an aligned state is
#: ``2 * 12 + 16 = 40`` wide -- the registered subject's ``state_dim`` -- and the env exposes 8 of
#: the 16 phases as actions (``envs/phase_control.py``'s transition rule).
N_INTERSECTIONS = 16
N_LANES = 240
N_INCOMING_LANES = 12
#: What the registered halting cross-check COVERS: the monitored INCOMING lanes, 16 x 12 = 192 --
#: not the network's 240 (48 boundary-outbound lanes enter no controlled intersection).
N_MONITORED_INCOMING_LANES = N_INTERSECTIONS * N_INCOMING_LANES
N_FILE_PHASES = 16
STATE_WIDTH = 40
N_ACTIONS = 8
HORIZON_SECONDS = 3600
N_DECISIONS = 360


def _sumo_available() -> bool:
    """Both halves: the Python bindings and the binary the env actually launches."""
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def _draws_root() -> Path:
    value = os.environ.get("RLTRAFFIC_DRAWS")
    return Path(value) if value else REPO_ROOT / "scenarios/draws"


def _grid4x4_parity_available(draw_id: int) -> bool:
    """Is P7.3d C1's grid4x4 parity configuration present for **this** draw?"""
    from offline.materialise_draws import parity_sumocfg_path

    return parity_sumocfg_path(GRID4X4_KEY, int(draw_id), out_root=_draws_root()).is_file()


def _resco_set() -> bool:
    value = os.environ.get("RLTRAFFIC_GRID4X4_RESCO")
    return bool(value) and (
        Path(value) / "resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml"
    ).is_file()


needs_resco = pytest.mark.skipif(
    not _resco_set(),
    reason="RLTRAFFIC_GRID4X4_RESCO is unset (or RESCO's grid4x4.net.xml is not under it): point "
    "it at the read-only, gitignored candidates root to build grid4x4's alignment",
)


# ==================================================================================
# T-regress (c), aligned-env half: the hangzhou key resolves to today's alignment
# ==================================================================================
def test_the_hz1x1_key_resolves_to_todays_declared_alignment() -> None:
    """*Mutation this is built against:* the default scenario changed, or the hangzhou key routed
    through the generic branch -- either makes this alignment differ from ``declared_alignment()``."""
    assert DECLARED_SCENARIO == "hangzhou_1x1_bc-tyc"
    resolved = alignment_for_scenario_key(HZ1X1_KEY)
    assert resolved == declared_alignment()
    assert resolved.scenario == "hangzhou_1x1_bc-tyc"
    assert resolved.phase_map_is_hangzhou_shaped is True
    assert list(resolved.intersections) == ["intersection_1_1"]
    assert resolved.intersections["intersection_1_1"].canonical_state_width() == 25
    assert alignment_for_scenario_key(HZ1X1_KEY, ("queue_length",)).metric_keys == ("queue_length",)


def test_an_unknown_scenario_key_has_no_alignment_and_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="cityflow_cologne3"):
        alignment_for_scenario_key("cityflow_cologne3")


def test_grid4x4s_alignment_is_refused_naming_the_variable_when_it_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RLTRAFFIC_GRID4X4_RESCO", raising=False)
    with pytest.raises(ValueError, match="RLTRAFFIC_GRID4X4_RESCO is unset"):
        alignment_for_scenario_key(GRID4X4_KEY)


@needs_resco
def test_grid4x4s_alignment_is_the_identity_phase_map_on_16_intersections_at_width_40() -> None:
    """A15(g)'s D verdict as the adapter consumes it: 16 <-> 16 phases, so the map is the IDENTITY
    (``cityflow_phase_for_sumo_phase``'s docstring says the hangzhou map is WRONG here), and the
    action count is the env's 8, not the 16 phases with a non-empty roadlink list."""
    from utils.cityflow_utils import parse_roadnet

    alignment = alignment_for_scenario_key(GRID4X4_KEY)

    roadnet = REPO_ROOT / "scenarios/grid4x4/grid4x4_roadnet_red.json"
    corpus_order = [ix.id for ix in parse_roadnet(roadnet).intersections]
    assert list(alignment.intersections) == corpus_order, "the CityFlow corpus's order, not sorted"
    assert len(corpus_order) == N_INTERSECTIONS
    assert alignment.scenario == "grid4x4"
    assert alignment.phase_map_is_hangzhou_shaped is False
    assert len(alignment.known_lanes) == N_LANES

    same_id = 0
    for ix_id, ix in alignment.intersections.items():
        assert ix.cityflow_num_phases == ix.sumo_num_phases == N_FILE_PHASES, ix_id
        assert ix.canonical_state_width() == ix.sumo_state_width() == STATE_WIDTH, ix_id
        assert ix.n_actions == N_ACTIONS, ix_id
        assert set(ix.correspondence) == set(ix.canonical_lanes), ix_id
        # THE MOVEMENT KEY DOES REAL WORK HERE, exactly as on hangzhou: the lane INDEX is reversed
        # between the two files (CityFlow counts a road's lanes left-to-right, SUMO right-to-left),
        # so `X_0` and `X_2` swap and only the middle lane keeps its id.  Pairing by id would
        # compare a right-turn queue with a left-turn queue on 8 of every 12 lanes -- P7.0's defect.
        for cityflow_lane, sumo_lane in ix.correspondence.items():
            road, index = cityflow_lane.rsplit("_", 1)
            assert sumo_lane == f"{road}_{2 - int(index)}", (ix_id, cityflow_lane, sumo_lane)
            same_id += int(cityflow_lane == sumo_lane)
        # ... and the PERMUTATION is nevertheless the identity, on all 16, as MEASURED from the two
        # files (2026-09-19): the CityFlow corpus's discovery order runs `_2, _1, _0` within a road
        # and SUMO's is sorted `_0, _1, _2`, so the two reversals cancel position by position.  I
        # first asserted the opposite by analogy with hangzhou ((0, 1, 2, 3, 6, 7, 5, 4) there) and
        # the files said otherwise.  Stated as a fact so that a change in either order fails here.
        assert list(ix.canonical_lanes) != sorted(ix.canonical_lanes), ix_id
        assert ix.permutation == tuple(range(N_INCOMING_LANES)), ix_id
    assert same_id == 4 * N_INTERSECTIONS, "only the middle lane of each incoming road keeps its id"


# ==================================================================================
# T-obs -- the first grid4x4 SUMO episode this project has ever run
# ==================================================================================
@needs_resco
@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
@pytest.mark.skipif(
    not _grid4x4_parity_available(T_OBS_DRAW),
    reason=f"P7.3d C1's grid4x4 parity configuration for draw {T_OBS_DRAW} is not under "
    "RLTRAFFIC_DRAWS (default: this tree's scenarios/draws); it is gitignored",
)
def test_t_obs_one_fixed_time_episode_on_draw_1000_reconstructs_on_16_intersections() -> None:
    """T-obs (load-bearing for ``BRIEF_39`` §0.9).  One fixed-time episode through
    ``aligned_observer_env_for_draw`` with the halting cross-check ON.

    *Mutation this is built against:* the halting check declared ON but covering a hangzhou-sized
    lane set (its 8 incoming lanes) -> the lane-second assertion dies.

    ⚠️ **BRIEF vs REPO, flagged in the packet.**  ``BRIEF_39`` specifies this count as
    ``240 x 3,600`` and the mutation as *"hz1x1's 16"*.  Both are TOTAL lane counts.  The registered
    instrument's cross-check iterates ``_monitored_incoming_lanes``
    (``offline/sumo_att_reference.py``), and its committed record says what that means: all 47
    halting-checked cells of ``docs/data/p7_3a_zero_shot.json`` carry ``halting_n_lane_seconds
    28,800 = 8 x 3,600`` -- hangzhou's 8 INCOMING lanes, not its 16.  On grid4x4 that is
    ``16 x 12 = 192`` lanes, and the first run of this test measured exactly ``691,200``.  The repo
    wins (``BRIEF_39`` header); widening the check to 240 lanes would be a new instrument
    (``DEFERRED`` 83, out of scope).  The count is asserted THREE ways so none can drift alone:
    derived from the env's own intersections, as the literal, and as NOT a hangzhou-sized one.
    """
    from offline.horizon_metric import horizon_rollout
    from offline.materialise_draws import parity_sumocfg_path
    from offline.sumo_att_reference import (
        build_policy,
        collect_style_args,
        reconstruct_sumo_episode,
    )

    config_path = parity_sumocfg_path(GRID4X4_KEY, T_OBS_DRAW, out_root=_draws_root())
    env = aligned_observer_env_for_draw(
        GRID4X4_KEY, T_OBS_DRAW, out_root=_draws_root(), halting_check=True, arm="fixedtime"
    )
    seen: dict[str, Any] = {"widths": set(), "n_intersections": set(), "decisions": 0, "actions": []}
    try:
        assert isinstance(env, AlignedEnv)
        ids = [str(ix.id) for ix in env.intersections]
        assert len(ids) == N_INTERSECTIONS
        incoming_per_intersection = [len(list(ix.incoming_lanes)) for ix in env.intersections]

        args = collect_style_args(
            "sumo", "fixedtime", config_path, episodes=1, base_seed=1000,
            sentinel_out_dir="/nonexistent",
        )
        policy = build_policy(env, args)

        def choose(_env: Any, info: dict[str, Any]) -> Any:
            payloads = info["intersections"]
            seen["n_intersections"].add(len(payloads))
            assert list(payloads) == ids, "C1: ordered by [ix.id for ix in env.intersections]"
            seen["widths"].update(len(payloads[ix_id]["state"]) for ix_id in ids)
            action = np.asarray(policy(info), dtype=np.int64).reshape(-1)
            seen["actions"].append(action)
            seen["decisions"] += 1
            return action

        horizon_rollout(env, choose, 1, 1000)
        built = reconstruct_sumo_episode(env.recorder)
        types_seen = {env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()}
        option = str(env._sumo.simulation.getOption("time-to-teleport"))
    finally:
        env.close()

    # --- the aligned frame, on every decision ---------------------------------------
    assert seen["n_intersections"] == {N_INTERSECTIONS}
    assert seen["widths"] == {STATE_WIDTH}, "assumption 4: the registered subject's state_dim"
    assert seen["decisions"] == N_DECISIONS
    actions = np.stack(seen["actions"])
    assert actions.shape == (N_DECISIONS, N_INTERSECTIONS)
    assert int(actions.min()) >= 0 and int(actions.max()) < N_ACTIONS

    # --- the observer -------------------------------------------------------------------
    assert built.n_observations == HORIZON_SECONDS
    assert built.horizon == float(HORIZON_SECONDS)
    assert int(built.n_teleports) == 0
    assert types_seen == {"cf_parity"}
    assert option == "-1"
    assert math.isfinite(built.e_sumo.value) and built.e_sumo.n_ids > 0
    assert built.e_sumo.value == built.e_sumo.total / built.e_sumo.n_ids, "re-derived, not trusted"
    assert built.n_intended >= built.n_departed >= built.n_arrived >= 0

    # --- the halting cross-check, ON, over EVERY monitored incoming lane for ALL 3,600 seconds
    assert incoming_per_intersection == [N_INCOMING_LANES] * N_INTERSECTIONS
    assert built.halting.n_seconds == HORIZON_SECONDS
    assert built.halting.n_lane_seconds == sum(incoming_per_intersection) * HORIZON_SECONDS
    assert built.halting.n_lane_seconds == N_MONITORED_INCOMING_LANES * HORIZON_SECONDS == 691_200
    assert built.halting.n_lane_seconds != 8 * HORIZON_SECONDS, "not a hangzhou-sized lane set"
    assert built.halting.n_missing == 0
    assert built.halting.n_disagreeing_lane_seconds == 0
    assert built.halting.max_abs_difference == 0


def test_the_scenarios_this_file_names_are_the_registered_parity_scenarios() -> None:
    """The keys above are the draws-tree keys C1 registered; a rename there must fail here."""
    assert parity.scenario_for_key(GRID4X4_KEY).external is not None
    assert parity.scenario_for_key(HZ1X1_KEY).external is None
