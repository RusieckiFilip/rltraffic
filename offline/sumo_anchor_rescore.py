"""P9.2 (``BRIEF_47`` §3.2-3.3): A27(f)'s aligned reference arms, rolled, and every transferred-DT rho re-scored beside the record.

Cell format version: ``p9.2-anchor-cell/1.0`` (:data:`CELL_FORMAT_VERSION`), one JSON file per cell under the run's
work directory.  Artifact format version: ``p9.2-sumo-anchors/1.0`` (:data:`ARTIFACT_FORMAT_VERSION`),
``docs/data/p9_2_sumo_anchors.json``.

**Label (A27(a), item 5): anchor-alignment control added after results, the registered rho and the H3 verdicts staying
the record.**  Nothing here changes a registered rho, verdict, artifact or anchor; every new value is reported BESIDE its
registered counterpart.

WHAT A CELL IS, AND ITS ALIGNMENT CONVENTION
--------------------------------------------
A cell is ``(backend, scenario, arm, draw_id)``: one episode of one reference arm on one held-out draw (1000-1099, D4)
from a fresh env at ``reset(seed=1000)`` (A18(c)).  ``att_*`` / ``e_sumo`` are the HORIZON values of that episode
(A1, A15).  ``actions[t]`` is the action decision ``t`` applied -- one ``int`` per decision on hz1x1, one row of ``int`` in
``intersection_ids`` order on grid4x4, as the registered anchor chunks record them; ``action_sequence_sha256`` is
``nortg_decomposition.action_sequence_sha256`` over that array.

* **SUMO** (``backend: "sumo"``): :func:`sumo_anchor_episode` -- the ANCHOR branch of ``transfer_curve.run_cell``, line for
  line, with the policy pluggable: the observed but UNWRAPPED env of ``observer_env_for_draw`` (A2: an anchor never goes
  through A16's door) built with the BASE arm's collection arguments, the halting cross-check on draw 1000 only (C2),
  ``_StepTap``, ``horizon_rollout(tap, choose, 1, 1000)``, ``reconstruct_sumo_episode``, and on grid4x4 A23's collision
  record and the per-intersection return by its two routes.  The frozen regime is A15's (vType parity, teleport-free) with
  A23's rule: on hz1x1 any teleport refuses the cell; on grid4x4 a collision is recorded and the cell kept, and a
  teleport no same-step collision explains refuses it -- and a refusal STOPS the run for a ruling.  ``run_cell`` itself
  cannot be reused: ``validate_cell_payload`` refuses any arm outside P7.3's declared set (``transfer_curve.py:2474``).
  The replica is proven to be the registered route by giving it the REGISTERED policies and reproducing committed anchor
  records under ``==`` (:func:`replica_gate`; ``tests/test_sumo_anchor_rescore.py``).
* **CityFlow** (``backend: "cityflow"``): :func:`cityflow_anchor_episode` -- ``admission_probe.probe_episode``, the
  per-episode body of P8.4b's ``att_rederivation.run_campaign``: ``att_ours`` from the horizon reader, ``att_engine`` from the
  engine at the horizon, and A11(b)'s companions, with the MaxPressure tier's own env settings.

THE TWO ARMS (A27(f))
---------------------
* ``maxpressure_1to1`` -- :class:`offline.policies.max_pressure_1to1.MaxPressure1to1Agent`, on both backends.
* ``fixedtime_kcf`` -- SUMO only: ``offline.policies.fixed_time.make_fixedtime`` through ``sumo_att_reference.build_policy``
  with the args ``collect_style_args`` builds for the registered k = 4 anchor (``transfer_curve.anchor_choose``), ``fixed_time_k``
  replaced by CityFlow's tuned k (:func:`kcf_for`: the CityFlow tier manifest's recorded ``fixed_time_k``, refused unless it
  is ``docs/data/fixed_time_sweep/README.md``'s chosen value).  On SUMO the schedule source is the equal split, exactly as
  for k = 4 (``resolve_plan`` cannot read a ``.sumocfg`` as a CityFlow config).

``departLane`` is set nowhere in the parity route files (``DEFERRED`` 116); every SUMO cell records the measured count of
``departLane`` attributes in its routes file, and the artifact names SUMO's default as a limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "BASE_ARM",
    "CELL_FORMAT_VERSION",
    "CITYFLOW_ARMS",
    "KCF_README_VALUES",
    "REPLICA_COMPARED_FIELDS",
    "REPLICA_DRAW",
    "SCENARIOS",
    "SUMO_ARMS",
    "anchor_status",
    "assert_work_dir",
    "cell_name",
    "cityflow_anchor_episode",
    "compare_with_registered",
    "declared_anchor_cells",
    "fixedtime_action_schedule",
    "kcf_for",
    "main",
    "replica_gate",
    "roll_cells",
    "sumo_anchor_episode",
    "sumo_policy",
    "validate_anchor_cell",
]

CELL_FORMAT_VERSION = "p9.2-anchor-cell/1.0"
ARTIFACT_FORMAT_VERSION = "p9.2-sumo-anchors/1.0"

#: The two scenarios of A27(f): ``transfer_curve.SCENARIO_KEY`` and ``transfer_curve.GRID4X4_SCENARIO_KEY``.
SCENARIOS: tuple[str, ...] = ("cityflow1x1", "cityflow_grid4x4")

#: A27(f): the two aligned arms on SUMO, and the set-semantics MaxPressure on CityFlow too.
SUMO_ARMS: tuple[str, ...] = ("maxpressure_1to1", "fixedtime_kcf")
CITYFLOW_ARMS: tuple[str, ...] = ("maxpressure_1to1",)

#: The registered arm whose collection arguments (and therefore whose env) a cell is built with.  A new arm differs from
#: its base in the policy and in nothing else.
BASE_ARM: Mapping[str, str] = {
    "maxpressure_1to1": "maxpressure",
    "fixedtime_kcf": "fixedtime",
    "maxpressure": "maxpressure",
    "fixedtime": "fixedtime",
}

#: CityFlow's tuned k (``docs/data/fixed_time_sweep/README.md``, chosen values; D5's protocol on draws 1-5).
KCF_README_VALUES: Mapping[str, int] = {"cityflow1x1": 6, "cityflow_grid4x4": 1}

#: The CityFlow fixed-time tier whose manifest records the k the CityFlow anchor ran at.
KCF_TIER_DIRS: Mapping[str, str] = {"cityflow1x1": "cf_hz1x1__fixedtime", "cityflow_grid4x4": "cf_grid4x4__fixedtime"}

#: P8.4b's scenario names for the two keys (``att_rederivation.run_campaign``'s map, inverted).
PROBE_SCENARIO: Mapping[str, str] = {"cityflow1x1": "hz1x1", "cityflow_grid4x4": "grid4x4"}

#: The held-out pool (D4) and the engine seed of every held-out episode (A18(c)).
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))
ENGINE_SEED = 1000

#: The replica gate's draw: not 1000, so the gate costs one plain episode per arm and scenario (draw 1000 carries the
#: 3.5x halting cross-check, Amendment C2).
REPLICA_DRAW = 1001

#: Every MEASUREMENT a registered anchor record carries (``docs/data/p7_3a_zero_shot.json``'s anchor rows, whose keys
#: ``tests/test_sumo_anchor_rescore.py`` re-derives); the replica must reproduce each under ``==``.  On grid4x4 the gate
#: adds ``intersection_ids`` and ``local_return``.
REPLICA_COMPARED_FIELDS: tuple[str, ...] = (
    "action_space_n", "att_env", "att_horizon", "att_running_mean", "config_sha256", "decisions", "e_sumo",
    "engine_seed_drawn", "engine_seed_requested", "episode_reward", "halting_checked", "halting_max_abs_difference",
    "halting_n_disagreeing_lane_seconds", "halting_n_lane_seconds", "horizon_vehicle_count",
    "max_abs_depart_clock_deviation", "mean_depart_delay", "n_arrived_never_observed_at_a_boundary", "n_created",
    "n_entered", "n_never_entered", "n_pending_at_horizon", "n_teleports", "n_vanished_without_arrival", "p_sumo",
    "routes_sha256", "time_to_teleport_option", "vehicle_types_seen", "w_sumo",
)
GRID_REPLICA_EXTRA_FIELDS: tuple[str, ...] = ("intersection_ids", "local_return")

#: The committed artifact each scenario's registered SUMO anchors are read from (P7.3b's and P7.3c's stage-1 anchors are
#: equal to these on every record, measured in the plan, V3).
REGISTERED_ANCHOR_ARTIFACT: Mapping[str, str] = {
    "cityflow1x1": "p7_3a_zero_shot.json",
    "cityflow_grid4x4": "p7_3d_grid4x4.json",
}

REPLICA_GATE_NAME = "replica_gate.json"
REPLICA_GATE_FORMAT_VERSION = "p9.2-replica-gate/1.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRAWS_ROOT = Path("/home/filip/rltraffic/scenarios/draws")
DEFAULT_OUTPUT_ROOT = Path("/home/filip/rltraffic/output")
DEFAULT_CORPUS_ROOT = Path("/home/filip/rltraffic/datasets_v11")
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_ROOT / "p9_2" / "anchors"
DEFAULT_DATA_DIR = _REPO_ROOT / "docs" / "data"
_HEX40 = re.compile(r"[0-9a-f]{40}")


# ----------------------------------------------------------------------
# The declaration
# ----------------------------------------------------------------------


def declared_anchor_cells() -> list[dict[str, Any]]:
    """A27(f)'s 600 cells: SUMO {hz1x1, grid4x4} x {``maxpressure_1to1``, ``fixedtime_kcf``} x draws, then CityFlow
    {hz1x1, grid4x4} x {``maxpressure_1to1``} x draws -- one episode per draw, as the registered anchors were rolled."""
    cells = [
        {"backend": "sumo", "scenario": scenario, "arm": arm, "draw_id": int(draw)}
        for scenario in SCENARIOS
        for arm in SUMO_ARMS
        for draw in HELD_OUT_DRAWS
    ]
    cells += [
        {"backend": "cityflow", "scenario": scenario, "arm": arm, "draw_id": int(draw)}
        for scenario in SCENARIOS
        for arm in CITYFLOW_ARMS
        for draw in HELD_OUT_DRAWS
    ]
    return cells


def cell_name(cell: Mapping[str, Any]) -> str:
    """One cell, one file; the name carries the whole identity."""
    return f"cell_{cell['backend']}_{cell['scenario']}_{cell['arm']}_draw{int(cell['draw_id']):04d}.json"


# ----------------------------------------------------------------------
# kcf and the policies
# ----------------------------------------------------------------------


def kcf_for(scenario: str, corpus_root: str | Path) -> dict[str, Any]:
    """CityFlow's tuned k for *scenario*: the fixed-time tier manifest's recorded ``fixed_time_k``, which IS the k the
    registered CityFlow anchor ran at (``method_tier_grid.fixedtime_collection_settings``), refused unless it equals the
    tuning README's chosen value."""
    from offline.method_tier_grid import fixedtime_collection_settings

    if scenario not in KCF_TIER_DIRS:
        raise ValueError(f"{scenario!r} is not a scenario of this task ({list(KCF_TIER_DIRS)})")
    manifest = Path(corpus_root) / KCF_TIER_DIRS[scenario] / "manifest.json"
    collected = fixedtime_collection_settings(manifest)
    k = int(collected["fixed_time_k"])
    if k != KCF_README_VALUES[scenario]:
        raise ValueError(
            f"{manifest} records fixed_time_k {k}, but the tuning README chose k = {KCF_README_VALUES[scenario]} for "
            f"{scenario}; refusing to call either 'CityFlow's tuned k' while they disagree"
        )
    return {
        "fixed_time_k": k,
        "manifest": str(manifest),
        "cityflow_schedule_source": collected["fixed_time_schedule_source"],
        "cityflow_plan_sha256": collected["fixed_time_plan_sha256"],
    }


def sumo_policy(
    env: Any, *, arm: str, config_path: str | Path, kcf: int | None
) -> tuple[Callable[[dict[str, Any]], Any], dict[str, Any]]:
    """The policy of one SUMO anchor arm, built on the UNTAPPED env as ``transfer_curve.anchor_choose`` builds it.

    The namespace is ``collect_style_args("sumo", <base arm>, cfg, episodes=1, base_seed=1000)`` -- the registered anchor's
    own -- so the registered arms ARE ``anchor_choose``'s policies (``build_policy``: ``collect.POLICIES[arm](env, args,
    default_rng(1000))``).  ``fixedtime_kcf`` changes ``fixed_time_k`` and nothing else; ``maxpressure_1to1`` swaps the
    frozen agent for the variant.  Returns the policy and its provenance.
    """
    import numpy as np

    from offline.policies.fixed_time import FixedTimeController, fixed_time_provenance
    from offline.policies.max_pressure_1to1 import (
        make_maxpressure_1to1,
        phase_multiplicities,
        roadlink_structure_digest,
    )
    from offline.sumo_att_reference import build_policy, collect_style_args

    if arm not in BASE_ARM:
        raise ValueError(f"{arm!r} is not an arm this module rolls ({sorted(BASE_ARM)})")
    base = BASE_ARM[arm]
    args = collect_style_args(
        "sumo", base, config_path, episodes=1, base_seed=ENGINE_SEED, sentinel_out_dir="/nonexistent"
    )
    if arm == "fixedtime_kcf":
        if kcf is None:
            raise ValueError("fixedtime_kcf needs CityFlow's tuned k (kcf_for); none was given")
        args.fixed_time_k = int(kcf)
    elif kcf is not None:
        raise ValueError(f"{arm} takes no kcf; only fixedtime_kcf replaces the registered k")

    intersections = list(env.intersections)
    if arm == "maxpressure_1to1":
        policy = make_maxpressure_1to1(env, args, np.random.default_rng(ENGINE_SEED))
        provenance: dict[str, Any] = {
            "arm": arm,
            "base_arm": base,
            "roadlink_structure_digest": roadlink_structure_digest(intersections),
            "multiplicities": phase_multiplicities(intersections),
        }
        return policy, provenance

    policy = build_policy(env, args)
    provenance = {"arm": arm, "base_arm": base, "roadlink_structure_digest": roadlink_structure_digest(intersections)}
    if base == "fixedtime":
        controller = getattr(policy, "__self__", None)
        if not isinstance(controller, FixedTimeController):
            raise TypeError(
                f"the fixed-time policy is a {type(controller).__name__}, not a FixedTimeController; the schedule this "
                "cell ran cannot be recorded"
            )
        provenance.update(fixed_time_provenance(args))
        provenance["cycle"] = {str(ix.id): list(controller.cycle_for(str(ix.id))) for ix in intersections}
    return policy, provenance


def fixedtime_action_schedule(
    policy: Callable[[dict[str, Any]], Any], ix_ids: Sequence[str], steps: int
) -> dict[str, list[int]]:
    """The action a fixed-time policy emits at decision steps ``0 .. steps - 1``, per intersection id.

    The controller is a pure function of ``info["step"]`` (``offline/policies/fixed_time.py``); with no legal-action mask
    in the info the scheduled action is returned as is, so this is the schedule itself, compared across backends by id.
    """
    import numpy as np

    schedule: dict[str, list[int]] = {str(ix): [] for ix in ix_ids}
    for step in range(int(steps)):
        actions = np.asarray(policy({"step": step, "intersections": {}})).reshape(-1)
        if len(actions) != len(ix_ids):
            raise ValueError(f"the policy returned {len(actions)} actions for {len(ix_ids)} intersections")
        for ix, action in zip(ix_ids, actions):
            schedule[str(ix)].append(int(action))
    return schedule


# ----------------------------------------------------------------------
# One SUMO episode: run_cell's anchor branch, the policy pluggable
# ----------------------------------------------------------------------


def _construction_cell(scenario: str, draw_id: int, arm: str) -> dict[str, Any]:
    """The cell ``transfer_curve``'s env helpers read: kind ``anchor``, the BASE arm, and the scenario key only on the grid
    (an hz1x1 P7.3 cell carries none, and ``scenario_of`` resolves its absence to hz1x1)."""
    import offline.transfer_curve as tc

    cell: dict[str, Any] = {"kind": "anchor", "subject": None, "arm": BASE_ARM[arm], "seed": None,
                            "draw_id": int(draw_id)}
    if scenario == tc.GRID4X4_SCENARIO_KEY:
        cell["scenario"] = scenario
    elif scenario != tc.SCENARIO_KEY:
        raise ValueError(f"{scenario!r} is not a scenario of this task ({list(SCENARIOS)})")
    return cell


def _git_provenance() -> dict[str, Any]:
    import offline.transfer_curve as tc

    return tc._git_provenance()


def sumo_anchor_episode(
    scenario: str, draw_id: int, arm: str, *, draws_root: str | Path, corpus_root: str | Path
) -> dict[str, Any]:
    """One observed SUMO episode of one anchor arm on one held-out draw, as a payload.  It writes NOTHING.

    The order is ``transfer_curve.run_cell``'s: the demand is pinned BEFORE a simulator starts
    (``demand_identity``), the env is built and refused if it is the wrong shape for an anchor
    (``assert_env_matches_cell``), the policy is built on the untapped env, the rollout runs on the tap, and
    ``att_env`` and ``att_horizon`` -- two routes to one quantity -- must agree under ``==``.
    """
    import offline.transfer_curve as tc
    from agent.utils.utils import Utils
    from offline.horizon_metric import horizon_rollout
    from offline.nortg_decomposition import action_sequence_sha256
    from offline.sumo_att_reference import reconstruct_sumo_episode

    cell = _construction_cell(scenario, draw_id, arm)
    grid = scenario == tc.GRID4X4_SCENARIO_KEY
    demand = tc.demand_identity(int(draw_id), out_root=draws_root, scenario=scenario)
    routes_path = Path(demand["config_path"]).parent / tc.PARITY_ROUTES_NAME
    depart_lane_attributes = routes_path.read_text(encoding="utf-8").count("departLane")
    kcf = kcf_for(scenario, corpus_root)["fixed_time_k"] if arm == "fixedtime_kcf" else None

    started = time.perf_counter()
    env = tc.env_for_cell(cell, out_root=draws_root)
    try:
        tc.assert_env_matches_cell(cell, env)
        intersections = list(env.intersections)
        action_counts = Utils.infer_action_counts(getattr(env, "action_space", None), intersections)
        n_actions = int(min(action_counts))
        policy, provenance = sumo_policy(env, arm=arm, config_path=demand["config_path"], kcf=kcf)
        actions: list[Any] = []

        def choose(_env: Any, info: Mapping[str, Any]) -> Any:
            import numpy as np

            action = policy(dict(info))
            flat = np.asarray(action).reshape(-1)
            actions.append([int(a) for a in flat] if grid else int(flat[0]))
            return action

        tap = tc._StepTap(env, collect_post_step=grid)
        rollout = horizon_rollout(tap, choose, 1, ENGINE_SEED)
        built = reconstruct_sumo_episode(env.recorder)
        collision_record = env.recorder.collision_record() if grid else None
        att_env = tc.att_env_from_info(tap.last_info or {})
        types_seen = sorted({env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()})
        option = str(env._sumo.simulation.getOption("time-to-teleport"))
        engine_seed_drawn = int(env._engine_seed)
        return_lanes = tc.grid4x4_return_lanes(intersections, tap.post_step[0]) if grid and tap.post_step else None
    finally:
        env.close()
    seconds = time.perf_counter() - started

    if att_env != rollout.att_horizon:
        raise ValueError(
            f"att_env {att_env!r} and att_horizon {rollout.att_horizon!r} disagree; they are two routes to one quantity, "
            "so the loop this cell ran is not the loop the registered anchors were measured by"
        )
    ix_ids = [str(ix.id) for ix in intersections]
    checked = tc.halting_check_for(int(draw_id))
    payload: dict[str, Any] = {
        "format_version": CELL_FORMAT_VERSION,
        "backend": "sumo",
        "scenario": scenario,
        "arm": arm,
        "base_arm": BASE_ARM[arm],
        "draw_id": int(draw_id),
        "kind": "anchor",
        "engine_seed_requested": ENGINE_SEED,
        "engine_seed_drawn": engine_seed_drawn,
        "decisions": len(actions),
        "actions": actions,
        "actions_in_range": bool(actions) and all(
            (len(row) == len(ix_ids) and all(0 <= int(a) < n_actions for a in row)) if grid else 0 <= int(row) < n_actions
            for row in actions
        ),
        "action_space_n": n_actions,
        "action_sequence_sha256": action_sequence_sha256(actions),
        "episode_reward": rollout.episode_reward,
        "att_horizon": rollout.att_horizon,
        "att_env": att_env,
        "att_running_mean": rollout.att_running_mean,
        "horizon_vehicle_count": rollout.final_vehicle_count,
        "e_sumo": built.e_sumo.value,
        "p_sumo": built.p_sumo.value,
        "w_sumo": built.w_sumo.value,
        "mean_depart_delay": built.mean_depart_delay,
        "n_created": built.e_sumo.n_ids,
        "n_entered": built.n_departed,
        "n_never_entered": built.n_never_inserted,
        "n_pending_at_horizon": built.n_pending_at_horizon,
        "n_teleports": built.n_teleports,
        "n_vanished_without_arrival": built.n_vanished_without_arrival,
        "n_arrived_never_observed_at_a_boundary": built.n_arrived_never_observed_at_a_boundary,
        "max_abs_depart_clock_deviation": built.max_abs_depart_clock_deviation,
        "n_observations": built.n_observations,
        "vehicle_types_seen": types_seen,
        "time_to_teleport_option": option,
        "halting_checked": checked,
        "halting_max_abs_difference": built.halting.max_abs_difference if checked else None,
        "halting_n_lane_seconds": built.halting.n_lane_seconds if checked else None,
        "halting_n_disagreeing_lane_seconds": built.halting.n_disagreeing_lane_seconds if checked else None,
        "config_sha256": demand["config_sha256"],
        "routes_sha256": demand["routes_sha256"],
        "depart_lane_attributes": depart_lane_attributes,
        "policy": provenance,
        "seconds": seconds,
        **_git_provenance(),
    }
    if grid:
        if return_lanes is None or len(tap.post_step) != len(actions):
            raise ValueError(
                f"{len(tap.post_step)} post-step info(s) for {len(actions)} decision(s); the per-intersection return is "
                "the sum over one post-step info per decision (B.7.1-2)"
            )
        by_reward, by_lanes = tc.per_intersection_local_returns(tap.post_step, return_lanes)
        payload["intersection_ids"] = ix_ids
        payload["local_return"] = {ix: float(by_reward[ix]) for ix in ix_ids}
        payload["local_return_from_lanes"] = {ix: float(by_lanes[ix]) for ix in ix_ids}
        payload.update({key: collision_record[key] for key in tc.GRID4X4_COLLISION_RECORD_FIELDS})
    return payload


# ----------------------------------------------------------------------
# One CityFlow episode: P8.4b's route
# ----------------------------------------------------------------------


def cityflow_anchor_episode(scenario: str, draw_id: int, arm: str, *, roots: Any) -> dict[str, Any]:
    """One CityFlow episode of ``maxpressure_1to1`` (or, for an identity check, the registered ``maxpressure``) on one
    held-out draw, by ``admission_probe.probe_episode`` with the MaxPressure tier's own env settings
    (``att_rederivation.rederivation_env_settings``) -- so the arm and the registered MaxPressure cell differ in the
    policy and in nothing else.  It writes NOTHING."""
    from offline.admission_probe import created_from_flow, probe_episode
    from offline.att_rederivation import rederivation_env_settings
    from offline.dt_gate import _maxpressure_factory
    from offline.materialise_draws import draw_config_path
    from offline.policies.max_pressure_1to1 import (
        maxpressure_1to1_factory,
        phase_multiplicities,
        roadlink_structure_digest,
    )

    if scenario not in PROBE_SCENARIO:
        raise ValueError(f"{scenario!r} is not a scenario of this task ({list(SCENARIOS)})")
    if arm not in ("maxpressure_1to1", "maxpressure"):
        raise ValueError(f"{arm!r} is not rolled on CityFlow by this task")
    probe_scenario = PROBE_SCENARIO[scenario]
    settings = rederivation_env_settings(probe_scenario, "maxpressure", roots)
    horizon = int(settings["max_steps"]) * int(settings["delta_time"])
    config = Path(draw_config_path(scenario, int(draw_id), out_root=roots.draws_root))
    created = created_from_flow(config.parent / "flow.json", horizon_seconds=horizon)
    inner = maxpressure_1to1_factory if arm == "maxpressure_1to1" else _maxpressure_factory
    structure: dict[str, Any] = {}

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], Any]:
        intersections = list(env.intersections)
        structure["roadlink_structure_digest"] = roadlink_structure_digest(intersections)
        structure["multiplicities"] = phase_multiplicities(intersections)
        return inner(env)

    episode = probe_episode(
        scenario=probe_scenario,
        tier="maxpressure",
        method=arm,
        arm=arm,
        seed=None,
        draw_id=int(draw_id),
        config_path=config,
        env_settings=settings,
        scenario_id=scenario,
        choose_action_factory=factory,
        engine_seed=ENGINE_SEED,
        created=created,
    )
    record = episode.as_record()
    return {
        "format_version": CELL_FORMAT_VERSION,
        "backend": "cityflow",
        "scenario": scenario,
        "arm": arm,
        "draw_id": int(draw_id),
        **{key: record[key] for key in (
            "att_ours", "att_engine", "att_difference", "created", "entered", "never_entered", "entered_fraction",
            "completed_at_horizon", "running_at_horizon", "waiting_at_horizon", "horizon_vehicle_count",
            "episode_reward", "seconds", "seconds_rollout",
        )},
        "engine_seed_requested": ENGINE_SEED,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "flow_sha256": hashlib.sha256((config.parent / "flow.json").read_bytes()).hexdigest(),
        **structure,
        **_git_provenance(),
    }


# ----------------------------------------------------------------------
# Validation, the barrier, the roll
# ----------------------------------------------------------------------


def validate_anchor_cell(payload: Mapping[str, Any], *, cell: Mapping[str, Any]) -> None:
    """Every property a cell must carry, each refusal naming its reason.  Run before a chunk is written, on every chunk a
    resumed roll finds, and by the report over every chunk it reads."""
    import offline.transfer_curve as tc
    from offline.transfer_calibration import EXPECTED_DECISIONS, EXPECTED_TIME_TO_TELEPORT, PARITY_VTYPE_ID

    label = cell_name(cell)
    if payload.get("format_version") != CELL_FORMAT_VERSION:
        raise ValueError(f"{label}: format_version {payload.get('format_version')!r} is not {CELL_FORMAT_VERSION!r}")
    for key in ("backend", "scenario", "arm", "draw_id"):
        if payload.get(key) != cell.get(key):
            raise ValueError(
                f"{label}: the chunk says {key}={payload.get(key)!r} and the cell says {cell.get(key)!r}; a chunk that "
                "describes another cell is not evidence about this one"
            )
    if payload.get("git_dirty") is not False:
        raise ValueError(
            f"{label}: git_dirty is {payload.get('git_dirty')!r}, not False; a cell rolled from a tree with uncommitted "
            "changes cannot be attributed to the commit it names"
        )
    commit = payload.get("git_commit")
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        raise ValueError(f"{label}: git_commit {commit!r} is not a 40-character hex commit")

    if payload["backend"] == "cityflow":
        for key in ("att_ours", "att_engine", "created", "entered", "never_entered", "horizon_vehicle_count",
                    "episode_reward", "roadlink_structure_digest", "multiplicities"):
            if key not in payload:
                raise ValueError(f"{label}: a CityFlow cell carries {key}, and this one does not")
        if int(payload["created"]) != int(payload["entered"]) + int(payload["never_entered"]):
            raise ValueError(
                f"{label}: created {payload['created']} is not entered {payload['entered']} + never entered "
                f"{payload['never_entered']}"
            )
        return

    if payload["backend"] != "sumo":
        raise ValueError(f"{label}: backend {payload['backend']!r} is neither 'sumo' nor 'cityflow'")
    grid = payload["scenario"] == tc.GRID4X4_SCENARIO_KEY
    if int(payload["decisions"]) != EXPECTED_DECISIONS or len(payload.get("actions") or []) != EXPECTED_DECISIONS:
        raise ValueError(f"{label}: {payload['decisions']} decisions, not the {EXPECTED_DECISIONS} decisions of the horizon")
    if payload.get("actions_in_range") is not True:
        raise ValueError(f"{label}: an action was outside the intersection's legal range")
    if str(payload["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
        raise ValueError(
            f"{label}: SUMO reported time-to-teleport {payload['time_to_teleport_option']!r}, not "
            f"{EXPECTED_TIME_TO_TELEPORT!r}; this is not A15(c)'s teleport-free regime"
        )
    if list(payload["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
        raise ValueError(
            f"{label}: the engine ran vehicle type(s) {payload['vehicle_types_seen']!r}, not [{PARITY_VTYPE_ID!r}]"
        )
    if int(payload["engine_seed_requested"]) != ENGINE_SEED:
        raise ValueError(f"{label}: engine_seed_requested {payload['engine_seed_requested']} is not A18(c)'s {ENGINE_SEED}")
    if payload["att_env"] != payload["att_horizon"]:
        raise ValueError(f"{label}: att_env and att_horizon, two routes to one quantity, disagree")
    checked = bool(payload.get("halting_checked"))
    if checked != tc.halting_check_for(int(payload["draw_id"])):
        raise ValueError(
            f"{label}: halting_checked is {checked} on draw {payload['draw_id']}; the cross-check runs on draw "
            f"{tc.HALTING_CHECK_DRAW} and nowhere else (Amendment C2)"
        )
    if checked and int(payload["halting_n_disagreeing_lane_seconds"]) != 0:
        raise ValueError(f"{label}: the recorder's halting classification disagreed with SUMO's own")
    if grid:
        tc._validate_collision_record(payload, label=label)
        ids = payload.get("intersection_ids")
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise ValueError(f"{label}: intersection_ids {ids!r} is not a list of distinct ids")
        split = sorted(ix for ix in ids if payload["local_return"][ix] != payload["local_return_from_lanes"][ix])
        if split:
            raise ValueError(f"{label}: the two per-intersection return routes disagree on {split[:4]}")
    elif int(payload["n_teleports"]) != 0:
        raise ValueError(f"{label}: {payload['n_teleports']} teleport(s) under A15(c)'s teleport-free regime on hz1x1")
    policy = payload.get("policy") or {}
    if payload["arm"] == "fixedtime_kcf":
        expected = KCF_README_VALUES[str(payload["scenario"])]
        if policy.get("fixed_time_k") != expected:
            raise ValueError(
                f"{label}: fixedtime_kcf ran at k = {policy.get('fixed_time_k')!r}, not the kcf {expected} of CityFlow's "
                "tuning"
            )
    if payload["arm"] == "maxpressure_1to1" and "multiplicities" not in policy:
        raise ValueError(f"{label}: a maxpressure_1to1 cell records its measured multiplicities, and this one does not")


def assert_work_dir(work_dir: str | Path, *, output_root: str | Path, repo_root: str | Path) -> Path:
    """Refuse a work directory outside ``<output_root>/p9_2`` or ``<output_root>/p9_2_runs``, or inside a repository tree.

    A27(a): new runs write only into new ``output/p9_*`` directories.  ``DEFERRED`` 55: a gitignored resource inside a
    worktree is deleted with it.  The check creates nothing.
    """
    path = Path(os.path.abspath(work_dir))
    repo = Path(os.path.abspath(repo_root))
    if path == repo or repo in path.parents:
        raise ValueError(
            f"{path} is inside the repository tree {repo}; a gitignored output there is deleted when the worktree is "
            "retired (DEFERRED 55)"
        )
    output = Path(os.path.abspath(output_root))
    allowed = (output / "p9_2", output / "p9_2_runs")
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError(
            f"{path} is not under {allowed[0]} or {allowed[1]}; A27(a): new runs write only into new output/p9_* "
            "directories, and this task's are p9_2 and p9_2_runs"
        )
    return Path(work_dir)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _shard(cells: Sequence[Mapping[str, Any]], *, shard: int, of: int) -> list[Mapping[str, Any]]:
    if int(of) < 1 or not 0 <= int(shard) < int(of):
        raise ValueError(f"shard {shard!r} of {of!r} is not a valid partition")
    ordered = sorted(cells, key=cell_name)
    return [cell for index, cell in enumerate(ordered) if index % int(of) == int(shard)]


def roll_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    work_dir: str | Path,
    shard: int,
    of: int,
    episode_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Roll this shard's cells that have no chunk yet, validating each payload BEFORE it is written.

    Resumable at cell grain: a chunk that exists and validates against its cell is skipped; one that exists and does NOT
    validate is refused and left exactly as it is (``PROJECT_PLAN`` §7: nothing in a campaign directory is overwritten).
    A payload that fails validation -- an A23 refusal included -- STOPS the shard with nothing written for that cell, and
    nothing is created before the first valid payload exists.
    """
    work = Path(work_dir)
    mine = _shard(cells, shard=shard, of=of)
    rolled: list[str] = []
    skipped = 0
    for cell in mine:
        path = work / cell_name(cell)
        if path.is_file():
            try:
                existing = json.loads(path.read_bytes())
                validate_anchor_cell(existing, cell=cell)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{path} exists and is not a valid chunk for this cell ({exc}); it is not overwritten -- copy it to "
                    "output/p9_2_runs/ with its sha256 and remove it deliberately before re-rolling"
                ) from exc
            skipped += 1
            continue
        payload = episode_fn(cell)
        validate_anchor_cell(payload, cell=cell)
        work.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, payload)
        rolled.append(path.name)
        print(f"  rolled {path.name} ({payload.get('seconds', 0.0):.1f} s)", flush=True)
    return {"rolled": len(rolled), "skipped": skipped, "rolled_names": rolled, "n_cells": len(mine)}


def anchor_status(work_dir: str | Path) -> dict[str, Any]:
    """Declared against present, recomputed from disk every time it is asked."""
    work = Path(work_dir)
    declared = declared_anchor_cells()
    present = [cell for cell in declared if (work / cell_name(cell)).is_file()]
    groups: dict[str, int] = {}
    for cell in present:
        key = f"{cell['backend']}/{cell['scenario']}/{cell['arm']}"
        groups[key] = groups.get(key, 0) + 1
    return {
        "declared": len(declared),
        "present": len(present),
        "missing": len(declared) - len(present),
        "complete": len(present) == len(declared),
        "by_group": dict(sorted(groups.items())),
        "replica_gate_present": (work / REPLICA_GATE_NAME).is_file(),
    }


# ----------------------------------------------------------------------
# The replica gate
# ----------------------------------------------------------------------


def compare_with_registered(
    payload: Mapping[str, Any], record: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, list[str]]:
    """Field NAMES only -- the ones that differ under ``==`` and the ones either side lacks; never a value."""
    differing: list[str] = []
    missing: list[str] = []
    for field in fields:
        if field not in payload or field not in record:
            missing.append(field)
        elif payload[field] != record[field]:
            differing.append(field)
    return {"differing": differing, "missing": missing}


def _registered_anchor_record(data_dir: Path, scenario: str, arm: str, draw_id: int) -> dict[str, Any]:
    cells = json.loads((data_dir / REGISTERED_ANCHOR_ARTIFACT[scenario]).read_bytes())["cells"]
    matches = [c for c in cells if c["kind"] == "anchor" and c["arm"] == arm and int(c["draw_id"]) == int(draw_id)]
    if len(matches) != 1:
        raise ValueError(
            f"{REGISTERED_ANCHOR_ARTIFACT[scenario]} holds {len(matches)} {arm} anchor record(s) for draw {draw_id}, not one"
        )
    return matches[0]


def replica_gate(
    *, draws_root: str | Path, corpus_root: str | Path, data_dir: str | Path, draw_id: int = REPLICA_DRAW,
    episode_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The registered ``maxpressure`` and ``fixedtime`` (k = 4) through :func:`sumo_anchor_episode`, on both scenarios,
    against the committed anchor records of P7.3a (hz1x1) and P7.3d (grid4x4) on every measurement they carry, under
    ``==``.  Passing proves the replica IS the registered route; the roll refuses to start without a passing gate."""
    run = sumo_anchor_episode if episode_fn is None else episode_fn
    data = Path(data_dir)
    checks: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        fields = list(REPLICA_COMPARED_FIELDS) + (
            list(GRID_REPLICA_EXTRA_FIELDS) if scenario == "cityflow_grid4x4" else []
        )
        for arm in ("maxpressure", "fixedtime"):
            payload = run(scenario, int(draw_id), arm, draws_root=draws_root, corpus_root=corpus_root)
            record = _registered_anchor_record(data, scenario, arm, int(draw_id))
            result = compare_with_registered(payload, record, fields)
            checks.append({
                "scenario": scenario,
                "arm": arm,
                "draw_id": int(draw_id),
                "artifact": REGISTERED_ANCHOR_ARTIFACT[scenario],
                "n_fields": len(fields),
                **result,
                "passed": not result["differing"] and not result["missing"],
                "seconds": payload.get("seconds"),
            })
    return {
        "format_version": REPLICA_GATE_FORMAT_VERSION,
        "what": (
            "the registered anchors rolled through this task's SUMO replica of run_cell's anchor branch, compared with "
            "their committed records on every measurement field under =="
        ),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        **_git_provenance(),
    }


# ----------------------------------------------------------------------
# rho under the pairs, and H3's clauses (C3)
# ----------------------------------------------------------------------

#: The four committed transferred-DT artifacts this task re-scores, pinned by sha256 (checked BEFORE parsing).
INPUT_ARTIFACTS: Mapping[str, str] = {
    "p7_3a_zero_shot.json": "e7a0d290a3179ffea8185496b593ee794d438c28f200e806f3c35566c389fb70",
    "p7_3b_anchor.json": "4cae233ce480a5596dc9a65ad53c1ec0cb20c17fa80008d1e1a13341703af1cc",
    "p7_3d_grid4x4.json": "c63c371ff14d208d16b9fbfa6d3daa31975679c90b20b5a4b44fbed60760b0b7",
    "p7_3c_grid4x4.json": "1bcea367d86fa1943d45c747634b7ed2e49cb3c174f44a532c7e8ba13ca66e1d",
}

#: A27(f)'s pairs, each a role -> arm map: the registered pair and the three beside it.
PAIRS: Mapping[str, Mapping[str, str]] = {
    "registered": {"maxpressure": "maxpressure", "fixedtime": "fixedtime"},
    "mp1to1_ftk4": {"maxpressure": "maxpressure_1to1", "fixedtime": "fixedtime"},
    "mpperlink_ftkcf": {"maxpressure": "maxpressure", "fixedtime": "fixedtime_kcf"},
    "mp1to1_ftkcf": {"maxpressure": "maxpressure_1to1", "fixedtime": "fixedtime_kcf"},
}
PAIR_DESCRIPTIONS: Mapping[str, str] = {
    "registered": "(MP per-link, FT k = 4) -- the registered anchors, the record",
    "mp1to1_ftk4": "(MP 1:1, FT k = 4)",
    "mpperlink_ftkcf": "(MP per-link, FT kcf)",
    "mp1to1_ftkcf": "(MP 1:1, FT kcf) -- the 'aligned anchors' of A27(f)'s sentences",
}
#: The pair A27(f)'s two sentences call "the aligned anchors": a MaxPressure that weighs upstream and downstream equally
#: and a fixed-time plan at CityFlow's tuned cycle.
ALIGNED_PAIR = "mp1to1_ftkcf"
_REGISTERED_ROLE_ARMS = frozenset({"fixedtime", "maxpressure"})

#: The registered verdicts, in P5.2c's vocabulary, as the registration records them: A19(a) on hz1x1 (clause 1 HOLDS,
#: clause 2 REFUTED -- "both entirely above 1" -- for both subjects; clause 3 VOID); A20(e) and P7.3d on grid4x4 (clause 1
#: holds, confirmatory; clause 2's inequality holds and is not scored); A24(d) and P7.3c (clause 3 (i), the criterion met).
#: ``tests/test_sumo_anchor_rho.py`` recomputes each from the committed artifacts by the functions below.
REGISTERED_H3_VERDICTS: Mapping[str, Any] = {
    "cityflow1x1": {
        "mappo1000": {"clause_1": "HELD", "clause_2": "FAILED"},
        "mix50": {"clause_1": "HELD", "clause_2": "FAILED"},
    },
    "cityflow_grid4x4": {"mappo1000_dt_nomix_h4": {"clause_1": "HELD", "clause_2": "HELD", "clause_3": "(i)"}},
}
REGISTERED_H3_SOURCES: Mapping[str, str] = {
    "cityflow1x1": "PREREGISTRATION A19(a): clause 1 HOLDS, clause 2 REFUTED (rho entirely above 1), clause 3 VOID",
    "cityflow_grid4x4": (
        "PREREGISTRATION A20(e) and P7.3d: clause 1 confirmatory and holds, clause 2 reported as the inequality it is "
        "(not scored); A24(d) and P7.3c: clause 3 (i), the criterion met"
    ),
}
CLAUSE_3_VOID = (
    "VOID (A19(a)): under this anchor pair the zero-shot rho is not below 1, so the gap clause 3 presupposes does not "
    "exist and k = 100 has nothing to close"
)

#: P7.3a's (subject, arm, role) groups in report's own order -- A17(c)'s subjects x the four declared arms, then the
#: anchor -- so a re-scored block lists its entries as the committed one does.
def _hz1x1_groups() -> list[tuple[str, str, str]]:
    import offline.transfer_curve as tc

    return [(subject, spec.name, spec.role) for subject in tc.SUBJECTS for spec in tc.DECLARED_ARMS] + [
        (tc.ANCHOR_SUBJECT, tc.ANCHOR_ARM, tc.ANCHOR_ROLE)
    ]


def load_inputs(data_dir: str | Path) -> dict[str, Any]:
    """The four committed artifacts, each refused unless its bytes hash to :data:`INPUT_ARTIFACTS`'s pin."""
    data = Path(data_dir)
    loaded: dict[str, Any] = {}
    for name, expected in INPUT_ARTIFACTS.items():
        raw = (data / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{data / name}: sha256 {digest} is not the pinned {expected}; a committed artifact is an input by "
                "digest and is never re-scored in another version"
            )
        loaded[name] = json.loads(raw)
    return loaded


def registered_anchors(artifact: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """``{draw: {"fixedtime": row, "maxpressure": row}}`` from an artifact's anchor records, one per draw and arm."""
    anchors: dict[int, dict[str, Any]] = {}
    for row in artifact["cells"]:
        if row["kind"] != "anchor" or row["arm"] not in _REGISTERED_ROLE_ARMS:
            continue
        slot = anchors.setdefault(int(row["draw_id"]), {})
        if row["arm"] in slot:
            raise ValueError(f"two {row['arm']} anchor records for draw {row['draw_id']}")
        slot[str(row["arm"])] = row
    incomplete = sorted(draw for draw, slot in anchors.items() if set(slot) != set(_REGISTERED_ROLE_ARMS))
    if incomplete:
        raise ValueError(f"draws {incomplete[:5]} lack one of the two registered anchors")
    return anchors


def pair_anchor_rows(
    pair: str, *, registered: Mapping[int, Mapping[str, Any]], new: Mapping[str, Mapping[int, Any]] | None
) -> dict[int, dict[str, Any]]:
    """The pair's anchors per draw, keyed by ROLE (``fixedtime`` / ``maxpressure``) as ``transfer_curve._rho_pair`` reads
    them: a registered arm's row from the committed artifact, a new arm's row from this task's cells.  Refused unless both
    roles cover exactly the registered anchors' draws."""
    if pair not in PAIRS:
        raise ValueError(f"{pair!r} is not one of A27(f)'s pairs ({list(PAIRS)})")
    by_role: dict[str, Mapping[int, Any]] = {}
    for role, arm in PAIRS[pair].items():
        if arm in _REGISTERED_ROLE_ARMS:
            by_role[role] = {int(draw): slot[arm] for draw, slot in registered.items()}
        else:
            if new is None or arm not in new:
                raise ValueError(f"pair {pair} needs the new arm {arm!r}, and no cells of it were given")
            by_role[role] = {int(draw): row for draw, row in new[arm].items()}
    draws = set(int(draw) for draw in registered)
    covered = {role: set(rows) for role, rows in by_role.items()}
    if any(cover != draws for cover in covered.values()):
        raise ValueError(
            f"pair {pair}: the anchors do not cover the same draws as the registered anchors "
            f"({ {role: len(cover) for role, cover in covered.items()} } of {len(draws)}); rho is paired per draw, and "
            "a ratio built from another draw's anchors normalises nothing"
        )
    return {draw: {role: by_role[role][draw] for role in ("fixedtime", "maxpressure")} for draw in sorted(draws)}


def rescored_rows(
    rows: Sequence[Mapping[str, Any]], anchors_by_draw: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Each row with ``rho_e_sumo`` and ``rho_att_env`` recomputed by ``transfer_curve._rho_pair`` -- the registered
    function -- against the SAME draw's anchors of the pair."""
    import offline.transfer_curve as tc

    out: list[dict[str, Any]] = []
    for row in rows:
        draw = int(row["draw_id"])
        if draw not in anchors_by_draw:
            raise ValueError(f"a row of draw {draw} has no anchors of its own draw in this pair")
        out.append({**row, **tc._rho_pair(row, anchors_by_draw[draw])})
    return out


def _role_anchor_rows(anchors_by_draw: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The pair's anchors as rows whose ``arm`` is the ROLE, which is how ``transfer_curve._paired_block`` and
    ``_grid4x4_estimates`` find them; the concrete arm is restored on the blocks afterwards (:func:`_concrete_arms`)."""
    return [
        {**dict(slot[role]), "arm": role, "kind": "anchor", "seed": None}
        for _draw, slot in sorted(anchors_by_draw.items())
        for role in ("fixedtime", "maxpressure")
    ]


def _concrete_arms(anchors_by_draw: Mapping[int, Mapping[str, Any]]) -> dict[str, str]:
    first = next(iter(anchors_by_draw.values()))
    return {role: str(first[role]["arm"]) for role in ("fixedtime", "maxpressure")}


def _relabel(paired: Mapping[str, Mapping[str, Any]], concrete: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    return {
        key: {role: {**block, "right_arm": concrete[role]} for role, block in by_role.items()}
        for key, by_role in paired.items()
    }


def hz1x1_rho_block(
    rows: Sequence[Mapping[str, Any]], anchors_by_draw: Mapping[int, Mapping[str, Any]], *,
    groups: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """``by_subject_arm`` and ``by_subject_arm_seed`` exactly as ``transfer_curve.report`` writes them for hz1x1 -- the
    per-draw seed means (``_seed_means_by_draw``), ``dt_gate.mean_ci95`` over them in draw order, the paired ATT blocks
    (``_paired_block``) -- plus each group's per-draw series, which the committed artifact does not carry."""
    import offline.transfer_curve as tc
    from offline.dt_gate import mean_ci95

    concrete = _concrete_arms(anchors_by_draw)
    paired_rows = [*rows, *_role_anchor_rows(anchors_by_draw)]
    by_seed: list[dict[str, Any]] = []
    by_arm: list[dict[str, Any]] = []
    per_draw: dict[str, dict[str, dict[str, float]]] = {}
    for subject, arm, role in groups:
        arm_rows = [r for r in rows if r["subject"] == subject and r["arm"] == arm]
        if not arm_rows:
            continue
        for seed in tc.TRAINING_SEEDS:
            seed_rows = [r for r in arm_rows if r["seed"] == seed]
            if not seed_rows:
                continue
            per_seed: dict[str, Any] = {"subject": subject, "arm": arm, "seed": int(seed), "n_draws": len(seed_rows)}
            for key in ("e_sumo", "att_env"):
                usable = [r[f"rho_{key}"] for r in seed_rows if r[f"rho_{key}"] is not None]
                per_seed[f"mean_rho_{key}"] = float(sum(usable) / len(usable)) if usable else None
                per_seed[f"n_draws_{key}"] = len(usable)
            by_seed.append(per_seed)
        entry: dict[str, Any] = {"subject": subject, "arm": arm, "role": role}
        for key in ("e_sumo", "att_env"):
            series = tc._seed_means_by_draw(arm_rows, f"rho_{key}")
            per_draw.setdefault(f"{subject}/{arm}", {})[key] = {str(draw): series[draw] for draw in sorted(series)}
            if not series:
                entry[key] = {
                    "n_draws": 0, "mean": None, "std": None, "ci95": None, "ci95_low": None, "ci95_high": None,
                    "why_empty": (
                        "every draw's denominator on this definition was exactly zero; see denominator_diagnostic and "
                        "excluded_draws"
                    ),
                }
                continue
            stats = mean_ci95([series[draw] for draw in sorted(series)])
            entry[key] = {
                "n_draws": stats.n, "mean": stats.mean, "std": stats.std, "ci95": stats.ci95,
                "ci95_low": stats.mean - stats.ci95, "ci95_high": stats.mean + stats.ci95,
            }
        paired = {
            key: {anchor: tc._paired_block(paired_rows, arm_rows, subject, arm, anchor, key)
                  for anchor in ("fixedtime", "maxpressure")}
            for key in ("e_sumo", "att_env")
        }
        entry["paired_att"] = _relabel(paired, concrete)
        by_arm.append(entry)
    return {"by_subject_arm": by_arm, "by_subject_arm_seed": by_seed, "per_draw": per_draw}


def grid4x4_rho_block(
    rows: Sequence[Mapping[str, Any]], anchors_by_draw: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """``transfer_curve._grid4x4_estimates`` -- P7.3d's own estimators, the per-draw block and the H3 inequalities -- on
    the pair's re-scored rows, the paired blocks relabelled with the pair's concrete arms."""
    import offline.transfer_curve as tc

    estimates = tc._grid4x4_estimates([*rows, *_role_anchor_rows(anchors_by_draw)], anchors_by_draw)
    registered = estimates["rho"]["registered_arm"]
    registered["paired_att"] = _relabel(registered["paired_att"], _concrete_arms(anchors_by_draw))
    return estimates


def clause_1_verdict(low: float, high: float) -> str:
    """H3 clause 1, *better than fixed-time* (rho > 0), in P5.2c's vocabulary on rho's 95 % interval: HELD iff the
    interval lies entirely above 0, FAILED iff entirely below, NOT RESOLVED if it touches or crosses 0."""
    from offline.tier_statements import interval_outcome

    return interval_outcome(float(low), float(high), predicted="positive")


def clause_2_verdict(low: float, high: float) -> str:
    """H3 clause 2, *worse than MaxPressure* (rho < 1): HELD iff the interval lies entirely below 1, FAILED iff entirely
    above 1 -- A19(a)'s REFUTED -- NOT RESOLVED if it touches or crosses 1."""
    from offline.tier_statements import interval_outcome

    return interval_outcome(float(low) - 1.0, float(high) - 1.0, predicted="negative")


def clause_3_verdict(delta: float, low: float, high: float, *, gap: float) -> dict[str, Any]:
    """A24(d)'s partition with the gap recomputed under the pair (``transfer_curve.p7_3c_verdict``, which takes the gap
    as a parameter); VOID by A19(a)'s logic when the pair leaves no positive gap, the partition reported beside."""
    import offline.transfer_curve as tc

    partition = tc.p7_3c_verdict(float(delta), float(low), float(high), gap=float(gap))
    positive = float(gap) > 0.0
    return {"verdict": partition if positive else CLAUSE_3_VOID, "partition": partition, "gap_positive": positive}


def _clause_3_with_gap(clause_3: Mapping[str, Any], gap: float, gap_att: float) -> dict[str, Any]:
    """P7.3c's ``clause_3`` block with every G-dependent field recomputed for *gap* (A27(f): "with G recomputed under the
    pair"): the verdict, the sentence, the closure fractions, the per-seed half-gap count and the att_env verdict.  The
    statistics of Delta_100 do not depend on G and are kept as ``p7_3c_estimates`` computed them."""
    import offline.transfer_curve as tc

    block = json.loads(json.dumps(clause_3))
    delta = block["delta_100"]
    primary = clause_3_verdict(delta["delta"], delta["lo"], delta["hi"], gap=gap)
    block["G"], block["G_att"], block["half_gap"] = gap, gap_att, gap / 2
    block["verdict"] = primary["verdict"]
    block["partition"] = primary["partition"]
    block["gap_positive"] = primary["gap_positive"]
    block["sentence"] = (
        tc._p7_3c_sentence(primary["partition"], delta["delta"], delta["lo"], delta["hi"], gap)
        if primary["gap_positive"] else CLAUSE_3_VOID
    )
    block["closure_fraction"] = (
        {"value": delta["delta"] / gap, "ci95_low": delta["lo"] / gap, "ci95_high": delta["hi"] / gap}
        if gap != 0.0 else None
    )
    att = block["att_env"]
    att_delta = att["delta_100"]
    secondary = clause_3_verdict(att_delta["delta"], att_delta["lo"], att_delta["hi"], gap=gap_att)
    att["closure_fraction"] = (
        {"value": att_delta["delta"] / gap_att, "ci95_low": att_delta["lo"] / gap_att,
         "ci95_high": att_delta["hi"] / gap_att}
        if gap_att != 0.0 else None
    )
    att["verdict"] = secondary["verdict"]
    att["agrees_with_primary"] = secondary["verdict"] == primary["verdict"]
    seeds = block["per_seed"]["seeds"]
    block["per_seed"]["n_at_least_half_gap"] = sum(1 for entry in seeds if entry["delta"] >= gap / 2)
    return block


def _json_normal(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _assert_same(mine: Any, committed: Any, label: str) -> None:
    if _json_normal(mine) != _json_normal(committed):
        raise ValueError(f"{label} does not reproduce the committed block under ==; the function is not the registered one")


def _assert_rows_reproduce(committed: Sequence[Mapping[str, Any]], rescored: Sequence[Mapping[str, Any]], label: str) -> None:
    for before, after in zip(committed, rescored):
        for key in ("rho_e_sumo", "rho_att_env"):
            if before[key] != after[key]:
                raise ValueError(
                    f"{label}: the {key} of {before['subject']}/{before['arm']} seed {before['seed']} draw "
                    f"{before['draw_id']} does not reproduce the committed value under =="
                )


def _dt_rows(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in artifact["cells"] if row["kind"] == "dt"]


def _p7_3c_blocks(
    p7_3c: Mapping[str, Any], c_anchors: Mapping[int, Mapping[str, Any]], d_rows: Sequence[Mapping[str, Any]],
    d_estimates: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], float, float]:
    """P7.3c's estimators under one pair: its rows re-scored, ``p7_3c_estimates`` with the pair's zero-shot series as rho0
    (P7.3d's DT rows under the same pair -- the source A24(c) reads rho0 from), and G = 1 - rho-bar0 on each definition."""
    import offline.transfer_curve as tc

    rows = rescored_rows(_dt_rows(p7_3c), c_anchors)
    estimates = tc.p7_3c_estimates(rows, d_estimates["rho"]["by_draw"], d_rows)
    registered = d_estimates["rho"]["registered_arm"]
    gap = 1.0 - float(registered["e_sumo"]["mean"])
    gap_att = 1.0 - float(registered["att_env"]["mean"])
    return rows, estimates, gap, gap_att


def registered_reproduction(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """THE TAUTOLOGY GUARD (``BRIEF_47`` §4 test 3): under the registered pair every committed per-cell rho, per-arm block,
    H3 block and P7.3c estimator block is reproduced under ``==`` by the functions this module uses, or it refuses."""
    import offline.transfer_curve as tc

    equal: list[str] = []
    counts: dict[str, int] = {}
    groups = _hz1x1_groups()
    for name, tag in (("p7_3a_zero_shot.json", "p7_3a"), ("p7_3b_anchor.json", "p7_3b")):
        artifact = inputs[name]
        # Through THE pair path the new pairs take, so the guard covers pair_anchor_rows too.
        anchors = pair_anchor_rows("registered", registered=registered_anchors(artifact), new=None)
        committed = _dt_rows(artifact)
        rows = rescored_rows(committed, anchors)
        _assert_rows_reproduce(committed, rows, name)
        counts[name] = len(rows)
        block = hz1x1_rho_block(rows, anchors, groups=groups)
        for key in ("by_subject_arm", "by_subject_arm_seed"):
            _assert_same(block[key], artifact["rho"][key], f"{tag}.{key}")
            equal.append(f"{tag}.{key}")
        if tag == "p7_3a":
            _assert_same(tc._h3_block(block["by_subject_arm"]), artifact["h3"], "p7_3a.h3")
            equal.append("p7_3a.h3")

    p7_3d = inputs["p7_3d_grid4x4.json"]
    d_anchors = pair_anchor_rows("registered", registered=registered_anchors(p7_3d), new=None)
    d_committed = _dt_rows(p7_3d)
    d_rows = rescored_rows(d_committed, d_anchors)
    _assert_rows_reproduce(d_committed, d_rows, "p7_3d_grid4x4.json")
    counts["p7_3d_grid4x4.json"] = len(d_rows)
    d_estimates = grid4x4_rho_block(d_rows, d_anchors)
    _assert_same(d_estimates["rho"], p7_3d["rho"], "p7_3d.rho")
    _assert_same(d_estimates["h3"], p7_3d["h3"], "p7_3d.h3")
    equal += ["p7_3d.rho", "p7_3d.h3"]

    p7_3c = inputs["p7_3c_grid4x4.json"]
    c_committed = _dt_rows(p7_3c)
    c_anchors = pair_anchor_rows("registered", registered=registered_anchors(p7_3c), new=None)
    c_rows, estimates, gap, gap_att = _p7_3c_blocks(p7_3c, c_anchors, d_rows, d_estimates)
    _assert_rows_reproduce(c_committed, c_rows, "p7_3c_grid4x4.json")
    counts["p7_3c_grid4x4.json"] = len(c_rows)
    for key in ("arms", "rho0", "total_effect", "adaptation_effect", "transfer", "adjacent_steps", "budget_secondary"):
        _assert_same(estimates[key], p7_3c[key], f"p7_3c.{key}")
        equal.append(f"p7_3c.{key}")
    committed_clause = {key: value for key, value in p7_3c["clause_3"].items() if key != "robustness"}
    _assert_same(estimates["clause_3"], committed_clause, "p7_3c.clause_3")
    equal.append("p7_3c.clause_3")
    if (gap, gap_att) != (tc.P7_3C_G, tc.P7_3C_G_ATT):
        raise ValueError(f"G recomputed under the registered pair is {(gap, gap_att)}, not A24's constants")
    recomputed = _clause_3_with_gap(estimates["clause_3"], gap, gap_att)
    _assert_same({key: recomputed[key] for key in committed_clause}, committed_clause, "p7_3c.clause_3_with_G_recomputed")
    equal.append("p7_3c.clause_3_with_G_recomputed")
    return {"all_equal": True, "n_rows": counts, "blocks_equal": equal}


def _paired_side(block: Mapping[str, Any], predicted: str) -> dict[str, Any]:
    from offline.tier_statements import interval_outcome

    return {
        "anchor": block["right_arm"],
        "mean_difference": block["mean_difference"],
        "ci95_low": block["ci95_low"],
        "ci95_high": block["ci95_high"],
        "wilcoxon_p": block["wilcoxon_p"],
        "wins": block["wins"],
        "losses": block["losses"],
        "verdict": interval_outcome(float(block["ci95_low"]), float(block["ci95_high"]), predicted=predicted),
    }


def _clause_entries(
    stats: Mapping[str, Mapping[str, Any]], paired: Mapping[str, Mapping[str, Any]], *, scenario: str, subject: str,
    clause_2_scored: bool,
) -> dict[str, Any]:
    """Clauses 1 and 2 for one subject: the verdict on rho's E_sumo interval, att_env's beside it, the paired ATT test row
    beside both (DT - FT predicted negative; DT - MP predicted positive), and the registered verdict."""
    registered = REGISTERED_H3_VERDICTS[scenario][subject]
    clauses: dict[str, Any] = {}
    for clause, rule, anchor, predicted in (
        ("clause_1", clause_1_verdict, "fixedtime", "negative"),
        ("clause_2", clause_2_verdict, "maxpressure", "positive"),
    ):
        primary = stats["e_sumo"]
        verdict = rule(primary["ci95_low"], primary["ci95_high"])
        co = stats["att_env"]
        side = _paired_side(paired["e_sumo"][anchor], predicted)
        clauses[clause] = {
            "inequality": "rho_sumo > 0" if clause == "clause_1" else "rho_sumo < 1",
            "definition": "e_sumo",
            "mean_rho": primary["mean"],
            "ci95_low": primary["ci95_low"],
            "ci95_high": primary["ci95_high"],
            "verdict": verdict,
            "att_env": {
                "mean_rho": co["mean"], "ci95_low": co["ci95_low"], "ci95_high": co["ci95_high"],
                "verdict": None if co["mean"] is None else rule(co["ci95_low"], co["ci95_high"]),
            },
            "paired_att_e_sumo": side,
            "paired_agrees": side["verdict"] == verdict,
            "registered_verdict": registered[clause],
            "agrees_with_registered": verdict == registered[clause],
            "scored": True if clause == "clause_1" else clause_2_scored,
        }
    return clauses


def rescore_pair(inputs: Mapping[str, Any], pair: str, *, new_rows: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every transferred-DT rho of the four artifacts and H3's clauses under ONE pair.  *new_rows* is
    ``{scenario: {arm: {draw: row}}}`` of this task's SUMO cells (``None`` for the registered pair)."""
    import offline.transfer_curve as tc

    hz_new = None if new_rows is None else new_rows.get("cityflow1x1")
    grid_new = None if new_rows is None else new_rows.get("cityflow_grid4x4")
    groups = _hz1x1_groups()
    result: dict[str, Any] = {"pair": pair, "roles": dict(PAIRS[pair]), "description": PAIR_DESCRIPTIONS[pair]}

    a_anchors = pair_anchor_rows(pair, registered=registered_anchors(inputs["p7_3a_zero_shot.json"]), new=hz_new)
    a_block = hz1x1_rho_block(rescored_rows(_dt_rows(inputs["p7_3a_zero_shot.json"]), a_anchors), a_anchors,
                              groups=groups)
    b_anchors = pair_anchor_rows(pair, registered=registered_anchors(inputs["p7_3b_anchor.json"]), new=hz_new)
    b_block = hz1x1_rho_block(rescored_rows(_dt_rows(inputs["p7_3b_anchor.json"]), b_anchors), b_anchors,
                              groups=groups)
    d_anchors = pair_anchor_rows(pair, registered=registered_anchors(inputs["p7_3d_grid4x4.json"]), new=grid_new)
    d_rows = rescored_rows(_dt_rows(inputs["p7_3d_grid4x4.json"]), d_anchors)
    d_estimates = grid4x4_rho_block(d_rows, d_anchors)
    c_anchors = pair_anchor_rows(pair, registered=registered_anchors(inputs["p7_3c_grid4x4.json"]), new=grid_new)
    _c_rows, c_estimates, gap, gap_att = _p7_3c_blocks(inputs["p7_3c_grid4x4.json"], c_anchors, d_rows, d_estimates)
    clause_3 = _clause_3_with_gap(c_estimates["clause_3"], gap, gap_att)

    h3: dict[str, Any] = {"cityflow1x1": {}, "cityflow_grid4x4": {}}
    for entry in a_block["by_subject_arm"]:
        if entry["arm"] == "b_mean_k100":
            h3["cityflow1x1"][entry["subject"]] = _clause_entries(
                entry, entry["paired_att"], scenario="cityflow1x1", subject=entry["subject"], clause_2_scored=True
            )
    registered_arm = d_estimates["rho"]["registered_arm"]
    grid_clauses = _clause_entries(
        registered_arm, registered_arm["paired_att"], scenario="cityflow_grid4x4", subject=tc.GRID4X4_SUBJECT,
        clause_2_scored=False,
    )
    expected_3 = REGISTERED_H3_VERDICTS["cityflow_grid4x4"][tc.GRID4X4_SUBJECT]["clause_3"]
    grid_clauses["clause_3"] = {
        "rule": "A24(d)'s partition, G = 1 - rho-bar0 recomputed under the pair; VOID by A19(a) when G <= 0",
        "G": gap,
        "G_att": gap_att,
        "half_gap": gap / 2,
        "delta_100": clause_3["delta_100"],
        "verdict": clause_3["verdict"],
        "partition": clause_3["partition"],
        "gap_positive": clause_3["gap_positive"],
        "sentence": clause_3["sentence"],
        "closure_fraction": clause_3["closure_fraction"],
        "att_env": clause_3["att_env"],
        "per_seed": clause_3["per_seed"],
        "registered_verdict": expected_3,
        "agrees_with_registered": clause_3["verdict"] == expected_3,
        "status": clause_3["status"],
    }
    h3["cityflow_grid4x4"][tc.GRID4X4_SUBJECT] = grid_clauses
    denominators = {
        scenario: {
            key: tc.denominator_diagnostic(
                {draw: tc.anchor_denominator(slot, key) for draw, slot in sorted(anchors.items())}
            )
            for key in ("e_sumo", "att_env")
        }
        for scenario, anchors in (("cityflow1x1", a_anchors), ("cityflow_grid4x4", d_anchors))
    }
    result.update(
        {
            "anchors": {scenario: _concrete_arms(anchors)
                        for scenario, anchors in (("cityflow1x1", a_anchors), ("cityflow_grid4x4", d_anchors))},
            "denominator_diagnostic": denominators,
            "p7_3a": {"by_subject_arm": a_block["by_subject_arm"], "by_subject_arm_seed": a_block["by_subject_arm_seed"],
                      "per_draw": a_block["per_draw"]},
            "p7_3b": {"by_subject_arm": b_block["by_subject_arm"], "by_subject_arm_seed": b_block["by_subject_arm_seed"],
                      "per_draw": b_block["per_draw"]},
            "p7_3d": {"rho": d_estimates["rho"]},
            "p7_3c": {key: c_estimates[key] for key in (
                "arms", "rho0", "total_effect", "adaptation_effect", "transfer", "adjacent_steps", "budget_secondary"
            )},
            "h3": h3,
        }
    )
    return result


# ----------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------


def _episode_for(args: argparse.Namespace) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    from offline.admission_probe import ProbeRoots

    roots = ProbeRoots(
        repo_root=_REPO_ROOT, corpus_root=Path(args.corpus_root), draws_root=Path(args.draws_root),
        output_root=Path(args.output_root), work_dir=Path(args.work_dir),
    )

    def episode(cell: Mapping[str, Any]) -> dict[str, Any]:
        if cell["backend"] == "sumo":
            return sumo_anchor_episode(
                str(cell["scenario"]), int(cell["draw_id"]), str(cell["arm"]),
                draws_root=args.draws_root, corpus_root=args.corpus_root,
            )
        return cityflow_anchor_episode(str(cell["scenario"]), int(cell["draw_id"]), str(cell["arm"]), roots=roots)

    return episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m offline.sumo_anchor_rescore",
        description="P9.2: A27(f)'s aligned anchors (maxpressure_1to1, fixedtime_kcf) and the rho re-scoring.",
    )
    parser.add_argument("--draws-root", default=str(DEFAULT_DRAWS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--torch-threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gate", help="the replica gate: registered anchors reproduced through the replica (writes its record)")
    roll = sub.add_parser("roll-anchors", help="roll this shard's declared cells (refuses without a passing gate)")
    roll.add_argument("--shard", type=int, default=0)
    roll.add_argument("--of", type=int, default=1)
    sub.add_parser("status", help="declared against present, from disk")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code (0 done, 2 refused)."""
    from offline.offline_baselines import pin_torch_threads

    args = build_parser().parse_args(argv)
    pin_torch_threads(int(args.torch_threads))
    work = Path(args.work_dir)
    if args.command == "status":
        print(json.dumps(anchor_status(work), indent=2, sort_keys=True), flush=True)
        return 0
    assert_work_dir(work, output_root=args.output_root, repo_root=_REPO_ROOT)
    if args.command == "gate":
        record = replica_gate(draws_root=args.draws_root, corpus_root=args.corpus_root, data_dir=args.data_dir)
        work.mkdir(parents=True, exist_ok=True)
        gate_path = work / REPLICA_GATE_NAME
        if gate_path.exists():
            raise FileExistsError(f"{gate_path} exists; the gate record is written once and never overwritten")
        _write_json_atomic(gate_path, record)
        for check in record["checks"]:
            print(
                f"replica {check['scenario']} {check['arm']} draw {check['draw_id']}: "
                f"{'REPRODUCED' if check['passed'] else 'NOT REPRODUCED'} on {check['n_fields']} fields "
                f"(differing {check['differing']}, missing {check['missing']})",
                flush=True,
            )
        return 0 if record["passed"] else 2
    gate_path = work / REPLICA_GATE_NAME
    if not gate_path.is_file() or json.loads(gate_path.read_bytes()).get("passed") is not True:
        print(f"REFUSED: no passing replica gate at {gate_path}; run the gate subcommand first", flush=True)
        return 2
    result = roll_cells(declared_anchor_cells(), work_dir=work, shard=args.shard, of=args.of,
                        episode_fn=_episode_for(args))
    print(f"shard {args.shard}/{args.of}: rolled {result['rolled']}, skipped {result['skipped']} of "
          f"{result['n_cells']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() by the tests
    raise SystemExit(main())
