"""P9.3 (A27 point 8): hangzhou_4x4_gudang -- draws, pre-flight, fixed-time sweep, corpus, A11 gate, declaration.

Phase A, complete: draws, pre-flight, the fixed-time sweep, the corpus (plan / run / verify), the teachers'
report, A11's gate and the declaration.  Two refusals are deliberate and wait on P9.1's code merge (M1):
the ``random_fresh`` tier needs the collector's ``--random-rng`` and its gate cell A27 (d)'s reference.

Artifact format versions: ``p9.3-preflight/1.0``, ``p9.3-fixed-time-sweep/1.0``, ``p9.3-teachers/1.0``,
``p9.3-engine-att-gate/1.0``, ``p9.3-declaration/1.0``.

WHAT THIS IS -- AND WHAT IT IS NOT
----------------------------------
``PREREGISTRATION`` A27 (i), label 8: a *conditional generalizability extension registered after the
grid results*.  hz4x4 tests real demand and real road geometry on the grid's 4x4 topology (the same
16-node lattice, the same eight selectable green phases); it is NOT a new topology, NOT a C3 target
(A15) and NOT a new claim.  This module is Phase A of ``BRIEF_48``: everything up to and including the
corpus, the engine-semantics gate and the tier declaration.  Phase B (the learners) is
``offline/hz4x4_phase_b.py`` and needs P9.1's code merge first.

THE ROOTS, AND WHY EVERY COMMAND RUNS FROM THE MAIN TREE
-------------------------------------------------------
``configs/sim/cityflow4x4.json`` names its scenario directory RELATIVELY, and both the collector and the
materialiser resolve it against the process working directory -- and the materialiser writes the
resolved path into every draw's ``cityflow.json``.  Run from a linked worktree, every materialised draw
would point into that worktree and break when it is retired (``DEFERRED`` 55).  So every subcommand runs
with the MAIN tree as its working directory and this branch's code on ``PYTHONPATH``, and every write
target and every embedded path is refused when it resolves inside a linked worktree.

ALIGNMENT CONVENTION
--------------------
Nothing here defines a new on-disk layout.  The corpus is contract C6 v1.1 written by
``offline.collect`` unchanged; a draw is ``materialised-draw/1.0`` written by
``offline.materialise_draws`` unchanged; every ATT read is either ``att_ours`` -- ``att_horizon``,
``info["average_travel_time"]`` after the last decision (A1) -- or ``att_engine`` --
``Engine::getAverageTravelTime`` at that same horizon (A11) -- through ``admission_probe.probe_episode``
or ``engine_att_reference``'s observer, the registered routes.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "corpus_verify_artifact",
    "ARTIFACT_FORMAT_CORPUS_VERIFY",
    "resolve_fixed_time_k",
    "fixed_time_k_from_ruling",
    "fixed_time_ruling_record",
    "ARTIFACT_FORMAT_RULING",
    "definition_check",
    "d5_t_regress",
    "summarise_admission",
    "d5_protocol",
    "d5_recorded_table",
    "D5_SCENARIOS",
    "D5Scenario",
    "A11_CONSEQUENCE_FAIL",
    "A11_CONSEQUENCE_PASS",
    "A27_LABEL",
    "BASE_SEED",
    "CORPUS_PREFIX",
    "CONVERGENCE_THRESHOLD",
    "CONVERGENCE_WINDOW",
    "DRAWS_HELD_OUT",
    "DRAWS_NOMINAL_AND_TRAINING",
    "EXPECTED_ACTIONS",
    "EXPECTED_INTERSECTIONS",
    "GRID_PROTOCOL_DIRS",
    "K_GRID",
    "K_SWEEP_DRAWS",
    "MAIN_CORPUS_ROOT",
    "MAIN_DRAWS_ROOT",
    "MAIN_OUTPUT_ROOT",
    "MAIN_TREE",
    "MATCHED_BATCH",
    "ONE_DAY_SECONDS",
    "PAPER_SENTENCE",
    "PREFLIGHT_MAPPO_EPISODES",
    "SCENARIO",
    "SCENARIO_ID",
    "SCENARIO_KEY",
    "SEEDS",
    "SOURCE_CONFIG",
    "SOURCE_VEHICLES",
    "TEACHER_CHECKPOINT_SUBDIR",
    "TEACHER_CONFIG",
    "TEACHER_EPISODES",
    "TIER_ORDER",
    "CollectionRun",
    "assert_eight_actions",
    "assert_main_tree_path",
    "assert_single_insertion",
    "assert_targets_absent",
    "assert_training_draws",
    "build_parser",
    "check_flow_digests",
    "check_random_rng_scheme",
    "choose_k",
    "chosen_k_from_artifact",
    "collector_supports_random_rng",
    "convergence",
    "derive_runs",
    "distinct_action_sequences",
    "exposure_table",
    "fixedtime_factory_for_k",
    "gate_cells_for",
    "gate_extremes",
    "gate_outcome",
    "go_no_go",
    "grid_protocol",
    "hz4x4_gate_episode",
    "ladder_att",
    "main",
    "materialise_hz4x4_draws",
    "mean_training_return",
    "preflight_decision",
    "project_teachers",
    "protocol_argv",
    "source_flow_entries",
    "summarise_sweep",
    "sweep_episode",
    "teachers_report",
    "tier_row_counts",
    "verify_run",
]

# ----------------------------------------------------------------------
# Declarations (docs/plans/p9.3.md; A27 (i); BRIEF_48 section 0)
# ----------------------------------------------------------------------

ARTIFACT_FORMAT_PREFLIGHT = "p9.3-preflight/1.0"
ARTIFACT_FORMAT_SWEEP = "p9.3-fixed-time-sweep/1.0"
ARTIFACT_FORMAT_TEACHERS = "p9.3-teachers/1.0"
ARTIFACT_FORMAT_GATE = "p9.3-engine-att-gate/1.0"
ARTIFACT_FORMAT_DECLARATION = "p9.3-declaration/1.0"

#: A27(a)'s label for point 8, verbatim.
A27_LABEL = "8 -- conditional generalizability extension registered after the grid results"

#: A27(i)'s paper sentence, verbatim (the author's).
PAPER_SENTENCE = (
    "hangzhou_4x4 tests real demand and real road geometry on the grid's 4×4 topology, with the same "
    "eight selectable green phases per intersection; it is not a new topology."
)

#: A11's consequence sentences, verbatim, one per branch.
A11_CONSEQUENCE_PASS = (
    "att_engine is the primary metric for every claim on S, and att_ours is reported beside it in every "
    "table with the three counts."
)
A11_CONSEQUENCE_FAIL = (
    "no claim on S rests on either definition alone; both are reported, and any comparison whose ordering "
    "differs between the definitions is reported as DEFINITION-DEPENDENT and never as a win."
)

SCENARIO = "hz4x4"
CORPUS_PREFIX = "cf_hz4x4"
#: The source sim config, relative to the MAIN tree, which is every command's working directory.
SOURCE_CONFIG = Path("configs/sim/cityflow4x4.json")
#: ``materialise_draws.scenario_key_for_config(SOURCE_CONFIG)`` and ``offline.collect``'s ``scenario_id``.
SCENARIO_KEY = "cityflow4x4"
SCENARIO_ID = "cityflow4x4"

MAIN_TREE = Path("/home/filip/rltraffic")
#: Equal to ``offline.collect.DEFAULT_DRAWS_ROOT`` (a test asserts it rather than trusting this line).
MAIN_DRAWS_ROOT = MAIN_TREE / "scenarios" / "draws"
MAIN_CORPUS_ROOT = MAIN_TREE / "datasets_v11"
MAIN_OUTPUT_ROOT = MAIN_TREE / "output"

#: Measured on 2026-10-15 (A27 (i)): 2,983 entries, every one a single insertion.
SOURCE_VEHICLES = 2983
EXPECTED_INTERSECTIONS = 16
#: The brief's pre-flight assertion: ``Utils.infer_action_counts`` gives 8 on every intersection.
EXPECTED_ACTIONS = 8

#: Draw pools (D4): 0 is the nominal control, 1-200 the training draws the grid's tiers use, 1000-1099 held out.
DRAWS_NOMINAL_AND_TRAINING: tuple[int, ...] = tuple(range(0, 201))
DRAWS_HELD_OUT: tuple[int, ...] = tuple(range(1000, 1100))
TRAINING_DRAW_MIN = 1
TRAINING_DRAW_MAX = 999

SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
BASE_SEED = 1000

#: D5's protocol as the brief restates it: k values, tuned on training draws 1-5.
K_GRID: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 14)
K_SWEEP_DRAWS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: A27(i)'s one-day rule for the teachers, in seconds.
ONE_DAY_SECONDS = 86_400
PREFLIGHT_MAPPO_EPISODES = 20
TEACHER_EPISODES = 1000
#: Relative to the repository this module lives in; loaded by ``experiments.config.load_config``.
TEACHER_CONFIG = Path("experiments/configs/p9_3_mappo_nominal_1000_hz4x4.json")
#: ``<checkpoint_dir>/<name>`` of the teacher config, relative to the main tree.
TEACHER_CHECKPOINT_SUBDIR = Path("output/checkpoints/p9_3_mappo_nominal_1000")
TEACHER_RESULTS = Path("output/p9_3_runs/p9_3_mappo_nominal_1000/results.json")

#: RUNSPEC_01 section 4's convergence criterion.
CONVERGENCE_WINDOW = 50
CONVERGENCE_THRESHOLD = 0.05

#: The ladder's tiers in collection order (A27 (i)); ``random_fresh`` is collected last.
TIER_ORDER: tuple[str, ...] = ("maxpressure", "fixedtime", "mappo1000", "random_fresh")

#: The grid tiers whose committed manifests ARE the collection protocol (BRIEF_48 section 0).
GRID_PROTOCOL_DIRS: Mapping[str, tuple[str, ...]] = {
    "maxpressure": ("cf_grid4x4__maxpressure",),
    "fixedtime": ("cf_grid4x4__fixedtime",),
    "mappo1000": tuple(f"cf_grid4x4__mappo1000__seed{seed}" for seed in SEEDS),
}

#: A27(b)'s matched per-step rows: bc_m 1,024 windows of K = 20, the two %BC arms 102 windows (the
#: declared rounding of 102.4), iql_m 20,480 transitions.
MATCHED_BATCH: Mapping[str, int] = {
    "bc_m_windows": 1024,
    "bc_top10_m_windows": 102,
    "iql_m_transitions": 20_480,
}


@dataclass(frozen=True)
class CollectionRun:
    """One ``offline.collect`` invocation for one hz4x4 tier directory, derived from the grid's protocol."""

    name: str
    tier: str
    seed: int | None
    out_dir: Path
    argv: tuple[str, ...]
    expected_episodes: int
    draw_ids: tuple[int, ...]


# ----------------------------------------------------------------------
# Draws (BRIEF_48 section 3.2)
# ----------------------------------------------------------------------


def _repo_root() -> Path:
    """The repository this module's code lives in (this branch's worktree when run from ``PYTHONPATH``)."""
    return Path(__file__).resolve().parent.parent


def _sha256_file(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refuse_existing(path: str | Path) -> Path:
    """An artifact is written once: refuse an existing destination before anything is computed."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"{destination} exists: an artifact of this task is written once and never replaced; "
            "remove it deliberately if a re-run is intended"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"the artifact's directory {destination.parent} does not exist; this task never creates one"
        )
    return destination


def _write_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    """``dt_gate.write_json_atomic`` behind :func:`_refuse_existing` -- validated, then written atomically."""
    from offline.dt_gate import write_json_atomic

    destination = _refuse_existing(path)
    write_json_atomic(dict(payload), destination)
    return destination


def _git() -> dict[str, Any]:
    """The code's commit and whether a TRACKED file differs from it (``compute_latency``'s provenance)."""
    from offline.compute_latency import _git_provenance

    return dict(_git_provenance())


def _protocol_env_settings(corpus_root: str | Path, tier_dir: str = "cf_grid4x4__maxpressure") -> dict[str, Any]:
    """The evaluation env settings the grid's protocol manifest records, via ``dt_gate.env_settings_from_manifest``."""
    from offline.dt_gate import env_settings_from_manifest

    return env_settings_from_manifest(Path(corpus_root) / tier_dir / "manifest.json")


def assert_main_tree_path(path: str | Path, *, what: str) -> Path:
    """Return the resolved *path*, or raise ``ValueError`` when it lies inside a linked worktree.

    ``DEFERRED`` 55: a gitignored resource written inside a linked worktree is deleted with the worktree,
    and a path EMBEDDED inside one breaks with it.  The signal is the repository marker itself --
    ``offline.materialise_draws._linked_worktree_marker``, reused rather than restated: a ``.git``
    DIRECTORY in a main tree, a ``.git`` FILE in a linked worktree.  A path that does not exist yet is
    judged by its nearest existing ancestor, so a draws root that has not been created is refused before
    it is.
    """
    from offline.materialise_draws import _linked_worktree_marker

    resolved = Path(path).resolve()
    marker = _linked_worktree_marker(resolved)
    if marker is not None:
        raise ValueError(
            f"{what} {resolved} lies inside a linked worktree (its repository marker {marker} is a file, "
            "not a directory). DEFERRED 55: anything written or embedded there is lost when the worktree "
            f"is retired, so this task writes and embeds main-tree paths only -- run from {MAIN_TREE} "
            "with this branch's code on PYTHONPATH"
        )
    return resolved


def assert_single_insertion(entries: Sequence[Mapping[str, Any]], *, where: str) -> None:
    """Raise unless every flow entry is one vehicle (``startTime == endTime``).

    An entry whose ``startTime`` and ``endTime`` differ is an aggregate flow that CityFlow expands into
    several vehicles (``offline.flow_randomizer``'s module docstring).  hz4x4's source has none and no
    draw of it may have one, because ``admission_probe.created_from_flow`` -- the gate's denominator --
    counts one vehicle per entry.
    """
    bad = [
        index
        for index, entry in enumerate(entries)
        if "startTime" not in entry or "endTime" not in entry or entry["startTime"] != entry["endTime"]
    ]
    if bad:
        raise ValueError(
            f"{where}: {len(bad)} of {len(entries)} flow entries are not single insertions "
            f"(startTime != endTime, or a time missing; the first is entry {bad[0]})"
        )


def source_flow_entries(
    source_config: str | Path = SOURCE_CONFIG, *, expected_vehicles: int = SOURCE_VEHICLES
) -> tuple[Path, bytes, list[dict[str, Any]]]:
    """The flow file *source_config* points at, its bytes and its entries, with the 2,983 asserted.

    The path is resolved by ``offline.collect._cityflow_flow_source`` -- the collector's and the
    materialiser's own resolution, against the working directory -- so the file checked here is the
    file both of them read.
    """
    import json

    from offline.collect import _cityflow_flow_source

    config = Path(source_config)
    if not config.is_file():
        raise FileNotFoundError(f"source sim config not found: {config} (run from {MAIN_TREE})")
    flow_path = _cityflow_flow_source(config)
    if not flow_path.is_file():
        raise FileNotFoundError(
            f"the flow file {config} points at does not exist: {flow_path}; a relative 'dir' resolves "
            f"against the working directory, so run from {MAIN_TREE}"
        )
    raw = flow_path.read_bytes()
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError(f"{flow_path}: a CityFlow flow file is a JSON list, got {type(entries).__name__}")
    assert_single_insertion(entries, where=str(flow_path))
    if len(entries) != int(expected_vehicles):
        raise ValueError(
            f"{flow_path} holds {len(entries)} entries, not the {int(expected_vehicles)} measured for "
            "hz4x4 on 2026-10-15 (A27 (i))"
        )
    return flow_path, raw, entries


def materialise_hz4x4_draws(
    draw_ids: Sequence[int],
    *,
    source_config: str | Path = SOURCE_CONFIG,
    out_root: str | Path = MAIN_DRAWS_ROOT,
    expected_vehicles: int = SOURCE_VEHICLES,
    dry_run: bool = False,
) -> list[Any]:
    """Materialise *draw_ids* through ``offline.materialise_draws.materialise``, guarded and verified.

    Four ordered phases -- the filesystem-mutation barrier is phases 1-2 before phase 3:

    1. **Validate**, nothing written: the out-root, and the scenario directory every ``cityflow.json``
       will embed (``materialise_draws._scenario_dir``, the function the materialiser itself uses), are
       refused inside a linked worktree; the source holds the measured 2,983 single insertions.
    2. **Plan in memory**: every requested draw is drawn and rendered by a
       :class:`offline.flow_randomizer.FlowRandomizer` pinned to the module's exported defaults -- the
       pinning ``materialise`` applies -- and checked single-insertion; draw 0 must render
       byte-identical to the source.
    3. **Write**: ``materialise`` itself, unchanged -- its own staging, its idempotence (``kept``) and its
       refusal of an existing draw that differs.  Its records' ``flow_sha256`` must equal the plan's.
    4. **Verify on disk** (skipped under *dry_run*): each ``flow.json`` re-read; its sha256 equal to the
       plan's, its entries single-insertion and as many as the record's ``n_vehicles``, draw 0 equal to
       the source byte for byte.  A mismatch raises: the plan and the materialiser disagree about a draw.
    """
    import hashlib
    import json

    from offline import materialise_draws as md
    from offline.flow_randomizer import (
        DEFAULT_BASE_SEED,
        DEFAULT_JITTER_SIGMA_S,
        DEFAULT_THIN_P,
        DEFAULT_VOLUME_SCALE,
        FlowRandomizer,
    )

    ids = [int(draw_id) for draw_id in draw_ids]
    if not ids:
        raise ValueError("no draw ids requested; nothing to materialise")

    # ---- phase 1: validation only -----------------------------------------
    assert_main_tree_path(out_root, what="the draws root")
    assert_main_tree_path(
        md._scenario_dir(source_config), what="the scenario directory every draw's cityflow.json embeds,"
    )
    flow_path, raw, _entries = source_flow_entries(source_config, expected_vehicles=expected_vehicles)

    # ---- phase 2: the plan, in memory ---------------------------------------
    randomizer = FlowRandomizer(
        flow_path,
        base_seed=DEFAULT_BASE_SEED,
        jitter_sigma_s=DEFAULT_JITTER_SIGMA_S,
        thin_p=DEFAULT_THIN_P,
        volume_scale=DEFAULT_VOLUME_SCALE,
    )
    planned: dict[int, str] = {}
    for draw_id in ids:
        entries, _provenance = randomizer.draw(draw_id)
        assert_single_insertion(entries, where=f"draw {draw_id} (planned in memory)")
        data = randomizer.render_cityflow_bytes(entries)
        if draw_id == 0 and data != raw:
            raise ValueError(
                "draw 0 does not render byte-identical to the source flow; it is the nominal control and "
                "must not perturb the demand even cosmetically (offline.flow_randomizer)"
            )
        planned[draw_id] = hashlib.sha256(data).hexdigest()

    # ---- phase 3: the materialiser's own write ------------------------------
    records = md.materialise(source_config, ids, out_root=out_root, dry_run=dry_run)
    if [int(record.draw_id) for record in records] != ids:
        raise ValueError(
            f"the materialiser returned draws {[r.draw_id for r in records]} for the request {ids}"
        )
    for record in records:
        if record.flow_sha256 != planned[int(record.draw_id)]:
            raise ValueError(
                f"draw {record.draw_id}: the materialiser built flow sha256 {record.flow_sha256} but the "
                f"plan rendered {planned[int(record.draw_id)]}; the two routes disagree about the demand"
            )
    if dry_run:
        return records

    # ---- phase 4: verification on disk --------------------------------------
    for record in records:
        draw_id = int(record.draw_id)
        data = Path(record.flow_path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != planned[draw_id]:
            raise ValueError(
                f"draw {draw_id}: {record.flow_path} on disk differs from the planned bytes "
                f"(sha256 {digest}, planned {planned[draw_id]})"
            )
        entries = json.loads(data)
        assert_single_insertion(entries, where=str(record.flow_path))
        if len(entries) != int(record.n_vehicles):
            raise ValueError(
                f"draw {draw_id}: {record.flow_path} holds {len(entries)} entries but its record says "
                f"{record.n_vehicles} vehicles"
            )
        if draw_id == 0 and data != raw:
            raise ValueError(f"draw 0: {record.flow_path} differs from the source flow {flow_path}")
    return records


# ----------------------------------------------------------------------
# Pre-flight (BRIEF_48 section 3.1)
# ----------------------------------------------------------------------


def assert_eight_actions(env: Any) -> list[int]:
    """``Utils.infer_action_counts`` on *env*, refused unless 16 intersections offer 8 actions each.

    The brief's pre-flight assertion, run on a CONSTRUCTED env rather than read from a road-network file:
    A27 (i)'s sentence -- the same eight selectable green phases as the grid -- is a claim about what the
    acyclic controller offers, and this is the controller's own count.
    """
    from agent.utils.utils import Utils

    intersections = list(env.intersections)
    counts = [int(c) for c in Utils.infer_action_counts(getattr(env, "action_space", None), intersections)]
    if len(intersections) != EXPECTED_INTERSECTIONS:
        raise ValueError(
            f"this env has {len(intersections)} intersections, not hz4x4's {EXPECTED_INTERSECTIONS}"
        )
    wrong = {str(ix.id): count for ix, count in zip(intersections, counts) if count != EXPECTED_ACTIONS}
    if wrong:
        raise ValueError(
            f"{len(wrong)} intersection(s) offer an action count other than {EXPECTED_ACTIONS}: {wrong}; "
            "A27 (i) registers eight selectable green phases on every hz4x4 intersection"
        )
    return counts


def time_maxpressure_episode(
    config_path: str | Path, settings: Mapping[str, Any], *, engine_seed: int = BASE_SEED
) -> dict[str, Any]:
    """One MaxPressure episode on one draw's config, timed; the decision count, and NO outcome number.

    The policy is ``dt_gate._maxpressure_factory``, the evaluation route's own.  The pre-flight produces
    no registered number, so neither the travel time nor the return is read or kept.
    """
    import time

    from experiments.config import EnvSpec
    from experiments.envs import make_env
    from offline.dt_gate import _maxpressure_factory

    started = time.perf_counter()
    env = make_env(
        EnvSpec(id=SCENARIO_ID, backend="cityflow", paths={"config": str(config_path)}, settings=dict(settings))
    )
    try:
        counts = assert_eight_actions(env)
        choose = _maxpressure_factory(env)
        rollout_started = time.perf_counter()
        info = env.reset(seed=int(engine_seed))
        decisions = 0
        for _ in range(int(env.max_steps)):
            _reward, terminated, truncated, info = env.step(choose(env, info))
            decisions += 1
            if terminated or truncated:
                break
        seconds_rollout = time.perf_counter() - rollout_started
    finally:
        env.close()
    return {
        "seconds": time.perf_counter() - started,
        "seconds_rollout": seconds_rollout,
        "decisions": decisions,
        "action_counts": counts,
    }


def mappo_preflight_worker(
    config_path: str,
    seed: int,
    episodes: int,
    collection_config: str,
    collection_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """One teacher seed's timing in its own process: the frozen trainer for *episodes*, then a C8 check.

    ``experiments.runner.limit_torch_threads`` runs FIRST, as ``run_cell`` runs it -- one torch thread per
    process, the teachers' regime.  The real teacher config is loaded and only ``train_episodes`` is
    replaced (``dataclasses.replace``), so the timed run is the teachers' recipe and nothing is retyped.
    The trained agent then acts once on an ``info`` from the COLLECTION env (the grid protocol's
    settings): contract C8 freezes the global metric set at training, so a set that differs between the
    teachers' env and the collector's raises here, now, rather than after a day of training.
    """
    import dataclasses
    import time

    import numpy as np

    from experiments.runner import _train_agent, limit_torch_threads

    threads = limit_torch_threads()
    from experiments.config import EnvSpec, load_config
    from experiments.envs import make_env

    config = load_config(config_path)
    env_specs = [spec for spec in config.environments if spec.id == "cf_hz4x4"]
    if len(env_specs) != 1 or len(config.agents) != 1:
        raise ValueError(f"{config_path} must declare exactly one cf_hz4x4 env and one agent")
    env_spec = dataclasses.replace(
        env_specs[0], settings={**env_specs[0].settings, "train_episodes": int(episodes)}
    )
    started = time.perf_counter()
    agent, train_returns = _train_agent(config.agents[0], env_spec, int(seed))
    seconds = time.perf_counter() - started

    env = make_env(
        EnvSpec(
            id=SCENARIO_ID,
            backend="cityflow",
            paths={"config": str(collection_config)},
            settings=dict(collection_settings),
        )
    )
    try:
        info = env.reset(seed=BASE_SEED)
        action = np.asarray(agent.act(info, explore=False, update_memory=False)).reshape(-1)
        metric_keys = sorted(info.get("metrics", {}))
    finally:
        env.close()
    if action.shape != (EXPECTED_INTERSECTIONS,):
        raise ValueError(f"the trained agent returned actions of shape {action.shape} on the collection env")
    return {
        "seed": int(seed),
        "episodes": int(episodes),
        "episodes_trained": len(train_returns),
        "seconds": float(seconds),
        "seconds_per_episode": float(seconds) / int(episodes),
        "torch_threads": int(threads),
        "c8_check": "passed: the trained agent acted on an info from the collection env without a width error",
        "collection_metric_keys": metric_keys,
    }


def canary_record(phase: str) -> dict[str, Any]:
    """``PROJECT_PLAN`` section 7's canary (2026-09-13), with the power block beside it.

    ``offline.transfer_calibration.canary_seconds`` times ``run_probe`` on cityflow1x1 draw 0 and
    ``check_canary`` compares its answers under ``==``; ``offline.compute_latency.power_block`` reads the
    supplies and the Windows power mode.  Both are reused, not restated.
    """
    from offline.compute_latency import power_block, power_regime_problems
    from offline.transfer_calibration import CANARY_MAX_SECONDS, canary_seconds, check_canary

    power = power_block()
    seconds, facts = canary_seconds()
    try:
        check_canary(facts)
        reproduced, failure = True, None
    except ValueError as exc:
        reproduced, failure = False, str(exc)
    return {
        "phase": phase,
        "seconds": float(seconds),
        "threshold_seconds": float(CANARY_MAX_SECONDS),
        "verdict": "at speed" if float(seconds) <= float(CANARY_MAX_SECONDS) else "throttled",
        "reproduced": reproduced,
        "reproduction_failure": failure,
        "recipe": "offline.transfer_calibration.canary_seconds + check_canary (cityflow1x1 draw 0, P4.3's settings)",
        "power": power,
        "power_problems": power_regime_problems(power),
    }


def go_no_go(projected_seconds: float, *, budget_seconds: float = ONE_DAY_SECONDS) -> str:
    """``"GO"`` iff the projected wall-clock fits the one-day rule (inclusive), else ``"NO-GO"``."""
    value = float(projected_seconds)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"a projected wall-clock must be a finite non-negative number, got {projected_seconds!r}")
    return "GO" if value <= float(budget_seconds) else "NO-GO"


def preflight_decision(
    projection: Mapping[str, Any], canaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Whether the projection may be quoted: both canaries at speed AND reproducing the engine's answers.

    ``PROJECT_PLAN`` section 7: a rate taken on a throttled machine may not be quoted, and a canary that
    does not reproduce the engine's recorded answers is a worse finding than a slow one.  Either way the
    go/no-go is NOT decided from this run.
    """
    phases = {str(record["phase"]): record for record in canaries}
    missing = [phase for phase in ("open", "close") if phase not in phases]
    if missing or len(phases) != len(canaries):
        raise ValueError(f"the pre-flight needs exactly one canary at each of open and close; missing {missing}")
    unreproduced = [phase for phase in ("open", "close") if phases[phase]["reproduced"] is not True]
    if unreproduced:
        return {
            "quotable": False,
            "go_no_go": None,
            "reason": f"the canary did not reproduce the engine's recorded answers at {unreproduced}: a finding "
            "about the engine, worse than a slow machine; nothing timed in this run is quoted",
        }
    throttled = [phase for phase in ("open", "close") if phases[phase]["verdict"] != "at speed"]
    if throttled:
        return {
            "quotable": False,
            "go_no_go": None,
            "reason": f"the canary was above its ceiling at {throttled}: no rate measured in that state may be "
            "quoted (PROJECT_PLAN section 7, 2026-09-13); re-run at reference",
        }
    return {
        "quotable": True,
        "go_no_go": projection["verdict"],
        "reason": "both canaries at speed and reproducing the engine's recorded answers",
    }


def project_teachers(
    timings: Sequence[Mapping[str, Any]],
    *,
    train_episodes: int = TEACHER_EPISODES,
    evaluation_episodes: int,
    budget_seconds: float = ONE_DAY_SECONDS,
) -> dict[str, Any]:
    """The teachers' projected wall-clock from the concurrent per-seed pre-flight timings.

    Each seed's cell trains ``train_episodes`` episodes and then evaluates ``evaluation_episodes`` (the
    agent's and every baseline's, ``experiments.runner.run_cell``); its projection is its own measured
    seconds per episode times that count -- an upper estimate for the evaluation episodes, which run no
    update.  The five cells run CONCURRENTLY (``--workers 5``, one per seed, 16 cores), so the wall-clock
    is the SLOWEST cell's projection, and the one-day rule is applied to it.
    """
    per_episode: dict[int, float] = {}
    for timing in timings:
        seed = int(timing["seed"])
        episodes = int(timing["episodes"])
        seconds = float(timing["seconds"])
        if seed in per_episode:
            raise ValueError(f"seed {seed} is timed twice; one concurrent process per seed is the regime")
        if episodes <= 0:
            raise ValueError(f"seed {seed}: a timing over {episodes} episodes cannot give a rate")
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"seed {seed}: {seconds!r} seconds is not a timing")
        per_episode[seed] = seconds / episodes
    missing = [seed for seed in SEEDS if seed not in per_episode]
    unexpected = sorted(seed for seed in per_episode if seed not in SEEDS)
    if missing or unexpected:
        raise ValueError(
            f"the pre-flight times every teacher seed {list(SEEDS)} concurrently; missing {missing}, "
            f"unexpected {unexpected}"
        )
    episodes_per_cell = int(train_episodes) + int(evaluation_episodes)
    projected = {str(seed): per_episode[seed] * episodes_per_cell for seed in SEEDS}
    wall_clock = max(projected.values())
    return {
        "seconds_per_episode": {str(seed): per_episode[seed] for seed in SEEDS},
        "episodes_per_cell": episodes_per_cell,
        "train_episodes": int(train_episodes),
        "evaluation_episodes": int(evaluation_episodes),
        "per_seed_projected_seconds": projected,
        "projected_wall_clock_seconds": wall_clock,
        "budget_seconds": float(budget_seconds),
        "verdict": go_no_go(wall_clock, budget_seconds=budget_seconds),
        "rule": "slowest concurrent seed x (train + evaluation episodes) <= one day (A27 (i))",
    }


# ----------------------------------------------------------------------
# The fixed-time sweep (D5; BRIEF_48 section 0)
# ----------------------------------------------------------------------


def assert_training_draws(draw_ids: Sequence[int]) -> None:
    """Refuse any draw id outside the training pool 1-999 (D4).

    Draw 0 is the nominal control and 1000-1099 the held-out pool: neither may tune a baseline or enter a
    training corpus, for any method (``PREREGISTRATION`` section 5, D4; ``recollect_v11``'s draw-pool rule).
    """
    outside = sorted({int(d) for d in draw_ids if not TRAINING_DRAW_MIN <= int(d) <= TRAINING_DRAW_MAX})
    if outside:
        raise ValueError(
            f"draw ids {outside} lie outside the training pool {TRAINING_DRAW_MIN}-{TRAINING_DRAW_MAX} (D4): "
            "draw 0 is the nominal control and 1000-1099 the held-out pool, and neither may tune or train anything"
        )


def fixedtime_factory_for_k(
    config_path: str | Path,
    k: int,
    *,
    k_grid: Sequence[int] = K_GRID,
    schedule_source: str = "equal_split",
    plan_sha256: str | None = None,
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], Any]], dict[str, Any]]:
    """The collector's fixed-time policy at *k* on one draw's config, with its schedule provenance.

    ``offline.policies.fixed_time.make_fixedtime`` with the ``SimpleNamespace`` the evaluation route builds
    (``method_tier_grid._fixedtime_factory``), so the swept policy is the policy ``offline.collect`` would
    collect with at that k.  The schedule must resolve to *schedule_source* with digest *plan_sha256*: on
    hz4x4 (the defaults) the equal split, which has no plan file and so no digest; for A.7's D5 check the
    source and digest the scenario's fixed-time tier was collected with (hz1x1: the shipped plan).  Anything
    else means the config points at a different scenario directory or plan, and is refused.
    """
    from types import SimpleNamespace

    import numpy as np

    from offline.policies.fixed_time import make_fixedtime, resolve_plan

    if int(k) not in tuple(int(value) for value in k_grid):
        raise ValueError(f"k = {k} is not in the declared grid {list(k_grid)}")
    args = SimpleNamespace(env_config=str(config_path), fixed_time_k=int(k), policy="fixedtime")
    resolution = resolve_plan(args)
    if resolution.source != schedule_source:
        raise ValueError(
            f"the fixed-time schedule on {config_path} resolves to {resolution.source!r}, not the declared "
            f"{schedule_source!r}; that is a different scenario directory or plan"
        )
    if resolution.sha256 != plan_sha256:
        raise ValueError(
            f"the fixed-time plan on {config_path} has sha256 {resolution.sha256!r}, not the declared {plan_sha256!r}; "
            "this would sweep a different plan from the one the tier was collected with"
        )

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], Any]:
        policy = make_fixedtime(env, args, np.random.default_rng(0))
        return lambda _env, info: policy(info)

    return factory, {
        "fixed_time_k": int(k),
        "schedule_source": resolution.source,
        "plan_sha256": resolution.sha256,
    }


def sweep_episode(
    k: int,
    draw_id: int,
    *,
    config_path: str | Path,
    env_settings: Mapping[str, Any],
    scenario: str = SCENARIO,
    scenario_id: str = SCENARIO_ID,
    k_grid: Sequence[int] = K_GRID,
    schedule_source: str = "equal_split",
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    """One sweep episode through ``admission_probe.probe_episode``: A11(b)'s five quantities and the seconds.

    ``probe_episode`` is the P8.4a route -- ``horizon_rollout`` for ``att_ours`` (``att_horizon``, A1) and
    ``Engine::getAverageTravelTime`` read before the env closes for ``att_engine`` -- with ``created``
    counted from the draw's own ``flow.json`` (``created_from_flow``), so every episode carries
    ``att_ours``, ``att_engine``, ``entered``, ``created`` and ``never_entered`` unconditionally.  Engine
    seed ``BASE_SEED``: episode 0 of every collected draw runs at 1000.  The defaults are hz4x4's; A.7's
    D5 check passes its scenario's name, env id, k grid and schedule (``BRIEF_48`` Amendment A: reuse with a
    scenario parameter, never a copy).
    """
    from offline.admission_probe import created_from_flow, probe_episode

    assert_training_draws([draw_id])
    factory, provenance = fixedtime_factory_for_k(
        config_path, k, k_grid=k_grid, schedule_source=schedule_source, plan_sha256=plan_sha256
    )
    horizon = int(env_settings["max_steps"]) * int(env_settings["delta_time"])
    created = created_from_flow(Path(config_path).parent / "flow.json", horizon_seconds=horizon)
    arm = f"fixedtime_k{int(k)}"
    episode = probe_episode(
        scenario=scenario,
        tier="fixedtime",
        method=arm,
        arm=arm,
        seed=None,
        draw_id=int(draw_id),
        config_path=config_path,
        env_settings=env_settings,
        scenario_id=scenario_id,
        choose_action_factory=factory,
        engine_seed=BASE_SEED,
        created=created,
    )
    return {
        "k": int(k),
        "draw_id": int(draw_id),
        "engine_seed": BASE_SEED,
        "att_ours": float(episode.att_ours),
        "att_engine": float(episode.att_engine),
        "created": int(episode.created),
        "entered": int(episode.entered),
        "never_entered": int(episode.never_entered),
        "entered_fraction": float(episode.entered_fraction),
        "completed_at_horizon": int(episode.completed_at_horizon),
        "running_at_horizon": int(episode.running_at_horizon),
        "waiting_at_horizon": int(episode.waiting_at_horizon),
        "seconds": float(episode.seconds),
        "seconds_rollout": float(episode.seconds_rollout),
        "schedule_source": provenance["schedule_source"],
        "plan_sha256": provenance["plan_sha256"],
    }


def summarise_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    k_grid: Sequence[int] = K_GRID,
    draws: Sequence[int] = K_SWEEP_DRAWS,
) -> dict[int, dict[str, Any]]:
    """Per k: the mean over its draws of ``att_engine`` and of ``att_ours``, with the draw ids.

    D5's objective is the mean over the five tuning draws; both definitions are carried because Rule R
    has not run on hz4x4 when k is chosen (``docs/plans/p9.3.md`` Q2).  Every (k, draw) must appear
    exactly once, every k of the declared grid must cover exactly the declared draws, and every value
    must be finite -- a mean over a ragged or malformed record set is refused rather than reported.  The
    defaults are hz4x4's grid and draws; A.7's D5 check passes each scenario's.
    """
    import numpy as np

    grid = tuple(int(k) for k in k_grid)
    expected_draws = [int(d) for d in draws]
    by_k: dict[int, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        k = int(record["k"])
        draw_id = int(record["draw_id"])
        assert_training_draws([draw_id])
        if k not in grid:
            raise ValueError(f"k = {k} is not in the declared grid {list(grid)}")
        for key in ("att_engine", "att_ours"):
            value = float(record[key])
            if not math.isfinite(value):
                raise ValueError(f"k = {k}, draw {draw_id}: {key} is {value!r}, not a finite travel time")
        slot = by_k.setdefault(k, {})
        if draw_id in slot:
            raise ValueError(f"k = {k}, draw {draw_id} appears twice")
        slot[draw_id] = record
    missing = [k for k in grid if k not in by_k]
    if missing:
        raise ValueError(f"no records for k = {missing}; the sweep covers {list(grid)}")
    summary: dict[int, dict[str, Any]] = {}
    for k in grid:
        present = sorted(by_k[k])
        if present != sorted(expected_draws):
            raise ValueError(f"k = {k} covers draws {present}, not the declared {sorted(expected_draws)}")
        engine = np.asarray([float(by_k[k][d]["att_engine"]) for d in present], dtype=np.float64)
        ours = np.asarray([float(by_k[k][d]["att_ours"]) for d in present], dtype=np.float64)
        summary[k] = {
            "n": len(present),
            "draw_ids": present,
            "mean_att_engine": float(engine.mean()),
            "mean_att_ours": float(ours.mean()),
        }
    return summary


def choose_k(summary: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The declared choice rule: both argmins agree and are interior or at the floor, else no k.

    ``docs/plans/p9.3.md`` section 4.3, declared before the sweep ran.  Statuses: ``CHOSEN`` (the two
    definitions' argmins are the same k and it is not the grid's upper edge; ``at_platform_floor`` flags
    k = 1, D5's limitation 2), ``DEFINITION-DEPENDENT`` (they differ -- Q2), ``TIE`` (an exact tie at the
    minimum under either definition -- no tie-break is invented), ``BOUNDARY_UPPER`` (both at k = 14, where
    D5 extended its sweep and this task does not -- Q3).  Anything but ``CHOSEN`` leaves ``chosen_k`` null
    and the collector refuses the fixed-time tier until the coordinator rules.
    """
    missing = [k for k in K_GRID if k not in summary]
    if missing:
        raise ValueError(f"the summary has no k = {missing}; the declared grid is {list(K_GRID)}")

    def minimisers(key: str) -> list[int]:
        values = {k: float(summary[k][key]) for k in K_GRID}
        best = min(values.values())
        return [k for k in K_GRID if values[k] == best]

    engine = minimisers("mean_att_engine")
    ours = minimisers("mean_att_ours")
    decision: dict[str, Any] = {
        "k_att_engine": engine[0] if len(engine) == 1 else None,
        "k_att_ours": ours[0] if len(ours) == 1 else None,
        "tied_at_minimum_att_engine": engine if len(engine) > 1 else [],
        "tied_at_minimum_att_ours": ours if len(ours) > 1 else [],
        "chosen_k": None,
        "at_platform_floor": False,
        "rule": (
            "docs/plans/p9.3.md section 4.3: chosen iff the argmin of the mean att_engine and the argmin of the "
            "mean att_ours over draws 1-5 are the same k and that k is not the grid's upper edge (14); a "
            "disagreement, an exact tie or the upper edge leaves no k and stops collection for a ruling"
        ),
    }
    if len(engine) > 1 or len(ours) > 1:
        decision["status"] = "TIE"
    elif engine[0] != ours[0]:
        decision["status"] = "DEFINITION-DEPENDENT"
    elif engine[0] == max(K_GRID):
        decision["status"] = "BOUNDARY_UPPER"
    else:
        decision["status"] = "CHOSEN"
        decision["chosen_k"] = int(engine[0])
        decision["at_platform_floor"] = engine[0] == min(K_GRID)
    return decision


#: Format of the fixed-time ruling file (``BRIEF_48`` Amendment A, A.2 finding 1 and A.4).
ARTIFACT_FORMAT_RULING = "p9.3-fixed-time-ruling/1.0"


def fixed_time_ruling_record(
    *, sweep_path: str | Path, k: int, amendment: str, brief_path: str | Path
) -> dict[str, Any]:
    """The ruling file's record: the sweep artifact by sha256, the ruled k and the amendment carrying the ruling.

    ``BRIEF_48`` Amendment A, A.2 finding 1 and A.4: the sweep artifact is never edited, so a ruling on a sweep that
    chose no k is a SEPARATE committed file.  It is built only for a sweep that chose none (a ``CHOSEN`` sweep needs no
    ruling), only for a k of the declared grid, and only when the brief on disk carries the named amendment -- the
    ruling must exist on ``main`` before a file may cite it.  The brief's digest is recorded beside its name.
    """
    import json

    sweep = Path(sweep_path)
    payload = json.loads(sweep.read_bytes())
    if payload.get("format_version") != ARTIFACT_FORMAT_SWEEP:
        raise ValueError(f"{sweep}: format {payload.get('format_version')!r} is not the sweep format {ARTIFACT_FORMAT_SWEEP!r}")
    decision = payload.get("decision") or {}
    status = decision.get("status")
    if status == "CHOSEN":
        raise ValueError(f"{sweep}: the sweep is CHOSEN (k = {decision.get('chosen_k')}); a ruling fills only a sweep "
                         "that chose no k")
    if int(k) not in K_GRID:
        raise ValueError(f"k = {k} is not in the declared grid {list(K_GRID)}")
    name = str(amendment).strip()
    if not name:
        raise ValueError("a ruling names the amendment that carries the author's ruling; none was given")
    brief = Path(brief_path)
    if name not in brief.read_text(encoding="utf-8"):
        raise ValueError(f"{brief} does not contain {name!r}: the amendment that carries the ruling must be on disk "
                         "(merge main first)")
    return {
        "format_version": ARTIFACT_FORMAT_RULING,
        "registered_in": "BRIEF_48 Amendment A, A.2 finding 1 and A.4",
        "sweep_artifact": str(sweep),
        "sweep_sha256": _sha256_file(sweep),
        "sweep_status": status,
        "sweep_k_att_engine": decision.get("k_att_engine"),
        "sweep_k_att_ours": decision.get("k_att_ours"),
        "chosen_k": int(k),
        "amendment": name,
        "brief": str(brief),
        "brief_sha256": _sha256_file(brief),
        "rule": "offline.hz4x4_ladder collect reads this k only when sweep_sha256 equals the committed sweep artifact's "
        "digest; the sweep artifact itself is never edited",
    }


def fixed_time_k_from_ruling(ruling_path: str | Path, sweep_path: str | Path) -> int:
    """The k a ruling file carries, refused unless it names this exact sweep artifact by sha256.

    The collector's half of A.2 finding 1: a ruling of another format, one naming no amendment, one whose
    ``sweep_sha256`` is not the digest of *sweep_path* as it is on disk, or one whose k is outside the declared grid
    is refused, so a ruling cannot be carried across to a different or re-run sweep.
    """
    import json

    path = Path(ruling_path)
    ruling = json.loads(path.read_bytes())
    if ruling.get("format_version") != ARTIFACT_FORMAT_RULING:
        raise ValueError(f"{path}: format {ruling.get('format_version')!r} is not the ruling format {ARTIFACT_FORMAT_RULING!r}")
    if not str(ruling.get("amendment") or "").strip():
        raise ValueError(f"{path} names no amendment carrying the author's ruling")
    actual = _sha256_file(sweep_path)
    if ruling.get("sweep_sha256") != actual:
        raise ValueError(f"{path} rules on a sweep with sha256 {ruling.get('sweep_sha256')!r}, but {sweep_path} hashes to "
                         f"{actual}; a ruling applies only to the exact artifact it names")
    k = ruling.get("chosen_k")
    if not isinstance(k, int) or isinstance(k, bool) or k not in K_GRID:
        raise ValueError(f"{path}: k = {k} is not in the declared grid {list(K_GRID)}")
    return int(k)


def resolve_fixed_time_k(sweep_path: str | Path, ruling_path: str | Path | None) -> int:
    """The fixed-time k to collect with: the sweep's own when it chose one, otherwise the ruling's.

    A ``CHOSEN`` sweep is its own authority and a ruling beside it is refused; a sweep that chose no k needs a ruling
    file (:func:`fixed_time_k_from_ruling`), and without one the refusal names the sweep's status.
    """
    import json

    decision = json.loads(Path(sweep_path).read_bytes()).get("decision") or {}
    if decision.get("status") == "CHOSEN":
        if ruling_path is not None:
            raise ValueError(f"{sweep_path}: the sweep chose k itself; a ruling may only fill a sweep that chose none")
        return chosen_k_from_artifact(sweep_path)
    if ruling_path is None:
        return chosen_k_from_artifact(sweep_path)
    return fixed_time_k_from_ruling(ruling_path, sweep_path)


def chosen_k_from_artifact(path: str | Path) -> int:
    """The k a committed sweep artifact chose, refused unless its status is ``CHOSEN``.

    The collector reads the tuned k from here and from nowhere else -- never from a flag -- so the k that
    collects the tier is the k D5's protocol recorded before collection.
    """
    import json

    payload = json.loads(Path(path).read_bytes())
    if payload.get("format_version") != ARTIFACT_FORMAT_SWEEP:
        raise ValueError(
            f"{path}: format {payload.get('format_version')!r} is not this task's sweep format {ARTIFACT_FORMAT_SWEEP!r}"
        )
    decision = payload.get("decision") or {}
    status = decision.get("status")
    if status != "CHOSEN":
        raise ValueError(
            f"{path}: the sweep's status is {status!r}, so no fixed-time k may be collected with until the "
            "coordinator rules (docs/plans/p9.3.md Q2/Q3)"
        )
    k = decision.get("chosen_k")
    if k not in K_GRID:
        raise ValueError(f"{path}: chosen_k {k!r} is not in the declared grid {list(K_GRID)}")
    return int(k)


# ----------------------------------------------------------------------
# A.7: the D5 definition check on hz1x1 and grid4x4 (BRIEF_48 Amendment A; EXPLORATION)
# ----------------------------------------------------------------------

ARTIFACT_FORMAT_D5_CHECK = "p9.3-d5-definition-check/1.0"

#: A.7's label, verbatim.
D5_CHECK_LABEL = (
    "exploration \u2014 a disclosure measurement on training draws 1\u20135; no registered k, tier, anchor, rho or "
    "verdict changes"
)

#: D5's README, the registered tuning record whose table the check must reproduce.
D5_README = Path("docs/data/fixed_time_sweep/README.md")


@dataclass(frozen=True)
class D5Scenario:
    """One scenario D5 tuned: its README row, its draws key and env id, its protocol manifest and its k grid."""

    name: str
    scenario_key: str
    scenario_id: str
    protocol_dir: str
    k_grid: tuple[int, ...]
    schedule_source: str


#: The two scenarios A.7 names, with the k grids D5's README records (a test holds them to the README and to A.7).
D5_SCENARIOS: tuple[D5Scenario, ...] = (
    D5Scenario(
        name="cf_hz1x1",
        scenario_key="cityflow1x1",
        scenario_id="cityflow1x1",
        protocol_dir="cf_hz1x1__fixedtime",
        k_grid=(2, 3, 4, 6, 8, 10, 14),
        schedule_source="shipped_plan",
    ),
    D5Scenario(
        name="cf_grid4x4",
        scenario_key="cityflow_grid4x4",
        scenario_id="cityflow_grid4x4",
        protocol_dir="cf_grid4x4__fixedtime",
        k_grid=(1, 2, 3, 4, 6),
        schedule_source="equal_split",
    ),
)


def d5_recorded_table(readme_path: str | Path = D5_README) -> dict[str, dict[str, Any]]:
    """D5's results table read from its README: per scenario, the printed ``att_horizon`` per k and the chosen k.

    Read, never retyped: the header row names the ``k=N`` columns and ``chosen``; each scenario row's cells are
    kept as the PRINTED two-decimal strings (bold markers stripped, an em dash meaning "not swept"), because
    A.7's T-regress compares to the printed decimals.  A README without the table, or a cell that is not a
    two-decimal number, is refused.
    """
    import re

    path = Path(readme_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header_at = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("| scenario |") and "| k=" in line), None
    )
    if header_at is None:
        raise ValueError(f"{path} holds no results table (a '| scenario | ... | k=N | ... | chosen |' header)")
    header = [cell.strip() for cell in lines[header_at].strip().strip("|").split("|")]
    k_columns = {index: int(name[2:]) for index, name in enumerate(header) if re.fullmatch(r"k=\d+", name)}
    if "chosen" not in header or not k_columns:
        raise ValueError(f"{path}: the results table's header {header} has no k columns or no 'chosen' column")
    chosen_at = header.index("chosen")
    table: dict[str, dict[str, Any]] = {}
    for line in lines[header_at + 2:]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(header):
            raise ValueError(f"{path}: the row {line!r} has {len(cells)} cells, the header {len(header)}")
        printed: dict[int, str] = {}
        for index, k in k_columns.items():
            value = cells[index].replace("**", "").strip()
            if value in ("—", "-", ""):
                continue
            if not re.fullmatch(r"\d+\.\d{2}", value):
                raise ValueError(f"{path}: {cells[0]} k={k} prints {cells[index]!r}, not a two-decimal number")
            printed[k] = value
        chosen = re.search(r"k=(\d+)", cells[chosen_at])
        if chosen is None:
            raise ValueError(f"{path}: {cells[0]}'s chosen cell {cells[chosen_at]!r} names no k")
        table[cells[0]] = {"att_horizon": printed, "chosen_k": int(chosen.group(1))}
    if not table:
        raise ValueError(f"{path}: the results table has no rows")
    return table


def d5_protocol(spec: D5Scenario, *, corpus_root: str | Path = MAIN_CORPUS_ROOT) -> tuple[dict[str, Any], str, str | None]:
    """The env settings, schedule source and plan digest D5's tier was collected with, read from its manifest.

    The scenario's own fixed-time tier (``datasets_v11/<scenario>__fixedtime``) is what D5's tuning chose the cycle
    for, so its manifest is the protocol: the env settings via ``dt_gate.env_settings_from_manifest`` (as the hz4x4
    sweep reads the grid's), and the schedule source and plan digest the swept policy must resolve to -- refused
    unless the source is the one A.7 declares for the scenario.
    """
    import json

    from offline.dt_gate import env_settings_from_manifest

    manifest = Path(corpus_root) / spec.protocol_dir / "manifest.json"
    meta = json.loads(manifest.read_bytes())["run_metadata"]
    if meta.get("behavior_policy") != "fixedtime":
        raise ValueError(f"{manifest} is not a fixed-time tier (behavior_policy {meta.get('behavior_policy')!r})")
    source = meta.get("fixed_time_schedule_source")
    if source != spec.schedule_source:
        raise ValueError(
            f"{manifest} records schedule source {source!r}; A.7 declares {spec.schedule_source!r} for {spec.name}"
        )
    return env_settings_from_manifest(manifest), str(source), meta.get("fixed_time_plan_sha256")


def summarise_admission(
    records: Sequence[Mapping[str, Any]], *, k_grid: Sequence[int], draws: Sequence[int]
) -> dict[int, dict[str, float]]:
    """Per k: the mean ``never_entered`` and the mean ``entered_fraction`` over the draws.

    A.7 asks for both beside the two ATT means, because the definitions part where vehicles are kept out of the
    network.  Every (k, draw) exactly once, every k covering exactly the declared draws, or the mean is refused.
    """
    import numpy as np

    grid = tuple(int(k) for k in k_grid)
    expected = sorted(int(d) for d in draws)
    by_k: dict[int, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        k = int(record["k"])
        draw_id = int(record["draw_id"])
        if k not in grid:
            raise ValueError(f"k = {k} is not in the declared grid {list(grid)}")
        slot = by_k.setdefault(k, {})
        if draw_id in slot:
            raise ValueError(f"k = {k}, draw {draw_id} appears twice")
        slot[draw_id] = record
    summary: dict[int, dict[str, float]] = {}
    for k in grid:
        present = sorted(by_k.get(k, {}))
        if present != expected:
            raise ValueError(f"k = {k} covers draws {present}, not the declared {expected}")
        never = np.asarray([float(by_k[k][d]["never_entered"]) for d in present], dtype=np.float64)
        fraction = np.asarray([float(by_k[k][d]["entered_fraction"]) for d in present], dtype=np.float64)
        summary[k] = {"mean_never_entered": float(never.mean()), "mean_entered_fraction": float(fraction.mean())}
    return summary


def d5_t_regress(summary: Mapping[int, Mapping[str, Any]], recorded: Mapping[int, str]) -> dict[str, Any]:
    """Every per-k mean ``att_ours`` printed to two decimals against D5's printed ``att_horizon``.

    A.7's load-bearing T-regress: one cell that does not print as recorded means the protocol is not D5's, and the
    caller stops.  The swept k grid must be exactly the recorded one -- a missing or extra k is a different protocol,
    refused rather than scored.
    """
    grid = sorted(int(k) for k in recorded)
    if sorted(int(k) for k in summary) != grid:
        raise ValueError(f"the sweep's k grid {sorted(int(k) for k in summary)} is not the recorded table's {grid}")
    cells: dict[int, dict[str, Any]] = {}
    for k in grid:
        mean = float(summary[k]["mean_att_ours"])
        printed = f"{mean:.2f}"
        cells[k] = {
            "mean_att_ours": mean,
            "computed_2dp": printed,
            "recorded": str(recorded[k]),
            "reproduces": printed == str(recorded[k]),
        }
    return {
        "rule": "BRIEF_48 A.7: every per-k mean att_ours, printed to two decimals, equals D5's printed att_horizon; one "
        "cell that does not is a STOP",
        "cells": cells,
        "all_reproduce": all(cell["reproduces"] for cell in cells.values()),
    }


def definition_check(summary: Mapping[int, Mapping[str, Any]], *, k_grid: Sequence[int]) -> dict[str, Any]:
    """Each definition's argmin, AGREE / DEFINITION-DEPENDENT / TIE, and each one's shortfall at the other's argmin.

    The shortfall of ``att_engine`` at ``att_ours``'s argmin is ``mean_att_engine[k_ours] - mean_att_engine[k_engine]``
    (seconds, >= 0), and its relative form divides by ``att_engine``'s own optimum; symmetrically for ``att_ours``.
    An exact tie at either minimum names no argmin and no shortfall (``TIE``): no tie-break is invented.
    """
    grid = tuple(int(k) for k in k_grid)
    missing = [k for k in grid if k not in summary]
    if missing:
        raise ValueError(f"the summary has no record for k = {', '.join(str(k) for k in missing)}")

    def minimisers(key: str) -> tuple[list[int], dict[int, float]]:
        values = {k: float(summary[k][key]) for k in grid}
        best = min(values.values())
        return [k for k in grid if values[k] == best], values

    engine, engine_values = minimisers("mean_att_engine")
    ours, ours_values = minimisers("mean_att_ours")
    check: dict[str, Any] = {
        "k_att_engine": engine[0] if len(engine) == 1 else None,
        "k_att_ours": ours[0] if len(ours) == 1 else None,
        "tied_at_minimum_att_engine": engine if len(engine) > 1 else [],
        "tied_at_minimum_att_ours": ours if len(ours) > 1 else [],
        "att_engine_shortfall_at_att_ours_argmin": None,
        "att_ours_shortfall_at_att_engine_argmin": None,
        "att_engine_relative_shortfall": None,
        "att_ours_relative_shortfall": None,
    }
    if len(engine) > 1 or len(ours) > 1:
        check["status"] = "TIE"
        return check
    k_engine, k_ours = engine[0], ours[0]
    engine_shortfall = engine_values[k_ours] - engine_values[k_engine]
    ours_shortfall = ours_values[k_engine] - ours_values[k_ours]
    check.update(
        {
            "status": "AGREE" if k_engine == k_ours else "DEFINITION-DEPENDENT",
            "att_engine_shortfall_at_att_ours_argmin": engine_shortfall,
            "att_ours_shortfall_at_att_engine_argmin": ours_shortfall,
            "att_engine_relative_shortfall": engine_shortfall / engine_values[k_engine],
            "att_ours_relative_shortfall": ours_shortfall / ours_values[k_ours],
        }
    )
    return check


# ----------------------------------------------------------------------
# The corpus (BRIEF_48 sections 0 and 3.3)
# ----------------------------------------------------------------------


def grid_protocol(corpus_root: str | Path, tier: str) -> list[dict[str, Any]]:
    """The run metadata of every grid directory of *tier*, its invariants checked.

    The grid's committed manifests ARE the protocol (``BRIEF_48`` section 0).  Before anything is derived
    from them, ``offline.recollect_v11.INVARIANTS`` -- the settings this project wrote down for the
    v1.1 campaign -- are asserted on each, and every draw id must lie in the training pool: deriving from a
    manifest inherits whatever is wrong in it, and these are the wrongs that were ever recorded.
    """
    import json

    from offline.recollect_v11 import INVARIANTS

    if tier not in GRID_PROTOCOL_DIRS:
        raise ValueError(
            f"{tier!r} has no grid protocol directory; the grid tiers that are the protocol are "
            f"{sorted(GRID_PROTOCOL_DIRS)} (random_fresh's generator is A27 (d)'s, not a grid manifest's)"
        )
    metadata: list[dict[str, Any]] = []
    for name in GRID_PROTOCOL_DIRS[tier]:
        manifest = Path(corpus_root) / name / "manifest.json"
        meta = json.loads(manifest.read_bytes())["run_metadata"]
        for key, expected in INVARIANTS.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"{manifest}: run_metadata[{key!r}] is {meta.get(key)!r}, expected {expected!r}; the protocol "
                    "manifest does not match the settings the v1.1 campaign is declared to have used"
                )
        ids = [int(d) for d in (meta.get("flow_draw_ids") or []) if d is not None]
        if not ids:
            raise ValueError(f"{manifest}: a nominal run with no flow draws is not the randomised-draw protocol")
        outside = sorted({d for d in ids if not TRAINING_DRAW_MIN <= d <= TRAINING_DRAW_MAX})
        if outside:
            raise ValueError(f"{manifest}: draw ids {outside} lie outside the training pool (D4)")
        metadata.append(meta)
    return metadata


def protocol_argv(
    run_metadata: Mapping[str, Any],
    *,
    env_config: str | Path,
    out_dir: str | Path,
    policy: str,
    episodes: int,
    draw_ids: Sequence[int],
    checkpoint: str | None = None,
    fixed_time_k: int | None = None,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    """``offline.collect`` arguments carrying every setting of a protocol manifest, in ``recollect_v11``'s form.

    Every env and reward setting, the base seed and the three randomiser parameters come from
    *run_metadata*; the caller supplies only what differs by scenario and tier (the config, the out-dir,
    the policy and its episodes, the draws, the checkpoint, the fixed-time k, and *extra* flags).  The
    randomiser parameters are passed explicitly although they equal the collector's defaults, so the
    command reproduces the manifest even if a default ever moved.  Contiguous draws use the half-open
    range flag, anything else is listed, as ``recollect_v11._draw_flags`` does.
    """
    ids = [int(d) for d in draw_ids]
    if not ids:
        raise ValueError("no draw ids: a randomised-draw collection names its draws")
    if len(set(ids)) != len(ids):
        raise ValueError(f"repeated draw ids in {ids}; each draw is deterministic and is collected once")
    assert_training_draws(ids)
    params = run_metadata["flow_randomizer_params"]
    argv: list[str] = [
        "--backend", str(run_metadata["backend"]),
        "--env-config", str(env_config),
        "--policy", str(policy),
        "--episodes", str(int(episodes)),
        "--base-seed", str(int(run_metadata["base_seed"])),
        "--max-steps", str(int(run_metadata["max_steps"])),
        "--delta-time", str(int(run_metadata["delta_time"])),
        "--control-mode", str(run_metadata["control_mode"]),
        "--global-reward-fn", str(run_metadata["global_reward_fn"]),
        "--local-reward-fn", str(run_metadata["local_reward_fn"]),
        "--global-reward-weight", str(float(run_metadata["global_reward_weight"])),
        "--state-features", *[str(feature) for feature in run_metadata["state_features"]],
        "--flow-jitter-sigma", str(float(params["jitter_sigma_s"])),
        "--flow-thin-p", str(float(params["thin_p"])),
        "--flow-volume-scale", str(float(params["volume_scale"])),
        "--out-dir", str(out_dir),
    ]
    if checkpoint is not None:
        argv += ["--checkpoint", str(checkpoint)]
    if fixed_time_k is not None:
        argv += ["--fixed-time-k", str(int(fixed_time_k))]
    argv += [str(token) for token in extra]
    if ids == list(range(ids[0], ids[0] + len(ids))):
        argv += ["--flow-draws-range", str(ids[0]), str(ids[0] + len(ids))]
    else:
        argv += ["--flow-draws", *[str(d) for d in ids]]
    return tuple(argv)


def collector_supports_random_rng(parser: argparse.ArgumentParser | None = None) -> bool:
    """Whether ``offline.collect``'s parser accepts ``--random-rng`` (P9.1's M1).

    Read from the parser the collector actually builds, never assumed from a merge having happened.
    """
    if parser is None:
        from offline.collect import build_parser as collect_parser

        parser = collect_parser()
    return any("--random-rng" in action.option_strings for action in parser._actions)


def derive_runs(
    *,
    corpus_root: str | Path,
    tiers: Sequence[str],
    chosen_k: int | None,
    env_config: str | Path,
    out_root: str | Path,
    random_rng_supported: bool,
) -> list[CollectionRun]:
    """Every hz4x4 collection run of *tiers*, derived from the grid's manifests with the declared substitutions.

    The substitutions, and nothing else (``docs/plans/p9.3.md`` section 4.4): the env config, the out-dir
    ``<out_root>/cf_hz4x4__<tier>[__seed<N>]``, the teacher checkpoint
    ``output/checkpoints/p9_3_mappo_nominal_1000/cf_hz4x4__mappo__seed<N>.pt`` (relative to the main tree, as
    the grid's are), and the fixed-time k the committed sweep chose.  ``random_fresh`` is A27 (d)'s tier --
    policy ``random``, ONE episode per draw on draws 1-200, ``--random-rng per-episode`` -- with the env
    settings of the grid's protocol, and is refused while the collector has no ``--random-rng`` (A27 (i):
    collected only with P9.1's fixed collector).  Runs come out in :data:`TIER_ORDER`, collection order.
    """
    import re

    names = [str(tier) for tier in tiers]
    unknown = [tier for tier in names if tier not in TIER_ORDER]
    if unknown:
        raise ValueError(f"unknown tier(s) {unknown}; hz4x4's tiers are {list(TIER_ORDER)}")
    if len(set(names)) != len(names):
        raise ValueError(f"a tier is named twice in {names}")
    root = Path(out_root)
    runs: list[CollectionRun] = []
    for tier in TIER_ORDER:
        if tier not in names:
            continue
        if tier == "random_fresh":
            if not random_rng_supported:
                raise ValueError(
                    "random_fresh is collected only with P9.1's fixed collector (A27 (d), (i)): offline.collect's "
                    "parser has no --random-rng yet; merge main after P9.1's M1"
                )
            meta = grid_protocol(corpus_root, "maxpressure")[0]
            name = f"{CORPUS_PREFIX}__random_fresh"
            ids = list(range(1, 201))
            argv = protocol_argv(meta, env_config=env_config, out_dir=root / name, policy="random", episodes=1,
                                 draw_ids=ids, extra=("--random-rng", "per-episode"))
            runs.append(CollectionRun(name=name, tier=tier, seed=None, out_dir=root / name, argv=argv,
                                      expected_episodes=len(ids), draw_ids=tuple(ids)))
            continue
        if tier == "fixedtime" and chosen_k is None:
            raise ValueError(
                "the fixedtime tier is collected at the k the committed k-sweep chose (D5: recorded before "
                "collection); there is none"
            )
        metadata = grid_protocol(corpus_root, tier)
        seeds: Sequence[int | None] = SEEDS if tier == "mappo1000" else (None,)
        for seed, meta in zip(seeds, metadata, strict=True):
            ids = [int(d) for d in meta["flow_draw_ids"] if d is not None]
            checkpoint: str | None = None
            name = f"{CORPUS_PREFIX}__{tier}"
            if tier == "mappo1000":
                recorded = str(meta.get("checkpoint") or "")
                if not re.search(rf"seed{seed}\.pt$", recorded):
                    raise ValueError(
                        f"the grid's mappo1000 protocol directory for seed {seed} records checkpoint {recorded!r}"
                    )
                checkpoint = str(TEACHER_CHECKPOINT_SUBDIR / f"cf_hz4x4__mappo__seed{seed}.pt")
                name = f"{CORPUS_PREFIX}__mappo1000__seed{seed}"
            argv = protocol_argv(
                meta,
                env_config=env_config,
                out_dir=root / name,
                policy=str(meta["behavior_policy"]),
                episodes=int(meta["episodes"]),
                draw_ids=ids,
                checkpoint=checkpoint,
                fixed_time_k=chosen_k if tier == "fixedtime" else None,
            )
            runs.append(CollectionRun(name=name, tier=tier, seed=seed, out_dir=root / name, argv=argv,
                                      expected_episodes=len(ids) * int(meta["episodes"]), draw_ids=tuple(ids)))
    return runs


#: Format of the committed ``collect verify`` record (``BRIEF_48`` Amendment B, B.5.1).
ARTIFACT_FORMAT_CORPUS_VERIFY = "p9.3-corpus-verify/1.0"


def corpus_verify_artifact(records: Sequence[Mapping[str, Any]], *, tiers: Sequence[str]) -> dict[str, Any]:
    """The record of one ``collect verify``: every run's verification, the totals and the code trees that collected them.

    ``BRIEF_48`` Amendment B, B.5.1: the verification is committed, not left in a terminal.  Each row is
    :func:`verify_run`'s, which raises on any mismatch, so a record exists only when EVERY run verified; an empty one
    proves nothing and is refused, and a run named twice is refused.  The ``git_hash`` of every manifest is gathered,
    so the record says which code tree collected the corpus.
    """
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("no run was verified; an empty verification record proves nothing")
    names = [str(row["name"]) for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"a run appears twice in {names}")
    return {
        "format_version": ARTIFACT_FORMAT_CORPUS_VERIFY,
        "registered_in": "BRIEF_48 Amendment B, B.2 and B.5.1",
        "label": A27_LABEL,
        "tiers": [str(tier) for tier in tiers],
        "runs": rows,
        "n_runs": len(rows),
        "n_episodes": sum(int(row["episodes"]) for row in rows),
        "n_flow_digests_checked": sum(int(row["flow_digests_checked"]) for row in rows),
        "collected_by_git_hashes": sorted({str(row["git_hash"]) for row in rows}),
        "all_verified": True,
        "checks": "offline.hz4x4_ladder.verify_run per run: the manifest's run metadata against the derived command as "
        "offline.collect's own parser reads it (settings, randomiser parameters, source flow path and sha256, metrics, "
        "epsilon, draws, checkpoint), the episode count, every draw's flow_draw_sha256 against the main tree's "
        "materialised flow.json, and a teacher checkpoint's recorded sha256 against the file",
    }


def assert_targets_absent(runs: Sequence[CollectionRun]) -> None:
    """Refuse when any run's out-dir exists and is not empty -- for ALL runs, before any runs.

    ``recollect_v11``'s rule 2: ``offline.collect`` refuses a populated out-dir, so a command that would
    trip it wastes its slot; and this task creates only NEW ``cf_hz4x4__*`` directories (``BRIEF_48``).
    """
    occupied = [
        str(run.out_dir)
        for run in runs
        if run.out_dir.exists() and (not run.out_dir.is_dir() or any(run.out_dir.iterdir()))
    ]
    if occupied:
        raise FileExistsError(
            f"{len(occupied)} collection target(s) already exist and are not empty: {occupied[:5]}; this task "
            "creates only new cf_hz4x4__* directories and never collects into an existing one"
        )


def check_random_rng_scheme(run_metadata: Mapping[str, Any]) -> None:
    """Refuse a ``random_fresh`` manifest whose generator scheme is not ``per-episode``.

    A27 (d): the registered random tiers replay two schedules because the generator restarts per draw;
    ``random_fresh`` exists to remove that, so a manifest that does not SAY ``per-episode`` -- under the key
    ``random_rng_scheme``, named by ``BRIEF_46`` section 3.3 and ``BRIEF_48`` section 5 -- is not that tier.
    """
    scheme = run_metadata.get("random_rng_scheme")
    if scheme is None:
        raise ValueError(
            "the manifest records no random_rng_scheme, so the generator this random tier was collected with "
            "is unknown; A27 (d) requires 'per-episode'"
        )
    if scheme != "per-episode":
        raise ValueError(
            f"the random tier was collected with random_rng_scheme {scheme!r}; A27 (d) requires 'per-episode' "
            "('per-draw' restarts the generator on every draw, the registered tiers' defect)"
        )


def check_flow_digests(
    manifest: Mapping[str, Any],
    *,
    draws_root: str | Path = MAIN_DRAWS_ROOT,
    scenario_key: str = SCENARIO_KEY,
) -> int:
    """Each collected draw's ``flow_draw_sha256`` against the materialised draw's ``flow.json``; the count checked.

    Two independent materialisations of the same ``(source, base_seed, draw_id)``: ``offline.collect``
    renders its own copy under ``<out-dir>/flows/`` and records its digest, ``offline.materialise_draws``
    wrote the main tree's.  Equal digests on every draw are the evidence that the corpus and the held-out
    evaluation stand on the same demand definition (``DEFERRED`` 55's regeneration argument, measured).
    Per draw first -- a recorded digest, a materialised file, equal digests -- and then the manifest's own
    consistency: a digest recorded for a draw the manifest does not list is refused too.
    """
    import hashlib

    from offline.materialise_draws import draw_dir

    meta = manifest["run_metadata"]
    ids = [int(d) for d in (meta.get("flow_draw_ids") or []) if d is not None]
    digests = {str(key): str(value) for key, value in (meta.get("flow_draw_sha256") or {}).items()}
    if not ids:
        raise ValueError("the manifest names no flow draws; there is nothing to check")
    for draw_id in ids:
        if str(draw_id) not in digests:
            raise ValueError(f"draw {draw_id}: the manifest records no flow_draw_sha256 for it")
        flow = draw_dir(scenario_key, draw_id, out_root=draws_root) / "flow.json"
        if not flow.is_file():
            raise FileNotFoundError(f"draw {draw_id}: no materialised flow at {flow}")
        actual = hashlib.sha256(flow.read_bytes()).hexdigest()
        if actual != digests[str(draw_id)]:
            raise ValueError(
                f"draw {draw_id}: the collector recorded flow sha256 {digests[str(draw_id)]} but the materialised "
                f"{flow} hashes to {actual}; the two materialisations of one draw disagree"
            )
    extra = sorted(set(int(key) for key in digests) - set(ids))
    if extra:
        raise ValueError(f"the manifest's flow_draw_sha256 names draws {extra} that its flow_draw_ids do not list")
    return len(ids)


def distinct_action_sequences(corpus_dir: str | Path) -> dict[str, int]:
    """Per intersection id, the number of distinct action sequences over the directory's episodes.

    One intersection's sequence is its ``(T,)`` action column; it is hashed by P5.3b's
    ``offline.nortg_decomposition.action_sequence_sha256`` (over the int64 bytes), the digest A27 (d)'s
    prediction names.  A27 (d) refuses a ``random_fresh`` tier unless every intersection carries 200.
    """
    from offline.nortg_decomposition import action_sequence_sha256
    from offline.trajectory_logger import load_episode

    paths = sorted(Path(corpus_dir).glob("ep*.npz"))
    if not paths:
        raise ValueError(f"{corpus_dir} holds no episode")
    seen: dict[str, set[str]] = {}
    for path in paths:
        episode = load_episode(path)
        for ix_id in episode.ix_ids:
            actions = episode.intersections[ix_id].action
            seen.setdefault(str(ix_id), set()).add(action_sequence_sha256([[int(a)] for a in actions]))
    return {ix_id: len(digests) for ix_id, digests in seen.items()}


# ----------------------------------------------------------------------
# The teachers (BRIEF_48 section 3.4; RUNSPEC_01 section 4)
# ----------------------------------------------------------------------


def convergence(
    train_returns: Sequence[float],
    *,
    window: int = CONVERGENCE_WINDOW,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    """RUNSPEC_01 section 4's criterion on one training-return curve: the final window AND the preceding one.

    With W = 50 and B the curve's length: ``m1`` = mean over [B-W, B), ``m2`` over [B-2W, B-W), ``m3`` over
    [B-3W, B-2W).  Converged iff ``|m1 - m2| / max(1, |m2|) < 0.05`` AND ``|m2 - m3| / max(1, |m3|) < 0.05``.
    This reading reproduces P2.1's recorded counts at B = 1000 (hz1x1 4/5, grid4x4 3/5, cologne3 0/5), and a
    test holds it to them.  Judged on the TRAINING curve, never on evaluation (RUNSPEC_01).
    """
    values = [float(v) for v in train_returns]
    width = int(window)
    if len(values) < 3 * width:
        raise ValueError(
            f"the criterion compares three consecutive {width}-episode windows, so it needs at least "
            f"{3 * width} episodes; got {len(values)}"
        )
    end = len(values)

    def mean(lo: int, hi: int) -> float:
        return math.fsum(values[lo:hi]) / (hi - lo)

    m_final = mean(end - width, end)
    m_preceding = mean(end - 2 * width, end - width)
    m_earlier = mean(end - 3 * width, end - 2 * width)
    rel_final = abs(m_final - m_preceding) / max(1.0, abs(m_preceding))
    rel_preceding = abs(m_preceding - m_earlier) / max(1.0, abs(m_earlier))
    return {
        "episodes": end,
        "window": width,
        "threshold": float(threshold),
        "m_final": m_final,
        "m_preceding": m_preceding,
        "m_earlier": m_earlier,
        "rel_final": rel_final,
        "rel_preceding": rel_preceding,
        "converged": bool(rel_final < float(threshold) and rel_preceding < float(threshold)),
    }


def teachers_report(
    results_path: str | Path, *, env_id: str = "cf_hz4x4", agent_id: str = "mappo"
) -> dict[str, Any]:
    """Per seed: the curve, ``train_sec`` (an uncontrolled clock), the checkpoint and the criterion.

    Read from the frozen harness's own ``results.json`` (``experiments.runner.run_matrix``): every seed's
    cell must be ``ok``, carry exactly :data:`TEACHER_EPISODES` training returns, and name a checkpoint that
    exists, whose sha256 is recorded.  The label is ``MAPPO@1000`` whatever the criterion says -- tiers are
    labelled by measured return (registration section 5), never by a convergence claim.
    """
    import json

    path = Path(results_path)
    payload = json.loads(path.read_bytes())
    by_seed: dict[int, dict[str, Any]] = {}
    for cell in payload.get("cells", []):
        if cell.get("env_id") != env_id:
            continue
        seed = int(cell["seed"])
        if seed in by_seed:
            raise ValueError(f"{path}: seed {seed} appears twice")
        if cell.get("status") != "ok":
            raise ValueError(f"{path}: seed {seed}'s cell is {cell.get('status')!r} ({cell.get('reason')})")
        policy = cell["policies"][agent_id]
        returns = [float(v) for v in policy["train_returns"]]
        if len(returns) != TEACHER_EPISODES:
            raise ValueError(f"{path}: seed {seed} carries {len(returns)} training returns, not {TEACHER_EPISODES}")
        checkpoint = Path(policy["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{path}: seed {seed}'s checkpoint {checkpoint} does not exist")
        timings = cell["timings"][agent_id]
        by_seed[seed] = {
            "train_returns": returns,
            "train_sec": float(timings["train_sec"]),
            "eval_sec": None if timings.get("eval_sec") is None else float(timings["eval_sec"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "convergence": convergence(returns),
        }
    missing = [seed for seed in SEEDS if seed not in by_seed]
    unexpected = sorted(seed for seed in by_seed if seed not in SEEDS)
    if missing or unexpected:
        raise ValueError(f"{path}: the teachers are seeds {list(SEEDS)}; missing {missing}, unexpected {unexpected}")
    return {
        "format_version": ARTIFACT_FORMAT_TEACHERS,
        "registered_in": "PREREGISTRATION A27 (i); BRIEF_48 section 3.4; RUNSPEC_01 section 4",
        "label": "MAPPO@1000",
        "label_rule": "tiers are labelled by measured return (registration section 5), never by a convergence "
        "claim; the teachers are MAPPO@1000 whatever the criterion says (P2.1's practice)",
        "results": str(path),
        "results_sha256": _sha256_file(path),
        "seeds": {str(seed): by_seed[seed] for seed in SEEDS},
        "converged_seeds": sum(1 for seed in SEEDS if by_seed[seed]["convergence"]["converged"]),
        "n_seeds": len(SEEDS),
        "criterion": "RUNSPEC_01 section 4: W = 50; |m1 - m2| / max(1, |m2|) < 0.05 for the final window AND the "
        "preceding one; P2.1 reported it per seed and resolved the scenario strictly (all seeds)",
        "clock": "train_sec is an uncontrolled clock (A27 (a)): wall-clock around experiments.runner._train_agent "
        "with five teachers training concurrently; no Best Performance regime, no canary, no benchmark claim",
    }


# ----------------------------------------------------------------------
# A11's engine-semantics gate on hz4x4 (offline.engine_att_reference)
# ----------------------------------------------------------------------


def gate_cells_for(tiers: Sequence[str]) -> tuple[Any, ...]:
    """hz4x4's gate cells: every tier x ``GATE_DRAWS``, seed ``GATE_SEED`` unless the tier is deterministic.

    ``engine_att_reference``'s own constants (``GATE_DRAWS`` 1000-1002, ``GATE_SEED`` 101,
    ``DETERMINISTIC_ANCHOR_TIERS``), imported, so the hz4x4 gate is the registered gate on hz4x4's tiers.
    No extreme cells are declared: they were enumerated from ``p8_4a_admission.json``, which has no hz4x4
    row (``docs/plans/p9.3.md`` Q1; :func:`gate_extremes` reports the gate's own).
    """
    from offline.engine_att_reference import (
        BEHAVIOUR_METHOD,
        DETERMINISTIC_ANCHOR_TIERS,
        GATE_DRAWS,
        GATE_SEED,
        GateCell,
    )

    names = [str(tier) for tier in tiers]
    unknown = [tier for tier in names if tier not in TIER_ORDER]
    if unknown:
        raise ValueError(f"unknown tier(s) {unknown}; hz4x4's tiers are {list(TIER_ORDER)}")
    if len(set(names)) != len(names):
        raise ValueError(f"a tier is named twice in {names}; one episode filed twice would be counted twice")
    cells = []
    for tier in names:
        seed = None if tier in DETERMINISTIC_ANCHOR_TIERS else int(GATE_SEED)
        for draw_id in GATE_DRAWS:
            cells.append(GateCell(scenario=SCENARIO, tier=tier, method=BEHAVIOUR_METHOD, seed=seed,
                                  draw_id=int(draw_id), role="tier"))
    return tuple(cells)


def hz4x4_gate_episode(
    *,
    cell: Any,
    config_path: str | Path,
    env_settings: Mapping[str, Any],
    scenario_id: str,
    choose_action_factory: Callable[[Any], Callable[[Any, dict[str, Any]], Any]],
    engine_seed: int,
    created: int,
) -> Any:
    """``engine_att_reference.gate_episode`` for hz4x4, line for line, with hz4x4's env id check.

    The registered function checks the env id against ``GATE_SCENARIOS``, which has no hz4x4 entry, so it
    cannot be called on hz4x4; ``BRIEF_48`` section 0 makes a scenario-parametrised path a new function here,
    never an edit.  Everything else is the registered instrument, imported: the observer env
    (``make_observer_env``), ``horizon_rollout``, ``read_admission_at_horizon``, the engine's own call, and
    ``reconstruct_episode`` -- in the same order, with the same ``finally`` and the same running-count check.
    """
    import time

    from offline.admission_probe import read_admission_at_horizon
    from offline.engine_att_reference import GateEpisode, make_observer_env, reconstruct_episode
    from offline.horizon_metric import horizon_rollout

    if str(cell.scenario) != SCENARIO:
        raise ValueError(f"cell {cell.arm} is on {cell.scenario!r}; this is hz4x4's gate")
    if str(scenario_id) != SCENARIO_ID:
        raise ValueError(
            f"cell {cell.arm} is on hz4x4, whose env id is {SCENARIO_ID!r}, but {scenario_id!r} was supplied; "
            "the two must agree or the episode is filed under the wrong network"
        )
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"draw {cell.draw_id} has no materialised sim config at {path}")

    started = time.perf_counter()
    env = make_observer_env(path, env_settings)
    try:
        choose_action = choose_action_factory(env)
        rollout_started = time.perf_counter()
        rollout = horizon_rollout(env, choose_action, episodes=1, seed=int(engine_seed))
        rollout_seconds = time.perf_counter() - rollout_started
        counts = read_admission_at_horizon(env, created=int(created))
        att_engine_call = float(env._eng.get_average_travel_time())
        built = reconstruct_episode(env.recorder)
    finally:
        env.close()
    seconds = time.perf_counter() - started

    if counts.running != int(rollout.final_vehicle_count):
        raise ValueError(
            f"{cell.arm} seed {cell.seed} draw {cell.draw_id}: the horizon reader reports "
            f"{rollout.final_vehicle_count} running vehicles but the engine reports {counts.running}; nothing may "
            "advance the simulation between the last step and the reconstruction read"
        )
    return GateEpisode(
        scenario=cell.scenario,
        tier=cell.tier,
        method=cell.method,
        arm=cell.arm,
        seed=None if cell.seed is None else int(cell.seed),
        draw_id=int(cell.draw_id),
        role=cell.role,
        engine_seed=int(engine_seed),
        att_reference_engine_population=float(built.engine_population.value),
        att_reference_entered_running=float(built.entered_running.value),
        att_reference_entered_population=float(built.entered_population.value),
        att_reference_metric_cadence=float(built.metric_cadence.value),
        att_engine_call=att_engine_call,
        att_ours=float(rollout.per_episode_horizon[0]),
        n_reference_ids=int(built.engine_population.n_ids),
        n_entered_ids=int(built.entered_running.n_ids),
        created_from_flow=int(created),
        entered=int(counts.entered),
        never_entered=int(counts.never_entered),
        admission_latency_mean=float(built.latency.mean),
        admission_latency_max=float(built.latency.maximum),
        n_admission_delayed=int(built.latency.n_delayed),
        interval=float(built.interval),
        n_observations=int(built.n_observations),
        seconds=seconds,
        seconds_rollout=rollout_seconds,
    )


def gate_extremes(episodes: Sequence[Any]) -> dict[str, Any]:
    """The minimum and maximum ``entered / created`` among the gate's own episodes (Q1 of the plan).

    A11 (4) asks for *"the minimum and maximum entered_fraction episodes available on S"*; on hz4x4 no other
    episode exists at gate time, so they are the gate's own cells -- present by construction, reported so a
    reader sees the censoring range the gate covered.  Ties are broken by the lowest ``(arm, seed, draw_id)``,
    the tie-break ``engine_att_reference.EXTREME_EPISODES`` registered.
    """
    rows = list(episodes)
    if not rows:
        raise ValueError("no gate episodes, so there are no extremes")

    def fraction(episode: Any) -> float:
        return int(episode.entered) / int(episode.created_from_flow)

    def key(episode: Any) -> tuple[str, int, int]:
        return (str(episode.arm), -1 if episode.seed is None else int(episode.seed), int(episode.draw_id))

    def describe(episode: Any, which: str) -> dict[str, Any]:
        return {
            "which": which,
            "tier": episode.tier,
            "method": episode.method,
            "arm": episode.arm,
            "seed": episode.seed,
            "draw_id": int(episode.draw_id),
            "entered": int(episode.entered),
            "created": int(episode.created_from_flow),
            "entered_fraction": fraction(episode),
        }

    lowest = min(fraction(e) for e in rows)
    highest = max(fraction(e) for e in rows)
    minimum = min((e for e in rows if fraction(e) == lowest), key=key)
    maximum = min((e for e in rows if fraction(e) == highest), key=key)
    return {
        "rule": "the minimum and maximum entered / created among the gate's own cells (no other hz4x4 episode "
        "exists at gate time); ties broken by the lowest (arm, seed, draw_id)",
        "min": describe(minimum, "min"),
        "max": describe(maximum, "max"),
    }


def gate_outcome(criteria: Any) -> dict[str, Any]:
    """A11's branch for hz4x4: PASS makes ``att_engine`` primary; FAIL reports both definitions.

    *criteria* is ``engine_att_reference.evaluate_scenario``'s ``ScenarioCriteria``, scored by the registered
    function itself.  This maps it onto A11's two consequence sentences and names the failed criteria; (3c)
    is reported and never gating (A12).  A ``passed`` flag that disagrees with its own criteria is refused.
    """
    named = (
        ("c1", criteria.c1_passed),
        ("c2", criteria.c2_exact),
        ("c3a", criteria.c3a_passed),
        ("c3b", criteria.c3b_passed),
        ("c4", criteria.c4_passed),
    )
    failed = [name for name, ok in named if not ok]
    passed = bool(criteria.passed)
    if passed != (not failed):
        raise ValueError(f"ScenarioCriteria.passed is {passed} but the failed criteria are {failed}")
    return {
        "passed": passed,
        "failed_criteria": failed,
        "primary": "att_engine" if passed else None,
        "consequence": A11_CONSEQUENCE_PASS if passed else A11_CONSEQUENCE_FAIL,
        "c3c_is_gating": False,
    }


# ----------------------------------------------------------------------
# The declaration (BRIEF_48 section 3.3)
# ----------------------------------------------------------------------


def _tier_episode_paths(dirs: Sequence[str | Path]) -> list[Path]:
    """Every ``ep*.npz`` of *dirs*, directories and episodes in sorted order -- ``att_ladder.tier_cells``' order."""
    paths: list[Path] = []
    for directory in sorted(Path(d) for d in dirs):
        episodes = sorted(directory.glob("ep*.npz"))
        if not episodes:
            raise ValueError(f"{directory} holds no episode")
        paths.extend(episodes)
    return paths


def ladder_att(dirs: Sequence[str | Path]) -> float:
    """``att_ladder``'s ``att_horizon_mean`` over the episodes of *dirs*: the mean of ``att_per_step[-1]``.

    ``offline.att_ladder.tier_cells`` reads a whole corpus root; this is its per-tier reduction -- the same
    order (sorted directories, sorted episodes), the same ``float64`` array, the same ``.mean()`` -- so a
    test can hold it to ``tier_sweep.TIERS``' recorded values under ``==`` on the grid, and the declaration
    run computes the hz4x4 value by ``tier_cells`` itself as its second route.
    """
    import numpy as np

    from offline.trajectory_logger import load_episode

    values: list[float] = []
    for path in _tier_episode_paths(dirs):
        episode = load_episode(path)
        if episode.att_per_step is None:
            raise ValueError(f"{path}: format {episode.format_version} has no att_per_step")
        values.append(float(episode.att_per_step[-1]))
    return float(np.asarray(values, dtype=np.float64).mean())


def mean_training_return(dirs: Sequence[str | Path]) -> float:
    """P5.2's mean training-set return: the mean over every stream of ``numpy.sum`` of its local reward.

    ``docs/plans/p5.2.md`` section 2.2's route, restated as code: ``numpy.sum`` over each episode's
    per-intersection ``local_reward`` array, over every episode x intersection stream of the tier.  A test
    holds it to ``tier_sweep.TIERS`` at the one decimal those values were recorded with.
    """
    import numpy as np

    from offline.trajectory_logger import load_episode

    totals: list[float] = []
    for path in _tier_episode_paths(dirs):
        episode = load_episode(path)
        for ix_id in episode.ix_ids:
            totals.append(float(np.sum(episode.intersections[ix_id].local_reward)))
    return float(np.mean(np.asarray(totals, dtype=np.float64)))


def tier_row_counts(dirs: Sequence[str | Path]) -> dict[str, int]:
    """Episodes, streams, decisions and the two top-decile row counts of one tier's training set.

    Counted from the episodes, never from the brief: decisions = sum of episode length x intersections.
    The top-decile rules are the registered filters' -- ``ceil(0.1 x n)`` streams and never fewer than one,
    over all streams for %BC (``offline_baselines.top_return_streams``) and per intersection for the
    per-intersection arm (``tier_sweep.per_intersection_top_streams``) -- times the episode length.
    """
    from offline.offline_baselines import TOP_RETURN_FRACTION
    from offline.trajectory_logger import load_episode

    episodes = 0
    decisions = 0
    lengths: set[int] = set()
    widths: set[int] = set()
    per_ix: dict[str, int] = {}
    for path in _tier_episode_paths(dirs):
        episode = load_episode(path)
        episodes += 1
        length = int(episode.episode_length)
        lengths.add(length)
        widths.add(len(episode.ix_ids))
        for ix_id in episode.ix_ids:
            per_ix[str(ix_id)] = per_ix.get(str(ix_id), 0) + 1
            decisions += length
    if len(lengths) != 1 or len(widths) != 1:
        raise ValueError(f"episode lengths {sorted(lengths)} / intersection counts {sorted(widths)} are not uniform")
    length = lengths.pop()
    streams = sum(per_ix.values())
    top_global = max(1, math.ceil(TOP_RETURN_FRACTION * streams)) * length
    top_perix = sum(max(1, math.ceil(TOP_RETURN_FRACTION * n)) for n in per_ix.values()) * length
    return {
        "episodes": episodes,
        "n_intersections": widths.pop(),
        "episode_length": length,
        "streams": streams,
        "decisions": decisions,
        "top_decile_decisions_global": top_global,
        "top_decile_decisions_perix": top_perix,
    }


def exposure_table(row_counts: Mapping[str, int], *, steps: int | None = None) -> dict[str, Any]:
    """A27(b)'s exposure (row visits = steps x decisions per step / decisions in the training set), per arm.

    Rows per step from the code constants -- ``tier_sweep.JOINT_BATCH_SIZE``, ``CONTEXT_LENGTH``,
    ``DECLARED_GRADIENT_STEPS``; ``offline_baselines.BC_BATCH_WINDOWS``, ``IQL_BATCH_TRANSITIONS`` -- and
    A27(b)'s matched batches (:data:`MATCHED_BATCH`); training decisions from *row_counts*, measured.  Every
    visit count is ``steps * rows / decisions`` in exact integer arithmetic with one correctly rounded
    division, so ``float(Fraction(steps * rows, decisions))`` reproduces it under ``==``.
    """
    from offline.offline_baselines import BC_BATCH_WINDOWS, IQL_BATCH_TRANSITIONS
    from offline.tier_sweep import CONTEXT_LENGTH, DECLARED_GRADIENT_STEPS, JOINT_BATCH_SIZE

    total_steps = int(DECLARED_GRADIENT_STEPS if steps is None else steps)
    decisions = int(row_counts["decisions"])
    n_ix = int(row_counts["n_intersections"])
    top_global = int(row_counts["top_decile_decisions_global"])
    top_perix = int(row_counts["top_decile_decisions_perix"])
    if decisions <= 0:
        raise ValueError(f"a training set of {decisions} decisions has no exposure")
    for label, value in (("global", top_global), ("per-intersection", top_perix)):
        if not 0 < value <= decisions:
            raise ValueError(f"the {label} top-decile row count {value} is not within (0, {decisions}] decisions")
    window = int(CONTEXT_LENGTH)
    arms = {
        "dt_nomix": (int(JOINT_BATCH_SIZE) * window * n_ix, decisions, "registered",
                     "tier_sweep.JOINT_BATCH_SIZE x CONTEXT_LENGTH x intersections"),
        "bc": (int(BC_BATCH_WINDOWS) * window, decisions, "registered",
               "offline_baselines.BC_BATCH_WINDOWS x CONTEXT_LENGTH"),
        "bc_top10": (int(BC_BATCH_WINDOWS) * window, top_global, "registered",
                     "offline_baselines.BC_BATCH_WINDOWS x CONTEXT_LENGTH, on the global top decile"),
        "bc_top10_perix": (int(BC_BATCH_WINDOWS) * window, top_perix, "registered",
                           "offline_baselines.BC_BATCH_WINDOWS x CONTEXT_LENGTH, on the per-intersection top decile"),
        "iql": (int(IQL_BATCH_TRANSITIONS), decisions, "registered", "offline_baselines.IQL_BATCH_TRANSITIONS"),
        "bc_m": (int(MATCHED_BATCH["bc_m_windows"]) * window, decisions, "matched (A27 (b))",
                 "1,024 windows x CONTEXT_LENGTH"),
        "bc_top10_m": (int(MATCHED_BATCH["bc_top10_m_windows"]) * window, top_global, "matched (A27 (b))",
                       "102 windows x CONTEXT_LENGTH (102.4 rounded down, declared), on the global top decile"),
        "bc_top10_perix_m": (int(MATCHED_BATCH["bc_top10_m_windows"]) * window, top_perix, "matched (A27 (b))",
                             "102 windows x CONTEXT_LENGTH, on the per-intersection top decile"),
        "iql_m": (int(MATCHED_BATCH["iql_m_transitions"]), decisions, "matched (A27 (b))", "20,480 transitions"),
    }
    return {
        "definition": "row visits = steps x decisions per step / decisions in the training set (A27 (b))",
        "steps": total_steps,
        "row_counts": {key: int(value) for key, value in row_counts.items()},
        "arms": {
            name: {
                "rows_per_step": rows,
                "training_decisions": basis,
                "visits": total_steps * rows / basis,
                "batch": kind,
                "rows_per_step_from": source,
            }
            for name, (rows, basis, kind, source) in arms.items()
        },
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The ``python -m offline.hz4x4_ladder`` parser.

    Every subcommand takes ``--torch-threads`` (default 1, ``BRIEF_48``'s "everywhere") and the three
    main-tree roots, so a command line names every place it reads or writes.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--torch-threads", type=int, default=1, help="torch intra-op threads (default 1)")
    common.add_argument("--draws-root", default=str(MAIN_DRAWS_ROOT), help="the main tree's materialised draws")
    common.add_argument("--corpus-root", default=str(MAIN_CORPUS_ROOT), help="the main tree's datasets_v11")
    common.add_argument("--output-root", default=str(MAIN_OUTPUT_ROOT), help="the main tree's output/")

    parser = argparse.ArgumentParser(
        prog="python -m offline.hz4x4_ladder",
        description="P9.3 (A27 point 8): hz4x4 draws, pre-flight, fixed-time sweep, corpus, A11 gate, declaration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    draws = sub.add_parser("draws", parents=[common], help="materialise draws 0-200 and 1000-1099 (BRIEF_48 3.2)")
    draws.add_argument("--draws-range", type=int, nargs=2, action="append", metavar=("START", "END"),
                       help="half-open [START, END); repeatable; default: 0 201 and 1000 1100")
    draws.add_argument("--dry-run", action="store_true", help="validate and plan, write nothing")

    preflight = sub.add_parser("preflight", parents=[common], help="the <= 1 h pre-flight (BRIEF_48 3.1)")
    preflight.add_argument("--out", required=True, help="docs/data/p9_3_preflight.json")
    preflight.add_argument("--episodes", type=int, default=PREFLIGHT_MAPPO_EPISODES,
                           help=f"MAPPO training episodes per seed (default {PREFLIGHT_MAPPO_EPISODES})")

    sweep = sub.add_parser("k-sweep", parents=[common], help="D5's fixed-time k sweep on draws 1-5")
    sweep.add_argument("--out", required=True, help="docs/data/p9_3_fixed_time_sweep.json")

    collect = sub.add_parser("collect", parents=[common], help="the four tiers by offline.collect")
    collect.add_argument("action", choices=("plan", "run", "verify"))
    collect.add_argument("--sweep", required=True, help="the committed docs/data/p9_3_fixed_time_sweep.json")
    collect.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=None,
                         help="default: every tier the collector can serve, in collection order")
    collect.add_argument("--python", default="/home/filip/rltraffic/.venv/bin/python",
                         help="the interpreter each collection subprocess runs")
    collect.add_argument("--ruling", default=None,
                         help="the committed fixed-time ruling file, needed when the sweep chose no k (A.4)")
    collect.add_argument("--out", default=None,
                         help="with verify: write the verification record here, once (BRIEF_48 B.5.1)")

    teachers = sub.add_parser("teachers", parents=[common], help="the teachers' curves and RUNSPEC_01's criterion")
    teachers.add_argument("--out", required=True, help="docs/data/p9_3_teachers.json")

    gate = sub.add_parser("gate", parents=[common], help="A11's engine-semantics gate on hz4x4")
    gate.add_argument("--out", required=True, help="docs/data/p9_3_engine_att_gate.json")
    gate.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=None)
    gate.add_argument("--device", default="cpu")

    declare = sub.add_parser("declare", parents=[common], help="tier specs and the exposure table")
    declare.add_argument("--out", required=True, help="docs/data/p9_3_declaration.json")
    declare.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=None)

    ruling = sub.add_parser("record-ruling", parents=[common],
                            help="write the fixed-time ruling file once the amendment carrying it is on disk")
    ruling.add_argument("--sweep", required=True, help="the committed docs/data/p9_3_fixed_time_sweep.json")
    ruling.add_argument("--k", type=int, required=True, help="the ruled fixed-time k")
    ruling.add_argument("--amendment", required=True, help="the amendment heading that carries the ruling")
    ruling.add_argument("--brief", default=None, help="default: this repository's docs/briefs/BRIEF_48_p9.3_a27c_hz4x4.md")
    ruling.add_argument("--out", required=True, help="docs/data/p9_3_fixed_time_ruling.json")

    d5 = sub.add_parser("d5-check", parents=[common],
                        help="A.7: D5's fixed-time sweep on hz1x1 and grid4x4 under both ATT definitions (exploration)")
    d5.add_argument("--out", required=True, help="docs/data/p9_3_d5_definition_check.json")
    return parser


def _run_draws(args: argparse.Namespace) -> int:
    """``draws``: both registered pools by default, through :func:`materialise_hz4x4_draws`."""
    ranges = args.draws_range or [[0, 201], [1000, 1100]]
    ids: list[int] = []
    for start, end in ranges:
        if int(end) <= int(start):
            raise ValueError(f"--draws-range is half-open [START, END); [{start}, {end}) selects nothing")
        ids.extend(range(int(start), int(end)))
    records = materialise_hz4x4_draws(ids, out_root=args.draws_root, dry_run=bool(args.dry_run))
    actions: dict[str, int] = {}
    for record in records:
        actions[record.action] = actions.get(record.action, 0) + 1
        print(f"{record.scenario_key} draw {record.draw_id:>4} [{record.pool}] {record.action}: "
              f"{record.n_vehicles} vehicles -> {record.directory}", flush=True)
    if args.dry_run:
        print(f"dry run: {len(records)} draws {actions}; planned in memory and matched the materialiser's own "
              "build, nothing written and nothing verified on disk", flush=True)
    else:
        print(f"done: {len(records)} draws {actions}; every flow.json verified on disk: single-insertion and equal "
              "to the in-memory plan", flush=True)
    return 0


def _run_preflight(args: argparse.Namespace) -> int:
    """``preflight``: the canary, the hz4x4 facts, the CityFlow and MAPPO timings, the projection, the decision."""
    import concurrent.futures
    import multiprocessing
    import os

    from experiments.config import load_config
    from offline.materialise_draws import draw_config_path

    out = _refuse_existing(args.out)
    draws_root = assert_main_tree_path(args.draws_root, what="the draws root")
    configs = {draw_id: draw_config_path(SCENARIO_KEY, draw_id, out_root=draws_root) for draw_id in (0, 1)}
    missing = [str(path) for path in configs.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"materialise the draws first (the draws subcommand); missing {missing}")
    settings = _protocol_env_settings(args.corpus_root)
    flow_path, _raw, entries = source_flow_entries(SOURCE_CONFIG)
    teacher_config = _repo_root() / TEACHER_CONFIG
    teacher = load_config(teacher_config)
    env_specs = [spec for spec in teacher.environments if spec.id == "cf_hz4x4"]
    if len(env_specs) != 1:
        raise ValueError(f"{teacher_config} must declare one cf_hz4x4 environment")
    teacher_settings = env_specs[0].settings
    evaluation_episodes = int(teacher_settings["eval_episodes"]) * (1 + len(teacher_settings["compare_with"]))
    main_config = MAIN_TREE / SOURCE_CONFIG
    code_config = _repo_root() / SOURCE_CONFIG

    canaries = [canary_record("open")]
    print(f"canary open: {canaries[0]['seconds']:.2f} s, {canaries[0]['verdict']}, "
          f"reproduced={canaries[0]['reproduced']}", flush=True)
    cityflow = []
    for draw_id in (0, 1):
        timing = time_maxpressure_episode(configs[draw_id], settings)
        cityflow.append({"draw_id": draw_id, "config": str(configs[draw_id]), **timing})
        print(f"maxpressure draw {draw_id}: {timing['seconds']:.2f} s ({timing['decisions']} decisions)", flush=True)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(SEEDS), mp_context=context) as pool:
        futures = [
            pool.submit(mappo_preflight_worker, str(teacher_config), seed, int(args.episodes),
                        str(configs[1]), dict(settings))
            for seed in SEEDS
        ]
        timings = [future.result() for future in futures]
    for timing in timings:
        print(f"mappo seed {timing['seed']}: {timing['seconds']:.1f} s over {timing['episodes']} episodes", flush=True)
    canaries.append(canary_record("close"))
    print(f"canary close: {canaries[1]['seconds']:.2f} s, {canaries[1]['verdict']}, "
          f"reproduced={canaries[1]['reproduced']}", flush=True)

    projection = project_teachers(timings, train_episodes=int(teacher_settings["train_episodes"]),
                                  evaluation_episodes=evaluation_episodes)
    decision = preflight_decision(projection, canaries)
    grid_seconds = _grid_mappo1000_seconds()
    artifact = {
        "format_version": ARTIFACT_FORMAT_PREFLIGHT,
        "registered_in": "PREREGISTRATION A27 (i); BRIEF_48 section 3.1",
        "label": A27_LABEL,
        "produces_no_registered_number": "timings and counts only: no travel time and no return is read or kept",
        "git": _git(),
        "machine": {"cpu_count": os.cpu_count(), "torch_threads_per_process": 1,
                    "mappo_processes": len(SEEDS), "start_method": "spawn"},
        "canaries": canaries,
        "hz4x4": {
            "source_flow": str(flow_path),
            "source_vehicles": len(entries),
            "single_insertion": True,
            "action_counts": cityflow[0]["action_counts"],
            "intersections": len(cityflow[0]["action_counts"]),
            "config_sha256_main_tree": _sha256_file(main_config),
            "config_sha256_code_tree": _sha256_file(code_config),
            "config_identical": _sha256_file(main_config) == _sha256_file(code_config),
        },
        "cityflow_maxpressure_episodes": cityflow,
        "mappo": {
            "config": str(teacher_config),
            "config_sha256": _sha256_file(teacher_config),
            "episodes_timed_per_seed": int(args.episodes),
            "per_seed": timings,
            "projection": projection,
        },
        "decision": decision,
        "workers": {
            "teachers": len(SEEDS),
            "teachers_reason": "one per seed (--workers 5), the regime timed above",
            "corpus": 1,
            "corpus_reason": "sequential, abort on the first failure (recollect_v11's campaign rules): eight runs, "
            "about 1,640 episodes",
        },
        "grid_records_for_comparison": {
            "mappo1000_train_sec_median": grid_seconds,
            "mappo1000_source": "docs/data/p8_2_compute.json rows[id == grid4x4.mappo1000].training[0].seconds.median",
            "maxpressure_episode_seconds": 1.16,
            "maxpressure_source": "the coordinator, 2026-10-15, as BRIEF_48 section 3.1 quotes it; not re-measured here",
            "note": "beside the hz4x4 numbers for scale only; no cross-session ratio is formed (PROJECT_PLAN section 7)",
        },
    }
    _write_artifact(artifact, out)
    print(f"wrote {out}: projected {projection['projected_wall_clock_seconds']:.0f} s, "
          f"decision {decision}", flush=True)
    return 0 if decision["quotable"] else 2


def _grid_mappo1000_seconds() -> float:
    """The grid MAPPO@1000 median ``train_sec`` read from ``docs/data/p8_2_compute.json``, never retyped."""
    import json

    payload = json.loads((_repo_root() / "docs" / "data" / "p8_2_compute.json").read_bytes())
    rows = [row for row in payload["rows"] if row.get("id") == "grid4x4.mappo1000"]
    if len(rows) != 1:
        raise ValueError("docs/data/p8_2_compute.json does not hold exactly one grid4x4.mappo1000 row")
    return float(rows[0]["training"][0]["seconds"]["median"])


def _tier_dir_names(tier: str) -> list[str]:
    """The corpus directory names of one hz4x4 tier: one per seed for ``mappo1000``, one otherwise."""
    if tier not in TIER_ORDER:
        raise ValueError(f"unknown tier {tier!r}; hz4x4's tiers are {list(TIER_ORDER)}")
    if tier == "mappo1000":
        return [f"{CORPUS_PREFIX}__mappo1000__seed{seed}" for seed in SEEDS]
    return [f"{CORPUS_PREFIX}__{tier}"]


def _collected_tiers(corpus_root: Path) -> list[str]:
    """The tiers whose every directory has a manifest, in collection order."""
    return [
        tier
        for tier in TIER_ORDER
        if all((corpus_root / name / "manifest.json").is_file() for name in _tier_dir_names(tier))
    ]


def _collection_command(run: CollectionRun, python: str) -> list[str]:
    """The full command one collection run executes: this branch's ``offline.collect``, ``-P``, from the main tree."""
    return [str(python), "-P", "-m", "offline.collect", *run.argv]


def _checkpoint_of(run: CollectionRun) -> str | None:
    argv = list(run.argv)
    return argv[argv.index("--checkpoint") + 1] if "--checkpoint" in argv else None


def verify_run(run: CollectionRun, *, draws_root: str | Path = MAIN_DRAWS_ROOT) -> dict[str, Any]:
    """After a run: its manifest equals the derived command, its draws equal the materialised ones.

    The manifest's run metadata is compared with the derived command as ``offline.collect``'s own parser
    reads it -- every setting the command determines, the randomiser's parameters, the source flow's path and digest,
    ``metrics`` and ``epsilon`` included (A.2 finding 2); every draw's ``flow_draw_sha256`` with the main tree's
    materialised ``flow.json``; a teacher
    checkpoint's recorded sha256 with the file; and for ``random_fresh`` the generator scheme
    (``per-episode``) and A27 (d)'s registered check -- one distinct action sequence per episode on every
    intersection.
    """
    import json

    from offline.collect import _cityflow_flow_source
    from offline.collect import build_parser as collect_parser

    manifest = json.loads((run.out_dir / "manifest.json").read_bytes())
    meta = manifest["run_metadata"]
    if len(manifest["episodes"]) != run.expected_episodes:
        raise ValueError(f"{run.name}: {len(manifest['episodes'])} episodes on disk, {run.expected_episodes} expected")
    args = collect_parser().parse_args(list(run.argv))
    # The collector's own resolution of the source flow (the working directory resolves a relative 'dir'), so the
    # expected path and digest are the ones the collector recorded -- BRIEF_48 Amendment A, A.2 finding 2.
    flow_source = _cityflow_flow_source(Path(args.env_config).resolve())
    expected: dict[str, Any] = {
        "backend": "cityflow",
        "scenario_id": SCENARIO_ID,
        "behavior_policy": args.policy,
        "episodes": args.episodes,
        "base_seed": args.base_seed,
        "max_steps": args.max_steps,
        "delta_time": args.delta_time,
        "control_mode": args.control_mode,
        "global_reward_fn": args.global_reward_fn,
        "local_reward_fn": args.local_reward_fn,
        "global_reward_weight": args.global_reward_weight,
        "state_features": list(args.state_features),
        "checkpoint": args.checkpoint,
        "flow_draw_ids": list(run.draw_ids),
        "env_paths": {"config": str(Path(args.env_config).resolve())},
        "flow_randomizer_params": {
            "base_seed": int(args.base_seed),
            "jitter_sigma_s": float(args.flow_jitter_sigma),
            "thin_p": float(args.flow_thin_p),
            "volume_scale": float(args.flow_volume_scale),
        },
        "flow_source_path": str(flow_source),
        "flow_source_sha256": _sha256_file(flow_source),
        "metrics": list(args.metrics) if args.metrics else None,
        "epsilon": float(args.epsilon) if args.policy == "mappo_eps" else None,
    }
    if run.tier == "fixedtime":
        expected["fixed_time_k"] = args.fixed_time_k
        expected["fixed_time_schedule_source"] = "equal_split"
    differing = {key: {"manifest": meta.get(key), "expected": value} for key, value in expected.items()
                 if meta.get(key) != value}
    if differing:
        raise ValueError(f"{run.name}: the manifest differs from the derived command: {differing}")
    record: dict[str, Any] = {
        "name": run.name,
        "episodes": len(manifest["episodes"]),
        "flow_digests_checked": check_flow_digests(manifest, draws_root=draws_root),
        "git_hash": manifest.get("git_hash"),
    }
    if run.tier == "mappo1000":
        checkpoint = MAIN_TREE / str(meta["checkpoint"])
        if _sha256_file(checkpoint) != meta.get("checkpoint_sha256"):
            raise ValueError(f"{run.name}: {checkpoint} is not the checkpoint the manifest records")
        record["checkpoint_sha256"] = meta["checkpoint_sha256"]
    if run.tier == "random_fresh":
        check_random_rng_scheme(meta)
        counts = distinct_action_sequences(run.out_dir)
        short = {ix_id: n for ix_id, n in counts.items() if n != run.expected_episodes}
        if len(counts) != EXPECTED_INTERSECTIONS or short:
            raise ValueError(
                f"{run.name}: A27 (d) refuses a random_fresh tier without {run.expected_episodes} distinct "
                f"per-intersection action sequences; these fall short: {short}"
            )
        record["distinct_sequences_per_intersection"] = counts
        record["random_rng_scheme"] = meta["random_rng_scheme"]
    return record


def _run_k_sweep(args: argparse.Namespace) -> int:
    """``k-sweep``: D5 on draws 1-5, both definitions, the declared choice rule, written before any collection."""
    from offline.materialise_draws import draw_config_path

    out = _refuse_existing(args.out)
    draws_root = assert_main_tree_path(args.draws_root, what="the draws root")
    assert_training_draws(K_SWEEP_DRAWS)
    manifest = Path(args.corpus_root) / "cf_grid4x4__fixedtime" / "manifest.json"
    settings = _protocol_env_settings(args.corpus_root, "cf_grid4x4__fixedtime")
    configs = {d: draw_config_path(SCENARIO_KEY, d, out_root=draws_root) for d in K_SWEEP_DRAWS}
    missing = [str(path) for path in configs.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"materialise the draws first (the draws subcommand); missing {missing}")
    records: list[dict[str, Any]] = []
    for k in K_GRID:
        for draw_id in K_SWEEP_DRAWS:
            record = sweep_episode(k, draw_id, config_path=configs[draw_id], env_settings=settings)
            records.append(record)
            print(f"k={k:>2} draw {draw_id}: att_engine {record['att_engine']:.4f}  att_ours {record['att_ours']:.4f}  "
                  f"({record['seconds']:.2f} s)", flush=True)
    summary = summarise_sweep(records)
    decision = choose_k(summary)
    artifact = {
        "format_version": ARTIFACT_FORMAT_SWEEP,
        "registered_in": "D5 (docs/data/fixed_time_sweep/README.md); PREREGISTRATION A27 (i); BRIEF_48 section 0",
        "label": A27_LABEL,
        "protocol": {
            "k_grid": list(K_GRID),
            "draws": list(K_SWEEP_DRAWS),
            "engine_seed": BASE_SEED,
            "episodes_per_draw": 1,
            "policy": "offline.policies.fixed_time.make_fixedtime, the collector's own, equal split (no shipped plan)",
            "cycle": "n_greens x delta_time x k with n_greens = 8",
            "route": "offline.admission_probe.probe_episode: att_ours = horizon_rollout's att_horizon (A1), "
            "att_engine = Engine::getAverageTravelTime at the horizon (A11); created from the draw's flow.json",
            "env_settings_from": str(manifest),
            "env_settings": settings,
        },
        "objective": "the mean over draws 1-5 under each definition; Rule R has not run on hz4x4 when k is chosen, "
        "so neither definition is primary and the two argmins must agree (docs/plans/p9.3.md section 4.3, Q2)",
        "git": _git(),
        "records": records,
        "summary": {str(k): summary[k] for k in K_GRID},
        "decision": decision,
        "recorded_before_collection": "committed before any hz4x4 corpus run; the collect subcommand reads the "
        "fixed-time k from this file and from nowhere else",
    }
    _write_artifact(artifact, out)
    print(f"wrote {out}: {decision['status']}, chosen k = {decision['chosen_k']} "
          f"(att_engine argmin {decision['k_att_engine']}, att_ours argmin {decision['k_att_ours']})", flush=True)
    return 0 if decision["status"] == "CHOSEN" else 2


def _run_collect(args: argparse.Namespace) -> int:
    """``collect plan|run|verify``: the tiers by ``offline.collect``, derived, guarded, sequential."""
    import json
    import os
    import subprocess

    from offline.materialise_draws import draw_config_path

    if args.out and args.action != "verify":
        raise ValueError("--out names collect verify's record; plan and run write no record")
    out = _refuse_existing(args.out) if args.out else None
    supported = collector_supports_random_rng()
    if args.tiers:
        tiers = list(args.tiers)
    else:
        tiers = [tier for tier in TIER_ORDER if tier != "random_fresh" or supported]
        if not supported:
            print("random_fresh is left out: offline.collect has no --random-rng yet (P9.1's M1); it is "
                  "collected last (A27 (i))", flush=True)
    chosen_k = resolve_fixed_time_k(args.sweep, args.ruling) if "fixedtime" in tiers else None
    corpus_root = assert_main_tree_path(args.corpus_root, what="the corpus root")
    draws_root = assert_main_tree_path(args.draws_root, what="the draws root")
    env_config = assert_main_tree_path(MAIN_TREE / SOURCE_CONFIG, what="the source sim config")
    runs = derive_runs(corpus_root=corpus_root, tiers=tiers, chosen_k=chosen_k, env_config=env_config,
                       out_root=corpus_root, random_rng_supported=supported)

    if args.action == "verify":
        records = []
        for run in runs:
            record = verify_run(run, draws_root=draws_root)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        print(f"verified {len(runs)} run(s)", flush=True)
        if out is not None:
            artifact = corpus_verify_artifact(records, tiers=tiers)
            artifact["git"] = _git()
            _write_artifact(artifact, out)
            print(f"wrote {out}", flush=True)
        return 0

    missing_checkpoints = [str(MAIN_TREE / c) for c in (_checkpoint_of(r) for r in runs)
                           if c is not None and not (MAIN_TREE / c).is_file()]
    missing_draws = sorted({d for r in runs for d in r.draw_ids
                            if not draw_config_path(SCENARIO_KEY, d, out_root=draws_root).is_file()})
    if args.action == "plan":
        for run in runs:
            print(f"{run.name}: {run.expected_episodes} episodes\n  " + " ".join(_collection_command(run, args.python)),
                  flush=True)
        print(f"plan: {len(runs)} run(s), {sum(r.expected_episodes for r in runs)} episodes; missing checkpoints "
              f"{missing_checkpoints}; draws not materialised {missing_draws}", flush=True)
        return 0

    assert_targets_absent(runs)
    if missing_checkpoints:
        raise FileNotFoundError(f"the teachers' checkpoints are missing: {missing_checkpoints}")
    if missing_draws:
        raise FileNotFoundError(f"draws {missing_draws[:5]} are not materialised under {draws_root}")
    env = dict(os.environ, PYTHONPATH=str(_repo_root()))
    for index, run in enumerate(runs, start=1):
        command = _collection_command(run, args.python)
        print(f"\n[{index}/{len(runs)}] {run.name}\n  {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=MAIN_TREE, env=env, check=False)
        if result.returncode != 0:
            print(f"ABORT: {run.name} exited {result.returncode}; {index - 1}/{len(runs)} run(s) completed", flush=True)
            return 1
        written = len(json.loads((run.out_dir / "manifest.json").read_bytes())["episodes"])
        if written != run.expected_episodes:
            print(f"ABORT: {run.name} wrote {written} episodes, {run.expected_episodes} expected", flush=True)
            return 1
    print(f"\nOK: {len(runs)}/{len(runs)} run(s), {sum(r.expected_episodes for r in runs)} episodes; next: "
          "collect verify", flush=True)
    return 0


def _run_teachers(args: argparse.Namespace) -> int:
    """``teachers``: the frozen harness's results.json turned into the curve-and-criterion record."""
    out = _refuse_existing(args.out)
    results = Path(args.output_root) / TEACHER_RESULTS.relative_to("output")
    report = teachers_report(results)
    report["git"] = _git()
    _write_artifact(report, out)
    print(f"wrote {out}: {report['converged_seeds']}/{report['n_seeds']} seeds meet RUNSPEC_01's criterion; "
          f"label {report['label']}", flush=True)
    return 0


def _run_gate(args: argparse.Namespace) -> int:
    """``gate``: A11's four criteria on hz4x4's collected tiers, scored by ``evaluate_scenario`` unchanged."""
    from offline import engine_att_reference as engine
    from offline.admission_probe import created_from_flow
    from offline.dt_gate import _mappo_factory, _maxpressure_factory, env_settings_from_manifest
    from offline.materialise_draws import draw_config_path
    from offline.method_tier_grid import _fixedtime_factory, fixedtime_collection_settings

    out = _refuse_existing(args.out)
    corpus_root = assert_main_tree_path(args.corpus_root, what="the corpus root")
    draws_root = assert_main_tree_path(args.draws_root, what="the draws root")
    tiers = list(args.tiers) if args.tiers else _collected_tiers(corpus_root)
    if not tiers:
        raise FileNotFoundError(f"no hz4x4 tier is collected under {corpus_root}")
    if "random_fresh" in tiers:
        raise ValueError(
            "random_fresh's gate cell is A27 (d)'s random_fresh reference (default_rng([seed, draw_id])), which "
            "P9.1's M1 implements; merge main after M1 and wire it before gating that tier "
            "(docs/plans/p9.3.md section 4.6)"
        )
    cells = gate_cells_for(tiers)
    episodes = []
    for cell in cells:
        names = _tier_dir_names(cell.tier)
        name = names[0] if cell.tier != "mappo1000" else f"{CORPUS_PREFIX}__mappo1000__seed{cell.seed}"
        manifest = corpus_root / name / "manifest.json"
        settings = env_settings_from_manifest(manifest)
        config = draw_config_path(SCENARIO_KEY, int(cell.draw_id), out_root=draws_root)
        horizon = int(settings["max_steps"]) * int(settings["delta_time"])
        created = created_from_flow(config.parent / "flow.json", horizon_seconds=horizon)
        if cell.tier == "maxpressure":
            factory = _maxpressure_factory
        elif cell.tier == "fixedtime":
            factory = _fixedtime_factory(config, fixedtime_collection_settings(manifest))
        else:
            import json

            meta = json.loads(manifest.read_bytes())["run_metadata"]
            checkpoint = MAIN_TREE / str(meta["checkpoint"])
            if _sha256_file(checkpoint) != meta["checkpoint_sha256"]:
                raise ValueError(f"{checkpoint} is not the checkpoint {manifest} records")
            factory = _mappo_factory(str(checkpoint), args.device)
        print(f"hz4x4/{cell.arm} seed {cell.seed} draw {cell.draw_id}", flush=True)
        episodes.append(hz4x4_gate_episode(cell=cell, config_path=config, env_settings=settings,
                                           scenario_id=SCENARIO_ID, choose_action_factory=factory,
                                           engine_seed=BASE_SEED, created=created))
    criteria = engine.evaluate_scenario(episodes)
    outcome = gate_outcome(criteria)
    artifact = {
        "format_version": ARTIFACT_FORMAT_GATE,
        "registered_in": "PREREGISTRATION A11 (c), amended by A12 and A13; A27 (i)",
        "label": A27_LABEL,
        "scenario": SCENARIO,
        "tiers": tiers,
        "tolerances": {"c1": engine.C1_TOLERANCE, "c3c_reported_not_gating": engine.C3C_TOLERANCE,
                       "c4_min_tiers": engine.C4_MIN_TIERS, "c4_min_draws": engine.C4_MIN_DRAWS},
        "coverage_note": (
            "Criterion (4) cannot be met on hz4x4: its corpus has four behaviour tiers against the registered floor "
            "of seven."
            if len(tiers) == 4
            else f"Criterion (4) cannot be met on hz4x4: this gate covers {len(tiers)} behaviour tier(s) against the "
            "registered floor of seven."
        ),
        "coverage_ruling": "BRIEF_48 Amendment A, A.3: the floor is the registered seven corpus tiers and is not re-read "
        "per scenario; A27 (i)'s fail branch applies",
        "episodes": [episode.as_record() for episode in episodes],
        "criteria": criteria.as_record(),
        "outcome": outcome,
        "extremes_available_on_s": gate_extremes(episodes),
        "thread_regime": engine.thread_regime(),
        "git": _git(),
    }
    _write_artifact(artifact, out)
    print(f"wrote {out}: {'PASS' if outcome['passed'] else 'FAIL'} {outcome['failed_criteria']}", flush=True)
    return 0


def _run_declare(args: argparse.Namespace) -> int:
    """``declare``: hz4x4's tier specs, the exposure table, the second ATT route and the grid T-regress."""
    import dataclasses
    import os
    import tempfile

    from offline.att_ladder import tier_cells
    from offline.tier_sweep import TIERS as GRID_TIERS
    from offline.tier_sweep import TierSpec

    out = _refuse_existing(args.out)
    corpus_root = assert_main_tree_path(args.corpus_root, what="the corpus root")
    tiers = list(args.tiers) if args.tiers else _collected_tiers(corpus_root)
    if not tiers:
        raise FileNotFoundError(f"no hz4x4 tier is collected under {corpus_root}")
    records: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        names = _tier_dir_names(tier)
        dirs = [corpus_root / name for name in names]
        counts = tier_row_counts(dirs)
        records[tier] = {
            "tier": tier,
            "dirs": names,
            "subsample": "none",
            "episode_count": counts["episodes"],
            "ladder_att": ladder_att(dirs),
            "mean_training_return": mean_training_return(dirs),
            "row_counts": counts,
            "exposure": exposure_table(counts),
        }
    with tempfile.TemporaryDirectory(prefix="p9_3_ladder_") as scratch:
        for tier in tiers:
            for name in _tier_dir_names(tier):
                os.symlink(corpus_root / name, Path(scratch) / name)
        second = {cell.tier: cell for cell in tier_cells(Path(scratch)) if cell.scenario == CORPUS_PREFIX}
    for tier in tiers:
        if second[tier].att_horizon_mean != records[tier]["ladder_att"] or second[tier].n_episodes != records[tier]["episode_count"]:
            raise ValueError(f"{tier}: att_ladder.tier_cells gives {second[tier].att_horizon_mean} over "
                             f"{second[tier].n_episodes} episodes, this module {records[tier]['ladder_att']}")
    regress: dict[str, dict[str, Any]] = {}
    for tier, spec in GRID_TIERS.items():
        dirs = [corpus_root / d for d in spec.dirs]
        value, mean_return = ladder_att(dirs), mean_training_return(dirs)
        regress[tier] = {"ladder_att": value, "recorded_ladder_att": spec.ladder_att,
                         "ladder_att_equal": value == spec.ladder_att, "mean_training_return": mean_return,
                         "recorded_mean_training_return": spec.mean_training_return,
                         "mean_training_return_equal_at_one_decimal": round(mean_return, 1) == spec.mean_training_return}
    failed = [t for t, r in regress.items() if not (r["ladder_att_equal"] and r["mean_training_return_equal_at_one_decimal"])]
    if failed:
        raise ValueError(f"T-regress failed on the grid's {failed}: this module does not reproduce tier_sweep.TIERS")
    field_names = [field.name for field in dataclasses.fields(TierSpec)]
    artifact = {
        "format_version": ARTIFACT_FORMAT_DECLARATION,
        "registered_in": "PREREGISTRATION A27 (i) and (b); BRIEF_48 section 3.3",
        "label": A27_LABEL,
        "paper_sentence": PAPER_SENTENCE,
        "tier_specs": {tier: {name: records[tier][name] for name in field_names} for tier in tiers},
        "tier_order_by_measured_att": sorted(tiers, key=lambda t: records[t]["ladder_att"]),
        "tiers": records,
        "ladder_att_second_route": "offline.att_ladder.tier_cells on a root holding only the hz4x4 directories, == "
        "this module's value on every tier",
        "t_regress_grid": regress,
        "git": _git(),
    }
    _write_artifact(artifact, out)
    print(f"wrote {out}: {len(tiers)} tier(s), order by measured ATT {artifact['tier_order_by_measured_att']}", flush=True)
    return 0


def _run_d5_check(args: argparse.Namespace) -> int:
    """``d5-check``: A.7's definition check on hz1x1 and grid4x4, one process, STOP on an unreproduced D5 cell.

    For each scenario in :data:`D5_SCENARIOS`: the k grid D5's README records (asserted equal to the declared one),
    the protocol read from the scenario's fixed-time manifest (:func:`d5_protocol`), training draws 1-5 from the main
    tree, one episode per (k, draw) through :func:`sweep_episode` with the scenario's parameters, then
    :func:`summarise_sweep`, :func:`d5_t_regress`, :func:`summarise_admission` and :func:`definition_check`.  The
    first scenario whose T-regress fails STOPS the check: its unreproduced cells are printed, nothing is written,
    and the exit code is 3 (A.7: "nothing else from the check is reported").
    """
    import json

    from offline.materialise_draws import draw_config_path

    out = _refuse_existing(args.out)
    draws_root = assert_main_tree_path(args.draws_root, what="the draws root")
    corpus_root = Path(args.corpus_root)
    readme = _repo_root() / D5_README
    table = d5_recorded_table(readme)
    assert_training_draws(K_SWEEP_DRAWS)
    scenarios: dict[str, dict[str, Any]] = {}
    for spec in D5_SCENARIOS:
        recorded = table[spec.name]["att_horizon"]
        if tuple(sorted(recorded)) != spec.k_grid:
            raise ValueError(f"{spec.name}: D5's README sweeps k = {sorted(recorded)}, the declared grid is {spec.k_grid}")
        settings, schedule_source, plan_sha256 = d5_protocol(spec, corpus_root=corpus_root)
        configs = {d: draw_config_path(spec.scenario_key, d, out_root=draws_root) for d in K_SWEEP_DRAWS}
        missing = [str(path) for path in configs.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{spec.name}: training draws are not materialised: {missing}")
        records: list[dict[str, Any]] = []
        for k in spec.k_grid:
            for draw_id in K_SWEEP_DRAWS:
                record = sweep_episode(
                    k, draw_id, config_path=configs[draw_id], env_settings=settings, scenario=spec.name,
                    scenario_id=spec.scenario_id, k_grid=spec.k_grid, schedule_source=schedule_source,
                    plan_sha256=plan_sha256,
                )
                records.append(record)
                print(f"{spec.name} k={k:>2} draw {draw_id}: att_engine {record['att_engine']:.4f}  "
                      f"att_ours {record['att_ours']:.4f}  never_entered {record['never_entered']}", flush=True)
        summary = summarise_sweep(records, k_grid=spec.k_grid, draws=K_SWEEP_DRAWS)
        regress = d5_t_regress(summary, recorded)
        if not regress["all_reproduce"]:
            failing = {k: cell for k, cell in regress["cells"].items() if not cell["reproduces"]}
            print(f"STOP: {spec.name}'s per-k mean att_ours does not reproduce D5's printed att_horizon at "
                  f"{json.dumps({str(k): [c['computed_2dp'], c['recorded']] for k, c in failing.items()})} "
                  "(computed, recorded); the protocol is not D5's and nothing else from the check is reported",
                  flush=True)
            return 3
        check = definition_check(summary, k_grid=spec.k_grid)
        chosen = int(table[spec.name]["chosen_k"])
        scenarios[spec.name] = {
            "k_grid": list(spec.k_grid),
            "draws": list(K_SWEEP_DRAWS),
            "engine_seed": BASE_SEED,
            "scenario_key": spec.scenario_key,
            "scenario_id": spec.scenario_id,
            "schedule_source": schedule_source,
            "plan_sha256": plan_sha256,
            "env_settings_from": str(corpus_root / spec.protocol_dir / "manifest.json"),
            "env_settings": settings,
            "records": records,
            "summary": {str(k): summary[k] for k in spec.k_grid},
            "admission": {str(k): value for k, value in summarise_admission(
                records, k_grid=spec.k_grid, draws=K_SWEEP_DRAWS).items()},
            "t_regress": {**regress, "cells": {str(k): cell for k, cell in regress["cells"].items()}},
            "definition_check": check,
            "d5_chosen_k": chosen,
            "d5_chosen_k_is_att_ours_argmin": check["k_att_ours"] == chosen,
        }
        print(f"{spec.name}: T-regress reproduces all {len(spec.k_grid)} cells; {check['status']} "
              f"(att_engine argmin {check['k_att_engine']}, att_ours argmin {check['k_att_ours']}, D5's k {chosen})",
              flush=True)
    hz4x4_sweep = _repo_root() / "docs" / "data" / "p9_3_fixed_time_sweep.json"
    hz4x4 = json.loads(hz4x4_sweep.read_bytes())
    artifact = {
        "format_version": ARTIFACT_FORMAT_D5_CHECK,
        "label": D5_CHECK_LABEL,
        "registered_in": "BRIEF_48 Amendment A, A.7 (2026-10-16); exploration, not part of A27",
        "fence": "re-tunes nothing: whatever it shows, no registered k changes, and the hz1x1 and grid4x4 fixed-time "
        "tiers and every anchor stay as registered (A.7)",
        "protocol": "D5 (docs/data/fixed_time_sweep/README.md) under both ATT definitions: the README's k grids, training "
        "draws 1-5 (the main tree's materialised draws), engine seed 1000, one episode per (k, draw), each through "
        "offline.admission_probe.probe_episode via offline.hz4x4_ladder.sweep_episode with the scenario's parameters",
        "d5_readme": str(D5_README),
        "d5_readme_sha256": _sha256_file(readme),
        "scenarios": scenarios,
        "hz4x4_for_reference": {
            "artifact": "docs/data/p9_3_fixed_time_sweep.json",
            "sha256": _sha256_file(hz4x4_sweep),
            "status": hz4x4["decision"]["status"],
            "k_att_engine": hz4x4["decision"]["k_att_engine"],
            "k_att_ours": hz4x4["decision"]["k_att_ours"],
        },
        "git": _git(),
    }
    _write_artifact(artifact, out)
    print(f"wrote {out}", flush=True)
    return 0


def _run_record_ruling(args: argparse.Namespace) -> int:
    """``record-ruling``: the fixed-time ruling file, built by :func:`fixed_time_ruling_record`, written once."""
    out = _refuse_existing(args.out)
    brief = Path(args.brief) if args.brief else _repo_root() / "docs" / "briefs" / "BRIEF_48_p9.3_a27c_hz4x4.md"
    record = fixed_time_ruling_record(sweep_path=args.sweep, k=int(args.k), amendment=str(args.amendment),
                                      brief_path=brief)
    record["git"] = _git()
    _write_artifact(record, out)
    print(f"wrote {out}: k = {record['chosen_k']} under {record['amendment']} (sweep {record['sweep_sha256'][:12]})",
          flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if int(args.torch_threads) != 1:
        raise SystemExit("--torch-threads must be 1: BRIEF_48 pins one torch thread everywhere")
    from experiments.runner import limit_torch_threads

    limit_torch_threads(int(args.torch_threads))
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "draws": _run_draws,
        "preflight": _run_preflight,
        "k-sweep": _run_k_sweep,
        "collect": _run_collect,
        "teachers": _run_teachers,
        "gate": _run_gate,
        "declare": _run_declare,
        "d5-check": _run_d5_check,
        "record-ruling": _run_record_ruling,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"hz4x4_ladder {args.command}: REFUSED: {exc}", flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
