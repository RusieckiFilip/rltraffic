"""P9.3 (A27 point 8): hangzhou_4x4_gudang -- draws, pre-flight, fixed-time sweep, corpus, A11 gate, declaration.

⚠️ PARTIAL.  The draws, the pre-flight and the CLI are implemented; the sweep, the collector, the
teachers' report, the gate and the declaration still raise :class:`NotImplementedError` (their tests were
written first, against this surface).

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
    """Refuse any draw id outside the training pool 1-999 (D4)."""
    raise NotImplementedError


def fixedtime_factory_for_k(
    config_path: str | Path, k: int
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], Any]], dict[str, Any]]:
    """The collector's fixed-time policy at *k* on one draw's config, with its schedule provenance."""
    raise NotImplementedError


def sweep_episode(
    k: int, draw_id: int, *, config_path: str | Path, env_settings: Mapping[str, Any]
) -> dict[str, Any]:
    """One sweep episode through ``admission_probe.probe_episode``: A11(b)'s five quantities and the seconds."""
    raise NotImplementedError


def summarise_sweep(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Per k: the mean over its draws of ``att_engine`` and of ``att_ours``, with the draw ids."""
    raise NotImplementedError


def choose_k(summary: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The declared choice rule: both argmins agree and are interior or at the floor, else no k."""
    raise NotImplementedError


def chosen_k_from_artifact(path: str | Path) -> int:
    """The k a committed sweep artifact chose, refused unless its status is ``CHOSEN``."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# The corpus (BRIEF_48 sections 0 and 3.3)
# ----------------------------------------------------------------------


def grid_protocol(corpus_root: str | Path, tier: str) -> list[dict[str, Any]]:
    """The run metadata of every grid directory of *tier*, its invariants checked."""
    raise NotImplementedError


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
    """``offline.collect`` arguments carrying every setting of a protocol manifest, in ``recollect_v11``'s form."""
    raise NotImplementedError


def collector_supports_random_rng(parser: argparse.ArgumentParser | None = None) -> bool:
    """Whether ``offline.collect``'s parser accepts ``--random-rng`` (P9.1's M1)."""
    raise NotImplementedError


def derive_runs(
    *,
    corpus_root: str | Path,
    tiers: Sequence[str],
    chosen_k: int | None,
    env_config: str | Path,
    out_root: str | Path,
    random_rng_supported: bool,
) -> list[CollectionRun]:
    """Every hz4x4 collection run of *tiers*, derived from the grid's manifests with the declared substitutions."""
    raise NotImplementedError


def assert_targets_absent(runs: Sequence[CollectionRun]) -> None:
    """Refuse when any run's out-dir exists and is not empty -- for ALL runs, before any runs."""
    raise NotImplementedError


def check_random_rng_scheme(run_metadata: Mapping[str, Any]) -> None:
    """Refuse a ``random_fresh`` manifest whose generator scheme is not ``per-episode``."""
    raise NotImplementedError


def check_flow_digests(
    manifest: Mapping[str, Any],
    *,
    draws_root: str | Path = MAIN_DRAWS_ROOT,
    scenario_key: str = SCENARIO_KEY,
) -> int:
    """Each collected draw's ``flow_draw_sha256`` against the materialised draw's ``flow.json``; the count checked."""
    raise NotImplementedError


def distinct_action_sequences(corpus_dir: str | Path) -> dict[str, int]:
    """Per intersection id, the number of distinct action sequences over the directory's episodes."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# The teachers (BRIEF_48 section 3.4; RUNSPEC_01 section 4)
# ----------------------------------------------------------------------


def convergence(
    train_returns: Sequence[float],
    *,
    window: int = CONVERGENCE_WINDOW,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    """RUNSPEC_01 section 4's criterion on one training-return curve: the final window AND the preceding one."""
    raise NotImplementedError


def teachers_report(
    results_path: str | Path, *, env_id: str = "cf_hz4x4", agent_id: str = "mappo"
) -> dict[str, Any]:
    """Per seed: the curve, ``train_sec`` (an uncontrolled clock), the checkpoint and the criterion."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# A11's engine-semantics gate on hz4x4 (offline.engine_att_reference)
# ----------------------------------------------------------------------


def gate_cells_for(tiers: Sequence[str]) -> tuple[Any, ...]:
    """hz4x4's gate cells: every tier x ``GATE_DRAWS``, seed ``GATE_SEED`` unless the tier is deterministic."""
    raise NotImplementedError


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
    """``engine_att_reference.gate_episode`` for hz4x4, line for line, with hz4x4's env id check."""
    raise NotImplementedError


def gate_extremes(episodes: Sequence[Any]) -> dict[str, Any]:
    """The minimum and maximum ``entered / created`` among the gate's own episodes (Q1 of the plan)."""
    raise NotImplementedError


def gate_outcome(criteria: Any) -> dict[str, Any]:
    """A11's branch for hz4x4: PASS makes ``att_engine`` primary; FAIL reports both definitions."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# The declaration (BRIEF_48 section 3.3)
# ----------------------------------------------------------------------


def ladder_att(dirs: Sequence[str | Path]) -> float:
    """``att_ladder``'s ``att_horizon_mean`` over the episodes of *dirs*: the mean of ``att_per_step[-1]``."""
    raise NotImplementedError


def mean_training_return(dirs: Sequence[str | Path]) -> float:
    """P5.2's mean training-set return: the mean over every stream of ``numpy.sum`` of its local reward."""
    raise NotImplementedError


def tier_row_counts(dirs: Sequence[str | Path]) -> dict[str, int]:
    """Episodes, streams, decisions and the two top-decile row counts of one tier's training set."""
    raise NotImplementedError


def exposure_table(row_counts: Mapping[str, int], *, steps: int | None = None) -> dict[str, Any]:
    """A27(b)'s exposure (row visits = steps x decisions per step / decisions in the training set), per arm."""
    raise NotImplementedError


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

    teachers = sub.add_parser("teachers", parents=[common], help="the teachers' curves and RUNSPEC_01's criterion")
    teachers.add_argument("--out", required=True, help="docs/data/p9_3_teachers.json")

    gate = sub.add_parser("gate", parents=[common], help="A11's engine-semantics gate on hz4x4")
    gate.add_argument("--out", required=True, help="docs/data/p9_3_engine_att_gate.json")
    gate.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=None)
    gate.add_argument("--device", default="cpu")

    declare = sub.add_parser("declare", parents=[common], help="tier specs and the exposure table")
    declare.add_argument("--out", required=True, help="docs/data/p9_3_declaration.json")
    declare.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=None)
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
    print(f"done: {len(records)} draws {actions}; every flow.json verified single-insertion and equal to "
          "the in-memory plan", flush=True)
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


def _not_yet(args: argparse.Namespace) -> int:
    raise NotImplementedError(f"{args.command} is implemented in the next commit")


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
        "k-sweep": _not_yet,
        "collect": _not_yet,
        "teachers": _not_yet,
        "gate": _not_yet,
        "declare": _not_yet,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"hz4x4_ladder {args.command}: REFUSED: {exc}", flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
