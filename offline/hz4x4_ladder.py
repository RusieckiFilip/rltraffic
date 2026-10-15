"""P9.3 (A27 point 8): hangzhou_4x4_gudang -- draws, pre-flight, fixed-time sweep, corpus, A11 gate, declaration.

⚠️ SKELETON.  Every function below raises :class:`NotImplementedError`; the constants are the
registered or measured declarations of ``docs/plans/p9.3.md`` and are real.  Tests are written
against this surface first, so each one fails for its own reason rather than sharing one import error.

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


def assert_main_tree_path(path: str | Path, *, what: str) -> Path:
    """Return the resolved *path*, or raise ``ValueError`` when it lies inside a linked worktree."""
    raise NotImplementedError


def assert_single_insertion(entries: Sequence[Mapping[str, Any]], *, where: str) -> None:
    """Raise unless every flow entry is one vehicle (``startTime == endTime``)."""
    raise NotImplementedError


def source_flow_entries(
    source_config: str | Path = SOURCE_CONFIG, *, expected_vehicles: int = SOURCE_VEHICLES
) -> tuple[Path, bytes, list[dict[str, Any]]]:
    """The flow file *source_config* points at, its bytes and its entries, with the 2,983 asserted."""
    raise NotImplementedError


def materialise_hz4x4_draws(
    draw_ids: Sequence[int],
    *,
    source_config: str | Path = SOURCE_CONFIG,
    out_root: str | Path = MAIN_DRAWS_ROOT,
    expected_vehicles: int = SOURCE_VEHICLES,
    dry_run: bool = False,
) -> list[Any]:
    """Materialise *draw_ids* through ``offline.materialise_draws.materialise``, guarded and verified."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# Pre-flight (BRIEF_48 section 3.1)
# ----------------------------------------------------------------------


def assert_eight_actions(env: Any) -> list[int]:
    """``Utils.infer_action_counts`` on *env*, refused unless 16 intersections offer 8 actions each."""
    raise NotImplementedError


def go_no_go(projected_seconds: float, *, budget_seconds: float = ONE_DAY_SECONDS) -> str:
    """``"GO"`` iff the projected wall-clock fits the one-day rule, else ``"NO-GO"``."""
    raise NotImplementedError


def project_teachers(
    timings: Sequence[Mapping[str, Any]],
    *,
    train_episodes: int = TEACHER_EPISODES,
    evaluation_episodes: int,
    budget_seconds: float = ONE_DAY_SECONDS,
) -> dict[str, Any]:
    """The teachers' projected wall-clock from the concurrent per-seed pre-flight timings."""
    raise NotImplementedError


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
    """The ``python -m offline.hz4x4_ladder`` parser."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
