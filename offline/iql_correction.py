"""P5.2b: the IQL random-tier correction run -- ONE registered cell of P5.2 re-run under its registered condition.

Format versions: the artifact ``p5.2b-correction/1.0`` (``output/p5_2b/artifacts/p5_2b_correction.json``, committed by
hand as ``docs/data/p5_2b_iql_correction.json``); the training record ``p5.2b-training/1.0``; the pre-flight record
``p5.2b-preflight/1.0``.  Alignment convention: none of these records a trajectory, so none applies.  The transition
table IQL trains on is ``offline_baselines.build_transitions``' (C6: the transition of decision ``t`` pairs observation
row ``t`` with row ``t + 1``, the final one bootstrapping from row ``T``), restricted to the declared streams.

Written against ``docs/briefs/BRIEF_44_p5.2b_iql_correction.md`` and its Amendment A, on the plan
``docs/plans/p5.2b.md`` @ ``b40bc3f`` (approved at gate G0).

THE DEFECT (``DEFERRED`` 106)
---------------------------
P5.2's ``_run_train_baselines`` filtered BC's and %BC's batches to the declared streams (``tier_sweep.py:2038``) but
built IQL's transition table from the WHOLE dataset (``:2078``; ``:1820`` at ``9460800``, the commit the original
checkpoints record).  On the random tier -- 400 episodes available, 200 selected, one per draw -- IQL trained on
400 x 16 x 360 = 2,304,000 transitions while every other arm trained on the selected 1,152,000.  The provenance's
``training_streams`` (3,200) and the reward scale both describe the selected set, which is why the record hid it.

WHAT THIS MODULE DOES, AND THROUGH WHOSE CODE
---------------------------------------------
1. **Train** five IQL seeds on the declared streams: the tier loaded by ``tier_sweep.tier_parts`` (P5.2's own loader),
   the table built by ``tier_sweep.iql_transition_table`` (the fix, ``BRIEF_44`` §4.1, which refuses unless the table
   holds exactly the declared streams' transitions), each seed trained by ``offline_baselines.train_iql`` with the
   arguments ``_run_train_baselines`` passes (``:2080-2087``).  The setup repeats fifteen lines of
   ``_run_train_baselines`` (``:2029-2058``) because calling it would also train BC and both %BC arms; the plan's
   T-train-args test proves the two hand ``train_iql`` the same table and the same arguments.
2. **Evaluate (i)** by P5.2's own path: ``tier_sweep.main`` with the ``evaluate`` subcommand, in-process, with the
   arguments ``output/p5_2/eval_random_iql.json`` records (draws, seeds, engine seed, ``deterministic``, declared steps),
   writing ``eval_random_iql.json`` in P5.2's format.
3. **Evaluate (ii)** by P8.4b's own cell runner, ``att_rederivation.run_campaign``, unmodified, on the same 500
   episodes: ``att_ours``, ``att_engine``, ``entered``, ``created``, ``never_entered`` per episode in P8.4b's format.  Its
   ``committed`` mapping is (i)'s output, so a cell's ``reproduces_committed: true`` means the two evaluation paths
   agree on that episode -- what it meant for the originals in P8.4b.
4. **Recompute** P5.2's random-tier statements that involve IQL -- Q1's ``iql@random`` entry and its aggregate (P5.2's
   ``score_level``), the ranking, Q2a, Q2b with IQL's five pairs (P5.2's ``concordance`` and ``predicted_order``), Q3a
   and Q3c -- under ``att_ours`` and ``att_engine``, before and after, with Q2b's hard subset under ``att_ours`` as an
   asserted invariance (Amendment A, A2: the IQL-free statements under ``att_engine`` are ``DEFERRED`` 109).  Q2a, Q3a,
   Q3c and the ranking have no scorer in ``tier_sweep.py`` (P5.2's coordinator scored them by hand at ``119cc48``), so
   they are composed here from P5.2's registered definitions (``docs/plans/p5.2.md`` §4) and its primitives
   (``dt_gate._per_draw_means``, ``dt_gate.mean_ci95``, sorted levels); T-reproduce (c) proves each composed value equal
   to the committed one.

WHERE THE CORRECTED CHECKPOINTS LIVE (Amendment A, A1.1)
-------------------------------------------------------
``output/p5_2b/p5_2/checkpoints/grid4x4_random_iql_seed<s>.pt``: P8.4b's runner resolves a grid4x4 random-tier
checkpoint only as ``<output_root>/p5_2/checkpoints/<name>`` (``admission_probe._method_checkpoint``), so handing it
``output_root=output/p5_2b`` makes it read this one physical copy -- the same files (i) reads through
``--checkpoint-dir`` and P8.2's builder will name.  No link, and no change to an existing module.

ONE ARITHMETIC ROUTE
--------------------
An arm's level is the mean of its 500 episodes by ``dt_gate.mean_ci95`` (P5.2's ``_cell`` route), the episodes taken
in ``(seed, draw)`` ascending order -- the order P5.2's writer produced them in and the order of P8.4b's cell file
names -- because numpy's sum depends on the order of its elements.  For ``att_ours`` this equals each committed eval
file's ``cell.att_horizon_mean``, which the loaders assert.  The coordinator's README note uses ``statistics.mean`` and
differs in the last bit for ``bc_top10@random``; no comparison here crosses the two routes under ``==``.

THE BARRIER
-----------
``output/p5_2/`` and ``output/p8_4b_rederivation/`` are the record of the defect and of the other arms; nothing here
writes, moves or deletes anything under them.  Every write goes through :func:`assert_target`: an allow-list (under the
out-root, which must resolve to ``output/p5_2b`` or under ``output/p5_2b_runs/``; or exactly
``output/SHA256SUMS_p5_2b.txt`` for the run) and then ``tier_sweep.assert_writable`` against every other child of the
real ``output/`` and the corpus.  Refusals happen before anything is created.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "ARM",
    "ARTIFACT_FORMAT_VERSION",
    "CHECKPOINT_SUBDIR",
    "DEFINITIONS",
    "METHOD",
    "NON_DT_METHODS",
    "PINS",
    "PREFLIGHT_FORMAT_VERSION",
    "Pins",
    "Protocol",
    "Roots",
    "SCENARIO",
    "TIER",
    "TRAINING_FORMAT_VERSION",
    "TrainingInputs",
    "assert_evaluated_models",
    "assert_out_root",
    "assert_p8_4b_campaign",
    "assert_protocol",
    "assert_target",
    "assert_training_inputs",
    "build_parser",
    "build_report",
    "cell_summary",
    "check",
    "checkpoint_dir",
    "committed_from_eval",
    "correction_block",
    "declared_selection",
    "episodes_from_rows",
    "evaluate_p5_2_stage",
    "evaluate_p8_4b_stage",
    "load_report_inputs",
    "main",
    "manifest_path",
    "narrowed_held_out_draws",
    "original_protocol",
    "original_training",
    "p5_2_evaluation_argv",
    "p5_2_sums",
    "preflight",
    "preflight_estimate",
    "protected_roots",
    "admission_summary",
    "levels_of",
    "p5_2_cells",
    "p8_4b_cells",
    "random_tier_statements",
    "read_p5_2_json",
    "rederivation_cells",
    "report_from_inputs",
    "report_stage",
    "resolved_checkpoints",
    "run_p5_2_evaluation",
    "run_rederivation",
    "run_root",
    "runs_root",
    "status",
    "timeouts_from_preflight",
    "train_seeds",
    "train_stage",
    "training_inputs",
    "write_once_json",
    "write_report",
    "write_run_manifest",
]

ARTIFACT_FORMAT_VERSION = "p5.2b-correction/1.0"
TRAINING_FORMAT_VERSION = "p5.2b-training/1.0"
PREFLIGHT_FORMAT_VERSION = "p5.2b-preflight/1.0"

#: P8.4b's scenario name for grid4x4 (its cell keys and file names carry it).
SCENARIO = "grid4x4"
TIER = "random"
METHOD = "iql"
#: The arm string P5.2's evaluate subcommand and P8.4b's cells both write.
ARM = f"{METHOD}@{TIER}"

#: The run's directory, the pre-flights' directory and the run's manifest, under (or directly in) the real ``output/``.
RUN_DIR = "p5_2b"
RUNS_DIR = "p5_2b_runs"
MANIFEST_NAME = "SHA256SUMS_p5_2b.txt"
#: Amendment A, A1.1: the layout P8.4b's resolver reads, under the out-root.
CHECKPOINT_SUBDIR = Path("p5_2") / "checkpoints"
REDERIVATION_SUBDIR = "rederivation"
ARTIFACTS_SUBDIR = "artifacts"
ARTIFACT_NAME = "p5_2b_correction.json"
TRAINING_RECORD_NAME = "training_random_iql.json"
#: P5.2's evaluate subcommand names the cell itself (``eval_<tier>_<method>.json`` without ``--seeds``).
EVAL_NAME = "eval_random_iql.json"
CANARY_NAMES: tuple[str, str] = ("canary_open.json", "canary_close.json")
LOGS_SUBDIR = "logs"

#: P5.2's record, read and never written: the manifest, the original cell, the random tier's declaration.
P5_2_SUMS_NAME = "SHA256SUMS_p5_2.txt"
P5_2_EVAL_RELPATH = "p5_2/eval_random_iql.json"
P8_4B_DIR = "p8_4b_rederivation"
DECLARATION_RELPATH = "docs/data/p5_2_declaration_random.json"
C1_NOTE_RELPATH = "docs/notes/readme_2026-10-05/c1_rule_r.json"
RANDOM_TIER_DIR = "cf_grid4x4__random"

#: Q3c's set (``docs/plans/p5.2.md`` §4 Q3c): the best non-DT arm is the lowest of these, measured on the tier.
NON_DT_METHODS: tuple[str, ...] = ("bc", "bc_top10", "bc_top10_perix", "iql")
#: A11(b)'s two ATT definitions, in the order every block reports them.
DEFINITIONS: tuple[str, ...] = ("att_ours", "att_engine")

#: A stage's timeout is this multiple of its pre-flight estimate (plan §13), and never below the floor.
TIMEOUT_FACTOR = 3.0
TIMEOUT_FLOOR_SECONDS = 600.0
#: The pre-flight's short training and its five draws per path (plan §8).
PREFLIGHT_STEPS = 2000
PREFLIGHT_DRAWS: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)
PREFLIGHT_SEED = 101


@dataclass(frozen=True)
class Pins:
    """The digests the inputs are read at.  Every one was computed on 2026-10-06 and is quoted in the plan (V1-V10)."""

    p5_2_sums_sha256: str
    declaration_sha256: str
    c1_note_sha256: str
    p8_4b_declared_cells_sha256: str


#: The real pins.  Tests pass their own for synthetic trees; the run and the pre-flight use these.
PINS = Pins(
    p5_2_sums_sha256="fde8309b4958231f8a7e3a33cb26674f645e1c295ad688900ac8e839a19ac19b",
    declaration_sha256="c8b8a35a2dd034434aef4b0c32de73627de7f35a8f7eb55b0bbf752631eadfe4",
    c1_note_sha256="643b73bca4de953c43b6adf1ac8d0485383de7b0afc1ae1ab738778327537e19",
    p8_4b_declared_cells_sha256="1f29b469dfb523ee3edf67acad341770ab18d1842427bcc6583e84f7b2cb8f4e",
)


@dataclass(frozen=True)
class Roots:
    """Where everything lives: this checkout (``docs/``), the MAIN tree's ``output/``, the v1.1 corpus, the materialised
    draws, and the out-root this invocation writes under (``output/p5_2b`` for the run)."""

    repo_root: Path
    output_root: Path
    corpus_root: Path
    draws_root: Path
    out_root: Path


@dataclass(frozen=True)
class Protocol:
    """P5.2's evaluation protocol for the cell, as ``output/p5_2/eval_random_iql.json`` records it."""

    seeds: tuple[int, ...]
    draw_ids: tuple[int, ...]
    engine_seed: int
    deterministic: bool
    declared_gradient_steps: int


@dataclass(frozen=True)
class TrainingInputs:
    """What ``_run_train_baselines`` hands ``train_iql``, built by P5.2's own loader and the fix."""

    parts: Mapping[str, Any]
    group: tuple[int, int]
    streams: tuple[Any, ...]
    scale: float
    table: Any
    provenance: dict[str, Any]


_SKELETON = "P5.2b C3 skeleton: implemented in C5-C6"


# ----------------------------------------------------------------------
# The barrier
# ----------------------------------------------------------------------


def run_root(output_root: str | Path) -> Path:
    """``<output>/p5_2b``, the run's directory (``BRIEF_44`` §3)."""
    raise NotImplementedError(_SKELETON)


def runs_root(output_root: str | Path) -> Path:
    """``<output>/p5_2b_runs``, under which every pre-flight writes (F9)."""
    raise NotImplementedError(_SKELETON)


def manifest_path(output_root: str | Path) -> Path:
    """``<output>/SHA256SUMS_p5_2b.txt``, the run's manifest."""
    raise NotImplementedError(_SKELETON)


def checkpoint_dir(out_root: str | Path) -> Path:
    """``<out-root>/p5_2/checkpoints``: the one directory (i), (ii) and P8.2 read the corrected checkpoints from."""
    raise NotImplementedError(_SKELETON)


def assert_out_root(out_root: str | Path, output_root: str | Path) -> Path:
    """The resolved out-root, or a refusal unless it resolves to ``<output>/p5_2b`` or strictly under
    ``<output>/p5_2b_runs`` -- compared against the un-resolved names, so a symlink at either cannot pass."""
    raise NotImplementedError(_SKELETON)


def protected_roots(roots: Roots) -> tuple[Path, ...]:
    """Every immediate child directory of the real ``output/`` except the one the out-root lives in, plus the corpus
    and the draws, resolved."""
    raise NotImplementedError(_SKELETON)


def assert_target(path: str | Path, roots: Roots, protected: Sequence[Path]) -> Path:
    """The resolved *path*, or a refusal unless it lies under the out-root (or is the run's manifest, for the run) and
    outside every protected root."""
    raise NotImplementedError(_SKELETON)


def write_once_json(path: str | Path, payload: Mapping[str, Any], roots: Roots, protected: Sequence[Path]) -> Path:
    """Write *payload* atomically behind the barrier, refusing an existing file (which it leaves untouched)."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# P5.2's record and P8.4b's campaign, read at their anchors
# ----------------------------------------------------------------------


def p5_2_sums(output_root: str | Path, *, pins: Pins = PINS) -> dict[str, str]:
    """``output/SHA256SUMS_p5_2.txt`` parsed, refused unless the file is at its pinned digest."""
    raise NotImplementedError(_SKELETON)


def read_p5_2_json(output_root: str | Path, relative: str, sums: Mapping[str, str]) -> dict[str, Any]:
    """A P5.2 JSON record (``relative`` to ``output/``), refused unless its sha256 is its manifest line's."""
    raise NotImplementedError(_SKELETON)


def original_protocol(payload: Mapping[str, Any]) -> Protocol:
    """The protocol a P5.2 eval record states: its seeds and draws (a full grid, each key once), engine seed,
    ``deterministic`` flag and declared steps; refused for any arm but ``iql@random``."""
    raise NotImplementedError(_SKELETON)


def assert_protocol(protocol: Protocol) -> None:
    """Refuse unless P5.2's evaluate subcommand would run this protocol: ``dt_gate.HELD_OUT_DRAWS`` and
    ``TRAINING_SEEDS`` equal the record's draws and seeds, the declared steps are 40,000, and the regime is the
    default one the record states (Amendment A, Q12)."""
    raise NotImplementedError(_SKELETON)


def original_training(output_root: str | Path, sums: Mapping[str, str]) -> dict[str, Any]:
    """What the five original checkpoints record, each read at its manifest digest; refused unless the five agree on
    every value but their seed's own."""
    raise NotImplementedError(_SKELETON)


def declared_selection(repo_root: str | Path, corpus_root: str | Path, *, pins: Pins = PINS) -> dict[str, Any]:
    """The SECOND route to the declared data: the declaration's ``selected_episodes`` x ``node_order``, each episode's
    length read from the corpus manifest, all with ``json`` -- never through the loader."""
    raise NotImplementedError(_SKELETON)


def assert_p8_4b_campaign(output_root: str | Path, *, pins: Pins = PINS) -> dict[str, Any]:
    """P8.4b's anchors (Amendment A, A1.4): the campaign manifest's digest recomputed by P8.4b's own
    ``campaign_manifest`` from its cell list, equal to the pin and to ``CAMPAIGN_COMPLETE``'s; the campaign complete."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def training_inputs(corpus_root: str | Path, *, tier: str = TIER) -> TrainingInputs:
    """``_run_train_baselines``' IQL inputs (``tier_sweep.py:2029-2058``), the table through ``iql_transition_table``."""
    raise NotImplementedError(_SKELETON)


def assert_training_inputs(inputs: TrainingInputs, declared: Mapping[str, Any],
                           original: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless the inputs are the declared data and P5.2's: the table's rows and streams equal the second route,
    the reward scale and the normalisation statistics equal what the original checkpoints record."""
    raise NotImplementedError(_SKELETON)


def correction_block(declared: Mapping[str, Any], original: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance block the corrected checkpoints carry beside P5.2's own keys."""
    raise NotImplementedError(_SKELETON)


def train_seeds(inputs: TrainingInputs, *, seeds: Sequence[int], gradient_steps: int, device: Any,
                checkpoint_dir: str | Path, protected: Sequence[Path], correction: Mapping[str, Any],
                log_every: int = 0) -> list[dict[str, Any]]:
    """Train each seed with ``train_iql`` as ``_run_train_baselines`` calls it, to a ``.partial`` moved into place."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# Evaluation (i): P5.2's own path
# ----------------------------------------------------------------------


def p5_2_evaluation_argv(roots: Roots, *, seeds: Sequence[int] | None = None,
                         checkpoint_dir: str | Path | None = None, work_dir: str | Path | None = None,
                         pins: Pins = PINS) -> list[str]:
    """``tier_sweep``'s argv for the cell, every protocol argument read from P5.2's record (plan §5.1's table)."""
    raise NotImplementedError(_SKELETON)


@contextlib.contextmanager
def narrowed_held_out_draws(draws: Sequence[int]) -> Iterator[None]:
    """Substitute ``dt_gate.HELD_OUT_DRAWS`` with a subset for the duration of one call -- the pre-flight's and
    T-reproduce (a)'s five draws.  The run never uses it."""
    raise NotImplementedError(_SKELETON)
    yield  # pragma: no cover


def run_p5_2_evaluation(argv: Sequence[str], *, draws: Sequence[int] | None = None) -> int:
    """``tier_sweep.main(argv)`` in-process; with *draws*, inside :func:`narrowed_held_out_draws`."""
    raise NotImplementedError(_SKELETON)


def assert_evaluated_models(payload: Mapping[str, Any], training_record: Mapping[str, Any]) -> None:
    """Refuse unless (i)'s ``model_provenance`` names, per seed, the record's checkpoint at its weight digest."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# Evaluation (ii): P8.4b's own cell runner
# ----------------------------------------------------------------------


def rederivation_cells(seeds: Sequence[int], draws: Sequence[int]) -> list[Any]:
    """The ``CellKey``s of the cell, in P8.4b's order."""
    raise NotImplementedError(_SKELETON)


def committed_from_eval(payload: Mapping[str, Any]) -> dict[tuple[str, str, int | None, int], float]:
    """(i)'s ``att_horizon`` per episode, keyed as P8.4b's ``committed_att_index`` keys a cell."""
    raise NotImplementedError(_SKELETON)


def resolved_checkpoints(roots: Roots, *, resolver_root: str | Path, seeds: Sequence[int]) -> dict[int, Path]:
    """Where P8.4b's resolver finds each seed's checkpoint under *resolver_root*."""
    raise NotImplementedError(_SKELETON)


def run_rederivation(cells: Sequence[Any], *, roots: Roots, resolver_root: str | Path, work_dir: str | Path,
                     committed: Mapping[tuple[str, str, int | None, int], float], protected: Sequence[Path],
                     engine_seed: int) -> dict[str, Any]:
    """P8.4b's runner on *cells*: the campaign manifest written once, ``run_campaign``, the completion marker."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# The recomputation
# ----------------------------------------------------------------------


def episodes_from_rows(rows: Sequence[Mapping[str, Any]], *, value_key: str) -> list[Any]:
    """``dt_gate.EpisodeResult``s in ``(seed, draw)`` order whose ``att_horizon`` carries ``rows[i][value_key]``."""
    raise NotImplementedError(_SKELETON)


def cell_summary(episodes: Sequence[Any]) -> dict[str, Any]:
    """Mean, sd, CI95 half-width and n by ``dt_gate.mean_ci95``, and the per-seed means."""
    raise NotImplementedError(_SKELETON)


def p5_2_cells(output_root: str | Path, sums: Mapping[str, str]) -> dict[tuple[str, str], list[Any]]:
    """P5.2's committed episodes of the 19 Q1 cells (``att_horizon``, i.e. ``att_ours``), each file read at its manifest
    digest and its mean asserted equal to the file's own ``cell.att_horizon_mean``."""
    raise NotImplementedError(_SKELETON)


def p8_4b_cells(output_root: str | Path, reference: Mapping[tuple[str, str], Sequence[Any]], *, definition: str,
                pins: Pins = PINS) -> dict[tuple[str, str], list[Any]]:
    """P8.4b's episodes of the same cells under *definition*, read by P8.4b's own file names after its anchors: every
    cell ``reproduces_committed`` and carries an ``att_ours`` equal to *reference*'s ``att_horizon`` for its key."""
    raise NotImplementedError(_SKELETON)


def levels_of(cells: Mapping[tuple[str, str], Sequence[Any]]) -> dict[tuple[str, str], float]:
    """Each cell's level: the mean of its episodes by ``dt_gate.mean_ci95`` (one arithmetic route)."""
    raise NotImplementedError(_SKELETON)


def admission_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A11(b)'s companions over a cell's P8.4b-format rows: the means of ``entered``, ``created``, ``never_entered``
    and ``entered_fraction``, and how many episodes left a vehicle never entered."""
    raise NotImplementedError(_SKELETON)


def random_tier_statements(cells: Mapping[tuple[str, str], Sequence[Any]], *,
                           with_hard_subset: bool) -> dict[str, Any]:
    """Every recomputed statement of plan §6 at the random tier, under the one definition *cells* carry."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def load_report_inputs(roots: Roots, *, pins: Pins = PINS) -> dict[str, Any]:
    """Every input of the artifact, each verified at its anchor before it is read."""
    raise NotImplementedError(_SKELETON)


def report_from_inputs(inputs: Mapping[str, Any], *, git: Mapping[str, Any]) -> dict[str, Any]:
    """The ``p5.2b-correction/1.0`` artifact from verified inputs; pure."""
    raise NotImplementedError(_SKELETON)


def build_report(roots: Roots, *, git: Mapping[str, Any], pins: Pins = PINS) -> dict[str, Any]:
    """:func:`load_report_inputs` then :func:`report_from_inputs`."""
    raise NotImplementedError(_SKELETON)


def write_report(roots: Roots, artifact: Mapping[str, Any], protected: Sequence[Path]) -> Path:
    """Write the artifact once, canonically, under ``<out-root>/artifacts/``."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# The run's stages
# ----------------------------------------------------------------------


def status(roots: Roots, *, pins: Pins = PINS) -> dict[str, str]:
    """Each stage's state from disk: ``absent``, ``complete`` or ``partial`` (a refusal for the driver)."""
    raise NotImplementedError(_SKELETON)


def check(roots: Roots, *, pins: Pins = PINS, require_cuda: bool = True) -> dict[str, Any]:
    """Every pre-token refusal the module owns (plan §8); writes nothing.  ``require_cuda`` is false only in tests."""
    raise NotImplementedError(_SKELETON)


def train_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """The five seeds and the training record, written once."""
    raise NotImplementedError(_SKELETON)


def evaluate_p5_2_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """(i): P5.2's evaluate subcommand on the corrected checkpoints."""
    raise NotImplementedError(_SKELETON)


def evaluate_p8_4b_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """(ii): P8.4b's runner on the corrected checkpoints, resumable per cell."""
    raise NotImplementedError(_SKELETON)


def write_run_manifest(roots: Roots, protected: Sequence[Path]) -> list[str]:
    """``<output>/SHA256SUMS_p5_2b.txt`` over every run file but ``artifacts/``, as ``<sha256>  p5_2b/<path>``, once."""
    raise NotImplementedError(_SKELETON)


def report_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """Build and write the artifact."""
    raise NotImplementedError(_SKELETON)


def preflight_estimate(*, table_seconds: float, rehearsal_steps: int, rehearsal_seconds: float,
                       episode_seconds_p5_2: float, episode_seconds_p8_4b: float, n_seeds: int,
                       declared_steps: int, n_episodes: int) -> dict[str, Any]:
    """The run's estimate from the pre-flight's measurements, and each stage's timeout: ``TIMEOUT_FACTOR`` x its
    estimate, never below ``TIMEOUT_FLOOR_SECONDS``.  An estimate, labelled one."""
    raise NotImplementedError(_SKELETON)


def timeouts_from_preflight(path: str | Path) -> dict[str, float]:
    """The stage timeouts of a pre-flight record, refused unless it is COMPLETE and both T-reproduce runs reproduced."""
    raise NotImplementedError(_SKELETON)


def preflight(roots: Roots, *, stamp: str, steps: int, draws: Sequence[int], pins: Pins = PINS) -> dict[str, Any]:
    """Gate G1's pre-flight (Amendment A, A3.1): a short training timed, T-reproduce (a) and (b) RUN on the original
    seed-101 checkpoint with their outcomes, the run's estimate and the stage timeouts."""
    raise NotImplementedError(_SKELETON)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> Any:
    """``check``, ``status``, ``train``, ``evaluate-p5-2``, ``evaluate-p8-4b``, ``manifest``, ``report``,
    ``preflight``, ``timeouts``."""
    raise NotImplementedError(_SKELETON)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns the process exit code."""
    raise NotImplementedError(_SKELETON)


if __name__ == "__main__":  # pragma: no cover - exercised by the driver
    raise SystemExit(main())
