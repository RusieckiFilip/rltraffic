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

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import re
import statistics
import sys
import time
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


_MODULE_ROOT = Path(__file__).resolve().parent.parent

#: The arguments ``_run_train_baselines`` passes ``train_iql`` that this module passes too (T-train-args compares them).
_CHECKPOINT_PATTERN = re.compile(r"^p5_2/checkpoints/grid4x4_random_iql_seed(\d+)\.pt$")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_bytes())


# ----------------------------------------------------------------------
# The barrier
# ----------------------------------------------------------------------


def run_root(output_root: str | Path) -> Path:
    """``<output>/p5_2b``, the run's directory (``BRIEF_44`` §3)."""
    return Path(output_root) / RUN_DIR


def runs_root(output_root: str | Path) -> Path:
    """``<output>/p5_2b_runs``, under which every pre-flight writes (F9)."""
    return Path(output_root) / RUNS_DIR


def manifest_path(output_root: str | Path) -> Path:
    """``<output>/SHA256SUMS_p5_2b.txt``, the run's manifest."""
    return Path(output_root) / MANIFEST_NAME


def checkpoint_dir(out_root: str | Path) -> Path:
    """``<out-root>/p5_2/checkpoints``: the one directory (i), (ii) and P8.2 read the corrected checkpoints from."""
    return Path(out_root) / CHECKPOINT_SUBDIR


def assert_out_root(out_root: str | Path, output_root: str | Path) -> Path:
    """The resolved out-root, or a refusal unless it resolves to ``<output>/p5_2b`` or strictly under
    ``<output>/p5_2b_runs`` -- compared against the un-resolved names, so a symlink at either cannot pass."""
    output = Path(output_root).resolve()
    resolved = Path(out_root).resolve()
    if resolved == output / RUN_DIR or (output / RUNS_DIR) in resolved.parents:
        return resolved
    raise PermissionError(
        f"{out_root} (resolving to {resolved}) is not an out-root of this task: the run writes under "
        f"{output / RUN_DIR} and a pre-flight strictly under {output / RUNS_DIR}. output/p5_2/ and "
        "output/p8_4b_rederivation/ are the record of the defect and of the other arms (BRIEF_44 §0)"
    )


def protected_roots(roots: Roots) -> tuple[Path, ...]:
    """Every immediate child directory of the real ``output/`` except the one the out-root lives in, plus the corpus
    and the draws, resolved."""
    from offline.tier_sweep import protected_roots_from

    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    top = out.relative_to(output.resolve()).parts[0]
    candidates: list[Path] = [Path(roots.corpus_root), Path(roots.draws_root)]
    if output.is_dir():
        for child in sorted(output.iterdir()):
            if child.is_dir() and child.name != top:
                candidates.append(child)
    return protected_roots_from(candidates)


def assert_target(path: str | Path, roots: Roots, protected: Sequence[Path]) -> Path:
    """The resolved *path*, or a refusal unless it lies under the out-root (or is the run's manifest, for the run) and
    outside every protected root."""
    from offline.tier_sweep import assert_writable

    output = Path(roots.output_root).resolve()
    out = assert_out_root(roots.out_root, roots.output_root)
    resolved = Path(path).resolve()
    is_manifest = resolved == output / MANIFEST_NAME and out == output / RUN_DIR
    if not (is_manifest or out in resolved.parents):
        raise PermissionError(
            f"{path} (resolving to {resolved}) is not a write target of this run: every file it writes lies under "
            f"{out}, and only the run itself writes {output / MANIFEST_NAME}"
        )
    return assert_writable(resolved, protected)


def write_once_json(path: str | Path, payload: Mapping[str, Any], roots: Roots, protected: Sequence[Path]) -> Path:
    """Write *payload* atomically behind the barrier, refusing an existing file (which it leaves untouched)."""
    from offline.dt_gate import write_json_atomic

    target = assert_target(path, roots, protected)
    if target.exists():
        raise FileExistsError(f"{target} already exists: every record of this run is written once")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(dict(payload), target)
    return Path(path)


# ----------------------------------------------------------------------
# P5.2's record and P8.4b's campaign, read at their anchors
# ----------------------------------------------------------------------


def p5_2_sums(output_root: str | Path, *, pins: Pins = PINS) -> dict[str, str]:
    """``output/SHA256SUMS_p5_2.txt`` parsed, refused unless the file is at its pinned digest."""
    from offline.tier_sweep import parse_sha256sums

    path = Path(output_root) / P5_2_SUMS_NAME
    digest = _sha256_file(path)
    if digest != pins.p5_2_sums_sha256:
        raise ValueError(
            f"{path} is at sha256 {digest}, not the pinned {pins.p5_2_sums_sha256}: P5.2's manifest is the anchor of "
            "every P5.2 file this task reads, so a changed manifest anchors nothing"
        )
    return parse_sha256sums(path.read_text(encoding="utf-8"))


def read_p5_2_json(output_root: str | Path, relative: str, sums: Mapping[str, str]) -> dict[str, Any]:
    """A P5.2 JSON record (``relative`` to ``output/``), refused unless its sha256 is its manifest line's."""
    expected = sums.get(relative)
    if expected is None:
        raise ValueError(f"{relative} has no line in {P5_2_SUMS_NAME}; an unlisted file cannot be shown to be P5.2's")
    path = Path(output_root) / relative
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(
            f"{path} is at sha256 {digest}, which does not match its line in {P5_2_SUMS_NAME} ({expected}); its bytes "
            "changed after P5.2 recorded them"
        )
    return json.loads(data)


def original_protocol(payload: Mapping[str, Any]) -> Protocol:
    """The protocol a P5.2 eval record states: its seeds and draws (a full grid, each key once), engine seed,
    ``deterministic`` flag and declared steps; refused for any arm but ``iql@random``."""
    if (payload.get("arm"), payload.get("method"), payload.get("tier")) != (ARM, METHOD, TIER):
        raise ValueError(f"the record is {payload.get('arm')!r}'s, not {ARM}'s: the protocol is read from {ARM}'s own record")
    episodes = list(payload["episodes"])
    if any(str(episode.get("arm")) != ARM for episode in episodes):
        raise ValueError(f"the record carries episodes of an arm other than {ARM}")
    keys = [(int(episode["seed"]), int(episode["draw_id"])) for episode in episodes]
    seeds = tuple(sorted({seed for seed, _ in keys}))
    draws = tuple(sorted({draw for _, draw in keys}))
    if len(keys) != len(set(keys)) or set(keys) != {(seed, draw) for seed in seeds for draw in draws}:
        raise ValueError(
            f"the record's {len(keys)} episodes are not a full grid of {len(seeds)} seeds x {len(draws)} draws with "
            "each key once, so it states no single protocol"
        )
    return Protocol(seeds=seeds, draw_ids=draws, engine_seed=int(payload["engine_seed"]),
                    deterministic=bool(payload["deterministic"]),
                    declared_gradient_steps=int(payload["declared_gradient_steps"]))


def assert_protocol(protocol: Protocol) -> None:
    """Refuse unless P5.2's evaluate subcommand would run this protocol: ``dt_gate.HELD_OUT_DRAWS`` and
    ``TRAINING_SEEDS`` equal the record's draws and seeds, the declared steps are 40,000, and the regime is the
    default one the record states (Amendment A, Q12)."""
    from offline import dt_gate
    from offline.tier_sweep import DECLARED_GRADIENT_STEPS

    held_out = tuple(int(draw) for draw in dt_gate.HELD_OUT_DRAWS)
    if held_out != protocol.draw_ids:
        raise ValueError(
            f"dt_gate.HELD_OUT_DRAWS ({len(held_out)} draws, {held_out[:3]}...) is not the record's draw set "
            f"({len(protocol.draw_ids)} draws): P5.2's evaluate subcommand rolls HELD_OUT_DRAWS, so it would not "
            "roll the recorded episodes"
        )
    seeds = tuple(int(seed) for seed in dt_gate.TRAINING_SEEDS)
    if seeds != protocol.seeds:
        raise ValueError(f"dt_gate.TRAINING_SEEDS {seeds} is not the record's seed set {protocol.seeds}")
    if protocol.declared_gradient_steps != DECLARED_GRADIENT_STEPS:
        raise ValueError(
            f"the record declares {protocol.declared_gradient_steps} gradient steps, not the registered "
            f"{DECLARED_GRADIENT_STEPS:,}"
        )
    if protocol.deterministic:
        raise ValueError(
            "the record says deterministic, and this run's regime is P5.2's default one (CUBLAS_WORKSPACE_CONFIG "
            "unset; Amendment A, Q12): the two would not be the same protocol"
        )


#: Per checkpoint, the provenance fields read (``name -> path``), every one equal across the five originals but the
#: seed's own.
_ORIGINAL_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tier", ("provenance", "tier")),
    ("training_streams", ("provenance", "training_streams")),
    ("gradient_steps", ("provenance", "gradient_steps")),
    ("declared_gradient_steps", ("provenance", "declared_gradient_steps")),
    ("batch_size", ("provenance", "batch_size")),
    ("learning_rate", ("provenance", "learning_rate")),
    ("weight_decay", ("provenance", "weight_decay")),
    ("grad_clip", ("provenance", "grad_clip")),
    ("device", ("provenance", "device")),
    ("torch_num_threads", ("provenance", "runtime", "torch_num_threads")),
    ("git_commit", ("provenance", "runtime", "git_commit")),
    ("torch_version", ("provenance", "runtime", "torch_version")),
    ("cuda_device_name", ("provenance", "runtime", "cuda_device_name")),
    ("training_rows", ("provenance", "diagnostics", "training_rows")),
    ("reward_scale", ("provenance", "diagnostics", "reward_scale")),
    ("tau", ("provenance", "diagnostics", "tau")),
    ("beta", ("provenance", "diagnostics", "beta")),
    ("gamma", ("provenance", "diagnostics", "gamma")),
    ("polyak", ("provenance", "diagnostics", "polyak")),
    ("weight_clip", ("provenance", "diagnostics", "weight_clip")),
)


def original_training(output_root: str | Path, sums: Mapping[str, str]) -> dict[str, Any]:
    """What the five original checkpoints record, each read at its manifest digest; refused unless the five agree on
    every value but their seed's own."""
    import torch

    output = Path(output_root)
    seeds = sorted(int(match.group(1)) for name in sums if (match := _CHECKPOINT_PATTERN.match(name)))
    if not seeds:
        raise ValueError(f"{P5_2_SUMS_NAME} lists no original IQL checkpoint of the random tier")
    common: dict[str, Any] | None = None
    stats: Any = None
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        relative = f"p5_2/checkpoints/grid4x4_random_iql_seed{seed}.pt"
        path = output / relative
        digest = _sha256_file(path)
        if digest != sums[relative]:
            raise ValueError(f"{path} is at sha256 {digest}, which does not match its line in {P5_2_SUMS_NAME}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values: dict[str, Any] = {}
        for name, route in _ORIGINAL_FIELDS:
            node: Any = payload
            for step in route:
                node = node[step]
            values[name] = node
        if common is None:
            common, stats = values, payload["stats"]
        elif values != common or payload["stats"] != stats:
            different = sorted(name for name in values if values[name] != common[name])
            raise ValueError(f"the original checkpoints disagree ({different or ['stats']}) at seed {seed}")
        per_seed[str(seed)] = {"checkpoint": f"output/{relative}", "sha256": digest,
                               "canonical_digest": payload["canonical_digest"], "training_rows": values["training_rows"]}
    return {"seeds": seeds, "common": dict(common or {}), "stats": stats, "per_seed": per_seed}


def declared_selection(repo_root: str | Path, corpus_root: str | Path, *, pins: Pins = PINS) -> dict[str, Any]:
    """The SECOND route to the declared data: the declaration's ``selected_episodes`` x ``node_order``, each episode's
    length read from the corpus manifest, all with ``json`` -- never through the loader."""
    path = Path(repo_root) / DECLARATION_RELPATH
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pins.declaration_sha256:
        raise ValueError(f"{path} is at sha256 {digest}, not the pinned {pins.declaration_sha256}")
    declaration = json.loads(data)
    manifest_file = Path(corpus_root) / RANDOM_TIER_DIR / "manifest.json"
    manifest = _read_json(manifest_file)
    length = {str(entry["filename"]): int(entry["episode_length"]) for entry in manifest["episodes"]}
    selected = list(declaration["selected_episodes"])
    nodes = [str(ix) for ix in declaration["node_order"]]
    if any(Path(str(entry["dataset_dir"])).name != RANDOM_TIER_DIR for entry in selected):
        raise ValueError(f"the declaration selects episodes outside {RANDOM_TIER_DIR}")
    keys = frozenset((RANDOM_TIER_DIR, str(entry["episode_file"]), ix) for entry in selected for ix in nodes)
    return {
        "episodes_available": len(length),
        "episodes_selected": len(selected),
        "nodes": len(nodes),
        "streams": len(keys),
        "rows": sum(length[str(entry["episode_file"])] for entry in selected) * len(nodes),
        "whole_tier_rows": sum(length.values()) * len(nodes),
        "declaration": {"file": DECLARATION_RELPATH, "sha256": digest},
        "corpus_manifest": {"file": f"datasets_v11/{RANDOM_TIER_DIR}/manifest.json",
                            "sha256": _sha256_file(manifest_file)},
        "keys": keys,
    }


def assert_p8_4b_campaign(output_root: str | Path, *, pins: Pins = PINS) -> dict[str, Any]:
    """P8.4b's anchors (Amendment A, A1.4): the campaign manifest's digest recomputed by P8.4b's own
    ``campaign_manifest`` from its cell list, equal to the pin and to ``CAMPAIGN_COMPLETE``'s; the campaign complete."""
    from offline.att_rederivation import COMPLETE_MARKER, MANIFEST_NAME as CAMPAIGN_MANIFEST
    from offline.att_rederivation import CellKey, campaign_manifest, campaign_status

    work = Path(output_root) / P8_4B_DIR
    recorded = _read_json(work / CAMPAIGN_MANIFEST)
    cells = []
    for key in recorded["cells"]:
        scenario, arm, seed, draw = str(key).split("|")
        cells.append(CellKey(scenario=scenario, arm=arm, seed=None if seed == "None" else int(seed),
                             draw_id=int(draw)))
    recomputed = campaign_manifest(cells, engine_seed=int(recorded["engine_seed"]))["declared_cells_sha256"]
    marker = _read_json(work / COMPLETE_MARKER)["declared_cells_sha256"]
    if not (recomputed == recorded["declared_cells_sha256"] == marker == pins.p8_4b_declared_cells_sha256):
        raise ValueError(
            f"P8.4b's campaign does not hold its anchors: declared_cells_sha256 recomputed {recomputed}, recorded "
            f"{recorded['declared_cells_sha256']}, CAMPAIGN_COMPLETE {marker}, pinned {pins.p8_4b_declared_cells_sha256}"
        )
    state = campaign_status(work)
    if not state["complete"]:
        raise ValueError(f"P8.4b's campaign is not complete on disk: {state['reason']}")
    return {"declared_cells_sha256": recomputed, "n_cells": len(cells), "engine_seed": int(recorded["engine_seed"]),
            "cells": [str(key) for key in recorded["cells"]]}


def _verified_note(repo_root: str | Path, pins: Pins) -> dict[str, Any]:
    path = Path(repo_root) / C1_NOTE_RELPATH
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pins.c1_note_sha256:
        raise ValueError(f"{path} is at sha256 {digest}, not the pinned {pins.c1_note_sha256}")
    return {"file": C1_NOTE_RELPATH, "sha256": digest, "payload": json.loads(data)}


def _verify_inputs(roots: Roots, pins: Pins) -> dict[str, Any]:
    """Every input the run reads, each at its anchor, before anything is written."""
    from offline.materialise_draws import draw_config_path

    output = Path(roots.output_root)
    sums = p5_2_sums(output, pins=pins)
    original_payload = read_p5_2_json(output, P5_2_EVAL_RELPATH, sums)
    protocol = original_protocol(original_payload)
    assert_protocol(protocol)
    original = original_training(output, sums)
    if tuple(original["seeds"]) != protocol.seeds:
        raise ValueError(f"the original checkpoints' seeds {original['seeds']} are not the record's {protocol.seeds}")
    declared = declared_selection(roots.repo_root, roots.corpus_root, pins=pins)
    if original["common"]["training_streams"] != declared["streams"]:
        raise ValueError(
            f"the original checkpoints record {original['common']['training_streams']} training streams and the "
            f"declaration {declared['streams']}"
        )
    campaign = assert_p8_4b_campaign(output, pins=pins)
    ours = sorted(f"{c.scenario}|{c.arm}|{c.seed}|{c.draw_id}"
                  for c in rederivation_cells(protocol.seeds, protocol.draw_ids))
    theirs = sorted(key for key in campaign["cells"] if key.startswith(f"{SCENARIO}|{ARM}|"))
    if ours != theirs or campaign["engine_seed"] != protocol.engine_seed:
        raise ValueError(
            f"the cell's {len(ours)} episodes and engine seed {protocol.engine_seed} are not P8.4b's {len(theirs)} "
            f"{ARM} cells at engine seed {campaign['engine_seed']}"
        )
    note = _verified_note(roots.repo_root, pins)
    missing = [draw for draw in protocol.draw_ids
               if not Path(draw_config_path("cityflow_grid4x4", draw, out_root=roots.draws_root)).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} held-out draws are not materialised under {roots.draws_root}: {missing[:5]}")
    return {"sums": sums, "p5_2_sums_sha256": pins.p5_2_sums_sha256, "original_payload": original_payload,
            "protocol": protocol, "original": original, "declared": declared, "campaign": campaign, "note": note}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def training_inputs(corpus_root: str | Path, *, tier: str = TIER) -> TrainingInputs:
    """``_run_train_baselines``' IQL inputs (``tier_sweep.py:2029-2058``), the table through ``iql_transition_table``."""
    import offline.tier_sweep as ts
    from offline.offline_baselines import iql_reward_scale

    parts = ts.tier_parts(tier, corpus_root)
    dataset = parts["dataset"]
    group = next(iter(dataset.groups))
    streams = tuple(parts["streams"])
    scale = iql_reward_scale([stream.total_return for stream in streams])
    table = ts.iql_transition_table(dataset, group=group, reward_scale=scale, streams=streams)
    provenance = {
        "tier": tier,
        "dataset_dirs": [str(d) for d in ts.tier_dirs(parts["spec"], corpus_root)],
        "scenario_id": ts.SCENARIO_ID,
        "training_streams": len(streams),
        "independent_per_intersection": True,
    }
    return TrainingInputs(parts=parts, group=group, streams=streams, scale=scale, table=table, provenance=provenance)


def assert_training_inputs(inputs: TrainingInputs, declared: Mapping[str, Any],
                           original: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless the inputs are the declared data and P5.2's: the table's rows and streams equal the second route,
    the reward scale and the normalisation statistics equal what the original checkpoints record."""
    rows = len(inputs.table)
    if rows != declared["rows"]:
        raise ValueError(f"the corrected table holds {rows} transitions; the declaration's second route gives {declared['rows']}")
    streams = frozenset((Path(str(s.dataset_dir)).name, str(s.episode_file), str(s.ix_id)) for s in inputs.streams)
    if streams != declared["keys"]:
        raise ValueError(
            f"the loader's {len(streams)} selected streams are not the declaration's {len(declared['keys'])} "
            f"({len(streams - declared['keys'])} extra, {len(declared['keys'] - streams)} missing)"
        )
    if inputs.scale != original["common"]["reward_scale"]:
        raise ValueError(f"the reward scale {inputs.scale!r} is not the {original['common']['reward_scale']!r} the original checkpoints record")
    if inputs.parts["dataset"].stats.to_json_obj() != original["stats"]:
        raise ValueError("the normalisation statistics are not the ones the original checkpoints record")
    return {"rows": rows, "streams": len(streams), "reward_scale": inputs.scale, "statistics_equal_the_originals": True}


def correction_block(declared: Mapping[str, Any], original: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance block the corrected checkpoints carry beside P5.2's own keys."""
    return {
        "task": "P5.2b",
        "brief": "BRIEF_44 and its Amendment A",
        "deferred": 106,
        "filter": "offline.tier_sweep.iql_transition_table",
        "declared": {"episodes": declared["episodes_selected"], "streams": declared["streams"],
                     "transitions": declared["rows"]},
        "original": {"episodes": declared["episodes_available"], "transitions": original["common"]["training_rows"],
                     "checkpoints": {seed: entry["sha256"] for seed, entry in original["per_seed"].items()}},
        "note": "the original checkpoints stay at output/p5_2/checkpoints/ as the record of the defect",
    }


def train_seeds(inputs: TrainingInputs, *, seeds: Sequence[int], gradient_steps: int, device: Any,
                checkpoint_dir: str | Path, protected: Sequence[Path], correction: Mapping[str, Any],
                log_every: int = 0) -> list[dict[str, Any]]:
    """Train each seed with ``train_iql`` as ``_run_train_baselines`` calls it, to a ``.partial`` moved into place."""
    import offline.offline_baselines as ob
    import offline.tier_sweep as ts
    from offline.dt_gate import runtime_provenance

    directory = Path(checkpoint_dir)
    planned: list[tuple[int, Path, Path]] = []
    for seed in seeds:
        final = directory / ts._checkpoint_name(TIER, METHOD, int(seed))
        partial = final.with_name(final.name + ".partial")
        for path in (final, partial):
            ts.assert_writable(path, protected)
            if path.exists():
                raise FileExistsError(f"{path} already exists: every checkpoint of this run is written once")
        planned.append((int(seed), final, partial))
    directory.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed, final, partial in planned:
        print(f"TRAIN {TIER}/{METHOD} seed {seed} -> {final}", flush=True)
        record = ob.train_iql(
            inputs.table, state_dim=inputs.group[0], n_actions=inputs.group[1], seed=seed,
            declared_gradient_steps=int(gradient_steps), batch_size=ob.IQL_BATCH_TRANSITIONS, device=device,
            checkpoint_path=partial, stats=inputs.parts["dataset"].stats, scenario_id=ts.SCENARIO_ID,
            provenance={**inputs.provenance, "correction": dict(correction), "runtime": runtime_provenance()},
            log_every=int(log_every),
        )
        ts.replace_guarded(partial, final, protected)
        file_sha256 = _sha256_file(final)
        if file_sha256 != record.file_sha256:
            raise ValueError(f"{final} changed between train_iql's write and its move into place")
        diagnostics = dict(record.diagnostics)
        runs.append({
            "tier": TIER, "method": METHOD, "seed": seed, "gradient_steps": int(record.gradient_steps),
            "declared_gradient_steps": int(record.declared_gradient_steps), "seconds": float(record.seconds),
            "final_loss": float(record.losses[-1]), "final_v_loss": diagnostics.get("final_v_loss"),
            "final_q_loss": diagnostics.get("final_q_loss"), "final_policy_loss": diagnostics.get("final_policy_loss"),
            "training_rows": diagnostics.get("training_rows"), "plateaued": bool(record.plateaued),
            "checkpoint_path": str(final), "canonical_digest": record.canonical_digest,
            "state_dict_sha256": ts.canonical_state_dict_digest(final), "file_sha256": file_sha256,
        })
        print(f"  done in {record.seconds:.1f}s", flush=True)
    return runs


# ----------------------------------------------------------------------
# Evaluation (i): P5.2's own path
# ----------------------------------------------------------------------


def p5_2_evaluation_argv(roots: Roots, *, seeds: Sequence[int] | None = None,
                         checkpoint_dir: str | Path | None = None, work_dir: str | Path | None = None,
                         pins: Pins = PINS) -> list[str]:
    """``tier_sweep``'s argv for the cell, every protocol argument read from P5.2's record (plan §5.1's table)."""
    sums = p5_2_sums(roots.output_root, pins=pins)
    protocol = original_protocol(read_p5_2_json(roots.output_root, P5_2_EVAL_RELPATH, sums))
    assert_protocol(protocol)
    checkpoints = Path(checkpoint_dir) if checkpoint_dir is not None else Path(roots.out_root) / CHECKPOINT_SUBDIR
    work = Path(work_dir) if work_dir is not None else Path(roots.out_root)
    argv = [
        "--corpus-root", str(roots.corpus_root), "--draws-root", str(roots.draws_root),
        "--work-dir", str(work), "--checkpoint-dir", str(checkpoints),
        "--reuse-root", str(Path(roots.output_root) / "p5_2"),
        "--engine-seed", str(protocol.engine_seed), "--gradient-steps", str(protocol.declared_gradient_steps),
        "--torch-threads", "1", "--tier", TIER,
    ]
    if seeds is not None:
        argv += ["--seeds", ",".join(str(int(seed)) for seed in seeds)]
    return argv + ["evaluate", "--method", METHOD]


@contextlib.contextmanager
def narrowed_held_out_draws(draws: Sequence[int]) -> Iterator[None]:
    """Substitute ``dt_gate.HELD_OUT_DRAWS`` with a subset for the duration of one call -- the pre-flight's and
    T-reproduce (a)'s five draws.  The run never uses it."""
    from offline import dt_gate

    saved = dt_gate.HELD_OUT_DRAWS
    narrowed = tuple(int(draw) for draw in draws)
    if not narrowed or len(set(narrowed)) != len(narrowed) or not set(narrowed) <= set(saved):
        raise ValueError(f"{narrowed} is not a subset of the held-out pool ({len(saved)} draws)")
    dt_gate.HELD_OUT_DRAWS = narrowed
    try:
        yield
    finally:
        dt_gate.HELD_OUT_DRAWS = saved


def run_p5_2_evaluation(argv: Sequence[str], *, draws: Sequence[int] | None = None) -> int:
    """``tier_sweep.main(argv)`` in-process; with *draws*, inside :func:`narrowed_held_out_draws`."""
    import offline.tier_sweep as ts

    if draws is None:
        return int(ts.main(list(argv)))
    with narrowed_held_out_draws(draws):
        return int(ts.main(list(argv)))


def assert_evaluated_models(payload: Mapping[str, Any], training_record: Mapping[str, Any]) -> None:
    """Refuse unless (i)'s ``model_provenance`` names, per seed, the record's checkpoint at its weight digest."""
    provenance = dict(payload.get("model_provenance") or {})
    runs = {str(run["seed"]): run for run in training_record["runs"]}
    if set(provenance) != set(runs):
        raise ValueError(f"(i)'s model_provenance names seeds {sorted(provenance)}, the training record {sorted(runs)}")
    for seed, run in sorted(runs.items()):
        entry = provenance[seed]
        if Path(str(entry.get("checkpoint_path", ""))).resolve() != Path(run["checkpoint_path"]).resolve():
            raise ValueError(f"(i)'s model_provenance for seed {seed} names {entry.get('checkpoint_path')}, not "
                             f"{run['checkpoint_path']}")
        if entry.get("state_dict_sha256") != run["state_dict_sha256"]:
            raise ValueError(f"(i)'s model_provenance for seed {seed} is weight digest {entry.get('state_dict_sha256')}, "
                             f"not the corrected checkpoint's {run['state_dict_sha256']}")


# ----------------------------------------------------------------------
# Evaluation (ii): P8.4b's own cell runner
# ----------------------------------------------------------------------


def rederivation_cells(seeds: Sequence[int], draws: Sequence[int]) -> list[Any]:
    """The ``CellKey``s of the cell, in P8.4b's order."""
    from offline.att_rederivation import CellKey

    return [CellKey(scenario=SCENARIO, arm=ARM, seed=int(seed), draw_id=int(draw))
            for seed in sorted(int(s) for s in seeds) for draw in sorted(int(d) for d in draws)]


def committed_from_eval(payload: Mapping[str, Any]) -> dict[tuple[str, str, int | None, int], float]:
    """(i)'s ``att_horizon`` per episode, keyed as P8.4b's ``committed_att_index`` keys a cell."""
    out: dict[tuple[str, str, int | None, int], float] = {}
    for episode in payload["episodes"]:
        if str(episode["arm"]) != ARM:
            raise ValueError(f"an episode of {episode['arm']!r} in {ARM}'s record")
        key = (SCENARIO, ARM, int(episode["seed"]), int(episode["draw_id"]))
        if key in out:
            raise ValueError(f"{key} appears twice in the record")
        out[key] = float(episode["att_horizon"])
    return out


def _probe_roots(roots: Roots, *, resolver_root: str | Path, work_dir: str | Path) -> Any:
    from offline.admission_probe import ProbeRoots

    return ProbeRoots(repo_root=Path(roots.repo_root), corpus_root=Path(roots.corpus_root),
                      draws_root=Path(roots.draws_root), output_root=Path(resolver_root), work_dir=Path(work_dir))


def resolved_checkpoints(roots: Roots, *, resolver_root: str | Path, seeds: Sequence[int]) -> dict[int, Path]:
    """Where P8.4b's resolver finds each seed's checkpoint under *resolver_root*."""
    from offline.att_rederivation import rederivation_checkpoint

    probe = _probe_roots(roots, resolver_root=resolver_root, work_dir=Path(resolver_root) / REDERIVATION_SUBDIR)
    return {int(seed): rederivation_checkpoint(SCENARIO, TIER, METHOD, int(seed), probe) for seed in seeds}


def run_rederivation(cells: Sequence[Any], *, roots: Roots, resolver_root: str | Path, work_dir: str | Path,
                     committed: Mapping[tuple[str, str, int | None, int], float], protected: Sequence[Path],
                     engine_seed: int) -> dict[str, Any]:
    """P8.4b's runner on *cells*: the campaign manifest written once, ``run_campaign``, the completion marker."""
    from offline.att_rederivation import COMPLETE_MARKER, MANIFEST_NAME as CAMPAIGN_MANIFEST
    from offline.att_rederivation import assert_manifest_agrees, campaign_manifest, campaign_status, run_campaign
    from offline.offline_baselines import pin_torch_threads
    from offline.tier_sweep import assert_writable, write_json_guarded

    work = Path(work_dir)
    manifest = campaign_manifest(list(cells), engine_seed=int(engine_seed))
    manifest_file = work / CAMPAIGN_MANIFEST
    assert_manifest_agrees(manifest_file, manifest)
    assert_writable(manifest_file, protected)
    assert_writable(work / COMPLETE_MARKER, protected)
    previous = pin_torch_threads(1)
    try:
        work.mkdir(parents=True, exist_ok=True)
        if not manifest_file.is_file():
            write_json_guarded(manifest, manifest_file, protected)
        outcome = run_campaign(list(cells), roots=_probe_roots(roots, resolver_root=resolver_root, work_dir=work),
                               engine_seed=int(engine_seed), device=None, committed=committed, protected=protected)
    finally:
        pin_torch_threads(previous)
    state = campaign_status(work)
    if state["complete"] and not state["marker_present"]:
        (work / COMPLETE_MARKER).write_text(json.dumps({"declared_cells_sha256": state["declared_cells_sha256"]})
                                            + "\n", encoding="utf-8")
        state = campaign_status(work)
    return {"outcome": outcome, "status": state}


# ----------------------------------------------------------------------
# The recomputation
# ----------------------------------------------------------------------


def episodes_from_rows(rows: Sequence[Mapping[str, Any]], *, value_key: str) -> list[Any]:
    """``dt_gate.EpisodeResult``s in ``(seed, draw)`` order whose ``att_horizon`` carries ``rows[i][value_key]``."""
    from offline.dt_gate import EpisodeResult

    episodes = sorted(
        (EpisodeResult(arm=str(row["arm"]), seed=int(row["seed"]), draw_id=int(row["draw_id"]),
                       att_horizon=float(row[value_key]),
                       horizon_vehicle_count=float(row.get("horizon_vehicle_count", 0.0)),
                       episode_reward=float(row.get("episode_reward", 0.0)))
         for row in rows),
        key=lambda episode: (episode.seed, episode.draw_id),
    )
    keys = [(episode.seed, episode.draw_id) for episode in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("an episode appears twice in one cell")
    return episodes


def levels_of(cells: Mapping[tuple[str, str], Sequence[Any]]) -> dict[tuple[str, str], float]:
    """Each cell's level: the mean of its episodes by ``dt_gate.mean_ci95`` (one arithmetic route)."""
    from offline.dt_gate import mean_ci95

    return {cell: float(mean_ci95([episode.att_horizon for episode in episodes]).mean)
            for cell, episodes in cells.items()}


def cell_summary(episodes: Sequence[Any]) -> dict[str, Any]:
    """Mean, sd, CI95 half-width and n by ``dt_gate.mean_ci95``, and the per-seed means."""
    from offline.dt_gate import mean_ci95

    stats = mean_ci95([episode.att_horizon for episode in episodes])
    by_seed: dict[int, list[float]] = {}
    for episode in episodes:
        by_seed.setdefault(int(episode.seed), []).append(float(episode.att_horizon))
    return {
        "n": int(stats.n), "mean": float(stats.mean), "std": float(stats.std), "ci95": float(stats.ci95),
        "ci95_low": float(stats.mean - stats.ci95), "ci95_high": float(stats.mean + stats.ci95),
        "per_seed_means": {str(seed): float(mean_ci95(values).mean) for seed, values in sorted(by_seed.items())},
    }


def p5_2_cells(output_root: str | Path, sums: Mapping[str, str]) -> dict[tuple[str, str], list[Any]]:
    """P5.2's committed episodes of the 19 Q1 cells (``att_horizon``, i.e. ``att_ours``), each file read at its manifest
    digest and its mean asserted equal to the file's own ``cell.att_horizon_mean``."""
    import offline.tier_sweep as ts

    out: dict[tuple[str, str], list[Any]] = {}
    for method, tier in ts.OUT_OF_SAMPLE_CELLS:
        payload = read_p5_2_json(output_root, f"p5_2/eval_{tier}_{method}.json", sums)
        if (payload["arm"], payload["method"], payload["tier"]) != (f"{method}@{tier}", method, tier):
            raise ValueError(f"p5_2/eval_{tier}_{method}.json is the record of {payload['arm']!r}")
        episodes = episodes_from_rows(payload["episodes"], value_key="att_horizon")
        level = levels_of({(method, tier): episodes})[(method, tier)]
        if level != float(payload["cell"]["att_horizon_mean"]):
            raise ValueError(f"{method}@{tier}: the level {level!r} by mean_ci95 in (seed, draw) order is not the "
                             f"record's own cell mean {payload['cell']['att_horizon_mean']!r}")
        out[(method, tier)] = episodes
    return out


def _p8_4b_rows(output_root: str | Path, method: str, tier: str, reference: Sequence[Any]) -> list[dict[str, Any]]:
    from offline.att_rederivation import CellKey, cell_file_name

    work = Path(output_root) / P8_4B_DIR
    arm = f"{method}@{tier}"
    rows = []
    for episode in reference:
        name = cell_file_name(CellKey(scenario=SCENARIO, arm=arm, seed=episode.seed, draw_id=episode.draw_id))
        row = _read_json(work / name)
        identity = (str(row["arm"]), int(row["seed"]), int(row["draw_id"]))
        if (row.get("reproduces_committed") is not True or identity != (arm, episode.seed, episode.draw_id)
                or float(row["att_ours"]) != episode.att_horizon):
            raise ValueError(f"P8.4b's {name} does not reproduce P5.2's committed att_horizon for its key")
        rows.append(row)
    return rows


def p8_4b_cells(output_root: str | Path, reference: Mapping[tuple[str, str], Sequence[Any]], *, definition: str,
                pins: Pins = PINS) -> dict[tuple[str, str], list[Any]]:
    """P8.4b's episodes of the same cells under *definition*, read by P8.4b's own file names after its anchors: every
    cell ``reproduces_committed`` and carries an ``att_ours`` equal to *reference*'s ``att_horizon`` for its key."""
    if definition not in DEFINITIONS:
        raise ValueError(f"{definition!r} is not one of {DEFINITIONS}")
    assert_p8_4b_campaign(output_root, pins=pins)
    return {cell: episodes_from_rows(_p8_4b_rows(output_root, cell[0], cell[1], episodes), value_key=definition)
            for cell, episodes in reference.items()}


def admission_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A11(b)'s companions over a cell's P8.4b-format rows: the means of ``entered``, ``created``, ``never_entered``
    and ``entered_fraction``, and how many episodes left a vehicle never entered."""
    from offline.dt_gate import mean_ci95

    ordered = sorted(rows, key=lambda row: (int(row["seed"]), int(row["draw_id"])))
    out: dict[str, Any] = {"n": len(ordered)}
    for name in ("entered", "created", "never_entered", "entered_fraction"):
        out[f"{name}_mean"] = float(mean_ci95([float(row[name]) for row in ordered]).mean)
    out["episodes_with_never_entered"] = sum(int(row["never_entered"]) > 0 for row in ordered)
    return out


def random_tier_statements(cells: Mapping[tuple[str, str], Sequence[Any]], *,
                           with_hard_subset: bool) -> dict[str, Any]:
    """Every recomputed statement of plan §6 at the random tier, under the one definition *cells* carry."""
    import offline.tier_sweep as ts
    from offline.dt_gate import _per_draw_means, mean_ci95

    levels = levels_of(cells)
    q1 = ts.score_level(levels)
    entry = next(cell for cell in q1["cells"] if cell["cell"] == [METHOD, TIER])
    measured = {method: levels[(method, TIER)] for method in ts.METHODS}
    order = sorted(ts.METHODS, key=lambda method: measured[method])
    predicted = list(ts.predicted_order(TIER))
    lowest = min(measured.values())
    firsts = [method for method in ts.METHODS if measured[method] == lowest]
    statements: dict[str, Any] = {
        "q1": {"iql_random": entry, "n_held": q1["n_held"], "n_cells": q1["n_cells"], "threshold": q1["threshold"],
               "band": q1["band"], "outcome": q1["outcome"],
               "rule": "tier_sweep.score_level: |measured - predicted| / predicted <= band per cell; k of N"},
        "ranking": {"order": order, "levels": measured,
                    "rule": "the six methods by level, lowest first (P5.2's measured ordering)"},
        "q2a": {"predicted_first": predicted[0], "measured_first": order[0],
                "tied_for_first": firsts if len(firsts) > 1 else [],
                "outcome": "HELD" if firsts == [predicted[0]] else "FAILED",
                "rule": "docs/plans/p5.2.md section 4 Q2a: HELD iff the predicted first-place arm is the measured "
                        "one; an exact tie for first is no unique first place"},
    }
    concordance = ts.concordance(predicted, order, measured_levels=measured)
    discordant = {tuple(pair) for pair in concordance["discordant"]}
    statements["q2b"] = {
        **concordance, "threshold": ts.CONCORDANCE_THRESHOLD,
        "outcome": "HELD" if concordance["n_concordant"] >= ts.CONCORDANCE_THRESHOLD else "FAILED",
        "iql_pairs": [{"pair": list(pair), "concordant": tuple(pair) not in discordant}
                      for pair in itertools.combinations(sorted(ts.METHODS), 2) if METHOD in pair],
        "rule": "tier_sweep.concordance and predicted_order, P5.2's own, with its declared tie rule",
    }
    if with_hard_subset:
        statements["q2b_hard"] = {**ts.concordance(predicted, order, subset=ts.HARD_SUBSET, measured_levels=measured),
                                  "rule": "the same, on tier_sweep.HARD_SUBSET; no threshold is registered"}
    others = [method for method in ts.METHODS if method != "dt_nomix"]
    statements["q3a"] = {
        "rank": 1 + sum(measured[method] < measured["dt_nomix"] for method in others), "of": len(ts.METHODS),
        "outcome": "HELD" if all(measured[method] > measured["dt_nomix"] for method in others) else "FAILED",
        "rule": "docs/plans/p5.2.md section 4 Q3a: dt_nomix has the lowest level of the six; rank counts the arms "
                "strictly lower",
    }
    best = min(NON_DT_METHODS, key=lambda method: measured[method])
    left = _per_draw_means(list(cells[("dt_nomix", TIER)]))
    right = _per_draw_means(list(cells[(best, TIER)]))
    if set(left) != set(right):
        raise ValueError(f"dt_nomix and {best} at {TIER} do not cover the same draws")
    shared = sorted(left)
    stats = mean_ci95([left[draw] - right[draw] for draw in shared])
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    statements["q3c"] = {
        "best_non_dt": best, "tied_best_non_dt": [m for m in NON_DT_METHODS if measured[m] == measured[best]][1:],
        "mean": float(stats.mean), "std": float(stats.std), "ci95_low": float(low), "ci95_high": float(high),
        "ci95_width": float(high - low), "n_draws": len(shared),
        "reading": "resolves against the DT" if low > 0 else ("resolves for the DT" if high < 0 else "NOT RESOLVED"),
        "rule": "docs/plans/p5.2.md section 4 Q3c and its common scoring: mean[ATT(dt_nomix) - ATT(best non-DT arm)] "
                "over per-draw means of the five seeds (dt_gate._per_draw_means), the 95 % CI by dt_gate.mean_ci95",
    }
    return statements


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def _run_files(out: Path) -> list[Path]:
    return sorted(path for path in out.rglob("*") if path.is_file()
                  and path.relative_to(out).parts[0] != ARTIFACTS_SUBDIR)


def _training_complete(out: Path, protocol: Protocol, declared_rows: int) -> tuple[str, dict[str, Any] | None]:
    record_path = out / TRAINING_RECORD_NAME
    directory = out / CHECKPOINT_SUBDIR
    files = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    if not record_path.is_file():
        return ("partial" if files else "absent"), None
    record = _read_json(record_path)
    runs = list(record.get("runs", []))
    expected = sorted(f"grid4x4_{TIER}_{METHOD}_seed{seed}.pt" for seed in protocol.seeds)
    if sorted(int(run["seed"]) for run in runs) != sorted(protocol.seeds) or files != expected:
        return "partial", record
    for run in runs:
        path = directory / f"grid4x4_{TIER}_{METHOD}_seed{int(run['seed'])}.pt"
        if (_sha256_file(path) != run["file_sha256"] or int(run["gradient_steps"]) != protocol.declared_gradient_steps
                or int(run["training_rows"]) != int(declared_rows)
                or Path(run["checkpoint_path"]).resolve() != path.resolve()):
            return "partial", record
    return "complete", record


def _corrected_rows(out: Path, protocol: Protocol, corrected_ours: Sequence[Any],
                    record: Mapping[str, Any]) -> list[dict[str, Any]]:
    from offline.att_rederivation import cell_file_name

    work = out / REDERIVATION_SUBDIR
    reference = {(episode.seed, episode.draw_id): episode.att_horizon for episode in corrected_ours}
    checkpoints = {int(run["seed"]): str(Path(run["checkpoint_path"]).resolve()) for run in record["runs"]}
    rows = []
    for cell in rederivation_cells(protocol.seeds, protocol.draw_ids):
        path = work / cell_file_name(cell)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: the corrected cell is incomplete")
        row = _read_json(path)
        key = (int(row["seed"]), int(row["draw_id"]))
        if (row.get("reproduces_committed") is not True or str(row["arm"]) != ARM or key != (cell.seed, cell.draw_id)
                or float(row["att_ours"]) != reference[key] or float(row["committed_att_ours"]) != reference[key]):
            raise ValueError(f"{path} does not reproduce (i)'s att_horizon for its episode")
        source = str(Path(str(row["policy_source"].get("checkpoint", ""))).resolve())
        if source != checkpoints[cell.seed]:
            raise ValueError(f"{path} was rolled from {source}, not the corrected checkpoint {checkpoints[cell.seed]}")
        rows.append(row)
    return rows


def _campaign_anchor(work: Path) -> dict[str, Any]:
    """A campaign directory's manifest, its digest recomputed by P8.4b's ``campaign_manifest`` from its own cell list
    and equal to the recorded one and to the completion marker's."""
    from offline.att_rederivation import COMPLETE_MARKER, MANIFEST_NAME as CAMPAIGN_MANIFEST
    from offline.att_rederivation import CellKey, campaign_manifest

    recorded = _read_json(work / CAMPAIGN_MANIFEST)
    cells = [CellKey(scenario=s, arm=a, seed=None if seed == "None" else int(seed), draw_id=int(d))
             for s, a, seed, d in (str(key).split("|") for key in recorded["cells"])]
    recomputed = campaign_manifest(cells, engine_seed=int(recorded["engine_seed"]))["declared_cells_sha256"]
    marker = _read_json(work / COMPLETE_MARKER)["declared_cells_sha256"]
    if not recomputed == recorded["declared_cells_sha256"] == marker:
        raise ValueError(f"{work}: declared_cells_sha256 recomputed {recomputed}, recorded "
                         f"{recorded['declared_cells_sha256']}, marker {marker}")
    return {"declared_cells_sha256": recomputed, "n_cells": len(cells), "engine_seed": int(recorded["engine_seed"])}


def _verify_run_manifest(roots: Roots) -> dict[str, Any]:
    output = Path(roots.output_root)
    out = Path(roots.out_root)
    path = manifest_path(output)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the report is built after the run's manifest")
    lines = path.read_text(encoding="utf-8").splitlines()
    listed = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        listed[name] = digest
    expected = {f"{RUN_DIR}/{p.relative_to(out).as_posix()}" for p in _run_files(out)}
    if set(listed) != expected:
        raise ValueError(f"{path} lists {len(listed)} files, the run directory holds {len(expected)} "
                         f"({sorted(set(listed) ^ expected)[:3]} differ)")
    for name, digest in listed.items():
        if _sha256_file(output / name) != digest:
            raise ValueError(f"{output / name} does not match its line in {path}")
    return {"file": f"output/{MANIFEST_NAME}", "sha256": _sha256_file(path), "files": len(lines)}


def _canary(out: Path, name: str) -> dict[str, Any]:
    path = out / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the training's seconds are recorded between two canaries")
    canary = _read_json(path)
    return {"file": f"output/{RUN_DIR}/{name}", "sha256": _sha256_file(path), "seconds": canary.get("seconds"),
            "verdict": canary.get("verdict"), "reproduced": canary.get("reproduced"),
            "threshold_seconds": canary.get("threshold_seconds"), "power": canary.get("power"),
            "written_utc": canary.get("written_utc")}


def load_report_inputs(roots: Roots, *, pins: Pins = PINS) -> dict[str, Any]:
    """Every input of the artifact, each verified at its anchor before it is read."""
    from offline.tier_sweep import cell_is_complete

    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    verified = _verify_inputs(roots, pins)
    protocol: Protocol = verified["protocol"]
    original_ours = p5_2_cells(output, verified["sums"])
    original_engine = p8_4b_cells(output, original_ours, definition="att_engine", pins=pins)
    original_rows = _p8_4b_rows(output, METHOD, TIER, original_ours[(METHOD, TIER)])
    state, record = _training_complete(out, protocol, verified["declared"]["rows"])
    if state != "complete" or record is None:
        raise ValueError(f"the training stage is {state}: the report needs the five corrected checkpoints and their record")
    canaries = {"open": _canary(out, CANARY_NAMES[0]), "close": _canary(out, CANARY_NAMES[1])}
    eval_path = out / EVAL_NAME
    if not eval_path.is_file():
        raise FileNotFoundError(f"{eval_path} is missing: (i) has not run")
    corrected_payload = _read_json(eval_path)
    if not cell_is_complete(corrected_payload, seeds=protocol.seeds, draws=protocol.draw_ids,
                            declared_steps=protocol.declared_gradient_steps, method=METHOD):
        raise ValueError(f"{eval_path} is not a complete cell of the protocol")
    assert_evaluated_models(corrected_payload, record)
    corrected_ours = episodes_from_rows(corrected_payload["episodes"], value_key="att_horizon")
    corrected_rows = _corrected_rows(out, protocol, corrected_ours, record)
    rederivation = _campaign_anchor(out / REDERIVATION_SUBDIR)
    manifest = _verify_run_manifest(roots)
    return {
        "verified": verified, "protocol": protocol, "record": record, "canaries": canaries,
        "original_ours": original_ours, "original_engine": original_engine, "original_rows": original_rows,
        "corrected_payload": corrected_payload, "corrected_ours": corrected_ours, "corrected_rows": corrected_rows,
        "rederivation": rederivation, "manifest": manifest,
        "files": {"training": {"file": f"output/{RUN_DIR}/{TRAINING_RECORD_NAME}",
                               "sha256": _sha256_file(out / TRAINING_RECORD_NAME)},
                  "p5_2_path": {"file": f"output/{RUN_DIR}/{EVAL_NAME}", "sha256": _sha256_file(eval_path)}},
    }


WHAT_THIS_DOES_NOT_SAY: tuple[str, ...] = (
    "The original cell is kept, unchanged, as the record of the defect: nothing under output/p5_2/ was written, moved "
    "or deleted, and P5.2's committed statements stand as written, with this artifact beside them.",
    "The other arms were not re-run: their att_ours is P5.2's committed evaluation and their att_engine P8.4b's "
    "committed campaign, each read at the anchors named in inputs.",
    "No other tier, method, scenario or seed was touched and no hypothesis is new (BRIEF_44: the author's ruling A of "
    "2026-10-05, a narrow exception to A25).",
    "The corrected weights are one realisation of P5.2's default CUDA regime, which does not reproduce bit for bit "
    "from run to run; no second realisation was trained (Amendment A, Q12).",
    "Q1's predictions were registered on att_horizon (att_ours); under att_engine they are compared with measurements "
    "of the other definition.",
    "The statements without an IQL arm (Q2b's hard subset under att_engine; Q3b, Q4, Q5 and Q6) are not recomputed: "
    "the correction cannot move them under att_ours, and their att_engine reading is the Rule-R recomputation of "
    "P5.2's statements, DEFERRED 109.",
    "P8.4b's att_engine values are pinned by its campaign's completeness and the cross-checks named in inputs, not by "
    "a digest manifest, which P8.4b never wrote (Amendment A, A1.4).",
    "The training seconds cover each seed's gradient loop only (offline_baselines.train_iql's time.time()), measured "
    "between two machine-health canaries.",
)


def report_from_inputs(inputs: Mapping[str, Any], *, git: Mapping[str, Any]) -> dict[str, Any]:
    """The ``p5.2b-correction/1.0`` artifact from verified inputs; pure."""
    verified = inputs["verified"]
    protocol: Protocol = inputs["protocol"]
    record = inputs["record"]
    declared = verified["declared"]
    original = verified["original"]
    corrected_engine = episodes_from_rows(inputs["corrected_rows"], value_key="att_engine")
    corrected_ours_ii = episodes_from_rows(inputs["corrected_rows"], value_key="att_ours")
    equal = sum(a.att_horizon == b.att_horizon for a, b in zip(inputs["corrected_ours"], corrected_ours_ii))
    before_ours = dict(inputs["original_ours"])
    after_ours = {**before_ours, (METHOD, TIER): list(inputs["corrected_ours"])}
    before_engine = dict(inputs["original_engine"])
    after_engine = {**before_engine, (METHOD, TIER): corrected_engine}
    seconds = [float(run["seconds"]) for run in record["runs"]]
    header = {key: record[key] for key in ("format_version", "seeds", "declared_gradient_steps", "batch_size",
                                           "regime", "concurrency", "table") if key in record}
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": ("P5.2b: P5.2's IQL cell at grid4x4's random tier, re-trained on the declared 200 episodes (DEFERRED "
                 "106), evaluated by P5.2's own path and by P8.4b's cell runner on the same episodes, and P5.2's "
                 "random-tier statements that involve IQL recomputed before and after under both ATT definitions"),
        "registered_in": ("docs/briefs/BRIEF_44_p5.2b_iql_correction.md and its Amendment A; the author's ruling A of "
                          "2026-10-05 (PROJECT_PLAN section 8), a narrow exception to A25"),
        "git": dict(git),
        "defect": {
            "deferred": 106,
            "mechanism": ("_run_train_baselines filtered BC's and %BC's batches to the declared streams but built "
                          "IQL's transition table from the whole dataset (offline_baselines.build_transitions reads "
                          "every episode record), so IQL at the random tier trained on every available episode"),
            "file_line": {"at_the_training_commit": "offline/tier_sweep.py:1820 at 9460800, the commit every "
                                                    "original checkpoint records",
                          "at_main_before_the_fix": "offline/tier_sweep.py:2078 at f077875"},
            "fixed_by": "offline.tier_sweep.iql_transition_table (BRIEF_44 section 4.1)",
            "original": {"training_rows": original["common"]["training_rows"],
                         "training_streams_recorded": original["common"]["training_streams"],
                         "episodes_available": declared["episodes_available"], "per_seed": original["per_seed"]},
            "declared": {key: declared[key] for key in ("episodes_selected", "streams", "nodes", "rows",
                                                        "whole_tier_rows", "declaration", "corpus_manifest")},
        },
        "training": {
            "record": inputs["files"]["training"], "header": header, "runs": list(record["runs"]),
            "seconds": {"per_seed": [{"seed": int(run["seed"]), "seconds": float(run["seconds"])}
                                     for run in record["runs"]],
                        "median": statistics.median(seconds), "min": min(seconds), "max": max(seconds),
                        "covers": "each seed's gradient loop (offline_baselines.train_iql, time.time() around it)"},
            "canaries": inputs["canaries"],
            "hyperparameters_of_the_originals": original["common"],
        },
        "evaluation": {
            "protocol": {"seeds": list(protocol.seeds), "draw_ids": list(protocol.draw_ids),
                         "engine_seed": protocol.engine_seed, "deterministic": protocol.deterministic,
                         "declared_gradient_steps": protocol.declared_gradient_steps,
                         "read_from": "output/p5_2/eval_random_iql.json"},
            "p5_2_path": {**inputs["files"]["p5_2_path"], "code": "offline.tier_sweep.main evaluate (in-process)"},
            "p8_4b_path": {"work_dir": f"output/{RUN_DIR}/{REDERIVATION_SUBDIR}",
                           "declared_cells_sha256": inputs["rederivation"]["declared_cells_sha256"],
                           "n_cells": inputs["rederivation"]["n_cells"],
                           "code": "offline.att_rederivation.run_campaign, its committed map (i)'s output"},
        },
        "cells": {
            "original": {"att_ours": cell_summary(inputs["original_ours"][(METHOD, TIER)]),
                         "att_engine": cell_summary(inputs["original_engine"][(METHOD, TIER)]),
                         "admission": admission_summary(inputs["original_rows"]),
                         "sources": {"att_ours": "output/p5_2/eval_random_iql.json",
                                     "att_engine": "output/p8_4b_rederivation/cell_grid4x4_iql_at_random_*"}},
            "corrected": {"att_ours": cell_summary(inputs["corrected_ours"]),
                          "att_ours_p8_4b_path": cell_summary(corrected_ours_ii),
                          "att_engine": cell_summary(corrected_engine),
                          "admission": admission_summary(inputs["corrected_rows"]),
                          "paths_agree": {"episodes": len(corrected_ours_ii), "equal": int(equal)}},
        },
        "statements": {
            "att_ours": {"before": random_tier_statements(before_ours, with_hard_subset=True),
                         "after": random_tier_statements(after_ours, with_hard_subset=True)},
            "att_engine": {"before": random_tier_statements(before_engine, with_hard_subset=False),
                           "after": random_tier_statements(after_engine, with_hard_subset=False)},
        },
        "not_recomputed": [
            {"statements": "Q3b, Q4, Q5, Q6 at the random tier", "reason": "no IQL arm: the correction cannot move them"},
            {"statements": "Q2b's hard subset under att_engine, and every IQL-free statement under att_engine",
             "reason": "the Rule-R recomputation of P5.2's statements, DEFERRED 109 (Amendment A, A2)"},
        ],
        "inputs": {
            "p5_2_manifest": {"file": f"output/{P5_2_SUMS_NAME}", "sha256": verified["p5_2_sums_sha256"]},
            "declaration": declared["declaration"], "corpus_manifest": declared["corpus_manifest"],
            "c1_note": {"file": verified["note"]["file"], "sha256": verified["note"]["sha256"]},
            "p8_4b_campaign": {"declared_cells_sha256": verified["campaign"]["declared_cells_sha256"],
                               "n_cells": verified["campaign"]["n_cells"],
                               "anchor": ("the digest recomputed by att_rederivation.campaign_manifest from the "
                                          "manifest's own cell list, equal to CAMPAIGN_COMPLETE's and to the pin; "
                                          "every cell read reproduces P5.2's committed att_horizon")},
            "run_manifest": inputs["manifest"],
        },
        "what_this_does_not_say": list(WHAT_THIS_DOES_NOT_SAY),
    }


def build_report(roots: Roots, *, git: Mapping[str, Any], pins: Pins = PINS) -> dict[str, Any]:
    """:func:`load_report_inputs` then :func:`report_from_inputs`."""
    return report_from_inputs(load_report_inputs(roots, pins=pins), git=git)


def write_report(roots: Roots, artifact: Mapping[str, Any], protected: Sequence[Path]) -> Path:
    """Write the artifact once, canonically, under ``<out-root>/artifacts/``."""
    path = Path(roots.out_root) / ARTIFACTS_SUBDIR / ARTIFACT_NAME
    target = assert_target(path, roots, protected)
    if target.exists():
        raise FileExistsError(f"{target} already exists: the artifact is written once")
    text = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ----------------------------------------------------------------------
# The run's stages
# ----------------------------------------------------------------------


def _rederivation_state(out: Path, protocol: Protocol, payload: Mapping[str, Any] | None,
                        record: Mapping[str, Any] | None) -> str:
    from offline.att_rederivation import campaign_status

    work = out / REDERIVATION_SUBDIR
    if not work.is_dir():
        return "absent"
    state = campaign_status(work)
    if not state["complete"] or not state["marker_present"] or payload is None or record is None:
        return "partial"
    try:
        _corrected_rows(out, protocol, episodes_from_rows(payload["episodes"], value_key="att_horizon"), record)
    except (ValueError, FileNotFoundError, KeyError):
        return "partial"
    return "complete"


def status(roots: Roots, *, pins: Pins = PINS) -> dict[str, str]:
    """Each stage's state from disk: ``absent``, ``complete`` or ``partial`` (a refusal for the driver)."""
    from offline.tier_sweep import cell_is_complete

    output = Path(roots.output_root)
    out = Path(roots.out_root)
    sums = p5_2_sums(output, pins=pins)
    protocol = original_protocol(read_p5_2_json(output, P5_2_EVAL_RELPATH, sums))
    declared = declared_selection(roots.repo_root, roots.corpus_root, pins=pins)
    training, record = _training_complete(out, protocol, declared["rows"])
    canaries = [(out / name).is_file() for name in CANARY_NAMES]
    state: dict[str, str] = {
        "training": training,
        "canaries": "complete" if all(canaries) else ("partial" if any(canaries) else "absent"),
    }
    eval_path = out / EVAL_NAME
    payload: Mapping[str, Any] | None = None
    if not eval_path.is_file():
        state["p5_2_eval"] = "absent"
    else:
        payload = _read_json(eval_path)
        complete = cell_is_complete(payload, seeds=protocol.seeds, draws=protocol.draw_ids,
                                    declared_steps=protocol.declared_gradient_steps, method=METHOD)
        if complete and record is not None and training == "complete":
            try:
                assert_evaluated_models(payload, record)
            except ValueError:
                complete = False
        state["p5_2_eval"] = "complete" if complete and training == "complete" else "partial"
    state["rederivation"] = _rederivation_state(out, protocol, payload if state["p5_2_eval"] == "complete" else None,
                                                record)
    state["manifest"] = "complete" if manifest_path(output).is_file() else "absent"
    state["report"] = "complete" if (out / ARTIFACTS_SUBDIR / ARTIFACT_NAME).is_file() else "absent"
    return state


def check(roots: Roots, *, pins: Pins = PINS, require_cuda: bool = True) -> dict[str, Any]:
    """Every pre-token refusal the module owns (plan §8); writes nothing.  ``require_cuda`` is false only in tests."""
    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    if out != output.resolve() / RUN_DIR:
        raise PermissionError(f"check is the run's: its out-root is {output / RUN_DIR}, not {out}")
    verified = _verify_inputs(roots, pins)
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device: P5.2 trained this cell on CUDA, and the correction may not change the regime")
    protected = protected_roots(roots)
    for target in (out / TRAINING_RECORD_NAME, checkpoint_dir(out) / "x.pt", out / REDERIVATION_SUBDIR / "x.json",
                   manifest_path(output)):
        assert_target(target, roots, protected)
    state = status(roots, pins=pins)
    if manifest_path(output).exists():
        raise FileExistsError(f"{manifest_path(output)} already exists: a run's manifest is written once and a run "
                              "with its manifest is final")
    partial = [stage for stage, value in state.items() if value == "partial" and stage != "rederivation"]
    if state["training"] != "complete" and state["canaries"] != "absent":
        partial.append("canaries")
    if partial:
        raise ValueError(f"the run directory is partial in {sorted(set(partial))}: move output/p5_2b aside by hand "
                         "and start again (the driver's restart rule); nothing here deletes")
    fresh = all(value == "absent" for value in state.values())
    declared = {key: value for key, value in verified["declared"].items() if key != "keys"}
    return {"ok": True, "state": "fresh" if fresh else "resumable", "status": state, "declared": declared,
            "protocol": {"seeds": list(verified["protocol"].seeds), "draws": len(verified["protocol"].draw_ids),
                         "engine_seed": verified["protocol"].engine_seed},
            "p8_4b_cells": verified["campaign"]["n_cells"]}


def _regime() -> dict[str, Any]:
    import torch

    regime = {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_num_threads": int(torch.get_num_threads()), "torch_version": str(torch.__version__),
        "device": "cuda", "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if (regime["omp_num_threads"], regime["mkl_num_threads"]) != ("1", "1") or regime["cublas_workspace_config"]:
        raise RuntimeError(f"the process regime is not P5.2's default one (OMP/MKL 1, CUBLAS_WORKSPACE_CONFIG unset): {regime}")
    return regime


def _canary_at_speed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the training starts after the opening canary")
    canary = _read_json(path)
    if canary.get("verdict") != "at speed" or canary.get("reproduced") is not True:
        raise ValueError(f"{path}: the canary is {canary.get('verdict')!r}, reproduced {canary.get('reproduced')!r}; "
                         "no training starts on a machine that is not at reference (PROJECT_PLAN section 7)")
    return canary


def train_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """The five seeds and the training record, written once."""
    import torch

    from offline.compute_latency import _git_provenance
    from offline.offline_baselines import IQL_BATCH_TRANSITIONS
    from offline.tier_sweep import configure_determinism

    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    verified = _verify_inputs(roots, pins)
    state = status(roots, pins=pins)
    if state["training"] != "absent" or (out / CANARY_NAMES[1]).exists():
        raise ValueError(f"the training stage is {state['training']}: a complete one is skipped by the driver and a "
                         "partial one is moved aside by hand")
    _canary_at_speed(out / CANARY_NAMES[0])
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: P5.2 trained this cell on CUDA")
    configure_determinism(False)
    torch.set_num_threads(1)
    regime = _regime()
    protected = protected_roots(roots)
    started = time.perf_counter()
    inputs = training_inputs(roots.corpus_root)
    table_seconds = time.perf_counter() - started
    checked = assert_training_inputs(inputs, verified["declared"], verified["original"])
    protocol: Protocol = verified["protocol"]
    runs = train_seeds(inputs, seeds=protocol.seeds, gradient_steps=protocol.declared_gradient_steps,
                       device=torch.device("cuda"), checkpoint_dir=checkpoint_dir(out), protected=protected,
                       correction=correction_block(verified["declared"], verified["original"]), log_every=0)
    record = {
        "format_version": TRAINING_FORMAT_VERSION, "tier": TIER, "method": METHOD, "seeds": list(protocol.seeds),
        "declared_gradient_steps": protocol.declared_gradient_steps, "batch_size": IQL_BATCH_TRANSITIONS,
        "regime": regime,
        "concurrency": {"value": 1, "basis": "the five seeds train in sequence in one process, and the driver "
                                              "refuses a second offline.iql_correction interpreter"},
        "table": {**checked, "episodes": verified["declared"]["episodes_selected"], "build_seconds": table_seconds,
                  "filter": "offline.tier_sweep.iql_transition_table"},
        "stage_seconds": time.perf_counter() - started,
        "canary_open": _canary(out, CANARY_NAMES[0]),
        "git": _git_provenance(),
        "runs": runs,
    }
    return write_once_json(out / TRAINING_RECORD_NAME, record, roots, protected)


def evaluate_p5_2_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """(i): P5.2's evaluate subcommand on the corrected checkpoints."""
    from offline.tier_sweep import cell_is_complete

    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    verified = _verify_inputs(roots, pins)
    protocol: Protocol = verified["protocol"]
    state = status(roots, pins=pins)
    if state["training"] != "complete":
        raise ValueError(f"(i) needs the complete training, and it is {state['training']}")
    path = out / EVAL_NAME
    if state["p5_2_eval"] == "complete":
        print(f"SKIP (i): {path} is complete", flush=True)
        return path
    if path.exists():
        raise ValueError(f"{path} exists but is not a complete cell of the corrected models: moved aside by hand")
    record = _read_json(out / TRAINING_RECORD_NAME)
    code = run_p5_2_evaluation(p5_2_evaluation_argv(roots, pins=pins))
    if code != 0:
        raise RuntimeError(f"P5.2's evaluate subcommand returned {code}")
    payload = _read_json(path)
    if not cell_is_complete(payload, seeds=protocol.seeds, draws=protocol.draw_ids,
                            declared_steps=protocol.declared_gradient_steps, method=METHOD):
        raise ValueError(f"{path} is not a complete cell of the protocol")
    assert_evaluated_models(payload, record)
    return path


def evaluate_p8_4b_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """(ii): P8.4b's runner on the corrected checkpoints, resumable per cell."""
    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    verified = _verify_inputs(roots, pins)
    protocol: Protocol = verified["protocol"]
    state = status(roots, pins=pins)
    if state["p5_2_eval"] != "complete":
        raise ValueError(f"(ii) checks every episode against (i), and (i) is {state['p5_2_eval']}")
    work = out / REDERIVATION_SUBDIR
    if state["rederivation"] == "complete":
        print(f"SKIP (ii): {work} is complete", flush=True)
        return work
    record = _read_json(out / TRAINING_RECORD_NAME)
    resolved = resolved_checkpoints(roots, resolver_root=out, seeds=protocol.seeds)
    for run in record["runs"]:
        path = resolved[int(run["seed"])]
        if path.resolve() != Path(run["checkpoint_path"]).resolve() or _sha256_file(path) != run["file_sha256"]:
            raise ValueError(f"P8.4b's resolver finds {path} for seed {run['seed']}, not the corrected checkpoint")
    payload = _read_json(out / EVAL_NAME)
    result = run_rederivation(rederivation_cells(protocol.seeds, protocol.draw_ids), roots=roots, resolver_root=out,
                              work_dir=work, committed=committed_from_eval(payload),
                              protected=protected_roots(roots), engine_seed=protocol.engine_seed)
    if result["outcome"]["n_refused"] or not result["status"]["complete"]:
        raise RuntimeError(f"(ii) refused {result['outcome']['n_refused']} cell(s); complete={result['status']['complete']}")
    _corrected_rows(out, protocol, episodes_from_rows(payload["episodes"], value_key="att_horizon"), record)
    return work


def write_run_manifest(roots: Roots, protected: Sequence[Path]) -> list[str]:
    """``<output>/SHA256SUMS_p5_2b.txt`` over every run file but ``artifacts/``, as ``<sha256>  p5_2b/<path>``, once."""
    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    if out != output.resolve() / RUN_DIR:
        raise PermissionError(f"only the run writes {MANIFEST_NAME}; {out} is a pre-flight")
    target = assert_target(manifest_path(output), roots, protected)
    if target.exists():
        raise FileExistsError(f"{target} already exists: the manifest is written once")
    lines = sorted(f"{_sha256_file(path)}  {RUN_DIR}/{path.relative_to(out).as_posix()}" for path in _run_files(out))
    if not lines:
        raise ValueError(f"{out} holds no file to list")
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    for line in target.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if _sha256_file(output / name) != digest:
            raise ValueError(f"{target}: {name} no longer matches its line")
    return lines


def report_stage(roots: Roots, *, pins: Pins = PINS) -> Path:
    """Build and write the artifact."""
    from offline.compute_latency import _git_provenance

    artifact = build_report(roots, git=_git_provenance(), pins=pins)
    return write_report(roots, artifact, protected_roots(roots))


def preflight_estimate(*, table_seconds: float, rehearsal_steps: int, rehearsal_seconds: float,
                       episode_seconds_p5_2: float, episode_seconds_p8_4b: float, n_seeds: int,
                       declared_steps: int, n_episodes: int) -> dict[str, Any]:
    """The run's estimate from the pre-flight's measurements, and each stage's timeout: ``TIMEOUT_FACTOR`` x its
    estimate, never below ``TIMEOUT_FLOOR_SECONDS``.  An estimate, labelled one."""
    per_step = float(rehearsal_seconds) / int(rehearsal_steps)
    training = float(table_seconds) + int(n_seeds) * int(declared_steps) * per_step
    p5_2 = int(n_episodes) * float(episode_seconds_p5_2)
    p8_4b = int(n_episodes) * float(episode_seconds_p8_4b)
    timeouts = {name: max(TIMEOUT_FLOOR_SECONDS, TIMEOUT_FACTOR * value)
                for name, value in (("train", training), ("evaluate_p5_2", p5_2), ("evaluate_p8_4b", p8_4b))}
    return {"seconds_per_step": per_step, "training_seconds": training, "evaluate_p5_2_seconds": p5_2,
            "evaluate_p8_4b_seconds": p8_4b, "total_seconds": training + p5_2 + p8_4b, "timeouts": timeouts,
            "note": (f"an estimate, not a bound: the training from {rehearsal_steps} rehearsal steps of one seed plus "
                     "the table's build, each evaluation from five episodes on its path; each timeout is "
                     f"{TIMEOUT_FACTOR:g} x its estimate, never below {TIMEOUT_FLOOR_SECONDS:g} s")}


def timeouts_from_preflight(path: str | Path) -> dict[str, float]:
    """The stage timeouts of a pre-flight record, refused unless it is COMPLETE and both T-reproduce runs reproduced."""
    record = _read_json(path)
    if record.get("format_version") != PREFLIGHT_FORMAT_VERSION or record.get("status") != "COMPLETE":
        raise ValueError(f"{path} is not a COMPLETE {PREFLIGHT_FORMAT_VERSION} record (status {record.get('status')!r})")
    for name in ("p5_2_path", "p8_4b_path"):
        if record.get("reproduce", {}).get(name, {}).get("reproduced") is not True:
            raise ValueError(f"{path}: T-reproduce on the {name} did not reproduce the committed episodes")
    timeouts = record["estimate"]["timeouts"]
    return {name: float(timeouts[name]) for name in ("train", "evaluate_p5_2", "evaluate_p8_4b")}


def _compare(mine: Mapping[str, Any], theirs: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    return [f"{field}: {mine.get(field)!r} != {theirs.get(field)!r}" for field in fields
            if mine.get(field) != theirs.get(field)]


def preflight(roots: Roots, *, stamp: str, steps: int, draws: Sequence[int], pins: Pins = PINS) -> dict[str, Any]:
    """Gate G1's pre-flight (Amendment A, A3.1): a short training timed, T-reproduce (a) and (b) RUN on the original
    seed-101 checkpoint with their outcomes, the run's estimate and the stage timeouts."""
    import torch

    from offline.att_rederivation import cell_file_name
    from offline.compute_latency import _git_provenance
    from offline.tier_sweep import configure_determinism

    output = Path(roots.output_root)
    out = assert_out_root(roots.out_root, output)
    if out == output.resolve() / RUN_DIR:
        raise PermissionError("the pre-flight never writes into the run's directory; its out-root is under p5_2b_runs/")
    protected = protected_roots(roots)
    verified = _verify_inputs(roots, pins)
    protocol: Protocol = verified["protocol"]
    reasons: list[str] = []
    try:
        canary = _canary_at_speed(out / CANARY_NAMES[0])
    except (ValueError, FileNotFoundError) as exc:
        canary = None
        reasons.append(str(exc))
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: the pre-flight times the training on the run's device")
    configure_determinism(False)
    torch.set_num_threads(1)
    regime = _regime()

    started = time.perf_counter()
    inputs = training_inputs(roots.corpus_root)
    table_seconds = time.perf_counter() - started
    checked = assert_training_inputs(inputs, verified["declared"], verified["original"])
    runs = train_seeds(inputs, seeds=(PREFLIGHT_SEED,), gradient_steps=int(steps), device=torch.device("cuda"),
                       checkpoint_dir=checkpoint_dir(out), protected=protected,
                       correction={**correction_block(verified["declared"], verified["original"]),
                                   "rehearsal": f"G1 pre-flight {stamp}: {steps} steps, never a candidate"})
    rehearsal = {"seed": PREFLIGHT_SEED, "steps": int(steps), "seconds": runs[0]["seconds"],
                 "table_seconds": table_seconds, "table": checked}

    original = verified["original_payload"]
    committed = {(int(e["seed"]), int(e["draw_id"])): e for e in original["episodes"] if int(e["seed"]) == PREFLIGHT_SEED}
    work_a = out / "reproduce_p5_2"
    argv = p5_2_evaluation_argv(roots, seeds=(PREFLIGHT_SEED,), checkpoint_dir=output / "p5_2" / "checkpoints",
                                work_dir=work_a, pins=pins)
    started = time.perf_counter()
    code = run_p5_2_evaluation(argv, draws=draws)
    seconds_a = time.perf_counter() - started
    rerun = _read_json(work_a / f"eval_{TIER}_{METHOD}_seed{PREFLIGHT_SEED}.json")["episodes"] if code == 0 else []
    differences_a = []
    for episode in rerun:
        differences_a += [f"draw {episode['draw_id']} {d}" for d in
                          _compare(episode, committed[(PREFLIGHT_SEED, int(episode["draw_id"]))],
                                   ("att_horizon", "horizon_vehicle_count", "episode_reward"))]
    reproduced_a = code == 0 and sorted(int(e["draw_id"]) for e in rerun) == sorted(int(d) for d in draws) \
        and not differences_a
    cells = rederivation_cells((PREFLIGHT_SEED,), draws)
    work_b = out / "reproduce_p8_4b"
    started = time.perf_counter()
    result = run_rederivation(cells, roots=roots, resolver_root=output, work_dir=work_b,
                              committed=committed_from_eval(original), protected=protected,
                              engine_seed=protocol.engine_seed)
    seconds_b = time.perf_counter() - started
    differences_b = []
    for cell in cells:
        mine_path = work_b / cell_file_name(cell)
        if not mine_path.is_file():
            differences_b.append(f"{cell_file_name(cell)} was not written")
            continue
        mine = _read_json(mine_path)
        theirs = _read_json(output / P8_4B_DIR / cell_file_name(cell))
        fields = sorted((set(mine) | set(theirs)) - {"seconds", "seconds_rollout"})
        differences_b += [f"{cell_file_name(cell)} {d}" for d in _compare(mine, theirs, fields)]
    reproduced_b = result["outcome"]["n_refused"] == 0 and not differences_b
    estimate = preflight_estimate(
        table_seconds=table_seconds, rehearsal_steps=int(steps), rehearsal_seconds=float(runs[0]["seconds"]),
        episode_seconds_p5_2=seconds_a / max(len(draws), 1), episode_seconds_p8_4b=seconds_b / max(len(draws), 1),
        n_seeds=len(protocol.seeds), declared_steps=protocol.declared_gradient_steps,
        n_episodes=len(protocol.seeds) * len(protocol.draw_ids))
    if not reproduced_a:
        reasons.append("T-reproduce (a) did not reproduce the committed episodes")
    if not reproduced_b:
        reasons.append("T-reproduce (b) did not reproduce P8.4b's committed cells")
    record = {
        "format_version": PREFLIGHT_FORMAT_VERSION, "stamp": stamp, "status": "FAILED" if reasons else "COMPLETE",
        "reasons": reasons, "git": _git_provenance(), "regime": regime,
        "canary_open": None if canary is None else _canary(out, CANARY_NAMES[0]),
        "rehearsal": rehearsal,
        "reproduce": {
            "p5_2_path": {"reproduced": bool(reproduced_a), "checkpoint": "output/p5_2/checkpoints/"
                          f"grid4x4_random_iql_seed{PREFLIGHT_SEED}.pt", "draws": [int(d) for d in draws],
                          "episodes": len(rerun), "differences": differences_a, "seconds": seconds_a,
                          "exit_code": code},
            "p8_4b_path": {"reproduced": bool(reproduced_b), "draws": [int(d) for d in draws],
                           "cells": len(cells), "refused": result["outcome"]["n_refused"],
                           "differences": differences_b, "seconds": seconds_b},
        },
        "estimate": estimate,
    }
    write_once_json(out / "preflight.json", record, roots, protected)
    return record


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> Any:
    """``check``, ``status``, ``train``, ``evaluate-p5-2``, ``evaluate-p8-4b``, ``manifest``, ``report``,
    ``preflight``, ``timeouts``."""
    parser = argparse.ArgumentParser(prog="python -m offline.iql_correction", allow_abbrev=False,
                                     description="P5.2b: the IQL random-tier correction run (BRIEF_44)")
    sub = parser.add_subparsers(dest="command", required=True)

    def roots(command: Any) -> Any:
        command.add_argument("--output-root", type=Path, default=Path("/home/filip/rltraffic/output"))
        command.add_argument("--corpus-root", type=Path, default=Path("/home/filip/rltraffic/datasets_v11"))
        command.add_argument("--draws-root", type=Path, default=Path("/home/filip/rltraffic/scenarios/draws"))
        command.add_argument("--repo-root", type=Path, default=_MODULE_ROOT)
        command.add_argument("--out-root", type=Path, default=None, help="default <output-root>/p5_2b")
        return command

    for name in ("check", "train", "evaluate-p5-2", "evaluate-p8-4b", "manifest", "report"):
        roots(sub.add_parser(name, allow_abbrev=False))
    roots(sub.add_parser("status", allow_abbrev=False)).add_argument("--stage", default=None)
    flight = roots(sub.add_parser("preflight", allow_abbrev=False))
    flight.add_argument("--stamp", required=True)
    flight.add_argument("--steps", type=int, default=PREFLIGHT_STEPS)
    sub.add_parser("timeouts", allow_abbrev=False).add_argument("--preflight-record", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "timeouts":
            values = timeouts_from_preflight(args.preflight_record)
            print(f"{values['train']:g} {values['evaluate_p5_2']:g} {values['evaluate_p8_4b']:g}", flush=True)
            return 0
        roots = Roots(repo_root=args.repo_root, output_root=args.output_root, corpus_root=args.corpus_root,
                      draws_root=args.draws_root, out_root=args.out_root or run_root(args.output_root))
        if args.command == "check":
            print(json.dumps(check(roots), indent=2, sort_keys=True), flush=True)
        elif args.command == "status":
            state = status(roots)
            print(state[args.stage] if args.stage else json.dumps(state, indent=2, sort_keys=True), flush=True)
        elif args.command == "train":
            print(f"training record: {train_stage(roots)}", flush=True)
        elif args.command == "evaluate-p5-2":
            print(f"(i): {evaluate_p5_2_stage(roots)}", flush=True)
        elif args.command == "evaluate-p8-4b":
            print(f"(ii): {evaluate_p8_4b_stage(roots)}", flush=True)
        elif args.command == "manifest":
            lines = write_run_manifest(roots, protected_roots(roots))
            print(f"{manifest_path(roots.output_root)}: {len(lines)} files", flush=True)
        elif args.command == "report":
            print(f"artifact: {report_stage(roots)}", flush=True)
        elif args.command == "preflight":
            record = preflight(roots, stamp=args.stamp, steps=args.steps, draws=PREFLIGHT_DRAWS)
            path = Path(roots.out_root) / "preflight.json"
            print(f"pre-flight {args.stamp}: {record['status']}" + (f" ({'; '.join(record['reasons'])})"
                                                                   if record["reasons"] else ""), flush=True)
            print(f"record {path} sha256 {_sha256_file(path)}", flush=True)
            return 0 if record["status"] == "COMPLETE" else 1
    except (ValueError, PermissionError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the driver
    raise SystemExit(main())
