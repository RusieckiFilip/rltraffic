"""P8.2: the compute-and-latency table -- parameters, training cost and decision latency, every number with its source.

Written against ``docs/briefs/BRIEF_43_p8.2_compute_latency.md`` §5 and its **Amendment A** (Q3: one row per arm with
its tiers inside and an ``architecture`` key; Q4: MAPPO's ``results.json`` digests pinned here as of 2026-10-03;
Q5/Q6: absences declared, never reconstructed; Q7: ``trained`` / ``deployed`` / ``stored`` /
``executed_per_decision``; Q13: parameters counted on all five seeds), on the plan ``docs/plans/p8.2.md`` @ ``a7c43e1``.

On-disk format
--------------
``p8.2-compute/1.0`` -- ``output/p8_2/artifacts/p8_2_compute.json``, committed by hand as ``docs/data/p8_2_compute.json``.
Its ``rows`` carry, per row: the claims it serves with the committed result each sits in; every checkpoint (tier,
seed, path, digest and the record that names it); the parameter counts (route A, the method's own loader, equal to
route B, the payload's parameter tensors, on every checkpoint); the training cost per (record, tier) -- median
[min-max] over seeds, never pooled across records, each with its regime and with what its wall time covers; the
environment interactions; the inference latency from the ``p8.2-latency/1.0`` records. **Every number is
``{value, source}``**, a source being ``{file, sha256, json_path}``, ``{file, sha256, line}`` or ``{measurement}``.
Alignment convention: not applicable -- the artifact records no trajectory.

The row-spec vocabulary
-----------------------
A training entry is a mapping: ``record`` (a :data:`PINNED_RECORDS` key), ``tier``, ``label``, ``seeds`` (how each
seed's object is found: ``{"kind": "list", "path": <JSON path of a list>, "match": {field: value}}`` or ``{"kind":
"dict", "path": <JSON path of a mapping>, "key": "<template with {seed}>"}``), ``covers`` (what its wall time
measures) and ValueRefs for ``seconds``, ``steps``, ``batch``, ``data`` and the ``regime`` fields (``device``,
``gpu``, ``torch``, ``threads``, ``concurrency``). A ValueRef is ``{"kind": "seed", "field": <dotted field under the
seed's object>}`` (one value per seed), ``{"kind": "record", "path": <JSON path>}``, ``{"kind": "checkpoint",
"path": <JSON path into the checkpoint payload>}`` (read from every seed's checkpoint, equal across seeds), ``{"kind":
"log", ...}`` (a manifest-pinned log's lines) or ``{"kind": "absent", "key": <a DECLARED_ABSENCES key>}``; ``data``
adds ``unit``.

Refusals, all before the artifact is written: a record that is absent or at a digest other than the pinned one; a
row with no source for a column; a value missing from its record that is not a declared absence; a latency run
without ``COMPLETE``, with a throttled or non-reproducing canary, with a record outside its manifest or whose
statistics do not recompute from its own nanoseconds; route A and route B disagreeing; a row whose checkpoints differ
in size.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "FORMAT_VERSION",
    "MAPPO_RESULTS_PINNED_ON",
    "PinnedRecord",
    "PINNED_RECORDS",
    "SOURCES_ADDED_AFTER_PLAN",
    "Roots",
    "TableRow",
    "TABLE_ROWS",
    "DECLARED_ABSENCES",
    "sha256_file",
    "json_get",
    "read_pinned",
    "parameter_names",
    "count_state_parameters",
    "count_payload_parameters",
    "count_loaded_parameters",
    "row_checkpoints",
    "training_block",
    "verify_latency_run",
    "inference_block",
    "build_artifact",
    "write_artifact",
    "build_parser",
    "main",
]

FORMAT_VERSION = "p8.2-compute/1.0"

#: Q4: the date the MAPPO ``results.json`` digests were pinned (no committed artifact or manifest named them before).
MAPPO_RESULTS_PINNED_ON = "2026-10-03"


@dataclass(frozen=True)
class PinnedRecord:
    """A record the builder reads: its root (``repo`` / ``output`` / ``corpus``), path, pinned digest and who names it."""

    key: str
    root: str
    relpath: str
    sha256: str
    named_by: str


#: Every record of ``docs/plans/p8.2.md`` §4, at the digest computed on 2026-10-03.
PINNED_RECORDS: dict[str, PinnedRecord] = {}

#: Committed records read beyond the plan's §4, each with the reason (disclosed in the Return Packet).
SOURCES_ADDED_AFTER_PLAN: dict[str, str] = {}


@dataclass(frozen=True)
class Roots:
    """The four trees the builder reads: this checkout (``docs/data``), the main tree's ``output/``, the v1.1 corpus,
    and the latency run directory with its manifest."""

    repo_root: Path
    output_root: Path
    corpus_root: Path
    latency_dir: Path | None = None
    manifest_path: Path | None = None


@dataclass(frozen=True)
class TableRow:
    """One row of the table (Q3): an arm as the paper names it, its tiers inside.

    ``family`` selects the parameter counting (``dt``, ``spatial_dt``, ``bc``, ``iql``, ``mappo``, ``heuristic``);
    ``latency_row`` names the ``offline.compute_latency`` row whose records give the inference column (``None`` with
    ``latency_note`` for the rows that reuse another row's measurement, Q14).
    """

    row_id: str
    scenario: str
    method: str
    configuration: str
    family: str
    claims: tuple[Mapping[str, str], ...]
    groups: tuple[Mapping[str, Any], ...]
    training: tuple[Mapping[str, Any], ...]
    interactions: str
    latency_row: str | None
    latency_note: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


TABLE_ROWS: tuple[TableRow, ...] = ()

#: Q5: absence key -> the reason a value no record holds is written as ``null``. A row spec declares an absence as
#: ``{"kind": "absent", "key": <key>}``; a key not in this mapping refuses, and so does a value missing from its
#: record without such a declaration.
DECLARED_ABSENCES: dict[str, str] = {}


def sha256_file(path: Path) -> str:
    """The sha256 of the file's bytes."""
    raise NotImplementedError


def json_get(document: Any, path: str) -> Any:
    """Resolve a JSON path of the form ``$.a.b[3]['c-d']`` against *document*; refuses a path that does not resolve."""
    raise NotImplementedError


def read_pinned(key: str, roots: Roots) -> tuple[Any, dict[str, str]]:
    """Load a pinned record and return it with its ``{file, sha256}``; refuses an absent file or another digest."""
    raise NotImplementedError


def parameter_names(module: Any) -> set[str]:
    """The names of *module*'s parameters -- ``named_parameters()``, never its buffers."""
    raise NotImplementedError


def count_state_parameters(state: Mapping[str, Any], skeleton: Any) -> int:
    """Sum ``numel()`` over the entries of *state* whose names are *skeleton*'s parameters; refuses a parameter the
    state lacks. Buffers and every other tensor of *state* are excluded."""
    raise NotImplementedError


def count_payload_parameters(family: str, payload: Mapping[str, Any]) -> dict[str, int]:
    """Route B: sum ``numel()`` over the payload's tensors whose names are the module skeleton's parameters.

    The skeleton is built from the payload's own configuration; non-module tensors (an optimiser's moments, a running
    normaliser) and buffers are excluded by construction. Returns ``trained`` and ``deployed`` for every family, plus
    ``stored`` for IQL and ``executed_per_decision`` for MAPPO.
    """
    raise NotImplementedError


def count_loaded_parameters(family: str, path: Path, *, declared_gradient_steps: int | None,
                            method: str | None = None) -> dict[str, int]:
    """Route A: load *path* through the method's own loader on a node-order stub env and sum ``numel()`` over the
    loaded model's ``parameters()``; the same keys as :func:`count_payload_parameters`."""
    raise NotImplementedError


def row_checkpoints(row: TableRow, roots: Roots) -> list[dict[str, Any]]:
    """Every checkpoint of *row* (tier x seed): its path, the digest its named source holds and that source;
    refuses a file whose sha256 is not the named one."""
    raise NotImplementedError


def training_block(row: TableRow, entry: Mapping[str, Any], roots: Roots) -> dict[str, Any]:
    """One (record, tier) training entry: per-seed seconds and steps with their JSON paths, median [min-max], batch,
    data size in its record's unit, what the wall time covers and the regime; a declared absence becomes ``null``
    with its reason, an undeclared one refuses."""
    raise NotImplementedError


def verify_latency_run(latency_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Refuse unless the run is ``COMPLETE``, both canaries are at speed and reproduced, every file is in the manifest
    at its digest, and every record's statistics recompute from its own nanoseconds; return the run's summary."""
    raise NotImplementedError


def inference_block(row: TableRow, run: Mapping[str, Any]) -> dict[str, Any]:
    """The inference column of *row*: per device the median and p95 in ms per decision and per intersection, the
    timed count and the record's ``{file, sha256}``; "not applicable" where the row has no tensor computation."""
    raise NotImplementedError


def build_artifact(roots: Roots, *, git: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``p8.2-compute/1.0`` artifact; every refusal of the module docstring runs before anything is returned."""
    raise NotImplementedError


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write the artifact once (an existing file refuses), canonical JSON."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    """The CLI: ``build`` (the artifact) and ``params`` (the parameter counts alone, for the reviewer)."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns the process exit code."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
