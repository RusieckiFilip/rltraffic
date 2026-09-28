"""P5.3c (``BRIEF_42``; ``PREREGISTRATION`` A26 as corrected by A26.1): H4's context-length sweep.

Formats written by this module, each carrying its version in the payload:

* ``p5.3c-reference-rows/1.0`` -- ``docs/data/p4_k20_att_engine_rows.json`` (C1): the per-draw ``att_engine`` and
  ``att_ours`` of P4's five K = 20 checkpoints on the 100 held-out draws, extracted from P8.4b's gitignored cells with
  every source's digest.  A26(c)'s PRECONDITION: committed before any training, and the campaign's reference gate
  compares against it under ``==``.
* ``dt-checkpoint/1.0`` -- the sixty checkpoints (C2): EXACTLY the payload ``offline.dt_gate.train_dt`` writes, published
  once through an in-memory serialisation; its provenance carries this module's block, versioned
  ``p5.3c-sweep-provenance/1.0`` (subject, arm, run, K, batch, corpus digests, the code's commit, ``deterministic``).
* ``p5.3c-train-run/1.0`` -- ``output/p5_3c_training/runs/<run>.json``: one run's seconds, losses and targets per step.
* ``p5.3c-k20-reproduction/1.0`` -- ``output/p5_3c_training/k20_reproduction.json``: the K = 20 reproduction MEASUREMENT.
* ``p5.3c-train-record/1.0`` -- ``output/p5_3c_training/p5_3c_train.json``: the training record, which the coordinator
  commits as ``docs/data/p5_3c_train.json`` at gate G3, before the evaluation token (A26(b)).
* ``p5.3c-train-timing-slot/1.0`` and ``p5.3c-train-timing/1.0`` -- the fenced timing run's records, under
  ``output/p5_3c_training/fenced_timing/<UTC stamp>/``, a directory nothing registered is written to or read from.

Alignment convention.  Contract C6 v1.1 (``docs/CONTRACTS.md``), unchanged: observation rows ``T + 1``, decision and
outcome rows ``T``, and the reward of decision ``t`` is the one the env returned from step ``t``.  A training window is
the K decision steps ENDING at its step ``t``, left-padded (``offline/dataset.py:777-820``); nothing about a window is
redefined here.  The per-episode ATT definitions are A11(b)'s: ``att_engine`` (the engine's pool-clock average over
every vehicle the demand created -- PRIMARY on hz1x1 under A11's Rule R and A15) and ``att_ours`` (P4's
``att_horizon``, co-reported).

C1 -- THE REFERENCE ROWS
------------------------
``extract-reference-rows`` reads, and before writing anything verifies:

1. P4's five checkpoints ``<output>/p4_dt/dt_seed<s>.pt``, each by file sha256 against ``docs/data/p4_gate.json``;
2. P8.4b's campaign manifest ``<output>/p8_4b_rederivation/campaign_manifest.json``, which must declare all 500
   ``hz1x1|dt@mappo1000|<seed>|<draw>`` cells, and whose ``engine_seed`` is recorded;
3. the cell files named by the producer's own rule (``offline.att_rederivation.cell_file_name``): all 500 present, no
   other file of that pattern in the directory, no (seed, draw) claimed by two files, each file's CONTENT naming the
   pair its NAME says, ``arm dt@mappo1000``, ``tier mappo1000``, ``method dt``, ``scenario hz1x1``, P8.4b's format,
   ``reproduces_committed true``, a policy source that is P4's checkpoint of that seed, and both definitions finite.

Then ONE write: the file is written once -- the same content again is a no-op, other content is refused -- through
``offline.few_shot``'s exclusive link.  The command refuses a dirty code tree, because the committed file records the
commit that wrote it (plan F8; ``BRIEF_42`` Amendment A, Q8).

C2 -- THE SIXTY TRAININGS, EACH WRITTEN ONCE (A26(a)-(b))
---------------------------------------------------------
The table: per subject (``mappo1000`` confirmatory, ``mix50`` exploratory) K in {1, 2, 5, 10, 20} at batch 64, and on
``mappo1000`` the two equal-supervision arms (K = 1 at batch 1,280, K = 2 at batch 640), each over the seeds
101 ... 505 -- sixty runs, nothing else trainable by name.  ONE route builds every run's inputs, P4.6's and P4.7's
(``offline/method_tier_grid.py``): the subject's train-split dataset at context length K, its size-matched training
streams, the prompt and scale the CORPUS gives (``assert_declaration_matches_corpus`` refuses a declaration the corpus
does not bear out, and returns the values), the stacked windows filtered to those streams.  ``train_dt`` is called ONCE
per run, unchanged, with ``context_length`` K, the run's batch, exactly 40,000 declared steps and ``raise_to None``; it
writes a STAGED file, whose payload is read back, checked, and published through ``offline.few_shot``'s in-memory
exclusive write, so the published bytes depend on the payload alone.  The resume decision, the attempt markers, the
manifest and the record are Python's, never the driver's.  The K = 20 reproduction is a MEASUREMENT recorded tensor for
tensor, never a stop (A26(b), A26.1(a)).  The registered regime: CUDA, ``CUBLAS_WORKSPACE_CONFIG`` unset, deterministic
algorithms off, one torch thread, a clean code tree, and a working directory at the code's commit (plan Q14).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.DTAgent import CHECKPOINT_FORMAT_VERSION, DTConfig
from offline.att_rederivation import CellKey, cell_file_name
from offline.dt_gate import (
    BATCH_SIZE,
    GRAD_CLIP,
    HELD_OUT_DRAWS,
    LEARNING_RATE,
    TRAINING_SEEDS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    stack_dataset,
    train_dt,
)
from offline.few_shot import _load_weights_only, write_payload_exclusive
from offline.method_tier_grid import (
    TierSpec,
    _component_streams,
    assert_declaration_matches_corpus,
    canonical_digest_of,
    statistics_digest,
    tier_dataset,
    tier_dirs,
    tier_spec,
    training_streams,
)
from offline.offline_baselines import filter_stacked_to_streams

__all__ = [
    "ATT_DEFINITIONS",
    "CONTEXT_LENGTHS",
    "EQUAL_SUPERVISION",
    "GRADIENT_STEPS",
    "K20_FORMAT_VERSION",
    "PRIMARY_ATT",
    "REFERENCE_ROWS_FORMAT_VERSION",
    "REFERENCE_ROWS_NAME",
    "RUN_RECORD_FORMAT_VERSION",
    "RunSpec",
    "SUBJECTS",
    "SweepInputs",
    "TIMING_BUDGET",
    "TIMING_RUN",
    "TRAIN_RECORD_FORMAT_VERSION",
    "TrainOutcome",
    "assert_fence",
    "assert_writable",
    "attempts_of",
    "build_train_record",
    "check_inputs",
    "compare_models",
    "enter_registered_regime",
    "k20_reproduction",
    "main",
    "next_attempt",
    "reference_rows_payload",
    "registered_arms",
    "registered_runs",
    "replay_index_stream",
    "resume_decision",
    "run_by_name",
    "subject_facts",
    "subject_spec",
    "summarize_timing",
    "supervised_targets_per_step",
    "train_run",
    "training_inputs",
    "validate_checkpoint",
    "validate_run",
    "verify_reference_inputs",
    "write_k20_record",
    "write_manifest",
    "write_reference_rows",
    "write_run_record",
    "write_train_record",
]

# ======================================================================================================================
# C1 -- the reference rows
# ======================================================================================================================

REFERENCE_ROWS_FORMAT_VERSION = "p5.3c-reference-rows/1.0"
REFERENCE_ROWS_NAME = "p4_k20_att_engine_rows.json"

#: A11(b)'s two definitions, the primary first (A11 Rule R; A15).
ATT_DEFINITIONS: tuple[str, ...] = ("att_engine", "att_ours")
PRIMARY_ATT = "att_engine"

#: The subject of the reference rows: P4's K = 20 model on the P4 validation scenario (A26(a)).
REFERENCE_SUBJECT: dict[str, Any] = {
    "tier": "mappo1000",
    "arm": "dt@mappo1000",
    "method": "dt",
    "scenario": "hz1x1",
    "scenario_id": "cityflow1x1",
    "context_length": 20,
}

#: Where P8.4b left its cells, and what it wrote into each (``offline/att_rederivation.py:1288-1298``).
REDERIVATION_DIRNAME = "p8_4b_rederivation"
REDERIVATION_MANIFEST_NAME = "campaign_manifest.json"
REDERIVATION_FORMAT_VERSION = "p8.4b-rederivation/1.0"
REDERIVATION_ARTIFACT_NAME = "p8_4b_rederivation.json"
P4_GATE_NAME = "p4_gate.json"
P4_CHECKPOINT_DIRNAME = "p4_dt"

_REQUIRED_CELL_KEYS: tuple[str, ...] = (
    "arm", "tier", "method", "scenario", "seed", "draw_id", "format_version", "reproduces_committed",
    "policy_source", *ATT_DEFINITIONS,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{path} is absent: {what}")
    return json.loads(path.read_bytes())


def _git_in_module_tree(*args: str) -> subprocess.CompletedProcess[str]:
    """``git`` run in THIS module's directory, never the process CWD (a driver's CWD may be another tree)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args=["git", *args], returncode=127, stdout="", stderr=str(exc))


def _code_provenance() -> dict[str, Any]:
    """The module tree's commit and whether it is dirty -- STRICT (``BRIEF_37`` J1(a)): a git failure raises.

    "Could not say" must never read as "clean", so neither call's failure is folded into a default.
    """
    head = _git_in_module_tree("rev-parse", "HEAD")
    commit = head.stdout.strip()
    if head.returncode != 0 or len(commit) != 40:
        raise ValueError(
            f"git could not name the code's commit (exit {head.returncode}: {head.stderr.strip()[:160]!r}); a record "
            "that cannot say which code wrote it is not provenance"
        )
    status = _git_in_module_tree("status", "--porcelain")
    if status.returncode != 0:
        raise ValueError(
            f"git could not say whether the code tree is clean (exit {status.returncode}: "
            f"{status.stderr.strip()[:160]!r}); an unmeasured tree is never recorded as a clean one"
        )
    return {"code_commit": commit, "code_dirty": bool(status.stdout.strip())}


def _declared_cells() -> dict[tuple[int, int], str]:
    """The 500 (seed, draw) pairs and the file each is written to, by the PRODUCER's naming rule."""
    return {
        (int(seed), int(draw)): cell_file_name(
            CellKey(
                scenario=str(REFERENCE_SUBJECT["scenario"]),
                arm=str(REFERENCE_SUBJECT["arm"]),
                seed=int(seed),
                draw_id=int(draw),
            )
        )
        for seed in TRAINING_SEEDS
        for draw in HELD_OUT_DRAWS
    }


def _manifest_key(seed: int, draw: int) -> str:
    """``offline.att_rederivation.campaign_manifest``'s cell key, ``scenario|arm|seed|draw``."""
    return f"{REFERENCE_SUBJECT['scenario']}|{REFERENCE_SUBJECT['arm']}|{int(seed)}|{int(draw)}"


def _verified_checkpoints(output_root: Path, data_dir: Path) -> dict[str, dict[str, str]]:
    gate = _read_json(data_dir / P4_GATE_NAME, "P4's gate artifact names the five checkpoints' digests")
    recorded = gate.get("checkpoints") or {}
    checkpoints: dict[str, dict[str, str]] = {}
    for seed in TRAINING_SEEDS:
        entry = recorded.get(str(seed))
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"{data_dir / P4_GATE_NAME} records no sha256 for seed {seed}")
        path = output_root / P4_CHECKPOINT_DIRNAME / f"dt_seed{seed}.pt"
        if not path.is_file():
            raise ValueError(f"{path} is absent: P4's checkpoint of seed {seed} is the reference rows' subject")
        digest = _sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"{path}: sha256 {digest} is not {P4_GATE_NAME}'s {entry['sha256']}; the rows would describe another "
                "checkpoint than the one P4 published"
            )
        checkpoints[str(seed)] = {
            "path": f"output/{P4_CHECKPOINT_DIRNAME}/dt_seed{seed}.pt",
            "sha256": digest,
            "checked_against": f"docs/data/{P4_GATE_NAME} checkpoints.{seed}.sha256",
        }
    return checkpoints


def reference_rows_payload(*, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """The reference rows, every source verified first.  Writes NOTHING; refuses with ``ValueError``."""
    out = Path(output_root)
    data = Path(data_dir)
    checkpoints = _verified_checkpoints(out, data)

    cells_dir = out / REDERIVATION_DIRNAME
    manifest_path = cells_dir / REDERIVATION_MANIFEST_NAME
    manifest = _read_json(manifest_path, "P8.4b's campaign manifest declares the cells it rolled")
    declared_by_campaign = set(manifest.get("cells") or [])
    declared = _declared_cells()

    names = set(declared.values())
    pattern = f"cell_{REFERENCE_SUBJECT['scenario']}_dt_at_{REFERENCE_SUBJECT['tier']}_seed*_draw*.json"
    extra = sorted(path.name for path in cells_dir.glob(pattern) if path.name not in names)
    if extra:
        raise ValueError(
            f"{cells_dir / extra[0]} is not one of the 500 declared cells ({len(extra)} such file(s)); a file of this "
            "pattern outside the declared set means the directory is not the one P8.4b wrote"
        )
    unlisted = [pair for pair in sorted(declared) if _manifest_key(*pair) not in declared_by_campaign]
    if unlisted:
        raise ValueError(
            f"(seed, draw) {unlisted[0]} is not declared in P8.4b's campaign manifest {manifest_path} "
            f"({len(unlisted)} such pair(s)); the rows may only come from cells that campaign declared"
        )

    absent = [declared[pair] for pair in sorted(declared) if not (cells_dir / declared[pair]).is_file()]
    if absent:
        raise ValueError(
            f"{len(absent)} of the 500 declared cells are absent from {cells_dir} (first: {absent[:3]}); the rows "
            "are extracted whole or not at all"
        )

    contents: dict[tuple[int, int], tuple[dict[str, Any], str]] = {}
    for pair in sorted(declared):
        raw = (cells_dir / declared[pair]).read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{declared[pair]} is a {type(payload).__name__}, not a cell record")
        for key in _REQUIRED_CELL_KEYS:
            if key not in payload:
                raise ValueError(f"{declared[pair]} lacks {key!r}")
        contents[pair] = (payload, _sha256_bytes(raw))

    claims = Counter((int(payload["seed"]), int(payload["draw_id"])) for payload, _digest in contents.values())
    duplicated = sorted(pair for pair, count in claims.items() if count > 1)
    if duplicated:
        pair = duplicated[0]
        holders = sorted(
            declared[name_pair]
            for name_pair, (payload, _digest) in contents.items()
            if (int(payload["seed"]), int(payload["draw_id"])) == pair
        )
        raise ValueError(
            f"(seed, draw) {pair} is claimed by {claims[pair]} files ({holders}); each pair must come from exactly "
            "one cell"
        )

    expected_fields = {
        "arm": REFERENCE_SUBJECT["arm"],
        "tier": REFERENCE_SUBJECT["tier"],
        "method": REFERENCE_SUBJECT["method"],
        "scenario": REFERENCE_SUBJECT["scenario"],
        "format_version": REDERIVATION_FORMAT_VERSION,
    }
    policies: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for pair in sorted(declared):
        payload, digest = contents[pair]
        name = declared[pair]
        seed, draw = pair
        claimed = (int(payload["seed"]), int(payload["draw_id"]))
        if claimed != pair:
            raise ValueError(f"{name} names {claimed} but its file name says {pair}")
        for field, expected in expected_fields.items():
            if payload[field] != expected:
                raise ValueError(f"{name}: {field} {payload[field]!r}, not {expected!r}")
        if payload["reproduces_committed"] is not True:
            raise ValueError(
                f"{name}: reproduces_committed is {payload['reproduces_committed']!r}; P8.4b re-derived this cell "
                "without reproducing P4's committed att_ours"
            )
        policy = payload["policy_source"]
        checkpoint = str(policy.get("checkpoint", "")) if isinstance(policy, dict) else ""
        if not checkpoint.endswith(f"/{P4_CHECKPOINT_DIRNAME}/dt_seed{seed}.pt"):
            raise ValueError(f"{name}: its policy source {checkpoint!r} is not P4's checkpoint of seed {seed}")
        policies.add((str(policy.get("kind")), str(policy.get("detail"))))
        values: dict[str, float] = {}
        for definition in ATT_DEFINITIONS:
            value = payload[definition]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name}: {definition} {value!r} is not a finite number")
            values[definition] = float(value)
        rows.append(
            {
                "seed": seed,
                "draw_id": draw,
                "att_engine": values["att_engine"],
                "att_ours": values["att_ours"],
                "source": name,
                "source_sha256": digest,
            }
        )
    if len(policies) != 1:
        raise ValueError(f"the 500 cells record {len(policies)} different policy sources: {sorted(policies)[:3]}")
    kind, detail = next(iter(policies))

    artifact_path = data / REDERIVATION_ARTIFACT_NAME
    artifact = _read_json(artifact_path, "P8.4b's committed artifact records the campaign's provenance")
    campaign = ((artifact.get("provenance") or {}).get("code_provenance")) or {}

    return {
        "format_version": REFERENCE_ROWS_FORMAT_VERSION,
        "role": (
            "A26(c)'s precondition: the per-draw att_engine (and att_ours) of P4's five K = 20 checkpoints on the 100 "
            "held-out draws, the reference the campaign's gate compares its re-evaluation of the same checkpoints "
            "against under ==. Committed before any training (BRIEF_42 C1)."
        ),
        "registered_in": "PREREGISTRATION A26(c) as corrected by A26.1; BRIEF_42 C1 and Amendment A (Q8)",
        "definitions": list(ATT_DEFINITIONS),
        "primary_definition": PRIMARY_ATT,
        "definition_note": (
            "att_engine is the engine's pool-clock average over every vehicle the demand created (A11 Rule R, A15: "
            "primary on hz1x1); att_ours is P4's att_horizon, co-reported"
        ),
        "subject": dict(REFERENCE_SUBJECT),
        "seeds": [int(seed) for seed in TRAINING_SEEDS],
        "draw_ids": [int(draw) for draw in HELD_OUT_DRAWS],
        "n_rows": len(rows),
        "engine_seed": int(manifest["engine_seed"]),
        "checkpoints": checkpoints,
        "source": {
            "directory": f"output/{REDERIVATION_DIRNAME}",
            "file_name_rule": "offline.att_rederivation.cell_file_name",
            "producer": "python -m offline.att_rederivation run (P8.4b)",
            "campaign_manifest": f"output/{REDERIVATION_DIRNAME}/{REDERIVATION_MANIFEST_NAME}",
            "campaign_manifest_sha256": _sha256_file(manifest_path),
            "declared_cells_sha256": manifest.get("declared_cells_sha256"),
            "campaign_provenance": {
                "artifact": f"docs/data/{REDERIVATION_ARTIFACT_NAME}",
                "artifact_sha256": _sha256_file(artifact_path),
                "git_commit": campaign.get("git_commit"),
                "code_dirty": campaign.get("code_dirty"),
                "note": (
                    "the cells themselves record no commit; this is the commit P8.4b's committed artifact records for "
                    "the code that wrote its report"
                ),
            },
            "policy_source": {"kind": kind, "detail": detail},
            "evaluation_device": {
                "recorded_by_the_cells": False,
                "note": (
                    "P8.4b's runner left --device unset (offline/att_rederivation.py:725), which the platform resolves "
                    "to CUDA when available (agent/utils/utils.py:32-35); A26.1(b) registers the sweep's evaluation "
                    "on CUDA for that reason"
                ),
            },
        },
        "extraction": {
            "command": "python -P -m offline.context_sweep extract-reference-rows",
            "module": "offline/context_sweep.py",
            **_code_provenance(),
        },
        "rows": rows,
    }


def write_reference_rows(payload: dict[str, Any], destination: str | Path) -> str:
    """Write *payload* ONCE; return the file's sha256.  The same content again is a no-op, other content refused."""
    from offline.few_shot import _link_exclusive

    target = Path(destination)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists() or target.is_symlink():
        if target.is_file() and target.read_bytes() == data:
            return _sha256_bytes(data)
        raise ValueError(f"{target} already exists and differs; it is written once and never replaced")
    if not target.parent.is_dir():
        raise ValueError(f"{target.parent} does not exist; nothing is created here")
    return _link_exclusive(data, target)


# ======================================================================================================================
# C2 -- the sixty trainings of A26(a)-(b), each written once
# ======================================================================================================================

#: A26(a)-(b): the two subjects, the five context lengths, the budget, the seeds.  ``mappo1000`` is the confirmatory
#: subject; ``mix50`` the exploratory second one.
SUBJECTS: tuple[str, ...] = ("mappo1000", "mix50")
CONTEXT_LENGTHS: tuple[int, ...] = (1, 2, 5, 10, 20)
#: The order the runs train in within a subject: K = 20 first, so the reproduction measurement exists early (plan §9).
#: The order moves no number: every run is a function of its own (subject, K, batch, seed).
RUN_ORDER_K: tuple[int, ...] = (20, 10, 5, 2, 1)
REFERENCE_K = 20
#: P4's batch (``offline/dt_gate.py:128``), 64 windows per step.
BASE_BATCH = BATCH_SIZE
#: A26(b)'s equal-supervision secondary, ``mappo1000`` only: K x batch = 20 x 64 = 1,280 targets per step at most.
EQUAL_SUPERVISION: tuple[tuple[str, int, int], ...] = (("mappo1000", 1, 1280), ("mappo1000", 2, 640))
#: A26(a): FIXED at 40,000 gradient steps for every K; P4's raise rule is not re-run per K.
GRADIENT_STEPS = 40_000

#: G5-like fenced timing (``BRIEF_42`` C2; Amendment A Q13): ONE ``mappo1000`` K = 5 run at B = 400, seed 101, and a
#: same-seed repeat, alone.  Not a registered configuration; its checkpoints are never evaluated.
TIMING_BUDGET = 400
TIMING_SLOTS: tuple[str, ...] = ("alone", "repeat")

RUN_RECORD_FORMAT_VERSION = "p5.3c-train-run/1.0"
TRAIN_RECORD_FORMAT_VERSION = "p5.3c-train-record/1.0"
K20_FORMAT_VERSION = "p5.3c-k20-reproduction/1.0"
TIMING_SLOT_FORMAT_VERSION = "p5.3c-train-timing-slot/1.0"
TIMING_FORMAT_VERSION = "p5.3c-train-timing/1.0"
#: The identity of the provenance block this module adds to ``train_dt``'s payload (whose format stays
#: ``dt-checkpoint/1.0``, so ``DTAgent.load`` reads it unchanged).
SWEEP_PROVENANCE_FORMAT_VERSION = "p5.3c-sweep-provenance/1.0"

TRAINING_DIRNAME = "p5_3c_training"
CHECKPOINTS_DIRNAME = "checkpoints"
RUNS_DIRNAME = "runs"
ATTEMPTS_DIRNAME = "attempts"
STAGING_DIRNAME = "staging"
FENCED_TIMING_DIRNAME = "fenced_timing"
K20_RECORD_NAME = "k20_reproduction.json"
TRAIN_RECORD_NAME = "p5_3c_train.json"
TIMING_RECORD_NAME = "timing.json"
TRAIN_MANIFEST_NAME = "SHA256SUMS_p5_3c_train.txt"
RUNS_OUTPUT_DIRNAME = "p5_3c_runs"
#: The ONLY entries under the output root the training half writes: default-deny, as ``nortg_campaign`` learned
#: (a list of fenced directories goes stale; a list of this task's own does not).
TRAINING_OUTPUT_ENTRIES: tuple[str, ...] = (TRAINING_DIRNAME, RUNS_OUTPUT_DIRNAME, TRAIN_MANIFEST_NAME)

#: The published K = 20 checkpoints the reproduction is measured against, and what names each by digest.
P4_7_CHECKPOINT_TEMPLATE = "p4_7/checkpoints/mix50_dt_seed{seed}.pt"
P4_7_MANIFEST_NAME = "SHA256SUMS_p4_7.txt"
P4_7_TRAINING_NAME = "p4_7_training.json"
P4_TRAINING_NAME = "p4_training.json"

_STAMP = re.compile(r"\d{8}T\d{6}Z")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_SUMS_LINE = re.compile(r"([0-9a-f]{64})  (\S+)")

WARMUP_AT_BUDGET = min(WARMUP_STEPS, max(1, GRADIENT_STEPS // 2))


@dataclass(frozen=True)
class RunSpec:
    """One registered training: the subject, the context length K, the batch (windows per step) and the seed."""

    subject: str
    k: int
    batch: int
    seed: int

    @property
    def name(self) -> str:
        """``<subject>_k<K>_b<batch>_seed<seed>`` -- the checkpoint's file stem and the run's name everywhere."""
        return f"{self.subject}_k{self.k}_b{self.batch}_seed{self.seed}"

    @property
    def arm(self) -> str:
        """``<subject>_k<K>_b<batch>`` -- the five seeds of one arm share it."""
        return f"{self.subject}_k{self.k}_b{self.batch}"


TIMING_RUN = RunSpec(subject="mappo1000", k=5, batch=BASE_BATCH, seed=101)


def registered_batches(subject: str, k: int) -> tuple[int, ...]:
    """The batches A26(b) registers for (*subject*, *k*): 64 always, plus an equal-supervision batch where one exists."""
    extra = tuple(batch for owner, context, batch in EQUAL_SUPERVISION if owner == subject and context == k)
    return (BASE_BATCH, *extra)


def validate_run(run: RunSpec) -> RunSpec:
    """Refuse anything A26 does not register -- the subject, K, the batch, the seed -- naming which."""
    if run.subject not in SUBJECTS:
        raise ValueError(f"subject {run.subject!r} is not one of A26's {list(SUBJECTS)}; nothing else is trained")
    if isinstance(run.k, bool) or not isinstance(run.k, int) or run.k not in CONTEXT_LENGTHS:
        raise ValueError(f"K {run.k!r} is not one of A26's {list(CONTEXT_LENGTHS)}; no other context length is trained")
    allowed = registered_batches(run.subject, run.k)
    if isinstance(run.batch, bool) or not isinstance(run.batch, int) or run.batch not in allowed:
        raise ValueError(
            f"batch {run.batch!r} is not registered for {run.subject} at K {run.k} (A26(b) registers {list(allowed)})"
        )
    if isinstance(run.seed, bool) or not isinstance(run.seed, int) or run.seed not in TRAINING_SEEDS:
        raise ValueError(f"seed {run.seed!r} is not one of the registered {list(TRAINING_SEEDS)}")
    return run


def registered_runs() -> tuple[RunSpec, ...]:
    """A26(a)-(b)'s sixty runs, in training order: per subject K = 20, 10, 5, 2, 1 at batch 64, then (``mappo1000``)
    the two equal-supervision arms; the seeds innermost."""
    runs: list[RunSpec] = []
    for subject in SUBJECTS:
        for k in RUN_ORDER_K:
            runs.extend(RunSpec(subject, k, BASE_BATCH, seed) for seed in TRAINING_SEEDS)
        for owner, k, batch in EQUAL_SUPERVISION:
            if owner == subject:
                runs.extend(RunSpec(owner, k, batch, seed) for seed in TRAINING_SEEDS)
    return tuple(validate_run(run) for run in runs)


def registered_arms() -> tuple[str, ...]:
    """The twelve arms, in training order."""
    return tuple(dict.fromkeys(run.arm for run in registered_runs()))


def run_by_name(name: str) -> RunSpec:
    """The registered run called *name*; refuses anything else by name."""
    for run in registered_runs():
        if run.name == name:
            return run
    raise ValueError(f"{name!r} is not one of the 60 registered runs (A26(a)-(b)); nothing else is trained under a name")


# ----------------------------------------------------------------------------------------------------------------------
# The training route: ONE route for both subjects (plan §9, A3), P4's and P4.7's inputs exactly
# ----------------------------------------------------------------------------------------------------------------------


def subject_spec(subject: str) -> TierSpec:
    """The registered subject's ``TierSpec`` (``offline/method_tier_grid.py:202-294``), refusing any other."""
    if subject not in SUBJECTS:
        raise ValueError(f"subject {subject!r} is not one of A26's {list(SUBJECTS)}")
    return tier_spec(subject)


@dataclass(frozen=True)
class SweepInputs:
    """Everything ``train_dt`` is handed for one (subject, K): the training batch and the subject's corpus facts."""

    subject: str
    k: int
    group: tuple[int, int]
    batch: dict[str, torch.Tensor]
    stats: Any
    scenario_id: str
    target_rtg: float
    rtg_scale: float
    statistics_digest: str
    dataset_dirs: tuple[str, ...]
    manifest_sha256: dict[str, str]
    training_draw_ids: tuple[int, ...]
    n_streams: int
    n_windows: int
    stream_keys_sha256: str
    subsample: str
    max_timestep: int
    window_ids: tuple[tuple[str, str, str, int], ...]


def training_inputs(spec: TierSpec, k: int, corpus_root: str | Path) -> SweepInputs:
    """The training batch of *spec* at context length *k*, built the way P4.6 and P4.7 built theirs.

    ``tier_dataset`` (the train split, statistics fitted there -- for ``mix50`` on the six-directory union, as P4.7 did,
    plan F7) -> ``stack_dataset`` -> ``training_streams`` -> ``assert_declaration_matches_corpus`` (which REFUSES unless
    the corpus gives the declared prompt and scale, and RETURNS them) -> ``filter_stacked_to_streams`` (dataset order,
    ``item_index`` kept).  The prompt and scale handed on are the values the corpus check returns, never the typed ones.
    For ``mappo1000`` the batch equals P4's ``stack_dataset`` batch tensor for tensor (plan V5).
    """
    context = int(k)
    if isinstance(k, bool) or context != k or context not in CONTEXT_LENGTHS:
        raise ValueError(f"K {k!r} is not one of A26's {list(CONTEXT_LENGTHS)}")
    root = Path(corpus_root)
    dataset = tier_dataset(spec, root, context_length=context)
    stacked = stack_dataset(dataset)
    groups = sorted(dataset.groups)
    if len(groups) != 1:
        raise ValueError(f"{spec.tier}: the subject has {len(groups)} (state_dim, n_actions) groups; it must have one")
    components = _component_streams(spec, root, context_length=context) if spec.subsample == "mixture" else None
    selected = training_streams(spec, dataset, component_streams=components)
    declared = assert_declaration_matches_corpus(spec, selected)
    batch = filter_stacked_to_streams(dataset, stacked, selected)
    del stacked
    window_ids = []
    for index in batch["item_index"].tolist():
        meta = dataset.item_meta(int(index))
        window_ids.append((str(meta.dataset_dir), str(meta.episode_file), str(meta.ix_id), int(meta.t)))
    dirs = tier_dirs(spec, root)
    keys = [list(stream.key) for stream in selected]
    return SweepInputs(
        subject=str(spec.tier),
        k=context,
        group=(int(groups[0][0]), int(groups[0][1])),
        batch=batch,
        stats=dataset.stats,
        scenario_id=str(dataset.episode_records[0].scenario_id),
        target_rtg=float(declared["target_rtg"]),
        rtg_scale=float(declared["rtg_scale"]),
        statistics_digest=statistics_digest(dataset),
        dataset_dirs=tuple(str(directory) for directory in dirs),
        manifest_sha256={str(directory): _sha256_file(directory / "manifest.json") for directory in dirs},
        training_draw_ids=tuple(int(draw) for draw in dataset.stats.draw_ids),
        n_streams=len(selected),
        n_windows=int(batch["state"].shape[0]),
        stream_keys_sha256=_sha256_bytes(json.dumps(keys).encode("utf-8")),
        subsample=str(spec.subsample),
        max_timestep=int(batch["timestep"].max()),
        window_ids=tuple(window_ids),
    )


def subject_facts(inputs: SweepInputs) -> dict[str, Any]:
    """The K-INVARIANT facts of a subject: what every registered checkpoint of it must carry, K aside."""
    return {
        "subject": inputs.subject,
        "scenario_id": inputs.scenario_id,
        "state_dim": inputs.group[0],
        "n_actions": inputs.group[1],
        "max_ep_len": inputs.max_timestep + 1,
        "target_rtg": inputs.target_rtg,
        "rtg_scale": inputs.rtg_scale,
        "stats": inputs.stats.to_json_obj(),
        "statistics_digest": inputs.statistics_digest,
        "dataset_dirs": list(inputs.dataset_dirs),
        "corpus_manifest_sha256": dict(inputs.manifest_sha256),
        "training_draw_ids": list(inputs.training_draw_ids),
        "n_streams": inputs.n_streams,
        "n_windows": inputs.n_windows,
        "stream_keys_sha256": inputs.stream_keys_sha256,
    }


def replay_index_stream(seed: int, count: int, batch: int, steps: int) -> Iterator[np.ndarray]:
    """The rows ``train_dt`` draws, step by step: ``np.random.default_rng(seed).integers(0, count, size=batch)``
    (``offline/dt_gate.py:828-835``) -- the same generator and the same call, so the same draws.  Pinned against the
    stream ``train_dt`` itself draws by ``tests/test_p5_3c_training.py``."""
    generator = np.random.default_rng(int(seed))
    for _step in range(int(steps)):
        yield generator.integers(0, int(count), size=int(batch)).astype(np.int64)


def supervised_targets_per_step(attention_mask: torch.Tensor, *, seed: int, batch: int, steps: int) -> dict[str, Any]:
    """Non-PAD targets per step over the run's index stream (Amendment A, A1.3): ``attention_mask`` marks exactly the
    positions whose action is a target (``offline/dataset.py``: ``action == PAD_ACTION`` iff the mask is False)."""
    per_window = attention_mask.sum(dim=1).to(torch.int64).numpy()
    totals = np.fromiter(
        (int(per_window[rows].sum()) for rows in replay_index_stream(seed, per_window.shape[0], batch, steps)),
        dtype=np.int64,
        count=int(steps),
    )
    total = int(totals.sum())
    return {
        "mean": float(total) / int(steps),
        "min": int(totals.min()),
        "max": int(totals.max()),
        "total": total,
        "steps": int(steps),
        "batch": int(batch),
        "rule": "non-PAD positions of the windows each step draws, replayed from the seed's index stream",
    }


# ----------------------------------------------------------------------------------------------------------------------
# Provenance and the registered regime
# ----------------------------------------------------------------------------------------------------------------------


def _regime() -> dict[str, Any]:
    """The numerical regime this process is in: the state that decides float reduction order on this machine."""
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _cwd_commit() -> str:
    """``git rev-parse HEAD`` in the process's working directory -- the commit ``train_dt``'s ``runtime_provenance`` stamps
    (``offline/dt_gate.py:661-667``).  Empty when git cannot name one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(Path.cwd()), capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def enter_registered_regime() -> dict[str, Any]:
    """The registered regime (``BRIEF_42`` C2; A26(b)), checked at process entry, before any CUDA work; refusals first.

    CUDA on this GPU, ``CUBLAS_WORKSPACE_CONFIG`` UNSET (P5.2's and P7.3c's non-deterministic regime: a set variable
    means the process was not started through the driver, which unsets it), deterministic algorithms OFF (P4's regime),
    one torch thread, a CLEAN code tree (every checkpoint records the commit it was trained at, J1(d)), and a working
    directory at that same commit (plan Q14: ``train_dt`` stamps the CWD's commit into every payload).
    """
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is not None:
        raise ValueError(
            f"CUBLAS_WORKSPACE_CONFIG is set ({configured!r}); the registered regime unsets it (P5.2's and P7.3c's "
            "non-deterministic regime) -- start through the driver, which does"
        )
    code = _code_provenance()
    if code["code_dirty"]:
        raise ValueError(
            f"the code tree at {code['code_commit'][:12]} is dirty; every checkpoint records the commit it was trained "
            "at (J1(d)), so a registered run trains from a clean tree"
        )
    cwd = _cwd_commit()
    if cwd != code["code_commit"]:
        raise ValueError(
            f"the working directory's commit {cwd[:12] or '(none)'} is not the code's {code['code_commit'][:12]}; "
            "train_dt stamps runtime_provenance() from the working directory (offline/dt_gate.py:661-667), so the "
            "driver runs from the run worktree (plan Q14)"
        )
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("deterministic algorithms are ON; A26(b) registers P4's regime, with no such flag")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; the registered runs train on cuda only (A26(b))")
    torch.set_num_threads(1)
    return {"code": code, "regime": _regime()}


def run_provenance(run: RunSpec, inputs: SweepInputs, code: Mapping[str, Any]) -> dict[str, Any]:
    """The block this task adds to ``train_dt``'s provenance (which adds seed, steps, recipe, device and ``runtime``).
    No wall-clock value: two same-seed runs must be able to agree byte for byte."""
    return {
        "sweep_format_version": SWEEP_PROVENANCE_FORMAT_VERSION,
        "campaign": "p5.3c",
        "registered_in": "PREREGISTRATION A26(a)-(b) as corrected by A26.1; BRIEF_42 C2",
        "run": run.name,
        "arm": run.arm,
        "subject": run.subject,
        "tier": run.subject,
        "registered_context_length": int(run.k),
        "registered_batch_size": int(run.batch),
        "dataset_dirs": list(inputs.dataset_dirs),
        "corpus_manifest_sha256": dict(inputs.manifest_sha256),
        "training_draw_ids": list(inputs.training_draw_ids),
        "statistics_digest": inputs.statistics_digest,
        "subsample": inputs.subsample,
        "training_streams": int(inputs.n_streams),
        "training_windows": int(inputs.n_windows),
        "stream_keys_sha256": inputs.stream_keys_sha256,
        "target_rule": (
            "target_rtg = the largest episode return over the training set, rtg_scale = the largest absolute one "
            "(offline.method_tier_grid.assert_declaration_matches_corpus, re-derived from the corpus)"
        ),
        "code_commit": str(code["code_commit"]),
        "code_dirty": bool(code["code_dirty"]),
        "deterministic": False,
        "regime": _regime(),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Layout, and the two fences
# ----------------------------------------------------------------------------------------------------------------------


def training_root(output_root: str | Path) -> Path:
    return Path(output_root) / TRAINING_DIRNAME


def checkpoints_dir(output_root: str | Path) -> Path:
    return training_root(output_root) / CHECKPOINTS_DIRNAME


def runs_dir(output_root: str | Path) -> Path:
    return training_root(output_root) / RUNS_DIRNAME


def attempts_dir(output_root: str | Path) -> Path:
    return training_root(output_root) / ATTEMPTS_DIRNAME


def staging_dir(output_root: str | Path) -> Path:
    return training_root(output_root) / STAGING_DIRNAME


def registered_destination(output_root: str | Path, run: RunSpec) -> Path:
    """``p5_3c_training/checkpoints/<name>.pt``: where a registered run is written, once."""
    return checkpoints_dir(output_root) / f"{run.name}.pt"


def run_record_path(output_root: str | Path, run: RunSpec) -> Path:
    return runs_dir(output_root) / f"{run.name}.json"


def k20_record_path(output_root: str | Path) -> Path:
    return training_root(output_root) / K20_RECORD_NAME


def train_record_path(output_root: str | Path) -> Path:
    return training_root(output_root) / TRAIN_RECORD_NAME


def train_manifest_path(output_root: str | Path) -> Path:
    return Path(output_root) / TRAIN_MANIFEST_NAME


def timing_destination(output_root: str | Path, stamp: str, slot: str) -> Path:
    """A fenced timing checkpoint: ``p5_3c_training/fenced_timing/<UTC stamp>/<slot>.pt``."""
    if _STAMP.fullmatch(str(stamp)) is None:
        raise ValueError(f"timing stamp {stamp!r} is not a UTC stamp like 20260928T230000Z")
    if slot not in TIMING_SLOTS:
        raise ValueError(f"timing slot {slot!r} is not one of {list(TIMING_SLOTS)}")
    return training_root(output_root) / FENCED_TIMING_DIRNAME / str(stamp) / f"{slot}.pt"


def assert_writable(
    path: str | Path, output_root: str | Path, *, entries: Sequence[str] = TRAINING_OUTPUT_ENTRIES
) -> Path:
    """DEFAULT-DENY: *path* must lie under *output_root* in one of *entries* (``nortg_campaign.assert_writable``'s rule).
    Whole path components are compared, never a string prefix; nothing is created."""
    target = Path(path)
    root = Path(output_root).resolve()
    try:
        relative = target.resolve().relative_to(root)
    except ValueError:
        raise ValueError(f"{target} is outside the output root {root}; nothing is written there") from None
    head = relative.parts[0] if relative.parts else ""
    if head not in entries:
        raise ValueError(
            f"{target}: output/{head} belongs to another campaign and is read-only here; this step writes only "
            f"{list(entries)} (default-deny, so a directory added later is protected without being named)"
        )
    return target


def assert_fence(path: str | Path, *, timing: bool) -> Path:
    """Refuse a TIMING checkpoint outside ``fenced_timing/`` and a REGISTERED one inside it."""
    candidate = Path(path)
    fenced = FENCED_TIMING_DIRNAME in candidate.parts
    if timing and not fenced:
        raise ValueError(f"{candidate} is not under {FENCED_TIMING_DIRNAME}/: a timing run's files never leave it")
    if not timing and fenced:
        raise ValueError(
            f"{candidate} lies under {FENCED_TIMING_DIRNAME}/, which holds the unregistered timing run only; a "
            "registered checkpoint is never written there and nothing there is ever evaluated"
        )
    return candidate


def _write_text_once(text: str, destination: Path, output_root: str | Path) -> str:
    """Write *text* ONCE (fenced, exclusive); the same text again is a no-op, other text refused.  Returns its sha256."""
    from offline.few_shot import _link_exclusive

    assert_writable(destination, output_root)
    data = text.encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and destination.read_bytes() == data:
            return _sha256_bytes(data)
        raise ValueError(f"{destination} already exists and differs; it is never rewritten -- a person moves it aside")
    if not destination.parent.is_dir():
        raise ValueError(f"{destination.parent} does not exist; nothing is created here")
    return _link_exclusive(data, destination)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------------------------------
# One training, written once
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainOutcome:
    """One written checkpoint: where, its sha256, and what ``train_dt``'s loop did."""

    destination: Path
    sha256: str
    steps: int
    warmup: int
    losses: tuple[float, ...]
    window_means: tuple[float, ...]
    seconds: float


def _check_staged(payload: Mapping[str, Any], run: RunSpec, inputs: SweepInputs, steps: int) -> None:
    """The staged payload carries the registered recipe, or nothing is written."""
    config = dict(payload.get("config") or {})
    provenance = dict(payload.get("provenance") or {})
    problems = []
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        problems.append("format_version")
    if config.get("context_length") != run.k or provenance.get("context_length") != run.k:
        problems.append("context_length")
    if config.get("rtg_mode") != "conditioned":
        problems.append("rtg_mode")
    if provenance.get("batch_size") != run.batch:
        problems.append("batch_size")
    if provenance.get("gradient_steps") != steps or provenance.get("declared_gradient_steps") != steps:
        problems.append("gradient_steps")
    if "raise_to" not in provenance or provenance.get("raise_to") is not None:
        problems.append("raise_to")
    if payload.get("target_rtg") != inputs.target_rtg or payload.get("rtg_scale") != inputs.rtg_scale:
        problems.append("prompt")
    if provenance.get("run") != run.name or provenance.get("seed") != run.seed:
        problems.append("run")
    if problems:
        raise ValueError(f"the staged payload of {run.name} does not carry the registered recipe {problems}; nothing written")


def train_run(
    run: RunSpec,
    inputs: SweepInputs,
    *,
    device: str,
    destination: str | Path,
    staging_dir: str | Path,
    code: Mapping[str, Any],
    budget: int = GRADIENT_STEPS,
    log_every: int = 0,
) -> TrainOutcome:
    """ONE call of ``train_dt`` with the registered recipe, then ONE exclusive write of its payload.

    Validation first (the run, the inputs' subject and K, the destination absent, its directory and the staging
    directory present).  ``train_dt`` (``offline/dt_gate.py:747-910``, unchanged) writes to a STAGED path it is given;
    the staged payload is read back weights-only, checked, and published through ``offline.few_shot``'s
    ``write_payload_exclusive`` -- ``torch.save`` into memory, then ``os.link`` refusing an existing file -- so the
    published bytes are a function of the payload alone, never of a file name (``few_shot.py:1018-1031``).  The staged
    file is removed on every path.  ``budget`` is 40,000 for every registered run; only the fenced timing run and the
    tests pass another.
    """
    validate_run(run)
    if inputs.subject != run.subject or inputs.k != run.k:
        raise ValueError(
            f"the inputs are {inputs.subject} at K {inputs.k}, not {run.subject} at K {run.k}; a run trains on its own"
        )
    steps = int(budget)
    if steps < 1:
        raise ValueError(f"budget {budget!r} is not a positive step count")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ValueError(f"{target} already exists; a checkpoint is written once (A26(b)) -- a person moves it aside")
    if not target.parent.is_dir():
        raise ValueError(f"{target.parent} does not exist; nothing is created here")
    staging = Path(staging_dir)
    if not staging.is_dir():
        raise ValueError(f"{staging} does not exist; nothing is created here")
    staged = staging / f".{target.stem}.{os.getpid()}.staging.pt"
    if staged.exists():
        raise ValueError(f"{staged} already exists; a staged file is never reused")
    provenance = run_provenance(run, inputs, code)
    try:
        result = train_dt(
            inputs.batch,
            state_dim=inputs.group[0],
            n_actions=inputs.group[1],
            seed=int(run.seed),
            declared_gradient_steps=steps,
            raise_to=None,
            context_length=int(run.k),
            batch_size=int(run.batch),
            device=torch.device(device),
            checkpoint_path=staged,
            stats=inputs.stats,
            scenario_id=inputs.scenario_id,
            target_rtg=float(inputs.target_rtg),
            rtg_scale=float(inputs.rtg_scale),
            provenance=provenance,
            log_every=int(log_every),
        )
        if int(result.gradient_steps) != steps or len(result.losses) != steps:
            raise ValueError(f"train_dt ran {len(result.losses)} steps, not the declared {steps}; nothing written")
        payload = _load_weights_only(staged)
        _check_staged(payload, run, inputs, steps)
        digest = write_payload_exclusive(payload, target)
    finally:
        staged.unlink(missing_ok=True)
    return TrainOutcome(
        destination=target,
        sha256=digest,
        steps=steps,
        warmup=int(payload["provenance"]["warmup_steps"]),
        losses=tuple(float(value) for value in result.losses),
        window_means=tuple(float(value) for value in result.window_means),
        seconds=float(result.seconds),
    )


def write_run_record(
    run: RunSpec,
    outcome: TrainOutcome,
    inputs: SweepInputs,
    *,
    output_root: str | Path,
    code: Mapping[str, Any],
    device: str,
) -> Path:
    """The run's seconds, losses and targets, beside the checkpoints -- written once, like the checkpoint.

    ``loss_per_supervised_target`` is ``train_dt``'s own 2,000-step window means of its per-step losses, each of which is
    the mean cross-entropy over that step's non-PAD targets (Amendment A, A1.3); the full curve is kept in ``losses``.
    """
    path = run_record_path(output_root, run)
    assert_writable(path, output_root)
    if path.exists():
        raise ValueError(f"{path} already exists; a run record is written once, with its checkpoint")
    if not path.parent.is_dir():
        raise ValueError(f"{path.parent} does not exist; the driver creates it after the token")
    if len(outcome.losses) != outcome.steps:
        raise ValueError(f"{run.name}: {len(outcome.losses)} losses for {outcome.steps} steps")
    digest = _sha256_file(outcome.destination)
    if digest != outcome.sha256:
        raise ValueError(f"{outcome.destination}: sha256 {digest} is not the written {outcome.sha256}")
    try:
        checkpoint = str(Path(outcome.destination).resolve().relative_to(Path(output_root).resolve()))
    except ValueError:
        checkpoint = str(outcome.destination)
    record = {
        "format_version": RUN_RECORD_FORMAT_VERSION,
        "run": run.name,
        "arm": run.arm,
        "subject": run.subject,
        "k": int(run.k),
        "batch": int(run.batch),
        "seed": int(run.seed),
        "steps": int(outcome.steps),
        "warmup_steps": int(outcome.warmup),
        "checkpoint": checkpoint,
        "checkpoint_sha256": outcome.sha256,
        "weights_sha256": canonical_digest_of(outcome.destination),
        "loop_seconds": float(outcome.seconds),
        "ms_per_step": float(outcome.seconds) / int(outcome.steps) * 1000.0,
        "final_loss": float(outcome.losses[-1]),
        "loss_per_supervised_target": [float(value) for value in outcome.window_means],
        "loss_per_supervised_target_rule": (
            "train_dt's window means of its per-step loss, the mean cross-entropy over the step's non-PAD targets "
            "(agent/DTAgent.py action_loss, ignore_index=PAD_ACTION)"
        ),
        "supervised_targets_per_step": supervised_targets_per_step(
            inputs.batch["attention_mask"], seed=run.seed, batch=run.batch, steps=outcome.steps
        ),
        "losses": [float(value) for value in outcome.losses],
        "device": str(device),
        "code_commit": str(code["code_commit"]),
        "written_utc": _utc_now(),
    }
    _write_text_once(json.dumps(record, indent=2, sort_keys=True) + "\n", path, output_root)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Written once, resumed in Python: validation, the decision, the attempt markers, the manifest
# ----------------------------------------------------------------------------------------------------------------------


def validate_checkpoint(run: RunSpec, *, output_root: str | Path, facts: Mapping[str, Any]) -> dict[str, bool]:
    """Every check a written checkpoint of *run* must pass, as named booleans; an unreadable file is refused."""
    path = assert_fence(registered_destination(output_root, run), timing=False)
    try:
        payload = _load_weights_only(path)
    except Exception as exc:  # any failure to read is the same finding: the file is not a checkpoint of this run
        raise ValueError(
            f"{path} exists but cannot be read ({type(exc).__name__}); it is never overwritten -- a person moves it aside"
        ) from exc
    config = dict(payload.get("config") or {})
    provenance = dict(payload.get("provenance") or {})
    expected_config = DTConfig(
        state_dim=int(facts["state_dim"]),
        n_actions=int(facts["n_actions"]),
        context_length=int(run.k),
        max_ep_len=int(facts["max_ep_len"]),
    ).to_json_obj()
    recipe = (
        provenance.get("learning_rate"), provenance.get("weight_decay"), provenance.get("grad_clip"),
        provenance.get("warmup_steps"),
    )
    commit = provenance.get("code_commit")
    return {
        "format_version": payload.get("format_version") == CHECKPOINT_FORMAT_VERSION,
        "context_length": config.get("context_length") == run.k and provenance.get("context_length") == run.k,
        "config": config == expected_config,
        "batch": provenance.get("batch_size") == run.batch,
        "budget": provenance.get("gradient_steps") == GRADIENT_STEPS
        and provenance.get("declared_gradient_steps") == GRADIENT_STEPS,
        "no_raise": "raise_to" in provenance and provenance.get("raise_to") is None,
        "recipe": recipe == (LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP, WARMUP_AT_BUDGET),
        "seed": provenance.get("seed") == run.seed,
        "names": (provenance.get("run"), provenance.get("arm"), provenance.get("subject"))
        == (run.name, run.arm, run.subject),
        "prompt": payload.get("target_rtg") == facts["target_rtg"] and payload.get("rtg_scale") == facts["rtg_scale"],
        "stats": payload.get("stats") == facts["stats"],
        "normalise": payload.get("normalise") is True,
        "scenario_id": payload.get("scenario_id") == facts["scenario_id"],
        "device": provenance.get("device") == "cuda",
        "deterministic": provenance.get("deterministic") is False,
        "code_commit": isinstance(commit, str) and _HEX40.fullmatch(commit) is not None,
        "sweep_format": provenance.get("sweep_format_version") == SWEEP_PROVENANCE_FORMAT_VERSION,
    }


def resume_decision(run: RunSpec, *, output_root: str | Path, facts: Mapping[str, Any]) -> str:
    """``"train"`` if *run*'s checkpoint is absent, ``"skip"`` if it exists AND validates; refused otherwise.

    Decided here, never by a ``[ -f ]`` in the driver: a checkpoint that exists and does not validate is never overwritten
    and never re-trained -- a run is re-trained only when its checkpoint was never written (A26(b): run once).
    """
    path = registered_destination(output_root, run)
    if not path.exists() and not path.is_symlink():
        return "train"
    checks = validate_checkpoint(run, output_root=output_root, facts=facts)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{path} exists but does not validate (failed: {failed}); it is never overwritten -- a person moves it aside"
        )
    return "skip"


def attempts_of(run: RunSpec, *, output_root: str | Path) -> tuple[int, ...]:
    """The attempt numbers recorded on disk for *run* (``attempts/<name>.<n>``), in order."""
    directory = attempts_dir(output_root)
    if not directory.is_dir():
        return ()
    prefix = f"{run.name}."
    numbers = [
        int(path.name[len(prefix):])
        for path in directory.iterdir()
        if path.name.startswith(prefix) and path.name[len(prefix):].isdigit()
    ]
    return tuple(sorted(numbers))


def next_attempt(run: RunSpec, *, output_root: str | Path) -> int:
    """Write the next attempt marker of *run*, exclusively, BEFORE its training starts; return its number.

    A marker with no checkpoint after it is a re-run of an infrastructure failure, counted in the record.
    """
    directory = attempts_dir(output_root)
    if not directory.is_dir():
        raise ValueError(f"{directory} does not exist; the driver creates it after the token")
    done = attempts_of(run, output_root=output_root)
    number = (done[-1] if done else 0) + 1
    text = f"attempt {number} of {run.name}, started {_utc_now()}\n"
    _write_text_once(text, directory / f"{run.name}.{number}", output_root)
    return number


def _manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _SUMS_LINE.fullmatch(line)
        if match is None or match.group(2) in entries:
            raise ValueError(f"{path}:{number}: not one sha256sum line: {line!r}")
        entries[match.group(2)] = match.group(1)
    return entries


def write_manifest(output_root: str | Path, *, runs: Sequence[RunSpec] | None = None) -> Path:
    """``SHA256SUMS_p5_3c_train.txt``: every declared checkpoint by digest, ordered by path, written once, re-verified.

    Refuses a stray entry in the checkpoints directory, an absent checkpoint, and an existing manifest with other
    content; the same content again is a no-op, so a restarted driver can reach the record.
    """
    root = Path(output_root)
    declared = list(registered_runs() if runs is None else runs)
    directory = checkpoints_dir(root)
    wanted = {f"{run.name}.pt" for run in declared}
    if directory.is_dir():
        stray = sorted(path.name for path in directory.iterdir() if path.name not in wanted)
        if stray:
            raise ValueError(
                f"{directory / stray[0]} is not one of the declared runs ({len(stray)} such entr"
                f"{'y' if len(stray) == 1 else 'ies'}); the manifest lists exactly the declared checkpoints"
            )
    lines: list[tuple[str, str]] = []
    for run in declared:
        path = registered_destination(root, run)
        if not path.is_file():
            raise ValueError(f"{path} is absent: every declared run needs its checkpoint before the manifest")
        lines.append((str(path.relative_to(root)), _sha256_file(path)))
    text = "".join(f"{digest}  {relative}\n" for relative, digest in sorted(lines))
    manifest = train_manifest_path(root)
    _write_text_once(text, manifest, root)
    for relative, digest in _manifest_entries(manifest).items():
        if _sha256_file(root / relative) != digest:
            raise ValueError(f"{root / relative} does not match {manifest}")
    return manifest


# ----------------------------------------------------------------------------------------------------------------------
# The K = 20 reproduction: a MEASUREMENT (A26(b); A26.1(a)), recorded, never a stop
# ----------------------------------------------------------------------------------------------------------------------


def compare_models(candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Tensor for tensor: ``torch.equal`` per parameter and the largest absolute difference in float64.

    A key present on one side only, a shape or dtype that differs -- each is RECORDED; nothing here raises on a
    difference, because A26(b) registers the reproduction as a measurement whose difference is reported, not a gate.
    """
    candidate_keys, reference_keys = set(candidate), set(reference)
    missing = sorted(reference_keys - candidate_keys)
    extra = sorted(candidate_keys - reference_keys)
    parameters: list[dict[str, Any]] = []
    for name in sorted(reference_keys & candidate_keys):
        left = candidate[name].detach().cpu()
        right = reference[name].detach().cpu()
        comparable = tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
        if comparable:
            difference = float((left.double() - right.double()).abs().max()) if right.numel() else 0.0
        else:
            difference = None
        parameters.append(
            {
                "name": name,
                "shape": list(right.shape),
                "dtype": str(right.dtype),
                "shape_and_dtype_equal": comparable,
                "equal": bool(comparable and torch.equal(left, right)),
                "max_abs_difference": difference,
            }
        )
    differing = [entry for entry in parameters if not entry["equal"]]
    largest = None
    if differing:
        largest = max(
            differing,
            key=lambda entry: math.inf if entry["max_abs_difference"] is None else entry["max_abs_difference"],
        )
    return {
        "keys_equal": not missing and not extra,
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
        "n_parameters": len(parameters),
        "n_parameters_differing": len(differing),
        "largest_abs_difference": 0.0 if largest is None else largest["max_abs_difference"],
        "largest_abs_difference_parameter": None if largest is None else largest["name"],
        "all_equal": not missing and not extra and not differing,
        "parameters": parameters,
    }


def _published_k20(output_root: Path, data_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    """The ten published K = 20 checkpoints, each verified by digest before it is compared with anything."""
    gate = (_read_json(data_dir / P4_GATE_NAME, "P4's gate artifact names its checkpoints' digests").get("checkpoints")
            or {})
    sums_path = output_root / P4_7_MANIFEST_NAME
    if not sums_path.is_file():
        raise ValueError(f"{sums_path} is absent: P4.7's manifest names its checkpoints' digests")
    sums = _manifest_entries(sums_path)
    p47_runs = _read_json(data_dir / P4_7_TRAINING_NAME, "P4.7's training record names its canonical digests").get("runs")
    canonical = {
        int(run["seed"]): str(run["canonical_digest"])
        for run in (p47_runs or [])
        if run.get("tier") == "mix50" and run.get("method") == "dt"
    }
    published: dict[tuple[str, int], dict[str, str]] = {}
    for seed in TRAINING_SEEDS:
        relative = f"{P4_CHECKPOINT_DIRNAME}/dt_seed{seed}.pt"
        expected = (gate.get(str(seed)) or {}).get("sha256")
        path = output_root / relative
        if not path.is_file() or not expected or _sha256_file(path) != expected:
            raise ValueError(f"{path} is absent or not at {P4_GATE_NAME}'s sha256 {expected}; only P4's published "
                             "checkpoint is compared")
        published[("mappo1000", seed)] = {"reference": relative, "sha256": expected,
                                          "checked_against": f"docs/data/{P4_GATE_NAME}"}
        relative = P4_7_CHECKPOINT_TEMPLATE.format(seed=seed)
        path = output_root / relative
        expected = sums.get(relative)
        if not path.is_file() or not expected or _sha256_file(path) != expected:
            raise ValueError(f"{path} is absent or not at {P4_7_MANIFEST_NAME}'s sha256 {expected}")
        if canonical.get(seed) != canonical_digest_of(path):
            raise ValueError(f"{path}: its weights are not {P4_7_TRAINING_NAME}'s canonical digest {canonical.get(seed)}")
        published[("mix50", seed)] = {"reference": relative, "sha256": expected,
                                      "checked_against": f"output/{P4_7_MANIFEST_NAME}; docs/data/{P4_7_TRAINING_NAME}"}
    return published


def k20_reproduction(output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Each subject's five K = 20 runs against the published checkpoint of their seed (P4's; P4.7's for ``mix50``)."""
    root = Path(output_root)
    published = _published_k20(root, Path(data_dir))
    subjects: dict[str, Any] = {}
    for subject in SUBJECTS:
        seeds: list[dict[str, Any]] = []
        for seed in TRAINING_SEEDS:
            run = RunSpec(subject, REFERENCE_K, BASE_BATCH, seed)
            candidate = registered_destination(root, run)
            record = _read_json(run_record_path(root, run), f"the run record of {run.name} names its checkpoint's digest")
            if not candidate.is_file():
                raise ValueError(f"{candidate} is absent: the K = 20 run of seed {seed} has not been written")
            candidate_sha = _sha256_file(candidate)
            if candidate_sha != record.get("checkpoint_sha256"):
                raise ValueError(f"{candidate}: sha256 {candidate_sha} is not its run record's")
            source = published[(subject, seed)]
            reference = root / source["reference"]
            comparison = compare_models(
                _load_weights_only(candidate)["model"], _load_weights_only(reference)["model"]
            )
            candidate_weights = canonical_digest_of(candidate)
            reference_weights = canonical_digest_of(reference)
            seeds.append(
                {
                    "seed": seed,
                    "candidate": str(candidate.relative_to(root)),
                    "candidate_sha256": candidate_sha,
                    "reference": source["reference"],
                    "reference_sha256": source["sha256"],
                    "reference_checked_against": source["checked_against"],
                    "candidate_weights_sha256": candidate_weights,
                    "reference_weights_sha256": reference_weights,
                    "weights_digests_equal": candidate_weights == reference_weights,
                    **comparison,
                }
            )
        subjects[subject] = {
            "all_equal": all(entry["all_equal"] for entry in seeds),
            "n_seeds_equal": sum(1 for entry in seeds if entry["all_equal"]),
            "seeds": seeds,
        }
    return {
        "format_version": K20_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A26(b) as corrected by A26.1(a); BRIEF_42 C2",
        "rule": "torch.equal per parameter; the largest absolute difference per parameter in float64",
        "never_a_stop": (
            "a difference is recorded with its magnitude and stops nothing: the sweep's own K = 20 arm is the sweep's "
            "reference, P4's checkpoints remain C1's published subject (A26(b))"
        ),
        "subjects": subjects,
    }


def write_k20_record(output_root: str | Path, record: Mapping[str, Any]) -> Path:
    path = k20_record_path(output_root)
    _write_text_once(json.dumps(record, indent=2, sort_keys=True) + "\n", path, output_root)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# The fenced timing run, and the training record
# ----------------------------------------------------------------------------------------------------------------------


def summarize_timing(stamp_dir: str | Path) -> dict[str, Any]:
    """The fenced timing's summary: ms per step and the device peak per slot, and the same-seed repeat compared by TWO
    routes (the file's sha256 and the weights-only digest).  C = 1, fixed (``BRIEF_42`` C2; Amendment A Q13)."""
    stamp = Path(stamp_dir)
    assert_fence(stamp / TIMING_RECORD_NAME, timing=True)
    slots: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name in TIMING_SLOTS:
        record = _read_json(stamp / f"{name}.json", f"timing slot {name}'s record")
        if record.get("format_version") != TIMING_SLOT_FORMAT_VERSION:
            raise ValueError(f"{stamp / f'{name}.json'}: format {record.get('format_version')!r}")
        samples_path = stamp / f"nvidia_smi_{name}.csv"
        if not samples_path.is_file():
            raise ValueError(f"{samples_path} is absent: the driver samples the device during every slot")
        samples = [float(value) for value in samples_path.read_text(encoding="utf-8").split()]
        if not samples:
            raise ValueError(f"{samples_path} holds no sample")
        records[name] = record
        slots[name] = {
            "steps": int(record["steps"]),
            "loop_seconds": float(record["loop_seconds"]),
            "ms_per_step": float(record["loop_seconds"]) / int(record["steps"]) * 1000.0,
            "build_seconds": float(record["build_seconds"]),
            "peak_allocated_mib": float(record["peak_allocated_mib"]),
            "device_peak_mib": max(samples),
            "checkpoint_sha256": str(record["checkpoint_sha256"]),
            "weights_sha256": str(record["weights_sha256"]),
        }
    return {
        "format_version": TIMING_FORMAT_VERSION,
        "stamp": stamp.name,
        "run": TIMING_RUN.name,
        "budget": TIMING_BUDGET,
        "slots": slots,
        "repeat": {
            "file_sha256_equal": slots["repeat"]["checkpoint_sha256"] == slots["alone"]["checkpoint_sha256"],
            "weights_sha256_equal": slots["repeat"]["weights_sha256"] == slots["alone"]["weights_sha256"],
        },
        "concurrency": 1,
        "concurrency_rule": "C = 1, fixed: the sixty runs train one at a time (BRIEF_42 C2; Amendment A Q13)",
        "device_name": records["alone"].get("device_name"),
        "code_commit": records["alone"].get("code_commit"),
    }


_TRAIN_RECORD_LIMITS: tuple[str, ...] = (
    "A difference in the K = 20 reproduction is recorded with its magnitude and stops nothing (A26(b)); the sweep's own "
    "K = 20 arm is the sweep's reference either way.",
    "The losses are training losses per supervised target (the mean cross-entropy over a step's non-PAD targets). They "
    "say nothing about held-out performance, and no checkpoint was selected by them: the payload after exactly 40,000 "
    "steps is the only one written.",
    "The loss per supervised target is comparable across K; the TOTAL supervision per step is not (64 targets per step "
    "at K = 1 against up to 1,280 at K = 20), which is what the equal-supervision arms exist to measure (A26(b)).",
    "CUDA training here carries no determinism flag (A26(b): P4's regime); a same-seed repeat's equality, where measured, "
    "is a property of this GPU and this torch build.",
)


def build_train_record(
    output_root: str | Path,
    *,
    corpus_root: str | Path,
    timing_path: str | Path,
    runs: Sequence[RunSpec] | None = None,
    specs: Mapping[str, TierSpec] | None = None,
) -> dict[str, Any]:
    """The training record: per run the fields ``BRIEF_42`` C2 lists, its checks and attempts; the subjects' corpus
    facts; the K = 20 record, the manifest and the timing record by digest.  Refuses rather than record an invalid or
    partial set."""
    root = Path(output_root)
    declared = list(registered_runs() if runs is None else runs)
    manifest = train_manifest_path(root)
    if not manifest.is_file():
        raise ValueError(f"{manifest} is absent: the manifest is written first, and the record repeats its digests")
    listed = _manifest_entries(manifest)
    timing = Path(timing_path)
    timing_record = _read_json(timing, "the fenced timing's record")
    if timing_record.get("format_version") != TIMING_FORMAT_VERSION:
        raise ValueError(f"{timing}: format {timing_record.get('format_version')!r}, not {TIMING_FORMAT_VERSION!r}")
    k20_path = k20_record_path(root)
    k20 = _read_json(k20_path, "the K = 20 reproduction record is written before the training record")
    if k20.get("format_version") != K20_FORMAT_VERSION:
        raise ValueError(f"{k20_path}: format {k20.get('format_version')!r}, not {K20_FORMAT_VERSION!r}")

    facts: dict[str, dict[str, Any]] = {}
    for subject in dict.fromkeys(run.subject for run in declared):
        spec = (specs or {}).get(subject) or subject_spec(subject)
        facts[subject] = subject_facts(training_inputs(spec, 1, corpus_root))

    entries: dict[str, Any] = {}
    for run in declared:
        path = registered_destination(root, run)
        relative = str(path.relative_to(root))
        if not path.is_file():
            raise ValueError(f"{path} is absent: the record describes a complete set only")
        digest = _sha256_file(path)
        if listed.get(relative) != digest:
            raise ValueError(f"{relative} is not in {manifest.name} at its current digest {digest}")
        checks = validate_checkpoint(run, output_root=root, facts=facts[run.subject])
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{run.name} does not validate (failed: {failed}); the record is never written for it")
        record_path = run_record_path(root, run)
        if not record_path.is_file():
            raise ValueError(
                f"{record_path} is absent: the checkpoint exists without its run record (the process died between the "
                "two writes); the coordinator decides"
            )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("format_version") != RUN_RECORD_FORMAT_VERSION or record.get("checkpoint_sha256") != digest:
            raise ValueError(f"{record_path} is not the run record of {relative} at {digest}")
        payload = _load_weights_only(path)
        provenance = payload["provenance"]
        if int(record.get("steps", -1)) != int(provenance["gradient_steps"]):
            raise ValueError(f"{record_path} and {relative} disagree on the step count")
        attempts = attempts_of(run, output_root=root)
        if not attempts:
            raise ValueError(f"{run.name} has a checkpoint but no attempt marker: it was not trained by the driver")
        entries[run.name] = {
            "subject": run.subject,
            "k": int(run.k),
            "batch": int(run.batch),
            "seed": int(run.seed),
            "arm": run.arm,
            "steps": int(provenance["gradient_steps"]),
            "warmup_steps": int(provenance["warmup_steps"]),
            "checkpoint": relative,
            "checkpoint_sha256": digest,
            "weights_sha256": canonical_digest_of(path),
            "loop_seconds": record["loop_seconds"],
            "ms_per_step": record["ms_per_step"],
            "final_loss": record["final_loss"],
            "loss_per_supervised_target": record["loss_per_supervised_target"],
            "supervised_targets_per_step": record["supervised_targets_per_step"],
            "attempts": len(attempts),
            "reruns": len(attempts) - 1,
            "code_commit": provenance["code_commit"],
            "device": provenance["device"],
            "checks": checks,
        }
        if run.k == REFERENCE_K and run.batch == BASE_BATCH:
            measured = (k20.get("subjects") or {}).get(run.subject) or {}
            entry = next((item for item in measured.get("seeds", []) if item.get("seed") == run.seed), None)
            if entry is None or entry.get("candidate_sha256") != digest:
                raise ValueError(f"{k20_path} does not measure {relative} at its manifest digest")
    return {
        "format_version": TRAIN_RECORD_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A26(a)-(b) as corrected by A26.1; BRIEF_42 C2 and Amendment A",
        "n_runs": len(entries),
        "runs": entries,
        "subjects": {
            subject: {key: value for key, value in fact.items() if key != "stats"} for subject, fact in facts.items()
        },
        "manifest": {"name": TRAIN_MANIFEST_NAME, "sha256": _sha256_file(manifest)},
        "k20_reproduction": {"path": str(k20_path.relative_to(root)), "sha256": _sha256_file(k20_path), "record": k20},
        "timing": {"path": str(timing), "sha256": _sha256_file(timing), "record": timing_record},
        "reruns_rule": "a re-run is an attempt marker with no checkpoint after it: an infrastructure failure",
        "what_this_does_not_say": list(_TRAIN_RECORD_LIMITS),
    }


def write_train_record(output_root: str | Path, record: Mapping[str, Any]) -> Path:
    path = train_record_path(output_root)
    _write_text_once(json.dumps(record, indent=2, sort_keys=True) + "\n", path, output_root)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# The inputs, by digest, before the canary and the token
# ----------------------------------------------------------------------------------------------------------------------


def verify_reference_inputs(*, output_root: str | Path, corpus_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """The ten published K = 20 checkpoints at their pins, and both subjects' corpus facts EQUAL to P4's and P4.7's
    committed values -- read from ``p4_training.json`` and the checkpoints, never typed.  No CUDA needed."""
    root = Path(output_root)
    data = Path(data_dir)
    published = _published_k20(root, data)
    p4_training = _read_json(data / P4_TRAINING_NAME, "P4's training record names its prompt and scale")
    p4 = _load_weights_only(root / published[("mappo1000", TRAINING_SEEDS[0])]["reference"])
    p47 = _load_weights_only(root / published[("mix50", TRAINING_SEEDS[0])]["reference"])
    mappo = subject_facts(training_inputs(subject_spec("mappo1000"), 1, corpus_root))
    mix = subject_facts(training_inputs(subject_spec("mix50"), 1, corpus_root))
    problems = []
    if not (mappo["target_rtg"] == p4_training["target_rtg"] == p4["target_rtg"]):
        problems.append("mappo1000 target_rtg")
    if not (mappo["rtg_scale"] == p4_training["rtg_scale"] == p4["rtg_scale"]):
        problems.append("mappo1000 rtg_scale")
    if mappo["stats"] != p4["stats"]:
        problems.append("mappo1000 statistics")
    if mappo["max_ep_len"] != p4["config"]["max_ep_len"]:
        problems.append("mappo1000 max_ep_len")
    if mix["target_rtg"] != p47["target_rtg"] or mix["rtg_scale"] != p47["rtg_scale"]:
        problems.append("mix50 prompt")
    if mix["statistics_digest"] != p47["provenance"]["statistics_digest"] or mix["stats"] != p47["stats"]:
        problems.append("mix50 statistics")
    if problems:
        raise ValueError(
            f"the corpus under {corpus_root} does not give P4's and P4.7's committed recipe ({problems}); A26(a) trains "
            "each subject on ITS corpus, so nothing is trained"
        )
    return {
        "published": {f"{subject}_seed{seed}": entry for (subject, seed), entry in published.items()},
        "subjects": {
            name: {key: fact[key] for key in ("target_rtg", "rtg_scale", "statistics_digest", "n_windows",
                                              "corpus_manifest_sha256")}
            for name, fact in (("mappo1000", mappo), ("mix50", mix))
        },
    }


def check_inputs(
    *, output_root: str | Path, corpus_root: str | Path, data_dir: str | Path, timing_path: str | Path | None = None
) -> dict[str, Any]:
    """Every input by digest, before the canary and the token; CUDA; and, for the trainings, the timing record and
    enough free device memory for its measured peak."""
    facts = verify_reference_inputs(output_root=output_root, corpus_root=corpus_root, data_dir=data_dir)
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; the registered runs and the fenced timing train on cuda only")
    free_bytes, _total = torch.cuda.mem_get_info()
    facts["cuda_free_mib"] = free_bytes / 2**20
    if timing_path is not None:
        timing = _read_json(Path(timing_path), "the fenced timing's record (the timing mode writes it)")
        if timing.get("format_version") != TIMING_FORMAT_VERSION or timing.get("concurrency") != 1:
            raise ValueError(f"{timing_path} is not a C = 1 timing record of this task")
        needed = float(timing["slots"]["alone"]["device_peak_mib"])
        if facts["cuda_free_mib"] < needed:
            raise ValueError(
                f"{facts['cuda_free_mib']:.0f} MiB free on the device, below the {needed:.0f} MiB the timing measured"
            )
        facts["timing_sha256"] = _sha256_file(Path(timing_path))
    return facts


# ======================================================================================================================
# The command line
# ======================================================================================================================


def _cmd_extract_reference_rows(args: argparse.Namespace) -> int:
    code = _code_provenance()
    if code["code_dirty"] and not args.allow_dirty:
        raise ValueError(
            f"the code tree at {code['code_commit'][:12]} is dirty; the committed rows record the commit that wrote "
            "them (plan F8), so extract from a clean, committed tree"
        )
    payload = reference_rows_payload(output_root=Path(args.output_root), data_dir=Path(args.data_dir))
    digest = write_reference_rows(payload, Path(args.out))
    print(
        f"extract-reference-rows: {payload['n_rows']} rows from {payload['source']['directory']} -> {args.out} "
        f"(sha256 {digest}); checkpoints 5/5 at p4_gate.json's digests; code {payload['extraction']['code_commit'][:12]}",
        flush=True,
    )
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    """The sixty registered run names, one per line, in training order."""
    for run in registered_runs():
        print(run.name)
    return 0


def _cmd_check_inputs(args: argparse.Namespace) -> int:
    facts = check_inputs(
        output_root=Path(args.output_root), corpus_root=Path(args.corpus_root), data_dir=Path(args.data_dir),
        timing_path=None if args.timing is None else Path(args.timing),
    )
    subjects = facts["subjects"]
    tail = "" if "timing_sha256" not in facts else f", timing record {facts['timing_sha256'][:12]}"
    print(
        f"check_inputs PASSED: the ten published K = 20 checkpoints at their pins; the corpus gives P4's recipe "
        f"(mappo1000 {subjects['mappo1000']['target_rtg']} / {subjects['mappo1000']['rtg_scale']}, statistics == P4's) "
        f"and P4.7's (mix50 {subjects['mix50']['target_rtg']} / {subjects['mix50']['rtg_scale']}, statistics "
        f"{subjects['mix50']['statistics_digest'][:12]}); CUDA {facts['cuda_free_mib']:.0f} MiB free{tail}",
        flush=True,
    )
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    """One run's decision, or -- ``--all``, the driver's pre-token scan -- every run's, with no stray checkpoint.

    The corpus facts are built only for a subject one of whose checkpoints EXISTS: an absent checkpoint trains, and
    deciding that needs nothing but its absence.
    """
    root = Path(args.output_root)
    cache: dict[str, dict[str, Any]] = {}

    def decide(run: RunSpec) -> str:
        path = registered_destination(root, run)
        if not path.exists() and not path.is_symlink():
            return "train"
        if run.subject not in cache:
            cache[run.subject] = subject_facts(training_inputs(subject_spec(run.subject), 1, Path(args.corpus_root)))
        return resume_decision(run, output_root=root, facts=cache[run.subject])

    if args.all:
        directory = checkpoints_dir(root)
        wanted = {f"{run.name}.pt" for run in registered_runs()}
        if directory.is_dir():
            stray = sorted(path.name for path in directory.iterdir() if path.name not in wanted)
            if stray:
                raise ValueError(
                    f"{directory / stray[0]} is not one of the 60 registered runs; a person moves it aside before the "
                    "trainings start"
                )
        counts = Counter(decide(run) for run in registered_runs())
        print(f"resume_decision: {counts['train']} to train, {counts['skip']} to skip", flush=True)
        return 0
    if args.run is None:
        raise ValueError("resume-decision needs --run NAME or --all")
    print(decide(run_by_name(args.run)), flush=True)
    return 0


def _cmd_attempt(args: argparse.Namespace) -> int:
    print(next_attempt(run_by_name(args.run), output_root=Path(args.output_root)), flush=True)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """One REGISTERED run on CUDA: the registered destination, the registered budget, then its run record."""
    run = run_by_name(args.run)
    if args.device != "cuda":
        raise ValueError(
            "the registered runs train on cuda only (A26(b)); a CPU training is a test, never a registered checkpoint"
        )
    root = Path(args.output_root)
    destination = assert_fence(registered_destination(root, run), timing=False)
    assert_writable(destination, root)
    for directory in (checkpoints_dir(root), runs_dir(root), staging_dir(root)):
        if not directory.is_dir():
            raise ValueError(f"{directory} does not exist; the driver creates it after the token")
    if run_record_path(root, run).exists():
        raise ValueError(f"{run_record_path(root, run)} already exists; a run record is written once, with its checkpoint")
    state = enter_registered_regime()
    inputs = training_inputs(subject_spec(run.subject), run.k, Path(args.corpus_root))
    print(
        f"context_sweep train {run.name}: subject {run.subject}, K {run.k}, batch {run.batch}, seed {run.seed}, "
        f"{GRADIENT_STEPS} steps, {inputs.n_windows} windows, device cuda",
        flush=True,
    )
    outcome = train_run(
        run, inputs, device="cuda", destination=destination, staging_dir=staging_dir(root), code=state["code"],
        budget=GRADIENT_STEPS, log_every=int(args.log_every),
    )
    written = write_run_record(run, outcome, inputs, output_root=root, code=state["code"], device="cuda")
    print(
        f"context_sweep train {run.name}: wrote {destination.name} (sha256 {outcome.sha256[:12]}), {outcome.steps} "
        f"steps in {outcome.seconds:.1f} s; run record {written.name}",
        flush=True,
    )
    return 0


def _cmd_compare_k20(args: argparse.Namespace) -> int:
    record = k20_reproduction(Path(args.output_root), Path(args.data_dir))
    path = write_k20_record(Path(args.output_root), record)
    subjects = record["subjects"]
    print(
        f"compare-k20: tensor-for-tensor equal on {subjects['mappo1000']['n_seeds_equal']}/5 mappo1000 and "
        f"{subjects['mix50']['n_seeds_equal']}/5 mix50 seeds; recorded in {path} (a measurement, never a stop)",
        flush=True,
    )
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    path = write_manifest(Path(args.output_root))
    print(f"manifest: {path} lists the {len(registered_runs())} checkpoints and was re-verified", flush=True)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    record = build_train_record(
        Path(args.output_root), corpus_root=Path(args.corpus_root), timing_path=Path(args.timing)
    )
    path = write_train_record(Path(args.output_root), record)
    print(f"record: {path} ({record['n_runs']} runs; the coordinator verifies it and commits it at G3)", flush=True)
    return 0


def _cmd_timing(args: argparse.Namespace) -> int:
    """The fenced timing (no token): one K = 5 run at B = 400, a slot at a time, or the summary of a stamp."""
    root = Path(args.output_root)
    if args.action == "summarize":
        if _STAMP.fullmatch(str(args.stamp)) is None:
            raise ValueError(f"timing stamp {args.stamp!r} is not a UTC stamp like 20260928T230000Z")
        stamp_dir = training_root(root) / FENCED_TIMING_DIRNAME / str(args.stamp)
        if not stamp_dir.is_dir():
            raise ValueError(f"{stamp_dir} is not an existing fenced timing directory")
        summary = summarize_timing(stamp_dir)
        out = stamp_dir / TIMING_RECORD_NAME
        _write_text_once(json.dumps(summary, indent=2, sort_keys=True) + "\n", out, root)
        for name, slot in summary["slots"].items():
            print(
                f"timing {name}: {slot['ms_per_step']:.2f} ms/step over {slot['steps']} steps, device peak "
                f"{slot['device_peak_mib']:.0f} MiB, allocated peak {slot['peak_allocated_mib']:.0f} MiB, inputs built "
                f"in {slot['build_seconds']:.1f} s",
                flush=True,
            )
        repeat = summary["repeat"]
        print(
            f"timing repeat: file sha256 {'EQUAL' if repeat['file_sha256_equal'] else 'DIFFERS'}, weights "
            f"{'EQUAL' if repeat['weights_sha256_equal'] else 'DIFFER'}; concurrency 1 (fixed); wrote {out}",
            flush=True,
        )
        return 0

    destination = assert_fence(timing_destination(root, str(args.stamp), str(args.slot)), timing=True)
    assert_writable(destination, root)
    stamp_dir = destination.parent
    if not stamp_dir.is_dir():
        raise ValueError(f"{stamp_dir} is not an existing fenced timing directory; the driver creates it")
    out = destination.with_suffix(".json")
    if out.exists():
        raise ValueError(f"{out} already exists; a timing record is written once")
    state = enter_registered_regime()
    started = time.perf_counter()
    inputs = training_inputs(subject_spec(TIMING_RUN.subject), TIMING_RUN.k, Path(args.corpus_root))
    build_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    outcome = train_run(
        TIMING_RUN, inputs, device="cuda", destination=destination, staging_dir=stamp_dir, code=state["code"],
        budget=TIMING_BUDGET, log_every=100,
    )
    record = {
        "format_version": TIMING_SLOT_FORMAT_VERSION,
        "slot": str(args.slot),
        "run": TIMING_RUN.name,
        "k": TIMING_RUN.k,
        "batch": TIMING_RUN.batch,
        "seed": TIMING_RUN.seed,
        "budget": TIMING_BUDGET,
        "steps": int(outcome.steps),
        "loop_seconds": float(outcome.seconds),
        "ms_per_step": float(outcome.seconds) / int(outcome.steps) * 1000.0,
        "build_seconds": float(build_seconds),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_sha256": outcome.sha256,
        "weights_sha256": canonical_digest_of(destination),
        "device_name": torch.cuda.get_device_name(0),
        "code_commit": state["code"]["code_commit"],
        "regime": state["regime"],
        "written_utc": _utc_now(),
    }
    _write_text_once(json.dumps(record, indent=2, sort_keys=True) + "\n", out, root)
    print(
        f"timing {args.slot}: {record['ms_per_step']:.2f} ms/step over {outcome.steps} steps, inputs built in "
        f"{build_seconds:.1f} s, peak {record['peak_allocated_mib']:.0f} MiB allocated",
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m offline.context_sweep", description=__doc__.splitlines()[0], allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        "extract-reference-rows", help="C1: P4's K = 20 rows from P8.4b's cells, written once", allow_abbrev=False
    )
    extract.add_argument("--output-root", required=True, help="the output tree holding p4_dt/ and p8_4b_rederivation/")
    extract.add_argument("--data-dir", required=True, help="the docs/data directory holding p4_gate.json")
    extract.add_argument("--out", required=True, help="the file to write: docs/data/p4_k20_att_engine_rows.json")
    extract.add_argument(
        "--allow-dirty", action="store_true", help="accept a dirty code tree; the payload then records code_dirty true"
    )
    extract.set_defaults(handler=_cmd_extract_reference_rows)

    def command(name: str, help_text: str, handler: Any, *, corpus: bool = False, data: bool = False) -> Any:
        sub = commands.add_parser(name, help=help_text, allow_abbrev=False)
        sub.add_argument("--output-root", required=True, help="the output tree (the MAIN tree's output/)")
        if corpus:
            sub.add_argument("--corpus-root", required=True, help="datasets_v11, passed as P4's absolute root")
        if data:
            sub.add_argument("--data-dir", required=True, help="the docs/data directory")
        sub.set_defaults(handler=handler)
        return sub

    runs = commands.add_parser("runs", help="C2: the sixty registered run names, in order", allow_abbrev=False)
    runs.set_defaults(handler=_cmd_runs)
    inputs = command("check-inputs", "C2: every input by digest, before the canary", _cmd_check_inputs, corpus=True,
                     data=True)
    inputs.add_argument("--timing", default=None, help="the fenced timing's timing.json (train mode)")
    resume = command("resume-decision", "C2: skip / train for one run, or the pre-token scan of all", _cmd_resume,
                     corpus=True)
    which = resume.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", default=None)
    which.add_argument("--all", action="store_true")
    attempt = command("attempt", "C2: write the next attempt marker of a run; print its number", _cmd_attempt)
    attempt.add_argument("--run", required=True)
    train = command("train", "C2: one registered run on CUDA, written once", _cmd_train, corpus=True)
    train.add_argument("--run", required=True, help="a registered run name, e.g. mappo1000_k20_b64_seed101")
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5000)
    command("compare-k20", "C2: the K = 20 reproduction measurement, recorded", _cmd_compare_k20, data=True)
    command("manifest", "C2: write and re-verify SHA256SUMS_p5_3c_train.txt", _cmd_manifest)
    record = command("record", "C2: write p5_3c_training/p5_3c_train.json", _cmd_record, corpus=True)
    record.add_argument("--timing", required=True, help="the fenced timing's timing.json")

    timing = commands.add_parser("timing", help="C2: the fenced timing run (no token)", allow_abbrev=False)
    actions = timing.add_subparsers(dest="action", required=True)
    slot = actions.add_parser("run", help="one fenced run: mappo1000, K 5, batch 64, seed 101, B 400",
                              allow_abbrev=False)
    slot.add_argument("--output-root", required=True)
    slot.add_argument("--corpus-root", required=True)
    slot.add_argument("--stamp", required=True)
    slot.add_argument("--slot", required=True, choices=TIMING_SLOTS)
    slot.set_defaults(handler=_cmd_timing)
    summary = actions.add_parser("summarize", help="the slots, the repeat by two routes", allow_abbrev=False)
    summary.add_argument("--output-root", required=True)
    summary.add_argument("--stamp", required=True)
    summary.set_defaults(handler=_cmd_timing)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; a refusal prints its reason and returns 2.  Anything else propagates, loudly."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"context_sweep {args.command}: REFUSED: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
