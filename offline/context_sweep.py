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
* ``p5.3c-cell/1.0`` -- ``output/p5_3c/cells/cell_<arm>_seed<s>_draw<d>.json`` (C3): one CityFlow episode of one cell --
  the cell's identity, ``admission_probe.AdmissionEpisode.as_record()`` under ``episode`` (both ATT definitions and the
  admission counts), the per-decision series (the return-to-go read BEFORE each decision, the reward read from that
  decision's ``info``, the action), the checkpoint and demand digests, the device, the thread count, the commit.
* ``p5.3c-reference-gate/1.0`` -- ``output/p5_3c/reference_gate.json``: A26(c)'s gate, written ONLY on a pass.
* ``p5.3c-reference-reroll/1.0`` -- ``output/p5_3c/g2/reference_reroll_check_<UTC>/``: the fenced pre-token re-roll.
* ``p5.3c-context-sweep/1.0`` -- ``output/p5_3c/artifacts/p5_3c_context_sweep.json``: the artifact (no series).

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

C3 -- THE CAMPAIGN AND THE REPORT (A26(c)-(e), A26.1(b)-(c))
------------------------------------------------------------
7,000 declared cells: the REFERENCE stage (P4's and P4.7's five published K = 20 checkpoints, 1,000 cells), A26(c)'s
GATE -- P4's 500 chunks equal to the committed rows under ``==`` on both definitions, or the campaign stops and no sweep
cell is rolled -- then the SWEEP (the sixty runs, 6,000 cells), each checkpoint read through the training record at its
pin.  One cell is P8.4b's path, unchanged: ``rtg_calibration.agent_with_target`` (load, then the subject's registered
prompt) acting greedily with ``update_memory=True`` inside ``admission_probe.probe_episode``, the subject's evaluation
settings, the draw's config and ``created`` count, engine seed 1000, the DT on CUDA.  The report refuses before it
writes; it computes, on ``att_engine`` for ``mappo1000``, T1 (the one-sided Wilcoxon, ``less``, of the per-draw rank
contrast ``s_d``), T2 and T3 (``greater``, of ``(A_d(K) - A_d(20)) - delta`` at K = 1 and 2), Holm within the three, the
registered outcome and its sentence; beside them, never deciding, everything A26(d) lists and the whole design on
``mix50``.  From the PINNED training record (``TRAIN_RECORD_SHA256``, gate G3, ``BRIEF_42`` Amendment C): C2's per-arm
training table computed, never retyped, beside the training-loss observation (C4.2), and C3's wording of the equal
K = 20 reproduction in the ``k20_reproduction`` section, made only when the record shows what it presumes.
Alignment: a chunk's series is aligned per decision -- ``rtg[t]`` read before decision ``t``, ``reward[t]`` the
reward the ``info`` of decision ``t`` carries (the previous step's), ``action[t]`` the action chosen at ``t``.
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
    # C3
    "ALPHA",
    "ARTIFACT_FORMAT_VERSION",
    "CELL_FORMAT_VERSION",
    "CampaignCell",
    "DELTA",
    "GATE_FORMAT_VERSION",
    "OneSidedWilcoxon",
    "RANK_CONTRAST",
    "REFERENCE_ROWS_SHA256",
    "REROLL_FORMAT_VERSION",
    "SENTENCE_TEMPLATES",
    "SHORTFALL_CLAUSE_TEMPLATE",
    "TRAIN_RECORD_SHA256",
    "build_context_sweep_artifact",
    "campaign_cell_by_name",
    "cell_decision_factory",
    "cell_payload",
    "check_campaign_inputs",
    "checkpoint_for_cell",
    "chunk_is_reusable",
    "confirmatory_family",
    "declared_campaign_cells",
    "holm",
    "load_reference_rows",
    "load_train_record",
    "one_sided_wilcoxon",
    "outcome_of",
    "outcome_sentence",
    "per_draw_means",
    "published_checkpoint",
    "rank_contrast",
    "reference_gate",
    "reroll_cells",
    "run_campaign_cell",
    "run_campaign_stage",
    "run_reference_reroll_check",
    "sweep_checkpoint",
    "validate_chunk",
    "verify_gate_record",
    "write_campaign_manifest",
    "write_chunk",
    "write_context_sweep_artifact",
    "write_gate_record",
    # C3, Amendment C
    "K20_READING",
    "k20_reading",
    "training_table",
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
# C3 -- the campaign: the declared cells, one cell, the chunks, the reference gate, the fenced re-roll
# ======================================================================================================================

CELL_FORMAT_VERSION = "p5.3c-cell/1.0"
GATE_FORMAT_VERSION = "p5.3c-reference-gate/1.0"
REROLL_FORMAT_VERSION = "p5.3c-reference-reroll/1.0"
ARTIFACT_FORMAT_VERSION = "p5.3c-context-sweep/1.0"

CAMPAIGN_DIRNAME = "p5_3c"
CELLS_DIRNAME = "cells"
FAILED_CHUNKS_DIRNAME = "failed"
ARTIFACTS_DIRNAME = "artifacts"
G2_DIRNAME = "g2"
ARTIFACT_NAME = "p5_3c_context_sweep.json"
GATE_RECORD_NAME = "reference_gate.json"
CAMPAIGN_MANIFEST_NAME = "SHA256SUMS_p5_3c.txt"
CAMPAIGN_OUTPUT_ENTRIES: tuple[str, ...] = (CAMPAIGN_DIRNAME, CAMPAIGN_MANIFEST_NAME)
#: Files the driver and ``record-canary`` put beside the chunks; nothing else may sit in ``cells/``.
CELLS_DIR_NON_CHUNKS: tuple[str, ...] = ("canary.json", "COMPLETE", "FAILED", FAILED_CHUNKS_DIRNAME)

STAGES: tuple[str, ...] = ("reference", "sweep")
#: The subject whose reference arm GATES the campaign (A26(c)); ``mix50``'s reference arm is reported, never gating.
GATED_SUBJECT = "mappo1000"
#: A26.1(b): the DT is evaluated on CUDA, as the reference rows were produced.
EVAL_DEVICE = "cuda"
EVAL_SCENARIO = "hz1x1"
EVAL_SCENARIO_KEY = "cityflow1x1"
ENGINE_SEED = 1000
CAMPAIGN_WORKERS = 12
#: Amendment D.2, D.2.2(1): the longest silence a stage (and the fenced re-roll) waits for its pool's NEXT result --
#: about 30 times a cell's wall time under twelve workers (results arrived every ~0.5 s, ~6 s a cell per worker, on
#: 2026-09-30), so a silence this long means only hung tasks remain: CityFlow's Engine::~Engine() race at an episode's
#: end (DEFERRED 104; docs/notes/P5.3c_HANG_2026-09-30.md).
STAGE_RESULT_TIMEOUT_S = 180.0
#: The rounds a cell is rolled in before a hang is its failure: a round is one pool over the cells still without a
#: result, the first pool being round 1; a cell that hung in every round fails as "hung 3 times".
STAGE_HANG_ROUNDS = 3
#: Amendment A, Q2 (with A.1): the fenced pre-token re-roll -- P4's five seeds on these three draws, fifteen cells.
REROLL_DRAWS: tuple[int, ...] = (1000, 1001, 1002)

#: ``docs/data/p4_k20_att_engine_rows.json`` as committed at C1b (``1d1f67f``), read at this digest or not at all.
REFERENCE_ROWS_SHA256 = "b36b8c7790f4740b65c158bba91ebc82a08a6acc9e45f8465a63710e078ed45b"
#: The sixty trainings' record, committed by the coordinator as ``docs/data/p5_3c_train.json`` on main at ``c55693a``,
#: gate G3 (A26(b): pinned before the evaluation token; BRIEF_42 Amendment C, C0 and C4.1).  Set in C3's commit, the
#: first after G3; were it ``None``, EVERY sweep checkpoint lookup would refuse, so no trained cell can be evaluated
#: against an unpinned record (``offline.transfer_curve``'s ``P7_3C_FINETUNE_SHA256`` pattern; a test keeps that
#: refusal exercised by setting it ``None``).
TRAIN_RECORD_DATA_NAME = "p5_3c_train.json"
TRAIN_RECORD_SHA256: str | None = "017808a5e84fada469d6b2d302889ad171b8e429b1612b8ac06c900ef3c6321a"
#: P8.4b's re-derived cells of P4.7's ``mix50`` K = 20 model: the ``mix50`` reference arm's per-draw comparison
#: (A26(c): reported, never gating); P5.3b read these same cells and committed their means.
P8_4B_MIX50_ARM = "dt@mix50"
P5_3B_ARTIFACT_NAME = "p5_3b_nortg.json"

#: A26(d): A6's margin as a superiority margin, a LITERAL (a test asserts it equals ``offline_baselines.DELTA_ATT`` and
#: ``docs/data/p4_4_baselines.json``'s).
DELTA = 0.6263
ALPHA = 0.05
#: The linear-trend contrast on the levels' RANKS, K ascending (A26(d)): ranks, not log K.
RANK_CONTRAST: tuple[int, ...] = (-2, -1, 0, 1, 2)
FAMILY: tuple[str, ...] = ("T1", "T2", "T3")
#: T2 and T3's context lengths, against the plateau K = 20.
SHORTFALL_TESTS: dict[str, int] = {"T2": 1, "T3": 2}

#: The device budget checked before the token: the fenced timing's per-process footprint (a TRAINING process,
#: 1,540 - 1,125 = 415 MiB, more than an evaluation process holds) x the twelve workers x 1.4.
PER_PROCESS_DEVICE_MIB = 415
EVAL_DEVICE_BUDGET_MIB = int(PER_PROCESS_DEVICE_MIB * CAMPAIGN_WORKERS * 1.4)


@dataclass(frozen=True)
class CampaignCell:
    """One declared cell: a stage, a subject's model (K, batch, seed) and one held-out draw."""

    stage: str
    subject: str
    k: int
    batch: int
    seed: int
    draw_id: int

    @property
    def arm(self) -> str:
        """``ref_<subject>_k20`` for a reference cell, the run's arm ``<subject>_k<K>_b<batch>`` for a sweep cell."""
        if self.stage == "reference":
            return f"ref_{self.subject}_k{self.k}"
        return f"{self.subject}_k{self.k}_b{self.batch}"

    @property
    def run(self) -> RunSpec | None:
        """The registered run of a sweep cell; ``None`` for a reference cell (the published checkpoints)."""
        if self.stage == "reference":
            return None
        return RunSpec(self.subject, self.k, self.batch, self.seed)

    @property
    def name(self) -> str:
        """The chunk's file name, ``cell_<arm>_seed<seed>_draw<draw>.json``."""
        return f"cell_{self.arm}_seed{self.seed}_draw{self.draw_id}.json"


#: The gated reference arm: P4's five K = 20 checkpoints (A26(c)).
GATED_ARM = f"ref_{GATED_SUBJECT}_k{REFERENCE_K}"


def declared_campaign_cells(stage: str | None = None) -> tuple[CampaignCell, ...]:
    """The 7,000 declared cells (1,000 reference, 6,000 sweep), or one stage's, in rolling order.

    ``reference``: per subject (``mappo1000``, then ``mix50``) the published K = 20 checkpoint of each seed on the 100
    held-out draws.  ``sweep``: the sixty registered runs in training order, each on the 100 draws.
    """
    if stage is not None and stage not in STAGES:
        raise ValueError(f"stage {stage!r} is not one of {list(STAGES)}")
    cells: list[CampaignCell] = []
    if stage in (None, "reference"):
        for subject in SUBJECTS:
            cells.extend(
                CampaignCell("reference", subject, REFERENCE_K, BASE_BATCH, seed, draw)
                for seed in TRAINING_SEEDS
                for draw in HELD_OUT_DRAWS
            )
    if stage in (None, "sweep"):
        for run in registered_runs():
            cells.extend(CampaignCell("sweep", run.subject, run.k, run.batch, run.seed, draw) for draw in HELD_OUT_DRAWS)
    return tuple(cells)


_CELLS_BY_NAME: dict[str, CampaignCell] = {}


def campaign_cell_by_name(name: str) -> CampaignCell:
    """The declared cell whose chunk is called *name*; refuses anything else."""
    if not _CELLS_BY_NAME:
        _CELLS_BY_NAME.update((cell.name, cell) for cell in declared_campaign_cells())
    cell = _CELLS_BY_NAME.get(str(name))
    if cell is None:
        raise ValueError(f"{name!r} is not one of the 7,000 declared cells (BRIEF_42 C3); nothing else is a chunk")
    return cell


def campaign_root(output_root: str | Path) -> Path:
    return Path(output_root) / CAMPAIGN_DIRNAME


def cells_dir(output_root: str | Path) -> Path:
    return campaign_root(output_root) / CELLS_DIRNAME


def chunk_path(output_root: str | Path, cell: CampaignCell) -> Path:
    return cells_dir(output_root) / cell.name


def gate_record_path(output_root: str | Path) -> Path:
    return campaign_root(output_root) / GATE_RECORD_NAME


def artifact_path(output_root: str | Path) -> Path:
    return campaign_root(output_root) / ARTIFACTS_DIRNAME / ARTIFACT_NAME


def campaign_manifest_path(output_root: str | Path) -> Path:
    return Path(output_root) / CAMPAIGN_MANIFEST_NAME


def g2_dir(output_root: str | Path) -> Path:
    return campaign_root(output_root) / G2_DIRNAME


def _write_campaign_text_once(text: str, destination: Path, output_root: str | Path) -> str:
    """:func:`_write_text_once` under the CAMPAIGN's writable entries: once; the same text again is a no-op."""
    from offline.few_shot import _link_exclusive

    assert_writable(destination, output_root, entries=CAMPAIGN_OUTPUT_ENTRIES)
    data = text.encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and destination.read_bytes() == data:
            return _sha256_bytes(data)
        raise ValueError(f"{destination} already exists and differs; it is never rewritten -- a person moves it aside")
    if not destination.parent.is_dir():
        raise ValueError(f"{destination.parent} does not exist; nothing is created here")
    return _link_exclusive(data, destination)


def published_checkpoint(subject: str, seed: int, *, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """ONE of the ten published K = 20 checkpoints, verified by the rules C2's ``_published_k20`` applies to all ten.

    P4's against ``docs/data/p4_gate.json``; P4.7's against ``output/SHA256SUMS_p4_7.txt`` and the canonical weights
    digest ``docs/data/p4_7_training.json`` records.  The path is relative to the output root.
    """
    root = Path(output_root)
    data = Path(data_dir)
    if subject not in SUBJECTS or isinstance(seed, bool) or int(seed) not in TRAINING_SEEDS:
        raise ValueError(f"({subject!r}, {seed!r}) is not one of the ten published K = 20 checkpoints")
    seed = int(seed)
    if subject == "mappo1000":
        gate = (_read_json(data / P4_GATE_NAME, "P4's gate artifact names its checkpoints' digests").get("checkpoints")
                or {})
        expected = (gate.get(str(seed)) or {}).get("sha256")
        relative = f"{P4_CHECKPOINT_DIRNAME}/dt_seed{seed}.pt"
        path = root / relative
        if not path.is_file() or not expected or _sha256_file(path) != expected:
            raise ValueError(f"{path} is absent or not at {P4_GATE_NAME}'s sha256 {expected}; only P4's published "
                             "checkpoint is its reference arm")
        return {"path": relative, "sha256": str(expected), "checked_against": [f"docs/data/{P4_GATE_NAME}"]}
    relative = P4_7_CHECKPOINT_TEMPLATE.format(seed=seed)
    path = root / relative
    sums_path = root / P4_7_MANIFEST_NAME
    if not sums_path.is_file():
        raise ValueError(f"{sums_path} is absent: P4.7's manifest names its checkpoints' digests")
    expected = _manifest_entries(sums_path).get(relative)
    if not path.is_file() or not expected or _sha256_file(path) != expected:
        raise ValueError(f"{path} is absent or not at {P4_7_MANIFEST_NAME}'s sha256 {expected}")
    p47_runs = _read_json(data / P4_7_TRAINING_NAME, "P4.7's training record names its canonical digests").get("runs")
    canonical = {
        int(run["seed"]): str(run["canonical_digest"])
        for run in (p47_runs or [])
        if run.get("tier") == "mix50" and run.get("method") == "dt"
    }
    if canonical.get(seed) != canonical_digest_of(path):
        raise ValueError(f"{path}: its weights are not {P4_7_TRAINING_NAME}'s canonical digest {canonical.get(seed)}")
    return {
        "path": relative,
        "sha256": str(expected),
        "checked_against": [f"output/{P4_7_MANIFEST_NAME}", f"docs/data/{P4_7_TRAINING_NAME}"],
    }


TRAIN_RECORD_PIN_LABEL = "docs/data/p5_3c_train.json at offline.context_sweep.TRAIN_RECORD_SHA256 (G3)"


def load_train_record(data_dir: str | Path) -> dict[str, Any]:
    """``docs/data/p5_3c_train.json`` at :data:`TRAIN_RECORD_SHA256`, the digest checked BEFORE it is parsed."""
    pin = TRAIN_RECORD_SHA256
    if pin is None:
        raise ValueError(
            "TRAIN_RECORD_SHA256 is not set: the sixty trainings' record is pinned in the first commit after gate G3 "
            "(A26(b): committed on main before the evaluation token), and until then no trained checkpoint is evaluated"
        )
    path = Path(data_dir) / TRAIN_RECORD_DATA_NAME
    if not path.is_file():
        raise ValueError(f"{path} is absent: the coordinator commits the training record at gate G3")
    digest = _sha256_file(path)
    if digest != pin:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {pin}; a record at another digest is not read")
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict) or record.get("format_version") != TRAIN_RECORD_FORMAT_VERSION:
        raise ValueError(f"{path} is not a {TRAIN_RECORD_FORMAT_VERSION} record")
    return record


def sweep_checkpoint(run: RunSpec, *, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """A registered run's checkpoint from the PINNED training record, hashed at consumption, its provenance checked.

    In order, each a refusal: the run is registered; the record is at its pin; it names the run as exactly that
    (subject, K, batch, seed); the checkpoint's path is the run's registered destination (never absolute, never under
    ``fenced_timing/``); the file's sha256 is the record's; the training manifest is at the record's digest and lists
    the file at that sha256; a weights-only load shows the run's K, batch, 40,000 steps, seed, name and the subject's
    registered prompt and scale.
    """
    run = validate_run(run)
    record = load_train_record(data_dir)
    entry = (record.get("runs") or {}).get(run.name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"the pinned training record names no run {run.name}")
    recorded = (entry.get("subject"), entry.get("k"), entry.get("batch"), entry.get("seed"))
    if recorded != (run.subject, run.k, run.batch, run.seed):
        raise ValueError(f"{run.name}: the pinned record says (subject, K, batch, seed) {recorded}")
    expected = Path(TRAINING_DIRNAME) / CHECKPOINTS_DIRNAME / f"{run.name}.pt"
    relative = Path(str(entry.get("checkpoint")))
    if relative.is_absolute() or FENCED_TIMING_DIRNAME in relative.parts or relative != expected:
        raise ValueError(
            f"{run.name}: the pinned record's checkpoint {relative} is not its registered destination {expected} "
            f"(never absolute, never under {FENCED_TIMING_DIRNAME}/: the timing run is never evaluated)"
        )
    root = Path(output_root)
    path = root / relative
    if not path.is_file():
        raise ValueError(f"{path}: {run.name}'s checkpoint is not on disk")
    digest = _sha256_file(path)
    if digest != entry.get("checkpoint_sha256"):
        raise ValueError(
            f"{path}: sha256 {digest} is not the pinned record's {entry.get('checkpoint_sha256')} for {run.name}; other "
            "weights under this name would evaluate as it"
        )
    manifest = train_manifest_path(root)
    if not manifest.is_file():
        raise ValueError(f"{manifest} is absent: {TRAIN_MANIFEST_NAME} lists the sixty ({run.name})")
    if _sha256_file(manifest) != (record.get("manifest") or {}).get("sha256"):
        raise ValueError(f"{manifest} is not at the pinned record's {TRAIN_MANIFEST_NAME} digest ({run.name})")
    if _manifest_entries(manifest).get(relative.as_posix()) != digest:
        raise ValueError(f"{relative} is not listed in {TRAIN_MANIFEST_NAME} at {digest} ({run.name})")
    payload = _load_weights_only(path)
    config = dict(payload.get("config") or {})
    provenance = dict(payload.get("provenance") or {})
    spec = subject_spec(run.subject)
    guard = {
        "context_length": config.get("context_length") == run.k,
        "batch": provenance.get("batch_size") == run.batch,
        "budget": provenance.get("gradient_steps") == GRADIENT_STEPS,
        "seed": provenance.get("seed") == run.seed,
        "run": provenance.get("run") == run.name,
        "prompt": payload.get("target_rtg") == float(spec.target_rtg)
        and payload.get("rtg_scale") == float(spec.rtg_scale),
    }
    failed = [name for name, passed in guard.items() if not passed]
    if failed:
        raise ValueError(f"{path}: {run.name}'s payload fails {failed}")
    return {"path": relative.as_posix(), "sha256": digest, "checked_against": [TRAIN_RECORD_PIN_LABEL, TRAIN_MANIFEST_NAME]}


def checkpoint_for_cell(cell: CampaignCell, *, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """The checkpoint a cell evaluates: a published one (reference) or a registered run's (sweep), verified."""
    if campaign_cell_by_name(cell.name) != cell:
        raise ValueError(f"{cell} is not a declared cell")
    if cell.run is None:
        return published_checkpoint(cell.subject, cell.seed, output_root=output_root, data_dir=data_dir)
    return sweep_checkpoint(cell.run, output_root=output_root, data_dir=data_dir)


def evaluation_env_settings(subject: str, corpus_root: str | Path) -> dict[str, Any]:
    """The subject's evaluation env settings, read from its collection manifests (P4.6's route; plan A7: equal to
    P8.4b's, which ``campaign-inputs`` checks before the token)."""
    from offline.method_tier_grid import env_settings_for_tiers

    return env_settings_for_tiers([subject_spec(subject)], corpus_root)


def _rederivation_settings(subject: str, corpus_root: str | Path) -> dict[str, Any]:
    """P8.4b's route to the same settings (``offline.att_rederivation.rederivation_env_settings``)."""
    from types import SimpleNamespace

    from offline.att_rederivation import rederivation_env_settings

    return rederivation_env_settings(EVAL_SCENARIO, subject, SimpleNamespace(corpus_root=Path(corpus_root)))


def draw_demand_identity(draw_id: int, *, draws_root: str | Path) -> dict[str, Any]:
    """A held-out draw's CityFlow config path and the sha256 of its config and flow files."""
    from offline.materialise_draws import FLOW_FILENAME, draw_config_path

    if isinstance(draw_id, bool) or int(draw_id) not in HELD_OUT_DRAWS:
        raise ValueError(f"draw {draw_id!r} is not one of the held-out draws 1000-1099")
    config = Path(draw_config_path(EVAL_SCENARIO_KEY, int(draw_id), out_root=draws_root))
    flow = config.parent / FLOW_FILENAME
    for path in (config, flow):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is absent: draw {draw_id} is not materialised")
    return {"config_path": config, "config_sha256": _sha256_file(config), "flow_sha256": _sha256_file(flow)}


def cell_decision_factory(
    checkpoint_path: str | Path, *, target_rtg: float, device: str, series: dict[str, list[Any]]
) -> Any:
    """``method_tier_grid._dt_factory``'s path, recording the per-decision series into *series*.

    The agent is ``rtg_calibration.agent_with_target`` -- load (refusing any step count but the declared 40,000), THEN
    apply the target, asserted to have taken -- and every decision ``act(info, explore=False, update_memory=True)``,
    exactly what produced the reference rows (``offline/att_rederivation.py:1096-1101``).  The series is P7.3a's
    ``dt_choose``'s: the return-to-go read BEFORE the call (``current_rtg`` is a pure read), the reward read from
    ``info``, the action.
    """

    def factory(env: Any) -> Any:
        from offline.rtg_calibration import agent_with_target

        agent = agent_with_target(
            env, checkpoint_path, declared_gradient_steps=GRADIENT_STEPS, target_rtg=float(target_rtg), device=device
        )
        ix_id = str(list(env.intersections)[0].id)

        def choose(_env: Any, info: Mapping[str, Any]) -> Any:
            series["rtg"].append(float(agent.current_rtg()[ix_id]))
            payload = info["intersections"][ix_id]
            series["reward"].append(None if "reward" not in payload else float(payload["reward"]))
            action = agent.act(info, explore=False, update_memory=True)
            series["action"].append(int(np.asarray(action).reshape(-1)[0]))
            return action

        return choose

    return factory


def cell_payload(
    cell: CampaignCell,
    *,
    episode: Mapping[str, Any],
    series: Mapping[str, Sequence[Any]],
    checkpoint: Mapping[str, Any],
    demand: Mapping[str, Any],
    target_rtg: float,
    device: str,
    code: Mapping[str, Any],
    canary_seconds: float | None,
    torch_num_threads: int,
) -> dict[str, Any]:
    """One cell's chunk, assembled and validated; pure.  Format ``p5.3c-cell/1.0``."""
    payload = {
        "format_version": CELL_FORMAT_VERSION,
        "name": cell.name,
        "stage": cell.stage,
        "arm": cell.arm,
        "subject": cell.subject,
        "k": int(cell.k),
        "batch": int(cell.batch),
        "seed": int(cell.seed),
        "draw_id": int(cell.draw_id),
        "run": None if cell.run is None else cell.run.name,
        "episode": dict(episode),
        "n_decisions": len(series["action"]),
        "series": {
            "rtg": [float(value) for value in series["rtg"]],
            "reward": [None if value is None else float(value) for value in series["reward"]],
            "action": [int(value) for value in series["action"]],
        },
        "checkpoint": dict(checkpoint),
        "demand": {"config_sha256": str(demand["config_sha256"]), "flow_sha256": str(demand["flow_sha256"])},
        "target_rtg": float(target_rtg),
        "device": str(device),
        "engine_seed": ENGINE_SEED,
        "scenario": EVAL_SCENARIO,
        "scenario_id": EVAL_SCENARIO_KEY,
        "torch_num_threads": int(torch_num_threads),
        "code_commit": str(code["code_commit"]),
        "code_dirty": bool(code["code_dirty"]),
        "canary_seconds": None if canary_seconds is None else float(canary_seconds),
    }
    validate_chunk(payload, cell)
    return payload


#: ``AdmissionEpisode.as_record()``'s keys, every one required in a chunk (A11(b) is unconditional).
EPISODE_KEYS: tuple[str, ...] = (
    "scenario", "tier", "method", "arm", "seed", "draw_id", "created", "entered", "never_entered", "entered_fraction",
    "completed_at_horizon", "running_at_horizon", "waiting_at_horizon", "att_ours", "att_engine", "att_difference",
    "horizon_vehicle_count", "episode_reward", "seconds", "seconds_rollout",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_chunk(payload: Mapping[str, Any], cell: CampaignCell) -> None:
    """Refuse a chunk that is not a complete, internally consistent chunk of *cell*, naming the FIELD.

    The identity is the cell's; the episode carries every A11(b) quantity, both ATT definitions finite, and names the
    cell; the series' three lists are equally long and ``n_decisions`` long; the checkpoint path is relative and never
    under ``fenced_timing/``, its digest a sha256; the demand's two digests too; CUDA (A26.1(b)), engine seed 1000, the
    scenario; a positive thread count; a 40-hex commit from a CLEAN tree (J1(d)); the subject's registered prompt.
    """

    def refuse(field: str, detail: str) -> None:
        raise ValueError(f"{cell.name}: field '{field}' {detail}")

    if not isinstance(payload, Mapping):
        raise ValueError(f"{cell.name}: the chunk is a {type(payload).__name__}, not an object")
    if payload.get("format_version") != CELL_FORMAT_VERSION:
        refuse("format_version", f"is {payload.get('format_version')!r}, not {CELL_FORMAT_VERSION!r}")
    identity = {
        "name": cell.name, "stage": cell.stage, "arm": cell.arm, "subject": cell.subject, "k": cell.k,
        "batch": cell.batch, "seed": cell.seed, "draw_id": cell.draw_id,
        "run": None if cell.run is None else cell.run.name,
    }
    for key, expected in identity.items():
        value = payload.get(key)
        if value != expected or isinstance(value, bool):
            refuse(key, f"is {value!r}, not {expected!r}")
    episode = payload.get("episode")
    if not isinstance(episode, Mapping):
        refuse("episode", "is not an object")
    for key in EPISODE_KEYS:
        if key not in episode:
            refuse(f"episode.{key}", "is absent: every A11(b) quantity is read from the live engine, never added later")
    for key, expected in (("scenario", EVAL_SCENARIO), ("tier", cell.subject), ("method", "dt"), ("arm", cell.arm),
                          ("seed", cell.seed), ("draw_id", cell.draw_id)):
        if episode[key] != expected:
            refuse(f"episode.{key}", f"is {episode[key]!r}, not {expected!r}")
    for key in ATT_DEFINITIONS:
        value = episode[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            refuse(f"episode.{key}", "is not a finite number")
    series = payload.get("series")
    if not isinstance(series, Mapping) or sorted(series) != ["action", "reward", "rtg"]:
        refuse("series", "is not the three lists rtg, reward, action")
    lengths = {name: len(series[name]) for name in ("rtg", "reward", "action")}
    if len(set(lengths.values())) != 1:
        refuse("series", f"lists differ in length {lengths}")
    if not all(_is_int(action) for action in series["action"]):
        refuse("series", "carries a non-integer action")
    n_decisions = payload.get("n_decisions")
    if not _is_int(n_decisions) or n_decisions < 1 or n_decisions != lengths["action"]:
        refuse("n_decisions", f"is {n_decisions!r}, not the series' {lengths['action']} decisions")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        refuse("checkpoint", "is not an object")
    path = checkpoint.get("path")
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        refuse("checkpoint.path", f"{path!r} is not a path relative to the output root")
    if FENCED_TIMING_DIRNAME in Path(path).parts:
        refuse("checkpoint.path", f"{path!r} lies under {FENCED_TIMING_DIRNAME}/, and nothing there is ever evaluated")
    if not isinstance(checkpoint.get("sha256"), str) or _HEX64.fullmatch(checkpoint["sha256"]) is None:
        refuse("checkpoint.sha256", "is not a sha256")
    demand = payload.get("demand")
    if not isinstance(demand, Mapping):
        refuse("demand", "is not an object")
    for key in ("config_sha256", "flow_sha256"):
        if not isinstance(demand.get(key), str) or _HEX64.fullmatch(demand[key]) is None:
            refuse(f"demand.{key}", "is not a sha256")
    if payload.get("device") != EVAL_DEVICE:
        refuse("device", f"is {payload.get('device')!r}: A26.1(b) evaluates the DT on {EVAL_DEVICE}")
    for key, expected in (("engine_seed", ENGINE_SEED), ("scenario", EVAL_SCENARIO), ("scenario_id", EVAL_SCENARIO_KEY)):
        if payload.get(key) != expected:
            refuse(key, f"is {payload.get(key)!r}, not {expected!r}")
    threads = payload.get("torch_num_threads")
    if not _is_int(threads) or threads < 1:
        refuse("torch_num_threads", f"is {threads!r}")
    commit = payload.get("code_commit")
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        refuse("code_commit", f"is {commit!r}, not a 40-hex commit")
    if payload.get("code_dirty") is not False:
        refuse("code_dirty", "is not false: a chunk rolled from a dirty tree records a commit it did not run (J1(d))")
    if payload.get("target_rtg") != float(subject_spec(cell.subject).target_rtg):
        refuse("target_rtg", f"is {payload.get('target_rtg')!r}, not the subject's registered prompt")
    canary = payload.get("canary_seconds")
    if canary is not None and (isinstance(canary, bool) or not isinstance(canary, (int, float))):
        refuse("canary_seconds", f"is {canary!r}")


def run_campaign_cell(
    cell: CampaignCell,
    *,
    output_root: str | Path,
    corpus_root: str | Path,
    draws_root: str | Path,
    data_dir: str | Path,
    canary_seconds: float | None,
    device: str = EVAL_DEVICE,
    episode_runner: Any = None,
) -> dict[str, Any]:
    """ONE CityFlow episode of *cell* through P8.4b's path, returned as a validated chunk; it writes NOTHING.

    The checkpoint is verified first; the env settings are the subject's; the draw's config, its ``created`` count
    from its flow file at the settings' horizon, engine seed 1000 -- the arguments P8.4b's runner handed
    ``admission_probe.probe_episode`` (``offline/att_rederivation.py:1219-1260``), with the subject's registered prompt
    and the device A26.1(b) registers.  *episode_runner* replaces ``probe_episode`` in tests only.
    """
    from offline.admission_probe import created_from_flow, probe_episode

    if campaign_cell_by_name(cell.name) != cell:
        raise ValueError(f"{cell} is not a declared cell")
    identity = checkpoint_for_cell(cell, output_root=output_root, data_dir=data_dir)
    spec = subject_spec(cell.subject)
    settings = evaluation_env_settings(cell.subject, corpus_root)
    horizon = int(settings["max_steps"]) * int(settings["delta_time"])
    demand = draw_demand_identity(cell.draw_id, draws_root=draws_root)
    config = Path(demand["config_path"])
    created = created_from_flow(config.parent / "flow.json", horizon_seconds=horizon)
    series: dict[str, list[Any]] = {"rtg": [], "reward": [], "action": []}
    target = float(spec.target_rtg)
    factory = cell_decision_factory(Path(output_root) / identity["path"], target_rtg=target, device=device, series=series)
    runner = probe_episode if episode_runner is None else episode_runner
    episode = runner(
        scenario=EVAL_SCENARIO,
        tier=cell.subject,
        method="dt",
        arm=cell.arm,
        seed=int(cell.seed),
        draw_id=int(cell.draw_id),
        config_path=config,
        env_settings=settings,
        scenario_id=EVAL_SCENARIO_KEY,
        choose_action_factory=factory,
        engine_seed=ENGINE_SEED,
        created=created,
    )
    return cell_payload(
        cell,
        episode=episode.as_record(),
        series=series,
        checkpoint=identity,
        demand=demand,
        target_rtg=target,
        device=device,
        code=_code_provenance(),
        canary_seconds=canary_seconds,
        torch_num_threads=torch.get_num_threads(),
    )


def write_chunk(payload: Mapping[str, Any], *, output_root: str | Path) -> Path:
    """Atomic tmp-then-replace inside ``cells/``: a chunk is whole or absent.

    The chunk is validated against the declared cell its name names BEFORE anything is touched, and ``cells/`` must
    already exist -- the driver makes it after the token; nothing here creates a directory.
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get("name"), str):
        raise ValueError("a chunk is an object naming its cell")
    cell = campaign_cell_by_name(payload["name"])
    validate_chunk(payload, cell)
    directory = cells_dir(output_root)
    if not directory.is_dir():
        raise ValueError(f"{directory} does not exist; the driver creates it after the token, and nothing is created here")
    destination = chunk_path(output_root, cell)
    assert_writable(destination, output_root, entries=CAMPAIGN_OUTPUT_ENTRIES)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


_CHANGED_SINCE: dict[str, list[str]] = {}


def code_changed_since(commit: str) -> list[str]:
    """Paths outside ``docs/`` that differ between *commit* and this module tree's HEAD (J1(c)); git failing raises.

    J1(c): commits may differ by documentation only -- a chunk rolled at a commit that differs from HEAD by ``docs/``
    alone is the same computation.  A commit git cannot resolve RAISES: that is an environment failure, not a verdict
    about the chunk, and the campaign stops rather than silently re-rolling (``offline.transfer_curve``'s rule).
    """
    head = _git_in_module_tree("rev-parse", "HEAD")
    key = f"{commit}..{head.stdout.strip()}"
    if key in _CHANGED_SINCE:
        return list(_CHANGED_SINCE[key])
    result = _git_in_module_tree("diff", "--name-only", str(commit), "HEAD")
    if head.returncode != 0 or result.returncode != 0:
        raise RuntimeError(
            f"git diff --name-only {commit} HEAD failed in the code tree (exit {result.returncode}: "
            f"{result.stderr.strip()[:160]!r}); a chunk whose commit cannot be resolved is not a verdict about the chunk"
        )
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("docs/")]
    _CHANGED_SINCE[key] = changed
    return list(changed)


def chunk_is_reusable(
    payload: Any,
    *,
    cell: CampaignCell,
    output_root: str | Path,
    draws_root: str | Path,
    data_dir: str | Path,
    identity: Mapping[str, Any] | None = None,
    demand: Mapping[str, Any] | None = None,
) -> bool:
    """May a restart SKIP this cell?  Only on evidence re-derived from disk; anything unreadable is ``False``.

    The chunk validates as *cell*'s; its commit differs from HEAD by ``docs/`` only (J1(c)); its checkpoint path and
    sha256 are the ones :func:`checkpoint_for_cell` re-derives now; its demand digests are the draw's files' now.
    A commit git cannot resolve raises instead (:func:`code_changed_since`).  *identity* and *demand* may be passed
    precomputed by the caller, which derived them by the same calls.
    """
    if not isinstance(payload, Mapping):
        return False
    try:
        validate_chunk(payload, cell)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if code_changed_since(str(payload["code_commit"])):
        return False
    try:
        checkpoint = identity if identity is not None else checkpoint_for_cell(
            cell, output_root=output_root, data_dir=data_dir
        )
        if (payload["checkpoint"]["path"], payload["checkpoint"]["sha256"]) != (checkpoint["path"], checkpoint["sha256"]):
            return False
        draw = demand if demand is not None else draw_demand_identity(cell.draw_id, draws_root=draws_root)
        if (payload["demand"]["config_sha256"], payload["demand"]["flow_sha256"]) != (
            draw["config_sha256"], draw["flow_sha256"]
        ):
            return False
    except (KeyError, TypeError, ValueError, AttributeError, OSError):
        return False
    return True


def move_aside(path: str | Path) -> Path:
    """Move an unusable chunk into ``cells/failed/``, never overwriting (the suffix grows)."""
    source = Path(path)
    destination = source.parent / FAILED_CHUNKS_DIRNAME / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{source.stem}.{suffix}{source.suffix}")
    source.replace(destination)
    return destination


def _pin_one_thread() -> None:
    """A pool worker's initializer: one torch thread (plan §10), whatever the parent had."""
    torch.set_num_threads(1)


def _campaign_worker(task: tuple[CampaignCell, dict[str, Any]]) -> dict[str, Any]:
    """One cell in one spawned process: rolled, validated, written.  A failure is RETURNED, never raised into the pool:
    one broken cell must not take the stage down, and the caller decides."""
    cell, kwargs = task
    try:
        payload = run_campaign_cell(
            cell,
            output_root=kwargs["output_root"],
            corpus_root=kwargs["corpus_root"],
            draws_root=kwargs["draws_root"],
            data_dir=kwargs["data_dir"],
            canary_seconds=kwargs["canary_seconds"],
        )
        path = write_chunk(payload, output_root=kwargs["output_root"])
        return {"name": path.name, "ok": True, "seconds": payload["episode"]["seconds"], "error": None}
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        return {"name": cell.name, "ok": False, "seconds": None, "error": f"{type(error).__name__}: {error}"}


def run_campaign_stage(
    *,
    stage: str,
    output_root: str | Path,
    corpus_root: str | Path,
    draws_root: str | Path,
    data_dir: str | Path,
    canary_seconds: float | None,
    workers: int = CAMPAIGN_WORKERS,
    worker: Any = None,
) -> dict[str, Any]:
    """Every declared cell of *stage* not already on disk as a reusable chunk, in a spawn pool.

    The sweep stage refuses unless A26(c)'s gate record exists, passed, and still names the 500 reference chunks as
    they are on disk (:func:`verify_gate_record`) -- the second line of the gate's stop.  The skip decision is Python's
    (:func:`chunk_is_reusable`); a chunk that exists and is not reusable is moved aside, never overwritten.  ``spawn``,
    each worker pinned to one torch thread; it prints each cell's NAME and whether it stood, never a number.

    A HUNG cell (Amendment D.2: CityFlow's destructor race at an episode's end, DEFERRED 104): the results are read with
    ``next(timeout=STAGE_RESULT_TIMEOUT_S)``; a silence that long terminates and joins the pool, moves every
    ``.cell_*.json.<pid>.tmp`` a killed worker left to ``cells/failed/``, prints each cell with neither a result nor a
    chunk as ``HUNG (round n): re-rolled`` and rolls those cells in a NEW pool of the same size and initializer.  A cell
    that hung in all :data:`STAGE_HANG_ROUNDS` rounds is a failure, ``hung 3 times``.  A cell that returned ``ok False``
    is a failure, never re-rolled.  ``n_hung`` counts the cells that hung at least once, ``hang_rounds`` the rounds that
    ended in a timeout.
    """
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} is not one of {list(STAGES)}")
    if isinstance(workers, bool) or int(workers) < 1:
        raise ValueError(f"the pool needs at least one worker, got {workers!r}")
    root = Path(output_root)
    if stage == "sweep":
        verify_gate_record(output_root=root, data_dir=data_dir)
    directory = cells_dir(root)
    if not directory.is_dir():
        raise ValueError(f"{directory} does not exist; the driver creates it after the token")
    declared = declared_campaign_cells(stage)
    # The INPUTS first, each once: a checkpoint or a draw that cannot be verified refuses the whole stage before any
    # chunk is touched -- it is an input's fault, never a chunk's, and no chunk is moved aside for it.
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in declared:
        key = _checkpoint_key(cell)
        if key not in identities:
            identities[key] = checkpoint_for_cell(cell, output_root=root, data_dir=data_dir)
    demands = {draw: draw_demand_identity(draw, draws_root=draws_root) for draw in HELD_OUT_DRAWS}
    todo: list[CampaignCell] = []
    reused = 0
    for cell in declared:
        path = chunk_path(root, cell)
        if path.exists() or path.is_symlink():
            try:
                payload = json.loads(path.read_bytes())
            except (ValueError, OSError):
                payload = None
            if chunk_is_reusable(
                payload, cell=cell, output_root=root, draws_root=draws_root, data_dir=data_dir,
                identity=identities[_checkpoint_key(cell)], demand=demands[cell.draw_id],
            ):
                reused += 1
                continue
            move_aside(path)
        todo.append(cell)
    kwargs = {
        "output_root": str(root),
        "corpus_root": str(corpus_root),
        "draws_root": str(draws_root),
        "data_dir": str(data_dir),
        "canary_seconds": canary_seconds,
    }
    results: list[dict[str, Any]] = []
    hung_ever: set[str] = set()
    hang_rounds = 0
    started = time.perf_counter()
    pending = list(todo)
    for round_number in range(1, STAGE_HANG_ROUNDS + 1):
        if not pending:
            break
        round_results, timed_out = _roll_round(pending, worker or _campaign_worker, kwargs, workers=int(workers))
        results.extend(round_results)
        if not timed_out:
            break
        hang_rounds += 1
        # The pool is terminated and joined: a worker killed mid-write leaves a partial write, never a chunk.
        for leftover in sorted(directory.glob(".cell_*.json.*.tmp")):
            move_aside(leftover)
        answered = {str(result["name"]) for result in round_results}
        hung = [cell for cell in pending if cell.name not in answered and not chunk_path(root, cell).exists()]
        hung_ever.update(cell.name for cell in hung)
        if round_number < STAGE_HANG_ROUNDS:
            for cell in hung:
                print(f"  {cell.name} HUNG (round {round_number}): re-rolled", flush=True)
            pending = hung
            continue
        for cell in hung:
            failure = {"name": cell.name, "ok": False, "seconds": None, "error": f"hung {STAGE_HANG_ROUNDS} times"}
            results.append(failure)
            print(f"  {cell.name} FAILED {failure['error']}", flush=True)
    failures = [result for result in results if not result["ok"]]
    return {
        "stage": stage,
        "n_declared": len(declared),
        "n_reused": reused,
        "n_rolled": len(results) - len(failures),
        "n_failed": len(failures),
        "n_hung": len(hung_ever),
        "hang_rounds": hang_rounds,
        "failures": failures,
        "wall_seconds": time.perf_counter() - started,
    }


def _roll_round(
    cells: Sequence[CampaignCell], worker: Any, kwargs: Mapping[str, Any], *, workers: int
) -> tuple[list[dict[str, Any]], bool]:
    """One round of a stage: *cells* in a NEW spawn pool, each worker pinned to one torch thread; every result printed
    as it arrives.  Returns the results and whether the round ended in a silence of ``STAGE_RESULT_TIMEOUT_S`` -- in
    which case the pool has been terminated and joined, its hung workers killed (Amendment D.2, D.2.2(1))."""
    from multiprocessing import TimeoutError as PoolTimeout
    from multiprocessing import get_context

    results: list[dict[str, Any]] = []
    context = get_context("spawn")
    with context.Pool(processes=int(workers), initializer=_pin_one_thread) as pool:
        arriving = pool.imap_unordered(worker, [(cell, dict(kwargs)) for cell in cells])
        for _cell in cells:
            try:
                result = arriving.next(timeout=STAGE_RESULT_TIMEOUT_S)
            except PoolTimeout:
                pool.terminate()
                pool.join()
                return results, True
            results.append(result)
            status = "ok" if result["ok"] else f"FAILED {result['error']}"
            print(f"  {result['name']} {status}", flush=True)
    return results, False


def load_reference_rows(data_dir: str | Path) -> dict[str, Any]:
    """``docs/data/p4_k20_att_engine_rows.json`` at :data:`REFERENCE_ROWS_SHA256`, the digest BEFORE the parse.

    Then: its format, and exactly one row per (seed, draw) of the five seeds on the 100 held-out draws.
    """
    path = Path(data_dir) / REFERENCE_ROWS_NAME
    if not path.is_file():
        raise ValueError(f"{path} is absent: A26(c)'s gate compares against the rows C1b committed")
    digest = _sha256_file(path)
    if digest != REFERENCE_ROWS_SHA256:
        raise ValueError(
            f"{path}: sha256 {digest} is not the pinned {REFERENCE_ROWS_SHA256} (docs/data/{REFERENCE_ROWS_NAME} as "
            "committed at C1b); rows at another digest are never the gate's"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get("format_version") != REFERENCE_ROWS_FORMAT_VERSION:
        raise ValueError(f"{path} is not a {REFERENCE_ROWS_FORMAT_VERSION} file")
    rows = payload.get("rows")
    pairs = sorted((int(row["seed"]), int(row["draw_id"])) for row in (rows or []))
    if pairs != sorted((seed, draw) for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS):
        raise ValueError(f"{path} does not hold exactly one row per (seed, draw) of the 500")
    return payload


def _gated_cells() -> list[CampaignCell]:
    return [cell for cell in declared_campaign_cells("reference") if cell.subject == GATED_SUBJECT]


def reference_gate(*, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """A26(c)'s gate: the 500 ``ref_mappo1000_k20`` chunks against the committed rows under ``==``.  Writes nothing.

    A missing chunk REFUSES (the gate compares a complete set); every present chunk is validated; each chunk's
    ``att_engine`` and ``att_ours`` are compared with its (seed, draw)'s row under ``==``.  The verdict names the cells
    and fields that differ, never a value, and records each compared chunk's sha256, so the record can later be
    checked against the chunks as they are.
    """
    root = Path(output_root)
    rows = load_reference_rows(data_dir)
    by_pair = {(int(row["seed"]), int(row["draw_id"])): row for row in rows["rows"]}
    cells = _gated_cells()
    missing = [cell.name for cell in cells if not chunk_path(root, cell).is_file()]
    if missing:
        raise ValueError(
            f"the reference gate compares a complete set only: {len(missing)} of the {len(cells)} {GATED_ARM} chunks "
            f"are absent ({missing[:5]}); nothing is written"
        )
    differing: list[dict[str, Any]] = []
    chunks: list[dict[str, str]] = []
    for cell in cells:
        path = chunk_path(root, cell)
        payload = json.loads(path.read_bytes())
        validate_chunk(payload, cell)
        row = by_pair[(cell.seed, cell.draw_id)]
        fields = [name for name in ATT_DEFINITIONS if payload["episode"][name] != row[name]]
        if fields:
            differing.append({"cell": cell.name, "fields": fields})
        chunks.append({"cell": cell.name, "sha256": _sha256_file(path)})
    return {
        "format_version": GATE_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A26(c) as corrected by A26.1(b); BRIEF_42 C3",
        "rule": (
            f"every {GATED_ARM} chunk's att_engine and att_ours == the committed row of its (seed, draw); a single "
            "difference stops the campaign before any sweep cell is rolled"
        ),
        "arm": GATED_ARM,
        "compared": list(ATT_DEFINITIONS),
        "rows": {"name": REFERENCE_ROWS_NAME, "sha256": REFERENCE_ROWS_SHA256},
        "n_declared": len(cells),
        "n_compared": len(chunks),
        "differing": differing,
        "passed": not differing and len(chunks) == len(cells),
        "chunks": chunks,
    }


def write_gate_record(output_root: str | Path, verdict: Mapping[str, Any]) -> Path:
    """A PASSING verdict, written once; a failing one is refused."""
    if (
        verdict.get("format_version") != GATE_FORMAT_VERSION
        or verdict.get("passed") is not True
        or verdict.get("differing")
    ):
        raise ValueError(
            "only a passing reference_gate verdict is written: a failed gate writes NOTHING and the campaign stops "
            "before any sweep cell (A26(c))"
        )
    path = gate_record_path(output_root)
    _write_campaign_text_once(json.dumps(dict(verdict), indent=2, sort_keys=True) + "\n", path, output_root)
    return path


def verify_gate_record(*, output_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """The written gate record, re-verified against the rows' pin and the 500 chunks as they are on disk now."""
    root = Path(output_root)
    path = gate_record_path(root)
    if not path.is_file():
        raise ValueError(
            f"{path} is absent: no sweep cell is rolled, and no report written, before A26(c)'s reference gate has "
            "passed and its record is written"
        )
    record = json.loads(path.read_bytes())
    if (
        not isinstance(record, dict)
        or record.get("format_version") != GATE_FORMAT_VERSION
        or record.get("passed") is not True
        or record.get("differing")
    ):
        raise ValueError(f"{path} is not a passing reference_gate record")
    if (record.get("rows") or {}).get("sha256") != REFERENCE_ROWS_SHA256:
        raise ValueError(f"{path}: the reference_gate record was written against rows at another digest")
    cells = _gated_cells()
    recorded = {str(entry["cell"]): str(entry["sha256"]) for entry in record.get("chunks") or []}
    if sorted(recorded) != sorted(cell.name for cell in cells):
        raise ValueError(f"{path} does not name exactly the {len(cells)} {GATED_ARM} chunks the reference_gate compared")
    for cell in cells:
        chunk = chunk_path(root, cell)
        if not chunk.is_file() or _sha256_file(chunk) != recorded[cell.name]:
            raise ValueError(
                f"{chunk.name} is not the chunk the reference_gate compared (it is absent or its digest moved since the "
                "gate); the gate's verdict no longer describes the chunks on disk"
            )
    return record


def reroll_cells() -> tuple[CampaignCell, ...]:
    """Amendment A, Q2: the campaign's OWN fifteen ``ref_mappo1000_k20`` cells on :data:`REROLL_DRAWS`."""
    return tuple(cell for cell in _gated_cells() if cell.draw_id in REROLL_DRAWS)


def _reroll_worker(task: tuple[CampaignCell, dict[str, Any]]) -> dict[str, Any]:
    """One re-roll in one spawned process, through :func:`run_campaign_cell` itself; a failure returned, not raised."""
    cell, kwargs = task
    try:
        payload = run_campaign_cell(
            cell,
            output_root=kwargs["output_root"],
            corpus_root=kwargs["corpus_root"],
            draws_root=kwargs["draws_root"],
            data_dir=kwargs["data_dir"],
            canary_seconds=kwargs["canary_seconds"],
        )
        return {"name": cell.name, "ok": True, "payload": payload, "error": None}
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        return {"name": cell.name, "ok": False, "payload": None, "error": f"{type(error).__name__}: {error}"}


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """A fenced record: created exclusively, never overwriting."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def run_reference_reroll_check(
    *,
    output_root: str | Path,
    corpus_root: str | Path,
    draws_root: str | Path,
    data_dir: str | Path,
    canary_seconds: float | None,
    workers: int = CAMPAIGN_WORKERS,
    worker: Any = None,
) -> dict[str, Any]:
    """The fifteen cells rolled at THIS commit through the campaign's own cell, compared with the rows; fenced.

    Amendment A, Q2 (with A.1: on CUDA): BEFORE the token.  The rows are loaded first, at their pin; the fifteen roll in
    ONE spawn pool through :func:`run_campaign_cell`; nothing is written until every roll has returned.  A roll that
    FAILED is not a verdict: the check raises naming the exception TYPES only and parks the messages in
    ``failures.json`` under the fence (a message can carry a value).  Otherwise one record per cell (the chunk under the
    key ``fenced``, never named or placed like a campaign chunk: ``g2/`` is never read by the gate or the report) and
    ``verdict.json``.  The returned lines -- MATCH, or NO MATCH and the differing field NAMES -- are what the driver
    prints and counts; never a value.

    A HUNG roll (Amendment D.2, D.2.2(2)): the results are read with ``next(timeout=STAGE_RESULT_TIMEOUT_S)``, as they
    arrive, and paired with their cells by NAME; a silence that long terminates and joins the pool, and every cell
    without a result is a FAILED roll (``HungRoll``) -- never retried here: the start is cheap and the author restarts.
    """
    from multiprocessing import TimeoutError as PoolTimeout
    from multiprocessing import get_context

    rows = load_reference_rows(data_dir)
    by_pair = {(int(row["seed"]), int(row["draw_id"])): row for row in rows["rows"]}
    cells = reroll_cells()
    if isinstance(workers, bool) or int(workers) < 1:
        raise ValueError(f"the pool needs at least one worker, got {workers!r}")
    root = Path(output_root)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = g2_dir(root) / f"reference_reroll_check_{stamp}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; a re-roll record is never overwritten")
    assert_writable(run_dir, root, entries=CAMPAIGN_OUTPUT_ENTRIES)
    kwargs = {
        "output_root": str(root),
        "corpus_root": str(corpus_root),
        "draws_root": str(draws_root),
        "data_dir": str(data_dir),
        "canary_seconds": canary_seconds,
    }
    context = get_context("spawn")
    arrived: list[dict[str, Any]] = []
    with context.Pool(processes=min(int(workers), len(cells)), initializer=_pin_one_thread) as pool:
        arriving = pool.imap_unordered(worker or _reroll_worker, [(cell, kwargs) for cell in cells])
        for _cell in cells:
            try:
                arrived.append(arriving.next(timeout=STAGE_RESULT_TIMEOUT_S))
            except PoolTimeout:
                pool.terminate()
                pool.join()
                break
    by_name = {str(result["name"]): result for result in arrived}
    if len(by_name) != len(arrived) or not set(by_name) <= {cell.name for cell in cells}:
        raise AssertionError("the rolls came back paired with other cells; refusing to compare them")
    hung_roll = {"ok": False, "payload": None,
                 "error": f"HungRoll: no result within {STAGE_RESULT_TIMEOUT_S:.0f} s; the pool was terminated "
                          "(DEFERRED 104)"}
    results = [by_name.get(cell.name) or {"name": cell.name, **hung_roll} for cell in cells]
    failures = [result for result in results if not result["ok"]]
    if failures:
        run_dir.mkdir(parents=True)
        _write_json_new(
            run_dir / "failures.json",
            {
                "format_version": REROLL_FORMAT_VERSION,
                "fenced": {"failures": [{"cell": result["name"], "error": result["error"]} for result in failures]},
            },
        )
        kinds = sorted({str(result["error"]).split(":", 1)[0] for result in failures})
        raise RuntimeError(
            f"reference_reroll_check could not run: {len(failures)} of {len(results)} roll(s) failed ({kinds}); no "
            f"verdict is invented. The messages are FENCED in {run_dir / 'failures.json'}"
        )
    lines: list[str] = []
    verdicts: list[dict[str, Any]] = []
    for cell, result in zip(cells, results):
        payload = result["payload"]
        validate_chunk(payload, cell)
        row = by_pair[(cell.seed, cell.draw_id)]
        differing = [name for name in ATT_DEFINITIONS if payload["episode"][name] != row[name]]
        if differing:
            lines.append(f"reference_reroll_check NO MATCH {cell.name} " + " ".join(differing))
        else:
            lines.append(f"reference_reroll_check MATCH {cell.name}")
        verdicts.append({"cell": cell.name, "match": not differing, "differing": differing})
    run_dir.mkdir(parents=True)
    for cell, result in zip(cells, results):
        _write_json_new(run_dir / cell.name, {"format_version": REROLL_FORMAT_VERSION, "fenced": result["payload"]})
    n_match = sum(1 for verdict in verdicts if verdict["match"])
    _write_json_new(
        run_dir / "verdict.json",
        {
            "format_version": REROLL_FORMAT_VERSION,
            "registered_in": "BRIEF_42 Amendment A, Q2, with Amendment A.1 (the device: CUDA)",
            "rule": "each re-rolled cell's att_engine and att_ours == the committed row of its (seed, draw)",
            "rows": {"name": REFERENCE_ROWS_NAME, "sha256": REFERENCE_ROWS_SHA256},
            "device": EVAL_DEVICE,
            "canary_seconds": canary_seconds,
            "cells": verdicts,
            "n_cells": len(cells),
            "n_match": n_match,
            "lines": lines,
        },
    )
    return {"lines": lines, "n_match": n_match, "run_dir": run_dir}


def undeclared_chunk_names(output_root: str | Path) -> list[str]:
    """Files in ``cells/`` that are neither a declared chunk nor one of :data:`CELLS_DIR_NON_CHUNKS`."""
    directory = cells_dir(output_root)
    if not directory.is_dir():
        return []
    declared = {cell.name for cell in declared_campaign_cells()}
    return sorted(
        path.name for path in directory.iterdir() if path.name not in declared and path.name not in CELLS_DIR_NON_CHUNKS
    )


def unresolvable_chunk_commits(output_root: str | Path) -> list[dict[str, str]]:
    """Chunks whose recorded commit git cannot resolve in this module tree (an unreadable chunk is left to the stage,
    which moves it aside)."""
    directory = cells_dir(output_root)
    if not directory.is_dir():
        return []
    known: dict[str, bool] = {}
    found: list[dict[str, str]] = []
    for cell in declared_campaign_cells():
        path = directory / cell.name
        if not path.is_file():
            continue
        try:
            commit = str(json.loads(path.read_bytes()).get("code_commit"))
        except (ValueError, OSError, AttributeError):
            continue
        if commit not in known:
            known[commit] = _git_in_module_tree("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0
        if not known[commit]:
            found.append({"chunk": cell.name, "code_commit": commit})
    return found


def _p8_4b_mix50_cells(output_root: str | Path) -> dict[tuple[int, int], dict[str, Any]]:
    """P8.4b's 500 ``dt@mix50`` cells by (seed, draw): each present, naming its own pair, both definitions."""
    directory = Path(output_root) / REDERIVATION_DIRNAME
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in TRAINING_SEEDS:
        for draw in HELD_OUT_DRAWS:
            path = directory / cell_file_name(
                CellKey(scenario=EVAL_SCENARIO, arm=P8_4B_MIX50_ARM, seed=int(seed), draw_id=int(draw))
            )
            if not path.is_file():
                raise ValueError(f"{path} is absent: the mix50 reference arm is compared with P8.4b's 500 cells (A26(c))")
            cell = json.loads(path.read_bytes())
            if (cell.get("arm"), cell.get("seed"), cell.get("draw_id")) != (P8_4B_MIX50_ARM, seed, draw):
                raise ValueError(f"{path} does not name ({P8_4B_MIX50_ARM}, {seed}, {draw})")
            found[(seed, draw)] = {
                "att_engine": cell["att_engine"], "att_ours": cell["att_ours"], "sha256": _sha256_file(path)
            }
    return found


def check_campaign_inputs(
    *, output_root: str | Path, corpus_root: str | Path, draws_root: str | Path, data_dir: str | Path
) -> dict[str, Any]:
    """Every campaign input by digest before the canary and the token; CUDA and the device budget.

    The rows at their pin; the training record at its pin (unset: refused); the ten published checkpoints, the record's
    K = 20 references and the rows' checkpoints all the same files; the sixty through the record and their manifest;
    both subjects' evaluation settings EQUAL to P8.4b's (plan A7); the 100 held-out draws; P8.4b's 500 ``mix50`` cells
    and P5.3b's committed means (the reported ``mix50`` comparison); then CUDA and free device memory for the twelve
    workers.  Writes nothing.
    """
    root = Path(output_root)
    data = Path(data_dir)
    rows = load_reference_rows(data)
    record = load_train_record(data)
    published = {
        (subject, seed): published_checkpoint(subject, seed, output_root=root, data_dir=data)
        for subject in SUBJECTS
        for seed in TRAINING_SEEDS
    }
    for seed in TRAINING_SEEDS:
        if (rows.get("checkpoints") or {}).get(str(seed), {}).get("sha256") != published[("mappo1000", seed)]["sha256"]:
            raise ValueError(f"the reference rows name another checkpoint of seed {seed} than P4's published one")
    k20 = ((record.get("k20_reproduction") or {}).get("record") or {}).get("subjects") or {}
    for subject in SUBJECTS:
        entries = {int(entry["seed"]): entry for entry in (k20.get(subject) or {}).get("seeds", [])}
        for seed in TRAINING_SEEDS:
            entry = entries.get(seed) or {}
            expected = published[(subject, seed)]
            if (entry.get("reference"), entry.get("reference_sha256")) != (expected["path"], expected["sha256"]):
                raise ValueError(f"the training record's K = 20 reference of {subject} seed {seed} is not the published file")
    sixty = {run.name: sweep_checkpoint(run, output_root=root, data_dir=data) for run in registered_runs()}
    for subject in SUBJECTS:
        ours = evaluation_env_settings(subject, corpus_root)
        theirs = _rederivation_settings(subject, corpus_root)
        if ours != theirs:
            raise ValueError(
                f"{subject}: the evaluation env settings differ from P8.4b's; the reference rows were rolled under "
                "P8.4b's, and the gate would compare two different episodes"
            )
    for draw in HELD_OUT_DRAWS:
        draw_demand_identity(draw, draws_root=draws_root)
    _p8_4b_mix50_cells(root)
    p5_3b = _read_json(data / P5_3B_ARTIFACT_NAME, "P5.3b's artifact carries the mix50 reference means")
    if "mix50" not in (p5_3b.get("reference_dt_cells") or {}):
        raise ValueError(f"{P5_3B_ARTIFACT_NAME} carries no reference_dt_cells.mix50")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available; A26.1(b) evaluates the DT on this GPU, as the reference rows were produced")
    free_bytes, _total = torch.cuda.mem_get_info()
    free_mib = free_bytes / 2**20
    if free_mib < EVAL_DEVICE_BUDGET_MIB:
        raise ValueError(
            f"{free_mib:.0f} MiB free on the device, below the {EVAL_DEVICE_BUDGET_MIB} MiB budget for "
            f"{CAMPAIGN_WORKERS} workers"
        )
    return {
        "rows_sha256": REFERENCE_ROWS_SHA256,
        "train_record_sha256": TRAIN_RECORD_SHA256,
        "n_published": len(published),
        "n_runs": len(sixty),
        "cuda_free_mib": free_mib,
        "device_budget_mib": EVAL_DEVICE_BUDGET_MIB,
    }


# ======================================================================================================================
# C3 -- the statistic: the one-sided Wilcoxon, the rank contrast, Holm, the outcome and its registered sentence
# ======================================================================================================================


@dataclass(frozen=True)
class OneSidedWilcoxon:
    """A one-sided Wilcoxon signed-rank test of the differences against zero (normal approximation)."""

    alternative: str
    n_used: int
    n_zero: int
    w_plus: float
    expected: float
    variance: float
    z: float
    p_value: float


def one_sided_wilcoxon(differences: Sequence[float], *, alternative: str) -> OneSidedWilcoxon:
    """``less`` (T1: s < 0) or ``greater`` (T2, T3: > 0); zeros dropped, ties averaged, continuity-corrected.

    NEW code -- the repository's ``dt_gate.wilcoxon_signed_rank`` is two-sided -- with the same approximation, tie
    correction and continuity correction, the tail made explicit (``docs/plans/p5.3c.md`` §7, route A):
    ``W+`` = the sum of the average ranks (``dt_gate._average_ranks``) of the positive differences' magnitudes,
    ``E = n(n+1)/4``, ``Var = n(n+1)(2n+1)/24 - sum(t^3 - t)/48`` over the tie groups; ``less``:
    ``z = (W+ - E + 0.5)/sqrt(Var)``, ``p = Phi(z)``; ``greater``: ``z = (W+ - E - 0.5)/sqrt(Var)``, ``p = Phi(-z)``;
    ``Phi`` the repository's correctly rounded ``_normal_cdf``.  No non-zero difference (or a zero variance) gives
    ``p = 1``.  Whenever ``W+ < E`` and the two-sided ``p < 1``, ``less`` gives exactly half the two-sided p.
    """
    from offline.dt_gate import _average_ranks, _normal_cdf

    if alternative not in ("less", "greater"):
        raise ValueError(f"alternative {alternative!r} is not 'less' or 'greater': A26(d)'s three tests are one-sided")
    data = np.asarray(list(differences), dtype=np.float64)
    if data.size == 0:
        raise ValueError("one_sided_wilcoxon received no differences")
    n_zero = int(np.count_nonzero(data == 0.0))
    nonzero = data[data != 0.0]
    n_used = int(nonzero.size)
    if n_used == 0:
        return OneSidedWilcoxon(alternative, 0, n_zero, 0.0, 0.0, 0.0, 0.0, 1.0)
    ranks, ties = _average_ranks(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    expected = n_used * (n_used + 1) / 4.0
    variance = n_used * (n_used + 1) * (2 * n_used + 1) / 24.0 - sum(t**3 - t for t in ties) / 48.0
    if variance <= 0.0:
        return OneSidedWilcoxon(alternative, n_used, n_zero, w_plus, expected, variance, 0.0, 1.0)
    shift = 0.5 if alternative == "less" else -0.5
    z = (w_plus - expected + shift) / math.sqrt(variance)
    p_value = _normal_cdf(z) if alternative == "less" else _normal_cdf(-z)
    return OneSidedWilcoxon(alternative, n_used, n_zero, w_plus, expected, variance, z, p_value)


def rank_contrast(values_by_k: Mapping[int, float]) -> float:
    """One draw's ``s_d``: ``sum(c_j * A_d(K_j))`` over K ascending, evaluated left to right.

    The coefficients :data:`RANK_CONTRAST` sit on the levels' RANKS (A26(d): no spacing is chosen after the fact), so a
    curve linear in the rank gives ten times its slope per rank.  The sum starts from 0 and adds the five products in
    ascending K -- the order G6's independent route repeats.
    """
    if sorted(values_by_k) != list(CONTEXT_LENGTHS):
        raise ValueError(f"the contrast needs exactly the five levels {list(CONTEXT_LENGTHS)}, got {sorted(values_by_k)}")
    total: float = 0
    for coefficient, k in zip(RANK_CONTRAST, CONTEXT_LENGTHS):
        total = total + coefficient * float(values_by_k[k])
    return float(total)


def holm(p_values: Sequence[float], *, alpha: float = ALPHA) -> tuple[bool, ...]:
    """Holm's step-down within the family, in the family's order; equal p-values broken by that order.

    Sorted ascending, the i-th smallest (i = 0, 1, ...) is rejected when ``p <= alpha / (m - i)``; the first that is
    not stops the procedure and nothing after it is rejected (``PREREGISTRATION`` §8; A26(d)).
    """
    values = [float(p) for p in p_values]
    m = len(values)
    order = sorted(range(m), key=lambda index: (values[index], index))
    rejected = [False] * m
    for step, index in enumerate(order):
        if values[index] <= alpha / (m - step):
            rejected[index] = True
        else:
            break
    return tuple(rejected)


def outcome_of(rejected: Mapping[str, bool]) -> str:
    """A26(d)'s partition: ``"i"`` all three reject, ``"ii"`` T1 rejects and T2 or T3 does not, ``"iii"`` T1 does not."""
    missing = [name for name in FAMILY if name not in rejected]
    if missing:
        raise ValueError(f"the family's decisions lack {missing}")
    if not rejected["T1"]:
        return "iii"
    if rejected["T2"] and rejected["T3"]:
        return "i"
    return "ii"


def format_number(value: float) -> str:
    """Plan Q16: four decimals, the minus sign U+2212."""
    return f"{float(value):.4f}".replace("-", "−")


def format_ci(low: float, high: float) -> str:
    """Plan Q16: ``[low, high]``."""
    return f"[{format_number(low)}, {format_number(high)}]"


#: A26(d)'s three registered sentences, verbatim, their slots in braces: ``{S}`` is "S [CI]", ``{G1}`` / ``{G2}`` are
#: "G [CI]"; in (ii) ``{NOT_SHOWN}`` names the context length(s) whose test did not reject, ``{FALL}`` agrees with it,
#: and ``{CLAUSE}`` is A26.1(c)'s sentence when exactly one of T2 and T3 rejected, placed right after the registered
#: sentence's parenthesis (empty otherwise).  A test finds each, its slots aside, in PREREGISTRATION's A26 row.
SENTENCE_TEMPLATES: dict[str, str] = {
    "i": (
        "Confirmatory (A26): on the P4 scenario, performance improves with context length — the registered trend "
        "contrast is {S} — and both context lengths DataLight tested (K = 1, 2) fall short of the K = 20 plateau by "
        "more than the registered margin δ = 0.6263 s of ATT: K = 1 by {G1} and K = 2 by {G2}. Context length "
        "accounts for a difference of that size on this corpus — a measured difference in ATT, not an explanation of "
        "a failure."
    ),
    "ii": (
        "Confirmatory (A26): on the P4 scenario, performance improves with context length — the registered trend "
        "contrast is {S} — but it is not shown that {NOT_SHOWN} {FALL} short of the K = 20 plateau by more than the "
        "registered margin δ = 0.6263 s of ATT (K = 1: {G1}; K = 2: {G2}).{CLAUSE} Context length matters measurably "
        "on this corpus; a material shortfall is not demonstrated for {NOT_SHOWN}."
    ),
    "iii": (
        "Confirmatory (A26): on the P4 scenario, an improvement with context length is not detected — the registered "
        "trend contrast is {S} — so this sweep gives no support, on this corpus, to context length as the explanation "
        "of DataLight's negative result."
    ),
}
#: A26.1(c)'s registered words for the (ii) case in which exactly one of T2 and T3 rejected.
SHORTFALL_CLAUSE_TEMPLATE = (
    "K = {k} falls short of the K = 20 plateau by more than the registered margin δ = 0.6263 s of ATT."
)


def _estimate(entry: Mapping[str, Any]) -> str:
    return f"{format_number(entry['mean'])} {format_ci(entry['ci95_low'], entry['ci95_high'])}"


def outcome_sentence(outcome: str, *, family: Mapping[str, Any]) -> str:
    """A26(d)'s registered sentence for *outcome*, filled from *family*; (ii) with A26.1(c)'s clause where it applies.

    Refuses an outcome the family's own Holm decisions do not give: the sentence is chosen by the partition, never
    handed in.
    """
    if outcome not in SENTENCE_TEMPLATES:
        raise ValueError(f"outcome {outcome!r} is not one of A26(d)'s {sorted(SENTENCE_TEMPLATES)}")
    rejected = dict(family["holm"]["rejected"])
    if outcome_of(rejected) != outcome:
        raise ValueError(f"outcome {outcome!r} is not the one the family's Holm decisions give ({outcome_of(rejected)!r})")
    slots = {
        "S": _estimate(family["S"]),
        "G1": _estimate(family["G"]["1"]),
        "G2": _estimate(family["G"]["2"]),
        "NOT_SHOWN": "",
        "FALL": "",
        "CLAUSE": "",
    }
    if outcome == "ii":
        not_shown = [k for test, k in SHORTFALL_TESTS.items() if not rejected[test]]
        shown = [k for test, k in SHORTFALL_TESTS.items() if rejected[test]]
        slots["NOT_SHOWN"] = " and ".join(f"K = {k}" for k in not_shown)
        slots["FALL"] = "falls" if len(not_shown) == 1 else "fall"
        if len(shown) == 1:
            slots["CLAUSE"] = " " + SHORTFALL_CLAUSE_TEMPLATE.format(k=shown[0])
    return SENTENCE_TEMPLATES[outcome].format(**slots)


def per_draw_means(values: Mapping[tuple[int, int], float]) -> dict[int, float]:
    """``A_d``: per draw, the mean over the five training seeds, the values in ascending seed order, ``np.mean`` in
    float64.

    Refuses any draw whose seeds are not EXACTLY :data:`TRAINING_SEEDS`, and an empty input: a per-draw mean over
    another seed set -- shared by every draw or not -- is not the registered unit (§2, §8; Amendment D, D4.3(a)).
    """
    by_draw: dict[int, dict[int, float]] = {}
    for (seed, draw), value in values.items():
        by_draw.setdefault(int(draw), {})[int(seed)] = float(value)
    expected = tuple(sorted(int(seed) for seed in TRAINING_SEEDS))
    seed_sets = {tuple(sorted(seeds)) for seeds in by_draw.values()}
    if seed_sets != {expected}:
        raise ValueError(
            f"the per-draw unit is the mean over exactly the seeds {list(expected)}; these draws carry "
            f"{sorted(seed_sets)}"
        )
    return {
        draw: float(np.mean(np.asarray([seeds[s] for s in sorted(seeds)], dtype=np.float64)))
        for draw, seeds in sorted(by_draw.items())
    }


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    from offline.dt_gate import mean_ci95

    cell = mean_ci95(values)
    return {
        "n": cell.n,
        "mean": cell.mean,
        "std": cell.std,
        "ci95_half_width": cell.ci95,
        "ci95_low": cell.mean - cell.ci95,
        "ci95_high": cell.mean + cell.ci95,
    }


def _test_record(result: OneSidedWilcoxon, **extra: Any) -> dict[str, Any]:
    return {
        "alternative": result.alternative,
        "n_used": result.n_used,
        "n_zero": result.n_zero,
        "w_plus": result.w_plus,
        "expected": result.expected,
        "variance": result.variance,
        "z": result.z,
        "p_value": result.p_value,
        **extra,
    }


def confirmatory_family(
    levels: Mapping[int, Mapping[int, float]], *, delta: float = DELTA, alpha: float = ALPHA
) -> dict[str, Any]:
    """T1, T2, T3 on the per-draw means of the five levels, Holm, the outcome, S and G1, G2 with their CIs.

    *levels* maps each K to ``{draw: A_d(K)}``; every level must carry the same draws.  T1: ``less`` on ``s_d``
    (:func:`rank_contrast`); T2 / T3: ``greater`` on ``(A_d(K) - A_d(20)) - delta`` for K = 1 / 2, the subtraction in
    that order; S the mean ``s_d`` and G_K the mean ``A_d(K) - A_d(20)``, each with ``dt_gate.mean_ci95``'s 95% CI.
    """
    if sorted(levels) != list(CONTEXT_LENGTHS):
        raise ValueError(f"the family needs exactly the five levels {list(CONTEXT_LENGTHS)}, got {sorted(levels)}")
    draw_sets = {tuple(sorted(int(d) for d in levels[k])) for k in CONTEXT_LENGTHS}
    if len(draw_sets) != 1:
        raise ValueError("the five levels do not carry the same draws; every contrast is per draw")
    draws = list(next(iter(draw_sets)))
    if not draws:
        raise ValueError("the levels carry no draw")
    contrasts = [rank_contrast({k: levels[k][d] for k in CONTEXT_LENGTHS}) for d in draws]
    tests: dict[str, Any] = {
        "T1": _test_record(
            one_sided_wilcoxon(contrasts, alternative="less"),
            statistic="the per-draw rank contrast s_d; the alternative s < 0 (ATT falls with K)",
        )
    }
    gaps: dict[str, Any] = {}
    for test, k in SHORTFALL_TESTS.items():
        gap = [float(levels[k][d]) - float(levels[REFERENCE_K][d]) for d in draws]
        shifted = [(float(levels[k][d]) - float(levels[REFERENCE_K][d])) - delta for d in draws]
        tests[test] = _test_record(
            one_sided_wilcoxon(shifted, alternative="greater"),
            k=k,
            statistic=f"(A_d({k}) - A_d(20)) - delta; the alternative > 0 (K = {k} falls short by more than delta)",
        )
        gaps[str(k)] = _mean_ci(gap)
    p_values = [tests[name]["p_value"] for name in FAMILY]
    decisions = holm(p_values, alpha=alpha)
    rejected = dict(zip(FAMILY, decisions))
    order = sorted(range(len(FAMILY)), key=lambda index: (p_values[index], index))
    return {
        "levels": list(CONTEXT_LENGTHS),
        "contrast": list(RANK_CONTRAST),
        "delta": delta,
        "alpha": alpha,
        "n_draws": len(draws),
        "draw_ids": draws,
        "S": _mean_ci(contrasts),
        "G": gaps,
        "tests": tests,
        "holm": {
            "rule": "step-down: the i-th smallest p (i = 0, 1, 2) rejects if p <= alpha / (3 - i); the first that does "
                    "not stops; equal p-values in the family's order T1, T2, T3",
            "order": [FAMILY[index] for index in order],
            "thresholds": {FAMILY[index]: alpha / (len(FAMILY) - step) for step, index in enumerate(order)},
            "rejected": rejected,
        },
        "outcome": outcome_of(rejected),
    }


# ======================================================================================================================
# C3 -- the report: refusals first, then ONE write
# ======================================================================================================================


#: Amendment C, C2's columns: what each of the per-arm table's numbers is, read from the pinned record's run entries.
TRAINING_TABLE_COLUMNS: dict[str, str] = {
    "loop_seconds": "the training loop's WALL seconds (the record's loop_seconds): mean, min and max over the five "
                    "seeds, and each seed's",
    "ms_per_step": "the mean over the five seeds of the record's ms_per_step",
    "final_loss_per_target": "the record's final_loss -- the last step's training loss per supervised target -- over "
                             "the five seeds: mean, min and max",
    "supervised_targets_per_step": "the mean over the five seeds of the record's supervised_targets_per_step.mean",
    "first_window_loss": "the mean over the five seeds of the first of the record's twenty loss_per_supervised_target "
                         "window means",
}


def training_table(record: Mapping[str, Any]) -> dict[str, Any]:
    """Amendment C, C2's per-arm table, computed from the PINNED training record (C4.2: never retyped).

    Per registered arm, over its five seeds in ascending seed order (float64; ``np.mean`` sums five values left to
    right from zero): the columns :data:`TRAINING_TABLE_COLUMNS` names.  Beside it, C2's training-loss observation as
    the record gives it: per subject, whether the mean final loss falls strictly as K rises through the batch-64 arms;
    per equal-supervision arm, whether its mean lies below its subject's K = 20 arm.  An observation, never a finding.
    """
    runs = record["runs"]
    arms: dict[str, dict[str, Any]] = {}
    for arm in registered_arms():
        entries = [runs[f"{arm}_seed{seed}"] for seed in TRAINING_SEEDS]
        loop = np.asarray([entry["loop_seconds"] for entry in entries], dtype=np.float64)
        loss = np.asarray([entry["final_loss"] for entry in entries], dtype=np.float64)
        arms[arm] = {
            "loop_seconds": {
                "mean": float(loop.mean()), "min": float(loop.min()), "max": float(loop.max()),
                "seeds": {str(seed): entry["loop_seconds"] for seed, entry in zip(TRAINING_SEEDS, entries)},
            },
            "ms_per_step": float(np.mean(np.asarray([entry["ms_per_step"] for entry in entries], dtype=np.float64))),
            "final_loss_per_target": {"mean": float(loss.mean()), "min": float(loss.min()), "max": float(loss.max())},
            "supervised_targets_per_step": float(np.mean(np.asarray(
                [entry["supervised_targets_per_step"]["mean"] for entry in entries], dtype=np.float64
            ))),
            "first_window_loss": float(np.mean(np.asarray(
                [entry["loss_per_supervised_target"][0] for entry in entries], dtype=np.float64
            ))),
        }
    mean_loss = {arm: row["final_loss_per_target"]["mean"] for arm, row in arms.items()}
    falls = {
        subject: all(
            mean_loss[f"{subject}_k{shorter}_b{BASE_BATCH}"] > mean_loss[f"{subject}_k{longer}_b{BASE_BATCH}"]
            for shorter, longer in zip(CONTEXT_LENGTHS, CONTEXT_LENGTHS[1:])
        )
        for subject in SUBJECTS
    }
    equal_supervision = {}
    for owner, k, batch in EQUAL_SUPERVISION:
        arm, k20 = f"{owner}_k{k}_b{batch}", f"{owner}_k{REFERENCE_K}_b{BASE_BATCH}"
        equal_supervision[arm] = {"mean": mean_loss[arm], "k20_arm": k20, "k20_mean": mean_loss[k20],
                                  "below_k20": mean_loss[arm] < mean_loss[k20]}
    return {
        "label": "BRIEF_42 Amendment C, C2's per-arm training table, computed here from the PINNED training record "
                 "(C4.2: never retyped); reported beside loss_per_target and the observation below, never deciding",
        "record": {"name": TRAIN_RECORD_DATA_NAME, "sha256": TRAIN_RECORD_SHA256},
        "columns": dict(TRAINING_TABLE_COLUMNS),
        "arms": arms,
        "wall_time": (
            "Loop seconds are WALL TIME on a machine that was not always quiet (Amendment C, C2 names two stalls); the "
            f"trainings are step-based, {GRADIENT_STEPS:,} steps each. A compute table uses per-arm medians or minima "
            "and names the stalls, never these means."
        ),
        "observation": {
            "label": "An observation, not a finding (Amendment C, C2): these are training losses per supervised "
                     "target and say nothing about held-out performance; H4 is decided by T1-T3 on the held-out draws "
                     "and by nothing here.",
            "final_loss_falls_with_k": falls,
            "final_loss_falls_with_k_rule": "the mean final loss per target strictly lower at each longer K, batch "
                                            f"{BASE_BATCH}, K in {list(CONTEXT_LENGTHS)}",
            "equal_supervision": equal_supervision,
        },
    }


#: Amendment C, C3: the wording of the EQUAL K = 20 result (a ruling on wording, not on a number) -- (i), (ii), (iii).
K20_READING: tuple[str, ...] = (
    "The sweep's K = 20 arm is P4's model, and mix50's is P4.7's: the K = 20 reproduction is equal on every tensor for "
    "all ten (subject, seed) pairs (A26(b)'s measurement). H4's K = 20 point is P4's checkpoint re-evaluated, and the "
    "equal-supervision and K < 20 arms differ from P4 by K (and batch) alone, trained by the same code path that "
    "reproduced P4 bit for bit.",
    "A26(c)'s reference gate -- P4's five checkpoints re-evaluated == the committed rows -- therefore also pins the "
    "sweep's own K = 20 cells: any difference between the K = 20 arm's cells and the reference rows would be an "
    "evaluation-path difference, never a model difference.",
    "The fenced timing run's same-seed repeat and this reproduction, both equal, are a property of THIS GPU, driver and "
    "torch build under the registered regime, stated as such, not as determinism of the method.",
)
K20_READING_NOT_MADE = (
    "Amendment C, C3's reading is NOT made: it presumes all ten (subject, seed) pairs equal on every tensor and the "
    "fenced timing run's same-seed repeat equal, and the pinned record gives {n_equal} of ten pairs equal and the "
    "repeat {repeat}. A difference is reported with its magnitude (the seeds above) and stops nothing (A26(b))."
)


def k20_reading(record: Mapping[str, Any]) -> dict[str, Any]:
    """Amendment C, C3's wording of the K = 20 reproduction, made only when the pinned record shows what it presumes.

    A (subject, seed) pair counts as equal when the record's comparison of it says ``all_equal`` with no parameter
    differing; a pair the record does not carry counts as not equal.  The timing run's repeat counts as equal when the
    embedded timing record says both its file and its weights digests are equal.  All ten and the repeat equal ->
    :data:`K20_READING`; anything else -> one sentence saying the reading is not made, and why.
    """
    subjects = ((record.get("k20_reproduction") or {}).get("record") or {}).get("subjects") or {}
    n_equal = 0
    for subject in SUBJECTS:
        by_seed = {int(entry["seed"]): entry for entry in (subjects.get(subject) or {}).get("seeds") or []}
        for seed in TRAINING_SEEDS:
            entry = by_seed.get(seed) or {}
            if entry.get("all_equal") is True and entry.get("n_parameters_differing") == 0:
                n_equal += 1
    repeat = ((record.get("timing") or {}).get("record") or {}).get("repeat") or {}
    repeat_equal = repeat.get("file_sha256_equal") is True and repeat.get("weights_sha256_equal") is True
    made = n_equal == len(SUBJECTS) * len(TRAINING_SEEDS) and repeat_equal
    sentences = list(K20_READING) if made else [
        K20_READING_NOT_MADE.format(n_equal=n_equal, repeat="equal" if repeat_equal else "not equal or not recorded")
    ]
    return {
        "source": "BRIEF_42 Amendment C, C3 (a ruling on wording, not on a number)",
        "made": made,
        "n_pairs_equal": n_equal,
        "timing_repeat_equal": repeat_equal,
        "sentences": sentences,
    }


_ARTIFACT_LIMITS: tuple[str, ...] = (
    "A26(d): the sweep tests DataLight's context lengths only; their prompt (a single hardcoded return) and their "
    "multi-agent data are not ours, and no sentence here speaks of them. Any other reading is exploratory and decides "
    "nothing.",
    "A26(b): K = 1 is the registered degenerate anchor -- a one-step return-conditioned policy on the current state, the "
    "current return-to-go token and the absolute timestep embedding -- not pure behaviour cloning.",
    "A26(b): the budget is fixed at 40,000 gradient steps for every K, which loads the supervision against small K; the "
    "two equal-supervision arms measure that at K = 1 and 2 only, and they are exploratory.",
    "A26(b) as corrected by A26.1(a): the K = 20 reproduction is a measurement; were the sweep's K = 20 to differ from "
    "P4's, the sweep's own K = 20 arm is the sweep's reference and P4's checkpoints remain C1's published subject.",
    "A26(d): delta = 0.6263 is A6's value used as a superiority margin; its transfer from att_ours's scale to "
    "att_engine's is a stated choice (P5.3b: 0.1180 against 0.1226), and the att_ours computation is co-reported.",
    "A26(e): the expectations are this row's expectations, held or refuted, never folded into (d)'s decision.",
    "A26(a): mix50 is an exploratory subject; its results, its partition included, decide nothing about H4.",
    "A26's stated limits: one scenario; the confirmatory decision on one corpus, the registered one; a fixed budget, its "
    "sensitivity checked at K in {1, 2} only.",
    'A26(f), in its own words: "UNCHANGED: §3.1\'s primary metric and A11/A13/A15\'s pair, §5\'s seed rule, §6\'s '
    "leakage rules (no training touches a held-out draw), §8's estimator and multiplicity; A6's δ keeps its value and "
    'gains this second, superiority use; A25 (the scope) as registered beside this row."',
    "A26.1(b): the DT is evaluated on CUDA, on this GPU, as every published DT number in this repository is.",
    "A25: after H4 and the compute-and-latency table no new experiment is run; H2 was registered and not tested, and "
    "nothing here bears on it.",
)


def _expected_checkpoints(record: Mapping[str, Any]) -> dict[tuple[Any, ...], tuple[str, str]]:
    """Every checkpoint a chunk may name, from the PINNED training record: the sixty runs and the ten published K = 20
    references its reproduction record compared them with."""
    expected: dict[tuple[Any, ...], tuple[str, str]] = {}
    for name, entry in (record.get("runs") or {}).items():
        expected[("sweep", str(name))] = (str(entry["checkpoint"]), str(entry["checkpoint_sha256"]))
    subjects = ((record.get("k20_reproduction") or {}).get("record") or {}).get("subjects") or {}
    for subject, block in subjects.items():
        for entry in block.get("seeds") or []:
            expected[("reference", str(subject), int(entry["seed"]))] = (
                str(entry["reference"]), str(entry["reference_sha256"])
            )
    return expected


def _checkpoint_key(cell: CampaignCell) -> tuple[Any, ...]:
    return ("sweep", cell.run.name) if cell.run is not None else ("reference", cell.subject, cell.seed)


def build_context_sweep_artifact(
    *, output_root: str | Path, corpus_root: str | Path, draws_root: str | Path, data_dir: str | Path
) -> dict[str, Any]:
    """The artifact, from the complete set of chunks; every refusal precedes every computation.  Writes nothing.

    Refuses, in order: the rows at another digest; the training record unpinned or at another digest; no
    ``canary.json`` (re-checked, ``transfer_calibration``'s reader); no passing gate record, or one the chunks no longer
    match; a file in ``cells/`` no declared cell names; any declared chunk absent (no estimator on a partial set, A26(c));
    a chunk that does not validate, was rolled by other code (J1(c)), on more than one torch thread, at a checkpoint the
    pinned record does not name, or on another demand than its draw's; P8.4b's ``mix50`` cells or P5.3b's artifact
    absent.

    From the pinned record, beside the statistic and deciding nothing (Amendment C): ``training_table`` (C2's table,
    :func:`training_table`) and ``k20_reproduction.reading`` (C3's wording, :func:`k20_reading`).

    THE REDUCTION ORDER, for G6's independent route: ``A_d(K)`` is ``np.mean`` over the five seeds' values in
    ascending seed order (float64); ``s_d`` sums ``c_j * A_d(K_j)`` over K ascending, left to right from zero; T2 / T3
    test ``(A_d(K) - A_d(20)) - delta``; S and G_K use ``dt_gate.mean_ci95``.  No per-decision series is copied here.
    """
    from offline.dt_gate import wilcoxon_signed_rank
    from offline.transfer_calibration import _read_canary_record

    root = Path(output_root)
    data = Path(data_dir)
    load_reference_rows(data)
    record = load_train_record(data)
    canary = _read_canary_record(cells_dir(root))
    gate = verify_gate_record(output_root=root, data_dir=data)
    stray = undeclared_chunk_names(root)
    if stray:
        raise ValueError(f"{cells_dir(root)} holds files no declared cell names ({stray[:5]}); the report reads the "
                         "declared set only")
    declared = declared_campaign_cells()
    missing = [cell.name for cell in declared if not chunk_path(root, cell).is_file()]
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(declared)} declared chunks are absent ({missing[:5]}); no estimator is computed "
            "on a partial set (A26(c))"
        )
    expected = _expected_checkpoints(record)
    demands = {draw: draw_demand_identity(draw, draws_root=draws_root) for draw in HELD_OUT_DRAWS}
    values: dict[tuple[str, int, int], dict[str, float]] = {}
    commits: set[str] = set()
    for cell in declared:
        payload = json.loads(chunk_path(root, cell).read_bytes())
        validate_chunk(payload, cell)
        commit = str(payload["code_commit"])
        changed = code_changed_since(commit)
        if changed:
            raise ValueError(
                f"{cell.name} was rolled at {commit[:12]}, whose code differs from HEAD outside docs/ ({changed[:3]}): "
                "J1(c) -- commits may differ by documentation only"
            )
        if payload["torch_num_threads"] != 1:
            raise ValueError(f"{cell.name}: rolled with torch_num_threads {payload['torch_num_threads']}, not the "
                             "campaign's one thread per worker")
        want = expected.get(_checkpoint_key(cell))
        if want is None or (payload["checkpoint"]["path"], payload["checkpoint"]["sha256"]) != want:
            raise ValueError(
                f"{cell.name}: its checkpoint {payload['checkpoint']['path']} at {payload['checkpoint']['sha256'][:12]} "
                "is not the one the pinned training record names"
            )
        demand = demands[cell.draw_id]
        if (payload["demand"]["config_sha256"], payload["demand"]["flow_sha256"]) != (
            demand["config_sha256"], demand["flow_sha256"]
        ):
            raise ValueError(f"{cell.name}: rolled on another demand than draw {cell.draw_id}'s files")
        values[(cell.arm, cell.seed, cell.draw_id)] = {name: float(payload["episode"][name]) for name in ATT_DEFINITIONS}
        commits.add(commit)
    mix_cells = _p8_4b_mix50_cells(root)
    p5_3b_path = data / P5_3B_ARTIFACT_NAME
    p5_3b = _read_json(p5_3b_path, "P5.3b's artifact carries the mix50 reference means")

    draws = list(HELD_OUT_DRAWS)
    means_cache: dict[tuple[str, str], dict[int, float]] = {}

    def a_d(arm: str, definition: str) -> dict[int, float]:
        key = (arm, definition)
        if key not in means_cache:
            means_cache[key] = per_draw_means(
                {(seed, draw): values[(arm, seed, draw)][definition] for seed in TRAINING_SEEDS for draw in draws}
            )
        return means_cache[key]

    def level_arms(subject: str) -> dict[int, str]:
        """The arms a subject's five levels are read from, K ascending: the sweep's own batch-64 arms."""
        return {k: f"{subject}_k{k}_b{BASE_BATCH}" for k in CONTEXT_LENGTHS}

    def levels(subject: str, definition: str) -> dict[int, dict[int, float]]:
        return {k: a_d(arm, definition) for k, arm in level_arms(subject).items()}

    def paired(left: dict[int, float], right: dict[int, float]) -> dict[str, Any]:
        left_values = [left[draw] for draw in draws]
        right_values = [right[draw] for draw in draws]
        differences = [a - b for a, b in zip(left_values, right_values)]
        return {
            **_mean_ci(differences),
            "wilcoxon_two_sided_p": wilcoxon_signed_rank(left_values, right_values).p_value,
            "wins": sum(1 for d in differences if d < 0),
            "losses": sum(1 for d in differences if d > 0),
            "ties": sum(1 for d in differences if d == 0),
        }

    family = confirmatory_family(levels("mappo1000", PRIMARY_ATT))
    mix_family = confirmatory_family(levels("mix50", PRIMARY_ATT))

    pairwise: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    gaps: dict[str, Any] = {}
    plateau: dict[str, Any] = {}
    for subject in SUBJECTS:
        pairwise[subject], per_seed[subject], gaps[subject], plateau[subject] = {}, {}, {}, {}
        for definition in ATT_DEFINITIONS:
            level = levels(subject, definition)
            pairwise[subject][definition] = [
                {"contrast": f"K = {left} - K = {right}", "left_k": left, "right_k": right,
                 **paired(level[left], level[right])}
                for index, left in enumerate(CONTEXT_LENGTHS)
                for right in CONTEXT_LENGTHS[index + 1:]
            ]
            by_seed: dict[str, float] = {}
            for seed in TRAINING_SEEDS:
                contrasts = [
                    rank_contrast({k: values[(f"{subject}_k{k}_b{BASE_BATCH}", seed, draw)][definition]
                                   for k in CONTEXT_LENGTHS})
                    for draw in draws
                ]
                by_seed[str(seed)] = float(np.mean(np.asarray(contrasts, dtype=np.float64)))
            seed_values = np.asarray(list(by_seed.values()), dtype=np.float64)
            per_seed[subject][definition] = {
                "seeds": by_seed, "mean": float(seed_values.mean()), "sd_ddof1": float(seed_values.std(ddof=1))
            }
            gap = {str(k): _mean_ci([level[k][draw] - level[REFERENCE_K][draw] for draw in draws])
                   for k in CONTEXT_LENGTHS}
            gaps[subject][definition] = gap
            plateau[subject][definition] = {
                "point_estimate_within_delta": {
                    "smallest_k": next((k for k in CONTEXT_LENGTHS if gap[str(k)]["mean"] <= DELTA), None),
                    "rule": "the smallest K whose paired mean G_K = mean(A_d(K) - A_d(20)) <= delta",
                },
                "ci_within_plus_minus_delta": {
                    "smallest_k": next(
                        (k for k in CONTEXT_LENGTHS
                         if -DELTA <= gap[str(k)]["ci95_low"] and gap[str(k)]["ci95_high"] <= DELTA),
                        None,
                    ),
                    "rule": "the smallest K whose 95% CI of G_K lies within [-delta, +delta] (A6's equivalence rule)",
                },
            }

    base = f"mappo1000_k{REFERENCE_K}_b{BASE_BATCH}"
    equal_supervision = {
        "label": "EXPLORATORY: the equal-supervision secondary (A26(b)), never deciding",
        "contrasts": {
            f"{owner}_k{k}_b{batch} - {owner}_k{k}_b{BASE_BATCH}": {
                definition: paired(a_d(f"{owner}_k{k}_b{batch}", definition), a_d(f"{owner}_k{k}_b{BASE_BATCH}", definition))
                for definition in ATT_DEFINITIONS
            }
            for owner, k, batch in EQUAL_SUPERVISION
        },
        "against_k20": {
            f"{owner}_k{k}_b{batch}": {
                definition: paired(a_d(f"{owner}_k{k}_b{batch}", definition), a_d(base, definition))
                for definition in ATT_DEFINITIONS
            }
            for owner, k, batch in EQUAL_SUPERVISION
        },
    }
    rejected = family["holm"]["rejected"]
    expectations = {
        "mappo1000": {
            "registered": "T1 is expected to reject with a small effect, and T2 and T3 are expected NOT to reject: "
                          "outcome (ii) (A26(e))",
            "refuted_iff": "T2 and T3 both reject after Holm",
            "verdict": "refuted" if rejected["T2"] and rejected["T3"] else "held",
            "observed_outcome": family["outcome"],
            "note": "reported as A26(e)'s expectation, held or refuted, never folded into (d)'s decision",
        },
        "mix50": {
            "registered": "K in {1, 2} are expected to be materially below K = 20 (exploratory; A26(e))",
            "refuted_iff": "K = 2 within delta of K = 20: mix50's T3 does not reject after Holm within mix50's own "
                           "family (plan Q10)",
            "verdict": "refuted" if not mix_family["holm"]["rejected"]["T3"] else "held",
            "G2": mix_family["G"]["2"],
            "note": "reported as A26(e)'s expectation, held or refuted, never folded into (d)'s decision",
        },
    }
    k20_subjects = ((record.get("k20_reproduction") or {}).get("record") or {}).get("subjects") or {}
    k20: dict[str, Any] = {}
    for subject in SUBJECTS:
        block = k20_subjects.get(subject) or {}
        sweep_arm = f"{subject}_k{REFERENCE_K}_b{BASE_BATCH}"
        reference_arm = f"ref_{subject}_k{REFERENCE_K}"
        k20[subject] = {
            "all_equal": block.get("all_equal"),
            "n_seeds_equal": block.get("n_seeds_equal"),
            "seeds": [
                {key: entry.get(key) for key in ("seed", "all_equal", "n_parameters_differing", "largest_abs_difference",
                                                  "largest_abs_difference_parameter")}
                for entry in block.get("seeds") or []
            ],
            "sweep_k20_beside_the_published": {
                definition: {
                    **paired(a_d(sweep_arm, definition), a_d(reference_arm, definition)),
                    "n_cells_equal": sum(
                        1 for seed in TRAINING_SEEDS for draw in draws
                        if values[(sweep_arm, seed, draw)][definition] == values[(reference_arm, seed, draw)][definition]
                    ),
                }
                for definition in ATT_DEFINITIONS
            },
        }
    k20["reading"] = k20_reading(record)
    loss_per_target: dict[str, Any] = {}
    for arm in registered_arms():
        entries = [(seed, record["runs"][f"{arm}_seed{seed}"]) for seed in TRAINING_SEEDS]
        loss_per_target[arm] = {
            "seeds": {
                str(seed): {key: entry[key] for key in ("final_loss", "loss_per_supervised_target",
                                                         "supervised_targets_per_step")}
                for seed, entry in entries
            },
            "mean_final_loss": float(np.mean(np.asarray([entry["final_loss"] for _seed, entry in entries],
                                                        dtype=np.float64))),
        }
    reference_arm = f"ref_mix50_k{REFERENCE_K}"
    differing = [
        f"cell_{reference_arm}_seed{seed}_draw{draw}.json"
        for (seed, draw), cell in sorted(mix_cells.items())
        if any(values[(reference_arm, seed, draw)][name] != cell[name] for name in ATT_DEFINITIONS)
    ]
    committed_means = (p5_3b.get("reference_dt_cells") or {}).get("mix50") or {}
    p5_3b_means: dict[str, Any] = {}
    for name in ATT_DEFINITIONS:
        recomputed = float(np.mean(np.asarray(
            [values[(reference_arm, seed, draw)][name] for seed in TRAINING_SEEDS for draw in draws], dtype=np.float64
        )))
        committed = float(committed_means[f"{name}_mean"])
        p5_3b_means[name] = {"committed": committed, "recomputed": recomputed, "equal": recomputed == committed,
                             "difference": recomputed - committed}
    counts = Counter(arm for arm, _seed, _draw in values)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A26 as corrected by A26.1; BRIEF_42 C3 and Amendments A, A.1, B, B.1, C",
        "role": "H4's context-length sweep: the confirmatory family on mappo1000, and everything A26(d) reports beside it",
        "n_cells": len(declared),
        "counts": dict(sorted(counts.items())),
        "definitions": list(ATT_DEFINITIONS),
        "primary_definition": PRIMARY_ATT,
        "delta": {
            "value": DELTA,
            "source": "A6; offline.offline_baselines.DELTA_ATT; docs/data/p4_4_baselines.json comparisons.madt_vs_bc.delta",
            "use": "a paired ATT superiority margin (A26(d))",
        },
        "alpha": ALPHA,
        "contrast": {"coefficients": list(RANK_CONTRAST), "levels": list(CONTEXT_LENGTHS),
                     "on": "the levels' ranks, K ascending (A26(d))"},
        "reduction_order": {
            "A_d": "np.mean over the five seeds' values in ascending seed order, float64",
            "s_d": "sum of c_j * A_d(K_j) over K ascending, evaluated left to right from zero",
            "T2_T3": "(A_d(K) - A_d(20)) - delta, the subtraction in that order",
            "S_G": "offline.dt_gate.mean_ci95 (ddof 1, 1.96 sd / sqrt(n))",
        },
        "confirmatory": {
            "registered_in": "PREREGISTRATION A26(d) as corrected by A26.1(c)",
            "subject": "mappo1000",
            "definition": PRIMARY_ATT,
            # Amendment D, D4.3(c): the arms the family was fed, in K order, and the K = 20 level's -- the sweep's own.
            "arms": list(level_arms("mappo1000").values()),
            "k20_arm": level_arms("mappo1000")[REFERENCE_K],
            "family": family,
            "outcome": family["outcome"],
            "sentence": outcome_sentence(family["outcome"], family=family),
            "tests_not_rejected": [name for name in FAMILY if not family["holm"]["rejected"][name]],
        },
        "att_ours": {
            "label": "co-reported beside the confirmatory family, never deciding (A26(d))",
            "subject": "mappo1000",
            "definition": "att_ours",
            "family": confirmatory_family(levels("mappo1000", "att_ours")),
        },
        "exploratory_mix50": {
            "label": "EXPLORATORY: the whole design on mix50, its own three-way partition reported without deciding "
                     "anything (A26(a), (d))",
            "subject": "mix50",
            "definition": PRIMARY_ATT,
            "family": mix_family,
            "outcome_label": mix_family["outcome"],
            "att_ours": {"family": confirmatory_family(levels("mix50", "att_ours"))},
        },
        "per_draw_means": {
            arm: {definition: {str(draw): value for draw, value in a_d(arm, definition).items()}
                  for definition in ATT_DEFINITIONS}
            for arm in sorted(counts)
        },
        "pairwise": pairwise,
        "per_seed": per_seed,
        "gaps": gaps,
        "plateau": plateau,
        "equal_supervision": equal_supervision,
        "expectations": expectations,
        "k20_reproduction": k20,
        "loss_per_target": loss_per_target,
        "training_table": training_table(record),
        "reference_gate": {
            "name": GATE_RECORD_NAME,
            "sha256": _sha256_file(gate_record_path(root)),
            "passed": gate["passed"],
            "n_compared": gate["n_compared"],
        },
        "mix50_reference": {
            "arm": reference_arm,
            "compared_with": "P8.4b's 500 dt@mix50 cells (the per-draw rows P5.3b read) and P5.3b's committed "
                             "reference_dt_cells.mix50 means",
            "never_gating": "A26(c): P4.7's reference arm is reported",
            "n_compared": len(mix_cells),
            "n_equal": len(mix_cells) - len(differing),
            "differing": differing,
            "p5_3b_means": p5_3b_means,
        },
        "canary": {"seconds": canary["seconds"], "facts": canary["facts"], "line": canary.get("line")},
        "inputs": {
            "reference_rows": {"name": REFERENCE_ROWS_NAME, "sha256": REFERENCE_ROWS_SHA256},
            "train_record": {"name": TRAIN_RECORD_DATA_NAME, "sha256": TRAIN_RECORD_SHA256},
            "p5_3b": {"name": P5_3B_ARTIFACT_NAME, "sha256": _sha256_file(p5_3b_path)},
        },
        "provenance": {
            "chunk_code_commits": sorted(commits),
            "report_code": _code_provenance(),
            "engine_seed": ENGINE_SEED,
            "device": EVAL_DEVICE,
            "held_out_draws": [draws[0], draws[-1]],
        },
        "what_this_does_not_say": list(_ARTIFACT_LIMITS),
    }


def write_context_sweep_artifact(output_root: str | Path, payload: Mapping[str, Any]) -> Path:
    """``output/p5_3c/artifacts/p5_3c_context_sweep.json``, written once (the same content again is a no-op)."""
    path = artifact_path(output_root)
    assert_writable(path, output_root, entries=CAMPAIGN_OUTPUT_ENTRIES)
    path.parent.mkdir(exist_ok=True)
    _write_campaign_text_once(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", path, output_root)
    return path


def write_campaign_manifest(output_root: str | Path) -> Path:
    """``output/SHA256SUMS_p5_3c.txt`` over ``output/p5_3c/``, ordered by path, written once and re-verified."""
    root = Path(output_root)
    campaign = campaign_root(root)
    if not campaign.is_dir():
        raise ValueError(f"{campaign} does not exist; there is nothing to list")
    lines = sorted((str(path.relative_to(root)), _sha256_file(path)) for path in campaign.rglob("*") if path.is_file())
    manifest = campaign_manifest_path(root)
    _write_campaign_text_once("".join(f"{digest}  {relative}\n" for relative, digest in lines), manifest, root)
    for relative, digest in _manifest_entries(manifest).items():
        if _sha256_file(root / relative) != digest:
            raise ValueError(f"{root / relative} does not match {manifest}")
    return manifest


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


def _cmd_campaign_inputs(args: argparse.Namespace) -> int:
    facts = check_campaign_inputs(
        output_root=Path(args.output_root), corpus_root=Path(args.corpus_root), draws_root=Path(args.draws_root),
        data_dir=Path(args.data_dir),
    )
    print(
        f"campaign_inputs PASSED: the reference rows at {str(facts['rows_sha256'])[:12]}; the training record at "
        f"{str(facts['train_record_sha256'])[:12]} with its {facts['n_runs']} checkpoints and their manifest; the "
        f"{facts['n_published']} published K = 20 checkpoints; both subjects' env settings == P8.4b's; the 100 held-out "
        f"draws; P8.4b's 500 mix50 cells; CUDA {facts['cuda_free_mib']:.0f} MiB free (budget "
        f"{facts['device_budget_mib']} MiB)",
        flush=True,
    )
    return 0


def _cmd_resume_check(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    stray = undeclared_chunk_names(root)
    unresolvable = unresolvable_chunk_commits(root)
    for name in stray:
        print(f"resume_check: {name} is not a declared chunk", flush=True)
    for entry in unresolvable:
        print(f"resume_check: {entry['chunk']} records a commit git cannot resolve ({entry['code_commit']})", flush=True)
    if stray or unresolvable:
        return 1
    print("resume_check PASSED: no undeclared file in cells/, every chunk's commit resolvable", flush=True)
    return 0


def _cmd_reroll_check(args: argparse.Namespace) -> int:
    try:
        result = run_reference_reroll_check(
            output_root=Path(args.output_root), corpus_root=Path(args.corpus_root), draws_root=Path(args.draws_root),
            data_dir=Path(args.data_dir), canary_seconds=float(args.canary_seconds), workers=int(args.workers),
        )
    except RuntimeError as exc:
        print(f"context_sweep reference-reroll-check: REFUSED: {exc}", flush=True)
        return 2
    for line in result["lines"]:
        print(line, flush=True)
    print(f"reference_reroll_check: {result['n_match']}/{len(result['lines'])} MATCH; records under {result['run_dir']}",
          flush=True)
    return 0 if result["n_match"] == len(result["lines"]) else 1


def _cmd_cells(args: argparse.Namespace) -> int:
    result = run_campaign_stage(
        stage=str(args.stage), output_root=Path(args.output_root), corpus_root=Path(args.corpus_root),
        draws_root=Path(args.draws_root), data_dir=Path(args.data_dir), canary_seconds=float(args.canary_seconds),
        workers=int(args.workers),
    )
    print(
        f"cells {result['stage']}: {result['n_declared']} declared, {result['n_reused']} reused, {result['n_rolled']} "
        f"rolled, {result['n_failed']} failed, in {result['wall_seconds']:.0f} s",
        flush=True,
    )
    return 0 if result["n_failed"] == 0 else 1


def _cmd_reference_gate(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    verdict = reference_gate(output_root=root, data_dir=Path(args.data_dir))
    if not verdict["passed"]:
        named = "; ".join(f"{entry['cell']}: {' '.join(entry['fields'])}" for entry in verdict["differing"])
        print(
            f"reference_gate FAILED: {len(verdict['differing'])} of {verdict['n_compared']} {GATED_ARM} chunks differ "
            f"from {REFERENCE_ROWS_NAME} -- {named}; NOTHING written, and no sweep cell may be rolled (A26(c))",
            flush=True,
        )
        return 1
    path = write_gate_record(root, verdict)
    print(
        f"reference_gate PASSED: {verdict['n_compared']}/{verdict['n_declared']} {GATED_ARM} chunks == "
        f"{REFERENCE_ROWS_NAME} on {' and '.join(ATT_DEFINITIONS)}; wrote {path}",
        flush=True,
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    payload = build_context_sweep_artifact(
        output_root=Path(args.output_root), corpus_root=Path(args.corpus_root), draws_root=Path(args.draws_root),
        data_dir=Path(args.data_dir),
    )
    path = write_context_sweep_artifact(Path(args.output_root), payload)
    print(f"report: wrote {path} ({payload['n_cells']} cells)", flush=True)
    return 0


def _cmd_campaign_manifest(args: argparse.Namespace) -> int:
    path = write_campaign_manifest(Path(args.output_root))
    n = len(path.read_text(encoding="utf-8").splitlines())
    print(f"campaign-manifest: {path} lists {n} files under {CAMPAIGN_DIRNAME}/ and was re-verified", flush=True)
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

    def campaign(name: str, help_text: str, handler: Any, *, inputs: bool = False, data: bool = False) -> Any:
        sub = command(name, help_text, handler, corpus=inputs, data=inputs or data)
        if inputs:
            sub.add_argument("--draws-root", required=True, help="the materialised draws (scenarios/draws)")
        return sub

    campaign("campaign-inputs", "C3: every campaign input by digest, CUDA and the device budget, before the canary",
             _cmd_campaign_inputs, inputs=True)
    command("resume-check", "C3: no undeclared chunk file and no unresolvable chunk commit, before the token",
            _cmd_resume_check)
    reroll = campaign("reference-reroll-check", "C3: the fifteen fenced reference cells, before the token",
                      _cmd_reroll_check, inputs=True)
    reroll.add_argument("--canary-seconds", type=float, required=True)
    reroll.add_argument("--workers", type=int, default=CAMPAIGN_WORKERS)
    cells = campaign("cells", "C3: roll one stage's declared cells not already on disk", _cmd_cells, inputs=True)
    cells.add_argument("--stage", required=True, choices=STAGES)
    cells.add_argument("--canary-seconds", type=float, required=True)
    cells.add_argument("--workers", type=int, default=CAMPAIGN_WORKERS)
    campaign("reference-gate", "C3: A26(c)'s gate on the 500 reference chunks; its record written on a pass only",
             _cmd_reference_gate, data=True)
    campaign("report", "C3: the artifact, from the complete set of 7,000 chunks", _cmd_report, inputs=True)
    command("campaign-manifest", "C3: write and re-verify SHA256SUMS_p5_3c.txt", _cmd_campaign_manifest)
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
