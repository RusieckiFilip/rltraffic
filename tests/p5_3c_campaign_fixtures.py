"""Fixtures for P5.3c C3 (``BRIEF_42`` C3): a fake campaign tree -- the seventy checkpoints by digest, the pinned training
record and reference rows, the held-out draws, P8.4b's ``mix50`` cells, P5.3b's committed means, and the 7,000 chunks.

The chunk FORMAT is stated here a second time, independently of ``offline.context_sweep.cell_payload``: a fixture chunk
must pass the module's ``validate_chunk``, and a test asserts the module assembles the same chunk from the same inputs.
Nothing here imports a simulator; the values are a deterministic function of the cell (:func:`fixture_att`).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
DRAWS: tuple[int, ...] = tuple(range(1000, 1100))
SUBJECTS: tuple[str, ...] = ("mappo1000", "mix50")
#: The registered prompt and scale of the two subjects (``offline/method_tier_grid.py``'s TIERS; A26(a)).
PROMPT: dict[str, tuple[float, float]] = {"mappo1000": (-5762.0, 9991.0), "mix50": (-5959.0, 40223.0)}
CANARY_LINE = (
    'canary 0.75 s {"att_horizon": 247.75089149261333, "decisions": 360, "local_return": -32648.0, '
    '"two_routes_agree": true}'
)
K_EFFECT: dict[int, float] = {1: 2.0, 2: 1.4, 5: 0.6, 10: 0.2, 20: 0.0}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def head_commit() -> str:
    """The repository's HEAD, which the fixture's chunks record (``code_changed_since`` is then empty)."""
    here = Path(__file__).resolve().parent
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(here), capture_output=True, text=True, check=True
    ).stdout.strip()


def fixture_att(stage: str, subject: str, k: int, batch: int, seed: int, draw: int) -> tuple[float, float]:
    """``(att_engine, att_ours)`` of one cell: a K effect falling with K, a batch, seed and draw effect, a residual."""
    effect = K_EFFECT[int(k)] - (0.3 if batch == 1280 else 0.2 if batch == 640 else 0.0)
    base = 100.0 if subject == "mappo1000" else 105.0
    digest = hashlib.sha256(f"{stage}|{subject}|{k}|{batch}|{seed}|{draw}".encode()).digest()
    residual = (int.from_bytes(digest[:4], "big") / 2**32 - 0.5) * 0.8
    engine = base + effect + 0.01 * (draw - 1000) + 0.001 * seed + residual
    ours = engine + 4.0 + (int.from_bytes(digest[4:8], "big") / 2**32 - 0.5) * 0.2
    return engine, ours


def arm_of(stage: str, subject: str, k: int, batch: int) -> str:
    return f"ref_{subject}_k{k}" if stage == "reference" else f"{subject}_k{k}_b{batch}"


def chunk_name(stage: str, subject: str, k: int, batch: int, seed: int, draw: int) -> str:
    return f"cell_{arm_of(stage, subject, k, batch)}_seed{seed}_draw{draw}.json"


# ======================================================================================================================
# The seventy checkpoints, the manifests and the pinned training record
# ======================================================================================================================

PUBLISHED_TEMPLATE: dict[str, str] = {
    "mappo1000": "p4_dt/dt_seed{seed}.pt",
    "mix50": "p4_7/checkpoints/mix50_dt_seed{seed}.pt",
}


def registered_table() -> list[tuple[str, int, int, int]]:
    """A26's sixty runs, written out HERE: (subject, K, batch, seed)."""
    rows: list[tuple[str, int, int, int]] = []
    for subject in SUBJECTS:
        for k in (20, 10, 5, 2, 1):
            rows.extend((subject, k, 64, seed) for seed in SEEDS)
        if subject == "mappo1000":
            rows.extend(("mappo1000", 1, 1280, seed) for seed in SEEDS)
            rows.extend(("mappo1000", 2, 640, seed) for seed in SEEDS)
    return rows


def _tiny_state(seed: int) -> dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    return {"embed.weight": torch.randn(4, 3, generator=generator), "head.bias": torch.randn(2, generator=generator)}


def sweep_payload(subject: str, k: int, batch: int, seed: int) -> dict[str, Any]:
    """A light checkpoint of a registered run: what the campaign's lookup reads from a trained payload."""
    name = f"{subject}_k{k}_b{batch}_seed{seed}"
    target, scale = PROMPT[subject]
    return {
        "format_version": "dt-checkpoint/1.0",
        "model": _tiny_state(seed * 100 + k * 3 + batch),
        "config": {"context_length": int(k)},
        "provenance": {
            "run": name, "subject": subject, "context_length": int(k), "batch_size": int(batch), "seed": int(seed),
            "gradient_steps": 40_000, "declared_gradient_steps": 40_000,
        },
        "target_rtg": target,
        "rtg_scale": scale,
    }


@dataclass
class CheckpointTree:
    output_root: Path
    data_dir: Path
    published: dict[tuple[str, int], dict[str, str]]
    runs: dict[str, dict[str, Any]]
    record_path: Path
    record_sha256: str


def build_checkpoint_tree(root: Path) -> CheckpointTree:
    """The ten published checkpoints at the digests their records name, the sixty runs, the training manifest and the
    training record as the coordinator would commit it (``docs/data/p5_3c_train.json``)."""
    from offline.few_shot import write_payload_exclusive
    from offline.method_tier_grid import canonical_digest_of

    output_root = Path(root) / "output"
    data_dir = Path(root) / "docs_data"
    data_dir.mkdir(parents=True)
    gate: dict[str, dict[str, str]] = {}
    sums: list[str] = []
    p47_runs: list[dict[str, Any]] = []
    published: dict[tuple[str, int], dict[str, str]] = {}
    for subject, template in PUBLISHED_TEMPLATE.items():
        for seed in SEEDS:
            relative = template.format(seed=seed)
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            target, scale = PROMPT[subject]
            payload = {
                "format_version": "dt-checkpoint/1.0", "model": _tiny_state(seed + (0 if subject == "mappo1000" else 7)),
                "config": {"context_length": 20}, "target_rtg": target, "rtg_scale": scale,
            }
            digest = write_payload_exclusive(payload, path)
            published[(subject, seed)] = {"path": relative, "sha256": digest}
            if subject == "mappo1000":
                gate[str(seed)] = {"path": f"output/{relative}", "sha256": digest}
            else:
                sums.append(f"{digest}  {relative}")
                p47_runs.append(
                    {"tier": "mix50", "method": "dt", "seed": seed, "file_sha256": digest,
                     "canonical_digest": canonical_digest_of(path)}
                )
    (data_dir / "p4_gate.json").write_text(json.dumps({"checkpoints": gate}) + "\n", encoding="utf-8")
    (output_root / "SHA256SUMS_p4_7.txt").write_text("".join(f"{line}\n" for line in sorted(sums)), encoding="utf-8")
    (data_dir / "p4_7_training.json").write_text(json.dumps({"runs": p47_runs}) + "\n", encoding="utf-8")

    runs: dict[str, dict[str, Any]] = {}
    manifest_lines: list[tuple[str, str]] = []
    for subject, k, batch, seed in registered_table():
        name = f"{subject}_k{k}_b{batch}_seed{seed}"
        relative = f"p5_3c_training/checkpoints/{name}.pt"
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = write_payload_exclusive(sweep_payload(subject, k, batch, seed), path)
        manifest_lines.append((relative, digest))
        runs[name] = {
            "subject": subject, "k": k, "batch": batch, "seed": seed, "arm": f"{subject}_k{k}_b{batch}",
            "steps": 40_000, "warmup_steps": 1_000, "checkpoint": relative, "checkpoint_sha256": digest,
            "weights_sha256": _sha256_bytes(name.encode()), "loop_seconds": 200.0 + k, "ms_per_step": 5.0,
            "final_loss": 0.01 * k + 0.001 * seed / 101, "loss_per_supervised_target": [1.0 / (i + 1) for i in range(20)],
            "supervised_targets_per_step": {"mean": float(k * batch), "min": batch, "max": k * batch,
                                            "total": k * batch * 40_000},
            "attempts": 1, "reruns": 0, "code_commit": "f" * 40, "device": "cuda",
        }
    manifest = output_root / "SHA256SUMS_p5_3c_train.txt"
    manifest.write_text("".join(f"{digest}  {relative}\n" for relative, digest in sorted(manifest_lines)), encoding="utf-8")
    k20 = {
        subject: {
            "all_equal": True,
            "n_seeds_equal": 5,
            "seeds": [
                {
                    "seed": seed,
                    "candidate": runs[f"{subject}_k20_b64_seed{seed}"]["checkpoint"],
                    "candidate_sha256": runs[f"{subject}_k20_b64_seed{seed}"]["checkpoint_sha256"],
                    "reference": published[(subject, seed)]["path"],
                    "reference_sha256": published[(subject, seed)]["sha256"],
                    "all_equal": True, "n_parameters_differing": 0, "largest_abs_difference": 0.0,
                    "largest_abs_difference_parameter": None,
                }
                for seed in SEEDS
            ],
        }
        for subject in SUBJECTS
    }
    record = {
        "format_version": "p5.3c-train-record/1.0",
        "n_runs": 60,
        "runs": runs,
        "manifest": {"name": manifest.name, "sha256": _sha256_file(manifest)},
        "k20_reproduction": {"path": "p5_3c_training/k20_reproduction.json", "sha256": "0" * 64,
                             "record": {"format_version": "p5.3c-k20-reproduction/1.0", "subjects": k20}},
        # The embedded timing record, as the pinned record carries it (Amendment C, C3(iii) reads its repeat).
        "timing": {"path": "p5_3c_training/fenced_timing/20260929T112505Z/timing.json", "sha256": "0" * 64,
                   "record": {"format_version": "p5.3c-train-timing/1.0",
                              "repeat": {"file_sha256_equal": True, "weights_sha256_equal": True}}},
    }
    record_path = data_dir / "p5_3c_train.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CheckpointTree(output_root, data_dir, published, runs, record_path, _sha256_file(record_path))


# ======================================================================================================================
# The draws, the reference rows, P8.4b's mix50 cells, P5.3b's means
# ======================================================================================================================


def build_draws(root: Path) -> Path:
    """One hundred held-out draws: a CityFlow config and a flow of single-vehicle entries each (``created_from_flow``'s
    shape)."""
    draws_root = Path(root) / "draws"
    for draw in DRAWS:
        directory = draws_root / "cityflow1x1" / f"draw_{draw:04d}"
        directory.mkdir(parents=True)
        (directory / "cityflow.json").write_text(json.dumps({"draw": draw, "flowFile": "flow.json"}) + "\n", encoding="utf-8")
        flow = [{"vehicle": {}, "route": ["a", "b"], "interval": 1.0, "startTime": t, "endTime": t}
                for t in range(draw % 7, 4000, 11)]
        (directory / "flow.json").write_text(json.dumps(flow) + "\n", encoding="utf-8")
    return draws_root


def build_rows(data_dir: Path, published: dict[tuple[str, int], dict[str, str]],
               values: Callable[..., tuple[float, float]] = fixture_att) -> tuple[Path, str]:
    """The reference rows of P4's arm as C1b committed them: the rows' values are ``ref_mappo1000_k20``'s."""
    rows = []
    for seed in SEEDS:
        for draw in DRAWS:
            engine, ours = values("reference", "mappo1000", 20, 64, seed, draw)
            rows.append({"seed": seed, "draw_id": draw, "att_engine": engine, "att_ours": ours,
                         "source": f"cell_hz1x1_dt_at_mappo1000_seed{seed}_draw{draw}.json", "source_sha256": "a" * 64})
    payload = {
        "format_version": "p5.3c-reference-rows/1.0",
        "definitions": ["att_engine", "att_ours"],
        "primary_definition": "att_engine",
        "n_rows": len(rows),
        "rows": rows,
        "seeds": list(SEEDS),
        "draw_ids": list(DRAWS),
        "engine_seed": 1000,
        "checkpoints": {
            str(seed): {"path": f"output/{published[('mappo1000', seed)]['path']}",
                        "sha256": published[("mappo1000", seed)]["sha256"]}
            for seed in SEEDS
        },
    }
    path = Path(data_dir) / "p4_k20_att_engine_rows.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, _sha256_file(path)


def build_p8_4b_mix50_cells(output_root: Path, values: Callable[..., tuple[float, float]] = fixture_att) -> None:
    """P8.4b's 500 re-derived ``dt@mix50`` cells, carrying the fixture's ``ref_mix50_k20`` values."""
    directory = Path(output_root) / "p8_4b_rederivation"
    directory.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for draw in DRAWS:
            engine, ours = values("reference", "mix50", 20, 64, seed, draw)
            cell = {
                "format_version": "p8.4b-rederivation/1.0", "arm": "dt@mix50", "tier": "mix50", "method": "dt",
                "scenario": "hz1x1", "seed": seed, "draw_id": draw, "att_engine": engine, "att_ours": ours,
                "reproduces_committed": True,
            }
            (directory / f"cell_hz1x1_dt_at_mix50_seed{seed}_draw{draw}.json").write_text(
                json.dumps(cell, sort_keys=True) + "\n", encoding="utf-8"
            )


def build_p5_3b(data_dir: Path, values: Callable[..., tuple[float, float]] = fixture_att) -> None:
    """P5.3b's committed ``reference_dt_cells.mix50`` means, computed here over the 500 values in (seed, draw) order."""
    engine = [values("reference", "mix50", 20, 64, seed, draw)[0] for seed in SEEDS for draw in DRAWS]
    ours = [values("reference", "mix50", 20, 64, seed, draw)[1] for seed in SEEDS for draw in DRAWS]
    payload = {
        "reference_dt_cells": {
            "mix50": {"arm": "dt@mix50", "att_engine_mean": float(np.mean(engine)), "att_ours_mean": float(np.mean(ours))}
        }
    }
    (Path(data_dir) / "p5_3b_nortg.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


# ======================================================================================================================
# The chunks
# ======================================================================================================================


def demand_of(draws_root: Path, draw: int) -> dict[str, str]:
    directory = Path(draws_root) / "cityflow1x1" / f"draw_{draw:04d}"
    return {"config_sha256": _sha256_file(directory / "cityflow.json"), "flow_sha256": _sha256_file(directory / "flow.json")}


def episode_record(stage: str, subject: str, k: int, batch: int, seed: int, draw: int,
                   values: Callable[..., tuple[float, float]] = fixture_att) -> dict[str, Any]:
    """``AdmissionEpisode.as_record()``'s twenty keys, with the fixture's two ATT values."""
    engine, ours = values(stage, subject, k, batch, seed, draw)
    return {
        "scenario": "hz1x1", "tier": subject, "method": "dt", "arm": arm_of(stage, subject, k, batch), "seed": seed,
        "draw_id": draw, "created": 300, "entered": 290, "never_entered": 10, "entered_fraction": 290 / 300,
        "completed_at_horizon": 250, "running_at_horizon": 40, "waiting_at_horizon": 10, "att_ours": ours,
        "att_engine": engine, "att_difference": ours - engine, "horizon_vehicle_count": 40.0, "episode_reward": -3000.0,
        "seconds": 2.9, "seconds_rollout": 2.5,
    }


def fixture_chunk(
    stage: str, subject: str, k: int, batch: int, seed: int, draw: int, *, checkpoint: dict[str, Any],
    demand: dict[str, str], commit: str, values: Callable[..., tuple[float, float]] = fixture_att,
) -> dict[str, Any]:
    """One chunk in the ``p5.3c-cell/1.0`` format, written out HERE (the module's ``cell_payload`` must agree)."""
    target = PROMPT[subject][0]
    return {
        "format_version": "p5.3c-cell/1.0",
        "name": chunk_name(stage, subject, k, batch, seed, draw),
        "stage": stage,
        "arm": arm_of(stage, subject, k, batch),
        "subject": subject,
        "k": k,
        "batch": batch,
        "seed": seed,
        "draw_id": draw,
        "run": None if stage == "reference" else f"{subject}_k{k}_b{batch}_seed{seed}",
        "episode": episode_record(stage, subject, k, batch, seed, draw, values),
        "n_decisions": 3,
        "series": {"rtg": [target, target + 10.0, target + 25.0], "reward": [0.0, -10.0, -15.0], "action": [0, 3, 1]},
        "checkpoint": checkpoint,
        "demand": demand,
        "target_rtg": target,
        "device": "cuda",
        "engine_seed": 1000,
        "scenario": "hz1x1",
        "scenario_id": "cityflow1x1",
        "torch_num_threads": 1,
        "code_commit": commit,
        "code_dirty": False,
        "canary_seconds": 0.75,
    }


def checkpoint_identity(tree: CheckpointTree, stage: str, subject: str, k: int, batch: int, seed: int) -> dict[str, Any]:
    if stage == "reference":
        entry = tree.published[(subject, seed)]
        return {"path": entry["path"], "sha256": entry["sha256"], "checked_against": ["fixture"]}
    entry = tree.runs[f"{subject}_k{k}_b{batch}_seed{seed}"]
    return {"path": entry["checkpoint"], "sha256": entry["checkpoint_sha256"], "checked_against": ["fixture"]}


def declared_table() -> list[tuple[str, str, int, int, int, int]]:
    """The 7,000 cells, written out HERE: (stage, subject, K, batch, seed, draw)."""
    cells: list[tuple[str, str, int, int, int, int]] = []
    for subject in SUBJECTS:
        cells.extend(("reference", subject, 20, 64, seed, draw) for seed in SEEDS for draw in DRAWS)
    for subject, k, batch, seed in registered_table():
        cells.extend(("sweep", subject, k, batch, seed, draw) for draw in DRAWS)
    return cells


def write_chunks(
    tree: CheckpointTree, draws_root: Path, *, commit: str, values: Callable[..., tuple[float, float]] = fixture_att,
    cells: list[tuple[str, str, int, int, int, int]] | None = None,
) -> Path:
    directory = tree.output_root / "p5_3c" / "cells"
    directory.mkdir(parents=True, exist_ok=True)
    demands = {draw: demand_of(draws_root, draw) for draw in DRAWS}
    for stage, subject, k, batch, seed, draw in cells if cells is not None else declared_table():
        chunk = fixture_chunk(
            stage, subject, k, batch, seed, draw, checkpoint=checkpoint_identity(tree, stage, subject, k, batch, seed),
            demand=demands[draw], commit=commit, values=values,
        )
        (directory / chunk["name"]).write_text(json.dumps(chunk, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return directory


@dataclass
class CampaignTree:
    root: Path
    output_root: Path
    data_dir: Path
    draws_root: Path
    corpus_root: Path
    rows_sha256: str
    record_sha256: str
    commit: str
    checkpoints: CheckpointTree


def build_campaign_tree(root: Path, *, values: Callable[..., tuple[float, float]] = fixture_att,
                        chunks: bool = True) -> CampaignTree:
    """Everything the report reads except the gate record and ``canary.json`` (the tests write those through the
    module and ``transfer_calibration.record_canary``)."""
    tree = build_checkpoint_tree(root)
    draws_root = build_draws(root)
    _rows, rows_sha = build_rows(tree.data_dir, tree.published, values)
    build_p8_4b_mix50_cells(tree.output_root, values)
    build_p5_3b(tree.data_dir, values)
    corpus_root = Path(root) / "corpus"
    corpus_root.mkdir()
    commit = head_commit()
    if chunks:
        write_chunks(tree, draws_root, commit=commit, values=values)
    return CampaignTree(Path(root), tree.output_root, tree.data_dir, draws_root, corpus_root, rows_sha,
                        tree.record_sha256, commit, tree)


def finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)
