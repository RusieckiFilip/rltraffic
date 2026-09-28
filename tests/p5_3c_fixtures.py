"""Shared fixtures for P5.3c's tests (``BRIEF_42``): a single-intersection synthetic corpus, a stub env, and a
fake output tree carrying the sources of the reference rows.

Nothing here reads or writes the real corpus, the real ``output/`` tree or ``docs/data/``.  Every helper writes under
the ``root`` it is handed, which is always a pytest ``tmp_path``.

THE CORPUS
----------
One collection directory per call, one ``.npz`` per draw plus a ``manifest.json``, laid out exactly as
``offline.trajectory_logger`` writes format ``1.0`` (the layout ``tests/test_offline_dataset.py`` builds for its
two-intersection fixture), but with ONE intersection, as on ``cityflow1x1``: C6's alignment convention -- observation
rows ``T + 1``, decision and outcome rows ``T``.  The values are chosen so that the tests can compare with ``==``:

* every state entry is distinct across (episode, t, feature) and exactly representable in float32;
* every reward is a NON-POSITIVE INTEGER (a queue length, as the corpus's reward is), so every return-to-go is exact
  in any summation order, and a return-to-go is largest in magnitude at ``t = 0`` exactly as on the real corpus;
* the availability mask BINDS on odd rows (only the logged action is legal there), unlike the real corpus.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: The fixture's single intersection and its shape.
IX_ID = "ix_solo"
SCENARIO_ID = "fixture_1ix"
STATE_DIM = 5
N_ACTIONS = 4
T_DECISIONS = 12

LANE_IDS: tuple[str, ...] = ("lane_a", "lane_b")
METRIC_KEYS: tuple[str, ...] = ("number_of_all_halting_vehicles_for_the_last_time_step_in_simulation",)

#: The env settings a collection manifest carries (``offline.dt_gate.env_settings_from_manifest`` reads these keys).
RUN_METADATA_SETTINGS: dict[str, Any] = {
    "max_steps": T_DECISIONS,
    "delta_time": 10,
    "control_mode": "acyclic",
    "state_features": ["lane_vehicle_count", "lane_waiting", "phase_onehot"],
    "global_reward_fn": "queue_length",
    "local_reward_fn": "queue_length",
    "global_reward_weight": 0.0,
}


def state_rows(episode: int, *, t_decisions: int = T_DECISIONS, state_dim: int = STATE_DIM) -> np.ndarray:
    """``(T + 1, state_dim)`` float32 states, distinct per (episode, t, feature), exact in float32."""
    rows = np.empty((t_decisions + 1, state_dim), dtype=np.float32)
    for t in range(t_decisions + 1):
        for d in range(state_dim):
            rows[t, d] = np.float32(t + 13 * d + 100 * episode) + np.float32(0.25)
    return rows


def actions(episode: int, *, t_decisions: int = T_DECISIONS, n_actions: int = N_ACTIONS) -> np.ndarray:
    """The logged actions, one per decision, int64."""
    return np.asarray([(t + episode) % n_actions for t in range(t_decisions)], dtype=np.int64)


def avail_mask(episode: int, *, t_decisions: int = T_DECISIONS, n_actions: int = N_ACTIONS) -> np.ndarray:
    """``(T + 1, n_actions)`` bool: all legal on even rows, only the logged action on odd rows (the mask binds)."""
    taken = actions(episode, t_decisions=t_decisions, n_actions=n_actions)
    mask = np.zeros((t_decisions + 1, n_actions), dtype=np.bool_)
    for t in range(t_decisions + 1):
        if t == t_decisions or t % 2 == 0:
            mask[t, :] = True
        else:
            mask[t, int(taken[t])] = True
    return mask


def local_rewards(episode: int, *, t_decisions: int = T_DECISIONS) -> np.ndarray:
    """Non-positive integral rewards (float32), different per episode so the returns differ."""
    # Negated as an INTEGER before the float conversion, so a zero reward is +0.0 and never -0.0.
    return np.asarray([float(-((3 * t + episode) % 7 + episode)) for t in range(t_decisions)], dtype=np.float32)


def _episode_arrays(episode: int, draw: int, *, t_decisions: int, state_dim: int, n_actions: int) -> dict[str, np.ndarray]:
    rewards = local_rewards(episode, t_decisions=t_decisions)
    waiting = [[(t + episode) % 3, (2 * t + episode) % 4] for t in range(t_decisions + 1)]
    return {
        "format_version": np.asarray("1.0"),
        "ix_ids": np.asarray([IX_ID], dtype=np.str_),
        "lane_ids": np.asarray(list(LANE_IDS), dtype=np.str_),
        "metric_keys": np.asarray(list(METRIC_KEYS), dtype=np.str_),
        "vehicle_count": np.asarray([10 + t for t in range(t_decisions + 1)], dtype=np.int64),
        "sim_time": np.asarray([10.0 * t for t in range(t_decisions + 1)], dtype=np.float32),
        "step": np.asarray(list(range(t_decisions + 1)), dtype=np.int64),
        "metrics": np.asarray([[float(sum(row))] for row in waiting], dtype=np.float32),
        "lane_vehicle_count": np.asarray([[v + 1 for v in row] for row in waiting], dtype=np.int32),
        "lane_waiting_vehicle_count": np.asarray(waiting, dtype=np.int32),
        "episode_length": np.asarray(t_decisions, dtype=np.int64),
        "terminated": np.asarray(False, dtype=np.bool_),
        "truncated": np.asarray(True, dtype=np.bool_),
        "engine_seed": np.asarray(1000 + episode, dtype=np.int64),
        "flow_draw": np.asarray(draw, dtype=np.int64),
        "global_reward": np.zeros(t_decisions, dtype=np.float32),
        "ix0_state": state_rows(episode, t_decisions=t_decisions, state_dim=state_dim),
        "ix0_avail_mask": avail_mask(episode, t_decisions=t_decisions, n_actions=n_actions),
        "ix0_current_phase": np.asarray([t % n_actions for t in range(t_decisions + 1)], dtype=np.int64),
        "ix0_time_in_phase": np.asarray([float(t % 4) for t in range(t_decisions + 1)], dtype=np.float32),
        "ix0_action": actions(episode, t_decisions=t_decisions, n_actions=n_actions),
        "ix0_local_reward": rewards,
    }


def _episode_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(arrays["ix0_action"].astype("<i8", copy=False).tobytes())
    digest.update(arrays["global_reward"].astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def write_single_ix_corpus(
    root: Path,
    name: str,
    *,
    draws: Sequence[int] = (1, 2, 3),
    first_episode: int = 0,
    t_decisions: int = T_DECISIONS,
    state_dim: int = STATE_DIM,
    n_actions: int = N_ACTIONS,
) -> Path:
    """One collection directory: an episode per draw, plus its manifest.  Returns the directory.

    ``first_episode`` offsets the episode index that seeds the values, so two directories written with different
    offsets hold different trajectories.
    """
    out_dir = Path(root) / name
    out_dir.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for offset, draw in enumerate(draws):
        episode = first_episode + offset
        arrays = _episode_arrays(episode, int(draw), t_decisions=t_decisions, state_dim=state_dim, n_actions=n_actions)
        filename = f"ep{offset:06d}_seed{1000 + episode}_draw{int(draw)}.npz"
        with open(out_dir / filename, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        entries.append(
            {
                "filename": filename,
                "episode_length": int(t_decisions),
                "total_global_reward": 0.0,
                "engine_seed": 1000 + episode,
                "flow_draw": int(draw),
                "episode_sha256": _episode_sha256(arrays),
            }
        )
    manifest = {
        "format_version": "1.0",
        "git_hash": "0" * 40,
        "lane_count": len(LANE_IDS),
        "lane_ids_sha256": hashlib.sha256("\n".join(LANE_IDS).encode("utf-8")).hexdigest(),
        "run_metadata": {
            "scenario_id": SCENARIO_ID,
            "backend": "cityflow",
            "behavior_policy": "fixture",
            **RUN_METADATA_SETTINGS,
            "max_steps": int(t_decisions),
        },
        "episodes": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir


# ======================================================================================================================
# A stub env for DTAgent rollouts (what DTAgent.__init__ reads, and the info dicts act() consumes)
# ======================================================================================================================


class StubIntersection:
    """An intersection object with the two attributes ``Utils.infer_action_counts`` may read."""

    def __init__(self, ix_id: str, num_phases: int) -> None:
        self.id = ix_id
        self.num_phases = num_phases


class StubEnv:
    """One intersection, a ``Discrete`` action space -- the hz1x1 shape."""

    def __init__(self, ix_id: str = IX_ID, n_actions: int = N_ACTIONS, max_steps: int = T_DECISIONS) -> None:
        from gymnasium import spaces

        self.intersections = [StubIntersection(ix_id, n_actions)]
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(n_actions)


def info_at(step: int, state: Sequence[float], avail: Sequence[bool], reward: float, ix_id: str = IX_ID) -> dict[str, Any]:
    """The ``info`` of decision *step*: the state row, the legal actions, and the reward of the PREVIOUS step."""
    legal = [a for a, ok in enumerate(avail) if bool(ok)]
    return {
        "step": int(step),
        "vehicle_count": 0,
        "intersections": {ix_id: {"state": [float(v) for v in state], "avail_actions": legal, "reward": float(reward)}},
    }


# ======================================================================================================================
# A fake output tree for the reference-rows extraction (C1): P4's five checkpoints, P8.4b's cells and manifest
# ======================================================================================================================

REFERENCE_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
REFERENCE_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))


def reference_values(seed: int, draw: int) -> tuple[float, float]:
    """The fixture's ``(att_engine, att_ours)`` for one (seed, draw): distinct, finite, not round numbers."""
    base = 100.0 + (seed % 1000) / 997.0 + (draw - 1000) / 101.0
    return base, base + 4.0 + draw / 7919.0


def reference_cell(seed: int, draw: int, *, output_root: Path) -> dict[str, Any]:
    """One P8.4b cell as ``offline.att_rederivation`` writes it for ``dt@mappo1000``."""
    engine, ours = reference_values(seed, draw)
    return {
        "arm": "dt@mappo1000",
        "att_difference": ours - engine,
        "att_engine": engine,
        "att_ours": ours,
        "committed_att_ours": ours,
        "completed_at_horizon": 1700,
        "created": 1800,
        "draw_id": int(draw),
        "entered": 1800,
        "entered_fraction": 1.0,
        "episode_reward": -7000.0,
        "format_version": "p8.4b-rederivation/1.0",
        "horizon_vehicle_count": 40.0,
        "method": "dt",
        "never_entered": 0,
        "policy_source": {
            "checkpoint": str(output_root / "p4_dt" / f"dt_seed{int(seed)}.pt"),
            "detail": "method_tier_grid._dt_factory with the declared target_rtg -5762.0",
            "kind": "checkpoint",
        },
        "reproduces_committed": True,
        "running_at_horizon": 40,
        "scenario": "hz1x1",
        "seconds": 2.5,
        "seconds_rollout": 2.4,
        "seed": int(seed),
        "tier": "mappo1000",
        "waiting_at_horizon": 0,
    }


def reference_cell_name(seed: int, draw: int) -> str:
    return f"cell_hz1x1_dt_at_mappo1000_seed{int(seed)}_draw{int(draw)}.json"


@dataclass(frozen=True)
class ReferenceTree:
    """Where the fake sources live: an output root and a docs/data directory."""

    output_root: Path
    data_dir: Path

    @property
    def cells(self) -> Path:
        return self.output_root / "p8_4b_rederivation"

    def cell_path(self, seed: int, draw: int) -> Path:
        return self.cells / reference_cell_name(seed, draw)

    def write_cell(self, seed: int, draw: int, payload: dict[str, Any]) -> None:
        self.cell_path(seed, draw).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reference_tree(root: Path) -> ReferenceTree:
    """The five fake P4 checkpoints, ``p4_gate.json`` naming their digests, P8.4b's 500 cells, its manifest, and
    ``p8_4b_rederivation.json``'s provenance -- the exact set of files the extraction reads."""
    output_root = Path(root) / "output"
    data_dir = Path(root) / "docs_data"
    tree = ReferenceTree(output_root=output_root, data_dir=data_dir)
    (output_root / "p4_dt").mkdir(parents=True)
    tree.cells.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    checkpoints: dict[str, dict[str, str]] = {}
    for seed in REFERENCE_SEEDS:
        path = output_root / "p4_dt" / f"dt_seed{seed}.pt"
        path.write_bytes(f"fixture stand-in for P4's checkpoint of seed {seed}\n".encode("utf-8"))
        checkpoints[str(seed)] = {"path": f"output/p4_dt/dt_seed{seed}.pt", "sha256": _file_sha256(path)}
    (data_dir / "p4_gate.json").write_text(
        json.dumps({"checkpoints": checkpoints, "engine_seed": 1000}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (data_dir / "p8_4b_rederivation.json").write_text(
        json.dumps(
            {
                "format_version": "p8.4b-rederivation/1.0",
                "provenance": {"code_provenance": {"git_commit": "3" * 40, "code_dirty": True, "git_dirty": True}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    keys: list[str] = ["grid4x4|bc@fixedtime|101|1000"]
    for seed in REFERENCE_SEEDS:
        for draw in REFERENCE_DRAWS:
            tree.write_cell(seed, draw, reference_cell(seed, draw, output_root=output_root))
            keys.append(f"hz1x1|dt@mappo1000|{seed}|{draw}")
    ordered = sorted(keys)
    (tree.cells / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "format_version": "p8.4b-rederivation/1.0",
                "n_cells": len(ordered),
                "declared_cells_sha256": hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest(),
                "engine_seed": 1000,
                "cells": ordered,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tree
