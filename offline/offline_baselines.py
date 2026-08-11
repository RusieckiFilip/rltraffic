"""P4.4: train BC, %BC and IQL on the P4 corpus and compare them against the DT, paired.

SKELETON -- signatures only.  Every callable raises ``NotImplementedError`` so the red-first
run reaches the real API surface instead of one shared ``ModuleNotFoundError``.

Artifact format version: ``p4.4-baselines/1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import EpisodeResult, WilcoxonResult

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_GRADIENT_STEPS",
    "DELTA_ATT",
    "IQL_BATCH_TRANSITIONS",
    "METHODS",
    "PairedComparison",
    "StreamReturn",
    "TrainRecord",
    "TransitionTable",
    "assert_campaign_complete",
    "baselines_artifact",
    "build_transitions",
    "equivalence_verdict",
    "filter_stacked_to_streams",
    "iql_reward_scale",
    "iql_targets",
    "main",
    "paired_comparison",
    "rank_biserial",
    "recovered_fraction",
    "stream_returns",
    "top_return_streams",
    "train_bc",
    "train_iql",
]

ARTIFACT_FORMAT_VERSION = "p4.4-baselines/1.0"

#: Declared before the first gradient step (docs/plans/p4.4.md section 3.1): the DT's own
#: REPORTED budget.  No raise is available to this task.
DECLARED_GRADIENT_STEPS = 40_000

#: PREREGISTRATION.md amendment A6, declared before any baseline existed.
DELTA_ATT = 0.6263

BC_BATCH_WINDOWS = 64
IQL_BATCH_TRANSITIONS = 1_280
TOP_RETURN_FRACTION = 0.10
IQL_TAU = 0.7
IQL_BETA = 3.0
IQL_GAMMA = 0.99
IQL_POLYAK = 0.005
IQL_WEIGHT_CLIP = 100.0

METHODS: tuple[str, ...] = ("bc", "bc_top10", "iql")


@dataclass(frozen=True)
class StreamReturn:
    """One (episode, intersection) stream and its undiscounted return."""

    dataset_dir: str
    episode_file: str
    ix_id: str
    ix_index: int
    episode_index: int
    flow_draw: int
    group: tuple[int, int]
    total_return: float


@dataclass(frozen=True)
class TransitionTable:
    """Flat ``(s, a, r, s')`` transitions.  No ``done``: every episode ends by truncation."""

    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    stream_index: torch.Tensor
    t: torch.Tensor
    reward_scale: float


@dataclass(frozen=True)
class TrainRecord:
    """One trained (method, seed): the reported checkpoint and its provenance."""

    method: str
    seed: int
    gradient_steps: int
    declared_gradient_steps: int
    losses: tuple[float, ...]
    window_means: tuple[float, ...]
    plateaued: bool
    checkpoint_path: str
    canonical_digest: str
    file_sha256: str
    seconds: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PairedComparison:
    """A paired per-draw comparison with the effect sizes section 8 makes mandatory."""

    left_arm: str
    right_arm: str
    n_shared_draws: int
    mean_left: float
    mean_right: float
    mean_difference: float
    ci95_half_width: float
    ci95_width: float
    ci95_low: float
    ci95_high: float
    median_difference: float
    wins: int
    losses: int
    ties: int
    rank_biserial: float
    wilcoxon: WilcoxonResult


def stream_returns(dataset: TrajectoryWindowDataset) -> tuple[StreamReturn, ...]:
    """Undiscounted return of every (episode, intersection) stream, from the loader only."""
    raise NotImplementedError


def top_return_streams(
    dataset: TrajectoryWindowDataset, fraction: float = TOP_RETURN_FRACTION
) -> tuple[StreamReturn, ...]:
    """The top *fraction* of streams by return -- %BC's filter."""
    raise NotImplementedError


def filter_stacked_to_streams(
    dataset: TrajectoryWindowDataset,
    stacked: dict[str, torch.Tensor],
    streams: Sequence[StreamReturn],
) -> dict[str, torch.Tensor]:
    """Restrict stacked windows to those belonging to *streams*."""
    raise NotImplementedError


def iql_reward_scale(returns: Sequence[float]) -> float:
    """IQL's published locomotion normalisation: ``1000 / (max - min)``."""
    raise NotImplementedError


def build_transitions(
    dataset: TrajectoryWindowDataset,
    *,
    group: tuple[int, int] | None = None,
    reward_scale: float = 1.0,
) -> TransitionTable:
    """Flat transitions including the final one, whose next state is observation row ``T``."""
    raise NotImplementedError


def iql_targets(
    table: TransitionTable, next_values: torch.Tensor, gamma: float = IQL_GAMMA
) -> torch.Tensor:
    """``r + gamma * V(s')`` for every transition -- no ``done`` term anywhere."""
    raise NotImplementedError


def train_bc(
    stacked: dict[str, torch.Tensor],
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    method: str,
    declared_gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: Any,
    scenario_id: str,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainRecord:
    """Train one BC (or %BC) seed for exactly *declared_gradient_steps* steps."""
    raise NotImplementedError


def train_iql(
    table: TransitionTable,
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    declared_gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: Any,
    scenario_id: str,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainRecord:
    """Train one IQL seed for exactly *declared_gradient_steps* steps."""
    raise NotImplementedError


def rank_biserial(result: WilcoxonResult) -> float:
    """Matched-pairs rank-biserial correlation ``(W+ - W-) / (W+ + W-)``."""
    raise NotImplementedError


def paired_comparison(
    left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]
) -> PairedComparison:
    """Paired per-draw comparison of two arms over their shared draws."""
    raise NotImplementedError


def equivalence_verdict(
    mean_difference: float, ci95_half_width: float, delta: float = DELTA_ATT
) -> str:
    """A6's verdict for a paired difference ``DT - baseline``."""
    raise NotImplementedError


def recovered_fraction(
    att_reference: float, att_arm: float, att_dt: float
) -> float:
    """``(reference - arm) / (reference - dt)`` -- reported unconditionally (A6 clarified)."""
    raise NotImplementedError


def assert_campaign_complete(
    requested: Sequence[tuple[str, int | None, int]],
    produced: Sequence[EpisodeResult],
) -> None:
    """Refuse to report a partial campaign: completed runs must equal runs requested."""
    raise NotImplementedError


def baselines_artifact(
    *,
    episodes: Sequence[EpisodeResult],
    training: dict[str, Any],
    gate_a: dict[str, Any],
    env_settings: dict[str, Any],
    engine_seed: int,
    delta: float = DELTA_ATT,
) -> dict[str, Any]:
    """The reported artifact: cells, paired comparisons, verdicts and recovered fractions."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
