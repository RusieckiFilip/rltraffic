"""Train the P4 Decision Transformer and evaluate the pre-registered P4.2 gate.

SKELETON -- signatures only.  Every body raises ``NotImplementedError`` so the red tests reach
the real API surface instead of one shared import error.

Artifact format version: ``p4-gate/1.0``.

WHAT THE GATE IS, AND WHERE IT IS MEASURED
------------------------------------------
``PREREGISTRATION.md`` section 9, verbatim and not renegotiable::

    ATT_MADT <= ATT_MaxPressure   AND   ATT_MADT <= 1.05 * ATT_best_online

Evaluated on the **registered held-out pool, all 100 draws 1000-1099** (BRIEF_10 section 8),
with the DT, MaxPressure and MAPPO@1000 measured live and **paired by draw id**.  Section 8 of
``PREREGISTRATION.md`` registered the held-out flow draw as the unit of replication on
2026-08-03, which is why the ladder's ``110.73`` -- measured on the *training* draws 1-200 --
is reported as a labelled secondary reading and is never the verdict.

Amendment A5 governs the reporting: ``vehicle_count`` at the horizon accompanies every ATT cell
**unconditionally**, every cell carries its **draw ids**, and a comparison without shared draws
is void.  Here every arm runs on the identical draw set by construction, so no cell needs
recomputing over an intersection.

THE LEAKAGE RULES THIS MODULE MECHANISES (``PREREGISTRATION.md`` section 6)
---------------------------------------------------------------------------
* **No online model selection.** The reported checkpoint is the one at the declared step count.
  :func:`load_gate_checkpoint` refuses a checkpoint whose recorded step count differs from the
  declared one, so "the best checkpoint" cannot be reported by accident.
* **The single budget raise reads the training curve only.** :func:`plateau_reached` consumes
  training-loss window means and nothing else; no evaluation number can reach it.
* **Statistics are fitted on the training split only** -- delegated to
  ``offline.dataset.TrajectoryWindowDataset``, which fits them only for ``split="train"``, and
  carried into the checkpoint so evaluation reuses the frozen numbers.
* **Held-out purity** is asserted from the artifact: the training and evaluation draw id sets
  must not intersect.

WHY THE EVALUATION ENV IS DERIVED FROM THE CORPUS MANIFEST
-----------------------------------------------------------
:func:`env_settings_from_manifest` reads ``max_steps``, ``delta_time``, ``control_mode``,
``state_features``, the reward functions and the global reward weight out of the collection
manifest instead of restating them.  A DT evaluated in an env whose state block differs from the
collection env reads a different feature space and produces a plausible, wrong number; deriving
the settings from the manifest makes that drift impossible to introduce silently, and
``tests/test_dt_gate.py`` additionally replays a stored episode to prove the reconstruction is
bit-exact.

WILCOXON
--------
scipy is not installed and no repo file imports it, so the paired signed-rank test required by
BRIEF_10 section 8 is implemented here in numpy/stdlib rather than added as a dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from offline.dataset import NormalizationStats

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CellStats",
    "EpisodeResult",
    "GateVerdict",
    "HELD_OUT_DRAWS",
    "TrainResult",
    "WilcoxonResult",
    "build_training_dataset",
    "env_settings_from_manifest",
    "evaluate_arm",
    "gate_verdict",
    "load_gate_checkpoint",
    "main",
    "mean_ci95",
    "plateau_reached",
    "stack_dataset",
    "train_dt",
    "wilcoxon_signed_rank",
    "window_means",
    "write_json_atomic",
]

ARTIFACT_FORMAT_VERSION = "p4-gate/1.0"

#: The registered held-out evaluation pool, used whole so no slice can be selected.
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))

#: The registered gate ratio against the best online policy.
GATE_RATIO = 1.05


@dataclass(frozen=True)
class EpisodeResult:
    """One rollout of one arm on one draw: the primary metric and A5's companion."""

    arm: str
    seed: int | None
    draw_id: int
    att_horizon: float
    horizon_vehicle_count: float
    episode_reward: float


@dataclass(frozen=True)
class CellStats:
    """Mean with a 95% CI, matching ``offline/att_ladder.py``'s convention (``ddof=1``)."""

    n: int
    mean: float
    std: float
    ci95: float


@dataclass(frozen=True)
class WilcoxonResult:
    """Paired signed-rank test over shared draws; ties get average ranks, zeros are dropped."""

    w_plus: float
    w_minus: float
    statistic: float
    n_used: int
    n_zero: int
    z: float
    p_value: float


@dataclass(frozen=True)
class GateVerdict:
    """The two registered inequalities, evaluated explicitly.  ``<=``: equality passes."""

    att_madt: float
    att_maxpressure: float
    att_best_online: float
    ratio: float
    threshold_online: float
    gate_a: bool
    gate_b: bool
    passed: bool


@dataclass(frozen=True)
class TrainResult:
    """One trained seed: the reported checkpoint and the curve that chose its length."""

    seed: int
    gradient_steps: int
    declared_gradient_steps: int
    losses: tuple[float, ...]
    window_means: tuple[float, ...]
    plateaued: bool
    checkpoint_path: str
    seconds: float


def mean_ci95(values: Sequence[float]) -> CellStats:
    """Mean, sample standard deviation (``ddof=1``) and a 95% CI half-width."""
    raise NotImplementedError


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test of ``x - y``.

    Zero differences are dropped (Wilcoxon's own convention), ties share an average rank, and
    the p-value uses the normal approximation with the tie correction and a continuity
    correction.  The approximation is what the registered analysis plan needs at n >= 20; it is
    reported with ``n_used`` so a caller can see how many pairs it actually rested on.
    """
    raise NotImplementedError


def gate_verdict(
    att_madt: float,
    att_maxpressure: float,
    att_best_online: float,
    ratio: float = GATE_RATIO,
) -> GateVerdict:
    """Evaluate both registered inequalities.  ``<=`` exactly as registered: equality passes."""
    raise NotImplementedError


def window_means(losses: Sequence[float], window: int) -> tuple[float, ...]:
    """Mean training loss per consecutive window of *window* gradient steps."""
    raise NotImplementedError


def plateau_reached(means: Sequence[float], tolerance: float = 0.05) -> bool:
    """Whether the last two consecutive relative changes are below *tolerance*.

    Mirrors amendment A3's criterion, applied to the training loss instead of the training
    return, and reads the training curve only -- no evaluation number can enter this decision.
    """
    raise NotImplementedError


def env_settings_from_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Env settings for evaluation, read out of the collection manifest rather than restated."""
    raise NotImplementedError


def build_training_dataset(
    dataset_dirs: Sequence[str | Path], context_length: int
) -> Any:
    """The training-split window dataset over *dataset_dirs*, with statistics fitted there."""
    raise NotImplementedError


def stack_dataset(dataset: Any) -> dict[str, torch.Tensor]:
    """Materialise every window once into contiguous tensors, in dataset order.

    Built by iterating the dataset's own ``__getitem__``, so the loader stays the single
    definition of what a window is; the tests re-check a random sample of rows against
    ``dataset[i]``.
    """
    raise NotImplementedError


def train_dt(
    stacked: dict[str, torch.Tensor],
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    declared_gradient_steps: int,
    raise_to: int | None,
    context_length: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: NormalizationStats,
    scenario_id: str,
    target_rtg: float,
    rtg_scale: float,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainResult:
    """Train one seed to the declared step count, applying the single pre-declared raise."""
    raise NotImplementedError


def load_gate_checkpoint(
    gym_env: Any, path: str | Path, declared_gradient_steps: int, device: str | None = None
) -> Any:
    """Load a DT checkpoint, refusing one whose recorded step count is not the declared one.

    This is the mechanical form of "no online model selection": a checkpoint saved at a
    different step -- an earlier one that scored better, say -- cannot be evaluated by this
    path at all.
    """
    raise NotImplementedError


def evaluate_arm(
    *,
    arm: str,
    seed: int | None,
    draw_ids: Sequence[int],
    config_for_draw: Callable[[int], Path],
    env_settings: dict[str, Any],
    scenario_id: str,
    choose_action_factory: Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]],
    engine_seed: int,
) -> list[EpisodeResult]:
    """Roll one arm over *draw_ids*, one episode per draw, through ``horizon_rollout``.

    ``episodes=1`` per draw is deliberate: at that setting the reader's ``final_vehicle_count``
    is that single episode's value, so P8.0 finding B2 -- a last-episode ``final_completed``
    mixed with a mean ``final_vehicle_count`` -- cannot arise.  ``final_completed`` is not read.
    """
    raise NotImplementedError


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Write *payload* as JSON atomically, after validation, into an existing directory."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``baselines`` (pre-training thresholds), ``train``, ``evaluate``."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
