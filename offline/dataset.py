"""Offline dataset: episode files -> per-intersection context windows for a Decision Transformer.

Reads the corpus written by :mod:`offline.trajectory_logger` (on-disk format versions "1.0"
and, once P2.6 lands, "1.1") through :func:`offline.trajectory_logger.load_episode` -- the
**only** reader of that format.  This module never calls ``np.load`` on an episode file.
Two readers of one format is how an off-by-one gets frozen into a data format.

Alignment convention (inherited from contract C6, not re-derived here)
---------------------------------------------------------------------
::

    observations (T+1 rows)   state, avail_mask, current_phase, time_in_phase, lane_*, att_per_step
    decisions    (T   rows)   action
    outcomes     (T   rows)   local_reward, global_reward

    row t   of an observation array = the state the agent saw before decision t
    row t+1 of an observation array = the post-step state that r_t was computed from

An item is the window of decision steps ending at ``t``, so ``t`` ranges over ``[0, T-1]``
and **observation row T is never an input**: it carries no action and no reward.  That is
how truncation is honoured -- episodes on this platform always end by time limit
(``terminated`` is hardcoded ``False``), and nothing here treats step ``T`` as absorbing or
appends a terminal zero.

Returns-to-go
-------------
::

    rtg[t] = sum(local_reward[t:])          undiscounted, INCLUSIVE of r_t

so ``rtg[t]`` pairs with state row ``t`` and action row ``t``, and ``rtg[T-1] == r_{T-1}``.
The stream is that intersection's own ``ix{i}_local_reward``.  It is purely local **for this
corpus** because collection used ``global_reward_weight = 0.0``; at a nonzero weight the
stored local reward is composite (``weight * global + local_fn``) and a global RTG alongside
it would double-count the global term.

The accumulation runs backwards in ``float64`` and is stored as ``float32``.  The order is
part of the definition: it is the same order a ``np.cumsum`` over the reversed array uses, so
the independent recomputation in the tests compares with ``==`` rather than a tolerance.

Padding
-------
Windows shorter than ``K`` are padded **on the left**.  Every padded position is zero --
``state`` 0.0, ``rtg`` 0.0, ``action`` 0, ``timestep`` 0, ``avail_mask`` all-False,
``attention_mask`` False.  Note that a padded ``action`` of 0 is indistinguishable from a
real action 0: a loss that ignores ``attention_mask`` trains on fabricated zero-actions
without failing.  The mask is not optional.

Draw-aware splitting (D4, registered in ``PREREGISTRATION.md`` section 5)
-------------------------------------------------------------------------
``nominal`` = draw 0 (source vehicle order, never pooled) · ``train`` = draws 1-999 ·
``heldout`` = draws 1000-1099, which must never enter any training corpus for any method.
Episodes are selected **by draw id read from the manifest**, never by file order, and a draw
outside the requested split raises rather than being skipped: a leak that raises is a bug, a
leak that skips is a published number.

Heterogeneous intersections
---------------------------
Intersections may differ in ``state_dim`` and ``n_actions`` (cologne3 does: 16/20/24 and
3/4/4), and C6 forbids padding across them.  Items are therefore not universally
collatable.  :attr:`TrajectoryWindowDataset.groups` maps ``(state_dim, n_actions)`` to the
indices in that group; batches must stay inside one group.  Mixing groups already fails
loudly -- ``torch.utils.data.default_collate`` raises when it tries to stack tensors of
different shapes, which is a mechanical guarantee rather than a convention -- but its message
names shapes, not intersections, so :func:`collate_windows` checks first and names both.

Format version and the ATT field
--------------------------------
Version support is delegated to :func:`offline.trajectory_logger.load_episode`; this module
adds only the rule that **one dataset may not span two format versions** (BRIEF_08 section 5:
a v1.0 and a v1.1 corpus must never be silently mixed).  ``att_per_step`` -- the per-row
observation array added in v1.1, whose horizon element is the registered primary metric -- is
surfaced on :class:`EpisodeRecord` exactly as the reader supplies it, and is ``None`` for
v1.0 files.  It is deliberately not part of an item: it is evaluation provenance, not a
model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "DRAW_SPLITS",
    "EpisodeRecord",
    "NormalizationStats",
    "RTG_QUANTILES",
    "RtgSummary",
    "STATS_VERSION",
    "TrajectoryWindowDataset",
    "WindowMeta",
    "collate_windows",
    "load_stats",
    "save_stats",
]

# Inclusive draw-id ranges.  Registered in PREREGISTRATION.md section 5 (decision D4,
# 2026-08-03) and binding on every collection and every loader.
DRAW_SPLITS: dict[str, tuple[int, int]] = {
    "nominal": (0, 0),
    "train": (1, 999),
    "heldout": (1000, 1099),
}

STATS_VERSION = "1.0"

# Recorded, never applied.  P4.3's probe-calibrated return prompting works in quantile
# space; computing these there from a second code path is how two pipelines start to
# disagree.
RTG_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

ITEM_KEYS: tuple[str, ...] = (
    "rtg",
    "state",
    "action",
    "avail_mask",
    "timestep",
    "attention_mask",
)


@dataclass(frozen=True)
class WindowMeta:
    """Provenance of one item, so any disagreement traces back to a row on disk."""

    episode_index: int
    dataset_dir: str
    episode_file: str
    scenario_id: str
    ix_id: str
    ix_index: int
    t: int
    flow_draw: int


@dataclass(frozen=True)
class EpisodeRecord:
    """One loaded episode's provenance and its v1.1-only ``att_per_step`` array."""

    dataset_dir: str
    episode_file: str
    scenario_id: str
    format_version: str
    flow_draw: int
    episode_length: int
    ix_ids: tuple[str, ...]
    att_per_step: np.ndarray | None


@dataclass(frozen=True)
class RtgSummary:
    """Descriptive statistics of one intersection's returns-to-go.  Nothing is applied."""

    count: int
    min: float
    max: float
    mean: float
    std: float
    quantiles: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class NormalizationStats:
    """State normalisation statistics frozen at construction, plus the RTG summary.

    Fitted on the **training split only** and carrying the exact draw ids they were fitted
    over: statistics fitted on held-out data are test-set leakage, a quieter form than
    training on it.  Keyed ``[scenario_id][ix_id]`` because intersections differ in
    ``state_dim`` both across and within scenarios.
    """

    stats_version: str
    split: str
    draw_ids: tuple[int, ...]
    dataset_dirs: tuple[str, ...]
    state_mean: dict[str, dict[str, np.ndarray]]
    state_std: dict[str, dict[str, np.ndarray]]
    row_count: dict[str, dict[str, int]]
    rtg: dict[str, dict[str, RtgSummary]]

    def normalize_state(self, scenario_id: str, ix_id: str, rows: np.ndarray) -> np.ndarray:
        """``(rows - mean) / std``, with a zero std left as 1.0 so a constant feature passes through."""
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping whose floats round-trip exactly."""
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> NormalizationStats:
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")


class TrajectoryWindowDataset(Dataset):
    """Per-intersection context windows over a set of collected dataset directories.

    See the module docstring for the alignment convention, the RTG definition, the padding
    convention and the draw-split rule.  Construction reads every selected episode once
    (through ``load_episode``), computes returns-to-go, and fits normalisation statistics
    unless frozen ones are supplied.  Nothing is written to disk.
    """

    def __init__(
        self,
        dataset_dirs: Sequence[str | Path],
        *,
        context_length: int,
        split: str = "train",
        draw_ids: Iterable[int] | None = None,
        stats: NormalizationStats | None = None,
        normalize: bool = True,
    ) -> None:
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    def __len__(self) -> int:
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    def item_meta(self, index: int) -> WindowMeta:
        """Provenance of item *index*: episode file, intersection id and ``t``."""
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    @property
    def episode_records(self) -> tuple[EpisodeRecord, ...]:
        """One record per loaded episode, in load order."""
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    @property
    def groups(self) -> dict[tuple[int, int], list[int]]:
        """``(state_dim, n_actions)`` -> item indices.  A batch must stay inside one group."""
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")

    @property
    def stats(self) -> NormalizationStats:
        """The frozen statistics: either fitted at construction or the ones supplied."""
        raise NotImplementedError("P3 skeleton: implementation follows the red tests")


def collate_windows(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack items, refusing a batch that straddles two ``(state_dim, n_actions)`` groups.

    ``default_collate`` already refuses such a batch, but names tensor shapes only.  This
    names both shapes so the reader knows which intersections collided.
    """
    raise NotImplementedError("P3 skeleton: implementation follows the red tests")


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    """Write *stats* as JSON, atomically, after all validation.

    Filesystem-mutation barrier: a rejected argument or a missing parent directory raises
    **before** anything is created, so a failed save leaves any previous file untouched and
    creates no directories.
    """
    raise NotImplementedError("P3 skeleton: implementation follows the red tests")


def load_stats(path: str | Path) -> NormalizationStats:
    raise NotImplementedError("P3 skeleton: implementation follows the red tests")
