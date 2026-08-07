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
The loop in :func:`_returns_to_go` is deliberately not written with ``np.cumsum`` -- a test
that recomputes a quantity with the same call it is checking verifies little.

Padding
-------
Windows shorter than ``K`` are padded **on the left**.  Padded positions are zero -- ``state``
0.0, ``rtg`` 0.0, ``timestep`` 0, ``avail_mask`` all-False, ``attention_mask`` False -- with
one deliberate exception: **``action`` pads with** :data:`PAD_ACTION` **= -1, not 0** (ruled
2026-08-07, superseding the brief).

``attention_mask`` remains the mechanism; ``-1`` is the tripwire behind it.  A padded 0 is a
*legal* action, so a loss that forgets the mask would train on fabricated action-0 targets in
silence; ``-1`` crashes both ``nn.Embedding`` and ``cross_entropy`` instead, and makes
``ignore_index=-1`` the explicit spelling of P4's loss.  ``action == -1`` holds exactly where
``attention_mask`` is False.  This is the **in-memory item only**: the on-disk format is
unchanged and no corpus needs regenerating.

Draw-aware splitting (D4, registered in ``PREREGISTRATION.md`` section 5)
-------------------------------------------------------------------------
``nominal`` = draw 0 (source vehicle order, never pooled) · ``train`` = draws 1-999 ·
``heldout`` = draws 1000-1099, which must never enter any training corpus for any method.
Episodes are selected **by draw id read from the manifest**, never by file order.  With
``draw_ids=None`` a draw outside the requested split **raises** rather than being skipped: a
leak that raises is a bug, a leak that skips is a published number.  With an explicit
``draw_ids=`` the requested ids are range-checked against the split *first*, and episodes
outside that explicit request are then filtered out -- a narrowing the caller asked for, not a
silent skip.  On both paths no episode outside the split can enter the dataset.  A
``flow_draw`` of ``-1`` -- the sentinel the logger writes when a collection ran with no draw
at all -- is refused under every split, because such a corpus has no draw identity to split on.

The manifest's draw id is cross-checked against the episode file's own ``flow_draw`` scalar,
because the split is only as trustworthy as the weaker of its two recorded sources.

Normalisation statistics
------------------------
Per ``(scenario_id, intersection_id)``: ``state_mean`` and ``state_std`` (``ddof=0``,
accumulated in float64, stored float32) over observation rows ``0..T-1`` -- the rows actually
fed to a model.  Applied as ``(x - mean) / std`` with a zero std left as 1.0, **before**
padding, so padded rows stay exactly 0.0 instead of ``-mean/std``.

Statistics are fitted **only when ``split="train"``**.  Fitting them on evaluation data is
test-set leakage, a quieter form than training on it; an evaluation dataset must therefore be
handed the frozen training statistics through ``stats=``.  A non-training split without
supplied statistics still constructs (its provenance and episodes are readable) but raises as
soon as normalised items or ``.stats`` are requested, rather than silently normalising with
numbers fitted on the evaluation pool.  ``save_stats`` records the split and the exact draw
ids the statistics were fitted over, so the question "what were these fitted on?" is
answerable from the file.

An RTG summary (min/max/mean/std and the quantiles in :data:`RTG_QUANTILES`) is recorded per
``(scenario_id, intersection_id)`` and **applied to nothing**: P4.3's probe-calibrated return
prompting works in quantile space, and computing those quantiles there from a second code
path is how two pipelines start to disagree.

Heterogeneous intersections
---------------------------
Intersections may differ in ``state_dim`` and ``n_actions`` (cologne3 does: 16/20/24 and
3/4/4), and C6 forbids padding across them.  Items are therefore not universally
collatable.  :attr:`TrajectoryWindowDataset.groups` maps ``(state_dim, n_actions)`` to the
indices in that group; batches must stay inside one group.  Mixing groups already fails
loudly -- ``torch.utils.data.default_collate`` raises when it tries to stack tensors of
different shapes, which is a mechanical guarantee rather than a convention -- but its message
names shapes only, so :func:`collate_windows` checks first and names both ``(D, A)`` pairs.

Format version and the ATT field
--------------------------------
Version support is delegated to :func:`offline.trajectory_logger.load_episode`; this module
adds only the rule that **one dataset may not span two format versions** (BRIEF_08 section 5:
a v1.0 and a v1.1 corpus must never be silently mixed).  ``att_per_step`` -- the per-row
observation array added in v1.1, whose horizon element is the registered primary metric -- is
surfaced on :class:`EpisodeRecord` exactly as the reader supplies it, and is ``None`` for
v1.0 files.  It is deliberately not part of an item: it is evaluation provenance, not a model
input.

⚠️ ``att_per_step`` is stored **float32** while an episode-horizon value recomputed in a
consumer is typically **float64** (measured 2026-08-07 by P2.6: bit-identical against
``np.float32(att_horizon)``, but 5.03e-06 apart when compared raw).  **Cast the float64 value
down with** ``np.float32(...)``; never widen the stored array instead, which hides a real
mismatch behind a tolerance.

Which spellings actually trip, measured on numpy 2.5.1 rather than assumed -- the exception is
the part that misleads::

    stored_f32 == <python float>                True    NEP 50: a Python float is a *weak*
                                                        scalar, so it is cast DOWN for you
    float(stored_f32) == computed_f64           False   both float64; nothing rounds it away
    stored_f32 == np.float64(computed)          False   a strong float64 promotes the pair UP
    np.array_equal(att_f32, <float64 array>)    False   a plain list builds a float64 array
    stored_f32 == np.float32(computed)          True    the correct comparison

A consumer is therefore safe *by accident* on a bare scalar comparison, and unsafe the moment
the value passes through ``float()``, ``np.float64`` or an array.  Do not rely on the accident.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from offline import trajectory_logger
from offline.trajectory_logger import load_episode

__all__ = [
    "ABSENT_DRAW",
    "DRAW_SPLITS",
    "EpisodeRecord",
    "NormalizationStats",
    "PAD_ACTION",
    "RTG_QUANTILES",
    "RtgSummary",
    "STATS_VERSION",
    "TrajectoryWindowDataset",
    "WindowMeta",
    "collate_windows",
    "load_stats",
    "save_stats",
]

# Padded action positions.  NOT 0, which is a legal action: -1 makes a forgotten
# attention_mask crash in nn.Embedding / cross_entropy instead of silently training on
# fabricated targets, and is the value P4's loss should pass as ignore_index.
PAD_ACTION: int = -1

# The logger writes flow_draw = -1 when an episode was collected with no draw at all.
ABSENT_DRAW: int = -1

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

MANIFEST_NAME = "manifest.json"


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
    """One loaded episode's provenance and its v1.1-only ``att_per_step`` array.

    ``att_per_step`` is ``None`` on v1.0 files.  It is float32; see the module docstring
    before comparing it to a float64 horizon value.
    """

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
        """``(rows - mean) / std``, with a zero std left as 1.0 so a constant feature passes through.

        Computed in float32 throughout: widening to float64 here would make the result
        depend on where normalisation happens rather than on the statistics.
        """
        try:
            mean = self.state_mean[scenario_id][ix_id]
            std = self.state_std[scenario_id][ix_id]
        except KeyError as exc:
            raise KeyError(
                f"no normalisation statistics for scenario {scenario_id!r} intersection "
                f"{ix_id!r}; the statistics were fitted on "
                f"{sorted(self.state_mean)} and cannot be reused here"
            ) from exc
        rows32 = np.asarray(rows, dtype=np.float32)
        if rows32.shape[-1] != mean.shape[-1]:
            raise ValueError(
                f"state width {rows32.shape[-1]} does not match the statistics for "
                f"{scenario_id}/{ix_id} (width {mean.shape[-1]})"
            )
        safe = np.where(std > 0, std, np.float32(1.0)).astype(np.float32)
        return (rows32 - mean) / safe

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping whose floats round-trip exactly.

        float32 values are widened to Python floats, which ``repr`` round-trips exactly, so
        ``save -> load -> save`` is byte-identical.
        """
        return {
            "stats_version": self.stats_version,
            "split": self.split,
            "draw_ids": list(self.draw_ids),
            "dataset_dirs": list(self.dataset_dirs),
            "state_mean": {
                scenario: {ix_id: [float(v) for v in vec] for ix_id, vec in per_ix.items()}
                for scenario, per_ix in self.state_mean.items()
            },
            "state_std": {
                scenario: {ix_id: [float(v) for v in vec] for ix_id, vec in per_ix.items()}
                for scenario, per_ix in self.state_std.items()
            },
            "row_count": {
                scenario: {ix_id: int(n) for ix_id, n in per_ix.items()}
                for scenario, per_ix in self.row_count.items()
            },
            "rtg": {
                scenario: {
                    ix_id: {
                        "count": int(summary.count),
                        "min": float(summary.min),
                        "max": float(summary.max),
                        "mean": float(summary.mean),
                        "std": float(summary.std),
                        "quantiles": [[float(q), float(v)] for q, v in summary.quantiles],
                    }
                    for ix_id, summary in per_ix.items()
                }
                for scenario, per_ix in self.rtg.items()
            },
        }

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> NormalizationStats:
        version = str(payload.get("stats_version", ""))
        if version != STATS_VERSION:
            raise ValueError(
                f"statistics version {version!r} is not readable by this build "
                f"(expected {STATS_VERSION!r})"
            )
        return cls(
            stats_version=version,
            split=str(payload["split"]),
            draw_ids=tuple(int(d) for d in payload["draw_ids"]),
            dataset_dirs=tuple(str(p) for p in payload["dataset_dirs"]),
            state_mean={
                scenario: {
                    ix_id: np.asarray(vec, dtype=np.float32) for ix_id, vec in per_ix.items()
                }
                for scenario, per_ix in payload["state_mean"].items()
            },
            state_std={
                scenario: {
                    ix_id: np.asarray(vec, dtype=np.float32) for ix_id, vec in per_ix.items()
                }
                for scenario, per_ix in payload["state_std"].items()
            },
            row_count={
                scenario: {ix_id: int(n) for ix_id, n in per_ix.items()}
                for scenario, per_ix in payload["row_count"].items()
            },
            rtg={
                scenario: {
                    ix_id: RtgSummary(
                        count=int(entry["count"]),
                        min=float(entry["min"]),
                        max=float(entry["max"]),
                        mean=float(entry["mean"]),
                        std=float(entry["std"]),
                        quantiles=tuple(
                            (float(q), float(v)) for q, v in entry["quantiles"]
                        ),
                    )
                    for ix_id, entry in per_ix.items()
                }
                for scenario, per_ix in payload["rtg"].items()
            },
        )


@dataclass(frozen=True)
class _Stream:
    """One (episode, intersection) pair: the arrays a window is cut from."""

    episode_index: int
    ix_id: str
    ix_index: int
    state: np.ndarray            # (T+1, D) float32 -- rows 0..T-1 are inputs
    avail_mask: np.ndarray       # (T+1, A) bool
    action: np.ndarray           # (T,)    int64
    rtg: np.ndarray              # (T,)    float32
    length: int                  # T


def _returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """Suffix sums of *rewards*, inclusive of ``r_t``, accumulated backwards in float64.

    Written as an explicit loop rather than ``np.cumsum`` on a reversed view: the tests
    recompute this with ``np.cumsum``, and a check that shares its implementation with the
    thing it checks is not a check.  The accumulation order is identical, so the two agree
    bit-for-bit and the test can assert ``==`` instead of a tolerance.
    """
    length = int(rewards.shape[0])
    out = np.empty(length, dtype=np.float64)
    total = 0.0
    for t in range(length - 1, -1, -1):
        total += float(rewards[t])
        out[t] = total
    return out.astype(np.float32)


def _supported_format_versions() -> tuple[str, ...]:
    """Whatever ``trajectory_logger`` says it can read -- never a second hardcoded list.

    P2.6 replaces the single ``FORMAT_VERSION`` with ``SUPPORTED_FORMAT_VERSIONS``; reading
    it through ``getattr`` means this module gains v1.1 support the moment that lands, and
    cannot drift from the reader in the meantime.
    """
    versions = getattr(trajectory_logger, "SUPPORTED_FORMAT_VERSIONS", None)
    if versions is None:
        return (str(trajectory_logger.FORMAT_VERSION),)
    return tuple(str(v) for v in versions)


def _draw_range_error(draws: Sequence[int], split: str, where: str) -> ValueError:
    """One message for every out-of-split draw, naming the ids and why it matters."""
    low, high = DRAW_SPLITS[split]
    held_low, held_high = DRAW_SPLITS["heldout"]
    offenders = sorted(set(int(d) for d in draws))
    held_out = [d for d in offenders if held_low <= d <= held_high]
    message = (
        f"{where}: draws {offenders} are outside the {split!r} split "
        f"(draws {low}-{high} inclusive)"
    )
    if held_out:
        message += (
            f"; draws {held_out} are in the registered held-out evaluation pool "
            f"({held_low}-{held_high}, decision D4 / PREREGISTRATION.md section 5) and must "
            "never enter a training corpus, for any method"
        )
    else:
        message += (
            "; draw 0 is the nominal control and is reported separately, never pooled with "
            "randomised draws"
        )
    return ValueError(message)


class TrajectoryWindowDataset(Dataset):
    """Per-intersection context windows over a set of collected dataset directories.

    See the module docstring for the alignment convention, the RTG definition, the padding
    convention and the draw-split rule.  Construction reads every selected episode once
    (through ``load_episode``), computes returns-to-go, and fits normalisation statistics
    when ``split="train"``.  **Nothing is written to disk.**

    Memory: only ``state``, ``avail_mask``, ``action`` and the computed RTG are retained per
    intersection; lane and metric arrays are released once the episode has been read.
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
        # -------- validation first; nothing is read or built until all of it passes ------
        context = int(context_length)
        if context < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length!r}")
        if split not in DRAW_SPLITS:
            raise ValueError(
                f"unknown split {split!r}; expected one of {sorted(DRAW_SPLITS)}"
            )
        low, high = DRAW_SPLITS[split]

        dirs = [Path(d) for d in dataset_dirs]
        if not dirs:
            raise ValueError("dataset_dirs is empty: nothing to load")
        for directory in dirs:
            if not (directory / MANIFEST_NAME).is_file():
                raise FileNotFoundError(
                    f"{directory}: no {MANIFEST_NAME}; a dataset directory is one collection "
                    "run written by offline.trajectory_logger"
                )

        requested: tuple[int, ...] | None = None
        if draw_ids is not None:
            requested = tuple(sorted({int(d) for d in draw_ids}))
            if not requested:
                raise ValueError("draw_ids is empty: pass None to take every draw in the split")
            outside = [d for d in requested if not low <= d <= high]
            if outside:
                raise _draw_range_error(outside, split, "requested draw_ids")

        manifests = [
            json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            for directory in dirs
        ]

        versions = {str(manifest.get("format_version")) for manifest in manifests}
        if len(versions) > 1:
            raise ValueError(
                f"these directories span format versions {sorted(versions)}; one dataset may "
                "not mix format versions, because the two corpora are not interchangeable "
                "(BRIEF_08 section 5)"
            )
        version = versions.pop()
        supported = _supported_format_versions()
        if version not in supported:
            raise ValueError(
                f"format version {version!r} is not readable by this build (supported: "
                f"{list(supported)}); offline.trajectory_logger.load_episode defines that set"
            )

        # -------- select episodes by draw id, never by file order -----------------------
        selected: list[tuple[Path, dict[str, Any], str, int]] = []
        found_draws: set[int] = set()
        for directory, manifest in zip(dirs, manifests):
            entries = manifest.get("episodes", [])
            listed = {str(entry["filename"]) for entry in entries}
            on_disk = {path.name for path in directory.glob("*.npz")}
            missing = sorted(listed - on_disk)
            extra = sorted(on_disk - listed)
            if missing:
                raise FileNotFoundError(
                    f"{directory}: the manifest lists {len(missing)} episode file(s) that are "
                    f"not on disk, first {missing[:3]}"
                )
            if extra:
                raise ValueError(
                    f"{directory}: {len(extra)} .npz file(s) on disk are absent from the "
                    f"manifest, first {extra[:3]}; loading them would be selecting data by "
                    "file order instead of by draw id"
                )

            scenario_id = str(manifest.get("run_metadata", {}).get("scenario_id", ""))
            if not scenario_id:
                raise ValueError(f"{directory}: manifest has no run_metadata.scenario_id")

            # The absent-draw sentinel is refused before the range check, and under every
            # split including an explicit draw_ids: an episode collected with no draw has no
            # draw identity, so there is no split it can honestly belong to.  Not reachable
            # from today's corpus (draws 1-200) but exactly the route a no-draw baseline
            # collection takes.
            absent = sorted(
                {
                    str(entry["filename"])
                    for entry in entries
                    if int(entry["flow_draw"]) == ABSENT_DRAW
                }
            )
            if absent:
                raise ValueError(
                    f"{directory}: {len(absent)} episode(s) record flow_draw "
                    f"{ABSENT_DRAW} -- the absent-draw sentinel the logger writes when a "
                    f"collection ran with no flow draw at all (first {absent[:3]}).  Such a "
                    "corpus carries no draw identity and cannot be assigned to the nominal, "
                    "training or held-out pool (decision D4); recollect it with a draw id"
                )

            out_of_split = sorted(
                {int(entry["flow_draw"]) for entry in entries if not low <= int(entry["flow_draw"]) <= high}
            )
            if out_of_split and requested is None:
                raise _draw_range_error(out_of_split, split, str(directory))

            for entry in entries:
                draw = int(entry["flow_draw"])
                if not low <= draw <= high:
                    continue                      # requested is not None: an explicit ask filters
                if requested is not None and draw not in requested:
                    continue
                found_draws.add(draw)
                selected.append((directory, entry, scenario_id, draw))

        if requested is not None:
            absent = sorted(set(requested) - found_draws)
            if absent:
                raise ValueError(
                    f"requested draws {absent} are not present in {[str(d) for d in dirs]}"
                )
        if not selected:
            raise ValueError(
                f"no episodes in the {split!r} split (draws {low}-{high}) across "
                f"{[str(d) for d in dirs]}"
            )

        # -------- read the episodes -----------------------------------------------------
        records: list[EpisodeRecord] = []
        streams: list[_Stream] = []
        for episode_index, (directory, entry, scenario_id, draw) in enumerate(selected):
            filename = str(entry["filename"])
            episode = load_episode(directory / filename)

            if str(episode.format_version) != version:
                raise ValueError(
                    f"{directory / filename}: episode reports format version "
                    f"{episode.format_version!r} while its manifest says {version!r}"
                )
            if int(episode.flow_draw) != draw:
                raise ValueError(
                    f"{directory / filename}: manifest flow_draw {draw} disagrees with the "
                    f"episode's own flow_draw {int(episode.flow_draw)}; the draw split is "
                    "only as trustworthy as the weaker of its two recorded sources"
                )

            length = int(episode.episode_length)
            for ix_index, ix_id in enumerate(episode.ix_ids):
                arrays = episode.intersections[ix_id]
                self._check_stream_shapes(directory / filename, ix_id, arrays, length)
                rewards = np.asarray(arrays.local_reward, dtype=np.float32)
                if bool(np.isnan(rewards).any()):
                    raise ValueError(
                        f"{directory / filename}: intersection {ix_id!r} has NaN local_reward, "
                        "which means the run had no local reward function; returns-to-go "
                        "cannot be built from it"
                    )
                streams.append(
                    _Stream(
                        episode_index=episode_index,
                        ix_id=ix_id,
                        ix_index=ix_index,
                        state=np.asarray(arrays.state, dtype=np.float32),
                        avail_mask=np.asarray(arrays.avail_mask, dtype=np.bool_),
                        action=np.asarray(arrays.action, dtype=np.int64),
                        rtg=_returns_to_go(rewards),
                        length=length,
                    )
                )

            records.append(
                EpisodeRecord(
                    dataset_dir=str(directory),
                    episode_file=filename,
                    scenario_id=scenario_id,
                    format_version=version,
                    flow_draw=draw,
                    episode_length=length,
                    ix_ids=tuple(episode.ix_ids),
                    att_per_step=getattr(episode, "att_per_step", None),
                )
            )

        # -------- flat item index, in load order --------------------------------------
        index: list[tuple[int, int]] = []
        groups: dict[tuple[int, int], list[int]] = {}
        for stream_index, stream in enumerate(streams):
            key = (int(stream.state.shape[1]), int(stream.avail_mask.shape[1]))
            for t in range(stream.length):
                groups.setdefault(key, []).append(len(index))
                index.append((stream_index, t))

        self._context_length = context
        self._split = split
        self._normalize = bool(normalize)
        self._dirs = tuple(str(d) for d in dirs)
        self._records = tuple(records)
        self._streams = tuple(streams)
        self._index = tuple(index)
        self._groups = groups
        self._draw_ids = tuple(sorted(found_draws))

        # -------- statistics: fitted on the training split only -------------------------
        if stats is not None:
            self._stats: NormalizationStats | None = stats
        elif split == "train":
            self._stats = self._fit_stats()
        else:
            self._stats = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_stream_shapes(path: Path, ix_id: str, arrays: Any, length: int) -> None:
        """Observations carry T+1 rows, decisions and outcomes T.  Verify, never assume."""
        expected = {
            "state": (length + 1, arrays.state.shape[1] if arrays.state.ndim == 2 else -1),
            "avail_mask": (
                length + 1,
                arrays.avail_mask.shape[1] if arrays.avail_mask.ndim == 2 else -1,
            ),
        }
        for name, (rows, _width) in expected.items():
            actual = getattr(arrays, name).shape[0]
            if actual != rows:
                raise ValueError(
                    f"{path}: intersection {ix_id!r} has {actual} rows of {name} for an "
                    f"episode of length {length} (expected {rows}); the file violates the "
                    "C6 alignment convention"
                )
        for name in ("action", "local_reward"):
            actual = getattr(arrays, name).shape[0]
            if actual != length:
                raise ValueError(
                    f"{path}: intersection {ix_id!r} has {actual} rows of {name} for an "
                    f"episode of length {length} (expected {length}); the file violates the "
                    "C6 alignment convention"
                )

    def _fit_stats(self) -> NormalizationStats:
        """Two passes in float64: mean, then variance about that mean (``ddof=0``).

        Two passes rather than a sum-of-squares shortcut: ``E[x^2] - mean^2`` cancels
        catastrophically on features with a large offset and a small spread, which is exactly
        what a queue-count state looks like late in a congested episode.
        """
        scenario_of = {i: record.scenario_id for i, record in enumerate(self._records)}

        totals: dict[tuple[str, str], np.ndarray] = {}
        counts: dict[tuple[str, str], int] = {}
        for stream in self._streams:
            key = (scenario_of[stream.episode_index], stream.ix_id)
            rows = stream.state[: stream.length]
            column_sum = np.sum(rows, axis=0, dtype=np.float64)
            if key in totals:
                totals[key] += column_sum
                counts[key] += rows.shape[0]
            else:
                totals[key] = column_sum
                counts[key] = rows.shape[0]
        means = {key: totals[key] / counts[key] for key in totals}

        squares: dict[tuple[str, str], np.ndarray] = {}
        rtg_values: dict[tuple[str, str], list[np.ndarray]] = {}
        for stream in self._streams:
            key = (scenario_of[stream.episode_index], stream.ix_id)
            rows = stream.state[: stream.length].astype(np.float64)
            deviation = np.sum((rows - means[key]) ** 2, axis=0)
            if key in squares:
                squares[key] += deviation
            else:
                squares[key] = deviation
            rtg_values.setdefault(key, []).append(stream.rtg)

        state_mean: dict[str, dict[str, np.ndarray]] = {}
        state_std: dict[str, dict[str, np.ndarray]] = {}
        row_count: dict[str, dict[str, int]] = {}
        rtg: dict[str, dict[str, RtgSummary]] = {}
        for (scenario_id, ix_id), mean in means.items():
            variance = squares[(scenario_id, ix_id)] / counts[(scenario_id, ix_id)]
            state_mean.setdefault(scenario_id, {})[ix_id] = mean.astype(np.float32)
            state_std.setdefault(scenario_id, {})[ix_id] = np.sqrt(variance).astype(np.float32)
            row_count.setdefault(scenario_id, {})[ix_id] = int(counts[(scenario_id, ix_id)])
            values = np.concatenate(rtg_values[(scenario_id, ix_id)]).astype(np.float64)
            rtg.setdefault(scenario_id, {})[ix_id] = RtgSummary(
                count=int(values.size),
                min=float(values.min()),
                max=float(values.max()),
                mean=float(values.mean()),
                std=float(values.std()),
                quantiles=tuple(
                    (float(q), float(np.quantile(values, q, method="linear")))
                    for q in RTG_QUANTILES
                ),
            )

        return NormalizationStats(
            stats_version=STATS_VERSION,
            split=self._split,
            draw_ids=self._draw_ids,
            dataset_dirs=self._dirs,
            state_mean=state_mean,
            state_std=state_std,
            row_count=row_count,
            rtg=rtg,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        position = int(index)
        if position < 0:
            position += len(self._index)
        if not 0 <= position < len(self._index):
            raise IndexError(f"item index {index} out of range for {len(self._index)} windows")

        stream_index, t = self._index[position]
        stream = self._streams[stream_index]
        context = self._context_length
        low = max(0, t - context + 1)
        span = t - low + 1
        start = context - span                      # padding sits on the LEFT

        state = np.zeros((context, stream.state.shape[1]), dtype=np.float32)
        rows = stream.state[low : t + 1]
        if self._normalize:
            record = self._records[stream.episode_index]
            rows = self._require_stats().normalize_state(record.scenario_id, stream.ix_id, rows)
        state[start:] = rows

        rtg = np.zeros((context, 1), dtype=np.float32)
        rtg[start:, 0] = stream.rtg[low : t + 1]

        action = np.full(context, PAD_ACTION, dtype=np.int64)
        action[start:] = stream.action[low : t + 1]

        avail_mask = np.zeros((context, stream.avail_mask.shape[1]), dtype=np.bool_)
        avail_mask[start:] = stream.avail_mask[low : t + 1]

        timestep = np.zeros(context, dtype=np.int64)
        timestep[start:] = np.arange(low, t + 1, dtype=np.int64)

        attention_mask = np.zeros(context, dtype=np.bool_)
        attention_mask[start:] = True

        return {
            "rtg": torch.from_numpy(rtg),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "avail_mask": torch.from_numpy(avail_mask),
            "timestep": torch.from_numpy(timestep),
            "attention_mask": torch.from_numpy(attention_mask),
        }

    def item_meta(self, index: int) -> WindowMeta:
        """Provenance of item *index*: episode file, intersection id and ``t``."""
        position = int(index)
        if position < 0:
            position += len(self._index)
        if not 0 <= position < len(self._index):
            raise IndexError(f"item index {index} out of range for {len(self._index)} windows")
        stream_index, t = self._index[position]
        stream = self._streams[stream_index]
        record = self._records[stream.episode_index]
        return WindowMeta(
            episode_index=stream.episode_index,
            dataset_dir=record.dataset_dir,
            episode_file=record.episode_file,
            scenario_id=record.scenario_id,
            ix_id=stream.ix_id,
            ix_index=stream.ix_index,
            t=int(t),
            flow_draw=record.flow_draw,
        )

    @property
    def episode_records(self) -> tuple[EpisodeRecord, ...]:
        """One record per loaded episode, in load order."""
        return self._records

    @property
    def groups(self) -> dict[tuple[int, int], list[int]]:
        """``(state_dim, n_actions)`` -> item indices.  A batch must stay inside one group."""
        return {key: list(value) for key, value in self._groups.items()}

    @property
    def stats(self) -> NormalizationStats:
        """The frozen statistics: either fitted at construction or the ones supplied."""
        return self._require_stats()

    def _require_stats(self) -> NormalizationStats:
        if self._stats is None:
            raise ValueError(
                f"no normalisation statistics: this dataset was built with "
                f"split={self._split!r}, and statistics are fitted only for split='train'.  "
                "THE FIX: pass stats= fitted on split='train' -- load them with "
                "offline.dataset.load_stats(path) from the file save_stats wrote during "
                "training, and hand them to this constructor.  Fitting them here would fit "
                "them on evaluation data, which is test-set leakage (decision D3).  "
                "normalize=False is a diagnostic escape hatch, not the intended path"
            )
        return self._stats


def collate_windows(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack items, refusing a batch that straddles two ``(state_dim, n_actions)`` groups.

    ``default_collate`` already refuses such a batch, but names tensor shapes only.  This
    names both ``(state_dim, n_actions)`` pairs.  It cannot name the intersections behind
    them -- an item deliberately carries no metadata -- so use
    :meth:`TrajectoryWindowDataset.groups` to build batch indices, and
    :meth:`TrajectoryWindowDataset.item_meta` to identify the offenders.
    """
    batch = list(items)
    if not batch:
        raise ValueError("collate_windows received an empty batch")
    shapes = sorted(
        {(int(item["state"].shape[-1]), int(item["avail_mask"].shape[-1])) for item in batch}
    )
    if len(shapes) > 1:
        raise ValueError(
            f"batch straddles {len(shapes)} intersection shapes {shapes} "
            "(state_dim, n_actions); C6 forbids padding across intersections, so a batch "
            "must stay inside one group -- see TrajectoryWindowDataset.groups"
        )
    return default_collate(batch)


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    """Write *stats* as JSON, atomically, after all validation.

    Filesystem-mutation barrier: a rejected argument or a missing parent directory raises
    **before** anything is created, so a failed save leaves any previous file untouched and
    creates no directories.
    """
    if not isinstance(stats, NormalizationStats):
        raise TypeError(
            f"save_stats expects a NormalizationStats, got {type(stats).__name__}"
        )
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"destination directory does not exist: {parent}; save_stats never creates one"
        )
    payload = json.dumps(stats.to_json_obj(), indent=2, sort_keys=True) + "\n"

    handle, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=".stats-", suffix=".json.tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_stats(path: str | Path) -> NormalizationStats:
    """Read statistics written by :func:`save_stats`."""
    source = Path(path)
    return NormalizationStats.from_json_obj(
        json.loads(source.read_text(encoding="utf-8"))
    )
