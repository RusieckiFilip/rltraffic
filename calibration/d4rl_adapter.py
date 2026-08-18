"""P8.3: the D4RL side of the IQL calibration -- numpy and torch only, no simulator, no MuJoCo.

Artifact format version: ``p8.3-d4rl-calibration/1.0``.

WHAT THIS FILE IS FOR
---------------------
``BRIEF_24`` asks whether our IQL implementation is sound, by running **our own unchanged**
``offline.offline_baselines.train_iql`` on ``halfcheetah-medium-expert-v2`` and reporting the D4RL
normalised score.  Our trainer is discrete-action -- a categorical policy over ``n_actions`` and a
``Q`` head with one output per action -- while HalfCheetah's action space is ``Box(-1, 1, (6,))``.
**The adapter is a codebook on the action space and nothing else.**  No line of ``train_iql``,
``train_bc``, ``iql_targets``, ``MLPTrunk`` or ``masked_action_logits`` is modified, and none is
copied here.  Every choice is enumerated in ``docs/plans/p8.3.md`` section 3, A1-A11.

WHY THE MODULE IS SPLIT IN TWO
------------------------------
This half imports **only** numpy and torch, so the project's own test suite exercises the
load-bearing arithmetic on every machine, forever.  ``calibration/d4rl_run.py`` is the only file
that touches ``h5py``, ``gymnasium`` or ``mujoco``; it is imported by nothing here and by nothing
in ``offline/``.  ``tests/test_d4rl_calibration.py`` asserts that mechanically (Gate F), because a
convention that depends on remembering is weaker than a scan.

WHY ``calibration/`` IS NOT A PACKAGE
-------------------------------------
It carries no ``__init__.py`` on purpose.  ``tests/test_packaging.py`` requires every top-level
directory holding one to appear in ``[tool.setuptools.packages.find].include``, and this code must
**not** ship in the wheel: it depends on packages outside the project's dependency set and no paper
number flows through it.  A directory that is not a distributable package is exactly what that
means.  It also makes binding condition (ii) structural -- what cannot be imported as a package
cannot be imported by ``offline/`` by accident.

PROVENANCE OF THE PUBLISHED CONSTANTS
-------------------------------------
The two normalisation constants are D4RL's own, read from
``https://raw.githubusercontent.com/Farama-Foundation/d4rl/master/d4rl/infos.py`` on 2026-08-18:
``REF_MIN_SCORE`` lines 128-132 and ``REF_MAX_SCORE`` lines 218-222 give ``-280.178953`` and
``12135.0`` for every halfcheetah dataset, and the loop at line 287 propagates the ``-v0`` entries
to the ``-v1``/``-v2`` names unchanged.

The comparator scores are from the IQL paper itself -- Kostrikov, Nair & Levine, *Offline
Reinforcement Learning with Implicit Q-Learning*, arXiv:2110.06169, Table 1, column header
``Dataset | BC | 10%BC | DT | AWAC | Onestep RL | TD3+BC | CQL | IQL (Ours)`` -- retrieved
2026-08-18 and **verified by double-computation**: the paper's own ``locomotion-v2 total`` row
reproduces exactly from the summed BC column (466.7) and 10%BC column (666.2).

⚠️ ``BRIEF_24`` originally quoted a different IQL figure here, relayed rather than read from the
source.  **It is withdrawn** (brief corrected at ``302ae87``), and by the coordinator's ruling it is
deliberately absent from this module -- not even as a struck-through comparison -- so that it cannot
reach a results table and cannot be averaged with the verified one.  ``docs/plans/p8.3.md`` section
0.2 carries the withdrawn value, which is where a retracted number belongs, and
``tests/test_d4rl_calibration.py`` fails if it reappears under ``calibration/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from offline.dataset import STATS_VERSION, NormalizationStats, RtgSummary
from offline.offline_baselines import TransitionTable

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CODEBOOK_ITERATIONS",
    "CODEBOOK_SEED",
    "CODEBOOK_SIZES",
    "CODEBOOK_SUBSAMPLE",
    "DATASET_NAME",
    "DATASET_URL",
    "EVAL_EPISODES_PER_SEED",
    "PRIMARY_CODEBOOK_SIZE",
    "PUBLISHED_SCORES",
    "REF_MAX_SCORE",
    "REF_MIN_SCORE",
    "SCENARIO_ID",
    "STREAM_ID",
    "ActionCodebook",
    "EpisodeSpans",
    "bc_windows",
    "build_transition_table",
    "episode_returns",
    "episode_spans",
    "normalization_stats",
    "normalized_score",
    "top_return_episodes",
]

ARTIFACT_FORMAT_VERSION = "p8.3-d4rl-calibration/1.0"

DATASET_NAME = "halfcheetah-medium-expert-v2"
#: From d4rl's own ``DATASET_URLS`` (``infos.py`` line 299 builds the ``-v2`` names).
DATASET_URL = (
    "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"
    "halfcheetah_medium_expert-v2.hdf5"
)

#: d4rl ``infos.py``: every halfcheetah dataset shares these, ``-v0`` through ``-v2``.
REF_MIN_SCORE = -280.178953
REF_MAX_SCORE = 12135.0

#: arXiv:2110.06169 Table 1, ``halfcheetah-medium-expert-v2``.  Keys are OUR arm names.
PUBLISHED_SCORES: dict[str, float] = {"bc": 55.2, "bc_top10": 92.9, "iql": 86.7}

#: Declared in ``docs/plans/p8.3.md`` section 3 before the first gradient step.  All three sizes
#: are fitted and their quantisation ceilings measured **without training**, which is what makes
#: the primary unselectable on a result.
CODEBOOK_SIZES: tuple[int, ...] = (8, 64, 256)
PRIMARY_CODEBOOK_SIZE = 64
CODEBOOK_SEED = 0
CODEBOOK_ITERATIONS = 25
CODEBOOK_SUBSAMPLE = 200_000

EVAL_EPISODES_PER_SEED = 10

#: The corpus keys ``NormalizationStats`` is indexed by.  One scenario, one "intersection": D4RL
#: locomotion is a single agent, and the statistics class is keyed ``[scenario_id][ix_id]``
#: because our intersections differ in width, not because anything here does.
SCENARIO_ID = "d4rl"
STREAM_ID = "agent"


def normalized_score(
    episode_return: float | Sequence[float] | np.ndarray,
    *,
    ref_min: float = REF_MIN_SCORE,
    ref_max: float = REF_MAX_SCORE,
) -> np.ndarray:
    """The D4RL normalised score, ``100 * (r - ref_min) / (ref_max - ref_min)``.

    Returns a float64 array so a scalar and a vector take the same code path.  ``ref_min`` maps to
    exactly ``0.0`` and ``ref_max`` to exactly ``100.0``; the test asserts both with ``==``.
    """
    span = float(ref_max) - float(ref_min)
    if span <= 0.0:
        raise ValueError(
            f"the reference scores do not span a positive range (min {ref_min}, max {ref_max})"
        )
    values = np.asarray(episode_return, dtype=np.float64)
    return 100.0 * (values - float(ref_min)) / span


@dataclass(frozen=True)
class ActionCodebook:
    """A discrete codebook over a continuous action space: the whole of the adapter.

    ``centroids`` is ``(size, action_dim)`` float64.  Encoding is nearest-centroid in Euclidean
    distance with ties broken by the lowest index; decoding returns the centroid itself, clipped
    into ``[-1, 1]`` and cast to float32.  See ``docs/plans/p8.3.md`` section 3, A1-A5.
    """

    centroids: np.ndarray

    @classmethod
    def fit(
        cls,
        actions: np.ndarray,
        size: int,
        *,
        seed: int = CODEBOOK_SEED,
        iterations: int = CODEBOOK_ITERATIONS,
        subsample: int = CODEBOOK_SUBSAMPLE,
    ) -> ActionCodebook:
        """Lloyd's algorithm with k-means++ initialisation, on a fixed seeded subsample.

        Deterministic under *seed*: two fits produce bit-identical centroids.  An empty cluster
        keeps its previous centroid rather than being re-seeded, so an iteration can never move a
        centroid to a point no action occupies.
        """
        rows = np.asarray(actions, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError(f"actions must be (n, action_dim), got shape {rows.shape}")
        count = int(rows.shape[0])
        width = int(size)
        if width < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        if count < width:
            raise ValueError(f"cannot fit {width} code words to {count} actions")

        generator = np.random.default_rng(int(seed))
        if 0 < int(subsample) < count:
            rows = rows[generator.permutation(count)[: int(subsample)]]
            count = int(rows.shape[0])

        # k-means++ seeding: the first centre uniformly, each next one with probability
        # proportional to its squared distance from the nearest centre chosen so far.
        centroids = np.empty((width, rows.shape[1]), dtype=np.float64)
        centroids[0] = rows[int(generator.integers(count))]
        closest = np.sum((rows - centroids[0]) ** 2, axis=1)
        for position in range(1, width):
            total = float(closest.sum())
            if total > 0.0:
                picked = int(generator.choice(count, p=closest / total))
            else:
                # Every point already coincides with a centre; any row is as good as any other,
                # and a uniform draw keeps the fit deterministic instead of raising.
                picked = int(generator.integers(count))
            centroids[position] = rows[picked]
            closest = np.minimum(closest, np.sum((rows - centroids[position]) ** 2, axis=1))

        book = cls(centroids=centroids)
        for _ in range(int(iterations)):
            labels = book.encode(rows)
            moved = book.centroids.copy()
            for code in range(width):
                members = rows[labels == code]
                if members.shape[0]:
                    moved[code] = members.mean(axis=0)
            if np.array_equal(moved, book.centroids):
                break
            book = cls(centroids=moved)
        return book

    @property
    def size(self) -> int:
        """How many code words."""
        return int(self.centroids.shape[0])

    @property
    def action_dim(self) -> int:
        """Width of the continuous action space."""
        return int(self.centroids.shape[1])

    @property
    def digest(self) -> str:
        """sha256 over the centroid bytes in little-endian order, plus the shape."""
        rows = np.ascontiguousarray(self.centroids, dtype=np.float64)
        payload = hashlib.sha256()
        payload.update(f"codebook|{rows.shape}|{rows.dtype}|".encode("utf-8"))
        payload.update(rows.astype(rows.dtype.newbyteorder("<"), copy=False).tobytes())
        return payload.hexdigest()

    def encode(self, actions: np.ndarray, *, chunk: int = 8192) -> np.ndarray:
        """``(n, action_dim) -> (n,)`` int64 nearest-centroid indices.

        Exact squared Euclidean distances, computed in float64 and in chunks.  The expanded
        ``|a|^2 - 2 a.c + |c|^2`` form is deliberately NOT used: it is faster and it moves ties,
        and the test that checks this function recomputes the same quantity by an independent
        route and asserts **exact** index equality.
        """
        rows = np.asarray(actions, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.action_dim:
            raise ValueError(
                f"actions must be (n, {self.action_dim}), got shape {rows.shape}"
            )
        step = max(1, int(chunk))
        out = np.empty(rows.shape[0], dtype=np.int64)
        for start in range(0, rows.shape[0], step):
            block = rows[start : start + step]
            distances = np.sum((block[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)
            out[start : start + block.shape[0]] = np.argmin(distances, axis=1)
        return out

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """``(n,) -> (n, action_dim)`` float32 actions, clipped into ``[-1, 1]``."""
        codes = np.asarray(indices, dtype=np.int64)
        if codes.ndim != 1:
            raise ValueError(f"indices must be one-dimensional, got shape {codes.shape}")
        if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= self.size):
            raise ValueError(
                f"code indices must lie in [0, {self.size}), got "
                f"[{int(codes.min())}, {int(codes.max())}]"
            )
        return np.clip(self.centroids[codes], -1.0, 1.0).astype(np.float32)

    def quantisation_error(self, actions: np.ndarray, *, chunk: int = 8192) -> dict[str, float]:
        """Mean and max L2 distance from an action to its code word.  Nothing is selected here."""
        rows = np.asarray(actions, dtype=np.float64)
        residual = rows - self.centroids[self.encode(rows, chunk=chunk)]
        distance = np.linalg.norm(residual, axis=1)
        return {
            "mean_l2": float(distance.mean()),
            "max_l2": float(distance.max()),
            "actions": int(rows.shape[0]),
        }


@dataclass(frozen=True)
class EpisodeSpans:
    """Half-open ``[start, stop)`` row ranges, one per episode, in dataset order."""

    start: np.ndarray
    stop: np.ndarray

    def __len__(self) -> int:
        """How many episodes."""
        return int(self.start.shape[0])

    @property
    def lengths(self) -> np.ndarray:
        """``stop - start`` per episode, int64."""
        return (self.stop - self.start).astype(np.int64)

    @property
    def rows(self) -> np.ndarray:
        """Every dataset row covered by an episode, in order, int64."""
        if not len(self):
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(
            [np.arange(a, b, dtype=np.int64) for a, b in zip(self.start, self.stop)]
        )


def episode_spans(timeouts: np.ndarray, terminals: np.ndarray) -> EpisodeSpans:
    """Split the flat D4RL arrays into episodes at every ``timeout`` or ``terminal`` flag.

    **A trailing partial episode -- rows after the last flag -- is dropped**, declared in
    ``docs/plans/p8.3.md`` before the data was read.  It would otherwise enter
    ``iql_reward_scale``'s ``max - min`` as a short, artificially small return and move the one
    data-dependent quantity in the IQL configuration.  The number of dropped rows is reported.
    """
    ends = np.asarray(timeouts, dtype=bool) | np.asarray(terminals, dtype=bool)
    if ends.ndim != 1:
        raise ValueError(f"the flags must be one-dimensional, got shape {ends.shape}")
    boundary = np.flatnonzero(ends).astype(np.int64)
    if boundary.size == 0:
        raise ValueError(
            "no episode boundary in this dataset: neither timeouts nor terminals is ever set, "
            "so the rows cannot be split into episodes"
        )
    stop = boundary + 1
    start = np.concatenate([np.zeros(1, dtype=np.int64), stop[:-1]])
    if np.any(stop <= start):
        raise ValueError("an episode of length zero was produced; the flags are not monotone")
    return EpisodeSpans(start=start, stop=stop)


def episode_returns(rewards: np.ndarray, spans: EpisodeSpans) -> np.ndarray:
    """Undiscounted return per episode, float64.  This is what ``iql_reward_scale`` consumes."""
    values = np.asarray(rewards, dtype=np.float64)
    return np.array(
        [float(values[a:b].sum()) for a, b in zip(spans.start, spans.stop)], dtype=np.float64
    )


def top_return_episodes(returns: Sequence[float] | np.ndarray, fraction: float) -> np.ndarray:
    """Indices of the top *fraction* of episodes by return, highest first.

    ``ceil(fraction * n)`` episodes, at least one.  **Whole episodes only** -- the unit our own
    ``top_return_streams`` selects is a stream, never a fragment.  Ties are broken by the lower
    episode index so the selection is deterministic.
    """
    values = np.asarray(list(returns), dtype=np.float64)
    if values.size == 0:
        raise ValueError("top_return_episodes received no returns")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    keep = max(1, int(np.ceil(float(fraction) * values.size)))
    # ``kind="stable"`` on the negated returns puts the highest first and leaves ties in
    # ascending episode order, which is what makes the selection reproducible.
    return np.argsort(-values, kind="stable")[:keep].astype(np.int64)


def normalization_stats(
    observations: np.ndarray,
    *,
    returns: np.ndarray | None = None,
    scenario_id: str = SCENARIO_ID,
    stream_id: str = STREAM_ID,
) -> NormalizationStats:
    """Frozen ``(mean, std)`` over the training observations, in the repo's own container.

    D4RL ships no train/test split, so the statistics are fitted on every row that is trained on --
    stated because our corpus fits them on the training split only and the difference matters when
    someone compares the two.  ``normalize_state`` leaves a zero std as ``1.0``.

    The ``rtg`` summary is descriptive only: neither BC nor IQL conditions on a return, and nothing
    here reads it back.  It is filled from *returns* when they are supplied and left **NaN** when
    they are not, because a zero-filled summary in a checkpoint reads like a measurement.
    """
    rows = np.asarray(observations, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"observations must be (n, state_dim), got shape {rows.shape}")
    if rows.shape[0] < 2:
        raise ValueError("at least two observations are needed for a standard deviation")

    if returns is None:
        summary = RtgSummary(
            count=0,
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            quantiles=(),
        )
    else:
        values = np.asarray(returns, dtype=np.float64)
        summary = RtgSummary(
            count=int(values.size),
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
            quantiles=tuple(
                (float(q), float(np.quantile(values, q))) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            ),
        )

    return NormalizationStats(
        stats_version=STATS_VERSION,
        split="train",
        draw_ids=(),
        dataset_dirs=(DATASET_NAME,),
        state_mean={scenario_id: {stream_id: rows.mean(axis=0).astype(np.float32)}},
        state_std={scenario_id: {stream_id: rows.std(axis=0).astype(np.float32)}},
        row_count={scenario_id: {stream_id: int(rows.shape[0])}},
        rtg={scenario_id: {stream_id: summary}},
    )


def build_transition_table(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    action_index: np.ndarray,
    rewards: np.ndarray,
    spans: EpisodeSpans,
    stats: NormalizationStats,
    reward_scale: float,
    scenario_id: str = SCENARIO_ID,
    stream_id: str = STREAM_ID,
) -> TransitionTable:
    """Our own ``TransitionTable``, built row-for-row the way ``build_transitions`` builds it.

    ``next_state`` is the dataset's ``next_observations`` throughout, **including the last
    transition of every episode**, which is exactly the bootstrap target contract C6 is about and
    the reason ``iql_targets`` carries no ``done`` term.  Halfcheetah never terminates early, so
    every episode boundary here is a timeout and the no-``done`` convention is correct rather than
    merely tolerated -- Gate B asserts that property of the data instead of assuming it.

    States are normalised with *stats*; rewards are multiplied by *reward_scale*, which comes from
    ``iql_reward_scale`` on the episode returns.

    ``observations``, ``next_observations``, ``action_index`` and ``rewards`` are all indexed by
    **dataset row**; the rows a trailing partial episode would have contributed are simply never
    selected.
    """
    rows = spans.rows
    scale = float(reward_scale)
    state = stats.normalize_state(
        scenario_id, stream_id, np.asarray(observations, dtype=np.float32)[rows]
    )
    next_state = stats.normalize_state(
        scenario_id, stream_id, np.asarray(next_observations, dtype=np.float32)[rows]
    )
    action = np.asarray(action_index, dtype=np.int64)[rows]
    if action.ndim != 1:
        raise ValueError(f"action_index must be one-dimensional, got shape {action.shape}")
    reward = np.asarray(rewards, dtype=np.float32)[rows] * np.float32(scale)

    stream_index = np.repeat(np.arange(len(spans), dtype=np.int64), spans.lengths)
    step = np.concatenate([np.arange(n, dtype=np.int64) for n in spans.lengths])

    return TransitionTable(
        state=torch.from_numpy(np.ascontiguousarray(state)),
        next_state=torch.from_numpy(np.ascontiguousarray(next_state)),
        action=torch.from_numpy(np.ascontiguousarray(action)),
        reward=torch.from_numpy(np.ascontiguousarray(reward)),
        stream_index=torch.from_numpy(stream_index),
        t=torch.from_numpy(step),
        reward_scale=scale,
    )


def bc_windows(
    *,
    observations: np.ndarray,
    action_index: np.ndarray,
    spans: EpisodeSpans,
    stats: NormalizationStats,
    n_actions: int,
    context_length: int,
    episodes: Sequence[int] | np.ndarray | None = None,
    scenario_id: str = SCENARIO_ID,
    stream_id: str = STREAM_ID,
) -> dict[str, torch.Tensor]:
    """The ``stacked`` mapping ``train_bc`` consumes: ``state``, ``action``, ``avail_mask``.

    **Non-overlapping windows of *context_length* consecutive rows, never crossing an episode
    boundary.**  A window that crossed one would train BC on a discontinuity IQL never sees.  The
    remainder rows at the end of an episode that is not a multiple of *context_length* are dropped;
    for this dataset every episode is exactly 1000 rows, so the count is reported and expected to
    be zero.

    ``avail_mask`` is all-``True``: every mask in our corpus is (P4.4's docstring, P3 review), so
    this preserves the property the training code was written against while still flowing through
    ``masked_action_logits`` on the same code path.

    Passing *episodes* restricts the windows to those episode indices -- this is how ``bc_top10``
    is built, and it is the only difference between the two BC arms.
    """
    length = int(context_length)
    if length < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")
    chosen = (
        np.arange(len(spans), dtype=np.int64)
        if episodes is None
        else np.asarray(list(episodes), dtype=np.int64)
    )

    starts: list[np.ndarray] = []
    for episode in chosen:
        begin = int(spans.start[episode])
        count = int(spans.lengths[episode]) // length
        if count:
            starts.append(begin + np.arange(count, dtype=np.int64) * length)
    if not starts:
        raise ValueError(
            f"no window of length {length} fits inside any of the {chosen.size} selected "
            f"episodes (longest is {int(spans.lengths[chosen].max())} rows)"
        )

    offsets = np.concatenate(starts)[:, None] + np.arange(length, dtype=np.int64)[None, :]
    state = stats.normalize_state(
        scenario_id, stream_id, np.asarray(observations, dtype=np.float32)[offsets.reshape(-1)]
    ).reshape(offsets.shape[0], length, -1)
    action = np.asarray(action_index, dtype=np.int64)[offsets]
    mask = np.ones((offsets.shape[0], length, int(n_actions)), dtype=np.bool_)

    return {
        "state": torch.from_numpy(np.ascontiguousarray(state)),
        "action": torch.from_numpy(np.ascontiguousarray(action)),
        "avail_mask": torch.from_numpy(mask),
    }
