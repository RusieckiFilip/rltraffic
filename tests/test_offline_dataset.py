"""Tests for ``offline.dataset`` -- per-intersection context windows over the offline corpus.

No corpus and no simulator required: every test here writes its own format-v1.0 episode
files.  The real-corpus tests live in ``tests/test_offline_dataset_corpus.py``.

THE FIXTURE IS BUILT SO THAT WRONG IMPLEMENTATIONS ARE VISIBLE, which is the point of the
2026-08-07 rule that a fixture satisfying its assertion by construction is a tautology:

* ``ix_ids`` is **not** sorted, and the two ids sort in the opposite order to their env
  order, so positional keying gives a different answer rather than the same one;
* the two intersections have different ``state_dim`` and ``n_actions`` (never padded across),
  which also gives the group tests two real groups;
* their reward streams differ at every step, so a positional mix-up changes the numbers;
* every state row differs from its neighbours, so a row-shifted window is visible;
* ``avail_mask`` genuinely binds -- on odd rows only the taken action is available -- because
  the real corpus's masks are all-True and would make the mask test vacuous;
* one reward is deliberately ``0`` (``ix_zulu``, t=2), which makes the RTG off-by-one
  *invisible at that row*: a miniature of the population measurement (all 4800 episodes,
  32000 streams -- blind on 0.84% of hz1x1 rows, 25.65% of cologne3, **34.35%** of grid4x4;
  ``docs/plans/p3.md`` section 4 quotes the earlier 138-episode sample, whose grid4x4 figure
  was 37.63%).  It is the reason the RTG assertions compare whole arrays, never a sampled row.
* all values are exactly representable in float32, so every comparison is ``==`` and never a
  tolerance.

DISCRIMINATING POWER.  Each alignment test reports the distance between the correct answer
and the deliberately wrong one via :func:`mismatch_rows`, and asserts that distance exactly
rather than asserting "it passed".  A check whose wrong answer is close to its right answer
certifies little however green it looks.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

import offline.dataset as dataset_module
from offline.dataset import (
    ABSENT_DRAW,
    DRAW_SPLITS,
    PAD_ACTION,
    RTG_QUANTILES,
    NormalizationStats,
    TrajectoryWindowDataset,
    collate_windows,
    load_stats,
    save_stats,
)
from offline.trajectory_logger import load_episode

# ----------------------------------------------------------------------
# Fixture definition
# ----------------------------------------------------------------------

# (intersection id, state_dim, n_actions).  Env order is NOT lexicographic order: "ix_zulu"
# sorts after "ix_alpha" but is stored first, exactly as env.intersections order can differ
# from sorted order on a real roadnet.
IX_SPECS: tuple[tuple[str, int, int], ...] = (
    ("ix_zulu", 3, 2),
    ("ix_alpha", 4, 3),
)
IX_IDS: tuple[str, ...] = tuple(spec[0] for spec in IX_SPECS)
assert IX_IDS != tuple(sorted(IX_IDS)), "fixture must distinguish env order from sorted order"

T = 8                       # decision rows; observations carry T+1 = 9
K = 3                       # context window used by the window/padding tests
SCENARIO_ID = "fixture_2ix"
LANE_IDS: tuple[str, ...] = ("lane_a", "lane_b", "lane_c")
METRIC_KEYS: tuple[str, ...] = (
    "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation",
)

# Base reward streams, one per intersection, integer-valued so math.fsum and a float64
# accumulation agree bit-for-bit.  ix_zulu carries the deliberate zero at t=2.
BASE_REWARDS: dict[str, tuple[int, ...]] = {
    "ix_zulu": (-3, -1, 0, -4, -2, -5, -1, -2),
    "ix_alpha": (-7, -2, -6, -1, -3, -8, -4, -6),
}
assert all(len(v) == T for v in BASE_REWARDS.values()), "reward streams must cover T rows"
assert 0 in BASE_REWARDS["ix_zulu"], "fixture needs a blind row for the off-by-one margin"
assert all(
    BASE_REWARDS["ix_zulu"][t] != BASE_REWARDS["ix_alpha"][t] for t in range(T)
), "streams must differ at every step so a positional mix-up changes the numbers"

# Waiting-vehicle totals per observation row.  Rows 1..T are pinned to -sum_j r_j[t] so the
# fixture reproduces the exact local identity of BRIEF_09 section 2:
#     sum_j ix{j}_local_reward[t] == -sum lane_waiting_vehicle_count[t+1]
# Row 0 is free and deliberately unequal to row 1, so the misaligned variant differs.
_REWARD_TOTALS: tuple[int, ...] = tuple(
    -(BASE_REWARDS["ix_zulu"][t] + BASE_REWARDS["ix_alpha"][t]) for t in range(T)
)
LANE_WAITING_ROWS: tuple[tuple[int, ...], ...] = (
    (3, 2, 2),                                    # row 0: total 7, != row 1's total
) + tuple(
    (total - 2 * (total // 3), total // 3, total // 3) for total in _REWARD_TOTALS
)
assert len(LANE_WAITING_ROWS) == T + 1
assert all(sum(row) == _REWARD_TOTALS[t] for t, row in enumerate(LANE_WAITING_ROWS[1:]))
assert sum(LANE_WAITING_ROWS[0]) != sum(LANE_WAITING_ROWS[1])


# The LAST state feature is constant everywhere -- every row, episode and intersection -- so
# its fitted std is exactly 0 and the zero-std branch of normalisation is reachable.  Without
# it that branch is dead code in the suite while being load-bearing on the real corpus, where
# 191/640 grid4x4 std entries are exactly 0 (review P3-10).
CONSTANT_FEATURE_VALUE = 7.0

# Per-step average travel time for the v1.1 fixture, as a consumer would compute it: in
# float64.  Deliberately NON-MONOTONE (P2.6 measured 407/407 real episodes falsifying the
# monotonicity the brief once asserted) and deliberately NOT exactly representable in
# float32, so the storage round-trip is lossy and the float32-vs-float64 comparison trap is
# reproducible rather than hypothetical.
ATT_SOURCE_FLOAT64: tuple[float, ...] = (
    10.1, 10.3, 10.7, 10.2, 11.4, 11.9, 12.35, 12.1, 12.8,
)
assert len(ATT_SOURCE_FLOAT64) == T + 1, "att_per_step is an OBSERVATION: T+1 rows"
assert any(
    ATT_SOURCE_FLOAT64[t + 1] < ATT_SOURCE_FLOAT64[t] for t in range(T)
), "fixture must be non-monotone, or it cannot detect a monotonicity assumption"
# The trap is asserted on the HORIZON element specifically, so it is the horizon that must
# lose precision.  "Some value in the sequence does" is not enough: the first draft of this
# fixture ended in 12.75, which is dyadic and therefore float32-exact, and the trap test
# failed against a perfectly correct loader.
assert float(np.float32(ATT_SOURCE_FLOAT64[-1])) != ATT_SOURCE_FLOAT64[-1], (
    "the horizon value must lose precision in float32, or the comparison trap is invisible"
)
assert all(
    float(np.float32(v)) != v for v in ATT_SOURCE_FLOAT64
), "every fixture value should be float32-lossy, so no assertion here is accidentally exact"


def _state_rows(ix_index: int, state_dim: int, ep_index: int) -> np.ndarray:
    """``t + 10*d + 100*ix + 1000*ep``, except a constant final column.

    Integer-valued, so pooled mean/std stay exact and the statistics test can assert ``==``.
    Every row differs from its neighbour (a shifted window is visible) and every intersection
    differs from the other (a positional mix-up is visible); the constant column reaches the
    zero-std guard.
    """
    rows = np.empty((T + 1, state_dim), dtype=np.float32)
    for t in range(T + 1):
        for d in range(state_dim):
            if d == state_dim - 1:
                rows[t, d] = CONSTANT_FEATURE_VALUE
            else:
                rows[t, d] = float(t + 10 * d + 100 * ix_index + 1000 * ep_index)
    return rows


def _actions(n_actions: int) -> np.ndarray:
    return np.asarray([t % n_actions for t in range(T)], dtype=np.int64)


def _avail_mask(n_actions: int) -> np.ndarray:
    """All actions available on even rows; on odd rows ONLY the action actually taken.

    The taken action is always available, so the loader's mask handling is tested against a
    mask that genuinely binds -- unlike the real corpus, where every mask is all-True.
    """
    taken = _actions(n_actions)
    mask = np.zeros((T + 1, n_actions), dtype=np.bool_)
    for t in range(T + 1):
        for a in range(n_actions):
            if t == T:
                mask[t, a] = True                       # no decision on the final row
            elif t % 2 == 0:
                mask[t, a] = True
            else:
                mask[t, a] = a == int(taken[t])
    return mask


def _episode_arrays(*, ep_index: int, draw: int, format_version: str, nan_rewards: bool) -> dict[str, np.ndarray]:
    """Build one episode exactly as ``TrajectoryLogger._build_arrays`` lays it out."""
    scale = ep_index + 1                                # episodes differ, so pooling is real
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(format_version),
        "ix_ids": np.asarray(list(IX_IDS), dtype=np.str_),
        "lane_ids": np.asarray(list(LANE_IDS), dtype=np.str_),
        "metric_keys": np.asarray(list(METRIC_KEYS), dtype=np.str_),
        "vehicle_count": np.asarray([10 + t for t in range(T + 1)], dtype=np.int64),
        "sim_time": np.asarray([10.0 * t for t in range(T + 1)], dtype=np.float32),
        "step": np.asarray(list(range(T + 1)), dtype=np.int64),
        "metrics": np.asarray(
            [[float(sum(LANE_WAITING_ROWS[t]) * scale)] for t in range(T + 1)],
            dtype=np.float32,
        ),
        "lane_vehicle_count": np.asarray(
            [[v * scale + 1 for v in row] for row in LANE_WAITING_ROWS], dtype=np.int32
        ),
        "lane_waiting_vehicle_count": np.asarray(
            [[v * scale for v in row] for row in LANE_WAITING_ROWS], dtype=np.int32
        ),
        "episode_length": np.asarray(T, dtype=np.int64),
        "terminated": np.asarray(False, dtype=np.bool_),
        "truncated": np.asarray(True, dtype=np.bool_),
        "engine_seed": np.asarray(1000 + ep_index, dtype=np.int64),
        "flow_draw": np.asarray(draw, dtype=np.int64),
    }

    # v1.1 = v1.0 plus exactly one array.  Stored float32, as the format requires, from a
    # float64 source -- which is precisely the lossy step the comparison trap lives in.
    if format_version == "1.1":
        arrays["att_per_step"] = np.asarray(ATT_SOURCE_FLOAT64, dtype=np.float32)

    global_reward = np.zeros(T, dtype=np.float32)
    for i, (ix_id, state_dim, n_actions) in enumerate(IX_SPECS):
        rewards = np.asarray(
            [r * scale for r in BASE_REWARDS[ix_id]], dtype=np.float32
        )
        if nan_rewards:
            rewards = np.full(T, np.nan, dtype=np.float32)
        else:
            global_reward += rewards
        arrays[f"ix{i}_state"] = _state_rows(i, state_dim, ep_index)
        arrays[f"ix{i}_avail_mask"] = _avail_mask(n_actions)
        arrays[f"ix{i}_current_phase"] = np.asarray(
            [t % n_actions for t in range(T + 1)], dtype=np.int64
        )
        arrays[f"ix{i}_time_in_phase"] = np.asarray(
            [float(t % 4) for t in range(T + 1)], dtype=np.float32
        )
        arrays[f"ix{i}_action"] = _actions(n_actions)
        arrays[f"ix{i}_local_reward"] = rewards
    arrays["global_reward"] = global_reward
    return arrays


def _digest(arrays: dict[str, np.ndarray]) -> str:
    """The logger's own recipe: actions in env order, then ``global_reward``, little-endian."""
    digest = hashlib.sha256()
    for i in range(len(IX_IDS)):
        digest.update(arrays[f"ix{i}_action"].astype("<i8", copy=False).tobytes())
    digest.update(arrays["global_reward"].astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def write_dataset_dir(
    root: Path,
    name: str,
    *,
    draws: Sequence[int] = (1, 2),
    scenario_id: str = SCENARIO_ID,
    format_version: str = "1.0",
    nan_rewards: bool = False,
    manifest_draw_override: dict[int, int] | None = None,
) -> Path:
    """Write a collection directory: one ``.npz`` per draw plus a ``manifest.json``.

    ``manifest_draw_override`` maps an episode index to the draw the *manifest* claims,
    leaving the ``.npz`` scalar untouched -- the disagreement the loader must refuse.
    """
    out_dir = root / name
    out_dir.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []
    for ep_index, draw in enumerate(draws):
        arrays = _episode_arrays(
            ep_index=ep_index, draw=draw, format_version=format_version, nan_rewards=nan_rewards
        )
        filename = f"ep{ep_index:06d}_seed{1000 + ep_index}_draw{draw}.npz"
        with open(out_dir / filename, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        claimed = (manifest_draw_override or {}).get(ep_index, draw)
        episodes.append(
            {
                "filename": filename,
                "episode_length": T,
                "total_global_reward": float(arrays["global_reward"].sum()),
                "engine_seed": 1000 + ep_index,
                "flow_draw": claimed,
                "episode_sha256": _digest(arrays),
            }
        )
    manifest = {
        "format_version": format_version,
        "git_hash": "0" * 40,
        "lane_count": len(LANE_IDS),
        "lane_ids_sha256": hashlib.sha256("\n".join(LANE_IDS).encode("utf-8")).hexdigest(),
        "run_metadata": {
            "scenario_id": scenario_id,
            "backend": "cityflow",
            "behavior_policy": "fixture",
            "global_reward_weight": 0.0,
            "local_reward_fn": "queue_length",
        },
        "episodes": episodes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_dir


# ----------------------------------------------------------------------
# Shared helpers -- used by BOTH the correctness assertion and its sensitivity control,
# so the control cannot pass by testing a different comparison.
# ----------------------------------------------------------------------


def mismatch_rows(actual: np.ndarray, expected: np.ndarray) -> int:
    """Number of positions where two arrays disagree.  0 means bit-identical."""
    return int(np.count_nonzero(np.asarray(actual) != np.asarray(expected)))


def raw_arrays(dataset_dir: Path, filename: str) -> dict[str, np.ndarray]:
    """Read an episode straight from the ``.npz`` -- the independent route."""
    with np.load(dataset_dir / filename) as data:
        return {key: data[key] for key in data.files}


def index_of(ds: TrajectoryWindowDataset, ix_id: str, t: int, episode_file: str | None = None) -> int:
    for i in range(len(ds)):
        meta = ds.item_meta(i)
        if meta.ix_id == ix_id and meta.t == t:
            if episode_file is None or meta.episode_file == episode_file:
                return i
    raise AssertionError(f"no item for ix_id={ix_id!r} t={t} file={episode_file!r}")


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A two-episode training-pool collection directory (draws 1 and 2)."""
    return write_dataset_dir(tmp_path, "fixture__policy")


# ----------------------------------------------------------------------
# The fixture itself must be a real format file
# ----------------------------------------------------------------------


def test_fixture_is_a_real_format_v10_file(corpus: Path) -> None:
    """``load_episode`` reads the fixture, so the fixture is not a private dialect.

    Without this control every other test in the file proves only that the loader agrees
    with the test author's idea of the on-disk format.
    """
    episode = load_episode(corpus / "ep000000_seed1000_draw1.npz")
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")

    assert episode.ix_ids == IX_IDS
    assert episode.episode_length == T
    assert episode.terminated is False and episode.truncated is True
    for i, ix_id in enumerate(IX_IDS):
        loaded = episode.intersections[ix_id]
        assert np.array_equal(loaded.state, written[f"ix{i}_state"])
        assert np.array_equal(loaded.action, written[f"ix{i}_action"])
        assert np.array_equal(loaded.local_reward, written[f"ix{i}_local_reward"])
        assert np.array_equal(loaded.avail_mask, written[f"ix{i}_avail_mask"])


def test_fixture_reproduces_the_local_reward_lane_identity(corpus: Path) -> None:
    """``sum_j local_reward[t] == -sum lane_waiting[t+1]``, with the misaligned margin reported.

    This is the exact identity BRIEF_09 section 2 verified bit-exactly on 92/92 hz1x1 and
    grid4x4 episodes.  Reproducing it in the fixture means the alignment convention is
    checked even when no corpus is present.
    """
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    local_sum = sum(written[f"ix{i}_local_reward"] for i in range(len(IX_IDS)))
    waiting = written["lane_waiting_vehicle_count"].astype(np.float32)

    aligned = -waiting[1:].sum(axis=1)
    misaligned = -waiting[:-1].sum(axis=1)

    assert mismatch_rows(local_sum, aligned) == 0
    # Discriminating power: the wrong pairing disagrees on 7 of 8 rows.  It coincides at
    # t=4 by arithmetic accident, which is exactly why the assertion is whole-array.
    assert mismatch_rows(local_sum, misaligned) == 7


# ----------------------------------------------------------------------
# Returns-to-go -- the load-bearing group
# ----------------------------------------------------------------------


def test_rtg_matches_reversed_cumsum_read_straight_from_the_npz(corpus: Path) -> None:
    """RTG equals ``np.cumsum`` over the reversed reward array, whole array, ``==``.

    The independent route reads the reward stream from the ``.npz`` and never touches the
    loader's helper (BRIEF_09 section 3.2).
    """
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    for filename in ("ep000000_seed1000_draw1.npz", "ep000001_seed1001_draw2.npz"):
        written = raw_arrays(corpus, filename)
        for i, ix_id in enumerate(IX_IDS):
            rewards = written[f"ix{i}_local_reward"]
            expected = np.cumsum(rewards[::-1].astype(np.float64))[::-1].astype(np.float32)
            item = ds[index_of(ds, ix_id, T - 1, filename)]
            actual = item["rtg"].numpy().reshape(-1)
            assert mismatch_rows(actual, expected) == 0


def test_rtg_matches_math_fsum_third_route(corpus: Path) -> None:
    """A third, exactly-rounded route: ``math.fsum`` of each suffix.

    Exact agreement with a float64 accumulation is guaranteed only for integer-valued
    rewards, so the precondition is asserted rather than assumed: the day a non-integer
    reward enters the corpus this test says so instead of quietly loosening.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    for i, ix_id in enumerate(IX_IDS):
        rewards = written[f"ix{i}_local_reward"].astype(np.float64)
        assert np.array_equal(rewards, np.floor(rewards)), "fsum route needs integer rewards"
        expected = np.asarray(
            [math.fsum(rewards[t:]) for t in range(T)], dtype=np.float32
        )
        actual = ds[index_of(ds, ix_id, T - 1)]["rtg"].numpy().reshape(-1)
        assert mismatch_rows(actual, expected) == 0


def test_rtg_endpoints_are_inclusive_of_the_current_reward(corpus: Path) -> None:
    """``rtg[T-1] == r[T-1]`` and ``rtg[0] == sum(r)`` -- the alignment stated as endpoints."""
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    for i, ix_id in enumerate(IX_IDS):
        rewards = written[f"ix{i}_local_reward"]
        rtg = ds[index_of(ds, ix_id, T - 1)]["rtg"].numpy().reshape(-1)
        assert rtg[T - 1] == rewards[T - 1]
        assert rtg[0] == np.float32(math.fsum(rewards.astype(np.float64)))


def test_offbyone_rtg_is_detected_and_its_margin_is_measured(corpus: Path) -> None:
    """The off-by-one differs from the correct RTG at exactly the nonzero-reward rows.

    ``rtg_correct[t] - rtg_offbyone[t] == r_t``, so this is the sensitivity control for the
    two tests above, and it measures the check's discriminating power instead of assuming
    it: ix_zulu carries a zero reward at t=2 and is therefore blind at that row.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    margins: dict[str, int] = {}
    for i, ix_id in enumerate(IX_IDS):
        rewards = written[f"ix{i}_local_reward"]
        correct = ds[index_of(ds, ix_id, T - 1)]["rtg"].numpy().reshape(-1)
        off_by_one = np.append(correct[1:], np.float32(0.0))   # exclusive of r_t
        margins[ix_id] = mismatch_rows(off_by_one, correct)
        assert margins[ix_id] == int(np.count_nonzero(rewards))
        assert np.array_equal(correct - off_by_one, rewards)

    assert margins["ix_zulu"] == T - 1, "one blind row expected: r=0 at t=2"
    assert margins["ix_alpha"] == T, "no zero rewards, so every row discriminates"


# ----------------------------------------------------------------------
# Windows, padding, dtypes
# ----------------------------------------------------------------------


def test_item_shapes_and_dtypes(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    for ix_id, state_dim, n_actions in IX_SPECS:
        item = ds[index_of(ds, ix_id, T - 1)]
        assert set(item) == {"rtg", "state", "action", "avail_mask", "timestep", "attention_mask"}
        assert item["rtg"].shape == (K, 1) and item["rtg"].dtype is torch.float32
        assert item["state"].shape == (K, state_dim) and item["state"].dtype is torch.float32
        assert item["action"].shape == (K,) and item["action"].dtype is torch.int64
        assert item["avail_mask"].shape == (K, n_actions) and item["avail_mask"].dtype is torch.bool
        assert item["timestep"].shape == (K,) and item["timestep"].dtype is torch.int64
        assert item["attention_mask"].shape == (K,) and item["attention_mask"].dtype is torch.bool


def test_full_window_rows_match_the_npz(corpus: Path) -> None:
    """A window ending at t covers rows t-K+1..t of every per-row array."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    t = T - 1
    for i, ix_id in enumerate(IX_IDS):
        item = ds[index_of(ds, ix_id, t, "ep000000_seed1000_draw1.npz")]
        lo = t - K + 1
        assert np.array_equal(item["state"].numpy(), written[f"ix{i}_state"][lo : t + 1])
        assert np.array_equal(item["action"].numpy(), written[f"ix{i}_action"][lo : t + 1])
        assert np.array_equal(item["avail_mask"].numpy(), written[f"ix{i}_avail_mask"][lo : t + 1])
        assert np.array_equal(item["timestep"].numpy(), np.arange(lo, t + 1))
        assert bool(item["attention_mask"].all())


def test_left_padding_marks_exactly_the_real_steps(corpus: Path) -> None:
    """For t < K-1 the mask is False on the left and True for the t+1 real steps."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    for t in range(K):
        item = ds[index_of(ds, "ix_zulu", t, "ep000000_seed1000_draw1.npz")]
        expected = np.asarray([False] * (K - 1 - t) + [True] * (t + 1))
        assert np.array_equal(item["attention_mask"].numpy(), expected)


def test_padded_positions_are_zero_except_the_action_tripwire(corpus: Path) -> None:
    """Padded positions are zero / False -- but ``action`` pads with -1, not 0.

    Ruled 2026-08-07, superseding the brief's "padded positions are zero": a padded 0 is a
    *legal* action, so a loss that forgets ``attention_mask`` would train on fabricated
    action-0 targets in silence.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    item = ds[index_of(ds, "ix_zulu", 0, "ep000000_seed1000_draw1.npz")]
    pad = ~item["attention_mask"].numpy()
    assert int(pad.sum()) == K - 1
    assert not item["state"].numpy()[pad].any()
    assert not item["rtg"].numpy().reshape(-1)[pad].any()
    assert not item["timestep"].numpy()[pad].any()
    assert not item["avail_mask"].numpy()[pad].any()
    assert np.array_equal(item["action"].numpy()[pad], np.full(K - 1, PAD_ACTION))
    # ...and the single real row is the episode's first row, not a shifted one.
    assert np.array_equal(item["state"].numpy()[-1], written["ix0_state"][0])


def test_padded_action_is_the_ignore_index_tripwire(corpus: Path) -> None:
    """``action == -1`` holds exactly where ``attention_mask`` is False, and nowhere else.

    That equivalence is what lets P4 spell its loss ``ignore_index=PAD_ACTION`` and get a
    crash rather than a silent wrong target if the mask is ever dropped.
    """
    assert PAD_ACTION == -1
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    checked_padded = 0
    for i in range(len(ds)):
        item = ds[i]
        action = item["action"].numpy()
        real = item["attention_mask"].numpy()
        assert np.array_equal(action == PAD_ACTION, ~real)
        assert (action[real] >= 0).all(), "a real action must never collide with the sentinel"
        checked_padded += int((~real).sum())
    assert checked_padded > 0, "fixture must contain padded windows for this to mean anything"


# ----------------------------------------------------------------------
# Keying, truncation, masks
# ----------------------------------------------------------------------


def test_items_are_keyed_by_intersection_id_not_position(corpus: Path) -> None:
    """Each item carries the arrays of ITS intersection, on a fixture whose ids are unsorted.

    The sorted-position answer is computed too and asserted to be *different*, so the test
    cannot pass under positional keying.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    sorted_ids = sorted(IX_IDS)
    for ix_id in IX_IDS:
        env_index = IX_IDS.index(ix_id)
        sorted_index = sorted_ids.index(ix_id)
        assert env_index != sorted_index, "fixture premise: the two orders must disagree"

        item = ds[index_of(ds, ix_id, T - 1)]
        by_id = written[f"ix{env_index}_local_reward"]
        by_position = written[f"ix{sorted_index}_local_reward"]
        expected = np.cumsum(by_id[::-1].astype(np.float64))[::-1].astype(np.float32)
        wrong = np.cumsum(by_position[::-1].astype(np.float64))[::-1].astype(np.float32)

        actual = item["rtg"].numpy().reshape(-1)
        assert mismatch_rows(actual, expected) == 0
        assert mismatch_rows(expected, wrong) == T, "positional keying must differ at every row"
        assert ds.item_meta(index_of(ds, ix_id, T - 1)).ix_index == env_index


def test_item_count_is_T_windows_per_intersection_per_episode(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    assert len(ds) == 2 * len(IX_SPECS) * T
    assert {ds.item_meta(i).t for i in range(len(ds))} == set(range(T))


def test_last_window_ends_at_the_last_decision_row_not_the_final_observation(corpus: Path) -> None:
    """Observation row T is never an input: the last window ends at row T-1.

    Truncation is honoured by never emitting the post-final observation, not by treating it
    as absorbing.  The check discriminates because state row T-1 differs from row T.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    assert max(ds.item_meta(i).t for i in range(len(ds))) == T - 1

    item = ds[index_of(ds, "ix_zulu", T - 1, "ep000000_seed1000_draw1.npz")]
    last_row = item["state"].numpy()[-1]
    assert np.array_equal(last_row, written["ix0_state"][T - 1])
    assert not np.array_equal(written["ix0_state"][T - 1], written["ix0_state"][T])
    assert not np.array_equal(last_row, written["ix0_state"][T])


def test_avail_mask_never_excludes_the_taken_action(corpus: Path) -> None:
    """On a mask that genuinely binds -- odd rows offer only the taken action."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    binding_rows = 0
    for i in range(len(ds)):
        item = ds[i]
        real = item["attention_mask"].numpy()
        mask = item["avail_mask"].numpy()
        actions = item["action"].numpy()
        for row in np.flatnonzero(real):
            assert bool(mask[row, actions[row]])
            binding_rows += int(not mask[row].all())
    assert binding_rows > 0, "fixture must contain masks that actually bind"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_two_constructions_give_byte_identical_items(corpus: Path) -> None:
    first = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    second = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    assert len(first) == len(second)
    for i in range(len(first)):
        a, b = first[i], second[i]
        for key in a:
            assert a[key].numpy().tobytes() == b[key].numpy().tobytes()


def test_seeded_dataloader_is_byte_identical_across_runs(corpus: Path) -> None:
    """Same seed, same indices, byte-identical batches -- twice over."""
    def run() -> list[bytes]:
        ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
        group = ds.groups[(IX_SPECS[0][1], IX_SPECS[0][2])]
        loader = DataLoader(
            Subset(ds, group),
            batch_size=4,
            shuffle=True,
            collate_fn=collate_windows,
            generator=torch.Generator().manual_seed(20260807),
        )
        return [
            batch[key].numpy().tobytes() for batch in loader for key in sorted(batch)
        ]

    assert run() == run()


# ----------------------------------------------------------------------
# Draw-aware splitting (D4)
# ----------------------------------------------------------------------


def test_split_ranges_match_the_registered_split() -> None:
    assert DRAW_SPLITS["nominal"] == (0, 0)
    assert DRAW_SPLITS["train"] == (1, 999)
    assert DRAW_SPLITS["heldout"] == (1000, 1099)


def test_heldout_draws_refused_for_training(tmp_path: Path) -> None:
    """Requesting 1000-1099 as training data raises; D4 forbids it for every method."""
    out_dir = write_dataset_dir(tmp_path, "held__policy", draws=(1000, 1001))
    with pytest.raises(ValueError, match="held-out"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train", draw_ids=[1000])


def test_out_of_split_draw_present_in_a_directory_raises(tmp_path: Path) -> None:
    """Encountered-but-not-requested held-out draws raise; they are never silently skipped."""
    out_dir = write_dataset_dir(tmp_path, "mixed__policy", draws=(1, 2, 1000))
    with pytest.raises(ValueError, match="1000"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train")


def test_draw_zero_is_refused_by_train_and_accepted_by_nominal(tmp_path: Path) -> None:
    """Draw 0 is the nominal control: never pooled with randomised draws."""
    nominal_dir = write_dataset_dir(tmp_path, "nominal__policy", draws=(0,))
    # match= names the condition: a bare "0" would be satisfied by almost any error text,
    # including one raised for an entirely different reason (review P3-08).
    with pytest.raises(ValueError, match=r"draws \[0\] are outside the 'train' split"):
        TrajectoryWindowDataset([nominal_dir], context_length=K, split="train")

    ds = TrajectoryWindowDataset([nominal_dir], context_length=K, split="nominal")
    assert {record.flow_draw for record in ds.episode_records} == {0}


def test_explicit_draw_ids_narrow_within_the_split(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", draw_ids=[2])
    assert {record.flow_draw for record in ds.episode_records} == {2}
    assert len(ds) == len(IX_SPECS) * T


def test_unknown_split_raises(corpus: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        TrajectoryWindowDataset([corpus], context_length=K, split="validation")


def test_invalid_context_length_raises(corpus: Path) -> None:
    with pytest.raises(ValueError, match="context_length"):
        TrajectoryWindowDataset([corpus], context_length=0, split="train")


def test_explicit_draw_ids_narrow_a_directory_that_also_holds_other_draws(tmp_path: Path) -> None:
    """The narrowing path, pinned: an explicit request filters, and cannot reach outside the split.

    "Raise, never skip" is unconditional only when ``draw_ids`` is None.  With an explicit
    request the ids are range-checked first and the rest are filtered out, which is a
    narrowing the caller asked for -- but it is a *different* code path and the docstring used
    to state the rule flatly (review P3-07).  No leak either way: the held-out episode below
    cannot be reached even by asking for it (see the held-out refusal test).
    """
    out_dir = write_dataset_dir(tmp_path, "wide__policy", draws=(1, 2, 1000))

    ds = TrajectoryWindowDataset([out_dir], context_length=K, split="train", draw_ids=[1])
    assert {record.flow_draw for record in ds.episode_records} == {1}
    assert len(ds) == len(IX_SPECS) * T

    # ...while the same directory without an explicit request refuses to load at all.
    with pytest.raises(ValueError, match=r"draws \[1000\] are outside the 'train' split"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train")


def test_absent_draw_sentinel_is_refused_under_every_split(tmp_path: Path) -> None:
    """``flow_draw = -1`` means "collected with no draw": no split can honestly hold it.

    Not reachable from today's corpus (draws 1-200), but it is exactly the route a no-draw
    baseline collection takes, and the old code refused it with a message about draw 0
    (review P3-06).
    """
    assert ABSENT_DRAW == -1
    out_dir = write_dataset_dir(tmp_path, "nodraw__policy", draws=(ABSENT_DRAW,))
    for split in ("train", "nominal", "heldout"):
        with pytest.raises(ValueError, match="absent-draw sentinel"):
            TrajectoryWindowDataset([out_dir], context_length=K, split=split)
    # ...and asking for it explicitly does not get around the refusal either.
    with pytest.raises(ValueError, match="outside the 'train' split"):
        TrajectoryWindowDataset(
            [out_dir], context_length=K, split="train", draw_ids=[ABSENT_DRAW]
        )


def test_manifest_and_npz_draw_disagreement_raises(tmp_path: Path) -> None:
    """The draw id has two recorded sources; they must agree or the split means nothing."""
    out_dir = write_dataset_dir(
        tmp_path, "liar__policy", draws=(1, 2), manifest_draw_override={1: 7}
    )
    with pytest.raises(ValueError, match="flow_draw"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train")


def test_nan_local_reward_raises(tmp_path: Path) -> None:
    """A corpus collected without a local reward function cannot produce an RTG."""
    out_dir = write_dataset_dir(tmp_path, "nan__policy", draws=(1,), nan_rewards=True)
    with pytest.raises(ValueError, match="NaN"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train")


# ----------------------------------------------------------------------
# Format versions and the v1.1 ATT field
# ----------------------------------------------------------------------


def test_mixed_format_versions_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A v1.0 and a v1.1 corpus must never be silently mixed (BRIEF_08 section 5).

    **Both versions are declared supported for this test**, through the same seam the loader
    reads at run time.  Without that, the pair would be refused for being *unsupported*
    rather than for being *mixed*, the homogeneity guard could be deleted with the test still
    green -- and it would go on passing at exactly the moment v1.1 lands and the guard's only
    real job begins (review P3-03).  The ``match=`` names the mixing, not the word "format".
    """
    monkeypatch.setattr(dataset_module, "_supported_format_versions", lambda: ("1.0", "1.1"))
    old = write_dataset_dir(tmp_path, "old__policy", draws=(1,), format_version="1.0")
    new = write_dataset_dir(tmp_path, "new__policy", draws=(2,), format_version="1.1")

    # The seam is real: each directory alone is now accepted, so only the mixing can raise.
    assert len(TrajectoryWindowDataset([old], context_length=K, split="train")) > 0

    with pytest.raises(ValueError, match=r"span format versions \['1\.0', '1\.1'\]"):
        TrajectoryWindowDataset([old, new], context_length=K, split="train")


def test_unsupported_format_version_is_rejected(tmp_path: Path) -> None:
    """An unknown version is refused by its own message, distinct from the mixing one."""
    out_dir = write_dataset_dir(tmp_path, "future__policy", draws=(1,), format_version="9.9")
    with pytest.raises(ValueError, match=r"'9\.9' is not readable by this build"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train")


def test_att_per_step_is_none_for_v10_episodes(corpus: Path) -> None:
    """v1.0 carries no ATT array; the field exists and is None rather than absent."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    assert [record.att_per_step for record in ds.episode_records] == [None, None]
    assert {record.format_version for record in ds.episode_records} == {"1.0"}


def test_att_per_step_is_populated_and_aligned_on_a_v11_fixture(tmp_path: Path) -> None:
    """The POSITIVE v1.1 branch: the ATT array arrives, with T+1 rows and float32 dtype.

    Until P2.6 merged, only the ``None`` branch of the ATT gate could be exercised, and this
    test could not exist.  It closes that merge gate on the synthetic side; the real-corpus
    side is in ``tests/test_offline_dataset_corpus.py``.

    ``att_per_step`` is an OBSERVATION, not an outcome: ``T+1`` rows against the ``T`` rows of
    ``action`` and ``local_reward``.  Getting that backwards is the same class of off-by-one
    the RTG tests exist to catch, so it is asserted by row count, not by eye.
    """
    out_dir = write_dataset_dir(tmp_path, "v11__policy", draws=(1, 2), format_version="1.1")
    ds = TrajectoryWindowDataset([out_dir], context_length=K, split="train")

    assert {record.format_version for record in ds.episode_records} == {"1.1"}
    for record in ds.episode_records:
        att = record.att_per_step
        assert att is not None, "a v1.1 episode must carry att_per_step, not None"
        assert att.shape == (T + 1,), "att_per_step is an observation: T+1 rows"
        assert att.dtype == np.float32
        # Same file, read by a route that does not go through the loader.
        written = raw_arrays(out_dir, record.episode_file)
        assert np.array_equal(att, written["att_per_step"])
        # T+1 observations against T outcomes, asserted rather than assumed.
        assert att.shape[0] == written["global_reward"].shape[0] + 1
        assert att.shape[0] == written["ix0_action"].shape[0] + 1


def test_att_horizon_comparison_forms_that_trip_on_float32_storage(tmp_path: Path) -> None:
    """Which float64-vs-float32 comparisons of the ATT horizon fail, measured not assumed.

    Carried from P2.6: the stored array is float32 while a consumer's horizon is float64, so
    "compare without casting and a correct corpus fails ``==``".  That is true of most
    spellings but **not all**, and the exception is the dangerous part to get wrong:

    * ``np.float32 == <python float>`` is **True** -- NumPy 2's NEP 50 treats a Python float
      as a *weak* scalar and casts it DOWN, so this spelling silently does the right thing;
    * every spelling where float64 is *strong* promotes UP and the equality fails:
      ``float(stored) == computed``, ``stored == np.float64(computed)``, and any array
      comparison against a float64 array -- which is what ``np.array_equal`` builds from a
      plain Python list.

    So a consumer is safe by accident on a bare scalar comparison and unsafe the moment the
    value passes through ``float()``, ``np.float64`` or an array.  Measured on numpy 2.5.1;
    if NEP 50 semantics change, this test says so rather than leaving stale guidance in the
    module docstring.
    """
    out_dir = write_dataset_dir(tmp_path, "v11__policy", draws=(1,), format_version="1.1")
    ds = TrajectoryWindowDataset([out_dir], context_length=K, split="train")
    att = ds.episode_records[0].att_per_step
    assert att is not None

    stored = att[-1]                                  # float32, straight out of the corpus
    computed = ATT_SOURCE_FLOAT64[-1]                 # what a consumer computes in float64
    assert isinstance(stored, np.float32) and isinstance(computed, float)

    # Safe by accident: the Python float is weak and gets cast down.
    assert stored == computed

    # Dangerous: every form in which float64 is strong.
    assert float(stored) != computed
    assert stored != np.float64(computed)
    assert not np.array_equal(att, np.asarray(ATT_SOURCE_FLOAT64))

    # Correct in every form: cast the float64 DOWN, never widen the stored array.
    assert stored == np.float32(computed)
    assert np.array_equal(att, np.asarray(ATT_SOURCE_FLOAT64, dtype=np.float32))


# ----------------------------------------------------------------------
# Normalisation statistics (P3.2)
# ----------------------------------------------------------------------


def _expected_mean_std(corpus: Path, ix_index: int, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent route: pool rows 0..T-1 from both ``.npz`` files and use ``math.fsum``.

    The fixture's values make every intermediate exact (integers, and a row count that is a
    power of two), so the comparison is ``==`` rather than a tolerance.
    """
    rows: list[np.ndarray] = []
    for filename in ("ep000000_seed1000_draw1.npz", "ep000001_seed1001_draw2.npz"):
        rows.append(raw_arrays(corpus, filename)[f"ix{ix_index}_state"][:T])
    pooled = np.concatenate(rows).astype(np.float64)
    n = pooled.shape[0]
    mean = np.asarray([math.fsum(pooled[:, d]) / n for d in range(state_dim)])
    var = np.asarray(
        [math.fsum((pooled[:, d] - mean[d]) ** 2) / n for d in range(state_dim)]
    )
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def test_state_statistics_match_an_independent_computation(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    stats = ds.stats
    for i, (ix_id, state_dim, _n_actions) in enumerate(IX_SPECS):
        mean, std = _expected_mean_std(corpus, i, state_dim)
        assert mismatch_rows(stats.state_mean[SCENARIO_ID][ix_id], mean) == 0
        assert mismatch_rows(stats.state_std[SCENARIO_ID][ix_id], std) == 0
        assert stats.row_count[SCENARIO_ID][ix_id] == 2 * T


def test_statistics_are_frozen_and_reused_not_refitted(corpus: Path, tmp_path: Path) -> None:
    """A second dataset over a different subset reuses the given statistics unchanged."""
    train = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    frozen = train.stats

    subset = TrajectoryWindowDataset(
        [corpus], context_length=K, split="train", draw_ids=[2], stats=frozen
    )
    assert subset.stats is frozen
    for ix_id in IX_IDS:
        assert mismatch_rows(
            subset.stats.state_mean[SCENARIO_ID][ix_id], frozen.state_mean[SCENARIO_ID][ix_id]
        ) == 0

    refitted = TrajectoryWindowDataset([corpus], context_length=K, split="train", draw_ids=[2])
    assert mismatch_rows(
        refitted.stats.state_mean[SCENARIO_ID]["ix_zulu"],
        frozen.state_mean[SCENARIO_ID]["ix_zulu"],
    ) > 0, "the two subsets must differ, or 'frozen' proves nothing"


def test_statistics_are_never_fitted_on_a_non_training_split(tmp_path: Path) -> None:
    """The D3 leakage guard, asserted: a non-train split refuses to fit its own statistics.

    This is the one registered ruling whose enforcement had no coverage at all (review
    P3-01): two mutants that fit statistics on held-out data -- eagerly at construction, or
    lazily on first use -- survived the entire suite while the behaviour shipped correct.
    The failure it guards is test-set leakage in its quietest form, so it is asserted on both
    routes into the statistics: the ``.stats`` property and a normalised item.
    """
    held = write_dataset_dir(tmp_path, "held__policy", draws=(1000, 1001))
    ds = TrajectoryWindowDataset([held], context_length=K, split="heldout")

    with pytest.raises(ValueError, match=r"pass stats= fitted on split='train'"):
        _ = ds.stats
    with pytest.raises(ValueError, match=r"pass stats= fitted on split='train'"):
        _ = ds[0]

    # normalize=False is the diagnostic escape hatch and must not need statistics.
    raw = TrajectoryWindowDataset([held], context_length=K, split="heldout", normalize=False)
    assert raw[0]["state"].shape == (K, IX_SPECS[0][1])


def test_heldout_data_is_normalised_with_training_statistics(tmp_path: Path) -> None:
    """The legitimate path stated positively: evaluation reuses train-fitted statistics.

    The provenance travels with the numbers -- the statistics used on held-out draws record
    ``split='train'`` and the *training* draw ids, so "what were these fitted on?" is
    answerable from the object rather than from trust.
    """
    train_dir = write_dataset_dir(tmp_path, "train__policy", draws=(1, 2))
    held = write_dataset_dir(tmp_path, "held__policy", draws=(1000, 1001))

    frozen = TrajectoryWindowDataset([train_dir], context_length=K, split="train").stats
    evaluation = TrajectoryWindowDataset(
        [held], context_length=K, split="heldout", stats=frozen
    )

    assert evaluation.stats is frozen
    assert evaluation.stats.split == "train"
    assert evaluation.stats.draw_ids == (1, 2)
    assert not set(evaluation.stats.draw_ids) & {1000, 1001}
    assert evaluation[0]["state"].shape == (K, IX_SPECS[0][1])


def test_statistics_record_the_split_and_the_exact_draw_ids(corpus: Path) -> None:
    """Statistics fitted on evaluation data are leakage; the provenance must be checkable."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    assert ds.stats.split == "train"
    assert ds.stats.draw_ids == (1, 2)
    assert all(DRAW_SPLITS["train"][0] <= d <= DRAW_SPLITS["train"][1] for d in ds.stats.draw_ids)


def test_normalized_states_use_the_statistics_and_padding_stays_zero(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=True)
    stats = ds.stats
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")

    item = ds[index_of(ds, "ix_zulu", T - 1, "ep000000_seed1000_draw1.npz")]
    mean = stats.state_mean[SCENARIO_ID]["ix_zulu"]
    std = stats.state_std[SCENARIO_ID]["ix_zulu"]
    raw_rows = written["ix0_state"][T - K : T]
    expected = (raw_rows - mean) / np.where(std > 0, std, np.float32(1.0))
    assert mismatch_rows(item["state"].numpy(), expected.astype(np.float32)) == 0

    padded = ds[index_of(ds, "ix_zulu", 0, "ep000000_seed1000_draw1.npz")]
    pad = ~padded["attention_mask"].numpy()
    assert not padded["state"].numpy()[pad].any(), "padding must stay zero, not -mean/std"


def test_constant_feature_gets_a_zero_std_and_is_not_divided_by_it(corpus: Path) -> None:
    """The zero-std guard, reached: a constant feature normalises to ``x - mean``, not NaN.

    Load-bearing on the real corpus, where 191/640 grid4x4 std entries are exactly 0 -- and
    untested until now, because the fixture had no constant column (review P3-10).  Removing
    the guard turns every normalised value of such a feature into ``0/0``.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=True)
    for ix_id, state_dim, _n_actions in IX_SPECS:
        std = ds.stats.state_std[SCENARIO_ID][ix_id]
        mean = ds.stats.state_mean[SCENARIO_ID][ix_id]
        assert std[state_dim - 1] == 0.0, "fixture premise: the last feature is constant"
        assert mean[state_dim - 1] == np.float32(CONSTANT_FEATURE_VALUE)
        assert (std[: state_dim - 1] > 0).all(), "the other features must vary"

        item = ds[index_of(ds, ix_id, T - 1)]
        column = item["state"].numpy()[:, state_dim - 1]
        assert np.isfinite(column).all(), "a zero std must not produce NaN or inf"
        assert np.array_equal(
            column, np.full(K, np.float32(CONSTANT_FEATURE_VALUE) - mean[state_dim - 1])
        )


def test_rtg_summary_is_recorded_for_every_intersection(corpus: Path) -> None:
    """Quantiles are recorded for P4.3 and nothing is applied to the RTG itself."""
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    for ix_id in IX_IDS:
        summary = ds.stats.rtg[SCENARIO_ID][ix_id]
        assert summary.count == 2 * T
        assert summary.min <= summary.mean <= summary.max
        assert [q for q, _ in summary.quantiles] == list(RTG_QUANTILES)
        values = [v for _, v in summary.quantiles]
        assert values == sorted(values)
        assert summary.min <= values[0] and values[-1] <= summary.max

    # Unnormalised RTG still equals the raw suffix sums.
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    rewards = written["ix0_local_reward"]
    expected = np.cumsum(rewards[::-1].astype(np.float64))[::-1].astype(np.float32)
    assert mismatch_rows(ds[index_of(ds, "ix_zulu", T - 1)]["rtg"].numpy().reshape(-1), expected) == 0


def test_rtg_quantiles_match_an_independent_interpolation(corpus: Path) -> None:
    """Quantiles recomputed from the ``.npz`` by an explicit linear interpolation.

    D3 recorded these so P4.3 need not compute them from a second code path; that only helps
    if they are pinned, and until now no test could tell ``method="linear"`` from
    ``method="lower"`` (review P3-11).

    Six of the seven quantile positions land on a dyadic fraction of an integer-valued RTG
    (``q*(n-1)`` for n=16 gives 1.5, 3.75, 7.5, 11.25, 13.5, 14.25), so ``a + (b-a)*frac`` is
    exact and the comparison is ``==``.  ``q=0.99`` gives 14.85, which is not dyadic, so it is
    asserted by bracketing between its two neighbours rather than with a tolerance.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=T, split="train", normalize=False)
    for i, (ix_id, _state_dim, _n_actions) in enumerate(IX_SPECS):
        pooled: list[float] = []
        for filename in ("ep000000_seed1000_draw1.npz", "ep000001_seed1001_draw2.npz"):
            rewards = raw_arrays(corpus, filename)[f"ix{i}_local_reward"]
            pooled.extend(
                float(v) for v in np.cumsum(rewards[::-1].astype(np.float64))[::-1]
            )
        values = sorted(pooled)
        n = len(values)
        assert n == 2 * T

        recorded = dict(ds.stats.rtg[SCENARIO_ID][ix_id].quantiles)
        for q in RTG_QUANTILES:
            position = q * (n - 1)
            low_index = math.floor(position)
            high_index = math.ceil(position)
            frac = position - low_index
            if frac == 0.0 or (frac * 4) == int(frac * 4):      # dyadic: exactly representable
                expected = values[low_index] + (values[high_index] - values[low_index]) * frac
                assert recorded[q] == expected, f"quantile {q} for {ix_id}"
            else:
                assert values[low_index] <= recorded[q] <= values[high_index]


def test_stats_json_round_trip_is_byte_identical(corpus: Path, tmp_path: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    first = tmp_path / "stats.json"
    second = tmp_path / "stats2.json"

    save_stats(ds.stats, first)
    reloaded = load_stats(first)
    save_stats(reloaded, second)
    assert first.read_bytes() == second.read_bytes()

    for ix_id in IX_IDS:
        assert mismatch_rows(
            reloaded.state_mean[SCENARIO_ID][ix_id], ds.stats.state_mean[SCENARIO_ID][ix_id]
        ) == 0
        assert mismatch_rows(
            reloaded.state_std[SCENARIO_ID][ix_id], ds.stats.state_std[SCENARIO_ID][ix_id]
        ) == 0
    assert reloaded.draw_ids == ds.stats.draw_ids
    assert reloaded.split == ds.stats.split


def test_save_stats_barrier_leaves_earlier_data_untouched(corpus: Path, tmp_path: Path) -> None:
    """Filesystem-mutation barrier: validation before any write, and no directory creation."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    path = tmp_path / "stats.json"
    save_stats(ds.stats, path)
    before = path.read_bytes()

    with pytest.raises(TypeError, match="NormalizationStats"):
        save_stats({"not": "stats"}, path)          # type: ignore[arg-type]
    assert path.read_bytes() == before

    missing = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError, match="destination directory does not exist"):
        save_stats(ds.stats, missing / "stats.json")
    assert not missing.exists(), "a failed save must not create directories"


def test_save_stats_never_opens_the_destination_for_writing(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomicity, mechanically: the payload lands in a temp file and arrives by rename.

    A fully non-atomic writer -- ``open(destination, "w")`` or ``Path.write_text`` -- was
    green under the previous barrier test, which only covered type validation and directory
    non-creation (review P3-02).

    **Both** ``builtins.open`` and ``io.open`` are spied on, and that is not belt-and-braces:
    a bare ``open(...)`` call resolves through ``builtins``, while ``Path.write_text`` goes to
    ``io.open`` directly and is invisible to a ``builtins``-only spy -- which is how the first
    version of this test passed under the very mutant it exists to catch.  This module's own
    writer reaches ``io.open`` too, but with an ``mkstemp`` **file descriptor**, never the
    destination path, which is exactly the distinction being asserted.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    path = tmp_path / "stats.json"
    save_stats(ds.stats, path)

    opened_for_writing: list[str] = []
    real_open = builtins.open

    def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            opened_for_writing.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    save_stats(ds.stats, path)

    assert opened_for_writing, "the spy caught nothing at all, so it proves nothing"

    assert str(path) not in opened_for_writing, (
        "the destination was opened for writing directly, so a crash mid-write would "
        f"truncate it; opened: {opened_for_writing}"
    )


def test_failed_rename_leaves_the_previous_file_and_no_debris(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename fails, the previous statistics survive byte-for-byte and no temp remains.

    The filesystem-mutation barrier has shipped broken twice in this project (P1 NB2, P2.0),
    both times because only the happy path was tested.
    """
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    path = tmp_path / "stats.json"
    save_stats(ds.stats, path)
    before = path.read_bytes()

    def exploding_replace(src: Any, dst: Any) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        save_stats(ds.stats, path)

    assert path.read_bytes() == before, "a failed save must leave the previous file intact"
    assert list(tmp_path.glob(".stats-*")) == [], "the temp file must be cleaned up"


# ----------------------------------------------------------------------
# Heterogeneous intersections
# ----------------------------------------------------------------------


def test_groups_separate_the_two_intersection_shapes(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    assert set(ds.groups) == {(3, 2), (4, 3)}
    for (state_dim, n_actions), indices in ds.groups.items():
        assert len(indices) == 2 * T
        for i in indices:
            item = ds[i]
            assert item["state"].shape == (K, state_dim)
            assert item["avail_mask"].shape == (K, n_actions)
    assert sorted(i for group in ds.groups.values() for i in group) == list(range(len(ds)))


def test_collate_windows_names_both_shapes_when_a_batch_straddles_groups(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    straddling = [ds[ds.groups[(3, 2)][0]], ds[ds.groups[(4, 3)][0]]]
    with pytest.raises(ValueError, match=r"\(3, 2\).*\(4, 3\)"):
        collate_windows(straddling)


def test_default_collate_also_refuses_a_straddling_batch(corpus: Path) -> None:
    """The docstring calls this a mechanical guarantee, so it is asserted, not assumed."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    straddling = [ds[ds.groups[(3, 2)][0]], ds[ds.groups[(4, 3)][0]]]
    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        default_collate(straddling)


def test_collate_windows_stacks_a_homogeneous_batch(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    indices = ds.groups[(3, 2)][:4]
    batch = collate_windows([ds[i] for i in indices])
    assert batch["state"].shape == (4, K, 3)
    assert batch["avail_mask"].shape == (4, K, 2)
    assert batch["rtg"].shape == (4, K, 1)
    for position, index in enumerate(indices):
        assert torch.equal(batch["state"][position], ds[index]["state"])


def test_item_meta_traces_back_to_a_row_on_disk(corpus: Path) -> None:
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    meta = ds.item_meta(index_of(ds, "ix_alpha", 5, "ep000001_seed1001_draw2.npz"))
    assert meta.ix_id == "ix_alpha"
    assert meta.ix_index == 1
    assert meta.t == 5
    assert meta.flow_draw == 2
    assert meta.scenario_id == SCENARIO_ID
    assert Path(meta.dataset_dir).name == corpus.name
    assert (corpus / meta.episode_file).exists()
