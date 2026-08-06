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
  *invisible at that row*: a miniature of the corpus-wide measurement in ``docs/plans/p3.md``
  section 4 (blind on 0.83% of hz1x1 rows, 25.31% of cologne3, 37.63% of grid4x4).  It is the
  reason the RTG assertions compare whole arrays and never a sampled row.
* all values are exactly representable in float32, so every comparison is ``==`` and never a
  tolerance.

DISCRIMINATING POWER.  Each alignment test reports the distance between the correct answer
and the deliberately wrong one via :func:`mismatch_rows`, and asserts that distance exactly
rather than asserting "it passed".  A check whose wrong answer is close to its right answer
certifies little however green it looks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from offline.dataset import (
    DRAW_SPLITS,
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


def _state_rows(ix_index: int, state_dim: int, ep_index: int) -> np.ndarray:
    """``t + 10*d + 100*ix + 1000*ep`` -- integer, so pooled mean/std stay exact.

    Every row differs from its neighbour (a shifted window is visible) and every
    intersection differs from the other (a positional mix-up is visible).
    """
    rows = np.empty((T + 1, state_dim), dtype=np.float32)
    for t in range(T + 1):
        for d in range(state_dim):
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


def test_padded_positions_are_zero(corpus: Path) -> None:
    """Every padded position is zero / False, so a masked loss reads nothing meaningful."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train", normalize=False)
    written = raw_arrays(corpus, "ep000000_seed1000_draw1.npz")
    item = ds[index_of(ds, "ix_zulu", 0, "ep000000_seed1000_draw1.npz")]
    pad = ~item["attention_mask"].numpy()
    assert int(pad.sum()) == K - 1
    assert not item["state"].numpy()[pad].any()
    assert not item["rtg"].numpy().reshape(-1)[pad].any()
    assert not item["action"].numpy()[pad].any()
    assert not item["timestep"].numpy()[pad].any()
    assert not item["avail_mask"].numpy()[pad].any()
    # ...and the single real row is the episode's first row, not a shifted one.
    assert np.array_equal(item["state"].numpy()[-1], written["ix0_state"][0])


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
    with pytest.raises(ValueError, match="0"):
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


def test_mixed_format_versions_are_rejected(tmp_path: Path) -> None:
    """A v1.0 and a v1.1 corpus must never be silently mixed (BRIEF_08 section 5)."""
    old = write_dataset_dir(tmp_path, "old__policy", draws=(1,), format_version="1.0")
    new = write_dataset_dir(tmp_path, "new__policy", draws=(2,), format_version="1.1")
    with pytest.raises(ValueError, match="format version"):
        TrajectoryWindowDataset([old, new], context_length=K, split="train")


def test_unsupported_format_version_is_rejected(tmp_path: Path) -> None:
    out_dir = write_dataset_dir(tmp_path, "future__policy", draws=(1,), format_version="9.9")
    with pytest.raises(ValueError, match="9.9"):
        TrajectoryWindowDataset([out_dir], context_length=K, split="train")


def test_att_per_step_is_none_for_v10_episodes(corpus: Path) -> None:
    """v1.0 carries no ATT array; the field exists and is None rather than absent."""
    ds = TrajectoryWindowDataset([corpus], context_length=K, split="train")
    assert [record.att_per_step for record in ds.episode_records] == [None, None]
    assert {record.format_version for record in ds.episode_records} == {"1.0"}


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
    with pytest.raises(FileNotFoundError, match="no_such_dir"):
        save_stats(ds.stats, missing / "stats.json")
    assert not missing.exists(), "a failed save must not create directories"


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
