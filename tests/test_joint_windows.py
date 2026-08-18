"""Joint decision instants: the grouping, and the proof it pairs the right rows.

The grouping is the pairing key between **corpus rows** and **graph nodes**.  If it is wrong, a
spatial mixing layer masks node ``i``'s tokens with node ``j``'s neighbours and trains perfectly
happily -- ``PROJECT_PLAN`` section 7's 2026-08-16 finding, one level up from lane ids.

Every structural claim here is recomputed by a route that does not call the module under test:
the grouping is rebuilt from ``TrajectoryWindowDataset.item_meta`` inside the test and compared
with ``==`` on exact integer arrays.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from offline.dataset import TrajectoryWindowDataset
from tests.test_offline_dataset import write_dataset_dir
from offline.joint_windows import (
    JOINT_INDEX_FORMAT_VERSION,
    build_joint_index,
    stack_joint,
)

GRID_NODE_IDS = (
    "A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3",
    "C0", "C1", "C2", "C3", "D0", "D1", "D2", "D3",
)
DRAWS = (1, 2, 3)
EPISODE_LENGTH = 360


def _corpus_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else Path(__file__).resolve().parents[1] / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected corpus root to run the joint-window tests"
        )
    return candidate


@pytest.fixture(scope="module")
def grid_dataset() -> TrajectoryWindowDataset:
    root = _corpus_root()
    directory = root / "cf_grid4x4__mappo1000__seed101"
    if not (directory / "manifest.json").is_file():
        pytest.skip(f"{directory} is not a collected dataset directory")
    return TrajectoryWindowDataset(
        [directory], context_length=20, split="train", draw_ids=list(DRAWS), normalize=True
    )


def _grouping_from_item_meta(dataset: TrajectoryWindowDataset) -> dict:
    """INDEPENDENT ROUTE: rebuild the grouping from the loader's own provenance answer."""
    groups: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for item in range(len(dataset)):
        meta = dataset.item_meta(item)
        groups[(meta.episode_index, meta.t)][meta.ix_id] = item
    return dict(groups)


# ----------------------------------------------------------------------
# Shape and completeness
# ----------------------------------------------------------------------


def test_the_index_has_one_row_per_decision_instant(grid_dataset):
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    assert index.n_nodes == 16
    assert index.n_windows == len(DRAWS) * EPISODE_LENGTH
    assert index.member_index.shape == (len(DRAWS) * EPISODE_LENGTH, 16)
    # The joint form holds every per-intersection window exactly once.
    assert index.member_index.size == len(grid_dataset)
    assert sorted(index.member_index.reshape(-1).tolist()) == list(range(len(grid_dataset)))


def test_the_index_carries_the_group_shape(grid_dataset):
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    assert index.state_dim == 40
    assert index.n_actions == 8
    assert index.member_index.dtype == np.int64
    assert index.episode_index.dtype == np.int64
    assert index.t.dtype == np.int64


def test_every_instant_spans_the_full_decision_range(grid_dataset):
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    for episode in np.unique(index.episode_index):
        steps = np.sort(index.t[index.episode_index == episode])
        assert steps.tolist() == list(range(EPISODE_LENGTH))


# ----------------------------------------------------------------------
# DOUBLE COMPUTE: the grouping, rebuilt from item_meta inside the test
# ----------------------------------------------------------------------


def test_the_grouping_reproduces_from_item_meta(grid_dataset):
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    expected = _grouping_from_item_meta(grid_dataset)

    assert len(expected) == index.n_windows
    rebuilt = np.array(
        [
            [expected[(int(ep), int(t))][node] for node in GRID_NODE_IDS]
            for ep, t in zip(index.episode_index, index.t)
        ],
        dtype=np.int64,
    )
    assert np.array_equal(index.member_index, rebuilt)


def test_every_member_is_the_node_it_claims_to_be(grid_dataset):
    """The alignment proof: column ``n`` of every row is ``node_ids[n]``, checked from provenance."""
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    for row in range(0, index.n_windows, 97):
        for column, node in enumerate(GRID_NODE_IDS):
            meta = grid_dataset.item_meta(int(index.member_index[row, column]))
            assert meta.ix_id == node
            assert meta.episode_index == int(index.episode_index[row])
            assert meta.t == int(index.t[row])


def test_a_permuted_node_order_permutes_the_columns(grid_dataset):
    """Order sensitivity: the same instants, different columns -- and the ids follow."""
    straight = build_joint_index(grid_dataset, GRID_NODE_IDS)
    swapped_ids = ("B1", *GRID_NODE_IDS[1:5], "A0", *GRID_NODE_IDS[6:])
    assert set(swapped_ids) == set(GRID_NODE_IDS) and swapped_ids != GRID_NODE_IDS
    permuted = build_joint_index(grid_dataset, swapped_ids)

    assert not np.array_equal(straight.member_index, permuted.member_index)
    assert np.array_equal(straight.member_index[:, 0], permuted.member_index[:, 5])
    assert np.array_equal(straight.member_index[:, 5], permuted.member_index[:, 0])
    assert permuted.node_ids == swapped_ids


# ----------------------------------------------------------------------
# Refusals: an incomplete instant is refused, never padded and never dropped
# ----------------------------------------------------------------------


def test_a_node_absent_from_the_corpus_is_refused(grid_dataset):
    with pytest.raises(ValueError, match="never present in the corpus"):
        build_joint_index(grid_dataset, (*GRID_NODE_IDS, "NOT_AN_INTERSECTION"))


def test_a_node_present_in_the_corpus_but_missing_from_the_order_is_refused(grid_dataset):
    with pytest.raises(ValueError, match="carries intersections the node order omits"):
        build_joint_index(grid_dataset, GRID_NODE_IDS[:-1])


def test_duplicate_node_ids_are_refused(grid_dataset):
    with pytest.raises(ValueError, match="duplicate node ids"):
        build_joint_index(grid_dataset, (*GRID_NODE_IDS[:-1], "A0"))


def test_an_empty_node_order_is_refused(grid_dataset):
    with pytest.raises(ValueError, match="node_ids is empty"):
        build_joint_index(grid_dataset, ())


# ----------------------------------------------------------------------
# stack_joint: the gathered batch IS the loader's own window
# ----------------------------------------------------------------------


def test_the_flat_rows_a_member_points_at_are_the_dataset_items(grid_dataset):
    """No re-slicing: what the model reads must be byte-identical to ``dataset[i]``."""
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    stacked = stack_joint(grid_dataset, index)
    member = stacked["member_index"]

    assert member.shape == (index.n_windows, 16)
    for row in range(0, index.n_windows, 211):
        for column in (0, 7, 15):
            flat_row = int(member[row, column])
            item = grid_dataset[int(index.member_index[row, column])]
            for key, value in item.items():
                assert torch.equal(stacked[key][flat_row], value), f"{key} at ({row}, {column})"


def test_a_gathered_batch_has_the_joint_shape(grid_dataset):
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    stacked = stack_joint(grid_dataset, index)
    rows = torch.tensor([0, 5, 900], dtype=torch.int64)
    members = stacked["member_index"][rows]

    assert stacked["state"][members].shape == (3, 16, 20, 40)
    assert stacked["rtg"][members].shape == (3, 16, 20, 1)
    assert stacked["action"][members].shape == (3, 16, 20)
    assert stacked["avail_mask"][members].shape == (3, 16, 20, 8)
    assert stacked["attention_mask"][members].shape == (3, 16, 20)


def test_the_nodes_of_one_instant_share_their_timestep(grid_dataset):
    """All 16 act simultaneously; a joint window that mixed instants would break the mixing layer."""
    index = build_joint_index(grid_dataset, GRID_NODE_IDS)
    stacked = stack_joint(grid_dataset, index)
    for row in (0, 19, 359, 700):
        members = stacked["member_index"][row]
        timesteps = stacked["timestep"][members]
        assert torch.equal(timesteps, timesteps[0].expand_as(timesteps))


def test_the_index_summary_records_its_format_and_shape(grid_dataset):
    payload = build_joint_index(grid_dataset, GRID_NODE_IDS).to_json_obj()
    assert payload["format_version"] == JOINT_INDEX_FORMAT_VERSION
    assert payload["n_nodes"] == 16
    assert payload["n_windows"] == len(DRAWS) * EPISODE_LENGTH
    assert payload["node_ids"] == list(GRID_NODE_IDS)
    assert payload["state_dim"] == 40
    assert payload["n_actions"] == 8


# ----------------------------------------------------------------------
# m5 -- the two shipped refusals in this module had no test.  Both are reachable
# and both are exercised live here, so "untested" and "unreachable" are no longer
# conflated (BRIEF_26 section 4 asks that they be told apart).
# ----------------------------------------------------------------------


def _single_group_corpus(root: Path, *, drop_second_in_episode: int | None) -> Path:
    """A corpus whose two intersections share one ``(state_dim, n_actions)`` shape.

    Built by narrowing the shared fixture's second intersection to the first's shape, so this
    test does not become a second reader of the on-disk format -- it reuses
    ``tests/test_offline_dataset._episode_arrays`` and edits two array widths.

    ``drop_second_in_episode`` removes the second intersection from ONE episode, which is what
    makes a decision instant incomplete while leaving the dataset single-group.
    """
    import numpy as np

    from tests.test_offline_dataset import T, _episode_arrays

    out = root / "single_group__policy"
    out.mkdir(parents=True)
    episodes = []
    for ep_index, draw in enumerate((1, 2)):
        arrays = dict(
            _episode_arrays(ep_index=ep_index, draw=draw, format_version="1.1", nan_rewards=False)
        )
        # narrow intersection 1 to intersection 0's shape -> a single (state_dim, n_actions) group
        arrays["ix1_state"] = np.ascontiguousarray(arrays["ix1_state"][:, :3]).astype(np.float32)
        arrays["ix1_avail_mask"] = np.ascontiguousarray(
            arrays["ix1_avail_mask"][:, :2]
        ).astype(np.bool_)
        arrays["ix1_action"] = np.clip(arrays["ix1_action"], 0, 1).astype(np.int64)
        arrays["ix1_current_phase"] = np.clip(arrays["ix1_current_phase"], 0, 1).astype(np.int64)
        if ep_index == drop_second_in_episode:
            for key in [k for k in arrays if k.startswith("ix1_")]:
                del arrays[key]
            arrays["ix_ids"] = np.asarray([str(arrays["ix_ids"][0])], dtype=np.str_)
        filename = f"ep{ep_index:06d}_seed{1000+ep_index}_draw{draw}.npz"
        with open(out / filename, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        episodes.append(
            {
                "filename": filename,
                "episode_length": T,
                "total_global_reward": float(arrays["global_reward"].sum()),
                "engine_seed": 1000 + ep_index,
                "flow_draw": draw,
                "episode_sha256": f"{ep_index:064d}",
            }
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": "1.1",
                "lane_count": 3,
                "lane_ids_sha256": "0" * 64,
                "git_hash": "0" * 40,
                "run_metadata": {"scenario_id": "fixture_2ix"},
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    return out


def test_the_incompleteness_refusal_is_REACHABLE_on_the_v1_1_format(tmp_path):
    """``joint_windows.py``'s incomplete-instant branch: reachable, and now exercised.

    BRIEF_26 section 4 asks whether this is reachable with the v1.1 on-disk format, because
    *untested-live* and *documented-unreachable* are different findings.  **It is reachable:**
    ``ix_ids`` is stored per episode, so two episodes in one corpus may carry different
    intersection sets at the same shape -- which is a single-group dataset with an instant that
    cannot be completed.
    """
    directory = _single_group_corpus(tmp_path, drop_second_in_episode=1)
    dataset = TrajectoryWindowDataset([directory], context_length=4, split="train")
    assert len(dataset.groups) == 1, f"fixture must be single-group, got {sorted(dataset.groups)}"

    with pytest.raises(ValueError, match="do not carry every intersection"):
        build_joint_index(dataset, ("ix_zulu", "ix_alpha"))


def test_a_complete_single_group_corpus_of_the_same_shape_is_accepted(tmp_path):
    """The matched half: without the dropped intersection the same fixture builds cleanly."""
    directory = _single_group_corpus(tmp_path, drop_second_in_episode=None)
    dataset = TrajectoryWindowDataset([directory], context_length=4, split="train")
    index = build_joint_index(dataset, ("ix_zulu", "ix_alpha"))
    assert index.n_nodes == 2
    assert index.member_index.shape == (2 * 8, 2)


def test_stack_joint_refuses_members_absent_from_the_stacked_group(tmp_path):
    """``joint_windows.py``'s second refusal: an index and tensors from different groups.

    Reachable on the two-group fixture: an index that claims group ``(3, 2)`` while carrying a
    member from group ``(4, 3)`` gathers a ``-1`` from the inverse table.
    """
    from offline.joint_windows import JointWindowIndex

    directory = write_dataset_dir(tmp_path, "twogroup__policy")
    dataset = TrajectoryWindowDataset([directory], context_length=4, split="train")
    assert sorted(dataset.groups) == [(3, 2), (4, 3)]

    small = dataset.groups[(3, 2)]
    other = dataset.groups[(4, 3)]
    poisoned = np.array([[small[0]], [other[0]]], dtype=np.int64)
    index = JointWindowIndex(
        node_ids=("ix_zulu",),
        member_index=poisoned,
        episode_index=np.array([0, 0], dtype=np.int64),
        t=np.array([0, 1], dtype=np.int64),
        state_dim=3,
        n_actions=2,
    )
    with pytest.raises(ValueError, match="absent from the stacked group"):
        stack_joint(dataset, index)
