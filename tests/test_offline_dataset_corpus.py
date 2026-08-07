"""Real-corpus tests for ``offline.dataset``.

Separate from ``tests/test_offline_dataset.py`` so that "how many corpus tests actually ran"
is answerable from the test report.  ``datasets*/`` is gitignored and lives outside this
worktree, so these tests locate it through ``RLTRAFFIC_CORPUS`` and **skip with a reason
naming that variable** when it is absent.  A green suite in which every corpus test skipped
is the failure mode, not the safeguard.

Each alignment test reports its discriminating power -- the distance between the correct
answer and the deliberately wrong one -- because a check that passes with a narrow margin
certifies almost nothing while looking rigorous.  Run with ``-s`` to see the reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from offline.dataset import DRAW_SPLITS, MANIFEST_NAME, TrajectoryWindowDataset

REPO_ROOT = Path(__file__).resolve().parents[1]

# Scenarios where sum_j local_reward[t] == -sum lane_waiting[t+1] is EXACT.  cologne3 is
# excluded by construction, not by a skip: its three controlled intersections do not cover
# every lane of the network, so the two sums are over different lane sets and the identity
# is structurally false there (BRIEF_09 section 2, measured 0/46 exact, max residual 549).
LANE_IDENTITY_SCENARIOS = ("cityflow1x1", "cityflow_grid4x4")

EPISODES_PER_DIR = 2        # the sampling protocol used for every population claim below


@pytest.fixture(scope="module")
def corpus_root() -> Path:
    """``RLTRAFFIC_CORPUS``, else ``<repo>/datasets``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets"
    if not candidate.is_dir():
        pytest.skip(
            f"offline corpus not found at {candidate}: set RLTRAFFIC_CORPUS to a "
            "collected datasets/ directory to run the corpus-backed tests"
        )
    return candidate


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected datasets_v11/ directory to run the v1.1 corpus tests"
        )
    return candidate


@pytest.fixture(scope="module")
def dataset_dirs(corpus_root: Path) -> dict[str, list[Path]]:
    """Collection directories grouped by ``scenario_id`` read from their manifests."""
    grouped: dict[str, list[Path]] = {}
    for manifest_path in sorted(corpus_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario = manifest["run_metadata"]["scenario_id"]
        grouped.setdefault(scenario, []).append(manifest_path.parent)
    if not grouped:
        pytest.skip(
            f"no collection directories under {corpus_root}: RLTRAFFIC_CORPUS must point "
            "at a datasets/ tree containing <scenario>__<tier>/manifest.json"
        )
    return grouped


def _episode_files(dataset_dir: Path, limit: int) -> list[str]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return [entry["filename"] for entry in manifest["episodes"][:limit]]


def _one_dir_per_scenario(dataset_dirs: dict[str, list[Path]]) -> dict[str, Path]:
    return {scenario: dirs[0] for scenario, dirs in sorted(dataset_dirs.items())}


# ----------------------------------------------------------------------
# RTG against the corpus
# ----------------------------------------------------------------------


def test_rtg_matches_reversed_cumsum_on_real_episodes(dataset_dirs: dict[str, list[Path]]) -> None:
    """The loader's RTG equals ``np.cumsum`` over the reversed reward array -- WHOLE array.

    ``context_length`` is set to the episode length so a single item carries all ``T`` rows of
    a stream; the earlier version used ``context_length=8`` and compared **8 of 360 rows**,
    where grid4x4's weakest stream discriminated on 2 rows at a maximum gap of 1.0, and
    16.6% of its streams were entirely blind on that window (review P3-04).  Whole-array
    comparison removes the dependence on which rows happen to land in the last window.

    The margin is reported per scenario rather than assumed, and asserted per stream.
    """
    report: list[str] = []
    checked = 0
    for scenario, dataset_dir in _one_dir_per_scenario(dataset_dirs).items():
        manifest = json.loads((dataset_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        length = int(manifest["episodes"][0]["episode_length"])
        ds = TrajectoryWindowDataset(
            [dataset_dir], context_length=length, split="train", draw_ids=[1], normalize=False
        )

        last_item: dict[tuple[str, str], int] = {}
        for i in range(len(ds)):
            meta = ds.item_meta(i)
            if meta.t == length - 1:
                last_item[(meta.episode_file, meta.ix_id)] = i

        rows = differing = 0
        weakest_rows: int | None = None
        weakest_gap: float | None = None
        for (episode_file, ix_id), index in sorted(last_item.items()):
            meta = ds.item_meta(index)
            with np.load(Path(meta.dataset_dir) / episode_file) as data:
                rewards = data[f"ix{meta.ix_index}_local_reward"]
                ix_ids = [str(v) for v in data["ix_ids"].tolist()]
            assert ix_ids[meta.ix_index] == ix_id, "ix_index must address the id it claims"

            expected = np.cumsum(rewards[::-1].astype(np.float64))[::-1].astype(np.float32)
            actual = ds[index]["rtg"].numpy().reshape(-1)
            assert actual.size == length, "the window must cover the whole stream"
            assert np.array_equal(actual, expected), (
                f"{scenario}: RTG mismatch for {ix_id} in {episode_file}"
            )

            # Discriminating power: the off-by-one differs from this exactly at r_t != 0.
            stream_differing = int(np.count_nonzero(rewards))
            stream_gap = float(np.abs(rewards).max())
            assert stream_differing > 0, (
                f"{scenario}: stream {ix_id} in {episode_file} has an all-zero reward stream, "
                "so an off-by-one RTG would be undetectable on it"
            )
            rows += length
            differing += stream_differing
            weakest_rows = stream_differing if weakest_rows is None else min(weakest_rows, stream_differing)
            weakest_gap = stream_gap if weakest_gap is None else min(weakest_gap, stream_gap)
            checked += 1

        report.append(
            f"[rtg] {scenario}: {len(last_item)} streams x {length} rows compared exactly; "
            f"under the off-by-one {100 * differing / rows:.1f}% of rows differ, weakest "
            f"stream {weakest_rows}/{length} rows at max gap {weakest_gap}"
        )
    assert checked > 0, "no corpus streams were checked"
    print("\n" + "\n".join(report))


def test_offbyone_rtg_margin_is_measured_on_the_corpus(dataset_dirs: dict[str, list[Path]]) -> None:
    """Report, per scenario, how far the off-by-one RTG sits from the correct one.

    ``rtg_correct[t] - rtg_offbyone[t] == r_t`` exactly, so the mutation is invisible at
    every row where ``r_t == 0``.  The per-row blindness is large on the multi-intersection
    scenarios; what the tests rely on is that no *stream* is blind.
    """
    report: list[str] = []
    for scenario, dirs in sorted(dataset_dirs.items()):
        rows = blind_rows = streams = blind_streams = 0
        worst_stream_rows = None
        for dataset_dir in dirs:
            for filename in _episode_files(dataset_dir, EPISODES_PER_DIR):
                with np.load(dataset_dir / filename) as data:
                    for i in range(len(data["ix_ids"])):
                        rewards = data[f"ix{i}_local_reward"]
                        differing = int(np.count_nonzero(rewards))
                        rows += rewards.size
                        blind_rows += rewards.size - differing
                        streams += 1
                        blind_streams += int(differing == 0)
                        if worst_stream_rows is None or differing < worst_stream_rows:
                            worst_stream_rows = differing
        assert blind_streams == 0, (
            f"{scenario}: {blind_streams} of {streams} streams have an all-zero reward "
            "stream, where an off-by-one RTG is undetectable"
        )
        report.append(
            f"[off-by-one] {scenario}: {streams} streams, {rows} rows, "
            f"blind rows {100 * blind_rows / rows:.2f}%, blind streams {blind_streams}, "
            f"weakest stream discriminates on {worst_stream_rows}/{rows // streams} rows"
        )
    print("\n" + "\n".join(report))


# ----------------------------------------------------------------------
# The exact local identity, and the alignment margin behind it
# ----------------------------------------------------------------------


def test_local_reward_lane_identity_and_its_alignment_margin(
    dataset_dirs: dict[str, list[Path]]
) -> None:
    """``sum_j ix{j}_local_reward[t] == -sum lane_waiting[t+1]``, bit-exact, and the
    misaligned pairing reported beside it as the discriminating margin.

    This ties the reward stream RTG is built from to raw lane counts by a route that shares
    no code with the loader.  cologne3 is excluded structurally (see the module constant).
    """
    report: list[str] = []
    for scenario in LANE_IDENTITY_SCENARIOS:
        if scenario not in dataset_dirs:
            continue
        episodes = exact = 0
        misaligned_rows = total_rows = 0
        for dataset_dir in dataset_dirs[scenario]:
            for filename in _episode_files(dataset_dir, EPISODES_PER_DIR):
                with np.load(dataset_dir / filename) as data:
                    n_ix = len(data["ix_ids"])
                    local_sum = sum(
                        data[f"ix{i}_local_reward"].astype(np.float32) for i in range(n_ix)
                    )
                    waiting = data["lane_waiting_vehicle_count"].astype(np.float32)
                aligned = -waiting[1:].sum(axis=1)
                misaligned = -waiting[:-1].sum(axis=1)
                episodes += 1
                exact += int(np.array_equal(local_sum, aligned))
                misaligned_rows += int(np.count_nonzero(local_sum != misaligned))
                total_rows += local_sum.size
        assert episodes > 0
        assert exact == episodes, (
            f"{scenario}: the local lane identity held on only {exact}/{episodes} episodes"
        )
        report.append(
            f"[lane identity] {scenario}: exact on {exact}/{episodes} episodes; "
            f"the misaligned pairing differs on {100 * misaligned_rows / total_rows:.2f}% "
            f"of {total_rows} rows"
        )
    assert report, "no scenario supported the lane identity check"
    print("\n" + "\n".join(report))


# ----------------------------------------------------------------------
# Corpus population claims
# ----------------------------------------------------------------------


def test_every_stored_avail_mask_is_all_true(dataset_dirs: dict[str, list[Path]]) -> None:
    """Falsifiable population claim: masking never binds anywhere in this corpus.

    Asserting "the mask never excludes the taken action" on this corpus would be a
    tautology -- the corpus was collected under ``control_mode: acyclic``, whose
    ``available_actions()`` returns every green phase.  The claim asserted instead is the
    one that can fail: every stored mask is entirely True.  It fails the day a corpus is
    collected under a cyclic mode, which is when the consumer needs to know.
    """
    streams = all_true = episodes = 0
    for _scenario, dirs in sorted(dataset_dirs.items()):
        for dataset_dir in dirs:
            for filename in _episode_files(dataset_dir, EPISODES_PER_DIR):
                with np.load(dataset_dir / filename) as data:
                    episodes += 1
                    for i in range(len(data["ix_ids"])):
                        mask = data[f"ix{i}_avail_mask"]
                        actions = data[f"ix{i}_action"]
                        streams += 1
                        all_true += int(bool(mask.all()))
                        taken = mask[np.arange(actions.size), actions]
                        assert bool(taken.all()), (
                            f"{dataset_dir.name}/{filename}: the mask excludes the action "
                            "actually taken -- that is a corpus defect, not a loader bug"
                        )
    assert streams > 0
    assert all_true == streams, (
        f"{all_true}/{streams} mask streams are all-True; a binding mask has appeared in "
        "the corpus, so the masking claim constraint in docs/plans/p3.md section 6 is stale"
    )
    print(f"\n[avail_mask] {episodes} episodes, {streams} streams, all-True: {all_true}/{streams}")


def test_all_corpus_draws_lie_in_the_training_pool(dataset_dirs: dict[str, list[Path]]) -> None:
    """Today's corpus is draws 1-200, so the held-out rule is satisfied trivially -- state it."""
    low, high = DRAW_SPLITS["train"]
    draws: set[int] = set()
    for dirs in dataset_dirs.values():
        for dataset_dir in dirs:
            manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
            draws.update(entry["flow_draw"] for entry in manifest["episodes"])
    assert draws, "no draws found in the corpus manifests"
    held_out = sorted(d for d in draws if DRAW_SPLITS["heldout"][0] <= d <= DRAW_SPLITS["heldout"][1])
    assert held_out == [], f"held-out draws present in the training corpus: {held_out}"
    assert all(low <= d <= high for d in draws), f"draws outside the training pool: {sorted(draws)}"
    print(f"\n[draws] {len(draws)} distinct draws, range {min(draws)}-{max(draws)}")


# ----------------------------------------------------------------------
# End to end on a real collection directory
# ----------------------------------------------------------------------


def test_att_per_step_is_populated_on_the_real_v11_corpus(corpus_v11_root: Path) -> None:
    """The POSITIVE v1.1 branch on real data -- the merge gate this task waited on.

    Exercised through the loader, so what is verified is the ATT gate's *populated* path
    end to end: the array survives ``load_episode``, reaches ``EpisodeRecord``, and carries
    the T+1 alignment an observation must have.
    """
    dataset_dir = sorted(p.parent for p in corpus_v11_root.glob("*/manifest.json"))[0]
    manifest = json.loads((dataset_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format_version"] == "1.1", f"{dataset_dir} is not a v1.1 corpus"

    ds = TrajectoryWindowDataset(
        [dataset_dir], context_length=8, split="train", draw_ids=[1], normalize=False
    )
    checked = 0
    for record in ds.episode_records:
        att = record.att_per_step
        assert record.format_version == "1.1"
        assert att is not None, f"{record.episode_file}: v1.1 episode with no att_per_step"
        assert att.dtype == np.float32
        assert att.shape == (record.episode_length + 1,), "att_per_step is an observation"

        with np.load(Path(record.dataset_dir) / record.episode_file) as data:
            assert np.array_equal(att, data["att_per_step"])
            assert att.shape[0] == data["global_reward"].shape[0] + 1
        # The horizon element is the registered primary metric (A1).  Compared as float32 --
        # see the comparison-form test in tests/test_offline_dataset.py for why.
        horizon = att[-1]
        assert np.isfinite(horizon) and horizon > 0
        assert horizon == att[record.episode_length], "the horizon is the last observation row"
        assert horizon != att[0], "ATT must evolve over an episode, or the field is inert"
        checked += 1
    assert checked > 0, "no v1.1 episodes were checked"
    print(f"\n[att v1.1] {dataset_dir.name}: {checked} episodes, horizon {att[-1]:.5f}")


def test_a_v10_and_a_v11_corpus_cannot_be_mixed(
    dataset_dirs: dict[str, list[Path]], corpus_v11_root: Path
) -> None:
    """The homogeneity guard against **both real corpora**, now that both versions load.

    This is the test review finding P3-03 said could not exist yet: until P2.6 merged, a
    mixed pair was refused for being *unsupported* rather than for being *mixed*, so the
    guard could be deleted with the suite still green.  Both directories below load happily
    on their own, so only the mixing can raise.
    """
    v10_dir = _one_dir_per_scenario(dataset_dirs)["cityflow1x1"]
    v11_dir = sorted(p.parent for p in corpus_v11_root.glob("*/manifest.json"))[0]

    for directory, expected in ((v10_dir, "1.0"), (v11_dir, "1.1")):
        alone = TrajectoryWindowDataset(
            [directory], context_length=4, split="train", draw_ids=[1], normalize=False
        )
        assert {r.format_version for r in alone.episode_records} == {expected}

    with pytest.raises(ValueError, match=r"span format versions \['1\.0', '1\.1'\]"):
        TrajectoryWindowDataset(
            [v10_dir, v11_dir], context_length=4, split="train", draw_ids=[1], normalize=False
        )


def test_items_match_the_npz_on_a_real_collection_directory(
    dataset_dirs: dict[str, list[Path]]
) -> None:
    """A window taken from a real episode carries that episode's rows, unshifted."""
    context = 5
    for scenario, dataset_dir in _one_dir_per_scenario(dataset_dirs).items():
        ds = TrajectoryWindowDataset(
            [dataset_dir], context_length=context, split="train", draw_ids=[1], normalize=False
        )
        assert len(ds) > 0, f"{scenario}: no items built from {dataset_dir}"

        index = len(ds) // 2
        meta = ds.item_meta(index)
        item = ds[index]
        with np.load(Path(meta.dataset_dir) / meta.episode_file) as data:
            states = data[f"ix{meta.ix_index}_state"]
            actions = data[f"ix{meta.ix_index}_action"]
            masks = data[f"ix{meta.ix_index}_avail_mask"]
            length = int(data["episode_length"])

        assert 0 <= meta.t <= length - 1, "no item may read the post-final observation row"
        real = item["attention_mask"].numpy()
        lo = meta.t + 1 - int(real.sum())
        assert np.array_equal(item["state"].numpy()[real], states[lo : meta.t + 1])
        assert np.array_equal(item["action"].numpy()[real], actions[lo : meta.t + 1])
        assert np.array_equal(item["avail_mask"].numpy()[real], masks[lo : meta.t + 1])
        assert np.array_equal(item["timestep"].numpy()[real], np.arange(lo, meta.t + 1))
