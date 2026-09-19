"""P7.3b section 3.3: the anchor's corpus, its naive in-domain prompt, its five seeds.

Written against ``BRIEF_38`` section 3.3 / section 4, Amendment A (A3 ``raise_to`` null, A5 the
route test), and ``docs/plans/p7.3b.md`` section 3.3.

**Nothing scientific is chosen here or in the module under test.** A18(a) fixes the corpus, the
recipe, the step count, the seeds and the prompt rule; these tests assert that what runs is what
was registered, and that a prompt cannot come from anywhere but the anchor's own training set.

The fixture is a MINIATURE of the real corpus, not a copy of it: one intersection, 25-wide state,
8 actions, integer reward streams, 200 episodes over draws 201-400 split across two directories.
Integer rewards matter -- they make the float32 corpus and a float64 ``np.cumsum`` agree bit for
bit, so the independent recomputation below can assert ``==`` rather than a tolerance.
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

from offline import anchor_training as at

MAIN_TREE = Path("/home/filip/rltraffic")
REAL_CORPUS_DIRS = (
    MAIN_TREE / "datasets_sumo_v11" / "hz1x1_sumo_maxpressure",
    MAIN_TREE / "datasets_sumo_v11" / "hz1x1_sumo_maxpressure_301_400",
)
REAL_OUTPUT_ROOT = MAIN_TREE / "output"

#: The fixture's episode length.  Short, because 200 episodes are written per test.
T = 12
IX = "intersection_1_1"


# ==================================================================================
# The miniature corpus
# ==================================================================================
def _rewards(ep_index: int, length: int = T) -> np.ndarray:
    """One episode's local reward stream: integer-valued, negative, and distinct per episode.

    Distinct per episode is the point: with identical streams every episode return would be the
    same and ``max`` could not be told from ``min``, which is exactly mutation M-NAIVE-1.
    """
    return np.asarray(
        [-(3 + ((ep_index * 7 + t * 5) % 29)) for t in range(length)], dtype=np.float32
    )


def _episode_arrays(
    *, ep_index: int, draw: int, state_dim: int, n_actions: int, length: int = T
) -> dict[str, np.ndarray]:
    """One episode laid out as ``TrajectoryLogger._build_arrays`` lays it out (v1.1)."""
    rewards = _rewards(ep_index, length)
    state = np.arange((length + 1) * state_dim, dtype=np.float32).reshape(length + 1, state_dim)
    state = state + float(ep_index)
    return {
        "format_version": np.asarray("1.1"),
        "ix_ids": np.asarray([IX], dtype=np.str_),
        "lane_ids": np.asarray([f"lane_{i}" for i in range(8)], dtype=np.str_),
        "metric_keys": np.asarray([], dtype=np.str_),
        "metrics": np.zeros((length + 1, 0), dtype=np.float32),
        "vehicle_count": np.asarray([10 + t for t in range(length + 1)], dtype=np.int64),
        "sim_time": np.asarray([10.0 * t for t in range(length + 1)], dtype=np.float32),
        "step": np.asarray(list(range(length + 1)), dtype=np.int64),
        "lane_vehicle_count": np.ones((length + 1, 8), dtype=np.int32),
        "lane_waiting_vehicle_count": np.ones((length + 1, 8), dtype=np.int32),
        "att_per_step": np.asarray([1.5 * t for t in range(length + 1)], dtype=np.float32),
        "episode_length": np.asarray(length, dtype=np.int64),
        "terminated": np.asarray(False, dtype=np.bool_),
        "truncated": np.asarray(True, dtype=np.bool_),
        "engine_seed": np.asarray(437485271, dtype=np.int64),
        "flow_draw": np.asarray(draw, dtype=np.int64),
        "ix0_state": state,
        "ix0_avail_mask": np.ones((length + 1, n_actions), dtype=np.bool_),
        "ix0_current_phase": np.asarray([t % n_actions for t in range(length + 1)], dtype=np.int64),
        "ix0_time_in_phase": np.asarray([float(t % 4) for t in range(length + 1)], dtype=np.float32),
        "ix0_action": np.asarray([t % n_actions for t in range(length)], dtype=np.int64),
        "ix0_local_reward": rewards,
        "global_reward": rewards.copy(),
    }


def _episode_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(arrays["ix0_action"].astype("<i8", copy=False).tobytes())
    digest.update(arrays["global_reward"].astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def write_corpus_dir(
    directory: Path,
    draws: Sequence[int],
    *,
    ep_offset: int = 0,
    state_dim: int = at.EXPECTED_STATE_DIM,
    n_actions: int = at.EXPECTED_N_ACTIONS,
    scenario_id: str = at.SCENARIO_ID,
    with_digest_file: bool = True,
) -> Path:
    """Write one collection directory, and its ``SHA256SUMS`` as section 3.2 writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    for offset, draw in enumerate(draws):
        ep_index = ep_offset + offset
        arrays = _episode_arrays(
            ep_index=ep_index, draw=draw, state_dim=state_dim, n_actions=n_actions
        )
        filename = f"ep{ep_index:06d}_seed1000_draw{draw}.npz"
        with open(directory / filename, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        episodes.append(
            {
                "filename": filename,
                "episode_length": T,
                "total_global_reward": float(arrays["global_reward"].sum()),
                "engine_seed": 1000,
                "flow_draw": int(draw),
                "episode_sha256": _episode_digest(arrays),
            }
        )
    manifest = {
        "format_version": "1.1",
        "git_hash": "0" * 40,
        "lane_count": 8,
        "lane_ids_sha256": hashlib.sha256(b"lanes").hexdigest(),
        "run_metadata": {
            "scenario_id": scenario_id,
            "backend": "sumo",
            "behavior_policy": "maxpressure",
            "base_seed": 1000,
            "global_reward_weight": 0.0,
            "local_reward_fn": "queue_length",
        },
        "episodes": episodes,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if with_digest_file:
        write_digest_file(directory)
    return directory


def write_digest_file(directory: Path) -> Path:
    """Section 3.2's recipe: the ``.npz`` files and ``manifest.json``, excluding this file."""
    names = sorted([p.name for p in directory.glob("*.npz")] + ["manifest.json"])
    lines = [
        f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}" for name in names
    ]
    path = directory / at.CORPUS_DIGEST_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_corpus(
    root: Path,
    *,
    draws: Sequence[int] = at.ANCHOR_DRAW_IDS,
    state_dim: int = at.EXPECTED_STATE_DIM,
    n_actions: int = at.EXPECTED_N_ACTIONS,
) -> tuple[Path, Path]:
    """The anchor's two directories: the A17(b) band and the band section 3.2 collected."""
    half = len(draws) // 2
    first = write_corpus_dir(
        root / "hz1x1_sumo_maxpressure", draws[:half], state_dim=state_dim, n_actions=n_actions
    )
    second = write_corpus_dir(
        root / "hz1x1_sumo_maxpressure_301_400",
        draws[half:],
        ep_offset=half,
        state_dim=state_dim,
        n_actions=n_actions,
    )
    return first, second


def fake_output_root(root: Path, *, training_draw_ids: Sequence[int] = range(1, 201)) -> Path:
    """The two seed-101 checkpoints ``disjointness_record`` reads ``training_draw_ids`` from.

    Fabricated rather than gated on the real ``output/``: the function under test only reads that
    one provenance field, and a fixture makes the disjointness refusal reachable in CI.  The real
    checkpoints are asserted separately, in the gated test at the end of this file.
    """
    from offline.transfer_calibration import SUBJECTS

    ids = [int(d) for d in training_draw_ids]
    for spec in SUBJECTS.values():
        directory = root / spec["subdir"]
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"provenance": {"training_draw_ids": ids}}, directory / f"{spec['stem']}101.pt"
        )
    return root


# ==================================================================================
# The independent route: np.cumsum over the raw .npz, touching neither module
# ==================================================================================
def naive_prompt_by_cumsum(directories: Sequence[Path]) -> tuple[float, float]:
    """A18(a)'s two quantities, recomputed from the reward streams and nothing else.

    ``RTG_t = sum_{u >= t} r_u``, so ``rtg[0]`` is the episode return.  This never constructs a
    ``TrajectoryWindowDataset``, never calls ``stack_dataset`` and never reads ``stats.rtg``: it
    opens the ``.npz`` files, reverses, cumulatively sums and reverses back.
    """
    returns: list[float] = []
    every_rtg: list[np.ndarray] = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.npz")):
            with np.load(path) as data:
                rewards = data["ix0_local_reward"].astype(np.float64)
            rtg = np.cumsum(rewards[::-1])[::-1]
            returns.append(float(rtg[0]))
            every_rtg.append(rtg)
    flat = np.concatenate(every_rtg)
    return max(returns), float(np.max(np.abs(flat)))


# ==================================================================================
# load_anchor_corpus -- the guard band
# ==================================================================================
def test_the_corpus_loads_from_both_directories_in_one_call(tmp_path: Path) -> None:
    """Section 3.3. Two v1.1 directories, ONE ``build_training_dataset`` call, 200 episodes.

    The union of draw ids is asserted to be exactly A18(a)'s band -- not merely its size, because
    200 episodes over the wrong 200 draws is the failure this is here to catch.
    """
    dirs = build_corpus(tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))

    assert corpus.facts.n_episodes == 200
    assert corpus.facts.state_dim == 25
    assert corpus.facts.n_actions == 8
    assert corpus.facts.draw_ids == at.ANCHOR_DRAW_IDS
    assert set(corpus.facts.draw_ids) == set(range(201, 401))
    assert len(corpus.facts.episodes) == 200
    assert corpus.facts.scenario_id == at.SCENARIO_ID
    assert corpus.facts.ix_id == IX
    # one window per decision per episode
    assert corpus.facts.n_windows == 200 * T
    assert tuple(corpus.stats.draw_ids) == at.ANCHOR_DRAW_IDS
    assert corpus.stacked["state"].shape == (200 * T, 20, 25)
    assert set(corpus.facts.corpus_digests) == {str(d) for d in dirs}


def test_a_thirty_two_wide_corpus_is_refused_because_it_never_went_through_a16s_door(
    tmp_path: Path,
) -> None:
    """Section 3.3: *a 32 here means the corpus is not canonical*.

    32 is ``sumo_state_width``; 25 is ``canonical_state_width``.  A CityFlow-frame model fed the
    raw SUMO frame does not error -- it produces a number that means nothing -- so the width is a
    refusal at load rather than a surprise at evaluation.
    """
    dirs = build_corpus(tmp_path / "corpus", state_dim=32)
    with pytest.raises(ValueError, match="25"):
        at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))


def test_a_corpus_that_is_not_two_hundred_episodes_is_refused(tmp_path: Path) -> None:
    """199 episodes would train a different model under the same name.

    ⚠️ **The match is the MANIFEST guard's own wording, and that is not fussiness.** A looser
    ``match="200"`` SURVIVED the mutation ``len(listed) != expected`` -> ``len(listed) < 1``:
    a second, later check -- the loader's own episode count -- then fired with a message that also
    contains "200", so the test looked green while the guard it names was dead. Matching the
    distinctive sentence is what tells the two refusals apart.
    """
    dirs = build_corpus(tmp_path / "corpus", draws=at.ANCHOR_DRAW_IDS[:-1])
    with pytest.raises(ValueError, match="these directories list 199"):
        at.load_anchor_corpus(
            dirs,
            output_root=fake_output_root(tmp_path / "output"),
            expected_draw_ids=at.ANCHOR_DRAW_IDS[:-1],
        )


def test_the_early_refusals_fire_before_a_single_episode_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A18(a): *asserted [...] before the first episode.* That is an ORDERING, so it is tested.

    ``build_training_dataset`` is the call that opens the ``.npz`` files. It is replaced by one
    that explodes, and both early refusals must still fire -- which they can only do if they
    precede it. This is also why the manifest-count check exists beside the loader's own count:
    it refuses before 200 episodes have been read and before the disjointness call.
    """
    import offline.dt_gate as dt_gate

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an episode was read before the early refusals had run")

    monkeypatch.setattr(dt_gate, "build_training_dataset", explode, raising=True)
    output_root = fake_output_root(tmp_path / "output")

    short = build_corpus(tmp_path / "short", draws=at.ANCHOR_DRAW_IDS[:-1])
    with pytest.raises(ValueError, match="these directories list 199"):
        at.load_anchor_corpus(
            short, output_root=output_root, expected_draw_ids=at.ANCHOR_DRAW_IDS[:-1]
        )

    band = tuple(range(200, 400))
    overlapping = build_corpus(tmp_path / "overlap", draws=band)
    with pytest.raises(ValueError, match="not disjoint"):
        at.load_anchor_corpus(overlapping, output_root=output_root, expected_draw_ids=band)

    # ...and the control: on a VALID corpus the exploding loader IS reached, so the two refusals
    # above are genuinely earlier rather than the loader simply never being called.
    good = build_corpus(tmp_path / "good")
    with pytest.raises(AssertionError, match="before the early refusals"):
        at.load_anchor_corpus(good, output_root=output_root)


def test_a_corpus_over_the_wrong_draws_is_refused_even_at_the_right_size(tmp_path: Path) -> None:
    """200 episodes over draws 401-600 is 200 episodes and is not A18(a)'s corpus."""
    wrong = tuple(range(401, 601))
    dirs = build_corpus(tmp_path / "corpus", draws=wrong)
    with pytest.raises(ValueError, match="draw"):
        at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))


def test_a_single_directory_is_refused_because_the_anchor_spans_two_bands(tmp_path: Path) -> None:
    """A18(a)'s corpus is 201-300 PLUS 301-400, and they are two collection runs."""
    one = write_corpus_dir(tmp_path / "corpus" / "only", at.ANCHOR_DRAW_IDS)
    with pytest.raises(ValueError, match="two"):
        at.load_anchor_corpus([one], output_root=fake_output_root(tmp_path / "output"))


def test_a_corpus_whose_bytes_moved_since_its_digest_file_is_refused(tmp_path: Path) -> None:
    """B3(a): *a digest checked once is not a digest checked when used.*

    The ``SHA256SUMS`` written in section 3.2 is re-verified HERE, at consumption, so an episode
    edited between collection and training is a refusal rather than a silently different corpus.
    """
    dirs = build_corpus(tmp_path / "corpus")
    victim = sorted(dirs[0].glob("*.npz"))[0]
    with np.load(victim) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["ix0_local_reward"] = arrays["ix0_local_reward"] - 1.0
    with open(victim, "wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ValueError, match=victim.name):
        at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))


def test_a_corpus_directory_without_a_digest_file_is_refused(tmp_path: Path) -> None:
    """Section 3.2 requires one in BOTH halves; a missing one is a gap, not a pass."""
    dirs = build_corpus(tmp_path / "corpus")
    (dirs[1] / at.CORPUS_DIGEST_NAME).unlink()
    with pytest.raises(FileNotFoundError, match=at.CORPUS_DIGEST_NAME):
        at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))


# ==================================================================================
# The disjointness assertion (section 3.1 / A18(a))
# ==================================================================================
def test_a_training_pool_overlapping_the_subjects_draws_is_refused_before_any_training(
    tmp_path: Path,
) -> None:
    """A18(a): *asserted by assert_probe_draws_disjoint before the first episode.*

    ``expected_draw_ids`` is passed so the band-identity check PASSES and the disjointness call is
    the thing that refuses -- otherwise this test could not tell the two apart, and removing the
    disjointness call would leave it green.
    """
    band = tuple(range(200, 400))  # draw 200 is a training draw of both subjects
    dirs = build_corpus(tmp_path / "corpus", draws=band)
    with pytest.raises(ValueError, match="not disjoint"):
        at.load_anchor_corpus(
            dirs,
            output_root=fake_output_root(tmp_path / "output"),
            expected_draw_ids=band,
        )


def test_a_training_pool_reaching_into_the_held_out_wall_is_refused(tmp_path: Path) -> None:
    """The other half of the same rule: the pool the result is measured over is off limits."""
    band = tuple(range(901, 1101))  # 1000-1099 are held out
    dirs = build_corpus(tmp_path / "corpus", draws=band)
    with pytest.raises(ValueError, match="held-out"):
        at.load_anchor_corpus(
            dirs,
            output_root=fake_output_root(tmp_path / "output"),
            expected_draw_ids=band,
        )


def test_the_disjointness_record_is_carried_into_the_facts(tmp_path: Path) -> None:
    """It is evidence, so it is recorded, not merely asserted and discarded."""
    dirs = build_corpus(tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))
    record = corpus.facts.disjointness
    assert record["disjoint"] is True
    assert record["n_probe_draws"] == 200
    assert [s["overlap_with_probe"] for s in record["sources"]] == [0] * len(record["sources"])


# ==================================================================================
# The naive in-domain rule -- three routes, ``==`` between all of them
# ==================================================================================
def test_the_naive_rule_matches_an_independent_cumsum_over_the_raw_npz(tmp_path: Path) -> None:
    """A18(a)'s two quantities, recomputed by a route that touches neither module.

    ``==`` and not ``approx``: the reward streams are integer-valued, so the float32 corpus and a
    float64 ``np.cumsum`` are exactly equal, and a rounding change must be a failure rather than
    noise.  The same holds on the real corpus -- verified at the plan gate, ``docs/plans/p7.3b.md``
    section 0.2.
    """
    dirs = build_corpus(tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))

    target_rtg, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)
    expected_target, expected_scale = naive_prompt_by_cumsum(dirs)

    assert target_rtg == expected_target
    assert rtg_scale == expected_scale
    # and the quantities are what A18(a) says they are, not merely equal to each other
    assert target_rtg == max(
        float(np.load(p)["ix0_local_reward"].astype(np.float64).sum())
        for d in dirs
        for p in sorted(Path(d).glob("*.npz"))
    ), "target_rtg is the MAXIMUM EPISODE RETURN of the training set"
    assert rtg_scale > 0.0, "rtg_scale is the largest ABSOLUTE rtg, so it is positive"
    assert rtg_scale != abs(target_rtg), (
        "the fixture must distinguish |max episode return| from max|RTG|, or a mutation "
        "confusing the two would survive"
    )


def test_the_stacked_route_and_p4s_statistics_route_agree_exactly(tmp_path: Path) -> None:
    """The module's own double-compute: the stacked route against P4's ``stats.rtg`` route.

    ``dt_gate._run_train:1381-1387`` is the only prior implementation of this rule, and the anchor
    must use the same one.  ``load_anchor_corpus`` recomputes both and raises on a disagreement;
    this asserts the two agree on a corpus where they should.
    """
    dirs = build_corpus(tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))

    target_rtg, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)

    summary = corpus.stats.rtg[at.SCENARIO_ID][IX]
    assert rtg_scale == max(abs(summary.min), abs(summary.max))
    assert rtg_scale == abs(summary.min), "the streams are negative, so |min| is the scale"


def test_left_padding_is_excluded_from_the_scale(tmp_path: Path) -> None:
    """The stacked ``rtg`` carries zeros where a window is left-padded (``dataset.py:789``).

    Those zeros are not RTG values.  They do not change the answer on a negative-reward corpus --
    which is exactly why this must be asserted rather than relied on: a corpus with positive
    rewards would make a scale computed over the padding wrong, silently.
    """
    dirs = build_corpus(tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))

    rtg = corpus.stacked["rtg"][:, :, 0]
    mask = corpus.stacked["attention_mask"]
    assert not bool(mask.all()), "the fixture must contain left-padded windows"
    assert float(rtg[~mask].abs().max()) == 0.0, "padding is zero"
    _, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)
    assert rtg_scale == float(rtg[mask].abs().max())


# ==================================================================================
# train_anchor and the artifact
# ==================================================================================
def _train_small(
    tmp_path: Path,
    steps: int = 3,
    seeds: Sequence[int] = (101, 202),
    corpus_root: Path | None = None,
) -> Any:
    """Train a few steps over a miniature corpus.

    *corpus_root* is separate from *tmp_path* so two runs can share ONE corpus directory. That
    distinction is load-bearing for the determinism test below: the checkpoint's provenance
    records ``dataset_dirs``, so two runs over two directories with identical CONTENT but
    different PATHS produce different checkpoint bytes -- correctly, and by design.
    """
    dirs = build_corpus(corpus_root if corpus_root is not None else tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))
    target_rtg, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results = at.train_anchor(
        corpus,
        checkpoint_dir=checkpoints,
        device=torch.device("cpu"),
        target_rtg=target_rtg,
        rtg_scale=rtg_scale,
        seeds=seeds,
        declared_gradient_steps=steps,
    )
    artifact = at.build_artifact(
        corpus,
        results,
        checkpoint_dir=checkpoints,
        target_rtg=target_rtg,
        rtg_scale=rtg_scale,
    )
    return corpus, results, artifact, checkpoints


def test_the_checkpoint_carries_a18as_recipe_and_the_anchors_own_prompt(tmp_path: Path) -> None:
    """P4's recipe verbatim: context 20, 3 layers, 1 head, d_model 128, dropout 0.1.

    Read back off the saved checkpoint, because that is what ``DTAgent.load`` will rebuild from --
    a constant that drifted in ``dt_gate`` or ``DTAgent`` must fail here rather than train a
    different model under A18(a)'s name.
    """
    _, _, _, checkpoints = _train_small(tmp_path)
    payload = torch.load(
        at.checkpoint_path_for_seed(checkpoints, 101), map_location="cpu", weights_only=False
    )
    config = payload["config"]
    assert config["context_length"] == 20
    assert config["n_layer"] == 3
    assert config["n_head"] == 1
    assert config["d_model"] == 128
    assert config["dropout"] == 0.1
    assert config["state_dim"] == 25
    assert config["n_actions"] == 8
    assert payload["scenario_id"] == at.SCENARIO_ID
    assert payload["normalise"] is True


def test_the_artifact_digests_match_the_checkpoints_on_disk(tmp_path: Path) -> None:
    """G1's shape: the record is only evidence if it names the bytes that exist."""
    _, results, artifact, checkpoints = _train_small(tmp_path)
    assert len(artifact["seeds"]) == 2
    for row in artifact["seeds"]:
        path = at.checkpoint_path_for_seed(checkpoints, int(row["seed"]))
        on_disk = hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["checkpoint_sha256"] == on_disk, f"seed {row['seed']}'s digest is stale"
        assert row["gradient_steps"] == 3
        assert isinstance(row["final_loss"], float)
        assert isinstance(row["seconds"], float)


def test_the_artifact_records_a18as_step_count_and_a3s_null_raise(tmp_path: Path) -> None:
    """Amendment A3: A18(a) fixes 40,000 outright and P4's plateau raise is not in the recipe."""
    assert at.DECLARED_GRADIENT_STEPS == 40_000
    assert at.RAISE_TO is None
    assert at.TRAINING_SEEDS == (101, 202, 303, 404, 505)

    _, _, artifact, _ = _train_small(tmp_path)
    assert artifact["raise_to"] is None
    assert json.dumps(artifact)  # the artifact must be serialisable, null and all
    assert '"raise_to": null' in json.dumps(artifact, indent=1)
    assert artifact["recipe"]["batch_size"] == 64
    assert artifact["recipe"]["learning_rate"] == 1e-4
    assert artifact["recipe"]["weight_decay"] == 1e-4
    assert artifact["recipe"]["warmup_steps"] == 1000
    assert artifact["recipe"]["grad_clip"] == 0.25
    assert artifact["recipe"]["context_length"] == 20


def test_the_artifact_carries_the_corpus_digests_the_draw_ids_and_the_prompt(
    tmp_path: Path,
) -> None:
    """Section 3.3's required contents, each checked against the thing it claims to describe."""
    corpus, _, artifact, _ = _train_small(tmp_path)

    assert artifact["format_version"] == at.ARTIFACT_FORMAT_VERSION
    assert artifact["n_episodes"] == 200
    assert artifact["episode_ids"] and len(artifact["episode_ids"]) == 200
    assert [int(e["flow_draw"]) for e in artifact["episode_ids"]] == list(at.ANCHOR_DRAW_IDS)

    for directory, digest in artifact["corpus"]["digest_files"].items():
        recorded = Path(directory) / at.CORPUS_DIGEST_NAME
        assert digest == hashlib.sha256(recorded.read_bytes()).hexdigest()

    target_rtg, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)
    assert artifact["target_rtg"] == target_rtg
    assert artifact["rtg_scale"] == rtg_scale
    assert artifact["prompt_rule"] == "naive_in_domain"
    assert artifact["normalisation_stats"]["draw_ids"] == list(at.ANCHOR_DRAW_IDS)
    assert artifact["disjointness"]["disjoint"] is True


def test_the_artifact_says_what_the_anchor_is_not(tmp_path: Path) -> None:
    """A18(a)'s sentence is registered and must travel with the record, not only with the packet.

    A18(b) registers why there is no online SUMO anchor; a reader of this file must not be able to
    take the anchor for an upper bound.
    """
    _, _, artifact, _ = _train_small(tmp_path)
    text = artifact["what_this_is_not"]
    assert "upper bound" in text
    assert "online" in text
    assert "k = 200" in artifact["role"]


def test_two_runs_of_one_seed_produce_the_same_weights(tmp_path: Path) -> None:
    """Determinism is a feature: the same seed, corpus and step count give the same bytes.

    Without this, a re-run that produced different weights would still match its OWN artifact and
    nothing would notice.  ``docs/plans/p5.3b.md`` section 2 records the precedent this rests on:
    a retrain at 40,000 steps reproduced the digest committed in ``p4_6_training.json``.

    ⚠️ **Both runs share ONE corpus directory, and that is required rather than tidy.** The
    checkpoint's provenance records ``dataset_dirs``, so two corpora with identical content at
    different paths give different checkpoint BYTES with identical WEIGHTS -- which is the
    property the next test pins deliberately.
    """
    corpus_root = tmp_path / "corpus"
    first = _train_small(tmp_path / "a", seeds=(101,), corpus_root=corpus_root)
    second = _train_small(tmp_path / "b", seeds=(101,), corpus_root=corpus_root)
    digest = [
        hashlib.sha256(at.checkpoint_path_for_seed(run[3], 101).read_bytes()).hexdigest()
        for run in (first, second)
    ]
    weights = [
        torch.load(at.checkpoint_path_for_seed(run[3], 101), map_location="cpu",
                   weights_only=False)["model"]
        for run in (first, second)
    ]
    assert set(weights[0]) == set(weights[1])
    for key in weights[0]:
        assert torch.equal(weights[0][key], weights[1][key]), f"{key} differs between runs"
    assert digest[0] == digest[1], "the checkpoint bytes must be reproducible, not merely the weights"


# ==================================================================================
# A5 -- the ROUTE test: the anchor's prompt never comes from P7.2b's calibration
# ==================================================================================
def test_the_prompt_never_reaches_p7_2b_s_calibration_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment A5. The proof is the ROUTE, never a value.

    On the 201-300 half the naive in-domain target and A17's Rule A ``q = 1.0`` are the SAME
    NUMBER by construction -- both are the maximum of the same 100 SUMO probe returns, -20809.0
    for both subjects in ``docs/data/p7_2b_calibration.json``.  A test that compared values would
    therefore be vacuous today and would only stop being vacuous by accident.  So: make the
    calibration loader explode, and require the anchor to be unaffected.
    """
    import offline.transfer_curve as tcv

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "the anchor's prompt must come from its own training set, never from P7.2b's "
            "CityFlow-trained calibration artifact"
        )

    monkeypatch.setattr(tcv, "load_calibration", explode, raising=True)
    monkeypatch.setattr(
        "offline.transfer_calibration.targets_for_subject", explode, raising=False
    )
    monkeypatch.setattr(tcv, "targets_for_subject", explode, raising=True)

    dirs = build_corpus(tmp_path / "corpus")
    corpus = at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))
    target_rtg, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)

    assert isinstance(target_rtg, float) and isinstance(rtg_scale, float)
    # The weaker, secondary assertion, labelled as such: today it happens to hold, and A5 says it
    # must not be relied on.
    calibration = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "data" / "p7_2b_calibration.json").read_bytes()
    )
    q1_targets = {
        float(row["target_rtg"])
        for rows in calibration["targets"].values()
        for row in rows
        if row["rule"] == "rule_a"
    }
    assert q1_targets == {-20809.0}, (
        "the collision Amendment A5 names is still in the artifact; if this changes, the packet's "
        "statement about it must change too"
    )


def test_the_module_never_imports_the_calibration_artifact_at_all() -> None:
    """The route, pinned statically as well: no name from P7.2b's calibration path appears here.

    A grep is weak on its own, which is why the executed route test above is the primary; this
    catches the case where a future edit adds the import without any test exercising it.
    """
    source = Path(at.__file__).read_text(encoding="utf-8")
    for forbidden in ("load_calibration", "targets_for_subject", "p7_2b_calibration"):
        assert forbidden not in source, (
            f"{forbidden!r} appears in offline/anchor_training.py; A18(a)'s prompt is the naive "
            "in-domain rule over the anchor's own 200 episodes"
        )


# ==================================================================================
# The real corpus (G2: the gated test names the artifacts it consumes)
# ==================================================================================
def _real_corpus_available() -> bool:
    return all(
        (d / "manifest.json").is_file() and (d / at.CORPUS_DIGEST_NAME).is_file()
        for d in REAL_CORPUS_DIRS
    ) and (REAL_OUTPUT_ROOT / "p4_dt" / "dt_seed101.pt").is_file()


@pytest.mark.skipif(
    not _real_corpus_available(),
    reason=(
        "needs the real anchor corpus and P4's checkpoints: "
        "datasets_sumo_v11/hz1x1_sumo_maxpressure{,_301_400}/ with SHA256SUMS, and "
        "output/p4_dt/dt_seed101.pt (both gitignored, main tree only)"
    ),
)
def test_the_real_two_hundred_episode_corpus_loads_and_its_prompt_recomputes(
    tmp_path: Path,
) -> None:
    """The one test that reads what the campaign will actually train on.

    Both quantities are recomputed by the ``np.cumsum`` route over the real ``.npz`` files and
    compared with ``==``.  This is the assertion ``docs/plans/p7.3b.md`` section 0.2 measured on
    half the corpus; here it runs on all 200 episodes.
    """
    corpus = at.load_anchor_corpus(REAL_CORPUS_DIRS, output_root=REAL_OUTPUT_ROOT)

    assert corpus.facts.n_episodes == 200
    assert corpus.facts.state_dim == 25, "A16's canonical width, not SUMO's 32"
    assert corpus.facts.n_actions == 8
    assert corpus.facts.draw_ids == at.ANCHOR_DRAW_IDS
    assert corpus.facts.n_windows == 200 * 360
    assert corpus.facts.disjointness["disjoint"] is True

    target_rtg, rtg_scale = at.naive_in_domain_prompt(corpus.stacked)
    expected_target, expected_scale = naive_prompt_by_cumsum(REAL_CORPUS_DIRS)
    assert target_rtg == expected_target
    assert rtg_scale == expected_scale

    # The half-corpus values, measured at the plan gate on draws 201-300 alone. The 200-episode
    # target is a max over a superset, so it can only rise or stay equal.
    assert target_rtg >= -20809.0
    assert rtg_scale >= 25257.0


def test_the_checkpoint_records_which_corpus_trained_it(tmp_path: Path) -> None:
    """The reason the determinism test must share one corpus directory, pinned as a property.

    Found by the determinism test failing on a fixture that built two corpora with identical
    content at different paths: the weights were bit-identical and the files were not. That is
    correct behaviour -- a checkpoint says which corpus produced it -- but nothing asserted it, so
    a future edit that dropped ``dataset_dirs`` from the provenance would look like a bug fix.
    """
    corpus_root = tmp_path / "shared"
    same_a = _train_small(tmp_path / "a", seeds=(101,), corpus_root=corpus_root)
    other_b = _train_small(tmp_path / "b", seeds=(101,))          # its own corpus, same content

    def payload(run: Any) -> dict[str, Any]:
        return torch.load(
            at.checkpoint_path_for_seed(run[3], 101), map_location="cpu", weights_only=False
        )

    first, second = payload(same_a), payload(other_b)
    assert first["provenance"]["dataset_dirs"] != second["provenance"]["dataset_dirs"]
    assert first["provenance"]["corpus_digest_files"] != second["provenance"]["corpus_digest_files"]
    # identical CONTENT, so the streams and therefore the weights agree exactly...
    for key in first["model"]:
        assert torch.equal(first["model"][key], second["model"][key])
    # ...and the FILES still differ, because provenance is part of the checkpoint.
    assert (
        hashlib.sha256(at.checkpoint_path_for_seed(same_a[3], 101).read_bytes()).hexdigest()
        != hashlib.sha256(at.checkpoint_path_for_seed(other_b[3], 101).read_bytes()).hexdigest()
    )
    assert first["provenance"]["tier"] == "anchor_k200"
    assert first["provenance"]["prompt_rule"] == "naive_in_domain"
    assert first["provenance"]["raise_to"] is None
    assert first["provenance"]["training_draw_ids"] == list(at.ANCHOR_DRAW_IDS)


def test_the_two_prompt_routes_are_actually_compared_and_a_disagreement_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_anchor_corpus``'s own double-compute must REFUSE, not prefer one route.

    A wrong prompt does not crash -- it produces a plausible number in a table -- so the stacked
    route and P4's ``stats.rtg`` route are computed independently and compared. This makes them
    disagree by one unit and requires the loader to raise; without the comparison the corpus would
    load happily and the disagreement would reach the campaign.
    """
    dirs = build_corpus(tmp_path / "corpus")
    real = at.naive_in_domain_prompt

    def skewed(stacked: Any) -> tuple[float, float]:
        target, scale = real(stacked)
        return target + 1.0, scale

    monkeypatch.setattr(at, "naive_in_domain_prompt", skewed, raising=True)
    with pytest.raises(ValueError, match="disagrees between its two routes"):
        at.load_anchor_corpus(dirs, output_root=fake_output_root(tmp_path / "output"))
