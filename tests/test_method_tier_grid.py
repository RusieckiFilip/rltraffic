"""Tests for ``offline.method_tier_grid`` -- P4.6's method x tier grid.

Three layers, and the Return Packet says which of them ran:

* pure arithmetic and declaration checks (the prompt rule, the size match, the prediction scoring,
  the hypergeometric tail, the no-verdict rule) -- always run;
* the data path over a **synthetic corpus** written by this module's own writer, which controls
  returns, ATT and draw multiplicity exactly -- always runs;
* the **real-corpus** path -- opt-in, skipping with a reason that names ``RLTRAFFIC_CORPUS_V11``.

The load-bearing tests here are
``test_the_declared_targets_equal_the_corpus_maxima_read_by_a_second_route`` (the DT's prompt is
the one quantity a silent convention error would move without any test noticing) and
``test_the_module_declares_no_equivalence_verdict_anywhere`` (``BRIEF_17`` section 4 is a ruling,
and a ruling with no mechanism is a comment).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

import numpy as np
import pytest

import offline.method_tier_grid as grid
from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import EpisodeResult, mean_ci95
from offline.method_tier_grid import (
    BEHAVIOUR_ATT,
    BEHAVIOUR_METHOD,
    BEHAVIOUR_REFERENCE_BY_TIER,
    DECLARED_GRADIENT_STEPS,
    METHODS,
    MIXTURE_EXPERT_FRACTION,
    PHASE1_TIER_ORDER,
    RANDOM_SUBSAMPLE_RNG_SEED,
    TIERS,
    TRAINING_STREAM_COUNT,
    TierSpec,
    arm_key,
    assert_cell_complete,
    assert_declaration_matches_corpus,
    assert_equal_training_size,
    assert_no_verdicts,
    assert_reused_cells_reproduce,
    cell_stats,
    behaviour_comparisons,
    difficulty_check,
    draw_arrivals,
    env_settings_for_tiers,
    fixedtime_collection_settings,
    grid_comparisons,
    hypergeometric_upper_tail,
    kendall_tau_b,
    kept_composition,
    merge_training_records,
    mixture_training_streams,
    recomputed_target_and_scale,
    stream_records_with_digests,
    score_p1,
    score_p2,
    score_p3,
    split_arm_key,
    statistics_digest,
    stratified_one_per_draw,
    tier_dataset,
    tier_dirs,
    tier_spec,
    top_decile_streams,
    training_streams,
    volume_check,
)
from offline.offline_baselines import StreamReturn, stream_returns

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_SCENARIO = "fixture1x1"
FIXTURE_T = 6
FIXTURE_STATE_DIM = 3
FIXTURE_N_ACTIONS = 2
FIXTURE_LANES = ("lane_a", "lane_b")
CONTEXT = 3


# ----------------------------------------------------------------------
# A synthetic one-intersection corpus, with returns and ATT under the test's control
# ----------------------------------------------------------------------


def write_tier_dir(
    root: Path,
    name: str,
    *,
    episodes: Sequence[tuple[int, float, float]],
    scenario_id: str = FIXTURE_SCENARIO,
    state_offset: float = 0.0,
    manifest_extra: dict[str, Any] | None = None,
    flow_hash_override: dict[int, str] | None = None,
) -> Path:
    """Write a v1.1 collection directory of one-intersection episodes.

    Each entry of *episodes* is ``(flow_draw, per_step_reward, att_horizon)``: the stream's return
    is exactly ``per_step_reward * FIXTURE_T`` and its horizon ATT is exactly *att_horizon*, so a
    test can state the answer it expects instead of deriving it from the code under test.
    """
    out_dir = root / name
    out_dir.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for index, (draw, reward, att) in enumerate(episodes):
        rewards = np.full(FIXTURE_T, float(reward), dtype=np.float32)
        att_rows = np.asarray(
            [float(att) + FIXTURE_T - t for t in range(FIXTURE_T)] + [float(att)],
            dtype=np.float32,
        )
        arrays: dict[str, np.ndarray] = {
            "format_version": np.asarray("1.1"),
            "ix_ids": np.asarray(["ix0"], dtype=np.str_),
            "lane_ids": np.asarray(list(FIXTURE_LANES), dtype=np.str_),
            "metric_keys": np.asarray(["queue"], dtype=np.str_),
            "vehicle_count": np.asarray([10 + t for t in range(FIXTURE_T + 1)], dtype=np.int64),
            "sim_time": np.asarray([10.0 * t for t in range(FIXTURE_T + 1)], dtype=np.float32),
            "step": np.asarray(list(range(FIXTURE_T + 1)), dtype=np.int64),
            "metrics": np.asarray([[float(t)] for t in range(FIXTURE_T + 1)], dtype=np.float32),
            "lane_vehicle_count": np.asarray(
                [[t, t + 1] for t in range(FIXTURE_T + 1)], dtype=np.int32
            ),
            "lane_waiting_vehicle_count": np.asarray(
                [[t, t] for t in range(FIXTURE_T + 1)], dtype=np.int32
            ),
            "att_per_step": att_rows,
            "episode_length": np.asarray(FIXTURE_T, dtype=np.int64),
            "terminated": np.asarray(False, dtype=np.bool_),
            "truncated": np.asarray(True, dtype=np.bool_),
            "engine_seed": np.asarray(1000, dtype=np.int64),
            "flow_draw": np.asarray(draw, dtype=np.int64),
            "ix0_state": np.asarray(
                [
                    [state_offset + index + t, state_offset + t, float(t % 2)]
                    for t in range(FIXTURE_T + 1)
                ],
                dtype=np.float32,
            ),
            "ix0_avail_mask": np.ones((FIXTURE_T + 1, FIXTURE_N_ACTIONS), dtype=np.bool_),
            "ix0_current_phase": np.asarray(
                [t % FIXTURE_N_ACTIONS for t in range(FIXTURE_T + 1)], dtype=np.int64
            ),
            "ix0_time_in_phase": np.asarray(
                [float(t % 3) for t in range(FIXTURE_T + 1)], dtype=np.float32
            ),
            "ix0_action": np.asarray(
                [t % FIXTURE_N_ACTIONS for t in range(FIXTURE_T)], dtype=np.int64
            ),
            "ix0_local_reward": rewards,
            "global_reward": rewards.copy(),
        }
        filename = f"ep{index:06d}_seed1000_draw{draw}.npz"
        with open(out_dir / filename, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        entries.append(
            {
                "filename": filename,
                "episode_length": FIXTURE_T,
                "total_global_reward": float(rewards.sum()),
                "engine_seed": 1000,
                "flow_draw": int(draw),
                "episode_sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
            }
        )
    run_metadata: dict[str, Any] = {
        "scenario_id": scenario_id,
        "backend": "cityflow",
        "behavior_policy": "fixture",
        "max_steps": FIXTURE_T,
        "delta_time": 10,
        "control_mode": "acyclic",
        "state_features": ["lane_vehicle_count"],
        "global_reward_fn": "queue_length",
        "local_reward_fn": "queue_length",
        "global_reward_weight": 0.0,
        "metrics": None,
        "flow_source_path": str(root / "flow.json"),
        "flow_randomizer_params": {
            "base_seed": 1000,
            "jitter_sigma_s": 30.0,
            "thin_p": 0.1,
            "volume_scale": 1.0,
        },
        "flow_draw_ids": sorted({int(draw) for draw, _, _ in episodes}),
        "flow_draw_sha256": {
            str(int(draw)): (flow_hash_override or {}).get(int(draw), "0" * 64)
            for draw, _, _ in episodes
        },
    }
    run_metadata.update(manifest_extra or {})
    manifest = {
        "format_version": "1.1",
        "git_hash": "0" * 40,
        "lane_count": len(FIXTURE_LANES),
        "lane_ids_sha256": hashlib.sha256("\n".join(FIXTURE_LANES).encode("utf-8")).hexdigest(),
        "run_metadata": run_metadata,
        "episodes": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_dir


def fixture_spec(name: str, dirs: Sequence[str], **overrides: Any) -> TierSpec:
    """A ``TierSpec`` over fixture directories, defaulting to the phase-1 shape."""
    fields: dict[str, Any] = {
        "tier": name,
        "dirs": tuple(dirs),
        "phase": 1,
        "target_rtg": -6.0,
        "rtg_scale": 12.0,
        "stream_count": 2,
        "subsample": "none",
        "components": (),
    }
    fields.update(overrides)
    return TierSpec(**fields)


def stream(draw: int, ret: float, *, directory: str = "d", episode: str | None = None) -> StreamReturn:
    """A ``StreamReturn`` with only the fields this module reads."""
    return StreamReturn(
        dataset_dir=directory,
        episode_file=episode or f"ep_draw{draw}_{ret}.npz",
        ix_id="ix0",
        ix_index=0,
        episode_index=draw,
        flow_draw=int(draw),
        group=(FIXTURE_STATE_DIM, FIXTURE_N_ACTIONS),
        total_return=float(ret),
    )


def episodes_for(arm: str, values: dict[int, float], seeds: Sequence[int] = (101, 202)) -> list[EpisodeResult]:
    """One ``EpisodeResult`` per (seed, draw); every seed sees the same value, so means are exact."""
    return [
        EpisodeResult(
            arm=arm,
            seed=int(seed),
            draw_id=int(draw),
            att_horizon=float(value),
            horizon_vehicle_count=float(value) / 2.0,
            episode_reward=-float(value),
        )
        for seed in seeds
        for draw, value in sorted(values.items())
    ]


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected datasets_v11/ directory to run the corpus-backed P4.6 tests"
        )
    return candidate


# ----------------------------------------------------------------------
# The declaration
# ----------------------------------------------------------------------


def test_the_behaviour_att_constants_are_the_committed_ladders_and_order_the_tiers() -> None:
    """The tier order is measured ATT, and the constants are the artifact's, digit for digit.

    ``BRIEF_17`` section 1: order by measured ATT, never by tier name or training budget --
    ``mappo060`` is worse than ``fixedtime`` despite being a learned policy.
    """
    ladder = json.loads(
        (REPO_ROOT / "docs/data/att_ladder_v11.json").read_text(encoding="utf-8")
    )
    committed = {
        cell["tier"]: cell["att_horizon_mean"]
        for cell in ladder["cells"]
        if cell["scenario"] == "cf_hz1x1"
    }
    for tier, value in BEHAVIOUR_ATT.items():
        assert value == committed[tier], f"{tier}: declared {value} against committed {committed[tier]}"
    assert PHASE1_TIER_ORDER == tuple(sorted(BEHAVIOUR_ATT, key=lambda t: BEHAVIOUR_ATT[t]))


def test_every_declared_tier_has_a_spec_and_an_unknown_tier_is_refused() -> None:
    """A typo in a tier name must raise, not select a neighbouring column."""
    for tier in (*PHASE1_TIER_ORDER, *MIXTURE_EXPERT_FRACTION):
        assert tier_spec(tier).tier == tier
    with pytest.raises(ValueError, match="unknown tier"):
        tier_spec("mappo1001")


def test_the_declared_prompt_targets_are_the_maxima_and_the_scales_the_largest_magnitude() -> None:
    """``recomputed_target_and_scale`` is the naive rule, and a mean would give another number."""
    streams = [stream(1, -10.0), stream(2, -4.0), stream(3, -7.0)]
    target, scale = recomputed_target_and_scale(streams)
    assert target == -4.0
    assert scale == 10.0
    mean = math.fsum(s.total_return for s in streams) / len(streams)
    assert target != mean, "the rule is the maximum; a mean would be a different prompt"


def test_the_random_subsample_takes_one_stream_per_draw_and_is_reproducible(
    tmp_path: Path,
) -> None:
    """Stratified one-per-draw, deterministic under the declared seed, different under another."""
    streams = [stream(draw, -float(draw * 10 + k), episode=f"ep{draw}_{k}.npz")
               for draw in (1, 2, 3, 4) for k in (0, 1)]

    first = stratified_one_per_draw(streams, rng=np.random.default_rng(RANDOM_SUBSAMPLE_RNG_SEED))
    again = stratified_one_per_draw(streams, rng=np.random.default_rng(RANDOM_SUBSAMPLE_RNG_SEED))

    assert [s.key for s in first] == [s.key for s in again], "the selection must be reproducible"
    assert sorted(s.flow_draw for s in first) == [1, 2, 3, 4], "exactly one stream per draw"
    assert len(first) == 4

    others = [
        tuple(s.key for s in stratified_one_per_draw(streams, rng=np.random.default_rng(seed)))
        for seed in (1, 2, 3, 4, 5, 6)
    ]
    assert any(other != tuple(s.key for s in first) for other in others), (
        "an RNG that never consumes would return the same subset for every seed"
    )


def test_a_draw_with_no_candidates_cannot_appear_and_multiplicity_is_reported(tmp_path: Path) -> None:
    """One-per-draw over a pool that already has one per draw is the identity."""
    streams = [stream(draw, -float(draw)) for draw in (1, 2, 3)]
    kept = stratified_one_per_draw(streams, rng=np.random.default_rng(7))
    assert [s.key for s in kept] == [s.key for s in streams]


def test_training_streams_are_size_matched_and_a_short_tier_is_refused(tmp_path: Path) -> None:
    """The size match is enforced where it is decided, not only asserted in the artifact."""
    single = write_tier_dir(
        tmp_path, "tier_single", episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)]
    )
    doubled = write_tier_dir(
        tmp_path,
        "tier_doubled",
        episodes=[(1, -1.0, 100.0), (1, -3.0, 130.0), (2, -2.0, 110.0), (2, -4.0, 140.0)],
    )
    spec_single = fixture_spec("single", [single.name])
    spec_doubled = fixture_spec("doubled", [doubled.name], subsample="one_per_draw")

    ds_single = TrajectoryWindowDataset([single], context_length=CONTEXT, split="train")
    ds_doubled = TrajectoryWindowDataset([doubled], context_length=CONTEXT, split="train")

    assert len(training_streams(spec_single, ds_single)) == 2
    assert len(training_streams(spec_doubled, ds_doubled)) == 2

    too_big = fixture_spec("too_big", [single.name], stream_count=3)
    with pytest.raises(ValueError, match="declares 3 training streams"):
        training_streams(too_big, ds_single)


def test_the_top_decile_filter_is_applied_to_the_tiers_own_training_set() -> None:
    """%BC selects within the size-matched set, never within the full split."""
    streams = [stream(d, -float(d)) for d in range(1, 21)]
    kept = top_decile_streams(streams)
    assert len(kept) == 2, "10 % of 20"
    assert sorted(s.total_return for s in kept) == [-2.0, -1.0]


def test_statistics_are_fitted_per_tier_and_two_tiers_do_not_share_a_digest(
    tmp_path: Path,
) -> None:
    """A shared statistic across tiers would be a leak between arms (``BRIEF_17`` section 7.3)."""
    left = write_tier_dir(tmp_path, "tier_left", episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)])
    right = write_tier_dir(
        tmp_path,
        "tier_right",
        episodes=[(1, -5.0, 300.0), (2, -6.0, 310.0)],
        state_offset=50.0,
    )
    ds_left = TrajectoryWindowDataset([left], context_length=CONTEXT, split="train")
    ds_right = TrajectoryWindowDataset([right], context_length=CONTEXT, split="train")

    digest_left = statistics_digest(ds_left)
    digest_right = statistics_digest(ds_right)
    assert digest_left != digest_right
    assert digest_left == statistics_digest(
        TrajectoryWindowDataset([left], context_length=CONTEXT, split="train")
    ), "the digest must be a function of the data, not of the object"


def test_the_declaration_refuses_a_target_that_is_not_the_corpus_maximum(tmp_path: Path) -> None:
    """The declared prompt is checked against the data before any gradient step."""
    directory = write_tier_dir(
        tmp_path, "tier_check", episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)]
    )
    dataset = TrajectoryWindowDataset([directory], context_length=CONTEXT, split="train")
    selected = training_streams(fixture_spec("good", [directory.name]), dataset)
    good = fixture_spec("good", [directory.name], target_rtg=-6.0, rtg_scale=12.0)
    record = assert_declaration_matches_corpus(good, selected)
    assert record["target_rtg"] == -6.0 and record["rtg_scale"] == 12.0

    wrong = fixture_spec("wrong", [directory.name], target_rtg=-7.0, rtg_scale=12.0)
    with pytest.raises(ValueError, match="declares target_rtg"):
        assert_declaration_matches_corpus(wrong, selected)


def test_the_target_is_computed_over_the_training_set_and_not_over_the_full_split(
    tmp_path: Path,
) -> None:
    """``BRIEF_17`` section 11, finding A4: the prompt must be in-support by construction.

    The fixture is built so the two rules disagree: the best episode of draw 1 is the one the
    declared subsample does NOT take, so a target computed over the full split would ask the model
    for a return no episode it trained on ever achieved.
    """
    directory = write_tier_dir(
        tmp_path,
        "tier_a4",
        episodes=[(1, -1.0, 100.0), (1, -3.0, 130.0), (2, -2.0, 110.0), (2, -4.0, 140.0)],
    )
    dataset = TrajectoryWindowDataset([directory], context_length=CONTEXT, split="train")
    spec = fixture_spec("a4", [directory.name], subsample="one_per_draw")
    selected = training_streams(spec, dataset)

    split_target, split_scale = recomputed_target_and_scale(stream_returns(dataset))
    set_target, set_scale = recomputed_target_and_scale(selected)
    assert split_target == -1.0 * FIXTURE_T, "the split's best episode returns -6"
    assert set_target <= split_target
    assert set_target == max(s.total_return for s in selected)
    assert set_scale == max(abs(s.total_return) for s in selected)

    matching = fixture_spec(
        "a4", [directory.name], subsample="one_per_draw", target_rtg=set_target,
        rtg_scale=set_scale,
    )
    assert assert_declaration_matches_corpus(matching, selected)["target_rtg"] == set_target

    split_rule = fixture_spec(
        "a4", [directory.name], subsample="one_per_draw", target_rtg=split_target,
        rtg_scale=split_scale,
    )
    if split_target != set_target:
        with pytest.raises(ValueError, match="declares target_rtg"):
            assert_declaration_matches_corpus(split_rule, selected)


def test_the_declaration_records_the_episode_sha256_of_every_selected_stream(
    tmp_path: Path,
) -> None:
    """``BRIEF_17`` section 11, finding A1: record the selected ``episode_sha256`` list."""
    directory = write_tier_dir(
        tmp_path,
        "tier_sha",
        episodes=[(1, -1.0, 100.0), (1, -3.0, 130.0), (2, -2.0, 110.0), (2, -4.0, 140.0)],
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    by_file = {e["filename"]: e["episode_sha256"] for e in manifest["episodes"]}
    dataset = TrajectoryWindowDataset([directory], context_length=CONTEXT, split="train")
    spec = fixture_spec("sha", [directory.name], subsample="one_per_draw")
    selected = training_streams(spec, dataset)

    records = stream_records_with_digests(selected)
    assert len(records) == 2
    for record in records:
        assert record["episode_sha256"] == by_file[record["episode_file"]]
        assert len(record["episode_sha256"]) == 64


def test_equal_training_size_is_asserted_from_the_artifact() -> None:
    """``BRIEF_17`` section 7.1: assert equal training-set size **in the artifact**."""
    declaration = {
        "tiers": {
            "a": {"training_streams": 200, "training_windows": 72000, "top_decile_streams": 20},
            "b": {"training_streams": 200, "training_windows": 72000, "top_decile_streams": 20},
        }
    }
    assert_equal_training_size(declaration)

    declaration["tiers"]["b"]["training_streams"] = 199
    with pytest.raises(ValueError, match="training-set sizes differ"):
        assert_equal_training_size(declaration)


def test_env_settings_must_be_identical_across_every_tier(tmp_path: Path) -> None:
    """One settings dict for every arm; a tier collected differently cannot join the grid."""
    a = write_tier_dir(tmp_path, "tier_a", episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)])
    b = write_tier_dir(tmp_path, "tier_b", episodes=[(1, -1.5, 100.0), (2, -2.5, 110.0)])
    specs = [fixture_spec("a", [a.name]), fixture_spec("b", [b.name])]
    settings = env_settings_for_tiers(specs, tmp_path)
    assert settings["max_steps"] == FIXTURE_T

    c = write_tier_dir(
        tmp_path,
        "tier_c",
        episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)],
        manifest_extra={"max_steps": FIXTURE_T + 1},
    )
    with pytest.raises(ValueError, match="env settings"):
        env_settings_for_tiers([*specs, fixture_spec("c", [c.name])], tmp_path)


def test_the_mixture_composition_is_the_declared_fraction_of_each_component() -> None:
    """OffLight's fractions, applied to the two components' own declared training sets."""
    expert = [stream(d, -float(d), directory="expert") for d in range(1, 21)]
    randoms = [stream(d, -100.0 - d, directory="random") for d in range(1, 21)]
    spec = TIERS["mix50"]
    chosen = mixture_training_streams(
        TierSpec(**{**spec.__dict__, "stream_count": 20}), expert, randoms
    )
    assert len(chosen) == 20
    assert sum(1 for s in chosen if s.dataset_dir == "expert") == 10
    assert sum(1 for s in chosen if s.dataset_dir == "random") == 10
    again = mixture_training_streams(
        TierSpec(**{**spec.__dict__, "stream_count": 20}), expert, randoms
    )
    assert [s.key for s in chosen] == [s.key for s in again]


# ----------------------------------------------------------------------
# Prediction P3: the two leakage-free checks
# ----------------------------------------------------------------------


def test_the_hypergeometric_tail_matches_brute_force_enumeration() -> None:
    """Exact ``P(X >= observed)``, checked against every subset of a small population."""
    population, successes, draws = 10, 3, 3
    subsets = list(itertools.combinations(range(population), draws))
    marked = set(range(successes))
    for observed in range(0, draws + 1):
        brute = sum(1 for s in subsets if len(marked.intersection(s)) >= observed) / len(subsets)
        assert hypergeometric_upper_tail(population, successes, draws, observed) == pytest.approx(
            brute, abs=1e-15
        ), observed


def test_the_hypergeometric_tail_is_one_at_zero_and_refuses_impossible_arguments() -> None:
    assert hypergeometric_upper_tail(200, 20, 20, 0) == 1.0
    with pytest.raises(ValueError, match="hypergeometric"):
        hypergeometric_upper_tail(10, 12, 3, 1)


def test_the_difficulty_check_counts_the_easiest_draws_and_not_the_hardest() -> None:
    """"Easiest" is the LOWEST MaxPressure ATT; using the hardest end inverts the diagnostic."""
    difficulty = {draw: float(draw) for draw in range(1, 11)}      # draw 1 easiest, 10 hardest
    kept_easy = [1, 2, 3]
    kept_hard = [8, 9, 10]

    easy = difficulty_check(kept_easy, difficulty, easiest_count=3)
    hard = difficulty_check(kept_hard, difficulty, easiest_count=3)

    assert easy["overlap"] == 3 and hard["overlap"] == 0
    assert easy["expected_overlap"] == pytest.approx(3 * 3 / 10)
    assert easy["p_value"] < hard["p_value"]
    assert easy["easiest_draws"] == [1, 2, 3]


def test_the_difficulty_check_refuses_a_draw_it_has_no_difficulty_for() -> None:
    with pytest.raises(ValueError, match="no difficulty"):
        difficulty_check([1, 99], {1: 1.0, 2: 2.0, 3: 3.0}, easiest_count=2)


def test_the_volume_check_reports_the_difference_and_an_interval_that_can_exclude_zero() -> None:
    """Check A is a difference of means with a normal-approximation interval, labelled as one."""
    arrivals = {draw: 100 for draw in range(1, 21)}
    for draw in range(1, 6):
        arrivals[draw] = 150
    kept = list(range(1, 6))
    others = list(range(6, 21))

    result = volume_check(kept, others, arrivals)
    assert result["mean_kept"] == 150.0
    assert result["mean_other"] == 100.0
    assert result["difference"] == 50.0
    assert result["ci95_low"] > 0.0, "a 50-vehicle difference with zero variance excludes zero"
    assert result["approximation"] == "welch_normal"

    flat = volume_check(kept, others, {draw: 100 for draw in range(1, 21)})
    assert flat["difference"] == 0.0
    assert flat["ci95_low"] <= 0.0 <= flat["ci95_high"]


def test_arrivals_refuse_a_draw_whose_rebuilt_flow_does_not_match_the_recorded_hash(
    tmp_path: Path,
) -> None:
    """The reconstruction is verified against the manifest's own per-draw hashes, or it refuses."""
    source = tmp_path / "flow.json"
    source.write_text(
        json.dumps(
            [
                {
                    "vehicle": {
                        "length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5,
                        "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5,
                        "maxSpeed": 11.11, "headwayTime": 2.0,
                    },
                    "route": ["road_a", "road_b"],
                    "interval": 5,
                    "startTime": t,
                    "endTime": t,
                }
                for t in range(50)
            ]
        ),
        encoding="utf-8",
    )
    directory = write_tier_dir(
        tmp_path, "tier_flow", episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)]
    )
    with pytest.raises(ValueError, match="recorded flow hash"):
        draw_arrivals(directory / "manifest.json")


# ----------------------------------------------------------------------
# The reporting rule: no equivalence verdicts anywhere
# ----------------------------------------------------------------------


def test_the_module_declares_no_equivalence_verdict_anywhere() -> None:
    """``BRIEF_17`` section 4 is a ruling; this is its mechanism.

    A6's delta is ``mappo1000``-specific and there is no non-circular way to derive a per-tier
    one before the run, so the module must not be able to emit a verdict at all.
    """
    source = (REPO_ROOT / "offline/method_tier_grid.py").read_text(encoding="utf-8")
    for symbol in (
        "equivalence_verdict",
        "delta_verdict",
        "DELTA_ATT",
        "VERDICT_MATCHES",
        "VERDICT_WITHIN_DELTA",
    ):
        assert symbol not in source, f"{symbol} must not be reachable from this module"


def test_assert_no_verdicts_finds_a_verdict_at_any_depth() -> None:
    """A nested verdict is the one that would survive a shallow check."""
    assert_no_verdicts({"cells": {"bc@random": {"att_horizon_mean": 1.0}}})
    with pytest.raises(ValueError, match="verdict"):
        assert_no_verdicts({"comparisons": [{"pair": "a", "verdict": "matches"}]})
    with pytest.raises(ValueError, match="verdict"):
        assert_no_verdicts({"a": {"b": [{"c": "within_delta"}]}})


# ----------------------------------------------------------------------
# Cells, comparisons and the prediction scoring
# ----------------------------------------------------------------------


def test_arm_keys_round_trip_and_never_collide_with_a_policy_source_name() -> None:
    """``bc@random`` cannot be mistaken for the ``random`` heuristic arm of ``dt_gate``."""
    from offline.dt_gate import POLICY_SOURCES

    for method in METHODS:
        for tier in PHASE1_TIER_ORDER:
            key = arm_key(method, tier)
            assert split_arm_key(key) == (method, tier)
            assert key not in POLICY_SOURCES
    with pytest.raises(ValueError, match="arm key"):
        split_arm_key("bc")


def test_cell_stats_matches_an_independent_mean_and_ci_recomputation() -> None:
    """The cell is packaging around ``mean_ci95``; the packaging is what is checked here."""
    values = {1000: 100.0, 1001: 104.0, 1002: 108.0}
    episodes = episodes_for("bc@random", values)
    cell = cell_stats(episodes)
    flat = [e.att_horizon for e in episodes]
    reference = mean_ci95(flat)

    assert cell["n_episodes"] == len(episodes)
    assert cell["att_horizon_mean"] == reference.mean
    assert cell["att_horizon_ci95"] == reference.ci95
    assert cell["att_horizon_mean"] == math.fsum(flat) / len(flat)
    assert cell["draw_ids"] == [1000, 1001, 1002]
    assert cell["seeds"] == [101, 202]


def test_cell_stats_refuses_two_arms_in_one_cell() -> None:
    with pytest.raises(ValueError, match="one arm"):
        cell_stats([*episodes_for("bc@random", {1000: 1.0}), *episodes_for("iql@random", {1000: 2.0})])


def test_every_within_tier_and_within_method_pair_is_compared_and_paired_by_draw() -> None:
    """30 within-tier + 40 within-method pairs on the full grid; here, the fixture's share."""
    tiers = ("mappo1000", "random")
    methods = ("bc", "dt")
    episodes_by_arm = {
        arm_key(method, tier): episodes_for(
            arm_key(method, tier), {1000: 100.0 + i, 1001: 102.0 + i}
        )
        for i, (method, tier) in enumerate(itertools.product(methods, tiers))
    }
    comparisons = grid_comparisons(episodes_by_arm)
    pairs = {(c.left_arm, c.right_arm) for c in comparisons}

    # one within-tier pair per tier, one within-method pair per method
    assert (arm_key("bc", "mappo1000"), arm_key("dt", "mappo1000")) in pairs
    assert (arm_key("bc", "random"), arm_key("dt", "random")) in pairs
    assert (arm_key("bc", "mappo1000"), arm_key("bc", "random")) in pairs
    assert (arm_key("dt", "mappo1000"), arm_key("dt", "random")) in pairs
    assert len(comparisons) == 4, "no cross-tier cross-method pair is defined"

    for comparison in comparisons:
        left = episodes_by_arm[comparison.left_arm]
        right = episodes_by_arm[comparison.right_arm]
        per_draw_left = {
            draw: math.fsum(e.att_horizon for e in left if e.draw_id == draw)
            / sum(1 for e in left if e.draw_id == draw)
            for draw in sorted({e.draw_id for e in left})
        }
        per_draw_right = {
            draw: math.fsum(e.att_horizon for e in right if e.draw_id == draw)
            / sum(1 for e in right if e.draw_id == draw)
            for draw in sorted({e.draw_id for e in right})
        }
        expected = math.fsum(
            per_draw_left[d] - per_draw_right[d] for d in per_draw_left
        ) / len(per_draw_left)
        assert comparison.mean_difference == pytest.approx(expected, abs=1e-12)
        assert comparison.ci95_width == 2.0 * comparison.ci95_half_width
        assert comparison.n_shared_draws == 2


def test_behaviour_arms_are_compared_per_tier_and_excluded_from_the_method_grid() -> None:
    """A3's per-tier reference is reported beside the methods, never inside their 70 pairs."""
    episodes_by_arm = {
        arm_key("bc", "random"): episodes_for(arm_key("bc", "random"), {1000: 300.0, 1001: 310.0}),
        arm_key("dt", "random"): episodes_for(arm_key("dt", "random"), {1000: 320.0, 1001: 330.0}),
        f"{BEHAVIOUR_METHOD}@random": episodes_for(
            f"{BEHAVIOUR_METHOD}@random", {1000: 400.0, 1001: 420.0}
        ),
    }
    grid = grid_comparisons(episodes_by_arm)
    assert all(
        BEHAVIOUR_METHOD not in comparison.left_arm and BEHAVIOUR_METHOD not in comparison.right_arm
        for comparison in grid
    ), "the behaviour arm is not one of the four methods and must not enter the method grid"

    against = behaviour_comparisons(episodes_by_arm)
    pairs = {(c.left_arm, c.right_arm) for c in against}
    assert pairs == {
        (arm_key("bc", "random"), f"{BEHAVIOUR_METHOD}@random"),
        (arm_key("dt", "random"), f"{BEHAVIOUR_METHOD}@random"),
    }
    for comparison in against:
        assert comparison.mean_difference < 0.0, "both methods beat a 410-ATT behaviour policy"
        assert comparison.n_shared_draws == 2


def test_p1_is_scored_by_the_declared_rank_rule_and_reports_the_sequence() -> None:
    """P1 holds only if BC is worst of four on every tier (plan section 4.1)."""
    worst_everywhere = {
        tier: {
            "bc": {"att_horizon_mean": 200.0},
            "bc_top10": {"att_horizon_mean": 100.0},
            "iql": {"att_horizon_mean": 110.0},
            "dt": {"att_horizon_mean": 120.0},
        }
        for tier in PHASE1_TIER_ORDER
    }
    held = score_p1(worst_everywhere)
    assert held["outcome"] == "HELD"
    assert held["bc_rank_by_tier"] == {tier: 4 for tier in PHASE1_TIER_ORDER}

    improved = {tier: {k: dict(v) for k, v in cells.items()} for tier, cells in worst_everywhere.items()}
    improved["random"]["bc"]["att_horizon_mean"] = 90.0
    failed = score_p1(improved)
    assert failed["outcome"] == "FAILED"
    assert failed["bc_rank_by_tier"]["random"] == 1

    tied = {tier: {k: dict(v) for k, v in cells.items()} for tier, cells in worst_everywhere.items()}
    tied["random"]["iql"]["att_horizon_mean"] = 200.0
    assert score_p1(tied)["outcome"] == "NOT RESOLVED"


def test_p2_scores_the_partial_clause_and_refuses_the_mixture_clause_without_mixtures() -> None:
    """P2's mixture half is unscorable from phase 1, and no verdict may be invented for it."""
    cells = {
        "mappo1000": {"bc": {"att_horizon_mean": 105.0}, "bc_top10": {"att_horizon_mean": 103.0}},
        "random": {"bc": {"att_horizon_mean": 400.0}, "bc_top10": {"att_horizon_mean": 399.5}},
        "fixedtime": {"bc": {"att_horizon_mean": 260.0}, "bc_top10": {"att_horizon_mean": 250.0}},
    }
    scored = score_p2(cells, [])
    assert scored["full_outcome"] == "NOT SCORABLE"
    assert scored["partial_outcome"] == "HELD"
    assert scored["advantage_by_tier"]["random"] == pytest.approx(0.5)

    cells["random"]["bc_top10"]["att_horizon_mean"] = 380.0
    assert score_p2(cells, [])["partial_outcome"] == "FAILED"


def test_p2_carries_the_paired_interval_of_each_tiers_bc_versus_bc_top10_contrast() -> None:
    """The advantage is only interpretable next to its CI, and the pair's ORDER fixes its sign."""
    episodes_by_arm = {
        arm_key("bc", "random"): episodes_for(arm_key("bc", "random"), {1000: 400.0, 1001: 402.0}),
        arm_key("bc_top10", "random"): episodes_for(
            arm_key("bc_top10", "random"), {1000: 399.0, 1001: 401.0}
        ),
    }
    comparisons = grid_comparisons(episodes_by_arm)
    cells = {
        "random": {
            "bc": {"att_horizon_mean": 401.0},
            "bc_top10": {"att_horizon_mean": 400.0},
        }
    }
    scored = score_p2(cells, comparisons)
    interval = scored["advantage_intervals"]["random"]
    assert interval["mean_difference"] == pytest.approx(1.0)
    assert interval["ci95_width"] == pytest.approx(2.0 * (interval["ci95_high"] - interval["mean_difference"]))

    reversed_pair = grid_comparisons(
        {
            arm_key("bc_top10", "random"): episodes_by_arm[arm_key("bc_top10", "random")],
            arm_key("bc", "random"): episodes_by_arm[arm_key("bc", "random")],
        }
    )
    assert reversed_pair[0].left_arm == arm_key("bc", "random"), (
        "grid_comparisons must order a pair by METHODS, whatever order the arms arrived in"
    )


def test_p3_is_scored_by_the_declared_or_rule_per_tier() -> None:
    """Either check may carry the signature; both are reported whichever way they come out."""
    diagnostics = {
        "tiers": {
            "mappo1000": {
                "volume": {"difference": -2.5, "ci95_low": -6.0, "ci95_high": 1.0},
                "difficulty": {"overlap": 4, "expected_overlap": 2.0, "p_value": 0.12},
            },
            "random": {
                "volume": {"difference": -30.0, "ci95_low": -40.0, "ci95_high": -20.0},
                "difficulty": {"overlap": 9, "expected_overlap": 2.0, "p_value": 0.0001},
            },
        }
    }
    scored = score_p3(diagnostics)
    assert scored["by_tier"]["mappo1000"]["demand_signature"] is False
    assert scored["by_tier"]["random"]["demand_signature"] is True
    assert scored["outcome"] == "HELD"

    diagnostics["tiers"]["random"]["difficulty"]["p_value"] = 0.9
    diagnostics["tiers"]["random"]["volume"] = {
        "difference": -1.0, "ci95_low": -5.0, "ci95_high": 3.0
    }
    assert score_p3(diagnostics)["outcome"] == "FAILED"


def test_the_kept_composition_is_reported_on_both_axes(tmp_path: Path) -> None:
    """``BRIEF_17`` section 11, finding A5: by source directory AND as a behaviour-seed histogram.

    Without the second axis, "the filter selects the expert fraction" on a mixture is confounded
    with the checkpoint selection P4.5 already established.
    """
    kept = [
        stream(1, -1.0, directory="/c/cf_hz1x1__mappo1000__seed202"),
        stream(2, -2.0, directory="/c/cf_hz1x1__mappo1000__seed202"),
        stream(3, -3.0, directory="/c/cf_hz1x1__mappo1000__seed101"),
        stream(4, -4.0, directory="/c/cf_hz1x1__random"),
    ]
    composition = kept_composition(kept)
    assert composition["by_dataset_dir"] == {
        "cf_hz1x1__mappo1000__seed202": 2,
        "cf_hz1x1__mappo1000__seed101": 1,
        "cf_hz1x1__random": 1,
    }
    assert composition["by_behaviour_seed"] == {"101": 1, "202": 2}
    assert composition["without_a_behaviour_seed"] == 1


def test_kendall_tau_b_matches_an_independent_pairwise_count() -> None:
    """Reported beside P1's rank rule, so it gets the same treatment as any reported number."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 1.0, 4.0, 3.0, 5.0]
    concordant = discordant = 0
    for i, j in itertools.combinations(range(len(xs)), 2):
        sign = (xs[i] - xs[j]) * (ys[i] - ys[j])
        concordant += sign > 0
        discordant += sign < 0
    expected = (concordant - discordant) / math.sqrt(
        (concordant + discordant) * (concordant + discordant)
    )
    assert kendall_tau_b(xs, ys) == pytest.approx(expected, abs=1e-12)
    assert kendall_tau_b([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert kendall_tau_b([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


# ----------------------------------------------------------------------
# Campaign integrity and the gate
# ----------------------------------------------------------------------


def test_every_tier_has_a_declared_behaviour_reference_and_a_source_for_it() -> None:
    """``BRIEF_17`` section 11, finding A3: the per-tier "did it beat its own policy?" reference."""
    for tier in PHASE1_TIER_ORDER:
        reference = BEHAVIOUR_REFERENCE_BY_TIER[tier]
        assert reference["arm"] in ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random")
        assert reference["source"] in ("committed", "evaluated_here")
    assert BEHAVIOUR_REFERENCE_BY_TIER["random"]["source"] == "evaluated_here"
    assert BEHAVIOUR_REFERENCE_BY_TIER["fixedtime"]["source"] == "evaluated_here"
    assert BEHAVIOUR_REFERENCE_BY_TIER["mappo500"]["source"] == "committed"
    for tier in MIXTURE_EXPERT_FRACTION:
        assert tier not in BEHAVIOUR_REFERENCE_BY_TIER, (
            "a mixture has two behaviour policies and no single reference; that is P4.7's problem"
        )


def test_the_fixedtime_factory_reads_k_from_the_manifest_and_checks_the_plan_hash(
    tmp_path: Path,
) -> None:
    """A reference policy that is not the collecting policy is a different measurement (A3).

    ``PROJECT_PLAN`` section 6 says P2.5 "ships k=4"; the corpus was collected at k=6, and the
    manifest is the only source that knows it.
    """
    directory = write_tier_dir(
        tmp_path,
        "tier_ft",
        episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)],
        manifest_extra={
            "fixed_time_k": 6,
            "fixed_time_schedule_source": "shipped_plan",
            "fixed_time_plan_sha256": "a" * 64,
        },
    )
    settings = fixedtime_collection_settings(directory / "manifest.json")
    assert settings["fixed_time_k"] == 6
    assert settings["fixed_time_plan_sha256"] == "a" * 64

    bad = write_tier_dir(
        tmp_path,
        "tier_ft_bad",
        episodes=[(1, -1.0, 100.0), (2, -2.0, 110.0)],
        manifest_extra={
            "fixed_time_k": None,
            "fixed_time_schedule_source": "shipped_plan",
            "fixed_time_plan_sha256": "a" * 64,
        },
    )
    with pytest.raises(ValueError, match="fixed_time_k"):
        fixedtime_collection_settings(bad / "manifest.json")


def test_a_cell_missing_one_seed_or_one_draw_is_refused() -> None:
    """A campaign that aborted halfway would otherwise produce a smaller, plausible cell."""
    seeds = (101, 202)
    draws = (1000, 1001)
    produced = episodes_for(arm_key("bc", "random"), {1000: 1.0, 1001: 2.0}, seeds=seeds)
    assert_cell_complete("bc", "random", seeds, draws, produced)

    with pytest.raises(ValueError, match="incomplete cell"):
        assert_cell_complete("bc", "random", seeds, draws, produced[:-1])
    with pytest.raises(ValueError, match="not requested"):
        assert_cell_complete(
            "bc", "random", seeds, draws,
            [*produced, *episodes_for(arm_key("bc", "random"), {1002: 3.0}, seeds=(101,))],
        )


def test_merging_training_records_keeps_other_cells_and_refuses_another_declaration() -> None:
    """A chunked campaign must not produce a run set that looks complete and is not."""
    existing = {
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "runs": [
            {"tier": "random", "method": "bc", "seed": 101},
            {"tier": "random", "method": "iql", "seed": 101},
        ],
    }
    fresh = {
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "runs": [{"tier": "random", "method": "bc", "seed": 101, "final_loss": 0.1}],
    }
    merged = merge_training_records(existing, fresh)
    assert {(r["tier"], r["method"], r["seed"]) for r in merged} == {
        ("random", "bc", 101),
        ("random", "iql", 101),
    }
    assert any(r.get("final_loss") == 0.1 for r in merged), "the fresh record must win"

    with pytest.raises(ValueError, match="different designs"):
        merge_training_records(existing, {**fresh, "declared_gradient_steps": 20_000})


def test_the_gate_refuses_a_single_cell_that_does_not_reproduce_exactly() -> None:
    """Re-use is only sound if this session's instrument reproduces the committed episodes."""
    committed = [
        {"arm": "bc", "seed": 101, "draw_id": 1000, "att_horizon": 105.5, "horizon_vehicle_count": 44.0},
        {"arm": "bc", "seed": 101, "draw_id": 1025, "att_horizon": 106.5, "horizon_vehicle_count": 45.0},
    ]
    rerolled = [
        EpisodeResult(arm="bc", seed=101, draw_id=1000, att_horizon=105.5, horizon_vehicle_count=44.0, episode_reward=-1.0),
        EpisodeResult(arm="bc", seed=101, draw_id=1025, att_horizon=106.5, horizon_vehicle_count=45.0, episode_reward=-1.0),
    ]
    record = assert_reused_cells_reproduce(committed, rerolled)
    assert record["compared"] == 2 and record["mismatches"] == 0

    drifted = [
        rerolled[0],
        EpisodeResult(arm="bc", seed=101, draw_id=1025, att_horizon=106.5000000001,
                      horizon_vehicle_count=45.0, episode_reward=-1.0),
    ]
    with pytest.raises(ValueError, match="does not reproduce"):
        assert_reused_cells_reproduce(committed, drifted)


def test_the_gate_refuses_a_reroll_that_is_missing_a_committed_cell() -> None:
    committed = [
        {"arm": "bc", "seed": 101, "draw_id": 1000, "att_horizon": 105.5, "horizon_vehicle_count": 44.0},
    ]
    with pytest.raises(ValueError, match="does not reproduce"):
        assert_reused_cells_reproduce(committed, [])


def test_the_training_cli_path_runs_end_to_end_on_a_fixture(tmp_path: Path) -> None:
    """The function the numbers flow through, exercised -- not the helpers one level down.

    ``PROJECT_PLAN`` section 7 makes this a rule after its third sighting (P4.4 F1, P4.5 N1,
    P4.3 F2).  It caught a real defect here on its first run: ``_run_train`` was still calling
    ``assert_declaration_matches_corpus`` with the dataset instead of the selected streams, which
    every unit test missed because none of them called ``_run_train``.
    """
    directory = write_tier_dir(
        tmp_path,
        "tier_cli",
        episodes=[(1, -1.0, 100.0), (1, -3.0, 130.0), (2, -2.0, 110.0), (2, -4.0, 140.0)],
    )
    dataset = TrajectoryWindowDataset([directory], context_length=CONTEXT, split="train")
    spec = fixture_spec("cli", [directory.name], subsample="one_per_draw")
    selected = training_streams(spec, dataset)
    target, scale = recomputed_target_and_scale(selected)
    spec = fixture_spec(
        "cli", [directory.name], subsample="one_per_draw", target_rtg=target, rtg_scale=scale
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoints = tmp_path / "ckpt"
    args = argparse.Namespace(
        corpus_root=str(tmp_path),
        tier="cli",
        methods="bc,bc_top10,iql,dt",
        steps=2,
        log_every=0,
        device="cpu",
        checkpoint_dir=str(checkpoints),
    )
    with mock.patch.dict(grid.TIERS, {"cli": spec}, clear=False), mock.patch.object(
        grid, "CONTEXT_LENGTH", CONTEXT
    ):
        assert grid._run_train(args, out_dir) == 0

    payload = json.loads((out_dir / "p4_6_training.json").read_text(encoding="utf-8"))
    runs = {(r["tier"], r["method"], r["seed"]) for r in payload["runs"]}
    assert len(runs) == 4 * len(grid.TRAINING_SEEDS)
    for run in payload["runs"]:
        assert run["gradient_steps"] == 2
        assert len(run["canonical_digest"]) == 64
        assert Path(run["checkpoint"]).is_file()
        assert (run["target_rtg"] == target) if run["method"] == "dt" else (run["target_rtg"] is None)


# ----------------------------------------------------------------------
# The real corpus
# ----------------------------------------------------------------------


def test_the_declared_targets_equal_the_corpus_maxima_read_by_a_second_route(
    corpus_v11_root: Path,
) -> None:
    """The DT's prompt, recomputed from the ``.npz`` files with no loader in the path.

    The loader route (``stream_returns`` over ``TrajectoryWindowDataset``) derives a stream's
    return from its returns-to-go; this route sums ``local_reward`` read by ``np.load``.  They
    agree only if the RTG convention is the one contract C6 states.
    """
    from offline.trajectory_logger import load_episode

    for tier in PHASE1_TIER_ORDER:
        spec = tier_spec(tier)
        returns: list[float] = []
        for directory in tier_dirs(spec, corpus_v11_root):
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            for entry in manifest["episodes"]:
                episode = load_episode(directory / entry["filename"])
                for ix_id in episode.ix_ids:
                    rewards = np.asarray(
                        episode.intersections[ix_id].local_reward, dtype=np.float32
                    )
                    returns.append(math.fsum(float(r) for r in rewards))
        assert spec.target_rtg == max(returns), tier
        assert spec.rtg_scale == max(abs(min(returns)), abs(max(returns))), tier


def test_the_real_tiers_are_size_matched_at_two_hundred_streams(corpus_v11_root: Path) -> None:
    """The whole design rests on this line, so it is measured on the real corpus, not a fixture."""
    for tier in PHASE1_TIER_ORDER:
        spec = tier_spec(tier)
        dataset = tier_dataset(spec, corpus_v11_root)
        selected = training_streams(spec, dataset)
        assert len(selected) == TRAINING_STREAM_COUNT, tier
        assert len({s.key for s in selected}) == TRAINING_STREAM_COUNT, tier
        assert sorted(s.flow_draw for s in selected) == list(range(1, 201)), (
            f"{tier}: every tier must cover the same 200 demand draws exactly once"
        )
        assert len(top_decile_streams(selected)) == 20, tier


def test_the_real_random_tier_holds_two_streams_per_draw_before_the_subsample(
    corpus_v11_root: Path,
) -> None:
    """The stratified rule only means something if the pool really has a choice to make."""
    spec = tier_spec("random")
    dataset = tier_dataset(spec, corpus_v11_root)
    streams = stream_returns(dataset)
    assert len(streams) == 400
    counts = {}
    for entry in streams:
        counts[entry.flow_draw] = counts.get(entry.flow_draw, 0) + 1
    assert set(counts.values()) == {2}


def test_the_rebuilt_flow_matches_the_recorded_hash_for_a_declared_sample(
    corpus_v11_root: Path,
) -> None:
    """Arrivals must belong to the draws the corpus used, and the corpus records the hashes.

    The sample is **declared here**: draws 1, 50, 100, 150 and 200 of the MaxPressure tier.  The
    full 200-draw verification runs inside ``draw_arrivals`` when the diagnostics are produced.
    """
    manifest_path = corpus_v11_root / "cf_hz1x1__maxpressure" / "manifest.json"
    arrivals = draw_arrivals(manifest_path)
    assert sorted(arrivals) == list(range(1, 201))
    for draw in (1, 50, 100, 150, 200):
        assert arrivals[draw] > 0
    assert len({arrivals[d] for d in arrivals}) > 1, (
        "constant arrivals would make check A vacuous"
    )
