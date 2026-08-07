"""Tests for ``offline.dt_gate`` -- the gate arithmetic, the leakage guards and the training loop.

Three layers, and the report says which of them ran:

* pure arithmetic (verdict, CIs, Wilcoxon, plateau rule) -- always runs;
* the training loop and the checkpoint guards over a synthetic corpus -- always runs;
* the corpus/simulator replay test -- **opt-in**, and it skips with a reason naming the
  variables it needs.  A green suite in which that one skipped is the failure mode, not the
  safeguard, so the Return Packet reports whether it ran.

The replay test is the load-bearing one: it is the only check that the env the DT is evaluated
in reproduces the env the corpus was collected in.  Without it every reported number rests on
an assumption about state features and reward configuration.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from offline.dataset import DRAW_SPLITS, TrajectoryWindowDataset
from offline.dt_gate import (
    ARTIFACT_FORMAT_VERSION,
    GATE_RATIO,
    HELD_OUT_DRAWS,
    build_training_dataset,
    env_settings_from_manifest,
    gate_verdict,
    load_gate_checkpoint,
    mean_ci95,
    plateau_reached,
    stack_dataset,
    train_dt,
    wilcoxon_signed_rank,
    window_means,
    write_json_atomic,
)

from tests.test_offline_dataset import write_dataset_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# The gate itself
# ----------------------------------------------------------------------


def test_gate_passes_only_when_both_inequalities_hold() -> None:
    both = gate_verdict(att_madt=100.0, att_maxpressure=176.5, att_best_online=105.0)
    assert (both.gate_a, both.gate_b, both.passed) == (True, True, True)

    beats_heuristic_only = gate_verdict(
        att_madt=150.0, att_maxpressure=176.5, att_best_online=105.0
    )
    assert (beats_heuristic_only.gate_a, beats_heuristic_only.gate_b) == (True, False)
    assert beats_heuristic_only.passed is False

    beats_online_only = gate_verdict(
        att_madt=104.0, att_maxpressure=100.0, att_best_online=105.0
    )
    assert (beats_online_only.gate_a, beats_online_only.gate_b) == (False, True)
    assert beats_online_only.passed is False


def test_equality_passes_because_the_registration_says_less_than_or_equal() -> None:
    """``ATT_MADT <= ...`` verbatim: a strict ``<`` would fail a run the registration passes."""
    exact = gate_verdict(att_madt=110.25, att_maxpressure=110.25, att_best_online=105.0)
    assert exact.threshold_online == 105.0 * GATE_RATIO
    assert (exact.gate_a, exact.gate_b, exact.passed) == (True, True, True)


def test_the_online_threshold_is_the_registered_five_percent_band() -> None:
    verdict = gate_verdict(att_madt=1.0, att_maxpressure=2.0, att_best_online=105.46135581970215)
    assert verdict.threshold_online == 105.46135581970215 * 1.05
    assert round(verdict.threshold_online, 2) == 110.73


# ----------------------------------------------------------------------
# Descriptives
# ----------------------------------------------------------------------


def test_mean_ci95_matches_an_independent_computation() -> None:
    values = [101.5, 99.25, 110.0, 97.75, 105.5, 102.0]
    stats = mean_ci95(values)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    assert stats.n == 6
    assert stats.mean == pytest.approx(mean, abs=1e-12)
    assert stats.std == pytest.approx(math.sqrt(variance), abs=1e-12)
    assert stats.ci95 == pytest.approx(1.96 * math.sqrt(variance) / math.sqrt(6), abs=1e-12)


def test_mean_ci95_of_a_single_value_has_no_spread() -> None:
    stats = mean_ci95([42.0])
    assert (stats.n, stats.mean, stats.std, stats.ci95) == (1, 42.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# Wilcoxon signed-rank, implemented here because scipy is not installed
# ----------------------------------------------------------------------


def test_wilcoxon_reproduces_a_hand_computed_example() -> None:
    """Differences 1, -2, 3, -4, 5, 6: ranks 1..6, W+ = 1+3+5+6 = 15, W- = 2+4 = 6."""
    x = [11.0, 8.0, 13.0, 6.0, 15.0, 16.0]
    y = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    result = wilcoxon_signed_rank(x, y)
    assert result.n_used == 6
    assert result.n_zero == 0
    assert result.w_plus == 15.0
    assert result.w_minus == 6.0
    assert result.statistic == 6.0
    assert result.w_plus + result.w_minus == 6 * 7 / 2


def test_wilcoxon_drops_zero_differences_and_averages_tied_ranks() -> None:
    """Differences 0, 0, 2, -2, 2: two zeros dropped; |d| = 2,2,2 share rank (1+2+3)/3 = 2."""
    x = [5.0, 7.0, 12.0, 8.0, 12.0]
    y = [5.0, 7.0, 10.0, 10.0, 10.0]
    result = wilcoxon_signed_rank(x, y)
    assert result.n_zero == 2
    assert result.n_used == 3
    assert result.w_plus == 4.0
    assert result.w_minus == 2.0


def test_wilcoxon_is_symmetric_under_swapping_the_arguments() -> None:
    generator = np.random.default_rng(3)
    x = generator.normal(size=40).tolist()
    y = generator.normal(size=40).tolist()
    forward = wilcoxon_signed_rank(x, y)
    backward = wilcoxon_signed_rank(y, x)
    assert forward.w_plus == backward.w_minus
    assert forward.statistic == backward.statistic
    assert forward.p_value == backward.p_value


def test_wilcoxon_separates_a_shifted_sample_from_an_unshifted_one() -> None:
    """Discriminating power: an obvious effect must reach a small p, a null must not."""
    generator = np.random.default_rng(11)
    base = generator.normal(size=100)
    shifted = wilcoxon_signed_rank((base + 3.0).tolist(), base.tolist())
    null = wilcoxon_signed_rank(base.tolist(), (base + 1e-9).tolist())
    assert shifted.p_value < 1e-10
    assert null.p_value > 1e-10
    assert shifted.p_value < null.p_value


def test_wilcoxon_refuses_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="paired"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


# ----------------------------------------------------------------------
# The declared budget and its single pre-declared raise (training curve only)
# ----------------------------------------------------------------------


def test_window_means_splits_the_curve_into_equal_windows() -> None:
    losses = [float(i) for i in range(10)]
    assert window_means(losses, 5) == (2.0, 7.0)


def test_window_means_refuses_a_curve_that_does_not_fill_its_windows() -> None:
    with pytest.raises(ValueError, match="window"):
        window_means([1.0, 2.0, 3.0], 2)


def test_plateau_needs_the_last_two_relative_changes_below_the_tolerance() -> None:
    assert plateau_reached([10.0, 5.0, 4.9, 4.85]) is True
    assert plateau_reached([10.0, 5.0, 4.9, 4.0]) is False   # last change 18%
    assert plateau_reached([10.0, 5.0, 4.0, 3.96]) is False  # previous change 20%


def test_plateau_needs_at_least_three_windows() -> None:
    with pytest.raises(ValueError, match="three"):
        plateau_reached([10.0, 9.9])


# ----------------------------------------------------------------------
# Held-out purity, mechanically
# ----------------------------------------------------------------------


def test_the_evaluation_pool_is_the_whole_registered_held_out_range() -> None:
    low, high = DRAW_SPLITS["heldout"]
    assert HELD_OUT_DRAWS == tuple(range(low, high + 1))
    assert len(HELD_OUT_DRAWS) == 100


def test_no_evaluation_draw_can_be_a_training_draw() -> None:
    train_low, train_high = DRAW_SPLITS["train"]
    training_pool = set(range(train_low, train_high + 1))
    assert training_pool.isdisjoint(set(HELD_OUT_DRAWS))


def test_a_held_out_draw_cannot_enter_the_training_dataset(tmp_path: Path) -> None:
    """The loader refuses it; this asserts the gate module asks for the training split."""
    dataset_dir = write_dataset_dir(tmp_path, "heldout__policy", draws=(1000, 1001))
    with pytest.raises(ValueError, match="held-out"):
        build_training_dataset([dataset_dir], context_length=4)


# ----------------------------------------------------------------------
# Training: the stacked tensors, determinism, and the no-model-selection guard
# ----------------------------------------------------------------------


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> TrajectoryWindowDataset:
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    return build_training_dataset([dataset_dir], context_length=4)


def test_stacked_tensors_reproduce_the_loader_item_by_item(
    synthetic_dataset: TrajectoryWindowDataset,
) -> None:
    """The loader stays the single definition of a window; stacking must not reinterpret it."""
    group = sorted(synthetic_dataset.groups)[0]
    indices = synthetic_dataset.groups[group]
    stacked = stack_dataset(synthetic_dataset)

    generator = np.random.default_rng(0)
    sample = generator.choice(len(indices), size=min(6, len(indices)), replace=False)
    for position in sample:
        index = indices[int(position)]
        item = synthetic_dataset[index]
        for key, value in item.items():
            assert torch.equal(stacked[key][index], value), f"{key} at item {index}"


def _train_once(
    dataset: TrajectoryWindowDataset, tmp_path: Path, seed: int, steps: int, name: str
) -> Any:
    group = sorted(dataset.groups)[0]
    state_dim, n_actions = group
    return train_dt(
        stack_dataset(dataset),
        state_dim=state_dim,
        n_actions=n_actions,
        seed=seed,
        declared_gradient_steps=steps,
        raise_to=None,
        context_length=4,
        batch_size=8,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / name,
        stats=dataset.stats,
        scenario_id="fixture_2ix",
        target_rtg=-1.0,
        rtg_scale=10.0,
        provenance={"tier": "fixture"},
    )


def test_training_is_byte_identical_for_the_same_seed(
    synthetic_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    first = _train_once(synthetic_dataset, tmp_path, 101, 6, "a.pt")
    second = _train_once(synthetic_dataset, tmp_path, 101, 6, "b.pt")
    left = torch.load(first.checkpoint_path, map_location="cpu", weights_only=False)
    right = torch.load(second.checkpoint_path, map_location="cpu", weights_only=False)
    assert first.losses == second.losses
    for key, tensor in left["model"].items():
        assert torch.equal(tensor, right["model"][key]), key


def test_a_different_seed_trains_to_different_weights(
    synthetic_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """The control that lets the determinism test above fail."""
    first = _train_once(synthetic_dataset, tmp_path, 101, 6, "a.pt")
    second = _train_once(synthetic_dataset, tmp_path, 202, 6, "c.pt")
    left = torch.load(first.checkpoint_path, map_location="cpu", weights_only=False)
    right = torch.load(second.checkpoint_path, map_location="cpu", weights_only=False)
    assert any(
        not torch.equal(tensor, right["model"][key]) for key, tensor in left["model"].items()
    )


def test_the_reported_step_count_is_the_declared_one(
    synthetic_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    result = _train_once(synthetic_dataset, tmp_path, 101, 6, "a.pt")
    assert result.gradient_steps == 6
    assert result.declared_gradient_steps == 6
    assert len(result.losses) == 6


def test_a_checkpoint_from_another_step_count_cannot_be_evaluated(
    synthetic_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """No online model selection: a better-scoring earlier checkpoint is unreachable here."""
    result = _train_once(synthetic_dataset, tmp_path, 101, 6, "a.pt")

    class _Env:
        class _Ix:
            id = "ix_alpha"
            num_phases = 3

        intersections = [_Ix()]
        max_steps = 8
        action_space = None

    with pytest.raises(ValueError, match="gradient steps"):
        load_gate_checkpoint(_Env(), result.checkpoint_path, declared_gradient_steps=20000)


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------


def test_write_json_atomic_refuses_a_missing_directory_and_creates_nothing(tmp_path: Path) -> None:
    target = tmp_path / "absent" / "out.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_json_atomic({"a": 1}, target)
    assert not (tmp_path / "absent").exists()


def test_write_json_atomic_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_json_atomic({"format_version": ARTIFACT_FORMAT_VERSION}, tmp_path / "out.json")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["out.json"]
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))[
        "format_version"
    ] == ARTIFACT_FORMAT_VERSION


def test_env_settings_come_from_the_manifest_not_from_a_restatement(tmp_path: Path) -> None:
    """A restated setting is a setting that can drift away from the collection env."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_metadata"].update(
        {
            "max_steps": 123,
            "delta_time": 7,
            "control_mode": "acyclic",
            "state_features": ["lane_vehicle_count"],
            "global_reward_fn": "queue_length",
            "local_reward_fn": "queue_length",
            "global_reward_weight": 0.0,
            "metrics": None,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    settings = env_settings_from_manifest(manifest_path)
    assert settings["max_steps"] == 123
    assert settings["delta_time"] == 7
    assert settings["state_features"] == ["lane_vehicle_count"]
    assert settings["global_reward_weight"] == 0.0


def test_env_settings_refuse_a_manifest_without_a_local_reward_function(tmp_path: Path) -> None:
    """Without a local reward the info carries no per-intersection reward, so RTG cannot advance."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_metadata"]["local_reward_fn"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="local_reward_fn"):
        env_settings_from_manifest(manifest_path)


# ----------------------------------------------------------------------
# The load-bearing one: the evaluation env must reproduce the collection env
# ----------------------------------------------------------------------


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture()
def hz1x1_maxpressure_run() -> Path:
    """The v1.1 cf_hz1x1 MaxPressure collection directory, or a skip naming the variable."""
    root = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(root) if root else REPO_ROOT / "datasets_v11"
    run_dir = candidate / "cf_hz1x1__maxpressure"
    if not (run_dir / "manifest.json").is_file():
        pytest.skip(
            f"v1.1 corpus not found at {run_dir}: set RLTRAFFIC_CORPUS_V11 to the collected "
            "datasets_v11/ directory to run the collection-env replay test"
        )
    return run_dir


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow engine is not importable here")
def test_the_evaluation_env_reproduces_a_stored_corpus_episode_bit_exactly(
    hz1x1_maxpressure_run: Path,
) -> None:
    """Replay a stored episode's own actions and demand; every stream must match exactly.

    This is the only check that the DT is evaluated in the env it was trained from.  A drift in
    the state block, the reward function or the global reward weight would leave every reported
    number plausible and wrong.
    """
    from experiments.config import EnvSpec
    from experiments.envs import make_env
    from offline.trajectory_logger import load_episode

    manifest = json.loads((hz1x1_maxpressure_run / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["episodes"][0]
    episode = load_episode(hz1x1_maxpressure_run / entry["filename"])
    ix_id = episode.ix_ids[0]
    arrays = episode.intersections[ix_id]

    settings = env_settings_from_manifest(hz1x1_maxpressure_run / "manifest.json")
    draw_config = hz1x1_maxpressure_run / "flows" / f"cityflow_draw{int(episode.flow_draw)}.json"
    assert draw_config.is_file(), f"the run's own drawn config is missing: {draw_config}"

    env = make_env(
        EnvSpec(
            id=manifest["run_metadata"]["scenario_id"],
            backend="cityflow",
            paths={"config": str(draw_config)},
            settings=settings,
        )
    )
    try:
        info = env.reset(seed=int(episode.engine_seed))
        states = [np.asarray(info["intersections"][ix_id]["state"], dtype=np.float32)]
        rewards: list[float] = []
        att: list[float] = [float(info["average_travel_time"])]
        vehicles: list[float] = [float(info["vehicle_count"])]
        for t in range(int(episode.episode_length)):
            _reward, _term, _trunc, info = env.step(
                np.asarray([arrays.action[t]], dtype=np.int64)
            )
            states.append(np.asarray(info["intersections"][ix_id]["state"], dtype=np.float32))
            rewards.append(float(info["intersections"][ix_id]["reward"]))
            att.append(float(info["average_travel_time"]))
            vehicles.append(float(info["vehicle_count"]))
    finally:
        env.close()

    assert np.array_equal(np.stack(states), arrays.state)
    assert np.array_equal(np.asarray(rewards, dtype=np.float32), arrays.local_reward)
    assert np.array_equal(
        np.asarray(vehicles, dtype=episode.vehicle_count.dtype), episode.vehicle_count
    )
    # float32 on disk against a float64 recomputation: cast DOWN, never widen the stored array
    # (offline/dataset.py's docstring, measured on numpy 2.5.1).
    assert episode.att_per_step is not None
    assert np.array_equal(np.float32(np.asarray(att)), episode.att_per_step)
