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
    EpisodeResult,
    _paired,
    _cell,
    build_training_dataset,
    env_settings_from_manifest,
    expected_reported_steps,
    gate_verdict,
    load_gate_checkpoint,
    mean_ci95,
    plateau_reached,
    policy_source_for,
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
    """Differences 0, 0, 2, 2, -2: two zeros dropped; |d| = 2,2,2 share rank (1+2+3)/3 = 2.

    THE SIGN ORDER IS LOAD-BEARING and was changed after a surviving mutant.  The original
    fixture used ``[+, -, +]``, for which average ranks and plain ordinal ranks give the *same*
    ``w_plus`` (1+3 = 2+2 = 4), so a tie-blind implementation passed it.  With ``[+, +, -]``
    the two disagree: average ranks give ``w_plus = 4``, ordinal ranks give ``3``.
    """
    x = [5.0, 7.0, 12.0, 12.0, 8.0]
    y = [5.0, 7.0, 10.0, 10.0, 10.0]
    result = wilcoxon_signed_rank(x, y)
    assert result.n_zero == 2
    assert result.n_used == 3
    assert result.w_plus == 4.0
    assert result.w_minus == 2.0
    assert result.statistic == 2.0


def test_wilcoxon_is_symmetric_under_swapping_the_arguments() -> None:
    generator = np.random.default_rng(3)
    x = generator.normal(size=40).tolist()
    y = generator.normal(size=40).tolist()
    forward = wilcoxon_signed_rank(x, y)
    backward = wilcoxon_signed_rank(y, x)
    assert forward.w_plus == backward.w_minus
    assert forward.statistic == backward.statistic
    assert forward.p_value == backward.p_value


def test_wilcoxon_separates_a_shifted_sample_from_a_genuine_null() -> None:
    """Discriminating power: an obvious effect must reach a small p, a null must not.

    The null is two INDEPENDENT samples.  It was originally written as ``base`` against
    ``base + 1e-9``, which is not a null at all -- see the test below, which now pins the
    property that mistake was hiding.
    """
    generator = np.random.default_rng(11)
    base = generator.normal(size=100)
    other = generator.normal(size=100)
    shifted = wilcoxon_signed_rank((base + 3.0).tolist(), base.tolist())
    null = wilcoxon_signed_rank(base.tolist(), other.tolist())
    assert shifted.p_value < 1e-10
    assert null.p_value > 0.05
    assert shifted.p_value < null.p_value


def test_wilcoxon_detects_a_tiny_but_perfectly_consistent_shift() -> None:
    """The signed-rank test reads signs and rank order, never magnitudes.

    100 differences of one sign is the most extreme outcome the test has, however small each
    difference is.  Pinned because assuming otherwise is exactly the error that made the
    previous test's original null case wrong.
    """
    generator = np.random.default_rng(11)
    base = generator.normal(size=100)
    result = wilcoxon_signed_rank(base.tolist(), (base + 1e-9).tolist())
    assert result.n_used == 100
    assert result.w_plus == 0.0
    assert result.w_minus == 100 * 101 / 2
    assert result.p_value < 1e-10


def test_wilcoxon_refuses_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="paired"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


def test_wilcoxon_z_and_p_match_a_variance_derived_from_first_principles() -> None:
    """Pins ``z`` and ``p``, including BOTH corrections, by a second derivation.

    Added after an independent review found that deleting the tie correction OR the continuity
    correction left every Wilcoxon test passing: all the others assert ``w_plus``/``w_minus`` or
    an inequality on ``p``, so nothing constrained the variance -- while the p-values go in the
    paper.

    The second route is a genuinely different derivation, not a copy of the implementation's
    expression.  Under the null each signed rank contributes ``+/- r_i`` with probability 1/2,
    so ``Var(W+) = sum(r_i^2) / 4``.  With average ranks that identity *automatically* carries
    the tie correction (verified here: 50.375, against 51.0 with no ties), which is why it kills
    a missing ``sum(t^3 - t)/48`` term.
    """
    differences = [1.0, 1.0, -2.0, 3.0, 3.0, 3.0, -4.0, 5.0]
    x = [10.0 + d for d in differences]
    y = [10.0] * len(differences)
    result = wilcoxon_signed_rank(x, y)

    ranks = {1.0: 1.5, 2.0: 3.0, 3.0: 5.0, 4.0: 7.0, 5.0: 8.0}   # by hand, average for ties
    signed = [(d, ranks[abs(d)]) for d in differences]
    w_plus = math.fsum(r for d, r in signed if d > 0)
    w_minus = math.fsum(r for d, r in signed if d < 0)
    n = len(differences)
    variance = math.fsum(r * r for _d, r in signed) / 4.0
    assert variance == 50.375
    statistic = min(w_plus, w_minus)
    expected_z = (statistic - n * (n + 1) / 4.0 + 0.5) / math.sqrt(variance)
    expected_p = min(1.0, 2.0 * (0.5 * math.erfc(-expected_z / math.sqrt(2.0))))

    assert result.w_plus == w_plus
    assert result.w_minus == w_minus
    assert result.statistic == statistic
    assert result.z == pytest.approx(expected_z, abs=1e-12)
    assert result.p_value == pytest.approx(expected_p, abs=1e-15)


def _result(arm: str, draw: int, att: float, seed: int | None = None) -> EpisodeResult:
    return EpisodeResult(
        arm=arm,
        seed=seed,
        draw_id=draw,
        att_horizon=att,
        horizon_vehicle_count=0.0,
        episode_reward=0.0,
    )


def test_pairing_uses_only_shared_draws_and_averages_seeds_within_a_draw() -> None:
    """A5 point 3: the paired unit is the draw, and seeds are averaged inside it."""
    left = [_result("madt", 1, 100.0, seed=1), _result("madt", 1, 200.0, seed=2)]
    left.append(_result("madt", 2, 50.0, seed=1))
    right = [_result("mp", 1, 300.0), _result("mp", 2, 60.0)]
    a, b, shared = _paired(left, right)
    assert shared == [1, 2]
    assert a == [150.0, 50.0]
    assert b == [300.0, 60.0]


def test_a_comparison_without_shared_draws_is_void_not_approximate() -> None:
    """Amendment A5 makes this binary and checkable, replacing A4's withdrawn 5% band."""
    left = [_result("madt", 1000, 100.0)]
    right = [_result("mp", 7, 100.0)]
    with pytest.raises(ValueError, match="void"):
        _paired(left, right)


# ----------------------------------------------------------------------
# The declared budget and its single pre-declared raise (training curve only)
# ----------------------------------------------------------------------


def test_window_means_splits_the_curve_into_equal_windows() -> None:
    losses = [float(i) for i in range(10)]
    assert window_means(losses, 5) == (2.0, 7.0)


def test_window_means_refuses_a_curve_that_does_not_fill_its_windows() -> None:
    with pytest.raises(ValueError, match="windows of"):
        window_means([1.0, 2.0, 3.0], 2)


def test_plateau_needs_the_last_two_relative_changes_below_the_tolerance() -> None:
    assert plateau_reached([10.0, 5.0, 4.9, 4.85]) is True
    assert plateau_reached([10.0, 5.0, 4.9, 4.0]) is False   # last change 18%
    assert plateau_reached([10.0, 5.0, 4.0, 3.96]) is False  # previous change 20%


def test_plateau_needs_at_least_three_windows() -> None:
    with pytest.raises(ValueError, match="least three"):
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
    """The loader stays the single definition of a window; stacking must not reinterpret it.

    Rows are addressed through the returned ``item_index`` rather than by dataset index: a
    stacked block covers ONE ``(state_dim, n_actions)`` group, so the two numbering schemes
    coincide only when the dataset has a single group.  This fixture has two, deliberately.
    """
    group = sorted(synthetic_dataset.groups)[0]
    stacked = stack_dataset(synthetic_dataset, group=group)
    indices = stacked["item_index"].tolist()
    assert indices == synthetic_dataset.groups[group]

    generator = np.random.default_rng(0)
    sample = generator.choice(len(indices), size=min(6, len(indices)), replace=False)
    for row in sample:
        index = indices[int(row)]
        item = synthetic_dataset[index]
        for key, value in item.items():
            assert torch.equal(stacked[key][int(row)], value), f"{key} at item {index}"


def test_stacking_a_multi_group_dataset_without_naming_the_group_raises(
    synthetic_dataset: TrajectoryWindowDataset,
) -> None:
    """C6 forbids padding across intersections, so silently taking the first group is wrong."""
    assert len(synthetic_dataset.groups) > 1
    with pytest.raises(ValueError, match="pass group="):
        stack_dataset(synthetic_dataset)


def _train_once(
    dataset: TrajectoryWindowDataset, tmp_path: Path, seed: int, steps: int, name: str
) -> Any:
    group = sorted(dataset.groups)[0]
    state_dim, n_actions = group
    return train_dt(
        stack_dataset(dataset, group=group),
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

    with pytest.raises(ValueError, match="gradient steps but"):
        load_gate_checkpoint(_Env(), result.checkpoint_path, declared_gradient_steps=20000)


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------


def test_a_heuristic_arm_is_never_recorded_as_running_from_a_checkpoint() -> None:
    """Queue item 0b needs ``policy_source`` to be true, not merely present.

    It was hardcoded to ``"checkpoint"`` in ``_cell`` and corrected only on the baselines path,
    so the GATE artifact -- the one later tasks are told to reuse -- claimed MaxPressure ran
    from a checkpoint. An independent review found it in the committed JSON.
    """
    assert policy_source_for("maxpressure") == "deterministic_heuristic"
    assert policy_source_for("fixedtime") == "deterministic_heuristic"
    assert policy_source_for("madt") == "checkpoint"
    assert policy_source_for("mappo1000") == "checkpoint"

    cell = _cell([_result("maxpressure", 1000, 176.0), _result("maxpressure", 1001, 177.0)])
    assert cell["policy_source"] == "deterministic_heuristic"
    assert _cell([_result("madt", 1000, 104.0)])["policy_source"] == "checkpoint"


def test_a_cell_that_straddles_two_arms_is_refused() -> None:
    with pytest.raises(ValueError, match="one arm"):
        _cell([_result("madt", 1000, 104.0), _result("maxpressure", 1000, 176.0)])


def test_the_reported_budget_is_checked_against_the_DECLARATION_not_against_itself() -> None:
    """The guard was vacuous: both sides of the comparison came from the same training run.

    ``_run_evaluate`` read ``reported_gradient_steps`` out of the artifact and handed it to
    ``load_gate_checkpoint``, which then compared 40000 to 40000. Here the declaration is the
    input and the artifact is what gets checked.
    """
    raised = {"declared_gradient_steps": 20000, "raise_to": 40000, "raise_taken": True,
              "reported_gradient_steps": 40000}
    plain = {"declared_gradient_steps": 20000, "raise_to": 40000, "raise_taken": False,
             "reported_gradient_steps": 20000}
    assert expected_reported_steps(raised, declared=20000) == 40000
    assert expected_reported_steps(plain, declared=20000) == 20000

    # trained longer than declared, without the raise ever being taken
    sneaky = {**plain, "reported_gradient_steps": 40000}
    with pytest.raises(ValueError, match="neither the declared"):
        expected_reported_steps(sneaky, declared=20000)

    # an artifact produced under a different declaration entirely
    with pytest.raises(ValueError, match="a declared budget"):
        expected_reported_steps(raised, declared=30000)

    # a third budget, reachable by neither the declaration nor the single raise
    with pytest.raises(ValueError, match="neither the declared"):
        expected_reported_steps({**raised, "reported_gradient_steps": 60000}, declared=20000)


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
    """Without a local reward the info carries no per-intersection reward, so RTG cannot advance.

    The manifest is otherwise COMPLETE: with keys missing, the missing-key check fires first and
    the test would pass on the wrong error.
    """
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_metadata"].update(
        {
            "max_steps": 8,
            "delta_time": 10,
            "control_mode": "acyclic",
            "state_features": ["lane_vehicle_count"],
            "global_reward_fn": "queue_length",
            "global_reward_weight": 0.0,
            "local_reward_fn": None,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="with local_reward_fn"):
        env_settings_from_manifest(manifest_path)


def test_env_settings_name_every_missing_key_rather_than_the_first(tmp_path: Path) -> None:
    """A manifest that cannot rebuild the env must say what it lacks, not fail one key at a time."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    with pytest.raises(ValueError, match="run_metadata is missing"):
        env_settings_from_manifest(dataset_dir / "manifest.json")


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
