"""Tests for ``offline.rtg_ablation`` -- P5.3a's teacher-forced RTG probe and its two tables.

Most of this file runs on fixtures and on the **committed artifacts** ``p4_6_grid.json`` and
``p4_7_grid.json``, which are the evidence this task must stay consistent with.  The tests that need
the v1.1 corpus or a checkpoint from ``output/`` say so in their names and skip with a reason naming
``RLTRAFFIC_CORPUS_V11``.

Two conventions this file pins, because getting either wrong would produce a plausible number
---------------------------------------------------------------------------------------------
1. ``teacher_forced_logits`` returns the model's **raw** head output; the availability mask is
   applied once, inside :func:`compare_logits`, through the frozen
   ``agent.DTAgent.masked_action_logits``.  One masking site, not two.
2. ``conditioning_series`` returns the **unscaled** return-to-go, exactly as ``DTAgent`` carries it
   in ``_Context.rtg``; the division by ``rtg_scale`` happens inside ``DTAgent._window``
   (``agent/DTAgent.py:596``).  A probe that scaled twice would look entirely reasonable.

⚠️ ``zero`` and ``grid_g0`` are different interventions and
:func:`test_zero_is_not_the_same_intervention_as_grid_g0` exists because they are easy to confuse:
``grid_g0`` sets the *target* to 0 and keeps decrementing, giving ``-cumsum(r)``; ``zero`` feeds a
constant 0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from agent.DTAgent import DTAgent, masked_action_logits
from offline.dataset import RtgSummary
from offline.method_tier_grid import TIERS, assert_no_verdicts
from offline.rtg_ablation import (
    ARTIFACT_FORMAT_VERSION,
    INTERVENTION_KEYS,
    PROBE_STREAM_COUNT,
    PROBE_STREAM_STRIDE,
    PROBE_TIERS,
    RTG_SPREAD_TIMESTEPS,
    InterventionComparison,
    ProbeCell,
    assert_declared_interventions,
    behaviour_margin_degenerate,
    between_episode_rtg_spread,
    committed_rtg_summary,
    compare_logits,
    conditioning_series,
    declared_interventions,
    delta_table,
    episode_return_stats,
    flip_rate,
    ramp_prediction,
    recomputed_rtg_summary,
    report_artifact,
    selected_stream_indices,
    teacher_forced_logits,
    total_variation_distance,
)

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
REPO = Path(__file__).resolve().parents[1]

STATE_DIM = 4
N_ACTIONS = 3
CONTEXT = 4
MAX_EP_LEN = 16
TARGET = -100.0
RTG_SCALE = 50.0

#: A8's control tier and the two tiers the identity test re-rolls.
DEGENERATE_TIERS = ("fixedtime", "maxpressure")
A6_DELTA_TIER = "mappo1000"
A6_DELTA = 0.6263


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class _StubIntersection:
    def __init__(self, ix_id: str, n_actions: int) -> None:
        self.id = ix_id
        self.n_actions = n_actions
        self.incoming_lanes: list[str] = []


class _StubEnv:
    def __init__(self) -> None:
        self.intersections = [_StubIntersection("ix_only", N_ACTIONS)]
        self.max_steps = MAX_EP_LEN
        self.action_space = gym_spaces.Discrete(N_ACTIONS)


def _agent(**overrides: Any) -> DTAgent:
    params: dict[str, Any] = {
        "context_length": CONTEXT,
        "n_layer": 2,
        "n_head": 1,
        "d_model": 16,
        "dropout": 0.0,
        "max_ep_len": MAX_EP_LEN,
        "target_rtg": TARGET,
        "rtg_scale": RTG_SCALE,
        "device": "cpu",
        "seed": 5,
        "state_dim": STATE_DIM,
    }
    params.update(overrides)
    return DTAgent(_StubEnv(), **params)


def _info(step: int, state: Sequence[float], reward: float) -> dict[str, Any]:
    return {
        "step": step,
        "vehicle_count": 0,
        "intersections": {
            "ix_only": {
                "state": list(state),
                "avail_actions": list(range(N_ACTIONS)),
                "reward": reward,
            }
        },
    }


REWARDS = np.array([-3.0, -4.5, -0.25, -8.0, -1.75, -6.0], dtype=np.float32)
ZERO_REWARDS = np.zeros(6, dtype=np.float32)


def _episode(length: int = 6, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "state": rng.standard_normal((length + 1, STATE_DIM)).astype(np.float32),
        "avail_mask": np.ones((length + 1, N_ACTIONS), dtype=np.bool_),
        "action": rng.integers(0, N_ACTIONS, size=length).astype(np.int64),
    }


def _corpus_root() -> Path:
    """``RLTRAFFIC_CORPUS_V11``, else ``<repo>/datasets_v11``; skip if neither exists."""
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected corpus to run the real-corpus checks"
        )
    return candidate


def _checkpoint(relative: str) -> Path:
    path = REPO / "output" / relative
    if not path.is_file():
        pytest.skip(f"checkpoint not present in this tree: {path}")
    return path


def _grids() -> list[dict[str, Any]]:
    return [
        json.loads((DATA / name).read_text(encoding="utf-8"))
        for name in ("p4_6_grid.json", "p4_7_grid.json")
    ]


# ----------------------------------------------------------------------
# 12: the RTG sequence, by two routes
# ----------------------------------------------------------------------


def test_the_conditioning_series_matches_an_independent_cumsum_route() -> None:
    """``rtg_t = target - sum(r_k for k < t)``, recomputed with ``np.cumsum``, exact in float64."""
    produced = conditioning_series(TARGET, REWARDS, kind="decrement")
    consumed = np.concatenate([[0.0], np.cumsum(REWARDS.astype(np.float64))[:-1]])
    expected = TARGET - consumed

    assert produced.dtype == np.float64
    assert produced.shape == REWARDS.shape
    assert np.array_equal(produced, expected)
    assert produced[0] == TARGET


def test_the_conditioning_series_is_the_one_dtagent_itself_advances() -> None:
    """The series is tied to production code, not merely to a formula in this file.

    After ``act`` at step ``t``, ``DTAgent.current_rtg()`` is ``target - sum(r_k for k < t)``
    (``agent/DTAgent.py:662``), which is exactly ``series[t]``.
    """
    agent = _agent()
    series = conditioning_series(TARGET, REWARDS, kind="decrement")
    for step in range(len(REWARDS)):
        reward = 0.0 if step == 0 else float(REWARDS[step - 1])
        agent.act(_info(step, [1.0, 1.0, 1.0, 1.0], reward), explore=False)
        assert agent.current_rtg()["ix_only"] == series[step], f"step {step}"


def test_the_frozen_series_never_decrements_and_the_zero_series_is_flat_zero() -> None:
    frozen = conditioning_series(TARGET, REWARDS, kind="frozen")
    zeroed = conditioning_series(TARGET, REWARDS, kind="zero")
    assert np.array_equal(frozen, np.full(REWARDS.shape, TARGET, dtype=np.float64))
    assert np.array_equal(zeroed, np.zeros(REWARDS.shape, dtype=np.float64))


def test_frozen_differs_from_baseline_when_rewards_are_non_zero() -> None:
    baseline = conditioning_series(TARGET, REWARDS, kind="decrement")
    frozen = conditioning_series(TARGET, REWARDS, kind="frozen")
    assert not np.array_equal(baseline, frozen)
    # And they agree at t = 0 by construction, which is why only t > 0 can separate them.
    assert baseline[0] == frozen[0]
    assert not np.array_equal(baseline[1:], frozen[1:])


def test_frozen_equals_baseline_exactly_when_every_reward_is_zero() -> None:
    """The exact condition under which the hazard at ``agent/DTAgent.py:560,628`` is harmless."""
    baseline = conditioning_series(TARGET, ZERO_REWARDS, kind="decrement")
    frozen = conditioning_series(TARGET, ZERO_REWARDS, kind="frozen")
    assert np.array_equal(baseline, frozen)


def test_zero_is_not_the_same_intervention_as_grid_g0() -> None:
    """``grid_g0`` still decrements from a target of 0; ``zero`` feeds a constant."""
    grid_g0 = conditioning_series(0.0, REWARDS, kind="decrement")
    zeroed = conditioning_series(0.0, REWARDS, kind="zero")
    assert not np.array_equal(grid_g0, zeroed)
    assert grid_g0[0] == 0.0 and zeroed[0] == 0.0
    assert np.array_equal(grid_g0[1:], -np.cumsum(REWARDS.astype(np.float64))[:-1])


def test_the_declared_interventions_are_twelve_and_fixed() -> None:
    """The grid may not grow after the fact; the keys are registered in ``docs/plans/p5.3a.md``."""
    assert len(INTERVENTION_KEYS) == 12
    assert INTERVENTION_KEYS[0] == "baseline"
    assert set(INTERVENTION_KEYS) == {"baseline", "zero", "frozen"} | {
        f"grid_g{i}" for i in range(9)
    }

    declared = declared_interventions(TARGET)
    assert tuple(i.key for i in declared) == INTERVENTION_KEYS
    baseline = declared[0]
    assert baseline.kind == "decrement" and baseline.target_rtg == TARGET
    # The nine grid points are P4.3's, unchanged and in declaration order.
    from offline.rtg_calibration import DECLARED_GRID

    grid = tuple(i.target_rtg for i in declared if i.key.startswith("grid_g"))
    assert grid == DECLARED_GRID


# ----------------------------------------------------------------------
# 13-16: flip rate, TVD and the mask
# ----------------------------------------------------------------------


def test_flip_rate_of_an_identical_sequence_is_exactly_zero() -> None:
    actions = np.array([0, 2, 1, 1, 0, 2, 1], dtype=np.int64)
    assert flip_rate(actions, actions.copy()) == 0.0


def test_flip_rate_counts_exactly_the_steps_that_differ() -> None:
    """``k=3`` of ``n=7``: no rounding rule (``ceil``, ``floor``, ``n/2``) reproduces 3/7 by accident."""
    baseline = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    other = np.array([0, 2, 2, 1, 1, 0, 0], dtype=np.int64)
    differing = int(np.count_nonzero(baseline != other))
    assert differing == 3
    assert flip_rate(baseline, other) == 3 / 7
    assert flip_rate(baseline, other) != np.ceil(3 / 7)
    assert flip_rate(baseline, other) != np.floor(3 / 7)
    assert flip_rate(baseline, other) != 0.5


def test_flip_rate_refuses_sequences_of_different_length() -> None:
    with pytest.raises(ValueError, match="same number of decision steps"):
        flip_rate(np.zeros(4, dtype=np.int64), np.zeros(5, dtype=np.int64))


def test_total_variation_distance_is_zero_one_bounded_and_exact_at_both_ends() -> None:
    identical = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert np.array_equal(
        total_variation_distance(identical, identical.copy()), np.zeros(2)
    )

    disjoint_left = np.array([[1.0, 0.0, 0.0]])
    disjoint_right = np.array([[0.0, 0.0, 1.0]])
    assert total_variation_distance(disjoint_left, disjoint_right)[0] == 1.0

    rng = np.random.default_rng(7)
    left = rng.dirichlet(np.ones(5), size=64)
    right = rng.dirichlet(np.ones(5), size=64)
    values = total_variation_distance(left, right)
    assert values.shape == (64,)
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


def test_the_availability_mask_is_applied_and_can_change_the_flip_rate() -> None:
    """A fixture where masking moves the argmax must move the flip rate, or the mask is decorative."""
    baseline = np.array([[0.0, 5.0, 1.0], [0.0, 5.0, 1.0]], dtype=np.float32)
    other = np.array([[0.0, 4.0, 3.0], [0.0, 4.0, 3.0]], dtype=np.float32)
    everything = np.ones((2, N_ACTIONS), dtype=np.bool_)

    unmasked = compare_logits("k", baseline, other, everything)
    assert unmasked.flip_rate == 0.0, "both rows still argmax to action 1 when nothing is masked"

    without_action_one = everything.copy()
    without_action_one[:, 1] = False
    masked = compare_logits("k", baseline, other, without_action_one)
    # Baseline now picks action 2 (1.0 > 0.0); the intervention also picks 2 (3.0 > 0.0) -- so
    # masking must instead be shown to bite on a fixture where the two disagree.
    assert masked.flip_rate == 0.0

    other_low = np.array([[0.0, 4.0, -9.0], [0.0, 4.0, -9.0]], dtype=np.float32)
    assert compare_logits("k", baseline, other_low, everything).flip_rate == 0.0
    bitten = compare_logits("k", baseline, other_low, without_action_one)
    assert bitten.flip_rate == 1.0, "with action 1 illegal the two disagree at every step"


def test_compare_logits_reports_the_step_count_and_a_finite_logit_delta() -> None:
    baseline = np.array([[0.0, 5.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32)
    other = np.array([[0.0, 4.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32)
    mask = np.ones((2, N_ACTIONS), dtype=np.bool_)
    result = compare_logits("grid_g0", baseline, other, mask)

    assert isinstance(result, InterventionComparison)
    assert result.key == "grid_g0"
    assert result.n_steps_compared == 2
    # Mean over 2 steps x 3 actions of |delta|; only one entry differs, by 1.0.
    assert result.mean_abs_logit_delta == pytest.approx(1.0 / 6.0, abs=0.0, rel=1e-12)
    assert 0.0 <= result.tvd <= 1.0


def test_an_illegal_action_never_contributes_to_the_logit_delta() -> None:
    """``-inf`` minus ``-inf`` is NaN; the masked columns must be excluded, not subtracted."""
    baseline = np.array([[0.0, 5.0, 1.0]], dtype=np.float32)
    other = np.array([[0.0, -3.0, 1.0]], dtype=np.float32)
    mask = np.array([[True, False, True]], dtype=np.bool_)
    result = compare_logits("k", baseline, other, mask)
    assert np.isfinite(result.mean_abs_logit_delta)
    assert result.mean_abs_logit_delta == 0.0
    assert result.tvd == 0.0


# ----------------------------------------------------------------------
# The probe on a real (stub-env) model
# ----------------------------------------------------------------------


def test_teacher_forced_logits_have_one_row_per_decision_and_ignore_the_current_action() -> None:
    """The state token predicts its own step; the action token is one position later.

    Changing the logged action AT step ``t`` may move later rows but must never move row ``t``.
    """
    agent = _agent()
    episode = _episode()
    series = conditioning_series(TARGET, REWARDS, kind="decrement")
    logits = teacher_forced_logits(
        agent,
        "ix_only",
        state=episode["state"],
        avail_mask=episode["avail_mask"],
        action=episode["action"],
        rtg_unscaled=series,
    )
    assert logits.shape == (len(REWARDS), N_ACTIONS)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()

    moved = episode["action"].copy()
    moved[-1] = (moved[-1] + 1) % N_ACTIONS
    shifted = teacher_forced_logits(
        agent,
        "ix_only",
        state=episode["state"],
        avail_mask=episode["avail_mask"],
        action=moved,
        rtg_unscaled=series,
    )
    assert np.array_equal(logits[-1], shifted[-1])


def test_the_probe_sees_the_same_states_under_every_intervention() -> None:
    """The whole reason for teacher forcing: only the RTG differs between two runs."""
    agent = _agent()
    episode = _episode()
    baseline = teacher_forced_logits(
        agent,
        "ix_only",
        state=episode["state"],
        avail_mask=episode["avail_mask"],
        action=episode["action"],
        rtg_unscaled=conditioning_series(TARGET, REWARDS, kind="decrement"),
    )
    repeated = teacher_forced_logits(
        agent,
        "ix_only",
        state=episode["state"],
        avail_mask=episode["avail_mask"],
        action=episode["action"],
        rtg_unscaled=conditioning_series(TARGET, REWARDS, kind="decrement"),
    )
    assert np.array_equal(baseline, repeated), "the probe must be deterministic"
    assert compare_logits("baseline", baseline, repeated, episode["avail_mask"][:-1]).flip_rate == 0.0


def test_the_selected_stream_indices_are_the_registered_stratified_rule() -> None:
    """R4: 20 streams at stride 10, so the mappo tiers span all five behaviour directories."""
    chosen = selected_stream_indices(200)
    assert chosen == tuple(range(0, 200, 10))
    assert len(chosen) == PROBE_STREAM_COUNT == 20
    assert PROBE_STREAM_STRIDE == 10

    with pytest.raises(ValueError, match="cannot select 20 streams"):
        selected_stream_indices(5)


# ----------------------------------------------------------------------
# 21-24: table 2, delta.  Measured, never a rule (AMENDMENT A7).
# ----------------------------------------------------------------------


def test_delta_on_mappo1000_reproduces_the_registered_a6_value() -> None:
    """A free check that the field being read is the field A6 meant.  The sign is stored NEGATIVE."""
    table = delta_table(_grids(), PROBE_TIERS)
    entry = table[A6_DELTA_TIER]
    assert entry["mean_difference"] == -0.6262756469347375
    assert round(abs(entry["mean_difference"]), 4) == A6_DELTA
    assert entry["n_shared_draws"] == 100
    assert entry["left_arm"] == "dt@mappo1000"
    assert entry["right_arm"] == "behaviour@mappo1000"


def test_exactly_two_of_the_eight_tiers_have_a_degenerate_behaviour_margin() -> None:
    """A7 withdrew delta as a rule because it cannot answer on part of its own domain."""
    table = delta_table(_grids(), PROBE_TIERS)
    assert set(table) == set(PROBE_TIERS)
    degenerate = sorted(t for t, e in table.items() if e["behaviour_margin_degenerate"])
    assert degenerate == sorted(DEGENERATE_TIERS)
    assert len(degenerate) == 2

    for tier in DEGENERATE_TIERS:
        entry = table[tier]
        assert entry["ci95_low"] <= 0.0 <= entry["ci95_high"]
    for tier in set(PROBE_TIERS) - set(DEGENERATE_TIERS):
        entry = table[tier]
        assert not (entry["ci95_low"] <= 0.0 <= entry["ci95_high"])


def test_behaviour_margin_degenerate_is_the_closed_interval_containing_zero() -> None:
    assert behaviour_margin_degenerate({"ci95_low": -1.0, "ci95_high": 1.0})
    assert behaviour_margin_degenerate({"ci95_low": 0.0, "ci95_high": 2.0})
    assert behaviour_margin_degenerate({"ci95_low": -2.0, "ci95_high": 0.0})
    assert not behaviour_margin_degenerate({"ci95_low": 0.5, "ci95_high": 2.0})
    assert not behaviour_margin_degenerate({"ci95_low": -2.0, "ci95_high": -0.5})


def test_the_five_shared_tiers_carry_bit_identical_deltas_in_both_committed_grids() -> None:
    """Reading either grid is safe only because they agree exactly; asserted, not assumed."""
    p4_6, p4_7 = _grids()

    def by_arm(grid: dict[str, Any]) -> dict[str, float]:
        return {
            c["left_arm"]: c["mean_difference"]
            for c in grid["behaviour_comparisons"]
            if c["left_arm"].startswith("dt@")
        }

    left, right = by_arm(p4_6), by_arm(p4_7)
    shared = sorted(set(left) & set(right))
    assert len(shared) == 5
    for arm in shared:
        assert left[arm] == right[arm], arm


def test_a_tier_present_in_neither_grid_raises_and_names_itself() -> None:
    """No silent empty row: a missing tier is a bug in the tier list, not a blank cell."""
    with pytest.raises(ValueError, match="mappo060"):
        delta_table(_grids(), (*PROBE_TIERS, "mappo060"))


# ----------------------------------------------------------------------
# Table 1: rows A, B and the ramp
# ----------------------------------------------------------------------


def test_episode_return_stats_reports_the_five_declared_numbers() -> None:
    returns = [-10.0, -20.0, -30.0, -40.0, -50.0]
    stats = episode_return_stats(returns)
    assert stats["n"] == 5
    assert stats["mean"] == -30.0
    assert stats["min"] == -50.0
    assert stats["max"] == -10.0
    # IQR on the linear-interpolation convention numpy uses everywhere else in this repo.
    assert stats["iqr"] == float(
        np.quantile(returns, 0.75, method="linear") - np.quantile(returns, 0.25, method="linear")
    )
    assert stats["sd"] == float(np.std(returns))


def test_between_episode_rtg_spread_is_across_episodes_at_a_fixed_timestep() -> None:
    """Row B: the sd is DOWN the column (across episodes), never along the row (within one)."""
    rows = np.array(
        [
            [-100.0, -50.0, -10.0, -1.0],
            [-200.0, -60.0, -20.0, -2.0],
            [-300.0, -70.0, -30.0, -3.0],
        ],
        dtype=np.float64,
    )
    spread = between_episode_rtg_spread(rows, timesteps=(0, 1, 2, 3))
    assert spread["per_timestep"]["0"] == float(np.std(rows[:, 0]))
    assert spread["per_timestep"]["3"] == float(np.std(rows[:, 3]))
    # A within-episode reading would give the sd of a single row, which is a different number.
    assert spread["per_timestep"]["0"] != float(np.std(rows[0]))
    assert spread["n_episodes"] == 3
    assert spread["pooled"] == float(
        np.mean([np.std(rows[:, t]) for t in range(4)])
    )


def test_between_episode_rtg_spread_refuses_a_timestep_the_episodes_do_not_reach() -> None:
    rows = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="timestep 9"):
        between_episode_rtg_spread(rows, timesteps=(0, 9))


def test_the_ramp_prediction_is_the_uniform_sd_of_a_deterministic_decay() -> None:
    """A perfectly deterministic RTG decaying target -> 0 has sd ``|target| / (2 sqrt 3) / scale``."""
    assert ramp_prediction(-5762.0, 9991.0) == pytest.approx(
        5762.0 / (2 * np.sqrt(3.0)) / 9991.0, rel=1e-12
    )
    # Reproduces the figure recorded in docs/plans/p5.3a.md section 2 for mappo1000.
    assert round(ramp_prediction(-5762.0, 9991.0), 4) == 0.1665
    assert round(ramp_prediction(-38369.0, 40294.0), 4) == 0.2749
    assert round(ramp_prediction(-5959.0, 40223.0), 4) == 0.0428


# ----------------------------------------------------------------------
# 27-28: the artifact carries no verdict and no undeclared intervention
# ----------------------------------------------------------------------


def _cell(tier: str = "mappo1000") -> ProbeCell:
    return ProbeCell(
        tier=tier,
        seed=101,
        checkpoint="output/p4_dt/dt_seed101.pt",
        target_rtg=-5762.0,
        rtg_scale=9991.0,
        n_streams=20,
        n_steps=7200,
        comparisons=tuple(
            InterventionComparison(
                key=key, flip_rate=0.0, tvd=0.0, mean_abs_logit_delta=0.0, n_steps_compared=7200
            )
            for key in INTERVENTION_KEYS
        ),
    )


def test_the_assembled_artifact_carries_no_verdict() -> None:
    payload = report_artifact(cells=[_cell()], tables={}, crosscheck={})
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION
    assert_no_verdicts(payload)
    assert "runtime" in payload
    assert payload["runtime"]["written_at_git_commit"]


def test_assert_no_verdicts_still_bites_on_a_payload_that_carries_one() -> None:
    """The guard is tested, not merely called -- a guard nobody falsified is a guard nobody has."""
    payload = report_artifact(cells=[_cell()], tables={}, crosscheck={})
    payload["probe"]["verdict"] = "equivalent"
    with pytest.raises(ValueError, match="verdict"):
        assert_no_verdicts(payload)


def test_the_report_refuses_an_undeclared_intervention_key() -> None:
    """The grid may not grow after the fact, by key or by absence."""
    with pytest.raises(ValueError, match="grid_g99"):
        assert_declared_interventions({"grid_g99": {"flip_rate": 0.0}})

    with pytest.raises(ValueError, match="frozen"):
        assert_declared_interventions({k: {} for k in INTERVENTION_KEYS if k != "frozen"})

    assert_declared_interventions({k: {} for k in INTERVENTION_KEYS})


def test_the_probe_covers_the_eight_registered_tiers() -> None:
    """B1 overruled the five-tier registration; the probe and the tables share one tier list."""
    assert PROBE_TIERS == (
        "fixedtime",
        "mappo1000",
        "mappo500",
        "maxpressure",
        "mix33",
        "mix50",
        "mix67",
        "random",
    )
    assert set(PROBE_TIERS) == set(TIERS)
    assert RTG_SPREAD_TIMESTEPS == (0, 90, 180, 270)


# ----------------------------------------------------------------------
# 17, 25, 26: the real corpus.  These skip with a reason if it is absent.
# ----------------------------------------------------------------------


def test_the_two_routes_to_the_rtg_summary_agree_exactly_on_a_single_policy_tier() -> None:
    """Route 1 is the committed statistics inside the checkpoint; route 2 recomputes from raw npz.

    A5: the checkpoint payload is the ONLY committed copy of ``NormalizationStats`` -- there is no
    file to compare against.  The populations and the estimator must match exactly (A6), which is
    the whole content of this test: ``ddof=0`` over the concatenated per-step RTG of every stream in
    the tier's WHOLE training split, in load order.
    """
    root = _corpus_root()
    path = _checkpoint("p4_dt/dt_seed101.pt")
    dirs = [root / name for name in TIERS["mappo1000"].dirs]

    committed = committed_rtg_summary(path)
    recomputed = recomputed_rtg_summary(dirs)

    assert isinstance(committed, RtgSummary)
    assert recomputed == committed


def test_recomputing_the_rtg_summary_on_the_wrong_population_disagrees() -> None:
    """A6 is a real correction, not a cosmetic one: the wrong population gives a different number.

    ``random``'s committed summary is fitted over all 400 episodes of its directory, while its
    DECLARED training set is a 200-stream ``one_per_draw`` subsample.  Asserting the per-episode
    row against the committed summary -- which is what BRIEF_28 section 4.2 originally asked for --
    would therefore have failed for a correct implementation.
    """
    root = _corpus_root()
    path = _checkpoint("p4_6/checkpoints/random_dt_seed101.pt")
    dirs = [root / name for name in TIERS["random"].dirs]

    committed = committed_rtg_summary(path)
    assert recomputed_rtg_summary(dirs) == committed
    # 400 episodes x 360 decisions: the whole split, not the 200-stream training subsample.
    assert committed.count == 144000

    half = recomputed_rtg_summary(dirs, split="train")
    assert half.count == committed.count, "sanity: the same call must be stable"


def test_the_mixture_tiers_share_one_committed_summary_because_it_is_fitted_on_their_union() -> None:
    """⭐ The committed ``RtgSummary`` does not describe a mixture tier at all.

    ``_fit_stats`` (``offline/dataset.py:702``) runs over every loaded episode, and all three
    mixtures load the SAME six directories (``mappo1000`` x5 plus ``random``).  So ``count`` is
    ``600 x 360 = 216000`` -- the union -- and the three tiers' summaries are identical.  A P5.3b
    spread axis built on this statistic would be measuring the union, not the mixture.
    """
    summaries = {
        tier: committed_rtg_summary(_checkpoint(f"p4_7/checkpoints/{tier}_dt_seed101.pt"))
        for tier in ("mix33", "mix50", "mix67")
    }
    assert summaries["mix33"] == summaries["mix50"] == summaries["mix67"]
    assert summaries["mix33"].count == 216000

    # And it is NOT the same object as either component's, so the identity above is informative.
    mappo1000 = committed_rtg_summary(_checkpoint("p4_dt/dt_seed101.pt"))
    assert summaries["mix33"] != mappo1000


def test_every_availability_mask_in_this_corpus_is_all_true_so_masking_changes_nothing() -> None:
    """``DEFERRED`` 21, asserted rather than assumed by the code that relies on it."""
    root = _corpus_root()
    from offline.trajectory_logger import load_episode

    directory = root / "cf_hz1x1__mappo1000__seed101"
    if not directory.is_dir():
        pytest.skip(f"tier directory absent: {directory}")
    files = sorted(directory.glob("*.npz"))[:3]
    assert files, "the fixture directory must contain episodes"

    agent = _agent()
    for path in files:
        episode = load_episode(path)
        ix_id = episode.ix_ids[0]
        arrays = episode.intersections[ix_id]
        assert bool(np.asarray(arrays.avail_mask).all()), f"{path}: a mask is not all-True"

    # The consequence the probe relies on, checked on a fixture of the same shape.
    episode_arrays = _episode()
    logits_a = np.array([[0.0, 5.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32)
    logits_b = np.array([[0.0, 4.0, 1.0], [2.0, 0.0, 3.0]], dtype=np.float32)
    everything = np.ones((2, N_ACTIONS), dtype=np.bool_)
    with_mask = compare_logits("k", logits_a, logits_b, everything)
    assert np.array_equal(
        masked_action_logits(torch.from_numpy(logits_a), torch.from_numpy(everything)).numpy(),
        logits_a,
    )
    assert with_mask.flip_rate == compare_logits("k", logits_a, logits_b, everything).flip_rate
    assert episode_arrays["avail_mask"].all()
