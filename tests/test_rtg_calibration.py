"""Tests for ``offline.rtg_calibration`` -- P4.3's probe, its two rules and the RTG landscape.

No simulator and no corpus are required by anything here except the two tests that say so in their
names and skip themselves.  The committed artifacts that P4 already produced (``p4_gate.json``,
``p4_training.json``, ``p4_dt_config.json``) ARE used as fixtures: they are the evidence this task
must remain consistent with, and reading them is what makes the declaration checkable rather than
self-referential.

Every assertion is exact where the types allow it.  The float64 identity in
:func:`test_rule_b_is_the_identity_in_domain_including_a_value_where_reassociation_is_not` is exact
by construction and the test pins the association that makes it so -- ``7.06 %`` of random
real-valued probe statistics fail under the other one (measured, 2026-08-13, 200,000 draws).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from agent.DTAgent import DTAgent
from offline.dt_gate import (
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    EpisodeResult,
    mean_ci95,
    runtime_provenance,
)
from offline.offline_baselines import assert_campaign_complete, paired_comparison
from offline.rtg_calibration import (
    DECLARED_GRID,
    GRID_POINT_KEYS,
    NAIVE_TARGET,
    PROBE_DRAW_START,
    PROBE_K_VALUES,
    PROBE_STATISTICS,
    RULE_A_K,
    RULE_A_POINT_KEY,
    RULE_A_QUANTILE,
    SECONDARY_QUANTILES,
    PointRun,
    ProbeEpisode,
    SupportCounts,
    agent_with_target,
    assert_probe_draws_disjoint,
    effect_size_sidecar,
    episode_return_two_routes,
    gate_a_result,
    grid_targets,
    in_support_counts,
    leaf_diff,
    pearson_r,
    point_artifact,
    probe_artifact,
    probe_draw_ids,
    probe_statistic,
    report_artifact,
    requested_runs,
    rule_a_target,
    rule_b_target,
    source_domain_ratio,
    training_rtg_range,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"

# The fixture DT: deliberately tiny, and a power-of-two RTG scale so `/ rtg_scale` is exact.
FIXTURE_STATE_DIM = 4
FIXTURE_N_ACTIONS = 3
FIXTURE_SCALE = 64.0
FIXTURE_STEPS = 4242


class _StubIntersection:
    def __init__(self, ix_id: str, num_phases: int, incoming_lanes: Sequence[str] = ()) -> None:
        self.id = ix_id
        self.num_phases = num_phases
        self.incoming_lanes = list(incoming_lanes)


class _StubEnv:
    """The smallest object ``DTAgent.__init__`` reads: intersections and an action space."""

    def __init__(self, specs: Sequence[tuple[str, int]], max_steps: int = 8) -> None:
        self.intersections = [_StubIntersection(i, n) for i, n in specs]
        self.max_steps = max_steps


def _info(step: int, state: Sequence[float], reward: float) -> dict[str, Any]:
    return {
        "step": step,
        "vehicle_count": 0,
        "intersections": {
            "ix_only": {
                "state": list(state),
                "avail_actions": list(range(FIXTURE_N_ACTIONS)),
                "reward": float(reward),
            }
        },
    }


def _write_checkpoint(path: Path, *, target_rtg: float, steps: int = FIXTURE_STEPS) -> None:
    """A real ``DTAgent`` checkpoint, saved through the real ``save()``."""
    agent = DTAgent(
        _StubEnv([("ix_only", FIXTURE_N_ACTIONS)]),
        context_length=4,
        n_layer=1,
        n_head=1,
        d_model=16,
        dropout=0.0,
        max_ep_len=32,
        target_rtg=target_rtg,
        rtg_scale=FIXTURE_SCALE,
        device="cpu",
        seed=11,
        state_dim=FIXTURE_STATE_DIM,
    )
    agent.save(str(path), provenance={"gradient_steps": int(steps)})


def _episode_result(arm: str, seed: int, draw: int, att: float, vehicles: float = 7.0,
                    reward: float = -1234.0) -> EpisodeResult:
    return EpisodeResult(
        arm=arm,
        seed=seed,
        draw_id=draw,
        att_horizon=att,
        horizon_vehicle_count=vehicles,
        episode_reward=reward,
    )


# ----------------------------------------------------------------------
# T1 -- the load-bearing one: the override must reach the MODEL
# ----------------------------------------------------------------------


def test_the_rtg_reaching_the_model_uses_the_OVERRIDE_not_the_checkpoints_target(
    tmp_path: Path,
) -> None:
    """The value spied at ``forward`` must be ``(override - sum rewards) / rtg_scale``, every step.

    THE TEST THIS MODULE EXISTS TO PASS.  ``DTAgent.load`` overwrites ``_target_rtg`` from the
    checkpoint (``agent/DTAgent.py:803``), so applying the override before the load leaves every
    grid point running P4's original target: the whole campaign becomes one point repeated, with
    plausible numbers and different labels.  Reading ``_target_rtg`` back would not catch it --
    the attribute is right in both orders if it is set last; what matters is which value the
    model was fed, so this test asserts on the tensor the model received.

    Exact by construction: ``rtg_scale`` is a power of two and every reward is integral, so
    ``(override - sum) / 64`` is representable in float32 and the assertion is ``==``.
    """
    path = tmp_path / "dt.pt"
    checkpoint_target = -100.0
    override = -1000.0
    _write_checkpoint(path, target_rtg=checkpoint_target)
    assert checkpoint_target != override, "the fixture must distinguish the two targets"

    env = _StubEnv([("ix_only", FIXTURE_N_ACTIONS)])
    agent = agent_with_target(
        env,
        path,
        declared_gradient_steps=FIXTURE_STEPS,
        target_rtg=override,
        device="cpu",
    )

    captured: list[torch.Tensor] = []
    original = agent.model.forward

    def spy(rtg, state, action, timestep, attention_mask=None, avail_mask=None):  # type: ignore[no-untyped-def]
        captured.append(rtg.detach().clone())
        return original(rtg, state, action, timestep, attention_mask, avail_mask)

    agent.model.forward = spy  # type: ignore[method-assign]

    rewards = [-3.0, -11.0, -7.0, -2.0, -19.0, -5.0]
    agent.act(_info(0, [0.0, 0.0, 0.0, 0.0], 0.0), explore=False)
    for step, reward in enumerate(rewards, start=1):
        agent.act(_info(step, [1.0, 2.0, 3.0, 4.0], reward), explore=False)

    assert len(captured) == len(rewards) + 1
    for step in range(len(rewards) + 1):
        expected = (override - math.fsum(rewards[:step])) / FIXTURE_SCALE
        got = float(captured[step][0, -1, 0])
        assert got == expected, f"step {step}: model saw {got}, expected {expected}"

    # The checkpoint's own target must appear nowhere in the trajectory the model saw.
    forbidden = checkpoint_target / FIXTURE_SCALE
    assert float(captured[0][0, -1, 0]) != forbidden


@pytest.mark.skipif(
    not os.environ.get("RLTRAFFIC_P4_DT_CHECKPOINTS"),
    reason="RLTRAFFIC_P4_DT_CHECKPOINTS is not set; the committed P4 checkpoints are not in tree",
)
def test_the_override_reaches_the_model_on_a_real_committed_checkpoint(tmp_path: Path) -> None:
    """T1 again, on the real weights the campaign runs, through the real loader path.

    The fixture version proves the ordering; this one proves it survives a 40,000-step checkpoint
    carrying frozen normalisation statistics, which is the configuration the campaign uses and the
    one whose normalisation branch P4's review found had never executed in a test.
    """
    root = Path(os.environ["RLTRAFFIC_P4_DT_CHECKPOINTS"])
    checkpoint = root / "dt_seed101.pt"
    if not checkpoint.is_file():
        pytest.skip(f"no checkpoint at {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dim = int(payload["config"]["state_dim"])
    n_actions = int(payload["config"]["n_actions"])
    scale = float(payload["rtg_scale"])
    override = -4321.0
    assert override != float(payload["target_rtg"])

    env = _StubEnv([("intersection_1_1", n_actions)])
    agent = agent_with_target(
        env,
        checkpoint,
        declared_gradient_steps=int(payload["provenance"]["gradient_steps"]),
        target_rtg=override,
        device="cpu",
    )

    captured: list[torch.Tensor] = []
    original = agent.model.forward

    def spy(rtg, state, action, timestep, attention_mask=None, avail_mask=None):  # type: ignore[no-untyped-def]
        captured.append(rtg.detach().clone())
        return original(rtg, state, action, timestep, attention_mask, avail_mask)

    agent.model.forward = spy  # type: ignore[method-assign]

    rewards = [-13.0, -21.0, -34.0]
    state = [float(i) for i in range(state_dim)]
    info = {
        "step": 0,
        "vehicle_count": 0,
        "intersections": {
            "intersection_1_1": {
                "state": state,
                "avail_actions": list(range(n_actions)),
                "reward": 0.0,
            }
        },
    }
    agent.act(info, explore=False)
    for step, reward in enumerate(rewards, start=1):
        info = json.loads(json.dumps(info))
        info["step"] = step
        info["intersections"]["intersection_1_1"]["reward"] = reward
        agent.act(info, explore=False)

    for step in range(len(rewards) + 1):
        # EXACT, not a tolerance: `_window` stores the quotient as float32
        # (agent/DTAgent.py:596), so the value the model receives is the float32 round-trip of
        # the float64 quotient.  Asserting against that pins the storage convention as well as
        # the arithmetic; asserting the float64 value with a tolerance would pin neither, and
        # this checkpoint's rtg_scale (9991.0) is not a power of two, so the two differ by up to
        # 1e-8 -- measured 5.8e-9 at step 0.
        expected = float(np.float32((override - math.fsum(rewards[:step])) / scale))
        assert float(captured[step][0, -1, 0]) == expected, f"step {step}"
    assert float(captured[0][0, -1, 0]) != float(payload["target_rtg"]) / scale


def test_agent_with_target_leaves_the_checkpoints_rtg_scale_untouched(tmp_path: Path) -> None:
    """T5: only the target varies in this task; ``rtg_scale`` is the normalisation divisor."""
    path = tmp_path / "dt.pt"
    _write_checkpoint(path, target_rtg=-100.0)
    env = _StubEnv([("ix_only", FIXTURE_N_ACTIONS)])

    for target in (0.0, -5762.0, -13000.0):
        agent = agent_with_target(
            env, path, declared_gradient_steps=FIXTURE_STEPS, target_rtg=target, device="cpu"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert float(payload["rtg_scale"]) == FIXTURE_SCALE
        # By effect, not by attribute: at step 0 the model input is target / scale.
        captured: list[torch.Tensor] = []
        original = agent.model.forward

        def spy(rtg, state, action, timestep, attention_mask=None, avail_mask=None, _c=captured, _o=original):  # type: ignore[no-untyped-def]
            _c.append(rtg.detach().clone())
            return _o(rtg, state, action, timestep, attention_mask, avail_mask)

        agent.model.forward = spy  # type: ignore[method-assign]
        agent.act(_info(0, [0.0, 0.0, 0.0, 0.0], 0.0), explore=False)
        assert float(captured[0][0, -1, 0]) == target / FIXTURE_SCALE


def test_a_checkpoint_saved_at_another_step_count_is_refused(tmp_path: Path) -> None:
    """The no-online-selection guard must survive the override wrapper, not be bypassed by it."""
    path = tmp_path / "dt.pt"
    _write_checkpoint(path, target_rtg=-100.0, steps=999)
    with pytest.raises(ValueError, match="checkpoint was saved at"):
        agent_with_target(
            _StubEnv([("ix_only", FIXTURE_N_ACTIONS)]),
            path,
            declared_gradient_steps=FIXTURE_STEPS,
            target_rtg=-1.0,
            device="cpu",
        )


# ----------------------------------------------------------------------
# T2 -- the probe band, asserted against what the corpus ACTUALLY used
# ----------------------------------------------------------------------


def test_the_probe_draws_are_disjoint_from_the_corpus_and_the_held_out_pool() -> None:
    """T2: disjointness is asserted against `p4_training.json`, not against a pool boundary.

    ``PREREGISTRATION.md`` section 5 registers 1-999 as the training pool, so a pool-boundary check
    would happily accept draw 200 -- which the corpus used.  The artifact naming the draws that
    were actually trained on is the only reference that can refuse it.
    """
    training = json.loads((DATA / "p4_training.json").read_text(encoding="utf-8"))
    used = training["training_draw_ids"]
    assert used[0] == 1 and used[-1] == 200 and len(used) == 200

    for k in PROBE_K_VALUES:
        draws = probe_draw_ids(k)
        assert len(draws) == k
        assert draws[0] == PROBE_DRAW_START
        assert_probe_draws_disjoint(
            draws, training_draw_ids=used, held_out_draws=HELD_OUT_DRAWS
        )
        assert not set(draws) & set(used)
        assert not set(draws) & set(HELD_OUT_DRAWS)


def test_the_probe_budgets_are_nested_so_the_k_ablation_is_a_prefix_ablation() -> None:
    """k=5's draws are a prefix of k=20's, which are a prefix of k=100's."""
    small, mid, large = (probe_draw_ids(k) for k in PROBE_K_VALUES)
    assert large[: len(mid)] == mid
    assert mid[: len(small)] == small


@pytest.mark.parametrize(
    "draws",
    [
        (200, 201, 202),          # touches the corpus
        (1000, 1001),             # touches the held-out pool
        (199,),                   # inside the corpus
    ],
)
def test_a_probe_touching_training_or_held_out_draws_is_refused(draws: tuple[int, ...]) -> None:
    training = json.loads((DATA / "p4_training.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not disjoint"):
        assert_probe_draws_disjoint(
            draws,
            training_draw_ids=training["training_draw_ids"],
            held_out_draws=HELD_OUT_DRAWS,
        )


# ----------------------------------------------------------------------
# T9 -- the probe return, by two routes that can disagree
# ----------------------------------------------------------------------


def test_the_two_return_routes_are_computed_independently_and_can_differ() -> None:
    """T9: the fixture is DELIBERATELY inconsistent, so returning one route twice is visible.

    On a real rollout the two routes agree exactly -- that is the identity
    ``local_reward == -sum(lane_waiting over incoming lanes)`` verified in
    ``metrics/base.py:329-345`` and ``rewards.py:48``.  A test built on a consistent fixture
    could not tell the two routes apart, so this one breaks the identity on purpose: the reward
    stream says -7 per step while the intersection's own lanes hold 5 waiting vehicles, and a
    third lane that belongs to no controlled intersection holds 100 (an implementation summing
    every lane in the info would report -103 per step).
    """
    infos = [
        {
            "step": t,
            "intersections": {"ix": {"reward": -7.0}},
            "lane_waiting_vehicle_count": {"in_a": 2, "in_b": 3, "elsewhere": 100},
        }
        for t in range(1, 6)
    ]
    from_rewards, from_lanes = episode_return_two_routes(
        infos, ix_id="ix", incoming_lanes=("in_a", "in_b")
    )
    assert from_rewards == -35.0                 # 5 steps x -7
    assert from_lanes == -25.0                   # 5 steps x -(2 + 3)
    assert from_rewards != from_lanes


def test_the_probe_artifact_refuses_an_episode_whose_two_routes_disagree() -> None:
    """A disagreement means the reward the RTG advances on is not the queue it claims to be."""
    good = ProbeEpisode(
        draw_id=201,
        local_return=-12000.0,
        local_return_from_lanes=-12000.0,
        att_horizon=176.0,
        horizon_vehicle_count=40.0,
        decisions=360,
    )
    bad = ProbeEpisode(
        draw_id=202,
        local_return=-12000.0,
        local_return_from_lanes=-11999.0,
        att_horizon=176.0,
        horizon_vehicle_count=40.0,
        decisions=360,
    )
    training = json.loads((DATA / "p4_training.json").read_text(encoding="utf-8"))
    kwargs: dict[str, Any] = {
        "training_draw_ids": training["training_draw_ids"],
        "best_source_return": NAIVE_TARGET,
        "rtg_range": (-9991.0, -6.0),
        "rtg_scale": 9991.0,
        "env_settings": {"max_steps": 360},
        "engine_seed": 1000,
        "scenario_id": "cityflow1x1",
    }
    payload = probe_artifact(episodes=[good], **kwargs)
    assert payload["n_episodes"] == 1

    with pytest.raises(ValueError, match="disagree between their two routes"):
        probe_artifact(episodes=[good, bad], **kwargs)


# ----------------------------------------------------------------------
# T4 / T11 / T12 -- the rules
# ----------------------------------------------------------------------


def test_rule_a_is_a_pure_function_of_the_probe_returns() -> None:
    """T4: a hand-computed answer, repeatability, and a signature that admits no outcome."""
    returns = [-30.0, -10.0, -20.0, -40.0]
    assert rule_a_target(returns, 1.0) == -10.0
    assert rule_a_target(returns, 0.0) == -40.0
    # numpy's linear interpolation on the sorted [-40, -30, -20, -10] at q = 0.5 is -25.
    assert rule_a_target(returns, 0.5) == -25.0
    assert rule_a_target(returns, 1.0) == rule_a_target(list(reversed(returns)), 1.0)

    names = set(inspect.signature(rule_a_target).parameters)
    assert names == {"returns", "quantile"}, (
        f"rule A must be a function of the probe returns alone, got {sorted(names)}"
    )


def test_rule_a_matches_an_independent_quantile_computation() -> None:
    """Recomputed by sorting and interpolating by hand, not by calling numpy again."""
    rng = np.random.default_rng(430)
    returns = list(-rng.uniform(5000.0, 20000.0, size=37))
    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        ordered = sorted(returns)
        position = q * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        expected = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        assert rule_a_target(returns, q) == pytest.approx(expected, rel=0.0, abs=1e-9)


def test_rule_a_refuses_an_empty_probe() -> None:
    with pytest.raises(ValueError, match="rule A is a quantile"):
        rule_a_target([], 1.0)


@pytest.mark.parametrize(
    "statistic_value",
    [
        -13112.0,                 # integral: a max of integral returns
        -9991.0,
        -14523.7,
        -18823.369238945274,      # measured counterexample: s * (best / s) != best here
        -8688.988980663957,
        -4560.384055280482,
    ],
)
def test_rule_b_is_the_identity_in_domain_including_a_value_where_reassociation_is_not(
    statistic_value: float,
) -> None:
    """T11 (BRIEF_15 section 12.4): in-domain the ratio is 1, so Rule B RETURNS the naive target.

    Asserted, not argued -- and the association is pinned, because it is what makes the identity
    exact.  ``best * (s / s)`` is exact for every ``s`` (the inner division yields exactly 1.0);
    ``s * (best / s)`` is not, and fails for **7.06 %** of random real statistics (measured over
    200,000 draws, 2026-08-13).  Three of the parametrised values are members of that failing set,
    so an implementation that reassociates dies here rather than in P7.
    """
    best = NAIVE_TARGET
    assert rule_b_target(
        best_source_return=best,
        probe_source_stat=statistic_value,
        probe_target_stat=statistic_value,
    ) == best


def test_the_reassociated_form_really_is_inexact_for_the_pinned_values() -> None:
    """A positive control for the test above: if this fails, that test proves nothing.

    Without it, the parametrised identity could be satisfied by any implementation and the
    "pins the association" claim would be decoration.
    """
    best = NAIVE_TARGET
    failing = [-18823.369238945274, -8688.988980663957, -4560.384055280482]
    for value in failing:
        assert value * (best / value) != best, f"{value} no longer discriminates"
        assert best * (value / value) == best


def test_rule_b_moves_the_target_with_the_ratio_when_the_domains_differ() -> None:
    """The mechanism: a target domain whose probe scores worse asks for proportionally less."""
    best = -6000.0
    got = rule_b_target(
        best_source_return=best, probe_source_stat=-12000.0, probe_target_stat=-18000.0
    )
    assert got == -9000.0                      # 1.5x worse probe -> 1.5x more negative target


def test_rule_b_refuses_a_zero_source_statistic() -> None:
    with pytest.raises(ValueError, match=r"Rule B\'s ratio is undefined"):
        rule_b_target(best_source_return=-1.0, probe_source_stat=0.0, probe_target_stat=-1.0)


def test_the_source_domain_ratio_is_reported_for_both_statistics_and_every_k() -> None:
    """T12: six numbers, hand-computed, with max and mean genuinely distinguished."""
    returns = [-10.0, -20.0, -30.0, -40.0]
    assert probe_statistic(returns, "max") == -10.0
    assert probe_statistic(returns, "mean") == -25.0
    assert source_domain_ratio(returns, best_source_return=-5.0, statistic="max") == 0.5
    assert source_domain_ratio(returns, best_source_return=-5.0, statistic="mean") == 0.2
    with pytest.raises(ValueError, match="unknown probe statistic"):
        probe_statistic(returns, "median")


def test_the_secondary_quantiles_are_declared_and_rule_a_evaluates_only_its_own_point() -> None:
    """BRIEF_15 section 12.6: the secondary quantiles are a labelling of the grid, not points on it."""
    assert SECONDARY_QUANTILES == (0.5, 0.75, 0.9)
    assert RULE_A_QUANTILE == 1.0 and RULE_A_K == 100
    assert RULE_A_POINT_KEY not in GRID_POINT_KEYS


# ----------------------------------------------------------------------
# T3 -- the in-support diagnostic
# ----------------------------------------------------------------------


def test_in_support_counts_match_an_independent_recomputation_and_partition_the_decisions() -> None:
    """T3: recomputed by a different route, and the three counts must sum to n."""
    lo, hi = -9991.0, -6.0
    target = -5762.0
    rng = np.random.default_rng(43)
    rewards = [-float(v) for v in rng.integers(1, 60, size=360)]
    series = [target - math.fsum(rewards[:t]) for t in range(360)]

    counts = in_support_counts(series, rtg_min=lo, rtg_max=hi)

    expected_in = sum(1 for v in series if not (v < lo or v > hi))
    expected_below = sum(1 for v in series if v < lo)
    expected_above = sum(1 for v in series if v > hi)
    assert (counts.in_support, counts.below, counts.above) == (
        expected_in,
        expected_below,
        expected_above,
    )
    assert counts.n == 360
    assert counts.in_support + counts.below + counts.above == counts.n
    assert counts.fraction == expected_in / 360
    assert counts.rtg_first == series[0]
    assert counts.rtg_last == series[-1]
    assert counts.rtg_min == min(series)
    assert counts.rtg_max == max(series)


def test_the_support_interval_is_closed_at_both_ends() -> None:
    """A value exactly ON either boundary is IN support; one ulp outside is not.

    The boundary is not decoration: the training minimum ``-9991`` is a target on the declared
    grid, so an episode really does start exactly on it.
    """
    lo, hi = -9991.0, -6.0
    counts = in_support_counts([lo, hi], rtg_min=lo, rtg_max=hi)
    assert (counts.in_support, counts.below, counts.above) == (2, 0, 0)

    outside = in_support_counts(
        [math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)], rtg_min=lo, rtg_max=hi
    )
    assert (outside.in_support, outside.below, outside.above) == (0, 1, 1)


def test_the_naive_and_zero_targets_have_the_diagnostic_the_withdrawal_rests_on() -> None:
    """``target = 0`` is 0.000 in-support -- the measurement that withdrew the selector.

    Registered prediction 1 of ``docs/plans/p4.3.md`` section 5, computed here from the rewards
    alone so that it is a property of the CONDITIONING RULE and not of any rollout: with every
    reward <= 0 the trajectory starts at the target and never decreases, so a target above the
    support can never enter it.
    """
    lo, hi = -9991.0, -6.0
    rewards = [-20.0] * 360
    zero_series = [0.0 - math.fsum(rewards[:t]) for t in range(360)]
    assert in_support_counts(zero_series, rtg_min=lo, rtg_max=hi).fraction == 0.0

    naive_series = [-5762.0 - math.fsum(rewards[:t]) for t in range(360)]
    naive = in_support_counts(naive_series, rtg_min=lo, rtg_max=hi)
    assert 0.0 < naive.fraction < 1.0
    assert naive.below == 0                       # it starts inside and walks upward
    assert naive.above > 0


def test_the_training_rtg_range_is_read_from_the_checkpoints_own_statistics(
    tmp_path: Path,
) -> None:
    """The support must come from the frozen stats, not from a constant restated in this module."""
    path = tmp_path / "dt.pt"
    _write_checkpoint(path, target_rtg=-100.0)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["stats"] = {
        "stats_version": "test",
        "split": "train",
        "draw_ids": [1, 2],
        "dataset_dirs": ["x"],
        "state_mean": {"sc": {"ix_only": [0.0] * FIXTURE_STATE_DIM}},
        "state_std": {"sc": {"ix_only": [1.0] * FIXTURE_STATE_DIM}},
        "row_count": {"sc": {"ix_only": 2}},
        "rtg": {
            "sc": {
                "ix_only": {
                    "count": 2,
                    "min": -4242.0,
                    "max": -17.0,
                    "mean": -100.0,
                    "std": 1.0,
                    "quantiles": [[0.5, -100.0]],
                }
            }
        },
    }
    torch.save(payload, path)
    assert training_rtg_range(path) == (-4242.0, -17.0)


def test_a_checkpoint_carrying_no_statistics_cannot_supply_a_support_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dt.pt"
    _write_checkpoint(path, target_rtg=-100.0)          # built without stats= -> stats is None
    with pytest.raises(ValueError, match="frozen training statistics"):
        training_rtg_range(path)


# ----------------------------------------------------------------------
# T6 / T7 -- the declaration and the campaign
# ----------------------------------------------------------------------


def test_the_grid_is_the_declared_grid_and_carries_the_naive_target() -> None:
    """T6: nine declared points, the naive one among them, keys stable and positional."""
    assert len(DECLARED_GRID) == 9
    assert len(GRID_POINT_KEYS) == len(DECLARED_GRID)
    assert NAIVE_TARGET in DECLARED_GRID
    assert DECLARED_GRID == tuple(sorted(DECLARED_GRID, reverse=True)), "declared in descending order"

    targets = grid_targets()
    assert tuple(targets) == GRID_POINT_KEYS
    assert tuple(targets.values()) == DECLARED_GRID


def test_the_naive_target_equals_the_configuration_p4_actually_reported() -> None:
    """The declared constant is checked against P4's committed configuration artifact.

    If these ever disagree, the grid is no longer anchored on the naive rule and Gate A is
    comparing two different configurations.
    """
    config = json.loads((DATA / "p4_dt_config.json").read_text(encoding="utf-8"))
    assert float(config["conditioning"]["target_rtg"]) == NAIVE_TARGET
    training = json.loads((DATA / "p4_training.json").read_text(encoding="utf-8"))
    assert float(training["target_rtg"]) == NAIVE_TARGET


def test_an_undeclared_point_cannot_enter_the_report() -> None:
    """T6's mutation guard: a grid derived from results is refused, by key AND by value."""
    probe = {"format_version": "p4.3-rtg/1.0"}
    common = {"probe": probe, "rtg_range": (-9991.0, -6.0), "checkpoints": {101: "a.pt"}}

    undeclared_key = [{"point_key": "dt_g99", "target_rtg": -1234.0, "rtg_scale": 9991.0}]
    with pytest.raises(ValueError, match="may not enter the report"):
        report_artifact(points=undeclared_key, **common)

    moved_value = [{"point_key": "dt_g3", "target_rtg": -3333.0, "rtg_scale": 9991.0}]
    with pytest.raises(ValueError, match="declared target"):
        report_artifact(points=moved_value, **common)


def test_the_requested_run_set_comes_from_the_declaration_not_from_the_data() -> None:
    """T7: completeness is only meaningful if the expectation is independent of the result."""
    requested = requested_runs("dt_g0", (101, 202), (1000, 1001, 1002))
    assert len(requested) == 6
    assert requested[0] == ("dt_g0", 101, 1000)
    assert set(requested) == {("dt_g0", s, d) for s in (101, 202) for d in (1000, 1001, 1002)}

    produced = [_episode_result("dt_g0", s, d, 100.0) for s in (101, 202) for d in (1000, 1001, 1002)]
    assert_campaign_complete(requested, produced)

    with pytest.raises(ValueError, match="incomplete campaign"):
        assert_campaign_complete(requested, produced[:-1])


def test_the_full_campaign_is_five_seeds_over_the_whole_registered_held_out_pool() -> None:
    """The declared design, restated where a test can see it: 5 x 100 per point."""
    requested = requested_runs("dt_g5", TRAINING_SEEDS, HELD_OUT_DRAWS)
    assert len(requested) == 500
    assert len({d for _, _, d in requested}) == 100
    assert min(d for _, _, d in requested) == 1000
    assert max(d for _, _, d in requested) == 1099


# ----------------------------------------------------------------------
# T8 -- Gate A
# ----------------------------------------------------------------------


def _reference_file(tmp_path: Path, episodes: Sequence[EpisodeResult]) -> Path:
    payload = {
        "format_version": "p4-gate/1.0",
        "cells": {"madt": {"att_horizon_mean": float(np.mean([e.att_horizon for e in episodes]))}},
        "episodes": [
            {
                "arm": e.arm,
                "seed": e.seed,
                "draw_id": e.draw_id,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "episode_reward": e.episode_reward,
            }
            for e in episodes
        ],
    }
    path = tmp_path / "p4_gate.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_gate_a_passes_only_when_every_episode_record_reproduces(tmp_path: Path) -> None:
    """T8: per-episode equality, not an aggregate -- and a mean-preserving swap must still fail.

    The compensating pair is the point.  A gate that compares cell means passes a run in which
    two draws swapped their values, which is exactly the failure mode a reproduction check is
    supposed to catch: the aggregate is the thing that reproduces trivially.
    """
    reference = [
        _episode_result("madt", 101, 1000, 106.4561500275786),
        _episode_result("madt", 101, 1001, 104.71978021978022),
        _episode_result("madt", 202, 1000, 103.5),
    ]
    path = _reference_file(tmp_path, reference)

    produced = [_episode_result("dt_g5", e.seed, e.draw_id, e.att_horizon) for e in reference]
    result = gate_a_result(produced, reference_path=path)
    assert result.status == "PASS"
    assert result.n_compared == 3 and result.n_mismatched == 0
    assert result.reference_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    tiny = [_episode_result("dt_g5", e.seed, e.draw_id, e.att_horizon) for e in reference]
    tiny[1] = _episode_result("dt_g5", 101, 1001, reference[1].att_horizon + 1e-9)
    failed = gate_a_result(tiny, reference_path=path)
    assert failed.status == "FAIL"
    assert failed.n_mismatched == 1
    assert any("1001" in m for m in failed.mismatches)

    delta = 0.25
    swapped = [_episode_result("dt_g5", e.seed, e.draw_id, e.att_horizon) for e in reference]
    swapped[0] = _episode_result("dt_g5", 101, 1000, reference[0].att_horizon + delta)
    swapped[1] = _episode_result("dt_g5", 101, 1001, reference[1].att_horizon - delta)
    mean_preserving = gate_a_result(swapped, reference_path=path)
    assert mean_preserving.reproduced_mean == pytest.approx(mean_preserving.reference_mean, abs=1e-12)
    assert mean_preserving.status == "FAIL"
    assert mean_preserving.n_mismatched == 2


def test_gate_a_refuses_a_run_that_did_not_cover_the_reference(tmp_path: Path) -> None:
    """A missing cell is not a pass with fewer comparisons."""
    reference = [
        _episode_result("madt", 101, 1000, 106.0),
        _episode_result("madt", 101, 1001, 104.0),
    ]
    path = _reference_file(tmp_path, reference)
    with pytest.raises(ValueError, match="does not cover"):
        gate_a_result([_episode_result("dt_g5", 101, 1000, 106.0)], reference_path=path)


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------


def _point_run(point_key: str = "dt_g5", target: float = NAIVE_TARGET) -> PointRun:
    episodes = tuple(
        _episode_result(point_key, seed, draw, 100.0 + seed / 1000 + draw / 10000)
        for seed in (101, 202)
        for draw in (1000, 1001)
    )
    support = tuple(
        SupportCounts(
            n=360,
            in_support=285,
            below=0,
            above=75,
            rtg_first=target,
            rtg_last=target + 7313.0,
            rtg_min=target,
            rtg_max=target + 7313.0,
        )
        for _ in episodes
    )
    index = tuple((int(e.seed), int(e.draw_id)) for e in episodes)
    return PointRun(
        point_key=point_key,
        target_rtg=target,
        rtg_scale=9991.0,
        episodes=episodes,
        support=support,
        support_index=index,
        canary_rtg_series=(target, target + 20.0, target + 41.0),
        canary_cell=(101, 1000),
    )


def test_a_point_artifact_carries_both_criteria_and_the_constants_it_ran_under() -> None:
    """Both criteria are reported for every point; neither is described as a selector."""
    payload = point_artifact(
        _point_run(),
        rtg_range=(-9991.0, -6.0),
        engine_seed=1000,
        env_settings={"max_steps": 360},
    )
    assert payload["format_version"] == "p4.3-rtg/1.0"
    assert payload["point_key"] == "dt_g5"
    assert payload["target_rtg"] == NAIVE_TARGET
    assert payload["rtg_scale"] == 9991.0
    assert payload["training_rtg_range"] == [-9991.0, -6.0]
    assert payload["cell"]["n_episodes"] == 4
    assert payload["in_support"]["mean_fraction"] == pytest.approx(285 / 360)
    assert payload["in_support"]["mean_above"] == 75.0
    assert payload["in_support"]["mean_below"] == 0.0
    assert payload["in_support"]["selects"] is False
    assert len(payload["episodes"]) == 4
    assert payload["canary"]["cell"] == [101, 1000]
    assert payload["canary"]["rtg_series"][0] == NAIVE_TARGET


def test_the_report_refuses_points_that_disagree_about_the_rtg_scale() -> None:
    """T5 at campaign level: one divisor for the whole task, or the points are not comparable."""
    points = [
        {"point_key": "dt_g5", "target_rtg": NAIVE_TARGET, "rtg_scale": 9991.0},
        {"point_key": "dt_g0", "target_rtg": 0.0, "rtg_scale": 5000.0},
    ]
    with pytest.raises(ValueError, match="disagree about rtg_scale"):
        report_artifact(
            points=points,
            probe={"format_version": "p4.3-rtg/1.0"},
            rtg_range=(-9991.0, -6.0),
            checkpoints={101: "a.pt"},
        )


def test_pearson_r_matches_an_independent_recomputation_and_the_known_extremes() -> None:
    """The correlation behind the withdrawn criterion's result (BRIEF_15 section 14.1).

    Recomputed from the centred cross-product by hand rather than by calling the same helper,
    and pinned at both extremes so a sign error or a dropped centring cannot survive.
    """
    assert pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0, abs=1e-12)
    assert pearson_r([1.0, 2.0, 3.0], [-2.0, -4.0, -6.0]) == pytest.approx(-1.0, abs=1e-12)

    rng = np.random.default_rng(1403)
    xs = list(rng.uniform(0.0, 1.0, size=17))
    ys = list(rng.uniform(100.0, 110.0, size=17))
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    cov = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = math.fsum((x - mean_x) ** 2 for x in xs)
    var_y = math.fsum((y - mean_y) ** 2 for y in ys)
    assert pearson_r(xs, ys) == pytest.approx(cov / math.sqrt(var_x * var_y), abs=1e-12)


def test_pearson_r_refuses_a_constant_series() -> None:
    """A zero-variance series has no correlation; returning 0.0 would read as 'unrelated'."""
    with pytest.raises(ValueError, match="zero variance"):
        pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def _landscape_points(atts: Sequence[float], fractions: Sequence[float]) -> list[dict[str, Any]]:
    """Evaluated grid points whose per-draw ATTs are PAIRED: the same draws under every point.

    Two noise components, and the second one exists because a mutation survived without it.  The
    **shared** draw effect is large and cancels under pairing -- that is what makes the paired CI
    narrower than the marginal one.  The **independent** jitter does not cancel, so the paired
    difference varies across draws and its CI has real width; without it every step's CI would be
    degenerate, every step would "resolve", and a `resolves` verdict computed from the point
    estimate's sign would be indistinguishable from one computed from the CI.
    """
    keys = list(grid_targets())
    rng = np.random.default_rng(1414)
    out: list[dict[str, Any]] = []
    for index, (att, fraction) in enumerate(zip(atts, fractions)):
        key = keys[index]
        out.append(
            {
                "point_key": key,
                "target_rtg": DECLARED_GRID[index],
                "rtg_scale": 9991.0,
                "episodes": [
                    {
                        "arm": key,
                        "seed": seed,
                        "draw_id": draw,
                        "att_horizon": (
                            att
                            + 0.5 * ((draw % 7) - 3)          # shared: cancels under pairing
                            + 0.1 * ((seed % 3) - 1)          # shared
                            + float(rng.normal(0.0, 0.15))    # independent: does not cancel
                        ),
                        "horizon_vehicle_count": 44.0,
                        "episode_reward": -7000.0,
                    }
                    for seed in (101, 202)
                    for draw in range(1000, 1020)
                ],
                "in_support": {
                    "mean_fraction": fraction,
                    "mean_below": 0.0,
                    "mean_above": 360.0 * (1.0 - fraction),
                },
            }
        )
    return out


def test_adjacent_steps_are_compared_PAIRED_and_each_carries_a_resolution_verdict() -> None:
    """BRIEF_15 section 14: adjacent cells share draws and seeds, so their difference is paired.

    Using a marginal CI for a paired quantity is the D1 defect from the P2.6 review.  The fixture
    makes the difference visible: a large per-draw spread is shared by both points, so the
    marginal CI is wide while the paired CI is narrow, and a report that used the marginal one
    would call a resolved step unresolved.
    """
    atts = [104.5564, 104.6121, 104.6928, 104.7700, 104.7633, 104.9558, 104.8640, 104.9000, 105.0]
    fractions = [0.0, 0.1775, 0.3348, 0.4651, 0.5787, 0.8180, 0.95, 1.0, 0.9]
    payload = report_artifact(
        points=_landscape_points(atts, fractions),
        probe={"format_version": "p4.3-rtg/1.0"},
        rtg_range=(-9991.0, -6.0),
        checkpoints={101: "a.pt"},
    )

    adjacent = payload["adjacent_comparisons"]
    assert len(adjacent) == len(DECLARED_GRID) - 1
    first = adjacent["dt_g0_to_dt_g1"]
    assert first["from_target"] == 0.0 and first["to_target"] == -1000.0
    # Paired: the shared per-draw structure cancels, so the difference is the point offset plus
    # only the independent jitter.
    assert first["mean_difference"] == pytest.approx(atts[1] - atts[0], abs=0.1)
    assert first["paired"] is True
    assert first["n_shared_draws"] == 20
    assert "rank_biserial" in first and "ci95_low" in first and "ci95_high" in first

    # The verdict must come from the CI, not from the sign of the point estimate.  The fixture
    # deliberately contains both kinds of step, so a verdict computed from the sign alone would
    # disagree with at least one of them -- which is how this assertion has teeth.
    for name, step in adjacent.items():
        expected = step["ci95_low"] > 0.0 or step["ci95_high"] < 0.0
        assert step["resolves"] == expected, name
        assert step["ci95_half_width"] > 0.0, f"{name}: a degenerate CI cannot test a verdict"
    verdicts = {name: step["resolves"] for name, step in adjacent.items()}
    assert any(verdicts.values()), "the fixture must contain a step that resolves"
    assert not all(verdicts.values()), "the fixture must contain a step that does NOT resolve"
    signs = {name: adjacent[name]["mean_difference"] != 0.0 for name in adjacent}
    assert verdicts != signs, (
        "every step's point estimate is non-zero, so a `resolves` computed from the sign would "
        "agree with the CI everywhere and this test could not tell them apart"
    )

    # The paired CI must be NARROWER than the marginal one the D1 defect would have used: the
    # fixture's per-draw spread is shared by both points, so pairing removes it.  The marginal
    # half-width is computed here from the episodes rather than read from the artifact, so the
    # comparison does not depend on the artifact's own aggregation.
    marginal_half = mean_ci95(
        [e["att_horizon"] for e in payload["points"][0]["episodes"]]
    ).ci95
    assert first["ci95_half_width"] < marginal_half, (
        f"paired {first['ci95_half_width']} is not narrower than marginal {marginal_half}; "
        "the fixture no longer distinguishes the two rulers"
    )


def test_the_report_carries_the_withdrawn_criterions_correlation_with_the_outcome() -> None:
    """BRIEF_15 section 14.1: reported as a RESULT, with the number, over the graded points."""
    atts = [104.5564, 104.6121, 104.6928, 104.7700, 104.7633, 104.9558, 104.8640, 104.9000, 105.0]
    fractions = [0.0, 0.1775, 0.3348, 0.4651, 0.5787, 0.8180, 0.95, 1.0, 0.9]
    payload = report_artifact(
        points=_landscape_points(atts, fractions),
        probe={"format_version": "p4.3-rtg/1.0"},
        rtg_range=(-9991.0, -6.0),
        checkpoints={101: "a.pt"},
    )
    block = payload["in_support_vs_att"]
    assert block["n_points"] == len(DECLARED_GRID)
    assert block["pearson_r"] == pytest.approx(pearson_r(fractions, atts), abs=1e-12)
    assert block["higher_in_support_is_better"] is False
    # The six-point subset the coordinator computed independently must be reported too, so the
    # two instruments can be compared on identical inputs.
    assert block["first_six_points_pearson_r"] == pytest.approx(
        pearson_r(fractions[:6], atts[:6]), abs=1e-12
    )


def test_the_report_labels_its_statistics_exploratory() -> None:
    """PREREGISTRATION.md section 2 files calibrated-vs-naive RTG prompting as exploratory."""
    points = [
        {
            "point_key": key,
            "target_rtg": target,
            "rtg_scale": 9991.0,
            "episodes": [
                {
                    "arm": key,
                    "seed": 101,
                    "draw_id": draw,
                    "att_horizon": 100.0 + draw / 100.0 + DECLARED_GRID.index(target) / 10.0,
                    "horizon_vehicle_count": 7.0,
                    "episode_reward": -1234.0,
                }
                for draw in (1000, 1001, 1002)
            ],
            "in_support": {"mean_fraction": 0.5, "mean_below": 1.0, "mean_above": 2.0},
        }
        for key, target in list(grid_targets().items())[:2]
    ]
    payload = report_artifact(
        points=points,
        probe={"format_version": "p4.3-rtg/1.0"},
        rtg_range=(-9991.0, -6.0),
        checkpoints={101: "a.pt"},
    )
    assert payload["analysis_status"] == "exploratory"
    assert "exploratory" in payload["role"].lower()
    assert payload["selection"]["att_may_select"] is False
    assert payload["selection"]["in_support_may_select"] is False


# ----------------------------------------------------------------------
# DEFERRED 31 and 39
# ----------------------------------------------------------------------


def test_the_effect_size_sidecar_recomputes_p4s_numbers_without_touching_its_artifact() -> None:
    """DEFERRED 31, ruled as a sidecar (BRIEF_15 section 12.3).

    The rank-biserial is checked against ``paired_comparison`` run on the artifact's own episode
    records -- a different route from whatever the sidecar uses internally -- and the committed
    file's digest is recorded and must still match afterwards.
    """
    gate_path = DATA / "p4_gate.json"
    before = hashlib.sha256(gate_path.read_bytes()).hexdigest()

    sidecar = effect_size_sidecar(gate_path)

    assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == before, "the artifact was edited"
    assert sidecar["source"]["path"].endswith("p4_gate.json")
    assert sidecar["source"]["sha256"] == before

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    episodes = [EpisodeResult(**e) for e in gate["episodes"]]
    thresholds = json.loads((DATA / "p4_heldout_thresholds.json").read_text(encoding="utf-8"))
    episodes.extend(EpisodeResult(**e) for e in thresholds["episodes"])
    by_arm: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        by_arm.setdefault(episode.arm, []).append(episode)

    assert set(sidecar["comparisons"]) == set(gate["wilcoxon"])
    for name, block in sidecar["comparisons"].items():
        right = name.split("_vs_")[1]
        expected = paired_comparison(by_arm["madt"], by_arm[right])
        assert block["rank_biserial"] == expected.rank_biserial
        assert block["mean_difference"] == expected.mean_difference
        assert block["ci95_low"] == expected.ci95_low
        assert block["ci95_high"] == expected.ci95_high
        assert block["n_shared_draws"] == expected.n_shared_draws
        assert block["p_value"] == gate["wilcoxon"][name]["p_value"]


def test_runtime_provenance_gains_the_split_without_changing_an_existing_field() -> None:
    """DEFERRED 39: additively only -- every existing key keeps its name and its meaning."""
    plain = runtime_provenance()
    for key in (
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "cuda_device_name",
        "torch_num_threads",
        "numpy_version",
        "python_version",
        "git_commit",
    ):
        assert key in plain, f"{key} disappeared; the change must be additive"

    assert plain["written_at_git_commit"] == plain["git_commit"] or plain["git_commit"] == ""
    assert plain["measurement_git_commits"] == []

    carried = runtime_provenance(measurement_git_commits=["bbb", "aaa", "bbb"])
    assert carried["measurement_git_commits"] == ["aaa", "bbb"]
    assert carried["git_commit"] == plain["git_commit"]
    assert carried["written_at_git_commit"] == plain["written_at_git_commit"]


def test_leaf_diff_names_every_leaf_that_moved() -> None:
    """The instrument DEFERRED 39's regeneration proof rests on, tested on a known difference."""
    left = {"a": 1, "b": {"c": [1, 2, {"d": "x"}], "e": None}, "f": 3.5}
    right = {"a": 1, "b": {"c": [1, 2, {"d": "y"}], "g": 7}, "f": 3.5}

    diff = leaf_diff(left, right)
    assert diff.changed == ("b.c[2].d",)
    assert diff.removed == ("b.e",)
    assert diff.added == ("b.g",)


def test_leaf_diff_reports_no_difference_for_an_identical_structure() -> None:
    left = {"x": [1, {"y": 2}], "z": "s"}
    diff = leaf_diff(left, json.loads(json.dumps(left)))
    assert (diff.removed, diff.added, diff.changed) == ((), (), ())
