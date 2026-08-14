"""Tests for ``offline.offline_baselines`` -- the %BC filter, IQL's transitions, and the verdict.

Three layers, and the Return Packet says which of them ran:

* pure arithmetic (effect sizes, the A6 verdict, the recovered fraction, campaign integrity)
  and the committed-artifact checks -- always run;
* the data path (stream returns, the top-10 % filter, the transition table, one training step
  of each method) over a synthetic corpus -- always runs;
* the real-corpus path -- **opt-in**, skipping with a reason that names the variable it needs.

The load-bearing test here is
``test_the_last_transitions_next_state_is_observation_row_T_read_by_a_different_path``.  The
window loader structurally never yields observation row ``T`` -- which is the entire reason the
transition table exists -- so a cross-check against the loader has near-zero power at exactly
the row at risk.  That test reads the row through plain ``np.load`` instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from agent.OfflineBaselines import canonical_state_dict_digest
import offline.offline_baselines as baselines_module
from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import (
    TRAINING_SEEDS,
    EpisodeResult,
    mean_ci95,
    stack_dataset,
    wilcoxon_signed_rank,
)
from offline.offline_baselines import (
    ARTIFACT_FORMAT_VERSION,
    DECLARED_GRADIENT_STEPS,
    DELTA_ATT,
    DELTA_ATT_DERIVATION,
    IQL_GAMMA,
    METHODS,
    assert_campaign_complete,
    baselines_artifact,
    build_transitions,
    equivalence_verdict,
    filter_stacked_to_streams,
    iql_reward_scale,
    iql_targets,
    load_baseline_checkpoint,
    merge_training_runs,
    paired_comparison,
    StreamReturn,
    rank_biserial,
    recovered_fraction,
    stream_returns,
    top_return_streams,
    train_bc,
    train_iql,
)

from tests.test_offline_dataset import write_dataset_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_SCENARIO = "fixture_2ix"
ALPHA_GROUP = (4, 3)                 # ix_alpha: state_dim 4, n_actions 3
ZULU_GROUP = (3, 2)                  # ix_zulu:  state_dim 3, n_actions 2
CONTEXT = 4
FIXTURE_T = 8


@pytest.fixture()
def fixture_dataset(tmp_path: Path) -> TrajectoryWindowDataset:
    """A real two-group dataset over two episodes of the shared fixture."""
    dataset_dir = write_dataset_dir(tmp_path, "fixture__policy")
    return TrajectoryWindowDataset([dataset_dir], context_length=CONTEXT, split="train")


# ----------------------------------------------------------------------
# Stream returns and the %BC filter (brief section 4, item 3)
# ----------------------------------------------------------------------


def test_stream_returns_equal_an_independent_sum_of_the_raw_reward_arrays(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """Second route: ``math.fsum`` over the ``.npz`` reward array, read by ``np.load``.

    The loader derives a stream's return from its returns-to-go; this recomputes it from the
    rewards on disk, so the two agree only if the RTG convention is the one the docstring
    claims.  The fixture's rewards are integral, so the comparison is exact.
    """
    returns = stream_returns(fixture_dataset)
    assert len(returns) == 4, "two episodes x two intersections"

    for entry in returns:
        index = {"ix_zulu": 0, "ix_alpha": 1}[entry.ix_id]
        with np.load(Path(entry.dataset_dir) / entry.episode_file) as raw:
            rewards = np.asarray(raw[f"ix{index}_local_reward"], dtype=np.float32)
        assert entry.total_return == math.fsum(float(r) for r in rewards), entry


def test_the_top_fraction_selects_exactly_the_highest_returning_streams(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """The fixture's ordering is known by construction, so the answer is not derived here.

    Rewards scale by episode index (episode 1 doubles episode 0), so the four stream returns
    are -18, -36, -37, -74 and their order is unambiguous.  At a fraction of 0.10 over 4
    streams the filter keeps ``ceil(0.4) = 1``: the single best stream, ix_zulu of episode 0.
    """
    everything = {(e.ix_id, e.episode_index): e.total_return for e in stream_returns(fixture_dataset)}
    assert everything == {
        ("ix_zulu", 0): -18.0,
        ("ix_alpha", 0): -37.0,
        ("ix_zulu", 1): -36.0,
        ("ix_alpha", 1): -74.0,
    }

    kept = top_return_streams(fixture_dataset, 0.10)
    assert len(kept) == 1
    assert (kept[0].ix_id, kept[0].episode_index) == ("ix_zulu", 0)

    half = top_return_streams(fixture_dataset, 0.50)
    assert [(k.ix_id, k.episode_index) for k in half] == [("ix_zulu", 0), ("ix_zulu", 1)]


def test_the_filter_rounds_up_and_never_returns_an_empty_set(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """``ceil`` of the fraction, and at least one stream however small the fraction is."""
    assert len(top_return_streams(fixture_dataset, 0.0001)) == 1
    assert len(top_return_streams(fixture_dataset, 0.26)) == 2      # ceil(1.04)
    assert len(top_return_streams(fixture_dataset, 1.0)) == 4


def test_filtering_stacked_windows_keeps_exactly_the_kept_streams_rows(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """Every surviving row must trace back through ``item_index`` to a kept stream."""
    stacked = stack_dataset(fixture_dataset, group=ALPHA_GROUP)
    kept = [s for s in stream_returns(fixture_dataset) if s.ix_id == "ix_alpha" and s.episode_index == 0]
    filtered = filter_stacked_to_streams(fixture_dataset, stacked, kept)

    assert int(filtered["state"].shape[0]) == FIXTURE_T
    assert int(stacked["state"].shape[0]) == 2 * FIXTURE_T, "the fixture must have rows to drop"
    for row in range(int(filtered["state"].shape[0])):
        meta = fixture_dataset.item_meta(int(filtered["item_index"][row]))
        assert (meta.ix_id, meta.episode_index) == ("ix_alpha", 0)
    # and the surviving rows are the same tensors the unfiltered stack carried
    for row in range(int(filtered["state"].shape[0])):
        source = int((stacked["item_index"] == filtered["item_index"][row]).nonzero()[0, 0])
        assert torch.equal(filtered["state"][row], stacked["state"][source])


# ----------------------------------------------------------------------
# IQL's transitions, and the boundary the registration is about
# ----------------------------------------------------------------------


def test_transitions_reproduce_the_loader_state_and_action_for_every_step(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """The cross-check that ties the second reader to the first, for ``t`` in ``[0, T-1]``."""
    table = build_transitions(fixture_dataset, group=ALPHA_GROUP)
    stacked = stack_dataset(fixture_dataset, group=ALPHA_GROUP)
    assert int(table.state.shape[0]) == int(stacked["state"].shape[0]) == 2 * FIXTURE_T

    for row in range(int(table.state.shape[0])):
        window = fixture_dataset[int(stacked["item_index"][row])]
        assert torch.equal(table.state[row], window["state"][-1]), f"state row {row}"
        assert int(table.action[row]) == int(window["action"][-1]), f"action row {row}"
        assert int(table.t[row]) == fixture_dataset.item_meta(
            int(stacked["item_index"][row])
        ).t


def test_transition_rewards_equal_the_returns_to_go_differences_exactly(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """``r_t == rtg[t] - rtg[t+1]``, and ``r_{T-1} == rtg[T-1]``.

    Exact rather than approximate: the corpus's rewards are integral, so both routes are
    exactly representable in float32 and a tolerance would hide a real disagreement.
    """
    table = build_transitions(fixture_dataset, group=ALPHA_GROUP)
    stacked = stack_dataset(fixture_dataset, group=ALPHA_GROUP)
    rtg = {}
    for row in range(int(stacked["item_index"].shape[0])):
        meta = fixture_dataset.item_meta(int(stacked["item_index"][row]))
        rtg[(meta.episode_file, meta.t)] = float(
            fixture_dataset[int(stacked["item_index"][row])]["rtg"][-1, 0]
        )

    for row in range(int(table.state.shape[0])):
        meta = fixture_dataset.item_meta(int(stacked["item_index"][row]))
        here = rtg[(meta.episode_file, meta.t)]
        later = rtg.get((meta.episode_file, meta.t + 1), 0.0)
        assert float(table.reward[row]) == here - later, f"row {row} at t={meta.t}"


def test_every_decision_keeps_its_transition_including_the_last(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """Dropping the final transition is the quiet way to avoid bootstrapping at the boundary."""
    table = build_transitions(fixture_dataset, group=ALPHA_GROUP)
    per_stream: dict[int, list[int]] = {}
    for row in range(int(table.t.shape[0])):
        per_stream.setdefault(int(table.stream_index[row]), []).append(int(table.t[row]))
    assert len(per_stream) == 2
    for stream, steps in per_stream.items():
        assert sorted(steps) == list(range(FIXTURE_T)), f"stream {stream}"


def test_the_last_transitions_next_state_is_observation_row_T_read_by_a_different_path(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """The row the loader can never show, read straight from the ``.npz`` with ``np.load``.

    Required by the brief's section 8.4 correction: the loader-based cross-check above has
    near-zero discriminating power at exactly the row the second reader exists for, because
    ``TrajectoryWindowDataset`` never yields observation row ``T``.  So this test reads row
    ``T`` by a different path and normalises it by hand from the fitted statistics, and it
    additionally asserts that row ``T-1`` gives a DIFFERENT answer -- otherwise the mutation
    ``row T -> row T-1`` would leave it green.
    """
    table = build_transitions(fixture_dataset, group=ALPHA_GROUP)
    stats = fixture_dataset.stats
    mean = np.asarray(stats.state_mean[FIXTURE_SCENARIO]["ix_alpha"], dtype=np.float32)
    std = np.asarray(stats.state_std[FIXTURE_SCENARIO]["ix_alpha"], dtype=np.float32)
    safe = np.where(std > 0, std, np.float32(1.0)).astype(np.float32)

    records = [r for r in fixture_dataset.episode_records]
    assert records, "the fixture must carry episode records"
    checked = 0
    for stream in range(int(table.stream_index.max()) + 1):
        rows = (table.stream_index == stream).nonzero().reshape(-1)
        last = int(rows[int(table.t[rows].argmax())])
        assert int(table.t[last]) == FIXTURE_T - 1

        record = records[stream]
        with np.load(Path(record.dataset_dir) / record.episode_file) as raw:
            raw_states = np.asarray(raw["ix1_state"], dtype=np.float32)
        assert raw_states.shape[0] == FIXTURE_T + 1, "observations carry T+1 rows"

        expected_T = (raw_states[FIXTURE_T] - mean) / safe
        expected_T_minus_1 = (raw_states[FIXTURE_T - 1] - mean) / safe
        got = table.next_state[last].numpy()
        assert np.array_equal(got, expected_T), f"stream {stream}: {got} vs {expected_T}"
        assert not np.array_equal(expected_T, expected_T_minus_1), (
            "the fixture's last two observation rows must differ, or this test cannot fail "
            "under the row T -> row T-1 mutation"
        )
        checked += 1
    assert checked == 2


def test_targets_bootstrap_through_the_horizon_with_no_terminal_term(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """``terminated`` is hardcoded ``False``: every episode ends by TIME-LIMIT truncation.

    ``PREREGISTRATION.md`` section 7 and the Decisions Log of 2026-07-26 make this a registered
    fairness constraint: treating the horizon as absorbing causes systematic value
    underestimation near episode end and would hand the DT an unearned win over its own
    baselines.  So the target is ``r + gamma * V(s')`` at EVERY transition, the last one
    included, and this test asserts the last rows specifically -- where a ``done`` mask would
    leave ``r`` alone.
    """
    table = build_transitions(fixture_dataset, group=ALPHA_GROUP)
    count = int(table.state.shape[0])
    next_values = torch.arange(1.0, count + 1.0, dtype=torch.float32)

    targets = iql_targets(table, next_values, gamma=IQL_GAMMA)
    expected = table.reward + IQL_GAMMA * next_values
    assert torch.equal(targets, expected)

    for stream in range(int(table.stream_index.max()) + 1):
        rows = (table.stream_index == stream).nonzero().reshape(-1)
        last = int(rows[int(table.t[rows].argmax())])
        bootstrap = IQL_GAMMA * float(next_values[last])
        assert abs(bootstrap) > 1e-6, "the bootstrap term must be non-zero at the last row"
        assert float(targets[last]) != float(table.reward[last]), (
            "the last transition must bootstrap through the truncation boundary"
        )


def test_a_multi_group_dataset_refuses_to_build_transitions_without_a_group(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    """C6 forbids padding across intersections, so the group must be explicit when ambiguous."""
    with pytest.raises(ValueError, match="pass group= to say which one to use"):
        build_transitions(fixture_dataset)
    assert build_transitions(fixture_dataset, group=ZULU_GROUP).state.shape[1] == 3


def test_the_reward_scale_is_iqls_published_formula_over_the_training_returns() -> None:
    """``1000 / (max - min)``, and the corpus's own measured returns give 1000/4229."""
    assert iql_reward_scale([-5762.0, -9991.0, -7000.0]) == 1000.0 / 4229.0
    with pytest.raises(ValueError, match="constant"):
        iql_reward_scale([-7.0, -7.0])


def test_the_reward_scale_is_applied_to_the_transition_rewards(
    fixture_dataset: TrajectoryWindowDataset,
) -> None:
    plain = build_transitions(fixture_dataset, group=ALPHA_GROUP, reward_scale=1.0)
    scaled = build_transitions(fixture_dataset, group=ALPHA_GROUP, reward_scale=0.25)
    assert torch.equal(scaled.reward, plain.reward * 0.25)
    assert scaled.reward_scale == 0.25


# ----------------------------------------------------------------------
# Effect sizes, the A6 verdict and the recovered fraction
# ----------------------------------------------------------------------


def _episodes(arm: str, values: dict[int, float], seed: int | None = 1) -> list[EpisodeResult]:
    return [
        EpisodeResult(
            arm=arm,
            seed=seed,
            draw_id=draw,
            att_horizon=value,
            horizon_vehicle_count=10.0 + draw,
            episode_reward=-100.0,
        )
        for draw, value in sorted(values.items())
    ]


def test_rank_biserial_matches_an_independent_derivation() -> None:
    """``(W+ - W-) / (W+ + W-)``, recomputed here from signed ranks counted by hand.

    Differences ``[+3, -1, +2, +4]`` give ``|d|`` ranks ``[3, 1, 2, 4]``: ``W+ = 3+2+4 = 9``,
    ``W- = 1``, so ``r = 8/10 = 0.8``.  The sign convention is pinned: a positive value means
    the FIRST argument was larger more often and by bigger ranks.
    """
    left = [10.0, 10.0, 10.0, 10.0]
    right = [7.0, 11.0, 8.0, 6.0]
    result = wilcoxon_signed_rank(left, right)
    assert (result.w_plus, result.w_minus) == (9.0, 1.0)
    assert rank_biserial(result) == 0.8
    assert rank_biserial(wilcoxon_signed_rank(right, left)) == -0.8


def test_a_paired_comparison_reports_the_ci_width_and_the_win_counts() -> None:
    """"Contains 0" is a failure to reject; the width says how large a difference is excluded."""
    dt = _episodes("madt", {1000: 100.0, 1001: 101.0, 1002: 102.0, 1003: 103.0})
    bc = _episodes("bc", {1000: 100.5, 1001: 101.5, 1002: 101.0, 1003: 103.5})
    comparison = paired_comparison(dt, bc)

    differences = [-0.5, -0.5, 1.0, -0.5]
    cell = mean_ci95(differences)
    assert comparison.n_shared_draws == 4
    assert comparison.mean_difference == cell.mean
    assert comparison.ci95_half_width == cell.ci95
    assert comparison.ci95_width == 2.0 * cell.ci95
    assert comparison.ci95_low == cell.mean - cell.ci95
    assert comparison.ci95_high == cell.mean + cell.ci95
    assert (comparison.wins, comparison.losses, comparison.ties) == (3, 1, 0)
    assert comparison.median_difference == float(np.median(differences))


def test_a_paired_comparison_without_shared_draws_is_void() -> None:
    """Amendment A5 point 3: a comparison without shared draws is void, not approximate."""
    with pytest.raises(ValueError, match="shared draws"):
        paired_comparison(_episodes("madt", {1000: 100.0}), _episodes("bc", {1001: 100.0}))


def test_the_verdict_covers_all_four_branches_including_the_one_A6_does_not_name() -> None:
    """A6 fixes three branches; the fourth is arithmetically possible and is not silently
    folded into "inconclusive", because a baseline beating the DT by more than delta is a
    decisive result in the other direction rather than an absence of one."""
    delta = 0.5
    assert equivalence_verdict(0.0, 0.2, delta) == "matches"
    assert equivalence_verdict(-1.0, 0.2, delta) == "dt_genuinely_better"
    assert equivalence_verdict(1.0, 0.2, delta) == "baseline_genuinely_better"
    assert equivalence_verdict(0.0, 2.0, delta) == "inconclusive_at_this_power"
    assert equivalence_verdict(-0.6, 0.3, delta) == "inconclusive_at_this_power"


def test_the_verdict_boundaries_are_closed_at_exactly_plus_or_minus_delta() -> None:
    """``lies entirely within [-delta, +delta]`` is inclusive; ``entirely below`` is not."""
    delta = 0.5
    # CI exactly [-0.5, +0.5] -> inside the closed interval -> matches.
    assert equivalence_verdict(0.0, 0.5, delta) == "matches"
    # CI exactly [-1.5, -0.5] -> its upper end is not BELOW -delta, so not "genuinely better".
    assert equivalence_verdict(-1.0, 0.5, delta) == "inconclusive_at_this_power"
    # One ulp lower and it is.
    assert equivalence_verdict(-1.0, 0.5 - 1e-12, delta) == "dt_genuinely_better"


def test_the_recovered_fraction_matches_the_paired_identity() -> None:
    """Two routes to one number, and the worked case A6's clarification names.

    A6's clarification states that a baseline landing 0.5 ATT worse than the DT recovers only
    20.2 % of the DT's margin.  With the committed cells (MAPPO@1000 105.5820, DT 104.9558)
    that is ``(105.5820 - 105.4558)/0.6263``; the identity ``1 - 0.5/delta`` gives the same
    number by different arithmetic.
    """
    mappo, dt = 105.58203462874322, 104.95575898180847
    margin = mappo - dt
    assert recovered_fraction(mappo, dt, dt) == 1.0
    assert recovered_fraction(mappo, mappo, dt) == 0.0

    worse_by_half = recovered_fraction(mappo, dt + 0.5, dt)
    assert worse_by_half == pytest.approx(1.0 - 0.5 / margin, abs=1e-12)
    assert round(100.0 * worse_by_half, 1) == 20.2


def test_the_recovered_fraction_refuses_a_degenerate_reference() -> None:
    with pytest.raises(ValueError, match="zero"):
        recovered_fraction(105.0, 104.0, 105.0)


def test_the_declared_delta_matches_the_committed_p4_artifacts() -> None:
    """delta is A6's declared literal AND a quantity recomputed from P4's own records.

    The constant in the source could drift from the artifact it was derived from without any
    test noticing, and it decides every verdict in this task.  Recomputed here from the 500
    MADT and 500 MAPPO@1000 per-episode records, paired by draw, then rounded as A6 states it.
    """
    gate = json.loads((REPO_ROOT / "docs/data/p4_gate.json").read_text(encoding="utf-8"))
    thresholds = json.loads(
        (REPO_ROOT / "docs/data/p4_heldout_thresholds.json").read_text(encoding="utf-8")
    )
    dt = [EpisodeResult(**e) for e in gate["episodes"] if e["arm"] == "madt"]
    mappo = [EpisodeResult(**e) for e in thresholds["episodes"] if e["arm"] == "mappo1000"]
    assert len(dt) == len(mappo) == 500

    comparison = paired_comparison(mappo, dt)
    assert comparison.n_shared_draws == 100
    assert round(comparison.mean_difference, 4) == DELTA_ATT


# ----------------------------------------------------------------------
# Campaign integrity and the artifact
# ----------------------------------------------------------------------


def test_a_partial_campaign_cannot_be_reported() -> None:
    """The condition attached to running the campaign in-session: completed == requested."""
    requested = [("bc", 101, 1000), ("bc", 101, 1001)]
    produced = _episodes("bc", {1000: 100.0}, seed=101)
    with pytest.raises(ValueError, match="incomplete"):
        assert_campaign_complete(requested, produced)

    complete = _episodes("bc", {1000: 100.0, 1001: 101.0}, seed=101)
    assert_campaign_complete(requested, complete)


def test_a_campaign_that_produced_an_unrequested_run_is_refused() -> None:
    requested = [("bc", 101, 1000)]
    produced = _episodes("bc", {1000: 100.0, 1099: 101.0}, seed=101)
    with pytest.raises(ValueError, match="not requested"):
        assert_campaign_complete(requested, produced)


def _artifact_episodes() -> list[EpisodeResult]:
    draws = {1000: 100.0, 1001: 101.0, 1002: 102.0}
    episodes: list[EpisodeResult] = []
    episodes += _episodes("madt", draws, seed=101)
    episodes += _episodes("mappo1000", {d: v + 0.6 for d, v in draws.items()}, seed=101)
    episodes += _episodes("maxpressure", {d: v + 70.0 for d, v in draws.items()}, seed=None)
    for method in METHODS:
        episodes += _episodes(method, {d: v + 0.1 for d, v in draws.items()}, seed=101)
    return episodes


def test_the_artifact_carries_every_unconditional_co_report_for_every_arm() -> None:
    """A5 makes ``vehicle_count`` and the draw ids unconditional; A6 adds the recovered
    fraction in every branch.  The artifact-writing path had zero tests in P2.6 and that is
    the defect this test exists to avoid repeating."""
    payload = baselines_artifact(
        episodes=_artifact_episodes(),
        training={"declared_gradient_steps": DECLARED_GRADIENT_STEPS, "seeds": []},
        gate_a={"status": "PASS", "compared": 0, "mismatches": 0},
        env_settings={"max_steps": 360},
        engine_seed=1000,
    )
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION
    assert payload["equivalence_margin_delta"] == DELTA_ATT

    for arm in ("madt", "mappo1000", "maxpressure", *METHODS):
        cell = payload["cells"][arm]
        assert cell["draw_ids"] == [1000, 1001, 1002]
        assert "horizon_vehicle_count_mean" in cell
        assert "att_horizon_mean" in cell
    assert payload["cells"]["maxpressure"]["policy_source"] == "deterministic_heuristic"
    assert payload["cells"]["bc"]["policy_source"] == "checkpoint"

    for method in METHODS:
        entry = payload["comparisons"][f"madt_vs_{method}"]
        assert entry["verdict"] in {
            "matches",
            "dt_genuinely_better",
            "baseline_genuinely_better",
            "inconclusive_at_this_power",
        }
        assert "recovered_fraction" in entry
        assert "ci95_width" in entry
        assert "rank_biserial" in entry
        assert entry["wilcoxon"]["p_value"] <= 1.0


def _training_payload(methods: Sequence[str], **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "training_draw_ids": [1, 2, 3],
        "scenario_id": "cityflow1x1",
        "group": [25, 8],
        "runs": [
            {"method": method, "seed": seed, "canonical_digest": f"{method}{seed}"}
            for method in methods
            for seed in (101, 202)
        ],
    }
    payload.update(overrides)
    return payload


def test_a_second_training_run_keeps_the_first_ones_methods() -> None:
    """Chunked training must not silently drop what an earlier chunk trained.

    The 40,000-step budget over three methods does not fit one job under the campaign's
    30-minute condition, so training runs in chunks -- and a chunk that overwrote the artifact
    would leave a run set that looks complete and is not.  Runs for a method the new chunk
    trained replace the old ones; every other method survives.
    """
    merged = merge_training_runs(
        _training_payload(["bc", "bc_top10"]), _training_payload(["iql"])
    )
    assert sorted({run["method"] for run in merged}) == ["bc", "bc_top10", "iql"]
    assert len(merged) == 6

    retrained = merge_training_runs(
        _training_payload(["bc", "iql"]),
        _training_payload(["bc"], runs=[{"method": "bc", "seed": 101, "canonical_digest": "new"}]),
    )
    assert [r["canonical_digest"] for r in retrained if r["method"] == "bc"] == ["new"]
    assert len([r for r in retrained if r["method"] == "iql"]) == 2


def test_merging_two_training_runs_from_different_designs_is_refused() -> None:
    """A merge across two declarations would mix two designs into one artifact."""
    for field_name, value in (
        ("declared_gradient_steps", 20_000),
        ("training_draw_ids", [1, 2]),
        ("group", [16, 4]),
    ):
        with pytest.raises(ValueError, match=field_name):
            merge_training_runs(
                _training_payload(["bc"]),
                _training_payload(["iql"], **{field_name: value}),
            )


def test_the_artifact_refuses_to_be_built_without_the_reference_arms() -> None:
    """No MADT cell means no comparison; no MAPPO@1000 cell means no recovered fraction."""
    without_dt = [e for e in _artifact_episodes() if e.arm != "madt"]
    with pytest.raises(ValueError, match="madt"):
        baselines_artifact(
            episodes=without_dt,
            training={},
            gate_a={},
            env_settings={},
            engine_seed=1000,
        )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def _train_one_bc(
    dataset: TrajectoryWindowDataset, path: Path, seed: int = 101, steps: int = 6
) -> Any:
    stacked = stack_dataset(dataset, group=ALPHA_GROUP)
    return train_bc(
        stacked,
        state_dim=ALPHA_GROUP[0],
        n_actions=ALPHA_GROUP[1],
        seed=seed,
        method="bc",
        declared_gradient_steps=steps,
        batch_size=4,
        device=torch.device("cpu"),
        checkpoint_path=path,
        stats=dataset.stats,
        scenario_id=FIXTURE_SCENARIO,
        provenance={"tier": "fixture"},
    )


def test_training_bc_writes_the_declared_checkpoint_with_its_canonical_digest(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    record = _train_one_bc(fixture_dataset, tmp_path / "bc_seed101.pt")
    assert record.gradient_steps == 6
    assert record.declared_gradient_steps == 6
    assert len(record.losses) == 6
    assert all(math.isfinite(loss) for loss in record.losses)

    payload = torch.load(tmp_path / "bc_seed101.pt", map_location="cpu", weights_only=False)
    assert payload["provenance"]["gradient_steps"] == 6
    assert record.canonical_digest == canonical_state_dict_digest(payload["model"])


def test_training_bc_is_reproducible_by_canonical_digest_and_seed_dependent(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """Determinism is claimed on the WEIGHTS, never on the file bytes (``DEFERRED`` 29)."""
    first = _train_one_bc(fixture_dataset, tmp_path / "a.pt", seed=101)
    again = _train_one_bc(fixture_dataset, tmp_path / "b.pt", seed=101)
    other = _train_one_bc(fixture_dataset, tmp_path / "c.pt", seed=202)
    assert first.canonical_digest == again.canonical_digest
    assert first.canonical_digest != other.canonical_digest
    assert first.file_sha256 != again.file_sha256, (
        "two names, two files: the file hash cannot carry a determinism claim"
    )


def test_training_refuses_a_checkpoint_directory_that_does_not_exist(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """Filesystem-mutation barrier: a refused run creates no directory and writes nothing."""
    missing = tmp_path / "absent" / "bc.pt"
    with pytest.raises(FileNotFoundError, match="checkpoint directory does not exist"):
        _train_one_bc(fixture_dataset, missing)
    assert not missing.parent.exists()


def test_the_checkpoint_directory_is_checked_before_the_first_gradient_step(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WRITTEN AFTER THE IMPLEMENTATION, and it exists because the test above is weaker than
    it looks: ``torch.save`` creates no directory either, so a run that trained first and only
    then refused would satisfy it.  The barrier is about ORDER, so this pins the order -- the
    loss function is replaced by a sentinel, and reaching it at all is the failure.
    """

    def sentinel(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("training started before the destination was validated")

    monkeypatch.setattr(baselines_module, "action_loss", sentinel)
    with pytest.raises(FileNotFoundError, match="checkpoint directory does not exist"):
        _train_one_bc(fixture_dataset, tmp_path / "absent" / "bc.pt", steps=50)


def test_bc_training_applies_the_availability_mask_the_way_the_dt_does(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """WRITTEN AFTER THE IMPLEMENTATION, to kill a mutant that survived: dropping the mask from
    BC's training loss left all 30 tests green.

    On the real corpus that mutant is *equivalent* -- every stored mask is all-True (P3 review,
    32000/32000 streams), so masking cannot change a gradient there and no reported number
    moves.  The fixture's masks do bind, so the intended behaviour is testable somewhere, and
    what is pinned is that BC's loss treats availability exactly as the DT's does rather than
    diverging silently for a scenario where it would matter.
    """
    stacked = stack_dataset(fixture_dataset, group=ALPHA_GROUP)
    assert not bool(stacked["avail_mask"].all()), "the fixture's masks must bind"

    masked = _train_one_bc(fixture_dataset, tmp_path / "masked.pt", steps=4)
    unmasked_stack = dict(stacked)
    unmasked_stack["avail_mask"] = torch.ones_like(stacked["avail_mask"])
    unmasked = train_bc(
        unmasked_stack,
        state_dim=ALPHA_GROUP[0],
        n_actions=ALPHA_GROUP[1],
        seed=101,
        method="bc",
        declared_gradient_steps=4,
        batch_size=4,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "unmasked.pt",
        stats=fixture_dataset.stats,
        scenario_id=FIXTURE_SCENARIO,
        provenance={},
    )
    assert masked.canonical_digest != unmasked.canonical_digest


def test_a_checkpoint_saved_at_another_step_count_cannot_be_evaluated(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """WRITTEN AFTER THE IMPLEMENTATION.  "No online model selection" made mechanical.

    ``PREREGISTRATION.md`` section 6.1 forbids reporting a checkpoint chosen by anything other
    than the declared budget, so a checkpoint recorded at a different step count is refused by
    the only path that loads one -- an earlier checkpoint that happened to score better cannot
    be evaluated at all.
    """
    path = tmp_path / "bc_seed101.pt"
    _train_one_bc(fixture_dataset, path, steps=6)
    assert load_baseline_checkpoint(None, path, 6)["provenance"]["gradient_steps"] == 6
    with pytest.raises(ValueError, match="forbids reporting a checkpoint chosen"):
        load_baseline_checkpoint(None, path, 7)


def test_training_iql_records_the_awr_weight_diagnostic(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path
) -> None:
    """No sweep ships, so the degeneracy diagnostic is the evidence about the transplant.

    ``docs/plans/p4.4.md`` section 3.5 declines the authorised tau/beta sweep because its
    selection criterion cannot rank either parameter; what ships instead is this diagnostic,
    computed on training data only and therefore selecting nothing.
    """
    table = build_transitions(fixture_dataset, group=ALPHA_GROUP, reward_scale=0.25)
    record = train_iql(
        table,
        state_dim=ALPHA_GROUP[0],
        n_actions=ALPHA_GROUP[1],
        seed=101,
        declared_gradient_steps=6,
        batch_size=4,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "iql_seed101.pt",
        stats=fixture_dataset.stats,
        scenario_id=FIXTURE_SCENARIO,
        provenance={"tier": "fixture"},
    )
    assert record.gradient_steps == 6
    assert all(math.isfinite(loss) for loss in record.losses)

    diagnostics = record.diagnostics
    for key in (
        "awr_weight_mean",
        "awr_weight_clipped_fraction",
        "awr_weight_effective_sample_size",
        "behaviour_agreement",
        "tau",
        "beta",
        "gamma",
        "reward_scale",
    ):
        assert key in diagnostics, key
    assert 0.0 <= diagnostics["awr_weight_clipped_fraction"] <= 1.0
    assert 0.0 <= diagnostics["behaviour_agreement"] <= 1.0
    assert diagnostics["reward_scale"] == 0.25


def test_the_held_out_pool_cannot_enter_a_training_dataset(tmp_path: Path) -> None:
    """The leakage guard stays in one place -- the loader -- and it raises rather than skips."""
    held_out = write_dataset_dir(tmp_path, "fixture__heldout", draws=(1000, 1001))
    with pytest.raises(ValueError, match="must never enter a training corpus"):
        TrajectoryWindowDataset([held_out], context_length=CONTEXT, split="train")


# ----------------------------------------------------------------------
# The real corpus -- opt-in, and it names the variable it needs
# ----------------------------------------------------------------------


def _corpus_dirs() -> list[Path]:
    root = os.environ.get("RLTRAFFIC_CORPUS_V11")
    if not root:
        pytest.skip(
            "RLTRAFFIC_CORPUS_V11 is not set: point it at the collected v1.1 corpus to run "
            "the real-tier path of P4.4"
        )
    dirs = sorted(Path(root).glob("cf_hz1x1__mappo1000__seed*"))
    if not dirs:
        pytest.skip(
            f"no cf_hz1x1__mappo1000__seed* directories under {root}: RLTRAFFIC_CORPUS_V11 "
            "must point at the v1.1 corpus root"
        )
    return dirs


def test_the_real_tier_filters_and_builds_transitions_consistently() -> None:
    """One pass over the tier P4.4 actually trains on, asserting the numbers this task rests on."""
    dirs = _corpus_dirs()
    dataset = TrajectoryWindowDataset(dirs, context_length=20, split="train")
    returns = stream_returns(dataset)
    assert len(returns) == 200
    assert max(r.total_return for r in returns) == -5762.0
    assert min(r.total_return for r in returns) == -9991.0

    kept = top_return_streams(dataset, 0.10)
    assert len(kept) == 20
    assert min(k.total_return for k in kept) == -6419.0

    assert iql_reward_scale([r.total_return for r in returns]) == 1000.0 / 4229.0

    table = build_transitions(dataset, reward_scale=1.0)
    assert int(table.state.shape[0]) == 72_000
    assert int(table.t.max()) == 359


def test_the_thread_pin_is_applied_by_the_cli_and_refuses_a_nonsense_count() -> None:
    """WRITTEN AFTER THE IMPLEMENTATION, and after a real hang rather than a hypothesis.

    A Gate A run under torch's default 16 threads deadlocked after roughly 600 rollouts, every
    OS thread parked in ``futex_do_wait`` with the process consuming no CPU.  ``main`` therefore
    pins the process to one thread -- the liveness fix ``experiments/runner.py`` applies and
    that ``offline/dt_gate.py`` deliberately never inherits.  The pin belongs to the entry
    point: importing this module must not change a caller's thread count, so it is asserted
    here that the default is 1 at the CLI and that the helper restores what it replaced.
    """
    before = torch.get_num_threads()
    try:
        assert baselines_module.pin_torch_threads(1) == before
        assert torch.get_num_threads() == 1
        assert baselines_module.pin_torch_threads(before) == 1
    finally:
        torch.set_num_threads(before)
    assert torch.get_num_threads() == before

    with pytest.raises(ValueError, match="torch thread count must be >= 1"):
        baselines_module.pin_torch_threads(0)
    assert baselines_module.build_parser().parse_args(
        ["--manifest", "m", "report"]
    ).torch_threads == 1


# ----------------------------------------------------------------------
# The IQL learning rule (BRIEF_12 F1, F2) -- characterisation tests written against code
# already known to be correct, so each ships with its mutation executed rather than red-first
# ----------------------------------------------------------------------

#: One state, two transitions with opposite rewards and self-loops.  With ``gamma`` patched to 0
#: the Q fixed point IS the reward, so V must converge to the tau-expectile of ``{+10, -10}``,
#: which is ``20*tau - 10``: **+4 at tau=0.7 and -4 if the asymmetry is reversed**.  An 8-unit
#: margin on a closed-form target, with no corpus and no simulator.
PROBE_STATE = [1.0, 0.0, 0.0]
PROBE_GOOD_REWARD = 10.0
PROBE_BAD_REWARD = -10.0
PROBE_STEPS = 400
PROBE_LEARNING_RATE = 0.02


def _probe_table() -> Any:
    """The two-transition fixture.  Row 0 takes the good action, row 1 the bad one."""
    state = torch.tensor([PROBE_STATE, PROBE_STATE], dtype=torch.float32)
    return baselines_module.TransitionTable(
        state=state,
        next_state=state.clone(),
        action=torch.tensor([0, 1], dtype=torch.int64),
        reward=torch.tensor([PROBE_GOOD_REWARD, PROBE_BAD_REWARD], dtype=torch.float32),
        stream_index=torch.tensor([0, 0], dtype=torch.int64),
        t=torch.tensor([0, 1], dtype=torch.int64),
        reward_scale=1.0,
    )


def _empty_stats() -> Any:
    from offline.dataset import NormalizationStats

    return NormalizationStats(
        stats_version="1.0",
        split="train",
        draw_ids=(1,),
        dataset_dirs=("fixture",),
        state_mean={},
        state_std={},
        row_count={},
        rtg={},
    )


def _train_probe_iql(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gamma: float,
    polyak: float,
    steps: int = PROBE_STEPS,
) -> Any:
    """Run the REAL ``train_iql`` on the probe fixture with the declared constants patched.

    Only the constants that make the fixed point computable in closed form are moved: the
    learning rule itself -- expectile, bootstrap, advantage weighting, target network -- is the
    shipped one.
    """
    monkeypatch.setattr(baselines_module, "IQL_GAMMA", gamma)
    monkeypatch.setattr(baselines_module, "IQL_POLYAK", polyak)
    monkeypatch.setattr(baselines_module, "LEARNING_RATE", PROBE_LEARNING_RATE)
    monkeypatch.setattr(baselines_module, "WARMUP_STEPS", 1)
    return baselines_module.train_iql(
        _probe_table(),
        state_dim=len(PROBE_STATE),
        n_actions=2,
        seed=7,
        declared_gradient_steps=steps,
        batch_size=64,
        device=torch.device("cpu"),
        checkpoint_path=path,
        stats=_empty_stats(),
        scenario_id="fixture",
        provenance={},
    )


def _probe_networks(path: Path) -> dict[str, Any]:
    """The four trained networks, rebuilt from the checkpoint the training loop wrote."""
    from agent.OfflineBaselines import MLPTrunk, TrunkConfig

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TrunkConfig.from_json_obj(payload["config"])
    out: dict[str, Any] = {}
    for prefix, width in (
        ("policy", config.n_actions),
        ("q", config.n_actions),
        ("v", 1),
        ("q_target", config.n_actions),
    ):
        net = MLPTrunk(config, width)
        net.load_state_dict(
            {
                key[len(prefix) + 1 :]: value
                for key, value in payload["model"].items()
                if key.startswith(prefix + ".")
            }
        )
        out[prefix] = net.eval()
    return out


def _probe_readings(path: Path) -> dict[str, Any]:
    nets = _probe_networks(path)
    state = torch.tensor([PROBE_STATE], dtype=torch.float32)
    with torch.no_grad():
        return {
            "v": float(nets["v"](state).reshape(-1)[0]),
            "q": nets["q"](state).reshape(-1).tolist(),
            "q_target": nets["q_target"](state).reshape(-1).tolist(),
            "policy": nets["policy"](state).reshape(-1).tolist(),
        }


def _expectile(values: Sequence[float], tau: float) -> float:
    """The tau-expectile of a two-point equal-mass distribution: ``tau*hi + (1-tau)*lo``."""
    hi, lo = max(values), min(values)
    return tau * hi + (1.0 - tau) * lo


def test_the_training_loop_bootstraps_through_the_horizon_at_its_call_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 + F2c: the guarantee must hold where ``train_iql`` CALLS the helper, not only inside it.

    The review found the docstring claim false as written: ``gamma`` and ``next_values`` are both
    supplied at the call site, so the identical semantic mutation placed there survived all 58
    tests while the same mutation inside :func:`iql_targets` was killed.  This test spies on the
    module-level name the loop resolves, so it sees exactly what the loop passes.

    A spy is the right level here and not a proxy: the claim is **contractual** -- that the
    declared bootstrap is the one the training path takes -- so what is asserted is what the call
    site passes.  (The semantic question, "does using the wrong quantity change the answer", is
    what the three tests below measure behaviourally.)
    """
    original = baselines_module.iql_targets
    seen: list[dict[str, Any]] = []

    def spy(table: Any, next_values: torch.Tensor, gamma: float = baselines_module.IQL_GAMMA) -> torch.Tensor:
        target = original(table, next_values, gamma=gamma)
        seen.append(
            {
                "gamma": gamma,
                "next_values": next_values.detach().clone(),
                "reward": table.reward.detach().clone(),
                "t": table.t.detach().clone(),
                "target": target.detach().clone(),
            }
        )
        return target

    monkeypatch.setattr(baselines_module, "iql_targets", spy)
    # The horizon is the point: t == 1 is the last transition of the fixture's only stream.
    _train_probe_iql(tmp_path / "iql.pt", monkeypatch, gamma=0.99, polyak=1.0, steps=10)

    assert len(seen) == 10, "the loop must take its targets through the helper on every step"
    horizon_rows = 0
    for step, call in enumerate(seen):
        assert call["gamma"] == baselines_module.IQL_GAMMA, (
            f"step {step}: the loop passed gamma={call['gamma']!r}, so the declared discount is "
            "not the one training used"
        )
        assert float(call["next_values"].abs().max()) > 0.0, (
            f"step {step}: every bootstrap value is zero, which is a terminal horizon by another "
            "name"
        )
        last = call["t"] == 1
        horizon_rows += int(last.sum())
        assert not bool(
            torch.any(call["target"][last] == call["reward"][last])
        ), f"step {step}: a last transition's target equals its reward, i.e. it did not bootstrap"
    assert horizon_rows > 0, "no last transition was ever sampled, so nothing was tested"


def test_the_expectile_target_is_read_from_the_frozen_target_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2a, behaviourally: using the online ``Q`` in the V loss must change the answer.

    A call-spy could only assert that the target network was consulted, which is weaker than
    "consulting the wrong one changes the result".  Polyak is patched to **0**, so ``q_target``
    stays at its initialisation while ``q`` learns the fixture's rewards; the two then differ by
    an order of magnitude and V's fixed point separates them.

    Measured when this test was written: V lands at **+0.484** against the frozen target's own
    expectile of **+0.472**, while the mutant that reads the online ``q`` lands at **+2.718**.
    """
    _train_probe_iql(tmp_path / "iql.pt", monkeypatch, gamma=0.0, polyak=0.0)
    reading = _probe_readings(tmp_path / "iql.pt")

    frozen = _expectile(reading["q_target"], baselines_module.IQL_TAU)
    online = _expectile(reading["q"], baselines_module.IQL_TAU)
    assert abs(online - frozen) > 2.0, (
        "the fixture must separate the two networks, or this test cannot discriminate: "
        f"online expectile {online:.3f} against frozen {frozen:.3f}"
    )
    assert abs(reading["v"] - frozen) < abs(reading["v"] - online), (
        f"V={reading['v']:.4f} tracks the ONLINE q (expectile {online:.4f}) rather than the "
        f"frozen target network (expectile {frozen:.4f})"
    )
    assert reading["v"] < 1.5, f"V={reading['v']:.4f} is not at the frozen target's scale"


def test_the_expectile_regression_leans_toward_the_upper_tail_as_tau_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2b: with ``tau = 0.7`` the value must sit ABOVE the mean of the two returns.

    The fixed point is closed-form: the tau-expectile of ``{+10, -10}`` is ``20*tau - 10``, so
    **+4 at tau=0.7 and -4 with the asymmetry reversed** -- an 8-unit margin either side of the
    mean of 0.  Measured when this test was written: **+2.524** against the reversed branch's
    **-5.701**, so the sign alone separates them.
    """
    _train_probe_iql(tmp_path / "iql.pt", monkeypatch, gamma=0.0, polyak=1.0)
    reading = _probe_readings(tmp_path / "iql.pt")

    analytic = 20.0 * baselines_module.IQL_TAU - 10.0
    assert analytic == pytest.approx(4.0), "the fixture's closed form must be the declared one"
    assert reading["q"][0] > 5.0 > -5.0 > reading["q"][1], (
        f"Q must learn the fixture's rewards before V's expectile means anything: {reading['q']}"
    )
    assert reading["v"] > 1.0, (
        f"V={reading['v']:.4f} does not lean toward the upper tail; tau={baselines_module.IQL_TAU} "
        f"puts the fixed point at {analytic:+.1f} and the reversed asymmetry at {-analytic:+.1f}"
    )


def test_advantage_weighting_clones_the_better_action_and_not_the_worse_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2d: AWR must up-weight the action with the HIGHER advantage.

    The fixture offers one state and two actions worth +10 and -10, so the extracted policy has
    exactly one defensible answer.  Measured when this test was written: the shipped code puts
    the argmax on action 0 with a logit gap of 32.7, and the sign-flipped advantage puts it on
    action 1 with a gap of 37.0 the other way -- a categorical flip, not a margin.
    """
    _train_probe_iql(tmp_path / "iql.pt", monkeypatch, gamma=0.0, polyak=1.0)
    reading = _probe_readings(tmp_path / "iql.pt")

    logits = reading["policy"]
    assert logits[0] > logits[1], (
        f"the extracted policy prefers the -10 action: logits {logits}; the advantage weighting "
        "is up-weighting the worse action"
    )
    assert logits[0] - logits[1] > 5.0, (
        f"the preference is too weak to be a decision: logits {logits}"
    )


# ----------------------------------------------------------------------
# The declared evaluation path, the delta-proximity guard and the CLI pin
# (BRIEF_12 F4, F7, F8) -- also characterisation tests, also mutation-proved
# ----------------------------------------------------------------------


def test_the_declared_evaluation_path_passes_greedy_selection_to_the_agent(
    fixture_dataset: TrajectoryWindowDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4: what ``_baseline_factory`` PASSES is the claim, so that is what is asserted.

    The declared evaluation path is greedy (``docs/plans/p4.4.md`` section 3.1), matching the
    DT's.  The obvious test -- show that ``explore=True`` changes the number -- is the wrong
    instrument: BC reproduces the logged action on 99.8 % of training positions, so a
    near-deterministic policy can make sampling and argmax agree and the mutation would survive
    as equivalent **while proving nothing about the guard**.  The contractual question is whether
    the declared quantity is the one handed to the agent, so the assertion is on the argument.

    ``load_baseline_checkpoint`` runs for real against a real checkpoint here; only the agent is
    a recorder, because a real agent would answer with an action rather than with its arguments.
    """
    path = tmp_path / "bc_seed101.pt"
    _train_one_bc(fixture_dataset, path, steps=2)

    seen: list[dict[str, Any]] = []

    class _Recorder:
        @classmethod
        def from_checkpoint(cls, env: Any, checkpoint: str, device: Any = None) -> Any:
            return cls()

        def act(self, info: Any, explore: bool = True, update_memory: bool = True) -> np.ndarray:
            seen.append({"explore": explore, "update_memory": update_memory})
            return np.zeros(1, dtype=np.int64)

    monkeypatch.setattr(baselines_module, "BCAgent", _Recorder)
    choose = baselines_module._baseline_factory("bc", str(path), 2, None)(object())
    choose(object(), {"step": 0})

    assert len(seen) == 1
    assert seen[0]["explore"] is False, (
        f"the declared evaluation path is greedy, but the factory passed explore="
        f"{seen[0]['explore']!r}"
    )
    assert seen[0]["update_memory"] is False


def test_a_ci_endpoint_landing_on_the_equivalence_margin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F7: the plan promised an ASSERTION and the first implementation only recorded the distance.

    A6's margin has a multiplier of 1.0 that is a **choice**, so a verdict decided at a distance
    smaller than the rounding of delta itself would be decided by the rounding.  Nothing reported
    is near it -- the observed distances are 0.157, 0.916 and 0.593 -- which is why installing
    the guard now costs nothing and proves something later.
    """
    draws = {1000: 100.0, 1001: 100.0, 1002: 100.0}
    episodes = _episodes("madt", draws, seed=101)
    episodes += _episodes("mappo1000", {d: v + 0.6 for d, v in draws.items()}, seed=101)
    # every paired difference is exactly -delta, so the CI is the single point -delta and both
    # endpoints sit on the margin
    episodes += _episodes("bc", {d: v + DELTA_ATT for d, v in draws.items()}, seed=101)

    with pytest.raises(ValueError, match="A6's multiplier of 1.0 is a CHOICE"):
        baselines_artifact(
            episodes=episodes,
            training={},
            gate_a={},
            env_settings={},
            engine_seed=1000,
        )


def test_the_committed_comparisons_are_far_from_the_margin_so_the_guard_never_fired() -> None:
    """The other half of F7: the guard must not have been silently load-bearing on the result."""
    artifact = json.loads(
        (REPO_ROOT / "docs/data/p4_4_baselines.json").read_text(encoding="utf-8")
    )
    for name, entry in artifact["comparisons"].items():
        distance = entry["distance_from_ci_endpoints_to_delta"]
        assert distance > baselines_module.DELTA_PROXIMITY_TOLERANCE, f"{name}: {distance}"
        assert entry["verdict"] == entry["verdict_at_full_precision_delta"], name


def test_main_applies_the_thread_pin_before_it_does_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8: the previous test of this name exercised the helper, never ``main`` (a test name is a
    claim).  Deleting the call from ``main`` survived all 58 tests.

    ``main`` is driven far enough to have pinned and then made to fail on a missing ``--out-dir``,
    so the assertion is about the pin and not about any subcommand's work.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": "1.1",
                "run_metadata": {
                    "scenario_id": "fixture",
                    "max_steps": 360,
                    "delta_time": 10,
                    "control_mode": "acyclic",
                    "state_features": ["lane_vehicle_count"],
                    "global_reward_fn": "queue_length",
                    "local_reward_fn": "queue_length",
                    "global_reward_weight": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    before = torch.get_num_threads()
    torch.set_num_threads(max(2, before))
    unpinned = torch.get_num_threads()
    assert unpinned != 1, "the test must start from a thread count the pin has to change"
    try:
        with pytest.raises(FileNotFoundError, match="--out-dir does not exist"):
            baselines_module.main(
                [
                    "--manifest",
                    str(manifest),
                    "--out-dir",
                    str(tmp_path / "absent"),
                    "report",
                ]
            )
        assert torch.get_num_threads() == 1, (
            "main() did not pin this process, so the evaluation path runs unpinned and can "
            "deadlock as it did on 2026-08-11"
        )
    finally:
        torch.set_num_threads(before)


# ----------------------------------------------------------------------
# What the top-return filter selected (BRIEF_12 F3, rescoped by section 7.5 as a RESULT)
# ----------------------------------------------------------------------


def test_the_exact_concentration_p_value_matches_a_hand_computed_case() -> None:
    """Two blocks of two, draw two: both from one block has probability ``2/6``.

    Small enough to compute by hand and therefore the only place the enumeration can be checked
    against something other than itself.  ``C(2,2) + C(2,2) = 2`` favourable draws of ``C(4,2) =
    6``, so ``P(max block count >= 2) = 1/3`` exactly.
    """
    assert baselines_module._exact_max_block_p_value([2, 2], 2, 2) == pytest.approx(2 / 6)
    # every draw has a maximum of at least one, so P(max >= 1) is exactly 1
    assert baselines_module._exact_max_block_p_value([2, 2], 2, 1) == 1.0
    # and no draw of two can put three in a block
    assert baselines_module._exact_max_block_p_value([2, 2], 2, 3) == 0.0


def test_the_permutation_cross_check_agrees_with_the_enumeration() -> None:
    """A second route to the same number under the SAME null.

    It therefore checks the enumeration's ARITHMETIC and leaves the multivariate-hypergeometric
    null itself untested -- independence of route is not independence of assumption.  Asserted
    within four Monte Carlo standard errors, which is a real tolerance rather than a generous one.
    """
    sizes = [40] * 5
    labels = [block for block, size in enumerate(sizes) for _ in range(size)]
    exact = baselines_module._exact_max_block_p_value(sizes, 20, 10)
    monte_carlo = baselines_module._permutation_max_block_p_value(
        labels, 20, 10, iterations=20_000, rng_seed=20_260_812
    )
    assert exact == pytest.approx(0.007225300, abs=5e-9)
    assert abs(monte_carlo["p_value"] - exact) < 4.0 * monte_carlo["monte_carlo_standard_error"]


def test_a_blocks_identity_must_agree_between_its_directory_and_its_manifest(
    tmp_path: Path,
) -> None:
    """F3 attributes a result to a CHECKPOINT, so a directory name alone is a label, not evidence."""
    directory = write_dataset_dir(tmp_path, "fixture__policy__seed101")
    stream = StreamReturn(
        dataset_dir=str(directory),
        episode_file="ep000000_seed1000_draw1.npz",
        ix_id="ix_alpha",
        ix_index=1,
        episode_index=0,
        flow_draw=1,
        group=ALPHA_GROUP,
        total_return=-37.0,
    )
    assert baselines_module._behaviour_seed(stream) == 101

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_metadata"]["checkpoint"] = "output/checkpoints/cf_hz1x1__mappo__seed202.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not trustworthy"):
        baselines_module._behaviour_seed(stream)


def test_the_composition_reports_concentration_and_the_correlation_it_rests_on(
    tmp_path: Path,
) -> None:
    """The whole F3 object, on a fixture whose blocks are known by construction.

    Three blocks of different sizes, and a held-out ATT that is worse exactly where the mean
    training return is worse -- so the correlation is -1 by construction and the block supplying
    the kept stream is the better checkpoint.  Three is the floor rather than a convenience: the
    function refuses two, because a correlation over two points is a line.

    The first draft of this test used two blocks and was refused by that guard.  The guard is
    right and the setup was wrong; it is recorded in the packet rather than quietly repaired.
    """
    first = write_dataset_dir(tmp_path, "fixture__policy__seed101", draws=(1,))
    second = write_dataset_dir(tmp_path, "fixture__policy__seed202", draws=(1, 2))
    third = write_dataset_dir(tmp_path, "fixture__policy__seed303", draws=(1, 2, 3))
    dataset = TrajectoryWindowDataset(
        [first, second, third], context_length=CONTEXT, split="train"
    )
    kept = [
        s for s in stream_returns(dataset)
        if s.ix_id == "ix_alpha" and "seed101" in s.dataset_dir
    ]
    assert len(kept) == 1

    composition = baselines_module.top_return_composition(
        dataset, kept, {101: 100.0, 202: 110.0, 303: 120.0}, permutations=2_000
    )
    assert composition["per_seed_kept_counts"] == {"101": 1, "202": 0, "303": 0}
    assert composition["per_seed_stream_counts"] == {"101": 2, "202": 4, "303": 6}
    assert composition["observed_max_block_count"] == 1
    assert composition["exact_p_value"] == 1.0          # any single draw concentrates trivially
    assert composition["pearson_n"] == 3
    assert composition["pearson_r_training_return_vs_heldout_att"] == pytest.approx(-1.0)
    assert "CONCENTRATION only" in composition["exact_p_value_tests"]
    assert "multivariate hypergeometric" in composition["exact_p_value_null"]


def test_the_composition_refuses_a_correlation_it_cannot_support(tmp_path: Path) -> None:
    """Three blocks is the floor: a correlation over two points is a line, not evidence."""
    first = write_dataset_dir(tmp_path, "fixture__policy__seed101", draws=(1,))
    dataset = TrajectoryWindowDataset([first], context_length=CONTEXT, split="train")
    kept = list(stream_returns(dataset))[:1]
    with pytest.raises(ValueError, match="too few to correlate"):
        baselines_module.top_return_composition(dataset, kept, {101: 100.0}, permutations=100)


def test_the_committed_composition_is_the_one_the_reported_filter_selected() -> None:
    """The artifact's composition must describe the streams the reported %BC run trained on."""
    training = json.loads(
        (REPO_ROOT / "docs/data/p4_4_training.json").read_text(encoding="utf-8")
    )
    block = training["top_return_filter"]
    composition = block.get("composition")
    if composition is None:
        pytest.skip(
            "docs/data/p4_4_training.json carries no composition block yet: run "
            "'python -m offline.offline_baselines ... compose' to record it"
        )
    assert sum(composition["per_seed_kept_counts"].values()) == block["streams_kept"]
    assert sum(composition["per_seed_stream_counts"].values()) == block["streams_total"]
    assert composition["exact_p_value"] == pytest.approx(0.007225300, abs=5e-9)
    assert composition["pearson_n"] == 5
    assert composition["pearson_r_training_return_vs_heldout_att"] < -0.9


# ----------------------------------------------------------------------
# Review findings F5 and F6 (DEFERRED 32 and 33), folded in by BRIEF_13 section 6
# ----------------------------------------------------------------------


def _report_fixture(
    tmp_path: Path,
    *,
    bc_seeds: Sequence[int] = TRAINING_SEEDS,
    trained_seeds: Sequence[int] | None = None,
    drop_one_bc_episode: bool = False,
) -> tuple[Path, Path]:
    """A complete, synthetic P4.4 campaign on disk: gate, training artifact and one eval file.

    Every arm is separated by whole ATT units, so no verdict lands anywhere near the equivalence
    margin and the guards under test are the only ones that can fire.
    """
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "out"
    work_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    def records(arm: str, base: float, seeds: Sequence[int | None]) -> list[dict[str, Any]]:
        return [
            {
                "arm": arm,
                "seed": seed,
                "draw_id": draw,
                "att_horizon": base + 0.01 * (draw - 1000),
                "horizon_vehicle_count": 40.0,
                "episode_reward": -100.0,
            }
            for seed in seeds
            for draw in baselines_module.HELD_OUT_DRAWS
        ]

    gate_episodes = (
        records("madt", 105.0, list(TRAINING_SEEDS))
        + records("mappo1000", 106.0, list(TRAINING_SEEDS))
        + records("maxpressure", 176.0, [None])
    )
    (work_dir / "gate_a.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "compared": len(gate_episodes),
                "mismatches": [],
                "arms": list(baselines_module.CITED_ARMS),
                "episodes": gate_episodes,
            }
        ),
        encoding="utf-8",
    )

    bc_episodes = records("bc", 104.0, list(bc_seeds))
    if drop_one_bc_episode:
        bc_episodes = bc_episodes[:-1]
    (work_dir / "eval_bc.json").write_text(
        json.dumps({"method": "bc", "episodes": bc_episodes}), encoding="utf-8"
    )

    (out_dir / "p4_4_training.json").write_text(
        json.dumps(
            {
                "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
                "declared_in": "a fixture",
                "raise_available": False,
                "seeds": list(TRAINING_SEEDS),
                "training_draw_ids": [1, 2, 3],
                "iql": {},
                "top_return_filter": {},
                "runs": [
                    {"method": "bc", "seed": seed, "checkpoint": "x.pt"}
                    for seed in (TRAINING_SEEDS if trained_seeds is None else trained_seeds)
                ],
            }
        ),
        encoding="utf-8",
    )
    return out_dir, work_dir


def _report_args(methods: str = "bc") -> Any:
    return argparse.Namespace(methods=methods, engine_seed=1000)


def test_the_report_derives_its_expected_runs_from_the_declaration_not_from_the_episodes(
    tmp_path: Path,
) -> None:
    """F5a: the completeness check must not be a tautology over the data it checks.

    ``_run_report`` once built its "requested runs" list from the very episodes it then checked,
    which can be restored with the whole suite green (review finding F5).  Dropping one episode
    from the evaluation file must therefore be fatal.

    Killed by: deriving ``requested`` from ``episodes``.
    """
    out_dir, work_dir = _report_fixture(tmp_path)
    assert baselines_module._run_report(_report_args(), {}, out_dir, work_dir) == 0
    assert (out_dir / "p4_4_baselines.json").is_file()

    out_dir, work_dir = _report_fixture(tmp_path / "second", drop_one_bc_episode=True)
    with pytest.raises(ValueError, match="incomplete campaign"):
        baselines_module._run_report(_report_args(), {}, out_dir, work_dir)


def test_a_method_trained_on_fewer_seeds_than_declared_cannot_be_reported(
    tmp_path: Path,
) -> None:
    """F5b: the expected seed set is the DECLARATION's, for every arm.

    It was ``TRAINING_SEEDS`` for ``madt``/``mappo1000`` and ``training["runs"]`` for the
    baselines, so a three-seed method passed: both sides of the comparison shrank together.

    Killed by: deriving the baseline arms' seed set from ``training["runs"]``.
    """
    out_dir, work_dir = _report_fixture(
        tmp_path, bc_seeds=(101, 202, 303), trained_seeds=(101, 202, 303)
    )
    with pytest.raises(ValueError, match="incomplete campaign"):
        baselines_module._run_report(_report_args(), {}, out_dir, work_dir)


def test_the_committed_training_artifact_trained_every_declared_seed(tmp_path: Path) -> None:
    """The F5b fix is a no-op on the merged result, and that is shown rather than asserted."""
    training = json.loads(
        (REPO_ROOT / "docs/data/p4_4_training.json").read_text(encoding="utf-8")
    )
    for method in METHODS:
        seeds = sorted(int(r["seed"]) for r in training["runs"] if r["method"] == method)
        assert seeds == sorted(TRAINING_SEEDS), method


def test_the_unbalanced_design_cross_check_fires_when_the_design_is_unbalanced() -> None:
    """F6a: a working guard with no regression protection -- it survives being disabled.

    The recovered fraction is computed twice, directly and through the paired route, and the two
    agree exactly only when seed is crossed with draw.  Dropping one arm episode breaks that.

    Killed by: removing the two-route comparison.
    """
    draws = {1000: 100.0, 1001: 101.0, 1002: 102.0}
    episodes = _episodes("madt", draws, seed=101) + _episodes("madt", draws, seed=202)
    episodes += _episodes("mappo1000", {d: v + 1.0 for d, v in draws.items()}, seed=101)
    episodes += _episodes("mappo1000", {d: v + 1.0 for d, v in draws.items()}, seed=202)
    balanced = _episodes("bc", {d: v + 3.0 for d, v in draws.items()}, seed=101)
    balanced += _episodes("bc", {d: v + 5.0 for d, v in draws.items()}, seed=202)

    artifact = baselines_artifact(
        episodes=episodes + balanced,
        training={},
        gate_a={},
        env_settings={},
        engine_seed=1000,
    )
    assert artifact["comparisons"]["madt_vs_bc"]["recovered_fraction"] == pytest.approx(
        artifact["comparisons"]["madt_vs_bc"]["recovered_fraction_paired_route"], abs=1e-12
    )

    with pytest.raises(ValueError, match="two routes"):
        baselines_artifact(
            episodes=episodes + balanced[:-1],
            training={},
            gate_a={},
            env_settings={},
            engine_seed=1000,
        )


def test_the_delta_rounding_cross_check_fires_on_a_verdict_that_turns_on_the_rounding() -> None:
    """F6b: the second guard with no regression protection.

    A6's multiplier of 1.0 is a choice, so a verdict that differs between the declared
    ``0.6263`` and its full-precision derivation ``0.6262756469347437`` is not a verdict.  Every
    paired difference here is exactly ``0.62629``, which lies between the two.

    Killed by: removing the disagreement check -- after which the proximity guard raises a
    DIFFERENT error, which is why ``match=`` names a phrase unique to THIS one's message.

    ⚠️ The first draft matched on ``"rounding"`` alone and the mutation SURVIVED, because the
    proximity guard's message also contains that word (*"decided by the margin's rounding"*).
    The test passed for the wrong reason and only the mutation exposed it.  Recorded here rather
    than quietly repaired -- and see the packet: the two guards are not independent, because the
    two deltas differ by 2.4353e-05, which is inside DELTA_PROXIMITY_TOLERANCE.
    """
    assert DELTA_ATT_DERIVATION < 0.62629 < DELTA_ATT, "the fixture must straddle the two deltas"
    draws = {1000: 100.0, 1001: 101.0, 1002: 102.0}
    episodes = _episodes("madt", draws, seed=101)
    episodes += _episodes("mappo1000", {d: v + 2.0 for d, v in draws.items()}, seed=101)
    episodes += _episodes("bc", {d: v - 0.62629 for d, v in draws.items()}, seed=101)

    with pytest.raises(ValueError, match="depends on delta's rounding"):
        baselines_artifact(
            episodes=episodes,
            training={},
            gate_a={},
            env_settings={},
            engine_seed=1000,
        )


# ----------------------------------------------------------------------
# P4.6 (BRIEF_17 section 11, finding A2): the reporting path is generalised additively.
# The default path must keep behaving EXACTLY as it did -- the regression gate that
# docs/data/p4_4_baselines.json regenerates byte-identically is what makes the generalisation
# safe, and these tests are its unit-level half.
# ----------------------------------------------------------------------


def test_the_artifact_can_report_against_a_behaviour_reference_that_is_not_mappo1000() -> None:
    """P4.6's random and fixedtime tiers have no mappo1000 arm; the report must still build."""
    draws = {1000: 300.0, 1001: 310.0, 1002: 320.0}
    episodes = _episodes("madt", draws, seed=101)
    episodes += _episodes("random", {d: v + 90.0 for d, v in draws.items()}, seed=101)
    for method in METHODS:
        episodes += _episodes(method, {d: v + 1.0 for d, v in draws.items()}, seed=101)

    payload = baselines_artifact(
        episodes=episodes,
        training={"declared_gradient_steps": DECLARED_GRADIENT_STEPS, "seeds": []},
        gate_a={"status": "PASS", "compared": 0, "mismatches": 0},
        env_settings={"max_steps": 360},
        engine_seed=1000,
        behaviour_reference="random",
        emit_verdicts=False,
    )
    assert payload["behaviour_reference"] == "random"
    assert "random_vs_madt" in payload["behaviour_policy_comparisons"]
    assert payload["cells"]["random"]["policy_source"] == "stochastic_heuristic"

    for method in METHODS:
        entry = payload["comparisons"][f"madt_vs_{method}"]
        assert "recovered_fraction" in entry, "the recovered fraction is reference-relative, not delta-relative"
        assert "verdict" not in entry
        assert "delta" not in entry
    for absent in (
        "equivalence_margin_delta",
        "equivalence_margin_delta_derivation",
        "decision_rule",
        "registered_forecasts",
    ):
        assert absent not in payload, f"{absent} is a verdict-mode key"
    assert "no equivalence verdict" in payload["reporting_rule"]


def test_the_artifact_without_its_behaviour_reference_is_still_refused() -> None:
    """Generalising the reference must not weaken the requirement that there IS one."""
    draws = {1000: 100.0}
    episodes = _episodes("madt", draws, seed=101) + _episodes("bc", draws, seed=101)
    with pytest.raises(ValueError, match="the artifact needs the 'random' arm"):
        baselines_artifact(
            episodes=episodes,
            training={"declared_gradient_steps": DECLARED_GRADIENT_STEPS, "seeds": []},
            gate_a={"status": "PASS", "compared": 0, "mismatches": 0},
            env_settings={"max_steps": 360},
            engine_seed=1000,
            behaviour_reference="random",
            emit_verdicts=False,
        )


def test_the_default_reporting_path_is_unchanged_by_the_generalisation() -> None:
    """The default call must produce exactly the keys and values it produced before P4.6.

    Byte-level, on everything but the self-describing ``runtime`` block: this is the unit-level
    form of the regression gate declared in ``docs/plans/p4.6.md`` section 15.2.
    """
    common = {
        "episodes": _artifact_episodes(),
        "training": {"declared_gradient_steps": DECLARED_GRADIENT_STEPS, "seeds": []},
        "gate_a": {"status": "PASS", "compared": 0, "mismatches": 0},
        "env_settings": {"max_steps": 360},
        "engine_seed": 1000,
    }
    implicit = baselines_artifact(**common)
    explicit = baselines_artifact(**common, behaviour_reference="mappo1000", emit_verdicts=True)

    implicit.pop("runtime")
    explicit.pop("runtime")
    assert json.dumps(implicit, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert implicit["equivalence_margin_delta"] == DELTA_ATT
    assert "behaviour_reference" not in implicit, (
        "the default payload must carry exactly the key set the committed artifact carries"
    )
    assert all(f"madt_vs_{method}" in implicit["comparisons"] for method in METHODS)
    assert all("verdict" in entry for entry in implicit["comparisons"].values())
