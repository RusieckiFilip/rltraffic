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
from offline.dt_gate import EpisodeResult, mean_ci95, stack_dataset, wilcoxon_signed_rank
from offline.offline_baselines import (
    ARTIFACT_FORMAT_VERSION,
    DECLARED_GRADIENT_STEPS,
    DELTA_ATT,
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
    with pytest.raises(ValueError, match="group"):
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
    with pytest.raises(FileNotFoundError, match="does not exist"):
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
    with pytest.raises(FileNotFoundError, match="does not exist"):
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
    with pytest.raises(ValueError, match="declared"):
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
    with pytest.raises(ValueError, match="held-out"):
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

    with pytest.raises(ValueError, match="must be >= 1"):
        baselines_module.pin_torch_threads(0)
    assert baselines_module.build_parser().parse_args(
        ["--manifest", "m", "report"]
    ).torch_threads == 1
