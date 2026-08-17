"""P5.1's declaration, its training loop and the scorers for the registered predictions.

⚠️ **The scorers are tested here because ``PREREGISTRATION`` A8(a) requires the rule AND its
scorer to predate every number.**  P4.7's review verified exactly that by timestamp; the same
property is arranged here by writing the scorers before the campaign runs and pinning their
boundary behaviour on synthetic cells whose answers are known by construction.

The prompt rule is double-computed: the test re-derives every intersection's declared target and
scale from the **raw** ``ix{i}_local_reward`` arrays, accumulating backwards in float64 and casting
to float32 -- the order ``offline/dataset.py::_returns_to_go`` documents -- and compares with
``==``, never a tolerance.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from offline.dt_gate import EpisodeResult
from offline.spatial_mixing import (
    ARTIFACT_FORMAT_VERSION,
    CONTEXT_LENGTH,
    DECLARATION_FORMAT_VERSION,
    DECLARED_GRADIENT_STEPS,
    DT_METHODS,
    EXPECTED_NODES,
    JOINT_BATCH_SIZE,
    METHODS,
    TIER_DIRS,
    assert_declaration_matches_corpus,
    collapse_criterion,
    declaration_artifact,
    per_node_prompts,
    per_seed_advantages,
    score_p1,
    score_p2,
    tier_dataset,
    tier_dirs,
    train_spatial_dt,
)

SEEDS = (101, 202, 303, 404, 505)


def _corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else Path(__file__).resolve().parents[1] / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(
            f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to a "
            "collected corpus root to run the P5.1 declaration tests"
        )
    return candidate


# ----------------------------------------------------------------------
# Constants that the plan and the campaign both depend on
# ----------------------------------------------------------------------


def test_the_declared_constants_are_the_ones_the_plan_registered():
    assert DECLARED_GRADIENT_STEPS == 40_000
    assert JOINT_BATCH_SIZE == 64
    assert CONTEXT_LENGTH == 20
    assert EXPECTED_NODES == 16
    assert METHODS == ("dt_spatial", "dt_nomix", "bc", "bc_top10", "iql")
    assert DT_METHODS == ("dt_spatial", "dt_nomix")
    assert len(TIER_DIRS) == 5


def test_the_batch_size_gives_the_same_epoch_count_as_the_hz1x1_dt_arms():
    """D5's justification, checked arithmetically rather than asserted in prose.

    P4/P4.6 trained on 72,000 hz1x1 windows at batch 64 for 40,000 steps.  grid4x4 has 72,000
    JOINT windows (200 episodes x 360 decisions), so the same batch size gives the same number of
    passes over the corpus -- which is why this batch size and not another.
    """
    hz1x1_windows = 200 * 360
    grid4x4_joint_windows = 200 * 360
    assert hz1x1_windows == grid4x4_joint_windows
    epochs = DECLARED_GRADIENT_STEPS * JOINT_BATCH_SIZE / grid4x4_joint_windows
    assert round(epochs, 1) == 35.6


# ----------------------------------------------------------------------
# The prompt rule, double-computed from the raw reward arrays
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_tier():
    """Three draws from one seed directory: enough streams for the rule, fast to build."""
    root = _corpus_root()
    directory = root / TIER_DIRS[0]
    if not (directory / "manifest.json").is_file():
        pytest.skip(f"{directory} is not a collected dataset directory")
    from offline.dataset import TrajectoryWindowDataset

    dataset = TrajectoryWindowDataset(
        [directory], context_length=CONTEXT_LENGTH, split="train",
        draw_ids=[1, 2, 3], normalize=True,
    )
    from offline.offline_baselines import stream_returns

    return dataset, stream_returns(dataset)


def _raw_returns_by_node(directory: Path, draws: set[int]) -> dict[str, list[float]]:
    """INDEPENDENT ROUTE: total return per (episode, intersection) from the raw npz arrays.

    Accumulated backwards in float64 and cast to float32, which is the order
    ``offline/dataset.py::_returns_to_go`` documents, so the comparison can be exact.
    """
    import json

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    out: dict[str, list[float]] = defaultdict(list)
    for entry in manifest["episodes"]:
        if int(entry["flow_draw"]) not in draws:
            continue
        payload = np.load(directory / entry["filename"], allow_pickle=True)
        ids = [str(x) for x in payload["ix_ids"]]
        for index, ix_id in enumerate(ids):
            rewards = payload[f"ix{index}_local_reward"]
            total = 0.0
            for t in range(int(rewards.shape[0]) - 1, -1, -1):
                total += float(rewards[t])
            out[ix_id].append(float(np.float32(total)))
    return dict(out)


def test_every_nodes_prompt_reproduces_from_the_raw_reward_arrays(small_tier):
    _, streams = small_tier
    prompts = per_node_prompts(streams)
    expected = _raw_returns_by_node(_corpus_root() / TIER_DIRS[0], {1, 2, 3})

    assert set(prompts) == set(expected) == set(f"{c}{r}" for c in "ABCD" for r in "0123")
    for ix_id, prompt in prompts.items():
        values = expected[ix_id]
        assert prompt.n_streams == len(values)
        assert prompt.target_rtg == max(values)
        assert prompt.rtg_scale == max(abs(min(values)), abs(max(values)))


def test_the_rule_reduces_to_p4_6_on_a_single_intersection(small_tier):
    """The property that makes this a disambiguation rather than a change (D3).

    Restricted to one intersection's streams, the per-node rule returns exactly what
    ``method_tier_grid.recomputed_target_and_scale`` returns for that same set.
    """
    from offline.method_tier_grid import recomputed_target_and_scale

    _, streams = small_tier
    prompts = per_node_prompts(streams)
    for ix_id, prompt in prompts.items():
        own = [s for s in streams if s.ix_id == ix_id]
        target, scale = recomputed_target_and_scale(own)
        assert prompt.target_rtg == target
        assert prompt.rtg_scale == scale


def test_the_prompts_differ_across_intersections_so_a_global_scalar_is_not_equivalent(small_tier):
    """Without this the whole per-node design would be a no-op and D3 would be arguing nothing."""
    _, streams = small_tier
    prompts = per_node_prompts(streams)
    targets = {p.target_rtg for p in prompts.values()}
    assert len(targets) > 1

    global_max = max(p.target_rtg for p in prompts.values())
    below = [ix for ix, p in prompts.items() if p.target_rtg < global_max]
    assert len(below) >= 12, (
        "a global scalar must be out of support for most nodes, or D3's argument does not apply "
        f"to this slice: only {len(below)} of 16 are below the global maximum"
    )


def test_a_node_with_no_streams_is_refused():
    with pytest.raises(ValueError, match="no streams"):
        per_node_prompts([])


# ----------------------------------------------------------------------
# The declaration guard
# ----------------------------------------------------------------------


def test_a_declaration_that_matches_the_corpus_is_accepted(small_tier):
    _, streams = small_tier
    prompts = per_node_prompts(streams)
    declared = {
        ix: {"target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale} for ix, p in prompts.items()
    }
    report = assert_declaration_matches_corpus(declared, streams)
    assert report["n_nodes"] == EXPECTED_NODES
    assert report["rule"].startswith("target_rtg = max")


def test_a_declaration_that_disagrees_by_one_unit_is_refused(small_tier):
    _, streams = small_tier
    prompts = per_node_prompts(streams)
    declared = {
        ix: {"target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale} for ix, p in prompts.items()
    }
    declared["C1"]["target_rtg"] += 1.0
    with pytest.raises(ValueError, match="declares target_rtg"):
        assert_declaration_matches_corpus(declared, streams)


def test_a_declaration_missing_a_node_is_refused(small_tier):
    _, streams = small_tier
    prompts = per_node_prompts(streams)
    declared = {
        ix: {"target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale}
        for ix, p in prompts.items()
        if ix != "D3"
    }
    with pytest.raises(ValueError, match="declaration covers"):
        assert_declaration_matches_corpus(declared, streams)


# ----------------------------------------------------------------------
# The registered prediction scorers, on synthetic cells with known answers
# ----------------------------------------------------------------------


def _cells(arm: str, values_by_draw: dict[int, float], seeds=SEEDS) -> list[EpisodeResult]:
    return [
        EpisodeResult(
            arm=arm, seed=seed, draw_id=draw, att_horizon=value,
            horizon_vehicle_count=10.0, episode_reward=-1.0,
        )
        for draw, value in values_by_draw.items()
        for seed in seeds
    ]


#: A STABLE seed per arm.  ``hash(str)`` is randomised per process unless PYTHONHASHSEED is set,
#: so deriving a fixture seed from it makes the test's own data differ between runs -- which is
#: exactly the non-determinism this project treats as a defect.  Found when a P2 fixture ranked
#: its arms differently on a second run.
_ARM_SEED: dict[str, int] = {
    "dt_spatial": 1, "dt_nomix": 2, "bc": 3, "bc_top10": 4, "iql": 5, "random": 6,
}


def _flat(arm: str, value: float, n: int = 100, jitter: float = 0.0) -> list[EpisodeResult]:
    """A synthetic cell whose realised mean is EXACTLY *value*, with real per-draw spread.

    The noise is centred (``noise -= noise.mean()``) so the arm ordering is fixed by construction
    and a rank test cannot flip on a draw of the fixture's own RNG, while the paired difference
    still carries the variance an interval test needs.
    """
    rng = np.random.default_rng(_ARM_SEED[arm])
    noise = rng.standard_normal(n)
    noise = noise - noise.mean()
    return _cells(
        arm,
        {
            1000 + offset: value + jitter * float(noise[offset])
            for offset in range(n)
        },
    )


def test_p1_holds_when_mixing_is_decisively_better():
    result = score_p1(_flat("dt_spatial", 100.0, jitter=1.0), _flat("dt_nomix", 120.0, jitter=1.0))
    assert result["outcome"] == "HELD"
    assert result["mean_difference"] < 0
    assert result["ci95_high"] < 0
    assert result["ci95_width"] > 0


def test_p1_fails_when_the_control_is_decisively_better():
    """The other direction is a decisive RESULT and is never folded into 'inconclusive'."""
    result = score_p1(_flat("dt_spatial", 130.0, jitter=1.0), _flat("dt_nomix", 100.0, jitter=1.0))
    assert result["outcome"] == "FAILED"
    assert result["ci95_low"] > 0


def test_p1_does_not_resolve_when_the_interval_spans_zero():
    result = score_p1(_flat("dt_spatial", 100.0, jitter=8.0), _flat("dt_nomix", 100.0, jitter=8.0))
    assert result["outcome"] == "NOT RESOLVED"
    assert result["ci95_low"] < 0 < result["ci95_high"]
    assert "ci95_width" in result


def test_p1_reports_the_per_seed_dimension_the_draw_level_ci_cannot_see():
    result = score_p1(_flat("dt_spatial", 100.0, jitter=1.0), _flat("dt_nomix", 120.0, jitter=1.0))
    assert set(result["per_seed"]["by_seed"]) == {str(s) for s in SEEDS}
    assert "between_seed_sd" in result["per_seed"]
    assert result["per_seed"]["n_seeds"] == 5


def test_p2_holds_only_when_the_rank_and_the_resolution_both_hold():
    episodes = {
        "dt_spatial": _flat("dt_spatial", 90.0, jitter=1.0),
        "dt_nomix": _flat("dt_nomix", 110.0, jitter=1.0),
        "bc": _flat("bc", 105.0, jitter=1.0),
        "bc_top10": _flat("bc_top10", 103.0, jitter=1.0),
        "iql": _flat("iql", 104.0, jitter=1.0),
    }
    result = score_p2(episodes)
    assert result["p2a"]["outcome"] == "HELD"
    assert result["p2a"]["rank"] == 1
    assert result["p2b"]["outcome"] == "HELD"
    assert result["outcome"] == "HELD"
    assert result["p2b"]["best_other_arm"] == "bc_top10"


def test_p2a_fails_when_a_baseline_leads_as_it_has_on_all_eight_prior_tiers():
    episodes = {
        "dt_spatial": _flat("dt_spatial", 105.0, jitter=1.0),
        "dt_nomix": _flat("dt_nomix", 110.0, jitter=1.0),
        "bc": _flat("bc", 106.0, jitter=1.0),
        "bc_top10": _flat("bc_top10", 100.0, jitter=1.0),
        "iql": _flat("iql", 104.0, jitter=1.0),
    }
    result = score_p2(episodes)
    assert result["p2a"]["outcome"] == "FAILED"
    assert result["p2a"]["rank"] == 3
    assert result["outcome"] == "FAILED"


def test_p2_reports_its_two_conjuncts_separately():
    """P4.7's M1: a conjunction whose parts differ in robustness is never one verdict."""
    episodes = {
        "dt_spatial": _flat("dt_spatial", 100.0, jitter=8.0),
        "dt_nomix": _flat("dt_nomix", 110.0, jitter=1.0),
        "bc": _flat("bc", 100.4, jitter=8.0),
        "bc_top10": _flat("bc_top10", 101.0, jitter=1.0),
        "iql": _flat("iql", 104.0, jitter=1.0),
    }
    result = score_p2(episodes)
    assert result["p2a"]["outcome"] == "HELD"
    assert result["p2b"]["outcome"] == "NOT RESOLVED"
    assert result["outcome"] == "FAILED"
    assert result["p2a"] is not result["p2b"]


def test_p2_refuses_a_cell_that_is_missing_an_arm():
    with pytest.raises(ValueError, match="P2 needs every method arm"):
        score_p2({"dt_spatial": _flat("dt_spatial", 100.0)})


def test_the_collapse_criterion_fires_only_against_the_shared_draw_reference():
    collapsed = collapse_criterion(_flat("dt_spatial", 400.0, jitter=5.0),
                                   _flat("random", 260.0, jitter=5.0))
    assert collapsed["outcome"] == "COLLAPSED"
    assert collapsed["mean_difference"] > 0
    assert collapsed["ci95_low"] > 0

    healthy = collapse_criterion(_flat("dt_spatial", 150.0, jitter=5.0),
                                 _flat("random", 260.0, jitter=5.0))
    assert healthy["outcome"] == "NOT COLLAPSED"


def test_the_collapse_criterion_records_dtlight_as_context_and_not_as_a_threshold():
    result = collapse_criterion(_flat("dt_spatial", 150.0), _flat("random", 260.0))
    assert result["dtlight_grid4x4_reference"] == 446.8
    assert "not our threshold" in result["dtlight_note"]
    # The verdict must not depend on that number in any way.
    assert result["outcome"] == "NOT COLLAPSED"


def test_the_collapse_criterion_does_not_fire_on_an_unresolved_difference():
    result = collapse_criterion(_flat("dt_spatial", 261.0, jitter=40.0),
                                _flat("random", 260.0, jitter=40.0))
    assert result["outcome"] == "NOT RESOLVED"


def test_per_seed_advantages_sees_a_seed_reversal_the_pooled_ci_hides():
    """P4.7's M1 reproduced deliberately: a mean advantage that reverses on two of five seeds."""
    left, right = [], []
    for index, seed in enumerate(SEEDS):
        offset = -30.0 if index < 3 else +25.0
        for draw in range(1000, 1100):
            left.append(EpisodeResult("dt_spatial", seed, draw, 100.0 + offset, 10.0, -1.0))
            right.append(EpisodeResult("dt_nomix", seed, draw, 100.0, 10.0, -1.0))

    report = per_seed_advantages(left, right)
    signs = [np.sign(v) for v in report["by_seed"].values()]
    assert sorted(signs) == [-1.0, -1.0, -1.0, 1.0, 1.0]
    assert report["n_seeds_favouring_left"] == 3
    assert report["reverses_on_n_seeds"] == 2
    assert report["between_seed_sd"] > 0


# ----------------------------------------------------------------------
# Training: the two arms differ only by the mask
# ----------------------------------------------------------------------


def test_the_two_arms_consume_the_same_batches_and_differ_only_by_the_mask(tmp_path, small_tier):
    """The primary comparison is only controlled if the batch stream is identical."""
    from offline.joint_windows import build_joint_index, stack_joint
    from offline.roadnet_graph import adjacency_from_sim_config

    dataset, streams = small_tier
    node_ids = tuple(f"{c}{r}" for c in "ABCD" for r in "0123")
    index = build_joint_index(dataset, node_ids)
    stacked = stack_joint(dataset, index)
    prompts = per_node_prompts(streams)
    repo = Path(__file__).resolve().parents[1]
    adjacency = adjacency_from_sim_config(
        repo / "configs" / "sim" / "cityflow_grid4x4.json", node_ids
    )

    common = dict(
        index=index, adjacency=adjacency, prompts=prompts, stats=dataset.stats,
        state_dim=40, n_actions=8, gradient_steps=3, batch_size=4,
        device=torch.device("cpu"), provenance={}, seed=101,
    )
    spatial = train_spatial_dt(
        stacked, method="dt_spatial", checkpoint_path=tmp_path / "s.pt", **common
    )
    control = train_spatial_dt(
        stacked, method="dt_nomix", checkpoint_path=tmp_path / "c.pt", **common
    )

    assert spatial.gradient_steps == control.gradient_steps == 3
    assert len(spatial.losses) == len(control.losses) == 3
    # Same seed, same data, same batch order -- the first loss differs ONLY because of the mask.
    assert spatial.losses != control.losses

    left = torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=False)
    right = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=False)
    assert left["config"]["spatial_mixing"] is True
    assert right["config"]["spatial_mixing"] is False
    assert np.array_equal(np.asarray(right["spatial_mask"]), np.eye(16, dtype=bool))
    assert not np.array_equal(np.asarray(left["spatial_mask"]), np.eye(16, dtype=bool))
    assert int(np.asarray(left["spatial_mask"]).sum()) == 2 * 24 + 16


def test_training_is_deterministic_for_a_given_seed(tmp_path, small_tier):
    from offline.joint_windows import build_joint_index, stack_joint
    from offline.roadnet_graph import adjacency_from_sim_config

    dataset, streams = small_tier
    node_ids = tuple(f"{c}{r}" for c in "ABCD" for r in "0123")
    index = build_joint_index(dataset, node_ids)
    stacked = stack_joint(dataset, index)
    repo = Path(__file__).resolve().parents[1]
    adjacency = adjacency_from_sim_config(
        repo / "configs" / "sim" / "cityflow_grid4x4.json", node_ids
    )
    common = dict(
        index=index, adjacency=adjacency, prompts=per_node_prompts(streams),
        stats=dataset.stats, state_dim=40, n_actions=8, gradient_steps=3, batch_size=4,
        device=torch.device("cpu"), provenance={}, seed=202, method="dt_spatial",
    )
    first = train_spatial_dt(stacked, checkpoint_path=tmp_path / "a.pt", **common)
    second = train_spatial_dt(stacked, checkpoint_path=tmp_path / "b.pt", **common)
    assert first.losses == second.losses


def test_an_unknown_method_is_refused(tmp_path, small_tier):
    from offline.joint_windows import build_joint_index, stack_joint
    from offline.roadnet_graph import adjacency_from_sim_config

    dataset, streams = small_tier
    node_ids = tuple(f"{c}{r}" for c in "ABCD" for r in "0123")
    index = build_joint_index(dataset, node_ids)
    repo = Path(__file__).resolve().parents[1]
    adjacency = adjacency_from_sim_config(
        repo / "configs" / "sim" / "cityflow_grid4x4.json", node_ids
    )
    with pytest.raises(ValueError, match="not a spatial arm"):
        train_spatial_dt(
            stack_joint(dataset, index), index=index, method="bc", seed=1,
            adjacency=adjacency, prompts=per_node_prompts(streams), stats=dataset.stats,
            state_dim=40, n_actions=8, gradient_steps=1, batch_size=2,
            device=torch.device("cpu"), checkpoint_path=tmp_path / "x.pt", provenance={},
        )


# ----------------------------------------------------------------------
# The declaration artifact
# ----------------------------------------------------------------------


def test_the_declaration_artifact_records_the_graph_and_every_prompt(small_tier):
    from offline.joint_windows import build_joint_index
    from offline.roadnet_graph import adjacency_from_sim_config

    dataset, streams = small_tier
    node_ids = tuple(f"{c}{r}" for c in "ABCD" for r in "0123")
    index = build_joint_index(dataset, node_ids)
    repo = Path(__file__).resolve().parents[1]
    adjacency = adjacency_from_sim_config(
        repo / "configs" / "sim" / "cityflow_grid4x4.json", node_ids
    )
    payload = declaration_artifact(
        _corpus_root(), adjacency, index, per_node_prompts(streams), dataset.stats
    )

    assert payload["format_version"] == DECLARATION_FORMAT_VERSION
    assert payload["graph"]["undirected_edges"] == 24
    assert payload["graph"]["degree_histogram"] == {"2": 4, "3": 8, "4": 4}
    assert payload["graph"]["roads_route_agrees"] is True
    assert payload["node_order"] == list(node_ids)
    assert len(payload["prompts"]) == EXPECTED_NODES
    assert payload["declared_gradient_steps"] == DECLARED_GRADIENT_STEPS
    assert payload["batch_size"] == JOINT_BATCH_SIZE
    assert payload["methods"] == list(METHODS)
    assert payload["prompt_rule"].startswith("target_rtg = max")


def test_the_artifact_version_is_stated():
    assert ARTIFACT_FORMAT_VERSION == "p5.1-spatial/1.0"
    assert DECLARATION_FORMAT_VERSION == "p5.1-declaration/1.0"


def test_a_smoke_artifact_can_never_be_mistaken_for_a_cell(tmp_path):
    """The smoke facility is quarantined BY NAME, so it cannot contaminate a campaign.

    ``--smoke-draws`` exists to execute the evaluation wiring once before an overnight run
    (coordinator ruling, 2026-08-17; PROJECT_PLAN section 7's rule of 2026-08-14).  It writes
    ``smoke_eval_<method>.json``; the report and the campaign's completeness check both read
    ``eval_<method>.json`` only.  A partial cell therefore cannot be picked up by either.
    """
    import argparse

    from offline.spatial_mixing import _run_report

    work = tmp_path / "work"
    work.mkdir()
    for method in METHODS:
        (work / f"smoke_eval_{method}.json").write_text(
            json.dumps(
                {
                    "arm": f"{method}@grid4x4_mappo1000", "smoke": True,
                    "cell": {"att_horizon_mean": 1.0},
                    "episodes": [
                        {"arm": f"{method}@grid4x4_mappo1000", "seed": 101, "draw_id": 1000,
                         "att_horizon": 1.0, "horizon_vehicle_count": 1.0, "episode_reward": -1.0}
                    ],
                }
            ),
            encoding="utf-8",
        )
    args = argparse.Namespace(work_dir=str(work), out_dir=str(tmp_path / "out"))
    # Every smoke file is present and NONE of them is a cell, so the report refuses.
    with pytest.raises(ValueError, match="the report is missing"):
        _run_report(args)
    assert not (tmp_path / "out" / "p5_1_grid.json").exists()


def test_the_campaign_completeness_check_reads_eval_not_smoke_eval():
    """The other half: the shipped script's glob must not match a smoke artifact."""
    script = (Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p5_1.sh").read_text(
        encoding="utf-8"
    )
    assert 'work / f"eval_{arm}.json"' in script
    assert "smoke" not in script.lower(), (
        "the campaign script must not know about the smoke facility at all; if it grows a "
        "smoke branch, a partial cell could reach the completeness check"
    )


def test_tier_dirs_refuses_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a collected dataset directory"):
        tier_dirs(tmp_path)
