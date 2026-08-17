"""P5.1: train and evaluate the spatial mixing layer on ``cf_grid4x4__mappo1000``.

Artifact format version: ``p5.1-spatial/1.0``.  Declaration format version:
``p5.1-declaration/1.0``.

WHAT THIS TASK DECIDES
----------------------
Every offline arm this project has measured is independent per intersection, and across eight
tiers the Decision Transformer leads zero (``PROJECT_PLAN`` section 1b, R7).  Section 1b's **R5**
leaves exactly two explanations standing -- the single-intersection setting, and the architecture --
and this is the only experiment that separates them.  The registered predictions, their scoring
rules and the declared collapse criterion are in ``docs/plans/p5.1.md``, committed before any code.

THE PROTOCOL IS P4'S, REUSED RATHER THAN RESTATED
--------------------------------------------------
Rollouts go through ``offline.dt_gate.evaluate_arm`` with env settings read from the collection
manifest by ``env_settings_from_manifest``; cells come from ``dt_gate._cell``, pairing from
``dt_gate._paired``, descriptives from ``mean_ci95``, the paired test from
``wilcoxon_signed_rank`` and the comparison object from ``offline_baselines.paired_comparison``.
Importing them -- including the private helpers -- is deliberate and is the convention
``offline/offline_baselines.py`` established: a second implementation of the same protocol is
exactly how two arms stop being comparable.

THE RETURN-PROMPT RULE, AND WHY IT IS PER INTERSECTION
--------------------------------------------------------
P4.6's rule is *target_rtg = max episode return in the training set; rtg_scale = max|return|*, one
scalar, because ``cf_hz1x1`` has one intersection.  Here it is applied **within each
intersection's own stream set**, which reduces exactly to P4.6 on a one-intersection scenario and
is therefore a disambiguation at a granularity that did not exist when the rule was written, not a
change to it.  Coordinator ruling of 2026-08-17; ``docs/plans/p5.1.md`` decision D3.

Measured on this tier over its 200 episodes: the global maximum return is **-64.0** while
**15 of 16** intersections have an own-best that never reaches it.  A single global scalar would
condition 15 of 16 nodes outside their own training support -- the infeasible-RTG failure mode
``PROJECT_PLAN`` section 9 rates as a live risk, and one that would risk *manufacturing* the
collapse this task exists to test for.

⚠️ NO EQUIVALENCE VERDICTS
---------------------------
``BRIEF_17`` section 4, inherited: A6's delta is ``mappo1000``-on-``cf_hz1x1``-specific and no
grid4x4 delta is derivable before the run without circularity.  Every pair reports a paired
difference, a CI, the CI **width** and a rank-biserial, and nothing here returns a verdict string.

⚠️ THE PER-SEED REPORT IS MANDATORY
-------------------------------------
``dt_gate._per_draw_means`` averages the five training seeds into each per-draw unit, so the
companion CIs carry **no information about the seed dimension**.  P4.7's review (M1) found that
blind spot after the fact, where between-seed sd reached **57.8 ATT**.  :func:`per_seed_advantages`
is computed and emitted beside every headline comparison so the same finding cannot recur here as
an omission.  ``docs/plans/p5.1.md`` decision D9.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from offline.dataset import NormalizationStats, TrajectoryWindowDataset
from offline.dt_gate import (
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    CellStats,
    EpisodeResult,
    mean_ci95,
)
from offline.joint_windows import JointWindowIndex
from offline.roadnet_graph import AdjacencySpec

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CONTEXT_LENGTH",
    "DECLARATION_FORMAT_VERSION",
    "DECLARED_GRADIENT_STEPS",
    "DT_METHODS",
    "EXPECTED_NODES",
    "JOINT_BATCH_SIZE",
    "METHODS",
    "NodePrompt",
    "SCENARIO_ID",
    "SCENARIO_KEY",
    "TIER",
    "TIER_DIRS",
    "TrainResult",
    "assert_declaration_matches_corpus",
    "build_parser",
    "collapse_criterion",
    "declaration_artifact",
    "main",
    "per_node_prompts",
    "per_seed_advantages",
    "score_p1",
    "score_p2",
    "tier_dataset",
    "tier_dirs",
    "train_spatial_dt",
]

ARTIFACT_FORMAT_VERSION = "p5.1-spatial/1.0"
DECLARATION_FORMAT_VERSION = "p5.1-declaration/1.0"

SCENARIO_KEY = "cityflow_grid4x4"
SCENARIO_ID = "cityflow_grid4x4"
TIER = "grid4x4_mappo1000"
TIER_DIRS: tuple[str, ...] = tuple(
    f"cf_grid4x4__mappo1000__seed{seed}" for seed in TRAINING_SEEDS
)

#: The two spatial arms plus the three per-intersection comparators (coordinator ruling D1).
METHODS: tuple[str, ...] = ("dt_spatial", "dt_nomix", "bc", "bc_top10", "iql")
DT_METHODS: tuple[str, ...] = ("dt_spatial", "dt_nomix")

#: The anchors: the policy that collected the tier, and the reference for the collapse criterion.
BEHAVIOUR_METHOD = "behaviour"
COLLAPSE_REFERENCE_METHOD = "random"

DECLARED_GRADIENT_STEPS = 40_000
CONTEXT_LENGTH = 20
EXPECTED_NODES = 16

#: 72,000 joint windows x 40,000 steps x 64 = 35.6 epochs -- the SAME number of passes over the
#: corpus P4/P4.6's DT arms had on cf_hz1x1 (72,000 windows, identical arithmetic).  Chosen for
#: that equality and for no other reason; see docs/plans/p5.1.md decision D5.
JOINT_BATCH_SIZE = 64

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 1_000
GRAD_CLIP = 0.25

#: DTLight Table 1's pure-offline Grid 4x4 figure.  Recorded for context ONLY: it is their
#: scenario, their corpus and their metric, and docs/plans/p5.1.md says explicitly that it is an
#: illustration and not our threshold.
DTLIGHT_GRID4X4_REFERENCE = 446.8


@dataclass(frozen=True)
class NodePrompt:
    """One intersection's declared return prompt, derived from its own training streams."""

    ix_id: str
    target_rtg: float
    rtg_scale: float
    n_streams: int
    return_min: float
    return_max: float


@dataclass(frozen=True)
class TrainResult:
    """One trained arm-seed: the curve, the budget and where the checkpoint landed."""

    method: str
    seed: int
    gradient_steps: int
    losses: tuple[float, ...]
    checkpoint_path: str
    seconds: float


def tier_dirs(corpus_root: str | Path) -> tuple[Path, ...]:
    """The five ``cf_grid4x4__mappo1000__seed*`` directories, refusing a missing one."""
    root = Path(corpus_root)
    out: list[Path] = []
    for name in TIER_DIRS:
        directory = root / name
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(
                f"{directory} is not a collected dataset directory (no manifest.json); the tier "
                f"needs all {len(TIER_DIRS)} seed directories and a partial tier may not train"
            )
        out.append(directory)
    return tuple(out)


def tier_dataset(
    corpus_root: str | Path, context_length: int = CONTEXT_LENGTH
) -> TrajectoryWindowDataset:
    """The tier's training-split window dataset, statistics fitted on the training split only.

    ``split="train"`` is not a default being accepted: it is the mechanism that makes a held-out
    draw raise instead of quietly entering the corpus.
    """
    return TrajectoryWindowDataset(
        list(tier_dirs(corpus_root)),
        context_length=int(context_length),
        split="train",
        normalize=True,
    )


def per_node_prompts(streams: Sequence[Any]) -> dict[str, NodePrompt]:
    """P4.6's rule applied within each intersection's own stream set.

    ``target_rtg[ix] = max`` total return over that intersection's streams;
    ``rtg_scale[ix] = max|total return|`` over the same.  A node with no streams is refused
    rather than defaulted.
    """
    grouped: dict[str, list[float]] = {}
    for stream in streams:
        grouped.setdefault(str(stream.ix_id), []).append(float(stream.total_return))
    if not grouped:
        raise ValueError(
            "no streams: the per-intersection prompt rule is undefined on an empty training set"
        )
    out: dict[str, NodePrompt] = {}
    for ix_id, values in sorted(grouped.items()):
        if not values:
            raise ValueError(f"intersection {ix_id!r} has no streams; its prompt is undefined")
        out[ix_id] = NodePrompt(
            ix_id=ix_id,
            target_rtg=max(values),
            rtg_scale=max(abs(min(values)), abs(max(values))),
            n_streams=len(values),
            return_min=min(values),
            return_max=max(values),
        )
    return out


def assert_declaration_matches_corpus(
    declared: Mapping[str, Mapping[str, float]], streams: Sequence[Any]
) -> dict[str, Any]:
    """Refuse to train unless every declared prompt is what the corpus says it is.

    Mirrors ``method_tier_grid.assert_declaration_matches_corpus``, per node.  A declaration that
    disagrees with the data may not train -- ``PROJECT_PLAN`` section 7, 2026-08-14: a declared
    value on a path that never runs is untested no matter how carefully it was written.
    """
    computed = per_node_prompts(streams)
    if set(declared) != set(computed):
        missing = sorted(set(computed) - set(declared))
        unknown = sorted(set(declared) - set(computed))
        raise ValueError(
            f"the declaration covers {len(declared)} intersection(s) and the corpus carries "
            f"{len(computed)}: missing {missing[:8]}, unknown {unknown[:8]}. Every controlled "
            "intersection needs its own declared prompt and none is defaulted"
        )
    for ix_id, prompt in computed.items():
        entry = declared[ix_id]
        if float(entry["target_rtg"]) != prompt.target_rtg:
            raise ValueError(
                f"{ix_id}: the declaration declares target_rtg {float(entry['target_rtg'])} but "
                f"this intersection's training streams have maximum return {prompt.target_rtg}; "
                "the prompt is that intersection's own split maximum (docs/plans/p5.1.md D3) and "
                "a declaration that disagrees with the data may not train"
            )
        if float(entry["rtg_scale"]) != prompt.rtg_scale:
            raise ValueError(
                f"{ix_id}: the declaration declares rtg_scale {float(entry['rtg_scale'])} but "
                f"this intersection's training streams have max|return| {prompt.rtg_scale}"
            )
    return {
        "n_nodes": len(computed),
        "rule": (
            "target_rtg = max episode return in THIS INTERSECTION's training streams; "
            "rtg_scale = max|return| over the same set"
        ),
        "reduces_to": (
            "method_tier_grid.recomputed_target_and_scale on a one-intersection scenario"
        ),
        "training_streams": sum(p.n_streams for p in computed.values()),
    }


def declaration_artifact(
    corpus_root: str | Path,
    adjacency: AdjacencySpec,
    index: JointWindowIndex,
    prompts: Mapping[str, NodePrompt],
    stats: NormalizationStats,
) -> dict[str, Any]:
    """The full pre-training record: graph, node order, prompts, corpus and budget."""
    from offline.roadnet_graph import assert_reproduces_from_roads

    roads = assert_reproduces_from_roads(adjacency)
    graph = adjacency.to_json_obj()
    graph["roads_route_agrees"] = bool(roads["agrees_with_lane_route"])
    graph["roads_route_directed_edges"] = int(roads["directed_edges"])
    return {
        "format_version": DECLARATION_FORMAT_VERSION,
        "tier": TIER,
        "scenario_id": SCENARIO_ID,
        "scenario_key": SCENARIO_KEY,
        "corpus_root": str(Path(corpus_root).resolve()),
        "dataset_dirs": [str(d) for d in tier_dirs(corpus_root)],
        "graph": graph,
        "node_order": list(index.node_ids),
        "joint_windows": index.n_windows,
        "state_dim": int(index.state_dim),
        "n_actions": int(index.n_actions),
        "prompts": {
            ix_id: {
                "target_rtg": prompt.target_rtg,
                "rtg_scale": prompt.rtg_scale,
                "n_streams": prompt.n_streams,
                "return_min": prompt.return_min,
                "return_max": prompt.return_max,
            }
            for ix_id, prompt in sorted(prompts.items())
        },
        "prompt_rule": (
            "target_rtg = max episode return in THIS INTERSECTION's training streams; "
            "rtg_scale = max|return| over the same set (docs/plans/p5.1.md D3)"
        ),
        "global_target_would_be": max(p.target_rtg for p in prompts.values()),
        "nodes_below_global_target": sorted(
            ix
            for ix, prompt in prompts.items()
            if prompt.target_rtg < max(p.target_rtg for p in prompts.values())
        ),
        "methods": list(METHODS),
        "seeds": list(TRAINING_SEEDS),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "batch_size": JOINT_BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "held_out_draws": list(HELD_OUT_DRAWS),
        "statistics_split": stats.split,
        "statistics_draw_ids": list(stats.draw_ids),
    }


def train_spatial_dt(
    stacked: Mapping[str, torch.Tensor],
    index: JointWindowIndex,
    *,
    method: str,
    seed: int,
    adjacency: AdjacencySpec,
    prompts: Mapping[str, NodePrompt],
    stats: NormalizationStats,
    state_dim: int,
    n_actions: int,
    gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    provenance: Mapping[str, Any],
    log_every: int = 0,
) -> TrainResult:
    """Train one spatial arm for exactly *gradient_steps* and save the checkpoint.

    ``method`` selects the arm: ``dt_spatial`` uses the derived graph, ``dt_nomix`` the identity.
    Both consume the same tensors, the same batch order and the same seed, so the only difference
    between the two runs is the mask.

    The flat tensors stay on the host and each batch is gathered and moved: the stacked state
    tensor for this tier is 3.43 GiB (measured), the per-batch gather is ~3 MB, and at the measured
    127.5 ms/step the transfer is not the bottleneck.
    """
    import time

    from agent.SpatialDTAgent import (
        SPATIAL_CHECKPOINT_FORMAT_VERSION,
        SpatialDecisionTransformer,
        SpatialDTConfig,
    )
    from agent.DTAgent import action_loss
    from agent.utils.utils import Utils

    if method not in DT_METHODS:
        raise ValueError(
            f"{method!r} is not a spatial arm; this trainer builds {list(DT_METHODS)}. The "
            "per-intersection comparators are trained by offline.offline_baselines, which is "
            "imported rather than duplicated"
        )
    total = int(gradient_steps)
    if total < 1:
        raise ValueError(f"gradient_steps must be >= 1, got {gradient_steps}")
    if list(index.node_ids) != list(adjacency.node_ids):
        raise ValueError(
            f"the joint index node order {list(index.node_ids)[:6]} is not the adjacency's "
            f"{list(adjacency.node_ids)[:6]}; the mask would apply to the wrong columns"
        )
    missing = [ix for ix in index.node_ids if ix not in prompts]
    if missing:
        raise ValueError(f"no declared prompt for {missing[:8]}")

    spatial_mixing = method == "dt_spatial"
    mask = adjacency.attention_mask(spatial_mixing=spatial_mixing)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = SpatialDTConfig(
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        n_nodes=index.n_nodes,
        context_length=int(stacked["state"].shape[1]),
        max_ep_len=int(stacked["timestep"].max()) + 1,
        spatial_mixing=spatial_mixing,
    )
    model = SpatialDecisionTransformer(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    warmup = min(WARMUP_STEPS, max(1, total // 2))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup)
    )

    members = stacked["member_index"]
    count = int(members.shape[0])
    if count < 1:
        raise ValueError("the joint index is empty")
    mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.bool_)).to(device)
    # (1, N, 1, 1): each node's RTG input is divided by ITS OWN scale.
    scale = torch.tensor(
        [prompts[ix].rtg_scale for ix in index.node_ids], dtype=torch.float32
    ).view(1, index.n_nodes, 1, 1).to(device)

    generator = np.random.default_rng(int(seed))
    losses: list[float] = []
    model.train()
    started = time.time()
    for step in range(total):
        rows = torch.from_numpy(
            generator.integers(0, count, size=int(batch_size)).astype(np.int64)
        )
        selected = members[rows]
        action = stacked["action"][selected].to(device)
        logits = model(
            stacked["rtg"][selected].to(device) / scale,
            stacked["state"][selected].to(device),
            action,
            stacked["timestep"][selected].to(device),
            mask_tensor,
            stacked["attention_mask"][selected].to(device),
            stacked["avail_mask"][selected].to(device),
        )
        loss = action_loss(logits, action)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  {method} seed {seed} step {step + 1}/{total} "
                f"loss {np.mean(losses[-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

    destination = Path(checkpoint_path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}; nothing is created here"
        )
    torch.save(
        {
            "format_version": SPATIAL_CHECKPOINT_FORMAT_VERSION,
            "config": config.to_json_obj(),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "target_rtg": {ix: prompts[ix].target_rtg for ix in index.node_ids},
            "rtg_scale": {ix: prompts[ix].rtg_scale for ix in index.node_ids},
            "normalise": True,
            "scenario_id": SCENARIO_ID,
            "stats": stats.to_json_obj(),
            "intersection_ids": list(index.node_ids),
            "spatial_mask": np.asarray(mask, dtype=np.bool_).tolist(),
            "provenance": {
                **dict(provenance),
                "method": method,
                "tier": TIER,
                "seed": int(seed),
                "gradient_steps": int(total),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": int(warmup),
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "spatial_mixing": spatial_mixing,
                "roadnet_sha256": adjacency.roadnet_sha256,
            },
        },
        destination,
    )
    return TrainResult(
        method=method,
        seed=int(seed),
        gradient_steps=total,
        losses=tuple(losses),
        checkpoint_path=str(destination),
        seconds=float(seconds),
    )


# ----------------------------------------------------------------------
# The registered predictions, scored.  A8(a): these predate every number.
# ----------------------------------------------------------------------


def per_seed_advantages(
    left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]
) -> dict[str, Any]:
    """``mean(left) - mean(right)`` within each training seed, plus the across-seed spread.

    The draw-level CI averages the seeds away; this is the dimension it cannot see (P4.7 M1).
    """
    left_by_seed: dict[int, list[float]] = {}
    right_by_seed: dict[int, list[float]] = {}
    for record in left:
        if record.seed is not None:
            left_by_seed.setdefault(int(record.seed), []).append(record.att_horizon)
    for record in right:
        if record.seed is not None:
            right_by_seed.setdefault(int(record.seed), []).append(record.att_horizon)

    shared = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared:
        raise ValueError(
            "the two arms share no training seed, so the per-seed report is undefined; a cell "
            "without seeds cannot expose the dimension P4.7's M1 found hidden"
        )
    advantages = {
        seed: float(np.mean(left_by_seed[seed]) - np.mean(right_by_seed[seed]))
        for seed in shared
    }
    values = np.array(list(advantages.values()), dtype=np.float64)
    favouring = int((values < 0).sum())
    return {
        "by_seed": {str(seed): value for seed, value in advantages.items()},
        "n_seeds": len(shared),
        "mean": float(values.mean()),
        "between_seed_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "n_seeds_favouring_left": favouring,
        "reverses_on_n_seeds": int(len(values) - favouring)
        if favouring >= len(values) - favouring
        else favouring,
        "note": (
            "the draw-level CI averages these seeds into each unit and carries no information "
            "about this dimension (P4.7 M1); lower ATT is better, so a NEGATIVE advantage "
            "favours the left arm"
        ),
    }


def _outcome_from_interval(low: float, high: float) -> str:
    """HELD below zero, FAILED above it, NOT RESOLVED across it.  No equivalence verdict."""
    if high < 0.0:
        return "HELD"
    if low > 0.0:
        return "FAILED"
    return "NOT RESOLVED"


def _paired_block(
    left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]
) -> dict[str, Any]:
    """The shared comparison payload: difference, CI, CI width, rank-biserial, per-seed."""
    from offline.offline_baselines import paired_comparison

    comparison = paired_comparison(left, right)
    return {
        "left_arm": comparison.left_arm,
        "right_arm": comparison.right_arm,
        "n_shared_draws": comparison.n_shared_draws,
        "draw_ids": list(comparison.draw_ids),
        "mean_left": comparison.mean_left,
        "mean_right": comparison.mean_right,
        "mean_difference": comparison.mean_difference,
        "ci95_low": comparison.ci95_low,
        "ci95_high": comparison.ci95_high,
        "ci95_width": comparison.ci95_width,
        "median_difference": comparison.median_difference,
        "wins": comparison.wins,
        "losses": comparison.losses,
        "ties": comparison.ties,
        "rank_biserial": comparison.rank_biserial,
        "p_value": comparison.wilcoxon.p_value,
        "per_seed": per_seed_advantages(left, right),
    }


def score_p1(
    spatial: Sequence[EpisodeResult], control: Sequence[EpisodeResult]
) -> dict[str, Any]:
    """P1: does spatial mixing beat its own no-mixing control on the same tier?

    HELD iff the 95 % CI of the mean paired difference lies entirely below 0; FAILED iff entirely
    above; NOT RESOLVED otherwise, always with the CI width.  ``docs/plans/p5.1.md`` section 4.
    """
    block = _paired_block(spatial, control)
    block["prediction"] = "P1"
    block["statement"] = (
        "spatial mixing beats its own no-mixing control on cf_grid4x4__mappo1000"
    )
    block["rule"] = (
        "HELD iff the 95% CI of mean(ATT_spatial - ATT_nomix) lies entirely below 0; FAILED iff "
        "entirely above 0, which is a decisive result in the other direction and is never folded "
        "into 'inconclusive'; NOT RESOLVED otherwise, reported with the CI width"
    )
    block["outcome"] = _outcome_from_interval(block["ci95_low"], block["ci95_high"])
    return block


def score_p2(
    episodes_by_method: Mapping[str, Sequence[EpisodeResult]]
) -> dict[str, Any]:
    """P2: does the DT with mixing reach the offline-method field it lost on all eight tiers?

    Two conjuncts, scored and reported **separately** -- P4.7's M1 showed a conjunction whose
    parts have different robustness must never be reported as one verdict.  ``P2a`` is the rank
    over the five method arms; ``P2b`` is whether the advantage over the best other arm resolves.
    """
    missing = [method for method in METHODS if method not in episodes_by_method]
    if missing:
        raise ValueError(
            f"P2 needs every method arm to be scorable and {missing} are absent. Without the "
            "per-intersection comparators the question 'does the DT reach the offline-method "
            "field' has no referent and the prediction is not scored at all"
        )
    means = {
        method: float(np.mean([r.att_horizon for r in episodes_by_method[method]]))
        for method in METHODS
    }
    ordered = sorted(means, key=lambda method: means[method])
    rank = ordered.index("dt_spatial") + 1

    others = [method for method in METHODS if method != "dt_spatial"]
    best_other = min(others, key=lambda method: means[method])
    block = _paired_block(episodes_by_method["dt_spatial"], episodes_by_method[best_other])

    p2a = {
        "conjunct": "P2a",
        "statement": "dt_spatial has the lowest held-out mean att_horizon of the five method arms",
        "rank": rank,
        "n_arms": len(METHODS),
        "means": means,
        "order": ordered,
        "outcome": "HELD" if rank == 1 else "FAILED",
        "note": (
            "on the eight prior single-intersection tiers the DT ranked 3/4, 3/4, 4/4, 2/4, 2/4 "
            "and 2/4 three times -- never 1 (PROJECT_PLAN section 1b, R7)"
        ),
    }
    p2b = dict(block)
    p2b["conjunct"] = "P2b"
    p2b["statement"] = "dt_spatial's paired advantage over the best other arm resolves"
    p2b["best_other_arm"] = best_other
    p2b["outcome"] = _outcome_from_interval(block["ci95_low"], block["ci95_high"])

    return {
        "prediction": "P2",
        "statement": (
            "the DT with mixing reaches the offline-method field it lost on all eight "
            "single-intersection tiers"
        ),
        "rule": "P2 HELDs only if BOTH conjuncts hold; each is reported separately",
        "p2a": p2a,
        "p2b": p2b,
        "outcome": "HELD" if p2a["outcome"] == "HELD" and p2b["outcome"] == "HELD" else "FAILED",
    }


def collapse_criterion(
    arm: Sequence[EpisodeResult], reference: Sequence[EpisodeResult]
) -> dict[str, Any]:
    """P3: is this arm worse than the ``random`` behaviour arm on the SAME held-out draws?

    Declared against a freshly rolled shared-draw reference rather than the training-draw ladder
    figure, because A5 makes a cross-draw-set comparison void.
    """
    block = _paired_block(arm, reference)
    block["prediction"] = "P3"
    block["statement"] = (
        "the arm's held-out mean att_horizon is worse than the random behaviour arm's, on the "
        "same draws"
    )
    block["rule"] = (
        "COLLAPSED iff mean(ATT_arm - ATT_random) > 0 with the 95% CI excluding 0; NOT COLLAPSED "
        "iff the CI lies entirely below 0; NOT RESOLVED otherwise"
    )
    outcome = _outcome_from_interval(block["ci95_low"], block["ci95_high"])
    block["outcome"] = {
        "FAILED": "COLLAPSED",
        "HELD": "NOT COLLAPSED",
        "NOT RESOLVED": "NOT RESOLVED",
    }[outcome]
    block["dtlight_grid4x4_reference"] = DTLIGHT_GRID4X4_REFERENCE
    block["dtlight_note"] = (
        "DTLight Table 1's pure-offline Grid 4x4 figure, 446.8 +/- 128.0, on THEIR scenario, "
        "THEIR corpus and THEIR metric. It is the prior evidence that made PROJECT_PLAN section 9 "
        "rate this risk High. It is context and it is not our threshold: no branch above reads it"
    )
    return block


def node_ids_from_corpus(corpus_root: str | Path) -> tuple[str, ...]:
    """The controlled intersection order, read from the CORPUS rather than from the roadnet.

    This is the pairing key.  It comes from the data the model is trained on, and
    ``derive_adjacency`` then refuses unless it is exactly the roadnet's controllable set --
    which is what makes the graph provably about these rows (``PROJECT_PLAN`` section 7,
    2026-08-16).
    """
    from offline.trajectory_logger import load_episode

    directory = tier_dirs(corpus_root)[0]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    first = sorted(str(entry["filename"]) for entry in manifest["episodes"])[0]
    episode = load_episode(directory / first)
    ids = tuple(str(ix) for ix in episode.ix_ids)
    if len(ids) != EXPECTED_NODES:
        raise ValueError(
            f"{directory / first} carries {len(ids)} intersections, expected {EXPECTED_NODES}; "
            "grid4x4 is a 16-intersection scenario and a different count means the wrong corpus"
        )
    return ids


def adjacency_for_corpus(
    corpus_root: str | Path, node_ids: Sequence[str], sim_config: str | Path | None = None
) -> AdjacencySpec:
    """The graph of the network THIS corpus was collected on, taken from its own manifest."""
    from offline.roadnet_graph import adjacency_from_sim_config

    if sim_config is None:
        manifest = json.loads(
            (tier_dirs(corpus_root)[0] / "manifest.json").read_text(encoding="utf-8")
        )
        sim_config = manifest["run_metadata"]["env_paths"]["config"]
    return adjacency_from_sim_config(sim_config, node_ids)


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``declare``, ``train``, ``evaluate``, ``report``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.spatial_mixing",
        description="P5.1: the spatial mixing layer on cf_grid4x4__mappo1000",
    )
    parser.add_argument("--corpus-root", default="datasets_v11")
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--work-dir", default="output/p5_1")
    parser.add_argument("--checkpoint-dir", default="output/p5_1/checkpoints")
    parser.add_argument(
        "--mappo-checkpoint-dir",
        default="/home/filip/rltraffic/output/checkpoints/p2_1_mappo_nominal_500/"
        "p2_1_mappo_nominal_1000",
        help="the MAPPO@1000 checkpoints that COLLECTED this tier (the behaviour anchor)",
    )
    parser.add_argument("--scenario-key", default=SCENARIO_KEY)
    parser.add_argument("--scenario-id", default=SCENARIO_ID)
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=DECLARED_GRADIENT_STEPS)
    parser.add_argument("--batch-size", type=int, default=JOINT_BATCH_SIZE)
    parser.add_argument("--log-every", type=int, default=0)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("declare", help="write the pre-training declaration artifact")
    train = sub.add_parser("train", help="train one spatial arm across every seed")
    train.add_argument("--method", required=True, choices=list(DT_METHODS))
    sub.add_parser(
        "train-baselines", help="train BC, %BC and IQL on the same tier, across every seed"
    )
    evaluate = sub.add_parser("evaluate", help="roll one arm over the held-out pool")
    evaluate.add_argument("--method", required=True, choices=[*METHODS, COLLAPSE_REFERENCE_METHOD])
    sub.add_parser("report", help="assemble the cells, comparisons and prediction scores")
    return parser


def _declaration_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Everything ``declare`` and ``train`` both need, built once and identically."""
    from offline.joint_windows import build_joint_index, stack_joint
    from offline.offline_baselines import stream_returns

    node_ids = node_ids_from_corpus(args.corpus_root)
    adjacency = adjacency_for_corpus(args.corpus_root, node_ids, args.sim_config)
    dataset = tier_dataset(args.corpus_root, CONTEXT_LENGTH)
    index = build_joint_index(dataset, node_ids)
    streams = stream_returns(dataset)
    prompts = per_node_prompts(streams)
    return {
        "node_ids": node_ids,
        "adjacency": adjacency,
        "dataset": dataset,
        "index": index,
        "streams": streams,
        "prompts": prompts,
        "stack": lambda: stack_joint(dataset, index),
    }


def _run_declare(args: argparse.Namespace) -> int:
    from offline.dt_gate import write_json_atomic

    parts = _declaration_inputs(args)
    payload = declaration_artifact(
        args.corpus_root, parts["adjacency"], parts["index"], parts["prompts"],
        parts["dataset"].stats,
    )
    declared = {
        ix: {"target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale}
        for ix, p in parts["prompts"].items()
    }
    payload["declaration_check"] = assert_declaration_matches_corpus(declared, parts["streams"])

    out = Path(args.out_dir) / "p5_1_declaration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload, out)
    print(f"declaration written to {out}")
    print(
        f"  graph: {payload['graph']['undirected_edges']} undirected edges, "
        f"degrees {payload['graph']['degree_histogram']}, "
        f"roads route agrees: {payload['graph']['roads_route_agrees']}"
    )
    print(
        f"  prompts: {len(payload['prompts'])} nodes, global target would be "
        f"{payload['global_target_would_be']}, "
        f"{len(payload['nodes_below_global_target'])} of {EXPECTED_NODES} below it"
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from offline.dt_gate import runtime_provenance

    parts = _declaration_inputs(args)
    declared = {
        ix: {"target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale}
        for ix, p in parts["prompts"].items()
    }
    assert_declaration_matches_corpus(declared, parts["streams"])

    stacked = parts["stack"]()
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    records: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        destination = checkpoints / f"{TIER}_{args.method}_seed{seed}.pt"
        print(f"TRAIN {args.method} seed {seed} -> {destination}", flush=True)
        result = train_spatial_dt(
            stacked,
            index=parts["index"],
            method=args.method,
            seed=int(seed),
            adjacency=parts["adjacency"],
            prompts=parts["prompts"],
            stats=parts["dataset"].stats,
            state_dim=int(parts["index"].state_dim),
            n_actions=int(parts["index"].n_actions),
            gradient_steps=int(args.gradient_steps),
            batch_size=int(args.batch_size),
            device=device,
            checkpoint_path=destination,
            provenance={"runtime": runtime_provenance()},
            log_every=int(args.log_every),
        )
        records.append(
            {
                "method": result.method,
                "seed": result.seed,
                "gradient_steps": result.gradient_steps,
                "seconds": result.seconds,
                "final_loss": result.losses[-1],
                "checkpoint_path": result.checkpoint_path,
            }
        )
        print(
            f"  done in {result.seconds:.1f}s, final loss {result.losses[-1]:.5f}", flush=True
        )

    from offline.dt_gate import write_json_atomic

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "tier": TIER,
            "method": args.method,
            "declared_gradient_steps": int(args.gradient_steps),
            "batch_size": int(args.batch_size),
            "runs": records,
        },
        work / f"training_{args.method}.json",
    )
    return 0


def _run_train_baselines(args: argparse.Namespace) -> int:
    """BC, %BC and IQL on the same tier, independent per intersection as they are by construction.

    ⚠️ **That asymmetry IS the experiment and is stated rather than hidden:** these arms cannot
    use the neighbour information ``dt_spatial`` is given, by construction and not by
    configuration.  The trainers, the batching, the loss and the optimiser are
    ``offline.offline_baselines``', imported unchanged.
    """
    from offline.dt_gate import runtime_provenance, stack_dataset, write_json_atomic
    from offline.offline_baselines import (
        IQL_BATCH_TRANSITIONS,
        build_transitions,
        filter_stacked_to_streams,
        iql_reward_scale,
        stream_returns,
        top_return_streams,
        train_bc,
        train_iql,
    )

    dataset = tier_dataset(args.corpus_root, CONTEXT_LENGTH)
    group = next(iter(dataset.groups))
    streams = stream_returns(dataset)
    kept = top_return_streams(dataset)
    scale = iql_reward_scale([s.total_return for s in streams])
    stacked = stack_dataset(dataset, group)
    batches = {
        "bc": stacked,
        "bc_top10": filter_stacked_to_streams(dataset, stacked, kept),
    }
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tier": TIER,
        "dataset_dirs": [str(d) for d in tier_dirs(args.corpus_root)],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": SCENARIO_ID,
        "training_streams": len(streams),
        "independent_per_intersection": True,
    }
    print(
        f"baselines: {len(streams)} streams, top decile {len(kept)}, "
        f"iql reward scale {scale}, device {device}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    table = None
    for method in ("bc", "bc_top10", "iql"):
        for seed in TRAINING_SEEDS:
            path = checkpoints / f"{TIER}_{method}_seed{seed}.pt"
            print(f"TRAIN {method} seed {seed} -> {path}", flush=True)
            if method == "iql":
                if table is None:
                    table = build_transitions(dataset, group=group, reward_scale=scale)
                    print(f"  transitions {len(table)}", flush=True)
                record = train_iql(
                    table, state_dim=group[0], n_actions=group[1], seed=int(seed),
                    declared_gradient_steps=int(args.gradient_steps),
                    batch_size=IQL_BATCH_TRANSITIONS, device=device, checkpoint_path=path,
                    stats=dataset.stats, scenario_id=SCENARIO_ID,
                    provenance={**provenance, "runtime": runtime_provenance()},
                    log_every=int(args.log_every),
                )
            else:
                record = train_bc(
                    batches[method], state_dim=group[0], n_actions=group[1], seed=int(seed),
                    method=method, declared_gradient_steps=int(args.gradient_steps),
                    batch_size=JOINT_BATCH_SIZE, device=device, checkpoint_path=path,
                    stats=dataset.stats, scenario_id=SCENARIO_ID,
                    provenance={**provenance, "runtime": runtime_provenance()},
                    log_every=int(args.log_every),
                )
            records.append(
                {"method": method, "seed": int(seed), "checkpoint_path": str(path),
                 "gradient_steps": int(args.gradient_steps)}
            )
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "tier": TIER,
            "methods": ["bc", "bc_top10", "iql"],
            "declared_gradient_steps": int(args.gradient_steps),
            "iql_reward_scale": scale,
            "top_decile_streams": len(kept),
            "training_streams": len(streams),
            "runs": records,
        },
        work / "training_baselines.json",
    )
    return 0


def assert_declared_budget(path: str | Path, declared: int, method: str) -> dict[str, Any]:
    """Refuse a checkpoint whose recorded step count is not the declared one.

    The mechanical form of "no online model selection" (``PREREGISTRATION`` section 6), mirroring
    ``dt_gate.load_gate_checkpoint``: a checkpoint saved at a different step -- an earlier one
    that scored better, say -- cannot be evaluated by this path at all.  The check runs **before**
    the agent is constructed, so a refusal builds nothing.

    It also refuses a checkpoint whose recorded ``spatial_mixing`` disagrees with the arm being
    evaluated, because the two arms are weight-compatible by design and nothing else distinguishes
    a ``dt_spatial`` file from a ``dt_nomix`` one.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    provenance = payload.get("provenance", {})
    recorded = provenance.get("gradient_steps")
    if recorded is None:
        raise ValueError(
            f"{path}: the checkpoint records no gradient step count, so it cannot be shown to be "
            "the pre-declared one; refusing rather than reporting an unidentifiable model"
        )
    if int(recorded) != int(declared):
        raise ValueError(
            f"{path}: checkpoint was saved at {int(recorded)} gradient steps but the declared "
            f"count is {int(declared)}. PREREGISTRATION.md section 6 forbids reporting a "
            "checkpoint chosen by anything other than the declared budget"
        )
    if method in DT_METHODS:
        expected = method == "dt_spatial"
        actual = bool(payload["config"]["spatial_mixing"])
        if actual != expected:
            raise ValueError(
                f"{path}: this file records spatial_mixing={actual} but is being evaluated as "
                f"{method!r}. The two arms are weight-compatible by design, so nothing else "
                "would have caught the swap"
            )
    return {"gradient_steps": int(recorded), "method": method}


def _mappo_checkpoint_for(seed: int, root: str | Path) -> Path:
    """The grid4x4 MAPPO@1000 checkpoint of one training seed, refusing a missing one."""
    path = Path(root) / f"cf_grid4x4__mappo__seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; the behaviour anchor is MAPPO@1000, the policy that "
            "collected this tier, and a missing checkpoint may not be substituted"
        )
    return path


def _arm_factory(method: str, args: argparse.Namespace, seed: int) -> Callable[[Any], Any]:
    """The action factory for one arm-seed.  One place, so no arm is wired twice."""
    from offline.method_tier_grid import _random_factory
    from offline.offline_baselines import _baseline_factory

    checkpoints = Path(args.checkpoint_dir)
    if method in DT_METHODS:
        path = str(checkpoints / f"{TIER}_{method}_seed{seed}.pt")

        def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
            from agent.SpatialDTAgent import SpatialDTAgent

            assert_declared_budget(path, int(args.gradient_steps), method)
            agent = SpatialDTAgent.from_checkpoint(env, path, device=args.device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        return factory
    if method in ("bc", "bc_top10", "iql"):
        path = str(checkpoints / f"{TIER}_{method}_seed{seed}.pt")
        return _baseline_factory(method, path, int(args.gradient_steps), args.device)
    if method == BEHAVIOUR_METHOD:
        from offline.dt_gate import _mappo_factory

        return _mappo_factory(str(_mappo_checkpoint_for(seed, args.mappo_checkpoint_dir)), args.device)
    if method == COLLAPSE_REFERENCE_METHOD:
        return _random_factory(int(seed))
    raise ValueError(f"no action factory is declared for {method!r}")


def _run_evaluate(args: argparse.Namespace) -> int:
    from offline.dt_gate import (
        _cell,
        env_settings_from_manifest,
        evaluate_arm,
        write_json_atomic,
    )
    from offline.materialise_draws import draw_config_path

    settings = env_settings_from_manifest(tier_dirs(args.corpus_root)[0] / "manifest.json")
    draws = list(HELD_OUT_DRAWS)
    arm = f"{args.method}@{TIER}"
    produced: list[EpisodeResult] = []
    for seed in TRAINING_SEEDS:
        print(f"{arm} seed {seed} over {len(draws)} draws", flush=True)
        produced.extend(
            evaluate_arm(
                arm=arm,
                seed=int(seed),
                draw_ids=draws,
                config_for_draw=lambda d: draw_config_path(
                    args.scenario_key, d, out_root=args.draws_root
                ),
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_arm_factory(args.method, args, int(seed)),
                engine_seed=int(args.engine_seed),
            )
        )

    expected = {(int(s), int(d)) for s in TRAINING_SEEDS for d in draws}
    got = {(int(r.seed), int(r.draw_id)) for r in produced}
    if got != expected:
        raise ValueError(
            f"{arm}: {len(got)} episodes against {len(expected)} requested "
            f"(missing {len(expected - got)}, unexpected {len(got - expected)})"
        )

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "arm": arm,
            "tier": TIER,
            "method": args.method,
            "declared_gradient_steps": int(args.gradient_steps),
            "engine_seed": int(args.engine_seed),
            "cell": _cell(produced),
            "episodes": [
                {
                    "arm": e.arm, "seed": e.seed, "draw_id": e.draw_id,
                    "att_horizon": e.att_horizon,
                    "horizon_vehicle_count": e.horizon_vehicle_count,
                    "episode_reward": e.episode_reward,
                }
                for e in produced
            ],
        },
        work / f"eval_{args.method}.json",
    )
    print(f"{arm}: {len(produced)} episodes written to {work / f'eval_{args.method}.json'}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Assemble the cells, every within-tier pair, and the three registered prediction scores."""
    from offline.dt_gate import write_json_atomic
    from offline.offline_baselines import paired_comparison

    work = Path(args.work_dir)
    episodes_by_method: dict[str, list[EpisodeResult]] = {}
    cells: dict[str, Any] = {}
    for method in (*METHODS, BEHAVIOUR_METHOD, COLLAPSE_REFERENCE_METHOD):
        path = work / f"eval_{method}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes_by_method[method] = [
            EpisodeResult(
                arm=r["arm"], seed=r["seed"], draw_id=int(r["draw_id"]),
                att_horizon=float(r["att_horizon"]),
                horizon_vehicle_count=float(r["horizon_vehicle_count"]),
                episode_reward=float(r["episode_reward"]),
            )
            for r in payload["episodes"]
        ]
        cells[method] = payload["cell"]

    missing = [m for m in METHODS if m not in episodes_by_method]
    if missing:
        raise ValueError(
            f"the report is missing {missing}; every method arm must be present before any "
            "prediction is scored, because P2 has no referent without the comparators"
        )

    present = [m for m in (*METHODS, BEHAVIOUR_METHOD, COLLAPSE_REFERENCE_METHOD)
               if m in episodes_by_method]
    comparisons = []
    for i, left in enumerate(present):
        for right in present[i + 1:]:
            comparison = paired_comparison(
                episodes_by_method[left], episodes_by_method[right]
            )
            entry = {
                "left_arm": comparison.left_arm, "right_arm": comparison.right_arm,
                "n_shared_draws": comparison.n_shared_draws,
                "mean_left": comparison.mean_left, "mean_right": comparison.mean_right,
                "mean_difference": comparison.mean_difference,
                "ci95_low": comparison.ci95_low, "ci95_high": comparison.ci95_high,
                "ci95_width": comparison.ci95_width,
                "rank_biserial": comparison.rank_biserial,
                "p_value": comparison.wilcoxon.p_value,
                "per_seed": per_seed_advantages(
                    episodes_by_method[left], episodes_by_method[right]
                ),
            }
            comparisons.append(entry)

    predictions: dict[str, Any] = {
        "P1": score_p1(episodes_by_method["dt_spatial"], episodes_by_method["dt_nomix"]),
        "P2": score_p2({m: episodes_by_method[m] for m in METHODS}),
    }
    if COLLAPSE_REFERENCE_METHOD in episodes_by_method:
        predictions["P3"] = {
            arm: collapse_criterion(
                episodes_by_method[arm], episodes_by_method[COLLAPSE_REFERENCE_METHOD]
            )
            for arm in DT_METHODS
        }

    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "tier": TIER,
        "scenario_id": SCENARIO_ID,
        "cells": cells,
        "comparisons": comparisons,
        "predictions": predictions,
        "no_equivalence_verdicts": (
            "BRIEF_17 section 4, inherited: A6's delta is mappo1000-on-cf_hz1x1-specific and no "
            "grid4x4 delta is derivable before the run without circularity. Every pair reports a "
            "paired difference, a CI, the CI width and a rank-biserial, and no verdict string"
        ),
        "asymmetry": (
            "bc, bc_top10 and iql are independent per intersection BY CONSTRUCTION and cannot "
            "use the neighbour information dt_spatial is given. That asymmetry is the experiment "
            "and is stated here rather than hidden"
        ),
    }
    out = Path(args.out_dir) / "p5_1_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload, out)

    print(f"report written to {out}")
    for method in present:
        print(f"  {method:12s} {cells[method]['att_horizon_mean']:9.4f}")
    print(f"  P1  {predictions['P1']['outcome']}")
    print(f"  P2a {predictions['P2']['p2a']['outcome']}  P2b {predictions['P2']['p2b']['outcome']}")
    if "P3" in predictions:
        for arm, block in predictions["P3"].items():
            print(f"  P3  {arm}: {block['outcome']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m offline.spatial_mixing``."""
    args = build_parser().parse_args(argv)
    if int(args.torch_threads) > 0:
        # PROJECT_PLAN section 9: the pin is a LIVENESS fix, not a performance one.
        torch.set_num_threads(int(args.torch_threads))
    if args.command == "declare":
        return _run_declare(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "train-baselines":
        return _run_train_baselines(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    return _run_report(args)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
