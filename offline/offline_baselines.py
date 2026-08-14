"""P4.4: train BC, %BC and IQL on the P4 corpus and compare them against the DT, paired.

Artifact format version: ``p4.4-baselines/1.0``.

WHAT THIS TASK DECIDES
----------------------
P4's Decision Transformer beat the policy whose data it trained on by **0.6263 ATT**.  Until a
behaviour-cloning model is measured on the same corpus, the same draws and the same protocol,
that margin cannot be attributed to anything.  ``PROJECT_PLAN.md`` section 1 has said since
2026-07-10: *"if BC matches MADT, sequence modeling adds nothing -- must be tested."*

THE PROTOCOL IS P4'S, REUSED RATHER THAN RESTATED
-------------------------------------------------
Every rollout goes through ``offline.dt_gate.evaluate_arm`` with env settings read from the
collection manifest by ``env_settings_from_manifest``; the cells are built by ``dt_gate._cell``,
the pairing by ``dt_gate._paired``, the descriptives by ``mean_ci95`` and the paired test by
``wilcoxon_signed_rank``.  Importing them -- including the two private helpers -- is deliberate:
a second implementation of the same protocol is exactly how two arms stop being comparable, and
these are the functions that produced the numbers this task compares against.

**Gate A** re-rolls all 1100 cited P4 episodes (MADT, MAPPO@1000, MaxPressure) on this
instrument and refuses to continue unless every one reproduces exactly.  A difference would mean
P4.4's numbers were measured on a different instrument than P4's, and no comparison between them
would mean anything.

THE DECISION RULE (``PREREGISTRATION.md`` A6, declared before any baseline existed)
-----------------------------------------------------------------------------------
``delta = 0.6263 ATT`` -- the DT's own paired margin over MAPPO@1000 on the 100 held-out draws.
With ``d = ATT_DT - ATT_baseline`` per shared draw and lower ATT better:

* CI(d) entirely within ``[-delta, +delta]``          -> the baseline **matches** the DT;
* CI(d) entirely below ``-delta``                     -> the DT is **genuinely better**;
* CI(d) entirely above ``+delta``                     -> the **baseline** is genuinely better --
  arithmetically possible, **not named by A6**, and reported separately rather than folded into
  "inconclusive", because it is a decisive result in the other direction;
* anything else                                       -> **inconclusive at this power**, with
  the CI width, because "contains 0" is a failure to reject and not a demonstration of
  equivalence.

**The recovered fraction ``(MAPPO@1000 - arm) / (MAPPO@1000 - DT)`` is reported unconditionally
beside every verdict, in every branch** (A6's clarification of 2026-08-11): delta equals the
effect under study, so a baseline landing 0.5 ATT worse than the DT sits inside the margin and
returns *matches* while having recovered only 20.2 % of the DT's margin over its own behaviour
policy.

WHAT A DIFFERENCE AGAINST THESE BASELINES DOES NOT SAY
-------------------------------------------------------
DT minus BC is a **combined** difference -- attention/context **plus** return-to-go conditioning
**plus** the timestep embedding.  It is never reported as "sequence modelling adds X"; P4.3 (RTG)
and P5.3 (no-RTG, context length) own the decomposition.  And IQL here is **untuned**, running
published D4RL-locomotion values transplanted onto a discrete 8-phase action space: a losing
untuned IQL cannot support any claim of MADT superiority over IQL, which is stated in the same
sentence as its number rather than in a caveats section.

TRUNCATION, AND WHY THERE IS NO ``done`` IN THIS FILE
------------------------------------------------------
``terminated`` is hardcoded ``False`` on this platform (``envs/base_traffic_env.py``) and every
episode ends by time-limit truncation, so the value learner **bootstraps through the boundary**:
the last transition's target is ``r + gamma * V(s_T)``, never ``r``.  Treating a timeout as
terminal causes systematic value underestimation near episode end and would hand the DT an
unearned win over its own baselines (``PREREGISTRATION.md`` section 7; Decisions Log 2026-07-26).

That is also why this module reads episodes through
``offline.trajectory_logger.load_episode``: ``TrajectoryWindowDataset`` structurally never yields
observation row ``T`` -- "observation row T is never an input" -- and row ``T`` is precisely the
bootstrap target of the final transition.  ``load_episode`` is the format's **single sanctioned
reader**, the one ``offline/dataset.py`` itself calls, so this is not a second reader of the
on-disk format; episode selection, the draw-split leakage guard and the normalisation statistics
all stay in ``TrajectoryWindowDataset``, and a test cross-checks every ``(s_t, a_t)`` against it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from agent.OfflineBaselines import (
    BCAgent,
    IQLAgent,
    MLPTrunk,
    TrunkConfig,
    action_loss,
    canonical_state_dict_digest,
    masked_action_logits,
)
from offline.dataset import NormalizationStats, TrajectoryWindowDataset
from offline.dt_gate import (
    GRAD_CLIP,
    HELD_OUT_DRAWS,
    LEARNING_RATE,
    PLATEAU_TOLERANCE,
    PLATEAU_WINDOW,
    TRAINING_SEEDS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EpisodeResult,
    WilcoxonResult,
    _cell,
    _paired,
    _sha256_file,
    build_training_dataset,
    env_settings_from_manifest,
    evaluate_arm,
    mean_ci95,
    runtime_provenance,
    stack_dataset,
    wilcoxon_signed_rank,
    window_means,
    plateau_reached,
    write_json_atomic,
)
from offline.trajectory_logger import load_episode

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_GRADIENT_STEPS",
    "DELTA_ATT",
    "DELTA_ATT_DERIVATION",
    "IQL_BATCH_TRANSITIONS",
    "METHODS",
    "PairedComparison",
    "StreamReturn",
    "TrainRecord",
    "TransitionTable",
    "assert_campaign_complete",
    "baselines_artifact",
    "build_transitions",
    "equivalence_verdict",
    "filter_stacked_to_streams",
    "iql_reward_scale",
    "iql_targets",
    "main",
    "merge_training_runs",
    "paired_comparison",
    "pin_torch_threads",
    "rank_biserial",
    "recovered_fraction",
    "stream_returns",
    "top_return_composition",
    "top_return_streams",
    "train_bc",
    "train_iql",
    # -- P4.5: which streams an arm sees, and nothing else -------------------------------
    "ArmSpec",
    "MATCHED_SUBSET_COUNT",
    "SELECTION_ARMS",
    "SELECTION_ARTIFACT_FORMAT_VERSION",
    "SELECTION_BASELINES_FORMAT_VERSION",
    "arm_spec_for_flags",
    "assert_reused_arm_reproduces",
    "assert_selection_design",
    "delta_verdict",
    "random_stream_subset",
    "select_arm_streams",
    "selection_artifact",
    "streams_from_datasets",
    "thread_regime",
]

ARTIFACT_FORMAT_VERSION = "p4.4-baselines/1.0"

#: Declared in ``docs/plans/p4.4.md`` section 3.1 before the first gradient step: the DT's own
#: REPORTED budget, which is what section 6.3's "same tuning budget" matches.  The DT has spent
#: its one pre-declared raise, so no raise is available to this task.
DECLARED_GRADIENT_STEPS = 40_000

#: ``PREREGISTRATION.md`` amendment A6's declared literal, and the full-precision quantity it was
#: rounded from (105.58203462874322 - 104.95575898180847, from the committed ``p4_gate.json``).
#: Both are recorded because the multiplier of 1.0 on the DT's margin is a CHOICE, as A6's own
#: clarification says, and a verdict must never turn on the rounding: the artifact recomputes
#: every verdict under both values and refuses to report if they disagree.
DELTA_ATT = 0.6263
DELTA_ATT_DERIVATION = 0.6262756469347437

#: How close a CI endpoint may come to +/- delta before the verdict is refused.  The plan's
#: section 5.3 promised this as an ASSERTION and the first implementation only recorded the
#: distance (review finding F7); observed distances are 0.157 / 0.916 / 0.593, so nothing
#: reported is near it, which is exactly when a guard should be installed rather than argued about.
DELTA_PROXIMITY_TOLERANCE = 1e-3

BC_BATCH_WINDOWS = 64
IQL_BATCH_TRANSITIONS = 1_280
TOP_RETURN_FRACTION = 0.10

#: Published IQL, D4RL locomotion.  Unswept -- see ``docs/plans/p4.4.md`` section 3.5 for the
#: decision and its reason (the authorised selection criterion cannot rank either parameter).
IQL_TAU = 0.7
IQL_BETA = 3.0
IQL_GAMMA = 0.99
IQL_POLYAK = 0.005
IQL_WEIGHT_CLIP = 100.0

METHODS: tuple[str, ...] = ("bc", "bc_top10", "iql")

#: The arms cited from P4 and re-rolled by Gate A before anything else is measured.
CITED_ARMS: tuple[str, ...] = ("madt", "mappo1000", "maxpressure")

VERDICT_MATCHES = "matches"
VERDICT_DT_BETTER = "dt_genuinely_better"
VERDICT_BASELINE_BETTER = "baseline_genuinely_better"
VERDICT_INCONCLUSIVE = "inconclusive_at_this_power"

#: How many rows the post-training diagnostics pass reads.  Deterministic and capped, so the
#: diagnostic costs the same on the fixture and on the 72,000-window tier.
DIAGNOSTIC_ROWS = 20_000


@dataclass(frozen=True)
class StreamReturn:
    """One (episode, intersection) stream and its undiscounted return.

    ``total_return`` is the stream's own ``rtg[t=0]``, i.e. ``sum(local_reward)`` -- the same
    quantity the ladder calls an episode return, per intersection.
    """

    dataset_dir: str
    episode_file: str
    ix_id: str
    ix_index: int
    episode_index: int
    flow_draw: int
    group: tuple[int, int]
    total_return: float

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity of the stream: directory, episode file, intersection id."""
        return (self.dataset_dir, self.episode_file, self.ix_id)


@dataclass(frozen=True)
class TransitionTable:
    """Flat ``(s, a, r, s')`` transitions.

    **There is no ``done`` field, and that is the point.**  Every episode on this platform ends
    by time-limit truncation, so every transition bootstraps -- including the last one of each
    stream, whose ``next_state`` is observation row ``T``.

    ``state`` and ``next_state`` are normalised with the frozen training-split statistics, the
    same ones the loader applied to its windows.
    """

    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    stream_index: torch.Tensor
    t: torch.Tensor
    reward_scale: float

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def select(self, index: torch.Tensor) -> TransitionTable:
        """A sub-table of the given rows, so a batch is still a table.

        The training loop takes its targets through :func:`iql_targets` on one of these rather
        than inlining ``r + gamma * V(s')``.  That call boundary is what makes the bootstrap
        observable; the test that pins it spies on the boundary and asserts on the arguments the
        loop passes, because the discount and the bootstrap value are chosen at the call site and
        not inside the helper (corrected 2026-08-12, review finding F1).
        """
        return TransitionTable(
            state=self.state[index],
            next_state=self.next_state[index],
            action=self.action[index],
            reward=self.reward[index],
            stream_index=self.stream_index[index],
            t=self.t[index],
            reward_scale=self.reward_scale,
        )

    def to(self, device: torch.device) -> TransitionTable:
        """The same table on *device*."""
        return TransitionTable(
            state=self.state.to(device),
            next_state=self.next_state.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            stream_index=self.stream_index.to(device),
            t=self.t.to(device),
            reward_scale=self.reward_scale,
        )


@dataclass(frozen=True)
class TrainRecord:
    """One trained (method, seed): the reported checkpoint and its provenance.

    ``canonical_digest`` is the identity of the weights; ``file_sha256`` is the identity of the
    file.  They answer different questions and both are kept (``DEFERRED`` 29).
    """

    method: str
    seed: int
    gradient_steps: int
    declared_gradient_steps: int
    losses: tuple[float, ...]
    window_means: tuple[float, ...]
    plateaued: bool
    checkpoint_path: str
    canonical_digest: str
    file_sha256: str
    seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedComparison:
    """A paired per-draw comparison with the effect sizes section 8 makes mandatory.

    ``mean_difference`` is ``mean(left - right)`` over the shared draws.  ``wins`` counts draws
    where the **left** arm had the lower (better) ATT.
    """

    left_arm: str
    right_arm: str
    n_shared_draws: int
    draw_ids: tuple[int, ...]
    mean_left: float
    mean_right: float
    mean_difference: float
    ci95_half_width: float
    ci95_width: float
    ci95_low: float
    ci95_high: float
    median_difference: float
    wins: int
    losses: int
    ties: int
    rank_biserial: float
    wilcoxon: WilcoxonResult

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-ready mapping, including the Wilcoxon result as a nested object."""
        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "n_shared_draws": self.n_shared_draws,
            "draw_ids": list(self.draw_ids),
            "mean_left": self.mean_left,
            "mean_right": self.mean_right,
            "mean_difference": self.mean_difference,
            "ci95_half_width": self.ci95_half_width,
            "ci95_width": self.ci95_width,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "median_difference": self.median_difference,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "rank_biserial": self.rank_biserial,
            "wilcoxon": {
                "w_plus": self.wilcoxon.w_plus,
                "w_minus": self.wilcoxon.w_minus,
                "statistic": self.wilcoxon.statistic,
                "n_used": self.wilcoxon.n_used,
                "n_zero": self.wilcoxon.n_zero,
                "z": self.wilcoxon.z,
                "p_value": self.wilcoxon.p_value,
            },
        }


# ----------------------------------------------------------------------
# The data path: stream returns, the %BC filter, and IQL's transitions
# ----------------------------------------------------------------------


def stream_returns(dataset: TrajectoryWindowDataset) -> tuple[StreamReturn, ...]:
    """Undiscounted return of every (episode, intersection) stream, from the loader alone.

    Read as the item at ``t = 0``, whose returns-to-go entry is ``sum(r_k for k >= 0)``.  The
    loader is the single definition of that quantity; a test recomputes it from the raw reward
    arrays by a second route.
    """
    out: list[StreamReturn] = []
    for index in range(len(dataset)):
        meta = dataset.item_meta(index)
        if meta.t != 0:
            continue
        item = dataset[index]
        out.append(
            StreamReturn(
                dataset_dir=meta.dataset_dir,
                episode_file=meta.episode_file,
                ix_id=meta.ix_id,
                ix_index=meta.ix_index,
                episode_index=meta.episode_index,
                flow_draw=meta.flow_draw,
                group=(int(item["state"].shape[-1]), int(item["avail_mask"].shape[-1])),
                total_return=float(item["rtg"][-1, 0]),
            )
        )
    return tuple(out)


def top_return_streams(
    dataset: TrajectoryWindowDataset, fraction: float = TOP_RETURN_FRACTION
) -> tuple[StreamReturn, ...]:
    """The top *fraction* of streams by return -- %BC's filter, declared and deterministic.

    ``ceil(fraction * n)`` streams, never fewer than one, ordered by descending return with ties
    broken by ``(dataset_dir, episode_file, ix_id)`` so the selection does not depend on load
    order.  On the P4 tier the cut is 20 of 200 streams at -6419.0, with the next stream at
    -6449.0: measured, and not a tie.
    """
    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")
    streams = stream_returns(dataset)
    if not streams:
        raise ValueError("this dataset has no streams to filter")
    keep = max(1, math.ceil(value * len(streams)))
    ordered = sorted(
        streams, key=lambda s: (-s.total_return, s.dataset_dir, s.episode_file, s.ix_id)
    )
    return tuple(ordered[:keep])


def _behaviour_seed(stream: StreamReturn) -> int:
    """The behaviour-policy seed a stream was collected under, by two routes that must agree.

    The directory name carries it (``cf_hz1x1__mappo1000__seed101``) and so does the manifest's
    ``run_metadata.checkpoint`` (``.../cf_hz1x1__mappo__seed101.pt``), which names the actual
    weights.  Both are read and compared, because the whole of F3's finding is an attribution to
    a checkpoint and a directory name alone is a label rather than evidence.  When a corpus
    carries no checkpoint field -- test fixtures do not -- the directory name stands alone and
    the cross-check is skipped rather than faked.
    """
    directory = Path(stream.dataset_dir)
    match = re.search(r"seed(\d+)$", directory.name)
    if match is None:
        raise ValueError(
            f"{directory.name}: no seed suffix, so this stream cannot be attributed to a "
            "behaviour policy; F3's composition is an attribution and refuses to guess"
        )
    seed = int(match.group(1))

    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        checkpoint = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "run_metadata", {}
        ).get("checkpoint")
        if checkpoint:
            recorded = re.search(r"seed(\d+)\.pt$", str(checkpoint))
            if recorded is None or int(recorded.group(1)) != seed:
                raise ValueError(
                    f"{directory.name}: the directory says seed {seed} but its manifest "
                    f"records checkpoint {checkpoint!r}; the block identity is not trustworthy "
                    "and the composition must not be reported from it"
                )
    return seed


def _exact_max_block_p_value(sizes: Sequence[int], draws: int, observed: int) -> float:
    """``P(max block count >= observed)`` under the multivariate hypergeometric null.

    The null, stated because a p-value without one is not interpretable: *drawing ``draws`` of
    ``sum(sizes)`` streams uniformly without replacement, from blocks of the given sizes.*  This
    enumerates every composition exactly, so no RNG seed enters a reported quantity and the
    number cannot be re-rolled.  :func:`_permutation_max_block_p_value` is the cross-check on the
    arithmetic -- **not** on the null, which both share.
    """
    blocks = [int(s) for s in sizes]
    if min(blocks, default=0) < 0 or draws < 0 or draws > sum(blocks):
        raise ValueError(f"cannot draw {draws} from blocks {blocks}")
    total = math.comb(sum(blocks), draws)
    accumulated = 0

    def walk(index: int, remaining: int, ways: int, hit: bool) -> None:
        nonlocal accumulated
        if index == len(blocks) - 1:
            if 0 <= remaining <= blocks[index] and (hit or remaining >= observed):
                accumulated += ways * math.comb(blocks[index], remaining)
            return
        for taken in range(0, min(blocks[index], remaining) + 1):
            walk(index + 1, remaining - taken, ways * math.comb(blocks[index], taken),
                 hit or taken >= observed)

    walk(0, draws, 1, False)
    return accumulated / total


def _permutation_max_block_p_value(
    labels: Sequence[int], draws: int, observed: int, *, iterations: int, rng_seed: int
) -> dict[str, Any]:
    """Monte Carlo under the same null -- a cross-check on the enumeration's arithmetic."""
    generator = np.random.default_rng(int(rng_seed))
    population = np.asarray(list(labels))
    hits = 0
    for _ in range(int(iterations)):
        picked = generator.choice(population.size, size=int(draws), replace=False)
        counts = np.bincount(population[picked])
        if int(counts.max()) >= observed:
            hits += 1
    estimate = hits / float(iterations)
    return {
        "p_value": estimate,
        "iterations": int(iterations),
        "rng_seed": int(rng_seed),
        "monte_carlo_standard_error": math.sqrt(
            max(estimate * (1.0 - estimate), 0.0) / float(iterations)
        ),
    }


def top_return_composition(
    dataset: TrajectoryWindowDataset,
    kept: Sequence[StreamReturn],
    heldout_att_by_seed: Mapping[int, float],
    *,
    permutations: int = 20_000,
    rng_seed: int = 20_260_812,
) -> dict[str, Any]:
    """What the top-return filter actually selected, as a RESULT rather than as provenance.

    The P4.4 packet's section 8.5 disclosed that %BC's margin *"may partly be memorisation of an
    easier subset -- I did not test that"*.  This tests it, and the answer is neither option that
    sentence offered: the filter is not selecting easier demand, and it is not mainly selecting
    better episodes within a policy -- **it is selecting better CHECKPOINTS.**

    The evidence is a deduction, and the p-value is its visible consequence rather than its
    support:

    1. per-seed training return ranks the five behaviour checkpoints, and per-seed **held-out**
       ATT ranks them almost identically (Pearson ``r``, reported with its ``n``), on **disjoint
       draw sets**, so the correlation cannot come from shared draw difficulty;
    2. the filter ranks streams by training return **deterministically**, so given (1) it selects
       the strongest checkpoints -- a consequence of the selection rule, not a coincidence;
    3. the observed concentration is what (2) looks like from outside, and D16 (seed identical to
       demand block) is what makes "seed" and "block" the same thing on this corpus.

    ⚠️ **The p-value tests CONCENTRATION -- that some block holds at least ``observed`` of the
    kept streams -- and NOT that the concentrated blocks are the best-performing ones.**  The
    identity claim rests on (1) and (2).
    """
    streams = stream_returns(dataset)
    labels = [_behaviour_seed(stream) for stream in streams]
    kept_labels = [_behaviour_seed(stream) for stream in kept]
    block_ids = sorted(set(labels))
    sizes = [labels.count(block) for block in block_ids]
    counts = {block: kept_labels.count(block) for block in block_ids}
    observed = max(counts.values())

    index = {block: position for position, block in enumerate(block_ids)}
    permutation = _permutation_max_block_p_value(
        [index[label] for label in labels],
        len(kept),
        observed,
        iterations=permutations,
        rng_seed=rng_seed,
    )
    exact = _exact_max_block_p_value(sizes, len(kept), observed)

    training_return = {
        block: float(np.mean([s.total_return for s, label in zip(streams, labels) if label == block]))
        for block in block_ids
    }
    shared = [block for block in block_ids if block in heldout_att_by_seed]
    if len(shared) < 3:
        raise ValueError(
            f"only {len(shared)} block(s) have a held-out ATT, which is too few to correlate; "
            "the composition is a claim about checkpoint quality and needs both measurements"
        )
    x = np.asarray([training_return[block] for block in shared], dtype=np.float64)
    y = np.asarray([float(heldout_att_by_seed[block]) for block in shared], dtype=np.float64)
    correlation = float(np.corrcoef(x, y)[0, 1])

    return {
        "role": (
            "what the top-return filter selected: a result about %BC's mechanism, not provenance"
        ),
        "per_seed_kept_counts": {str(block): counts[block] for block in block_ids},
        "per_seed_stream_counts": {str(block): size for block, size in zip(block_ids, sizes)},
        "observed_max_block_count": int(observed),
        "exact_p_value": exact,
        "exact_p_value_null": (
            "multivariate hypergeometric: drawing "
            f"{len(kept)} of {len(streams)} streams uniformly WITHOUT replacement from "
            f"{len(block_ids)} blocks of {sizes[0] if len(set(sizes)) == 1 else sizes}, "
            f"P(max block count >= {observed})"
        ),
        "exact_p_value_tests": (
            "CONCENTRATION only -- that some block holds at least the observed count. It does "
            "NOT test that the concentrated blocks are the best-performing ones; that rests on "
            "the correlation below and on the filter being a deterministic rank by return"
        ),
        "permutation_cross_check": permutation,
        "permutation_cross_check_role": (
            "a second route to the same number under the SAME null, so it checks the "
            "enumeration's arithmetic and leaves the null itself untested"
        ),
        "per_seed_training_return_mean": {
            str(block): training_return[block] for block in block_ids
        },
        "per_seed_heldout_att": {
            str(block): float(heldout_att_by_seed[block]) for block in shared
        },
        "pearson_r_training_return_vs_heldout_att": correlation,
        "pearson_n": len(shared),
        "pearson_draw_sets": (
            "DISJOINT: training returns over draws 1-200, held-out ATT over draws 1000-1099, so "
            "the correlation cannot be produced by shared draw difficulty"
        ),
        "licenses": (
            "on this corpus the top-10% return filter performed CHECKPOINT SELECTION, so %BC's "
            "advantage over BC is at least partly an effect of training on the strongest "
            "behaviour policies rather than of filtering episode quality within a policy"
        ),
        "does_not_license": (
            "that return filtering does checkpoint selection in general: n = "
            f"{len(shared)} seeds, one tier, one scenario, one backend. A correlation this "
            "strong over this few points is a strong hint and a weak law"
        ),
        "decisive_test_not_run_here": (
            "BC trained on the two strongest behaviour seeds only, against %BC; registered as "
            "P4.5 because it needs a training run, which BRIEF_12 section 5 forbids in this round"
        ),
    }


def filter_stacked_to_streams(
    dataset: TrajectoryWindowDataset,
    stacked: dict[str, torch.Tensor],
    streams: Sequence[StreamReturn],
) -> dict[str, torch.Tensor]:
    """Restrict stacked windows to those belonging to *streams*, keeping ``item_index``.

    Provenance survives the filter: row ``r`` of the result is still
    ``dataset[item_index[r]]``, so a filtered row can be traced back to its episode file.
    """
    if "item_index" not in stacked:
        raise ValueError(
            "the stacked mapping carries no item_index, so its rows cannot be traced back to "
            "streams; build it with offline.dt_gate.stack_dataset"
        )
    wanted = {stream.key for stream in streams}
    if not wanted:
        raise ValueError("no streams were given: a filter that keeps nothing is not a filter")
    keep = torch.tensor(
        [
            (
                dataset.item_meta(int(index)).dataset_dir,
                dataset.item_meta(int(index)).episode_file,
                dataset.item_meta(int(index)).ix_id,
            )
            in wanted
            for index in stacked["item_index"]
        ],
        dtype=torch.bool,
    )
    if not bool(keep.any()):
        raise ValueError(
            "the filter kept no windows: the requested streams are not present in this stack, "
            "which usually means the streams and the stack belong to different groups"
        )
    return {name: value[keep] for name, value in stacked.items()}


# ----------------------------------------------------------------------
# P4.5: which behaviour checkpoints an arm's streams came from, and nothing else
# ----------------------------------------------------------------------

SELECTION_ARTIFACT_FORMAT_VERSION = "p4.5-selection/1.0"
SELECTION_BASELINES_FORMAT_VERSION = "p4.5-selection-baselines/1.0"

#: The size every matched arm is held to.  ``docs/plans/p4.5.md`` section 3: %BC trains on 20
#: streams, so an arm that answers "is it the SEEDS?" must train on 20 too, or it answers "is it
#: the amount of data?" instead.
MATCHED_SUBSET_COUNT = 20

VERDICT_WITHIN_DELTA = "within_delta"
VERDICT_LEFT_BETTER = "left_genuinely_better"
VERDICT_RIGHT_BETTER = "right_genuinely_better"
VERDICT_PAIR_INCONCLUSIVE = "inconclusive_at_this_power"


#: Per-seed held-out ATT of the behaviour policy, COPIED from the committed
#: ``docs/data/p4_4_training.json`` composition block rather than recomputed here, so this task
#: cannot produce a second version of a merged number.  A test asserts the equality field by
#: field, and a second test recomputes the RANKING from the 500 raw episode records.
BEHAVIOUR_HELDOUT_ATT: Mapping[int, float] = {
    101: 103.60869401265231,
    202: 103.52858962932616,
    303: 107.79803977256651,
    404: 105.99759066310882,
    505: 106.9772590660622,
}

#: The two best and the two worst behaviour checkpoints on the held-out pool, DERIVED from the
#: measurement above rather than typed.  ``BRIEF_13`` section 10.7 warns that the worst two are
#: 505 and 303 and not 303+404 -- a mistake that is invisible in a shell command and fatal to the
#: ordering prediction, which is exactly why this is a computation and not a literal.
_BEHAVIOUR_RANKING: tuple[int, ...] = tuple(
    sorted(BEHAVIOUR_HELDOUT_ATT, key=lambda seed: BEHAVIOUR_HELDOUT_ATT[seed])
)
BEST_TWO_BEHAVIOUR_SEEDS: tuple[int, ...] = tuple(sorted(_BEHAVIOUR_RANKING[:2]))
WORST_TWO_BEHAVIOUR_SEEDS: tuple[int, ...] = tuple(sorted(_BEHAVIOUR_RANKING[-2:]))

#: The five-seed mixture, and the two gaps the secondary prediction is measured against.  All
#: three are COMPUTED from the per-seed measurement rather than typed: the first draft of this
#: file carried a hand-copied 2.0534450018170614 for the second gap and the true value is
#: 2.053444999417053, which is the same class of defect as the brief's own 2.05.
#: ``_BEHAVIOUR_MIXTURE_ATT`` equals the committed 500-episode ``mappo1000`` cell mean exactly
#: under ``==``, which a test asserts across the two artifacts.
_BEHAVIOUR_MIXTURE_ATT = sum(BEHAVIOUR_HELDOUT_ATT.values()) / float(len(BEHAVIOUR_HELDOUT_ATT))
BEHAVIOUR_GAP_TO_BEST_TWO = _BEHAVIOUR_MIXTURE_ATT - sum(
    BEHAVIOUR_HELDOUT_ATT[seed] for seed in BEST_TWO_BEHAVIOUR_SEEDS
) / 2.0
BEHAVIOUR_GAP_TO_BEST_SINGLE = _BEHAVIOUR_MIXTURE_ATT - min(BEHAVIOUR_HELDOUT_ATT.values())


@dataclass(frozen=True)
class ArmSpec:
    """One arm's declared stream selection: the selector, its pool and its size.

    A declaration, not a configuration.  The CLI must agree with it or refuse to run, because a
    shell typo that redefined an arm would be invisible in every artifact this task writes.

    ``behaviour_seeds`` empty means "every seed in the corpus"; ``count`` ``None`` means "the
    whole pool".
    """

    arm: str
    selector: str
    behaviour_seeds: tuple[int, ...]
    count: int | None
    role: str


#: The four new arms of P4.5, declared before the first gradient step
#: (``docs/plans/p4.5.md`` section 3).  ``bc_top10`` is not here: it was trained by P4.4 and its
#: 500 episodes are re-used rather than re-rolled.
SELECTION_ARMS: Mapping[str, ArmSpec] = {
    "bc_best2_20": ArmSpec(
        arm="bc_best2_20",
        selector="random_subset",
        behaviour_seeds=BEST_TWO_BEHAVIOUR_SEEDS,
        count=MATCHED_SUBSET_COUNT,
        role="seed identity at MATCHED SIZE: the decisive arm against bc_any_20",
    ),
    "bc_any_20": ArmSpec(
        arm="bc_any_20",
        selector="random_subset",
        behaviour_seeds=(),
        count=MATCHED_SUBSET_COUNT,
        role="size alone, seeds unmatched: a mixture of all five behaviour modes",
    ),
    "bc_worst2_20": ArmSpec(
        arm="bc_worst2_20",
        selector="random_subset",
        behaviour_seeds=WORST_TWO_BEHAVIOUR_SEEDS,
        count=MATCHED_SUBSET_COUNT,
        role="the low end of the same axis: two arms give a difference, three give an ordering",
    ),
    "bc_best2_all": ArmSpec(
        arm="bc_best2_all",
        selector="datasets",
        behaviour_seeds=BEST_TWO_BEHAVIOUR_SEEDS,
        count=None,
        role="data quantity from good seeds; SECONDARY, because it is size-matched to nothing",
    ),
}

#: The reported order, which fixes the orientation of every pair: for ``i < j`` the difference is
#: ``mean(arms[i] - arms[j])`` over the shared draws.  Declared, so no pair can be reported in
#: whichever direction reads better.
SELECTION_ARM_ORDER: tuple[str, ...] = (
    "bc_top10",
    "bc_best2_20",
    "bc_any_20",
    "bc_worst2_20",
    "bc_best2_all",
)

#: The arm whose episodes are re-used from the merged P4.4 artifact instead of re-rolled.
REUSED_ARM = "bc_top10"

#: Corpus facts of the P4 tier, declared here so a validator has a reference that does NOT come
#: from the payload it validates (review finding N1).  Both are asserted against the real corpus
#: by ``test_the_real_tier_filters_and_builds_transitions_consistently``'s neighbours: 200 streams
#: over 5 behaviour seeds is 40 each, and 72,000 windows over 200 streams is 360 rows each.
DECISION_ROWS_PER_STREAM = 360
STREAMS_PER_BEHAVIOUR_SEED = 40

#: The contrast the whole task turns on, and the one whose CI bounds a null (plan section 2.2).
DECISIVE_CONTRAST = ("bc_best2_20", "bc_any_20")
POWER_CONTRAST = ("bc_best2_20", "bc_worst2_20")


def _stream_key(stream: StreamReturn) -> tuple[str, str, str]:
    """The canonical identity of a stream, and the order every selector returns."""
    return (stream.dataset_dir, stream.episode_file, stream.ix_id)


def streams_from_datasets(
    dataset: TrajectoryWindowDataset, dataset_dirs: Sequence[str | Path]
) -> tuple[StreamReturn, ...]:
    """Every stream collected under one of *dataset_dirs*, in canonical order.

    The membership test is on the stream's OWN ``dataset_dir``, so an arm's pool is a property of
    the data rather than of the call that asked for it.  Paths are compared as ``str(Path(x))``
    on both sides -- separators normalised, symlinks deliberately NOT resolved, because a
    resolved match between two different spellings of a directory would be a silent success where
    a refusal is wanted.

    A requested directory that yields no stream raises, naming both sides: an arm whose pool
    quietly came out smaller than it was declared to be is the failure this task cannot afford.
    """
    wanted = [str(Path(directory)) for directory in dataset_dirs]
    if not wanted:
        raise ValueError("no dataset directories were given: an arm needs a pool to draw from")
    streams = stream_returns(dataset)
    present = sorted({stream.dataset_dir for stream in streams})
    empty = [directory for directory in wanted if directory not in set(present)]
    if empty:
        raise ValueError(
            f"these dataset directories yield no streams: {empty} -- the dataset carries "
            f"{present}; a pool smaller than declared would change what the arm measures"
        )
    keep = [stream for stream in streams if stream.dataset_dir in set(wanted)]
    return tuple(sorted(keep, key=_stream_key))


def random_stream_subset(
    streams: Sequence[StreamReturn], count: int, rng: np.random.Generator
) -> tuple[StreamReturn, ...]:
    """A uniform sample of *count* streams without replacement, deterministic given *rng*.

    **The contract, because a recorded rng seed has to be enough to regenerate a subset years
    later:** the pool is sorted into canonical order ``(dataset_dir, episode_file, ix_id)``
    FIRST, the draw is ``rng.choice(len(pool), size=count, replace=False)`` over that order, and
    the result is returned in canonical order again.

    Canonicalising before the draw is what makes the SELECTION independent of the caller's
    ordering; canonicalising after is what makes downstream row order independent of draw order.
    Both are needed and they are different properties.
    """
    canonical = sorted(streams, key=_stream_key)
    total = len(canonical)
    wanted = int(count)
    if wanted < 1:
        raise ValueError(f"a subset needs at least one stream, got count={count!r}")
    if wanted > total:
        raise ValueError(
            f"cannot draw {wanted} streams without replacement from a pool of {total}; "
            "clamping would silently shrink the arm"
        )
    positions = rng.choice(total, size=wanted, replace=False)
    drawn = [canonical[int(position)] for position in positions]
    return tuple(sorted(drawn, key=_stream_key))


def arm_spec_for_flags(
    arm: str,
    *,
    selector: str,
    behaviour_seeds: Sequence[int],
    count: int | None,
) -> ArmSpec:
    """The declared spec of *arm*, refusing any flag that disagrees with it.

    The CLI carries ``--stream-selector`` and its parameters because the brief asks for them; it
    carries this check because a typed flag is an instruction and :data:`SELECTION_ARMS` is the
    declaration, and when the two disagree the declaration wins or the arm is not the arm the
    plan registered.
    """
    spec = SELECTION_ARMS.get(str(arm))
    if spec is None:
        raise ValueError(
            f"unknown arm {arm!r}; the declared arms are {sorted(SELECTION_ARMS)}"
        )
    if str(selector) != spec.selector:
        raise ValueError(
            f"{arm}: the declared selector is {spec.selector!r} but the flags ask for "
            f"{selector!r}; the declaration wins"
        )
    seeds = tuple(sorted(int(seed) for seed in behaviour_seeds))
    if seeds != spec.behaviour_seeds:
        raise ValueError(
            f"{arm}: the declared behaviour seeds are {list(spec.behaviour_seeds)} but the flags "
            f"ask for {list(seeds)}; the declaration wins"
        )
    asked = None if count is None else int(count)
    if asked != spec.count:
        raise ValueError(
            f"{arm}: the declared count is {spec.count} but the flags ask for {asked}; the "
            "declaration wins, because a matched-size design is the whole experiment"
        )
    return spec


def _dirs_for_behaviour_seeds(
    dataset_dirs: Sequence[str | Path], seeds: Sequence[int]
) -> tuple[str, ...]:
    """The directories collected under *seeds*, by the same suffix rule as ``_behaviour_seed``."""
    by_seed: dict[int, list[str]] = {}
    for directory in dataset_dirs:
        path = Path(directory)
        match = re.search(r"seed(\d+)$", path.name)
        if match is None:
            raise ValueError(
                f"{path.name}: no seed suffix, so this directory cannot be attributed to a "
                "behaviour policy and an arm cannot be built from it"
            )
        by_seed.setdefault(int(match.group(1)), []).append(str(path))
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(
            f"the corpus has no directory for behaviour seed(s) {missing}; it carries "
            f"{sorted(by_seed)}"
        )
    return tuple(sorted(name for seed in seeds for name in by_seed[int(seed)]))


def select_arm_streams(
    dataset: TrajectoryWindowDataset,
    spec: ArmSpec,
    *,
    dataset_dirs: Sequence[str | Path],
    rng: np.random.Generator,
) -> tuple[StreamReturn, ...]:
    """The streams *spec* selects from *dataset*, in canonical order.

    *dataset* is always the FULL training split over every directory -- never a dataset rebuilt
    over a subset -- because rebuilding would refit :class:`NormalizationStats` and make the arms
    incomparable in a way no downstream test could see.
    """
    if spec.behaviour_seeds:
        pool = streams_from_datasets(
            dataset, _dirs_for_behaviour_seeds(dataset_dirs, spec.behaviour_seeds)
        )
        observed = {_behaviour_seed(stream) for stream in pool}
        if observed != set(spec.behaviour_seeds):
            raise ValueError(
                f"{spec.arm}: the pool drawn for behaviour seeds "
                f"{list(spec.behaviour_seeds)} actually contains {sorted(observed)}; the "
                "directory names and the streams disagree and the arm is refused"
            )
    else:
        pool = tuple(sorted(stream_returns(dataset), key=_stream_key))

    if spec.selector == "random_subset":
        if spec.count is None:
            raise ValueError(f"{spec.arm}: a random subset needs a declared count")
        return random_stream_subset(pool, int(spec.count), rng)
    if spec.selector == "datasets":
        if spec.count is not None:
            raise ValueError(
                f"{spec.arm}: the 'datasets' selector takes every stream of its directories, so "
                f"a count of {spec.count} would describe a different arm"
            )
        return pool
    if spec.selector == "top_return":
        return top_return_streams(dataset, TOP_RETURN_FRACTION)
    raise ValueError(
        f"{spec.arm}: unknown selector {spec.selector!r}; known selectors are "
        "('top_return', 'datasets', 'random_subset')"
    )


def thread_regime() -> dict[str, Any]:
    """The thread regime of this process, READ AT CALL TIME, to sit beside a timing.

    ``BRIEF_13`` section 11.1: ``docs/data/p4_4_training.json`` carries 15 per-run ``seconds``
    and records ``torch_num_threads = 1``, while ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``
    appear nowhere in it -- and those are a **different knob**, the one that fixed this task's
    test-suite hang.  A timing without its thread regime is not reproducible, and two timings
    from different regimes look comparable and are not.

    Read at call time and never cached: a block captured at import would record the regime of the
    interpreter's startup rather than of the run being timed.  ``runtime_provenance`` is NOT
    extended to carry this -- it lives in the merged ``offline/dt_gate.py``.
    """
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "torch_get_num_threads": int(torch.get_num_threads()),
        "torch_get_num_interop_threads": int(torch.get_num_interop_threads()),
        "read": "at call time, beside the timing it describes",
    }


def delta_verdict(
    mean_difference: float, ci95_half_width: float, delta: float = DELTA_ATT
) -> str:
    """A6's decision with ARM-NEUTRAL names, for a paired difference ``left - right``.

    The same arithmetic as :func:`equivalence_verdict` and deliberately a second implementation
    of it: that function's names (``dt_genuinely_better``) describe a DT-versus-baseline pair, and
    writing one of them into an artifact describing a BC-versus-BC pair would be a false label on
    disk -- the defect class review finding F1 was raised for.  A test asserts the two agree
    under a documented name map over a grid, so the names differ and the decision cannot.

    Lower ATT is better, so ``left_genuinely_better`` means the CI lies entirely below ``-delta``.
    """
    half = float(ci95_half_width)
    margin = float(delta)
    if half < 0.0:
        raise ValueError(f"ci95_half_width must be >= 0, got {ci95_half_width!r}")
    if margin <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta!r}")
    low = float(mean_difference) - half
    high = float(mean_difference) + half
    if low >= -margin and high <= margin:
        return VERDICT_WITHIN_DELTA
    if high < -margin:
        return VERDICT_LEFT_BETTER
    if low > margin:
        return VERDICT_RIGHT_BETTER
    return VERDICT_PAIR_INCONCLUSIVE


def assert_selection_design(
    payload: Mapping[str, Any],
    *,
    declaration: Mapping[str, ArmSpec] | None = None,
    training_seeds: Sequence[int] | None = None,
    held_out_draws: Sequence[int] | None = None,
    rows_per_stream: int | None = None,
    streams_per_behaviour_seed: int | None = None,
) -> None:
    """Refuse a selection that would invalidate the design, BEFORE anything is written.

    **EVERY REFERENCE COMES FROM THE DECLARATION AND NONE FROM THE PAYLOAD** -- the arms and their
    sizes from :data:`SELECTION_ARMS`, the training seeds from ``TRAINING_SEEDS``, the evaluation
    pool from ``HELD_OUT_DRAWS``, the geometry from the two corpus constants.  The payload's own
    ``held_out_draws`` and ``matched_arms`` fields are **cross-checked against those constants and
    never used as the reference**, so a payload that disagrees raises instead of being believed.

    ⚠️ **Corrected 2026-08-12, review finding N1, and this docstring is part of the fix.**  The
    first version read its pool from ``payload["held_out_draws"]``, its sizes from the block being
    checked and its seed set from the first arm's own record.  The reviewer emptied the pool
    field, planted a real leak at ``flow_draw = 1042`` and the whole suite stayed green -- the
    same defect class this task was commissioned to fix in ``_run_report``, in a function whose
    docstring promised the guarantee it could not give.

    Six invariants, each of which can void the result on its own:

    1. **no held-out draw enters training**, against ``HELD_OUT_DRAWS``;
    2. **every arm records a subset for every declared training seed**, against ``TRAINING_SEEDS``;
    3. **every arm's size is the declared one**, against ``SELECTION_ARMS`` (``count``, or
       ``streams_per_behaviour_seed x len(behaviour_seeds)`` for a whole-pool arm);
    4. **every stream came from a declared behaviour checkpoint**, and the recorded composition
       recomputes from the stream list -- the draw must be auditable rather than asserted;
    5. **rows equal ``rows_per_stream x len(streams)``**, and all matched arms have equal rows --
       the decisive comparison would otherwise measure data quantity while claiming to measure
       seed identity;
    6. **all arms share one normalisation digest** -- refitting per arm would make the arms
       incomparable silently.

    The keyword arguments exist so a test can supply its own declaration; ``None`` means "the
    module's", resolved **at call time** rather than bound at import, which is what every
    production call uses.  A caller may substitute a declaration but **cannot** make the payload
    its own reference, which is the property that was missing.
    """
    declaration = SELECTION_ARMS if declaration is None else declaration
    training_seeds = TRAINING_SEEDS if training_seeds is None else training_seeds
    held_out_draws = HELD_OUT_DRAWS if held_out_draws is None else held_out_draws
    rows_per_stream = (
        DECISION_ROWS_PER_STREAM if rows_per_stream is None else rows_per_stream
    )
    streams_per_behaviour_seed = (
        STREAMS_PER_BEHAVIOUR_SEED
        if streams_per_behaviour_seed is None
        else streams_per_behaviour_seed
    )
    arms: Mapping[str, Any] = payload["arms"]
    if not arms:
        raise ValueError("this selection declares no arms")

    undeclared = sorted(set(arms) - set(declaration))
    if undeclared:
        raise ValueError(
            f"the payload carries arm(s) {undeclared} that are not declared in the arm table "
            f"{sorted(declaration)}; an undeclared arm has no reference to be checked against"
        )

    held_out = {int(draw) for draw in held_out_draws}
    if "held_out_draws" in payload:
        recorded = {int(draw) for draw in payload["held_out_draws"]}
        if recorded != held_out:
            raise ValueError(
                f"the payload records a held-out pool of {len(recorded)} draw(s) that is not the "
                f"registered pool of {len(held_out)}; the pool is a declaration and a payload "
                "that disagrees with it is refused rather than believed"
            )

    expected_seeds = sorted(str(int(seed)) for seed in training_seeds)
    expected_matched = sorted(
        arm for arm in arms if declaration[arm].count is not None
    )
    if "matched_arms" in payload and sorted(payload["matched_arms"]) != expected_matched:
        raise ValueError(
            f"the payload calls {sorted(payload['matched_arms'])} the matched arms while the "
            f"declaration makes them {expected_matched}; the matched set decides which arms the "
            "size invariant protects and it is not the payload's to choose"
        )

    rows: dict[tuple[str, str], int] = {}
    for arm in sorted(arms):
        block = arms[arm]
        spec = declaration[arm]
        expected_count = (
            int(spec.count)
            if spec.count is not None
            else int(streams_per_behaviour_seed) * len(spec.behaviour_seeds or tuple(range(5)))
        )
        if int(block["declared_count"]) != expected_count:
            raise ValueError(
                f"{arm}: the payload declares {block['declared_count']} streams but the arm "
                f"table declares {expected_count}; the size is the experiment and it is not the "
                "payload's to redefine"
            )
        if sorted(block["per_training_seed"]) != expected_seeds:
            raise ValueError(
                f"{arm}: records subsets for training seeds {sorted(block['per_training_seed'])} "
                f"while the declared seeds are {expected_seeds}; a missing per training seed "
                "record makes the draw an assertion rather than an audit"
            )

        for seed in expected_seeds:
            entry = block["per_training_seed"][seed]
            streams = entry["streams"]
            if len(streams) != expected_count:
                raise ValueError(
                    f"{arm} seed {seed}: {len(streams)} streams against a declared count of "
                    f"{expected_count}"
                )
            if "per_behaviour_seed_composition" not in entry:
                raise ValueError(
                    f"{arm} seed {seed}: no per-behaviour-seed composition, so which "
                    "checkpoints this subset came from is not recorded"
                )
            composition = {int(k): int(v) for k, v in entry["per_behaviour_seed_composition"].items()}
            if sum(composition.values()) != expected_count:
                raise ValueError(
                    f"{arm} seed {seed}: the composition {composition} does not sum to the "
                    f"declared count {expected_count}"
                )
            recomputed: dict[int, int] = {}
            for stream in streams:
                key = int(stream["behaviour_seed"])
                recomputed[key] = recomputed.get(key, 0) + 1
            if recomputed != composition:
                raise ValueError(
                    f"{arm} seed {seed}: the recorded composition {composition} is not the one "
                    f"its own stream list implies ({recomputed}); the record is not an audit of "
                    "the draw it describes"
                )
            if spec.behaviour_seeds:
                foreign = sorted(set(recomputed) - set(spec.behaviour_seeds))
                if foreign:
                    raise ValueError(
                        f"{arm} seed {seed}: streams from behaviour checkpoint(s) {foreign}, "
                        f"which the arm table does not declare ({list(spec.behaviour_seeds)}); "
                        "the arm's label would not describe its data"
                    )
            leaked = sorted({int(s["flow_draw"]) for s in streams} & held_out)
            if leaked:
                raise ValueError(
                    f"{arm} seed {seed}: training streams drawn from held-out draws {leaked}; "
                    "the evaluation pool is not held out and no number here may be reported"
                )
            recorded_rows = int(entry["training_rows"])
            if recorded_rows != int(rows_per_stream) * len(streams):
                raise ValueError(
                    f"{arm} seed {seed}: {recorded_rows} training rows against "
                    f"{rows_per_stream} x {len(streams)} = "
                    f"{int(rows_per_stream) * len(streams)} implied by its own stream list"
                )
            rows[(arm, seed)] = recorded_rows

    matched_rows = sorted({rows[(arm, seed)] for arm in expected_matched for seed in expected_seeds})
    if len(matched_rows) != 1:
        raise ValueError(
            f"the matched arms {expected_matched} do not have equal training rows: "
            f"{matched_rows}; the decisive comparison would measure data quantity while claiming "
            "to measure seed identity"
        )

    digests = sorted({str(arms[arm]["normalisation_digest"]) for arm in sorted(arms)})
    if len(digests) != 1:
        raise ValueError(
            f"the arms carry {len(digests)} different normalisation digests {digests}; the "
            "statistics must be the full training split's for every arm or the arms are not "
            "comparable"
        )


def assert_reused_arm_reproduces(
    committed: Sequence[EpisodeResult], rerolled: Sequence[EpisodeResult]
) -> dict[str, Any]:
    """Gate B: re-used episodes are only sound if THIS instrument reproduces them exactly.

    ``BRIEF_13`` section 4 requires ``bc_top10``'s 500 episodes to be re-used rather than
    re-rolled -- correct, because re-rolling a settled number is a second measurement of it.  But
    re-use across sessions is only sound if the instrument is the same one, which is what P4.4's
    Gate A established for P4's arms and what this establishes for P4.4's.

    Exact equality on all three reported fields, never a tolerance: a 1e-12 drift is a different
    instrument, not a rounding.  The re-rolled values are discarded by the caller either way --
    they are an instrument check and never a datum.
    """
    want = {(e.arm, e.seed, e.draw_id): e for e in committed}
    got = {(e.arm, e.seed, e.draw_id): e for e in rerolled}
    if not want:
        raise ValueError("no committed episodes were given, so there is nothing to reproduce")
    missing = sorted(str(key) for key in set(want) - set(got))
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(want)} committed episodes were not re-rolled, first "
            f"{missing[:5]}; the gate compares exactly the cells it declared"
        )
    extra = sorted(str(key) for key in set(got) - set(want))
    if extra:
        raise ValueError(
            f"{len(extra)} re-rolled episode(s) have no committed counterpart, first "
            f"{extra[:5]}"
        )

    fields = ("att_horizon", "horizon_vehicle_count", "episode_reward")
    mismatches: list[dict[str, Any]] = []
    for key in sorted(want, key=lambda k: (str(k[0]), str(k[1]), int(k[2]))):
        for name in fields:
            reference = getattr(want[key], name)
            produced = getattr(got[key], name)
            if reference != produced:
                mismatches.append(
                    {
                        "arm": key[0],
                        "seed": key[1],
                        "draw_id": key[2],
                        "field": name,
                        "committed": reference,
                        "rerolled": produced,
                        "difference": produced - reference,
                    }
                )
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"the re-used arm does not reproduce on this instrument: {len(mismatches)} "
            f"mismatch(es), first arm {first['arm']} seed {first['seed']} draw "
            f"{first['draw_id']} field {first['field']} committed {first['committed']!r} "
            f"against {first['rerolled']!r}; no number may be re-used across two instruments"
        )
    return {
        "status": "PASS",
        "compared": len(want),
        "mismatches": [],
        "fields_compared": list(fields),
        "comparison": "exact equality (==), never a tolerance",
        "role": (
            "re-use of a committed arm's episodes is only sound if this session's instrument "
            "reproduces them; the re-rolled values are discarded and never reported"
        ),
    }


def _selection_pairs(present: Sequence[str]) -> list[tuple[str, str]]:
    """Every unordered pair of the arms present, oriented by :data:`SELECTION_ARM_ORDER`."""
    ordered = [arm for arm in SELECTION_ARM_ORDER if arm in set(present)]
    return [
        (ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]


def selection_artifact(
    *,
    episodes: Sequence[EpisodeResult],
    selection: Mapping[str, Any],
    gate_b: Mapping[str, Any],
    env_settings: Mapping[str, Any],
    engine_seed: int,
    delta: float = DELTA_ATT,
) -> dict[str, Any]:
    """The reported P4.5 artifact: cells, every pair, and the three registered predictions.

    Artifact format version: ``p4.5-selection-baselines/1.0``.

    **Every pair is reported unconditionally** -- mean difference, 95 % CI, CI width,
    rank-biserial and the neutral delta verdict -- so no reader depends on delta, which is
    IMPORTED from A6 rather than derived for this comparison.

    **The three predictions are scored by the rules fixed in ``docs/plans/p4.5.md`` section 2.1
    before the run**: the registered forecast on the point estimate, an equivalence claim on the
    whole CI, and the ordering on the three cell means.  Both readings of the primary are
    reported whatever they say; if they disagree, the declared name for that outcome is
    *"consistent with equivalence, not demonstrated at this power"* and it may not be written up
    as a match.

    **No DT arm and no DT comparison appear here at all** (``docs/reviews/P4.4.md`` section 8.6
    binds until P4.3 has run).
    """
    assert_selection_design(selection)
    by_arm = _grouped(episodes)
    for arm in sorted(selection["arms"]):
        if arm not in by_arm:
            raise ValueError(
                f"the selection records arm {arm!r} but no episode does; a trained arm that was "
                "never evaluated must not be reported as part of this comparison"
            )
    if REUSED_ARM not in by_arm:
        raise ValueError(
            f"the {REUSED_ARM!r} arm is missing: it is the measured reference the primary "
            "prediction is about, and it is re-used from the merged P4.4 artifact"
        )
    # Enforced rather than incidental (review finding N7): the previous version simply never
    # built a comparison for an arm outside SELECTION_ARM_ORDER, so a test asserting "no madt
    # cell" could not fail on a BC-only fixture and guaranteed nothing.
    foreign = sorted(set(by_arm) - set(SELECTION_ARM_ORDER))
    if foreign:
        raise ValueError(
            f"episodes for arm(s) {foreign} were passed to the P4.5 artifact, which reports only "
            f"{list(SELECTION_ARM_ORDER)}; docs/reviews/P4.4.md section 8.6 binds until P4.3 has "
            "run, so no DT arm and no DT-versus-baseline comparison may appear here"
        )

    cells = {arm: _cell(results) for arm, results in sorted(by_arm.items())}
    comparisons: dict[str, Any] = {}
    for left, right in _selection_pairs(sorted(by_arm)):
        comparison = paired_comparison(by_arm[left], by_arm[right])
        verdict = delta_verdict(comparison.mean_difference, comparison.ci95_half_width, delta)
        alternative = delta_verdict(
            comparison.mean_difference, comparison.ci95_half_width, DELTA_ATT_DERIVATION
        )
        # Review finding N4: the packet's section 0.1 reads as though this guard ran, and it did
        # not exist.  Added here rather than reworded, because a guard that can only ever REFUSE
        # to emit a verdict is safe to add after the fact -- it cannot manufacture one.  It is an
        # assertion only and adds no field, so the reported artifact stays numerically identical;
        # the observed distances are in the Return Packet.  The smallest is 0.01296, 13x this
        # tolerance, so it did not fire on the committed result.
        distance = min(
            abs(comparison.ci95_low + delta),
            abs(comparison.ci95_low - delta),
            abs(comparison.ci95_high + delta),
            abs(comparison.ci95_high - delta),
        )
        if distance <= DELTA_PROXIMITY_TOLERANCE:
            # The wording deliberately does NOT reuse baselines_artifact's sentence: two raise
            # messages sharing a phrase make every match= on that phrase ambiguous, which is the
            # class BRIEF_14 section 7 measures.  "the P4.5 pair" appears at this site only.
            raise ValueError(
                f"the P4.5 pair {left}_vs_{right} has a CI endpoint {distance:.3e} from the "
                f"margin delta={delta}, inside the {DELTA_PROXIMITY_TOLERANCE} proximity "
                "tolerance; delta's multiplier is a CHOICE, so this verdict would be decided by "
                "the margin's rounding rather than by the data. Report the CI and the distance "
                "instead of a verdict"
            )
        comparisons[f"{left}_vs_{right}"] = {
            **comparison.to_json_obj(),
            "delta": float(delta),
            "delta_verdict": verdict,
            "delta_verdict_at_full_precision_delta": alternative,
            "delta_verdict_turns_on_the_rounding": verdict != alternative,
        }

    spread = (
        sum(BEHAVIOUR_HELDOUT_ATT[seed] for seed in WORST_TWO_BEHAVIOUR_SEEDS) / 2.0
        - sum(BEHAVIOUR_HELDOUT_ATT[seed] for seed in BEST_TWO_BEHAVIOUR_SEEDS) / 2.0
    )
    decisive = comparisons[f"{DECISIVE_CONTRAST[0]}_vs_{DECISIVE_CONTRAST[1]}"]
    power = comparisons[f"{POWER_CONTRAST[0]}_vs_{POWER_CONTRAST[1]}"]
    null_bound = {
        "contrast": f"{DECISIVE_CONTRAST[0]}_vs_{DECISIVE_CONTRAST[1]}",
        "x": max(abs(decisive["ci95_low"]), abs(decisive["ci95_high"])),
        "x_definition": (
            "max(|ci95_low|, |ci95_high|) of the decisive contrast: the largest effect the data "
            "leave standing, not the half-width around a convenient centre"
        ),
        "behaviour_spread_att": spread,
        "behaviour_spread_source": (
            "mean held-out ATT of the two worst behaviour checkpoints minus that of the two "
            "best, from the committed P4.4 measurement; not re-measured here"
        ),
        "power_contrast": f"{POWER_CONTRAST[0]}_vs_{POWER_CONTRAST[1]}",
        "power_contrast_mean_difference": power["mean_difference"],
        "power_contrast_ci95_width": power["ci95_width"],
        "how_a_null_must_be_phrased": (
            "no effect larger than +/-X found, against a "
            f"{spread:.4f} ATT spread in the behaviour policies themselves"
        ),
    }

    predictions = _score_selection_predictions(cells, comparisons, delta)
    per_seed = {
        arm: {
            str(seed): float(
                np.mean([e.att_horizon for e in results if e.seed == seed])
            )
            for seed in sorted({e.seed for e in results if e.seed is not None})
        }
        for arm, results in sorted(by_arm.items())
    }
    spreads = {
        arm: (max(values.values()) - min(values.values())) if values else float("nan")
        for arm, values in per_seed.items()
    }
    return {
        "format_version": SELECTION_BASELINES_FORMAT_VERSION,
        "role": (
            "P4.5: does %BC's advantage come from WHICH behaviour checkpoints produced its "
            "training streams? Matched-size arms on the registered held-out pool, paired by draw"
        ),
        "evaluation_pool": "registered held-out draws 1000-1099 (PREREGISTRATION.md D4)",
        "draw_ids": sorted({episode.draw_id for episode in episodes}),
        "engine_seed": int(engine_seed),
        "env_settings": {k: v for k, v in env_settings.items() if k != "compare_with"},
        "equivalence_margin_delta": float(delta),
        "equivalence_margin_delta_derivation": DELTA_ATT_DERIVATION,
        "delta_provenance": (
            "IMPORTED from PREREGISTRATION.md A6, where it is the DT's own margin over its "
            "behaviour mixture. It is re-used as this project's registered equivalence scale for "
            "this scenario and is NOT derived for this comparison -- a choice, stated rather "
            "than defended, which is why every pair also carries its mean difference, CI, width "
            "and rank-biserial unconditionally"
        ),
        "arm_order": list(SELECTION_ARM_ORDER),
        "pair_orientation": (
            "for i < j in arm_order, mean_difference is mean(arms[i] - arms[j]) over shared draws"
        ),
        "no_dt_comparison": (
            "docs/reviews/P4.4.md section 8.6 binds until P4.3 has run: the DT is prompted at a "
            "target its own P4 review showed is not its best, so no DT-versus-baseline sentence "
            "and no DT arm appears in this task"
        ),
        "cells": cells,
        "per_seed_att_horizon_mean": per_seed,
        "per_seed_spread": spreads,
        "per_seed_spread_role": (
            "plan section 2.3: bc_any_20 is predicted to carry the largest subset-induced "
            "variance of the three matched arms, because its draw varies in WHICH SEEDS appear "
            "as well as in which streams. Predicted before the run, not observed after it"
        ),
        "comparisons": comparisons,
        "registered_predictions": predictions,
        "null_bound": null_bound,
        "reused_arm": {
            "arm": REUSED_ARM,
            "source": "docs/data/p4_4_baselines.json",
            "n_episodes": len(by_arm[REUSED_ARM]),
            "statement": (
                "these episodes are re-used from the merged P4.4 artifact, not re-rolled: the "
                "same model on the same draws, and re-rolling a settled number would be a "
                "second measurement of it"
            ),
            "instrument_check": "see gate_b, which is what makes the re-use sound",
        },
        "gate_b": dict(gate_b),
        "selection": dict(selection),
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
        "runtime": runtime_provenance(),
        "thread_regime": thread_regime(),
    }


def _score_selection_predictions(
    cells: Mapping[str, Any], comparisons: Mapping[str, Any], delta: float
) -> dict[str, Any]:
    """Score the three predictions registered in ``docs/plans/p4.5.md`` section 2 before the run.

    The scoring rules are section 2.1's and were fixed before any number existed: the forecast on
    the point estimate, the equivalence claim on the whole CI, the ordering on the cell means.
    """
    out: dict[str, Any] = {
        "registered_in": (
            "docs/plans/p4.5.md section 2, committed before the first gradient step; scoring "
            "rules in section 2.1, fixed at the same commit"
        )
    }

    primary = comparisons["bc_top10_vs_bc_best2_20"]
    difference = float(primary["mean_difference"])
    within_ci = (
        primary["ci95_low"] >= -float(delta) and primary["ci95_high"] <= float(delta)
    )
    point_holds = abs(difference) <= float(delta)
    out["primary_bc_best2_20_within_delta_of_bc_top10"] = {
        "statement": (
            "bc_best2_20 lands within delta of bc_top10: matched-size random sampling from the "
            "two best checkpoints reproduces %BC"
        ),
        "scored_by": "abs(mean_paired_difference) <= delta",
        "paired_mean_difference_bc_top10_minus_bc_best2_20": difference,
        "delta": float(delta),
        "held": bool(point_holds),
        "equivalence_demonstrated_at_this_power": bool(within_ci),
        "ci95": [primary["ci95_low"], primary["ci95_high"]],
        "ci95_width": primary["ci95_width"],
        "readings_agree": bool(point_holds == within_ci),
        "name_when_they_disagree": (
            "consistent with equivalence, not demonstrated at this power -- reported with the CI "
            "width and NOT written up as a match"
        ),
    }

    secondary = comparisons["bc_best2_20_vs_bc_any_20"]
    any_minus_best2 = -float(secondary["mean_difference"])
    excludes_zero = not (secondary["ci95_low"] <= 0.0 <= secondary["ci95_high"])
    out["secondary_bc_any_20_worse_than_bc_best2_20"] = {
        "statement": (
            "bc_any_20 lands worse than bc_best2_20 by an amount comparable to the "
            "behaviour-policy gap between the best two seeds and the five-seed mixture"
        ),
        "scored_by": (
            "directional: the paired mean difference is positive and its 95% CI excludes 0. The "
            "brief gives no numeric threshold for 'comparable' and none was invented afterwards"
        ),
        "mean_difference_any_minus_best2": any_minus_best2,
        "ci95_excludes_zero": bool(excludes_zero),
        "held": bool(any_minus_best2 > 0.0 and excludes_zero),
        "reference_gap_to_best_two_mean": BEHAVIOUR_GAP_TO_BEST_TWO,
        "reference_gap_to_best_single_seed": BEHAVIOUR_GAP_TO_BEST_SINGLE,
        "reference_gap_note": (
            "BRIEF_13 section 3 called 2.05 the gap to the best two; section 10.1 accepted that "
            "it is the gap to the best SINGLE seed (202). Both are reported and neither is "
            "load-bearing"
        ),
    }

    ordering_arms = ("bc_best2_20", "bc_any_20", "bc_worst2_20")
    means = [float(cells[arm]["att_horizon_mean"]) for arm in ordering_arms]
    out["ordering_best2_then_any_then_worst2"] = {
        "statement": (
            "bc_best2_20 < bc_any_20 < bc_worst2_20 in ATT (lower is better) -- monotone in "
            "behaviour-seed quality. The mechanism claim: not 'does seed identity matter' but "
            "'does performance TRACK behaviour-mode quality'"
        ),
        "scored_by": "the three cell means, strictly ordered",
        "arms": list(ordering_arms),
        "cell_means": means,
        "held": bool(means[0] < means[1] < means[2]),
        "adjacent_contrasts": {
            "bc_best2_20_vs_bc_any_20": {
                "mean_difference": comparisons["bc_best2_20_vs_bc_any_20"]["mean_difference"],
                "ci95": [
                    comparisons["bc_best2_20_vs_bc_any_20"]["ci95_low"],
                    comparisons["bc_best2_20_vs_bc_any_20"]["ci95_high"],
                ],
                "ci95_excludes_zero": not (
                    comparisons["bc_best2_20_vs_bc_any_20"]["ci95_low"]
                    <= 0.0
                    <= comparisons["bc_best2_20_vs_bc_any_20"]["ci95_high"]
                ),
            },
            "bc_any_20_vs_bc_worst2_20": {
                "mean_difference": comparisons["bc_any_20_vs_bc_worst2_20"]["mean_difference"],
                "ci95": [
                    comparisons["bc_any_20_vs_bc_worst2_20"]["ci95_low"],
                    comparisons["bc_any_20_vs_bc_worst2_20"]["ci95_high"],
                ],
                "ci95_excludes_zero": not (
                    comparisons["bc_any_20_vs_bc_worst2_20"]["ci95_low"]
                    <= 0.0
                    <= comparisons["bc_any_20_vs_bc_worst2_20"]["ci95_high"]
                ),
            },
        },
        "if_all_three_land_together": (
            "seed identity does nothing and F3's reading collapses -- learned decisively rather "
            "than from one null contrast, which is why this arm exists"
        ),
    }
    return out


def iql_reward_scale(returns: Sequence[float]) -> float:
    """IQL's published locomotion normalisation: ``1000 / (max_return - min_return)``.

    Computed on the **training split only**, like every other statistic here.  It is the one
    data-dependent quantity in the IQL configuration; the rest are published constants applied
    unchanged.
    """
    values = np.asarray(list(returns), dtype=np.float64)
    if values.size == 0:
        raise ValueError("iql_reward_scale received no returns")
    span = float(values.max() - values.min())
    if span <= 0.0:
        raise ValueError(
            "the returns are constant, so IQL's reward normalisation is undefined "
            f"(min == max == {float(values.min())})"
        )
    return 1000.0 / span


def _resolve_group(
    dataset: TrajectoryWindowDataset, group: tuple[int, int] | None
) -> tuple[int, int]:
    groups = dataset.groups
    if group is None:
        if len(groups) != 1:
            raise ValueError(
                f"this dataset has {len(groups)} (state_dim, n_actions) groups "
                f"{sorted(groups)}; pass group= to say which one to use. C6 forbids padding "
                "across intersections, so each group needs its own tensors"
            )
        return next(iter(groups))
    key = (int(group[0]), int(group[1]))
    if key not in groups:
        raise ValueError(f"group {key} is not present; this dataset has {sorted(groups)}")
    return key


def build_transitions(
    dataset: TrajectoryWindowDataset,
    *,
    group: tuple[int, int] | None = None,
    reward_scale: float = 1.0,
) -> TransitionTable:
    """Flat transitions for one group, including the final one of every stream.

    The final transition's ``next_state`` is **observation row ``T``** -- the post-step state
    after the last decision, which the window loader never yields and which is exactly the
    bootstrap target the registered fairness constraint is about.  Episodes are read through
    ``offline.trajectory_logger.load_episode``, the format's single sanctioned reader, and are
    selected from ``dataset.episode_records`` so the draw-split leakage guard stays in the
    loader.

    States are normalised with the dataset's frozen statistics; rewards are the stream's own
    ``local_reward``, multiplied by *reward_scale*.
    """
    key = _resolve_group(dataset, group)
    stats = dataset.stats
    scale = float(reward_scale)

    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    stream_ids: list[np.ndarray] = []
    steps: list[np.ndarray] = []

    stream_index = 0
    for record in dataset.episode_records:
        episode = load_episode(Path(record.dataset_dir) / record.episode_file)
        length = int(record.episode_length)
        for ix_id in record.ix_ids:
            arrays = episode.intersections[ix_id]
            observation = np.asarray(arrays.state, dtype=np.float32)
            mask = np.asarray(arrays.avail_mask, dtype=np.bool_)
            if (int(observation.shape[1]), int(mask.shape[1])) != key:
                continue
            if observation.shape[0] != length + 1:
                raise ValueError(
                    f"{record.episode_file}: intersection {ix_id!r} has "
                    f"{observation.shape[0]} observation rows for an episode of length "
                    f"{length} (expected {length + 1}); the file violates the C6 alignment "
                    "convention and the final bootstrap target cannot be read from it"
                )
            rows = stats.normalize_state(record.scenario_id, ix_id, observation)
            states.append(rows[:length])
            next_states.append(rows[1 : length + 1])
            actions.append(np.asarray(arrays.action, dtype=np.int64))
            rewards.append(np.asarray(arrays.local_reward, dtype=np.float32) * np.float32(scale))
            stream_ids.append(np.full(length, stream_index, dtype=np.int64))
            steps.append(np.arange(length, dtype=np.int64))
            stream_index += 1

    if not states:
        raise ValueError(
            f"no stream of group {key} in this dataset; it has {sorted(dataset.groups)}"
        )
    return TransitionTable(
        state=torch.from_numpy(np.concatenate(states)),
        next_state=torch.from_numpy(np.concatenate(next_states)),
        action=torch.from_numpy(np.concatenate(actions)),
        reward=torch.from_numpy(np.concatenate(rewards)),
        stream_index=torch.from_numpy(np.concatenate(stream_ids)),
        t=torch.from_numpy(np.concatenate(steps)),
        reward_scale=scale,
    )


def iql_targets(
    table: TransitionTable, next_values: torch.Tensor, gamma: float = IQL_GAMMA
) -> torch.Tensor:
    """``r + gamma * V(s')`` for every transition -- **no ``done`` term anywhere**.

    This is the registered fairness constraint in one line: ``terminated`` is hardcoded ``False``
    and every episode ends by time-limit truncation, so the horizon is not absorbing and the last
    transition of every stream bootstraps like any other.

    ⚠️ **The guarantee is at the CALL SITE, not here, and the test that provides it says so**
    (corrected 2026-08-12, review finding F1).  ``gamma`` and ``next_values`` are both supplied by
    :func:`train_iql`, so a mutation placed there -- ``gamma=0.0``, or a zeroed bootstrap value --
    leaves this function untouched and once survived the whole suite.  What guards the path
    training actually takes is
    ``tests/test_offline_baselines.py::test_the_training_loop_bootstraps_through_the_horizon_at_its_call_site``,
    which spies on the module-level name the loop resolves and asserts on what the loop passes.
    """
    values = next_values.reshape(-1)
    if values.shape != table.reward.shape:
        raise ValueError(
            f"next_values has {tuple(values.shape)} entries against {tuple(table.reward.shape)} "
            "transitions"
        )
    return table.reward + float(gamma) * values


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def _schedule(optimiser: torch.optim.Optimizer, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    """The DT's linear warmup, shortened only when the run is shorter than the warmup."""
    warmup = min(WARMUP_STEPS, max(1, total // 2))
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup)
    )


def _curve(losses: Sequence[float], total: int) -> tuple[tuple[float, ...], bool]:
    """Window means of the training loss and the plateau diagnostic, as P4 computes them."""
    window = min(PLATEAU_WINDOW, total)
    while total % window:
        window -= 1
    means = window_means(losses, window)
    return means, (plateau_reached(means) if len(means) >= 3 else False)


def _prepare_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Filesystem-mutation barrier: refuse before training, and create no directory."""
    destination = Path(checkpoint_path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}; nothing is created here"
        )
    return destination


def _diagnostic_rows(count: int, device: torch.device) -> torch.Tensor:
    """A deterministic, capped row sample for the post-training diagnostics."""
    if count <= DIAGNOSTIC_ROWS:
        return torch.arange(count, dtype=torch.int64, device=device)
    stride = count // DIAGNOSTIC_ROWS
    return torch.arange(0, stride * DIAGNOSTIC_ROWS, stride, dtype=torch.int64, device=device)


def train_bc(
    stacked: dict[str, torch.Tensor],
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    method: str,
    declared_gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: NormalizationStats,
    scenario_id: str,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainRecord:
    """Train one BC (or %BC) seed for exactly *declared_gradient_steps* steps.

    The batches, the loss and the optimiser are the DT's: uniform sampling with replacement from
    a ``numpy`` generator seeded by *seed*, ``action_loss`` with ``ignore_index = -1`` over every
    valid window position, masked logits, AdamW with the DT's learning rate, weight decay,
    warmup and gradient clip.  **The only difference from the DT is the architecture** -- a
    per-position MLP that sees one state and nothing else.
    """
    from agent.utils.utils import Utils

    total = int(declared_gradient_steps)
    if total < 1:
        raise ValueError(f"declared_gradient_steps must be >= 1, got {declared_gradient_steps}")
    count = int(stacked["state"].shape[0])
    if count < 1:
        raise ValueError("the stacked dataset is empty")
    destination = _prepare_checkpoint_path(checkpoint_path)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = TrunkConfig(state_dim=int(state_dim), n_actions=int(n_actions))
    model = MLPTrunk(config, int(n_actions)).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = _schedule(optimiser, total)

    tensors = {key: value.to(device) for key, value in stacked.items()}
    generator = np.random.default_rng(int(seed))
    losses: list[float] = []
    model.train()
    started = time.time()
    for step in range(total):
        index = torch.from_numpy(
            generator.integers(0, count, size=int(batch_size)).astype(np.int64)
        ).to(device)
        action = tensors["action"][index]
        logits = masked_action_logits(
            model(tensors["state"][index]), tensors["avail_mask"][index]
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

    means, plateaued = _curve(losses, total)
    diagnostics = _bc_diagnostics(model, tensors, device)
    diagnostics["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    diagnostics["method"] = method
    diagnostics["training_rows"] = count

    merged = {f"policy.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    digest = canonical_state_dict_digest(merged)
    torch.save(
        {
            "format_version": "bc-checkpoint/1.0",
            "config": config.to_json_obj(),
            "model": merged,
            "canonical_digest": digest,
            "normalise": True,
            "scenario_id": str(scenario_id),
            "stats": stats.to_json_obj(),
            "intersection_ids": [],
            "provenance": {
                **dict(provenance),
                "method": method,
                "seed": int(seed),
                "gradient_steps": int(total),
                "declared_gradient_steps": int(declared_gradient_steps),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "window_means": list(means),
                "plateaued": bool(plateaued),
                "diagnostics": diagnostics,
                "runtime": runtime_provenance(),
            },
        },
        destination,
    )
    return TrainRecord(
        method=method,
        seed=int(seed),
        gradient_steps=total,
        declared_gradient_steps=int(declared_gradient_steps),
        losses=tuple(losses),
        window_means=means,
        plateaued=bool(plateaued),
        checkpoint_path=str(destination),
        canonical_digest=digest,
        file_sha256=_sha256_file(destination),
        seconds=float(seconds),
        diagnostics=diagnostics,
    )


def _bc_diagnostics(
    model: MLPTrunk, tensors: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    """How often the trained policy reproduces the logged action, on TRAINING data only.

    A mechanism number, not a selection criterion: nothing in this task chooses a model, a
    hyperparameter or a checkpoint by it.
    """
    rows = _diagnostic_rows(int(tensors["state"].shape[0]), device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = masked_action_logits(
                model(tensors["state"][rows]), tensors["avail_mask"][rows]
            )
            action = tensors["action"][rows]
            valid = action >= 0
            agreement = (logits.argmax(dim=-1) == action) & valid
            total = int(valid.sum())
    finally:
        model.train(was_training)
    return {
        "behaviour_agreement": float(int(agreement.sum()) / total) if total else float("nan"),
        "diagnostic_positions": total,
    }


def train_iql(
    table: TransitionTable,
    *,
    state_dim: int,
    n_actions: int,
    seed: int,
    declared_gradient_steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path,
    stats: NormalizationStats,
    scenario_id: str,
    provenance: dict[str, Any],
    log_every: int = 0,
) -> TrainRecord:
    """Train one IQL seed for exactly *declared_gradient_steps* steps.

    Published D4RL-locomotion hyperparameters, unswept: expectile ``tau = 0.7``, AWR temperature
    ``beta = 3.0``, ``gamma = 0.99``, Polyak ``0.005``, weight clip ``100``.  One step updates
    ``V``, then ``Q``, then the policy, then the target network -- the published order.

    The Q target comes from :func:`iql_targets`, which carries no ``done`` term, so the final
    transition of every episode bootstraps through the truncation boundary.

    The policy loss is computed on **unmasked** logits: the transition table carries no
    availability mask, the logged action is legal by construction, and every mask in this corpus
    is all-``True`` (P3 review), so masking here would be a no-op with a second convention to
    keep consistent.  ``act()`` masks at evaluation, where it can bind.
    """
    from agent.utils.utils import Utils

    total = int(declared_gradient_steps)
    if total < 1:
        raise ValueError(f"declared_gradient_steps must be >= 1, got {declared_gradient_steps}")
    count = len(table)
    if count < 1:
        raise ValueError("the transition table is empty")
    destination = _prepare_checkpoint_path(checkpoint_path)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = TrunkConfig(state_dim=int(state_dim), n_actions=int(n_actions))
    policy = MLPTrunk(config, int(n_actions)).to(device)
    q = MLPTrunk(config, int(n_actions)).to(device)
    value = MLPTrunk(config, 1).to(device)
    q_target = MLPTrunk(config, int(n_actions)).to(device)
    q_target.load_state_dict(q.state_dict())
    for parameter in q_target.parameters():
        parameter.requires_grad_(False)

    optimisers = {
        "policy": torch.optim.AdamW(
            policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        ),
        "q": torch.optim.AdamW(q.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        "v": torch.optim.AdamW(value.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    }
    schedules = {name: _schedule(opt, total) for name, opt in optimisers.items()}

    device_table = table.to(device)
    generator = np.random.default_rng(int(seed))
    losses: list[float] = []
    component: dict[str, list[float]] = {"v": [], "q": [], "policy": []}
    for module in (policy, q, value):
        module.train()
    started = time.time()
    for step in range(total):
        index = torch.from_numpy(
            generator.integers(0, count, size=int(batch_size)).astype(np.int64)
        ).to(device)
        batch = device_table.select(index)
        action = batch.action.reshape(-1, 1)

        with torch.no_grad():
            q_taken = q_target(batch.state).gather(1, action).reshape(-1)

        # 1. V: expectile regression of the target Q onto V.
        v_pred = value(batch.state).reshape(-1)
        residual = q_taken - v_pred
        weight = torch.where(
            residual < 0, torch.full_like(residual, 1.0 - IQL_TAU), torch.full_like(residual, IQL_TAU)
        )
        v_loss = (weight * residual.pow(2)).mean()
        optimisers["v"].zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), GRAD_CLIP)
        optimisers["v"].step()
        schedules["v"].step()

        # 2. Q: TD regression onto r + gamma * V(s'), bootstrapping through every boundary.
        with torch.no_grad():
            next_value = value(batch.next_state).reshape(-1)
            target = iql_targets(batch, next_value, gamma=IQL_GAMMA)
        q_pred = q(batch.state).gather(1, action).reshape(-1)
        q_loss = F.mse_loss(q_pred, target)
        optimisers["q"].zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), GRAD_CLIP)
        optimisers["q"].step()
        schedules["q"].step()

        # 3. Policy: advantage-weighted regression onto the logged action.
        with torch.no_grad():
            advantage = q_taken - v_pred.detach()
            awr_weight = torch.clamp(torch.exp(IQL_BETA * advantage), max=IQL_WEIGHT_CLIP)
        log_probs = torch.log_softmax(policy(batch.state), dim=-1).gather(1, action).reshape(-1)
        policy_loss = -(awr_weight * log_probs).mean()
        optimisers["policy"].zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimisers["policy"].step()
        schedules["policy"].step()

        # 4. Polyak update of the target network.
        with torch.no_grad():
            for online, target_parameter in zip(q.parameters(), q_target.parameters()):
                target_parameter.mul_(1.0 - IQL_POLYAK).add_(online, alpha=IQL_POLYAK)

        losses.append(float(policy_loss.detach()))
        component["v"].append(float(v_loss.detach()))
        component["q"].append(float(q_loss.detach()))
        component["policy"].append(float(policy_loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  iql seed {seed} step {step + 1}/{total} "
                f"policy {np.mean(component['policy'][-log_every:]):.5f} "
                f"q {np.mean(component['q'][-log_every:]):.5f} "
                f"v {np.mean(component['v'][-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

    means, plateaued = _curve(losses, total)
    diagnostics = _iql_diagnostics(policy, q_target, value, device_table, device)
    diagnostics.update(
        {
            "tau": IQL_TAU,
            "beta": IQL_BETA,
            "gamma": IQL_GAMMA,
            "polyak": IQL_POLYAK,
            "weight_clip": IQL_WEIGHT_CLIP,
            "reward_scale": float(table.reward_scale),
            "hyperparameter_provenance": (
                "published IQL D4RL-locomotion values, unswept, transplanted onto a discrete "
                "8-phase action space; see docs/plans/p4.4.md section 3.5 for why no sweep ships"
            ),
            "final_v_loss": component["v"][-1],
            "final_q_loss": component["q"][-1],
            "final_policy_loss": component["policy"][-1],
            "parameter_count": int(
                sum(p.numel() for module in (policy, q, value) for p in module.parameters())
            ),
            "training_rows": count,
        }
    )

    merged: dict[str, torch.Tensor] = {}
    for prefix, module in (("policy", policy), ("q", q), ("v", value), ("q_target", q_target)):
        for key, tensor in module.state_dict().items():
            merged[f"{prefix}.{key}"] = tensor.detach().cpu()
    digest = canonical_state_dict_digest(merged)
    torch.save(
        {
            "format_version": "iql-checkpoint/1.0",
            "config": config.to_json_obj(),
            "model": merged,
            "canonical_digest": digest,
            "normalise": True,
            "scenario_id": str(scenario_id),
            "stats": stats.to_json_obj(),
            "intersection_ids": [],
            "provenance": {
                **dict(provenance),
                "method": "iql",
                "seed": int(seed),
                "gradient_steps": int(total),
                "declared_gradient_steps": int(declared_gradient_steps),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "window_means": list(means),
                "plateaued": bool(plateaued),
                "diagnostics": diagnostics,
                "runtime": runtime_provenance(),
            },
        },
        destination,
    )
    return TrainRecord(
        method="iql",
        seed=int(seed),
        gradient_steps=total,
        declared_gradient_steps=int(declared_gradient_steps),
        losses=tuple(losses),
        window_means=means,
        plateaued=bool(plateaued),
        checkpoint_path=str(destination),
        canonical_digest=digest,
        file_sha256=_sha256_file(destination),
        seconds=float(seconds),
        diagnostics=diagnostics,
    )


def _iql_diagnostics(
    policy: MLPTrunk,
    q_target: MLPTrunk,
    value: MLPTrunk,
    table: TransitionTable,
    device: torch.device,
) -> dict[str, Any]:
    """Whether the transplanted configuration degenerated -- computed on TRAINING data only.

    ``docs/plans/p4.4.md`` section 3.5 declines the authorised tau/beta sweep because its
    selection criterion cannot rank either parameter (beta enters no value loss at all, and tau
    defines the expectile loss).  What ships instead is this: if every AWR weight is clipped, or
    every weight is ~1, the extraction has collapsed toward uniform or toward plain BC, and the
    number says so mechanistically.  **Nothing here selects anything.**
    """
    rows = _diagnostic_rows(len(table), device)
    sample = table.select(rows)
    modes = [module.training for module in (policy, q_target, value)]
    for module in (policy, q_target, value):
        module.eval()
    try:
        with torch.no_grad():
            action = sample.action.reshape(-1, 1)
            q_taken = q_target(sample.state).gather(1, action).reshape(-1)
            v_pred = value(sample.state).reshape(-1)
            advantage = q_taken - v_pred
            raw = torch.exp(IQL_BETA * advantage)
            weights = torch.clamp(raw, max=IQL_WEIGHT_CLIP)
            agreement = (policy(sample.state).argmax(dim=-1) == sample.action).float().mean()
    finally:
        for module, mode in zip((policy, q_target, value), modes):
            module.train(mode)

    weights64 = weights.double()
    total = float(weights64.sum())
    quantiles = torch.quantile(
        weights64, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=weights64.device)
    )
    return {
        "awr_weight_mean": float(weights64.mean()),
        "awr_weight_q10": float(quantiles[0]),
        "awr_weight_median": float(quantiles[1]),
        "awr_weight_q90": float(quantiles[2]),
        "awr_weight_clipped_fraction": float((raw >= IQL_WEIGHT_CLIP).double().mean()),
        "awr_weight_near_zero_fraction": float((weights64 < 0.01).double().mean()),
        "awr_weight_effective_sample_size": (
            float(total * total / float((weights64 * weights64).sum()) / weights64.numel())
            if total > 0
            else 0.0
        ),
        "advantage_mean": float(advantage.double().mean()),
        "advantage_std": float(advantage.double().std(unbiased=True)),
        "behaviour_agreement": float(agreement),
        "diagnostic_positions": int(weights64.numel()),
    }


# ----------------------------------------------------------------------
# Effect sizes, the A6 verdict and the recovered fraction
# ----------------------------------------------------------------------


def rank_biserial(result: WilcoxonResult) -> float:
    """Matched-pairs rank-biserial correlation ``(W+ - W-) / (W+ + W-)``.

    The effect size that belongs beside a signed-rank p-value: the test is scale-free, so a
    tiny but perfectly consistent shift is highly significant, and the p-value alone cannot say
    whether the difference matters.  Positive means the FIRST argument of the test was larger
    more often and by larger ranks.
    """
    total = float(result.w_plus) + float(result.w_minus)
    if total <= 0.0:
        return 0.0
    return (float(result.w_plus) - float(result.w_minus)) / total


def paired_comparison(
    left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]
) -> PairedComparison:
    """Paired per-draw comparison of two arms over their shared draws.

    The per-draw unit is the mean over training seeds, as in P4, so seed and draw stay crossed.
    Amendment A5 point 3 makes a comparison without shared draws **void**, which
    ``dt_gate._paired`` enforces by raising.
    """
    left_values, right_values, shared = _paired(left, right)
    arms = (sorted({r.arm for r in left}), sorted({r.arm for r in right}))
    if len(arms[0]) != 1 or len(arms[1]) != 1:
        raise ValueError(f"a paired comparison needs one arm on each side, got {arms}")

    differences = [a - b for a, b in zip(left_values, right_values)]
    cell = mean_ci95(differences)
    result = wilcoxon_signed_rank(left_values, right_values)
    return PairedComparison(
        left_arm=arms[0][0],
        right_arm=arms[1][0],
        n_shared_draws=len(shared),
        draw_ids=tuple(shared),
        mean_left=float(np.mean(left_values)),
        mean_right=float(np.mean(right_values)),
        mean_difference=cell.mean,
        ci95_half_width=cell.ci95,
        ci95_width=2.0 * cell.ci95,
        ci95_low=cell.mean - cell.ci95,
        ci95_high=cell.mean + cell.ci95,
        median_difference=float(np.median(differences)),
        wins=int(sum(1 for d in differences if d < 0)),
        losses=int(sum(1 for d in differences if d > 0)),
        ties=int(sum(1 for d in differences if d == 0)),
        rank_biserial=rank_biserial(result),
        wilcoxon=result,
    )


def equivalence_verdict(
    mean_difference: float, ci95_half_width: float, delta: float = DELTA_ATT
) -> str:
    """A6's verdict for a paired difference ``DT - baseline`` (lower ATT is better).

    ``[-delta, +delta]`` is **closed**: a CI landing exactly on the margin lies within it.
    "Entirely below ``-delta``" is **strict**, so a CI whose upper end touches ``-delta`` is
    inconclusive rather than decisive -- the conservative reading in both directions.

    A6 names three branches.  The fourth, a CI entirely above ``+delta``, is returned as
    :data:`VERDICT_BASELINE_BETTER` instead of being folded into "inconclusive": it is a
    decisive result in the other direction, and reporting it as an absence of one would be
    wrong.  The asymmetry is raised with the coordinator rather than resolved here.
    """
    half = float(ci95_half_width)
    margin = float(delta)
    if half < 0.0:
        raise ValueError(f"ci95_half_width must be >= 0, got {ci95_half_width!r}")
    if margin <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta!r}")
    low = float(mean_difference) - half
    high = float(mean_difference) + half
    if low >= -margin and high <= margin:
        return VERDICT_MATCHES
    if high < -margin:
        return VERDICT_DT_BETTER
    if low > margin:
        return VERDICT_BASELINE_BETTER
    return VERDICT_INCONCLUSIVE


def recovered_fraction(att_reference: float, att_arm: float, att_dt: float) -> float:
    """``(reference - arm) / (reference - dt)`` -- reported unconditionally (A6 clarified).

    1.0 means the arm recovered the DT's entire margin over the behaviour policy, 0.0 means it
    recovered none of it, and a negative value means it lands below the behaviour policy.  It
    carries what the binary verdict compresses away: a baseline inside the equivalence margin
    can still have recovered only a fifth of the effect under study.
    """
    denominator = float(att_reference) - float(att_dt)
    if denominator == 0.0:
        raise ValueError(
            "the reference and the DT have the same ATT, so the recovered fraction divides by "
            "zero; there is no margin to recover"
        )
    return (float(att_reference) - float(att_arm)) / denominator


# ----------------------------------------------------------------------
# Campaign integrity and the artifact
# ----------------------------------------------------------------------


def assert_campaign_complete(
    requested: Sequence[tuple[str, int | None, int]],
    produced: Sequence[EpisodeResult],
) -> None:
    """Refuse to report a partial campaign: completed runs must equal runs requested.

    The condition attached to running the evaluation in-session.  A campaign that aborted
    halfway would otherwise produce a smaller, quieter, perfectly plausible cell.
    """
    from collections import Counter

    want = Counter(tuple(item) for item in requested)
    got = Counter((r.arm, r.seed, r.draw_id) for r in produced)
    missing = sorted(str(k) for k in (want - got))
    extra = sorted(str(k) for k in (got - want))
    if missing:
        raise ValueError(
            f"incomplete campaign: {len(missing)} of {sum(want.values())} requested runs are "
            f"missing, first {missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"{len(extra)} produced run(s) were not requested, first {extra[:5]}; a cell must "
            "describe exactly the runs the campaign asked for"
        )


def merge_training_runs(
    existing: Mapping[str, Any], fresh: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Runs of *fresh*, plus the runs of *existing* for methods *fresh* did not train.

    Training the three methods for 40,000 steps does not fit one job under the campaign's
    30-minute condition, so it runs in chunks -- and a chunk that simply overwrote the artifact
    would leave a run set that looks complete and is not.  A merge across two different
    declarations is refused rather than reconciled: the budget, the training draws and the
    intersection group must match, because mixing two designs in one artifact is exactly the
    kind of quiet defect this project keeps finding.
    """
    for field_name in ("declared_gradient_steps", "training_draw_ids", "scenario_id", "group"):
        if field_name in existing and field_name in fresh:
            if existing[field_name] != fresh[field_name]:
                raise ValueError(
                    f"refusing to merge two training artifacts that disagree on "
                    f"{field_name}: {existing[field_name]!r} against {fresh[field_name]!r}; "
                    "they describe two different designs"
                )
    trained_now = {run["method"] for run in fresh["runs"]}
    kept = [run for run in existing.get("runs", []) if run["method"] not in trained_now]
    return [*kept, *fresh["runs"]]


def _grouped(episodes: Sequence[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    out: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        out.setdefault(episode.arm, []).append(episode)
    return out


def baselines_artifact(
    *,
    episodes: Sequence[EpisodeResult],
    training: dict[str, Any],
    gate_a: dict[str, Any],
    env_settings: dict[str, Any],
    engine_seed: int,
    delta: float = DELTA_ATT,
    behaviour_reference: str = "mappo1000",
    emit_verdicts: bool = True,
) -> dict[str, Any]:
    """The reported artifact: cells, paired comparisons, verdicts and recovered fractions.

    Every quantity A5 and A6 make unconditional is present for every arm: ``att_horizon``,
    ``vehicle_count`` at the horizon, the draw ids, the CI width, the effect size, the verdict
    and the recovered fraction.  The verdict is computed under **both** the declared ``delta``
    and its full-precision derivation, and the artifact refuses to be built if the two disagree
    -- A6's multiplier is a choice, and a verdict must not turn on a rounding.

    **Generalised additively for P4.6** (``BRIEF_17`` section 11, finding A2), because this
    function hard-required the ``mappo1000`` arm and emitted a verdict unconditionally, and a
    data-quality ladder has a different behaviour policy on every tier:

    * ``behaviour_reference`` names the arm the recovered fraction and the behaviour comparisons
      are taken against.  It defaults to ``"mappo1000"``, which is P4.4's arm.
    * ``emit_verdicts=False`` removes every equivalence-verdict quantity -- the verdict itself,
      its full-precision cross-check, ``delta``, the CI-endpoint distance, the two top-level
      margin fields, the decision rule and the delta-scored forecasts -- and records the reporting
      rule and the reference instead.  ``BRIEF_17`` section 4 forbids per-tier verdicts, and A6's
      delta is a ``mappo1000`` quantity that cannot be derived per tier without circularity.

    **The defaults reproduce this function's pre-P4.6 output exactly**, key for key and value for
    value; ``docs/data/p4_4_baselines.json`` regenerating byte-identically through this path
    (outside its self-describing ``runtime`` block) is the declared regression gate.
    """
    by_arm = _grouped(episodes)
    for required in ("madt", behaviour_reference):
        if required not in by_arm:
            raise ValueError(
                f"the artifact needs the {required!r} arm: without it there is no comparison "
                f"and no recovered fraction to report; present arms are {sorted(by_arm)}"
            )

    cells = {arm: _cell(results) for arm, results in sorted(by_arm.items())}
    att = {arm: cell["att_horizon_mean"] for arm, cell in cells.items()}
    reference, dt_mean = att[behaviour_reference], att["madt"]

    comparisons: dict[str, Any] = {}
    for method in METHODS:
        if method not in by_arm:
            continue
        comparison = paired_comparison(by_arm["madt"], by_arm[method])
        verdicts: dict[str, Any] = {}
        if emit_verdicts:
            verdict = equivalence_verdict(
                comparison.mean_difference, comparison.ci95_half_width, delta
            )
            alternative = equivalence_verdict(
                comparison.mean_difference, comparison.ci95_half_width, DELTA_ATT_DERIVATION
            )
            if verdict != alternative:
                raise ValueError(
                    f"the verdict for {method!r} depends on delta's rounding: {verdict!r} at "
                    f"delta={delta} and {alternative!r} at delta={DELTA_ATT_DERIVATION}. A6's "
                    "multiplier is a choice and a verdict must not turn on it; report this instead "
                    "of picking one"
                )
            distance = min(
                abs(comparison.ci95_low + delta),
                abs(comparison.ci95_low - delta),
                abs(comparison.ci95_high + delta),
                abs(comparison.ci95_high - delta),
            )
            if distance <= DELTA_PROXIMITY_TOLERANCE:
                raise ValueError(
                    f"a CI endpoint for {method!r} sits {distance:.3e} from the equivalence margin "
                    f"delta={delta}, which is within {DELTA_PROXIMITY_TOLERANCE}. A6's multiplier of "
                    "1.0 is a CHOICE, so a verdict decided at that distance would be decided by the "
                    "margin's rounding rather than by the data; report the CI and the distance "
                    "instead of a verdict"
                )
            verdicts = {
                "verdict": verdict,
                "verdict_at_full_precision_delta": alternative,
                "delta": float(delta),
                "distance_from_ci_endpoints_to_delta": distance,
            }
        direct = recovered_fraction(reference, att[method], dt_mean)
        # Second route to the same number: 1 - mean(arm - DT) / (reference - DT).  The two agree
        # exactly only for a balanced design, which is what "seed crossed with draw" gives.
        paired_route = 1.0 - (-comparison.mean_difference) / (reference - dt_mean)
        if abs(direct - paired_route) > 1e-9:
            raise ValueError(
                f"the recovered fraction for {method!r} disagrees between its two routes: "
                f"{direct} against {paired_route}; that means the seed-by-draw design is not "
                "balanced and the cell mean is not the mean of the per-draw means"
            )
        comparisons[f"madt_vs_{method}"] = {
            **comparison.to_json_obj(),
            **verdicts,
            "recovered_fraction": direct,
            "recovered_fraction_paired_route": paired_route,
        }

    behaviour: dict[str, Any] = {}
    for arm in ("madt", *METHODS):
        if arm in by_arm and arm != behaviour_reference:
            behaviour[f"{behaviour_reference}_vs_{arm}"] = paired_comparison(
                by_arm[behaviour_reference], by_arm[arm]
            ).to_json_obj()

    draw_ids = sorted({episode.draw_id for episode in episodes})
    verdict_fields: dict[str, Any] = (
        {
            "equivalence_margin_delta": float(delta),
            "equivalence_margin_delta_derivation": DELTA_ATT_DERIVATION,
            "decision_rule": (
                "PREREGISTRATION.md A6: matches iff the 95% CI of the paired per-draw difference "
                "(DT - baseline) lies entirely within [-delta, +delta]; the DT is genuinely better "
                "iff it lies entirely below -delta; a CI entirely above +delta is the baseline "
                "genuinely better, a branch A6 does not name; anything else is inconclusive at this "
                "power, reported with the CI width"
            ),
            "registered_forecasts": _forecast_outcomes(comparisons, by_arm, delta),
        }
        if emit_verdicts
        else {
            "behaviour_reference": behaviour_reference,
            "reporting_rule": (
                "paired mean differences with 95 % CIs, CI widths and rank-biserial effect sizes; "
                "no equivalence verdict is issued, because A6's delta is a mappo1000 quantity and "
                "no per-tier margin can be derived before the run without using its own result "
                "(BRIEF_17 section 4)"
            ),
        }
    )
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "P4.4: BC, %BC and IQL against the P4 Decision Transformer on the registered "
            "held-out pool, paired by draw"
        ),
        "evaluation_pool": "registered held-out draws 1000-1099 (PREREGISTRATION.md D4)",
        "draw_ids": draw_ids,
        "engine_seed": int(engine_seed),
        "env_settings": {k: v for k, v in env_settings.items() if k != "compare_with"},
        **verdict_fields,
        "attribution_caveat": (
            "DT minus baseline is a COMBINED difference: attention/context plus return-to-go "
            "conditioning plus the timestep embedding. It is not a sequence-modelling effect; "
            "P4.3 (RTG) and P5.3 (no-RTG, context length K) own the decomposition"
        ),
        "iql_caveat": (
            "IQL here is untuned, running published D4RL-locomotion values transplanted onto a "
            "discrete 8-phase action space: a losing untuned IQL is not evidence that MADT "
            "outperforms IQL"
        ),
        "cells": cells,
        "comparisons": comparisons,
        "behaviour_policy_comparisons": behaviour,
        "gate_a": dict(gate_a),
        "training": dict(training),
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
        "runtime": runtime_provenance(),
    }


def _forecast_outcomes(
    comparisons: Mapping[str, Any],
    by_arm: Mapping[str, list[EpisodeResult]],
    delta: float,
) -> dict[str, Any]:
    """Score the forecasts registered in ``docs/plans/p4.4.md`` section 4 before training."""
    out: dict[str, Any] = {
        "registered_in": "docs/plans/p4.4.md section 4, before the first gradient step"
    }
    if "madt_vs_bc" in comparisons:
        difference = float(comparisons["madt_vs_bc"]["mean_difference"])
        out["primary_bc_within_delta_of_the_dt"] = {
            "statement": "BC lands within delta of the DT on the paired held-out mean",
            "paired_mean_difference_dt_minus_bc": difference,
            "delta": float(delta),
            "held": bool(abs(difference) <= float(delta)),
        }
    if "bc" in by_arm and "bc_top10" in by_arm:
        secondary = paired_comparison(by_arm["bc"], by_arm["bc_top10"])
        out["secondary_bc_top10_within_delta_of_bc"] = {
            "statement": "%BC lands within delta of BC on the paired held-out mean",
            "paired_mean_difference_bc_minus_bc_top10": secondary.mean_difference,
            "delta": float(delta),
            "held": bool(abs(secondary.mean_difference) <= float(delta)),
        }
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``gate`` (reproduce P4), ``train``, ``evaluate`` (one method), ``report``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.offline_baselines",
        description="Train and evaluate P4.4's offline baselines: BC, %BC and IQL.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="a collection manifest of the training scenario; the evaluation env settings are "
        "read from it rather than restated",
    )
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--scenario-key", default="cityflow1x1")
    parser.add_argument("--scenario-id", default="cityflow1x1")
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--work-dir", default="output/p4_4")
    parser.add_argument("--checkpoint-dir", default="output/p4_4/checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch threads for this process; 1 is the default because the unpinned path "
        "DEADLOCKS on this workload (see pin_torch_threads) and reproduces P4 bit-identically",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="re-roll P4's cited arms and refuse on any difference")
    gate.add_argument("--dt-checkpoint", action="append", default=[], metavar="SEED=PATH")
    gate.add_argument("--mappo-checkpoint", action="append", default=[], metavar="SEED=PATH")
    gate.add_argument("--dt-steps", type=int, default=40_000)
    gate.add_argument("--reference", default="docs/data/p4_gate.json")
    gate.add_argument("--thresholds", default="docs/data/p4_heldout_thresholds.json")

    train = sub.add_parser("train", help="train every method and seed to the declared budget")
    train.add_argument("--dataset-dir", action="append", required=True)
    train.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    train.add_argument("--methods", default=",".join(METHODS))
    train.add_argument("--log-every", type=int, default=2000)
    # P4.5.  The default reproduces P4.4's path exactly; any other value trains ONE declared arm
    # across the five training seeds and writes the selection artifact instead.
    train.add_argument(
        "--stream-selector",
        default="top_return",
        choices=("top_return", "datasets", "random_subset"),
        help="which streams the arm trains on; 'top_return' is P4.4's %%BC filter",
    )
    train.add_argument("--selection-arm", default=None, choices=sorted(SELECTION_ARMS))
    train.add_argument("--selector-seed", action="append", type=int, default=[])
    train.add_argument("--subset-count", type=int, default=None)

    evaluate = sub.add_parser("evaluate", help="evaluate ONE method over the held-out pool")
    evaluate.add_argument(
        "--method", required=True, choices=[*METHODS, *sorted(SELECTION_ARMS)]
    )
    evaluate.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)

    report = sub.add_parser("report", help="merge the per-method runs into the artifact")
    report.add_argument("--methods", default=",".join(METHODS))
    # P4.6 (BRIEF_17 section 11, finding A2): a ladder tier's behaviour policy is not mappo1000,
    # and BRIEF_17 section 4 forbids per-tier equivalence verdicts.  Both defaults are P4.4's.
    report.add_argument("--behaviour-reference", default="mappo1000")
    report.add_argument(
        "--no-verdicts",
        action="store_true",
        help="report differences, CIs, widths and effect sizes without any equivalence verdict",
    )

    gate_b = sub.add_parser(
        "gate-selection",
        help="Gate B: prove this instrument reproduces the arm whose episodes P4.5 re-uses",
    )
    gate_b.add_argument("--reused-arm", default=REUSED_ARM, choices=list(METHODS))
    gate_b.add_argument("--baselines", default=None, help="default: <out-dir>/p4_4_baselines.json")
    gate_b.add_argument("--training", default=None, help="default: <out-dir>/p4_4_training.json")
    gate_b.add_argument("--draw", action="append", type=int, default=[])
    gate_b.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)

    report_selection = sub.add_parser(
        "report-selection", help="merge P4.5's arms into docs/data/p4_5_baselines.json"
    )
    report_selection.add_argument(
        "--baselines", default=None, help="default: <out-dir>/p4_4_baselines.json"
    )

    compose = sub.add_parser(
        "compose",
        help="record WHAT the top-return filter selected in the training artifact; reads the "
        "corpus and the committed artifacts only -- no training, no rollouts",
    )
    compose.add_argument("--dataset-dir", action="append", required=True)
    compose.add_argument("--baselines", default=None, help="default: <out-dir>/p4_4_baselines.json")
    compose.add_argument("--permutations", type=int, default=20_000)
    compose.add_argument("--rng-seed", type=int, default=20_260_812)
    return parser


def pin_torch_threads(threads: int) -> int:
    """Pin this process's torch thread count and return what it was before.

    **This is a liveness fix, not a performance fix**, and it is applied in ``main`` only --
    never at import, so importing this module changes nothing about a caller's process.

    Measured here on 2026-08-11, and the reason this function exists: a Gate A run under torch's
    default 16 threads **deadlocked** after roughly 600 rollouts, with all 32 OS threads parked
    in ``futex_do_wait`` and the process consuming no CPU at all.  That is the liveness class
    ``experiments/runner.py`` pins against (P0.3-fix, ``docs/returns/P0.3-fix.md``), and
    ``offline/dt_gate.py`` deliberately does not import that module, so nothing was pinning this
    path.  At one thread the same workload runs to completion at **1.05 s/episode against ~5 s**,
    and -- verified before adopting it -- reproduces P4's committed cells **bit-identically** on
    MADT, MAPPO@1000 and MaxPressure alike, so the pin buys liveness and speed without moving a
    number.  ``runtime_provenance`` records the resulting count in every artifact.
    """
    count = int(threads)
    if count < 1:
        raise ValueError(f"torch thread count must be >= 1, got {threads!r}")
    previous = int(torch.get_num_threads())
    torch.set_num_threads(count)
    return previous


def _checkpoint_map(specs: Iterable[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for spec in specs:
        seed, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"expected SEED=PATH, got {spec!r}")
        out[int(seed)] = path
    return out


def load_baseline_checkpoint(
    gym_env: Any, path: str | Path, declared_gradient_steps: int
) -> dict[str, Any]:
    """Read a baseline checkpoint, refusing one whose step count is not the declared one.

    The mechanical form of "no online model selection" (``PREREGISTRATION.md`` section 6.1),
    mirroring ``dt_gate.load_gate_checkpoint``: a checkpoint saved at a different step -- an
    earlier one that scored better, say -- cannot be evaluated by this path at all.  The check
    runs before any agent is constructed, so a refusal builds nothing.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    recorded = payload.get("provenance", {}).get("gradient_steps")
    if recorded is None:
        raise ValueError(
            f"{path}: the checkpoint records no gradient step count, so it cannot be shown to "
            "be the pre-declared one; refusing rather than reporting an unidentifiable model"
        )
    if int(recorded) != int(declared_gradient_steps):
        raise ValueError(
            f"{path}: checkpoint was saved at {int(recorded)} gradient steps but the declared "
            f"count is {int(declared_gradient_steps)}. PREREGISTRATION.md section 6 forbids "
            "reporting a checkpoint chosen by anything other than the declared budget"
        )
    return payload


def _baseline_factory(method: str, path: str, declared: int, device: str | None) -> Callable[[Any], Any]:
    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        load_baseline_checkpoint(env, path, declared)
        agent_class = IQLAgent if method == "iql" else BCAgent
        agent = agent_class.from_checkpoint(env, path, device=device)
        return lambda _env, info: agent.act(info, explore=False, update_memory=False)

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.materialise_draws import draw_config_path

    args = build_parser().parse_args(argv)
    pin_torch_threads(args.torch_threads)
    settings = env_settings_from_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"--out-dir does not exist: {out_dir}")

    def config_for_draw(draw_id: int) -> Path:
        return draw_config_path(args.scenario_key, draw_id, out_root=args.draws_root)

    if args.command == "gate":
        return _run_gate(args, settings, config_for_draw, work_dir)
    if args.command == "gate-selection":
        return _run_gate_selection(args, settings, config_for_draw, out_dir, work_dir)
    if args.command == "train":
        if args.stream_selector != "top_return" or args.selection_arm is not None:
            return _run_train_selection(args, out_dir)
        return _run_train(args, out_dir)
    if args.command == "evaluate":
        return _run_evaluate(args, settings, config_for_draw, out_dir, work_dir)
    if args.command == "compose":
        return _run_compose(args, out_dir)
    if args.command == "report-selection":
        return _run_report_selection(args, settings, out_dir, work_dir)
    return _run_report(args, settings, out_dir, work_dir)


def _run_compose(args: argparse.Namespace, out_dir: Path) -> int:
    """Patch the training artifact with what the top-return filter selected.

    Everything is validated **before the first byte is written**, and the central validation is
    a refusal rather than a formality: the streams recomputed here must be exactly the streams
    the recorded training run kept.  Patching a committed record is only acceptable because that
    check makes "this composition describes that run" a verified statement.  A refused compose
    leaves the artifact untouched and creates nothing.
    """
    from offline.dt_gate import CONTEXT_LENGTH

    destination = out_dir / "p4_4_training.json"
    training = json.loads(destination.read_text(encoding="utf-8"))
    recorded = training.get("top_return_filter")
    if recorded is None:
        raise ValueError(f"{destination}: no top_return_filter block to describe")

    dataset = build_training_dataset(args.dataset_dir, CONTEXT_LENGTH)
    kept = top_return_streams(dataset, float(recorded["fraction"]))

    recomputed = sorted((k.episode_file, int(k.flow_draw), float(k.total_return)) for k in kept)
    from_artifact = sorted(
        (str(entry["episode_file"]), int(entry["flow_draw"]), float(entry["return"]))
        for entry in recorded["kept_streams"]
    )
    if recomputed != from_artifact:
        raise ValueError(
            f"the streams recomputed from {args.dataset_dir} are not the streams "
            f"{destination} records as kept ({len(recomputed)} against {len(from_artifact)}, "
            "first difference at "
            f"{next((a, b) for a, b in zip(recomputed, from_artifact) if a != b)}); this "
            "composition would describe a different run and is refused"
        )

    baselines = Path(args.baselines) if args.baselines else out_dir / "p4_4_baselines.json"
    episodes = json.loads(baselines.read_text(encoding="utf-8"))["episodes"]
    by_seed: dict[int, list[float]] = {}
    for episode in episodes:
        if episode["arm"] == "mappo1000" and episode["seed"] is not None:
            by_seed.setdefault(int(episode["seed"]), []).append(float(episode["att_horizon"]))
    if not by_seed:
        raise ValueError(f"{baselines}: no mappo1000 episodes, so no per-seed held-out ATT")

    composition = top_return_composition(
        dataset,
        kept,
        {seed: float(np.mean(values)) for seed, values in by_seed.items()},
        permutations=int(args.permutations),
        rng_seed=int(args.rng_seed),
    )
    composition["heldout_att_source"] = str(baselines)
    composition["heldout_att_arm"] = "mappo1000"

    # Validation is complete; only now is anything written.
    training["top_return_filter"] = {**recorded, "composition": composition}
    write_json_atomic(training, destination)
    print(
        f"composition recorded in {destination}\n"
        f"  per-seed kept counts {composition['per_seed_kept_counts']}\n"
        f"  exact p {composition['exact_p_value']:.9f}  "
        f"(permutation {composition['permutation_cross_check']['p_value']:.6f} "
        f"+/- {composition['permutation_cross_check']['monte_carlo_standard_error']:.6f})\n"
        f"  pearson r {composition['pearson_r_training_return_vs_heldout_att']:.4f} "
        f"n={composition['pearson_n']}",
        flush=True,
    )
    return 0


def _episode_key(episode: EpisodeResult) -> tuple[str, int | None, int]:
    return (episode.arm, episode.seed, episode.draw_id)


def _run_gate(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    work_dir: Path,
) -> int:
    """Gate A: re-roll P4's cited arms here and refuse to continue on any difference."""
    from offline.dt_gate import _maxpressure_factory, _mappo_factory, load_gate_checkpoint

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    committed = {
        _episode_key(EpisodeResult(**e)): EpisodeResult(**e)
        for e in [*reference["episodes"], *thresholds["episodes"]]
        if e["arm"] in CITED_ARMS
    }
    if not committed:
        raise ValueError("no cited episodes found in the reference artifacts")

    draws = list(HELD_OUT_DRAWS)
    dt_paths = _checkpoint_map(args.dt_checkpoint)
    mappo_paths = _checkpoint_map(args.mappo_checkpoint)
    work_dir.mkdir(parents=True, exist_ok=True)

    produced: list[EpisodeResult] = []
    mismatches: list[dict[str, Any]] = []

    def compare(results: Sequence[EpisodeResult]) -> None:
        for result in results:
            expected = committed.get(_episode_key(result))
            if expected is None:
                raise ValueError(f"{_episode_key(result)} is not in the committed reference")
            if (
                result.att_horizon != expected.att_horizon
                or result.horizon_vehicle_count != expected.horizon_vehicle_count
                or result.episode_reward != expected.episode_reward
            ):
                mismatches.append(
                    {
                        "key": list(_episode_key(result)),
                        "measured": [result.att_horizon, result.horizon_vehicle_count],
                        "committed": [expected.att_horizon, expected.horizon_vehicle_count],
                    }
                )
                raise ValueError(
                    f"GATE A FAILED at {_episode_key(result)}: measured "
                    f"{result.att_horizon!r} against the committed {expected.att_horizon!r}. "
                    "P4.4 would be measuring on a different instrument than P4; stopping"
                )
        produced.extend(results)

    print(f"Gate A: MaxPressure over {len(draws)} held-out draws", flush=True)
    compare(
        evaluate_arm(
            arm="maxpressure",
            seed=None,
            draw_ids=draws,
            config_for_draw=config_for_draw,
            env_settings=settings,
            scenario_id=args.scenario_id,
            choose_action_factory=_maxpressure_factory,
            engine_seed=args.engine_seed,
        )
    )
    for seed, path in sorted(mappo_paths.items()):
        print(f"Gate A: MAPPO@1000 seed {seed}", flush=True)
        compare(
            evaluate_arm(
                arm="mappo1000",
                seed=seed,
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_mappo_factory(path, args.device),
                engine_seed=args.engine_seed,
            )
        )
    for seed, path in sorted(dt_paths.items()):
        print(f"Gate A: MADT seed {seed}", flush=True)

        def dt_factory(env: Any, path: str = path) -> Callable[[Any, dict[str, Any]], np.ndarray]:
            agent = load_gate_checkpoint(env, path, int(args.dt_steps), device=args.device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        compare(
            evaluate_arm(
                arm="madt",
                seed=seed,
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=dt_factory,
                engine_seed=args.engine_seed,
            )
        )

    assert_campaign_complete(list(committed), produced)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "Gate A: P4's cited arms re-rolled on this instrument before any baseline number "
            "was measured; exact equality per (arm, seed, draw) or the task stops"
        ),
        "status": "PASS",
        "compared": len(produced),
        "mismatches": len(mismatches),
        "arms": sorted({e.arm for e in produced}),
        "engine_seed": int(args.engine_seed),
        "env_settings": {k: v for k, v in settings.items() if k != "compare_with"},
        "episodes": [
            {
                "arm": e.arm,
                "seed": e.seed,
                "draw_id": e.draw_id,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "episode_reward": e.episode_reward,
            }
            for e in produced
        ],
        "runtime": runtime_provenance(),
    }
    write_json_atomic(payload, work_dir / "gate_a.json")
    print(
        f"\nGATE A PASS: {len(produced)} episodes reproduced exactly, 0 mismatches", flush=True
    )
    return 0


def _run_train(args: argparse.Namespace, out_dir: Path) -> int:
    """Train every requested method for every seed, at the declared budget."""
    from agent.utils.utils import Utils
    from offline.dt_gate import CONTEXT_LENGTH

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; known methods are {list(METHODS)}")

    dataset = build_training_dataset(args.dataset_dir, CONTEXT_LENGTH)
    stacked = stack_dataset(dataset)                        # one group, or it raises
    (state_dim, n_actions) = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    streams = stream_returns(dataset)
    kept = top_return_streams(dataset, TOP_RETURN_FRACTION)
    scale = iql_reward_scale([s.total_return for s in streams])

    device = torch.device(args.device) if args.device else Utils.resolve_device(None)
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tier": "mappo1000",
        "dataset_dirs": [str(d) for d in args.dataset_dir],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
    }

    print(
        f"training windows {len(dataset)}  state_dim {state_dim}  n_actions {n_actions}\n"
        f"streams {len(streams)}  top-10% kept {len(kept)} (cut "
        f"{min(k.total_return for k in kept)}, next "
        f"{max((s.total_return for s in streams if s not in kept), default=float('nan'))})\n"
        f"iql reward scale {scale}  device {device}",
        flush=True,
    )

    records: list[TrainRecord] = []
    table: TransitionTable | None = None
    for method in methods:
        for seed in TRAINING_SEEDS:
            path = checkpoints / f"{method}_seed{seed}.pt"
            if method == "iql":
                if table is None:
                    table = build_transitions(dataset, reward_scale=scale)
                    print(f"transitions {len(table)}", flush=True)
                records.append(
                    train_iql(
                        table,
                        state_dim=state_dim,
                        n_actions=n_actions,
                        seed=seed,
                        declared_gradient_steps=int(args.steps),
                        batch_size=IQL_BATCH_TRANSITIONS,
                        device=device,
                        checkpoint_path=path,
                        stats=dataset.stats,
                        scenario_id=scenario_id,
                        provenance=provenance,
                        log_every=args.log_every,
                    )
                )
            else:
                batch = stacked
                if method == "bc_top10":
                    batch = filter_stacked_to_streams(dataset, stacked, kept)
                records.append(
                    train_bc(
                        batch,
                        state_dim=state_dim,
                        n_actions=n_actions,
                        seed=seed,
                        method=method,
                        declared_gradient_steps=int(args.steps),
                        batch_size=BC_BATCH_WINDOWS,
                        device=device,
                        checkpoint_path=path,
                        stats=dataset.stats,
                        scenario_id=scenario_id,
                        provenance=provenance,
                        log_every=args.log_every,
                    )
                )
            last = records[-1]
            print(
                f"  {last.method} seed {last.seed}: {last.seconds:.1f}s  "
                f"final loss {last.losses[-1]:.5f}  plateaued {last.plateaued}  "
                f"digest {last.canonical_digest[:12]}",
                flush=True,
            )

    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": (
            "training record for P4.4's offline baselines; the reported checkpoint is the one "
            "at the declared step count and no raise is available to this task"
        ),
        "declared_gradient_steps": int(args.steps),
        "declared_in": "docs/plans/p4.4.md section 3.1, before the first gradient step",
        "raise_available": False,
        "seeds": list(TRAINING_SEEDS),
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
        "group": [int(state_dim), int(n_actions)],
        "window_count": len(dataset),
        "batch_sizes": {"bc": BC_BATCH_WINDOWS, "bc_top10": BC_BATCH_WINDOWS, "iql": IQL_BATCH_TRANSITIONS},
        "top_return_filter": {
            "fraction": TOP_RETURN_FRACTION,
            "streams_total": len(streams),
            "streams_kept": len(kept),
            "cut_return": min(k.total_return for k in kept),
            "next_return_below_the_cut": max(
                (s.total_return for s in streams if s.key not in {k.key for k in kept}),
                default=float("nan"),
            ),
            "kept_streams": [
                {"episode_file": k.episode_file, "flow_draw": k.flow_draw, "return": k.total_return}
                for k in kept
            ],
        },
        "iql": {
            "tau": IQL_TAU,
            "beta": IQL_BETA,
            "gamma": IQL_GAMMA,
            "polyak": IQL_POLYAK,
            "weight_clip": IQL_WEIGHT_CLIP,
            "reward_scale": scale,
            "sweep": "none",
            "sweep_decision": (
                "declared in docs/plans/p4.4.md section 3.5 before the first gradient step: the "
                "authorised criterion (training-split expectile/TD loss) cannot rank either "
                "parameter it is authorised to rank -- beta appears in no value loss, and "
                "selecting it on the policy loss provably drives beta to 0, at which point IQL's "
                "policy extraction IS plain BC; tau defines the expectile loss, so two tau values "
                "are values of two different objectives"
            ),
            "reported_as": "untuned",
        },
        "runs": [
            {
                "method": r.method,
                "seed": r.seed,
                "gradient_steps": r.gradient_steps,
                "plateaued": r.plateaued,
                "window_means": list(r.window_means),
                "final_loss": r.losses[-1],
                "seconds": r.seconds,
                "checkpoint": r.checkpoint_path,
                "canonical_digest": r.canonical_digest,
                "file_sha256": r.file_sha256,
                "diagnostics": r.diagnostics,
            }
            for r in records
        ],
        "runtime": runtime_provenance(),
    }
    destination = out_dir / "p4_4_training.json"
    if destination.is_file():
        # A chunked run: keep the runs an earlier chunk trained, and refuse to merge two
        # different declarations.  See merge_training_runs.
        payload["runs"] = merge_training_runs(
            json.loads(destination.read_text(encoding="utf-8")), payload
        )
    write_json_atomic(payload, destination)
    return 0


def training_artifact_name(method: str) -> str:
    """Which training artifact declares *method*, derived from the arm rather than passed.

    A P4.5 arm cannot be evaluated against P4.4's declaration and vice versa: the budget check
    and the leakage check in ``_run_evaluate`` are only meaningful against the declaration that
    actually produced the checkpoint.
    """
    return (
        "p4_5_selection.json" if method in SELECTION_ARMS else "p4_4_training.json"
    )


#: The cells Gate B re-rolls, declared in ``docs/plans/p4.5.md`` section 5 before it ran, so the
#: sample cannot be chosen after seeing a disagreement.
GATE_B_DRAWS: tuple[int, ...] = (1000, 1025, 1050, 1075, 1099)


def _run_gate_selection(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
    work_dir: Path,
) -> int:
    """Gate B: the instrument that produced the re-used episodes must be this one.

    ``BRIEF_13`` section 4 re-uses ``bc_top10``'s 500 committed episodes rather than re-rolling
    them, which is right -- re-rolling a settled number is a second measurement of it.  This is
    what makes that sound: **weight identity by canonical digest** (section 10.3's substitution;
    a file hash depends on the filename, so it proves transport and not weights) and **path
    identity** by re-rolling the declared cells and requiring exact equality.

    A refusal is BLOCKED: nothing is written and no P4.5 number may be reported.
    """
    started = time.time()
    training_path = Path(args.training) if args.training else out_dir / "p4_4_training.json"
    baselines_path = Path(args.baselines) if args.baselines else out_dir / "p4_4_baselines.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    committed_all = json.loads(baselines_path.read_text(encoding="utf-8"))["episodes"]

    arm = str(args.reused_arm)
    runs = [r for r in training["runs"] if r["method"] == arm]
    if not runs:
        raise ValueError(f"{training_path}: records no runs for the re-used arm {arm!r}")

    # -------- weight identity, before a single rollout ---------------------------------
    weights: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: int(r["seed"])):
        path = Path(run["checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        digest = canonical_state_dict_digest(payload["model"])
        file_hash = _sha256_file(path)
        if digest != run["canonical_digest"]:
            raise ValueError(
                f"{path}: canonical digest {digest} does not match the {run['canonical_digest']} "
                f"recorded in {training_path}; these are not the weights that produced the "
                "episodes P4.5 re-uses"
            )
        weights.append(
            {
                "seed": int(run["seed"]),
                "checkpoint": str(path),
                "canonical_digest": digest,
                "canonical_digest_matches": True,
                "file_sha256": file_hash,
                "file_sha256_matches": file_hash == run.get("file_sha256"),
                "file_sha256_role": (
                    "transport integrity only: a file hash depends on the filename (DEFERRED 29)"
                ),
            }
        )

    # -------- path identity: re-roll the declared cells ---------------------------------
    draws = sorted(set(args.draw)) if args.draw else list(GATE_B_DRAWS)
    leaked = sorted(set(draws) - set(HELD_OUT_DRAWS))
    if leaked:
        raise ValueError(f"gate draws {leaked} are not in the registered held-out pool")
    committed = [
        EpisodeResult(**e)
        for e in committed_all
        if e["arm"] == arm and int(e["draw_id"]) in set(draws)
    ]
    rerolled: list[EpisodeResult] = []
    for run in sorted(runs, key=lambda r: int(r["seed"])):
        print(f"gate B: {arm} seed {run['seed']} over {len(draws)} declared draws", flush=True)
        rerolled.extend(
            evaluate_arm(
                arm=arm,
                seed=int(run["seed"]),
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_baseline_factory(
                    arm, run["checkpoint"], int(args.steps), args.device
                ),
                engine_seed=args.engine_seed,
            )
        )
    record = assert_reused_arm_reproduces(committed, rerolled)

    # Validation is complete; only now is anything written.
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": SELECTION_ARTIFACT_FORMAT_VERSION,
        "role": (
            "Gate B: this session's instrument reproduces the committed episodes P4.5 re-uses "
            "for the bc_top10 arm; the re-rolled values are discarded and never reported"
        ),
        "reused_arm": arm,
        "declared_draws": draws,
        "declared_in": "docs/plans/p4.5.md section 5, before it ran",
        "weights": weights,
        "sources": {"training": str(training_path), "baselines": str(baselines_path)},
        "seconds": time.time() - started,
        "thread_regime": thread_regime(),
        "runtime": runtime_provenance(),
        **record,
    }
    write_json_atomic(payload, work_dir / "gate_b.json")
    print(
        f"GATE B PASS: {record['compared']} episodes reproduce exactly, "
        f"{len(weights)} checkpoints match by canonical digest",
        flush=True,
    )
    return 0


def _run_train_selection(args: argparse.Namespace, out_dir: Path) -> int:
    """Train ONE declared arm across the five training seeds, and record what it saw.

    The dataset is built over **every** dataset directory and only then filtered, so the
    normalisation statistics are the full training split's for every arm; rebuilding over a
    subset would refit them and make the arms incomparable in a way no later test could see.

    The subset is drawn from the TRAINING SEED's own generator, so the five-seed spread averages
    over five subsets rather than resting on one lucky draw -- which is why this arm's CI covers
    subset variance as well as training variance.
    """
    from agent.utils.utils import Utils
    from offline.dt_gate import CONTEXT_LENGTH

    if args.selection_arm is None:
        raise ValueError(
            "--stream-selector on the selection path needs --selection-arm; the declaration in "
            f"SELECTION_ARMS is what fixes the design ({sorted(SELECTION_ARMS)})"
        )
    if str(args.methods) != ",".join(METHODS):
        raise ValueError(
            f"--methods {args.methods!r} has no meaning on the selection path: one invocation "
            "trains exactly the arm named by --selection-arm"
        )
    spec = arm_spec_for_flags(
        args.selection_arm,
        selector=args.stream_selector,
        behaviour_seeds=args.selector_seed,
        count=args.subset_count,
    )

    dataset = build_training_dataset(args.dataset_dir, CONTEXT_LENGTH)
    stacked = stack_dataset(dataset)
    (state_dim, n_actions) = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    stats_json = json.dumps(dataset.stats.to_json_obj(), sort_keys=True)
    normalisation_digest = hashlib.sha256(stats_json.encode("utf-8")).hexdigest()

    device = torch.device(args.device) if args.device else Utils.resolve_device(None)
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tier": "mappo1000",
        "dataset_dirs": [str(d) for d in args.dataset_dir],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
        "selection_arm": spec.arm,
        "selector": spec.selector,
        "behaviour_seeds": list(spec.behaviour_seeds),
    }
    print(
        f"arm {spec.arm}: selector {spec.selector}, behaviour seeds "
        f"{list(spec.behaviour_seeds) or 'ALL'}, count {spec.count}\n"
        f"training windows {len(dataset)}  streams {len(stream_returns(dataset))}  "
        f"device {device}  normalisation {normalisation_digest[:12]}",
        flush=True,
    )

    records: list[TrainRecord] = []
    per_training_seed: dict[str, Any] = {}
    declared_count: int | None = None
    for seed in TRAINING_SEEDS:
        rng = np.random.default_rng(int(seed))
        selected = select_arm_streams(
            dataset, spec, dataset_dirs=args.dataset_dir, rng=rng
        )
        if declared_count is None:
            declared_count = len(selected)
        elif len(selected) != declared_count:
            raise ValueError(
                f"{spec.arm}: seed {seed} selected {len(selected)} streams against "
                f"{declared_count} for an earlier seed; the arm's size must not vary by seed"
            )
        batch = filter_stacked_to_streams(dataset, stacked, selected)
        rows = int(batch["state"].shape[0])
        composition: dict[str, int] = {}
        for stream in selected:
            key = str(_behaviour_seed(stream))
            composition[key] = composition.get(key, 0) + 1

        record = train_bc(
            batch,
            state_dim=state_dim,
            n_actions=n_actions,
            seed=seed,
            method=spec.arm,
            declared_gradient_steps=int(args.steps),
            batch_size=BC_BATCH_WINDOWS,
            device=device,
            checkpoint_path=checkpoints / f"{spec.arm}_seed{seed}.pt",
            stats=dataset.stats,
            scenario_id=scenario_id,
            provenance=provenance,
            log_every=args.log_every,
        )
        records.append(record)
        per_training_seed[str(int(seed))] = {
            "rng_seed": int(seed),
            "rng": "numpy.random.default_rng(training_seed)",
            "training_rows": rows,
            "per_behaviour_seed_composition": composition,
            "streams": [
                {
                    "dataset_dir": s.dataset_dir,
                    "episode_file": s.episode_file,
                    "ix_id": s.ix_id,
                    "flow_draw": s.flow_draw,
                    "total_return": s.total_return,
                    "behaviour_seed": _behaviour_seed(s),
                }
                for s in selected
            ],
        }
        print(
            f"  {spec.arm} seed {seed}: {record.seconds:.1f}s  rows {rows}  "
            f"composition {composition}  final loss {record.losses[-1]:.5f}  "
            f"digest {record.canonical_digest[:12]}",
            flush=True,
        )

    arms_block = {
        spec.arm: {
            "selector": spec.selector,
            "selector_parameters": {
                "behaviour_seeds": list(spec.behaviour_seeds),
                "count": spec.count,
            },
            "role": spec.role,
            "declared_count": int(declared_count or 0),
            "pool": (
                f"streams of behaviour seeds {list(spec.behaviour_seeds)}"
                if spec.behaviour_seeds
                else "every stream of the training split"
            ),
            "normalisation_digest": normalisation_digest,
            "per_training_seed": per_training_seed,
        }
    }
    payload: dict[str, Any] = {
        "format_version": SELECTION_ARTIFACT_FORMAT_VERSION,
        "role": (
            "P4.5: which streams each arm trained on, and which behaviour checkpoints produced "
            "them. The arms differ in that and in nothing else"
        ),
        "declared_gradient_steps": int(args.steps),
        "declared_in": "docs/plans/p4.5.md section 3, before the first gradient step",
        "raise_available": False,
        "seeds": list(TRAINING_SEEDS),
        "training_draw_ids": list(dataset.stats.draw_ids),
        "held_out_draws": list(HELD_OUT_DRAWS),
        "scenario_id": scenario_id,
        "group": [int(state_dim), int(n_actions)],
        "window_count": len(dataset),
        "streams_total": len(stream_returns(dataset)),
        "batch_size": BC_BATCH_WINDOWS,
        "normalisation": {
            "digest": normalisation_digest,
            "source": (
                "the FULL training split over every dataset directory; every arm filters that "
                "same stack rather than rebuilding a dataset over its own directories"
            ),
        },
        "arms": arms_block,
        "runs": [
            {
                "method": r.method,
                "seed": r.seed,
                "gradient_steps": r.gradient_steps,
                "plateaued": r.plateaued,
                "window_means": list(r.window_means),
                "final_loss": r.losses[-1],
                "seconds": r.seconds,
                "thread_regime": thread_regime(),
                "checkpoint": r.checkpoint_path,
                "canonical_digest": r.canonical_digest,
                "file_sha256": r.file_sha256,
                "diagnostics": r.diagnostics,
            }
            for r in records
        ],
        "runtime": runtime_provenance(),
    }

    destination = out_dir / "p4_5_selection.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        payload["runs"] = merge_training_runs(existing, payload)
        payload["arms"] = {**existing.get("arms", {}), **arms_block}

    payload["matched_arms"] = sorted(
        arm for arm in payload["arms"] if SELECTION_ARMS[arm].count is not None
    )
    # Everything is validated before the first byte is written: a refused arm leaves the
    # artifact of the arms already trained exactly as it was.
    assert_selection_design(payload)
    write_json_atomic(payload, destination)
    return 0


def _run_report_selection(
    args: argparse.Namespace, settings: dict[str, Any], out_dir: Path, work_dir: Path
) -> int:
    """Merge P4.5's four arms and the re-used one into the reported artifact."""
    gate = json.loads((work_dir / "gate_b.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise ValueError(
            f"Gate B did not pass ({gate.get('status')!r}); bc_top10's episodes may not be "
            "re-used until this session's instrument is shown to reproduce them"
        )
    selection = json.loads((out_dir / "p4_5_selection.json").read_text(encoding="utf-8"))
    missing_arms = sorted(set(SELECTION_ARMS) - set(selection["arms"]))
    if missing_arms:
        raise ValueError(
            f"the selection artifact is missing declared arm(s) {missing_arms}; the design is "
            "four new arms and a partial one answers a different question"
        )

    episodes: list[EpisodeResult] = []
    for arm in sorted(SELECTION_ARMS):
        payload = json.loads((work_dir / f"eval_{arm}.json").read_text(encoding="utf-8"))
        episodes.extend(EpisodeResult(**e) for e in payload["episodes"])

    baselines_path = Path(args.baselines) if args.baselines else out_dir / "p4_4_baselines.json"
    merged = json.loads(baselines_path.read_text(encoding="utf-8"))
    reused = [EpisodeResult(**e) for e in merged["episodes"] if e["arm"] == REUSED_ARM]
    if not reused:
        raise ValueError(f"{baselines_path}: carries no {REUSED_ARM!r} episodes to re-use")

    # The expected run set is the DECLARED design and never the episodes being checked.
    requested = [
        (arm, seed, draw)
        for arm in (*sorted(SELECTION_ARMS), REUSED_ARM)
        for seed in TRAINING_SEEDS
        for draw in HELD_OUT_DRAWS
    ]
    assert_campaign_complete(requested, [*episodes, *reused])

    artifact = selection_artifact(
        episodes=[*episodes, *reused],
        selection=selection,
        gate_b={
            key: gate[key]
            for key in ("status", "compared", "mismatches", "declared_draws", "weights")
            if key in gate
        },
        env_settings=settings,
        engine_seed=int(args.engine_seed),
    )
    write_json_atomic(artifact, out_dir / "p4_5_baselines.json")

    print("\narm              att_horizon   +/- CI    veh     n", flush=True)
    for arm in SELECTION_ARM_ORDER:
        cell = artifact["cells"][arm]
        print(
            f"  {arm:14s} {cell['att_horizon_mean']:10.4f} {cell['att_horizon_ci95']:8.4f} "
            f"{cell['horizon_vehicle_count_mean']:7.2f} {cell['n_episodes']:5d}",
            flush=True,
        )
    print("", flush=True)
    for name, entry in artifact["comparisons"].items():
        print(
            f"  {name:36s} diff {entry['mean_difference']:+.4f} "
            f"CI [{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}] "
            f"width {entry['ci95_width']:.4f}  p {entry['wilcoxon']['p_value']:.3e}  "
            f"r {entry['rank_biserial']:+.3f}  -> {entry['delta_verdict']}",
            flush=True,
        )
    print("", flush=True)
    for name, entry in artifact["registered_predictions"].items():
        if isinstance(entry, dict) and "held" in entry:
            print(f"  {name}: held={entry['held']}", flush=True)
    bound = artifact["null_bound"]
    print(
        f"  null bound X = {bound['x']:.4f} on {bound['contrast']}, against a "
        f"{bound['behaviour_spread_att']:.4f} ATT behaviour spread; power contrast width "
        f"{bound['power_contrast_ci95_width']:.4f}",
        flush=True,
    )
    return 0


def _run_evaluate(
    args: argparse.Namespace,
    settings: dict[str, Any],
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
    work_dir: Path,
) -> int:
    """Evaluate ONE method over the held-out pool, so no single job runs long."""
    training = json.loads(
        (out_dir / training_artifact_name(args.method)).read_text(encoding="utf-8")
    )
    if int(training["declared_gradient_steps"]) != int(args.steps):
        raise ValueError(
            f"the training artifact was produced under a declared budget of "
            f"{training['declared_gradient_steps']} steps but this evaluation was asked for "
            f"{int(args.steps)}; PREREGISTRATION.md section 6 fixes the budget before training"
        )
    leaked = sorted(set(HELD_OUT_DRAWS) & set(training["training_draw_ids"]))
    if leaked:
        raise ValueError(
            f"evaluation draws {leaked[:5]} are also training draws; the held-out pool is not "
            "held out and no number from this run may be reported"
        )

    runs = [r for r in training["runs"] if r["method"] == args.method]
    if not runs:
        raise ValueError(f"the training artifact records no runs for method {args.method!r}")

    draws = list(HELD_OUT_DRAWS)
    requested = [(args.method, int(r["seed"]), d) for r in runs for d in draws]
    produced: list[EpisodeResult] = []
    for run in runs:
        print(f"{args.method} seed {run['seed']} over {len(draws)} draws", flush=True)
        produced.extend(
            evaluate_arm(
                arm=args.method,
                seed=int(run["seed"]),
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=_baseline_factory(
                    args.method, run["checkpoint"], int(args.steps), args.device
                ),
                engine_seed=args.engine_seed,
            )
        )
    assert_campaign_complete(requested, produced)

    work_dir.mkdir(parents=True, exist_ok=True)
    cell = _cell(produced)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "method": args.method,
            "declared_gradient_steps": int(args.steps),
            "engine_seed": int(args.engine_seed),
            "cell": cell,
            "episodes": [
                {
                    "arm": e.arm,
                    "seed": e.seed,
                    "draw_id": e.draw_id,
                    "att_horizon": e.att_horizon,
                    "horizon_vehicle_count": e.horizon_vehicle_count,
                    "episode_reward": e.episode_reward,
                }
                for e in produced
            ],
            "runtime": runtime_provenance(),
        },
        work_dir / f"eval_{args.method}.json",
    )
    print(
        f"  {args.method}: att_horizon {cell['att_horizon_mean']:.4f} "
        f"+/- {cell['att_horizon_ci95']:.4f}  vehicle_count "
        f"{cell['horizon_vehicle_count_mean']:.2f}  n={cell['n_episodes']}",
        flush=True,
    )
    return 0


def _run_report(
    args: argparse.Namespace, settings: dict[str, Any], out_dir: Path, work_dir: Path
) -> int:
    """Merge Gate A and the per-method runs into the one committed artifact."""
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    gate = json.loads((work_dir / "gate_a.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise ValueError(
            f"Gate A did not pass ({gate.get('status')!r}); no baseline number may be reported "
            "against P4's arms until the instrument reproduces them"
        )
    training = json.loads((out_dir / "p4_4_training.json").read_text(encoding="utf-8"))

    episodes = [EpisodeResult(**e) for e in gate["episodes"]]
    for method in methods:
        payload = json.loads((work_dir / f"eval_{method}.json").read_text(encoding="utf-8"))
        episodes.extend(EpisodeResult(**e) for e in payload["episodes"])

    # The expected run set is derived from the DECLARED design -- the cited arms, the registered
    # training seeds and the registered held-out pool -- and never from the episodes being
    # checked.  Deriving it from them would make the completeness check a tautology, which is
    # what it was in the first draft of this function.
    #
    # Corrected 2026-08-12 (review finding F5, DEFERRED 32): the baseline arms' seed set was
    # derived from ``training["runs"]`` -- from DATA -- while ``madt``/``mappo1000`` correctly
    # used ``TRAINING_SEEDS``.  A method trained on three seeds therefore passed, because both
    # sides of the comparison shrank together; a test now reproduces that and the loop is uniform.
    # This changes no reported number: every method in the committed artifact was trained on all
    # five declared seeds, which a second test asserts from the artifact itself.
    # P4.6's two flags are read with getattr so the generalisation is additive for EVERY caller,
    # including a hand-built Namespace: a caller that predates them keeps P4.4's behaviour exactly.
    behaviour_reference = str(getattr(args, "behaviour_reference", "mappo1000"))
    emit_verdicts = not bool(getattr(args, "no_verdicts", False))

    requested: list[tuple[str, int | None, int]] = [
        ("maxpressure", None, draw) for draw in HELD_OUT_DRAWS
    ]
    for arm in ("madt", behaviour_reference, *methods):
        requested += [(arm, seed, draw) for seed in TRAINING_SEEDS for draw in HELD_OUT_DRAWS]
    assert_campaign_complete(requested, episodes)
    artifact = baselines_artifact(
        episodes=episodes,
        training={
            key: training[key]
            for key in (
                "declared_gradient_steps",
                "declared_in",
                "raise_available",
                "seeds",
                "training_draw_ids",
                "top_return_filter",
                "iql",
                "runs",
            )
            if key in training
        },
        gate_a={key: gate[key] for key in ("status", "compared", "mismatches", "arms")},
        env_settings=settings,
        engine_seed=int(args.engine_seed),
        behaviour_reference=behaviour_reference,
        emit_verdicts=emit_verdicts,
    )
    write_json_atomic(artifact, out_dir / "p4_4_baselines.json")

    print("\narm            att_horizon   +/- CI    veh     n", flush=True)
    for arm, cell in sorted(artifact["cells"].items()):
        print(
            f"  {arm:12s} {cell['att_horizon_mean']:10.4f} {cell['att_horizon_ci95']:8.4f} "
            f"{cell['horizon_vehicle_count_mean']:7.2f} {cell['n_episodes']:5d}",
            flush=True,
        )
    print("", flush=True)
    for name, entry in sorted(artifact["comparisons"].items()):
        # The verdict is absent under --no-verdicts, which is a mode and not a missing value.
        verdict = f"  -> {entry['verdict']}" if "verdict" in entry else ""
        print(
            f"  {name:20s} diff {entry['mean_difference']:+.4f} "
            f"CI [{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}] "
            f"width {entry['ci95_width']:.4f}  p {entry['wilcoxon']['p_value']:.3e}  "
            f"r {entry['rank_biserial']:+.3f}  recovered {entry['recovered_fraction']:+.4f}"
            f"{verdict}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
