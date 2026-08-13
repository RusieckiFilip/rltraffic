"""P4.6 -- the method x tier grid: four offline methods across the C1 data-quality ladder.

**Artifact formats written by this module** (every one carries its version in the payload):

* ``p4.6-grid-declaration/1.0`` -- ``docs/data/p4_6_declaration.json``: which streams each tier
  trains on, the declared per-tier prompt target and scale, the statistics digests and the sizes.
  Written **before** the first gradient step.
* ``p4.6-selection-diagnostics/1.0`` -- ``docs/data/p4_6_selection_diagnostics.json``: prediction
  P3's two leakage-free checks per tier.  No rollouts, no gradient steps.
* ``p4.6-grid-training/1.0`` -- ``docs/data/p4_6_training.json``: one record per
  ``(tier, method, seed)``.
* ``p4.6-grid/1.0`` -- ``docs/data/p4_6_grid.json``: the cells, every per-episode record, the paired
  comparisons and the scored predictions.

**Conventions this module is bound by, stated because a reader must not have to infer them.**

*Alignment* is contract C6's, unchanged: observations carry ``T+1`` rows, decisions and outcomes
``T``, and a stream's ``total_return`` is ``sum(local_reward)``.  Nothing here re-derives it; the
windows come from :class:`offline.dataset.TrajectoryWindowDataset` and the returns from
:func:`offline.offline_baselines.stream_returns`.

*Size matching* (``BRIEF_17`` section 7.1): **every tier trains on exactly 200 streams**, which on
``cf_hz1x1`` is 200 episodes because each episode carries exactly one intersection.  ``random`` holds
400 (two per draw) and is subsampled **one per draw** by a declared RNG, so the tiers match on demand
coverage as well as on count.

*Statistics* (``BRIEF_17`` section 7.3, fixed in ``docs/plans/p4.6.md`` section 3.3): every per-tier
statistic -- state normalisation, the RTG summary and hence ``rtg_scale``, the DT's naive
``target_rtg`` and IQL's ``reward_scale`` -- is fitted on that tier's **full training split**, while
the training **set** is the size-matched subsample.  The two coincide for every tier except
``random``.  Statistics are never shared across tiers: that would be a leak between arms.

*The prompt* (``BRIEF_17`` section 5): one declared naive target per tier,
``max(training-split episode return)``, and **no sweep** -- P4.3 measured 0.9026 ATT of movement over
a 13,000-wide grid, so a per-tier sweep would chase an effect smaller than the differences this grid
exists to measure.

*No equivalence verdicts* (``BRIEF_17`` section 4): this module reports paired mean differences,
their 95 % CIs, the CI **width** and the rank-biserial effect size, and **issues no verdict**.  It
deliberately imports no verdict symbol from :mod:`offline.offline_baselines`, and
:func:`assert_no_verdicts` re-checks the emitted payload.

*Naming* (contract C9): the Decision Transformer arm's key here is **``dt``**.  It is the same arm
that P4.4's merged ``docs/data/p4_4_baselines.json`` calls ``madt``; that key is not renamed, and the
alias is recorded in :data:`REUSED_ARM_KEYS`.  The string "MADT" is not a model name and is not
written in prose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import (
    CONTEXT_LENGTH,
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    CellStats,
    EpisodeResult,
    build_training_dataset,
    env_settings_from_manifest,
    evaluate_arm,
    mean_ci95,
    runtime_provenance,
    stack_dataset,
    write_json_atomic,
)
from offline.offline_baselines import (
    BC_BATCH_WINDOWS,
    GATE_B_DRAWS,
    IQL_BATCH_TRANSITIONS,
    TOP_RETURN_FRACTION,
    PairedComparison,
    StreamReturn,
    build_transitions,
    filter_stacked_to_streams,
    iql_reward_scale,
    paired_comparison,
    pin_torch_threads,
    stream_returns,
    thread_regime,
)

# ----------------------------------------------------------------------
# Formats and the declared design
# ----------------------------------------------------------------------

DECLARATION_FORMAT_VERSION = "p4.6-grid-declaration/1.0"
DIAGNOSTICS_FORMAT_VERSION = "p4.6-selection-diagnostics/1.0"
TRAINING_FORMAT_VERSION = "p4.6-grid-training/1.0"
GRID_FORMAT_VERSION = "p4.6-grid/1.0"

#: Declared in ``docs/plans/p4.6.md`` section 3.4 before the first gradient step.  No raise is
#: available to this task, exactly as in P4.4.
DECLARED_GRADIENT_STEPS = 40_000

#: Every tier trains on this many streams (``BRIEF_17`` section 7.1).
TRAINING_STREAM_COUNT = 200

#: The RNG that subsamples ``random`` from 400 streams to 200, one per draw.  Declared in
#: ``docs/plans/p4.6.md`` section 3.2; the repo's date convention (P4.4 declared ``20_260_812``).
RANDOM_SUBSAMPLE_RNG_SEED = 20_260_813

#: Mixture tiers draw with ``default_rng(MIXTURE_RNG_BASE + round(100 * fraction))``.
MIXTURE_RNG_BASE = 20_260_813

#: The four methods, in a fixed order so every table is written the same way.
METHODS: tuple[str, ...] = ("bc", "bc_top10", "iql", "dt")

#: The tier whose four cells are re-used rather than re-run, under the gate of section 7.
REUSED_TIER = "mappo1000"

#: This task's arm key -> the key the merged P4.4 artifact uses (contract C9 rule 4).
REUSED_ARM_KEYS: Mapping[str, str] = {
    "bc": "bc",
    "bc_top10": "bc_top10",
    "iql": "iql",
    "dt": "madt",
}

#: Behaviour ATT of each tier's own data, measured over draws 1-200 and read from
#: ``docs/data/att_ladder_v11.json``.  **This is the only ordering used for tiers**: never the tier
#: name, never the training budget (``BRIEF_17`` section 1).
#: Copied digit for digit from that artifact and asserted against it by a test, rather than read at
#: run time: the ordering is a declaration of this task, and a declaration that silently follows a
#: file it does not control is not a declaration.
BEHAVIOUR_ATT: Mapping[str, float] = {
    "mappo1000": 105.46135581970215,
    "mappo500": 107.49564304351807,
    "maxpressure": 176.50006713867188,
    "fixedtime": 262.08867095947267,
    "random": 422.5187595367432,
}


@dataclass(frozen=True)
class TierSpec:
    """One column of the grid: where its data is, what it trains on and what the DT is asked for.

    ``target_rtg`` and ``rtg_scale`` are **declared** here and re-derived from the corpus by
    :func:`assert_declaration_matches_corpus` before any training; a disagreement raises.
    """

    tier: str
    dirs: tuple[str, ...]
    phase: int
    target_rtg: float
    rtg_scale: float
    stream_count: int
    subsample: str
    components: tuple[str, ...] = ()


#: The five single-controller tiers of phase 1 and the three mixture tiers of phase 2.
TIERS: Mapping[str, TierSpec] = {
    "mappo1000": TierSpec(
        tier="mappo1000",
        dirs=tuple(f"cf_hz1x1__mappo1000__seed{seed}" for seed in TRAINING_SEEDS),
        phase=1,
        target_rtg=-5762.0,
        rtg_scale=9991.0,
        stream_count=200,
        subsample="none",
    ),
    "mappo500": TierSpec(
        tier="mappo500",
        dirs=tuple(f"cf_hz1x1__mappo500__seed{seed}" for seed in TRAINING_SEEDS),
        phase=1,
        target_rtg=-6362.0,
        rtg_scale=11043.0,
        stream_count=200,
        subsample="none",
    ),
    "maxpressure": TierSpec(
        tier="maxpressure",
        dirs=("cf_hz1x1__maxpressure",),
        phase=1,
        target_rtg=-13112.0,
        rtg_scale=24115.0,
        stream_count=200,
        subsample="none",
    ),
    "fixedtime": TierSpec(
        tier="fixedtime",
        dirs=("cf_hz1x1__fixedtime",),
        phase=1,
        target_rtg=-29707.0,
        rtg_scale=33225.0,
        stream_count=200,
        subsample="none",
    ),
    "random": TierSpec(
        tier="random",
        dirs=("cf_hz1x1__random",),
        phase=1,
        target_rtg=-38369.0,
        rtg_scale=40294.0,
        stream_count=200,
        subsample="one_per_draw",
    ),
    "mix33": TierSpec(
        tier="mix33",
        dirs=(
            *(f"cf_hz1x1__mappo1000__seed{seed}" for seed in TRAINING_SEEDS),
            "cf_hz1x1__random",
        ),
        phase=2,
        target_rtg=-5762.0,
        rtg_scale=40294.0,
        stream_count=200,
        subsample="mixture",
        components=("mappo1000", "random"),
    ),
    "mix50": TierSpec(
        tier="mix50",
        dirs=(
            *(f"cf_hz1x1__mappo1000__seed{seed}" for seed in TRAINING_SEEDS),
            "cf_hz1x1__random",
        ),
        phase=2,
        target_rtg=-5762.0,
        rtg_scale=40294.0,
        stream_count=200,
        subsample="mixture",
        components=("mappo1000", "random"),
    ),
    "mix67": TierSpec(
        tier="mix67",
        dirs=(
            *(f"cf_hz1x1__mappo1000__seed{seed}" for seed in TRAINING_SEEDS),
            "cf_hz1x1__random",
        ),
        phase=2,
        target_rtg=-5762.0,
        rtg_scale=40294.0,
        stream_count=200,
        subsample="mixture",
        components=("mappo1000", "random"),
    ),
}

#: Expert fraction of each mixture tier (OffLight Fig. 8: 33 / 50 / 67 % expert, remainder random).
MIXTURE_EXPERT_FRACTION: Mapping[str, float] = {"mix33": 0.33, "mix50": 0.50, "mix67": 0.67}

#: Phase-1 tiers in measured-quality order, best behaviour data first.
PHASE1_TIER_ORDER: tuple[str, ...] = ("mappo1000", "mappo500", "maxpressure", "fixedtime", "random")

#: Phase-2 tiers, in expert-fraction order.
MIXTURE_TIER_ORDER: tuple[str, ...] = ("mix33", "mix50", "mix67")

#: P3 check B: the draws whose MaxPressure ATT is lowest form the "easiest" set, and the top decile
#: has this many streams, so the two sets are the same size and the null is exactly hypergeometric.
EASIEST_DRAW_COUNT = 20

#: P3's declared significance level (``docs/plans/p4.6.md`` section 4.3).
P3_ALPHA = 0.05

#: The tier that defines draw difficulty for P3 check B.
DIFFICULTY_TIER = "maxpressure"


# ----------------------------------------------------------------------
# Declaration: which streams a tier trains on, and what the DT is asked for
# ----------------------------------------------------------------------


def tier_spec(tier: str) -> TierSpec:
    """The declared spec of *tier*, or a refusal naming the tiers this task declares."""
    raise NotImplementedError


def tier_dirs(spec: TierSpec, corpus_root: str | Path) -> tuple[Path, ...]:
    """Absolute dataset directories of *spec*, in declaration order."""
    raise NotImplementedError


def tier_dataset(
    spec: TierSpec, corpus_root: str | Path, *, context_length: int = CONTEXT_LENGTH
) -> TrajectoryWindowDataset:
    """The training-split window dataset over *spec*'s directories, statistics fitted there."""
    raise NotImplementedError


def stratified_one_per_draw(
    streams: Sequence[StreamReturn], *, rng: np.random.Generator
) -> tuple[StreamReturn, ...]:
    """Exactly one stream per ``flow_draw``, chosen by *rng* from that draw's candidates."""
    raise NotImplementedError


def training_streams(
    spec: TierSpec,
    dataset: TrajectoryWindowDataset,
    *,
    component_streams: Mapping[str, Sequence[StreamReturn]] | None = None,
) -> tuple[StreamReturn, ...]:
    """The size-matched training set of *spec*: ``spec.stream_count`` streams, deterministic."""
    raise NotImplementedError


def mixture_training_streams(
    spec: TierSpec,
    expert: Sequence[StreamReturn],
    random_pool: Sequence[StreamReturn],
) -> tuple[StreamReturn, ...]:
    """A mixture tier's streams: ``round(count * fraction)`` expert, the rest random."""
    raise NotImplementedError


def top_decile_streams(streams: Sequence[StreamReturn]) -> tuple[StreamReturn, ...]:
    """%BC's filter applied to a tier's own training set: the top ``TOP_RETURN_FRACTION``."""
    raise NotImplementedError


def recomputed_target_and_scale(streams: Sequence[StreamReturn]) -> tuple[float, float]:
    """``(max return, max|return|)`` over *streams* -- the naive prompt rule, from the data."""
    raise NotImplementedError


def assert_declaration_matches_corpus(
    spec: TierSpec, dataset: TrajectoryWindowDataset
) -> dict[str, Any]:
    """Refuse to train unless the declared target and scale are what the corpus says they are."""
    raise NotImplementedError


def statistics_digest(dataset: TrajectoryWindowDataset) -> str:
    """sha256 over the tier's serialised normalisation statistics."""
    raise NotImplementedError


def assert_equal_training_size(declaration: Mapping[str, Any]) -> None:
    """Every tier in the declaration trains on the same number of streams and windows."""
    raise NotImplementedError


def env_settings_for_tiers(
    tiers: Sequence[str], corpus_root: str | Path
) -> dict[str, Any]:
    """The one evaluation settings dict, asserted identical across every tier's manifest."""
    raise NotImplementedError


def declaration_artifact(
    corpus_root: str | Path,
    tiers: Sequence[str],
    *,
    context_length: int = CONTEXT_LENGTH,
) -> dict[str, Any]:
    """The full declaration: selections, targets, scales, digests, sizes, order and predictions."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# Prediction P3: the two leakage-free checks, per tier
# ----------------------------------------------------------------------


def draw_arrivals(manifest_path: str | Path) -> dict[int, int]:
    """Vehicles per draw, rebuilt from the collection's own randomizer and hash-verified."""
    raise NotImplementedError


def difficulty_by_draw(corpus_root: str | Path) -> dict[int, float]:
    """Per-draw ``att_horizon`` of the MaxPressure tier -- the declared difficulty ranking."""
    raise NotImplementedError


def hypergeometric_upper_tail(
    population: int, successes: int, draws: int, observed: int
) -> float:
    """Exact ``P(X >= observed)`` for a hypergeometric draw, in integer arithmetic."""
    raise NotImplementedError


def volume_check(
    kept_draws: Sequence[int], other_draws: Sequence[int], arrivals: Mapping[int, int]
) -> dict[str, Any]:
    """P3 check A: kept-versus-discarded mean arrivals, with a normal-approximation interval."""
    raise NotImplementedError


def difficulty_check(
    kept_draws: Sequence[int],
    difficulty: Mapping[int, float],
    *,
    easiest_count: int = EASIEST_DRAW_COUNT,
) -> dict[str, Any]:
    """P3 check B: overlap of the kept draws with the easiest draws, and its exact p-value."""
    raise NotImplementedError


def selection_diagnostics_artifact(
    corpus_root: str | Path,
    tiers: Sequence[str],
    *,
    context_length: int = CONTEXT_LENGTH,
) -> dict[str, Any]:
    """P3's two checks for every tier, whichever way they come out."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# Gate G: the re-used mappo1000 column
# ----------------------------------------------------------------------


def canonical_digests(checkpoints: Mapping[str, str]) -> dict[str, str]:
    """Canonical state-dict digest of every checkpoint file, keyed as given."""
    raise NotImplementedError


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes -- an identity of the container, not of the weights."""
    raise NotImplementedError


def assert_reused_checkpoint_identity(
    training_artifact: Mapping[str, Any],
    gate_artifact_p4: Mapping[str, Any],
    *,
    baselines_root: str | Path,
    dt_root: str | Path,
) -> dict[str, Any]:
    """The 15 baseline digests and the 5 DT file hashes must be the committed ones."""
    raise NotImplementedError


def assert_reused_cells_reproduce(
    committed: Sequence[Mapping[str, Any]],
    rerolled: Sequence[EpisodeResult],
) -> dict[str, Any]:
    """Re-rolled cells must equal the committed ones exactly; any difference is BLOCKED."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------


def arm_key(method: str, tier: str) -> str:
    """``"<method>@<tier>"`` -- the arm identity carried by every episode record."""
    raise NotImplementedError


def split_arm_key(arm: str) -> tuple[str, str]:
    """Inverse of :func:`arm_key`."""
    raise NotImplementedError


def merge_training_records(
    existing: Mapping[str, Any], fresh: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Fresh records plus the existing ones this chunk did not train, keyed by tier/method/seed."""
    raise NotImplementedError


def assert_cell_complete(
    method: str,
    tier: str,
    seeds: Sequence[int],
    draws: Sequence[int],
    produced: Sequence[EpisodeResult],
) -> None:
    """A cell must contain exactly the runs the declaration asks for."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def cell_stats(episodes: Sequence[EpisodeResult]) -> dict[str, Any]:
    """One reported cell: n, ATT mean/std/CI95, the companion vehicle count, draws and seeds."""
    raise NotImplementedError


def grid_comparisons(
    episodes_by_arm: Mapping[str, Sequence[EpisodeResult]],
) -> list[PairedComparison]:
    """Every within-tier method pair and every within-method tier pair, paired on shared draws."""
    raise NotImplementedError


def score_p1(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """P1: BC's rank per tier, the primary rule of plan section 4.1 and its continuous companion."""
    raise NotImplementedError


def score_p2(
    cells: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[PairedComparison],
) -> dict[str, Any]:
    """P2: %BC's advantage over BC per tier, scored fully only when the mixtures are present."""
    raise NotImplementedError


def score_p3(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """P3: the demand-signature rule of plan section 4.3, applied per tier."""
    raise NotImplementedError


def kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall's tau-b, reported beside P1's rank rule and carrying no threshold."""
    raise NotImplementedError


def assert_no_verdicts(payload: Any) -> None:
    """Refuse to emit an equivalence verdict anywhere in the artifact (``BRIEF_17`` section 4)."""
    raise NotImplementedError


def grid_artifact(
    declaration: Mapping[str, Any],
    training: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    gate: Mapping[str, Any],
    episodes_by_arm: Mapping[str, Sequence[EpisodeResult]],
) -> dict[str, Any]:
    """The reported artifact: cells, episodes, comparisons, predictions and provenance."""
    raise NotImplementedError


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``declare``, ``diagnostics``, ``gate``, ``train``, ``evaluate``, ``report``."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
