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

The four filenames above are the ``--artifact-prefix p4_6`` default (P4.7, coordinator RULING 3 of
2026-08-15).  **The format versions are NOT renamed with the prefix**: they name the schema, which
P4.7 re-uses unchanged, and renaming a version string because a caller changed would make two
identical layouts look like two formats.

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
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from offline.dataset import TrajectoryWindowDataset
from offline.dt_gate import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    EpisodeResult,
    build_training_dataset,
    env_settings_from_manifest,
    evaluate_arm,
    load_gate_checkpoint,
    mean_ci95,
    runtime_provenance,
    stack_dataset,
    write_json_atomic,
)
from offline.offline_baselines import (
    BC_BATCH_WINDOWS,
    GATE_B_DRAWS,
    IQL_BATCH_TRANSITIONS,
    IQL_BETA,
    IQL_GAMMA,
    IQL_POLYAK,
    IQL_TAU,
    IQL_WEIGHT_CLIP,
    TOP_RETURN_FRACTION,
    PairedComparison,
    StreamReturn,
    TransitionTable,
    _baseline_factory,
    build_transitions,
    filter_stacked_to_streams,
    iql_reward_scale,
    paired_comparison,
    pin_torch_threads,
    stream_returns,
    thread_regime,
    train_bc,
    train_iql,
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

#: The behaviour policy of a tier is an arm too, but it is NOT one of the four methods: it never
#: enters the method grid's pairs.  ``BRIEF_17`` section 11, finding A3.
BEHAVIOUR_METHOD = "behaviour"

#: The tier whose four cells are re-used rather than re-run, under the gate of section 7.
REUSED_TIER = "mappo1000"

#: This task's arm key -> the key the merged P4.4 artifact uses (contract C9 rule 4).
REUSED_ARM_KEYS: Mapping[str, str] = {
    "bc": "bc",
    "bc_top10": "bc_top10",
    "iql": "iql",
    "dt": "madt",
}

#: ⚠️ **A TIER LABEL, MEASURED ON THE TRAINING DRAWS (1-200), AND NEVER A COMPARATOR.**
#: Each tier's own corpus ATT, read from ``docs/data/att_ladder_v11.json``.  It orders the tiers --
#: the only ordering this task uses, never the tier name and never the training budget
#: (``BRIEF_17`` section 1) -- and it must **never** be placed beside a held-out cell:
#: ``PREREGISTRATION`` A5 makes a comparison across different draw sets **void**, and the held-out
#: behaviour cells in ``behaviour_cells`` are the only lawful reference for "did the method beat the
#: policy that produced its data?".
#:
#: **Renamed from ``BEHAVIOUR_ATT`` on 2026-08-14 (``BRIEF_18`` finding F2), because the old name
#: invited exactly that void reading and the report's own summary table had taken it**: it printed
#: this training-draw value beside four held-out means, with a **maximum substitution error of
#: 6.3652 ATT** and **one of twenty statements reversing sign** (`bc@random` read +5.4356 "worse"
#: against the true −0.9296 "better").  The rename is necessary and not sufficient -- the offending
#: site was a *use* -- so every consumption site is enumerated and classified in
#: ``docs/returns/P4.6.md`` section 5a.
#:
#: Copied digit for digit from the ladder artifact and asserted against it by a test, rather than
#: read at run time: the ordering is a declaration of this task, and a declaration that silently
#: follows a file it does not control is not a declaration.
TIER_LABEL_ATT_TRAINING_DRAWS: Mapping[str, float] = {
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


_MAPPO1000_DIRS = tuple(f"cf_hz1x1__mappo1000__seed{seed}" for seed in TRAINING_SEEDS)
_MIXTURE_DIRS = (*_MAPPO1000_DIRS, "cf_hz1x1__random")


#: The five single-controller tiers of phase 1 and the three mixture tiers of phase 2.
TIERS: Mapping[str, TierSpec] = {
    "mappo1000": TierSpec(
        tier="mappo1000",
        dirs=_MAPPO1000_DIRS,
        phase=1,
        target_rtg=-5762.0,
        rtg_scale=9991.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="none",
    ),
    "mappo500": TierSpec(
        tier="mappo500",
        dirs=tuple(f"cf_hz1x1__mappo500__seed{seed}" for seed in TRAINING_SEEDS),
        phase=1,
        target_rtg=-6362.0,
        rtg_scale=11043.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="none",
    ),
    "maxpressure": TierSpec(
        tier="maxpressure",
        dirs=("cf_hz1x1__maxpressure",),
        phase=1,
        target_rtg=-13112.0,
        rtg_scale=24115.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="none",
    ),
    "fixedtime": TierSpec(
        tier="fixedtime",
        dirs=("cf_hz1x1__fixedtime",),
        phase=1,
        target_rtg=-29707.0,
        rtg_scale=33225.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="none",
    ),
    "random": TierSpec(
        tier="random",
        dirs=("cf_hz1x1__random",),
        phase=1,
        target_rtg=-38369.0,
        rtg_scale=40294.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="one_per_draw",
    ),
    # ⚠️ **THE THREE MIXTURE TARGETS WERE CORRECTED IN P4.7, BEFORE ANY MIXTURE NUMBER EXISTED**
    # (``docs/plans/p4.7.md`` section 3.5, coordinator RULING 1 of 2026-08-15).  Until then all three
    # declared ``target_rtg=-5762.0, rtg_scale=40294.0`` -- the maximum and the largest absolute
    # return of the six directories' **600-episode union**, from ``docs/plans/p4.6.md`` section 8.
    # But :func:`assert_declaration_matches_corpus` computes both over the **composed 200-stream
    # training set** (``BRIEF_17`` section 11 finding A4, which supersedes that plan's section 3.3),
    # and **neither the union's best expert stream nor its worst random stream is drawn into any of
    # the three mixtures** -- measured, not assumed.  So the old declaration was not merely loose: it
    # was **unrunnable**, and ``train --tier mix33`` would have raised before its first gradient step.
    # It went unnoticed because the path had never been executed; that is the general lesson recorded
    # in ``PROJECT_PLAN`` section 7 (``a91ddda``) and it is why this comment is here rather than in a
    # packet.  **The rule, not the numbers, is the declaration:** ``target_rtg = max(total_return)``
    # and ``rtg_scale = max|total_return|`` over the composed training set, which is exactly what
    # :func:`recomputed_target_and_scale` returns.  The three pairs below are that rule evaluated on
    # the deterministic composition, re-derived from the raw ``.npz`` by an independent route in
    # ``tests/test_mixture_tiers.py``.
    "mix33": TierSpec(
        tier="mix33",
        dirs=_MIXTURE_DIRS,
        phase=2,
        target_rtg=-5994.0,
        rtg_scale=40223.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="mixture",
        components=("mappo1000", "random"),
    ),
    "mix50": TierSpec(
        tier="mix50",
        dirs=_MIXTURE_DIRS,
        phase=2,
        target_rtg=-5959.0,
        rtg_scale=40223.0,
        stream_count=TRAINING_STREAM_COUNT,
        subsample="mixture",
        components=("mappo1000", "random"),
    ),
    "mix67": TierSpec(
        tier="mix67",
        dirs=_MIXTURE_DIRS,
        phase=2,
        target_rtg=-5959.0,
        rtg_scale=40123.0,
        stream_count=TRAINING_STREAM_COUNT,
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

#: z for the normal-approximation interval of P3's volume check, declared with the check.
VOLUME_CHECK_Z = 1.96

#: The policy that COLLECTED each tier, and where its held-out cell comes from
#: (``BRIEF_17`` section 11, finding A3).  ``committed`` cells are re-used from a merged artifact;
#: ``evaluated_here`` cells are rolled by this task through the same ``evaluate_arm``, using the
#: collection factory itself rather than a fresh implementation of the same idea (amendment A2's
#: error class).  **Mixture tiers have two behaviour policies and therefore no single reference;
#: they are deliberately absent.**
BEHAVIOUR_REFERENCE_BY_TIER: Mapping[str, Mapping[str, str]] = {
    "mappo1000": {
        "arm": "mappo1000",
        "source": "committed",
        "artifact": "docs/data/p4_heldout_thresholds.json",
    },
    "mappo500": {
        "arm": "mappo500",
        "source": "committed",
        "artifact": "docs/data/p4_heldout_thresholds.json",
    },
    "maxpressure": {
        "arm": "maxpressure",
        "source": "committed",
        "artifact": "docs/data/p4_heldout_thresholds.json",
    },
    "fixedtime": {
        "arm": "fixedtime",
        "source": "evaluated_here",
        "artifact": "offline/policies/fixed_time.py::make_fixedtime, k from the tier's manifest",
    },
    "random": {
        "arm": "random",
        "source": "evaluated_here",
        "artifact": "offline/collect.py::_make_random, one fresh default_rng(seed) per draw",
    },
}

#: Strings that would constitute an equivalence verdict.  ``BRIEF_17`` section 4 forbids all of them.
_VERDICT_VALUES = frozenset(
    {
        "matches",
        "within_delta",
        "dt_genuinely_better",
        "baseline_genuinely_better",
        "inconclusive_at_this_power",
        "left_genuinely_better",
        "right_genuinely_better",
        "equivalent",
    }
)


# ----------------------------------------------------------------------
# The declaration
# ----------------------------------------------------------------------


#: Default filename prefix of the four artifacts this module writes.  ``--artifact-prefix`` overrides
#: it, and **the default reproduces every pre-P4.7 invocation byte for byte**.
DEFAULT_ARTIFACT_PREFIX = "p4_6"


def artifact_path(out_dir: str | Path, name: str, prefix: str = DEFAULT_ARTIFACT_PREFIX) -> Path:
    """``<out_dir>/<prefix>_<name>.json`` -- the one place a reported filename is decided.

    Added in P4.7 under coordinator RULING 3 of 2026-08-15, **additively**: phase 2 re-uses this
    module's ``train`` and ``evaluate`` unchanged but must not write into P4.6's merged records, and
    the alternative -- isolating them in a ``docs/data/p4_7/`` directory under their P4.6 names --
    *protects the artifact and does nothing for the reader*.  A ``p4_6_training.json`` holding P4.7's
    runs is the misnaming hazard that defeated an expert reader on ``BEHAVIOUR_ATT`` two days
    earlier, rebuilt one directory deeper.  The prefix is therefore part of the name, not part of the
    path.
    """
    token = str(prefix).strip()
    if not token:
        raise ValueError("artifact prefix must be a non-empty string, e.g. 'p4_6' or 'p4_7'")
    return Path(out_dir) / f"{token}_{name}.json"


def _artifact_prefix(args: argparse.Namespace) -> str:
    """The prefix this invocation writes under, defaulting for a namespace built by hand.

    ``argparse`` always supplies the attribute, so this default fires only for a caller that builds
    a :class:`argparse.Namespace` itself -- which P4.6's ``test_the_training_cli_path_runs_end_to_end
    _on_a_fixture`` does, and which is a legitimate way to test a subcommand without a parser.
    **Reading it through ``getattr`` is what lets P4.7 add the flag without editing a single
    pre-existing test**, and the fallback is the documented default, so such a caller gets exactly
    the filenames it got before this argument existed.
    """
    return str(getattr(args, "artifact_prefix", DEFAULT_ARTIFACT_PREFIX))


def tier_spec(tier: str) -> TierSpec:
    """The declared spec of *tier*, or a refusal naming the tiers this task declares."""
    key = str(tier)
    if key not in TIERS:
        raise ValueError(f"unknown tier {key!r}; this task declares {sorted(TIERS)}")
    return TIERS[key]


def _as_spec(tier: TierSpec | str) -> TierSpec:
    return tier if isinstance(tier, TierSpec) else tier_spec(tier)


def tier_dirs(spec: TierSpec, corpus_root: str | Path) -> tuple[Path, ...]:
    """Absolute dataset directories of *spec*, in declaration order."""
    root = Path(corpus_root)
    return tuple(root / name for name in spec.dirs)


def tier_dataset(
    spec: TierSpec, corpus_root: str | Path, *, context_length: int = CONTEXT_LENGTH
) -> TrajectoryWindowDataset:
    """The training-split window dataset over *spec*'s directories, statistics fitted there."""
    return build_training_dataset(tier_dirs(spec, corpus_root), int(context_length))


def stratified_one_per_draw(
    streams: Sequence[StreamReturn], *, rng: np.random.Generator
) -> tuple[StreamReturn, ...]:
    """Exactly one stream per ``flow_draw``, chosen by *rng* from that draw's candidates.

    Draws are visited in ascending order and each draw's candidates are sorted by
    ``(dataset_dir, episode_file, ix_id)``, so the selection depends on the RNG and on nothing
    else -- not on load order, not on the filesystem.
    """
    by_draw: dict[int, list[StreamReturn]] = {}
    for entry in streams:
        by_draw.setdefault(int(entry.flow_draw), []).append(entry)
    chosen: list[StreamReturn] = []
    for draw in sorted(by_draw):
        candidates = sorted(by_draw[draw], key=lambda s: s.key)
        index = int(rng.integers(0, len(candidates)))
        chosen.append(candidates[index])
    return tuple(chosen)


def training_streams(
    spec: TierSpec,
    dataset: TrajectoryWindowDataset,
    *,
    component_streams: Mapping[str, Sequence[StreamReturn]] | None = None,
) -> tuple[StreamReturn, ...]:
    """The size-matched training set of *spec*: ``spec.stream_count`` streams, deterministic."""
    if spec.subsample == "mixture":
        if component_streams is None:
            raise ValueError(
                f"{spec.tier}: a mixture tier needs its components' training sets; pass "
                "component_streams={'mappo1000': [...], 'random': [...]}"
            )
        missing = [name for name in spec.components if name not in component_streams]
        if missing:
            raise ValueError(f"{spec.tier}: component_streams is missing {missing}")
        selected = mixture_training_streams(
            spec,
            component_streams[spec.components[0]],
            component_streams[spec.components[1]],
        )
    else:
        pool = stream_returns(dataset)
        if spec.subsample == "one_per_draw":
            selected = stratified_one_per_draw(
                pool, rng=np.random.default_rng(RANDOM_SUBSAMPLE_RNG_SEED)
            )
        elif spec.subsample == "none":
            selected = tuple(sorted(pool, key=lambda s: s.key))
        else:
            raise ValueError(
                f"{spec.tier}: unknown subsample rule {spec.subsample!r}; the declared rules are "
                "'none', 'one_per_draw' and 'mixture'"
            )

    if len(selected) != int(spec.stream_count):
        raise ValueError(
            f"{spec.tier}: the spec declares {int(spec.stream_count)} training streams but the "
            f"selection holds {len(selected)}; every tier trains on the same number of streams "
            "(BRIEF_17 section 7.1) and a tier that cannot supply them may not enter the grid"
        )
    return selected


def mixture_training_streams(
    spec: TierSpec,
    expert: Sequence[StreamReturn],
    random_pool: Sequence[StreamReturn],
) -> tuple[StreamReturn, ...]:
    """A mixture tier's streams: ``round(count * fraction)`` expert, the rest random."""
    if spec.tier not in MIXTURE_EXPERT_FRACTION:
        raise ValueError(f"{spec.tier!r} is not a mixture tier; {sorted(MIXTURE_EXPERT_FRACTION)}")
    fraction = float(MIXTURE_EXPERT_FRACTION[spec.tier])
    total = int(spec.stream_count)
    n_expert = int(round(total * fraction))
    n_random = total - n_expert
    if n_expert > len(expert) or n_random > len(random_pool):
        raise ValueError(
            f"{spec.tier}: needs {n_expert} expert and {n_random} random streams but the "
            f"components hold {len(expert)} and {len(random_pool)}"
        )
    rng = np.random.default_rng(MIXTURE_RNG_BASE + int(round(100 * fraction)))
    expert_sorted = sorted(expert, key=lambda s: s.key)
    random_sorted = sorted(random_pool, key=lambda s: s.key)
    expert_index = sorted(int(i) for i in rng.choice(len(expert_sorted), size=n_expert, replace=False))
    random_index = sorted(int(i) for i in rng.choice(len(random_sorted), size=n_random, replace=False))
    return (
        *(expert_sorted[i] for i in expert_index),
        *(random_sorted[i] for i in random_index),
    )


def top_decile_streams(
    streams: Sequence[StreamReturn], fraction: float = TOP_RETURN_FRACTION
) -> tuple[StreamReturn, ...]:
    """%BC's filter applied to a tier's own training set: the top *fraction* by return.

    The rule is :func:`offline.offline_baselines.top_return_streams`', transcribed to operate on a
    stream sequence rather than on a whole dataset -- ``ceil(fraction * n)``, never fewer than one,
    ordered by descending return with ties broken by ``(dataset_dir, episode_file, ix_id)``.  The
    tier's training set is the size-matched subsample, never the full split, so the filter can only
    select data the arm actually trains on.
    """
    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")
    if not streams:
        raise ValueError("this tier has no streams to filter")
    keep = max(1, math.ceil(value * len(streams)))
    ordered = sorted(
        streams, key=lambda s: (-s.total_return, s.dataset_dir, s.episode_file, s.ix_id)
    )
    return tuple(ordered[:keep])


def recomputed_target_and_scale(streams: Sequence[StreamReturn]) -> tuple[float, float]:
    """``(max return, max|return|)`` over *streams* -- the naive prompt rule, from the data."""
    if not streams:
        raise ValueError("no streams: the naive target is undefined")
    returns = [float(s.total_return) for s in streams]
    return max(returns), max(abs(min(returns)), abs(max(returns)))


def assert_declaration_matches_corpus(
    spec: TierSpec, selected: Sequence[StreamReturn]
) -> dict[str, Any]:
    """Refuse to train unless the declared target and scale are what the corpus says they are.

    Computed over the **training set** -- the streams the models actually see -- so the prompt is
    in-support by construction (``BRIEF_17`` section 11, finding A4; ``docs/plans/p4.6.md`` section
    15.4, which supersedes that plan's section 3.3).  The rule itself is P4's, recorded in
    ``docs/data/p4_dt_config.json``: *target_rtg = max episode return in the training split;
    rtg_scale = max|rtg|*.
    """
    pool = list(selected)
    target, scale = recomputed_target_and_scale(pool)
    if float(spec.target_rtg) != target:
        raise ValueError(
            f"{spec.tier}: the spec declares target_rtg {float(spec.target_rtg)} but this tier's "
            f"training split has maximum return {target}; the prompt is the split maximum "
            "(BRIEF_17 section 5) and a declaration that disagrees with the data may not train"
        )
    if float(spec.rtg_scale) != scale:
        raise ValueError(
            f"{spec.tier}: the spec declares rtg_scale {float(spec.rtg_scale)} but this tier's "
            f"training split has max|return| {scale}"
        )
    return {
        "tier": spec.tier,
        "target_rtg": target,
        "rtg_scale": scale,
        "training_streams": len(pool),
        "training_return_min": min(float(s.total_return) for s in pool),
        "training_return_max": max(float(s.total_return) for s in pool),
        "rule": "target_rtg = max episode return in the TRAINING SET; rtg_scale = max|rtg|",
        "computed_over": "the size-matched training set (BRIEF_17 section 11, finding A4)",
    }


def statistics_digest(dataset: TrajectoryWindowDataset) -> str:
    """sha256 over the tier's serialised normalisation statistics."""
    payload = json.dumps(dataset.stats.to_json_obj(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_equal_training_size(declaration: Mapping[str, Any]) -> None:
    """Every tier in the declaration trains on the same number of streams and windows."""
    tiers = declaration.get("tiers", {})
    if not tiers:
        raise ValueError("the declaration carries no tiers, so there is no size to compare")
    for field in ("training_streams", "training_windows", "top_decile_streams"):
        values = {tier: int(entry[field]) for tier, entry in tiers.items()}
        distinct = sorted(set(values.values()))
        if len(distinct) > 1:
            raise ValueError(
                f"training-set sizes differ across tiers on {field}: {values}; P4.5's confound "
                "returns the moment two arms differ in how much data they saw "
                "(BRIEF_17 section 7.1)"
            )


def env_settings_for_tiers(
    tiers: Sequence[TierSpec | str], corpus_root: str | Path
) -> dict[str, Any]:
    """The one evaluation settings dict, asserted identical across every tier's manifest."""
    seen: dict[str, list[str]] = {}
    settings: dict[str, Any] | None = None
    for tier in tiers:
        spec = _as_spec(tier)
        for directory in tier_dirs(spec, corpus_root):
            candidate = env_settings_from_manifest(directory / "manifest.json")
            key = json.dumps(candidate, sort_keys=True, default=str)
            seen.setdefault(key, []).append(str(directory))
            if settings is None:
                settings = candidate
    if settings is None:
        raise ValueError("no tiers were given, so there are no env settings to agree on")
    if len(seen) > 1:
        summary = {sorted(paths)[0]: len(paths) for _, paths in seen.items()}
        raise ValueError(
            f"these tiers do not share one set of env settings ({len(seen)} distinct sets, "
            f"first directory of each: {summary}); arms evaluated under different env settings "
            "are not comparable and may not enter one grid"
        )
    return settings


def _stream_record(entry: StreamReturn) -> dict[str, Any]:
    return {
        "dataset_dir": entry.dataset_dir,
        "episode_file": entry.episode_file,
        "ix_id": entry.ix_id,
        "flow_draw": int(entry.flow_draw),
        "total_return": float(entry.total_return),
    }


def _behaviour_seed_of(dataset_dir: str) -> str | None:
    """The behaviour seed a MAPPO tier directory encodes, or ``None`` for a policy without one."""
    name = Path(dataset_dir).name
    marker = "__seed"
    if marker not in name:
        return None
    return name.rsplit(marker, 1)[1]


def stream_records_with_digests(
    streams: Sequence[StreamReturn],
) -> list[dict[str, Any]]:
    """Stream records carrying each episode's ``episode_sha256`` from its own manifest.

    ``BRIEF_17`` section 11, finding A1: the selection is only reproducible from the artifact if
    the artifact identifies the episodes by content and not only by filename.
    """
    manifests: dict[str, dict[str, str]] = {}
    out: list[dict[str, Any]] = []
    for entry in streams:
        directory = entry.dataset_dir
        if directory not in manifests:
            payload = json.loads(
                (Path(directory) / "manifest.json").read_text(encoding="utf-8")
            )
            manifests[directory] = {
                str(item["filename"]): str(item["episode_sha256"])
                for item in payload["episodes"]
            }
        digest = manifests[directory].get(entry.episode_file)
        if digest is None:
            raise ValueError(
                f"{directory}: the manifest has no entry for {entry.episode_file!r}, so the "
                "selected episode cannot be identified by content"
            )
        out.append({**_stream_record(entry), "episode_sha256": digest})
    return out


def kept_composition(streams: Sequence[StreamReturn]) -> dict[str, Any]:
    """What a selection is made of, on BOTH axes (``BRIEF_17`` section 11, finding A5).

    By dataset directory -- on a mixture that is expert against random -- and as a behaviour-seed
    histogram in P4.4's F3 form.  Without the second axis, "the filter selected the expert
    fraction" is confounded with the checkpoint selection P4.5 measured.
    """
    by_dir: dict[str, int] = {}
    by_seed: dict[str, int] = {}
    without_seed = 0
    for entry in streams:
        name = Path(entry.dataset_dir).name
        by_dir[name] = by_dir.get(name, 0) + 1
        seed = _behaviour_seed_of(entry.dataset_dir)
        if seed is None:
            without_seed += 1
        else:
            by_seed[seed] = by_seed.get(seed, 0) + 1
    return {
        "by_dataset_dir": by_dir,
        "by_behaviour_seed": dict(sorted(by_seed.items())),
        "without_a_behaviour_seed": without_seed,
        "total": len(streams),
    }


def declaration_artifact(
    corpus_root: str | Path,
    tiers: Sequence[str],
    *,
    context_length: int = CONTEXT_LENGTH,
) -> dict[str, Any]:
    """The full declaration: selections, targets, scales, digests, sizes, order and predictions."""
    specs = [_as_spec(tier) for tier in tiers]
    settings = env_settings_for_tiers(specs, corpus_root)

    component_sets: dict[str, tuple[StreamReturn, ...]] = {}
    entries: dict[str, Any] = {}
    for spec in specs:
        dataset = tier_dataset(spec, corpus_root, context_length=context_length)
        pool = stream_returns(dataset)
        selected = training_streams(spec, dataset, component_streams=component_sets or None)
        declared = assert_declaration_matches_corpus(spec, selected)
        component_sets[spec.tier] = selected
        kept = top_decile_streams(selected)
        windows = len(selected) * int(dataset.episode_records[0].episode_length)
        entries[spec.tier] = {
            "phase": spec.phase,
            "dataset_dirs": [str(path) for path in tier_dirs(spec, corpus_root)],
            "subsample": spec.subsample,
            "subsample_rng_seed": (
                RANDOM_SUBSAMPLE_RNG_SEED if spec.subsample == "one_per_draw" else None
            ),
            "tier_label_att_training_draws": TIER_LABEL_ATT_TRAINING_DRAWS.get(spec.tier),
            "behaviour_reference": BEHAVIOUR_REFERENCE_BY_TIER.get(spec.tier),
            "target_rtg": declared["target_rtg"],
            "rtg_scale": declared["rtg_scale"],
            "target_rule": declared["rule"],
            "split_streams": len(pool),
            "split_return_min": min(float(s.total_return) for s in pool),
            "split_return_max": max(float(s.total_return) for s in pool),
            "training_return_min": declared["training_return_min"],
            "training_return_max": declared["training_return_max"],
            "statistics_digest": statistics_digest(dataset),
            "statistics_fitted_on": (
                "state normalisation is fitted by the loader over every episode of the tier's "
                "training split; the target, the RTG scale and IQL's reward scale are computed "
                "over the training SET (docs/plans/p4.6.md section 15.4)"
            ),
            "training_streams": len(selected),
            "training_windows": windows,
            "training_draws": sorted({int(s.flow_draw) for s in selected}),
            "top_decile_streams": len(kept),
            "top_decile_fraction": TOP_RETURN_FRACTION,
            "iql_reward_scale": iql_reward_scale([s.total_return for s in selected]),
            "training_composition": kept_composition(selected),
            "top_decile_composition": kept_composition(kept),
            "streams": stream_records_with_digests(selected),
            "top_decile": stream_records_with_digests(kept),
        }

    payload = {
        "format_version": DECLARATION_FORMAT_VERSION,
        "role": (
            "P4.6: what each tier trains on and what its Decision Transformer is prompted with, "
            "declared before the first gradient step"
        ),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "declared_in": "docs/plans/p4.6.md sections 3 to 5",
        "raise_available": False,
        "methods": list(METHODS),
        "seeds": list(TRAINING_SEEDS),
        "held_out_draws": list(HELD_OUT_DRAWS),
        "context_length": int(context_length),
        "batch_sizes": {
            "bc": BC_BATCH_WINDOWS,
            "bc_top10": BC_BATCH_WINDOWS,
            "iql": IQL_BATCH_TRANSITIONS,
            "dt": BATCH_SIZE,
        },
        "iql": {
            "tau": IQL_TAU,
            "beta": IQL_BETA,
            "gamma": IQL_GAMMA,
            "polyak": IQL_POLYAK,
            "weight_clip": IQL_WEIGHT_CLIP,
            "sweep": "none",
            "sweep_decision": (
                "P4.4's analytic refusal, inherited unchanged: beta appears in no value loss, so "
                "the authorised criterion cannot rank it, and selecting it on the policy loss "
                "drives beta to 0, at which point IQL's extraction IS behaviour cloning"
            ),
            "reported_as": "untuned",
        },
        "tier_order_by_tier_label_att": [
            tier for tier in (*PHASE1_TIER_ORDER, *MIXTURE_TIER_ORDER) if tier in entries
        ],
        "reused_tier": REUSED_TIER,
        "reused_arm_keys": dict(REUSED_ARM_KEYS),
        "behaviour_reference_by_tier": {
            tier: dict(entry) for tier, entry in BEHAVIOUR_REFERENCE_BY_TIER.items()
        },
        "prompt_rule": (
            "each tier's DT is conditioned on that tier's naive target, max(episode return over "
            "the tier's TRAINING SET); no sweep (BRIEF_17 section 5 and section 11 finding A4)"
        ),
        "reporting_rule": (
            "paired mean differences, 95 % CIs, CI widths and rank-biserial effect sizes; NO "
            "equivalence verdict anywhere (BRIEF_17 section 4)"
        ),
        "env_settings": {k: v for k, v in settings.items() if k != "compare_with"},
        "tiers": entries,
        "runtime": runtime_provenance(),
    }
    assert_equal_training_size(payload)
    assert_no_verdicts(payload)
    return payload


# ----------------------------------------------------------------------
# Prediction P3: the two leakage-free checks, per tier
# ----------------------------------------------------------------------


def draw_arrivals(manifest_path: str | Path) -> dict[int, int]:
    """Vehicles per draw, rebuilt from the collection's own randomizer and hash-verified.

    Nothing is written: the draw is rendered in memory and digested.  **Every draw's digest must
    equal the one the manifest recorded at collection time**, or this refuses -- an arrivals figure
    attributed to the wrong demand would be worse than no figure at all.
    """
    from offline.flow_randomizer import FlowRandomizer

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metadata = payload.get("run_metadata", {})
    for key in ("flow_source_path", "flow_randomizer_params", "flow_draw_sha256", "flow_draw_ids"):
        if key not in metadata:
            raise ValueError(
                f"{manifest_path}: run_metadata has no {key!r}, so the demand of each draw cannot "
                "be rebuilt and verified"
            )
    params = metadata["flow_randomizer_params"]
    randomizer = FlowRandomizer(
        metadata["flow_source_path"],
        base_seed=int(params["base_seed"]),
        jitter_sigma_s=float(params["jitter_sigma_s"]),
        thin_p=float(params["thin_p"]),
        volume_scale=float(params["volume_scale"]),
    )
    recorded_source = metadata.get("flow_source_sha256")
    if recorded_source is not None and randomizer.source_sha256 != recorded_source:
        raise ValueError(
            f"{manifest_path}: the source flow file hashes {randomizer.source_sha256} but the "
            f"manifest recorded {recorded_source}; this is not the demand the corpus was collected "
            "from"
        )

    arrivals: dict[int, int] = {}
    hashes = metadata["flow_draw_sha256"]
    for draw in sorted(int(d) for d in metadata["flow_draw_ids"]):
        entries, provenance = randomizer.draw(draw)
        digest = hashlib.sha256(randomizer.render_cityflow_bytes(entries)).hexdigest()
        recorded = hashes.get(str(draw))
        if recorded is None:
            raise ValueError(
                f"{manifest_path}: draw {draw} has no recorded flow hash, so its arrivals cannot "
                "be attributed to the demand the corpus used"
            )
        if digest != recorded:
            raise ValueError(
                f"{manifest_path}: draw {draw} rebuilds to {digest} against the recorded flow hash "
                f"{recorded}; the arrivals this would report belong to a different demand"
            )
        arrivals[draw] = int(provenance.n_vehicles)
    return arrivals


def difficulty_by_draw(corpus_root: str | Path) -> dict[int, float]:
    """Per-draw ``att_horizon`` of the MaxPressure tier -- the declared difficulty ranking."""
    from offline.trajectory_logger import load_episode

    spec = tier_spec(DIFFICULTY_TIER)
    difficulty: dict[int, float] = {}
    for directory in tier_dirs(spec, corpus_root):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["episodes"]:
            draw = int(entry["flow_draw"])
            if draw in difficulty:
                raise ValueError(
                    f"{directory}: draw {draw} appears more than once in the difficulty tier; a "
                    "draw with two MaxPressure episodes has no single difficulty"
                )
            episode = load_episode(directory / entry["filename"])
            att = getattr(episode, "att_per_step", None)
            if att is None:
                raise ValueError(
                    f"{directory / str(entry['filename'])}: no att_per_step, so this corpus is "
                    "format v1.0 and carries no horizon ATT to rank draws by"
                )
            difficulty[draw] = float(np.asarray(att)[-1])
    return difficulty


def hypergeometric_upper_tail(
    population: int, successes: int, draws: int, observed: int
) -> float:
    """Exact ``P(X >= observed)`` for a hypergeometric draw, in integer arithmetic.

    No RNG and no new dependency: the tail is a ratio of binomial coefficients, computed with
    ``math.comb`` and divided once at the end.
    """
    n_pop, n_succ, n_draw, x = int(population), int(successes), int(draws), int(observed)
    if n_pop < 0 or not 0 <= n_succ <= n_pop or not 0 <= n_draw <= n_pop:
        raise ValueError(
            f"hypergeometric arguments are impossible: population {n_pop}, successes {n_succ}, "
            f"draws {n_draw}"
        )
    if x <= 0:
        return 1.0
    total = math.comb(n_pop, n_draw)
    numerator = sum(
        math.comb(n_succ, i) * math.comb(n_pop - n_succ, n_draw - i)
        for i in range(x, min(n_succ, n_draw) + 1)
    )
    return numerator / total


def volume_check(
    kept_draws: Sequence[int], other_draws: Sequence[int], arrivals: Mapping[int, int]
) -> dict[str, Any]:
    """P3 check A: kept-versus-discarded mean arrivals, with a normal-approximation interval."""
    kept = [int(d) for d in kept_draws]
    others = [int(d) for d in other_draws]
    if len(kept) < 2 or len(others) < 2:
        raise ValueError(
            f"the volume check needs at least two draws on each side, got {len(kept)} and "
            f"{len(others)}"
        )
    missing = sorted({d for d in (*kept, *others) if d not in arrivals})
    if missing:
        raise ValueError(f"no arrivals recorded for draws {missing[:5]}")

    left = np.asarray([float(arrivals[d]) for d in kept], dtype=np.float64)
    right = np.asarray([float(arrivals[d]) for d in others], dtype=np.float64)
    difference = float(left.mean() - right.mean())
    standard_error = float(
        math.sqrt(left.var(ddof=1) / left.size + right.var(ddof=1) / right.size)
    )
    half = VOLUME_CHECK_Z * standard_error
    return {
        "n_kept": int(left.size),
        "n_other": int(right.size),
        "mean_kept": float(left.mean()),
        "mean_other": float(right.mean()),
        "difference": difference,
        "standard_error": standard_error,
        "ci95_low": difference - half,
        "ci95_high": difference + half,
        "excludes_zero": bool((difference - half) > 0.0 or (difference + half) < 0.0),
        "approximation": "welch_normal",
        "z": VOLUME_CHECK_Z,
    }


def difficulty_check(
    kept_draws: Sequence[int],
    difficulty: Mapping[int, float],
    *,
    easiest_count: int = EASIEST_DRAW_COUNT,
    withdrawn_reason: str | None = None,
) -> dict[str, Any]:
    """P3 check B: overlap of the kept draws with the easiest draws, and its exact p-value.

    ``withdrawn_reason`` marks the result **withdrawn in the artifact itself**, not only in prose
    (``BRIEF_18`` finding F3).  It exists for one measured situation: on the tier whose own
    behaviour policy DEFINES the difficulty ranking, "top decile by return" and "easiest draws by
    that policy's ATT" are two functions of the same episodes, so the overlap is close to a
    tautology and the hypergeometric null -- which assumes the two sets are independent -- does not
    apply.  A withdrawn check's numbers are still reported, because hiding them would be a second
    error; :func:`score_p3` refuses to let a withdrawn check contribute to a signature.
    """
    kept = sorted({int(d) for d in kept_draws})
    unknown = [d for d in kept if d not in difficulty]
    if unknown:
        raise ValueError(
            f"no difficulty is recorded for draws {unknown[:5]}; check B ranks the kept draws "
            f"against the {DIFFICULTY_TIER} tier and cannot rank a draw that tier never ran"
        )
    ordered = sorted(difficulty, key=lambda d: (difficulty[d], d))
    easiest = sorted(ordered[: int(easiest_count)])
    overlap = len(set(kept) & set(easiest))
    population = len(difficulty)
    result = {
        "population": population,
        "kept_count": len(kept),
        "easiest_count": int(easiest_count),
        "easiest_draws": easiest,
        "kept_draws": kept,
        "overlap": overlap,
        "expected_overlap": len(kept) * int(easiest_count) / population,
        "p_value": hypergeometric_upper_tail(population, int(easiest_count), len(kept), overlap),
        "null": "hypergeometric, drawing the kept set uniformly from the training draws",
        "withdrawn": False,
        "withdrawn_reason": None,
    }
    if withdrawn_reason is not None:
        result["withdrawn"] = True
        result["withdrawn_reason"] = str(withdrawn_reason)
    return result


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman's rank correlation, average ranks on ties, computed without a new dependency.

    Used for one job (``BRIEF_18`` finding F3): measuring how far each tier's stream returns already
    determine the difficulty ranking check B scores them against.  On the tier whose own policy
    defines that ranking the answer is near −1, which is what makes the check circular there.
    """
    if len(xs) != len(ys):
        raise ValueError(f"spearman_rho needs equal lengths, got {len(xs)} and {len(ys)}")
    if len(xs) < 3:
        raise ValueError(f"spearman_rho needs at least three points, got {len(xs)}")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    left, right = ranks(list(xs)), ranks(list(ys))
    n = float(len(left))
    mean_left, mean_right = sum(left) / n, sum(right) / n
    cov = math.fsum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    var_left = math.fsum((a - mean_left) ** 2 for a in left)
    var_right = math.fsum((b - mean_right) ** 2 for b in right)
    if var_left == 0.0 or var_right == 0.0:
        return 0.0
    return cov / math.sqrt(var_left * var_right)


def return_versus_difficulty_rho(
    streams: Sequence[StreamReturn], difficulty: Mapping[int, float]
) -> float:
    """How far a tier's own stream returns already determine the difficulty ranking (F3)."""
    pairs = [(s.total_return, difficulty[int(s.flow_draw)]) for s in streams if int(s.flow_draw) in difficulty]
    if len(pairs) < 3:
        raise ValueError("too few streams overlap the difficulty ranking to correlate them")
    return spearman_rho([p[0] for p in pairs], [p[1] for p in pairs])


def selection_diagnostics_artifact(
    corpus_root: str | Path,
    tiers: Sequence[str],
    *,
    context_length: int = CONTEXT_LENGTH,
) -> dict[str, Any]:
    """P3's two checks for every tier, whichever way they come out."""
    specs = [_as_spec(tier) for tier in tiers]
    difficulty = difficulty_by_draw(corpus_root)
    difficulty_spec = tier_spec(DIFFICULTY_TIER)
    arrivals = draw_arrivals(tier_dirs(difficulty_spec, corpus_root)[0] / "manifest.json")

    agreement = _assert_flow_hashes_agree(specs, corpus_root)

    component_sets: dict[str, tuple[StreamReturn, ...]] = {}
    entries: dict[str, Any] = {}
    for spec in specs:
        dataset = tier_dataset(spec, corpus_root, context_length=context_length)
        selected = training_streams(spec, dataset, component_streams=component_sets or None)
        component_sets[spec.tier] = selected
        kept = top_decile_streams(selected)
        kept_draws = sorted({int(s.flow_draw) for s in kept})
        other_draws = sorted({int(s.flow_draw) for s in selected} - set(kept_draws))
        rho = return_versus_difficulty_rho(selected, difficulty)
        # F3: on the tier whose own policy DEFINES the difficulty ranking, check B scores a
        # selection against a ranking that selection is a function of.  Withdrawn there, by tier
        # identity rather than by a threshold on rho -- a threshold would be a second chosen
        # constant, and the circularity is structural, not empirical.
        withdrawn = (
            (
                f"check B ranks the kept draws by the {DIFFICULTY_TIER} tier's own att_horizon, and "
                f"this IS that tier: its top decile by return and the easiest draws by its own ATT "
                f"are two functions of the same episodes (measured Spearman rho = {rho:.4f} between "
                "stream return and own-tier difficulty, against -0.23 or weaker on every other "
                "tier).  The hypergeometric null assumes the two sets are independent and they are "
                "not, so the p-value does not mean what it appears to mean and this check is "
                "withdrawn on this tier (BRIEF_18 finding F3)"
            )
            if spec.tier == DIFFICULTY_TIER
            else None
        )
        entries[spec.tier] = {
            "kept_streams": len(kept),
            "kept_draws": kept_draws,
            "return_versus_difficulty_rho": rho,
            "volume": volume_check(kept_draws, other_draws, arrivals),
            "difficulty": difficulty_check(
                kept_draws, difficulty, withdrawn_reason=withdrawn
            ),
        }

    return {
        "format_version": DIAGNOSTICS_FORMAT_VERSION,
        "role": (
            "P4.6 prediction P3: does the top-decile filter select easier demand?  Two checks per "
            "tier, on training data only, with no rollout and no gradient step"
        ),
        "checks": {
            "volume": (
                "vehicles per draw, rebuilt from the collection's own FlowRandomizer and verified "
                "against every recorded per-draw flow hash; kept versus discarded means with a "
                "Welch normal-approximation interval"
            ),
            "difficulty": (
                f"per-draw att_horizon of the {DIFFICULTY_TIER} tier; the {EASIEST_DRAW_COUNT} "
                "lowest are the easiest draws, and the overlap's null is hypergeometric"
            ),
        },
        "alpha": P3_ALPHA,
        "arrivals_source": str(tier_dirs(difficulty_spec, corpus_root)[0] / "manifest.json"),
        "arrivals_draws_verified": len(arrivals),
        "flow_hash_agreement": agreement,
        "difficulty_draws": len(difficulty),
        "tiers": entries,
        "runtime": runtime_provenance(),
    }


def _assert_flow_hashes_agree(
    specs: Sequence[TierSpec], corpus_root: str | Path
) -> dict[str, Any]:
    """Every tier must record the same demand for the draws it holds, or they are not comparable."""
    reference: dict[str, str] = {}
    checked = 0
    for spec in specs:
        for directory in tier_dirs(spec, corpus_root):
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            hashes = manifest.get("run_metadata", {}).get("flow_draw_sha256", {})
            for draw, digest in hashes.items():
                checked += 1
                if draw in reference and reference[draw] != digest:
                    raise ValueError(
                        f"draw {draw} was collected from two different demands "
                        f"({reference[draw]} against {digest} in {directory}); the tiers are not "
                        "comparable on this draw"
                    )
                reference[draw] = digest
    return {"draws": len(reference), "hash_comparisons": checked, "disagreements": 0}


# ----------------------------------------------------------------------
# Gate G: the re-used mappo1000 column
# ----------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes -- an identity of the container, not of the weights."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest_of(path: str | Path) -> str:
    """Canonical state-dict digest of a checkpoint file (``DEFERRED`` 29's identity)."""
    from agent.OfflineBaselines import canonical_state_dict_digest

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return canonical_state_dict_digest(payload["model"])


def assert_reused_checkpoint_identity(
    training_artifact: Mapping[str, Any],
    gate_artifact_p4: Mapping[str, Any],
    *,
    baselines_root: str | Path,
    dt_root: str | Path,
) -> dict[str, Any]:
    """The 15 baseline digests and the 5 DT file hashes must be the committed ones."""
    baselines: list[dict[str, Any]] = []
    for run in training_artifact.get("runs", []):
        path = Path(baselines_root) / f"{run['method']}_seed{int(run['seed'])}.pt"
        if not path.is_file():
            raise ValueError(f"re-used checkpoint is missing: {path}")
        digest = canonical_digest_of(path)
        if digest != run["canonical_digest"]:
            raise ValueError(
                f"{path}: canonical digest {digest} is not the committed "
                f"{run['canonical_digest']}; the re-used column would describe different weights"
            )
        baselines.append(
            {
                "method": run["method"],
                "seed": int(run["seed"]),
                "path": str(path),
                "canonical_digest": digest,
                "file_sha256": file_sha256(path),
            }
        )

    dt: list[dict[str, Any]] = []
    for seed, entry in sorted(gate_artifact_p4.get("checkpoints", {}).items()):
        path = Path(dt_root) / f"dt_seed{int(seed)}.pt"
        if not path.is_file():
            raise ValueError(f"re-used DT checkpoint is missing: {path}")
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"{path}: file sha256 {digest} is not the committed {entry['sha256']}; the "
                "re-used DT column would describe a different checkpoint"
            )
        dt.append(
            {
                "seed": int(seed),
                "path": str(path),
                "file_sha256": digest,
                # Recorded here for the first time: p4_training.json never carried a canonical
                # digest for these five checkpoints (P4.6 plan section 7.2).
                "canonical_digest": canonical_digest_of(path),
            }
        )

    if len(baselines) != 15 or len(dt) != 5:
        raise ValueError(
            f"expected 15 baseline and 5 DT checkpoints, verified {len(baselines)} and {len(dt)}"
        )
    return {"baselines": baselines, "dt": dt, "verified": len(baselines) + len(dt)}


def assert_reused_cells_reproduce(
    committed: Sequence[Mapping[str, Any]],
    rerolled: Sequence[EpisodeResult],
) -> dict[str, Any]:
    """Re-rolled cells must equal the committed ones exactly; any difference is BLOCKED.

    Exact equality on the float, never a tolerance: a tolerance here would accept precisely the
    drift the gate exists to detect.
    """
    produced = {
        (str(r.arm), int(r.seed) if r.seed is not None else -1, int(r.draw_id)): r
        for r in rerolled
    }
    mismatches: list[str] = []
    compared = 0
    for record in committed:
        key = (
            str(record["arm"]),
            int(record["seed"]) if record.get("seed") is not None else -1,
            int(record["draw_id"]),
        )
        match = produced.get(key)
        if match is None:
            mismatches.append(f"{key}: not re-rolled")
            continue
        compared += 1
        if float(match.att_horizon) != float(record["att_horizon"]):
            mismatches.append(
                f"{key}: att_horizon {match.att_horizon!r} against committed "
                f"{record['att_horizon']!r}"
            )
        if float(match.horizon_vehicle_count) != float(record["horizon_vehicle_count"]):
            mismatches.append(
                f"{key}: horizon_vehicle_count {match.horizon_vehicle_count!r} against committed "
                f"{record['horizon_vehicle_count']!r}"
            )
    if mismatches:
        raise ValueError(
            f"the re-used column does not reproduce: {len(mismatches)} difference(s) over "
            f"{len(committed)} committed cells, first {mismatches[:3]}; P4.6 is BLOCKED because a "
            "column may only be re-used by an instrument that reproduces it"
        )
    return {"compared": compared, "mismatches": 0, "comparison": "exact float equality"}


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------


def fixedtime_collection_settings(manifest_path: str | Path) -> dict[str, Any]:
    """The fixed-time configuration the tier was COLLECTED with, read from its manifest.

    ``BRIEF_17`` section 11, finding A3: ``PROJECT_PLAN`` section 6 says P2.5 "ships k=4", which is
    true of P2.5 and false of this corpus -- ``cf_hz1x1__fixedtime`` was collected at ``k = 6``.  A
    reference line drawn with a different ``k`` is a policy that generated none of these episodes.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metadata = payload.get("run_metadata", {})
    k = metadata.get("fixed_time_k")
    if k is None:
        raise ValueError(
            f"{manifest_path}: run_metadata records no fixed_time_k, so the schedule this tier was "
            "collected with is unknown and may not be guessed"
        )
    return {
        "fixed_time_k": int(k),
        "fixed_time_schedule_source": metadata.get("fixed_time_schedule_source"),
        "fixed_time_plan_sha256": metadata.get("fixed_time_plan_sha256"),
    }


def _fixedtime_factory(
    config_path: str | Path, settings: Mapping[str, Any]
) -> Callable[[Any], Any]:
    """The collection factory itself, with the plan hash asserted against the manifest's."""
    from types import SimpleNamespace

    from offline.policies.fixed_time import make_fixedtime, resolve_plan

    args = SimpleNamespace(
        env_config=str(config_path),
        fixed_time_k=int(settings["fixed_time_k"]),
        policy="fixedtime",
    )
    resolution = resolve_plan(args)
    if resolution.sha256 != settings["fixed_time_plan_sha256"]:
        raise ValueError(
            f"the fixed-time plan resolves to {resolution.sha256!r} but the tier was collected "
            f"with {settings['fixed_time_plan_sha256']!r}; this would measure a different policy"
        )
    if resolution.source != settings["fixed_time_schedule_source"]:
        raise ValueError(
            f"the fixed-time schedule source is {resolution.source!r} but the tier was collected "
            f"with {settings['fixed_time_schedule_source']!r}"
        )

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        policy = make_fixedtime(env, args, np.random.default_rng(0))
        return lambda _env, info: policy(info)

    return factory


def _random_factory(seed: int) -> Callable[[Any], Any]:
    """``offline/collect.py``'s own random policy, with a fresh generator per draw.

    Collection built ``numpy.random.default_rng(base_seed)`` once per draw
    (``offline/collect.py:717``); this builds one per draw too, seeded with the registered training
    seed so the arm has the five-seed structure every other arm in the grid has.
    """
    from types import SimpleNamespace

    from offline.collect import _make_random

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        policy = _make_random(env, SimpleNamespace(), np.random.default_rng(int(seed)))
        return lambda _env, info: policy(info)

    return factory


def arm_key(method: str, tier: str) -> str:
    """``"<method>@<tier>"`` -- the arm identity carried by every episode record."""
    if method not in METHODS and method != BEHAVIOUR_METHOD:
        raise ValueError(f"unknown method {method!r}; this task declares {list(METHODS)}")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; this task declares {sorted(TIERS)}")
    return f"{method}@{tier}"


def split_arm_key(arm: str) -> tuple[str, str]:
    """Inverse of :func:`arm_key`."""
    text = str(arm)
    if text.count("@") != 1:
        raise ValueError(f"{text!r} is not an arm key of this task; expected '<method>@<tier>'")
    method, tier = text.split("@")
    return method, tier


def transition_stream_keys(
    dataset: TrajectoryWindowDataset, group: tuple[int, int]
) -> list[tuple[str, str, str]]:
    """The stream identity of every ``stream_index`` in :func:`build_transitions`' table.

    Transcribed from that function's own iteration order -- episode records in load order, each
    record's intersections in ``ix_ids`` order, skipping any stream of another group -- so a row
    can be traced back to the episode it came from.  A test checks the mapping against the rewards
    themselves rather than against this docstring.
    """
    from offline.trajectory_logger import load_episode

    keys: list[tuple[str, str, str]] = []
    for record in dataset.episode_records:
        episode = load_episode(Path(record.dataset_dir) / record.episode_file)
        for ix_id in record.ix_ids:
            arrays = episode.intersections[ix_id]
            shape = (int(arrays.state.shape[1]), int(arrays.avail_mask.shape[1]))
            if shape != tuple(group):
                continue
            keys.append((record.dataset_dir, record.episode_file, ix_id))
    return keys


def filter_transitions_to_streams(
    table: TransitionTable,
    stream_keys: Sequence[tuple[str, str, str]],
    streams: Sequence[StreamReturn],
) -> TransitionTable:
    """Restrict a transition table to the rows of *streams*, keeping their order."""
    wanted = {s.key for s in streams}
    keep_index = [i for i, key in enumerate(stream_keys) if key in wanted]
    if len(keep_index) != len(wanted):
        raise ValueError(
            f"the transition table holds {len(keep_index)} of the {len(wanted)} requested streams; "
            "the table and the selection describe different data"
        )
    selector = torch.zeros(len(table), dtype=torch.bool)
    lookup = torch.as_tensor(keep_index, dtype=torch.int64)
    selector |= torch.isin(table.stream_index, lookup)
    return table.select(torch.nonzero(selector, as_tuple=True)[0])


def merge_training_records(
    existing: Mapping[str, Any], fresh: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Fresh records plus the existing ones this chunk did not train, keyed by tier/method/seed."""
    for field in ("declared_gradient_steps", "context_length"):
        if field in existing and field in fresh and existing[field] != fresh[field]:
            raise ValueError(
                f"refusing to merge two training artifacts that disagree on {field}: "
                f"{existing[field]!r} against {fresh[field]!r}; they describe different designs"
            )
    trained_now = {
        (run["tier"], run["method"], int(run["seed"])) for run in fresh.get("runs", [])
    }
    kept = [
        run
        for run in existing.get("runs", [])
        if (run["tier"], run["method"], int(run["seed"])) not in trained_now
    ]
    return [*kept, *fresh.get("runs", [])]


def assert_cell_complete(
    method: str,
    tier: str,
    seeds: Sequence[int | None],
    draws: Sequence[int],
    produced: Sequence[EpisodeResult],
) -> None:
    """A cell must contain exactly the runs the declaration asks for.

    ``seeds=(None,)`` is a deterministic arm -- the ``fixedtime`` behaviour reference -- and is
    written as one run per draw rather than as a missing seed.
    """
    from collections import Counter

    arm = f"{method}@{tier}"
    want = Counter(
        (arm, -1 if seed is None else int(seed), int(draw)) for seed in seeds for draw in draws
    )
    got = Counter((r.arm, int(r.seed) if r.seed is not None else -1, int(r.draw_id)) for r in produced)
    missing = sorted(str(k) for k in (want - got))
    extra = sorted(str(k) for k in (got - want))
    if missing:
        raise ValueError(
            f"incomplete cell {arm}: {len(missing)} of {sum(want.values())} requested runs are "
            f"missing, first {missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"{len(extra)} produced run(s) were not requested for {arm}, first {extra[:5]}; a cell "
            "must describe exactly the runs the campaign asked for"
        )


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def cell_stats(episodes: Sequence[EpisodeResult]) -> dict[str, Any]:
    """One reported cell: n, ATT mean/std/CI95, the companion vehicle count, draws and seeds."""
    arms = sorted({r.arm for r in episodes})
    if len(arms) != 1:
        raise ValueError(f"a cell must describe one arm, got {arms}")
    att = mean_ci95([r.att_horizon for r in episodes])
    vehicles = mean_ci95([r.horizon_vehicle_count for r in episodes])
    method, tier = split_arm_key(arms[0])
    return {
        "arm": arms[0],
        "method": method,
        "tier": tier,
        "n_episodes": len(episodes),
        "att_horizon_mean": att.mean,
        "att_horizon_std": att.std,
        "att_horizon_ci95": att.ci95,
        "horizon_vehicle_count_mean": vehicles.mean,
        "horizon_vehicle_count_std": vehicles.std,
        "draw_ids": sorted({r.draw_id for r in episodes}),
        "seeds": sorted({r.seed for r in episodes if r.seed is not None}),
    }


def grid_comparisons(
    episodes_by_arm: Mapping[str, Sequence[EpisodeResult]],
) -> list[PairedComparison]:
    """Every within-tier method pair and every within-method tier pair, paired on shared draws.

    No cross-tier cross-method pair is defined: two arms differing in both axes answer no question
    this grid asks, and reporting one would invite a reader to attribute a difference to either.
    """
    present = {}
    for arm in episodes_by_arm:
        method, tier = split_arm_key(arm)
        present[(method, tier)] = arm

    tier_order = list((*PHASE1_TIER_ORDER, *MIXTURE_TIER_ORDER))
    out: list[PairedComparison] = []
    for tier in tier_order:
        methods = [m for m in METHODS if (m, tier) in present]
        for i, left in enumerate(methods):
            for right in methods[i + 1 :]:
                out.append(
                    paired_comparison(
                        episodes_by_arm[present[(left, tier)]],
                        episodes_by_arm[present[(right, tier)]],
                    )
                )
    for method in METHODS:
        tiers = [t for t in tier_order if (method, t) in present]
        for i, left in enumerate(tiers):
            for right in tiers[i + 1 :]:
                out.append(
                    paired_comparison(
                        episodes_by_arm[present[(method, left)]],
                        episodes_by_arm[present[(method, right)]],
                    )
                )
    return out


def behaviour_comparisons(
    episodes_by_arm: Mapping[str, Sequence[EpisodeResult]],
) -> list[PairedComparison]:
    """Each method against the policy that COLLECTED its tier, on the same 100 draws.

    ``BRIEF_17`` section 11, finding A3: C1's most valuable per-tier sentence is *"does an offline
    method beat the policy that produced its data?"*, and it is only answerable over shared draw
    ids -- substituting the tier's training-draw ATT is void under ``PREREGISTRATION`` A5.  The
    behaviour arm is not one of the four methods, so it never enters :func:`grid_comparisons`.
    """
    present = {}
    for arm in episodes_by_arm:
        method, tier = split_arm_key(arm)
        present[(method, tier)] = arm

    out: list[PairedComparison] = []
    for tier in (*PHASE1_TIER_ORDER, *MIXTURE_TIER_ORDER):
        reference = present.get((BEHAVIOUR_METHOD, tier))
        if reference is None:
            continue
        for method in METHODS:
            arm = present.get((method, tier))
            if arm is None:
                continue
            out.append(
                paired_comparison(episodes_by_arm[arm], episodes_by_arm[reference])
            )
    return out


#: A signed-rank effect size computed on this few non-tied pairs describes the residue, not the
#: arms.  Declared as a fraction of the shared draws so it scales with the design (``BRIEF_18``
#: section 5): with 100 shared draws it flags any comparison resting on fewer than 10 of them.
TIED_PAIR_CAVEAT_FRACTION = 0.10


def comparison_json(comparison: PairedComparison) -> dict[str, Any]:
    """A paired comparison as JSON, **carrying its own caveat when its effect size is unusable**.

    ``dt@fixedtime`` against its behaviour policy is the case this exists for: the two arms are
    identical, so 98 of 100 per-draw differences are exactly zero, Wilcoxon drops them and runs on
    **two float residues** -- returning ``rank_biserial = -1.0``, the same value 45 genuine effects
    in this artifact carry.  **A figure script sorting by |r| cannot tell them apart, and prose in a
    Return Packet does not reach a figure script**, so the caveat is a FIELD.
    """
    payload = comparison.to_json_obj()
    n_used = int(comparison.wilcoxon.n_used)
    n_zero = int(comparison.wilcoxon.n_zero)
    floor = TIED_PAIR_CAVEAT_FRACTION * float(comparison.n_shared_draws)
    if n_zero > 0 and n_used < floor:
        payload["effect_size_caveat"] = (
            f"rank_biserial and p_value rest on {n_used} non-tied pair(s) of "
            f"{comparison.n_shared_draws} shared draws ({n_zero} exact ties), which is below the "
            f"declared {TIED_PAIR_CAVEAT_FRACTION:.0%} floor.  The two arms are (near-)identical on "
            "these draws; the tie count is the result and these two statistics are not comparable "
            "with the artifact's other pairs"
        )
        payload["effect_size_usable"] = False
    else:
        payload["effect_size_usable"] = True
    return payload


def _ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Ranks by ascending value, 1 = lowest; exact ties take the average rank."""
    ordered = sorted(values, key=lambda k: (values[k], k))
    out: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and values[ordered[stop + 1]] == values[ordered[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            out[ordered[position]] = average
        index = stop + 1
    return out


def score_p1(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """P1: BC's rank per tier, the primary rule of plan section 4.1 and its continuous companion.

    **HELD** iff BC is worst of the four on every tier present; **FAILED** if it is not worst
    somewhere; **NOT RESOLVED** if an exact tie makes the rank ambiguous.  The plan discloses that
    BC is already rank 4 on ``mappo1000`` in P4.4's committed cells, so this is the falsifiable
    form of "worsens as quality falls" given that anchor.
    """
    ranks: dict[str, float] = {}
    gaps: dict[str, float] = {}
    tied: list[str] = []
    for tier, methods in cells.items():
        means = {m: float(entry["att_horizon_mean"]) for m, entry in methods.items()}
        if "bc" not in means or len(means) < 2:
            continue
        rank = _ranks(means)["bc"]
        if rank != int(rank):
            tied.append(tier)
        ranks[tier] = rank
        gaps[tier] = means["bc"] - min(v for m, v in means.items() if m != "bc")

    order = [t for t in (*PHASE1_TIER_ORDER, *MIXTURE_TIER_ORDER) if t in ranks]
    worst_rank = {tier: float(len(cells[tier])) for tier in ranks}
    if tied:
        outcome = "NOT RESOLVED"
    elif all(ranks[tier] == worst_rank[tier] for tier in ranks):
        outcome = "HELD"
    else:
        outcome = "FAILED"
    quality_index = [float(order.index(t) + 1) for t in order]
    return {
        "prediction": (
            "P1 -- BC's rank among the four methods worsens as tier quality falls"
        ),
        "primary_rule": (
            "HELD iff BC's rank is the worst of the methods present on every tier; FAILED if it "
            "is better than worst on any tier; NOT RESOLVED on an exact tie"
        ),
        "outcome": outcome,
        "bc_rank_by_tier": {tier: int(ranks[tier]) if ranks[tier] == int(ranks[tier]) else ranks[tier] for tier in order},
        "tiers_in_quality_order": order,
        "tied_tiers": tied,
        "bc_gap_to_best_other": {tier: gaps[tier] for tier in order},
        "kendall_tau_b_quality_vs_gap": (
            kendall_tau_b(quality_index, [gaps[t] for t in order]) if len(order) > 1 else None
        ),
        "secondary_note": (
            "the gap and its tau-b carry no threshold; they are reported because a rank pinned at "
            "its floor compresses away the magnitude"
        ),
    }


def score_p2(
    cells: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[PairedComparison],
) -> dict[str, Any]:
    """P2: %BC's advantage over BC per tier, scored fully only when the mixtures are present."""
    advantage: dict[str, float] = {}
    for tier, methods in cells.items():
        if "bc" in methods and "bc_top10" in methods:
            advantage[tier] = float(methods["bc"]["att_horizon_mean"]) - float(
                methods["bc_top10"]["att_horizon_mean"]
            )
    intervals: dict[str, Any] = {}
    for comparison in comparisons:
        try:
            left_method, left_tier = split_arm_key(comparison.left_arm)
            right_method, right_tier = split_arm_key(comparison.right_arm)
        except ValueError:
            continue
        if left_tier == right_tier and {left_method, right_method} == {"bc", "bc_top10"}:
            if left_method != "bc":
                # grid_comparisons emits pairs in METHODS order, so bc is always on the left.  A
                # reversed pair would silently flip the sign of every advantage reported here, so
                # it is refused rather than accommodated by a branch no caller can reach.
                raise ValueError(
                    f"the bc/bc_top10 comparison for {left_tier} arrived as "
                    f"{comparison.left_arm} vs {comparison.right_arm}; P2's advantage is defined "
                    "as bc minus bc_top10 and the pair order fixes its sign"
                )
            intervals[left_tier] = {
                "mean_difference": comparison.mean_difference,
                "ci95_low": comparison.ci95_low,
                "ci95_high": comparison.ci95_high,
                "ci95_width": comparison.ci95_width,
                "rank_biserial": comparison.rank_biserial,
            }

    single = [t for t in PHASE1_TIER_ORDER if t in advantage]
    mixtures = [t for t in MIXTURE_TIER_ORDER if t in advantage]
    partial = "NOT SCORABLE"
    if "random" in advantage and len(single) > 1:
        minimum = min(advantage[t] for t in single)
        partial = "HELD" if advantage["random"] == minimum else "FAILED"

    full = "NOT SCORABLE"
    if len(mixtures) == len(MIXTURE_TIER_ORDER) and "random" in advantage:
        everything = {**advantage}
        argmin = min(everything, key=lambda t: (everything[t], t))
        argmax = max(everything, key=lambda t: (everything[t], t))
        full = "HELD" if argmin == "random" and argmax in MIXTURE_TIER_ORDER else "FAILED"

    return {
        "prediction": (
            "P2 -- %BC's advantage over BC is LARGEST on the heterogeneous mixtures and SMALLEST "
            "on random"
        ),
        "full_rule": (
            "HELD iff the advantage attains its minimum at random and its maximum at a mixture "
            "tier; scorable only when all three mixture tiers are present"
        ),
        "partial_rule": (
            "phase 1 alone scores only the clause it can see: random attains the minimum among "
            "the single-controller tiers"
        ),
        "full_outcome": full,
        "partial_outcome": partial,
        "advantage_by_tier": advantage,
        "advantage_intervals": intervals,
        "mixtures_present": mixtures,
        "note": (
            "the mixture half of P2 is unscorable without phase 2 and no verdict is issued on it"
        ),
    }


def score_p3(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """P3: the demand-signature rule of plan section 4.3, applied per tier."""
    by_tier: dict[str, Any] = {}
    for tier, entry in diagnostics.get("tiers", {}).items():
        volume = entry["volume"]
        difficulty = entry["difficulty"]
        excludes_zero = bool(
            volume.get(
                "excludes_zero",
                float(volume["ci95_low"]) > 0.0 or float(volume["ci95_high"]) < 0.0,
            )
        )
        # A withdrawn check contributes nothing to a signature, whatever its p-value says
        # (BRIEF_18 finding F3).  Its numbers are still reported: withdrawing a check is not the
        # same as hiding it.
        withdrawn = bool(difficulty.get("withdrawn", False))
        significant = (not withdrawn) and float(difficulty["p_value"]) < P3_ALPHA
        by_tier[tier] = {
            "volume_difference": float(volume["difference"]),
            "volume_ci95": [float(volume["ci95_low"]), float(volume["ci95_high"])],
            "volume_excludes_zero": excludes_zero,
            "difficulty_overlap": int(difficulty["overlap"]),
            "difficulty_expected": float(difficulty["expected_overlap"]),
            "difficulty_p_value": float(difficulty["p_value"]),
            "difficulty_withdrawn": withdrawn,
            "difficulty_withdrawn_reason": difficulty.get("withdrawn_reason"),
            "return_versus_difficulty_rho": entry.get("return_versus_difficulty_rho"),
            "demand_signature": bool(significant or excludes_zero),
            "signature_carried_by": (
                [name for name, on in (("volume", excludes_zero), ("difficulty", significant)) if on]
                or None
            ),
        }
    if "random" not in by_tier:
        outcome = "NOT SCORABLE"
    else:
        outcome = "HELD" if by_tier["random"]["demand_signature"] else "FAILED"
    return {
        "prediction": (
            "P3 -- on random, %BC's top decile carries a DEMAND signature; on mappo1000 it did not"
        ),
        "rule": (
            "a tier carries a demand signature iff the difficulty check's exact p-value is below "
            f"{P3_ALPHA} OR the volume check's 95 % interval excludes zero; P3 is scored on random"
        ),
        "outcome": outcome,
        "by_tier": by_tier,
        "multiplicity": (
            "none applied, declared in advance: the rule is fixed on one tier and the other "
            "columns are descriptive"
        ),
    }


def kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall's tau-b, reported beside P1's rank rule and carrying no threshold."""
    left = [float(x) for x in xs]
    right = [float(y) for y in ys]
    if len(left) != len(right):
        raise ValueError(f"kendall_tau_b needs equal lengths, got {len(left)} and {len(right)}")
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = left[i] - left[j]
            dy = right[i] - right[j]
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                tied_x += 1
            elif dy == 0.0:
                tied_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    )
    if denominator == 0.0:
        return 0.0
    return (concordant - discordant) / denominator


def assert_no_verdicts(payload: Any) -> None:
    """Refuse to emit an equivalence verdict anywhere in the artifact (``BRIEF_17`` section 4).

    A6's delta is ``mappo1000``-specific and no per-tier delta can be derived before this run
    without using the run's own result, so a verdict here would be circular.  The check walks the
    whole payload because a nested verdict is exactly the one a shallow check would miss.
    """
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if "verdict" in str(key).lower():
                    raise ValueError(
                        f"{path}.{key}: this task issues no equivalence verdict anywhere "
                        "(BRIEF_17 section 4); report the difference, its CI and its width instead"
                    )
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str) and node in _VERDICT_VALUES:
            raise ValueError(
                f"{path}: {node!r} is an equivalence verdict and this task issues none "
                "(BRIEF_17 section 4)"
            )

    walk(payload, "$")


def measurement_commits(inputs: Sequence[Mapping[str, Any]]) -> list[str]:
    """The commits that produced this report's INPUTS, de-duplicated (``DEFERRED`` 39).

    A ``tmux`` campaign is maximally chunked -- one tier's cells are measured hours apart from
    another's -- so a single ``runtime.git_commit`` describes when the report was assembled and
    nothing about what produced its numbers.  Measured precedent: ``output/p4_4/gate_a.json``
    carries ``738884b`` while its three ``eval_*.json`` carry ``c13aaa9``.
    ``runtime_provenance`` checks each of these for reachability from ``HEAD`` and moves the
    unreachable ones into their own field rather than dropping them.
    """
    seen: set[str] = set()
    for payload in inputs:
        commit = str(payload.get("runtime", {}).get("git_commit") or "")
        if commit:
            seen.add(commit)
    return sorted(seen)


def grid_artifact(
    declaration: Mapping[str, Any],
    training: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    gate: Mapping[str, Any],
    episodes_by_arm: Mapping[str, Sequence[EpisodeResult]],
    *,
    inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The reported artifact: cells, episodes, comparisons, predictions and provenance."""
    if gate.get("status") != "PASS":
        raise ValueError(
            f"the re-use gate did not pass ({gate.get('status')!r}); no P4.6 number may be "
            "reported until this session's instrument reproduces the column it re-uses"
        )

    cells_by_tier: dict[str, dict[str, Any]] = {}
    behaviour_cells: dict[str, Any] = {}
    flat_cells: dict[str, Any] = {}
    for arm, episodes in episodes_by_arm.items():
        method, tier = split_arm_key(arm)
        cell = cell_stats(episodes)
        flat_cells[arm] = cell
        if method == BEHAVIOUR_METHOD:
            behaviour_cells[tier] = {**cell, "reference": BEHAVIOUR_REFERENCE_BY_TIER.get(tier)}
            continue
        cells_by_tier.setdefault(tier, {})[method] = cell

    incomplete = {
        tier: sorted(set(METHODS) - set(methods))
        for tier, methods in cells_by_tier.items()
        if set(methods) != set(METHODS)
    }
    if incomplete:
        raise ValueError(
            f"these tiers are missing methods {incomplete}; a tier enters the artifact only when "
            "all four of its methods are complete (docs/plans/p4.6.md section 12.2)"
        )

    comparisons = grid_comparisons(episodes_by_arm)
    tiers_present = [
        tier for tier in (*PHASE1_TIER_ORDER, *MIXTURE_TIER_ORDER) if tier in cells_by_tier
    ]
    payload = {
        "format_version": GRID_FORMAT_VERSION,
        "role": (
            "P4.6: four offline methods across the C1 data-quality ladder, at equal training-set "
            "size, on the registered held-out pool"
        ),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "declared_in": "docs/plans/p4.6.md, committed before the first gradient step",
        "methods": list(METHODS),
        "seeds": list(TRAINING_SEEDS),
        "held_out_draws": list(HELD_OUT_DRAWS),
        "tiers_present": tiers_present,
        "tier_order_by_tier_label_att": tiers_present,
        "tier_label_att_training_draws": {
            tier: TIER_LABEL_ATT_TRAINING_DRAWS[tier]
            for tier in tiers_present
            if tier in TIER_LABEL_ATT_TRAINING_DRAWS
        },
        "phase": 1 if not any(tier in MIXTURE_TIER_ORDER for tier in tiers_present) else 2,
        "reporting_rule": (
            "paired mean differences with 95 % CIs, CI widths and rank-biserial effect sizes; NO "
            "equivalence verdict anywhere (BRIEF_17 section 4).  A6's delta is mappo1000-specific "
            "and no per-tier margin can be derived before this run without using its own result"
        ),
        "multiplicity": (
            f"{len(comparisons)} paired comparisons, no correction applied and none needed: no "
            "decision rule is attached to any of them"
        ),
        "cells": flat_cells,
        "cells_by_tier": cells_by_tier,
        "behaviour_cells": behaviour_cells,
        "behaviour_comparisons": [
            comparison_json(c) for c in behaviour_comparisons(episodes_by_arm)
        ],
        "behaviour_reference_rule": (
            "each method against the policy that COLLECTED its tier, over the same 100 held-out "
            "draws; substituting a tier's training-draw ATT would be void under PREREGISTRATION "
            "A5, which is why random and fixedtime were evaluated here (BRIEF_17 section 11, A3)"
        ),
        "comparisons": [comparison_json(c) for c in comparisons],
        "predictions": {
            "P1": score_p1(cells_by_tier),
            "P2": score_p2(cells_by_tier, comparisons),
            "P3": score_p3(diagnostics),
        },
        "declaration": {
            "format_version": declaration.get("format_version"),
            "tiers": {
                tier: {
                    key: entry[key]
                    for key in (
                        "training_streams",
                        "training_windows",
                        "top_decile_streams",
                        "target_rtg",
                        "rtg_scale",
                        "statistics_digest",
                        "subsample",
                        "iql_reward_scale",
                    )
                }
                for tier, entry in declaration.get("tiers", {}).items()
            },
        },
        "training": {
            "format_version": training.get("format_version"),
            "runs": [
                {
                    key: run[key]
                    for key in ("tier", "method", "seed", "gradient_steps", "final_loss", "canonical_digest")
                    if key in run
                }
                for run in training.get("runs", [])
            ],
        },
        "gate": gate,
        "diagnostics": diagnostics,
        "episodes": [
            {
                "arm": e.arm,
                "seed": e.seed,
                "draw_id": e.draw_id,
                "att_horizon": e.att_horizon,
                "horizon_vehicle_count": e.horizon_vehicle_count,
                "episode_reward": e.episode_reward,
            }
            for arm in sorted(episodes_by_arm)
            for e in episodes_by_arm[arm]
        ],
        "runtime": runtime_provenance(measurement_commits(inputs)),
    }
    assert_equal_training_size(declaration)
    assert_no_verdicts(payload)
    return payload


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``declare``, ``diagnostics``, ``gate``, ``train``, ``evaluate``, ``report``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.method_tier_grid",
        description="P4.6: four offline methods across the C1 data-quality ladder.",
    )
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--scenario-key", default="cityflow1x1")
    parser.add_argument("--scenario-id", default="cityflow1x1")
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--work-dir", default="output/p4_6")
    parser.add_argument("--checkpoint-dir", default="output/p4_6/checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    parser.add_argument(
        "--artifact-prefix",
        default=DEFAULT_ARTIFACT_PREFIX,
        help="filename prefix of the four artifacts this module writes and reads "
        f"(default: {DEFAULT_ARTIFACT_PREFIX}, which reproduces every pre-P4.7 invocation "
        "exactly; P4.7's phase-2 campaign passes p4_7 so it cannot write into P4.6's records)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch threads for this process; 1 is the default because the unpinned path "
        "deadlocks on this workload (DEFERRED 41) and reproduces P4 bit-identically",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    declare = sub.add_parser("declare", help="write the declaration; no training, no rollouts")
    declare.add_argument("--tiers", default=",".join(PHASE1_TIER_ORDER))

    diagnostics = sub.add_parser("diagnostics", help="prediction P3's two checks, per tier")
    diagnostics.add_argument("--tiers", default=",".join(PHASE1_TIER_ORDER))

    gate = sub.add_parser("gate", help="Gate G: prove this session reproduces the re-used column")
    gate.add_argument("--p4-4-training", default="docs/data/p4_4_training.json")
    gate.add_argument("--p4-4-baselines", default="docs/data/p4_4_baselines.json")
    gate.add_argument("--p4-gate", default="docs/data/p4_gate.json")
    gate.add_argument("--baselines-checkpoint-dir", required=True)
    gate.add_argument("--dt-checkpoint-dir", required=True)

    train = sub.add_parser("train", help="train one tier's methods across the five seeds")
    train.add_argument("--tier", required=True, choices=sorted(TIERS))
    train.add_argument("--methods", default=",".join(METHODS))
    train.add_argument("--log-every", type=int, default=5000)

    evaluate = sub.add_parser("evaluate", help="evaluate ONE cell over the held-out pool")
    evaluate.add_argument("--tier", required=True, choices=sorted(TIERS))
    evaluate.add_argument("--method", required=True, choices=list(METHODS))

    behaviour = sub.add_parser(
        "behaviour",
        help="evaluate a tier's COLLECTING policy on the held-out pool (BRIEF_17 section 11, A3)",
    )
    behaviour.add_argument("--tier", required=True, choices=sorted(BEHAVIOUR_REFERENCE_BY_TIER))

    report = sub.add_parser("report", help="assemble the reported artifact")
    report.add_argument("--tiers", default=",".join(PHASE1_TIER_ORDER))
    return parser


def _requested_tiers(value: str) -> list[str]:
    tiers = [t.strip() for t in str(value).split(",") if t.strip()]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise ValueError(f"unknown tier(s) {unknown}; this task declares {sorted(TIERS)}")
    return tiers


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    from offline.materialise_draws import draw_config_path

    args = build_parser().parse_args(argv)
    pin_torch_threads(args.torch_threads)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"--out-dir does not exist: {out_dir}")

    def config_for_draw(draw_id: int) -> Path:
        return draw_config_path(args.scenario_key, draw_id, out_root=args.draws_root)

    if args.command == "declare":
        return _run_declare(args, out_dir)
    if args.command == "diagnostics":
        return _run_diagnostics(args, out_dir)
    if args.command == "gate":
        return _run_gate(args, config_for_draw, work_dir)
    if args.command == "train":
        return _run_train(args, out_dir)
    if args.command == "evaluate":
        return _run_evaluate(args, config_for_draw, out_dir, work_dir)
    if args.command == "behaviour":
        return _run_behaviour(args, config_for_draw, work_dir)
    return _run_report(args, out_dir, work_dir)


def _run_behaviour(
    args: argparse.Namespace,
    config_for_draw: Callable[[int], Path],
    work_dir: Path,
) -> int:
    """Roll a tier's own collecting policy over the held-out pool, through the same instrument."""
    spec = tier_spec(args.tier)
    reference = BEHAVIOUR_REFERENCE_BY_TIER[spec.tier]
    if reference["source"] != "evaluated_here":
        raise ValueError(
            f"{spec.tier}'s behaviour cell is committed in {reference['artifact']}; re-rolling a "
            "settled number is a second measurement of it, not a check on this one"
        )

    settings = env_settings_for_tiers([spec], args.corpus_root)
    draws = list(HELD_OUT_DRAWS)
    arm = arm_key(BEHAVIOUR_METHOD, spec.tier)
    provenance: dict[str, Any] = {"policy": reference["arm"], "factory": reference["artifact"]}
    produced: list[EpisodeResult] = []

    if spec.tier == "fixedtime":
        collected = fixedtime_collection_settings(
            tier_dirs(spec, args.corpus_root)[0] / "manifest.json"
        )
        provenance.update(collected)
        seeds: list[int | None] = [None]
        factories = {None: _fixedtime_factory(config_for_draw(draws[0]), collected)}
    elif spec.tier == "random":
        seeds = list(TRAINING_SEEDS)
        factories = {seed: _random_factory(int(seed)) for seed in TRAINING_SEEDS}
        provenance["rng"] = "numpy.random.default_rng(training seed), rebuilt per draw"
    else:  # pragma: no cover - the two branches above are the only evaluated_here tiers
        raise ValueError(f"no behaviour factory is declared for {spec.tier!r}")

    for seed in seeds:
        print(f"{arm} seed {seed} over {len(draws)} draws", flush=True)
        produced.extend(
            evaluate_arm(
                arm=arm,
                seed=None if seed is None else int(seed),
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=factories[seed],
                engine_seed=args.engine_seed,
            )
        )
    assert_cell_complete(BEHAVIOUR_METHOD, spec.tier, seeds, draws, produced)

    cell = cell_stats(produced)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": GRID_FORMAT_VERSION,
            "arm": arm,
            "tier": spec.tier,
            "method": BEHAVIOUR_METHOD,
            "role": (
                "the policy that collected this tier, evaluated on the registered held-out pool "
                "so the per-tier comparison is over shared draw ids (PREREGISTRATION A5)"
            ),
            "behaviour_policy": provenance,
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
        work_dir / f"eval_{spec.tier}_{BEHAVIOUR_METHOD}.json",
    )
    print(
        f"  {arm}: att_horizon {cell['att_horizon_mean']:.4f} +/- {cell['att_horizon_ci95']:.4f}  "
        f"vehicle_count {cell['horizon_vehicle_count_mean']:.2f}  n={cell['n_episodes']}",
        flush=True,
    )
    return 0


def _run_declare(args: argparse.Namespace, out_dir: Path) -> int:
    """Everything is validated before the first byte is written."""
    tiers = _requested_tiers(args.tiers)
    payload = declaration_artifact(args.corpus_root, tiers)
    write_json_atomic(payload, artifact_path(out_dir, "declaration", _artifact_prefix(args)))
    for tier in tiers:
        entry = payload["tiers"][tier]
        print(
            f"{tier:12s} streams {entry['training_streams']:4d}  windows "
            f"{entry['training_windows']:6d}  top decile {entry['top_decile_streams']:3d}  "
            f"target {entry['target_rtg']:.1f}  scale {entry['rtg_scale']:.1f}  "
            f"stats {entry['statistics_digest'][:12]}",
            flush=True,
        )
    return 0


def _run_diagnostics(args: argparse.Namespace, out_dir: Path) -> int:
    tiers = _requested_tiers(args.tiers)
    payload = selection_diagnostics_artifact(args.corpus_root, tiers)
    write_json_atomic(
        payload, artifact_path(out_dir, "selection_diagnostics", _artifact_prefix(args))
    )
    for tier in tiers:
        entry = payload["tiers"][tier]
        volume = entry["volume"]
        difficulty = entry["difficulty"]
        print(
            f"{tier:12s} arrivals kept {volume['mean_kept']:.1f} other {volume['mean_other']:.1f} "
            f"diff {volume['difference']:+.1f} CI [{volume['ci95_low']:+.1f}, "
            f"{volume['ci95_high']:+.1f}]  overlap {difficulty['overlap']}/"
            f"{difficulty['kept_count']} (expected {difficulty['expected_overlap']:.1f}, "
            f"p {difficulty['p_value']:.4g})",
            flush=True,
        )
    return 0


def _run_gate(
    args: argparse.Namespace,
    config_for_draw: Callable[[int], Path],
    work_dir: Path,
) -> int:
    """Gate G: identity of the re-used checkpoints, then exact re-rolling of declared cells."""
    from offline.rtg_calibration import agent_with_target

    training = json.loads(Path(args.p4_4_training).read_text(encoding="utf-8"))
    baselines = json.loads(Path(args.p4_4_baselines).read_text(encoding="utf-8"))
    p4_gate = json.loads(Path(args.p4_gate).read_text(encoding="utf-8"))

    identity = assert_reused_checkpoint_identity(
        training,
        p4_gate,
        baselines_root=args.baselines_checkpoint_dir,
        dt_root=args.dt_checkpoint_dir,
    )
    print(f"checkpoint identity verified: {identity['verified']} checkpoints", flush=True)

    settings = env_settings_for_tiers([REUSED_TIER], args.corpus_root)
    draws = list(GATE_B_DRAWS)
    committed: list[dict[str, Any]] = []
    rerolled: list[EpisodeResult] = []

    for method in METHODS:
        source_arm = REUSED_ARM_KEYS[method]
        committed.extend(
            record
            for record in baselines["episodes"]
            if record["arm"] == source_arm and int(record["draw_id"]) in set(draws)
        )
        for seed in TRAINING_SEEDS:
            if method == "dt":
                path = str(Path(args.dt_checkpoint_dir) / f"dt_seed{seed}.pt")
                factory = _gate_dt_factory(path, int(args.steps), args.device)
            else:
                path = str(Path(args.baselines_checkpoint_dir) / f"{method}_seed{seed}.pt")
                factory = _baseline_factory(method, path, int(args.steps), args.device)
            rerolled.extend(
                EpisodeResult(
                    arm=source_arm,
                    seed=result.seed,
                    draw_id=result.draw_id,
                    att_horizon=result.att_horizon,
                    horizon_vehicle_count=result.horizon_vehicle_count,
                    episode_reward=result.episode_reward,
                )
                for result in evaluate_arm(
                    arm=source_arm,
                    seed=int(seed),
                    draw_ids=draws,
                    config_for_draw=config_for_draw,
                    env_settings=settings,
                    scenario_id=args.scenario_id,
                    choose_action_factory=factory,
                    engine_seed=args.engine_seed,
                )
            )
        print(f"  re-rolled {source_arm} over {len(draws)} draws x {len(TRAINING_SEEDS)} seeds", flush=True)

    reproduction = assert_reused_cells_reproduce(committed, rerolled)

    # The DT arm is evaluated in this task through agent_with_target (which asserts the model
    # conditions on exactly the declared target) while P4 used load_gate_checkpoint.  One cell
    # decides whether the two paths are the same path.
    cross_path: dict[str, Any] = {}
    dt_path = str(Path(args.dt_checkpoint_dir) / f"dt_seed{TRAINING_SEEDS[0]}.pt")
    target = float(tier_spec(REUSED_TIER).target_rtg)

    def declared_factory(env: Any) -> Any:
        agent = agent_with_target(
            env, dt_path, declared_gradient_steps=int(args.steps), target_rtg=target,
            device=args.device,
        )
        return lambda _env, info: agent.act(info, explore=False, update_memory=True)

    declared_results = evaluate_arm(
        arm="madt",
        seed=int(TRAINING_SEEDS[0]),
        draw_ids=draws,
        config_for_draw=config_for_draw,
        env_settings=settings,
        scenario_id=args.scenario_id,
        choose_action_factory=declared_factory,
        engine_seed=args.engine_seed,
    )
    reference = {
        (int(r.seed), int(r.draw_id)): r
        for r in rerolled
        if r.arm == "madt" and int(r.seed) == int(TRAINING_SEEDS[0])
    }
    differences = [
        f"draw {r.draw_id}: {r.att_horizon!r} against {reference[(int(r.seed), int(r.draw_id))].att_horizon!r}"
        for r in declared_results
        if float(r.att_horizon) != float(reference[(int(r.seed), int(r.draw_id))].att_horizon)
    ]
    cross_path = {
        "cells": len(declared_results),
        "differences": differences,
        "identical": not differences,
        "role": (
            "load_gate_checkpoint (P4's path) against agent_with_target at the declared target "
            "(this task's path) on one seed and the declared gate draws"
        ),
    }
    if differences:
        raise ValueError(
            "the declared-target evaluation path does not reproduce P4's path on the gate cells: "
            f"{differences[:3]}; the new tiers would be measured by a different instrument"
        )

    payload = {
        "format_version": GRID_FORMAT_VERSION,
        "role": "Gate G: the re-used mappo1000 column, verified before any P4.6 training",
        "status": "PASS",
        "declared_draws": draws,
        "declared_in": "docs/plans/p4.6.md section 7; the draws are P4.5's GATE_B_DRAWS, inherited",
        "checkpoint_identity": identity,
        "reproduction": reproduction,
        "cross_path_check": cross_path,
        "engine_seed": int(args.engine_seed),
        "env_settings": {k: v for k, v in settings.items() if k != "compare_with"},
        "runtime": runtime_provenance(),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload, work_dir / "gate.json")
    print(
        f"GATE G PASS: {reproduction['compared']} cells reproduce exactly, "
        f"{identity['verified']} checkpoints identical",
        flush=True,
    )
    return 0


def _gate_dt_factory(path: str, declared: int, device: str | None) -> Callable[[Any], Any]:
    """P4's own evaluation path, unchanged: load the checkpoint and act greedily."""

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        agent = load_gate_checkpoint(env, path, declared, device=device)
        return lambda _env, info: agent.act(info, explore=False, update_memory=True)

    return factory


def _dt_factory(path: str, declared: int, target: float, device: str | None) -> Callable[[Any], Any]:
    """This task's DT path: the declared target is applied AFTER load and asserted to have taken."""
    from offline.rtg_calibration import agent_with_target

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        agent = agent_with_target(
            env, path, declared_gradient_steps=declared, target_rtg=target, device=device
        )
        return lambda _env, info: agent.act(info, explore=False, update_memory=True)

    return factory


def _component_streams(
    spec: TierSpec, corpus_root: str | Path, *, context_length: int
) -> dict[str, tuple[StreamReturn, ...]]:
    """A mixture tier's components' own declared training sets."""
    out: dict[str, tuple[StreamReturn, ...]] = {}
    for name in spec.components:
        component = tier_spec(name)
        dataset = tier_dataset(component, corpus_root, context_length=context_length)
        out[name] = training_streams(component, dataset)
    return out


def _run_train(args: argparse.Namespace, out_dir: Path) -> int:
    """Train one tier's methods across the five seeds, at the declared budget."""
    from agent.utils.utils import Utils
    from offline.dt_gate import train_dt

    spec = tier_spec(args.tier)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; known methods are {list(METHODS)}")

    dataset = tier_dataset(spec, args.corpus_root)
    stacked = stack_dataset(dataset)
    group = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    components = (
        _component_streams(spec, args.corpus_root, context_length=CONTEXT_LENGTH)
        if spec.subsample == "mixture"
        else None
    )
    selected = training_streams(spec, dataset, component_streams=components)
    # The prompt, the RTG scale and IQL's reward scale are all computed over the TRAINING SET, so
    # the declaration is checked after the selection and never before it (BRIEF_17 section 11, A4).
    declared = assert_declaration_matches_corpus(spec, selected)
    kept = top_decile_streams(selected)
    scale = iql_reward_scale([s.total_return for s in selected])
    digest = statistics_digest(dataset)

    device = torch.device(args.device) if args.device else Utils.resolve_device(None)
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tier": spec.tier,
        "dataset_dirs": [str(d) for d in tier_dirs(spec, args.corpus_root)],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
        "statistics_digest": digest,
        "subsample": spec.subsample,
        "training_streams": len(selected),
    }

    full_batch = filter_stacked_to_streams(dataset, stacked, selected)
    top_batch = filter_stacked_to_streams(dataset, stacked, kept)
    print(
        f"tier {spec.tier}: split streams {len(stream_returns(dataset))}  training streams "
        f"{len(selected)}  windows {int(full_batch['state'].shape[0])}  top decile {len(kept)} "
        f"({int(top_batch['state'].shape[0])} windows)\n"
        f"target {declared['target_rtg']}  scale {declared['rtg_scale']}  iql reward scale "
        f"{scale}  stats {digest[:12]}  device {device}",
        flush=True,
    )

    table: TransitionTable | None = None
    runs: list[dict[str, Any]] = []
    for method in methods:
        for seed in TRAINING_SEEDS:
            path = checkpoints / f"{spec.tier}_{method}_seed{seed}.pt"
            started = time.time()
            if method == "iql":
                if table is None:
                    full = build_transitions(dataset, group=group, reward_scale=scale)
                    table = filter_transitions_to_streams(
                        full, transition_stream_keys(dataset, group), selected
                    )
                    print(f"transitions {len(table)} of {len(full)}", flush=True)
                record = train_iql(
                    table,
                    state_dim=group[0],
                    n_actions=group[1],
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
                runs.append(_train_record(spec, method, record, path))
            elif method == "dt":
                result = train_dt(
                    full_batch,
                    state_dim=group[0],
                    n_actions=group[1],
                    seed=seed,
                    declared_gradient_steps=int(args.steps),
                    raise_to=None,
                    context_length=CONTEXT_LENGTH,
                    batch_size=BATCH_SIZE,
                    device=device,
                    checkpoint_path=path,
                    stats=dataset.stats,
                    scenario_id=scenario_id,
                    target_rtg=float(spec.target_rtg),
                    rtg_scale=float(spec.rtg_scale),
                    provenance=provenance,
                    log_every=args.log_every,
                )
                runs.append(_train_record(spec, method, result, path))
            else:
                record = train_bc(
                    top_batch if method == "bc_top10" else full_batch,
                    state_dim=group[0],
                    n_actions=group[1],
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
                runs.append(_train_record(spec, method, record, path))
            print(
                f"  {spec.tier} {method} seed {seed}: {time.time() - started:.1f}s  "
                f"final loss {runs[-1]['final_loss']:.5f}  digest {runs[-1]['canonical_digest'][:12]}",
                flush=True,
            )

    payload: dict[str, Any] = {
        "format_version": TRAINING_FORMAT_VERSION,
        "role": "P4.6 training record; the reported checkpoint is the one at the declared budget",
        "declared_gradient_steps": int(args.steps),
        "declared_in": "docs/plans/p4.6.md section 3.4, before the first gradient step",
        "raise_available": False,
        "context_length": CONTEXT_LENGTH,
        "seeds": list(TRAINING_SEEDS),
        "runs": runs,
        "runtime": runtime_provenance(),
    }
    destination = artifact_path(out_dir, "training", _artifact_prefix(args))
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        payload["runs"] = merge_training_records(existing, payload)
    write_json_atomic(payload, destination)
    return 0


def _train_record(spec: TierSpec, method: str, record: Any, path: Path) -> dict[str, Any]:
    """One training run, in the same shape whichever trainer produced it."""
    digest = canonical_digest_of(path)
    return {
        "tier": spec.tier,
        "method": method,
        "seed": int(record.seed),
        "gradient_steps": int(record.gradient_steps),
        "plateaued": bool(record.plateaued),
        "window_means": list(record.window_means),
        "final_loss": float(record.losses[-1]),
        "seconds": float(record.seconds),
        "checkpoint": str(path),
        "canonical_digest": digest,
        "file_sha256": file_sha256(path),
        "diagnostics": getattr(record, "diagnostics", None),
        "thread_regime": thread_regime(),
        "target_rtg": float(spec.target_rtg) if method == "dt" else None,
    }


def _run_evaluate(
    args: argparse.Namespace,
    config_for_draw: Callable[[int], Path],
    out_dir: Path,
    work_dir: Path,
) -> int:
    """Evaluate ONE cell over the held-out pool, so no single job runs long."""
    spec = tier_spec(args.tier)
    training = json.loads(
        artifact_path(out_dir, "training", _artifact_prefix(args)).read_text(encoding="utf-8")
    )
    if int(training["declared_gradient_steps"]) != int(args.steps):
        raise ValueError(
            f"the training artifact was produced under a declared budget of "
            f"{training['declared_gradient_steps']} steps but this evaluation was asked for "
            f"{int(args.steps)}; PREREGISTRATION.md section 6 fixes the budget before training"
        )
    runs = [
        run
        for run in training["runs"]
        if run["tier"] == spec.tier and run["method"] == args.method
    ]
    if len(runs) != len(TRAINING_SEEDS):
        raise ValueError(
            f"the training artifact records {len(runs)} runs for {args.method}@{spec.tier}, not "
            f"{len(TRAINING_SEEDS)}; a cell is evaluated only when its whole seed set is trained"
        )

    settings = env_settings_for_tiers([spec], args.corpus_root)
    draws = list(HELD_OUT_DRAWS)
    arm = arm_key(args.method, spec.tier)
    produced: list[EpisodeResult] = []
    for run in sorted(runs, key=lambda r: int(r["seed"])):
        seed = int(run["seed"])
        digest = canonical_digest_of(run["checkpoint"])
        if digest != run["canonical_digest"]:
            raise ValueError(
                f"{run['checkpoint']}: canonical digest {digest} is not the trained "
                f"{run['canonical_digest']}; this is not the model the training artifact records"
            )
        factory = (
            _dt_factory(run["checkpoint"], int(args.steps), float(spec.target_rtg), args.device)
            if args.method == "dt"
            else _baseline_factory(args.method, run["checkpoint"], int(args.steps), args.device)
        )
        print(f"{arm} seed {seed} over {len(draws)} draws", flush=True)
        produced.extend(
            EpisodeResult(
                arm=arm,
                seed=result.seed,
                draw_id=result.draw_id,
                att_horizon=result.att_horizon,
                horizon_vehicle_count=result.horizon_vehicle_count,
                episode_reward=result.episode_reward,
            )
            for result in evaluate_arm(
                arm=arm,
                seed=seed,
                draw_ids=draws,
                config_for_draw=config_for_draw,
                env_settings=settings,
                scenario_id=args.scenario_id,
                choose_action_factory=factory,
                engine_seed=args.engine_seed,
            )
        )
    assert_cell_complete(args.method, spec.tier, TRAINING_SEEDS, draws, produced)

    cell = cell_stats(produced)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": GRID_FORMAT_VERSION,
            "arm": arm,
            "tier": spec.tier,
            "method": args.method,
            "declared_gradient_steps": int(args.steps),
            "engine_seed": int(args.engine_seed),
            "target_rtg": float(spec.target_rtg) if args.method == "dt" else None,
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
        work_dir / f"eval_{spec.tier}_{args.method}.json",
    )
    print(
        f"  {arm}: att_horizon {cell['att_horizon_mean']:.4f} +/- {cell['att_horizon_ci95']:.4f}  "
        f"vehicle_count {cell['horizon_vehicle_count_mean']:.2f}  n={cell['n_episodes']}",
        flush=True,
    )
    return 0


def _episodes_from_records(records: Sequence[Mapping[str, Any]], arm: str) -> list[EpisodeResult]:
    return [
        EpisodeResult(
            arm=arm,
            seed=int(record["seed"]) if record.get("seed") is not None else None,
            draw_id=int(record["draw_id"]),
            att_horizon=float(record["att_horizon"]),
            horizon_vehicle_count=float(record["horizon_vehicle_count"]),
            episode_reward=float(record["episode_reward"]),
        )
        for record in records
    ]


def _run_report(args: argparse.Namespace, out_dir: Path, work_dir: Path) -> int:
    """Assemble the reported artifact from the gate, the declaration and the per-cell runs."""
    tiers = _requested_tiers(args.tiers)
    gate = json.loads((work_dir / "gate.json").read_text(encoding="utf-8"))
    declaration = json.loads(
        artifact_path(out_dir, "declaration", _artifact_prefix(args)).read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        artifact_path(out_dir, "selection_diagnostics", _artifact_prefix(args)).read_text(
            encoding="utf-8"
        )
    )
    training = json.loads(
        artifact_path(out_dir, "training", _artifact_prefix(args)).read_text(encoding="utf-8")
    )
    baselines = json.loads(Path("docs/data/p4_4_baselines.json").read_text(encoding="utf-8"))

    # DEFERRED 39, used from the start here because a tmux campaign is maximally chunked: every
    # input's own write-time commit is carried into the report rather than one commit standing for
    # all of them (BRIEF_17 section 12).
    inputs: list[Mapping[str, Any]] = [gate, declaration, diagnostics, training, baselines]

    episodes_by_arm: dict[str, list[EpisodeResult]] = {}
    for tier in tiers:
        for method in METHODS:
            arm = arm_key(method, tier)
            if tier == REUSED_TIER:
                source = REUSED_ARM_KEYS[method]
                records = [r for r in baselines["episodes"] if r["arm"] == source]
                if not records:
                    raise ValueError(
                        f"docs/data/p4_4_baselines.json carries no episodes for {source!r}, so "
                        f"{arm} cannot be re-used"
                    )
                episodes_by_arm[arm] = _episodes_from_records(records, arm)
            else:
                path = work_dir / f"eval_{tier}_{method}.json"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{path} does not exist: {arm} has not been evaluated, and a tier enters "
                        "the artifact only when all four of its methods are complete"
                    )
                payload = json.loads(path.read_text(encoding="utf-8"))
                inputs.append(payload)
                episodes_by_arm[arm] = _episodes_from_records(payload["episodes"], arm)
            assert_cell_complete(
                method, tier, TRAINING_SEEDS, list(HELD_OUT_DRAWS), episodes_by_arm[arm]
            )

        # The behaviour reference of this tier: committed where one exists, rolled here where none
        # did (BRIEF_17 section 11, finding A3).
        reference = BEHAVIOUR_REFERENCE_BY_TIER.get(tier)
        if reference is None:
            continue
        arm = arm_key(BEHAVIOUR_METHOD, tier)
        if reference["source"] == "committed":
            thresholds = json.loads(
                Path(reference["artifact"]).read_text(encoding="utf-8")
            )
            records = [r for r in thresholds["episodes"] if r["arm"] == reference["arm"]]
            if not records:
                raise ValueError(
                    f"{reference['artifact']} carries no episodes for {reference['arm']!r}, so "
                    f"{tier}'s behaviour reference cannot be re-used"
                )
            episodes_by_arm[arm] = _episodes_from_records(records, arm)
        else:
            path = work_dir / f"eval_{tier}_{BEHAVIOUR_METHOD}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} does not exist: {tier}'s behaviour policy has not been evaluated on "
                    "the held-out pool, and its training-draw ATT is not a substitute (A5)"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            inputs.append(payload)
            episodes_by_arm[arm] = _episodes_from_records(payload["episodes"], arm)
        seeds: list[int | None] = sorted(
            {e.seed for e in episodes_by_arm[arm]}, key=lambda s: (s is not None, s)
        )
        assert_cell_complete(
            BEHAVIOUR_METHOD, tier, seeds, list(HELD_OUT_DRAWS), episodes_by_arm[arm]
        )

    payload = grid_artifact(
        declaration, training, diagnostics, gate, episodes_by_arm, inputs=inputs
    )
    payload["reused"] = {
        "tier": REUSED_TIER,
        "source": "docs/data/p4_4_baselines.json",
        "arm_keys": dict(REUSED_ARM_KEYS),
        "gate": gate["status"],
        "note": (
            "P4.4's merged cells, re-used rather than re-run; the DT arm is that artifact's "
            "'madt' key under contract C9's neutral alias"
        ),
    }
    assert_no_verdicts(payload)
    write_json_atomic(payload, artifact_path(out_dir, "grid", _artifact_prefix(args)))

    # F2 (BRIEF_18 section 2): this row USED to print the tier label -- a mean over training draws
    # 1-200 -- beside four held-out means, which is the comparison PREREGISTRATION A5 makes void.  It
    # now reads the held-out behaviour cell, and every column states its draw range.
    print("tier          behaviour[1000-1099]  " + "  ".join(f"{m}[1000-1099]" for m in METHODS),
          flush=True)
    for tier in payload["tiers_present"]:
        row = payload["cells_by_tier"][tier]
        reference = payload["behaviour_cells"].get(tier)
        behaviour = float("nan") if reference is None else reference["att_horizon_mean"]
        print(
            f"{tier:12s} {behaviour:20.4f}  "
            + "  ".join(
                f"{method} {row[method]['att_horizon_mean']:8.4f}" for method in METHODS
            ),
            flush=True,
        )
    for name in ("P1", "P2", "P3"):
        prediction = payload["predictions"][name]
        outcome = prediction.get("outcome") or prediction.get("partial_outcome")
        print(f"{name}: {outcome}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
