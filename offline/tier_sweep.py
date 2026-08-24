"""P5.2: the spatial mixing layer across the grid4x4 ladder, and the head-count 2x2.

Artifact format version: ``p5.2-tier-sweep/1.0``.  Declaration format version:
``p5.2-declaration/1.0``.

⚠️ SKELETON.  Every function below raises :class:`NotImplementedError`; the constants are the
REGISTERED declarations from ``docs/plans/p5.2.md`` and are real.  Tests are written against this
surface first, so each one fails for its own reason rather than sharing a single import error.

WHAT THIS TASK DECIDES
----------------------
P5.1 measured spatial mixing on one tier of one scenario.  P5.2 asks two questions off the same
cells: **does the harm survive the data-quality axis**, and — the question ``BRIEF_27`` A1 added —
**does ``dt_nomix``'s lead survive it**.  Before either, it measures the head count, because P5.1's
negative result is about a single-head configuration (``PROJECT_PLAN`` section 1b, C3).  The
registered predictions, their bands, their thresholds and the stop rule are in
``docs/plans/p5.2.md``, committed before this file existed.

ALIGNMENT CONVENTION — INHERITED, NOT REDEFINED
-----------------------------------------------
The token layout, the left padding, the ``PAD_ACTION = -1`` tripwire and the convention that **the
state token at ``3t+1`` predicts ``a_t``** come from ``agent/DTAgent.py`` and ``offline/dataset.py``
unchanged.  Joint grouping is ``offline.joint_windows.build_joint_index``, which reads
``TrajectoryWindowDataset.item_meta`` rather than index arithmetic.  Nothing here redefines a window.

THE PROTOCOL IS P4'S AND P5.1'S, REUSED RATHER THAN RESTATED
-------------------------------------------------------------
Rollouts go through ``offline.dt_gate.evaluate_arm``; cells come from ``dt_gate._cell``, pairing
from ``dt_gate._paired``, descriptives from ``mean_ci95``, the paired test from
``wilcoxon_signed_rank`` and the comparison object from ``offline_baselines.paired_comparison``.
``bc``, ``bc_top10`` and ``iql`` are trained by ``offline.offline_baselines``' own trainers.  A
second implementation of a protocol is exactly how two arms stop being comparable.

🚨 THE FILESYSTEM-MUTATION BARRIER (docs/plans/p5.2.md section 3.2)
-------------------------------------------------------------------
``output/p5_1/`` holds the ONLY copy of the evidence behind a merged, independently reviewed
result: ``output/`` is gitignored and there is no backup.  **Every write and every delete in this
module goes through :func:`assert_writable`**, which compares ``Path.resolve()`` against the
resolved protected roots — so a relative path, a ``..`` traversal or a symlink cannot pass a string
comparison.  A refused run creates nothing and destroys nothing, and the resume path writes to a
temporary name, validates, and only then moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ANCHOR_BAND",
    "ARTIFACT_FORMAT_VERSION",
    "CONCORDANCE_THRESHOLD",
    "CONTEXT_LENGTH",
    "DECLARATION_FORMAT_VERSION",
    "DECLARED_GRADIENT_STEPS",
    "DT_METHODS",
    "EXPECTED_NODES",
    "EpisodeRef",
    "HARD_SUBSET",
    "HEAD_METHODS",
    "JOINT_BATCH_SIZE",
    "LEVEL_BAND",
    "LEVEL_THRESHOLD",
    "LR_PROBE_EXPECTED",
    "LR_PROBE_STEPS",
    "METHODS",
    "N_HEAD_BY_METHOD",
    "OUT_OF_SAMPLE_CELLS",
    "PREDICTED_LEVELS",
    "RANDOM_SUBSAMPLE_RNG_SEED",
    "REUSED_CELLS",
    "REUSED_TIER",
    "SCENARIO_ID",
    "SCENARIO_KEY",
    "TIERS",
    "TIER_ORDER",
    "TierSpec",
    "assert_cells_identical",
    "assert_reused_digest",
    "assert_writable",
    "cell_is_complete",
    "concordance",
    "episode_key_set",
    "episodes_in",
    "lr_multiplier",
    "parse_sha256sums",
    "per_intersection_top_streams",
    "predicted_order",
    "protected_roots_from",
    "replace_guarded",
    "score_level",
    "selected_episodes",
    "sha256_of",
    "stop_rule_verdict",
    "streams_of_episodes",
    "tier_dirs",
    "tier_spec",
    "warmup_for",
    "write_json_guarded",
]

ARTIFACT_FORMAT_VERSION = "p5.2-tier-sweep/1.0"
DECLARATION_FORMAT_VERSION = "p5.2-declaration/1.0"

SCENARIO_KEY = "cityflow_grid4x4"
SCENARIO_ID = "cityflow_grid4x4"

#: The six method arms.  ``bc_top10_perix`` is the NEW arm declared BESIDE the global filter and
#: never as a re-specification of it (``BRIEF_27`` section 4.3).
METHODS: tuple[str, ...] = (
    "dt_spatial",
    "dt_nomix",
    "bc",
    "bc_top10",
    "bc_top10_perix",
    "iql",
)

#: Every arm this module trains with the joint spatial trainer, at either head count.
DT_METHODS: tuple[str, ...] = ("dt_spatial", "dt_nomix", "dt_spatial_h4", "dt_nomix_h4")

#: The 4-head half of A2's 2x2.  Trained at ``mappo1000`` only; the 1-head half is P5.1's.
HEAD_METHODS: tuple[str, ...] = ("dt_spatial_h4", "dt_nomix_h4")

#: Head count per arm.  ``d_model = 128`` is divisible by both, and both give 853,128 parameters.
N_HEAD_BY_METHOD: Mapping[str, int] = {
    "dt_spatial": 1,
    "dt_nomix": 1,
    "dt_spatial_h4": 4,
    "dt_nomix_h4": 4,
}

BEHAVIOUR_METHOD = "behaviour"
COLLAPSE_REFERENCE_METHOD = "random"

DECLARED_GRADIENT_STEPS = 40_000
CONTEXT_LENGTH = 20
EXPECTED_NODES = 16
JOINT_BATCH_SIZE = 64

#: The RNG that size-matches the ``random`` tier from 400 episodes to 200, one per draw
#: (``docs/plans/p5.2.md`` D10).  Date convention: P4.4 used 20_260_812, P4.6 20_260_813.
RANDOM_SUBSAMPLE_RNG_SEED = 20_260_818

#: The tier whose seven cells are reused verbatim rather than re-run (``BRIEF_27`` A3).
REUSED_TIER = "mappo1000"

#: The seven arms reused from ``output/p5_1/``.  ``bc_top10_perix`` and the two 4-head arms are NOT
#: here: they did not exist in P5.1 and are run at this tier.
REUSED_CELLS: tuple[str, ...] = (
    "dt_spatial",
    "dt_nomix",
    "bc",
    "bc_top10",
    "iql",
    "behaviour",
    "random",
)

#: Tiers in ``cf_grid4x4``'s OWN measured ATT order (``BRIEF_27`` B0: no hz1x1 number may be quoted
#: as grid4x4's).  ``fixedtime`` joined under H1 when E1's zero envelope freed regime (c)'s budget --
#: the contingency B1 and F4 pre-declared, firing as written.
TIER_ORDER: tuple[str, ...] = ("mappo1000", "maxpressure", "fixedtime", "random")


@dataclass(frozen=True)
class TierSpec:
    """One ladder rung: where its episodes live, how many there are, and how it is size-matched."""

    tier: str
    dirs: tuple[str, ...]
    subsample: str
    episode_count: int
    ladder_att: float
    mean_training_return: float


#: The declared tiers.  ``ladder_att`` is read from ``docs/data/att_ladder_v11.json`` and
#: ``mean_training_return`` was computed from the raw ``.npz`` while writing the plan; both are
#: recorded so a figure can be ordered by measurement rather than by tier name.
TIERS: Mapping[str, TierSpec] = {
    "mappo1000": TierSpec(
        tier="mappo1000",
        dirs=tuple(f"cf_grid4x4__mappo1000__seed{seed}" for seed in (101, 202, 303, 404, 505)),
        subsample="none",
        episode_count=200,
        ladder_att=160.33195198059082,
        mean_training_return=-165.6,
    ),
    "maxpressure": TierSpec(
        tier="maxpressure",
        dirs=("cf_grid4x4__maxpressure",),
        subsample="none",
        episode_count=200,
        ladder_att=167.49201210021974,
        mean_training_return=-251.5,
    ),
    "random": TierSpec(
        tier="random",
        dirs=("cf_grid4x4__random",),
        subsample="one_per_draw",
        episode_count=200,
        ladder_att=257.7325928115845,
        mean_training_return=-918.6,
    ),
    "fixedtime": TierSpec(
        tier="fixedtime",
        dirs=("cf_grid4x4__fixedtime",),
        subsample="none",
        episode_count=200,
        ladder_att=206.93176498413087,
        mean_training_return=-492.6,
    ),
}

# ----------------------------------------------------------------------
# The registered predictions (docs/plans/p5.2.md section 4).  These predate every P5.2 number and
# are pinned here so the scorer cannot be written to whatever the campaign returns.
# ----------------------------------------------------------------------

#: Rule R' applied to every arm x tier cell.  ``mappo1000``'s method entries other than
#: ``bc_top10_perix`` are P5.1's MEASURED values, reused, and are not predictions.
PREDICTED_LEVELS: Mapping[str, Mapping[str, float]] = {
    "dt_spatial": {
        "mappo1000": 197.4126, "maxpressure": 208.2667,
        "fixedtime": 256.3101, "random": 312.9006,
    },
    "dt_nomix": {
        "mappo1000": 157.8477, "maxpressure": 166.5265,
        "fixedtime": 204.9412, "random": 250.1900,
    },
    "bc": {
        "mappo1000": 168.9806, "maxpressure": 175.7248,
        "fixedtime": 222.1050, "random": 272.1254,
    },
    "bc_top10": {
        "mappo1000": 749.5796, "maxpressure": 751.0633,
        "fixedtime": 1042.7531, "random": 1242.6948,
    },
    "bc_top10_perix": {
        "mappo1000": 165.7657, "maxpressure": 166.0938,
        "fixedtime": 230.5996, "random": 274.8156,
    },
    "iql": {
        "mappo1000": 275.8354, "maxpressure": 264.0804,
        "fixedtime": 360.8493, "random": 414.9930,
    },
    "behaviour": {
        "mappo1000": 160.2780, "maxpressure": 167.4920,
        "fixedtime": 206.9318, "random": 260.3602,
    },
}

#: The 13 cells scored by Q1.  Seen cells are excluded so a free hit cannot enter the denominator
#: (``BRIEF_27`` B5.2); ``behaviour@maxpressure`` is scored separately as Q1b's instrument check.
OUT_OF_SAMPLE_CELLS: tuple[tuple[str, str], ...] = tuple(
    (method, tier)
    for tier in ("maxpressure", "fixedtime", "random")
    for method in METHODS
) + (("bc_top10_perix", "mappo1000"),)

#: Q1's band, fixed in the commit that registered it and NOT widenable afterwards.  Calibration
#: behind the choice: median relative error 23.9 %, max 378.5 %, 3 of 5 inside the band.
LEVEL_BAND = 0.30

#: The registered rate: 9 of 13 out-of-sample cells, declared when N was 13.
REGISTERED_LEVEL_RATE = (9, 13)


def level_threshold_for(n_cells: int) -> int:
    """``k = ceil(9/13 * N)`` -- the registered rate carried to any denominator (``BRIEF_27`` I3).

    ⚠️ **This is a carry-across, not a fresh judgement.**  Adding ``fixedtime`` grew the
    out-of-sample set from 13 cells to 19, and moving a threshold after data exists is loosening --
    so the rule that transports it is named and applied mechanically rather than re-decided.

    ✅ **It SELF-CHECKS, which is why it is the right rule rather than a plausible one: at ``N = 13``
    it returns 9, reproducing the originally registered threshold exactly.**  At ``N = 19`` it
    returns **14** (73.68 %); 13/19 is 68.42 %, below the registered 69.23 %, and is excluded.
    """
    import math

    numerator, denominator = REGISTERED_LEVEL_RATE
    return math.ceil(numerator / denominator * int(n_cells))


#: Q1's aggregate threshold at the current denominator.  19 cells -> 14.
LEVEL_THRESHOLD = level_threshold_for(len(OUT_OF_SAMPLE_CELLS))

#: Q1b's tighter band for the behaviour anchor.  P5.1's two anchors landed 0.03 % and 1.0 % away.
ANCHOR_BAND = 0.10

#: Q2b's registered threshold, per out-of-sample tier, over all 15 unordered arm pairs.
CONCORDANCE_THRESHOLD = 12

#: Q2b-hard's subset (``BRIEF_27`` C4): ``iql`` and ``bc_top10`` are predicted far from the rest and
#: touch 9 of the 15 pairs, so the remaining 6 carry the ordering the paper's sentence depends on.
#: NO threshold is registered on this subset -- it is reported as a count out of 6.
HARD_SUBSET: tuple[str, ...] = ("dt_nomix", "bc_top10_perix", "bc", "dt_spatial")

#: C2/D2's probe.  Step 249 is the discriminating point (181.4x a cosine's margin at step 500);
#: step 500 is a clean linear check; step 999 is where the ramp first reaches exactly 1.0.
LR_PROBE_STEPS: tuple[int, ...] = (0, 249, 500, 999, 1000, 1001, 40000)
LR_PROBE_EXPECTED: tuple[float, ...] = (0.001, 0.250000, 0.501, 1.0, 1.0, 1.0, 1.0)


#: I1/J1's envelope replicate: BOTH arms, seed 202, on the ``random`` tier, unconditional.
#: The seed is PRE-DECLARED and is not re-selected after phase B -- E1 chose 202 for its largest
#: per-seed effect, and that principle cannot be applied to a tier whose per-seed spread does not
#: exist until phase B has run, so applying it later would be post-hoc selection on a quantity
#: related to the one being measured.
REPLICATE_TIER = "random"
REPLICATE_SEED = 202
REPLICATE_ARMS: tuple[str, ...] = ("dt_spatial", "dt_nomix")

#: J2(b): the replicate's artifacts carry a DISTINCT key, so no cache, resume branch or report path
#: can serve phase B's own cell in their place -- which would return zero by construction.
REPLICATE_SUFFIX = "_replicate"


def cell_filename(tier: str, method: str, seed: int | None = None, replicate: bool = False) -> str:
    """The evaluation artifact name for one cell.  One place, so no path is spelled twice."""
    tag = "" if seed is None else f"_seed{int(seed)}"
    return f"eval_{tier}_{method}{tag}{REPLICATE_SUFFIX if replicate else ''}.json"


def campaign_cell_manifest() -> tuple[dict[str, Any], ...]:
    """Every cell the campaign must produce, DERIVED FROM THE DECLARED CONSTANTS.

    ⚠️ **This is what the completeness assertion checks against -- never the files on disk.**  A
    campaign that verified itself against its own output would report success for whatever it
    happened to produce; ``PROJECT_PLAN`` section 7 records a campaign that ran *"on to a clean-looking
    end"* with half its output missing.

    🔒 **The two I1/J1 replicate cells are enumerated HERE, before phases A/C/B run (``BRIEF_27``
    K2).**  Their wiring is deliberately not in the campaign script yet -- inventing it while the
    ladder runs is how a replicate ends up re-evaluating the cell it is supposed to be independent of
    -- but listing them means a campaign that lacks them **refuses to report itself complete**, which
    is a mechanism rather than a promise to remember.
    """
    out: list[dict[str, Any]] = []
    for arm in REUSED_CELLS:
        out.append({
            "cell": f"eval_{arm}.json", "tier": REUSED_TIER, "method": arm,
            "role": "reused", "produced_by": "P5.1", "source": "reuse_root",
        })
    for method in HEAD_METHODS:
        out.append({
            "cell": cell_filename(REUSED_TIER, method), "tier": REUSED_TIER, "method": method,
            "role": "head_2x2", "produced_by": "phase A", "source": "work_dir",
        })
    out.append({
        "cell": cell_filename(REUSED_TIER, "bc_top10_perix"), "tier": REUSED_TIER,
        "method": "bc_top10_perix", "role": "new_arm_at_reused_tier",
        "produced_by": "phase C", "source": "work_dir",
    })
    for tier in TIER_ORDER:
        if tier == REUSED_TIER:
            continue
        for method in METHODS:
            out.append({
                "cell": cell_filename(tier, method), "tier": tier, "method": method,
                "role": "ladder", "produced_by": "phase B", "source": "work_dir",
            })
        if tier != "random":
            out.append({
                "cell": cell_filename(tier, BEHAVIOUR_METHOD), "tier": tier,
                "method": BEHAVIOUR_METHOD, "role": "behaviour_anchor",
                "produced_by": "phase B", "source": "work_dir",
            })
    out.append({
        "cell": cell_filename(REUSED_TIER, COLLAPSE_REFERENCE_METHOD),
        "tier": "all", "method": COLLAPSE_REFERENCE_METHOD,
        "role": "shared_collapse_reference",
        "produced_by": "gate 1", "source": "work_dir",
        "note": (
            "one cell serves every tier and doubles as the random tier's behaviour anchor: env "
            "settings are identical across all four tier manifests (verified) and the factory is a "
            "function of the seed alone (docs/plans/p5.2.md D12)"
        ),
    })
    for method in REPLICATE_ARMS:
        out.append({
            "cell": cell_filename(REUSED_TIER, method, REPLICATE_SEED), "tier": REUSED_TIER,
            "method": method, "seed": REPLICATE_SEED, "role": "envelope_replicate_E1",
            "produced_by": "phase E1", "source": "work_dir",
            "note": (
                "one seed, so 100 episodes and not 500; the `seed` field is what tells the "
                "completeness assertion that, and omitting it made two correct cells report as "
                "defective"
            ),
        })
    for method in REPLICATE_ARMS:
        out.append({
            "cell": cell_filename(REPLICATE_TIER, method, REPLICATE_SEED, replicate=True),
            "tier": REPLICATE_TIER, "method": method, "seed": REPLICATE_SEED,
            "role": "envelope_replicate_I1_J1", "produced_by": "NOT YET WIRED",
            "source": "work_dir",
            "note": (
                "BRIEF_27 K2: enumerated before phases A/C/B so the completeness assertion refuses "
                "a campaign that lacks it. Distinct key per J2(b); it is an independent training "
                "run from scratch, never a re-evaluation of phase B's checkpoint"
            ),
        })
    return tuple(out)


@dataclass(frozen=True)
class EpisodeRef:
    """One episode of a tier, identified the way the loader identifies it."""

    dataset_dir: str
    episode_file: str
    episode_index: int
    flow_draw: int

    @property
    def key(self) -> tuple[str, str]:
        """Identity of the episode: directory and file name."""
        return (self.dataset_dir, self.episode_file)


# ----------------------------------------------------------------------
# D1 -- the filesystem-mutation barrier.  One helper; every write and delete goes through it.
# ----------------------------------------------------------------------


def protected_roots_from(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    """Resolve *paths* into the read-only roots this task may never write under.

    Resolution happens here, once, so :func:`assert_writable` compares two already-resolved paths
    and can never be handed a root that is itself a symlink.
    """
    return tuple(Path(p).resolve() for p in paths)


def assert_writable(path: str | Path, protected: Sequence[Path]) -> Path:
    """Return the resolved *path*, or raise if it lies at or under a protected root.

    The comparison is on ``Path.resolve()`` and on path COMPONENTS, never on the string: a
    relative path, a ``..`` traversal and a symlink all resolve into the protected root and are
    refused, while a sibling whose name merely starts with the root's -- ``p5_1_backup`` -- is not.
    """
    resolved = Path(path).resolve()
    for root in protected:
        if resolved == root or root in resolved.parents:
            raise PermissionError(
                f"{resolved} lies at or under {root}, which is read-only to this task. "
                "P5.1's cells and checkpoints are the only copy of the evidence behind a merged, "
                "independently reviewed result: output/ is gitignored and there is no backup. "
                "This task reads them and never opens one for writing "
                "(docs/plans/p5.2.md section 3.2)"
            )
    return resolved


def write_json_guarded(
    payload: Mapping[str, Any], path: str | Path, protected: Sequence[Path]
) -> None:
    """``dt_gate.write_json_atomic`` behind the barrier: guarded, then validated, then written.

    The guard runs before anything is opened or created, so a refused write leaves the tree
    byte-for-byte as it was -- no file, no directory, no truncation.
    """
    from offline.dt_gate import write_json_atomic

    destination = assert_writable(path, protected)
    write_json_atomic(dict(payload), destination)


def replace_guarded(
    source: str | Path, destination: str | Path, protected: Sequence[Path]
) -> None:
    """``os.replace`` behind the barrier, for the write-validate-move resume path.

    BOTH ends are guarded, because a move is a write at the destination **and** a delete at the
    source, and both are checked before either happens.
    """
    import os

    origin = assert_writable(source, protected)
    target = assert_writable(destination, protected)
    os.replace(origin, target)


# ----------------------------------------------------------------------
# Tiers, and B4's size match.
# ----------------------------------------------------------------------


def tier_spec(tier: str) -> TierSpec:
    """The declared spec of *tier*, refusing an undeclared name."""
    key = str(tier)
    if key not in TIERS:
        raise ValueError(
            f"unknown tier {key!r}; this task declares {sorted(TIERS)}. The running order is "
            f"{list(TIER_ORDER)} and 'fixedtime' is the pre-declared optional fourth tier"
        )
    return TIERS[key]


def tier_dirs(spec: TierSpec, corpus_root: str | Path) -> tuple[Path, ...]:
    """The tier's collected dataset directories, refusing a missing one."""
    root = Path(corpus_root)
    out: list[Path] = []
    for name in spec.dirs:
        directory = root / name
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(
                f"{directory} is not a collected dataset directory (no manifest.json); tier "
                f"{spec.tier!r} needs all {len(spec.dirs)} directory/ies and a partial tier may "
                "not train"
            )
        out.append(directory)
    return tuple(out)


def episodes_in(dataset: Any) -> tuple[EpisodeRef, ...]:
    """Every episode the dataset carries, read from ``item_meta`` and not from a layout.

    Same rule as ``joint_windows.build_joint_index``: the loader's own answer to *what is this
    row* is the only source of provenance, because index arithmetic keeps producing plausible
    answers after a change to selection or ordering.
    """
    seen: dict[tuple[str, str], EpisodeRef] = {}
    for item in range(len(dataset)):
        meta = dataset.item_meta(item)
        key = (str(meta.dataset_dir), str(meta.episode_file))
        if key not in seen:
            seen[key] = EpisodeRef(
                dataset_dir=str(meta.dataset_dir),
                episode_file=str(meta.episode_file),
                episode_index=int(meta.episode_index),
                flow_draw=int(meta.flow_draw),
            )
    return tuple(seen[key] for key in sorted(seen))


def selected_episodes(
    spec: TierSpec,
    episodes: Sequence[EpisodeRef],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[EpisodeRef, ...]:
    """The tier's size-matched training episodes: all of them, or one per draw (B4).

    Pure in *episodes* so the selection can be tested without loading a 400-episode tier; the
    caller composes it as ``selected_episodes(spec, episodes_in(dataset))``.

    ⚠️ UNIT: the selection is over EPISODES, not streams.  P4.6's ``one_per_draw`` selected one
    stream per draw because on ``cf_hz1x1`` a stream IS an episode.  Here a stream is an
    ``(episode, intersection)`` pair, and 200 of those would be 12.5 episodes' worth of nodes --
    from which ``build_joint_index`` could construct no joint window at all, because it refuses any
    decision instant missing an intersection.  200 episodes x 16 nodes = 3,200 streams is the size
    match P4.6's rule expresses (``docs/plans/p5.2.md`` section 2.3).
    """
    if spec.subsample == "none":
        chosen = tuple(sorted(episodes, key=lambda e: e.key))
    elif spec.subsample == "one_per_draw":
        generator = rng if rng is not None else np.random.default_rng(RANDOM_SUBSAMPLE_RNG_SEED)
        by_draw: dict[int, list[EpisodeRef]] = {}
        for episode in episodes:
            by_draw.setdefault(int(episode.flow_draw), []).append(episode)
        picked: list[EpisodeRef] = []
        for draw in sorted(by_draw):
            candidates = sorted(by_draw[draw], key=lambda e: e.key)
            picked.append(candidates[int(generator.integers(0, len(candidates)))])
        chosen = tuple(picked)
    else:
        raise ValueError(
            f"{spec.tier}: unknown subsample rule {spec.subsample!r}; the declared rules are "
            "'none' and 'one_per_draw'"
        )
    if len(chosen) != int(spec.episode_count):
        raise ValueError(
            f"{spec.tier}: the spec declares {int(spec.episode_count)} training episodes but the "
            f"selection holds {len(chosen)}. Every tier trains on the same number of episodes "
            "(docs/plans/p5.2.md D10), so tier is not confounded with training-set size, and a "
            "tier that cannot supply them may not enter the sweep"
        )
    return chosen


def streams_of_episodes(
    streams: Sequence[Any], episodes: Sequence[EpisodeRef]
) -> tuple[Any, ...]:
    """Every stream belonging to *episodes*, in the order *streams* already has."""
    wanted = {episode.key for episode in episodes}
    return tuple(
        stream
        for stream in streams
        if (str(stream.dataset_dir), str(stream.episode_file)) in wanted
    )


def per_intersection_top_streams(
    streams: Sequence[Any], fraction: float = 0.10
) -> tuple[Any, ...]:
    """``top_return_streams``' rule applied WITHIN each intersection's own stream set.

    ``ceil(fraction * n_ix)`` streams per intersection, never fewer than one, ordered by
    descending return with ties broken by ``(dataset_dir, episode_file, ix_id)`` -- the same rule
    and the same tie-break as ``offline_baselines.top_return_streams``, applied per node.

    ⚠️ THIS IS A NEW ARM, NOT A RE-SPECIFICATION of the global filter, which stays in the study and
    is reported beside it (``BRIEF_27`` section 4.3).  On a tier whose intersections carry equal
    stream counts it keeps the SAME TOTAL as the global filter -- 20 x 16 = 320 on a 200-episode
    grid4x4 tier -- so the two arms differ in WHICH streams they keep and not in how many, which is
    what makes the contrast controlled.

    Why it exists: per-intersection returns on a network are dominated by LOAD rather than by
    control quality, so a GLOBAL return quantile is a load sorter.  Measured on every grid4x4 tier
    while writing the plan, the global decile draws from only 6-11 of the 16 nodes.
    """
    import math

    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")
    if not streams:
        raise ValueError("this dataset has no streams to filter")
    grouped: dict[str, list[Any]] = {}
    for stream in streams:
        grouped.setdefault(str(stream.ix_id), []).append(stream)
    kept: list[Any] = []
    for ix_id in sorted(grouped):
        members = grouped[ix_id]
        keep = max(1, math.ceil(value * len(members)))
        ordered = sorted(
            members,
            key=lambda s: (
                -float(s.total_return),
                str(s.dataset_dir),
                str(s.episode_file),
                str(s.ix_id),
            ),
        )
        kept.extend(ordered[:keep])
    return tuple(kept)


# ----------------------------------------------------------------------
# C2 / D2 -- the schedule the campaign actually runs, computed rather than trained.
# ----------------------------------------------------------------------


def warmup_for(total: int) -> int:
    """The warmup length for a budget of *total* steps.

    ``WARMUP_STEPS`` is IMPORTED from ``offline.spatial_mixing`` rather than restated, so the two
    trainers cannot drift on the constant -- only on the formula, which is what obligation 6b's
    probe compares.  Warmup is a function of the BUDGET: 100 -> 50, 500 -> 250, 40,000 -> 1000.
    """
    from offline.spatial_mixing import WARMUP_STEPS

    return min(int(WARMUP_STEPS), max(1, int(total) // 2))


def lr_multiplier(step: int, warmup: int) -> float:
    """The learning-rate multiplier at *step* under a *warmup*-step linear ramp.

    ``min(1.0, (step + 1) / warmup)``.  The ``+ 1`` is not cosmetic: it is why step 0 is
    ``1/warmup`` rather than 0, and it is what the registered probe pins at step 0.
    """
    return min(1.0, (int(step) + 1) / int(warmup))


# ----------------------------------------------------------------------
# F6 -- the numerical regime.  A launch parameter, acting at PROCESS ENTRY.
# ----------------------------------------------------------------------

#: The cuBLAS workspace settings that make its reductions reproducible.  PyTorch requires one of
#: these to be exported BEFORE the CUDA context is created; there is no way to set it afterwards.
ACCEPTED_CUBLAS_WORKSPACE_CONFIGS: tuple[str, ...] = (":4096:8", ":16:8")


def configure_determinism(deterministic: bool) -> dict[str, Any]:
    """Put THIS PROCESS into the requested numerical regime, at entry, before any CUDA work.

    ⚠️ ``torch.use_deterministic_algorithms`` is PROCESS-GLOBAL and ``CUBLAS_WORKSPACE_CONFIG``
    must be exported **before the CUDA context exists** (``BRIEF_27`` F6a).  ``deterministic`` is
    therefore the right *interface* and cannot be a per-call *parameter*: this function is called
    once, from ``main``, and :func:`train_tier_dt` only ever asserts that the process is in the
    state it was asked for.

    **A flag that silently fails to take effect is worse than no flag**, because it produces a run
    that believes it is reproducible and is not.  So both preconditions are REFUSALS: the
    environment variable must already be set to an accepted value, and CUDA must not be initialised.

    Measured on this machine, campaign shape, 100 steps: default CUDA does **not** reproduce itself
    (63/66 and 61/66 tensors differ, worst |dw| 3.774e-06); with this flag it reproduces exactly
    (0/66), at +10.3 % wall clock.  Determinism does NOT make the method stable -- it makes our
    numbers reproducible (``docs/plans/p5.2.md`` section 4.7).
    """
    import os

    import torch

    if not deterministic:
        return {
            "deterministic": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "note": (
                "default numerical regime: the same regime P5.1's merged cells were trained in, "
                "and NOT reproducible run to run on CUDA -- the envelope is measured by E1"
            ),
        }
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in ACCEPTED_CUBLAS_WORKSPACE_CONFIGS:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG is {configured!r} and deterministic training needs one of "
            f"{list(ACCEPTED_CUBLAS_WORKSPACE_CONFIGS)}. It must be exported BEFORE this process "
            "started -- offline/campaigns/p5_2.sh exports it -- because cuBLAS reads it when the "
            "CUDA context is created and setting it now would not take effect"
        )
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        raise RuntimeError(
            "the CUDA context is already initialised, so requesting determinism now would not "
            "reliably take effect; configure_determinism must run at process entry, before any "
            "tensor reaches the device"
        )
    torch.use_deterministic_algorithms(True)
    return {
        "deterministic": True,
        "cublas_workspace_config": configured,
        "note": (
            "deterministic regime: reproducible run to run on CUDA, measured 0/66 tensors "
            "differing at campaign shape, at +10.3 % wall clock"
        ),
    }


def assert_process_regime(deterministic: bool) -> None:
    """Refuse to train unless the process is in the regime the caller asked for."""
    import torch

    actual = bool(torch.are_deterministic_algorithms_enabled())
    if actual != bool(deterministic):
        raise RuntimeError(
            f"this process has deterministic algorithms {'on' if actual else 'off'} but the run "
            f"asked for {'on' if deterministic else 'off'}. The regime is set once at process "
            "entry by configure_determinism(); it is not a per-call toggle, and training under a "
            "regime the checkpoint will not record is how a cell becomes unattributable"
        )


# ----------------------------------------------------------------------
# Obligation 6 -- the trainer.  Tier- and head-parameterised; equivalent to P5.1's at 1 head.
# ----------------------------------------------------------------------


def train_tier_dt(
    stacked: Mapping[str, Any],
    index: Any,
    *,
    tier: str,
    method: str,
    seed: int,
    adjacency: Any,
    prompts: Mapping[str, Any],
    stats: Any,
    state_dim: int,
    n_actions: int,
    gradient_steps: int,
    batch_size: int,
    device: Any,
    checkpoint_path: str | Path,
    protected: Sequence[Path],
    provenance: Mapping[str, Any],
    deterministic: bool = False,
    log_every: int = 0,
) -> Any:
    """Train one spatial arm at one tier and head count, and save the checkpoint.

    ⚠️ **This is ``spatial_mixing.train_spatial_dt`` with two axes added -- the TIER and the HEAD
    COUNT -- and nothing else.**  Every constant it uses is IMPORTED from that module rather than
    restated, so the two cannot drift on a value; obligation 6 tests that they do not drift on the
    FORM either, by requiring exact equality of the loss sequence and every ``state_dict`` tensor
    at ``n_head = 1`` on CPU, where the C1 control proved reproduction is available.

    ⚠️ **What obligation 6 licenses and what it does not:** it licenses the CODE PATH -- this
    trainer is P5.1's at one head.  It says nothing about artifacts; B3's digest check and the
    ``random``-anchor re-roll license those.  Neither substitutes for the other.
    """
    import time

    import torch

    from agent.DTAgent import action_loss
    from agent.SpatialDTAgent import (
        SPATIAL_CHECKPOINT_FORMAT_VERSION,
        SpatialDecisionTransformer,
        SpatialDTConfig,
    )
    from agent.utils.utils import Utils
    from offline.spatial_mixing import GRAD_CLIP, LEARNING_RATE, WEIGHT_DECAY

    if method not in DT_METHODS:
        raise ValueError(
            f"{method!r} is not a spatial arm; this trainer builds {list(DT_METHODS)}. The "
            "per-intersection comparators are trained by offline.offline_baselines, imported "
            "rather than duplicated"
        )
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; this task declares {sorted(TIERS)}")
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
    assert_process_regime(deterministic)
    # Filesystem-mutation barrier: the destination is guarded BEFORE any work, so a refused run
    # trains nothing and writes nothing (docs/plans/p5.2.md section 3.2).
    destination = assert_writable(checkpoint_path, protected)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}; nothing is created here"
        )

    n_head = int(N_HEAD_BY_METHOD[method])
    spatial = method in ("dt_spatial", "dt_spatial_h4")
    mask = adjacency.attention_mask(spatial_mixing=spatial)

    Utils.seed_everything(int(seed), seed_python_random=False)
    config = SpatialDTConfig(
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        n_nodes=index.n_nodes,
        context_length=int(stacked["state"].shape[1]),
        max_ep_len=int(stacked["timestep"].max()) + 1,
        spatial_mixing=spatial,
        n_head=n_head,
    )
    model = SpatialDecisionTransformer(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    warmup = warmup_for(total)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: lr_multiplier(step, warmup)
    )

    members = stacked["member_index"]
    count = int(members.shape[0])
    if count < 1:
        raise ValueError("the joint index is empty")
    mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.bool_)).to(device)
    scale = (
        torch.tensor([prompts[ix].rtg_scale for ix in index.node_ids], dtype=torch.float32)
        .view(1, index.n_nodes, 1, 1)
        .to(device)
    )

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
                f"  {tier}/{method} seed {seed} step {step + 1}/{total} "
                f"loss {np.mean(losses[-log_every:]):.5f}",
                flush=True,
            )
    seconds = time.time() - started

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
                "tier": tier,
                "n_head": n_head,
                "deterministic": bool(deterministic),
                "seed": int(seed),
                "gradient_steps": int(total),
                "batch_size": int(batch_size),
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": int(warmup),
                "grad_clip": GRAD_CLIP,
                "device": str(device),
                "spatial_mixing": spatial,
                "roadnet_sha256": adjacency.roadnet_sha256,
            },
        },
        destination,
    )
    from offline.spatial_mixing import TrainResult

    return TrainResult(
        method=method,
        seed=int(seed),
        gradient_steps=total,
        losses=tuple(losses),
        checkpoint_path=str(destination),
        seconds=float(seconds),
    )


# ----------------------------------------------------------------------
# B3 -- the reuse gate.
# ----------------------------------------------------------------------


def sha256_of(path: str | Path) -> str:
    """The SHA-256 hex digest of a file's bytes."""
    import hashlib

    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse a ``sha256sum`` manifest into ``{relative path: digest}``."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if not name:
            raise ValueError(
                f"cannot parse sha256sum line {line!r}; the expected format is "
                "'<digest>  <path>' with two spaces, which is what sha256sum writes"
            )
        out[name.strip()] = digest.strip()
    return out


def assert_reused_digest(
    path: str | Path, expected: str, *, relative_name: str | None = None
) -> str:
    """Re-verify a reused artifact's digest AT CONSUMPTION, returning what was measured.

    ``BRIEF_27`` B3(a): a digest checked once is not a digest checked when used.  The whole point
    is that this runs in the campaign, against the file the campaign is about to read, rather than
    once at securing time.
    """
    measured = sha256_of(path)
    if measured != str(expected):
        raise ValueError(
            f"{relative_name or path}: sha256 digest {measured} does not match the recorded "
            f"{expected}. This artifact is being consumed as a reused P5.1 cell and its bytes have "
            "changed since output/SHA256SUMS_p5_1.txt was written; refusing rather than reporting "
            "a number whose provenance cannot be established"
        )
    return measured


def episode_key_set(payload: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    """The ``(seed, draw_id)`` set an evaluation payload covers."""
    return frozenset(
        (int(episode["seed"]), int(episode["draw_id"])) for episode in payload["episodes"]
    )


#: The per-episode fields the reuse gate compares.  ``att_horizon`` is the primary (A1) and
#: ``horizon_vehicle_count`` is A5's unconditional co-report, so a drift in either matters.
_COMPARED_EPISODE_FIELDS: tuple[str, ...] = (
    "att_horizon",
    "horizon_vehicle_count",
    "episode_reward",
)


def assert_cells_identical(
    left: Mapping[str, Any], right: Mapping[str, Any], *, expected_n: int
) -> dict[str, Any]:
    """Refuse unless both cells hold EXACTLY *expected_n* episodes and every value matches.

    The non-empty assertion comes FIRST, so ``found no differences`` can never mean ``compared
    nothing`` (``BRIEF_27`` B3b): two empty cells are trivially equal, and a gate that accepted
    them would pass on precisely the failure it exists to detect.

    Comparison is by ``(seed, draw_id)`` key, so a reordered payload is still identical -- episode
    order is not a property of the measurement -- and by ``==`` on the values, because this gate
    tests a CPU-deterministic policy re-rolled on regenerated demand, where exact equality is the
    correct bar.
    """
    for name, payload in (("left", left), ("right", right)):
        count = len(payload["episodes"])
        if count != int(expected_n):
            raise ValueError(
                f"the {name} cell holds {count} episodes and the gate requires exactly "
                f"{int(expected_n)}; refusing to report agreement over a set that is not the one "
                "declared, because 'found no differences' must never mean 'compared nothing'"
            )
    left_keys, right_keys = episode_key_set(left), episode_key_set(right)
    if len(left_keys) != int(expected_n) or left_keys != right_keys:
        missing = sorted(left_keys - right_keys)[:8]
        unexpected = sorted(right_keys - left_keys)[:8]
        raise ValueError(
            f"the two cells cover different (seed, draw) sets or carry duplicates: "
            f"{len(left_keys)} distinct keys on the left against {len(right_keys)} on the right, "
            f"missing {missing}, unexpected {unexpected}"
        )
    by_key = {
        (int(e["seed"]), int(e["draw_id"])): e for e in right["episodes"]
    }
    differences: list[str] = []
    for episode in left["episodes"]:
        key = (int(episode["seed"]), int(episode["draw_id"]))
        other = by_key[key]
        for field in _COMPARED_EPISODE_FIELDS:
            if float(episode[field]) != float(other[field]):
                differences.append(
                    f"{key} {field}: {float(episode[field])!r} != {float(other[field])!r}"
                )
    if differences:
        raise ValueError(
            f"{len(differences)} value(s) differ between the two cells over "
            f"{int(expected_n)} compared episodes; the first are {differences[:5]}. The reuse "
            "gate REFUSES AND STOPS rather than falling back to re-running, because a fallback "
            "hides exactly the drift the gate exists to detect (BRIEF_27 B3c)"
        )
    return {
        "n_compared": int(expected_n),
        "fields": list(_COMPARED_EPISODE_FIELDS),
        "n_values_compared": int(expected_n) * len(_COMPARED_EPISODE_FIELDS),
    }


# ----------------------------------------------------------------------
# D1(c) -- resume, and the half-written cell it must never mistake for a complete one.
# ----------------------------------------------------------------------


def cell_is_complete(
    payload: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    draws: Sequence[int],
    declared_steps: int,
    method: str,
) -> bool:
    """Whether an evaluation payload is a finished cell at the declared budget.

    Completeness is a SET identity over ``(seed, draw_id)`` and not a count: a cell holding one
    episode twice and another not at all has the right length and is not complete.  The arm and the
    budget are checked too, because a rehearsal at 3 steps, or the other arm's file, would
    otherwise be resumed straight into a reported cell.

    Used by the resume path, which is why it is conservative in one direction only: anything it
    cannot positively confirm is INCOMPLETE, and an incomplete cell is re-run rather than trusted.
    """
    if str(payload.get("method")) != str(method):
        return False
    if int(payload.get("declared_gradient_steps", -1)) != int(declared_steps):
        return False
    expected = {(int(s), int(d)) for s in seeds for d in draws}
    episodes = payload.get("episodes", [])
    if len(episodes) != len(expected):
        return False
    return episode_key_set(payload) == expected


# ----------------------------------------------------------------------
# The registered scorers.  Every threshold they read is a module constant declared above.
# ----------------------------------------------------------------------


def score_level(
    measured: Mapping[tuple[str, str], float],
    *,
    predicted: Mapping[str, Mapping[str, float]] = PREDICTED_LEVELS,
    cells: Sequence[tuple[str, str]] = OUT_OF_SAMPLE_CELLS,
    band: float = LEVEL_BAND,
    threshold: int = LEVEL_THRESHOLD,
) -> dict[str, Any]:
    """Score Q1: per-cell relative error against the band, then ``k of N`` against the threshold.

    The per-cell rule is ``|measured - predicted| / predicted <= band`` -- SYMMETRIC, so an arm
    that beats its prediction by more than the band misses it just as one that misses upward does.
    Registered that way deliberately: a one-sided rule would score a collapse as a failure of the
    prediction and an improvement as a success of it.

    Only *cells* are scored, and the reused ``mappo1000`` cells are not among them, so a seen cell
    cannot enter the denominator even if the caller supplies it (``BRIEF_27`` B5.2).
    """
    scored: list[dict[str, Any]] = []
    missing = [cell for cell in cells if cell not in measured]
    if missing:
        raise ValueError(
            f"no measurement for {missing[:8]}; every registered out-of-sample cell must be "
            "present before the aggregate is scored, because a missing cell would silently "
            "shrink the denominator the threshold is stated against"
        )
    for arm, tier in cells:
        expected = float(predicted[arm][tier])
        actual = float(measured[(arm, tier)])
        relative = abs(actual - expected) / expected
        scored.append(
            {
                "cell": [arm, tier],
                "predicted": expected,
                "measured": actual,
                "relative_error": relative,
                "held": bool(relative <= float(band)),
            }
        )
    n_held = sum(1 for cell in scored if cell["held"])
    return {
        "rule": (
            "|measured - predicted| / predicted <= band, per cell; the aggregate HOLDS iff at "
            "least `threshold` of the enumerated out-of-sample cells hold"
        ),
        "band": float(band),
        "threshold": int(threshold),
        "n_cells": len(scored),
        "n_held": n_held,
        "outcome": "HELD" if n_held >= int(threshold) else "FAILED",
        "cells": scored,
    }


def predicted_order(tier: str, arms: Sequence[str] = METHODS) -> tuple[str, ...]:
    """The predicted ordering of *arms* on *tier*, best (lowest ATT) first.

    Derived from the registered table rather than retyped, so the ordering and the levels cannot
    disagree with each other.
    """
    return tuple(sorted(arms, key=lambda arm: float(PREDICTED_LEVELS[arm][tier])))


def concordance(
    predicted: Sequence[str], measured: Sequence[str], *, subset: Sequence[str] | None = None
) -> dict[str, Any]:
    """Count the unordered arm pairs ordered as predicted, optionally within *subset* only.

    ``subset`` implements C4's hard-subset report: ``iql`` and ``bc_top10`` are predicted far from
    the other four and touch **9 of the 15** pairs, so a 15-pair count can read 9/15 while every
    ordering the paper's sentence depends on is reversed.  No threshold is registered on the
    subset -- it is reported as a count out of 6 beside the registered 15-pair count.
    """
    import itertools

    keep = set(subset) if subset is not None else None
    left = [arm for arm in predicted if keep is None or arm in keep]
    right = [arm for arm in measured if keep is None or arm in keep]
    if set(left) != set(right):
        raise ValueError(
            f"the predicted and measured orderings cover different arms: "
            f"{sorted(set(left) ^ set(right))}"
        )
    rank_predicted = {arm: i for i, arm in enumerate(left)}
    rank_measured = {arm: i for i, arm in enumerate(right)}
    pairs = list(itertools.combinations(sorted(left), 2))
    concordant = [
        (a, b)
        for a, b in pairs
        if (rank_predicted[a] < rank_predicted[b]) == (rank_measured[a] < rank_measured[b])
    ]
    return {
        "n_pairs": len(pairs),
        "n_concordant": len(concordant),
        "arms": list(left),
        "discordant": [list(pair) for pair in pairs if pair not in set(concordant)],
    }


def canonical_state_dict_digest(checkpoint_path: str | Path) -> str:
    """SHA-256 over the WEIGHTS alone, keys sorted -- not the file, which carries provenance.

    ``DEFERRED`` 29 records this from the other side: a checkpoint's FILE hash depends on its
    filename and provenance block, so *"reproduces byte-identically"* is not testable at file level.
    E1 met that live -- the two runs' file hashes differed on ``git_commit``, ``deterministic``,
    ``n_head`` and a changed ``tier`` label while saying nothing about the weights.

    This digests ``state_dict`` tensors in sorted-key order, as raw bytes with their dtype and
    shape, so two checkpoints agree here **iff their weights agree**.
    """
    import hashlib

    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = payload["model"]
    digest = hashlib.sha256()
    for key in sorted(model):
        tensor = model[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_independent_replicate(
    replicate_checkpoint: str | Path, original_checkpoint: str | Path
) -> dict[str, str]:
    """J2(c): refuse to report an envelope unless the two runs produced DIFFERENT weights.

    🚨 **EQUAL digests are a REFUSAL, not a zero.**  They mean either that two identical models were
    compared -- a defect, and the one a re-evaluation of the same checkpoint would produce **by
    construction** -- or that training reproduced exactly, which would contradict the C1 control and
    invalidate the measurement's premise.  Either way it forces a look instead of silently reporting
    the answer the question was asked to avoid.

    Derived from what happened in E1: its ``+0.0000`` was only interpretable because the two
    checkpoints were checked BY HAND and found to differ (66/66 tensors, worst 1.22e-04).  F7 did
    not require that, and without it the result was indistinguishable from a self-comparison.  The
    machinery now does what was done manually.
    """
    replicate = canonical_state_dict_digest(replicate_checkpoint)
    original = canonical_state_dict_digest(original_checkpoint)
    if replicate == original:
        raise ValueError(
            f"the replicate and the original share a canonical state_dict digest ({replicate}). "
            "An envelope may not be reported from this pair: either the same checkpoint was "
            "compared with itself -- which returns zero by construction -- or training reproduced "
            "exactly, which contradicts the C1 control and invalidates the premise. Refusing "
            "rather than reporting the answer the question was asked to avoid (BRIEF_27 J2c)"
        )
    return {
        "replicate_state_dict_sha256": replicate,
        "original_state_dict_sha256": original,
        "note": (
            "canonical digest over sorted state_dict tensor bytes, NOT the file sha256, which "
            "differs on provenance alone (DEFERRED 29)"
        ),
    }


def episodes_of_seed(payload: Mapping[str, Any], seed: int) -> dict[int, dict[str, float]]:
    """``{draw_id: episode}`` for one training seed, refusing duplicate draws."""
    out: dict[int, dict[str, float]] = {}
    for episode in payload["episodes"]:
        if int(episode["seed"]) != int(seed):
            continue
        draw = int(episode["draw_id"])
        if draw in out:
            raise ValueError(
                f"draw {draw} appears twice for seed {seed}; a paired comparison needs one "
                "episode per draw and a duplicate would silently weight it twice"
            )
        out[draw] = episode
    if not out:
        raise ValueError(f"no episodes for seed {seed}; nothing to pair")
    return out


def paired_replicate_report(
    replicate: Mapping[str, Any],
    published: Mapping[str, Any],
    *,
    seed: int,
    metric: str = "att_horizon",
) -> dict[str, Any]:
    """E1/F7: the paired per-draw difference between a replicate cell and P5.1's own, at one seed.

    Two point estimates cannot be judged -- ``72.07`` against ``68.5`` and against ``20.1`` look the
    same on the page until the per-draw scatter is known.  This returns ``mean(replicate -
    published)`` over the SHARED draw ids with its 95 % CI and whether that interval excludes zero.

    🚨 **If the interval excludes zero, two independent training runs of the same code at the same
    seed produced measurably different policies.** That is a finding about our own reproducibility
    and it is reported as one, not reconciled.

    ⭐ The attribution is clean because obligation 6 proved ``train_tier_dt`` reproduces
    ``spatial_mixing.train_spatial_dt`` byte-exactly at one head on CPU: any difference measured here
    on CUDA is **device nondeterminism alone**, not the trainer change.
    """
    from offline.dt_gate import mean_ci95

    left = episodes_of_seed(replicate, seed)
    right = episodes_of_seed(published, seed)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError(
            f"the two cells share no draw ids at seed {seed}: {sorted(left)[:5]} against "
            f"{sorted(right)[:5]}. A5 makes an unshared comparison void"
        )
    if len(shared) != len(left) or len(shared) != len(right):
        raise ValueError(
            f"the two cells cover different draw sets at seed {seed}: {len(left)} and "
            f"{len(right)} episodes with {len(shared)} shared; refusing a partial pairing"
        )
    differences = [float(left[d][metric]) - float(right[d][metric]) for d in shared]
    stats = mean_ci95(differences)
    # mean_ci95 returns a HALF-WIDTH in `ci95`; the interval is mean +/- that.
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    return {
        "metric": metric,
        "seed": int(seed),
        "n_shared_draws": len(shared),
        "mean_difference": stats.mean,
        "std": stats.std,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_width": high - low,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "mean_replicate": sum(float(left[d][metric]) for d in shared) / len(shared),
        "mean_published": sum(float(right[d][metric]) for d in shared) / len(shared),
    }


def stop_rule_verdict(ci_low: float, ci_high: float) -> str:
    """Q0's stop rule: ``STOP`` iff the CI of ``d4`` lies entirely below zero.

    ``entirely below`` is STRICT.  A CI whose upper end is exactly zero has not resolved, and a
    straddling CI is not a reversal -- the plan says phase B proceeds in that case and the result
    is reported as *at 4 heads the contrast does not resolve*.
    """
    if float(ci_high) < 0.0:
        return "STOP"
    return "CONTINUE"


# ----------------------------------------------------------------------
# The tier's inputs, built once and identically for every command.
# ----------------------------------------------------------------------


def node_ids_from_corpus(spec: TierSpec, corpus_root: str | Path) -> tuple[str, ...]:
    """The controlled intersection order, read from the CORPUS rather than from the roadnet.

    This is the pairing key: it comes from the data the model trains on, and ``derive_adjacency``
    then refuses unless it is exactly the roadnet's controllable set, which is what makes the graph
    provably about these rows (``PROJECT_PLAN`` section 7, 2026-08-16).  P5.1's function of the
    same name is tier-fixed; this one takes the tier, and both read the manifest the same way.
    """
    import json as _json

    from offline.trajectory_logger import load_episode

    directory = tier_dirs(spec, corpus_root)[0]
    manifest = _json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    first = sorted(str(entry["filename"]) for entry in manifest["episodes"])[0]
    episode = load_episode(directory / first)
    ids = tuple(str(ix) for ix in episode.ix_ids)
    if len(ids) != EXPECTED_NODES:
        raise ValueError(
            f"{directory / first} carries {len(ids)} intersections, expected {EXPECTED_NODES}; "
            "grid4x4 is a 16-intersection scenario and a different count means the wrong corpus"
        )
    return ids


def adjacency_for_tier(
    spec: TierSpec, corpus_root: str | Path, node_ids: Sequence[str],
    sim_config: str | Path | None = None,
) -> Any:
    """The graph of the network THIS tier was collected on, taken from its own manifest."""
    import json as _json

    from offline.roadnet_graph import adjacency_from_sim_config

    if sim_config is None:
        manifest = _json.loads(
            (tier_dirs(spec, corpus_root)[0] / "manifest.json").read_text(encoding="utf-8")
        )
        sim_config = manifest["run_metadata"]["env_paths"]["config"]
    return adjacency_from_sim_config(sim_config, node_ids)


def tier_dataset(spec: TierSpec, corpus_root: str | Path, context_length: int) -> Any:
    """The tier's training-split window dataset, statistics fitted on the training split only.

    ``split="train"`` is not a default being accepted: it is the mechanism that makes a held-out
    draw raise instead of quietly entering the corpus.
    """
    from offline.dataset import TrajectoryWindowDataset

    return TrajectoryWindowDataset(
        list(tier_dirs(spec, corpus_root)),
        context_length=int(context_length),
        split="train",
        normalize=True,
    )


def tier_parts(
    tier: str, corpus_root: str | Path, *, sim_config: str | Path | None = None,
    context_length: int = CONTEXT_LENGTH,
) -> dict[str, Any]:
    """Everything every command needs for one tier, built once and identically.

    The size match (B4) happens HERE, before anything downstream sees the corpus, so the DT arms
    and the per-intersection comparators are guaranteed to train on the same episode set.
    """
    from offline.joint_windows import build_joint_index, stack_joint
    from offline.offline_baselines import stream_returns
    from offline.spatial_mixing import per_node_prompts

    spec = tier_spec(tier)
    node_ids = node_ids_from_corpus(spec, corpus_root)
    adjacency = adjacency_for_tier(spec, corpus_root, node_ids, sim_config)
    dataset = tier_dataset(spec, corpus_root, context_length)
    all_episodes = episodes_in(dataset)
    chosen = selected_episodes(spec, all_episodes)
    all_streams = stream_returns(dataset)
    streams = streams_of_episodes(all_streams, chosen)
    prompts = per_node_prompts(streams)

    def stack() -> dict[str, Any]:
        """The joint tensors, restricted to the size-matched episode set."""
        index = build_joint_index(dataset, node_ids)
        stacked = stack_joint(dataset, index)
        keep = {e.episode_index for e in chosen}
        rows = np.array(
            [i for i, ep in enumerate(index.episode_index) if int(ep) in keep], dtype=np.int64
        )
        if rows.size == 0:
            raise ValueError(f"{spec.tier}: the size match selected no joint windows")
        out = dict(stacked)
        out["member_index"] = stacked["member_index"][rows]
        return out

    index = build_joint_index(dataset, node_ids)
    return {
        "spec": spec,
        "node_ids": node_ids,
        "adjacency": adjacency,
        "dataset": dataset,
        "index": index,
        "episodes_all": all_episodes,
        "episodes": chosen,
        "streams": streams,
        "streams_all": all_streams,
        "prompts": prompts,
        "stack": stack,
    }


# ----------------------------------------------------------------------
# The declaration, and the two gates the campaign runs before any training.
# ----------------------------------------------------------------------


def declaration_artifact(parts: Mapping[str, Any], corpus_root: str | Path) -> dict[str, Any]:
    """The full pre-training record for one tier: graph, node order, selection, prompts, budget."""
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
    from offline.roadnet_graph import assert_reproduces_from_roads
    from offline.spatial_mixing import assert_declaration_matches_corpus

    spec: TierSpec = parts["spec"]
    adjacency = parts["adjacency"]
    roads = assert_reproduces_from_roads(adjacency)
    graph = adjacency.to_json_obj()
    graph["roads_route_agrees"] = bool(roads["agrees_with_lane_route"])
    graph["roads_route_directed_edges"] = int(roads["directed_edges"])
    declared = {
        ix: {"target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale}
        for ix, p in parts["prompts"].items()
    }
    check = assert_declaration_matches_corpus(declared, parts["streams"])
    return {
        "format_version": DECLARATION_FORMAT_VERSION,
        "tier": spec.tier,
        "scenario_id": SCENARIO_ID,
        "scenario_key": SCENARIO_KEY,
        "corpus_root": str(Path(corpus_root).resolve()),
        "dataset_dirs": [str(d) for d in tier_dirs(spec, corpus_root)],
        "graph": graph,
        "node_order": list(parts["index"].node_ids),
        "state_dim": int(parts["index"].state_dim),
        "n_actions": int(parts["index"].n_actions),
        "ladder_att": spec.ladder_att,
        "ladder_att_source": (
            "docs/data/att_ladder_v11.json, randomised draws 1-200. PROJECT_PLAN section 10 item 0 "
            "discharges this ladder for figures and for the prediction rule's T(t) term; the "
            "nominal draw-0 comparison and anything from rederive_anchors.py stay blocked. "
            "DEFERRED 27: the ladder tool's JSON-writing path has zero tests, so the artifact's "
            "CONTENT is verified and the PATH that wrote it is not. Corroboration: P5.1's two "
            "anchors landed 0.0540 and 2.6276 ATT (0.03 % and 1.0 %) from these values on "
            "independently rolled held-out draws"
        ),
        "mean_training_return": spec.mean_training_return,
        "subsample": spec.subsample,
        "subsample_rng_seed": (
            RANDOM_SUBSAMPLE_RNG_SEED if spec.subsample == "one_per_draw" else None
        ),
        "episodes_available": len(parts["episodes_all"]),
        "episodes_selected": len(parts["episodes"]),
        "selected_episodes": [
            {"dataset_dir": e.dataset_dir, "episode_file": e.episode_file,
             "episode_index": e.episode_index, "flow_draw": e.flow_draw}
            for e in parts["episodes"]
        ],
        "training_streams": len(parts["streams"]),
        "prompts": {
            ix: {
                "target_rtg": p.target_rtg, "rtg_scale": p.rtg_scale, "n_streams": p.n_streams,
                "return_min": p.return_min, "return_max": p.return_max,
            }
            for ix, p in sorted(parts["prompts"].items())
        },
        "prompt_rule": (
            "target_rtg = max episode return in THIS INTERSECTION's training streams; rtg_scale = "
            "max|return| over the same set (docs/plans/p5.2.md D3, inherited from P5.1's D3)"
        ),
        "declaration_check": check,
        "methods": list(METHODS),
        "head_methods": list(HEAD_METHODS),
        "n_head_by_method": dict(N_HEAD_BY_METHOD),
        "seeds": list(TRAINING_SEEDS),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "batch_size": JOINT_BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "held_out_draws": list(HELD_OUT_DRAWS),
        "statistics_split": parts["dataset"].stats.split,
        "statistics_draw_ids": list(parts["dataset"].stats.draw_ids),
    }


def verify_reuse_gate(
    reuse_root: str | Path, checksums: str | Path, arms: Sequence[str] = REUSED_CELLS
) -> dict[str, Any]:
    """B3(a): re-verify every reused cell's digest AT CONSUMPTION, not once at securing time.

    Returns the source path and measured digest per arm, for the artifact.  Any mismatch RAISES,
    and the campaign stops rather than falling back to re-running (B3c).
    """
    root = Path(reuse_root)
    recorded = parse_sha256sums(Path(checksums).read_text(encoding="utf-8"))
    prefix = root.name
    out: dict[str, Any] = {}
    for arm in arms:
        name = f"{prefix}/eval_{arm}.json"
        if name not in recorded:
            raise ValueError(
                f"{name} is not in {checksums}; every reused cell must carry a recorded digest, "
                "and an unrecorded one cannot be shown to be the reviewed artifact"
            )
        path = root / f"eval_{arm}.json"
        digest = assert_reused_digest(path, recorded[name], relative_name=name)
        out[arm] = {"source": str(path), "sha256": digest}
    return out


# ----------------------------------------------------------------------
# CLI.  The regime is configured ONCE, at process entry, before anything touches CUDA.
# ----------------------------------------------------------------------


def build_parser() -> Any:
    """CLI: ``declare``, ``train``, ``evaluate``, ``verify-reuse``, ``replicate-report``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m offline.tier_sweep",
        description="P5.2: the spatial DT across the grid4x4 ladder, and the head-count 2x2.",
    )
    parser.add_argument("--corpus-root", default="datasets_v11")
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--work-dir", default="output/p5_2")
    parser.add_argument("--checkpoint-dir", default="output/p5_2/checkpoints")
    parser.add_argument(
        "--reuse-root",
        default="/home/filip/rltraffic/output/p5_1",
        help="P5.1's secured cells. READ-ONLY: every write resolving under it is refused.",
    )
    parser.add_argument(
        "--checksums", default="/home/filip/rltraffic/output/SHA256SUMS_p5_1.txt"
    )
    parser.add_argument(
        "--mappo-checkpoint-dir", default="/home/filip/rltraffic/output/checkpoints"
    )
    parser.add_argument("--scenario-key", default=SCENARIO_KEY)
    parser.add_argument("--scenario-id", default=SCENARIO_ID)
    parser.add_argument("--engine-seed", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=DECLARED_GRADIENT_STEPS)
    parser.add_argument("--batch-size", type=int, default=JOINT_BATCH_SIZE)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="run in the deterministic numerical regime (F6: acts at process entry; the campaign "
             "script must export CUBLAS_WORKSPACE_CONFIG before this process starts)",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="comma-separated training seeds; defaults to the registered five. E1 uses '202'.",
    )
    parser.add_argument("--tier", default=None, help=f"one of {sorted(TIERS)}")
    parser.add_argument(
        "--replicate", action="store_true",
        help="I1/J1: write under the DISTINCT replicate key (J2b), so the independent run cannot "
             "be served by, or serve, phase B's own checkpoint and cell",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("declare", help="write the pre-training declaration for one tier")
    train = sub.add_parser("train", help="train one spatial arm")
    train.add_argument("--method", required=True, choices=list(DT_METHODS))
    evaluate = sub.add_parser("evaluate", help="roll one arm over the held-out pool")
    evaluate.add_argument("--method", required=True)
    sub.add_parser(
        "declare-campaign", help="write the campaign-wide declaration of every expected cell"
    )
    sub.add_parser(
        "check-factories",
        help="Gate 0: every declared cell must have a constructible action factory",
    )
    sub.add_parser(
        "assert-complete", help="refuse unless every cell the declaration names exists"
    )
    sub.add_parser(
        "train-baselines", help="train bc, bc_top10, bc_top10_perix and iql on one tier"
    )
    sub.add_parser("verify-reuse", help="B3: re-verify the reused cells' digests at consumption")
    sub.add_parser(
        "stop-rule",
        help="Q0: score d4 from the two 4-head cells; exit 3 (STOP) if its CI is entirely below 0",
    )
    replicate = sub.add_parser(
        "replicate-report", help="E1/F7: the paired per-draw report against P5.1's cells"
    )
    replicate.add_argument("--seed", type=int, default=202)
    envelope = sub.add_parser(
        "envelope-report",
        help="I1/J1: the random-tier envelope, replicate vs phase B, under J2's conditions",
    )
    envelope.add_argument("--seed", type=int, default=REPLICATE_SEED)
    return parser


def _seeds_of(args: Any) -> tuple[int, ...]:
    from offline.dt_gate import TRAINING_SEEDS

    if not args.seeds:
        return tuple(TRAINING_SEEDS)
    return tuple(int(s) for s in str(args.seeds).split(",") if s.strip())


def _protected_of(args: Any) -> tuple[Path, ...]:
    return protected_roots_from([args.reuse_root])


def _require_tier(args: Any) -> str:
    if not args.tier:
        raise SystemExit(f"--tier is required; this task declares {sorted(TIERS)}")
    return str(args.tier)


#: O4: a training record is keyed by these, and a resume MERGES into it rather than overwriting.
TRAINING_RECORD_KEY: tuple[str, ...] = ("tier", "method", "seed")


def training_record_name(tier: str, method: str, replicate: bool = False) -> str:
    """The training record's filename.

    ⚠️ **It carries ``REPLICATE_SUFFIX`` too, and its absence was a J2(b) LEAK that destroyed data.**
    The suffix covered the checkpoint name and the evaluation cell but not this path, so phase R's
    replicate wrote into phase B's record and overwrote it.  A "distinct artifact key" is only
    distinct if it covers every artifact class the run writes.
    """
    return f"training_{tier}_{method}{REPLICATE_SUFFIX if replicate else ''}.json"


def load_existing_runs(path: str | Path) -> list[dict[str, Any]]:
    """The ``runs`` already recorded at *path*, or an empty list if there is no record yet."""
    import json as _json

    target = Path(path)
    if not target.is_file():
        return []
    payload = _json.loads(target.read_text(encoding="utf-8"))
    return [dict(run) for run in payload.get("runs", [])]


def merge_training_runs(
    existing: Sequence[Mapping[str, Any]], produced: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``BRIEF_27`` O4: merge by ``(tier, method, seed)`` instead of overwriting.

    🚨 **THIS EXISTS BECAUSE ITS ABSENCE DESTROYED SIX RECORDS IN AN UN-BACKED-UP TREE.**  The old
    code held only the seeds trained in the CURRENT invocation and wrote ``"runs": records`` over
    whatever was there.  A resume, which skips every seed already on disk, therefore wrote an EMPTY
    list over a complete one -- and that is exactly what happened to all four ``*_baselines.json``
    records, each of which had held 20 runs.

    A run produced now REPLACES the entry with the same key -- a re-run of one seed should update
    it, not duplicate it -- and every other entry is preserved.  The result is sorted so the file
    is stable across invocations and a diff shows real changes rather than reordering.
    """
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in list(existing) + list(produced):
        key = tuple(run.get(field) for field in TRAINING_RECORD_KEY)
        by_key[key] = dict(run)
    return [by_key[key] for key in sorted(by_key, key=lambda k: tuple(str(part) for part in k))]


def _checkpoint_name(tier: str, method: str, seed: int, replicate: bool = False) -> str:
    """J2(b): the replicate's checkpoint carries a DISTINCT key, so no resume branch, cache or
    report path can serve phase B's own checkpoint in its place -- which would make the envelope
    zero by construction."""
    return f"grid4x4_{tier}_{method}_seed{seed}{REPLICATE_SUFFIX if replicate else ''}.pt"


def _run_declare(args: Any) -> int:
    protected = _protected_of(args)
    tier = _require_tier(args)
    parts = tier_parts(tier, args.corpus_root, sim_config=args.sim_config)
    payload = declaration_artifact(parts, args.corpus_root)
    out = Path(args.out_dir) / f"p5_2_declaration_{tier}.json"
    assert_writable(out, protected)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_guarded(payload, out, protected)
    print(f"declaration written to {out}")
    print(
        f"  graph: {payload['graph']['undirected_edges']} undirected edges, "
        f"degrees {payload['graph']['degree_histogram']}, "
        f"roads route agrees: {payload['graph']['roads_route_agrees']}"
    )
    print(
        f"  size match: {payload['episodes_selected']} of {payload['episodes_available']} "
        f"episodes ({payload['subsample']}), {payload['training_streams']} streams"
    )
    return 0


def _run_train(args: Any) -> int:
    import torch

    from offline.dt_gate import runtime_provenance

    protected = _protected_of(args)
    tier = _require_tier(args)
    parts = tier_parts(tier, args.corpus_root, sim_config=args.sim_config)
    stacked = parts["stack"]()
    checkpoints = Path(args.checkpoint_dir)
    assert_writable(checkpoints, protected)
    checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    records: list[dict[str, Any]] = []
    for seed in _seeds_of(args):
        destination = checkpoints / _checkpoint_name(
            tier, args.method, seed, replicate=bool(getattr(args, "replicate", False))
        )
        if destination.is_file():
            print(f"SKIP {tier}/{args.method} seed {seed}: checkpoint on disk", flush=True)
            continue
        print(f"TRAIN {tier}/{args.method} seed {seed} -> {destination}", flush=True)
        partial = destination.with_name(destination.name + ".partial")
        assert_writable(partial, protected)
        partial.unlink(missing_ok=True)
        result = train_tier_dt(
            stacked, parts["index"], tier=tier, method=args.method, seed=int(seed),
            adjacency=parts["adjacency"], prompts=parts["prompts"],
            stats=parts["dataset"].stats, state_dim=int(parts["index"].state_dim),
            n_actions=int(parts["index"].n_actions), gradient_steps=int(args.gradient_steps),
            batch_size=int(args.batch_size), device=device, checkpoint_path=partial,
            protected=protected, provenance={"runtime": runtime_provenance()},
            deterministic=bool(args.deterministic), log_every=int(args.log_every),
        )
        replace_guarded(partial, destination, protected)
        records.append({
            "tier": tier, "method": args.method, "seed": int(seed),
            "gradient_steps": result.gradient_steps, "seconds": result.seconds,
            "final_loss": result.losses[-1], "checkpoint_path": str(destination),
            "deterministic": bool(args.deterministic),
        })
        print(f"  done in {result.seconds:.1f}s, final loss {result.losses[-1]:.5f}", flush=True)
    work = Path(args.work_dir)
    assert_writable(work, protected)
    work.mkdir(parents=True, exist_ok=True)
    # O4: merge by (tier, method, seed).  A resume skips the seeds already on disk, so writing
    # `records` alone would overwrite a complete record with a partial one -- or an empty one.
    record_path = work / training_record_name(
        tier, args.method, replicate=bool(getattr(args, "replicate", False))
    )
    merged = merge_training_runs(load_existing_runs(record_path), records)
    write_json_guarded(
        {
            "format_version": ARTIFACT_FORMAT_VERSION, "tier": tier, "method": args.method,
            "declared_gradient_steps": int(args.gradient_steps),
            "batch_size": int(args.batch_size), "deterministic": bool(args.deterministic),
            "replicate": bool(getattr(args, "replicate", False)),
            "runs": merged,
            "runs_this_invocation": len(records),
        },
        record_path, protected,
    )
    return 0


def _run_train_baselines(args: Any) -> int:
    """BC, %BC, per-intersection %BC and IQL on one tier, on the SAME size-matched episode set.

    ⚠️ These arms are independent per intersection BY CONSTRUCTION and cannot use the neighbour
    information ``dt_spatial`` is given.  **That asymmetry IS the experiment** and is stated rather
    than hidden.  The trainers, batching, loss and optimiser are ``offline.offline_baselines``',
    imported unchanged; the only new selection is ``per_intersection_top_streams``.
    """
    import torch

    from offline.dt_gate import runtime_provenance, stack_dataset
    from offline.offline_baselines import (
        IQL_BATCH_TRANSITIONS,
        build_transitions,
        filter_stacked_to_streams,
        iql_reward_scale,
        top_return_streams,
        train_bc,
        train_iql,
    )

    protected = _protected_of(args)
    tier = _require_tier(args)
    parts = tier_parts(tier, args.corpus_root, sim_config=args.sim_config)
    dataset = parts["dataset"]
    group = next(iter(dataset.groups))
    streams = parts["streams"]
    kept_global = tuple(s for s in top_return_streams(dataset) if s in set(streams))
    kept_perix = per_intersection_top_streams(streams)
    scale = iql_reward_scale([s.total_return for s in streams])
    stacked_all = stack_dataset(dataset, group)
    stacked = filter_stacked_to_streams(dataset, stacked_all, streams)
    batches = {
        "bc": stacked,
        "bc_top10": filter_stacked_to_streams(dataset, stacked_all, kept_global),
        "bc_top10_perix": filter_stacked_to_streams(dataset, stacked_all, kept_perix),
    }
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoints = Path(args.checkpoint_dir)
    assert_writable(checkpoints, protected)
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tier": tier,
        "dataset_dirs": [str(d) for d in tier_dirs(parts["spec"], args.corpus_root)],
        "scenario_id": SCENARIO_ID,
        "training_streams": len(streams),
        "independent_per_intersection": True,
    }
    print(
        f"baselines @{tier}: {len(streams)} streams, global decile {len(kept_global)}, "
        f"per-intersection decile {len(kept_perix)}, iql reward scale {scale}, device {device}",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    table = None
    for method in ("bc", "bc_top10", "bc_top10_perix", "iql"):
        for seed in _seeds_of(args):
            path = checkpoints / _checkpoint_name(tier, method, seed)
            if path.is_file():
                print(f"SKIP {tier}/{method} seed {seed}: checkpoint on disk", flush=True)
                continue
            assert_writable(path, protected)
            partial = path.with_name(path.name + ".partial")
            partial.unlink(missing_ok=True)
            print(f"TRAIN {tier}/{method} seed {seed} -> {path}", flush=True)
            if method == "iql":
                if table is None:
                    table = build_transitions(dataset, group=group, reward_scale=scale)
                    print(f"  transitions {len(table)}", flush=True)
                train_iql(
                    table, state_dim=group[0], n_actions=group[1], seed=int(seed),
                    declared_gradient_steps=int(args.gradient_steps),
                    batch_size=IQL_BATCH_TRANSITIONS, device=device, checkpoint_path=partial,
                    stats=dataset.stats, scenario_id=SCENARIO_ID,
                    provenance={**provenance, "runtime": runtime_provenance()},
                    log_every=int(args.log_every),
                )
            else:
                train_bc(
                    batches[method], state_dim=group[0], n_actions=group[1], seed=int(seed),
                    method=method, declared_gradient_steps=int(args.gradient_steps),
                    batch_size=JOINT_BATCH_SIZE, device=device, checkpoint_path=partial,
                    stats=dataset.stats, scenario_id=SCENARIO_ID,
                    provenance={**provenance, "runtime": runtime_provenance()},
                    log_every=int(args.log_every),
                )
            replace_guarded(partial, path, protected)
            records.append({"tier": tier, "method": method, "seed": int(seed),
                            "checkpoint_path": str(path),
                            "gradient_steps": int(args.gradient_steps)})
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # O4: merge by (tier, method, seed).  All four of this task's *_baselines.json records were
    # destroyed by the absence of this line: on a resume every seed is skipped, `records` is empty,
    # and the old code wrote that empty list over 20 complete entries.
    record_path = work / f"training_{tier}_baselines.json"
    merged = merge_training_runs(load_existing_runs(record_path), records)
    write_json_guarded(
        {
            "format_version": ARTIFACT_FORMAT_VERSION, "tier": tier,
            "methods": ["bc", "bc_top10", "bc_top10_perix", "iql"],
            "declared_gradient_steps": int(args.gradient_steps),
            "iql_reward_scale": scale,
            "global_decile_streams": len(kept_global),
            "per_intersection_decile_streams": len(kept_perix),
            "training_streams": len(streams),
            "runs": merged,
            "runs_this_invocation": len(records),
        },
        record_path, protected,
    )
    return 0


def _arm_factory(tier: str, method: str, args: Any, seed: int) -> Any:
    """The action factory for one arm-seed.  One place, so no arm is wired twice."""
    from offline.method_tier_grid import _random_factory
    from offline.offline_baselines import _baseline_factory

    checkpoints = Path(args.checkpoint_dir)
    if method in DT_METHODS:
        path = str(
            checkpoints
            / _checkpoint_name(
                tier, method, seed, replicate=bool(getattr(args, "replicate", False))
            )
        )

        def factory(env: Any) -> Any:
            from agent.SpatialDTAgent import SpatialDTAgent

            agent = SpatialDTAgent.from_checkpoint(env, path, device=args.device)
            return lambda _env, info: agent.act(info, explore=False, update_memory=True)

        return factory
    if method in ("bc", "bc_top10", "bc_top10_perix", "iql"):
        path = str(checkpoints / _checkpoint_name(tier, method, seed))
        return _baseline_factory(method, path, int(args.gradient_steps), args.device)
    if method == COLLAPSE_REFERENCE_METHOD:
        return _random_factory(int(seed))
    if method == BEHAVIOUR_METHOD:
        if tier == "random":
            return _random_factory(int(seed))
        if tier == "maxpressure":
            from offline.dt_gate import _maxpressure_factory

            return _maxpressure_factory
        if tier == "fixedtime":
            # P4.6's factory, imported rather than reimplemented, because it does one thing more
            # than build the controller: it ASSERTS the resolved plan's hash and schedule source
            # against the ones recorded in the tier's own manifest, so an evaluation that would
            # measure a DIFFERENT fixed-time policy than the one that collected the corpus is
            # refused instead of reported.  Fixed-time is deterministic, so the factory ignores
            # the seed -- the five seed slots hold five identical rollouts, exactly as the
            # maxpressure anchor does (measured: all five per-seed means 167.561778).
            from offline.materialise_draws import draw_config_path
            from offline.method_tier_grid import (
                _fixedtime_factory,
                fixedtime_collection_settings,
            )

            collected = fixedtime_collection_settings(
                tier_dirs(tier_spec(tier), args.corpus_root)[0] / "manifest.json"
            )
            from offline.dt_gate import HELD_OUT_DRAWS

            return _fixedtime_factory(
                draw_config_path(
                    args.scenario_key, HELD_OUT_DRAWS[0], out_root=args.draws_root
                ),
                collected,
            )
        if tier == "mappo1000":
            from offline.dt_gate import _mappo_factory

            path = Path(args.mappo_checkpoint_dir) / f"cf_grid4x4__mappo__seed{seed}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist; the behaviour anchor needs it")
            return _mappo_factory(str(path), args.device)
    raise ValueError(f"no action factory is declared for {method!r} at tier {tier!r}")


def assert_factories_constructible(args: Any) -> dict[str, Any]:
    """Gate 0: every cell the DECLARATION names must have a CONSTRUCTIBLE action factory.

    🚨 **Why this exists, and it is a defect this campaign actually paid for.**
    ``_arm_factory``'s ``behaviour`` branch had no ``fixedtime`` case, so it fell through to its
    own ``raise``.  The failure was **deterministic and reachable only at evaluation time**, three
    days into a 53 h campaign, and it recurred on every resume.

    ⚠️ **``assert-complete`` cannot catch this class.** It compares the declared cells to the cells
    on disk, so it only ever fires **after** the work that was supposed to produce them -- it tells
    you a cell is missing, never that a cell is unproducible.  This asserts the *capability* before
    any training starts: it walks the declared ``(tier, method)`` matrix, builds each factory, and
    refuses on anything that raises.  Seconds, no training, no environment.

    Building the factory does not construct an env -- the returned closure takes one -- so this is
    cheap, but it DOES exercise the real preconditions: the MAPPO anchor's checkpoint must exist,
    and the fixed-time plan must resolve and hash-match its tier's manifest.
    """
    checked: list[dict[str, str]] = []
    problems: list[str] = []
    seed = int(_seeds_of(args)[0])
    for entry in campaign_cell_manifest():
        if entry.get("source") == "reuse_root":
            continue
        tier, method = str(entry["tier"]), str(entry["method"])
        if tier == "all":
            tier = REUSED_TIER
        try:
            _arm_factory(tier, method, args, seed)
        except Exception as exc:  # noqa: BLE001 - any failure here is a refusal
            problems.append(f"{method}@{tier}: {type(exc).__name__}: {exc}")
            continue
        checked.append({"tier": tier, "method": method})
    return {"n_checked": len(checked), "problems": problems, "ok": not problems}


def _run_check_factories(args: Any) -> int:
    report = assert_factories_constructible(args)
    print(f"action factories constructible: {report['n_checked']} checked")
    if not report["ok"]:
        print("\nREFUSED -- these declared cells have no constructible factory:")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    return 0


def _run_evaluate(args: Any) -> int:
    from offline.dt_gate import (
        HELD_OUT_DRAWS,
        _cell,
        env_settings_from_manifest,
        evaluate_arm,
    )
    from offline.materialise_draws import draw_config_path

    protected = _protected_of(args)
    tier = _require_tier(args)
    spec = tier_spec(tier)
    settings = env_settings_from_manifest(tier_dirs(spec, args.corpus_root)[0] / "manifest.json")
    draws = list(HELD_OUT_DRAWS)
    arm = f"{args.method}@{tier}"
    produced: list[Any] = []
    for seed in _seeds_of(args):
        print(f"{arm} seed {seed} over {len(draws)} draws", flush=True)
        produced.extend(
            evaluate_arm(
                arm=arm, seed=int(seed), draw_ids=draws,
                config_for_draw=lambda d: draw_config_path(
                    args.scenario_key, d, out_root=args.draws_root
                ),
                env_settings=settings, scenario_id=args.scenario_id,
                choose_action_factory=_arm_factory(tier, args.method, args, int(seed)),
                engine_seed=int(args.engine_seed),
            )
        )
    expected = {(int(s), int(d)) for s in _seeds_of(args) for d in draws}
    got = {(int(r.seed), int(r.draw_id)) for r in produced}
    if got != expected:
        raise ValueError(
            f"{arm}: {len(got)} episodes against {len(expected)} requested "
            f"(missing {len(expected - got)}, unexpected {len(got - expected)})"
        )
    work = Path(args.work_dir)
    assert_writable(work, protected)
    work.mkdir(parents=True, exist_ok=True)
    seeds_tag = "" if not args.seeds else f"_seed{'_'.join(str(s) for s in _seeds_of(args))}"
    if getattr(args, "replicate", False):
        seeds_tag += REPLICATE_SUFFIX
    write_json_guarded(
        {
            "format_version": ARTIFACT_FORMAT_VERSION, "arm": arm, "tier": tier,
            "method": args.method, "declared_gradient_steps": int(args.gradient_steps),
            "engine_seed": int(args.engine_seed), "deterministic": bool(args.deterministic),
            "cell": _cell(produced),
            "episodes": [
                {"arm": e.arm, "seed": e.seed, "draw_id": e.draw_id,
                 "att_horizon": e.att_horizon,
                 "horizon_vehicle_count": e.horizon_vehicle_count,
                 "episode_reward": e.episode_reward}
                for e in produced
            ],
        },
        work / f"eval_{tier}_{args.method}{seeds_tag}.json", protected,
    )
    print(f"{arm}: {len(produced)} episodes written")
    return 0


def score_stop_rule(
    spatial: Mapping[str, Any], nomix: Mapping[str, Any], seeds: Sequence[int]
) -> dict[str, Any]:
    """Q0's stop rule, scored from the two 4-head cells: ``d4`` paired per draw over shared draws.

    ⚠️ **The quantity this fires on must BE the mixing contrast, not merely be named one.**  A
    mutation that voided the mixing flag on ``dt_spatial_h4`` would make ``d4`` about zero, the CI
    would straddle, the rule would NOT fire, and the ladder would run on a quantity that is not the
    contrast.  What guarantees it is not this function but
    ``test_each_arm_of_the_2x2_gets_the_graph_its_name_promises``, which asserts at the ARTIFACT that
    each arm's saved mask has an off-diagonal edge iff its name says it mixes.
    """
    from offline.dt_gate import mean_ci95

    per_draw: dict[str, dict[int, list[float]]] = {}
    for name, payload in (("spatial", spatial), ("nomix", nomix)):
        grouped: dict[int, list[float]] = {}
        for episode in payload["episodes"]:
            if int(episode["seed"]) not in {int(s) for s in seeds}:
                continue
            grouped.setdefault(int(episode["draw_id"]), []).append(float(episode["att_horizon"]))
        per_draw[name] = grouped
    shared = sorted(set(per_draw["spatial"]) & set(per_draw["nomix"]))
    if not shared:
        raise ValueError("the two 4-head cells share no draw ids; A5 makes the comparison void")
    differences = [
        sum(per_draw["spatial"][d]) / len(per_draw["spatial"][d])
        - sum(per_draw["nomix"][d]) / len(per_draw["nomix"][d])
        for d in shared
    ]
    stats = mean_ci95(differences)
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    verdict = stop_rule_verdict(low, high)
    return {
        "quantity": "d4 = ATT(dt_spatial_h4) - ATT(dt_nomix_h4), paired per shared draw",
        "n_shared_draws": len(shared),
        "mean_difference": stats.mean,
        "std": stats.std,
        "ci95_low": low,
        "ci95_high": high,
        # D8: every reported pair carries the CI WIDTH, not only its endpoints.  It is also what
        # discriminates a paired difference from an unpaired one when the point estimate agrees.
        "ci95_width": high - low,
        "verdict": verdict,
        "reading": (
            "STOP: the CI lies entirely below zero, so spatial mixing HELPS at 4 heads and P5.1's "
            "sign has reversed -- the ladder would be measuring the wrong architecture"
            if verdict == "STOP"
            else (
                "CONTINUE: the CI straddles zero, so the contrast DOES NOT RESOLVE at 4 heads; "
                "this is not a reversal and phase B proceeds"
                if low <= 0.0 <= high
                else "CONTINUE: the CI lies entirely above zero, so mixing still hurts at 4 heads"
            )
        ),
    }


def _run_stop_rule(args: Any) -> int:
    """Score Q0 and REFUSE to continue if the sign reversed.  Exit 3 means STOP."""
    import json as _json

    protected = _protected_of(args)
    tier = _require_tier(args)
    work = Path(args.work_dir)
    spatial = _json.loads(
        (work / f"eval_{tier}_dt_spatial_h4.json").read_text(encoding="utf-8")
    )
    nomix = _json.loads((work / f"eval_{tier}_dt_nomix_h4.json").read_text(encoding="utf-8"))
    report = score_stop_rule(spatial, nomix, _seeds_of(args))
    print(
        f"  d4 = {report['mean_difference']:+.4f}  "
        f"CI [{report['ci95_low']:+.4f}, {report['ci95_high']:+.4f}]  "
        f"over {report['n_shared_draws']} shared draws"
    )
    print(f"  VERDICT: {report['verdict']} -- {report['reading']}")
    write_json_guarded(
        {"format_version": ARTIFACT_FORMAT_VERSION, "measurement": "Q0 stop rule", **report},
        work / f"stop_rule_{tier}.json", protected,
    )
    if report["verdict"] == "STOP":
        marker = work / "STOPPED_BY_RULE"
        assert_writable(marker, protected)
        marker.write_text(
            f"d4 = {report['mean_difference']:.4f} CI [{report['ci95_low']:.4f}, "
            f"{report['ci95_high']:.4f}] lies entirely below zero.\n"
            "The ladder sweep is NOT run: at 4 heads spatial mixing helps, P5.1's sign has "
            "reversed, and the sweep would be measuring the wrong architecture "
            "(docs/plans/p5.2.md section 4, Q0).\n",
            encoding="utf-8",
        )
        return 3
    return 0


def _run_declare_campaign(args: Any) -> int:
    """Write the campaign-wide declaration: every cell the campaign owes, before it runs."""
    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

    protected = _protected_of(args)
    manifest = campaign_cell_manifest()
    payload = {
        "format_version": DECLARATION_FORMAT_VERSION,
        "scenario_id": SCENARIO_ID,
        "regime": "default CUDA (BRIEF_27 H1); CUBLAS_WORKSPACE_CONFIG unset (G2)",
        "tier_order": list(TIER_ORDER),
        "tier_order_basis": "cf_grid4x4's own measured att_horizon (BRIEF_27 B0)",
        "methods": list(METHODS),
        "head_methods": list(HEAD_METHODS),
        "seeds": list(TRAINING_SEEDS),
        "held_out_draws": list(HELD_OUT_DRAWS),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "reused_tier": REUSED_TIER,
        "reused_cells": list(REUSED_CELLS),
        "envelope_replicate": {
            "tier": REPLICATE_TIER, "seed": REPLICATE_SEED, "arms": list(REPLICATE_ARMS),
            "suffix": REPLICATE_SUFFIX,
            "seed_is_pre_declared": True,
            "note": (
                "BRIEF_27 I1/J1: unconditional, both arms, seed pre-declared. If phase B shows 202 "
                "is unrepresentative of this tier's per-seed spread that is a DISCLOSURE, not a "
                "reason to re-select"
            ),
        },
        "out_of_sample_cells": [list(cell) for cell in OUT_OF_SAMPLE_CELLS],
        "level_band": LEVEL_BAND,
        "level_threshold": LEVEL_THRESHOLD,
        "level_threshold_rule": (
            f"k = ceil({REGISTERED_LEVEL_RATE[0]}/{REGISTERED_LEVEL_RATE[1]} * N); at N = "
            f"{REGISTERED_LEVEL_RATE[1]} it returns {level_threshold_for(REGISTERED_LEVEL_RATE[1])}, "
            "reproducing the originally registered threshold (BRIEF_27 I3)"
        ),
        "expected_cells": [dict(entry) for entry in manifest],
        "n_expected_cells": len(manifest),
    }
    out = Path(args.out_dir) / "p5_2_declaration.json"
    assert_writable(out, protected)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_guarded(payload, out, protected)
    print(f"campaign declaration written to {out}")
    roles: dict[str, int] = {}
    for entry in manifest:
        roles[str(entry["role"])] = roles.get(str(entry["role"]), 0) + 1
    for role, count in sorted(roles.items()):
        print(f"  {role:28s} {count}")
    print(f"  {'TOTAL':28s} {len(manifest)}")
    return 0


def assert_campaign_complete(
    declaration: Mapping[str, Any], work_dir: str | Path, reuse_root: str | Path
) -> dict[str, Any]:
    """Refuse unless every cell the DECLARATION names exists and is a finished cell.

    ⚠️ **Expected is derived from the declaration, never from the files being checked.**  A campaign
    that verified itself against its own output would report success for whatever it happened to
    produce -- the failure ``PROJECT_PLAN`` section 7 records as running *"on to a clean-looking end"*
    with half its output missing.
    """
    import json

    from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

    work, reuse = Path(work_dir), Path(reuse_root)
    problems: list[str] = []
    present: list[str] = []
    for entry in declaration["expected_cells"]:
        root = reuse if entry.get("source") == "reuse_root" else work
        path = root / str(entry["cell"])
        if not path.is_file():
            problems.append(f"{entry['cell']}: missing ({entry['role']}, {entry['produced_by']})")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds = [int(entry["seed"])] if entry.get("seed") is not None else list(TRAINING_SEEDS)
        got = episode_key_set(payload)
        want = {(int(s), int(d)) for s in seeds for d in HELD_OUT_DRAWS}
        if got != want or len(payload["episodes"]) != len(want):
            problems.append(
                f"{entry['cell']}: {len(payload['episodes'])} episodes over {len(got)} distinct "
                f"keys, expected exactly {len(want)}"
            )
            continue
        present.append(str(entry["cell"]))
    return {
        "n_expected": len(declaration["expected_cells"]),
        "n_present": len(present),
        "problems": problems,
        "complete": not problems,
    }


def _run_assert_complete(args: Any) -> int:
    import json

    declaration = json.loads(
        (Path(args.out_dir) / "p5_2_declaration.json").read_text(encoding="utf-8")
    )
    report = assert_campaign_complete(declaration, args.work_dir, args.reuse_root)
    print(f"expected cells: {report['n_expected']}   present: {report['n_present']}")
    if not report["complete"]:
        print("\nCAMPAIGN INCOMPLETE:")
        for problem in report["problems"][:20]:
            print(f"  - {problem}")
        return 1
    print("complete == declared, every cell holds its declared seeds x 100 draws")
    return 0


def positive_control_resolution(
    published: Mapping[str, Any], seed: int, epsilon: float = 1e-12
) -> dict[str, Any]:
    """Prove the detector resolves *epsilon* by perturbing one episode and re-running it.

    ⚠️ **This is what separated E1's zero from a self-comparison, and F7 did not require it.**  A
    reported ``+0.0000`` means nothing unless the instrument that produced it is shown to move when
    something moves.  One episode of the published cell is shifted by ``epsilon`` in a COPY -- the
    artifact on disk is never touched -- and the report must then differ from zero.
    """
    import copy

    perturbed = copy.deepcopy(dict(published))
    perturbed["episodes"] = [dict(e) for e in published["episodes"]]
    target = next(e for e in perturbed["episodes"] if int(e["seed"]) == int(seed))
    target["att_horizon"] = float(target["att_horizon"]) + float(epsilon)
    report = paired_replicate_report(perturbed, published, seed=seed)
    detected = report["mean_difference"] != 0.0
    if not detected:
        raise ValueError(
            f"the positive control did not fire: a {epsilon} shift in one episode produced a mean "
            "difference of exactly zero, so this detector cannot resolve it and any zero it "
            "reports is uninterpretable"
        )
    return {
        "epsilon": float(epsilon),
        "perturbed_episodes": 1,
        "observed_mean_difference": report["mean_difference"],
        "detected": bool(detected),
        "note": (
            "run on a COPY; the artifact on disk is untouched. Without this a zero envelope is "
            "indistinguishable from a self-comparison"
        ),
    }


def _run_envelope_report(args: Any) -> int:
    """I1/J1: the random-tier envelope, under J2's three conditions.

    Compares the REPLICATE (an independent training run from scratch under a distinct key) against
    phase B's own cell at the same pre-declared seed, for both arms and for ``d1``.
    """
    import json as _json

    from offline.dt_gate import mean_ci95

    protected = _protected_of(args)
    tier = _require_tier(args)
    seed = int(args.seed)
    work = Path(args.work_dir)
    checkpoints = Path(args.checkpoint_dir)

    # Every input is checked to EXIST before anything is loaded, so a missing artifact says which
    # one and what produces it rather than surfacing as a torch traceback.
    required: list[tuple[str, Path]] = []
    for method in REPLICATE_ARMS:
        required.append((
            f"phase R checkpoint for {method} (run: --replicate --seeds {seed} train)",
            checkpoints / _checkpoint_name(tier, method, seed, replicate=True),
        ))
        required.append((
            f"phase B checkpoint for {method} (produced by phase B)",
            checkpoints / _checkpoint_name(tier, method, seed),
        ))
        required.append((
            f"phase R cell for {method} (run: --replicate --seeds {seed} evaluate)",
            work / cell_filename(tier, method, seed, replicate=True),
        ))
        required.append((
            f"phase B cell for {method} (produced by phase B)",
            work / cell_filename(tier, method),
        ))
    absent = [(what, path) for what, path in required if not path.is_file()]
    if absent:
        lines = "\n".join(f"  - {what}: {path}" for what, path in absent)
        raise SystemExit(
            f"the envelope report needs {len(absent)} artifact(s) that do not exist:\n{lines}\n"
            "Phase R must run before its report; nothing is computed from a partial input."
        )

    # J2(c) FIRST: no envelope is computed, let alone reported, until the two runs are shown to
    # have produced DIFFERENT weights.  Equal digests are a refusal, not a zero.
    digests: dict[str, Any] = {}
    for method in REPLICATE_ARMS:
        digests[method] = assert_independent_replicate(
            checkpoints / _checkpoint_name(tier, method, seed, replicate=True),
            checkpoints / _checkpoint_name(tier, method, seed),
        )
        print(
            f"  {method}: replicate {digests[method]['replicate_state_dict_sha256'][:16]}... vs "
            f"original {digests[method]['original_state_dict_sha256'][:16]}... DIFFER",
            flush=True,
        )

    blocks: dict[str, Any] = {}
    per_draw: dict[str, dict[int, float]] = {}
    controls: dict[str, Any] = {}
    for method in REPLICATE_ARMS:
        replicate = _json.loads(
            (work / cell_filename(tier, method, seed, replicate=True)).read_text(encoding="utf-8")
        )
        published = _json.loads(
            (work / cell_filename(tier, method)).read_text(encoding="utf-8")
        )
        blocks[method] = paired_replicate_report(replicate, published, seed=seed)
        controls[method] = positive_control_resolution(published, seed)
        per_draw[f"{method}_replicate"] = {
            int(e["draw_id"]): float(e["att_horizon"])
            for e in replicate["episodes"] if int(e["seed"]) == seed
        }
        per_draw[f"{method}_published"] = {
            int(e["draw_id"]): float(e["att_horizon"])
            for e in published["episodes"] if int(e["seed"]) == seed
        }

    shared = sorted(
        set(per_draw["dt_spatial_replicate"]) & set(per_draw["dt_nomix_replicate"])
        & set(per_draw["dt_spatial_published"]) & set(per_draw["dt_nomix_published"])
    )
    if len(shared) != 100:
        raise ValueError(f"expected 100 shared draws for d1, found {len(shared)}")
    differences = [
        (per_draw["dt_spatial_replicate"][d] - per_draw["dt_nomix_replicate"][d])
        - (per_draw["dt_spatial_published"][d] - per_draw["dt_nomix_published"][d])
        for d in shared
    ]
    stats = mean_ci95(differences)
    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95
    blocks["d1"] = {
        "metric": "att_horizon", "seed": seed, "n_shared_draws": len(shared),
        "mean_difference": stats.mean, "std": stats.std,
        "ci95_low": low, "ci95_high": high, "ci95_width": high - low,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "d1_replicate": sum(
            per_draw["dt_spatial_replicate"][d] - per_draw["dt_nomix_replicate"][d] for d in shared
        ) / len(shared),
        "d1_published": sum(
            per_draw["dt_spatial_published"][d] - per_draw["dt_nomix_published"][d] for d in shared
        ) / len(shared),
    }

    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "measurement": (
            "I1/J1 -- the nondeterminism envelope on the random tier "
            "(docs/plans/p5.2.md section 4.8)"
        ),
        "tier": tier, "seed": seed, "regime": "default CUDA",
        "seed_is_pre_declared": True,
        "seed_note": (
            "202 was declared before phase B ran, because the principle that chose it for E1 -- "
            "the largest per-seed effect -- cannot be applied to a tier whose per-seed spread does "
            "not exist until phase B has run. If 202 is unrepresentative of this tier's spread "
            "that is a DISCLOSURE, not a reason to re-select (BRIEF_27 J1)"
        ),
        "independence": digests,
        "positive_control": controls,
        "attribution": (
            "obligation 6 proved train_tier_dt reproduces spatial_mixing.train_spatial_dt "
            "byte-exactly at one head on CPU, so any difference here is device nondeterminism "
            "alone and not a trainer change"
        ),
        "registered_reading": (
            "BRIEF_27 I2, registered before this number existed: a NON-ZERO envelope is a FINDING "
            "-- nondeterminism reaching the metric where the learned policy separates phases less "
            "confidently -- and a ZERO is not the only acceptable outcome. Both are publishable"
        ),
        "mappo1000_envelope_for_comparison": 0.0,
        "mappo1000_scope_limit": (
            "BRIEF_27 H2: E1's 0.0000 is a mappo1000-only measurement and is NOT established for "
            "the degraded tiers; decision margins are a property of how confidently the learned "
            "policy separates phases"
        ),
        "blocks": blocks,
    }
    out = work / f"envelope_{tier}_seed{seed}.json"
    write_json_guarded(payload, out, protected)
    for name, block in blocks.items():
        print(
            f"  {name:11s} mean {block['mean_difference']:+9.4f}  "
            f"CI [{block['ci95_low']:+9.4f}, {block['ci95_high']:+9.4f}]  "
            f"excludes zero: {block['excludes_zero']}"
        )
    print(f"envelope report written to {out}")
    return 0


def _run_verify_reuse(args: Any) -> int:
    record = verify_reuse_gate(args.reuse_root, args.checksums)
    for arm, entry in sorted(record.items()):
        print(f"  {arm:16s} {entry['sha256'][:16]}...  {entry['source']}")
    print(f"verified {len(record)} reused cells at consumption")
    return 0


def _run_replicate_report(args: Any) -> int:
    """E1/F7: the paired per-draw comparison of the replicate against P5.1's own seed-202 cells."""
    import json as _json

    protected = _protected_of(args)
    tier = _require_tier(args)
    work = Path(args.work_dir)
    reuse = Path(args.reuse_root)
    seed = int(args.seed)
    blocks: dict[str, Any] = {}
    per_draw: dict[str, dict[int, float]] = {}
    for method in ("dt_spatial", "dt_nomix"):
        replicate = _json.loads(
            (work / f"eval_{tier}_{method}_seed{seed}.json").read_text(encoding="utf-8")
        )
        published = _json.loads((reuse / f"eval_{method}.json").read_text(encoding="utf-8"))
        blocks[method] = paired_replicate_report(replicate, published, seed=seed)
        per_draw[f"{method}_replicate"] = {
            int(e["draw_id"]): float(e["att_horizon"])
            for e in replicate["episodes"] if int(e["seed"]) == seed
        }
        per_draw[f"{method}_published"] = {
            int(e["draw_id"]): float(e["att_horizon"])
            for e in published["episodes"] if int(e["seed"]) == seed
        }
    # d1 = spatial - nomix, per draw, in each of the two runs; then the paired difference of those.
    from offline.dt_gate import mean_ci95

    shared = sorted(
        set(per_draw["dt_spatial_replicate"]) & set(per_draw["dt_nomix_replicate"])
        & set(per_draw["dt_spatial_published"]) & set(per_draw["dt_nomix_published"])
    )
    if len(shared) != 100:
        raise ValueError(f"expected 100 shared draws for d1, found {len(shared)}")
    d1_replicate = [
        per_draw["dt_spatial_replicate"][d] - per_draw["dt_nomix_replicate"][d] for d in shared
    ]
    d1_published = [
        per_draw["dt_spatial_published"][d] - per_draw["dt_nomix_published"][d] for d in shared
    ]
    stats = mean_ci95([a - b for a, b in zip(d1_replicate, d1_published)])
    blocks["d1"] = {
        "metric": "att_horizon", "seed": seed, "n_shared_draws": len(shared),
        "mean_difference": stats.mean, "std": stats.std,
        "ci95_low": stats.mean - stats.ci95, "ci95_high": stats.mean + stats.ci95,
        "ci95_width": 2 * stats.ci95,
        "excludes_zero": bool(stats.mean - stats.ci95 > 0 or stats.mean + stats.ci95 < 0),
        "d1_replicate": sum(d1_replicate) / len(d1_replicate),
        "d1_published": sum(d1_published) / len(d1_published),
    }
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "measurement": "E1 -- the nondeterminism envelope (docs/plans/p5.2.md section 4.7)",
        "tier": tier, "seed": seed, "regime": "default CUDA",
        "attribution": (
            "obligation 6 proved train_tier_dt reproduces spatial_mixing.train_spatial_dt "
            "byte-exactly at one head on CPU -- identical loss sequence and all 66 state_dict "
            "tensors -- so any difference measured here is attributable to DEVICE "
            "NONDETERMINISM ALONE and not to the trainer change"
        ),
        "published_d1_seed202": 72.07412354024478,
        "blocks": blocks,
    }
    out = work / f"e1_envelope_{tier}_seed{seed}.json"
    write_json_guarded(payload, out, protected)
    for name, block in blocks.items():
        print(
            f"  {name:11s} mean {block['mean_difference']:+9.4f}  "
            f"CI [{block['ci95_low']:+9.4f}, {block['ci95_high']:+9.4f}]  "
            f"excludes zero: {block['excludes_zero']}"
        )
    print(f"E1 report written to {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m offline.tier_sweep``."""
    import torch

    args = build_parser().parse_args(argv)
    # F6(a): the regime is configured ONCE, here, before anything can touch CUDA.
    configure_determinism(bool(args.deterministic))
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if args.command == "declare":
        return _run_declare(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "declare-campaign":
        return _run_declare_campaign(args)
    if args.command == "check-factories":
        return _run_check_factories(args)
    if args.command == "assert-complete":
        return _run_assert_complete(args)
    if args.command == "train-baselines":
        return _run_train_baselines(args)
    if args.command == "verify-reuse":
        return _run_verify_reuse(args)
    if args.command == "stop-rule":
        return _run_stop_rule(args)
    if args.command == "replicate-report":
        return _run_replicate_report(args)
    if args.command == "envelope-report":
        return _run_envelope_report(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - exercised via the campaign script
    raise SystemExit(main())
