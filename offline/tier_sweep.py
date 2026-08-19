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
#: as grid4x4's).  ``fixedtime`` is the pre-declared optional fourth tier and is not in this tuple.
TIER_ORDER: tuple[str, ...] = ("mappo1000", "maxpressure", "random")


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
    "dt_spatial": {"mappo1000": 197.4126, "maxpressure": 208.2667, "random": 312.9006},
    "dt_nomix": {"mappo1000": 157.8477, "maxpressure": 166.5265, "random": 250.1900},
    "bc": {"mappo1000": 168.9806, "maxpressure": 175.7248, "random": 272.1254},
    "bc_top10": {"mappo1000": 749.5796, "maxpressure": 751.0633, "random": 1242.6948},
    "bc_top10_perix": {"mappo1000": 165.7657, "maxpressure": 166.0938, "random": 274.8156},
    "iql": {"mappo1000": 275.8354, "maxpressure": 264.0804, "random": 414.9930},
    "behaviour": {"mappo1000": 160.2780, "maxpressure": 167.4920, "random": 260.3602},
}

#: The 13 cells scored by Q1.  Seen cells are excluded so a free hit cannot enter the denominator
#: (``BRIEF_27`` B5.2); ``behaviour@maxpressure`` is scored separately as Q1b's instrument check.
OUT_OF_SAMPLE_CELLS: tuple[tuple[str, str], ...] = (
    ("dt_spatial", "maxpressure"),
    ("dt_nomix", "maxpressure"),
    ("bc", "maxpressure"),
    ("bc_top10", "maxpressure"),
    ("bc_top10_perix", "maxpressure"),
    ("iql", "maxpressure"),
    ("dt_spatial", "random"),
    ("dt_nomix", "random"),
    ("bc", "random"),
    ("bc_top10", "random"),
    ("bc_top10_perix", "random"),
    ("iql", "random"),
    ("bc_top10_perix", "mappo1000"),
)

#: Q1's band and threshold, fixed in the commit that registered them and NOT widenable afterwards.
#: Calibration behind the choice: median relative error 23.9 %, max 378.5 %, 3 of 5 inside the band.
LEVEL_BAND = 0.30
LEVEL_THRESHOLD = 9

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
