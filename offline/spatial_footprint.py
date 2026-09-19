"""P5.4: the trained-model footprint of the spatial layer on ``cf_grid4x4__mappo1000``.

Artifact format version: ``p5.4-footprint/1.0``.

WHAT THIS MEASURES, AND WHAT DECIDED IT
----------------------------------------
``PREREGISTRATION.md`` A22 (``v2.2-prereg-a22`` -> ``f966f33``) declares an influence matrix **F (16 x 16)** per
checkpoint: over 200 real joint windows drawn from the corpus the models trained on, node ``j``'s whole state sequence
is replaced by the same node's sequence from another real window, and ``F[i, j]`` is the mean absolute change of node
``i``'s **last-step action logits**.  ``F[i, i]`` is that substitution applied to node ``i`` itself.  The statistic is

    r = mean_i( mean over ONE-HOP neighbours j of F[i, j] )  /  mean_i F[i, i]

and the footprint is **TRIVIAL iff the median of r over the five spatial seeds is below 0.10**.  Nothing in this
module chooses any of that: the threshold, the statistic, the perturbation, the window count and both paper sentences
are registered, and the seed (``20260919``) and the full-window rule were fixed at the plan gate before any
trained-model number existed (``docs/plans/p5.4.md``, ``BRIEF_40`` Amendment A).

🚨 THE SUPPORT CHECK IS A DEFECT TEST, AND ITS FIRST FORM WOULD HAVE FIRED ON A CORRECT MODEL (A22.1)
------------------------------------------------------------------------------------------------------
A22(b) originally required ``F[i, j] == 0`` for **every** non-neighbour pair, calling the hard ``-inf`` attention bias
an identity.  That is an identity for **one** spatial layer.  :class:`~agent.SpatialDTAgent.SpatialDecisionTransformer`
repeats its block ``n_layer`` times (``agent/SpatialDTAgent.py:282-285``), each block applying the *same* one-hop bias,
and every registered checkpoint records ``n_layer = 3`` -- so information travels **three hops** and a correct model is
influenced by every intersection within three hops of it.  On the 4x4 lattice the 240 ordered pairs split by hop
distance into ``{1: 48, 2: 68, 3: 64, 4: 40, 5: 16, 6: 4}``: of the 192 non-neighbour pairs, **132 are within reach and
60 are beyond it**.

**A22.1 is the registered correction and is what this module implements:** exact zero is required only for pairs **more
than ``n_layer`` hops apart on the recorded mask** (the 60), and for every off-diagonal pair of a ``nomix`` checkpoint
(the identity's reach is itself).  Influence at two and three hops is **expected** and is reported per hop,
descriptively, with no threshold attached.  The comparison is ``torch.equal`` on the raw per-window logits, never on a
mean; a violation refuses the artifact and stops the task.

WHAT IS REUSED RATHER THAN RE-DERIVED
--------------------------------------
Windows come from the corpus's own builders (``offline.joint_windows.build_joint_index`` / ``stack_joint``) over
``offline.dataset.TrajectoryWindowDataset``; the tier, its directories, its graph and its node order come from
``offline.tier_sweep``, which is what trained these checkpoints.  Normalisation is
:meth:`agent.SpatialDTAgent.SpatialDTAgent._normalise_state` -- the method ``act`` itself calls -- and it is
cross-checked against the dataset's own route (``offline/dataset.py:795``) under exact equality.  The return-to-go is
divided by each node's ``rtg_scale`` exactly as the trainer divided it (``offline/tier_sweep.py:867-884``).  A second
implementation of any of these is how two measurements stop being comparable.

ALIGNMENT
---------
Contract C6's convention is inherited, not restated: a window is the K decision steps ending at ``t``, the state token
at ``3t+1`` predicts action ``t``, and the readout is the **last** step's ``masked_action_logits`` as the model emits
them (``-inf`` at illegal actions), before any argmax.  Only windows with K real steps are used, so no padded row ever
enters F.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "A22_ATT",
    "A22_CHECKPOINTS",
    "A22_OUTCOMES",
    "CANONICAL_CHECKPOINT_DIR",
    "CANONICAL_CORPUS_ROOT",
    "CANONICAL_EVAL_DIR",
    "CheckpointFootprint",
    "CheckpointPin",
    "EVAL_CELLS",
    "FORMAT_VERSION",
    "Influence",
    "N_WINDOWS",
    "Outcome",
    "RunContext",
    "Subject",
    "TIER",
    "TORCH_THREADS",
    "TRIVIAL_THRESHOLD",
    "VERDICT_NON_TRIVIAL",
    "VERDICT_TRIVIAL",
    "WINDOW_SEED",
    "WindowRef",
    "WindowSample",
    "build_artifact",
    "build_parser",
    "checkpoint_pin",
    "cyclic_derangement",
    "draw_windows",
    "eligible_instants",
    "footprint_dataset",
    "footprint_ratio",
    "hop_distances",
    "influence",
    "load_subject",
    "main",
    "median_verdict",
    "model_inputs",
    "must_be_zero_pairs",
    "per_hop_report",
    "per_seed_att",
    "reachability",
    "support_violations",
    "tier_directories",
    "verdict",
    "verify_checkpoint",
    "verify_eval_cell",
    "write_artifact",
]

FORMAT_VERSION = "p5.4-footprint/1.0"

#: The tier the ten registered checkpoints trained on.  Nothing else may be measured here (A22(a)).
TIER = "mappo1000"

#: Declared in ``docs/plans/p5.4.md`` before any trained-model number existed and accepted at gate G0
#: (``BRIEF_40`` Amendment A, A22.1(e)).  One integer drives the window draw and then the pairing.
WINDOW_SEED = 20_260_919

#: A22(a): 200 real windows.
N_WINDOWS = 200

#: A22(c): TRIVIAL iff the median of r over the five spatial seeds is BELOW this.  The boundary itself is NON-TRIVIAL.
TRIVIAL_THRESHOLD = 0.10

VERDICT_TRIVIAL = "TRIVIAL"
VERDICT_NON_TRIVIAL = "NON-TRIVIAL"

#: Pinned so the artifact regenerates byte-identically; CPU reduction order can depend on the thread count.
TORCH_THREADS = 8

#: Where the inputs live in the repository.  Recorded in the artifact regardless of the directories the run was
#: pointed at, because a digest identifies a file and an absolute path identifies a machine.
CANONICAL_CHECKPOINT_DIR = "output/p5_2/checkpoints"
CANONICAL_EVAL_DIR = "output/p5_2"
CANONICAL_CORPUS_ROOT = "datasets_v11"

_SPATIAL_ARM = "dt_spatial_h4"
_NOMIX_ARM = "dt_nomix_h4"


@dataclass(frozen=True)
class CheckpointPin:
    """One registered checkpoint: A22(a)'s prefix, and the full digest measured 2026-09-19.

    A22(a) registers **eight hex characters**.  No tracked file carries the full digests (``git grep``: zero hits),
    so they are pinned here and every one is asserted to begin with its registered prefix by
    ``tests/test_spatial_footprint.py``.
    """

    arm: str
    seed: int
    filename: str
    a22_prefix: str
    sha256: str

    @property
    def spatial_mixing(self) -> bool:
        """Whether this arm's recorded mask is the lattice (True) or the identity (False)."""
        return self.arm == _SPATIAL_ARM


@dataclass(frozen=True)
class Outcome:
    """One of A22(d)'s two pre-written outcomes, carried verbatim so neither can be edited afterwards."""

    verdict: str
    sentence: str
    a22_d_clause: str


@dataclass(frozen=True)
class WindowRef:
    """Where one drawn window came from, so G1 can recompute it by its own route."""

    dataset_dir: str
    episode_file: str
    flow_draw: int
    t: int


@dataclass(frozen=True)
class WindowSample:
    """The 200 joint windows, their donors, and the tensors gathered for them."""

    node_ids: tuple[str, ...]
    joint_rows: np.ndarray
    episode_index: np.ndarray
    t: np.ndarray
    donor: np.ndarray
    refs: tuple[WindowRef, ...]
    tensors: Mapping[str, torch.Tensor]

    @property
    def n_windows(self) -> int:
        """How many joint windows were drawn."""
        return int(self.joint_rows.shape[0])


@dataclass(frozen=True)
class Subject:
    """One loaded checkpoint: the agent that ``act`` would use, plus the facts the run checks it against."""

    pin: CheckpointPin
    sha256: str
    agent: Any
    config: Any
    mask: np.ndarray
    node_ids: tuple[str, ...]
    stats_payload: Mapping[str, Any]
    rtg_scale: Mapping[str, float]
    scenario_id: str
    normalise: bool

    @property
    def n_layer(self) -> int:
        """The depth of the stack, which is the reach of the spatial layer (A22.1)."""
        return int(self.config.n_layer)


@dataclass(frozen=True)
class Influence:
    """F, the per-pair exact-equality result, and the legal-action counts the mean was taken over."""

    F: np.ndarray
    unchanged: np.ndarray
    legal_counts: np.ndarray


@dataclass(frozen=True)
class CheckpointFootprint:
    """Everything measured for one checkpoint."""

    pin: CheckpointPin
    sha256: str
    n_layer: int
    spatial_mixing: bool
    F: np.ndarray
    r: float
    att: float
    violations: tuple[tuple[str, str], ...]
    per_hop: Mapping[str, Any]
    legal_min: int
    legal_max: int


@dataclass(frozen=True)
class RunContext:
    """The provenance every checkpoint in one run shares."""

    node_ids: tuple[str, ...]
    corpus_dirs: tuple[str, ...]
    manifest_sha256: Mapping[str, str]
    stats_match_corpus: bool
    roadnet_sha256: str
    degree_histogram: Mapping[int, int]
    undirected_edges: int
    neighbours: np.ndarray
    hop_distances: np.ndarray
    context_length: int
    joint_windows_total: int
    eligible_windows: int
    torch_threads: int
    eval_cells: Mapping[str, Mapping[str, str]]


def _checkpoint_pins() -> tuple[CheckpointPin, ...]:
    """A22(a)'s ten checkpoints, in the registration's order: five spatial, then five nomix."""
    from offline.tier_sweep import _checkpoint_name

    digests: tuple[tuple[str, int, str], ...] = (
        (_SPATIAL_ARM, 101, "4d52b8b15502d398d0ee0f2a40abac83421f7082f2e40182a03a4a3743f50477"),
        (_SPATIAL_ARM, 202, "b34756f487914fdf0e146fbc7a7d3db18fe7cb658cb2191b9f2f4160715b63ae"),
        (_SPATIAL_ARM, 303, "92db70379e4c01074adad8eb28d39e74eb17e54b570ea0fa5938c772dc857d59"),
        (_SPATIAL_ARM, 404, "028643bc135911e241a1c7607615eca088b08f9ce0f042cdd64fe09666dfcb09"),
        (_SPATIAL_ARM, 505, "e34d5753eae6a6c76ff2113d7b62e73105e93e133a16055dcb9630cf643ed489"),
        (_NOMIX_ARM, 101, "329fb6b87fc8cf5c27530b2fe4d45212ea9c0775db3fb1e636a6bbc0a5004279"),
        (_NOMIX_ARM, 202, "f541358530399cc8282d4b8af63da7a83195bc421706714df0d460ff0d132fcc"),
        (_NOMIX_ARM, 303, "48076dab687da6b88ee0059b12e5edd49cea7d363deec84d30597bdc26e29906"),
        (_NOMIX_ARM, 404, "4b61bc064af2984e0d2e9666f03485ca901b87337a3a30c29c77302813eb4729"),
        (_NOMIX_ARM, 505, "09bd310ddc6ed337835d8a8fe9d29a16968cf2d339fabaae91927b4808e34cdc"),
    )
    return tuple(
        CheckpointPin(
            arm=arm,
            seed=seed,
            filename=_checkpoint_name(TIER, arm, seed),
            a22_prefix=digest[:8],
            sha256=digest,
        )
        for arm, seed, digest in digests
    )


A22_CHECKPOINTS: tuple[CheckpointPin, ...] = _checkpoint_pins()

#: A22(c)'s per-seed held-out ATT, to two decimals as registered.  Recomputed from the eval cells at run time and
#: refused on disagreement, so the artifact cannot quote a number the cell no longer holds.
A22_ATT: Mapping[tuple[str, int], float] = {
    (_SPATIAL_ARM, 101): 204.89,
    (_SPATIAL_ARM, 202): 192.79,
    (_SPATIAL_ARM, 303): 193.91,
    (_SPATIAL_ARM, 404): 176.46,
    (_SPATIAL_ARM, 505): 159.59,
    (_NOMIX_ARM, 101): 157.96,
    (_NOMIX_ARM, 202): 159.51,
    (_NOMIX_ARM, 303): 159.25,
    (_NOMIX_ARM, 404): 157.61,
    (_NOMIX_ARM, 505): 157.80,
}

#: The two P5.2 evaluation cells, pinned by digest: they are gitignored, so the artifact is where they are recorded.
EVAL_CELLS: Mapping[str, Mapping[str, str]] = {
    _SPATIAL_ARM: {
        "filename": f"eval_{TIER}_{_SPATIAL_ARM}.json",
        "sha256": "c3fcf047f3c4724695e96d39a049ed6c0d36ef52a5cc86d31d54022cb4429faa",
    },
    _NOMIX_ARM: {
        "filename": f"eval_{TIER}_{_NOMIX_ARM}.json",
        "sha256": "620a3e0a71109e206502ce1f3732f8146ad83766c1c7bf0080260d90ce807aea",
    },
}

#: A22(d), verbatim.  ``tests/test_spatial_footprint.py`` re-reads both clauses out of ``PREREGISTRATION.md``.
A22_OUTCOMES: tuple[Outcome, ...] = (
    Outcome(
        verdict=VERDICT_NON_TRIVIAL,
        sentence="spatial mixing harms at the best-data tier",
        a22_d_clause=(
            "*NON-TRIVIAL:* the paper's current sentence stands — **spatial mixing harms at the "
            "best-data tier** — and gains a measured basis: the trained models used "
            "cross-intersection information, and the harm is attributable to it."
        ),
    ),
    Outcome(
        verdict=VERDICT_TRIVIAL,
        sentence="adding the cross-intersection attention path destabilises training",
        a22_d_clause=(
            "*TRIVIAL:* the sentence *spatial mixing harms* is **NOT written**, because the trained "
            "models did not mix; it becomes **adding the cross-intersection attention path "
            "destabilises training**, citing the per-seed spread (sd 17.70 against 0.88 on the h4 "
            "arms; 30.36 against 0.104 on P5.1's) as the mechanism."
        ),
    ),
)

_ITEM_KEYS: tuple[str, ...] = (
    "rtg",
    "state",
    "action",
    "timestep",
    "avail_mask",
    "attention_mask",
)


# ----------------------------------------------------------------------
# The registered inputs, and their refusals.
# ----------------------------------------------------------------------


def checkpoint_pin(arm: str, seed: int) -> CheckpointPin:
    """The registered pin for one ``(arm, seed)``, refusing any pair A22(a) does not register."""
    for pin in A22_CHECKPOINTS:
        if pin.arm == str(arm) and pin.seed == int(seed):
            return pin
    raise ValueError(
        f"({arm!r}, {seed}) is not one of A22(a)'s ten registered checkpoints: "
        f"{sorted((p.arm, p.seed) for p in A22_CHECKPOINTS)}"
    )


def verify_checkpoint(path: str | Path, arm: str, seed: int) -> str:
    """The file's sha256, refused unless it equals A22(a)'s pin for *arm* and *seed*."""
    from offline.tier_sweep import sha256_of

    pin = checkpoint_pin(arm, seed)
    digest = str(sha256_of(path))
    if digest != pin.sha256:
        raise ValueError(
            f"{path} hashes to {digest[:8]}… but A22(a) registers {pin.a22_prefix}… for {arm} "
            f"seed {seed}. The ten checkpoints are registered by digest; a different file is a "
            "different model and its footprint is not the one A22 declared"
        )
    return digest


def verify_eval_cell(path: str | Path, arm: str) -> str:
    """The evaluation cell's sha256, refused unless it equals the pin for *arm*."""
    from offline.tier_sweep import sha256_of

    if str(arm) not in EVAL_CELLS:
        raise ValueError(f"{arm!r} has no registered evaluation cell; expected {sorted(EVAL_CELLS)}")
    expected = str(EVAL_CELLS[str(arm)]["sha256"])
    digest = str(sha256_of(path))
    if digest != expected:
        raise ValueError(
            f"{path} hashes to {digest[:8]}… but P5.4 pins {expected[:8]}… for {arm}. A22(c)'s ATT "
            "column is read from this cell, so a changed file would put a number beside r that "
            "P5.2 never produced"
        )
    return digest


def tier_directories(corpus_root: str | Path, tier: str = TIER) -> tuple[Path, ...]:
    """The tier's five corpus directories, refusing any tier other than ``mappo1000``.

    The refusal precedes every filesystem call: A22(a) registers one tier, and a run pointed at another would
    produce a plausible F against a corpus these models never saw.
    """
    if str(tier) != TIER:
        raise ValueError(
            f"P5.4 measures the {TIER!r} tier only, not {tier!r}: A22(a) registers ten checkpoints "
            "trained on it, and their normalisation statistics and return prompts come from it"
        )
    from offline.tier_sweep import tier_dirs, tier_spec

    return tuple(tier_dirs(tier_spec(TIER), corpus_root))


def footprint_dataset(
    corpus_root: str | Path,
    *,
    context_length: int,
    tier: str = TIER,
    draw_ids: Sequence[int] | None = None,
) -> Any:
    """The tier's training-split dataset with ``normalize=False``.

    Raw rows come out and normalisation is applied by ``act``'s own route (:func:`model_inputs`).  ``split="train"``
    is the mechanism that makes a held-out draw raise rather than enter the corpus.
    """
    from offline.dataset import TrajectoryWindowDataset

    directories = tier_directories(corpus_root, tier)
    return TrajectoryWindowDataset(
        [str(directory) for directory in directories],
        context_length=int(context_length),
        split="train",
        draw_ids=None if draw_ids is None else list(draw_ids),
        normalize=False,
    )


# ----------------------------------------------------------------------
# The windows and the pairing.
# ----------------------------------------------------------------------


def cyclic_derangement(count: int, rng: np.random.Generator) -> np.ndarray:
    """A single random cycle over ``count`` positions: a permutation with **no fixed point**.

    ``donor[sigma[k]] = sigma[(k + 1) % count]``.  Every window donates once and receives once, and no window is its
    own donor -- by construction rather than by rejection, so the draw stays deterministic in the seed.
    """
    size = int(count)
    if size < 2:
        raise ValueError(
            f"a pairing needs at least two windows, got {count}: with one window the donor could "
            "only be the window itself, and the substitution would measure nothing"
        )
    sigma = np.asarray(rng.permutation(size), dtype=np.int64)
    donor = np.empty(size, dtype=np.int64)
    donor[sigma] = np.roll(sigma, -1)
    return donor


def eligible_instants(t: np.ndarray, context_length: int) -> np.ndarray:
    """The joint rows a window may be drawn from: those with ``t >= context_length - 1``.

    A22.1(e) fixes full windows only.  An instant earlier than that is left-padded, and a padded donor block would
    put normalised zeros -- the per-feature mean, not an observation -- into a real position of the recipient.
    """
    steps = np.asarray(t, dtype=np.int64)
    span = int(context_length)
    if span < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")
    return np.flatnonzero(steps >= span - 1)


def draw_windows(
    dataset: Any,
    node_ids: Sequence[str],
    *,
    context_length: int,
    n_windows: int = N_WINDOWS,
    seed: int = WINDOW_SEED,
) -> WindowSample:
    """Draw *n_windows* full joint windows and pair them, through the corpus's own builders.

    Eligible instants are those with ``t >= context_length - 1``: a full window, no left padding, so every
    substituted row is a real observation (A22.1(e)).  One generator seeded with *seed* draws the windows first and
    the pairing second.

    Every alignment the gather depends on is verified from the loader's own ``item_meta`` rather than assumed: the
    node order of each instant, its ``t``, its episode, the last timestep of the gathered tensors and the absence of
    padding.
    """
    from offline.joint_windows import build_joint_index, stack_joint

    order = tuple(str(node) for node in node_ids)
    span = int(context_length)
    index = build_joint_index(dataset, order)

    eligible = eligible_instants(index.t, span)
    if eligible.size < int(n_windows):
        raise ValueError(
            f"only {eligible.size} of {index.n_windows} joint instants carry {span} real steps, "
            f"fewer than the {n_windows} windows requested; a padded window would put zero rows "
            "into a real position and A22.1(e) fixes full windows only"
        )

    rng = np.random.default_rng(int(seed))
    chosen = np.asarray(rng.choice(eligible, size=int(n_windows), replace=False), dtype=np.int64)
    donor = cyclic_derangement(int(n_windows), rng)

    stacked = stack_joint(dataset, index)
    members = stacked["member_index"][torch.from_numpy(chosen)]
    tensors = {key: stacked[key][members].contiguous() for key in _ITEM_KEYS}

    items = np.asarray(index.member_index)[chosen]
    refs: list[WindowRef] = []
    for position, row in enumerate(chosen.tolist()):
        metas = [dataset.item_meta(int(item)) for item in items[position]]
        if [str(meta.ix_id) for meta in metas] != list(order):
            raise ValueError(
                f"joint instant {row} does not carry the node order {order[:4]}…; the gather and "
                "the adjacency would index different intersections"
            )
        steps = {int(meta.t) for meta in metas}
        episodes = {int(meta.episode_index) for meta in metas}
        if steps != {int(index.t[row])} or episodes != {int(index.episode_index[row])}:
            raise ValueError(
                f"joint instant {row} spans steps {sorted(steps)} and episodes {sorted(episodes)}; "
                "a joint window is one decision instant of one episode"
            )
        refs.append(
            WindowRef(
                dataset_dir=Path(str(metas[0].dataset_dir)).name,
                episode_file=str(metas[0].episode_file),
                flow_draw=int(metas[0].flow_draw),
                t=int(metas[0].t),
            )
        )

    expected_t = torch.from_numpy(np.asarray(index.t)[chosen]).view(-1, 1).expand(
        int(n_windows), len(order)
    )
    if not torch.equal(tensors["timestep"][:, :, -1], expected_t.contiguous()):
        raise ValueError(
            "the last timestep of the gathered windows is not the instant they were drawn for; "
            "the flat rows and the joint index disagree"
        )
    if not bool(tensors["attention_mask"].all()):
        raise ValueError(
            "a drawn window carries padding, which A22.1(e) excludes: only windows with "
            f"{span} real steps may be used"
        )

    return WindowSample(
        node_ids=order,
        joint_rows=chosen,
        episode_index=np.asarray(index.episode_index)[chosen],
        t=np.asarray(index.t)[chosen],
        donor=donor,
        refs=tuple(refs),
        tensors=tensors,
    )


# ----------------------------------------------------------------------
# The subject, and the inputs ``act`` would build.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _NodeOrderIntersection:
    """What ``Utils.infer_action_counts`` reads when there is no gym action space."""

    id: str
    num_phases: int


class _NodeOrderEnv:
    """The smallest object ``SpatialDTAgent`` reads at construction: a node order and an action count.

    No environment is built here and none is needed: the checkpoint carries its own graph, statistics and return
    prompts, and :meth:`SpatialDTAgent.load` checks this order against the one it recorded.
    """

    def __init__(self, node_ids: Sequence[str], n_actions: int) -> None:
        self.intersections = [
            _NodeOrderIntersection(id=str(node), num_phases=int(n_actions)) for node in node_ids
        ]
        self.action_space = None


def load_subject(path: str | Path, node_ids: Sequence[str], arm: str, seed: int) -> Subject:
    """Load one checkpoint through ``SpatialDTAgent.from_checkpoint`` on a node-order-only env.

    Going through the agent runs every refusal its ``load`` carries -- format version, action count, node count,
    intersection order, normalise-without-statistics and a missing mask -- and gives back the object whose
    ``_normalise_state`` and mask the measurement then uses.
    """
    from agent.SpatialDTAgent import SpatialDTAgent, SpatialDTConfig

    pin = checkpoint_pin(arm, seed)
    digest = verify_checkpoint(path, arm, seed)
    order = tuple(str(node) for node in node_ids)

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    config = SpatialDTConfig.from_json_obj(payload["config"])
    agent = SpatialDTAgent.from_checkpoint(
        _NodeOrderEnv(order, config.n_actions), str(path), device="cpu"
    )

    provenance = dict(payload.get("provenance", {}))
    if str(provenance.get("tier", "")) != TIER:
        raise ValueError(
            f"{path} was trained on tier {provenance.get('tier')!r}, not {TIER!r}"
        )
    if str(provenance.get("method", "")) != pin.arm or int(provenance.get("seed", -1)) != pin.seed:
        raise ValueError(
            f"{path} records method {provenance.get('method')!r} seed {provenance.get('seed')!r}, "
            f"not the registered {pin.arm!r} seed {pin.seed}"
        )
    if config.spatial_mixing is not pin.spatial_mixing:
        raise ValueError(
            f"{path} records spatial_mixing={config.spatial_mixing} but {pin.arm} is the "
            f"{'mixing' if pin.spatial_mixing else 'control'} arm"
        )
    mask = np.asarray(agent.spatial_mask, dtype=np.bool_)
    if not np.array_equal(mask, np.asarray(payload["spatial_mask"], dtype=np.bool_)):
        raise ValueError(f"{path}: the agent did not adopt the mask the checkpoint recorded")
    if not bool(np.diagonal(mask).all()):
        raise ValueError(
            f"{path}: the recorded mask has a closed diagonal, so its boolean power is not the "
            "receptive field and A22.1's reach cannot be computed from it"
        )
    if tuple(agent.intersection_ids) != order:
        raise ValueError(
            f"{path} records the order {tuple(agent.intersection_ids)[:4]}…, not the corpus's "
            f"{order[:4]}…"
        )
    if not bool(payload.get("normalise", False)) or payload.get("stats") is None:
        raise ValueError(
            f"{path} carries no normalisation statistics; these models were trained on normalised "
            "states and feeding them raw ones would measure a different model"
        )

    return Subject(
        pin=pin,
        sha256=digest,
        agent=agent,
        config=config,
        mask=mask,
        node_ids=order,
        stats_payload=dict(payload["stats"]),
        rtg_scale={str(k): float(v) for k, v in payload["rtg_scale"].items()},
        scenario_id=str(payload["scenario_id"]),
        normalise=bool(payload["normalise"]),
    )


def model_inputs(sample: WindowSample, subject: Subject) -> dict[str, torch.Tensor]:
    """The model-ready tensors for *sample* under *subject*'s payload.

    State: ``act``'s own per-intersection normalisation, cross-checked against the dataset's block route under exact
    equality.  Return-to-go: divided by each node's ``rtg_scale`` as the trainer divided it.  Actions, timesteps and
    both masks come with the window untouched.
    """
    from offline.dataset import NormalizationStats

    if tuple(sample.node_ids) != tuple(subject.node_ids):
        raise ValueError(
            f"the sample's node order {sample.node_ids[:4]}… is not the checkpoint's "
            f"{subject.node_ids[:4]}…"
        )
    raw = sample.tensors["state"]
    n_windows, n_nodes, span, width = raw.shape
    if n_nodes != len(subject.node_ids):
        raise ValueError(f"the sample carries {n_nodes} nodes, the checkpoint {len(subject.node_ids)}")

    reference_stats = NormalizationStats.from_json_obj(dict(subject.stats_payload))
    normalised = torch.empty_like(raw)
    for position, ix_id in enumerate(subject.node_ids):
        block = raw[:, position].reshape(-1, width).numpy()
        # act's own route, row by row, exactly as SpatialDTAgent.act normalises one decision's state.
        through_act = np.stack(
            [subject.agent._normalise_state(ix_id, row) for row in block]
        ).astype(np.float32)
        # The dataset's route over the whole block, as a second computation of the same quantity.
        if subject.normalise:
            through_dataset = reference_stats.normalize_state(subject.scenario_id, ix_id, block)
        else:
            through_dataset = np.asarray(block, dtype=np.float32)
        if not np.array_equal(through_act, through_dataset):
            raise ValueError(
                f"the agent's normalisation of {ix_id} differs from the dataset's route; the "
                "windows the model is being fed are not the windows it was trained on"
            )
        normalised[:, position] = torch.from_numpy(
            through_act.reshape(n_windows, span, width)
        )

    scale = torch.tensor(
        [float(subject.rtg_scale[ix_id]) for ix_id in subject.node_ids], dtype=torch.float32
    ).view(1, n_nodes, 1, 1)

    return {
        "rtg": sample.tensors["rtg"] / scale,
        "state": normalised,
        "action": sample.tensors["action"],
        "timestep": sample.tensors["timestep"],
        "attention_mask": sample.tensors["attention_mask"],
        "avail_mask": sample.tensors["avail_mask"],
    }


# ----------------------------------------------------------------------
# Reach, and the support the mask actually has.
# ----------------------------------------------------------------------


def reachability(mask: np.ndarray, hops: int) -> np.ndarray:
    """Which nodes reach which within *hops* applications of *mask* (its boolean matrix power).

    The recorded masks carry self-loops, so this is exactly the receptive field of a *hops*-layer stack.
    """
    allowed = np.asarray(mask, dtype=np.bool_)
    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError(f"the mask must be square, got {allowed.shape}")
    depth = int(hops)
    if depth < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")
    out = np.eye(allowed.shape[0], dtype=np.bool_)
    for _ in range(depth):
        out = (out.astype(np.int64) @ allowed.astype(np.int64)) > 0
    return np.asarray(out, dtype=np.bool_)


def hop_distances(mask: np.ndarray) -> np.ndarray:
    """``(N, N)`` graph distance in hops, ``-1`` where unreachable; the diagonal is 0."""
    allowed = np.asarray(mask, dtype=np.bool_)
    size = allowed.shape[0]
    distances = np.full((size, size), -1, dtype=np.int64)
    for depth in range(size):
        reached = reachability(allowed, depth)
        fresh = reached & (distances < 0)
        distances[fresh] = depth
        if bool(reached.all()):
            break
    return distances


def must_be_zero_pairs(mask: np.ndarray, n_layer: int) -> np.ndarray:
    """A22.1(b): the pairs whose influence must be **exactly** zero -- those beyond the receptive field.

    Spatial: more than ``n_layer`` hops apart on the recorded mask.  Nomix: every off-diagonal pair, because the
    identity's reach is itself at any depth.
    """
    return ~reachability(mask, int(n_layer))


# ----------------------------------------------------------------------
# The measurement.
# ----------------------------------------------------------------------


def _last_step_logits(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    spatial_mask: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """``masked_action_logits`` at the last decision of every window and node, as the model emits them."""
    logits = model(
        inputs["rtg"],
        state,
        inputs["action"],
        inputs["timestep"],
        spatial_mask,
        inputs["attention_mask"],
        inputs.get("avail_mask"),
    )
    return logits[:, :, -1, :]


def influence(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    spatial_mask: np.ndarray,
    donor: np.ndarray,
) -> Influence:
    """Substitute each node's state block in turn and measure what moved.

    One base forward plus one forward per node, all at the same shape so the two are comparable bit for bit.  The
    model runs in eval mode under ``no_grad``, as ``act`` runs it, so dropout is off.

    Two independent routes to the same fact, and they must agree: ``F[i, j] == 0.0`` exactly, and ``torch.equal`` on
    the raw per-window logits.  A disagreement is refused rather than reported.
    """
    state = inputs["state"]
    n_windows, n_nodes = int(state.shape[0]), int(state.shape[1])
    pairing = np.asarray(donor, dtype=np.int64)
    if pairing.shape != (n_windows,):
        raise ValueError(
            f"the pairing has {pairing.shape} entries for {n_windows} windows"
        )
    if bool(np.any(pairing == np.arange(n_windows))):
        raise ValueError(
            "a window is its own donor, so its substitution would change nothing and would enter "
            "F as a zero"
        )
    mask_tensor = torch.from_numpy(np.asarray(spatial_mask, dtype=np.bool_))
    donor_tensor = torch.from_numpy(pairing)

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            base = _last_step_logits(model, inputs, mask_tensor, state)
            avail = inputs.get("avail_mask")
            legal = (
                torch.ones_like(base, dtype=torch.bool)
                if avail is None
                else avail[:, :, -1, :].to(torch.bool)
            )
            counts = legal.sum(-1)
            if int(counts.min()) < 1:
                raise ValueError(
                    "a window has a node with no legal action at its last step; the mean absolute "
                    "change over legal actions is undefined there"
                )
            divisor = counts.to(torch.float64)

            matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
            unchanged = np.zeros((n_nodes, n_nodes), dtype=np.bool_)
            for node in range(n_nodes):
                perturbed = state.clone()
                perturbed[:, node] = state[donor_tensor, node]
                moved = _last_step_logits(model, inputs, mask_tensor, perturbed)
                difference = (moved.to(torch.float64) - base.to(torch.float64)).abs()
                # -inf at an illegal action is a constant, not an output: -inf - -inf is NaN, so the
                # mean runs over the legal entries and torch.where keeps the NaN out of it.
                selected = torch.where(legal, difference, torch.zeros_like(difference))
                per_window = selected.sum(-1) / divisor
                matrix[:, node] = per_window.mean(0).numpy()
                for other in range(n_nodes):
                    unchanged[other, node] = bool(torch.equal(moved[:, other], base[:, other]))
    finally:
        model.train(was_training)

    if not np.array_equal(matrix == 0.0, unchanged):
        raise ValueError(
            "the mean absolute change and the exact-equality check disagree about which pairs "
            "moved; one of the two routes is wrong and neither may be reported"
        )
    return Influence(F=matrix, unchanged=unchanged, legal_counts=counts.numpy())


def support_violations(
    result: Influence, must_zero: np.ndarray, node_ids: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """The ``(i, j)`` id pairs that had to be exactly zero and were not."""
    order = [str(node) for node in node_ids]
    required = np.asarray(must_zero, dtype=np.bool_)
    if required.shape != result.unchanged.shape:
        raise ValueError(
            f"the must-be-zero set is {required.shape} and the footprint is "
            f"{result.unchanged.shape}"
        )
    if len(order) != required.shape[0]:
        raise ValueError(f"{len(order)} node ids for a {required.shape[0]}-node footprint")
    offenders = np.argwhere(required & ~result.unchanged)
    return tuple((order[int(i)], order[int(j)]) for i, j in offenders)


def footprint_ratio(F: np.ndarray, neighbours: np.ndarray) -> float:
    """A22(c)'s r: mean one-hop neighbour influence over mean self influence, in float64.

    *neighbours* must carry no diagonal entry: ``F[i, i]`` is the denominator, and counting it in the numerator
    would inflate r by the very quantity it is measured against.
    """
    matrix = np.asarray(F, dtype=np.float64)
    adjacency = np.asarray(neighbours, dtype=np.bool_)
    if matrix.shape != adjacency.shape or matrix.ndim != 2:
        raise ValueError(f"F is {matrix.shape} and the neighbour set is {adjacency.shape}")
    if bool(np.diagonal(adjacency).any()):
        raise ValueError(
            "the neighbour set carries a diagonal entry; A22(c)'s numerator is the mean over "
            "NEIGHBOURS and its denominator is the self influence, so counting F[i, i] in both "
            "would inflate r by its own denominator"
        )
    empty = np.flatnonzero(~adjacency.any(axis=1))
    if empty.size:
        raise ValueError(
            f"nodes {empty[:8].tolist()} have no neighbour, so the mean over neighbours is "
            "undefined for them"
        )
    per_node = np.array(
        [matrix[i, adjacency[i]].mean() for i in range(matrix.shape[0])], dtype=np.float64
    )
    numerator = float(per_node.mean())
    denominator = float(np.diagonal(matrix).mean())
    if denominator == 0.0:
        raise ValueError(
            "the mean self influence is exactly 0, so r is undefined: this model's own state does "
            "not move its logits at all, which is a defect and not a footprint"
        )
    return numerator / denominator


def verdict(median_r: float) -> str:
    """``TRIVIAL`` iff *median_r* is **below** 0.10; the threshold itself is ``NON-TRIVIAL`` (A22(c))."""
    return VERDICT_TRIVIAL if float(median_r) < TRIVIAL_THRESHOLD else VERDICT_NON_TRIVIAL


def median_verdict(values: Sequence[float]) -> tuple[float, str]:
    """The median of the five spatial seeds' r, and the verdict it selects."""
    ratios = [float(value) for value in values]
    if not ratios:
        raise ValueError("no r values: the median over the spatial seeds is undefined")
    median = float(statistics.median(ratios))
    return median, verdict(median)


def per_hop_report(F: np.ndarray, distances: np.ndarray, n_layer: int) -> dict[str, Any]:
    """A22.1(b)'s descriptive block: mean F at each hop within reach, and the maximum beyond it.

    Distances are measured on the **lattice**, for both arms, so the control's zeros sit in the same columns as the
    treatment's numbers.  No threshold attaches to any of this.
    """
    matrix = np.asarray(F, dtype=np.float64)
    hops = np.asarray(distances, dtype=np.int64)
    if matrix.shape != hops.shape:
        raise ValueError(f"F is {matrix.shape} and the distances are {hops.shape}")
    depth = int(n_layer)

    mean_by_hop: dict[str, float | None] = {}
    pairs_by_hop: dict[str, int] = {}
    for hop in range(1, depth + 1):
        at_hop = hops == hop
        pairs_by_hop[str(hop)] = int(at_hop.sum())
        mean_by_hop[str(hop)] = float(matrix[at_hop].mean()) if bool(at_hop.any()) else None

    beyond = (hops > depth) | (hops < 0)
    return {
        "n_layer": depth,
        "mean_by_hop": mean_by_hop,
        "pairs_by_hop": pairs_by_hop,
        "pairs_beyond_reach": int(beyond.sum()),
        "max_abs_beyond_reach": float(np.abs(matrix[beyond]).max()) if bool(beyond.any()) else 0.0,
        "distances_measured_on": "the lattice (undirected relation plus self-loops), for both arms",
    }


def per_seed_att(payload: Mapping[str, Any], arm: str, seed: int) -> float:
    """One seed's held-out ATT from a P5.2 evaluation cell, by P5.x's own per-seed convention."""
    from offline.dt_gate import HELD_OUT_DRAWS
    from offline.tier_sweep import episodes_of_seed

    if str(payload.get("method")) != str(arm) or str(payload.get("tier")) != TIER:
        raise ValueError(
            f"this cell is {payload.get('method')!r} at {payload.get('tier')!r}, not {arm!r} at "
            f"{TIER!r}"
        )
    episodes = episodes_of_seed(payload, int(seed))
    if tuple(sorted(episodes)) != tuple(HELD_OUT_DRAWS):
        raise ValueError(
            f"{arm} seed {seed} carries {len(episodes)} draws, not the {len(HELD_OUT_DRAWS)} "
            "held-out draws A22(c)'s ATT is the mean over"
        )
    value = float(np.mean([float(episode["att_horizon"]) for episode in episodes.values()]))
    registered = float(A22_ATT[(str(arm), int(seed))])
    if round(value, 2) != registered:
        raise ValueError(
            f"{arm} seed {seed} means {value:.4f}, which does not round to A22(c)'s {registered}; "
            "the registration quotes this cell and the two must be the same measurement"
        )
    return value


# ----------------------------------------------------------------------
# The artifact.
# ----------------------------------------------------------------------


def _protocol_block(context: RunContext) -> dict[str, Any]:
    """Every choice the measurement made, in the words of the registration that fixed it."""
    return {
        "window_seed": WINDOW_SEED,
        "n_windows": N_WINDOWS,
        "context_length": context.context_length,
        "eligibility": "joint instants with t >= context_length - 1 (a full window, no left padding)",
        "joint_windows_total": context.joint_windows_total,
        "eligible_windows": context.eligible_windows,
        "draw": (
            "one numpy default_rng(window_seed) draws the windows first "
            "(choice without replacement over the eligible rows, ascending) and the pairing second"
        ),
        "pairing": (
            "a single random cyclic derangement: donor[sigma[k]] = sigma[(k + 1) % n_windows], so "
            "every window donates once, receives once, and is never its own donor"
        ),
        "perturbation": (
            "node j's whole K-step state block is replaced by node j's block from its donor window; "
            "return-to-go, actions, timesteps, both masks and every other node are untouched"
        ),
        "readout": (
            "the last decision step's masked_action_logits as the model emits them, before argmax; "
            "the mean absolute change is taken over the legal actions in float64, then over windows"
        ),
        "support_check": (
            "torch.equal on the raw per-window logits, for every pair more than n_layer hops apart "
            "on the recorded mask (A22.1(b)); a violation refuses this artifact"
        ),
        "statistic": (
            "r = mean_i(mean over one-hop neighbours j of F[i, j]) / mean_i F[i, i], float64; "
            "TRIVIAL iff the median over the five spatial seeds is below 0.10"
        ),
        "normalisation": (
            "SpatialDTAgent._normalise_state -- the method act calls -- with the checkpoint's own "
            "statistics, cross-checked row for row against NormalizationStats.normalize_state and "
            "refused on any difference"
        ),
        "rtg": "the window's return-to-go divided by each node's recorded rtg_scale, as the trainer divided it",
        "model_mode": "eval, torch.no_grad, CPU; dropout off",
        "torch_threads": context.torch_threads,
    }


def build_artifact(
    footprints: Sequence[CheckpointFootprint], sample: WindowSample, context: RunContext
) -> dict[str, Any]:
    """Assemble the artifact.  Violations are **recorded**, not refused here: :func:`write_artifact` is the gate."""
    from offline.dt_gate import runtime_provenance

    if not footprints:
        raise ValueError("no footprints to assemble")

    spatial = [item for item in footprints if item.spatial_mixing]
    nomix = [item for item in footprints if not item.spatial_mixing]
    if not spatial:
        raise ValueError("no spatial checkpoint: the verdict is the median over the spatial seeds")

    ordered_spatial = sorted(spatial, key=lambda item: item.pin.seed)
    median, chosen = median_verdict([item.r for item in ordered_spatial])

    def att_of(group: Sequence[CheckpointFootprint]) -> list[float]:
        return [item.att for item in sorted(group, key=lambda entry: entry.pin.seed)]

    checkpoints = [
        {
            "arm": item.pin.arm,
            "seed": item.pin.seed,
            "path": f"{CANONICAL_CHECKPOINT_DIR}/{item.pin.filename}",
            "sha256": item.sha256,
            "a22_prefix": item.pin.a22_prefix,
            "spatial_mixing": bool(item.spatial_mixing),
            "n_layer": int(item.n_layer),
            "F": [[float(value) for value in row] for row in np.asarray(item.F, dtype=np.float64)],
            "r": float(item.r),
            "neighbour_to_self": {
                "self_mean": float(np.diagonal(np.asarray(item.F, dtype=np.float64)).mean()),
            },
            "att_heldout_p5_2": float(item.att),
            "per_hop": dict(item.per_hop),
            "legal_actions_at_last_step": {"min": int(item.legal_min), "max": int(item.legal_max)},
            "support_check": {
                "passed": not item.violations,
                "violations": [list(pair) for pair in item.violations],
                "rule": (
                    "exactly zero beyond n_layer hops on the recorded mask (spatial) or at every "
                    "off-diagonal pair (nomix), compared with torch.equal per window"
                ),
            },
        }
        for item in sorted(footprints, key=lambda entry: (entry.pin.arm, entry.pin.seed))
    ]

    return {
        "format_version": FORMAT_VERSION,
        "task": "P5.4",
        "registration": {
            "amendment": "A22",
            "correction": "A22.1",
            "tag": "v2.3-prereg-a22-1",
            "threshold": TRIVIAL_THRESHOLD,
            "rule": "TRIVIAL iff the median of r over the five spatial seeds is below 0.10",
        },
        "protocol": _protocol_block(context),
        "corpus": {
            "tier": TIER,
            "root": CANONICAL_CORPUS_ROOT,
            "dirs": list(context.corpus_dirs),
            "manifest_sha256": dict(context.manifest_sha256),
            "statistics_equal_checkpoint_payloads": bool(context.stats_match_corpus),
        },
        "graph": {
            "roadnet_sha256": context.roadnet_sha256,
            "degree_histogram": {str(k): int(v) for k, v in dict(context.degree_histogram).items()},
            "undirected_edges": int(context.undirected_edges),
            "neighbour_rule": (
                "the undirected relation without self-loops, derived by offline.roadnet_graph from "
                "the corpus's own network; the same set is used for both arms"
            ),
            # Recorded so r and the per-hop block are recomputable from this file alone, without
            # re-deriving the graph from the road network.
            "neighbour_mask": [
                [bool(value) for value in row]
                for row in np.asarray(context.neighbours, dtype=np.bool_)
            ],
            "hop_distances": [
                [int(value) for value in row]
                for row in np.asarray(context.hop_distances, dtype=np.int64)
            ],
        },
        "node_ids": list(context.node_ids),
        "windows": {
            "n": sample.n_windows,
            "joint_rows": [int(row) for row in sample.joint_rows],
            "donor": [int(value) for value in sample.donor],
            "pairs_within_same_episode": int(
                np.sum(np.asarray(sample.episode_index)[np.asarray(sample.donor)] == np.asarray(sample.episode_index))
            ),
            "refs": [
                {
                    "dataset_dir": ref.dataset_dir,
                    "episode_file": ref.episode_file,
                    "flow_draw": ref.flow_draw,
                    "t": ref.t,
                }
                for ref in sample.refs
            ],
        },
        "checkpoints": checkpoints,
        "eval_cells": {arm: dict(cell) for arm, cell in context.eval_cells.items()},
        "summary": {
            "r_by_seed": {
                "dt_spatial_h4": {
                    str(item.pin.seed): float(item.r) for item in ordered_spatial
                },
                "dt_nomix_h4": {
                    str(item.pin.seed): float(item.r)
                    for item in sorted(nomix, key=lambda entry: entry.pin.seed)
                },
            },
            "median_r_spatial": median,
            "threshold": TRIVIAL_THRESHOLD,
            "verdict": chosen,
            "att_heldout_by_seed": {
                "dt_spatial_h4": {
                    str(item.pin.seed): float(item.att) for item in ordered_spatial
                },
                "dt_nomix_h4": {
                    str(item.pin.seed): float(item.att)
                    for item in sorted(nomix, key=lambda entry: entry.pin.seed)
                },
            },
            "att_sd_across_seeds": {
                "dt_spatial_h4": (
                    float(statistics.stdev(att_of(spatial))) if len(spatial) > 1 else None
                ),
                "dt_nomix_h4": (
                    float(statistics.stdev(att_of(nomix))) if len(nomix) > 1 else None
                ),
            },
        },
        "paper_sentences": {
            outcome.verdict: {
                "sentence": outcome.sentence,
                "a22_d_clause": outcome.a22_d_clause,
                "chosen": outcome.verdict == chosen,
            }
            for outcome in A22_OUTCOMES
        },
        "what_this_does_not_say": [
            "one tier only: cf_grid4x4__mappo1000, the tier these ten checkpoints trained on",
            "200 windows, drawn once and shared by all ten checkpoints",
            "full 20-step windows only, so the first 19 decisions of every episode are not represented",
            "the last decision step's logits only, and a state-only substitution",
            "r is a ratio of means over one-hop neighbours; two- and three-hop influence is reported "
            "per hop but enters no statistic and no verdict",
            "the per-hop block measures reach, not use",
            "n = 5 for any relation between r and ATT, which is descriptive and untested",
        ],
        "runtime": runtime_provenance(),
    }


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    """Validate the assembled artifact and write it atomically.

    Filesystem-mutation barrier: every refusal below precedes any filesystem call, so a refused run leaves a previous
    artifact untouched and creates no directory.  Refused: a support violation, a non-finite F, a nomix r that is not
    exactly 0.0, influence beyond the receptive field, a digest that is not A22(a)'s, a verdict that disagrees with
    the median it records, and a missing or edited A22(d) clause.
    """
    from offline.dt_gate import write_json_atomic

    if str(artifact.get("format_version")) != FORMAT_VERSION:
        raise ValueError(
            f"format version {artifact.get('format_version')!r} is not {FORMAT_VERSION!r}"
        )

    entries = list(artifact.get("checkpoints", ()))
    if not entries:
        raise ValueError("the artifact carries no checkpoint")

    spatial_ratios: list[float] = []
    for entry in entries:
        arm, seed = str(entry["arm"]), int(entry["seed"])
        pin = checkpoint_pin(arm, seed)
        if str(entry["sha256"]) != pin.sha256:
            raise ValueError(
                f"{arm} seed {seed} records digest {str(entry['sha256'])[:8]}…, not A22(a)'s "
                f"{pin.a22_prefix}…"
            )
        violations = list(entry["support_check"]["violations"])
        if violations:
            raise ValueError(
                f"the support check failed for {arm} seed {seed} on {len(violations)} pair(s), "
                f"first {violations[:4]}: influence reached beyond the recorded mask's "
                "receptive field. A22.1(b) makes that a defect in the mask's application to "
                "trained models, so nothing is written and the task stops"
            )
        matrix = np.asarray(entry["F"], dtype=np.float64)
        if not bool(np.isfinite(matrix).all()):
            raise ValueError(f"{arm} seed {seed} has a non-finite influence entry")
        beyond = float(entry["per_hop"]["max_abs_beyond_reach"])
        if beyond != 0.0:
            raise ValueError(
                f"{arm} seed {seed} records influence {beyond} beyond its receptive field, which "
                "the support check should already have refused"
            )
        if bool(entry["spatial_mixing"]):
            spatial_ratios.append(float(entry["r"]))
        elif float(entry["r"]) != 0.0:
            raise ValueError(
                f"the control {arm} seed {seed} has r = {entry['r']!r}, not exactly 0.0; with the "
                "identity mask no neighbour can influence anything and r is 0 by construction"
            )

    summary = dict(artifact.get("summary", {}))
    median, chosen = median_verdict(spatial_ratios)
    if float(summary.get("median_r_spatial", float("nan"))) != median:
        raise ValueError(
            f"the recorded median {summary.get('median_r_spatial')!r} is not the median "
            f"{median!r} of the per-seed r this artifact carries"
        )
    if str(summary.get("verdict")) != chosen:
        raise ValueError(
            f"the recorded verdict {summary.get('verdict')!r} is not the verdict {chosen!r} that "
            f"the threshold {TRIVIAL_THRESHOLD} selects for median {median!r}"
        )

    sentences = dict(artifact.get("paper_sentences", {}))
    if set(sentences) != {outcome.verdict for outcome in A22_OUTCOMES}:
        raise ValueError(
            f"the artifact carries {sorted(sentences)} where A22(d) writes both outcomes: the road "
            "not taken stays visible"
        )
    for outcome in A22_OUTCOMES:
        block = dict(sentences[outcome.verdict])
        if str(block.get("a22_d_clause")) != outcome.a22_d_clause:
            raise ValueError(f"A22(d)'s {outcome.verdict} clause is not carried verbatim")
        if str(block.get("sentence")) != outcome.sentence:
            raise ValueError(f"A22(d)'s {outcome.verdict} sentence is not carried verbatim")
        if bool(block.get("chosen")) != (outcome.verdict == chosen):
            raise ValueError(
                f"{outcome.verdict} is marked chosen={block.get('chosen')!r} under verdict {chosen!r}"
            )

    write_json_atomic(dict(artifact), path)


# ----------------------------------------------------------------------
# The command line.
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI: where the checkpoints, the corpus, the evaluation cells and the artifact are."""
    parser = argparse.ArgumentParser(
        description="P5.4: the trained-model footprint of the spatial layer (A22 / A22.1)"
    )
    parser.add_argument("--checkpoint-dir", default=CANONICAL_CHECKPOINT_DIR)
    parser.add_argument("--corpus-root", default=CANONICAL_CORPUS_ROOT)
    parser.add_argument("--eval-dir", default=CANONICAL_EVAL_DIR)
    parser.add_argument("--out", default="docs/data/p5_2_spatial_footprint.json")
    parser.add_argument("--threads", type=int, default=TORCH_THREADS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Measure all ten checkpoints and write the artifact.  Returns non-zero without writing on a violation."""
    from offline.dataset import NormalizationStats
    from offline.tier_sweep import (
        adjacency_for_tier,
        node_ids_from_corpus,
        sha256_of,
        tier_spec,
    )

    args = build_parser().parse_args(None if argv is None else list(argv))
    torch.set_num_threads(int(args.threads))

    checkpoint_dir = Path(args.checkpoint_dir)
    eval_dir = Path(args.eval_dir)
    corpus_root = Path(args.corpus_root)

    # ---- inputs, all verified before anything is loaded or measured -------------------------
    paths = {
        (pin.arm, pin.seed): checkpoint_dir / pin.filename for pin in A22_CHECKPOINTS
    }
    for (arm, seed), path in paths.items():
        verify_checkpoint(path, arm, seed)

    eval_payloads: dict[str, Any] = {}
    eval_cells: dict[str, dict[str, str]] = {}
    for arm, cell in EVAL_CELLS.items():
        cell_path = eval_dir / str(cell["filename"])
        digest = verify_eval_cell(cell_path, arm)
        eval_payloads[arm] = json.loads(cell_path.read_text(encoding="utf-8"))
        eval_cells[arm] = {
            "path": f"{CANONICAL_EVAL_DIR}/{cell['filename']}",
            "sha256": digest,
        }

    spec = tier_spec(TIER)
    directories = tier_directories(corpus_root)
    node_ids = tuple(node_ids_from_corpus(spec, corpus_root))
    adjacency = adjacency_for_tier(spec, corpus_root, node_ids)
    lattice = np.asarray(adjacency.attention_mask(spatial_mixing=True), dtype=np.bool_)
    neighbours = np.asarray(adjacency.undirected, dtype=np.bool_)
    distances = hop_distances(lattice)

    subjects = [
        load_subject(paths[(pin.arm, pin.seed)], node_ids, pin.arm, pin.seed)
        for pin in A22_CHECKPOINTS
    ]
    spans = {subject.config.context_length for subject in subjects}
    depths = {subject.n_layer for subject in subjects}
    if len(spans) != 1 or len(depths) != 1:
        raise ValueError(
            f"the ten checkpoints disagree on context_length {sorted(spans)} or n_layer "
            f"{sorted(depths)}; one window set cannot serve them"
        )
    span = int(spans.pop())
    for subject in subjects:
        expected = adjacency.attention_mask(spatial_mixing=subject.pin.spatial_mixing)
        if not np.array_equal(subject.mask, np.asarray(expected, dtype=np.bool_)):
            raise ValueError(
                f"{subject.pin.arm} seed {subject.pin.seed} records a mask that is not its arm's "
                "graph; the footprint would be measured under a graph the model never trained on"
            )

    # ---- the corpus, and the proof it is the one these models were normalised against --------
    dataset = footprint_dataset(corpus_root, context_length=span)
    fitted = dataset.stats.to_json_obj()
    stats_match = True
    for subject in subjects:
        payload = dict(subject.stats_payload)
        for key in ("state_mean", "state_std", "row_count", "draw_ids", "split", "rtg"):
            if payload[key] != fitted[key]:
                stats_match = False
                raise ValueError(
                    f"{subject.pin.arm} seed {subject.pin.seed} carries {key} that the loaded "
                    "corpus does not reproduce; this is not the corpus it was trained on"
                )
        NormalizationStats.from_json_obj(payload)

    from offline.joint_windows import build_joint_index

    index = build_joint_index(dataset, node_ids)
    eligible = int(eligible_instants(index.t, span).size)
    sample = draw_windows(dataset, node_ids, context_length=span)

    context = RunContext(
        node_ids=node_ids,
        corpus_dirs=tuple(directory.name for directory in directories),
        manifest_sha256={
            directory.name: str(sha256_of(directory / "manifest.json"))
            for directory in directories
        },
        stats_match_corpus=stats_match,
        roadnet_sha256=str(adjacency.roadnet_sha256),
        degree_histogram=adjacency.degree_histogram(),
        undirected_edges=len(adjacency.undirected_edges()),
        neighbours=neighbours,
        hop_distances=distances,
        context_length=span,
        joint_windows_total=int(index.n_windows),
        eligible_windows=eligible,
        torch_threads=int(args.threads),
        eval_cells=eval_cells,
    )

    # ---- the measurement ---------------------------------------------------------------------
    footprints: list[CheckpointFootprint] = []
    for subject in subjects:
        inputs = model_inputs(sample, subject)
        result = influence(subject.agent.model, inputs, subject.mask, sample.donor)
        violations = support_violations(
            result, must_be_zero_pairs(subject.mask, subject.n_layer), node_ids
        )
        ratio = footprint_ratio(result.F, neighbours)
        footprints.append(
            CheckpointFootprint(
                pin=subject.pin,
                sha256=subject.sha256,
                n_layer=subject.n_layer,
                spatial_mixing=subject.pin.spatial_mixing,
                F=result.F,
                r=ratio,
                att=per_seed_att(eval_payloads[subject.pin.arm], subject.pin.arm, subject.pin.seed),
                violations=violations,
                per_hop=per_hop_report(result.F, distances, subject.n_layer),
                legal_min=int(result.legal_counts.min()),
                legal_max=int(result.legal_counts.max()),
            )
        )
        print(
            f"  {subject.pin.arm} seed {subject.pin.seed}: r = {ratio:.6f}, "
            f"violations {len(violations)}",
            flush=True,
        )

    artifact = build_artifact(footprints, sample, context)
    offenders = [
        (entry["arm"], entry["seed"], entry["support_check"]["violations"])
        for entry in artifact["checkpoints"]
        if entry["support_check"]["violations"]
    ]
    if offenders:
        for arm, seed, pairs in offenders:
            print(
                f"SUPPORT VIOLATION {arm} seed {seed}: {len(pairs)} pair(s), first {pairs[:4]}",
                flush=True,
            )
        print("nothing written: A22.1(b) makes this a defect, not a result", flush=True)
        return 1

    write_artifact(artifact, args.out)
    summary = artifact["summary"]
    print(
        f"median r = {summary['median_r_spatial']:.6f} over the five spatial seeds; "
        f"verdict {summary['verdict']} (threshold {TRIVIAL_THRESHOLD})"
    )
    print(f"artifact written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
