"""P7.3b section 3.3 -- the C3 curve's FULL-RETRAIN ANCHOR: corpus, prompt, five seeds.

Registered design, implemented here and chosen nowhere in this file
------------------------------------------------------------------
``PREREGISTRATION`` A18(a), quoted in ``docs/briefs/BRIEF_38_p7.3b_anchor.md`` section 1:

    a DT trained from scratch, on SUMO, on 200 MaxPressure probe episodes -- the few-shot band
    201-300 (A17(b)) plus draws 301-400 [...] training-pool draws disjoint from both subjects'
    training draws (1-200) and from the held-out pool (1000-1099), asserted by
    ``assert_probe_draws_disjoint`` before the first episode.  Training recipe: P4's, unchanged
    [...] 40,000 gradient steps, five seeds 101/202/303/404/505, ``target_rtg`` = the maximum
    episode return of the training set and ``rtg_scale`` = the largest absolute RTG of the
    training set (the naive in-domain rule), state normalisation fitted on the anchor's own 200
    episodes.

**It is the curve's k = 200 endpoint.  It is NOT an upper bound on achievable SUMO performance and
NOT a target-domain online policy** (A18(a)); A18(b) registers why there is no online SUMO anchor.
Nothing in this module may be read as one.

What this module does NOT do
----------------------------
It runs no evaluation cell, computes no rho and writes no ``docs/data/p7_3b_anchor.json``.  That is
section 3.4's, in :mod:`offline.transfer_curve`.  The one artifact written here,
``docs/data/p7_3b_anchor_training.json``, is the **committed digest record the evaluation pins
against** (section 3.3): the anchor's checkpoints have no ``SHA256SUMS_p4_*`` and neither
``p4_gate.json`` nor ``p4_7_training.json`` will ever name them, so ``checkpoint_identity`` has
nothing to check against until this file exists.

The three routes to the naive in-domain rule, and why there are three
--------------------------------------------------------------------
``target_rtg`` and ``rtg_scale`` are the prompt.  A wrong one does not crash: it produces a
plausible number in a table.  So:

1. :func:`naive_in_domain_prompt` computes both from the **stacked tensors alone**, using
   ``timestep`` to find the ``t = 0`` windows and ``attention_mask`` to exclude left padding;
2. :func:`load_anchor_corpus` recomputes both from the **dataset's own fitted statistics** --
   P4's route at ``offline/dt_gate.py:1381-1387``, lifted unchanged -- and **raises** if the two
   disagree, rather than preferring one;
3. ``tests/test_anchor_training.py`` recomputes both a third time by ``np.cumsum`` over the raw
   ``.npz`` reward streams, touching neither this module nor :mod:`offline.dataset`, and asserts
   ``==``.

Alignment convention of the corpus this module reads
----------------------------------------------------
Format version **1.1**, C6: outcomes are ``T`` rows and observations are ``T + 1``; the ``info``
returned by step ``t`` describes the state after step ``t``.  ``RTG_t = sum_{u >= t} r_u``, so the
``t = 0`` return-to-go **is** the episode return -- which is what makes A18(a)'s *maximum episode
return* readable off a ``t = 0`` window.  The state must be **25 wide**: that is A16's canonical
frame, and a 32 here means the corpus was logged without ``align_info`` and is not comparable with
anything the paper reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from offline.dataset import NormalizationStats

__all__ = [
    "ANCHOR_DRAW_IDS",
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_NAME",
    "AnchorCorpus",
    "AnchorFacts",
    "CHECKPOINT_STEM",
    "CORPUS_DIGEST_NAME",
    "DECLARED_GRADIENT_STEPS",
    "EXPECTED_EPISODES",
    "EXPECTED_N_ACTIONS",
    "EXPECTED_STATE_DIM",
    "RAISE_TO",
    "SCENARIO_ID",
    "TRAINING_SEEDS",
    "build_artifact",
    "build_parser",
    "checkpoint_path_for_seed",
    "corpus_digest_file_sha256",
    "load_anchor_corpus",
    "main",
    "naive_in_domain_prompt",
    "train_anchor",
    "verify_corpus_digests",
]

#: A18(a)'s training pool: the A17(b) probe band plus the band P7.3b collected.
ANCHOR_DRAW_IDS: tuple[int, ...] = tuple(range(201, 401))

#: Asserted, never inferred.  200 episodes is the registered corpus size; a 199 would train a
#: different model under the same name.
EXPECTED_EPISODES = 200

#: A16's canonical frame.  ``sumo_state_width`` is 32 and ``canonical_state_width`` is 25; the
#: corpus manifest records both, and only the second may reach a model.
EXPECTED_STATE_DIM = 25

#: hz1x1 has eight phases.  Taken from the corpus and checked, not assumed from a literal at a
#: call site -- ``transfer_curve.py:1094-1096`` records why that distinction cost a defect.
EXPECTED_N_ACTIONS = 8

SCENARIO_ID = "cityflow1x1"

#: A18(a) fixes 40,000 outright.  P4's gate raised 20,000 -> 40,000 on an all-seeds plateau rule;
#: that raise is **not** part of the registered anchor recipe, so ``raise_to`` is None and is
#: recorded as ``null`` (Amendment A3).
DECLARED_GRADIENT_STEPS = 40_000
RAISE_TO: int | None = None

#: A18(a)'s five, and P4's five: the same seeds, which is what "P4's recipe unchanged" means.
TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: Written by ``BRIEF_38`` section 3.2 inside each corpus directory, over the ``.npz`` files and
#: ``manifest.json`` and excluding itself.
CORPUS_DIGEST_NAME = "SHA256SUMS"

CHECKPOINT_STEM = "anchor_dt_seed"
ARTIFACT_NAME = "p7_3b_anchor_training.json"
ARTIFACT_FORMAT_VERSION = "p7.3b-anchor-training/1.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AnchorFacts:
    """Everything the artifact records about the corpus, read from it rather than declared."""

    n_episodes: int
    n_windows: int
    state_dim: int
    n_actions: int
    scenario_id: str
    ix_id: str
    draw_ids: tuple[int, ...]
    episodes: tuple[Mapping[str, Any], ...]
    dataset_dirs: tuple[str, ...]
    corpus_digests: Mapping[str, str]
    disjointness: Mapping[str, Any]


@dataclass(frozen=True)
class AnchorCorpus:
    """The stacked tensors, the statistics fitted on these episodes only, and the facts."""

    stacked: dict[str, torch.Tensor]
    stats: NormalizationStats
    facts: AnchorFacts


def corpus_digest_file_sha256(directory: str | Path) -> str:
    """The sha256 OF the directory's ``SHA256SUMS`` file -- one string that names 101 files.

    Recorded in the artifact so a reader can bind the whole corpus half with a single value; the
    file it digests lists every ``.npz`` and ``manifest.json`` individually.
    """
    path = Path(directory) / CORPUS_DIGEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: this corpus half carries no {CORPUS_DIGEST_NAME}. BRIEF_38 section 3.2 "
            "writes one in BOTH halves, and a half without it has no integrity evidence at all"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_corpus_digests(directory: str | Path) -> int:
    """Re-verify every entry of the directory's ``SHA256SUMS``; return how many were checked.

    ``BRIEF_27`` B3(a)'s rule, applied to a corpus rather than a checkpoint: **a digest checked
    once is not a digest checked when used.**  Section 3.2 wrote these digests and Amendment A2
    compared them against the coordinator's independent record; this re-derives them from the
    bytes on disk at the moment they are about to become a model, so an episode edited in between
    is a refusal rather than a silently different corpus.
    """
    root = Path(directory)
    path = root / CORPUS_DIGEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: this corpus half carries no {CORPUS_DIGEST_NAME}. BRIEF_38 section 3.2 "
            "writes one in BOTH halves, and a half without it has no integrity evidence at all"
        )
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        recorded, _, name = line.partition("  ")
        member = root / name.strip()
        if not member.is_file():
            raise ValueError(
                f"{member}: listed in {CORPUS_DIGEST_NAME} and not on disk; the corpus is not the "
                "one those digests were taken over"
            )
        actual = hashlib.sha256(member.read_bytes()).hexdigest()
        if actual != recorded.strip():
            raise ValueError(
                f"{name.strip()}: sha256 {actual} is not the {recorded.strip()} that "
                f"{CORPUS_DIGEST_NAME} records. The bytes moved between collection and training, "
                "so this is not the corpus BRIEF_38 section 3.2 gated"
            )
        checked += 1
    # A digest file that lists nothing verifies vacuously, which is the one way this check could
    # pass while proving nothing.
    if checked == 0:
        raise ValueError(f"{path}: lists no entries, so verifying it proves nothing")
    return checked


def _manifest_draws(directory: Path) -> list[tuple[int, dict[str, Any]]]:
    """``(flow_draw, manifest entry)`` per episode, read from the manifest and not from the files.

    Read before any episode is opened: A18(a) requires the disjointness assertion **before the
    first episode**, and that is only literally true if the draw ids are known from the manifest.
    """
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: a dataset directory is one collection run written by "
            "offline.trajectory_logger, and it has no manifest"
        )
    manifest = json.loads(path.read_bytes())
    version = str(manifest.get("format_version"))
    if version != "1.1":
        raise ValueError(
            f"{directory}: format_version {version!r}, not '1.1'. A18(a)'s corpus is C6 v1.1 "
            "episodes logged through A16's align_info; a v1.0 half is not interchangeable"
        )
    return [(int(entry["flow_draw"]), dict(entry)) for entry in manifest.get("episodes", [])]


def load_anchor_corpus(
    dataset_dirs: Sequence[str | Path],
    *,
    output_root: str | Path,
    context_length: int | None = None,
    expected_draw_ids: Sequence[int] = ANCHOR_DRAW_IDS,
    expected_episodes: int = EXPECTED_EPISODES,
    expected_state_dim: int = EXPECTED_STATE_DIM,
    expected_n_actions: int = EXPECTED_N_ACTIONS,
) -> AnchorCorpus:
    """A18(a)'s 200 episodes, from both v1.1 directories in ONE loader call.

    The order below is the design, not an accident of drafting.  **Everything that can refuse
    precedes everything that costs**, and the disjointness assertion precedes the first episode
    being read rather than merely the first gradient step:

    1. exactly two directories, each with a manifest and a digest file;
    2. every digest re-verified from the bytes on disk (:func:`verify_corpus_digests`);
    3. the draw ids read from the MANIFESTS -- their count and their identity as a set;
    4. :func:`offline.transfer_calibration.disjointness_record`, which wraps
       ``assert_probe_draws_disjoint`` and asserts against the **union** of both subjects'
       ``provenance.training_draw_ids``, ``p4_7_declaration``'s 152 ``mix50`` draws and the
       held-out pool, so no single weak source can make it pass alone;
    5. only now, the loader -- ``build_training_dataset`` over BOTH directories in one call, which
       is what makes the statistics fit A18(a)'s *own 200 episodes* and nothing else;
    6. the shape, from the loaded data: 200 episodes, ``state_dim`` 25, ``n_actions`` 8;
    7. the stacked tensors, one group or it raises (C6 forbids padding across intersections);
    8. the naive in-domain rule computed by two independent routes and compared.

    ``expected_draw_ids`` is a parameter with A18(a)'s band as its default -- a test points it
    elsewhere so the disjointness refusal in step 4 is REACHABLE, which it would not be if step 3
    always refused first.  Nothing else may pass anything but the default.
    """
    from offline.dt_gate import CONTEXT_LENGTH, build_training_dataset, stack_dataset
    from offline.transfer_calibration import disjointness_record

    context = CONTEXT_LENGTH if context_length is None else int(context_length)

    # ---- 1. the two directories --------------------------------------------------------------
    dirs = [Path(d) for d in dataset_dirs]
    if len(dirs) != 2:
        raise ValueError(
            f"the anchor's corpus is two directories -- the A17(b) band 201-300 and the band "
            f"BRIEF_38 section 3.2 collected, 301-400 -- and {len(dirs)} were given: "
            f"{[str(d) for d in dirs]}"
        )
    for directory in dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory}: not a directory")

    # ---- 2. the digests, re-derived at consumption --------------------------------------------
    digests = {str(directory): corpus_digest_file_sha256(directory) for directory in dirs}
    n_verified = sum(verify_corpus_digests(directory) for directory in dirs)

    # ---- 3. the draw ids, from the manifests, before an episode is opened ---------------------
    listed: list[tuple[int, dict[str, Any]]] = []
    for directory in dirs:
        listed.extend(_manifest_draws(directory))
    draw_ids = tuple(sorted(int(draw) for draw, _ in listed))
    wanted = tuple(sorted(int(d) for d in expected_draw_ids))

    if len(listed) != int(expected_episodes):
        raise ValueError(
            f"the anchor's corpus is {expected_episodes} episodes and these directories list "
            f"{len(listed)}. A18(a) registers 200 MaxPressure probe episodes; a different count "
            "trains a different model under the same name"
        )
    if len(set(draw_ids)) != len(draw_ids):
        duplicated = sorted({d for d in draw_ids if draw_ids.count(d) > 1})
        raise ValueError(
            f"the corpus repeats {len(duplicated)} draw id(s) (first {duplicated[:5]}); one "
            "episode per draw is A18(a)'s rule and a repeat weights that demand twice"
        )
    if draw_ids != wanted:
        missing = sorted(set(wanted) - set(draw_ids))
        extra = sorted(set(draw_ids) - set(wanted))
        raise ValueError(
            f"the corpus covers a different draw band: {len(missing)} expected draw(s) absent "
            f"(first {missing[:5]}) and {len(extra)} unexpected (first {extra[:5]}). A18(a)'s "
            f"training pool is {wanted[0]}-{wanted[-1]}"
        )

    # ---- 4. disjointness, BEFORE the first episode is read ------------------------------------
    disjointness = disjointness_record(list(draw_ids), output_root=output_root)

    # ---- 5. the loader, both directories in ONE call ------------------------------------------
    dataset = build_training_dataset(dirs, context)

    # ---- 6. the shape, from the loaded data ---------------------------------------------------
    records = dataset.episode_records
    if len(records) != int(expected_episodes):
        raise ValueError(
            f"the loader returned {len(records)} episodes, not {expected_episodes}; the manifests "
            "and the files on disk disagree about how many episodes this corpus has"
        )
    groups = dataset.groups
    if len(groups) != 1:
        raise ValueError(
            f"this corpus has {len(groups)} (state_dim, n_actions) groups {sorted(groups)}; the "
            "anchor trains one intersection and C6 forbids padding across intersections"
        )
    (state_dim, n_actions) = next(iter(groups))
    if int(state_dim) != int(expected_state_dim):
        raise ValueError(
            f"state_dim is {state_dim}, not {expected_state_dim}. A16 makes align_info the only "
            "door into the canonical frame: 25 is canonical_state_width and 32 is SUMO's own "
            "sumo_state_width, so a 32 here means this corpus was logged WITHOUT the door and a "
            "model trained on it is not comparable with anything the paper reports"
        )
    if int(n_actions) != int(expected_n_actions):
        raise ValueError(
            f"n_actions is {n_actions}, not {expected_n_actions}; hz1x1 has eight phases and a "
            "different action bound is a different action space"
        )

    scenario_id = records[0].scenario_id
    ix_id = records[0].ix_ids[0]

    # ---- 7. the stacked tensors ---------------------------------------------------------------
    stacked = stack_dataset(dataset)

    facts = AnchorFacts(
        n_episodes=len(records),
        n_windows=int(stacked["state"].shape[0]),
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        scenario_id=str(scenario_id),
        ix_id=str(ix_id),
        draw_ids=draw_ids,
        episodes=tuple(
            {
                "dataset_dir": record.dataset_dir,
                "episode_file": record.episode_file,
                "flow_draw": int(record.flow_draw),
                "episode_length": int(record.episode_length),
            }
            for record in sorted(records, key=lambda r: int(r.flow_draw))
        ),
        dataset_dirs=tuple(str(d) for d in dirs),
        corpus_digests=digests,
        disjointness={**disjointness, "n_files_verified": n_verified},
    )
    corpus = AnchorCorpus(stacked=stacked, stats=dataset.stats, facts=facts)

    # ---- 8. the prompt, by two routes, compared -----------------------------------------------
    _assert_prompt_routes_agree(corpus, dataset)
    return corpus


def _assert_prompt_routes_agree(corpus: AnchorCorpus, dataset: Any) -> None:
    """The module's own double-compute of A18(a)'s two quantities.

    Route A is :func:`naive_in_domain_prompt`, over the stacked tensors alone.  Route B is P4's,
    at ``offline/dt_gate.py:1381-1387``, lifted here unchanged -- ``stats.rtg``'s summary for the
    scale, and the ``t = 0`` windows found through ``item_meta`` for the target.  They are two
    routes to one pair of numbers, and a disagreement is a refusal rather than a preference: the
    prompt is not the sort of quantity where one may pick the answer one likes.
    """
    target_a, scale_a = naive_in_domain_prompt(corpus.stacked)

    summary = dataset.stats.rtg[corpus.facts.scenario_id][corpus.facts.ix_id]
    scale_b = max(abs(summary.min), abs(summary.max))
    target_b = max(
        float(corpus.stacked["rtg"][index][-1, 0])
        for index in range(len(dataset))
        if dataset.item_meta(index).t == 0
    )
    if target_a != target_b or scale_a != scale_b:
        raise ValueError(
            "the naive in-domain rule disagrees between its two routes: the stacked tensors give "
            f"(target {target_a!r}, scale {scale_a!r}) and P4's statistics route gives "
            f"(target {target_b!r}, scale {scale_b!r}). A18(a)'s prompt is one pair of numbers "
            "and a wrong one does not crash -- it publishes"
        )


def naive_in_domain_prompt(stacked: Mapping[str, torch.Tensor]) -> tuple[float, float]:
    """A18(a)'s naive in-domain rule: ``(target_rtg, rtg_scale)`` from the stacked tensors alone.

    *"target_rtg = the maximum episode return of the training set and rtg_scale = the largest
    absolute RTG of the training set."*

    Two conventions of :mod:`offline.dataset` make this readable, and both are load-bearing:

    * **padding sits on the LEFT** (``dataset.py:789``), so a window at ``t = 0`` has one real row
      and it is the LAST one.  ``timestep[:, -1] == 0`` therefore selects exactly the ``t = 0``
      windows, and ``rtg[:, -1, 0]`` is their return-to-go at decision 0 -- which, under
      ``RTG_t = sum_{u >= t} r_u``, **is the episode return**;
    * the padded positions carry ``0.0`` in ``rtg``, which is NOT an RTG value.  They are excluded
      through ``attention_mask``.

    ⚠️ **Honest note on that second point, so it is not mistaken for a guard.**  Removing the mask
    here **cannot change this function's result, on any corpus**: padding contributes ``0.0``, and
    ``|0.0|`` never raises a maximum of absolute values.  The exclusion is written because the
    STATISTIC is the part most likely to be edited -- a future change to a mean, a minimum or a
    quantile over the same tensor would silently absorb tens of thousands of padded zeros -- and
    because ``rtg[:, :, 0][mask]`` states what the quantity is taken over.  The corresponding
    mutation is therefore EQUIVALENT, and it is reported as equivalent rather than dressed up with
    a test that could not kill it.
    """
    rtg = stacked["rtg"]
    timestep = stacked["timestep"]
    mask = stacked["attention_mask"]

    at_episode_start = timestep[:, -1] == 0
    if not bool(at_episode_start.any()):
        raise ValueError(
            "no window starts at t = 0, so the training set's episode returns are not present in "
            "these tensors and the maximum episode return cannot be read off them"
        )
    target_rtg = float(rtg[at_episode_start][:, -1, 0].max())

    real = rtg[:, :, 0][mask]
    if real.numel() == 0:
        raise ValueError("every stacked position is padding; there is no RTG to scale by")
    rtg_scale = float(real.abs().max())
    if rtg_scale == 0.0:
        raise ValueError(
            "the largest absolute RTG is 0.0, so dividing by it in train_dt would be a division "
            "by zero; a corpus whose every return-to-go is zero has no prompt scale"
        )
    return target_rtg, rtg_scale


def checkpoint_path_for_seed(checkpoint_dir: str | Path, seed: int) -> Path:
    """Where one anchor seed's checkpoint lives.  Pure path arithmetic."""
    return Path(checkpoint_dir) / f"{CHECKPOINT_STEM}{int(seed)}.pt"


def train_anchor(
    corpus: AnchorCorpus,
    *,
    checkpoint_dir: str | Path,
    device: torch.device,
    target_rtg: float,
    rtg_scale: float,
    seeds: Sequence[int] = TRAINING_SEEDS,
    declared_gradient_steps: int = DECLARED_GRADIENT_STEPS,
    log_every: int = 0,
) -> list[Any]:
    """Train one DT per seed to A18(a)'s recipe, verbatim, and return the ``TrainResult``s.

    **Everything about the recipe is P4's and is read from P4's own constants**, never restated
    here: ``train_dt`` holds ``LEARNING_RATE``, ``WEIGHT_DECAY``, ``WARMUP_STEPS`` and
    ``GRAD_CLIP``, and ``DTConfig`` holds the three layers, the single head, ``d_model`` 128 and
    dropout 0.1.  This function passes the corpus, the seeds, the step count and the prompt, and
    nothing else -- which is what *"P4's recipe, unchanged"* has to mean if it is to be checkable.

    ``rtg_mode`` is deliberately NOT passed: ``train_dt``'s default is the conditioned mode, and
    ``tests/test_train_dt_rtg_mode.py`` is the guard that makes changing that default a failing
    test rather than a silent campaign.

    ``raise_to`` is :data:`RAISE_TO`, which is ``None`` (Amendment A3).  P4's gate raised
    20,000 -> 40,000 on an all-seeds plateau rule; A18(a) fixes 40,000 outright, so there is no
    raise to orchestrate and the artifact records ``null``.
    """
    from offline.dt_gate import BATCH_SIZE, CONTEXT_LENGTH, train_dt

    destination = Path(checkpoint_dir)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination}; nothing is created here"
        )
    requested = [int(seed) for seed in seeds]
    if not requested:
        raise ValueError("the anchor trains at least one seed")
    if len(set(requested)) != len(requested):
        raise ValueError(f"the seed list repeats: {requested}")

    provenance = {
        "tier": "anchor_k200",
        "role": "PREREGISTRATION A18(a): the C3 curve's full-retrain anchor, k = 200",
        "dataset_dirs": list(corpus.facts.dataset_dirs),
        "corpus_digest_files": dict(corpus.facts.corpus_digests),
        "training_draw_ids": list(corpus.facts.draw_ids),
        "scenario_id": corpus.facts.scenario_id,
        "prompt_rule": "naive_in_domain",
    }

    results: list[Any] = []
    for seed in requested:
        results.append(
            train_dt(
                corpus.stacked,
                state_dim=corpus.facts.state_dim,
                n_actions=corpus.facts.n_actions,
                seed=seed,
                declared_gradient_steps=int(declared_gradient_steps),
                raise_to=RAISE_TO,
                context_length=CONTEXT_LENGTH,
                batch_size=BATCH_SIZE,
                device=device,
                checkpoint_path=checkpoint_path_for_seed(destination, seed),
                stats=corpus.stats,
                scenario_id=corpus.facts.scenario_id,
                target_rtg=float(target_rtg),
                rtg_scale=float(rtg_scale),
                provenance=provenance,
                log_every=int(log_every),
            )
        )
    return results


#: A18(a)'s own words about what the anchor is, carried IN the artifact because the packet does
#: not travel with it.  Quoted, not paraphrased.
WHAT_THIS_IS_NOT = (
    "It is the curve's k = 200 endpoint and the paper says exactly what it is: what "
    "target-domain probe data alone buys the same architecture -- NOT an upper bound on "
    "achievable SUMO performance, and NOT a target-domain online policy. A18(b) registers why "
    "there is no online SUMO anchor (no MAPPO-on-SUMO path, contract C8's metric-set defect, "
    ">= 1,000 on-policy episodes per seed); its absence is deferred to P11 and is named as a "
    "limitation in the paper's C3 section, not in a footnote."
)

ROLE = (
    "the C3 transfer curve's k = 200 endpoint: a DT trained from scratch on SUMO, on 200 "
    "MaxPressure probe episodes over draws 201-400 (PREREGISTRATION A18(a))"
)


def build_artifact(
    corpus: AnchorCorpus,
    results: Sequence[Any],
    *,
    checkpoint_dir: str | Path,
    target_rtg: float,
    rtg_scale: float,
) -> dict[str, Any]:
    """The committed digest record section 3.4 pins the anchor's checkpoints and prompt against.

    Why this file has to exist, rather than the anchor reusing an existing record: G1 requires a
    checkpoint to be pinned against a COMMITTED artifact at consumption, and
    ``transfer_curve.CHECKPOINT_RECORD`` maps ``mappo1000 -> p4_gate.json`` and
    ``mix50 -> p4_7_training.json``.  Neither will ever name an anchor checkpoint, and every
    ``SHA256SUMS_p4_*`` is gitignored, so without this file there is nothing a reader of the
    repository could check the anchor's weights against.

    Every digest here is recomputed from the file on disk at the moment the artifact is built.
    """
    from offline.dt_gate import BATCH_SIZE, CONTEXT_LENGTH, GRAD_CLIP, LEARNING_RATE
    from offline.dt_gate import WARMUP_STEPS, WEIGHT_DECAY, runtime_provenance

    destination = Path(checkpoint_dir)
    seeds: list[dict[str, Any]] = []
    for result in results:
        path = checkpoint_path_for_seed(destination, int(result.seed))
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}: seed {result.seed} reports a checkpoint that is not on disk; an artifact "
                "that named it would be a record of nothing"
            )
        seeds.append(
            {
                "seed": int(result.seed),
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "final_loss": float(result.losses[-1]),
                "gradient_steps": int(result.gradient_steps),
                "declared_gradient_steps": int(result.declared_gradient_steps),
                "seconds": float(result.seconds),
                "plateaued": bool(result.plateaued),
                "window_means": [float(v) for v in result.window_means],
            }
        )

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION A18(a); BRIEF_38 section 3.3; Amendment A3",
        "role": ROLE,
        "what_this_is_not": WHAT_THIS_IS_NOT,
        "scenario_id": corpus.facts.scenario_id,
        "intersection_id": corpus.facts.ix_id,
        "prompt_rule": "naive_in_domain",
        "prompt_rule_definition": (
            "target_rtg = the maximum episode return of the training set; rtg_scale = the largest "
            "absolute RTG of the training set. Computed from the anchor's OWN 200 episodes and "
            "never from P7.2b's calibration artifact, which registers CityFlow-trained subjects"
        ),
        "target_rtg": float(target_rtg),
        "rtg_scale": float(rtg_scale),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "raise_to": RAISE_TO,
        "raise_to_note": (
            "null by Amendment A3: A18(a) fixes 40,000 steps outright, and P4's all-seeds plateau "
            "raise is not part of the registered anchor recipe"
        ),
        "recipe": {
            "source": "P4's, unchanged -- offline/dt_gate.py and agent/DTAgent.py's DTConfig",
            "context_length": int(CONTEXT_LENGTH),
            "batch_size": int(BATCH_SIZE),
            "learning_rate": float(LEARNING_RATE),
            "weight_decay": float(WEIGHT_DECAY),
            "warmup_steps": int(WARMUP_STEPS),
            "grad_clip": float(GRAD_CLIP),
            "optimiser": "AdamW",
            "n_layer": 3,
            "n_head": 1,
            "d_model": 128,
            "dropout": 0.1,
        },
        "n_episodes": corpus.facts.n_episodes,
        "n_windows": corpus.facts.n_windows,
        "state_dim": corpus.facts.state_dim,
        "n_actions": corpus.facts.n_actions,
        "corpus": {
            "dataset_dirs": list(corpus.facts.dataset_dirs),
            "digest_files": dict(corpus.facts.corpus_digests),
            "digest_file_name": CORPUS_DIGEST_NAME,
            "draw_ids": list(corpus.facts.draw_ids),
        },
        "episode_ids": [dict(entry) for entry in corpus.facts.episodes],
        "disjointness": dict(corpus.facts.disjointness),
        "normalisation_stats": corpus.stats.to_json_obj(),
        "seeds": seeds,
        "runtime": runtime_provenance(),
    }


def write_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomic tmp-then-replace, so the artifact is either whole or absent."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """The CLI the staged training script drives."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.anchor_training",
        description=(
            "Train the C3 curve's full-retrain anchor (PREREGISTRATION A18(a)): 200 MaxPressure "
            "SUMO episodes, P4's recipe unchanged, 40,000 steps, five seeds, the naive in-domain "
            "prompt. Writes five checkpoints and one committed digest record."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        metavar="PATH",
        help="a v1.1 corpus directory; pass it TWICE (draws 201-300 and 301-400)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="root holding P4's checkpoints, which disjointness_record reads training draws from",
    )
    parser.add_argument("--checkpoint-dir", required=True, help="where the five checkpoints go")
    parser.add_argument(
        "--out-path", required=True, help=f"where to write {ARTIFACT_NAME}"
    )
    parser.add_argument("--device", default=None, help="torch device; default resolves it")
    parser.add_argument(
        "--steps",
        type=int,
        default=DECLARED_GRADIENT_STEPS,
        help=f"gradient steps per seed (A18(a): {DECLARED_GRADIENT_STEPS})",
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load, assert and report the prompt; train nothing and write nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: load, prompt, train five seeds, write the artifact.  Returns an exit code."""
    from agent.utils.utils import Utils

    args = build_parser().parse_args(argv)

    corpus = load_anchor_corpus(args.dataset_dir, output_root=args.output_root)
    target_rtg, rtg_scale = naive_in_domain_prompt(corpus.stacked)
    print(
        f"corpus     {corpus.facts.n_episodes} episodes  {corpus.facts.n_windows} windows  "
        f"state_dim {corpus.facts.state_dim}  n_actions {corpus.facts.n_actions}\n"
        f"draws      {corpus.facts.draw_ids[0]}-{corpus.facts.draw_ids[-1]}  "
        f"disjoint {corpus.facts.disjointness['disjoint']}  "
        f"{corpus.facts.disjointness['n_files_verified']} files re-verified\n"
        f"prompt     target_rtg {target_rtg!r}  rtg_scale {rtg_scale!r}  (naive in-domain)",
        flush=True,
    )
    if args.dry_run:
        print("dry run: nothing trained, nothing written", flush=True)
        return 0

    device = torch.device(args.device) if args.device else Utils.resolve_device(None)
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    print(f"device     {device}  steps {int(args.steps)}  seeds {list(TRAINING_SEEDS)}", flush=True)

    results = train_anchor(
        corpus,
        checkpoint_dir=checkpoints,
        device=device,
        target_rtg=target_rtg,
        rtg_scale=rtg_scale,
        declared_gradient_steps=int(args.steps),
        log_every=int(args.log_every),
    )
    for result in results:
        print(
            f"  seed {result.seed}  {result.gradient_steps} steps  "
            f"final loss {result.losses[-1]:.6f}  {result.seconds:.1f} s",
            flush=True,
        )

    artifact = build_artifact(
        corpus,
        results,
        checkpoint_dir=checkpoints,
        target_rtg=target_rtg,
        rtg_scale=rtg_scale,
    )
    written = write_artifact(artifact, args.out_path)
    print(f"artifact   {written}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
