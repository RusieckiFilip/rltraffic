"""P5.3b -- the ``dt_nortg`` campaign: does removing the return prompt cost anything?

Artifact format version: ``p5.3b-nortg/1.0`` -- ``docs/data/p5_3b_nortg.json``.

The question, and why it is not the one P5.3 was created to ask
---------------------------------------------------------------
P5.3a settled that the return prompt is a **strong lever on the policy** (499 of 500 held-out
episodes move between targets ``0`` and ``-13000``) and a **weak one on mean quality** (+0.9026).
So the knob exists.  What is unproven is that it is worth anything, and that is what this campaign
measures: **fifteen ``dt_nortg`` cells trained with ``rtg_mode="zero"``**, paired against the
committed ``dt`` column of P4.6/P4.7 over the same corpus, seeds, budget and held-out draws.

Conventions this module is bound by, stated because a reader must not have to infer them
-----------------------------------------------------------------------------------------
*Alignment* is contract C6's, unchanged; nothing here re-derives it.

*The arm key is* ``dt_nortg@<tier>``, produced by :func:`nortg_arm_key` and **not** by
``method_tier_grid.arm_key``, which validates against ``METHODS``.  ``BRIEF_30`` section 6.5 forbids
adding to ``METHODS`` because ``method_tier_grid.py:1701`` records that ``grid_comparisons`` emits
pairs in that order, so an entry would change the comparison enumeration of two merged artifacts.
``assert_cell_complete``, ``cell_stats`` and ``paired_comparison`` do not validate against
``METHODS`` and are therefore reused unchanged.

*The paired protocol is the repo's, imported and CALLED* -- ``dt_gate._paired``,
``dt_gate.wilcoxon_signed_rank`` and ``offline_baselines.paired_comparison``.
``docs/reviews/P5.2.md`` **MJ-4** found a packet whose docstring claimed exactly this reuse while
*"none of which was imported or called"*; three tests here fail if any of the three calls is
removed.  **Both arms are sorted by ``(seed, draw_id)`` before pairing**, so the float reduction
order inside ``_per_draw_means`` is fixed and an independent recomputation can assert exact
equality rather than a tolerance.

*The probe is P5.3a's*.  ``BRIEF_30`` section 4.4 asks for ``offline/rtg_ablation.py probe``; that
CLI resolves checkpoints through ``_CHECKPOINT_LAYOUT``, keyed by tier, and cannot address a
``dt_nortg`` file.  :func:`probe_nortg_cell` calls ``rtg_ablation.probe_cell`` directly with an
explicit path -- the same instrument one layer down, with ``offline/rtg_ablation.py`` unmodified
(``docs/plans/p5.3b.md`` section 8 F2, confirmed by ``BRIEF_30`` AMENDMENT A4).

⛔ *No equivalence threshold and no equivalence verdict.*  ``PREREGISTRATION`` A7 withdrew the
per-tier delta rule on 2026-08-25 because it spanned eleven orders of magnitude and could not return
one of its answers on part of its domain.  This module reports the paired difference, its 95 % CI,
the per-seed reversals and the tier's own ``dt`` ATT beside them.  **A CI containing 0 is a failure
to reject, never a demonstration of equivalence** -- and that disclaimer is emitted, not merely
implied.  :func:`assert_no_verdicts` re-checks the payload.

*Division of validation labour.*  :func:`report_artifact` validates the **design** -- the tier set,
the fifteen cells, arm validity, verdict-freedom -- and treats ``episodes`` as an opaque payload.
The **data** is validated by ``main``, which calls ``assert_cell_complete`` for every cell before a
report is assembled, and by ``tests/test_p5_3b_artifact.py``, which pins the shipped bytes.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from offline.dataset import NormalizationStats
from offline.dt_gate import (
    BATCH_SIZE,
    HELD_OUT_DRAWS,
    TRAINING_SEEDS,
    EpisodeResult,
    _paired,
    evaluate_arm,
    runtime_provenance,
    stack_dataset,
    train_dt,
    wilcoxon_signed_rank,
    write_json_atomic,
)
from offline.method_tier_grid import (
    CONTEXT_LENGTH,
    DECLARED_GRADIENT_STEPS,
    TIERS,
    TierSpec,
    _component_streams,
    _dt_factory,
    assert_cell_complete,
    assert_declaration_matches_corpus,
    assert_no_verdicts,
    assert_reused_cells_reproduce,
    assert_reused_checkpoint_identity,
    canonical_digest_of,
    cell_stats,
    env_settings_for_tiers,
    file_sha256,
    measurement_commits,
    tier_dataset,
    tier_dirs,
    tier_spec,
    training_streams,
)
from offline.offline_baselines import (
    filter_stacked_to_streams,
    paired_comparison,
    pin_torch_threads,
    thread_regime,
)
from offline.rtg_ablation import INTERVENTION_KEYS, probe_cell

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "COMPARED_PAYLOAD_KEYS",
    "CONTROL_CELL",
    "CONTROL_COMMITTED_DIGEST",
    "EXCLUDED_PAYLOAD_KEYS",
    "FENCED_OUTPUT_DIRS",
    "GATE_1B_CELLS",
    "NORTG_METHOD",
    "NORTG_RTG_MODE",
    "NORTG_TIERS",
    "TIER_GRID_ARTIFACT",
    "TrainingInputs",
    "assert_arm_validity",
    "assert_committed_dt_agrees_across_grids",
    "assert_payload_matches_committed",
    "assert_reused_dt_identity",
    "assert_writable",
    "committed_dt_episodes",
    "evaluate_cell",
    "main",
    "nortg_arm_key",
    "paired_stats",
    "per_seed_differences",
    "probe_nortg_cell",
    "report_artifact",
    "row_b_pooled_scaled",
    "score_q1",
    "score_q2",
    "score_q3",
    "select_tiers",
    "train_cell",
    "training_inputs",
]

ARTIFACT_FORMAT_VERSION = "p5.3b-nortg/1.0"

#: This task's arm.  It is NOT added to ``method_tier_grid.METHODS`` -- see the module docstring.
NORTG_METHOD = "dt_nortg"
REFERENCE_METHOD = "dt"

#: The ablation, one mode only.  ``rtg_shuffled`` remains unregistered (``BRIEF_30`` section 6.3).
NORTG_RTG_MODE = "zero"

#: The headline tier of ``BRIEF_28`` section 9's rule; clause (i).
DECLARED_TIER = "mappo1000"

#: The rule's output, evaluated on P5.3a's row B before any P5.3b number existed.  Re-derived from
#: ``docs/data/p5_3a_rtg_probe.json`` by :func:`select_tiers` and asserted against this constant by
#: ``tests/test_nortg_campaign.py`` and by ``main`` at run time, so it is a checked answer rather
#: than a remembered one.
NORTG_TIERS: tuple[str, ...] = ("mappo1000", "mix50", "random")

#: Gate 1b (``BRIEF_30`` AMENDMENT A1): one committed ``dt`` cell re-rolled per tier, because the
#: three ``dt`` columns have three different provenances -- ``output/p4_dt/`` (P4's reused column,
#: in **no** integrity manifest, ``DEFERRED`` 56), ``output/p4_7/``, ``output/p4_6/``.
GATE_1B_CELLS: tuple[tuple[str, int], ...] = tuple((tier, 101) for tier in NORTG_TIERS)

#: Gate 2's control cell (``BRIEF_30`` section 4.1).  **Not ``random``**: its conditioned DT is
#: already RTG-inert at the argmax, so a control there would pass whether or not ``rtg_mode``
#: reached the trainer.  ``mappo500`` over ``maxpressure`` because P5.3a measured its ``zero`` flip
#: rate at 0.002361-0.005417 against 0.000139-0.000278; and it is not a campaign tier.
CONTROL_CELL: tuple[str, int] = ("mappo500", 101)
CONTROL_COMMITTED_DIGEST = "5d98d5351198c45054cce1e38b810dabd789708e71e3563e9428d37a49e0e563"

#: AMENDMENT A5.  ``canonical_digest_of`` hashes ``payload["model"]`` alone, so ``target_rtg`` and
#: ``rtg_scale`` -- **which are the prompt** -- are invisible to it.  Gate 2 compares every other
#: key.  ``provenance`` is excluded because it legitimately differs (seed, timings, device, commit).
COMPARED_PAYLOAD_KEYS: tuple[str, ...] = (
    "config",
    "format_version",
    "intersection_ids",
    "normalise",
    "rtg_scale",
    "scenario_id",
    "stats",
    "target_rtg",
)
EXCLUDED_PAYLOAD_KEYS: tuple[str, ...] = ("model", "provenance")

#: 🚨 **A5 versus the repo, and the repo wins (CLAUDE.md section 2).**  AMENDMENT A5 asks for the
#: whole payload except ``model`` and ``provenance`` to be *equal*.  It cannot be, and the reason is
#: **P5.3a's merged change, not this task's**: ``DTConfig.to_json_obj`` emits ``rtg_mode``
#: unconditionally, so a checkpoint written today carries a **9**-key config where the committed
#: P4.6 one carries **8** (measured: the committed ``mappo500`` seed 101 config is exactly
#: ``context_length, d_model, dropout, max_ep_len, n_actions, n_head, n_layer, state_dim``).
#: ``BRIEF_30`` section 6.8 and ``docs/plans/p5.3b.md`` section 8 F4 both predicted it.
#:
#: The allowance below is **narrower than skipping ``config``, not wider than comparing it**: every
#: shared key must be equal, the candidate may not LOSE a key, it may gain only this one, and the
#: value it gains must be the pre-P5.3a behaviour.  A schema fact becomes a checked claim.
CONFIG_KEYS_ADDED_AFTER_P4: Mapping[str, Any] = {"rtg_mode": "conditioned"}

#: Directories under ``output/`` that belong to a merged campaign.  Never written, never deleted
#: (``BRIEF_30`` section 6.6).  ``p5_3b`` is deliberately absent: it is this task's own.
FENCED_OUTPUT_DIRS: tuple[str, ...] = (
    "p4_3", "p4_4", "p4_5", "p4_6", "p4_7", "p4_dt", "p4_probe",
    "p5_1", "p5_2", "p5_3a", "p7_0", "p8_3", "checkpoints",
    "checkpoints.pre_c8_migration",
)

#: Which merged grid holds each tier's committed ``dt`` column.
TIER_GRID_ARTIFACT: Mapping[str, str] = {
    "mappo1000": "p4_6_grid.json",
    "mix50": "p4_7_grid.json",
    "random": "p4_6_grid.json",
}

#: Which merged training record holds each tier's committed ``dt`` canonical digest.  ``mappo1000``
#: has **none**: ``p4_training.json`` never carried one (``method_tier_grid.py:1233-1235``), which
#: is why its identity goes through ``assert_reused_checkpoint_identity``'s file-sha256 route.
TIER_TRAINING_ARTIFACT: Mapping[str, str | None] = {
    "mappo1000": None,
    "mix50": "p4_7_training.json",
    "random": "p4_6_training.json",
}

TIER_CHECKPOINT_TEMPLATE: Mapping[str, str] = {
    "mappo1000": "p4_dt/dt_seed{seed}.pt",
    "mix50": "p4_7/checkpoints/mix50_dt_seed{seed}.pt",
    "random": "p4_6/checkpoints/random_dt_seed{seed}.pt",
}

#: The integrity manifest covering each tier's ``dt`` checkpoints, at consumption
#: (``BRIEF_27`` B3(a)).  ⚠️ ``mappo1000`` has **none** -- ``DEFERRED`` 56.
TIER_MANIFEST: Mapping[str, str | None] = {
    "mappo1000": None,
    "mix50": "SHA256SUMS_p4_7.txt",
    "random": "SHA256SUMS_p4_6.txt",
}

SCENARIO_KEY = "cityflow1x1"
SCENARIO_ID = "cityflow1x1"
ENGINE_SEED = 1000

_LIMITATIONS: tuple[str, ...] = (
    "A null on a tier is not 'the prompt is useless'. It is 'removing the prompt did not change "
    "mean held-out ATT on this corpus, at this budget, at 5 seeds, on this tier'. A 95 % CI "
    "containing 0 is a failure to reject, never a demonstration of equivalence.",
    "The three tiers differ in more than return spread: composition, state coverage and data "
    "quality move together. Row B is an axis we can measure, not one we can isolate, so Q1's "
    "ordering is consistent with hypothesis C4 rather than evidence for it.",
    "random's DT is 4x worse in ATT than the other two (420.3764 against 104.9558 and 107.7026). A "
    "difference measured there is not comparable in magnitude to one measured on mappo1000.",
    "mix50's NormalizationStats is fitted on the UNION of all three mixtures' six directories and "
    "is identical for mix33, mix50 and mix67: count 216000, raw std 13155.3172. That is how P4.7 "
    "trained it, so reusing it is consistent -- but the summary is not a property of mix50.",
    "Q1 is scored on the RAW ATT scale, which is the conservative choice: if paired differences "
    "scaled with baseline ATT, random would show the largest raw difference, which is the opposite "
    "of what Q1 predicts. The scale-normalised column is a reading aid and is not a scored claim.",
)


# ----------------------------------------------------------------------
# Identity, and the output fence
# ----------------------------------------------------------------------


def nortg_arm_key(tier: str) -> str:
    """``"dt_nortg@<tier>"``, refusing a tier this task does not declare."""
    if str(tier) not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; the platform declares {sorted(TIERS)}")
    return f"{NORTG_METHOD}@{tier}"


def assert_writable(path: str | Path) -> Path:
    """Refuse any path inside a merged campaign's output directory, and return *path* unchanged.

    A path is fenced when a component ``output`` is **immediately followed** by a fenced name.
    Neither a string prefix nor a bare component test would do: a prefix test makes ``output/p5_3a``
    and ``output/p5_3b`` indistinguishable, and a bare component test refuses this task's own
    ``output/p5_3b/checkpoints/`` because ``output/checkpoints/`` is itself fenced.  Both traps are
    covered by tests.  P5.2's BL-1 destroyed six irrecoverable training records in an un-backed-up
    tree; ``output/`` is gitignored and has no backup.
    """
    target = Path(path)
    parts = Path(target).resolve().parts
    for index, part in enumerate(parts[:-1]):
        head = parts[index + 1]
        if part == "output" and head in FENCED_OUTPUT_DIRS:
            raise ValueError(
                f"{target}: output/{head} belongs to another campaign and is read-only here; "
                f"P5.3b writes only under output/p5_3b (BRIEF_30 section 6.6)"
            )
    return target


# ----------------------------------------------------------------------
# The tier rule, re-evaluated rather than remembered
# ----------------------------------------------------------------------


def row_b_pooled_scaled(probe_artifact: Mapping[str, Any]) -> dict[str, float]:
    """Row B's pooled between-episode sd of the **scaled** RTG, per tier, from P5.3a's artifact.

    Row B and not ``RtgSummary.std``: the marginal statistic is 65-93 % within-episode ramp on the
    single-policy tiers and is fitted on the wrong population entirely for the mixtures
    (``docs/plans/p5.3a.md`` section 2, and ``docs/returns/P5.3a.md``'s hand-forward).
    """
    spread = probe_artifact.get("tables", {}).get("spread")
    if not isinstance(spread, Mapping) or not spread:
        raise ValueError(
            "the P5.3a probe artifact carries no tables.spread block, so row B cannot be read and "
            "the tier rule cannot be evaluated"
        )
    pooled: dict[str, float] = {}
    for tier, entry in spread.items():
        try:
            pooled[str(tier)] = float(entry["between_episode_rtg_scaled"]["pooled"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{tier}: row B is missing from the spread table ({error})") from error
    return pooled


def select_tiers(probe_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """``BRIEF_28`` section 9's rule, evaluated on row B.  Registered 2026-08-24, before the data.

    *"(i) ``mappo1000``, the headline tier; (ii) the tier with the largest measured between-episode
    scaled-RTG sd; (iii) the tier with the smallest.  If (ii) or (iii) is ``mappo1000``, take the
    next one in that direction."*  Ties break by tier name ascending, declared here because P5.2's
    **MJ-5** found three reported integers resting on an undeclared tie-break.
    """
    pooled = row_b_pooled_scaled(probe_artifact)
    order = sorted(pooled, key=lambda tier: (pooled[tier], tier))
    if len(order) < 3:
        raise ValueError(f"the rule needs at least three tiers, got {order}")

    fallback_fired = False
    widest = order[-1]
    if widest == DECLARED_TIER:
        widest = order[-2]
        fallback_fired = True
    narrowest = order[0]
    if narrowest == DECLARED_TIER:
        narrowest = order[1]
        fallback_fired = True

    values = sorted(pooled.values())
    return {
        "rule": (
            "BRIEF_28 section 9, registered 2026-08-24 before any spread number existed: (i) the "
            "declared headline tier, (ii) the largest between-episode scaled-RTG sd, (iii) the "
            "smallest; if (ii) or (iii) is the headline tier, take the next one in that direction"
        ),
        "axis": "row B: pooled between-episode sd of the scaled RTG, over the declared 200-stream "
                "training set",
        "source": "docs/data/p5_3a_rtg_probe.json tables.spread.<tier>."
                  "between_episode_rtg_scaled.pooled",
        "row_b_pooled_scaled": dict(sorted(pooled.items())),
        "declared": DECLARED_TIER,
        "widest": widest,
        "narrowest": narrowest,
        "fallback_fired": fallback_fired,
        "tie_break": "tier name ascending",
        "ties_present": len(set(values)) != len(values),
        "spread_ratio_widest_over_narrowest": pooled[widest] / pooled[narrowest],
        "tiers": sorted({DECLARED_TIER, widest, narrowest}),
    }


def assert_selection_still_holds(probe_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """The registered tier set must still be what the rule returns.  Checked at run time."""
    selection = select_tiers(probe_artifact)
    if tuple(selection["tiers"]) != NORTG_TIERS:
        raise ValueError(
            f"the tier rule now returns {selection['tiers']} but this task is registered for "
            f"{list(NORTG_TIERS)}; a registration may not follow its inputs silently"
        )
    return selection


# ----------------------------------------------------------------------
# Training: the same inputs method_tier_grid._run_train builds
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingInputs:
    """Everything ``train_dt`` needs for one tier, built exactly as ``_run_train`` builds it."""

    tier: str
    spec: TierSpec
    group: tuple[int, int]
    batch: dict[str, torch.Tensor]
    stats: NormalizationStats
    scenario_id: str
    provenance: dict[str, Any]
    declared: Mapping[str, Any]
    n_streams: int
    n_windows: int


def training_inputs(tier: str, corpus_root: str | Path) -> TrainingInputs:
    """One tier's training batch, by ``method_tier_grid._run_train``'s own recipe.

    ⚠️ **This function is validated by Gate 2, not by a unit test.**  The control cell retrains
    through these inputs and must reproduce ``p4_6_training.json``'s committed canonical digest; a
    batch that differed from ``_run_train``'s in any way would not reproduce it.  That coupling is
    why the two share this one definition instead of each carrying their own.
    """
    spec = tier_spec(tier)
    dataset = tier_dataset(spec, corpus_root)
    stacked = stack_dataset(dataset)
    group = next(iter(dataset.groups))
    scenario_id = dataset.episode_records[0].scenario_id
    components = (
        _component_streams(spec, corpus_root, context_length=CONTEXT_LENGTH)
        if spec.subsample == "mixture"
        else None
    )
    selected = training_streams(spec, dataset, component_streams=components)
    # The prompt, the RTG scale and the reward scale are computed over the TRAINING SET, so the
    # declaration is checked after the selection and never before it (BRIEF_17 section 11, A4).
    declared = assert_declaration_matches_corpus(spec, selected)
    batch = filter_stacked_to_streams(dataset, stacked, selected)
    provenance = {
        "tier": spec.tier,
        "dataset_dirs": [str(d) for d in tier_dirs(spec, corpus_root)],
        "training_draw_ids": list(dataset.stats.draw_ids),
        "scenario_id": scenario_id,
        "statistics_digest": None,
        "subsample": spec.subsample,
        "training_streams": len(selected),
    }
    from offline.method_tier_grid import statistics_digest

    provenance["statistics_digest"] = statistics_digest(dataset)
    return TrainingInputs(
        tier=spec.tier,
        spec=spec,
        group=(int(group[0]), int(group[1])),
        batch=batch,
        stats=dataset.stats,
        scenario_id=scenario_id,
        provenance=provenance,
        declared=declared,
        n_streams=len(selected),
        n_windows=int(batch["state"].shape[0]),
    )


def train_cell(
    tier: str,
    seed: int,
    *,
    corpus_root: str | Path,
    checkpoint_dir: str | Path,
    device: str | None = None,
    steps: int = DECLARED_GRADIENT_STEPS,
    log_every: int = 0,
    inputs: TrainingInputs | None = None,
) -> dict[str, Any]:
    """Train one ``dt_nortg`` cell.  ``rtg_mode="zero"`` is the whole intervention."""
    from agent.utils.utils import Utils

    prepared = inputs if inputs is not None else training_inputs(tier, corpus_root)
    directory = assert_writable(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = assert_writable(directory / f"{tier}_{NORTG_METHOD}_seed{int(seed)}.pt")
    resolved = torch.device(device) if device else Utils.resolve_device(None)

    started = time.time()
    result = train_dt(
        prepared.batch,
        state_dim=prepared.group[0],
        n_actions=prepared.group[1],
        seed=int(seed),
        declared_gradient_steps=int(steps),
        raise_to=None,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        device=resolved,
        checkpoint_path=destination,
        stats=prepared.stats,
        scenario_id=prepared.scenario_id,
        target_rtg=float(prepared.spec.target_rtg),
        rtg_scale=float(prepared.spec.rtg_scale),
        provenance={**prepared.provenance, "rtg_mode": NORTG_RTG_MODE, "campaign": "p5.3b"},
        log_every=int(log_every),
        rtg_mode=NORTG_RTG_MODE,
    )
    seconds = time.time() - started
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    if payload["config"].get("rtg_mode") != NORTG_RTG_MODE:
        raise ValueError(
            f"{destination}: the checkpoint records rtg_mode "
            f"{payload['config'].get('rtg_mode')!r}, not {NORTG_RTG_MODE!r}; rtg_mode did not "
            "reach the training path and the arm would be indistinguishable from dt"
        )
    return {
        "tier": tier,
        "method": NORTG_METHOD,
        "seed": int(seed),
        "rtg_mode": NORTG_RTG_MODE,
        "gradient_steps": int(result.gradient_steps),
        "declared_gradient_steps": int(steps),
        "plateaued": bool(result.plateaued),
        "final_loss": float(result.losses[-1]),
        "seconds": float(seconds),
        "train_dt_seconds": float(result.seconds),
        "checkpoint": str(destination),
        "canonical_digest": canonical_digest_of(destination),
        "file_sha256": file_sha256(destination),
        "target_rtg": float(prepared.spec.target_rtg),
        "rtg_scale": float(prepared.spec.rtg_scale),
        "training_streams": prepared.n_streams,
        "training_windows": prepared.n_windows,
        "thread_regime": thread_regime(),
    }


# ----------------------------------------------------------------------
# Evaluation, through P4.6's own instrument
# ----------------------------------------------------------------------


def evaluate_cell(
    tier: str,
    seed: int,
    *,
    checkpoint: str | Path,
    corpus_root: str | Path,
    draws_root: str | Path,
    engine_seed: int = ENGINE_SEED,
    device: str | None = None,
    arm: str | None = None,
    draws: Sequence[int] | None = None,
) -> list[EpisodeResult]:
    """Roll one cell over the held-out pool through ``evaluate_arm`` and ``_dt_factory``.

    ``_dt_factory`` is P4.6's own DT evaluation path -- load, **then** apply the declared target,
    then act greedily -- reused rather than re-implemented, because ``DTAgent.load`` overwrites
    ``_target_rtg`` from the payload and a target passed to the constructor is silently discarded
    (``offline/rtg_calibration.py``'s ``agent_with_target``).  The declared ``target_rtg`` is the
    tier's own, exactly as the ``dt`` arm used: the ablation is in the weights, not in the prompt
    handed to the harness, and Q3 proves the model ignores it.
    """
    from offline.materialise_draws import draw_config_path

    spec = tier_spec(tier)
    settings = env_settings_for_tiers([spec], corpus_root)
    factory = _dt_factory(
        str(checkpoint), DECLARED_GRADIENT_STEPS, float(spec.target_rtg), device
    )
    return list(
        evaluate_arm(
            arm=arm or nortg_arm_key(tier),
            seed=int(seed),
            draw_ids=list(draws if draws is not None else HELD_OUT_DRAWS),
            config_for_draw=lambda draw: draw_config_path(
                SCENARIO_KEY, int(draw), out_root=draws_root
            ),
            env_settings=settings,
            scenario_id=SCENARIO_ID,
            choose_action_factory=factory,
            engine_seed=int(engine_seed),
        )
    )


# ----------------------------------------------------------------------
# The reused dt column
# ----------------------------------------------------------------------


def nortg_cell_record(episodes: Sequence[EpisodeResult], seed: int) -> dict[str, Any]:
    """``cell_stats`` plus the scalar ``seed`` this campaign's cells are keyed by.

    ``method_tier_grid.cell_stats`` emits ``seeds`` -- a **list**, because a P4.6 cell spans all
    five training seeds.  P5.3b evaluates one seed per job so its cells are per-seed, and the
    artifact is keyed by ``(tier, seed)``.  Adding the scalar here rather than at the call site
    keeps one definition of what a P5.3b cell record is, and gives it a test that does not need a
    simulator.
    """
    record = cell_stats(episodes)
    if record["seeds"] != [int(seed)]:
        raise ValueError(
            f"a P5.3b cell describes exactly one training seed; cell_stats reports "
            f"{record['seeds']} for seed {int(seed)}"
        )
    return {**record, "seed": int(seed)}


def committed_dt_episodes(tier: str, *, data_dir: str | Path) -> list[EpisodeResult]:
    """The committed ``dt@<tier>`` per-episode records, read from the merged grid artifact."""
    grid = json.loads(
        (Path(data_dir) / TIER_GRID_ARTIFACT[str(tier)]).read_text(encoding="utf-8")
    )
    arm = f"{REFERENCE_METHOD}@{tier}"
    records = [entry for entry in grid["episodes"] if entry["arm"] == arm]
    expected = len(TRAINING_SEEDS) * len(HELD_OUT_DRAWS)
    if len(records) != expected:
        raise ValueError(
            f"{arm}: the committed grid holds {len(records)} episodes, not {expected}; the paired "
            "comparison would rest on an incomplete column"
        )
    return [
        EpisodeResult(
            arm=arm,
            seed=int(record["seed"]),
            draw_id=int(record["draw_id"]),
            att_horizon=float(record["att_horizon"]),
            horizon_vehicle_count=float(record["horizon_vehicle_count"]),
            episode_reward=float(record["episode_reward"]),
        )
        for record in records
    ]


def assert_committed_dt_agrees_across_grids(tier: str, *, data_dir: str | Path) -> dict[str, Any]:
    """For a tier present in both merged grids, the two copies must be bit-identical.

    ``mappo1000`` and ``random`` appear in ``p4_6_grid.json`` **and** ``p4_7_grid.json`` (P4.7
    re-reports phase 1).  Asserting they agree makes the choice of file provably immaterial instead
    of merely conventional.
    """
    root = Path(data_dir)
    arm = f"{REFERENCE_METHOD}@{tier}"
    copies: dict[str, list[tuple[int, int, float, float, float]]] = {}
    for name in ("p4_6_grid.json", "p4_7_grid.json"):
        grid = json.loads((root / name).read_text(encoding="utf-8"))
        rows = sorted(
            (
                int(e["seed"]),
                int(e["draw_id"]),
                float(e["att_horizon"]),
                float(e["horizon_vehicle_count"]),
                float(e["episode_reward"]),
            )
            for e in grid["episodes"]
            if e["arm"] == arm
        )
        if rows:
            copies[name] = rows
    if len(copies) < 2:
        return {"tier": tier, "grids": sorted(copies), "compared": False,
                "reason": "the tier appears in only one merged grid"}
    (first, left), (second, right) = sorted(copies.items())
    if left != right:
        raise ValueError(
            f"{arm}: {first} and {second} disagree on the committed column, so which file is read "
            "would change the reported comparison"
        )
    return {"tier": tier, "grids": [first, second], "compared": True, "n_episodes": len(left)}


def _manifest_digests(path: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        digests[name.strip()] = digest.strip()
    return digests


def assert_reused_dt_identity(
    *, data_dir: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Gate 1: the reused ``dt`` checkpoints are the committed weights, checked AT CONSUMPTION.

    ``BRIEF_27`` B3(a): *a digest checked once is not a digest checked when used.*  Two routes,
    because they protect different things -- a canonical ``state_dict`` digest against the merged
    training record, and a file sha256 against the campaign's integrity manifest.

    🚨 **The three tiers are NOT equally protected.**  ``output/p4_dt/`` appears in **no** integrity
    manifest (``DEFERRED`` 56), and ``p4_training.json`` never carried a canonical digest for those
    five checkpoints (``method_tier_grid.py:1233-1235``), so ``mappo1000`` rests on a
    filename-dependent file sha256 (``DEFERRED`` 29) with nothing behind it.  Gate 1b's behavioural
    re-roll is the compensating control for that recorded gap, and the record below says so rather
    than letting the three tiers read as equal.
    """
    root = Path(output_root)
    data = Path(data_dir)
    tiers: dict[str, Any] = {}

    for tier in NORTG_TIERS:
        template = TIER_CHECKPOINT_TEMPLATE[tier]
        manifest_name = TIER_MANIFEST[tier]
        manifest = _manifest_digests(root / manifest_name) if manifest_name else {}
        training_name = TIER_TRAINING_ARTIFACT[tier]
        committed: dict[int, str] = {}
        if training_name:
            training = json.loads((data / training_name).read_text(encoding="utf-8"))
            committed = {
                int(run["seed"]): str(run["canonical_digest"])
                for run in training["runs"]
                if run["tier"] == tier and run["method"] == REFERENCE_METHOD
            }
            if sorted(committed) != sorted(TRAINING_SEEDS):
                raise ValueError(
                    f"{tier}: {training_name} records dt digests for seeds {sorted(committed)}, "
                    f"not {list(TRAINING_SEEDS)}"
                )

        seeds: list[dict[str, Any]] = []
        for seed in TRAINING_SEEDS:
            relative = template.format(seed=seed)
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"reused dt checkpoint is missing: {path}")
            digest = canonical_digest_of(path)
            if committed and digest != committed[int(seed)]:
                raise ValueError(
                    f"{path}: canonical digest {digest} is not the committed {committed[int(seed)]}"
                    "; the reused column would describe different weights"
                )
            sha = file_sha256(path)
            if manifest_name:
                if relative not in manifest:
                    raise ValueError(
                        f"{relative} is not listed in {manifest_name}; a checkpoint outside its "
                        "own campaign manifest has no integrity record at consumption"
                    )
                if manifest[relative] != sha:
                    raise ValueError(
                        f"{path}: file sha256 {sha} is not {manifest_name}'s {manifest[relative]}"
                    )
            seeds.append(
                {
                    "seed": int(seed),
                    "path": str(path),
                    "canonical_digest": digest,
                    "file_sha256": sha,
                    "canonical_digest_checked_against": training_name,
                    "manifest_checked_against": manifest_name,
                }
            )

        tiers[tier] = {
            "checkpoints": seeds,
            "canonical_digest_route": bool(training_name),
            "manifest_route": bool(manifest_name),
            "grid_artifact": TIER_GRID_ARTIFACT[tier],
            "cross_grid_agreement": assert_committed_dt_agrees_across_grids(tier, data_dir=data),
        }

    p4_dt = tiers[DECLARED_TIER]
    p4_dt["deferred_56"] = (
        "output/p4_dt/ appears in no integrity manifest and p4_training.json never carried a "
        "canonical digest for these five checkpoints, so this tier's identity rests on a "
        "filename-dependent file sha256 alone. Gate 1b's behavioural re-roll is the compensating "
        "control. The three tiers are not equally protected."
    )
    p4_dt["file_sha256_route"] = assert_reused_checkpoint_identity(
        json.loads((data / "p4_4_training.json").read_text(encoding="utf-8")),
        json.loads((data / "p4_gate.json").read_text(encoding="utf-8")),
        baselines_root=root / "p4_4" / "checkpoints",
        dt_root=root / "p4_dt",
    )
    return {
        "role": "Gate 1: reused dt checkpoint identity, verified at consumption (BRIEF_27 B3(a))",
        "tiers": tiers,
        "manifest_coverage": {tier: TIER_MANIFEST[tier] for tier in NORTG_TIERS},
    }


def assert_payload_matches_committed(
    candidate: str | Path, committed: str | Path
) -> dict[str, Any]:
    """AMENDMENT A5: every payload key except ``model`` and ``provenance`` must be equal.

    ``canonical_digest_of`` hashes ``payload["model"]`` alone
    (``offline/method_tier_grid.py:1180-1185``), so ``target_rtg`` and ``rtg_scale`` -- **which are
    the prompt** -- are invisible to it.  A thread-through that perturbed either would leave the
    digest green and change every number in the campaign.  ``model`` is excluded because the digest
    covers it; ``provenance`` is excluded because it legitimately records the seed, the timings, the
    device and the write-time commit.  The exclusion is named, never silent.
    """
    left = torch.load(Path(candidate), map_location="cpu", weights_only=False)
    right = torch.load(Path(committed), map_location="cpu", weights_only=False)
    if set(left) != set(right):
        raise ValueError(
            f"the two payloads do not carry the same keys: {sorted(set(left) ^ set(right))}; a "
            "payload that gained or lost a key must not slip through uncompared"
        )
    compared = tuple(sorted(set(left) - set(EXCLUDED_PAYLOAD_KEYS)))
    if compared != COMPARED_PAYLOAD_KEYS:
        raise ValueError(
            f"the compared key set is {list(compared)} but this task registered "
            f"{list(COMPARED_PAYLOAD_KEYS)}; the comparison must cover every key the digest cannot"
        )
    differing: list[str] = []
    details: dict[str, Any] = {}
    for key in compared:
        if key == "config":
            reasons = _config_differences(left[key], right[key])
            if reasons:
                differing.append(key)
                details[key] = reasons
        elif left[key] != right[key]:
            differing.append(key)
            details[key] = [f"{left[key]!r} against {right[key]!r}"]
    if differing:
        raise ValueError(
            f"payload keys differ outside model and provenance: {differing}; the canonical digest "
            f"cannot see any of these, and target_rtg and rtg_scale ARE the prompt. {details}"
        )
    return {
        "candidate": str(candidate),
        "committed": str(committed),
        "compared_keys": list(compared),
        "excluded_keys": list(EXCLUDED_PAYLOAD_KEYS),
        "excluded_because": {
            "model": "covered by the canonical state_dict digest",
            "provenance": "legitimately differs: seed, timings, device, write-time git commit",
        },
        "config_key_allowance": {
            "may_be_gained": dict(CONFIG_KEYS_ADDED_AFTER_P4),
            "why": "DTConfig.to_json_obj has emitted rtg_mode unconditionally since P5.3a, so a "
                   "checkpoint written today carries a 9-key config where a P4.6-era one carries "
                   "8. Every shared key must still be equal, no key may be lost, and the gained "
                   "value must be the pre-P5.3a behaviour.",
        },
        "differing_keys": differing,
    }


def _config_differences(candidate: Mapping[str, Any], committed: Mapping[str, Any]) -> list[str]:
    """Every way the retrained config may differ from the committed one, as reasons.

    An empty list is the only acceptable answer; see :data:`CONFIG_KEYS_ADDED_AFTER_P4` for the one
    declared allowance and why it is narrower than skipping the key.
    """
    reasons: list[str] = []
    lost = sorted(set(committed) - set(candidate))
    if lost:
        reasons.append(f"the retrained config LOST {lost}")
    gained = sorted(set(candidate) - set(committed))
    undeclared = [key for key in gained if key not in CONFIG_KEYS_ADDED_AFTER_P4]
    if undeclared:
        reasons.append(f"the retrained config gained undeclared key(s) {undeclared}")
    for key in gained:
        if key in CONFIG_KEYS_ADDED_AFTER_P4:
            expected = CONFIG_KEYS_ADDED_AFTER_P4[key]
            if candidate[key] != expected:
                reasons.append(
                    f"{key} is {candidate[key]!r}, not the pre-P5.3a behaviour {expected!r}"
                )
    reasons.extend(
        f"{key}: {candidate[key]!r} against committed {committed[key]!r}"
        for key in sorted(set(candidate) & set(committed))
        if candidate[key] != committed[key]
    )
    return reasons


# ----------------------------------------------------------------------
# The paired statistics -- imported and CALLED (docs/reviews/P5.2.md MJ-4)
# ----------------------------------------------------------------------


def _sorted_for_pairing(episodes: Sequence[EpisodeResult]) -> list[EpisodeResult]:
    """``(seed, draw_id)`` order, so ``_per_draw_means``' float reduction order is fixed."""
    return sorted(episodes, key=lambda e: (int(e.seed if e.seed is not None else -1), int(e.draw_id)))


def paired_stats(
    dt_episodes: Sequence[EpisodeResult], nortg_episodes: Sequence[EpisodeResult]
) -> dict[str, Any]:
    """Paired per-draw comparison of ``dt`` against ``dt_nortg`` over their shared draws.

    Sign convention, registered in ``docs/plans/p5.3b.md`` section 3.1:
    ``mean_difference = mean(ATT_dt - ATT_dt_nortg)``, **left = dt**.  Lower ATT is better, so a
    negative difference means the prompted arm is better.

    Three repo functions are called, not merely described: ``dt_gate._paired`` supplies the shared
    draws (``PREREGISTRATION`` A5 point 3 makes a comparison without them **void**),
    ``offline_baselines.paired_comparison`` supplies the headline, and
    ``dt_gate.wilcoxon_signed_rank`` is run a **second time on the same vectors** and compared to
    the one inside the ``PairedComparison``.  A disagreement is a refusal, not a warning.
    """
    left_episodes = _sorted_for_pairing(dt_episodes)
    right_episodes = _sorted_for_pairing(nortg_episodes)
    left_values, right_values, shared = _paired(left_episodes, right_episodes)
    comparison = paired_comparison(left_episodes, right_episodes)
    reference = wilcoxon_signed_rank(left_values, right_values)

    inner = comparison.wilcoxon
    fields = ("w_plus", "w_minus", "statistic", "n_used", "n_zero", "z", "p_value")
    disagreement = [
        field
        for field in fields
        if getattr(reference, field) != getattr(inner, field)
    ]
    if disagreement:
        raise ValueError(
            f"the two Wilcoxon routes disagree on {disagreement}; the same test run on the same "
            "paired vectors must give the same answer, so this is a defect in the pairing rather "
            "than a numerical nicety"
        )

    differences = [a - b for a, b in zip(left_values, right_values)]
    return {
        "paired": comparison.to_json_obj(),
        "abs_mean_difference": abs(float(comparison.mean_difference)),
        "mean_absolute_difference": float(np.mean(np.abs(np.asarray(differences, np.float64)))),
        "n_shared_draws": len(shared),
        "sign_convention": "mean(ATT_dt - ATT_dt_nortg); lower ATT is better, so a negative value "
                           "means the prompted arm is better",
        "wilcoxon_second_route_agrees": True,
    }


def per_seed_differences(
    dt_episodes: Sequence[EpisodeResult],
    nortg_episodes: Sequence[EpisodeResult],
    pooled_difference: float,
) -> dict[str, Any]:
    """The seed dimension the per-draw comparison averages away.

    ``d_s`` averages draws inside a seed while the pooled difference averages seeds inside a draw,
    so the two are the same quantity in exact arithmetic and may differ in the last bits.  Both are
    reported with their measured difference and **no equality is asserted between them**: that
    would condemn a correct implementation (``docs/plans/p5.3b.md`` section 3.4).

    A seed whose difference is exactly ``0.0`` counts as a reversal -- the conservative direction.
    """
    def by_seed(episodes: Sequence[EpisodeResult]) -> dict[int, float]:
        buckets: dict[int, list[float]] = {}
        for episode in episodes:
            buckets.setdefault(int(episode.seed), []).append(float(episode.att_horizon))
        return {seed: float(np.mean(values)) for seed, values in buckets.items()}

    left, right = by_seed(dt_episodes), by_seed(nortg_episodes)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("the two arms share no training seed, so the seed dimension is empty")
    differences = {seed: left[seed] - right[seed] for seed in shared}
    pooled = float(pooled_difference)
    mean_of_per_seed = float(np.mean(np.asarray(list(differences.values()), np.float64)))
    return {
        "per_seed": {str(seed): value for seed, value in differences.items()},
        "n_seeds": len(shared),
        "seeds_reversed": int(sum(1 for value in differences.values() if value * pooled <= 0.0)),
        "reversal_rule": "sign(d_s) differs from sign(the pooled difference), counting an exact "
                         "zero as a reversal",
        "pooled_difference": pooled,
        "mean_of_per_seed_differences": mean_of_per_seed,
        "difference_between_the_two_averaging_orders": mean_of_per_seed - pooled,
        "note": "d_s averages draws inside a seed; the pooled difference averages seeds inside a "
                "draw. Equal in exact arithmetic, and no equality is asserted between them.",
    }


# ----------------------------------------------------------------------
# Arm validity -- P5.3a's probe, pointed at the new checkpoints
# ----------------------------------------------------------------------


def probe_nortg_cell(
    tier: str,
    seed: int,
    *,
    checkpoint_path: str | Path,
    corpus_root: str | Path,
    device: str | None = None,
    streams: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """P5.3a's teacher-forced probe on one ``dt_nortg`` checkpoint, plus its recorded ``rtg_mode``.

    ``rtg_ablation.probe_cell`` is called with an explicit ``checkpoint_path`` because its CLI
    resolves paths through ``_CHECKPOINT_LAYOUT``, keyed by tier, and knows nothing of a
    ``dt_nortg`` file.  Same instrument, one layer down; ``offline/rtg_ablation.py`` is unmodified.
    """
    cell = probe_cell(
        tier,
        int(seed),
        checkpoint_path=checkpoint_path,
        corpus_root=corpus_root,
        device=device,
        streams=streams,
    )
    payload = cell.to_json_obj()
    config = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)["config"]
    payload["rtg_mode"] = str(config.get("rtg_mode", "conditioned"))
    return payload


def assert_arm_validity(probe_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Q3, the GATE: every ``dt_nortg`` checkpoint ignores a token it was never shown.

    ⚠️ Q3 and Q2 are different claims and must not be conflated.  Q3 says the *trained* model
    ignores the token; Q2 says training without it cost nothing on a tier.  Anything non-zero here
    means ``rtg_mode`` did not reach the training path -- a wiring finding, not a scientific one.
    """
    cells = list(probe_cells)
    expected = {(tier, int(seed)) for tier in NORTG_TIERS for seed in TRAINING_SEEDS}
    got = {(str(cell["tier"]), int(cell["seed"])) for cell in cells}
    if got != expected or len(cells) != len(expected):
        raise ValueError(
            f"the declared cell set is 3 tiers x 5 seeds = {len(expected)}; got {len(cells)} "
            f"records covering {len(got)} distinct cells, missing {sorted(expected - got)}"
        )

    checked = 0
    max_flip = 0.0
    max_tvd = 0.0
    for cell in cells:
        where = f"{cell['tier']}@{cell['seed']}"
        interventions = cell["interventions"]
        if sorted(interventions) != sorted(INTERVENTION_KEYS):
            raise ValueError(
                f"{where}: the twelve declared interventions are {sorted(INTERVENTION_KEYS)}, got "
                f"{sorted(interventions)}; the grid may not grow after the fact"
            )
        if str(cell.get("rtg_mode")) != NORTG_RTG_MODE:
            raise ValueError(
                f"{where}: the checkpoint records rtg_mode {cell.get('rtg_mode')!r}, so rtg_mode "
                f"did not reach the training path and this arm is not the ablation it claims"
            )
        for key, values in interventions.items():
            flip = float(values["flip_rate"])
            if flip != 0.0:
                raise ValueError(
                    f"{where} intervention {key}: flip_rate is {flip!r}, so this checkpoint "
                    "did not ignore the return token it was never trained with; rtg_mode did not "
                    "reach the training path"
                )
            max_flip = max(max_flip, flip)
            max_tvd = max(max_tvd, float(values.get("tvd", 0.0)))
            checked += 1

    return {
        "role": "Q3, the arm-validity GATE: a model trained without the return token must be "
                "insensitive to it under every declared intervention",
        "n_cells": len(cells),
        "n_values_checked": checked,
        "max_flip_rate": max_flip,
        "max_tvd": max_tvd,
        "interventions": list(INTERVENTION_KEYS),
        "cells": cells,
    }


# ----------------------------------------------------------------------
# Scoring the registered predictions
# ----------------------------------------------------------------------


def score_q1(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Q1: the paired absolute difference is largest on ``mix50`` and smallest on ``random``.

    The scored quantity is ``abs(mean_difference)`` on the **raw ATT scale**, registered in
    ``docs/plans/p5.3b.md`` section 3.2 and confirmed by ``BRIEF_30`` AMENDMENT A2.  It is the
    quantity Q2's CI is about, so the two predictions stay on one scale.

    ⭐ The raw scale is the **conservative** choice: the three ``dt`` ATTs are 105, 108 and 420, so
    if differences scaled with baseline ATT then ``random`` would show the largest raw difference --
    the opposite of what Q1 predicts.  The normalised column below is a reading aid and is **not** a
    scored claim; switching to it after seeing the result is forbidden by the plan's section 3.3.

    **Endpoints, never a trend** -- section 1b's R3 was falsified on exactly a monotonicity claim.
    """
    magnitudes = {tier: float(entry["abs_mean_difference"]) for tier, entry in comparisons.items()}
    order = sorted(magnitudes, key=lambda tier: (magnitudes[tier], tier))
    largest, smallest = order[-1], order[0]
    values = sorted(magnitudes.values())
    secondary_abs = {
        tier: float(entry.get("mean_absolute_difference", float("nan")))
        for tier, entry in comparisons.items()
    }
    normalised = {
        tier: magnitudes[tier] / float(entry["att_dt_mean"])
        for tier, entry in comparisons.items()
        if float(entry.get("att_dt_mean", 0.0))
    }
    secondary_order = sorted(secondary_abs, key=lambda tier: (secondary_abs[tier], tier))
    normalised_order = sorted(normalised, key=lambda tier: (normalised[tier], tier))
    return {
        "prediction": "the paired absolute difference between the dt and dt_nortg arms is largest "
                      "on mix50 and smallest on random",
        "registered_in": "BRIEF_30 section 3 Q1; scoring quantity fixed in docs/plans/p5.3b.md "
                         "section 3.2 and confirmed by AMENDMENT A2",
        "quantity": "abs(mean_difference) of the paired per-draw comparison",
        "scale": "raw ATT",
        "scale_is_conservative": (
            "if paired differences scaled with baseline ATT, random (dt ATT 420.38) would show the "
            "largest raw difference, which is the opposite of what Q1 predicts"
        ),
        "abs_mean_difference": dict(sorted(magnitudes.items())),
        "largest": largest,
        "smallest": smallest,
        "tie_break": "tier name ascending",
        "ties_present": len(set(values)) != len(values),
        "holds": bool(largest == "mix50" and smallest == "random"),
        "secondary_not_registered": {
            "mean_absolute_difference": dict(sorted(secondary_abs.items())),
            "mean_absolute_difference_largest": secondary_order[-1],
            "mean_absolute_difference_smallest": secondary_order[0],
            "normalised_by_att_dt_mean": dict(sorted(normalised.items())),
            "normalised_largest": normalised_order[-1] if normalised_order else None,
            "normalised_smallest": normalised_order[0] if normalised_order else None,
            "status": "reported, not scored; the registered ordering is the raw one above",
        },
    }


def score_q2(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Q2: ``random`` is a null control predicted from an independent instrument.

    P5.3a measured ``random``'s conditioned DT at ``flip_rate = 0.000000``, 0 of 7200, on every
    intervention.  If the token carries nothing there, training without it should cost nothing
    there.  A large CI-excluding difference on ``random`` **indicts the wiring before it indicts the
    science** -- the same direction of inference as A8's ``fixedtime`` prediction in P5.3a.

    ⚠️ No equivalence verdict and no threshold: ``PREREGISTRATION`` A7 withdrew the per-tier delta
    rule, and a CI containing 0 is a failure to reject, never a demonstration of equivalence.
    """
    if "random" not in comparisons:
        raise ValueError("Q2 is a prediction about the random tier and it is not in the comparisons")
    paired = comparisons["random"]["paired"]
    low, high = float(paired["ci95_low"]), float(paired["ci95_high"])
    contains = low <= 0.0 <= high
    reading = (
        "the 95 % CI of the paired difference contains 0: a FAILURE TO REJECT the null of no "
        "difference, and never a demonstration of equivalence"
        if contains
        else "the 95 % CI of the paired difference excludes 0: removing the prompt changed mean "
             "held-out ATT on this tier, at this budget, at 5 seeds"
    )
    return {
        "prediction": "dt - dt_nortg on random has a 95 % CI containing 0",
        "registered_in": "BRIEF_30 section 3 Q2",
        "basis": "P5.3a measured random's conditioned DT at flip_rate 0.000000, 0 of 7200, on "
                 "every intervention -- an independent instrument",
        "tier": "random",
        "mean_difference": float(paired["mean_difference"]),
        "ci95_low": low,
        "ci95_high": high,
        "ci_contains_zero": bool(contains),
        "holds": bool(contains),
        "reading": reading,
        "if_falsified": "a large CI-excluding difference on random indicts the wiring before it "
                        "indicts the science",
    }


def score_q3(probe_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Q3 as a scored record.  The refusal lives in :func:`assert_arm_validity`; this reports it."""
    record = assert_arm_validity(probe_cells)
    return {
        "prediction": "every dt_nortg checkpoint shows flip_rate exactly 0.0 under P5.3a's probe "
                      "on all 12 interventions, and carries rtg_mode == 'zero'",
        "registered_in": "BRIEF_30 section 3 Q3",
        "status": "a GATE, not a result",
        "n_cells": record["n_cells"],
        "n_values_checked": record["n_values_checked"],
        "max_flip_rate": record["max_flip_rate"],
        "holds": bool(record["max_flip_rate"] == 0.0),
    }


# ----------------------------------------------------------------------
# The artifact
# ----------------------------------------------------------------------


def report_artifact(
    *,
    cells: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    probe_cells: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    selection: Mapping[str, Any],
    timings: Mapping[str, Any],
    measurement_inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble the one committed artifact.  Validates the design; ``main`` validates the data."""
    if sorted(comparisons) != sorted(NORTG_TIERS):
        raise ValueError(
            f"the registered tier set is {list(NORTG_TIERS)} and the comparisons cover "
            f"{sorted(comparisons)}; a tier may not be added after the numbers exist "
            "(BRIEF_30 section 6.1)"
        )
    expected = {(tier, int(seed)) for tier in NORTG_TIERS for seed in TRAINING_SEEDS}
    got = {(str(cell["tier"]), int(cell["seed"])) for cell in cells}
    if got != expected or len(list(cells)) != len(expected):
        raise ValueError(
            f"the declared cell set is 3 tiers x 5 seeds = {len(expected)}; got "
            f"{len(list(cells))} records covering {len(got)} distinct cells"
        )

    arm_validity = assert_arm_validity(probe_cells)
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": "P5.3b: fifteen dt_nortg cells trained with rtg_mode='zero', paired against the "
                "committed dt column of P4.6/P4.7 over the same corpus, seeds, budget and held-out "
                "draws. It reports measured quantities and issues no equivalence verdict.",
        "registered_in": "docs/plans/p5.3b.md; predictions Q1-Q3 in BRIEF_30 section 3",
        "method": NORTG_METHOD,
        "reference_method": REFERENCE_METHOD,
        "rtg_mode": NORTG_RTG_MODE,
        "tiers": list(NORTG_TIERS),
        "seeds": list(TRAINING_SEEDS),
        "held_out_draws": list(HELD_OUT_DRAWS),
        "declared_gradient_steps": DECLARED_GRADIENT_STEPS,
        "context_length": CONTEXT_LENGTH,
        "tier_selection": dict(selection),
        "cells": [dict(cell) for cell in cells],
        "episodes": [dict(entry) for entry in episodes],
        "comparisons": {tier: dict(entry) for tier, entry in comparisons.items()},
        "reference_dt_cells": {
            tier: {
                "arm": f"{REFERENCE_METHOD}@{tier}",
                "source": TIER_GRID_ARTIFACT[tier],
                "att_horizon_mean": float(comparisons[tier]["att_dt_mean"]),
                "reused": "read, never retrained (BRIEF_30 section 6.2)",
            }
            for tier in NORTG_TIERS
        },
        "arm_validity": arm_validity,
        "predictions": {
            "Q1": score_q1(comparisons),
            "Q2": score_q2(comparisons),
            "Q3": score_q3(probe_cells),
        },
        "gates": dict(gates),
        "timings_seconds": dict(timings),
        "equivalence": (
            "NONE. PREREGISTRATION A7 withdrew the per-tier delta rule on 2026-08-25 because it "
            "spanned eleven orders of magnitude. This task defines no threshold and issues no "
            "verdict. A 95 % CI containing 0 is a failure to reject, never a demonstration of "
            "equivalence."
        ),
        "limitations": list(_LIMITATIONS),
        "runtime": runtime_provenance(measurement_commits(list(measurement_inputs))),
    }
    assert_no_verdicts(payload)
    return payload


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``gate1``, ``control``, ``train``, ``evaluate``, ``probe``, ``report``."""
    parser = argparse.ArgumentParser(
        prog="python -m offline.nortg_campaign",
        description="P5.3b: the dt_nortg campaign on hz1x1.",
    )
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--work-dir", default="output/p5_3b")
    parser.add_argument("--checkpoint-dir", default="output/p5_3b/checkpoints")
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--engine-seed", type=int, default=ENGINE_SEED)
    parser.add_argument("--steps", type=int, default=DECLARED_GRADIENT_STEPS)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10_000)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gate1", help="reused dt identity, manifests, and the per-tier re-rolls")
    sub.add_parser("control", help="Gate 2: retrain the committed control cell and compare")
    train = sub.add_parser("train", help="train one tier's five dt_nortg seeds")
    train.add_argument("--tier", required=True, choices=list(NORTG_TIERS))
    evaluate = sub.add_parser("evaluate", help="evaluate ONE cell over the held-out pool")
    evaluate.add_argument("--tier", required=True, choices=list(NORTG_TIERS))
    evaluate.add_argument("--seed", required=True, type=int, choices=list(TRAINING_SEEDS))
    probe = sub.add_parser("probe", help="Gate 3: arm validity over one tier's five checkpoints")
    probe.add_argument("--tier", required=True, choices=list(NORTG_TIERS))
    sub.add_parser("report", help="assemble docs/data/p5_3b_nortg.json")
    return parser


def _chunk(work: Path, name: str) -> dict[str, Any]:
    path = work / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: this report needs every campaign chunk; run `train` and `evaluate` for it "
            "first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code."""
    args = build_parser().parse_args(argv)
    work = assert_writable(args.work_dir)
    out_dir = Path(args.out_dir)
    data_dir = Path(args.out_dir)

    if args.command == "report":
        return _run_report(args, work, out_dir, data_dir)

    pin_torch_threads(args.torch_threads)
    if args.command == "gate1":
        return _run_gate1(args, work, data_dir)
    if args.command == "control":
        return _run_control(args, work, data_dir)
    if args.command == "train":
        return _run_train(args, work)
    if args.command == "evaluate":
        return _run_evaluate(args, work)
    return _run_probe(args, work)


def _run_gate1(args: argparse.Namespace, work: Path, data_dir: Path) -> int:
    identity = assert_reused_dt_identity(data_dir=data_dir, output_root=args.output_root)
    rerolls: dict[str, Any] = {}
    for tier, seed in GATE_1B_CELLS:
        checkpoint = Path(args.output_root) / TIER_CHECKPOINT_TEMPLATE[tier].format(seed=seed)
        started = time.time()
        produced = evaluate_cell(
            tier,
            seed,
            checkpoint=checkpoint,
            corpus_root=args.corpus_root,
            draws_root=args.draws_root,
            engine_seed=args.engine_seed,
            device=args.device,
            arm=f"{REFERENCE_METHOD}@{tier}",
        )
        committed = [
            entry
            for entry in json.loads(
                (data_dir / TIER_GRID_ARTIFACT[tier]).read_text(encoding="utf-8")
            )["episodes"]
            if entry["arm"] == f"{REFERENCE_METHOD}@{tier}" and int(entry["seed"]) == seed
        ]
        if len(committed) != len(HELD_OUT_DRAWS):
            raise ValueError(
                f"{tier} seed {seed}: {len(committed)} committed episodes, not "
                f"{len(HELD_OUT_DRAWS)}; 'found no differences' must never be 'compared nothing'"
            )
        rerolls[tier] = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "seconds": time.time() - started,
            **assert_reused_cells_reproduce(committed, produced),
        }
        print(f"gate 1b {tier} seed {seed}: {rerolls[tier]['compared']} episodes reproduce", flush=True)

    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "gate_1": identity,
            "gate_1b": {
                "role": "the reused dt column re-rolls bit-exactly under this campaign's own "
                        "harness; on mappo1000 it is the compensating control for DEFERRED 56",
                "cells": rerolls,
            },
            "runtime": runtime_provenance(),
        },
        assert_writable(work / "gate1.json"),
    )
    return 0


def _run_control(args: argparse.Namespace, work: Path, data_dir: Path) -> int:
    tier, seed = CONTROL_CELL
    inputs = training_inputs(tier, args.corpus_root)
    scratch = assert_writable(work / "control")
    scratch.mkdir(parents=True, exist_ok=True)
    destination = assert_writable(scratch / f"{tier}_dt_seed{seed}.pt")
    from agent.utils.utils import Utils

    device = torch.device(args.device) if args.device else Utils.resolve_device(None)
    started = time.time()
    train_dt(
        inputs.batch,
        state_dim=inputs.group[0],
        n_actions=inputs.group[1],
        seed=seed,
        declared_gradient_steps=int(args.steps),
        raise_to=None,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        device=device,
        checkpoint_path=destination,
        stats=inputs.stats,
        scenario_id=inputs.scenario_id,
        target_rtg=float(inputs.spec.target_rtg),
        rtg_scale=float(inputs.spec.rtg_scale),
        provenance=inputs.provenance,
        log_every=int(args.log_every),
    )
    seconds = time.time() - started
    digest = canonical_digest_of(destination)
    if digest != CONTROL_COMMITTED_DIGEST:
        raise ValueError(
            f"{destination}: canonical digest {digest} is not the committed "
            f"{CONTROL_COMMITTED_DIGEST}; threading rtg_mode through train_dt moved the "
            "conditioned path and every merged DT number would be affected"
        )
    committed = Path(args.output_root) / "p4_6" / "checkpoints" / f"{tier}_dt_seed{seed}.pt"
    payload_record = assert_payload_matches_committed(destination, committed)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "role": "Gate 2: the committed control cell retrained through the modified train_dt",
            "cell": {"tier": tier, "seed": seed},
            "seconds": seconds,
            "canonical_digest": digest,
            "committed_digest": CONTROL_COMMITTED_DIGEST,
            "payload_comparison": payload_record,
            "runtime": runtime_provenance(),
        },
        assert_writable(work / "control.json"),
    )
    print(f"gate 2: digest {digest[:12]} reproduces in {seconds:.1f}s", flush=True)
    return 0


def _run_train(args: argparse.Namespace, work: Path) -> int:
    inputs = training_inputs(args.tier, args.corpus_root)
    print(
        f"tier {args.tier}: training streams {inputs.n_streams}  windows {inputs.n_windows}  "
        f"target {inputs.spec.target_rtg}  scale {inputs.spec.rtg_scale}  "
        f"rtg_mode {NORTG_RTG_MODE}",
        flush=True,
    )
    runs = []
    for seed in TRAINING_SEEDS:
        record = train_cell(
            args.tier,
            int(seed),
            corpus_root=args.corpus_root,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            steps=int(args.steps),
            log_every=int(args.log_every),
            inputs=inputs,
        )
        runs.append(record)
        print(
            f"  {args.tier} dt_nortg seed {seed}: {record['seconds']:.1f}s  "
            f"final loss {record['final_loss']:.5f}  digest {record['canonical_digest'][:12]}",
            flush=True,
        )
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "tier": args.tier,
            "rtg_mode": NORTG_RTG_MODE,
            "declared_gradient_steps": int(args.steps),
            "training_streams": inputs.n_streams,
            "training_windows": inputs.n_windows,
            "normalisation_note": (
                "mix50's NormalizationStats is fitted on the UNION of the six directories the "
                "three mixtures share and is identical for mix33/mix50/mix67 (count 216000, raw "
                "std 13155.3172). That is how P4.7 trained it, so reusing it is consistent, but "
                "the summary is not a property of mix50."
            ),
            "runs": runs,
            "runtime": runtime_provenance(),
        },
        assert_writable(work / f"train_{args.tier}.json"),
    )
    return 0


def _run_evaluate(args: argparse.Namespace, work: Path) -> int:
    training = _chunk(work, f"train_{args.tier}.json")
    runs = [run for run in training["runs"] if int(run["seed"]) == int(args.seed)]
    if len(runs) != 1:
        raise ValueError(
            f"{args.tier} seed {args.seed}: the training chunk records {len(runs)} runs, not 1"
        )
    run = runs[0]
    digest = canonical_digest_of(run["checkpoint"])
    if digest != run["canonical_digest"]:
        raise ValueError(
            f"{run['checkpoint']}: canonical digest {digest} is not the trained "
            f"{run['canonical_digest']}; this is not the model the training chunk records"
        )
    arm = nortg_arm_key(args.tier)
    print(f"{arm} seed {args.seed} over {len(HELD_OUT_DRAWS)} draws", flush=True)
    started = time.time()
    produced = evaluate_cell(
        args.tier,
        int(args.seed),
        checkpoint=run["checkpoint"],
        corpus_root=args.corpus_root,
        draws_root=args.draws_root,
        engine_seed=args.engine_seed,
        device=args.device,
    )
    seconds = time.time() - started
    assert_cell_complete(NORTG_METHOD, args.tier, [int(args.seed)], list(HELD_OUT_DRAWS), produced)
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "arm": arm,
            "tier": args.tier,
            "seed": int(args.seed),
            "checkpoint": run["checkpoint"],
            "canonical_digest": digest,
            "seconds": seconds,
            "seconds_per_episode": seconds / len(produced),
            "cell": nortg_cell_record(produced, int(args.seed)),
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
        assert_writable(work / f"eval_{args.tier}_seed{args.seed}.json"),
    )
    print(f"  {arm} seed {args.seed}: {seconds:.1f}s ({seconds / len(produced):.3f}s/episode)", flush=True)
    return 0


def _run_probe(args: argparse.Namespace, work: Path) -> int:
    training = _chunk(work, f"train_{args.tier}.json")
    from offline.rtg_ablation import _tier_streams

    streams = _tier_streams(args.tier, args.corpus_root)
    cells = []
    timings = {}
    for run in sorted(training["runs"], key=lambda r: int(r["seed"])):
        started = time.time()
        cell = probe_nortg_cell(
            args.tier,
            int(run["seed"]),
            checkpoint_path=run["checkpoint"],
            corpus_root=args.corpus_root,
            device=args.device,
            streams=streams,
        )
        elapsed = time.time() - started
        timings[f"{args.tier}@{run['seed']}"] = elapsed
        cells.append(cell)
        worst = max(float(v["flip_rate"]) for v in cell["interventions"].values())
        print(
            f"probe {args.tier} seed {run['seed']}: rtg_mode {cell['rtg_mode']}  "
            f"max flip_rate {worst:.6f}  n={cell['n_steps']} in {elapsed:.1f}s",
            flush=True,
        )
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "tier": args.tier,
            "cells": cells,
            "timings_seconds": timings,
            "runtime": runtime_provenance(),
        },
        assert_writable(work / f"probe_{args.tier}.json"),
    )
    return 0


def _run_report(args: argparse.Namespace, work: Path, out_dir: Path, data_dir: Path) -> int:
    """Read every chunk, validate the data, assemble, then write.  Validation precedes the write."""
    chunks: list[Mapping[str, Any]] = []
    cells: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    probe_cells: list[dict[str, Any]] = []
    comparisons: dict[str, dict[str, Any]] = {}
    timings: dict[str, Any] = {"train_seconds": {}, "evaluate_seconds": {}, "probe_seconds": {}}

    for tier in NORTG_TIERS:
        training = _chunk(work, f"train_{tier}.json")
        probe = _chunk(work, f"probe_{tier}.json")
        chunks.extend((training, probe))
        probe_cells.extend(probe["cells"])
        for run in training["runs"]:
            timings["train_seconds"][f"{tier}@{run['seed']}"] = run["seconds"]
        timings["probe_seconds"].update(probe.get("timings_seconds", {}))

        produced: list[EpisodeResult] = []
        for seed in TRAINING_SEEDS:
            chunk = _chunk(work, f"eval_{tier}_seed{seed}.json")
            chunks.append(chunk)
            timings["evaluate_seconds"][f"{tier}@{seed}"] = chunk["seconds"]
            cells.append(chunk["cell"])
            episodes.extend(chunk["episodes"])
            produced.extend(
                EpisodeResult(
                    arm=e["arm"],
                    seed=int(e["seed"]),
                    draw_id=int(e["draw_id"]),
                    att_horizon=float(e["att_horizon"]),
                    horizon_vehicle_count=float(e["horizon_vehicle_count"]),
                    episode_reward=float(e["episode_reward"]),
                )
                for e in chunk["episodes"]
            )
        assert_cell_complete(NORTG_METHOD, tier, list(TRAINING_SEEDS), list(HELD_OUT_DRAWS), produced)

        reference = committed_dt_episodes(tier, data_dir=data_dir)
        stats = paired_stats(reference, produced)
        pooled = float(stats["paired"]["mean_difference"])
        stats["per_seed"] = per_seed_differences(reference, produced, pooled)
        stats["att_dt_mean"] = float(
            json.loads((data_dir / TIER_GRID_ARTIFACT[tier]).read_text(encoding="utf-8"))["cells"][
                f"{REFERENCE_METHOD}@{tier}"
            ]["att_horizon_mean"]
        )
        stats["att_dt_nortg_mean"] = float(
            np.mean([e.att_horizon for e in produced])
        )
        comparisons[tier] = stats

    gate1 = _chunk(work, "gate1.json")
    control = _chunk(work, "control.json")
    chunks.extend((gate1, control))
    probe_artifact = json.loads(
        (data_dir / "p5_3a_rtg_probe.json").read_text(encoding="utf-8")
    )
    payload = report_artifact(
        cells=cells,
        episodes=episodes,
        comparisons=comparisons,
        probe_cells=probe_cells,
        gates={
            "gate_1": gate1["gate_1"],
            "gate_1b": gate1["gate_1b"],
            "gate_2": {k: v for k, v in control.items() if k != "runtime"},
        },
        selection=assert_selection_still_holds(probe_artifact),
        timings=timings,
        measurement_inputs=chunks,
    )
    if not payload["runtime"]["measurement_git_commits"]:
        raise ValueError(
            f"the report was assembled from {len(chunks)} chunk payloads but recorded no "
            "measurement commits; runtime.git_commit would then describe only when the report was "
            "written, which is the defect DEFERRED 39 exists to prevent"
        )
    write_json_atomic(payload, out_dir / "p5_3b_nortg.json")
    for tier in NORTG_TIERS:
        paired = comparisons[tier]["paired"]
        print(
            f"{tier}: dt - dt_nortg {paired['mean_difference']:+.4f} "
            f"[{paired['ci95_low']:+.4f}, {paired['ci95_high']:+.4f}]  "
            f"reversals {comparisons[tier]['per_seed']['seeds_reversed']}/5",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
