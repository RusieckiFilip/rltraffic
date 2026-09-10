"""P5.3b BL-2(b): A13(b)'s three-component decomposition, MEASURED on the campaign's own arms.

Artifact format version: ``p5.3b-decomposition/1.0`` -- ``docs/data/p5_3b_decomposition.json``.

WHY THIS EXISTS, AND WHY IT IS NOT ARITHMETIC
---------------------------------------------
``PREREGISTRATION`` **A13(b)** makes the three-component decomposition a **required reported
quantity for every report of the two ATT definitions' difference**.  P5.3b reports that difference
in three places and the decomposition nowhere, so the campaign is in breach of a registered
requirement.  ``docs/reviews/P5.3b.md`` **BL-2(b)** filed it.

🚨 **It cannot be computed from the five A11(b) quantities.**  The terms need reconstructions ``P``
(admitted population on the POOL clock) and ``W_running`` (admitted population on the ADMISSION
clock), which exist only as the per-second reconstructions built by
``offline/engine_att_reference.py``'s ``EngineObservationRecorder``.  The campaign's episodes carry
``att_ours``, ``att_engine``, ``entered``, ``created`` and ``never_entered`` -- the endpoints, not
the intermediates.  **So this module re-rolls all 30 cells under the Gate 0 observer.**

⛔ **And the tautology this replaces, because it was believed for two days.**
``-409.1450 - (-194.4757) = -214.669 = (-2.112) - 212.557`` is the identity
``delta_engine - delta_ours == delta(engine - ours)``.  It is true of any two numbers whatsoever and
says **nothing** about what the gap is made of.  ``docs/reviews/P5.3b.md`` attributed the gap to the
never-admitted population on that basis; the 2026-09-10 Decisions Log retracts the attribution.
**No sentence about what the 215 IS may be written from this module's output except one this module
measured.**

THE ORIENTATION -- READ THIS BEFORE TRUSTING A SIGN
--------------------------------------------------
``PREREGISTRATION`` A13(b) writes the identity as

    att_engine - att_ours = POPULATION (E - P) + CLOCK ORIGIN (P - W_running) + CADENCE (W_running - C)

``offline/engine_att_reference.py:853-892`` implements the **exact negation** of that, term by term::

    term_population   = entered_population - engine_population      #  P - E
    term_clock_origin = entered_running    - entered_population     #  W_running - P
    term_cadence      = att_ours           - entered_running        #  C - W_running
    identity          = "att_ours - att_engine = population + clock_origin + cadence"

**This module reports the REPO's orientation** (``BRIEF_33`` AMENDMENT A3): re-signing the terms
would mean a second implementation of ``_decomposition_summary``, which ``BRIEF_33`` section 3.1
forbids, and *the registered quantity is the decomposition, not its sign convention*.  Every artifact
carries :data:`ORIENTATION` and :data:`ORIENTATION_NOTE` so the negation is stated rather than
inferred.  ⚠️ **Consequence a reader must not trip over: ``term_clock_origin`` is expected NEGATIVE**
under this orientation -- it is minus the insertion-buffer wait of vehicles that DID enter.

WHAT IS REUSED AND NEVER RE-IMPLEMENTED
---------------------------------------
The env, the rollout and all four reconstructions are ``engine_att_reference.gate_episode``, which is
merged, reviewed and carries Gate 0's verdict; this module is a harness over it and
**``offline/engine_att_reference.py`` is READ-ONLY here**.  The per-cell preamble is
``nortg_campaign.evaluate_cell``'s (``:574-643``), imported element for element -- ``tier_spec``,
``env_settings_for_tiers``, ``_dt_factory`` with the tier's declared ``target_rtg``,
``draw_config_path``, ``created_from_flow``, ``HELD_OUT_DRAWS``, ``ENGINE_SEED``.  The **only**
difference from the campaign's own evaluation is that ``gate_episode`` replaces ``probe_episode`` and
the action factory is wrapped by a recorder.  ⚠️ :func:`_decomposition_summary` is imported under its
private name from a module this task may not edit; that is disclosed rather than worked around,
because the alternative is the re-implementation section 3.1 forbids.

🔒 THE LOAD-BEARING CHECK
-------------------------
Gate 0 established ``deviation_c1 == 0.0`` and ``deviation_c3c == 0.0`` on 46 episodes of **other
arms**.  **Whether the observed DT trajectory equals the campaign's is a NEW FACT**, and
:func:`decomposition_artifact` is what establishes it: it refuses to build -- so ``report`` writes
**nothing** -- unless, on every one of the 3,000 episodes, ``att_engine_call == committed_att_engine``
and ``att_ours == committed_att_ours`` under ``==``, and ``deviation_c1``, ``deviation_c3c`` and
``decomposition_residual`` are all exactly ``0.0``.  A decomposition of a *different* trajectory is
not the required quantity.

⭐ **The chunk's own ``reproduces_committed`` flag is NEVER trusted** (``BRIEF_33`` AMENDMENT A4
calls this the load-bearing design decision of the task).  ``report`` re-reads the committed pair
from its sources -- ``docs/data/p5_3b_nortg.json`` for ``dt_nortg``,
``nortg_campaign.rederived_dt_episodes`` for ``dt`` -- and compares those against the chunk's stored
``committed_*`` as well.  A chunk that agrees with itself and with nothing else is refused.  P8.4b's
``reproduces_committed`` was read as evidence once already.

⚠️ **``decomposition_residual`` is algebraically ``deviation_c1``.**  The three terms telescope to
``att_ours - engine_population`` while the total is ``att_ours - att_engine_call``, so the residual is
``|att_engine_call - engine_population|`` in exact arithmetic.  It therefore corroborates criterion 1
and **nothing further**; it is not independent evidence about the decomposition, and the artifact says
so in ``what_this_does_not_say``.

THE FILESYSTEM-MUTATION BARRIER
-------------------------------
``output/`` is gitignored, has no backup and holds every checkpoint under twelve manifests.  Every
destination goes through :func:`assert_decomposition_writable`, which is **default-deny with its own
allow-list** -- it permits ``output/p5_3b_decomp`` and ``output/SHA256SUMS_p5_3b_decomp.txt`` and
refuses every other component under ``output/``, **including ``p5_3b``, this campaign's own
predecessor**, and including directories that do not exist yet.  It is deliberately a **separate
function with a separate name** from ``nortg_campaign.assert_writable``: one name meaning two
different allow-lists is ``DEFERRED`` 57's class.  All validation completes before the first byte is
written, and a refused destination creates no directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from offline import nortg_campaign
from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ALLOWED_DECOMP_OUTPUT_ENTRIES",
    "CONTRAST_TOLERANCE",
    "DECOMP_METHODS",
    "DEFAULT_WORK_DIRNAME",
    "IDENTITY",
    "NORTG_CHECKPOINT_TEMPLATE",
    "NORTG_MANIFEST",
    "ORIENTATION",
    "ORIENTATION_NOTE",
    "TERM_KEYS",
    "ActionRecorder",
    "assert_chunks_complete",
    "assert_contrast_consistent",
    "assert_decomposition_writable",
    "assert_identity_is_exact",
    "assert_nortg_checkpoint_identity",
    "assert_rows_reproduce_committed",
    "action_sequence_sha256",
    "build_parser",
    "chunk_name",
    "chunk_is_reusable",
    "committed_reference_rows",
    "contrast_summaries",
    "declared_contrasts",
    "decomposition_artifact",
    "diff_json_paths",
    "main",
    "per_arm_summaries",
    "recording_factory",
    "run_cell",
]

#: Bumped whenever the on-disk layout of ``docs/data/p5_3b_decomposition.json`` changes (C6).
ARTIFACT_FORMAT_VERSION = "p5.3b-decomposition/1.0"

#: The two arms of the contrast.  ``dt`` is P4.6/P4.7's committed column, re-rolled here under the
#: observer; ``dt_nortg`` is P5.3b's ablated column.  Both carry the tier's declared ``target_rtg``
#: -- the ablation is in the weights, not in the prompt handed to the harness.
DECOMP_METHODS: tuple[str, ...] = ("dt", "dt_nortg")

#: DEFAULT-DENY.  The **only** two entries under ``output/`` this task may write.  ⚠️ ``p5_3b`` is
#: deliberately absent: this campaign reads its predecessor's tree and never writes to it.
ALLOWED_DECOMP_OUTPUT_ENTRIES: tuple[str, ...] = (
    "p5_3b_decomp",
    "SHA256SUMS_p5_3b_decomp.txt",
)

DEFAULT_WORK_DIRNAME = "p5_3b_decomp"

#: The ablated checkpoints, relative to ``--output-root``, and the manifest their sha256 is checked
#: against AT CONSUMPTION (``BRIEF_27`` B3(a): a digest checked once is not a digest checked when
#: used).  The ``dt`` half resolves through ``nortg_campaign.TIER_CHECKPOINT_TEMPLATE`` instead.
NORTG_CHECKPOINT_TEMPLATE = "p5_3b/checkpoints/{tier}_dt_nortg_seed{seed}.pt"
NORTG_MANIFEST = "SHA256SUMS_p5_3b.txt"

#: The three terms, in the order the identity states them.
TERM_KEYS: tuple[str, ...] = ("population", "clock_origin", "cadence")

ORIENTATION = "att_ours - att_engine"
IDENTITY = "att_ours - att_engine = population + clock_origin + cadence"
ORIENTATION_NOTE = (
    "This is the NEGATION of the orientation PREREGISTRATION A13(b) writes, which states "
    "att_engine - att_ours = (E - P) + (P - W_running) + (W_running - C). "
    "offline/engine_att_reference.py:853-892 implements the negation term by term and this module "
    "reports the implementation's orientation (BRIEF_33 AMENDMENT A3), because re-signing would be "
    "a second implementation of a reviewed function. Consequence: clock_origin is expected NEGATIVE "
    "here -- it is minus the insertion-buffer wait of vehicles that DID enter the network."
)

#: The per-tier consistency check's tolerance.  ⚠️ ``==`` is deliberately NOT used at tier level:
#: ``delta_ours - delta_engine`` and the sum of the three term contrasts reduce the same 3,000 floats
#: in **different summation orders**, so exact equality would condemn a correct implementation.
#: **Per episode the identity is exact and is checked exactly** by :func:`assert_identity_is_exact`.
CONTRAST_TOLERANCE = 1e-9


# ----------------------------------------------------------------------
# The fence
# ----------------------------------------------------------------------


def assert_decomposition_writable(path: str | Path) -> Path:
    """DEFAULT-DENY: refuse anything under an ``output/`` that is not this task's own.

    A path is allowed when the component immediately after ``output`` is in
    :data:`ALLOWED_DECOMP_OUTPUT_ENTRIES`; **everything else under ``output/`` is refused**,
    including directories that do not exist yet and including ``output/p5_3b``, which this campaign
    READS and must never write.

    Matching is on whole **path components** of the resolved path, never on a string prefix: a prefix
    test makes ``output/p5_3b`` and ``output/p5_3b_decomp`` indistinguishable, and that pair is
    exactly the trap here.  Returns the path unresolved, so it composes.
    """
    target = Path(path)
    parts = Path(target).resolve().parts
    for index, part in enumerate(parts[:-1]):
        head = parts[index + 1]
        if part == "output" and head not in ALLOWED_DECOMP_OUTPUT_ENTRIES:
            raise ValueError(
                f"{target}: output/{head} belongs to another campaign and is read-only here; "
                f"P5.3b-fix writes only {list(ALLOWED_DECOMP_OUTPUT_ENTRIES)} (BRIEF_33 section "
                "3.6). This fence is default-deny, so a directory added to output/ after this code "
                "was written is protected without being named -- and output/p5_3b in particular is "
                "READ by this task and must never be written by it"
            )
    return target


# ----------------------------------------------------------------------
# Action recording -- BL-2(c)'s half
# ----------------------------------------------------------------------


def action_sequence_sha256(actions: Sequence[Any]) -> str:
    """sha256 over the ``int64`` bytes of the ``(T, n_intersections)`` action array.

    Convention, stated because a digest with an unstated convention is not reproducible: the
    sequence is stacked in decision order into a C-contiguous array of dtype ``int64`` and shape
    ``(T, n_intersections)``, and the digest is over ``.tobytes()``.

    ⚠️ **Named ``action_sequence_sha256`` and never ``episode_sha256``.**  Contract C6's
    ``episode_sha256`` is over actions **and rewards**; one name meaning two different digests is
    ``DEFERRED`` 57/70/71's class, which this repo has now filed three times.
    """
    if len(actions) == 0:
        raise ValueError("an empty action sequence has no digest; an episode makes >= 1 decision")
    stacked = np.asarray(
        [np.asarray(action, dtype=np.int64).reshape(-1) for action in actions], dtype=np.int64
    )
    return hashlib.sha256(np.ascontiguousarray(stacked).tobytes()).hexdigest()


@dataclass
class ActionRecorder:
    """Every action one episode's policy returned, recorded without touching the trajectory.

    The recorder stores a **copy** of each action and the wrapped callable returns the inner
    callable's own object unchanged, so wrapping is inert by construction; ``test_nortg_decomposition``
    pins that with a stub and the 3,000-episode reproduction check would catch it anyway.
    """

    actions: list[np.ndarray] = field(default_factory=list)
    n_actions: tuple[int, ...] | None = None

    def bind(self, n_actions: Sequence[int]) -> None:
        """Fix the per-intersection action counts, from ``Utils.infer_action_counts``.

        The histogram width is the env's DECLARED action count, never ``max(observed) + 1``: a
        collapsed policy that only ever emits phase 0 must produce a full-width histogram of zeros
        rather than a one-column one that looks like a different env.
        """
        self.n_actions = tuple(int(n) for n in n_actions)
        self.actions = []

    def record(self, action: Any) -> None:
        """Store a copy of one decision."""
        stored = np.array(action, copy=True)
        if not np.issubdtype(stored.dtype, np.integer):
            raise TypeError(
                f"an action of dtype {stored.dtype} is not an integer phase index; casting it "
                "silently would truncate, and contract C1 says actions are one int per intersection"
            )
        self.actions.append(stored.reshape(-1))

    def summary(self) -> dict[str, Any]:
        """``n_decisions``, ``action_counts`` (per intersection, per action index) and the digest."""
        if self.n_actions is None:
            raise ValueError("the recorder was never bound to an env's declared action counts")
        if not self.actions:
            raise ValueError("no decision was recorded, so this episode has no action sequence")
        stacked = np.asarray(self.actions, dtype=np.int64)
        if stacked.shape[1] != len(self.n_actions):
            raise ValueError(
                f"actions are {stacked.shape[1]} wide but the env declares "
                f"{len(self.n_actions)} intersections"
            )
        counts: list[list[int]] = []
        for index, width in enumerate(self.n_actions):
            column = stacked[:, index]
            if int(column.max()) >= width or int(column.min()) < 0:
                raise ValueError(
                    f"intersection {index} played action {int(column.max())} but declares {width} "
                    "actions; an out-of-range action would vanish from the histogram rather than "
                    "being reported"
                )
            counts.append([int((column == action).sum()) for action in range(width)])
        return {
            "n_decisions": int(stacked.shape[0]),
            "action_counts": counts,
            "action_sequence_sha256": action_sequence_sha256(self.actions),
        }


def recording_factory(
    inner: Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]],
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]], ActionRecorder]:
    """Wrap a ``choose_action`` factory so every decision is recorded and none is altered.

    Returns ``(factory, recorder)``.  The wrapped factory binds the recorder's histogram width from
    the env it is handed, then returns a callable that defers to the inner one, records a copy and
    **returns the inner callable's object itself** -- not a copy, not a cast.
    """
    recorder = ActionRecorder()

    def factory(env: Any) -> Callable[[Any, dict[str, Any]], np.ndarray]:
        from agent.utils.utils import Utils

        recorder.bind(Utils.infer_action_counts(env.action_space, env.intersections))
        chosen = inner(env)

        def recording_choose_action(env_: Any, info: dict[str, Any]) -> np.ndarray:
            action = chosen(env_, info)
            recorder.record(action)
            return action

        return recording_choose_action

    return factory, recorder


# ----------------------------------------------------------------------
# Reading the committed columns
# ----------------------------------------------------------------------


def committed_reference_rows(
    *, data_dir: str | Path, rederivation_dir: str | Path
) -> dict[tuple[str, str, int, int], dict[str, float]]:
    """The committed ``(att_engine, att_ours)`` pair for every cell of both arms.

    Keyed ``(method, tier, seed, draw_id)``.  ``dt_nortg`` comes from
    ``docs/data/p5_3b_nortg.json``'s own ``episodes``; ``dt`` from
    ``nortg_campaign.rederived_dt_episodes``, which verifies every row against the committed
    ``p4_6``/``p4_7`` grids before returning it.  **This is the source of truth the chunks are
    checked against**, and it is read at report time rather than inherited from the chunks.
    """
    data = Path(data_dir)
    rows: dict[tuple[str, str, int, int], dict[str, float]] = {}

    payload = json.loads((data / "p5_3b_nortg.json").read_text(encoding="utf-8"))
    for entry in payload["episodes"]:
        key = (
            nortg_campaign.NORTG_METHOD,
            str(entry["tier"]),
            int(entry["seed"]),
            int(entry["draw_id"]),
        )
        rows[key] = {
            "att_engine": float(entry["att_engine"]),
            "att_ours": float(entry["att_ours"]),
        }

    for tier in nortg_campaign.NORTG_TIERS:
        for entry in nortg_campaign.rederived_dt_episodes(
            tier, rederivation_dir=rederivation_dir, data_dir=data
        ):
            key = (
                nortg_campaign.REFERENCE_METHOD,
                tier,
                int(entry["seed"]),
                int(entry["draw_id"]),
            )
            rows[key] = {
                "att_engine": float(entry["att_engine"]),
                "att_ours": float(entry["att_ours"]),
            }

    expected = len(DECOMP_METHODS) * len(nortg_campaign.NORTG_TIERS) * len(TRAINING_SEEDS) * len(
        HELD_OUT_DRAWS
    )
    if len(rows) != expected:
        raise ValueError(
            f"the committed reference covers {len(rows)} cells, not {expected}; a reproduction "
            "check over a partial reference is not the check it claims to be"
        )
    return rows


def declared_contrasts(data_dir: str | Path) -> dict[str, dict[str, float]]:
    """Per tier, the committed ``mean_difference`` under each definition.

    Read from ``docs/data/p5_3b_nortg.json``'s ``comparisons.<tier>.by_definition``.  The sign
    convention is ``paired_stats``': ``mean(ATT_dt - ATT_dt_nortg)``, left = ``dt``.
    """
    payload = json.loads((Path(data_dir) / "p5_3b_nortg.json").read_text(encoding="utf-8"))
    declared: dict[str, dict[str, float]] = {}
    for tier in nortg_campaign.NORTG_TIERS:
        by_definition = payload["comparisons"][tier]["by_definition"]
        declared[tier] = {
            definition: float(by_definition[definition]["paired"]["mean_difference"])
            for definition in nortg_campaign.ATT_DEFINITIONS
        }
    return declared


def assert_nortg_checkpoint_identity(*, output_root: str | Path) -> dict[str, Any]:
    """The fifteen ablated checkpoints are the ones ``SHA256SUMS_p5_3b.txt`` recorded.

    Checked AT CONSUMPTION, in this process, before a single episode is rolled.  Refuses a
    checkpoint absent from the manifest as loudly as one whose digest moved: a file outside its own
    campaign manifest has no integrity record at all (``DEFERRED`` 56's lesson).
    """
    from offline.method_tier_grid import file_sha256

    root = Path(output_root)
    manifest = nortg_campaign._manifest_digests(root / NORTG_MANIFEST)
    checkpoints: list[dict[str, Any]] = []
    for tier in nortg_campaign.NORTG_TIERS:
        for seed in TRAINING_SEEDS:
            relative = NORTG_CHECKPOINT_TEMPLATE.format(tier=tier, seed=seed)
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"the ablated checkpoint is missing: {path}")
            if relative not in manifest:
                raise ValueError(
                    f"{relative} is not listed in {NORTG_MANIFEST}; a checkpoint outside its own "
                    "campaign manifest has no integrity record at consumption"
                )
            digest = file_sha256(path)
            if manifest[relative] != digest:
                raise ValueError(
                    f"{path}: file sha256 {digest} is not {NORTG_MANIFEST}'s {manifest[relative]}; "
                    "the ablated column would describe different weights"
                )
            checkpoints.append(
                {
                    "tier": tier,
                    "seed": int(seed),
                    "path": str(path),
                    "file_sha256": digest,
                    "manifest_checked_against": NORTG_MANIFEST,
                }
            )
    return {
        "role": "the ablated dt_nortg checkpoints, verified at consumption (BRIEF_27 B3(a))",
        "n_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


# ----------------------------------------------------------------------
# Rolling a cell
# ----------------------------------------------------------------------


def chunk_name(method: str, tier: str, seed: int) -> str:
    """``decomp_<method>_<tier>_seed<seed>.json``.

    ⚠️ ``BRIEF_33`` section 3.1 writes ``decomp_<arm>_<tier>_seed<seed>.json``; a literal ``arm``
    (``dt_nortg@mix50``) would repeat the tier and put ``@`` in a filename, so ``<arm>`` is read as
    the method label, following the repo's own ``REDERIVED_CELL_TEMPLATE``
    (``cell_{scenario}_{method}_at_{tier}_seed{seed}_draw{draw}.json``).
    """
    return f"decomp_{method}_{tier}_seed{int(seed)}.json"


def chunk_is_reusable(payload: Mapping[str, Any]) -> bool:
    """May a restart SKIP this cell?  Only if the chunk is complete AND clean.

    Complete = every one of ``HELD_OUT_DRAWS`` present; clean = ``n_mismatches == 0``; and the
    format version must match.  ⚠️ ``offline/campaigns/p5_3b.sh`` skips on ``[ -f ]`` alone, and
    ``assert_probe_cell_is_ablated``'s docstring records what that cost: *"the driver skips a tier
    whose probe chunk exists, so a bad chunk survived every restart."*  A partial or mismatching
    chunk is re-run and overwritten.
    """
    if str(payload.get("format_version")) != ARTIFACT_FORMAT_VERSION:
        return False
    if not bool(payload.get("is_complete")):
        return False
    if int(payload.get("n_mismatches", 1)) != 0:
        return False
    if sorted(int(draw) for draw in payload.get("draws", ())) != sorted(
        int(draw) for draw in HELD_OUT_DRAWS
    ):
        return False
    return len(payload.get("episodes", ())) == len(HELD_OUT_DRAWS)


def run_cell(
    method: str,
    tier: str,
    seed: int,
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    committed: Mapping[tuple[str, str, int, int], Mapping[str, float]],
    corpus_root: str | Path,
    draws_root: str | Path,
    engine_seed: int,
    device: str | None = None,
    draws: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Roll one cell over the held-out pool under the Gate 0 observer.

    The preamble is ``nortg_campaign.evaluate_cell``'s, element for element; the rolled function is
    ``engine_att_reference.gate_episode`` and the factory is wrapped by :func:`recording_factory`.
    Returns the chunk payload, including ``n_mismatches`` -- counted here so the driver can stop on
    the first bad cell rather than an hour later, and re-counted independently at report time.
    """
    from offline.admission_probe import created_from_flow
    from offline.engine_att_reference import GateCell, gate_episode
    from offline.materialise_draws import draw_config_path
    from offline.method_tier_grid import (
        DECLARED_GRADIENT_STEPS,
        _dt_factory,
        env_settings_for_tiers,
        tier_spec,
    )

    if method not in DECOMP_METHODS:
        raise ValueError(f"{method!r} is not one of {list(DECOMP_METHODS)}")

    spec = tier_spec(tier)
    settings = env_settings_for_tiers([spec], corpus_root)
    horizon = int(settings["max_steps"]) * int(settings["delta_time"])
    base_factory = _dt_factory(
        str(checkpoint), DECLARED_GRADIENT_STEPS, float(spec.target_rtg), device
    )
    ids = [int(draw) for draw in (draws if draws is not None else HELD_OUT_DRAWS)]

    episodes: list[dict[str, Any]] = []
    mismatches = 0
    started = time.perf_counter()
    for draw_id in ids:
        config = Path(
            draw_config_path(nortg_campaign.SCENARIO_KEY, draw_id, out_root=draws_root)
        )
        # a fresh recorder per episode: the digest is a property of ONE episode's decisions
        factory, recorder = recording_factory(base_factory)
        episode = gate_episode(
            cell=GateCell(
                scenario=nortg_campaign.PROBE_SCENARIO,
                tier=tier,
                method=method,
                seed=int(seed),
                draw_id=draw_id,
                role="tier",
            ),
            config_path=config,
            env_settings=settings,
            scenario_id=nortg_campaign.SCENARIO_ID,
            choose_action_factory=factory,
            engine_seed=int(engine_seed),
            created=created_from_flow(config.parent / "flow.json", horizon_seconds=horizon),
        )
        record = episode.as_record()
        reference = committed[(method, tier, int(seed), draw_id)]
        reproduces = bool(
            float(record["att_engine_call"]) == float(reference["att_engine"])
            and float(record["att_ours"]) == float(reference["att_ours"])
        )
        if not reproduces:
            mismatches += 1
        record.update(
            {
                "committed_att_engine": float(reference["att_engine"]),
                "committed_att_ours": float(reference["att_ours"]),
                "reproduces_committed": reproduces,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": str(checkpoint_sha256),
                **recorder.summary(),
            }
        )
        episodes.append(record)
    seconds = time.perf_counter() - started

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "method": method,
        "tier": tier,
        "seed": int(seed),
        "arm": f"{method}@{tier}",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "draws": ids,
        "is_complete": sorted(ids) == sorted(int(d) for d in HELD_OUT_DRAWS),
        "n_mismatches": mismatches,
        "episodes": episodes,
        "seconds": seconds,
        "seconds_per_episode": seconds / len(ids) if ids else 0.0,
    }


# ----------------------------------------------------------------------
# Validation -- all of it, before anything is written
# ----------------------------------------------------------------------


def assert_chunks_complete(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Thirty cells, one hundred draws each, no duplicates.

    *"Compared nothing" must never read as "found no differences"* -- so an incomplete cell set is a
    refusal, not a smaller sample.
    """
    expected = {
        (method, tier, int(seed))
        for method in DECOMP_METHODS
        for tier in nortg_campaign.NORTG_TIERS
        for seed in TRAINING_SEEDS
    }
    seen: list[tuple[str, str, int]] = [
        (str(chunk["method"]), str(chunk["tier"]), int(chunk["seed"])) for chunk in chunks
    ]
    duplicates = sorted({key for key in seen if seen.count(key) > 1})
    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            f"the decomposition needs exactly {len(expected)} cells (2 methods x 3 tiers x 5 "
            f"seeds) and got {len(chunks)}: {len(missing)} missing {missing[:3]}, "
            f"{len(duplicates)} duplicated {duplicates[:3]}, {len(unexpected)} unexpected "
            f"{unexpected[:3]}. 'Compared nothing' must never read as 'found no differences'"
        )

    wanted = sorted(int(draw) for draw in HELD_OUT_DRAWS)
    total = 0
    for chunk in chunks:
        got = sorted(int(row["draw_id"]) for row in chunk["episodes"])
        if got != wanted:
            absent = sorted(set(wanted) - set(got))
            raise ValueError(
                f"{chunk['method']}@{chunk['tier']} seed {chunk['seed']}: {len(got)} draws, not "
                f"{len(wanted)}; an incomplete cell makes the contrast void, first missing "
                f"{absent[:3]}"
            )
        total += len(got)

    return {
        "n_cells": len(chunks),
        "n_episodes": total,
        "n_draws_per_cell": len(wanted),
        "methods": list(DECOMP_METHODS),
        "rule": "2 methods x 3 tiers x 5 seeds x 100 held-out draws, exactly",
    }


def assert_rows_reproduce_committed(
    rows: Sequence[Mapping[str, Any]],
    *,
    committed: Mapping[tuple[str, str, int, int], Mapping[str, float]],
) -> dict[str, Any]:
    """🔒 Every episode reproduces the committed cell under BOTH definitions, under ``==``.

    Four equalities per row, and the last two are what make the first two mean anything:

    1. ``att_engine_call == committed_att_engine``   (the chunk is self-consistent)
    2. ``att_ours == committed_att_ours``
    3. ``committed_att_engine == committed[key]["att_engine"]``  (and the chunk's idea of "committed"
    4. ``committed_att_ours == committed[key]["att_ours"]``       really is the committed value)

    ⭐ Without 3 and 4 a chunk could agree with itself and with nothing else.  The chunk's own
    ``reproduces_committed`` flag is not read.
    """
    failures: list[str] = []
    for row in rows:
        key = (str(row["method"]), str(row["tier"]), int(row["seed"]), int(row["draw_id"]))
        reference = committed.get(key)
        if reference is None:
            raise ValueError(
                f"{key} has no committed cell to reproduce; the reference column does not cover "
                "every episode this artifact would report"
            )
        checks = (
            ("att_engine_call", float(row["att_engine_call"]), float(row["committed_att_engine"])),
            ("att_ours", float(row["att_ours"]), float(row["committed_att_ours"])),
            (
                "committed_att_engine",
                float(row["committed_att_engine"]),
                float(reference["att_engine"]),
            ),
            ("committed_att_ours", float(row["committed_att_ours"]), float(reference["att_ours"])),
        )
        for name, produced, expected in checks:
            if produced != expected:
                failures.append(f"{key} {name}: {produced!r} != {expected!r}")

    if failures:
        raise ValueError(
            f"{len(failures)} of {len(rows) * 4} equalities do not reproduce the committed cell "
            f"under exact float equality; first {failures[:3]}. The observer perturbed the "
            "trajectory, or this harness differs from the campaign's -- and a decomposition of a "
            "DIFFERENT trajectory is not the quantity A13(b) requires"
        )

    return {
        "n_episodes": len(rows),
        "n_reproducing": len(rows),
        "n_equalities": len(rows) * 4,
        "comparison": "exact float equality (==) on both definitions, against the committed "
                      "sources and against the chunk's own record of them",
        "reads_the_chunks_own_flag": False,
    }


def assert_identity_is_exact(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """🔒 The decomposition identity, checked on the STORED terms and by a second route.

    Route one: ``term_population + term_clock_origin + term_cadence == att_ours - att_engine_call``,
    computed from the values as they sit in the file.  Route two: every derived key is compared
    against ``engine_att_reference.episode_from_record(row)``'s recomputation, which reads only the
    constructor fields and ignores the stored derived ones.  A tampered ``term_*`` fails route one; a
    tampered ``att_reference_*`` fails route two.

    Also asserts ``deviation_c1``, ``deviation_c3c`` and ``decomposition_residual`` are exactly
    ``0.0`` -- with the caveat, stated in the module docstring, that the residual is algebraically
    ``deviation_c1`` and so corroborates criterion 1 and nothing further.

    ⚠️ **DISCRIMINATING POWER OF ROUTE ONE, measured by mutation and stated rather than implied**
    (``PROJECT_PLAN`` section 7: *a check must report its discriminating power*).  Deleting route one
    kills no test, and that is not a gap in the tests -- it is analytic.  If route two passes then
    every stored ``term_*`` equals its recomputation, and the recomputed terms telescope to
    ``att_ours - att_reference_engine_population``; the zero-deviation assertion then forces
    ``att_reference_engine_population == att_engine_call``, so the stored terms necessarily sum to
    ``att_ours - att_engine_call``.  **Route one is therefore IMPLIED by route two plus
    ``deviation_c1 == 0``, and no single-field tamper can kill it alone.**  It is kept because it is
    a direct, cheap statement of the identity that does not depend on ``episode_from_record``, and
    because it fires first and gives the clearer message -- not because it adds independent
    evidence.  ⭐ Route TWO is the one with independent power: it catches a moved reconstruction that
    leaves the stored terms summing correctly.
    """
    from offline.engine_att_reference import episode_from_record

    derived = (
        "deviation_c1",
        "deviation_c3c",
        "difference_c3a_running",
        "difference_c3a_population",
        "term_population",
        "term_clock_origin",
        "term_cadence",
        "decomposition_residual",
    )
    # ⚠️ ``difference_ours_minus_engine`` is NOT a GateEpisode property -- ``as_record`` computes it
    # inline (``engine_att_reference.py:907``), so ``getattr`` cannot reach it.  It is recomputed
    # explicitly below rather than dropped, because it is the quantity the whole artifact is about.
    identity_failures: list[str] = []
    route_two_failures: list[str] = []
    zero_failures: list[str] = []
    residual_max = 0.0

    for row in rows:
        key = (str(row["method"]), str(row["tier"]), int(row["seed"]), int(row["draw_id"]))

        # Route one: the STORED terms against the STORED total, exactly.
        stored_sum = (
            float(row["term_population"])
            + float(row["term_clock_origin"])
            + float(row["term_cadence"])
        )
        total = float(row["att_ours"]) - float(row["att_engine_call"])
        if stored_sum != total:
            identity_failures.append(f"{key}: terms sum to {stored_sum!r}, total is {total!r}")

        # Route two: every derived key recomputed from the CONSTRUCTOR fields alone.
        # ``episode_from_record`` ignores the stored derived keys and rebuilds them, so this
        # catches a moved reconstruction that route one cannot see.
        rebuilt = episode_from_record(row)
        for name in derived:
            recomputed = float(getattr(rebuilt, name))
            if float(row[name]) != recomputed:
                route_two_failures.append(
                    f"{key} {name}: stored {row[name]!r} != recomputed {recomputed!r}"
                )
        rebuilt_total = float(rebuilt.att_ours) - float(rebuilt.att_engine_call)
        if float(row["difference_ours_minus_engine"]) != rebuilt_total:
            route_two_failures.append(
                f"{key} difference_ours_minus_engine: stored "
                f"{row['difference_ours_minus_engine']!r} != recomputed {rebuilt_total!r}"
            )

        for name in ("deviation_c1", "deviation_c3c", "decomposition_residual"):
            if float(row[name]) != 0.0:
                zero_failures.append(f"{key} {name} = {row[name]!r}")
        residual_max = max(residual_max, float(row["decomposition_residual"]))

    if identity_failures:
        raise ValueError(
            f"the decomposition identity fails on {len(identity_failures)} of {len(rows)} episodes "
            f"(stored term_* do not sum to att_ours - att_engine_call); first "
            f"{identity_failures[:3]}"
        )
    if route_two_failures:
        raise ValueError(
            f"{len(route_two_failures)} stored derived values disagree with the recomputation from "
            f"their own reconstruction fields; first {route_two_failures[:3]}. A term that sums "
            "correctly but is not what the reconstructions imply is still wrong"
        )
    if zero_failures:
        raise ValueError(
            f"{len(zero_failures)} episodes carry a non-zero deviation or residual; first "
            f"{zero_failures[:3]}. Gate 0 established c1 = c3c = 0.0 exactly, and anything else "
            "means this is not the trajectory the campaign measured"
        )

    return {
        "n_episodes": len(rows),
        "identity": IDENTITY,
        "residual_max": residual_max,
        "routes": "stored terms against their own total, and every derived key recomputed from the "
                  "constructor fields by engine_att_reference.episode_from_record",
        "residual_is_not_independent": (
            "decomposition_residual is algebraically |att_engine_call - "
            "att_reference_engine_population| = deviation_c1, so it corroborates criterion 1 and "
            "nothing further"
        ),
    }


def per_arm_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``_decomposition_summary``'s shape per arm, plus per-seed means.

    The summary itself is ``engine_att_reference._decomposition_summary``, imported and called --
    ``BRIEF_33`` section 3.4 forbids re-implementing it.
    """
    # ⚠️ Imported under its private name from a module this task may NOT edit.  Disclosed in the
    # packet rather than worked around: the only alternative is the re-implementation section 3.4
    # forbids, and a second implementation of a reviewed summary is the worse of the two.
    from offline.engine_att_reference import _decomposition_summary, episode_from_record

    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)

    summaries: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        rebuilt = [episode_from_record(row) for row in arm_rows]
        summary = dict(_decomposition_summary(rebuilt))
        per_seed: dict[str, dict[str, float]] = {}
        for seed in TRAINING_SEEDS:
            seed_rows = [row for row in arm_rows if int(row["seed"]) == int(seed)]
            if not seed_rows:
                continue
            per_seed[str(seed)] = {
                key: float(np.mean([float(row[f"term_{key}"]) for row in seed_rows]))
                for key in TERM_KEYS
            } | {
                "total": float(
                    np.mean(
                        [
                            float(row["att_ours"]) - float(row["att_engine_call"])
                            for row in seed_rows
                        ]
                    )
                )
            }
        summary["per_seed"] = per_seed
        summary["orientation"] = ORIENTATION
        summaries[arm] = summary
    return summaries


def contrast_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per tier, ``dt - dt_nortg`` for each term, paired exactly as ``paired_stats`` pairs.

    Per-draw mean over seeds first, then the mean over the 100 draws -- the registered pairing unit.
    The plain 500-episode mean is reported beside it with the difference between the two averaging
    orders, exactly as ``per_seed_differences`` does, and **no equality is asserted between them**:
    they are the same quantity in exact arithmetic and may differ in the last bits.
    """

    def quantity(row: Mapping[str, Any], key: str) -> float:
        if key == "total":
            return float(row["att_ours"]) - float(row["att_engine_call"])
        return float(row[f"term_{key}"])

    keys = (*TERM_KEYS, "total")
    out: dict[str, Any] = {}
    for tier in nortg_campaign.NORTG_TIERS:
        tier_rows = [row for row in rows if str(row["tier"]) == tier]
        entry: dict[str, Any] = {}
        for key in keys:
            sides: dict[str, dict[int, list[float]]] = {method: {} for method in DECOMP_METHODS}
            for row in tier_rows:
                sides[str(row["method"])].setdefault(int(row["draw_id"]), []).append(
                    quantity(row, key)
                )
            shared = sorted(set(sides["dt"]) & set(sides[nortg_campaign.NORTG_METHOD]))
            if not shared:
                raise ValueError(f"{tier}: the two arms share no draw, so {key} cannot be paired")
            # the registered pairing unit: per-draw mean over seeds, then the mean over draws
            per_draw = [
                float(np.mean(sides["dt"][draw]))
                - float(np.mean(sides[nortg_campaign.NORTG_METHOD][draw]))
                for draw in shared
            ]
            paired_mean = float(np.mean(np.asarray(per_draw, dtype=np.float64)))
            plain = float(
                np.mean([quantity(r, key) for r in tier_rows if str(r["method"]) == "dt"])
            ) - float(
                np.mean(
                    [
                        quantity(r, key)
                        for r in tier_rows
                        if str(r["method"]) == nortg_campaign.NORTG_METHOD
                    ]
                )
            )
            entry[key] = paired_mean
            entry[f"{key}_plain_500_episode_mean"] = plain
            entry[f"{key}_difference_between_the_two_averaging_orders"] = plain - paired_mean
        entry["n_shared_draws"] = len(
            sorted({int(row["draw_id"]) for row in tier_rows})
        )
        entry["sign_convention"] = (
            "dt - dt_nortg for every term, paired per draw (mean over seeds first), matching "
            "paired_stats' registered unit"
        )
        entry["orientation"] = ORIENTATION
        out[tier] = entry
    return out


def assert_contrast_consistent(
    contrast: Mapping[str, Mapping[str, Any]], declared: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """🔒 The measured decomposition explains the difference the campaign already reported.

    Per tier, ``delta_ours - delta_engine`` from ``p5_3b_nortg.json`` must equal
    ``delta_population + delta_clock_origin + delta_cadence`` within :data:`CONTRAST_TOLERANCE`.
    This is the check that ties BL-2(b)'s measurement to the headline it decomposes; without it the
    module could produce three internally consistent terms of some *other* quantity.
    """
    checked: dict[str, Any] = {}
    failures: list[str] = []
    for tier in nortg_campaign.NORTG_TIERS:
        entry = contrast[tier]
        declared_gap = float(declared[tier]["att_ours"]) - float(declared[tier]["att_engine"])
        measured_gap = sum(float(entry[key]) for key in TERM_KEYS)
        gap = declared_gap - measured_gap
        if abs(gap) >= CONTRAST_TOLERANCE:
            failures.append(
                f"{tier}: the campaign reports delta_ours - delta_engine = {declared_gap!r} and "
                f"the measured terms sum to {measured_gap!r}, a gap of {gap!r}"
            )
        checked[tier] = {
            "delta_engine": float(declared[tier]["att_engine"]),
            "delta_ours": float(declared[tier]["att_ours"]),
            "delta_ours_minus_delta_engine": declared_gap,
            "sum_of_term_contrasts": measured_gap,
            "identity_gap": gap,
        }
    if failures:
        raise ValueError(
            "the measured decomposition does not explain the definition difference the campaign "
            f"reports, beyond {CONTRAST_TOLERANCE}: {failures}. Three internally consistent terms "
            "of some other quantity would look exactly like this"
        )
    return {
        "tolerance": CONTRAST_TOLERANCE,
        "why_not_exact": (
            "delta_ours - delta_engine and the sum of the three term contrasts reduce the same "
            "3,000 floats in different summation orders, so == would condemn a correct "
            "implementation. The PER-EPISODE identity is checked exactly by assert_identity_is_exact"
        ),
        "per_tier": checked,
    }


def decomposition_artifact(
    *,
    chunks: Sequence[Mapping[str, Any]],
    committed: Mapping[tuple[str, str, int, int], Mapping[str, float]],
    declared: Mapping[str, Mapping[str, float]],
    provenance: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate everything, then assemble.  **Raises rather than returning a partial artifact.**

    Order is the barrier: :func:`assert_chunks_complete` ->
    :func:`assert_rows_reproduce_committed` -> :func:`assert_identity_is_exact` ->
    :func:`per_arm_summaries` / :func:`contrast_summaries` -> :func:`assert_contrast_consistent`.
    The caller writes only what this returns, so a refusal anywhere leaves the destination untouched.
    """
    completeness = assert_chunks_complete(chunks)
    rows = [dict(row) for chunk in chunks for row in chunk["episodes"]]
    reproduction = assert_rows_reproduce_committed(rows, committed=committed)
    identity = assert_identity_is_exact(rows)
    per_arm = per_arm_summaries(rows)
    contrast = contrast_summaries(rows)
    consistency = assert_contrast_consistent(contrast, declared)
    for tier, record in consistency["per_tier"].items():
        contrast[tier].update(record)

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "role": "PREREGISTRATION A13(b)'s three-component decomposition, MEASURED on all 30 P5.3b "
                "cells under the Gate 0 per-second observer",
        "registered_in": "PREREGISTRATION A13(b); commissioned by BRIEF_33 section 3.4 after "
                         "docs/reviews/P5.3b.md BL-2(b)",
        "orientation": ORIENTATION,
        "orientation_note": ORIENTATION_NOTE,
        "identity": IDENTITY,
        "scenario": nortg_campaign.PROBE_SCENARIO,
        "methods": list(DECOMP_METHODS),
        "tiers": list(nortg_campaign.NORTG_TIERS),
        "seeds": list(TRAINING_SEEDS),
        "held_out_draws": list(HELD_OUT_DRAWS),
        "engine_seed": nortg_campaign.ENGINE_SEED,
        "episodes": rows,
        "summary": {
            "completeness": completeness,
            "reproduction": reproduction,
            "identity": identity,
            "consistency": consistency,
            "per_arm": per_arm,
            "contrast": contrast,
        },
        "provenance": dict(provenance or {}),
        "runtime": dict(runtime or {}),
        "timing": dict(timing or {}),
        "what_this_does_not_say": [
            "It does not say WHY the ablated mix50 arm collapses. The terms are a measurement of "
            "what the two ATT definitions' difference is MADE OF, not of what caused the policy to "
            "behave as it did.",
            "residual_max corroborates criterion 1 and nothing further: decomposition_residual is "
            "algebraically |att_engine_call - att_reference_engine_population| = deviation_c1.",
            "The orientation is att_ours - att_engine, the NEGATION of the one A13(b) writes, so "
            "clock_origin is negative here where A13(b)'s writing would make it positive.",
            "It settles nothing about BL-1: the artifact makes BL-1's guarding test WRITABLE from "
            "a clone by carrying committed_att_engine for all 1,500 dt rows, and that test is not "
            "written here.",
        ],
    }


def diff_json_paths(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    """Every JSON path at which two payloads differ, dotted, sorted.

    Used by the ``diff-paths`` subcommand to prove that regenerating
    ``docs/data/p5_3b_nortg.json`` moved only the keys ``BRIEF_33`` section 3.5 enumerates.  A path
    present in one payload and absent from the other counts as differing.
    """

    def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        if isinstance(node, Mapping):
            for key, value in node.items():
                flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                flat.update(flatten(value, f"{prefix}[{index}]"))
        else:
            flat[prefix] = node
        return flat

    left, right = flatten(baseline), flatten(candidate)
    differing = {key for key in set(left) | set(right) if left.get(key, _ABSENT) != right.get(key, _ABSENT)}
    return sorted(differing)


class _Absent:
    """A sentinel distinct from every JSON value, so ``null`` and *absent* are not confused."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT = _Absent()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``run``, ``report``, ``diff-paths``.

    ``allow_abbrev=False`` deliberately, following ``engine_att_reference``: an abbreviated flag that
    silently resolves to a different option is a class of defect this repo has already paid for.
    """
    parser = argparse.ArgumentParser(
        prog="python -m offline.nortg_decomposition",
        description="P5.3b-fix: A13(b)'s decomposition, measured on all 30 cells.",
        allow_abbrev=False,
    )
    # ⚠️ --corpus-root is NOT required globally, unlike nortg_campaign's: `report` and `diff-paths`
    # roll nothing, and requiring a corpus to read committed JSON would be a path that has to be
    # supplied and is never used.  `run` refuses without it.
    parser.add_argument("--corpus-root", default=None)
    parser.add_argument("--draws-root", default="scenarios/draws")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--engine-seed", type=int, default=nortg_campaign.ENGINE_SEED)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record a chunk from a modified working tree on purpose (AMENDMENT C2's mechanism, "
        "inherited): without it every writing subcommand REFUSES a dirty or undeterminable tree",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="roll ONE cell over the held-out pool under the observer")
    run.add_argument("--method", required=True, choices=list(DECOMP_METHODS))
    run.add_argument("--tier", required=True, choices=list(nortg_campaign.NORTG_TIERS))
    run.add_argument("--seed", required=True, type=int, choices=list(TRAINING_SEEDS))
    run.add_argument("--draws", type=int, nargs="+", default=None)
    sub.add_parser("report", help="assemble docs/data/p5_3b_decomposition.json")
    diff = sub.add_parser("diff-paths", help="which JSON paths differ between two payloads")
    diff.add_argument("--baseline", required=True)
    diff.add_argument("--candidate", required=True)
    diff.add_argument("--allow", nargs="*", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch.  ``main`` validates the work directory through the fence before anything else."""
    args = build_parser().parse_args(argv)
    work = assert_decomposition_writable(
        args.work_dir or (Path(args.output_root) / DEFAULT_WORK_DIRNAME)
    )
    out_dir = Path(args.out_dir)

    if args.command == "diff-paths":
        return _run_diff_paths(args)
    if args.command == "report":
        return _run_report(args, work, out_dir)

    from offline.offline_baselines import pin_torch_threads

    pin_torch_threads(args.torch_threads)
    return _run_cell_command(args, work, out_dir)


def _run_cell_command(args: argparse.Namespace, work: Path, out_dir: Path) -> int:
    """One cell.  Skips only a chunk that is COMPLETE and CLEAN; otherwise re-runs and overwrites."""
    from offline.dt_gate import runtime_provenance
    from offline.method_tier_grid import file_sha256

    if args.corpus_root is None:
        raise ValueError("`run` needs --corpus-root; it builds env settings from the tier manifest")

    destination = assert_decomposition_writable(
        work / chunk_name(args.method, args.tier, args.seed)
    )
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if chunk_is_reusable(existing):
            print(f"{destination.name}: complete and clean, skipping", flush=True)
            return 0
        print(f"{destination.name}: incomplete or mismatching, re-running", flush=True)

    tree = nortg_campaign.assert_recordable_tree(args.allow_dirty)
    root = Path(args.output_root)
    if args.method == nortg_campaign.NORTG_METHOD:
        assert_nortg_checkpoint_identity(output_root=root)
        checkpoint = root / NORTG_CHECKPOINT_TEMPLATE.format(tier=args.tier, seed=args.seed)
    else:
        nortg_campaign.assert_reused_dt_identity(data_dir=out_dir, output_root=root)
        checkpoint = root / nortg_campaign.TIER_CHECKPOINT_TEMPLATE[args.tier].format(
            seed=args.seed
        )

    committed = committed_reference_rows(
        data_dir=out_dir, rederivation_dir=nortg_campaign.default_rederivation_dir(root)
    )
    chunk = run_cell(
        args.method,
        args.tier,
        args.seed,
        checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        committed=committed,
        corpus_root=args.corpus_root,
        draws_root=args.draws_root,
        engine_seed=args.engine_seed,
        device=args.device,
        draws=args.draws,
    )
    chunk["tree"] = tree
    chunk["runtime"] = runtime_provenance()

    work.mkdir(parents=True, exist_ok=True)
    nortg_campaign.write_json_atomic(chunk, destination)
    print(
        f"{chunk['arm']} seed {chunk['seed']}: {len(chunk['episodes'])} episodes, "
        f"{chunk['n_mismatches']} mismatches, {chunk['seconds_per_episode']:.3f} s/episode",
        flush=True,
    )
    # non-zero on a mismatch so the driver stops on the first bad cell rather than an hour later;
    # the chunk is written first, so the evidence survives, and chunk_is_reusable refuses to skip it
    return 1 if chunk["n_mismatches"] else 0


def _run_report(args: argparse.Namespace, work: Path, out_dir: Path) -> int:
    """Read every chunk, validate all of it, then write.  Validation precedes the write."""
    from offline.dt_gate import runtime_provenance
    from offline.method_tier_grid import file_sha256, measurement_commits

    chunks: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for method in DECOMP_METHODS:
        for tier in nortg_campaign.NORTG_TIERS:
            for seed in TRAINING_SEEDS:
                path = work / chunk_name(method, tier, seed)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{path}: this report needs every cell chunk; run `run` for it first"
                    )
                chunks.append(json.loads(path.read_text(encoding="utf-8")))
                provenance.append({"path": str(path), "sha256": file_sha256(path)})

    root = Path(args.output_root)
    committed = committed_reference_rows(
        data_dir=out_dir, rederivation_dir=nortg_campaign.default_rederivation_dir(root)
    )
    payload = decomposition_artifact(
        chunks=chunks,
        committed=committed,
        declared=declared_contrasts(out_dir),
        provenance={
            "chunks": provenance,
            "committed_dt_nortg": str(Path(out_dir) / "p5_3b_nortg.json"),
            "committed_dt": str(nortg_campaign.default_rederivation_dir(root)),
        },
        runtime=runtime_provenance(measurement_commits(chunks)),
        timing={
            "per_cell_seconds": {
                f"{chunk['arm']}@seed{chunk['seed']}": float(chunk["seconds"]) for chunk in chunks
            },
            "concurrent_load": _CONCURRENT_LOAD_NOTE,
        },
    )
    nortg_campaign.write_json_atomic(payload, out_dir / "p5_3b_decomposition.json")
    print(
        f"wrote {out_dir / 'p5_3b_decomposition.json'}: {len(payload['episodes'])} episodes, "
        f"{payload['summary']['reproduction']['n_reproducing']} reproducing",
        flush=True,
    )
    return 0


def _run_diff_paths(args: argparse.Namespace) -> int:
    """Print every differing JSON path, and refuse any that is not in ``--allow``."""
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    differing = diff_json_paths(baseline, candidate)
    allowed = tuple(args.allow)
    unexpected = [
        path for path in differing if not any(path == a or path.startswith(a) for a in allowed)
    ]
    for path in differing:
        marker = "  " if path not in unexpected else "! "
        print(f"{marker}{path}")
    print(f"{len(differing)} differing paths, {len(unexpected)} outside the enumerated set")
    if unexpected:
        print(
            "REFUSED: the paths marked ! are not in the enumerated set. Each one is a FINDING, "
            "not something to add to the list",
            flush=True,
        )
        return 1
    return 0


#: ``PROJECT_PLAN`` section 7 (:1577): *a timing must also record the concurrent load*.  The
#: 2026-09-10 Decisions Log makes both halves binding -- the machine stays quiet, AND the record
#: says what else ran, because (a) cannot be applied retroactively.  A wall clock carries no
#: signature of the load it was taken under.
_CONCURRENT_LOAD_NOTE = (
    "Recorded per PROJECT_PLAN section 7: this campaign ran 5 cells concurrently, one torch thread "
    "each, on one GPU. Any other load during the run is stated in docs/returns/P5.3b-fix.md; a "
    "wall clock carries no signature of the load it was taken under."
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
