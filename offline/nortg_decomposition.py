"""P5.3b BL-2(b): A13(b)'s three-component decomposition, MEASURED on the campaign's own arms.

⚠️ SKELETON.  Every function below raises :class:`NotImplementedError`; the constants are real.
Tests are written against this surface first, so each one fails for its own reason rather than
sharing a single import error.

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
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError

    def record(self, action: Any) -> None:
        """Store a copy of one decision."""
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        """``n_decisions``, ``action_counts`` (per intersection, per action index) and the digest."""
        raise NotImplementedError


def recording_factory(
    inner: Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]],
) -> tuple[Callable[[Any], Callable[[Any, dict[str, Any]], np.ndarray]], ActionRecorder]:
    """Wrap a ``choose_action`` factory so every decision is recorded and none is altered.

    Returns ``(factory, recorder)``.  The wrapped factory binds the recorder's histogram width from
    the env it is handed, then returns a callable that defers to the inner one, records a copy and
    **returns the inner callable's object itself** -- not a copy, not a cast.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def declared_contrasts(data_dir: str | Path) -> dict[str, dict[str, float]]:
    """Per tier, the committed ``mean_difference`` under each definition.

    Read from ``docs/data/p5_3b_nortg.json``'s ``comparisons.<tier>.by_definition``.  The sign
    convention is ``paired_stats``': ``mean(ATT_dt - ATT_dt_nortg)``, left = ``dt``.
    """
    raise NotImplementedError


def assert_nortg_checkpoint_identity(*, output_root: str | Path) -> dict[str, Any]:
    """The fifteen ablated checkpoints are the ones ``SHA256SUMS_p5_3b.txt`` recorded.

    Checked AT CONSUMPTION, in this process, before a single episode is rolled.  Refuses a
    checkpoint absent from the manifest as loudly as one whose digest moved: a file outside its own
    campaign manifest has no integrity record at all (``DEFERRED`` 56's lesson).
    """
    raise NotImplementedError


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
    raise NotImplementedError


def chunk_is_reusable(payload: Mapping[str, Any]) -> bool:
    """May a restart SKIP this cell?  Only if the chunk is complete AND clean.

    Complete = every one of ``HELD_OUT_DRAWS`` present; clean = ``n_mismatches == 0``; and the
    format version must match.  ⚠️ ``offline/campaigns/p5_3b.sh`` skips on ``[ -f ]`` alone, and
    ``assert_probe_cell_is_ablated``'s docstring records what that cost: *"the driver skips a tier
    whose probe chunk exists, so a bad chunk survived every restart."*  A partial or mismatching
    chunk is re-run and overwritten.
    """
    raise NotImplementedError


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
    raise NotImplementedError


# ----------------------------------------------------------------------
# Validation -- all of it, before anything is written
# ----------------------------------------------------------------------


def assert_chunks_complete(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Thirty cells, one hundred draws each, no duplicates.

    *"Compared nothing" must never read as "found no differences"* -- so an incomplete cell set is a
    refusal, not a smaller sample.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    """
    raise NotImplementedError


def per_arm_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``_decomposition_summary``'s shape per arm, plus per-seed means.

    The summary itself is ``engine_att_reference._decomposition_summary``, imported and called --
    ``BRIEF_33`` section 3.4 forbids re-implementing it.
    """
    raise NotImplementedError


def contrast_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per tier, ``dt - dt_nortg`` for each term, paired exactly as ``paired_stats`` pairs.

    Per-draw mean over seeds first, then the mean over the 100 draws -- the registered pairing unit.
    The plain 500-episode mean is reported beside it with the difference between the two averaging
    orders, exactly as ``per_seed_differences`` does, and **no equality is asserted between them**:
    they are the same quantity in exact arithmetic and may differ in the last bits.
    """
    raise NotImplementedError


def assert_contrast_consistent(
    contrast: Mapping[str, Mapping[str, Any]], declared: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """🔒 The measured decomposition explains the difference the campaign already reported.

    Per tier, ``delta_ours - delta_engine`` from ``p5_3b_nortg.json`` must equal
    ``delta_population + delta_clock_origin + delta_cadence`` within :data:`CONTRAST_TOLERANCE`.
    This is the check that ties BL-2(b)'s measurement to the headline it decomposes; without it the
    module could produce three internally consistent terms of some *other* quantity.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def diff_json_paths(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    """Every JSON path at which two payloads differ, dotted, sorted.

    Used by the ``diff-paths`` subcommand to prove that regenerating
    ``docs/data/p5_3b_nortg.json`` moved only the keys ``BRIEF_33`` section 3.5 enumerates.  A path
    present in one payload and absent from the other counts as differing.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``run``, ``report``, ``diff-paths``.

    ``allow_abbrev=False`` deliberately, following ``engine_att_reference``: an abbreviated flag that
    silently resolves to a different option is a class of defect this repo has already paid for.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch.  ``main`` validates the work directory through the fence before anything else."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
