"""P7.3a: the zero-shot point of the C3 transfer curve, and the artifact that reports it.

Artifact format version: ``p7.3a-zero-shot/1.0`` -- ``docs/data/p7_3a_zero_shot.json``, written by
:func:`report` from the per-cell chunks under ``output/p7_3a/``.  Amendment B2's stage-1 artifact
is ``docs/data/p7_3a_zero_shot_stage1.json``, the same block layout over the confirmatory cells.

WHAT THIS MODULE EXECUTES, AND WHAT IT MAY NOT DECIDE
------------------------------------------------------
Everything scientific is registered and READ, never chosen here: **H3** and its test row
(``PREREGISTRATION`` §2), **rho** (§3.4), **A15** (the definition pair, the counts every cell
carries, the teleport-free regime, the admitted pair), **A16** (the door), **A17** (Rule B with
``S = mean`` at k = 100 as the registered prompt, the two subjects, the exploratory contrast, the
consistency gate) and **A18(c)** (the held-out seed rule).  The four arms were closed by
``BRIEF_37`` §2 before any cell ran, and the two stages by Amendment B1.

RHO, AND WHY THERE ARE TWO NUMBERS AND ONE FORMULA
---------------------------------------------------
``PREREGISTRATION`` §3.4 fixes ONE formula, computed **within** a backend::

    rho = (ATT_fixedtime - ATT_policy) / (ATT_fixedtime - ATT_maxpressure)

fixed-time is 0 and MaxPressure is 1 **by construction**, and values outside [0, 1] are expected and
are never clipped.  A15(a) then observes that §3.4 never fixed *which* ATT definition each backend
computes it on, and A15(b) requires **both** on every SUMO cell.  So this module reports rho twice
-- once on ``e_sumo`` (the pool-clock, all-created twin, A15's primary) and once on ``att_env`` (the
admitted pair beside it) -- and that, not two formulas, is what ``BRIEF_37`` §3.5's *"both
definitions"* means (Amendment A1).

⚠️ ``e_sumo`` AND ``att_reference_created_population`` ARE ONE QUANTITY UNDER TWO NAMES
(Amendment C6): :func:`offline.sumo_att_reference.reconstruct_sumo_episode` calls it ``e_sumo`` and
the artifact key it is written to in P7.1's freeze is ``att_reference_created_population``.  A15's
text is correct and is not amended; this sentence is the alias, recorded once.

THE DOOR IS FOR THE DT, NOT FOR THE ANCHORS (Amendment A2)
-----------------------------------------------------------
A DT cell runs on an **observed, aligned** env; an anchor cell runs on an **observed, unwrapped**
one.  ``align_info`` drops outgoing lanes and re-keys the survivors to CityFlow ids, while
MaxPressure's pressure is a difference over the env's own SUMO lane ids -- wrapping it raises
``KeyError`` on an outgoing lane, measured before it was written down.  A16 is untouched: the door
is the only route into *a CityFlow-trained model's* frame, and an anchor has no frame to enter.

THE FENCE IS LIFTED HERE, FOR THESE ARMS, AND FOR NOTHING ELSE
----------------------------------------------------------------
P7.2b fenced ``att_horizon``, ``episode_reward``, ``rtg_last`` and the RTG series because A3 said
*until P7.3's brief is written*.  It is written: ``BRIEF_37`` §2 lifts the fence for the four
declared arms and the three anchors on the **held-out pool**, and for nothing else.  P7.2b's smoke
on draw 5 stays fenced.  :func:`report` refuses any cell whose arm is not declared.

P7.3d'S GRID4X4 CHUNK -- format ``p7.3d-grid4x4/1.0`` (``BRIEF_39``, B.6 fix round)
------------------------------------------------------------------------------------
A cell whose ``scenario`` is ``cityflow_grid4x4`` is written by :func:`run_cell` as its own format
(:data:`GRID4X4_ARTIFACT_FORMAT_VERSION`); an hz1x1 chunk -- no ``scenario`` key -- is exactly
what it always was.  The grid4x4 chunk carries every episode-level field of the hz1x1 chunk, plus
``scenario``, ``intersection_ids`` (the ENV's order, contract C1) and ``actions``, and every
per-intersection quantity as a mapping keyed by id: ``target_rtg``, ``rtg_first``, ``rtg_last``,
``rtg_series``, ``reward_series``, ``rtg_advanced_every_decision``, ``n_decisions_in_support``,
``support_range`` and ``in_support_counts``.  ``calibration_sha256`` is
:data:`P7_3D_CALIBRATION_SHA256`; ``checkpoint_sha256`` is checked against A20(a)'s pin.
Every grid4x4 chunk -- anchor and DT alike (B.7.1-2) -- also records ``local_return`` and
``local_return_from_lanes``: intersection *i*'s episode return under the collection reward, the sum
over the 360 POST-STEP infos of its reward, by the probe's two routes and equal under ``==``.  It
is the quantity per-intersection rho is defined on, and it is NOT ``sum(reward_series[i])``, which
lacks the last decision's reward (the series is read before each act, below).

**Alignment convention.**  ``actions[t][j]`` is the action decision ``t`` applied at
``intersection_ids[j]``, ``t = 0 .. decisions - 1``.  ``rtg_series[i][t]`` is intersection
``i``'s return-to-go read BEFORE the agent acts on ``info_t``, and ``reward_series[i][t]`` is the
reward ``info_t`` carries for ``i`` -- the reset info's at ``t = 0``, the post-step info of
decision ``t - 1`` after that.  So Amendment D1's shift-by-one holds per intersection,
``rtg[i][t] - rtg[i][t-1] == -reward[i][t-1]`` for ``t >= 2``, while at ``t = 1`` the agent forces
the step-0 reward to zero (``transfer_calibration.rtg_advanced_every_decision``'s rule; the reset
info carries ``-0.0`` on every real episode).  ``rtg[i][0] == target_rtg[i]`` is the per-id
refusal :func:`assert_rtg_first_matches_targets` applies, and ``episode_reward`` is NOT the sum of
any reward series (it is the env's scalar, as on hz1x1).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ANCHOR_ARMS",
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_ARMS",
    "HALTING_CHECK_DRAW",
    "HELD_OUT_DRAWS",
    "P7_2B_CALIBRATION_SHA256",
    "RANDOM_POLICY_SEEDS",
    "STAGE_CONFIRMATORY",
    "STAGE_GRID4X4",
    "grid4x4_cells",
    "STAGES",
    "SUBJECTS",
    "TRAINING_SEEDS",
    "ArmSpec",
    "anchor_choose",
    "artifact_name_for_stage",
    "assert_env_matches_cell",
    "att_env_from_info",
    "build_parser",
    "cell_chunk_name",
    "check_campaign_inputs",
    "checkpoint_identity",
    "checkpoint_identity_for",
    "chunk_is_reusable",
    "chunk_path",
    "compare_reroll_payloads",
    "declared_cells",
    "demand_identity",
    "demand_identity_for",
    "dt_choose",
    "dt_reroll_check_cell",
    "env_for_cell",
    "grid4x4_checkpoint_identity",
    "grid4x4_support_ranges",
    "halting_check_for",
    "load_calibration",
    "main",
    "pilot_cells",
    "report",
    "run_dt_reroll_check",
    "run_pilot",
    "run_stage",
    "reusable_chunk_at",
    "rho",
    "run_cell",
    "scenario_of",
    "targets_for_subject",
    "unresolvable_chunk_commits",
    "validate_cell_payload",
    "write_chunk",
    "write_manifest",
]

ARTIFACT_FORMAT_VERSION = "p7.3a-zero-shot/1.0"

#: A15(g)'s admitted pair; the only scenario this task evaluates.
SCENARIO_KEY = "cityflow1x1"

#: A18(c): every held-out episode is one per draw on a fresh env at ``reset(seed=1000)``.
HELD_OUT_DRAWS: tuple[int, ...] = tuple(range(1000, 1100))
ENGINE_SEED = 1000

#: A17(c)'s two zero-shot subjects, and P4's five training seeds.
SUBJECTS: tuple[str, ...] = ("mappo1000", "mix50")
TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: The anchors rho is defined against (§3.4), plus ``random`` which the C1 ladder normalises on.
#: ``fixedtime`` and ``maxpressure`` are one episode per draw; ``random`` is five POLICY seeds at
#: the same engine seed (A18(c)), because a single random episode is a draw from a distribution
#: rather than an anchor.
ANCHOR_ARMS: tuple[str, ...] = ("fixedtime", "maxpressure", "random")
RANDOM_POLICY_SEEDS: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)

#: Amendment C2: the recorder's halting cross-check is value-neutral (A9/A9b reproduce P7.1's
#: frozen ``att_env`` and ``e_sumo`` bit-for-bit with it ON and OFF) and costs 3.5x, so it runs on
#: a DECLARED SUBSET -- every cell on this draw -- and nowhere else.  47 cells of 4,700.
HALTING_CHECK_DRAW = 1000


class ArmSpec:
    """One declared arm: how its target is found in P7.2b's artifact, and what it is for.

    ``BRIEF_37`` §2 closed this set **before any cell ran**, which is what A17(d) requires of a
    reduction: Rule B's k = 5 and k = 20 targets are the few-shot prompts A18(d) attaches to
    fine-tuned models and are deliberately NOT evaluated zero-shot.
    """

    __slots__ = ("name", "rule", "statistic", "k", "role")

    def __init__(self, name: str, rule: str, statistic: str, k: int | None, role: str) -> None:
        self.name = name
        self.rule = rule
        self.statistic = statistic
        self.k = k
        self.role = role

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ArmSpec({self.name!r}, role={self.role!r})"


#: The four, in the order the packet reports them.  ``role`` is checked against the artifact's own
#: ``role`` field rather than assumed: the registered prompt is whichever row P7.2b marked, and a
#: disagreement between that mark and this table is a refusal, not a preference.
DECLARED_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("b_mean_k100", "rule_b", "mean", 100, "registered_prompt"),
    ArmSpec("b_max_k100", "rule_b", "max", 100, "ablation"),
    ArmSpec("a_q1.0", "rule_a", "q1.0", 100, "ablation"),
    ArmSpec("naive", "naive", "none", None, "ablation"),
)

#: Amendment B1's two declared stages.  **Stage 2 is UNCONDITIONAL**: it runs whatever stage 1
#: shows, exactly as P7.3b runs whatever the zero-shot number is (§7).  This is a sequence, not a
#: cut, and nothing registered moves between them.
STAGE_CONFIRMATORY = "confirmatory"

#: P7.3b section 3.4's third stage.  ⚠️ **It is NOT part of ``declared_cells(None)``**, and that
#: asymmetry is Amendment A1, required rather than tolerated: ``declared_cells(None)`` keeps
#: returning P7.3a's 4,700 and its two stages keep partitioning that set element for element, so
#: adding the anchor cannot disturb anything already published.  The anchor's 700 cells are
#: reachable only through this stage BY NAME, and they live in their own work directory.
STAGE_ANCHOR = "anchor"
STAGES: tuple[str, ...] = (STAGE_CONFIRMATORY, "rest", STAGE_ANCHOR)

#: A18(a)'s full-retrain anchor, as a SUBJECT.  Deliberately **not** added to
#: ``transfer_calibration.SUBJECTS``: that table means *A17(c)'s two CityFlow-trained subjects*,
#: it is what ``_h3_block``, ``_contrast_block`` and ``_in_support_block`` are written against,
#: and ``targets_for_subject`` would raise for a subject P7.2b's artifact never registered.
ANCHOR_SUBJECT = "anchor_k200"

#: Its one arm.  A18(a)'s prompt is the naive in-domain rule over the anchor's OWN 200 episodes --
#: the maximum episode return and the largest absolute RTG -- and NOT any row of P7.2b's
#: calibration artifact, which registers CityFlow-trained subjects only.
ANCHOR_ARM = "naive_in_domain"

ANCHOR_ROLE = "anchor_k200_endpoint"

#: The committed digest record the anchor is pinned against, and its digest.  It exists because
#: ``CHECKPOINT_RECORD`` maps the two CityFlow subjects to ``p4_gate.json`` and
#: ``p4_7_training.json``, and neither will ever name an anchor checkpoint (P7.3a packet section
#: 19.14).  Same shape as :data:`P7_2B_CALIBRATION_SHA256`: a DECLARATION that these numbers came
#: from THAT file, which moves only in a commit that also moves the file.
P7_3B_TRAINING_NAME = "p7_3b_anchor_training.json"
P7_3B_TRAINING_SHA256 = "edca65f96f060d77a8ef9620deaa2c4a76c4c77eae0f7a9e05c47587a06dd62a"

#: Where the anchor's five checkpoints live under ``output/``.
ANCHOR_CHECKPOINT_SUBDIR = "p7_3b_anchor/checkpoints"
ANCHOR_CHECKPOINT_STEM = "anchor_dt_seed"


def targets_for_subject(
    subject: str, artifact: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The four declared arms' targets for *subject*, READ from P7.2b's artifact.

    Looked up by ``(rule, statistic, k)`` and cross-checked against the artifact's own ``role``:
    ``b_mean_k100`` must be the row P7.2b marked ``registered_prompt`` and the other three must be
    ``ablation``.  A table that disagreed with the artifact would silently re-register the prompt,
    which is the one thing A17 does not permit this task to do.
    """
    rows = artifact["targets"][subject]
    out: dict[str, dict[str, Any]] = {}
    for spec in DECLARED_ARMS:
        matching = [
            row
            for row in rows
            if row["rule"] == spec.rule
            and row["statistic"] == spec.statistic
            and row["k"] == spec.k
        ]
        if len(matching) != 1:
            raise ValueError(
                f"{subject}: {len(matching)} rows match arm {spec.name} "
                f"({spec.rule}, {spec.statistic}, k={spec.k}); exactly one is required"
            )
        row = matching[0]
        if row["role"] != spec.role:
            raise ValueError(
                f"{subject}: arm {spec.name} is declared {spec.role!r} but P7.2b's artifact marks "
                f"that row {row['role']!r}. The artifact is the registration; refusing to evaluate "
                "a prompt under a role this task assigned it"
            )
        out[spec.name] = {
            "target_rtg": float(row["target_rtg"]),
            "rule": spec.rule,
            "statistic": spec.statistic,
            "k": spec.k,
            "role": spec.role,
            "in_support": row["in_support"],
        }
    registered = [name for name, row in out.items() if row["role"] == "registered_prompt"]
    if registered != ["b_mean_k100"]:
        raise ValueError(
            f"{subject}: the registered prompt resolved to {registered}, not ['b_mean_k100']"
        )
    return out


#: P7.3d's ONE stage (A21(a)): `b_mean_k100` x 5 seeds x 100 draws + the two anchors = 700 cells.
STAGE_GRID4X4 = "grid4x4_confirmatory"


def grid4x4_cells() -> list[dict[str, Any]]:
    """A21(a)'s 700 cells: one subject, one arm, five seeds, the two anchors, 100 held-out draws.

    A SEPARATE declaration from :func:`declared_cells`, for Amendment A1's reason on P7.3b: the
    hangzhou declaration -- and therefore every count P7.3a's artifact rests on -- is exactly what
    it was before this scenario existed.  `naive` and `random` are NOT here: A21(b) removed them
    by declaration before any grid4x4 SUMO number existed, and their absence is scope, not an
    omission (the paper says so in A21's own words).
    """
    cells: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        for draw in HELD_OUT_DRAWS:
            cells.append(
                {
                    "kind": "dt",
                    "subject": GRID4X4_SUBJECT,
                    "arm": GRID4X4_ARM,
                    "seed": int(seed),
                    "draw_id": int(draw),
                    "scenario": GRID4X4_SCENARIO_KEY,
                    "stage": STAGE_GRID4X4,
                }
            )
    for arm in ("fixedtime", "maxpressure"):
        for draw in HELD_OUT_DRAWS:
            cells.append(
                {
                    "kind": "anchor",
                    "subject": None,
                    "arm": arm,
                    "seed": None,
                    "draw_id": int(draw),
                    "scenario": GRID4X4_SCENARIO_KEY,
                    "stage": STAGE_GRID4X4,
                }
            )
    return cells


def declared_cells(stage: str | None = None) -> list[dict[str, Any]]:
    """Every cell the campaign runs, as declared -- optionally only one stage's.

    A cell is ``(kind, subject, arm, seed, draw)``.  ``kind`` is ``"dt"`` or ``"anchor"`` and it is
    what decides whether the env goes through A16's door (Amendment A2), so it is part of the
    cell's identity rather than something inferred later from the arm's name.

    Amendment B1's stages:

    * **confirmatory** -- ``b_mean_k100`` x 2 subjects x 5 seeds x 100 draws (1,000), plus
      ``fixedtime`` and ``maxpressure`` on the same draws (200): the arm H3 tests and the two
      anchors §3.4's rho is defined against. 1,200 cells.
    * **rest** -- the three ablation arms x 2 subjects x 5 seeds (3,000) and ``random`` x 5 policy
      seeds (500). 3,500 cells.
    """
    if stage == STAGE_GRID4X4:
        return grid4x4_cells()
    if stage is not None and stage not in STAGES:
        raise ValueError(f"{stage!r} is not one of {list(STAGES) + [STAGE_GRID4X4]}")
    # Amendment A1: the anchor stage is a SEPARATE declaration, so everything below -- and
    # therefore declared_cells(None) -- is exactly what it was before P7.3b existed.
    if stage == STAGE_ANCHOR:
        return anchor_cells()

    cells: list[dict[str, Any]] = []
    for spec in DECLARED_ARMS:
        in_stage1 = spec.name == "b_mean_k100"
        for subject in SUBJECTS:
            for seed in TRAINING_SEEDS:
                for draw in HELD_OUT_DRAWS:
                    cells.append(
                        {
                            "kind": "dt",
                            "subject": subject,
                            "arm": spec.name,
                            "seed": int(seed),
                            "draw_id": int(draw),
                            "stage": STAGE_CONFIRMATORY if in_stage1 else "rest",
                        }
                    )
    for arm in ANCHOR_ARMS:
        seeds = RANDOM_POLICY_SEEDS if arm == "random" else (None,)
        for seed in seeds:
            for draw in HELD_OUT_DRAWS:
                cells.append(
                    {
                        "kind": "anchor",
                        "subject": None,
                        "arm": arm,
                        "seed": None if seed is None else int(seed),
                        "draw_id": int(draw),
                        "stage": "rest" if arm == "random" else STAGE_CONFIRMATORY,
                    }
                )
    if stage is not None:
        cells = [cell for cell in cells if cell["stage"] == stage]
    return cells


def anchor_cells() -> list[dict[str, Any]]:
    """P7.3b section 3.4's 700 cells: the anchor on the held-out pool, and rho's two denominators.

    * **500** ``dt`` cells -- :data:`ANCHOR_SUBJECT` x five training seeds x the 100 held-out
      draws, one episode per draw on a fresh env at ``reset(seed=1000)`` (A18(c));
    * **200** anchor cells -- ``fixedtime`` and ``maxpressure`` on the same draws, **re-rolled**.
      They are re-rolled rather than reused because P7.3a's chunks stopped being reusable at
      ``d92947e``, and because rho's denominator must come from the same code as its numerator.

    ``random`` is **not** re-rolled (Amendment A4): rho's denominator is fixed-time and MaxPressure
    only, and P7.3a's 500 ``random`` cells stay where they are.  The artifact says so, so no reader
    infers a ``random`` comparison this task did not run.

    ``kind`` stays ``"dt"`` for the anchor's own cells, deliberately: that is what makes
    :func:`assert_env_matches_cell` require ``AlignedEnv`` (Amendment A2 -- the anchor was TRAINED
    in the canonical frame, so it must be evaluated in it), what keeps
    :func:`validate_cell_payload`'s ``rtg_first == target_rtg`` refusal in force, and what routes
    the cell through :func:`dt_choose`'s ``act(info, explore=False, update_memory=True)``
    (``BRIEF_36`` E4).  A new ``kind`` would have walked around all three.
    """
    cells: list[dict[str, Any]] = [
        {
            "kind": "dt",
            "subject": ANCHOR_SUBJECT,
            "arm": ANCHOR_ARM,
            "seed": int(seed),
            "draw_id": int(draw),
            "stage": STAGE_ANCHOR,
        }
        for seed in TRAINING_SEEDS
        for draw in HELD_OUT_DRAWS
    ]
    cells.extend(
        {
            "kind": "anchor",
            "subject": None,
            "arm": arm,
            "seed": None,
            "draw_id": int(draw),
            "stage": STAGE_ANCHOR,
        }
        for arm in ("fixedtime", "maxpressure")
        for draw in HELD_OUT_DRAWS
    )
    return cells


def cell_chunk_name(cell: Mapping[str, Any]) -> str:
    """A chunk's file name; one cell, one file, and the name carries the whole identity."""
    subject = cell["subject"] or "anchor"
    seed = "none" if cell["seed"] is None else int(cell["seed"])
    stem = f"cell_{subject}_{cell['arm']}_seed{seed}_draw{int(cell['draw_id']):04d}.json"
    # P7.3b's stage-identity lesson, one level up: two campaigns that share a naming scheme will
    # eventually share a directory.  The scenario is prefixed for every scenario EXCEPT the
    # default -- hz1x1's 5,404 chunks on disk are keyed by the old names and may not move.
    scenario = str(cell.get("scenario") or SCENARIO_KEY)
    if scenario == SCENARIO_KEY:
        return stem
    return f"cell_{scenario}_{stem[len('cell_'):]}"


def rho(att_arm: float, att_fixedtime: float, att_maxpressure: float) -> float:
    """``PREREGISTRATION`` §3.4, verbatim: the ONE registered formula, within one backend.

    ``rho = (ATT_fixedtime - ATT_arm) / (ATT_fixedtime - ATT_maxpressure)``

    fixed-time is 0 and MaxPressure is 1 **by construction** -- substituting either anchor for
    ``att_arm`` gives exactly that, which is what T7 asserts rather than approximates.  §3.4:
    *"Values may exceed 100 or fall below 0; that is expected and is not clipped."*  Nothing here
    clips, and a caller that wanted to would be changing a registered definition.

    The anchors must be **the same draw's**.  Pairing is per draw because the demand differs by
    draw and a ratio built from another draw's denominator is not a normalisation of anything; the
    caller is responsible for that pairing and :func:`report` refuses when it is broken.
    """
    denominator = float(att_fixedtime) - float(att_maxpressure)
    if denominator == 0.0:
        raise ValueError(
            "the two anchors have equal ATT on this draw, so rho's denominator is zero and the "
            "normalisation is undefined; reporting it as any finite number would be an invention"
        )
    return (float(att_fixedtime) - float(att_arm)) / denominator


# ======================================================================================
# Section 3.5b -- the pins, the cell runner, the chunks and the artifact
# (Amendments E2, E3(a)-(f), G1-G4)
# ======================================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Roots are PARAMETERS with today's paths as defaults (``BRIEF_37`` §0.9, ``DEFERRED`` 82): this
#: module adds no new hardcoded absolute path to a call site, and a test points them elsewhere.
DEFAULT_DRAWS_ROOT = Path("/home/filip/rltraffic/scenarios/draws")
DEFAULT_OUTPUT_ROOT = Path("/home/filip/rltraffic/output")
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_ROOT / "p7_3a"
DEFAULT_DATA_DIR = _REPO_ROOT / "docs" / "data"

#: E3(a): A17's targets are READ from P7.2b's artifact, and the artifact is pinned by digest.  The
#: pin is a DECLARATION -- *these targets came from THAT file* -- and the file is the evidence; a
#: target read from an unpinned artifact can be edited between the calibration and the campaign
#: without leaving a trace.  It moves only in a commit that also moves the artifact.
#: P7.3d's artifact version.  grid4x4 gets its OWN, rather than overloading P7.3a's: one string
#: describing two artifacts is how a reader comes to believe a cell set is something it is not.
GRID4X4_ARTIFACT_FORMAT_VERSION = "p7.3d-grid4x4/1.0"

#: The grid4x4 scenario key, its registered subject (A20(a)) and its one registered arm (A21(a)).
GRID4X4_SCENARIO_KEY = "cityflow_grid4x4"
GRID4X4_SUBJECT = "mappo1000_dt_nomix_h4"
GRID4X4_ARM = "b_mean_k100"

#: A21(b)'s two scope items, VERBATIM from ``PREREGISTRATION.md`` (the enumerators ``(i)`` and
#: ``(ii)`` omitted): the paper states both as SCOPE, in these words, and the grid4x4 artifact carries
#: them first under ``what_this_does_not_say``.  The non-ASCII characters (the minus sign U+2212, rho
#: U+03C1, the em dash U+2014) are escaped so this source stays ASCII; a test extracts both items
#: from the registration and compares them character for character.
A21_SCOPE_SENTENCES: tuple[str, str] = (
    "The `naive` arm is not evaluated on grid4x4, so the calibrated-versus-naive contrast (A17(a); "
    "P7.3a's exploratory contrast) stays hz1x1-only, where it was flat: −0.0021 and −0.0179 "
    "on `E_sumo` for the two subjects. The paper reports that contrast as a single-intersection "
    "result and says grid4x4 did not test it.",
    "The `random` anchor is not evaluated on grid4x4, so ρ_random — the C1 ladder's "
    "normaliser, −1.300 / −3.604 on hz1x1 SUMO — is ABSENT on this scenario. ρ "
    "itself does not depend on it: its anchors are fixed-time and MaxPressure. The paper says the "
    "random-normalised number is not reported for grid4x4 SUMO and why.",
)

#: E3(a)'s rule for grid4x4: the 16 targets are READ from P7.3d's calibration artifact, pinned by
#: digest, and never recomputed -- a target read from an unpinned artifact can be edited between
#: the calibration and the campaign without leaving a trace.  It moves only in a commit that also
#: moves the artifact.
P7_3D_CALIBRATION_NAME = "p7_3d_calibration.json"
P7_3D_CALIBRATION_SHA256 = "3e9df8eed4af2e42c132087e711751bc4c265edcef75dd88e24e6143e82f9723"

P7_2B_CALIBRATION_NAME = "p7_2b_calibration.json"
P7_2B_CALIBRATION_SHA256 = "92b1592de637cee187c56b988ce320611d89706c8f9f02c34fe7ebbe81658d86"

#: G1 -- the checkpoint pins, and why they are these files.
#:
#: ``BRIEF_37`` §3.5 and Amendment E3(b) named ``SHA256SUMS_p4_6.txt``.  It lists ``p4_6/checkpoints/*``
#: and **no** ``p4_dt/`` path, the repository has recorded that since ``BRIEF_27``
#: (``nortg_campaign.TIER_MANIFEST["mappo1000"] = None``, ``DEFERRED`` 56), and every
#: ``SHA256SUMS_*`` file is gitignored, so it is not a record a reader of the repository can check.
#: G1's ruling: **both subjects are pinned against COMMITTED artifacts**, and the local manifest is
#: checked as well wherever it happens to list the file.
CHECKPOINT_RECORD: Mapping[str, str] = {
    "mappo1000": "p4_gate.json",
    "mix50": "p4_7_training.json",
    # P7.3b section 3.3's artifact.  The anchor is the first subject in this project whose
    # committed record was written BY the task that consumes it, which is why that file is
    # digest-pinned here as well as hashed at consumption.
    ANCHOR_SUBJECT: P7_3B_TRAINING_NAME,
}

#: The gitignored campaign manifest that also lists the file, where one exists.  ``None`` for
#: ``mappo1000`` is ``DEFERRED`` 56 recorded rather than worked around.
LOCAL_CHECKPOINT_MANIFEST: Mapping[str, str | None] = {
    "mappo1000": None,
    "mix50": "SHA256SUMS_p4_7.txt",
    # Written last and atomically by P7.3b's training driver, then re-verified by it.
    ANCHOR_SUBJECT: "SHA256SUMS_p7_3b_anchor.txt",
}

#: ``p4_7_training.json`` carries four METHODS per (tier, seed) -- ``bc``, ``bc_top10``, ``iql``,
#: ``dt`` -- so ``(tier, seed)`` matches four rows, three of which are other checkpoints.  The DT is
#: the one A17(c) registered.
MIX50_TRAINING_METHOD = "dt"

#: The parity artefacts of one draw, and the provenance file that records both digests (G2).
PARITY_CONFIG_NAME = "noteleport.sumocfg"
PARITY_ROUTES_NAME = "routes.rou.xml"
PARITY_PROVENANCE_NAME = "provenance.json"

#: The declared arm names, anchors included: what ``report``'s fence admits and nothing else.
DECLARED_ARM_NAMES: frozenset[str] = frozenset(
    [spec.name for spec in DECLARED_ARMS] + list(ANCHOR_ARMS) + [ANCHOR_ARM]
)

#: P7.2b's fenced marker.  A chunk that carried it would be a fenced quantity on its way into
#: ``docs/data/``; the last refusal in :func:`report` scans the serialised bytes for it.
FENCED_KEY = "fenced_do_not_report"

#: The author's ruling of 2026-09-17 (``PROJECT_PLAN`` Decisions Log): this text goes in the
#: ARTIFACT -- in the env-ATT block and on every H3 clause computed on env ATT -- and not only in
#: the packet, **because the packet does not travel with the artifact**.  The numbers are P7.1's
#: frozen teleport-free anchors, quoted exactly rather than rounded.
ATT_ENV_CAVEAT = (
    "CO-REPORTED, NOT THE REGISTERED PRIMARY (A15). On P7.1's frozen teleport-free anchors "
    "(nominal demand, five engine seeds), fixed-time minus MaxPressure on this definition was "
    "-1.50, +2.08, +4.92, +13.14, +18.70 s, negative at seed 1000, the engine seed every cell "
    "here uses; on the primary pool-clock definition it was +206.5 to +252.6 s. A per-draw ratio "
    "with a denominator this small can be dominated by a few draws and can change sign. Read "
    "this definition's rho only with denominator_diagnostic beside it."
)


def _sha256_file(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomic, sorted, newline-terminated -- so an artifact is either whole or absent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def _git(*arguments: str) -> str:
    """Run one git command in THIS module's tree, and RAISE if it cannot answer.

    ⚠️ **Amendment J1(a): this deliberately does NOT reuse ``materialise_draws._git_commit``.**
    That helper returns ``dirty = False`` whenever ``git status`` cannot run, so an UNMEASURED tree
    records as a clean one -- and ``BRIEF_37`` §7's gate, *the zero-shot artifact's provenance is
    clean*, would then be satisfied by the absence of evidence rather than by evidence.  A
    provenance this task cannot measure is one it refuses to guess.
    """
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {_REPO_ROOT} with exit {result.returncode}: "
            f"{result.stderr.strip()!r}. P7.3a records provenance it has MEASURED or it records "
            "nothing: a helper that falls back on 'clean' turns an unmeasurable tree into a clean "
            "one, which is the reading section 7's gate must never be given"
        )
    return result.stdout


def _git_provenance() -> dict[str, Any]:
    """The commit this code is at, and whether the tree carrying it has uncommitted changes.

    Measured strictly (J1(a)).  ``git_dirty`` is ``True`` for ANY entry in ``git status
    --porcelain``, untracked files included: an untracked file in the tree a campaign runs from is
    exactly what J1(e)'s dedicated worktree exists to prevent.
    """
    commit = _git("rev-parse", "HEAD").strip()
    status = [line for line in _git("status", "--porcelain").splitlines() if line.strip()]
    return {"git_commit": commit, "git_dirty": bool(status)}


def _paths_changed_between(commit: str, other: str = "HEAD") -> list[str]:
    """The paths that differ between two commits, as ``git diff --name-only`` reports them."""
    return [
        line.strip()
        for line in _git("diff", "--name-only", commit, other).splitlines()
        if line.strip()
    ]


def code_changed_since(commit: str, other: str = "HEAD") -> list[str]:
    """Paths outside ``docs/`` that differ between *commit* and *other*.

    Amendment J1(c): **commits may differ only by documentation.**  A chunk rolled before the
    stage-1 artifact's own commit -- which is a docs commit -- is still the same computation, so a
    campaign must not re-roll 1,200 cells because a packet was written; a chunk rolled by different
    CODE is a different computation and is not reusable.  An unknown revision makes :func:`_git`
    raise, which is the right answer for a chunk whose provenance cannot be resolved at all.
    """
    return [
        path for path in _paths_changed_between(commit, other) if not path.startswith("docs/")
    ]


def _data_dir(data_dir: str | Path | None) -> Path:
    return DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)


def load_calibration(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """E3(a): P7.2b's calibration artifact, digest-checked BEFORE it is parsed.

    ``targets_for_subject`` takes a mapping and reads no file, which is why the digest could not be
    tested at §3.5a: this is the seam T6's second mutation needs.  The comparison is on the whole
    digest, not a prefix.
    """
    path = Path(artifact_path) if artifact_path is not None else DEFAULT_DATA_DIR / P7_2B_CALIBRATION_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path}: P7.2b's calibration artifact is not here")
    digest = _sha256_file(path)
    if digest != P7_2B_CALIBRATION_SHA256:
        raise ValueError(
            f"{path.name}: sha256 {digest} is not the pinned {P7_2B_CALIBRATION_SHA256}. A17's "
            "targets are READ from this artifact and never recomputed, so a file that has moved "
            "under the pin is a different registration; refusing rather than evaluating prompts "
            "nobody registered"
        )
    return json.loads(path.read_bytes())


def load_anchor_training(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """P7.3b section 3.3's training record, digest-checked BEFORE it is parsed.

    The same shape as :func:`load_calibration`, for the same reason (E3(a)): the pin is a
    DECLARATION -- *the anchor's prompt and its checkpoint digests came from THAT file* -- and the
    file is the evidence.  A target read from an unpinned artifact can be edited between the
    training and the campaign without leaving a trace.

    ⚠️ **This is the anchor's ONLY prompt source.**  Amendment A5: the proof that it is not
    P7.2b's calibration artifact is the ROUTE, never a value -- on the 201-300 half the naive
    in-domain target and A17's Rule A ``q = 1.0`` were the same number by construction.  (On the
    full 200 episodes they are not: -20625.0 against -20809.0.  That is a fact about this corpus,
    not a guarantee, and nothing here relies on it.)
    """
    path = (
        _data_dir(None) / P7_3B_TRAINING_NAME if artifact_path is None else Path(artifact_path)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: the anchor's committed training record is not here. Section 3.3 writes it "
            "and the evaluation pins every anchor checkpoint and the prompt against it"
        )
    digest = _sha256_file(path)
    if digest != P7_3B_TRAINING_SHA256:
        raise ValueError(
            f"{path}: sha256 {digest} is not the pinned {P7_3B_TRAINING_SHA256}. The record the "
            "anchor's prompt and checkpoint digests come from has moved; refusing to read targets "
            "from an artifact this code does not name"
        )
    return json.loads(path.read_bytes())


def anchor_seed_record(seed: int, artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    """One seed's row in the training record, found by seed and required to be unique."""
    matching = [row for row in artifact.get("seeds", ()) if int(row["seed"]) == int(seed)]
    if len(matching) != 1:
        raise ValueError(
            f"{P7_3B_TRAINING_NAME}: {len(matching)} rows match seed {seed}; exactly one is "
            f"required. The record declares seeds {[int(r['seed']) for r in artifact.get('seeds', ())]}"
        )
    return matching[0]


def anchor_subject_facts(artifact: Mapping[str, Any]) -> Any:
    """The anchor's :class:`~offline.transfer_calibration.SubjectFacts`, from its own record.

    ``subject_facts`` reads a CityFlow subject's five checkpoints and raises for anything outside
    A17(c)'s two; the anchor's equivalent is read from the committed training artifact, which
    carries the same quantities.  ``support_range_over_the_split`` is the fitted ``stats.rtg``
    range over the anchor's own 200 episodes, and ``training_set_return_min`` is ``-rtg_scale``,
    exactly as ``subject_facts`` defines them (Amendment A1: the diagnostic never selects).
    """
    from offline.transfer_calibration import SubjectFacts

    stats = artifact["normalisation_stats"]["rtg"]
    pairs = [(s, i, v) for s, per_ix in stats.items() for i, v in per_ix.items()]
    if len(pairs) != 1:
        raise ValueError(
            f"{P7_3B_TRAINING_NAME}: the statistics cover {len(pairs)} (scenario, intersection) "
            "pairs; the anchor trains one intersection and has one support range"
        )
    summary = pairs[0][2]
    scale = float(artifact["rtg_scale"])
    return SubjectFacts(
        subject=ANCHOR_SUBJECT,
        best_source_return=float(artifact["target_rtg"]),
        rtg_scale=scale,
        support_range_over_the_split=(float(summary["min"]), float(summary["max"])),
        n_rows=int(summary["count"]),
        training_set_return_min=-scale,
        checkpoints=tuple(str(row["checkpoint"]) for row in artifact["seeds"]),
        state_dim=int(artifact["state_dim"]),
        context_length=int(artifact["recipe"]["context_length"]),
    )


def checkpoint_path_for(subject: str, seed: int, *, output_root: str | Path) -> Path:
    """Where one subject-seed checkpoint lives, from P7.2b's own subject table."""
    from offline.transfer_calibration import SUBJECTS as SUBJECT_LAYOUT

    if subject == ANCHOR_SUBJECT:
        return Path(output_root) / ANCHOR_CHECKPOINT_SUBDIR / f"{ANCHOR_CHECKPOINT_STEM}{int(seed)}.pt"
    if subject not in SUBJECT_LAYOUT:
        raise ValueError(f"unknown subject {subject!r}; A17(c) registers {sorted(SUBJECT_LAYOUT)}")
    spec = SUBJECT_LAYOUT[subject]
    return Path(output_root) / spec["subdir"] / f"{spec['stem']}{int(seed)}.pt"


def _committed_checkpoint_digest(subject: str, seed: int, *, data_dir: Path) -> tuple[str, str]:
    """The digest a COMMITTED artifact records for this checkpoint, and the artifact's name."""
    record_name = CHECKPOINT_RECORD[subject]
    record_path = data_dir / record_name
    if not record_path.is_file():
        raise FileNotFoundError(
            f"{record_path}: {subject}'s checkpoints are pinned against this committed artifact "
            "(G1), and it is not here"
        )
    record = json.loads(record_path.read_bytes())

    if subject == ANCHOR_SUBJECT:
        # Digest-pinned BEFORE it is parsed, unlike the two P4 records: this file was written by
        # the same task that consumes it, so the pin is what stops that being circular.
        return str(anchor_seed_record(seed, load_anchor_training(record_path))["checkpoint_sha256"]), record_name

    if subject == "mappo1000":
        entry = record.get("checkpoints", {}).get(str(int(seed)))
        if not isinstance(entry, Mapping) or "sha256" not in entry:
            raise ValueError(
                f"{record_name}: no checkpoints[{seed!r}].sha256 for {subject}; a checkpoint absent "
                "from its committed record has no integrity evidence at consumption"
            )
        return str(entry["sha256"]), record_name

    matching = [
        row
        for row in record.get("runs", ())
        if str(row.get("tier")) == "mix50"
        and str(row.get("method")) == MIX50_TRAINING_METHOD
        and int(row.get("seed", -1)) == int(seed)
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{record_name}: {len(matching)} rows match (tier=mix50, "
            f"method={MIX50_TRAINING_METHOD}, seed={seed}); exactly one is required. The key "
            "carries the method because this artifact records four methods per (tier, seed) and "
            "three of them are other checkpoints"
        )
    return str(matching[0]["file_sha256"]), record_name


def checkpoint_identity(
    subject: str, seed: int, *, output_root: str | Path, data_dir: str | Path | None = None
) -> dict[str, Any]:
    """G1: this checkpoint's digest, recomputed from the file and checked at CONSUMPTION.

    ``BRIEF_27`` B3(a)'s rule: *a digest checked once is not a digest checked when used.*  The file
    is hashed here, every record that names it is compared, and the record names are returned so the
    chunk can say which evidence exists for which subject instead of implying they are equal --
    ``mappo1000`` has no campaign manifest at all (``DEFERRED`` 56).
    """
    path = checkpoint_path_for(subject, seed, output_root=output_root)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: {subject} seed {seed}'s checkpoint is not on disk")
    digest = _sha256_file(path)

    data = _data_dir(data_dir)
    committed, record_name = _committed_checkpoint_digest(subject, seed, data_dir=data)
    checked_against = [record_name]
    if digest != committed:
        raise ValueError(
            f"{path}: file sha256 {digest} is not the {committed} that {record_name} records for "
            f"{subject} seed {seed}. Different weights under the same filename would evaluate as "
            "the registered subject and could not be told apart afterwards"
        )

    manifest_name = LOCAL_CHECKPOINT_MANIFEST[subject]
    manifest_path = Path(output_root) / manifest_name if manifest_name else None
    if manifest_path is not None and manifest_path.is_file():
        listed = _manifest_digests(manifest_path)
        relative = str(path.relative_to(Path(output_root)))
        if relative not in listed:
            raise ValueError(
                f"{relative} is not listed in {manifest_name}, which does list this campaign's "
                "other checkpoints; a partial manifest is a gap, not a pass"
            )
        if listed[relative] != digest:
            raise ValueError(
                f"{path}: file sha256 {digest} is not {manifest_name}'s {listed[relative]}"
            )
        checked_against.append(manifest_name)

    return {
        "subject": subject,
        "seed": int(seed),
        "path": str(path),
        "file_sha256": digest,
        "sha256_checked_against": checked_against,
        "local_manifest": manifest_name,
        "deferred_56": manifest_name is None,
    }


def _manifest_digests(path: Path) -> dict[str, str]:
    """A ``sha256sum``-format manifest as ``{relative path: digest}``."""
    digests: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        digests[name.strip()] = digest.strip()
    return digests


def demand_identity(
    draw_id: int, *, out_root: str | Path, scenario: str = SCENARIO_KEY
) -> dict[str, Any]:
    """G2: the draw's demand, pinned by TWO digests against P7.2a's own provenance.

    The ``.sumocfg`` only *names* the routes file, so a regenerated ``routes.rou.xml`` leaves the
    cfg's bytes -- and its digest -- untouched.  rho pairs an arm with the two anchors of the same
    draw, and all three cells must provably have run the same demand, across a resume and across
    the days a campaign may span.
    """
    from offline.materialise_draws import parity_sumocfg_path

    config_path = parity_sumocfg_path(str(scenario), int(draw_id), out_root=out_root)
    parity_dir = config_path.parent
    routes_path = parity_dir / PARITY_ROUTES_NAME
    provenance_path = parity_dir / PARITY_PROVENANCE_NAME
    for path in (config_path, routes_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"draw {draw_id}: {path} is absent; P7.2a materialises the band into the MAIN "
                "tree's scenarios/draws and this task only reads it"
            )

    recorded = json.loads(provenance_path.read_bytes()).get("files", {})
    identity = {
        "draw_id": int(draw_id),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "routes_sha256": _sha256_file(routes_path),
    }
    for name, key in ((PARITY_CONFIG_NAME, "config_sha256"), (PARITY_ROUTES_NAME, "routes_sha256")):
        if name not in recorded:
            raise ValueError(
                f"draw {draw_id}: {PARITY_PROVENANCE_NAME} records no digest for {name}, so the "
                "demand cannot be pinned to what P7.2a generated"
            )
        if str(recorded[name]) != identity[key]:
            raise ValueError(
                f"draw {draw_id}: {name} hashes to {identity[key]} but {PARITY_PROVENANCE_NAME} "
                f"records {recorded[name]}. The demand on disk is not the demand P7.2a "
                "materialised; every number computed on it would describe a different scenario"
            )
    return identity


def _load_p7_3d_calibration(data_dir: str | Path | None) -> tuple[Path, dict[str, Any]]:
    """``p7_3d_calibration.json``, digest-checked BEFORE it is parsed (E3(a)'s rule for grid4x4)."""
    path = _data_dir(data_dir) / P7_3D_CALIBRATION_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent; P7.3d C3a writes it from both probe halves' chunks"
        )
    digest = _sha256_file(path)
    if digest != P7_3D_CALIBRATION_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}, not the pinned {P7_3D_CALIBRATION_SHA256}. The targets "
            "are a registered quantity; this module reads them from THAT file and no other"
        )
    return path, json.loads(path.read_bytes())


def grid4x4_support_ranges(*, data_dir: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """The 16 training-support ranges, READ from the same digest-pinned artifact as the targets.

    ``support_range`` per intersection is what C3a recorded from the subject's checkpoints
    (``stats["rtg"]`` min and max, Amendment A4: it bounds the support and selects nothing).  A cell
    counts its decisions against it per intersection; reading it here, rather than re-opening five
    checkpoints per cell, keeps the prompt and the diagnostic on ONE pinned file.  Returned in the
    artifact's ``intersection_ids`` order.
    """
    _path, payload = _load_p7_3d_calibration(data_dir)
    ranges: dict[str, tuple[float, float]] = {}
    for ix_id in payload["intersection_ids"]:
        low, high = payload["per_intersection"][ix_id]["support_range"]
        ranges[str(ix_id)] = (float(low), float(high))
    return ranges


def load_grid4x4_targets(*, data_dir: str | Path | None = None) -> dict[str, float]:
    """The 16 registered prompts, READ from the digest-pinned calibration artifact.

    The target of intersection *i* at the registered budget (A20(b): ``k = 100``, and only that
    budget carries the role).  The artifact is checked against
    :data:`P7_3D_CALIBRATION_SHA256` before a value is taken from it, and the role is checked on
    every intersection: a budget the registration records and never evaluates must not be able to
    become the prompt by an edit to one field.
    """
    path, payload = _load_p7_3d_calibration(data_dir)
    key = f"k{int(payload['registered_k'])}"
    targets: dict[str, float] = {}
    for ix_id in payload["intersection_ids"]:
        cell = payload["per_intersection"][ix_id]["budgets"][key]
        if str(cell["role"]) != "registered_prompt":
            raise ValueError(
                f"{path}: intersection {ix_id!r} carries role {cell['role']!r} at {key}, not "
                "'registered_prompt'"
            )
        targets[str(ix_id)] = float(cell["target"])
    return targets


def assert_rtg_first_matches_targets(
    rtg_series: Mapping[str, Sequence[float]],
    targets: Mapping[str, float],
    *,
    label: str,
) -> None:
    """``rtg_first_i == target_i`` for EVERY intersection, refusing by name (§3 C3b).

    ``BRIEF_38`` §2's seam 2 on hz1x1 -- *the only guard that the prompt took effect* -- one level
    up.  Fifteen intersections agreeing must not be able to hide the sixteenth, so every id is
    checked and every disagreement is named with both numbers.  Compared under ``==``: both sides
    are the same float, read back from the same mapping, and a tolerance here would accept a
    target that was rounded on its way into the model.
    """
    missing = sorted(ix for ix in targets if ix not in rtg_series)
    extra = sorted(ix for ix in rtg_series if ix not in targets)
    if missing or extra:
        raise ValueError(
            f"{label}: the RTG series do not cover exactly the targeted intersections (missing "
            f"{missing[:4]}, unexpected {extra[:4]})"
        )
    empty = sorted(ix for ix, series in rtg_series.items() if not list(series))
    if empty:
        raise ValueError(
            f"{label}: intersection(s) {empty[:4]} recorded NO decision, so there is no rtg_first "
            "to compare; an empty series is not a passing one"
        )
    disagreeing = [
        (ix, float(list(rtg_series[ix])[0]), float(targets[ix]))
        for ix in sorted(targets)
        if float(list(rtg_series[ix])[0]) != float(targets[ix])
    ]
    if disagreeing:
        raise ValueError(
            f"{label}: {len(disagreeing)} intersection(s) did not condition on their own target "
            f"(first: {disagreeing[:3]}); the prompt did not reach them"
        )


# ======================================================================================
# The B.6 fix round: ONE scenario-aware call per identity, used by run_cell, chunk_is_reusable
# AND report (BRIEF_39 Amendment B.6-2(2); docs/reviews/P7.3d-preflight.md, blocker B2)
# ======================================================================================

#: The scenarios a cell may name.  hz1x1 is the ABSENT default -- P7.3a's and P7.3b's 5,404
#: chunks carry no ``scenario`` key, and their bytes may not move (T-regress).
REGISTERED_SCENARIOS: tuple[str, ...] = (SCENARIO_KEY, GRID4X4_SCENARIO_KEY)

#: The label a grid4x4 chunk's ``sha256_checked_against`` carries for A20(a)'s committed pin, and
#: the gitignored campaign manifest that also lists the five files.
GRID4X4_CHECKPOINT_PIN_LABEL = (
    "A20(a): offline.transfer_calibration.GRID4X4_CHECKPOINT_SHA256 (pinned at fcf22fc)"
)
GRID4X4_LOCAL_MANIFEST = "SHA256SUMS_p5_2.txt"


def scenario_of(cell: Mapping[str, Any]) -> str:
    """The cell's scenario key, resolved STRICTLY.

    Absent or ``None`` is hz1x1, which is what every P7.3a / P7.3b cell and chunk is.  Anything not
    in :data:`REGISTERED_SCENARIOS` refuses by name: a key this module does not register must never
    fall through to hz1x1's pins -- the coincidence-dependent seam Amendments A.1-2 and B4 warn
    about, where a branch taken by default is a branch nobody decided.
    """
    value = cell.get("scenario")
    scenario = SCENARIO_KEY if value is None else str(value)
    if scenario not in REGISTERED_SCENARIOS:
        raise ValueError(
            f"scenario {scenario!r} is not one this module runs cells on ({list(REGISTERED_SCENARIOS)}); "
            "refusing rather than resolving it to hz1x1's demand, checkpoints and calibration"
        )
    return scenario


def demand_identity_for(cell: Mapping[str, Any], *, out_root: str | Path) -> dict[str, Any]:
    """:func:`demand_identity` for THIS cell's scenario -- the one call all three sites make.

    B2, the blocker that could make a number wrong: ``run_cell``, ``chunk_is_reusable`` and
    ``report`` each called ``demand_identity(draw_id, out_root=...)`` with no scenario, so a grid4x4
    anchor cell ran against hz1x1's configuration path and recorded hz1x1's digests, and the two
    checks that re-derive the identity re-derived the SAME default and found it consistent.  One
    function, called by all three, is what makes the mismatch visible instead of invisible.
    """
    return demand_identity(int(cell["draw_id"]), out_root=out_root, scenario=scenario_of(cell))


def grid4x4_checkpoint_identity(seed: int, *, output_root: str | Path) -> dict[str, Any]:
    """The registered grid4x4 subject's checkpoint, hashed at CONSUMPTION against A20(a)'s pin.

    ``checkpoint_identity`` pins hz1x1's subjects against their committed training records
    (``CHECKPOINT_RECORD``), and it refuses this subject by design.  A20(a) registers the grid4x4
    subject BY DIGEST, and those five digests were first committed in
    :data:`offline.transfer_calibration.GRID4X4_CHECKPOINT_SHA256` (``fcf22fc``); the file is
    hashed here and compared with that pin, and with ``SHA256SUMS_p5_2.txt`` wherever that
    gitignored manifest exists -- a manifest that exists and does not list the file is a gap, not a
    pass, the rule :func:`checkpoint_identity` applies.  The returned mapping has the same keys.
    """
    from offline.transfer_calibration import (
        GRID4X4_CHECKPOINT_SHA256,
        GRID4X4_CHECKPOINT_STEM,
        GRID4X4_CHECKPOINT_SUBDIR,
    )

    seed = int(seed)
    if seed not in GRID4X4_CHECKPOINT_SHA256:
        raise ValueError(
            f"{GRID4X4_SUBJECT}: seed {seed} is not one of A20(a)'s {sorted(GRID4X4_CHECKPOINT_SHA256)}"
        )
    root = Path(output_root)
    path = root / GRID4X4_CHECKPOINT_SUBDIR / f"{GRID4X4_CHECKPOINT_STEM}{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: {GRID4X4_SUBJECT} seed {seed}'s checkpoint is not on disk")
    digest = _sha256_file(path)
    pinned = GRID4X4_CHECKPOINT_SHA256[seed]
    if digest != pinned:
        raise ValueError(
            f"{path}: file sha256 {digest} is not A20(a)'s pinned {pinned} for {GRID4X4_SUBJECT} "
            f"seed {seed}. A20(a) registers the subject BY DIGEST; different weights under the same "
            "filename would evaluate as the registered subject and could not be told apart"
        )
    checked_against = [GRID4X4_CHECKPOINT_PIN_LABEL]
    manifest_path = root / GRID4X4_LOCAL_MANIFEST
    if manifest_path.is_file():
        listed = _manifest_digests(manifest_path)
        relative = str(path.relative_to(root))
        if relative not in listed:
            raise ValueError(
                f"{relative} is not listed in {GRID4X4_LOCAL_MANIFEST}, which does exist; a partial "
                "manifest is a gap, not a pass"
            )
        if listed[relative] != digest:
            raise ValueError(
                f"{path}: file sha256 {digest} is not {GRID4X4_LOCAL_MANIFEST}'s {listed[relative]}"
            )
        checked_against.append(GRID4X4_LOCAL_MANIFEST)
    return {
        "subject": GRID4X4_SUBJECT,
        "seed": seed,
        "path": str(path),
        "file_sha256": digest,
        "sha256_checked_against": checked_against,
        "local_manifest": GRID4X4_LOCAL_MANIFEST,
        "deferred_56": False,
    }


def checkpoint_identity_for(
    cell: Mapping[str, Any], *, output_root: str | Path, data_dir: str | Path | None = None
) -> dict[str, Any]:
    """The checkpoint identity for THIS cell's scenario -- again one call for all three sites.

    hz1x1 goes to :func:`checkpoint_identity`, unchanged, so every P7.3a / P7.3b digest check is
    byte-for-byte what it was; grid4x4 goes to :func:`grid4x4_checkpoint_identity`.  At ``8c79778``
    the grid4x4 subject reached ``checkpoint_identity`` and raised ``unknown subject`` -- inside
    ``run_cell`` for all 500 DT cells, and inside ``chunk_is_reusable``, where the exception was
    caught and read as *not reusable*, so every restart would have re-rolled every DT chunk (M2).
    """
    if scenario_of(cell) == GRID4X4_SCENARIO_KEY:
        if str(cell.get("subject")) != GRID4X4_SUBJECT:
            raise ValueError(
                f"{cell.get('subject')!r} is not the registered grid4x4 subject {GRID4X4_SUBJECT!r} "
                "(A20(a))"
            )
        return grid4x4_checkpoint_identity(int(cell["seed"]), output_root=output_root)
    return checkpoint_identity(
        str(cell["subject"]), int(cell["seed"]), output_root=output_root, data_dir=data_dir
    )


#: C4's reference cells: rho's two anchors on the first three held-out draws, frozen so the
#: campaign's own chunks for those six cells can be checked against them (A9, *the instrument
#: regenerates*).  P7.3a had P7.1's frozen values to check against; grid4x4 has none until this
#: file exists.
REFERENCE_CELL_DRAWS: tuple[int, ...] = (1000, 1001, 1002)
REFERENCE_CELL_ARMS: tuple[str, ...] = ("fixedtime", "maxpressure")
REFERENCE_CELLS_FORMAT_VERSION = "p7.3d-reference-cells/1.0"

#: The only fields of a frozen reference row that a re-roll may differ on: WALL CLOCKS.  Measured
#: 2026-09-22 on the first run of the pedigree check -- all 34 recorded fields of all six episodes
#: reproduced except these two, which are ``time.perf_counter`` differences and reproduce on no
#: machine.  They are named here rather than compared loosely, so that a field which stops
#: reproducing has to be added to this tuple by someone who then has to justify it.  The same
#: shape as ``_PUBLISHED_FIELDS``' exclusion of ``seconds`` in the campaign's own comparisons.
_PEDIGREE_WALL_CLOCK_FIELDS: tuple[str, ...] = ("seconds", "seconds_rollout")
P7_3D_REFERENCE_CELLS_NAME = "p7_3d_reference_cells.json"

#: m1 (B.6-2(4)): the campaign's two other committed inputs, pinned by digest exactly as
#: :data:`P7_3D_CALIBRATION_SHA256` is -- a DECLARATION that the driver ran against THESE files,
#: which moves only in a commit that also moves the file.  Measured from the committed files at
#: ``e14d950`` (``9c4a979`` and ``d465ec3`` wrote them); a test recomputes both.
P7_3D_REFERENCE_CELLS_SHA256 = "5265f0d5d8b1bc11b4d40fdcde6821391940139dfe9e7fc61723fa98c6121b2c"
P7_3D_CAP_E_NAME = "p7_3d_cap_e.json"
P7_3D_CAP_E_SHA256 = "949015b3d1be8b96272b0826fa750497f7c57a0d285f6ceb7baefe8f9da9e0cd"


#: B.7-2(iii): the fields a campaign chunk must reproduce, bit for bit, against its frozen C4 row --
#: every field BOTH records, as ``(frozen row key, chunk key)``, the aliases being the same
#: quantity under ``run_cell``'s names (``n_created`` is ``e_sumo.n_ids``, ``n_entered`` is
#: ``n_departed``, ``n_never_entered`` is ``n_never_inserted``).  ⚠️ ``episode_reward``, which the
#: amendment names, is NOT in the frozen record (its row keys, read from disk), so it cannot be
#: compared; ``e_sumo_total``, ``n_intended`` and ``n_arrived`` are in the frozen record and not in
#: a chunk; ``seconds`` is a clock.
REFERENCE_CELL_COMPARED_FIELDS: tuple[tuple[str, str], ...] = (
    ("e_sumo", "e_sumo"),
    ("att_env", "att_env"),
    ("n_teleports", "n_teleports"),
    ("decisions", "decisions"),
    ("n_observations", "n_observations"),
    ("engine_seed_requested", "engine_seed_requested"),
    ("engine_seed_drawn", "engine_seed_drawn"),
    ("e_sumo_n_ids", "n_created"),
    ("n_departed", "n_entered"),
    ("n_never_inserted", "n_never_entered"),
    ("n_pending_at_horizon", "n_pending_at_horizon"),
    ("n_vanished_without_arrival", "n_vanished_without_arrival"),
    ("vehicle_types_seen", "vehicle_types_seen"),
    ("time_to_teleport_option", "time_to_teleport_option"),
    ("config_sha256", "config_sha256"),
    ("routes_sha256", "routes_sha256"),
    ("halting_checked", "halting_checked"),
)
#: Compared only where the frozen run CHECKED halting: the frozen row records the recorder's 0 on an
#: unchecked draw, a chunk records ``None`` (measured on the committed artifact's draws 1001-1002).
REFERENCE_CELL_HALTING_COUNTS: tuple[str, ...] = (
    "halting_n_lane_seconds",
    "halting_n_disagreeing_lane_seconds",
)
REFERENCE_CELL_NOT_COMPARED: tuple[str, ...] = ("e_sumo_total", "n_intended", "n_arrived", "seconds")


def load_reference_cells(*, data_dir: str | Path | None = None) -> dict[str, Any]:
    """C4's six frozen anchors, digest-checked BEFORE they are parsed, and refused unless whole.

    The same shape as :func:`load_grid4x4_targets`: :data:`P7_3D_REFERENCE_CELLS_SHA256` is a
    DECLARATION that the campaign's regeneration check reads THAT file.  Every row must carry every
    field :func:`reference_cell_differences` compares -- a key missing from the frozen row would
    compare ``None`` with ``None`` and pass, which is the one way this check could say nothing.
    """
    path = _data_dir(data_dir) / P7_3D_REFERENCE_CELLS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is absent; P7.3d C4 writes it")
    digest = _sha256_file(path)
    if digest != P7_3D_REFERENCE_CELLS_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}, not the pinned {P7_3D_REFERENCE_CELLS_SHA256}; the "
            "regeneration anchors are read from THAT file and no other"
        )
    payload = json.loads(path.read_bytes())
    rows = {(str(row["arm"]), int(row["draw_id"])): row for row in payload["cells"]}
    expected = {(arm, draw) for arm in REFERENCE_CELL_ARMS for draw in REFERENCE_CELL_DRAWS}
    if set(rows) != expected or len(payload["cells"]) != len(expected):
        raise ValueError(f"{path} does not hold exactly the six cells {sorted(expected)}")
    needed = [key for key, _ in REFERENCE_CELL_COMPARED_FIELDS] + list(REFERENCE_CELL_HALTING_COUNTS)
    for key, row in sorted(rows.items()):
        absent = [name for name in needed if name not in row]
        if absent:
            raise ValueError(f"{path}: the frozen row {key} records no {absent}")
    return payload


def reference_cell_differences(chunk: Mapping[str, Any], frozen: Mapping[str, Any]) -> list[str]:
    """The CHUNK's names of every compared field on which it is not the frozen row, under ``==``.

    Bit-for-bit: ``==`` on the values as recorded, the aliases of :data:`REFERENCE_CELL_COMPARED_FIELDS`
    applied, and the two halting counts compared only where the frozen run checked halting.
    """
    differing: list[str] = []
    for frozen_key, chunk_key in REFERENCE_CELL_COMPARED_FIELDS:
        if frozen_key not in frozen:
            raise ValueError(f"the frozen row records no {frozen_key!r}; nothing can be compared")
        if chunk.get(chunk_key) != frozen[frozen_key]:
            differing.append(chunk_key)
    if frozen.get("halting_checked") is True:
        differing.extend(key for key in REFERENCE_CELL_HALTING_COUNTS if chunk.get(key) != frozen.get(key))
    return differing


def grid4x4_return_lanes(intersections: Sequence[Any], info: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each intersection's incoming lanes, in the frame *info* keys its lane counts in.

    The SUMO probe read the second return route off the PLAIN env's info with each intersection's
    own SUMO incoming lanes (``transfer_calibration._roll_sumo_probe_episode_per_intersection``); an
    anchor cell's observed-but-unwrapped info is keyed the same way.  A DT cell's info is ALIGNED:
    ``align_info`` re-keys the lane counts to canonical ids and records ``lane_id_translation``
    (canonical -> SUMO), so the SAME SUMO lanes are read here through its inverse.  A SUMO lane
    without a canonical partner refuses -- dropping it would shorten the second route silently.
    """
    translation = info.get("lane_id_translation")
    lanes: dict[str, list[str]] = {}
    if translation is None:
        for ix in intersections:
            lanes[str(ix.id)] = [str(lane) for lane in ix.incoming_lanes]
        return lanes
    inverse: dict[str, str] = {}
    for canonical, sumo in translation.items():
        if str(sumo) in inverse:
            raise ValueError(f"the lane translation maps two canonical lanes onto {sumo!r}")
        inverse[str(sumo)] = str(canonical)
    for ix in intersections:
        missing = [str(lane) for lane in ix.incoming_lanes if str(lane) not in inverse]
        if missing:
            raise ValueError(
                f"intersection {ix.id!r}: SUMO incoming lane(s) {missing[:3]} have no canonical "
                "partner in the aligned info's lane_id_translation, so the lane route cannot be read"
            )
        lanes[str(ix.id)] = [inverse[str(lane)] for lane in ix.incoming_lanes]
    return lanes


def per_intersection_local_returns(
    post_step: Sequence[Mapping[str, Any]], lanes_by_ix: Mapping[str, Sequence[str]]
) -> tuple[dict[str, float], dict[str, float]]:
    """B.7.1-2's ``R_i``: each intersection's episode return under the collection reward, two routes.

    The sum over the POST-STEP infos -- one per decision, 360 on the horizon -- of the intersection's
    reward, by :func:`offline.rtg_calibration.episode_return_two_routes`, the probe's own function,
    and refused per intersection unless the reward route and the lane route agree under ``==``.
    ⚠️ This is NOT ``sum(reward_series[i])``: that series is read BEFORE each ``act`` (D1), so it
    carries the reset info's reward and not the last decision's.
    """
    from offline.rtg_calibration import episode_return_two_routes

    by_reward: dict[str, float] = {}
    by_lanes: dict[str, float] = {}
    for ix_id, lanes in lanes_by_ix.items():
        from_rewards, from_lanes = episode_return_two_routes(
            post_step, ix_id=str(ix_id), incoming_lanes=list(lanes)
        )
        if from_rewards != from_lanes:
            raise ValueError(
                f"the two return routes disagree on intersection {str(ix_id)!r} ({from_rewards!r} "
                f"from the reward stream against {from_lanes!r} from its lanes' waiting counts); "
                "A17(b)'s equality holds under == per intersection or the return is not recorded"
            )
        by_reward[str(ix_id)] = from_rewards
        by_lanes[str(ix_id)] = from_lanes
    return by_reward, by_lanes


def grid4x4_reference_cells(
    *,
    out_root: str | Path,
    draws: Sequence[int] = REFERENCE_CELL_DRAWS,
    arms: Sequence[str] = REFERENCE_CELL_ARMS,
) -> list[dict[str, Any]]:
    """Roll rho's two anchors on the first three held-out draws and record BOTH ATT definitions.

    The same construction one campaign cell uses -- ``env_for_cell``'s anchor branch, the observed
    but UNWRAPPED env (``BRIEF_37`` A2), ``anchor_choose`` through ``collect.POLICIES``,
    ``horizon_rollout(..., seed=ENGINE_SEED)`` -- so that a campaign chunk for one of these six
    cells is comparable with the frozen row cell by cell rather than merely in spirit.  The
    halting cross-check follows C2's convention (ON for draw 1000 only), because the campaign's
    own chunks for these cells will carry it.
    """
    import time

    from offline.horizon_metric import horizon_rollout
    from offline.sumo_att_reference import reconstruct_sumo_episode

    rows: list[dict[str, Any]] = []
    for arm in arms:
        for draw_id in draws:
            cell = {
                "kind": "anchor", "subject": None, "arm": str(arm), "seed": None,
                "draw_id": int(draw_id), "scenario": GRID4X4_SCENARIO_KEY,
            }
            demand = demand_identity(int(draw_id), out_root=out_root, scenario=GRID4X4_SCENARIO_KEY)
            started = time.perf_counter()
            env = env_for_cell(cell, out_root=out_root)
            try:
                assert_env_matches_cell(cell, env)
                choose, diagnostics = anchor_choose(
                    env, cell=cell, config_path=demand["config_path"]
                )
                tap = _StepTap(env)
                rollout = horizon_rollout(tap, choose, 1, ENGINE_SEED)
                built = reconstruct_sumo_episode(env.recorder)
                att_env = att_env_from_info(tap.last_info or {})
                types_seen = sorted(
                    {env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()}
                )
                option = str(env._sumo.simulation.getOption("time-to-teleport"))
                engine_seed_drawn = int(env._engine_seed)
            finally:
                env.close()
            seconds = time.perf_counter() - started

            if att_env != rollout.att_horizon:
                raise ValueError(
                    f"{arm} draw {draw_id}: att_env {att_env!r} and att_horizon "
                    f"{rollout.att_horizon!r} disagree; they are two routes to ONE quantity (G4)"
                )
            rows.append(
                {
                    "arm": str(arm),
                    "draw_id": int(draw_id),
                    "scenario": GRID4X4_SCENARIO_KEY,
                    "e_sumo": float(built.e_sumo.value),
                    "e_sumo_total": float(built.e_sumo.total),
                    "e_sumo_n_ids": int(built.e_sumo.n_ids),
                    "att_env": float(att_env),
                    "decisions": len(diagnostics["actions"]),
                    "n_observations": int(built.n_observations),
                    "n_teleports": int(built.n_teleports),
                    "n_intended": int(built.n_intended),
                    "n_departed": int(built.n_departed),
                    "n_arrived": int(built.n_arrived),
                    "n_never_inserted": int(built.n_never_inserted),
                    "n_pending_at_horizon": int(built.n_pending_at_horizon),
                    "n_vanished_without_arrival": int(built.n_vanished_without_arrival),
                    "vehicle_types_seen": types_seen,
                    "time_to_teleport_option": option,
                    "engine_seed_requested": ENGINE_SEED,
                    "engine_seed_drawn": engine_seed_drawn,
                    "halting_checked": halting_check_for(int(draw_id)),
                    "halting_n_lane_seconds": int(built.halting.n_lane_seconds),
                    "halting_n_disagreeing_lane_seconds": int(
                        built.halting.n_disagreeing_lane_seconds
                    ),
                    "config_sha256": demand["config_sha256"],
                    "routes_sha256": demand["routes_sha256"],
                    "seconds": seconds,
                }
            )
    return rows


def grid4x4_cityflow_pedigree(
    *,
    draws_root: str | Path,
    output_root: str | Path,
    corpus_root: str | Path,
    repo_root: str | Path,
    reference_artifact: str | Path,
) -> dict[str, Any]:
    """Re-run the six CityFlow reference episodes and compare EVERY field with what P8.4b froze.

    Amendment B3/Q3.  The episodes are rolled by ``engine_att_reference.run_cells`` -- the SAME
    function that produced the committed rows, not a re-implementation of it -- and every numeric
    field of every row is compared under ``==``.  G1 already checked that the held-out draws'
    ``flow.json`` are byte-identical to what P8.4a committed; this checks that the same demand
    still produces the same numbers, which is the half a digest cannot answer.
    """
    from offline.admission_probe import ProbeRoots
    from offline.engine_att_reference import default_work_dir, gate_cells, run_cells

    committed = json.loads(Path(reference_artifact).read_bytes())
    frozen = {
        (str(row["arm"]).split("@")[1], int(row["draw_id"])): row
        for row in committed["episodes"]
        if row.get("scenario") == "grid4x4"
        and str(row["arm"]) in ("behaviour@fixedtime", "behaviour@maxpressure")
    }
    if len(frozen) != 6:
        raise ValueError(
            f"{reference_artifact} carries {len(frozen)} grid4x4 anchor rows, not the six "
            "P8.4b froze on draws 1000-1002"
        )

    output_root = Path(output_root)
    roots = ProbeRoots(
        repo_root=Path(repo_root),
        corpus_root=Path(corpus_root),
        draws_root=Path(draws_root),
        output_root=output_root,
        work_dir=default_work_dir(output_root),
    )
    cells = [
        cell
        for cell in gate_cells("grid4x4")
        if cell.tier in REFERENCE_CELL_ARMS and int(cell.draw_id) in REFERENCE_CELL_DRAWS
    ]
    episodes = run_cells(cells, roots=roots, engine_seed=ENGINE_SEED, device="cpu")

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        key = (str(episode.tier), int(episode.draw_id))
        expected = frozen[key]
        observed = {
            field: getattr(episode, field)
            for field in sorted(expected)
            if hasattr(episode, field) and field not in _PEDIGREE_WALL_CLOCK_FIELDS
        }
        differing = sorted(
            field for field, value in observed.items() if value != expected[field]
        )
        rows.append(
            {
                "arm": key[0],
                "draw_id": key[1],
                "n_fields_compared": len(observed),
                "fields_excluded": list(_PEDIGREE_WALL_CLOCK_FIELDS),
                "matches": not differing,
                "differing": differing,
                "att_ours": float(episode.att_ours),
                "att_engine": float(episode.att_reference_engine_population),
            }
        )

    matching = [row for row in rows if row["matches"]]
    return {
        "what": (
            "the six CityFlow reference episodes of docs/data/p8_4b_g0_reference.json, re-rolled "
            "through engine_att_reference.run_cells -- the function that produced them -- and "
            "compared field by field under ==, excluding the two wall clocks "
            f"{list(_PEDIGREE_WALL_CLOCK_FIELDS)}, which are perf_counter differences"
        ),
        "reference_artifact": Path(reference_artifact).name,
        "reference_artifact_sha256": _sha256_file(reference_artifact),
        "n_checked": len(rows),
        "n_matching": len(matching),
        "all_match": len(matching) == len(rows) == 6,
        "rows": rows,
    }


def write_reference_cells_artifact(
    *,
    out_path: str | Path,
    cells: Sequence[Mapping[str, Any]],
    pedigree: Mapping[str, Any],
) -> dict[str, Any]:
    """The C4 artifact: the six SUMO anchors and the CityFlow pedigree verdict, refusals first.

    Every refusal precedes the write and the write is atomic, so a failed check leaves no file:
    the six cells must all be present, each under the registered regime, and the pedigree must
    have matched on all six.  A9's rule made into an artifact -- the campaign's own chunks for
    these six cells are later required to reproduce them bit for bit.
    """
    rows = list(cells)
    expected = {(arm, draw) for arm in REFERENCE_CELL_ARMS for draw in REFERENCE_CELL_DRAWS}
    seen = {(str(row["arm"]), int(row["draw_id"])) for row in rows}
    if seen != expected or len(rows) != 6:
        raise ValueError(
            f"the reference set must be exactly the {len(expected)} cells "
            f"{sorted(expected)}; got {len(rows)}: {sorted(seen)}"
        )
    from offline.transfer_calibration import PARITY_VTYPE_ID

    for row in rows:
        label = f"{row['arm']} draw {row['draw_id']}"
        if int(row["n_teleports"]) != 0:
            raise ValueError(
                f"{label}: {row['n_teleports']} teleport(s) under a configuration that requested "
                "time-to-teleport -1 (A15(c)); a frozen reference may not carry one"
            )
        if list(row["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
            raise ValueError(f"{label}: the engine ran {row['vehicle_types_seen']!r}")
        if str(row["time_to_teleport_option"]) != "-1":
            raise ValueError(f"{label}: time-to-teleport {row['time_to_teleport_option']!r}")
    if not pedigree.get("all_match") or int(pedigree.get("n_matching", 0)) != 6:
        raise ValueError(
            f"the CityFlow pedigree did not match on all six episodes "
            f"({pedigree.get('n_matching')} of {pedigree.get('n_checked')}); the held-out draws "
            "no longer reproduce the numbers P8.4b froze, which stops the task"
        )

    artifact = {
        "format_version": REFERENCE_CELLS_FORMAT_VERSION,
        "registered_in": "BRIEF_39 C4, Amendment B3/Q3; PREREGISTRATION A9, A15(b)",
        "scenario_key": GRID4X4_SCENARIO_KEY,
        "what_this_is": (
            "The two rho anchors on the first three held-out draws, under the registered regime, "
            "with BOTH ATT definitions -- the values the campaign's own chunks for these six "
            "cells must reproduce bit for bit (A9: the instrument regenerates). grid4x4 had no "
            "frozen anchor until this file; P7.3a could check against P7.1's."
        ),
        "engine_seed": ENGINE_SEED,
        "draws": list(REFERENCE_CELL_DRAWS),
        "arms": list(REFERENCE_CELL_ARMS),
        "n_cells": len(rows),
        "cells": sorted(rows, key=lambda row: (str(row["arm"]), int(row["draw_id"]))),
        "pedigree": dict(pedigree),
        "what_this_does_not_say": [
            "These are ANCHOR values -- fixed-time and MaxPressure -- not an evaluation of any "
            "subject. rho is computed from them; they are not themselves a result.",
            "Three draws are not the held-out pool: they are the instrument-regeneration check, "
            "and the campaign reports rho over all 100.",
        ],
        **_git_provenance(),
    }
    _write_json(out_path, artifact)
    return artifact


def halting_check_for(draw_id: int) -> bool:
    """Amendment C2: ON for every cell on :data:`HALTING_CHECK_DRAW`, OFF everywhere else."""
    return int(draw_id) == HALTING_CHECK_DRAW


def env_for_cell(
    cell: Mapping[str, Any],
    *,
    out_root: str | Path,
    sentinel_out_dir: str | Path = "/nonexistent",
) -> Any:
    """A2/G3: the observed env this cell runs in -- aligned for a DT, UNWRAPPED for an anchor.

    The module docstring says why: ``align_info`` drops outgoing lanes and re-keys the survivors to
    CityFlow ids, and an anchor's pressure is a difference over the env's own SUMO lane ids.  The
    arm is passed through so the anchor's env is built with its own name in the collection args,
    exactly as P7.1 built it.
    """
    import offline.aligned_env as aligned_env

    kind = str(cell["kind"])
    draw_id = int(cell["draw_id"])
    factory = (
        aligned_env.aligned_observer_env_for_draw
        if kind == "dt"
        else aligned_env.observer_env_for_draw
    )
    arm = "maxpressure" if kind == "dt" else str(cell["arm"])
    # P7.3d: the cell's own scenario, defaulting to hz1x1 so every P7.3a/P7.3b call is unchanged.
    return factory(
        str(cell.get("scenario") or SCENARIO_KEY),
        draw_id,
        out_root=out_root,
        halting_check=halting_check_for(draw_id),
        arm=arm,
        sentinel_out_dir=sentinel_out_dir,
    )


def assert_env_matches_cell(cell: Mapping[str, Any], env: Any) -> None:
    """Amendment C8: the cell builder's explicit refusal, on every cell.

    Two directions, because both are silent failures.  An anchor through A16's door raises
    ``KeyError: 'road_1_1_2_0'`` deep inside ``_phase_pressure`` -- after a SUMO process has started
    -- or, worse, would not raise at all if the door ever stopped dropping outgoing lanes.  A DT
    cell on a raw ``SumoEnv`` sees 32-wide info in SUMO lane order, which a CityFlow-trained model
    consumes happily and turns into a number that means nothing.
    """
    from offline.aligned_env import AlignedEnv

    kind = str(cell["kind"])
    aligned = isinstance(env, AlignedEnv)
    if kind == "anchor" and aligned:
        raise TypeError(
            f"{cell['arm']} is an anchor and must run on the observed, UNWRAPPED env: align_info "
            "drops outgoing lanes and re-keys the rest to CityFlow ids, while the anchor's pressure "
            "is a difference over the env's own SUMO lane ids (Amendment A2, measured). A16 is "
            "untouched by this -- the door is the route into a CityFlow-trained MODEL's frame, and "
            "an anchor has no frame to enter"
        )
    if kind == "dt" and not aligned:
        raise TypeError(
            f"a {kind} cell must run behind AlignedEnv: A16 makes align_info the only door into the "
            "canonical 25-wide frame, and a CityFlow-trained model handed raw 32-wide SUMO info "
            "would produce a number with no meaning rather than an error"
        )


def dt_choose(
    env: Any,
    *,
    checkpoint_path: str | Path,
    target_rtg: float | Mapping[str, float],
    declared_gradient_steps: int | None = None,
    device: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """The DT's decision function, and the per-decision series a reviewer re-derives from.

    ``BRIEF_36`` E4: **every** decision is ``act(info, explore=False, update_memory=True)``.  The
    default is ``explore=True``, which samples from the masked softmax through an unseeded
    ``torch.multinomial`` -- P7.2b's smoke took that default once and ``n_decisions_in_support``
    moved 271 -> 231 between two runs of the same seed, checkpoint and draw.

    ``DEFERRED`` 81 closes here: the RTG series and the per-decision reward series are both
    recorded, so ``rtg_advanced_every_decision`` is re-derivable by a reviewer instead of being a
    flag they must trust.  The RTG is read BEFORE the call, which is what makes the shift-by-one
    rule in ``transfer_calibration.rtg_advanced_every_decision`` the right comparison.
    """
    import numpy as np

    from offline.rtg_calibration import agent_with_target
    from offline.transfer_calibration import DECLARED_GRADIENT_STEPS

    steps = DECLARED_GRADIENT_STEPS if declared_gradient_steps is None else int(declared_gradient_steps)

    if isinstance(target_rtg, Mapping):
        # P7.3d: the multi-intersection subject, in the spatial checkpoint format (Amendment A1).
        # Sixteen series, keyed by id in the ENV's order (contract C1), and sixteen refusals --
        # fifteen intersections agreeing must not be able to hide the sixteenth.
        from offline.rtg_calibration import spatial_agent_with_targets

        agent = spatial_agent_with_targets(
            env, checkpoint_path, declared_gradient_steps=steps,
            targets=target_rtg, device=device,
        )
        ix_ids = [str(ix.id) for ix in env.intersections]
        diagnostics = {
            "agent": agent,
            "intersections": list(ix_ids),
            "rtg_series": {ix: [] for ix in ix_ids},
            "reward_series": {ix: [] for ix in ix_ids},
            "actions": [],
        }

        def choose_many(_env: Any, info: Mapping[str, Any]) -> Any:
            # Read BEFORE the call, which is what makes Amendment D1's shift-by-one the right
            # comparison: rtg[t] - rtg[t-1] == -reward[t-1], per intersection.
            resting = agent.current_rtg()
            for ix in ix_ids:
                diagnostics["rtg_series"][ix].append(float(resting[ix]))
                payload = info["intersections"][ix]
                diagnostics["reward_series"][ix].append(
                    None if "reward" not in payload else float(payload["reward"])
                )
            action = agent.act(info, explore=False, update_memory=True)
            diagnostics["actions"].append(
                [int(a) for a in np.asarray(action).reshape(-1)]
            )
            return action

        return choose_many, diagnostics

    agent = agent_with_target(
        env, checkpoint_path, declared_gradient_steps=steps, target_rtg=float(target_rtg)
    )
    ix_id = str(list(env.intersections)[0].id)

    diagnostics: dict[str, Any] = {
        "agent": agent,
        "intersection": ix_id,
        "rtg_series": [],
        "reward_series": [],
        "actions": [],
    }

    def choose(_env: Any, info: Mapping[str, Any]) -> Any:
        diagnostics["rtg_series"].append(float(agent.current_rtg()[ix_id]))
        payload = info["intersections"][ix_id]
        diagnostics["reward_series"].append(
            None if "reward" not in payload else float(payload["reward"])
        )
        action = agent.act(info, explore=False, update_memory=True)
        diagnostics["actions"].append(int(np.asarray(action).reshape(-1)[0]))
        return action

    return choose, diagnostics


def anchor_choose(
    env: Any, *, cell: Mapping[str, Any], config_path: str | Path
) -> tuple[Any, dict[str, Any]]:
    """The anchor's decision function, through ``collect.POLICIES`` as P7.1 ran it.

    ``build_policy`` seeds ``default_rng(args.base_seed)``, so ``random``'s five POLICY seeds enter
    there (G6) and vary the action stream ONLY: the engine seed stays ``horizon_rollout``'s
    :data:`ENGINE_SEED` for every cell, which is A18(c)'s rule.  ``fixedtime`` carries
    ``COLLECT_SETTINGS``' ``--fixed-time-k 4``, the schedule P7.1's frozen anchor rows were measured
    with; nothing about the anchors is chosen here.

    It returns the same ``(choose, diagnostics)`` pair :func:`dt_choose` does, and for the same
    reason: the decision COUNT and the action range are properties of the episode that ran, and a
    cell that reported the registered horizon without counting it would satisfy
    :func:`validate_cell_payload` by construction.  An anchor keeps no RTG, so those series stay
    empty rather than being invented.
    """
    import numpy as np

    from offline.sumo_att_reference import build_policy, collect_style_args

    arm = str(cell["arm"])
    policy_seed = cell.get("seed") if arm == "random" else None
    base_seed = ENGINE_SEED if policy_seed is None else int(policy_seed)
    args = collect_style_args(
        "sumo", arm, config_path, episodes=1, base_seed=base_seed, sentinel_out_dir="/nonexistent"
    )
    policy = build_policy(env, args)
    diagnostics: dict[str, Any] = {"rtg_series": [], "reward_series": [], "actions": []}
    # P7.3d (B.6 fix round, F-B6-5): on grid4x4 EVERY intersection's action is recorded, one row
    # per decision, so `actions_in_range` checks sixteen intersections rather than the first. Keyed
    # on the SCENARIO, not on a count (A.1-2): hz1x1 keeps its one int per decision, unchanged.
    per_intersection = scenario_of(cell) == GRID4X4_SCENARIO_KEY

    def choose(_env: Any, info: Mapping[str, Any]) -> Any:
        action = policy(info)
        flat = np.asarray(action).reshape(-1)
        diagnostics["actions"].append(
            [int(a) for a in flat] if per_intersection else int(flat[0])
        )
        return action

    return choose, diagnostics


def att_env_from_info(info: Mapping[str, Any]) -> float:
    """G4: the env's own ATT at the horizon, SUBSCRIPTED -- never defaulted.

    ``horizon_rollout`` reads ``info.get("average_travel_time", 0.0)`` (``horizon_metric.py:99``),
    and that default is the hazard: a missing key would make ``att_horizon`` 0.0, which is not a
    measurement but the absence of one wearing the same type.  P7.1's own expression subscripts the
    key (``sumo_att_reference.py:1242``); this is that expression with its failure named.
    """
    if "average_travel_time" not in info:
        raise ValueError(
            "the final info carries no 'average_travel_time', so att_env cannot be measured. "
            "horizon_rollout would substitute 0.0 here and report it as an ATT; refusing instead, "
            "because a defaulted metric is indistinguishable from a real one once written down"
        )
    return float(info["average_travel_time"])


class _StepTap:
    """Forwards every call, and remembers the last ``info`` ``step`` returned (G4's second route).

    ``horizon_rollout`` keeps no reference to the final ``info``: it appends
    ``info.get("average_travel_time", 0.0)`` and returns the reduction.  The tap gives the cell the
    same ``info`` P7.1's ``_record`` reads, so ``att_env`` and ``att_horizon`` are computed by two
    independent routes and compared under ``==`` rather than one being assumed from the other.

    The policy is built on the UNTAPPED env and only the rollout is tapped, so nothing about agent
    construction or MaxPressure's roadnet introspection passes through this object.
    """

    def __init__(self, env: Any, *, collect_post_step: bool = False) -> None:
        self._env = env
        self.last_info: dict[str, Any] | None = None
        # B.7.1-2: on a grid4x4 cell the tap also keeps every POST-STEP info, the input of the
        # per-intersection return (the reset info is not among them: no reward precedes the first
        # action). Off by default, so an hz1x1 cell runs exactly as it always did.
        self._collect = bool(collect_post_step)
        self.post_step: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        info = self._env.reset(**kwargs)
        self.last_info = info
        return info

    def step(self, action: Any) -> tuple[Any, bool, bool, dict[str, Any]]:
        reward, terminated, truncated, info = self._env.step(action)
        self.last_info = info
        if self._collect:
            self.post_step.append(info)
        return reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        if name in {"_env", "last_info", "_collect", "post_step"}:
            raise AttributeError(name)
        return getattr(self._env, name)


def validate_cell_payload(
    payload: Mapping[str, Any], *, cell: Mapping[str, Any] | None = None
) -> None:
    """Every property a cell must carry, each refusal naming its own reason.

    Run by :func:`run_cell` before a chunk is written and again by :func:`report` over every chunk
    it reads: a chunk can be hand-made, and a resumed campaign can carry chunks written by an older
    revision.  Digest comparisons against files on disk are NOT here -- they need roots, and they
    live in :func:`report` and :func:`chunk_is_reusable`.
    """
    from offline.transfer_calibration import (
        EXPECTED_DECISIONS,
        EXPECTED_TIME_TO_TELEPORT,
        PARITY_VTYPE_ID,
    )

    label = cell_chunk_name(payload)
    # P7.3d (B.6 fix round): the scenario decides the format and the pins. A grid4x4 chunk is its
    # own format (`p7.3d-grid4x4/1.0`, the module docstring's last section); an hz1x1 chunk is
    # checked exactly as it always was.
    grid = scenario_of(payload) == GRID4X4_SCENARIO_KEY
    expected_version = GRID4X4_ARTIFACT_FORMAT_VERSION if grid else ARTIFACT_FORMAT_VERSION
    if payload.get("format_version") != expected_version:
        raise ValueError(
            f"{label}: format_version {payload.get('format_version')!r} is not "
            f"{expected_version!r}"
        )
    if cell is not None:
        # ⚠️ `stage` IS part of the identity (reviewer A's minor 9, closed here because P7.3b made
        # it load-bearing). `cell_chunk_name` derives the name from (subject, arm, seed, draw) and
        # NOT from the stage, so P7.3a's `cell_anchor_fixedtime_seednone_draw1000.json` and the
        # anchor stage's re-rolled denominator share a file name. They live in different work
        # directories, so nothing collides on disk -- but without this key a P7.3a chunk offered to
        # `report --stage anchor` would validate as one of the anchor's own, and 200 of the 700
        # cells would be another campaign's. The name is not changed: P7.3a's chunks are on disk
        # under it and renaming them would make its artifact unregenerable at its own commit.
        # `scenario` joined the identity in the B.6 fix round: an hz1x1 chunk offered to a grid4x4
        # cell (or the reverse) is another campaign's evidence, whatever its name says. On hz1x1
        # both sides lack the key, so every P7.3a / P7.3b comparison is unchanged.
        for key in ("kind", "subject", "arm", "seed", "draw_id", "stage", "scenario"):
            if payload.get(key) != cell.get(key):
                raise ValueError(
                    f"{label}: the chunk says {key}={payload.get(key)!r} and the cell it was asked "
                    f"for says {cell.get(key)!r}; a chunk that describes another cell is not "
                    "evidence about this one"
                )
    arm = str(payload.get("arm"))
    if arm not in DECLARED_ARM_NAMES:
        raise ValueError(
            f"{label}: {arm!r} is not a declared arm. BRIEF_37 §2 closed the set before any cell "
            f"ran ({sorted(DECLARED_ARM_NAMES)}); Rule B at k = 5 and k = 20 are the few-shot "
            "prompts A18(d) attaches to fine-tuned models, and a zero-shot outcome for one of them "
            "is an evaluation nobody registered"
        )
    if grid:
        # A21(a)/(b): the grid4x4 declaration has ONE subject with ONE arm and rho's two anchors.
        # `naive` and `random` are in DECLARED_ARM_NAMES because hz1x1 evaluated them; on this
        # scenario they were removed BY DECLARATION before any grid4x4 SUMO number existed.
        grid_arms = (GRID4X4_ARM, "fixedtime", "maxpressure")
        kind = str(payload.get("kind"))
        if arm not in grid_arms:
            raise ValueError(
                f"{label}: {arm!r} is not an arm of the grid4x4 declaration {list(grid_arms)}; "
                "A21(b) removed `naive` and `random` from this scenario by declaration"
            )
        if kind == "dt" and (str(payload.get("subject")) != GRID4X4_SUBJECT or arm != GRID4X4_ARM):
            raise ValueError(
                f"{label}: a grid4x4 DT cell is {GRID4X4_SUBJECT!r} under {GRID4X4_ARM!r} (A20(a), "
                f"A21(a)), not {payload.get('subject')!r} under {arm!r}"
            )
        if kind == "anchor" and (payload.get("subject") is not None or arm == GRID4X4_ARM):
            raise ValueError(f"{label}: an anchor cell has no subject and no prompt")
    if int(payload["n_teleports"]) != 0:
        raise ValueError(
            f"{label}: {payload['n_teleports']} teleport(s) under A15(c)'s teleport-free regime"
        )
    if str(payload["time_to_teleport_option"]) != EXPECTED_TIME_TO_TELEPORT:
        raise ValueError(
            f"{label}: SUMO reported time-to-teleport {payload['time_to_teleport_option']!r}, not "
            f"{EXPECTED_TIME_TO_TELEPORT!r}; this is not the teleport-free regime A15(c) registers"
        )
    if list(payload["vehicle_types_seen"]) != [PARITY_VTYPE_ID]:
        raise ValueError(
            f"{label}: the engine ran {payload['vehicle_types_seen']!r}, not [{PARITY_VTYPE_ID!r}]"
        )
    if int(payload["decisions"]) != EXPECTED_DECISIONS:
        raise ValueError(
            f"{label}: {payload['decisions']} decisions, not {EXPECTED_DECISIONS}; the episode did "
            "not run to the horizon"
        )
    if payload.get("actions_in_range") is not True:
        raise ValueError(f"{label}: an action was outside the intersection's legal range")
    if grid:
        # The action MATRIX a grid4x4 chunk records (F-B6-5): one row per decision, one column per
        # intersection in `intersection_ids` order. `actions_in_range` above was computed over all
        # of it; this checks the shape it was computed over.
        ids = payload.get("intersection_ids")
        actions = payload.get("actions")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{label}: intersection_ids {ids!r} is not a list of distinct ids")
        if (
            not isinstance(actions, list)
            or len(actions) != int(payload["decisions"])
            or any(not isinstance(row, list) or len(row) != len(ids) for row in actions)
        ):
            raise ValueError(
                f"{label}: the action matrix is not {payload['decisions']} decisions x "
                f"{len(ids)} intersections"
            )
        # B.7.1-2: the per-intersection episode return, by the probe's two routes, on EVERY grid4x4
        # chunk -- the anchors' as well as the DT's -- re-checked here at consumption.
        for key in ("local_return", "local_return_from_lanes"):
            value = payload.get(key)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{label}: a grid4x4 chunk records local_return and local_return_from_lanes per "
                    f"intersection (B.7.1-2); {key} is {type(value).__name__}"
                )
            if sorted(value) != sorted(ids):
                raise ValueError(
                    f"{label}: {key} covers {sorted(value)[:4]}..., not exactly the recorded "
                    "intersection_ids"
                )
        split = sorted(
            ix for ix in ids if payload["local_return"][ix] != payload["local_return_from_lanes"][ix]
        )
        if split:
            raise ValueError(
                f"{label}: the two per-intersection return routes disagree on {split[:4]}; the "
                "return is recorded only where both routes agree under =="
            )
    # The calibration the cell's prompts came from, per scenario: P7.2b's artifact for hz1x1,
    # P7.3d's per-intersection artifact for grid4x4 (B.6-2(2); at 8c79778 a grid4x4 chunk would have
    # claimed hz1x1's calibration).
    expected_calibration = P7_3D_CALIBRATION_SHA256 if grid else P7_2B_CALIBRATION_SHA256
    if str(payload.get("calibration_sha256")) != expected_calibration:
        raise ValueError(
            f"{label}: it records targets from calibration sha256 "
            f"{payload.get('calibration_sha256')!r}, not the pinned {expected_calibration!r}"
        )
    # Amendment A5, checked at consumption as well as at production: an anchor cell must name the
    # committed training record its prompt came from, and NOTHING ELSE may name it. The second
    # half matters as much as the first -- a CityFlow subject's cell carrying this field would be
    # a cell whose prompt source cannot be told from the anchor's.
    is_anchor_subject = str(payload.get("subject")) == ANCHOR_SUBJECT
    recorded_training = payload.get("anchor_training_sha256")
    if is_anchor_subject and recorded_training != P7_3B_TRAINING_SHA256:
        raise ValueError(
            f"{label}: it records anchor_training_sha256 {recorded_training!r}, not the pinned "
            f"{P7_3B_TRAINING_SHA256!r}. The anchor's prompt and its checkpoint digest both come "
            f"from {P7_3B_TRAINING_NAME}, and a cell that cannot name that record is not evidence "
            "about the anchor A18(a) registers"
        )
    if not is_anchor_subject and recorded_training is not None:
        raise ValueError(
            f"{label}: subject {payload.get('subject')!r} records anchor_training_sha256 "
            f"{recorded_training!r}. Only {ANCHOR_SUBJECT!r} is prompted from "
            f"{P7_3B_TRAINING_NAME}"
        )
    if is_anchor_subject and str(payload.get("arm")) != ANCHOR_ARM:
        raise ValueError(
            f"{label}: {ANCHOR_SUBJECT} ran arm {payload.get('arm')!r}. A18(a) gives the anchor "
            f"exactly one prompt, the naive in-domain rule ({ANCHOR_ARM!r}); a calibrated arm on "
            "the anchor is an evaluation nobody registered"
        )
    if int(payload["engine_seed_requested"]) != ENGINE_SEED:
        raise ValueError(
            f"{label}: engine_seed_requested {payload['engine_seed_requested']} is not A18(c)'s "
            f"{ENGINE_SEED}"
        )

    # ---- J1(b): provenance, checked where it is consumed -------------------------------------
    # Every chunk has recorded these since section 3.5b and NOTHING read them: the pilot's 33 real
    # chunks all carry `git_dirty: true`. A cell rolled from a tree with uncommitted changes cannot
    # be attributed to any commit, so it is not evidence about the code the artifact names.
    if payload.get("git_dirty") is not False:
        raise ValueError(
            f"{label}: git_dirty is {payload.get('git_dirty')!r}, not False. This cell was rolled "
            "from a tree with uncommitted changes, so the code that produced it is not the code "
            "any commit names. Run the campaign from the dedicated worktree (J1(e))"
        )
    commit = payload.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or not all(
        character in "0123456789abcdef" for character in commit.lower()
    ):
        raise ValueError(
            f"{label}: git_commit {commit!r} is not a 40-character hex commit; a cell whose "
            "provenance cannot be resolved to a revision is not evidence about one"
        )

    checked = bool(payload.get("halting_checked"))
    if checked != halting_check_for(int(payload["draw_id"])):
        raise ValueError(
            f"{label}: halting_checked is {checked} on draw {payload['draw_id']}, but Amendment C2 "
            f"declares the cross-check ON for draw {HALTING_CHECK_DRAW} and OFF elsewhere"
        )
    if checked and int(payload["halting_n_disagreeing_lane_seconds"]) != 0:
        raise ValueError(
            f"{label}: the recorder's halting classification disagreed with SUMO's own on "
            f"{payload['halting_n_disagreeing_lane_seconds']} lane-second(s). Amendment C2: a "
            "recorder disagreement is a finding that stops the campaign, exactly as a teleport is"
        )

    if str(payload.get("kind")) == "dt" and grid:
        # THE 16-ID REFUSAL, with a call site at last (B.6-2(2), the pre-flight's M3): reached from
        # run_cell on the finished chunk, from chunk_is_reusable, and from report over every chunk.
        targets = payload.get("target_rtg")
        series = payload.get("rtg_series")
        rewards = payload.get("reward_series")
        if not all(isinstance(value, Mapping) for value in (targets, series, rewards)):
            raise ValueError(
                f"{label}: a grid4x4 DT chunk records target_rtg, rtg_series and reward_series PER "
                "INTERSECTION, as mappings keyed by id"
            )
        if sorted(targets) != sorted(payload["intersection_ids"]):
            raise ValueError(
                f"{label}: the targets cover {sorted(targets)[:4]}..., not exactly the recorded "
                "intersection_ids (A17(e) applies the rule per intersection)"
            )
        if payload.get("rtg_first") != targets:
            first = payload.get("rtg_first") or {}
            differing = sorted(ix for ix in targets if first.get(ix) != targets[ix])
            raise ValueError(
                f"{label}: rtg_first is not the declared target on {differing[:4]}; the prompt did "
                "not take effect there"
            )
        assert_rtg_first_matches_targets(series, targets, label=label)
        uneven = sorted(
            ix
            for ix in targets
            if ix not in rewards
            or len(series[ix]) != len(rewards[ix])
            or len(series[ix]) != int(payload["decisions"])
        )
        if uneven:
            raise ValueError(
                f"{label}: intersection(s) {uneven[:4]} do not record one RTG and one reward per "
                "decision (DEFERRED 81: both are stored so a reviewer re-derives D1's rule)"
            )
    elif str(payload.get("kind")) == "dt":
        if payload.get("rtg_first") != payload.get("target_rtg"):
            raise ValueError(
                f"{label}: rtg_first {payload.get('rtg_first')!r} is not the declared target "
                f"{payload.get('target_rtg')!r}; the prompt did not take effect and the cell ran "
                "some other conditioning"
            )
        series = payload.get("rtg_series")
        rewards = payload.get("reward_series")
        if not series or rewards is None or len(series) != len(rewards):
            raise ValueError(
                f"{label}: the RTG series and the reward series are recorded per decision and must "
                "agree in length (DEFERRED 81: both are stored so a reviewer re-derives the "
                "advance rule instead of trusting a flag)"
            )


def chunk_path(cell: Mapping[str, Any], *, work_dir: str | Path) -> Path:
    """Pure path arithmetic: where one cell's chunk lives."""
    return Path(work_dir) / cell_chunk_name(cell)


def write_chunk(payload: Mapping[str, Any], *, work_dir: str | Path) -> Path:
    """E3(d): atomic tmp-then-replace, so a chunk is either whole or absent."""
    path = chunk_path(payload, work_dir=work_dir)
    _write_json(path, payload)
    return path


def move_aside(path: str | Path) -> Path:
    """Move an unusable chunk into ``failed/`` rather than overwriting it.

    A chunk that failed its own re-validation is evidence about a run, and ``report``'s glob must
    not see it.  Never overwritten: the suffix grows instead.
    """
    source = Path(path)
    destination = source.parent / "failed" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{source.stem}.{suffix}{source.suffix}")
    source.replace(destination)
    return destination


def chunk_is_reusable(
    payload: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
) -> bool:
    """E3(d): may a restart SKIP this cell?  Only on evidence RE-DERIVED from disk.

    ``offline/campaigns/p5_3b.sh`` skipped on ``[ -f ]`` alone and a bad chunk survived every
    restart; ``nortg_decomposition.chunk_is_reusable`` records what that cost.  The stored verdict
    is exactly what a half-written or hand-edited chunk would lie about, so the checkpoint's, the
    cfg's and the routes file's digests are recomputed from the files themselves and compared with
    what the chunk claims (G2: a cfg digest alone would not notice a regenerated routes file).

    Any disagreement, and any chunk this function cannot read, returns ``False``: the cell is
    re-run.  A file we cannot read is a file we have no evidence about, and a predicate that
    crashed on it would take the driver down on every restart.
    """
    if not isinstance(payload, Mapping):
        return False
    try:
        validate_cell_payload(payload, cell=cell)
        # J1(c): rolled by THIS code, or re-rolled. A docs-only difference is not a difference in
        # the computation. `_git` raises on an unknown revision and that exception is deliberately
        # NOT caught below: a git that cannot answer is an environment failure, not a verdict about
        # this chunk, and the campaign must stop rather than silently re-roll 4,700 cells.
        if code_changed_since(str(payload["git_commit"])):
            return False
        # B.6-2(2): the SAME scenario-aware calls run_cell and report make. At 8c79778 this line
        # re-derived hz1x1's demand for a grid4x4 cell and found a chunk carrying hz1x1's digests
        # CONSISTENT -- the pre-flight's sandbox reused it.
        demand = demand_identity_for(cell, out_root=out_root)
        if str(payload["config_sha256"]) != demand["config_sha256"]:
            return False
        if str(payload["routes_sha256"]) != demand["routes_sha256"]:
            return False
        if str(cell["kind"]) == "dt":
            identity = checkpoint_identity_for(cell, output_root=output_root, data_dir=data_dir)
            if str(payload["checkpoint_sha256"]) != identity["file_sha256"]:
                return False
            # A grid4x4 chunk's 16 prompts, re-derived from the digest-pinned artifact: a chunk
            # conditioned on other targets is internally consistent, so validate cannot see it.
            if scenario_of(cell) == GRID4X4_SCENARIO_KEY and dict(payload["target_rtg"]) != (
                load_grid4x4_targets(data_dir=data_dir)
            ):
                return False
    except (KeyError, TypeError, ValueError, AttributeError, FileNotFoundError):
        return False
    return True


def reusable_chunk_at(path: str | Path, **kwargs: Any) -> bool:
    """:func:`chunk_is_reusable` over a file that may not exist or may not parse."""
    target = Path(path)
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_bytes())
    except (ValueError, OSError):
        return False
    return chunk_is_reusable(payload, **kwargs)


def run_cell(
    cell: Mapping[str, Any],
    *,
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
    calibration: Mapping[str, Any] | None = None,
    canary_seconds: float | None = None,
) -> dict[str, Any]:
    """One observed SUMO episode for one cell, as a validated chunk.  It writes NOTHING.

    The order is the design: the demand and the checkpoint are pinned BEFORE a simulator starts, the
    env is built and its shape refused if it does not match the cell (C8), the policy is built on
    the untapped env, the rollout runs on the tap (G4), and the payload is validated in full before
    it is returned.  The caller writes it -- so a cell that fails leaves no chunk, no directory and
    no trace except the exception.

    **It branches on ``cell["scenario"]``** (B.6 fix round, B.6-2(2)).  An hz1x1 cell -- no
    ``scenario`` key -- takes exactly the path it always took and returns exactly the payload it
    always returned.  A grid4x4 cell reads its demand through :func:`demand_identity_for`, its
    checkpoint through :func:`checkpoint_identity_for` (A20(a)'s pin), its 16 prompts through
    :func:`load_grid4x4_targets` and their support ranges through :func:`grid4x4_support_ranges` --
    both from the ONE digest-pinned ``p7_3d_calibration.json`` -- loads the subject through
    ``dt_choose``'s per-intersection branch (``spatial_agent_with_targets``), and returns a
    ``p7.3d-grid4x4/1.0`` chunk (the module docstring's last section), whose 16-id refusal runs in
    :func:`validate_cell_payload` on the finished payload before it is returned.
    """
    import time

    from agent.utils.utils import Utils
    from offline.horizon_metric import horizon_rollout
    from offline.rtg_calibration import in_support_counts
    from offline.sumo_att_reference import reconstruct_sumo_episode
    from offline.transfer_calibration import rtg_advanced_every_decision, subject_facts

    cell = dict(cell)
    kind = str(cell["kind"])
    draw_id = int(cell["draw_id"])
    arm = str(cell["arm"])
    grid = scenario_of(cell) == GRID4X4_SCENARIO_KEY

    demand = demand_identity_for(cell, out_root=out_root)
    checkpoint: dict[str, Any] | None = None
    target_rtg: float | dict[str, float] | None = None
    facts = None
    grid_support: dict[str, tuple[float, float]] | None = None
    anchor_training_sha256: str | None = None
    if kind == "dt" and grid:
        # P7.3d: the registered subject under its ONE registered arm (A20(a), A21(a)); the 16
        # prompts and their support ranges READ from the digest-pinned artifact, never recomputed.
        if str(cell["subject"]) != GRID4X4_SUBJECT or arm != GRID4X4_ARM:
            raise ValueError(
                f"a grid4x4 DT cell is {GRID4X4_SUBJECT!r} under {GRID4X4_ARM!r}, not "
                f"{cell['subject']!r} under {arm!r}"
            )
        target_rtg = load_grid4x4_targets(data_dir=data_dir)
        grid_support = grid4x4_support_ranges(data_dir=data_dir)
        checkpoint = checkpoint_identity_for(cell, output_root=output_root, data_dir=data_dir)
    elif kind == "dt":
        subject = str(cell["subject"])
        if subject == ANCHOR_SUBJECT:
            # ⚠️ AMENDMENT A5, AND IT IS A ROUTE RATHER THAN A VALUE. The anchor's prompt is the
            # naive in-domain rule over its OWN 200 SUMO episodes, read from section 3.3's
            # committed record. `load_calibration` and `targets_for_subject` are NOT reached on
            # this branch at all -- P7.2b's artifact registers CityFlow-trained subjects, and a
            # target taken from it would be conditioning the anchor on the source domain's
            # returns. The test monkeypatches both to raise and requires this cell to run anyway.
            training = load_anchor_training(_data_dir(data_dir) / P7_3B_TRAINING_NAME)
            anchor_training_sha256 = P7_3B_TRAINING_SHA256
            target_rtg = float(training["target_rtg"])
            facts = anchor_subject_facts(training)
        else:
            # J3 (reviewer min-6): through data_dir, as the checkpoint pins already are. A run
            # pointed at a different --data-dir would otherwise pin its checkpoints against one
            # set of records and read its prompts from another.
            artifact = (
                load_calibration(_data_dir(data_dir) / P7_2B_CALIBRATION_NAME)
                if calibration is None
                else calibration
            )
            target_rtg = float(targets_for_subject(subject, artifact)[arm]["target_rtg"])
            facts = subject_facts(subject, output_root=output_root)
        checkpoint = checkpoint_identity_for(cell, output_root=output_root, data_dir=data_dir)

    started = time.perf_counter()
    env = env_for_cell(cell, out_root=out_root)
    diagnostics: dict[str, Any] = {"rtg_series": [], "reward_series": [], "actions": []}
    try:
        # ⚠️ FIRST, before a policy is built or a single step is taken (reviewer MIN-1). The
        # refusal's purpose is to refuse BEFORE a SUMO process has run an episode; a mutant that
        # moved it after the rollout still raised, and T5 stayed green, which is why the ORDER is
        # now pinned by a test rather than by this comment.
        assert_env_matches_cell(cell, env)
        # The action bound is the ENV's, through the helper CLAUDE.md rule 5 requires
        # (Utils.infer_action_counts falls back to ix.num_phases). hz1x1 has 8 phases, so the
        # literal 8 this replaces was right here and silently wrong anywhere else -- a dimension
        # assumption P7.3b would have inherited.
        intersections = list(env.intersections)
        action_counts = Utils.infer_action_counts(getattr(env, "action_space", None), intersections)
        n_actions = int(min(action_counts))
        if kind == "dt":
            choose, diagnostics = dt_choose(
                env, checkpoint_path=checkpoint["path"], target_rtg=target_rtg  # type: ignore[index]
            )
        else:
            choose, diagnostics = anchor_choose(env, cell=cell, config_path=demand["config_path"])
        # B.7.1-2: a grid4x4 cell also keeps its POST-STEP infos (hz1x1: off, unchanged).
        tap = _StepTap(env, collect_post_step=grid)
        rollout = horizon_rollout(tap, choose, 1, ENGINE_SEED)
        built = reconstruct_sumo_episode(env.recorder)
        att_env = att_env_from_info(tap.last_info or {})
        types_seen = sorted({env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()})
        option = str(env._sumo.simulation.getOption("time-to-teleport"))
        engine_seed_drawn = int(env._engine_seed)
        # The second route's lanes, read while the env is open, in the frame the infos use.
        return_lanes = (
            grid4x4_return_lanes(intersections, tap.post_step[0]) if grid and tap.post_step else None
        )
    finally:
        env.close()
    seconds = time.perf_counter() - started

    if att_env != rollout.att_horizon:
        raise ValueError(
            f"att_env {att_env!r} and att_horizon {rollout.att_horizon!r} disagree. They are two "
            "routes to ONE quantity -- the final info's average_travel_time, and horizon_rollout's "
            "last sample of the same key (G4) -- so a difference means the loop this cell ran is "
            "not the loop P7.1 measured, and no comparison with the frozen anchors would be valid"
        )

    if grid:
        if return_lanes is None or len(tap.post_step) != len(diagnostics["actions"]):
            raise ValueError(
                f"{cell_chunk_name(cell)}: {len(tap.post_step)} post-step info(s) for "
                f"{len(diagnostics['actions'])} decision(s); the per-intersection return is the sum "
                "over one post-step info per decision (B.7.1-2)"
            )
        return _grid4x4_payload(
            cell,
            diagnostics=diagnostics,
            ix_ids=[str(ix.id) for ix in intersections],
            n_actions=n_actions,
            rollout=rollout,
            built=built,
            att_env=att_env,
            types_seen=types_seen,
            option=option,
            engine_seed_drawn=engine_seed_drawn,
            demand=demand,
            checkpoint=checkpoint,
            targets=target_rtg,  # type: ignore[arg-type]
            support_ranges=grid_support,
            local_returns=per_intersection_local_returns(tap.post_step, return_lanes),
            canary_seconds=canary_seconds,
            seconds=seconds,
        )

    actions = diagnostics["actions"]
    rtg_series = diagnostics["rtg_series"]
    reward_series = diagnostics["reward_series"]
    support = None
    if kind == "dt" and rtg_series and facts is not None:
        counts = in_support_counts(
            rtg_series,
            rtg_min=facts.support_range_over_the_split[0],
            rtg_max=facts.support_range_over_the_split[1],
        )
        support = {
            "n_decisions_in_support": counts.in_support,
            "n_decisions_below": counts.below,
            "n_decisions_above": counts.above,
            "support_range": list(facts.support_range_over_the_split),
            "training_set_return_min": facts.training_set_return_min,
        }

    checked = halting_check_for(draw_id)
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        **cell,
        "policy_seed": int(cell["seed"]) if kind == "anchor" and arm == "random" else None,
        "engine_seed_requested": ENGINE_SEED,
        "engine_seed_drawn": engine_seed_drawn,
        # Counted, for both kinds: a decision count taken from the declared horizon rather than
        # from the episode would make validate_cell_payload's check a tautology.
        "decisions": len(actions),
        "actions_in_range": bool(actions)
        and all(0 <= int(action) < n_actions for action in actions),
        "action_space_n": n_actions,
        "episode_reward": rollout.episode_reward,
        "att_horizon": rollout.att_horizon,
        "att_env": att_env,
        "att_running_mean": rollout.att_running_mean,
        "horizon_vehicle_count": rollout.final_vehicle_count,
        "e_sumo": built.e_sumo.value,
        "p_sumo": built.p_sumo.value,
        "w_sumo": built.w_sumo.value,
        "mean_depart_delay": built.mean_depart_delay,
        "n_created": built.e_sumo.n_ids,
        "n_entered": built.n_departed,
        "n_never_entered": built.n_never_inserted,
        "n_pending_at_horizon": built.n_pending_at_horizon,
        "n_teleports": built.n_teleports,
        "n_vanished_without_arrival": built.n_vanished_without_arrival,
        "n_arrived_never_observed_at_a_boundary": built.n_arrived_never_observed_at_a_boundary,
        "max_abs_depart_clock_deviation": built.max_abs_depart_clock_deviation,
        "n_observations": built.n_observations,
        "vehicle_types_seen": types_seen,
        "time_to_teleport_option": option,
        "halting_checked": checked,
        "halting_max_abs_difference": built.halting.max_abs_difference if checked else None,
        "halting_n_lane_seconds": built.halting.n_lane_seconds if checked else None,
        "halting_n_disagreeing_lane_seconds": (
            built.halting.n_disagreeing_lane_seconds if checked else None
        ),
        "config_sha256": demand["config_sha256"],
        "routes_sha256": demand["routes_sha256"],
        "calibration_sha256": P7_2B_CALIBRATION_SHA256,
        # None on every cell but the anchor's, and on the anchor's it is the record its prompt and
        # its checkpoint digest BOTH came from. Recorded per cell as evidence, checked at
        # consumption by validate_cell_payload, and published once in the artifact's `inputs`.
        "anchor_training_sha256": anchor_training_sha256,
        "checkpoint": None if checkpoint is None else checkpoint["path"],
        "checkpoint_sha256": None if checkpoint is None else checkpoint["file_sha256"],
        "sha256_checked_against": None if checkpoint is None else checkpoint["sha256_checked_against"],
        "target_rtg": target_rtg,
        "rtg_first": rtg_series[0] if rtg_series else None,
        "rtg_last": rtg_series[-1] if rtg_series else None,
        "rtg_series": rtg_series or None,
        "reward_series": reward_series or None,
        "rtg_advanced_every_decision": (
            rtg_advanced_every_decision(rtg_series, reward_series) if rtg_series else None
        ),
        "n_decisions_in_support": None if support is None else support["n_decisions_in_support"],
        "support_range": None if support is None else support["support_range"],
        "in_support_counts": support,
        "canary_seconds": None if canary_seconds is None else float(canary_seconds),
        "seconds": seconds,
        **_git_provenance(),
    }
    validate_cell_payload(payload, cell=cell)
    return payload


def _grid4x4_payload(
    cell: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any],
    ix_ids: Sequence[str],
    n_actions: int,
    rollout: Any,
    built: Any,
    att_env: float,
    types_seen: Sequence[str],
    option: str,
    engine_seed_drawn: int,
    demand: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    targets: Mapping[str, float] | None,
    support_ranges: Mapping[str, tuple[float, float]] | None,
    local_returns: tuple[Mapping[str, float], Mapping[str, float]],
    canary_seconds: float | None,
    seconds: float,
) -> dict[str, Any]:
    """A grid4x4 cell's chunk (``p7.3d-grid4x4/1.0``), validated before it is returned.

    ``local_returns`` is :func:`per_intersection_local_returns`' pair over this episode's post-step
    infos -- recorded on the anchor cells AND the DT cells (B.7.1-2), because per-intersection rho
    divides one by the other and both sides must be the same 360-decision sum.

    The episode-level fields are the hz1x1 chunk's, computed from the same objects.  What differs
    is that every per-intersection quantity is a mapping keyed by id -- the RTG and reward series
    (D1's shift-by-one per id), ``rtg_first`` / ``rtg_last``, the in-support counts against that
    intersection's own range, ``rtg_advanced_every_decision`` -- and that the full action matrix is
    recorded, one row per decision in ``intersection_ids`` order, so ``actions_in_range`` is
    computed over all sixteen intersections and a re-roll can be compared action by action
    (B.5-1).  ``calibration_sha256`` is P7.3d's artifact, never P7.2b's.
    """
    from offline.rtg_calibration import in_support_counts
    from offline.transfer_calibration import rtg_advanced_every_decision

    kind = str(cell["kind"])
    draw_id = int(cell["draw_id"])
    actions = [list(row) for row in diagnostics["actions"]]
    per_ix: dict[str, Any] = {
        "target_rtg": None,
        "rtg_first": None,
        "rtg_last": None,
        "rtg_series": None,
        "reward_series": None,
        "rtg_advanced_every_decision": None,
        "n_decisions_in_support": None,
        "support_range": None,
        "in_support_counts": None,
    }
    if kind == "dt":
        if targets is None or support_ranges is None:
            raise ValueError("a grid4x4 DT cell needs its 16 targets and their support ranges")
        rtg_series = {ix: list(diagnostics["rtg_series"][ix]) for ix in ix_ids}
        reward_series = {ix: list(diagnostics["reward_series"][ix]) for ix in ix_ids}
        counts = {
            ix: in_support_counts(
                rtg_series[ix], rtg_min=support_ranges[ix][0], rtg_max=support_ranges[ix][1]
            )
            for ix in ix_ids
        }
        per_ix = {
            "target_rtg": {ix: float(targets[ix]) for ix in ix_ids},
            "rtg_first": {ix: rtg_series[ix][0] if rtg_series[ix] else None for ix in ix_ids},
            "rtg_last": {ix: rtg_series[ix][-1] if rtg_series[ix] else None for ix in ix_ids},
            "rtg_series": rtg_series,
            "reward_series": reward_series,
            "rtg_advanced_every_decision": {
                ix: rtg_advanced_every_decision(rtg_series[ix], reward_series[ix]) for ix in ix_ids
            },
            "n_decisions_in_support": {ix: counts[ix].in_support for ix in ix_ids},
            "support_range": {ix: list(support_ranges[ix]) for ix in ix_ids},
            "in_support_counts": {
                ix: {
                    "in_support": counts[ix].in_support,
                    "below": counts[ix].below,
                    "above": counts[ix].above,
                }
                for ix in ix_ids
            },
        }

    checked = halting_check_for(draw_id)
    payload: dict[str, Any] = {
        "format_version": GRID4X4_ARTIFACT_FORMAT_VERSION,
        **dict(cell),
        "policy_seed": None,
        "engine_seed_requested": ENGINE_SEED,
        "engine_seed_drawn": engine_seed_drawn,
        "decisions": len(actions),
        "intersection_ids": list(ix_ids),
        "actions": actions,
        "actions_in_range": bool(actions)
        and all(
            len(row) == len(ix_ids) and all(0 <= int(a) < n_actions for a in row) for row in actions
        ),
        "action_space_n": n_actions,
        "episode_reward": rollout.episode_reward,
        "att_horizon": rollout.att_horizon,
        "att_env": att_env,
        "att_running_mean": rollout.att_running_mean,
        "horizon_vehicle_count": rollout.final_vehicle_count,
        "e_sumo": built.e_sumo.value,
        "p_sumo": built.p_sumo.value,
        "w_sumo": built.w_sumo.value,
        "mean_depart_delay": built.mean_depart_delay,
        "n_created": built.e_sumo.n_ids,
        "n_entered": built.n_departed,
        "n_never_entered": built.n_never_inserted,
        "n_pending_at_horizon": built.n_pending_at_horizon,
        "n_teleports": built.n_teleports,
        "n_vanished_without_arrival": built.n_vanished_without_arrival,
        "n_arrived_never_observed_at_a_boundary": built.n_arrived_never_observed_at_a_boundary,
        "max_abs_depart_clock_deviation": built.max_abs_depart_clock_deviation,
        "n_observations": built.n_observations,
        "vehicle_types_seen": list(types_seen),
        "time_to_teleport_option": option,
        "halting_checked": checked,
        "halting_max_abs_difference": built.halting.max_abs_difference if checked else None,
        "halting_n_lane_seconds": built.halting.n_lane_seconds if checked else None,
        "halting_n_disagreeing_lane_seconds": (
            built.halting.n_disagreeing_lane_seconds if checked else None
        ),
        "config_sha256": demand["config_sha256"],
        "routes_sha256": demand["routes_sha256"],
        "calibration_sha256": P7_3D_CALIBRATION_SHA256,
        "anchor_training_sha256": None,
        "checkpoint": None if checkpoint is None else checkpoint["path"],
        "checkpoint_sha256": None if checkpoint is None else checkpoint["file_sha256"],
        "sha256_checked_against": (
            None if checkpoint is None else list(checkpoint["sha256_checked_against"])
        ),
        **per_ix,
        "local_return": {ix: float(local_returns[0][ix]) for ix in ix_ids},
        "local_return_from_lanes": {ix: float(local_returns[1][ix]) for ix in ix_ids},
        "canary_seconds": None if canary_seconds is None else float(canary_seconds),
        "seconds": seconds,
        **_git_provenance(),
    }
    validate_cell_payload(payload, cell=cell)
    return payload


#: The only chunk fields that may reach ``docs/data/``.  A whitelist, not a blacklist: a field added
#: to the chunk later is excluded by default rather than by being remembered.
_PUBLISHED_FIELDS: tuple[str, ...] = (
    "kind", "subject", "arm", "seed", "policy_seed", "draw_id", "stage", "action_space_n",
    "att_env", "att_horizon", "att_running_mean", "e_sumo", "p_sumo", "w_sumo",
    "mean_depart_delay", "episode_reward", "horizon_vehicle_count", "decisions",
    "engine_seed_requested", "engine_seed_drawn",
    "n_created", "n_entered", "n_never_entered", "n_pending_at_horizon", "n_teleports",
    "n_vanished_without_arrival", "n_arrived_never_observed_at_a_boundary",
    "max_abs_depart_clock_deviation", "vehicle_types_seen", "time_to_teleport_option",
    "halting_checked", "halting_max_abs_difference", "halting_n_lane_seconds",
    "halting_n_disagreeing_lane_seconds",
    "target_rtg", "rtg_first", "rtg_last", "rtg_series", "reward_series",
    "rtg_advanced_every_decision", "n_decisions_in_support", "support_range",
    "checkpoint", "checkpoint_sha256", "sha256_checked_against",
    "config_sha256", "routes_sha256", "calibration_sha256",
    "canary_seconds", "seconds", "git_commit", "git_dirty",
)


def _read_canary_record(work: Path) -> dict[str, Any]:
    """Read ``<work>/canary.json`` and RE-RUN the correctness half on its facts.

    Amendment E1.4: the record is a CLAIM, and a claim is not evidence -- ``report`` checks the
    facts itself rather than trusting whatever wrote the file.  A missing file is a refusal, not a
    fallback: falling back on the chunks' own ``canary_seconds`` is precisely how P7.2b's run 3
    published run 1's canary as its own.
    """
    from offline.transfer_calibration import CANARY_RECORD_NAME, check_canary

    path = work / CANARY_RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: this run wrote no {CANARY_RECORD_NAME}. The driver writes it from the canary "
            "line immediately after the token is consumed; a work directory without one is not a "
            "run, and the chunks' canary_seconds is the canary of whoever ROLLED them"
        )
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict) or not isinstance(record.get("facts"), dict):
        raise ValueError(f"{CANARY_RECORD_NAME}: the canary record has no 'facts' object")
    check_canary(record["facts"])
    return {**record, "seconds": float(record["seconds"])}


def anchor_denominator(anchors: Mapping[str, Mapping[str, Any]], key: str) -> float:
    """``ATT_fixedtime - ATT_maxpressure`` on one definition, for one draw."""
    return float(anchors["fixedtime"][key]) - float(anchors["maxpressure"][key])


def denominator_diagnostic(denominators: Mapping[int, float]) -> dict[str, Any]:
    """What rho's denominator looked like across the pool, on one ATT definition.

    The author's ruling of 2026-09-17.  rho is a per-draw ratio, so a denominator near zero makes
    that draw's ratio enormous and a denominator below zero flips its sign; on the CO-REPORTED env
    definition P7.1's frozen anchors put the difference between -1.50 and +18.70 s, which is small
    enough for a handful of draws to dominate the mean.  This block is what :data:`ATT_ENV_CAVEAT`
    tells a reader to read that definition's rho beside, and it is computed from the ANCHOR chunks
    rather than from any arm's outcome.
    """
    values = [float(value) for value in denominators.values()]
    non_positive = sorted(draw for draw, value in denominators.items() if float(value) <= 0.0)
    return {
        "what_this_is": (
            "ATT_fixedtime - ATT_maxpressure per draw, from the anchor cells of that draw; rho's "
            "denominator. Near zero the ratio is dominated by that draw; below zero it flips sign"
        ),
        "n_draws": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "n_non_positive": len(non_positive),
        "n_below_one_second": sum(1 for value in values if abs(value) < 1.0),
        "draw_ids_non_positive": non_positive,
    }


def _rho_pair(
    chunk: Mapping[str, Any], anchors: Mapping[str, Mapping[str, Any]]
) -> dict[str, float | None]:
    """rho under BOTH registered definitions, against the anchors OF THE SAME DRAW.

    ⚠️ **The two definitions differ in what an exactly-zero denominator means, and the author ruled
    on it before any campaign number existed** (2026-09-17).  On the CO-REPORTED env definition a
    zero is reachable -- P7.1 measured -1.50 s at seed 1000, the seed every cell here uses -- so
    that draw's env rho is ``None``, the draw is recorded, and ``report`` carries on.  On the
    REGISTERED PRIMARY it is not: the same anchors differ by 206.5-252.6 s there, so a zero would
    be a finding about the instrument, and :func:`rho` keeps its refusal.
    """
    pair: dict[str, float | None] = {}
    for key in ("e_sumo", "att_env"):
        denominator = anchor_denominator(anchors, key)
        if key == "att_env" and denominator == 0.0:
            pair["rho_att_env"] = None
            continue
        pair[f"rho_{key}"] = rho(
            float(chunk[key]),
            float(anchors["fixedtime"][key]),
            float(anchors["maxpressure"][key]),
        )
    return pair


def declarations_for(
    stage: str | None, cells: Sequence[Mapping[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(every declared cell of ANY stage, this stage's slice)`` -- the pair ``report`` checks against.

    Extracted so the campaign's own path is reachable by a test.  With ``cells=None`` -- which is
    what the driver passes and what a committed artifact requires -- nothing else in ``report``
    exercises it, and a mutation collapsing the two branches SURVIVED the whole suite until this
    existed.

    **Amendment A1 lives here.**  ``declared_cells(None)`` is P7.3a's 4,700 and stays that way, so
    the anchor stage needs its own "every declared cell of ANY stage" set rather than being folded
    into the campaign's.  The two declarations are disjoint by construction and live in separate
    work directories; the completeness check and the "undeclared cell" refusal are only as good as
    the declaration they compare against, so getting this pair wrong is how a chunk nobody declared
    would reach an artifact.
    """
    if cells is not None:
        supplied = [dict(cell) for cell in cells]
        return supplied, supplied
    # B.6-2(3), the pre-flight's B3: the grid4x4 stage is its OWN whole declaration, exactly as the
    # anchor stage is. At 8c79778 it fell through to `None` -- hz1x1's 4,700 -- so a COMPLETE grid4x4
    # campaign would have been refused as 700 chunks "not declared cells of ANY stage" (measured
    # overlap 0: every grid4x4 name carries the scenario prefix).
    if stage in (STAGE_ANCHOR, STAGE_GRID4X4):
        whole: str | None = stage
    else:
        whole = None
    return list(declared_cells(whole)), list(declared_cells(stage))


def report(
    *,
    work_dir: str | Path,
    out_path: str | Path,
    output_root: str | Path,
    out_root: str | Path,
    data_dir: str | Path | None = None,
    stage: str | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
    stage1_path: str | Path | None = None,
) -> dict[str, Any]:
    """E3(e)/B2: the committed artifact.  EVERY refusal precedes EVERY write, including the last.

    The order below is the design, not an accident of drafting:

    1. the caller-supplied cell set may not produce a committed artifact;
    2. P7.2b's calibration artifact still hashes to its pin;
    3. this run's canary, re-checked rather than believed;
    4. every chunk validated in full, again;
    5. completeness against a DECLARATION that exists independently of the chunks;
    6. the digests re-derived from disk -- demand (G2) and checkpoint (G1);
    7. the pairing, per draw, refused when broken;
    8. rho, the aggregates, H3's inequalities, the exploratory contrast, the diagnostics;
    9. a scan over the SERIALISED bytes -- and only then ``_write_json``.

    Step 9 exists because step 4 can be widened by a future edit and a reordered write would publish
    before anything checked it.  ``tests/test_transfer_curve.py`` reaches it with delivered code.

    **The declaration's scenario selects the body** (``BRIEF_39`` Amendment B.7-2).  A grid4x4
    declaration reads P7.3d's calibration at step 2, refuses at step 6 a DT chunk whose support
    ranges are not the pinned ones, and after the pairing REFUSES unless C4's six reference cells
    reproduce bit for bit (:func:`_grid4x4_reference_check`); it then writes the
    ``p7.3d-grid4x4/1.0`` artifact (:func:`_grid4x4_artifact`) through the same step 9
    (:func:`_publish_artifact`).  An hz1x1 declaration runs exactly the code it always ran.
    """
    from offline.dt_gate import EpisodeResult, mean_ci95
    from offline.offline_baselines import paired_comparison
    from offline.transfer_calibration import CANARY_MAX_SECONDS, CANARY_RECORD_NAME

    work = Path(work_dir)
    target_path = Path(out_path)
    data = _data_dir(data_dir)

    # ---------------------------------------------------------------- 1. the cell-set guard
    # The WHOLE declaration, and this stage's slice of it. Both are derived from the declaration
    # and never from the chunks on disk, which is what stops the completeness check being a
    # tautology (PROJECT_PLAN section 7).
    all_declared, declared = declarations_for(stage, cells)
    if cells is not None:
        if stage is not None:
            declared = [cell for cell in declared if cell["stage"] == stage]
        committed_dir = (_REPO_ROOT / "docs" / "data").resolve()
        if target_path.resolve().parent == committed_dir:
            raise ValueError(
                f"{target_path}: refusing to write the committed artifact from a caller-supplied "
                "cell set. The completeness check is only as good as the declaration it compares "
                "against, so the campaign passes none and gets declared_cells(stage) -- all 4,700. "
                "The parameter exists so a test can declare a small campaign of its own"
            )
    if not declared:
        raise ValueError(f"the declared cell set for stage {stage!r} is empty")
    # B.7-2: the SCENARIO of the declaration selects the artifact's body, so the driver's
    # `grid4x4_confirmatory` stage and a caller-supplied grid4x4 set take the same one, and hz1x1's
    # path is the code it always was. One artifact describes one scenario.
    declared_scenarios = sorted({scenario_of(cell) for cell in declared})
    if len(declared_scenarios) != 1:
        raise ValueError(
            f"the declared cells span the scenarios {declared_scenarios}; one artifact describes "
            "one scenario"
        )
    grid = declared_scenarios == [GRID4X4_SCENARIO_KEY]

    # ---------------------------------------------------------------- 2. the calibration pin
    # hz1x1: P7.2b's artifact. grid4x4 (B.7-2(ii)): P7.3d's per-intersection artifact, never P7.2b's.
    grid_calibration: dict[str, Any] | None = None
    if grid:
        calibration = None
        _calibration_path, grid_calibration = _load_p7_3d_calibration(data)
    else:
        calibration = load_calibration(data / P7_2B_CALIBRATION_NAME)

    # ---------------------------------------------------------------- 3. this run's canary
    canary_record = _read_canary_record(work)

    # ---------------------------------------------------------------- 4. every chunk, validated
    chunks: dict[str, dict[str, Any]] = {}
    for path in sorted(work.glob("cell_*.json")):
        payload = json.loads(path.read_bytes())
        validate_cell_payload(payload)
        name = cell_chunk_name(payload)
        if name != path.name:
            raise ValueError(
                f"{path.name}: its content names the cell {name}; a chunk under another cell's "
                "filename was skipped as complete on every restart once already (BRIEF_33 C1.2)"
            )
        chunks[name] = payload

    # ---------------------------------------------------------------- 5. completeness
    declared_by_name = {cell_chunk_name(cell): cell for cell in declared}
    all_declared_names = {cell_chunk_name(cell) for cell in all_declared}
    missing = sorted(set(declared_by_name) - set(chunks))
    if missing:
        raise ValueError(
            f"{len(missing)} declared cell(s) have no chunk (first: {missing[:3]}); the campaign is "
            "incomplete and a partial artifact would report an arm on fewer draws than registered"
        )
    # ⚠️ Reviewer MIN-5: the two stages SHARE one work directory (B1), so a chunk that is a
    # declared cell of the OTHER stage is not an intruder -- it is the rest of the campaign. Before
    # this, `report --stage confirmatory` refused as soon as stage 2 wrote its first chunk, which
    # made the stage-1 artifact regenerable only from a tree that stops existing the moment stage 2
    # starts -- while B2 checks its byte-identity at the recording commit. Counted, never read.
    outside_stage = sorted(set(chunks) & set(all_declared_names) - set(declared_by_name))
    extra = sorted(set(chunks) - set(all_declared_names))
    if extra:
        raise ValueError(
            f"{len(extra)} chunk(s) are not declared cells of ANY stage (first: {extra[:3]}); an "
            "undeclared cell reaching the artifact is an evaluation nobody registered"
        )
    # Every chunk on disk is validated, whatever stage it belongs to: an undeclared ARM anywhere in
    # the work directory refuses, which is B2's rule and is what keeps the filtering above from
    # becoming a way to hide a cell.
    for name, payload in chunks.items():
        validate_cell_payload(payload, cell=declared_by_name.get(name))
    chunks = {name: payload for name, payload in chunks.items() if name in declared_by_name}

    # ---------------------------------------------------------------- 6. the digests, from disk
    # J1(c) at the artifact: `report` re-derives the provenance verdict rather than trusting that
    # `chunk_is_reusable` ran. A resumed campaign can carry chunks a previous revision wrote, and
    # the artifact claims every cell in it was produced by the code it names.
    chunk_commits_by_stage: dict[str, set[str]] = {}
    for name, payload in sorted(chunks.items()):
        commit = str(payload["git_commit"])
        chunk_commits_by_stage.setdefault(str(payload.get("stage")), set()).add(commit)
        changed = code_changed_since(commit)
        if changed:
            raise ValueError(
                f"{name}: it was rolled at {commit}, which differs from HEAD outside docs/ "
                f"({changed[:3]}, {len(changed)} path(s)). Commits may differ only by "
                "documentation; this cell was produced by different code and must be re-rolled"
            )

    demand_by_draw: dict[int, dict[str, Any]] = {}
    identity_by_checkpoint: dict[tuple[str, int], dict[str, Any]] = {}
    grid_targets: dict[str, float] | None = None
    pinned_ranges: dict[str, tuple[float, float]] | None = None
    for name, payload in sorted(chunks.items()):
        draw_id = int(payload["draw_id"])
        # B.6-2(2): the SAME scenario-aware call run_cell and chunk_is_reusable make. One stage is
        # one declaration, so every chunk here shares the stage's scenario and the draw is a key.
        if draw_id not in demand_by_draw:
            demand_by_draw[draw_id] = demand_identity_for(payload, out_root=out_root)
        demand = demand_by_draw[draw_id]
        for key in ("config_sha256", "routes_sha256"):
            if str(payload[key]) != demand[key]:
                raise ValueError(
                    f"{name}: {key} {payload[key]!r} is not draw {draw_id}'s {demand[key]!r}. The "
                    "cell ran on demand that is not what is on disk now, so it cannot be paired "
                    "with the anchors of this draw"
                )
        if str(payload["kind"]) == "dt":
            key_pair = (str(payload["subject"]), int(payload["seed"]))
            if key_pair not in identity_by_checkpoint:
                identity_by_checkpoint[key_pair] = checkpoint_identity_for(
                    payload, output_root=output_root, data_dir=data
                )
            if scenario_of(payload) == GRID4X4_SCENARIO_KEY:
                # The 16 prompts, re-derived from the digest-pinned artifact: a chunk conditioned
                # on other targets is internally consistent, so step 4 cannot see it.
                if grid_targets is None:
                    grid_targets = load_grid4x4_targets(data_dir=data)
                if dict(payload["target_rtg"]) != grid_targets:
                    raise ValueError(
                        f"{name}: its target_rtg is not the 16 registered prompts of "
                        f"{P7_3D_CALIBRATION_NAME}; the cell conditioned on something else"
                    )
                # B.7-2(i): the in-support block sums the chunks' per-id counts against the
                # pinned ranges, so a chunk counted against another range is refused here.
                if pinned_ranges is None:
                    pinned_ranges = grid4x4_support_ranges(data_dir=data)
                recorded = {
                    str(ix): [float(v) for v in value]
                    for ix, value in dict(payload["support_range"]).items()
                }
                if recorded != {ix: [low, high] for ix, (low, high) in pinned_ranges.items()}:
                    raise ValueError(
                        f"{name}: its support_range is not {P7_3D_CALIBRATION_NAME}'s "
                        "per-intersection range, so its in-support counts were taken against "
                        "another range"
                    )
            declared_sha = identity_by_checkpoint[key_pair]["file_sha256"]
            if str(payload["checkpoint_sha256"]) != declared_sha:
                raise ValueError(
                    f"{name}: it records checkpoint sha256 {payload['checkpoint_sha256']!r}, not "
                    f"the {declared_sha!r} that {key_pair[0]} seed {key_pair[1]}'s committed record "
                    "pins today; the weights that produced this cell are not the registered ones"
                )

    # ---------------------------------------------------------------- 7-8. pairing, rho, blocks
    anchors_by_draw: dict[int, dict[str, Mapping[str, Any]]] = {}
    for payload in chunks.values():
        if payload["arm"] in ("fixedtime", "maxpressure"):
            anchors_by_draw.setdefault(int(payload["draw_id"]), {})[str(payload["arm"])] = payload

    rows: list[dict[str, Any]] = []
    for name in sorted(chunks):
        payload = chunks[name]
        draw_id = int(payload["draw_id"])
        anchors = anchors_by_draw.get(draw_id, {})
        if set(anchors) != {"fixedtime", "maxpressure"}:
            raise ValueError(
                f"{name}: draw {draw_id} carries anchors {sorted(anchors)}, so rho has no "
                "denominator of its own draw. Pairing is PER DRAW because the demand differs by "
                "draw, and a ratio built from another draw's anchors normalises nothing"
            )
        published = _GRID4X4_PUBLISHED_FIELDS if grid else _PUBLISHED_FIELDS
        row = {field: payload.get(field) for field in published}
        row.update(_rho_pair(payload, anchors))
        rows.append(row)

    if grid:
        # B.7-2: grid4x4's body. (iii) first -- the six reference cells, bit for bit, a REFUSAL
        # that precedes every aggregate and the write -- then (i), (ii) and (iv).
        reference_record = _grid4x4_reference_check(chunks, data_dir=data)
        artifact = _grid4x4_artifact(
            rows=rows,
            chunks=chunks,
            stage=stage,
            n_declared=len(declared),
            outside_stage=outside_stage,
            chunk_commits_by_stage=chunk_commits_by_stage,
            cells_supplied=cells is not None,
            anchors_by_draw=anchors_by_draw,
            canary_record=canary_record,
            calibration_payload=grid_calibration or {},
            reference_record=reference_record,
            identity_by_checkpoint=identity_by_checkpoint,
            demand_by_draw=demand_by_draw,
        )
        return _publish_artifact(artifact, target_path)

    # ---- rho's denominator, per definition, from the ANCHOR cells (2026-09-17 ruling) ---------
    denominators: dict[str, dict[int, float]] = {
        key: {
            draw_id: anchor_denominator(anchors, key)
            for draw_id, anchors in sorted(anchors_by_draw.items())
        }
        for key in ("e_sumo", "att_env")
    }
    excluded_env_draws = [
        {
            "draw_id": draw_id,
            "att_fixedtime": float(anchors_by_draw[draw_id]["fixedtime"]["att_env"]),
            "att_maxpressure": float(anchors_by_draw[draw_id]["maxpressure"]["att_env"]),
            "why": (
                "the two anchors have EXACTLY equal env ATT on this draw, so rho's denominator is "
                "zero and the ratio is undefined; excluded from this definition's means, CI and "
                "H3 clauses, and reported here"
            ),
        }
        for draw_id, value in sorted(denominators["att_env"].items())
        if value == 0.0
    ]

    by_seed: list[dict[str, Any]] = []
    by_arm: list[dict[str, Any]] = []
    # The anchor's (subject, arm) is APPENDED, so P7.3a's entries keep their order and its
    # artifact is unchanged: with no anchor rows on disk the extra entry is skipped by the same
    # `if not arm_rows: continue` that every absent arm already takes.
    for subject, spec in [(s, spec) for s in SUBJECTS for spec in DECLARED_ARMS] + [
        (ANCHOR_SUBJECT, ArmSpec(ANCHOR_ARM, "naive_in_domain", "none", None, ANCHOR_ROLE))
    ]:
        arm_rows = [r for r in rows if r["subject"] == subject and r["arm"] == spec.name]
        if not arm_rows:
            continue
        for seed in TRAINING_SEEDS:
            seed_rows = [r for r in arm_rows if r["seed"] == seed]
            if not seed_rows:
                continue
            # An EXCLUDED draw (rho None) is dropped from that definition's mean and from the
            # n reported beside it -- never averaged in as a zero (2026-09-17 ruling).
            per_seed: dict[str, Any] = {
                "subject": subject,
                "arm": spec.name,
                "seed": int(seed),
                "n_draws": len(seed_rows),
            }
            for key in ("e_sumo", "att_env"):
                usable = [r[f"rho_{key}"] for r in seed_rows if r[f"rho_{key}"] is not None]
                per_seed[f"mean_rho_{key}"] = (
                    float(sum(usable) / len(usable)) if usable else None
                )
                per_seed[f"n_draws_{key}"] = len(usable)
            by_seed.append(per_seed)
        entry: dict[str, Any] = {"subject": subject, "arm": spec.name, "role": spec.role}
        for key in ("e_sumo", "att_env"):
            per_draw = _seed_means_by_draw(arm_rows, f"rho_{key}")
            if not per_draw:
                # Every draw undefined on this definition. Reported as such and carried past:
                # refusing here would decide the handling of an undefined ratio after the
                # campaign's numbers exist, which is what the 2026-09-17 ruling closes.
                #
                # ⚠️ CORRECTED 2026-09-18 (BRIEF_38 §2, Finding 4). Until then the sentence
                # above was a claim the code did not honour: the None below reached
                # `_h3_block`, where `stats["mean"] > 0.0` raised
                # `TypeError: '>' not supported between instances of 'NoneType' and 'float'`,
                # and `_contrast_block`, where `mean - mean` raised
                # `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'`.
                # Both fired AFTER every refusal had passed -- at the point where the artifact
                # was about to be written -- so the excluded definition took the REGISTERED
                # PRIMARY down with it. Both blocks now carry the None through as None; a
                # verdict is not invented and a zero is not substituted.
                entry[key] = {
                    "n_draws": 0,
                    "mean": None,
                    "std": None,
                    "ci95": None,
                    "ci95_low": None,
                    "ci95_high": None,
                    "why_empty": (
                        "every draw's denominator on this definition was exactly zero; see "
                        "denominator_diagnostic and excluded_draws"
                    ),
                }
                continue
            stats = mean_ci95([per_draw[d] for d in sorted(per_draw)])
            entry[key] = {
                "n_draws": stats.n,
                "mean": stats.mean,
                "std": stats.std,
                "ci95": stats.ci95,
                "ci95_low": stats.mean - stats.ci95,
                "ci95_high": stats.mean + stats.ci95,
            }
        entry["paired_att"] = {
            key: {
                anchor: _paired_block(rows, arm_rows, subject, spec.name, anchor, key)
                for anchor in ("fixedtime", "maxpressure")
            }
            for key in ("e_sumo", "att_env")
        }
        by_arm.append(entry)

    artifact: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "registered_in": "PREREGISTRATION H3, §3.4, A15, A16, A17, A18(c)",
        "scenario_key": SCENARIO_KEY,
        "stage": stage,
        "n_cells_declared": len(declared),
        # Declared cells of the OTHER stage that share this work directory (B1). Counted so the
        # artifact says what else was on disk, and never read into any number (reviewer MIN-5).
        "n_chunks_outside_stage": len(outside_stage),
        # J1(c): the commits the CELLS were rolled by, per stage. The top-level git_commit below is
        # this report's own, taken at write time, and the two are routinely different -- the
        # stage-1 artifact is written by a docs commit.
        "chunk_commits_by_stage": {
            stage_name: sorted(commits)
            for stage_name, commits in sorted(chunk_commits_by_stage.items())
        },
        "cell_set_source": "declared_cells()" if cells is None else "caller-supplied declaration",
        "halting_check_draw": HALTING_CHECK_DRAW,
        "cells": rows,
        "rho": {
            "formula": "rho = (ATT_fixedtime - ATT_arm) / (ATT_fixedtime - ATT_maxpressure)",
            "definitions": {
                "e_sumo": {
                    "what": (
                        "A15's primary: the pool-clock ATT over the all-created population, from "
                        "the observer. P7.1's freeze writes the same quantity under the key "
                        "att_reference_created_population (Amendment C6); one quantity, two names"
                    ),
                    "role": "REGISTERED PRIMARY (A15)",
                    "denominator_diagnostic": denominator_diagnostic(denominators["e_sumo"]),
                    "zero_denominator_rule": (
                        "REFUSED. On P7.1's frozen anchors this difference is +206.5 to +252.6 s, "
                        "so a zero here is a finding about the instrument rather than a property "
                        "of a draw, and rho() raises"
                    ),
                },
                "att_env": {
                    "what": "the admitted pair beside it: the env's own metric at the horizon",
                    "role": "co-reported (A15)",
                    "caveat": ATT_ENV_CAVEAT,
                    "denominator_diagnostic": denominator_diagnostic(denominators["att_env"]),
                    "zero_denominator_rule": (
                        "RECORDED AND EXCLUDED, and report carries on (author's ruling, "
                        "2026-09-17). The draw's cells keep rho_att_env: null, the draw is listed "
                        "in excluded_draws, and it counts towards no mean, no CI and no H3 clause "
                        "on this definition"
                    ),
                    "excluded_draws": excluded_env_draws,
                    "n_draws_used": len(denominators["att_env"]) - len(excluded_env_draws),
                    "n_draws_total": len(denominators["att_env"]),
                },
            },
            "estimator": {
                "method": (
                    "analytic normal approximation, 1.96*s/sqrt(n) over the per-draw seed means "
                    "(offline.dt_gate.mean_ci95, the helper P5.3b's decomposition reached through "
                    "paired_comparison)"
                ),
                "resampling_seed": None,
                "why_no_seed": (
                    "there is no resampling: the interval is analytic and deterministic. BRIEF_37 "
                    "§3.5 asked for the helper's resampling seed; the repository has no bootstrap "
                    "estimator, and inventing one would be a NEW estimator, which the same sentence "
                    "forbids (plan §0.4, Amendment A3)"
                ),
                "unit": "one paired evaluation draw; seeds averaged within a draw, as in P4",
            },
            "not_clipped": (
                "PREREGISTRATION §3.4: values may exceed 1 or fall below 0 and are not clipped. "
                "fixed-time is 0 and MaxPressure is 1 by construction"
            ),
            "by_subject_arm_seed": by_seed,
            "by_subject_arm": by_arm,
        },
        "h3": _h3_block(by_arm),
        "contrast": _contrast_block(by_arm),
        "in_support": _in_support_block(rows, calibration),
        "canary": {
            "seconds": canary_record["seconds"],
            "threshold_seconds": CANARY_MAX_SECONDS,
            "verdict": "at speed" if canary_record["seconds"] <= CANARY_MAX_SECONDS else "throttled",
            "observed": dict(canary_record["facts"]),
            "source": (
                f"{CANARY_RECORD_NAME} in the work directory, written by the driver from the canary "
                "line right after the token; report re-ran check_canary on these facts before "
                "writing this file"
            ),
            "git_commit": canary_record.get("git_commit"),
            "git_dirty": canary_record.get("git_dirty"),
            "chunk_canaries_by_stage": _canaries_by_stage(chunks),
            "why_more_than_one_is_expected": (
                "Amendment B1 runs the campaign in two declared stages under two tokens and the "
                "driver re-runs the canary at each stage's start, so the chunks legitimately carry "
                "one canary per stage. P7.2b's single-canary rule was a one-stage rule"
            ),
        },
        "what_this_does_not_say": (
            ANCHOR_WHAT_THIS_DOES_NOT_SAY
            if stage == STAGE_ANCHOR
            else (
                "This is the ZERO-SHOT point only. No model was fine-tuned, no anchor corpus was "
                "collected and no k-shot curve is reported here; the few-shot points and the anchor are "
                "P7.3b's. rho is computed WITHIN SUMO against anchors on the same draws, so it is not a "
                "cross-backend comparison of absolute travel times. The calibrated-vs-naive contrast is "
                "exploratory (PREREGISTRATION §2, A17(d)) and is never promoted to a claim. The "
                "in-support diagnostic selects nothing. H3's third clause -- that the gap closes "
                "substantially by k = 100 -- is not tested here."
            )
        ),
        "inputs": {
            "calibration_sha256": P7_2B_CALIBRATION_SHA256,
            "checkpoints": sorted(
                (identity_by_checkpoint[key] for key in identity_by_checkpoint),
                key=lambda entry: (entry["subject"], entry["seed"]),
            ),
            "deferred_56": (
                "output/p4_dt/ appears in no SHA256SUMS_* manifest, and every such manifest is "
                "gitignored in any case. Both subjects are therefore pinned against COMMITTED "
                "artifacts (Amendment G1): p4_gate.json for mappo1000 and p4_7_training.json's "
                "(tier, method, seed) row for mix50. sha256_checked_against on every cell names "
                "what was compared, so the two subjects' evidence is not implied to be equal. This "
                "is the first artifact in the project to pin the subjects' checkpoints by digest"
            ),
            "demand_by_draw": {
                str(draw): {k: v for k, v in demand.items() if k != "config_path"}
                for draw, demand in sorted(demand_by_draw.items())
            },
        },
        **_git_provenance(),
    }

    if stage == STAGE_ANCHOR:
        artifact["anchor"] = _anchor_block(rows, data)

    if stage1_path is not None:
        artifact["stage1_artifact"] = _stage1_block(Path(stage1_path), rows)

    # ---------------------------------------------------------------- 9. the LAST refusal
    return _publish_artifact(artifact, target_path)


def _publish_artifact(artifact: dict[str, Any], target_path: Path) -> dict[str, Any]:
    """Step 9 of :func:`report`, for BOTH bodies: the last refusals over the SERIALISED bytes, then
    the ONE write.  Extracted unchanged in the B.7 round so the grid4x4 body cannot publish by a
    different route than hz1x1's."""
    serialised = json.dumps(artifact, indent=2, sort_keys=True)
    if FENCED_KEY in serialised:
        raise AssertionError(
            f"{FENCED_KEY} reached the artifact: a fenced quantity from P7.2b's smoke is on its way "
            "into docs/data/. The scan is over the serialised bytes because a key-shaped check "
            "cannot see a fenced name carried in a VALUE"
        )
    published_arms = {str(row["arm"]) for row in json.loads(serialised)["cells"]}
    if not published_arms <= DECLARED_ARM_NAMES:
        raise AssertionError(
            f"undeclared arm(s) {sorted(published_arms - DECLARED_ARM_NAMES)} reached the "
            "artifact; BRIEF_37 §2 lifted the fence for the declared arms and for nothing else"
        )
    _write_json(target_path, artifact)
    return artifact



#: A18(a)'s own sentence about what the anchor is NOT, quoted verbatim.  It lives in the ARTIFACT
#: because the packet does not travel with it (the author's ruling of 2026-09-17 on ATT_ENV_CAVEAT,
#: applied to the same problem).
ANCHOR_WHAT_IT_IS_NOT = (
    "It is the curve's k = 200 endpoint and the paper says exactly what it is: what "
    "target-domain probe data alone buys the same architecture -- not an upper bound on "
    "achievable SUMO performance, and not a target-domain online policy."
)

ANCHOR_WHAT_THIS_DOES_NOT_SAY = (
    "This is the FULL-RETRAIN ANCHOR only -- the C3 curve's k = 200 endpoint (PREREGISTRATION "
    "A18(a)). " + ANCHOR_WHAT_IT_IS_NOT + " No few-shot point is reported here: k in {5, 20, 100} "
    "is P7.3c, deferred by the author on 2026-09-18, and grid4x4 is P7.3d. "
    "**A18(b): there is NO online SUMO anchor** -- no MAPPO-on-SUMO path exists, contract C8's "
    "metric-set defect (DEFERRED 75(e)) stands, and it would need >= 1,000 on-policy episodes per "
    "seed; it is deferred to P11 and is named as a limitation in the paper's C3 section, not in a "
    "footnote. Nothing here is a verdict on P7.3a's zero-shot number: the anchor is a "
    "measurement reported beside it, and H3's clauses are P7.3a's and are NOT recomputed by this "
    "artifact. rho is computed WITHIN SUMO against anchors on the same draws, so it is not a "
    "cross-backend comparison of absolute travel times. The `random` arm was NOT re-rolled for "
    "this stage (Amendment A4): rho's denominator is fixed-time and MaxPressure only, so no "
    "reader should infer a `random` comparison this stage ran. The in-support diagnostic selects "
    "nothing."
)


def _anchor_block(rows: Sequence[Mapping[str, Any]], data_dir: Path) -> dict[str, Any]:
    """P7.3b section 3.4: the anchor reported as the curve's endpoint, and interpreted nowhere.

    Everything here is read from the committed training record rather than recomputed, so the
    artifact says which file its prompt came from and a reader can check that file's digest.
    """
    training = load_anchor_training(data_dir / P7_3B_TRAINING_NAME)
    facts = anchor_subject_facts(training)
    anchor_rows = [r for r in rows if r["subject"] == ANCHOR_SUBJECT]
    return {
        "subject": ANCHOR_SUBJECT,
        "arm": ANCHOR_ARM,
        "role": ANCHOR_ROLE,
        "registered_in": "PREREGISTRATION A18(a), tagged v1.8-prereg-a18 (045e9be)",
        "what_it_is_not": ANCHOR_WHAT_IT_IS_NOT,
        "no_online_anchor": (
            "A18(b), registered before any target-domain number existed: there is no online SUMO "
            "anchor in this paper. Deferred to P11 and named as a limitation in the C3 section"
        ),
        "prompt": {
            "rule": "naive_in_domain",
            "definition": str(training["prompt_rule_definition"]),
            "target_rtg": float(training["target_rtg"]),
            "rtg_scale": float(training["rtg_scale"]),
            "source_artifact": P7_3B_TRAINING_NAME,
            "source_sha256": P7_3B_TRAINING_SHA256,
            "not_from": (
                "docs/data/p7_2b_calibration.json. That artifact registers CityFlow-trained "
                "subjects; conditioning the anchor on the SOURCE domain's returns would make it a "
                "different experiment. Amendment A5 requires the proof to be the ROUTE rather than "
                "a value, and the route is tested: load_calibration and targets_for_subject are "
                "never reached on this subject's branch of run_cell"
            ),
        },
        "training": {
            "declared_gradient_steps": int(training["declared_gradient_steps"]),
            "raise_to": training["raise_to"],
            "recipe": dict(training["recipe"]),
            "n_episodes": int(training["n_episodes"]),
            "draw_band": [
                int(training["corpus"]["draw_ids"][0]),
                int(training["corpus"]["draw_ids"][-1]),
            ],
            "corpus_digest_files": dict(training["corpus"]["digest_files"]),
            "disjoint_from_the_subjects_and_the_pool": bool(training["disjointness"]["disjoint"]),
            "support_range_over_the_split": list(facts.support_range_over_the_split),
            "training_set_return_min": facts.training_set_return_min,
            "seeds": [
                {"seed": int(r["seed"]), "checkpoint_sha256": str(r["checkpoint_sha256"]),
                 "final_loss": float(r["final_loss"]), "seconds": float(r["seconds"])}
                for r in training["seeds"]
            ],
        },
        "n_cells": len(anchor_rows),
        "h3_is_not_recomputed_here": (
            "H3's clauses are about b_mean_k100 and are P7.3a's; docs/data/p7_3a_zero_shot.json "
            "carries them. This artifact's h3 block therefore has no subject in it, which is the "
            "honest result of running report over a work directory that holds no b_mean_k100 cell"
        ),
        "reported_not_interpreted": (
            "rho is reported under both registered definitions with its CIs. This artifact draws "
            "no conclusion about whether the anchor beats, matches or trails the zero-shot point"
        ),
    }


def _seed_means_by_draw(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, float]:
    """The per-draw unit: the mean over training seeds, as in P4, so seed and draw stay crossed.

    A ``None`` rho is an EXCLUDED draw (the author's 2026-09-17 ruling on an exactly-zero env-ATT
    denominator), not a zero: it is dropped here rather than averaged in, and the draws that
    survive are what the reported ``n`` counts.
    """
    buckets: dict[int, list[float]] = {}
    for row in rows:
        if row.get(key) is None:
            continue
        buckets.setdefault(int(row["draw_id"]), []).append(float(row[key]))
    return {draw: sum(values) / len(values) for draw, values in buckets.items()}


def _paired_block(
    rows: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    subject: str,
    arm: str,
    anchor: str,
    key: str,
) -> dict[str, Any]:
    """G7/A3: the registered paired comparison on ATT, with the Wilcoxon beside the analytic CI.

    ``paired_comparison`` pairs per draw over shared draws and averages seeds within a draw
    (``dt_gate._per_draw_means``); A5 point 3 makes a comparison without shared draws void, which it
    raises on.  The arm names are qualified with the subject so each side carries exactly one.
    """
    from offline.dt_gate import EpisodeResult
    from offline.offline_baselines import paired_comparison

    def episodes(source: Sequence[Mapping[str, Any]], label: str) -> list[EpisodeResult]:
        return [
            EpisodeResult(
                arm=label,
                seed=None if row["seed"] is None else int(row["seed"]),
                draw_id=int(row["draw_id"]),
                att_horizon=float(row[key]),
                horizon_vehicle_count=float(row["horizon_vehicle_count"]),
                episode_reward=float(row["episode_reward"]),
            )
            for row in source
        ]

    anchor_rows = [r for r in rows if r["arm"] == anchor]
    comparison = paired_comparison(
        episodes(arm_rows, f"{subject}:{arm}"), episodes(anchor_rows, anchor)
    )
    return {
        "att_definition": key,
        "left_arm": comparison.left_arm,
        "right_arm": comparison.right_arm,
        "n_shared_draws": comparison.n_shared_draws,
        "mean_left": comparison.mean_left,
        "mean_right": comparison.mean_right,
        "mean_difference": comparison.mean_difference,
        "ci95_low": comparison.ci95_low,
        "ci95_high": comparison.ci95_high,
        "median_difference": comparison.median_difference,
        "wins": comparison.wins,
        "losses": comparison.losses,
        "ties": comparison.ties,
        "rank_biserial": comparison.rank_biserial,
        "wilcoxon_p": comparison.wilcoxon.p_value,
        "direction": "mean(arm - anchor) on ATT; NEGATIVE means the arm had the lower travel time",
    }


def _satisfies(stats: Mapping[str, Any], test: Any) -> bool | None:
    """``test(stats)``, or ``None`` when this definition has no usable draw (Finding 4).

    The empty block :func:`report` writes for an all-excluded definition sets ``mean``, ``std``,
    ``ci95``, ``ci95_low`` and ``ci95_high`` to ``None`` **together**, so ``mean`` is a sufficient
    test for all of them.  ``None`` here means *this definition had nothing to test*, which is not
    the same fact as ``False`` (*it was tested and the inequality did not hold*) -- and writing
    ``False`` there would be this code inventing a verdict about draws it excluded.
    """
    if stats.get("mean") is None:
        return None
    return bool(test(stats))


def _h3_block(by_arm: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H3's first two clauses as the registered INEQUALITIES, reported and not interpreted.

    *"Zero-shot transfer is positive but incomplete -- better than fixed-time, worse than
    within-backend MaxPressure -- and closes substantially by k = 100."*  Clause 3 is P7.3b's and is
    named here as absent rather than left to be inferred.  There is no verdict sentence: the
    registered claim is the inequality, and a word like "confirmed" would be this task deciding
    something A17 gave it no licence to decide.
    """
    registered = [entry for entry in by_arm if entry["arm"] == "b_mean_k100"]
    clauses: list[dict[str, Any]] = []
    for inequality, clause, point_test, interval_test in (
        (
            "rho_sumo(b_mean_k100) > 0",
            "better than the within-backend fixed-time anchor",
            lambda stats: stats["mean"] > 0.0,
            lambda stats: stats["ci95_low"] > 0.0,
        ),
        (
            "rho_sumo(b_mean_k100) < 1",
            "worse than within-backend MaxPressure",
            lambda stats: stats["mean"] < 1.0,
            lambda stats: stats["ci95_high"] < 1.0,
        ),
    ):
        for key in ("e_sumo", "att_env"):
            # The caveat travels ON THE CLAUSE, not only in the definitions block and not only in
            # the packet (author's ruling, 2026-09-17): a reader who opens the artifact and looks
            # at one H3 clause on the co-reported definition must see it there.
            caveat = {"caveat": ATT_ENV_CAVEAT} if key == "att_env" else {}
            clauses.append(
                {
                    "inequality": inequality,
                    "clause": clause,
                    "definition": key,
                    **caveat,
                    # ⚠️ Amendment H7: this used to be one boolean called `holds`, which anyone
                    # opening the artifact would read as "confirmed" -- a verdict on a point
                    # estimate, next to a registered test row that is a PAIRED comparison. Two
                    # mechanical facts now sit side by side and neither is a verdict: whether the
                    # mean is on the claimed side, and whether the WHOLE analytic interval is.
                    # Finding 4: `_satisfies` returns None on an all-excluded definition rather
                    # than raising TypeError on `None > 0.0`, which used to take the whole report
                    # down -- including the REGISTERED PRIMARY, which had lost nothing.
                    "point_estimate_satisfies": {
                        str(e["subject"]): _satisfies(e[key], point_test) for e in registered
                    },
                    "ci95_entirely_satisfies": {
                        str(e["subject"]): _satisfies(e[key], interval_test) for e in registered
                    },
                    "mean_rho": {str(e["subject"]): e[key]["mean"] for e in registered},
                    "ci95_low": {str(e["subject"]): e[key]["ci95_low"] for e in registered},
                    "ci95_high": {str(e["subject"]): e[key]["ci95_high"] for e in registered},
                }
            )
    return {
        "hypothesis": (
            "Zero-shot transfer is positive but incomplete -- better than fixed-time, worse than "
            "within-backend MaxPressure -- and closes substantially by k = 100"
        ),
        "registered_arm": "b_mean_k100",
        "primary_definition": "e_sumo",
        "test_row": (
            "MADT zero-shot in SUMO vs the within-backend fixed-time anchor, per paired scenario; "
            "unit: paired evaluation draw"
        ),
        "clauses": clauses,
        "third_clause": (
            "'closes substantially by k = 100' is P7.3b's and is NOT tested by this artifact"
        ),
        "reported_not_interpreted": (
            "the inequality and its value are reported; this artifact draws no conclusion from "
            "them. point_estimate_satisfies and ci95_entirely_satisfies are arithmetic on the "
            "numbers beside them, not verdicts: the registered test row is the paired comparison, "
            "which is reported in rho.by_subject_arm[*].paired_att"
        ),
    }


def _contrast_block(by_arm: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A17(d)'s calibrated-vs-naive contrast: EXPLORATORY, with its registered direction beside it."""
    by_subject: dict[str, Any] = {}
    for subject in SUBJECTS:
        calibrated = [e for e in by_arm if e["subject"] == subject and e["arm"] == "b_mean_k100"]
        naive = [e for e in by_arm if e["subject"] == subject and e["arm"] == "naive"]
        if not calibrated or not naive:
            continue
        by_subject[subject] = {
            key: {
                "calibrated_mean_rho": calibrated[0][key]["mean"],
                "naive_mean_rho": naive[0][key]["mean"],
                # Finding 4: `None - None` is a TypeError. An undefined difference is reported as
                # None -- never as 0.0, which would read as "the two prompts agreed".
                "difference": (
                    None
                    if calibrated[0][key]["mean"] is None or naive[0][key]["mean"] is None
                    else calibrated[0][key]["mean"] - naive[0][key]["mean"]
                ),
            }
            for key in ("e_sumo", "att_env")
        }
    return {
        "status": "exploratory",
        "registered_direction": (
            "A17(d): calibrated >= naive, and the gap larger on mix50. Registered as a DIRECTION "
            "for an exploratory contrast, never as a confirmatory test"
        ),
        "never_promoted": (
            "PREREGISTRATION §2 and A17(d): this contrast is exploratory and is not promoted to a "
            "claim whichever way it comes out"
        ),
        "by_subject": by_subject,
        "arms_present": sorted({str(entry["arm"]) for entry in by_arm}),
    }


def _in_support_block(
    rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """The in-support diagnostic, declared and observed side by side.  Nothing selects on it.

    Amendment E4 recorded BEFORE any number existed that ``a_q1.0``'s target is out of support for
    ``mappo1000`` (below its training-return range, margin -10818) and inside it for ``mix50``.
    This block reports the artifact's own field so that fact is read from the registration rather
    than discovered afterwards and offered as an explanation of a result.
    """
    declared: dict[str, Any] = {}
    for subject in SUBJECTS:
        targets = targets_for_subject(subject, calibration)
        declared[subject] = {name: row["in_support"] for name, row in targets.items()}

    observed: dict[str, Any] = {}
    for row in rows:
        if row["kind"] != "dt" or row["n_decisions_in_support"] is None:
            continue
        key = f"{row['subject']}/{row['arm']}"
        bucket = observed.setdefault(key, {"n_cells": 0, "total_decisions_in_support": 0})
        bucket["n_cells"] += 1
        bucket["total_decisions_in_support"] += int(row["n_decisions_in_support"])

    return {
        "what_this_is": (
            "a reliability diagnostic, withdrawn as a selection criterion on 2026-08-13 "
            "(BRIEF_15 §12.1): target = 0 scores 0.000 here and was the best point measured. "
            "Nothing in this artifact selects on it"
        ),
        "declared_by_the_calibration_artifact": declared,
        "observed_decisions": observed,
        "amendment_e4": (
            "recorded before any number existed: a_q1.0's target is OUT OF SUPPORT for mappo1000 "
            "and in support for mix50, as the calibration artifact states"
        ),
    }


def _canaries_by_stage(chunks: Mapping[str, Mapping[str, Any]]) -> dict[str, list[float]]:
    """The canary each stage's chunks were rolled under -- one per stage, not one per campaign."""
    buckets: dict[str, set[float]] = {}
    for payload in chunks.values():
        seconds = payload.get("canary_seconds")
        if seconds is None:
            raise ValueError(
                f"{cell_chunk_name(payload)}: no canary_seconds. Every cell records the rate basis "
                "of the run that rolled it, or its timing cannot be compared with any other"
            )
        buckets.setdefault(str(payload.get("stage")), set()).add(float(seconds))
    return {stage: sorted(values) for stage, values in sorted(buckets.items())}


def _stage1_block(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Amendment B2: the stage-1 artifact's digest, and its rows compared cell for cell.

    Stage 2 is unconditional (B1), so the confirmatory rows must not move when the rest arrives.
    The comparison is on the rows themselves, under ``==``; the digest is recorded so a reader can
    redo it against the committed stage-1 file instead of trusting this sentence.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: the stage-1 artifact was named but is not here, so its rows cannot be "
            "compared with the final ones"
        )
    stage1 = json.loads(path.read_bytes())
    final_by_name = {cell_chunk_name(row): row for row in rows}
    moved: list[str] = []
    for row in stage1["cells"]:
        name = cell_chunk_name(row)
        if name not in final_by_name or final_by_name[name] != row:
            moved.append(name)
    if moved:
        raise ValueError(
            f"{len(moved)} stage-1 cell(s) differ between the stage-1 artifact and this one "
            f"(first: {moved[:3]}). Amendment B2: the stages are a sequence, not a cut, and a "
            "confirmatory row that moved once the rest of the campaign arrived would mean the "
            "number reported early is not the number reported finally"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "n_cells": len(stage1["cells"]),
        "rows_identical": True,
    }


# ======================================================================================
# B.7-2: report's grid4x4 BODY -- the artifact `p7.3d-grid4x4/1.0`
# ======================================================================================

#: The grid4x4 artifact's published row.  hz1x1's scalar fields, plus the grid4x4 chunk's
#: per-intersection SUMMARIES (the 16 returns, the 16 first/last RTGs, the 16 in-support counts) --
#: and deliberately NOT the per-decision RTG / reward series or the 360 x 16 action matrix: on sixteen
#: intersections they would make the committed file about 130 MB (hz1x1's, with one intersection's
#: series, is 64.8 MB), and every one of them stays in the chunks, which the manifest covers.  A
#: whitelist, as ``_PUBLISHED_FIELDS`` is: a field added to the chunk later is excluded by default.
_GRID4X4_PUBLISHED_FIELDS: tuple[str, ...] = (
    "kind", "subject", "arm", "seed", "policy_seed", "draw_id", "stage", "scenario",
    "action_space_n", "att_env", "att_horizon", "att_running_mean", "e_sumo", "p_sumo", "w_sumo",
    "mean_depart_delay", "episode_reward", "horizon_vehicle_count", "decisions",
    "engine_seed_requested", "engine_seed_drawn",
    "n_created", "n_entered", "n_never_entered", "n_pending_at_horizon", "n_teleports",
    "n_vanished_without_arrival", "n_arrived_never_observed_at_a_boundary",
    "max_abs_depart_clock_deviation", "vehicle_types_seen", "time_to_teleport_option",
    "halting_checked", "halting_max_abs_difference", "halting_n_lane_seconds",
    "halting_n_disagreeing_lane_seconds",
    "intersection_ids", "local_return", "target_rtg", "rtg_first", "rtg_last",
    "rtg_advanced_every_decision", "n_decisions_in_support", "support_range", "in_support_counts",
    "checkpoint", "checkpoint_sha256", "sha256_checked_against",
    "config_sha256", "routes_sha256", "calibration_sha256",
    "canary_seconds", "seconds", "git_commit", "git_dirty",
)

#: The co-reported definition's caveat for THIS scenario.  hz1x1's :data:`ATT_ENV_CAVEAT` quotes
#: P7.1's frozen hangzhou anchors; no grid4x4 value is quoted here, because none existed when the
#: text was written -- the denominator diagnostic beside it is where grid4x4's numbers are.
GRID4X4_ATT_ENV_CAVEAT = (
    "CO-REPORTED, NOT THE REGISTERED PRIMARY (A15). rho on this definition is a per-draw ratio whose "
    "denominator -- ATT_fixedtime - ATT_maxpressure on the env's own metric -- can be small or "
    "negative on a draw, and such a draw can dominate the mean or flip its sign (on hz1x1 it did). "
    "Read this definition's rho only with denominator_diagnostic beside it."
)

#: Everything else the grid4x4 artifact does NOT say, after A21(b)'s two items.
GRID4X4_WHAT_THIS_DOES_NOT_SAY: tuple[str, ...] = (
    "This is the ZERO-SHOT point on grid4x4 only: no model was fine-tuned and no anchor was trained "
    "on this scenario, so H3's third clause is void here (A19), and nothing here is a curve.",
    "rho is computed WITHIN SUMO against anchors on the same draws; it is not a cross-backend "
    "comparison of absolute travel times.",
    "Every per-intersection breakdown -- per-intersection rho and the in-support block -- is "
    "exploratory (A20(e)): no CI is promoted, and nothing selects on it.",
    "The per-decision RTG and reward series and the 360 x 16 action matrices are not republished "
    "here; they are in the chunks under output/p7_3d/cells/, covered by output/SHA256SUMS_p7_3d.txt.",
)


def _grid4x4_reference_check(
    chunks: Mapping[str, Mapping[str, Any]], *, data_dir: Path
) -> dict[str, Any]:
    """B.7-2(iii), A9's *the instrument regenerates*: the six reference cells, bit for bit, or REFUSE.

    Every refusal is collected first and raised together, naming each cell and each field, so a
    reader sees the whole disagreement at once.  A reference cell with no chunk refuses as well.
    """
    frozen = load_reference_cells(data_dir=data_dir)
    rows = {(str(row["arm"]), int(row["draw_id"])): row for row in frozen["cells"]}
    problems: list[str] = []
    record_cells: list[dict[str, Any]] = []
    for arm in REFERENCE_CELL_ARMS:
        for draw in REFERENCE_CELL_DRAWS:
            name = cell_chunk_name(
                {"subject": None, "arm": arm, "seed": None, "draw_id": draw,
                 "scenario": GRID4X4_SCENARIO_KEY}
            )
            chunk = chunks.get(name)
            if chunk is None:
                problems.append(f"{arm} draw {draw} has no chunk")
                continue
            differing = reference_cell_differences(chunk, rows[(arm, draw)])
            if differing:
                problems.append(f"{arm} draw {draw} differs on {differing}")
            record_cells.append({"arm": arm, "draw_id": draw, "reproduces": not differing})
    if problems:
        raise ValueError(
            f"the reference cell(s) do not reproduce {P7_3D_REFERENCE_CELLS_NAME} bit-for-bit "
            "(A9; BRIEF_39 C4, B.7-2(iii)): " + "; ".join(problems) + ". The instrument that "
            "produced this campaign is not the instrument that froze the anchors"
        )
    return {
        "what": (
            "A9's check that the instrument regenerates: the campaign's six anchor chunks on draws "
            "1000-1002 against the values C4 froze, under == on every field both records"
        ),
        "artifact": P7_3D_REFERENCE_CELLS_NAME,
        "artifact_sha256": P7_3D_REFERENCE_CELLS_SHA256,
        "n_checked": len(record_cells),
        "all_reproduce": all(cell["reproduces"] for cell in record_cells),
        "compared_fields": [list(pair) for pair in REFERENCE_CELL_COMPARED_FIELDS],
        "halting_counts_compared_where_the_frozen_run_checked": list(REFERENCE_CELL_HALTING_COUNTS),
        "not_compared": {
            "fields": list(REFERENCE_CELL_NOT_COMPARED),
            "why": (
                "e_sumo_total, n_intended and n_arrived are recorded by the frozen artifact and not "
                "by a chunk; seconds is a clock. episode_reward, which BRIEF_39 B.7-2(iii) names, "
                "is not recorded by the frozen artifact, so it cannot be compared"
            ),
        },
        "cells": record_cells,
    }


def _definition_stats(per_draw: Mapping[int, float]) -> dict[str, Any]:
    """The per-draw values' mean and analytic 95 % CI (``dt_gate.mean_ci95``), or the empty form."""
    from offline.dt_gate import mean_ci95

    if not per_draw:
        return {
            "n_draws": 0, "mean": None, "std": None, "ci95": None, "ci95_low": None,
            "ci95_high": None,
            "why_empty": "every draw's denominator on this definition was exactly zero",
        }
    stats = mean_ci95([per_draw[draw] for draw in sorted(per_draw)])
    return {
        "n_draws": stats.n,
        "mean": stats.mean,
        "std": stats.std,
        "ci95": stats.ci95,
        "ci95_low": stats.mean - stats.ci95,
        "ci95_high": stats.mean + stats.ci95,
    }


def _grid4x4_per_intersection_rho(
    rows: Sequence[Mapping[str, Any]], ix_ids: Sequence[str]
) -> dict[str, Any]:
    """B.7.1-2's DESCRIPTIVE block: rho per intersection on the per-intersection collection return.

    ``rho_i,d = (R_ft,i,d - R_arm,i,d) / (R_ft,i,d - R_mp,i,d)`` through :func:`rho` -- the one
    registered formula, applied to ``local_return[i]`` -- per cell, then averaged over seeds within
    a draw, then over the usable draws.  A draw whose two anchors tie exactly on intersection *i* is
    excluded for that intersection and listed.  No CI is computed: A20(e) makes every
    per-intersection breakdown exploratory.
    """
    anchors: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["arm"] in ("fixedtime", "maxpressure"):
            anchors.setdefault(int(row["draw_id"]), {})[str(row["arm"])] = row["local_return"]
    dt_rows = [row for row in rows if row["kind"] == "dt"]
    per_intersection: dict[str, Any] = {}
    for ix in ix_ids:
        excluded = sorted(
            draw
            for draw, pair in anchors.items()
            if float(pair["fixedtime"][ix]) - float(pair["maxpressure"][ix]) == 0.0
        )
        buckets: dict[int, list[float]] = {}
        for row in dt_rows:
            draw = int(row["draw_id"])
            if draw in excluded:
                continue
            pair = anchors[draw]
            buckets.setdefault(draw, []).append(
                rho(
                    float(row["local_return"][ix]),
                    float(pair["fixedtime"][ix]),
                    float(pair["maxpressure"][ix]),
                )
            )
        per_draw = {draw: sum(values) / len(values) for draw, values in buckets.items()}
        ordered = [per_draw[draw] for draw in sorted(per_draw)]
        per_intersection[str(ix)] = {
            "mean_rho": (sum(ordered) / len(ordered)) if ordered else None,
            "n_draws_used": len(ordered),
            "n_draws_excluded": len(excluded),
            "draw_ids_excluded": excluded,
        }
    return {
        "status": "exploratory and descriptive (A20(e)); no CI is computed, none is promoted",
        "definition": (
            "rho_i,d = (R_fixedtime,i,d - R_arm,i,d) / (R_fixedtime,i,d - R_maxpressure,i,d) "
            "(BRIEF_39 Amendment B.7.1-2), where R is local_return[i]: intersection i's episode "
            "return under the collection reward, the sum over the 360 POST-STEP infos of its reward, "
            "recorded on every cell by the probe's two routes (reward stream == minus lane waiting "
            "counts); seeds averaged within a draw; the mean over the draws whose denominator is "
            "not exactly zero for that intersection, the others excluded and listed"
        ),
        "not_the_reward_series": (
            "B.7.1-2 names the DT's value as sum(reward_series[i]); that series is read BEFORE each "
            "act (D1), so it carries the reset info's reward and not the last decision's, and would "
            "divide a 359-decision sum by 360-decision anchors. Every arm uses local_return instead"
        ),
        "per_intersection": per_intersection,
    }


def _grid4x4_in_support_block(
    dt_rows: Sequence[Mapping[str, Any]], calibration_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """B.7-2(i): per intersection, the pinned range, the target's position, the chunks' counts.

    The ranges and the registered targets' positions are READ from ``p7_3d_calibration.json`` (the
    chunks' ranges were refused at step 6 unless equal to them); the counts are summed from the
    chunks' own per-id counts and refused unless they add up to the decisions recorded.
    """
    key = f"k{int(calibration_payload['registered_k'])}"
    per_intersection: dict[str, Any] = {}
    for ix in calibration_payload["intersection_ids"]:
        entry = calibration_payload["per_intersection"][ix]
        budget = entry["budgets"][key]
        counts = [row["in_support_counts"][ix] for row in dt_rows]
        n_decisions = sum(int(row["decisions"]) for row in dt_rows)
        inside = sum(int(count["in_support"]) for count in counts)
        below = sum(int(count["below"]) for count in counts)
        above = sum(int(count["above"]) for count in counts)
        if inside + below + above != n_decisions:
            raise ValueError(
                f"intersection {ix!r}: the in-support counts ({inside} + {below} + {above}) do not "
                f"add up to the {n_decisions} decisions the cells recorded"
            )
        per_intersection[str(ix)] = {
            "support_range": [float(value) for value in entry["support_range"]],
            "registered_target": float(budget["target"]),
            "target_position": str(budget["in_support"]["position"]),
            "n_cells": len(dt_rows),
            "n_decisions": n_decisions,
            "decisions_in_support": inside,
            "decisions_below": below,
            "decisions_above": above,
        }
    return {
        "what_this_is": (
            "a reliability diagnostic, withdrawn as a selection criterion on 2026-08-13 (BRIEF_15 "
            "§12.1): per intersection, how many of the registered subject's decisions conditioned "
            "on a return-to-go inside that intersection's training support. Nothing selects on it"
        ),
        "ranges_from": P7_3D_CALIBRATION_NAME,
        "ranges_sha256": P7_3D_CALIBRATION_SHA256,
        "per_intersection": per_intersection,
    }


def _grid4x4_h3_block(registered: Mapping[str, Any]) -> dict[str, Any]:
    """H3 on this scenario as A20(e) registers it: clause 1 CONFIRMATORY, clause 2 reported and NOT
    scored, clause 3 void (A19).  Arithmetic beside the inequalities, never a verdict word."""
    clauses: list[dict[str, Any]] = []
    for clause, inequality, meaning, status, scored in (
        (
            1, "rho_sumo(b_mean_k100) > 0", "better than the within-backend fixed-time anchor",
            "CONFIRMATORY on this scenario (A20(e); H3's test row is per paired scenario)", True,
        ),
        (
            2, "rho_sumo(b_mean_k100) < 1", "worse than within-backend MaxPressure",
            "reported as the inequality it is (A20(e)); not scored", False,
        ),
    ):
        for key in ("e_sumo", "att_env"):
            stats = registered[key]
            entry: dict[str, Any] = {
                "clause": clause,
                "inequality": inequality,
                "meaning": meaning,
                "status": status,
                "definition": key,
                "subject": GRID4X4_SUBJECT,
                "n_draws": stats["n_draws"],
                "mean_rho": stats["mean"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            }
            if key == "att_env":
                entry["caveat"] = GRID4X4_ATT_ENV_CAVEAT
            if scored:
                entry["point_estimate_satisfies"] = _satisfies(stats, lambda s: s["mean"] > 0.0)
                entry["ci95_entirely_satisfies"] = _satisfies(stats, lambda s: s["ci95_low"] > 0.0)
            clauses.append(entry)
    return {
        "hypothesis": (
            "Zero-shot transfer is positive but incomplete -- better than fixed-time, worse than "
            "within-backend MaxPressure -- and closes substantially by k = 100"
        ),
        "scenario_key": GRID4X4_SCENARIO_KEY,
        "subject": GRID4X4_SUBJECT,
        "registered_arm": GRID4X4_ARM,
        "primary_definition": "e_sumo",
        "test_row": (
            "MADT zero-shot in SUMO vs the within-backend fixed-time anchor, per paired scenario; "
            "unit: paired evaluation draw (rho.registered_arm.paired_att)"
        ),
        "clauses": clauses,
        "clause_3": (
            "void (A19): 'closes substantially by k = 100' is not tested on grid4x4 -- no few-shot "
            "point and no anchor exist on this scenario"
        ),
        "reported_not_interpreted": (
            "the inequalities and their values are reported; this artifact draws no conclusion. "
            "point_estimate_satisfies and ci95_entirely_satisfies on clause 1 are arithmetic on the "
            "numbers beside them, not verdicts"
        ),
    }


def _grid4x4_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]],
    stage: str | None,
    n_declared: int,
    outside_stage: Sequence[str],
    chunk_commits_by_stage: Mapping[str, set[str]],
    cells_supplied: bool,
    anchors_by_draw: Mapping[int, Mapping[str, Mapping[str, Any]]],
    canary_record: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    identity_by_checkpoint: Mapping[tuple[str, int], Mapping[str, Any]],
    demand_by_draw: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """The ``p7.3d-grid4x4/1.0`` artifact: B.7-2 (i)-(iv) and B.7.1-2's per-intersection block."""
    from offline.transfer_calibration import CANARY_MAX_SECONDS, CANARY_RECORD_NAME

    dt_rows = [row for row in rows if row["kind"] == "dt"]

    by_seed: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        seed_rows = [row for row in dt_rows if row["seed"] == seed]
        if not seed_rows:
            continue
        entry: dict[str, Any] = {"seed": int(seed), "n_draws": len(seed_rows)}
        for key in ("e_sumo", "att_env"):
            usable = [row[f"rho_{key}"] for row in seed_rows if row[f"rho_{key}"] is not None]
            entry[f"mean_rho_{key}"] = float(sum(usable) / len(usable)) if usable else None
            entry[f"n_draws_{key}"] = len(usable)
        by_seed.append(entry)

    per_draw = {key: _seed_means_by_draw(dt_rows, f"rho_{key}") for key in ("e_sumo", "att_env")}
    seeds_by_draw: dict[int, int] = {}
    for row in dt_rows:
        seeds_by_draw[int(row["draw_id"])] = seeds_by_draw.get(int(row["draw_id"]), 0) + 1
    by_draw = {
        str(draw): {
            "e_sumo": per_draw["e_sumo"].get(draw),
            "att_env": per_draw["att_env"].get(draw),
            "n_seeds": count,
        }
        for draw, count in sorted(seeds_by_draw.items())
    }

    registered: dict[str, Any] = {
        "subject": GRID4X4_SUBJECT,
        "arm": GRID4X4_ARM,
        "role": "registered_prompt (A20(b), A21(a))",
        "e_sumo": _definition_stats(per_draw["e_sumo"]),
        "att_env": _definition_stats(per_draw["att_env"]),
    }
    registered["paired_att"] = {
        key: {
            anchor: _paired_block(rows, dt_rows, GRID4X4_SUBJECT, GRID4X4_ARM, anchor, key)
            for anchor in ("fixedtime", "maxpressure")
        }
        for key in ("e_sumo", "att_env")
    }

    denominators: dict[str, dict[int, float]] = {
        key: {
            draw: anchor_denominator(anchors, key) for draw, anchors in sorted(anchors_by_draw.items())
        }
        for key in ("e_sumo", "att_env")
    }
    excluded_env_draws = [
        {
            "draw_id": draw,
            "att_fixedtime": float(anchors_by_draw[draw]["fixedtime"]["att_env"]),
            "att_maxpressure": float(anchors_by_draw[draw]["maxpressure"]["att_env"]),
            "why": (
                "the two anchors have EXACTLY equal env ATT on this draw, so rho's denominator is "
                "zero and the ratio is undefined; excluded from this definition's means, CI and "
                "H3 clauses, and reported here"
            ),
        }
        for draw, value in sorted(denominators["att_env"].items())
        if value == 0.0
    ]
    ix_ids = [str(ix) for ix in calibration_payload["intersection_ids"]]

    return {
        "format_version": GRID4X4_ARTIFACT_FORMAT_VERSION,
        "registered_in": (
            "PREREGISTRATION H3, §3.4, A15, A16, A17(e), A18(c), A20, A21; BRIEF_39 Amendments B.7 "
            "and B.7.1"
        ),
        "scenario_key": GRID4X4_SCENARIO_KEY,
        "subject": GRID4X4_SUBJECT,
        "arms": {
            "evaluated": {"registered": [GRID4X4_ARM], "anchors": ["fixedtime", "maxpressure"]},
            "not_evaluated": {
                "naive": "A21(b)(i): removed by declaration before any grid4x4 SUMO number existed",
                "random": "A21(b)(ii): removed by declaration before any grid4x4 SUMO number existed",
                "b_max_k100": "A20(b): not evaluated on grid4x4 (it tracked b_mean within 0.01 on hz1x1)",
                "a_q1.0": "A20(b): not evaluated on grid4x4 (out of support for mappo1000 on hz1x1)",
            },
        },
        "stage": stage,
        "n_cells_declared": n_declared,
        "n_chunks_outside_stage": len(outside_stage),
        "chunk_commits_by_stage": {
            stage_name: sorted(commits)
            for stage_name, commits in sorted(chunk_commits_by_stage.items())
        },
        "cell_set_source": "caller-supplied declaration" if cells_supplied else "declared_cells()",
        "halting_check_draw": HALTING_CHECK_DRAW,
        "intersection_ids": ix_ids,
        "cells": [dict(row) for row in rows],
        "series_location": (
            "the per-decision RTG and reward series and the action matrices of every cell are in "
            "its chunk under output/p7_3d/cells/, covered by output/SHA256SUMS_p7_3d.txt; they are "
            "not republished here (about 130 MB on sixteen intersections)"
        ),
        "reference_cells": dict(reference_record),
        "rho": {
            "formula": "rho = (ATT_fixedtime - ATT_arm) / (ATT_fixedtime - ATT_maxpressure)",
            "definitions": {
                "e_sumo": {
                    "what": (
                        "A15's primary: the pool-clock ATT over the all-created population, from "
                        "the observer (P7.1's key att_reference_created_population; one quantity)"
                    ),
                    "role": "REGISTERED PRIMARY (A15)",
                    "denominator_diagnostic": denominator_diagnostic(denominators["e_sumo"]),
                    "zero_denominator_rule": (
                        "REFUSED: on the registered primary a zero denominator is a finding about "
                        "the instrument, and rho() raises (the author's ruling of 2026-09-17)"
                    ),
                },
                "att_env": {
                    "what": "the admitted pair beside it: the env's own metric at the horizon",
                    "role": "co-reported (A15)",
                    "caveat": GRID4X4_ATT_ENV_CAVEAT,
                    "denominator_diagnostic": denominator_diagnostic(denominators["att_env"]),
                    "zero_denominator_rule": (
                        "RECORDED AND EXCLUDED (the author's ruling of 2026-09-17): the draw's cells "
                        "keep rho_att_env null, the draw is listed in excluded_draws, and it counts "
                        "towards no mean, no CI and no H3 clause on this definition"
                    ),
                    "excluded_draws": excluded_env_draws,
                    "n_draws_used": len(denominators["att_env"]) - len(excluded_env_draws),
                    "n_draws_total": len(denominators["att_env"]),
                },
            },
            "estimator": {
                "method": (
                    "analytic normal approximation, 1.96*s/sqrt(n) over the per-draw seed means "
                    "(offline.dt_gate.mean_ci95), as on hz1x1"
                ),
                "resampling_seed": None,
                "unit": "one paired evaluation draw; seeds averaged within a draw, as in P4",
            },
            "not_clipped": (
                "PREREGISTRATION §3.4: values may exceed 1 or fall below 0 and are not clipped. "
                "fixed-time is 0 and MaxPressure is 1 by construction"
            ),
            "by_seed": by_seed,
            "by_draw": by_draw,
            "registered_arm": registered,
        },
        "per_intersection_rho": _grid4x4_per_intersection_rho(rows, ix_ids),
        "h3": _grid4x4_h3_block(registered),
        "in_support": _grid4x4_in_support_block(dt_rows, calibration_payload),
        "canary": {
            "seconds": canary_record["seconds"],
            "threshold_seconds": CANARY_MAX_SECONDS,
            "verdict": "at speed" if canary_record["seconds"] <= CANARY_MAX_SECONDS else "throttled",
            "observed": dict(canary_record["facts"]),
            "source": (
                f"{CANARY_RECORD_NAME} in the work directory, written by the driver from the canary "
                "line right after the token; report re-ran check_canary on these facts"
            ),
            "git_commit": canary_record.get("git_commit"),
            "git_dirty": canary_record.get("git_dirty"),
            "chunk_canaries_by_stage": _canaries_by_stage(chunks),
        },
        "what_this_does_not_say": [*A21_SCOPE_SENTENCES, *GRID4X4_WHAT_THIS_DOES_NOT_SAY],
        "inputs": {
            "calibration_artifact": P7_3D_CALIBRATION_NAME,
            "calibration_sha256": P7_3D_CALIBRATION_SHA256,
            "reference_cells_artifact": P7_3D_REFERENCE_CELLS_NAME,
            "reference_cells_sha256": P7_3D_REFERENCE_CELLS_SHA256,
            "checkpoints": sorted(
                (dict(identity) for identity in identity_by_checkpoint.values()),
                key=lambda entry: (entry["subject"], entry["seed"]),
            ),
            "demand_by_draw": {
                str(draw): {k: v for k, v in demand.items() if k != "config_path"}
                for draw, demand in sorted(demand_by_draw.items())
            },
        },
        **_git_provenance(),
    }


# ======================================================================================
# The pool: one process per worker, one env per cell, resume by content
# ======================================================================================

#: Amendment C3: 12 workers, measured 1.27x better than 8 on this machine's 16 cores.
DEFAULT_WORKERS = 12

#: F1's pre-flight pilot.  **Draw 5 is P7.2b's FENCED smoke draw and is deliberately NOT in the
#: held-out pool** -- *"a pilot on the held-out pool would BE the experiment, run before the token"*
#: (``docs/plans/p7.3a_amendment_a_measurements.md`` §A6).  Two seeds across both subjects and all
#: four declared arms, so the pilot exercises the target lookup for every arm rather than running
#: one arm sixteen times: the pilot's FIRST purpose is the mechanics -- destruction and resume, one
#: env per process, chunks resumable by content -- and its rate is a by-product (F1).
PILOT_DRAW = 5
PILOT_SEEDS: tuple[int, ...] = (101, 202)
PILOT_STAGE = "pilot"


def pilot_cells() -> list[dict[str, Any]]:
    """F1's sixteen cells, declared as a set before any of them runs.

    ``PILOT_STAGE`` is not one of :data:`STAGES`: the pilot is not part of the campaign, and a
    pilot chunk is therefore not a declared cell of any stage, which is what makes :func:`report`
    refuse a pilot work directory outright.  The fence is that refusal, not the instruction never
    to call ``report`` on it.
    """
    return [
        {
            "kind": "dt",
            "subject": subject,
            "arm": spec.name,
            "seed": int(seed),
            "draw_id": PILOT_DRAW,
            "stage": PILOT_STAGE,
        }
        for subject in SUBJECTS
        for spec in DECLARED_ARMS
        for seed in PILOT_SEEDS
    ]


def anchor_pilot_cells() -> list[dict[str, Any]]:
    """The anchor's own pre-flight pilot: its cell on the FENCED smoke draw, both pilot seeds.

    P7.3b's schedule may not be quoted from P7.3a's campaign -- F3: *the header must never quote a
    rate for a machine state the run did not have* -- and the anchor runs a different checkpoint
    per cell through a different prompt source.  This is the smallest set that exercises that path
    end to end.

    Like :func:`pilot_cells` these carry ``PILOT_STAGE``, which is deliberately not one of
    :data:`STAGES`, so a pilot chunk is a declared cell of no stage and :func:`report` refuses the
    directory outright.  The two rho anchors are included because they go through the same runner
    and A6 measured them SLOWER than a DT cell, so a DT-only pilot would under-count.
    """
    cells: list[dict[str, Any]] = [
        {
            "kind": "dt",
            "subject": ANCHOR_SUBJECT,
            "arm": ANCHOR_ARM,
            "seed": int(seed),
            "draw_id": PILOT_DRAW,
            "stage": PILOT_STAGE,
        }
        for seed in PILOT_SEEDS
    ]
    cells.extend(
        {
            "kind": "anchor",
            "subject": None,
            "arm": arm,
            "seed": None,
            "draw_id": PILOT_DRAW,
            "stage": PILOT_STAGE,
        }
        for arm in ("fixedtime", "maxpressure")
    )
    return cells


def _worker(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """One cell in one process.  Top-level so ``spawn`` can pickle it.

    A failure is RETURNED, never raised into the pool: one broken cell must not take the campaign
    down, and the caller decides whether to stop.  Nothing is written unless the cell validated.
    """
    cell, kwargs = task
    try:
        payload = run_cell(cell, **kwargs["run"])
        path = write_chunk(payload, work_dir=kwargs["work_dir"])
        return {"name": path.name, "ok": True, "seconds": payload["seconds"], "error": None}
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        return {
            "name": cell_chunk_name(cell),
            "ok": False,
            "seconds": None,
            "error": f"{type(error).__name__}: {error}",
        }


def run_stage(
    *,
    work_dir: str | Path,
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
    stage: str | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
    canary_seconds: float | None = None,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
) -> dict[str, Any]:
    """Every cell of *stage* that is not already on disk as a reusable chunk.

    **The skip decision is in Python, not in the shell.**  ``offline/campaigns/p5_3b.sh`` skipped on
    ``[ -f ]`` alone, and a bad chunk then survived every restart until someone deleted it by hand;
    :func:`chunk_is_reusable` re-derives the verdict from the chunk's own content and from the files
    on disk instead.  A chunk that exists and is NOT reusable is moved aside to ``failed/`` -- never
    overwritten, because it is evidence about a run.

    ``spawn`` rather than ``fork``: a forked worker inherits the parent's traci module state, and
    one env per cell means the simulator is created and closed inside the worker that uses it.
    """
    from multiprocessing import get_context

    work = Path(work_dir)
    declared = list(declared_cells(stage) if cells is None else cells)
    if limit is not None:
        declared = declared[: int(limit)]

    identity = {"out_root": str(out_root), "output_root": str(output_root), "data_dir": None if data_dir is None else str(data_dir)}
    todo: list[dict[str, Any]] = []
    reused: list[str] = []
    for cell in declared:
        path = chunk_path(cell, work_dir=work)
        if reusable_chunk_at(path, cell=cell, **identity):
            reused.append(path.name)
            continue
        if path.exists():
            move_aside(path)
        todo.append(dict(cell))

    kwargs = {
        "run": {**identity, "canary_seconds": canary_seconds},
        "work_dir": str(work),
    }
    import time

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    if todo:
        work.mkdir(parents=True, exist_ok=True)
        context = get_context("spawn")
        with context.Pool(processes=max(1, int(workers))) as pool:
            for result in pool.imap_unordered(_worker, [(cell, kwargs) for cell in todo]):
                results.append(result)
                # The cell's NAME and whether it stood, never a number it produced.
                status = "ok" if result["ok"] else f"FAILED {result['error']}"
                print(f"  {result['name']} {status}", flush=True)
    wall_seconds = time.perf_counter() - started

    failures = [result for result in results if not result["ok"]]
    return {
        "stage": stage,
        "n_declared": len(declared),
        "n_reused": len(reused),
        "n_rolled": len(results) - len(failures),
        "n_failed": len(failures),
        "failures": failures,
        # Wall clocks only -- the in-process seconds of each cell and the pool's own elapsed time.
        # Nothing here is an outcome: a duration is not an ATT (F1's fence).
        "seconds": [result["seconds"] for result in results if result["seconds"] is not None],
        "wall_seconds": wall_seconds,
    }


def run_pilot(
    *,
    work_dir: str | Path,
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    transcript_path: str | Path | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """F1's pre-flight pilot: :func:`pilot_cells` through the REAL pool, and nothing published.

    **FENCED.**  Every chunk it writes contains an outcome -- it has to, because the thing under
    test is the runner that produces them -- but no ATT, no rho and no return is printed,
    summarised or returned from here, and :func:`report` is never called on the directory.  That is
    a procedure, so the mechanical half matters more: a pilot cell is not a declared cell of any
    stage, so ``report`` refuses this work directory if anyone ever points it at one.

    The canary runs FIRST and both halves must pass: a rate measured on a throttled machine is not
    a rate (``PROJECT_PLAN`` §7), and the transcript carries the canary line so the number can be
    read against the machine state it was taken on (A11).

    Returns wall clocks and counts only.
    """
    import time

    from offline.transfer_calibration import canary_seconds as measure_canary
    from offline.transfer_calibration import check_canary, format_canary_line

    canary, facts = measure_canary()
    # PRINT FIRST, THEN CHECK (J3, reviewer min-7; the `canary` subcommand has done this since H8).
    # The transcript is written after the run, so on a failed correctness half the pane is the only
    # place the observed values would ever appear.
    canary_line = format_canary_line(canary, facts)
    print(canary_line, flush=True)
    check_canary(facts)

    # `cells` lets P7.3b hand in `anchor_pilot_cells()`; the default is F1's sixteen.
    # Whatever is passed, every cell must carry PILOT_STAGE -- the fence is that a pilot
    # chunk is a declared cell of no stage, so `report` refuses the directory outright.
    cells = list(pilot_cells() if cells is None else cells)
    off_stage = sorted({str(c.get('stage')) for c in cells if c.get('stage') != PILOT_STAGE})
    if off_stage:
        raise ValueError(
            f"a pilot cell set may only carry stage {PILOT_STAGE!r}; got {off_stage}. A cell "
            "on a campaign stage would write a chunk report would then publish, and the pilot "
            "is fenced precisely because its outcomes are not evidence about anything"
        )
    started = time.perf_counter()
    summary = run_stage(
        work_dir=work_dir,
        out_root=out_root,
        output_root=output_root,
        data_dir=data_dir,
        cells=cells,
        canary_seconds=canary,
        workers=workers,
    )
    elapsed = time.perf_counter() - started

    rolled = list(summary.get("seconds", ()))
    # The rate basis is the POOL's wall over the cells it actually rolled, which is what A6's
    # "wall (16 cells)" means and what F3 will compare a pad re-run against. `elapsed` is this
    # route's own clock and additionally covers the reusability scan, so it is reported beside it
    # rather than in place of it -- and a pilot that reused every chunk has no rate at all.
    pool_wall = float(summary.get("wall_seconds", elapsed))
    n_rolled = int(summary["n_rolled"])
    record: dict[str, Any] = {
        "what_this_is": (
            "BRIEF_37 Amendment F1's pre-flight pilot: the section 3.5b runner through the real "
            "pool. A MECHANICS check first and a rate second"
        ),
        "fenced": (
            "no ATT, no rho and no episode outcome is printed, summarised or published here; "
            "report is never called on this work directory, and report REFUSES it. MEASURED "
            "2026-09-18 on a copy of this directory: the refusal that actually fires is the "
            "COMPLETENESS one -- 700 declared cells have no chunk -- because report checks "
            "missing before extra. The undeclared-cell refusal is real and is reached only if "
            "completeness passes, which it cannot here. The earlier wording named the second as "
            "if it were the first; the fence holds either way, and a claim about WHICH refusal "
            "guards a file is the kind of thing that stops being true silently"
        ),
        "draw_id": PILOT_DRAW,
        "why_this_draw": (
            "draw 5 is P7.2b's fenced smoke draw and is not in the held-out pool; a pilot on the "
            "pool would BE the experiment, run before the token"
        ),
        "n_cells": len(cells),
        "subjects": sorted({str(cell["subject"]) for cell in cells}),
        "arms": sorted({str(cell["arm"]) for cell in cells}),
        # ⚠️ `seed` is None on an ANCHOR cell -- fixedtime and MaxPressure have no training seed --
        # and this used to be `int(cell["seed"])` over every cell. F1's sixteen are all `dt`, so
        # nothing noticed until P7.3b's pilot included rho's two denominators: all four cells
        # rolled fine and then the TRANSCRIPT raised TypeError, losing the rate the pilot exists to
        # measure. Found by RUNNING the pilot, which is what a pre-flight is for.
        "seeds": sorted({int(cell["seed"]) for cell in cells if cell["seed"] is not None}),
        "n_cells_without_a_training_seed": sum(1 for cell in cells if cell["seed"] is None),
        "workers": int(workers),
        "halting_check": halting_check_for(PILOT_DRAW),
        "canary_seconds": canary,
        "canary_line": canary_line,
        "n_rolled": summary["n_rolled"],
        "n_reused": summary["n_reused"],
        "n_failed": summary["n_failed"],
        "failures": summary["failures"],
        "wall_seconds": pool_wall,
        "route_wall_seconds": elapsed,
        "effective_seconds_per_cell": (pool_wall / n_rolled) if n_rolled else None,
        "in_process_mean_seconds": (sum(rolled) / len(rolled)) if rolled else None,
        "speed_up": (sum(rolled) / pool_wall) if rolled and pool_wall > 0 else None,
        "a11": (
            "measured under thermal constraint unless a cooling-pad canary says otherwise; the "
            "only rate that governs a stage is the canary the driver runs at that stage's start"
        ),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_git_provenance(),
    }

    if transcript_path is not None:
        _write_pilot_transcript(Path(transcript_path), record)
    return record


def _write_pilot_transcript(path: Path, record: Mapping[str, Any]) -> None:
    """The pilot's transcript: the canary line, the date, n and the rate -- and no outcome.

    Appended, never truncated, like every log this project writes: a second pilot (F3's pad re-run)
    must not destroy the first one's record, because the comparison between them is the point.
    """
    lines = [
        "=== P7.3a pre-flight pilot (BRIEF_37 Amendment F1 / I5(2))",
        f"date: {record['date']}",
        f"git_commit: {record['git_commit']}  git_dirty: {record['git_dirty']}",
        record["canary_line"],
        f"cells: n = {record['n_cells']} DT cells on draw {record['draw_id']} "
        f"(FENCED smoke draw), subjects {record['subjects']}, arms {record['arms']}, "
        f"seeds {record['seeds']}",
        f"workers: {record['workers']}   halting_check: {record['halting_check']}",
        f"rolled: {record['n_rolled']}   reused: {record['n_reused']}   "
        f"failed: {record['n_failed']}",
        f"wall: {record['wall_seconds']:.2f} s",
        f"effective s/cell: {record['effective_seconds_per_cell']:.3f}"
        if record["effective_seconds_per_cell"] is not None
        else "effective s/cell: n/a",
        f"in-process mean s/cell: {record['in_process_mean_seconds']:.2f}"
        if record["in_process_mean_seconds"] is not None
        else "in-process mean s/cell: n/a",
        f"speed-up: {record['speed_up']:.2f}x" if record["speed_up"] is not None else "speed-up: n/a",
        f"A11: {record['a11']}",
        f"FENCED: {record['fenced']}",
    ]
    for failure in record["failures"]:
        lines.append(f"FAILURE {failure['name']}: {failure['error']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n\n")


# ======================================================================================
# The B.6 fix round's driver-facing pieces: the artifact's name per stage, the manifest, the
# inputs by digest (m1), git resolvability before the token (m2), and B.5-1's DT re-roll check
# ======================================================================================

#: B.6-2(3): the P7.3d manifest covers the directory of this name and nothing else, and sits
#: beside it, as every campaign's ``output/SHA256SUMS_<campaign>.txt`` does.
MANIFEST_CAMPAIGN_NAME = "p7_3d"
MANIFEST_NAME = "SHA256SUMS_p7_3d.txt"


def artifact_name_for_stage(stage: str | None) -> str:
    """One stage, one artifact name (B.6-2(3)).

    P7.3a's and P7.3b's names are what they were; the grid4x4 stage writes ``p7_3d_grid4x4.json``,
    the name its driver's header promises -- at ``8c79778`` it fell through to P7.3a's
    ``p7_3a_zero_shot.json``.  Writing one campaign's artifact under another's name is how the
    zero-shot point would be overwritten by something else.  (The ``name = ...`` chain is the
    spelling ``main`` used before the extraction; P7.3b's text pin reads it.)
    """
    if stage == STAGE_GRID4X4:
        name = "p7_3d_grid4x4.json"
    elif stage == STAGE_ANCHOR:
        name = "p7_3b_anchor.json"
    elif stage == STAGE_CONFIRMATORY:
        name = "p7_3a_zero_shot_stage1.json"
    else:
        name = "p7_3a_zero_shot.json"
    return name


def _campaign_files(root: Path) -> list[str]:
    """Every regular file under *root*, as sorted POSIX paths relative to it (no symlinks)."""
    import os

    found: list[str] = []
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        subdirectories.sort()
        for name in names:
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def write_manifest(*, campaign_dir: str | Path) -> dict[str, Any]:
    """``output/SHA256SUMS_p7_3d.txt`` over ``output/p7_3d/`` ONLY: atomic, then re-verified.

    B.6-2(3).  Refusals first -- the directory must be named ``p7_3d`` and hold at least one file --
    then every regular file under it is hashed into ``<parent>/SHA256SUMS_p7_3d.txt.tmp`` in
    ``sha256sum`` format (two spaces; paths relative to the parent, so ``sha256sum -c`` run from
    ``output/`` checks it), the temporary file REPLACES the manifest in one ``os.replace``, and
    the written manifest is read back and every listed file hashed again: the file set must not
    have moved and every digest must still hold.  ``p7_3b_anchor.sh:353-357``'s shape, in Python so
    a test can reach every branch.
    """
    import os

    root = Path(campaign_dir)
    if root.name != MANIFEST_CAMPAIGN_NAME:
        raise ValueError(
            f"{root}: the P7.3d manifest covers a directory named {MANIFEST_CAMPAIGN_NAME!r} and "
            "nothing else -- another campaign's files in this manifest would be certified by a "
            "run that did not produce them"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    files = _campaign_files(root)
    if not files:
        raise ValueError(f"{root} holds no file; an empty manifest certifies nothing")

    lines = [f"{_sha256_file(root / relative)}  {root.name}/{relative}" for relative in files]
    target = root.parent / MANIFEST_NAME
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)

    listed = _manifest_digests(target)
    now = sorted(f"{root.name}/{relative}" for relative in _campaign_files(root))
    if sorted(listed) != now:
        raise ValueError(
            f"{target}: the file set moved while the manifest was written (listed "
            f"{len(listed)}, on disk {len(now)}); re-run the manifest from a quiet directory"
        )
    moved = sorted(name for name, digest in listed.items() if _sha256_file(root.parent / name) != digest)
    if moved:
        raise ValueError(
            f"{target}: re-verification failed on {moved[:4]} -- a file changed after it was hashed"
        )
    return {"path": str(target), "n_files": len(lines), "sha256": _sha256_file(target)}


def check_campaign_inputs(
    *, data_dir: str | Path, output_root: str | Path, draws_root: str | Path
) -> list[str]:
    """m1 (B.6-2(4)): the campaign's inputs BY DIGEST, before the token.  Raises naming each miss.

    The driver checked ``p7_3d_calibration.json``, ``p7_3d_reference_cells.json`` and
    ``p7_3d_cap_e.json`` by EXISTENCE only; here each is hashed and compared with the pin this
    module carries.  With them, and for the same reason: the five checkpoints against A20(a)
    (``BRIEF_39`` §3 C6 lists them *at their digests*), and RESCO's network through the relative
    reference draw 1000's parity provenance records (Amendment A8) -- the one input a parity
    config's digest does not cover, because the config only NAMES the net.  Returns one line per
    input, every digest printed, so the capture records what the run started from.
    """
    from offline.materialise_draws import parity_sumocfg_path
    from offline.parity import GRID4X4_SCENARIO
    from offline.transfer_calibration import (
        GRID4X4_CHECKPOINT_SHA256,
        GRID4X4_CHECKPOINT_STEM,
        GRID4X4_CHECKPOINT_SUBDIR,
    )

    data = Path(data_dir)
    checks: list[tuple[str, Path, str]] = [
        (name, data / name, pin)
        for name, pin in (
            (P7_3D_CALIBRATION_NAME, P7_3D_CALIBRATION_SHA256),
            (P7_3D_REFERENCE_CELLS_NAME, P7_3D_REFERENCE_CELLS_SHA256),
            (P7_3D_CAP_E_NAME, P7_3D_CAP_E_SHA256),
        )
    ]
    for seed, pin in sorted(GRID4X4_CHECKPOINT_SHA256.items()):
        checks.append(
            (
                f"{GRID4X4_SUBJECT} seed{seed}",
                Path(output_root) / GRID4X4_CHECKPOINT_SUBDIR / f"{GRID4X4_CHECKPOINT_STEM}{seed}.pt",
                pin,
            )
        )
    problems: list[str] = []
    parity_dir = parity_sumocfg_path(
        GRID4X4_SCENARIO_KEY, HALTING_CHECK_DRAW, out_root=draws_root
    ).parent
    provenance_path = parity_dir / PARITY_PROVENANCE_NAME
    net_pin = GRID4X4_SCENARIO.external.net_sha256  # type: ignore[union-attr]
    if provenance_path.is_file():
        net = json.loads(provenance_path.read_bytes())["net"]
        if str(net["sha256"]) != net_pin:
            problems.append(
                f"{provenance_path} records the network at sha256 {net['sha256']}, not {net_pin}"
            )
        checks.append(
            ("the RESCO network grid4x4.net.xml (A8)", parity_dir / str(net["reference"]), net_pin)
        )
    else:
        problems.append(f"{provenance_path} is absent, so the network reference cannot be resolved")

    lines: list[str] = []
    for label, path, pin in checks:
        if not path.is_file():
            problems.append(f"{label}: {path} is absent")
            continue
        digest = _sha256_file(path)
        if digest != pin:
            problems.append(f"{label}: {path} has sha256 {digest}, not the pinned {pin}")
        else:
            lines.append(f"input {label}: sha256 {digest} (the pin)")
    if problems:
        raise ValueError(
            f"{len(problems)} input(s) are not what the module pins: " + "; ".join(problems)
        )
    return lines


def unresolvable_chunk_commits(*, work_dir: str | Path, stage: str) -> list[dict[str, str]]:
    """m2 (B.6-2(4)): the chunks whose ``git_commit`` git cannot resolve -- found BEFORE the token.

    :func:`chunk_is_reusable` lets ``_git``'s ``RuntimeError`` propagate on an unknown revision, on
    purpose (J1(c): a git that cannot answer is an environment failure, not a verdict).  Inside
    ``run_stage`` that raise came AFTER ``rm -f "$TOKEN"``: the token lost, nothing created, a new
    token needed (the pre-flight's m2).  This walks the SAME path up to that call -- every declared
    chunk on disk that parses and validates against its cell -- and resolves each distinct commit
    now.  A chunk that fails validation is moved aside by ``run_stage`` and never reaches git, so it
    is not reported.  Reads only; creates nothing.
    """
    work = Path(work_dir)
    verdicts: dict[str, str | None] = {}
    problems: list[dict[str, str]] = []
    for cell in declared_cells(stage):
        path = chunk_path(cell, work_dir=work)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_bytes())
        except (ValueError, OSError):
            continue
        if not isinstance(payload, Mapping):
            continue
        try:
            validate_cell_payload(payload, cell=cell)
        except (KeyError, TypeError, ValueError, AttributeError, FileNotFoundError):
            continue
        commit = str(payload["git_commit"])
        if commit not in verdicts:
            try:
                code_changed_since(commit)
                verdicts[commit] = None
            except RuntimeError as error:
                verdicts[commit] = str(error)
        if verdicts[commit] is not None:
            problems.append({"chunk": path.name, "git_commit": commit, "error": str(verdicts[commit])})
    return problems


#: B.5-1: the format of each record the re-roll check writes, the ONE clock field a chunk carries
#: (``run_cell``'s ``time.perf_counter`` difference; ``canary_seconds`` is the value the caller
#: passes, identical by construction), and the fields B.5-1 names -- each must be PRESENT in every
#: roll, or the comparison refuses: a comparison over absent fields is not a comparison.
REROLL_FORMAT_VERSION = "p7.3d-dt-reroll-check/1.0"
REROLL_CLOCK_FIELDS: tuple[str, ...] = ("seconds",)
REROLL_COMPARED_FIELDS: tuple[str, ...] = (
    "actions", "rtg_series", "reward_series", "e_sumo", "att_env", "episode_reward", "n_teleports",
)
REROLL_SEED = 101


def dt_reroll_check_cell() -> dict[str, Any]:
    """B.5-1's ONE cell: the registered subject, seed 101, the registered prompt, draw 5.

    Draw 5 is the fenced smoke draw -- outside the held-out pool and the probe band (Amendment
    B.2-2) -- and the stage is :data:`PILOT_STAGE`, a stage no declaration contains, so a stray
    re-roll chunk is a declared cell of nothing and ``report`` would refuse it.
    """
    return {
        "kind": "dt",
        "subject": GRID4X4_SUBJECT,
        "arm": GRID4X4_ARM,
        "seed": REROLL_SEED,
        "draw_id": PILOT_DRAW,
        "scenario": GRID4X4_SCENARIO_KEY,
        "stage": PILOT_STAGE,
    }


def _canonical_bytes(value: Any) -> bytes:
    """The bytes :func:`_write_json` writes for *value* -- what a chunk IS on disk."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compare_reroll_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every roll against the FIRST, field by field, under ``==`` -- clocks excluded BY NAME.

    Each field is compared as the bytes it is written as (:func:`_canonical_bytes`), which is
    ``==`` on the evidence itself: it is exact for every finite float (``repr`` round-trips), it
    tells ``-0.0`` from ``0.0``, and it does not call two bit-identical ``NaN`` values different.
    The sha256 of each whole roll minus the clocks is the SECOND route to the same verdict -- all
    digests equal if and only if no field differs -- and a disagreement between the two routes
    raises rather than choosing one.  Returns names and digests only, never a value.
    """
    rolls = [dict(payload) for payload in payloads]
    if len(rolls) < 2:
        raise ValueError("the re-roll check needs at least two rolls of the cell to compare")
    for index, roll in enumerate(rolls):
        missing = [name for name in REROLL_COMPARED_FIELDS if name not in roll]
        if missing:
            raise ValueError(
                f"roll {index} carries no {missing}; B.5-1 compares these fields and a comparison "
                "over absent fields is not a comparison"
            )
    stripped = [
        {key: value for key, value in roll.items() if key not in REROLL_CLOCK_FIELDS}
        for roll in rolls
    ]
    import hashlib

    digests = [hashlib.sha256(_canonical_bytes(roll)).hexdigest() for roll in stripped]
    reference = stripped[0]
    differing: set[str] = set()
    for other in stripped[1:]:
        for key in set(reference) | set(other):
            if key not in reference or key not in other:
                differing.add(key)
            elif _canonical_bytes(reference[key]) != _canonical_bytes(other[key]):
                differing.add(key)
    if (not differing) != (len(set(digests)) == 1):
        raise AssertionError(
            "the field-by-field comparison and the whole-roll digests disagree; refusing to choose"
        )
    return {
        "verdict": "IDENTICAL" if not differing else "NOT IDENTICAL",
        "differing_fields": sorted(differing),
        "sha256_minus_clocks": digests,
        "n_rolls": len(rolls),
        "n_fields_compared": len(reference),
        "clock_fields_excluded": list(REROLL_CLOCK_FIELDS),
    }


def _reroll_worker(task: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    """One roll of the re-roll cell in one spawned process, through :func:`run_cell` itself.

    A failure is RETURNED, as :func:`_worker` returns it; the caller decides.  Top-level so
    ``spawn`` can pickle it.
    """
    cell, kwargs, role = task
    try:
        return {"role": role, "ok": True, "payload": run_cell(cell, **kwargs), "error": None}
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        return {"role": role, "ok": False, "payload": None, "error": f"{type(error).__name__}: {error}"}


def run_dt_reroll_check(
    *,
    g2_dir: str | Path,
    out_root: str | Path,
    output_root: str | Path,
    data_dir: str | Path | None = None,
    canary_seconds: float | None = None,
    workers: int = DEFAULT_WORKERS,
    worker: Any = None,
) -> dict[str, Any]:
    """B.5-1: ONE DT cell rolled at W = 1 and inside a *workers*-wide pool, compared under ``==``.

    No DT evaluation cell had ever been rolled twice and compared (B.5-1, checked from disk by the
    coordinator): every bitwise claim so far was a recomputation or a non-DT re-roll.  The cell
    (:func:`dt_reroll_check_cell`) is rolled through :func:`run_cell` -- the campaign's own code
    path, device choice included -- once in a spawn pool of ONE worker, then *workers* times in ONE
    spawn pool of that width, so the campaign's pooling (twelve concurrent processes on one GPU) is
    exercised; the pool is G2's ``measure_pool`` shape, twelve copies of the same cell, and EVERY
    pooled roll is compared with the W = 1 roll (:func:`compare_reroll_payloads`).

    Nothing is written until every roll has returned and been compared.  Then one record per roll
    goes to ``<g2_dir>/dt_reroll_check_<UTC>/<role>.json`` with the whole chunk under
    :data:`FENCED_KEY` (``report`` never globs ``g2/``), plus ``verdict.json``; an existing run
    directory refuses, so no record is ever overwritten.  A roll that FAILED is not a verdict: the
    check raises naming the exception types only -- a failure message can carry an outcome value
    (B.2-2(3)'s known route) -- and parks the messages under the fence in ``failures.json``.

    The returned ``line`` is what the driver prints and copies into the campaign capture's header:
    IDENTICAL or NOT IDENTICAL, the differing field NAMES, the sha256 of each roll minus the
    clocks, and no value.
    """
    import time
    from multiprocessing import get_context

    cell = dt_reroll_check_cell()
    width = int(workers)
    if width < 1:
        raise ValueError(f"the pool needs at least one worker, got {workers}")
    run_dir = Path(g2_dir) / f"dt_reroll_check_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; a re-roll record is never overwritten")
    kwargs = {
        "out_root": str(out_root),
        "output_root": str(output_root),
        "data_dir": None if data_dir is None else str(data_dir),
        "canary_seconds": canary_seconds,
    }
    roll = _reroll_worker if worker is None else worker
    roles = [f"pool_{index:02d}" for index in range(width)]
    context = get_context("spawn")
    with context.Pool(processes=1) as pool:
        single = pool.apply(roll, ((cell, kwargs, "w1"),))
    with context.Pool(processes=width) as pool:
        pooled = list(pool.imap_unordered(roll, [(cell, kwargs, role) for role in roles]))
    results = [single, *sorted(pooled, key=lambda result: str(result["role"]))]

    failures = [result for result in results if not result["ok"]]
    if failures:
        _write_json(
            run_dir / "failures.json",
            {
                "format_version": REROLL_FORMAT_VERSION,
                FENCED_KEY: {
                    "cell": cell,
                    "failures": [{"role": r["role"], "error": r["error"]} for r in failures],
                },
                **_git_provenance(),
            },
        )
        kinds = sorted({str(result["error"]).split(":", 1)[0] for result in failures})
        raise RuntimeError(
            f"dt_reroll_check could not run: {len(failures)} of {len(results)} roll(s) failed "
            f"({kinds}); no verdict is invented. The messages are FENCED in "
            f"{run_dir / 'failures.json'} because a failure message can carry an outcome value"
        )

    comparison = compare_reroll_payloads([result["payload"] for result in results])
    name = cell_chunk_name(cell)
    by_role = dict(zip([str(r["role"]) for r in results], comparison["sha256_minus_clocks"]))
    if comparison["verdict"] == "IDENTICAL":
        line = (
            f"dt_reroll_check IDENTICAL: {len(results)} rolls of {name} (1 at W=1, {width} in one "
            f"{width}-worker pool) agree under == on all {comparison['n_fields_compared']} fields "
            f"but the clock {list(REROLL_CLOCK_FIELDS)}; sha256 minus clocks "
            f"{comparison['sha256_minus_clocks'][0]} on all {len(results)}"
        )
    else:
        line = (
            f"dt_reroll_check NOT IDENTICAL: {len(results)} rolls of {name} (1 at W=1, {width} in "
            f"one {width}-worker pool) differ on {comparison['differing_fields']}; sha256 minus "
            "clocks by roll: " + ", ".join(f"{role} {digest}" for role, digest in by_role.items())
        )

    for result in results:
        _write_json(
            run_dir / f"{result['role']}.json",
            {
                "format_version": REROLL_FORMAT_VERSION,
                "role": result["role"],
                "sha256_minus_clocks": by_role[str(result["role"])],
                "clock_fields_excluded": list(REROLL_CLOCK_FIELDS),
                FENCED_KEY: result["payload"],
            },
        )
    record = {
        "format_version": REROLL_FORMAT_VERSION,
        "what_this_is": (
            "BRIEF_39 Amendment B.5-1: one DT cell on the fenced smoke draw rolled once at W = 1 and "
            f"{width} times in one {width}-worker pool, every pooled roll compared with the W = 1 "
            "roll under == with the clock excluded by name. Measurement of the instrument; no "
            "outcome value is recorded outside the fenced per-roll files"
        ),
        "line": line,
        "verdict": comparison["verdict"],
        "differing_fields": comparison["differing_fields"],
        "sha256_minus_clocks": by_role,
        "clock_fields_excluded": list(REROLL_CLOCK_FIELDS),
        "compared_fields_required": list(REROLL_COMPARED_FIELDS),
        "n_fields_compared": comparison["n_fields_compared"],
        "cell": cell,
        "workers": width,
        "canary_seconds": canary_seconds,
        **_git_provenance(),
    }
    _write_json(run_dir / "verdict.json", record)
    return {
        "verdict": comparison["verdict"],
        "line": line,
        "differing_fields": comparison["differing_fields"],
        "sha256_minus_clocks": by_role,
        "run_dir": str(run_dir),
        "n_rolls": len(results),
    }


def build_parser() -> Any:
    """CLI: ``cells``, ``pilot``, ``report``, ``canary``, ``record-canary``, and P7.3d's
    ``check-inputs``, ``resume-check``, ``dt-reroll-check`` and ``manifest`` (B.6 fix round).

    ⚠️ The roots are options of the PARENT parser, so they come BEFORE the subcommand:
    ``python -m offline.transfer_curve --draws-root D ... cells --stage S``.  P7.3d's driver put
    them after it and refused at its own canary on every machine (the pre-flight's B1).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m offline.transfer_curve",
        description=(
            "P7.3a: the zero-shot point of the C3 transfer curve. Every root is a parameter with "
            "today's path as its default (BRIEF_37 section 0.9)."
        ),
    )
    parser.add_argument("--draws-root", default=str(DEFAULT_DRAWS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--canary-seconds", type=float, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cells = subparsers.add_parser("cells", help="roll every cell of a stage that is not on disk")
    cells.add_argument("--stage", choices=[*STAGES, STAGE_GRID4X4], default=None)
    cells.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    cells.add_argument("--limit", type=int, default=None)

    pilot = subparsers.add_parser(
        "pilot", help="F1's pre-flight pilot: 16 fenced DT cells on draw 5 through the real pool"
    )
    pilot.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    pilot.add_argument(
        "--anchor",
        action="store_true",
        help="P7.3b: pilot the ANCHOR's cells instead of F1's sixteen (anchor_pilot_cells)",
    )
    pilot.add_argument(
        "--transcript",
        default=str(DEFAULT_OUTPUT_ROOT / "p7_3a_runs" / "preflight_pilot.txt"),
    )

    gate = subparsers.add_parser(
        "a17f", help="A17(f): every logged episode reproduces P7.2b's probe return bit-for-bit"
    )
    gate.add_argument("--corpus-dir", required=True)

    report_parser = subparsers.add_parser("report", help="write the committed artifact")
    report_parser.add_argument("--stage", choices=[*STAGES, STAGE_GRID4X4], default=None)
    report_parser.add_argument("--stage1-path", default=None)

    subparsers.add_parser("canary", help="the machine-health canary (PROJECT_PLAN section 7)")
    record = subparsers.add_parser("record-canary", help="park the canary line in the work directory")
    record.add_argument("--line", required=True)

    subparsers.add_parser(
        "check-inputs", help="P7.3d m1: the campaign's inputs by DIGEST against the module's pins"
    )
    resume = subparsers.add_parser(
        "resume-check", help="P7.3d m2: refuse if a chunk on disk records a commit git cannot resolve"
    )
    resume.add_argument("--stage", choices=[*STAGES, STAGE_GRID4X4], required=True)
    reroll = subparsers.add_parser(
        "dt-reroll-check",
        help="B.5-1: one fenced DT cell rolled at W=1 and in one pool, compared under ==",
    )
    reroll.add_argument("--g2-dir", required=True)
    reroll.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    manifest = subparsers.add_parser(
        "manifest", help="P7.3d: output/SHA256SUMS_p7_3d.txt over output/p7_3d/ only, re-verified"
    )
    manifest.add_argument("--campaign-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a non-zero code on a refusal rather than raising into the driver."""
    from offline.transfer_calibration import (
        canary_seconds as measure_canary,
        check_canary,
        format_canary_line,
        record_canary,
    )

    args = build_parser().parse_args(argv)
    work = Path(args.work_dir)

    if args.command == "canary":
        seconds, facts = measure_canary()
        # PRINT FIRST, THEN CHECK (reviewer MIN-6, matching P7.2b): on a failed correctness half the
        # OBSERVED values must be visible in the pane and in canary.log, or the refusal says only
        # that something was wrong. The driver captures this line with `tee -a`, so printed is also
        # recorded -- and `set -euo pipefail` is what makes the failure still stop the run.
        print(format_canary_line(seconds, facts), flush=True)
        check_canary(facts)
        return 0

    if args.command == "record-canary":
        print(record_canary(args.line, work_dir=work), flush=True)
        return 0

    if args.command == "a17f":
        # The gate lives in transfer_calibration, where P7.2b's probe does; this is the CLI the
        # P7.3a driver runs it from. It RAISES on a mismatch -- 100/100 or P7.3 stops -- so the
        # driver's `|| fail` sees a non-zero exit and the campaign never reaches an evaluation cell.
        #
        # ⚠️ THE BAND IS PASSED EXPLICITLY, AND THAT IS THE WHOLE OF AMENDMENT H1.
        # `assert_logged_corpus_matches_probe` defaults `draw_ids=None` to *the draws present in
        # the corpus*, so a gate called without a band checks the corpus against itself: the
        # coordinator ran this CLI on the ONE-draw smoke corpus and got `n_checked: 1,
        # all_match: true, exit 0`. A 99-draw corpus would have passed, and P7.3b's few-shot
        # source would have been short a draw with nothing saying so. The expectation is derived
        # from the REGISTERED band, never from the data (PROJECT_PLAN section 7).
        from offline.transfer_calibration import (
            PROBE_DRAW_END_DEFAULT,
            PROBE_DRAW_START_DEFAULT,
            assert_logged_corpus_matches_probe,
        )

        artifact_path = Path(args.data_dir) / P7_2B_CALIBRATION_NAME
        # Reviewer MIN-7: the gate reads the same artifact the prompts come from, so it is pinned
        # here too -- a gate that checked a corpus against a file that had moved proves nothing.
        load_calibration(artifact_path)

        band = range(PROBE_DRAW_START_DEFAULT, PROBE_DRAW_END_DEFAULT)
        expected = len(band)
        record = assert_logged_corpus_matches_probe(
            args.corpus_dir, artifact_path, draw_ids=band
        )
        if record["n_checked"] != expected or record["n_matching"] != expected:
            raise ValueError(
                f"A17(f) checked {record['n_checked']} draw(s) and matched {record['n_matching']} "
                f"of the declared {expected} ({PROBE_DRAW_START_DEFAULT}-{PROBE_DRAW_END_DEFAULT - 1}); "
                "100/100 or P7.3 stops"
            )
        # The one line the coordinator reads at the stage-1 checkpoint, before stage 2's token.
        print(f"A17(f) {record['n_matching']}/{expected}", flush=True)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        return 0

    if args.command == "pilot":
        record = run_pilot(
            work_dir=work,
            out_root=args.draws_root,
            output_root=args.output_root,
            data_dir=args.data_dir,
            workers=args.workers,
            transcript_path=args.transcript,
            cells=anchor_pilot_cells() if getattr(args, "anchor", False) else None,
        )
        # Counts and clocks only. `report` is never called on a pilot directory, and would refuse.
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        return 1 if record["n_failed"] else 0

    if args.command == "cells":
        summary = run_stage(
            work_dir=work,
            out_root=args.draws_root,
            output_root=args.output_root,
            data_dir=args.data_dir,
            stage=args.stage,
            canary_seconds=args.canary_seconds,
            workers=args.workers,
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 1 if summary["n_failed"] else 0

    if args.command == "check-inputs":
        for line in check_campaign_inputs(
            data_dir=args.data_dir, output_root=args.output_root, draws_root=args.draws_root
        ):
            print(line, flush=True)
        return 0

    if args.command == "resume-check":
        problems = unresolvable_chunk_commits(work_dir=work, stage=args.stage)
        if problems:
            import sys

            for problem in problems:
                print(
                    f"resume-check: {problem['chunk']} records git_commit {problem['git_commit']}, "
                    f"which git cannot resolve: {problem['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            return 2
        print("resume-check: every chunk on disk that reaches git records a resolvable commit", flush=True)
        return 0

    if args.command == "dt-reroll-check":
        result = run_dt_reroll_check(
            g2_dir=args.g2_dir,
            out_root=args.draws_root,
            output_root=args.output_root,
            data_dir=args.data_dir,
            canary_seconds=args.canary_seconds,
            workers=args.workers,
        )
        # The ONE line the driver copies into the capture's header: a verdict, field names and
        # digests -- never a value (B.5-1, B.2-2's fence).
        print(result["line"], flush=True)
        return 0 if result["verdict"] == "IDENTICAL" else 2

    if args.command == "manifest":
        record = write_manifest(campaign_dir=args.campaign_dir)
        print(f"manifest {record['path']}: {record['n_files']} files, re-verified", flush=True)
        return 0

    # One stage, one artifact name. The anchor's is its own file: it is a different declaration
    # over a different work directory, and writing it under P7.3a's name would overwrite the
    # zero-shot point with the curve's endpoint.
    name = artifact_name_for_stage(args.stage)
    artifact = report(
        work_dir=work,
        out_path=Path(args.out_dir) / name,
        output_root=args.output_root,
        out_root=args.draws_root,
        data_dir=args.data_dir,
        stage=args.stage,
        stage1_path=args.stage1_path,
    )
    print(f"wrote {Path(args.out_dir) / name} with {len(artifact['cells'])} cells", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
