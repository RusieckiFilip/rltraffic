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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ANCHOR_ARMS",
    "ARTIFACT_FORMAT_VERSION",
    "DECLARED_ARMS",
    "HALTING_CHECK_DRAW",
    "HELD_OUT_DRAWS",
    "RANDOM_POLICY_SEEDS",
    "STAGE_CONFIRMATORY",
    "STAGES",
    "SUBJECTS",
    "TRAINING_SEEDS",
    "ArmSpec",
    "cell_chunk_name",
    "declared_cells",
    "rho",
    "targets_for_subject",
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
STAGES: tuple[str, ...] = (STAGE_CONFIRMATORY, "rest")


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
    if stage is not None and stage not in STAGES:
        raise ValueError(f"{stage!r} is not one of {list(STAGES)}")

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


def cell_chunk_name(cell: Mapping[str, Any]) -> str:
    """A chunk's file name; one cell, one file, and the name carries the whole identity."""
    subject = cell["subject"] or "anchor"
    seed = "none" if cell["seed"] is None else int(cell["seed"])
    return f"cell_{subject}_{cell['arm']}_seed{seed}_draw{int(cell['draw_id']):04d}.json"


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
