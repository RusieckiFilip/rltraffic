"""P9.4 plan sections 5 and 6 (``docs/plans/p9.4_mix50_fresh.md``): what is reported, and the sentence each outcome
selects.  EXPLORATORY, ADDED AFTER THE RESULTS; no multiplicity correction (plan section 5).

FORMATS (every artifact carries its version)
--------------------------------------------
* ``p9.4-zero-shot/1.0`` -- ``docs/data/p9_4_zero_shot.json``: P7.3a's block layout (``p7.3a-zero-shot/1.0``'s ``cells``,
  ``rho``, ``h3``, ``contrast`` and ``in_support``), computed by P7.3a's own functions on ``mix50_fresh``'s 1,000 cells
  against the REGISTERED anchor pair; the same blocks under the ALIGNED pair (P9.2's functions); the guards; R1-R7;
  the readings; and the sentence S0-S13 each template selects, its slots filled.
* ``p9.4-cityflow/1.0`` -- ``docs/data/p9_4_cityflow.json``: the 500 CityFlow records in P9.1's hz1x1 format
  (``corrective_report._joined_row``: route (i)'s ``att_ours`` joined to route (ii)'s ``att_engine`` and A11(b)'s
  companions, refused unless the two routes agree under ``==``).
* ``p9.4-training/1.0`` -- ``docs/data/p9_4_training.json``: every run's record, the control record and each
  session's code commit.

ALIGNMENT CONVENTION AND THE UNIT
---------------------------------
Every value is one episode's horizon value from ``reset(seed=1000)`` on one held-out draw (A18(c)).  The unit is the
PAIRED DRAW: each draw's value is the mean over the five seeds (``transfer_curve._seed_means_by_draw``,
``dt_gate._per_draw_means``), and a comparison pairs two arms on the same draws.  CI = mean +/- 1.96 * sd / sqrt(100)
(``dt_gate.mean_ci95``, ddof 1); the test is ``dt_gate.wilcoxon_signed_rank``, two-sided.  A difference is
``left - right`` on ATT, so NEGATIVE means the left arm had the lower travel time; a WIN is a draw where it did.  rho is
``(ATT_fixedtime - ATT_arm) / (ATT_fixedtime - ATT_maxpressure)`` per row against the same draw's anchors of the pair
(``transfer_curve._rho_pair``), never clipped.  In SUMO ``e_sumo`` is primary and ``att_env`` beside it (A15); in
CityFlow ``att_engine`` is primary and ``att_ours`` beside it (A11(c), Rule R).

THE NUMBERS IN THE SENTENCES
----------------------------
A branch is chosen from the exact values.  The printed values share one number of decimals -- one for seconds, two for
rho by default -- raised until every printed value keeps its exact sign relative to 0 (and, for rho, to 1), so a
printed interval never contradicts its branch.  A p-value prints with three decimals, or two significant digits below
0.001.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import offline.p9_4_mix50_fresh as p94

ZERO_SHOT_FORMAT_VERSION = "p9.4-zero-shot/1.0"
CITYFLOW_FORMAT_VERSION = "p9.4-cityflow/1.0"
TRAINING_FORMAT_VERSION = "p9.4-training/1.0"

#: The (subject, arm, role) groups the rho blocks are written for, in P7.3a's order (calibrated first).
NEW_GROUPS: tuple[tuple[str, str, str], ...] = (
    (p94.SUBJECT, "b_mean_k100", "registered_prompt"), (p94.SUBJECT, "naive", "ablation"),
)
ORIGINAL_GROUPS: tuple[tuple[str, str, str], ...] = (
    (p94.REGISTERED_TIER, "b_mean_k100", "registered_prompt"), (p94.REGISTERED_TIER, "naive", "ablation"),
)

#: The two anchor pairs (plan 4.1), role -> the arm of ``p9_2_sumo_anchors.json``'s ``by_draw``.
PAIRS: Mapping[str, Mapping[str, str]] = {
    "registered": {"fixedtime": "fixedtime", "maxpressure": "maxpressure"},
    "aligned": {"fixedtime": "fixedtime_kcf", "maxpressure": "maxpressure_1to1"},
}

#: The instruction's sign convention for R4 (A17(d) states its direction on rho).
R4_DIRECTION = ("A17(d)'s registered direction is calibrated rho >= source rho; on ATT the same direction is "
                "calibrated - source <= 0 (a difference is calibrated - source; negative means the calibrated prompt "
                "had the lower travel time)")


# ----------------------------------------------------------------------
# Anchors, pairs and the rho blocks: P7.3a's and P9.2's functions
# ----------------------------------------------------------------------


def anchors_for_pair(by_draw: Mapping[str, Mapping[str, Mapping[str, Any]]], pair: str) -> dict[int, dict[str, Any]]:
    """``{draw: {"fixedtime": row, "maxpressure": row}}`` for one of plan 4.1's two pairs, from
    ``p9_2_sumo_anchors.json``'s ``anchors.sumo.cityflow1x1.by_draw``: each row the committed values with its concrete
    ``arm`` and ``draw_id`` added -- the shape ``transfer_curve._rho_pair`` and ``sumo_anchor_rescore.hz1x1_rho_block``
    read."""
    if pair not in PAIRS:
        raise ValueError(f"{pair!r} is not one of plan 4.1's anchor pairs {sorted(PAIRS)}")
    return {
        int(draw): {role: {**dict(slot[arm]), "arm": arm, "draw_id": int(draw), "kind": "anchor", "seed": None,
                           "subject": None}
                    for role, arm in PAIRS[pair].items()}
        for draw, slot in by_draw.items()
    }


def pair_block(rows: Sequence[Mapping[str, Any]], anchors: Mapping[int, Mapping[str, Any]],
               groups: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    """The rows re-scored against *anchors* by ``transfer_curve._rho_pair`` (``sumo_anchor_rescore.rescored_rows``), and
    ``by_subject_arm`` / ``by_subject_arm_seed`` exactly as ``transfer_curve.report`` writes them for hz1x1
    (``sumo_anchor_rescore.hz1x1_rho_block``: per-draw seed means, ``mean_ci95``, the paired ATT blocks), plus
    ``transfer_curve._h3_block`` on them."""
    import offline.sumo_anchor_rescore as sar
    import offline.transfer_curve as tc

    rescored = sar.rescored_rows(rows, anchors)
    block = sar.hz1x1_rho_block(rescored, anchors, groups=groups)
    return {"rows": rescored, **block, "h3": tc._h3_block(block["by_subject_arm"])}


def paired(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], key: str, *, left_label: str,
           right_label: str) -> dict[str, Any]:
    """``offline_baselines.paired_comparison`` of two sets of rows on *key*, in ``transfer_curve._paired_block``'s
    shape: per-draw seed means, the mean of ``left - right`` with its ``mean_ci95`` interval, wins (negative draws),
    losses, ties and the Wilcoxon p.  Both sides must cover the SAME draws."""
    from offline.dt_gate import EpisodeResult
    from offline.offline_baselines import paired_comparison

    left_draws = sorted({int(row["draw_id"]) for row in left})
    right_draws = sorted({int(row["draw_id"]) for row in right})
    if left_draws != right_draws or not left_draws:
        raise ValueError(f"{left_label} and {right_label} cover different draw sets ({len(left_draws)} and "
                         f"{len(right_draws)} draws); the unit is the paired draw")

    def episodes(rows: Sequence[Mapping[str, Any]], label: str) -> list[Any]:
        return [EpisodeResult(arm=label, seed=None if row.get("seed") is None else int(row["seed"]),
                              draw_id=int(row["draw_id"]), att_horizon=float(row[key]),
                              horizon_vehicle_count=float(row.get("horizon_vehicle_count", 0.0)),
                              episode_reward=float(row.get("episode_reward", 0.0)))
                for row in rows]

    comparison = paired_comparison(episodes(left, left_label), episodes(right, right_label))
    return {
        "att_definition": key, "left_arm": comparison.left_arm, "right_arm": comparison.right_arm,
        "n_shared_draws": comparison.n_shared_draws, "mean_left": comparison.mean_left,
        "mean_right": comparison.mean_right, "mean_difference": comparison.mean_difference,
        "ci95_low": comparison.ci95_low, "ci95_high": comparison.ci95_high,
        "median_difference": comparison.median_difference, "wins": comparison.wins, "losses": comparison.losses,
        "ties": comparison.ties, "rank_biserial": comparison.rank_biserial, "wilcoxon_p": comparison.wilcoxon.p_value,
        "direction": f"mean({left_label} - {right_label}) on {key}; NEGATIVE means {left_label} had the lower travel time",
    }


def calibrated_minus_source(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """R4's paired difference, by P7.3a's own ``_paired_block``: the calibrated arm against the source arm (``naive``)
    of the same subject, per draw."""
    import offline.transfer_curve as tc

    calibrated = [row for row in rows if row["subject"] == p94.SUBJECT and row["arm"] == "b_mean_k100"]
    source = [row for row in rows if row["subject"] == p94.SUBJECT and row["arm"] == "naive"]
    if not calibrated or {int(r["draw_id"]) for r in calibrated} != {int(r["draw_id"]) for r in source}:
        raise ValueError("R4 pairs the calibrated and the source arm on the same draws")
    return tc._paired_block([*rows], calibrated, p94.SUBJECT, "b_mean_k100", "naive", key)


def contrast(block: Mapping[str, Any]) -> dict[str, Any]:
    """R4's difference of mean rho: P7.3a's contrast block (``transfer_curve._contrast_block``) for ``mix50_fresh``."""
    import offline.transfer_curve as tc

    return {**tc._contrast_block(block["by_subject_arm"], subjects=(p94.SUBJECT,)), "sign_convention": R4_DIRECTION}


def entry_of(block: Mapping[str, Any], subject: str, arm: str) -> dict[str, Any]:
    """One (subject, arm) entry of a pair block's ``by_subject_arm``."""
    found = [entry for entry in block["by_subject_arm"] if entry["subject"] == subject and entry["arm"] == arm]
    if len(found) != 1:
        raise ValueError(f"{len(found)} entries for {subject}/{arm}")
    return found[0]


# ----------------------------------------------------------------------
# Readings
# ----------------------------------------------------------------------


def difference_reading(low: float, high: float) -> str:
    """``below`` if the whole interval lies below 0, ``above`` if above, ``contains`` otherwise (an endpoint ON 0
    contains it)."""
    if float(high) < 0.0:
        return "below"
    if float(low) > 0.0:
        return "above"
    return "contains"


def rho_reading(low: float, high: float) -> str:
    """S5's six branches, exhaustive and exclusive for ``low <= high`` (an endpoint ON 0 or 1 contains it)."""
    low, high = float(low), float(high)
    if high < 0.0:
        return "below_0"
    if low > 1.0:
        return "above_1"
    if low > 0.0 and high < 1.0:
        return "inside_0_1"
    if low > 0.0:
        return "contains_1_above_0"
    if high < 1.0:
        return "contains_0_below_1"
    return "contains_0_and_1"


# ----------------------------------------------------------------------
# Formatting: the printed values keep their sides
# ----------------------------------------------------------------------

_DEFAULT_DECIMALS = {"s": 1, "rho": 2}
_REFERENCES = {"s": (0.0,), "rho": (0.0, 1.0)}


def _side(value: float, reference: float) -> int:
    return (value > reference) - (value < reference)


def _decimals(values: Sequence[float], unit: str) -> int:
    for decimals in range(_DEFAULT_DECIMALS[unit], 7):
        if all(_side(round(float(v), decimals), ref) == _side(float(v), ref)
               for v in values for ref in _REFERENCES[unit]):
            return decimals
    return 6


def _fmt(value: float, decimals: int) -> str:
    text = f"{float(value):.{decimals}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def fmt_number(value: float, *, unit: str) -> str:
    return _fmt(value, _decimals([value], unit))


def fmt_interval(low: float, high: float, *, unit: str) -> tuple[str, str]:
    decimals = _decimals([low, high], unit)
    return _fmt(low, decimals), _fmt(high, decimals)


def fmt_estimate(mean: float, low: float, high: float, *, unit: str) -> tuple[str, str, str]:
    decimals = _decimals([mean, low, high], unit)
    return _fmt(mean, decimals), _fmt(low, decimals), _fmt(high, decimals)


def fmt_p(p: float) -> str:
    return f"{float(p):.2g}" if float(p) < 0.001 else f"{float(p):.3f}"


# ----------------------------------------------------------------------
# The sentences (plan section 6), one per outcome
# ----------------------------------------------------------------------

MINUS = "−"
RHO = "ρ"


def _sentence(template: str, branch: str, text: str) -> dict[str, str]:
    return {"template": template, "branch": branch, "text": text}


def sentence_s0() -> dict[str, str]:
    return _sentence("S0", "always", (
        "The random half of the 50 % mixture came from the original random collection, whose episodes replay two "
        "action schedules. After the transfer results were known, we retrained this DT with the random half drawn from "
        "the corrected collection on the same draws. The state normalisation and the return scale were refitted on the "
        "new data by the same rules, and the recipe, the prompts and the evaluation stayed the same. We report the "
        "retrained model as exploratory and the original one in the supplement."))


def _vs_fixed_time(template: str, block: Mapping[str, Any], name: str) -> dict[str, str]:
    d, lo, hi = fmt_estimate(block["mean_difference"], block["ci95_low"], block["ci95_high"], unit="s")
    p = fmt_p(block["wilcoxon_p"])
    tail = f"DT {MINUS} fixed-time {d} s [{lo}, {hi}] (paired Wilcoxon signed-rank, p = {p})."
    branch = difference_reading(block["ci95_low"], block["ci95_high"])
    head = "Transferred zero-shot, the DT from the corrected mixture"
    if branch == "below":
        return _sentence(template, branch, f"{head} was faster than {name} on {block['wins']} of 100 draws: {tail}")
    if branch == "above":
        return _sentence(template, branch, f"{head} was slower than {name} on {block['losses']} of 100 draws: {tail}")
    return _sentence(template, branch, f"{head} did not differ measurably from {name}: {tail}")


def sentence_s1(block: Mapping[str, Any]) -> dict[str, str]:
    return _vs_fixed_time("S1", block, "SUMO's fixed-time plan")


def sentence_s2(block: Mapping[str, Any]) -> dict[str, str]:
    return _vs_fixed_time("S2", block, "fixed-time at CityFlow's tuned cycle")


def _vs_maxpressure(template: str, block: Mapping[str, Any], name: str) -> dict[str, str]:
    d, lo, hi = fmt_estimate(block["mean_difference"], block["ci95_low"], block["ci95_high"], unit="s")
    value = f"DT {MINUS} MaxPressure {d} s [{lo}, {hi}]"
    branch = difference_reading(block["ci95_low"], block["ci95_high"])
    if branch == "below":
        return _sentence(template, branch, f"It was also faster than {name}: {value}, on {block['wins']} of 100 draws.")
    if branch == "above":
        return _sentence(template, branch, f"It was slower than {name}: {value}, on {block['losses']} of 100 draws.")
    return _sentence(template, branch, f"It did not differ measurably from {name}: {value}.")


def sentence_s3(block: Mapping[str, Any]) -> dict[str, str]:
    return _vs_maxpressure("S3", block, "the registered MaxPressure")


def sentence_s4(block: Mapping[str, Any]) -> dict[str, str]:
    return _vs_maxpressure("S4", block, "the MaxPressure that counts each lane once")


RHO_ENDINGS: Mapping[str, str] = {
    "above_1": "beyond MaxPressure's 1",
    "contains_1_above_0": "not distinguishable from MaxPressure's 1",
    "inside_0_1": "between fixed-time (0) and MaxPressure (1)",
    "contains_0_below_1": "not distinguishable from fixed-time's 0",
    "contains_0_and_1": "not distinguishable from either anchor",
    "below_0": "below fixed-time's 0",
}


def _rho_sentence(template: str, rho: Mapping[str, Any], anchors: str) -> dict[str, str]:
    r, lo, hi = fmt_estimate(rho["mean"], rho["ci95_low"], rho["ci95_high"], unit="rho")
    branch = rho_reading(rho["ci95_low"], rho["ci95_high"])
    return _sentence(template, branch, f"Its score was {RHO} = {r} [{lo}, {hi}] {anchors}, {RHO_ENDINGS[branch]}.")


def sentence_s5(rho: Mapping[str, Any]) -> dict[str, str]:
    return _rho_sentence("S5", rho, "under the registered anchors")


def sentence_s6(rho: Mapping[str, Any]) -> dict[str, str]:
    return _rho_sentence("S6", rho, "under the aligned anchors, which count each lane once and use CityFlow's tuned cycle")


def sentence_s7(rho: Mapping[str, Any], *, anchors: str) -> dict[str, str]:
    """S7 (registered anchors) and S7b (aligned): "better than fixed-time" iff the rho CI lies above 0, "worse than
    MaxPressure" iff it lies below 1."""
    if anchors not in ("registered", "aligned"):
        raise ValueError(f"anchors {anchors!r} is neither 'registered' nor 'aligned'")
    better = float(rho["ci95_low"]) > 0.0
    worse = float(rho["ci95_high"]) < 1.0
    text = (f"Read against the registered transfer hypothesis, the corrected model {'is' if better else 'is not'} better "
            f"than fixed-time and {'is' if worse else 'is not'} worse than MaxPressure under the {anchors} anchors.")
    return _sentence("S7" if anchors == "registered" else "S7b", f"better={better},worse={worse}", text)


def _three(template: str, block: Mapping[str, Any], below: str, contains: str, above: str, name: str,
           *, counts: bool = False) -> dict[str, str]:
    d, lo, hi = fmt_estimate(block["mean_difference"], block["ci95_low"], block["ci95_high"], unit="s")
    value = f"{name} {d} s [{lo}, {hi}]"
    branch = difference_reading(block["ci95_low"], block["ci95_high"])
    if branch == "below":
        return _sentence(template, branch, f"{below}: {value}" + (f", on {block['wins']} of 100 draws." if counts else "."))
    if branch == "above":
        return _sentence(template, branch, f"{above}: {value}" + (f", on {block['losses']} of 100 draws." if counts else "."))
    return _sentence(template, branch, f"{contains}: {value}.")


def sentence_s8(block: Mapping[str, Any]) -> dict[str, str]:
    return _three("S8", block, "The calibrated prompt made the model faster than its source prompt",
                  "The calibrated and the source prompt did not differ measurably",
                  "The source prompt made the model faster than the calibrated one", f"calibrated {MINUS} source")


def sentence_s9(block: Mapping[str, Any]) -> dict[str, str]:
    return _three("S9", block, "Correcting the random half made the transferred DT faster",
                  "Correcting the random half did not change the transferred DT measurably",
                  "Correcting the random half made the transferred DT slower", f"corrected {MINUS} original")


READING_NAMES: Mapping[str, str] = {
    "S1": "the comparison with SUMO's fixed-time plan",
    "S3": "the comparison with the registered MaxPressure",
    "S5": f"{RHO} under the registered anchors",
    "S6": f"{RHO} under the aligned anchors",
}
DIFFERENCE_WORDS: Mapping[str, Mapping[str, str]] = {
    "S1": {"below": "faster than SUMO's fixed-time plan",
           "contains": "not measurably different from SUMO's fixed-time plan",
           "above": "slower than SUMO's fixed-time plan"},
    "S3": {"below": "faster than the registered MaxPressure",
           "contains": "not measurably different from the registered MaxPressure",
           "above": "slower than the registered MaxPressure"},
}


def _branch_words(reading: str, branch: str) -> str:
    return RHO_ENDINGS[branch] if reading in ("S5", "S6") else DIFFERENCE_WORDS[reading][branch]


def sentence_s10(corrected: Mapping[str, str], original: Mapping[str, str]) -> dict[str, str]:
    """S10: the readings of S1, S3, S5 and S6 for the corrected and the original model (calibrated arm)."""
    if set(corrected) != set(READING_NAMES) or set(original) != set(READING_NAMES):
        raise ValueError(f"S10 compares exactly the readings {sorted(READING_NAMES)}")
    differing = [name for name in ("S1", "S3", "S5", "S6") if corrected[name] != original[name]]
    if not differing:
        return _sentence("S10", "all the same",
                         "Every reading of the transfer test is the same for the corrected and the original model.")
    names = [READING_NAMES[name] for name in differing]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    changes = "; ".join(f"{_branch_words(name, original[name])} becomes {_branch_words(name, corrected[name])}"
                        for name in differing)
    return _sentence("S10", "otherwise", f"The correction changes {listed}: {changes}.")


def sentence_s11(att: Mapping[str, Any], *, never_entered: float) -> dict[str, str]:
    m, lo, hi = fmt_estimate(att["mean"], att["ci95_low"], att["ci95_high"], unit="s")
    return _sentence("S11", "always", (
        f"In CityFlow, on the same 100 held-out draws, the DT from the corrected mixture had an average travel time of "
        f"{m} s [{lo}, {hi}], with {float(never_entered):.1f} vehicles per episode that never entered."))


def sentence_s12(block: Mapping[str, Any]) -> dict[str, str]:
    return _three("S12", block, "In CityFlow the corrected model was faster than the original one",
                  "In CityFlow the corrected and the original model did not differ measurably",
                  "In CityFlow the corrected model was slower than the original one", f"corrected {MINUS} original",
                  counts=True)


def sentence_s12b(engine: Mapping[str, Any], ours: Mapping[str, Any]) -> dict[str, str] | None:
    """S12b, only if ``att_ours`` selects a different branch of S12 than ``att_engine``."""
    if difference_reading(ours["ci95_low"], ours["ci95_high"]) == difference_reading(engine["ci95_low"],
                                                                                     engine["ci95_high"]):
        return None
    d, lo, hi = fmt_estimate(ours["mean_difference"], ours["ci95_low"], ours["ci95_high"], unit="s")
    return _sentence("S12b", difference_reading(ours["ci95_low"], ours["ci95_high"]), (
        f"Timed from entry instead of from creation, the reading changes: corrected {MINUS} original {d} s [{lo}, {hi}]."))


def sentence_s13(block: Mapping[str, Any]) -> dict[str, str]:
    head = "Transferred zero-shot, the DT from the corrected mixture"
    recipe = "the same recipe trained offline in SUMO on 200 MaxPressure episodes"
    return _three("S13", block, f"{head} was faster than {recipe}", f"{head} did not differ measurably from {recipe}",
                  f"{head} was slower than {recipe}", f"transferred {MINUS} SUMO-trained", counts=True)


# ----------------------------------------------------------------------
# The sources, loaded and checked once (every refusal before any number is computed)
# ----------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402 - the report's second half
from pathlib import Path  # noqa: E402


@dataclass(frozen=True)
class Sources:
    """Everything the three artifacts are built from.  :func:`load_sources` fills it from disk after every check;
    a test builds one by hand."""

    new_rows: list[dict[str, Any]]
    by_draw: dict[str, Any]
    original_rows: list[dict[str, Any]]
    anchor_k200_rows: list[dict[str, Any]]
    cityflow_rows: list[dict[str, Any]]
    cityflow_original: list[dict[str, Any]]
    targets_table: list[dict[str, Any]]
    p7_3a_rho: dict[str, Any]
    guards: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)
    training_runs: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    code_commits: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    declaration: dict[str, Any] = field(default_factory=dict)


def _published(payload: Mapping[str, Any], anchors: Mapping[str, Any]) -> dict[str, Any]:
    import offline.transfer_curve as tc

    return tc._published_row(payload, anchors, grid=False)


def _listing_digest(directory: Path, pattern: str) -> dict[str, Any]:
    """sha256 over the sorted ``name sha256`` lines of the files *pattern* matches: one digest for a directory."""
    import hashlib

    lines = [f"{path.name} {p94.file_sha256(path)}" for path in sorted(Path(directory).glob(pattern))]
    return {"n_files": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()}


def load_sources(roots: p94.Roots) -> Sources:
    """Read and check everything, before any number: the committed declaration at its pin; the control at 5 of 5; the
    guard record passed and written by this code; every one of the 1,000 cells' chunks present, valid for its cell and
    reusable (``p9_4_evaluation.chunk_problems``: the validator, this code, today's demand, the checkpoint digest, the
    declared prompt) and no chunk beside them; route (i) complete and route (ii) complete with no refusal, joined
    episode by episode under ``==``; and the committed artifacts read by name."""
    import json

    import offline.corrective_report as crep
    import offline.p9_4_evaluation as ev
    import offline.transfer_calibration as tcal
    import offline.transfer_curve as tc
    from offline import att_rederivation as ar

    declaration = p94.load_declaration(roots)
    control = p94.assert_control_passed(roots)
    guard_path = ev.guards_record_path(roots)
    if not guard_path.is_file():
        raise ValueError(f"{guard_path} is absent: the guards come before the cells")
    guards = json.loads(guard_path.read_text(encoding="utf-8"))
    ev.assert_guards_authorise(roots, guards)

    data = roots.data_dir
    anchors_doc = json.loads((data / p94.P9_2_ANCHORS).read_bytes())
    by_draw = anchors_doc["anchors"]["sumo"][ev.SCENARIO_KEY]["by_draw"]
    registered = anchors_for_pair(by_draw, "registered")

    declared = ev.new_cells()
    names = {ev.cell_path(roots, "new", cell).name for cell in declared}
    extra = sorted(path.name for path in ev.cells_dir(roots).glob("cell_*.json") if path.name not in names)
    if extra:
        raise ValueError(f"{len(extra)} chunk(s) in {ev.cells_dir(roots)} are not declared cells (first {extra[:3]})")
    new_rows = []
    for cell in declared:
        path = ev.cell_path(roots, "new", cell)
        if not path.is_file():
            raise ValueError(f"{path} is absent: the campaign is incomplete and a partial artifact is refused")
        payload = json.loads(path.read_text(encoding="utf-8"))
        problems = ev.chunk_problems(roots, "new", cell, payload, declaration)
        if problems:
            raise ValueError(f"{path.name}: {'; '.join(problems)}")
        new_rows.append(_published(payload, registered[int(cell["draw_id"])]))

    p7_3a = json.loads((data / p94.P7_3A_ARTIFACT).read_bytes())
    original_rows = [dict(row) for row in p7_3a["cells"] if row["kind"] == "dt" and row["subject"] == p94.REGISTERED_TIER
                     and row["arm"] in p94.SUMO_ARMS]
    p7_3b = json.loads((data / p94.P7_3B_ARTIFACT).read_bytes())
    anchor_k200_rows = [dict(row) for row in p7_3b["cells"] if row.get("subject") == "anchor_k200"]

    route_i_path = ev.route_i_output(roots)
    route_i = json.loads(route_i_path.read_text(encoding="utf-8"))
    ev.assert_route_i_complete(route_i)
    status = ar.campaign_status(ev.route_ii_work(roots))
    if not status["complete"] or status["refused"]:
        raise ValueError(f"route (ii) is not complete without refusals: {status}")
    work = ev.route_ii_work(roots)
    source = (f"new: route (i) output/{route_i_path.relative_to(Path(roots.output_root)).as_posix()}; route (ii) "
              f"output/{work.relative_to(Path(roots.output_root)).as_posix()}, att_ours equal under ==")
    cityflow_rows = []
    for episode in sorted(route_i["episodes"], key=lambda e: (int(e["seed"]), int(e["draw_id"]))):
        rerolled = crep._rerolled(work, ev.CITYFLOW_SCENARIO, ev.CITYFLOW_ARM, int(episode["seed"]),
                                  int(episode["draw_id"]), "route (ii) has no episode")
        cityflow_rows.append(crep._joined_row(ev.CITYFLOW_ARM, p94.TIER, episode, rerolled,
                                              att_ours=float(episode["att_horizon"]), source=source,
                                              refusal="the two routes disagree"))
    cityflow_original = [dict(row) for row in json.loads((data / p94.P9_1_IQL_TARGETS).read_bytes())["records"][
        f"p4_7:dt@{p94.REGISTERED_TIER}"]]

    facts = ev.new_subject_facts(roots, declaration)
    calibration = tc.load_calibration(data / tc.P7_2B_CALIBRATION_NAME)
    probe = [(int(record["draw_id"]), float(record["local_return"])) for record in calibration["probe"]]
    table = tcal.targets_table(calibration["statistics"], probe, {p94.SUBJECT: facts})[p94.SUBJECT]
    sessions = crep.driver_sessions(roots)  # type: ignore[arg-type]
    code_commits = {stage: crep.stage_code_commit(sessions, stage) for stage in ("train", "cityflow", "sumo")}
    training_runs = [json.loads(p94.run_record_path(roots, run).read_text(encoding="utf-8"))
                     for run in p94.training_runs() if p94.run_status(roots, run) == "done"]
    if len(training_runs) != len(p94.training_runs()):
        raise ValueError("every training run is done before the report")
    inputs = {
        "docs_data_sha256": {name: p94.file_sha256(data / name) for name in (
            p94.DECLARATION_NAME, p94.P7_3A_ARTIFACT, p94.P9_2_ANCHORS, p94.P7_3B_ARTIFACT, p94.P9_1_IQL_TARGETS,
            tc.P7_2B_CALIBRATION_NAME)},
        "route_i_sha256": p94.file_sha256(route_i_path),
        "route_ii": _listing_digest(work, "cell_*.json"),
        "sumo_cells": _listing_digest(ev.cells_dir(roots), "cell_*.json"),
        "guards_sha256": p94.file_sha256(guard_path),
        "control_sha256": p94.file_sha256(p94.control_record_path(roots)),
    }
    return Sources(new_rows=new_rows, by_draw=by_draw, original_rows=original_rows, anchor_k200_rows=anchor_k200_rows,
                   cityflow_rows=cityflow_rows, cityflow_original=cityflow_original, targets_table=table,
                   p7_3a_rho=p7_3a["rho"], guards=guards, control=control, training_runs=training_runs,
                   sessions=sessions, code_commits=code_commits, inputs=inputs, declaration=declaration)


# ----------------------------------------------------------------------
# The blocks R1-R7 and the readings, from a Sources (pure)
# ----------------------------------------------------------------------


def _rho_stats(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: dict(entry[key]) for key in ("e_sumo", "att_env")}


def _seed_means(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """Per seed: the mean over its draws, with ``mean_ci95``'s n."""
    from offline.dt_gate import mean_ci95

    out = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        stats = mean_ci95([float(row[key]) for row in rows if int(row["seed"]) == seed])
        out[str(seed)] = {"n_draws": stats.n, "mean": stats.mean}
    return out


def _draw_stats(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """The mean over draws of each draw's seed mean, with ``mean_ci95``'s interval (the unit of plan section 5)."""
    import offline.transfer_curve as tc
    from offline.dt_gate import mean_ci95

    per_draw = tc._seed_means_by_draw(rows, key)
    stats = mean_ci95([per_draw[draw] for draw in sorted(per_draw)])
    return {"n_draws": stats.n, "mean": stats.mean, "std": stats.std, "ci95": stats.ci95,
            "ci95_low": stats.mean - stats.ci95, "ci95_high": stats.mean + stats.ci95}


def _arm_rows(rows: Sequence[Mapping[str, Any]], subject: str, arm: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["subject"] == subject and row["arm"] == arm]


def readings_of(registered: Mapping[str, Any], aligned: Mapping[str, Any], subject: str, arm: str) -> dict[str, str]:
    """The four readings S10 compares, and R5's: S1, S3 (paired differences on ``e_sumo`` against the registered pair),
    S5 and S6 (rho's interval under each pair), plus S2 and S4 (against the aligned pair)."""
    reg, al = entry_of(registered, subject, arm), entry_of(aligned, subject, arm)

    def diff(entry: Mapping[str, Any], role: str) -> str:
        block = entry["paired_att"]["e_sumo"][role]
        return difference_reading(block["ci95_low"], block["ci95_high"])

    return {"S1": diff(reg, "fixedtime"), "S2": diff(al, "fixedtime"), "S3": diff(reg, "maxpressure"),
            "S4": diff(al, "maxpressure"), "S5": rho_reading(reg["e_sumo"]["ci95_low"], reg["e_sumo"]["ci95_high"]),
            "S6": rho_reading(al["e_sumo"]["ci95_low"], al["e_sumo"]["ci95_high"])}


def analysis(sources: Sources) -> dict[str, Any]:
    """R1-R7, the readings and the per-arm extras, from the sources by P7.3a's, P9.2's and ``dt_gate``'s functions."""
    import offline.transfer_curve as tc

    registered = pair_block(sources.new_rows, anchors_for_pair(sources.by_draw, "registered"), NEW_GROUPS)
    aligned = pair_block(sources.new_rows, anchors_for_pair(sources.by_draw, "aligned"), NEW_GROUPS)
    original_registered = pair_block(sources.original_rows, anchors_for_pair(sources.by_draw, "registered"),
                                     ORIGINAL_GROUPS)
    original_aligned = pair_block(sources.original_rows, anchors_for_pair(sources.by_draw, "aligned"), ORIGINAL_GROUPS)
    arms = {"calibrated": "b_mean_k100", "source": "naive"}
    r1, r2, r3, r5, r7, also = {}, {}, {}, {}, {}, {}
    for label, arm in arms.items():
        reg, al = entry_of(registered, p94.SUBJECT, arm), entry_of(aligned, p94.SUBJECT, arm)
        r1[label] = {"fixedtime_k4": {k: reg["paired_att"][k]["fixedtime"] for k in ("e_sumo", "att_env")},
                     "fixedtime_tuned_k": {k: al["paired_att"][k]["fixedtime"] for k in ("e_sumo", "att_env")}}
        r2[label] = {"maxpressure_per_link": {k: reg["paired_att"][k]["maxpressure"] for k in ("e_sumo", "att_env")},
                     "maxpressure_lane_once": {k: al["paired_att"][k]["maxpressure"] for k in ("e_sumo", "att_env")}}
        r3[label] = {
            pair: {**_rho_stats(entry),
                   "readings": {key: {"rho > 0": entry[key]["ci95_low"] > 0.0, "rho < 1": entry[key]["ci95_high"] < 1.0}
                                for key in ("e_sumo", "att_env")}}
            for pair, entry in (("registered", reg), ("aligned", al))
        }
        corrected_rows = _arm_rows(sources.new_rows, p94.SUBJECT, arm)
        original_rows = _arm_rows(sources.original_rows, p94.REGISTERED_TIER, arm)
        corrected_readings = readings_of(registered, aligned, p94.SUBJECT, arm)
        original_readings = readings_of(original_registered, original_aligned, p94.REGISTERED_TIER, arm)
        r5[label] = {
            **{key: paired(corrected_rows, original_rows, key, left_label="corrected", right_label="original")
               for key in ("e_sumo", "att_env")},
            "rho": {pair: {"corrected": _rho_stats(entry_of(blocks[0], p94.SUBJECT, arm)),
                           "original": _rho_stats(entry_of(blocks[1], p94.REGISTERED_TIER, arm))}
                    for pair, blocks in (("registered", (registered, original_registered)),
                                         ("aligned", (aligned, original_aligned)))},
            "readings": {"corrected": corrected_readings, "original": original_readings,
                         "all_agree": corrected_readings == original_readings},
        }
        r7[label] = {"e_sumo": paired(corrected_rows, sources.anchor_k200_rows, "e_sumo", left_label="transferred",
                                      right_label="sumo_trained_anchor_k200")}
        also[label] = {
            "per_seed_means": {key: _seed_means(corrected_rows, key) for key in ("e_sumo", "att_env")},
            "decisions_in_support": {"mean_per_cell": sum(int(r["n_decisions_in_support"]) for r in corrected_rows)
                                     / len(corrected_rows), "n_cells": len(corrected_rows)},
            "never_entered_per_episode": sum(int(r["n_never_entered"]) for r in corrected_rows) / len(corrected_rows),
            "teleports": sum(int(r["n_teleports"]) for r in corrected_rows),
            "vanished_without_arrival": sum(int(r["n_vanished_without_arrival"]) for r in corrected_rows),
            "collisions": "hz1x1 cells refuse any teleport (A15(c); validate_cell_payload), so a collision teleport "
                          "cannot reach this artifact; the counts above are the cells' own",
        }
    r4 = {"paired": {key: calibrated_minus_source(registered["rows"], key) for key in ("e_sumo", "att_env")},
          "contrast": {"registered": contrast(registered), "aligned": contrast(aligned)},
          "direction": R4_DIRECTION}
    cityflow_new = sources.cityflow_rows
    r6 = {
        **{key: {**_draw_stats(cityflow_new, key), "per_seed_means": _seed_means(cityflow_new, key)}
           for key in ("att_engine", "att_ours")},
        "companions": {key: sum(float(r[key]) for r in cityflow_new) / len(cityflow_new)
                       for key in ("entered", "created", "never_entered", "entered_fraction")},
        "corrected_minus_original": {key: paired(cityflow_new, sources.cityflow_original, key, left_label="corrected",
                                                 right_label="original") for key in ("att_engine", "att_ours")},
        "primary": "att_engine (A11(c), Rule R); att_ours beside it",
    }
    for block in (*(b for arm in r1.values() for v in arm.values() for b in v.values()),
                  *(b for arm in r2.values() for v in arm.values() for b in v.values()),
                  *r4["paired"].values(), *(r5[label][key] for label in r5 for key in ("e_sumo", "att_env")),
                  *(r7[label]["e_sumo"] for label in r7), *r6["corrected_minus_original"].values()):
        if block["n_shared_draws"] != len(p94.HELD_OUT_DRAWS):
            raise ValueError(f"a paired block covers {block['n_shared_draws']} draws, not the "
                             f"{len(p94.HELD_OUT_DRAWS)} held-out draws")
    return {"registered": registered, "aligned": aligned, "r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6,
            "r7": r7, "also": also, "in_support": tc._in_support_block(
                sources.new_rows, {"targets": {p94.SUBJECT: sources.targets_table}}, subjects=(p94.SUBJECT,))}


def choose_sentences(result: Mapping[str, Any]) -> list[dict[str, str]]:
    """Plan section 6: every template's branch, chosen by the numbers (calibrated arm, ``e_sumo``; CityFlow on
    ``att_engine``), its slots filled.  S12b appears only when ``att_ours`` selects another branch of S12."""
    r1, r2, r3, r4, r5, r6, r7 = (result[name] for name in ("r1", "r2", "r3", "r4", "r5", "r6", "r7"))
    sentences = [
        sentence_s0(),
        sentence_s1(r1["calibrated"]["fixedtime_k4"]["e_sumo"]),
        sentence_s2(r1["calibrated"]["fixedtime_tuned_k"]["e_sumo"]),
        sentence_s3(r2["calibrated"]["maxpressure_per_link"]["e_sumo"]),
        sentence_s4(r2["calibrated"]["maxpressure_lane_once"]["e_sumo"]),
        sentence_s5(r3["calibrated"]["registered"]["e_sumo"]),
        sentence_s6(r3["calibrated"]["aligned"]["e_sumo"]),
        sentence_s7(r3["calibrated"]["registered"]["e_sumo"], anchors="registered"),
        sentence_s7(r3["calibrated"]["aligned"]["e_sumo"], anchors="aligned"),
        sentence_s8(r4["paired"]["e_sumo"]),
        sentence_s9(r5["calibrated"]["e_sumo"]),
        sentence_s10({k: r5["calibrated"]["readings"]["corrected"][k] for k in READING_NAMES},
                     {k: r5["calibrated"]["readings"]["original"][k] for k in READING_NAMES}),
        sentence_s11(r6["att_engine"], never_entered=r6["companions"]["never_entered"]),
        sentence_s12(r6["corrected_minus_original"]["att_engine"]),
    ]
    s12b = sentence_s12b(r6["corrected_minus_original"]["att_engine"], r6["corrected_minus_original"]["att_ours"])
    if s12b is not None:
        sentences.append(s12b)
    sentences.append(sentence_s13(r7["calibrated"]["e_sumo"]))
    return sentences


def _rho_layout(sources: Sources, block: Mapping[str, Any], anchors: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """P7.3a's ``rho`` block for these rows: its committed texts (read from ``p7_3a_zero_shot.json``), the denominator
    diagnostics recomputed (``transfer_curve.denominator_diagnostic`` / ``anchor_denominator``) and the estimates."""
    import offline.transfer_curve as tc

    definitions = {key: dict(value) for key, value in sources.p7_3a_rho["definitions"].items()}
    for key in ("e_sumo", "att_env"):
        definitions[key]["denominator_diagnostic"] = tc.denominator_diagnostic(
            {draw: tc.anchor_denominator(slot, key) for draw, slot in sorted(anchors.items())})
    excluded = [draw for draw, slot in sorted(anchors.items()) if tc.anchor_denominator(slot, "att_env") == 0.0]
    if excluded:
        raise ValueError(f"draws {excluded} have a zero env-ATT denominator under this pair; the 2026-09-17 ruling's "
                         "exclusion is not implemented for P9.4, which stops here")
    definitions["att_env"].update({"excluded_draws": [], "n_draws_used": len(anchors), "n_draws_total": len(anchors)})
    return {"formula": sources.p7_3a_rho["formula"], "definitions": definitions,
            "estimator": dict(sources.p7_3a_rho["estimator"]), "not_clipped": sources.p7_3a_rho["not_clipped"],
            "by_subject_arm_seed": block["by_subject_arm_seed"], "by_subject_arm": block["by_subject_arm"],
            "per_draw": block["per_draw"]}


WHAT_THIS_DOES_NOT_SAY = (
    "P9.4 is EXPLORATORY and was added after the results (plan section 1): it is not a registered test, and A25(c) as "
    "A27 amends it allows no further run, so it is a deviation with its own row in the supplement's deviation table. Of "
    "P7.3a's four prompt arms two are repeated (b_mean_k100, naive); a_q1.0 and b_max_k100 are not. No multiplicity "
    "correction is applied. rho is computed WITHIN SUMO against anchors of the same draws. R7 compares models that "
    "differ in several things at once (the simulator and the controller that produced their logs, the prompt rule, the "
    "alignment of observations), so its difference is reported without a cause.")


def build_zero_shot(sources: Sources) -> dict[str, Any]:
    """``docs/data/p9_4_zero_shot.json``: P7.3a's blocks under the registered pair, the aligned pair beside them, the
    guards, R1-R7, the readings and the sentences."""
    import offline.transfer_curve as tc

    result = analysis(sources)
    registered, aligned = result["registered"], result["aligned"]
    return {
        "format_version": ZERO_SHOT_FORMAT_VERSION,
        "block_layout": "p7.3a-zero-shot/1.0's cells, rho, h3, contrast and in_support, computed by P7.3a's functions",
        "task": "P9.4", "label": p94.LABEL, "registered_in": None, "plan": {"path": p94.PLAN_PATH, "tags": dict(p94.PLAN_TAGS)},
        "subject": p94.SUBJECT, "scenario_key": "cityflow1x1", "n_cells_declared": len(sources.new_rows),
        "cells": [dict(row) for row in sources.new_rows],
        "rho": _rho_layout(sources, registered, anchors_for_pair(sources.by_draw, "registered")),
        "h3": registered["h3"],
        "contrast": contrast(registered),
        "in_support": result["in_support"],
        "aligned": {"anchors": dict(PAIRS["aligned"]),
                    "rho": _rho_layout(sources, aligned, anchors_for_pair(sources.by_draw, "aligned")),
                    "h3": aligned["h3"], "contrast": contrast(aligned)},
        "anchor_source": "docs/data/p9_2_sumo_anchors.json anchors.sumo.cityflow1x1.by_draw; registered pair (fixedtime, "
                         "maxpressure), aligned pair (fixedtime_kcf, maxpressure_1to1)",
        "guards": {"g1": sources.guards.get("g1"), "g2": sources.guards.get("g2"), "passed": sources.guards.get("passed"),
                   "control": {"n_seeds_equal": sources.control.get("n_seeds_equal"),
                               "passed": sources.control.get("passed")}},
        **{name: result[name] for name in ("r1", "r2", "r3", "r4", "r5", "r6", "r7")},
        "also": result["also"],
        "statistics": {"unit": "the paired draw: each draw's value is the mean of the five seeds",
                       "ci": "mean +/- 1.96 * sd / sqrt(n), sd with ddof 1 (offline.dt_gate.mean_ci95)",
                       "test": "offline.dt_gate.wilcoxon_signed_rank, two-sided",
                       "multiplicity": "none: the task is exploratory (plan section 5)",
                       "sumo_primary": "e_sumo (A15), att_env beside it",
                       "cityflow_primary": "att_engine (A11(c), Rule R), att_ours beside it"},
        "sentences": choose_sentences(result),
        "what_this_does_not_say": WHAT_THIS_DOES_NOT_SAY,
        "inputs": dict(sources.inputs),
        "sessions": list(sources.sessions),
        "code_commits": dict(sources.code_commits),
        **tc._git_provenance(),
    }


def build_cityflow(sources: Sources) -> dict[str, Any]:
    """``docs/data/p9_4_cityflow.json``: the 500 records in P9.1's hz1x1 format."""
    import offline.transfer_curve as tc

    return {"format_version": CITYFLOW_FORMAT_VERSION, "task": "P9.4", "label": p94.LABEL,
            "record_format": "P9.1's hz1x1 records (offline/corrective_report.py, p9.1-report/1.0's EPISODE_FIELDS)",
            "records": {f"p9_4:hz1x1:dt@{p94.TIER}": [dict(row) for row in sources.cityflow_rows]},
            "original": f"docs/data/{p94.P9_1_IQL_TARGETS} records p4_7:dt@{p94.REGISTERED_TIER}",
            "inputs": {key: sources.inputs.get(key) for key in ("route_i_sha256", "route_ii")},
            "code_commit": sources.code_commits.get("cityflow"), **tc._git_provenance()}


def build_training(sources: Sources) -> dict[str, Any]:
    """``docs/data/p9_4_training.json``: every run's record, the control record and the train sessions' commit."""
    import offline.transfer_curve as tc

    return {"format_version": TRAINING_FORMAT_VERSION, "task": "P9.4", "label": p94.LABEL, "clock": p94.CLOCK_LABEL,
            "runs": [dict(run) for run in sources.training_runs], "control": dict(sources.control),
            "declaration_sha256": sources.inputs.get("docs_data_sha256", {}).get(p94.DECLARATION_NAME),
            "code_commit": sources.code_commits.get("train"), **tc._git_provenance()}


def write_artifacts(roots: p94.Roots, out_dir: str | Path | None = None) -> dict[str, str]:
    """Load, build and write the three artifacts into *out_dir* (``docs/data/`` by default).  Every refusal precedes
    every write; an existing file is replaced only by the identical bytes."""
    sources = load_sources(roots)
    target = Path(out_dir) if out_dir is not None else roots.data_dir
    payloads = {"p9_4_zero_shot.json": build_zero_shot(sources), "p9_4_cityflow.json": build_cityflow(sources),
                "p9_4_training.json": build_training(sources)}
    texts = {name: p94.canonical_text(payload) for name, payload in payloads.items()}
    for name, text in texts.items():
        path = target / name
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{path} exists with other content; an artifact is not rewritten")
    written = {}
    for name, text in texts.items():
        path = target / name
        if not path.exists():
            with open(path, "x", encoding="utf-8") as handle:
                handle.write(text)
        written[name] = p94.file_sha256(path)
    return written
