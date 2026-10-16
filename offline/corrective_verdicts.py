"""P9.1's verdicts (``BRIEF_46`` 3.4 ``rescore``; A27(b)-(e), (g)): the registered rules applied to the new cells.

Nothing here re-implements a rule.  The paired contrast is P5.2c's (``tier_statements.contrast``: per-draw means across
the seeds, ``dt_gate.mean_ci95``, refused unless both cells cover the identical ``(seed, draw)`` grid); HELD / FAILED /
NOT RESOLVED is P5.2c's interval rule (``tier_statements.interval_outcome``, ``conjunction``); the re-scored grid
statements are P5.2c's scorers (``tier_statements.all_statements``) on a book with the new cell substituted; collapse is
Q6's two-valued vocabulary.  Primary definition ``att_engine``; ``att_ours`` beside (A27(a)).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PRIMARY = "att_engine"
DEFINITIONS: tuple[str, ...] = ("att_engine", "att_ours")

#: A27(b): the four grid baselines whose matched arm replaces them, on the four grid tiers.
A27B_KEYS: tuple[tuple[str, str], ...] = tuple(
    (method, tier) for tier in ("mappo1000", "maxpressure", "fixedtime", "random")
    for method in ("bc", "bc_top10", "bc_top10_perix", "iql")
)


def _numbers(block: Mapping[str, Any]) -> dict[str, Any]:
    """The interval and the per-seed block of a P5.2c contrast, as an artifact carries them."""
    keep = ("mean_difference", "ci95_low", "ci95_high", "ci95_half_width", "ci95_width", "n_draws", "n_seeds",
            "exactly_zero", "sign", "reading", "per_seed", "seed_reversal_qualifier", "discriminability",
            "paired_comparison")
    return {key: block[key] for key in keep}


def contrast_both(left: Any, right: Any, *, tier: str) -> dict[str, dict[str, Any]]:
    """``left - right`` by P5.2c's contrast under both definitions."""
    import offline.tier_statements as tst

    return {definition: tst.contrast(left, right, definition=definition, tier=tier) for definition in DEFINITIONS}


def predicted_contrast(left: Any, right: Any, *, tier: str, predicted: str) -> dict[str, Any]:
    """``left - right`` predicted *predicted*, HELD / FAILED / NOT RESOLVED by P5.2c's interval rule under each
    definition; the verdict is ``att_engine``'s, ``att_ours`` beside."""
    import offline.tier_statements as tst

    by_definition = {}
    for definition, block in contrast_both(left, right, tier=tier).items():
        by_definition[definition] = {**_numbers(block), "outcome": tst.interval_outcome(
            block["ci95_low"], block["ci95_high"], predicted=predicted)}
    return {"left": left.spec.cell_id, "right": right.spec.cell_id, "left_arm": left.spec.method,
            "right_arm": right.spec.method, "tier": tier, "predicted": predicted,
            "verdict": by_definition[PRIMARY]["outcome"], "verdict_definition": PRIMARY, "by_definition": by_definition}


#: A27(e)'s four contrasts, in the registration's order; each is predicted NEGATIVE (the left arm's ATT is lower).
MIXTURE_CONTRASTS: tuple[tuple[str, str], ...] = (("bc_top10", "bc"), ("bc_top10", "iql"), ("iql", "dt"), ("dt", "bc"))
#: A27(e)'s "every arm within 5 s of bc_top10": the three other predicted arms (``iql_t`` is reported beside).
WITHIN_ARMS: tuple[str, ...] = ("dt", "bc", "iql")
WITHIN_SECONDS = 5.0
WITHIN_RULE = (
    "within 5 s read literally, |arm - bc_top10| < 5 s, three-valued by P5.2c's interval rule applied to both edges of "
    "the paired interval: HELD iff the interval lies strictly inside (-5, +5), FAILED iff it lies entirely above +5 or "
    "entirely below -5, NOT RESOLVED otherwise; the verdict over the arms is P5.2c's conjunction"
)


def mixture_orderings(cells: Mapping[str, Any], *, tier: str) -> dict[str, Any]:
    """A27(e): the four ordering contrasts, each predicted negative, under P5.2c's rule."""
    contrasts = [predicted_contrast(cells[left], cells[right], tier=tier, predicted="negative")
                 for left, right in MIXTURE_CONTRASTS]
    return {"contrasts": contrasts, "registered_order": "bc_top10 < iql < dt < bc (att_horizon on both parent tiers, P4.6)",
            "rule": "P5.2c: HELD / FAILED / NOT RESOLVED by the side of zero of the paired interval over per-draw means "
                    "across the five seeds, n = 100, dt_gate.mean_ci95; under att_engine, att_ours beside (A27(e))"}


def window_outcome(low: float, high: float, *, seconds: float = WITHIN_SECONDS) -> str:
    """:data:`WITHIN_RULE` on one interval, by ``tier_statements.interval_outcome`` on each edge."""
    import offline.tier_statements as tst

    upper = tst.interval_outcome(low - seconds, high - seconds, predicted="negative")
    lower = tst.interval_outcome(low + seconds, high + seconds, predicted="positive")
    if upper == "HELD" and lower == "HELD":
        return "HELD"
    if upper == "FAILED" or lower == "FAILED":
        return "FAILED"
    return "NOT RESOLVED"


def within_seconds(cells: Mapping[str, Any], *, tier: str, reference: str = "bc_top10",
                   arms: Sequence[str] = WITHIN_ARMS, seconds: float = WITHIN_SECONDS) -> dict[str, Any]:
    """A27(e): every arm within *seconds* of *reference*, per arm and by conjunction."""
    import offline.tier_statements as tst

    rows = []
    for arm in arms:
        by_definition = {}
        for definition, block in contrast_both(cells[arm], cells[reference], tier=tier).items():
            by_definition[definition] = {**_numbers(block), "outcome": window_outcome(
                block["ci95_low"], block["ci95_high"], seconds=seconds)}
        rows.append({"arm": arm, "reference": reference, "verdict": by_definition[PRIMARY]["outcome"],
                     "by_definition": by_definition})
    by_definition_verdict = {definition: tst.conjunction([row["by_definition"][definition]["outcome"] for row in rows])
                             for definition in DEFINITIONS}
    return {"arms": rows, "seconds": float(seconds), "rule": WITHIN_RULE, "verdict": by_definition_verdict[PRIMARY],
            "verdict_by_definition": by_definition_verdict}


#: A27(g)'s four registered recipes, one ceiling.
BUDGET_ARMS: tuple[str, ...] = ("dt", "bc", "bc_top10", "iql")
CONCORDANCE_RULE = (
    "for each of the six arm pairs, the predicted direction is the sign of the registered 40,000-step paired point "
    "difference (left - right) and the 100,000-step paired interval is judged by P5.2c's rule against it: HELD = "
    "concordant and resolved, FAILED = reversed and resolved, NOT RESOLVED otherwise (also where the 40,000-step "
    "difference is an exact tie, which predicts no order); the ordering's verdict is P5.2c's conjunction over the pairs"
)


def budget_block(cells_40k: Mapping[str, Any], cells_100k: Mapping[str, Any], *, tier: str,
                 arms: Sequence[str] = BUDGET_ARMS) -> dict[str, Any]:
    """A27(g): per arm the paired 100k - 40k difference; the ordering's concordance over the arm pairs."""
    import offline.tier_statements as tst

    differences = []
    for arm in arms:
        blocks = contrast_both(cells_100k[arm], cells_40k[arm], tier=tier)
        differences.append({"arm": arm, "left": cells_100k[arm].spec.cell_id, "right": cells_40k[arm].spec.cell_id,
                            "by_definition": {d: _numbers(b) for d, b in blocks.items()}})
    concordance = {}
    for definition in DEFINITIONS:
        pairs = []
        for i, left in enumerate(arms):
            for right in arms[i + 1:]:
                before = tst.contrast(cells_40k[left], cells_40k[right], definition=definition, tier=tier)
                after = tst.contrast(cells_100k[left], cells_100k[right], definition=definition, tier=tier)
                point = float(before["mean_difference"])
                predicted = "negative" if point < 0.0 else "positive" if point > 0.0 else None
                outcome = ("NOT RESOLVED" if predicted is None else
                           tst.interval_outcome(after["ci95_low"], after["ci95_high"], predicted=predicted))
                pairs.append({"left_arm": left, "right_arm": right, "predicted": predicted, "outcome": outcome,
                              "note": ("an exact tie at 40,000 steps predicts no order" if predicted is None else
                                       "the 40,000-step order, judged on the 100,000-step interval"),
                              "at_40k": _numbers(before), "at_100k": _numbers(after)})
        outcomes = [pair["outcome"] for pair in pairs]
        concordance[definition] = {"pairs": pairs, "n_held": outcomes.count("HELD"),
                                   "n_failed": outcomes.count("FAILED"), "n_not_resolved": outcomes.count("NOT RESOLVED"),
                                   "verdict": tst.conjunction(outcomes)}
    return {"tier": tier, "arms": list(arms), "differences": differences, "concordance": concordance,
            "concordance_rule": CONCORDANCE_RULE,
            "difference_rule": "per arm, 100,000 - 40,000 steps, paired per draw across the five seeds (A27(g)); no "
                               "direction is predicted, so the reading is the interval's side of zero"}


def paired_differences(pairs: Mapping[str, tuple[Any, Any]], *, left: str, right: str) -> dict[str, Any]:
    """``left - right`` per tier, both definitions; the count of intervals excluding zero and the range of the point
    differences (A27(c)'s sentence: across the tiers, with a CI that excludes zero on k of them)."""
    tiers = []
    for tier, (left_cell, right_cell) in pairs.items():
        by_definition = {}
        for definition, block in contrast_both(left_cell, right_cell, tier=tier).items():
            by_definition[definition] = {**_numbers(block), "excludes_zero": bool(
                block["ci95_high"] < 0.0 or block["ci95_low"] > 0.0)}
        tiers.append({"tier": tier, "left": left_cell.spec.cell_id, "right": right_cell.spec.cell_id,
                      "by_definition": by_definition})
    count = {d: sum(entry["by_definition"][d]["excludes_zero"] for entry in tiers) for d in DEFINITIONS}
    span = {d: [min(e["by_definition"][d]["mean_difference"] for e in tiers),
                max(e["by_definition"][d]["mean_difference"] for e in tiers)] for d in DEFINITIONS}
    return {"left_arm": left, "right_arm": right, "tiers": tiers, "n_excluding_zero": count, "range": span}


def rescore_p5_2(books: Mapping[str, Mapping[tuple[str, str], Any]],
                 substitution: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    """Every P5.2c statement on the REGISTERED book (``corrected``: P5.2b's random-tier IQL cell) with *substitution*'s
    cells in place of their originals, by P5.2c's own ``all_statements``.  Both versions are handed the same book, so
    every entry comes back ``version: "both"``."""
    import offline.tier_statements as tst

    book = dict(books["corrected"])
    unknown = sorted(set(substitution) - set(book))
    if unknown:
        raise ValueError(f"{unknown} are not cells of P5.2c's book; a substitution replaces a registered cell")
    book.update(substitution)
    return tst.all_statements({"corrected": book, "as_committed": book})


def _key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (entry["id"], entry["tier"], entry["arm"], entry["definition"])


def moved_block(original: Sequence[Mapping[str, Any]], rescored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per statement: the registered decision on the original book (its ``both`` / ``corrected`` entry) beside the
    re-scored one.  ``moved`` iff the statement is a verdict and its verdict differs (A27(b): a verdict MOVES iff the
    registered rule's decision changes); ``reading_changed`` for every kind."""
    before = {_key(e): e for e in original if e["version"] in ("both", "corrected")}
    after = {_key(e): e for e in rescored}
    if set(before) != set(after):
        raise ValueError(f"the original and the re-scored books do not score the same statements "
                         f"({len(before)} and {len(after)})")
    out = []
    for key in sorted(before, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        old, new = before[key], after[key]
        out.append({"id": key[0], "tier": key[1], "arm": key[2], "definition": key[3], "kind": old["kind"],
                    "primary": key[3] == PRIMARY, "original_verdict": old["verdict"], "verdict": new["verdict"],
                    "original_reading": old["reading"], "reading": new["reading"],
                    "moved": bool(old["kind"] == "verdict" and old["verdict"] != new["verdict"]),
                    "reading_changed": bool(old["reading"] != new["reading"]),
                    "original_inputs": list(old["inputs"]), "inputs": list(new["inputs"])})
    return out


def collapse_rows(arms: Mapping[str, Any], anchors: Mapping[str, Any], *, tier: str) -> dict[str, Any]:
    """Every arm against every random reference: Q6's two-valued rule (COLLAPSED iff the arm is worse than the
    reference with the paired CI excluding 0), counted per reference (A27(d): every collapse check against both)."""
    rows = []
    for arm, cell in arms.items():
        for name, anchor in anchors.items():
            by_definition = {}
            for definition, block in contrast_both(cell, anchor, tier=tier).items():
                by_definition[definition] = {**_numbers(block), "verdict": (
                    "COLLAPSED" if block["ci95_low"] > 0.0 else "NOT COLLAPSED")}
            rows.append({"arm": arm, "anchor": name, "left": cell.spec.cell_id, "right": anchor.spec.cell_id,
                         "verdict": by_definition[PRIMARY]["verdict"], "by_definition": by_definition})
    counts = {d: {name: sum(r["by_definition"][d]["verdict"] == "COLLAPSED" for r in rows if r["anchor"] == name)
                  for name in anchors} for d in DEFINITIONS}
    return {"tier": tier, "rows": rows, "collapsed_count": counts,
            "rule": "Q6's two-valued vocabulary (P5.2c): COLLAPSED iff the arm's mean is worse than the reference's "
                    "with the paired CI excluding 0, else NOT COLLAPSED; against each random reference (A27(d))"}


def _by_seed(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    out: dict[int, list[Mapping[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (int(r["seed"]), int(r["draw_id"]))):
        out.setdefault(int(row["seed"]), []).append(row)
    keys = [(int(r["seed"]), int(r["draw_id"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("an episode appears twice in a P3a record")
    return out


def _columns(row: Mapping[str, Any]) -> list[str]:
    return [str(v) for v in row["per_intersection_sha256"]]


P3A_RULE = (
    "A27(d)'s P3a, the author's wording: on no seed does the DT emit the same action sequence on two of the 100 held-out "
    "draws (nortg_decomposition.action_sequence_sha256 over the int64 bytes of the episode's action array); on the grid "
    "an episode counts as repeated if any intersection's sequence equals that intersection's sequence on another draw "
    "of the same seed; HELD iff no episode repeats, FAILED if any does"
)


def p3a(rows: Sequence[Mapping[str, Any]], *, per_intersection: bool) -> dict[str, Any]:
    """P3a's count: the episodes whose sequence (the joint array, or on the grid any intersection's column) appears on
    another draw of the same seed."""
    from collections import Counter

    repeated: list[list[int]] = []
    for seed, seed_rows in _by_seed(rows).items():
        if per_intersection:
            n = len(_columns(seed_rows[0]))
            counts = [Counter(_columns(row)[i] for row in seed_rows) for i in range(n)]
            hits = [row for row in seed_rows if any(counts[i][_columns(row)[i]] > 1 for i in range(n))]
        else:
            joint = Counter(str(row["action_sequence_sha256"]) for row in seed_rows)
            hits = [row for row in seed_rows if joint[str(row["action_sequence_sha256"])] > 1]
        repeated += [[seed, int(row["draw_id"])] for row in hits]
    return {"episodes": len(rows), "repeated_episodes": len(repeated), "repeated": repeated,
            "unit": "per intersection" if per_intersection else "the joint action array",
            "verdict": "HELD" if not repeated else "FAILED", "rule": P3A_RULE}


def p3a_beside(rows: Sequence[Mapping[str, Any]], *, references: Mapping[str, Sequence[Mapping[str, Any]]],
               training: Mapping[str, Any], per_intersection: bool = False) -> dict[str, Any]:
    """A27(d)'s beside-table, without a prediction: per seed, the distinct sequences and the share of draws carrying
    the most common one, identity with each random reference's episode on the same ``(seed, draw)``, and replays of
    the tier's training schedules (``training``: the joint digests and each intersection's, from the corpus)."""
    from collections import Counter

    reference_index = {name: {(int(r["seed"]), int(r["draw_id"])): r for r in ref_rows}
                       for name, ref_rows in references.items()}
    per_seed = {}
    for seed, seed_rows in _by_seed(rows).items():
        joint = Counter(str(row["action_sequence_sha256"]) for row in seed_rows)
        entry: dict[str, Any] = {
            "episodes": len(seed_rows), "distinct_sequences": len(joint),
            "most_common_share": max(joint.values()) / len(seed_rows),
            "identical_to_reference": {name: sum(
                1 for row in seed_rows if (seed, int(row["draw_id"])) in index and
                str(index[(seed, int(row["draw_id"]))]["action_sequence_sha256"]) == str(row["action_sequence_sha256"]))
                for name, index in reference_index.items()},
            "replays_a_training_schedule": sum(str(row["action_sequence_sha256"]) in training["joint"]
                                               for row in seed_rows),
        }
        if per_intersection:
            n = len(_columns(seed_rows[0]))
            columns = [Counter(_columns(row)[i] for row in seed_rows) for i in range(n)]
            entry["per_intersection_distinct"] = [len(c) for c in columns]
            entry["per_intersection_most_common_share"] = [max(c.values()) / len(seed_rows) for c in columns]
            entry["episodes_with_an_intersection_identical_to_reference"] = {name: sum(
                1 for row in seed_rows if (seed, int(row["draw_id"])) in index and any(
                    a == b for a, b in zip(_columns(row), _columns(index[(seed, int(row["draw_id"]))]))))
                for name, index in reference_index.items()}
            entry["episodes_replaying_an_intersections_training_schedule"] = sum(
                1 for row in seed_rows if any(_columns(row)[i] in training["per_intersection"][i] for i in range(n)))
        per_seed[str(seed)] = entry
    return {"per_seed": per_seed, "prediction": None,
            "note": "reported beside P3a without a prediction (A27(d)); P5.3b's old-tier numbers: 1 sequence, 100 % of "
                    "draws, on every seed"}


def a6_block(dt: Any, baseline: Any, *, delta: float) -> dict[str, Any]:
    """P4.4's registered statement (A6) for ``dt - baseline``, by its own route: ``offline_baselines.paired_comparison``
    (per-draw means over the seeds) and ``equivalence_verdict`` at the registered delta, the full-precision delta's
    verdict beside it -- under both definitions."""
    from offline import offline_baselines as ob
    from offline.iql_correction import episodes_from_rows

    by_definition = {}
    for definition in DEFINITIONS:
        comparison = ob.paired_comparison(episodes_from_rows(dt.rows, value_key=definition),
                                          episodes_from_rows(baseline.rows, value_key=definition))
        by_definition[definition] = {
            "mean_difference": float(comparison.mean_difference), "ci95_low": float(comparison.ci95_low),
            "ci95_high": float(comparison.ci95_high), "ci95_half_width": float(comparison.ci95_half_width),
            "n_shared_draws": int(comparison.n_shared_draws),
            "verdict": ob.equivalence_verdict(comparison.mean_difference, comparison.ci95_half_width, delta),
            "verdict_at_full_precision_delta": ob.equivalence_verdict(
                comparison.mean_difference, comparison.ci95_half_width, ob.DELTA_ATT_DERIVATION),
        }
    return {"left": dt.spec.cell_id, "right": baseline.spec.cell_id, "delta": float(delta),
            "rule": "PREREGISTRATION.md A6 via offline_baselines.equivalence_verdict (P4.4's own route)",
            "by_definition": by_definition}


def a6_rescore(dt: Any, original: Any, rescored: Any, *, delta: float) -> dict[str, Any]:
    """A6 on the original baseline and on its replacement; ``moved`` per definition iff the verdict differs."""
    before = a6_block(dt, original, delta=delta)
    after = a6_block(dt, rescored, delta=delta)
    return {"original": before, "rescored": after, "moved": {
        d: before["by_definition"][d]["verdict"] != after["by_definition"][d]["verdict"] for d in DEFINITIONS}}


def contrast_entry(left: Any, right: Any, *, tier: str) -> dict[str, Any]:
    """``left - right`` under both definitions, with no prediction: the interval and its reading (the beside-numbers and
    the sentences' contrasts)."""
    return {"left": left.spec.cell_id, "right": right.spec.cell_id, "left_arm": left.spec.method,
            "right_arm": right.spec.method, "tier": tier,
            "by_definition": {d: _numbers(b) for d, b in contrast_both(left, right, tier=tier).items()}}
