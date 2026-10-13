"""P5.2c (``BRIEF_45`` and its Amendment A): ``offline/tier_statements.py`` -- P5.2's registered statements scored by
committed code under both ATT definitions and both random-tier IQL cells.

Written against a signature-only skeleton, so every test reaches the real API and is red for its own reason first.  Every
expected value is computed HERE by a second route -- plain loops, ``statistics``, ``itertools.permutations`` over the
raw rows -- never by the scorer under test.

Ungated (CI): T-contrast, T-outcome, T-ties, T-composer, T-q0, T-q4b, T-q5, T-q6, T-versions, T-definitions, T-inputs
(a synthetic ``output/`` and repository, ``tests/p5_2c_fixtures.py``), T-barrier, T-canonical, T-cli, T-code-root, T-quotes.
Gated on ``RLTRAFFIC_OUTPUT_ROOT`` (the main tree's ``output/``; each skip names it): T-inputs-real, T-reproduce-record,
T-reproduce-P8.4b, T-reproduce-P5.2b, T-reproduce-P5.1, T-note, T-companions, T-first-time, T-check-real and T-regress.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import offline.iql_correction as ic
import offline.tier_statements as tst
import offline.tier_sweep as ts
from offline.dt_gate import EpisodeResult
from tests import p5_2c_fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = REPO_ROOT / "docs" / "data" / "p5_2_statements.json"
SYNTHETIC_GIT: dict[str, Any] = {"commit": "0" * 40, "dirty": False, "code_root": "synthetic"}
SYNTHETIC_CODE: list[dict[str, Any]] = [{"module": "offline.tier_statements", "path": "offline/tier_statements.py",
                                         "sha256": "0" * 64}]


# ----------------------------------------------------------------------
# The test's own arithmetic (the second route)
# ----------------------------------------------------------------------


def seq_mean(values: Sequence[float]) -> float:
    """A plain left-to-right sum divided by n: what ``numpy.mean`` computes for fewer than eight values."""
    total = 0.0
    for value in values:
        total += float(value)
    return total / len(values)


def by_draw(rows: Sequence[Mapping[str, Any]], definition: str) -> dict[int, list[float]]:
    """Each draw's values in seed order."""
    out: dict[int, list[float]] = {}
    for r in sorted(rows, key=lambda r: (int(r["seed"]), int(r["draw_id"]))):
        out.setdefault(int(r["draw_id"]), []).append(float(r[definition]))
    return out


def by_seed(rows: Sequence[Mapping[str, Any]], definition: str) -> dict[int, list[float]]:
    """Each seed's values in draw order."""
    out: dict[int, list[float]] = {}
    for r in sorted(rows, key=lambda r: (int(r["seed"]), int(r["draw_id"]))):
        out.setdefault(int(r["seed"]), []).append(float(r[definition]))
    return out


def own_interval(differences: Sequence[float]) -> tuple[float, float, float, float]:
    """``statistics``' mean and sample sd, and ``1.96 * sd / sqrt(n)``: (mean, sd, low, high)."""
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half = 1.96 * sd / math.sqrt(len(differences))
    return mean, sd, mean - half, mean + half


def own_outcome(low: float, high: float, predicted: str) -> str:
    if predicted == "negative":
        return "HELD" if high < 0 else ("FAILED" if low > 0 else "NOT RESOLVED")
    return "HELD" if low > 0 else ("FAILED" if high < 0 else "NOT RESOLVED")


def own_paired(left: tst.Cell, right: tst.Cell, definition: str) -> list[float]:
    """Per-draw differences of the seed means, by statistics.mean, draws ascending."""
    a, b = by_draw(left.rows, definition), by_draw(right.rows, definition)
    assert sorted(a) == sorted(b)
    return [statistics.mean(a[d]) - statistics.mean(b[d]) for d in sorted(a)]


def own_contrast_outcome(left: tst.Cell, right: tst.Cell, definition: str, predicted: str) -> str:
    """The registered interval rule on THIS file's interval: an exact zero has CI [0, 0] and resolves to neither side."""
    _, _, low, high = own_interval(own_paired(left, right, definition))
    return own_outcome(low, high, predicted)


def own_conjunction(outcomes: Sequence[str]) -> str:
    """``BRIEF_45`` section 3.3's conjunction, written out: FAILED if any, HELD if all, NOT RESOLVED otherwise."""
    values = list(outcomes)
    return "FAILED" if "FAILED" in values else ("HELD" if all(v == "HELD" for v in values) else "NOT RESOLVED")


def seq_contrast(left: tst.Cell, right: tst.Cell, definition: str) -> float:
    """The registered route's mean, by plain loops: exact for fewer than eight seeds and eight draws (``numpy`` sums
    such short vectors left to right)."""
    a, b = by_draw(left.rows, definition), by_draw(right.rows, definition)
    assert sorted(a) == sorted(b) and len(a) < 8 and all(len(v) < 8 for v in a.values())
    return seq_mean([seq_mean(a[d]) - seq_mean(b[d]) for d in sorted(a)])


def own_level(c: tst.Cell, definition: str) -> float:
    return statistics.mean(float(r[definition]) for r in c.rows)


def consistent_orders(levels: Mapping[str, float]) -> list[tuple[str, ...]]:
    """Every order of the arms consistent with *levels*: all permutations, kept iff levels never decrease."""
    arms = list(levels)
    return [order for order in itertools.permutations(arms)
            if all(levels[a] <= levels[b] for a, b in zip(order, order[1:]))]


def own_concordant(predicted: Sequence[str], order: Sequence[str], arms: Sequence[str]) -> int:
    keep = [a for a in predicted if a in arms]
    rank_p = {a: i for i, a in enumerate(keep)}
    rank_m = {a: i for i, a in enumerate(a for a in order if a in arms)}
    return sum(1 for a, b in itertools.combinations(sorted(keep), 2)
               if (rank_p[a] < rank_p[b]) == (rank_m[a] < rank_m[b]))


def own_concordant_rule(predicted: Sequence[str], levels: Mapping[str, float], arms: Sequence[str]) -> int:
    """The declared tie rule by its definition: a pair is concordant iff its levels differ and are ordered as predicted."""
    keep = [a for a in predicted if a in arms]
    rank_p = {a: i for i, a in enumerate(keep)}
    return sum(1 for a, b in itertools.combinations(keep, 2)
               if levels[a] != levels[b] and (rank_p[a] < rank_p[b]) == (levels[a] < levels[b]))


def pick(statements: Sequence[Mapping[str, Any]], statement_id: str, *, tier: str | None, definition: str,
         version: str = "corrected", arm: str | None = None) -> Mapping[str, Any]:
    found = [s for s in statements if s["id"] == statement_id and s["tier"] == tier and s["definition"] == definition
             and s.get("arm") == arm and s["version"] in (version, "both")]
    assert len(found) == 1, f"{statement_id} {tier} {definition} {version} {arm}: {len(found)} entries"
    return found[0]


def contains(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every field of *expected* present in *actual* and equal (dicts may gain keys; lists must be equal)."""
    problems: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a dict, got {type(actual).__name__}"]
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                problems.extend(contains(value, actual[key], f"{path}.{key}"))
        return problems
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return [f"{path}: {expected!r} != {actual!r}"]
        for i, (e, a) in enumerate(zip(expected, actual)):
            problems.extend(contains(e, a, f"{path}[{i}]"))
        return problems
    if expected != actual or type(expected) is not type(actual) and not (
            isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool)):
        return [f"{path}: {expected!r} != {actual!r}"]
    return problems


def five_seed_cells(values_left: Mapping[int, Sequence[float]], values_right: Mapping[int, Sequence[float]],
                    seeds: Sequence[int] = (101, 202, 303, 404, 505)) -> tuple[tst.Cell, tst.Cell]:
    left = {(s, d): v for d, vs in values_left.items() for s, v in zip(seeds, vs)}
    right = {(s, d): v for d, vs in values_right.items() for s, v in zip(seeds, vs)}
    return fx.simple_cell("dt_spatial", "maxpressure", left), fx.simple_cell("dt_nomix", "maxpressure", right)


@pytest.fixture
def books() -> dict[str, dict[tuple[str, str], tst.Cell]]:
    return tst.make_books(fx.synthetic_cells())


# ----------------------------------------------------------------------
# T-contrast
# ----------------------------------------------------------------------


def test_t_contrast_route_pin_is_numpy_per_draw_and_not_the_compensated_sum() -> None:
    """F1 (Amendment A, A2 Q1): the per-draw mean is ``dt_gate._per_draw_means`` (``numpy.mean``, a plain sequential sum
    over five seeds), not Python's compensated built-in ``sum``.  On these values the two routes differ at the result."""
    left = {1000: [0.1, 0.2, 0.3, 0.7, 0.9], 1001: [1.1, 2.2, 3.3, 0.1, 0.2], 1002: [0.3, 0.6, 0.1, 0.2, 0.7]}
    right = {d: [0.0] * 5 for d in left}
    sequential = seq_mean([seq_mean(left[d]) - seq_mean(right[d]) for d in sorted(left)])
    compensated = seq_mean([sum(left[d]) / 5 - sum(right[d]) / 5 for d in sorted(left)])
    assert sequential != compensated, "the fixture no longer discriminates the two routes"
    a, b = five_seed_cells(left, right)
    block = tst.contrast(a, b, definition="att_ours", tier="maxpressure")
    assert block["mean_difference"] == sequential
    assert block["n_draws"] == 3 and block["n_seeds"] == 5


def test_t_contrast_equals_statistics_on_dyadic_values_where_every_route_is_exact() -> None:
    """Mean, sd, half-width, low, high and width against ``statistics`` and ``1.96 * sd / sqrt(n)``, under ``==``, on
    values exact in binary: four seeds (a per-draw mean divides by 4) and five draws (the variance divides by 4)."""
    seeds = (101, 202, 303, 404)
    left = {(s, d): 100.0 + (s % 7) / 16.0 + (d - 1000) * 1.5 + (s * d % 5) / 8.0 for s in seeds for d in range(1000, 1005)}
    right = {(s, d): 99.0 + (s % 5) / 16.0 + (d % 3) / 4.0 for s in seeds for d in range(1000, 1005)}
    a = fx.simple_cell("dt_spatial", "random", left)
    b = fx.simple_cell("dt_nomix", "random", right)
    differences = own_paired(a, b, "att_ours")
    mean, sd, low, high = own_interval(differences)
    block = tst.contrast(a, b, definition="att_ours", tier="random")
    assert block["mean_difference"] == mean
    assert block["std"] == sd
    assert block["ci95_half_width"] == 1.96 * sd / math.sqrt(5)
    assert (block["ci95_low"], block["ci95_high"]) == (low, high)
    assert block["ci95_width"] == high - low
    assert block["n_draws"] == 5 and block["exactly_zero"] is False


def test_t_contrast_episode_level_pairing_would_change_n_and_the_interval() -> None:
    """The paired unit is the DRAW (n = draws), never the episode (n = seeds x draws): P8.4b's wiring defect."""
    seeds = (101, 202, 303, 404)
    left = {(s, d): 10.0 + (s % 9) + d % 4 for s in seeds for d in range(1000, 1006)}
    right = {(s, d): 9.0 + (s % 4) / 2.0 + d % 3 for s in seeds for d in range(1000, 1006)}
    a, b = fx.simple_cell("bc", "fixedtime", left), fx.simple_cell("bc_top10", "fixedtime", right)
    block = tst.contrast(a, b, definition="att_ours", tier="fixedtime")
    _, _, low, high = own_interval(own_paired(a, b, "att_ours"))
    per_episode = [left[k] - right[k] for k in sorted(left)]
    _, _, low_e, high_e = own_interval(per_episode)
    assert (low, high) != (low_e, high_e)
    assert block["n_draws"] == 6
    assert (block["ci95_low"], block["ci95_high"]) == (low, high)


def test_t_contrast_exact_zero_is_ci_zero_not_resolved_sign_zero_and_named() -> None:
    values = {(s, d): 150.0 + s / 64.0 + d / 32.0 for s in (101, 202, 303) for d in (1000, 1001, 1002, 1003)}
    a = fx.simple_cell("dt_spatial", "fixedtime", values)
    b = fx.simple_cell("dt_nomix", "fixedtime", values)
    block = tst.contrast(a, b, definition="att_ours", tier="fixedtime")
    assert block["exactly_zero"] is True
    assert (block["mean_difference"], block["ci95_low"], block["ci95_high"]) == (0.0, 0.0, 0.0)
    assert block["sign"] == "zero"
    assert tst.interval_outcome(block["ci95_low"], block["ci95_high"], predicted="negative") == "NOT RESOLVED"
    assert tst.interval_outcome(block["ci95_low"], block["ci95_high"], predicted="positive") == "NOT RESOLVED"
    assert block["discriminability"]["identical"] is True
    d8 = block["paired_comparison"]
    assert d8["wilcoxon"]["n_used"] == 0
    assert d8["rank_biserial"] == 0.0 and d8["rank_biserial_defined"] is False
    assert "undefined" in d8["rank_biserial_note"]
    assert "identical episodes" in block["reading"]


def test_t_contrast_a_straddling_interval_is_not_resolved() -> None:
    seeds = (101, 202, 303)
    left = {(s, d): 100.0 + (1.0 if d % 2 else -1.0) + s / 1000.0 for s in seeds for d in range(1000, 1006)}
    right = {(s, d): 100.0 + s / 1000.0 for s in seeds for d in range(1000, 1006)}
    a, b = fx.simple_cell("bc", "random", left), fx.simple_cell("iql", "random", right)
    block = tst.contrast(a, b, definition="att_ours", tier="random")
    assert block["ci95_low"] < 0.0 < block["ci95_high"]
    assert tst.interval_outcome(block["ci95_low"], block["ci95_high"], predicted="negative") == "NOT RESOLVED"
    assert tst.interval_outcome(block["ci95_low"], block["ci95_high"], predicted="positive") == "NOT RESOLVED"


def test_t_contrast_a_reversing_seed_emits_the_qualifier_naming_it() -> None:
    seeds = (101, 202, 303)
    shift = {101: -2.0, 202: -3.0, 303: +1.0}
    left = {(s, d): 50.0 + shift[s] + (d % 3) / 8.0 for s in seeds for d in range(1000, 1004)}
    right = {(s, d): 50.0 + (d % 3) / 8.0 for s in seeds for d in range(1000, 1004)}
    a, b = fx.simple_cell("dt_spatial", "random", left), fx.simple_cell("dt_nomix", "random", right)
    block = tst.contrast(a, b, definition="att_ours", tier="random")
    own = {s: seq_mean(by_seed(a.rows, "att_ours")[s]) - seq_mean(by_seed(b.rows, "att_ours")[s]) for s in seeds}
    assert block["per_seed"]["by_seed"] == {str(s): v for s, v in own.items()}
    assert block["per_seed"]["reverses_against_pooled_sign"] == 1
    assert block["seed_reversal_qualifier"] is not None and "303" in block["seed_reversal_qualifier"]


def test_t_contrast_refuses_a_missing_draw() -> None:
    seeds = (101, 202)
    full = {(s, d): 10.0 + d % 3 for s in seeds for d in (1000, 1001, 1002)}
    short = {k: v for k, v in full.items() if k != (202, 1002)}
    with pytest.raises(ValueError, match=r"same \(seed, draw\) grid"):
        tst.contrast(fx.simple_cell("bc", "random", full), fx.simple_cell("iql", "random", short),
                     definition="att_ours", tier="random")


def test_t_contrast_refuses_mismatched_seeds() -> None:
    left = {(s, d): 10.0 + d % 3 for s in (101, 202) for d in (1000, 1001)}
    right = {(s, d): 11.0 + d % 3 for s in (101, 303) for d in (1000, 1001)}
    with pytest.raises(ValueError, match=r"same \(seed, draw\) grid"):
        tst.contrast(fx.simple_cell("bc", "random", left), fx.simple_cell("iql", "random", right),
                     definition="att_ours", tier="random")


def test_t_contrast_refuses_a_duplicate_episode() -> None:
    good = fx.simple_cell("bc", "random", {(s, d): 10.0 + d for s in (101, 202) for d in (1000, 1001)})
    doubled = tst.Cell(spec=good.spec, rows=good.rows + (dict(good.rows[0]),))
    with pytest.raises(ValueError, match="appears twice"):
        tst.contrast(doubled, good, definition="att_ours", tier="random")


def test_t_contrast_carries_d8_from_paired_comparison_and_it_agrees_with_this_files_own_counts() -> None:
    seeds = (101, 202, 303)
    left = {(s, d): 20.0 + ((s + 3 * d) % 7) / 4.0 for s in seeds for d in range(1000, 1009)}
    right = {(s, d): 20.5 + ((2 * s + d) % 5) / 4.0 for s in seeds for d in range(1000, 1009)}
    a, b = fx.simple_cell("bc", "maxpressure", left), fx.simple_cell("dt_nomix", "maxpressure", right)
    block = tst.contrast(a, b, definition="att_ours", tier="maxpressure")
    left_draw = {d: seq_mean(v) for d, v in by_draw(a.rows, "att_ours").items()}
    right_draw = {d: seq_mean(v) for d, v in by_draw(b.rows, "att_ours").items()}
    differences = [left_draw[d] - right_draw[d] for d in sorted(left_draw)]
    d8 = block["paired_comparison"]
    assert d8["wins"] == sum(1 for x in differences if x < 0)
    assert d8["losses"] == sum(1 for x in differences if x > 0)
    assert d8["ties"] == sum(1 for x in differences if x == 0)
    assert d8["median_difference"] == statistics.median(differences)
    nonzero = [x for x in differences if x != 0]
    magnitudes = sorted(abs(x) for x in nonzero)
    rank = {}
    for value in set(magnitudes):
        positions = [i + 1 for i, m in enumerate(magnitudes) if m == value]
        rank[value] = sum(positions) / len(positions)
    w_plus = sum(rank[abs(x)] for x in nonzero if x > 0)
    w_minus = sum(rank[abs(x)] for x in nonzero if x < 0)
    assert d8["rank_biserial"] == (w_plus - w_minus) / (w_plus + w_minus)
    assert d8["rank_biserial_defined"] is True
    assert set(d8["wilcoxon"]) == {"w_plus", "w_minus", "statistic", "n_used", "n_zero", "z", "p_value"}


def test_m20_a_one_ulp_perturbation_of_paired_comparisons_mean_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Amendment A, A2 Q8 and A4.4 (M20): the module refuses unless ``paired_comparison``'s mean and interval equal its
    own under ``==`` -- so a one-ulp disagreement, which an ``allclose`` would pass, is refused."""
    import dataclasses

    original = tst.paired_comparison

    def nudged(left: Any, right: Any) -> Any:
        result = original(left, right)
        return dataclasses.replace(result, mean_difference=result.mean_difference + math.ulp(result.mean_difference))

    monkeypatch.setattr(tst, "paired_comparison", nudged)
    seeds = (101, 202)
    a = fx.simple_cell("bc", "random", {(s, d): 30.0 + d % 4 + s / 100.0 for s in seeds for d in range(1000, 1004)})
    b = fx.simple_cell("iql", "random", {(s, d): 29.0 + d % 3 for s in seeds for d in range(1000, 1004)})
    with pytest.raises(ValueError, match="paired_comparison"):
        tst.contrast(a, b, definition="att_ours", tier="random")


# ----------------------------------------------------------------------
# T-outcome
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("low", "high", "negative", "positive"), [
    (-2.0, -1.0, "HELD", "FAILED"),
    (1.0, 2.0, "FAILED", "HELD"),
    (-1.0, 1.0, "NOT RESOLVED", "NOT RESOLVED"),
    (-1.0, 0.0, "NOT RESOLVED", "NOT RESOLVED"),
    (0.0, 1.0, "NOT RESOLVED", "NOT RESOLVED"),
    (0.0, 0.0, "NOT RESOLVED", "NOT RESOLVED"),
])
def test_t_outcome_the_interval_rule_for_both_predicted_signs(low: float, high: float, negative: str,
                                                             positive: str) -> None:
    assert tst.interval_outcome(low, high, predicted="negative") == negative == own_outcome(low, high, "negative")
    assert tst.interval_outcome(low, high, predicted="positive") == positive == own_outcome(low, high, "positive")


def test_t_outcome_refuses_an_unknown_predicted_sign() -> None:
    with pytest.raises(ValueError, match="predicted"):
        tst.interval_outcome(-1.0, 1.0, predicted="zero")


@pytest.mark.parametrize(("outcomes", "expected"), [
    (["HELD", "HELD", "HELD"], "HELD"),
    (["HELD", "NOT RESOLVED", "HELD"], "NOT RESOLVED"),
    (["HELD", "NOT RESOLVED", "FAILED"], "FAILED"),
    (["FAILED", "FAILED"], "FAILED"),
    (["NOT RESOLVED"], "NOT RESOLVED"),
])
def test_t_outcome_the_conjunction_rule(outcomes: list[str], expected: str) -> None:
    assert tst.conjunction(outcomes) == expected


def test_t_outcome_sign_of() -> None:
    assert [tst.sign_of(v) for v in (-0.5, 0.0, -0.0, 2.0)] == ["negative", "zero", "zero", "positive"]


# ----------------------------------------------------------------------
# T-ties
# ----------------------------------------------------------------------


TIED_LEVELS = {"dt_spatial": 207.0, "dt_nomix": 207.0, "bc": 207.25, "bc_top10": 207.5, "bc_top10_perix": 207.25,
               "iql": 204.0}


def test_t_ties_two_exact_ties_every_integer_and_its_range_by_enumeration() -> None:
    predicted = list(ts.predicted_order("fixedtime"))
    audit = tst.tie_audit(TIED_LEVELS, predicted)
    orders = consistent_orders(TIED_LEVELS)
    assert len(orders) == 4 == audit["orders_enumerated"]
    counts = [own_concordant(predicted, order, ts.METHODS) for order in orders]
    hard = [own_concordant(predicted, order, ts.HARD_SUBSET) for order in orders]
    assert audit["q2b"]["under_rule"] == own_concordant_rule(predicted, TIED_LEVELS, ts.METHODS) == min(counts)
    assert audit["q2b"]["range"] == [min(counts), max(counts)]
    assert audit["q2b_hard"]["under_rule"] == own_concordant_rule(predicted, TIED_LEVELS, ts.HARD_SUBSET) == min(hard)
    assert audit["q2b_hard"]["range"] == [min(hard), max(hard)]
    assert audit["first_place"]["possible"] == sorted({order[0] for order in orders})
    ranks = [order.index("dt_nomix") + 1 for order in orders]
    assert audit["rank"]["under_rule"] == 1 + sum(1 for a, v in TIED_LEVELS.items() if v < TIED_LEVELS["dt_nomix"])
    assert audit["rank"]["range"] == [min(ranks), max(ranks)]
    assert audit["rank"]["tied_with"] == ["dt_spatial"]
    assert [sorted(g) for g in audit["tied_groups"]] == [["dt_nomix", "dt_spatial"], ["bc", "bc_top10_perix"]]


def test_t_ties_a_tie_for_first_is_no_unique_first_place() -> None:
    levels = dict(TIED_LEVELS, iql=207.0)
    audit = tst.tie_audit(levels, list(ts.predicted_order("fixedtime")))
    assert audit["first_place"]["under_rule"] is None
    assert audit["first_place"]["possible"] == ["dt_nomix", "dt_spatial", "iql"]
    assert audit["orders_enumerated"] == len(consistent_orders(levels)) == 12


def test_t_ties_without_ties_every_range_is_a_point() -> None:
    levels = {"dt_spatial": 168.0, "dt_nomix": 167.25, "bc": 166.0, "bc_top10": 680.0, "bc_top10_perix": 210.25,
              "iql": 199.0}
    predicted = list(ts.predicted_order("maxpressure"))
    audit = tst.tie_audit(levels, predicted)
    order = sorted(levels, key=levels.get)
    assert audit["orders_enumerated"] == 1
    assert audit["q2b"]["range"] == [own_concordant(predicted, order, ts.METHODS)] * 2
    assert audit["first_place"]["under_rule"] == "bc"
    assert audit["rank"]["under_rule"] == 2 and audit["rank"]["range"] == [2, 2] and audit["rank"]["tied_with"] == []


def test_t_ties_the_level_groups_keep_exact_equality_only() -> None:
    levels = {"a": 1.0, "b": 1.0 + 2 ** -40, "c": 1.0, "d": 0.5}
    assert tst.level_groups(levels) == [["d"], ["a", "c"], ["b"]]


# ----------------------------------------------------------------------
# T-composer
# ----------------------------------------------------------------------


def _oos_episodes(cells: Mapping[str, tst.Cell], definition: str, *, iql: str = tst.ORIGINAL_IQL
                  ) -> dict[tuple[str, str], list[EpisodeResult]]:
    out = {}
    for method, tier in ts.OUT_OF_SAMPLE_CELLS:
        cell_id = iql if (method, tier) == ("iql", "random") else f"p5_2:{method}@{tier}"
        out[(method, tier)] = ic.episodes_from_rows(cells[cell_id].rows, value_key=definition)
    return out


def _tied_cells() -> dict[str, tst.Cell]:
    """The synthetic cells with two exact ties at ``random``: dt_spatial = dt_nomix, and bc = iql (a tie for first)."""
    cells = fx.synthetic_cells()

    def copy(source: str, target: str) -> None:
        spec = cells[target].spec
        cells[target] = tst.Cell(spec=spec, rows=tuple(dict(r, arm=spec.arm) for r in cells[source].rows))

    copy("p5_2:dt_nomix@random", "p5_2:dt_spatial@random")
    copy("p5_2:iql@random", "p5_2:bc@random")
    return cells


@pytest.mark.parametrize("with_hard_subset", [True, False])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("definition", ["att_ours", "att_engine"])
def test_t_composer_equals_random_tier_statements_field_by_field(with_hard_subset: bool, tied: bool,
                                                                 definition: str) -> None:
    """Amendment A, A2 Q2: at ``random`` the tier-generic composer returns every field P5.2b's composer returns, equal;
    it may add fields and never change one."""
    cells = _tied_cells() if tied else fx.synthetic_cells()
    episodes = _oos_episodes(cells, definition)
    theirs = ic.random_tier_statements(episodes, with_hard_subset=with_hard_subset)
    mine = tst.tier_composer(episodes, tier="random", with_hard_subset=with_hard_subset)
    assert contains(theirs, mine) == []
    if tied:
        assert theirs["q2a"]["tied_for_first"] == ["bc", "iql"]
        assert theirs["q2b"]["n_tied"] == 2


def test_t_composer_on_maxpressure_matches_this_files_own_ordering() -> None:
    cells = fx.synthetic_cells()
    episodes = _oos_episodes(cells, "att_ours")
    mine = tst.tier_composer(episodes, tier="maxpressure", with_hard_subset=True)
    levels = {m: own_level(cells[f"p5_2:{m}@maxpressure"], "att_ours") for m in ts.METHODS}
    order = sorted(ts.METHODS, key=levels.get)
    predicted = list(ts.predicted_order("maxpressure"))
    assert mine["ranking"]["order"] == order
    assert mine["q2a"]["measured_first"] == order[0] == "bc"
    assert mine["q2b"]["n_concordant"] == own_concordant_rule(predicted, levels, ts.METHODS)
    assert mine["q2b_hard"]["n_concordant"] == own_concordant_rule(predicted, levels, ts.HARD_SUBSET)
    assert mine["q3a"]["rank"] == 1 + sum(1 for m in ts.METHODS if levels[m] < levels["dt_nomix"])
    best = min(tst.NON_DT_METHODS, key=levels.get)
    assert mine["q3c"]["best_non_dt"] == best
    assert mine["q3c"]["mean"] == seq_contrast(cells["p5_2:dt_nomix@maxpressure"], cells[f"p5_2:{best}@maxpressure"],
                                               "att_ours")


# ----------------------------------------------------------------------
# T-q0
# ----------------------------------------------------------------------


#: ``spatial_mixing.per_seed_advantages``' keys: P5.1's per-seed schema, which every per-seed block carries.
P5_1_PER_SEED_KEYS = {"by_seed", "n_seeds", "mean", "between_seed_sd", "min", "max", "n_seeds_favouring_left",
                      "reverses_on_n_seeds", "reverses_against_pooled_sign", "definitions", "note"}


def test_t_q0_the_interaction_its_per_seed_values_and_the_three_verdicts(books: Any) -> None:
    """``BRIEF_45`` Amendment B, B2.12 and B2.2: I's sd and CI by ``statistics.stdev`` and ``1.96 * sd / sqrt(n)`` under
    ``==``, d4's interval the same way, and Q0a, Q0b and the stop rule derived from THIS test's intervals.  On these
    values (``att_ours``) the two routes agree exactly -- measured; were they ever to differ, this fails visibly.  I's
    per-seed block is P5.1's worker over composite rows, and its values are computed here by a loop: per seed, the mean
    over draws of ``dt_spatial_h4 - dt_nomix_h4`` minus that of ``dt_spatial - dt_nomix``."""
    book = books["corrected"]
    entries = {e["id"]: e for e in tst.score_q0(book, definition="att_ours")}
    s4, n4 = book[("dt_spatial_h4", "mappo1000")], book[("dt_nomix_h4", "mappo1000")]
    s1, n1 = book[("dt_spatial", "mappo1000")], book[("dt_nomix", "mappo1000")]
    d4 = [seq_mean(a) - seq_mean(b) for a, b in zip(by_draw(s4.rows, "att_ours").values(),
                                                     by_draw(n4.rows, "att_ours").values())]
    d1 = [seq_mean(a) - seq_mean(b) for a, b in zip(by_draw(s1.rows, "att_ours").values(),
                                                     by_draw(n1.rows, "att_ours").values())]
    interaction = [a - b for a, b in zip(d4, d1)]
    assert entries["Q0a"]["numbers"]["mean_difference"] == seq_mean(d4)
    assert entries["Q0.d1"]["numbers"]["mean_difference"] == seq_mean(d1)
    q0b = entries["Q0b"]["numbers"]
    assert q0b["mean_difference"] == seq_mean(interaction)
    assert (q0b["draws_negative"], q0b["draws_zero"], q0b["draws_positive"]) == (
        sum(x < 0 for x in interaction), sum(x == 0 for x in interaction), sum(x > 0 for x in interaction))
    mean_i, sd_i, low_i, high_i = own_interval(interaction)
    assert (q0b["mean_difference"], q0b["std"], q0b["ci95_half_width"], q0b["ci95_low"], q0b["ci95_high"]) == (
        mean_i, sd_i, 1.96 * sd_i / math.sqrt(len(interaction)), low_i, high_i)
    mean_4, sd_4, low4, high4 = own_interval(d4)
    q0a = entries["Q0a"]["numbers"]
    assert (q0a["mean_difference"], q0a["std"], q0a["ci95_low"], q0a["ci95_high"]) == (mean_4, sd_4, low4, high4)
    assert entries["Q0a"]["verdict"] == own_outcome(low4, high4, "positive")
    assert entries["Q0b"]["verdict"] == ("HELD" if low_i <= 0.0 <= high_i else "FAILED")
    assert entries["Q0.stop_rule"]["verdict"] == ("STOP" if high4 < 0.0 else "CONTINUE")
    ours = {name: by_seed(c.rows, "att_ours") for name, c in (("s4", s4), ("n4", n4), ("s1", s1), ("n1", n1))}
    per_seed = {s: seq_mean([a - b for a, b in zip(ours["s4"][s], ours["n4"][s])])
                - seq_mean([a - b for a, b in zip(ours["s1"][s], ours["n1"][s])]) for s in fx.SEEDS}
    block = q0b["per_seed"]
    assert set(block) == P5_1_PER_SEED_KEYS
    assert block["by_seed"] == {str(s): v for s, v in per_seed.items()}
    favouring = sum(1 for v in per_seed.values() if v < 0)
    assert block["n_seeds_favouring_left"] == favouring
    assert block["reverses_on_n_seeds"] == min(favouring, len(per_seed) - favouring)
    pooled = seq_mean(list(per_seed.values()))
    assert block["reverses_against_pooled_sign"] == sum(1 for v in per_seed.values()
                                                        if (v < 0) != (pooled < 0) and v != 0.0)
    assert q0b["rank_biserial"] is None and "difference of two differences" in q0b["rank_biserial_note"]
    assert entries["Q0.d1"]["verdict"] is None and entries["Q0.d1"]["kind"] == "quantity"


def test_t_q0_the_stop_rule_fires_on_a_d4_entirely_below_zero(books: Any) -> None:
    book = dict(books["corrected"])
    nomix = book[("dt_nomix_h4", "mappo1000")]
    spatial = book[("dt_spatial_h4", "mappo1000")]
    book[("dt_spatial_h4", "mappo1000")] = tst.Cell(
        spec=spatial.spec, rows=tuple(dict(r, arm=spatial.spec.arm, att_ours=r["att_ours"] - 5.0,
                                           att_engine=r["att_engine"] - 5.0) for r in nomix.rows))
    entries = {e["id"]: e for e in tst.score_q0(book, definition="att_ours")}
    assert entries["Q0.stop_rule"]["verdict"] == "STOP"
    assert entries["Q0a"]["verdict"] == "FAILED"


# ----------------------------------------------------------------------
# T-q4b
# ----------------------------------------------------------------------


def _with_d(book: Mapping[tuple[str, str], tst.Cell], tier: str, shift: float) -> dict[tuple[str, str], tst.Cell]:
    """A book whose ``dt_spatial`` at *tier* is ``dt_nomix`` + *shift* on every episode, so ``d(tier) = shift``."""
    out = dict(book)
    nomix, spatial = book[("dt_nomix", tier)], book[("dt_spatial", tier)]
    out[("dt_spatial", tier)] = tst.Cell(spec=spatial.spec, rows=tuple(
        dict(r, arm=spatial.spec.arm, att_ours=r["att_ours"] + shift, att_engine=r["att_engine"] + shift)
        for r in nomix.rows))
    return out


def test_t_q4b_the_four_rung_chain_and_the_three_rung_bullet_can_disagree(books: Any) -> None:
    book = books["corrected"]
    for tier, shift in (("mappo1000", 1.0), ("maxpressure", 2.0), ("fixedtime", 9.0), ("random", 3.0)):
        book = _with_d(book, tier, shift)
    entries = {e["id"]: e for e in tst.score_q4(book, definition="att_ours")}
    assert entries["Q4b.three_rung"]["verdict"] == "HELD"
    assert entries["Q4b"]["verdict"] == "FAILED"
    assert entries["Q4b"]["numbers"]["strictly_decreasing"] is False
    assert [r["tier"] for r in entries["Q4b"]["numbers"]["chain"]] == list(tst.TIERS)
    assert [r["tier"] for r in entries["Q4b.three_rung"]["numbers"]["chain"]] == ["mappo1000", "maxpressure", "random"]


def test_t_q4b_an_inverted_chain_is_named_and_a_non_monotone_one_is_not(books: Any) -> None:
    entries = {e["id"]: e for e in tst.score_q4(books["corrected"], definition="att_ours")}
    chain = [r["mean_difference"] for r in entries["Q4b"]["numbers"]["chain"]]
    assert all(a > b for a, b in zip(chain, chain[1:]))
    assert entries["Q4b"]["verdict"] == "FAILED" and "inverted" in entries["Q4b"]["reading"]
    book = books["corrected"]
    for tier, shift in (("mappo1000", 4.0), ("maxpressure", 1.0), ("fixedtime", 3.0), ("random", 2.0)):
        book = _with_d(book, tier, shift)
    mixed = {e["id"]: e for e in tst.score_q4(book, definition="att_ours")}
    assert mixed["Q4b"]["verdict"] == "FAILED" and "inverted" not in mixed["Q4b"]["reading"]


def test_t_q4a_the_exact_zero_at_fixedtime_is_not_resolved_and_the_aggregate_follows(books: Any) -> None:
    """``BRIEF_45`` Amendment B, B2.12: each tier's outcome from THIS test's interval, and the aggregate from those."""
    book = books["corrected"]
    entries = {(e["id"], e["tier"]): e for e in tst.score_q4(book, definition="att_ours")}
    fixedtime = entries[("Q4a", "fixedtime")]
    assert fixedtime["numbers"]["exactly_zero"] is True and fixedtime["verdict"] == "NOT RESOLVED"
    own = {t: own_contrast_outcome(book[("dt_spatial", t)], book[("dt_nomix", t)], "att_ours", "positive")
           for t in tst.OUT_OF_SAMPLE_TIERS}
    assert {t: entries[("Q4a", t)]["verdict"] for t in tst.OUT_OF_SAMPLE_TIERS} == own
    assert entries[("Q4a.aggregate", None)]["verdict"] == own_conjunction(list(own.values()))


# ----------------------------------------------------------------------
# T-q5
# ----------------------------------------------------------------------


def test_t_q5b_a_failed_mappo1000_tier_makes_the_aggregate_failed(books: Any) -> None:
    """M7: Q5b is registered on all FOUR tiers; dropping ``mappo1000`` would hide this tier's failure."""
    book = dict(books["corrected"])
    top10 = book[("bc_top10", "mappo1000")]
    perix = book[("bc_top10_perix", "mappo1000")]
    book[("bc_top10_perix", "mappo1000")] = tst.Cell(spec=perix.spec, rows=tuple(
        dict(r, arm=perix.spec.arm, att_ours=r["att_ours"] + 10.0, att_engine=r["att_engine"] + 10.0)
        for r in top10.rows))
    entries = {(e["id"], e["tier"]): e for e in tst.score_q5(book, definition="att_ours")}
    assert entries[("Q5b", "mappo1000")]["verdict"] == "FAILED"
    assert entries[("Q5b.aggregate", None)]["verdict"] == "FAILED"
    assert entries[("Q5b.aggregate", None)]["numbers"]["tiers"] == {
        t: entries[("Q5b", t)]["verdict"] for t in tst.TIERS}


def test_t_q5_each_tier_by_its_registered_sign(books: Any) -> None:
    book = books["corrected"]
    entries = {(e["id"], e["tier"]): e for e in tst.score_q5(book, definition="att_ours")}
    pattern = {"Q5a": (("bc_top10", "bc"), {t: "positive" for t in tst.OUT_OF_SAMPLE_TIERS}),
               "Q5b": (("bc_top10_perix", "bc_top10"), {t: "negative" for t in tst.TIERS}),
               "Q5c": (("bc_top10_perix", "bc"), {"mappo1000": "negative", "maxpressure": "negative",
                                                    "fixedtime": "positive", "random": "positive"})}
    for statement, ((left, right), signs) in pattern.items():
        for tier, sign in signs.items():
            differences = own_paired(book[(left, tier)], book[(right, tier)], "att_ours")
            exact = all(x == 0 for x in differences)
            entry = entries[(statement, tier)]
            assert entry["numbers"]["predicted_sign"] == sign
            if exact:
                assert entry["verdict"] == "NOT RESOLVED"
            else:
                _, _, low, high = own_interval(differences)
                assert entry["verdict"] == own_outcome(low, high, sign), (statement, tier)
        assert {t for (s, t) in entries if s == statement} == set(signs)


# ----------------------------------------------------------------------
# T-q6
# ----------------------------------------------------------------------


def test_t_q6_reads_the_shared_anchor_not_the_behaviour_cell(books: Any) -> None:
    book = books["corrected"]
    anchor = book[tst.ANCHOR]
    assert own_level(anchor, "att_ours") != own_level(book[("behaviour", "mappo1000")], "att_ours")
    entries = {(e["arm"], e["tier"]): e for e in tst.score_q6(book, definition="att_ours")}
    assert set(entries) == {(a, t) for a in tst.DT_ARMS for t in tst.TIERS}
    for (arm, tier), entry in entries.items():
        assert entry["numbers"]["mean_difference"] == seq_contrast(book[(arm, tier)], anchor, "att_ours")
        assert entry["numbers"]["mean_difference"] != seq_contrast(book[(arm, tier)], book[("behaviour", "mappo1000")],
                                                                   "att_ours")
        assert entry["inputs"][1] == tst.ANCHOR_CELL
        _, _, low, _ = own_interval(own_paired(book[(arm, tier)], anchor, "att_ours"))  # Amendment B, B2.12
        assert entry["verdict"] == ("COLLAPSED" if low > 0.0 else "NOT COLLAPSED")
        assert entry["numbers"]["p5_1_outcome"] in ("COLLAPSED", "NOT COLLAPSED", "NOT RESOLVED")
        assert entry["numbers"]["seen"] is (tier == "mappo1000")


def test_t_q6_refuses_when_collapse_criterion_disagrees_with_the_registered_route(books: Any,
                                                                                  monkeypatch: pytest.MonkeyPatch
                                                                                  ) -> None:
    """Amendment A, A2 Q7 (F9): P5.1's three-valued outcome comes from ``collapse_criterion`` imported unchanged, and the
    module refuses unless its interval equals the module's own under ``==``."""
    original = tst.collapse_criterion

    def nudged(arm: Any, reference: Any) -> dict[str, Any]:
        block = original(arm, reference)
        return dict(block, ci95_low=block["ci95_low"] + math.ulp(block["ci95_low"]))

    monkeypatch.setattr(tst, "collapse_criterion", nudged)
    with pytest.raises(ValueError, match="collapse_criterion"):
        tst.score_q6(books["corrected"], definition="att_ours")


def test_t_q6_the_anchor_identity_is_refused_when_one_episode_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    import offline.dt_gate as dt_gate

    monkeypatch.setattr(dt_gate, "HELD_OUT_DRAWS", fx.DRAWS)
    monkeypatch.setattr(dt_gate, "TRAINING_SEEDS", fx.SEEDS)
    cells = fx.synthetic_cells()
    p5_1, p5_2 = cells[tst.ANCHOR_CELL], cells[tst.REROLLED_ANCHOR_CELL]

    def payload(c: tst.Cell) -> dict[str, Any]:
        return {"episodes": [{"seed": r["seed"], "draw_id": r["draw_id"], "att_horizon": r["att_ours"],
                              "horizon_vehicle_count": r["horizon_vehicle_count"],
                              "episode_reward": r["episode_reward"]} for r in c.rows]}

    tst.assert_anchor_identity(payload(p5_1), payload(p5_2), p5_1.rows, p5_2.rows)
    altered = payload(p5_2)
    altered["episodes"][3] = dict(altered["episodes"][3], att_horizon=altered["episodes"][3]["att_horizon"] + 0.25)
    with pytest.raises(ValueError, match="differ"):
        tst.assert_anchor_identity(payload(p5_1), altered, p5_1.rows, p5_2.rows)
    engine_rows = [dict(r) for r in p5_2.rows]
    engine_rows[5]["att_engine"] += 0.5
    with pytest.raises(ValueError, match="att_engine"):
        tst.assert_anchor_identity(payload(p5_1), payload(p5_2), p5_1.rows, engine_rows)


# ----------------------------------------------------------------------
# T-versions and T-definitions
# ----------------------------------------------------------------------


def test_t_versions_the_iql_free_statements_are_stored_once_and_the_iql_ones_twice(books: Any) -> None:
    statements = tst.all_statements(books)
    for s in statements:
        involves_corrected = tst.CORRECTED_IQL in s["inputs"] or tst.ORIGINAL_IQL in s["inputs"]
        if s["version"] == "both":
            assert not involves_corrected, (s["id"], s["tier"])
        else:
            assert s["version"] in tst.VERSIONS
    q1 = {s["version"]: s for s in statements if s["id"] == "Q1" and s["definition"] == "att_ours"}
    assert set(q1) == {"corrected", "as_committed"}
    iql = {v: next(c for c in q1[v]["numbers"]["cells"] if c["cell"] == ["iql", "random"])["measured"] for v in q1}
    corrected = fx.synthetic_cells()[tst.CORRECTED_IQL]
    original = fx.synthetic_cells()[tst.ORIGINAL_IQL]
    # Amendment B, B2.12: the levels by THIS file's mean (statistics.mean), not the module's route.
    assert iql["corrected"] == own_level(corrected, "att_ours")
    assert iql["as_committed"] == own_level(original, "att_ours")
    assert iql["corrected"] != iql["as_committed"]
    for statement_id in ("Q2a", "Q2b", "D9", "Q3a", "Q3c", "ranking"):
        versions = {s["version"] for s in statements if s["id"] == statement_id and s["tier"] == "random"}
        assert versions == {"corrected", "as_committed"}, statement_id
    hard = [s for s in statements if s["id"] == "Q2b_hard" and s["tier"] == "random"]
    assert {s["version"] for s in hard} == {"both"}


def test_t_versions_an_iql_free_statement_that_differs_between_versions_is_refused(books: Any) -> None:
    """M12: the identity of every IQL-free statement across versions is asserted, not assumed."""
    altered = {version: dict(book) for version, book in books.items()}
    bc = altered["corrected"][("bc", "maxpressure")]
    altered["corrected"][("bc", "maxpressure")] = tst.Cell(spec=bc.spec, rows=tuple(
        dict(r, att_ours=r["att_ours"] + 1.0) for r in bc.rows))
    with pytest.raises(ValueError, match="differs between versions"):
        tst.all_statements(altered)


def test_t_versions_the_books_differ_only_at_the_random_tier_iql_cell() -> None:
    """M3: the corrected book reads P5.2b's cell, the as-committed one P5.2's; nothing else differs."""
    books = tst.make_books(fx.synthetic_cells())
    assert books["corrected"][("iql", "random")].spec.cell_id == tst.CORRECTED_IQL
    assert books["as_committed"][("iql", "random")].spec.cell_id == tst.ORIGINAL_IQL
    differing = [k for k in books["corrected"] if books["corrected"][k] is not books["as_committed"][k]]
    assert differing == [("iql", "random")]
    assert books["corrected"][("behaviour", "random")].spec.cell_id == tst.ANCHOR_CELL
    assert books["corrected"][tst.ANCHOR].spec.cell_id == tst.ANCHOR_CELL


def _engine_straddles_at_maxpressure(books: Mapping[str, Mapping[tuple[str, str], tst.Cell]]
                                     ) -> dict[str, dict[tuple[str, str], tst.Cell]]:
    """``dt_spatial@maxpressure``'s ``att_engine`` made ``dt_nomix``'s plus an alternating term: ``d`` straddles zero
    under ``att_engine`` while ``att_ours`` is unchanged."""
    out = {version: dict(book) for version, book in books.items()}
    for book in out.values():
        nomix, spatial = book[("dt_nomix", "maxpressure")], book[("dt_spatial", "maxpressure")]
        engine = {(r["seed"], r["draw_id"]): r["att_engine"] for r in nomix.rows}
        book[("dt_spatial", "maxpressure")] = tst.Cell(spec=spatial.spec, rows=tuple(
            dict(r, att_engine=engine[(r["seed"], r["draw_id"])] + (0.5 if r["draw_id"] % 2 else -0.5))
            for r in spatial.rows))
    return out


def test_t_definitions_definition_dependent_is_true_iff_the_two_verdicts_differ(books: Any) -> None:
    statements = tst.all_statements(_engine_straddles_at_maxpressure(books))
    ours = pick(statements, "Q4a", tier="maxpressure", definition="att_ours")
    engine = pick(statements, "Q4a", tier="maxpressure", definition="att_engine")
    assert ours["verdict"] == "HELD" and engine["verdict"] == "NOT RESOLVED"
    assert ours["definition_dependent"] is True and engine["definition_dependent"] is True
    assert engine["primary"] is True and ours["primary"] is False
    for s in statements:
        if s["kind"] == "verdict":
            other = pick(statements, s["id"], tier=s["tier"], arm=s.get("arm"), version=s["version"]
                         if s["version"] != "both" else "corrected",
                         definition="att_engine" if s["definition"] == "att_ours" else "att_ours")
            assert s["definition_dependent"] is (s["verdict"] != other["verdict"]), (s["id"], s["tier"])
            assert s["definition_dependent_on"] == "verdict"


def test_t_definitions_q3c_and_the_leaders_are_compared_on_their_reading(books: Any) -> None:
    statements = tst.all_statements(books)
    for s in statements:
        if s["id"] == "Q3c":
            assert s["kind"] in ("reading", "context") and s["verdict"] is None
            assert s["definition_dependent_on"] == "reading"
            other = pick(statements, "Q3c", tier=s["tier"], version=s["version"] if s["version"] != "both"
                         else "corrected", definition="att_engine" if s["definition"] == "att_ours" else "att_ours")
            assert s["definition_dependent"] is (s["reading"] != other["reading"])
    for block in tst.leaders_block(books):
        assert block["definition_dependent_on"] == "reading"


# ----------------------------------------------------------------------
# The statements on the synthetic book: every id, tier and definition present
# ----------------------------------------------------------------------


def test_every_registered_statement_is_present_under_both_definitions(books: Any) -> None:
    statements = tst.all_statements(books)
    keys = {(s["id"], s["tier"], s["definition"]) for s in statements}
    expected = set()
    for definition in tst.DEFINITIONS:
        expected |= {("Q0.d1", "mappo1000", definition), ("Q0a", "mappo1000", definition),
                     ("Q0b", "mappo1000", definition), ("Q0.stop_rule", "mappo1000", definition),
                     ("Q1", None, definition), ("Q1b.both", None, definition), ("Q2b.aggregate", None, definition),
                     ("Q3b.aggregate", None, definition), ("Q4a.aggregate", None, definition),
                     ("Q4b", None, definition), ("Q4b.three_rung", None, definition),
                     ("Q5a.aggregate", None, definition), ("Q5b.aggregate", None, definition),
                     ("Q5c.aggregate", None, definition)}
        expected |= {("Q1b", t, definition) for t in ("maxpressure", "fixedtime")}
        for tier in tst.OUT_OF_SAMPLE_TIERS:
            expected |= {(i, tier, definition) for i in ("ranking", "Q2a", "Q2b", "Q2b_hard", "D9", "Q3a", "Q3b",
                                                          "Q3c", "Q4a", "Q5a")}
        expected |= {(i, "mappo1000", definition) for i in ("Q3a", "Q3b", "Q3c")}
        expected |= {(i, t, definition) for i in ("Q4.d", "Q5b", "Q5c", "Q6") for t in tst.TIERS}
    assert keys == expected
    for s in statements:
        assert s["rule"], (s["id"], s["tier"])
        for quote in s["rule"]:
            assert quote["file"] in tst.CITABLE_FILES
    context = [s for s in statements if s["tier"] == "mappo1000" and s["id"] in ("Q3a", "Q3b", "Q3c")]
    assert context and all(s["kind"] == "context" and s["verdict"] is None for s in context)


def test_q1b_both_cells_and_its_two_sources(books: Any) -> None:
    entries = {(e["id"], e["tier"]): e for e in tst.score_q1(books["corrected"], definition="att_ours")}
    for tier in ("maxpressure", "fixedtime"):
        level = own_level(books["corrected"][("behaviour", tier)], "att_ours")
        predicted = ts.PREDICTED_LEVELS["behaviour"][tier]
        held = abs(level - predicted) / predicted <= ts.ANCHOR_BAND
        assert entries[("Q1b", tier)]["verdict"] == ("HELD" if held else "FAILED")
    both = entries[("Q1b.both", None)]
    assert both["verdict"] == ("HELD" if all(entries[("Q1b", t)]["verdict"] == "HELD"
                                             for t in ("maxpressure", "fixedtime")) else "FAILED")
    assert "both cells hold" in both["reading"]
    assert {q["lines"][0] for q in both["rule"]} >= {576, 497}


def test_q3b_reads_the_shared_anchor_as_behaviour_at_random_and_reports_the_sub_prediction(books: Any) -> None:
    """``BRIEF_45`` Amendment B, B2.1: ``matched`` stays as the registered rule gives it; the block gains
    ``exactly_zero`` and a reading, which at ``fixedtime`` says the match comes from identical episodes."""
    book = books["corrected"]
    entries = {(e["id"], e["tier"]): e for e in tst.score_q3b(book, definition="att_ours")}
    assert entries[("Q3b", "random")]["inputs"][1] == tst.ANCHOR_CELL
    for tier, predicted_to_resolve in (("maxpressure", False), ("fixedtime", False), ("random", True)):
        differences = own_paired(book[("dt_nomix", tier)], book[("behaviour", tier)], "att_ours")
        _, _, low, high = own_interval(differences)
        resolved = low > 0.0 or high < 0.0
        sub = entries[("Q3b", tier)]["numbers"]["sub_prediction"]
        assert {k: sub[k] for k in ("predicted_to_resolve", "resolved", "matched", "exactly_zero")} == {
            "predicted_to_resolve": predicted_to_resolve, "resolved": resolved,
            "matched": resolved == predicted_to_resolve, "exactly_zero": all(x == 0.0 for x in differences)}
        assert sub["reading"].startswith("matched" if resolved == predicted_to_resolve else "not matched")
    fixedtime = entries[("Q3b", "fixedtime")]
    assert fixedtime["numbers"]["exactly_zero"] is True and fixedtime["verdict"] == "NOT RESOLVED"
    assert fixedtime["numbers"]["sub_prediction"]["exactly_zero"] is True
    assert "the match comes from identical episodes" in fixedtime["numbers"]["sub_prediction"]["reading"]
    for tier in ("maxpressure", "random"):
        assert "identical episodes" not in entries[("Q3b", tier)]["numbers"]["sub_prediction"]["reading"]


# ----------------------------------------------------------------------
# Leaders, ties and the collapse audit (descriptive)
# ----------------------------------------------------------------------


def test_the_leaders_block_on_every_tier_with_ties_grouped(books: Any) -> None:
    blocks = tst.leaders_block(books)
    keys = {(b["tier"], b["definition"], b["version"]) for b in blocks}
    for definition in tst.DEFINITIONS:
        for tier in ("mappo1000", "maxpressure", "fixedtime"):
            assert (tier, definition, "both") in keys
        assert {("random", definition, v) for v in tst.VERSIONS} <= keys
    fixedtime = next(b for b in blocks if b["tier"] == "fixedtime" and b["definition"] == "att_ours")
    assert fixedtime["leaders"] == ["iql"]
    assert sorted(fixedtime["runners_up"]) == ["dt_nomix", "dt_spatial"]
    assert any(len(group) == 2 for group in fixedtime["order"])
    maxpressure = next(b for b in blocks if b["tier"] == "maxpressure" and b["definition"] == "att_ours")
    assert maxpressure["leaders"] == ["bc"] and maxpressure["runners_up"] == ["dt_nomix"]
    lead = maxpressure["leader_minus_runner_up"][0]
    assert lead["mean_difference"] == seq_contrast(books["corrected"][("bc", "maxpressure")],
                                                   books["corrected"][("dt_nomix", "maxpressure")], "att_ours")


def test_the_ties_block_lists_the_identities_and_every_tie_dependent_integer(books: Any) -> None:
    blocks = tst.ties_block(books)
    fixedtime = next(b for b in blocks if b["tier"] == "fixedtime" and b["definition"] == "att_ours")
    groups = sorted(sorted(g) for g in fixedtime["equal_level_groups"])
    assert groups == [["bc", "bc_top10_perix"], ["behaviour", "dt_nomix", "dt_spatial"]]
    identical = {tuple(sorted(p["pair"])) for p in fixedtime["identical_episode_pairs"]}
    assert identical == {("bc", "bc_top10_perix"), ("behaviour", "dt_nomix"), ("behaviour", "dt_spatial"),
                         ("dt_nomix", "dt_spatial")}
    names = {i["name"] for i in fixedtime["integers"]}
    assert {"Q2b", "Q2b_hard", "Q3a", "first_place"} <= names
    q2b = next(i for i in fixedtime["integers"] if i["name"] == "Q2b" and i["scope"] == "pooled")
    levels = {m: own_level(books["corrected"][(m, "fixedtime")], "att_ours") for m in ts.METHODS}
    predicted = list(ts.predicted_order("fixedtime"))
    counts = [own_concordant(predicted, order, ts.METHODS) for order in consistent_orders(levels)]
    assert q2b["under_rule"] == min(counts) and q2b["range"] == [min(counts), max(counts)]
    assert {i["scope"] for i in fixedtime["integers"]} >= {"pooled", *(f"seed {s}" for s in fx.SEEDS)}


def own_collapses(book: Mapping[tuple[str, str], tst.Cell], arm: str, tier: str, definition: str) -> bool:
    """Q6's rule on THIS file's interval: COLLAPSED iff the arm is worse than the shared anchor with the CI excluding 0."""
    _, _, low, _ = own_interval(own_paired(book[(arm, tier)], book[tst.ANCHOR], definition))
    return low > 0.0


def own_sentence_holds(books: Mapping[str, Mapping[tuple[str, str], tst.Cell]], definition: str, version: str) -> bool:
    """The README sentence in its own words: no DT arm collapses on any tier, and every non-DT arm on at least one."""
    book, tiers = books[version], ("mappo1000", "maxpressure", "fixedtime", "random")
    dt_never = not any(own_collapses(book, arm, t, definition) for arm in ("dt_spatial", "dt_nomix") for t in tiers)
    every_other = all(any(own_collapses(book, arm, t, definition) for t in tiers)
                      for arm in ("bc", "bc_top10", "bc_top10_perix", "iql"))
    return dt_never and every_other


def test_the_descriptive_collapse_audit_applies_q6s_rule_to_every_non_dt_arm(books: Any) -> None:
    audit = tst.collapse_audit(books)
    assert audit["registered"] is False
    assert audit["readme_sentence"] == tst.README_COLLAPSE_SENTENCE
    rows = audit["rows"]
    keys = {(r["arm"], r["tier"], r["definition"], r["version"]) for r in rows}
    for definition in tst.DEFINITIONS:
        for arm in tst.NON_DT_METHODS:
            for tier in tst.TIERS:
                if (arm, tier) == ("iql", "random"):
                    assert {(arm, tier, definition, v) for v in tst.VERSIONS} <= keys
                else:
                    assert (arm, tier, definition, "both") in keys
    for r in rows:
        book = books["corrected" if r["version"] == "both" else r["version"]]
        differences = own_paired(book[(r["arm"], r["tier"])], book[tst.ANCHOR], r["definition"])
        _, _, low, _ = own_interval(differences) if any(differences) else (0, 0, 0.0, 0)
        assert r["verdict"] == ("COLLAPSED" if low > 0 else "NOT COLLAPSED")
    for verdict in audit["sentence_holds"]:
        assert verdict["holds"] is own_sentence_holds(books, verdict["definition"], verdict["version"])


# ----------------------------------------------------------------------
# The cells block
# ----------------------------------------------------------------------


def test_the_cells_block_carries_every_quantity_a11b_and_d15_require() -> None:
    cells = fx.synthetic_cells()
    block = tst.cells_block(cells)
    assert set(block) == {spec.cell_id for spec in tst.CELL_SPECS}
    for cell_id, entry in block.items():
        c = cells[cell_id]
        for definition in tst.DEFINITIONS:
            values = [float(r[definition]) for r in c.rows]
            assert entry[definition]["n"] == len(values)
            assert entry[definition]["median"] == float(np.percentile(np.asarray(values), 50, method="linear"))
            q25, q75 = np.percentile(np.asarray(values), [25, 75], method="linear")
            assert entry[definition]["iqr"] == [float(q25), float(q75)]
        for name in ("entered", "created", "never_entered"):
            assert entry["companions"][f"{name}_mean"] == statistics.mean(int(r[name]) for r in c.rows)
        assert entry["companions"]["episodes_with_never_entered"] == sum(1 for r in c.rows if r["never_entered"] > 0)
        assert entry["horizon_vehicle_count_mean"] == statistics.mean(r["horizon_vehicle_count"] for r in c.rows)
        assert entry["method"] == c.spec.method and entry["tier"] == c.spec.tier and entry["source"] == c.spec.source


# ----------------------------------------------------------------------
# T-inputs (a synthetic output/ and repository)
# ----------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> fx.SyntheticTree:
    synthetic = fx.write_synthetic_tree(tmp_path)
    fx.install_protocol(monkeypatch, synthetic)
    return synthetic


def test_t_inputs_the_synthetic_tree_loads_and_its_cells_are_the_written_ones(tree: fx.SyntheticTree) -> None:
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    assert set(inputs.cells) == {spec.cell_id for spec in tst.CELL_SPECS}
    for cell_id, c in inputs.cells.items():
        assert [dict(r) for r in c.rows] == [dict(r) for r in tree.cells[cell_id].rows], cell_id
    provenance = inputs.provenance
    output = tree.roots.output_root
    from offline.att_rederivation import CellKey, cell_file_name

    pairs = []
    for spec in tst.CELL_SPECS:
        if spec.source == "p5_2b":
            continue
        for r in tree.cells[spec.cell_id].rows:
            relative = f"{tst.P8_4B_DIR}/" + cell_file_name(CellKey(scenario="grid4x4", arm=spec.arm, seed=r["seed"],
                                                                    draw_id=r["draw_id"]))
            pairs.append((relative, fx.sha256_file(output / relative)))
    text = "".join(f"{digest}  {path}\n" for path, digest in sorted(pairs))
    assert provenance["p8_4b_rows"]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert provenance["p8_4b_rows"]["n"] == len(pairs) == 31 * len(tree.seeds) * len(tree.draws)
    corrected = sorted(p.relative_to(output).as_posix()
                       for p in (output / tst.P5_2B_REDERIVATION_DIR).glob("cell_*.json"))
    text = "".join(f"{fx.sha256_file(output / p)}  {p}\n" for p in corrected)
    assert provenance["p5_2b_corrected_rows"]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rewrite(path: Path, edit: Any) -> None:
    payload = json.loads(path.read_bytes())
    edit(payload)
    fx.write_json(path, payload)


def _resum(tree: fx.SyntheticTree, name: str, directory: str, field: str) -> fx.SyntheticTree:
    """Rewrite a manifest after an edit and re-pin it, so only the check under test can fail."""
    import dataclasses

    fx.write_sums(tree.roots.output_root, name, directory)
    pins = dataclasses.replace(tree.pins, **{field: fx.sha256_file(tree.roots.output_root / name)})
    return dataclasses.replace(tree, pins=pins)


def test_t_inputs_refuses_a_manifest_off_its_pin(tree: fx.SyntheticTree) -> None:
    sums = tree.roots.output_root / tst.P5_1_SUMS_NAME
    sums.write_text(sums.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the pinned"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_an_eval_file_off_its_manifest_line(tree: fx.SyntheticTree) -> None:
    path = tree.roots.output_root / "p5_1" / "eval_bc.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=r"does not match its line in SHA256SUMS_p5_1\.txt"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_an_incomplete_cell(tree: fx.SyntheticTree) -> None:
    _rewrite(tree.roots.output_root / "p5_2" / "eval_maxpressure_bc.json",
             lambda p: p.__setitem__("episodes", p["episodes"][:-1]))
    tree = _resum(tree, tst.P5_2_SUMS_NAME, "p5_2", "p5_2_sums_sha256")
    with pytest.raises(ValueError, match="not a complete cell"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_a_p8_4b_row_that_does_not_reproduce_its_eval_file(tree: fx.SyntheticTree) -> None:
    """M15: every P8.4b row passes ``iql_correction._p8_4b_rows``."""
    from offline.att_rederivation import CellKey, cell_file_name

    name = cell_file_name(CellKey(scenario="grid4x4", arm="bc@fixedtime", seed=202, draw_id=1001))
    _rewrite(tree.roots.output_root / tst.P8_4B_DIR / name, lambda p: p.__setitem__("att_ours", p["att_ours"] + 1.0))
    with pytest.raises(ValueError, match="does not reproduce"):
        tst.load_inputs(tree.roots, pins=tree.pins)


@pytest.mark.parametrize("arm", ["bc@maxpressure", "dt_nomix@grid4x4_mappo1000", "iql@random"])
def test_t_inputs_refuses_an_altered_att_engine_by_the_notes_means(tree: fx.SyntheticTree, arm: str) -> None:
    """F13 / M16: P8.4b has no per-file manifest, so an edited ``att_engine`` passes every other check; the README note's
    means are the anchor on EVERY tier."""
    from offline.att_rederivation import CellKey, cell_file_name

    name = cell_file_name(CellKey(scenario="grid4x4", arm=arm, seed=101, draw_id=1002))
    _rewrite(tree.roots.output_root / tst.P8_4B_DIR / name,
             lambda p: p.__setitem__("att_engine", p["att_engine"] + 0.5))
    with pytest.raises(ValueError, match="does not hold"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_a_corrected_row_off_p5_2bs_committed_cells_block(tree: fx.SyntheticTree) -> None:
    from offline.att_rederivation import CellKey, cell_file_name

    name = cell_file_name(CellKey(scenario="grid4x4", arm="iql@random", seed=303, draw_id=1000))
    _rewrite(tree.roots.output_root / tst.P5_2B_REDERIVATION_DIR / name,
             lambda p: p.__setitem__("att_engine", p["att_engine"] + 0.25))
    tree = _resum(tree, tst.P5_2B_SUMS_NAME, tst.P5_2B_RUN_DIR, "p5_2b_sums_sha256")
    with pytest.raises(ValueError, match="cells.corrected"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_a_rerolled_anchor_that_differs_from_p5_1s(tree: fx.SyntheticTree) -> None:
    """M14: D12's one anchor is asserted, not assumed."""
    def edit(payload: dict[str, Any]) -> None:
        payload["episodes"][2]["episode_reward"] -= 1.0  # a field only tier_sweep.assert_cells_identical compares

    _rewrite(tree.roots.output_root / "p5_2" / "eval_mappo1000_random.json", edit)
    tree = _resum(tree, tst.P5_2_SUMS_NAME, "p5_2", "p5_2_sums_sha256")
    with pytest.raises(ValueError, match="differ"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_d17_a_deterministic_mappo1000_dt_run_is_refused(tree: fx.SyntheticTree) -> None:
    """``BRIEF_45`` section 3.2, Q0: E1 measured zero, so D17's deterministic 1-head pair was never trained -- verified on
    disk, so ``d1`` is P5.1's default-regime pair."""
    _rewrite(tree.roots.output_root / "p5_2" / "training_mappo1000_dt_nomix.json",
             lambda p: p["runs"][0].__setitem__("deterministic", True))
    tree = _resum(tree, tst.P5_2_SUMS_NAME, "p5_2", "p5_2_sums_sha256")
    with pytest.raises(ValueError, match="D17"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_d17_a_complete_one_head_mappo1000_cell_in_p5_2_is_refused(tree: fx.SyntheticTree) -> None:
    import shutil

    output = tree.roots.output_root
    shutil.copy(output / "p5_1" / "eval_dt_spatial.json", output / "p5_2" / "eval_mappo1000_dt_spatial.json")
    tree = _resum(tree, tst.P5_2_SUMS_NAME, "p5_2", "p5_2_sums_sha256")
    with pytest.raises(ValueError, match="D17"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_an_altered_note(tree: fx.SyntheticTree) -> None:
    note = tree.roots.repo_root / tst.C1_NOTE
    note.write_bytes(note.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not the pinned"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_refuses_a_p5_2b_artifact_off_its_pin(tree: fx.SyntheticTree) -> None:
    artifact = tree.roots.repo_root / tst.P5_2B_ARTIFACT
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not the pinned"):
        tst.load_inputs(tree.roots, pins=tree.pins)


def test_t_inputs_the_local_manifest_reader_equals_read_p5_2_json_and_refuses_the_same_tampering(
        tree: fx.SyntheticTree) -> None:
    """Amendment A, A2 Q3: the restated reader returns ``read_p5_2_json``'s payload and refuses what it refuses."""
    output = tree.roots.output_root
    sums = tst.read_manifest(output, tst.P5_2_SUMS_NAME, tree.pins.p5_2_sums_sha256)
    assert sums == ts.parse_sha256sums((output / tst.P5_2_SUMS_NAME).read_text(encoding="utf-8"))
    relative = "p5_2/eval_random_bc.json"
    assert tst.read_listed_json(output, relative, sums, manifest_name=tst.P5_2_SUMS_NAME) == ic.read_p5_2_json(
        output, relative, sums)
    with pytest.raises(ValueError, match="has no line"):
        tst.read_listed_json(output, "p5_2/not_listed.json", sums, manifest_name=tst.P5_2_SUMS_NAME)
    with pytest.raises(ValueError, match="has no line"):
        ic.read_p5_2_json(output, "p5_2/not_listed.json", sums)
    (output / relative).write_bytes((output / relative).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="does not match its line"):
        tst.read_listed_json(output, relative, sums, manifest_name=tst.P5_2_SUMS_NAME)
    with pytest.raises(ValueError, match="does not match its line"):
        ic.read_p5_2_json(output, relative, sums)


def test_t_inputs_the_restated_note_check_equals_the_original_at_random(tree: fx.SyntheticTree) -> None:
    output = tree.roots.output_root
    campaign = ic.assert_p8_4b_campaign(output, pins=ic.Pins(
        p5_2_sums_sha256=tree.pins.p5_2_sums_sha256, declaration_sha256="unused", c1_note_sha256=tree.pins.c1_note_sha256,
        p8_4b_declared_cells_sha256=tree.pins.p8_4b_declared_cells_sha256, corpus_manifest_sha256="unused"))
    note = json.loads((tree.roots.repo_root / tst.C1_NOTE).read_bytes())
    arms = {m: f"{m}@random" for m in ts.METHODS}
    assert tst.assert_note_means(output, campaign["cells"], note, tier="random", arms=arms) == ic.assert_c1_note_means(
        output, campaign["cells"], note)
    note["tiers"]["grid4x4/random"]["att_engine"]["means"]["bc"] += 1.0
    with pytest.raises(ValueError, match="does not hold"):
        ic.assert_c1_note_means(output, campaign["cells"], note)
    with pytest.raises(ValueError, match="does not hold"):
        tst.assert_note_means(output, campaign["cells"], note, tier="random", arms=arms)


def test_t_inputs_the_synthetic_build_passes_every_comparison_of_the_record(tree: fx.SyntheticTree) -> None:
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    artifact = tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())
    comparison = artifact["comparison_with_the_record"]
    assert comparison["p5_2b_artifact"]["equal"] is True
    assert comparison["p5_1_grid"]["equal"] is True
    for row in comparison["p8_4b"]:
        assert row["record_route_reproduces"] is True
    assert comparison["p5_2_stop_rule_record"]["record_route_reproduces"] is True
    assert len(comparison["q5"]) == 3 and all(r["statement"] == tst.Q5_STATEMENT for r in comparison["q5"])
    assert {(r["statement"], r["tier"]) for r in comparison["never_scored"]} == set(tst.NEVER_SCORED)


def test_a_record_row_neither_equal_nor_explained_refuses_the_build(tree: fx.SyntheticTree) -> None:
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    statements = tst.all_statements(tst.make_books(inputs.cells))
    q1 = pick(statements, "Q1", tier=None, definition="att_ours", version="as_committed")
    good = f"{q1['verdict']}, {q1['numbers']['n_held']} of {q1['numbers']['n_cells']}"
    record = (tst.RecordRow("docs/returns/P5.2.md", (25, 25), good, "q1.outcome"),)
    artifact = tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=record)
    assert artifact["comparison_with_the_record"]["literal_records"][0]["equal"] is True
    wrong = (tst.RecordRow("docs/returns/P5.2.md", (25, 25), "HELD, 99 of 19", "q1.outcome"),)
    with pytest.raises(ValueError, match="neither equal nor explained"):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=wrong)


def _repin_document(tree: fx.SyntheticTree, relative: str, field: str, edit: Any) -> fx.SyntheticTree:
    """Edit a committed synthetic document and re-pin it, so only the comparison under test can refuse."""
    import dataclasses

    _rewrite(tree.roots.repo_root / relative, edit)
    pins = dataclasses.replace(tree.pins, **{field: fx.sha256_file(tree.roots.repo_root / relative)})
    return dataclasses.replace(tree, pins=pins)


def test_the_build_refuses_a_p5_2b_statement_it_does_not_reproduce_field_by_field(tree: fx.SyntheticTree) -> None:
    def edit(payload: dict[str, Any]) -> None:
        payload["statements"]["att_ours"]["before"]["q2b"]["n_concordant"] += 1

    tree = _repin_document(tree, tst.P5_2B_ARTIFACT, "p5_2b_artifact_sha256", edit)
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    with pytest.raises(ValueError, match="not reproduced field by field"):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())


def test_the_build_refuses_a_p8_4b_record_its_own_route_does_not_reproduce(tree: fx.SyntheticTree) -> None:
    def edit(payload: dict[str, Any]) -> None:
        block = payload["contrasts"][0]["by_definition"]["att_engine"]
        block["mean_difference"] = block["mean_difference"] + 2 ** -30

    tree = _repin_document(tree, tst.P8_4B_ARTIFACT, "p8_4b_artifact_sha256", edit)
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    with pytest.raises(ValueError, match="does not reproduce V4"):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())


def test_the_build_refuses_a_p5_1_record_it_does_not_reproduce_under_equality(tree: fx.SyntheticTree) -> None:
    def edit(payload: dict[str, Any]) -> None:
        payload["predictions"]["P3"]["dt_nomix"]["ci95_high"] += 2 ** -30

    tree = _repin_document(tree, tst.P5_1_GRID, "p5_1_grid_sha256", edit)
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    with pytest.raises(ValueError, match="are not reproduced under =="):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())


def test_the_build_refuses_a_stop_rule_record_its_own_route_does_not_reproduce(tree: fx.SyntheticTree) -> None:
    _rewrite(tree.roots.output_root / tst.P5_2_STOP_RULE_RECORD,
             lambda p: p.__setitem__("ci95_low", p["ci95_low"] - 2 ** -30))
    tree = _resum(tree, tst.P5_2_SUMS_NAME, "p5_2", "p5_2_sums_sha256")
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    with pytest.raises(ValueError, match="score_stop_rule over the two 4-head records does not reproduce"):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())


def test_a_declared_exception_is_verified_not_trusted(tree: fx.SyntheticTree) -> None:
    """``BRIEF_45`` section 1.4: an exception holds only if the recorded integer is reachable by a tie-break order and
    the rule gives another; a false explanation refuses the build, and so does one declared where the two agree."""
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    statements = tst.all_statements(tst.make_books(inputs.cells))
    q2b = [pick(statements, "Q2b", tier=t, definition="att_ours", version="as_committed")["numbers"]
           for t in tst.OUT_OF_SAMPLE_TIERS]
    rule = [e["n_concordant"] for e in q2b]
    reachable = [e["range"][1] if e["range"][1] != e["n_concordant"] else e["n_concordant"] for e in q2b]
    assert reachable != rule, "the synthetic tree no longer has a tie-dependent Q2b count"
    good = (tst.RecordRow("docs/returns/P5.2.md", (27, 27), " · ".join(map(str, reachable)), "q2b.counts",
                          "deferred_108"),)
    artifact = tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=good)
    row = artifact["comparison_with_the_record"]["literal_records"][0]
    assert row["equal"] is False and row["exception"] == "deferred_108"
    unreachable = [r + 20 for r in rule]
    bad = (tst.RecordRow("docs/returns/P5.2.md", (27, 27), " · ".join(map(str, unreachable)), "q2b.counts",
                         "deferred_108"),)
    with pytest.raises(ValueError, match="does not verify"):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=bad)
    agreeing = (tst.RecordRow("docs/returns/P5.2.md", (27, 27), " · ".join(map(str, rule)), "q2b.counts",
                              "deferred_108"),)
    with pytest.raises(ValueError, match="would be false"):
        tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=agreeing)


# ----------------------------------------------------------------------
# T-barrier, T-canonical, T-cli, T-code-root
# ----------------------------------------------------------------------


@pytest.fixture
def cli(tree: fx.SyntheticTree, monkeypatch: pytest.MonkeyPatch) -> fx.SyntheticTree:
    """``main`` on the synthetic tree: its pins, no literal record, a synthetic git block and code list."""
    monkeypatch.setattr(tst, "PINS", tree.pins)
    monkeypatch.setattr(tst, "RECORD", ())
    monkeypatch.setattr(tst, "git_block", lambda: dict(SYNTHETIC_GIT))
    monkeypatch.setattr(tst, "assert_code_under_root", lambda root: list(SYNTHETIC_CODE))
    return tree


def _argv(command: str, tree: fx.SyntheticTree, out: Path | str | None = None) -> list[str]:
    argv = [command, "--repo-root", str(tree.roots.repo_root), "--output-root", str(tree.roots.output_root)]
    return argv + (["--out", str(out)] if out is not None else [])


def test_t_barrier_a_control_path_is_written_once(cli: fx.SyntheticTree, tmp_path: Path) -> None:
    out = cli.roots.repo_root / "docs" / "data" / "p5_2_statements.json"
    assert tst.main(_argv("build", cli, out)) == 0
    written = out.read_bytes()
    assert written.endswith(b"\n") and json.loads(written)["format_version"] == "p5.2-statements/1.0"
    leftovers = [p.name for p in out.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


@pytest.mark.parametrize("where", ["absolute", "dotdot", "symlink"])
def test_t_barrier_refuses_an_out_path_under_the_output_root(cli: fx.SyntheticTree, where: str,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    output = cli.roots.output_root
    if where == "absolute":
        out = output / "p5_2" / "statements.json"
    elif where == "dotdot":
        out = cli.roots.repo_root / ".." / "output" / "statements.json"
    else:
        link = cli.roots.repo_root / "docs" / "linked"
        link.symlink_to(output, target_is_directory=True)
        out = link / "statements.json"
    before = fx.snapshot(output)
    assert tst.main(_argv("build", cli, out)) == 2
    assert "lies at or under" in capsys.readouterr().err
    assert fx.snapshot(output) == before


def test_t_barrier_refuses_a_tree_holding_p5_2s_manifest_whatever_the_output_root_says(
        cli: fx.SyntheticTree, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = tmp_path / "another_output"
    (other / "sub").mkdir(parents=True)
    (other / tst.P5_2_SUMS_NAME).write_text("x\n", encoding="utf-8")
    assert tst.main(_argv("build", cli, other / "sub" / "statements.json")) == 2
    assert "holds SHA256SUMS_p5_2.txt" in capsys.readouterr().err
    assert not (other / "sub" / "statements.json").exists()


def test_t_barrier_refuses_an_input_as_the_out_path(cli: fx.SyntheticTree, capsys: pytest.CaptureFixture[str]) -> None:
    """``BRIEF_45`` Amendment B, B2.9: the input-equality guard runs before the exists check, so its own message is the
    one an input as the out path receives."""
    target = cli.roots.repo_root / tst.P8_4B_ARTIFACT
    before = target.read_bytes()
    assert tst.main(_argv("build", cli, target)) == 2
    err = capsys.readouterr().err
    assert "is an input of this artifact" in err and "already exists" not in err
    assert target.read_bytes() == before


def test_t_barrier_an_existing_artifact_is_refused_and_left_byte_identical(cli: fx.SyntheticTree,
                                                                           capsys: pytest.CaptureFixture[str]) -> None:
    """M18: written once."""
    out = cli.roots.repo_root / "docs" / "data" / "p5_2_statements.json"
    out.write_bytes(b"{}\n")
    stat = out.stat().st_mtime_ns
    assert tst.main(_argv("build", cli, out)) == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_bytes() == b"{}\n" and out.stat().st_mtime_ns == stat


def test_t_barrier_refuses_a_missing_parent_and_creates_no_directory(cli: fx.SyntheticTree,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    out = cli.roots.repo_root / "docs" / "new_dir" / "p5_2_statements.json"
    before = fx.directories(cli.roots.repo_root)
    assert tst.main(_argv("build", cli, out)) == 2
    assert "does not exist" in capsys.readouterr().err
    assert fx.directories(cli.roots.repo_root) == before


def test_t_barrier_refuses_a_non_json_out_path(cli: fx.SyntheticTree, capsys: pytest.CaptureFixture[str]) -> None:
    assert tst.main(_argv("build", cli, cli.roots.repo_root / "docs" / "data" / "statements.txt")) == 2
    assert ".json" in capsys.readouterr().err


def test_t_barrier_a_failing_input_check_leaves_no_file_no_temporary_and_no_directory(
        cli: fx.SyntheticTree, capsys: pytest.CaptureFixture[str]) -> None:
    out = cli.roots.repo_root / "docs" / "data" / "p5_2_statements.json"
    sums = cli.roots.output_root / tst.P5_1_SUMS_NAME
    sums.write_text(sums.read_text() + "\n", encoding="utf-8")
    before_files, before_dirs = fx.snapshot(cli.roots.repo_root), fx.directories(cli.roots.repo_root)
    assert tst.main(_argv("build", cli, out)) == 2
    assert "not the pinned" in capsys.readouterr().err
    assert fx.snapshot(cli.roots.repo_root) == before_files and fx.directories(cli.roots.repo_root) == before_dirs


def test_t_canonical_two_builds_are_byte_identical_sorted_finite_relative(tree: fx.SyntheticTree) -> None:
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    first = tst.serialize(tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=()))
    second = tst.serialize(tst.build_artifact(tst.load_inputs(tree.roots, pins=tree.pins), git=SYNTHETIC_GIT,
                                              code=SYNTHETIC_CODE, record=()))
    assert first == second
    assert first.endswith(b"}\n") and not first.endswith(b"\n\n")
    payload = json.loads(first)
    assert (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") == first
    roots = (str(tree.roots.repo_root), str(tree.roots.output_root), str(REPO_ROOT))

    def strings(node: Any, path: str = "") -> list[tuple[str, str]]:
        if isinstance(node, dict):
            return [s for k, v in node.items() for s in strings(v, f"{path}.{k}")]
        if isinstance(node, list):
            return [s for i, v in enumerate(node) for s in strings(v, f"{path}[{i}]")]
        return [(path, node)] if isinstance(node, str) else []

    for path, value in strings(payload):
        if path == ".git.code_root":
            # Exempt by ruling (BRIEF_45 Amendment B, B2.8): compute_latency._git_provenance returns the checkout's
            # absolute path, P5.2b's committed artifact records it the same way, and the packet discloses it.
            continue
        assert not value.startswith("/") and not any(root in value for root in roots), (path, value)
    assert payload["format_version"] == "p5.2-statements/1.0"
    assert set(payload) == {"format_version", "role", "registered_in", "git", "provenance", "definitions", "cells",
                            "statements", "ties", "leaders", "descriptive_collapse_audit", "comparison_with_the_record",
                            "changes", "what_this_does_not_say"}


def test_t_cli_check_writes_nothing_and_build_writes_exactly_one_file(cli: fx.SyntheticTree, tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture[str]) -> None:
    before = fx.snapshot(tmp_path)
    assert tst.main(_argv("check", cli)) == 0
    assert fx.snapshot(tmp_path) == before
    out = capsys.readouterr().out
    assert tst.P5_1_SUMS_NAME in out and tst.P8_4B_ARTIFACT in out
    target = cli.roots.repo_root / "docs" / "data" / "p5_2_statements.json"
    assert tst.main(_argv("check", cli, target)) == 0
    assert not target.exists() and fx.snapshot(tmp_path) == before
    assert tst.main(_argv("build", cli, target)) == 0
    after = fx.snapshot(tmp_path)
    assert set(after) - set(before) == {target.relative_to(tmp_path).as_posix()}
    assert all(after[k] == v for k, v in before.items())


def test_t_cli_build_refuses_a_dirty_tree_and_check_reports_it(cli: fx.SyntheticTree,
                                                               monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    """M17."""
    monkeypatch.setattr(tst, "git_block", lambda: dict(SYNTHETIC_GIT, dirty=True))
    target = cli.roots.repo_root / "docs" / "data" / "p5_2_statements.json"
    assert tst.main(_argv("build", cli, target)) == 2
    assert "dirty" in capsys.readouterr().err
    assert not target.exists()
    assert tst.main(_argv("check", cli)) == 0
    assert "dirty" in capsys.readouterr().out


def test_t_cli_build_requires_out(cli: fx.SyntheticTree, capsys: pytest.CaptureFixture[str]) -> None:
    assert tst.main(_argv("build", cli)) == 2
    assert "--out" in capsys.readouterr().err


def test_t_code_root_records_each_module_relative_with_its_sha256() -> None:
    entries = tst.assert_code_under_root(REPO_ROOT)
    assert [e["module"] for e in entries] == list(tst.CODE_MODULES)
    for entry in entries:
        path = REPO_ROOT / entry["path"]
        assert not entry["path"].startswith("/") and entry["path"].startswith("offline/")
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_t_code_root_a_module_outside_the_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not resolve under"):
        tst.assert_code_under_root(tmp_path)


def test_t_code_root_a_loaded_offline_module_from_another_tree_is_refused(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    stray = types.ModuleType("offline._p5_2c_stray")
    stray.__file__ = str(tmp_path / "offline" / "_p5_2c_stray.py")
    monkeypatch.setitem(sys.modules, "offline._p5_2c_stray", stray)
    with pytest.raises(ValueError, match="does not resolve under"):
        tst.assert_code_under_root(REPO_ROOT)


def test_t_code_root_the_declared_modules_are_exactly_the_offline_modules_an_import_loads() -> None:
    """``CODE_MODULES`` is what ``provenance.code`` hashes; a fresh interpreter importing the module loads exactly these
    ``offline`` modules, so no module whose code runs is left out of the record."""
    import subprocess

    probe = ("import sys, offline.tier_statements; "
             "print(sorted(n for n in sys.modules if n == 'offline' or n.startswith('offline.')))")
    completed = subprocess.run([sys.executable, "-P", "-c", probe], capture_output=True, text=True, timeout=300,
                               env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, check=True)
    assert json.loads(completed.stdout.strip().replace("'", '"')) == sorted(tst.CODE_MODULES)


def test_m19_main_refuses_a_repo_root_the_running_code_is_not_under(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    """Amendment A, A4.1 (M19): the check runs in ``main`` before anything is read."""
    (tmp_path / "output").mkdir()
    argv = ["check", "--repo-root", str(tmp_path), "--output-root", str(tmp_path / "output")]
    assert tst.main(argv) == 2
    assert "does not resolve under" in capsys.readouterr().err


# ----------------------------------------------------------------------
# T-quotes (ungated: every cited file is history and is committed)
# ----------------------------------------------------------------------


def _norm(text: str) -> str:
    """This file's own normalisation, written independently of ``tst.normalise``."""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        while stripped.startswith(">"):
            stripped = stripped[1:].strip()
        kept.append(stripped)
    joined = " ".join(kept).replace("**", "").replace("\\*", "*").replace("−", "-")
    return re.sub(r"\s+", " ", joined).strip()


def _cited(file: str, lines: tuple[int, int]) -> str:
    content = (REPO_ROOT / file).read_text(encoding="utf-8").split("\n")
    assert 1 <= lines[0] <= lines[1] <= len(content), (file, lines)
    return "\n".join(content[lines[0] - 1:lines[1]])


def test_t_quotes_every_rule_quotation_appears_at_its_cited_lines() -> None:
    checked = 0
    for key, quotes in tst.RULES.items():
        for quote in quotes:
            assert quote.file in tst.CITABLE_FILES, (key, quote.file)
            assert _norm(quote.text) in _norm(_cited(quote.file, quote.lines)), (key, quote)
            checked += 1
    assert checked >= 40


def test_t_quotes_every_record_literal_appears_at_its_cited_lines() -> None:
    for row in tst.RECORD:
        assert row.file in tst.CITABLE_FILES
        assert _norm(row.recorded) in _norm(_cited(row.file, row.lines)), row
    for quote in tst.Q5_RECORDS:
        assert quote.file in tst.CITABLE_FILES, quote  # Amendment B, B2.11
        assert _norm(quote.text) in _norm(_cited(quote.file, quote.lines)), quote
    assert {row.exception for row in tst.RECORD} == {None, *tst.RECORD_EXCEPTIONS}


def _line_citations(text: str) -> list[str]:
    """Every ``file:line`` citation in *text*, in three forms (Amendment B, B2.11): a path with an extension
    (``docs/plans/p5.2.md:523``), the same in backticks before the colon (`` `docs/returns/P5.2.md`:31 ``), and an
    extension-less upper-case name (``PROJECT_PLAN:12``, ``BRIEF_27:1274``)."""
    paths = re.findall(r"`?([A-Za-z0-9_./-]+\.(?:md|py|json|txt))`?:(\d+)", text)
    names = re.findall(r"(?<![A-Za-z0-9_./-])([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*)`?:(\d+)", text)
    return [file for file, _ in paths] + [name for name, _ in names]


def _not_citable(citation: str) -> bool:
    """A path must be one of the history files; a bare name must begin one of their file names."""
    if "/" in citation or "." in citation:
        return citation not in tst.CITABLE_FILES
    return not any(Path(file).name.startswith(citation) for file in tst.CITABLE_FILES)


def test_t_quotes_no_line_pinned_citation_points_into_a_living_document() -> None:
    """Amendment A, A4.2: ``file:line`` only into history; the plan, README, DEFERRED, CLAUDE.md and BRIEF_45 never.
    The scan catches the backticked and the extension-less forms too (Amendment B, B2.11), shown on samples first."""
    living = {"docs/PROJECT_PLAN.md", "README.md", "docs/notes/DEFERRED.md", "CLAUDE.md",
              "docs/briefs/BRIEF_45_p5.2c_statements.md"}
    assert not living & set(tst.CITABLE_FILES)
    caught = _line_citations("`docs/PROJECT_PLAN.md`:12, PROJECT_PLAN:12, README:3 and `BRIEF_45_p5.2c_statements.md`:7")
    assert [c for c in caught if _not_citable(c)] == ["docs/PROJECT_PLAN.md", "BRIEF_45_p5.2c_statements.md",
                                                     "PROJECT_PLAN", "README"]
    accepted = _line_citations("`docs/returns/P5.2.md:31`, docs/plans/p5.2.md:523, BRIEF_27:1274, PREREGISTRATION:12")
    assert len(accepted) == 4 and [c for c in accepted if _not_citable(c)] == []
    source = (REPO_ROOT / "offline" / "tier_statements.py").read_text(encoding="utf-8")
    citations = _line_citations(source)
    assert citations, "the pattern found nothing, so it checks nothing"
    assert [c for c in citations if _not_citable(c)] == [], "a line-pinned citation points outside the history files"


def test_t_quotes_the_modules_normalisation_agrees_with_this_files() -> None:
    samples = ["> **A** b\n>   c \\* d", "x − y  **z**", "plain"]
    assert [tst.normalise(s) for s in samples] == [_norm(s) for s in samples]


# ----------------------------------------------------------------------
# Amendment B (gate G1): a test for each surviving coordinator mutant (B1), and the rulings' tests (B2).  Every
# expected answer is this file's own: raw rows, ``statistics``, the registered rule written out here.
# ----------------------------------------------------------------------


def _shifted(book: Mapping[tuple[str, str], tst.Cell], key: tuple[str, str], shift: Any
             ) -> dict[tuple[str, str], tst.Cell]:
    """A copy of *book* whose cell at *key* has ``shift(row)`` added to both definitions on every row."""
    out = dict(book)
    c = book[key]
    out[key] = tst.Cell(spec=c.spec, rows=tuple(dict(r, att_ours=r["att_ours"] + shift(r),
                                                     att_engine=r["att_engine"] + shift(r)) for r in c.rows))
    return out


def _as(book: Mapping[tuple[str, str], tst.Cell], target: tuple[str, str], source: tuple[str, str], shift: Any = None
        ) -> dict[tuple[str, str], tst.Cell]:
    """A copy of *book* whose cell at *target* holds *source*'s episodes (plus ``shift(row)``), under its own spec."""
    out = dict(book)
    spec = book[target].spec
    step = shift or (lambda r: 0.0)
    out[target] = tst.Cell(spec=spec, rows=tuple(dict(r, arm=spec.arm, att_ours=r["att_ours"] + step(r),
                                                      att_engine=r["att_engine"] + step(r)) for r in book[source].rows))
    return out


def _both(books: Mapping[str, Mapping[tuple[str, str], tst.Cell]], edit: Any
          ) -> dict[str, dict[tuple[str, str], tst.Cell]]:
    """The same edit applied to both versions' books."""
    return {version: edit(book) for version, book in books.items()}


def _alternating(r: Mapping[str, Any]) -> float:
    """+1.0 on odd draws and -0.75 on even ones: a mean of +0.125 over four draws, with a CI that straddles zero."""
    return 1.0 if int(r["draw_id"]) % 2 else -0.75


def test_b1_cm17_q3b_aggregate_reads_all_three_tiers(books: Any) -> None:
    """CM17 (``BRIEF_45`` Amendment B, B1): the fixture's Q3b is HELD / NOT RESOLVED / HELD by this file's intervals,
    so the conjunction over all three tiers is NOT RESOLVED; an aggregate ignoring ``fixedtime`` would be HELD."""
    book = books["corrected"]
    own = {t: own_contrast_outcome(book[("dt_nomix", t)], book[("behaviour", t)], "att_ours", "negative")
           for t in tst.OUT_OF_SAMPLE_TIERS}
    assert own == {"maxpressure": "HELD", "fixedtime": "NOT RESOLVED", "random": "HELD"}, "the data no longer discriminate"
    entries = {(e["id"], e["tier"]): e for e in tst.score_q3b(book, definition="att_ours")}
    assert {t: entries[("Q3b", t)]["verdict"] for t in own} == own
    assert entries[("Q3b.aggregate", None)]["verdict"] == own_conjunction(list(own.values()))


def test_b1_cm9_each_definitions_per_seed_block_reads_that_definitions_values() -> None:
    """CM9: a contrast's D9 per-seed block is computed on the definition it is under.  Here ``att_ours`` favours the left
    arm by 1.0 on every seed while ``att_engine`` reverses on seed 303, so a block read on ``att_ours`` is caught."""
    seeds, draws = (101, 202, 303), range(1000, 1004)
    engine_shift = {101: -2.0, 202: -1.5, 303: 0.75}
    values = {"att_ours": ({(s, d): 50.0 + (d % 3) / 8.0 for s in seeds for d in draws},
                           {(s, d): 51.0 + (d % 3) / 8.0 for s in seeds for d in draws}),
              "att_engine": ({(s, d): 43.0 + engine_shift[s] + (d % 2) / 4.0 for s in seeds for d in draws},
                             {(s, d): 43.0 + (d % 2) / 4.0 for s in seeds for d in draws})}
    a = fx.simple_cell("dt_spatial", "random", values["att_ours"][0], values["att_engine"][0])
    b = fx.simple_cell("dt_nomix", "random", values["att_ours"][1], values["att_engine"][1])
    blocks = {}
    for definition, (left, right) in values.items():
        blocks[definition] = tst.contrast(a, b, definition=definition, tier="random")
        own = {str(s): seq_mean([left[(s, d)] for d in draws]) - seq_mean([right[(s, d)] for d in draws]) for s in seeds}
        assert blocks[definition]["per_seed"]["by_seed"] == own, definition
    assert blocks["att_ours"]["per_seed"]["by_seed"] != blocks["att_engine"]["per_seed"]["by_seed"]
    assert blocks["att_ours"]["seed_reversal_qualifier"] is None
    assert "(303)" in blocks["att_engine"]["seed_reversal_qualifier"]


def test_cm13_a_contrast_zero_on_some_draws_only_is_not_exactly_zero() -> None:
    """CM13 (killed at G1 only through the real record's three exact zeros): ``exactly_zero`` means EVERY per-draw
    difference is zero.  Here two of four draws differ by exactly zero and two by 1.5, so it is False, and the reading is
    the interval's, by this file's interval."""
    seeds, draws = (101, 202, 303), range(1000, 1004)
    left = {(s, d): 60.0 + s / 64.0 + (d % 2) * 1.5 for s in seeds for d in draws}
    right = {(s, d): 60.0 + s / 64.0 for s in seeds for d in draws}
    a, b = fx.simple_cell("bc", "maxpressure", left), fx.simple_cell("iql", "maxpressure", right)
    differences = own_paired(a, b, "att_ours")
    assert sorted(differences) == [0.0, 0.0, 1.5, 1.5]
    block = tst.contrast(a, b, definition="att_ours", tier="maxpressure")
    assert block["exactly_zero"] is False
    _, _, low, high = own_interval(differences)
    assert block["reading"] == ("resolves: the CI lies entirely above zero" if low > 0.0 else
                                "does not resolve: the CI includes zero")


def _own_seed_firsts(book: Mapping[tuple[str, str], tst.Cell], tier: str, definition: str) -> dict[str, list[str]]:
    """Per seed, the arms with the lowest level by ``statistics.mean`` over that seed's rows."""
    seeds = sorted({int(r["seed"]) for r in book[("iql", tier)].rows})
    out = {}
    for seed in seeds:
        levels = {m: statistics.mean(float(r[definition]) for r in book[(m, tier)].rows if int(r["seed"]) == seed)
                  for m in ts.METHODS}
        out[str(seed)] = [m for m in ts.METHODS if levels[m] == min(levels.values())]
    return out


def test_b1_cm7_the_leaders_per_seed_first_places_are_read_under_each_definition(books: Any) -> None:
    """CM7: at ``maxpressure``, ``dt_nomix``'s ``att_engine`` is lowered by 10 on seed 303 only, so under ``att_engine``
    seed 303's first place is ``dt_nomix`` while ``att_ours`` keeps ``bc`` first on every seed."""
    def lower(book: Mapping[tuple[str, str], tst.Cell]) -> dict[tuple[str, str], tst.Cell]:
        out = dict(book)
        c = book[("dt_nomix", "maxpressure")]
        out[("dt_nomix", "maxpressure")] = tst.Cell(spec=c.spec, rows=tuple(
            dict(r, att_engine=r["att_engine"] - (10.0 if r["seed"] == 303 else 0.0)) for r in c.rows))
        return out

    altered = _both(books, lower)
    blocks = {b["definition"]: b for b in tst.leaders_block(altered) if b["tier"] == "maxpressure"}
    for definition in tst.DEFINITIONS:
        assert blocks[definition]["per_seed_first_places"] == _own_seed_firsts(altered["corrected"], "maxpressure",
                                                                                definition)
    assert blocks["att_engine"]["per_seed_first_places"]["303"] == ["dt_nomix"]
    assert all(first == ["bc"] for first in blocks["att_ours"]["per_seed_first_places"].values())


def _lead_reading(high: float) -> str:
    return "resolves (the CI lies entirely below zero)" if high < 0.0 else "does not resolve (the CI includes zero)"


def test_b1_cm16_a_lead_whose_ci_straddles_zero_does_not_resolve(books: Any) -> None:
    """CM16: at ``maxpressure`` ``dt_nomix`` is made ``bc``'s episodes plus +1.0 on odd draws and -0.75 on even ones, so
    ``bc`` still leads (by 0.125) and the lead's CI, by this file's interval, straddles zero."""
    altered = _both(books, lambda book: _as(book, ("dt_nomix", "maxpressure"), ("bc", "maxpressure"), _alternating))
    book = altered["corrected"]
    _, _, low, high = own_interval(own_paired(book[("bc", "maxpressure")], book[("dt_nomix", "maxpressure")],
                                              "att_ours"))
    assert low < 0.0 < high, "the lead no longer straddles zero"
    block = next(b for b in tst.leaders_block(altered) if b["tier"] == "maxpressure" and b["definition"] == "att_ours")
    assert block["leaders"] == ["bc"] and block["runners_up"] == ["dt_nomix"]
    assert block["reading"] == f"bc leads dt_nomix; the lead {_lead_reading(high)}"


def test_b1_cm8_the_sentence_fails_when_one_non_dt_arm_never_collapses(books: Any) -> None:
    """CM8: ``iql`` collapses only at ``mappo1000`` in the fixture; lowered by 40 there it collapses nowhere, so the
    README sentence (every non-DT arm collapses somewhere) fails under both definitions and both versions."""
    altered = _both(books, lambda book: _shifted(book, ("iql", "mappo1000"), lambda r: -40.0))
    audit = tst.collapse_audit(altered)
    assert len(audit["sentence_holds"]) == 4
    for verdict in audit["sentence_holds"]:
        book, definition = altered[verdict["version"]], verdict["definition"]
        assert not any(own_collapses(book, "iql", t, definition) for t in tst.TIERS)
        assert all(any(own_collapses(book, arm, t, definition) for t in tst.TIERS)
                   for arm in ("bc", "bc_top10", "bc_top10_perix"))
        assert own_sentence_holds(altered, definition, verdict["version"]) is False
        assert verdict["holds"] is False


def test_b1_cm10_a_dt_arm_whose_ci_against_the_anchor_straddles_zero_is_not_collapsed(books: Any) -> None:
    """CM10: ``dt_spatial`` at ``random`` made the anchor's episodes plus +1.0 on odd draws and -0.75 on even ones --
    worse on average, the CI straddling zero -- so NOT COLLAPSED by Q6's rule on this file's interval."""
    book = _as(books["corrected"], ("dt_spatial", "random"), tst.ANCHOR, _alternating)
    _, _, low, high = own_interval(own_paired(book[("dt_spatial", "random")], book[tst.ANCHOR], "att_ours"))
    assert low < 0.0 < high, "the CI no longer straddles zero"
    entry = next(e for e in tst.score_q6(book, definition="att_ours") if (e["arm"], e["tier"]) == ("dt_spatial", "random"))
    assert entry["verdict"] == ("COLLAPSED" if low > 0.0 else "NOT COLLAPSED")


def test_b1_cm12_a_predicted_first_place_tied_for_first_is_not_held(books: Any) -> None:
    """CM12: at ``random`` the predicted first, ``dt_nomix``, is made ``iql``'s episodes (``iql`` is first there): the two
    tie exactly for first, so there is no unique first place and Q2a is FAILED by the registered rule."""
    book = _as(books["corrected"], ("dt_nomix", "random"), ("iql", "random"))
    predicted = list(ts.predicted_order("random"))
    levels = {m: own_level(book[(m, "random")], "att_ours") for m in ts.METHODS}
    firsts = [m for m in ts.METHODS if levels[m] == min(levels.values())]
    assert predicted[0] in firsts and len(firsts) > 1, "the data no longer tie the predicted first for first"
    entry = next(e for e in tst.score_q2_q3(book, definition="att_ours") if (e["id"], e["tier"]) == ("Q2a", "random"))
    assert entry["verdict"] == ("HELD" if firsts == [predicted[0]] else "FAILED")


def test_b1_cm4_q1b_holds_only_within_ten_percent_of_the_predicted_level(books: Any) -> None:
    """CM4: ``behaviour`` at ``maxpressure`` moved to 1.2 x its predicted level: 20 % off, so FAILED by the registered
    +-10 % anchor band (``tier_sweep.ANCHOR_BAND``), which the +-30 % level band would pass."""
    assert ts.ANCHOR_BAND == 0.10
    predicted = ts.PREDICTED_LEVELS["behaviour"]["maxpressure"]
    start = own_level(books["corrected"][("behaviour", "maxpressure")], "att_ours")
    book = _shifted(books["corrected"], ("behaviour", "maxpressure"), lambda r: 1.2 * predicted - start)
    measured = own_level(book[("behaviour", "maxpressure")], "att_ours")
    relative = abs(measured - predicted) / predicted
    assert 0.10 < relative <= 0.30, "the cell no longer separates the two bands"
    entries = {(e["id"], e["tier"]): e for e in tst.score_q1(book, definition="att_ours")}
    assert entries[("Q1b", "maxpressure")]["verdict"] == ("HELD" if relative <= 0.10 else "FAILED")
    assert entries[("Q1b.both", None)]["verdict"] == "FAILED"


def test_b1_cm14_a_chain_with_two_equal_rungs_is_not_strictly_increasing(books: Any) -> None:
    """CM14: d = 1, 2, 2, 3 along the four-rung chain: not strictly increasing, not strictly decreasing, so Q4b FAILED
    and not monotone -- by this file's strict comparison of its own d."""
    book = books["corrected"]
    for tier, shift in (("mappo1000", 1.0), ("maxpressure", 2.0), ("fixedtime", 2.0), ("random", 3.0)):
        book = _with_d(book, tier, shift)
    d = [seq_contrast(book[("dt_spatial", t)], book[("dt_nomix", t)], "att_ours") for t in tst.TIERS]
    assert d == [1.0, 2.0, 2.0, 3.0]
    increasing = all(a < b for a, b in zip(d, d[1:]))
    entries = {e["id"]: e for e in tst.score_q4(book, definition="att_ours")}
    assert entries["Q4b"]["verdict"] == ("HELD" if increasing else "FAILED")
    assert entries["Q4b"]["reading"] == "d is not monotone along the chain"
    assert entries["Q4b.three_rung"]["verdict"] == "HELD"


def test_b1_cm1_an_exactly_zero_interaction_is_held_by_the_registered_inclusion_rule(books: Any) -> None:
    """CM1 reversed (``BRIEF_45`` Amendment B, B2.1): Q0b is HELD iff the CI of mean(I) includes 0
    (``docs/plans/p5.2.md:523-524``); an exactly-zero I has CI [0, 0], which includes 0, so it is HELD and labelled.  Both
    pairs are made identical here, so d4 = d1 = 0 on every draw.  Q0a, a predicted-sign rule, stays NOT RESOLVED."""
    book = _as(books["corrected"], ("dt_spatial_h4", "mappo1000"), ("dt_nomix_h4", "mappo1000"))
    book = _as(book, ("dt_spatial", "mappo1000"), ("dt_nomix", "mappo1000"))
    s4, n4 = book[("dt_spatial_h4", "mappo1000")], book[("dt_nomix_h4", "mappo1000")]
    s1, n1 = book[("dt_spatial", "mappo1000")], book[("dt_nomix", "mappo1000")]
    interaction = [a - b for a, b in zip(own_paired(s4, n4, "att_ours"), own_paired(s1, n1, "att_ours"))]
    assert interaction and all(x == 0.0 for x in interaction)
    _, _, low, high = own_interval(interaction)
    entries = {e["id"]: e for e in tst.score_q0(book, definition="att_ours")}
    assert entries["Q0b"]["verdict"] == ("HELD" if low <= 0.0 <= high else "FAILED")
    assert entries["Q0b"]["numbers"]["exactly_zero"] is True
    assert entries["Q0b"]["reading"].startswith("exactly zero (identical episodes)")
    assert entries["Q0a"]["verdict"] == own_outcome(0.0, 0.0, "positive")
    assert "the registered rule decides the verdict" in tst.definitions_block()["exact_zero"]


def _broken_tie_cells() -> dict[str, tst.Cell]:
    """``bc`` at ``random`` is ``iql``'s episodes +0.5 on seed 101 and -0.5 on seed 202: an exact pooled tie for first
    (the sums are exact), broken one way on seed 101, the other on 202, and kept on 303."""
    cells = fx.synthetic_cells()
    spec, offset = cells["p5_2:bc@random"].spec, {101: 0.5, 202: -0.5, 303: 0.0}
    cells["p5_2:bc@random"] = tst.Cell(spec=spec, rows=tuple(
        dict(r, arm=spec.arm, att_ours=r["att_ours"] + offset[r["seed"]], att_engine=r["att_engine"] + offset[r["seed"]])
        for r in cells["p5_2:iql@random"].rows))
    return cells


def test_b2_3_a_pooled_tie_for_first_has_no_first_under_the_rule_and_reversals_against_the_pooled_set() -> None:
    """``BRIEF_45`` Amendment B, B2.3: beside P5.2b's fields, ``measured_first_under_rule`` (null when the pooled first is
    tied) and the seeds whose set of first places differs from the pooled SET -- by ``statistics.mean`` here."""
    expected = {"unique": ("iql", []), "tied": (None, []), "broken": (None, [101, 202])}
    for name, cells in (("unique", fx.synthetic_cells()), ("tied", _tied_cells()), ("broken", _broken_tie_cells())):
        book = {(m, "random"): cells[f"p5_2:{m}@random"] for m in ts.METHODS}
        pooled = {m: own_level(book[(m, "random")], "att_ours") for m in ts.METHODS}
        firsts = {m for m in ts.METHODS if pooled[m] == min(pooled.values())}
        reversing = [int(s) for s, f in _own_seed_firsts(book, "random", "att_ours").items() if set(f) != firsts]
        q2a = tst.tier_composer(_oos_episodes(cells, "att_ours"), tier="random", with_hard_subset=True)["q2a"]
        assert q2a["measured_first_under_rule"] == (next(iter(firsts)) if len(firsts) == 1 else None), name
        assert q2a["reverses_on_seeds_against_pooled_firsts"] == reversing, name
        assert (q2a["measured_first_under_rule"], reversing) == expected[name], name
    tied = tst.tier_composer(_oos_episodes(_tied_cells(), "att_ours"), tier="random", with_hard_subset=True)["q2a"]
    assert tied["reverses_on_seeds"] == list(fx.SEEDS)  # P5.2b's field, kept: every seed differs from the tie-break


def test_b2_4_the_file_run_by_path_from_another_tree_is_refused(tmp_path: Path) -> None:
    """``BRIEF_45`` Amendment B, B2.4 (R2 F3.1): the running file itself must lie under ``<--repo-root>/offline``.  A copy
    of ``offline/`` run by path, importing every other module from this tree, is refused before any input is read; the
    same command on this tree's own file passes the check and stops at the empty output root instead."""
    import shutil
    import subprocess

    copy = tmp_path / "copy"
    shutil.copytree(REPO_ROOT / "offline", copy / "offline", ignore=shutil.ignore_patterns("__pycache__"))
    output = tmp_path / "empty_output"
    output.mkdir()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}

    def run(script: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-P", str(script), "check", "--repo-root", str(REPO_ROOT),
                               "--output-root", str(output)], capture_output=True, text=True, timeout=600, env=env,
                              cwd=tmp_path)

    script = (copy / "offline" / "tier_statements.py").resolve()
    refused = run(script)
    assert refused.returncode == 2, refused.stderr
    assert "REFUSED: ValueError" in refused.stderr and str(script) in refused.stderr, refused.stderr
    assert "does not resolve under" in refused.stderr
    control = run(REPO_ROOT / "offline" / "tier_statements.py")
    assert control.returncode == 2 and "REFUSED: FileNotFoundError" in control.stderr, control.stderr
    assert "does not resolve under" not in control.stderr


def test_b2_4_provenance_code_hashes_the_file_that_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BRIEF_45`` Amendment B, B2.4: ``provenance.code`` records, for ``offline.tier_statements``, the file whose code is
    running -- even when ``sys.modules`` holds another module under that name (here one whose file is another module
    of this tree, so the location check itself passes)."""
    stand_in = types.ModuleType("offline.tier_statements")
    stand_in.__file__ = str(REPO_ROOT / "offline" / "tier_sweep.py")
    monkeypatch.setitem(sys.modules, "offline.tier_statements", stand_in)
    entries = {e["module"]: e for e in tst.assert_code_under_root(REPO_ROOT)}
    running = Path(tst.__file__).resolve()
    assert entries["offline.tier_statements"]["path"] == running.relative_to(REPO_ROOT).as_posix()
    assert entries["offline.tier_statements"]["sha256"] == hashlib.sha256(running.read_bytes()).hexdigest()


def _components(tree: fx.SyntheticTree) -> tuple[tst.Inputs, list[dict[str, Any]], Any, Any]:
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    books = tst.make_books(inputs.cells)
    return inputs, tst.all_statements(books), tst.collapse_audit(books), tst.ties_block(books)


def _ulps_away(x: float, n: int) -> float:
    y = x + n * math.ulp(x)
    assert y - x == n * math.ulp(x) and math.ulp(y) == math.ulp(x), "not an exact shift within one binade"
    return y


#: The three values of each record B2.5 bounds (``BRIEF_45`` Amendment B.1, B.1.1, CB5c).
ULP_KEYS = ["mean_difference", "ci95_low", "ci95_high"]


@pytest.mark.parametrize("key", ULP_KEYS)
@pytest.mark.parametrize(("contrast_id", "statement_id", "definition"), [
    ("V4", "Q0.d1", "att_engine"), ("V5", "Q0a", "att_engine"), ("V5", "Q0a", "att_ours")])
def test_b2_5_a_value_five_ulps_from_p8_4bs_record_is_refused_and_four_are_accepted(
        tree: fx.SyntheticTree, contrast_id: str, statement_id: str, definition: str, key: str) -> None:
    """``BRIEF_45`` Amendment B, B2.5: the route explanation is verified, not asserted -- each difference from P8.4b's V4
    and V5 is at most ``4 * math.ulp(max(|a|, |b|))``.  Amendment B.1, B.1.1 (CB5c): over each record's three values,
    for V4 (this artifact's ``d1``) and V5 (``d4``).  One value of the statement is set 4 and then 5 ulps from the
    record's own value (P8.4b's route still reproduces the record), and the refusal names that value.  V4 under
    ``att_ours`` is not a case: there ``d1`` is also P5.1's P1, which the build compares under ``==`` before it reads
    the bound, so a moved value is refused by that comparison first."""
    import copy

    inputs, statements, audit, ties = _components(tree)
    recorded = next(c for c in inputs.records["p8_4b_artifact"]["contrasts"]
                    if c["contrast_id"] == contrast_id)["by_definition"][definition][key]

    def moved(n: int) -> list[dict[str, Any]]:
        out = copy.deepcopy(statements)
        entry = next(s for s in out if s["id"] == statement_id and s["definition"] == definition)
        entry["numbers"][key] = _ulps_away(recorded, n)
        return out

    accepted = tst.comparison_with_the_record(moved(4), audit, ties, inputs, record=())
    row = next(r for r in accepted["p8_4b"] if (r["contrast_id"], r["definition"]) == (contrast_id, definition))
    assert row["recorded"][key] == recorded
    assert row["this_artifact"][key] - recorded == 4 * math.ulp(recorded)
    assert row["equal"] is False and row["record_route_reproduces"] is True
    refusal = re.escape(f"P8.4b's {contrast_id} under {definition}, {key}: ") + ".*more than 4 ulps"
    with pytest.raises(ValueError, match=refusal):
        tst.comparison_with_the_record(moved(5), audit, ties, inputs, record=())


@pytest.mark.parametrize("key", ULP_KEYS)
def test_b2_5_a_stop_rule_record_five_ulps_away_is_refused_and_four_are_accepted(tree: fx.SyntheticTree,
                                                                                 monkeypatch: pytest.MonkeyPatch,
                                                                                 key: str) -> None:
    """``BRIEF_45`` Amendment B, B2.5, for P5.2's stop-rule record: one of the record's values moved 4 and 5 ulps
    (re-listed in its manifest), with ``tier_sweep.score_stop_rule`` standing in to reproduce the moved record, so the
    bound alone decides.  Amendment B.1, B.1.1 (CB5c): over the record's three values; this artifact's value is the
    unmoved record's, so the distance is exactly 4 and then 5 ulps, and the refusal names the moved value."""
    path = tree.roots.output_root / tst.P5_2_STOP_RULE_RECORD
    original = json.loads(path.read_bytes())[key]
    monkeypatch.setattr(ts, "score_stop_rule", lambda *args: json.loads(path.read_bytes()))

    def build(n: int) -> dict[str, Any]:
        _rewrite(path, lambda p: p.__setitem__(key, _ulps_away(original, n)))
        moved = _resum(tree, tst.P5_2_SUMS_NAME, "p5_2", "p5_2_sums_sha256")
        inputs = tst.load_inputs(moved.roots, pins=moved.pins)
        return tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())

    stop = build(4)["comparison_with_the_record"]["p5_2_stop_rule_record"]
    assert stop["this_artifact"][key] == original
    assert stop["recorded"][key] - original == 4 * math.ulp(original)
    assert stop["equal"] is False
    with pytest.raises(ValueError, match=re.escape(f"P5.2's stop-rule record, {key}: ") + ".*more than 4 ulps"):
        build(5)


def test_b2_6_the_gate_2_record_cites_1274_to_1281_and_quotes_the_sentence_that_is_the_error(
        tree: fx.SyntheticTree) -> None:
    """``BRIEF_45`` Amendment B, B2.6: A3's GATE 2 row cites ``BRIEF_27``:1274-1281 and quotes the sentence that is the
    error, beside the heading's claim; T-quotes holds the text at those lines."""
    gate2 = [q for q in tst.Q5_RECORDS if q.file == "docs/briefs/BRIEF_27_p5.2_tier_sweep.md"]
    assert len(gate2) == 1 and gate2[0].lines == (1274, 1281)
    text = _norm(gate2[0].text)
    assert "THE LOAD-SORTER MECHANISM IS CONFIRMED, decisively and twice" in text
    assert "the `fixedtime` null is *predicted* by the mechanism" in text
    assert text in _norm(_cited(gate2[0].file, gate2[0].lines))
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    q5 = tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())["comparison_with_the_record"]["q5"]
    assert [r["record"]["source"] for r in q5] == ["docs/returns/P5.2.md:30", "docs/returns/P5.2.md:572-573",
                                                   "docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1274-1281"]


def test_b2_7_q6_in_never_scored_names_its_numbers_and_the_recorded_verdict(tree: fx.SyntheticTree) -> None:
    """``BRIEF_45`` Amendment B, B2.7: Q6's verdict was recorded (``docs/returns/P5.2.md:31``); its numbers were not."""
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    never = tst.build_artifact(inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE, record=())[
        "comparison_with_the_record"]["never_scored"]
    for entry in never:
        if entry["statement"] == "Q6":
            assert entry["recorded"] == ("Q6's numbers: never scored; the verdict was recorded "
                                         "(`docs/returns/P5.2.md:31`)")
        else:
            assert entry["recorded"] == "never scored in any committed record, under either definition (BRIEF_45 " \
                                        "section 1.3)"
    assert sum(1 for entry in never if entry["statement"] == "Q6") == 1
    assert "no DT arm collapsed on any tier" in _norm(_cited("docs/returns/P5.2.md", (31, 31)))


def test_b2_9_a_dangling_symlink_at_the_out_path_is_refused_and_nothing_is_written(
        cli: fx.SyntheticTree, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``BRIEF_45`` Amendment B, B2.9 (R2 F8.1): the UNRESOLVED out path is examined, so a symlink there -- dangling or
    not -- is refused; nothing is written at its target and the link is left as it was."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "target.json"
    out = cli.roots.repo_root / "docs" / "data" / "p5_2_statements.json"
    out.symlink_to(target)
    assert out.is_symlink() and not out.exists()
    assert tst.main(_argv("build", cli, out)) == 2
    assert "symlink" in capsys.readouterr().err
    assert list(elsewhere.iterdir()) == [] and out.is_symlink() and not out.exists()


def test_b2_13_what_this_does_not_say_states_what_definition_dependent_does_not_compare() -> None:
    """``BRIEF_45`` Amendment B, B2.13 (R1 F-12)."""
    assert ("Q0b's sign and Q4b's chain shape are reported per definition beside the flag, and the flag does not "
            "compare them") in " ".join(tst.WHAT_THIS_DOES_NOT_SAY)


def test_b2_14_p8_4bs_campaign_manifest_and_marker_are_listed_with_their_sha256(tree: fx.SyntheticTree) -> None:
    """``BRIEF_45`` Amendment B, B2.14: P8.4b's ``campaign_manifest.json`` and ``CAMPAIGN_COMPLETE`` in
    ``provenance.inputs``, each with the sha256 this test computes from the file."""
    inputs = tst.load_inputs(tree.roots, pins=tree.pins)
    listed = {entry["file"]: entry["sha256"] for entry in inputs.provenance["inputs"]}
    for name in ("campaign_manifest.json", "CAMPAIGN_COMPLETE"):
        path = tree.roots.output_root / tst.P8_4B_DIR / name
        assert listed[f"output/{tst.P8_4B_DIR}/{name}"] == hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# Amendment B.1 (gate G1.1): the four survivors (B.1.1) and Q0b's exact-zero label (B.1.2(1)).  Every expected answer is
# this file's own: raw rows, ``statistics``, the rule written out here.  (CB5c is the two B2.5 tests above.)
# ----------------------------------------------------------------------


#: Seed offsets summing to zero: added to a cell's episodes they change three of every four ``(seed, draw)`` values and
#: no draw's seed sum, which stays exact because every fixture value is dyadic.
ZERO_SUM_OFFSETS = {101: 0.5, 202: -0.5, 303: 0.0}

#: Q0b's reading on an exactly-zero I from pairs that are not both identical (``BRIEF_45`` Amendment B.1, B.1.2(1)).
NOT_IDENTICAL_LABEL = "exactly zero on every draw, from episodes that are not identical (d4 equals d1 draw by draw)"


def _own_identical(a: tst.Cell, b: tst.Cell, definition: str) -> bool:
    """True iff the two cells hold equal values on every ``(seed, draw)``, read from the rows."""
    left = {(int(r["seed"]), int(r["draw_id"])): float(r[definition]) for r in a.rows}
    right = {(int(r["seed"]), int(r["draw_id"])): float(r[definition]) for r in b.rows}
    assert left.keys() == right.keys()
    return all(left[key] == right[key] for key in left)


def _own_i_by_seed(book: Mapping[tuple[str, str], tst.Cell], definition: str) -> dict[int, float]:
    """I per training seed by this file's loop: the mean over draws of ``dt_spatial_h4 - dt_nomix_h4`` minus that of
    ``dt_spatial - dt_nomix`` (``seq_mean``, exact on the fixture's dyadic values)."""
    rows = {m: by_seed(book[(m, "mappo1000")].rows, definition)
            for m in ("dt_spatial_h4", "dt_nomix_h4", "dt_spatial", "dt_nomix")}
    return {s: seq_mean([a - b for a, b in zip(rows["dt_spatial_h4"][s], rows["dt_nomix_h4"][s])])
            - seq_mean([a - b for a, b in zip(rows["dt_spatial"][s], rows["dt_nomix"][s])])
            for s in sorted(rows["dt_spatial_h4"])}


def test_b11_cb2_q0bs_qualifier_names_d4_against_d1_from_this_files_per_seed_loop(books: Any) -> None:
    """CB2 (``BRIEF_45`` Amendment B.1, B.1.1): ``dt_spatial_h4`` + 25 on seed 101 makes I reverse on that seed alone.
    Q0b's per-seed values are this test's loop's, and its ``seed_reversal_qualifier`` begins with the sentence built here
    from that loop, naming d4 first: a worker called with the two names swapped reads "d1 vs d4"."""
    book = _shifted(books["corrected"], ("dt_spatial_h4", "mappo1000"), lambda r: 25.0 if r["seed"] == 101 else 0.0)
    for definition in tst.DEFINITIONS:
        per_seed = _own_i_by_seed(book, definition)
        pooled = seq_mean(list(per_seed.values()))
        reversing = [str(s) for s, v in per_seed.items() if (v < 0) != (pooled < 0) and v != 0.0]
        assert reversing == ["101"], "the data no longer discriminate"
        expected = (f"⚠️ ORDERING NOT ROBUST ACROSS SEEDS: d4 vs d1 reverses on {len(reversing)} of "
                    f"{len(per_seed)} training seeds ({', '.join(reversing)})")
        numbers = next(e for e in tst.score_q0(book, definition=definition) if e["id"] == "Q0b")["numbers"]
        assert numbers["per_seed"]["by_seed"] == {str(s): v for s, v in per_seed.items()}, definition
        qualifier = numbers["seed_reversal_qualifier"]
        assert qualifier is not None and qualifier.startswith(expected), qualifier


def test_b11_cb1a_q3bs_exact_zero_from_episodes_that_differ_is_not_called_identical(books: Any) -> None:
    """CB1a (``BRIEF_45`` Amendment B.1, B.1.1): at ``maxpressure`` ``dt_nomix`` holds ``behaviour``'s episodes moved by
    :data:`ZERO_SUM_OFFSETS`, so ``dt_nomix - behaviour`` is exactly zero on every draw from episodes that differ --
    both facts read from the rows here.  The sub-prediction's reading names episodes that are not identical, and
    ``discriminability.identical`` is False."""
    book = _as(books["corrected"], ("dt_nomix", "maxpressure"), ("behaviour", "maxpressure"),
               lambda r: ZERO_SUM_OFFSETS[r["seed"]])
    left, right = book[("dt_nomix", "maxpressure")], book[("behaviour", "maxpressure")]
    for definition in tst.DEFINITIONS:
        differences = own_paired(left, right, definition)
        assert differences and all(x == 0.0 for x in differences), "no longer exactly zero on every draw"
        identical = _own_identical(left, right, definition)
        assert identical is False, "the episodes no longer differ"
        _, _, low, high = own_interval(differences)
        resolved = low > 0.0 or high < 0.0
        entry = next(e for e in tst.score_q3b(book, definition=definition) if (e["id"], e["tier"]) == ("Q3b",
                                                                                                         "maxpressure"))
        numbers = entry["numbers"]
        assert numbers["exactly_zero"] is True
        assert numbers["discriminability"]["identical"] is identical
        sub = numbers["sub_prediction"]
        assert (sub["exactly_zero"], sub["resolved"], sub["matched"]) == (True, resolved, not resolved)
        assert "episodes that are not identical" in sub["reading"]
        assert "identical episodes" not in sub["reading"], sub["reading"]


@pytest.mark.parametrize(("low", "high", "ruled"), [(0.0, 1.0, "HELD"), (-1.0, 0.0, "HELD"), (2.0 ** -10, 1.0, "FAILED")])
def test_b11_cb0_a_ci_touching_zero_at_one_end_is_included_by_q0bs_rule(books: Any, monkeypatch: pytest.MonkeyPatch,
                                                                        low: float, high: float, ruled: str) -> None:
    """CB0 (``BRIEF_45`` Amendment B.1, B.1.1): Q0b is HELD iff the CI of mean(I) INCLUDES 0
    (``docs/plans/p5.2.md:523-524``), and a CI touching zero at one end includes it.  ``composite_contrast`` is
    substituted in the module's namespace by the real block carrying this CI, with ``exactly_zero`` False; the expected
    verdict is this test's own ``low <= 0.0 <= high``, and the three cases are the ruling's."""
    real = tst.composite_contrast

    def touching(left: Any, right: Any, *, definition: str, tier: str) -> dict[str, Any]:
        mean = (low + high) / 2.0
        return {**real(left, right, definition=definition, tier=tier), "mean_difference": mean, "ci95_low": low,
                "ci95_high": high, "ci95_half_width": (high - low) / 2.0, "ci95_width": high - low,
                "exactly_zero": False, "sign": "positive" if mean > 0.0 else "negative"}

    monkeypatch.setattr(tst, "composite_contrast", touching)
    q0b = next(e for e in tst.score_q0(books["corrected"], definition="att_ours") if e["id"] == "Q0b")
    assert (q0b["numbers"]["ci95_low"], q0b["numbers"]["ci95_high"], q0b["numbers"]["exactly_zero"]) == (low, high, False)
    own = "HELD" if low <= 0.0 <= high else "FAILED"
    assert own == ruled, "the cases no longer discriminate"
    assert q0b["verdict"] == own
    assert q0b["reading"].startswith("the CI of mean(I) includes zero" if own == "HELD" else
                                     "the CI of mean(I) excludes zero")


def _assert_q0b_exactly_zero_and_not_identical(book: Mapping[tuple[str, str], tst.Cell],
                                               pairs: Mapping[str, bool]) -> None:
    """Under both definitions: I is exactly zero on every draw and the pairs' identity is *pairs* (each read from the
    rows here); Q0b is HELD by this file's inclusion rule, its reading begins with :data:`NOT_IDENTICAL_LABEL`, and its
    ``identical_pairs`` and ``identical`` are this file's."""
    s4, n4 = book[("dt_spatial_h4", "mappo1000")], book[("dt_nomix_h4", "mappo1000")]
    s1, n1 = book[("dt_spatial", "mappo1000")], book[("dt_nomix", "mappo1000")]
    for definition in tst.DEFINITIONS:
        interaction = [a - b for a, b in zip(own_paired(s4, n4, definition), own_paired(s1, n1, definition))]
        assert interaction and all(x == 0.0 for x in interaction), "I is no longer exactly zero on every draw"
        own_pairs = {"d4": _own_identical(s4, n4, definition), "d1": _own_identical(s1, n1, definition)}
        assert own_pairs == dict(pairs), "the data no longer discriminate"
        _, _, low, high = own_interval(interaction)
        q0b = next(e for e in tst.score_q0(book, definition=definition) if e["id"] == "Q0b")
        assert q0b["verdict"] == ("HELD" if low <= 0.0 <= high else "FAILED")
        assert q0b["reading"].startswith(NOT_IDENTICAL_LABEL), q0b["reading"]
        assert q0b["numbers"]["exactly_zero"] is True
        assert q0b["numbers"]["identical_pairs"] == own_pairs
        assert q0b["numbers"]["identical"] is (own_pairs["d4"] and own_pairs["d1"])


def test_b11_2_an_i_exactly_zero_because_the_horizon_changes_nothing_is_held_and_not_identical(books: Any) -> None:
    """``BRIEF_45`` Amendment B.1, B.1.2(1): s4's episodes are s1's and n4's are n1's while s differs from n, so I is
    exactly zero on every draw from pairs that are not identical.  Q0b is HELD (B2.1), its reading begins with the
    not-identical label, and ``identical`` is False."""
    book = _as(books["corrected"], ("dt_spatial_h4", "mappo1000"), ("dt_spatial", "mappo1000"))
    book = _as(book, ("dt_nomix_h4", "mappo1000"), ("dt_nomix", "mappo1000"))
    _assert_q0b_exactly_zero_and_not_identical(book, {"d4": False, "d1": False})


def test_b11_2_identical_needs_both_pairs_identical(books: Any) -> None:
    """``BRIEF_45`` Amendment B.1, B.1.2(1): ``identical`` is True iff BOTH pairs are.  Here d4's pair is identical
    (``dt_spatial_h4`` holds ``dt_nomix_h4``'s episodes) and d1's is not, yet is exactly zero draw by draw
    (``dt_spatial`` is ``dt_nomix`` moved by :data:`ZERO_SUM_OFFSETS`), so I is exactly zero and not identical."""
    book = _as(books["corrected"], ("dt_spatial_h4", "mappo1000"), ("dt_nomix_h4", "mappo1000"))
    book = _as(book, ("dt_spatial", "mappo1000"), ("dt_nomix", "mappo1000"), lambda r: ZERO_SUM_OFFSETS[r["seed"]])
    _assert_q0b_exactly_zero_and_not_identical(book, {"d4": True, "d1": False})


# ----------------------------------------------------------------------
# Real data (gated on RLTRAFFIC_OUTPUT_ROOT)
# ----------------------------------------------------------------------


def _output_root() -> Path:
    value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    candidate = Path(value) if value else REPO_ROOT / "output"
    marker = candidate / "p8_4b_rederivation" / "CAMPAIGN_COMPLETE"
    if not marker.is_file():
        pytest.skip(f"{marker} not found: set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/ (it is gitignored)")
    return candidate


@pytest.fixture(scope="module")
def real_inputs() -> tst.Inputs:
    return tst.load_inputs(tst.Roots(repo_root=REPO_ROOT, output_root=_output_root()))


@pytest.fixture(scope="module")
def real_artifact(real_inputs: tst.Inputs) -> dict[str, Any]:
    return tst.build_artifact(real_inputs, git=SYNTHETIC_GIT, code=SYNTHETIC_CODE)


def _real_rows(output: Path, arm: str) -> list[dict[str, Any]]:
    """P8.4b's rows of one arm, read by this test from the files, (seed, draw) order."""
    from offline.att_rederivation import CellKey, cell_file_name

    rows = []
    for seed in (101, 202, 303, 404, 505):
        for draw in range(1000, 1100):
            path = output / tst.P8_4B_DIR / cell_file_name(CellKey(scenario="grid4x4", arm=arm, seed=seed,
                                                                   draw_id=draw))
            rows.append(json.loads(path.read_bytes()))
    return rows


def test_t_inputs_real_every_anchor_holds_and_the_two_row_digests_recompute(real_inputs: tst.Inputs) -> None:
    output = _output_root()
    provenance = real_inputs.provenance
    for name, pin in ((tst.P5_1_SUMS_NAME, tst.PINS.p5_1_sums_sha256), (tst.P5_2_SUMS_NAME, tst.PINS.p5_2_sums_sha256),
                      (tst.P5_2B_SUMS_NAME, tst.PINS.p5_2b_sums_sha256)):
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == pin
    from offline.att_rederivation import CellKey, cell_file_name

    pairs = []
    for spec in tst.CELL_SPECS:
        if spec.source == "p5_2b":
            continue
        for seed in (101, 202, 303, 404, 505):
            for draw in range(1000, 1100):
                relative = f"{tst.P8_4B_DIR}/" + cell_file_name(CellKey(scenario="grid4x4", arm=spec.arm, seed=seed,
                                                                        draw_id=draw))
                pairs.append((relative, hashlib.sha256((output / relative).read_bytes()).hexdigest()))
    text = "".join(f"{digest}  {path}\n" for path, digest in sorted(pairs))
    assert provenance["p8_4b_rows"]["n"] == len(pairs) == 15_500
    assert provenance["p8_4b_rows"]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    corrected = sorted((output / tst.P5_2B_REDERIVATION_DIR).glob("cell_*.json"))
    assert len(corrected) == 500
    text = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output).as_posix()}\n"
                   for p in corrected)
    assert provenance["p5_2b_corrected_rows"]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert provenance["p8_4b_campaign"]["declared_cells_sha256"] == tst.PINS.p8_4b_declared_cells_sha256
    assert provenance["p5_2b_rederivation"]["declared_cells_sha256"] == \
        tst.PINS.p5_2b_rederivation_declared_cells_sha256


def _q(artifact: Mapping[str, Any], statement_id: str, tier: str | None, *, definition: str = "att_ours",
       version: str = "as_committed", arm: str | None = None) -> Mapping[str, Any]:
    return pick(artifact["statements"], statement_id, tier=tier, definition=definition, version=version, arm=arm)


def _f4(x: float) -> str:
    return "0.0000" if x == 0 else f"{x:+.4f}"


def test_t_reproduce_record_every_value_p5_2s_record_states_at_its_precision(real_artifact: dict[str, Any]) -> None:
    """Load-bearing.  Every literal below is the record's (``docs/returns/P5.2.md`` section 1 and ``BRIEF_27`` GATE 2,
    the line in each comment), written here independently of ``tst.RECORD``.  The two exceptions are asserted AS
    exceptions: the declared rule gives another integer, and the recorded one lies in the range."""
    a = real_artifact
    q1 = _q(a, "Q1", None)  # P5.2.md:25, BRIEF_27:1216
    assert (q1["verdict"], q1["numbers"]["n_held"], q1["numbers"]["n_cells"], q1["numbers"]["threshold"]) == (
        "HELD", 15, 19, 14)
    misses = {tuple(c["cell"]): c for c in q1["numbers"]["cells"] if not c["held"]}
    expected = {("bc_top10", "fixedtime"): ("207.47", "80.1"), ("iql", "fixedtime"): ("203.95", "43.5"),
                ("bc_top10", "random"): ("350.34", "71.8"), ("iql", "random"): ("190.96", "54.0")}  # BRIEF_27:1219-1221
    assert set(misses) == set(expected)
    for cell, (measured, percent) in expected.items():
        assert f"{misses[cell]['measured']:.2f}" == measured
        assert f"{100 * misses[cell]['relative_error']:.1f}" == percent
    for tier, percent in (("maxpressure", "0.04"), ("fixedtime", "0.09")):  # P5.2.md:25
        entry = _q(a, "Q1b", tier)
        assert entry["verdict"] == "HELD" and f"{100 * entry['numbers']['relative_error']:.2f}" == percent
    tiers = tst.OUT_OF_SAMPLE_TIERS
    assert [_q(a, "Q2a", t)["numbers"]["measured_first"] for t in tiers] == ["bc", "iql", "iql"]  # P5.2.md:26
    assert [_q(a, "Q2a", t)["numbers"]["predicted_first"] for t in tiers] == ["bc_top10_perix", "dt_nomix", "dt_nomix"]
    assert [_q(a, "Q2a", t)["verdict"] for t in tiers] == ["FAILED"] * 3
    q2b = [_q(a, "Q2b", t) for t in tiers]  # P5.2.md:27 records 11 · 8 · 8
    assert [e["numbers"]["n_concordant"] for e in q2b] == [11, 7, 8]
    # DEFERRED 108 (Amendment B, B2.10): the recorded fixedtime count, parsed from its record row, lies in the range.
    row = next(r for r in a["comparison_with_the_record"]["literal_records"]
               if r["source"] == "docs/returns/P5.2.md:27" and r["value"] == "q2b.counts")
    recorded = [int(x) for x in re.findall(r"\d+", row["recorded"])]
    low, high = q2b[1]["numbers"]["range"]
    assert (low, high) == (7, 9) and recorded == [11, 8, 8]
    assert low <= recorded[1] <= high and recorded[1] != q2b[1]["numbers"]["n_concordant"]
    assert [e["verdict"] for e in q2b] == ["FAILED"] * 3
    hard = [_q(a, "Q2b_hard", t)["numbers"] for t in tiers]  # P5.2.md:87, :92-93
    assert [h["n_concordant"] for h in hard] == [2, 2, 3]
    assert [h["range"] for h in hard] == [[2, 2], [2, 4], [3, 3]]
    q3a = [_q(a, "Q3a", t)["numbers"] for t in tiers]  # P5.2.md:28 records 2/6 · 3/6 · 3/6
    assert [e["rank"] for e in q3a] == [2, 2, 3]
    assert q3a[1]["rank_range"] == [2, 3] and q3a[1]["tied_with"] == ["dt_spatial"]
    for tier, (mean, low, high) in (("maxpressure", ("-0.7294", "-0.9139", "-0.5450")),
                                    ("random", ("-5.0360", "-5.5005", "-4.5716"))):  # P5.2.md:74
        numbers = _q(a, "Q3b", tier)["numbers"]
        assert (_f4(numbers["mean_difference"]), _f4(numbers["ci95_low"]), _f4(numbers["ci95_high"])) == (mean, low, high)
    assert _q(a, "Q3b", "fixedtime")["numbers"]["exactly_zero"] is True  # P5.2.md:75
    for tier, (mean, low, high, agree) in (
            ("mappo1000", ("+39.5649", "+36.0510", "+43.0787", "5/5")),
            ("maxpressure", ("+0.7267", "+0.6206", "+0.8328", "5/5")),
            ("random", ("-0.7063", "-1.3172", "-0.0953", "4/5"))):  # P5.2.md:42-45, BRIEF_27:1260-1263
        numbers = _q(a, "Q4.d", tier)["numbers"]
        assert (_f4(numbers["mean_difference"]), _f4(numbers["ci95_low"]), _f4(numbers["ci95_high"])) == (mean, low, high)
        n = numbers["per_seed"]["n_seeds"]
        assert f"{n - numbers['per_seed']['reverses_against_pooled_sign']}/{n}" == agree
    assert _q(a, "Q4.d", "fixedtime")["numbers"]["exactly_zero"] is True
    assert "inverted" in _q(a, "Q4b", None)["reading"]  # P5.2.md:29
    for tier, (mean, low, high) in (("maxpressure", ("-469.69", "-477.81", "-461.58")),
                                    ("fixedtime", ("-0.28", "-0.67", "+0.11")),
                                    ("random", ("-49.16", "-55.26", "-43.05"))):  # BRIEF_27:1276-1277
        numbers = _q(a, "Q5b", tier)["numbers"]
        assert (f"{numbers['mean_difference']:+.2f}", f"{numbers['ci95_low']:+.2f}",
                f"{numbers['ci95_high']:+.2f}") == (mean, low, high)
    for arm in tst.DT_ARMS:  # P5.2.md:31
        for tier in tst.TIERS:
            assert _q(a, "Q6", tier, arm=arm)["verdict"] == "NOT COLLAPSED"
    d4, i = _q(a, "Q0a", "mappo1000")["numbers"], _q(a, "Q0b", "mappo1000")["numbers"]  # BRIEF_27:1622-1628
    assert (_f4(d4["mean_difference"]), _f4(d4["ci95_low"]), _f4(d4["ci95_high"])) == ("+27.1004", "+23.8150",
                                                                                         "+30.3858")
    assert (_f4(i["mean_difference"]), _f4(i["ci95_low"]), _f4(i["ci95_high"])) == ("-12.4645", "-16.3575", "-8.5714")
    assert i["draws_negative"] == 70 and i["n_draws"] == 100
    assert _q(a, "Q0.stop_rule", "mappo1000")["verdict"] == "CONTINUE"  # P5.2.md:24
    levels = _q(a, "ranking", "maxpressure")["numbers"]["levels"]  # P5.2.md:265-266
    assert (f"{levels['bc_top10']:.2f}", f"{levels['bc_top10_perix']:.2f}") == ("679.96", "210.27")
    fixedtime = _q(a, "ranking", "fixedtime")["numbers"]["levels"]  # docs/plans/p5.2.md:606-607
    assert (repr(fixedtime["dt_spatial"]), repr(fixedtime["dt_nomix"])) == ("207.11922169043655",) * 2
    assert (repr(fixedtime["bc"]), repr(fixedtime["bc_top10_perix"])) == ("207.1886372764493",) * 2
    rows = a["comparison_with_the_record"]["literal_records"]
    assert len(rows) == len(tst.RECORD)
    for row in rows:
        assert row["equal"] or row["exception"] in tst.RECORD_EXCEPTIONS, row
        assert not (row["equal"] and row["exception"]), row
    assert {r["exception"] for r in rows if not r["equal"]} == set(tst.RECORD_EXCEPTIONS)


def test_t_reproduce_p8_4b_v4_and_v5_in_the_four_part_exact_form(real_inputs: tst.Inputs,
                                                                 real_artifact: dict[str, Any]) -> None:
    """F1 (Amendment A, A2 Q1): (a) the verdicts equal V4's and V5's; (b) P8.4b's own route over the rows the artifact
    read reproduces V4 and V5 exactly; (c) ``d1`` under ``att_ours`` is P5.1's committed P1; (d) the comparison states
    both values and their difference, and this test recomputes it.  No tolerance anywhere."""
    from offline.att_rederivation import _paired_differences
    from offline.dt_gate import mean_ci95
    from offline.spatial_mixing import _outcome_from_interval

    committed = json.loads((REPO_ROOT / tst.P8_4B_ARTIFACT).read_bytes())
    v = {c["contrast_id"]: c["by_definition"] for c in committed["contrasts"] if c["contrast_id"] in ("V4", "V5")}
    cells = real_inputs.cells
    pairs = {"V4": ("p5_1:dt_spatial@grid4x4_mappo1000", "p5_1:dt_nomix@grid4x4_mappo1000"),
             "V5": ("p5_2:dt_spatial_h4@mappo1000", "p5_2:dt_nomix_h4@mappo1000")}
    rows = {r["contrast_id"] + "/" + r["definition"]: r for r in real_artifact["comparison_with_the_record"]["p8_4b"]}
    for definition in tst.DEFINITIONS:
        d1 = _q(real_artifact, "Q0.d1", "mappo1000", definition=definition)["numbers"]
        d4 = _q(real_artifact, "Q0a", "mappo1000", definition=definition)["numbers"]
        assert _outcome_from_interval(d1["ci95_low"], d1["ci95_high"]) == v["V4"][definition]["outcome"]
        assert _q(real_artifact, "Q0.stop_rule", "mappo1000", definition=definition)["verdict"] == \
            v["V5"][definition]["outcome"]
        for name, (left, right) in pairs.items():
            block = {cells[c].spec.arm: {(r["seed"], r["draw_id"]): {definition: r[definition]} for r in cells[c].rows}
                     for c in (left, right)}
            stats = mean_ci95(_paired_differences(block, cells[left].spec.arm, cells[right].spec.arm, definition))
            assert (stats.mean, stats.mean - stats.ci95, stats.mean + stats.ci95) == (
                v[name][definition]["mean_difference"], v[name][definition]["ci95_low"],
                v[name][definition]["ci95_high"])
            mine = d1 if name == "V4" else d4
            row = rows[f"{name}/{definition}"]
            assert row["recorded"]["mean_difference"] == v[name][definition]["mean_difference"]
            assert row["this_artifact"]["mean_difference"] == mine["mean_difference"]
            assert row["equal"] is (mine["mean_difference"] == v[name][definition]["mean_difference"]
                                    and mine["ci95_low"] == v[name][definition]["ci95_low"]
                                    and mine["ci95_high"] == v[name][definition]["ci95_high"])
            assert row["record_route_reproduces"] is True
    grid = json.loads((REPO_ROOT / tst.P5_1_GRID).read_bytes())["predictions"]["P1"]
    d1 = _q(real_artifact, "Q0.d1", "mappo1000")["numbers"]
    assert (d1["mean_difference"], d1["ci95_low"], d1["ci95_high"]) == (grid["mean_difference"], grid["ci95_low"],
                                                                         grid["ci95_high"])
    v4 = rows["V4/att_ours"]
    assert v4["equal"] is False and v4["difference"]["mean_difference"] == (
        d1["mean_difference"] - v["V4"]["att_ours"]["mean_difference"])


def test_t_reproduce_p5_2b_the_random_tier_on_both_versions_field_by_field(real_artifact: dict[str, Any]) -> None:
    committed = json.loads((REPO_ROOT / tst.P5_2B_ARTIFACT).read_bytes())
    a = real_artifact
    for definition in tst.DEFINITIONS:
        for when, version in (("before", "as_committed"), ("after", "corrected")):
            theirs = json.loads(json.dumps(committed["statements"][definition][when]))
            theirs["q1"].pop("predictions_registered_on")
            mine = {"ranking": _q(a, "ranking", "random", definition=definition, version=version)["numbers"],
                    "q2a": _q(a, "Q2a", "random", definition=definition, version=version)["numbers"],
                    "q2b": _q(a, "Q2b", "random", definition=definition, version=version)["numbers"],
                    "q2b_hard": _q(a, "Q2b_hard", "random", definition=definition, version=version)["numbers"],
                    "per_seed": _q(a, "D9", "random", definition=definition, version=version)["numbers"],
                    "q3a": _q(a, "Q3a", "random", definition=definition, version=version)["numbers"],
                    "q3c": _q(a, "Q3c", "random", definition=definition, version=version)["numbers"]}
            q1 = _q(a, "Q1", None, definition=definition, version=version)["numbers"]
            mine["q1"] = {"iql_random": next(c for c in q1["cells"] if c["cell"] == ["iql", "random"]),
                          **{k: q1[k] for k in ("n_held", "n_cells", "threshold", "band", "outcome", "rule")}}
            if definition == "att_engine":
                theirs.pop("q2b_hard", None)
            # The artifact's Q1 entry carries tier_sweep.score_level's own rule text; the composer's one-line summary of
            # it is the one q1 field not stored (the module's own comparison block checks the composer's output whole).
            theirs["q1"].pop("rule")
            assert contains(theirs, json.loads(json.dumps(mine))) == [], (definition, when)
    corrected = committed["cells"]["corrected"]
    block = a["cells"][tst.CORRECTED_IQL]
    for definition in tst.DEFINITIONS:
        for key in ("n", "mean", "std", "ci95", "ci95_low", "ci95_high", "per_seed_means"):
            assert block[definition][key] == corrected[definition][key], (definition, key)
    assert a["comparison_with_the_record"]["p5_2b_artifact"]["equal"] is True


def test_t_reproduce_p5_1_d1_is_p1_and_q6_at_mappo1000_is_p3(real_artifact: dict[str, Any]) -> None:
    grid = json.loads((REPO_ROOT / tst.P5_1_GRID).read_bytes())["predictions"]
    d1 = _q(real_artifact, "Q0.d1", "mappo1000")["numbers"]
    p1 = grid["P1"]
    for key in ("mean_difference", "ci95_low", "ci95_high"):
        assert d1[key] == p1[key], key
    for key in ("rank_biserial", "median_difference", "wins", "losses", "ties"):
        assert d1["paired_comparison"][key] == p1[key], key
    assert d1["paired_comparison"]["wilcoxon"]["p_value"] == p1["p_value"]
    for arm in tst.DT_ARMS:
        q6 = _q(real_artifact, "Q6", "mappo1000", arm=arm)["numbers"]
        p3 = grid["P3"][arm]
        for key in ("mean_difference", "ci95_low", "ci95_high"):
            assert q6[key] == p3[key], (arm, key)
        assert q6["p5_1_outcome"] == p3["outcome"]
        assert q6["paired_comparison"]["rank_biserial"] == p3["rank_biserial"]


def test_t_note_every_tiers_six_means_equal_the_notes_by_its_own_route(real_inputs: tst.Inputs) -> None:
    """``BRIEF_45`` section 5: means only -- the note's intervals use z = 1.959963984540054 and are never compared."""
    note = json.loads((REPO_ROOT / tst.C1_NOTE).read_bytes())
    books = tst.make_books(real_inputs.cells)
    for tier in tst.TIERS:
        for definition in tst.DEFINITIONS:
            for method in ts.METHODS:
                c = books["as_committed"][(method, tier)]
                mine = statistics.mean(float(r[definition]) for r in c.rows)
                assert mine == note["tiers"][f"grid4x4/{tier}"][definition]["means"][method], (tier, method, definition)


def test_t_companions_a11bs_five_quantities_on_every_cell(real_inputs: tst.Inputs,
                                                          real_artifact: dict[str, Any]) -> None:
    output = _output_root()
    for spec in tst.CELL_SPECS:
        entry = real_artifact["cells"][spec.cell_id]
        if spec.source == "p5_2b":
            rows = [json.loads(p.read_bytes()) for p in (output / tst.P5_2B_REDERIVATION_DIR).glob("cell_*.json")]
        else:
            rows = _real_rows(output, spec.arm)
        rows.sort(key=lambda r: (r["seed"], r["draw_id"]))
        assert len(rows) == 500
        for definition in tst.DEFINITIONS:
            assert entry[definition]["n"] == 500
        assert entry["att_engine"]["mean"] == float(np.mean(np.asarray([r["att_engine"] for r in rows])))
        for name in ("entered", "created", "never_entered"):
            assert entry["companions"][f"{name}_mean"] == statistics.mean(int(r[name]) for r in rows), (spec, name)
        assert entry["companions"]["entered_fraction_mean"] == float(np.mean(np.asarray(
            [r["entered_fraction"] for r in rows])))
        assert entry["companions"]["episodes_with_never_entered"] == sum(1 for r in rows if r["never_entered"] > 0)


def test_t_first_time_scores_each_against_this_files_own_recomputation(real_inputs: tst.Inputs,
                                                                       real_artifact: dict[str, Any]) -> None:
    """``BRIEF_45`` section 1.3's never-scored list, each verdict recomputed here from the rows by ``statistics`` (per
    draw the mean over the five seeds; ``1.96 * sd / 10``), and the values at the four decimals the coordinator's
    independent route reported (``BRIEF_45`` Amendment A, A0.3-A0.4)."""
    books = tst.make_books(real_inputs.cells)

    def own(left: tuple[str, str], right: tuple[str, str], definition: str, version: str = "corrected"
            ) -> tuple[float, float, float]:
        book = books[version]
        mean, _, low, high = own_interval(own_paired(book[left], book[right], definition))
        return mean, low, high

    expected_q5 = {  # BRIEF_45 Amendment A, A0.3 (att_ours), at four decimals
        ("Q5a", "maxpressure"): ("+514.4262", "+508.0807", "+520.7716"),
        ("Q5a", "fixedtime"): ("+0.2816", "-0.1093", "+0.6725"),
        ("Q5a", "random"): ("+60.9885", "+55.0077", "+66.9693"),
        ("Q5c", "mappo1000"): ("+23.2396", "+19.1155", "+27.3637"),
        ("Q5c", "maxpressure"): ("+44.7326", "+40.1822", "+49.2831"),
        ("Q5c", "random"): ("+11.8324", "+6.7711", "+16.8937"),
        ("Q5b", "mappo1000"): ("-557.3594", "-565.4166", "-549.3021"),
    }
    pairs = {"Q5a": ("bc_top10", "bc"), "Q5b": ("bc_top10_perix", "bc_top10"), "Q5c": ("bc_top10_perix", "bc")}
    for (statement, tier), literal in expected_q5.items():
        left, right = pairs[statement]
        mean, low, high = own((left, tier), (right, tier), "att_ours")
        numbers = _q(real_artifact, statement, tier)["numbers"]
        assert (_f4(numbers["mean_difference"]), _f4(numbers["ci95_low"]), _f4(numbers["ci95_high"])) == literal
        assert (_f4(mean), _f4(low), _f4(high)) == literal
    assert _q(real_artifact, "Q5c", "fixedtime")["numbers"]["exactly_zero"] is True
    for definition in tst.DEFINITIONS:
        assert [_q(real_artifact, f"{s}.aggregate", None, definition=definition)["verdict"]
                for s in ("Q5a", "Q5b", "Q5c")] == ["NOT RESOLVED", "NOT RESOLVED", "FAILED"]
        assert _q(real_artifact, "Q0a", "mappo1000", definition=definition)["verdict"] == "HELD"
        q0b = _q(real_artifact, "Q0b", "mappo1000", definition=definition)
        assert q0b["verdict"] == "FAILED" and q0b["numbers"]["sign"] == "negative"
        for tier in ("maxpressure", "fixedtime"):
            entry = _q(real_artifact, "Q3c", tier, definition=definition)
            best = entry["numbers"]["best_non_dt"]
            mean, low, high = own(("dt_nomix", tier), (best, tier), definition)
            reading = "resolves against the DT" if low > 0 else ("resolves for the DT" if high < 0 else "NOT RESOLVED")
            assert entry["reading"] == reading and entry["verdict"] is None
            assert entry["numbers"]["registered_value"] == {"maxpressure": 0.4326, "fixedtime": -17.1639}[tier]
        for arm in tst.DT_ARMS:
            for tier in tst.TIERS:
                mean, low, high = own((arm, tier), tst.ANCHOR, definition)
                entry = _q(real_artifact, "Q6", tier, definition=definition, arm=arm)
                assert entry["verdict"] == ("COLLAPSED" if low > 0 else "NOT COLLAPSED")
                assert high < 0, (arm, tier, definition)
    q3c = _q(real_artifact, "Q3c", "maxpressure", definition="att_engine")
    assert q3c["reading"] == "NOT RESOLVED" and q3c["definition_dependent"] is True


def test_t_check_real_main_accepts_the_real_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    output = _output_root()
    assert tst.main(["check", "--repo-root", str(REPO_ROOT), "--output-root", str(output)]) == 0
    out = capsys.readouterr().out
    assert tst.PINS.p8_4b_declared_cells_sha256 in out and "15500" in out


def test_t_regress_the_committed_artifact_regenerates_byte_for_byte_from_output(real_inputs: tst.Inputs) -> None:
    """``BRIEF_45`` section 5: given the build's own git and code blocks (the two values a regeneration cannot derive:
    they name the tree and the code that ran), ``docs/data/p5_2_statements.json`` is regenerated byte for byte."""
    if not COMMITTED_ARTIFACT.is_file():
        pytest.skip(f"{COMMITTED_ARTIFACT} not found: the artifact is built and committed at gate G2 (the plan's C7)")
    committed = COMMITTED_ARTIFACT.read_bytes()
    payload = json.loads(committed)
    regenerated = tst.serialize(tst.build_artifact(real_inputs, git=payload["git"], code=payload["provenance"]["code"]))
    assert hashlib.sha256(regenerated).hexdigest() == hashlib.sha256(committed).hexdigest()
    assert regenerated == committed


# ----------------------------------------------------------------------
# Amendment B (gate G1), real data: the survivors' gated assertions (B1) and B2.14
# ----------------------------------------------------------------------


def test_b1_cm17_real_q3b_aggregate_is_not_resolved_under_both_definitions(real_inputs: tst.Inputs,
                                                                          real_artifact: dict[str, Any]) -> None:
    """CM17 on real data (``BRIEF_45`` Amendment B, B1): HELD / NOT RESOLVED (``fixedtime``'s exact zero) / HELD by this
    file's intervals, so the aggregate is NOT RESOLVED under both definitions."""
    book = tst.make_books(real_inputs.cells)["corrected"]
    for definition in tst.DEFINITIONS:
        own = {t: own_contrast_outcome(book[("dt_nomix", t)], book[("behaviour", t)], definition, "negative")
               for t in tst.OUT_OF_SAMPLE_TIERS}
        assert own == {"maxpressure": "HELD", "fixedtime": "NOT RESOLVED", "random": "HELD"}, definition
        assert {t: _q(real_artifact, "Q3b", t, definition=definition)["verdict"] for t in own} == own
        aggregate = _q(real_artifact, "Q3b.aggregate", None, definition=definition)["verdict"]
        assert aggregate == own_conjunction(list(own.values())) == "NOT RESOLVED"


def test_b1_cm9_real_an_att_engine_per_seed_block_against_the_raw_rows(real_artifact: dict[str, Any]) -> None:
    """CM9 on real data: Q4's d at ``maxpressure`` under ``att_engine``, its per-seed block against P8.4b's rows read by
    this test -- per seed ``numpy.mean`` over the draws in draw order, the worker's own summation, so ``==`` holds while
    the data path is this file's.  The ``att_ours`` block differs, so a block read on ``att_ours`` is caught."""
    output = _output_root()
    left, right = _real_rows(output, "dt_spatial@maxpressure"), _real_rows(output, "dt_nomix@maxpressure")
    own = {str(seed): float(np.mean([r["att_engine"] for r in left if r["seed"] == seed])
                            - np.mean([r["att_engine"] for r in right if r["seed"] == seed]))
           for seed in (101, 202, 303, 404, 505)}
    blocks = {d: _q(real_artifact, "Q4.d", "maxpressure", definition=d)["numbers"]["per_seed"]["by_seed"]
              for d in tst.DEFINITIONS}
    assert blocks["att_engine"] == own
    assert blocks["att_ours"] != blocks["att_engine"]


def test_b1_cm7_real_maxpressure_first_places_per_seed_under_each_definition(real_artifact: dict[str, Any]) -> None:
    """CM7 on real data (``BRIEF_45`` Amendment B, B1, verified by the coordinator): at ``maxpressure`` under
    ``att_engine`` seed 303's first place is ``dt_nomix`` and ``bc`` holds the other four; under ``att_ours`` ``bc`` holds
    all five -- by this file's per-seed means over P8.4b's rows."""
    output = _output_root()
    rows = {m: _real_rows(output, f"{m}@maxpressure") for m in ts.METHODS}
    expected = {"att_ours": {s: ["bc"] for s in ("101", "202", "303", "404", "505")},
                "att_engine": {"101": ["bc"], "202": ["bc"], "303": ["dt_nomix"], "404": ["bc"], "505": ["bc"]}}
    for definition in tst.DEFINITIONS:
        firsts = {}
        for seed in (101, 202, 303, 404, 505):
            levels = {m: statistics.mean(float(r[definition]) for r in rows[m] if r["seed"] == seed) for m in ts.METHODS}
            firsts[str(seed)] = [m for m in ts.METHODS if levels[m] == min(levels.values())]
        assert firsts == expected[definition], definition
        block = next(b for b in real_artifact["leaders"] if b["tier"] == "maxpressure" and b["definition"] == definition)
        assert block["per_seed_first_places"] == firsts, definition


def test_b1_cm16_real_maxpressures_lead_resolves_under_att_ours_and_not_under_att_engine(
        real_inputs: tst.Inputs, real_artifact: dict[str, Any]) -> None:
    """CM16 on real data: ``bc - dt_nomix`` at ``maxpressure`` is -1.2993 [-2.3592, -0.2395] under ``att_ours`` and
    -0.9059 [-1.9735, +0.1617] under ``att_engine`` (``BRIEF_45`` Amendment B, B0's verification): the lead resolves under
    the first only, by this file's interval -- the README's sentence."""
    book = tst.make_books(real_inputs.cells)["corrected"]
    expected = {"att_ours": ("-1.2993", "-2.3592", "-0.2395"), "att_engine": ("-0.9059", "-1.9735", "+0.1617")}
    for definition in tst.DEFINITIONS:
        mean, _, low, high = own_interval(own_paired(book[("bc", "maxpressure")], book[("dt_nomix", "maxpressure")],
                                                     definition))
        assert (_f4(mean), _f4(low), _f4(high)) == expected[definition], definition
        block = next(b for b in real_artifact["leaders"] if b["tier"] == "maxpressure" and b["definition"] == definition)
        assert block["leaders"] == ["bc"] and block["runners_up"] == ["dt_nomix"]
        assert block["reading"] == f"bc leads dt_nomix; the lead {_lead_reading(high)}", definition


def test_b2_14_real_p8_4bs_campaign_manifest_and_marker_are_listed_with_their_sha256(real_inputs: tst.Inputs) -> None:
    output = _output_root()
    listed = {entry["file"]: entry["sha256"] for entry in real_inputs.provenance["inputs"]}
    for name in ("campaign_manifest.json", "CAMPAIGN_COMPLETE"):
        digest = hashlib.sha256((output / tst.P8_4B_DIR / name).read_bytes()).hexdigest()
        assert listed[f"output/{tst.P8_4B_DIR}/{name}"] == digest


# ----------------------------------------------------------------------
# Amendment B.1 (gate G1.1), real data: CB2's gated assertion
# ----------------------------------------------------------------------


#: I's per-seed values on the real data at four decimals, by the coordinator's independent route (``json`` and
#: ``statistics``, nothing from ``offline/``): ``BRIEF_45`` Amendment B.1, B.1.0(5).
REAL_I_BY_SEED = {
    "att_ours": {"101": "+45.8935", "202": "-38.7945", "303": "-33.8405", "404": "-13.8980", "505": "-21.6828"},
    "att_engine": {"101": "+46.4239", "202": "-39.1119", "303": "-34.1478", "404": "-14.1020", "505": "-21.9003"},
}


def test_b11_cb2_real_q0bs_per_seed_values_and_qualifier_under_both_definitions(real_artifact: dict[str, Any]) -> None:
    """CB2 on real data (``BRIEF_45`` Amendment B.1, B.1.1): under both definitions Q0b's ``by_seed`` equals B.1.0(5)'s
    values at four decimals, and its qualifier names d4 against d1, reversing on seed 101 alone."""
    for definition in tst.DEFINITIONS:
        numbers = _q(real_artifact, "Q0b", "mappo1000", definition=definition)["numbers"]
        assert {s: _f4(v) for s, v in numbers["per_seed"]["by_seed"].items()} == REAL_I_BY_SEED[definition], definition
        qualifier = numbers["seed_reversal_qualifier"]
        assert qualifier is not None and qualifier.startswith(
            "⚠️ ORDERING NOT ROBUST ACROSS SEEDS: d4 vs d1 reverses on 1 of 5 training seeds (101)"), qualifier
