"""P5.3c C3 (``BRIEF_42`` §4 T-T1 / T-T2 / T-holm; ``PREREGISTRATION`` A26(d) as corrected by A26.1(c)): the statistic.

What this file pins, and the named mutations it is built against:

* **T-T1 (load-bearing)** -- the per-draw linear-trend contrast ``s_d = sum(c_j A_d(K_j))``, ``c = (-2, -1, 0, +1, +2)``
  on the levels' RANKS, K ascending: an exact linear trend in rank gives ``s_d`` = the slope x 10 and a flat curve 0;
  the ONE-SIDED Wilcoxon (``less``) equals an independent second route on 100 draws, statistic and p under ``==``; a
  curve on which ATT RISES with K does NOT reject T1.  *Mutations:* the contrast on log K -> the exact-trend fixture
  dies; a two-sided p -> the direction fixture dies.
* **T-T2 (load-bearing)** -- the one-sided Wilcoxon (``greater``) on ``(A_d(K) - A_d(20)) - delta``: a K = 1 (and K = 2)
  gap of EXACTLY delta on every draw does not reject, delta + 0.01 on every draw does.  *Mutation:* the ``>`` boundary
  moved to ``>=`` -> the exactly-delta fixture dies.
* **T-holm** -- Holm's step-down within {T1, T2, T3} at alpha = 0.05, against the test's own implementation:
  (0.01, 0.02, 0.04) -> all reject; (0.01, 0.03, 0.04) -> (reject, not, not) (Amendment A, A1.2); the partition on all
  eight reject patterns; the registered sentence chosen and filled, (ii) naming the test(s) that did not reject and, when
  exactly one did, A26.1(c)'s clause.  *Mutation:* Holm replaced by the unadjusted alpha -> the second fixture dies.
* The sentence templates are the REGISTERED text: each, its slots aside, is found in ``PREREGISTRATION.md``'s A26 row (the
  clause in A26.1's) -- a second route against transcription.  delta is the literal 0.6263, equal to A6's constant and
  to P4.4's committed artifact.

ROUTE B (the second route, written here and nowhere else, ``docs/plans/p5.3c.md`` §7 and Appendix A.4): ranks by
COUNTING, the mean as ``sum(r)/2``, the variance as ``sum(r^2)/4`` -- the exact conditional variance under ties,
algebraically the closed form with its tie correction, computed from the ranks themselves.  ``W+``, ``E`` and ``Var`` are
multiples of 1/16 below 2**53, so both routes compute them exactly and ``z`` and ``p`` are then the same IEEE operations.
Both use the repository's correctly rounded normal CDF, which is cross-checked against ``statistics.NormalDist``.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import pytest

import offline.context_sweep as cs
from offline.dt_gate import _normal_cdf, wilcoxon_signed_rank
from offline.offline_baselines import DELTA_ATT

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAWS = tuple(range(1000, 1100))
LEVELS = (1, 2, 5, 10, 20)
MINUS = "−"


def route_b(differences: list[float] | np.ndarray, alternative: str) -> tuple[float, float, float, float, float, int, int]:
    """The second route: ranks by counting, E = sum(r)/2, Var = sum(r^2)/4, the tail rule written out separately."""
    values = [float(x) for x in differences if float(x) != 0.0]
    n_zero = sum(1 for x in differences if float(x) == 0.0)
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0, n_zero
    magnitudes = [abs(v) for v in values]
    ranks = []
    for m in magnitudes:
        below = sum(1 for other in magnitudes if other < m)
        equal = sum(1 for other in magnitudes if other == m)
        ranks.append(below + (equal + 1) / 2.0)
    w_plus = sum(r for r, v in zip(ranks, values) if v > 0)
    expected = sum(ranks) / 2.0
    variance = sum(r * r for r in ranks) / 4.0
    if alternative == "less":
        z = (w_plus - expected + 0.5) / math.sqrt(variance)
        p = _normal_cdf(z)
    else:
        z = (w_plus - expected - 0.5) / math.sqrt(variance)
        p = _normal_cdf(-z)
    return w_plus, expected, variance, z, p, len(values), n_zero


def holm_by_hand(p_values: tuple[float, ...], alpha: float = 0.05) -> tuple[bool, ...]:
    """The test's own Holm: sort ascending (ties by position), compare the i-th with alpha/(m - i), stop at the first
    failure."""
    m = len(p_values)
    order = sorted(range(m), key=lambda index: (p_values[index], index))
    decision = [False] * m
    for step, index in enumerate(order):
        if p_values[index] <= alpha / (m - step):
            decision[index] = True
        else:
            break
    return tuple(decision)


def _levels(value) -> dict[int, dict[int, float]]:
    """``{K: {draw: A_d(K)}}`` from ``value(rank, K, draw)``."""
    return {k: {d: float(value(rank, k, d)) for d in DRAWS} for rank, k in enumerate(LEVELS)}


# ----------------------------------------------------------------------------------------------------------------------
# delta, the contrast, the formats
# ----------------------------------------------------------------------------------------------------------------------


def test_delta_is_the_literal_a6_value_equal_to_the_constant_and_to_p4_4s_artifact() -> None:
    source = (REPO_ROOT / "offline" / "context_sweep.py").read_text(encoding="utf-8")
    assert re.search(r"^DELTA = 0\.6263$", source, flags=re.MULTILINE), "delta must be a literal constant"
    baselines = json.loads((REPO_ROOT / "docs" / "data" / "p4_4_baselines.json").read_bytes())
    assert cs.DELTA == DELTA_ATT == baselines["comparisons"]["madt_vs_bc"]["delta"] == 0.6263
    assert cs.ALPHA == 0.05
    assert cs.FAMILY == ("T1", "T2", "T3") and cs.SHORTFALL_TESTS == {"T2": 1, "T3": 2}


def test_the_contrast_is_on_the_levels_ranks_in_ascending_k_and_an_exact_rank_trend_gives_ten_times_the_slope() -> None:
    """*Mutation:* the contrast on log K -> this dies (the values here are linear in the RANK, not in log K)."""
    assert cs.RANK_CONTRAST == (-2, -1, 0, 1, 2) and cs.CONTEXT_LENGTHS == LEVELS
    for intercept, slope in ((100.0, -0.25), (87.5, 0.125), (104.0, -3.0), (100.0, 0.0)):
        values = {k: intercept + slope * rank for rank, k in enumerate(LEVELS)}
        assert cs.rank_contrast(values) == 10.0 * slope
    # The evaluation order is documented (G6's independent route): left to right over K ascending, from zero.
    values = {1: 101.3, 2: 100.9, 5: 100.8, 10: 100.1, 20: 99.95}
    by_hand = 0
    for coefficient, k in zip((-2, -1, 0, 1, 2), LEVELS):
        by_hand = by_hand + coefficient * values[k]
    assert cs.rank_contrast(values) == by_hand
    with pytest.raises(ValueError, match="levels"):
        cs.rank_contrast({1: 1.0, 2: 1.0, 5: 1.0, 10: 1.0})


def test_numbers_are_four_decimals_with_the_unicode_minus_and_a_ci_in_brackets() -> None:
    assert cs.format_number(-1.23456) == f"{MINUS}1.2346"
    assert cs.format_number(0.5) == "0.5000"
    assert cs.format_number(12.0) == "12.0000"
    assert cs.format_ci(-0.1, 0.2) == f"[{MINUS}0.1000, 0.2000]"
    assert "-" not in cs.format_number(-3.0) + cs.format_ci(-2.0, -1.0)


def test_the_per_draw_mean_is_over_the_seeds_in_ascending_seed_order_by_np_mean() -> None:
    """The reduction order G6's independent route must repeat, pinned: seeds ascending, ``np.mean`` in float64."""
    values = {}
    for d in DRAWS[:7]:
        for seed, v in zip((505, 101, 303, 202, 404), (0.1 * d, 1e-3 * d, 7.3, 1e-9, 33.33 + d)):
            values[(seed, d)] = v
    means = cs.per_draw_means(values)
    assert sorted(means) == list(DRAWS[:7])
    for d in DRAWS[:7]:
        ordered = [values[(seed, d)] for seed in (101, 202, 303, 404, 505)]
        assert means[d] == float(np.mean(np.asarray(ordered, dtype=np.float64)))
    with pytest.raises(ValueError, match="seed"):
        cs.per_draw_means({(101, 1000): 1.0, (202, 1000): 2.0, (101, 1001): 1.0})


# ----------------------------------------------------------------------------------------------------------------------
# The one-sided Wilcoxon against the second route
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_the_one_sided_wilcoxon_equals_the_second_route_under_equality_on_100_draws(alternative: str) -> None:
    rng = np.random.default_rng(20260929)
    for trial in range(60):
        base = rng.normal(loc=rng.normal(0.0, 0.3), scale=1.0, size=100)
        base = np.round(base, int(rng.integers(0, 3)))  # rounding makes ties and zeros
        a = cs.one_sided_wilcoxon(base, alternative=alternative)
        w_plus, expected, variance, z, p, n_used, n_zero = route_b(base, alternative)
        assert (a.w_plus, a.expected, a.variance, a.z, a.p_value) == (w_plus, expected, variance, z, p), trial
        assert (a.n_used, a.n_zero, a.alternative) == (n_used, n_zero, alternative)
        libm = statistics.NormalDist().cdf(a.z if alternative == "less" else -a.z)
        assert abs(a.p_value - libm) <= 1e-15
    assert cs.one_sided_wilcoxon(np.zeros(100), alternative=alternative).p_value == 1.0
    with pytest.raises(ValueError, match="alternative"):
        cs.one_sided_wilcoxon([1.0, 2.0], alternative="two-sided")


def test_the_lower_tail_is_half_the_reviewed_two_sided_p_whenever_the_statistic_lies_below_its_mean() -> None:
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(200):
        base = np.round(rng.normal(loc=-0.1, scale=1.0, size=100), 1)
        less = cs.one_sided_wilcoxon(base, alternative="less")
        two = wilcoxon_signed_rank(base, np.zeros_like(base))
        if less.w_plus < less.expected and two.p_value < 1.0:
            assert less.p_value == two.p_value / 2.0
            checked += 1
    assert checked >= 100


# ----------------------------------------------------------------------------------------------------------------------
# T1 and T2/T3 through the family
# ----------------------------------------------------------------------------------------------------------------------


def test_a_curve_on_which_att_rises_with_k_does_not_reject_t1_although_a_two_sided_test_would() -> None:
    """*Mutation:* a two-sided p used for T1 -> this dies."""
    levels = _levels(lambda rank, k, d: 100.0 + 0.5 * rank + 0.001 * (d - 1000))
    family = cs.confirmatory_family(levels)
    s_values = [cs.rank_contrast({k: levels[k][d] for k in LEVELS}) for d in DRAWS]
    assert all(s > 0 for s in s_values)
    t1 = family["tests"]["T1"]
    assert t1["alternative"] == "less" and t1["p_value"] == 1.0
    assert family["holm"]["rejected"]["T1"] is False and family["outcome"] == "iii"
    assert wilcoxon_signed_rank(s_values, [0.0] * len(s_values)).p_value < 1e-10


def test_a_flat_curve_gives_zero_contrasts_and_no_rejection() -> None:
    # Multiples of 1/64: every partial sum of the contrast is exact, so a flat curve gives exactly 0.0.
    family = cs.confirmatory_family(_levels(lambda rank, k, d: 100.0 + (d - 1000) / 64.0))
    assert family["S"]["mean"] == 0.0 and family["tests"]["T1"]["n_zero"] == 100
    assert family["tests"]["T1"]["p_value"] == 1.0 and family["outcome"] == "iii"


@pytest.mark.parametrize(("test", "k"), [("T2", 1), ("T3", 2)])
def test_a_gap_of_exactly_delta_does_not_reject_and_delta_plus_a_hundredth_does(test: str, k: int) -> None:
    """*Mutation:* the ``>`` boundary moved to ``>=`` -> the exactly-delta case dies."""
    bases = (0.0, 0.25)

    def exact(rank: int, level: int, d: int) -> float:
        base = bases[d % 2]
        return base + cs.DELTA if level == k else base

    levels = _levels(exact)
    assert all((levels[k][d] - levels[20][d]) - cs.DELTA == 0.0 for d in DRAWS), "the fixture's precondition"
    at_delta = cs.confirmatory_family(levels)["tests"][test]
    assert at_delta["alternative"] == "greater" and at_delta["k"] == k
    assert at_delta["n_zero"] == 100 and at_delta["p_value"] == 1.0

    above = _levels(lambda rank, level, d: 100.0 + (cs.DELTA + 0.01 if level == k else 0.0) + 0.001 * (d - 1000))
    assert all((above[k][d] - above[20][d]) - cs.DELTA > 0.0 for d in DRAWS)
    family = cs.confirmatory_family(above)
    assert family["tests"][test]["p_value"] < 0.05 / 3
    assert family["holm"]["rejected"][test] is True


def test_the_family_reports_s_and_the_two_gaps_with_their_cis_recomputed_here() -> None:
    rng = np.random.default_rng(404)
    noise = {(k, d): float(rng.normal(0.0, 0.8)) for k in LEVELS for d in DRAWS}
    levels = _levels(lambda rank, k, d: 101.0 - 0.3 * rank + noise[(k, d)])
    family = cs.confirmatory_family(levels)

    s_values = []
    for d in DRAWS:
        total = 0
        for coefficient, k in zip((-2, -1, 0, 1, 2), LEVELS):
            total = total + coefficient * levels[k][d]
        s_values.append(total)
    s = np.asarray(s_values, dtype=np.float64)
    assert family["S"]["mean"] == float(s.mean())
    assert family["S"]["ci95_half_width"] == 1.96 * float(s.std(ddof=1)) / math.sqrt(100)
    for test, k in (("T2", 1), ("T3", 2)):
        gaps = np.asarray([levels[k][d] - levels[20][d] for d in DRAWS], dtype=np.float64)
        entry = family["G"][str(k)]
        assert entry["mean"] == float(gaps.mean())
        assert entry["ci95_low"] == float(gaps.mean()) - 1.96 * float(gaps.std(ddof=1)) / math.sqrt(100)
        shifted = [(levels[k][d] - levels[20][d]) - cs.DELTA for d in DRAWS]
        assert family["tests"][test]["p_value"] == route_b(shifted, "greater")[4]
    assert family["tests"]["T1"]["p_value"] == route_b(s_values, "less")[4]
    p = tuple(family["tests"][name]["p_value"] for name in ("T1", "T2", "T3"))
    assert tuple(family["holm"]["rejected"][name] for name in ("T1", "T2", "T3")) == holm_by_hand(p)
    assert family["draw_ids"] == list(DRAWS) and family["n_draws"] == 100
    with pytest.raises(ValueError, match="draw"):
        cs.confirmatory_family({**levels, 5: {d: v for d, v in levels[5].items() if d != 1050}})


# ----------------------------------------------------------------------------------------------------------------------
# Holm and the outcome
# ----------------------------------------------------------------------------------------------------------------------


def test_holm_steps_down_as_registered() -> None:
    """*Mutation:* Holm replaced by the unadjusted alpha -> the second assertion dies."""
    assert cs.holm((0.01, 0.02, 0.04)) == (True, True, True)
    assert cs.holm((0.01, 0.03, 0.04)) == (True, False, False)
    assert holm_by_hand((0.01, 0.03, 0.04)) == (True, False, False)
    assert cs.holm((0.04, 0.01, 0.02)) == (True, True, True)
    assert cs.holm((0.06, 0.001, 0.001)) == (False, True, True)
    assert cs.holm((0.02, 0.02, 0.02)) == (False, False, False)
    grid = (0.0001, 0.01, 0.0166, 0.0167, 0.02, 0.025, 0.03, 0.05, 0.051, 0.5)
    for p in itertools.product(grid, repeat=3):
        assert cs.holm(p) == holm_by_hand(p), p


def test_the_outcome_partition_on_every_reject_pattern() -> None:
    for t1, t2, t3 in itertools.product((False, True), repeat=3):
        outcome = cs.outcome_of({"T1": t1, "T2": t2, "T3": t3})
        expected = "iii" if not t1 else ("i" if t2 and t3 else "ii")
        assert outcome == expected, (t1, t2, t3)


# ----------------------------------------------------------------------------------------------------------------------
# The registered sentences
# ----------------------------------------------------------------------------------------------------------------------


def _registration_row(marker: str) -> str:
    rows = [
        line for line in (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8").splitlines()
        if line.startswith(f"| 2026-09-28 | **{marker} — ")
    ]
    assert len(rows) == 1, marker
    return rows[0]


def _as_pattern(template: str) -> str:
    parts = re.split(r"\{[A-Za-z0-9_]+\}", template)
    return "(.*?)".join(re.escape(part) for part in parts)


def test_the_sentence_templates_are_the_registered_text_of_a26_and_a26_1() -> None:
    a26 = _registration_row("A26")
    a26_1 = _registration_row("A26.1")
    assert sorted(cs.SENTENCE_TEMPLATES) == ["i", "ii", "iii"]
    for outcome, template in cs.SENTENCE_TEMPLATES.items():
        assert template.startswith("Confirmatory (A26): on the P4 scenario, ")
        assert re.search(_as_pattern(template), a26), outcome
    assert re.search(_as_pattern(cs.SHORTFALL_CLAUSE_TEMPLATE), a26_1)
    assert cs.SHORTFALL_CLAUSE_TEMPLATE.startswith("K = {k} falls short of the K = 20 plateau")


def _family(t1: bool, t2: bool, t3: bool) -> dict:
    return {
        "S": {"mean": -1.23456, "ci95_low": -2.0, "ci95_high": -0.46912},
        "G": {
            "1": {"mean": 1.5, "ci95_low": 0.75, "ci95_high": 2.25},
            "2": {"mean": 0.3, "ci95_low": -0.1, "ci95_high": 0.7},
        },
        "holm": {"rejected": {"T1": t1, "T2": t2, "T3": t3}},
    }


def test_the_sentence_of_each_outcome_is_filled_from_the_family() -> None:
    s = f"{MINUS}1.2346 [{MINUS}2.0000, {MINUS}0.4691]"
    g1 = "1.5000 [0.7500, 2.2500]"
    g2 = f"0.3000 [{MINUS}0.1000, 0.7000]"
    assert cs.outcome_sentence("i", family=_family(True, True, True)) == (
        f"Confirmatory (A26): on the P4 scenario, performance improves with context length — the registered trend "
        f"contrast is {s} — and both context lengths DataLight tested (K = 1, 2) fall short of the K = 20 plateau by "
        f"more than the registered margin δ = 0.6263 s of ATT: K = 1 by {g1} and K = 2 by {g2}. Context length accounts "
        "for a difference of that size on this corpus — a measured difference in ATT, not an explanation of a failure."
    )
    assert cs.outcome_sentence("ii", family=_family(True, False, False)) == (
        f"Confirmatory (A26): on the P4 scenario, performance improves with context length — the registered trend "
        f"contrast is {s} — but it is not shown that K = 1 and K = 2 fall short of the K = 20 plateau by more than the "
        f"registered margin δ = 0.6263 s of ATT (K = 1: {g1}; K = 2: {g2}). Context length matters measurably on this "
        "corpus; a material shortfall is not demonstrated for K = 1 and K = 2."
    )
    assert cs.outcome_sentence("ii", family=_family(True, True, False)) == (
        f"Confirmatory (A26): on the P4 scenario, performance improves with context length — the registered trend "
        f"contrast is {s} — but it is not shown that K = 2 falls short of the K = 20 plateau by more than the "
        f"registered margin δ = 0.6263 s of ATT (K = 1: {g1}; K = 2: {g2}). K = 1 falls short of the K = 20 plateau by "
        "more than the registered margin δ = 0.6263 s of ATT. Context length matters measurably on this corpus; a "
        "material shortfall is not demonstrated for K = 2."
    )
    assert cs.outcome_sentence("ii", family=_family(True, False, True)) == (
        f"Confirmatory (A26): on the P4 scenario, performance improves with context length — the registered trend "
        f"contrast is {s} — but it is not shown that K = 1 falls short of the K = 20 plateau by more than the "
        f"registered margin δ = 0.6263 s of ATT (K = 1: {g1}; K = 2: {g2}). K = 2 falls short of the K = 20 plateau by "
        "more than the registered margin δ = 0.6263 s of ATT. Context length matters measurably on this corpus; a "
        "material shortfall is not demonstrated for K = 1."
    )
    assert cs.outcome_sentence("iii", family=_family(False, True, True)) == (
        f"Confirmatory (A26): on the P4 scenario, an improvement with context length is not detected — the registered "
        f"trend contrast is {s} — so this sweep gives no support, on this corpus, to context length as the explanation "
        "of DataLight's negative result."
    )
    with pytest.raises(ValueError, match="outcome"):
        cs.outcome_sentence("ii", family=_family(True, True, True))
    with pytest.raises(ValueError, match="outcome"):
        cs.outcome_sentence("iv", family=_family(True, True, True))
