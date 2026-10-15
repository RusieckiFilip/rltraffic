"""P9.1's verdicts (``offline/corrective_verdicts.py``; ``BRIEF_46`` 3.4 ``rescore``, 4.5-4.6; A27(b)-(e), (g)).

Every rule here is a REGISTERED rule applied to new cells, imported, never re-implemented: P5.2c's paired contrast
(``tier_statements.contrast``: per-draw means across the seeds, ``dt_gate.mean_ci95``) and its HELD / FAILED / NOT
RESOLVED interval rule (``tier_statements.interval_outcome``, ``conjunction``), P5.2c's statement scorers
(``all_statements``) for the re-scored grid statements, Q6's two-valued collapse vocabulary.  Primary definition
``att_engine``, ``att_ours`` beside (A27(a)).  Alignment: a contrast pairs two cells on the identical ``(seed, draw)``
grid and is refused otherwise.

The expected values are recomputed here by a second route (per-draw means by hand on dyadic values, where every route is
exact), never by calling the function under test.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import offline.corrective_verdicts as cv
import offline.tier_statements as tst
from tests import p5_2c_fixtures as fx

SEEDS = (101, 202, 303)
DRAWS = (1000, 1001, 1002, 1003)


def _cell(method: str, tier: str, level: float, *, wobble: float = 0.25, engine_shift: float = -7.0,
          per_draw: Mapping[int, float] | None = None) -> tst.Cell:
    """A cell at *level* with a dyadic per-episode term (exact in binary); ``att_engine`` = ``att_ours`` + shift."""
    ours, engine = {}, {}
    for s in SEEDS:
        for d in DRAWS:
            value = level + wobble * (((s // 101) + 2 * (d - 1000)) % 4 - 1.5) + (per_draw or {}).get(d, 0.0)
            ours[(s, d)] = value
            engine[(s, d)] = value + engine_shift
    return fx.simple_cell(method, tier, ours, engine, source="p9_1")


def _own_interval(left: tst.Cell, right: tst.Cell, definition: str) -> tuple[float, float, float]:
    """mean and 95 % interval of the per-draw differences of per-draw means across the seeds (P5.2c's unit)."""
    def by_draw(cell: tst.Cell) -> dict[int, float]:
        out: dict[int, list[float]] = {}
        for row in cell.rows:
            out.setdefault(int(row["draw_id"]), []).append(float(row[definition]))
        return {d: math.fsum(v) / len(v) for d, v in out.items()}
    a, b = by_draw(left), by_draw(right)
    diffs = [a[d] - b[d] for d in sorted(a)]
    mean = math.fsum(diffs) / len(diffs)
    half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return mean, mean - half, mean + half


def _own_outcome(low: float, high: float, predicted: str) -> str:
    if predicted == "negative":
        return "HELD" if high < 0 else "FAILED" if low > 0 else "NOT RESOLVED"
    return "HELD" if low > 0 else "FAILED" if high < 0 else "NOT RESOLVED"


# ----------------------------------------------------------------------
# A27(e): the four ordering contrasts and the 5-s check
# ----------------------------------------------------------------------


def _mixture(levels: Mapping[str, float]) -> dict[str, tst.Cell]:
    return {arm: _cell(arm, "mix_m1000_m500", level) for arm, level in levels.items()}


def test_the_four_registered_contrasts_each_predicted_negative_under_att_engine_with_att_ours_beside() -> None:
    cells = _mixture({"bc_top10": 103.0, "iql": 104.5, "dt": 106.0, "bc": 108.0, "iql_t": 104.0})
    block = cv.mixture_orderings(cells, tier="mix_m1000_m500")
    assert [(c["left_arm"], c["right_arm"]) for c in block["contrasts"]] == [
        ("bc_top10", "bc"), ("bc_top10", "iql"), ("iql", "dt"), ("dt", "bc")]
    for entry in block["contrasts"]:
        left, right = cells[entry["left_arm"]], cells[entry["right_arm"]]
        for definition in ("att_engine", "att_ours"):
            mean, low, high = _own_interval(left, right, definition)
            got = entry["by_definition"][definition]
            assert got["mean_difference"] == pytest.approx(mean, abs=1e-12)
            assert got["outcome"] == _own_outcome(low, high, "negative") == "HELD"
        assert entry["verdict"] == entry["by_definition"]["att_engine"]["outcome"]
        assert entry["predicted"] == "negative"


def test_a_reversed_pair_fails_and_only_that_pair() -> None:
    cells = _mixture({"bc_top10": 103.0, "iql": 106.5, "dt": 104.0, "bc": 108.0})
    verdicts = {(c["left_arm"], c["right_arm"]): c["verdict"] for c in cv.mixture_orderings(
        cells, tier="mix_m1000_m500")["contrasts"]}
    assert verdicts == {("bc_top10", "bc"): "HELD", ("bc_top10", "iql"): "HELD", ("iql", "dt"): "FAILED",
                        ("dt", "bc"): "HELD"}


@pytest.mark.parametrize(("gap", "expected"), [
    (2.0, "HELD"), (-2.0, "HELD"), (8.0, "FAILED"), (-8.0, "FAILED"), (5.0, "NOT RESOLVED"), (-5.0, "NOT RESOLVED"),
])
def test_within_five_seconds_is_the_two_sided_window_by_the_interval_rule(gap: float, expected: str) -> None:
    """``within 5 s`` read literally: |arm - bc_top10| < 5 s, three-valued by P5.2c's interval rule on both edges --
    HELD iff the interval lies strictly inside (-5, +5), FAILED iff it lies entirely outside, NOT RESOLVED otherwise."""
    cells = _mixture({"bc_top10": 103.0, "dt": 103.0 + gap, "bc": 103.5, "iql": 103.25})
    block = cv.within_seconds(cells, tier="mix_m1000_m500")
    by_arm = {entry["arm"]: entry for entry in block["arms"]}
    mean, low, high = _own_interval(cells["dt"], cells["bc_top10"], "att_engine")
    assert by_arm["dt"]["by_definition"]["att_engine"]["mean_difference"] == pytest.approx(mean, abs=1e-12)
    inside = -5.0 < low and high < 5.0
    outside = high < -5.0 or low > 5.0
    assert by_arm["dt"]["verdict"] == ("HELD" if inside else "FAILED" if outside else "NOT RESOLVED") == expected
    assert [entry["arm"] for entry in block["arms"]] == ["dt", "bc", "iql"]
    assert block["verdict"] == tst.conjunction([entry["verdict"] for entry in block["arms"]])


# ----------------------------------------------------------------------
# A27(g): the paired 100k - 40k differences and the concordance
# ----------------------------------------------------------------------


def _budget(levels: Mapping[str, float], *, tier: str = "mappo1000") -> dict[str, tst.Cell]:
    return {arm: _cell(arm, tier, level) for arm, level in levels.items()}


def test_each_arm_is_paired_100k_minus_40k_on_the_same_draws() -> None:
    at_40k = _budget({"dt": 105.0, "bc": 106.0, "bc_top10": 103.0, "iql": 104.0})
    at_100k = _budget({"dt": 104.5, "bc": 106.25, "bc_top10": 103.0, "iql": 103.0})
    block = cv.budget_block(at_40k, at_100k, tier="mappo1000")
    for entry in block["differences"]:
        mean, low, high = _own_interval(at_100k[entry["arm"]], at_40k[entry["arm"]], "att_engine")
        got = entry["by_definition"]["att_engine"]
        assert got["mean_difference"] == pytest.approx(mean, abs=1e-12)
        assert (got["ci95_low"], got["ci95_high"]) == pytest.approx((low, high), abs=1e-12)
    assert [e["arm"] for e in block["differences"]] == ["dt", "bc", "bc_top10", "iql"]


def test_the_concordance_judges_each_pairs_100k_interval_against_its_40k_order() -> None:
    at_40k = _budget({"dt": 105.0, "bc": 107.0, "bc_top10": 103.0, "iql": 104.0})
    same = cv.budget_block(at_40k, _budget({"dt": 105.5, "bc": 107.5, "bc_top10": 103.5, "iql": 104.5}),
                           tier="mappo1000")["concordance"]["att_engine"]
    assert len(same["pairs"]) == 6 and same["n_held"] == 6 and same["verdict"] == "HELD"
    swapped = cv.budget_block(at_40k, _budget({"dt": 103.0, "bc": 107.0, "bc_top10": 105.0, "iql": 104.0}),
                              tier="mappo1000")["concordance"]["att_engine"]
    outcomes = {(p["left_arm"], p["right_arm"]): p["outcome"] for p in swapped["pairs"]}
    assert outcomes[("dt", "bc_top10")] == "FAILED" and swapped["verdict"] == "FAILED"
    for pair in swapped["pairs"]:
        mean40, _, _ = _own_interval(at_40k[pair["left_arm"]], at_40k[pair["right_arm"]], "att_engine")
        assert pair["predicted"] == ("negative" if mean40 < 0 else "positive")


def test_a_pair_tied_exactly_at_40k_has_no_registered_order_and_is_not_resolved() -> None:
    at_40k = _budget({"dt": 104.0, "bc": 107.0, "bc_top10": 103.0, "iql": 104.0})
    at_100k = _budget({"dt": 103.0, "bc": 107.0, "bc_top10": 102.0, "iql": 105.0})
    pairs = {(p["left_arm"], p["right_arm"]): p for p in cv.budget_block(
        at_40k, at_100k, tier="mappo1000")["concordance"]["att_engine"]["pairs"]}
    tie = pairs[("dt", "iql")]
    assert tie["predicted"] is None and tie["outcome"] == "NOT RESOLVED" and "exact tie" in tie["note"]


# ----------------------------------------------------------------------
# A27(c): iql_t - iql per tier
# ----------------------------------------------------------------------


def test_iql_t_minus_iql_is_paired_per_tier_and_counts_the_intervals_excluding_zero() -> None:
    pairs = {"random": (_cell("iql_t", "random", 190.0), _cell("iql", "random", 198.5)),
             "maxpressure": (_cell("iql_t", "maxpressure", 199.0, wobble=0.5), _cell("iql", "maxpressure", 199.0))}
    block = cv.paired_differences(pairs, left="iql_t", right="iql")
    by_tier = {entry["tier"]: entry for entry in block["tiers"]}
    for tier, (left, right) in pairs.items():
        mean, low, high = _own_interval(left, right, "att_engine")
        got = by_tier[tier]["by_definition"]["att_engine"]
        assert got["mean_difference"] == pytest.approx(mean, abs=1e-12)
        assert got["excludes_zero"] == (high < 0.0 or low > 0.0)
    assert block["n_excluding_zero"]["att_engine"] == sum(
        by_tier[t]["by_definition"]["att_engine"]["excludes_zero"] for t in pairs)
    assert block["range"]["att_engine"] == [min(by_tier[t]["by_definition"]["att_engine"]["mean_difference"] for t in pairs),
                                            max(by_tier[t]["by_definition"]["att_engine"]["mean_difference"] for t in pairs)]


# ----------------------------------------------------------------------
# moved: the registered rule's decision, original book against substituted book
# ----------------------------------------------------------------------


@pytest.fixture
def books() -> dict[str, dict[tuple[str, str], tst.Cell]]:
    return tst.make_books(fx.synthetic_cells())


def test_the_tautology_guard_substituting_every_original_for_itself_moves_nothing(books: Any) -> None:
    original = tst.all_statements(books)
    same = cv.rescore_p5_2(books, {key: books["corrected"][key] for key in cv.A27B_KEYS})
    assert len(cv.A27B_KEYS) == 16
    block = cv.moved_block(original, same)
    assert block and all(entry["moved"] is False for entry in block)
    assert all(entry["reading_changed"] is False for entry in block)


def test_moved_is_true_exactly_where_the_registered_verdict_differs(books: Any) -> None:
    original = tst.all_statements(books)
    worse = {("bc_top10", "random"): fx.simple_cell("bc_top10", "random", {
        (s, d): 150.0 + fx._term("x", s, d) for s in fx.SEEDS for d in fx.DRAWS}, source="p9_1",
        arm="bc_top10_m@random")}
    rescored = cv.rescore_p5_2(books, worse)
    block = cv.moved_block(original, rescored)
    index = {(e["id"], e["tier"], e["arm"], e["definition"]): e for e in original
             if e["version"] in ("both", "corrected")}
    moved = [entry for entry in block if entry["moved"]]
    assert moved, "a %BC cell moved from 350 s to 150 s on random must move a registered verdict"
    for entry in block:
        before = index[(entry["id"], entry["tier"], entry["arm"], entry["definition"])]
        assert entry["original_verdict"] == before["verdict"]
        assert entry["moved"] == (before["kind"] == "verdict" and entry["original_verdict"] != entry["verdict"])


def test_moved_refuses_two_statement_sets_that_do_not_match(books: Any) -> None:
    original = tst.all_statements(books)
    with pytest.raises(ValueError, match="do not score the same statements"):
        cv.moved_block(original, [e for e in original if e["id"] != "Q6"])


# ----------------------------------------------------------------------
# A27(d): the collapse counts against both random references, and P3a
# ----------------------------------------------------------------------


def test_collapse_is_q6s_two_valued_rule_against_each_reference_and_counted_per_reference() -> None:
    arms = {"dt_nomix": _cell("dt_nomix", "random_fresh", 250.0), "iql_m": _cell("iql_m", "random_fresh", 190.0),
            "bc_m": _cell("bc_m", "random_fresh", 262.0)}
    anchors = {"registered_five_schedule": _cell("random", "mappo1000", 255.0),
               "random_fresh": _cell("behaviour", "random_fresh", 245.0)}
    block = cv.collapse_rows(arms, anchors, tier="random_fresh")
    for row in block["rows"]:
        _, low, _ = _own_interval(arms[row["arm"]], anchors[row["anchor"]], "att_engine")
        assert row["by_definition"]["att_engine"]["verdict"] == ("COLLAPSED" if low > 0.0 else "NOT COLLAPSED")
    counts = block["collapsed_count"]["att_engine"]
    assert counts == {name: sum(r["by_definition"]["att_engine"]["verdict"] == "COLLAPSED" for r in block["rows"]
                                if r["anchor"] == name) for name in anchors}
    assert counts["random_fresh"] == 2 and counts["registered_five_schedule"] == 1


def _p3a_row(seed: int, draw: int, joint: str, columns: Sequence[str]) -> dict[str, Any]:
    return {"seed": seed, "draw_id": draw, "att_ours": 1.0, "n_decisions": 360, "action_sequence_sha256": joint,
            "per_intersection_sha256": list(columns)}


def test_p3a_counts_an_episode_repeated_if_its_seed_emits_its_sequence_on_another_draw() -> None:
    rows = [_p3a_row(101, d, f"j{d}", [f"a{d}"]) for d in DRAWS]
    rows += [_p3a_row(202, 1000, "same", ["c"]), _p3a_row(202, 1001, "x", ["d"]), _p3a_row(202, 1002, "same", ["c"]),
             _p3a_row(202, 1003, "y", ["e"])]
    rows += [_p3a_row(303, d, f"j{d}", [f"a{d}"]) for d in DRAWS]
    block = cv.p3a(rows, per_intersection=False)
    assert block["repeated_episodes"] == 2 and block["repeated"] == [[202, 1000], [202, 1002]]
    assert block["episodes"] == 12 and block["verdict"] == "FAILED"
    assert cv.p3a([r for r in rows if r["seed"] != 202], per_intersection=False)["verdict"] == "HELD"


def test_on_the_grid_one_intersections_repeat_is_enough() -> None:
    rows = [_p3a_row(101, 1000, "j0", ["a", "b", "c"]), _p3a_row(101, 1001, "j1", ["a2", "b2", "c"]),
            _p3a_row(101, 1002, "j2", ["a3", "b3", "c3"])]
    joint = cv.p3a(rows, per_intersection=False)
    grid = cv.p3a(rows, per_intersection=True)
    assert joint["repeated_episodes"] == 0 and joint["verdict"] == "HELD"
    assert grid["repeated_episodes"] == 2 and grid["repeated"] == [[101, 1000], [101, 1001]]
    assert grid["verdict"] == "FAILED"


def test_the_beside_table_counts_distinct_sequences_the_most_common_share_and_the_identities() -> None:
    dt = [_p3a_row(101, 1000, "s", ["s"]), _p3a_row(101, 1001, "s", ["s"]), _p3a_row(101, 1002, "t", ["t"]),
          _p3a_row(101, 1003, "r", ["r"])]
    references = {"registered_five_schedule": [_p3a_row(101, d, "s" if d == 1001 else f"ref{d}", ["z"]) for d in DRAWS],
                  "random_fresh": [_p3a_row(101, d, f"fresh{d}", ["z"]) for d in DRAWS]}
    training = {"joint": {"t", "q"}, "per_intersection": [{"t", "q"}]}
    table = cv.p3a_beside(dt, references=references, training=training)
    row = table["per_seed"]["101"]
    assert row["distinct_sequences"] == 3 and row["most_common_share"] == 0.5
    assert row["identical_to_reference"] == {"registered_five_schedule": 1, "random_fresh": 0}
    assert row["replays_a_training_schedule"] == 1


# ----------------------------------------------------------------------
# A27(c): P4.4's A6 statement re-scored with iql_t (the registered route, reproduced first)
# ----------------------------------------------------------------------


def _committed_cell(arm: str) -> tst.Cell:
    import json
    from pathlib import Path

    payload = json.loads((Path(__file__).resolve().parents[1] / "docs/data/p4_4_baselines.json").read_text())
    values = {(int(e["seed"]), int(e["draw_id"])): {
        "att_ours": float(e["att_horizon"]), "att_engine": float(e["att_horizon"]), "created": 1, "entered": 1,
        "never_entered": 0, "entered_fraction": 1.0, "horizon_vehicle_count": float(e["horizon_vehicle_count"]),
        "episode_reward": float(e["episode_reward"])} for e in payload["episodes"] if e["arm"] == arm}
    return fx.cell(fx.spec(arm, "mappo1000", source="p4_4", arm=arm), values)


def test_the_a6_route_reproduces_p4_4s_committed_madt_vs_iql_exactly() -> None:
    """The registered route on the committed episodes gives the committed numbers under == (att_ours = att_horizon)."""
    import json
    from pathlib import Path

    committed = json.loads((Path(__file__).resolve().parents[1] / "docs/data/p4_4_baselines.json").read_text())
    registered = committed["comparisons"]["madt_vs_iql"]
    block = cv.a6_block(_committed_cell("madt"), _committed_cell("iql"), delta=float(registered["delta"]))
    ours = block["by_definition"]["att_ours"]
    assert (ours["mean_difference"], ours["ci95_low"], ours["ci95_high"]) == (
        registered["mean_difference"], registered["ci95_low"], registered["ci95_high"])
    assert ours["verdict"] == registered["verdict"] == "baseline_genuinely_better"
    assert ours["verdict_at_full_precision_delta"] == registered["verdict_at_full_precision_delta"]
    assert block["delta"] == 0.6263


def test_the_a6_rescore_moves_when_the_baselines_verdict_changes() -> None:
    dt = _cell("madt", "mappo1000", 105.0)
    same = cv.a6_rescore(dt, _cell("iql", "mappo1000", 103.0), _cell("iql_t", "mappo1000", 103.25), delta=0.6263)
    assert same["moved"] == {"att_engine": False, "att_ours": False}
    moved = cv.a6_rescore(dt, _cell("iql", "mappo1000", 103.0), _cell("iql_t", "mappo1000", 105.25), delta=0.6263)
    assert moved["original"]["by_definition"]["att_ours"]["verdict"] == "baseline_genuinely_better"
    assert moved["rescored"]["by_definition"]["att_ours"]["verdict"] == "matches"
    assert moved["moved"] == {"att_engine": True, "att_ours": True}


# ----------------------------------------------------------------------
# Added after the C6b mutation table (V3, V12, V13, V17 survived on fixtures that could not tell them apart)
# ----------------------------------------------------------------------


def test_the_verdict_is_att_engines_where_the_two_definitions_disagree() -> None:
    """V3: att_ours says dt is worse than bc, att_engine says better; the verdict is att_engine's."""
    cells = _mixture({"bc_top10": 103.0, "iql": 104.5, "bc": 108.0})
    cells["dt"] = _cell("dt", "mix_m1000_m500", 109.0, engine_shift=-9.0)
    dt_bc = next(c for c in cv.mixture_orderings(cells, tier="mix_m1000_m500")["contrasts"]
                 if (c["left_arm"], c["right_arm"]) == ("dt", "bc"))
    assert dt_bc["by_definition"]["att_ours"]["outcome"] == "FAILED"
    assert dt_bc["by_definition"]["att_engine"]["outcome"] == "HELD" == dt_bc["verdict"]


def test_an_interval_entirely_above_zero_also_excludes_zero() -> None:
    """V12: iql_t worse than iql with the interval above zero counts as excluding zero."""
    pairs = {"fixedtime": (_cell("iql_t", "fixedtime", 210.0), _cell("iql", "fixedtime", 204.0))}
    block = cv.paired_differences(pairs, left="iql_t", right="iql")
    assert block["tiers"][0]["by_definition"]["att_engine"]["ci95_low"] > 0.0
    assert block["n_excluding_zero"] == {"att_engine": 1, "att_ours": 1}


def test_an_empty_substitution_rescored_is_the_registered_book_itself(books: Any) -> None:
    """V13: the registered book is P5.2c's ``corrected`` one (P5.2b's random-tier IQL cell); with nothing substituted,
    every statement's verdict and reading come back unchanged."""
    block = cv.moved_block(tst.all_statements(books), cv.rescore_p5_2(books, {}))
    assert not any(entry["moved"] or entry["reading_changed"] for entry in block)
    assert any(tst.CORRECTED_IQL in entry["inputs"] for entry in block)


def test_a_positive_mean_whose_interval_includes_zero_is_not_collapsed() -> None:
    """V17: Q6's rule reads the interval's lower end, not the mean."""
    arm = _cell("dt_nomix", "random_fresh", 250.0, per_draw={1000: 3.0, 1001: -2.5, 1002: 2.0, 1003: -1.5})
    anchor = _cell("random", "mappo1000", 250.0)
    row = cv.collapse_rows({"dt_nomix": arm}, {"registered_five_schedule": anchor}, tier="random_fresh")["rows"][0]
    numbers = row["by_definition"]["att_engine"]
    assert numbers["mean_difference"] > 0.0 and numbers["ci95_low"] <= 0.0
    assert row["verdict"] == "NOT COLLAPSED"
