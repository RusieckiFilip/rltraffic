"""P9.4 plan sections 5 and 6 (``docs/plans/p9.4_mix50_fresh.md``): the report's arithmetic against HAND-COMPUTED values
-- the paired CI, the Wilcoxon p, wins and losses, rho per draw and its CI -- through the report's own call sites, and
the branch every template S0-S13 selects.

The data are four draws and two seeds chosen so every quantity has a closed form.  The expected values are computed
here in plain Python (sums, the sample standard deviation, 1.96 * sd / sqrt(n), and the Wilcoxon normal approximation
with its continuity correction written out from ``dt_gate.wilcoxon_signed_rank``'s documented definition), never by
calling the function the report calls.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import offline.p9_4_report as rep

DRAWS = (1000, 1001, 1002, 1003)
SEEDS = (101, 202)
SEED_OFFSET = {101: 0.0, 202: 2.0}


def _dt_row(subject: str, arm: str, seed: int, draw: int, e_sumo: float, att_env: float) -> dict[str, Any]:
    return {"kind": "dt", "subject": subject, "arm": arm, "seed": seed, "draw_id": draw, "e_sumo": e_sumo,
            "att_env": att_env, "horizon_vehicle_count": 40.0, "episode_reward": -7000.0, "n_decisions_in_support": 300,
            "n_never_entered": 10 + seed % 7, "n_teleports": 0, "n_vanished_without_arrival": 0, "stage": "p9_4"}


def rows() -> list[dict[str, Any]]:
    out = []
    for k, draw in enumerate(DRAWS):
        for seed in SEEDS:
            out.append(_dt_row("mix50_fresh", "b_mean_k100", seed, draw, 500.0 + k + SEED_OFFSET[seed], 330.0 + k))
            out.append(_dt_row("mix50_fresh", "naive", seed, draw, 520.0 + 2 * k + SEED_OFFSET[seed], 331.0 + k))
    return out


def by_draw() -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for k, draw in enumerate(DRAWS):
        out[str(draw)] = {
            "fixedtime": {"e_sumo": 600.0 + 10 * k, "att_env": 350.0 + k, "horizon_vehicle_count": 1.0, "episode_reward": -1.0},
            "maxpressure": {"e_sumo": 400.0 + 5 * k, "att_env": 320.0 + k, "horizon_vehicle_count": 1.0, "episode_reward": -1.0},
            "fixedtime_kcf": {"e_sumo": 560.0 + 10 * k, "att_env": 352.0 + k, "horizon_vehicle_count": 1.0, "episode_reward": -1.0},
            "maxpressure_1to1": {"e_sumo": 380.0 + 5 * k, "att_env": 300.0 + k, "horizon_vehicle_count": 1.0,
                                 "episode_reward": -1.0},
        }
    return out


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def _wilcoxon_p_no_ties(differences: list[float]) -> float:
    """The documented normal approximation for distinct non-zero |d| (no tie correction needed)."""
    nonzero = [d for d in differences if d != 0.0]
    n = len(nonzero)
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    rank = {i: r + 1 for r, i in enumerate(order)}
    w_plus = sum(rank[i] for i in range(n) if nonzero[i] > 0)
    w_minus = sum(rank[i] for i in range(n) if nonzero[i] < 0)
    statistic = min(w_plus, w_minus)
    z = (statistic - n * (n + 1) / 4.0 + 0.5) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return min(1.0, 2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


# ----------------------------------------------------------------------
# R1-R3 through the report's pair block: hand-computed
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def registered() -> dict[str, Any]:
    return rep.pair_block(rows(), rep.anchors_for_pair(by_draw(), "registered"), rep.NEW_GROUPS)


@pytest.fixture(scope="module")
def aligned() -> dict[str, Any]:
    return rep.pair_block(rows(), rep.anchors_for_pair(by_draw(), "aligned"), rep.NEW_GROUPS)


def _entry(block: dict[str, Any], arm: str) -> dict[str, Any]:
    (entry,) = [e for e in block["by_subject_arm"] if e["subject"] == "mix50_fresh" and e["arm"] == arm]
    return entry


def test_r1_dt_minus_fixed_time_is_the_hand_computed_paired_difference(registered: dict[str, Any]) -> None:
    block = _entry(registered, "b_mean_k100")["paired_att"]["e_sumo"]["fixedtime"]
    # Per draw: seed mean of the DT (501 + k) minus fixed-time (600 + 10k) = -99 - 9k.
    differences = [-99.0 - 9.0 * k for k in range(4)]
    mean, low, high = _mean_ci(differences)
    assert block["mean_difference"] == pytest.approx(mean, rel=1e-12) and mean == -112.5
    assert (block["ci95_low"], block["ci95_high"]) == (pytest.approx(low, rel=1e-12), pytest.approx(high, rel=1e-12))
    assert (block["wins"], block["losses"], block["ties"], block["n_shared_draws"]) == (4, 0, 0, 4)
    assert block["wilcoxon_p"] == pytest.approx(_wilcoxon_p_no_ties(differences), rel=1e-12)
    assert block["right_arm"] == "fixedtime"


def test_r1_at_the_tuned_cycle_and_r2_lane_once_use_the_aligned_anchors(aligned: dict[str, Any]) -> None:
    entry = _entry(aligned, "b_mean_k100")["paired_att"]["e_sumo"]
    ft = entry["fixedtime"]
    mp = entry["maxpressure"]
    assert (ft["right_arm"], mp["right_arm"]) == ("fixedtime_kcf", "maxpressure_1to1")
    assert ft["mean_difference"] == pytest.approx(_mean_ci([-59.0 - 9.0 * k for k in range(4)])[0], rel=1e-12)
    # DT (501 + k) minus lane-once MaxPressure (380 + 5k) = 121 - 4k: every draw a loss.
    differences = [121.0 - 4.0 * k for k in range(4)]
    assert mp["mean_difference"] == pytest.approx(_mean_ci(differences)[0], rel=1e-12)
    assert (mp["wins"], mp["losses"]) == (0, 4)
    assert mp["wilcoxon_p"] == pytest.approx(_wilcoxon_p_no_ties(differences), rel=1e-12)


def test_r3_rho_is_the_per_draw_seed_mean_of_the_per_row_ratio(registered: dict[str, Any]) -> None:
    entry = _entry(registered, "b_mean_k100")["e_sumo"]
    per_draw = []
    for k in range(4):
        ft, mp = 600.0 + 10 * k, 400.0 + 5 * k
        per_draw.append(sum((ft - (500.0 + k + SEED_OFFSET[s])) / (ft - mp) for s in SEEDS) / 2)
    mean, low, high = _mean_ci(per_draw)
    assert entry["mean"] == pytest.approx(mean, rel=1e-12) and entry["n_draws"] == 4
    assert (entry["ci95_low"], entry["ci95_high"]) == (pytest.approx(low, rel=1e-12), pytest.approx(high, rel=1e-12))
    assert per_draw[0] == pytest.approx(99.0 / 200.0, rel=1e-15)


def test_r4_calibrated_minus_source_is_paired_per_draw(registered: dict[str, Any]) -> None:
    block = rep.calibrated_minus_source(registered["rows"], "e_sumo")
    # Per draw: (501 + k) - (521 + 2k) = -20 - k.
    differences = [-20.0 - k for k in range(4)]
    mean, low, high = _mean_ci(differences)
    assert block["mean_difference"] == pytest.approx(mean, rel=1e-12)
    assert (block["ci95_low"], block["ci95_high"]) == (pytest.approx(low, rel=1e-12), pytest.approx(high, rel=1e-12))
    assert (block["wins"], block["losses"], block["right_arm"]) == (4, 0, "naive")
    contrast = rep.contrast(registered)
    calibrated, naive = _entry(registered, "b_mean_k100")["e_sumo"]["mean"], _entry(registered, "naive")["e_sumo"]["mean"]
    assert contrast["by_subject"]["mix50_fresh"]["e_sumo"]["difference"] == pytest.approx(calibrated - naive, rel=1e-12)
    assert "calibrated - source <= 0" in rep.R4_DIRECTION and "calibrated rho >= source rho" in rep.R4_DIRECTION


def test_a_paired_comparison_of_two_models_pairs_seed_means_on_shared_draws() -> None:
    left = [_dt_row("mix50_fresh", "b_mean_k100", s, d, 100.0 + k + s / 1000.0, 0.0)
            for k, d in enumerate(DRAWS) for s in SEEDS]
    right = [_dt_row("mix50", "b_mean_k100", s, d, 103.0 + 2 * k, 0.0) for k, d in enumerate(DRAWS) for s in SEEDS]
    block = rep.paired(left, right, "e_sumo", left_label="corrected", right_label="original")
    seed_mean = sum(s / 1000.0 for s in SEEDS) / 2
    differences = [(100.0 + k + seed_mean) - (103.0 + 2 * k) for k in range(4)]
    assert block["mean_difference"] == pytest.approx(_mean_ci(differences)[0], rel=1e-12)
    assert (block["left_arm"], block["right_arm"], block["wins"], block["losses"]) == ("corrected", "original", 4, 0)
    with pytest.raises(ValueError, match="draw"):
        rep.paired(left, [r for r in right if r["draw_id"] != 1003], "e_sumo", left_label="a", right_label="b")


def test_the_registered_pair_reproduces_the_p9_2_anchor_rows_it_was_built_from() -> None:
    anchors = rep.anchors_for_pair(by_draw(), "aligned")
    assert anchors[1002]["fixedtime"]["arm"] == "fixedtime_kcf" and anchors[1002]["fixedtime"]["e_sumo"] == 580.0
    assert anchors[1002]["maxpressure"]["arm"] == "maxpressure_1to1" and anchors[1002]["maxpressure"]["draw_id"] == 1002
    with pytest.raises(ValueError, match="pair"):
        rep.anchors_for_pair(by_draw(), "mp1to1_ftk4")


# ----------------------------------------------------------------------
# Readings and formatting
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("low", "high", "reading"), [
    (-3.0, -0.5, "below"), (-3.0, 0.0, "contains"), (0.0, 2.0, "contains"), (-1.0, 1.0, "contains"), (0.25, 2.0, "above"),
])
def test_a_difference_reading_is_the_side_of_zero_the_whole_interval_lies_on(low: float, high: float, reading: str) -> None:
    assert rep.difference_reading(low, high) == reading


@pytest.mark.parametrize(("low", "high", "reading"), [
    (1.01, 1.5, "above_1"), (1.0, 1.5, "contains_1_above_0"), (0.5, 1.2, "contains_1_above_0"),
    (0.2, 0.8, "inside_0_1"), (-0.2, 0.8, "contains_0_below_1"), (0.0, 0.8, "contains_0_below_1"),
    (-0.1, 1.0, "contains_0_and_1"), (-0.5, -0.01, "below_0"),
])
def test_a_rho_reading_is_one_of_the_six_branches(low: float, high: float, reading: str) -> None:
    assert rep.rho_reading(low, high) == reading


def test_formatting_keeps_every_printed_interval_on_its_branchs_side() -> None:
    assert rep.fmt_interval(-3.21, -0.04, unit="s") == ("-3.21", "-0.04")
    assert rep.fmt_interval(-12.34, -5.67, unit="s") == ("-12.3", "-5.7")
    assert rep.fmt_interval(-0.04, 0.0, unit="s") == ("-0.04", "0.00")
    assert rep.fmt_interval(1.004, 1.3, unit="rho") == ("1.004", "1.300")
    assert rep.fmt_estimate(-0.03, -0.06, -0.01, unit="s") == ("-0.03", "-0.06", "-0.01")
    assert rep.fmt_number(-0.04, unit="s") == "-0.04" and rep.fmt_number(-0.0, unit="s") == "0.0"
    assert rep.fmt_p(0.0123) == "0.012" and rep.fmt_p(3.2e-05) == "3.2e-05" and rep.fmt_p(1.0) == "1.000"


# ----------------------------------------------------------------------
# The sentences: each template's branch chosen by the numbers, its slots filled
# ----------------------------------------------------------------------


def _paired(mean: float, low: float, high: float, wins: int, losses: int, p: float = 0.0123) -> dict[str, Any]:
    return {"mean_difference": mean, "ci95_low": low, "ci95_high": high, "wins": wins, "losses": losses,
            "ties": 100 - wins - losses, "wilcoxon_p": p}


def _rho(mean: float, low: float, high: float) -> dict[str, Any]:
    return {"mean": mean, "ci95_low": low, "ci95_high": high}


def test_s0_is_the_plans_text() -> None:
    assert rep.sentence_s0()["text"].startswith("The random half of the 50 % mixture came from the original random")
    assert rep.sentence_s0()["text"].endswith("We report the retrained model as exploratory and the original one in the "
                                              "supplement.")


def test_s1_and_s2_choose_faster_unchanged_or_slower_from_the_interval() -> None:
    faster = rep.sentence_s1(_paired(-20.0, -30.0, -10.0, 90, 10))
    assert faster["branch"] == "below" and faster["text"] == (
        "Transferred zero-shot, the DT from the corrected mixture was faster than SUMO's fixed-time plan on 90 of 100 "
        "draws: DT − fixed-time -20.0 s [-30.0, -10.0] (paired Wilcoxon signed-rank, p = 0.012).")
    same = rep.sentence_s1(_paired(-2.0, -5.0, 1.0, 55, 45, 0.31))
    assert same["branch"] == "contains" and "did not differ measurably from SUMO's fixed-time plan" in same["text"]
    slower = rep.sentence_s2(_paired(12.0, 4.0, 20.0, 30, 70))
    assert slower["branch"] == "above" and slower["text"].startswith(
        "Transferred zero-shot, the DT from the corrected mixture was slower than fixed-time at CityFlow's tuned cycle "
        "on 70 of 100 draws")


def test_s3_and_s4_name_the_maxpressure_they_compare_with() -> None:
    s3 = rep.sentence_s3(_paired(-8.0, -12.0, -4.0, 80, 20))
    assert s3["text"] == ("It was also faster than the registered MaxPressure: DT − MaxPressure -8.0 s [-12.0, -4.0], "
                          "on 80 of 100 draws.")
    s4 = rep.sentence_s4(_paired(5.0, 1.0, 9.0, 35, 65))
    assert s4["branch"] == "above" and s4["text"] == (
        "It was slower than the MaxPressure that counts each lane once: DT − MaxPressure 5.0 s [1.0, 9.0], on 65 of "
        "100 draws.")
    assert rep.sentence_s3(_paired(0.5, -1.0, 2.0, 50, 50))["text"] == (
        "It did not differ measurably from the registered MaxPressure: DT − MaxPressure 0.5 s [-1.0, 2.0].")


@pytest.mark.parametrize(("interval", "branch", "ending"), [
    ((1.3, 1.1, 1.5), "above_1", "under the registered anchors, beyond MaxPressure's 1."),
    ((0.95, 0.85, 1.05), "contains_1_above_0", "under the registered anchors, not distinguishable from MaxPressure's 1."),
    ((0.5, 0.3, 0.7), "inside_0_1", "under the registered anchors, between fixed-time (0) and MaxPressure (1)."),
    ((0.05, -0.1, 0.2), "contains_0_below_1", "under the registered anchors, not distinguishable from fixed-time's 0."),
    ((0.5, -0.2, 1.2), "contains_0_and_1", "under the registered anchors, not distinguishable from either anchor."),
    ((-0.3, -0.5, -0.1), "below_0", "under the registered anchors, below fixed-time's 0."),
])
def test_s5_has_six_branches_and_s6_names_the_aligned_anchors(interval: tuple[float, float, float], branch: str,
                                                               ending: str) -> None:
    s5 = rep.sentence_s5(_rho(*interval))
    assert s5["branch"] == branch and s5["text"].endswith(ending) and s5["text"].startswith("Its score was ρ = ")
    s6 = rep.sentence_s6(_rho(*interval))
    assert s6["branch"] == branch and "under the aligned anchors, which count each lane once and use CityFlow's tuned " \
                                      "cycle" in s6["text"]


def test_s5_fills_rho_and_its_interval() -> None:
    assert rep.sentence_s5(_rho(1.234, 1.1, 1.367))["text"] == (
        "Its score was ρ = 1.23 [1.10, 1.37] under the registered anchors, beyond MaxPressure's 1.")


def test_s7_reads_the_registered_hypothesis_from_the_rho_interval() -> None:
    both = rep.sentence_s7(_rho(0.5, 0.3, 0.7), anchors="registered")
    assert both["text"] == ("Read against the registered transfer hypothesis, the corrected model is better than "
                            "fixed-time and is worse than MaxPressure under the registered anchors.")
    neither = rep.sentence_s7(_rho(1.3, 1.1, 1.5), anchors="aligned")
    assert neither["text"] == ("Read against the registered transfer hypothesis, the corrected model is better than "
                               "fixed-time and is not worse than MaxPressure under the aligned anchors.")
    assert rep.sentence_s7(_rho(0.05, -0.1, 0.2), anchors="registered")["text"].count("is not") == 1
    assert (both["template"], neither["template"]) == ("S7", "S7b")


def test_s8_and_s9_have_three_branches_without_counts() -> None:
    assert rep.sentence_s8(_paired(-3.0, -5.0, -1.0, 70, 30))["text"] == (
        "The calibrated prompt made the model faster than its source prompt: calibrated − source -3.0 s [-5.0, -1.0].")
    assert rep.sentence_s8(_paired(3.0, 1.0, 5.0, 30, 70))["text"].startswith("The source prompt made the model faster")
    assert rep.sentence_s9(_paired(0.2, -1.0, 1.4, 50, 50))["text"] == (
        "Correcting the random half did not change the transferred DT measurably: corrected − original 0.2 s "
        "[-1.0, 1.4].")
    assert rep.sentence_s9(_paired(-4.0, -6.0, -2.0, 80, 20))["branch"] == "below"


def test_s10_compares_the_four_readings_of_both_models() -> None:
    same = {"S1": "below", "S3": "above", "S5": "inside_0_1", "S6": "contains_1_above_0"}
    assert rep.sentence_s10(same, dict(same))["text"] == (
        "Every reading of the transfer test is the same for the corrected and the original model.")
    changed = rep.sentence_s10({**same, "S5": "contains_1_above_0"}, {**same, "S5": "above_1"})
    assert changed["branch"] == "otherwise"
    assert changed["text"] == ("The correction changes ρ under the registered anchors: beyond MaxPressure's 1 "
                               "becomes not distinguishable from MaxPressure's 1.")


def test_s11_s12_and_s12b_read_cityflow_on_att_engine_with_att_ours_beside() -> None:
    s11 = rep.sentence_s11(_rho(105.33, 103.0, 107.7), never_entered=1.25)
    assert s11["text"] == ("In CityFlow, on the same 100 held-out draws, the DT from the corrected mixture had an average "
                           "travel time of 105.3 s [103.0, 107.7], with 1.2 vehicles per episode that never entered.")
    s12 = rep.sentence_s12(_paired(-1.5, -2.5, -0.5, 70, 30))
    assert s12["text"] == ("In CityFlow the corrected model was faster than the original one: corrected − original "
                           "-1.5 s [-2.5, -0.5], on 70 of 100 draws.")
    assert rep.sentence_s12b(_paired(-1.5, -2.5, -0.5, 70, 30), _paired(-1.0, -2.0, 0.1, 60, 40))["text"] == (
        "Timed from entry instead of from creation, the reading changes: corrected − original -1.0 s [-2.0, 0.1].")
    assert rep.sentence_s12b(_paired(-1.5, -2.5, -0.5, 70, 30), _paired(-1.0, -2.0, -0.1, 60, 40)) is None


def test_s13_compares_with_the_dt_trained_offline_in_sumo() -> None:
    s13 = rep.sentence_s13(_paired(-6.0, -9.0, -3.0, 75, 25))
    assert s13["text"] == ("Transferred zero-shot, the DT from the corrected mixture was faster than the same recipe "
                           "trained offline in SUMO on 200 MaxPressure episodes: transferred − SUMO-trained -6.0 s "
                           "[-9.0, -3.0], on 75 of 100 draws.")
    assert rep.sentence_s13(_paired(6.0, 3.0, 9.0, 25, 75))["text"].endswith("on 75 of 100 draws.")
    assert rep.sentence_s13(_paired(1.0, -1.0, 3.0, 50, 50))["branch"] == "contains"
