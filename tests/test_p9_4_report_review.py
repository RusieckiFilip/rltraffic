"""P9.4's report after the 2026-11-19 review (review C; ``docs/returns/P9.4.md``): the printed numbers, R5's readings as
plan section 5 defines them, the wiring the earlier fixture could not see (its paired intervals all had zero width),
``load_sources``' refusals, and the artifacts' atomic writes.

Every expected number here is computed in plain Python or typed out, never by the function under test.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

import offline.att_rederivation as ar
import offline.corrective_report as crep
import offline.method_tier_grid as grid
import offline.p9_4_evaluation as ev
import offline.p9_4_mix50_fresh as p94
import offline.p9_4_report as rep
import offline.transfer_calibration as tcal
from tests.test_p9_4_report_build import DRAWS, OFFSET, SEEDS, _f, sources

REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# The printed numbers
# ----------------------------------------------------------------------


def test_printed_values_keep_their_sides_their_order_and_their_distinctness() -> None:
    # Review C: -0.05 and -0.15 both printed "-0.1" at one decimal, the mean ON its interval's end.
    assert rep.fmt_estimate(-0.05, -0.15, 0.05, unit="s") == ("-0.05", "-0.15", "0.05")
    # Review C: the six-decimal cap printed [-2.000000, 0.000000] for an interval wholly below 0.
    assert rep.fmt_interval(-2.0, -4e-7, unit="s") == ("-2.0000000", "-0.0000004")
    assert rep.fmt_estimate(1.0004, 0.9996, 1.0012, unit="rho") == ("1.0004", "0.9996", "1.0012")
    assert rep.fmt_estimate(-20.04, -30.3, -9.87, unit="s") == ("-20.0", "-30.3", "-9.9")
    assert rep.fmt_estimate(0.75, 0.75, 0.75, unit="rho") == ("0.75", "0.75", "0.75")  # equal values print equal
    assert (rep.fmt_number(0.04, unit="n"), rep.fmt_number(13.0, unit="n"), rep.fmt_number(1.25, unit="n")) == (
        "0.04", "13.0", "1.2")
    with pytest.raises(ValueError, match="cannot be printed"):
        rep.fmt_interval(-1e-13, 1.0, unit="s")


@pytest.mark.parametrize("p, text", [
    (0.0123, "0.012"), (1.0, "1.000"), (0.346, "0.346"), (0.05, "0.050"), (0.01, "0.010"), (0.001, "0.001"),
    (0.0496, "0.0496"), (0.0504, "0.0504"), (0.0099, "0.0099"), (0.0012, "0.0012"),
    (0.0009996, "9.996 × 10⁻⁴"), (3.2e-05, "3.2 × 10⁻⁵"), (3.9e-18, "3.9 × 10⁻¹⁸"),
])
def test_a_p_value_prints_on_its_own_side_of_0_05_0_01_and_0_001(p: float, text: str) -> None:
    assert rep.fmt_p(p) == text


@pytest.mark.parametrize("p", [math.nan, -0.1, 0.0, 1.5])
def test_a_p_value_that_is_not_a_probability_of_the_test_is_refused(p: float) -> None:
    with pytest.raises(ValueError, match="not a p-value"):
        rep.fmt_p(p)


def test_s11_prints_the_never_entered_count_on_its_side_of_zero() -> None:
    text = rep.sentence_s11({"mean": 105.33, "ci95_low": 103.0, "ci95_high": 107.7}, never_entered=0.04)["text"]
    assert text.endswith("with 0.04 vehicles per episode that never entered.")


# ----------------------------------------------------------------------
# R5's readings (plan section 5): R1's and R2's CI sides, R3's two flags
# ----------------------------------------------------------------------


def _entry(fixed_time: tuple[float, float], max_pressure: tuple[float, float], rho: tuple[float, float]) -> dict[str, Any]:
    return {"paired_att": {"e_sumo": {"fixedtime": {"ci95_low": fixed_time[0], "ci95_high": fixed_time[1]},
                                      "maxpressure": {"ci95_low": max_pressure[0], "ci95_high": max_pressure[1]}}},
            "e_sumo": {"ci95_low": rho[0], "ci95_high": rho[1]}}


def test_rho_flags_are_r3s_readings() -> None:
    assert rep.rho_flags(0.98, 1.02) == {"rho > 0": True, "rho < 1": False}
    assert rep.rho_flags(-0.2, 0.9) == {"rho > 0": False, "rho < 1": True}
    assert rep.rho_flags(0.0, 1.0) == {"rho > 0": False, "rho < 1": False}  # ON 0 is not above 0; ON 1, not below it


def test_r5_reads_r1_to_r3_as_plan_section_5_defines_their_readings() -> None:
    aligned = _entry((-1.0, 2.0), (-1.0, 3.0), (0.2, 0.8))
    corrected = rep.r1_r3_readings(_entry((-9.0, -8.0), (1.0, 2.0), (1.686, 1.767)), aligned, "e_sumo")
    original = rep.r1_r3_readings(_entry((-9.0, -8.0), (1.0, 2.0), (0.9, 1.1)), aligned, "e_sumo")
    # S5's branch moves (beyond 1 -> contains 1 and lies above 0); R3's two readings do not.
    assert corrected == original == {
        "R1 fixed-time k = 4": "below", "R1 fixed-time tuned k": "contains",
        "R2 MaxPressure per-link": "above", "R2 MaxPressure lane-once": "contains",
        "R3 registered rho > 0": True, "R3 registered rho < 1": False,
        "R3 aligned rho > 0": True, "R3 aligned rho < 1": True,
    }
    assert rep.rho_reading(1.686, 1.767) != rep.rho_reading(0.9, 1.1)


def test_the_artifacts_r5_readings_and_s10_branches_come_from_their_own_functions() -> None:
    built_sources = sources(original_shift=-200.0)
    result = rep.analysis(built_sources)
    for label, arm in (("calibrated", "b_mean_k100"), ("source", "naive")):
        readings = result["r5"][label]["readings"]
        for key in ("e_sumo", "att_env"):
            corrected = rep.r1_r3_readings(rep.entry_of(result["registered"], p94.SUBJECT, arm),
                                           rep.entry_of(result["aligned"], p94.SUBJECT, arm), key)
            assert readings[key]["corrected"] == corrected
            assert readings[key]["all_agree"] is (readings[key]["corrected"] == readings[key]["original"])
        assert readings["all_agree"] is readings["e_sumo"]["all_agree"]
        assert set(result["r5"][label]["s10_branches"]["corrected"]) == {"S1", "S3", "S5", "S6"}
    # The original model 200 s faster: R2 per-link flips (it beat MaxPressure), so the readings do not all agree.
    assert result["r5"]["calibrated"]["readings"]["e_sumo"]["original"]["R2 MaxPressure per-link"] == "below"
    assert result["r5"]["calibrated"]["readings"]["all_agree"] is False


# ----------------------------------------------------------------------
# Wiring with intervals of non-zero width, arms that differ, and every seed visible
# ----------------------------------------------------------------------


def _noise(draw: int) -> float:
    return float((draw * 7) % 13) - 6.0


def _noisy(built: rep.Sources) -> rep.Sources:
    """The calibrated arm's rows moved by a per-draw term, so its paired intervals and its rho intervals have width."""
    rows = [{**row, "e_sumo": row["e_sumo"] + (_noise(int(row["draw_id"])) if row["arm"] == "b_mean_k100" else 0.0)}
            for row in built.new_rows]
    return rep.Sources(**{**built.__dict__, "new_rows": rows})


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    half = 1.96 * math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def test_r1s_interval_is_the_per_draw_mean_difference_with_its_own_width() -> None:
    built = _noisy(sources())
    result = rep.analysis(built)
    block = result["r1"]["calibrated"]["fixedtime_k4"]["e_sumo"]
    per_draw = []
    for draw in DRAWS:
        seeds = [r["e_sumo"] for r in built.new_rows if r["arm"] == "b_mean_k100" and r["draw_id"] == draw]
        per_draw.append(sum(seeds) / len(seeds) - built.by_draw[str(draw)]["fixedtime"]["e_sumo"])
    mean, low, high = _mean_ci(per_draw)
    assert high - low > 1.0
    assert (block["mean_difference"], block["ci95_low"], block["ci95_high"]) == pytest.approx((mean, low, high), rel=1e-12)


def test_a_rho_interval_that_contains_1_reads_not_below_1() -> None:
    built = sources()
    near_mp = [{**row, "e_sumo": 400.0 + _f(int(row["draw_id"])) + OFFSET[int(row["seed"])] + _noise(int(row["draw_id"]))}
               if row["arm"] == "b_mean_k100" else row for row in built.new_rows]
    result = rep.analysis(rep.Sources(**{**built.__dict__, "new_rows": near_mp}))
    registered = result["r3"]["calibrated"]["registered"]
    assert registered["e_sumo"]["ci95_low"] < 1.0 < registered["e_sumo"]["ci95_high"]
    assert registered["readings"]["e_sumo"] == {"rho > 0": True, "rho < 1": False}


def test_s10_reads_the_calibrated_arm_not_the_source_arm() -> None:
    built = sources()
    original = [{**row, "e_sumo": row["e_sumo"] - (100.0 if row["arm"] == "b_mean_k100" else 0.0)}
                for row in [{**r, "subject": "mix50"} for r in built.new_rows]]
    artifact = rep.build_zero_shot(rep.Sources(**{**built.__dict__, "original_rows": original}))
    s10 = next(s for s in artifact["sentences"] if s["template"] == "S10")
    assert s10["branch"] == "otherwise" and "the comparison with the registered MaxPressure" in s10["text"]
    source = artifact["r5"]["source"]["s10_branches"]
    assert source["corrected"] == source["original"]  # the source arm reads the same: S10 is not the source arm's


def test_per_seed_means_and_r7_for_the_source_arm_are_their_own_rows() -> None:
    built = sources()
    result = rep.analysis(built)
    mean_f = sum(_f(draw) for draw in DRAWS) / len(DRAWS)
    per_seed = result["also"]["calibrated"]["per_seed_means"]["e_sumo"]
    assert {seed: entry["mean"] for seed, entry in per_seed.items()} == pytest.approx(
        {str(seed): 450.0 + mean_f + OFFSET[seed] for seed in SEEDS}, rel=1e-12)
    assert {entry["n_draws"] for entry in per_seed.values()} == {100}
    # The anchor rows are the calibrated rows + 3 s: the source arm (470 + f) is 17 s slower than them.
    assert result["r7"]["source"]["e_sumo"]["mean_difference"] == pytest.approx(17.0, abs=1e-9)
    assert result["r7"]["calibrated"]["e_sumo"]["mean_difference"] == pytest.approx(-3.0, abs=1e-9)


# ----------------------------------------------------------------------
# load_sources: every refusal before any number
# ----------------------------------------------------------------------


def _cell_row(payload: dict[str, Any], anchors: Any) -> dict[str, Any]:
    return {"kind": "dt", "subject": payload["subject"], "arm": payload["arm"], "seed": payload["seed"],
            "draw_id": payload["draw_id"], "e_sumo": 450.0, "att_env": 320.0}


def _on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> p94.Roots:
    """A minimal campaign on disk: the 1,000 chunks, route (i)'s record, the guard and control records and every run
    record, with the registered checks that need simulators or a git history replaced at their seams."""
    roots = p94.Roots(repo_root=REPO_ROOT, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus",
                      draws_root=tmp_path / "draws")
    declaration = p94.load_declaration(roots)
    monkeypatch.setattr(p94, "assert_control_passed", lambda r: {"passed": True, "n_seeds_equal": 5})
    monkeypatch.setattr(ev, "assert_guards_authorise", lambda r, record: None)
    monkeypatch.setattr(ev, "chunk_problems", lambda *args, **kwargs: [])
    monkeypatch.setattr(rep, "_published", _cell_row)
    for path, payload in ((ev.guards_record_path(roots), {"passed": True}), (p94.control_record_path(roots), {})):
        p94.write_once(path, payload, roots)
    for cell in ev.new_cells():
        p94.write_once(ev.cell_path(roots, "new", cell), {key: cell[key] for key in ("subject", "arm", "seed", "draw_id")},
                       roots)
    episodes = [{"arm": ev.CITYFLOW_ARM, "seed": seed, "draw_id": draw, "att_horizon": 100.0 + draw - 1000,
                 "horizon_vehicle_count": 40.0, "episode_reward": -1.0} for seed in p94.SEEDS for draw in p94.HELD_OUT_DRAWS]
    p94.write_once(ev.route_i_output(roots), {"format_version": grid.GRID_FORMAT_VERSION, "arm": ev.CITYFLOW_ARM,
                                              "target_rtg": p94.SOURCE_TARGET, "episodes": episodes}, roots)
    monkeypatch.setattr(ar, "campaign_status", lambda work: {"complete": True, "refused": 0})
    monkeypatch.setattr(crep, "_rerolled", lambda work, scenario, arm, seed, draw, missing: {"seed": seed, "draw_id": draw})
    monkeypatch.setattr(crep, "_joined_row", lambda arm, tier, first, rerolled, **kwargs: {
        "arm": arm, "seed": first["seed"], "draw_id": first["draw_id"], "att_ours": first["att_horizon"],
        "att_engine": first["att_horizon"] - 2.0})
    facts = declaration["normalisation"]
    monkeypatch.setattr(ev, "new_subject_facts", lambda r, d: tcal.SubjectFacts(
        subject=p94.SUBJECT, best_source_return=-5959.0, rtg_scale=float(declaration["rtg_scale"]),
        support_range_over_the_split=tuple(facts["support_range_over_the_split"]), n_rows=int(facts["n_rows"]),
        training_set_return_min=-float(declaration["rtg_scale"]), checkpoints=(), state_dim=25, context_length=20))
    monkeypatch.setattr(crep, "driver_sessions", lambda r: [])
    monkeypatch.setattr(crep, "stage_code_commit", lambda sessions, stage: {"commit": "c" * 40})
    monkeypatch.setattr(p94, "run_status", lambda r, run, **kwargs: "done")
    for run in p94.training_runs():
        p94.write_once(p94.run_record_path(roots, run), {"run": run.name}, roots)
    return roots


def test_load_sources_reads_a_complete_campaign_with_every_design_complete(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = rep.load_sources(_on_disk(tmp_path, monkeypatch))
    assert (len(loaded.new_rows), len(loaded.original_rows), len(loaded.anchor_k200_rows), len(loaded.cityflow_rows),
            len(loaded.cityflow_original), len(loaded.training_runs)) == (1000, 1000, 500, 500, 500, 10)
    assert {(row["kind"], row["arm"]) for row in loaded.anchor_k200_rows} == {("dt", "naive_in_domain")}


def test_load_sources_refuses_a_chunk_that_is_not_a_declared_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _on_disk(tmp_path, monkeypatch)
    p94.write_once(ev.cells_dir(roots) / "cell_extra.json", {}, roots)
    with pytest.raises(ValueError, match="are not declared cells"):
        rep.load_sources(roots)


def test_load_sources_refuses_a_chunk_that_may_not_be_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _on_disk(tmp_path, monkeypatch)
    monkeypatch.setattr(ev, "chunk_problems", lambda r, kind, cell, payload, d, **kw: (
        ["rolled by other code"] if (cell["arm"], cell["seed"], cell["draw_id"]) == ("naive", 303, 1050) else []))
    with pytest.raises(ValueError, match="rolled by other code"):
        rep.load_sources(roots)


def test_load_sources_refuses_a_missing_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _on_disk(tmp_path, monkeypatch)
    ev.cell_path(roots, "new", ev.new_cells()[17]).unlink()
    with pytest.raises(ValueError, match="is absent: the campaign is incomplete"):
        rep.load_sources(roots)


def test_load_sources_refuses_route_ii_with_a_refusal_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _on_disk(tmp_path, monkeypatch)
    monkeypatch.setattr(ar, "campaign_status", lambda work: {"complete": True, "refused": 1})
    with pytest.raises(ValueError, match="not complete without refusals"):
        rep.load_sources(roots)


def test_load_sources_refuses_before_every_training_run_is_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _on_disk(tmp_path, monkeypatch)
    monkeypatch.setattr(p94, "run_status", lambda r, run, **kw: "missing" if run.seed == 404 and run.group == "hz1x1"
                        else "done")
    with pytest.raises(ValueError, match="every training run is done"):
        rep.load_sources(roots)


def test_load_sources_refuses_a_non_finite_cell_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _on_disk(tmp_path, monkeypatch)
    monkeypatch.setattr(rep, "_published", lambda payload, anchors: {
        **_cell_row(payload, anchors), "att_env": math.nan if payload["draw_id"] == 1099 else 320.0})
    with pytest.raises(ValueError, match="not finite"):
        rep.load_sources(roots)


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows[1:], "not the 1000 cells"),
    (lambda rows: [*rows[:-1], rows[0]], "appear more than once"),
    (lambda rows: [{**rows[0], "e_sumo": math.inf}, *rows[1:]], "not finite"),
])
def test_a_source_design_must_be_complete_once_each_and_finite(change: Any, message: str) -> None:
    rows = [{"arm": arm, "seed": seed, "draw_id": draw, "e_sumo": 1.0} for arm in p94.SUMO_ARMS for seed in p94.SEEDS
            for draw in p94.HELD_OUT_DRAWS]
    rep.assert_design(rows, "new", arms=p94.SUMO_ARMS, keys=("e_sumo",))
    with pytest.raises(ValueError, match=message):
        rep.assert_design(change(rows), "new", arms=p94.SUMO_ARMS, keys=("e_sumo",))


# ----------------------------------------------------------------------
# The artifacts are written atomically
# ----------------------------------------------------------------------


def test_a_failed_artifact_write_leaves_no_file_and_no_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rep, "load_sources", lambda roots: sources())

    def refuse(source: Any, destination: Any) -> None:
        raise OSError("simulated failure at the link")

    monkeypatch.setattr(os, "link", refuse)
    with pytest.raises(OSError, match="simulated failure"):
        rep.write_artifacts(None, out_dir=tmp_path)  # type: ignore[arg-type]
    assert sorted(path.name for path in tmp_path.iterdir()) == []
