"""P9.4's report assembled end to end on a synthetic ``Sources`` over the full 100 held-out draws: the P7.3a blocks, R1-R7
wired to the right arms and anchors with hand-known values, the sentences following the numbers, and the writes refusing
other content.  ``load_sources``' checks are exercised by the evaluation tests and by the run itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.p9_4_report as rep

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAWS = tuple(range(1000, 1100))
SEEDS = (101, 202, 303, 404, 505)
OFFSET = {101: -2.0, 202: -1.0, 303: 0.0, 404: 1.0, 505: 2.0}  # sums to 0: each draw's seed mean is the base value


def _f(draw: int) -> float:
    return float((draw * 37) % 11)


def _row(subject: str, arm: str, seed: int, draw: int, e_sumo: float, att_env: float, **extra: Any) -> dict[str, Any]:
    return {"kind": "dt", "subject": subject, "arm": arm, "seed": seed, "draw_id": draw, "e_sumo": e_sumo,
            "att_env": att_env, "horizon_vehicle_count": 50.0, "episode_reward": -7000.0, "n_decisions_in_support": 300,
            "n_never_entered": 12, "n_teleports": 0, "n_vanished_without_arrival": 0, **extra}


def new_rows(shift: float = 0.0) -> list[dict[str, Any]]:
    """Calibrated: 450 + f(draw) on e_sumo; source: 470 + f(draw); every seed offset cancels in the draw's mean."""
    out = []
    for draw in DRAWS:
        for seed in SEEDS:
            out.append(_row("mix50_fresh", "b_mean_k100", seed, draw, 450.0 + _f(draw) + OFFSET[seed] + shift,
                            320.0 + _f(draw) + OFFSET[seed]))
            out.append(_row("mix50_fresh", "naive", seed, draw, 470.0 + _f(draw) + OFFSET[seed] + shift,
                            325.0 + _f(draw) + OFFSET[seed]))
    return out


def by_draw() -> dict[str, Any]:
    out = {}
    for draw in DRAWS:
        base = {"horizon_vehicle_count": 1.0, "episode_reward": -1.0}
        out[str(draw)] = {
            "fixedtime": {**base, "e_sumo": 600.0 + _f(draw), "att_env": 350.0 + _f(draw)},
            "maxpressure": {**base, "e_sumo": 400.0 + _f(draw), "att_env": 330.0 + _f(draw)},
            "fixedtime_kcf": {**base, "e_sumo": 560.0 + _f(draw), "att_env": 348.0 + _f(draw)},
            "maxpressure_1to1": {**base, "e_sumo": 380.0 + _f(draw), "att_env": 310.0 + _f(draw)},
        }
    return out


def cityflow(base: float, ours_gap: float = 4.0) -> list[dict[str, Any]]:
    return [{"arm": "dt@x", "tier": "x", "seed": seed, "draw_id": draw, "att_ours": base + ours_gap + _f(draw) + OFFSET[seed],
             "att_engine": base + _f(draw) + OFFSET[seed], "entered": 1800, "created": 1813, "never_entered": 13,
             "entered_fraction": 1800 / 1813, "episode_reward": -7000.0, "horizon_vehicle_count": 45.0, "source": "s"}
            for seed in SEEDS for draw in DRAWS]


def targets_table() -> list[dict[str, Any]]:
    def row(rule: str, statistic: str, k: int | None, role: str, target: float) -> dict[str, Any]:
        return {"rule": rule, "statistic": statistic, "k": k, "role": role, "target_rtg": target,
                "in_support": {"position": "inside", "margin": 0.0, "range": [-41000.0, -6.0], "target_rtg": target}}

    return [row("rule_b", "mean", 100, "registered_prompt", -7431.0185327454665),
            row("rule_b", "max", 100, "ablation", -6000.0), row("rule_a", "q1.0", 100, "ablation", -20809.0),
            row("naive", "none", None, "ablation", -5959.0)]


def sources(*, original_shift: float = 5.0, anchor_shift: float = 3.0, cityflow_original_base: float = 108.0,
            original_ours_gap: float = 6.0) -> Any:
    p7_3a = json.loads((REPO_ROOT / "docs" / "data" / "p7_3a_zero_shot.json").read_bytes())
    rows = new_rows()
    original = [{**r, "subject": "mix50", "e_sumo": r["e_sumo"] + original_shift + (3.0 if r["arm"] == "naive" else 0.0)}
                for r in rows]
    anchor = [_row("anchor_k200", "naive_in_domain", r["seed"], r["draw_id"], r["e_sumo"] + anchor_shift, r["att_env"])
              for r in rows if r["arm"] == "b_mean_k100"]
    return rep.Sources(new_rows=rows, by_draw=by_draw(), original_rows=original, anchor_k200_rows=anchor,
                       cityflow_rows=cityflow(105.0), cityflow_original=cityflow(cityflow_original_base, ours_gap=original_ours_gap),
                       targets_table=targets_table(), p7_3a_rho=p7_3a["rho"],
                       guards={"g1": {"passed": True}, "g2": {"passed": True}, "passed": True},
                       control={"n_seeds_equal": 5, "passed": True})


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return rep.build_zero_shot(sources())


def test_the_artifact_carries_p7_3a_blocks_for_mix50_fresh(artifact: dict[str, Any]) -> None:
    assert artifact["format_version"] == "p9.4-zero-shot/1.0" and len(artifact["cells"]) == 1000
    for key in ("cells", "rho", "h3", "contrast", "in_support"):
        assert key in artifact
    entries = {(e["subject"], e["arm"]) for e in artifact["rho"]["by_subject_arm"]}
    assert entries == {("mix50_fresh", "b_mean_k100"), ("mix50_fresh", "naive")}
    assert len(artifact["rho"]["by_subject_arm_seed"]) == 10
    assert sorted(artifact["contrast"]["by_subject"]) == ["mix50_fresh"]
    assert sorted(artifact["in_support"]["declared_by_the_calibration_artifact"]) == ["mix50_fresh"]
    assert artifact["in_support"]["observed_decisions"]["mix50_fresh/naive"] == {"n_cells": 500,
                                                                                 "total_decisions_in_support": 150000}
    assert {c["definition"] for c in artifact["h3"]["clauses"]} == {"e_sumo", "att_env"}


def test_the_rho_layout_is_p7_3as_texts_with_the_diagnostics_recomputed(artifact: dict[str, Any]) -> None:
    p7_3a = json.loads((REPO_ROOT / "docs" / "data" / "p7_3a_zero_shot.json").read_bytes())["rho"]
    for key in ("what", "role", "zero_denominator_rule"):
        assert artifact["rho"]["definitions"]["e_sumo"][key] == p7_3a["definitions"]["e_sumo"][key]
    assert artifact["rho"]["definitions"]["att_env"]["caveat"] == p7_3a["definitions"]["att_env"]["caveat"]
    assert (artifact["rho"]["estimator"], artifact["rho"]["formula"]) == (p7_3a["estimator"], p7_3a["formula"])
    diagnostic = artifact["rho"]["definitions"]["e_sumo"]["denominator_diagnostic"]
    assert (diagnostic["n_draws"], diagnostic["min"], diagnostic["max"]) == (100, 200.0, 200.0)
    aligned = artifact["aligned"]["rho"]["definitions"]["att_env"]["denominator_diagnostic"]
    assert (aligned["min"], aligned["max"]) == (38.0, 38.0)


def test_r1_to_r3_use_the_right_anchor_of_the_right_pair(artifact: dict[str, Any]) -> None:
    r1, r2, r3 = artifact["r1"]["calibrated"], artifact["r2"]["calibrated"], artifact["r3"]["calibrated"]
    assert r1["fixedtime_k4"]["e_sumo"]["mean_difference"] == pytest.approx(-150.0, abs=1e-9)
    assert r1["fixedtime_tuned_k"]["e_sumo"]["mean_difference"] == pytest.approx(-110.0, abs=1e-9)
    assert r2["maxpressure_per_link"]["e_sumo"]["mean_difference"] == pytest.approx(50.0, abs=1e-9)
    assert r2["maxpressure_lane_once"]["e_sumo"]["mean_difference"] == pytest.approx(70.0, abs=1e-9)
    assert r2["maxpressure_lane_once"]["e_sumo"]["right_arm"] == "maxpressure_1to1"
    assert (r1["fixedtime_k4"]["e_sumo"]["wins"], r2["maxpressure_per_link"]["e_sumo"]["losses"]) == (100, 100)
    # rho per row: (FT - DT) / (FT - MP); the seed offsets are linear, so each draw's mean is 150 / 200 = 0.75.
    assert r3["registered"]["e_sumo"]["mean"] == pytest.approx(0.75, abs=1e-12)
    assert r3["aligned"]["e_sumo"]["mean"] == pytest.approx(110.0 / 180.0, abs=1e-12)
    assert r3["registered"]["readings"]["e_sumo"] == {"rho > 0": True, "rho < 1": True}
    assert artifact["r1"]["source"]["fixedtime_k4"]["e_sumo"]["mean_difference"] == pytest.approx(-130.0, abs=1e-9)


def test_r4_r5_r6_and_r7_pair_the_right_rows(artifact: dict[str, Any]) -> None:
    assert artifact["r4"]["paired"]["e_sumo"]["mean_difference"] == pytest.approx(-20.0, abs=1e-9)
    assert artifact["r4"]["paired"]["e_sumo"]["right_arm"] == "naive"
    assert artifact["r4"]["contrast"]["registered"]["by_subject"]["mix50_fresh"]["e_sumo"]["difference"] == pytest.approx(
        0.75 - 130.0 / 200.0, abs=1e-12)
    assert artifact["r5"]["calibrated"]["e_sumo"]["mean_difference"] == pytest.approx(-5.0, abs=1e-9)
    assert artifact["r5"]["calibrated"]["e_sumo"]["wins"] == 100
    assert artifact["r5"]["calibrated"]["att_env"]["mean_difference"] == pytest.approx(0.0, abs=1e-12)
    assert artifact["r7"]["calibrated"]["e_sumo"]["mean_difference"] == pytest.approx(-3.0, abs=1e-9)
    assert artifact["r7"]["calibrated"]["e_sumo"]["right_arm"] == "sumo_trained_anchor_k200"
    r6 = artifact["r6"]
    mean_engine = sum(105.0 + _f(d) for d in DRAWS) / 100
    assert r6["att_engine"]["mean"] == pytest.approx(mean_engine, abs=1e-9)
    assert r6["att_ours"]["mean"] == pytest.approx(mean_engine + 4.0, abs=1e-9)
    assert r6["corrected_minus_original"]["att_engine"]["mean_difference"] == pytest.approx(-3.0, abs=1e-9)
    assert r6["corrected_minus_original"]["att_ours"]["mean_difference"] == pytest.approx(-5.0, abs=1e-9)
    per_draw = [105.0 + _f(d) for d in DRAWS]
    sd = (sum((v - mean_engine) ** 2 for v in per_draw) / 99) ** 0.5
    assert r6["att_engine"]["n_draws"] == 100
    assert r6["att_engine"]["ci95_high"] - r6["att_engine"]["ci95_low"] == pytest.approx(2 * 1.96 * sd / 10.0, rel=1e-9)
    assert r6["companions"]["never_entered"] == 13.0
    assert set(r6["att_engine"]["per_seed_means"]) == {str(s) for s in SEEDS}


def test_every_template_appears_once_in_order_and_follows_the_numbers(artifact: dict[str, Any]) -> None:
    templates = [s["template"] for s in artifact["sentences"]]
    assert templates == ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S7b", "S8", "S9", "S10", "S11", "S12", "S13"]
    branches = {s["template"]: s["branch"] for s in artifact["sentences"]}
    assert (branches["S1"], branches["S3"], branches["S5"], branches["S8"], branches["S9"], branches["S13"]) == (
        "below", "above", "inside_0_1", "below", "below", "below")
    # The original model is 5 s slower everywhere: every interval moves, and no reading changes side.
    assert branches["S10"] == "all the same"
    assert all(not s["text"].endswith("..") for s in artifact["sentences"])
    texts = {s["template"]: s["text"] for s in artifact["sentences"]}
    assert "DT \u2212 fixed-time -150.0 s [-150.0, -150.0]" in texts["S1"]
    assert "DT \u2212 fixed-time -110.0 s [-110.0, -110.0]" in texts["S2"]
    assert "corrected \u2212 original -3.0 s [-3.0, -3.0]" in texts["S12"]
    assert "\u03c1 = 0.75 [0.75, 0.75]" in texts["S5"]
    assert "\u03c1 = 0.61 [0.61, 0.61] under the aligned anchors" in texts["S6"]
    assert "corrected \u2212 original -5.0 s [-5.0, -5.0]" in texts["S9"]
    assert "DT \u2212 MaxPressure 50.0 s [50.0, 50.0], on 100 of 100 draws" in texts["S3"]
    assert "DT \u2212 MaxPressure 70.0 s [70.0, 70.0], on 100 of 100 draws" in texts["S4"]
    assert "calibrated \u2212 source -20.0 s [-20.0, -20.0]" in texts["S8"]
    assert "transferred \u2212 SUMO-trained -3.0 s [-3.0, -3.0], on 100 of 100 draws" in texts["S13"]
    mean_engine = sum(105.0 + _f(d) for d in DRAWS) / 100
    assert f"average travel time of {mean_engine:.1f} s" in texts["S11"] and "with 13.0 vehicles" in texts["S11"]
    assert artifact["r5"]["source"]["e_sumo"]["mean_difference"] == pytest.approx(-8.0, abs=1e-9)


def test_s7_and_s7b_read_their_own_pair() -> None:
    """Lane-once MaxPressure made slower than the DT: rho beyond 1 under the aligned pair only."""
    built_sources = sources()
    slow = {draw: {**slot, "maxpressure_1to1": {**slot["maxpressure_1to1"], "e_sumo": 500.0 + _f(int(draw))}}
            for draw, slot in built_sources.by_draw.items()}
    built = rep.build_zero_shot(rep.Sources(**{**built_sources.__dict__, "by_draw": slow}))
    texts = {s["template"]: s["text"] for s in built["sentences"]}
    assert texts["S7"].endswith("is better than fixed-time and is worse than MaxPressure under the registered anchors.")
    assert texts["S7b"].endswith("is better than fixed-time and is not worse than MaxPressure under the aligned anchors.")


def test_s10_names_every_reading_the_correction_changes() -> None:
    """An original model 200 s faster: S3 flips (it beat MaxPressure), and rho moves beyond 1 under both pairs."""
    built = rep.build_zero_shot(sources(original_shift=-200.0))
    s10 = next(s for s in built["sentences"] if s["template"] == "S10")
    assert s10["branch"] == "otherwise"
    assert s10["text"] == (
        "The correction changes the comparison with the registered MaxPressure, \u03c1 under the registered anchors "
        "and \u03c1 under the aligned anchors: faster than the registered MaxPressure becomes slower than the "
        "registered MaxPressure; beyond MaxPressure's 1 becomes between fixed-time (0) and MaxPressure (1); beyond "
        "MaxPressure's 1 becomes between fixed-time (0) and MaxPressure (1).")
    assert built["r5"]["calibrated"]["readings"]["all_agree"] is False


def test_identical_models_read_the_same_and_s12b_appears_only_when_att_ours_differs() -> None:
    same = rep.build_zero_shot(sources(original_shift=0.0, cityflow_original_base=105.0, original_ours_gap=4.0))
    s10 = next(s for s in same["sentences"] if s["template"] == "S10")
    assert s10["branch"] == "all the same"
    assert "S12b" not in {s["template"] for s in same["sentences"]}
    assert next(s for s in same["sentences"] if s["template"] == "S12")["branch"] == "contains"
    differing = rep.build_zero_shot(sources(original_shift=0.0, cityflow_original_base=105.0))
    s12b = next(s for s in differing["sentences"] if s["template"] == "S12b")
    assert s12b["branch"] == "below" and "-2.0 s [-2.0, -2.0]" in s12b["text"]


def test_a_campaign_missing_the_same_draw_everywhere_is_refused_at_the_draw_count() -> None:
    full = sources()
    keep = lambda rows: [r for r in rows if r["draw_id"] != 1099]  # noqa: E731 - one filter, used five times
    short = rep.Sources(**{**full.__dict__, "new_rows": keep(full.new_rows), "original_rows": keep(full.original_rows),
                           "anchor_k200_rows": keep(full.anchor_k200_rows), "cityflow_rows": keep(full.cityflow_rows),
                           "cityflow_original": keep(full.cityflow_original),
                           "by_draw": {k: v for k, v in full.by_draw.items() if k != "1099"}})
    with pytest.raises(ValueError, match="not the 100 held-out draws"):
        rep.build_zero_shot(short)


def test_a_comparison_over_fewer_than_the_100_draws_is_refused() -> None:
    short = sources()
    trimmed = [r for r in short.cityflow_original if r["draw_id"] != 1099]
    with pytest.raises(ValueError, match="different draw sets"):
        rep.build_zero_shot(rep.Sources(**{**short.__dict__, "cityflow_original": trimmed}))


def test_the_writes_refuse_other_content_and_repeat_only_the_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rep, "load_sources", lambda roots: sources())
    first = rep.write_artifacts(None, out_dir=tmp_path)  # type: ignore[arg-type]
    assert sorted(first) == ["p9_4_cityflow.json", "p9_4_training.json", "p9_4_zero_shot.json"]
    assert rep.write_artifacts(None, out_dir=tmp_path) == first  # type: ignore[arg-type]
    (tmp_path / "p9_4_training.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not rewritten"):
        rep.write_artifacts(None, out_dir=tmp_path)  # type: ignore[arg-type]
    assert (tmp_path / "p9_4_training.json").read_text(encoding="utf-8") == "{}"
