"""Tests for ``BRIEF_48`` Amendment A, A.7: the D5 definition check on hz1x1 and grid4x4 (EXPLORATION).

Written before the implementation.  The check re-runs D5's fixed-time protocol (``docs/data/fixed_time_sweep/README.md``)
under BOTH ATT definitions through the hz4x4 sweep's own functions, given a scenario parameter.  Two things are
load-bearing and tested here:

* the T-REGRESS -- every per-k mean ``att_ours`` must print, to two decimals, as D5's recorded ``att_horizon``; one cell
  that does not reproduce stops the check (A.7: "the protocol is not D5's");
* the REGRET ARITHMETIC -- each definition's argmin, the AGREE / DEFINITION-DEPENDENT status, and each definition's
  shortfall at the other's argmin, recomputed here by exact rational arithmetic.

A third test pins the refactor: the hz4x4 defaults still produce a committed hz4x4 sweep record exactly.  Tests that
need the main tree's draws and corpus or the native engine skip with the reason when those are absent.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import offline.hz4x4_ladder as hz

REPO = Path(__file__).resolve().parents[1]
README = REPO / "docs" / "data" / "fixed_time_sweep" / "README.md"
HZ4X4_SWEEP = REPO / "docs" / "data" / "p9_3_fixed_time_sweep.json"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except ImportError:
        return False
    return True


needs_cityflow = pytest.mark.skipif(
    not _cityflow_available(), reason="the native cityflow engine is not importable here"
)
needs_main_tree = pytest.mark.skipif(
    not (hz.MAIN_CORPUS_ROOT / "cf_hz1x1__fixedtime" / "manifest.json").is_file()
    or not (hz.MAIN_DRAWS_ROOT / "cityflow1x1" / "draw_0005" / "cityflow.json").is_file()
    or not (hz.MAIN_DRAWS_ROOT / "cityflow_grid4x4" / "draw_0005" / "cityflow.json").is_file(),
    reason="the main tree's v1.1 corpus manifests or its materialised draws 1-5 are not on this machine",
)

#: A.7's own transcription of D5's table (BRIEF_48 Amendment A, 2026-10-16): the second source the README parse is held to.
AMENDMENT_TRANSCRIPTION = {
    "cf_hz1x1": {2: "318.81", 3: "286.65", 4: "270.53", 6: "261.86", 8: "265.48", 10: "274.69", 14: "291.47"},
    "cf_grid4x4": {1: "206.09", 2: "261.65", 3: "297.54", 4: "290.57", 6: "341.05"},
}


# ======================================================================
# D5's recorded table, read from the README rather than retyped
# ======================================================================


def test_d5_recorded_table_is_parsed_from_the_readme_and_matches_the_amendments_transcription() -> None:
    table = hz.d5_recorded_table(README)
    for scenario, cells in AMENDMENT_TRANSCRIPTION.items():
        assert table[scenario]["att_horizon"] == cells, scenario
    assert table["cf_hz1x1"]["chosen_k"] == 6
    assert table["cf_grid4x4"]["chosen_k"] == 1
    from offline.recollect_v11 import TUNED_FIXED_TIME_K

    assert {s: table[s]["chosen_k"] for s in AMENDMENT_TRANSCRIPTION} == {
        s: TUNED_FIXED_TIME_K[s] for s in AMENDMENT_TRANSCRIPTION
    }
    assert [spec.name for spec in hz.D5_SCENARIOS] == ["cf_hz1x1", "cf_grid4x4"]
    for spec in hz.D5_SCENARIOS:
        assert spec.k_grid == tuple(sorted(AMENDMENT_TRANSCRIPTION[spec.name])), spec.name


def test_d5_recorded_table_refuses_a_readme_without_the_results_table(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# no table here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="results table"):
        hz.d5_recorded_table(path)


# ======================================================================
# The T-regress: every printed cell, to two decimals
# ======================================================================


def _summary(means_ours: dict[int, float], means_engine: dict[int, float] | None = None) -> dict[int, dict[str, Any]]:
    engine = means_ours if means_engine is None else means_engine
    return {
        k: {"n": 5, "draw_ids": [1, 2, 3, 4, 5], "mean_att_ours": means_ours[k], "mean_att_engine": engine[k]}
        for k in means_ours
    }


def test_t_regress_reproduces_only_when_every_cell_prints_as_recorded() -> None:
    recorded = {2: "318.81", 3: "286.65", 6: "261.86"}
    exact = hz.d5_t_regress(_summary({2: 318.8125, 3: 286.6484375, 6: 261.859375}), recorded)
    assert exact["all_reproduce"] is True
    assert {k: cell["computed_2dp"] for k, cell in exact["cells"].items()} == recorded
    off = hz.d5_t_regress(_summary({2: 318.8125, 3: 286.65625, 6: 261.859375}), recorded)
    assert off["all_reproduce"] is False
    assert [k for k, cell in off["cells"].items() if not cell["reproduces"]] == [3]
    assert off["cells"][3]["computed_2dp"] == "286.66"
    with pytest.raises(ValueError, match="k grid"):
        hz.d5_t_regress(_summary({2: 318.8125, 3: 286.6484375}), recorded)


# ======================================================================
# The regret arithmetic
# ======================================================================


def test_definition_check_agrees_with_zero_shortfall() -> None:
    summary = _summary({1: 230.0, 2: 220.0, 3: 225.0}, {1: 250.0, 2: 240.0, 3: 244.0})
    check = hz.definition_check(summary, k_grid=(1, 2, 3))
    assert check["status"] == "AGREE"
    assert (check["k_att_engine"], check["k_att_ours"]) == (2, 2)
    assert check["att_engine_shortfall_at_att_ours_argmin"] == 0.0
    assert check["att_ours_shortfall_at_att_engine_argmin"] == 0.0


def test_definition_check_measures_each_definitions_shortfall_at_the_others_argmin() -> None:
    ours = {1: 589.5, 2: 533.625, 3: 536.0, 4: 576.25}
    engine = {1: 652.25, 2: 553.125, 3: 544.0625, 4: 580.0}
    check = hz.definition_check(_summary(ours, engine), k_grid=(1, 2, 3, 4))
    assert check["status"] == "DEFINITION-DEPENDENT"
    assert (check["k_att_engine"], check["k_att_ours"]) == (3, 2)
    engine_short = Fraction(553.125) - Fraction(544.0625)
    ours_short = Fraction(536.0) - Fraction(533.625)
    assert check["att_engine_shortfall_at_att_ours_argmin"] == float(engine_short) == 9.0625
    assert check["att_ours_shortfall_at_att_engine_argmin"] == float(ours_short) == 2.375
    assert check["att_engine_relative_shortfall"] == float(engine_short / Fraction(544.0625))
    assert check["att_ours_relative_shortfall"] == float(ours_short / Fraction(533.625))


def test_definition_check_refuses_to_name_an_argmin_on_an_exact_tie() -> None:
    summary = _summary({1: 230.0, 2: 220.0, 3: 220.0}, {1: 250.0, 2: 240.0, 3: 244.0})
    check = hz.definition_check(summary, k_grid=(1, 2, 3))
    assert check["status"] == "TIE"
    assert check["k_att_ours"] is None
    assert check["att_engine_shortfall_at_att_ours_argmin"] is None
    with pytest.raises(ValueError, match="k = 4"):
        hz.definition_check(summary, k_grid=(1, 2, 3, 4))


def test_summarise_admission_means_never_entered_and_the_entered_fraction() -> None:
    records = [
        {"k": k, "draw_id": d, "never_entered": n, "entered_fraction": f}
        for k, rows in {1: [(1, 4, 0.5), (2, 0, 1.0)], 2: [(1, 2, 0.75), (2, 2, 0.75)]}.items()
        for d, n, f in rows
    ]
    summary = hz.summarise_admission(records, k_grid=(1, 2), draws=(1, 2))
    assert summary == {
        1: {"mean_never_entered": 2.0, "mean_entered_fraction": 0.75},
        2: {"mean_never_entered": 2.0, "mean_entered_fraction": 0.75},
    }
    with pytest.raises(ValueError, match="draws"):
        hz.summarise_admission(records[:-1], k_grid=(1, 2), draws=(1, 2))


# ======================================================================
# The schedule is D5's, and the protocol reproduces D5's registered cells live
# ======================================================================


@needs_main_tree
def test_fixedtime_factory_for_k_holds_the_declared_schedule_and_its_digest() -> None:
    from offline.materialise_draws import draw_config_path

    hz1x1 = next(spec for spec in hz.D5_SCENARIOS if spec.name == "cf_hz1x1")
    config = draw_config_path(hz1x1.scenario_key, 1, out_root=hz.MAIN_DRAWS_ROOT)
    manifest = json.loads((hz.MAIN_CORPUS_ROOT / hz1x1.protocol_dir / "manifest.json").read_bytes())["run_metadata"]
    _factory, provenance = hz.fixedtime_factory_for_k(
        config, 6, k_grid=hz1x1.k_grid, schedule_source="shipped_plan", plan_sha256=manifest["fixed_time_plan_sha256"]
    )
    assert provenance["schedule_source"] == "shipped_plan"
    assert provenance["plan_sha256"] == manifest["fixed_time_plan_sha256"]
    with pytest.raises(ValueError, match="equal_split"):
        hz.fixedtime_factory_for_k(config, 6, k_grid=hz1x1.k_grid)
    with pytest.raises(ValueError, match="sha256"):
        hz.fixedtime_factory_for_k(config, 6, k_grid=hz1x1.k_grid, schedule_source="shipped_plan", plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="k = 5"):
        hz.fixedtime_factory_for_k(config, 5, k_grid=hz1x1.k_grid, schedule_source="shipped_plan",
                                   plan_sha256=manifest["fixed_time_plan_sha256"])


@needs_main_tree
@needs_cityflow
def test_the_registered_cells_of_d5_reproduce_live_through_the_sweeps_own_functions() -> None:
    from offline.materialise_draws import draw_config_path

    recorded = hz.d5_recorded_table(README)
    for spec, k in ((next(s for s in hz.D5_SCENARIOS if s.name == "cf_hz1x1"), 6),
                    (next(s for s in hz.D5_SCENARIOS if s.name == "cf_grid4x4"), 1)):
        settings, schedule_source, plan_sha256 = hz.d5_protocol(spec, corpus_root=hz.MAIN_CORPUS_ROOT)
        assert schedule_source == spec.schedule_source
        records = [
            hz.sweep_episode(k, d, config_path=draw_config_path(spec.scenario_key, d, out_root=hz.MAIN_DRAWS_ROOT),
                             env_settings=settings, scenario=spec.name, scenario_id=spec.scenario_id,
                             k_grid=spec.k_grid, schedule_source=schedule_source, plan_sha256=plan_sha256)
            for d in hz.K_SWEEP_DRAWS
        ]
        mean = sum(r["att_ours"] for r in records) / len(records)
        assert f"{mean:.2f}" == recorded[spec.name]["att_horizon"][k], spec.name
        assert all(r["created"] == r["entered"] + r["never_entered"] for r in records)


@needs_main_tree
@needs_cityflow
def test_the_hz4x4_defaults_still_produce_a_committed_hz4x4_sweep_record() -> None:
    from offline.materialise_draws import draw_config_path

    artifact = json.loads(HZ4X4_SWEEP.read_bytes())
    (committed,) = [r for r in artifact["records"] if (r["k"], r["draw_id"]) == (3, 1)]
    record = hz.sweep_episode(3, 1, config_path=draw_config_path(hz.SCENARIO_KEY, 1, out_root=hz.MAIN_DRAWS_ROOT),
                              env_settings=artifact["protocol"]["env_settings"])
    for key in ("att_ours", "att_engine", "created", "entered", "never_entered", "entered_fraction",
                "schedule_source", "plan_sha256"):
        assert record[key] == committed[key], key


def test_d5_protocol_refuses_a_manifest_with_another_schedule_source(tmp_path: Path) -> None:
    hz1x1 = next(spec for spec in hz.D5_SCENARIOS if spec.name == "cf_hz1x1")
    directory = tmp_path / hz1x1.protocol_dir
    directory.mkdir()
    meta = {"behavior_policy": "fixedtime", "fixed_time_schedule_source": "equal_split", "fixed_time_plan_sha256": None}
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": meta}), encoding="utf-8")
    with pytest.raises(ValueError, match="shipped_plan"):
        hz.d5_protocol(hz1x1, corpus_root=tmp_path)
    meta["behavior_policy"] = "maxpressure"
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": meta}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a fixed-time tier"):
        hz.d5_protocol(hz1x1, corpus_root=tmp_path)
