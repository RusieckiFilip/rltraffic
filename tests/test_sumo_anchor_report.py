"""Tests for ``offline/sumo_anchor_rescore.py``, part 3 -- the artifact ``docs/data/p9_2_sumo_anchors.json`` (``BRIEF_47`` §3.3).

* **A synthetic end-to-end report**: a work directory of 600 stand-in cells whose new anchors carry the REGISTERED anchors'
  values under the new arm names, so every pair must reproduce the registered rho -- built by the real report over the real
  committed artifacts, with the two provenance substitutions every regression in this repository uses (a chunk's commit
  is not a real commit here) and nothing else.
* **Refusals before any write**: a missing cell, an undeclared file, an invalid chunk, a missing or failed replica gate, the
  load-bearing CityFlow hz1x1 identity broken.
* **The label** is A27(a)'s item 5, verbatim from ``PREREGISTRATION.md``.
* **T-regress**: the committed artifact regenerates byte-identically from the run's work directory (gated: until the
  campaign has run neither exists, and the test says so).
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import pytest

import offline.sumo_anchor_rescore as sar
from tests.test_sumo_anchor_rescore import _gate_record, _sumo_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
COMMIT = "b" * 40
#: BRIEF_47 Amendment A.2.1, verbatim: hz1x1's clause 3 when clause 2 HOLDS under a pair.
PREMISE_HOLDS = "PREMISE HOLDS UNDER THIS PAIR — NOT SCOREABLE: no few-shot curve was registered on hz1x1"

_SUMO_COPIED = (
    "e_sumo", "att_env", "att_horizon", "att_running_mean", "p_sumo", "w_sumo", "mean_depart_delay", "episode_reward",
    "horizon_vehicle_count", "n_created", "n_entered", "n_never_entered", "n_pending_at_horizon",
    "n_vanished_without_arrival", "n_arrived_never_observed_at_a_boundary", "max_abs_depart_clock_deviation",
    "engine_seed_drawn",
)


@functools.lru_cache(maxsize=None)
def _registered_sumo(scenario: str) -> dict[int, dict[str, dict[str, Any]]]:
    """Parsed once per scenario: the committed artifact is up to 65 MB and 400 stand-in cells read it."""
    artifact = json.loads((DATA / sar.REGISTERED_ANCHOR_ARTIFACT[scenario]).read_bytes())
    return sar.registered_anchors(artifact)


@functools.lru_cache(maxsize=None)
def _escalated(arm: str) -> dict[int, dict[str, Any]]:
    return {
        int(e["draw_id"]): e
        for e in json.loads((DATA / "p8_4a_admission_escalated.json").read_bytes())["episodes"]
        if e["scenario"] == "hz1x1" and e["arm"] == arm
    }


def _grid_cityflow_registered() -> dict[int, dict[str, dict[str, Any]]]:
    """A stand-in for grid4x4's registered CityFlow anchors (in the campaign: P8.4b's cells under output/)."""
    return {
        draw: {
            arm: {"att_ours": 200.0 + bump + draw / 1000.0, "att_engine": 199.0 + bump, "created": 1300, "entered": 1290,
                  "never_entered": 10, "horizon_vehicle_count": 40.0, "episode_reward": -900.0 - bump,
                  "source": f"stand-in {arm}"}
            for arm, bump in (("maxpressure", 0.0), ("fixedtime", 30.0))
        }
        for draw in sar.HELD_OUT_DRAWS
    }


def _write_work_dir(tmp_path: Path, *, mp_shift: float = 0.0) -> Path:
    """600 stand-in cells; the SUMO ``maxpressure_1to1`` cells' E_sumo moved by *mp_shift* seconds."""
    work = tmp_path / "output" / "p9_2" / "anchors"
    work.mkdir(parents=True)
    escalated_mp = _escalated("behaviour@maxpressure")
    grid_registered = _grid_cityflow_registered()
    for cell in sar.declared_anchor_cells():
        scenario, arm, draw = cell["scenario"], cell["arm"], int(cell["draw_id"])
        if cell["backend"] == "sumo":
            payload = _sumo_payload(scenario, arm, draw)
            source = _registered_sumo(scenario)[draw][sar.BASE_ARM[arm]]
            payload.update({key: source[key] for key in _SUMO_COPIED})
            if arm == "maxpressure_1to1":
                payload["e_sumo"] = source["e_sumo"] + mp_shift
            payload["git_commit"] = COMMIT
            if scenario == "cityflow_grid4x4":
                payload["intersection_ids"] = list(source["intersection_ids"])
                payload["local_return"] = dict(source["local_return"])
                payload["local_return_from_lanes"] = dict(source["local_return"])
                payload["actions"] = [[0] * 16 for _ in range(360)]
        else:
            values = escalated_mp[draw] if scenario == "cityflow1x1" else {
                **grid_registered[draw]["maxpressure"], "att_ours": grid_registered[draw]["maxpressure"]["att_ours"] - 1.0
            }
            payload = {
                "format_version": sar.CELL_FORMAT_VERSION, "backend": "cityflow", "scenario": scenario, "arm": arm,
                "draw_id": draw,
                **{key: values[key] for key in ("att_ours", "att_engine", "created", "entered", "never_entered",
                                                 "horizon_vehicle_count", "episode_reward")},
                "roadlink_structure_digest": "f" * 64, "multiplicities": {"summary": {"selectable_greens": {}}},
                "policy_class": "MaxPressure1to1Agent", "action_sequence_sha256": "9" * 64, "decisions": 360,
                "engine_seed_requested": 1000, "seconds": 1.0, "git_commit": COMMIT, "git_dirty": False,
            }
        (work / sar.cell_name(cell)).write_text(json.dumps(payload), encoding="utf-8")
    # A gate record that authorises (BRIEF_47 Amendment A.2.2): its eight declared checks, clean provenance.
    (work / sar.REPLICA_GATE_NAME).write_text(json.dumps(_gate_record(git_commit=COMMIT)), encoding="utf-8")
    return work


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two substitutions: a stand-in chunk's commit is not a real commit, and the report's own provenance is pinned."""
    monkeypatch.setattr(sar, "_code_changed_since", lambda commit: [], raising=True)
    monkeypatch.setattr(sar, "_git_provenance", lambda: {"git_commit": COMMIT, "git_dirty": False}, raising=True)
    monkeypatch.setattr(sar, "sumo_demand_digests", lambda scenario, draw, draws_root: {
        "config_sha256": "c" * 64, "routes_sha256": "d" * 64}, raising=True)


def _report(work: Path) -> dict[str, Any]:
    return sar.build_report(
        work_dir=work, data_dir=DATA, output_root=work.parent.parent, draws_root=work,
        cityflow_registered={"cityflow1x1": sar.cityflow_registered_anchors("cityflow1x1", data_dir=DATA,
                                                                            output_root=None),
                             "cityflow_grid4x4": _grid_cityflow_registered()},
    )


# ----------------------------------------------------------------------
# The synthetic end-to-end report
# ----------------------------------------------------------------------


def test_a_report_whose_new_anchors_equal_the_registered_ones_reproduces_the_registered_rho_under_every_pair(
    tmp_path: Path, patched: None
) -> None:
    artifact = _report(_write_work_dir(tmp_path))
    assert artifact["format_version"] == "p9.2-sumo-anchors/1.0"
    assert artifact["registered_reproduced"]["all_equal"] is True
    registered = artifact["pairs"]["registered"]
    for pair in ("mp1to1_ftk4", "mpperlink_ftkcf", "mp1to1_ftkcf"):
        block = artifact["pairs"][pair]
        for mine, theirs in zip(block["p7_3a"]["by_subject_arm"], registered["p7_3a"]["by_subject_arm"]):
            assert (mine["e_sumo"], mine["att_env"]) == (theirs["e_sumo"], theirs["att_env"])
        assert block["p7_3d"]["rho"]["registered_arm"]["e_sumo"] == registered["p7_3d"]["rho"]["registered_arm"]["e_sumo"]
        assert block["p7_3c"]["arms"] == registered["p7_3c"]["arms"]
    assert artifact["h3_summary"]["cityflow1x1"]["mappo1000"]["mp1to1_ftkcf"] == {
        "clause_1": "HELD", "clause_2": "FAILED", "clause_3": "VOID", "agrees_with_registered": True
    }
    assert artifact["h3_summary"]["cityflow_grid4x4"]["mappo1000_dt_nomix_h4"]["mp1to1_ftkcf"] == {
        "clause_1": "HELD", "clause_2": "HELD", "clause_3": "(i)", "agrees_with_registered": True
    }


def test_a_pair_that_moves_a_verdict_is_reported_as_differing_and_the_sentences_use_the_aligned_pair(
    tmp_path: Path, patched: None
) -> None:
    """MP 1:1 300 s BETTER on E_sumo than the registered MaxPressure on every draw: hz1x1's rho falls below 1 under the
    pairs that use it (clause 2 HELD, against A19(a)'s FAILED), and not under the pair that keeps the registered MP."""
    artifact = _report(_write_work_dir(tmp_path, mp_shift=-300.0))
    summary = artifact["h3_summary"]["cityflow1x1"]["mappo1000"]
    assert summary["mp1to1_ftk4"] == {
        "clause_1": "HELD", "clause_2": "HELD", "clause_3": PREMISE_HOLDS, "agrees_with_registered": False
    }
    assert summary["mp1to1_ftkcf"]["agrees_with_registered"] is False
    assert summary["mpperlink_ftkcf"] == {
        "clause_1": "HELD", "clause_2": "FAILED", "clause_3": "VOID", "agrees_with_registered": True
    }
    values = artifact["sentence_values"]
    aligned = next(e for e in artifact["pairs"]["mp1to1_ftkcf"]["p7_3a"]["by_subject_arm"]
                   if e["subject"] == "mappo1000" and e["arm"] == "b_mean_k100")["e_sumo"]
    assert values["pair"] == "mp1to1_ftkcf"
    assert values["hz1x1"]["mappo1000"]["rho_aligned"]["mean"] == aligned["mean"]
    assert values["hz1x1"]["mappo1000"]["rho_aligned"]["mean"] < 1.0 < values["hz1x1"]["mappo1000"]["rho_registered"]["mean"]


def test_the_report_carries_every_anchor_on_every_draw_under_both_backends(tmp_path: Path, patched: None) -> None:
    artifact = _report(_write_work_dir(tmp_path))
    sumo = artifact["anchors"]["sumo"]
    for scenario in sar.SCENARIOS:
        by_draw = sumo[scenario]["by_draw"]
        assert sorted(by_draw) == [str(d) for d in sar.HELD_OUT_DRAWS]
        assert sorted(by_draw["1042"]) == ["fixedtime", "fixedtime_kcf", "maxpressure", "maxpressure_1to1"]
        registered = _registered_sumo(scenario)[1042]["fixedtime"]
        assert by_draw["1042"]["fixedtime"]["e_sumo"] == registered["e_sumo"]
        assert by_draw["1042"]["fixedtime_kcf"]["e_sumo"] == registered["e_sumo"]
    cityflow = artifact["anchors"]["cityflow"]
    assert sorted(cityflow["cityflow1x1"]["by_draw"]["1000"]) == ["fixedtime", "maxpressure", "maxpressure_1to1"]
    identity = cityflow["cityflow1x1"]["identity_with_registered_maxpressure"]
    assert (identity["n_draws"], identity["n_identical"]) == (100, 100)
    grid = cityflow["cityflow_grid4x4"]["beside_registered_maxpressure"]
    assert grid["n_draws"] == 100 and grid["n_att_ours_differing"] == 100


def _a27_row() -> str:
    """A27's registration row, found by its prefix and required unique -- never by line number (CLAUDE.md section 2:
    section 12 is newest-first, so every new row moves every older one)."""
    rows = [line for line in (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("| 2026-10-15 | **A27")]
    assert len(rows) == 1, len(rows)
    return rows[0]


def test_the_label_is_a27a_item_5_verbatim() -> None:
    row = _a27_row()
    start = row.index("5 — ") + len("5 — ")
    assert row[start:].startswith(sar.LABEL_POINT_5)
    assert row[start + len(sar.LABEL_POINT_5)] == ";"


def test_the_report_names_departlane_and_what_it_does_not_say(tmp_path: Path, patched: None) -> None:
    artifact = _report(_write_work_dir(tmp_path))
    assert artifact["label"] == sar.LABEL_POINT_5
    assert artifact["depart_lane"]["n_parity_route_files"] == 200
    assert artifact["depart_lane"]["n_departlane_attributes"] == 0
    assert "default" in artifact["depart_lane"]["limitation"]
    joined = " ".join(artifact["what_this_does_not_say"])
    assert "No DT was re-evaluated" in joined and "registered" in joined


# ----------------------------------------------------------------------
# Refusals, each before any write
# ----------------------------------------------------------------------


def test_a_missing_cell_refuses(tmp_path: Path, patched: None) -> None:
    work = _write_work_dir(tmp_path)
    (work / sar.cell_name(sar.declared_anchor_cells()[17])).unlink()
    with pytest.raises(ValueError, match="no chunk"):
        _report(work)


def test_an_undeclared_file_refuses(tmp_path: Path, patched: None) -> None:
    work = _write_work_dir(tmp_path)
    (work / "cell_sumo_cityflow1x1_maxpressure_1to1_draw0999.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a declared cell"):
        _report(work)


def test_an_invalid_chunk_refuses(tmp_path: Path, patched: None) -> None:
    work = _write_work_dir(tmp_path)
    path = work / sar.cell_name(sar.declared_anchor_cells()[5])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["n_teleports"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="teleport"):
        _report(work)


def test_a_missing_or_failed_replica_gate_refuses(tmp_path: Path, patched: None) -> None:
    work = _write_work_dir(tmp_path)
    gate = work / sar.REPLICA_GATE_NAME
    gate.write_text(json.dumps({"passed": False, "checks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="replica gate"):
        _report(work)
    gate.unlink()
    with pytest.raises(ValueError, match="replica gate"):
        _report(work)


def _only_draw_1001s_checks(record: dict[str, Any]) -> None:
    record["checks"] = [check for check in record["checks"] if check["draw_id"] == 1001]


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        pytest.param(lambda record: record.update(git_dirty=True), "git_dirty", id="dirty-tree"),
        pytest.param(_only_draw_1001s_checks, "lacks", id="only-draw-1001s-four-checks"),
        pytest.param(lambda record: record.update(git_commit="e" * 40), "differs from HEAD outside docs/",
                     id="run-by-other-code"),
    ],
)
def test_the_report_refuses_a_passed_gate_record_that_does_not_authorise(
    tmp_path: Path, patched: None, monkeypatch: pytest.MonkeyPatch, plant: Any, match: str
) -> None:
    """BRIEF_47 Amendment A.2.2: ``passed`` alone authorises nothing.  The gate record's OWN provenance is checked -- its
    commit "e" * 40 is the only one the stand-in J1(c) comparison reports as other code, so the chunks (at "b" * 40) are
    not what refuses -- and so are its declared checks (only draw 1001's four, as the gate wrote them before C8)."""
    work = _write_work_dir(tmp_path)
    record = _gate_record(git_commit=COMMIT)
    plant(record)
    (work / sar.REPLICA_GATE_NAME).write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(sar, "_code_changed_since",
                        lambda commit: ["offline/sumo_anchor_rescore.py"] if commit == "e" * 40 else [], raising=True)
    with pytest.raises(ValueError, match=match):
        _report(work)


def test_hz1x1s_clause_3_enters_the_sentence_values_under_the_aligned_pair(tmp_path: Path, patched: None) -> None:
    """A.2.1 and A.5: hz1x1's sentence asks whether H3's clauses agree with their registered verdicts, clause 3 included."""
    registered = _report(_write_work_dir(tmp_path / "same"))
    assert {subject: registered["sentence_values"]["hz1x1"][subject]["clause_3"] for subject in ("mappo1000", "mix50")} == {
        "mappo1000": "VOID", "mix50": "VOID"
    }
    moved = _report(_write_work_dir(tmp_path / "moved", mp_shift=-300.0))
    assert {subject: moved["sentence_values"]["hz1x1"][subject]["clause_3"] for subject in ("mappo1000", "mix50")} == {
        "mappo1000": PREMISE_HOLDS, "mix50": PREMISE_HOLDS
    }


def test_the_load_bearing_cityflow_hz1x1_identity_broken_refuses(tmp_path: Path, patched: None) -> None:
    work = _write_work_dir(tmp_path)
    cell = next(c for c in sar.declared_anchor_cells() if c["backend"] == "cityflow" and c["scenario"] == "cityflow1x1")
    path = work / sar.cell_name(cell)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["att_engine"] = payload["att_engine"] + 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="load-bearing"):
        _report(work)


def test_write_report_writes_once_and_never_into_the_worktree_from_a_test_root(tmp_path: Path, patched: None) -> None:
    artifact = _report(_write_work_dir(tmp_path))
    out = tmp_path / "artifact.json"
    sar.write_report(artifact, out)
    assert json.loads(out.read_text(encoding="utf-8"))["format_version"] == "p9.2-sumo-anchors/1.0"
    with pytest.raises(FileExistsError, match="exists"):
        sar.write_report(artifact, out)


# ----------------------------------------------------------------------
# The registered CityFlow anchors, read from their records
# ----------------------------------------------------------------------


def test_hz1x1s_registered_cityflow_anchors_are_p8_4as_records_and_reproduce_the_thresholds() -> None:
    anchors = sar.cityflow_registered_anchors("cityflow1x1", data_dir=DATA, output_root=None)
    assert sorted(anchors) == list(sar.HELD_OUT_DRAWS)
    thresholds = {int(e["draw_id"]): e for e in json.loads((DATA / "p4_heldout_thresholds.json").read_bytes())["episodes"]
                  if e["arm"] == "maxpressure"}
    assert all(anchors[d]["maxpressure"]["att_ours"] == thresholds[d]["att_horizon"] for d in sar.HELD_OUT_DRAWS)
    assert anchors[1000]["maxpressure"]["att_engine"] == _escalated("behaviour@maxpressure")[1000]["att_engine"]


def test_grid4x4s_registered_cityflow_anchors_come_from_p8_4bs_cells_checked_against_p5_2() -> None:
    value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    output = Path(value) if value else REPO_ROOT / "output"
    if not (output / "p8_4b_rederivation" / "CAMPAIGN_COMPLETE").is_file():
        pytest.skip(f"P8.4b's re-derivation cells are not under {output}: set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/")
    anchors = sar.cityflow_registered_anchors("cityflow_grid4x4", data_dir=DATA, output_root=output)
    assert sorted(anchors) == list(sar.HELD_OUT_DRAWS)
    p5_2 = {int(e["draw_id"]): e["att_horizon"]
            for e in json.loads((output / "p5_2" / "eval_maxpressure_behaviour.json").read_bytes())["episodes"]
            if e["seed"] == 101}
    assert all(anchors[d]["maxpressure"]["att_ours"] == p5_2[d] for d in sar.HELD_OUT_DRAWS)
    assert all(len(anchors[d]["fixedtime"]["source_sha256"]) == 64 for d in sar.HELD_OUT_DRAWS)


# ----------------------------------------------------------------------
# T-regress (gated until the campaign has run)
# ----------------------------------------------------------------------


def test_t_regress_the_committed_artifact_regenerates_byte_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = DATA / sar.ARTIFACT_NAME
    value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    output = Path(value) if value else REPO_ROOT / "output"
    work = output / "p9_2" / "anchors"
    if not committed.is_file() or not (work / sar.REPLICA_GATE_NAME).is_file():
        pytest.skip(
            f"{committed.name} is committed only after the campaign (G4) and its work directory {work} is gitignored: "
            "set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/ once both exist"
        )
    recorded = json.loads(committed.read_bytes())
    monkeypatch.setattr(sar, "_code_changed_since", lambda commit: [], raising=True)
    monkeypatch.setattr(sar, "_git_provenance", lambda: {"git_commit": recorded["git_commit"],
                                                        "git_dirty": recorded["git_dirty"]}, raising=True)
    artifact = sar.build_report(work_dir=work, data_dir=DATA, output_root=output,
                                draws_root=output.parent / "scenarios" / "draws")
    out = tmp_path / sar.ARTIFACT_NAME
    sar.write_report(artifact, out)
    assert out.read_bytes() == committed.read_bytes()
