"""Tests for ``offline/demand_shift.py``, part 2 -- the evaluation by two routes, the guard, and the artifact
``docs/data/p9_2_demand_shift.json`` (``BRIEF_47`` §3.4, §4 test 4).

* **pure**: the declaration (15,600 stage-1 cells, 3,000 stage-2), the ``random_fresh`` schedule as a function of
  ``(seed, draw)`` only (recomputed by hand), the two-route agreement refusing a disagreement, rho on a file's own anchors
  with exact dyadic values and the co-reported zero-denominator rule, the report over a small declared set with both random
  references beside each other, and ``what_this_is_not`` verbatim against the registration.
* **gated**: one bc-tyc held-out cell reproduced under ``==`` through the evaluation path against P8.4b's committed cell (the
  tautology guard); sigma = 0 across engine seeds; one kn-hz draw materialised into ``tmp_path`` and one cell rolled on it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import offline.demand_shift as ds

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
COMMIT = "c" * 40


# ----------------------------------------------------------------------
# The declaration
# ----------------------------------------------------------------------


def test_the_stage_1_declaration_is_15600_cells_and_stage_2_is_3000() -> None:
    cells = ds.declared_cells(stage=1)
    assert len(cells) == 15600
    per_file = {name: [c for c in cells if c["file"] == name] for name in ds.SHIFTED_FILES}
    assert {name: len(group) for name, group in per_file.items()} == {"kn-hz": 5200, "qc-yn": 5200, "sb-sx": 5200}
    arms = sorted({c["arm"] for c in cells})
    assert arms == sorted([
        "dt@mappo1000", "bc@mappo1000", "bc_top10@mappo1000", "iql@mappo1000",
        "dt@mix50", "bc@mix50", "bc_top10@mix50", "iql@mix50",
        "behaviour@fixedtime", "behaviour@maxpressure", "behaviour@random", "random_fresh",
    ])
    assert {c["seed"] for c in cells if c["arm"] in ("behaviour@fixedtime", "behaviour@maxpressure")} == {None}
    assert {c["seed"] for c in cells if c["arm"] == "random_fresh"} == {101, 202, 303, 404, 505}
    assert sorted({c["draw_id"] for c in cells}) == list(range(1000, 1100))
    names = [ds.cell_name(c) for c in cells]
    assert len(set(names)) == len(names)
    stage_2 = ds.declared_cells(stage=2)
    assert len(stage_2) == 3000 and {c["arm"] for c in stage_2} == {"iql_t@mappo1000", "iql_t@mix50"}


# ----------------------------------------------------------------------
# random_fresh: a function of (seed, draw) only
# ----------------------------------------------------------------------


class _Ix:
    def __init__(self, ix_id: str) -> None:
        self.id = ix_id
        self.num_phases = 8


def _stub_env() -> SimpleNamespace:
    return SimpleNamespace(intersections=[_Ix("a"), _Ix("b")], action_space=None, control_mode="acyclic")


def _info() -> dict[str, Any]:
    return {"intersections": {ix: {"avail_actions": list(range(8))} for ix in ("a", "b")}}


def _schedule(seed: int, draw: int, steps: int = 30) -> list[list[int]]:
    choose = ds.random_fresh_factory(seed, draw)(_stub_env())
    return [[int(a) for a in choose(None, _info())] for _ in range(steps)]


def test_the_random_fresh_schedule_depends_on_seed_and_draw_only() -> None:
    assert _schedule(101, 1000) == _schedule(101, 1000)
    assert _schedule(101, 1000) != _schedule(101, 1001)
    assert _schedule(101, 1000) != _schedule(202, 1000)


def test_the_random_fresh_schedule_is_default_rng_seed_draw_by_hand() -> None:
    """A27(d)'s formula recomputed WITHOUT collect: one ``integers(8)`` per intersection per step, in id order."""
    rng = np.random.default_rng([101, 1000])
    by_hand = [[int(rng.integers(8)) for _ix in ("a", "b")] for _ in range(30)]
    assert _schedule(101, 1000) == by_hand


# ----------------------------------------------------------------------
# The two routes, and the cell record
# ----------------------------------------------------------------------


def _route_results(att: float, *, att_engine: float = 100.0, digest: str = "d" * 64) -> tuple[Any, Any]:
    p4_6 = SimpleNamespace(att_horizon=att, horizon_vehicle_count=10.0, episode_reward=-50.0, action_sequence_sha256=digest)
    p8_4b = SimpleNamespace(
        att_ours=att, att_engine=att_engine, created=100, entered=99, never_entered=1, entered_fraction=0.99,
        horizon_vehicle_count=10.0, episode_reward=-50.0, seconds=1.0, seconds_rollout=0.9,
        action_sequence_sha256=digest,
    )
    return p4_6, p8_4b


def test_the_two_routes_must_agree_on_att_ours_under_equality() -> None:
    cell = {"file": "kn-hz", "scenario_key": "cityflow1x1_config3", "arm": "behaviour@maxpressure", "seed": None,
            "draw_id": 1000, "stage": 1}
    record = ds.cell_record(cell, *_route_results(150.0), source={"kind": "algorithmic"})
    assert (record["att_ours"], record["att_ours_p4_6_route"], record["att_engine"]) == (150.0, 150.0, 100.0)
    p4_6, p8_4b = _route_results(150.0)
    p4_6.att_horizon = float(np.nextafter(150.0, np.inf))
    with pytest.raises(ValueError, match="two routes"):
        ds.cell_record(cell, p4_6, p8_4b, source={"kind": "algorithmic"})
    p4_6, p8_4b = _route_results(150.0)
    p4_6.action_sequence_sha256 = "e" * 64
    with pytest.raises(ValueError, match="two routes"):
        ds.cell_record(cell, p4_6, p8_4b, source={"kind": "algorithmic"})


# ----------------------------------------------------------------------
# rho on a file's own anchors, both definitions
# ----------------------------------------------------------------------


def _rec(arm: str, seed: int | None, draw: int, ours: float, engine: float) -> dict[str, Any]:
    return {"file": "kn-hz", "arm": arm, "seed": seed, "draw_id": draw, "att_ours": ours, "att_engine": engine,
            "horizon_vehicle_count": 10.0, "episode_reward": -1.0}


def test_rho_on_the_files_own_anchors_is_exact_and_a_zero_denominator_is_excluded() -> None:
    """FT 300 / MP 200 on att_ours (draws 1000, 1001); the arm at 250 and 225 -> rho 0.5 and 0.75.  On att_engine draw
    1001's anchors are equal, so that draw is excluded from att_engine and listed, never a zero."""
    records = [
        _rec("behaviour@fixedtime", None, 1000, 300.0, 310.0), _rec("behaviour@maxpressure", None, 1000, 200.0, 210.0),
        _rec("behaviour@fixedtime", None, 1001, 300.0, 250.0), _rec("behaviour@maxpressure", None, 1001, 200.0, 250.0),
    ]
    records += [_rec("dt@mappo1000", seed, 1000, 250.0, 260.0) for seed in (101, 202, 303, 404, 505)]
    records += [_rec("dt@mappo1000", seed, 1001, 225.0, 200.0) for seed in (101, 202, 303, 404, 505)]
    block = ds.file_block(records, arms=["dt@mappo1000"])
    rho = block["arms"]["dt@mappo1000"]["rho"]
    s = math.sqrt(((0.5 - 0.625) ** 2 + (0.75 - 0.625) ** 2) / (2 - 1))
    assert (rho["att_ours"]["mean"], rho["att_ours"]["std"], rho["att_ours"]["n_draws"]) == (0.625, s, 2)
    assert rho["att_engine"]["n_draws"] == 1 and rho["att_engine"]["mean"] == (310.0 - 260.0) / (310.0 - 210.0)
    assert block["anchors"]["excluded_draws"]["att_engine"] == [1001]
    raw = block["arms"]["dt@mappo1000"]["att"]["att_ours"]
    assert (raw["mean"], raw["n_draws"]) == ((250.0 + 225.0) / 2, 2)


# ----------------------------------------------------------------------
# what_this_is_not, verbatim
# ----------------------------------------------------------------------


def test_what_this_is_not_carries_a27h_s_three_sentences_and_a25a_s_amended_sentence_verbatim() -> None:
    # A27's row is found by its prefix and required unique -- never by line number (CLAUDE.md section 2: section 12 is
    # newest-first, so every new registration row moves every older one).
    rows = [line for line in (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("| 2026-10-15 | **A27")]
    assert len(rows) == 1, len(rows)
    row = rows[0]
    plain = row.replace("**", "").replace("*", "")
    assert len(ds.WHAT_THIS_IS_NOT) == 4
    for sentence in ds.WHAT_THIS_IS_NOT:
        assert sentence in plain, sentence[:60]
    assert ds.WHAT_THIS_IS_NOT[3].startswith("H2's registered 2×2 design remains untested")
    assert ds.LABEL == "exploration, not H2"
    assert "7 — exploration, not H2;" in row


# ----------------------------------------------------------------------
# A guard record authorises only with clean provenance and every declared check (BRIEF_47 Amendment A.2.2)
# ----------------------------------------------------------------------

#: The guard's declared cells, written out: the eight stage-1 learner arms at seed 101 and the three registered references
#: (the two deterministic ones in their single slot), all on bc-tyc draw 1000; then the engine-seed check's four files.
GUARD_ARMS: list[tuple[str, int | None]] = [
    ("dt@mappo1000", 101), ("bc@mappo1000", 101), ("bc_top10@mappo1000", 101), ("iql@mappo1000", 101),
    ("dt@mix50", 101), ("bc@mix50", 101), ("bc_top10@mix50", 101), ("iql@mix50", 101),
    ("behaviour@fixedtime", None), ("behaviour@maxpressure", None), ("behaviour@random", 101),
]
ENGINE_SEED_CELLS: list[str] = [
    f"cell_{name}_behaviour_at_maxpressure_seednone_draw1000.json" for name in ("bc-tyc", "kn-hz", "qc-yn", "sb-sx")
]


def _guard_record(**overrides: Any) -> dict[str, Any]:
    """A guard record as ``run_guard`` writes one that passed."""
    record: dict[str, Any] = {
        "format_version": "p9.2-demand-shift-guard/1.0",
        "what": "stand-in",
        "cells": [{"arm": arm, "seed": seed, "draw_id": 1000, "differing": [], "passed": True,
                   "committed_source": "stand-in", "committed_sha256": "0" * 64} for arm, seed in GUARD_ARMS],
        "random_fresh_digest_bc_tyc_seed101_draw1000": "a" * 64,
        "engine_seed_check": {
            "checks": [{"cell": name, "engine_seeds": [1000, 1001, 1002], "att_ours": [150.0] * 3,
                        "att_engine": [140.0] * 3, "passed": True} for name in ENGINE_SEED_CELLS],
            "passed": True,
        },
        "passed": True,
        "git_commit": COMMIT,
        "git_dirty": False,
    }
    record.update(overrides)
    return record


@pytest.fixture()
def same_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in record's commit is not a real commit: J1(c)'s comparison is substituted, and nothing else."""
    monkeypatch.setattr(ds, "_code_changed_since", lambda commit: [], raising=True)


def test_a_guard_record_with_clean_provenance_and_every_declared_check_authorises(same_code: None) -> None:
    ds.assert_guard_authorises(_guard_record(), source="stand-in")
    assert [(cell["arm"], cell["seed"]) for cell in ds.declared_guard_cells()] == GUARD_ARMS
    assert [ds.cell_name(cell) for cell in ds.declared_engine_seed_cells()] == ENGINE_SEED_CELLS


def _seed_check(index: int, **fields: Any) -> Any:
    return lambda record: record["engine_seed_check"]["checks"][index].update(fields)


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        pytest.param(lambda r: r.update(git_dirty=True), "git_dirty", id="dirty-tree"),
        pytest.param(lambda r: r.pop("git_dirty"), "git_dirty", id="dirtiness-not-measured"),
        pytest.param(lambda r: r.update(git_commit="HEAD"), "40-character", id="commit-not-a-commit"),
        pytest.param(lambda r: r.update(passed=False), "did not pass", id="guard-failed"),
        pytest.param(lambda r: r.update(format_version="p9.2-demand-shift-guard/0.9"), "format_version",
                     id="other-format"),
        pytest.param(lambda r: r["cells"].pop(7), "lacks", id="an-arm-missing"),
        pytest.param(lambda r: r["cells"].append(dict(r["cells"][0])), "twice", id="an-arm-twice"),
        pytest.param(lambda r: r["cells"][2].update(seed=202), "checked at seed 202", id="an-arm-at-another-seed"),
        pytest.param(lambda r: r["cells"][2].update(draw_id=1001), "checked on draw 1001", id="an-arm-on-another-draw"),
        pytest.param(lambda r: r["cells"][4].update(passed=False), "did not reproduce", id="an-arm-failed"),
        pytest.param(lambda r: r["cells"][4].update(differing=["att_engine"]), "did not reproduce",
                     id="a-differing-field-marked-passed"),
        pytest.param(lambda r: r["cells"].append({**r["cells"][0], "arm": "random_fresh"}), "not a declared",
                     id="an-undeclared-arm"),
        pytest.param(lambda r: r["engine_seed_check"].update(passed=False), "engine-seed check", id="seed-block-failed"),
        pytest.param(lambda r: r["engine_seed_check"]["checks"].pop(3), "engine-seed check", id="a-file-unchecked"),
        pytest.param(_seed_check(1, passed=False), "engine-seed check", id="a-file-check-failed"),
        pytest.param(_seed_check(2, engine_seeds=[1000, 1001]), "engine seeds", id="two-engine-seeds-only"),
    ],
)
def test_a_guard_record_is_refused_for_each_planted_defect(same_code: None, plant: Any, match: str) -> None:
    record = _guard_record()
    plant(record)
    with pytest.raises(ValueError, match=match):
        ds.assert_guard_authorises(record, source="stand-in")


def test_a_guard_record_run_by_other_code_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def changed(commit: str) -> list[str]:
        asked.append(commit)
        return ["offline/demand_shift.py"]

    monkeypatch.setattr(ds, "_code_changed_since", changed, raising=True)
    with pytest.raises(ValueError, match="differs from HEAD outside docs/"):
        ds.assert_guard_authorises(_guard_record(), source="stand-in")
    assert asked == [COMMIT]


def _write_guard(work: Path, record: dict[str, Any]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / ds.GUARD_NAME).write_text(json.dumps(record), encoding="utf-8")


def test_evaluate_rolls_nothing_past_a_passed_guard_record_from_a_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_code: None
) -> None:
    """The case a bare ``passed`` check let through: the record says passed, its tree was dirty."""
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    _write_guard(work, _guard_record(git_dirty=True))

    def never(cell: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a cell was rolled past a guard record that does not authorise")

    monkeypatch.setattr(ds, "roll_cell", never, raising=True)
    code = ds.main(["--output-root", str(tmp_path / "output"), "--work-dir", str(work), "evaluate", "--stage", "1",
                    "--shard", "0", "--of", "5200"])
    assert code == 2
    assert sorted(path.name for path in work.iterdir()) == [ds.GUARD_NAME]


def test_evaluate_rolls_its_shard_past_a_guard_record_that_authorises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_code: None
) -> None:
    """The control of the test above: the same command past an authorising record rolls its shard's three cells."""
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    _write_guard(work, _guard_record())
    monkeypatch.setattr(ds, "roll_cell", lambda cell, **_kwargs: _fake_roll(cell), raising=True)
    code = ds.main(["--output-root", str(tmp_path / "output"), "--work-dir", str(work), "evaluate", "--stage", "1",
                    "--shard", "0", "--of", "5200"])
    assert code == 0
    assert len([path for path in work.iterdir() if path.name.startswith("cell_")]) == 3


def test_check_guard_answers_0_only_for_a_record_that_authorises_and_writes_nothing(
    tmp_path: Path, same_code: None, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    command = ["--work-dir", str(work), "check-guard"]
    assert ds.main(command) == 2
    assert not work.exists(), "a check of a missing record creates nothing"
    for refused in (_guard_record(passed=False), _guard_record(git_dirty=True), _guard_record(cells=[])):
        _write_guard(work, refused)
        assert ds.main(command) == 2
    assert capsys.readouterr().out.count("REFUSED") == 4
    _write_guard(work, _guard_record())
    assert ds.main(command) == 0
    assert sorted(path.name for path in work.iterdir()) == [ds.GUARD_NAME]


# ----------------------------------------------------------------------
# The report over a small declared set
# ----------------------------------------------------------------------


def _small_declaration() -> list[dict[str, Any]]:
    return [c for c in ds.declared_cells(stage=1) if c["file"] == "kn-hz" and c["draw_id"] in (1000, 1001)]


def _write_cells(work: Path, cells: list[dict[str, Any]]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells):
        base = 300.0 if cell["arm"] == "behaviour@fixedtime" else 200.0 if cell["arm"] == "behaviour@maxpressure" else (
            400.0 if cell["arm"] in ("behaviour@random", "random_fresh") else 250.0)
        att = base + cell["draw_id"] - 1000 + (cell["seed"] or 0) / 1000.0
        record = {
            "format_version": ds.CELL_FORMAT_VERSION, **cell, "att_ours": att, "att_ours_p4_6_route": att,
            "att_engine": att - 3.0, "created": 743, "entered": 740, "never_entered": 3, "entered_fraction": 740 / 743,
            "horizon_vehicle_count": 10.0, "episode_reward": -float(index), "seconds": 1.0, "seconds_rollout": 0.9,
            "action_sequence_sha256": f"{index:064x}", "policy_source": {"kind": "stand-in"}, "checkpoint_sha256": None,
            "git_commit": COMMIT, "git_dirty": False,
        }
        (work / ds.cell_name(cell)).write_text(json.dumps(record), encoding="utf-8")
    # A guard record that authorises (BRIEF_47 Amendment A.2.2): every declared check, clean provenance.
    (work / ds.GUARD_NAME).write_text(json.dumps(_guard_record()), encoding="utf-8")


def _audit_file(tmp_path: Path) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"format_version": ds.AUDIT_FORMAT_VERSION, "passed": True, "files": {}, "sources": {},
                                "expected_flow_sha256": {}}), encoding="utf-8")
    return path


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "_code_changed_since", lambda commit: [], raising=True)
    monkeypatch.setattr(ds, "_git_provenance", lambda: {"git_commit": COMMIT, "git_dirty": False}, raising=True)


def test_the_report_over_a_declared_set_carries_both_random_references_beside_each_other(
    tmp_path: Path, patched: None
) -> None:
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    cells = _small_declaration()
    _write_cells(work, cells)
    artifact = ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)
    assert artifact["label"] == "exploration, not H2"
    assert artifact["what_this_is_not"] == list(ds.WHAT_THIS_IS_NOT)
    block = artifact["files"]["kn-hz"]
    dt = block["arms"]["dt@mappo1000"]
    assert sorted(dt["paired"]) == ["behaviour@fixedtime", "behaviour@random", "random_fresh"]
    assert dt["paired"]["behaviour@fixedtime"]["att_ours"]["mean_difference"] < 0.0
    assert block["random_references"] == {"registered": "behaviour@random", "fresh": "random_fresh"}
    assert len(artifact["cells"]) == len(cells)
    assert artifact["stage_2"]["status"] == "pending"
    assert artifact["no_single_definition_claim"].startswith("Both ATT definitions")


def test_the_report_refuses_a_missing_cell_and_a_missing_guard(tmp_path: Path, patched: None) -> None:
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    cells = _small_declaration()
    _write_cells(work, cells)
    (work / ds.cell_name(cells[3])).unlink()
    with pytest.raises(ValueError, match="no record"):
        ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)
    _write_cells(work, cells)
    (work / ds.GUARD_NAME).unlink()
    with pytest.raises(ValueError, match="guard"):
        ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        pytest.param(lambda record: record.update(git_dirty=True), "git_dirty", id="dirty-tree"),
        pytest.param(lambda record: record["engine_seed_check"]["checks"].pop(), "engine-seed check",
                     id="a-file-unchecked"),
        pytest.param(lambda record: record.update(git_commit="e" * 40), "differs from HEAD outside docs/",
                     id="run-by-other-code"),
    ],
)
def test_the_report_refuses_a_passed_guard_record_that_does_not_authorise(
    tmp_path: Path, patched: None, monkeypatch: pytest.MonkeyPatch, plant: Any, match: str
) -> None:
    """BRIEF_47 Amendment A.2.2: the guard record's OWN provenance and declared checks -- its commit "e" * 40 is the only
    one the stand-in J1(c) comparison reports as other code, so the cell records (at "c" * 40) are not what refuses."""
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    cells = _small_declaration()
    _write_cells(work, cells)
    record = _guard_record()
    plant(record)
    _write_guard(work, record)
    monkeypatch.setattr(ds, "_code_changed_since",
                        lambda commit: ["offline/demand_shift.py"] if commit == "e" * 40 else [], raising=True)
    with pytest.raises(ValueError, match=match):
        ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)


def test_the_report_refuses_a_record_whose_two_routes_disagree(tmp_path: Path, patched: None) -> None:
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    cells = _small_declaration()
    _write_cells(work, cells)
    path = work / ds.cell_name(cells[0])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["att_ours_p4_6_route"] = record["att_ours"] + 1.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="two routes"):
        ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)


def test_the_committed_artifact_is_never_written_from_a_partial_declaration(tmp_path: Path, patched: None) -> None:
    """The refusal leaves the committed artifact exactly as it was -- absent before G4, its committed bytes after.  (Until
    G4 part 2 this asserted the file's absence, which held only while no artifact existed; ``BRIEF_47`` Amendment D.4.)"""
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    cells = _small_declaration()
    _write_cells(work, cells)
    artifact = ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)
    committed = DATA / ds.ARTIFACT_NAME
    before = committed.read_bytes() if committed.exists() else None
    with pytest.raises(ValueError, match="partial"):
        ds.write_report(artifact, committed)
    assert (committed.read_bytes() if committed.exists() else None) == before


def test_t_regress_the_committed_artifact_regenerates_byte_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``docs/data/p9_2_demand_shift.json`` was built in the run tree that rolled its cells and copied byte for byte (D.4);
    this branch's code regenerates it from the same records with the two provenance substitutions every regression in this
    repository uses (a record's commit is compared as the run tree's, the artifact's own provenance is the recorded one) and
    nothing else.  Gated: the records live under the main tree's ``output/``."""
    committed = DATA / ds.ARTIFACT_NAME
    value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    output = Path(value) if value else REPO_ROOT / "output"
    work = output / "p9_2" / "demand_shift"
    if not committed.is_file() or not (work / ds.GUARD_NAME).is_file():
        pytest.skip(
            f"{committed.name} is committed only after the campaign (G4) and its work directory {work} is gitignored: "
            "set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/ once both exist"
        )
    recorded = json.loads(committed.read_bytes())
    monkeypatch.setattr(ds, "_code_changed_since", lambda commit: [], raising=True)
    monkeypatch.setattr(ds, "_git_provenance", lambda: {"git_commit": recorded["git_commit"],
                                                       "git_dirty": recorded["git_dirty"]}, raising=True)
    artifact = ds.build_report(work_dir=work, audit_path=DATA / ds.AUDIT_NAME)
    out = tmp_path / ds.ARTIFACT_NAME
    ds.write_report(artifact, out)
    assert out.read_bytes() == committed.read_bytes()


# ----------------------------------------------------------------------
# Gated: the guard on real files
# ----------------------------------------------------------------------


def _roots() -> Any:
    from offline.admission_probe import ProbeRoots

    draws = Path(os.environ.get("RLTRAFFIC_DRAWS") or REPO_ROOT / "scenarios" / "draws")
    output = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT") or REPO_ROOT / "output")
    corpus = Path(os.environ.get("RLTRAFFIC_CORPUS_V11") or REPO_ROOT / "datasets_v11")
    if not (draws / "cityflow1x1" / "draw_1000" / "cityflow.json").is_file():
        pytest.skip(f"bc-tyc's held-out draws are not at {draws}: set RLTRAFFIC_DRAWS to the main tree's scenarios/draws")
    if not (output / "p8_4b_rederivation" / "CAMPAIGN_COMPLETE").is_file() or not (output / "p4_dt").is_dir():
        pytest.skip(f"the checkpoints and P8.4b's cells are not under {output}: set RLTRAFFIC_OUTPUT_ROOT to output/")
    if not (corpus / "cf_hz1x1__maxpressure" / "manifest.json").is_file():
        pytest.skip(f"the v1.1 corpus manifests are not at {corpus}: set RLTRAFFIC_CORPUS_V11 to datasets_v11/")
    try:
        import cityflow  # noqa: F401
    except ImportError:
        pytest.skip("the CityFlow engine is not installed")
    return ProbeRoots(repo_root=REPO_ROOT, corpus_root=corpus, draws_root=draws, output_root=output,
                      work_dir=output / "unused")


def test_one_bc_tyc_held_out_cell_reproduces_p8_4bs_committed_cell_through_the_evaluation_path() -> None:
    """The tautology guard: dt@mappo1000, seed 101, draw 1000 on bc-tyc, BOTH routes, every compared field ``==``."""
    roots = _roots()
    cell = {"file": "bc-tyc", "scenario_key": "cityflow1x1", "arm": "dt@mappo1000", "seed": 101, "draw_id": 1000,
            "stage": 0}
    record = ds.roll_cell(cell, roots=roots, device="cpu")
    committed = ds.p8_4b_committed(cell, output_root=roots.output_root)
    assert {f: record[f] for f in ds.GUARD_FIELDS} == {f: committed[f] for f in ds.GUARD_FIELDS}


def test_cityflow_ignores_the_engine_seed_on_this_network() -> None:
    roots = _roots()
    result = ds.engine_seed_check({"file": "bc-tyc", "scenario_key": "cityflow1x1", "arm": "behaviour@maxpressure",
                                   "seed": None, "draw_id": 1000, "stage": 0}, roots=roots, device="cpu")
    assert result["passed"] is True and result["engine_seeds"] == [1000, 1001, 1002]
    assert len(set(result["att_ours"])) == 1 and len(set(result["att_engine"])) == 1


def test_one_shifted_draw_materialised_into_tmp_path_rolls_one_valid_cell(tmp_path: Path) -> None:
    from offline.admission_probe import ProbeRoots
    from offline.materialise_draws import materialise

    roots = _roots()
    draws = tmp_path / "draws"
    materialise(REPO_ROOT / ds.SHIFTED_FILES["kn-hz"]["config"], [1000], out_root=draws)
    shifted_roots = ProbeRoots(repo_root=REPO_ROOT, corpus_root=roots.corpus_root, draws_root=draws,
                               output_root=roots.output_root, work_dir=roots.work_dir)
    cell = {"file": "kn-hz", "scenario_key": "cityflow1x1_config3", "arm": "random_fresh", "seed": 101, "draw_id": 1000,
            "stage": 1}
    record = ds.roll_cell(cell, roots=shifted_roots, device="cpu")
    assert record["created"] <= 743 and record["att_ours"] == record["att_ours_p4_6_route"]
    assert record["created"] == record["entered"] + record["never_entered"]
    shutil.rmtree(draws)


# ----------------------------------------------------------------------
# The sharded, resumable loop and the command line's refusals
# ----------------------------------------------------------------------


def _fake_roll(cell: dict[str, Any]) -> dict[str, Any]:
    p4_6, p8_4b = _route_results(100.0 + cell["draw_id"] - 1000)
    record = ds.cell_record(cell, p4_6, p8_4b, source={"kind": "stand-in"})
    record["git_dirty"] = False
    record["git_commit"] = COMMIT
    return record


def test_the_loop_partitions_by_shard_resumes_and_writes_nothing_for_a_refused_cell(tmp_path: Path) -> None:
    cells = _small_declaration()[:6]
    work = tmp_path / "w"
    seen: list[str] = []
    for shard in range(2):
        seen += ds.evaluate_cells(cells, work_dir=work, shard=shard, of=2, roll_fn=_fake_roll)["rolled_names"]
    assert sorted(seen) == sorted(ds.cell_name(c) for c in cells)
    again = ds.evaluate_cells(cells, work_dir=work, shard=0, of=1, roll_fn=_fake_roll)
    assert (again["rolled"], again["skipped"]) == (0, 6)

    def dirty(cell: dict[str, Any]) -> dict[str, Any]:
        record = _fake_roll(cell)
        record["git_dirty"] = True
        return record

    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="git_dirty"):
        ds.evaluate_cells(cells, work_dir=fresh, shard=0, of=1, roll_fn=dirty)
    assert not fresh.exists()


def test_evaluate_refuses_without_a_passing_guard(tmp_path: Path) -> None:
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    code = ds.main(["--output-root", str(tmp_path / "output"), "--work-dir", str(work), "evaluate", "--stage", "1"])
    assert code == 2
    assert not work.exists()


def test_stage_2_refuses_without_pins(tmp_path: Path) -> None:
    cell = ds.declared_cells(stage=2)[0]
    with pytest.raises(ValueError, match="pending"):
        ds._factory_for(cell, roots=None, device="cpu", config=tmp_path / "x.json", iql_t_pins=None)


def test_the_report_compares_the_random_fresh_digest_across_files_with_bc_tycs(tmp_path: Path, patched: None) -> None:
    """A27(h): every legal-action mask is all-true, so (seed 101, draw 1000)'s schedule is the same on every file; the
    report RECORDS whether the digests agree rather than assuming it."""
    work = tmp_path / "output" / "p9_2" / "demand_shift"
    cells = _small_declaration()
    _write_cells(work, cells)
    guard = json.loads((work / ds.GUARD_NAME).read_text(encoding="utf-8"))
    fresh = next(c for c in cells if c["arm"] == "random_fresh" and c["seed"] == 101 and c["draw_id"] == 1000)
    digest = json.loads((work / ds.cell_name(fresh)).read_text(encoding="utf-8"))["action_sequence_sha256"]
    guard["random_fresh_digest_bc_tyc_seed101_draw1000"] = digest
    (work / ds.GUARD_NAME).write_text(json.dumps(guard), encoding="utf-8")
    artifact = ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)
    assert artifact["random_fresh_seed101_draw1000"] == {"digests": {"bc-tyc": digest, "kn-hz": digest},
                                                          "identical": True}
    guard["random_fresh_digest_bc_tyc_seed101_draw1000"] = "0" * 64
    (work / ds.GUARD_NAME).write_text(json.dumps(guard), encoding="utf-8")
    artifact = ds.build_report(work_dir=work, audit_path=_audit_file(tmp_path), declared=cells)
    assert artifact["random_fresh_seed101_draw1000"]["identical"] is False
