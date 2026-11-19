"""P9.4 plan section 4 and 4.3 (``docs/plans/p9.4_mix50_fresh.md``): the CityFlow routes, the SUMO cells and the guards,
on synthetic records -- the simulators replaced by spies at the registered seams -- plus, at the end, the checks that
need the real checkpoints, corpus and draws, which skip without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import offline.att_rederivation as ar
import offline.method_tier_grid as grid
import offline.p9_4_evaluation as ev
import offline.p9_4_mix50_fresh as p94
import offline.transfer_calibration as tcal
import offline.transfer_curve as tc

REPO_ROOT = Path(__file__).resolve().parent.parent


def _roots(tmp_path: Path, repo_root: Path | None = None) -> p94.Roots:
    (tmp_path / "output").mkdir(exist_ok=True)
    return p94.Roots(repo_root=repo_root or REPO_ROOT, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")


def _checkpoint_payload(*, target: float = -5959.0, scale: float = 41000.0, steps: int = 40000,
                        low: float = -41000.0, high: float = -6.0, count: int = 1000) -> dict[str, Any]:
    return {"model": {"w": torch.zeros(2)}, "target_rtg": target, "rtg_scale": scale,
            "config": {"state_dim": 25, "context_length": 20}, "provenance": {"gradient_steps": steps},
            "stats": {"rtg": {"cityflow1x1": {"intersection_1_1": {"min": low, "max": high, "count": count}}}}}


def _train_fake(roots: p94.Roots, run: p94.Run, payload: dict[str, Any] | None = None) -> Path:
    path = p94.checkpoint_path(roots, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload or _checkpoint_payload(), path)
    record = {"format_version": p94.RUN_FORMAT_VERSION, "run": run.name, "group": run.group, "tier": run.tier,
              "seed": int(run.seed), "declared_gradient_steps": 40000, "gradient_steps": 40000,
              "file_sha256": p94.file_sha256(path), "canonical_digest": grid.canonical_digest_of(path)}
    p94.write_once(p94.run_record_path(roots, run), record, roots)
    return path


DECLARATION = {"rtg_scale": 41000.0, "prompts": {"b_mean_k100": {"target_rtg": -7431.0185327454665},
                                                 "naive": {"target_rtg": -5959.0}},
               "normalisation": {"support_range_over_the_split": [-41000.0, -6.0], "n_rows": 1000}}


# ----------------------------------------------------------------------
# The declared cells
# ----------------------------------------------------------------------


def test_the_sumo_cells_are_plan_4_1s_thousand_and_the_guards_plan_4_3s() -> None:
    cells, g1, g2 = ev.new_cells(), ev.g1_cells(), ev.g2_cells()
    assert (len(cells), len(g1), len(g2)) == (1000, 100, 40)
    assert {(c["subject"], c["stage"]) for c in cells} == {("mix50_fresh", "p9_4")}
    assert {c["arm"] for c in cells} == {"b_mean_k100", "naive"} == {c["arm"] for c in g1}
    assert {c["draw_id"] for c in cells} == set(range(1000, 1100))
    assert {(c["subject"], c["stage"]) for c in g1} == {("mix50", "p9_4_g1")}
    assert {c["draw_id"] for c in g1} == {c["draw_id"] for c in g2} == set(range(1000, 1010))
    assert {c["arm"] for c in g2} == {"fixedtime", "maxpressure", "fixedtime_kcf", "maxpressure_1to1"}
    assert len({tc.cell_chunk_name(c) for c in cells}) == 1000


# ----------------------------------------------------------------------
# The checkpoints at consumption, and their facts
# ----------------------------------------------------------------------


def test_the_checkpoint_identity_is_hashed_at_consumption_and_checked_against_its_record(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    run = p94.new_runs()[0]
    path = _train_fake(roots, run)
    identity = ev.new_checkpoint_identity(roots, run.seed)
    assert identity == {"subject": "mix50_fresh", "seed": 101, "path": str(path), "file_sha256": p94.file_sha256(path),
                        "sha256_checked_against": ["output/p9_4_runs/training/hz1x1/mix50_fresh_dt_seed101.json"],
                        "local_manifest": None, "deferred_56": False}
    torch.save(_checkpoint_payload(target=-1.0), path)
    with pytest.raises(ValueError, match="digests are not the ones"):
        ev.new_checkpoint_identity(roots, run.seed)
    with pytest.raises(ValueError, match="not trained"):
        ev.new_checkpoint_identity(roots, 202)


def test_the_facts_are_read_from_the_five_checkpoints_and_must_agree_and_be_declared(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    for run in p94.new_runs():
        _train_fake(roots, run)
    facts = ev.new_subject_facts(roots, DECLARATION)
    assert (facts.subject, facts.best_source_return, facts.rtg_scale, facts.support_range_over_the_split,
            facts.n_rows, facts.training_set_return_min) == ("mix50_fresh", -5959.0, 41000.0, (-41000.0, -6.0), 1000,
                                                             -41000.0)
    with pytest.raises(ValueError, match="not the declared"):
        ev.new_subject_facts(roots, {**DECLARATION, "rtg_scale": 40223.0})
    paths = [p94.checkpoint_path(roots, run) for run in p94.new_runs()]
    with pytest.raises(ValueError, match="gradient steps"):
        ev.facts_from_checkpoints("mix50_fresh", paths, steps=20)
    torch.save(_checkpoint_payload(high=-7.0), paths[3])
    with pytest.raises(ValueError, match="disagree"):
        ev.facts_from_checkpoints("mix50_fresh", paths)


# ----------------------------------------------------------------------
# Route (i) and route (ii)
# ----------------------------------------------------------------------


def _route_i(seeds: Any = p94.SEEDS, draws: Any = p94.HELD_OUT_DRAWS) -> dict[str, Any]:
    return {"format_version": grid.GRID_FORMAT_VERSION, "arm": ev.CITYFLOW_ARM, "target_rtg": -5959.0,
            "episodes": [{"arm": ev.CITYFLOW_ARM, "seed": s, "draw_id": d, "att_horizon": 100.0 + s + d / 1000.0,
                          "horizon_vehicle_count": 40.0, "episode_reward": -7000.0} for s in seeds for d in draws]}


def test_route_i_must_be_the_complete_cell() -> None:
    ev.assert_route_i_complete(_route_i())
    short = _route_i()
    short["episodes"] = short["episodes"][:-1]
    for bad in (short, {**_route_i(), "target_rtg": -7431.0}, {**_route_i(), "arm": "dt@mix50"}):
        with pytest.raises(ValueError, match="not the complete"):
            ev.assert_route_i_complete(bad)
    doubled = _route_i()
    doubled["episodes"] = doubled["episodes"] + doubled["episodes"][:1]
    with pytest.raises(ValueError, match="not the complete"):
        ev.assert_route_i_complete(doubled)


def test_route_ii_checks_route_i_values_with_p9_4s_builders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    p94.write_once(ev.route_i_output(roots), _route_i(), roots)
    with pytest.raises(FileNotFoundError, match="declare route"):
        ev.run_route_ii(roots, DECLARATION)
    ev.declare_route_ii(roots)
    ev.declare_route_ii(roots)  # a no-op the second time
    seen: dict[str, Any] = {}

    def campaign(keys: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(keys=list(keys), **kwargs)
        return {"rolled": len(keys), "n_refused": 0, "refused": []}

    monkeypatch.setattr(ar, "run_campaign", campaign)
    ev.run_route_ii(roots, DECLARATION, shard=1, of=3)
    assert seen["keys"] == ar.shard_cells(ev.route_ii_keys(), shard=1, of=3)
    assert seen["committed"][("hz1x1", ev.CITYFLOW_ARM, 303, 1050)] == 100.0 + 303 + 1.05
    assert len(seen["committed"]) == 500 and seen["engine_seed"] == 1000
    assert seen["roots"].work_dir == ev.route_ii_work(roots)
    with pytest.raises(ValueError, match="not a cell of P9.4"):
        seen["factory_builder"]("hz1x1", "dt@mix50", 101, None, device=None, config_path=None)
    with pytest.raises(ValueError, match="not P9.4's tier"):
        seen["env_settings_builder"]("hz1x1", "mix50", None)
    monkeypatch.setattr(ar, "run_campaign", lambda keys, **kwargs: {"rolled": 0, "n_refused": 1, "refused": [
        {"arm": ev.CITYFLOW_ARM, "seed": 101, "draw_id": 1000, "reason": "ValueError: does not reproduce"}]})
    with pytest.raises(RuntimeError, match="does not reproduce route"):
        ev.run_route_ii(roots, DECLARATION)


def test_a_route_ii_manifest_of_another_campaign_is_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    other = ar.campaign_manifest(ev.route_ii_keys(seeds=[101]), engine_seed=1000)
    p94.write_once(ev.route_ii_work(roots) / ar.MANIFEST_NAME, other, roots)
    with pytest.raises(ValueError, match="different"):
        ev.declare_route_ii(roots)


def test_committed_values_refuse_a_duplicate_episode() -> None:
    payload = _route_i(seeds=[101], draws=[1000])
    payload["episodes"].append(dict(payload["episodes"][0]))
    with pytest.raises(ValueError, match="twice"):
        ev.committed_from_route_i(payload)


# ----------------------------------------------------------------------
# SUMO: the new cell's inputs, the chunks on disk, the pool
# ----------------------------------------------------------------------


def test_a_new_cell_hands_run_cell_the_declared_prompt_facts_identity_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    for run in p94.new_runs():
        _train_fake(roots, run)
    seen: dict[str, Any] = {}

    def run_cell(cell: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        target = kwargs["subject_inputs"]["target_rtg"]
        return {"target_rtg": target, "rtg_first": target}

    monkeypatch.setattr(tc, "run_cell", run_cell)
    cell = ev.new_cells()[0]
    ev.new_cell_payload(roots, {**cell, "arm": "naive"}, DECLARATION)
    inputs = seen["subject_inputs"]
    assert inputs["target_rtg"] == -5959.0 and inputs["declared_gradient_steps"] == 40000
    assert inputs["checkpoint"]["file_sha256"] == p94.file_sha256(p94.checkpoint_path(roots, p94.new_runs()[0]))
    assert inputs["facts"].subject == "mix50_fresh"
    assert (seen["out_root"], seen["output_root"], seen["data_dir"]) == (roots.draws_root, roots.output_root,
                                                                       roots.data_dir)
    ev.new_cell_payload(roots, cell, DECLARATION)
    assert seen["subject_inputs"]["target_rtg"] == -7431.0185327454665
    monkeypatch.setattr(tc, "run_cell", lambda cell, **kwargs: {"target_rtg": -1.0, "rtg_first": -1.0})
    with pytest.raises(ValueError, match="not the declared"):
        ev.new_cell_payload(roots, cell, DECLARATION)
    with pytest.raises(ValueError, match="not one of P9.4"):
        ev.new_cell_payload(roots, {**cell, "subject": "mix50"}, DECLARATION)


def test_a_chunk_that_may_not_be_reused_stops_the_stage_and_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    cells = ev.new_cells()[:3]
    path = ev.cell_path(roots, "new", cells[1])
    p94.write_once(path, {"a": 1}, roots)
    before = path.read_bytes()
    monkeypatch.setattr(ev, "chunk_problems", lambda roots, kind, cell, payload, declaration, **kw: [])
    assert [c["seed"] for c in ev.pending(roots, "new", cells, DECLARATION)] == [cells[0]["seed"], cells[2]["seed"]]
    assert len(ev.pending(roots, "new", cells, DECLARATION)) == 2
    monkeypatch.setattr(ev, "chunk_problems", lambda roots, kind, cell, payload, declaration, **kw: ["invalid: x"])
    with pytest.raises(ValueError, match="left untouched"):
        ev.pending(roots, "new", cells, DECLARATION)
    assert path.read_bytes() == before and sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_chunk_problems_names_every_reason_a_new_chunk_is_not_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    cell = ev.new_cells()[0]
    monkeypatch.setattr(tc, "validate_cell_payload", lambda payload, cell=None: None)
    monkeypatch.setattr(tc, "code_changed_since", lambda commit, other="HEAD": ["offline/x.py"] if commit == "old" else [])
    monkeypatch.setattr(tc, "demand_identity_for", lambda cell, *, out_root: {"config_sha256": "c", "routes_sha256": "r"})
    monkeypatch.setattr(ev, "new_checkpoint_identity", lambda roots, seed, steps=40000: {"file_sha256": "k" * 64})
    good = {"git_commit": "new", "config_sha256": "c", "routes_sha256": "r", "checkpoint_sha256": "k" * 64,
            "target_rtg": -7431.0185327454665}
    assert ev.chunk_problems(roots, "new", cell, good, DECLARATION) == []
    bad = {**good, "git_commit": "old", "routes_sha256": "x", "checkpoint_sha256": "z", "target_rtg": -5959.0}
    problems = ev.chunk_problems(roots, "new", cell, bad, DECLARATION)
    assert len(problems) == 4 and any("other code" in p for p in problems)
    assert any("routes_sha256" in p for p in problems) and any("checkpoint" in p for p in problems)
    assert any("prompt" in p for p in problems)

    def invalid(payload: Any, cell: Any = None) -> None:
        raise ValueError("git_dirty is True")

    monkeypatch.setattr(tc, "validate_cell_payload", invalid)
    assert ev.chunk_problems(roots, "new", cell, good, DECLARATION) == ["invalid: git_dirty is True"]


class _InlinePool:
    """A stand-in for a spawn pool: the tasks run in this process, in order."""

    def __init__(self, processes: int) -> None:
        self.processes = processes

    def __enter__(self) -> _InlinePool:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def imap_unordered(self, function: Any, tasks: Any) -> Any:
        return [function(task) for task in tasks]


INLINE = SimpleNamespace(Pool=_InlinePool)


def test_roll_works_in_batches_of_workers_and_stops_between_them(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    done: list[str] = []

    def worker(task: Any) -> dict[str, Any]:
        kind, cell, context = task
        done.append(f"{cell['seed']}/{cell['draw_id']}")
        if len(done) == 4:
            p94.stop_file(roots).parent.mkdir(parents=True, exist_ok=True)
            p94.stop_file(roots).touch()
        return {"name": f"c{len(done)}", "kind": kind, "ok": True, "seconds": 1.0, "error": None}

    tasks = [("new", cell) for cell in ev.new_cells()[:9]]
    outcome = ev.roll(roots, tasks, DECLARATION, workers=2, deadline_unix=None, worker=worker, context_factory=INLINE)
    assert len(done) == 4 and outcome == {"rolled": 4, "left": 5, "stopped_by": "the stop file", "seconds": [1.0] * 4}
    with pytest.raises(ValueError, match="at most 6"):
        ev.roll(roots, tasks, DECLARATION, workers=7, deadline_unix=None, worker=worker, context_factory=INLINE)


def test_roll_stops_after_a_failed_cell_and_names_it(tmp_path: Path) -> None:
    roots = _roots(tmp_path)

    def worker(task: Any) -> dict[str, Any]:
        kind, cell, context = task
        ok = cell["draw_id"] != 1002
        return {"name": f"d{cell['draw_id']}", "kind": kind, "ok": ok, "seconds": 1.0 if ok else None,
                "error": None if ok else "ValueError: 1 teleport(s)"}

    tasks = [("new", cell) for cell in ev.new_cells()[:6]]
    with pytest.raises(RuntimeError, match=r"d1002: ValueError: 1 teleport"):
        ev.roll(roots, tasks, DECLARATION, workers=2, deadline_unix=None, worker=worker, context_factory=INLINE)


# ----------------------------------------------------------------------
# G1 and G2: compared under ==, the stage stopped on a difference
# ----------------------------------------------------------------------


@pytest.fixture()
def guard_repo(tmp_path: Path) -> Path:
    """A repo root whose docs/data holds small stand-ins of the two committed artifacts the guards read."""
    data = tmp_path / "repo" / "docs" / "data"
    data.mkdir(parents=True)
    cells = [{"kind": "dt", "subject": "mix50", "arm": c["arm"], "seed": c["seed"], "draw_id": c["draw_id"],
              "e_sumo": 500.0 + c["seed"] + c["draw_id"], "att_env": 300.0 + c["draw_id"]} for c in ev.g1_cells()]
    (data / p94.P7_3A_ARTIFACT).write_text(json.dumps({"cells": cells}), encoding="utf-8")
    by_draw = {str(d): {arm: {"e_sumo": 400.0 + d, "att_env": 350.0 + d,
                              **({"action_sequence_sha256": f"{arm}{d}"} if arm.endswith(("kcf", "1to1")) else {})}
                        for arm in p94.G2_ARMS} for d in p94.GUARD_DRAWS}
    anchors = {"anchors": {"sumo": {"cityflow1x1": {"fields": ["e_sumo", "att_env"], "by_draw": by_draw}}}}
    (data / p94.P9_2_ANCHORS).write_text(json.dumps(anchors), encoding="utf-8")
    return tmp_path / "repo"


def _write_guard_chunks(roots: p94.Roots, *, g1_shift: float = 0.0, g2_hash: str | None = None) -> None:
    for c in ev.g1_cells():
        p94.write_once(ev.cell_path(roots, "g1", c), {"e_sumo": 500.0 + c["seed"] + c["draw_id"] + g1_shift,
                                                      "att_env": 300.0 + c["draw_id"]}, roots)
    for c in ev.g2_cells():
        d, arm = c["draw_id"], c["arm"]
        p94.write_once(ev.cell_path(roots, "g2", c), {"e_sumo": 400.0 + d, "att_env": 350.0 + d,
                                                      "action_sequence_sha256": g2_hash or f"{arm}{d}"}, roots)


def test_the_guards_pass_only_when_every_compared_field_is_equal(tmp_path: Path, guard_repo: Path) -> None:
    roots = _roots(tmp_path, guard_repo)
    _write_guard_chunks(roots)
    assert ev.g1_comparison(roots, ev.g1_cells())["passed"] is True
    assert ev.g2_comparison(roots, ev.g2_cells())["passed"] is True


def test_a_guard_over_fewer_than_its_declared_cells_does_not_pass(tmp_path: Path, guard_repo: Path) -> None:
    roots = _roots(tmp_path, guard_repo)
    _write_guard_chunks(roots)
    short_g1 = ev.g1_comparison(roots, ev.g1_cells()[:99])
    short_g2 = ev.g2_comparison(roots, ev.g2_cells()[1:])
    assert (short_g1["mismatches"], short_g2["mismatches"]) == ([], [])
    assert (short_g1["passed"], short_g2["passed"]) == (False, False)


def test_g1_reports_the_differing_field_with_both_values(tmp_path: Path, guard_repo: Path) -> None:
    roots = _roots(tmp_path, guard_repo)
    _write_guard_chunks(roots, g1_shift=1e-9)
    result = ev.g1_comparison(roots, ev.g1_cells())
    assert result["passed"] is False and len(result["mismatches"]) == 100
    first = result["mismatches"][0]
    assert first["differing"] == ["e_sumo"] and first["rerolled"]["e_sumo"] != first["committed"]["e_sumo"]


def test_g2_compares_the_action_digest_where_the_record_carries_one(tmp_path: Path, guard_repo: Path) -> None:
    roots = _roots(tmp_path, guard_repo)
    _write_guard_chunks(roots, g2_hash="other")
    result = ev.g2_comparison(roots, ev.g2_cells())
    assert result["passed"] is False
    assert len(result["mismatches"]) == 20
    assert all("_fixedtime_kcf_" in m["cell"] or "_maxpressure_1to1_" in m["cell"] for m in result["mismatches"])
    assert all(m["differing"] == ["action_sequence_sha256"] for m in result["mismatches"])


def test_a_failed_guard_writes_its_record_and_stops_before_the_first_new_cell(
    tmp_path: Path, guard_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path, guard_repo)
    _write_guard_chunks(roots, g1_shift=0.5)
    monkeypatch.setattr(ev, "pending", lambda roots, kind, cells, declaration, **kw: [])
    rolled: list[str] = []
    monkeypatch.setattr(ev, "roll", lambda roots, tasks, declaration, **kw: rolled.extend(k for k, _ in tasks) or {
        "rolled": len(tasks), "left": 0, "stopped_by": None, "seconds": []})
    with pytest.raises(ev.GuardFailed, match="G1 FAILED"):
        ev.sumo_session(roots, DECLARATION, workers=2, deadline_unix=None)
    assert "new" not in rolled
    record = json.loads(ev.guards_record_path(roots).read_text(encoding="utf-8"))
    assert (record["passed"], record["g1"]["passed"], record["g2"]["passed"]) == (False, False, True)
    with pytest.raises(ev.GuardFailed, match="G1 FAILED"):
        ev.sumo_session(roots, DECLARATION, workers=2, deadline_unix=None)


def test_passed_guards_from_this_code_authorise_the_new_cells(
    tmp_path: Path, guard_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path, guard_repo)
    _write_guard_chunks(roots)
    monkeypatch.setattr(ev, "pending", lambda roots, kind, cells, declaration, **kw: [dict(c) for c in cells][:1])
    kinds: list[str] = []
    monkeypatch.setattr(ev, "roll", lambda roots, tasks, declaration, **kw: kinds.extend(k for k, _ in tasks) or {
        "rolled": len(tasks), "left": 0, "stopped_by": None, "seconds": []})
    monkeypatch.setattr(tc, "_git_provenance", lambda: {"git_commit": "a" * 40, "git_dirty": False})
    monkeypatch.setattr(tc, "code_changed_since", lambda commit, other="HEAD": [])
    # Since review B (2026-11-19) authorisation re-validates every guard chunk; these chunks are stand-ins, so the
    # validator seam is stubbed like the two above (tests/test_p9_4_evaluation_review.py covers its refusals).
    monkeypatch.setattr(ev, "chunk_problems", lambda *args, **kwargs: [])
    outcome = ev.sumo_session(roots, DECLARATION, workers=2, deadline_unix=None)
    assert kinds == ["g1", "g2", "new"] and outcome["guards"]["passed"] is True
    monkeypatch.setattr(tc, "code_changed_since", lambda commit, other="HEAD": ["offline/p9_4_evaluation.py"])
    with pytest.raises(ValueError, match="other or uncommitted code"):
        ev.sumo_session(roots, DECLARATION, workers=2, deadline_unix=None)


def test_a_session_that_ends_during_the_guards_rolls_no_new_cell(
    tmp_path: Path, guard_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path, guard_repo)
    kinds: list[str] = []
    monkeypatch.setattr(ev, "pending", lambda roots, kind, cells, declaration, **kw: [dict(c) for c in cells])
    monkeypatch.setattr(ev, "roll", lambda roots, tasks, declaration, **kw: kinds.extend(k for k, _ in tasks) or {
        "rolled": 3, "left": len(tasks) - 3, "stopped_by": "the hours bound", "seconds": []})
    outcome = ev.sumo_session(roots, DECLARATION, workers=3, deadline_unix=0.0)
    assert outcome["new"] is None and outcome["stopped_by"] == "the hours bound"
    assert set(kinds) == {"g1", "g2"} and not ev.guards_record_path(roots).exists()


# ----------------------------------------------------------------------
# The real checkpoints, corpus and draws (skip without them)
# ----------------------------------------------------------------------


def _main_roots(tmp_path: Path) -> p94.Roots:
    output = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT") or REPO_ROOT / "output")
    corpus = Path(os.environ.get("RLTRAFFIC_CORPUS_V11") or REPO_ROOT / "datasets_v11")
    draws = Path(os.environ.get("RLTRAFFIC_DRAWS_ROOT") or REPO_ROOT / "scenarios" / "draws")
    if not (output / "p4_7" / "checkpoints" / "mix50_dt_seed101.pt").is_file():
        pytest.skip(f"no P4.7 checkpoint under {output}: set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/")
    if not (corpus / "cf_hz1x1__random" / "manifest.json").is_file():
        pytest.skip(f"no corpus under {corpus}: set RLTRAFFIC_CORPUS_V11")
    if not (draws / "cityflow1x1" / "draw_1000" / "cityflow.json").is_file():
        pytest.skip(f"no materialised held-out draws under {draws}: set RLTRAFFIC_DRAWS_ROOT")
    return p94.Roots(repo_root=REPO_ROOT, output_root=output, corpus_root=corpus, draws_root=draws)


def test_on_the_main_tree_the_facts_reading_is_subject_facts_for_mix50(tmp_path: Path) -> None:
    roots = _main_roots(tmp_path)
    paths = [Path(roots.output_root) / "p4_7" / "checkpoints" / f"mix50_dt_seed{s}.pt" for s in p94.SEEDS]
    assert ev.facts_from_checkpoints("mix50", paths) == tcal.subject_facts("mix50", output_root=roots.output_root)


def test_on_the_main_tree_route_i_reproduces_p4_7s_committed_episode(tmp_path: Path) -> None:
    """The route (i) composition on the registered mix50 DT reproduces P4.7's committed att_horizon under ==."""
    pytest.importorskip("cityflow")
    roots = _main_roots(tmp_path)
    committed = next(e for e in json.loads((REPO_ROOT / "docs" / "data" / p94.P4_7_GRID).read_bytes())["episodes"]
                     if e["arm"] == "dt@mix50" and e["seed"] == 101 and e["draw_id"] == 1000)
    (episode,) = ev.evaluate_route_i(roots, grid.TIERS["mix50"],
                                     {101: Path(roots.output_root) / "p4_7" / "checkpoints" / "mix50_dt_seed101.pt"},
                                     target_rtg=-5959.0, draws=[1000], steps=40000)
    assert (episode.arm, episode.seed, episode.draw_id) == ("dt@mix50", 101, 1000)
    assert episode.att_horizon == committed["att_horizon"]
    assert episode.episode_reward == committed["episode_reward"]
