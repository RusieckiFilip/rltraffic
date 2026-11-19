"""P9.4's CityFlow routes, SUMO guards and run_cell hook after the 2026-11-19 review (review B and the pre-flight review
D; ``docs/returns/P9.4.md``): a route (ii) refusal persists and is never rewritten, every sibling shard stops at its next
episode once one exists, the routes roll what plan 4.2 says (spied at the registered seams), G1 compares att_env too,
a guard record authorises only if it re-derives from its chunks today, and the hook refuses every registered name.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import offline.att_rederivation as ar
import offline.dt_gate as gate
import offline.method_tier_grid as grid
import offline.p9_4_evaluation as ev
import offline.p9_4_mix50_fresh as p94
import offline.transfer_curve as tc
from tests.test_p9_4_hooks import _NotReached, _cell, _inputs, no_inputs_resolved  # noqa: F401 - a fixture

REPO_ROOT = Path(__file__).resolve().parent.parent


def _roots(tmp_path: Path) -> p94.Roots:
    (tmp_path / "output").mkdir(exist_ok=True)
    return p94.Roots(repo_root=REPO_ROOT, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")


def _route_i_on_disk(roots: p94.Roots) -> None:
    episodes = [{"arm": ev.CITYFLOW_ARM, "seed": seed, "draw_id": draw, "att_horizon": 100.0, "horizon_vehicle_count": 1.0,
                 "episode_reward": -1.0} for seed in p94.SEEDS for draw in p94.HELD_OUT_DRAWS]
    p94.write_once(ev.route_i_output(roots), {"format_version": grid.GRID_FORMAT_VERSION, "arm": ev.CITYFLOW_ARM,
                                              "target_rtg": p94.SOURCE_TARGET, "episodes": episodes}, roots)
    ev.declare_route_ii(roots)


def _digest(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns


# ----------------------------------------------------------------------
# Route (ii): a refusal persists, stops every shard, and is never rewritten (review B's blocking finding, D's N6)
# ----------------------------------------------------------------------


def test_route_ii_refuses_to_start_while_a_refusal_is_on_disk_and_rewrites_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _route_i_on_disk(roots)
    refusal = ev.route_ii_work(roots) / "refused_hz1x1__dt@mix50_fresh__seed101__draw1000.json"
    refusal.write_text('{"reason": "the two routes disagree"}\n', encoding="utf-8")
    before = _digest(refusal)

    def not_reached(*args: Any, **kwargs: Any) -> Any:
        raise _NotReached("run_campaign started although a refusal is on disk")

    monkeypatch.setattr(ar, "run_campaign", not_reached)
    with pytest.raises(RuntimeError, match="refusal record"):
        ev.run_route_ii(roots, {"rtg_scale": 41135.0}, shard=0, of=6)
    assert _digest(refusal) == before


def test_a_refusal_in_any_shard_stops_every_other_shard_at_its_next_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _route_i_on_disk(roots)
    seen: list[bool] = []

    def campaign(keys: Any, **kwargs: Any) -> dict[str, Any]:
        for index in range(5):
            stop = kwargs["should_stop"]()
            seen.append(stop)
            if stop:
                return {"n_refused": 0, "refused": [], "stopped": True, "not_reached": 5 - index}
            if index == 1:  # a sibling shard refuses an episode while this one rolls
                (ev.route_ii_work(roots) / "refused_hz1x1__dt@mix50_fresh__seed505__draw1099.json").write_text(
                    "{}\n", encoding="utf-8")
        return {"n_refused": 0, "refused": [], "stopped": False, "not_reached": 0}

    monkeypatch.setattr(ar, "run_campaign", campaign)
    with pytest.raises(RuntimeError, match="refusal record"):
        ev.run_route_ii(roots, {"rtg_scale": 41135.0}, shard=1, of=6)
    assert seen == [False, False, True]


def test_cityflow_is_not_complete_while_a_refusal_is_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    _route_i_on_disk(roots)
    for refused, complete in ((1, False), (0, True)):
        monkeypatch.setattr(ar, "campaign_status", lambda work, n=refused: {"complete": True, "refused": n})
        assert ev.cityflow_status(roots)["complete"] is complete


# ----------------------------------------------------------------------
# The two routes roll what plan 4.2 says (review B's M1, M2, M16)
# ----------------------------------------------------------------------


def test_route_i_rolls_each_seeds_checkpoint_at_the_source_prompt_on_engine_seed_1000(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    factories: list[tuple[Any, ...]] = []
    rollouts: list[dict[str, Any]] = []

    def factory(*args: Any) -> str:
        factories.append(args)
        return f"factory-{args[1]}"

    def evaluate(**kwargs: Any) -> list[Any]:
        rollouts.append(kwargs)
        return [SimpleNamespace(seed=kwargs["seed"], draw_id=draw, att_horizon=100.0, horizon_vehicle_count=1.0,
                                episode_reward=-1.0) for draw in kwargs["draw_ids"]]

    monkeypatch.setattr(grid, "cell_factory", factory)
    monkeypatch.setattr(grid, "env_settings_for_tiers", lambda specs, corpus_root: {"settings": specs[0].tier})
    monkeypatch.setattr(gate, "evaluate_arm", evaluate)
    monkeypatch.setattr(ev, "new_checkpoint_identity", lambda r, seed, steps=40000: {"path": f"/ckpt/{seed}.pt"})
    payload = ev.route_i_payload(roots, {"rtg_scale": 41135.0}, seeds=(101, 202), draws=(1000, 1001))
    assert factories == [("dt", "/ckpt/101.pt", 40000, -5959.0, None), ("dt", "/ckpt/202.pt", 40000, -5959.0, None)]
    assert [(r["seed"], r["draw_ids"], r["engine_seed"], r["scenario_id"], r["choose_action_factory"], r["arm"],
             r["env_settings"]) for r in rollouts] == [
        (seed, [1000, 1001], 1000, "cityflow1x1", f"factory-/ckpt/{seed}.pt", "dt@mix50_fresh", {"settings": "mix50_fresh"})
        for seed in (101, 202)]
    assert payload["target_rtg"] == -5959.0 and payload["engine_seed"] == 1000


def test_route_ii_builds_route_is_factory_at_the_source_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    factories: list[tuple[Any, ...]] = []
    monkeypatch.setattr(grid, "cell_factory", lambda *args: factories.append(args) or "factory")
    monkeypatch.setattr(ev, "new_checkpoint_identity", lambda r, seed, steps=40000: {"path": f"/ckpt/{seed}.pt"})
    built, detail = ev.factory_builder(roots)("hz1x1", "dt@mix50_fresh", 303, None, device="cpu", config_path=None)
    assert (built, factories, detail["checkpoint"]) == ("factory", [("dt", "/ckpt/303.pt", 40000, -5959.0, "cpu")],
                                                        "/ckpt/303.pt")


# ----------------------------------------------------------------------
# G1 compares att_env as well as e_sumo (review B's M4)
# ----------------------------------------------------------------------


def _p7_3a_records() -> dict[tuple[Any, ...], dict[str, Any]]:
    cells = json.loads((REPO_ROOT / "docs" / "data" / p94.P7_3A_ARTIFACT).read_bytes())["cells"]
    return {(r["kind"], r["subject"], r["arm"], r["seed"], int(r["draw_id"])): r for r in cells}


def _write_g1(roots: p94.Roots, changed: dict[str, float] | None = None) -> None:
    committed = _p7_3a_records()
    for index, cell in enumerate(ev.g1_cells()):
        record = committed[(cell["kind"], cell["subject"], cell["arm"], cell["seed"], int(cell["draw_id"]))]
        payload = {field: record[field] for field in ("e_sumo", "att_env")}
        if index == 37 and changed:
            payload.update(changed)
        p94.write_once(ev.cell_path(roots, "g1", cell), payload, roots)


def test_g1_fails_on_a_difference_in_att_env_alone(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    committed = _p7_3a_records()
    cell = ev.g1_cells()[37]
    att_env = committed[(cell["kind"], cell["subject"], cell["arm"], cell["seed"], int(cell["draw_id"]))]["att_env"]
    _write_g1(roots, {"att_env": att_env + 1e-9})
    comparison = ev.g1_comparison(roots, ev.g1_cells())
    assert comparison["passed"] is False
    assert [m["differing"] for m in comparison["mismatches"]] == [["att_env"]]


# ----------------------------------------------------------------------
# A guard record authorises only if it re-derives from its chunks today (review B, D's N10)
# ----------------------------------------------------------------------


def _write_g2(roots: p94.Roots) -> None:
    anchors = json.loads((REPO_ROOT / "docs" / "data" / p94.P9_2_ANCHORS).read_bytes())["anchors"]["sumo"]["cityflow1x1"]
    for cell in ev.g2_cells():
        record = anchors["by_draw"][str(cell["draw_id"])][cell["arm"]]
        fields = [*anchors["fields"], *(["action_sequence_sha256"] if "action_sequence_sha256" in record else [])]
        p94.write_once(ev.cell_path(roots, "g2", cell), {field: record[field] for field in fields}, roots)


@pytest.fixture()
def guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[p94.Roots, dict[str, Any]]:
    roots = _roots(tmp_path)
    _write_g1(roots)
    _write_g2(roots)
    monkeypatch.setattr(ev, "chunk_problems", lambda *args, **kwargs: [])
    monkeypatch.setattr(tc, "code_changed_since", lambda commit, other="HEAD": [])
    record = {"format_version": ev.GUARDS_FORMAT_VERSION, "task": "P9.4", "g1": ev.g1_comparison(roots, ev.g1_cells()),
              "g2": ev.g2_comparison(roots, ev.g2_cells()), "git_commit": "c" * 40, "git_dirty": False, "passed": True}
    assert record["g1"]["passed"] and record["g2"]["passed"]
    return roots, record


def test_a_passed_guard_record_that_re_derives_from_its_chunks_authorises(guarded: tuple[p94.Roots, dict[str, Any]]) -> None:
    roots, record = guarded
    ev.assert_guards_authorise(roots, record)


def test_a_guard_record_whose_chunk_changed_is_refused(guarded: tuple[p94.Roots, dict[str, Any]]) -> None:
    roots, record = guarded
    path = ev.cell_path(roots, "g1", ev.g1_cells()[5])
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    p94.write_once(path, {**payload, "e_sumo": payload["e_sumo"] + 1.0}, roots)
    with pytest.raises(ValueError, match="does not re-derive from its chunks"):
        ev.assert_guards_authorise(roots, record)


def test_a_guard_record_against_another_artifact_is_refused(guarded: tuple[p94.Roots, dict[str, Any]]) -> None:
    roots, record = guarded
    stale = {**record, "g2": {**record["g2"], "against_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="does not re-derive from its chunks"):
        ev.assert_guards_authorise(roots, stale)


def test_a_guard_record_with_a_chunk_that_may_not_be_reused_is_refused(
    guarded: tuple[p94.Roots, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, record = guarded
    monkeypatch.setattr(ev, "chunk_problems", lambda r, kind, cell, payload, d, **kw: (
        ["rolled by other code"] if kind == "g2" and cell["draw_id"] == 1004 else []))
    with pytest.raises(ValueError, match="may not be reused"):
        ev.assert_guards_authorise(roots, record)


def test_a_fresh_guard_record_is_checked_before_it_authorises(guarded: tuple[p94.Roots, dict[str, Any]],
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    roots, record = guarded
    checked: list[dict[str, Any]] = []
    monkeypatch.setattr(tc, "_git_provenance", lambda: {"git_commit": "c" * 40, "git_dirty": False})
    monkeypatch.setattr(ev, "assert_guards_authorise", lambda r, written: checked.append(dict(written)))
    outcome = ev.guards(roots, {"prompts": {}}, workers=1, deadline_unix=None)
    assert outcome["record"]["passed"] is True and checked == [outcome["record"]]


# ----------------------------------------------------------------------
# The hook refuses every registered name (review B's M13 and the docstring's claim)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("cell", "why"), [
    (_cell(subject=tc.GRID4X4_SUBJECT), "registered subject"),
    (_cell(subject="ft_k5"), "registered subject"),
    (_cell(subject="scratch_k100"), "registered subject"),
    (_cell(stage=None), "unregistered stage"),
    (_cell(stage=""), "unregistered stage"),
    *((_cell(stage=stage), "unregistered stage") for stage in ("confirmatory", "rest", "anchor", "grid4x4_confirmatory",
                                                               "p7_3c")),
])
def test_the_hook_refuses_every_registered_subject_and_stage(no_inputs_resolved: None, cell: dict[str, Any],
                                                             why: str) -> None:
    with pytest.raises(ValueError, match=why):
        tc.run_cell(cell, out_root="/nonexistent", output_root="/nonexistent",
                    subject_inputs=_inputs(str(cell.get("subject") or "mix50_fresh")))


def test_the_hook_admits_p9_4s_own_cell() -> None:
    tc._assert_subject_inputs_admissible(_cell(), _inputs("mix50_fresh"))
