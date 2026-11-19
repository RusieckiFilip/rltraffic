"""P9.4's training, declaration, barrier and command line after the 2026-11-19 review (reviews A, B and the pre-flight
review D; ``docs/returns/P9.4.md``): the session ends at its MEASURED hours bound, train_run hands train_dt P9.1's
arguments, the declared inputs are refused on any other statistics digest, plan 2.2's source-target check and check (v)
refuse what they must, the control record counts only real comparisons and re-derives from its checkpoints, a symlinked
task directory is refused, a failed write leaves nothing, and every evaluation command needs the passed control.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import offline.dt_gate as gate
import offline.method_tier_grid as grid
import offline.p9_4_evaluation as ev
import offline.p9_4_mix50_fresh as p94
import offline.p9_4_smoke as smoke
from tests.test_p9_4_training import _fake_trained, session, stubbed_control  # noqa: F401 - fixtures

REPO_ROOT = Path(__file__).resolve().parent.parent


def _roots(tmp_path: Path) -> p94.Roots:
    for name in ("output", "corpus", "draws"):
        (tmp_path / name).mkdir(exist_ok=True)
    return p94.Roots(repo_root=REPO_ROOT, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")


def _argv(tmp_path: Path, *rest: str) -> list[str]:
    _roots(tmp_path)
    return ["--repo-root", str(REPO_ROOT), "--output-root", str(tmp_path / "output"), "--corpus-root",
            str(tmp_path / "corpus"), "--draws-root", str(tmp_path / "draws"), *rest]


# ----------------------------------------------------------------------
# The session ends at its measured hours bound (review A; plan section 7)
# ----------------------------------------------------------------------


def test_a_session_ends_between_runs_at_its_measured_hours_bound(session: SimpleNamespace,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 5000.0}
    fake = p94.train_run

    def two_hours(roots: p94.Roots, run: p94.Run, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        result = fake(roots, run, inputs, **kwargs)
        clock["now"] += 2 * 3600.0  # far beyond the planning estimate of 240 s
        return result

    monkeypatch.setattr(p94, "train_run", two_hours)
    summary = p94.train_session(session.roots, hours=0.5, device="cpu", enter_regime=False, clock=lambda: clock["now"])
    assert summary["ended_by"] == "the hours bound"
    assert session.trained == ["mix50_dt_seed101"]
    p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False, clock=lambda: 0.0)
    assert session.trained == [run.name for run in p94.training_runs()]  # the next session resumes from disk


def test_the_plan_only_command_refuses_after_a_failed_control(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    p94.write_once(p94.control_record_path(roots), {"format_version": p94.CONTROL_FORMAT_VERSION,
                                                    "declared_gradient_steps": 40000, "passed": False,
                                                    "n_seeds_equal": 3, "seeds": [], "library_versions": {}}, roots)
    with pytest.raises(ValueError, match="control did not pass"):
        p94.train_plan(roots, hours=1.0)


# ----------------------------------------------------------------------
# train_run hands train_dt P9.1's arguments (review A: no test pinned them)
# ----------------------------------------------------------------------


def test_train_run_hands_train_dt_p9_1s_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    run = p94.new_runs()[2]
    stats = object()
    inputs = SimpleNamespace(spec=SimpleNamespace(tier="mix50_fresh", target_rtg=-5959.0, rtg_scale=41135.0),
                             full_batch="the full batch", top_batch="the top decile", group=(25, 4),
                             dataset=SimpleNamespace(stats=stats), scenario_id="cityflow1x1",
                             provenance={"tier": "mix50_fresh"}, digest="d" * 64, selected=tuple(range(200)))
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_train_dt(batch: Any, **kwargs: Any) -> Any:
        calls.append((batch, kwargs))
        torch.save({"model": {"w": torch.zeros(1)}}, kwargs["checkpoint_path"])
        steps = kwargs["declared_gradient_steps"]
        return SimpleNamespace(gradient_steps=steps, losses=[0.5] * steps, seconds=1.0, window_means=[0.5],
                               plateaued=False)

    monkeypatch.setattr(gate, "train_dt", fake_train_dt)
    monkeypatch.setattr(gate, "runtime_provenance", lambda: {})
    p94.train_run(roots, run, inputs, device="cpu", steps=7, log_every=3)
    final = p94.checkpoint_path(roots, run)
    assert calls == [("the full batch", {
        "state_dim": 25, "n_actions": 4, "seed": 303, "declared_gradient_steps": 7, "raise_to": None,
        "context_length": 20, "batch_size": 64, "device": "cpu", "checkpoint_path": final.with_name(final.name + ".partial"),
        "stats": stats, "scenario_id": "cityflow1x1", "target_rtg": -5959.0, "rtg_scale": 41135.0,
        "provenance": {"tier": "mix50_fresh", "task": "P9.4", "label": p94.LABEL, "run": run.name, "group": "hz1x1",
                       "plan": p94.PLAN_PATH},
        "log_every": 3})]


def test_the_budget_is_the_registered_one_everywhere() -> None:
    declaration = json.loads((REPO_ROOT / "docs" / "data" / p94.DECLARATION_NAME).read_bytes())
    assert p94.GRADIENT_STEPS == grid.DECLARED_GRADIENT_STEPS == declaration["recipe"]["gradient_steps"] == 40000


# ----------------------------------------------------------------------
# The declared inputs, plan 2.2's source target and check (v)
# ----------------------------------------------------------------------


def _inputs_like(declared: dict[str, Any], *, digest: str) -> SimpleNamespace:
    return SimpleNamespace(
        selected=[SimpleNamespace(dataset_dir=row["dataset_dir"], episode_file=row["episode_file"], ix_id=row["ix_id"])
                  for row in declared["streams"]],
        declared={"target_rtg": declared["target_rtg"], "rtg_scale": declared["rtg_scale"]}, digest=digest,
        spec=SimpleNamespace(tier="x"))


def test_both_tiers_inputs_are_refused_on_another_statistics_digest(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    declaration = p94.load_declaration(roots)
    committed = json.loads((REPO_ROOT / "docs" / "data" / p94.P4_7_DECLARATION).read_bytes())["tiers"]["mix50"]
    for build, declared, digest in (
        (lambda: p94.new_tier_inputs(roots, declaration), declaration, declaration["normalisation"]["statistics_digest"]),
        (lambda: p94.control_inputs(roots), committed, committed["statistics_digest"]),
    ):
        fake = _inputs_like(declared, digest="0" * 64)
        monkeypatch.setattr(p94, "training_inputs", lambda spec, corpus_root, f=fake: f)
        with pytest.raises(ValueError, match="statistics digest"):
            build()
        fake.digest = digest
        assert build() is fake


def _rows(expert: list[float], random_half: list[float]) -> list[dict[str, Any]]:
    return ([{"half": "expert", "total_return": value} for value in expert]
            + [{"half": "random", "total_return": value} for value in random_half])


def test_plan_2_2s_source_target_is_the_highest_return_and_lies_in_the_expert_half_only() -> None:
    assert p94.source_target_check(_rows([-5959.0, -7000.0], [-20000.0]), -5959.0) == {
        "training_return_max": -5959.0, "at_maximum": ["expert"], "passed": True}
    for rows, target in ((_rows([-7000.0], [-5959.0]), -5959.0),           # the maximum in the random half
                         (_rows([-5959.0], [-5959.0]), -5959.0),           # in both halves
                         (_rows([-5900.0], [-20000.0]), -5900.0),          # not the source target
                         (_rows([-5959.0, -5000.0], [-20000.0]), -5959.0)):  # not the highest return
        with pytest.raises(ValueError, match="plan 2.2"):
            p94.source_target_check(rows, target)


def test_check_v_needs_both_digests_as_sha256_hex() -> None:
    good = {"episode_file": "a.npz", "episode_sha256": "a" * 64, "trajectory_digest": "a" * 64, "file_sha256": "b" * 64}
    assert p94.check_v([good])["passed"] is True
    for bad in ({**good, "episode_sha256": None, "trajectory_digest": None},
                {**good, "episode_sha256": "x" * 64, "trajectory_digest": "x" * 64},
                {**good, "file_sha256": "Z" * 64}):
        assert p94.check_v([good, bad])["passed"] is False


# ----------------------------------------------------------------------
# The control record counts only real comparisons and re-derives from its checkpoints (review A, D's N10)
# ----------------------------------------------------------------------


def test_a_control_seed_with_no_parameter_compared_does_not_count(tmp_path: Path, stubbed_control: dict[int, bool],
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    for run in p94.control_runs():
        _fake_trained(roots, run)
    monkeypatch.setattr(p94, "control_comparison", lambda candidate, reference: {
        "keys_equal": True, "missing_in_candidate": [], "extra_in_candidate": [], "n_parameters": 0,
        "n_parameters_differing": 0, "largest_abs_difference": 0.0, "largest_abs_difference_parameter": None,
        "all_equal": True})
    record = p94.control_record(roots)
    assert (record["passed"], record["n_seeds_equal"]) == (False, 0)


def test_a_passed_control_record_must_re_derive_from_the_five_checkpoints(tmp_path: Path,
                                                                          stubbed_control: dict[int, bool]) -> None:
    roots = _roots(tmp_path)
    for run in p94.control_runs():
        _fake_trained(roots, run)
    record = p94.ensure_control_record(roots)
    assert record is not None and p94.assert_control_passed(roots)["passed"] is True
    path = p94.control_record_path(roots)
    for change in (lambda r: r["seeds"].pop(), lambda r: r["seeds"][2].update(candidate_file_sha256="0" * 64),
                   lambda r: r["seeds"][4].update(n_parameters=0)):
        edited = json.loads(path.read_text(encoding="utf-8"))
        change(edited)
        path.unlink()
        p94.write_once(path, edited, roots)
        with pytest.raises(ValueError, match="does not re-derive"):
            p94.assert_control_passed(roots)
        path.unlink()
        p94.write_once(path, record, roots)
    path.unlink()
    p94.write_once(path, {**record, "declared_gradient_steps": 20}, roots)
    with pytest.raises(ValueError, match="not this task's control record"):
        p94.assert_control_passed(roots)


# ----------------------------------------------------------------------
# The barrier: a symlinked task directory, a failed write, the stall dump, the smoke's roots (review D's N1-N3, N11)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("leaf", [p94.RUN_DIR, p94.RUNS_DIR])
def test_a_symlinked_task_directory_is_refused_before_any_byte(tmp_path: Path, leaf: str) -> None:
    roots = _roots(tmp_path)
    elsewhere = tmp_path / "output" / "p7_3a"
    elsewhere.mkdir()
    (tmp_path / "output" / leaf).symlink_to(elsewhere)
    with pytest.raises(PermissionError, match="is a symlink"):
        p94.write_once(tmp_path / "output" / leaf / "new.json", {"x": 1}, roots)
    assert list(elsewhere.iterdir()) == []


def test_a_failed_write_leaves_no_file_and_no_temporary(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(ValueError, match="Out of range float values"):
        p94.write_once(roots.run_root / "nan.json", {"x": float("nan")}, roots)
    assert not roots.run_root.exists() or list(roots.run_root.iterdir()) == []


def test_the_stall_dump_passes_the_barrier_before_it_is_opened(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.txt"
    with pytest.raises(PermissionError, match="P9.4 writes nowhere else"):
        p94.main(_argv(tmp_path, "route-i", "--stall-dump", str(outside)))
    assert not outside.exists()


@pytest.mark.parametrize("scratch", ["/", "/home", "/home/filip"])
def test_the_smoke_root_may_not_hold_the_main_tree_or_the_repository(scratch: str) -> None:
    with pytest.raises(ValueError, match="outside"):
        p94.smoke_roots(Path(scratch), repo_root=REPO_ROOT)


def test_the_smoke_copies_its_references_into_real_files_of_its_scratch_root_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names = [f"p4_7/checkpoints/mix50_dt_seed{seed}.pt" for seed in p94.SEEDS] + ["SHA256SUMS_p4_7.txt"]
    for name in names:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(name.encode("utf-8"))
    roots = p94.smoke_roots(tmp_path / "smoke", repo_root=REPO_ROOT)
    target = Path(roots.output_root) / names[0]
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "victim.pt")  # dangling: a copy would create the victim through it
    with pytest.raises(ValueError, match="symlink"):
        smoke.copy_references(roots, source_output=source)
    assert not (tmp_path / "victim.pt").exists()
    target.unlink()
    (Path(roots.output_root) / "p4_7").rename(tmp_path / "moved")
    (Path(roots.output_root) / "p4_7").symlink_to(tmp_path / "moved")
    with pytest.raises(ValueError, match="outside the smoke's output root"):
        smoke.copy_references(roots, source_output=source)
    assert list((tmp_path / "moved" / "checkpoints").iterdir()) == []


# ----------------------------------------------------------------------
# Every evaluation command needs the passed control and the five new runs (review B)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("command", [["route-i"], ["declare-route-ii"], ["route-ii", "--shard", "0", "--of", "6"],
                                     ["sumo", "--workers", "6", "--deadline-unix", "1"]])
def test_every_evaluation_command_refuses_before_the_passed_control(tmp_path: Path, command: list[str],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p94, "load_declaration", lambda roots: {"streams": []})
    with pytest.raises(ValueError, match="control has not run"):
        p94.main(_argv(tmp_path, *command))
    assert not ev.route_ii_work(_roots(tmp_path)).exists() and not ev.guards_dir(_roots(tmp_path)).exists()


def test_check_cityflow_refuses_while_a_route_ii_refusal_is_on_disk(tmp_path: Path, stubbed_control: dict[int, bool],
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    for run in p94.training_runs():
        _fake_trained(roots, run)
    p94.ensure_control_record(roots)
    monkeypatch.setattr(p94, "load_declaration", lambda r: {"streams": []})
    monkeypatch.setattr(p94, "assert_held_out_draws", lambda r, *, backend: None)
    assert p94.check(roots, stage="cityflow") == {"stage": "cityflow", "ok": True}
    work = ev.route_ii_work(roots)
    work.mkdir(parents=True)
    refusal = work / "refused_hz1x1__dt@mix50_fresh__seed101__draw1000.json"
    refusal.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(refusal.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="refusal record"):
        p94.check(roots, stage="cityflow")
    assert hashlib.sha256(refusal.read_bytes()).hexdigest() == digest and os.path.isfile(refusal)


# ----------------------------------------------------------------------
# On the corpus: build_declaration at this commit reproduces the committed declaration (review A)
# ----------------------------------------------------------------------


def test_on_the_corpus_build_declaration_reproduces_the_committed_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    committed = json.loads((REPO_ROOT / "docs" / "data" / p94.DECLARATION_NAME).read_bytes())
    corpus = Path(committed["dataset_dirs"][0]).parent
    configured = os.environ.get("RLTRAFFIC_CORPUS_V11")
    if not configured or Path(configured).resolve() != corpus.resolve():
        pytest.skip(f"the declared corpus {corpus} is not RLTRAFFIC_CORPUS_V11: set it to re-derive the declaration")
    monkeypatch.setattr(p94, "_code_provenance", lambda: {"code_commit": committed["git"]["commit"], "code_dirty": False})
    roots = p94.Roots(repo_root=REPO_ROOT, output_root=REPO_ROOT / "unused", corpus_root=corpus,
                      draws_root=REPO_ROOT / "unused")
    assert json.loads(p94.canonical_text(p94.build_declaration(roots))) == committed
