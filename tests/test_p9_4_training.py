"""P9.4 plan section 3 (``docs/plans/p9.4_mix50_fresh.md``): the barrier, the run records, the control and the training
sessions -- the control first, 5 of 5 or nothing else trains -- on synthetic runs, with the trainer replaced by a spy
that writes what ``train_run`` writes.  The real trainer, corpus and checkpoints are read only by the tests at the end,
which skip without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import offline.method_tier_grid as grid
import offline.p9_4_mix50_fresh as p94

REPO_ROOT = Path(__file__).resolve().parent.parent


def _roots(tmp_path: Path) -> p94.Roots:
    (tmp_path / "output").mkdir(exist_ok=True)
    return p94.Roots(repo_root=REPO_ROOT, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")


# ----------------------------------------------------------------------
# The barrier and the writes
# ----------------------------------------------------------------------


def test_the_barrier_admits_only_the_two_task_directories(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    assert p94.assert_target(roots.run_root / "x.json", roots) == (roots.run_root / "x.json").resolve()
    assert p94.assert_target(roots.runs_root / "a" / "b.json", roots).name == "b.json"
    for outside in (tmp_path / "output" / "p9_1" / "x.json", tmp_path / "output" / "p9_4_backup" / "x.json",
                    roots.run_root / ".." / "p4_7" / "x.json"):
        with pytest.raises(PermissionError, match="P9.4 writes nowhere else"):
            p94.assert_target(outside, roots)
    link = roots.run_root / "escape"
    roots.run_root.mkdir(parents=True)
    link.symlink_to(tmp_path)
    with pytest.raises(PermissionError, match="P9.4 writes nowhere else"):
        p94.assert_target(link / "x.json", roots)


def test_write_once_never_overwrites_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    path = roots.runs_root / "r.json"
    p94.write_once(path, {"a": 1}, roots)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(FileExistsError, match="File exists"):
        p94.write_once(path, {"a": 2}, roots)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(p.name for p in path.parent.iterdir()) == ["r.json"]


def test_a_declaration_is_rewritten_only_as_the_identical_bytes(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    path = roots.run_root / "declaration.json"
    p94.write_declaration(path, {"k": [1, 2]}, roots)
    before = path.read_bytes()
    p94.write_declaration(path, {"k": [1, 2]}, roots)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="not rewritten"):
        p94.write_declaration(path, {"k": [1, 3]}, roots)
    assert path.read_bytes() == before


# ----------------------------------------------------------------------
# Run records and their status, computed from disk
# ----------------------------------------------------------------------


def _fake_trained(roots: p94.Roots, run: p94.Run, *, steps: int = p94.GRADIENT_STEPS, weight: float = 0.0) -> None:
    """What train_run leaves on disk: the checkpoint and its write-once record."""
    checkpoint = p94.checkpoint_path(roots, run)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.full((2,), float(weight))}}, checkpoint)
    record = {"format_version": p94.RUN_FORMAT_VERSION, "run": run.name, "group": run.group, "tier": run.tier,
              "seed": int(run.seed), "declared_gradient_steps": int(steps), "gradient_steps": int(steps),
              "seconds": 1.5, "file_sha256": p94.file_sha256(checkpoint),
              "canonical_digest": grid.canonical_digest_of(checkpoint)}
    p94.write_once(p94.run_record_path(roots, run), record, roots)


def test_run_status_is_missing_done_or_a_refusal(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    run = p94.control_runs()[0]
    assert p94.run_status(roots, run) == "missing"
    _fake_trained(roots, run)
    assert p94.run_status(roots, run) == "done"
    with pytest.raises(ValueError, match="record says"):
        p94.run_status(roots, run, steps=20)
    torch.save({"model": {"w": torch.ones(2)}}, p94.checkpoint_path(roots, run))
    with pytest.raises(ValueError, match="digests are not the ones"):
        p94.run_status(roots, run)


def test_run_status_refuses_a_partial_and_a_half_written_run(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    run, other = p94.new_runs()[:2]
    partial = p94.checkpoint_path(roots, run).with_suffix(".pt.partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x")
    with pytest.raises(ValueError, match="interrupted run"):
        p94.run_status(roots, run)
    checkpoint = p94.checkpoint_path(roots, other)
    torch.save({"model": {"w": torch.zeros(2)}}, checkpoint)
    with pytest.raises(ValueError, match="half-written"):
        p94.run_status(roots, other)


def test_the_runs_are_the_control_five_then_the_new_five() -> None:
    names = [run.name for run in p94.training_runs()]
    assert names == [f"mix50_dt_seed{s}" for s in p94.SEEDS] + [f"mix50_fresh_dt_seed{s}" for s in p94.SEEDS]
    assert {run.group for run in p94.control_runs()} == {"control"}
    assert {run.group for run in p94.new_runs()} == {"hz1x1"}


# ----------------------------------------------------------------------
# The control: 5 of 5, or nothing else trains
# ----------------------------------------------------------------------


@pytest.fixture()
def stubbed_control(monkeypatch: pytest.MonkeyPatch) -> dict[int, bool]:
    """The comparison and the reference resolution replaced: seed -> whether it reproduces."""
    outcome = {seed: True for seed in p94.SEEDS}
    monkeypatch.setattr(p94, "reference_checkpoint", lambda roots, seed: {
        "path": f"/ref/mix50_dt_seed{seed}.pt", "file_sha256": "r" * 64, "checked_against": ["x"]})

    def compare(candidate: Any, reference: Any) -> dict[str, Any]:
        seed = int(str(candidate).rsplit("seed", 1)[1].split(".")[0])
        equal = outcome[seed]
        return {"keys_equal": True, "missing_in_candidate": [], "extra_in_candidate": [], "n_parameters": 37,
                "n_parameters_differing": 0 if equal else 3, "largest_abs_difference": 0.0 if equal else 2.5e-7,
                "largest_abs_difference_parameter": None if equal else "blocks.0.attn.w", "all_equal": equal}

    monkeypatch.setattr(p94, "control_comparison", compare)
    monkeypatch.setattr(p94, "library_versions", lambda: {"torch": "t"})
    return outcome


def test_the_control_passes_at_five_of_five_only(tmp_path: Path, stubbed_control: dict[int, bool]) -> None:
    roots = _roots(tmp_path)
    for run in p94.control_runs()[:4]:
        _fake_trained(roots, run)
    with pytest.raises(ValueError, match="five runs or none"):
        p94.control_record(roots)
    _fake_trained(roots, p94.control_runs()[4])
    assert p94.control_record(roots)["passed"] is True
    stubbed_control[303] = False
    record = p94.control_record(roots)
    assert (record["passed"], record["n_seeds_equal"]) == (False, 4)
    report = p94.control_report(record)
    assert "4 of 5" in report and "seed 303: all_equal False" in report and "2.5e-07" in report
    assert "library versions" in report


def test_assert_control_passed_refuses_absent_and_failed_records(tmp_path: Path, stubbed_control: dict[int, bool]) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(ValueError, match="has not run"):
        p94.assert_control_passed(roots)
    for run in p94.control_runs():
        _fake_trained(roots, run)
    stubbed_control[505] = False
    p94.ensure_control_record(roots)
    with pytest.raises(ValueError, match="did not pass"):
        p94.assert_control_passed(roots)


# ----------------------------------------------------------------------
# The session: refusals first, the control's verdict before every new-tier run, STOP between runs
# ----------------------------------------------------------------------


@pytest.fixture()
def session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stubbed_control: dict[int, bool]) -> SimpleNamespace:
    roots = _roots(tmp_path)
    trained: list[str] = []
    declaration = {"streams": [], "rtg_scale": 41000.0}
    monkeypatch.setattr(p94, "load_declaration", lambda roots: declaration)
    monkeypatch.setattr(p94, "assert_files_as_declared", lambda declaration: {"n_files": 200, "all_as_declared": True})
    monkeypatch.setattr(p94, "control_inputs", lambda roots: SimpleNamespace(tier="mix50"))
    monkeypatch.setattr(p94, "new_tier_inputs", lambda roots, declaration: SimpleNamespace(tier="mix50_fresh"))

    def fake_train_run(roots: p94.Roots, run: p94.Run, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        assert inputs.tier == run.tier
        trained.append(run.name)
        _fake_trained(roots, run, steps=kwargs.get("steps", p94.GRADIENT_STEPS))
        if run.name == "mix50_dt_seed202":
            p94.stop_file(roots).parent.mkdir(parents=True, exist_ok=True)
        return {"seconds": 1.5}

    monkeypatch.setattr(p94, "train_run", fake_train_run)
    return SimpleNamespace(roots=roots, trained=trained, outcome=stubbed_control)


def test_a_session_trains_the_control_then_the_new_tier_after_a_passed_record(session: SimpleNamespace) -> None:
    summary = p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    assert session.trained == [run.name for run in p94.training_runs()]
    assert summary["control"] == {"passed": True, "n_seeds_equal": 5}
    assert summary["missing_left"] == []
    assert json.loads(p94.control_record_path(session.roots).read_text(encoding="utf-8"))["passed"] is True


def test_a_failed_control_stops_the_session_before_any_new_tier_run(session: SimpleNamespace) -> None:
    session.outcome[404] = False
    with pytest.raises(p94.ControlFailed, match="4 of 5"):
        p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    assert session.trained == [run.name for run in p94.control_runs()]
    with pytest.raises(p94.ControlFailed, match="4 of 5"):
        p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    assert session.trained == [run.name for run in p94.control_runs()]


def test_no_new_tier_run_starts_without_a_passed_record_even_if_the_record_is_never_written(
    session: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard before every new-tier run stands on its own: with the record never written, the first new-tier run is
    refused and nothing of the new tier trains."""
    monkeypatch.setattr(p94, "ensure_control_record", lambda roots, **kwargs: None)
    with pytest.raises(ValueError, match="has not run"):
        p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    assert session.trained == [run.name for run in p94.control_runs()]


def test_the_stop_file_ends_a_session_between_runs_and_the_next_resumes_from_disk(session: SimpleNamespace) -> None:
    stop = p94.stop_file(session.roots)
    stop.parent.mkdir(parents=True, exist_ok=True)

    original = p94.train_run

    def stopping(roots: p94.Roots, run: p94.Run, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(roots, run, inputs, **kwargs)
        if run.name == "mix50_dt_seed202":
            stop.touch()
        return result

    p94.train_run = stopping  # type: ignore[assignment]
    try:
        summary = p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    finally:
        p94.train_run = original  # type: ignore[assignment]
    assert summary["ended_by"] == "the stop file"
    assert session.trained == ["mix50_dt_seed101", "mix50_dt_seed202"]
    with pytest.raises(ValueError, match="no session starts"):
        p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    stop.unlink()
    p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    assert session.trained == [run.name for run in p94.training_runs()]


def test_the_hours_bound_plans_the_first_run_always_and_stops_at_the_bound(session: SimpleNamespace) -> None:
    one = p94.train_plan(session.roots, hours=0.01)
    assert [run.name for run in one.runs] == ["mix50_dt_seed101"]
    two = p94.train_plan(session.roots, hours=(2 * p94.PLANNED_RUN_SECONDS + p94.PLANNED_INPUT_SECONDS) / 3600.0)
    assert [run.name for run in two.runs] == ["mix50_dt_seed101", "mix50_dt_seed202"]
    with pytest.raises(ValueError, match="positive number of hours"):
        p94.train_plan(session.roots, hours=0.0)


def test_a_session_refuses_before_any_run_when_a_declared_file_moved(
    session: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def moved(declaration: Any) -> Any:
        raise ValueError("1 declared file(s) no longer hash to their declared sha256")

    monkeypatch.setattr(p94, "assert_files_as_declared", moved)
    with pytest.raises(ValueError, match="no longer hash"):
        p94.train_session(session.roots, hours=6.0, device="cpu", enter_regime=False)
    assert session.trained == []


def test_amendment_a1s_rehash_compares_every_file_with_its_declared_sha256(tmp_path: Path) -> None:
    files = []
    for i in range(p94.STREAM_COUNT):
        path = tmp_path / f"ep{i}.npz"
        path.write_bytes(f"episode {i}".encode())
        files.append({"dataset_dir": str(tmp_path), "episode_file": path.name, "file_sha256": p94.file_sha256(path)})
    assert p94.assert_files_as_declared({"streams": files})["all_as_declared"] is True
    with pytest.raises(ValueError, match="no longer hash"):
        p94.assert_files_as_declared({"streams": files[:-1]})
    (tmp_path / "ep7.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="no longer hash"):
        p94.assert_files_as_declared({"streams": files})


def test_the_declaration_must_be_pinned_and_unmoved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = p94.Roots(repo_root=tmp_path, output_root=tmp_path, corpus_root=tmp_path, draws_root=tmp_path)
    monkeypatch.setattr(p94, "DECLARATION_SHA256", None)
    with pytest.raises(ValueError, match="not pinned yet"):
        p94.load_declaration(roots)
    (tmp_path / "docs" / "data").mkdir(parents=True)
    path = tmp_path / "docs" / "data" / p94.DECLARATION_NAME
    path.write_text(json.dumps({"format_version": p94.DECLARATION_FORMAT_VERSION, "tier": p94.TIER}), encoding="utf-8")
    monkeypatch.setattr(p94, "DECLARATION_SHA256", p94.file_sha256(path))
    assert p94.load_declaration(roots)["tier"] == "mix50_fresh"
    path.write_text(json.dumps({"format_version": p94.DECLARATION_FORMAT_VERSION, "tier": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="has moved"):
        p94.load_declaration(roots)


def test_inputs_are_refused_unless_they_are_the_declared_ones() -> None:
    selected = [SimpleNamespace(dataset_dir=f"/c/d{i}", episode_file=f"e{i}.npz", ix_id="intersection_1_1")
                for i in range(3)]
    rows = [{"dataset_dir": s.dataset_dir, "episode_file": s.episode_file, "ix_id": s.ix_id} for s in selected]
    inputs = SimpleNamespace(selected=selected, declared={"target_rtg": -5959.0, "rtg_scale": 41000.0}, digest="ab" * 32,
                             spec=SimpleNamespace(tier="mix50_fresh"))
    p94.assert_inputs_as_declared(inputs, expected_streams=rows, target_rtg=-5959.0, rtg_scale=41000.0,
                                  statistics_digest="ab" * 32)
    with pytest.raises(ValueError, match="declared order"):
        p94.assert_inputs_as_declared(inputs, expected_streams=list(reversed(rows)), target_rtg=-5959.0,
                                      rtg_scale=41000.0, statistics_digest=None)
    with pytest.raises(ValueError, match="target/scale"):
        p94.assert_inputs_as_declared(inputs, expected_streams=rows, target_rtg=-5959.0, rtg_scale=40223.0,
                                      statistics_digest=None)
    with pytest.raises(ValueError, match="statistics digest"):
        p94.assert_inputs_as_declared(inputs, expected_streams=rows, target_rtg=-5959.0, rtg_scale=41000.0,
                                      statistics_digest="cd" * 32)


def test_the_recipe_read_from_the_code_is_the_plans() -> None:
    record = p94.recipe_record()
    assert {key: record[key] for key in p94.PLAN_RECIPE} == dict(p94.PLAN_RECIPE)


# ----------------------------------------------------------------------
# The real corpus and checkpoints (skip without them)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not (candidate / "cf_hz1x1__random" / "manifest.json").is_file():
        pytest.skip(f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to the main tree's corpus")
    return candidate


def test_on_the_corpus_the_composition_is_method_tier_grids_training_inputs_for_mix50(corpus_root: Path) -> None:
    """The control trains through p94.training_inputs: for the registered mix50 it must BE the registered inputs."""
    mine = p94.training_inputs(grid.TIERS["mix50"], corpus_root)
    registered = grid.training_inputs(grid.TIERS["mix50"], corpus_root)
    assert [s.key for s in mine.selected] == [s.key for s in registered.selected]
    assert [s.key for s in mine.kept] == [s.key for s in registered.kept]
    assert (mine.declared, mine.scale, mine.digest, mine.provenance, mine.group, mine.scenario_id) == (
        registered.declared, registered.scale, registered.digest, registered.provenance, registered.group,
        registered.scenario_id)
    for batch in ("full_batch", "top_batch"):
        left, right = getattr(mine, batch), getattr(registered, batch)
        assert sorted(left) == sorted(right)
        for key in left:
            assert torch.equal(left[key], right[key]), f"{batch}[{key}]"


def _output_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    candidate = Path(env_value) if env_value else REPO_ROOT / "output"
    if not (candidate / "p4_7" / "checkpoints" / "mix50_dt_seed101.pt").is_file():
        pytest.skip(f"no P4.7 checkpoint under {candidate}: set RLTRAFFIC_OUTPUT_ROOT to the main tree's output/")
    return candidate


def test_on_the_output_root_p4_7s_references_resolve_at_their_pins_and_reproduce_themselves(tmp_path: Path) -> None:
    output = _output_root()
    roots = p94.Roots(repo_root=REPO_ROOT, output_root=output, corpus_root=tmp_path, draws_root=tmp_path)
    for seed in p94.SEEDS:
        reference = p94.reference_checkpoint(roots, seed)
        assert reference["path"].endswith(f"p4_7/checkpoints/mix50_dt_seed{seed}.pt")
        assert "p7_3a_zero_shot.json inputs.checkpoints" in reference["checked_against"]
    path = p94.reference_checkpoint(roots, 101)["path"]
    same = p94.control_comparison(path, path)
    other = p94.control_comparison(path, p94.reference_checkpoint(roots, 202)["path"])
    assert (same["all_equal"], same["n_parameters_differing"]) == (True, 0)
    assert other["all_equal"] is False and other["largest_abs_difference"] > 0.0
