"""P9.4's command line (``python -m offline.p9_4_mix50_fresh``): every stage's pre-token check refuses what plan section
7 forbids -- ``cityflow`` and ``sumo`` before the control's 5 of 5 and the new tier's five runs, any stage while STOP
exists -- and writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import offline.method_tier_grid as grid
import offline.p9_4_mix50_fresh as p94

REPO_ROOT = Path(__file__).resolve().parent.parent


def _argv(tmp_path: Path, *rest: str) -> list[str]:
    for name in ("output", "corpus", "draws"):
        (tmp_path / name).mkdir(exist_ok=True)
    return ["--repo-root", str(REPO_ROOT), "--output-root", str(tmp_path / "output"), "--corpus-root",
            str(tmp_path / "corpus"), "--draws-root", str(tmp_path / "draws"), *rest]


def _roots(tmp_path: Path) -> p94.Roots:
    _argv(tmp_path)
    return p94.Roots(repo_root=REPO_ROOT, output_root=tmp_path / "output", corpus_root=tmp_path / "corpus",
                     draws_root=tmp_path / "draws")


def _trained(roots: p94.Roots, run: p94.Run) -> None:
    path = p94.checkpoint_path(roots, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.zeros(1)}}, path)
    p94.write_once(p94.run_record_path(roots, run), {
        "format_version": p94.RUN_FORMAT_VERSION, "run": run.name, "group": run.group, "tier": run.tier,
        "seed": run.seed, "declared_gradient_steps": 40000, "gradient_steps": 40000,
        "file_sha256": p94.file_sha256(path), "canonical_digest": grid.canonical_digest_of(path)}, roots)


@pytest.fixture()
def quiet_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The checks that need the corpus, the draws or a GPU, replaced by recorders; the rest run for real."""
    called: list[str] = []
    monkeypatch.setattr(p94, "load_declaration", lambda roots: called.append("declaration") or {"streams": []})
    monkeypatch.setattr(p94, "assert_files_as_declared", lambda declaration: called.append("rehash") or {})
    monkeypatch.setattr(p94, "reference_checkpoint", lambda roots, seed: called.append(f"reference{seed}") or {})
    monkeypatch.setattr(p94, "assert_held_out_draws", lambda roots, *, backend: called.append(f"draws:{backend}"))
    monkeypatch.setattr(p94, "assert_cuda", lambda: called.append("cuda"))
    return called


def test_check_train_runs_every_refusal_and_writes_nothing(tmp_path: Path, quiet_checks: list[str]) -> None:
    assert p94.main(_argv(tmp_path, "check", "--stage", "train")) == 0
    assert quiet_checks == ["declaration", "rehash", *(f"reference{s}" for s in p94.SEEDS), "cuda"]
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == []


@pytest.mark.parametrize("stage", ["cityflow", "sumo"])
def test_cityflow_and_sumo_refuse_before_the_controls_five_of_five(tmp_path: Path, quiet_checks: list[str],
                                                                    stage: str) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(ValueError, match="control has not run"):
        p94.main(_argv(tmp_path, "check", "--stage", stage))
    p94.write_once(p94.control_record_path(roots), {"format_version": p94.CONTROL_FORMAT_VERSION,
                                                    "declared_gradient_steps": 40000, "passed": False,
                                                    "n_seeds_equal": 4, "seeds": [], "library_versions": {}}, roots)
    with pytest.raises(ValueError, match="did not pass"):
        p94.main(_argv(tmp_path, "check", "--stage", stage))


@pytest.mark.parametrize("stage", ["cityflow", "sumo"])
def test_cityflow_and_sumo_refuse_until_the_new_tiers_five_runs_are_done(tmp_path: Path, quiet_checks: list[str],
                                                                         stage: str) -> None:
    roots = _roots(tmp_path)
    # Since review D (2026-11-19) a passed record must re-derive from the five control checkpoints it compared.
    for run in p94.control_runs():
        _trained(roots, run)
    seeds = [{"seed": run.seed, "candidate": str(p94.checkpoint_path(roots, run)),
              "candidate_file_sha256": p94.file_sha256(p94.checkpoint_path(roots, run)), "all_equal": True,
              "keys_equal": True, "n_parameters": 1} for run in p94.control_runs()]
    p94.write_once(p94.control_record_path(roots), {"format_version": p94.CONTROL_FORMAT_VERSION,
                                                    "declared_gradient_steps": 40000, "passed": True,
                                                    "n_seeds_equal": 5, "seeds": seeds, "library_versions": {}}, roots)
    for run in p94.new_runs()[:4]:
        _trained(roots, run)
    with pytest.raises(ValueError, match="mix50_fresh_dt_seed505 is not trained"):
        p94.main(_argv(tmp_path, "check", "--stage", stage))
    _trained(roots, p94.new_runs()[4])
    assert p94.main(_argv(tmp_path, "check", "--stage", stage)) == 0
    assert f"draws:{'cityflow' if stage == 'cityflow' else 'sumo'}" in quiet_checks


@pytest.mark.parametrize("stage", ["train", "cityflow", "sumo"])
def test_no_stage_passes_its_check_while_the_stop_file_exists(tmp_path: Path, quiet_checks: list[str],
                                                              stage: str) -> None:
    roots = _roots(tmp_path)
    p94.stop_file(roots).parent.mkdir(parents=True)
    p94.stop_file(roots).touch()
    with pytest.raises(ValueError, match="no session starts"):
        p94.main(_argv(tmp_path, "check", "--stage", stage))
    assert p94.stop_file(roots).exists()


def test_an_unknown_stage_or_a_bad_worker_count_is_refused_by_the_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        p94.main(_argv(tmp_path, "check", "--stage", "evaluate"))
    with pytest.raises(SystemExit, match="2"):
        p94.main(_argv(tmp_path, "sumo", "--workers", "7", "--deadline-unix", "1"))


def test_the_status_is_computed_from_disk(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    roots = _roots(tmp_path)
    _trained(roots, p94.control_runs()[0])
    assert p94.main(_argv(tmp_path, "status")) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["training"]["mix50_dt_seed101"] == "done" and status["training"]["mix50_fresh_dt_seed101"] == "missing"
    assert status["control"] is None and status["sumo"]["new"]["missing"] == 1000
    assert status["cityflow"]["route_i_complete"] is False


def test_the_train_command_exits_3_with_the_report_when_the_control_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(roots: Any, **kwargs: Any) -> Any:
        raise p94.ControlFailed("CONTROL: 4 of 5 seeds reproduce P4.7's mix50 checkpoints")

    monkeypatch.setattr(p94, "train_session", failing)
    assert p94.main(_argv(tmp_path, "train", "--hours", "2")) == 3
    assert "CONTROL: 4 of 5" in capsys.readouterr().err


def test_the_smoke_root_must_be_outside_the_main_trees_output_and_any_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        p94.smoke_roots(Path("/home/filip/rltraffic/output/p9_4_smoke"), repo_root=REPO_ROOT)
    with pytest.raises(ValueError, match="outside"):
        p94.smoke_roots(REPO_ROOT / "scratch", repo_root=REPO_ROOT)
    roots = p94.smoke_roots(tmp_path / "smoke", repo_root=REPO_ROOT)
    assert roots.output_root == (tmp_path / "smoke" / "output").resolve()
