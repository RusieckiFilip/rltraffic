"""P9.1's training path is the campaign's first training commit's, byte for byte (``BRIEF_46`` Amendment E, E.4-4).

Runs 1-3 of 255 were trained at ``cb61ee7`` (stage ``20261028T151149Z``); every later run is trained by the sessions'
commit.  They come from ONE training path iff what computes a run is unchanged.  Two methods, both against
``git show cb61ee7:<file>``:

* **Static, inside ``offline/corrective_round.py``:** the closure of ``train_run`` -- every module-level function, class
  and constant it reaches by name, transitively -- together with the inputs ``train`` hands it (``_hz1x1_inputs``,
  ``_grid_inputs``) and the run list and status it resumes from (``training_runs``, ``training_cells``, ``run_status``,
  ``recipe_fields``, ``checkpoint_path``, ``run_record_path``): the same set of names at both commits, each with the same
  source text.
* **Dynamic, every other repository file:** a real two-step training through every trainer branch of ``train_run`` (hz1x1:
  IQL, DT and BC through ``method_tier_grid``; the grid's matched BC and IQL through ``offline_baselines``), and the grid
  DT's own trainer (``tier_sweep.train_tier_dt``) on a synthetic joint stack, traced with ``sys.setprofile``: every
  repository file a traced call executes in, other than ``corrective_round.py``, is byte-identical at both commits.

Skipped where the commit is not in the repository (a shallow clone).
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

import offline.corrective_round as cr
import offline.tier_sweep as ts
from tests.test_corrective_round_train import _cell, _grid_fixture, _hz1x1_fixture, _two_steps

REPO = Path(__file__).resolve().parents[1]
FIRST_TRAINING_COMMIT = "cb61ee7"
STATIC_ROOTS = ("train_run", "_hz1x1_inputs", "_grid_inputs", "training_runs", "training_cells", "run_status",
                "recipe_fields", "checkpoint_path", "run_record_path")


def _available() -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{FIRST_TRAINING_COMMIT}^{{commit}}"],
                          capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _available(), reason=f"commit {FIRST_TRAINING_COMMIT} (the campaign's first "
                                                         "trainings) is not in this repository")


def _at_commit(relative: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{FIRST_TRAINING_COMMIT}:{relative}"], capture_output=True,
                          check=True).stdout


def _definitions(source: str) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def _closure(source: str, roots: tuple[str, ...]) -> dict[str, str]:
    """Every module-level definition *roots* reach by name, transitively, with its source text (decorators included)."""
    definitions = _definitions(source)
    lines = source.splitlines(keepends=True)
    out: dict[str, str] = {}
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in out or name not in definitions:
            continue
        node = definitions[name]
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        out[name] = "".join(lines[first - 1:node.end_lineno])
        stack += [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in definitions]
    return out


def test_train_run_and_every_definition_it_reaches_in_corrective_round_are_the_first_trainings() -> None:
    head = _closure((REPO / "offline" / "corrective_round.py").read_text(encoding="utf-8"), STATIC_ROOTS)
    first = _closure(_at_commit("offline/corrective_round.py").decode("utf-8"), STATIC_ROOTS)
    assert sorted(head) == sorted(first)
    assert [name for name in sorted(head) if head[name] != first[name]] == []
    assert {"train_run", "assert_target", "_write_once", "LABELS", "CLOCK_LABEL", "Roots", "Cell"} <= set(head)


def _executed(call: Callable[[], Any]) -> set[Path]:
    files: set[str] = set()

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            files.add(frame.f_code.co_filename)

    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(None)
    repo = REPO.resolve()
    out = set()
    for name in files:
        path = Path(name).resolve()
        if path.is_file() and path.is_relative_to(repo) and path.relative_to(repo).parts[0] != "tests":
            out.add(path)
    return out


def test_every_repository_file_a_real_training_executes_is_the_first_trainings(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from tests.test_tier_sweep import _tiny_joint_setup

    (tmp_path / "hz").mkdir()
    (tmp_path / "grid").mkdir()
    executed: set[Path] = set()
    hz_roots = _hz1x1_fixture(tmp_path / "hz", monkeypatch)
    _two_steps(monkeypatch, "hz1x1", "iql_t", "dt", "bc")
    iql = _cell("hz1x1", "maxpressure", "iql_t")
    hz_cells = [iql, dataclasses.replace(iql, arm="dt"), dataclasses.replace(iql, arm="bc")]
    executed |= _executed(lambda: cr.train(hz_roots, cells=hz_cells, device="cpu", seeds=(101,)))
    grid_roots = _grid_fixture(tmp_path / "grid", monkeypatch)
    _two_steps(monkeypatch, "grid4x4", "bc_m", "iql_m")
    grid_cells = [_cell("grid4x4", "random", "bc_m"), _cell("grid4x4", "random", "iql_m")]
    executed |= _executed(lambda: cr.train(grid_roots, cells=grid_cells, device="cpu", seeds=(101,)))
    assert all(cr.run_status(hz_roots, c, 101) == "done" for c in hz_cells)
    assert all(cr.run_status(grid_roots, c, 101) == "done" for c in grid_cells)
    setup = _tiny_joint_setup()
    executed |= _executed(lambda: ts.train_tier_dt(
        setup["stacked"], setup["index"], tier="random_fresh", method="dt_nomix", seed=101,
        adjacency=setup["adjacency"], prompts=setup["prompts"], stats=setup["stats"], state_dim=setup["state_dim"],
        n_actions=setup["n_actions"], gradient_steps=2, batch_size=2, device=torch.device("cpu"),
        checkpoint_path=tmp_path / "dt.pt", protected=(), provenance={}))
    names = {p.relative_to(REPO.resolve()).as_posix() for p in executed}
    assert {"offline/method_tier_grid.py", "offline/offline_baselines.py", "offline/dt_gate.py", "offline/tier_sweep.py",
            "offline/a27_arms.py", "agent/OfflineBaselines.py", "agent/SpatialDTAgent.py"} <= names, sorted(names)
    changed = sorted(name for name in names if name != "offline/corrective_round.py"
                     and _at_commit(name) != (REPO / name).read_bytes())
    assert changed == []
