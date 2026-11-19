"""P9.4's driver after the 2026-11-19 pre-flight review (review D; ``docs/returns/P9.4.md``): every run-tree check FAILS
CLOSED -- a tree whose git state cannot be read is refused, never read as clean (D's blocking finding) -- an ignored
file is refused like an untracked one, the commit is the full sha the report later reads, a missing argument exits 2,
and nothing is overwritten by a redirection.  The behavioural checks run copies of the driver in scratch trees, each
copy's ``MAIN`` pointed at a scratch main tree whose interpreter is the one running these tests (the git checks come
after the interpreter check, and a runner has no main tree); none gets past the module check, so nothing is run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p9_4_mix50_fresh.sh"
SHA = "0123456789abcdef0123456789abcdef01234567"


def _run(script: Path, *argv: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "WORKERS" and not key.startswith("P94_")}
    return subprocess.run(["bash", str(script), *argv], capture_output=True, text=True, env=env, cwd=cwd)


def _copy_driver(root: Path) -> Path:
    """The driver copied into *root*'s ``offline/campaigns/``, its ``MAIN`` a scratch main tree beside *root* whose
    ``.venv/bin/python`` is this interpreter."""
    main = root.parent / f"{root.name}_main"
    (main / ".venv" / "bin").mkdir(parents=True)
    (main / ".venv" / "bin" / "python").symlink_to(sys.executable)
    script = root / "offline" / "campaigns" / DRIVER.name
    script.parent.mkdir(parents=True)
    text = DRIVER.read_text(encoding="utf-8")
    assert text.count("\nMAIN=/home/filip/rltraffic\n") == 1
    script.write_text(text.replace("\nMAIN=/home/filip/rltraffic\n", f"\nMAIN={main}\n"), encoding="utf-8")
    return script


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", "-c", "user.name=p94", "-c", "user.email=p94@example.invalid", "-c",
                             "core.hooksPath=/dev/null", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


@pytest.mark.parametrize("argv", [[], ["train"], ["train", SHA, "extra"]])
def test_a_missing_or_an_extra_argument_is_refused_with_exit_2(argv: list[str], tmp_path: Path) -> None:
    result = _run(DRIVER, *argv, cwd=tmp_path)
    assert result.returncode == 2 and "usage" in result.stderr


@pytest.mark.parametrize("commit", ["512c101", "task/p9.4-mix50-fresh", "HEAD", SHA.upper(), SHA + "0"])
def test_the_commit_must_be_the_full_40_character_sha(commit: str, tmp_path: Path) -> None:
    result = _run(DRIVER, "train", commit, cwd=tmp_path)
    assert result.returncode == 2 and "full 40-character sha" in result.stderr


def test_a_tree_whose_git_state_cannot_be_read_is_refused_not_read_as_clean(tmp_path: Path) -> None:
    script = _copy_driver(tmp_path / "not_a_repository")
    result = _run(script, "train", SHA, cwd=tmp_path)
    assert result.returncode == 2 and "git cannot read the state of" in result.stderr


@pytest.fixture()
def detached_tree(tmp_path: Path) -> tuple[Path, str]:
    """A scratch repository holding only the driver and a .gitignore, detached at its one commit."""
    tree = tmp_path / "tree"
    script = _copy_driver(tree)
    (tree / ".gitignore").write_text("*.so\n__pycache__/\n", encoding="utf-8")
    _git(tree, "init", "-q")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "a scratch commit")
    head = _git(tree, "rev-parse", "HEAD")
    _git(tree, "checkout", "-q", "--detach", head)
    return script, head


def test_a_clean_detached_tree_at_its_commit_passes_every_git_check(detached_tree: tuple[Path, str]) -> None:
    script, head = detached_tree
    result = _run(script, "train", head, cwd=script.parents[2])
    # Stopped after them, at the module check: the scratch tree holds no module ("does not import"), or, where the
    # repository is installed in editable mode (a CI runner), the module resolves elsewhere ("loads from").
    assert result.returncode == 2 and ("does not import" in result.stderr or "loads from" in result.stderr)
    for message in ("git cannot", "ignored files", "is not at", "on a branch"):
        assert message not in result.stderr


def test_an_ignored_file_in_the_run_tree_is_refused_like_an_untracked_one(detached_tree: tuple[Path, str]) -> None:
    script, head = detached_tree
    tree = script.parents[2]
    (tree / "offline" / "p9_4_evaluation.so").write_bytes(b"\x7fELF")  # CPython would import it before the .py
    result = _run(script, "train", head, cwd=tree)
    assert result.returncode == 2 and "ignored files" in result.stderr and "p9_4_evaluation.so" in result.stderr


def test_a_tree_at_another_commit_or_on_a_branch_is_refused(detached_tree: tuple[Path, str]) -> None:
    script, head = detached_tree
    tree = script.parents[2]
    result = _run(script, "train", "f" * 40, cwd=tree)
    assert result.returncode == 2 and "is not at" in result.stderr
    _git(tree, "checkout", "-q", "-b", "a-branch")
    result = _run(script, "train", head, cwd=tree)
    assert result.returncode == 2 and "on a branch" in result.stderr


def test_nothing_is_overwritten_by_a_redirection_and_each_status_call_has_its_own_file() -> None:
    code = [line for line in DRIVER.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    assert "set -o noclobber" in code and not any(">|" in line for line in code)
    text = "\n".join(code)
    assert "stage_status status_complete" in text and "stage_status status_end" in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text  # nothing is written into the run tree between two sessions
