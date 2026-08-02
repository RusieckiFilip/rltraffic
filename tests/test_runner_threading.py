"""Every cell must run its torch ops single-threaded, in the worker process.

torch's default intra-op pool takes one thread per core, so N worker processes
ask for 16N threads on a 16-core box. Measured 2026-07-27 on a reduced
``p0_baselines`` (6 cells): unpinned ``--workers 3`` ran at 0.29x the speed of a
single worker and ``--workers 6`` never finished, while pinned ``--workers 6``
reached 5.80x. Even ``--workers 1`` gains 1.37x, because these MLPs are too small
to pay for a thread pool. Full measurement: ``docs/notes/P0.3_spawn_attempt.md``
section 5.

The property under test is therefore about a *child* process, and the only
honest way to check it is to read ``torch.get_num_threads()`` inside the child
and carry the number back to the parent. Asserting it in the parent's own
process would prove nothing about the workers.

Two hazards these tests are built around:

* **Vacuity.** A forked child inherits the parent's thread count, so a test run
  in an already-pinned parent would pass without the pin doing anything. After
  this change every ``run_matrix(workers=1)`` elsewhere in the suite pins the
  pytest process to 1, so this is a live risk, not a theoretical one. Each test
  therefore forces the parent to a known non-1 value first.
* **Order dependence.** ``_restore_torch_threads`` puts the parent's thread
  count back after every test here, so the file is safe under ``pytest-randomly``
  and does not leak its forcing into the rest of the suite.

This file emits four ``DeprecationWarning: This process (pid=...) is
multi-threaded, use of fork() may lead to deadlocks in the child`` -- one per
forked worker. That is expected and is **not** to be silenced: reading
``torch.get_num_threads()`` requires importing torch into the parent, and a
torch-loaded parent is exactly the fork configuration Python 3.12 warns about.
The production parent does not have it, because ``experiments/runner.py`` imports
torch only inside ``limit_torch_threads`` (measured 2026-08-02: with warnings
forced visible, a parent importing only ``experiments.runner`` forks 2 workers and
emits **0** such warnings; adding ``import torch`` to that same parent emits 2).
The tests therefore run in a strictly harsher configuration than the CLI does.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

import experiments.runner as runner
from experiments.config import load_config

# Reused rather than reimplemented: the fake env and config builder already
# exercise the full train -> eval -> aggregate path without a simulator.
from tests.test_experiments_runner_smoke import FakeEnv, _smoke_config

# A parent value that is neither the pinned target (1) nor the machine default
# (one thread per core), so an assertion of "== 1" can only pass if something
# actively lowered it.
PARENT_THREADS = 2


@pytest.fixture(autouse=True)
def _restore_torch_threads() -> Iterator[None]:
    """Undo this file's thread forcing so test order cannot change any outcome."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def _probe_pinned_threads(_: int) -> tuple[int, int]:
    """Pin, then report the count read back independently, plus the pid that read it.

    Runs in a worker process. ``limit_torch_threads`` returns the count it
    believes it set; this deliberately ignores that and re-reads torch directly,
    so the test cannot be satisfied by a helper that returns the right number
    without changing anything.
    """
    runner.limit_torch_threads()
    return os.getpid(), torch.get_num_threads()


def _recording_make_env(out_dir: Path, env_spec: Any) -> FakeEnv:
    """Stand in for ``runner.make_env`` and record the caller's thread count.

    One file per pid, appended to, so a cell whose later ``make_env`` calls ran
    unpinned would show up as a second line rather than overwrite the first.
    The filesystem is the return channel: this runs in a forked child, whose
    only shared state with the parent is ``tmp_path``.
    """
    (out_dir / f"threads_{os.getpid()}.txt").open("a", encoding="utf-8").write(
        f"{torch.get_num_threads()}\n"
    )
    return FakeEnv(max_steps=env_spec.settings["max_steps"])


def _read_observations(out_dir: Path) -> dict[int, list[int]]:
    """pid -> every thread count that pid recorded, parsed back from disk."""
    observations: dict[int, list[int]] = {}
    for path in sorted(out_dir.glob("threads_*.txt")):
        pid = int(path.stem.split("_")[1])
        observations[pid] = [
            int(line) for line in path.read_text(encoding="utf-8").split() if line
        ]
    return observations


def test_worker_process_runs_single_threaded() -> None:
    """The property, measured in the child and returned to the parent."""
    torch.set_num_threads(PARENT_THREADS)
    assert torch.get_num_threads() == PARENT_THREADS, "parent forcing did not take"

    # timeout: this test forks from a parent that has imported torch, which is the
    # configuration Python 3.12 warns about ("may lead to deadlocks in the child").
    # The warning is expected and must not be dodged -- the pool path is the thing
    # under test. Bounding the wait means a deadlock would surface as a failed test
    # rather than a suite that hangs forever. 60 s against a ~2.5 s file is far too
    # wide to flake. run_matrix (T2) cannot be bounded this way: it calls
    # future.result() with no timeout, and runner.py is frozen beyond this task.
    with ProcessPoolExecutor(max_workers=2) as pool:
        observed = list(pool.map(_probe_pinned_threads, range(2), timeout=60))

    assert len(observed) == 2
    for pid, threads in observed:
        assert pid != os.getpid(), "measured in the parent, so it proves nothing"
        assert threads == 1, f"worker {pid} ran with {threads} torch threads"


def test_pool_path_pins_every_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_matrix(workers=2)`` pins inside the workers, not just in the helper."""
    if multiprocessing.get_start_method() != "fork":
        pytest.skip(
            "the parent-side monkeypatch of runner.make_env only reaches the child "
            "under the fork start method"
        )

    out_dir = tmp_path / "observations"
    out_dir.mkdir()
    monkeypatch.setattr(
        runner, "backend_ready", lambda backend, paths, libsumo=False: (True, "")
    )
    monkeypatch.setattr(
        runner, "make_env", functools.partial(_recording_make_env, out_dir)
    )

    torch.set_num_threads(PARENT_THREADS)
    config = load_config(_smoke_config(tmp_path, seeds=[7, 8]))
    report = runner.run_matrix(config, workers=2, verbose=False)

    for cell in report["cells"]:
        assert cell["status"] == "ok", cell.get("reason")

    observations = _read_observations(out_dir)
    assert observations, "no worker recorded a thread count; the cells never ran"
    for pid, counts in observations.items():
        assert pid != os.getpid(), f"cell ran in the parent process {pid}, not a worker"
        assert counts, f"worker {pid} recorded no thread count"
        assert set(counts) == {1}, f"worker {pid} recorded thread counts {counts}"


def test_sequential_path_is_pinned_and_still_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``workers=1`` still produces a complete cell, and pins the process it runs in."""
    monkeypatch.setattr(
        runner, "backend_ready", lambda backend, paths, libsumo=False: (True, "")
    )
    monkeypatch.setattr(
        runner, "make_env", lambda env_spec: FakeEnv(max_steps=env_spec.settings["max_steps"])
    )

    torch.set_num_threads(PARENT_THREADS)
    config = load_config(_smoke_config(tmp_path))
    report = runner.run_matrix(config, workers=1, verbose=False)

    cell = report["cells"][0]
    assert cell["status"] == "ok", cell.get("reason")
    assert set(cell["policies"]) == {"dqn", "Random"}
    assert report["aggregated"]["fake"]["dqn"]["episode_reward"]["n"] == 1

    # The sequential path runs run_cell in this very process, so the pin is
    # observable here directly. This is the documented side effect, not a leak:
    # the 1.37x row of the benchmark is exactly this case.
    assert torch.get_num_threads() == 1


def test_thread_limit_is_a_named_constant() -> None:
    """The count is a documented constant, not a literal buried in the call site."""
    assert runner.CELL_TORCH_THREADS == 1
    assert runner.limit_torch_threads.__doc__, "the constant's justification must be documented"
