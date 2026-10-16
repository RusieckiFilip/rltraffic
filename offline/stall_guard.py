"""A no-progress watchdog for P9.1's evaluation workers (``BRIEF_46`` Amendment B, B.3; ``DEFERRED`` 104).

No format is written here and no number passes through it: the module only ends a worker that has stopped making
progress, so that the driver can start it again.

WHY
---
CityFlow's ``Engine::~Engine()`` can wait forever at a two-party barrier when its controller thread has already left
its loop (``docs/notes/P5.3c_HANG_2026-09-30.md``, about 2 in 1,200 episodes under load).  The wait is inside C++, in
``pthread_cond_wait``, entered from a Python object's deallocation, so the GIL is HELD: no Python thread can run, and a
Python-level timer would never fire.  ``faulthandler``'s watchdog is a C thread that needs no GIL: armed with
``exit=True`` it writes every thread's stack to a file and calls ``_exit(1)``.  The hang never changes a number -- it
comes after the episode, every write is atomic (``dt_gate.write_json_atomic``), the engine is seeded -- so a worker
that restarts and resumes from the files on disk reproduces exactly what it would have produced (B.3).

THE PROTOCOL
------------
* A worker (``corrective_round`` ``evaluate``, ``old-cells``, ``p3a-references`` with ``--stall-dump PATH``) arms a
  :class:`StallGuard` on PATH before it rolls anything, and keeps it armed until the process ends, so a hang while the
  interpreter destroys the last engine at exit is caught as well.
* Progress re-arms it: every ``CityFlowEnv.reset`` and ``CityFlowEnv.step`` (:func:`install` wraps both for the rest
  of the process; ``envs/cityflow_env.py`` is not edited, and the wrapped methods are called with their arguments
  unchanged).  Re-arming is throttled to once per :data:`REARM_SECONDS`, because each re-arm starts a new C thread.
* :data:`STALL_SECONDS` without progress: PATH receives faulthandler's dump, whose first characters are
  :data:`BANNER`, and the process exits 1.  A worker that fails any other way leaves PATH empty.
* The driver (``offline/campaigns/p9_1_corrective.sh``, ``restarting``) restarts a worker on exactly that exit -- code 1
  AND the banner at the head of a fresh dump file -- at most ``MAX_RESTARTS`` times, logging the time and the dumped
  stack; any other non-zero exit stops the stage.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

#: B.3's "no progress for 10 minutes".
STALL_SECONDS: float = 600.0
#: The shortest interval between two re-arms: a re-arm joins faulthandler's watchdog thread and starts a new one, so
#: re-arming on every step of every episode would start a thread per step.  A stall is therefore detected after
#: between ``STALL_SECONDS - REARM_SECONDS`` and ``STALL_SECONDS`` seconds without progress.
REARM_SECONDS: float = 10.0
#: The first characters faulthandler writes on a timeout (``"Timeout (0:10:00)!"``); the driver's only stall signal.
BANNER: str = "Timeout ("
#: The methods that mark progress on every env class :func:`install` wraps.
PROGRESS_METHODS: tuple[str, ...] = ("reset", "step")


class StallGuard:
    """``faulthandler.dump_traceback_later(seconds, exit=True)`` on *dump_path*, armed at construction and re-armed
    by :meth:`progress` at most once per *rearm_seconds*."""

    def __init__(self, dump_path: str | Path, *, seconds: float = STALL_SECONDS, rearm_seconds: float = REARM_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not float(seconds) > 0.0 or float(rearm_seconds) < 0.0 or float(rearm_seconds) >= float(seconds):
            raise ValueError(f"a stall window of {seconds} s re-armed at most every {rearm_seconds} s cannot work")
        self.seconds = float(seconds)
        self.rearm_seconds = float(rearm_seconds)
        self._clock = clock
        # Truncated now and kept open for the process: faulthandler writes to its descriptor from a C thread, and the
        # driver reads an EMPTY file as "no stall".
        self._file = open(dump_path, "w", encoding="utf-8")  # noqa: SIM115 - held for the guard's life
        self._armed_at = 0.0
        self.arm()

    def arm(self) -> None:
        """Start a fresh *seconds* window now."""
        import faulthandler

        faulthandler.dump_traceback_later(self.seconds, exit=True, file=self._file)
        self._armed_at = self._clock()

    def progress(self) -> None:
        """Progress was made: re-arm, unless the last arm is younger than *rearm_seconds*."""
        if self._clock() - self._armed_at >= self.rearm_seconds:
            self.arm()

    def cancel(self) -> None:
        """Disarm and close the dump file (a worker that finished; a test)."""
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        self._file.close()


def wrap_progress(guard: StallGuard, classes: Sequence[type]) -> Callable[[], None]:
    """Make every :data:`PROGRESS_METHODS` call on each of *classes* report progress to *guard* first, then run the
    method unchanged; returns the callable that restores each class exactly as it was (an inherited method is
    inherited again)."""
    saved: list[tuple[type, str, Any]] = []

    def wrapped(original: Callable[..., Any]) -> Callable[..., Any]:
        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            guard.progress()
            return original(self, *args, **kwargs)

        method.__name__ = getattr(original, "__name__", "method")
        method.__qualname__ = f"stall_guard.{getattr(original, '__qualname__', 'method')}"
        method.__doc__ = getattr(original, "__doc__", None)
        return method

    for cls in classes:
        for name in PROGRESS_METHODS:
            saved.append((cls, name, cls.__dict__.get(name, _INHERITED)))
            setattr(cls, name, wrapped(getattr(cls, name)))

    def restore() -> None:
        for cls, name, own in reversed(saved):
            if own is _INHERITED:
                delattr(cls, name)
            else:
                setattr(cls, name, own)

    return restore


#: Marks a method the class did not define itself, so :func:`wrap_progress`'s restore deletes the wrapper.
_INHERITED = object()


def install(dump_path: str | Path, *, seconds: float = STALL_SECONDS,
            rearm_seconds: float = REARM_SECONDS) -> StallGuard:
    """A worker's guard for the rest of its process: armed now, re-armed by every ``CityFlowEnv.reset`` and ``step``,
    never cancelled (the interpreter's exit destroys the last engines, which is the exposed path too)."""
    from envs.cityflow_env import CityFlowEnv

    guard = StallGuard(dump_path, seconds=seconds, rearm_seconds=rearm_seconds)
    wrap_progress(guard, [CityFlowEnv])
    return guard
