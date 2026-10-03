"""P8.2: the policy's decision latency, timed alone in real CityFlow episodes -- and no outcome whatsoever.

Written against ``docs/briefs/BRIEF_43_p8.2_compute_latency.md`` §4 and its **Amendment A** (A2's rulings Q8–Q14, A3's
two added requirements), on the plan ``docs/plans/p8.2.md`` approved at ``a7c43e1``.

On-disk formats
---------------
* ``p8.2-latency/1.0`` -- one JSON record per (row, device), ``<run dir>/<row>_<device>.json``, written ONCE (atomic,
  exclusive). It carries the per-decision nanoseconds of three episodes, the statistics over the timed decisions,
  the regime, the machine and the checkpoint's digest.
* ``p8.2-canary/1.0`` -- ``canary_open.json`` / ``canary_close.json``: the machine-health canary's seconds, its
  verdict and whether the engine reproduced draw 0 -- as a boolean, never the observed values.
* ``p8.2-latency-run/1.0`` -- ``run.json``: the row order, every attempt of every supervised process and the run's
  terminal status, beside the markers ``COMPLETE`` / ``FAILED``.
* ``p8.2-preflight/1.0`` -- the G1 pre-flight's record: the measured process wall times, the per-scenario timeouts
  and the run's expected duration derived from them.

Alignment convention: not applicable -- no trajectory is recorded. The episode loop mirrors
``offline.horizon_metric.horizon_rollout``'s order (``info = env.reset(seed=1000)``; then ``max_steps`` times: decide,
step) with every accumulation removed: the reward is received and discarded, no metric is read, and no ``info``
field reaches a record.

What is timed
-------------
Only the decision call -- the function the row's OWN evaluation factory returns, imported and never reimplemented.
The env's ``step`` is outside the timer. On CUDA, ``torch.cuda.synchronize()`` runs before BOTH clock readings.
The first ``WARMUP`` decisions of each episode are recorded and excluded from the statistics by count.

Teardown and CityFlow's destructor race (Amendment A3.1, ``DEFERRED`` 104)
-------------------------------------------------------------------------
``CityFlowEnv.reset`` reuses its engine; only ``close`` (or garbage collection) destroys it, and that destructor can
hang forever. A row process therefore keeps all three episode envs OPEN until its record is written, writes the
record atomically and exclusively, and only then closes them. The driver supervises each process under a timeout:
an attempt that hangs before its record exists is killed and re-run (at most ``MAX_ATTEMPTS``); a record written by
a process that then hangs is kept; a process that EXITS without a record fails the run at once.

What this module does NOT do
----------------------------
It trains nothing, evaluates nothing and records no ATT, reward, return, queue or any other episode quantity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "FORMAT_VERSION",
    "CANARY_FORMAT_VERSION",
    "RUN_FORMAT_VERSION",
    "PREFLIGHT_FORMAT_VERSION",
    "TIMING_DRAWS",
    "ENGINE_SEED",
    "WARMUP",
    "DECISIONS_PER_EPISODE",
    "MIN_TIMED",
    "MAX_ATTEMPTS",
    "FORBIDDEN_KEY_TOKENS",
    "ScenarioSpec",
    "SCENARIOS",
    "LatencyRow",
    "ROWS",
    "row_by_id",
    "time_episode",
    "latency_stats",
    "find_outcome_keys",
    "assert_cpu_regime",
    "configure_regime",
    "write_once",
    "build_record",
    "build_canary_record",
    "run_row",
    "supervise",
    "run_jobs",
    "build_run_record",
    "write_manifest",
    "run_all",
    "derive_timeouts",
    "expected_duration",
    "build_parser",
    "main",
]

FORMAT_VERSION = "p8.2-latency/1.0"
CANARY_FORMAT_VERSION = "p8.2-canary/1.0"
RUN_FORMAT_VERSION = "p8.2-latency-run/1.0"
PREFLIGHT_FORMAT_VERSION = "p8.2-preflight/1.0"

#: Q8: three held-out draws, one episode each. No outcome is recorded (T-no-outcome), and the held-out draws are the
#: deployment distribution.
TIMING_DRAWS: tuple[int, ...] = (1000, 1001, 1002)

#: The evaluations' engine seed: ``horizon_rollout`` resets with ``seed + ep`` and ep is 0 for one episode per env.
ENGINE_SEED = 1000

#: §5.2: decisions 0..19 of every episode are warm-up (the K = 20 window is full from index 19).
WARMUP = 20

#: ``envs/base_traffic_env.py:604-605``: never terminated, truncated at ``max_steps`` = 360.
DECISIONS_PER_EPISODE = 360

#: The brief: the p95 must rest on at least 1,000 timed decisions; 3 x (360 - 20) = 1,020.
MIN_TIMED = 1000

#: Amendment A3.1: at most three attempts per supervised process.
MAX_ATTEMPTS = 3

#: T-no-outcome: a record key whose ``_``-separated tokens include one of these names an episode quantity.
FORBIDDEN_KEY_TOKENS: tuple[str, ...] = (
    "att",
    "reward",
    "rewards",
    "return",
    "returns",
    "travel",
    "queue",
    "queues",
    "waiting",
    "vehicle",
    "vehicles",
    "throughput",
    "delay",
    "completed",
    "entered",
    "metric",
    "metrics",
    "rho",
    "pressure",
)


@dataclass(frozen=True)
class ScenarioSpec:
    """One CityFlow scenario of the table: the draws' scenario key, the env id and the intersection count."""

    name: str
    scenario_key: str
    scenario_id: str
    n_intersections: int


SCENARIOS: Mapping[str, ScenarioSpec] = {}


@dataclass(frozen=True)
class LatencyRow:
    """One row of the latency registry: what is timed, through which factory, with which prompt and settings.

    ``checkpoint`` is relative to the MAIN tree (the output root's parent); ``sha256`` is the representative
    checkpoint's digest (seed 101 of the row's headline tier, ``docs/plans/p8.2.md`` Appendix A) and
    ``sha256_source`` names the record that holds it as ``(kind, file, locator)``: ``("json", <file under the repo
    root>, <JSON path>)``, ``("sums", <manifest under the output root>, <listed relative path>)`` or
    ``("corpus_manifest", <manifest under the corpus root>, <JSON path>)``. ``settings_corpus_dir`` is the collection
    whose manifest gives the row's evaluation env settings (``offline.dt_gate.env_settings_from_manifest``).
    """

    row_id: str
    scenario: str
    kind: str
    checkpoint: str | None
    sha256: str | None
    sha256_source: tuple[str, str, str] | None
    method: str | None
    declared_gradient_steps: int | None
    target_rtg: float | None
    target_source: str | None
    fine_tune_k: int | None
    settings_corpus_dir: str
    devices: tuple[str, ...]


#: The 42 latency rows of ``docs/plans/p8.2.md`` §3 (the 36 checkpoint rows of Appendix A and six heuristics).
ROWS: tuple[LatencyRow, ...] = ()


def row_by_id(row_id: str) -> LatencyRow:
    """The registry row named *row_id*; refuses an unknown id."""
    raise NotImplementedError


def time_episode(
    env: Any,
    choose: Callable[[Any, dict[str, Any]], Any],
    *,
    engine_seed: int = ENGINE_SEED,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], int] | None = None,
) -> list[int]:
    """Roll one episode and return the nanoseconds of every decision call, in order.

    Only ``choose(env, info)`` sits between the two clock readings; ``env.step`` is outside. ``sync`` (CUDA's
    ``torch.cuda.synchronize``) runs immediately before BOTH readings when given.
    """
    raise NotImplementedError


def latency_stats(per_episode_ns: Sequence[Sequence[int]], *, warmup: int = WARMUP) -> dict[str, Any]:
    """Statistics over the timed decisions: the first *warmup* of EACH episode are excluded by count.

    ``median_ns`` is ``numpy.median``; ``p95_ns`` is the nearest-rank value (``numpy.percentile(..., method=
    "inverted_cdf")``, an observed value). Milliseconds are the same values divided by 10**6.
    """
    raise NotImplementedError


def find_outcome_keys(payload: Any) -> list[str]:
    """Every key, at any depth, whose ``_``-separated tokens include one of :data:`FORBIDDEN_KEY_TOKENS`."""
    raise NotImplementedError


def assert_cpu_regime(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Refuse unless torch runs ONE thread, ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are ``"1"`` and
    ``CUBLAS_WORKSPACE_CONFIG`` is unset; return the regime block a record carries."""
    raise NotImplementedError


def configure_regime(device: str) -> dict[str, Any]:
    """Pin torch to one thread, then :func:`assert_cpu_regime` -- for the CPU and the CUDA rows alike."""
    raise NotImplementedError


def write_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Write *payload* as JSON to *path* atomically and exclusively.

    The bytes go to a temporary file under ``<parent>/partial/`` first and are then hard-linked to *path*; an
    existing *path* refuses and is left untouched, and a failure before the link leaves no file at *path*.
    """
    raise NotImplementedError


def build_record(
    row: LatencyRow,
    device: str,
    per_episode_ns: Sequence[Sequence[int]],
    *,
    draws: Sequence[int],
    prompt: Mapping[str, Any] | None,
    factory: str,
    regime: Mapping[str, Any],
    machine: Mapping[str, Any],
    load_before: Sequence[float],
    load_after: Sequence[float],
    git: Mapping[str, Any],
    warmup: int = WARMUP,
) -> dict[str, Any]:
    """The ``p8.2-latency/1.0`` record; refuses an episode whose decision count is not 360, fewer than
    :data:`MIN_TIMED` timed decisions, or any key that names an episode quantity."""
    raise NotImplementedError


def build_canary_record(phase: str, seconds: float, *, reproduced: bool, git: Mapping[str, Any]) -> dict[str, Any]:
    """The ``p8.2-canary/1.0`` record: seconds, the 2.0 s threshold, the verdict and ``reproduced`` -- no value of
    the canary episode itself."""
    raise NotImplementedError


def run_row(
    row: LatencyRow,
    device: str,
    *,
    out_dir: Path,
    output_root: Path,
    corpus_root: Path,
    draws_root: Path,
    data_dir: Path,
    env_builder: Callable[[int], Any] | None = None,
    factory_builder: Callable[[], Callable[[Any], Callable[[Any, dict[str, Any]], Any]]] | None = None,
    writer: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> Path:
    """The row process: time three episodes and write the record BEFORE any env is closed.

    ``env_builder``, ``factory_builder`` and ``writer`` exist for tests; the defaults build the real CityFlow env of
    each draw, the row's own evaluation factory and :func:`write_once`.
    """
    raise NotImplementedError


def supervise(
    label: str,
    command: Sequence[str],
    *,
    record_path: Path,
    timeout_s: float,
    max_attempts: int = MAX_ATTEMPTS,
    env: Mapping[str, str] | None = None,
    log_dir: Path | None = None,
    kill_grace_s: float = 5.0,
) -> dict[str, Any]:
    """Run *command* under *timeout_s*, at most *max_attempts* times (Amendment A3.1).

    A timeout kills the process group and is recorded as ``hung``. A valid record at *record_path* ends the
    supervision (status ``ok``, or ``kept`` when the process hung or failed AFTER writing it). A process that exits
    without a record fails at once; a process that hangs without one is re-run, and after the last attempt the
    outcome is ``failed`` with a reason naming *label*.
    """
    raise NotImplementedError


def run_jobs(
    jobs: Sequence[tuple[str, Sequence[str], Path, float]],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    env: Mapping[str, str] | None = None,
    log_dir: Path | None = None,
    kill_grace_s: float = 5.0,
) -> dict[str, Any]:
    """Supervise *jobs* (label, command, record path, timeout) one after another; the first failure stops the
    sequence and the result names it."""
    raise NotImplementedError


def build_run_record(
    stamp: str,
    rows: Sequence[LatencyRow],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    status: str,
    reason: str | None,
    git: Mapping[str, Any],
    timeouts: Mapping[str, float],
) -> dict[str, Any]:
    """The ``p8.2-latency-run/1.0`` record: the declared row order, every supervised outcome with its attempts, the
    timeouts, the terminal status and, on ``FAILED``, the reason naming what failed."""
    raise NotImplementedError


def write_manifest(latency_dir: Path, manifest_path: Path) -> list[str]:
    """Write ``sha256  relative/path`` for every file under *latency_dir* (sorted), once, then re-verify every line
    against the files; returns the lines. An existing manifest refuses."""
    raise NotImplementedError


def run_all(
    *,
    stamp: str,
    rows: Sequence[LatencyRow],
    latency_root: Path,
    manifest_path: Path,
    command_for: Callable[[LatencyRow, str, Path], Sequence[str]],
    canary_command_for: Callable[[str, Path], Sequence[str]],
    timeouts: Mapping[str, float],
    canary_timeout_s: float,
    git: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    kill_grace_s: float = 5.0,
) -> int:
    """The timing run: the opening canary, every (row, device) process in order, the closing canary, ``run.json``,
    ``COMPLETE`` or ``FAILED``, and on ``COMPLETE`` the manifest. Refuses a stamp that exists or a manifest that
    exists before creating anything. Returns 0 on ``COMPLETE``, 1 otherwise."""
    raise NotImplementedError


def derive_timeouts(measured_seconds: Mapping[tuple[str, str], float], *, factor: float = 3.0,
                    floor_s: float = 120.0) -> dict[str, float]:
    """Per-scenario timeout from the pre-flight: ``max(floor_s, factor x`` the slowest measured process of that
    scenario``)``, rounded up to a whole second."""
    raise NotImplementedError


def expected_duration(measured_seconds: Mapping[tuple[str, str], float], canary_seconds: float,
                      rows: Sequence[LatencyRow] | None = None) -> dict[str, Any]:
    """An UPPER-BOUND estimate of the run: every (row, device) process costed at its scenario's measured process
    on that device (the pre-flight measures the most expensive row of each scenario), plus two canaries."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    """The CLI: ``run-row``, ``canary``, ``run-all``, ``preflight``."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns the process exit code."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
