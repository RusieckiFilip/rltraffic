"""P7.3d gate G2: the one-cell measurement — TIME and MEMORY, and no outcome whatsoever.

Written against ``docs/briefs/BRIEF_39_p7.3d_grid4x4_zero_shot.md`` §3 C5 and **Amendment B.2**,
which moved this measurement onto **draw 5** and fenced it.

WHY THIS EXISTS
---------------
The costing note (`docs/notes/GRID4X4_ZERO_SHOT_COSTING_2026-09-17.md`) gives grid4x4's SUMO cost
as a RANGE over an unmeasured multiplier, and says so: *"the first act of any grid4x4 work is a
one-cell measurement, which replaces every range above."*  ``WORKERS`` and the campaign driver's
RSS refusal budget are set from what this module measures, and the driver's header cites it with
its date and its canary.

⛔ **THIS MODULE PRODUCES NO OUTCOME, BY CONSTRUCTION (Amendment B.2-2).**
It measures wall seconds, in-process seconds, peak RSS and GPU memory.  It does **not** compute,
print, return or store an average travel time, an ``e_sumo``, an episode return, a return-to-go
series or an action — from any cell, on any draw.  ``reconstruct_sumo_episode`` IS called, because
the campaign's cells call it and a rate that skipped it would be a rate for a different cell, and
its result is discarded unread except for the structural counters below.

**What a G2 record may carry, and why each is not an outcome:** the decision count (360) and the
observation count (3,600) are properties of the horizon; ``n_teleports``, the effective vehicle
type set and the ``time-to-teleport`` option are the campaign's own refusal checks — a cell that
did not run under parity must be visible as such; every other field is a duration or a byte count.

**The one quantity in the capture that looks like an outcome is the CANARY's**, and it is not this
scenario's: ``format_canary_line`` prints hangzhou draw 0's ``local_return`` and ``att_horizon``,
which are the committed constants ``CANARY_REFERENCE_LOCAL_RETURN`` and
``CANARY_REFERENCE_ATT_HORIZON`` in :mod:`offline.transfer_calibration`.  The canary's correctness
half exists precisely to compare them (``BRIEF_36`` E1.2), they are already in
``docs/data/p4_3_probe.json``, and they say nothing about grid4x4.

WHERE IT RUNS, AND ON WHAT
--------------------------
**Draw 5** — P7.2b's smoke draw, outside the held-out pool 1000–1099 and outside the probe band
201–300 (Amendment B.2-2(1)), exactly as P7.3a's and P7.3b's pilots ran.  grid4x4's draw 5 has no
``parity/`` (C1 rendered 201–300 and 1000–1099), so :func:`ensure_draw_parity` renders it with the
same tool and records the provenance it wrote.

⚠️ **The halting cross-check is OFF here**, because draw 5 is not Amendment C2's declared subset.
The campaign runs it ON for the 7 cells of draw 1000 only, and T-obs measured that it roughly
doubles an episode (108 s against 53 s, no canary).  A schedule built from these numbers must add
that cost for those 7 cells and not for the other 693.

On-disk format
--------------
``p7.3d-g2/1.0``.  One JSON record per run under ``<work_dir>/g2/``, whose entire measurement body
sits under the key ``fenced_do_not_report`` — the key
:func:`offline.transfer_curve.report` refuses to emit (``FENCED_KEY``) — in a directory no
``report`` ever globs.  Alignment convention: not applicable; this file records no trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "G2_FORMAT_VERSION",
    "FENCED_KEY",
    "GRID4X4_KEY",
    "G2_DRAW",
    "DT_SEED",
    "POOL_SIZES",
    "PeakMemory",
    "MemorySampler",
    "ensure_draw_parity",
    "measure_single_cell",
    "measure_pool",
    "build_parser",
    "main",
]

G2_FORMAT_VERSION = "p7.3d-g2/1.0"

#: The key every measurement body sits under.  Named identically to
#: ``offline.transfer_calibration.FENCED_KEY`` / ``offline.transfer_curve.FENCED_KEY``, whose
#: ``report`` raises if it reaches an artifact.
FENCED_KEY = "fenced_do_not_report"

GRID4X4_KEY = "cityflow_grid4x4"

#: Amendment B.2-2(1): the smoke draw, outside the held-out pool and outside the probe band.
G2_DRAW = 5

#: A20(a)'s first seed.  One seed is a timing cell; five would be five timings of one number.
DT_SEED = 101

#: The pool sizes the RSS scaling is measured at (§3 C5).
POOL_SIZES: tuple[int, ...] = (4, 8, 12)

_SAMPLE_SECONDS = 0.25


def _rss_kib(pid: int) -> int:
    """This process's resident set in KiB, or 0 if it has gone."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _hwm_kib(pid: int) -> int:
    """This process's PEAK resident set in KiB (``VmHWM``), or 0 if it has gone."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _descendants(pid: int) -> list[int]:
    """Every live descendant of *pid*, from ``/proc`` alone -- no new dependency."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        # The command name can contain spaces and parentheses, so ppid is read after the last ')'.
        tail = stat.rsplit(")", 1)[-1].split()
        if len(tail) < 2:
            continue
        try:
            children.setdefault(int(tail[1]), []).append(int(entry.name))
        except ValueError:
            continue
    out: list[int] = []
    frontier = [int(pid)]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, []):
            out.append(child)
            frontier.append(child)
    return out


def _nvidia_used_mib() -> int | None:
    """Total GPU memory in use, from ``nvidia-smi``; ``None`` when there is no GPU to ask."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


@dataclass
class PeakMemory:
    """What a sampler observed: peaks in KiB (host) and MiB (device), and its own sample count."""

    peak_tree_rss_kib: int = 0
    peak_self_rss_kib: int = 0
    peak_gpu_used_mib: int | None = None
    n_samples: int = 0
    max_live_processes: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "peak_tree_rss_mib": round(self.peak_tree_rss_kib / 1024, 1),
            "peak_self_rss_mib": round(self.peak_self_rss_kib / 1024, 1),
            "peak_gpu_used_mib": self.peak_gpu_used_mib,
            "n_samples": self.n_samples,
            "max_live_processes": self.max_live_processes,
            "sample_interval_seconds": _SAMPLE_SECONDS,
        }


class MemorySampler:
    """Samples the whole process tree's resident set while a measurement runs.

    A worker pool's cost is the SUM over its live processes, not the maximum of one: twelve
    workers that each fit are a host refusal if their total does not.  ``resource.getrusage``
    cannot answer that (``RUSAGE_CHILDREN``'s ``ru_maxrss`` is the maximum over children, never the
    sum), so the tree is walked in ``/proc`` on a timer, which is also what makes the peak a peak
    rather than an end-of-run reading.
    """

    def __init__(self, *, interval: float = _SAMPLE_SECONDS, sample_gpu: bool = True) -> None:
        self._interval = float(interval)
        self._sample_gpu = bool(sample_gpu)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = PeakMemory()

    def _sample_once(self) -> None:
        pid = os.getpid()
        pids = [pid, *_descendants(pid)]
        total = sum(_rss_kib(one) for one in pids)
        self.peak.peak_tree_rss_kib = max(self.peak.peak_tree_rss_kib, total)
        self.peak.peak_self_rss_kib = max(self.peak.peak_self_rss_kib, _rss_kib(pid))
        self.peak.max_live_processes = max(self.peak.max_live_processes, len(pids))
        self.peak.n_samples += 1
        if self._sample_gpu:
            used = _nvidia_used_mib()
            if used is not None:
                current = self.peak.peak_gpu_used_mib
                self.peak.peak_gpu_used_mib = used if current is None else max(current, used)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample_once()

    def __enter__(self) -> "MemorySampler":
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="g2-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample_once()


# ----------------------------------------------------------------------
# The cells.  Each returns TIMINGS AND MEMORY and nothing else.
# ----------------------------------------------------------------------


def ensure_draw_parity(
    draw_id: int = G2_DRAW,
    *,
    scenario_key: str = GRID4X4_KEY,
    out_root: str | Path,
    env_config: str | Path,
) -> dict[str, Any]:
    """Render one draw's ``parity/`` with C1's own tool, and return what its provenance records.

    Amendment B.2-2(1): draw 5 has no parity directory because C1 rendered 201-300 and 1000-1099.
    ``materialise_parity`` is idempotent -- an existing identical directory is ``kept`` -- and it
    never writes to the parent, so running this twice is a no-op and running it at all adds one
    directory under the gitignored draws tree.
    """
    from offline.materialise_draws import (
        load_parity_provenance,
        materialise_parity,
        parity_sumocfg_path,
    )

    (record,) = materialise_parity(env_config, [int(draw_id)], out_root=out_root)
    provenance = load_parity_provenance(scenario_key, int(draw_id), out_root=out_root)
    return {
        "draw_id": int(draw_id),
        "action": record.action,
        "parent_action": record.parent_action,
        "config_path": str(parity_sumocfg_path(scenario_key, int(draw_id), out_root=out_root)),
        "format_version": provenance["format_version"],
        "n_vehicles": provenance["n_vehicles"],
        "n_bound": provenance["n_bound"],
        "vtype_attributes": provenance["vtype_attributes"],
        "net_sha256": provenance["net"]["sha256"],
        "route_template": provenance.get("route_template"),
        "files": provenance["files"],
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
    }


def _checkpoint_path(seed: int, *, output_root: str | Path) -> Path:
    from offline.transfer_calibration import (
        GRID4X4_CHECKPOINT_STEM,
        GRID4X4_CHECKPOINT_SUBDIR,
    )

    return Path(output_root) / GRID4X4_CHECKPOINT_SUBDIR / f"{GRID4X4_CHECKPOINT_STEM}{seed}.pt"


def measure_single_cell(
    arm: str,
    *,
    draw_id: int = G2_DRAW,
    scenario_key: str = GRID4X4_KEY,
    draws_root: str | Path,
    output_root: str | Path,
    seed: int = DT_SEED,
    sample_gpu: bool = True,
) -> dict[str, Any]:
    """One grid4x4 SUMO cell in THIS process: timings, memory, and the structural counters.

    *arm* is ``dt`` -- the registered subject at *seed*, under **the checkpoint's own in-domain
    prompt**, which Amendment B.2-1 records as the only prompt that exists before the SUMO probe
    runs -- or ``fixedtime``, one of rho's two anchors.

    The DT cell runs on the ALIGNED observer env and the anchor on the observed but UNWRAPPED one,
    which is ``BRIEF_37`` Amendment A2's rule and the shape ``transfer_curve.env_for_cell`` uses;
    the halting cross-check is OFF (draw 5 is not Amendment C2's subset).

    ⛔ Every number this returns is a duration, a byte count or a structural counter.  The
    episode's outcome is computed -- the campaign's cells compute it, so a rate without it would
    be a rate for a different cell -- and discarded unread.
    """
    import torch

    from offline.aligned_env import aligned_observer_env_for_draw, observer_env_for_draw
    from offline.horizon_metric import horizon_rollout
    from offline.materialise_draws import parity_sumocfg_path
    from offline.sumo_att_reference import (
        build_policy,
        collect_style_args,
        reconstruct_sumo_episode,
    )

    if arm not in {"dt", "fixedtime"}:
        raise ValueError(f"unknown G2 arm {arm!r}; the two measured cells are 'dt' and 'fixedtime'")
    config_path = parity_sumocfg_path(scenario_key, int(draw_id), out_root=draws_root)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"draw {draw_id} has no parity configuration at {config_path}; G2 renders it first "
            "(ensure_draw_parity), because C1 rendered 201-300 and 1000-1099 only"
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    with MemorySampler(sample_gpu=sample_gpu) as sampler:
        if arm == "dt":
            from agent.SpatialDTAgent import SpatialDTAgent

            env = aligned_observer_env_for_draw(
                scenario_key, int(draw_id), out_root=draws_root,
                halting_check=False, arm="maxpressure",
            )
        else:
            env = observer_env_for_draw(
                scenario_key, int(draw_id), out_root=draws_root,
                halting_check=False, arm=arm,
            )
        try:
            build_started = time.perf_counter()
            if arm == "dt":
                checkpoint = _checkpoint_path(seed, output_root=output_root)
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"no checkpoint at {checkpoint}")
                agent = SpatialDTAgent.from_checkpoint(env, str(checkpoint))

                def choose(_env: Any, info: Mapping[str, Any]) -> Any:
                    return agent.act(dict(info), explore=False, update_memory=True)
            else:
                args = collect_style_args(
                    "sumo", arm, config_path, episodes=1, base_seed=1000,
                    sentinel_out_dir="/nonexistent",
                )
                policy = build_policy(env, args)

                def choose(_env: Any, info: Mapping[str, Any]) -> Any:
                    return policy(dict(info))

            build_seconds = time.perf_counter() - build_started
            rollout_started = time.perf_counter()
            horizon_rollout(env, choose, 1, 1000)
            rollout_seconds = time.perf_counter() - rollout_started

            reconstruct_started = time.perf_counter()
            built = reconstruct_sumo_episode(env.recorder)
            reconstruct_seconds = time.perf_counter() - reconstruct_started
            # Structural counters only -- the campaign's own refusal checks.  `built` carries
            # e_sumo and the decomposition; neither is read, recorded or printed.
            structural = {
                "n_observations": int(built.n_observations),
                "n_teleports": int(built.n_teleports),
                "vehicle_types_seen": sorted(
                    {env._sumo.vehicle.getTypeID(v) for v in env._sumo.vehicle.getIDList()}
                ),
                "time_to_teleport_option": str(
                    env._sumo.simulation.getOption("time-to-teleport")
                ),
                "n_intersections": len(list(env.intersections)),
            }
        finally:
            env.close()
    in_process_seconds = time.perf_counter() - started

    record: dict[str, Any] = {
        "arm": arm,
        "draw_id": int(draw_id),
        "scenario_key": scenario_key,
        "seed": int(seed) if arm == "dt" else None,
        "halting_check": False,
        "in_process_seconds": in_process_seconds,
        "env_and_agent_build_seconds": build_seconds,
        "rollout_seconds": rollout_seconds,
        "reconstruct_seconds": reconstruct_seconds,
        "memory": sampler.peak.as_record(),
        "self_hwm_mib": round(_hwm_kib(os.getpid()) / 1024, 1),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "structural": structural,
    }
    if torch.cuda.is_available():
        record["torch_cuda_max_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / (1024 * 1024), 1
        )
        record["torch_cuda_max_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / (1024 * 1024), 1
        )
    return record


def _pool_worker(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """One cell in one spawned process.  Top-level so ``spawn`` can pickle it."""
    arm, kwargs = task
    try:
        # The parent samples the whole tree; a worker sampling too would multiply nvidia-smi calls.
        record = measure_single_cell(arm, sample_gpu=False, **kwargs)
        return {"ok": True, **record}
    except Exception as exc:  # noqa: BLE001 - a worker reports its failure as data
        return {"ok": False, "arm": arm, "error": f"{type(exc).__name__}: {exc}"}


def measure_pool(
    workers: int,
    *,
    arm: str = "dt",
    draw_id: int = G2_DRAW,
    scenario_key: str = GRID4X4_KEY,
    draws_root: str | Path,
    output_root: str | Path,
    seed: int = DT_SEED,
) -> dict[str, Any]:
    """*workers* copies of the same cell, concurrently, with the tree's peak RSS sampled.

    ``spawn`` rather than ``fork``, as ``transfer_curve.run_stage`` uses and for its reason: a
    forked worker inherits the parent's traci module state.  Every worker runs the SAME cell on
    draw 5 -- the point is the memory and the throughput of W concurrent SUMO processes, not W
    different numbers, and W identical cells cannot be mistaken for a result.
    """
    from multiprocessing import get_context

    kwargs = {
        "draw_id": int(draw_id),
        "scenario_key": scenario_key,
        "draws_root": str(draws_root),
        "output_root": str(output_root),
        "seed": int(seed),
    }
    started = time.perf_counter()
    with MemorySampler() as sampler:
        context = get_context("spawn")
        with context.Pool(processes=max(1, int(workers))) as pool:
            results = list(pool.imap_unordered(_pool_worker, [(arm, kwargs)] * int(workers)))
    wall_seconds = time.perf_counter() - started

    ok = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    in_process = sorted(float(r["in_process_seconds"]) for r in ok)
    return {
        "workers": int(workers),
        "arm": arm,
        "draw_id": int(draw_id),
        "n_ok": len(ok),
        "n_failed": len(failures),
        "failures": [r["error"] for r in failures],
        "wall_seconds": wall_seconds,
        "effective_seconds_per_cell": (wall_seconds / len(ok)) if ok else None,
        "in_process_seconds": in_process,
        "in_process_mean_seconds": (sum(in_process) / len(in_process)) if in_process else None,
        "speed_up": (sum(in_process) / wall_seconds) if in_process and wall_seconds else None,
        "memory": sampler.peak.as_record(),
        "worker_peak_self_rss_mib": sorted(
            float(r["memory"]["peak_self_rss_mib"]) for r in ok
        ),
    }


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------


@dataclass
class _Section:
    """One labelled block of the transcript, printed as it completes."""

    name: str
    body: dict[str, Any] = field(default_factory=dict)


def _print_memory(label: str, memory: Mapping[str, Any]) -> None:
    print(
        f"  {label}: peak tree RSS {memory['peak_tree_rss_mib']} MiB, peak self RSS "
        f"{memory['peak_self_rss_mib']} MiB, peak GPU used {memory['peak_gpu_used_mib']} MiB, "
        f"{memory['max_live_processes']} processes, {memory['n_samples']} samples",
        flush=True,
    )


def run_g2(
    *,
    draws_root: str | Path,
    output_root: str | Path,
    work_dir: str | Path,
    env_config: str | Path,
    pool_sizes: Sequence[int] = POOL_SIZES,
    draw_id: int = G2_DRAW,
    seed: int = DT_SEED,
) -> dict[str, Any]:
    """The whole gate: canary, the parity render, two single cells, then the pool scaling."""
    from offline.transfer_calibration import (
        canary_seconds,
        check_canary,
        format_canary_line,
    )

    sections: list[_Section] = []

    print("== canary (both halves), before anything is measured", flush=True)
    seconds, facts = canary_seconds()
    line = format_canary_line(seconds, facts)
    print(f"  {line}", flush=True)
    check_canary(facts)
    print("  canary correctness half: PASSED (hangzhou draw 0 reproduced its committed values)",
          flush=True)
    sections.append(_Section("canary", {"seconds": seconds, "line": line}))

    print(f"== parity for {GRID4X4_KEY} draw {draw_id} (B.2-2(1): the smoke draw)", flush=True)
    parity_record = ensure_draw_parity(
        draw_id, out_root=draws_root, env_config=env_config
    )
    print(
        f"  {parity_record['action']}: {parity_record['n_bound']}/{parity_record['n_vehicles']} "
        f"bound, format {parity_record['format_version']}, net {parity_record['net_sha256'][:12]}",
        flush=True,
    )
    sections.append(_Section("parity", parity_record))

    single: dict[str, Any] = {}
    for arm in ("dt", "fixedtime"):
        print(f"== single worker: {arm} cell, draw {draw_id}", flush=True)
        record = measure_single_cell(
            arm, draw_id=draw_id, draws_root=draws_root, output_root=output_root, seed=seed
        )
        single[arm] = record
        print(
            f"  in-process {record['in_process_seconds']:.2f} s "
            f"(build {record['env_and_agent_build_seconds']:.2f}, "
            f"rollout {record['rollout_seconds']:.2f}, "
            f"reconstruct {record['reconstruct_seconds']:.2f}); "
            f"self HWM {record['self_hwm_mib']} MiB",
            flush=True,
        )
        _print_memory("memory", record["memory"])
        structural = record["structural"]
        print(
            f"  structural: {structural['n_intersections']} intersections, "
            f"{structural['n_observations']} observations, "
            f"{structural['n_teleports']} teleports, types {structural['vehicle_types_seen']}, "
            f"time-to-teleport {structural['time_to_teleport_option']!r}",
            flush=True,
        )
    sections.append(_Section("single_worker", single))

    pools: dict[str, Any] = {}
    for workers in pool_sizes:
        print(f"== pool of {workers} workers: {workers} x the dt cell, draw {draw_id}", flush=True)
        record = measure_pool(
            workers, draw_id=draw_id, draws_root=draws_root, output_root=output_root, seed=seed
        )
        pools[f"w{workers}"] = record
        effective = record["effective_seconds_per_cell"]
        speed_up = record["speed_up"]
        effective_text = "n/a" if effective is None else f"{effective:.2f}"
        speed_up_text = "n/a" if speed_up is None else f"{speed_up:.2f}x"
        print(
            f"  {record['n_ok']} ok, {record['n_failed']} failed; wall "
            f"{record['wall_seconds']:.2f} s; effective {effective_text} s/cell; "
            f"speed-up {speed_up_text}",
            flush=True,
        )
        for failure in record["failures"]:
            print(f"    FAILED: {failure}", flush=True)
        _print_memory("memory", record["memory"])
    sections.append(_Section("pool_scaling", pools))

    from offline.transfer_curve import _git_provenance

    body = {section.name: section.body for section in sections}
    body["machine"] = _machine_record()
    record = {
        "format_version": G2_FORMAT_VERSION,
        "gate": "G2",
        "what_this_is": (
            "A TIMING AND MEMORY measurement on the smoke draw (BRIEF_39 Amendment B.2). It "
            "carries no travel time, no e_sumo, no return, no return-to-go and no action, from "
            "any cell. Nothing here is an evaluation of any arm, and nothing here may be reported "
            "as a result."
        ),
        FENCED_KEY: body,
        **_git_provenance(),
    }
    target = Path(work_dir) / "g2" / "g2_measurement.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    print(f"== wrote {target} (every measurement under {FENCED_KEY!r})", flush=True)
    return record


def _machine_record() -> dict[str, Any]:
    import torch

    record: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        record["cuda_device_name"] = torch.cuda.get_device_name(0)
        record["cuda_total_mib"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
        )
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    record["host_mem_total_mib"] = round(int(line.split()[1]) / 1024, 1)
                elif line.startswith("MemAvailable:"):
                    record["host_mem_available_mib"] = round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m offline.g2_measure",
        description=(
            "P7.3d gate G2: the one-cell measurement. Times and memory only; no outcome of any "
            "kind is computed for reporting, printed or stored."
        ),
    )
    parser.add_argument("--draws-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--draw", type=int, default=G2_DRAW)
    parser.add_argument("--seed", type=int, default=DT_SEED)
    parser.add_argument("--workers", type=int, nargs="*", default=list(POOL_SIZES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_g2(
        draws_root=args.draws_root,
        output_root=args.output_root,
        work_dir=args.work_dir,
        env_config=args.env_config,
        pool_sizes=tuple(args.workers),
        draw_id=int(args.draw),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
