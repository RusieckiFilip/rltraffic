"""P8.2 gate G3 (BRIEF_43 Amendment B.1, B.1.4): read the timing run's 78 latency records from disk by a THIRD route.

Imports nothing from the project. For every record: the protocol fields, the power block, the device evidence, the
regime, the provenance, the checkpoint file re-hashed, no outcome key; then n_timed, the median and the nearest-rank p95
recomputed from the record's own per-decision nanoseconds -- the median with ``statistics.median``, the p95 by its
DEFINITION (the smallest observed value v with #{x <= v} >= 0.95 n, found by cumulative counting, not by index
arithmetic) -- and every published millisecond figure compared with the recomputed nanoseconds. Writes g3_records.json.

usage: python g3_records.py <run_dir> <main_tree> <out_json>
"""
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

RUN, MAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FORBIDDEN = {"att", "reward", "rewards", "return", "returns", "travel", "queue", "queues", "waiting", "vehicle",
             "vehicles", "throughput", "delay", "completed", "entered", "metric", "metrics", "rho", "pressure"}
BEST = "ded574b5-45a0-4f42-8737-46345c09c238"
N_IX = {"hz1x1": 1, "grid4x4": 16}


def keys(node, trail=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield f"{trail}.{k}", str(k)
            yield from keys(v, f"{trail}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from keys(v, f"{trail}[{i}]")


def p95_by_definition(values):
    n = len(values)
    cum = 0
    for v, c in sorted(Counter(values).items()):
        cum += c
        if cum * 100 >= 95 * n:
            return v
    raise AssertionError("unreachable")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


run = json.loads((RUN / "run.json").read_text())
cells = [label.rsplit("_", 1) for label in run["row_order"]]
problems, out, hashed = [], [], {}
for row_id, device in cells:
    name = f"{row_id}_{device}.json"
    r = json.loads((RUN / name).read_text())
    p = []
    if r.get("format_version") != "p8.2-latency/1.1": p.append("format")
    if (r.get("row"), r.get("device")) != (row_id, device): p.append("row/device")
    scen = row_id.split(".")[0]
    if r.get("scenario") != scen: p.append("scenario")
    if r.get("draws") != [1000, 1001, 1002] or [e["draw"] for e in r["episodes"]] != [1000, 1001, 1002]: p.append("draws")
    if r.get("warmup") != 20 or r.get("engine_seed") != 1000 or r.get("max_steps") != 360: p.append("protocol")
    eps = [e["decision_ns"] for e in r["episodes"]]
    if [len(e) for e in eps] != [360, 360, 360]: p.append("episode lengths")
    if not all(isinstance(x, int) and x > 0 for e in eps for x in e): p.append("non-positive or non-integer ns")
    bad_keys = [t for t, k in keys(r) if any(tok in FORBIDDEN for tok in k.lower().split("_"))]
    if bad_keys: p.append(f"outcome keys {bad_keys[:3]}")
    g = r.get("git") or {}
    if g.get("commit") != run["git"]["commit"] or g.get("dirty") is not False: p.append("git")
    reg = r.get("regime") or {}
    if (reg.get("torch_num_threads"), reg.get("OMP_NUM_THREADS"), reg.get("MKL_NUM_THREADS"), reg.get("CUBLAS_WORKSPACE_CONFIG"),
            reg.get("device")) != (1, "1", "1", None, device) or not str(reg.get("tracers", "")).startswith("none"):
        p.append(f"regime {reg}")
    m = r.get("machine") or {}
    pw = m.get("power") or {}
    mains = any(i.get("type") == "Mains" and i.get("online") == 1 for i in (pw.get("supplies") or {}).get("items", []))
    if not mains or (pw.get("windows") or {}).get("ac_overlay") != BEST: p.append("power")
    ev = r.get("device_evidence") or {}
    if device == "cpu" and ev.get("cuda_initialized"): p.append("cpu row touched CUDA")
    if device == "cuda" and not ((ev.get("cuda_max_memory_allocated") or 0) > 0 and ev.get("cuda_initialized")): p.append("cuda row no memory")
    ck = r.get("checkpoint")
    if ck is not None:
        path = MAIN / ck["path"]
        hashed.setdefault(ck["path"], sha256(path))
        if hashed[ck["path"]] != ck["sha256"]: p.append("checkpoint digest")
    # the third route
    timed = [x for e in eps for x in e[20:]]
    n = len(timed)
    med = statistics.median(timed)
    p95 = p95_by_definition(timed)
    if n != 1020 or r["n_timed"] != n: p.append(f"n_timed {r['n_timed']} vs {n}")
    if r["median_ns"] != med: p.append(f"median_ns {r['median_ns']} vs {med}")
    if r["p95_ns"] != p95: p.append(f"p95_ns {r['p95_ns']} vs {p95}")
    if r["median_ms"] != med / 1e6 or r["p95_ms"] != p95 / 1e6: p.append("ms")
    nix = N_IX[scen]
    if r["n_intersections"] != nix or r["per_intersection"]["median_ms"] != (med / 1e6) / nix or r["per_intersection"]["p95_ms"] != (p95 / 1e6) / nix:
        p.append("per-intersection")
    warm = [x for e in eps for x in e[:20]]
    out.append({"row": row_id, "device": device, "median_ms": med / 1e6, "p95_ms": p95 / 1e6, "n_timed": n,
                "max_ms": max(timed) / 1e6, "min_ms": min(timed) / 1e6,
                "first_decision_ms": [e[0] / 1e6 for e in eps], "warmup_median_ms": statistics.median(warm) / 1e6,
                "per_episode_median_ms": [statistics.median(e[20:]) / 1e6 for e in eps],
                "load_before": r["load_before"], "load_after": r["load_after"], "cuda_mib": (ev.get("cuda_max_memory_allocated") or 0) / 2**20,
                "kernel": m.get("kernel_release"), "cpu": m.get("cpu_model"), "gpu": m.get("gpu_name"), "driver": m.get("gpu_driver"),
                "torch": m.get("torch_version"), "factory": r.get("factory"), "problems": p})
    if p:
        problems.append((name, p))
OUT.write_text(json.dumps({"run": run["stamp"], "cells": out, "problems": problems, "checkpoints_rehashed": len(hashed)}, indent=1))
print(f"cells read: {len(out)} | problems: {problems if problems else 'none'} | checkpoint files re-hashed: {len(hashed)}")
print("machine:", sorted({(c['cpu'], c['gpu'], c['driver'], c['torch'], c['kernel']) for c in out}))
