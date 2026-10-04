"""Run g11_cell.py for every (row, device) cell, one process each, sequentially, each under `timeout`, with the
driver's environment (OMP/MKL = 1, no CUBLAS, no tracer variable, -P). Results -> g11_cells.jsonl (no timing value)."""
import json, os, subprocess, sys, time
from pathlib import Path
WORK_TREE = "/home/filip/rltraffic-p82-g11"
MAIN = "/home/filip/rltraffic"
HERE = Path(__file__).resolve().parent
OUT = HERE / "g11_cells.jsonl"
LOGS = HERE / "g11_cells_logs"
LOGS.mkdir(exist_ok=True)
SCRATCH = HERE / "g11_scratch_out"
SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0, WORK_TREE)
import offline.compute_latency as cl  # noqa: E402
cells = [(row.row_id, device) for row in cl.ROWS for device in row.devices]
env = {k: v for k, v in os.environ.items() if k not in ("CUBLAS_WORKSPACE_CONFIG", "COVERAGE_PROCESS_START",
       "COVERAGE_PROCESS_CONFIG", "PYTHONTRACEMALLOC", "PYTHONDEVMODE", "PYTHONMALLOC", "PYTHONPROFILEIMPORTTIME")}
env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
started = time.monotonic()
with open(OUT, "a", encoding="utf-8") as sink:
    sink.write(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": len(cells)}) + "\n")
    for row_id, device in cells:
        t0 = time.monotonic()
        with open(LOGS / f"{row_id}_{device}.log", "w", encoding="utf-8") as err:
            proc = subprocess.run(["timeout", "-k", "10", "300", f"{MAIN}/.venv/bin/python", "-P", str(HERE / "g11_cell.py"),
                                   WORK_TREE, row_id, device, str(SCRATCH)],
                                  cwd=MAIN, env=env, stdout=subprocess.PIPE, stderr=err, text=True, check=False)
        line = next((x for x in proc.stdout.splitlines() if x.startswith("G11CELL ")), None)
        result = json.loads(line[len("G11CELL "):]) if line else {"row": row_id, "device": device}
        result.update(exit=proc.returncode, done_line="G11DONE" in proc.stdout, wall_s=round(time.monotonic() - t0, 1))
        sink.write(json.dumps(result, sort_keys=True) + "\n")
        sink.flush()
    sink.write(json.dumps({"done": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.monotonic() - started, 1)}) + "\n")
