"""Run g1_cell.py for every (row, device) cell of the latency registry, one process each, sequentially, each under
`timeout`. Results (one JSON line per cell, no timing value) -> g1_cells.jsonl; stderr per cell -> g1_cells_logs/."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK_TREE = "/home/filip/rltraffic-p82-g1"
MAIN = "/home/filip/rltraffic"
HERE = Path(__file__).resolve().parent
OUT = HERE / "g1_cells.jsonl"
LOGS = HERE / "g1_cells_logs"
LOGS.mkdir(exist_ok=True)
sys.path.insert(0, WORK_TREE)
import offline.compute_latency as cl  # noqa: E402

cells = [(row.row_id, device) for row in cl.ROWS for device in row.devices]
only = set(sys.argv[1:])
started = time.monotonic()
with open(OUT, "a", encoding="utf-8") as sink:
    sink.write(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": len(cells)}) + "\n")
    for row_id, device in cells:
        if only and row_id not in only:
            continue
        log = LOGS / f"{row_id}_{device}.log"
        with open(log, "w", encoding="utf-8") as err:
            proc = subprocess.run(
                ["timeout", "-k", "10", "300", f"{MAIN}/.venv/bin/python", str(HERE / "g1_cell.py"), WORK_TREE, row_id,
                 device],
                cwd=MAIN, stdout=subprocess.PIPE, stderr=err, text=True, check=False,
            )
        line = next((x for x in proc.stdout.splitlines() if x.startswith("G1CELL ")), None)
        result = json.loads(line[len("G1CELL "):]) if line else {"row": row_id, "device": device}
        result["exit"] = proc.returncode
        sink.write(json.dumps(result, sort_keys=True) + "\n")
        sink.flush()
    sink.write(json.dumps({"done": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           "elapsed_s": round(time.monotonic() - started, 1)}) + "\n")
