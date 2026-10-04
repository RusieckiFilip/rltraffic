"""G1.1 dry run of ONE latency cell through the REAL run_row (BRIEF_43 Amendment B): three episodes, every new check of
the row process (B2's device evidence, B5.3's tracer refusal, B1.1's power block), the record built and validated by
build_record -- and then DISCARDED by an injected writer that prints only non-timing facts. No timing value, no
statistic and no episode quantity is printed or kept.

usage: python -P g11_cell.py <work_tree> <row_id> <device> <scratch_out_dir>
"""
import os
import sys

WORK_TREE, ROW_ID, DEVICE, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, WORK_TREE)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import offline.compute_latency as cl  # noqa: E402

assert cl.__file__ == f"{WORK_TREE}/offline/compute_latency.py", cl.__file__
MAIN = Path("/home/filip/rltraffic")
row = cl.row_by_id(ROW_ID)


def discard(path, record):
    power = (record.get("machine") or {}).get("power") or {}
    keep = {
        "row": record["row"], "device": record["device"], "format_version": record["format_version"],
        "n_timed": record["n_timed"], "episodes": [len(e["decision_ns"]) for e in record["episodes"]],
        "n_intersections": record["n_intersections"], "device_evidence": record.get("device_evidence"),
        "tracers": record["regime"].get("tracers"), "threads": record["regime"]["torch_num_threads"],
        "kernel_release": record["machine"].get("kernel_release"),
        "power_problems": cl.power_regime_problems(power),
        "power_ac_overlay_name": (power.get("windows") or {}).get("ac_overlay_name"),
        "outcome_keys": cl.find_outcome_keys(record),
        "checkpoint_ok": (record["checkpoint"] is None and row.checkpoint is None)
        or (record["checkpoint"] or {}).get("sha256") == row.sha256,
        "git_dirty": record["git"].get("dirty"), "git_commit": (record["git"].get("commit") or "")[:7],
    }
    print("G11CELL " + json.dumps(keep, sort_keys=True), flush=True)


cl.run_row(row, DEVICE, out_dir=Path(OUT_DIR), output_root=MAIN / "output", corpus_root=MAIN / "datasets_v11",
           draws_root=MAIN / "scenarios" / "draws", data_dir=Path(WORK_TREE) / "docs" / "data", writer=discard)
print("G11DONE", flush=True)
os._exit(0)
