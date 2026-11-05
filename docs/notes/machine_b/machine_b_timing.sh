#!/usr/bin/env bash
# MACHINE B TIMING for P9.3's training (rltraffic; BRIEF_48 I.6, 2026-10-29). Written by the coordinator.
#
# The pre-flight's step 5, which never ran: step 4's bitwise gate stopped the pre-flight, and the author replaced that
# gate the same day with a numerical comparison (machine_b_compare.py), which B passed. Run alone and as B's sessions
# will run: from the run tree, one thread, CUDA, and no CPU kernel settings (on B they change nothing: its CPU cell's
# checkpoint was byte-identical with and without them).
#   1. the timing: each arm at 2,000 steps on maxpressure seed 101, CUDA, with a GPU trace every second; per arm the
#      training seconds, setup and peak GPU memory beside G3b's laptop timings; B's whole training projected; and B's
#      sessions: the queue's own planner (plan_session, on G3b's estimates) at the largest --hours that keeps every
#      session within SESSION_HOURS (6) by B's measured timings
#   2. the CPU cell maxpressure/dt_nomix seed 101 at ONE step: a record keeps only its last step's loss, so this
#      record's loss is the first step's, for the comparison the author asked for (the laptop's: 2.133220911026001)
# Run on machine B, in tmux, as:   bash /home/filip/rltraffic/docs/notes/machine_b/machine_b_timing.sh
# It writes ONLY under $OUT (default /home/filip/machine_b_timing) and refuses to start if $OUT exists. About 30 minutes.
# QUICK=1 is the coordinator's mechanics test on the laptop: 20 timing steps, and the session plan computed from G3b's
# own estimates (it must reproduce the laptop's plan of BRIEF_48 I.4); never on machine B.
set -u -o pipefail

OUT=${OUT:-/home/filip/machine_b_timing}
QUICK=${QUICK:-0}
SESSION_HOURS=${SESSION_HOURS:-6}
M=/home/filip/rltraffic
R=/home/filip/rltraffic-p93-run-c
PY=$M/.venv/bin/python
RUN_COMMIT=4dcbedc25021634c199897beffeab4d806b9afc2
ARMS="dt_nomix bc_m bc_top10_m bc_top10_perix_m iql_m"

if [ -e "$OUT" ]; then echo "REFUSED: $OUT exists; it is never overwritten"; exit 2; fi
mkdir -p "$OUT/roots" || exit 2

fail() { echo "TIMING FAILED at step $1: $2"; echo "TIMING EXIT 1"; exit 1; }
step() { echo; echo "=== $(date '+%F %T %Z') step $1: $2"; }
train_cell() {  # train_cell <output root> <tier> <arm> <device> <steps>: one cell through the frozen CLI, from the run tree
  (cd "$M" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$R" "$PY" -P -m offline.hz4x4_phase_b train \
    --tier "$2" --arm "$3" --seed 101 --device "$4" --gradient-steps "$5" --output-root "$1" --torch-threads 1)
}

main() {
  echo "MACHINE B TIMING (QUICK=$QUICK) into $OUT; this script at $(git -C "$M" rev-parse --short HEAD)"
  echo "host: $(hostname)"; lscpu | grep 'Model name'
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader || fail 0 "nvidia-smi"
  [ "$(git -C "$R" rev-parse HEAD)" = "$RUN_COMMIT" ] || fail 0 "the run tree is not at $RUN_COMMIT"
  [ -z "$(git -C "$R" status --porcelain)" ] || fail 0 "the run tree is not clean"
  echo "run tree: $RUN_COMMIT, clean"

  local steps=2000
  if [ "$QUICK" = 1 ]; then steps=20; fi
  step 1 "the timing: each arm at $steps steps on maxpressure seed 101, CUDA, GPU trace every second"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv -l 1 > "$OUT/gpu_timing.csv" 2>&1 &
  local trace=$! t0 t1 arm
  for arm in $ARMS; do
    echo "--- timing $arm"
    t0=$(date +%s.%N)
    train_cell "$OUT/roots/timing" maxpressure "$arm" cuda "$steps" || { kill "$trace"; fail 1 "$arm"; }
    t1=$(date +%s.%N)
    echo "WALL $arm $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }') $t0 $t1" | tee -a "$OUT/timing_wall.txt"
  done
  kill "$trace"
  (cd "$M" && PYTHONPATH="$R" "$PY" -P - "$OUT" "$steps" "$SESSION_HOURS") <<'PYEOF' || fail 1 "the timing summary"
import csv, json, sys, time
from pathlib import Path

import offline.hz4x4_phase_b as pb  # the run tree's: G3b's planning numbers, the queue's cell order and its planner

out, steps, session_hours = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
quick = steps != 2000
trace = []
with open(out / "gpu_timing.csv") as handle:
    for row in csv.reader(handle):
        if len(row) > 1 and row[1].strip().endswith("MiB"):
            stamp = time.mktime(time.strptime(row[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S"))
            trace.append((stamp, int(row[1].strip().split()[0])))
cell_b = {}
for line in (out / "timing_wall.txt").read_text().splitlines():
    _, arm, wall, t0, t1 = line.split()
    record = json.loads((out / "roots/timing/p9_3/train" / f"hz4x4_maxpressure_{arm}_seed101.json").read_text())
    seconds = float(record["seconds"])
    setup = float(wall) - seconds
    cell_b[arm] = seconds * (40_000 / steps) + setup
    peak = max([mib for stamp, mib in trace if float(t0) - 1 <= stamp <= float(t1) + 1] or [0])
    print(f"{arm:17s} B: {seconds:8.2f} s training, {setup:6.1f} s setup, peak {peak:5d} MiB | laptop (G3b): "
          f"{pb.TIMING_SECONDS_2000[arm]:7.2f} s, {pb.SETUP_SECONDS[arm]:5.1f} s | training ratio B/laptop "
          f"{seconds * (2000 / steps) / pb.TIMING_SECONDS_2000[arm]:5.2f} | one 40,000-step cell on B {cell_b[arm] / 3600:.2f} h")
cells = pb.training_cells()
total_b = sum(cell_b[arm] for _, arm, _ in cells)
total_laptop = sum(pb.planned_seconds(arm) for _, arm, _ in cells)
note = "  (QUICK: not a projection)" if quick else ""
print(f"projected P9.3 training on B: {total_b / 3600:.2f} h for {len(cells)} cells, against G3b's {total_laptop / 3600:.2f} h "
      f"on the laptop; factor {total_b / total_laptop:.2f}{note}")
print(f"peak GPU memory used during the timing (the whole card): {max([mib for _, mib in trace] or [0])} MiB")

# B's sessions: what train-queue --hours H selects (its own planner, on G3b's estimates), at the largest H, in steps of
# 0.05, at which EVERY session the queue can plan -- from any pending cell on, so a session ended early by STOP or a
# failure changes nothing -- takes at most session_hours by B's measured cells. QUICK times nothing worth projecting,
# so it plans on G3b's own estimates: that must reproduce the laptop's plan (BRIEF_48 I.4).
basis = {arm: pb.planned_seconds(arm) for arm in pb.ARMS} if quick else cell_b


def seconds_of(session):
    return sum(basis[arm] for _, arm, _ in session)


def sessions(hours):
    pending, plan = list(cells), []
    while pending:
        chosen = pb.plan_session(pending, hours=hours, max_cells=None)
        plan.append(chosen)
        pending = pending[len(chosen):]
    return plan


for hundredths in range(1200, 49, -5):
    hours = hundredths / 100
    if all(seconds_of(pb.plan_session(cells[i:], hours=hours, max_cells=None)) <= session_hours * 3600
           for i in range(len(cells))):
        break
else:
    raise SystemExit(f"no --hours from 12 down to 0.5 keeps every session within {session_hours:g} h")
plan = sessions(hours)
label = "G3b's estimates (QUICK)" if quick else "B, measured"
print(f"B's sessions: train-queue --hours {hours:.2f}; every session it can plan, from any pending cell, takes at most "
      f"{session_hours:g} h ({label}); the queue itself prints G3b's planned hours. From the first cell:")
for number, s in enumerate(plan, 1):
    first, last = s[0], s[-1]
    print(f"  session {number:2d}: {len(s):3d} cells, {'/'.join(map(str, first))} .. {'/'.join(map(str, last))}, "
          f"{seconds_of(s) / 3600:.2f} h ({label}); the queue prints "
          f"{sum(pb.planned_seconds(arm) for _, arm, _ in s) / 3600:.2f} h")
print(f"  {len(plan)} sessions, {sum(basis[arm] for _, arm, _ in cells) / 3600:.2f} h in all ({label})")
PYEOF

  step 2 "the CPU cell maxpressure/dt_nomix seed 101 at one step: the first step's loss"
  train_cell "$OUT/roots/cpu_1step" maxpressure dt_nomix cpu 1 || fail 2 "the CPU cell"
  "$PY" - "$OUT/roots/cpu_1step/p9_3/train/hz4x4_maxpressure_dt_nomix_seed101.json" <<'PYEOF' || fail 2 "its record"
import json, sys
record = json.load(open(sys.argv[1]))
print(f"CPU cell at one step: loss {record['final_loss']!r} (the laptop's 2.133220911026001), canonical digest "
      f"{record['canonical_digest']} (the laptop's 1c6e4d5ffc0a6a3ac4e3df34ed42f9793c6adab01cab5ae9790d4b6e2698bf22)")
PYEOF

  echo
  echo "TIMING DONE $(date '+%F %T %Z')"
  echo "TIMING EXIT 0"
}

main 2>&1 | tee -a "$OUT/timing.log"
exit "${PIPESTATUS[0]}"
