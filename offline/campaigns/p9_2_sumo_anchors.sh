#!/usr/bin/env bash
# P9.2 -- A27(f)'s aligned anchors (BRIEF_47 section 3.3): the replica gate, then 600 cells in SHARDS tmux windows.
#
# USAGE (the author, in a tmux session he starts; RUN_TREE is the detached run worktree at the reviewed commit):
#   tmux new -s p92-sumo
#   bash RUN_TREE/offline/campaigns/p9_2_sumo_anchors.sh 2>&1 | tee -a /home/filip/rltraffic/output/p9_2_runs/logs/sumo_driver.log; echo "DRIVER EXIT: ${PIPESTATUS[0]}"
#
# WHAT IT DOES, IN THIS ORDER (every step that can refuse PRECEDES the token's consumption):
#   1. refuses a run tree with uncommitted changes (every cell records git_dirty and the validator refuses True);
#   2. refuses without the token  output/p9_2_runs/TOKEN_p92_sumo  (written by the author on the coordinator's
#      Amendment; it is consumed only after step 5 passes);
#   3. the machine-health canary (PROJECT_PLAN section 7), logged;
#   4. the replica gate -- the registered maxpressure and fixedtime (k = 4) on draws 1001 and 1000 (1000's cells run the
#      halting cross-check) of hz1x1 and grid4x4 through this task's SUMO replica, eight episodes, every measurement
#      field == the committed P7.3a / P7.3d record -- written once to output/p9_2/anchors/replica_gate.json; a failed
#      gate exits 2 and nothing else runs; a record left by an earlier start is not re-run;
#   5. check-gate: the record, written now or left by an earlier start, must authorise the roll -- passed, run from a
#      clean tree at this code, all eight declared checks (BRIEF_47 Amendment A.2.2) -- or the driver exits 2 and the
#      token is KEPT (Amendment A.2.3);
#   6. consumes the token and opens one tmux window per shard (roll-anchors --shard i --of SHARDS), each logging through
#      tee to output/p9_2_runs/logs/sumo_shard<i>.log and ending with "SHARD i EXIT: <code>".
#
# P92_PYTHON, P92_OUTPUT_ROOT, P92_RUNS, P92_SESSION and SHARDS override the defaults (tests/test_p9_2_drivers.py).
#
# RESUME: every cell is one JSON file written atomically; a shard skips every chunk that exists and validates, refuses
# (and leaves untouched) one that does not, and STOPS at a refused episode (an A23 teleport no collision explains, any
# hz1x1 teleport). To resume, re-run a shard's line from its window -- no token is needed for a shard -- or the whole
# driver with a new token (an existing gate record is not re-run; it is checked). Status at any time:
#   /home/filip/rltraffic/.venv/bin/python -m offline.sumo_anchor_rescore status
#
# COST (measured at G2, docs/returns/P9.2-preflight.md): 400 SUMO episodes (hz1x1 ~19 s and grid4x4 ~50 s each under
# concurrency, draw 1000 about 3.5x for its halting cross-check) and 200 CityFlow episodes (~1.4 s); the gate is eight
# SUMO episodes on one worker, four of them on draw 1000.
set -euo pipefail

RUN_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${P92_PYTHON:-/home/filip/rltraffic/.venv/bin/python}
OUTPUT=${P92_OUTPUT_ROOT:-/home/filip/rltraffic/output}
RUNS=${P92_RUNS:-$OUTPUT/p9_2_runs}
WORK=$OUTPUT/p9_2/anchors
LOGS=$RUNS/logs
TOKEN=$RUNS/TOKEN_p92_sumo
SHARDS=${SHARDS:-3}
SESSION=${P92_SESSION:-p92-sumo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$RUN_TREE"
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING TO START: $RUN_TREE has uncommitted changes; every cell would record git_dirty and be refused" >&2
  exit 2
fi
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run token at $TOKEN" >&2
  exit 2
fi
mkdir -p "$LOGS"
echo "=== run tree $RUN_TREE at $(git rev-parse HEAD); token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
"$PY" -m offline.transfer_calibration canary 2>&1 | tee -a "$LOGS/sumo_canary.log"
if [ -f "$WORK/replica_gate.json" ]; then
  echo "=== the replica gate record exists; it is not re-run, and it is checked before the token is consumed"
else
  "$PY" -m offline.sumo_anchor_rescore --output-root "$OUTPUT" --work-dir "$WORK" gate 2>&1 | tee -a "$LOGS/sumo_gate.log"
fi
if ! "$PY" -m offline.sumo_anchor_rescore --output-root "$OUTPUT" --work-dir "$WORK" check-gate 2>&1 \
    | tee -a "$LOGS/sumo_gate.log"; then
  echo "REFUSING TO CONSUME THE TOKEN: the replica gate record at $WORK/replica_gate.json does not authorise a roll;" \
    "the token is kept at $TOKEN" >&2
  exit 2
fi
rm -f "$TOKEN"
echo "=== token consumed; opening $SHARDS shard window(s) in tmux session $SESSION"
for i in $(seq 0 $((SHARDS - 1))); do
  tmux new-window -t "$SESSION" -n "shard$i" \
    "bash -c 'set -o pipefail; cd $RUN_TREE; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; $PY -m offline.sumo_anchor_rescore --output-root $OUTPUT --work-dir $WORK roll-anchors --shard $i --of $SHARDS 2>&1 | tee -a $LOGS/sumo_shard$i.log; echo \"SHARD $i EXIT: \${PIPESTATUS[0]}\" | tee -a $LOGS/sumo_shard$i.log; exec bash'"
done
echo "=== shards started; this window may be closed"
