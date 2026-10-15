#!/usr/bin/env bash
# P9.2 -- A27(h)'s EXPLORATORY demand-shift evaluation (exploration, not H2; BRIEF_47 section 3.4): materialise the three
# files' held-out draws into the main tree, the guard, then stage STAGE's cells in SHARDS tmux windows.
#
# USAGE (the author, in a tmux session he starts; RUN_TREE is the detached run worktree at the reviewed commit):
#   tmux new -s p92-shift
#   bash RUN_TREE/offline/campaigns/p9_2_demand_shift.sh 2>&1 | tee -a /home/filip/rltraffic/output/p9_2_runs/logs/shift_driver.log; echo "DRIVER EXIT: ${PIPESTATUS[0]}"
#
# WHAT IT DOES, IN THIS ORDER (every step that can refuse PRECEDES the token's consumption):
#   1. refuses a run tree with uncommitted changes, and a missing committed audit (docs/data/p9_2_demand_shift_audit.json,
#      written at G2 BEFORE any draw);
#   2. refuses without the token  output/p9_2_runs/TOKEN_p92_shift;
#   3. the machine-health canary, logged;
#   4. materialises draws 1000-1099 of kn-hz, qc-yn and sb-sx into /home/filip/rltraffic/scenarios/draws FROM THE MAIN
#      TREE'S ROOT (the rendered cityflow.json embeds the scenario dir resolved against the cwd; DEFERRED 55), with the
#      run tree's code (PYTHONPATH + python -P); every flow.json is checked against the audit's digest; re-running keeps
#      every draw;
#   5. the guard -- one bc-tyc held-out cell per arm through the evaluation path, == P8.4b's committed cell, and the
#      engine-seed check on bc-tyc and each shifted file -- written once to output/p9_2/demand_shift/guard.json; a failed
#      guard exits 2 and nothing else runs;
#   6. consumes the token and opens one tmux window per shard (evaluate --stage STAGE --shard i --of SHARDS).
#
# RESUME: one JSON record per cell, atomic; a shard skips every record that exists and validates and STOPS at a refused
# cell (the two routes disagreeing, a dirty tree). Re-run a shard's line from its window, or the driver with a new token
# (existing draws are kept, a written guard is not re-run). Status:
#   /home/filip/rltraffic/.venv/bin/python -m offline.demand_shift status
# Stage 2 (iql_t, P9.1's checkpoints) is STAGE=2 with IQL_T_PINS=<pin file>, once the coordinator's amendment names it.
set -euo pipefail

RUN_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/filip/rltraffic/.venv/bin/python
MAIN_TREE=/home/filip/rltraffic
OUTPUT=${P92_OUTPUT_ROOT:-$MAIN_TREE/output}
RUNS=${P92_RUNS:-$OUTPUT/p9_2_runs}
WORK=$OUTPUT/p9_2/demand_shift
LOGS=$RUNS/logs
TOKEN=$RUNS/TOKEN_p92_shift
AUDIT=$RUN_TREE/docs/data/p9_2_demand_shift_audit.json
SHARDS=${SHARDS:-3}
STAGE=${STAGE:-1}
SESSION=${P92_SESSION:-p92-shift}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$RUN_TREE"
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING TO START: $RUN_TREE has uncommitted changes; every record would carry git_dirty and be refused" >&2
  exit 2
fi
if [ ! -f "$AUDIT" ]; then
  echo "REFUSING TO START: no committed audit at $AUDIT; the audit precedes any draw (A27(h))" >&2
  exit 2
fi
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run token at $TOKEN" >&2
  exit 2
fi
PINS_ARG=()
if [ "$STAGE" = "2" ]; then
  if [ -z "${IQL_T_PINS:-}" ] || [ ! -f "$IQL_T_PINS" ]; then
    echo "REFUSING TO START: stage 2 needs IQL_T_PINS, the pin file the coordinator's amendment names" >&2
    exit 2
  fi
  PINS_ARG=(--iql-t-pins "$IQL_T_PINS")
fi
mkdir -p "$LOGS"
echo "=== run tree $RUN_TREE at $(git rev-parse HEAD); token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
"$PY" -m offline.transfer_calibration canary 2>&1 | tee -a "$LOGS/shift_canary.log"
(cd "$MAIN_TREE" && PYTHONPATH="$RUN_TREE" "$PY" -P -m offline.demand_shift --audit "$AUDIT" materialise) 2>&1 \
  | tee -a "$LOGS/shift_materialise.log"
if [ -f "$WORK/guard.json" ]; then
  echo "=== the guard record exists; evaluate re-checks that it passed"
else
  "$PY" -m offline.demand_shift --output-root "$OUTPUT" --work-dir "$WORK" guard 2>&1 | tee -a "$LOGS/shift_guard.log"
fi
rm -f "$TOKEN"
echo "=== token consumed; opening $SHARDS shard window(s) for stage $STAGE in tmux session $SESSION"
for i in $(seq 0 $((SHARDS - 1))); do
  tmux new-window -t "$SESSION" -n "stage${STAGE}_shard$i" \
    "bash -c 'set -o pipefail; cd $RUN_TREE; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; $PY -m offline.demand_shift --output-root $OUTPUT --work-dir $WORK evaluate --stage $STAGE --shard $i --of $SHARDS ${PINS_ARG[*]} 2>&1 | tee -a $LOGS/shift_stage${STAGE}_shard$i.log; echo \"SHARD $i EXIT: \${PIPESTATUS[0]}\" | tee -a $LOGS/shift_stage${STAGE}_shard$i.log; exec bash'"
done
echo "=== shards started; this window may be closed"
