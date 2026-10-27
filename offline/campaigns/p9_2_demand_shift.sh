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
#      guard exits 2 and nothing else runs; a record left by an earlier start is not re-run;
#   6. check-guard: the record, written now or left by an earlier start, must authorise the evaluation -- passed, run
#      from a clean tree at this code, every declared arm and engine-seed check (BRIEF_47 Amendment A.2.2) -- or the
#      driver exits 2 and the token is KEPT (Amendment A.2.3);
#   7. consumes the token and opens one tmux window per shard, each running this driver's SHARD MODE (below).
#
# THE HANG (BRIEF_47 Amendment B, B.3; DEFERRED 104; docs/notes/P5.3c_HANG_2026-09-30.md): CityFlow's Engine::~Engine() can
# wait forever at a two-party barrier, holding the main thread and the GIL; this campaign destroys about 31,000 engines.
# SHARD MODE -- `bash RUN_TREE/offline/campaigns/p9_2_demand_shift.sh --run-shard i` -- runs `evaluate` for shard i under
# offline.demand_shift's per-cell watchdog (WATCHDOG_SECONDS, 600 s: a cell still running then is cut, every thread's stack
# is dumped into the shard's own log and the process exits 1) and RESTARTS the shard when it exits 1 with faulthandler's
# banner in the part of its log that attempt wrote: it resumes from the records on disk, needs no new token, and each
# restart is logged with the time and the dump's top frame; at most MAX_RESTARTS (20) restarts per shard. Any other
# non-zero exit stops the shard as before. The shard's log is $LOGS/shift_stage<STAGE>_shard<i>.log, ending with
# "SHARD i EXIT: <code>".
#
# P92_PYTHON, P92_MAIN_TREE (the tree step 4 materialises from), P92_OUTPUT_ROOT, P92_RUNS, P92_SESSION, SHARDS, STAGE,
# IQL_T_PINS, P92_WATCHDOG_SECONDS and P92_MAX_RESTARTS override the defaults (tests/test_p9_2_drivers.py,
# tests/test_demand_shift_watchdog.py).
#
# RESUME: one JSON record per cell, atomic; a shard skips every record that exists and validates and STOPS at a refused
# cell (the two routes disagreeing, a dirty tree). Re-run a stopped shard from its window with its shard-mode line (the
# environment as the window's command sets it), or the driver with a new token (existing draws are kept; a written guard
# is not re-run, it is checked). Status:
#   /home/filip/rltraffic/.venv/bin/python -m offline.demand_shift status
# Stage 2 (iql_t, P9.1's checkpoints) is STAGE=2 with IQL_T_PINS=<pin file>, once the coordinator's amendment names it.
set -euo pipefail

RUN_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${P92_PYTHON:-/home/filip/rltraffic/.venv/bin/python}
MAIN_TREE=${P92_MAIN_TREE:-/home/filip/rltraffic}
OUTPUT=${P92_OUTPUT_ROOT:-$MAIN_TREE/output}
RUNS=${P92_RUNS:-$OUTPUT/p9_2_runs}
WORK=$OUTPUT/p9_2/demand_shift
LOGS=$RUNS/logs
TOKEN=$RUNS/TOKEN_p92_shift
AUDIT=$RUN_TREE/docs/data/p9_2_demand_shift_audit.json
SHARDS=${SHARDS:-3}
STAGE=${STAGE:-1}
SESSION=${P92_SESSION:-p92-shift}
# B.3: the per-cell watchdog's timeout (offline.demand_shift.WATCHDOG_SECONDS), the restarts allowed per shard, and
# faulthandler's banner -- the first line it writes on a timeout, the one signal that restarts a shard.
WATCHDOG_SECONDS=${P92_WATCHDOG_SECONDS:-600}
MAX_RESTARTS=${P92_MAX_RESTARTS:-20}
WATCHDOG_BANNER='^Timeout \([0-9]+:[0-9]{2}:[0-9]{2}(\.[0-9]+)?\)!$'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

PINS_ARG=()
if [ "$STAGE" = "2" ] && [ -n "${IQL_T_PINS:-}" ]; then
  PINS_ARG=(--iql-t-pins "$IQL_T_PINS")
fi

# Stage 2 runs only with the pin file the coordinator's amendment names; $1 says what is refused.
refuse_without_pins() {
  if [ "$STAGE" = "2" ] && { [ -z "${IQL_T_PINS:-}" ] || [ ! -f "$IQL_T_PINS" ]; }; then
    echo "REFUSING TO $1: stage 2 needs IQL_T_PINS, the pin file the coordinator's amendment names" >&2
    exit 2
  fi
}

# B.3: shard $1 under the per-cell watchdog, restarted while it exits 1 with faulthandler's banner in the part of its log
# this attempt wrote, at most MAX_RESTARTS times; returns the last attempt's exit code.
run_shard() {
  local i=$1 log restarts=0 attempts=0 code=0 before segment frame
  log=$LOGS/shift_stage${STAGE}_shard$1.log
  mkdir -p "$LOGS"
  touch "$log"
  while :; do
    before=$(stat -c %s "$log")
    attempts=$((attempts + 1))
    code=0
    "$PY" -m offline.demand_shift --output-root "$OUTPUT" --work-dir "$WORK" evaluate --stage "$STAGE" --shard "$i" \
      --of "$SHARDS" --watchdog-seconds "$WATCHDOG_SECONDS" --watchdog-file "$log" "${PINS_ARG[@]}" 2>&1 \
      | tee -a "$log" || code=$?
    if [ "$code" -eq 0 ]; then
      break
    fi
    # Read this attempt's part of the log whole before searching it: grep stopping early must not SIGPIPE the reader.
    segment=$(tail -c "+$((before + 1))" "$log")
    if [ "$code" -ne 1 ] || ! grep -qE "$WATCHDOG_BANNER" <<<"$segment"; then
      echo "SHARD $i STOPPED: exit $code without the watchdog's dump; not restarted" | tee -a "$log"
      break
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "SHARD $i STOPPED: the watchdog cut it on $attempts attempts; the cap of $MAX_RESTARTS restarts is reached" \
        | tee -a "$log"
      break
    fi
    restarts=$((restarts + 1))
    frame=$(grep -m1 -E '^  File ' <<<"$segment" || true)
    echo "SHARD $i RESTART $restarts/$MAX_RESTARTS at $(date -u +%Y-%m-%dT%H:%M:%SZ): the watchdog cut a cell after" \
      "$WATCHDOG_SECONDS s; resuming from the records on disk; top frame: ${frame#  }" | tee -a "$log"
  done
  echo "SHARD $i EXIT: $code" | tee -a "$log"
  return "$code"
}

cd "$RUN_TREE"
if [ "${1:-}" = "--run-shard" ]; then
  refuse_without_pins "RUN A SHARD"
  if ! [[ "${2:-}" =~ ^[0-9]+$ ]] || [ "$2" -ge "$SHARDS" ]; then
    echo "REFUSING TO RUN A SHARD: '${2:-}' is not a shard of $SHARDS" >&2
    exit 2
  fi
  shard_code=0
  run_shard "$2" || shard_code=$?
  exit "$shard_code"
fi
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
refuse_without_pins "START"
mkdir -p "$LOGS"
echo "=== run tree $RUN_TREE at $(git rev-parse HEAD); token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
"$PY" -m offline.transfer_calibration canary 2>&1 | tee -a "$LOGS/shift_canary.log"
(cd "$MAIN_TREE" && PYTHONPATH="$RUN_TREE" "$PY" -P -m offline.demand_shift --audit "$AUDIT" materialise) 2>&1 \
  | tee -a "$LOGS/shift_materialise.log"
if [ -f "$WORK/guard.json" ]; then
  echo "=== the guard record exists; it is not re-run, and it is checked before the token is consumed"
else
  "$PY" -m offline.demand_shift --output-root "$OUTPUT" --work-dir "$WORK" guard 2>&1 | tee -a "$LOGS/shift_guard.log"
fi
if ! "$PY" -m offline.demand_shift --output-root "$OUTPUT" --work-dir "$WORK" check-guard 2>&1 \
    | tee -a "$LOGS/shift_guard.log"; then
  echo "REFUSING TO CONSUME THE TOKEN: the guard record at $WORK/guard.json does not authorise an evaluation;" \
    "the token is kept at $TOKEN" >&2
  exit 2
fi
rm -f "$TOKEN"
echo "=== token consumed; opening $SHARDS shard window(s) for stage $STAGE in tmux session $SESSION"
for i in $(seq 0 $((SHARDS - 1))); do
  tmux new-window -t "$SESSION" -n "stage${STAGE}_shard$i" \
    "bash -c 'cd $RUN_TREE; P92_PYTHON=$PY P92_OUTPUT_ROOT=$OUTPUT P92_RUNS=$RUNS STAGE=$STAGE SHARDS=$SHARDS IQL_T_PINS=${IQL_T_PINS:-} P92_WATCHDOG_SECONDS=$WATCHDOG_SECONDS P92_MAX_RESTARTS=$MAX_RESTARTS bash $RUN_TREE/offline/campaigns/p9_2_demand_shift.sh --run-shard $i; exec bash'"
done
echo "=== shards started; this window may be closed"
