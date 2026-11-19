#!/usr/bin/env bash
# P9.4 -- the transfer test's second DT retrained on a corrected 50 % mixture (docs/plans/p9.4_mix50_fresh.md, tags
# p9.4-plan and p9.4-plan-a1).  EXPLORATORY, ADDED AFTER THE RESULTS.  Three stages, each run by the author in tmux.
#
# 0. USAGE -- from a DETACHED RUN WORKTREE at the reviewed commit, named by its FULL 40-character sha (a tree on a
#    branch, at another commit, with any uncommitted, untracked or ignored file, or whose git state cannot be read is
#    refused), one tmux session per stage, started by the author:
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p94-run <commit>
#      Step 1, open a pane:    mkdir -p /home/filip/rltraffic/output/p9_4_runs && tmux new -s p94-<stage>
#      Step 2, at ITS PROMPT:  set -o pipefail; bash /home/filip/rltraffic-p94-run/offline/campaigns/p9_4_mix50_fresh.sh <stage> <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_4_runs/<stage>_capture.txt; echo "DRIVER EXIT $?"
#    `set -o pipefail` makes `$?` the driver's status rather than tee's; tee's -i keeps Ctrl-C's lines in the capture.
#    Every session takes its OWN token, output/p9_4_runs/TOKEN_<stage>, which the author creates with `touch` and the
#    driver consumes with `rm` after every refusal has passed.  A session runs at most P94_SESSION_HOURS
#    hours without a break (default 3; it says so first), ends by itself between units of work at that bound or when
#    output/p9_4_runs/STOP exists, and the next session resumes from disk, skipping every finished unit and re-running
#    none.  While STOP exists no session starts; nothing here creates or deletes it.  The stages, in this order:
#      train     GPU.  Plan section 3's control first: the registered mix50 retrained, five seeds, its record written as
#                soon as the fifth is done -- every parameter torch.equal to P4.7's checkpoint or the stage stops (exit 3,
#                the report printed) and nothing else is trained or evaluated; then mix50_fresh's five seeds.  Before
#                the first gradient step of every session: the committed declaration at its pin and Amendment A1's
#                re-hash of the 200 declared files.
#      cityflow  CPU, only after the control passed: route (i) (500 episodes, one unit), route (ii)'s manifest declared
#                once, then route (ii) in WORKERS shards (default 6, at most 6), any disagreement stopping the stage.
#      sumo      CPU, only after the control passed: G1 (100 cells) and G2 (40 cells) compared under == with their
#                committed values -- any difference stops the stage (exit 3) before its first new cell -- then the
#                1,000 cells in a pool of WORKERS processes.
#    The regime: OMP/MKL one thread; CUBLAS_WORKSPACE_CONFIG UNSET; the working directory and PYTHONPATH the run tree,
#    -P on every interpreter call (train_dt stamps the working directory's commit; P5.3c's regime requires them equal).
#
# 1. WHAT IT PRODUCES -- only under output/p9_4/ and output/p9_4_runs/ (the module's barrier refuses anything else).
#    Nothing under output/ is deleted, moved or overwritten; a file that may not be reused stops the stage untouched.
#    The driver's own logs are new files per session, written under `set -o noclobber`; no interpreter writes into the
#    run tree (PYTHONDONTWRITEBYTECODE), so the next session's clean-tree check sees the tree as it was checked out.
#
# 2. HANGS (DEFERRED 104): CityFlow's engine destructor can hang at an env's close.  Every CityFlow worker -- route (i)
#    and route (ii)'s shards -- runs under `restarting` (P9.1's loop): armed by --stall-dump and --stall-seconds
#    (offline/stall_guard.py), a worker with no env reset or step for STALL_SECONDS (120 s; P94_STALL_SECONDS) writes
#    its stack to $LOGS/<worker>.stall<n>.txt and exits 1, and the driver starts it again, at most MAX_RESTARTS (60;
#    P94_MAX_RESTARTS) times; it resumes from the files on disk.  Any other non-zero exit stops the stage.

set -euo pipefail
set -o noclobber

MAIN=/home/filip/rltraffic
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
CORPUS=$MAIN/datasets_v11
DRAWS=$MAIN/scenarios/draws
RUNS=$OUTPUT/p9_4_runs
WORKERS=${WORKERS:-6}
SESSION_HOURS=${P94_SESSION_HOURS:-3}
STALL_SECONDS=${P94_STALL_SECONDS:-120}
MAX_RESTARTS=${P94_MAX_RESTARTS:-60}
STALL_BANNER='Timeout ('

WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 2
}

fail() {
  printf 'FAILED: %s\n' "$*" >&2
  exit 1
}

[ "$#" -eq 2 ] || refuse "usage: p9_4_mix50_fresh.sh train|cityflow|sumo <the reviewed commit's full 40-character sha>"
STAGE=$1
COMMIT=$2
case "$STAGE" in train|cityflow|sumo) ;; *) refuse "unknown stage $STAGE (train, cityflow or sumo)" ;; esac
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || refuse "the commit must be the full 40-character sha the report reads back (git rev-parse <commit>), not $COMMIT"
[[ "$WORKERS" =~ ^[1-6]$ ]] || refuse "WORKERS must be 1-6 (plan section 7), not $WORKERS"
[[ "$SESSION_HOURS" =~ ^[0-9]*([.][0-9]+)?$ && "$SESSION_HOURS" =~ [1-9] ]] \
  || refuse "P94_SESSION_HOURS must be a positive number of hours, not $SESSION_HOURS"
[[ "$STALL_SECONDS" =~ ^[0-9]+$ ]] && [ "$STALL_SECONDS" -gt 10 ] \
  || refuse "P94_STALL_SECONDS must be a whole number of seconds above the guard's 10 s re-arm interval, not $STALL_SECONDS"
[[ "$MAX_RESTARTS" =~ ^[0-9]+$ ]] || refuse "P94_MAX_RESTARTS must be a non-negative whole number, not $MAX_RESTARTS"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset CUBLAS_WORKSPACE_CONFIG
for variable in COVERAGE_PROCESS_START COVERAGE_PROCESS_CONFIG PYTHONTRACEMALLOC PYTHONDEVMODE PYTHONMALLOC PYTHONPROFILEIMPORTTIME; do
  [ -z "${!variable+set}" ] || refuse "$variable is set: a tracer, profiler or debug allocator does not belong in a run"
done

[ -x "$PY" ] || refuse "no interpreter at $PY"
# Every git check FAILS CLOSED (review D, 2026-11-19): a git that cannot answer is a refusal, never an empty answer.
porcelain=$(git -C "$WORK_TREE" status --porcelain --ignored) \
  || refuse "git cannot read the state of $WORK_TREE; a tree whose state cannot be read is not run"
[ -z "$porcelain" ] \
  || refuse "$WORK_TREE has uncommitted changes or untracked or ignored files: $(printf '%s' "$porcelain" | head -3 | tr '\n' ' ')"
head=$(git -C "$WORK_TREE" rev-parse --verify HEAD) || refuse "git cannot read the HEAD of $WORK_TREE"
[ "$head" = "$COMMIT" ] || refuse "$WORK_TREE is not at $COMMIT (its HEAD is $head)"
on_branch=0
git -C "$WORK_TREE" symbolic-ref -q HEAD >/dev/null || on_branch=$?
case "$on_branch" in
  0) refuse "$WORK_TREE is on a branch; the run tree is a DETACHED worktree at the reviewed commit" ;;
  1) ;;
  *) refuse "git cannot tell whether $WORK_TREE is on a branch (git symbolic-ref exited $on_branch)" ;;
esac
if pgrep -f '^[^ ]*python[^ ]* -P -m offline[.]p9_4_mix50_fresh (.* )?(train|route-i|declare-route-ii|route-ii|sumo)( |$)' >/dev/null; then
  refuse "another P9.4 stage is running (a training, a CityFlow route or a SUMO session)"
fi
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.p9_4_mix50_fresh as m; print(m.__file__)') \
  || refuse "the P9.4 module does not import from $WORK_TREE"
[ "$LOADED" = "$WORK_TREE/offline/p9_4_mix50_fresh.py" ] || refuse "the module loads from $LOADED, not from $WORK_TREE"
[ ! -e "$RUNS/STOP" ] || refuse "$RUNS/STOP exists: no session starts while it exists (the author removes it, by hand, to start one)"

cd "$WORK_TREE"
cr() {
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.p9_4_mix50_fresh --repo-root "$WORK_TREE" --output-root "$OUTPUT" \
    --corpus-root "$CORPUS" --draws-root "$DRAWS" "$@"
}

stage_check() {
  cr check --stage "$STAGE" >/dev/null || refuse "the module's pre-run check refused (its reason is above)"
}
stage_check
TOKEN=$RUNS/TOKEN_$STAGE
[ -f "$TOKEN" ] || refuse "no token at $TOKEN (the author creates it with touch to start a session)"
rm -- "$TOKEN"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGS=$RUNS/logs/$STAMP
mkdir -p "$LOGS"
echo "token consumed; stage $STAGE $STAMP from $WORK_TREE at $COMMIT; logs $LOGS"

TRACE_PID=""
start_trace() {
  # A Unix-time GPU sample every 20 s: a suspension of the machine shows as a gap between two samples.
  ( while :; do
      printf '%s,%s\n' "$(date -u +%s.%N)" "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
      sleep 20
    done ) >>"$LOGS/gpu_trace.csv" 2>/dev/null &
  TRACE_PID=$!
}
stop_trace() {
  [ -z "$TRACE_PID" ] || kill "$TRACE_PID" 2>/dev/null || true
}
trap stop_trace EXIT

run_logged() {
  local label=$1
  shift
  local code=0
  "$@" >"$LOGS/$label.log" 2>&1 || code=$?
  if [ "$code" -eq 3 ]; then
    tail -40 "$LOGS/$label.log" | sed 's/^/    /'
    fail "$label: a stopping control failed (plan 4.3 / section 3); its report is above and in $LOGS/$label.log -- nothing else runs, the author decides"
  fi
  [ "$code" -eq 0 ] || { tail -20 "$LOGS/$label.log" | sed 's/^/    /'; fail "$label exited $code (log $LOGS/$label.log)"; }
  echo "$label: ok"
}

# P9.1's loop: run one CityFlow worker with --stall-dump; while it exits 1 with faulthandler's banner at the head of its
# own fresh stall file, log it and start it again -- it resumes from the files on disk -- at most MAX_RESTARTS times.
restarting() {
  local label=$1 restarts=0 code dump first
  shift
  while :; do
    dump=$LOGS/$label.stall$restarts.txt
    code=0
    "$@" --stall-dump "$dump" --stall-seconds "$STALL_SECONDS" || code=$?
    [ "$code" -ne 0 ] || return 0
    first=$(head -c "${#STALL_BANNER}" "$dump" 2>/dev/null || true)
    if [ "$code" -ne 1 ] || [ "$first" != "$STALL_BANNER" ]; then
      return "$code"
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      printf '%s %s: no progress again after %s restarts; stopping (its stack in %s)\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$MAX_RESTARTS" "$dump" >>"$LOGS/restarts.log"
      echo "$label: no progress after $MAX_RESTARTS restarts; stopping (log $LOGS/restarts.log)" >&2
      return 1
    fi
    restarts=$((restarts + 1))
    {
      printf '%s %s: no progress (window %s s, stall file %s); restart %s of at most %s; the dumped stack:\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$STALL_SECONDS" "$dump" "$restarts" "$MAX_RESTARTS"
      sed 's/^/    /' "$dump"
    } >>"$LOGS/restarts.log"
    echo "$label: restart $restarts of at most $MAX_RESTARTS (its stack in $dump)" >&2
  done
}

shards() {
  local what=$1 pids=() k code=0
  shift
  for k in $(seq 0 $((WORKERS - 1))); do
    restarting "${what}_shard$k" cr "$what" --shard "$k" --of "$WORKERS" "$@" >"$LOGS/${what}_shard$k.log" 2>&1 &
    pids+=("$!")
  done
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || { code=1; echo "$what shard $k FAILED; the last lines of its log:"; tail -15 "$LOGS/${what}_shard$k.log" | sed 's/^/    /'; }
  done
  [ "$code" -eq 0 ] || fail "a $what shard failed; finished episodes are on disk, start again with a new token"
  echo "$what: $WORKERS shards ended cleanly"
}

session_over() {
  [ -e "$RUNS/STOP" ] && return 0
  "$PY" -c 'import sys, time; sys.exit(0 if time.time() >= float(sys.argv[1]) else 1)' "$DEADLINE"
}

# One file per call (noclobber refuses a second write to a name): status_complete.json, then possibly status_end.json.
stage_status() {
  local name=$1 code=0
  cr status >"$LOGS/$name.json" 2>"$LOGS/$name.err" || code=$?
  [ "$code" -eq 0 ] || fail "the status refused with $code (log $LOGS/$name.err)"
}

session_end() {
  local why="the hours bound ($SESSION_HOURS h)"
  [ ! -e "$RUNS/STOP" ] || why="the stop file $RUNS/STOP"
  stage_status status_end
  echo "stage $STAGE $STAMP: SESSION ENDED at $why -- what is left is in $LOGS/status_end.json; the next session resumes from disk under a new TOKEN_$STAGE"
  exit 0
}

complete_or_end() {
  local key=$1
  stage_status status_complete
  if "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))[sys.argv[2]]["complete"] is True else 1)' "$LOGS/status_complete.json" "$key"; then
    echo "stage $STAGE $STAMP: COMPLETE (status $LOGS/status_complete.json)"
  elif session_over; then
    session_end
  else
    fail "the $STAGE stage is not complete on disk although no stop came: read $LOGS/status_complete.json"
  fi
}

case "$STAGE" in
  train)
    start_trace
    echo "train session: it runs at most about $SESSION_HOURS hours without a break (P94_SESSION_HOURS; its plan below, in planning estimates) and ends between runs, at its plan's end or at $RUNS/STOP"
    cr train --hours "$SESSION_HOURS" --plan-only
    run_logged train cr train --hours "$SESSION_HOURS"
    sed -n '/^SESSION PLAN /p; /^CONTROL: /p; /^  seed /p; /^SESSION SUMMARY /p' "$LOGS/train.log"
    echo "stage train $STAMP: SESSION COMPLETE -- the next session resumes from disk under a new TOKEN_train"
    ;;
  cityflow)
    DEADLINE=$("$PY" -c 'import sys, time; print(f"{time.time() + float(sys.argv[1]) * 3600.0:.3f}")' "$SESSION_HOURS")
    echo "cityflow session: it runs at most about $SESSION_HOURS hours without a break (P94_SESSION_HOURS), plus the unit in flight (route (i) is one unit of 500 episodes), and ends between units at Unix $DEADLINE or at $RUNS/STOP"
    run_logged route_i restarting route_i cr route-i --deadline-unix "$DEADLINE"
    if session_over; then session_end; fi
    run_logged declare_route_ii cr declare-route-ii
    shards route-ii --deadline-unix "$DEADLINE"
    complete_or_end cityflow
    ;;
  sumo)
    DEADLINE=$("$PY" -c 'import sys, time; print(f"{time.time() + float(sys.argv[1]) * 3600.0:.3f}")' "$SESSION_HOURS")
    echo "sumo session: it runs at most about $SESSION_HOURS hours without a break (P94_SESSION_HOURS), plus the batch of $WORKERS cells in flight, and ends between batches at Unix $DEADLINE or at $RUNS/STOP; G1 and G2 come first"
    run_logged sumo cr sumo --workers "$WORKERS" --deadline-unix "$DEADLINE"
    sed -n '/^GUARDS: /p; /^SUMO SESSION /p' "$LOGS/sumo.log"
    complete_or_end sumo
    ;;
esac
