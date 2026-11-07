#!/usr/bin/env bash
# P9.1 -- A27's corrective round, points 1, 2, 3, 4, 6 (BRIEF_46; PREREGISTRATION.md A27 at v2.8-prereg-a27): the two
# random_fresh tiers collected and declared, then 255 trainings (serial, one GPU process, one torch thread), every new
# cell evaluated by BOTH routes (P5.2/P4.6's CLI evaluate and P8.4b's run_campaign, att_ours equal under ==), P3a's
# action sequences, and the registered hz1x1 cells A27 compares with re-rolled through P8.4b's own route.
#
# 0. USAGE -- from a DETACHED RUN WORKTREE at the reviewed commit (a tree on a branch, at another commit or with
#    uncommitted changes is refused), in ONE tmux session started by the author:
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p91-run <commit>
#      Step 1, open a pane:      mkdir -p /home/filip/rltraffic/output/p9_1_runs && tmux new -s p91
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p91-run/offline/campaigns/p9_1_corrective.sh <stage> <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/<stage>_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
#    Every long stage runs in SESSIONS of a few hours (the author's rule, 2026-10-28; BRIEF_46 Amendment E; PROJECT_PLAN
#    section 7 rule (5)): each session takes its OWN token (output/p9_1_runs/TOKEN_<stage>, created by the author with
#    `touch`; consumed by `rm` after every refusal has passed), says first how many hours it runs (P91_SESSION_HOURS,
#    default 6), ends by itself between units of work at that bound or when output/p9_1_runs/STOP exists, and the next
#    session resumes from disk, skipping every finished unit and re-running none.  While STOP exists no train or
#    evaluate session starts (collect and d4rl start as before: E.4-2d); nothing here creates or deletes it.  Stages:
#      collect    CPU, ~15 min (done 2026-10-16): collect-random-fresh on hz1x1 and grid4x4, then the declarations.
#      train      GPU, one session = P91_SESSION_HOURS of PLANNED runs: `corrective_round train --hours` plans the next
#                 missing runs in training_runs() order that fit (planning estimates, the pre-flight's per-arm figures;
#                 the first always included), prints the plan, trains exactly those, checks STOP before each run, and
#                 closes with a summary (runs trained and their seconds, the next missing run, the planned hours left).
#                 ~40.5 planned GPU-hours in all (BRIEF_46 E.5: about seven sessions of six hours).  A Unix-time GPU trace
#                 per session.
#      evaluate   CPU, after the LAST training (refused while any run is missing): the evaluation declared once (a later
#                 session verifies it unchanged), then the 53 new cells and the 38 old hz1x1 cells (18,200 episodes)
#                 with WORKERS shards (default 6, at most 6: PROJECT_PLAN section 10's cap), then P3a's registered
#                 references and the status.  A session ends between units -- a cell for route (i) and P3a, an episode
#                 for route (ii) and the old cells -- at P91_SESSION_HOURS or at STOP, and says how much is left.
#      campaign   RETIRED (BRIEF_46 Amendment E, E.4-2c): it could end only by Ctrl+C, losing the run in flight.
#      d4rl       GPU, ~35-45 min: P8.3's IQL arm re-run as iql_t (--iql-target-dropout off), five seeds, in P8.3's
#                 venv, beside the registered artifact (--compare-with); outputs under /home/filip/d4rl-calib-out/p9_1/.
#                 Never two trainings on the GPU at once.
#    The regime: OMP/MKL one thread; CUBLAS_WORKSPACE_CONFIG UNSET (P5.2's default regime, in which every registered
#    grid DT was trained: deterministic False).  A27(a): seconds are an UNCONTROLLED clock -- no power check, no canary.
#    KEEP THE MACHINE AWAKE for the whole stage (Windows: never sleep while plugged in).  Every trainer times itself with
#    time.time(), so a suspension inside a run adds hours to its seconds (the pre-flight lost 10 h 45 min that way); the
#    stage's Unix-time GPU trace ($LOGS/gpu_trace.csv) lets the report flag such a run, but cannot repair its number.
#
# 1. WHAT IT PRODUCES -- only under output/p9_1/, output/p9_1_runs/ and the two new corpus directories
#    datasets_v11/cf_hz1x1__random_fresh and cf_grid4x4__random_fresh (collect), and /home/filip/d4rl-calib-out/p9_1/
#    (d4rl).  Nothing under output/ is deleted, moved or overwritten; a corpus directory that exists is refused.
#
# 2. HANGS (DEFERRED 104; BRIEF_46 Amendment B, B.3): CityFlow's engine destructor can hang at an env's close, holding
#    the GIL.  Every worker that rolls CityFlow episodes -- the evaluate and old-cells shards, P3a's references -- runs
#    under `restarting`: armed by --stall-dump and --stall-seconds (offline/stall_guard.py), a worker with no env reset
#    or step for STALL_SECONDS (120 s, BRIEF_46 Amendment B.2; P91_STALL_SECONDS overrides) writes every thread's stack
#    to $LOGS/<worker>.stall<n>.txt and exits 1, and the driver starts it again, with no new token; it resumes from the
#    files on disk (route (ii) and the old cells per episode, route (i) and P3a per cell, so a cell in flight is rolled
#    again from scratch -- seeded, the same values).  At most MAX_RESTARTS (60, Amendment B.2; P91_MAX_RESTARTS
#    overrides) per worker, each logged in $LOGS/restarts.log with the time, the window and the dumped stack; any
#    other non-zero exit stops the stage as before.  The collect stage runs without it (B.3).

set -euo pipefail

MAIN=/home/filip/rltraffic
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
CORPUS=$MAIN/datasets_v11
DRAWS=$MAIN/scenarios/draws
RUNS=$OUTPUT/p9_1_runs
WORKERS=${WORKERS:-6}
D4RL_PY=/home/filip/d4rl-calib-venv/bin/python
D4RL_DATA=/home/filip/d4rl-calib-data/halfcheetah_medium_expert-v2.hdf5
D4RL_OUT=/home/filip/d4rl-calib-out/p9_1
# BRIEF_46 Amendment E: the hours a session runs without a break (the train stage's plan, the evaluate stage's deadline).
SESSION_HOURS=${P91_SESSION_HOURS:-6}
# B.3, its values corrected by BRIEF_46 Amendment B.2 (2026-10-27): a worker with no env reset or step for
# STALL_SECONDS is cut (offline/stall_guard.py re-arms at most once per 10 s, so after 110-120 s) and restarted at most
# MAX_RESTARTS times; P91_STALL_SECONDS and P91_MAX_RESTARTS override them, the window is passed to every worker as
# --stall-seconds, and every restart line prints both.  STALL_BANNER is the first characters faulthandler writes on a
# timeout -- the only signal that restarts a worker (offline/stall_guard.py BANNER).
STALL_SECONDS=${P91_STALL_SECONDS:-120}
MAX_RESTARTS=${P91_MAX_RESTARTS:-60}
STALL_BANNER='Timeout ('

WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 2
}

fail() {
  printf 'FAILED: %s\n' "$*" >&2
  exit 1
}

STAGE=${1:?usage: p9_1_corrective.sh collect|train|evaluate|d4rl <commit>}
COMMIT=${2:?usage: p9_1_corrective.sh collect|train|evaluate|d4rl <commit>}
case "$STAGE" in collect|train|evaluate|campaign|d4rl) ;; *) refuse "unknown stage $STAGE" ;; esac
[[ "$WORKERS" =~ ^[1-6]$ ]] || refuse "WORKERS must be 1-6 (PROJECT_PLAN section 10), not $WORKERS"
[[ "$STALL_SECONDS" =~ ^[0-9]+$ ]] && [ "$STALL_SECONDS" -gt 10 ] \
  || refuse "P91_STALL_SECONDS must be a whole number of seconds above the guard's 10 s re-arm interval, not $STALL_SECONDS"
[[ "$MAX_RESTARTS" =~ ^[0-9]+$ ]] || refuse "P91_MAX_RESTARTS must be a non-negative whole number, not $MAX_RESTARTS"
# BRIEF_46 Amendment E: the session rules -- P91_SESSION_HOURS here, the stop file below -- bind the train and evaluate
# sessions only; collect and d4rl start exactly as before E.4 (E.4-2d: "d4rl is unchanged").
case "$STAGE" in train|evaluate) [[ "$SESSION_HOURS" =~ ^[0-9]*([.][0-9]+)?$ && "$SESSION_HOURS" =~ [1-9] ]] \
  || refuse "P91_SESSION_HOURS must be a positive number of hours, not $SESSION_HOURS" ;; esac
[ "$STAGE" != campaign ] || refuse "the campaign stage is retired (BRIEF_46 Amendment E, E.4-2c): run the train stage in sessions, then the evaluate stage in sessions"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset CUBLAS_WORKSPACE_CONFIG
for variable in COVERAGE_PROCESS_START COVERAGE_PROCESS_CONFIG PYTHONTRACEMALLOC PYTHONDEVMODE PYTHONMALLOC PYTHONPROFILEIMPORTTIME; do
  [ -z "${!variable+set}" ] || refuse "$variable is set: a tracer, profiler or debug allocator does not belong in a run"
done

[ -x "$PY" ] || refuse "no interpreter at $PY"
[ -z "$(git -C "$WORK_TREE" status --porcelain --untracked-files=no)" ] || refuse "$WORK_TREE has uncommitted changes"
[ "$(git -C "$WORK_TREE" rev-parse HEAD)" = "$(git -C "$WORK_TREE" rev-parse --verify --quiet "$COMMIT^{commit}" || true)" ] || refuse "$WORK_TREE is not at $COMMIT"
if git -C "$WORK_TREE" symbolic-ref -q HEAD >/dev/null; then
  refuse "$WORK_TREE is on a branch; the run tree is a DETACHED worktree at the reviewed commit"
fi
if pgrep -f '^[^ ]*python[^ ]* -P -m (offline[.]corrective_round (.* )?(train|declare|declare-evaluation|evaluate|old-cells|p3a-references|collect-random-fresh)( |$)|calibration[.]d4rl_run )' >/dev/null; then
  refuse "another P9.1 stage is running (a training, a collection, a declaration, an evaluation or the D4RL re-run)"
fi
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.corrective_round as m; print(m.__file__)')
[ "$LOADED" = "$WORK_TREE/offline/corrective_round.py" ] || refuse "the module loads from $LOADED, not from $WORK_TREE"
case "$STAGE" in train|evaluate) [ ! -e "$RUNS/STOP" ] || refuse "$RUNS/STOP exists: no session starts while it exists (the author removes it, by hand, to start one)" ;; esac

cd "$MAIN"
cr() {
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.corrective_round --repo-root "$WORK_TREE" --output-root "$OUTPUT" \
    --corpus-root "$CORPUS" --draws-root "$DRAWS" "$@"
}

stage_check() {
  if [ "$STAGE" = d4rl ]; then
    [ -x "$D4RL_PY" ] || refuse "no interpreter at $D4RL_PY (P8.3's venv)"
    [ -f "$D4RL_DATA" ] || refuse "no dataset at $D4RL_DATA"
    [ -f "$WORK_TREE/docs/data/p8_3_d4rl_calibration.json" ] || refuse "the registered P8.3 artifact is missing"
    [ ! -e "$D4RL_OUT" ] || [ -z "$(ls -A "$D4RL_OUT")" ] || refuse "$D4RL_OUT is not empty; the re-run is written once"
  else
    cr check --stage "$STAGE" >/dev/null || refuse "the module's pre-run check refused (its reason is above)"
  fi
}
stage_check
TOKEN=$RUNS/TOKEN_$STAGE
[ -f "$TOKEN" ] || refuse "no token at $TOKEN (the author's, after the brief clears the stage)"
rm -- "$TOKEN"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGS=$RUNS/logs/$STAMP
mkdir -p "$LOGS"
echo "token consumed; stage $STAGE $STAMP from $WORK_TREE at $COMMIT; logs $LOGS"

TRACE_PID=""
start_trace() {
  # A Unix-time GPU sample every 20 s for the whole stage: a suspension of the machine shows as a gap between two
  # samples, and the report flags every run whose [started_unix, finished_unix] spans one (its seconds are wall time).
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
  [ "$code" -eq 0 ] || { tail -20 "$LOGS/$label.log" | sed 's/^/    /'; fail "$label exited $code (log $LOGS/$label.log)"; }
  echo "$label: ok"
}

# B.3: run one worker with --stall-dump; while it exits 1 with faulthandler's banner at the head of its own fresh stall
# file (no progress: offline/stall_guard.py), log the time and the dumped stack to $LOGS/restarts.log and start it again
# -- it resumes from the files on disk -- at most MAX_RESTARTS times.  Any other exit is returned as it is: 0 goes on,
# anything else stops the stage as before.
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
      printf '%s %s: no progress again after %s restarts; stopping (its stack in %s; the window was %s s)\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$MAX_RESTARTS" "$dump" "$STALL_SECONDS" >>"$LOGS/restarts.log"
      echo "$label: no progress after $MAX_RESTARTS restarts; stopping (log $LOGS/restarts.log)" >&2
      return 1
    fi
    restarts=$((restarts + 1))
    {
      printf '%s %s: no progress -- its stall timeout fired (window %s s, its stall file %s); restart %s of at most %s; the dumped stack:\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$STALL_SECONDS" "$dump" "$restarts" "$MAX_RESTARTS"
      sed 's/^/    /' "$dump"
    } >>"$LOGS/restarts.log"
    echo "$label: restart $restarts of at most $MAX_RESTARTS (no progress for its $STALL_SECONDS s window; its stack in $dump)" >&2
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

# BRIEF_46 Amendment E, E.4-2b: an evaluate session ends between its steps once its deadline has passed or STOP exists
# (each worker has already ended between units of work); session_end says how much is left, from the status on disk.
session_over() {
  [ -e "$RUNS/STOP" ] && return 0
  "$PY" -c 'import sys, time; sys.exit(0 if time.time() >= float(sys.argv[1]) else 1)' "$DEADLINE"
}
session_end() {
  local why="the hours bound ($SESSION_HOURS h)" code=0
  [ ! -e "$RUNS/STOP" ] || why="the stop file $RUNS/STOP"
  cr status >"$LOGS/status.json" 2>"$LOGS/status.err" || code=$?
  [ "$code" -eq 0 ] || fail "the status refused with $code (log $LOGS/status.err)"
  "$PY" -c 'import json, sys
s = json.load(open(sys.argv[1]))
e, old = s["evaluation"], s["old_cells_hz1x1"]
ii = ", ".join("%s %s of %s" % (k, v.get("missing"), v.get("declared")) for k, v in sorted(e["route_ii"].items()))
print("left: route (i) %d of %d cells; route (ii) episodes %s; P3a %d of %d cells; old-cell episodes %s of %s" % (
    e["cells_declared"] - e["route_i_cells_done"], e["cells_declared"], ii,
    e["p3a_cells_declared"] - e["p3a_cells_done"], e["p3a_cells_declared"], old.get("missing"), old.get("declared")))' "$LOGS/status.json"
  echo "stage evaluate $STAMP: SESSION ENDED at $why -- the next session resumes from disk under a new TOKEN_evaluate"
  exit 0
}

case "$STAGE" in
  collect)
    run_logged collect_hz1x1 cr collect-random-fresh --scenario hz1x1
    run_logged collect_grid4x4 cr collect-random-fresh --scenario grid4x4
    run_logged declare_mixture cr declare --what mixture
    run_logged declare_random_fresh_hz1x1 cr declare --what random_fresh --scenario hz1x1
    run_logged declare_random_fresh_grid4x4 cr declare --what random_fresh --scenario grid4x4
    echo "stage collect $STAMP: COMPLETE -- next: plan Q2's typed commit, then the train sessions"
    ;;
  train)
    start_trace
    echo "train session: it runs about $SESSION_HOURS hours without a break (P91_SESSION_HOURS; its plan below, in planning estimates) and ends between runs, at its plan's end or at $RUNS/STOP"
    cr train --hours "$SESSION_HOURS" --plan-only
    run_logged train cr train --hours "$SESSION_HOURS"
    sed -n '/^SESSION: /p; /^SESSION SUMMARY /p' "$LOGS/train.log"
    echo "stage train $STAMP: SESSION COMPLETE -- the next session resumes from disk under a new TOKEN_train"
    ;;
  evaluate)
    DEADLINE=$("$PY" -c 'import sys, time; print(f"{time.time() + float(sys.argv[1]) * 3600.0:.3f}")' "$SESSION_HOURS")
    echo "evaluate session: it runs about $SESSION_HOURS hours without a break (P91_SESSION_HOURS), PLUS the unit of work in flight in each shard at that moment (a route (i) cell is 500 episodes in one call: 7 min 47 s for an hz1x1 cell in the session smoke, longer on the grid), and ends between units at Unix $DEADLINE or at $RUNS/STOP"
    run_logged declare_evaluation cr declare-evaluation
    shards evaluate --deadline-unix "$DEADLINE"
    if session_over; then session_end; fi
    shards old-cells --deadline-unix "$DEADLINE"
    if session_over; then session_end; fi
    run_logged p3a_references restarting p3a_references cr p3a-references --deadline-unix "$DEADLINE"
    status_code=0
    cr status >"$LOGS/status.json" 2>"$LOGS/status.err" || status_code=$?
    [ "$status_code" -eq 0 ] || fail "the status refused with $status_code (log $LOGS/status.err)"
    if "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["complete"] is True else 1)' "$LOGS/status.json"; then
      echo "stage evaluate $STAMP: COMPLETE (status $LOGS/status.json)"
    elif session_over; then
      session_end
    else
      fail "the evaluation is not complete on disk although no stop came: read $LOGS/status.json"
    fi
    ;;
  d4rl)
    mkdir -p "$D4RL_OUT"
    cd "$WORK_TREE"
    run_logged d4rl env PYTHONPATH="$WORK_TREE" "$D4RL_PY" -P -m calibration.d4rl_run --threads 1 train --dataset "$D4RL_DATA" --out "$D4RL_OUT" --arms iql --iql-target-dropout off --compare-with "$WORK_TREE/docs/data/p8_3_d4rl_calibration.json"
    echo "stage d4rl $STAMP: COMPLETE (outputs $D4RL_OUT)"
    ;;
esac
