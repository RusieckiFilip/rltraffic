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
#    Stages, in this order, each under its OWN token (output/p9_1_runs/TOKEN_<stage>, created by the author with `touch` when the
#    brief's amendment clears the stage; consumed by `rm` after every refusal has passed):
#      collect    CPU, ~15 min: collect-random-fresh on hz1x1 and grid4x4 (draws 1-200, --random-rng per-episode), then
#                 declare the mixture and both random_fresh tiers.  STOP: plan Q2's reviewed commit types hz1x1
#                 random_fresh's prompt from its declaration before the next stage can pass its check.
#      campaign   GPU then CPU: train every missing run serially (~35 GPU-hours, the pre-flight's measurement); then
#                 declare the evaluation's three manifests from this one process; then evaluate the 53 new cells and
#                 re-roll the 21 old hz1x1 cells, each with WORKERS shards (default 6, at most 6: PROJECT_PLAN section
#                 10's cap per session); then P3a's registered references and the status.  Resumable: everything done
#                 on disk is skipped, a half-written run is refused.  Evaluation starts only after the LAST training run:
#                 a run in progress leaves its .partial on disk, which the status refuses by design.
#      d4rl       GPU, ~10 min: P8.3's IQL arm re-run as iql_t (--iql-target-dropout off), five seeds, in P8.3's venv,
#                 beside the registered artifact (--compare-with); outputs under /home/filip/d4rl-calib-out/p9_1/.  Run
#                 it when no training runs (never two trainings on the GPU at once).
#    The regime: OMP/MKL one thread; CUBLAS_WORKSPACE_CONFIG UNSET (P5.2's default regime, in which every registered
#    grid DT was trained: deterministic False).  A27(a): seconds are an UNCONTROLLED clock -- no power check, no canary.
#
# 1. WHAT IT PRODUCES -- only under output/p9_1/, output/p9_1_runs/ and the two new corpus directories
#    datasets_v11/cf_hz1x1__random_fresh and cf_grid4x4__random_fresh (collect), and /home/filip/d4rl-calib-out/p9_1/
#    (d4rl).  Nothing under output/ is deleted, moved or overwritten; a corpus directory that exists is refused.
#
# 2. HANGS (DEFERRED 104): CityFlow's engine destructor can hang at an env's close.  A hung evaluation worker is
#    killed by hand (its pane, Ctrl-C) and the stage started again with a new token: every finished episode is on disk
#    and is skipped; an episode in flight is rolled again from scratch.

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

WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 2
}

fail() {
  printf 'FAILED: %s\n' "$*" >&2
  exit 1
}

STAGE=${1:?usage: p9_1_corrective.sh collect|campaign|d4rl <commit>}
COMMIT=${2:?usage: p9_1_corrective.sh collect|campaign|d4rl <commit>}
case "$STAGE" in collect|campaign|d4rl) ;; *) refuse "unknown stage $STAGE" ;; esac
[[ "$WORKERS" =~ ^[1-6]$ ]] || refuse "WORKERS must be 1-6 (PROJECT_PLAN section 10), not $WORKERS"

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

shards() {
  local what=$1 pids=() k code=0
  for k in $(seq 0 $((WORKERS - 1))); do
    cr "$what" --shard "$k" --of "$WORKERS" >"$LOGS/${what}_shard$k.log" 2>&1 &
    pids+=("$!")
  done
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || { code=1; echo "$what shard $k FAILED; the last lines of its log:"; tail -15 "$LOGS/${what}_shard$k.log" | sed 's/^/    /'; }
  done
  [ "$code" -eq 0 ] || fail "a $what shard failed; finished episodes are on disk, start again with a new token"
  echo "$what: $WORKERS shards ok"
}

case "$STAGE" in
  collect)
    run_logged collect_hz1x1 cr collect-random-fresh --scenario hz1x1
    run_logged collect_grid4x4 cr collect-random-fresh --scenario grid4x4
    run_logged declare_mixture cr declare --what mixture
    run_logged declare_random_fresh_hz1x1 cr declare --what random_fresh --scenario hz1x1
    run_logged declare_random_fresh_grid4x4 cr declare --what random_fresh --scenario grid4x4
    echo "stage collect $STAMP: COMPLETE -- next: plan Q2's typed commit, then the campaign stage under its own token"
    ;;
  campaign)
    start_trace
    run_logged train cr train
    run_logged declare_evaluation cr declare-evaluation
    shards evaluate
    shards old-cells
    run_logged p3a_references cr p3a-references
    status_code=0
    cr status >"$LOGS/status.json" 2>"$LOGS/status.err" || status_code=$?
    [ "$status_code" -eq 0 ] || fail "the status refused with $status_code (log $LOGS/status.err)"
    "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["complete"] is True else 1)' "$LOGS/status.json" || fail "the campaign is not complete on disk: read $LOGS/status.json"
    echo "stage campaign $STAMP: COMPLETE (status $LOGS/status.json)"
    ;;
  d4rl)
    mkdir -p "$D4RL_OUT"
    cd "$WORK_TREE"
    run_logged d4rl env PYTHONPATH="$WORK_TREE" "$D4RL_PY" -P -m calibration.d4rl_run --threads 1 train --dataset "$D4RL_DATA" --out "$D4RL_OUT" --arms iql --iql-target-dropout off --compare-with "$WORK_TREE/docs/data/p8_3_d4rl_calibration.json"
    echo "stage d4rl $STAMP: COMPLETE (outputs $D4RL_OUT)"
    ;;
esac
