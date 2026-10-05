#!/usr/bin/env bash
# P8.2 -- the TIMING RUN (gate G2) and its PRE-FLIGHT (gate G1): every latency row of offline/compute_latency.py timed
# alone in real CityFlow episodes, sequentially, one process per (row, device), under ONE token (BRIEF_43 §4,
# Amendment A: Q8-Q14, A3.1, A3.2; Amendment B: B1, B5.3). No outcome of any episode is recorded or printed.
#
# 0. USAGE
#    The PRE-FLIGHT (G1, the implementer), from a committed, clean task worktree; no token:
#        mkdir -p /home/filip/rltraffic/output/p8_2_runs
#        bash offline/campaigns/p8_2_latency.sh --preflight 2>&1 | tee -i -a /home/filip/rltraffic/output/p8_2_runs/preflight_capture.txt
#    It times hz1x1.dt_k20 and grid4x4.dt_nomix_h4 (in each scenario the row whose network does the most work per
#    decision, by architecture) on both devices between two canaries and writes output/p8_2_runs/preflight_<UTC>/
#    preflight.json with the per-scenario timeouts, the canary timeout and an ESTIMATE of the run's duration. The record G1 accepts is then pinned below (PREFLIGHT_RECORD and
#    PREFLIGHT_SHA256): the run reads its timeouts from it and refuses while it is UNSET or at another digest.
#
#    The TIMING RUN (G2, the author), the FOREGROUND form, from the DETACHED RUN WORKTREE the coordinator creates at the
#    reviewed, pushed commit (a tree at any other commit, on a branch or with uncommitted changes is refused):
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p82-run <commit>
#      Step 1, open a pane:      mkdir -p /home/filip/rltraffic/output/p8_2_runs && tmux new -s p82_latency
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p82-run/offline/campaigns/p8_2_latency.sh <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p8_2_runs/latency_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
#    The token is output/p8_2_runs/TOKEN_latency, created by the author when the brief names the run commit (touch it).
#    Start on a quiet machine, on mains power with the Windows power mode set to Best Performance (Amendment B, B1):
#    the run calls power-check among its checks and refuses before the token otherwise, each canary re-checks before it
#    times anything, and a regime lost by the closing canary makes the run FAILED. The canary's timing half refuses
#    above 2.0 s at the start, and a closing canary above 2.0 s makes the run FAILED (PROJECT_PLAN §7's canary rule).
#    Both modes refuse, before any interpreter starts, if COVERAGE_PROCESS_START, COVERAGE_PROCESS_CONFIG,
#    PYTHONTRACEMALLOC, PYTHONDEVMODE, PYTHONMALLOC or PYTHONPROFILEIMPORTTIME is set (B5.3). ${PIPESTATUS[0]} is the
#    driver's status; tee's -i ignores the interrupt, so Ctrl-C's lines reach the capture.
#
# 1. WHAT IT PRODUCES (the run)
#      output/p8_2/latency/<UTC>/<row>_<device>.json   one p8.2-latency/1.1 record per (row, device), written once
#      output/p8_2/latency/<UTC>/canary_{open,close}.json, run.json, logs/, partial/, and COMPLETE or FAILED
#      output/SHA256SUMS_p8_2.txt                       over output/p8_2/latency/, written once on COMPLETE, re-verified
#    Nothing under docs/, the corpus, the draws, another campaign's directory or any worktree.
#
# 2. HANGS (DEFERRED 104, Amendment A3.1): CityFlow's engine destructor can hang forever at an env's close. Each row
#    process keeps its three envs open until its record is written; the run supervises every process (rows and canaries)
#    under the pre-flight's timeout, kills a hung attempt's process group and re-runs it, at most three attempts; a record
#    written before a hang is kept; a process that exits without its record FAILS the run, naming the row. Nothing to do
#    by hand: a hang costs one timeout.
#
# 3. RESTART: a FAILED run leaves its directory as it is (it is a record); start again with a new token. A COMPLETE run
#    is final: the manifest exists and the driver refuses.

set -euo pipefail

MAIN=/home/filip/rltraffic
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
CORPUS=$MAIN/datasets_v11
DRAWS=$MAIN/scenarios/draws
RUNS=$OUTPUT/p8_2_runs
TOKEN=$RUNS/TOKEN_latency
MANIFEST=$OUTPUT/SHA256SUMS_p8_2.txt

# The G1 pre-flight whose timeouts the run uses, relative to $OUTPUT, and its sha256 (UNSET refuses the run).
# Pinned at C3c: the pre-flight of 2026-10-03 16:58 UTC at d33254a (COMPLETE; canaries 0.64 / 0.76 s at speed,
# reproduced; process wall times 5.4 / 7.9 s hz1x1 cpu / cuda, 34.3 / 12.6 s grid4x4 cpu / cuda) gives 120 s for
# hz1x1, grid4x4 and the canary (the floor; 3 x 34.3 s = 103 s) and an estimated run of about 19.0 min (an estimate,
# not a bound: Amendment B, B6.5). Its directory and capture are listed in
# output/p8_2_runs/SHA256SUMS_preflight_20261003T165853Z.txt. Amendment B (B1.4) keeps it pinned and not re-run.
PREFLIGHT_RECORD=p8_2_runs/preflight_20261003T165853Z/preflight.json
PREFLIGHT_SHA256=1abf1120fbc1dcba846a4e31de4a1526812fe058381fca76a05b321232e27868

WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=$WORK_TREE/docs/data

refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 2
}

MODE=run
if [ "${1:-}" = "--preflight" ]; then
  MODE=preflight
fi

# The regime, before any interpreter starts: one thread for OMP and MKL, no cuBLAS workspace setting.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset CUBLAS_WORKSPACE_CONFIG
# Amendment B, B5.3: every measured interpreter runs untraced, unprofiled and on the default allocator.
for variable in COVERAGE_PROCESS_START COVERAGE_PROCESS_CONFIG PYTHONTRACEMALLOC PYTHONDEVMODE PYTHONMALLOC PYTHONPROFILEIMPORTTIME; do
  [ -z "${!variable+set}" ] || refuse "$variable is set: a tracer, profiler or debug allocator would slow every measured process"
done

[ -x "$PY" ] || refuse "no interpreter at $PY"
[ -z "$(git -C "$WORK_TREE" status --porcelain --untracked-files=no)" ] || refuse "$WORK_TREE has uncommitted changes"
# An INTERPRETER running the module (`<python> -P -m offline.compute_latency ...`, how every child starts), anchored
# at the command line's start, so a shell or an editor that merely mentions the module is not taken for a live run.
if pgrep -f '^[^ ]*python[^ ]* -P -m offline[.]compute_latency' >/dev/null; then
  refuse "another offline.compute_latency process is running"
fi
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.compute_latency as m; print(m.__file__)')
[ "$LOADED" = "$WORK_TREE/offline/compute_latency.py" ] || refuse "the module loads from $LOADED, not from $WORK_TREE"

cd "$MAIN"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

if [ "$MODE" = preflight ]; then
  mkdir -p "$RUNS"
  echo "preflight $STAMP from $WORK_TREE at $(git -C "$WORK_TREE" rev-parse HEAD)"
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.compute_latency preflight --stamp "$STAMP" --python "$PY" --work-tree "$WORK_TREE" --output-root "$OUTPUT" --corpus-root "$CORPUS" --draws-root "$DRAWS" --data-dir "$DATA"
  exit $?
fi

# The run: every refusal below precedes the token.
COMMIT=${1:?usage: p8_2_latency.sh <commit> | --preflight}
[ "$(git -C "$WORK_TREE" rev-parse HEAD)" = "$(git -C "$WORK_TREE" rev-parse --verify --quiet "$COMMIT^{commit}" || true)" ] || refuse "$WORK_TREE is not at $COMMIT"
if git -C "$WORK_TREE" symbolic-ref -q HEAD >/dev/null; then
  refuse "$WORK_TREE is on a branch; the run tree is a DETACHED worktree at the reviewed commit"
fi
[ "$PREFLIGHT_SHA256" != UNSET ] || refuse "no G1 pre-flight is pinned (PREFLIGHT_SHA256 is UNSET)"
[ "$(sha256sum "$OUTPUT/$PREFLIGHT_RECORD" 2>/dev/null | cut -d' ' -f1)" = "$PREFLIGHT_SHA256" ] || refuse "$OUTPUT/$PREFLIGHT_RECORD is absent or not at its pinned digest $PREFLIGHT_SHA256"
TIMEOUTS=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.compute_latency timeouts --preflight-record "$OUTPUT/$PREFLIGHT_RECORD") || refuse "the pinned pre-flight record yields no timeouts"
read -r T_HZ T_GRID T_CANARY <<<"$TIMEOUTS"
[ ! -e "$MANIFEST" ] || refuse "$MANIFEST exists: a completed timing run is final"
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.compute_latency power-check || refuse "the power regime is not mains + Windows power mode Best Performance (Amendment B, B1): set it, then start again"
[ -f "$TOKEN" ] || refuse "no token at $TOKEN (the author's, after Amendment B)"

rm -- "$TOKEN"
echo "token consumed; run $STAMP from $WORK_TREE at $COMMIT; timeouts hz1x1 ${T_HZ} s, grid4x4 ${T_GRID} s, canary ${T_CANARY} s"
set +e
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.compute_latency run-all --stamp "$STAMP" --python "$PY" --work-tree "$WORK_TREE" --output-root "$OUTPUT" --corpus-root "$CORPUS" --draws-root "$DRAWS" --data-dir "$DATA" --timeout-hz1x1 "$T_HZ" --timeout-grid4x4 "$T_GRID" --canary-timeout "$T_CANARY"
CODE=$?
set -e
echo "run $STAMP: driver exit $CODE"
exit "$CODE"
