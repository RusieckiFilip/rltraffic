#!/usr/bin/env bash
# P5.2b -- the IQL random-tier CORRECTION RUN (gate G2) and its PRE-FLIGHT (gate G1): five IQL seeds re-trained on the
# random tier's declared 200 episodes, then evaluated by P5.2's own path and by P8.4b's cell runner on the same 500
# episodes (BRIEF_44 §3-§4 and its Amendment A; docs/plans/p5.2b.md §8). One process per stage, in sequence, under ONE
# token. No episode's outcome is printed.
#
# 0. USAGE
#    The PRE-FLIGHT (G1, the implementer), from the committed, clean task worktree; no token:
#        mkdir -p /home/filip/rltraffic/output/p5_2b_runs
#        bash offline/campaigns/p5_2b_correction.sh --preflight 2>&1 | tee -i -a /home/filip/rltraffic/output/p5_2b_runs/preflight_capture.txt
#    It writes output/p5_2b_runs/preflight_<UTC>/: the two canaries around a 2,000-step training of seed 101 on the
#    corrected table, T-reproduce (a) and (b) RUN on the ORIGINAL seed-101 checkpoint (draws 1000-1004) with their
#    outcomes beside the timings, the run's estimate and the stage timeouts (preflight.json, whose sha256 the module
#    prints). A T-reproduce failure makes the record FAILED and stops the task before any token (Amendment A, A3.1). The
#    record G1 accepts is then pinned below (PREFLIGHT_RECORD and PREFLIGHT_SHA256): the run reads its timeouts from it
#    and refuses while it is UNSET or at another digest.
#
#    The RUN (G2, the author), the FOREGROUND form, from the DETACHED RUN WORKTREE the coordinator creates at the
#    reviewed, pushed commit (a tree at any other commit, on a branch or with uncommitted changes is refused):
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p52b-run <commit>
#      Step 1, open a pane:      mkdir -p /home/filip/rltraffic/output/p5_2b_runs && tmux new -s p52b
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p52b-run/offline/campaigns/p5_2b_correction.sh <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p5_2b_runs/correction_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
#    The token is output/p5_2b_runs/TOKEN_correction, created by the author when the brief names the run commit (touch it).
#    Start on a quiet machine, on mains power with the Windows power mode set to Best Performance: the run calls P8.2's
#    power-check among its checks and refuses before the token otherwise; the opening canary (P8.2's, which re-checks the
#    power regime) must be at speed before the training starts, and the closing one is recorded beside the seconds
#    (PROJECT_PLAN §7's canary rule). Both modes refuse, before any interpreter starts, if COVERAGE_PROCESS_START,
#    COVERAGE_PROCESS_CONFIG, PYTHONTRACEMALLOC, PYTHONDEVMODE, PYTHONMALLOC or PYTHONPROFILEIMPORTTIME is set.
#    ${PIPESTATUS[0]} is the driver's status; tee's -i ignores the interrupt, so Ctrl-C's lines reach the capture.
#
# 1. WHAT IT PRODUCES (the run)
#      output/p5_2b/p5_2/checkpoints/grid4x4_random_iql_seed<s>.pt   the corrected checkpoints (Amendment A, A1.1)
#      output/p5_2b/training_random_iql.json, canary_open.json, canary_close.json
#      output/p5_2b/training_random_iql.late_close.json            only if the closing canary was taken late (see 3.)
#      output/p5_2b/eval_random_iql.json                         (i), P5.2's evaluate subcommand
#      output/p5_2b/rederivation/                                (ii), P8.4b's cell runner
#      output/p5_2b/logs/<stage>.attempt<n>.log
#      output/SHA256SUMS_p5_2b.txt                               every run file but artifacts/, written once
#      output/p5_2b/artifacts/p5_2b_correction.json              the artifact, built after the run's file list
#    Nothing under output/p5_2/, output/p8_4b_rederivation/, docs/, the corpus, the draws or any worktree.
#
# 2. HANGS (DEFERRED 104): CityFlow's engine destructor can hang at an env's close. Each evaluation stage runs under the
#    pinned pre-flight's timeout, at most three attempts: (i) restarts from scratch (P5.2's writer writes once, at its
#    end), (ii) resumes per cell. timeout -k kills a hung attempt.
#
# 3. RESTART: a FAILED run leaves output/p5_2b/ as it is (it is a record). Start again with a new token: a stage already
#    complete on disk is skipped and (ii) resumes per cell. A training complete on disk is never trained again (one
#    realisation: Amendment A, Q12; BRIEF_44 Amendment B, B1.3): if its closing canary failed, or the run stopped before
#    it, the restart runs close-late -- the training record's write-once addendum saying its seconds are bracketed by the
#    opening canary only -- and then takes the closing canary alone, late. A PARTIAL training -- a checkpoint or a
#    .partial without the full record, or a canary without its complete training -- is refused before the token:
#    move output/p5_2b aside by hand first (mv output/p5_2b output/p5_2b.failed_<UTC>); nothing in this run deletes. The
#    file list is written only when every stage is complete, and a run whose file list is written is final.

set -euo pipefail

MAIN=/home/filip/rltraffic
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
CORPUS=$MAIN/datasets_v11
DRAWS=$MAIN/scenarios/draws
RUN=$OUTPUT/p5_2b
RUNS=$OUTPUT/p5_2b_runs
TOKEN=$RUNS/TOKEN_correction
MANIFEST=$OUTPUT/SHA256SUMS_p5_2b.txt
CANARY_TIMEOUT=120
MAX_ATTEMPTS=3

# The G1 pre-flight whose timeouts the run uses, relative to $OUTPUT, and its sha256 (UNSET refuses the run): the record
# gate G1 read from disk and kept pinned (BRIEF_44 Amendment B, B0 and B2).
PREFLIGHT_RECORD=p5_2b_runs/preflight_20261006T195405Z/preflight.json
PREFLIGHT_SHA256=a280452734494479f6d2941b09ff12c16825155080404ee7ce84b95752c4bb4b

WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 2
}

fail() {
  printf 'FAILED: %s\n' "$*" >&2
  exit 1
}

MODE=run
if [ "${1:-}" = "--preflight" ]; then
  MODE=preflight
fi

# The regime, before any interpreter starts: one thread for OMP and MKL, no cuBLAS workspace setting (P5.2's default
# regime, the one the original cell trained in: Amendment A, Q12), no tracer, profiler or debug allocator.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset CUBLAS_WORKSPACE_CONFIG
for variable in COVERAGE_PROCESS_START COVERAGE_PROCESS_CONFIG PYTHONTRACEMALLOC PYTHONDEVMODE PYTHONMALLOC PYTHONPROFILEIMPORTTIME; do
  [ -z "${!variable+set}" ] || refuse "$variable is set: a tracer, profiler or debug allocator would slow every timed process"
done

[ -x "$PY" ] || refuse "no interpreter at $PY"
[ -z "$(git -C "$WORK_TREE" status --porcelain --untracked-files=no)" ] || refuse "$WORK_TREE has uncommitted changes"
# An INTERPRETER running the module (`<python> -P -m offline.iql_correction ...`, how every stage starts), anchored at
# the command line's start, so a shell or an editor that merely mentions the module is not taken for a live run.
if pgrep -f '^[^ ]*python[^ ]* -P -m offline[.]iql_correction' >/dev/null; then
  refuse "another offline.iql_correction process is running"
fi
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.iql_correction as m; print(m.__file__)')
[ "$LOADED" = "$WORK_TREE/offline/iql_correction.py" ] || refuse "the module loads from $LOADED, not from $WORK_TREE"

ROOTS=(--output-root "$OUTPUT" --corpus-root "$CORPUS" --draws-root "$DRAWS" --repo-root "$WORK_TREE")
cd "$MAIN"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# One module stage in this shell: -P, the run tree on PYTHONPATH, the cwd the main tree.
ic() {
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.iql_correction "$@"
}

if [ "$MODE" = preflight ]; then
  PRE=$RUNS/preflight_$STAMP
  mkdir -p "$PRE"
  echo "pre-flight $STAMP from $WORK_TREE at $(git -C "$WORK_TREE" rev-parse HEAD)"
  CODE=0
  timeout -k 30 "$CANARY_TIMEOUT" env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.compute_latency canary --phase open --out-dir "$PRE" || CODE=$?
  [ "$CODE" -ne 0 ] || ic preflight "${ROOTS[@]}" --out-root "$PRE" --stamp "$STAMP" || CODE=$?
  timeout -k 30 "$CANARY_TIMEOUT" env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.compute_latency canary --phase close --out-dir "$PRE" || CODE=$?
  echo "pre-flight $STAMP: driver exit $CODE ($PRE)"
  exit "$CODE"
fi

# The run: every refusal below precedes the token.
COMMIT=${1:?usage: p5_2b_correction.sh <commit> | --preflight}
[ "$(git -C "$WORK_TREE" rev-parse HEAD)" = "$(git -C "$WORK_TREE" rev-parse --verify --quiet "$COMMIT^{commit}" || true)" ] || refuse "$WORK_TREE is not at $COMMIT"
if git -C "$WORK_TREE" symbolic-ref -q HEAD >/dev/null; then
  refuse "$WORK_TREE is on a branch; the run tree is a DETACHED worktree at the reviewed commit"
fi
[ "$PREFLIGHT_SHA256" != UNSET ] || refuse "no G1 pre-flight is pinned (PREFLIGHT_SHA256 is UNSET)"
[ "$(sha256sum "$OUTPUT/$PREFLIGHT_RECORD" 2>/dev/null | cut -d' ' -f1)" = "$PREFLIGHT_SHA256" ] || refuse "$OUTPUT/$PREFLIGHT_RECORD is absent or not at its pinned digest $PREFLIGHT_SHA256"
TIMEOUTS=$(ic timeouts --preflight-record "$OUTPUT/$PREFLIGHT_RECORD") || refuse "the pinned pre-flight record yields no stage timeouts"
read -r T_TRAIN T_EVAL_I T_EVAL_II <<<"$TIMEOUTS"
[ ! -e "$MANIFEST" ] || refuse "$MANIFEST exists: a completed run is final"
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.compute_latency power-check || refuse "the power regime is not mains + Windows power mode Best Performance: set it, then start again"
ic check "${ROOTS[@]}" --out-root "$RUN" || refuse "the module's pre-run check refused (its reasons are above)"
[ -f "$TOKEN" ] || refuse "no token at $TOKEN (the author's, after the brief names the run commit)"

rm -- "$TOKEN"
echo "token consumed; run $STAMP from $WORK_TREE at $COMMIT"
echo "stage timeouts in seconds: ${T_TRAIN} (training), ${T_EVAL_I} (i), ${T_EVAL_II} (ii); canary ${CANARY_TIMEOUT}"
mkdir -p "$RUN/logs"

# attempt <label> <seconds> <attempts> <command...>: the command under its timeout, again on a failure or a hang.
attempt() {
  local label=$1 seconds=$2 max=$3
  shift 3
  local n code
  for n in $(seq 1 "$max"); do
    code=0
    echo "$label: try $n of $max, timeout ${seconds} s"
    timeout -k 30 "$seconds" "$@" >>"$RUN/logs/$label.attempt$n.log" 2>&1 || code=$?
    [ "$code" -ne 0 ] || { echo "$label: try $n ok"; return 0; }
    echo "$label: try $n FAILED (exit $code; 124 or 137 is the timeout); the last lines of its log:"
    tail -15 "$RUN/logs/$label.attempt$n.log" | sed 's/^/    /'
  done
  return 1
}

if [ "$(ic status "${ROOTS[@]}" --out-root "$RUN" --stage training)" != complete ]; then
  attempt canary_open "$CANARY_TIMEOUT" 1 env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.compute_latency canary --phase open --out-dir "$RUN" || fail "the opening canary refused or failed: nothing was trained"
  attempt training "$T_TRAIN" 1 env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.iql_correction train "${ROOTS[@]}" --out-root "$RUN" || fail "the training stage failed: its partial files are a record, move output/p5_2b aside by hand"
  attempt canary_close "$CANARY_TIMEOUT" 1 env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.compute_latency canary --phase close --out-dir "$RUN" || fail "the closing canary failed after a complete training: start again with a new token; the closing canary is then taken alone, late, and nothing is trained again"
elif [ "$(ic status "${ROOTS[@]}" --out-root "$RUN" --stage canaries)" = closing_pending ]; then
  echo "the training is complete on disk and its closing canary is missing: it is taken now, alone and late; nothing is trained again"
  ic close-late "${ROOTS[@]}" --out-root "$RUN" || fail "the late closing canary's mark was not written"
  attempt canary_close_late "$CANARY_TIMEOUT" 1 env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.compute_latency canary --phase close --out-dir "$RUN" || fail "the late closing canary failed: start again with a new token; nothing is trained again"
else
  echo "SKIP: the training stage and its two canaries are complete on disk"
fi
attempt evaluate_p5_2 "$T_EVAL_I" "$MAX_ATTEMPTS" env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.iql_correction evaluate-p5-2 "${ROOTS[@]}" --out-root "$RUN" || fail "(i) failed on every attempt"
attempt evaluate_p8_4b "$T_EVAL_II" "$MAX_ATTEMPTS" env PYTHONPATH="$WORK_TREE" "$PY" -P -m offline.iql_correction evaluate-p8-4b "${ROOTS[@]}" --out-root "$RUN" || fail "(ii) failed on every attempt"
ic manifest "${ROOTS[@]}" --out-root "$RUN" || fail "the run's file list was not written"
ic report "${ROOTS[@]}" --out-root "$RUN" || fail "the artifact was not built (the run's records are complete; rebuild it with the same command)"
echo "run $STAMP: COMPLETE"
