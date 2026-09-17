#!/usr/bin/env bash
# P7.3a — the zero-shot point of the C3 transfer curve (H3's confirmatory test).
#
# Shape: offline/campaigns/p7_2b_calibration.sh, whose own header credits p7_1_metric_freeze.sh.
#
# 0. USAGE — the STAGE is an ARGUMENT, and that is Amendment B1's whole point.
#      bash offline/campaigns/p7_3a_zero_shot.sh confirmatory
#      bash offline/campaigns/p7_3a_zero_shot.sh rest
#    ⚠️ STAGE 2 IS UNCONDITIONAL. It runs whatever stage 1 showed, exactly as P7.3b runs whatever
#    the zero-shot number is (BRIEF_37 section 7). The stage is an argument precisely so that no
#    decision the driver makes can depend on a number it has just seen. Each stage needs its OWN
#    token and re-runs the canary at its start.
#
# 1. WHAT IT PRODUCES
#    datasets_sumo_v11/hz1x1_sumo_maxpressure/   the v1.1 SUMO corpus, draws 201-300 (stage
#                                                'confirmatory' only; P7.3b's few-shot source)
#    output/p7_3a/cell_*.json                    one chunk per cell, atomic and resumable
#    output/p7_3a/failed/                        chunks that failed their own re-validation
#    docs/data/p7_3a_zero_shot_stage1.json       Amendment B2's confirmatory artifact
#    docs/data/p7_3a_zero_shot.json              the final artifact, citing the stage-1 sha256
#    output/SHA256SUMS_p7_3a.txt                 written last, atomically, then re-verified
#
# 2. ORDERING — every check that can refuse PRECEDES the token, so a refused start consumes
#    nothing and changes nothing at all. Order: interpreter → import → lock → GROUP LEADER →
#    stage argument → inputs → CANARY → TRAP → token → work. The canary is inside that fence
#    deliberately: a throttled machine must not burn the author's one-shot authorisation.
#    ⚠️ THE TRAP IS INSTALLED BEFORE THE TOKEN IS CONSUMED. It used to be installed after in
#    P7.2b, and a signal in that window destroyed the authorisation while leaving neither FAILED
#    nor COMPLETE — the operator could not tell whether the run had started.
#
# 3. A17(f) RUNS BEFORE ANY EVALUATION CELL, AND STOPS THE DRIVER.
#    The collection stage is ~20 minutes and it is where a wiring defect surfaces; the evaluation
#    pool is hours. 100/100 or P7.3 stops (PREREGISTRATION A17(f), BRIEF_37 section 3.2). The
#    coordinator reads the gate's output from disk before stage 2's token is written.
#
# 4. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL.
#    offline/transfer_curve.py::chunk_is_reusable re-derives a chunk's verdict from its own content
#    AND from the files on disk — the checkpoint's digest, the .sumocfg's and the routes file's —
#    because the stored verdict is exactly what a half-written chunk would lie about.
#    ⛔ THERE ARE DELIBERATELY NO `[ -f ]` GUARDS BELOW. offline/campaigns/p5_3b.sh had one and a
#    bad chunk then survived every restart until someone deleted it by hand.
#
# 5. NOTHING UNDER scenarios/draws/ IS WRITTEN. The parity configs are P7.2a's output and this
#    campaign opens them read-only.
#
# 6. TIME — from F1's PRE-FLIGHT PILOT, run through this runner at this worker count.
#    Pilot: `python -m offline.transfer_curve … pilot --workers 12`, 16 DT cells on draw 5 (the
#    FENCED smoke draw), both subjects, all four declared arms, seeds 101/202, halting OFF (draw 5
#    is not Amendment C2's declared subset). Transcript: output/p7_3a_runs/preflight_pilot.txt.
#    Date 2026-09-17. n = 2 runs x 16 cells. A11 labels every figure: measured on the thermally
#    constrained machine, no cooling pad.
#      run 1   canary 0.92 s   wall 32.05 s   2.003 s/cell effective   in-process mean 14.34 s
#      run 2   canary 0.77 s   wall 32.55 s   2.035 s/cell effective   in-process mean 14.86 s
#      speed-up 7.16x / 7.30x on 16 cores;  0 failures in 32 cells
#    SCHEDULE, stated with what it does and does not cover:
#      stage 1 (1,200 cells) ≈ 40 min   ·   full campaign (4,700 cells) ≈ 2.6 h
#    ⚠️ THAT PROJECTION IS MEASURED ON **DT CELLS ONLY**. The campaign's 700 anchor cells
#    (fixedtime 100, maxpressure 100, random 5x100) were NOT run by this pilot; A6's in-process
#    anchor times with halting ON (fixedtime 38.65 s, maxpressure 44.66 s, random 51.24 s, n = 2
#    each) sit ABOVE the DT cell's 31.64 s, so those 700 cells may be slower and the figures above
#    are a lower bound for them. The 47 cells carrying C2's halting check cost about 3.5x (A9b),
#    which adds roughly 4 min.
#    This replaces Amendment C7's reading (stage 1 ≈ 1 h, full ≈ 3–3.5 h), which came from the A6
#    HARNESS pilot at 5.393 s/cell with the halting check ON for every cell; under C2 it is ON for
#    47 of 4,700, which is most of the difference.
#    ⚠️ F3: if the cooling pad's canary differs from 0.92 s / 0.77 s by more than 10 %, the pilot is
#    re-run on the pad before the token and this block is replaced by that run's rate — the header
#    must never quote a rate for a machine state the run did not have.
#
# 7. THE WORKTREE HAS NO .venv. The interpreter is the main tree's, as P7.2b's driver does.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p7_3a
LOGS=$WORK/logs
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
CORPUS=$MAIN/datasets_sumo_v11/hz1x1_sumo_maxpressure
# Amendment C3: measured 1.27x better than 8 on this machine's 16 cores. A variable, so an
# operator on a smaller machine lowers it without editing the campaign's logic.
WORKERS=12
CANARY_MAX_SECONDS=2.0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ⚠️ INPUTS come from the run worktree's COMMITTED docs/data; OUTPUTS go to $WORK/artifacts, which
# is inside the manifest and outside the tree (Finding 2, the author's ruling of 2026-09-17).
# Measured by the coordinator: writing the stage-1 artifact into $WORK_TREE/docs/data -- where it is
# NOT gitignored -- left `?? docs/data/p7_3a_zero_shot_stage1.json` in the tree, so J1(d)'s own
# dirty-tree refusal then blocked stage 2 before its token. The artifacts are copied into the task
# branch BY HAND afterwards (see the NEXT block), which is also when they are reviewed.
ARTIFACTS=$WORK/artifacts

COMMON=(--draws-root "$DRAWS" --output-root "$MAIN/output" --work-dir "$WORK"
        --data-dir "$DATA" --out-dir "$ARTIFACTS")

# ---------------------------------------------------------------- preconditions
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  echo "  The worktree has no .venv of its own; the main tree's is the one to use." >&2
  exit 2
fi

cd "$WORK_TREE"
if ! PYTHONPATH=$WORK_TREE "$PY" -P -c "import offline.transfer_curve" 2>/dev/null; then
  echo "REFUSING TO START: offline.transfer_curve does not import from $WORK_TREE" >&2
  exit 2
fi

STAGE=${1:-}
case "$STAGE" in
  confirmatory|rest) ;;
  *)
    echo "REFUSING TO START: the stage must be 'confirmatory' or 'rest', got '${STAGE}'" >&2
    echo "  Amendment B1 declares both before any cell runs, and stage 2 is UNCONDITIONAL." >&2
    exit 2
    ;;
esac

# ⚠️ The pattern requires `python` before the module name, and this process and its parent are
# excluded: a bare match also matches the SHELL running this script, because the module name
# appears in its own command line, and the driver would refuse to start because of itself.
ALIVE=$(pgrep -f 'python.*offline\.transfer_curve' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: cells from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo '  Wait until `pgrep -f transfer_curve` is empty. Nothing has been consumed.' >&2
  exit 3
fi

# `kill -- -$$` in the signal handler is a NO-OP unless this script leads its own process group.
# ⚠️ CORRECTED (Amendment J3, to what the reviewer MEASURED). The earlier sentence here claimed a
# backgrounded script does not lead its group. Measured: a plain `&` from a non-interactive shell
# DID lead its own group, and what produced a non-leader was `set +m` -- job control switched off.
# The check below is unchanged; only the sentence explaining it was wrong.
if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux foreground pane" >&2
  echo "  The handler kills the process group, and that is a no-op from a non-leader, so an" >&2
  echo "  interrupted run would leave the worker pool writing. Nothing consumed." >&2
  exit 2
fi

# Amendment J3 (reviewer min-2): a shell that STARTS with SIGINT ignored cannot trap it -- bash
# does not let a non-interactive shell trap a signal that was ignored on entry -- so Ctrl-C would
# do nothing at all while the pool kept writing and the token was already gone. Signal 2's bit in
# the SigIgn mask is 0x2. Fail closed, like the group-leader check above.
SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  echo "REFUSING TO START: SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" >&2
  echo "  bash cannot trap a signal that was ignored on entry, so the trap below would be a" >&2
  echo "  no-op and Ctrl-C could not stop an interrupted run. Start the driver from a shell" >&2
  echo "  that does not ignore SIGINT -- a tmux foreground pane does not. Nothing consumed." >&2
  exit 2
fi

# Amendment J1(d): a multi-hour stage must not run from a tree that is being edited. ONE untracked
# file makes every chunk rolled after it `git_dirty: true`, and validate_cell_payload then refuses
# those cells one at a time, hours in. J1(e): the campaign runs from a dedicated worktree
# (/home/filip/rltraffic-p73a-run) detached at the reviewed commit, which no session edits.
WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "REFUSING TO START: the worktree $WORK_TREE is DIRTY" >&2
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  echo "  Every cell rolled from this tree would record git_dirty: true, which report refuses" >&2
  echo "  one cell at a time. Run the campaign from the dedicated worktree detached at the" >&2
  echo "  reviewed commit (Amendment J1(e)). Nothing has been consumed." >&2
  exit 2
fi

# The pool's ends and the corpus band's ends. Not every draw: 100 stat() calls before a token is
# noise, and a missing middle draw refuses inside the cell that needs it, naming it.
for draw in 201 300 1000 1099; do
  cfg=$DRAWS/cityflow1x1/draw_$(printf '%04d' "$draw")/parity/noteleport.sumocfg
  if [ ! -f "$cfg" ]; then
    echo "REFUSING TO START: P7.2a's parity configuration is missing: $cfg" >&2
    exit 2
  fi
done
for seed in 101 202 303 404 505; do
  for ckpt in "$MAIN/output/p4_dt/dt_seed${seed}.pt" \
              "$MAIN/output/p4_7/checkpoints/mix50_dt_seed${seed}.pt"; do
    if [ ! -f "$ckpt" ]; then
      echo "REFUSING TO START: a registered subject's checkpoint is missing: $ckpt" >&2
      exit 2
    fi
  done
done

# ---------------------------------------------------------------- the canary
# PROJECT_PLAN section 7's rule; recipe in BRIEF_36 section 3.3. A guest that reports a low load
# can still be running on a throttled host, so the rate basis is measured in THIS session or not
# written down. BOTH halves: the timing says the machine is at speed, and check_canary says the
# ENGINE still computes what it computed when the references were measured.
# ⚠️ THE `-a` IS LOAD-BEARING (Amendment E1.3 item 3). Plain `tee /dev/stderr` re-opens stderr with
# O_TRUNC, so when an operator runs this driver with `>> log 2>&1` the open resets the file offset
# and destroys everything already written. P7.2b lost run 3's capture exactly that way.
echo "=== canary (PROJECT_PLAN section 7; recipe BRIEF_36 section 3.3)"
# ⚠️ Amendment J3 (reviewer min-4): the failure is CAUGHT here and gets its own refusal, because
# `fail()` exits 1 and writes FAILED into the work directory -- an end state that says a run
# started. A canary that refuses has consumed nothing, and says so, and exits 2 like every other
# pre-token refusal. `set -euo pipefail` is what makes the pipeline's failure visible at all.
if ! CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" canary | tee -a /dev/stderr); then
  echo "REFUSING TO START: the canary FAILED -- its timing or its correctness half" >&2
  echo "  The observed line is above: the canary prints before it checks, so the values are" >&2
  echo "  visible even when the check refuses. A correctness failure means the ENGINE did not" >&2
  echo "  reproduce draw 0, which is a finding about the engine and not a rate question." >&2
  echo "  Nothing has been consumed." >&2
  exit 2
fi
echo "$CANARY_LINE"
CANARY=$(echo "$CANARY_LINE" | awk '{print $2}')
if awk -v c="$CANARY" -v m="$CANARY_MAX_SECONDS" 'BEGIN { exit !(c > m) }'; then
  echo "REFUSING TO START: canary $CANARY s exceeds $CANARY_MAX_SECONDS s -- the machine is" >&2
  echo "  throttled and no rate measured here would be a rate. Check mains power and the" >&2
  echo "  cooling pad, then start again. Nothing has been consumed." >&2
  exit 2
fi

# ---------------------------------------------------------------- the trap, BEFORE the token
fail() {
  local where=$1
  if [ -d "$WORK" ]; then
    echo "CAMPAIGN FAILED at $where" | tee "$WORK/FAILED"
  else
    echo "CAMPAIGN FAILED at $where (before any work directory existed)"
  fi
  exit 1
}

# The pid of the stage currently in flight. Global, so the signal handler can reach it: bash traps
# do not see a caller's locals.
CELL_PID=""

on_signal() {
  trap '' INT TERM
  if [ -d "$WORK" ]; then
    echo "CAMPAIGN INTERRUPTED by a signal" | tee "$WORK/FAILED" >&2
  else
    echo "CAMPAIGN INTERRUPTED by a signal (before any work directory existed)" >&2
  fi
  [ -z "$CELL_PID" ] || kill -TERM "$CELL_PID" 2>/dev/null || true
  sleep 2
  [ -z "$CELL_PID" ] || kill -KILL "$CELL_PID" 2>/dev/null || true
  kill -- -$$ 2>/dev/null || true
  exit 130
}
# ⚠️ HUP as well as INT and TERM (Amendment J2). A closed tmux pane or a dropped SSH session sends
# exactly SIGHUP, and it was not trapped: measured, a HUP during A17(f) exited 129 with the token
# consumed, canary.json and the logs written, and NEITHER FAILED NOR COMPLETE -- the end state the
# trap-before-token rule exists to prevent, reached by the one signal the trap did not name.
trap on_signal INT TERM HUP

# ---------------------------------------------------------------- the token
TOKEN=$WORK/AUTHORISED_TO_RUN
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run authorisation token at $TOKEN" >&2
  echo "  The author authorises one run with:" >&2
  echo "    mkdir -p $WORK && date -Is > $TOKEN" >&2
  echo "  The token is deleted on start, so it authorises exactly one run -- and each of the" >&2
  echo "  two declared stages needs its own." >&2
  exit 2
fi
echo "=== authorised by token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
rm -f "$TOKEN"
echo "=== token consumed and deleted; a restart needs a new one"

# Only NOW may anything be created or cleared.
mkdir -p "$LOGS"
mkdir -p "$ARTIFACTS"
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

# Amendment E1.2: the canary's observed values reach a MANIFESTED file, not only the pane.
# Appended, never truncated, like the stage logs -- and written here, AFTER the token, so
# "a refused start creates nothing" still holds.
echo "=== canary  stage $STAGE  run at $(date -Is)" >> "$LOGS/canary.log"
echo "$CANARY_LINE" >> "$LOGS/canary.log"

# Amendment E1.4: and into a MACHINE-READABLE record `report` reads back and re-checks, because a
# log is only ever read by a human. Without it `report` built the canary block from the CHUNKS --
# which a re-roll reuses unchanged -- so P7.2b's run 3 published run 1's canary as its own.
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve \
  "${COMMON[@]}" record-canary --line "$CANARY_LINE" || fail "record-canary"

START=$(date +%s)

# Amendment I3(3): when a stage has more than one path into it, the path taken is written as the
# FIRST line of that run's entry -- so the stage-1 checkpoint reads ONE file (logs/a17f.log) and
# still knows whether the corpus was collected by this run or accepted from a previous one.
STAGE_NOTE=""

run_stage() {
  local label=$1; shift
  local log=$LOGS/${label}.log
  echo "=== $label"
  # ⚠️ APPENDED, never truncated: a restart must not destroy the first run's narrative record.
  if [ -n "$STAGE_NOTE" ]; then echo "$STAGE_NOTE" >> "$log"; fi
  echo "=== $label  run at $(date -Is)" >> "$log"
  PYTHONPATH=$WORK_TREE "$PY" -P -m "$@" >> "$log" 2>&1 &
  CELL_PID=$!
  if ! wait "$CELL_PID"; then
    CELL_PID=""
    echo "--- tail of $log ---" >&2
    tail -5 "$log" >&2
    fail "$label"
  fi
  CELL_PID=""
  tail -1 "$log"
}

# ---------------------------------------------------------------- stage 1: collection + A17(f)
# Only in the confirmatory run: the corpus is collected once and A17(f) gates the whole of P7.3.
#
# ⚠️ AMENDMENT H2 — THE STAGE IS RESTARTABLE, AND THAT IS WHY IT BRANCHES.
# `offline.collect` refuses a populated --out-dir (trajectory_logger.py:450), so after ANY failure
# past this point -- the pool, the report, the manifest -- a restart used to die here, on the one
# stage that had already succeeded. If the corpus is already on disk we do NOT collect again; we
# run A17(f) over it, which is the only thing that decides whether it is usable.
#
# ⛔ --overwrite IS NEVER PASSED, and there is no branch below that deletes anything. A corpus
# A17(f) refuses is named draw by draw in its own message and is moved aside BY HAND: deleting a
# corpus the driver cannot prove is bad is not the driver's decision, and a silent re-collection
# would replace a corpus that a previous run's gate had already blessed.
#
# The collect argv is the one tests/test_transfer_calibration.py's T1/T2 fixture verified end to
# end -- same flags, same order -- so the corpus this driver writes is the corpus those tests
# checked. --flow-draws-range is half-open: 201 301 is draws 201..300.
if [ "$STAGE" = "confirmatory" ]; then
  if [ -d "$CORPUS" ]; then
    echo "=== corpus already at $CORPUS -- not collecting again; A17(f) decides whether it stands"
    STAGE_NOTE="branch: corpus ALREADY PRESENT at $CORPUS -- collection skipped by this run"
    run_stage a17f offline.transfer_curve "${COMMON[@]}" a17f --corpus-dir "$CORPUS"
    STAGE_NOTE=""
  else
    run_stage collect offline.collect \
      --backend sumo \
      --env-config "$WORK_TREE/configs/sim/cityflow1x1.json" \
      --policy maxpressure \
      --flow-draws-range 201 301 \
      --episodes 1 \
      --base-seed 1000 \
      --global-reward-weight 0.0 \
      --local-reward-fn queue_length \
      --out-dir "$CORPUS" \
      --draws-root "$DRAWS"

    STAGE_NOTE="branch: corpus COLLECTED by this run into $CORPUS"
    run_stage a17f offline.transfer_curve "${COMMON[@]}" a17f --corpus-dir "$CORPUS"
    STAGE_NOTE=""
  fi
fi

# ---------------------------------------------------------------- stage 2: the evaluation pool
run_stage "cells_$STAGE" offline.transfer_curve "${COMMON[@]}" \
  --canary-seconds "$CANARY" cells --stage "$STAGE" --workers "$WORKERS"

if [ "$STAGE" = "confirmatory" ]; then
  run_stage report_confirmatory offline.transfer_curve "${COMMON[@]}" \
    report --stage confirmatory
else
  run_stage report_final offline.transfer_curve "${COMMON[@]}" \
    report --stage1-path "$ARTIFACTS/p7_3a_zero_shot_stage1.json"
fi

# ---------------------------------------------------------------- manifest
( cd "$MAIN/output" \
  && find p7_3a -type f \( -name '*.json' -o -name '*.log' \) -print \
     | LC_ALL=C sort | xargs sha256sum > SHA256SUMS_p7_3a.txt.tmp \
  && mv SHA256SUMS_p7_3a.txt.tmp SHA256SUMS_p7_3a.txt ) || fail "manifest write"

( cd "$MAIN/output" && for manifest in SHA256SUMS_*.txt; do
    sha256sum -c "$manifest" >/dev/null || { echo "MANIFEST MISMATCH: $manifest" >&2; exit 1; }
  done ) || fail "manifest re-verification"

ELAPSED=$(( $(date +%s) - START ))
echo "CAMPAIGN COMPLETE ($STAGE) in ${ELAPSED}s ($((ELAPSED / 60)) min)" | tee "$WORK/COMPLETE"

cat <<'NEXT'
=== NEXT (manual)
  1. the coordinator reads A17(f)'s 100/100 line from disk BEFORE stage 2's token is written
  2. COPY THE ARTIFACTS INTO THE TASK BRANCH BY HAND, then commit them there:
       cp $WORK/artifacts/p7_3a_zero_shot_stage1.json <task worktree>/docs/data/
       cp $WORK/artifacts/p7_3a_zero_shot.json        <task worktree>/docs/data/
       git -C <task worktree> add docs/data/p7_3a_zero_shot*.json && git -C <task worktree> commit
     They are written to $WORK/artifacts and NOT into the run worktree, because an artifact
     written there is an untracked file that the dirty-tree refusal then blocks the next stage on
     (Finding 2). The copy is by hand and it is also when the artifact is read.
  3. the artifact REPORTS; it registers nothing, and it interprets nothing. H3's clauses are
     inequalities with values beside them; the contrast is exploratory; the gate to P7.3b is a
     property of the PIPELINE and never of the number (BRIEF_37 section 7).
NEXT
