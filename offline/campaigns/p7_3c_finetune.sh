#!/usr/bin/env bash
# P7.3c — C4: the TRAINING driver — A24(b)'s thirty fine-tunes, each written exactly once, and G5's FENCED timing
# (BRIEF_41 C3–C4; PREREGISTRATION A24(b), A18(d); Amendment A Q7, Q9, Q11, Q12, Q22).
#
# Shape: offline/campaigns/p7_3c_corpus.sh (C2) — the run worktree, the derived tree, the two-step foreground start,
# -P on every interpreter call, the liveness guard, the group-leader and SigIgn checks, every refusal before the token,
# the trap before the token, record-canary after it — with Amendment B's B3 fixes applied from the start: the run
# worktree AND its commit enforced (B3.1); FAILED on every path after the token, through an EXIT trap keyed on a
# success flag, written with printf before any echo, and `tee -i -a` in the documented start line (B3.4).
#
# 0. USAGE — both modes FOREGROUND from the DETACHED RUN WORKTREE (J1(e)), which the coordinator creates at the
#    reviewed, pushed commit; that commit is the second argument, and a tree at any other commit is refused:
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p73c-run <commit>
#
#      Step 1, open a pane:      tmux new -s p73c_finetune
#      Step 2 (timing), at ITS PROMPT:    bash /home/filip/rltraffic-p73c-run/offline/campaigns/p7_3c_finetune.sh timing <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p7_3c_runs/finetune_timing_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -i -a /home/filip/rltraffic/output/p7_3c_runs/finetune_timing_capture.txt
#      Step 2 (train), at ITS PROMPT:     bash /home/filip/rltraffic-p73c-run/offline/campaigns/p7_3c_finetune.sh train <commit> <timing-stamp> 2>&1 | tee -i -a /home/filip/rltraffic/output/p7_3c_runs/finetune_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -i -a /home/filip/rltraffic/output/p7_3c_runs/finetune_capture.txt
#
#    TIMING (gate G5, Amendment A Q22: the implementer starts this pane) takes NO token: a configuration A24 does not
#    register (k 5, seed 101, B 400), every write under output/p7_3c_training/fenced_timing/<UTC stamp>/, which the
#    evaluation path refuses.  TRAIN (gate G6) consumes the author's token, output/p7_3c_runs/TOKEN_finetune, and takes
#    G5's stamp: its concurrency is read from fenced_timing/<stamp>/timing.json and re-derived there by the rule.
#    Start on an idle machine, on mains power: the canary's timing half refuses above 2.0 s.
#
#    THE SECOND HALF OF STEP 2 (BRIEF_39 Amendment B.5-3): `${PIPESTATUS[0]}` is the DRIVER's status, so the pane and
#    the capture's last line agree.  `tee -i` ignores the interrupt, so Ctrl-C's lines reach the capture (B3.4).
#    ⛔ NOT `tmux new -s NAME '<cmd>'`: the script would not lead its own process group, and the guard refuses.
#
# 1. WHAT IT PRODUCES
#      output/p7_3c_training/checkpoints/<run>.pt        the thirty checkpoints, each written once (os.link)
#      output/p7_3c_training/runs/<run>.json             each run's seconds and losses, written once with it
#      output/p7_3c_training/attempts/<run>.<n>          one marker per start of a run, BEFORE it trains
#      output/p7_3c_training/starts/<UTC>/canary.json    this start's canary; COMPLETE or FAILED beside it
#      output/SHA256SUMS_p7_3c_finetune.txt              the manifest, sha256sum's format, re-verified
#      output/p7_3c_training/p7_3c_finetune.json         the record the coordinator commits on main at G7 (Q9)
#    Timing: fenced_timing/<UTC>/{alone,pair1,pair2,triple1,triple2,triple3,repeat}.{pt,json}, build_k100.json,
#    nvidia_smi_<phase>.csv, timing.json, COMPLETE or FAILED.  Nothing under docs/, datasets or any worktree.
#    The driver prints no evaluation outcome: none exists here, and the training losses stay in the run records.
#
# 2. RESUME, AND WHAT IS NEVER DONE.  Every decision is offline.few_shot's, never a `[ -f ]` here: a checkpoint that
#    exists AND validates is skipped; one that exists and does NOT validate refuses the start (resume-decision --all,
#    before the canary and the token) and is never overwritten; an absent one trains, after its attempt marker.  A
#    marker with no checkpoint after it is a re-run of an infrastructure failure (A24(b)), counted in the record.  The
#    only deletion is the consumed token; nothing is moved.
#
# 3. ORDERING — arguments → interpreter → not the implementer's tree → THE RUN TREE → ITS COMMIT → import (-P) from
#    that tree → no live runner → GROUP LEADER → SigIgn → dirty tree → check-inputs (the calibration artifact, the five
#    sources and the corpus's sums by digest, G3's gate record, CUDA; for train, G5's record and free device memory) →
#    [train: the concurrency, the resume scan] → canary, BOTH halves → TRAPS → [train: token] → the start directory →
#    record-canary → the runs → manifest (re-verified, 30 lines) → record, LAST → COMPLETE.
#
# 4. -P ON EVERY INTERPRETER CALL; the cwd is the MAIN tree, PYTHONPATH the tree this copy lives in, and the import
#    check asserts WHICH files loaded.  THE REGIME IS P5.2's (A24(b)): OMP and MKL at one thread and
#    CUBLAS_WORKSPACE_CONFIG UNSET, exactly as offline/campaigns/p5_2.sh lines 98-117 do outside their deterministic
#    regime; the CUDA commands refuse the variable set and pin one torch thread (P5.2's --torch-threads 1).
#
# 5. CONCURRENCY — MEASURED, NOT ASSUMED (G5): the C in {1, 2, 3} with the largest aggregate throughput whose device
#    peak is <= 80 % of 16,303 MiB, a tie to the smaller C, fixed before any measurement (plan section 8) and applied
#    in offline.few_shot.choose_concurrency.  NOT YET MEASURED at this commit: G5's numbers go here, into the plan
#    and into the packet, each with its canary and date.

set -euo pipefail

MODE=${1:-}
EXPECTED_COMMIT=${2:-}
TIMING_STAMP=${3:-}

MAIN=/home/filip/rltraffic
# J1(e) / BRIEF_39 Amendment B.7.1-1: the tree this copy of the script lives in, never a hardcoded one.
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
IMPLEMENTER_TREE=/home/filip/rltraffic-p73c
RUN_TREE=/home/filip/rltraffic-p73c-run
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
TRAINING=$OUTPUT/p7_3c_training
FENCED=$TRAINING/fenced_timing
CORPUS=$MAIN/datasets_sumo_v11/grid4x4_sumo_maxpressure
GATE_RECORD=$OUTPUT/p7_3c_corpus/a17f_gate.json
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
TOKEN=$OUTPUT/p7_3c_runs/TOKEN_finetune
CANARY_MAX_SECONDS=2.0
SAMPLE_MS=250
EXPECTED_MANIFEST_LINES=30

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# P5.2's non-deterministic regime UNSETS this rather than merely not setting it (offline/campaigns/p5_2.sh lines
# 101-117): it constrains cuBLAS's workspace and so its GEMM selection, and the launch shell may carry it.
unset CUBLAS_WORKSPACE_CONFIG

SUCCESS=0
MARKER_DIR=""
SAMPLER=""
PIDS=()
NAMES=()

echo "=== P7.3c fine-tune driver (${MODE:-no mode}): WORK_TREE $WORK_TREE (derived from this script's own location)"

refuse() {
  echo "REFUSING TO START: $1" >&2
  shift
  local line
  for line in "$@"; do
    echo "  $line" >&2
  done
  exit 2
}

# ---------------------------------------------------------------- arguments and the tree
if [ "$MODE" != "timing" ] && [ "$MODE" != "train" ]; then
  refuse "the mode is '$MODE': it is 'timing' (G5, fenced, no token) or 'train' (G6's token, the thirty)"
fi
if ! [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  refuse "the second argument must be the full 40-hex commit the run worktree is at (B3.1), not '$EXPECTED_COMMIT'"
fi
if [ "$MODE" = "train" ] && ! [[ "$TIMING_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  refuse "train takes G5's timing stamp as its third argument (fenced_timing/<stamp>/timing.json), not '$TIMING_STAMP'"
fi
if [ ! -x "$PY" ]; then
  refuse "no interpreter at $PY" "The worktree has no .venv of its own; the main tree's is the one to use."
fi
if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then
  refuse "this copy of the driver is in the implementer's worktree $WORK_TREE" \
    "J1(e): trainings run from the DETACHED run worktree $RUN_TREE, created at the reviewed commit."
fi
if [ "$WORK_TREE" != "$RUN_TREE" ]; then
  refuse "this copy is in $WORK_TREE, not the run worktree $RUN_TREE" \
    "B3.1: the driver runs from the one tree the coordinator created for it, and from no other."
fi
HEAD_COMMIT=$(git -C "$WORK_TREE" rev-parse HEAD)
if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then
  refuse "the run worktree is at $HEAD_COMMIT, not $EXPECTED_COMMIT" \
    "B3.1: the commit named at the start is the one the checkpoints will record, and it must be the reviewed one."
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.few_shot as f, offline.transfer_calibration as t; print(f.__file__); print(t.__file__)' 2>/dev/null || true)
N_LOADED=0
while IFS= read -r MODULE_FILE; do
  case "$MODULE_FILE" in
    "$WORK_TREE"/*) N_LOADED=$((N_LOADED + 1)) ;;
    *) refuse "a training module loaded from '${MODULE_FILE:-nothing}', not $WORK_TREE" "Nothing has been consumed." ;;
  esac
done <<< "$LOADED"
if [ "$N_LOADED" -ne 2 ]; then
  refuse "expected offline.few_shot and offline.transfer_calibration from $WORK_TREE, got: ${LOADED:-nothing}"
fi

ALIVE=$(pgrep -f 'python.*offline\.(collect|transfer_calibration|transfer_curve|few_shot|g2_measure)' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: another run is still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo "  Wait until it has ended. Nothing has been consumed." >&2
  exit 3
fi

if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  refuse "not a process-group leader; run in a tmux FOREGROUND pane (section 0)" \
    "The signal handler kills the process group, which is a no-op from a non-leader. Nothing consumed."
fi

SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  refuse "SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" "bash cannot trap a signal ignored on entry."
fi

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  refuse "the worktree $WORK_TREE is DIRTY" "J1(d): every checkpoint records the commit it was trained at."
fi

# ---------------------------------------------------------------- shared stages
# The layout half of the barrier: every directory this driver would create or write into is absent or a directory.
# A file in its place would let a start consume the token and then die at its first mkdir.
refuse_non_directories() {
  local target
  for target in "$@"; do
    if [ -e "$target" ] && [ ! -d "$target" ]; then
      refuse "$target exists and is not a directory. Nothing has been consumed."
    fi
  done
}

canary_both_halves() {
  CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration --draws-root "$DRAWS" --output-root "$OUTPUT" --work-dir "$TRAINING" canary) || {
    echo "${CANARY_LINE:-the canary printed no line}"
    refuse "the canary failed its correctness half. NOTHING has been consumed." \
      "A correctness failure means the ENGINE did not reproduce draw 0: a finding, not a rate question."
  }
  echo "$CANARY_LINE"
  CANARY=$(echo "$CANARY_LINE" | awk '{print $2}')
  if awk -v c="$CANARY" -v m="$CANARY_MAX_SECONDS" 'BEGIN { exit !(c > m) }'; then
    refuse "canary $CANARY s exceeds $CANARY_MAX_SECONDS s -- the machine is throttled." \
      "Check mains power and the cooling pad, then start again. Nothing has been consumed."
  fi
}

on_exit() {
  local status=$?
  if [ -n "$SAMPLER" ]; then
    kill "$SAMPLER" 2>/dev/null || true
  fi
  if [ "$SUCCESS" -ne 1 ] && [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ] && [ ! -e "$MARKER_DIR/FAILED" ]; then
    printf 'FINETUNE %s FAILED (exit %s)\n' "$MODE" "$status" > "$MARKER_DIR/FAILED"
  fi
}

fail() {
  if [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ]; then
    printf 'FINETUNE %s FAILED at %s\n' "$MODE" "$1" > "$MARKER_DIR/FAILED"
  fi
  echo "FINETUNE $MODE FAILED at $1"
  exit 1
}

on_signal() {
  trap '' INT TERM HUP
  if [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ]; then
    printf 'FINETUNE %s INTERRUPTED by a signal\n' "$MODE" > "$MARKER_DIR/FAILED"
  fi
  echo "FINETUNE $MODE INTERRUPTED by a signal: whatever was written stays where it is; killing the process group" >&2
  kill -- -$$ 2>/dev/null || true
  exit 130
}

install_traps() {
  trap on_exit EXIT
  trap on_signal INT TERM HUP
}

# ---------------------------------------------------------------- G5: the fenced timing (no token)
timing_slot() {
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot timing run --output-root "$OUTPUT" --corpus-dir "$CORPUS" --data-dir "$DATA" --stamp "$STAMP" --slot "$1" 2>&1 | sed -u "s/^/[$1] /"
}

run_slots() {
  local slot index status phase=${1%%[0-9]*}
  local pids=() failed=()
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms "$SAMPLE_MS" > "$STAMP_DIR/nvidia_smi_$phase.csv" &
  SAMPLER=$!
  for slot in "$@"; do
    timing_slot "$slot" &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do
    status=0
    wait "${pids[$index]}" || status=$?
    if [ "$status" -ne 0 ]; then
      failed+=("${@:$((index + 1)):1}")
    fi
  done
  kill "$SAMPLER" 2>/dev/null || true
  wait "$SAMPLER" 2>/dev/null || true
  SAMPLER=""
  if [ "${#failed[@]}" -ne 0 ]; then
    fail "timing $*"
  fi
}

run_timing() {
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  STAMP_DIR=$FENCED/$STAMP
  refuse_non_directories "$TRAINING" "$FENCED"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    refuse "nvidia-smi is not installed: G5 samples the device's memory with it"
  fi
  INPUTS_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot check-inputs --output-root "$OUTPUT" --corpus-dir "$CORPUS" --gate-record "$GATE_RECORD" --data-dir "$DATA") || {
    echo "${INPUTS_LINE:-check-inputs printed no result line}"
    refuse "check-inputs did not pass; the line above names the input. Nothing has been written."
  }
  echo "$INPUTS_LINE"
  if [ -e "$STAMP_DIR" ]; then
    refuse "$STAMP_DIR exists; a timing stamp is never reused. Nothing has been written."
  fi
  canary_both_halves
  install_traps
  mkdir -p "$STAMP_DIR"
  MARKER_DIR=$STAMP_DIR
  echo "P7.3c G5 — the FENCED timing (k 5, seed 101, B 400; not a registered configuration)"
  echo "  commit       $HEAD_COMMIT"
  echo "  code         $(echo "$LOADED" | tr '\n' ' ')"
  echo "  stamp dir    $STAMP_DIR"
  echo "  inputs       $INPUTS_LINE"
  echo "  canary       $CANARY_LINE"
  echo "  started      $(date -Is)"
  START=$(date +%s)
  run_slots alone
  run_slots pair1 pair2
  run_slots triple1 triple2 triple3
  run_slots repeat
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot timing build --output-root "$OUTPUT" --corpus-dir "$CORPUS" --data-dir "$DATA" --stamp "$STAMP" --k 100 || fail "timing build"
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot timing summarize --output-root "$OUTPUT" --stamp "$STAMP" || fail "timing summarize"
  SUCCESS=1
  echo "TIMING COMPLETE in $(( $(date +%s) - START ))s, stamp $STAMP  $(date -Is)" | tee "$STAMP_DIR/COMPLETE"
  echo "NEXT: the numbers go into this header, the plan and the packet; train takes the stamp $STAMP."
}

# ---------------------------------------------------------------- G6: the thirty (the author's token)
train_one() {
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot train --run "$1" --device cuda --output-root "$OUTPUT" --corpus-dir "$CORPUS" --data-dir "$DATA" 2>&1 | sed -u "s/^/[$1] /"
}

drain() {
  local index status
  local failed=()
  for index in "${!PIDS[@]}"; do
    status=0
    wait "${PIDS[$index]}" || status=$?
    if [ "$status" -ne 0 ]; then
      failed+=("${NAMES[$index]}")
    fi
  done
  PIDS=()
  NAMES=()
  if [ "${#failed[@]}" -ne 0 ]; then
    fail "train ${failed[*]}"
  fi
}

run_train() {
  TIMING=$FENCED/$TIMING_STAMP/timing.json
  refuse_non_directories "$TRAINING" "$TRAINING/starts" "$TRAINING/checkpoints" "$TRAINING/runs" "$TRAINING/attempts"
  INPUTS_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot check-inputs --output-root "$OUTPUT" --corpus-dir "$CORPUS" --gate-record "$GATE_RECORD" --data-dir "$DATA" --timing "$TIMING") || {
    echo "${INPUTS_LINE:-check-inputs printed no result line}"
    refuse "check-inputs did not pass; the line above names the input. Nothing has been consumed."
  }
  echo "$INPUTS_LINE"
  CONCURRENCY=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot timing concurrency --output-root "$OUTPUT" --timing "$TIMING") || {
    echo "${CONCURRENCY:-timing concurrency printed nothing}"
    refuse "G5's record at $TIMING gives no concurrency by the rule. Nothing has been consumed."
  }
  SCAN_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot resume-decision --all --output-root "$OUTPUT" --data-dir "$DATA") || {
    echo "${SCAN_LINE:-the resume scan printed nothing}"
    refuse "the resume scan refused; the line above names the file. Nothing has been consumed." \
      "A checkpoint that exists and does not validate is never overwritten: a person moves it aside."
  }
  echo "$SCAN_LINE"
  canary_both_halves
  install_traps
  if [ ! -f "$TOKEN" ]; then
    refuse "no run token at $TOKEN" "The author writes it (gate G6, channel (a)). Nothing has been consumed."
  fi
  echo "=== authorised by the token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
  START_DIR=$TRAINING/starts/$(date -u +%Y%m%dT%H%M%SZ)
  # From the token's deletion to FAILED's directory there is nothing that can fail but the mkdir itself (B3.4).
  rm -f "$TOKEN"
  mkdir -p "$START_DIR"
  MARKER_DIR=$START_DIR
  echo "=== token consumed and deleted; another start needs a new one"
  mkdir -p "$TRAINING/checkpoints" "$TRAINING/runs" "$TRAINING/attempts"

  echo "P7.3c C4 — the thirty fine-tunes (A24(b)): each written once, at concurrency $CONCURRENCY from G5"
  echo "  commit       $HEAD_COMMIT"
  echo "  code         $(echo "$LOADED" | tr '\n' ' ')"
  echo "  corpus       $CORPUS"
  echo "  timing       $TIMING"
  echo "  inputs       $INPUTS_LINE"
  echo "  scan         $SCAN_LINE"
  echo "  canary       $CANARY_LINE"
  echo "  start dir    $START_DIR"
  echo "  started      $(date -Is)"
  START=$(date +%s)

  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration --draws-root "$DRAWS" --output-root "$OUTPUT" --work-dir "$START_DIR" record-canary --line "$CANARY_LINE" || fail "record-canary"

  RUN_NAMES=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot runs) || fail "runs"
  for RUN in $RUN_NAMES; do
    DECISION=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot resume-decision --run "$RUN" --output-root "$OUTPUT" --data-dir "$DATA") || fail "resume-decision $RUN"
    if [ "$DECISION" = "skip" ]; then
      echo "  $RUN: skip -- its checkpoint exists and validates"
      continue
    fi
    ATTEMPT=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot attempt --run "$RUN" --output-root "$OUTPUT") || fail "attempt $RUN"
    echo "  $RUN: train, attempt $ATTEMPT"
    train_one "$RUN" &
    PIDS+=("$!")
    NAMES+=("$RUN")
    if [ "${#PIDS[@]}" -ge "$CONCURRENCY" ]; then
      drain
    fi
  done
  drain

  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot manifest --output-root "$OUTPUT" || fail "manifest"
  ( cd "$OUTPUT" && sha256sum -c --quiet SHA256SUMS_p7_3c_finetune.txt ) || fail "manifest verify"
  N_LINES=$(wc -l < "$OUTPUT/SHA256SUMS_p7_3c_finetune.txt")
  if [ "$N_LINES" -ne "$EXPECTED_MANIFEST_LINES" ]; then
    fail "manifest count ($N_LINES lines, not $EXPECTED_MANIFEST_LINES)"
  fi
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.few_shot record --output-root "$OUTPUT" --corpus-dir "$CORPUS" --timing "$TIMING" --data-dir "$DATA" || fail "record"
  SUCCESS=1
  echo "FINETUNE RUN COMPLETE in $(( $(date +%s) - START ))s  $(date -Is)" | tee "$START_DIR/COMPLETE"
  echo "NEXT: the coordinator verifies the thirty from disk and pins their digests on main (gate G7)."
}

case "$MODE" in
  timing) run_timing ;;
  train) run_train ;;
esac
