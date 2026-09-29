#!/usr/bin/env bash
# P5.3c -- C2: the TRAINING driver -- A26's sixty runs, each written exactly once, and the FENCED timing run
# (BRIEF_42 C2; Amendment A Q13, Q14, Q15; PREREGISTRATION A26(a)-(b) as corrected by A26.1).
#
# Shape: offline/campaigns/p7_3c_finetune.sh (P7.3c's training driver), with its B3 and J fixes from the start: the run
# worktree AND its commit enforced; -P on every interpreter call; the liveness guard; the group-leader and SigIgn
# checks; every refusal before the token; the traps before the token; FAILED on every path after it, through an EXIT
# trap keyed on a success flag, written with printf before any echo; the tee start lines below.
#
# 0. USAGE -- both modes FOREGROUND from the DETACHED RUN WORKTREE, which the coordinator creates at the reviewed,
#    pushed commit; that commit is the second argument, and a tree at any other commit is refused:
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p53c-run <commit>
#
#      Step 1, open a pane:      mkdir -p /home/filip/rltraffic/output/p5_3c_runs && tmux new -s p53c_train
#      Step 2 (timing), at ITS PROMPT:    bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_train.sh timing <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p5_3c_runs/train_timing_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -i -a /home/filip/rltraffic/output/p5_3c_runs/train_timing_capture.txt
#      Step 2 (train), at ITS PROMPT:     bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_train.sh train <commit> <timing-stamp> 2>&1 | tee -i -a /home/filip/rltraffic/output/p5_3c_runs/train_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -i -a /home/filip/rltraffic/output/p5_3c_runs/train_capture.txt
#
#    TIMING takes NO token (Amendment A Q13: the implementer starts it): ONE mappo1000 K = 5 run at B = 400, seed 101,
#    then its same-seed repeat, alone, every write under output/p5_3c_training/fenced_timing/<UTC stamp>/, a directory
#    nothing registered is written to or read from.  TRAIN consumes the author's token, output/p5_3c_runs/TOKEN_train
#    (gate G2), and takes the timing's stamp: check-inputs requires free device memory for its measured peak.
#    Start on an idle machine, on mains power: the canary's timing half refuses above 2.0 s.
#
#    Step 1's mkdir exists so the capture file can be created before the driver's first line.  The second half of step
#    2: `${PIPESTATUS[0]}` is the DRIVER's status, so the pane and the capture's last line agree; tee's -i ignores the
#    interrupt, so Ctrl-C's lines reach the capture.
#    NOT `tmux new -s NAME '<cmd>'`: the script would not lead its own process group, and the guard refuses.
#
# 1. WHAT IT PRODUCES
#      output/p5_3c_training/checkpoints/<run>.pt        the sixty checkpoints, each written once (os.link)
#      output/p5_3c_training/runs/<run>.json             each run's seconds, losses and targets, written once with it
#      output/p5_3c_training/attempts/<run>.<n>          one marker per start of a run, BEFORE it trains
#      output/p5_3c_training/staging/                    train_dt's staged file while a run trains, removed on every path
#      output/p5_3c_training/starts/<UTC>/canary.json    this start's canary; COMPLETE or FAILED beside it
#      output/p5_3c_training/k20_reproduction.json       the K = 20 reproduction MEASUREMENT (A26(b)): a record, never a stop
#      output/SHA256SUMS_p5_3c_train.txt                 the manifest, sha256sum's format, re-verified, 60 lines
#      output/p5_3c_training/p5_3c_train.json            the record the coordinator commits on main at G3
#    Timing: fenced_timing/<UTC>/{alone,repeat}.{pt,json}, nvidia_smi_{alone,repeat}.csv, timing.json, COMPLETE or
#    FAILED.  Nothing under docs/, the corpus, another campaign's output directory or any worktree.  The driver prints
#    no evaluation outcome: none exists here.
#
# 2. RESUME, AND WHAT IS NEVER DONE.  Every decision is offline.context_sweep's, never a `[ -f ]` here: a checkpoint
#    that exists AND validates is skipped; one that exists and does NOT validate refuses the start (resume-decision
#    --all, before the canary and the token) and is never overwritten; an absent one trains, after its attempt marker.
#    A marker with no checkpoint after it is a re-run of an infrastructure failure, counted in the record.  The only
#    deletion is the consumed token; nothing is moved.
#
# 3. ORDERING -- arguments -> interpreter -> not the implementer's tree -> THE RUN TREE -> ITS COMMIT -> the working
#    directory is the run tree -> the modules load (-P) from it -> no live runner -> GROUP LEADER -> SigIgn -> dirty tree
#    -> the layout -> check-inputs (the ten published K = 20 checkpoints by digest; both subjects' recipe from the corpus
#    against P4's and P4.7's committed values; CUDA; for train, the timing record and free device memory) -> [train: the
#    resume scan] -> the canary, BOTH halves -> TRAPS -> [train: token] -> the start directory -> record-canary -> the
#    runs, one at a time -> compare-k20 -> manifest (re-verified, 60 lines) -> record, LAST -> COMPLETE.
#
# 4. -P ON EVERY INTERPRETER CALL, PYTHONPATH the run tree, and the WORKING DIRECTORY the run tree too (plan Q14):
#    train_dt stamps the working directory's commit into every payload, and the module refuses a working directory at
#    another commit than the code's.  THE REGIME IS P5.2's / P7.3c's: OMP and MKL at one thread and
#    CUBLAS_WORKSPACE_CONFIG UNSET; the Python side refuses the variable set, refuses deterministic algorithms and pins
#    one torch thread.  The corpus root is P4's absolute one, because the statistics record their directories.
#
# 5. TIME -- MEASURED BY THE TIMING MODE BEFORE THE TOKEN; its numbers are written here from its record, with its canary.
#    Stamp 20260929T112505Z (2026-09-29, 11:25 UTC), the run tree at 8327128c41a3, the machine quiet (load 0.00) and on
#    mains power; canary 1.03 s, both correctness halves the reference's (-32648.0, 247.75089149261333, 360 decisions);
#    "TIMING COMPLETE in 17s", "DRIVER EXIT: 0".  Record output/p5_3c_training/fenced_timing/20260929T112505Z/
#    timing.json (sha256 46fbccaba701...); capture output/p5_3c_runs/train_timing_capture.txt.
#    ONE configuration -- mappo1000_k5_b64_seed101, 400 steps (train_dt's warm-up 200), alone, the registered regime:
#                ms/step   input build   device peak (nvidia-smi, whole GPU)   allocated peak (torch, this process)
#      alone      8.25       2.16 s        1,540 MiB                             97.8 MiB
#      repeat     7.01       2.23 s        1,516 MiB                             97.8 MiB
#    The GPU held 1,125 / 1,136 MiB before each slot (the Windows host's baseline): one run adds 415 / 380 MiB.
#    The same-seed repeat by two routes: file sha256 EQUAL (f796315069cf...), weights sha256 EQUAL (bf8a146f80be...).
#    An observation at this one configuration, NOT a property of the regime (deterministic algorithms are OFF, and
#    P7.3c's repeat differed); nothing relies on it -- the K = 20 reproduction is a measurement, never a stop.
#    Measured at K = 5, batch 64, 400 steps; the sixty's total is an EXTRAPOLATION across K in {1, 2, 10, 20}, the two
#    equal-supervision arms and sixty input builds, not a measurement: 60 x 40,000 steps at the measured 7.01-8.25
#    ms/step is 4.7-5.5 h of steps (280-330 s a run), plus sixty builds of about 2.2 s.  Not measured: ms/step at the
#    other K (20 of the 60 runs are K >= 10) and at 3,840 tokens a step (K = 20's and both equal-supervision arms', 20
#    runs; the measured point is 960), the builds at other K, the per-run process starts and calls, thermal drift.
#    RECORDED, not measured here: train_dt's own loop seconds for the DT at K = 20, batch 64, 40,000 steps on this GPU
#    were 193.4-205.0 s (4.84-5.12 ms/step) in P4.7's fifteen runs (docs/data/p4_7_training.json) and 202.3-204.1 s
#    in three of P4's five (docs/data/p4_training.json; the other two 356.2 s and 14,018.0 s) -- below this 400-step
#    rate, whose short loop amortises less of the process's first CUDA steps (their share is not measured).  At that
#    recorded rate the sixty's steps take 3.2-3.4 h: an extrapolation too.

set -euo pipefail

MODE=${1:-}
EXPECTED_COMMIT=${2:-}
TIMING_STAMP=${3:-}

MAIN=/home/filip/rltraffic
# The tree this copy of the script lives in, never a hardcoded one.
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
IMPLEMENTER_TREE=/home/filip/rltraffic-p53c
RUN_TREE=/home/filip/rltraffic-p53c-run
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
TRAINING=$OUTPUT/p5_3c_training
FENCED=$TRAINING/fenced_timing
CORPUS=$MAIN/datasets_v11
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
TOKEN=$OUTPUT/p5_3c_runs/TOKEN_train
CANARY_MAX_SECONDS=2.0
SAMPLE_MS=250
EXPECTED_MANIFEST_LINES=60

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# P5.2's non-deterministic regime UNSETS this rather than merely not setting it: it constrains cuBLAS's workspace and
# so its GEMM selection, and the launch shell may carry it.
unset CUBLAS_WORKSPACE_CONFIG

SUCCESS=0
MARKER_DIR=""
SAMPLER=""

echo "=== P5.3c training driver (${MODE:-no mode}): WORK_TREE $WORK_TREE (derived from this script's own location)"

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
  refuse "the mode is '$MODE': it is 'timing' (fenced, no token) or 'train' (G2's token, the sixty)"
fi
if ! [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  refuse "the second argument must be the full 40-hex commit the run worktree is at, not '$EXPECTED_COMMIT'"
fi
if [ "$MODE" = "train" ] && ! [[ "$TIMING_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  refuse "train takes the timing's stamp as its third argument (fenced_timing/<stamp>/timing.json), not '$TIMING_STAMP'"
fi
if [ ! -x "$PY" ]; then
  refuse "no interpreter at $PY" "The worktree has no .venv of its own; the main tree's is the one to use."
fi
if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then
  refuse "this copy of the driver is in the implementer's worktree $WORK_TREE" \
    "Trainings run from the DETACHED run worktree $RUN_TREE, created at the reviewed commit."
fi
if [ "$WORK_TREE" != "$RUN_TREE" ]; then
  refuse "this copy is in $WORK_TREE, not the run worktree $RUN_TREE" \
    "The driver runs from the one tree the coordinator created for it, and from no other."
fi
HEAD_COMMIT=$(git -C "$WORK_TREE" rev-parse HEAD)
if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then
  refuse "the run worktree is at $HEAD_COMMIT, not $EXPECTED_COMMIT" \
    "The commit named at the start is the one every checkpoint will record, and it must be the reviewed one."
fi

# Plan Q14: train_dt stamps the WORKING DIRECTORY's commit, so the working directory is the run tree.
cd "$WORK_TREE"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.context_sweep as c, offline.transfer_calibration as t; print(c.__file__); print(t.__file__)' 2>/dev/null || true)
N_LOADED=0
while IFS= read -r MODULE_FILE; do
  case "$MODULE_FILE" in
    "$WORK_TREE"/*) N_LOADED=$((N_LOADED + 1)) ;;
    *) refuse "a training module loaded from '${MODULE_FILE:-nothing}', not $WORK_TREE" "Nothing has been consumed." ;;
  esac
done <<< "$LOADED"
if [ "$N_LOADED" -ne 2 ]; then
  refuse "expected offline.context_sweep and offline.transfer_calibration from $WORK_TREE, got: ${LOADED:-nothing}"
fi

ALIVE=$(pgrep -f 'python.*offline\.(context_sweep|collect|transfer_calibration|transfer_curve|few_shot|g2_measure)' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
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
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  refuse "the worktree $WORK_TREE is DIRTY" "Every checkpoint records the commit it was trained at."
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
  # SIGKILL, which no trap can swallow: the sampler must never outlive the driver (it would hold the capture's pipe).
  if [ -n "$SAMPLER" ]; then
    kill -9 "$SAMPLER" 2>/dev/null || true
  fi
  if [ "$SUCCESS" -ne 1 ] && [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ] && [ ! -e "$MARKER_DIR/FAILED" ]; then
    printf 'TRAIN %s FAILED (exit %s)\n' "$MODE" "$status" > "$MARKER_DIR/FAILED"
  fi
}

fail() {
  if [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ]; then
    printf 'TRAIN %s FAILED at %s\n' "$MODE" "$1" > "$MARKER_DIR/FAILED"
  fi
  echo "TRAIN $MODE FAILED at $1"
  exit 1
}

on_signal() {
  trap '' INT TERM HUP
  if [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ]; then
    printf 'TRAIN %s INTERRUPTED by a signal\n' "$MODE" > "$MARKER_DIR/FAILED"
  fi
  echo "TRAIN $MODE INTERRUPTED by a signal: whatever was written stays where it is; killing the process group" >&2
  kill -- -$$ 2>/dev/null || true
  exit 130
}

install_traps() {
  trap on_exit EXIT
  trap on_signal INT TERM HUP
}

# ---------------------------------------------------------------- the fenced timing run (no token)
# The device sampler.  A SIGTERM that reaches the background child BEFORE it has become nvidia-smi lands in this shell's
# inherited trap and is lost, and the sampler then outlives its slot: the wait below hangs, and an orphaned sampler holds
# the capture's pipe open.  Found by the stubbed-slot test, whose slot ends within milliseconds (2026-09-29).  So the
# slot starts only once the child IS nvidia-smi, and the stop is bounded: SIGTERM, then SIGKILL after ten seconds.
start_sampler() {
  local tries=0
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms "$SAMPLE_MS" > "$STAMP_DIR/nvidia_smi_$1.csv" &
  SAMPLER=$!
  until [ "$(cat "/proc/$SAMPLER/comm" 2>/dev/null)" = "nvidia-smi" ]; do
    tries=$((tries + 1))
    if [ "$tries" -gt 200 ]; then
      fail "the device sampler for $1 did not start"
    fi
    sleep 0.05
  done
}

stop_sampler() {
  local pid=$SAMPLER tries=0
  kill "$pid" 2>/dev/null || true
  # An exited child that is not yet reaped still answers kill -0; its /proc state letter is Z.
  while [ -e "/proc/$pid" ] && [ "$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)" != "Z" ] && [ "$tries" -lt 40 ]; do
    sleep 0.25
    tries=$((tries + 1))
  done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  SAMPLER=""
}

sample_slot() {
  local status=0
  start_sampler "$1"
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep timing run --output-root "$OUTPUT" --corpus-root "$CORPUS" --stamp "$STAMP" --slot "$1" 2>&1 | sed -u "s/^/[$1] /" || status=$?
  stop_sampler
  if [ "$status" -ne 0 ]; then
    fail "timing $1"
  fi
}

run_timing() {
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  STAMP_DIR=$FENCED/$STAMP
  refuse_non_directories "$TRAINING" "$FENCED"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    refuse "nvidia-smi is not installed: the timing samples the device's memory with it"
  fi
  INPUTS_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep check-inputs --output-root "$OUTPUT" --corpus-root "$CORPUS" --data-dir "$DATA") || {
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
  echo "P5.3c C2 -- the FENCED timing (mappo1000, K 5, batch 64, seed 101, B 400; not a registered configuration)"
  echo "  commit       $HEAD_COMMIT"
  echo "  code         $(echo "$LOADED" | tr '\n' ' ')"
  echo "  stamp dir    $STAMP_DIR"
  echo "  inputs       $INPUTS_LINE"
  echo "  canary       $CANARY_LINE"
  echo "  started      $(date -Is)"
  START=$(date +%s)
  sample_slot alone
  sample_slot repeat
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep timing summarize --output-root "$OUTPUT" --stamp "$STAMP" || fail "timing summarize"
  SUCCESS=1
  echo "TIMING COMPLETE in $(( $(date +%s) - START ))s, stamp $STAMP  $(date -Is)" | tee "$STAMP_DIR/COMPLETE"
  echo "NEXT: the numbers go into this header, the packet and the plan; train takes the stamp $STAMP."
}

# ---------------------------------------------------------------- the sixty (the author's token)
train_one() {
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep train --run "$1" --device cuda --output-root "$OUTPUT" --corpus-root "$CORPUS" 2>&1 | sed -u "s/^/[$1] /"
}

run_train() {
  TIMING=$FENCED/$TIMING_STAMP/timing.json
  refuse_non_directories "$TRAINING" "$TRAINING/starts" "$TRAINING/checkpoints" "$TRAINING/runs" "$TRAINING/attempts" "$TRAINING/staging"
  INPUTS_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep check-inputs --output-root "$OUTPUT" --corpus-root "$CORPUS" --data-dir "$DATA" --timing "$TIMING") || {
    echo "${INPUTS_LINE:-check-inputs printed no result line}"
    refuse "check-inputs did not pass; the line above names the input. Nothing has been consumed."
  }
  echo "$INPUTS_LINE"
  SCAN_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep resume-decision --all --output-root "$OUTPUT" --corpus-root "$CORPUS") || {
    echo "${SCAN_LINE:-the resume scan printed nothing}"
    refuse "the resume scan refused; the line above names the file. Nothing has been consumed." \
      "A checkpoint that exists and does not validate is never overwritten: a person moves it aside."
  }
  echo "$SCAN_LINE"
  canary_both_halves
  install_traps
  if [ ! -f "$TOKEN" ]; then
    refuse "no run token at $TOKEN" "The author writes it (gate G2, channel (a)). Nothing has been consumed."
  fi
  echo "=== authorised by the token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(< "$TOKEN")"
  START_DIR=$TRAINING/starts/$(date -u +%Y%m%dT%H%M%SZ)
  # From the token's deletion to FAILED's directory there is nothing that can fail but the mkdir itself.
  rm -f "$TOKEN"
  mkdir -p "$START_DIR"
  MARKER_DIR=$START_DIR
  echo "=== token consumed and deleted; another start needs a new one"
  mkdir -p "$TRAINING/checkpoints" "$TRAINING/runs" "$TRAINING/attempts" "$TRAINING/staging"

  echo "P5.3c C2 -- the sixty trainings (A26(a)-(b)): each written once, one at a time"
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

  RUN_NAMES=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep runs) || fail "runs"
  for RUN in $RUN_NAMES; do
    DECISION=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep resume-decision --run "$RUN" --output-root "$OUTPUT" --corpus-root "$CORPUS") || fail "resume-decision $RUN"
    if [ "$DECISION" = "skip" ]; then
      echo "  $RUN: skip -- its checkpoint exists and validates"
      continue
    fi
    ATTEMPT=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep attempt --run "$RUN" --output-root "$OUTPUT") || fail "attempt $RUN"
    echo "  $RUN: train, attempt $ATTEMPT"
    train_one "$RUN" || fail "train $RUN"
  done

  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep compare-k20 --output-root "$OUTPUT" --data-dir "$DATA" || fail "compare-k20"
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep manifest --output-root "$OUTPUT" || fail "manifest"
  ( cd "$OUTPUT" && sha256sum -c --quiet SHA256SUMS_p5_3c_train.txt ) || fail "manifest verify"
  N_LINES=$(wc -l < "$OUTPUT/SHA256SUMS_p5_3c_train.txt")
  if [ "$N_LINES" -ne "$EXPECTED_MANIFEST_LINES" ]; then
    fail "manifest count ($N_LINES lines, not $EXPECTED_MANIFEST_LINES)"
  fi
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep record --output-root "$OUTPUT" --corpus-root "$CORPUS" --timing "$TIMING" || fail "record"
  SUCCESS=1
  echo "TRAINING RUN COMPLETE in $(( $(date +%s) - START ))s  $(date -Is)" | tee "$START_DIR/COMPLETE"
  echo "NEXT: the coordinator verifies the sixty from disk and pins docs/data/p5_3c_train.json on main (gate G3)."
}

case "$MODE" in
  timing) run_timing ;;
  train) run_train ;;
esac
