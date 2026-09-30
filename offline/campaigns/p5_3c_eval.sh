#!/usr/bin/env bash
# P5.3c -- C3: the evaluation CAMPAIGN -- A26(c)'s reference arms, the GATE, then the sixty runs on the 100 held-out draws,
# the report, under ONE token (BRIEF_42 C3; PREREGISTRATION A26 as corrected by A26.1; Amendments A, A.1, B, B.1, C).
#
# Shape: offline/campaigns/p7_3c_grid4x4.sh (P7.3c's campaign), with the P5.3c training driver's fixes from the start:
# the run worktree AND its commit enforced; -P on every interpreter call; the liveness guard; the group-leader and SigIgn
# checks; every refusal before the token; the traps before the token; FAILED on every path after it (beside the token if
# the work directory could not be made), written with printf before any echo; the device sampler that cannot outlive
# its driver; the tee start line below.
#
# 0. USAGE -- the FOREGROUND form, from the DETACHED RUN WORKTREE, which the coordinator creates at the reviewed, pushed
#    commit before the token; that commit is the one argument, and a tree at any other commit is refused:
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p53c-run <commit>
#
#      Step 1, open a pane:      mkdir -p /home/filip/rltraffic/output/p5_3c_runs && tmux new -s p53c_campaign
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p53c-run/offline/campaigns/p5_3c_eval.sh <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p5_3c_runs/campaign_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -i -a /home/filip/rltraffic/output/p5_3c_runs/campaign_capture.txt
#
#    The token is output/p5_3c_runs/TOKEN_campaign (gate G5, the author's).  Start on an idle machine, on mains power: the
#    canary's timing half refuses above 2.0 s.  Step 1's mkdir exists so the capture file can be created before the
#    driver's first line; `${PIPESTATUS[0]}` is the DRIVER's status; tee's -i ignores the interrupt, so Ctrl-C's lines
#    reach the capture.  NOT `tmux new -s NAME '<cmd>'`: the script would not lead its own process group, and the guard
#    refuses.
#
#    THE CANARY HANG (DEFERRED 101; BRIEF_42 Amendment B, B5), a rule for every start: the machine is quiet at every
#    start; if the driver prints nothing for a minute after `resume_check PASSED`, the canary is hung: Ctrl-C -- the
#    canary precedes the token, so NOTHING is consumed -- and start again. A second occurrence is a finding for the
#    plan, not a rate question.
#
# 1. WHAT IT PRODUCES
#      output/p5_3c/g2/reference_reroll_check_<UTC>/   BEFORE the token: the fifteen reference cells re-rolled (fenced)
#      output/p5_3c/cells/canary.json                  the canary line, right after the token
#      output/p5_3c/cells/cell_<arm>_seed<s>_draw<d>.json   one chunk per cell of both stages, atomic, resumable
#      output/p5_3c/cells/failed/                      chunks that failed their own re-check (moved, never overwritten)
#      output/p5_3c/nvidia_smi_<UTC>.csv               the device's memory over the two stages (the peak is printed)
#      output/p5_3c/reference_gate.json                A26(c)'s gate record, written ONLY on a pass
#      output/p5_3c/artifacts/p5_3c_context_sweep.json the artifact C4 commits BY HAND
#      output/SHA256SUMS_p5_3c.txt                     over output/p5_3c/ ONLY, written once, re-verified
#      output/p5_3c/cells/COMPLETE  or  .../FAILED     the terminal marker, on EVERY path after the token
#      output/p5_3c_runs/FAILED_campaign               instead, if the token was consumed and the work directory could
#                                                      not be made
#    Nothing under docs/, the corpus, another campaign's directory or any worktree.
#
# 2. THE STAGES AND THE GATE, under ONE token, in one work directory:
#      reference       1,000 cells: P4's five published K = 20 checkpoints (ref_mappo1000_k20) and P4.7's five
#                      (ref_mix50_k20) on the 100 held-out draws
#      reference-gate  A26(c): the 500 ref_mappo1000_k20 chunks == docs/data/p4_k20_att_engine_rows.json under ==, both
#                      definitions, or the campaign STOPS here: FAILED, and no sweep cell exists (the sweep stage
#                      refuses on its own without the gate's record -- the second line)
#      sweep           6,000 cells: the sixty registered runs, each through the pinned training record
#      report, then the manifest, then COMPLETE.
#    It prints NO outcome: `cells` prints each cell's name and whether it stood, the gate a verdict and cell and field
#    NAMES, `report` the artifact's path and its cell count.  G6 recomputes T1-T3 from the raw chunks first.
#    RESUME -- two leftovers of a stopped run need a hand, each named when it bites (Amendment D, D4.1(b)):
#      a restart that re-rolled a reference chunk finds the gate record stale -- `reference-gate` refuses "already
#      exists and differs" after the token; move output/p5_3c/reference_gate.json aside by hand and start again (with
#      a new token: this refusal comes after one was consumed);
#      after a Ctrl-C a worker's .cell_<...>.json.<pid>.tmp may remain in cells/ -- `resume-check` names it before
#      the token; move it aside by hand.
#    `<name> HUNG (round n): re-rolled` lines are the stage's own re-rolls of a hung cell (Amendment D.2), not a
#    failure; a restart after C3.2 re-rolls every chunk rolled by the earlier code (J1(c)).
#
# 3. ORDERING -- every check that can refuse PRECEDES the token, so a refused start consumes nothing: the commit argument
#    -> the interpreter -> not the implementer's tree -> the run tree and its commit -> the cwd -> the modules from the
#    run tree -> no live runner -> GROUP LEADER -> SigIgn -> the committed inputs exist -> the dirty tree -> the RSS
#    budget -> the layout -> campaign-inputs (everything BY DIGEST: the rows, the training record at its pin with the
#    sixty and their manifest, the ten published checkpoints, both subjects' env settings == P8.4b's, the draws, CUDA and
#    free device memory) -> resume-check (no undeclared chunk file, every chunk's commit resolvable) -> the canary, BOTH
#    halves -> the fifteen fenced re-rolls, fifteen MATCH lines counted -> the TRAPS -> the token -> record-canary -> the
#    device sampler -> reference -> the gate -> sweep -> report -> manifest -> COMPLETE.
#
# 4. -P ON EVERY INTERPRETER CALL, PYTHONPATH the run tree, the cwd the MAIN tree (P7.3c's convention; plan Q14): the
#    chunks record the MODULE tree's commit, read strictly.  The regime: OMP and MKL at one thread, every pool worker one
#    torch thread, CUBLAS_WORKSPACE_CONFIG unset; the DT on CUDA (A26.1(b)).  The skip decision is Python's (a chunk is
#    reused only by its content, J1(c)).
#
# 5. TIME AND MEMORY -- NOT MEASURED AT THIS COMMIT: 7,000 cells at P5.3b's 2.9 s per hz1x1 episode single-process is
#    5.6 h of episodes on one worker; at 12 workers the rate is the pre-flight's (G4) to measure, and the driver prints
#    its own wall clock.  Device: the fenced timing's 415 MiB per TRAINING process x 12 x 1.4 is the budget
#    campaign-inputs requires free (an evaluation process holds less).  Host: P7.3d's G2 measured a 17,297 MiB tree at 12
#    workers (1,441 MiB per torch-and-CUDA process); its x 1.4, 24,216 MiB, is borrowed as this driver's budget -- an
#    upper bound (hz1x1's CityFlow env is smaller than grid4x4's SUMO), not a measurement of this campaign.
#    A hung cell costs STAGE_RESULT_TIMEOUT_S (180 s) plus a re-roll, detected when the stage's other results have all
#    arrived (Amendment D.2: CityFlow's destructor race at an episode's end, DEFERRED 104).

set -euo pipefail

EXPECTED_COMMIT=${1:-}

MAIN=/home/filip/rltraffic
# The tree this copy of the script lives in, never a hardcoded one.
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
IMPLEMENTER_TREE=/home/filip/rltraffic-p53c
RUN_TREE=/home/filip/rltraffic-p53c-run
PY=$MAIN/.venv/bin/python
OUTPUT=$MAIN/output
CAMPAIGN_DIR=$OUTPUT/p5_3c
WORK=$CAMPAIGN_DIR/cells
ARTIFACTS=$CAMPAIGN_DIR/artifacts
G2_DIR=$CAMPAIGN_DIR/g2
CORPUS=$MAIN/datasets_v11
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
TOKEN=$OUTPUT/p5_3c_runs/TOKEN_campaign
CANARY_MAX_SECONDS=2.0
SAMPLE_MS=1000
WORKERS=12
REROLL_CELLS=15
PEAK_TREE_RSS_MIB=17297
RSS_BUDGET_MIB=24216

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# The training driver's regime, kept: the variable constrains cuBLAS's workspace, and the launch shell may carry it.
unset CUBLAS_WORKSPACE_CONFIG

# The roots every campaign command takes (options of the SUBCOMMAND, after its name).
COMMON=(--output-root "$OUTPUT" --corpus-root "$CORPUS" --data-dir "$DATA" --draws-root "$DRAWS")

SUCCESS=0
MARKER_DIR=""
SAMPLER=""
# The token is consumed BEFORE the work directory exists, so a failure in between -- a mkdir that fails -- leaves FAILED
# beside the token instead of nowhere. Before the token nothing is written but the fenced re-roll's records under
# output/p5_3c/g2/ (section 1).
TOKEN_CONSUMED=0
FAILED_FALLBACK=$(dirname "$TOKEN")/FAILED_campaign

echo "=== P5.3c campaign driver: WORK_TREE $WORK_TREE (derived from this script's own location)"

refuse() {
  echo "REFUSING TO START: $1" >&2
  shift
  local line
  for line in "$@"; do
    echo "  $line" >&2
  done
  exit 2
}

# ---------------------------------------------------------------- the argument and the tree
if ! [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  refuse "the argument must be the full 40-hex commit the run worktree is at, not '$EXPECTED_COMMIT'"
fi
if [ ! -x "$PY" ]; then
  refuse "no interpreter at $PY" "The worktree has no .venv of its own; the main tree's is the one to use."
fi
if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then
  refuse "this copy of the driver is in the implementer's worktree $WORK_TREE" \
    "The campaign runs from the DETACHED run worktree $RUN_TREE, created at the reviewed commit."
fi
if [ "$WORK_TREE" != "$RUN_TREE" ]; then
  refuse "this copy is in $WORK_TREE, not the run worktree $RUN_TREE" \
    "The driver runs from the one tree the coordinator created for it, and from no other."
fi
HEAD_COMMIT=$(git -C "$WORK_TREE" rev-parse HEAD)
if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then
  refuse "the run worktree is at $HEAD_COMMIT, not $EXPECTED_COMMIT" \
    "The commit named at the start is the one every chunk records, and it must be the reviewed one."
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.context_sweep as c, offline.transfer_calibration as t; print(c.__file__); print(t.__file__)' 2>/dev/null || true)
N_LOADED=0
while IFS= read -r MODULE_FILE; do
  case "$MODULE_FILE" in
    "$WORK_TREE"/*) N_LOADED=$((N_LOADED + 1)) ;;
    *) refuse "a campaign module loaded from '${MODULE_FILE:-nothing}', not $WORK_TREE" "Nothing has been consumed." ;;
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

# ---------------------------------------------------------------- the inputs
for INPUT in "$DATA/p4_k20_att_engine_rows.json" "$DATA/p5_3c_train.json" "$DATA/p4_gate.json" \
             "$DATA/p4_7_training.json" "$DATA/p5_3b_nortg.json"; do
  if [ ! -f "$INPUT" ]; then
    refuse "missing committed input $INPUT" "Existence first; campaign-inputs then reads each at its pin. Nothing consumed."
  fi
done

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  refuse "the worktree $WORK_TREE is DIRTY" \
    "J1(d): one untracked file makes every chunk rolled after it code_dirty true, and the chunk refuses it."
fi

AVAILABLE_MIB=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
if [ -n "$AVAILABLE_MIB" ] && [ "$AVAILABLE_MIB" -lt "$RSS_BUDGET_MIB" ]; then
  refuse "${AVAILABLE_MIB} MiB available, below the ${RSS_BUDGET_MIB} MiB budget" \
    "P7.3d's G2 measured ${PEAK_TREE_RSS_MIB} MiB at ${WORKERS} workers, x 1.4 (section 5). Nothing has been consumed."
fi

# The layout half of the barrier: every directory this driver would create or write into is absent or a directory.
refuse_non_directories() {
  local target
  for target in "$@"; do
    if [ -e "$target" ] && [ ! -d "$target" ]; then
      refuse "$target exists and is not a directory. Nothing has been consumed."
    fi
  done
}
refuse_non_directories "$CAMPAIGN_DIR" "$WORK" "$ARTIFACTS" "$G2_DIR"

# Existence is not identity: every input by digest, and the device, before anything runs.
INPUTS_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep campaign-inputs "${COMMON[@]}") || {
  echo "${INPUTS_LINE:-campaign-inputs printed no result line}"
  refuse "campaign-inputs did not pass; the line above names the input. Nothing has been consumed."
}
echo "$INPUTS_LINE"

# Two things the pool would otherwise meet only AFTER the token: a chunk file no declared cell names, and a chunk whose
# commit git cannot resolve.
RESUME_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep resume-check --output-root "$OUTPUT") || {
  echo "${RESUME_LINE:-resume-check printed no line}"
  refuse "resume-check found a file that would stop the campaign after the token" \
    "The lines above name it. Move it aside by hand, or make its commit reachable. Nothing has been consumed."
}
echo "$RESUME_LINE"

# ---------------------------------------------------------------- the canary, BOTH halves
CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration --draws-root "$DRAWS" --output-root "$OUTPUT" --work-dir "$WORK" canary) || {
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

# ---------------------------------------------------------------- the fifteen fenced re-rolls (Amendment A, Q2; A.1)
if ! REFERENCE_LINES=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep reference-reroll-check "${COMMON[@]}" --canary-seconds "$CANARY" --workers "$WORKERS"); then
  echo "${REFERENCE_LINES:-reference_reroll_check printed no result line}"
  refuse "reference_reroll_check did not pass (Amendment A, Q2): a divergence is a finding for a registration decision" \
    "The token has NOT been consumed; the fenced record is under $G2_DIR."
fi
echo "$REFERENCE_LINES"
N_REFERENCE_RESULTS=$(grep -cE '^reference_reroll_check (MATCH|NO MATCH) ' <<< "$REFERENCE_LINES" || true)
N_REFERENCE_MATCH=$(grep -c '^reference_reroll_check MATCH ' <<< "$REFERENCE_LINES" || true)
if [ "$N_REFERENCE_RESULTS" -ne "$REROLL_CELLS" ] || [ "$N_REFERENCE_MATCH" -ne "$REROLL_CELLS" ]; then
  refuse "reference_reroll_check printed $N_REFERENCE_RESULTS result line(s), $N_REFERENCE_MATCH of them MATCH" \
    "Fifteen MATCH lines, one per re-rolled cell, are required. Nothing has been consumed."
fi

# ---------------------------------------------------------------- the traps, BEFORE the token
# Where FAILED goes: the work directory once it exists; before that, and only once the token is consumed, beside the
# token; before the token, nowhere -- a refusal consumes and writes nothing.
failed_marker() {
  if [ -n "$MARKER_DIR" ] && [ -d "$MARKER_DIR" ]; then
    echo "$MARKER_DIR/FAILED"
  elif [ "$TOKEN_CONSUMED" -eq 1 ]; then
    echo "$FAILED_FALLBACK"
  fi
}

on_exit() {
  local status=$?
  local marker
  # SIGKILL, which no trap can swallow: the sampler must never outlive the driver (it would hold the capture's pipe).
  if [ -n "$SAMPLER" ]; then
    kill -9 "$SAMPLER" 2>/dev/null || true
  fi
  marker=$(failed_marker)
  if [ "$SUCCESS" -ne 1 ] && [ -n "$marker" ] && [ ! -e "$marker" ]; then
    printf 'CAMPAIGN FAILED (exit %s)\n' "$status" > "$marker"
  fi
}

fail() {
  local marker
  marker=$(failed_marker)
  if [ -n "$marker" ]; then
    printf 'CAMPAIGN FAILED at %s\n' "$1" > "$marker"
  fi
  echo "CAMPAIGN FAILED at $1"
  exit 1
}

on_signal() {
  trap '' INT TERM HUP
  local marker
  marker=$(failed_marker)
  if [ -n "$marker" ]; then
    printf 'CAMPAIGN INTERRUPTED by a signal\n' > "$marker"
  fi
  echo "CAMPAIGN INTERRUPTED by a signal: every completed chunk is on disk, and running again resumes from them by" >&2
  echo "  CONTENT. Killing the process group." >&2
  kill -- -$$ 2>/dev/null || true
  exit 130
}

# The device sampler (the training driver's, 2026-09-29): the stages start only once the child IS nvidia-smi, and the
# stop is bounded -- SIGTERM, then SIGKILL after ten seconds.
start_sampler() {
  local tries=0
  SAMPLES=$CAMPAIGN_DIR/nvidia_smi_$1.csv
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms "$SAMPLE_MS" > "$SAMPLES" &
  SAMPLER=$!
  until [ "$(cat "/proc/$SAMPLER/comm" 2>/dev/null)" = "nvidia-smi" ]; do
    tries=$((tries + 1))
    if [ "$tries" -gt 200 ]; then
      fail "the device sampler did not start"
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

trap on_exit EXIT
trap on_signal INT TERM HUP

# ---------------------------------------------------------------- the ONE token
if [ ! -f "$TOKEN" ]; then
  refuse "no run token at $TOKEN" "The author writes it (gate G5, channel (a)). Nothing has been consumed."
fi
echo "=== authorised by the token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(< "$TOKEN")"
rm -f "$TOKEN"
TOKEN_CONSUMED=1
echo "=== token consumed and deleted; another start needs a new one"

# Only NOW may anything be created or cleared.
mkdir -p "$WORK" "$ARTIFACTS"
MARKER_DIR=$WORK
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

STARTED_UTC=$(date -u +%Y%m%dT%H%M%SZ)
echo "P5.3c C3 -- the context-length campaign (A26(c)): 7,000 cells, the reference gate, ONE token"
echo "  commit       $HEAD_COMMIT"
echo "  code         $(echo "$LOADED" | tr '\n' ' ')"
echo "  cwd          $(pwd)   (the MAIN tree)"
echo "  stages       reference (1,000) -> reference-gate -> sweep (6,000) -> report"
echo "  workers      $WORKERS"
echo "  memory       ${AVAILABLE_MIB} MiB available, budget ${RSS_BUDGET_MIB} MiB"
echo "  inputs       $INPUTS_LINE"
echo "  canary       $CANARY_LINE"
echo "  re-roll      $N_REFERENCE_MATCH/$REROLL_CELLS MATCH"
echo "  started      $(date -Is)"
echo ""
START=$(date +%s)

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration --draws-root "$DRAWS" --output-root "$OUTPUT" --work-dir "$WORK" record-canary --line "$CANARY_LINE" || fail "record-canary"

start_sampler "$STARTED_UTC"

# ---------------------------------------------------------------- the reference stage, then THE GATE
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep cells "${COMMON[@]}" --stage reference --canary-seconds "$CANARY" --workers "$WORKERS" || fail "cells reference"

# A26(c): P4's 500 rows reproduced under ==, or the campaign STOPS here and no sweep cell exists. The module prints ONE
# verdict line: cell and field names, never a value.
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep reference-gate --output-root "$OUTPUT" --data-dir "$DATA" || fail "reference-gate"

# ---------------------------------------------------------------- the sixty runs
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep cells "${COMMON[@]}" --stage sweep --canary-seconds "$CANARY" --workers "$WORKERS" || fail "cells sweep"

stop_sampler
DEVICE_PEAK=$(sort -n "$SAMPLES" | tail -1)
echo "  device       peak ${DEVICE_PEAK:-unknown} MiB over the two stages (nvidia-smi, the whole GPU; $SAMPLES)"

# ---------------------------------------------------------------- the artifact, the manifest
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep report "${COMMON[@]}" || fail "report"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.context_sweep campaign-manifest --output-root "$OUTPUT" || fail "campaign-manifest"
( cd "$OUTPUT" && sha256sum -c --quiet SHA256SUMS_p5_3c.txt ) || fail "manifest verify"

ELAPSED=$(( $(date +%s) - START ))
# COMPLETE first, THEN success: a final write that fails still leaves on_exit's FAILED marker (Amendment D, D4.1(a)).
printf 'CAMPAIGN COMPLETE in %ss  %s\n' "$ELAPSED" "$(date -Is)" > "$WORK/COMPLETE"
SUCCESS=1
echo ""
echo "CAMPAIGN COMPLETE in ${ELAPSED}s  $(date -Is)"
echo "NEXT: the coordinator reads the capture, then recomputes T1-T3 from the raw chunks BEFORE opening the artifact (G6)."
