#!/usr/bin/env bash
# P7.3c -- C7: the grid4x4 few-shot CAMPAIGN, A24(c)'s three stages under ONE token (BRIEF_41 C7; PREREGISTRATION A24).
#
# Shape: offline/campaigns/p7_3d_grid4x4.sh (P7.3d's campaign), with Amendment B's B3 corrections that the P7.3c drivers
# carry from the start: the run worktree AND its commit enforced (B3.1); FAILED on every path after the token, through
# an EXIT trap keyed on a success flag, written with printf before any echo, and `tee -i -a` in the documented start
# line (B3.4).
#
# 0. USAGE -- the FOREGROUND form, from the DETACHED RUN WORKTREE, which the coordinator creates (or moves) at the
#    reviewed, pushed commit before the token (G8):
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p73c-run <commit>
#
#      Step 1, open a pane:      tmux new -s p73c_campaign
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p73c-run/offline/campaigns/p7_3c_grid4x4.sh <commit> 2>&1 | tee -i -a /home/filip/rltraffic/output/p7_3c_runs/campaign_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -i -a /home/filip/rltraffic/output/p7_3c_runs/campaign_capture.txt
#
#    <commit> is the full 40-hex commit the run worktree is at; the driver refuses any other (B3.1).  Why a run
#    worktree, why the second half of step 2, and why never `tmux new -s NAME '<cmd>'`: P7.3d's driver header,
#    section 0 -- the same reasons hold here.
#
# 1. WHAT IT PRODUCES
#      output/p7_3c/g2/dt_reroll_check_<UTC>/            BEFORE the token: B.5-1's fenced DT re-roll (P7.3d's, Q18)
#      output/p7_3c/g2/reference_reroll_check_<UTC>/     BEFORE the token: the six reference cells re-rolled (fenced)
#      output/p7_3c/cells/canary.json                    the canary line, right after the token
#      output/p7_3c/cells/cell_cityflow_grid4x4_*.json   one chunk per cell of all three stages, atomic, resumable
#      output/p7_3c/cells/failed/                        chunks that failed their own re-check
#      output/p7_3c/cells/COMPLETE  or  .../FAILED       the terminal marker, on EVERY path after the token
#      output/p7_3c_runs/FAILED_campaign                 the marker instead, if the token was consumed and the work
#                                                        directory could not be made (Amendment H, H3.2)
#      output/p7_3c/artifacts/p7_3c_grid4x4.json         the artifact C8 commits BY HAND
#      output/SHA256SUMS_p7_3c.txt                       over output/p7_3c/ ONLY, atomic, re-verified
#    It does NOT write docs/data/.
#
# 2. THREE STAGES, ONE TOKEN (A24(c)), in this order, in one work directory:
#      p7_3c_reproduce   700 cells: P7.3d's declared set, re-rolled
#      stage1-check      AUTOMATIC, with no human step: all 700 against docs/data/p7_3d_grid4x4.json's records (its
#                        sha256 first) on every field but the six bookkeeping ones. Anything but REPRODUCED writes
#                        FAILED and stops BEFORE stage 2's first cell
#      p7_3c_primary     1,500 cells: ft_k5, ft_k20, ft_k100
#      p7_3c_controls    2,500 cells: zs_k5, zs_k20, scratch_k100, ft_k100 at B = 1,000 and 16,000 -- whatever
#                        stage 2 shows
#      report --stage p7_3c, then the manifest, then COMPLETE.
#    It prints NO outcome of any cell (Amendment B, B2): `cells` prints each cell's name and whether it stood,
#    `stage1-check` a verdict, a cell name and field NAMES, and `report` the artifact's path and its cell count.
#
# 3. ORDERING -- every check that can refuse PRECEDES the token, so a refused start consumes nothing: the commit
#    argument -> the interpreter -> not the implementer's tree -> the run tree and its commit -> the module loads from
#    it -> no live runner -> GROUP LEADER -> SigIgn -> the committed inputs exist -> the inputs BY DIGEST
#    (`check-inputs --campaign p7_3c`: P7.3d's, the zero-shot artifact, the training record, its manifest and the
#    thirty, frozen parts included and nothing under fenced_timing/) -> the dirty tree -> the RSS budget -> the layout
#    -> every chunk's commit resolvable and every chunk file a declared cell's (`resume-check`; the second is Amendment
#    H, H3.3) -> the canary, BOTH halves -> dt_reroll_check ->
#    reference_reroll_check -> the TRAPS -> the token -> record-canary -> the three stages and the gate -> report ->
#    manifest -> COMPLETE.  The two pre-token writes are the two re-roll checks' fenced records under g2/.
#
# 4-6. `-P` ON EVERY INTERPRETER CALL, THE CWD RULE (the MAIN tree, PYTHONPATH the run worktree), THE SKIP DECISION IN
#    PYTHON (chunk_is_reusable, by content, through the same calls run_cell makes): P7.3d's driver header, 4-6.
#
# 7. TIME AND MEMORY -- P7.3d's gate G2 (2026-09-20, this machine, canary 0.76 s): W = 12 at 4.88 s/cell and a tree
#    RSS of 17,297 MiB, W = 8 at 9.67 s/cell. The cell and the architecture are P7.3d's, so the budget is P7.3d's (a
#    different measurement would be a finding): 4,700 cells x 4.88-9.67 s = 6.4-12.6 h, plus the halting cross-check on
#    draw 1000's 47 cells. An ESTIMATE from P7.3d's rates, not measured on this commit; the driver prints its own wall
#    clock and the packet reports the observed one.

set -euo pipefail

EXPECTED_COMMIT=${1:-}

MAIN=/home/filip/rltraffic
# J1(e) / BRIEF_39 Amendment B.7.1-1: the tree this copy of the script lives in, never a hardcoded one.
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
IMPLEMENTER_TREE=/home/filip/rltraffic-p73c
RUN_TREE=/home/filip/rltraffic-p73c-run
PY=$MAIN/.venv/bin/python
CAMPAIGN_DIR=$MAIN/output/p7_3c
WORK=$CAMPAIGN_DIR/cells
ARTIFACTS=$CAMPAIGN_DIR/artifacts
G2_DIR=$CAMPAIGN_DIR/g2
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
TOKEN=$MAIN/output/p7_3c_runs/TOKEN_campaign
CANARY_MAX_SECONDS=2.0

# P7.3d's gate G2 (section 7): its best measured throughput, and its measured peak tree RSS x 1.4.
WORKERS=12
PEAK_TREE_RSS_MIB=17297
PER_WORKER_RSS_MIB=1441
RSS_BUDGET_MIB=24216
GPU_PEAK_MIB=4595

: "${RLTRAFFIC_GRID4X4_RESCO:=$MAIN/scenarios/grid4x4_candidates}"
export RLTRAFFIC_GRID4X4_RESCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# The roots are options of the module's PARENT parser, so every call puts them BEFORE the subcommand (P7.3d's B1).
COMMON=(--draws-root "$DRAWS" --output-root "$MAIN/output" --work-dir "$WORK"
        --data-dir "$DATA" --out-dir "$ARTIFACTS")

SUCCESS=0
MARKER_DIR=""
# Amendment H, H3.2: the token is consumed BEFORE the work directory exists, so a failure in between -- a mkdir that
# fails -- leaves FAILED beside the token instead of nowhere. Before the token nothing is written, on any path.
TOKEN_CONSUMED=0
FAILED_FALLBACK=$(dirname "$TOKEN")/FAILED_campaign

echo "=== P7.3c campaign driver: WORK_TREE $WORK_TREE (derived from this script's own location)"

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
  refuse "the argument must be the full 40-hex commit the run worktree is at (B3.1), not '$EXPECTED_COMMIT'"
fi
if [ ! -x "$PY" ]; then
  refuse "no interpreter at $PY" "The worktree has no .venv of its own; the main tree's is the one to use."
fi
if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then
  refuse "this copy of the driver is in the implementer's worktree $WORK_TREE" \
    "J1(e): the campaign runs from the DETACHED run worktree $RUN_TREE, created at the reviewed commit."
fi
if [ "$WORK_TREE" != "$RUN_TREE" ]; then
  refuse "this copy is in $WORK_TREE, not the run worktree $RUN_TREE" \
    "B3.1: the driver runs from the one tree the coordinator created for it, and from no other."
fi
HEAD_COMMIT=$(git -C "$WORK_TREE" rev-parse HEAD)
if [ "$HEAD_COMMIT" != "$EXPECTED_COMMIT" ]; then
  refuse "the run worktree is at $HEAD_COMMIT, not $EXPECTED_COMMIT" \
    "B3.1: the commit named at the start is the one every chunk records, and it must be the reviewed one."
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.transfer_curve as m; print(m.__file__)' 2>/dev/null || true)
case "$LOADED" in
  "$WORK_TREE"/*) ;;
  *)
    refuse "offline.transfer_curve loaded from '${LOADED:-nothing}', not $WORK_TREE" \
      "The cwd is the MAIN tree; the CODE must be the run worktree's, or the chunks record another commit."
    ;;
esac

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

# ---------------------------------------------------------------- the inputs
for INPUT in "$DATA/p7_3d_calibration.json" "$DATA/p7_3d_reference_cells.json" "$DATA/p7_3d_cap_e.json" \
             "$DATA/p7_3d_grid4x4.json" "$DATA/p7_3c_finetune.json"; do
  if [ ! -f "$INPUT" ]; then
    refuse "missing committed input $INPUT"
  fi
done

MISSING=0
for DRAW in $(seq 1000 1099); do
  if [ ! -f "$DRAWS/cityflow_grid4x4/draw_${DRAW}/parity/noteleport.sumocfg" ]; then
    echo "REFUSING TO START: missing parity configuration for draw $DRAW" >&2
    MISSING=$((MISSING + 1))
    [ "$MISSING" -ge 3 ] && break
  fi
done
if [ "$MISSING" -ne 0 ]; then
  refuse "the held-out band is not rendered" "P7.3d's C1 renders it; this driver writes nothing under scenarios/."
fi

# Existence is not identity: every input by digest against the module's pins, the thirty through the pinned record.
if ! PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" check-inputs --campaign p7_3c; then
  refuse "an input is not what the module pins (F3.2)" \
    "The error above names it. A campaign on moved inputs would record THEIR numbers under THIS registration."
fi

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  refuse "the worktree $WORK_TREE is DIRTY" \
    "J1(d): one untracked file makes every chunk rolled after it git_dirty: true, and report refuses them."
fi

AVAILABLE_MIB=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
if [ -n "$AVAILABLE_MIB" ] && [ "$AVAILABLE_MIB" -lt "$RSS_BUDGET_MIB" ]; then
  refuse "${AVAILABLE_MIB} MiB available, below the ${RSS_BUDGET_MIB} MiB budget" \
    "P7.3d's G2 measured ${PEAK_TREE_RSS_MIB} MiB at ${WORKERS} workers (${PER_WORKER_RSS_MIB} MiB each, GPU ${GPU_PEAK_MIB} MiB), x 1.4."
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

# Two things report or the pool would otherwise meet only AFTER the token are found here instead: a chunk whose
# git_commit git cannot resolve, and a chunk file no declared cell names (Amendment H, H3.3).
if ! PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" resume-check --stage p7_3c; then
  refuse "resume-check found a chunk that would stop the campaign after the token" \
    "The lines above name it: an undeclared chunk file, or a commit git cannot resolve. Move it aside by hand, or make its" \
    "commit reachable. Nothing has been consumed."
fi

# ---------------------------------------------------------------- the canary, BOTH halves
CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" canary) || {
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

# ---------------------------------------------------------------- the two re-roll checks, P7.3d's (Q18)
if ! REROLL_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" --canary-seconds "$CANARY" dt-reroll-check --g2-dir "$G2_DIR" --workers "$WORKERS"); then
  echo "${REROLL_LINE:-dt_reroll_check printed no result line}"
  refuse "dt_reroll_check did not report IDENTICAL (BRIEF_39 Amendment B.5-1)." \
    "The token has NOT been consumed; the fenced record is under $G2_DIR."
fi
echo "$REROLL_LINE"

if ! REFERENCE_LINES=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" --canary-seconds "$CANARY" reference-reroll-check --g2-dir "$G2_DIR" --workers "$WORKERS"); then
  echo "${REFERENCE_LINES:-reference_reroll_check printed no result line}"
  refuse "reference_reroll_check did not pass (BRIEF_39 Amendment B.7.4-1)." \
    "The token has NOT been consumed; the fenced record is under $G2_DIR."
fi
echo "$REFERENCE_LINES"
REFERENCE_RESULTS=$(grep -E '^reference_reroll_check (MATCH|NO MATCH) ' <<< "$REFERENCE_LINES" || true)
N_REFERENCE_RESULTS=$(grep -cE '^reference_reroll_check (MATCH|NO MATCH) ' <<< "$REFERENCE_LINES" || true)
N_REFERENCE_MATCH=$(grep -c '^reference_reroll_check MATCH ' <<< "$REFERENCE_LINES" || true)
if [ "$N_REFERENCE_RESULTS" -ne 6 ] || [ "$N_REFERENCE_MATCH" -ne 6 ]; then
  refuse "reference_reroll_check printed $N_REFERENCE_RESULTS result line(s), $N_REFERENCE_MATCH of them MATCH" \
    "Six MATCH lines, one per reference cell, are required (B.7.4-1). Nothing has been consumed."
fi

# ---------------------------------------------------------------- the traps, BEFORE the token
# Where FAILED goes: the work directory once it exists; before that, and only once the token is consumed, beside the
# token (H3.2); before the token, nowhere -- a refusal consumes and writes nothing.
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

trap on_exit EXIT
trap on_signal INT TERM HUP

# ---------------------------------------------------------------- the ONE token
if [ ! -f "$TOKEN" ]; then
  refuse "no run token at $TOKEN" "The author writes it (channel (a)). Nothing has been consumed."
fi
echo "=== authorised by the token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(< "$TOKEN")"
rm -f "$TOKEN"
TOKEN_CONSUMED=1
echo "=== token consumed and deleted; another start needs a new one"

# Only NOW may anything be created or cleared.
mkdir -p "$WORK" "$ARTIFACTS"
MARKER_DIR=$WORK
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

echo "P7.3c C7 -- the grid4x4 few-shot campaign (A24(c): three stages, 4,700 cells, ONE token)"
echo "  commit       $HEAD_COMMIT"
echo "  code         $LOADED"
echo "  cwd          $(pwd)   (the MAIN tree)"
echo "  stages       p7_3c_reproduce (700) -> stage1-check -> p7_3c_primary (1,500) -> p7_3c_controls (2,500)"
echo "  workers      $WORKERS"
echo "  memory       ${AVAILABLE_MIB} MiB available, budget ${RSS_BUDGET_MIB} MiB"
echo "  schedule     6.4-12.6 h (P7.3d's G2 rates at W = 12 and W = 8; an estimate)"
echo "  canary       $CANARY_LINE"
echo "  re-roll      $REROLL_LINE"
while IFS= read -r REFERENCE_LINE; do
  echo "  reference    $REFERENCE_LINE"
done <<< "$REFERENCE_RESULTS"
echo "  started      $(date -Is)"
echo ""

START=$(date +%s)

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  record-canary --line "$CANARY_LINE" || fail "record-canary"

# ---------------------------------------------------------------- stage 1, then the gate
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  --canary-seconds "$CANARY" cells --stage p7_3c_reproduce --workers "$WORKERS" || fail "cells p7_3c_reproduce"

# A24(c): all 700 reproduce P7.3d's committed records, or the campaign STOPS here -- no human step, and no stage-2
# cell exists. The module prints ONE line: a verdict, a cell name and field names, never a value.
STAGE1_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" stage1-check) || {
  echo "${STAGE1_LINE:-stage1-check printed no line}"
  fail "stage1-check"
}
echo "$STAGE1_LINE"

# ---------------------------------------------------------------- stages 2 and 3, whatever stage 2 shows
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  --canary-seconds "$CANARY" cells --stage p7_3c_primary --workers "$WORKERS" || fail "cells p7_3c_primary"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  --canary-seconds "$CANARY" cells --stage p7_3c_controls --workers "$WORKERS" || fail "cells p7_3c_controls"

# ---------------------------------------------------------------- the artifact, the manifest
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  report --stage p7_3c || fail "report"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  manifest --campaign-dir "$CAMPAIGN_DIR" || fail "manifest"

ELAPSED=$(( $(date +%s) - START ))
SUCCESS=1
printf 'CAMPAIGN COMPLETE in %ss  %s\n' "$ELAPSED" "$(date -Is)" > "$WORK/COMPLETE"
echo ""
echo "CAMPAIGN COMPLETE in ${ELAPSED}s  $(date -Is)"
echo "NEXT: the implementer copies $ARTIFACTS/p7_3c_grid4x4.json into the task branch BY HAND (C8), states its"
echo "      measured digest, and writes the packet. Nothing here is copied automatically."
