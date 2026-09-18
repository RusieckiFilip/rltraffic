#!/usr/bin/env bash
# P7.3b — the C3 curve's FULL-RETRAIN ANCHOR (PREREGISTRATION A18(a)), k = 200.
#
# Shape: offline/campaigns/p7_3a_zero_shot.sh, whose own header credits p7_2b_calibration.sh.
#
# 0. USAGE — the STAGE is an ARGUMENT, for the same reason it is one in P7.3a: no decision this
#    driver makes may depend on a number it has just seen.
#      bash offline/campaigns/p7_3b_anchor.sh anchor
#    There is exactly one campaign stage here (A18(a) declares one arm, one prompt), and the
#    argument is still required so that starting the wrong thing is a refusal rather than a
#    default.
#
# 1. WHAT IT PRODUCES
#    output/p7_3b_anchor/cell_*.json          one chunk per cell, atomic and resumable
#    output/p7_3b_anchor/failed/              chunks that failed their own re-validation
#    output/p7_3b_anchor/artifacts/p7_3b_anchor.json    the artifact section 3.4 commits
#    output/SHA256SUMS_p7_3b_anchor.txt       rewritten last, atomically, then re-verified
#
#    ⚠️ It does NOT write docs/data/. The artifact goes to $WORK/artifacts — inside the manifest
#    and outside the worktree — and is copied into the task branch BY HAND afterwards, which is
#    also when it is reviewed. P7.3a Finding 2, the author's ruling of 2026-09-17: writing it into
#    $WORK_TREE/docs/data leaves an untracked file that the dirty-tree refusal then blocks on.
#
# 2. ORDERING — every check that can refuse PRECEDES the token, so a refused start consumes
#    nothing and changes nothing. Order: interpreter → import → lock → GROUP LEADER → SigIgn →
#    stage argument → inputs → dirty tree → CANARY (both halves) → TRAP → token → work.
#    ⚠️ THE TRAP IS INSTALLED BEFORE THE TOKEN IS CONSUMED (Amendment J2/J3): a signal in that
#    window destroyed P7.2b's authorisation while leaving neither FAILED nor COMPLETE.
#
# 3. THE COLLECTION AND THE TRAINING ARE ALREADY DONE, AND THIS DRIVER KNOWS IT.
#    BRIEF_38 section 3.5 lists collect → re-collection check → train → cells → report. The first
#    three ran under CLAUDE.md section 5 in tmux panes started by the author (105 SUMO episodes on
#    2026-09-18, 5 x 40,000 gradient steps on 2026-09-18) and were verified from disk by the
#    coordinator. The branches below therefore ACCEPT what is on disk and check it rather than
#    redoing it — the same shape as P7.3a's H2 corpus branch, and for the same reason: a stage
#    that already succeeded must not die on a restart.
#    ⛔ NOTHING BELOW DELETES OR OVERWRITES A CORPUS, A CHECKPOINT OR A CHUNK. `--overwrite` is
#    never passed. A corpus or a checkpoint this driver cannot vouch for is named and the run
#    stops; moving it aside is a human decision.
#
# 4. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL.
#    offline/transfer_curve.py::chunk_is_reusable re-derives a chunk's verdict from its own content
#    AND from the files on disk. ⛔ THERE ARE DELIBERATELY NO `[ -f ]` GUARDS OVER CHUNKS below;
#    offline/campaigns/p5_3b.sh had one and a bad chunk survived every restart.
#
# 5. NOTHING UNDER scenarios/draws/ IS WRITTEN. P7.2a's parity configs are opened read-only.
#
# 6. TIME — THIS TASK'S OWN PILOT, measured 2026-09-18 (F3), with its canary and its n.
#    WHERE IT RAN: /home/filip/rltraffic-p73b, `git status --porcelain` EMPTY, detached at
#    5849d59a799400c807f8d3205084e043bf41f9f6 — and all four chunks record that commit with
#    `git_dirty: false`, which is checked rather than asserted.
#    WHAT IT RAN: `pilot --anchor --workers 12` — anchor_pilot_cells(), 4 FENCED cells on draw 5
#    (P7.2b's smoke draw, NOT in the held-out pool): the anchor at seeds 101 and 202, plus both
#    rho denominators, which A6 measured SLOWER than a DT cell. n = 2 runs x 4 cells.
#    Transcripts: output/p7_3b_runs/preflight_pilot_anchor{,_resumed}.txt, capture
#    preflight_pilot_capture.txt.
#
#      clean run     canary 0.76 s   wall 19.53 s   4.883 s/cell   in-process mean 17.02 s  3.49x
#      resumed run   canary 0.80 s   4 of 4 REUSED, 0 rolled, 0 failed — no rate, correctly null
#      0 failures in 4 cells; halting check OFF (draw 5 is not Amendment C2's declared subset)
#
#    A11's LABEL, and it travels with every figure above: measured on the thermally constrained
#    laptop, no cooling pad, on mains, with a 0.76 s canary in the same command. Both canaries are
#    far below the 2.0 s threshold, so this is a CLEAN-machine rate — unlike P7.3a's stage 2.
#
#    ⚠️ THE SCHEDULE IS **NOT** 700 x 4.883 s. Four cells cannot saturate twelve workers: the pool
#    runs all four at once, so the wall is roughly ONE cell's time and `4.883 s/cell` is
#    wall/4 — an OVERSTATEMENT of the per-cell cost at scale, not a rate. The honest basis is the
#    in-process mean divided by the speed-up a FULL pool achieves, and P7.3a measured that twice
#    on this machine at these settings: 1,200 cells at 7.36x and 3,500 cells at 11.92x.
#      700 cells x 17.02 s / 11.92  ≈  1,000 s ≈ 17 min   (optimistic, the larger stage's scaling)
#      700 cells x 17.02 s /  7.36  ≈  1,619 s ≈ 27 min   (conservative, the smaller stage's)
#    ⇒ EXPECT 17–27 MINUTES. The driver prints its own wall clock and writes COMPLETE; that
#    number, not this one, is what the packet reports.
#
#    ⚠️ F3: the comparison basis is THIS pilot's canary, 0.76 s. If the machine's canary at the
#    campaign's start differs from it by more than 10 %, this block is replaced by a re-run before
#    the token — the header must never quote a rate for a machine state the run did not have. The
#    driver re-runs the canary at its own start and refuses above 2.0 s regardless.
#
#    FOUND BY RUNNING THIS PILOT, and it is why a pre-flight is not a formality: run_pilot's
#    transcript did `int(cell["seed"])` over every cell, and an anchor cell has seed None. All
#    four cells rolled and the SUMMARY then raised TypeError, losing the rate. F1's sixteen cells
#    are all `dt`, so nothing had noticed. Fixed, and pinned by a test.
#
# 7. THE WORKTREE HAS NO .venv. The interpreter is the main tree's, as P7.2b's and P7.3a's are.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p7_3b_anchor
LOGS=$WORK/logs
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
ARTIFACTS=$WORK/artifacts
CKPT=$WORK/checkpoints
CORPUS_A=$MAIN/datasets_sumo_v11/hz1x1_sumo_maxpressure
CORPUS_B=$MAIN/datasets_sumo_v11/hz1x1_sumo_maxpressure_301_400
RECHECK=$MAIN/output/p7_3b_recollect_check/scratch_301_400
# Amendment C3: measured 1.27x better than 8 on this machine's 16 cores.
WORKERS=12
CANARY_MAX_SECONDS=2.0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

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
# Measured under J3: a plain `&` from a non-interactive shell DID lead its group; what produced a
# non-leader was `set +m`, job control switched off.
if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux foreground pane" >&2
  echo "  The handler kills the process group, and that is a no-op from a non-leader, so an" >&2
  echo "  interrupted run would leave the worker pool writing. Nothing consumed." >&2
  exit 2
fi

# Amendment J3: a shell that STARTS with SIGINT ignored cannot trap it -- bash does not let a
# non-interactive shell trap a signal that was ignored on entry -- so Ctrl-C would do nothing at
# all while the pool kept writing and the token was already gone. Signal 2's bit is 0x2.
SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  echo "REFUSING TO START: SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" >&2
  echo "  bash cannot trap a signal that was ignored on entry, so the trap below would be a" >&2
  echo "  no-op and Ctrl-C could not stop an interrupted run. Start the driver from a shell" >&2
  echo "  that does not ignore SIGINT -- a tmux foreground pane does not. Nothing consumed." >&2
  exit 2
fi

STAGE=${1:-}
case "$STAGE" in
  anchor) ;;
  *)
    echo "REFUSING TO START: the stage must be 'anchor', got '${STAGE}'" >&2
    echo "  A18(a) declares one arm and one prompt for this task, so there is one campaign" >&2
    echo "  stage; the argument is required so that starting the wrong thing is a refusal." >&2
    exit 2
    ;;
esac

# Amendment J1(d)/(e): a multi-hour stage must not run from a tree that is being edited. ONE
# untracked file makes every chunk rolled after it `git_dirty: true`, and validate_cell_payload
# then refuses those cells one at a time, hours in.
WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "REFUSING TO START: the worktree $WORK_TREE is DIRTY" >&2
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  echo "  Every cell rolled from this tree would record git_dirty: true, which report refuses" >&2
  echo "  one cell at a time. Run the campaign from a dedicated worktree detached at the" >&2
  echo "  reviewed commit (Amendment J1(e)). Nothing has been consumed." >&2
  exit 2
fi

# The pool's ends and the anchor's training band's ends. Not every draw: 100 stat() calls before a
# token is noise, and a missing middle draw refuses inside the cell that needs it, naming it.
for draw in 201 300 301 400 1000 1099; do
  cfg=$DRAWS/cityflow1x1/draw_$(printf '%04d' "$draw")/parity/noteleport.sumocfg
  if [ ! -f "$cfg" ]; then
    echo "REFUSING TO START: P7.2a's parity configuration is missing: $cfg" >&2
    exit 2
  fi
done

# Section 3.3's inputs: both corpus halves, each with the digest file section 3.2 wrote.
for d in "$CORPUS_A" "$CORPUS_B"; do
  for f in manifest.json SHA256SUMS; do
    if [ ! -f "$d/$f" ]; then
      echo "REFUSING TO START: $d/$f is missing -- section 3.2's corpus is incomplete" >&2
      exit 2
    fi
  done
done
if [ ! -d "$RECHECK" ]; then
  echo "REFUSING TO START: the five-draw re-collection check's scratch corpus is missing:" >&2
  echo "  $RECHECK" >&2
  echo "  There is no A17(f) for the 301-400 band; that bit-for-bit comparison is its only" >&2
  echo "  integrity evidence, and this driver will not run cells without it on disk." >&2
  exit 2
fi

# The anchor's five checkpoints and its committed record.
if [ ! -f "$DATA/p7_3b_anchor_training.json" ]; then
  echo "REFUSING TO START: $DATA/p7_3b_anchor_training.json is missing" >&2
  echo "  The anchor's prompt and every checkpoint digest are pinned against it (G1)." >&2
  exit 2
fi
for seed in 101 202 303 404 505; do
  if [ ! -f "$CKPT/anchor_dt_seed${seed}.pt" ]; then
    echo "REFUSING TO START: the anchor's checkpoint is missing: $CKPT/anchor_dt_seed${seed}.pt" >&2
    echo "  Run output/p7_3b_runs/train_anchor.sh first (section 3.3, in tmux)." >&2
    exit 2
  fi
done
# The P4 subjects' checkpoints are NOT required here: this stage evaluates the anchor only.

# ---------------------------------------------------------------- the canary
# PROJECT_PLAN section 7; recipe BRIEF_36 section 3.3. BOTH halves: the timing says the machine is
# at speed, and check_canary says the ENGINE still computes what it computed when the references
# were measured. ⚠️ THE `-a` IS LOAD-BEARING: plain `tee /dev/stderr` re-opens stderr with O_TRUNC,
# so under `>> log 2>&1` the open resets the file offset. P7.2b lost run 3's capture that way.
echo "=== canary (PROJECT_PLAN section 7; recipe BRIEF_36 section 3.3)"
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
# HUP as well as INT and TERM (Amendment J2). A closed tmux pane or a dropped SSH session sends
# exactly SIGHUP, and it was not trapped: measured, a HUP during A17(f) exited 129 with the token
# consumed and NEITHER FAILED NOR COMPLETE.
trap on_signal INT TERM HUP

# ---------------------------------------------------------------- the token
TOKEN=$WORK/AUTHORISED_TO_RUN
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run authorisation token at $TOKEN" >&2
  echo "  The author authorises one run with:" >&2
  echo "    mkdir -p $WORK && date -Is > $TOKEN" >&2
  echo "  The token is deleted on start, so it authorises exactly one run." >&2
  exit 2
fi
echo "=== authorised by token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
rm -f "$TOKEN"
echo "=== token consumed and deleted; a restart needs a new one"

# Only NOW may anything be created or cleared.
mkdir -p "$LOGS" "$ARTIFACTS"
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

# Amendment E1.2: the canary's observed values reach a MANIFESTED file, not only the pane.
# Appended, never truncated -- and written here, AFTER the token, so "a refused start creates
# nothing" still holds.
echo "=== canary  stage $STAGE  run at $(date -Is)" >> "$LOGS/canary.log"
echo "$CANARY_LINE" >> "$LOGS/canary.log"

# Amendment E1.4: and into a MACHINE-READABLE record `report` reads back and re-checks, because a
# log is only ever read by a human. Without it `report` built the canary block from the CHUNKS --
# which a re-roll reuses unchanged -- so P7.2b's run 3 published run 1's canary as its own.
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve \
  "${COMMON[@]}" record-canary --line "$CANARY_LINE" || fail "record-canary"

START=$(date +%s)
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

# ---------------------------------------------------------------- inputs, re-derived not trusted
# Amendment I3(3): when a stage has more than one path into it, the path taken is written as the
# FIRST line of that run's entry. Here every input already exists (section 3, above), so the note
# records that this run ACCEPTED them rather than produced them -- and the digests are re-verified
# from the bytes on disk by load_anchor_corpus's own check, reached through the dry run below.
STAGE_NOTE="branch: corpus, re-collection check and checkpoints ALL PRESENT -- accepted and re-verified by this run, not produced by it"
run_stage verify_inputs offline.anchor_training \
  --dataset-dir "$CORPUS_A" --dataset-dir "$CORPUS_B" \
  --output-root "$MAIN/output" --checkpoint-dir "$CKPT" \
  --out-path "$ARTIFACTS/unused_dry_run.json" --dry-run
STAGE_NOTE=""

# ---------------------------------------------------------------- the evaluation pool
run_stage "cells_$STAGE" offline.transfer_curve "${COMMON[@]}" \
  --canary-seconds "$CANARY" cells --stage "$STAGE" --workers "$WORKERS"

run_stage report_anchor offline.transfer_curve "${COMMON[@]}" \
  report --stage "$STAGE"

# ---------------------------------------------------------------- manifest
# Rewritten over the whole of p7_3b_anchor/, so it covers the checkpoints section 3.3 wrote AND
# the chunks and artifact this run wrote, then re-verified.
( cd "$MAIN/output" \
  && find p7_3b_anchor -type f \( -name '*.pt' -o -name '*.json' -o -name '*.log' \) -print \
     | LC_ALL=C sort | xargs sha256sum > SHA256SUMS_p7_3b_anchor.txt.tmp \
  && mv SHA256SUMS_p7_3b_anchor.txt.tmp SHA256SUMS_p7_3b_anchor.txt ) || fail "manifest write"
( cd "$MAIN/output" && sha256sum -c SHA256SUMS_p7_3b_anchor.txt --quiet ) || fail "manifest verify"

ELAPSED=$(( $(date +%s) - START ))
echo "CAMPAIGN COMPLETE ($STAGE) in ${ELAPSED}s" | tee "$WORK/COMPLETE"
echo "=== canary was $CANARY s; artifact at $ARTIFACTS/p7_3b_anchor.json"
echo "=== NEXT (by hand, and reviewed there):"
echo "    cp $ARTIFACTS/p7_3b_anchor.json $WORK_TREE/docs/data/"
