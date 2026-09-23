#!/usr/bin/env bash
# P7.3d — C6: the grid4x4 ZERO-SHOT CAMPAIGN, one declared stage of 700 cells (PREREGISTRATION A21).
#
# Shape: offline/campaigns/p7_3b_anchor.sh, with the two P7.3d drivers' corrections folded in, and
# the B.6 fix round's (BRIEF_39 Amendment B.6; docs/reviews/P7.3d-preflight.md) -- see section 8.
#
# 0. USAGE — the FOREGROUND form, and it is not the one-liner (BRIEF_39 Amendment B.3-1).
#
#      Step 1, open a pane:      tmux new -s p73d_cells
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p73d/offline/campaigns/p7_3d_grid4x4.sh confirmatory 2>&1 | tee -a /home/filip/rltraffic/output/p7_3d_runs/campaign_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -a /home/filip/rltraffic/output/p7_3d_runs/campaign_capture.txt
#
#    THE SECOND HALF OF STEP 2 (Amendments B.5-3 and B.6-2(6)): the pipeline's own status is
#    tee's, so without it the pane reports 0 even when the driver refuses. `${PIPESTATUS[0]}` is
#    expanded when the echo runs, after the first pipeline has finished, so it is the DRIVER's exit
#    status; the echo is teed as well, so the capture's last line and the pane's agree. The author
#    still reads the capture's last lines and the COMPLETE / FAILED file (section 1).
#
#    ⛔ WHY NOT `tmux new -s NAME '<cmd>'`: that form runs the command through a non-interactive
#    shell with job control OFF, so this script does NOT lead its own process group and the guard
#    below refuses (measured 2026-09-21; the two refusals at the top of g2_capture.txt are exactly
#    this). The guard exists so Ctrl-C reaches the worker pool, and the driver deliberately does
#    NOT re-exec itself under `setsid` to dodge it — that would silently change which process the
#    author's signal lands on. T-driver EXECUTES step 2's line through tmux and asserts the
#    group-leader check passes and the pane's last line carries this driver's exit code.
#
# 1. WHAT IT PRODUCES
#      output/p7_3d/g2/dt_reroll_check_<UTC>/              BEFORE the token: B.5-1's thirteen fenced
#                                                          rolls of one draw-5 DT cell + verdict.json
#      output/p7_3d/cells/canary.json                      the canary line, right after the token
#      output/p7_3d/cells/cell_cityflow_grid4x4_*.json     one chunk per cell, atomic, resumable
#      output/p7_3d/cells/failed/                          chunks that failed their own re-check
#      output/p7_3d/cells/COMPLETE  or  .../FAILED         the terminal marker, on EVERY path after
#                                                          the token (a signal writes FAILED too)
#      output/p7_3d/artifacts/p7_3d_grid4x4.json           the artifact C7 commits BY HAND
#      output/SHA256SUMS_p7_3d.txt                         over output/p7_3d/ ONLY, rewritten last,
#                                                          atomically (tmp -> mv), re-verified
#    ⚠️ It does NOT write docs/data/. P7.3a's Finding 2: writing into $WORK_TREE/docs/data leaves
#    an untracked file that the dirty-tree refusal then blocks on.
#
# 2. ONE STAGE, ONE TOKEN (A21(a)). 700 cells = b_mean_k100 x 5 seeds x 100 draws (500) + fixedtime
#    (100) + maxpressure (100). There is no `naive` arm and no `random` anchor on this scenario:
#    A21(b) removed them BY DECLARATION before any grid4x4 SUMO number existed, and the paper
#    reports that as scope in A21's own words. The stage argument is still required, so starting
#    the wrong thing is a refusal rather than a default.
#
# 3. ORDERING — every check that can refuse PRECEDES the token, so a refused start consumes
#    nothing: interpreter → import from the worktree → no live runner → GROUP LEADER → SigIgn →
#    stage argument → inputs exist → inputs BY DIGEST (m1) → dirty tree → RSS budget → every chunk's
#    commit resolvable by git (m2) → canary, BOTH halves → dt_reroll_check (B.5-1) → TRAP → token →
#    record-canary → cells → report → manifest → COMPLETE. The trap is installed BEFORE the token
#    is consumed (J2/J3): a signal in that window destroyed P7.2b's authorisation while leaving
#    neither FAILED nor COMPLETE. The ONE pre-token write is dt_reroll_check's fenced record in g2/,
#    which B.5-1 orders; nothing under cells/ or artifacts/ exists before the token.
#
# 4. `-P` ON EVERY INTERPRETER CALL (Amendment B.3-2). Without it Python prepends the cwd — the
#    MAIN tree, by rule 5 — to sys.path, `offline` resolves to the main tree's package, and the
#    import fails. That is what made G2's first hand-over refuse (1066aff).
#
# 5. THE CWD RULE (Amendment B3/Q4): cd to the MAIN tree so any render embeds the main tree's
#    absolute `dir` (Amendment A6, DEFERRED 87), with PYTHONPATH pointing at the task worktree and
#    an assertion of WHICH file loaded. Nothing under scenarios/draws/ is written by this driver;
#    the parity configs are opened read-only.
#
# 6. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL. transfer_curve.chunk_is_reusable re-derives
#    a chunk's verdict from its own content and from the files on disk -- the demand, the checkpoint
#    and the 16 prompts through the SAME scenario-aware calls run_cell makes (B.6-2(2)). ⛔ There
#    are deliberately no `[ -f ]` guards over chunks; offline/campaigns/p5_3b.sh had one and a bad
#    chunk survived every restart.
#
# 7. TIME AND MEMORY — from gate G2, measured 2026-09-20 at commit 07ab267, canary 0.76 s, on this
#    machine (16 cores, 48,174 MiB host, RTX 5080 Laptop 16,303 MiB):
#      one cell, single worker, halting OFF   35.5 s in-process, 1,280 MiB
#      W = 4    11.67 s/cell   3.89x    tree RSS  6,562 MiB
#      W = 8     9.67 s/cell   7.81x    tree RSS 11,856 MiB
#      W = 12    4.88 s/cell  11.39x    tree RSS 17,297 MiB, GPU 4,595 MiB
#    ⚠️ The W = 8 row is SLOWER PER CELL than W = 12 and that is UNEXPLAINED (the spread inside
#    each pool was under 0.5 s, so it is systematic). Amendment B.3-3: the schedule is therefore a
#    RANGE, not a point — 700 x 4.88 s ≈ 57 min at the W = 12 rate to 700 x 9.67 s ≈ 113 min at the
#    W = 8 rate — plus draw 1000's 7 cells carrying the halting cross-check, which roughly doubles
#    an episode. The driver prints its own wall clock and the packet reports the observed one.
#    The single-worker figure is corroborated by C5's sequential MaxPressure episode at 29.9 s.
#    BEFORE the token, dt_reroll_check adds about 95 s: one DT cell alone (G2: 35.5 s) and then
#    twelve copies in one 12-worker pool (G2's W = 12 pool: 58.6 s wall).
#
#    DEMAND BASIS (Amendment B.2-3b), because SUMO wall time scales with vehicle count and the
#    schedule above was measured on draw 5: draw 5 carries 1,335 vehicles against the held-out
#    band's mean of 1,327.6 (sd 12.3, range 1,298-1,358) — +0.56 %, z = +0.60, inside the band's
#    own spread. Recomputed from the flow files 2026-09-22. RSS is insensitive to a 0.6 % demand
#    difference; the schedule inherits it and says so.
#
# 8. THE B.6 FIX ROUND (gate G4 NOT CLEAR). The pre-flight found the SHAPE right and the WIRING
#    wrong, and every piece below is now executed by tests/test_p7_3d_campaign_path.py through
#    THIS file, redirected, on a clean snapshot of the tree:
#    B1  the roots ("${COMMON[@]}") are options of the module's PARENT parser, so they come BEFORE
#        the subcommand; after it, argparse refused at the canary on every machine.
#    B3  report names its artifact p7_3d_grid4x4.json, the grid4x4 stage is its own whole
#        declaration, and the manifest is a real subcommand.
#    M1  FAILED / COMPLETE files on every terminal path after the token (p7_3b_anchor.sh's shape).
#    m1  the three committed inputs, the five checkpoints and RESCO's network BY DIGEST.
#    m2  a chunk whose commit git cannot resolve refuses BEFORE the token, not inside the pool.
#    F-B6-1  `cells` receives --canary-seconds: without it every chunk recorded canary_seconds
#            None, and report raises on the first such chunk -- after 700 cells.
#    F-B6-2  the canary's TIMING half (2.0 s, p7_3b_anchor.sh's) was not checked at all.
#    B.5-1   dt_reroll_check: IDENTICAL or the token is not consumed.
#    B.5-3   the pane carries the driver's exit code (section 0).

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=/home/filip/rltraffic-p73d
PY=$MAIN/.venv/bin/python
CAMPAIGN_DIR=$MAIN/output/p7_3d
WORK=$CAMPAIGN_DIR/cells
ARTIFACTS=$CAMPAIGN_DIR/artifacts
G2_DIR=$CAMPAIGN_DIR/g2
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
TOKEN=$MAIN/output/p7_3d_runs/TOKEN_confirmatory
CALIBRATION=$DATA/p7_3d_calibration.json
REFERENCE_CELLS=$DATA/p7_3d_reference_cells.json
CAP_E=$DATA/p7_3d_cap_e.json
CANARY_MAX_SECONDS=2.0

# From gate G2 (§7). WORKERS is its best measured throughput; the budget is its measured peak
# tree RSS x 1.4, and the per-worker figure is that peak divided by the workers.
WORKERS=12
PEAK_TREE_RSS_MIB=17297
PER_WORKER_RSS_MIB=1441
RSS_BUDGET_MIB=24216
GPU_PEAK_MIB=4595

: "${RLTRAFFIC_GRID4X4_RESCO:=$MAIN/scenarios/grid4x4_candidates}"
export RLTRAFFIC_GRID4X4_RESCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ⚠️ B1: these are options of the module's PARENT parser. Every call below puts them BEFORE the
# subcommand (p7_3b_anchor.sh:230's shape); after it, argparse exits 2 with "unrecognized arguments".
COMMON=(--draws-root "$DRAWS" --output-root "$MAIN/output" --work-dir "$WORK"
        --data-dir "$DATA" --out-dir "$ARTIFACTS")

# ---------------------------------------------------------------- preconditions
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  echo "  The worktree has no .venv of its own; the main tree's is the one to use." >&2
  exit 2
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.transfer_curve as m; print(m.__file__)' 2>/dev/null || true)
case "$LOADED" in
  "$WORK_TREE"/*) ;;
  *)
    echo "REFUSING TO START: offline.transfer_curve loaded from '${LOADED:-nothing}', not $WORK_TREE" >&2
    echo "  The cwd is the MAIN tree (§5); the CODE must still be the task worktree's, or this" >&2
    echo "  campaign is rolled by another commit than the one its chunks will record." >&2
    exit 2
    ;;
esac

ALIVE=$(pgrep -f 'python.*offline\.(transfer_curve|transfer_calibration|g2_measure)' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: cells from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo '  Wait until `pgrep -f transfer_curve` is empty. Nothing has been consumed.' >&2
  exit 3
fi

if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux FOREGROUND pane" >&2
  echo "  Open a pane with 'tmux new -s p73d_cells', then type the command at its prompt (§0)." >&2
  echo "  The one-line 'tmux new -s NAME <cmd>' form runs a non-interactive shell with job" >&2
  echo "  control off, and this handler kills the process group, which is a no-op from a" >&2
  echo "  non-leader -- an interrupted run would leave the pool writing. Nothing consumed." >&2
  exit 2
fi

SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  echo "REFUSING TO START: SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" >&2
  echo "  bash cannot trap a signal ignored on entry, so Ctrl-C could not stop the pool." >&2
  exit 2
fi

STAGE=${1:-}
# ⚠️ The user-facing argument is `confirmatory`; the MODULE's stage name is `grid4x4_confirmatory`,
# because hz1x1 already has a `confirmatory` stage of 1,200 cells and passing the bare word through
# would have selected THAT declaration -- 1,200 hangzhou cells rolled under a grid4x4 token. Found
# by running the driver's own documented form before it was committed. The two names are mapped
# here, once, and a test asserts the mapped value is what reaches the module.
STAGE_ARG=grid4x4_confirmatory
case "$STAGE" in
  confirmatory) ;;
  *)
    echo "REFUSING TO START: the stage must be 'confirmatory', got '${STAGE}'" >&2
    echo "  A21(a) declares ONE stage of 700 cells for this scenario; the argument is required" >&2
    echo "  so that starting the wrong thing is a refusal rather than a default." >&2
    exit 2
    ;;
esac

for INPUT in "$CALIBRATION" "$REFERENCE_CELLS" "$CAP_E"; do
  if [ ! -f "$INPUT" ]; then
    echo "REFUSING TO START: missing committed input $INPUT" >&2
    exit 2
  fi
done

for SEED in 101 202 303 404 505; do
  CKPT=$MAIN/output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed${SEED}.pt
  if [ ! -f "$CKPT" ]; then
    echo "REFUSING TO START: missing checkpoint $CKPT" >&2
    echo "  A20(a) registers the subject by five digests; the loader checks them." >&2
    exit 2
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
  echo "  Gate G1 recorded 200/200 exact in docs/data/p7_3d_cap_e.json; the band is rendered by C1." >&2
  exit 2
fi

# m1 (B.6-2(4)): existence is not identity. Each input is hashed against the pin in the module --
# the three committed artifacts, A20(a)'s five checkpoints, and RESCO's network (Amendment A8).
if ! PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" check-inputs; then
  echo "REFUSING TO START: an input is not at the digest the module pins (B.6-2(4), m1)" >&2
  echo "  The error above names each one. A campaign run against moved inputs would record THEIR" >&2
  echo "  numbers under THIS registration. Nothing has been consumed." >&2
  exit 2
fi

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "REFUSING TO START: the worktree $WORK_TREE is DIRTY" >&2
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  echo "  J1(d): ONE untracked file makes every chunk rolled after it git_dirty: true, and" >&2
  echo "  validate_cell_payload then refuses those cells one at a time, hours in." >&2
  exit 2
fi

AVAILABLE_MIB=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
if [ -n "$AVAILABLE_MIB" ] && [ "$AVAILABLE_MIB" -lt "$RSS_BUDGET_MIB" ]; then
  echo "REFUSING TO START: ${AVAILABLE_MIB} MiB available, below the ${RSS_BUDGET_MIB} MiB budget" >&2
  echo "  Gate G2 measured a peak of ${PEAK_TREE_RSS_MIB} MiB at ${WORKERS} workers" >&2
  echo "  (${PER_WORKER_RSS_MIB} MiB per worker, GPU ${GPU_PEAK_MIB} MiB); the budget is that peak" >&2
  echo "  x 1.4. A pool that runs out of memory hours in loses the cells it was rolling." >&2
  exit 2
fi

# m2 (B.6-2(4)): a chunk whose git_commit git cannot resolve makes chunk_is_reusable RAISE (J1(c),
# deliberately). Inside the pool that happened AFTER `rm -f "$TOKEN"`; it is found here instead.
if ! PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" resume-check --stage "$STAGE_ARG"; then
  echo "REFUSING TO START: a chunk on disk records a commit git cannot resolve (B.6-2(4), m2)" >&2
  echo "  The line above names it. Inside the pool it would raise AFTER the token is consumed." >&2
  echo "  Move it aside by hand, or make its commit reachable, then start again. Nothing consumed." >&2
  exit 2
fi

# ---------------------------------------------------------------- the canary, BOTH halves
# The correctness half runs in the module (check_canary: the ENGINE still reproduces draw 0); the
# TIMING half is here, p7_3b_anchor.sh:239-245's shape (F-B6-2: this driver had no timing half).
CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" canary) || {
  echo "${CANARY_LINE:-the canary printed no line}"
  echo "REFUSING TO START: the canary failed its correctness half. NOTHING has been consumed." >&2
  echo "  The observed line is above: a correctness failure means the ENGINE did not reproduce" >&2
  echo "  draw 0, which is a finding about the engine and not a rate question." >&2
  exit 2
}
echo "$CANARY_LINE"
CANARY=$(echo "$CANARY_LINE" | awk '{print $2}')
if awk -v c="$CANARY" -v m="$CANARY_MAX_SECONDS" 'BEGIN { exit !(c > m) }'; then
  echo "REFUSING TO START: canary $CANARY s exceeds $CANARY_MAX_SECONDS s -- the machine is" >&2
  echo "  throttled and no rate measured here would be a rate. Check mains power and the" >&2
  echo "  cooling pad, then start again. Nothing has been consumed." >&2
  exit 2
fi

# ---------------------------------------------------------------- B.5-1: the DT re-roll check
# One fenced DT cell (seed 101, draw 5, the registered prompt, reset(seed=1000)) rolled once at
# W = 1 and twelve times in one 12-worker pool, compared under == with the clock excluded by name.
# It prints ONE line -- IDENTICAL or NOT IDENTICAL, field names, digests, no value -- and writes its
# fenced record under g2/, never under cells/. Anything but IDENTICAL: the token is NOT consumed.
if ! REROLL_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" --canary-seconds "$CANARY" dt-reroll-check --g2-dir "$G2_DIR" --workers "$WORKERS"); then
  echo "${REROLL_LINE:-dt_reroll_check printed no result line}"
  echo "REFUSING TO START: dt_reroll_check did not report IDENTICAL (Amendment B.5-1)." >&2
  echo "  A DT cell that does not reproduce under the campaign's own pooling is a property of the" >&2
  echo "  instrument the coordinator rules on (channel (c)) BEFORE 700 cells, not after. The" >&2
  echo "  token has NOT been consumed; the fenced record is under $G2_DIR." >&2
  exit 2
fi
echo "$REROLL_LINE"

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

on_signal() {
  trap '' INT TERM HUP
  if [ -d "$WORK" ]; then
    echo "CAMPAIGN INTERRUPTED by a signal" | tee "$WORK/FAILED" >&2
  else
    echo "CAMPAIGN INTERRUPTED by a signal (before any work directory existed)" >&2
  fi
  echo "  Every completed chunk is on disk; running this script again resumes from them by" >&2
  echo "  CONTENT (section 6). Killing the process group." >&2
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM HUP

# ---------------------------------------------------------------- the token
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run token at $TOKEN" >&2
  echo "  The author writes it (CLAUDE.md §5, channel (a)). Nothing has been consumed." >&2
  exit 2
fi
echo "=== authorised by the token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
rm -f "$TOKEN"
echo "=== token consumed and deleted; another start needs a new one"

# Only NOW may anything be created or cleared.
mkdir -p "$WORK" "$ARTIFACTS"
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

echo "P7.3d C6 — the grid4x4 zero-shot campaign (A21: ONE stage, 700 cells)"
echo "  commit       $(git -C "$WORK_TREE" rev-parse HEAD)"
echo "  code         $LOADED"
echo "  cwd          $(pwd)   (the MAIN tree: Amendment A6's render rule)"
echo "  stage        $STAGE   (500 dt + 100 fixedtime + 100 maxpressure)"
echo "  workers      $WORKERS   (gate G2's best measured throughput on 16 cores)"
echo "  memory       ${AVAILABLE_MIB} MiB available, budget ${RSS_BUDGET_MIB} MiB"
echo "  schedule     57-113 min (G2's W=12 and W=8 rates; the W=8 row is unexplained)"
echo "  canary       $CANARY_LINE"
echo "  re-roll      $REROLL_LINE"
echo "  started      $(date -Is)"
echo ""

START=$(date +%s)

# Amendment E1.4: the canary's observed values reach a MANIFESTED, machine-readable record right
# after the token; report reads it back and re-checks it rather than trusting the chunks' copy.
PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  record-canary --line "$CANARY_LINE" || fail "record-canary"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  --canary-seconds "$CANARY" cells --stage "$STAGE_ARG" --workers "$WORKERS" || fail "cells"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  report --stage "$STAGE_ARG" || fail "report"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_curve "${COMMON[@]}" \
  manifest --campaign-dir "$CAMPAIGN_DIR" || fail "manifest"

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "CAMPAIGN COMPLETE ($STAGE) in ${ELAPSED}s  $(date -Is)" | tee "$WORK/COMPLETE"
echo "NEXT: the implementer copies $ARTIFACTS/p7_3d_grid4x4.json into the task branch BY HAND,"
echo "      states its measured digest, and writes the packet. Nothing here is copied automatically."
