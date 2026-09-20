#!/usr/bin/env bash
# P7.3d — GATE G2: the one-cell measurement (BRIEF_39 §3 C5, Amendment B.2).
#
# Shape: offline/campaigns/p7_3b_anchor.sh, minus everything that does not apply to a gate that
# consumes no token and writes no cell.
#
# 0. USAGE — from a tmux pane the AUTHOR starts (CLAUDE.md §5; channel (a)):
#      tmux new -s p73d_g2 'bash /home/filip/rltraffic-p73d/offline/campaigns/p7_3d_g2.sh 2>&1 | tee -a /home/filip/rltraffic/output/p7_3d_runs/g2_capture.txt'
#    No argument. No token: G2 consumes no authorisation because it produces no cell and no
#    number — it is the measurement that lets a schedule be written at all.
#
# 1. WHAT IT PRODUCES, AND WHAT IT CANNOT
#      output/p7_3d/g2/g2_measurement.json   the record, ENTIRELY under `fenced_do_not_report`
#      scenarios/draws/cityflow_grid4x4/draw_0005/parity/   rendered by C1's own tool if absent
#    ⛔ It writes no cell chunk, touches no campaign work directory, and reports NO OUTCOME:
#    no ATT, no e_sumo, no return, no return-to-go, no action, from any cell, on any draw
#    (Amendment B.2-2(2)). `g2/` is a directory no `report` globs.
#
# 2. WHERE IT RUNS — draw 5, the smoke draw, OUTSIDE the held-out pool 1000–1099 and outside the
#    probe band 201–300 (Amendment B.2-2(1)). §3 C5's "draw 1000, the naive prompt" predates A21
#    and was corrected by B.2: the DT cell here runs under the CHECKPOINT'S OWN in-domain prompt,
#    which is the only prompt that exists before the SUMO probe, and it is a TIMING cell.
#
# 3. THE CWD RULE, CARRIED BY THIS SCRIPT (Amendment B3/Q4)
#    The rendered CityFlow parents embed an absolute `dir` resolved against the process working
#    directory, so every grid4x4 render must run with the cwd in the MAIN tree (Amendment A6,
#    DEFERRED 87) while the CODE is the task worktree's. Hence: cd $MAIN, PYTHONPATH=$WORK_TREE,
#    and an assertion that the module actually loaded from the worktree.
#
# 4. ORDERING — every check that can refuse precedes any measurement, so a refused start changes
#    nothing: interpreter → import from the worktree → no live runner → group leader → SigIgn →
#    inputs (RESCO root, checkpoint, draws tree, sim config) → dirty tree → trap → work.
#    The canary (both halves) is the FIRST thing the Python does, before any grid4x4 work.
#
# 5. TIME — UNKNOWN, which is why this gate exists. The only grid4x4 SUMO data point in the
#    project is T-obs (2026-09-19): one fixed-time episode on draw 1000 with the halting check ON
#    took 108 s, and the same test with the check over 8 lanes took 53 s for the whole pytest
#    process. NEITHER IS A RATE — no canary was taken beside them, the check is OFF here, and the
#    DT cell's cost is not the fixed-time cell's. Expect roughly: 2 single cells + 24 pooled cells.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=/home/filip/rltraffic-p73d
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p7_3d
DRAWS=$MAIN/scenarios/draws
ENV_CONFIG=$WORK_TREE/configs/sim/cityflow_grid4x4.json
DRAW=5
SEED=101
POOL_SIZES="4 8 12"
: "${RLTRAFFIC_GRID4X4_RESCO:=$MAIN/scenarios/grid4x4_candidates}"
export RLTRAFFIC_GRID4X4_RESCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---------------------------------------------------------------- preconditions
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  echo "  The worktree has no .venv of its own; the main tree's is the one to use." >&2
  exit 2
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -c 'import offline.g2_measure as m; print(m.__file__)' 2>/dev/null || true)
case "$LOADED" in
  "$WORK_TREE"/*) ;;
  *)
    echo "REFUSING TO START: offline.g2_measure loaded from '${LOADED:-nothing}', not $WORK_TREE" >&2
    echo "  The cwd is the MAIN tree (§3) so a rendered draw embeds the main tree's dir; the" >&2
    echo "  CODE must still be the task worktree's, or this gate measures another commit." >&2
    exit 2
    ;;
esac

ALIVE=$(pgrep -f 'python.*offline\.(transfer_curve|g2_measure)' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: cells from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo "  A second SUMO pool would make every memory number here meaningless." >&2
  exit 3
fi

if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux foreground pane" >&2
  echo "  The handler kills the process group, and that is a no-op from a non-leader, so an" >&2
  echo "  interrupted run would leave the worker pool holding SUMO processes." >&2
  exit 2
fi

SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  echo "REFUSING TO START: SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" >&2
  echo "  bash cannot trap a signal ignored on entry, so Ctrl-C could not stop the pool." >&2
  exit 2
fi

if [ ! -f "$RLTRAFFIC_GRID4X4_RESCO/resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml" ]; then
  echo "REFUSING TO START: RLTRAFFIC_GRID4X4_RESCO=$RLTRAFFIC_GRID4X4_RESCO holds no grid4x4 net" >&2
  echo "  grid4x4's SUMO side is RESCO's, read-only and gitignored; it has no default path." >&2
  exit 2
fi

CHECKPOINT=$MAIN/output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed${SEED}.pt
for INPUT in "$CHECKPOINT" "$ENV_CONFIG" "$DRAWS/cityflow_grid4x4/draw_0005/flow.json"; do
  if [ ! -f "$INPUT" ]; then
    echo "REFUSING TO START: missing input $INPUT" >&2
    exit 2
  fi
done

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "REFUSING TO START: the worktree $WORK_TREE is DIRTY" >&2
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  echo "  A measurement is attributed to a commit; a dirty tree has no commit to attribute it to." >&2
  exit 2
fi

on_signal() {
  echo "" >&2
  echo "G2 INTERRUPTED — killing the process group; nothing was consumed, no cell written." >&2
  trap - INT TERM HUP
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM HUP

echo "P7.3d GATE G2 — the one-cell measurement (BRIEF_39 Amendment B.2)"
echo "  commit      $(git -C "$WORK_TREE" rev-parse HEAD)"
echo "  code        $LOADED"
echo "  cwd         $(pwd)   (the MAIN tree: Amendment A6's render rule)"
echo "  draw        $DRAW  (the smoke draw — NOT the held-out pool, NOT the probe band)"
echo "  dt seed     $SEED, under the CHECKPOINT'S OWN in-domain prompt (B.2-1)"
echo "  pools       $POOL_SIZES"
echo "  resco       $RLTRAFFIC_GRID4X4_RESCO"
echo "  started     $(date -Is)"
echo "  NO OUTCOME IS MEASURED OR PRINTED: no travel time, no reconstructed per-vehicle"
echo "     average, no episode return, no conditioning series, no action — from any cell,"
echo "     on any draw (Amendment B.2-2). The executable text of this driver does not even"
echo "     contain those field names, and a test asserts it."
echo ""

# shellcheck disable=SC2086
PYTHONPATH=$WORK_TREE "$PY" -m offline.g2_measure \
  --draws-root "$DRAWS" \
  --output-root "$MAIN/output" \
  --work-dir "$WORK" \
  --env-config "$ENV_CONFIG" \
  --draw "$DRAW" \
  --seed "$SEED" \
  --workers $POOL_SIZES

echo ""
echo "G2 COMPLETE  $(date -Is)"
echo "NEXT: the implementer reads the capture, sets WORKERS and the RSS refusal budget from it,"
echo "      and writes C5's SUMO probe script. Nothing here is a result."
