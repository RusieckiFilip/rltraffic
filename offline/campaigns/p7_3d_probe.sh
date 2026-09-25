#!/usr/bin/env bash
# P7.3d — C5: the SUMO PROBE, 100 MaxPressure episodes on grid4x4 draws 201–300, per intersection.
#
# Shape: offline/campaigns/p7_3d_g2.sh (the gate that measured this run's rate), which took its
# shape from offline/campaigns/p7_3b_anchor.sh.
#
# 0. USAGE — from a tmux pane the AUTHOR starts (CLAUDE.md §5; channel (a)):
#      tmux new -s p73d_probe 'bash /home/filip/rltraffic-p73d/offline/campaigns/p7_3d_probe.sh 2>&1 | tee -a /home/filip/rltraffic/output/p7_3d_runs/probe_capture.txt'
#    No argument. No token: A17(b)'s probe produces the ANCHOR-side inputs of Rule B — MaxPressure
#    returns, not an evaluation of any arm — which is why Amendment B.2-2(4) lets it run on the
#    probe band without one. Nothing here evaluates the subject.
#
# 1. WHAT IT PRODUCES
#      output/p7_3d/calibration/probe_sumo_draw_0201.json … draw_0300.json   one chunk per draw
#      output/p7_3d/calibration/failed/                                      chunks that failed
#    The CityFlow half (probe_cityflow_draw_*.json, 100 chunks) is ALREADY THERE, measured
#    in-session on 2026-09-19 under Amendment A4. This run adds the SUMO half beside it; the
#    calibration artifact is assembled from both afterwards, from the chunks and not from memory.
#    ⛔ It writes no cell, no campaign chunk and no docs/data/ file.
#
# 2. RESUMABLE BY CONTENT, so an interrupted run is restarted by re-running this script.
#    offline/transfer_calibration.py::per_intersection_chunk_is_reusable re-derives a chunk's
#    verdict from its own content — the two return routes re-compared PER INTERSECTION, and on
#    this side A17(b)'s engine reads too (0 teleports, exactly [cf_parity], "-1"). A chunk that
#    fails its own re-validation is moved to failed/ and re-rolled. ⛔ There are deliberately no
#    `[ -f ]` guards over chunks here; offline/campaigns/p5_3b.sh had one and a bad chunk survived
#    every restart.
#
# 3. THE CWD RULE, CARRIED BY THIS SCRIPT (Amendment B3/Q4) — as in the G2 driver, and `-P` on
#    BOTH interpreter calls. Without it Python prepends the cwd (the MAIN tree, by the render
#    rule) to sys.path and `offline` resolves to the main tree's package, which does not carry
#    this module; that is what made G2's first hand-over refuse on 2026-09-20.
#
# 4. ORDERING — every check that can refuse precedes the first episode: interpreter → import from
#    the worktree → no live runner → group leader → SigIgn → inputs (RESCO root, the 100 parity
#    configs, the CityFlow half) → dirty tree → free memory → trap → work. The canary (both
#    halves) is the first thing the Python does.
#
# 5. TIME AND MEMORY — FROM GATE G2, measured 2026-09-20 at commit 07ab267, canary 0.76 s, on this
#    machine (16 cores, 48,174 MiB host, RTX 5080 Laptop 16,303 MiB):
#      one grid4x4 SUMO cell, single worker, halting check OFF   35.5 s in-process, 1,280 MiB
#      W = 4    wall 46.7 s   11.67 s/cell   speed-up 3.89x    tree RSS  6,562 MiB
#      W = 8    wall 77.4 s    9.67 s/cell   speed-up 7.81x    tree RSS 11,856 MiB
#      W = 12   wall 58.6 s    4.88 s/cell   speed-up 11.39x   tree RSS 17,297 MiB, GPU 4,595 MiB
#    ⚠️ The W = 8 row is SLOWER PER CELL than W = 12 (75.5 s against 55.8 s in-process) and that is
#    UNEXPLAINED; within each pool the spread was under 0.5 s, so it is systematic, not noise. It
#    does not change the choice — 12 is the best measured throughput and matches BRIEF_37 C3's
#    ruling of 12 workers on these 16 cores — but the schedule below carries both rates.
#    THIS PROBE: 100 episodes, MaxPressure, no DT and no observer, so a cell is CHEAPER than G2's.
#    Expect ≈ 100 x 4.88 s ≈ 8 min at W = 12, and ≈ 11 min at the W = 8 per-cell rate. The probe is
#    sequential by construction (one env per draw, in-process) — see 6.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=/home/filip/rltraffic-p73d
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p7_3d/calibration
DRAWS=$MAIN/scenarios/draws
SCENARIO=cityflow_grid4x4
FIRST_DRAW=201
LAST_DRAW=300
ENGINE_SEED=1000

# 6. WORKERS — set from G2 (§5). The probe runs SEQUENTIALLY in one process: A17(b) fixes one
#    episode per draw on a fresh env and the 100 episodes are ~8 min at G2's rate, so a pool would
#    buy minutes at the cost of a second concurrency regime to reason about. WORKERS is recorded
#    here because the CAMPAIGN driver (C6) takes it from this same measurement and the number must
#    have one home.
WORKERS=12
# The RSS refusal budget, from G2's measured peak of 17,297 MiB at W = 12 (1,441 MiB per worker)
# with a 1.4x margin. This probe uses one worker and about 1,300 MiB, so the check below is a
# pre-flight for the CAMPAIGN's regime, not for this run's: it refuses here if the machine could
# not hold the campaign either, which is the cheapest moment to find that out.
PEAK_TREE_RSS_MIB=17297
PER_WORKER_RSS_MIB=1441
RSS_BUDGET_MIB=24216
GPU_PEAK_MIB=4595

: "${RLTRAFFIC_GRID4X4_RESCO:=$MAIN/scenarios/grid4x4_candidates}"
export RLTRAFFIC_GRID4X4_RESCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---------------------------------------------------------------- preconditions
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  exit 2
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.transfer_calibration as m; print(m.__file__)' 2>/dev/null || true)
case "$LOADED" in
  "$WORK_TREE"/*) ;;
  *)
    echo "REFUSING TO START: offline.transfer_calibration loaded from '${LOADED:-nothing}', not $WORK_TREE" >&2
    echo "  The cwd is the MAIN tree (§3); the CODE must still be the task worktree's." >&2
    exit 2
    ;;
esac

ALIVE=$(pgrep -f 'python.*offline\.(transfer_curve|transfer_calibration|g2_measure)' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: episodes from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  exit 3
fi

if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux foreground pane" >&2
  echo "  The handler kills the process group, and that is a no-op from a non-leader." >&2
  exit 2
fi

SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  echo "REFUSING TO START: SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" >&2
  exit 2
fi

if [ ! -f "$RLTRAFFIC_GRID4X4_RESCO/resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml" ]; then
  echo "REFUSING TO START: RLTRAFFIC_GRID4X4_RESCO=$RLTRAFFIC_GRID4X4_RESCO holds no grid4x4 net" >&2
  exit 2
fi

MISSING=0
for DRAW in $(seq "$FIRST_DRAW" "$LAST_DRAW"); do
  CONFIG=$(printf '%s/%s/draw_%04d/parity/noteleport.sumocfg' "$DRAWS" "$SCENARIO" "$DRAW")
  if [ ! -f "$CONFIG" ]; then
    echo "REFUSING TO START: missing parity configuration $CONFIG" >&2
    MISSING=$((MISSING + 1))
    [ "$MISSING" -ge 3 ] && break
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "  P7.3d C1 renders the probe band; gate G1 recorded 200/200 exact in docs/data/p7_3d_cap_e.json." >&2
  exit 2
fi

CITYFLOW_CHUNKS=$(find "$WORK" -maxdepth 1 -name 'probe_cityflow_draw_*.json' 2>/dev/null | wc -l)
if [ "$CITYFLOW_CHUNKS" -ne 100 ]; then
  echo "REFUSING TO START: $CITYFLOW_CHUNKS of 100 CityFlow probe chunks in $WORK" >&2
  echo "  Rule B's ratio needs BOTH domains over the SAME draws; the CityFlow half was measured" >&2
  echo "  in-session on 2026-09-19 under Amendment A4 and must be on disk before this half runs." >&2
  exit 2
fi

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "REFUSING TO START: the worktree $WORK_TREE is DIRTY" >&2
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  echo "  Every chunk records the commit it was rolled at; a dirty tree has none to record." >&2
  exit 2
fi

AVAILABLE_MIB=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
if [ -n "$AVAILABLE_MIB" ] && [ "$AVAILABLE_MIB" -lt "$RSS_BUDGET_MIB" ]; then
  echo "REFUSING TO START: ${AVAILABLE_MIB} MiB available, below the ${RSS_BUDGET_MIB} MiB budget" >&2
  echo "  The budget is gate G2's measured peak of ${PEAK_TREE_RSS_MIB} MiB at ${WORKERS} workers" >&2
  echo "  (${PER_WORKER_RSS_MIB} MiB per worker) with a 1.4x margin. This probe itself needs about" >&2
  echo "  1,300 MiB; the check is a pre-flight for the CAMPAIGN's regime, and the cheapest moment" >&2
  echo "  to learn the machine cannot hold it is now." >&2
  exit 2
fi

on_signal() {
  echo "" >&2
  echo "PROBE INTERRUPTED — killing the process group. Every completed chunk is on disk and" >&2
  echo "  re-running this script resumes from them (§2); nothing was consumed." >&2
  trap - INT TERM HUP
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM HUP

echo "P7.3d C5 — the SUMO probe, per intersection (A17(b), Amendment A4)"
echo "  commit          $(git -C "$WORK_TREE" rev-parse HEAD)"
echo "  code            $LOADED"
echo "  cwd             $(pwd)   (the MAIN tree: Amendment A6's render rule)"
echo "  scenario        $SCENARIO, draws ${FIRST_DRAW}-${LAST_DRAW}, one episode each"
echo "  engine seed     $ENGINE_SEED on a fresh env per draw (A18(c))"
echo "  cityflow half   $CITYFLOW_CHUNKS chunks already on disk"
echo "  memory          ${AVAILABLE_MIB} MiB available, budget ${RSS_BUDGET_MIB} MiB (G2: peak ${PEAK_TREE_RSS_MIB} MiB at W=${WORKERS}, GPU ${GPU_PEAK_MIB} MiB)"
echo "  schedule        G2's rate gives about 8 min; the unexplained W=8 rate gives about 11"
echo "  started         $(date -Is)"
echo ""

PYTHONPATH=$WORK_TREE "$PY" -P - <<PYTHON
import sys, time
import offline.transfer_calibration as tc

seconds, facts = tc.canary_seconds()
line = tc.format_canary_line(seconds, facts)
print("canary (both halves), before the first episode", flush=True)
print("  " + line, flush=True)
tc.check_canary(facts)
print("  canary correctness half: PASSED", flush=True)

started = time.perf_counter()
returns = tc.run_sumo_probe_per_intersection(
    range($FIRST_DRAW, $LAST_DRAW + 1),
    scenario_key="$SCENARIO",
    out_root="$DRAWS",
    work_dir="$WORK",
    engine_seed=$ENGINE_SEED,
    canary_seconds=seconds,
)
wall = time.perf_counter() - started
n_draws = len(returns)
ids = sorted({ix for per in returns.values() for ix in per})
print("", flush=True)
print(f"{n_draws} draws, {len(ids)} intersections each", flush=True)
print(f"wall {wall:.1f} s = {wall / max(1, n_draws):.2f} s/episode", flush=True)
if n_draws != 100 or len(ids) != 16:
    print(f"INCOMPLETE: {n_draws} of 100 draws, {len(ids)} of 16 intersections", flush=True)
    sys.exit(1)
PYTHON

echo ""
echo "PROBE COMPLETE  $(date -Is)"
echo "NEXT: the implementer assembles docs/data/p7_3d_calibration.json from BOTH halves' chunks,"
echo "      read back from disk. Nothing in this run is an evaluation of any arm."
