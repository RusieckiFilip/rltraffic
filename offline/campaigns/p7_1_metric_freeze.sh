#!/usr/bin/env bash
# P7.1 half A driver (BRIEF_34 sections 2.2 and 5 G3, Amendment D4) -- run from the task WORKTREE.
#
#   tmux new -s p71freeze
#   date -Is > /home/filip/rltraffic/output/p7_1/AUTHORISED_TO_RUN   # the AUTHOR does this,
#                                                                    # after the pre-flight is CLEAR
#   bash /home/filip/rltraffic-p53b/offline/campaigns/p7_1_metric_freeze.sh
#
# ⚠️ output/p7_1/ MUST NOT EXIST at the first start, so the token is written into a directory the
# author creates for it:  mkdir -p /home/filip/rltraffic/output/p7_1 && date -Is > .../AUTHORISED_TO_RUN
# That is the ONLY directory creation this campaign needs before the token check, and it is the
# author's, not the script's.
#
# Fails closed; idempotent and restartable. Skeleton of offline/campaigns/p5_3b_decomp.sh, with the
# differences this task needs stated rather than left to a diff.
#
# ---------------------------------------------------------------------------
# 1. IT WRITES TO THE **MAIN** TREE, NOT THE WORKTREE (Decisions Log 2026-09-10)
# ---------------------------------------------------------------------------
# WORK=$MAIN/output/p7_1. Review MJ-9 found P5.3b's whole evidence base living only in a worktree,
# one `git worktree prune` from gone. The fence (sumo_att_reference.assert_metric_freeze_writable)
# is what makes writing there safe: it is DEFAULT-DENY on whole path components and allows exactly
# `p7_1` and `SHA256SUMS_p7_1.txt` -- refusing, in particular, `output/p7_0`, which this campaign
# READS on every cell and must never write.
#
# ---------------------------------------------------------------------------
# 2. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL
# ---------------------------------------------------------------------------
# `run-sumo` / `run-cityflow` inspect their own chunk and re-run unless it is COMPLETE, at this
# format version, and for THIS EXACT CELL -- backend, arm, observer flag and teleport regime, in
# the header AND in every row. ⛔ THERE ARE DELIBERATELY NO `[ -f ]` GUARDS BELOW: a bad chunk that
# survived a restart is what `assert_probe_cell_is_ablated`'s docstring records the cost of.
#
# ---------------------------------------------------------------------------
# 3. ONE-SHOT RUN AUTHORISATION (2026-09-10 ruling -- a mechanism, not a courtesy)
# ---------------------------------------------------------------------------
# The author writes the token; this script requires it and DELETES it as its first mutation, so one
# token buys exactly one run. Every check that can refuse -- cwd, import, start lock -- PRECEDES the
# token, so a refused start consumes nothing and changes nothing at all.
#
# ---------------------------------------------------------------------------
# 4. THE SCHEDULE, FROM MEASURED RATES ONLY (Amendment E1 item 5)
# ---------------------------------------------------------------------------
# ⚠️ The first version of this header projected ~17 min from an ASSUMED ~1.5 s CityFlow episode and
# an ASSUMED ~1.5 min A4. Both were wrong, and A4 by 5x. Every rate below is now a measurement with
# its n; this project has twice shipped a wrong schedule from an unmeasured rate.
#
#   hz1x1 parity, maxpressure, seed 1000, one episode each (P7.1 G1):
#     frozen env, no observer .................... 13.64 s/episode   (n = 1)
#     observer, no halting check ................. 14.63 s/episode   (n = 1, +19 %)
#     observer + halting cross-check ............. 44.94 s/episode   (n = 1, 3.30x)
#   hz1x1 nominal, CityFlow, maxpressure, observed (P7.1 E1):
#     Gate-0 observer ............................  6.97 s/episode   (n = 1)
#   hz4x4 gudang, random, observed, no halting check (pre-flight PART 1 completion):
#     observer ................................... 438.69 s/episode  (n = 1)  <- 8.6x the 51.0 s
#                                                  bare rate the 2026-09-11 survey measured
#
# The halting check queries every vehicle's speed every second; it is 2.9x on its own, which
# falsified the plan's A7. It runs on the FIRST EPISODE of each observed SUMO arm only
# (--halting-episodes 1), and every row records the lane-seconds it actually covered. One episode
# is 28,800 lane-seconds; G1 measured max abs difference 0 over them.
#
# Projection for the stages AS THIS SCRIPT RUNS THEM, so the log can be compared against it:
#     smoke           1 observed episode with the halting check ........  0.7 min
#     A1  observed    3 arms x (1 x 44.94 + 4 x 14.63) ................   5.2 min
#     A1  unobserved  3 arms x 5 x 13.64 ..............................   3.4 min
#     A1b observed    3 arms x (1 x 44.94 + 4 x 14.63) ................   5.2 min
#     A2  cityflow    3 arms x 5 x 6.97 ...............................   1.7 min
#     A4  hz4x4 gudang, 1 episode ......................................  7.3 min
#     report + manifest ................................................  0.5 min
#     ----------------------------------------------------------------  --------
#     TOTAL, as configured .............................................  24 min
#
# ⚠️ PLAN FOR ~33 MIN, NOT 24. Amendment E1 item 5 states G3 at ~33 min, which is the same rates
# with the halting cross-check on EVERY observed episode (3 arms x 5 x 44.94 twice = 22.5 min
# instead of 10.4). That is the conservative bound: it is what the run costs if --halting-episodes
# is ever raised to 5, and a schedule that is an upper bound is the useful kind. Both figures are
# stated because they differ by the --halting-episodes flag on the A1/A1b stages, not by an unknown.
#
# Either way this is inside section 7's one-hour trigger; the PRE-FLIGHT was required by trigger
# (b) instead -- this writes under output/ while reading output/p7_0.
#
# ---------------------------------------------------------------------------
# 5. WHAT THE STAGES ARE, AND WHY A1 AND A1b ARE BOTH RUN
# ---------------------------------------------------------------------------
# A1  regime `parity`     : P7.0's own .sumocfg, teleports ENABLED (SUMO's 300 s default). This is
#                           the arm the reproduction check is defined against, and it is labelled
#                           *teleports enabled* wherever it is reported.
# A1b regime `noteleport` : the NEW ..._parity_noteleport.sumocfg (Amendment D2), same network and
#                           same parity routes, `time-to-teleport -1`. `run-sumo --regime noteleport`
#                           also ASSERTS n_teleports == 0 and n_vanished_without_arrival == 0 on
#                           every episode, so the regime cannot be reported without being checked.
#                           It has NO P7.0 counterpart and its rows are reported as unverified.
# A2  cityflow            : the same three arms through engine_att_reference's Layer A.
# A4  hz4x4 gudang        : ONE episode, TIMING ONLY. Its ATT is not a result and the artifact says
#                           so: the shipped gudang route file binds no parity vType. It is the one
#                           stage a restart used to re-roll; it now skips a complete chunk, which
#                           is worth 7.3 min (Amendment E1 item 4, mn-1).
#
# ⚠️ STAGE ORDER: the smoke runs FIRST, not between A4 and report as Amendment D4 lists it. D4's
# line is a list of what must be present; running the cheapest failing stage first means a broken
# tree costs 45 s rather than 24 min. Recorded here because the pre-flight noticed the difference
# (PART 1 note, PART 2 MINOR 7) and a reader should not have to.
#
# ---------------------------------------------------------------------------
# 6. THE MANIFEST INCLUDES smoke/ THIS TIME, AND THAT IS DELIBERATE (Amendment D1)
# ---------------------------------------------------------------------------
# p5_3b_decomp.sh PRUNED smoke/ because that smoke predated the campaign commit and listing it would
# have made a later tidy-up break `sha256sum -c`. Here the smoke is produced BY THIS RUN, at this
# commit, and the packet cites the file rather than a table typed from a terminal -- so it is part
# of the evidence and is listed. Nothing under output/p7_1/ is deleted after the manifest is written.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=/home/filip/rltraffic-p53b
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p7_1
LOGS=$WORK/logs
SMOKE=$WORK/smoke
ARMS=(fixedtime maxpressure random)

COMMON=(--output-root "$MAIN/output"
        --work-dir "$WORK"
        --out-dir "$WORK_TREE/docs/data"
        --episodes 5
        --base-seed 1000)

# SUMO is single-threaded per process and this campaign runs one cell at a time, but the pin is
# re-asserted here as well as in the tmux shell: P4.6 section 12's condition 3.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# ⚠️ RLTRAFFIC_OUTPUT_ROOT / RLTRAFFIC_CORPUS_V11 are deliberately NOT exported: they are what stop
# the TESTS skipping, and no test runs here. The CLI flags above fix the campaign's paths.

# ---------------------------------------------------------------------------
# PRECONDITIONS -- every one of them BEFORE the token is touched
# ---------------------------------------------------------------------------
# `offline` is importable only from a tree root: the venv has no `offline` on sys.path, and the
# worktree has no .venv of its own. A condition the operator must remember is DEFERRED 61's class,
# so it is enforced instead.
cd "$WORK_TREE"

# ⚠️ ORDER: the interpreter check comes FIRST. Behind the import check it was unreachable (PART 1's
# minor): a missing $PY fails the import check first, and reports it as a wrong cwd, which is the
# one diagnosis it is not.
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  echo "  The worktree has no .venv of its own; this campaign runs the MAIN tree's." >&2
  exit 2
fi

if ! $PY -c "import offline.sumo_att_reference" >/dev/null 2>&1; then
  echo "REFUSING TO START: cannot import offline.sumo_att_reference from $PWD" >&2
  echo "  This is what a wrong cwd looks like. Nothing has been consumed." >&2
  exit 2
fi

# pre-flight M3's class: SIGINT to the tmux pane can kill the driver while its python child keeps
# writing. A second start while one is alive would interleave two runs into one work dir.
# ⚠️ The pattern requires `python` before the module name, and this process and its parent are
# excluded: a bare match also matches the SHELL running this script, because the module name appears
# in its own command line, and the driver would refuse to start because of itself.
ALIVE=$(pgrep -f 'python.*offline\.sumo_att_reference' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: cells from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo '  Wait until `pgrep -f sumo_att_reference` is empty. Nothing has been consumed.' >&2
  exit 3
fi

# The A1b configuration must exist before the token is spent: A1b is half the point of this run.
NOTELEPORT=$WORK_TREE/scenarios/hangzhou_1x1_bc-tyc_18041610_1h_parity/hangzhou_1x1_bc-tyc_18041610_1h_parity_noteleport.sumocfg
if [ ! -f "$NOTELEPORT" ]; then
  echo "REFUSING TO START: $NOTELEPORT is missing, so stage A1b cannot run" >&2
  exit 2
fi

# P7.0's episodes are READ by every parity cell. A missing or altered tree would make the
# reproduction check vacuous rather than failing, so it is verified before anything is spent.
if [ ! -d "$MAIN/output/p7_0" ]; then
  echo "REFUSING TO START: $MAIN/output/p7_0 is missing; the reproduction check has no reference" >&2
  exit 2
fi
if ! ( cd "$MAIN/output" && sha256sum -c SHA256SUMS_p7_0.txt >/dev/null 2>&1 ); then
  echo "REFUSING TO START: output/SHA256SUMS_p7_0.txt does not verify; P7.0's episodes are the" >&2
  echo "  reference this campaign reproduces against and they must be intact first." >&2
  exit 2
fi

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

# Only NOW may anything be created or cleared. A refused start above left the tree exactly as the
# previous run left it.
mkdir -p "$LOGS" "$SMOKE"
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

fail() { echo "CAMPAIGN FAILED at $1" | tee "$WORK/FAILED"; exit 1; }

# The pid of the cell currently in flight. Global, so the signal handler can reach it: bash traps do
# not see a caller's locals.
CELL_PID=""

# Ctrl-C in tmux signals the process GROUP, and a child started with `&` does not die with the
# shell. The handler kills the cell explicitly by pid, then the group; and it writes FAILED FIRST,
# so the record survives even if the group kill takes the script with it.
on_signal() {
  trap '' INT TERM
  echo "CAMPAIGN INTERRUPTED by a signal" | tee "$WORK/FAILED" >&2
  [ -z "$CELL_PID" ] || kill -TERM "$CELL_PID" 2>/dev/null || true
  sleep 2
  [ -z "$CELL_PID" ] || kill -KILL "$CELL_PID" 2>/dev/null || true
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM

STARTED=$(date +%s)

# One cell, run in the background so the trap can reach it by pid, then waited for. `run-*` exits
# non-zero when an episode does not reproduce its P7.0 cell or when a regime assertion fails; the
# chunk is written first, so the evidence survives the stop.
run_cell() {
  local label=$1; shift
  local log=$LOGS/${label}.log
  echo "=== $label"
  # ⚠️ APPENDED, never truncated (Amendment E1 item 4, mn-2): a restart used to replace the first
  # run's per-cell log with the one line "skipping", destroying the only narrative record of what
  # the cell actually did. The banner separates the runs.
  echo "=== $label  run at $(date -Is)" >> "$log"
  "$PY" -m offline.sumo_att_reference "${COMMON[@]}" "$@" >> "$log" 2>&1 &
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

# ---------------------------------------------------------------------------
# SMOKE -- Amendment D1: the record is a file, produced by this run at this commit
# ---------------------------------------------------------------------------
# One observed maxpressure episode into output/p7_1/smoke/, with the halting cross-check on, so the
# packet cites a file rather than a terminal. It uses its own work dir, so it can never be mistaken
# for an A1 cell or skipped into one.
echo "=== smoke: 1 observed maxpressure episode into $SMOKE"
echo "=== smoke  run at $(date -Is)" >> "$LOGS/smoke.log"
"$PY" -m offline.sumo_att_reference \
    --output-root "$MAIN/output" --work-dir "$SMOKE" --out-dir "$WORK_TREE/docs/data" \
    --episodes 1 --base-seed 1000 \
    run-sumo --arm maxpressure --regime parity --halting-episodes 1 \
    >> "$LOGS/smoke.log" 2>&1 || { tail -5 "$LOGS/smoke.log" >&2; fail "smoke"; }
tail -1 "$LOGS/smoke.log"

# ---------------------------------------------------------------------------
# A1 -- the three anchors, teleports ENABLED, observed then unobserved
# ---------------------------------------------------------------------------
for ARM in "${ARMS[@]}"; do
  run_cell "a1_sumo_${ARM}" run-sumo --arm "$ARM" --regime parity --halting-episodes 1
done
for ARM in "${ARMS[@]}"; do
  run_cell "a1_sumo_${ARM}_unobserved" run-sumo --arm "$ARM" --regime parity --no-observer
done

# ---------------------------------------------------------------------------
# A1b -- the same three anchors with teleporting DISABLED (Amendment D2)
# ---------------------------------------------------------------------------
for ARM in "${ARMS[@]}"; do
  run_cell "a1b_sumo_${ARM}_noteleport" run-sumo --arm "$ARM" --regime noteleport --halting-episodes 1
done

# ---------------------------------------------------------------------------
# A2 -- the three CityFlow anchors through engine_att_reference's Layer A
# ---------------------------------------------------------------------------
for ARM in "${ARMS[@]}"; do
  run_cell "a2_cityflow_${ARM}" run-cityflow --arm "$ARM"
done

# ---------------------------------------------------------------------------
# A4 -- hz4x4 gudang, ONE episode, TIMING ONLY
# ---------------------------------------------------------------------------
run_cell "a4_timing_hz4x4" timing-hz4x4 --episodes 1

# ---------------------------------------------------------------------------
# REPORT -- validates every episode of every chunk before writing a byte
# ---------------------------------------------------------------------------
echo "=== assembling docs/data/p7_1_metric_freeze.json"
run_cell "report" report

# ---------------------------------------------------------------------------
# MANIFEST -- written LAST, from a stable tree, smoke/ INCLUDED (Amendment D1)
# ---------------------------------------------------------------------------
# tmp + mv so an interrupted manifest step cannot leave a half-written file that verifies nothing.
echo "=== writing output/SHA256SUMS_p7_1.txt (smoke/ included)"
( cd "$MAIN/output" \
  && find p7_1 -type f \( -name '*.json' -o -name '*.log' \) -print \
     | LC_ALL=C sort | xargs sha256sum > SHA256SUMS_p7_1.txt.tmp \
  && mv SHA256SUMS_p7_1.txt.tmp SHA256SUMS_p7_1.txt ) || fail "manifest write"
echo "    $(wc -l < "$MAIN/output/SHA256SUMS_p7_1.txt") entries"

# Re-verify every manifest in the tree this campaign read from.
echo "=== re-verifying every manifest in $MAIN/output"
( cd "$MAIN/output"
  status=0
  for M in SHA256SUMS_*.txt; do
    if out=$(sha256sum -c "$M" 2>&1); then
      printf '  %-32s %s OK\n' "$M" "$(printf '%s\n' "$out" | grep -c ': OK$')"
    else
      printf '  %-32s FAILED\n' "$M"; printf '%s\n' "$out" | grep -v ': OK$' | head -5; status=1
    fi
  done
  exit $status
) || fail "manifest re-verification"

ELAPSED=$(( $(date +%s) - STARTED ))
echo "CAMPAIGN COMPLETE in ${ELAPSED}s ($((ELAPSED / 60)) min)" | tee "$WORK/COMPLETE"

cat <<'NEXT'

=== NEXT STEPS, run by hand and NOT by this script ===
  1. Commit docs/data/p7_1_metric_freeze.json in the worktree.
  2. Read, in this order, before writing a word of the freeze document:
       - reproduction.n_equal / n_verified      (SUMO's determinism, as measured)
       - cadence.n_with_zero_cadence_term       (Amendment A5: is the third term zero, and on how
                                                 many episodes, and what do the two counters say)
       - cells.sumo__* vs cells.sumo_noteleport__*   (A1b's effect size, per arm, both definitions)
       - halting_threshold.max_abs_difference   (with its n_lane_seconds)
       - timing.*                               (the hz1x1 and hz4x4 SUMO rates, each with its n)
  3. docs/notes/P7.1_FREEZE.md section "Metric" is DRAFTED from those numbers. It proposes; it does
     not register. The amendment is the coordinator's (BRIEF_34 section 2.3).
NEXT
