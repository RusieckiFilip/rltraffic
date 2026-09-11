#!/usr/bin/env bash
# P5.3b-fix BL-2(b) driver -- A13(b)'s decomposition on all 30 cells, run from the task WORKTREE.
#
#   tmux new -s p53bdecomp
#   date -Is > /home/filip/rltraffic/output/p5_3b_decomp/AUTHORISED_TO_RUN   # the AUTHOR does this
#   bash /home/filip/rltraffic-p53b/offline/campaigns/p5_3b_decomp.sh
#
# Fails closed; idempotent and restartable. Skeleton of offline/campaigns/p5_3b.sh with four
# deliberate differences, each ruled or disclosed:
#
# ---------------------------------------------------------------------------
# 1. IT WRITES TO THE **MAIN** TREE, NOT THE WORKTREE (Decisions Log 2026-09-10)
# ---------------------------------------------------------------------------
# p5_3b.sh set WORK=$WORK_TREE/output/p5_3b, and review MJ-9 found the entire evidence base --
# 27 entries, 15 checkpoints, the 39-line manifest -- existing ONLY in a worktree, one
# `git worktree prune` from gone. A new directory in the main tree needs no securing step.
#   WORK=$MAIN/output/p5_3b_decomp
# The fence (nortg_decomposition.assert_decomposition_writable) is what makes that safe: it allows
# exactly p5_3b_decomp and SHA256SUMS_p5_3b_decomp.txt and refuses every other component under
# output/, INCLUDING p5_3b, which this campaign reads and must never write.
#
# ---------------------------------------------------------------------------
# 2. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL (BRIEF_33 AMENDMENT A4)
# ---------------------------------------------------------------------------
# p5_3b.sh skips a stage on `[ -f "$WORK/chunk.json" ]` alone. assert_probe_cell_is_ablated's own
# docstring records what that costs: "the driver skips a tier whose probe chunk exists, so a bad
# chunk survived every restart." Here `run` inspects its own chunk and re-runs unless it is
# COMPLETE (all 100 draws) and CLEAN (n_mismatches == 0). One Python start per cell costs ~2 s
# against an hour of rollouts. ⛔ THERE ARE DELIBERATELY NO `[ -f ]` GUARDS BELOW.
#
# ---------------------------------------------------------------------------
# 3. ONE-SHOT RUN AUTHORISATION (2026-09-10 ruling -- a mechanism, not a courtesy)
# ---------------------------------------------------------------------------
# The author writes the token; the script requires it and DELETES it as its first action, so one
# token buys exactly one run. The check PRECEDES every mutation this script makes -- including the
# FAILED/COMPLETE wipe AND the `mkdir -p` p5_3b.sh did first -- so a refused start changes nothing
# at all. Validate-then-mutate, applied to the driver itself.
#
# ---------------------------------------------------------------------------
# 4. THE SCHEDULE IS MEASURED, NOT INHERITED (BRIEF_33 AMENDMENTS A5 and B1)
# ---------------------------------------------------------------------------
# ⛔ DO NOT quote a parallel speedup extrapolated from a small sample. BRIEF_33 assumed 5x and
# printed "~35 min"; AMENDMENT A applied a 30-episode 2.55x to an already-contended per-process rate
# and printed "61-69 min". Both were wrong, in opposite directions, and neither needed estimating:
# output/p5_3b/ holds three directly measured 500-episode five-tuple wall clocks. Re-measured from
# the committed eval chunks (max `seconds` per five-tuple / 500 episodes):
#
#     mappo1000  273.8 s -> 0.548 s/episode      mix50  297.7 s -> 0.595 s/episode
#     random     304.5 s -> 0.609 s/episode      (5 concurrent cells, 100 draws each)
#
# The observer's overhead on a DT episode was the one unmeasured term; the G2 smoke measured it
# over 10 episodes: 1.964 s/episode serial under the observer against 1.85 s without it, a ratio of
# 1.061. So 3000 x 0.609 x 1.061 = ~32 min wall at 5 workers, ~1.64 h serial. The packet reports
# this run's own wall clock beside that projection.
#
# 3,000 episodes = 30 cells x 100 held-out draws, 6 groups of 5 concurrent cells.
#
# ---------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES **NOT** DO
# ---------------------------------------------------------------------------
# It does not run `nortg_campaign report`. That regenerates docs/data/p5_3b_nortg.json and
# BRIEF_33 section 3.5 requires it to run FROM A CLEAN TREE, while this script's own last stage
# writes docs/data/p5_3b_decomposition.json into the worktree. The two steps are separated on
# purpose; the NEXT STEPS block at the end says so.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=/home/filip/rltraffic-p53b
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p5_3b_decomp
LOGS=$WORK/logs
SEEDS=(101 202 303 404 505)

COMMON=(--corpus-root $MAIN/datasets_v11
        --draws-root $MAIN/scenarios/draws
        --output-root $MAIN/output
        --work-dir $WORK
        --out-dir $WORK_TREE/docs/data
        --torch-threads 1)

# One torch thread per process is P4.6's protocol, and it is what the reused dt column was measured
# under. Gate 1b proved this harness matches that column bit-for-bit; changing the thread regime
# would put that at risk for a speedup the GPU will not give anyway.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# ⚠️ RLTRAFFIC_CORPUS_V11 / RLTRAFFIC_OUTPUT_ROOT are deliberately NOT exported: AMENDMENT E5 --
# they are what stop the TESTS skipping, and no test runs here. The CLI flags above fix the
# campaign's paths. Two mechanisms, two jobs.

# ---------------------------------------------------------------------------
# PRECONDITIONS -- every one of them BEFORE the token is touched (pre-flight M2)
# ---------------------------------------------------------------------------
# `offline` is importable only from a tree root: the venv has no `offline` on sys.path. The header
# said "run from the task WORKTREE" and gave no `cd`, so from any other cwd all five cells died with
# ModuleNotFoundError -- AFTER the token had been burnt and FAILED written. A condition the operator
# must remember is DEFERRED 61's class; this is the same condition, enforced.
cd "$WORK_TREE"

if ! $PY -c "import offline.nortg_decomposition" >/dev/null 2>&1; then
  echo "REFUSING TO START: cannot import offline.nortg_decomposition from $PWD" >&2
  echo "  This is what a wrong cwd looks like. Nothing has been consumed." >&2
  exit 2
fi

# pre-flight M3: SIGINT to the tmux pane killed the driver while the five python cells SURVIVED and
# kept writing. A second start while they run was accepted, which would double the GPU load and make
# the artifact's own concurrent-load statement false. Refuse rather than overlap.
# ⚠️ The pattern requires `python` before the module name, and this process and its parent are
# excluded: a bare `offline\.nortg_decomposition` also matches the SHELL that is running this
# script whenever the module name appears in its own command line, which would make the driver
# refuse to start because of itself. It still fails CLOSED -- an unrecognised match refuses --
# because overlapping two campaigns makes the artifact's concurrent-load statement false.
ALIVE=$(pgrep -f 'python.*offline\.nortg_decomposition' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: cells from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo '  Wait until `pgrep -f nortg_decomposition` is empty. Nothing has been consumed.' >&2
  exit 3
fi

TOKEN=$WORK/AUTHORISED_TO_RUN
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run authorisation token at $TOKEN" >&2
  echo "  The author authorises one run with:" >&2
  echo "    date -Is > $TOKEN" >&2
  echo "  The token is deleted on start, so it authorises exactly one run." >&2
  exit 2
fi
echo "=== authorised by token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
rm -f "$TOKEN"
echo "=== token consumed and deleted; a restart needs a new one"

# Only NOW may anything be created or cleared. A refused start above left FAILED, COMPLETE and the
# directory tree exactly as the previous run left them.
mkdir -p "$LOGS"
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

fail() { echo "CAMPAIGN FAILED at $1" | tee "$WORK/FAILED"; exit 1; }

# The pids of the cells currently in flight. Global, so the signal handler can reach them: bash
# traps do not see a caller's locals.
CELL_PIDS=()

# pre-flight M3. Ctrl-C in tmux signals the process GROUP, and a child started with `&` does not die
# with the shell -- the reviewer's three test children ran to completion after the driver exited 130.
# So the handler kills the cells explicitly, by pid, then the group; and it writes FAILED FIRST, so
# the record survives even if the group kill takes the script with it.
on_signal() {
  trap '' INT TERM
  echo "CAMPAIGN INTERRUPTED by a signal; stopping ${#CELL_PIDS[@]} cell(s)" | tee "$WORK/FAILED" >&2
  for pid in "${CELL_PIDS[@]:-}"; do [ -z "$pid" ] || kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${CELL_PIDS[@]:-}"; do [ -z "$pid" ] || kill -KILL "$pid" 2>/dev/null || true; done
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM

# The failure path's reaper. ⚠️ m4, corrected: `wait` below returns only once EVERY cell has
# finished, so by the time this runs the other four have already completed and written their chunks.
# It is a belt-and-braces kill of anything still alive, not a way to stop four healthy cells early.
reap() {
  local pids=("$@")
  for pid in "${pids[@]:-}"; do [ -z "$pid" ] || kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do [ -z "$pid" ] || wait "$pid" 2>/dev/null || true; done
}

STARTED=$(date +%s)

# ---------------------------------------------------------------------------
# THE 30 CELLS -- six groups of five concurrent seeds
# ---------------------------------------------------------------------------
# mix50 first, and both its arms before any other tier: it is the tier the -409 lives on, so an
# interrupted campaign still holds a complete contrast rather than half of three.
for TIER in mix50 mappo1000 random; do
  for METHOD in dt_nortg dt; do
    echo "=== $METHOD@$TIER: 5 seeds in parallel, each pinned to one torch thread"
    pids=()
    CELL_PIDS=()
    for SEED in "${SEEDS[@]}"; do
      $PY -m offline.nortg_decomposition "${COMMON[@]}" \
          run --method "$METHOD" --tier "$TIER" --seed "$SEED" \
          > "$LOGS/decomp_${METHOD}_${TIER}_seed${SEED}.log" 2>&1 &
      pids+=($!)
      CELL_PIDS+=($!)
    done
    failed=0
    for pid in "${pids[@]:-}"; do [ -z "$pid" ] || wait "$pid" || failed=1; done
    CELL_PIDS=()
    if [ "$failed" -ne 0 ]; then
      echo "--- tail of every log for $METHOD@$TIER ---" >&2
      for SEED in "${SEEDS[@]}"; do
        echo "  [$SEED] $(tail -3 "$LOGS/decomp_${METHOD}_${TIER}_seed${SEED}.log" | tr '\n' ' ')" >&2
      done
      reap "${pids[@]:-}"
      # `run` exits non-zero when a cell has ANY episode that does not reproduce the committed
      # value. The chunk is still written, so the evidence survives; chunk_is_reusable refuses to
      # skip it, so a restart re-runs it rather than inheriting it.
      fail "run $METHOD@$TIER"
    fi
    grep -h "s/episode" "$LOGS"/decomp_${METHOD}_${TIER}_seed*.log || true
  done
done

# ---------------------------------------------------------------------------
# REPORT -- the last stage, and it validates all 3,000 episodes before writing a byte
# ---------------------------------------------------------------------------
echo "=== assembling docs/data/p5_3b_decomposition.json (validates 3,000 episodes first)"
$PY -m offline.nortg_decomposition "${COMMON[@]}" report > "$LOGS/report.log" 2>&1 \
  || { tail -20 "$LOGS/report.log" >&2; fail "report"; }
tail -3 "$LOGS/report.log"

# ---------------------------------------------------------------------------
# MANIFEST -- written LAST, from a stable tree
# ---------------------------------------------------------------------------
# m1: `smoke/` is PRUNED. It is G2's evidence, produced at an earlier commit (5000cc26, not this
# run's), and listing it would make deleting it later break `sha256sum -c` on this campaign's own
# manifest -- turning a tidy-up into a false integrity failure. The write is tmp + mv so an
# interrupted manifest step cannot leave a half-written one that verifies nothing.
echo "=== writing output/SHA256SUMS_p5_3b_decomp.txt (smoke/ pruned)"
( cd "$MAIN/output" \
  && find p5_3b_decomp -path 'p5_3b_decomp/smoke' -prune -o \
       -type f \( -name '*.json' -o -name '*.log' \) -print \
     | LC_ALL=C sort | xargs sha256sum > SHA256SUMS_p5_3b_decomp.txt.tmp \
  && mv SHA256SUMS_p5_3b_decomp.txt.tmp SHA256SUMS_p5_3b_decomp.txt ) || fail "manifest write"
echo "    $(wc -l < "$MAIN/output/SHA256SUMS_p5_3b_decomp.txt") entries"

# Re-verify every manifest this campaign READ, in the tree it read them from (p5_3b.sh AMENDMENT C6).
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
  1. Commit docs/data/p5_3b_decomposition.json in the worktree.
  2. From the now-CLEAN tree, regenerate the campaign artifact:
       python -m offline.nortg_campaign --corpus-root <corpus> --output-root /home/filip/rltraffic/output \
           --work-dir /home/filip/rltraffic/output/p5_3b --out-dir <worktree>/docs/data report
     BRIEF_33 section 3.5 requires runtime.git_dirty == false, which is why this is a separate step.
  3. Diff the regenerated artifact against the merged one and check the changed paths are a SUBSET
     of section 3.5's enumeration:
       git show bd36a0a:docs/data/p5_3b_nortg.json > /tmp/p5_3b_nortg.bd36a0a.json
       python -m offline.nortg_decomposition diff-paths --baseline /tmp/p5_3b_nortg.bd36a0a.json \
           --candidate docs/data/p5_3b_nortg.json \
           --allow format_version \
                   predictions.Q1.holds predictions.Q1.holds_rule \
                   predictions.Q1.largest_limb.registered_tier predictions.Q1.largest_limb.as_registered \
                   predictions.Q1.smallest_limb.registered_tier predictions.Q1.smallest_limb.as_registered \
                   comparisons.mappo1000.definition_difference_decomposition \
                   comparisons.mix50.definition_difference_decomposition \
                   comparisons.random.definition_difference_decomposition \
                   mechanism runtime.
     ⚠️ mn-5: the enumeration is PER-TIER and per-field on purpose. A blanket `comparisons` prefix
     tolerates a moved mean_difference, and a blanket `predictions.Q1` tolerates a moved `largest`.
     Anything outside that set is a FINDING, not something to add to the list.
NEXT
