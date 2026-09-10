#!/usr/bin/env bash
# P5.3b campaign driver -- the dt_nortg arm on hz1x1, run from the task WORKTREE.
#
#   tmux new -s p53b
#   bash /home/filip/rltraffic-p53b/offline/campaigns/p5_3b.sh
#
# Fails closed; idempotent and restartable -- every stage is skipped when its chunk exists.
#
# ---------------------------------------------------------------------------
# AMENDMENT E5 -- TWO MECHANISMS, TWO JOBS. They are not interchangeable.
# ---------------------------------------------------------------------------
#   --corpus-root / --draws-root / --output-root   fix the CAMPAIGN's input paths
#   RLTRAFFIC_CORPUS_V11 / RLTRAFFIC_OUTPUT_ROOT   are what stop the GATE-2 and E1 TESTS skipping
# `RLTRAFFIC_CORPUS` appears zero times in nortg_campaign.py; it is a test-side gate only.
#
# The worktree carries none of the gitignored inputs (corpus 3.8 G, draws 179 M, reused
# checkpoints 477 M, P8.4b's 38,502 re-derived cells), so every INPUT root points at the main
# tree and every OUTPUT path stays in the worktree. Nothing here writes outside $WORK.
#
# ---------------------------------------------------------------------------
# GATE 0, DISCHARGED FROM COMMITTED CONSTANTS (AMENDMENT E3) -- not re-estimated
# ---------------------------------------------------------------------------
#   offline/att_rederivation.py:89  MEASURED_SECONDS_PER_EPISODE = {"hz1x1": 1.29, ...}
#   offline/att_rederivation.py:92  DEFAULT_WORKERS = 5
#   training    15 cells x 212.4 s (measured G0-c)      = 53.1 min serial
#   evaluation  15 cells x 100 draws x 1.29 s = 32.2 min serial -> 6.5 min at 5 workers
#   gate 1b      3 cells x 100 draws x 1.29 s =  6.5 min serial   (matches E1's own figure)
#   gate 2       1 retrain                     =  3.5 min
#   probe       15 cells x ~4 s                =  1.0 min
#   TOTAL                                      ~= 70 min
# NOTE: an earlier estimate of ~2 h used 500 episodes per cell. A P4.6 cell is 5 seeds x 100
# draws; a P5.3b cell is ONE seed x 100 draws. Quoting the committed rate caught that.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=/home/filip/rltraffic-p53b
PY=$MAIN/.venv/bin/python
WORK=$WORK_TREE/output/p5_3b
LOGS=$WORK/logs
SEEDS=(101 202 303 404 505)

COMMON=(--corpus-root $MAIN/datasets_v11
        --draws-root $MAIN/scenarios/draws
        --output-root $MAIN/output
        --out-dir $WORK_TREE/docs/data
        --work-dir $WORK
        --checkpoint-dir $WORK/checkpoints
        --torch-threads 1)

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RLTRAFFIC_CORPUS_V11=$MAIN/datasets_v11
export RLTRAFFIC_OUTPUT_ROOT=$MAIN/output

mkdir -p "$LOGS" "$WORK/checkpoints"
cd "$WORK_TREE"

fail() { echo "CAMPAIGN FAILED at $1" | tee "$WORK/FAILED"; exit 1; }

# ---------------------------------------------------------------------------
# ONE-SHOT RUN AUTHORISATION (2026-09-10). The author writes the token; the script
# requires it and DELETES it as its first action, so one token buys exactly one run.
#
# WHY: this campaign was started three times and only the first was instructed. The
# second and third were the implementer's own initiative, and the third came after the
# author had deliberately stopped the machine for the night, resting on a causal claim
# that turned out to be false.
#
# A token cannot make an uninstructed start IMPOSSIBLE. It makes it NON-REFLEXIVE, and
# reflex is what happened. Deleting it before any work begins means a crash-and-retry
# loop cannot re-arm itself.
#
#   To authorise one run:  date -Is > /home/filip/rltraffic-p53b/output/p5_3b/AUTHORISED_TO_RUN
# ---------------------------------------------------------------------------
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

# Only NOW may the run markers be cleared. The token check above is pure validation, so a
# refused start leaves FAILED exactly as the previous run left it -- filesystem-mutation
# barrier, applied to the driver itself.
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

# AMENDMENT C5: reap the fan-out's children before exiting, or a restart gives 8-10 concurrent
# evaluations on one GPU against this script's own one-thread-per-cell protocol.
reap() {
  local pids=("$@")
  for pid in "${pids[@]:-}"; do [ -z "$pid" ] || kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do [ -z "$pid" ] || wait "$pid" 2>/dev/null || true; done
}

# Gate 1 (reused identity at consumption) + Gate 1b (three re-rolls, the INSTRUMENT check, E1).
if [ ! -f "$WORK/gate1.json" ]; then
  echo "=== gate 1 + 1b: reused dt identity, and one re-roll per tier through evaluate_cell"
  $PY -m offline.nortg_campaign "${COMMON[@]}" gate1 > "$LOGS/gate1.log" 2>&1 || fail "gate1"
fi

# Gate 2: the control cell retrained through the modified train_dt.
if [ ! -f "$WORK/control.json" ]; then
  echo "=== gate 2: control retrain of mappo500 seed 101 against its committed digest"
  $PY -m offline.nortg_campaign "${COMMON[@]}" control > "$LOGS/control.log" 2>&1 || fail "control"
fi

# Training -- serial: five concurrent CUDA jobs contend and the column is only ~53 min.
# Widest row-B spread first, so an interrupted campaign still spans the axis.
for TIER in mix50 mappo1000 random; do
  if [ ! -f "$WORK/train_$TIER.json" ]; then
    echo "=== training $TIER (5 seeds, rtg_mode=zero, 40000 steps)"
    $PY -m offline.nortg_campaign "${COMMON[@]}" train --tier "$TIER" \
        > "$LOGS/train_$TIER.log" 2>&1 || fail "train $TIER"
  fi
done

# Evaluation -- five cells concurrent, each pinned to one torch thread (P4.6's protocol).
for TIER in mix50 mappo1000 random; do
  echo "=== evaluating $TIER (5 seeds in parallel, each pinned to one torch thread)"
  pids=()
  for SEED in "${SEEDS[@]}"; do
    [ -f "$WORK/eval_${TIER}_seed${SEED}.json" ] && continue
    $PY -m offline.nortg_campaign "${COMMON[@]}" evaluate --tier "$TIER" --seed "$SEED" \
        > "$LOGS/eval_${TIER}_seed${SEED}.log" 2>&1 &
    pids+=($!)
  done
  failed=0
  for pid in "${pids[@]:-}"; do [ -z "$pid" ] || wait "$pid" || failed=1; done
  if [ "$failed" -ne 0 ]; then reap "${pids[@]:-}"; fail "evaluate $TIER"; fi
done

# Gate 3 -- arm validity, enforced where it is measured (AMENDMENT C3).
for TIER in mappo1000 mix50 random; do
  if [ ! -f "$WORK/probe_$TIER.json" ]; then
    echo "=== gate 3: probing $TIER"
    $PY -m offline.nortg_campaign "${COMMON[@]}" probe --tier "$TIER" \
        > "$LOGS/probe_$TIER.log" 2>&1 || fail "probe $TIER"
  fi
done

echo "=== assembling docs/data/p5_3b_nortg.json"
$PY -m offline.nortg_campaign "${COMMON[@]}" report > "$LOGS/report.log" 2>&1 || fail "report"
tail -8 "$LOGS/report.log"

echo "=== writing output/SHA256SUMS_p5_3b.txt"
( cd "$WORK_TREE/output" && find p5_3b -type f \( -name '*.pt' -o -name '*.json' \) \
    | LC_ALL=C sort | xargs sha256sum > SHA256SUMS_p5_3b.txt )
wc -l < "$WORK_TREE/output/SHA256SUMS_p5_3b.txt"

# AMENDMENT C6 -- re-verify every manifest the campaign READ, in the tree it read them from.
echo "=== re-verifying the reused manifests in $MAIN/output"
( cd "$MAIN/output"
  status=0
  for M in SHA256SUMS_*.txt; do
    if out=$(sha256sum -c "$M" 2>&1); then
      printf '  %-28s %s OK\n' "$M" "$(printf '%s\n' "$out" | grep -c ': OK$')"
    else
      printf '  %-28s FAILED\n' "$M"; printf '%s\n' "$out" | grep -v ': OK$' | head -5; status=1
    fi
  done
  exit $status
) || fail "manifest re-verification"

echo "CAMPAIGN COMPLETE" | tee "$WORK/COMPLETE"
