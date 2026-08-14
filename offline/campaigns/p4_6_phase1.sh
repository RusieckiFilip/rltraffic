#!/usr/bin/env bash
#
# P4.6 PHASE 1 CAMPAIGN -- launch this in tmux, from the user's own shell.
# =====================================================================================
#   tmux new -s p46
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#   bash /home/filip/rltraffic-p46/offline/campaigns/p4_6_phase1.sh 2>&1 | tee \
#        /home/filip/rltraffic-p46/output/p4_6/logs/phase1.log
#   # detach with C-b d ; reattach with: tmux attach -t p46
#
# BRIEF_17 section 12: the implementer writes this script, the USER launches it, and the
# implementer never sleep-polls it.  CLAUDE.md section 5 already bound this ("long simulation runs
# ... go to a tmux session started by the user").
#
# WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
# -----------------------------------------------------------------------------------
# Does:  the 4 evaluations of the already-trained `random` tier, then train+evaluate `fixedtime`,
#        `maxpressure` and `mappo500`, then assemble the report.
# Does NOT: retrain `random` (20 checkpoints exist), re-roll either behaviour cell (both exist),
#        re-run Gate G (PASS, and re-running a passed gate is a second measurement of it), or
#        touch phase 2 -- the mixture tiers are a separate script by section 12's condition 6.
#
# RESUMABLE AT TIER GRANULARITY (section 12.4 as scoped down): a tier whose 20 checkpoints exist is
# not retrained and a cell whose eval JSON exists is not re-rolled; every skip is logged BY NAME.
#
# FAILS CLOSED: `set -euo pipefail`, and a final assertion that the completed cells equal the
# requested cells, exiting non-zero on any mismatch (section 12, condition 2 -- PROJECT_PLAN
# section 7's rule of 2026-08-06, which a campaign has already violated once by running "on to a
# clean-looking end" with half its output missing).
#
# EXPECTED WALL CLOCK: about 3 h 20 m -- 3 tiers x ~60 min training (BC ~100 s, %BC ~100 s,
# IQL ~320 s, DT ~210 s per seed, five seeds each) plus 4 x ~10 min of evaluation per tier, the
# four cells of a tier running concurrently, each pinned to one torch thread.
#
set -euo pipefail

ROOT=/home/filip/rltraffic-p46
PY=/home/filip/rltraffic/.venv/bin/python
CORPUS=/home/filip/rltraffic/datasets_v11
WORK=$ROOT/output/p4_6
LOGS=$WORK/logs

# Condition 3: the pin is re-asserted INSIDE the script, not only exported in the tmux shell.
# DEFERRED 41 has two sightings, one inside pytest; an unpinned job wedges at ~0 % CPU.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

COMMON=(--corpus-root "$CORPUS"
        --draws-root "$ROOT/scenarios/draws"
        --out-dir "$ROOT/docs/data"
        --work-dir "$WORK"
        --checkpoint-dir "$WORK/checkpoints"
        --torch-threads 1)

TIERS=(random fixedtime maxpressure mappo500)
METHODS=(bc bc_top10 iql dt)

mkdir -p "$LOGS"
cd "$ROOT"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
say()   { echo "[$(stamp)] $*"; }

say "P4.6 phase 1 starting; pin OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
say "tiers: ${TIERS[*]}   methods: ${METHODS[*]}"

# -------------------------------------------------------------------------------------
# Preconditions: this script measures cells, it does not re-establish the gate.
# -------------------------------------------------------------------------------------
[ -f "$WORK/gate.json" ] || { say "FATAL: $WORK/gate.json is missing -- Gate G must pass first"; exit 2; }
$PY - "$WORK/gate.json" <<'PYEOF'
import json, sys
gate = json.load(open(sys.argv[1]))
if gate.get("status") != "PASS":
    raise SystemExit(f"FATAL: Gate G status is {gate.get('status')!r}, not PASS")
PYEOF
say "Gate G: PASS (checkpoint identity + 100 re-rolled cells, verified before this campaign)"

for f in eval_random_behaviour.json eval_fixedtime_behaviour.json; do
  [ -f "$WORK/$f" ] || { say "FATAL: $WORK/$f is missing -- the behaviour cells are inputs here"; exit 2; }
done
say "behaviour cells present: random (n=500) and fixedtime (n=100); neither is re-rolled"

# -------------------------------------------------------------------------------------
# Training and evaluation, tier by tier, in the declared order of docs/plans/p4.6.md 12.2
# -------------------------------------------------------------------------------------
for TIER in "${TIERS[@]}"; do
  # find, not ls+glob: an unmatched glob under `set -o pipefail` would abort the campaign
  TRAINED=$(find "$WORK/checkpoints" -maxdepth 1 -name "${TIER}_*_seed*.pt" | wc -l)
  if [ "$TRAINED" -eq 20 ]; then
    say "SKIP training $TIER: 20 checkpoints already on disk"
  else
    say "TRAIN $TIER (4 methods x 5 seeds, 40,000 steps each)"
    $PY -m offline.method_tier_grid "${COMMON[@]}" train --tier "$TIER" \
        > "$LOGS/train_$TIER.log" 2>&1
    say "TRAIN $TIER COMPLETE"
  fi

  PIDS=()
  RUNNING=()
  for METHOD in "${METHODS[@]}"; do
    if [ -f "$WORK/eval_${TIER}_${METHOD}.json" ]; then
      say "SKIP evaluate ${METHOD}@${TIER}: cell already on disk"
      continue
    fi
    $PY -m offline.method_tier_grid "${COMMON[@]}" evaluate --tier "$TIER" --method "$METHOD" \
        > "$LOGS/eval_${TIER}_${METHOD}.log" 2>&1 &
    PIDS+=("$!")
    RUNNING+=("${METHOD}@${TIER}")
  done
  if [ "${#PIDS[@]}" -gt 0 ]; then
    say "EVALUATE ${RUNNING[*]} (concurrent, one torch thread each)"
    STATUS=0
    for i in "${!PIDS[@]}"; do
      # Every pid is waited on before the script reacts, so one failure cannot leave the others
      # orphaned and unreported.
      if ! wait "${PIDS[$i]}"; then
        say "FAILED: ${RUNNING[$i]} -- see $LOGS/eval_${TIER}_*.log"
        STATUS=1
      fi
    done
    [ "$STATUS" -eq 0 ] || exit 1
  fi
  say "TIER $TIER COMPLETE (training + 4 cells)"
done

# -------------------------------------------------------------------------------------
# Verify by EFFECT, not by status: completed cells must equal requested cells.
# -------------------------------------------------------------------------------------
say "verifying the campaign by its artifacts"
$PY - "$WORK" "${TIERS[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
from offline.method_tier_grid import METHODS, REUSED_TIER

work = Path(sys.argv[1])
tiers = sys.argv[2:]

# Requested is derived from the DECLARATION -- the tier list this script was asked to run, the
# registered seeds and the registered held-out pool -- and never from the files being checked.
requested = {
    (tier, method) for tier in tiers for method in METHODS if tier != REUSED_TIER
}
completed: set[tuple[str, str]] = set()
problems: list[str] = []

for tier, method in sorted(requested):
    path = work / f"eval_{tier}_{method}.json"
    if not path.is_file():
        problems.append(f"{method}@{tier}: no cell at {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    want = {(int(s), int(d)) for s in TRAINING_SEEDS for d in HELD_OUT_DRAWS}
    got = {(int(e["seed"]), int(e["draw_id"])) for e in episodes}
    if got != want:
        problems.append(
            f"{method}@{tier}: {len(got)} episodes against {len(want)} requested "
            f"(missing {len(want - got)}, unexpected {len(got - want)})"
        )
        continue
    if int(payload["declared_gradient_steps"]) != 40_000:
        problems.append(f"{method}@{tier}: budget {payload['declared_gradient_steps']}, not 40000")
        continue
    completed.add((tier, method))

training = json.loads((Path("docs/data/p4_6_training.json")).read_text(encoding="utf-8"))
runs = {(r["tier"], r["method"], int(r["seed"])) for r in training["runs"]}
for tier in tiers:
    if tier == REUSED_TIER:
        continue
    for method in METHODS:
        for seed in TRAINING_SEEDS:
            if (tier, method, int(seed)) not in runs:
                problems.append(f"{method}@{tier} seed {seed}: no training record")

print(f"requested cells: {len(requested)}   completed cells: {len(completed)}")
for tier in tiers:
    done = sorted(m for (t, m) in completed if t == tier)
    print(f"  {tier:12s} {len(done)}/4  {done}")
if problems or completed != requested:
    print("\nCAMPAIGN INCOMPLETE:")
    for problem in problems[:20]:
        print(f"  - {problem}")
    raise SystemExit(1)
print("completed == requested, every cell holds 5 seeds x 100 draws at 40,000 steps")
PYEOF

say "ALL CELLS PRESENT -- assembling the report"
$PY -m offline.method_tier_grid "${COMMON[@]}" report --tiers "$(IFS=,; echo "${TIERS[*]}"),mappo1000" \
    > "$LOGS/report.log" 2>&1
tail -12 "$LOGS/report.log"

say "PHASE 1 COMPLETE"
echo "COMPLETE $(stamp)" > "$WORK/PHASE1_COMPLETE"
