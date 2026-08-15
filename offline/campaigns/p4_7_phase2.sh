#!/usr/bin/env bash
#
# P4.7 PHASE 2 CAMPAIGN -- the three mixture tiers.  Launch this in tmux, from the user's own shell.
# =====================================================================================
#   tmux new -s p47
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#   mkdir -p /home/filip/rltraffic-p47/output/p4_7/logs        # <-- REQUIRED BEFORE THE PIPE
#   bash /home/filip/rltraffic-p47/offline/campaigns/p4_7_phase2.sh 2>&1 | tee \
#        /home/filip/rltraffic-p47/output/p4_7/logs/phase2.log
#   # detach with C-b d ; reattach with: tmux attach -t p47
#
# ⚠️ THE mkdir LINE IS NOT DECORATION.  The shell opens `tee`'s target BEFORE the script runs, so a
# log directory this script creates does not yet exist when the pipeline is built.  Measured on the
# real launch of 2026-08-15: `tee` failed with "No such file or directory", the campaign ran to
# completion regardless, and `phase2.log` was never written -- it was recovered from the tmux pane
# afterwards.  Same content, different provenance, and P4.6's phase1.log was inside its checksums.
#
# BRIEF_17 section 12, inherited by BRIEF_19 section 7: the implementer writes this script, the USER
# launches it, and the implementer never sleep-polls it.  CLAUDE.md section 5 already bound this.
# ⚠️ NO `until`-POLL ANYWHERE, deliberately: section 12.2 measured `pgrep -f` matching the polling
# shell's OWN command line, so `! pgrep ...` is always false and the loop hangs forever if the job
# dies without writing its file.  That hang is live in this repo's history, twice.
#
# WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
# -----------------------------------------------------------------------------------
# Does:  train and evaluate the 12 new cells -- 3 mixture tiers x 4 methods x 5 seeds, 40,000
#        gradient steps each, on the registered held-out pool (draws 1000-1099).
# Does NOT: re-run Gate G or Gate D (both PASS on disk, and re-running a passed gate is a second
#        measurement of it, not a check on this one); re-evaluate any phase-1 cell (all 20 are
#        RE-USED under Gate P1's bit-identity check); build the constructed behaviour reference or
#        the report (both cost zero compute and are done in-session, where their assertions can be
#        read); touch docs/data/p4_6_*.json -- every artifact here is written under
#        --artifact-prefix p4_7.
#
# RESUMABLE AT TIER AND CELL GRANULARITY (section 12.4 as scoped down by 12.1): a tier whose 20
# checkpoints exist is not retrained and a cell whose eval JSON exists is not re-rolled; every skip
# is logged BY NAME.  Nothing already computed is recomputed.
#
# FAILS CLOSED: `set -euo pipefail`, and a final assertion that the completed cells equal the
# requested cells -- derived from the DECLARATION, never from the files being checked -- exiting
# non-zero on any mismatch (section 12 condition 2; PROJECT_PLAN section 7's rule of 2026-08-06,
# which a campaign has already violated once by running "on to a clean-looking end" with half its
# output missing).
#
# EXPECTED WALL CLOCK: about 3 h 45 m -- 3 tiers x ~60 min training (BC ~100 s, %BC ~100 s, IQL
# ~320 s, DT ~210 s per seed, five seeds each) plus ~15 min of evaluation per tier, the four cells
# of a tier running concurrently, each pinned to one torch thread.
#
set -euo pipefail

ROOT=/home/filip/rltraffic-p47
PY=/home/filip/rltraffic/.venv/bin/python
CORPUS=/home/filip/rltraffic/datasets_v11
WORK=$ROOT/output/p4_7
LOGS=$WORK/logs

# Condition 3: the pin is re-asserted INSIDE the script, not only exported in the tmux shell.
# DEFERRED 41 has two sightings, one inside pytest; an unpinned job wedges at ~0 % CPU and costs the
# whole campaign.  Exporting it here turns "remember to pin each job" into "the environment is
# pinned", which is this project's standing preference for mechanism over intention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

COMMON=(--corpus-root "$CORPUS"
        --draws-root "$ROOT/scenarios/draws"
        --out-dir "$ROOT/docs/data"
        --work-dir "$WORK"
        --checkpoint-dir "$WORK/checkpoints"
        --artifact-prefix p4_7
        --torch-threads 1)

TIERS=(mix33 mix50 mix67)
METHODS=(bc bc_top10 iql dt)

# ⚠️ The checkpoint directory is created HERE, not by the first trainer.  Measured before this
# script was handed over: `find <missing dir> ... | wc -l` exits 1 under `set -o pipefail`, so the
# resumability probe below would abort the campaign at the first tier -- on a fresh worktree, i.e.
# exactly the case P4.7 runs in.  P4.6's script carried the same construct and never hit it because
# its directory already existed.
mkdir -p "$LOGS" "$WORK/checkpoints"
cd "$ROOT"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
say()   { echo "[$(stamp)] $*"; }

say "P4.7 phase 2 starting; pin OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
say "tiers: ${TIERS[*]}   methods: ${METHODS[*]}"

# -------------------------------------------------------------------------------------
# Preconditions: this script measures cells, it does not establish a gate.
# -------------------------------------------------------------------------------------
for GATE in gate.json gate_d.json; do
  [ -f "$WORK/$GATE" ] || { say "FATAL: $WORK/$GATE is missing -- it must pass before any cell"; exit 2; }
done
[ -f "$ROOT/docs/data/p4_7_declaration.json" ] || {
  say "FATAL: docs/data/p4_7_declaration.json is missing -- the declaration is written BEFORE training"
  exit 2
}
$PY - "$WORK/gate.json" "$WORK/gate_d.json" <<'PYEOF'
import json, sys
for path in sys.argv[1:]:
    payload = json.load(open(path))
    if payload.get("status") != "PASS":
        raise SystemExit(f"FATAL: {path} status is {payload.get('status')!r}, not PASS")
PYEOF
say "Gate G and Gate D: PASS"

DRAWS=$(find "$ROOT/scenarios/draws/cityflow1x1" -maxdepth 1 -type d -name 'draw_10??' | wc -l)
[ "$DRAWS" -eq 100 ] || { say "FATAL: $DRAWS materialised held-out draws, expected 100"; exit 2; }
say "held-out demand present: 100 draws, Gate D verified against the 5 survivors"

# -------------------------------------------------------------------------------------
# Training and evaluation, tier by tier, in expert-fraction order
# -------------------------------------------------------------------------------------
for TIER in "${TIERS[@]}"; do
  # find, not ls+glob: an unmatched glob under `set -o pipefail` would abort the campaign
  TRAINED=$(find "$WORK/checkpoints" -maxdepth 1 -name "${TIER}_*_seed*.pt" 2>/dev/null | wc -l)
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
from offline.method_tier_grid import METHODS, MIXTURE_EXPERT_FRACTION

work = Path(sys.argv[1])
tiers = sys.argv[2:]

# Requested is derived from the DECLARATION -- the registered tier set, seeds and held-out pool --
# and never from the files being checked.
unknown = [t for t in tiers if t not in MIXTURE_EXPERT_FRACTION]
if unknown:
    raise SystemExit(f"FATAL: {unknown} are not mixture tiers")
requested = {(tier, method) for tier in tiers for method in METHODS}
completed: set[tuple[str, str]] = set()
problems: list[str] = []

for tier, method in sorted(requested):
    path = work / f"eval_{tier}_{method}.json"
    if not path.is_file():
        problems.append(f"{method}@{tier}: no cell at {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    want = {(int(s), int(d)) for s in TRAINING_SEEDS for d in HELD_OUT_DRAWS}
    got = {(int(e["seed"]), int(e["draw_id"])) for e in payload["episodes"]}
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

training = json.loads(Path("docs/data/p4_7_training.json").read_text(encoding="utf-8"))
runs = {(r["tier"], r["method"], int(r["seed"])) for r in training["runs"]}
for tier in tiers:
    for method in METHODS:
        for seed in TRAINING_SEEDS:
            if (tier, method, int(seed)) not in runs:
                problems.append(f"{method}@{tier} seed {seed}: no training record")

print(f"requested cells: {len(requested)}   completed cells: {len(completed)}")
for tier in tiers:
    done = sorted(m for (t, m) in completed if t == tier)
    print(f"  {tier:8s} {len(done)}/4  {done}")
if problems or completed != requested:
    print("\nCAMPAIGN INCOMPLETE:")
    for problem in problems[:20]:
        print(f"  - {problem}")
    raise SystemExit(1)
print("completed == requested, every cell holds 5 seeds x 100 draws at 40,000 steps")
PYEOF

say "ALL 12 CELLS PRESENT"
say "NEXT, in-session and costing no compute: the constructed behaviour reference, Gate P1 and the report"
echo "COMPLETE $(stamp)" > "$WORK/PHASE2_COMPLETE"
