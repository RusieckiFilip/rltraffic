#!/usr/bin/env bash
#
# P5.1 CAMPAIGN -- the spatial mixing layer on cf_grid4x4__mappo1000.  Launch this in tmux, from
# the user's own shell.
# =====================================================================================
#   tmux new -s p51
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#   mkdir -p /home/filip/rltraffic-p51/output/p5_1/logs        # <-- REQUIRED BEFORE THE PIPE
#   bash /home/filip/rltraffic-p51/offline/campaigns/p5_1.sh 2>&1 | tee \
#        /home/filip/rltraffic-p51/output/p5_1/logs/campaign.log
#   # detach with C-b d ; reattach with: tmux attach -t p51
#
# ⚠️ THE mkdir LINE IS NOT DECORATION.  The shell opens `tee`'s target BEFORE the script runs, so a
# log directory this script creates does not yet exist when the pipeline is built.  Measured on
# P4.7's real launch of 2026-08-15: `tee` failed with "No such file or directory", the campaign ran
# to completion regardless, and phase2.log was never written -- it was recovered from the tmux pane
# afterwards.  Same content, different provenance.
#
# BRIEF_17 section 12, inherited by BRIEF_22 section 7: the implementer writes this script, the
# USER launches it, and the implementer never sleep-polls it.  CLAUDE.md section 5 already bound
# this.  ⚠️ NO `until`-POLL ANYWHERE, deliberately: section 12.2 measured `pgrep -f` matching the
# polling shell's OWN command line, so `! pgrep ...` is always false and the loop hangs forever if
# the job dies without writing its file.  That hang is live in this repo's history, twice.
#
# WHAT IT DOES
# -----------------------------------------------------------------------------------
# Does:  train and evaluate 5 method arms x 5 seeds at 40,000 gradient steps each on the
#        registered held-out pool (draws 1000-1099), plus the MAPPO@1000 behaviour anchor and the
#        `random` reference the declared collapse criterion is scored against.
# Does NOT: touch docs/data/p4_*.json or any merged artifact; re-materialise draws (done in-session
#        and byte-verified against the five survivors in the main tree); run any tier but
#        mappo1000 -- section 10's CUT 2 keeps P5.1 on ONE tier and drops the sweep.
#
# RESUMABLE AT SEED GRANULARITY FOR TRAINING, CELL GRANULARITY FOR EVALUATION.  Every skip is
# logged BY NAME and nothing already computed is recomputed:
#   * each training seed whose checkpoint exists AT THE DECLARED BUDGET is skipped, so a crash at
#     seed four of five costs one seed (~85 min) and not one arm (~7 h).  Arm granularity alone is
#     what this script had first, and it would have thrown away up to four hours of finished work;
#   * a checkpoint that exists but records a DIFFERENT budget, or the other arm's mixing flag, is
#     REFUSED rather than reused or overwritten -- the concrete hazard is a rehearsal or smoke run
#     whose 3-step checkpoints sit in the campaign's directory;
#   * training writes to a `.partial` sibling and renames on success, because `torch.save` is not
#     atomic and a kill mid-save would otherwise leave a truncated file that aborts the resume;
#   * each evaluation cell whose `eval_<arm>.json` exists is not re-rolled, and that artifact is
#     written through `write_json_atomic` (temp file + `os.replace`), so a kill leaves the previous
#     file or nothing, never a partial cell.
#
# FAILS CLOSED: `set -euo pipefail`, and a final assertion that the completed cells equal the cells
# derived from the DECLARATION, never from the files being checked -- exiting non-zero on any
# mismatch (PROJECT_PLAN section 7's rule of 2026-08-06, which a campaign has already violated once
# by running "on to a clean-looking end" with half its output missing).
#
# EXPECTED WALL CLOCK: about 17 h.  Training measured on this machine at 127.5 ms/step for
# dt_spatial and 107.1 ms/step for dt_nomix (batch 64 joint windows, RTX 5080), i.e. 85.0 and 71.4
# min per seed -- about 13 h for the two spatial arms across five seeds.  The three baselines add
# roughly 45 min, evaluation roughly 2.5 h and the two anchors roughly 1 h.
# ⚠️ ONLY the two ms/step figures are measurements; everything else here is an estimate.
#
# 🚨 CUDA IS MANDATORY.  The same model measures 5440 ms/step on CPU -- 60 h per seed, 25 days for
# the campaign.  The script refuses to start without a visible GPU rather than silently running a
# job that cannot finish.
#
set -euo pipefail

ROOT=/home/filip/rltraffic-p51
PY=/home/filip/rltraffic/.venv/bin/python
CORPUS=/home/filip/rltraffic/datasets_v11

# REHEARSAL KNOBS.  Both default to the declared campaign values and exist so this script's own
# control flow -- the resume branches and the completeness assertion -- can be EXECUTED before an
# overnight handover rather than read.  PROJECT_PLAN section 7 (2026-08-14): a declared path with
# no execution behind it is untested however carefully it was written, and P4.7's resumability
# probe aborted at tier 1 on exactly this class, after handover.
#
# ⚠️ A REHEARSAL CANNOT MASQUERADE AS A CAMPAIGN.  The completeness assertion below reads the
# declared budget out of every cell and refuses anything but 40,000, so a reduced-budget run exits
# non-zero and CAMPAIGN_COMPLETE is never written.  The per-seed resume guard additionally REFUSES
# to reuse a checkpoint whose recorded budget is not the declared one, so rehearsal checkpoints
# left in a campaign directory abort the campaign loudly instead of entering a reported cell.
STEPS=${P51_STEPS:-40000}
WORK=${P51_WORK:-$ROOT/output/p5_1}
# The report lands here.  Overridable for the same reason as WORK: a rehearsal that reaches the
# report must NOT write docs/data/p5_1_grid.json into the repository.
OUT=${P51_OUT:-$ROOT/docs/data}
LOGS=$WORK/logs

# The pin is re-asserted INSIDE the script, not only exported in the tmux shell.  DEFERRED 41 has
# two sightings, one inside pytest; an unpinned job wedges at ~0 % CPU and costs the whole
# campaign.  Exporting it here turns "remember to pin each job" into "the environment is pinned".
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

COMMON=(--corpus-root "$CORPUS"
        --draws-root "$ROOT/scenarios/draws"
        --out-dir "$OUT"
        --work-dir "$WORK"
        --checkpoint-dir "$WORK/checkpoints"
        --gradient-steps "$STEPS"
        --torch-threads 1)

SPATIAL_ARMS=(dt_spatial dt_nomix)
BASELINE_ARMS=(bc bc_top10 iql)
ANCHORS=(behaviour random)
ALL_ARMS=("${SPATIAL_ARMS[@]}" "${BASELINE_ARMS[@]}" "${ANCHORS[@]}")
SEEDS=(101 202 303 404 505)

mkdir -p "$LOGS" "$WORK/checkpoints"
cd "$ROOT"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
say()   { echo "[$(stamp)] $*"; }

if [ "$STEPS" -ne 40000 ]; then
  say "################################################################"
  say "REHEARSAL: STEPS=$STEPS, not the declared 40000."
  say "This exercises the control flow ONLY.  The completeness assertion"
  say "will refuse these cells and CAMPAIGN_COMPLETE will NOT be written."
  say "################################################################"
fi
say "P5.1 campaign starting; pin OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
say "work dir: $WORK   out dir: $OUT"
say "arms: ${ALL_ARMS[*]}   seeds: ${SEEDS[*]}"

# -------------------------------------------------------------------------------------
# Preconditions.  Each is a refusal, not a warning.
# -------------------------------------------------------------------------------------
$PY - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit(
        "FATAL: no CUDA device. The spatial model measures 5440 ms/step on CPU -- 60 h per seed. "
        "Refusing to start a job that cannot finish rather than running it silently."
    )
print(f"CUDA: {torch.cuda.get_device_name(0)}")
PYEOF
say "CUDA present"

DRAWS=$(find "$ROOT/scenarios/draws/cityflow_grid4x4" -maxdepth 1 -type d -name 'draw_10??' | wc -l)
[ "$DRAWS" -eq 100 ] || { say "FATAL: $DRAWS materialised held-out draws, expected 100"; exit 2; }
say "held-out demand present: 100 draws, byte-verified against the 5 survivors in the main tree"

[ -f "$ROOT/docs/data/p5_1_declaration.json" ] || {
  say "FATAL: docs/data/p5_1_declaration.json is missing -- the declaration is written BEFORE training"
  exit 2
}
say "declaration present"

# -------------------------------------------------------------------------------------
# Training.  The two spatial arms first: they are the primary comparison and the long pole.
# -------------------------------------------------------------------------------------
for ARM in "${SPATIAL_ARMS[@]}"; do
  # find, not ls+glob: an unmatched glob under `set -o pipefail` would abort the campaign.
  # ⚠️ No `2>/dev/null` here, deliberately.  With it, a missing checkpoints directory aborts the
  # script SILENTLY -- measured: exit 1 straight after "declaration present", with no reason on
  # screen.  The mkdir above makes that unreachable, but a guard whose failure is invisible is
  # how P4.7's probe went unnoticed until after handover.
  # This shell guard is only a FAST PATH that avoids rebuilding the dataset when an arm is wholly
  # done.  The real resumability is per SEED and lives in offline.spatial_mixing._run_train, which
  # skips each checkpoint already on disk at the declared budget and refuses one that disagrees.
  # Arm granularity alone would lose up to four hours when a run dies at seed four of five.
  TRAINED=$(find "$WORK/checkpoints" -maxdepth 1 -name "grid4x4_mappo1000_${ARM}_seed*.pt" | wc -l)
  if [ "$TRAINED" -eq 5 ]; then
    say "SKIP training $ARM: 5 checkpoints already on disk"
  else
    say "TRAIN $ARM (5 seeds x 40,000 steps; ~85 min/seed measured for dt_spatial)"
    $PY -m offline.spatial_mixing "${COMMON[@]}" train --method "$ARM" \
        > "$LOGS/train_$ARM.log" 2>&1
    say "TRAIN $ARM COMPLETE"
  fi
done

BASE_TRAINED=$(find "$WORK/checkpoints" -maxdepth 1 -name "grid4x4_mappo1000_bc_seed*.pt" -o \
                    -name "grid4x4_mappo1000_bc_top10_seed*.pt" -o \
                    -name "grid4x4_mappo1000_iql_seed*.pt" | wc -l)
if [ "$BASE_TRAINED" -eq 15 ]; then
  say "SKIP training baselines: 15 checkpoints already on disk"
else
  say "TRAIN baselines (bc, bc_top10, iql -- independent per intersection BY CONSTRUCTION)"
  $PY -m offline.spatial_mixing "${COMMON[@]}" train-baselines \
      > "$LOGS/train_baselines.log" 2>&1
  say "TRAIN baselines COMPLETE"
fi

# -------------------------------------------------------------------------------------
# Evaluation.  One process per arm, run sequentially: each already holds the GPU.
# -------------------------------------------------------------------------------------
for ARM in "${ALL_ARMS[@]}"; do
  if [ -f "$WORK/eval_${ARM}.json" ]; then
    say "SKIP evaluate $ARM: cell already on disk"
    continue
  fi
  say "EVALUATE $ARM (5 seeds x 100 held-out draws)"
  $PY -m offline.spatial_mixing "${COMMON[@]}" evaluate --method "$ARM" \
      > "$LOGS/eval_$ARM.log" 2>&1
  say "EVALUATE $ARM COMPLETE"
done

# -------------------------------------------------------------------------------------
# Verify by EFFECT, not by status: completed cells must equal cells derived from the declaration.
# -------------------------------------------------------------------------------------
say "verifying the campaign by its artifacts"
$PY - "$WORK" <<'PYEOF'
import json
import sys
from pathlib import Path

from offline.dt_gate import HELD_OUT_DRAWS, TRAINING_SEEDS
from offline.spatial_mixing import BEHAVIOUR_METHOD, COLLAPSE_REFERENCE_METHOD, METHODS

work = Path(sys.argv[1])

# Requested is derived from the DECLARATION -- the registered arm set, seeds and held-out pool --
# and never from the files being checked.
requested = {*METHODS, BEHAVIOUR_METHOD, COLLAPSE_REFERENCE_METHOD}
completed: set[str] = set()
problems: list[str] = []

for arm in sorted(requested):
    path = work / f"eval_{arm}.json"
    if not path.is_file():
        problems.append(f"{arm}: no cell at {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    want = {(int(s), int(d)) for s in TRAINING_SEEDS for d in HELD_OUT_DRAWS}
    got = {(int(e["seed"]), int(e["draw_id"])) for e in payload["episodes"]}
    if got != want:
        problems.append(
            f"{arm}: {len(got)} episodes against {len(want)} requested "
            f"(missing {len(want - got)}, unexpected {len(got - want)})"
        )
        continue
    if arm in METHODS and int(payload["declared_gradient_steps"]) != 40_000:
        problems.append(f"{arm}: budget {payload['declared_gradient_steps']}, not 40000")
        continue
    completed.add(arm)

print(f"requested arms: {len(requested)}   completed arms: {len(completed)}")
for arm in sorted(requested):
    print(f"  {arm:12s} {'OK' if arm in completed else 'MISSING'}")
if problems or completed != requested:
    print("\nCAMPAIGN INCOMPLETE:")
    for problem in problems[:20]:
        print(f"  - {problem}")
    raise SystemExit(1)
print("completed == requested, every cell holds 5 seeds x 100 draws")
PYEOF

say "ALL 7 CELLS PRESENT"
say "REPORT (costs no compute; its assertions are readable in-session)"
$PY -m offline.spatial_mixing "${COMMON[@]}" report > "$LOGS/report.log" 2>&1
cat "$LOGS/report.log"
echo "COMPLETE $(stamp)" > "$WORK/CAMPAIGN_COMPLETE"
say "P5.1 campaign complete"
