#!/usr/bin/env bash
#
# P5.2 CAMPAIGN -- the spatial DT across the grid4x4 ladder, and the head-count 2x2.
# Launch this in tmux, from the user's own shell.
# =====================================================================================
#   tmux new -s p52
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#   mkdir -p /home/filip/rltraffic-p52/output/p5_2/logs      # <-- REQUIRED BEFORE THE PIPE
#   bash /home/filip/rltraffic-p52/offline/campaigns/p5_2.sh 2>&1 | tee \
#        /home/filip/rltraffic-p52/output/p5_2/logs/campaign.log
#   # detach with C-b d ; reattach with: tmux attach -t p52
#
# ⚠️ THE mkdir LINE IS NOT DECORATION.  The shell opens `tee`'s target BEFORE the script runs, so a
# log directory this script creates does not yet exist when the pipeline is built.  This has cost
# this project a log TWICE -- P4.7 on 2026-08-15 and P5.1 on 2026-08-17 -- and both were recovered
# only from the tmux pane afterwards.  This script also never clears the log directory afterwards.
#
# ⚠️ DO NOT export CUBLAS_WORKSPACE_CONFIG in the launch shell.  E1 measures the envelope of the
# regime P5.1 ran in, and P5.1 exported OMP and MKL only.  This script sets the variable ONLY under
# P52_DETERMINISTIC=1 and actively UNSETS it otherwise, so a launcher whose shell happens to carry
# it cannot silently move E1 into a different environment (BRIEF_27 G2).  F6(a) still holds for the
# deterministic regime: `--deterministic` REFUSES rather than proceeding when the variable is
# absent, because a flag that silently fails to take effect produces a run that BELIEVES it is
# reproducible and is not.
#
# BRIEF_17 section 12: the implementer writes this script, the USER launches it, and the
# implementer never sleep-polls it.  ⚠️ NO `until`-POLL ANYWHERE, deliberately: `pgrep -f` matches
# the polling shell's OWN command line, so `! pgrep ...` is always false and the loop hangs forever
# if the job dies.  That hang is live in this repo's history twice.
#
# 🚨 output/p5_1/ IS READ-ONLY TO THIS CAMPAIGN.  It holds the only copy of the evidence behind a
# merged, independently reviewed result; output/ is gitignored and there is no backup.  Every write
# and delete in offline/tier_sweep.py resolves its path and REFUSES anything at or under it, and
# the precondition block below fails fast if the work directory is pointed inside it -- at launch,
# where it costs a second, rather than ten hours in.
#
# WHAT IT DOES, IN ORDER
# -----------------------------------------------------------------------------------
#   GATE 0   preconditions: CUDA, the read-only path check, the regime check, the held-out draws,
#            the per-tier declaration and the CAMPAIGN declaration (which enumerates every cell)
#   GATE 1   re-verify P5.1's seven cells AT CONSUMPTION (B3a) and re-roll the `random` anchor,
#            requiring EXACT equality with P5.1's 500 episodes.  MISMATCH => REFUSE AND STOP
#   PHASE E1 replicate dt_spatial + dt_nomix at seed 202; the envelope (F7).  DONE 2026-08-19,
#            returned +0.0000 on all three blocks; skips on its existing artifacts
#   PHASE A  the head-count 2x2 at mappo1000, then the STOP RULE -- which HALTS the campaign if
#            CI(d4) lies entirely below zero, writing STOPPED_BY_RULE
#   PHASE C  bc_top10_perix at mappo1000 -- the only NEW arm at the reused tier (A3)
#   PHASE B  the ladder: maxpressure, fixedtime, random; 6 method arms + the behaviour anchor
#   VERIFY   completed cells must equal the cells the CAMPAIGN DECLARATION names
#
# ⚠️ STILL DEFERRED AND DELIBERATELY NOT HERE: the I1/J1 random-tier replicate.  It compares against
# phase B's own random-tier cell, and inventing that wiring while the ladder runs is how a replicate
# ends up re-evaluating the cell it is supposed to be independent of -- the zero-by-construction J2
# exists to refuse.  Its two cells ARE enumerated in the campaign declaration (K2), so the
# completeness assertion refuses to write CAMPAIGN_COMPLETE until they exist.  That is a mechanism,
# not a promise to remember.
#
# EXPECTED WALL CLOCK: about 50 h remaining.  Measured from P5.1's own campaign log: 59 min per DT
# arm-seed at 40,000 steps, 42 min for a tier's baselines, ~28 min per evaluation cell of 5 seeds x
# 100 draws.  ⚠️ Only those rates are measurements; the total is an estimate.
#
# RESUME IS SAFE BECAUSE THE WRITER IS ATOMIC, NOT BECAUSE THE READER IS CLEVER.  The conditions
# below are bare existence tests -- `[ -f cell.json ]` and a checkpoint count of 5 -- which would be
# D1(c)'s exact failure if a crash could leave a truncated artifact at a final name.  It cannot:
# training writes to `<name>.partial` and `replace_guarded` does an `os.replace` behind the barrier,
# and evaluation writes through `write_json_atomic`.  A kill therefore leaves a `.partial` or the
# previous file, never a half-written one at the name the resume tests, so an existence test is
# sufficient BY CONSTRUCTION -- and D1(c) is satisfied at the WRITER, which is the right place.
#
set -euo pipefail

ROOT=/home/filip/rltraffic-p52
PY=/home/filip/rltraffic/.venv/bin/python
CORPUS=/home/filip/rltraffic/datasets_v11
REUSE=/home/filip/rltraffic/output/p5_1
CHECKSUMS=/home/filip/rltraffic/output/SHA256SUMS_p5_1.txt

# REHEARSAL KNOBS.  Both default to the declared campaign values and exist so this script's own
# control flow can be EXECUTED before an overnight handover rather than read (PROJECT_PLAN
# section 7, 2026-08-14: a declared path with no execution behind it is untested however carefully
# it was written).
STEPS=${P52_STEPS:-40000}
WORK=${P52_WORK:-$ROOT/output/p5_2}
OUT=${P52_OUT:-$ROOT/docs/data}
LOGS=$WORK/logs
E1_SEED=202
TIER=mappo1000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 🚨 CUBLAS_WORKSPACE_CONFIG IS SET ONLY UNDER --deterministic (BRIEF_27 G2).  An earlier version of
# this script exported it unconditionally, commented "harmless in the default regime".  It is not
# neutral -- it is part of the determinism recipe, constraining cuBLAS workspace and thereby GEMM
# kernel selection between runs.  E1 exists to measure the run-to-run envelope OF THE REGIME P5.1
# RAN IN, and P5.1 exported OMP and MKL only (verified: offline/campaigns/p5_1.sh lines 90-91, and
# CUBLAS appears nowhere else on main).  A replicate carrying a variable P5.1 lacked would conflate
# the noise being measured with a systematic effect of the cuBLAS configuration.
#
# The default branch UNSETS it rather than merely not setting it, because the launcher's own shell
# may carry it: E1 must reproduce P5.1's environment exactly, and inheriting is how it would fail
# to, silently.
DETERMINISTIC=${P52_DETERMINISTIC:-0}
if [ "$DETERMINISTIC" = "1" ]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
else
  unset CUBLAS_WORKSPACE_CONFIG
fi

COMMON=(--corpus-root "$CORPUS"
        --draws-root "$ROOT/scenarios/draws"
        --reuse-root "$REUSE"
        --checksums "$CHECKSUMS"
        --out-dir "$OUT"
        --work-dir "$WORK"
        --checkpoint-dir "$WORK/checkpoints"
        --gradient-steps "$STEPS"
        --torch-threads 1)

mkdir -p "$LOGS" "$WORK/checkpoints"
cd "$ROOT"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
say()   { echo "[$(stamp)] $*"; }

if [ "$STEPS" -ne 40000 ]; then
  say "################################################################"
  say "REHEARSAL: STEPS=$STEPS, not the declared 40000."
  say "This exercises the control flow ONLY.  E1's numbers are NOT a"
  say "measurement of the envelope at any budget but the declared one."
  say "################################################################"
fi
say "P5.2 campaign starting; pin OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
say "work dir: $WORK   out dir: $OUT   tier: $TIER   E1 seed: $E1_SEED"
if [ "$DETERMINISTIC" = "1" ]; then
  say "regime: DETERMINISTIC (CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG)"
  COMMON+=(--deterministic)
else
  say "regime: DEFAULT CUDA, CUBLAS_WORKSPACE_CONFIG unset -- P5.1's exact environment (G2)"
fi

# -------------------------------------------------------------------------------------
# GATE 0 -- preconditions.  Each is a refusal, not a warning, and each costs seconds.
# -------------------------------------------------------------------------------------
$PY - <<'PYEOF'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(
        "FATAL: no CUDA device. The spatial model measures 5440 ms/step on CPU -- 60 h per seed. "
        "Refusing to start a job that cannot finish rather than running it silently."
    )
print(f"CUDA: {torch.cuda.get_device_name(0)}")
PYEOF
say "CUDA present"

# D1(a) at launch: the work directory must not resolve inside the read-only tree.  The module
# refuses every such write anyway; this fails in one second instead of at the first write.
$PY - "$WORK" "$OUT" "$REUSE" <<'PYEOF'
import sys
from pathlib import Path

from offline.tier_sweep import assert_writable, protected_roots_from

work, out, reuse = (Path(p) for p in sys.argv[1:4])
protected = protected_roots_from([reuse])
for label, path in (("--work-dir", work), ("--out-dir", out)):
    try:
        assert_writable(path, protected)
    except PermissionError as exc:
        sys.exit(f"FATAL: {label} resolves inside the read-only tree.\n{exc}")
print(f"read-only guard: {reuse} is protected; work and out directories are outside it")
PYEOF
say "read-only path check passed"

# G2 at launch: the regime E1 runs in must be P5.1's, and it is asserted in the CHILD process --
# where it matters -- rather than in this shell.
$PY - "$DETERMINISTIC" <<'PYEOF'
import os
import sys

import torch

wanted_deterministic = sys.argv[1] == "1"
configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if wanted_deterministic:
    if configured != ":4096:8":
        sys.exit(f"FATAL: deterministic regime requested but CUBLAS_WORKSPACE_CONFIG={configured!r}")
else:
    if configured is not None:
        sys.exit(
            f"FATAL: CUBLAS_WORKSPACE_CONFIG={configured!r} leaked into the DEFAULT regime. E1 must "
            "reproduce P5.1's environment exactly, and P5.1 exported OMP and MKL only. Unset it."
        )
    if torch.are_deterministic_algorithms_enabled():
        sys.exit("FATAL: deterministic algorithms are on in the default regime")
print(
    f"regime check: CUBLAS_WORKSPACE_CONFIG={configured!r}, "
    f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}"
)
PYEOF
say "regime check passed (E1 runs in P5.1's environment: OMP and MKL only)"

DRAWS=$(find "$ROOT/scenarios/draws/cityflow_grid4x4" -maxdepth 1 -type d -name 'draw_10??' 2>/dev/null | wc -l)
if [ "$DRAWS" -ne 100 ]; then
  say "FATAL: $DRAWS materialised held-out draws, expected 100."
  say "Re-materialise them and byte-verify against the five survivors in the main tree:"
  say "  $PY -m offline.materialise_draws --env-config configs/sim/cityflow_grid4x4.json ..."
  exit 2
fi
say "held-out demand present: 100 draws"

[ -f "$OUT/p5_2_declaration_${TIER}.json" ] || {
  say "declaration missing; writing it now (it is a precondition, not an output)"
  $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" declare
}
say "declaration present"

# K2: the campaign-wide declaration enumerates EVERY cell the campaign owes -- including the two
# I1/J1 replicate cells, whose wiring is deliberately deferred.  The completeness assertion at the
# end derives its expectation from THIS FILE, never from the files being checked, so a campaign
# that lacks them refuses to report itself complete rather than depending on anyone remembering.
[ -f "$OUT/p5_2_declaration.json" ] || \
  $PY -m offline.tier_sweep "${COMMON[@]}" declare-campaign
say "campaign declaration present (expected cells enumerated, replicates included)"

# -------------------------------------------------------------------------------------
# GATE 1 -- the reuse gate (B3).  Digests AT CONSUMPTION, then the random-anchor re-roll.
# -------------------------------------------------------------------------------------
say "GATE 1: verifying P5.1's seven reused cells at consumption"
$PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" verify-reuse

if [ ! -f "$WORK/eval_${TIER}_random.json" ]; then
  say "GATE 1: re-rolling the random anchor (CPU-deterministic policy; exact equality is the bar)"
  $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" evaluate --method random \
      > "$LOGS/eval_random.log" 2>&1
fi
$PY - "$WORK/eval_${TIER}_random.json" "$REUSE/eval_random.json" <<'PYEOF'
import json
import sys
from pathlib import Path

from offline.tier_sweep import assert_cells_identical

left = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
right = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
report = assert_cells_identical(left, right, expected_n=500)
print(
    f"random anchor reproduces P5.1 EXACTLY: {report['n_compared']} episodes, "
    f"{report['n_values_compared']} values, fields {report['fields']}"
)
PYEOF
say "GATE 1 PASSED: the draws, the env and the evaluation harness reproduce P5.1 exactly"

# -------------------------------------------------------------------------------------
# PHASE E1 -- the nondeterminism envelope.  Default CUDA, seed 202, both arms.
# -------------------------------------------------------------------------------------
for ARM in dt_spatial dt_nomix; do
  CKPT="$WORK/checkpoints/grid4x4_${TIER}_${ARM}_seed${E1_SEED}.pt"
  if [ -f "$CKPT" ]; then
    say "SKIP E1 training $ARM seed $E1_SEED: checkpoint already on disk"
  else
    say "E1 TRAIN $ARM seed $E1_SEED (~59 min measured for a DT arm-seed at 40,000 steps)"
    $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" --seeds "$E1_SEED" \
        train --method "$ARM" > "$LOGS/e1_train_$ARM.log" 2>&1
    say "E1 TRAIN $ARM COMPLETE"
  fi
done

for ARM in dt_spatial dt_nomix; do
  if [ -f "$WORK/eval_${TIER}_${ARM}_seed${E1_SEED}.json" ]; then
    say "SKIP E1 evaluate $ARM: cell already on disk"
  else
    say "E1 EVALUATE $ARM seed $E1_SEED over 100 held-out draws"
    $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" --seeds "$E1_SEED" \
        evaluate --method "$ARM" > "$LOGS/e1_eval_$ARM.log" 2>&1
    say "E1 EVALUATE $ARM COMPLETE"
  fi
done

say "E1 REPORT (F7: paired per-draw intervals against P5.1's own seed-202 cells)"
$PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" replicate-report --seed "$E1_SEED" \
    | tee "$LOGS/e1_report.log"

echo "E1 COMPLETE $(stamp)" > "$WORK/E1_COMPLETE"
say "E1 complete"

# -------------------------------------------------------------------------------------
# PHASE A -- the head-count 2x2, and the STOP RULE.  Reported first (A2).
# -------------------------------------------------------------------------------------
say "PHASE A: the 4-head pair at $TIER (the 1-head pair is P5.1's, reused)"
for ARM in dt_spatial_h4 dt_nomix_h4; do
  CKPT_COUNT=$(find "$WORK/checkpoints" -maxdepth 1 -name "grid4x4_${TIER}_${ARM}_seed*.pt" | wc -l)
  if [ "$CKPT_COUNT" -eq 5 ]; then
    say "SKIP training $ARM: 5 checkpoints already on disk"
  else
    say "TRAIN $ARM (5 seeds x $STEPS steps)"
    $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" train --method "$ARM" \
        > "$LOGS/train_${TIER}_$ARM.log" 2>&1
  fi
  if [ -f "$WORK/eval_${TIER}_${ARM}.json" ]; then
    say "SKIP evaluate $ARM: cell already on disk"
  else
    say "EVALUATE $ARM (5 seeds x 100 held-out draws)"
    $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" evaluate --method "$ARM" \
        > "$LOGS/eval_${TIER}_$ARM.log" 2>&1
  fi
done

# 🚨 THE STOP RULE IS ENFORCED HERE, BY THE SCRIPT, NOT REMEMBERED.  Exit 3 means the CI of d4 lies
# entirely below zero: spatial mixing HELPS at 4 heads, P5.1's sign has reversed, and the ladder
# would be measuring the wrong architecture.  `set -e` is deliberately suspended for this one call
# so the campaign can halt CLEANLY with its own message rather than dying on a non-zero exit.
say "SCORING THE STOP RULE (Q0)"
set +e
$PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" stop-rule | tee "$LOGS/stop_rule.log"
RULE_STATUS=${PIPESTATUS[0]}
set -e
if [ "$RULE_STATUS" -eq 3 ]; then
  say "################################################################"
  say "STOP RULE FIRED.  The ladder sweep is NOT run."
  say "$WORK/STOPPED_BY_RULE records the interval that fired it."
  say "This is a REGISTERED OUTCOME, not a failure: report it and stop."
  say "################################################################"
  exit 0
elif [ "$RULE_STATUS" -ne 0 ]; then
  say "FATAL: the stop rule could not be scored (exit $RULE_STATUS)"
  exit "$RULE_STATUS"
fi
say "stop rule did not fire; phase B proceeds"

# -------------------------------------------------------------------------------------
# PHASE C -- the new baseline arm at the reused tier.  Only NEW arms run at mappo1000 (A3).
# -------------------------------------------------------------------------------------
say "PHASE C: bc_top10_perix at $TIER (the only new baseline arm at the reused tier)"
$PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" train-baselines \
    > "$LOGS/train_${TIER}_baselines.log" 2>&1
[ -f "$WORK/eval_${TIER}_bc_top10_perix.json" ] || \
  $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$TIER" evaluate --method bc_top10_perix \
      > "$LOGS/eval_${TIER}_bc_top10_perix.log" 2>&1

# -------------------------------------------------------------------------------------
# PHASE B -- the ladder.  Four tiers in measured-ATT order; mappo1000 is reused, not re-run.
# -------------------------------------------------------------------------------------
for LADDER_TIER in maxpressure fixedtime random; do
  say "PHASE B: tier $LADDER_TIER"
  [ -f "$OUT/p5_2_declaration_${LADDER_TIER}.json" ] || \
    $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$LADDER_TIER" declare
  for ARM in dt_spatial dt_nomix; do
    TRAINED=$(find "$WORK/checkpoints" -maxdepth 1 -name "grid4x4_${LADDER_TIER}_${ARM}_seed*.pt" | wc -l)
    if [ "$TRAINED" -eq 5 ]; then
      say "SKIP training $LADDER_TIER/$ARM: 5 checkpoints on disk"
    else
      say "TRAIN $LADDER_TIER/$ARM (5 seeds)"
      $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$LADDER_TIER" train --method "$ARM" \
          > "$LOGS/train_${LADDER_TIER}_$ARM.log" 2>&1
    fi
  done
  say "TRAIN $LADDER_TIER baselines (bc, bc_top10, bc_top10_perix, iql)"
  $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$LADDER_TIER" train-baselines \
      > "$LOGS/train_${LADDER_TIER}_baselines.log" 2>&1
  for ARM in dt_spatial dt_nomix bc bc_top10 bc_top10_perix iql behaviour; do
    # The `random` tier's behaviour policy IS the shared random anchor (D12), verified common:
    # env settings identical across all four tier manifests, and the factory is a function of the
    # seed alone.  Rolling it twice would be two names for one measurement.
    if [ "$LADDER_TIER" = "random" ] && [ "$ARM" = "behaviour" ]; then
      say "SKIP $LADDER_TIER/behaviour: it IS the shared random anchor (D12)"
      continue
    fi
    if [ -f "$WORK/eval_${LADDER_TIER}_${ARM}.json" ]; then
      say "SKIP evaluate $LADDER_TIER/$ARM: cell on disk"
    else
      say "EVALUATE $LADDER_TIER/$ARM"
      $PY -m offline.tier_sweep "${COMMON[@]}" --tier "$LADDER_TIER" evaluate --method "$ARM" \
          > "$LOGS/eval_${LADDER_TIER}_$ARM.log" 2>&1
    fi
  done
done

# -------------------------------------------------------------------------------------
# Verify by EFFECT: completed cells must equal the cells the DECLARATION names.
# -------------------------------------------------------------------------------------
say "verifying the campaign against its declaration"
set +e
$PY -m offline.tier_sweep "${COMMON[@]}" assert-complete | tee "$LOGS/assert_complete.log"
COMPLETE_STATUS=${PIPESTATUS[0]}
set -e
if [ "$COMPLETE_STATUS" -ne 0 ]; then
  say "################################################################"
  say "CAMPAIGN INCOMPLETE -- see above.  This is EXPECTED at this point:"
  say "the two I1/J1 replicate cells are declared and not yet wired, so"
  say "the assertion refuses by design until they exist (BRIEF_27 K2)."
  say "PHASES_COMPLETE is still written; CAMPAIGN_COMPLETE is NOT."
  say "################################################################"
else
  echo "CAMPAIGN COMPLETE $(stamp)" > "$WORK/CAMPAIGN_COMPLETE"
  say "complete == declared"
fi

say "================================================================"
say "PHASES A, C and B ARE COMPLETE."
say "STILL OWED before the report, and NOT in this script: the random-tier"
say "envelope replicate (I1/J1) -- BOTH arms at seed 202 under a DISTINCT"
say "artifact key, with J2's canonical state_dict digest assertion."
say "================================================================"
echo "PHASES COMPLETE $(stamp)" > "$WORK/PHASES_COMPLETE"
