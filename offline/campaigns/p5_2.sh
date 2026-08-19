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
#   GATE 0   preconditions: CUDA, the read-only path check, the held-out draws, the declaration
#   GATE 1   re-verify P5.1's seven cells AT CONSUMPTION (B3a) and re-roll the `random` anchor,
#            requiring EXACT equality with P5.1's 500 episodes.  MISMATCH => REFUSE AND STOP
#   PHASE E1 replicate dt_spatial + dt_nomix at seed 202, 40,000 steps, DEFAULT CUDA; evaluate;
#            report the paired per-draw intervals against P5.1's own seed-202 cells (F7)
#            >>> THE CAMPAIGN STOPS HERE AND RETURNS FOR THE REGIME RULING (F1) <<<
#
# Phases A, B and C are NOT in this script yet, deliberately: the numerical regime they run under
# is chosen after E1 returns, and writing them now would bake in a choice the author has deferred.
#
# EXPECTED WALL CLOCK: about 3 h for E1.  Measured from P5.1's own campaign log: 59 min per DT
# arm-seed at 40,000 steps and ~28 min per evaluation cell of 5 seeds x 100 draws, so one seed's
# evaluation is ~6 min.  ⚠️ Only those two rates are measurements; the total is an estimate.
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
say "================================================================"
say "E1 IS COMPLETE AND THE CAMPAIGN STOPS HERE BY DESIGN (BRIEF_27 F1)."
say "The numerical regime for phases A/B/C is the author's ruling and it"
say "follows from the numbers above.  Do not start phase A from this script."
say "================================================================"
