#!/usr/bin/env bash
#
# P8.4a CAMPAIGN -- how many vehicles each arm admits, and both ATT definitions on every episode.
# =====================================================================================
#   tmux new -s p84a
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#   mkdir -p /home/filip/rltraffic/output/p8_4a/logs        # <-- REQUIRED BEFORE THE PIPE
#   bash /home/filip/rltraffic-p84a/offline/campaigns/p8_4a.sh 2>&1 | tee \
#        /home/filip/rltraffic/output/p8_4a/logs/campaign.log
#   # detach with C-b d ; reattach with: tmux attach -t p84a
#
# ⚠️ THE mkdir LINE IS NOT DECORATION.  The shell opens `tee`'s target BEFORE the script runs, so a
# log directory this script creates does not yet exist when the pipeline is built.  Measured on
# P4.7's real launch of 2026-08-15: `tee` failed with "No such file or directory", the campaign ran
# to completion regardless, and the log was never written.
#
# 🚨 IT RUNS FROM THE MAIN TREE, AND THAT IS A REQUIREMENT RATHER THAN A CONVENIENCE.
# ------------------------------------------------------------------------------------
# A materialised `cityflow.json` embeds `dir` as an ABSOLUTE path resolved against the process
# working directory, and `_existing_conflict` compares rendered files BYTE-FOR-BYTE before it looks
# at any provenance field.  Re-materialising from the task worktree therefore reports `cityflow.json
# differs byte-for-byte` on draws that are in fact correct -- MEASURED on 2026-08-28, and it would
# have been a FALSE `BLOCKED` under Amendment A1's stop rule.  So: working directory is the MAIN
# tree, and the code comes from the worktree via PYTHONPATH with PYTHONSAFEPATH=1, which is what
# stops the main tree's own `offline` package (which has no `admission_probe`) from shadowing it.
#
# ⚠️ NOT the provenance path fields.  `source_config`, `source_flow` and `source_roadnet` USED to be
# compared as identity and are not since Amendment E3 (`DEFERRED` 61); an earlier version of this
# header blamed them, which is the same misreading Amendment D1 made.  The remaining cause is the
# rendered config, and Amendment E4 rules that it is not normalised -- `restore-draws` detects a
# non-main working directory and refuses with a message that says so.
#
# 🚨 AND THE CWD HAS A PROVENANCE COST, recorded rather than hidden (review BL-2 / Amendment I2):
# `dt_gate.runtime_provenance` reads `git rev-parse HEAD` in the CWD, so the `runtime.git_commit` in
# anything this script writes is the MAIN TREE's HEAD -- another task's branch, containing none of
# this code.  `admission_probe.code_provenance()` records the real code root and commit beside it.
#
# WHAT IT DOES
# -----------------------------------------------------------------------------------
# Does:  replay 39 cells -- P4.6's five hz1x1 tiers x (bc, bc_top10, iql, dt) plus their five
#        behaviour anchors, and P5.2's grid4x4 `mappo1000` and `random` tiers x (dt_spatial,
#        dt_nomix, bc, bc_top10, bc_top10_perix, iql) plus their two anchors -- over the ten
#        held-out draws 1000-1009 at five seeds, recording `created`, `entered`, `never_entered`,
#        `att_ours` and `att_engine` per episode.
# Does NOT: train anything, collect anything, touch a committed artifact, or write anywhere under
#        `output/` except `output/p8_4a/`.  Every write goes through `tier_sweep.assert_writable`
#        with every sibling `output/*` directory passed as a protected root.
#
# GATE -1 IS NOT IN THIS SCRIPT.  It was run in-session and its result is in
# `output/p8_4a/draw_restoration.json`: all five surviving grid4x4 draws came back `kept`, i.e.
# byte-identical, and 1005-1009 were written.  This script REFUSES to start if that file is absent,
# because the alternative is measuring grid4x4 on demand that is not P5.2's, undetectably.
#
# RESUMABLE AT CELL GRANULARITY.  A cell whose `admission_<scenario>_<tier>_<method>.json` exists is
# skipped BY NAME.  Each cell is written through `write_json_atomic` (temp file + `os.replace`), so
# a kill leaves the previous file or nothing, never a partial cell.
#
# FAILS CLOSED: `set -euo pipefail`, plus a final assertion that every declared cell has a file and
# that every cell's replay reproduced its committed `att_horizon` EXACTLY.  A cell that does not
# reproduce exits non-zero: the replay would not be faithful and no admission number from it would
# be interpretable.
#
# EXPECTED WALL CLOCK.  Measured on this machine on 2026-08-28, four full cells of 50 episodes
# each, per-episode cost including the per-draw `torch.load` that `evaluate_arm`'s contract puts in
# front of every draw:
#   hz1x1   bc@random          1.333 s/episode
#   hz1x1   dt@random          1.746 s/episode
#   grid4x4 bc@random          2.165 s/episode
#   grid4x4 dt_spatial@random  3.440 s/episode
# Per-class projection over the 1170 hz1x1 and 700 grid4x4 episodes: about 28 + 30 = 58 min.
# Charging EVERY arm at its scenario's DT rate gives 74 min, which is a worst case and not the
# projection -- the BC rates supersede it for the 19 non-DT arms.  See docs/plans/p8.4a.md section 8.
# ⚠️ Serial on purpose.  Parallel cells would finish sooner and would also inflate every per-cell
# timing, and BRIEF_31's Definition of Done makes that timing a deliverable for P8.4b's cost model.
#
set -euo pipefail

WORKTREE="${WORKTREE:-/home/filip/rltraffic-p84a}"
MAIN="${MAIN:-/home/filip/rltraffic}"
PY="${PY:-$MAIN/.venv/bin/python}"
WORK="${WORK:-$MAIN/output/p8_4a}"
export WORKTREE WORK
ENGINE_SEED="${ENGINE_SEED:-1000}"

say() { printf '\n=== %s ===\n' "$*"; }
fail() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

cd "$MAIN" || fail "the main tree $MAIN does not exist"
export PYTHONSAFEPATH=1
export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

COMMON=(
  --repo-root "$WORKTREE"
  --corpus-root datasets_v11
  --draws-root scenarios/draws
  --output-root output
  --work-dir "$WORK"
  --engine-seed "$ENGINE_SEED"
)

# ---------------------------------------------------------------------------------------------
# GATE 0: preconditions.  Nothing is rolled until every one of them holds.
# ---------------------------------------------------------------------------------------------
say "GATE 0 preconditions"
[ -d "$MAIN/datasets_v11" ] || fail "no corpus at $MAIN/datasets_v11"
[ -f "$WORK/draw_restoration.json" ] || fail \
  "Gate -1 has not run: $WORK/draw_restoration.json is absent. Run restore-draws for both scenarios
   and confirm every survivor came back 'kept' before any cell is rolled."

$PY - <<'PYEOF' || fail "the Gate -1 report does not certify every draw as reproduced"
import json, pathlib, os, sys
work = pathlib.Path(os.environ.get("WORK", "/home/filip/rltraffic/output/p8_4a"))
report = json.loads((work / "draw_restoration.json").read_bytes())
missing = [s for s in ("hz1x1", "grid4x4") if s not in report]
if missing:
    sys.exit(f"draw_restoration.json covers {sorted(report)}, missing {missing}")
for scenario, record in sorted(report.items()):
    bad = sorted(d for d, a in record["actions"].items() if a not in ("kept", "written"))
    if bad or not record["survivors_reproduced"]:
        sys.exit(f"{scenario}: draws {bad} did not reproduce")
    print(f"  {scenario}: {len(record['actions'])} draws, survivors reproduced byte-identically")
PYEOF

for draw in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009; do
  for key in cityflow1x1 cityflow_grid4x4; do
    [ -f "$MAIN/scenarios/draws/$key/draw_$draw/cityflow.json" ] || \
      fail "draw $draw of $key is not materialised"
  done
done

# CAPABILITY, not completeness.  `assert_cell_complete`-style checks only ever fire AFTER the work
# that was supposed to produce a cell; this asserts up front that every declared cell CAN be
# produced.  The concrete precedent: tier_sweep's `_arm_factory` had no `fixedtime` branch, and the
# failure was deterministic, reachable only at evaluation time, and surfaced three days into a 53 h
# campaign (offline/tier_sweep.py::assert_factories_constructible).  Seconds, no environment, no
# rollout -- and it also verifies every checkpoint's existence, every MAPPO anchor's recorded
# sha256, the fixed-time plan hash, and that a complete committed reference exists for every cell.
say "GATE 0 capability check"
$PY - <<'PYEOF' || fail "at least one declared cell cannot be produced or has no committed reference"
import os, pathlib, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from offline.admission_probe import (
    BEHAVIOUR_METHOD, PROBE_DRAWS, PROBE_SCENARIOS, ProbeRoots, build_factory,
    committed_reference, seeds_for,
)
from offline.materialise_draws import draw_config_path

roots = ProbeRoots(
    repo_root=pathlib.Path(os.environ["WORKTREE"]), corpus_root=pathlib.Path("datasets_v11"),
    draws_root=pathlib.Path("scenarios/draws"), output_root=pathlib.Path("output"),
    work_dir=pathlib.Path(os.environ["WORK"]),
)
ok, problems = 0, []
for name, spec in sorted(PROBE_SCENARIOS.items()):
    for tier in spec.tiers:
        for method in (*spec.methods, BEHAVIOUR_METHOD):
            config = draw_config_path(spec.scenario_key, PROBE_DRAWS[0], out_root=roots.draws_root)
            try:
                for seed in seeds_for(name, tier, method):
                    build_factory(name, tier, method, seed, roots, device=None, config_path=config)
                reference, _ = committed_reference(name, tier, method, roots)
                wanted = {(s, d) for s in seeds_for(name, tier, method) for d in PROBE_DRAWS}
                absent = sorted(wanted - set(reference))
                if absent:
                    raise ValueError(f"{len(absent)} committed reference keys absent, e.g. {absent[:3]}")
                ok += 1
            except Exception as exc:
                problems.append(f"{name}/{method}@{tier}: {type(exc).__name__}: {exc}")
print(f"  cells constructible with a complete committed reference: {ok}")
for problem in problems:
    print(f"  PROBLEM: {problem}")
if problems:
    sys.exit(1)
PYEOF
say "GATE 0 PASSED: the corpus, the draws, the Gate -1 certificate and all 39 factories are present"

# ---------------------------------------------------------------------------------------------
# The 39 cells.  Order is cheapest scenario first, so a failure surfaces early.
# ---------------------------------------------------------------------------------------------
mkdir -p "$WORK/logs"

run_cell() {
  local scenario="$1" tier="$2" method="$3"
  local file="$WORK/admission_${scenario}_${tier}_${method}.json"
  if [ -f "$file" ]; then
    printf '  SKIP %s/%s@%s -- %s exists\n' "$scenario" "$method" "$tier" "$(basename "$file")"
    return 0
  fi
  say "PROBE $scenario/$method@$tier"
  $PY -m offline.admission_probe "${COMMON[@]}" \
      probe --scenario "$scenario" --tier "$tier" --method "$method" \
      2>&1 | tee "$WORK/logs/${scenario}_${tier}_${method}.log"
  local status="${PIPESTATUS[0]}"
  [ "$status" -eq 0 ] || fail "$scenario/$method@$tier exited $status (a non-zero exit here means
     the replay did NOT reproduce its committed att_horizon exactly)"
}

for tier in random fixedtime maxpressure mappo500 mappo1000; do
  for method in bc bc_top10 iql dt behaviour; do
    run_cell hz1x1 "$tier" "$method"
  done
done

for tier in random mappo1000; do
  for method in bc bc_top10 bc_top10_perix iql dt_nomix dt_spatial behaviour; do
    run_cell grid4x4 "$tier" "$method"
  done
done

# ---------------------------------------------------------------------------------------------
# Completeness, checked against the DECLARED inventory and never against the files on disk.
# ---------------------------------------------------------------------------------------------
say "COMPLETENESS"
$PY - <<'PYEOF' || fail "the campaign is incomplete or a cell did not reproduce"
import json, os, pathlib, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from offline.admission_probe import BEHAVIOUR_METHOD, PROBE_SCENARIOS, work_file_name

work = pathlib.Path(os.environ.get("WORK", "/home/filip/rltraffic/output/p8_4a"))
missing, inexact = [], []
declared = 0
for name, spec in sorted(PROBE_SCENARIOS.items()):
    for tier in spec.tiers:
        for method in (*spec.methods, BEHAVIOUR_METHOD):
            declared += 1
            path = work / work_file_name(name, tier, method)
            if not path.is_file():
                missing.append(path.name)
                continue
            payload = json.loads(path.read_bytes())
            if not payload["reference"]["exact"]:
                inexact.append(payload["arm"])
print(f"  declared cells: {declared}, missing: {len(missing)}, inexact: {len(inexact)}")
if missing:
    sys.exit(f"missing cells: {missing}")
if inexact:
    sys.exit(f"cells that did not reproduce their committed att_horizon: {inexact}")
PYEOF

say "REPORT"
$PY -m offline.admission_probe "${COMMON[@]}" report --out docs/data/p8_4a_admission.json \
    2>&1 | tee "$WORK/logs/report.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "the report exited non-zero"

say "CAMPAIGN COMPLETE"
touch "$WORK/CAMPAIGN_COMPLETE"

# ---------------------------------------------------------------------------------------------
# PHASE 2 -- the registered escalation, run only when explicitly asked for.
# ---------------------------------------------------------------------------------------------
# `docs/plans/p8.4a.md` section 4, approved as Amendment A3: **any arm with `deficit > 0` at all --
# close OR falsified -- is re-run over the full 100 held-out draws before anything is said about
# it.**  The permissive `Delta` that governs the VERDICT is only acceptable because this ESCALATION
# threshold sits at zero, and neither may be loosened without the other being re-argued.
#
# The cell list is DERIVED from the scored artifact by `escalation-plan`, never hand-written, and it
# carries the behaviour anchor of every escalated arm's tier: E1 is a comparison, and a 100-draw arm
# may only be compared against a 100-draw anchor.
#
# 🔒 THE TWO GRAINS ARE NEVER MIXED.  Escalated cells are written as `*_full.json` and reported into
# `docs/data/p8_4a_admission_escalated.json`; the 10-draw artifact is left exactly as it is.  A
# report whose cells span two draw counts REFUSES.
#
# MEASURED COST, from this campaign's own per-cell rates: 14 cells, 6200 episodes, **3.44 h serial**
# -- so Amendment A2's one-hour rule sends this phase to a user-started tmux, not to a session.
#
#   bash offline/campaigns/p8_4a.sh --escalate
#
if [ "${1:-}" = "--escalate" ]; then
  say "PHASE 2: the registered escalation to $(printf '%s' "100") held-out draws"
  [ -f "$WORK/CAMPAIGN_COMPLETE" ] || fail "phase 1 has not completed; there is nothing to escalate"

  PLAN="$($PY -m offline.admission_probe "${COMMON[@]}" escalation-plan \
          --artifact docs/data/p8_4a_admission.json | grep -v '^#')"
  [ -n "$PLAN" ] || say "nothing to escalate: no arm had a deficit above zero"

  while read -r scenario tier method; do
    [ -n "$scenario" ] || continue
    file="$WORK/admission_${scenario}_${tier}_${method}_full.json"
    if [ -f "$file" ]; then
      printf '  SKIP %s/%s@%s -- %s exists\n' "$scenario" "$method" "$tier" "$(basename "$file")"
      continue
    fi
    say "ESCALATE $scenario/$method@$tier over the full held-out pool"
    $PY -m offline.admission_probe "${COMMON[@]}" \
        probe --scenario "$scenario" --tier "$tier" --method "$method" --escalated \
        2>&1 | tee "$WORK/logs/${scenario}_${tier}_${method}_full.log"
    status="${PIPESTATUS[0]}"
    [ "$status" -eq 0 ] || fail "$scenario/$method@$tier exited $status at the escalated grain"
  done <<< "$PLAN"

  say "ESCALATED REPORT"
  $PY -m offline.admission_probe "${COMMON[@]}" report --escalated \
      --out docs/data/p8_4a_admission_escalated.json 2>&1 | tee "$WORK/logs/report_escalated.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "the escalated report exited non-zero"
  say "ESCALATION COMPLETE"
  touch "$WORK/ESCALATION_COMPLETE"
fi
