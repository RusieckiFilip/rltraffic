#!/usr/bin/env bash
# P7.2b — the target-domain SUMO probe and A17's return-prompt calibration.
#
# Shape: offline/campaigns/p7_1_metric_freeze.sh, whose own header credits p5_3b_decomp.sh.
#
# 1. WHAT IT PRODUCES
#    output/p7_2b/probe_draw_0201.json … 0300.json   one MaxPressure episode per draw, on SUMO
#    output/p7_2b/smoke_{mappo1000,mix50}.json       the FENCED mechanics proof on draw 5
#    docs/data/p7_2b_calibration.json                the committed artifact
#    output/SHA256SUMS_p7_2b.txt                     written last, atomically
#    It EVALUATES NOTHING: no held-out draw, no rho, no ATT or return of any DT (BRIEF_36 §2).
#
# 2. ORDERING — every check that can refuse PRECEDES the token, so a refused start consumes
#    nothing and changes nothing at all. Order: interpreter → import → lock → GROUP LEADER →
#    inputs → CANARY → TRAP → token → work. The canary is inside that fence deliberately: a
#    throttled machine must not burn the author's one-shot authorisation (BRIEF_35 Amendment D3.1
#    is the reason the canary exists).
#    ⚠️ THE TRAP IS INSTALLED BEFORE THE TOKEN IS CONSUMED (Amendment B1, pre-flight minor 2).
#    It used to be installed after, and a signal in that window destroyed the one-shot
#    authorisation while leaving neither FAILED nor COMPLETE -- the operator could not tell whether
#    the run had started. From the first destructive line onward, a signal now writes FAILED.
#    The handler writes it only if $WORK already exists, so a signal arriving BEFORE the token
#    check still creates nothing.
#
# 3. THE SKIP DECISION IS IN PYTHON, NOT IN THE SHELL.
#    offline/transfer_calibration.py::chunk_is_reusable re-derives a chunk's verdict from its own
#    numbers — the two return routes are compared again, the teleport count and the vehicle type
#    set are re-read from the chunk — because the stored verdict is exactly what a half-written
#    chunk would lie about. ⛔ THERE ARE DELIBERATELY NO `[ -f ]` GUARDS BELOW.
#
# 4. NOTHING UNDER scenarios/draws/ IS WRITTEN. The 102 parity configs it reads are P7.2a's
#    output and this campaign opens them read-only.
#
# 5. TIME, from measurements with their n and never from a guess:
#      canary            ≈ 0.9 s   (measured 2026-09-14; refuses above 2.0 s)
#      probe, 100 draws  ≈ 18 min  (10.85 s/episode measured on the fenced draw 5, 2026-09-14;
#                                   P7.1's registered bare MaxPressure rate is 11.84 s, n = 5)
#      smoke ×2          ≈ 30 s
#      report            seconds
#    Nominal ≈ 19 min; conservative upper bound ≈ 25 min.
#
# 6. THE WORKTREE HAS NO .venv. The interpreter is the main tree's, as p7_1's driver does.

set -euo pipefail

MAIN=/home/filip/rltraffic
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$MAIN/.venv/bin/python
WORK=$MAIN/output/p7_2b
LOGS=$WORK/logs
DRAWS=$MAIN/scenarios/draws
CANARY_MAX_SECONDS=2.0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

COMMON=(--draws-root "$DRAWS" --output-root "$MAIN/output" --work-dir "$WORK"
        --out-dir "$WORK_TREE/docs/data")

# ---------------------------------------------------------------- preconditions
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  echo "  The worktree has no .venv of its own; the main tree's is the one to use." >&2
  exit 2
fi

cd "$WORK_TREE"
if ! PYTHONPATH=$WORK_TREE "$PY" -P -c "import offline.transfer_calibration" 2>/dev/null; then
  echo "REFUSING TO START: offline.transfer_calibration does not import from $WORK_TREE" >&2
  exit 2
fi

# ⚠️ The pattern requires `python` before the module name, and this process and its parent are
# excluded: a bare match also matches the SHELL running this script, because the module name
# appears in its own command line, and the driver would refuse to start because of itself.
ALIVE=$(pgrep -f 'python.*offline\.transfer_calibration' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: cells from another run are still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo '  Wait until `pgrep -f transfer_calibration` is empty. Nothing has been consumed.' >&2
  exit 3
fi

# Amendment B3 (pre-flight minor 3): `kill -- -$$` in the signal handler is a NO-OP unless this
# script leads its own process group, which a tmux foreground pane gives and `bash script.sh &`
# does not. Fail closed, and turn BRIEF_34 E2's operator condition into something the driver checks.
if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux foreground pane" >&2
  echo "  The signal handler kills the process group, and that is a no-op from a background" >&2
  echo "  job, so an interrupted run would leave the python child writing. Nothing consumed." >&2
  exit 2
fi

for draw in 201 300 5; do
  cfg=$DRAWS/cityflow1x1/draw_$(printf '%04d' "$draw")/parity/noteleport.sumocfg
  if [ ! -f "$cfg" ]; then
    echo "REFUSING TO START: P7.2a's parity configuration is missing: $cfg" >&2
    exit 2
  fi
done
for ckpt in "$MAIN/output/p4_dt/dt_seed101.pt" "$MAIN/output/p4_7/checkpoints/mix50_dt_seed101.pt"; do
  if [ ! -f "$ckpt" ]; then
    echo "REFUSING TO START: a registered subject's checkpoint is missing: $ckpt" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------- the canary
# PROJECT_PLAN §7's rule; recipe in BRIEF_36 §3.3. A guest that reports a low load can still be
# running on a throttled host, so the rate basis is measured in THIS session or not written down.
echo "=== canary (PROJECT_PLAN section 7; recipe BRIEF_36 section 3.3)"
# `| tee /dev/stderr` so the OBSERVED values reach the pane even when the stage exits non-zero.
# Without it the command substitution swallows them and `set -euo pipefail` aborts here with
# nothing visible -- which is the same "the values exist on no disk and no screen" defect
# Amendment E1.2 was written about. Verified by experiment: with tee, a failing canary prints its
# line, does NOT reach the token, and exits 1; `pipefail` is what makes the pipeline's failure
# propagate.
CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" canary | tee /dev/stderr)
echo "$CANARY_LINE"
CANARY=$(echo "$CANARY_LINE" | awk '{print $2}')
if awk -v c="$CANARY" -v m="$CANARY_MAX_SECONDS" 'BEGIN { exit !(c > m) }'; then
  echo "REFUSING TO START: canary $CANARY s exceeds $CANARY_MAX_SECONDS s -- the machine is" >&2
  echo "  throttled and no rate measured here would be a rate. Check mains power and the" >&2
  echo "  performance plan, then start again. Nothing has been consumed." >&2
  exit 2
fi

# ---------------------------------------------------------------- the trap, BEFORE the token
# Amendment B1: installed here so that from the first destructive line onward -- and the token's
# deletion is the first one -- a signal writes FAILED. Both writers guard on $WORK existing, so a
# signal arriving before the token check still leaves the tree exactly as it found it.
fail() {
  local where=$1
  if [ -d "$WORK" ]; then
    echo "CAMPAIGN FAILED at $where" | tee "$WORK/FAILED"
  else
    echo "CAMPAIGN FAILED at $where (before any work directory existed)"
  fi
  exit 1
}

# The pid of the stage currently in flight. Global, so the signal handler can reach it: bash traps
# do not see a caller's locals.
CELL_PID=""

on_signal() {
  trap '' INT TERM
  if [ -d "$WORK" ]; then
    echo "CAMPAIGN INTERRUPTED by a signal" | tee "$WORK/FAILED" >&2
  else
    echo "CAMPAIGN INTERRUPTED by a signal (before any work directory existed)" >&2
  fi
  [ -z "$CELL_PID" ] || kill -TERM "$CELL_PID" 2>/dev/null || true
  sleep 2
  [ -z "$CELL_PID" ] || kill -KILL "$CELL_PID" 2>/dev/null || true
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM

# ---------------------------------------------------------------- the token
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

# Only NOW may anything be created or cleared.
mkdir -p "$LOGS"
rm -f "$WORK/FAILED" "$WORK/COMPLETE"

# Amendment E1.2: the canary's observed values reach a MANIFESTED file, not only the pane.
# Runs 1 and 2 printed them to stdout and nowhere else, so their correctness half exists on no
# disk. Appended, never truncated, like the stage logs -- and written here, AFTER the token, so
# Amendment B1's "a refused start creates nothing" still holds.
echo "=== canary  run at $(date -Is)" >> "$LOGS/canary.log"
echo "$CANARY_LINE" >> "$LOGS/canary.log"

START=$(date +%s)

run_stage() {
  local label=$1; shift
  local log=$LOGS/${label}.log
  echo "=== $label"
  # ⚠️ APPENDED, never truncated: a restart must not destroy the first run's narrative record.
  echo "=== $label  run at $(date -Is)" >> "$log"
  PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration \
    "${COMMON[@]}" --canary-seconds "$CANARY" "$@" >> "$log" 2>&1 &
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

# ---------------------------------------------------------------- stages
# The probe first: the smoke's prompt IS the Rule B mean k=100 target, which does not exist until
# the band has run (Amendment A4).
run_stage probe        probe --draws-range 201 301
run_stage smoke_mappo1000 smoke --subject mappo1000
run_stage smoke_mix50     smoke --subject mix50
run_stage report       report

# ---------------------------------------------------------------- manifest
( cd "$MAIN/output" \
  && find p7_2b -type f \( -name '*.json' -o -name '*.log' \) -print \
     | LC_ALL=C sort | xargs sha256sum > SHA256SUMS_p7_2b.txt.tmp \
  && mv SHA256SUMS_p7_2b.txt.tmp SHA256SUMS_p7_2b.txt ) || fail "manifest write"

( cd "$MAIN/output" && for manifest in SHA256SUMS_*.txt; do
    sha256sum -c "$manifest" >/dev/null || { echo "MANIFEST MISMATCH: $manifest" >&2; exit 1; }
  done ) || fail "manifest re-verification"

ELAPSED=$(( $(date +%s) - START ))
echo "CAMPAIGN COMPLETE in ${ELAPSED}s ($((ELAPSED / 60)) min)" | tee "$WORK/COMPLETE"

cat <<'NEXT'
=== NEXT (manual)
  1. git add docs/data/p7_2b_calibration.json && commit on the task branch
  2. read, in this order: targets[*] with role registered_prompt; statistics per k;
     disjointness.sources; smoke[*] (mechanics only -- the outcome is fenced)
  3. the artifact REPORTS; it registers nothing. A17 is the registration.
NEXT
