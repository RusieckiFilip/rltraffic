#!/usr/bin/env bash
# P7.3c — C2: the grid4x4 SUMO CORPUS — A17(b)'s probe replayed through the trajectory logger, one MaxPressure
# episode per draw on the probe band 201–300, into ONE directory (BRIEF_41 C1–C2; PREREGISTRATION A24(b), A17(f)).
#
# Shape: offline/campaigns/p7_3d_grid4x4.sh, the corrected P7.3d driver — the run worktree, the derived tree, the
# two-step foreground start, -P on every interpreter call, the liveness guard, the group-leader and SigIgn checks,
# every refusal before the token, the trap before the token, record-canary after it, COMPLETE / FAILED after it.
#
# 0. USAGE — the FOREGROUND form (BRIEF_39 Amendment B.3-1), from the DETACHED RUN WORKTREE (J1(e)), which the
#    coordinator creates at the reviewed, pushed commit before the token:
#        git -C /home/filip/rltraffic worktree add --detach /home/filip/rltraffic-p73c-run <commit>
#
#      Step 1, open a pane:      tmux new -s p73c_corpus
#      Step 2, at ITS PROMPT:    bash /home/filip/rltraffic-p73c-run/offline/campaigns/p7_3c_corpus.sh 2>&1 | tee -a /home/filip/rltraffic/output/p7_3c_runs/corpus_capture.txt; echo "DRIVER EXIT: ${PIPESTATUS[0]}" | tee -a /home/filip/rltraffic/output/p7_3c_runs/corpus_capture.txt
#
#    The token is the author's (gate G2, channel (a)): /home/filip/rltraffic/output/p7_3c_runs/TOKEN_corpus.
#    Start on an idle machine, on mains power (PROJECT_PLAN §7's canary rule): the canary's timing half refuses
#    above 2.0 s.
#
#    THE SECOND HALF OF STEP 2 (Amendment B.5-3): the pipeline's own status is tee's, so without it the pane reports
#    0 even when the driver refuses; `${PIPESTATUS[0]}` is expanded after the pipeline finishes, so it is the
#    DRIVER's status, and the echo is teed too, so the capture's last line and the pane's agree.
#    ⛔ NOT `tmux new -s NAME '<cmd>'`: that runs a non-interactive shell with job control off, the script does not
#    lead its own process group, and the guard below refuses (measured on P7.3d, 2026-09-21).
#
# 1. WHAT IT PRODUCES
#      /home/filip/rltraffic/datasets_sumo_v11/grid4x4_sumo_maxpressure/   THE CORPUS: 100 .npz, manifest.json,
#                                                                         SHA256SUMS (101 lines, sha256sum's format)
#      /home/filip/rltraffic/output/p7_3c_corpus/canary.json              the canary line, right after the token
#      /home/filip/rltraffic/output/p7_3c_corpus/a17f_gate.json           A17(f)'s record — written ONLY if the gate passes
#      /home/filip/rltraffic/output/p7_3c_corpus/COMPLETE  or  FAILED     the terminal marker, on every path after the token
#    It writes nothing under docs/, scenarios/ or any worktree.  collect prints one line per episode with its GLOBAL
#    return: MaxPressure on the probe band, whose per-intersection returns are already published in
#    docs/data/p7_3d_calibration.json.  No outcome of any evaluated arm exists here, and none is printed.
#
# 2. THE BARRIER, AND WHY THIS RUN IS NOT RESUMABLE.  Every refusal precedes the token and writes NOTHING.  The
#    corpus directory and the run directory must each be ABSENT OR EMPTY at the start; this driver never deletes,
#    moves or overwrites either, and `--overwrite` is never passed.  offline.collect is one process with one
#    manifest (Amendment A, Q11), so an interrupted run leaves a partial corpus and FAILED behind; starting again
#    is REFUSED until a person moves both directories aside (to output/p7_3c_runs/, never deleting them).  An
#    episode with a teleport or a collision — counted on EVERY simulated second (DEFERRED 93) — is refused before
#    it is written and stops the run: a finding for the coordinator, not a re-roll.
#
# 3. ORDERING — interpreter → the tree this copy lives in → not the implementer's tree → import (-P) from that
#    tree → no live runner → GROUP LEADER → SigIgn → RESCO's network → the 100 parity configs → dirty tree →
#    THE BARRIER → free memory → corpus-preflight (the two committed artifacts at their pins, grid4x4's alignment
#    and RESCO's digests, the band disjoint from the subject's training draws and the held-out pool) → canary,
#    BOTH halves → TRAP → token → record-canary → collect-corpus → SHA256SUMS (atomic, re-verified, 101 lines) →
#    corpus-gate, LAST → COMPLETE.  The trap is installed BEFORE the token is consumed (BRIEF_37 J2/J3).
#
# 4. -P ON EVERY INTERPRETER CALL (BRIEF_39 B.3-2): without it Python prepends the cwd — the MAIN tree — to sys.path
#    and `offline` resolves to the main tree's package.  PYTHONPATH names the tree this copy lives in, and the
#    import check asserts WHICH files loaded.
#
# 5. THE CWD RULE: cd to the MAIN tree; the parity configs are opened read-only, and offline.collect resolves the
#    CityFlow flow file (for flow_draw_sha256 only) against it.  The corpus argv is built by the module from
#    offline.transfer_gate.COLLECT_SETTINGS — the probe's own settings — never retyped here.
#
# 6. WORKERS = 1 — Amendment A (Q11): offline.collect writes one manifest from one process; a pool would need a
#    manifest merge nobody has reviewed.  G1's reviewer mandate carries this run's pre-flight checklist.
#
# 7. TIME AND MEMORY — MEASURED 2026-09-25 19:15 UTC at commit 97cfd98 (C1 merged with main), on this machine,
#    idle (load 0.57), through this driver's own module call (collect-corpus), draws 201 and 201-202:
#      canary before 1.08 s (the first, cold call), after 0.84 s; both correctness halves -32648.0 / 247.75089149261333
#      1 draw    wall 35.90 s   peak RSS 672 MiB
#      2 draws   wall 69.73 s   peak RSS 671 MiB     => 33.83 s per episode, about 2.1 s of process start
#    ⇒ 100 draws ≈ 2.1 + 100 × 33.83 s ≈ 3,385 s ≈ 56 min, one process. The plain-env probe ran 29.9 s per episode
#    (P7.3d); the observer's per-second reads cost the difference. The memory floor above (4,096 MiB) is about six
#    times the measured peak. Evidence: output/p7_3c_runs/c2/schedule_measurement_20260925T191457Z/ (the two fenced
#    scratch corpora, time -v, both canaries; the gate on the two-draw corpus: 32/32 MATCH, 0 events). The driver
#    prints its own wall clock and writes COMPLETE; that number, not this one, is what the packet reports.

set -euo pipefail

MAIN=/home/filip/rltraffic
# J1(e) / BRIEF_39 Amendment B.7.1-1: the tree this copy of the script lives in, never a hardcoded one.
WORK_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
IMPLEMENTER_TREE=/home/filip/rltraffic-p73c
RUN_TREE=/home/filip/rltraffic-p73c-run
PY=$MAIN/.venv/bin/python
SCENARIO=cityflow_grid4x4
DRAWS_RANGE=(201 301)
N_DRAWS=100
CORPUS=$MAIN/datasets_sumo_v11/grid4x4_sumo_maxpressure
RUN_DIR=$MAIN/output/p7_3c_corpus
DRAWS=$MAIN/scenarios/draws
DATA=$WORK_TREE/docs/data
TOKEN=$MAIN/output/p7_3c_runs/TOKEN_corpus
CANARY_MAX_SECONDS=2.0
WORKERS=1
EXPECTED_SUMS_LINES=101
MIN_AVAILABLE_MIB=4096

: "${RLTRAFFIC_GRID4X4_RESCO:=$MAIN/scenarios/grid4x4_candidates}"
export RLTRAFFIC_GRID4X4_RESCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Options of the module's PARENT parser: every call puts them BEFORE the subcommand (P7.3d's B.6 finding B1).
COMMON=(--draws-root "$DRAWS" --output-root "$MAIN/output" --work-dir "$RUN_DIR")

echo "=== P7.3c corpus driver: WORK_TREE $WORK_TREE (derived from this script's own location)"

# ---------------------------------------------------------------- preconditions
if [ ! -x "$PY" ]; then
  echo "REFUSING TO START: no interpreter at $PY" >&2
  echo "  The worktree has no .venv of its own; the main tree's is the one to use." >&2
  exit 2
fi

if [ "$WORK_TREE" = "$IMPLEMENTER_TREE" ]; then
  echo "REFUSING TO START: this copy of the driver is in the implementer's worktree $WORK_TREE" >&2
  echo "  J1(e): the corpus is collected from the DETACHED run worktree $RUN_TREE, created at the" >&2
  echo "  reviewed commit and edited by no session. Nothing has been consumed." >&2
  exit 2
fi

cd "$MAIN"
LOADED=$(PYTHONPATH=$WORK_TREE "$PY" -P -c 'import offline.transfer_calibration as t, offline.collect as c; print(t.__file__); print(c.__file__)' 2>/dev/null || true)
N_LOADED=0
while IFS= read -r MODULE_FILE; do
  case "$MODULE_FILE" in
    "$WORK_TREE"/*) N_LOADED=$((N_LOADED + 1)) ;;
    *)
      echo "REFUSING TO START: a corpus module loaded from '${MODULE_FILE:-nothing}', not $WORK_TREE" >&2
      echo "  The cwd is the MAIN tree (section 5); the CODE must still be this tree's. Nothing consumed." >&2
      exit 2
      ;;
  esac
done <<< "$LOADED"
if [ "$N_LOADED" -ne 2 ]; then
  echo "REFUSING TO START: expected offline.transfer_calibration and offline.collect from $WORK_TREE, got:" >&2
  echo "${LOADED:-nothing}" | sed 's/^/    /' >&2
  exit 2
fi

ALIVE=$(pgrep -f 'python.*offline\.(collect|transfer_calibration|transfer_curve|few_shot|g2_measure)' 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true)
if [ -n "$ALIVE" ]; then
  echo "REFUSING TO START: another run is still alive:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo "$ALIVE" | tr '\n' ' ') >&2 2>/dev/null || echo "$ALIVE" >&2
  echo "  Wait until it has ended. Nothing has been consumed." >&2
  exit 3
fi

if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then
  echo "REFUSING TO START: not a process-group leader; run in a tmux FOREGROUND pane" >&2
  echo "  Open a pane with 'tmux new -s p73c_corpus', then type the command at its prompt (section 0)." >&2
  echo "  The handler kills the process group, which is a no-op from a non-leader. Nothing consumed." >&2
  exit 2
fi

SIGIGN_MASK=$(awk '/^SigIgn:/ { print $2 }' /proc/$$/status)
if [ -n "$SIGIGN_MASK" ] && [ $(( 0x$SIGIGN_MASK & 0x2 )) -ne 0 ]; then
  echo "REFUSING TO START: SIGINT is IGNORED in this shell (SigIgn $SIGIGN_MASK)" >&2
  echo "  bash cannot trap a signal ignored on entry, so Ctrl-C could not stop the run." >&2
  exit 2
fi

if [ ! -f "$RLTRAFFIC_GRID4X4_RESCO/resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml" ]; then
  echo "REFUSING TO START: RLTRAFFIC_GRID4X4_RESCO=$RLTRAFFIC_GRID4X4_RESCO holds no grid4x4 network" >&2
  exit 2
fi

MISSING=0
for DRAW in $(seq "${DRAWS_RANGE[0]}" $(( DRAWS_RANGE[1] - 1 ))); do
  CONFIG=$(printf '%s/%s/draw_%04d/parity/noteleport.sumocfg' "$DRAWS" "$SCENARIO" "$DRAW")
  if [ ! -f "$CONFIG" ]; then
    echo "REFUSING TO START: missing parity configuration $CONFIG" >&2
    MISSING=$((MISSING + 1))
    [ "$MISSING" -ge 3 ] && break
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "  P7.3d C1 rendered the probe band 201-300 into the MAIN tree's scenarios/draws. Nothing consumed." >&2
  exit 2
fi

WORK_TREE_DIRTY=$(git -C "$WORK_TREE" status --porcelain)
if [ -n "$WORK_TREE_DIRTY" ]; then
  echo "REFUSING TO START: the worktree $WORK_TREE is DIRTY" >&2
  echo "$WORK_TREE_DIRTY" | sed 's/^/    /' >&2
  echo "  J1(d): the manifest records the commit the corpus was collected at; a dirty tree has none." >&2
  exit 2
fi

# ---------------------------------------------------------------- THE BARRIER
# Nothing exists yet that this run could damage, and nothing it finds will be touched: every write happens after
# the token, into directories that were absent or empty when every check passed.
for TARGET in "$CORPUS" "$RUN_DIR"; do
  if [ -e "$TARGET" ] && [ ! -d "$TARGET" ]; then
    echo "REFUSING TO START: $TARGET exists and is not a directory. Nothing has been consumed." >&2
    exit 2
  fi
  if [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET")" ]; then
    echo "REFUSING TO START: $TARGET is not empty" >&2
    ls -A "$TARGET" | head -5 | sed 's/^/    /' >&2
    echo "  The filesystem-mutation barrier: this driver never deletes, moves or overwrites a corpus or its" >&2
    echo "  run record. A person moves both aside (to output/p7_3c_runs/) first. Nothing has been consumed." >&2
    exit 2
  fi
done

AVAILABLE_MIB=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
if [ -n "$AVAILABLE_MIB" ] && [ "$AVAILABLE_MIB" -lt "$MIN_AVAILABLE_MIB" ]; then
  echo "REFUSING TO START: ${AVAILABLE_MIB} MiB available, below the ${MIN_AVAILABLE_MIB} MiB floor" >&2
  echo "  One collection process with SUMO measured below it (section 7). Nothing has been consumed." >&2
  exit 2
fi

# The two committed artifacts at their pins, grid4x4's alignment (RESCO's digests), and the band disjoint from the
# draws the subject's five checkpoints trained on and from the held-out pool (BRIEF_41 C1(v), Amendment A Q21).
if ! PREFLIGHT_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" corpus-preflight --draws-range "${DRAWS_RANGE[@]}" --data-dir "$DATA"); then
  echo "${PREFLIGHT_LINE:-corpus-preflight printed no result line}"
  echo "REFUSING TO START: corpus-preflight did not pass; the line above names the check. Nothing consumed." >&2
  exit 2
fi
echo "$PREFLIGHT_LINE"

# ---------------------------------------------------------------- the canary, BOTH halves
CANARY_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" canary) || {
  echo "${CANARY_LINE:-the canary printed no line}"
  echo "REFUSING TO START: the canary failed its correctness half. NOTHING has been consumed." >&2
  echo "  A correctness failure means the ENGINE did not reproduce draw 0: a finding, not a rate question." >&2
  exit 2
}
echo "$CANARY_LINE"
CANARY=$(echo "$CANARY_LINE" | awk '{print $2}')
if awk -v c="$CANARY" -v m="$CANARY_MAX_SECONDS" 'BEGIN { exit !(c > m) }'; then
  echo "REFUSING TO START: canary $CANARY s exceeds $CANARY_MAX_SECONDS s -- the machine is throttled." >&2
  echo "  Check mains power and the cooling pad, then start again. Nothing has been consumed." >&2
  exit 2
fi

# ---------------------------------------------------------------- the trap, BEFORE the token
fail() {
  local where=$1
  if [ -d "$RUN_DIR" ]; then
    echo "CORPUS RUN FAILED at $where" | tee "$RUN_DIR/FAILED"
  else
    echo "CORPUS RUN FAILED at $where (before the run directory existed)"
  fi
  exit 1
}

on_signal() {
  trap '' INT TERM HUP
  if [ -d "$RUN_DIR" ]; then
    echo "CORPUS RUN INTERRUPTED by a signal" | tee "$RUN_DIR/FAILED" >&2
  else
    echo "CORPUS RUN INTERRUPTED by a signal (before the run directory existed)" >&2
  fi
  echo "  The partial corpus stays where it is; a new start is refused until both directories are moved" >&2
  echo "  aside (section 2). Killing the process group." >&2
  kill -- -$$ 2>/dev/null || true
  exit 130
}
trap on_signal INT TERM HUP

# ---------------------------------------------------------------- the token
if [ ! -f "$TOKEN" ]; then
  echo "REFUSING TO START: no run token at $TOKEN" >&2
  echo "  The author writes it (gate G2, channel (a)). Nothing has been consumed." >&2
  exit 2
fi
echo "=== authorised by the token written $(stat -c '%y' "$TOKEN" | cut -d. -f1): $(cat "$TOKEN")"
rm -f "$TOKEN"
echo "=== token consumed and deleted; another start needs a new one"

# Only NOW may anything be created.
mkdir -p "$RUN_DIR"

echo "P7.3c C2 — the grid4x4 SUMO corpus (A24(b): the probe band, one episode per draw, logged)"
echo "  commit       $(git -C "$WORK_TREE" rev-parse HEAD)"
echo "  code         $(echo "$LOADED" | tr '\n' ' ')"
echo "  cwd          $(pwd)   (the MAIN tree, section 5)"
echo "  draws        ${DRAWS_RANGE[0]}-$(( DRAWS_RANGE[1] - 1 )) ($N_DRAWS), reset(seed=1000) on a fresh env per draw (A18(c))"
echo "  corpus       $CORPUS"
echo "  workers      $WORKERS   (Amendment A, Q11: one process, one manifest)"
echo "  memory       ${AVAILABLE_MIB} MiB available, floor ${MIN_AVAILABLE_MIB} MiB"
echo "  canary       $CANARY_LINE"
echo "  preflight    $PREFLIGHT_LINE"
echo "  started      $(date -Is)"
echo ""

START=$(date +%s)

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" \
  record-canary --line "$CANARY_LINE" || fail "record-canary"

PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" collect-corpus --corpus-dir "$CORPUS" --draws-range "${DRAWS_RANGE[@]}" || fail "collect-corpus"

# SHA256SUMS over the 100 episodes and the manifest, sha256sum's own format, written atomically (tmp -> mv).
( cd "$CORPUS" \
  && find . -maxdepth 1 -type f \( -name '*.npz' -o -name 'manifest.json' \) -printf '%f\n' \
     | LC_ALL=C sort | xargs sha256sum > SHA256SUMS.tmp \
  && mv SHA256SUMS.tmp SHA256SUMS ) || fail "SHA256SUMS write"
( cd "$CORPUS" && sha256sum -c --quiet SHA256SUMS ) || fail "SHA256SUMS verify"
N_SUMS=$(wc -l < "$CORPUS/SHA256SUMS")
if [ "$N_SUMS" -ne "$EXPECTED_SUMS_LINES" ]; then
  echo "SHA256SUMS lists $N_SUMS file(s), not $EXPECTED_SUMS_LINES ($N_DRAWS episodes and the manifest)" >&2
  fail "SHA256SUMS count"
fi

# A17(f)'s gate, LAST: 100 x 16 returns == the probe's by id, zero events counted every simulated second, the
# engine seed every committed zero-shot cell records. Its record is written only if it passes.
GATE_LINE=$(PYTHONPATH=$WORK_TREE "$PY" -P -m offline.transfer_calibration "${COMMON[@]}" corpus-gate --corpus-dir "$CORPUS" --draws-range "${DRAWS_RANGE[@]}" --data-dir "$DATA" --record "$RUN_DIR/a17f_gate.json") || {
  echo "${GATE_LINE:-corpus-gate printed no result line}"
  fail "corpus-gate"
}
echo "$GATE_LINE"

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "CORPUS RUN COMPLETE in ${ELAPSED}s  $(date -Is)" | tee "$RUN_DIR/COMPLETE"
echo "NEXT: the coordinator verifies the corpus from disk (gate G3): 100/100, A17(f) 1,600/1,600, zero events."
