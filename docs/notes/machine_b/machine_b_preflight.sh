#!/usr/bin/env bash
# MACHINE B PRE-FLIGHT for P9.3's training (rltraffic; BRIEF_48 I.6). Written by the coordinator, 2026-10-28.
#
# Run on machine B, after the setup steps of docs/notes/MACHINE_B.md, as:
#   bash /home/filip/rltraffic/docs/notes/machine_b/machine_b_preflight.sh
# It runs, in order, and stops at the first failure:
#   1. identity: OS, glibc, CPU, GPU; the six Python 3.12.13 packages' installed files, the interpreter, `pip freeze`
#      (against requirements-frozen.txt, line for line) and every pinned package's installed library files (by RECORD)
#      against the laptop's; torch and CUDA; CityFlow; the run tree at 4dcbedc, clean, its two campaign modules' sha256;
#      the eight corpora's files against their manifest, and the manifest against its committed sha256
#   2. P9.3's seven test files at 4dcbedc, in a scratch clone (0 failed required; skips for absent draws are expected)
#   3. the CUDA smoke: every (tier, arm) cell, seed 101, 2 gradient steps -- 20 cells
#   4. one CPU cell (maxpressure / dt_nomix / seed 101, 2 steps) at the laptop's CPU kernel level: its canonical digest
#      MUST equal the laptop's (the author's gate); the same cell at B's own kernel level is reported beside it
#   5. the timing: each arm at 2,000 steps on maxpressure seed 101, CUDA, with a GPU trace every second, to size B's
#      sessions (no speed gate: the author's ruling of 2026-10-28)
# Steps 2-5 run at the laptop's CPU kernel level (AVX2 for MKL, torch, oneDNN and numpy), as B's sessions will: the
# laptop's CPU has no AVX-512, and there these settings change nothing (the CPU cell's digest is the same either way).
# It writes ONLY under $OUT (default /home/filip/machine_b_out) and refuses to start if $OUT exists. About 1 hour.
# QUICK=1 is the coordinator's mechanics test on the laptop: 2 smoke cells, 20 timing steps, and the laptop's one
# package beyond the frozen record (py-spy) tolerated; never on machine B.
set -u -o pipefail

OUT=${OUT:-/home/filip/machine_b_out}
QUICK=${QUICK:-0}
M=/home/filip/rltraffic
R=/home/filip/rltraffic-p93-run-c
N=$M/docs/notes/machine_b
DRIVE=${DRIVE:-/home/filip/machine_b_drive}
PY=$M/.venv/bin/python
SITE=$M/.venv/lib/python3.12/site-packages
RUN_COMMIT=4dcbedc25021634c199897beffeab4d806b9afc2
SHA_PB=5ee2edab67a6c53d65ac99e6861544e6122663409fc62501df4edbfcb2defe5f
SHA_LADDER=bd305b2c91644ad634b86ab8a68146dd908f0561596e7740ce951cc52e2b19b8
SHA_CORPORA_MANIFEST=7d6232c815e501656a09f492c190990672e887b74b877a40b7e5074e832e22a0
LAPTOP_CPU_CELL=4c03f73dbc700daf34978b31654b12b58dd3b02415e0a45d7b88f857282b6354
DEBS="libpython3.12 libpython3.12-dev libpython3.12-stdlib python3.12 python3.12-dev python3.12-venv"
TIERS="maxpressure fixedtime mappo1000 random_fresh"
ARMS="dt_nomix bc_m bc_top10_m bc_top10_perix_m iql_m"
TESTS="tests/test_hz4x4_phase_b_eval.py tests/test_hz4x4_report.py tests/test_hz4x4_phase_b.py tests/test_hz4x4_stall_and_records.py tests/test_hz4x4_ladder.py tests/test_p9_3_d5_definition_check.py tests/test_hz4x4_ruling_and_verify.py"

if [ -e "$OUT" ]; then echo "REFUSED: $OUT exists; it is never overwritten"; exit 2; fi
mkdir -p "$OUT/roots" || exit 2

fail() { echo "PREFLIGHT FAILED at step $1: $2"; echo "PREFLIGHT EXIT 1"; exit 1; }
step() { echo; echo "=== $(date '+%F %T %Z') step $1: $2"; }
laptop_kernels() {  # run "$@" with every CPU math library held to the laptop's kernel level (AVX2)
  (export MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 \
     NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"; "$@")
}
train_cell() {  # train_cell <output root> <tier> <arm> <device> <steps>: one cell through the frozen CLI, from the run tree
  (cd "$M" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$R" "$PY" -P -m offline.hz4x4_phase_b train \
    --tier "$2" --arm "$3" --seed 101 --device "$4" --gradient-steps "$5" --output-root "$1" --torch-threads 1)
}
train() { laptop_kernels train_cell "$@"; }

main() {
  echo "MACHINE B PRE-FLIGHT (QUICK=$QUICK) into $OUT; references from $N at $(git -C "$M" rev-parse --short HEAD)"

  step 1 "identity"
  echo "host: $(hostname)"; grep PRETTY_NAME /etc/os-release; ldd --version | head -1; echo "kernel: $(uname -r)"
  echo "cpus: $(nproc)"; lscpu | grep 'Model name'; free -g | sed -n 2p
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader || fail 1 "nvidia-smi"
  # The interpreter: the six packages' installed files, then the version string, of /usr/bin and of the venv.
  python3 -I "$N/machine_b_digests.py" deb $DEBS > "$OUT/python_debs.txt" || fail 1 "the six Python packages"
  diff "$N/laptop_python_debs.txt" "$OUT/python_debs.txt" || fail 1 "the Python 3.12.13 files differ from the laptop's"
  echo "python packages: all six equal to the laptop's (installed files, by sha256)"
  local version venv_version
  version=$(/usr/bin/python3.12 -c 'import sys; print(sys.version)') || fail 1 "/usr/bin/python3.12 does not start"
  venv_version=$("$PY" -c 'import sys; print(sys.version)') || fail 1 "the venv's interpreter does not start"
  echo "python: $version"
  [ "$version" = "$(cat "$N/laptop_python.txt")" ] && [ "$venv_version" = "$version" ] || fail 1 "not the laptop's interpreter"
  # The venv: pip freeze against the frozen record, line for line (its editable self-entry aside).
  "$PY" -m pip freeze --exclude-editable 2>/dev/null | LC_ALL=C sort > "$OUT/pip_freeze.txt" || fail 1 "pip freeze"
  grep -v -E '^#|^$|^-e ' "$M/requirements-frozen.txt" | LC_ALL=C sort > "$OUT/frozen_record.txt"
  local extra missing
  extra=$(LC_ALL=C comm -13 "$OUT/frozen_record.txt" "$OUT/pip_freeze.txt")
  missing=$(LC_ALL=C comm -23 "$OUT/frozen_record.txt" "$OUT/pip_freeze.txt")
  if [ -n "$missing" ] || { [ -n "$extra" ] && ! { [ "$QUICK" = 1 ] && [ "$extra" = "py-spy==0.4.2" ]; }; }; then
    echo "only in the frozen record: ${missing:-none}"; echo "only in this venv: ${extra:-none}"
    fail 1 "pip freeze differs from requirements-frozen.txt"
  fi
  echo "pip freeze: equal to requirements-frozen.txt ($(wc -l < "$OUT/frozen_record.txt") lines)${extra:+; QUICK tolerated: $extra}"
  # Every pinned package's installed library files, by its RECORD, against the laptop's.
  python3 -I "$N/machine_b_digests.py" record "$SITE" > "$OUT/record_digests_all.txt" || fail 1 "the RECORD digests"
  awk 'NR == FNR { want[$1] = 1; next } ($1 in want)' "$N/laptop_record_digests.txt" "$OUT/record_digests_all.txt" \
    | LC_ALL=C sort > "$OUT/record_digests.txt"
  diff <(LC_ALL=C sort "$N/laptop_record_digests.txt") "$OUT/record_digests.txt" \
    || fail 1 "an installed package's files differ from the laptop's"
  echo "packages: all $(wc -l < "$OUT/record_digests.txt") pinned packages' installed files equal to the laptop's (by RECORD)"
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version(), '| gpu', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))" || fail 1 "torch or CUDA"
  "$PY" -c "import cityflow; print('cityflow:', cityflow.__file__)" || fail 1 "cityflow"
  echo "cpu avx512 flags: $(lscpu | grep -o -E 'avx512[a-z_0-9]*' | sort -u | tr '\n' ' ')"
  "$PY" -c "import torch; print('torch cpu kernels, own choice:', torch.backends.cpu.get_cpu_capability())"
  laptop_kernels "$PY" -c "import torch; print('torch cpu kernels, at the laptop level:', torch.backends.cpu.get_cpu_capability())" \
    || fail 1 "the laptop's kernel level is not available on this CPU"
  [ "$(git -C "$R" rev-parse HEAD)" = "$RUN_COMMIT" ] || fail 1 "the run tree is not at $RUN_COMMIT"
  [ -z "$(git -C "$R" status --porcelain)" ] || fail 1 "the run tree is not clean"
  echo "$SHA_PB  $R/offline/hz4x4_phase_b.py" | sha256sum --quiet -c || fail 1 "hz4x4_phase_b.py differs"
  echo "$SHA_LADDER  $R/offline/hz4x4_ladder.py" | sha256sum --quiet -c || fail 1 "hz4x4_ladder.py differs"
  echo "run tree: $RUN_COMMIT, clean, both campaign modules equal"
  echo "$SHA_CORPORA_MANIFEST  $DRIVE/p93_corpora.sha256" | sha256sum --quiet -c || fail 1 "the corpora's manifest differs"
  (cd "$M" && sha256sum --quiet -c "$DRIVE/p93_corpora.sha256") || fail 1 "a corpus file differs from the laptop's"
  echo "corpora: all $(wc -l < "$DRIVE/p93_corpora.sha256") files equal to the laptop's"

  step 2 "P9.3's seven test files at $RUN_COMMIT, in a scratch clone"
  git clone -q "$M" "$OUT/roots/clone" && git -C "$OUT/roots/clone" checkout -q "$RUN_COMMIT" || fail 2 "the scratch clone"
  laptop_kernels bash -c "cd '$OUT/roots/clone' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '$PY' -m pytest -q -rs -p no:cacheprovider $TESTS" \
    > "$OUT/pytest.txt" 2>&1
  local code=$?
  tail -n 25 "$OUT/pytest.txt"
  [ "$code" -eq 0 ] || fail 2 "pytest exit $code (full output in $OUT/pytest.txt)"

  step 3 "the CUDA smoke: every (tier, arm), seed 101, 2 steps"
  local n=0 tier arm
  for tier in $TIERS; do
    for arm in $ARMS; do
      if [ "$QUICK" = 1 ] && [ "$n" -ge 2 ]; then break 2; fi
      echo "--- smoke $tier/$arm"
      train "$OUT/roots/smoke" "$tier" "$arm" cuda 2 || fail 3 "$tier/$arm"
      n=$((n + 1))
    done
  done
  echo "smoke: $n cell(s) trained, $(ls "$OUT/roots/smoke/p9_3/train" | wc -l) record(s)"

  step 4 "one CPU cell: maxpressure/dt_nomix seed 101, 2 steps -- at the laptop's kernel level (the gate), then B's own"
  train "$OUT/roots/cpu" maxpressure dt_nomix cpu 2 || fail 4 "the CPU cell"
  train_cell "$OUT/roots/cpu_own_kernels" maxpressure dt_nomix cpu 2 || fail 4 "the CPU cell at B's own kernel level"
  "$PY" - "$OUT/roots" "$LAPTOP_CPU_CELL" <<'PYEOF' || fail 4 "the CPU cell differs from the laptop's"
import json, sys
from pathlib import Path
roots, laptop = Path(sys.argv[1]), sys.argv[2]
cell = "p9_3/train/hz4x4_maxpressure_dt_nomix_seed101.json"
gate = json.loads((roots / "cpu" / cell).read_text())["canonical_digest"]
own = json.loads((roots / "cpu_own_kernels" / cell).read_text())["canonical_digest"]
print("CPU cell, the laptop's kernel level:", gate, "|", "EQUAL" if gate == laptop else "DIFFERENT", "to the laptop's", laptop)
print("CPU cell, B's own kernel level (reported):", own, "|", "EQUAL" if own == laptop else "DIFFERENT")
raise SystemExit(0 if gate == laptop else 1)
PYEOF

  local steps=2000
  if [ "$QUICK" = 1 ]; then steps=20; fi
  step 5 "the timing: each arm at $steps steps on maxpressure seed 101, CUDA, GPU trace every second"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv -l 1 > "$OUT/gpu_timing.csv" 2>&1 &
  local trace=$! t0 t1
  for arm in $ARMS; do
    echo "--- timing $arm"
    t0=$(date +%s.%N)
    train "$OUT/roots/timing" maxpressure "$arm" cuda "$steps" || { kill "$trace"; fail 5 "$arm"; }
    t1=$(date +%s.%N)
    echo "WALL $arm $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }') $t0 $t1" | tee -a "$OUT/timing_wall.txt"
  done
  kill "$trace"
  "$PY" - "$OUT" "$steps" <<'PYEOF' || fail 5 "the timing summary"
import csv, json, sys, time
from pathlib import Path
out, steps = Path(sys.argv[1]), int(sys.argv[2])
# G3b's laptop timings (3ea8418, RTX 5080 Laptop GPU): 2,000-step seconds and setup, as in hz4x4_phase_b at 4dcbedc.
g3b = {"dt_nomix": (172.69, 49.4), "bc_m": (15.72, 61.4), "bc_top10_m": (4.85, 54.4), "bc_top10_perix_m": (4.82, 48.1),
       "iql_m": (64.72, 51.7)}
trace = []
with open(out / "gpu_timing.csv") as handle:
    for row in csv.reader(handle):
        if len(row) > 1 and row[1].strip().endswith("MiB"):
            stamp = time.mktime(time.strptime(row[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S"))
            trace.append((stamp, int(row[1].strip().split()[0])))
total_b = total_laptop = 0.0
for line in (out / "timing_wall.txt").read_text().splitlines():
    _, arm, wall, t0, t1 = line.split()
    seconds_laptop, setup_laptop = g3b[arm]
    record = json.loads((out / "roots/timing/p9_3/train" / f"hz4x4_maxpressure_{arm}_seed101.json").read_text())
    seconds = float(record["seconds"])
    setup = float(wall) - seconds
    total_b += 20 * (seconds * (40_000 / steps) + setup)
    total_laptop += 20 * (seconds_laptop * 20 + setup_laptop)
    peak = max([mib for stamp, mib in trace if float(t0) - 1 <= stamp <= float(t1) + 1] or [0])
    print(f"{arm:17s} B: {seconds:8.2f} s training, {setup:6.1f} s setup, peak {peak:5d} MiB | laptop: {seconds_laptop:7.2f} s, "
          f"{setup_laptop:5.1f} s | training ratio B/laptop {seconds * (2000 / steps) / seconds_laptop:5.2f}")
note = "" if steps == 2000 else "  (QUICK: not a projection)"
print(f"projected P9.3 training on B: {total_b / 3600:.2f} h against the laptop's planned {total_laptop / 3600:.2f} h; "
      f"factor {total_b / total_laptop:.2f}{note}")
print(f"peak GPU memory used during the timing (the whole card): {max([mib for _, mib in trace] or [0])} MiB")
PYEOF

  echo
  echo "PREFLIGHT DONE $(date '+%F %T %Z')"
  echo "PREFLIGHT EXIT 0"
}

main 2>&1 | tee -a "$OUT/preflight.log"
exit "${PIPESTATUS[0]}"
