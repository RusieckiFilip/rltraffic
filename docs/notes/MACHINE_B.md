# MACHINE B — a second workstation for P9.3's training (2026-10-28)

**Status (2026-10-29):** machine B **PASSES** its pre-flight under the author's ruling of 2026-10-29, and P9.3 trains on B. The
pre-flight ran on B from 00:04 to 00:46. Steps 1–3 passed. Step 4's bitwise CPU gate failed, and the author replaced it with a
numerical comparison, which B passes (see "Machine B's pre-flight" below). Step 5, the timing, never ran; it runs alone
(`machine_b_timing.sh`, B11–B13), and it sizes B's sessions (`BRIEF_48` I.7). The procedure's first version (USB, 1:1 copy) was
replaced on 2026-10-28 and never run. The scripts and references are in `docs/notes/machine_b/`.

## The rulings
- **The coordinator (`BRIEF_48` I.6):** P9.3's whole training (100 cells) runs on machine B; P9.1 stays whole on the laptop. A
  campaign is never split across the two GPUs. A27 names no hardware, so this is a disclosed methods fact, not a deviation.
- **The author, on the method (2026-10-28, night):**
  - no USB stick; transfer through the author's Google Drive;
  - no 1:1 copy where a download will do, because training on the RTX 3070 cannot be bit-identical to the laptop anyway, so pinned
    versions are enough;
  - the code from GitHub (`task/p9.3-a27c` pushed);
  - the venv built on B from `requirements-frozen.txt`'s exact pins, and CityFlow built from the vendored tree;
  - through Drive only what exists nowhere online: the eight hz4x4 corpora and the six Python 3.12.13 packages, with sha256
    manifests checked on both sides;
  - the venv checked by package versions and wheel hashes, not file by file.
- **The author, on the gate (the same night):** the 2.3× speed rule is dropped, and P9.3 moves to B whatever its speed. Only a
  pre-flight failure keeps it on the laptop: a test failing, a cell erroring or running out of memory, or the CPU cell differing from
  the laptop's. The timings are still taken, to size B's sessions.
- **The author, on the CPU gate (2026-10-29, after B's pre-flight):** the bitwise gate was meant to catch a software difference, and
  step 1 already proves the software identical. Bitwise equality across CPU vendors is not promised even at one instruction-set level,
  because MKL and oneDNN choose kernels and blocking by CPU model. The gate is replaced by a numerical comparison of B's CPU
  checkpoint against the laptop's, tensor by tensor (largest absolute and relative difference), plus the losses. If the differences
  are at floating-point rounding size, B passes and P9.3 trains on B.

## What machine B shares with the laptop, and how each item is checked
| Item | On machine B | Checked by the pre-flight |
|---|---|---|
| OS | Ubuntu 26.04 (WSL2), the laptop's release | reported: glibc, kernel |
| Python 3.12.13 | the laptop's own six deadsnakes packages (`3.12.13-1+resolute1`), repacked with `dpkg-repack`, via Drive | every installed file of each package, by sha256, equal to the laptop's (`laptop_python_debs.txt`); the version string |
| The packages | from PyPI and the PyTorch cu128 index with `machine_b_pins.txt`: the frozen record's 52 pins, each with the sha256 of the one wheel the laptop has installed; pip 26.1.2 first | pip refuses any other file. `pip freeze` must equal `requirements-frozen.txt` line for line, its editable self-entry aside. Each package's installed library files, as its RECORD lists them, must equal the laptop's (`laptop_record_digests.txt`, 53 packages) |
| CityFlow | built on B from the vendored `CityFlow/` | imports (its bytes differ; see below) |
| Code | GitHub: `main`, and `task/p9.3-a27c` with the run tree at `4dcbedc` | the commit, a clean tree, both campaign modules' sha256 |
| Corpora | Drive | every file against the manifest; the manifest against its committed sha256 `7d6232c8…` |
| CPU arithmetic | B's own CPU (AMD Ryzen 5 3600X; the laptop's is Intel). The pre-flight held the four libraries to AVX2; B's sessions do not, since on B the setting changed nothing | **the gate (the author, 2026-10-29):** the 2-step CPU cell agrees with the laptop's to floating-point rounding (`machine_b_compare.py`). The bitwise digest is reported, not required |
| GPU | NVIDIA GeForce RTX 3070 (a desktop card), 8 GB, driver 591.74, 100 W power limit | the one intended difference |

## Verified by the coordinator on the laptop, before machine B
- **The wheels.** All 53 pinned wheels (the 52 and pip) were downloaded from the two indexes B will use. Each one's files equal the
  laptop's installed files: version, file set, and every file's sha256 and size.
  - The laptop's own installed files are unchanged since installation: 19,265 files checked against their RECORDs, 0 changed.
  - **triton 3.6.0 is the one trap.** PyPI's file of that name has another hash (`74caf5e3…`) and is not the laptop's; the PyTorch
    index's file is. Given the hashes, pip can only take the laptop's.
- **A fresh venv, built with B's exact commands** (`/home/filip/b_install_test`):
  - `pip freeze` equals the frozen record line for line, except the CityFlow line, which names the test's path.
  - All 53 packages' installed files equal the laptop's.
  - CityFlow builds and imports. Its binary differs from the laptop's installed one: even a rebuild on the laptop differs.
- **The kernel level.** The laptop's CPU (Intel Core Ultra 9 275HX) has no AVX-512, and torch reports AVX2. The CPU cell gives the
  same digest whether or not the four libraries are held to AVX2. So the setting changes nothing on the laptop, and on B it makes the
  gate test the software rather than the CPU's instruction set.
  - ⚠️ **Corrected 2026-10-29: the last clause was wrong.** Holding the instruction set equal does not make the arithmetic equal:
    the libraries also choose kernels and blocking by CPU model, so a different CPU can round differently at the same level. B's AMD
    CPU did exactly that, and the coordinator's gate failed on it (the author's ruling above replaced it).
- **The pre-flight script, in QUICK mode on the laptop:** see "The coordinator's run of the pre-flight script" below.

## What the method gives up, plainly (the author's question)
**Nothing that bears on a number in the paper.**
- The interpreter is the laptop's binary, every installed file checked.
- The 53 packages are the laptop's, every installed library file checked through its RECORD. That is what the file-by-file copy
  proved for them.
- The code and the corpora are identical.
- **Changed on 2026-10-29: B's CPU arithmetic agrees with the laptop's to rounding, not bit for bit.** The 2-step CPU cell differs
  by 1.7e-7 at most, and its loss is bitwise equal. This bears on no number either:
  - P9.3 trains on B's GPU, not its CPU;
  - every input computed on B's CPU before training (the normalisation statistics, the prompts, the configuration) is equal to the
    laptop's.

**What the 1:1 copy would have shown and this does not:**
1. **CityFlow's compiled extension** is rebuilt on B from the same vendored source, so its bytes differ. CityFlow takes no part in
   training, and P9.3's evaluation runs on the laptop; on B, only the pre-flight's live tests use it.
2. **Files outside the packages' own directories** are not compared: the console scripts in `.venv/bin` (written at install), the
   `.pyc` caches (compiled from identical sources) and the editable project link. Training imports none of them as code.
3. **The OS packages other than Python** (glibc's patch level and the like) are whatever Ubuntu 26.04 ships on B today. They are
   reported, not equalized; the 1:1 copy did not equalize them either.

**What limits reproducibility is unchanged by the method: B's GPU.** Its training cannot be bit-identical to the laptop's, so P9.3's
checkpoints reproduce only on B's GPU model with this software. The paper names the machine for point 8.
- **Measured on 2026-10-29** (`cpu_cell_comparison.txt`): the same 2-step cell on the two GPUs differs by up to 4.0e-4 in a weight,
  in 835,976 of 847,112 elements, and its loss by 1.6e-4 (relative).
- The laptop's own CPU and GPU differ by as much (4.0e-4; loss 2.5e-4). There the cause is known: the model trains with dropout 0.1,
  and the CPU and CUDA draw its masks from different generators. Why the two GPUs differ this much has not been established.
- So a cell trained on B is not the laptop's cell plus rounding, and no contrast may pair the two. None does: every one of P9.3's
  cells trains on B (`BRIEF_48` I.6).

**If anything, the method adds something.** The frozen record's packages now have a public, hash-pinned installer
(`machine_b_pins.txt`), proven equal to the laptop's installed files. The record itself is unchanged and still states what ran. The
one piece that is not public is Python 3.12.13 itself: deadsnakes no longer serves it, so it travels as the laptop's own packages.

## The steps
### On the laptop
- **L1 (WSL, about 3 minutes; run it once).** Repack the six Python packages into the Drive archive, which the coordinator has
  already filled with the corpora and their manifest:
  ```
  sudo apt install -y dpkg-repack && cd /home/filip/machine_b_drive && mkdir -p debs && cd debs && sudo dpkg-repack libpython3.12 libpython3.12-dev libpython3.12-stdlib python3.12 python3.12-dev python3.12-venv && sudo chown filip:filip ./*.deb && sha256sum ./*.deb > SHA256SUMS && cd /home/filip/machine_b_drive && tar -rf machine_b_drive.tar debs && sha256sum machine_b_drive.tar > machine_b_drive.tar.sha256 && cat machine_b_drive.tar.sha256 debs/SHA256SUMS
  ```
  - It prints seven lines: the archive's hash and the six packages'.
  - If it stops partway, do not run it again; report it, because a second run would append the packages twice.
- **L2 (the laptop's browser): upload to Google Drive.**
  1. Open `https://drive.google.com`, signed in to the account with the free space.
  2. Click **New → File upload**.
  3. In the dialog's file-name box, type `\\wsl.localhost\Ubuntu\home\filip\machine_b_drive` and press Enter.
  4. Select `machine_b_drive.tar` and `machine_b_drive.tar.sha256` (Ctrl+click), then click Open.
  5. Wait for "2 uploads complete" (about 0.79 GB).

### On machine B
- **B1 (Windows PowerShell).** Run `wsl --list --online`, then `wsl --install -d Ubuntu-26.04`, using the name the list prints.
  - Create the user `filip`.
  - If 26.04 is not listed, stop and report.
  - The 24.04 instance stays unused.
- **B2 (in the Ubuntu 26.04 terminal, about 10 minutes):**
  ```
  sudo apt update && sudo apt full-upgrade -y && sudo apt install -y git tmux build-essential cmake && grep PRETTY_NAME /etc/os-release && nvidia-smi
  ```
- **B3 (B's browser): download from Google Drive.** Open `https://drive.google.com`.
  - Right-click `machine_b_drive.tar` and choose Download. Drive says it cannot scan the file for viruses; click "Download anyway".
  - Then right-click `machine_b_drive.tar.sha256` and choose Download.
  - Download the two files one at a time: selecting both makes Drive zip them.
  - If the browser saves a second copy as `machine_b_drive (1).tar`, delete the extra copy from Downloads.
- **B4. Copy the files into WSL and check them:**
  ```
  mkdir -p /home/filip/machine_b_drive && cp /mnt/c/Users/*/Downloads/machine_b_drive.tar /mnt/c/Users/*/Downloads/machine_b_drive.tar.sha256 /home/filip/machine_b_drive/ && cd /home/filip/machine_b_drive && sha256sum -c machine_b_drive.tar.sha256 && tar -xf machine_b_drive.tar p93_corpora.sha256 debs && (cd debs && sha256sum -c SHA256SUMS)
  ```
  Every line must end in `OK`.
- **B5. The laptop's Python 3.12.13:**
  ```
  sudo apt install -y /home/filip/machine_b_drive/debs/*.deb && sudo apt-mark hold libpython3.12 libpython3.12-dev libpython3.12-stdlib python3.12 python3.12-dev python3.12-venv && /usr/bin/python3.12 -c 'import sys; print(sys.version)'
  ```
  It must print `3.12.13 (main, May  6 2026, 18:46:41) [GCC 15.2.0]`.
- **B6. The code, from GitHub** (the repository is public; no login needed):
  ```
  git clone https://github.com/RusieckiFilip/rltraffic.git /home/filip/rltraffic && cd /home/filip/rltraffic && git branch task/p9.3-a27c origin/task/p9.3-a27c && git config core.hooksPath /home/filip/rltraffic/githooks && git worktree add --detach /home/filip/rltraffic-p93-run-c 4dcbedc && git -C /home/filip/rltraffic-p93-run-c rev-parse HEAD
  ```
  It must print `4dcbedc25021634c199897beffeab4d806b9afc2`.
- **B7. The corpora:**
  ```
  tar -C /home/filip/rltraffic -xf /home/filip/machine_b_drive/machine_b_drive.tar datasets_v11 && cd /home/filip/rltraffic && sha256sum --quiet -c /home/filip/machine_b_drive/p93_corpora.sha256 && echo "corpora: all files OK"
  ```
- **B8. The venv (about 15–30 minutes; downloads about 3.8 GB):**
  ```
  cd /home/filip/rltraffic && /usr/bin/python3.12 -m venv .venv && .venv/bin/python -m pip install --require-hashes --no-deps -r docs/notes/machine_b/machine_b_pip.txt && .venv/bin/python -m pip install --require-hashes --no-deps -r docs/notes/machine_b/machine_b_pins.txt && (cd CityFlow && ../.venv/bin/python -m pip install --no-deps --no-build-isolation .) && .venv/bin/python -m pip install --no-deps --no-build-isolation -e . && .venv/bin/python -m pip freeze --exclude-editable | wc -l
  ```
  It ends by printing `53`.
- **B9. The pre-flight: about 1 hour without a break.** Run `tmux new -s preflight`, then:
  ```
  bash /home/filip/rltraffic/docs/notes/machine_b/machine_b_preflight.sh
  ```
  - It is done when its last line is `PREFLIGHT EXIT 0`.
  - If it prints `PREFLIGHT FAILED at step N`, do not repeat it; report.
- **B10. Results back through Google Drive:**
  - In B's WSL:
    ```
    cd /home/filip && tar -cf machine_b_out.tar machine_b_out && sha256sum machine_b_out.tar > machine_b_out.tar.sha256
    ```
  - In B's browser: **New → File upload**, type `\\wsl.localhost\Ubuntu-26.04\home\filip` (the distro name `wsl -l` shows), select
    the two `machine_b_out.tar*` files and upload them.
  - In the laptop's browser: download each of the two files, one at a time.
  - In the laptop's WSL:
    ```
    cp /mnt/c/Users/*/Downloads/machine_b_out.tar /mnt/c/Users/*/Downloads/machine_b_out.tar.sha256 /home/filip/ && cd /home/filip && sha256sum -c machine_b_out.tar.sha256 && tar -xf machine_b_out.tar && ls machine_b_out
    ```
  - Then say "machine B pre-flight done".
- **B11 (Windows PowerShell on B, once): B must not sleep or hibernate while it trains.** B hibernated at about 02:10 after the
  pre-flight; a session runs up to 6 hours. Run:
  ```
  powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0
  ```
  Both mean "never" while on mains power. The screen may still turn off.
- **B12. The timing: about half an hour (20–30 minutes) without a break.** In B's WSL, fetch the script, then run it in tmux:
  ```
  git -C /home/filip/rltraffic pull --ff-only && git -C /home/filip/rltraffic log -1 --format='%h %s' | cut -c1-72
  ```
  Then `tmux new -s timing`, and inside it:
  ```
  bash /home/filip/rltraffic/docs/notes/machine_b/machine_b_timing.sh
  ```
  - It is done when its last line is `TIMING EXIT 0`.
  - If it prints `TIMING FAILED at step N`, do not repeat it; report.
- **B13. Results back through Google Drive**, as in B10 with the new name:
  - In B's WSL:
    ```
    cd /home/filip && tar -cf machine_b_timing.tar machine_b_timing && sha256sum machine_b_timing.tar > machine_b_timing.tar.sha256
    ```
  - Upload the two `machine_b_timing.tar*` files from `\\wsl.localhost\Ubuntu-26.04\home\filip`, and download each of them in the
    laptop's browser, one at a time.
  - In the laptop's WSL:
    ```
    cp /mnt/c/Users/*/Downloads/machine_b_timing.tar /mnt/c/Users/*/Downloads/machine_b_timing.tar.sha256 /home/filip/ && cd /home/filip && sha256sum -c machine_b_timing.tar.sha256 && tar -xf machine_b_timing.tar -C /home/filip/machine_b_result && ls /home/filip/machine_b_result/machine_b_timing
    ```
  - Then say "machine B timing done".

## What passes
**P9.3 moves to machine B unless the pre-flight fails** (the author's gate):
- a check of step 1 fails;
- a test fails;
- a smoke or timing cell errors or runs out of GPU memory;
- or the CPU cell differs from the laptop's by more than floating-point rounding. As amended by the author on 2026-10-29, this is
  `machine_b_compare.py`'s bound; the bitwise digest it replaced is reported only.

**The timings size B's sessions.** The frozen planner's estimates are the laptop's, so `machine_b_timing.sh` finds the `--hours` at
which every session the queue can plan takes at most 6 hours on B, by B's measured cells, and prints the sessions.

**The memory to expect,** measured on the laptop (the whole card, about 0.9 GB of it the desktop's own): `bc_m` 7.2 GB, `dt_nomix`
3.8 GB, `bc_top10_m` and `bc_top10_perix_m` 3.6 GB, `iql_m` 2.0 GB. So `bc_m` needs about 6.3 GB of B's 8 GB.

## Machine B's pre-flight, 2026-10-29 (verified by the coordinator from its output)
The output came back through Drive as `/home/filip/machine_b_out.tar` (sha256 `8d3dc160…`, checked). The author extracted it to
`/home/filip/machine_b_result/machine_b_out`, which equals the tar (`tar -d`).
- **Step 1 (00:04:53), identity: every check passed.**
  - B is `DESKTOP-H2DV9E3`: Ubuntu 26.04.1 LTS, glibc 2.43-2ubuntu2.4 (the laptop's), WSL kernel 6.18.54.1.
  - CPU: AMD Ryzen 5 3600X, 12 threads, no AVX-512; torch chooses AVX2. WSL has 15 GB.
  - GPU: NVIDIA GeForce RTX 3070, driver 591.74, 8,192 MiB, power limit 100 W. torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900,
    capability (8, 6).
  - Equal to the laptop's: the six Python packages' files, the interpreter string, `pip freeze` (53 lines) and all 53 packages'
    RECORD digests. The run tree is at `4dcbedc`, clean, with both modules equal. All 2,408 corpus files are equal.
- **Step 2 (00:05:08): 138 passed, 21 skipped, 0 failed** (269.9 s); 138 + 21 is the laptop's 159.
  - Every skip names data that only the laptop's main tree holds and that training does not read: hz4x4's held-out draws
    (`scenarios/draws`), the grid's `cf_grid4x4` corpus, its committed evaluation output, P2.1's results, and the v1.1 manifests
    with draws 1–5.
- **Step 3 (00:09:41–00:43:22):** all 20 (tier, arm) cells trained on CUDA at 2 steps; 20 records. Setup took about 77 s per
  `dt_nomix` cell and 106–110 s per other cell.
- **Step 4 (00:43:24–00:46:20):** the author's report says 00:43–00:44, which is the first of its two cells.
  - The CPU cell's digest is `84da36bc…`, at the laptop's kernel level and at B's own, and the two checkpoints are byte-identical.
    The laptop's is `4c03f73d…`.
  - So the bitwise gate failed, and the pre-flight stopped before step 5.

**The comparison, under the author's ruling.** The script is `machine_b_compare.py`, committed in `b029604` before it was run; its
output is `cpu_cell_comparison.txt`.
- **The bound, fixed before the comparison ran:**
  - every tensor's largest absolute difference is at most 1e-5, a tenth of one AdamW step. The cell trains at learning rate 1e-4
    with warmup 1, so each step moves nearly every weight by about 1e-4;
  - the recorded loss agrees within 1e-5, relative;
  - the checkpoint's corpus-derived contents are equal.
- **The reference** is the laptop's CPU cell, regenerated at `4dcbedc` into `output/machine_b/laptop_cpu_2step`. Its digest is
  `4c03f73d…` again; the files are hashed in `output/machine_b/SHA256SUMS`.
- **Result: within the bound.**
  - 46 of the 66 tensors differ, in 11,444 of 847,112 elements (1.4 %). The other 20 are bitwise equal.
  - The largest absolute difference is 1.7e-7, in `blocks.0.spatial.proj.weight`. That is about 1/580 of one optimiser step, 58
    times inside the bound, and about 23 ulps of that tensor's largest weight.
  - Relative to each tensor's largest weight, the difference is at most 1.9e-6.
  - In ulps of each weight's own value, the largest distance is 16,384. It is at a weight near zero, where one ulp is about 1e-11.
  - The step-2 loss is bitwise equal: 2.0727832317352295 on both machines.
  - All 9 corpus-derived entries (configuration, normalisation statistics, prompts, mask) are equal.
- **The first step's loss**, which the author's ruling also names, is not in any record: a record keeps only the last step's loss.
  - The laptop's 1-step cell gives 2.133220911026001 (`output/machine_b/laptop_cpu_1step`).
  - B's comes from B12's step 2, and its 1-step checkpoint gets the same comparison.
- **So B passes, and P9.3 trains on B.**
- **The kernel-level settings are dropped from B's sessions.** On B they changed nothing (its two checkpoints were byte-identical),
  and the gate they served is gone.

## The coordinator's run of the pre-flight script on the laptop (QUICK mode)
On 2026-10-28, 22:53–23:07, with 2 smoke cells and 20 timing steps, the result was `PREFLIGHT EXIT 0`.
- **Step 1:** every check passed. The six Python packages' files and the interpreter matched. `pip freeze` equalled the frozen record,
  with the laptop's own `py-spy` tolerated in QUICK mode only. All 53 packages' files matched, torch saw CUDA on the RTX 5080, and the
  run tree and the 2,408 corpus files matched. The CPU has no AVX-512 flags, and torch reports AVX2 both by its own choice and at the
  laptop level.
- **Step 2:** 159 passed (333 s).
- **Step 3:** 2 cells trained.
- **Step 4:** the CPU cell was EQUAL at the laptop's kernel level and at the machine's own.
- **Step 5:** the summary and the per-arm peak memory were read (`bc_m` 7,215 MiB, `dt_nomix` 3,809 MiB).

The run proves the script's mechanics on the machine its references come from, not machine B.

**The timing script, in QUICK mode on the laptop (2026-10-29, 11:23–11:34): `TIMING EXIT 0`.**
- All five arms trained at 20 steps, and the GPU trace was read: `bc_m` peaked at 6,777 MiB for the whole card.
- In QUICK mode the session planner runs on G3b's own estimates, and it reproduced I.4's plan exactly: `--hours 6.00`; sessions of
  20, 9, 23, 23, 20 and 5 cells; 5.80, 5.76, 5.72, 5.72, 5.80 and 1.87 hours; 30.67 hours in all.
  - The search was then tightened so that the bound holds for a session that starts from any pending cell. The summary's final
    text was re-run on the same run's data, with the same result.
- The 1-step CPU cell gave loss 2.133220911026001 and digest `1c6e4d5f…`, equal to the reference made 17 minutes earlier.

## What follows (`BRIEF_48` I.7)
- **B's session command** is I.4's, with these additions:
  - the host's name and the GPU line printed into the capture;
  - `--hours` as `machine_b_timing.sh` prints it, B's measured timings;
  - no kernel-level settings (the author's ruling of 2026-10-29 made them moot).
- **B stays awake** (B11) for every session.
- **During a session on B, no GPU-heavy Windows program runs on B.** An out-of-memory error fails the cell in flight and stops the
  session, but nothing finished is lost.
- **Back to the laptop, tier by tier, through Drive.** Each tier's `train/`, `checkpoints/` and `train_logs/` and the capture travel
  with a sha256 manifest made on B and checked on the laptop. The records' paths are the same on both machines.
- **G4 discloses** that every training cell ran on machine B (named, with its GPU, driver and power cap) and that evaluation ran on
  the laptop's CPU.
