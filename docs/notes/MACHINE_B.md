# MACHINE B — a second workstation for P9.3's training (2026-10-28)

**Status:** procedure issued (this version replaces the USB / 1:1-copy version of the same evening, which was never run). Machine B's
pre-flight has not run yet. Its scripts and references are in `docs/notes/machine_b/`.

## The rulings
- **The coordinator (`BRIEF_48` I.6):** P9.3's whole training (100 cells) runs on machine B; P9.1 stays whole on the laptop. A
  campaign is never split across the two GPUs. A27 names no hardware, so this is a disclosed methods fact, not a deviation.
- **The author, on the method (2026-10-28, night):**
  - no USB stick; transfer through his Google Drive;
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

## What machine B shares with the laptop, and how each item is checked
| Item | On machine B | Checked by the pre-flight |
|---|---|---|
| OS | Ubuntu 26.04 (WSL2), the laptop's release | reported: glibc, kernel |
| Python 3.12.13 | the laptop's own six deadsnakes packages (`3.12.13-1+resolute1`), repacked with `dpkg-repack`, via Drive | every installed file of each package, by sha256, equal to the laptop's (`laptop_python_debs.txt`); the version string |
| The packages | from PyPI and the PyTorch cu128 index with `machine_b_pins.txt`: the frozen record's 52 pins, each with the sha256 of the one wheel the laptop has installed; pip 26.1.2 first | pip refuses any other file. `pip freeze` must equal `requirements-frozen.txt` line for line, its editable self-entry aside. Each package's installed library files, as its RECORD lists them, must equal the laptop's (`laptop_record_digests.txt`, 53 packages) |
| CityFlow | built on B from the vendored `CityFlow/` | imports (its bytes differ; see below) |
| Code | GitHub: `main`, and `task/p9.3-a27c` with the run tree at `4dcbedc` | the commit, a clean tree, both campaign modules' sha256 |
| Corpora | Drive | every file against the manifest; the manifest against its committed sha256 `7d6232c8…` |
| CPU arithmetic | MKL, torch, oneDNN and numpy held to the laptop's kernel level (AVX2), for every pre-flight step and every session | **the gate:** the 2-step CPU cell's canonical digest equals the laptop's `4c03f73d…`; B's own-kernel result is reported beside it |
| GPU | RTX 3070 Laptop, 8 GB, 100 W cap | the one intended difference |

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
- **The pre-flight script, in QUICK mode on the laptop:** see "The coordinator's run of the pre-flight script" below.

## What the method gives up, plainly (the author's question)
**Nothing that bears on a number in the paper.**
- The interpreter is the laptop's binary, every installed file checked.
- The 53 packages are the laptop's, every installed library file checked through its RECORD. That is what the file-by-file copy
  proved for them.
- The code and the corpora are identical, and the CPU arithmetic is the laptop's (the gate).

**What the 1:1 copy would have shown and this does not:**
1. **CityFlow's compiled extension** is rebuilt on B from the same vendored source, so its bytes differ. CityFlow takes no part in
   training, and P9.3's evaluation runs on the laptop; on B, only the pre-flight's live tests use it.
2. **Files outside the packages' own directories** are not compared: the console scripts in `.venv/bin` (written at install), the
   `.pyc` caches (compiled from identical sources) and the editable project link. Training imports none of them as code.
3. **The OS packages other than Python** (glibc's patch level and the like) are whatever Ubuntu 26.04 ships on B today. They are
   reported, not equalized; the 1:1 copy did not equalize them either.

**What limits reproducibility is unchanged by the method: B's GPU.** Its training cannot be bit-identical to the laptop's, so P9.3's
checkpoints reproduce only on B's GPU model with this software. The paper names the machine for point 8.

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

## What passes
**P9.3 moves to machine B unless the pre-flight fails** (the author's gate):
- a check of step 1 fails;
- a test fails;
- a smoke or timing cell errors or runs out of GPU memory;
- or the CPU cell at the laptop's kernel level differs from the laptop's.

**The timings size B's sessions.** B's factor against the laptop's 30.67 planned hours sets each session's `--hours` (the frozen
planner's estimates are the laptop's).

**The memory to expect,** measured on the laptop (the whole card, about 0.9 GB of it the desktop's own): `bc_m` 7.2 GB, `dt_nomix`
3.8 GB, `bc_top10_m` and `bc_top10_perix_m` 3.6 GB, `iql_m` 2.0 GB. So `bc_m` needs about 6.3 GB of B's 8 GB.

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

## What follows (`BRIEF_48` I.7)
- **B's session command** is I.4's, with these additions:
  - the host's name and the GPU line printed into the capture;
  - the four kernel-level settings;
  - `--hours` set from B's factor.
- **During a session on B, no GPU-heavy Windows program runs on B.** An out-of-memory error fails the cell in flight and stops the
  session, but nothing finished is lost.
- **Back to the laptop, tier by tier, through Drive.** Each tier's `train/`, `checkpoints/` and `train_logs/` and the capture travel
  with a sha256 manifest made on B and checked on the laptop. The records' paths are the same on both machines.
- **G4 discloses** that every training cell ran on machine B (named, with its GPU, driver and power cap) and that evaluation ran on
  the laptop's CPU.
