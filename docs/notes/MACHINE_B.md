# MACHINE B — a second workstation for P9.3's training (2026-10-28)

**Status:** procedure issued; machine B's pre-flight not yet run. The ruling is `BRIEF_48` I.6. This note is the procedure and its
record. The coordinator updates it when the pre-flight returns.

## Why, in one paragraph
The author added a second computer (machine B): WSL2 on Windows, an NVIDIA GeForce RTX 3070 Laptop GPU with 8 GB and a 100 W power
cap. **P9.3's whole training (100 cells) moves there if its pre-flight passes. P9.1 stays whole on the laptop.**
- P9.3's cells are all new; no hz4x4 learner was ever trained. So no before/after contrast mixes the two GPUs.
- P9.1's contrasts (`iql_t − iql` and the rest) are against cells trained on the laptop, and keep that hardware.
- A campaign is never split across the two GPUs: the hardware would become a factor inside one comparison.
- A27 names no hardware (checked in its row), so this is a disclosed methods fact, not a deviation.

## The target: everything identical except the GPU
| Item | The laptop (machine A) | Machine B, after this procedure |
|---|---|---|
| OS | Ubuntu 26.04 LTS, glibc 2.43 (`2.43-2ubuntu2.4`) | Ubuntu 26.04 LTS (glibc reported by the pre-flight) |
| Interpreter | `3.12.13 (main, May  6 2026, 18:46:41) [GCC 15.2.0]`, deadsnakes `3.12.13-1+resolute1` | the same packages, repacked from the laptop |
| venv | `/home/filip/rltraffic/.venv`: the frozen record plus `py-spy` | the same files, checked one by one |
| Code | run tree `/home/filip/rltraffic-p93-run-c` at `4dcbedc` | the same commit, from a git bundle |
| Corpora | the eight `datasets_v11/cf_hz4x4__*` | the same files, checked one by one |
| GPU | RTX 5080 Laptop, 16 GB, driver 610.74 | RTX 3070 Laptop, 8 GB, 100 W, driver 591.74 |
| CPU | Intel Core Ultra 9 275HX, 16 threads | reported by the pre-flight |

**Why Ubuntu 26.04 and the laptop's own Python packages.** The author asked how to get exactly 3.12.13.
- The deadsnakes PPA that installed it on the laptop now serves only 3.12.16.
- On Ubuntu 24.04, `/usr/bin/python3.12` is the system's own 3.12.3, which the OS needs, so it cannot be replaced.
- A source build of 3.12.13 gives the right version number but a different binary.
- So the laptop's installed packages are repacked (`dpkg-repack`) and installed on Ubuntu 26.04, the laptop's release. The interpreter
  is then the same binary, glibc the same release, and the copied venv (whose CityFlow was compiled on the laptop) loads unchanged.

## Part A — on the laptop
- **A1 (the author; needs sudo; about 2 minutes).** Repack the six Python packages into the package's `debs/`:
  ```
  sudo apt install -y dpkg-repack && cd /home/filip/transfer_b/debs && sudo dpkg-repack libpython3.12 libpython3.12-dev libpython3.12-stdlib python3.12 python3.12-dev python3.12-venv && sudo chown filip:filip ./*.deb && sha256sum ./*.deb > SHA256SUMS && cat SHA256SUMS
  ```
  It prints six lines, one per `.deb`.
- **A2 (the coordinator; done).** The package `/home/filip/transfer_b`: the git bundle, the venv tar and the corpora tar with
  per-file manifests, the laptop's reference files, the pre-flight script, a README, and `SHA256SUMS` (Part E).
- **A3 (the author).** Copy `/home/filip/transfer_b` to a USB drive formatted **exFAT or NTFS**, not FAT32, because `venv.tar` is
  7 GB. With the drive at `E:`:
  ```
  sudo mkdir -p /mnt/e && sudo mount -t drvfs E: /mnt/e; cp -r /home/filip/transfer_b /mnt/e/ && cd /mnt/e/transfer_b && sha256sum -c SHA256SUMS && (cd debs && sha256sum -c SHA256SUMS)
  ```
  Every line ends in `OK`. (The first part prints an error if the drive is already mounted; that is harmless.)

## Part B — on machine B
- **B1 (Windows PowerShell).** Install Ubuntu 26.04 as a NEW WSL distribution, with the user `filip`:
  ```
  wsl --list --online
  wsl --install -d Ubuntu-26.04
  ```
  - Use the 26.04 name exactly as the list prints it.
  - The 24.04 instance stays as it is, unused; every step below runs in the 26.04 one (`wsl -d Ubuntu-26.04`).
  - If 26.04 is not listed, say so and stop.
- **B2 (Ubuntu 26.04).**
  ```
  sudo apt update && sudo apt full-upgrade -y && sudo apt install -y git tmux && grep PRETTY_NAME /etc/os-release && nvidia-smi
  ```
  It must print `Ubuntu 26.04` and the RTX 3070.
- **B3. The package from the USB drive** (at `E:`), checked:
  ```
  sudo mkdir -p /mnt/e && sudo mount -t drvfs E: /mnt/e; cp -r /mnt/e/transfer_b /home/filip/ && cd /home/filip/transfer_b && sha256sum -c SHA256SUMS && (cd debs && sha256sum -c SHA256SUMS)
  ```
  Every line ends in `OK`.
- **B4. The laptop's Python 3.12.13:**
  ```
  sudo apt install -y /home/filip/transfer_b/debs/*.deb && sudo apt-mark hold libpython3.12 libpython3.12-dev libpython3.12-stdlib python3.12 python3.12-dev python3.12-venv && /usr/bin/python3.12 -c 'import sys; print(sys.version)'
  ```
  It must print `3.12.13 (main, May  6 2026, 18:46:41) [GCC 15.2.0]`.
- **B5. The repository and the run tree:**
  ```
  git clone -b main /home/filip/transfer_b/rltraffic.bundle /home/filip/rltraffic && cd /home/filip/rltraffic && git branch task/p9.3-a27c origin/task/p9.3-a27c && git config core.hooksPath /home/filip/rltraffic/githooks && git worktree add --detach /home/filip/rltraffic-p93-run-c 4dcbedc && git -C /home/filip/rltraffic-p93-run-c rev-parse HEAD
  ```
  It must print `4dcbedc25021634c199897beffeab4d806b9afc2`.
- **B6. The venv and the corpora:**
  ```
  tar -C /home/filip/rltraffic -xf /home/filip/transfer_b/venv.tar && tar -C /home/filip/rltraffic -xf /home/filip/transfer_b/datasets_p93.tar && ls /home/filip/rltraffic/datasets_v11
  ```
  It lists the eight `cf_hz4x4__` directories. Run nothing with the venv's Python before B7: its files are checked first.
- **B7. The pre-flight: about 1 hour without a break.** In tmux:
  ```
  tmux new -s preflight
  bash /home/filip/transfer_b/machine_b_preflight.sh
  ```
  - It is done when its last line is `PREFLIGHT EXIT 0`.
  - If it prints `PREFLIGHT FAILED at step N`, it stops there. Do not repeat it; report.
  - It writes only under `/home/filip/machine_b_out` and refuses to start if that exists.
- **B8. Bring the result back** (about 100 MB):
  ```
  cp -r /home/filip/machine_b_out /mnt/e/
  ```
  Then on the laptop:
  ```
  cp -r /mnt/e/machine_b_out /home/filip/
  ```
  Then say "machine B pre-flight done".

## Part C — what the pre-flight checks, and what passes
1. **Identity**, each a hard check:
   - the venv's every file equals the laptop's;
   - the interpreter string equals the laptop's, and so does `pip freeze`;
   - torch sees CUDA, and CityFlow imports;
   - the run tree is at `4dcbedc` and clean, with `hz4x4_phase_b.py` `5ee2edab…` and `hz4x4_ladder.py` `bd305b2c…`;
   - the corpora's every file equals the laptop's.
2. **P9.3's seven test files at `4dcbedc`:** 0 failed. Skips for data absent from machine B (the evaluation draws) are expected, and
   the coordinator reads each reason.
3. **The CUDA smoke:** all 20 (tier, arm) cells train 2 steps on the RTX 3070. That is every distinct training path on this machine
   (§7 (4)). Seeds add no path.
4. **One CPU cell's canonical digest beside the laptop's** (`4c03f73d…`): reported, not a gate. EQUAL means the CPU arithmetic is the
   laptop's, bit for bit. DIFFERENT would come from the CPU, since every file is equal.
5. **The timing** (each arm at 2,000 steps, as G3b timed the laptop): B's projected hours for the 100 cells, its factor against the
   laptop's 30.67 planned hours, and the GPU's peak memory.
   - **P9.3 moves to machine B unless either:**
     - the factor exceeds about 2.3, when P9.3 would finish later on B than on the laptop after P9.1's ≈ 40.5 h; or
     - any cell runs out of GPU memory in the smoke or the timing.
   - Otherwise P9.3 trains on the laptop after P9.1, under I.4 as written.
   - **The memory to expect,** measured on the laptop in the coordinator's QUICK run (the whole card; the desktop's own use is about
     0.9 GB): `bc_m` 7.2 GB, `dt_nomix` 3.8 GB, `bc_top10_m` and `bc_top10_perix_m` 3.6 GB, `iql_m` 2.0 GB.
     - `bc_m` therefore needs about 6.3 GB of B's 8 GB. It should fit, with little to spare.

## Part D — after the pre-flight (in `BRIEF_48` I.7)
- **The sessions on B use I.4's command,** with two additions: the host's name and the GPU line are printed into the capture.
  `--hours` is B's hours divided by the factor, because the frozen planner's estimates are the laptop's.
- **Back to the laptop, tier by tier** (once a tier's 25 cells exist): the `train/`, `checkpoints/` and `train_logs/` files and the
  capture, copied with a sha256 manifest made on B and checked on the laptop.
  - The records' paths are the same on both machines (`/home/filip/rltraffic/...`), so the laptop's evaluation reads them unchanged.
  - Nothing on the laptop is overwritten: P9.3 has trained nothing there.
- **During a session on B, no GPU-heavy Windows program runs on B.** `bc_m` leaves under 2 GB of the card free; an out-of-memory error
  fails the cell in flight and stops the session, though nothing finished is lost.
- **G4 discloses it:** every training cell was trained on machine B (named, with its GPU, driver and power cap); evaluation ran on the
  laptop's CPU.

## Part E — the package, and the coordinator's test of the pre-flight script
**`/home/filip/transfer_b`** (7.6 GB before the author adds `debs/`). Its `SHA256SUMS` has sha256 `ce6959f7…75e6c5` and reads:
```
d7795734a041ca843ed208db9bfe2a163ab9e7bcc4003b100aa28cf135425fe6  README.txt
c35474a8611068c956da89a18cd9e5d4a824257ba450e6183d3ea0f817908a50  rltraffic.bundle
6f8e21dcd65a5bfc892e6736a0a23ac285265120235c8ec78782577f2b514aba  venv.tar
48140a3cb5d4a72340d3e7341d596b9f32a533caf32d71e90d4b3d26004e5223  venv.sha256
1f0cb68c7da70beed8f4b2be9ce1b8b858a92a4056a0f69bf27d78b07d72a912  datasets_p93.tar
7d6232c815e501656a09f492c190990672e887b74b877a40b7e5074e832e22a0  datasets_p93.sha256
2c843276572e8952f1c95a3a82e123bfde96d429267e9d179624edd98e25035d  machine_b_preflight.sh
0d78d3aaae6f417f5ff5c6858fc3cf7641589c8ea444ab41282c43edc0440bc8  laptop_reference/cpu_cell_record.json
7c1082bcacd56e991f9e30c3791329459290dbc1bfb64ebad281d3273a8b182b  laptop_reference/laptop_env.txt
b49c13b434ff3d3b0add0490e9f8a6b46291a497d6615340a2da1752b77d2a22  laptop_reference/pip_freeze.txt
2d1a63461c47a6522cda425dbae623ad4250eeed8470493301ee56b08a145884  laptop_reference/python_version.txt
```
- The bundle verifies: `main` `fd035f6`, `task/p9.3-a27c` `4dcbedc`.
- `venv.sha256` covers 26,797 files and `datasets_p93.sha256` 2,408.

**The script, tested on the laptop in QUICK mode** (2 smoke cells, 20 timing steps), 21:52–22:17, `PREFLIGHT EXIT 0`:
- every identity check passed;
- the seven files: 159 passed (681 s, beside another session's pytest);
- 2 smoke cells trained;
- the CPU cell's canonical digest, written into a different output root, was EQUAL to the reference;
- the timing summary and the GPU trace were read.

The test proves the script's mechanics on the machine whose files it compares against, not machine B.
