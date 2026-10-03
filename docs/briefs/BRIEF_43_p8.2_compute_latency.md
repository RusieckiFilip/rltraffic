# BRIEF_43 — P8.2: the compute-and-latency table — training cost, inference ms per decision, parameter counts of the paper's methods (A25(c): measurements of existing artifacts, no hypothesis evaluated)

**Mode:** Explore → Plan (gate G0: `docs/plans/p8.2.md`) → Code (tests first) → the timing run (gate G2) → Commit. Branch
`task/p8.2-compute-latency` from `main`. One task, one session.

**Registered as:** `PREREGISTRATION` A25(c) (`v2.6-prereg-a25-a26`): *"P8.2's compute-and-latency table (training time, inference ms per
decision, parameter counts — measurements of existing artifacts, no new evaluation of any hypothesis). After those, the paper is written
from what is on `main`."* This is the LAST measurement before the paper.

## 0. Frozen interface contracts (read them from disk; do not infer)
- `docs/CONTRACTS.md` (C6 v1.1; the env API `info = env.reset(seed=...)`, `reward, terminated, truncated, info = env.step(action)`).
- Frozen paths (CLAUDE.md §1) untouched; `agent/DTAgent.py`, `offline/dataset.py`, `offline/dt_gate.py`, `offline/offline_baselines.py`,
  `offline/method_tier_grid.py`, `offline/context_sweep.py` and every existing `offline/` module UNCHANGED — new code goes in new files.
- No new dependency (numpy / stdlib / torch / pytest). `py-spy` in the venv is a diagnostic tool, never imported.
- Every checkpoint and record is read at a digest that a committed artifact or manifest names; nothing is retrained, re-evaluated or moved.

## 1. Why this task exists
Reviewers of offline-RL and TSC papers ask three things the paper cannot yet answer from `main`: what each method COSTS to train, what it
costs to RUN per decision, and how BIG it is. Two comparisons carry the paper's argument and need these numbers: (i) **offline against
online** — the offline methods learn from a logged corpus with ZERO environment interaction, against the MAPPO teacher's training on
360,000 simulated decisions per seed (`output/experiments/p2_1_mappo_nominal_{060,1000}/results.json`: `train_sec` per cell, CPU,
`thread_num 1`); (ii) **H4's context length** — K = 1 is not worse than K = 20 on the P4 scenario (Amendment E of `BRIEF_42`), and its
inference cost and size are the practical side of that result. Most of the table already exists in committed records (§3); the one
measurement that does not is the policy's decision latency, because every recorded rollout time includes the simulator's.

## 2. Scope fence — what NOT to build
- No training of any model; no evaluation episode whose outcome is recorded; no ATT, reward or metric of any timing episode is written
  anywhere (the timing harness records times and decision counts ONLY — so nothing it produces can be read as a result).
- No change to any existing module, driver or committed artifact; no new registration; no MAPPO re-run (its training cost is READ).
- No CQL, no new baseline, no new scenario; `cf_cologne3` out (no claim of the paper uses it).
- No GPU-only number without its CPU counterpart: a traffic controller is deployed on a CPU; the CPU single-thread latency is the
  primary inference figure, the GPU's beside it.

## 3. What the table holds
**Rows — every (scenario, method, configuration) whose RESULT the paper reports under C1, C3 and H4 (A25(d)), at its published
configuration.** G0's plan enumerates them from the committed artifacts with the checkpoint paths and the training record of each; the
expected list, to be confirmed or corrected with evidence:
- `cf_hz1x1` (C1, H4): the DT at P4's recipe (K = 20; `output/p4_dt/`, `docs/data/p4_training.json`); the DT at K ∈ {1, 2, 5, 10} and the
  two equal-supervision arms (`docs/data/p5_3c_train.json`); BC, %BC (`bc_top10`) and IQL across the ladder tiers
  (`docs/data/p4_4_training.json`, `p4_6_training.json`, `p4_7_training.json`); the MAPPO teachers 060 and 1000; MaxPressure; fixed-time.
- `cf_grid4x4` (C1, C3): `dt_spatial`, `dt_nomix`, BC, %BC, IQL at the tiers the paper reports (P5.1 / P5.2's records); the C3 DT and its
  SUMO fine-tunes (`docs/data/p7_3b_anchor_training.json`, `docs/data/p7_3c_finetune.json`); the MAPPO teachers; MaxPressure; fixed-time.

**Columns:**
1. **Parameters** — counted from each checkpoint by loading it through its own loader and summing `numel()` over the model's
   `parameters()`; for methods that train more networks than they deploy (IQL's Q and V networks beside its policy; MAPPO's critic beside
   its actor) BOTH counts — *trained* and *deployed*. Cross-checked against every `parameter_count` a record already holds.
2. **Training cost** — gradient steps, batch, the size of the training data (windows or transitions), and wall seconds as the committed
   record holds them (median [min–max] over seeds, never the mean: Amendment C of `BRIEF_42` names two wall-time stalls caused by the
   author's game), with each record's device, GPU name, torch version and thread regime; **environment interactions** — 0 for every offline
   method, and for MAPPO `train_episodes × max_steps` decisions per seed (and × intersections where the plan shows the count is per
   intersection), with `train_sec` as recorded and its regime from `docs/briefs/RUNSPEC_01_p2.1_mappo.md` (the worker count the manifest
   does not record — found, or stated as unknown).
3. **Inference — ms per decision**, NEW, measured by §4's harness: median and p95 over all timed decisions, on CPU with ONE thread
   (primary) and on CUDA (beside it), per scenario; one decision = the agent's whole decision for one control step (all 16 intersections on
   grid4x4, through the agent's own deployment path, batched or not as the agent does it), and per intersection beside it.

## 4. The latency harness — `offline/compute_latency.py` (new)
- Loads each row's checkpoint at its recorded digest through the method's own loader (for the DT: `rtg_calibration.agent_with_target`
  with the registered prompt, as every evaluation did) and runs **real CityFlow episodes on held-out draws**, wrapping ONLY the agent's
  decision call (`act(info, explore=False, update_memory=True)` or the method's equivalent) in the timer — the env's `step` is outside it.
- Clock: `time.perf_counter_ns`; on CUDA `torch.cuda.synchronize()` before both readings; the first W decisions of each episode are
  warm-up and excluded (W in the plan, ≥ 20 for the DT, whose history fills over K steps — the latency at a full window is the one that
  matters).
- Regime: CPU runs with `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG` unset; the run
  records the CPU model (`/proc/cpuinfo`), the GPU name and driver (`nvidia-smi`), torch's version, and the machine's load before and
  after; a canary before the first timing (the CityFlow canary's 2.0 s ceiling) refuses a throttled machine.
- N: the plan proposes the number of episodes per row per device (expected 3 episodes × 360 decisions, on draws 1000–1002), so that the
  p95 rests on ≥ 1,000 decisions; rows run SEQUENTIALLY, one process at a time — no pool: concurrency is exactly what a latency figure
  must not include.
- Output: `output/p8_2/latency/<UTC stamp>/<row>_<device>.json` (per row: the decision count, the per-decision nanoseconds, the median,
  the p95, the regime, the checkpoint digest — and NO outcome of the episode), written once each; a manifest
  `output/SHA256SUMS_p8_2.txt`.

## 5. The table builder — `offline/compute_table.py` (new)
Reads the committed training records and the MAPPO manifests by digest, counts the parameters from the checkpoints, reads the latency
records, and writes the artifact `output/p8_2/artifacts/p8_2_compute.json` (format `p8.2-compute/1.0`): the rows, every number with its
SOURCE (file, sha256, the JSON path or the measurement), the hardware block, and `what_this_does_not_say` (wall times are indicative on a
shared laptop and were measured in different sessions; latency is the policy's alone, not the controller's end-to-end loop; no number here
evaluates any hypothesis). Refuses rather than writes when a record is absent, at another digest, or a row has no source.

## 6. Tests — first, red for their own reasons; the named mutations executed and pasted
- **T-params (load-bearing):** each method's parameter count by two routes — the model's `parameters()` and an independent sum over the
  checkpoint's state_dict tensors that are parameters (buffers excluded) — equal; trained ≥ deployed. *Mutation:* buffers counted → dies.
- **T-timer (load-bearing):** a fake env whose `step` sleeps 50 ms and a fake agent whose decision sleeps 2 ms → the harness reports
  ≈ 2 ms per decision (within 1 ms), never ≈ 52 ms; the warm-up decisions excluded by count; the CUDA path calls `synchronize` (a spy).
  *Mutations:* the timer around `step` → dies; warm-up not excluded → dies.
- **T-stats:** the median and the p95 by a second route (`numpy.percentile` against a sorted-index computation written in the test) on a
  fixed sample.
- **T-sources:** every training-cost number in the artifact equals the committed record's value at the JSON path the artifact names, read
  by the test with `json`; a record at another digest refuses; a row without a source refuses.
- **T-no-outcome:** the latency record of a fake episode contains no key that names a metric (`att`, `reward`, `travel`, `queue`, …).
- **T-regime:** the CPU timing refuses unless the process runs one torch thread with `CUBLAS_WORKSPACE_CONFIG` unset.
Suite discipline: gated tests name their artifact; `check_test_hygiene.sh` and `check_english.sh` falsified first; mutants COMMITTED in a
throwaway worktree; **the new test files also run once in a depth-1 clone made by `git clone --depth 1 --branch <branch> file://…`**
(`BRIEF_42` Amendment F.1).

## 7. Gates, in order
| # | Gate | Runs it | Checks | Stops the task if | You learn it by |
|---|---|---|---|---|---|
| G0 | Plan | coordinator, from `docs/plans/p8.2.md` | the row inventory with its evidence; every source; the latency protocol (N, W, draws, devices, the regime); MAPPO's regime from RUNSPEC_01 | a row or a source is wrong | Amendment A on `main` |
| G1 | Code review + the timing run's pre-flight | coordinator + one reviewer (≤ 15 min, findings file); the coordinator's mutants | T-params, T-timer, the sources, the refusals, no outcome recorded | a defect | Amendment B |
| G2 | The timing run | **the author** — one command, a quiet machine | — | — | the capture |
| G3 | Read | coordinator, from disk: the latency records first, the medians and p95 recomputed from the per-decision nanoseconds by an independent route; every table number traced to its source | the table | a number untraceable or wrong | Amendment C |
| G4 | Packet | the implementer → **"P8.2 done"** | §9 | — | — |
| G5 | Merge review | coordinator spawns one (two mandates: the numbers to their sources; the code by mutation) + the depth-1 run | — | a blocker | the merge, §6's box ticked, the CI ceiling if it moves |

## 8. Definition of Done
- [ ] `docs/plans/p8.2.md` approved (G0); every row of the paper's C1, C3 and H4 results present, each with its sources.
- [ ] `offline/compute_latency.py`, `offline/compute_table.py` and their tests; every §6 test red first, then green; every named mutation
      executed and pasted; hygiene and English; the depth-1 run.
- [ ] The timing run complete on a quiet machine (G2), the latency records under their manifest.
- [ ] `docs/data/p8_2_compute.json` committed by hand, byte-identical to the builder's output, with a T-regress that regenerates it.
- [ ] `docs/returns/P8.2.md` per §9; then "P8.2 done". No frozen file, no existing module, no new dependency touched.

## 9. Return Packet
`docs/returns/TEMPLATE.md`, plus: the row inventory and why each row is in the paper; the table itself (rendered) with every number's
source; the hardware block; the timing capture; the regime of every recorded training time; what the table does not say; the
AI-assistance record's four lines; one paragraph on what the paper's compute section will assume.

---

# ✅ AMENDMENT A — 2026-10-03, gate G0: PLAN APPROVED (`docs/plans/p8.2.md` @ `a7c43e1`, 507 lines) — three errors in THIS brief corrected, every question ruled as proposed, two requirements added

## A0 — Verdict
Approved. The plan inventories every C1, C3 and H4 row with its committed evidence, hashed **465 / 465** candidate checkpoints against
the digests their sources name (0 mismatches), and found three errors in this brief, each with evidence. **Verified by the coordinator
(2026-10-03, by running commands):** the DT's parameter count is **647,176** at K = 20 (P4), K = 1 and K = 10 (P5.3c) and for P7.3b's
anchor; the anchor's checkpoint and record say `cityflow1x1` / `intersection_1_1` / `state_dim 25`; MAPPO@1000's checkpoints hold
`steps_done 360000` with actors 20,872 + critic 20,353 (hz1x1) and 364,672 + 101,008 (grid4x4); `MAPPOAgent.act` calls
`self.estimate_values(joint_state)` after the action loop on the deterministic path too (`agent/MAPPOAgent.py:247`); `p4_gate.json`'s
cells are `madt, mappo1000, mappo500, maxpressure`; `output/p5_1/training_baselines.json`'s runs carry no seconds.

## A1 — Corrections to this brief (the coordinator's errors, logged in `PROJECT_PLAN` §8)
1. **The MAPPO rows.** §3 named "the MAPPO teachers 060 and 1000"; the paper REPORTS MAPPO@1000 (both scenarios) and **MAPPO@500
   (hz1x1, the P4 gate's second online baseline)**; MAPPO@060 appears only as a corpus-ladder tier. Rows per Q1.
2. **P7.3b's anchor is an hz1x1 model.** §3 filed `p7_3b_anchor_training.json` under `cf_grid4x4`; it belongs to hz1x1's C3 rows. The
   grid4x4 C3 model is **P5.2's `dt_nomix_h4`** at `mappo1000` and its SUMO fine-tunes are P7.3c's.
3. **K does not change the DT's size.** §1 (and the plan row of 2026-10-03) implied K = 1 is smaller; the parameter count is identical
   at every K (the K window enters only tensor shapes, `agent/DTAgent.py:319`). K = 1 is cheaper PER DECISION, not smaller. Q15's
   sentence goes into `what_this_does_not_say`.

## A2 — Rulings: Q1–Q15 ALL ACCEPTED AS PROPOSED
Q1 (MAPPO@1000 both scenarios, MAPPO@500 hz1x1, MAPPO@060 both as the labelled ladder-tier, low-interaction point; 200 and grid4x4@500
out) · Q2 (rows (a)–(g) in) · Q3 (one row per arm, tiers inside, an `architecture` key) · Q4 (the three `results.json` digests pinned in
the builder as of 2026-10-03, with the plan's three corroborations stated in the artifact) · Q5 ("not recorded", with the reason, never a
reconstruction) · Q6 (the random-tier DT seconds from the manifest-pinned logs, labelled) · Q7 (`trained`, `deployed`, `stored`,
`executed_per_decision`) · Q8 (draws 1000–1002: no outcome is recorded, T-no-outcome guarantees it, and the held-out draws are the
deployment distribution; the reason stated in the artifact) · Q9 (canaries before and after; a closing canary above 2.0 s refuses the
run) · Q10 (the lean driver, run from a detached worktree at the reviewed commit, the author's token `output/p8_2_runs/TOKEN_latency`) ·
Q11 (nearest-rank p95) · Q12 (heuristics on CUDA "not applicable") · Q13 (seed 101 timed; parameters counted on all five, equal within
a row) · Q14 (the C3 rows timed on CityFlow, the deployment domain stated) · Q15 (the size sentence).

## A3 — Two requirements ADDED
1. **CityFlow's destructor race (`DEFERRED` 104) inside the timing run.** Each (row, device) process rolls three real episodes, ≈ 240
   in all; at the rate observed on 2026-09-30 (≈ 0.11 % of episodes, every one at an episode's end) a hang is likely enough to plan for.
   Required: the row process writes its latency record (exclusively) as soon as the last timed decision of its third episode is taken,
   BEFORE any env is torn down; the driver runs each row process under a timeout (the plan proposes it from the G1 pre-flight) with at
   most THREE attempts; an attempt that times out is killed and recorded as `hung` in the run record; a record already written by a
   process that then hangs is kept (the exclusive write refuses a second); no record is ever written from a partial attempt. Tests,
   red first: a fake row process that hangs AFTER writing → killed, record kept, no re-run; one that hangs BEFORE writing → killed and
   re-run, at most three attempts, then the run FAILS naming the row.
2. **MAPPO on CUDA.** MAPPO was trained on CPU (`results.json` `device cpu`). If its existing factory cannot place the agent on CUDA
   without a change to frozen code, its CUDA cell is *"not applicable: the deployed path is CPU-only"* with the code line; if it can, it
   is timed like every other row. Confirmed in C2 by reading the code, and stated.

## A4 — Next
C1 (tests and signature-only skeletons, red for their own reasons, A3's two driver tests included) · C2 (the implementation, green,
the mutations committed and pasted) · C3 (the driver and the G1 pre-flight: two rows with the canary, the per-row timeout and the run's
expected duration derived from it) — committed on the branch, NOT pushed; the new test files once in a depth-1 clone (F.1's command);
then **"P8.2 C1–C3 done"** → gate G1.

---

# ⚠️ AMENDMENT B — 2026-10-03, gate G1: FIX FIRST (C4–C5), then gate G1.1, then the token — no number found wrong at `1873996`, seven load-bearing properties untested, and the machine in Windows' battery-favouring power mode

## B0 — Verdict and what was verified (by the coordinator, by running commands; the record: `docs/reviews/P8.2-G1.md`)
**FIX FIRST.** Nothing the table would publish was found wrong at `1873996`, and the pre-flight stands. But seven of the coordinator's
thirteen mutants SURVIVED — each a property a number in the table rests on that no test holds — and the timing run would have measured in
the power mode that favours the battery. Verified:
1. **The pre-flight**, from disk: 14 / 14 manifest lines `OK`; `preflight.json` at the pinned `1abf1120…`; its timeouts (120 s ×3) and
   its expected duration (1,138.49 s) recomputed independently and equal.
2. **The 78-cell smoke** (`docs/notes/p8_2_g1/`): every (row, device) cell of the registry built through `_evaluation_inputs`,
   `_row_factory` and `_make_env` exactly as `run_row` builds it, ONE episode each, no record, no timing kept. **All 78 exited 0 with 360
   decisions; every torch cell's parameters on the requested device, no CPU cell initialised CUDA, every CUDA cell allocated memory;
   every DT at its checkpoint's own context length** (H4 1/2/5/10/20, `k1_b1280` 1, `k2_b640` 2, all other DTs 20); parameter totals as
   G0. Slowest one-episode cell, start and load included, on a loaded machine: 23 s (16 s on grid4x4) — far inside 120 s.
3. **Three read-only reviewers** (≤ 15 min each, findings files verbatim in the review record): R1 the harness and driver, R2 the
   builder's mechanics, R3 the row specs against the records — all PASS-WITH-NOTES, no blocker. R3: 72 / 72 pins at their digests,
   every training value the right seed's and field, all seven declared absences real, the five outliers the only seeds above 2×.
4. **Thirteen coordinator mutants**, committed at `1873996` in a throwaway worktree (specs `docs/notes/p8_2_g1/mutants_1873996.json`):
   **KILLED 6** (CM3 warm-up once not per episode; CM7 the closing synchronize; CM8 median → mean; CM9 MAPPO's ×16; CM10 IQL's
   deployed = trained in both routes; CM12 a non-reproducing canary) · **SURVIVED 7**: **CM1** a CUDA DT row timed on CPU; **CM2** CPU
   baseline rows placed on CUDA; **CM4** the builder publishing µs as ms; **CM5** the row process timing a checkpoint at another
   digest; **CM6** the children started without `-P`; **CM11** the builder accepting a record of another checkpoint; **CM13** the real
   row process never synchronising on CUDA (R1's MAJOR 10a, falsified here rather than taken on its word).

## B1 — The power regime (the author's reviewer asked for it; it is a finding, not a formality)
**Verified 2026-10-03, read-only:** `/sys/class/power_supply/AC1` is `type Mains`, `online 1` — **the power source IS visible from WSL2**
(the interim packet's *"not visible from WSL2"* is wrong; a claim of impossibility made without the check); Windows agrees
(`PowerLineStatus Online`, `Win32_Battery.BatteryStatus 2`). The power plan is Balanced (`381b4222-…`), and **the AC power-mode overlay
is `961cc777-2547-4f9d-8174-7d86181b8a7a` — "Better Battery", which Windows 11 labels "Best power efficiency"** (`reg.exe query
HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes`, `ActiveOverlayAcPowerScheme`). Microsoft's documentation of the slider
(*Customize the Windows performance power slider*, learn.microsoft.com) gives the overlays — `961cc777-…` Better Battery, `3af9b8d9-7c97-
431d-ad78-34a8bfea439f` Better Performance (the out-of-box default), `ded574b5-45a0-4f42-8737-46345c09c238` Best Performance — and says
the overlay sets the CPU's processor power management and that **power throttling is engaged in every mode but Best Performance**. The
driver's header already asks for "mains power and a performance plan"; nothing checked it. Required, tests red first with injected
readers (a fake `power_supply` root, a fake `reg.exe` output):
1. **Every latency record's machine block** gains `kernel_release` (`platform.release()`, what `uname -r` prints) and a `power` block:
   each supply under `/sys/class/power_supply` (type, online), the Windows active scheme GUID and the AC and DC overlay GUIDs as read,
   the overlay's name from the documented table above (anything else `unknown`), and how each was read. No key may contain a
   `FORBIDDEN_KEY_TOKENS` token.
2. **The run REFUSES before the token** unless a `Mains` supply is online AND the AC overlay is `ded574b5-…` (Best Performance) — a
   subcommand the driver calls among its pre-token checks; a source that cannot be read refuses.
3. **`run_all` re-checks at the opening canary and at the closing one**; a regime that is not mains + Best Performance at the close makes
   the run FAILED. **The builder refuses a record** whose `power` block is not mains + Best Performance.
4. **The pre-flight stays pinned and is NOT re-run:** its only products are the timeouts (at the 120 s floor, 3.5× its slowest process)
   and an estimate, both conservative under a faster mode; B0.2's smoke is the independent bound.

## B2 — Where the model ran: evidence in the record (CM1, CM2)
The record's `device` is the CLI argument; nothing observed confirms it. Required: right after the last episode — **before**
`_machine_block()` or anything else touches CUDA — the row process records `cuda_initialized` (`torch.cuda.is_initialized()`) and, when
initialised, `cuda_max_memory_allocated`; **a CPU row refuses if CUDA was initialised; a CUDA row refuses unless memory was allocated.**
The killing tests run the REAL path: the existing gated `hz1x1.bc` CPU row test (CM2 dies there) and a new gated real-row test of
`hz1x1.dt_k20` on CUDA that skips naming CUDA when none is present (CM1 dies there: with the synchronize wired, CUDA initialises and
allocates nothing).

## B3 — The CUDA synchronize on the real path (CM13; R1 MAJOR 10a)
A test drives `run_row(..., "cuda", env_builder=…, factory_builder=…)` with `configure_regime` and `_cuda_sync` monkeypatched (no GPU
needed) and asserts the spy fires immediately before both clock readings of every decision. CM13 must die.

## B4 — The builder's latency verification (CM4, CM11; R2 MAJOR-1…4 and its minors)
1. **An independent route:** the builder recomputes `n_timed`, the median and the nearest-rank p95 from each record's nanoseconds by
   its OWN code (sorted values; rank `ceil(0.95 n)`; the even-count median as the mean of the two middle values), never through
   `compute_latency.latency_stats`; and it derives **every published millisecond figure from those verified nanoseconds** — `median_ms`,
   `p95_ms` and both per-intersection figures (÷ the scenario's intersection count from the registry) — never from the record's own ms
   fields (or it verifies each of them equal). CM4 must die.
2. **Refuse, never skip,** with one test each, red first: a record of another format version (select records by the registry's expected
   file names, not by the `*_c*.json` glob that also matches `canary_close.json`); a missing (row, device) cell of the registry; an
   unknown row id; a row or device that disagrees with the file name; a checkpoint other than the registered one (CM11 must die);
   `warmup` ≠ 20, `draws` ≠ (1000, 1001, 1002), episodes ≠ 3, `engine_seed` ≠ 1000; a record whose `git.commit` is not the run's or
   whose tree was dirty; a record outside B1's power regime.
3. **Join the registry to the table row** (R2 MAJOR-4): the latency row's checkpoint path and sha256 must equal the table row's seed-101
   checkpoint of its representative tier as `row_checkpoints` verifies it, and route A hashes the file it loads. Test: a registry whose
   `h4.k1` names the K = 20 file refuses (K does not change the size, so route A = route B cannot catch it).
4. `_hardware` reports BOTH devices' regimes; `write_artifact` is atomic as well as exclusive; a JSON `null` present in a record is a
   value only where `DECLARED_ABSENCES` declares it (steps, batch, data, regime alike); a non-heuristic row without training entries
   refuses; an outlier note must be non-empty; the docstring's "every number is `{value, source}`" is made true or corrected to what is
   true (derived statistics sit beside the sourced per-seed values they come from).

## B5 — The row process (CM5, CM6; R1 MINOR 4a)
1. A test that `_verified_checkpoint` refuses a file at another digest (CM5 must die).
2. A test that every child command line the run builds carries `-P` (`_commands`) (CM6 must die).
3. **The measured process refuses an active tracer or profiler** — `sys.gettrace()`, `sys.getprofile()`, any `sys.monitoring` tool in
   use, `tracemalloc.is_tracing()`, `sys.flags.dev_mode` — and the driver refuses before the token if any of `COVERAGE_PROCESS_START`,
   `COVERAGE_PROCESS_CONFIG`, `PYTHONTRACEMALLOC`, `PYTHONDEVMODE`, `PYTHONMALLOC`, `PYTHONPROFILEIMPORTTIME` is set (the venv's
   coverage hook would otherwise trace every child and inflate every median, and the 2.0 s canary would not see a 2× slowdown).

## B6 — Text the table would print (R3; every number in a note is a number of the table)
1. `_P4_SUSPEND` says "202.4"; the record's minimum is **202.3** (`docs/data/p4_training.json $.seeds[2].seconds` = 202.348…). Fix it, and
   re-read EVERY number quoted in a declared note from its record — the packet lists each with its JSON path.
2. **MAPPO's worker count is an inference, not a measurement:** `kind` `inferred`, value 6, with its basis (the training run's file
   mtimes, plan V7, at the plan's stated confidence) and "not recorded by the manifest"; "16 logical CPUs" removed unless a record of the
   2026-08-06 run holds it.
3. `hz1x1.random` gains its C3 claim (P7.3a's zero-shot arm `random`, plan p8.2.md:173).
4. Units that cannot be misread: grid4x4 BC and IQL data as **per-intersection** windows / transitions beside the DT's joint windows (no
   reader may conclude BC saw 16× the data); MAPPO's batch labelled the PPO `minibatch_size`.
5. The module docstring's "every record of plan §4" states what is pinned, what is not, and why; `expected_duration`'s "upper bound"
   becomes "an estimate" with its premise, and names the pre-flight row as `dt_nomix_h4` (R1 MINOR 9a).
6. Optional (R3 NOTEs, not required): claims that point at a file-level key (`$.format_version`, `$.h3`) point at the row's own result
   where one exists.

## B7 — `what_this_does_not_say` gains
*"The CPU is a hybrid-core laptop part (Intel Core Ultra 9 275HX) running Linux under WSL2: the Windows host schedules the guest's
virtual CPUs onto performance or efficiency cores and the guest cannot pin them, so the single-thread figure is this machine's in the
recorded regime (mains, Windows power mode Best Performance), not a property of one core type."*

## B8 — Accepted as proposed
Q-G1-1 (the pre-flight and its 120 s timeouts; the pinned record stays) · Q-G1-2 (the outlier rule — it is what makes the stalls
checkable; R3 confirmed the five and that they are the only seeds above 2×) · Q-G1-3 (the five added sources; R3 confirmed each reason)
· Q-G1-4 superseded by B1 (the power source is visible and is now checked) · the plan's dated erratum (36 representatives, not 40).

## B9 — Next
**C4** (the tests for B1–B6, red for their own reasons — the seven survivors' killing tests among them) · **C5** (the fixes, green) — at
most two source files per commit, the commits named in the packet · the whole suite · the three test files once in a depth-1 clone (F.1's
command) · the seven survivors re-run against the new code (CM1, CM2, CM4, CM5, CM6, CM11, CM13; same semantics, their target lines
adapted), each committed in a throwaway worktree and pasted KILLED, plus a mutant per new refusal of B1, B4.2 and B5.3 · the interim
packet updated → **"P8.2 C4–C5 done"** → **gate G1.1**: the coordinator re-runs all thirteen and checks B1–B7, then Amendment B.1 names
the run worktree's commit, and only then does the author set Best Performance on mains and create the token.
