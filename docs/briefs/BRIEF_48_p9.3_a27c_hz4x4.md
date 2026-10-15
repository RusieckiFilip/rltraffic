# BRIEF_48 — P9.3 (A27-C): hangzhou_4x4_gudang — teachers, corpus and ladder on a real-demand 4×4, conditional, hard stop 2026-10-23 (A27 point 8)

**Mode:** the LEAN PROTOCOL (`PROJECT_PLAN` §7, 2026-10-15): plan and build in one go (`docs/plans/p9.3.md` committed, then
proceed). Tests first. Mutation testing for number-computing code. One session, one branch, in parallel with P9.1 and P9.2.
**Branch:** `task/p9.3-a27c` from `main`, worktree `/home/filip/rltraffic-p93`.
**Interpreter:** `/home/filip/rltraffic/.venv/bin/python` by absolute path.
**Roots (the MAIN tree's):** draws `/home/filip/rltraffic/scenarios/draws` (`--out-root` there; `DEFERRED` 55), corpus
`/home/filip/rltraffic/datasets_v11` (you create only `cf_hz4x4__*` directories), output `/home/filip/rltraffic/output` (you create
only `output/p9_3/`, `output/p9_3_runs/`, `output/checkpoints/p9_3_mappo_nominal_1000/`). `--torch-threads 1` everywhere.
**Registration:** `PREREGISTRATION.md` A27 (i) (and (b)–(d) for the recipes), tag `v2.8-prereg-a27`. **GATE 0 — the pre-flight may
run before the tag (it produces no registered number); nothing else does.** The registration wins over this brief.
**Label (A27):** conditional generalizability extension registered after the grid results; no new claim. The paper says exactly: hz4x4 tests real
demand and real geometry on the grid's topology, not a new topology (32 nodes of which 16 signalised, 80 directed roads, the same
16-node / 48-internal-edge lattice as `grid4x4_roadnet_red.json`; 9 file phases per intersection against the grid's 16; 2,983 vehicles
against 1,473 — measured 2026-10-15).
**Hard stop:** not merged to `main` by 2026-10-23 → dropped; the branch stays; nothing of it is cited; the paper names it as a
limitation. Reductions, in order, declared in A27(i): the spatial DT (already out); the `mappo1000` tier if its teacher cannot be
trained within one day.
**Dependency:** Phase B and the `random_fresh` tier need P9.1's CODE (its M1 merge: `--iql-target-dropout`, `--bc-batch-windows`,
`--iql-batch-transitions`, `--random-rng per-episode`). Do everything else first; after M1, `git merge main`.

## 0. Frozen interface contracts
- `docs/CONTRACTS.md`; the frozen set of `CLAUDE.md` §1. `experiments/**/*.py` frozen; **a NEW config under `experiments/configs/`
  is allowed** (`experiments/configs/p9_3_mappo_nominal_1000_hz4x4.json`, modelled on `p2_1_mappo_nominal_1000.json`: the same
  `lr 0.0003`, `rollout_size 1024`, `minibatch_size 128`, `hidden_dim 128`, `train_episodes 1000`, `max_steps 360`, `delta_time 10`,
  `control_mode acyclic`, `global_reward_fn queue_length`, seeds 101–505, env `cf_hz4x4` on `../../configs/sim/cityflow4x4.json`).
- The grid's collection protocol is the contract for the corpus (read the manifests of `datasets_v11/cf_grid4x4__*`, `run_metadata`):
  `mappo1000` — 5 seeds × draws 1–40 × 1 episode from the nominal MAPPO@1000 checkpoints; `maxpressure` — draws 1–200 × 1;
  `fixedtime` — draws 1–200 × 1 at the tuned k; `random_fresh` — draws 1–200 × 1 with `--random-rng per-episode` — **collected ONLY with P9.1's fixed collector; if
  Phase A starts before that merge, this tier is collected last** (A27(i)); reward settings `--local-reward-fn queue_length --global-reward-weight 0.0`; `--base-seed 1000`.
- D5's tuning protocol for fixed-time (`docs/data/fixed_time_sweep/README.md`): k ∈ {1, 2, 3, 4, 6, 8, 10, 14} on draws 1–5,
  objective the registered primary (now `att_engine` where its gate passes, `att_ours` beside), the chosen k recorded BEFORE collection.
- A11's engine-semantics gate (`offline/engine_att_reference.py`, P8.4b G0): four criteria with observed values on hz4x4 before
  `att_engine` is primary there; on FAIL both definitions are reported and no single-definition claim is made.
- `offline/collect.py`, `offline/materialise_draws.py`, `offline/flow_randomizer.py` (it handles hangzhou_4x4's flow format —
  `flow_randomizer.py:36-47, 350`), `offline/tier_sweep.py`, `offline/spatial_mixing.py`, `offline/offline_baselines.py`,
  `offline/att_rederivation.py`: UNCHANGED, import only; a scenario-parametrised path that today assumes `cf_grid4x4` is a new
  function in your module, not an edit (if impossible, stop and say so in the plan).

## 1. Why this task exists
Every multi-intersection result stands on one synthetic grid. hz4x4 has real demand and real geometry on the same lattice, so it
tests realism, not topology. The platform already supports it (`configs/sim/cityflow4x4.json`; the thesis measured MAPPO unconverged
at 200 episodes there — the reason the teacher's feasibility is a pre-flight question).

## 2. Scope fence
- No spatial DT. No SUMO side (hz4x4 is audited and NOT a C3 target, A15). No new claim: the ladder is reported as a replication
  beside the grid's, labelled as A27(i) says.
- No change to the grid's or hz1x1's artifacts, registries or corpora. No MAPPO hyper-parameter search. No timing claim.
- Not more than ONE day of wall-clock for the MAPPO teachers; if the pre-flight projects more, the tier is dropped and the ladder is
  three tiers (A27(i)).

## 3. What the task produces — Phase A (now)
### 3.1 `docs/data/p9_3_preflight.json` (≤ 1 hour of running)
CityFlow episode seconds on hz4x4 (MaxPressure, draw 0 and draw 1, 360 steps); MAPPO seconds per training episode (20 episodes
with the config above, one seed); the projection for 1,000 episodes × 5 seeds in parallel; the go/no-go for the teacher by the
one-day rule; the number of CPU workers you will use. For comparison, the grid's records: MAPPO@1000 3,463 s per seed median
(`docs/data/p8_2_compute.json`), one MaxPressure episode 1.16 s (coordinator, 2026-10-15).
### 3.2 Draws
`offline.materialise_draws --env-config configs/sim/cityflow4x4.json --draws-range 0 201` and `--draws-range 1000 1100`, out-root
the main tree (scenario key `cityflow4x4`); the materialised flow is single-insertion like the source (2,983 entries, asserted).
### 3.3 `offline/hz4x4_ladder.py` (the plan may rename it)
`k-sweep` (D5, writes `docs/data/p9_3_fixed_time_sweep.json` before any collection) · `collect` (the four tiers by `offline.collect`
exactly as §0; refuses an existing directory) · `gate` (A11's four criteria via `engine_att_reference`, `docs/data/p9_3_engine_att_gate.json`)
· `declare` (tier specs for hz4x4 mirroring `tier_sweep.TIERS`'s fields, `ladder_att` and `mean_training_return` computed, the
exposure table of A27(b) for hz4x4's row counts) · Phase B's `train` / `evaluate` / `old-cells` / `report`.
### 3.4 The teachers
`experiments/run.py experiments/configs/p9_3_mappo_nominal_1000_hz4x4.json --workers 5 --no-plot` in tmux (one worker per seed; the
runner pins one torch thread per process — `docs/patches/runner_thread_pinning.patch`); checkpoints under
`output/checkpoints/p9_3_mappo_nominal_1000/`; the training curves and the registered convergence criterion reported as P2.1 reported
them (`RUNSPEC_01`), the label `MAPPO@1000` regardless (tiers are labelled by measured return, §5 of the registration).

## 4. Phase B (after P9.1's M1 merge)
`dt_nomix` (the no-mixing spatial-DT arm of `tier_sweep`, `JOINT_BATCH_SIZE 64`, K = 20, 40,000 steps, the declared prompt rule),
`bc_m` (1,024 windows), `bc_top10_m` and `bc_top10_perix_m` (102 windows), `iql_m` (20,480 transitions, `--iql-target-dropout off`),
five seeds, on every tier collected; evaluation on draws 1000–1099 × 5 seeds by both routes (`att_ours` under `==` between them;
`att_engine` and companions by the observer route); references on the held-out draws: `maxpressure`, `fixedtime` (tuned k),
`random` (registered scheme) and `random_fresh` (A27(d)'s reference), and the MAPPO teachers as `behaviour`. Artifact
`docs/data/p9_3_hz4x4.json`: A27's label; the gate's outcome and which definition is primary; per-episode records of every cell;
every training run's `seconds` and `gradient_steps` with the uncontrolled-clock label; the exposure table; leaders with per-seed
blocks under the registered rules; `what_this_is_not` (not a C3 target; not a new claim; a replication).

## 5. Tests (first; minimum)
Draw materialisation idempotence and refusal outside the main tree; the k-sweep's objective recomputed independently from the raw
episode arrays; the collector invocation records `random_rng_scheme = per-episode` for the random tier and the manifest check refuses
`per-draw` there; the gate's four criteria with mutants on each tolerance; the declaration's exposure arithmetic; the two
evaluation routes' agreement stop; report T-regress; mutants for every leader/verdict path.

## 6. Gates, in order (lean)
- **G0 — plan** committed; the pre-flight run and its JSON committed (no token needed: ≤ 1 h, writes under `output/p9_3/` only).
- **G1 — code** (tests, mutants, suite green, hygiene/English) → `docs/returns/P9.3-code.md` → one mandate at most.
- **G2 — pre-flight → token** for the teachers and the corpus: `docs/returns/P9.3-preflight.md` with the exact tmux commands
  (`p93-mappo`, then `p93-corpus`), the worker counts, the projected wall-clock, the k-sweep result, and the go/no-go. The coordinator
  relays; the run is cleared by an Amendment to this brief.
- **G3 — Phase A runs**; the gate; `git merge main` after P9.1's M1; **G3b — Phase B pre-flight → token** (`p93-train`, `p93-eval`; the
  GPU plan: Phase B's DT trainings — ≈ 1 h each on the grid — run after P9.1's trainings finish, or interleaved at the coordinator's
  instruction, never two DT trainings on the GPU at once).
- **G4 — artifact and packet** `docs/returns/P9.3.md` with the "for the paper" section.
- **G5 — one review, at most one fix round → merge by 2026-10-23 or the hard stop fires.**

## 7. Definition of Done
- [ ] Pre-flight JSON committed; draws 0–200 and 1000–1099 materialised in the main tree; k-sweep recorded before collection; the
      teachers trained (or the tier dropped by the one-day rule, stated); the four tiers collected by the grid's protocol with the
      fresh random generator; the A11 gate run with observed values.
- [ ] Phase B's five arms × five seeds on every tier, evaluated on 100 held-out draws, both definitions, references beside; the
      artifact with per-episode records, training seconds/steps (uncontrolled clock), A27's label and T-regress.
- [ ] No registered artifact changed; zero frozen-file changes; zero new dependencies; suite green; packets with the AI record.

## 8. Return Packets
`docs/returns/P9.3-code.md`, `docs/returns/P9.3-preflight.md`, `docs/returns/P9.3.md` by `docs/returns/TEMPLATE.md`.
