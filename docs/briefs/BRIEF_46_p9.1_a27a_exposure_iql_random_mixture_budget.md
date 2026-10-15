# BRIEF_46 — P9.1 (A27-A): grid exposure parity, IQL targets without dropout, a true random tier, the overlap mixture, and A8(b)'s budget ablation (A27 points 1, 2, 3, 4, 6)

**Mode:** the LEAN PROTOCOL of the author's ruling of 2026-10-15 (`PROJECT_PLAN` §7, 2026-10-15). Plan and build in one go: write
`docs/plans/p9.1.md` and commit it, then proceed — no stop at G0 unless this brief is ambiguous, and then the ambiguity goes into
the plan as a numbered open question and you stop only on that item. Tests first, always. Mutation testing for code that computes a
reported number; your own mutants suffice unless the review finds something. One session, one branch.
**Branch:** `task/p9.1-a27a` from `main`, in its own worktree: `/home/filip/rltraffic-p91`.
**Interpreter:** worktrees have no `.venv`; use `/home/filip/rltraffic/.venv/bin/python` (and `.../bin/pytest`) by absolute path.
**Roots (the MAIN tree's, by absolute path, every command):** corpus `/home/filip/rltraffic/datasets_v11`, draws
`/home/filip/rltraffic/scenarios/draws`, output `/home/filip/rltraffic/output`. You create ONLY `output/p9_1/` and
`output/p9_1_runs/` there. Nothing under `output/` is deleted, moved or overwritten. `--torch-threads 1` on every training and
evaluation (N2, `PROJECT_PLAN` §10 "inherited by every later task").
**Registration:** `PREREGISTRATION.md` A27 (tag `v2.8-prereg-a27`), sections (a)–(e) and (g). **GATE 0 — no command that produces a
number registered there runs until `git merge main && git tag -l v2.8-prereg-a27` prints the tag.** Code and tests may be written
before. Read A27 from `PREREGISTRATION.md` on `main`, not from this brief; where the two disagree, the registration wins and you say so
in the packet.
**Authorised by:** the author's ruling of 2026-10-15 and his addendum of the same day (`PROJECT_PLAN` §8, 2026-10-15).
**This task has TWO merges:** M1, the code (flags, registries, tests, mutants — no numbers), as early as possible, because P9.3's
Phase B waits on it; M2, the campaign's artifacts and packet.

## 0. Frozen interface contracts (read them from disk; do not infer)
- `docs/CONTRACTS.md` (C6 v1.1). The frozen set of `CLAUDE.md` §1 untouched — `algorithms/**`, `agent/base.py`, `agent/utils/utils.py`,
  `agent/MAPPOAgent.py`, `envs/**`, `experiments/**/*.py`, `scripts/**`, `.claude/**`. No new dependency (the D4RL re-run uses the
  venv P8.3 recorded, outside the repo — §3.6). No existing test changes (additions only). Registered artifacts under `docs/data/`
  are read, never edited.
- **Every existing module keeps its registered behaviour under default flags, and a test proves it:** `offline/offline_baselines.py`,
  `offline/method_tier_grid.py`, `offline/tier_sweep.py`, `offline/collect.py`, `offline/dt_gate.py`, `offline/iql_correction.py`,
  `offline/att_rederivation.py`, `offline/tier_statements.py`. Reuse their functions; never copy one with a change.
- **The token mechanism:** the campaign refuses to start without the author's token exactly as `offline/iql_correction.py`'s campaign
  does — read its mechanism from the source and reuse it; do not invent a second one.
- Arm and tier names are A27's: `iql_t`, `bc_m`, `iql_m`, `bc_top10_m`, `bc_top10_perix_m`, `random_fresh`, the mixture tier named
  after its pair (A27(e)), the 100,000-step arms suffixed `_100k`. No registered name is reused for a new cell.

## 1. Why this task exists (A27 (b)–(e), (g); the author's ruling, points 1–4 and 6)
1. **Exposure (point 1 — control added after results).** On the grid the DTs see every training row 711.1 times, BC and IQL 44.4,
   the top-decile arms 444.4 (A27(b) defines exposure and derives the numbers from `tier_sweep.JOINT_BATCH_SIZE = 64`,
   `CONTEXT_LENGTH = 20`, `offline_baselines.BC_BATCH_WINDOWS = 64`, `IQL_BATCH_TRANSITIONS = 1_280`, 40,000 steps, and the tiers'
   1,152,000 / 115,200 rows). Every grid DT-versus-baseline contrast is confounded by that gap.
2. **IQL targets (point 2 — bug fix).** `offline_baselines.py:1981` puts `policy`, `q`, `value` in `train()` and never leaves it;
   `q_target` is never switched; `MLPTrunk` has `dropout = 0.1` (`agent/OfflineBaselines.py`). So `q_taken = q_target(s)` (`:1990`)
   and `next_value = value(s')` (`:2008`) carry dropout noise into the TD targets.
3. **Random tier (point 3 — bug fix in data generation).** `collect.py:981` re-creates the random policy's RNG from `--base-seed` per
   draw: the hz1x1 random tier holds 2 action schedules over 400 episodes (200/200), the training subsample 116/84 (measured
   2026-10-15, `PROJECT_PLAN` §8).
4. **Overlap mixture (point 4 — test registered after results).** The registered mixtures' top decile is all-expert by construction
   (`DEFERRED` 46). A27(e) fixes the pair by the author's ruling after the coordinator's measurement that `mappo1000 + mappo200`
   is degenerate too (0 of 300 compositions put a `mappo200` stream in the decile).
5. **Budget (point 6 — test registered after results, A8(b)).** Registered 2026-08-12, never run; 127 of 160 hz1x1 runs had not
   met the plateau criterion at 40,000 steps (`plateaued` flags, measured 2026-10-15).

## 2. Scope fence — what NOT to build
- No DT is re-trained for point 1; no spatial DT anywhere in this task; no change to any DT recipe, prompt rule or K.
- No hz1x1 exposure arms (A27(b): hz1x1's asymmetry is stated in the paper, not re-run).
- No second IQL variant (the advantage's Q term follows the target flag — A27(c) says so; do not add an arm that keeps it).
- No mixture other than A27(e)'s pair; no retuning of anything; no change to the registered mixtures or random tiers.
- No edit to `algorithms/max_pressure.py`, to any `docs/data/p4_*`, `p5_*`, `p8_*` artifact, to `docs/returns/P5.2*.md`, or to
  `README.md` (the coordinator updates it at merge if a sentence becomes false).
- No SUMO, no hz4x4 (P9.2, P9.3). No timing claim: `seconds` are recorded and labelled an uncontrolled clock, nothing more.
- No new statistics beyond the registered rules: the paper track computes its own (hierarchical bootstrap, IQM, Holm) from the
  per-episode records you commit.

## 3. What the task produces
### 3.1 `offline/offline_baselines.py` — the IQL target flag (commit C1, with its tests)
`train_iql(..., target_dropout: str = "active")`. Under `"off"`: every forward of `q_target` and the forward of `value` on
`batch.next_state` run with the module in `eval()` inside the existing `torch.no_grad()` blocks, and the module is returned to
`train()` immediately after; the forward of `value` on `batch.state` (the expectile regression's prediction), the online `q`, the
`policy`, every hyper-parameter and the warm-up are untouched. Record `target_dropout` in the checkpoint provenance and in the
training record's diagnostics. Under `"active"` the code path is the registered one: prove it by regenerating a registered IQL
checkpoint tensor-for-tensor (pick `output/p4_4/checkpoints/iql_seed101.pt`'s recipe from its provenance; CUDA, one torch thread, as
P4.4 ran it; if equality fails, say so with the first differing tensor — do not loosen the test to `allclose`). Expose the flag on
every CLI that trains IQL (`method_tier_grid train`, `tier_sweep train-baselines`, `iql_correction train` if it is the route you
reuse for the grid random tier) as `--iql-target-dropout {active,off}`.
**Also `train_bc` / `train_iql` batch sizes:** both already take `batch_size`; expose `--bc-batch-windows` (default 64) and
`--iql-batch-transitions` (default 1280) on the same CLIs, recorded in provenance. Arm names carry the suffix (`_m`) so no registered
checkpoint path or cell name can be overwritten: refuse to write a checkpoint whose path exists.

### 3.2 `offline/method_tier_grid.py` and `offline/tier_sweep.py` — registries and wiring (C2)
- hz1x1 `TIERS` gains `random_fresh` (dirs `("cf_hz1x1__random_fresh",)`, `subsample="none"`, `stream_count=200`) and the mixture
  tier of A27(e) (components per the registered pair, `subsample="mixture"`, composed by the SAME rule, RNG and seed as `mix50`;
  quote the rule from the source in the plan). Their `target_rtg` / `rtg_scale` are the declared rule's values, re-derived by
  `assert_declaration_matches_corpus` before training, exactly as the mixtures were (`method_tier_grid.py:252-263`'s comment is the
  history to read first).
- grid `TIERS` gains `random_fresh` (dirs `("cf_grid4x4__random_fresh",)`, `subsample="none"`, `episode_count=200`; `ladder_att` and
  `mean_training_return` computed and recorded by the declaration step, not typed).
- The `--tier` enums of `train`/`evaluate` accept the new tiers; `--steps 100000` is accepted for the `_100k` arms and the declared
  value is recorded in every artifact that pins `declared_gradient_steps` (the pins for registered arms stay at 40,000 — a test asserts
  a registered arm name with 100,000 steps is REFUSED).

### 3.3 `offline/collect.py` — the random generator flag (C3)
`--random-rng {per-draw,per-episode}`, default `per-draw`. `per-episode`: `np.random.default_rng([base_seed, draw_id, episode_index])`
built for every episode (the draw loop at `collect.py:965-1000` is where the per-draw generator is created today — `:981`). Write
`random_rng_scheme` into `run_metadata`. **Load-bearing test:** under the default, re-collect ONE registered random episode
(`cf_hz1x1__random/ep000000_seed1000_draw1.npz`, draw 1, engine seed 1000) into `tmp_path` and compare the `.npz` arrays byte for
byte with the corpus file; CityFlow is deterministic on hz1x1, so any difference is a regression. Under `per-episode`, two episodes of
one draw must differ and two runs of the same (draw, episode) must be identical.

### 3.4 The campaign module `offline/corrective_round.py` (C4–C6; the plan may rename it, with the reason)
Subcommands, each writing only into `output/p9_1/` / `output/p9_1_runs/` or the one artifact it names, after every input check:
- `declare` — for `random_fresh` (both scenarios) and the mixture: the declaration as P4.6/P5.2 write it, plus the EXPOSURE TABLE
  computed from the code constants (A27(b)'s numbers must come out: 711.1 / 44.4 / 44.4 / 444.4 and, matched, 711.1 / 711.1 / 708.3 /
  708.3 — a test pins them) and, for the mixture, the kept decile's SOURCE COMPOSITION and the pair's top-decile statistic (A27(e)
  P4a is decided here, before training). Refuse a `random_fresh` tier that does not hold 200 distinct per-intersection action sequences.
- `collect-random-fresh` — the two tiers by `offline.collect` with `--random-rng per-episode`, draws 1–200, `--episodes 1`,
  `--base-seed 1000`, the registered reward settings (`--local-reward-fn queue_length --global-reward-weight 0.0`), into
  `datasets_v11/cf_hz1x1__random_fresh` and `cf_grid4x4__random_fresh` (new directories; refuse if they exist).
- `train` — every arm of A27 (b), (c), (d), (e), (g) by the existing training functions with the flags above: hz1x1 `iql_t` on 8
  tiers; `random_fresh` arms (`dt`, `bc`, `bc_top10`, `iql`, `iql_t`); mixture arms (`dt`, `bc`, `bc_top10`, `iql`, `iql_t`); budget
  arms (`dt_100k`, `bc_100k`, `bc_top10_100k`, `iql_100k`, `iql_t_100k`) on `mappo1000` and `mix50`; grid `bc_m`, `bc_top10_m`,
  `bc_top10_perix_m`, `iql_m` on 4 tiers, `iql_t` on 4 tiers, `random_fresh` arms (`dt_nomix`, `bc_m`, `bc_top10_m`,
  `bc_top10_perix_m`, `iql_m`). Five seeds each. Resumable: a checkpoint on disk with a matching provenance is skipped, never
  re-trained; status is COMPUTED from disk.
- `evaluate` — every new cell on draws 1000–1099 × 5 seeds by BOTH routes P5.2b used: the P5.2/P4.6 route for `att_ours`
  (`method_tier_grid evaluate` / `tier_sweep evaluate`) and P8.4b's route for `att_engine` and the companions
  (`offline/att_rederivation.py`'s per-second observer, as `iql_correction evaluate-p8-4b` drives it). Per cell the two routes'
  `att_ours` must agree under `==`; a disagreement stops the campaign.
- `old-cells` — for every registered cell a new cell is compared with (A27 (b)–(e), (g)): copy its per-episode records (both
  definitions, companions) from the committed artifact that holds them; where the committed record lacks `att_engine` (every
  `docs/data/p4_*` / `p5_2_*` cell P8.4b did not re-derive; the old IQL cells that exist only under `output/`), re-roll the cell through
  the P8.4b route and accept it only if its `att_ours` reproduces the committed value under `==` (`att_rederivation`'s
  `reproduces_committed`). Also copy each compared training run's `seconds` and `gradient_steps` from its committed training record
  or checkpoint provenance, naming the source path. Nothing is recomputed from a description.
- `rescore` — A27(b): every P5.2 statement contrasting a DT with a baseline, re-scored with the matched arm via
  `offline/tier_statements.py`'s scorers (import; do not re-implement), original beside, both definitions, verdict `moved: true/false`
  by the registered rule; A27(c): the IQL statements of P4.4, P4.6, P4.7, P5.2 (P5.2b's random cell as the registered one) with
  `iql_t`; A27(d): the collapse checks against BOTH random references (registered `random` anchor and `random_fresh` reference);
  A27(e): P4a/P4b/P4c; A27(g): the paired 100k − 40k differences and the ordering concordance.
- `report` — the artifacts of §3.5, pure functions of their inputs (sorted keys, no rounding, no timestamp, git commit recorded, dirty
  must be false), each with a T-regress test that rebuilds it byte for byte from the run directories.
- `preflight`, `status`, and the token check.

### 3.5 Artifacts (all under `docs/data/`, format version in each, A27's label in each)
`p9_1_exposure_parity.json` · `p9_1_iql_targets.json` (with a `d4rl` section, §3.6) · `p9_1_random_fresh.json` · `p9_1_mixture.json` ·
`p9_1_budget.json`. Each carries: `label` (A27's, verbatim), `registered_in` ("A27(x)"), inputs by digest, the declarations, **the
per-episode records of every new cell and every compared old cell** (`arm`, `tier`, `seed`, `draw_id`, `att_ours`, `att_engine`,
`entered`, `created`, `never_entered`, `episode_reward`, `horizon_vehicle_count`, `source` = committed path or "re-rolled, reproduces
committed att_ours"), **every training run's `seconds` and `gradient_steps` with `"clock": "uncontrolled; no Best Performance regime,
no canary; not a benchmark"`**, the registered-rule verdicts with per-seed blocks, the predictions' verdicts, and `what_this_is_not`.

### 3.6 P8.3's D4RL calibration, re-run with the fix (part of point 2, not optional — the author's addendum, item 5)
`calibration/d4rl_run.py` runs in `/home/filip/d4rl-calib-venv` (verified 2026-10-15: imports `mujoco` 3.11.0, `gymnasium` 1.3.0;
dataset `/home/filip/d4rl-calib-data/halfcheetah_medium_expert-v2.hdf5`, sha256 `015db75d…`). Read whether it calls
`offline_baselines.train_iql` or carries its own loop; the flag must reach whichever computes the targets, and the plan says which.
Re-run the IQL arm only, five seeds, `--iql-target-dropout off`, the protocol of `docs/plans/p8.3.md` unchanged (10 evaluation
episodes per seed), outputs under `/home/filip/d4rl-calib-out/p9_1/`; the per-seed returns of the new arm AND of the registered three
arms (from `docs/data/p8_3_d4rl_calibration.json`) go into `p9_1_iql_targets.json`'s `d4rl` section with the `bc − iql` contrast
recomputed beside the registered −542.05 [−1668, +584]. Expected cost ≈ 100 s of training per seed (P8.3's record); if the pre-flight
measures more than one hour in total, stop and say so — do not skip silently.

## 4. Tests (written first; the list is the minimum)
1. `tests/test_iql_target_dropout.py` — **load-bearing:** a forward hook (or a recording subclass) asserts `q_target.training is
   False` and `value.training is False` during the target forwards under `"off"` and `True` under `"active"`; `value.training is True`
   during the `state` forward under both; and the provenance records the flag. **Mutation you run and paste:** replace the `eval()`
   calls with `train()` in the implementation — the test must FAIL.
2. Registered-behaviour tests: the IQL checkpoint regenerated tensor-for-tensor under defaults (§3.1); the random episode regenerated
   byte for byte under `per-draw` (§3.3); a `bc` checkpoint regenerated under `--bc-batch-windows 64`.
3. `tests/test_exposure_table.py` — the declaration's exposure arithmetic from the constants equals A27(b)'s eight numbers.
4. Registry tests: the new tiers declare and re-derive; a registered arm name at 100,000 steps is refused; a checkpoint path that
   exists is refused; the mixture's decile composition is computed by an independent route (raw `.npz` sums, as
   `tests/test_mixture_tiers.py` does) and equals the declaration's.
5. `tests/test_corrective_round.py` — the two evaluation routes' `att_ours` agreement check stops on a disagreement; `old-cells`
   refuses a re-rolled cell whose `att_ours` does not reproduce; `rescore` reproduces P5.2c's verdict for one unchanged statement
   (a tautology guard: the matched arm replaced by the original arm must give `moved: false`); T-regress for every artifact.
6. Mutants for every scorer path you add (the `moved` decision, the concordance, the P4a/P4b/P4c rules, the collapse counts): list
   them, run them, paste the kill table.

## 5. Gates, in order (lean)
- **G0 — plan.** `docs/plans/p9.1.md` committed on the branch: the exposure arithmetic from the code, the mixture rule quoted from
  the source, the D4RL route, the commit list, the campaign's cell count and your timing estimate from the registered records
  (medians, hz1x1: dt 204 s, bc 110 s, bc_top10 120 s, iql 361 s per run at the registered batches; grid: dt_nomix 3,487 s, iql
  298 s; `docs/data/p8_2_compute.json`) with the matched batches' cost measured on ONE run each. No stop.
- **G1 — code merge M1.** Commits C1–C3 (and C4's registries) with tests and mutants, the whole suite run, `scripts/check_test_hygiene.sh`
  and `scripts/check_english.sh` clean → packet `docs/returns/P9.1-code.md` → the coordinator reviews (at most one mandate) → merges
  to `main`. P9.3 Phase B starts from this merge. **You continue on the branch meanwhile; after the merge, `git merge main`.**
- **G2 — pre-flight, then the token.** `docs/returns/P9.1-preflight.md`: the cell count by arm, the measured cost of one matched BC
  run and one matched IQL run on the grid, one registered cell reproduced end-to-end through your `evaluate` (both routes, `==`),
  the `declare` output for the mixture (P4a decided), the exact campaign command (ONE tmux session, training serial, evaluation
  with `--workers ≤ 6`), the resume behaviour, and the GPU plan of `PROJECT_PLAN` §10 (2026-10-15). The coordinator relays the
  command and the token request to the author in one message. The campaign is cleared when this brief carries an Amendment saying
  so; until then build the report's tests.
- **G3 — the campaign** under the author's token, chunked, resumable, status from disk; the D4RL re-run in its venv.
- **G4 — artifacts and packet.** `report` from a clean committed tree; `docs/returns/P9.1.md` by `docs/returns/TEMPLATE.md`, with the
  real pytest tail, `git diff --stat`, the kill tables, the AI-assistance record, and a "for the paper" section: the new numbers
  beside the old, which registered verdicts moved, which draft sentence changes (as the coordinator's note will state it).
- **G5 — one review, at most one fix round** (only findings that change a reported number or a claim; the rest to `DEFERRED`) → merge
  M2 → the coordinator hands the author the CI ceiling command.

## 6. Definition of Done
- [ ] A27(c): `q_target` and `value(s')` run in `eval()` under the flag; the test of §4.1 exists and its mutant fails; every IQL cell
      (hz1x1 8 tiers, grid 4 tiers, P8.3) has a before and an after in the repository, per-episode (per-seed for D4RL).
- [ ] A27(b): `bc_m`, `bc_top10_m`, `bc_top10_perix_m`, `iql_m` exist on every grid tier beside the originals; the exposure table is
      in the artifact and equals A27(b).
- [ ] A27(d): two `random_fresh` tiers with 200 distinct schedules each; the hz1x1 and grid arms trained and evaluated; both random
      references on the held-out draws; P3a/P3b with verdicts.
- [ ] A27(e): the mixture's decile composition declared before training; five arms × five seeds; P4a/P4b/P4c with verdicts.
- [ ] A27(g): the ten `_100k` arms on `mappo1000` and `mix50`; the paired differences and the concordance.
- [ ] Every artifact carries the per-episode records (new and compared-old, both definitions, companions), the training seconds and
      steps with the uncontrolled-clock label, A27's label, and a T-regress. No registered artifact or checkpoint changed
      (`git diff --stat` and `sha256sum` of `output/SHA256SUMS_*.txt` entries you touched nothing of).
- [ ] Zero frozen-file changes; zero new dependencies; suite green; hygiene and English checks clean; packets written.

## 7. Return Packets
`docs/returns/P9.1-code.md` (M1) and `docs/returns/P9.1.md` (M2), by `docs/returns/TEMPLATE.md`, each with the AI-assistance record
(`CLAUDE.md` §8). State which A27 section each number answers and which amendments of this brief the packet was written against.
