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
1. **Exposure (point 1 — control added after results).** Exposure = row visits = steps × decisions per step / decisions in the
   training set (A27(b)): on the grid the DTs 711.1, BC and IQL 44.4, the top-decile arms 444.4; the rounding of %BC's 102.4 windows
   to 102 (708.3 visits) is declared in A27(b). The numbers come from `tier_sweep.JOINT_BATCH_SIZE = 64`,
   `CONTEXT_LENGTH = 20`, `offline_baselines.BC_BATCH_WINDOWS = 64`, `IQL_BATCH_TRANSITIONS = 1_280`, 40,000 steps, and the tiers'
   1,152,000 / 115,200 rows). Every grid DT-versus-baseline contrast is confounded by that gap.
2. **IQL targets (point 2 — bug fix, results before and after).** `offline_baselines.py:1981` puts `policy`, `q`, `value` in `train()` and never leaves it;
   `q_target` is never switched; `MLPTrunk` has `dropout = 0.1` (`agent/OfflineBaselines.py`). So `q_taken = q_target(s)` (`:1990`)
   and `next_value = value(s')` (`:2008`) carry dropout noise into the TD targets.
3. **Random tier (point 3 — data-collection bug fix, results before and after).** `collect.py:981` re-creates the random policy's RNG from `--base-seed` per
   draw: the hz1x1 random tier holds 2 action schedules over 400 episodes (200/200), the training subsample 116/84 (measured
   2026-10-15, `PROJECT_PLAN` §8). The grid's random tier has the same defect: 2 joint schedules over its 400 episodes (200/200) and
   2 per intersection on all 16. On both scenarios the two logged schedules are `default_rng(1000)`'s first and second 360-step
   blocks (every legal-action mask is all-true), none of the reference seeds 101–505 regenerates either, and P5.3b's attractor
   `47c0ff33…` is the logged engine-seed-1001 schedule itself (all measured 2026-10-15).
4. **Overlap mixture (point 4 — test registered after results).** The registered mixtures' top decile is all-expert by construction
   (`DEFERRED` 46). A27(e) replaces the ruling's pair by `mappo1000 + mappo500` (expert fraction 0.50) under the author's delegation, after the
   registered functions showed `mappo1000 + mappo200` all-expert at every proportion down to 10 %.
5. **Budget (point 6 — A8(b)'s secondary, registered 2026-08-12 after P4.4's primary results and executed after all primary
   results; the `mix50` part beyond A8(b)).** Registered 2026-08-12, never run; 127 of 160 hz1x1 runs had not met the plateau
   criterion at 40,000 steps (`plateaued` flags, measured 2026-10-15), and of those 127, 111 were still falling over the last three
   `window_means` — the draft's count; A27(g) carries the author's one-line command, which prints `160 127 111 11 5`.

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
`train_iql(..., target_dropout: str = "active")`. **Three quantities carry dropout noise in the registered loop** (A27(c)): the
expectile target `q_target(s, a)` (`:1991`), the TD target `value(s')` (`:2008`), and `value(s)` inside the AWR advantage
(`v_pred.detach()`, `:2022`); nothing ever calls `q_target.train()` — it is in train mode because that is `nn.Module`'s state after
construction. Under `"off"` those three forwards run in `eval()` under `no_grad` — `q_target(s)`, `value(s')`, and the advantage's
`value(s)` as a SEPARATE forward computed BEFORE `optimisers["v"].step()` (so the registered update order is kept: the advantage uses
V from before its update in the same step; published IQL uses V after — not changed) — and each module returns to `train()`
immediately after. The trained forwards — `policy(s)`, `q(s)`, and `value(s)` for the expectile loss — keep dropout 0.1 in train mode.
Under `"active"` the advantage uses `v_pred.detach()` exactly as today (no extra forward): the registered path, byte for byte. Every
hyper-parameter and the warm-up untouched. Record `target_dropout` in the checkpoint provenance and in the
training record's diagnostics. Because `q_target` and `value` are locals of `train_iql`, the test of §4.1 needs forward hooks that
record `module.training` at each forward (or a small refactor exposing the modules to a test seam — say which in the plan). Under `"active"` the code path is the registered one: prove it by regenerating a registered IQL
checkpoint tensor-for-tensor (pick `output/p4_4/checkpoints/iql_seed101.pt`'s recipe from its provenance; CUDA, one torch thread, as
P4.4 ran it; if equality fails, say so with the first differing tensor — do not loosen the test to `allclose`). Expose the flag on
every CLI that trains IQL (`method_tier_grid train`, `tier_sweep train-baselines`, `iql_correction train` if it is the route you
reuse for the grid random tier) as `--iql-target-dropout {active,off}`.
**Also `train_bc` / `train_iql` batch sizes:** both already take `batch_size`; expose `--bc-batch-windows` (default 64) and
`--iql-batch-transitions` (default 1280) on the same CLIs, recorded in provenance. Arm names carry the suffix (`_m`) so no registered
checkpoint path or cell name can be overwritten: refuse to write a checkpoint whose path exists.

### 3.2 `offline/method_tier_grid.py` and `offline/tier_sweep.py` — registries and wiring (C2)
- hz1x1 `TIERS` gains `random_fresh` (dirs `("cf_hz1x1__random_fresh",)`, `subsample="none"`, `stream_count=200`; its state
  normalisation, target, scale and IQL reward scale are fitted on its own 200 episodes — A27(d) says the old tier's were fitted on
  400, and the artifact must say so too) and the mixture tier `mix_m1000_m500` of A27(e): components `mappo1000` (stronger) and
  `mappo500` (weaker, in the `random_pool` slot of `mixture_training_streams`), expert fraction 0.50, the SAME rule, RNG and seed as
  `mix50` (so its 100 `mappo1000` streams are `mix50`'s); the declaration must reproduce A27(e)'s design facts exactly — kept decile
  14 `mappo1000` (seeds 101: 5, 202: 6, 404: 1, 505: 2) + 6 `mappo500` (101: 3, 202: 1, 404: 2), decile returns [−6,627, −5,959] —
  and refuse to train if it does not. Their `target_rtg` / `rtg_scale` are the declared rule's values, re-derived by
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
  708.3 — a test pins them) and, for the mixture, the kept decile's SOURCE COMPOSITION by source and by behaviour seed, which must equal A27(e)'s
  design facts (a refusal otherwise). Refuse a `random_fresh` tier that does not hold 200 distinct per-intersection action sequences.
- `collect-random-fresh` — the two tiers by `offline.collect` with `--random-rng per-episode`, draws 1–200, `--episodes 1`,
  `--base-seed 1000`, the registered reward settings (`--local-reward-fn queue_length --global-reward-weight 0.0`), into
  `datasets_v11/cf_hz1x1__random_fresh` and `cf_grid4x4__random_fresh` (new directories; refuse if they exist).
- `train` — every arm of A27 (b), (c), (d), (e), (g) by the existing training functions with the flags above: hz1x1 `iql_t` on 8
  tiers; `random_fresh` arms (`dt`, `bc`, `bc_top10`, `iql`, `iql_t`); mixture arms (`dt`, `bc`, `bc_top10`, `iql`, `iql_t`); budget
  arms — the FOUR registered recipes only, one ceiling, as A8(b) requires: `dt_100k`, `bc_100k`, `bc_top10_100k`, `iql_100k` (IQL with
  its registered targets; no `iql_t` at 100,000) on `mappo1000` (A8(b)) and on `mix50` (beyond A8(b), labelled an exploratory
  extension in the artifact); grid `bc_m`, `bc_top10_m`,
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
  `iql_t`; A27(d): every "worse than random" and collapse check against BOTH random references (the registered FIVE-SCHEDULE anchor and
  the `random_fresh` reference), and A27(d)'s P3a as the author worded it — per (scenario, seed), the `action_sequence_sha256`
  (`offline/nortg_decomposition.py`) of every held-out episode, the joint array on hz1x1 and each intersection's column on the grid,
  with the number of distinct sequences, the share of draws carrying the most common one, identity with either random reference's
  episode on the same (seed, draw), and identity with any of the tier's 200 training schedules; P3a is FALSIFIED by any repeat (on the
  grid: any intersection's sequence equal to that intersection's sequence on another draw of the same seed); A27(e): the four ordering contrasts (`bc_top10` vs `bc`, `bc_top10` vs `iql`, `iql` vs `dt`, `dt` vs
  `bc`) and the "every arm within 5 s of `bc_top10`" check, each HELD / FAILED / NOT RESOLVED by P5.2c's rule under `att_engine`,
  `att_ours` beside; A27(g): the paired 100k − 40k differences and the ordering concordance, same rule.
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
1. `tests/test_iql_target_dropout.py` — **load-bearing:** forward hooks record `module.training` at every forward of `q_target`,
   `value`, `q` and `policy` during one real update step: under `"off"` the three target forwards (`q_target(s)`, `value(s')`, the
   advantage's `value(s)`) record `False` and every trained forward (`policy(s)`, `q(s)`, the expectile-loss `value(s)`) records
   `True`; under `"active"` every forward records `True`; the provenance records the flag; and the advantage's V is shown to come from
   the pre-update parameters under both modes (compare against a V forward taken after the step). **Mutation you run and paste:** replace the `eval()`
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
- [ ] A27(c): all three target forwards — `q_target(s, a)`, `value(s')` and the advantage's `value(s)` as a separate pre-update forward — run in `eval()` under the flag (the author's edit 1, 2026-10-15); the trained forwards keep dropout 0.1; the test of §4.1 exists and its mutant fails; every IQL cell
      (hz1x1 8 tiers, grid 4 tiers, P8.3) has a before and an after in the repository, per-episode (per-seed for D4RL).
- [ ] A27(b): `bc_m`, `bc_top10_m`, `bc_top10_perix_m`, `iql_m` exist on every grid tier beside the originals; the exposure table is
      in the artifact and equals A27(b).
- [ ] A27(d): two `random_fresh` tiers with 200 distinct schedules each; the hz1x1 and grid arms trained and evaluated; both random
      references on the held-out draws; P3a (no repeated sequence per seed; per intersection on the grid) and P3b with verdicts, and the
      beside-table (distinct sequences per seed, most-common share, reference identity, training-schedule replay) next to P5.3b's
      old-tier numbers (1 sequence, 100 % of draws).
- [ ] A27(e): `mix_m1000_m500` declared with A27(e)'s design facts reproduced; five arms × five seeds; the four ordering contrasts
      and the collapse check with HELD / FAILED / NOT RESOLVED verdicts under `att_engine`, `att_ours` beside.
- [ ] A27(g): the four registered recipes at 100,000 steps on `mappo1000` and on `mix50` (the latter labelled beyond A8(b)); the
      paired differences and the concordance.
- [ ] Every artifact carries the per-episode records (new and compared-old, both definitions, companions), the training seconds and
      steps with the uncontrolled-clock label, A27's label, and a T-regress. No registered artifact or checkpoint changed
      (`git diff --stat` and `sha256sum` of `output/SHA256SUMS_*.txt` entries you touched nothing of).
- [ ] Zero frozen-file changes; zero new dependencies; suite green; hygiene and English checks clean; packets written.

## 7. Return Packets
`docs/returns/P9.1-code.md` (M1) and `docs/returns/P9.1.md` (M2), by `docs/returns/TEMPLATE.md`, each with the AI-assistance record
(`CLAUDE.md` §8). State which A27 section each number answers and which amendments of this brief the packet was written against.

---

# ℹ️ AMENDMENT A — 2026-10-16 (no gate): a timing window you may have used was loaded by the coordinator, and the machine-load rule while P9.2's and P9.3's runs go

**A contaminated window.** Between about 11:28 and 12:10 on 2026-10-16 the coordinator's verification runs (P9.2's and P9.3's test
files and mutants, several of which hung) and two review mandates loaded the machine: the 1-minute load average reached **33.6 at
11:46** on 16 CPUs. Your `time_iqlm.py` was running in that window (started about 11:19, seen at 94 % CPU at 11:50). **No timing taken
in that window is quotable**, in the pre-flight or anywhere else: re-take it, with the canary before and after. The fault is the
coordinator's; the window is recorded in `PROJECT_PLAN` §8 (2026-10-16).

**From now until P9.2's and P9.3's Phase A runs have ended** (P9.3's teachers start today, about 52 minutes on 5 CPU workers, then its
corpus on 1; P9.2's shards, 6 workers for up to about 3.5 h, start after its C8):
1. Beside every timing you quote, record the 1-minute load average (`uptime`) at its start and at its end. **A timing whose load
   average exceeds 12 at either end is not quotable.**
2. At most one pytest process at a time in your session, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; no mutation campaign while a
   timing runs. Today's hung test processes waited on futexes at about 0 % CPU while about five test processes ran at once — the
   signature of an unpinned forked worker, which `experiments/runner.py:limit_torch_threads` documents.
3. CPU workers: §10's cap of 12 across the round still binds. Your GPU training stage is one process; a CPU evaluation stage of yours
   uses at most `12 − (workers running in P9.2 and P9.3)`, and states the number it used.

---

# ⚠️ AMENDMENT B — 2026-10-16, gates G1 (M1, reviewed late) and G2: M1 is FIX FIRST, small (branch `task/p9.1-m1`); the `collect` stage is CLEARED now; the `campaign` stage waits for the M1 merge, Q2's typed commit, Q8's cells and the stall remedy (Amendment C clears it); Q1, Q2, Q6, Q7, Q8 ruled; the author ruled Q10

Written against `docs/returns/P9.1-code.md` (M1, `784893d`) and `docs/returns/P9.1-preflight.md` (`140149e`), read from disk, and
against one review mandate's findings (`review_p91_m1`, read-only, run on `784893d`). Merge `main` to read it.

## B.1 M1 — the review: no blocker; one MAJOR; FIX FIRST on a branch of its own
**Verified by the reviewer, with no defect:** under the default flags `train_iql` makes no extra forward and keeps the forward
order, and an independent recomputation gives the same fixture digest on `main` (`9503282`) and on `784893d` (`de85ece6…`); under
`"off"` exactly A27(c)'s three target forwards run in `eval()` under `no_grad`, the advantage's V is a separate forward taken before
the V step, every module returns to `train()` in a `finally`, and the trained forwards keep dropout; `collect.py`'s default branch is
the old two lines verbatim, `per-episode` seeds `default_rng([base_seed, draw_id, index])` and the manifest records the scheme; a
sha256 over both modules' `TIERS`, `BEHAVIOUR_REFERENCE_BY_TIER`, `MIXTURE_EXPERT_FRACTION` and `METHODS` is identical on `main`
and `784893d`; the matched batches (1,024 / 102 / 20,480) and A27(b)'s exposure numbers; `--compare-with` reads the committed D4RL
artifact. The seven new test files: 144 passed, 2 skipped (the two CUDA regenerations; the GPU is closed to the reviewer's sandbox).
**The fix round (the lean protocol: only findings that change a reported number or a claim):**
1. **MAJOR — a registered recipe is not refused on the grid's `random_fresh`.** `tier_sweep.assert_a27_baselines_run` (and
   `baselines_plan`) skip the check whenever `tier in A27_TIERS`, so `train-baselines --tier random_fresh` on the GRID trains `bc` at
   64 windows, `bc_top10`, `bc_top10_perix` and `iql` at 1,280 transitions with dropout active — under their registered names — and
   with `--iql-target-dropout off` also `iql_t`. A27(d) declares exactly `dt_nomix`, `bc_m`, `bc_top10_m`, `bc_top10_perix_m`,
   `iql_m` for the grid (and `dt`, `bc`, `bc_top10`, `iql`, `iql_t` for hz1x1). The campaign's own cell list trains exactly those
   (your pre-flight's table), so no planned number is wrong — but the refusal is the guarantee, and an unmatched, dropout-active grid
   cell could otherwise be scored as the grid's `random_fresh` baseline. **Fix:** refuse, on each scenario's `random_fresh`, every
   recipe outside A27(d)'s list for that scenario; **`tests/test_tier_sweep_a27.py:186-203` asserts the defect and is CORRECTED** —
   this amendment authorises that edit in writing: its docstring credits A27(d) with hz1x1's list applied to the grid, so the test is
   wrong, not the refusal. Red first: the corrected test fails on `784893d`.
2. **A claim in the packet is false and is corrected:** `docs/returns/P9.1-code.md` (Q2) says that until ruled *"only the hz1x1
   random_fresh DT is blocked"*; in fact `method_tier_grid.assert_declaration_matches_corpus` runs unconditionally and the tier's
   `target_rtg` / `rtg_scale` are NaN, so EVERY hz1x1 `random_fresh` arm is refused until the typed commit. The refusal is safe; the
   sentence goes.
3. **Q5 rides in M1:** `att_rederivation.run_campaign(..., factory_builder=None)` (C5's default-preserving hook) moves into the M1
   branch with its default-identity test, because P9.3's Phase B evaluates new arm names by P8.4b's route.
**To `DEFERRED` (no number or claim changes; logged as row 123):** the CPU suite cannot see a reordered dropout draw in the
registered IQL path (the reviewer's mutant — `value(s)` moved before `q_target(s)` — survives the CPU tests; only the CUDA-gated
regeneration kills it); `d4rl_run` accepts `--compare-with` under `"active"` and refuses only after all training; `d4rl_run main()`'s
protocol check has no test; `assert_distinct_schedules` does not check the intersection count.
**How M1 now merges:** a new branch `task/p9.1-m1` from `784893d`, with the commits for items 1–3 and their tests (one mutant per
fix, red first), the seven M1 test files and the changed modules' regression suites green, and the CUDA regeneration tests run once
(outside the sandbox) → **"P9.1 M1-fix ready"**. The coordinator re-runs the corrected test, the reviewer's MAJOR as a mutant, and
the two CUDA regenerations, merges `task/p9.1-m1` into `main` (M1), and hands the author the CI ceiling patch for M1's skips (your
Q4: eight tests that skip on a corpus-less, GPU-less runner). Then `git merge main` into `task/p9.1-a27a` (the conflicts are yours to
resolve, toward the fixed code).

## B.2 Rulings on Q1–Q10
- **Q1 — the episode index is WITHIN the draw** (implemented). A27(d) writes `default_rng([base_seed, draw_id, episode_index])` and
  does not define the index; every `random_fresh` tier has one episode per draw, and the within-draw index makes each episode's
  generator a function of (base seed, draw, its index in the draw) alone — independent of how a run is chunked, so one episode can
  be re-collected by itself. The manifest records the scheme. P9.3's hz4x4 tier inherits it.
- **Q2 — APPROVED as proposed:** `collect` (CPU) → a reviewed commit that types hz1x1 `random_fresh`'s `target_rtg` / `rtg_scale` from
  `output/p9_1/declaration_random_fresh_hz1x1.json` with `tests/test_random_fresh_prompt.py` re-deriving both from the raw `.npz`
  → `campaign` from a run tree at that commit. (With B.1.2's correction: every hz1x1 `random_fresh` arm waits for that commit.)
- **Q3 —** fixed by P5.2d, merged at `9503282`; `main`'s CI is read by the coordinator.
- **Q4 —** the coordinator measures the ceiling after M1's merge, by the registered route.
- **Q6 — two-sided, as implemented.** *"Every arm within 5 s of `bc_top10`"* reads |arm − `bc_top10`| < 5 s; HELD strictly inside
  (−5, +5) by P5.2c's interval rule, FAILED entirely outside, NOT RESOLVED otherwise.
- **Q7 — your reading, with the conjunction made explicit:** each arm pair's predicted direction is the sign of its registered
  40,000-step paired point difference; its 100,000-step paired interval is judged against it by P5.2c's rule; an exact 40,000-step
  tie predicts no order (NOT RESOLVED). The ordering's verdict: FAILED if any pair FAILED (the ordering *changes at* those pairs),
  HELD if all six HELD, NOT RESOLVED otherwise. The author's sentence has no branch for NOT RESOLVED: if it occurs, the G4 packet
  proposes the words and the author rules them.
- **Q8 — YES.** A27(c)'s governing clause is *"every registered statement in which IQL appears"*; its parenthetical list is the
  coordinator's enumeration and misses P4.6/P4.7's P1 (BC's rank among the four methods, IQL one of them) and the `*_vs_iql`
  behaviour comparisons. The governing clause decides, as it did for the grid's `mappo1000` draws (`BRIEF_48` Amendment A, A.2): the
  13 old hz1x1 cells (6,500 episodes) join the old-cell manifest BEFORE the campaign stage, and P1 and the `*_vs_iql` comparisons are
  re-scored with `iql_t` beside the original, by their registered rules, under both definitions.
- **Q9 — relayed to the author:** keep the machine awake for the whole campaign (Windows: never sleep while plugged in).
- **Q10 — THE AUTHOR'S RULING (2026-10-16), quoted:** *"no canary re-take and no power-mode change; size the campaign on the
  load-recorded re-take timings, since A27(a) already labels every timing an uncontrolled clock."* So: the campaign is sized on the
  re-take (≈ 40.5 GPU-h serial, plus ≈ 7 h of evaluation); Amendment A's *"canary before and after"* no longer applies to P9.1 (the
  load averages beside each timing stay); no stage of this round gates on P8.2's power-mode canary; every training second is reported
  as A27(a)'s uncontrolled clock, with the pause flags.

## B.3 A run-feasibility requirement for the campaign stage: the engine-destructor hang (`DEFERRED` 104)
CityFlow's `Engine::~Engine()` can wait forever at a two-party barrier when its controller thread has already left its loop
(`docs/notes/P5.3c_HANG_2026-09-30.md`, ≈ 2 in 1,200 episodes under load). It struck again on 2026-10-16: P9.3's A.7 sweep hung once
in 27 episodes while P9.3's teachers ran, and the coordinator's own test copies hung under load. It never changes a number (the hang
is after the episode, every write is atomic, the engine is seeded), but your evaluation stage destroys tens of thousands of engines,
most of them right after the last step — the exposed path. **Required before the `campaign` token, as P5.3c's C3.2 did at its stage:**
every worker process that rolls CityFlow episodes (route (i), route (ii), the old cells) is restarted automatically when it makes no
progress for 10 minutes — for example `faulthandler.dump_traceback_later(600, exit=True)` re-armed per episode, with the driver
re-running the shard on that exit and resuming from the files on disk; at most 20 restarts per shard, each logged with the time and
the dumped stack; any other non-zero exit stops the stage as now; a restart inside a stage needs no new token. Tests: a fake worker
blocked in a C-level wait is restarted and completes; a failing worker is not restarted; a healthy worker is never killed.
**The `collect` stage runs without it:** `offline.collect` writes the episode's `.npz` before the engine is freed (the less exposed
path; P9.3's 400-episode corpus ran without a hang today). If `collect` prints no line for 10 minutes, the author says "P9.1 collect
hung", and the coordinator rules the recovery.

## B.4 The tokens
- **`collect` — CLEARED**, from a detached run tree at `140149e` (`/home/filip/rltraffic-p91-run`, created by the coordinator): its
  code path (`offline.collect` and the declarations) is not touched by B.1's fix, and the reviewer verified the collector itself.
  CPU, one process, ≈ 400 episodes.
- **`campaign` — NOT YET.** Amendment C clears it once: M1 is merged (B.1) and merged into this branch; Q2's typed commit exists;
  Q8's 13 cells are in the old-cell manifest; B.3 is built and tested. Say **"P9.1 campaign ready"** with that commit. The GPU plan
  stands: training serial on the GPU, then evaluation with `WORKERS` ≤ 6.
- **`d4rl` — after the campaign's training**, from the same run tree, when nothing trains (unchanged).

---

# ✅ AMENDMENT B.1 — 2026-10-27: M1 MERGED into `main` (by the commit that carries this note); the author ruled points 1–7 finished in full

**The author's ruling (2026-10-27), quoted:** *"Points 1-7: finish all in full, and record the slip past 21 Oct as a deviation."* Nothing
in your campaign is cut — points 1, 2, 3, 4 and 6 run as planned (deviation D1, `docs/notes/A27_DEVIATIONS.md`).

**M1 verified by the coordinator, from disk, before the merge:**
- `task/p9.1-m1` at `0095758`: the three fixes read (B.1(1)'s `assert_declared_on_random_fresh` at all three training paths, with
  `dt_nomix` among `tier_sweep.DT_METHODS`; B.1(3)'s hook default-identical by construction; B.1(2)'s sentence).
- In a scratch git copy: M1's eight test files with the corpus set, **165 passed, 2 skipped** (the CUDA pair); the review's MAJOR as a
  mutant (the train-baselines guard removed) killed by 3 tests; the two CUDA regenerations, outside the sandbox, **2 passed** in 388 s
  — the registered P4.4 IQL and P4.6 BC checkpoints regenerate tensor for tensor under M1's code.
- On the merged tree, in CI's mode, M1's files with the suites of the modules it changes: **787 passed, 3 skipped**; the guard clean.

**Your next step:** `git merge --no-edit main` into `task/p9.1-a27a` (the `offline/tier_sweep.py` conflict your trial met, resolved as
`P9.1_evidence/campaign_ready_m1_merge_resolution.diff` records), the trial's 26 files and the whole suite re-run on that commit (one
pytest process, `OMP_NUM_THREADS=1`), then **"P9.1 campaign ready"** with that commit. Amendment C then creates the run tree there and
clears the `campaign` token. The CI skip ceiling for M1's eight corpus-gated tests is measured from this merge's CI run and handed to
the author as a patch (your Q4).

---

# ⚖️ AMENDMENT B.2 — 2026-10-27: the stall remedy's values CORRECTED for your campaign too (your Q12) — 120 s and 60 restarts per shard

`BRIEF_46` Amendment B, B.3 set 600 s and 20 restarts without computing the hangs the evaluation stage will meet — the coordinator's
error, found by P9.2's arithmetic (`BRIEF_47` Amendment D, D.3). Your evaluation destroys roughly 70,000 engines over six shards
(route (i) and route (ii) for 53 cells, 38 old cells, P3a's references), ≈ 12,000 per shard: at `DEFERRED` 104's rate a shard expects
≈ 20 hangs, so a cap of 20 stops a healthy shard about half the time and 600 s per hang idles ≈ 3.3 h per shard. **Ruled:** your
`StallGuard` cuts after **120 s** without progress (it re-arms on reset and step at most once per 10 s, so a stall is cut after
110–120 s; the longest legitimate pause between two steps is a checkpoint load, seconds) and a shard stops after **60** restarts.
Make them the defaults in `offline/stall_guard.py` and the driver (with an environment override, as P9.2's driver has one), pin both
by a test, and include the change in the commit you name in "P9.1 campaign ready".

---

# ⚠️ AMENDMENT B.2.1 — 2026-10-27 (no gate): your readiness at `9fc8ba2` predates Amendment B.2 — merge `main` again, make B.2's change, then say "P9.1 campaign ready"

- Your note `d9ef0b9` (14:50) names `9fc8ba2`, your merge of `main` at `c579424` (12:31). Amendment B.2 reached `main` at 13:31,
  inside the P9.2 merge `e4a5ab0`, after that merge. So at `9fc8ba2`, `offline/stall_guard.py` still has `STALL_SECONDS = 600.0`, and
  B.2's condition (*"include the change in the commit you name in 'P9.1 campaign ready'"*) is not met. This is not your error: the
  ruling landed after your merge.
- **Next:**
  1. `git merge --no-edit main`. Since your merge, `main` has changed only P9.2's files (`offline/demand_shift.py`,
     `offline/sumo_anchor_rescore.py`, `offline/policies/max_pressure_1to1.py`, the two P9.2 drivers and their tests) and docs.
  2. B.2's change: `StallGuard` cuts after 120 s and a shard stops after 60 restarts, as the defaults in `offline/stall_guard.py`
     and the driver, with an environment override, pinned by a test that runs red on 600 first.
  3. Re-run the trial's 26 files on that commit: one pytest process, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- **One audit (the class that turned `main` red today, `BRIEF_47` Amendment E):** `offline/campaigns/p9_1_corrective.sh` sets
  `MAIN=/home/filip/rltraffic` (line 49), derives `PY`, `OUTPUT`, `CORPUS` and `DRAWS` from it, and runs `cd "$MAIN"` (line 100).
  CI's runners have no such path. List every test that runs the driver, or reaches any other `/home/filip/rltraffic` literal in
  your code, and say how it behaves there: overridden, skipped with a named reason, or not reached. Fix a test that would fail
  there in the same commit, with an override like P9.2's `P92_*` variables.
- Then **"P9.1 campaign ready"** with that commit. Amendment C creates the run tree there and clears the token. The GPU is idle
  until then, and every idle hour comes out of P9.3's margin to 2026-11-07.

---

# ⛔ AMENDMENT C — 2026-10-27: the campaign is NOT cleared. B.2 is verified, but `tier_sweep.train_tier_dt` refuses the grid's `random_fresh` tier and would stop the campaign at training run 231 of 255. FIX FIRST (M1b), then a training smoke over every distinct cell

## C.1 Verified at `d0c834d`, from disk
- **The merge `f84c506`:** parents `d9ef0b9` and `7d4cbb3`.
- **B.2's change, read:** the guard's 120.0; the driver's `P91_STALL_SECONDS` (120) and `P91_MAX_RESTARTS` (60), validated before
  anything runs; `--stall-seconds` handed to every worker.
- **Tested in a scratch clone at `d0c834d`:** `tests/test_corrective_round_driver.py` and `tests/test_stall_guard.py`, one process,
  every `RLTRAFFIC_*` unset (CI's setting):
  - control: **40 passed**;
  - the guard back at 600.0 → **3 failed**;
  - the driver's cap back at 20 → **2 failed**;
  - **the driver with every `/home/filip/rltraffic` replaced by a path that does not exist → 40 passed**, so your audit's claim
    holds.
- **Q16, ruled:** no `P91_MAIN` override. No test reaches a `MAIN`-derived path (shown above), the campaign runs on this machine, and
  an override would change the campaign commit for symmetry alone.

## C.2 ⛔ The blocker: found by P9.3 (its plan `fc0fda2`, §10.2 C2), reproduced by the coordinator at `d0c834d`
- **The check.** `offline/tier_sweep.py`'s `train_tier_dt` has `if tier not in TIERS: raise ValueError(…)` (line 841). `TIERS` is the
  grid's four tiers. `random_fresh` lives in `A27_TIERS`, which `tier_spec` accepts and this check does not.
- **Reproduced.** Called with `tier="random_fresh", method="dt_nomix"`, it raises `ValueError: unknown tier 'random_fresh'; this task
  declares ['fixedtime', 'mappo1000', 'maxpressure', 'random']` before touching any data. The coordinator ran this in the clone.
- **Where the campaign reaches it.** `corrective_round.train` calls `ts.train_tier_dt(…, tier=cell.tier, …)` for
  `Cell("grid4x4", "random_fresh", "dt_nomix")`. In `training_runs()` order these are **runs 231–235 of 255**. The stage needs ≈ 40.5
  GPU-hours, so it would have trained for ≈ 35 GPU-hours and then stopped, with 21 runs untrained and no evaluation. A fix at that
  point would have split the campaign across two commits.
- **Why no test saw it.** The training tests replace the trainer with a spy, so the refusal is never reached. hz1x1's trainer already
  accepts A27's tiers (`method_tier_grid`, line 1490: `if tier not in TIERS and tier not in A27_TIERS`). The grid's was missed in M1,
  **and the coordinator's M1 review missed it too**: B.1 read the three training paths but ran none of them on `random_fresh`.

## C.3 FIX FIRST: M1b, on its own branch from `main` (as M1 was), because P9.3's Phase B needs the same line (its Q-B1)
1. Branch `task/p9.1-m1b` from `main` at `0e38175` or later, in its own worktree.
2. **The test first, red on the unchanged trainer.** `train_tier_dt` on the grid's `random_fresh` passes the tier check and trains,
   for example 2 gradient steps on CPU on a small synthetic stack. Its checkpoint record carries `tier` `random_fresh`. An undeclared
   name is still refused.
3. **The fix.** The check becomes `if tier not in TIERS and tier not in A27_TIERS:`, the mirror of `method_tier_grid`'s, and its
   message names A27's tiers. Nothing else in the trainer changes.
4. **A mutant.** Restore the old check: the test must fail. Report it killed against a control.
5. **Runs.** The touched test file and `tier_sweep`'s own test files, in one pytest process with
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; then hygiene, English and the guard. Then say **"P9.1 M1b ready"** with the commit. The
   coordinator merges it into `main`.

## C.4 Then on `task/p9.1-a27a`, with the training smoke (§7's CI rule (4): a spy proves the call; only the real callee proves the run)
1. `git merge --no-edit task/p9.1-m1b`, or merge `main` once M1b is on it. Either way it is the same commit.
2. **The training smoke, REQUIRED before the token.**
   - What: every distinct training cell of `training_runs()` (its 51 (scenario, tier, arm) cells), trained for ONE seed.
   - How: through the campaign's own training function, on the real corpora, with only two substitutions: the gradient steps (2) and
     the output root (a scratch directory). No spy and no stub trainer.
   - Where: on CPU in your session, or on CUDA outside its sandbox, as M1's regenerations ran.
   - Report: 51 of 51 completed, with seconds per cell. A refusal stops the smoke and is reported, not worked around.
3. **The evaluation smoke, where it needs no code change.**
   - What: each of the 53 new evaluation cells, for one episode of one held-out draw, through both routes, `att_ours` equal under
     `==`, using the smoke's checkpoints in the scratch root.
   - If the module cannot be pointed at the scratch root without a code change, do not change it. Instead, list which evaluation
     paths G2's pre-flight already ran and which no run has reached yet; the coordinator rules.
4. **The trial's 26 files** on the merge commit, in one process with one thread.
5. Then say **"P9.1 campaign ready"** with that commit. The coordinator verifies it, creates the run tree there and clears the token
   in Amendment D. Every hour the GPU idles comes out of P9.3's margin to 2026-11-07.

## C.5 2026-10-28: P9.3 no longer needs M1b; M1b itself is UNCHANGED
P9.3 found a second grid constant in `train_tier_dt`: the trainer writes the grid's `SCENARIO_ID` into every DT checkpoint. So
hz4x4's DT trains through P9.3's mirror of the trainer, which the coordinator verified (`BRIEF_48` Amendment H.2).

C.3's reason "because P9.3's Phase B needs the same line" therefore no longer holds. M1b's content and route stay exactly as C.3 and
C.4 say:
- one line, a red-first test and a mutant;
- "P9.1 M1b ready", then the coordinator merges it into `main`;
- C.4 on `task/p9.1-a27a`.

The scenario id is the grid's own on the grid, so nothing changes for P9.1.

---

# ✅ AMENDMENT D — 2026-10-28: the campaign is CLEARED. M1b verified and merged into `main` (`ff236af`); the run tree `/home/filip/rltraffic-p91-run-c` at `cb61ee7`; the token; the campaign does not wait for P9.3's smoke

## D.1 Verified by the coordinator, from disk
- **M1b** (`44d8c02`, fix `c0237de`):
  - The diff is the one line and its message, plus the test.
  - The test calls the real trainer.
  - In a scratch clone the test is green at `44d8c02`. With the old check restored it is red with C.2's own refusal
    (`unknown tier 'random_fresh'; this task declares ['fixedtime', 'mappo1000', 'maxpressure', 'random']`).
  - `tests/test_tier_sweep.py` and `tests/test_tier_sweep_a27.py`: 184 passed, 1 skipped.
  - No `/home/filip/rltraffic` in the diff.
  - **Merged into `main` as `ff236af`**, two parents (`53b994f`, `44d8c02`), with no code beyond `44d8c02`'s.
- **The campaign commit `cb61ee7`** is `c99d0c4` (`d0c834d` plus §14, docs) merged with `44d8c02`, with no conflict.
  - Its code delta from `d0c834d`, whose B.2 was verified in Amendment C, is M1b plus `main`'s own changes since then: P9.2's
    shift driver and tests, and the CI ceiling.
  - None of those is in the campaign's import path.
- **The training smoke** (`22513bd`, `P9.1_evidence/campaign_ready_c4_training_smoke.txt`):
  - It ran at HEAD `cb61ee7`: 51 of 51 cells, 2 gradient steps each, `smoke exit=0`.
  - Cell 47 (grid4x4 `random_fresh` `dt_nomix`) trained, in 7.60 s.
  - A full `p9.1-run/1.0` record in the smoke root carries `checkpoint`, `file_sha256`, `canonical_digest`, `declared_steps` and
    `gradient_steps`.
- **The evaluation smoke:**
  - `SUMMARY`: new cells 53 of 53 equal under `==`, P3a 6 of 6, old cells 38 of 38.
  - The main tree's `output/p9_1_runs/` still holds only the collect stage's capture and logs.
- **Your 17 changed test files at `cb61ee7`,** in a clone, one process, one thread, with the corpus and output roots set: **289
  passed, 6 skipped.** The skips are 5 T-regress (the campaign's artifacts do not exist yet) and 1 CUDA-trained checkpoint.
- **The run tree** `/home/filip/rltraffic-p91-run-c`:
  - The coordinator created it at 15:43, detached at `cb61ee7`, with no tracked change.
  - From it, against the main tree's roots, `corrective_round check --stage campaign` printed `{"ok": true, "stage": "campaign"}`.
  - The module loads from the run tree.

## D.2 Rulings
1. **C.4.3's unreached list is ACCEPTED as listed.**
   - **(1) hz1x1's route-(i) CLI on the A27 arms.** Its unreached checks are:
     - the training aggregate as the CLI reads it;
     - the five-run check;
     - the per-seed canonical digest;
     - A8(b)'s ceilings;
     - the evaluation file.

     Each compares a record the campaign writes with a file it writes. `write_training_aggregates` rebuilds the aggregate from the
     per-run records every time.
   - **So none of them can invalidate a trained checkpoint.** A refusal there costs an evaluation-side fix and a CPU re-run, never a
     GPU hour.
     - The factory and the rollout already ran for every A27 arm.
     - The CLI itself ran end to end on a registered hz1x1 cell (G2).
   - **(2) and (3)**, the full-volume orchestration and `declare-evaluation` on the real lists: the same reasoning. Both are
     unit-tested.
   - **If any of them refuses during the campaign,** the stage stops as designed. Report it; the trainings stand, and nothing is
     re-run without a ruling.
   - §7's rule (4) gains this clarification.
2. **P9.3's smoke and the GPU: the campaign does NOT wait for it.**
   - P9.3's H.3 smoke trains two steps per cell: a few GPU-seconds per cell beside about 53 s of CPU setup. Its records are never
     quoted.
   - Your runs' seconds are an uncontrolled clock (A27(a)), and the stage records its own GPU trace.
   - If the smoke is still on the GPU when the campaign starts, both run. The overlap is disclosed in your G4 packet (from the trace)
     and in P9.3's smoke note.
   - What stays forbidden is two CAMPAIGN trainings at once: P9.3's `p93-train` waits for your last training.
3. **The d4rl stage runs after P9.3's trainings, not before the campaign.**
   - The driver runs one P9.1 stage at a time: its `pgrep` refuses while your evaluation shards run.
   - Running it before the campaign would put about 40 minutes (five seeds at the pre-flight's 382–538 s) on the critical path.
   - Its token comes in a later amendment, once its prerequisites are re-read. They are on disk today: P8.3's venv, the dataset, and
     no `/home/filip/d4rl-calib-out/p9_1`.
4. **P9.2's stage 2 starts during the campaign,** from your on-disk records of the ten hz1x1 `iql_t` runs on `mappo1000` and `mix50`
   (runs 21–25 and 31–35 of 255, about 4 h in; `BRIEF_47` Amendment G). Nothing of yours changes.

## D.3 THE TOKEN — the `campaign` stage (the author's tmux)
Run it from the run tree above, in ONE tmux session. **Keep the machine awake for the whole stage** (Windows: never sleep while plugged
in). A suspension inside a run adds hours to its seconds.
```
# Step 1, open a pane:
mkdir -p /home/filip/rltraffic/output/p9_1_runs && tmux new -s p91
# Step 2, at ITS PROMPT (one line):
touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_campaign && bash /home/filip/rltraffic-p91-run-c/offline/campaigns/p9_1_corrective.sh campaign cb61ee7 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/campaign_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
```
- **First line:** `token consumed; stage campaign <stamp> from /home/filip/rltraffic-p91-run-c at cb61ee7; logs …`.
- **Then, in order:**
  - `train: ok` after about 40.5 GPU-hours, 255 runs serially;
  - `declare_evaluation: ok`;
  - `evaluate: 6 shards ok`;
  - `old-cells: 6 shards ok`;
  - `p3a_references: ok`.
- **Done when** the capture ends `stage campaign <stamp>: COMPLETE (status …)` and `DRIVER EXIT 0`.
- **On a stop** (`REFUSED:` or `FAILED:`): the capture names the step and its log. Nothing is re-run without a ruling. The stage is
  resumable from disk under a new token.
- **Two milestones to tell the coordinator about:**
  - P9.2's stage 2 can start when ten records exist. This prints 10:
    `ls /home/filip/rltraffic/output/p9_1_runs/training/hz1x1/{mappo1000,mix50}__iql_t__seed*.json | wc -l`
  - P9.3's `p93-train` can start once `train: ok` is in the capture.

## D.4 Meanwhile
- Nothing to build for the campaign; the run tree and the evidence stay as they are.
- Write G4's `report` path tests-first where it does not wait for data, as your plan says.
- d4rl's input comes last (D.2-3).

---

# ⚖️ AMENDMENT E — 2026-10-28: the author STOPPED the campaign stage; no hard stop (D8); every long run in SESSIONS. Session 1 = the d4rl stage (CLEARED). The training resumes after SESSION CONTROL is built (FIX FIRST)

## E.1 The author's rulings (2026-10-28, about 16:40), in his words
1. *"There is no hard stop any more: the 7 Nov date is withdrawn; record it as a deviation."* This is `A27_DEVIATIONS` D8.
2. *"No computation runs 50 h non-stop on this laptop. Split the rest of the campaign into sessions of a few hours each (roughly 6–9
   sessions in all), which the author starts and stops himself; each session must end cleanly between trainings, so nothing finished
   is lost or re-run. The same rule applies to P9.3's p93-train (~31 h) and to every other long run."* This is now `PROJECT_PLAN` §7
   rule (5) and `CLAUDE.md` §5.
3. Every run handed to him states first how many hours it runs without a break. Nothing under `output/` is deleted or moved without
   his ruling.

Amendment D's D.2-3 (d4rl after P9.3's trainings) and D.3 (one `campaign` stage) are superseded by E.3 and E.4.

## E.2 What Ctrl+C left: stage `20261028T151149Z`, 16:11:49–16:30:31, read by the coordinator
- **Runs 1–3 are DONE:** hz1x1 `iql_t@random`, seeds 101, 202 and 303.
  - Each has its checkpoint and record, written at 16:17:40, 16:23:31 and 16:29:28.
  - Each trained 40,000 steps on CUDA, in 344.0 s, 350.7 s and 356.8 s.
  - The module's `run_status` reads all three as done: recipe and canonical digest.
- **Run 4 (seed 404) was interrupted about a minute in,** at about step 5,000 of 40,000. The trainer writes nothing until its end, so
  there is no checkpoint, no `.partial` and no record. `run_status` reads it as missing, and the first training session trains it from
  the start.
- **Runs 4–255:** 252 missing; nothing damaged.
- **The stage's logs** are kept as its record: `output/p9_1_runs/logs/20261028T151149Z/train.log` ends with the `KeyboardInterrupt`
  traceback, and `gpu_trace.csv` holds 57 samples, its loop ended with the driver.
- The capture holds the token line only. The token was consumed at the start, and no process of the stage is left.
- **Handling: nothing is deleted or moved.** The next training session reads runs 1–3 from disk and skips them. Your report names this
  session: runs 1–3 at `cb61ee7`, stage `20261028T151149Z`.

## E.3 Session 1: the `d4rl` stage, CLEARED — about 35–45 minutes without a break
- **Why now:** D.2-3 deferred it only to keep about 40 minutes off a critical date. There is no date now, and the GPU is idle while E.4
  is built.
- **Verified by the coordinator:**
  - **The prerequisites** are on disk: P8.3's venv, the dataset, and no `/home/filip/d4rl-calib-out/p9_1`.
  - **The flags:** the runner accepts every flag the driver passes (`--threads`, `--dataset`, `--out`, `--arms`,
    `--iql-target-dropout`, `--compare-with`). Its defaults are the registered five seeds and 40,000 steps.
  - **The training path:** the pre-flight ran one full seed through the same runner (382.1 s; `g2_d4rl_one_seed.txt`).
  - **The comparison, which the pre-flight did not reach.** `--compare-with` runs AFTER all five seeds, so a refusal there would lose
    them. The coordinator ran it on the registered artifact, in P8.3's venv, from the run tree:
    - the registered contrasts recompute EXACTLY;
    - `compare_with_registered` completes on five stand-in `iql_t` cells of ten episodes each.

    Its other checks (dataset, codebook, budget, episodes) run before the first gradient step.
- **Do not stop it early.** It writes each seed's checkpoint into `/home/filip/d4rl-calib-out/p9_1` as it goes, so a stopped run
  leaves files that block a re-run.
- **The command**, in the author's tmux `p91` (its pane is at a prompt), as one line:
  `touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_d4rl && bash /home/filip/rltraffic-p91-run-c/offline/campaigns/p9_1_corrective.sh d4rl cb61ee7 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/d4rl_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"`
- **Done when** the capture shows `d4rl: ok` and `stage d4rl <stamp>: COMPLETE (outputs /home/filip/d4rl-calib-out/p9_1)`, and the
  pane shows `DRIVER EXIT 0`.
- P9.3's two-step smoke may share the GPU meanwhile (D.2-2's reasoning). The d4rl artifact's seconds are an uncontrolled clock.

## E.4 ⛔ FIX FIRST, before any training session: SESSION CONTROL (on `task/p9.1-a27a`, tests first)
The `campaign` stage cannot end between trainings. Its only stop is Ctrl+C, which loses the run in flight: a minute today, up to an
hour for a grid `dt_nomix` run. Build the following.
1. **`corrective_round train`:**
   - **`--hours H`.** At the start, plan the session: the next missing runs in `training_runs()` order whose planned seconds sum to at
     most H hours. The first run is always included. Print the plan (first and last run, count, planned hours), train exactly those
     runs, then exit 0.
     - The planned seconds are your pre-flight's per-arm figures (the 40.5 h table), labelled in the code as planning estimates and
       never as measurements of this campaign.
   - **`--max-runs N`**, for tests and smokes.
   - **A stop file, `output/p9_1_runs/STOP`.** It is checked before each run. When it exists, the session ends after the run in flight
     and says so. The code never deletes it, and refuses to START a session while it exists.
   - **A closing summary:** the runs trained (first to last) and their seconds, the next missing run, and the planned hours left.
2. **The driver:**
   - **(a) A `train` stage:** training only, with its own token `TOKEN_train` per session, the preamble checks, and a GPU trace per
     session. It passes `P91_SESSION_HOURS` (default 6) as `--hours`.
   - **(b) An `evaluate` stage,** after the last training. It refuses while any run is missing. It declares the evaluation once; a
     later session verifies the declaration unchanged instead of refusing. Then come the `evaluate` and `old-cells` shards, P3a's
     references and the status.
     - A session ends between units of work: a cell for route (i) and P3a, an episode for route (ii) and the old cells.
     - It ends at `P91_SESSION_HOURS` or at the stop file, and says how many units are left.
     - A later session resumes from disk.
   - **(c)** The `campaign` stage refuses, pointing to `train` and `evaluate`.
   - **(d)** `d4rl` is unchanged.
3. **Tests first, red for the right reason.** Mutants:
   - the hours plan ignored;
   - the stop file ignored;
   - a run started after the stop file, and a run interrupted by it;
   - `evaluate` started with a missing run;
   - a second declaration that differs.
4. **Proof that no training computation changed.** `train_run` and everything it calls must be byte-identical to `cb61ee7`'s. Name
   the functions and the method. Then runs 1–3 and the rest come from one training path, and your report names both commits.
5. **A session smoke (§7 rule (4)),** on a scratch root with C.4.2's two-step wrapper:
   - a `train` session with `--max-runs 3`;
   - the stop file before run 2 of the next session;
   - a resumed session;
   - `evaluate` on those checkpoints with a short deadline, then resumed to the end.
6. **Then:** the trial's 26 files and the changed test files on the new commit, and say **"P9.1 sessions ready"** with the commit. The
   coordinator verifies it, creates a new run tree there, and clears the first TRAINING session in Amendment F.

## E.5 The session plan, indicative
These are planning estimates from your pre-flight's figures. Today's three runs took 344–357 s against the plan's 414 s. The driver's
printed plan governs each session, and the author may set `P91_SESSION_HOURS` per session.

| session | what | planned hours |
|---|---|---|
| 1 | d4rl (E.3) | ≈ 0.6–0.75 |
| T1 | runs 4–65: hz1x1 `iql_t` (the rest) and hz1x1 `random_fresh`; P9.2's ten `iql_t` records are in it | ≈ 6.0 |
| T2 | runs 66 to about 121: the mixture and most of the 100k budget | ≈ 6.0 |
| T3 | to about run 150: the rest of the budget and the grid's matched `mappo1000` tier | ≈ 5.3 |
| T4 | to about run 186: matched `maxpressure`, then `fixedtime`'s BC arms | ≈ 5.7 |
| T5 | to about run 209: `fixedtime`'s `iql_m`, then the `random` tier | ≈ 5.9 |
| T6 | to about run 233: the grid's `iql_t`, then three `dt_nomix` on `random_fresh` | ≈ 5.5 |
| T7 | to run 255: the grid `random_fresh` arms | ≈ 6.0 |
| E1, E2 | the evaluation: 53 new cells, 38 old cells, P3a, status (≈ 7 h on six workers) | ≈ 3.5 each |

That is ten sessions at 6 h. Seven-hour training sessions would make about nine.

## E.6 Meanwhile
The run tree `/home/filip/rltraffic-p91-run-c` stays, because d4rl runs from it. `output/p9_1*` stays untouched. P9.2's ten records come
in T1.

## E.7 2026-10-28, evening: session 1 (`d4rl`) COMPLETE; the result VERIFIED; the GPU order
**The capture** (`output/p9_1_runs/d4rl_capture.txt`): the token line, then `d4rl: ok` and `stage d4rl 20261028T192029Z: COMPLETE
(outputs /home/filip/d4rl-calib-out/p9_1)`. Five seeds trained from 20:20:29 to 20:47:25, about 5.3 min each; the capture's last write is 20:47:31. The author confirmed the pane's `DRIVER EXIT 0` at 21:01, so all three of E.3's done-when conditions hold.

**Verified by the coordinator, from the artifact** `p8_3_train_k64_40000_iql_t.json` (sha256 `e168b203…`):
- `target_dropout` `off`; 40,000 steps; seeds 101–505; 10 evaluation episodes per seed.
- `runtime.git_commit` `cb61ee7`, not dirty; the same GPU, CUDA and torch as the registered run.
- `registered_contrasts_reproduced` true. The registered `bc_vs_iql` re-reads as −542.0464 [−1668.0995, 584.0066].
- **The three new contrasts were recomputed by the coordinator with the standard library** (per-seed means of the episode returns;
  mean ± 1.96 · sd / √5). They equal the artifact's:

  | Contrast (left − `iql_t`, raw return) | Mean [95 % CI] | Seeds won by the left arm |
  |---|---|---|
  | `bc_vs_iql_t` | **+3554.20 [3163.06, 3945.34]** | 5 of 5 |
  | `bc_top10_vs_iql_t` | +1629.50 [1172.01, 2086.98] | 5 of 5 |
  | `iql_vs_iql_t` | +4096.25 [3134.56, 5057.93] | 5 of 5 |

- **`iql_t`'s per-seed returns:** −58.82, −80.99, −57.17, −57.60, −57.17 (mean −62.35). The registered `iql`'s were 4814.13, 4544.28,
  2372.29, 4956.87 and 3481.90.
- **The training was unstable on every seed, in the artifact's diagnostics and the log.** For seed 101:
  - the Q loss rose from 0.12 at step 5,000 to 60.03 at 40,000, against the registered run's 0.08 to 0.56;
  - the V loss rose from 0.03 to 7.71, against 0.03 to 0.19;
  - the advantage mean is −3.55 (registered −0.14), and 49.7 % of AWR weights are near zero (registered 5.0 %).
  - Seed 303 is the same: Q loss 96.83, near-zero weights 50.0 %.

  These are observations. **No cause is claimed:** the registered design changes only the target forwards' module mode.
- **The traffic runs so far do not show the same signature.** Runs 1–3 (hz1x1 `iql_t@random`) have Q losses FALLING, from about 2,960
  at step 5,000 to 59–91 at 40,000 (`output/p9_1_runs/logs/20261028T151149Z/train.log`). Their returns are not evaluated yet.
- **Secured** (§7, *an output that exists in one place is not a record*): the JSON and the five checkpoints were copied to
  `output/p9_1_runs/d4rl_outputs_20261028T192029Z/`, with `SHA256SUMS.source` re-verified after the copy (6 of 6 OK).

**What it means for the paper.** A27(c)'s registered sentence reads *"…on the D4RL calibration the BC − IQL difference in raw return
becomes Δ′ [CI], against −542.05 [−1668.10, 584.01] before the fix."* Δ′ is **+3554.20 [3163.06, 3945.34]**. Nothing in the procedure
changes. The traffic `iql_t` arms run as registered.

**For G4** (no new code path):
- `p9_1_iql_targets.json`'s `d4rl` section takes this artifact byte for byte, as planned (J1(c)), cited by its sha256.
- Beside it, descriptively and labelled as diagnostics (not registered statistics), put each D4RL cell's final Q and V losses and AWR
  near-zero fraction for `iql` and `iql_t`. Do the same for every traffic `iql_t` run whose record or log carries them.

**The GPU order changed** (`BRIEF_48` I.3-4). P9.3's training sessions run while you build E.4. Your first training session goes next
once Amendment F clears it, after the P9.3 session then running ends. Report "P9.1 sessions ready" as E.4 says.

---

# ✅ AMENDMENT F — 2026-10-29: SESSION CONTROL VERIFIED at `c58afb8`; training session T1 CLEARED from the run tree `/home/filip/rltraffic-p91-run-d` (runs #4–#65, about 6 h without a break); Q18 ruled

Your packet: §17 of `docs/returns/P9.1-campaign-ready.md` (`9eb908a`, docs only), on `c58afb8`. It reached the coordinator on
the afternoon of 2026-10-29. The laptop's GPU is free: P9.3 now trains on machine B (`BRIEF_48` I.7).

## F.1 Verified by the coordinator, from disk
- **The code that changed since `cb61ee7`** is:
  - `offline/corrective_round.py` and `offline/campaigns/p9_1_corrective.sh`;
  - `offline/att_rederivation.py`, for the opt-in `should_stop`;
  - `offline/corrective_report.py`, for G4's report path (D.4).

  No other code file differs, so every trainer module and `agent/` are byte-identical.
- **One training path (E.4-4), by the coordinator's own `ast` script:**
  - In `offline/corrective_round.py`, 9 definitions changed: `build_parser`, `check`, `evaluate`, `evaluate_cell`, `main`,
    `record_registered_references`, `roll_old_cells`, `run_route_ii` and `train`. 11 were added and none removed.
  - The training roots are `train_run`; `_hz1x1_inputs` and `_grid_inputs`; and `training_runs`, `training_cells`, `run_status`,
    `recipe_fields`, `checkpoint_path` and `run_record_path`. Their closure over module-level names has the same 21 definitions at
    both commits, and none changed. It equals your list.
  - `train`'s 13-line input block is identical. The rest of its diff is the session wrapper: the stop-file refusal, the plan, the
    check before each run and the summary.
  - **So runs 1–3 (`cb61ee7`) and runs 4–255 (`c58afb8`) come from one training path.**
- **The session smoke (E.4-5),** read from its root `/tmp/claude-1000/e4/smoke` by the coordinator's own script:
  - 7 run records, #1–#7, each at 2 steps, and each checkpoint's sha256 is its record's.
  - They were written in the transcript's order: #1–#3; then #4 alone, during which STOP appeared; then #5–#7 after the refused
    start.
  - Route (ii) holds 1,000 episode files. The 66 of `iql_t@random` seed 101 written before evaluate session 3 began (draws
    1000–1065) were not rewritten.
  - No STOP is left.
- **The whole suite at `c58afb8`,** in a scratch clone: one process, one thread, no corpus variables (CI's setting), HEAD held, no
  tracked change. **3,956 passed, 3 failed, 273 skipped** in 954.8 s.
  - The 3 failures are the tmux tests that the sandbox refuses (`error connecting to /tmp/tmux-1000/default (Operation not
    permitted)`), as on 2026-10-16: `test_the_headers_own_line_passes_the_guard_and_the_pane_carries_the_drivers_exit_code` in
    `test_p7_3c_campaign.py`, `test_p7_3c_corpus_driver.py` and `test_p7_3d_campaign_path.py`.
  - A second run, which names its failures, gave the same three and the same counts (935.9 s).
- **The coordinator's mutants** on the session control, each caught by your two session test files (control: 62 passed):

  | Mutant | Result |
  |---|---|
  | K1: the planner ignores the hours bound | KILLED |
  | K2: the stop file ignored between runs | KILLED |
  | K3: a session starts while the stop file exists | KILLED |
  | K4: the driver's stop refusal binds every stage again (`11476c0`'s line) | KILLED |
  | K5: hz1x1 `iql_t` planned without its factor 1.15 | KILLED |
  | K6: the driver trains without `--hours` | KILLED |
  | K7: a later run pulled past one that does not fit | KILLED |
  | K8: the driver's default session is 60 hours | KILLED |

  Each file was restored by `git checkout`, and the clone was clean after every mutant.
- **The run tree** `/home/filip/rltraffic-p91-run-d`:
  - The coordinator created it at 14:04:38 on 2026-10-29, detached at `c58afb8`, with no tracked change.
  - From it, on the main roots, read-only (the listing of `output/p9_1_runs` was unchanged):
    - the module loads from the run tree;
    - `check --stage train` printed `{"ok": true, "stage": "train"}`;
    - `train --hours 6 --plan-only` planned runs #4 (hz1x1 `iql_t@random` seed 404) to #65 (hz1x1 `iql_t@random_fresh` seed 505):
      62 runs, 5.95 planned hours, 252 missing, 3 done. That is E.5's T1:
      - hz1x1 `iql_t`: `random` seeds 404 and 505, then `fixedtime`, `maxpressure`, `mappo500`, `mappo1000`, `mix33`, `mix50`
        and `mix67`, five seeds each;
      - hz1x1 `random_fresh`: `dt`, `bc`, `bc_top10`, `iql` and `iql_t`, five seeds each.
    - P9.2's ten records (`mappo1000` and `mix50` `iql_t`) are in it.
    - The 255 runs plan 40.55 h in all; 34.26 planned hours are left after T1.

## F.2 Rulings
1. **The deviation in `11476c0` is closed by `c58afb8`.** The driver's hours check and its stop refusal open with
   `case "$STAGE" in train|evaluate)`, and restoring the old line is killed (your S1; the coordinator's K4).
2. **The evaluate overrun is accepted as disclosed.** An evaluate session runs `P91_SESSION_HOURS` plus the unit in flight in each
   shard. Before the evaluate sessions are cleared, report how long a grid route (i) cell takes, from the pre-flight or measured, so
   that their hours can be stated first.
3. **Q18: yes, in the waiting time.** Build E.7's G4 diagnostics tests first, on the branch:
   - CPU only, one process, one thread, nothing on the GPU;
   - the run tree and `output/p9_1*` untouched;
   - first read which of the run record and `train.log` carries the values;
   - label them as diagnostics, not registered statistics (E.7).
4. **`train`'s docstring needs one correction.** It now says an interrupted call leaves no checkpoint and no record. That holds for
   an interruption during training. An interruption while a checkpoint is written leaves a `.partial`, which `run_status` refuses
   (unchanged code). Restore that case in your next commit.
5. **The two tests written after the code are accepted.** Mutants E11 and E3 show that each catches what it guards.

## F.3 ✅ THE TOKEN: training session T1 (the author's tmux `p91`)
**T1 runs about 6 hours without a break:** 5.95 planned hours. Runs 1–3 took about 15 % less than planned, so it may end earlier. It
trains runs #4–#65 and ends by itself at the end of its plan.
- **Before it:** nothing else trains on the laptop's GPU; P9.3 trains on machine B. Keep the laptop plugged in and awake.
- **The command,** at the pane's prompt, as one line. It first checks that torch sees CUDA, because the trainer falls back to the
  CPU without a word, and only then creates the token:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T1 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
  - The CUDA check was run on the laptop by the coordinator: it passes outside the sandbox and refuses inside it, where the GPU is
    blocked. `NO CUDA: T1 not started` means nothing started and no token was created.
- **The pane** shows the token line, `train session: it runs about 6 hours without a break …` and the plan (`SESSION: runs #4 …
  to #65 …: 62 runs, 5.95 planned hours`). Then it is quiet until the end. Progress is in
  `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 62 runs trained …; ended by the plan; 190 runs missing, 34.26 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight, up to about
  7 minutes later. Remove the file by hand before the next session.
- **A failure** stops the session at the failed run (`FAILED: train exited …`, `DRIVER EXIT 1`). Report it; nothing is re-run
  without a ruling.
- **Ctrl+C is the emergency stop only.** It loses the run in flight. If it lands while a checkpoint is being written, the next
  session refuses that run's `.partial`, and the coordinator rules.

## F.4 Meanwhile
- Q18 (F.2-3) and F.2-4's docstring.
- Nothing on the GPU while T1 runs.
- After T1, the coordinator reads its records and names T2. P9.2's stage 2 follows T1 (`BRIEF_47` Amendment H).

# ✅ AMENDMENT G — 2026-10-29, night: T1 VERIFIED from disk; training session T2 CLEARED (runs #66–#125, about 6 h without a break); each run's code commit comes from its session's token line

## G.1 T1, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261029T154931Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the plan (#4–#65, 62 runs, 5.95 planned hours);
  - `train: ok`;
  - the summary (`62 runs trained in 17508.5 s (their records' clock); ended by the plan; 190 runs missing, 34.26 planned hours
    left`);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`.
- **The times (CET):**
  - the token at 16:49:31, the first run started at 16:49:39;
  - the last run finished at 21:42:21, and the capture's last write is 21:42:23;
  - 4.88 h on the wall, and 4.86 h of training on the records' clock: 0.82 of the 5.95 planned hours.
- **`train.log`:** 62 `TRAIN` lines and 62 `done in` lines; no warning, error, traceback or NaN line.
- **The coordinator's own script** (CPU, read-only) over runs 1–65:
  - The code's order of runs 1–65 equals the manifest written out by hand.
  - **Every record** has:
    - format `p9.1-run/1.0`, its cell, seed, recipe and label;
    - `gradient_steps` equal to `declared_steps` (40,000), and its checkpoint's path;
    - a finite final loss and finite window means.
  - **Every checkpoint's** sha256, recomputed, equals the record's `file_sha256`, and its canonical digest equals the record's. 0 problems.
  - **T1's 62 runs** ran on `cuda` with OMP and MKL at one thread, in run order, none overlapping, all inside the session.
  - **Runs 1–3** are untouched (their files' times are before T1).
  - `training_status`: 255 declared, 65 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 877 samples, the largest gap 20.1 s (no suspension). Inside runs, utilisation was at least 16 % (median
    42 %); memory peaked at 2,860 MiB.
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against the planning estimates:

  | Arm | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|
  | `iql_t`, the eight tiers | 37 | 329.8–347.6 s, 338.1 s | 415.1 s |
  | `random_fresh` `dt` | 5 | 187.8–284.3 s, 212.3 s | 204.0 s |
  | `random_fresh` `bc` | 5 | 87.2–88.6 s, 88.0 s | 110.0 s |
  | `random_fresh` `bc_top10` | 5 | 86.8–88.3 s, 87.5 s | 120.0 s |
  | `random_fresh` `iql` | 5 | 295.5–299.2 s, 297.3 s | 361.0 s |
  | `random_fresh` `iql_t` | 5 | 312.9–316.6 s, 314.4 s | 415.1 s |
- **P9.2's ten** (runs 21–25 and 31–35) are among the verified runs; `BRIEF_47` G.5 gives them to P9.2.

## G.2 Found, and ruled: the checkpoints do not carry the code's commit
- **The checkpoints' `runtime.git_commit` is the main tree's HEAD when they were written:**
  - `265ecad` for runs 1–3;
  - `9d3e042` for T1's 62 runs.

  Both are documentation commits on `main`, which never held the training code. The driver runs from the main tree, and the
  provenance reads that tree's HEAD. The run records carry no commit.
- **Ruling:** G4 names each run's code commit by its session:
  - the session is the one whose window holds the run's `started_unix`;
  - its token line names the commit and the run tree, which the driver checks are clean at that commit before it starts:
    `campaign_capture.txt` gives `cb61ee7` for runs 1–3, and `train_capture.txt` gives each training session's.
  - The checkpoint's `git_commit` is reported only as what it is, the main tree's HEAD at write time, never as the code's commit.

## G.3 ✅ THE TOKEN: training session T2 (the author's tmux `p91`)
**T2 runs about 6 hours without a break:** 5.95 planned hours. T1 ran at 0.82 of its plan. T2's 100,000-step arms are planned at
2.5 times their 40,000-step cost, so it may end nearer 5 hours. It ends by itself at the end of its plan.
- **What it trains:** runs #66–#125, 60 runs.
  - hz1x1 `mix_m1000_m500`: `dt`, `bc`, `bc_top10`, `iql` and `iql_t`, five seeds each (25 runs);
  - A27(g)'s 100,000-step arms on `mappo1000`: `dt_100k`, `bc_100k`, `bc_top10_100k` and `iql_100k` (20 runs);
  - the same on `mix50`, without `iql_100k` (15 runs). `mix50`'s `iql_100k` opens T3.
- **Verified before clearing** (CPU, read-only; the listings of `output/p9_1*` and the corpus root unchanged):
  - from the run tree, `check --stage train` printed `{"ok": true, "stage": "train"}`;
  - `train --hours 6 --plan-only` planned #66 (hz1x1 `dt@mix_m1000_m500` seed 101) to #125 (hz1x1 `bc_top10_100k@mix50` seed 505):
    60 runs, 5.95 planned hours, 190 missing, 65 done;
  - T2's three tiers build their training inputs from the run tree's code: `mix_m1000_m500` in 7.3 s, `mappo1000` in 3.0 s,
    `mix50` in 9.5 s. Each has 200 streams, 20 kept, and a finite prompt. `check` does not build them, and a failure at #66 would
    cost a session start.
- **Before it:** nothing else trains on the laptop's GPU. P9.2's stage 2 may run on the CPU beside it (`BRIEF_47` G.2-3), once its
  token is given. Keep the laptop plugged in and awake.
- **The command** is T1's, with its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T2 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line and the plan (`SESSION: runs #66 … to #125 …: 60 runs, 5.95 planned hours`).
  Then it is quiet until the end. Progress is in `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 60 runs trained …; ended by the plan; 130 runs missing, 28.31 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight, up to about
  15 minutes later (`iql_100k` is planned at 15 minutes). Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3.

## G.4 After T2
- **Five more sessions at `--hours 6`, as planned today:**

  | Session | Runs | Planned hours |
  |---|---|---|
  | T3 | #126–#154 | 5.92 |
  | T4 | #155–#187 | 5.70 |
  | T5 | #188–#211 | 5.98 |
  | T6 | #212–#234 | 5.68 |
  | T7 | #235–#255 | 5.03 |

  Each session's own plan decides at its start. T3 trains the grid's first runs; their inputs are checked before T3 is cleared.
- **Meanwhile:** Q18 (F.2-3), F.2-4's docstring, and G.2's ruling in the report. Nothing on the GPU while T2 runs.

# ✅ AMENDMENT H — 2026-10-30: T2 VERIFIED from disk; training session T3 CLEARED (runs #126–#154, about 6 h without a break), the grid's first runs

## H.1 T2, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261029T220237Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the plan (#66–#125, 60 runs, 5.95 planned hours);
  - `train: ok`;
  - the summary (`60 runs trained in 19045.2 s (their records' clock); ended by the plan; 130 runs missing, 28.31 planned hours
    left`);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`.
- **The times (CET):**
  - the token at 23:02:37 on 2026-10-29, the first run started at 23:02:47;
  - the last run finished at 04:20:30 on 2026-10-30, and the capture's last write is 04:20:31;
  - 5.30 h on the wall, and 5.29 h of training on the records' clock: 0.89 of the 5.95 planned hours.
- **`train.log`:** 60 `TRAIN` lines and 60 `done in` lines; no warning, error, traceback or NaN line.
- **The coordinator's own script** (CPU, read-only) over runs 1–125:
  - The code's order of all 255 runs (cell, A27 point, seed) equals the manifest written out by hand from 3.4 and G.3, with the
    grid's tiers in `tier_sweep.TIER_ORDER`.
  - **Every record** has:
    - format `p9.1-run/1.0`, its cell, seed, recipe and label;
    - `gradient_steps` equal to `declared_steps`, both written out by hand: 100,000 for the 35 `_100k` runs, 40,000 for the rest;
    - for `iql_100k`, the registered targets (`target_dropout` `active`), as 3.4 requires;
    - its checkpoint's path, a finite final loss and finite window means.
  - **Every checkpoint's** sha256, recomputed, equals the record's `file_sha256`, and its canonical digest equals the record's.
    0 problems.
  - **T2's 60 runs** ran on `cuda` with OMP and MKL at one thread, in run order, none overlapping, all inside the session.
  - **Runs 1–65** are untouched: their files' times are before T2. P9.2's ten pinned checkpoints are among them.
  - `training_status`: 255 declared, 125 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 952 samples, the largest gap 20.1 s (no suspension). Inside runs, utilisation was at least 18 % (median
    43 %); memory peaked at 2,909 MiB.
  - **G.2, seen again within one session:** the checkpoints' `runtime.git_commit` is `2328305` for runs #66–#73 and `70c72ac` for
    #74–#125, because `70c72ac` was committed on `main` at 23:23:23. Both are documentation commits. T2's code commit is `c58afb8`,
    from its token line.
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against the planning estimates:

  | Arm | Tier | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|---|
  | `dt` | `mix_m1000_m500` | 5 | 173.4–178.6 s, 175.5 s | 204.0 s |
  | `bc` | `mix_m1000_m500` | 5 | 85.7–93.3 s, 89.5 s | 110.0 s |
  | `bc_top10` | `mix_m1000_m500` | 5 | 91.5–98.1 s, 94.4 s | 120.0 s |
  | `iql` | `mix_m1000_m500` | 5 | 315.6–321.7 s, 317.3 s | 361.0 s |
  | `iql_t` | `mix_m1000_m500` | 5 | 338.5–342.3 s, 340.3 s | 415.1 s |
  | `dt_100k` | `mappo1000` | 5 | 494.8–515.7 s, 506.2 s | 510.0 s |
  | `dt_100k` | `mix50` | 5 | 510.2–518.2 s, 515.0 s | 510.0 s |
  | `bc_100k` | `mappo1000` | 5 | 238.4–243.2 s, 241.0 s | 275.0 s |
  | `bc_100k` | `mix50` | 5 | 240.8–252.5 s, 245.3 s | 275.0 s |
  | `bc_top10_100k` | `mappo1000` | 5 | 236.4–242.4 s, 238.5 s | 300.0 s |
  | `bc_top10_100k` | `mix50` | 5 | 240.1–244.5 s, 242.4 s | 300.0 s |
  | `iql_100k` | `mappo1000` | 5 | 799.6–811.4 s, 803.7 s | 902.5 s |

  `dt_100k` is the only arm that reached its plan.

## H.2 T3, checked before clearing (CPU, read-only; the listings of `output/p9_1*` and of the corpus root unchanged)
- From the run tree, `check --stage train` printed `{"ok": true, "stage": "train"}`.
- `train --hours 6 --plan-only` planned #126 (hz1x1 `iql_100k@mix50` seed 101) to #154 (grid4x4 `bc_m@maxpressure` seed 404):
  29 runs, 5.92 planned hours, 130 missing, 125 done. That is G.4's T3.
- **T3's three tiers build their training inputs** from the run tree's code, by the calls `train` makes:
  - hz1x1 `mix50` in 29.8 s: 200 streams, 20 kept, a finite prompt.
  - grid4x4 `mappo1000` in 124.3 s and `maxpressure` in 115.1 s. Each has:
    - 3,200 streams;
    - 320 kept by the global and 320 by the per-intersection top-10 % selection;
    - a finite IQL reward scale (1.295 and 3.636);
    - BC windows of 1,152,000 × 20 (115,200 for each top-10 % selection).
- **Host memory.** `train` keeps every tier's inputs until the session ends, so at #151 it holds `mappo1000`'s inputs while it
  builds `maxpressure`'s. The coordinator ran that sequence: 17.3 GiB held, and 21.4 GiB at the peak. The machine has 37.3 GiB
  available and 16 GiB of swap.
- **GPU memory.** The pre-flight's grid runs on this RTX 5080 (16 GB) peaked at 5,673 MiB (`bc_m`) and 7,837 MiB (`iql_m`), as
  whole-device figures. `train_bc` and `train_iql` copy their data to the GPU inside each run, so nothing builds up through the
  input cache.
- **The hours.** The grid's figures are single runs of seed 101 on `mappo1000`:
  - `bc_m` took 413.0 s and 543.0 s, and `iql_m` 2,077.3 s and 2,187.3 s; the plan uses the larger of each;
  - `bc_top10_m` and `bc_top10_perix_m` (97 s) are the plan's figures, not measured; their batch is 102, against `bc_m`'s 1,024.

  With T2's `iql_100k` (803.7 s), T3 trains for 5.3–5.8 h, plus about 5 minutes to build its inputs.

## H.3 ✅ THE TOKEN: training session T3 (the author's tmux `p91`)
**T3 runs about 6 hours without a break:** 5.92 planned hours, or 5.3–5.8 h from the measured figures (H.2). It ends by itself at
the end of its plan.
- **What it trains:** runs #126–#154, 29 runs.
  - hz1x1 `iql_100k` on `mix50` (5 runs);
  - the grid's first runs: A27(b)'s matched arms on `mappo1000`, that is `bc_m`, `bc_top10_m`, `bc_top10_perix_m` and `iql_m`
    (20 runs);
  - `bc_m` on `maxpressure`, seeds 101–404 (4 runs). Its seed 505 opens T4.
- **Before it:** nothing else trains on the laptop's GPU. P9.2's stage 2 may run on the CPU beside it (`BRIEF_47` H.5). Keep the
  laptop plugged in and awake.
- **The command** is T2's, with its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T3 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line and the plan (`SESSION: runs #126 … to #154 …: 29 runs, 5.92 planned
  hours`). Then it is quiet until the end. Progress is in `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 29 runs trained …; ended by the plan; 101 runs missing, 22.39 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight, up to about
  37 minutes later (`iql_m` is planned at 36.5 minutes). Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3.

## H.4 After T3
- **Four more sessions, as G.4 planned:** T4 (#155–#187, 5.70 planned hours), then T5, T6 and T7. Each session's own plan decides
  at its start.
- **The grid's other inputs are checked before the session that first builds them:** `fixedtime` (T4), `random`, and
  `random_fresh` with `dt_nomix`'s parts.
- **Meanwhile:** as G.4. Nothing else on the GPU while T3 runs.

# ✅ AMENDMENT I — 2026-10-30: T3 VERIFIED from disk; training session T4 CLEARED (runs #155–#187, about 5 h without a break), the grid's `maxpressure` and `fixedtime` arms

## I.1 T3, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261030T114038Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the plan (#126–#154, 29 runs, 5.92 planned hours);
  - `train: ok`;
  - the summary (`29 runs trained in 19611.6 s (their records' clock); ended by the plan; 101 runs missing, 22.39 planned hours
    left`);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`.
- **The times (CET):**
  - the token at 12:40:38 on 2026-10-30, the first run started at 12:40:51;
  - the last run finished at 18:10:06, and the capture's last write is 18:10:09;
  - 5.49 h on the wall, and 5.45 h of training on the records' clock: 0.92 of the 5.92 planned hours.
- **The coordinator's own script** (CPU, read-only) over runs 1–154:
  - The code's order of all 255 runs (cell, A27 point, seed) equals the manifest written out by hand from 3.4 and G.3.
  - **Every record** has:
    - format `p9.1-run/1.0`, its cell, seed, recipe and label;
    - `gradient_steps` equal to `declared_steps`, both written out by hand: 100,000 for the 40 `_100k` runs, 40,000 for the rest;
    - its checkpoint's path, a finite final loss and finite window means.
  - **T3's recipes, also written out by hand:**
    - hz1x1 `iql_100k` with the registered targets (`target_dropout` `active`);
    - the grid's `bc_m` at batch 1,024, and `bc_top10_m` and `bc_top10_perix_m` at 102;
    - `iql_m` at 20,480, with `target_dropout` `off`.
  - **Every checkpoint's** sha256, recomputed, equals the record's `file_sha256`, and its canonical digest equals the record's.
    0 problems.
  - **T3's 29 runs** ran on `cuda` with OMP and MKL at one thread, in run order, none overlapping, all inside the session.
  - **Runs 1–125** are untouched: their files' times are before T3.
  - `training_status`: 255 declared, 154 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 987 samples, the largest gap 20.1 s (no suspension). Inside runs, utilisation was at least 27 % (median
    96 %); memory peaked at 7,617 MiB (`iql_m`), below the pre-flight's 7,837 MiB.
  - **G.2 again:** the checkpoints' `runtime.git_commit` is `3a538f5` for all 29. It was committed on `main` at 12:46:05, and T3's
    first checkpoint was written at 12:58:29. T3's code commit is `c58afb8`, from its token line.
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against the planning estimates. These are the grid arms'
  first five-seed figures: the pre-flight had measured `bc_m` and `iql_m` on seed 101 only, and neither top-10 % arm (H.2).

  | Arm | Tier | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|---|
  | hz1x1 `iql_100k` | `mix50` | 5 | 798.2–1,055.3 s, 940.8 s | 902.5 s |
  | `bc_m` | `mappo1000` | 5 | 475.8–509.5 s, 484.2 s | 543.0 s |
  | `bc_m` | `maxpressure` | 4 | 478.2–490.4 s, 486.5 s | 543.0 s |
  | `bc_top10_m` | `mappo1000` | 5 | 92.7–98.6 s, 96.0 s | 97.0 s |
  | `bc_top10_perix_m` | `mappo1000` | 5 | 93.6–98.3 s, 96.5 s | 97.0 s |
  | `iql_m` | `mappo1000` | 5 | 1,890.8–1,929.2 s, 1,915.6 s | 2,187.3 s |

- **Beside P9.2's stage 2.** Stage 2 ran on 3 CPU shards from 12:41:08 to 13:32:55 (`BRIEF_47` Amendment I).
  - The first three `iql_100k` runs (#126–#128) ran beside it and took 1,005.6–1,055.3 s.
  - #129 (beside it for 15 s) and #130 took 798.8 s and 798.2 s, as T2's `iql_100k` on `mappo1000` did (799.6–811.4 s).

  The shards slowed the GPU run by 26–32 %, about 12 minutes of T3's wall time in all. The author's reading is confirmed by the
  records. No reported number changes: the records' clock is uncontrolled (A27(a)).
- **The grid's inputs, built inside T3:** 56 s for `mappo1000` (between #130 and #131), and 71 s for `maxpressure` (between #150
  and #151).

## I.2 T4, checked before clearing (CPU, read-only; the listings of `output/p9_1*` and of the corpus root unchanged)
- From the run tree (clean at `c58afb8`), `check --stage train` printed `{"ok": true, "stage": "train"}`.
- `train --hours 6 --plan-only` planned #155 (grid4x4 `bc_m@maxpressure` seed 505) to #187 (grid4x4 `iql_m@fixedtime` seed 202):
  33 runs, 5.70 planned hours, 101 missing, 154 done. That is G.4's T4.
- **`fixedtime`, the tier T4 builds for the first time,** from the run tree's code, by the call `train` makes:
  - built in 44.2 s;
  - 3,200 streams;
  - 320 kept by the global and 320 by the per-intersection top-10 % selection;
  - a finite IQL reward scale (2.024);
  - BC windows of 1,152,000 × 20 (115,200 for each top-10 % selection), and the group (40, 8).
- **Host memory.** `train` keeps every tier's inputs until the session ends, so at #171 it holds `maxpressure`'s inputs while it
  builds `fixedtime`'s. The coordinator ran that sequence: 16.5 GiB held, and 20.6 GiB at the peak. The machine had 36.7 GiB
  available. `maxpressure`'s inputs built as in H.2 (3,200 streams, 320 kept, IQL reward scale 3.636).
- **GPU memory.** T4 trains T3's four grid arms on two other tiers. T3's peak was 7,617 MiB of the 16 GB.
- **The hours.** By T3's measured figures per arm, T4 trains 5.07 h (the means) to 5.15 h (the maxima), plus about 2 minutes to
  build its two tiers' inputs. The plan's 5.70 h uses the pre-flight's larger figures.

## I.3 ✅ THE TOKEN: training session T4 (the author's tmux `p91`)
**T4 runs about 5 hours without a break:** 5.1–5.2 h by T3's measured figures (I.2), and at most its 5.70 planned hours. It ends by
itself at the end of its plan.
- **What it trains:** runs #155–#187, 33 runs.
  - `bc_m` on `maxpressure`, seed 505;
  - A27(b)'s other three arms on `maxpressure`: `bc_top10_m`, `bc_top10_perix_m` and `iql_m` (15 runs);
  - on `fixedtime`: `bc_m`, `bc_top10_m` and `bc_top10_perix_m` (15 runs), and `iql_m` seeds 101 and 202. Its seeds 303–505 open
    T5.
- **Before it:** nothing else trains on the laptop's GPU, and no long CPU job runs beside it (I.1: three CPU shards slowed T3's GPU
  run by 26–32 %). Keep the laptop plugged in and awake.
- **The command** is T3's, with its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T4 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line and the plan (`SESSION: runs #155 … to #187 …: 33 runs, 5.70 planned
  hours`). Then it is quiet until the end. Progress is in `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 33 runs trained …; ended by the plan; 68 runs missing, 16.69 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight, up to about
  37 minutes later (`iql_m` is planned at 36.5 minutes, and measured at 32). Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3.

## I.4 After T4
- **Three more sessions, as G.4 planned:** T5 (#188–#211, 5.98 planned hours), T6 and T7. Each session's own plan decides at its
  start.
- **The grid's other inputs are checked before the session that first builds them:** `random` (T5), and `random_fresh` with
  `dt_nomix`'s parts.
- **Meanwhile:** as G.4. Nothing else on the GPU while T4 runs.

# ✅ AMENDMENT J — 2026-10-31, night: T4 VERIFIED from disk; training session T5 CLEARED (runs #188–#211, about 5.5 h without a break), the grid's `random` arms and its first `iql_t`

## J.1 T4, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261031T095739Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the plan (#155–#187, 33 runs, 5.70 planned hours);
  - `train: ok`;
  - the summary (`33 runs trained in 17416.4 s (their records' clock); ended by the plan; 68 runs missing, 16.69 planned hours
    left`);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`.
- **The times (CET):**
  - the token at 10:57:39 on 2026-10-31, the first run started at 10:58:40;
  - the last run finished at 15:50:04, and the capture's last write is 15:50:05;
  - 4.87 h on the wall, and 4.84 h of training on the records' clock: 0.85 of the 5.70 planned hours.
- **The coordinator's own script** (I.1's, moved to runs 1–187; CPU, read-only):
  - The code's order of all 255 runs equals the manifest written out by hand.
  - **Every record and checkpoint of runs 1–187:** its cell, seed, recipe, label and steps, the checkpoint's sha256 and canonical
    digest, a finite final loss and finite window means. 0 problems.
  - **T4's recipes, written out by hand:** `bc_m` at batch 1,024, `bc_top10_m` and `bc_top10_perix_m` at 102, `iql_m` at 20,480
    with `target_dropout` `off`.
  - **T4's 33 runs** ran on `cuda` with OMP and MKL at one thread, 40,000 steps each, in run order, none overlapping, all inside the
    session.
  - **Runs 1–154** are untouched: their files' times are before T4.
  - `training_status`: 255 declared, 187 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 876 samples, the largest gap 20.1 s. Inside runs, utilisation was at least 30 % (median 96 %); memory peaked
    at 7,416 MiB.
  - **G.2 again:** the checkpoints' `runtime.git_commit` is `52f75a6` for #155 (written at 11:09:37) and `341c352` for #156–#187,
    none dirty: `main`'s HEAD at each write, from the coordinator's two docs commits that morning. T4's code commit is `c58afb8`,
    from its token line.
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against the planning estimates:

  | Arm | Tier | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|---|
  | `bc_m` | `maxpressure` | 1 (#155) | 653.8 s | 543.0 s |
  | `bc_top10_m` | `maxpressure` | 5 | 89.3–100.6 s, 93.8 s | 97.0 s |
  | `bc_top10_perix_m` | `maxpressure` | 5 | 92.1–94.9 s, 93.6 s | 97.0 s |
  | `iql_m` | `maxpressure` | 5 | 1,795.3–1,857.1 s, 1,810.7 s | 2,187.3 s |
  | `bc_m` | `fixedtime` | 5 | 446.0–450.4 s, 448.4 s | 543.0 s |
  | `bc_top10_m` | `fixedtime` | 5 | 88.7–103.7 s, 96.4 s | 97.0 s |
  | `bc_top10_perix_m` | `fixedtime` | 5 | 88.9–103.7 s, 92.9 s | 97.0 s |
  | `iql_m` | `fixedtime` | 2 | 1,785.0–1,798.6 s, 1,791.8 s | 2,187.3 s |

  - #155 took 34 % longer than the same arm's four `maxpressure` runs in T3 (478.2–490.4 s). The cause is not measured.
  - The other runs match T3's figures or beat them: `iql_m` ran about 6 % faster. No reported number changes, because the records'
    clock is uncontrolled (A27(a)).
- **`plateaued`** is true for 31 runs and false for #170 and #187 (both `iql_m`). It is recorded per run; no gate of the campaign
  depends on it.
- **The grid's inputs, built inside T4:** `fixedtime` in 52 s, between #170 (13:56:12) and #171 (13:57:04).

## J.2 T5, checked before clearing (CPU, read-only; the listings of `output/p9_1*` and of the corpus root unchanged)
- From the run tree (clean at `c58afb8`), `check --stage train` printed `{"ok": true, "stage": "train"}`.
- `train --hours 6 --plan-only` planned #188 (grid4x4 `iql_m@fixedtime` seed 303) to #211 (grid4x4 `iql_t@mappo1000` seed 101):
  24 runs, 5.98 planned hours, 68 missing, 187 done. That is G.4's T5.
- **`random`, the tier T5 builds for the first time,** from the run tree's code, by the call `train` makes:
  - 3,200 streams;
  - 320 kept by the global and 320 by the per-intersection top-10 % selection;
  - a finite IQL reward scale (0.743);
  - BC windows of 1,152,000 × 20 (115,200 for each top-10 % selection), and the group (40, 8);
  - its corpus `cf_grid4x4__random`. It took 354 s to build in the check.
- **Host memory: T5's margin is the smallest so far.** `train` keeps every tier's inputs until the session ends:
  - T5 holds `fixedtime`'s inputs (#188–#190);
  - it builds `random`'s at #191 while holding `fixedtime`'s;
  - it builds `mappo1000`'s at #211, for the grid's first `iql_t`.

  The coordinator ran that sequence. The process held 11.3 GiB after `fixedtime` and 20.4 GiB after `random`, with a peak of
  31.3 GiB during `random`'s build. With all three tiers it held 23.9 GiB.

  The machine has 47.0 GiB, of which 37.3 GiB were available before the check, and 16 GiB of swap. So T5 fits only if nothing else
  large runs on the laptop's WSL beside it.
- **GPU memory.** T5 trains T4's arms on `random`, and `iql_t` at batch 1,280 (A27(c)), smaller than `iql_m`'s 20,480. T4's peak was
  7,416 MiB of the 16 GB.
- **The hours.** By T4's measured figures per arm, T5 trains 4.98 h (the means) to 5.14 h (the maxima), with `iql_t`'s planned 343 s.
  Building three tiers' inputs adds up to about 10 minutes. The plan's 5.98 h uses the pre-flight's larger figures.

## J.3 ✅ THE TOKEN: training session T5 (the author's tmux `p91`)
**T5 runs about 5.5 hours without a break:** 5.0–5.2 h of training by T4's measured figures (J.2), plus up to about 10 minutes to
build its inputs. It ends by itself at the end of its plan.
- **What it trains:** runs #188–#211, 24 runs.
  - `iql_m` on `fixedtime`, seeds 303–505 (3 runs);
  - A27(b)'s four arms on `random`: `bc_m`, `bc_top10_m`, `bc_top10_perix_m` and `iql_m` (20 runs);
  - the grid's first `iql_t` (A27(c)), on `mappo1000`, seed 101. Its seeds 202–505 open T6.
- **Before it and while it runs:**
  - nothing else trains on the laptop's GPU;
  - no long CPU job and nothing memory-heavy runs on the laptop's WSL beside it (J.2), so P9.3's evaluation waits for T5's end;
  - keep the laptop plugged in and awake.
- **The command** is T4's, with its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T5 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line and the plan (`SESSION: runs #188 … to #211 …: 24 runs, 5.98 planned
  hours`). Then it is quiet until the end. Progress is in `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 24 runs trained …; ended by the plan; 44 runs missing, 10.71 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight, up to about
  37 minutes later (`iql_m` is planned at 36.5 minutes, and measured at 30–31). Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3. A run killed for memory leaves no checkpoint and no record, so it reads as missing: report it.

## J.4 After T5
- **T6 as G.4 planned it would hold four grid tiers at once:** `iql_t` on `mappo1000`, `maxpressure`, `fixedtime` and `random`
  (#212–#230), then `random_fresh`'s `dt_nomix` parts (#231–#234). T5's three tiers already peak at 31.3 GiB.
  - Before T6, the coordinator runs that sequence once T5 has ended; the measurement cannot run beside T5.
  - If it does not fit, T6 is split at a tier boundary by a smaller session bound (`P91_SESSION_HOURS`). That changes no code.
- **Still checked before the session that first builds them:** `random_fresh`'s `dt_nomix` parts (T6) and its grid inputs (T7).
- **Meanwhile:** nothing else on the GPU while T5 runs.

## J.5 2026-11-01: the author's ruling on the order
*"T5 has not started and will not start before tomorrow morning; the laptop is idle tonight. My ruling on the order: P9.3 goes
first."*
- P9.3's evaluation (`BRIEF_48` Amendment J, about 3 h on the CPU) runs first. T5 starts after it ends.
- J.3's clearance and command are unchanged. J.2's memory check stands: T5 runs with nothing large beside it.

---

# ✅ AMENDMENT K — 2026-11-02, evening: T5 VERIFIED from disk; four grid tiers do NOT fit in host memory (measured), so training session T6 is CLEARED as runs #212–#225 at `P91_SESSION_HOURS=1.4` (about 1.4 h without a break), the grid's `iql_t` on three tiers

## K.1 T5, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261102T131153Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the plan (#188–#211, 24 runs, 5.98 planned hours);
  - `train: ok`;
  - the summary (`24 runs trained in 17232.1 s (their records' clock); ended by the plan; 44 runs missing, 10.71 planned hours
    left`);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`.
- **The times (CET, 2026-11-02):**
  - the token at 14:11:53, after P9.3's evaluation (J.5), and the first run started at 14:12:56;
  - the last run finished at 19:03:21, and the capture's last write is 19:03:24;
  - 4.86 h on the wall, and 4.79 h of training on the records' clock: 0.80 of the 5.98 planned hours. J.3 said about 5.5 h.
- **The coordinator's own script** (J.1's, moved to runs 1–211; CPU, read-only):
  - The code's order of all 255 runs equals the manifest written out by hand.
  - **Every record and checkpoint of runs 1–211:** its cell, seed, recipe, label and steps, the checkpoint's sha256 and canonical
    digest, a finite final loss and finite window means. 0 problems.
  - **T5's recipes, written out by hand:** `bc_m` at batch 1,024, `bc_top10_m` and `bc_top10_perix_m` at 102, `iql_m` at 20,480
    with `target_dropout` `off`, and the grid's `iql_t` at 1,280 with `target_dropout` `off` (A27(c)).
  - **T5's 24 runs** ran on `cuda` with OMP and MKL at one thread, 40,000 steps each, in run order, none overlapping, all inside the
    session.
  - **Runs 1–187** are untouched: their files' times are before T5.
  - `training_status`: 255 declared, 211 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 873 samples, the largest gap 20.1 s. Inside runs, utilisation was at least 32 % (median 96 %); memory peaked
    at 7,606 MiB.
  - **G.2 again:** every checkpoint's `runtime.git_commit` is `f939f17`, not dirty. That was `main`'s HEAD from 14:20:56, before
    #188 was written at 14:33:08. T5's code commit is `c58afb8`, from its token line.
- **The inputs, built inside T5:**
  - `fixedtime` before #188, within the 63 s from the token to its start;
  - `random` in 117 s, between #190 (15:17:13) and #191 (15:19:10);
  - `mappo1000` in 58 s, between #210 (18:56:49) and #211 (18:57:47).
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against the planning estimates:

  | Arm | Tier | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|---|
  | `iql_m` | `fixedtime` | 3 (#188–#190) | 1,205.0–1,439.5 s, 1,284.3 s | 2,187.3 s |
  | `bc_m` | `random` | 5 | 482.0–489.0 s, 484.5 s | 543.0 s |
  | `bc_top10_m` | `random` | 5 | 88.4–96.0 s, 92.5 s | 97.0 s |
  | `bc_top10_perix_m` | `random` | 5 | 89.8–94.4 s, 92.3 s | 97.0 s |
  | `iql_m` | `random` | 5 | 1,919.7–1,948.7 s, 1,940.1 s | 2,187.3 s |
  | `iql_t` | `mappo1000` | 1 (#211) | 331.9 s | 342.7 s |

  - #188 and #189 took 1,205–1,209 s, a third less than T4's two `iql_m` runs on `fixedtime` (1,785.0–1,798.6 s). #190 took
    1,439.5 s.
  - During those three runs the GPU's memory was at most 1,863 MiB, against 7,221 MiB for the same arm in T4. They were the
    session's first runs. The cause of either difference is not measured.
  - `iql_m` on `random` took 1,920–1,949 s, about 7 % more than on `maxpressure` in T4.
  - No reported number changes, because the records' clock is uncontrolled (A27(a)).
- **`plateaued`** is true for 17 runs and false for 7: #188–#190, #206, #209 and #210 (`iql_m`), and #211 (`iql_t`). It is
  recorded per run; no gate of the campaign depends on it.

## K.2 T6, checked before clearing (CPU, read-only; the listings of `output/p9_1*` and of the corpus root unchanged)
- From the run tree (clean and detached at `c58afb8`), `check --stage train` printed `{"ok": true, "stage": "train"}`.
- `train --hours 6 --plan-only` planned #212 to #234: 23 runs, 5.68 planned hours, 44 missing, 211 done. That is J.4's T6:
  - `iql_t` on `mappo1000` (4 runs), `maxpressure`, `fixedtime` and `random` (5 each);
  - then `random_fresh`'s `dt_nomix` (4 runs).
- **The memory check.** The coordinator built T6's inputs in `train`'s order and kept each one, as `train` does. A guard thread
  would stop the check when the machine's available memory fell below 5 GiB. Before the check, 41.8 GiB of the 47.0 GiB were
  available, with 16 GiB of swap.

  | After | Built in | Held | Peak | Machine available |
  |---|---|---|---|---|
  | `mappo1000` | 55.6 s | 11.3 GiB | 15.4 GiB | 30.5 GiB |
  | and `maxpressure` | 53.5 s | 16.6 GiB | 20.6 GiB | 25.1 GiB |
  | and `fixedtime` | 48.8 s | 21.8 GiB | 25.8 GiB | 20.1 GiB |
  | and `random` | stopped by the guard | — | 37.8 GiB when stopped | 5.0 GiB |

  - **So T6 as planned does not fit.** `random`'s build needs about 20 GiB above what the process already holds (J.2: from
    11.3 GiB to a 31.3 GiB peak). With three tiers held, that is about 42 GiB.
  - **Each built tier** has 3,200 streams, 320 kept by each top-10 % selection, BC windows of 1,152,000 × 20 (115,200 for each
    top-10 % selection) and the group (40, 8).
    - Each IQL reward scale is finite: 1.295 (`mappo1000`), 3.636 (`maxpressure`) and 2.024 (`fixedtime`).
    - `mappo1000` reads its five teacher-seed corpora.
  - **`random_fresh`'s DT parts were not reached.** They are checked before the session that builds them (K.4).
- **The split, as J.4 set it: T6 ends at #225, before `random`'s build.**
  - `P91_SESSION_HOURS=1.4` plans exactly #212–#225: 14 runs, 1.33 planned hours (4,797.8 s). The bound is 5,040 s, and #226 would
    bring the plan to 5,140.5 s. `train --hours 1.4 --plan-only` printed that plan.
  - No code changes.
  - T6's peak is then the measured 25.8 GiB plus the CUDA context's host memory. That leaves about 16 GiB of the 41.8 GiB,
    less the CUDA context's share.
- **The hours.** At #211's measured 331.9 s per `iql_t` run, T6 trains about 1.3 h, plus about 3 minutes for its three builds.

## K.3 ✅ THE TOKEN: training session T6 (the author's tmux `p91`)
**T6 runs about 1.4 hours without a break:** about 1.3 h of training by #211's measured figure (K.2), plus about 3 minutes to build
its inputs. It ends by itself at the end of its plan.
- **What it trains:** runs #212–#225, 14 runs of the grid's `iql_t` (A27(c)):
  - on `mappo1000`, seeds 202–505 (4 runs);
  - on `maxpressure` and on `fixedtime`, seeds 101–505 (10 runs).
- **Before it and while it runs:**
  - nothing else trains on the laptop's GPU;
  - nothing memory-heavy runs on the laptop's WSL beside it. The coordinator's next memory check waits for its end;
  - P9.3's test runs, one process at a time, may run beside it. T6's margin is about 16 GiB;
  - keep the laptop plugged in and awake.
- **The command** is T5's, with `P91_SESSION_HOURS=1.4` and its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T6 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=1.4 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line (`it runs about 1.4 hours without a break`) and the plan (`SESSION: runs
  #212 … to #225 …: 14 runs, 1.33 planned hours`). Then it is quiet until the end. Progress is in
  `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 14 runs trained …; ended by the plan; 30 runs missing, 9.38 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight, up to about
  6 minutes later (`iql_t` is planned at 5.7 minutes, and measured at 5.5). Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3. A run killed for memory leaves no checkpoint and no record, so it reads as missing: report it.

## K.4 After T6: the rest of the training, in two sessions
Each session is checked before it is cleared, as T6 was. The figures below are planning estimates.
- **T7: #226–#235, 10 runs, 5.32 planned hours, at `P91_SESSION_HOURS=5.4`.**
  - It trains `iql_t` on `random` (5 runs) and `random_fresh`'s `dt_nomix` (5 runs).
  - The bound is 19,440 s, and #236 would bring the plan to 19,691.5 s.
  - It holds `random`'s inputs and `random_fresh`'s DT parts, and builds one stack per DT run. The coordinator measures that after
    T6, as J.4 required for the DT parts.
  - Its hours rest on `dt_nomix`'s planned 3,487 s per run, and are checked against a measured figure before T7 is cleared.
- **T8: #236–#255, 20 runs, 4.06 planned hours, at 6 h.**
  - It trains `random_fresh`'s four A27(b) arms: `bc_m`, `bc_top10_m`, `bc_top10_perix_m` and `iql_m`.
  - It builds `random_fresh`'s grid inputs, which J.4 checks before the session that first builds them.
- After T8, all 255 runs are trained.

---

# ✅ AMENDMENT L — 2026-11-03: T6 VERIFIED from disk; T7's memory measured and the grid's `dt_nomix` TIMED (1.09 of its plan); training session T7 CLEARED (runs #226–#235, about 6 h without a break)

## L.1 T6, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261103T102301Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the session line (`it runs about 1.4 hours without a break`) and the plan (#212–#225, 14 runs, 1.33 planned hours);
  - `train: ok`;
  - the summary (`14 runs trained in 5060.1 s (their records' clock); ended by the plan; 30 runs missing, 9.38 planned hours
    left`), whose next missing run is #226 (`grid4x4:iql_t@random`, seed 101);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`.
- **The times (CET, 2026-11-03):**
  - the token at 11:23:01, and the first run started at 11:24:05;
  - the last run finished at 12:50:28, and the capture's last write is 12:50:31;
  - 1.46 h on the wall, and 1.41 h of training on the records' clock: 1.05 of the 1.33 planned hours. K.3 said about 1.4 h.
- **The coordinator's own script** (K.1's, moved to runs 1–225; CPU, read-only):
  - The code's order of all 255 runs equals the manifest written out by hand.
  - **Every record and checkpoint of runs 1–225:** its cell, seed, recipe, label and steps, the checkpoint's sha256 and canonical
    digest, a finite final loss and finite window means. 0 problems.
  - **T6's recipe, written out by hand:** the grid's `iql_t` at batch 1,280 with `target_dropout` `off` (A27(c)), in all 14 runs.
  - **T6's 14 runs** ran on `cuda` with OMP and MKL at one thread, 40,000 steps each, in run order, none overlapping, all inside the
    session.
  - **Runs 1–211** are untouched: their files' times are before T6.
  - `training_status`: 255 declared, 225 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 262 samples, the largest gap 20.1 s. Inside runs, utilisation was at least 28 % (median 49 %); memory peaked
    at 1,280 MiB.
  - **G.2 again:** the checkpoints' `runtime.git_commit` is `f3a0e05` for #212 (written at 11:29:55) and `dd97de7` for #213–#225,
    none dirty: `main`'s HEAD at each write, since the coordinator's docs commit `dd97de7` landed at 11:35:20. T6's code commit is
    `c58afb8`, from its token line.
- **The inputs, built inside T6:**
  - `mappo1000` before #212, within the 64 s from the token to its start;
  - `maxpressure` in 62 s, between #215 (11:48:30) and #216 (11:49:32);
  - `fixedtime` in 47 s, between #220 (12:20:28) and #221 (12:21:15).
- **Measured seconds per tier** (the records' clock, uncontrolled, A27(a)), against the planning estimate:

  | Arm | Tier | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|---|
  | `iql_t` | `mappo1000` | 4 (#212–#215) | 345.4–387.8 s, 364.9 s | 342.7 s |
  | `iql_t` | `maxpressure` | 5 | 365.5–373.9 s, 370.2 s | 342.7 s |
  | `iql_t` | `fixedtime` | 5 | 341.3–371.8 s, 349.9 s | 342.7 s |

  - All 14 took 341.3–387.8 s, with a mean of 361.4 s: 1.05 of the plan's 342.7 s (P8.2's `iql` median of 298 s, times A27(c)'s
    1.15).
  - The GPU's memory was at most 1,280 MiB, against 7,549 MiB during #211 in T5, which followed `iql_m` (batch 20,480) in the same
    process. As with #188–#190 (K.1), the cause is not measured, and no reported number changes (A27(a)).
- **`plateaued`** is true for 3 runs (#212, #213 and #215) and false for 11. It is recorded per run; no gate of the campaign depends
  on it.

## L.2 T7, checked before clearing (the listings of `output/p9_1*` and of the corpus root unchanged)
- From the run tree (clean and detached at `c58afb8`), `check --stage train` printed `{"ok": true, "stage": "train"}`.
- `train --hours 5.4 --plan-only` planned #226 to #235: 10 runs, 5.32 planned hours (19,148.5 s), 30 missing, 225 done. That is
  K.4's T7:
  - the grid's `iql_t` on `random` (5 runs);
  - then `random_fresh`'s `dt_nomix` (5 runs).

  At 5.3 h the plan ends at #234. At 5.5 h it takes #236, `random_fresh`'s first `bc_m`, whose grid inputs T8 builds after its own
  check (L.4).
- **The memory check.** The coordinator built T7's inputs in `train`'s order and kept each one, as `train` does. That includes
  the IQL table, which `train` builds at the first `iql_t` run and keeps on the inputs. A guard thread would stop the check when
  the machine's available memory fell below 5 GiB. Before the check, 42.4 GiB of the 47.0 GiB were available.

  | After | Built in | Held | Machine available |
  |---|---|---|---|
  | `random`'s grid inputs | 112.2 s | 15.5 GiB (peak 26.3 GiB) | 27.4 GiB (lowest 16.6 GiB, during the build) |
  | and their IQL table (1,152,000 transitions) | 5.1 s | 15.4 GiB | 27.4 GiB |
  | and `random_fresh`'s DT parts | 5.0 s | 15.4 GiB | 27.4 GiB |
  | and one DT run's stack | 30.1 s | 18.9 GiB | 23.8 GiB |
  | during the timing below, on `cuda` | — | 19.2 GiB | 23.8 GiB |

  - **So T7's peak is `random`'s build, at the session's start:** 26.3 GiB, with about 16 GiB of the machine left. For the rest
    of the session it holds about 19 GiB, with about 24 GiB left, the CUDA context included.
  - **`random`'s inputs** are J.2's: 3,200 streams, 320 kept by each top-10 % selection, the group (40, 8) and a finite IQL
    reward scale, from `cf_grid4x4__random`.
  - **`random_fresh`'s DT parts:** 16 nodes, 200 of 200 episodes chosen, 3,200 streams, `state_dim` 40 and `n_actions` 8. A27(d)'s
    distinct-schedule check passed.
  - **The stack:** windows of 1,152,000 × 20 with 40 state features, 4.07 GiB by its arrays' sizes, and the member index
    (72,000, 16).
    - Each DT run builds its own: `train_run` passes `parts["stack"]()` straight to the trainer, so one stack is held at a time.
    - Releasing it took the process from 19.2 to 15.8 GiB.
- **The timing of the grid's `dt_nomix`.** It is the campaign's first measured figure for that arm; K.4 required one before T7.
  - **The run:** the run tree's `train_tier_dt`, on the stack above, on `cuda`, at two budgets.
    - It used the campaign's arguments (batch 64, deterministic off, OMP and MKL at one thread), except the budget, the
      checkpoint path and the log interval.
    - The checkpoints went to the coordinator's scratchpad; they are not runs of the campaign.
  - **The result:**
    - 300 steps took 28.0 s, and 2,300 steps took 218.1 s, measured over the gradient loop, as a run record's seconds;
    - that is 95.0 ms per step, from the difference, which cancels the start-up;
    - a 40,000-step run projects to 3,801 s, 1.09 of the plan's 3,487 s;
    - torch's allocated CUDA memory peaked at 2,382 MiB.
  - **The plan's 3,487 s** is P5.1's median on `mappo1000` (3,305–3,515 s, `docs/data/p8_2_compute.json`). P5.2's runs on the
    grid's other tiers took 3,027–3,988 s.
- **The hours. By the measured figures, T7 trains about 5.8 h:**
  - five `dt_nomix` runs at 3,801 s: 5.28 h;
  - five `iql_t` runs at T6's 341–388 s: 0.47–0.54 h.

  Building `random`'s inputs, `random_fresh`'s parts and five stacks adds about 5 minutes. **That makes about 5.9 h in all**, or
  about 6.2 h if each DT run took P5's slowest 3,988 s.
  - The plan's 5.32 h rests on the planned 3,487 s. So the pane's session line, which prints the bound (`about 5.4 hours`), is
    about half an hour short.
  - The bound stays at 5.4 h, because 5.5 h would take #236 and build `random_fresh`'s grid inputs before their check.
  - Sessions of about 6 h have been cleared under the author's rule since T1 (F.3), so T7 is not split.

## L.3 ✅ THE TOKEN: training session T7 (the author's tmux `p91`)
**T7 runs about 6 hours without a break:**
- about 5.8 h of training by the coordinator's measured figures (L.2);
- plus about 5 minutes to build its inputs.

It ends by itself at the end of its plan. Its pane says "about 5.4 hours": that is the bound, and the measured figures say
about 6.
- **What it trains:** runs #226–#235, 10 runs:
  - the grid's `iql_t` on `random`, seeds 101–505 (5 runs; A27(c));
  - `random_fresh`'s `dt_nomix`, seeds 101–505 (5 runs; A27(d)).
- **Before it and while it runs:**
  - nothing else trains on the laptop's GPU;
  - nothing memory-heavy runs on the laptop's WSL beside it;
  - P9.3's test runs and its preview (`BRIEF_48` J.9), one process at a time, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, may run
    beside it. T7's margin is about 16 GiB during `random`'s build, in its first two minutes, and about 24 GiB after;
  - keep the laptop plugged in and awake.
- **The command** is T6's, with `P91_SESSION_HOURS=5.4` and its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T7 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=5.4 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line (`it runs about 5.4 hours without a break`) and the plan (`SESSION: runs
  #226 … to #235 …: 10 runs, 5.32 planned hours`). Then it is quiet until the end. Progress is in
  `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 10 runs trained …; ended by the plan; 20 runs missing, 4.06 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight:
  - up to about 7 minutes later during the `iql_t` runs;
  - up to about 64 minutes later during the `dt_nomix` runs, which are planned at 58 minutes and measured at about 63.

  Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3. A run killed for memory leaves no checkpoint and no record, so it reads as missing: report it.

## L.4 After T7: T8, the last training session
- **T8: #236–#255, 20 runs, 4.06 planned hours, at `P91_SESSION_HOURS=6`** (K.4).
  - It trains `random_fresh`'s four A27(b) arms: `bc_m`, `bc_top10_m`, `bc_top10_perix_m` and `iql_m`.
  - By T5's measured figures for the same arms on `random`, that is about 3.6 h of training.
- **Before T8 is cleared,** the coordinator checks `random_fresh`'s grid inputs in a fresh process, as T8 builds them (J.4).
- After T8, all 255 runs are trained. P9.1's evaluation is cleared in its own amendment, and P9.3's `old-cells` waits for P9.1's
  merge (`BRIEF_48` J.9).

---

# ✅ AMENDMENT M — 2026-11-04: T7 VERIFIED from disk (`dt_nomix` at 0.98 of its measured projection); `random_fresh`'s grid inputs checked in a fresh process; training session T8, the last, CLEARED (runs #236–#255, about 3.7 h without a break)

## M.1 T7, verified by the coordinator from disk
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261103T152830Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the session line (`it runs about 5.4 hours without a break`) and the plan (#226–#235, 10 runs, 5.32 planned hours);
  - `train: ok`;
  - the summary (`10 runs trained in 20291.1 s (their records' clock); ended by the plan; 20 runs missing, 4.06 planned hours
    left`), whose next missing run is #236 (`grid4x4:bc_m@random_fresh`, seed 101);
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`. The stage's log counts 10 runs trained and 225 skipped as done, and has no warning, error or
  traceback line.
- **The times (CET, 2026-11-03):**
  - the token at 16:28:30, and the first run started at 16:30:04;
  - the last run finished at 22:11:12, and the capture's last write is 22:11:13;
  - 5.71 h on the wall, and 5.64 h of training on the records' clock: 1.06 of the 5.32 planned hours. L.3 said about 6 h.
- **The coordinator's own script** (L.1's, moved to runs 1–235; CPU, read-only):
  - The code's order of all 255 runs equals the manifest written out by hand.
  - **Every record and checkpoint of runs 1–235:** its cell, seed, recipe, label and steps, the checkpoint's sha256 and canonical
    digest, a finite final loss and finite window means. 0 problems.
  - **T7's recipes, written out by hand:**
    - the grid's `iql_t` at batch 1,280 with `target_dropout` `off` (A27(c)), in #226–#230;
    - the grid's `dt_nomix` at 64 joint windows, with no IQL targets and `deterministic` `false` in its record (P5.1 / P5.2's
      registered recipe on A27(d)'s tier), in #231–#235.

    Each T7 checkpoint's A27 provenance (arm, base, targets, registration, point and campaign) equals the hand-written fields.
  - **T7's 10 runs** ran on `cuda` with OMP and MKL at one thread, 40,000 steps each, in run order, none overlapping, all inside
    the session.
  - **Runs 1–225** are untouched: their files' times are before T7.
  - `training_status`: 255 declared, 235 done. No `.partial`, STOP or token is left.
  - **The GPU trace:** 1,026 samples, the largest gap 20.1 s. Memory peaked at 3,428 MiB during `dt_nomix` and at 1,276 MiB
    during `iql_t`; utilisation's median was 95 % and 46 %.
    - 7 samples read 0 % utilisation: one during `random`'s build, before #226, and six in the first 2–22 s of the `dt_nomix`
      runs, one or two in each, while each run builds its own stack on the CPU.
  - **G.2 again:** the checkpoints' `runtime.git_commit` is `63a6dde` for #226 (written at 16:35:51) and `2a0d3d8` for
    #227–#235, none dirty: `main`'s HEAD at each write, since the coordinator's docs commits landed at 16:35:04 and 16:37:20.
    T7's code commit is `c58afb8`, from its token line.
- **The inputs, built inside T7:**
  - `random`'s grid inputs before #226, within the 94 s from the token to its start, and their IQL table inside #226, whose wall
    exceeds its gradient loop by 5.8 s;
  - `random_fresh`'s DT parts in the 5.7 s between #230 (16:58:06) and #231 (16:58:12);
  - each `dt_nomix` run's stack inside that run: its wall exceeds its gradient loop by 31.1–34.9 s (L.2 measured 30.1 s).
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against the planning estimate:

  | Arm | Tier | Runs | Measured: min–max, mean | Planned |
  |---|---|---|---|---|
  | `iql_t` | `random` | 5 (#226–#230) | 330.9–341.1 s, 335.1 s | 342.7 s |
  | `dt_nomix` | `random_fresh` | 5 (#231–#235) | 3,587.5–3,792.9 s, 3,723.1 s | 3,487 s |

  - `dt_nomix`'s mean is 1.07 of the plan and 0.98 of the coordinator's projection of 3,801 s (L.2). All five lie inside P5's
    3,027–3,988 s for the grid's DT runs.
  - `iql_t` on `random` took 0.98 of the plan.
- **`plateaued`** is true for #228 and for all five `dt_nomix` runs, and false for the other four `iql_t` runs. It is recorded
  per run; no gate of the campaign depends on it.
- **Beside T7:**
  - The coordinator's own seven-file pytest run (16:28:01–16:33:12, `BRIEF_48` J.10) ran during `random`'s build and the first
    three minutes of #226, as L.3 allows.
  - #226 took 341.1 s, the slowest of the five `iql_t` runs and 3 % above the fastest. Its seconds are uncontrolled (A27(a)),
    and no number changes.
  - The coordinator knows of no other run beside T7. P9.3's branch has no commit since `cb6a942`.

## M.2 T8, checked before clearing (the listings of `output/p9_1*` and of the corpus root unchanged)
- **Before the check:**
  - No trainer or pytest process ran; this was checked outside the sandbox, read-only.
  - The GPU held no compute process, and 42.6 GiB of the machine were available.
  - The listing of `output/p9_1*` differs from the one taken after L.2's check by T7's 23 new paths, the capture's growth and
    four directories' times. The 23 are 10 checkpoints, 10 records, and the session's log directory with its two files. No path
    is missing, and the corpus listing is unchanged.
- **The module's own checks.** From the run tree (clean and detached at `c58afb8`):
  - `check --stage train` printed `{"ok": true, "stage": "train"}`.
  - `train --plan-only` at 4.1 h and at 6 h planned the same: #236 to #255, 20 runs, 4.06 planned hours (14,621.5 s), 20
    missing, 235 done. At 4.0 h the plan ends at #254.
- **The plan's 20 runs** are `random_fresh`'s four A27(b) arms, seeds 101–505 each, in the hand-written order:
  - #236–#240 `bc_m`;
  - #241–#245 `bc_top10_m`;
  - #246–#250 `bc_top10_perix_m`;
  - #251–#255 `iql_m`.

  Their recipes equal A27(b)'s, written out by hand: 1,024, 102 and 102 windows, and 20,480 transitions with A27(c)'s targets;
  40,000 steps each.
- **The memory check, in a fresh process.** The coordinator built T8's inputs as `train` builds them at #236:
  - A27(d)'s distinct-schedule check, then `tier_sweep.baselines_inputs`;
  - kept with their IQL table, which `train` builds at #251 and keeps on the inputs.

  A guard thread would stop the check below 5 GiB available; 42.4 GiB were available before it.

  | After | Built in | Held | Machine available |
  |---|---|---|---|
  | `random_fresh`'s grid inputs | 57.1 s | 11.4 GiB (peak 15.4 GiB) | 31.3 GiB (lowest 27.4 GiB, during the build) |
  | and their IQL table (1,152,000 transitions) | 2.6 s | 11.4 GiB | 31.4 GiB |

  - **The inputs:**
    - A27(d)'s distinct-schedule check passed;
    - 3,200 streams, from `cf_grid4x4__random_fresh`, with 320 kept by each top-10 % selection;
    - the group (40, 8) and a finite IQL reward scale;
    - the windowed batches: 1,152,000 windows for `bc` (4.06 GiB), and 115,200 for each top-10 % arm (0.41 GiB each).
  - **So T8 peaks at its start, at 15.4 GiB, with about 27 GiB of the machine left.** That is lower than `random`'s 26.3 GiB
    (L.2). `random`'s directory holds 400 episodes and `random_fresh`'s 200, and each tier trains on 3,200 streams. The
    coordinator did not measure the cause further.
- **The hours.** T8 trains the four arms that T5 trained on `random` (#191–#210), on inputs of the same size: 3,200 streams, and
  1,152,000 windows and transitions.
  - Those 20 runs took 13,047.3 s on the records' clock, 3.62 h: 0.89 of their plan. From first start to last end they took
    3.63 h.

  | Arm | On `random` in T5: min–max, mean | Planned |
  |---|---|---|
  | `bc_m` | 482.0–489.0 s, 484.5 s | 543.0 s |
  | `bc_top10_m` | 88.4–96.0 s, 92.5 s | 97.0 s |
  | `bc_top10_perix_m` | 89.8–94.4 s, 92.3 s | 97.0 s |
  | `iql_m` | 1,919.7–1,948.7 s, 1,940.1 s | 2,187.3 s |

  - **So T8 runs about 3.7 h:** 3.6 h of training and about a minute for its inputs. At its plan's figures it would take about
    4.1 h.
  - **The bound is 4.1 h, not K.4's and L.4's 6 h.**
    - In the train stage the bound only chooses the plan. There is no deadline: the session ends at its plan's end or at STOP.
    - 4.1 h and 6 h plan the same 20 runs (checked above).
    - The pane's session line prints the bound, so it now says about 4.1 hours instead of 6.

## M.3 ✅ THE TOKEN: training session T8 (the author's tmux `p91`)
**T8 runs about 3.7 hours without a break:**
- about 3.6 h of training, by T5's measured figures for the same arms on `random` (M.2);
- plus about a minute to build its inputs.

It ends by itself at the end of its plan. Its pane says "about 4.1 hours": that is the bound. T8 is the last training session;
after it, all 255 runs are trained.
- **What it trains:** runs #236–#255, 20 runs: `random_fresh`'s `bc_m`, `bc_top10_m`, `bc_top10_perix_m` and `iql_m`, seeds
  101–505 each (A27(b) on A27(d)'s tier).
- **Before it and while it runs:**
  - nothing else trains on the laptop's GPU;
  - nothing memory-heavy runs on the laptop's WSL beside it;
  - P9.3's test runs may run beside it, one process at a time, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. T8's margin is
    about 27 GiB during its inputs' build, in its first minute, and about 31 GiB after;
  - keep the laptop plugged in and awake.
- **The command** is T7's, with `P91_SESSION_HOURS=4.1` and its own refusal text. At the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: T8 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_train && P91_SESSION_HOURS=4.1 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh train c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/train_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows the token line, the session line (`it runs about 4.1 hours without a break`) and the plan (`SESSION: runs
  #236 … to #255 …: 20 runs, 4.06 planned hours`). Then it is quiet until the end. Progress is in
  `output/p9_1_runs/logs/<stamp>/train.log`.
- **Done when** the capture shows:
  - `train: ok`;
  - the summary (`SESSION: 20 runs trained …; ended by the plan; 0 runs missing, 0.00 planned hours left`);
  - `stage train <stamp>: SESSION COMPLETE …`.

  The pane must also show `DRIVER EXIT 0`.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. The session ends after the run in flight:
  - up to about 8 minutes later during the `bc_m` runs, and about 2 minutes during the top-10 % arms;
  - up to about 33 minutes later during the `iql_m` runs.

  Remove the file by hand before the next session.
- **A failure, and Ctrl+C:** as F.3. A run killed for memory leaves no checkpoint and no record, so it reads as missing: report it.

## M.4 After T8
- The coordinator verifies T8 from disk, as M.1 does T7. All 255 runs are then trained.
- P9.1's evaluation is cleared in its own amendment, after its own check.
- P9.3's `old-cells` waits for P9.1's merge (`BRIEF_48` J.9).

---

# ✅ AMENDMENT N — 2026-11-04, evening: T8 VERIFIED from disk — P9.1's TRAINING COMPLETE, 255 of 255 runs; the evaluate stage's check UNDER WAY (cleared in Amendment O)

## N.1 T8, verified by the coordinator from disk: P9.1's training is COMPLETE
- **The capture** (`output/p9_1_runs/train_capture.txt`) shows, in order:
  - the token line (`stage train 20261104T150356Z from /home/filip/rltraffic-p91-run-d at c58afb8`);
  - the session line (`it runs about 4.1 hours without a break`) and the plan (#236–#255, 20 runs, 4.06 planned hours);
  - `train: ok`;
  - the summary (`20 runs trained in 12373.8 s (their records' clock); ended by the plan; 0 runs missing, 0.00 planned hours
    left`), whose next missing run is `null`;
  - `SESSION COMPLETE`.

  The author saw `DRIVER EXIT 0`. The stage's log counts 20 runs trained and 235 skipped as done, and has no warning, error or
  traceback line.
- **The times (CET, 2026-11-04):**
  - the token at 16:03:56, and the first run started at 16:04:59;
  - the last run finished at 19:31:25, and the capture's last write is 19:31:28;
  - 3.46 h on the wall, and 3.44 h of training on the records' clock: 0.85 of the 4.06 planned hours. M.3 said about 3.7 h.
- **The coordinator's own script** (M.1's, moved to all 255 runs; CPU, read-only):
  - The code's order of all 255 runs equals the manifest written out by hand.
  - **Every record and checkpoint of all 255 runs:** its cell, seed, recipe, label and steps, the checkpoint's sha256 and canonical
    digest, a finite final loss and finite window means. 0 problems.
  - **T8's recipes, written out by hand from A27(b):** `bc_m` at 1,024 windows, `bc_top10_m` and `bc_top10_perix_m` at 102, and
    `iql_m` at 20,480 transitions with `target_dropout` `off`. None of their records has a `deterministic` field, which only the DT
    records carry.

    Each T8 checkpoint's A27 provenance (arm, base, targets, registration, point and campaign) equals the hand-written fields. Its
    registration is `A27(b)` for the three BC arms and `A27(b) with A27(c)'s targets` for `iql_m`, and its point is `A27(d)`.
  - **T8's 20 runs** ran on `cuda` with OMP and MKL at one thread, 40,000 steps each, in run order, none overlapping, all inside the
    session.
  - **Runs 1–235** are untouched: their files' times are before T8.
  - `training_status`: 255 declared, **255 done**. No `.partial`, STOP or token is left.
  - **The GPU trace:** 622 samples, the largest gap 20.2 s. Memory peaked at 7,827 MiB during `iql_m` and at 7,824 MiB during
    `bc_m`, of the GPU's 16,303 MiB. Utilisation's median was 96 % inside the runs. The one sample at 0 % lies outside every run.
  - **G.2 again:** the checkpoints' `runtime.git_commit` is `dc99753` for all 20, none dirty: `main`'s HEAD throughout T8, since the
    coordinator's docs commit landed at 14:01:50, before it. T8's code commit is `c58afb8`, from its token line.
- **The inputs, built inside T8:** `random_fresh`'s grid inputs in the 63.9 s between the token and #236 (M.2 measured 57.1 s), and
  their IQL table inside #251, whose wall exceeds its gradient loop by 3.1 s.
- **Measured seconds per arm** (the records' clock, uncontrolled, A27(a)), against T5's figures on `random` (M.2) and the plan:

  | Arm | Runs | On `random_fresh` (T8): min–max, mean | On `random` (T5): mean | Planned |
  |---|---|---|---|---|
  | `bc_m` | #236–#240 | 455.8–481.7 s, 473.3 s | 484.5 s | 543.0 s |
  | `bc_top10_m` | #241–#245 | 82.5–87.0 s, 84.8 s | 92.5 s | 97.0 s |
  | `bc_top10_perix_m` | #246–#250 | 82.9–85.6 s, 83.9 s | 92.3 s | 97.0 s |
  | `iql_m` | #251–#255 | 1,786.8–1,883.4 s, 1,832.8 s | 1,940.1 s | 2,187.3 s |

  The 20 runs took 0.95 of the same arms' 13,047.3 s on `random`.
- **`plateaued`** is true for the 15 BC-family runs and false for the five `iql_m` runs. It is recorded per run; no gate of the
  campaign depends on it.
- **Beside T8:** the coordinator ran nothing, and knows of no other run. P9.3's two docs commits (15:06 and 15:08) came before it.
- **The whole training:** 255 runs from 16:11 CET on 2026-10-28 (run #1) to 19:31 CET on 2026-11-04 (run #255), in the campaign
  stage's three runs and sessions T1–T8. They took 36.00 h on the records' clock, 0.89 of the 40.55 planned hours.

## N.2 The coordinator's error: a directory created under `output/`
- To read T8's log, the coordinator ran `cd` into `output/p9_1_runs/logs/20261104T150356Z` in its main shell, against its own rule.
  It ran only reading commands there (`ls`, `wc`, `grep`, `head`, `tail`) and left at once.
- **The harness then created an empty `.claude/.cc-writes` directory in that log directory,** at 20:44:11. Nothing else changed:
  `train.log` and `gpu_trace.csv` keep their sizes and their times (19:31:26 and 19:31:24), as N.1 read them.
- **The directory stays where it is, recorded here, until the author rules on it.** Nothing under `output/` is deleted without that
  ruling. Seven older empty `.claude` directories lie under `output/`, from earlier sessions (2026-10-27 to 2026-10-29).

## N.3 Next
- **The evaluate stage is cleared in its own amendment (O), after its check, which is under way.** F.2-2 asked P9.1 to report how
  long a grid route (i) cell takes before the evaluate sessions are cleared. P9.1's branch has no commit since `9eb908a`
  (2026-10-28), so the coordinator is timing the units itself, as L.2 timed `dt_nomix`'s training. Nothing starts before O.
- **Meanwhile, for P9.1, on its branch** (nothing in the run tree): Q18 (F.2-3, E.7's G4 diagnostics, tests first), F.2-4's
  docstring, and G.2's ruling in the report. They were set on 2026-10-29, they are needed before G4, and the branch has no commit
  since. Tests run on the CPU only: one process, one thread, nothing on the GPU.
- **P9.3** waits for P9.1's merge (`BRIEF_48` J.9, J.11).

---

# ✅ AMENDMENT O — 2026-11-04, night: the evaluate stage CHECKED and its units TIMED by the coordinator (F.2-2); evaluate session E1 CLEARED (`P91_SESSION_HOURS=4`: about 4 h without a break, at most about 4 h 40 min)

## O.1 The evaluate stage, checked before clearing (read-only on `output/` and the corpus)
- **What it does,** read in the run tree's driver and module at `c58afb8`. Each session runs, in order:
  1. **`declare-evaluation`,** in one process. The first session writes three manifests, and a later one checks them unchanged.
     They declare 13,500 hz1x1 and 13,000 grid route (ii) episodes, and 18,200 old-cell episodes. None is on disk yet.
  2. **`evaluate`,** in six shards. Each shard takes every sixth of the 53 new cells, sorted by scenario, tier and arm, so the grid's
     cells come first. Each cell runs:
     - **route (i):** the CLI's own `evaluate`, 500 episodes (5 seeds × 100 held-out draws) in one call, which cannot stop inside;
     - **route (ii):** P8.4b's runner on the same 500 episodes, each reproducing route (i)'s `att_ours` under `==`. It can stop
       between episodes;
     - **P3a,** for its four new cells: the 500 episodes rolled once more, each alone, with their actions recorded. This is one unit,
       which cannot stop inside.
  3. **`old-cells`,** in six shards: the 38 registered hz1x1 cells, 18,200 episodes, each reproducing its committed `att_ours` under
     `==`. They can stop between episodes, and they start only after all six `evaluate` shards have ended.
  4. **`p3a-references`,** in one process: P3a's two registered random references, 500 episodes each. It can stop between the two.
  5. **`status`:** `COMPLETE`, or the units left.

  A session ends at `P91_SESSION_HOURS` or at the stop file, **plus the unit in flight in each shard** (F.2-2). Any refusal stops
  the stage: a disagreement between the routes is a finding, not a cell to skip.
- **The shards** (`shard_of`, every sixth of the sorted cells):
  - **shard 0:** four grid cells; then the grid's `dt_nomix@random_fresh` (route (i), route (ii) and P3a: 1,500 grid DT episodes);
    then four hz1x1 cells;
  - **shards 1–5:** four or five grid cells, then four or five hz1x1 cells. P3a's other new cells are shard 1's hz1x1
    `behaviour@random_fresh`, shard 2's hz1x1 `dt@random_fresh` and shard 5's grid `behaviour@random_fresh`.
- **The module's own checks.** From the run tree (clean and detached at `c58afb8`):
  - `check --stage evaluate` printed `{"ok": true, "stage": "evaluate"}`;
  - `status` reads 255 of 255 runs trained, 0 of 53 route (i) cells, no route (ii) or old-cell manifest yet, and 0 of 6 P3a cells.
- **The listings.** The corpus listing is unchanged. Under `output/p9_1*`, the coordinator's checks and timing changed nothing
  after N.2's directory.

## O.2 F.2-2: how long the units take, MEASURED by the coordinator
F.2-2 asked P9.1 to report how long a grid route (i) cell takes, so that the evaluate sessions' hours could be stated first. P9.1's
branch has no commit since `9eb908a` (2026-10-28), so the coordinator timed the units itself, as L.2 timed `dt_nomix`'s training.
- **The method: the pre-flight's own form** (`_preflight_a27_grid_path`):
  - route (i) by the campaign's own argv (`route_i_command`), with `--seeds 101` and the first held-out draws through
    `narrowed_held_out_draws`; for hz1x1, its two training aggregates were rebuilt from the run records into the scratchpad;
  - route (ii) by `run_campaign` with the campaign's `a27_factory_builder`, and P3a by `record_episode_actions`;
  - the old cells by `run_campaign` with P8.4b's own builder, on their committed values.

  The checkpoints were read from `output/p9_1_runs/checkpoints`, and every file was written under the coordinator's scratchpad. The
  runs were on `cuda`, with OMP and MKL at one thread, outside the sandbox. The scripts printed seconds, counts and flags only, and no
  outcome value was read.
- **Every timed episode reproduced its value under `==`.** 1,530 route (ii) episodes (30 on the grid and 1,500 on hz1x1) and 20 P3a
  episodes reproduced route (i), and 40 old-cell episodes reproduced their committed `att_ours`. No episode was refused.
- **The grid, alone** (10 episodes each, draws 1000–1009), **and six processes at once** (`dt_nomix` for 10 episodes, five BC and
  IQL arms on `fixedtime` for 40 each):

  | Unit | Alone, s per episode | Six at once, s per episode | One 500-episode unit, six at once |
  |---|---|---|---|
  | `dt_nomix@random_fresh`: route (i) / route (ii) / P3a | 3.13 / 2.91 / 2.72 | 4.23 in route (i) | route (i) about 35 min; P3a about 32 min |
  | `bc_m@fixedtime`: route (i) / route (ii) | 1.98 / 1.96 | 2.78 in route (i) | route (i) about 23 min |
  | `bc_top10_m` and `bc_top10_perix_m` (`fixedtime`): route (i) | — | 2.78 | about 23 min |
  | `iql_m@fixedtime` and `iql_t@fixedtime`: route (i) | 2.23 (`iql_m`) | 2.86 and 2.87 | about 24 min |
  | `behaviour@random_fresh`: route (i) / route (ii) / P3a | 1.40 / 1.45 / 1.53 | — | about 16–18 min |

  - **Six at once is 1.28–1.41 times slower than alone:** `iql_m` 1.28, `dt_nomix` 1.35 and `bc_m` 1.41.
  - `behaviour@random_fresh` ran beside one other process, the coordinator's hz1x1 timing.
  - With six processes, the GPU held at most 2,171 MiB of its 16,303, at a median utilisation of 27 %. The host kept at least
    37.6 GiB available, and no process exceeded 1.26 GiB.
  - **C.4's smoke figures are not the campaign's.** On its two-step checkpoints, the grid's `dt_nomix` took 9.3 s per route (ii)
    episode; the trained checkpoints take 2.9–3.1 s.
- **hz1x1, alone,** over whole cells of 500 episodes (O.3-6), **and the old cells, alone,** over the first 10 committed episodes of
  four cells:

  | Unit | Alone, s per episode | One 500-episode unit, six at once (× 1.40) |
  |---|---|---|
  | hz1x1 `dt@random_fresh`: route (i) / route (ii) | 1.85 / 1.81 | route (i) about 22 min |
  | hz1x1 `iql_t@mappo1000`: route (i) / route (ii) | 1.05 / 1.06 | route (i) about 12 min |
  | hz1x1 `bc_100k@mappo1000`: route (i) / route (ii) | 0.95 / 0.98 | route (i) about 11 min |
  | old cells `iql@maxpressure`, `madt`, `mappo1000` and `behaviour@fixedtime` | 1.01, 1.73, 1.69 and 0.66 | per episode |

  - **The old cells take 6.05 h alone in all, 0.78–1.23 h per shard.** This weights each old cell's committed episodes by its
    family's rate: the BC and IQL arms as measured, the registered DTs as `madt`, the MAPPO behaviour cells as `mappo1000`, and the
    rule-based ones as `behaviour@fixedtime`.
  - The hz1x1 `behaviour@random_fresh` cell is not timed. It is taken at 1.0 s per episode.
- **The projection.** The coordinator's script runs each shard's units in the module's order, at these rates. While n shards run,
  each is slowed by 1 + k (n − 1). With k from 0.07 to 0.09, that is 1.35–1.45 for six, around the measured 1.28–1.41.
  - **The whole stage takes about 8.0–8.5 h.**
    - The six `evaluate` shards end after 4.8–6.5 h.
    - The old cells wait for the last of them, and end about 1.6–1.7 h later.
    - P3a's references take about 0.35 h after that.
  - **At a 4-h deadline,** each shard is either between two route (ii) episodes or finishing an hz1x1 route (i) cell with minutes
    left. So the session ends at about 4.1 h, with 0.6–1.9 h of alone-work left per shard.
  - **The next session** runs the rest of `evaluate` (about 2.0–2.4 h), then the old cells and P3a's references: about 3.9–4.5 h
    in all.
  - The grid's rates come from its first 10 or 40 held-out draws, and the draws' demand varies. E1's own records will show by
    how much.

## O.3 Rulings
1. **F.2-2 is answered by O.2's measurement.** P9.1's own report of it is no longer needed.
2. **The evaluation runs in sessions of about four hours.**
   - E1 is cleared now at `P91_SESSION_HOURS=4`, with six workers (the cap).
   - E2 is cleared after E1 is verified, with its hours from E1's own records: the route (i) files' times and the shards' logs.
   - A third, short session follows only if E2 would otherwise run past about five hours.
3. **E1's hours.** It runs about 4 hours without a break.
   - At its deadline, the projection finds each shard either between two route (ii) episodes or minutes from the end of an hz1x1
     route (i) cell. So E1 should end at about 4.1 h.
   - The longest unit that cannot stop is the grid's `dt_nomix` route (i), about 35 minutes with six shards running. So E1 runs at
     most about 4 h 40 min, unless a stall restart re-rolls a unit that was in flight at the deadline.
4. **The command checks CUDA first**, as the training's did. The workers run their policies on the GPU (`device` None), as the
   coordinator's timing did. On the CPU they would be slower than every figure above.
5. **Beside the evaluation:**
   - no training and nothing else on the GPU;
   - nothing memory-heavy on the laptop's WSL;
   - P9.1's own work (O.5) may run one pytest process at a time, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`;
   - P9.3 runs nothing (`BRIEF_48` J.11).

   Its values do not depend on the machine's load, only its hours do.
6. **The coordinator's error in the timing.** `method_tier_grid` binds the held-out draws when it is imported, so
   `narrowed_held_out_draws` does not narrow them there. The coordinator's hz1x1 timing therefore rolled each cell's 500 episodes
   instead of 10: about an hour of one CPU core, every file in its scratchpad, and no value read. The campaign never narrows draws,
   so nothing in it changes.
7. **The coordinator's timing outputs are kept** in its scratchpad: route (i) for three whole hz1x1 cells, and the grid's first
   draws. At E1's verification the coordinator compares them with E1's own route (i) files by program, printing only counts of equal
   episodes: the same code, checkpoints and seeds must give the same values. `/tmp` does not survive a restart of WSL; if they are
   lost, that check is skipped and said so.

## O.4 ✅ THE TOKEN: evaluate session E1 (the author's tmux `p91`)
**E1 runs about 4 hours without a break, and at most about 4 h 40 min** (O.3-3). It ends by itself.
- **What it runs:**
  - `declare-evaluation`, which writes the three manifests once;
  - the six `evaluate` shards until the deadline, 4 h after the session's start. Each shard then finishes its unit in flight.

  The old cells and P3a's references come in a later session.
- **Before it:**
  - the coordinator's timing has ended (at 22:04:46), and no other P9.1 stage runs;
  - nothing runs on the GPU;
  - keep the laptop plugged in and awake.
- **The command,** at the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: E1 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_evaluate && P91_SESSION_HOURS=4 WORKERS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh evaluate c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/evaluate_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows:
  - the token line;
  - the session line (`evaluate session: it runs about 4 hours without a break (P91_SESSION_HOURS), PLUS the unit of work in
    flight in each shard …`). Its "7 min 47 s" is the smoke's hz1x1 cell: O.2 has the campaign's figures;
  - `declare_evaluation: ok`.

  Then it is quiet until the deadline. Progress is in `output/p9_1_runs/logs/<stamp>/evaluate_shard<k>.log`.
- **Done when** the capture shows:
  - `evaluate: 6 shards ended cleanly`;
  - the `left:` line (route (i) cells, route (ii) episodes per scenario, P3a cells and old-cell episodes left);
  - `stage evaluate <stamp>: SESSION ENDED at the hours bound (4 h) …`.

  The pane must also show `DRIVER EXIT 0`. If everything is done, the last line is `stage evaluate <stamp>: COMPLETE …` instead.
- **To end it early:** `touch /home/filip/rltraffic/output/p9_1_runs/STOP`. Each shard ends after its unit in flight: seconds in a
  route (ii), up to about 35 minutes in a grid route (i) or P3a. The session then ends `at the stop file`. Remove the file by hand
  before the next session.
- **A failure:** `evaluate shard <k> FAILED` with the last lines of its log, then `FAILED: …` and `DRIVER EXIT 1`. Report it; nothing
  is re-run without a ruling. A route (ii) episode that does not reproduce route (i) is a finding.
- **A stall** (DEFERRED 104): a shard with no progress for 120 s is restarted, at most 60 times, and each restart is logged in
  `restarts.log`. A restart rolls the unit in flight again from scratch, with the same seeds and values.
- **Ctrl+C is the emergency stop only.** It loses the units in flight.

## O.5 After E1, and meanwhile
- **After E1:** the coordinator verifies it from disk (the route (i) files, the route (ii) episodes, P3a, the restarts, the logs),
  then clears E2, its hours first.
- **Meanwhile, for P9.1, on its branch** (nothing in the run tree): Q18 (F.2-3, E.7's G4 diagnostics, tests first), F.2-4's
  docstring, and G.2's ruling in the report. These were set on 2026-10-29 and are needed before G4; the branch has no commit since.
  Tests run CPU only, one process, one thread, nothing on the GPU.
- **P9.3** waits for P9.1's merge (`BRIEF_48` J.9, J.11).

---

# ✅ AMENDMENT P — 2026-11-05, evening: evaluate session E1 VERIFIED from disk; E2 CLEARED (`P91_SESSION_HOURS=4`: about 4 h without a break, at most about 4 h 30 min); a third session, E3, of about 1.5–1.8 h follows (O.3-2)

## P.1 E1, verified from disk (read-only; the scripts printed counts, flags, provenance and seconds; no outcome value was read)
- **The session** ran from the run tree at `c58afb8`:
  - token at 11:10:34 CET (stage `20261105T101034Z`), deadline 15:10:34, the six shards started at 11:10:36;
  - three shards ended at the deadline, between two route (ii) episodes, 0.9–2.6 s after it;
  - three finished the hz1x1 route (i) cell in flight, 179, 309 and 331 s after it;
  - the capture's last line was written at 15:16:08: **4 h 05 min 35 s after the token** (O.3-3: about 4.1 h, at most about
    4 h 40 min).

  The capture holds the token line, the session line, `declare_evaluation: ok`, `evaluate: 6 shards ended cleanly`, the `left:`
  line and `SESSION ENDED at the hours bound (4 h)`. The author's pane showed `DRIVER EXIT 0`.
- **The 53 cells and the six shards,** written out by hand (the cells, sorted, every sixth), equal the code's.
- **Route (i): 33 of 53 cell files,** the grid's 26 and hz1x1's 7. Each was checked for:
  - 500 episodes over the 5 seeds × 100 held-out draws, of its own arm, every value a finite float;
  - engine seed 1000, the declared steps, the tier, and the cell's seeds and draws;
  - its mean, recomputed from its episodes, equal to within 10⁻¹² relative;
  - a write inside E1.

  **Provenance.** Each grid file names each seed's checkpoint (125 of 125), and its weight digest equals the digest recomputed from
  the checkpoint, both by `tier_sweep`'s function and by the coordinator's own loop (125 of 125). Each checkpoint's file sha256
  equals its training record (125 of 125). The hz1x1 files record `cuda`, one thread, and main's HEAD at the write (`25410f4`), as the
  checkpoints do (G.2). The code that ran is the run tree's at `c58afb8`, which the driver checks before it starts. The training
  aggregates that route (i) wrote for hz1x1 (`output/p9_1/p9_1_training.json`, 90 runs; `p9_1_100k_training.json`, 40) equal their
  rebuild from the run records.
- **Route (ii).** Both manifests equal the module's and the hand-written cell lists (13,500 hz1x1 and 13,000 grid episodes, engine
  seed 1000), and were written once, at 11:10:36. There are 14,405 episode files (the grid's 12,508, hz1x1's 1,897) and **0
  refused**. For every one:
  - it is declared, and its names are its file's;
  - `reproduces_committed` is true, `att_ours == committed_att_ours`, and `committed_att_ours ==` route (i)'s `att_horizon` for the
    same episode;
  - it was written inside E1.

  Every trained arm's episode names its run's checkpoint (13,905), and the grid's behaviour cell is algorithmic (500).
- **P3a: 1 of 6 cells,** the grid's `behaviour@random_fresh`, written at 14:25:45. It has 500 episodes over the 5 × 100 keys, and
  `att_ours ==` route (i) in 500 of 500. Each episode's action-sequence sha256 is well formed (500 distinct), and so is each of its
  16 per-intersection sha256s. The other three new cells are E2's; the two registered references are `p3a-references`'.
- **The old cells** are declared (18,200 episodes, the manifest equal to the module's). None was rolled: they start only after the
  last `evaluate` shard (O.1).
- **The status, recomputed,** equals the capture's `left:` line and the session's `status.json`:
  - route (i): 20 of 53 cells left;
  - route (ii): 492 grid and 11,603 hz1x1 episodes left;
  - P3a: 5 of 6 cells left;
  - old cells: 18,200 of 18,200 episodes left.
- **The logs.** Each shard's log holds only:
  - the CLIs' seed lines and the grid's `episodes written` lines;
  - route (ii)'s progress lines, with `refused=0` throughout;
  - the module's result: stopped by the hours bound, and the cells left;
  - hz1x1 route (i)'s summary lines. These print means, so they were counted and not read.

  There is no other line: no warning, error or traceback. There is no `restarts.log`, and the six stall files are empty. No STOP,
  token, `.partial` or `.tmp` file is left.
- **Nothing else changed.**
  - Training: 255 of 255; no checkpoint or record was written during E1.
  - Under `output/p9_1*`, against the listing taken after the coordinator's timing: 572 of the 575 paths are unchanged. The other
    three are directories that gained E1's files, and none was removed. The only new files are E1's: route (i), route (ii), P3a,
    the old cells' manifest, the logs, the capture and the two aggregates.
  - Nothing else under `output/` was written after E1's start, and the corpus listing is unchanged.
  - After E1, outside the sandbox, read-only: no P9.1, P9.3 or pytest process, and no process on the GPU.
- **O.3-7's check.** The coordinator's route (i) outputs of 2026-11-04 equal E1's route (i) files, episode by episode, in
  `att_horizon`, `episode_reward` and `horizon_vehicle_count` under `==`:
  - the grid: the first 10 or 40 draws of seed 101, in ten outputs over seven cells, three of them timed both alone and six at once
    (250 episodes);
  - hz1x1's whole `bc_100k@mappo1000` cell (500).

  The same code, checkpoints and seeds gave the same values alone, six at once and in the campaign. hz1x1's other two timed cells,
  `dt@random_fresh` and `iql_t@mappo1000`, are E2's; their outputs are kept for E2's check.

## P.2 E1's hours, and the coordinator's error in O.2's projection
- **The grid ran as O.2 expected:** six at once, 1.32–1.56 times slower than alone.
  - BC and IQL arms: route (i) and route (ii) 2.75–3.27 s per episode, so 23–27 min per 500-episode unit;
  - `behaviour@random_fresh`: 1.97–2.18 s per episode;
  - `dt_nomix@random_fresh`: route (i) 4.73 s per episode, 39.4 min (O.2: about 35 min); route (ii) 4.54 s.
- **hz1x1 did not.** With six shards running (mostly four hz1x1 beside two grid), hz1x1's BC and IQL cells ran 2.17–2.26 times
  slower than alone, and its DT 1.63–1.64 times:
  - `bc_100k@mappo1000`, the very cell O.2 timed alone, took 2.10 s per route (i) episode against 0.95 alone, and 2.16 per route
    (ii) episode against 0.98;
  - `iql_t@fixedtime` took 2.29 / 2.39 (against O.2's `iql_t@mappo1000` alone, 1.05 / 1.06);
  - `dt_100k@mappo1000` took 3.03 / 2.97 (against O.2's `dt@random_fresh` alone, 1.85 / 1.81).

  O.2 took 1.35–1.45 for every unit. **That was the coordinator's error: it carried the grid's six-at-once factor over to hz1x1
  without measuring hz1x1 six at once.** Why hz1x1 is hit harder was not measured. It changes the hours, not the values (P.1:
  every value reproduced under `==`, alone, six at once and in the campaign).
- **The consequence.** E1 itself ended as O.3-3 said. The rest, though, takes longer than O.2's "E2 about 3.9–4.5 h".
  - The coordinator's projection, rerun with E1's rates (and again with every slowdown 15 % larger), gives about 5.5–5.8 h without a
    deadline:
    - the `evaluate` shards end after 1.9–3.1 h;
    - the old cells end about 2.2–2.3 h after the last of them;
    - P3a's references take 0.35 h after that.
  - The old cells' weighting reproduces O.2's 0.78–1.23 alone-hours per shard.

  That is past O.3-2's five hours, so **a third session, E3, follows**, as O.3-2 provides.
- **E1 ran beside P9.1's morning work.** P9.1's branch gained three commits during E1: `8ccea70` at 11:11 (a merge of main at
  `25410f4`), `8ab9761` at 11:37 and `11616ea` at 11:44 (Q18, F.2-4 and G.2, with their test runs). O.3-5 allowed this, and E1's
  first grid cells (11:10–11:58) ran at the later cells' rates. The coordinator verifies those commits in their own amendment,
  before G4. Nothing in E2 depends on them: the run tree stays at `c58afb8`.

## P.3 Rulings
1. **E1 is verified.**
2. **E2 runs at `P91_SESSION_HOURS=4`, with six workers, and E3 follows it.** The evaluation's sessions stay at about four hours
   (O.3-2), and the whole rest, about 5.5–5.8 h, is past five.
3. **E2's hours.** By the projection, from E1's rates and with every slowdown 15 % larger:
   - the six `evaluate` shards end at about 3.0–3.1 h;
   - the old cells then run until the deadline and stop between episodes, with about 40–50 % of their work done;
   - so E2 ends at about 4.0 h.

   If the `evaluate` shards are still running at the deadline, the longest unit that cannot stop is an hz1x1 DT route (i) or P3a,
   about 25–27 min with six running. So **E2 runs at most about 4 h 30 min.** The grid's `dt_nomix` P3a, about 35 min, comes early
   in shard 0, after its 396 route (ii) episodes, and ends at about 1.1 h.
4. **E3** is cleared after E2's verification, with its hours from E2's records. It runs the old cells' rest (about 1.1–1.5 h),
   then P3a's two references (about 0.35 h): about 1.5–1.8 h in all.
5. **Beside E2,** as O.3-5:
   - no training and nothing else on the GPU;
   - nothing memory-heavy;
   - P9.1's own work one pytest process at a time, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`;
   - P9.3 runs nothing.
6. **The coordinator's outputs for O.3-7** stay in its scratchpad for E2's check of hz1x1's `dt@random_fresh` and
   `iql_t@mappo1000`. If they are lost, that check is skipped and said so.

## P.4 ✅ THE TOKEN: evaluate session E2 (the author's tmux `p91`)
**E2 runs about 4 hours without a break, and at most about 4 h 30 min** (P.3-3). It ends by itself.
- **What it runs:**
  - `declare-evaluation`, which finds the three manifests unchanged and writes nothing;
  - the six `evaluate` shards, each resuming from disk, to the end of their cells (about 3.0–3.1 h);
  - then the six `old-cells` shards, until the deadline 4 h after the session's start.

  At the deadline each shard ends between two episodes, or finishes its unit in flight.
- **Before it:**
  - no other P9.1 stage runs, and nothing runs on the GPU;
  - `output/p9_1_runs/STOP` does not exist;
  - keep the laptop plugged in and awake.
- **The command,** at the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: E2 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_evaluate && P91_SESSION_HOURS=4 WORKERS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh evaluate c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/evaluate_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows:
  - the token line, the session line and `declare_evaluation: ok`;
  - after about 3 h, `evaluate: 6 shards ended cleanly`;
  - at about 4 h, `old-cells: 6 shards ended cleanly`, the `left:` line and `stage evaluate <stamp>: SESSION ENDED at the hours
    bound (4 h) …`;
  - then `DRIVER EXIT 0`.
- **Done when** the capture shows those lines under E2's stamp and the pane shows `DRIVER EXIT 0`. The expected `left:` line:
  - route (i) 0 of 53 cells;
  - route (ii) episodes grid4x4 0 of 13000, hz1x1 0 of 13500;
  - P3a 2 of 6 cells;
  - roughly half of the 18,200 old-cell episodes.

  If the `evaluate` shards are still running at the deadline, `old-cells` does not start, and the `left:` line shows what is left
  of them.
- **STOP, a failure, a stall and Ctrl+C:** as O.4. A route (ii) episode that does not reproduce route (i), or an old-cell episode
  that does not reproduce its committed value, is a finding: the shard fails, and nothing is re-run without a ruling.

## P.5 After E2, and meanwhile
- **After E2:** the coordinator verifies it from disk, as in P.1, now with the old cells' episodes and O.3-7's two hz1x1 cells.
  Then it clears E3, its hours first.
- **P9.1's commits `8ab9761` and `11616ea`** (Q18, F.2-4, G.2; Q19 and Q20 offered) are verified by the coordinator in their own
  amendment, before G4.
- **P9.3** waits for P9.1's merge (`BRIEF_48` J.9, J.11).

---

# ✅ AMENDMENT Q — 2026-11-06: evaluate session E2 VERIFIED from disk (one stall, restarted by B.3's guard as designed); E3, the last session, CLEARED (`P91_SESSION_HOURS=2`: about 1 h without a break, at most about 2 h 15 min)

## Q.1 E2, verified from disk (read-only; the scripts printed counts, flags, provenance and seconds; no outcome value was read)
- **The session** ran from the run tree at `c58afb8`, which its token line names:
  - token at 21:03:18 CET (stage `20261105T200318Z`), deadline 01:03:18;
  - the six `evaluate` shards started at 21:03:20, resumed from disk, and skipped E1's finished cells in 1.5–1.6 s;
  - they ended after their last cells, all before the deadline: at 1.70, 1.96, 1.99, 2.38, 2.61 and 2.74 h;
  - the six `old-cells` shards started at 23:47:41 and ran to the deadline, each stopping between two episodes 0.3–1.9 s after
    it;
  - the capture's last line was written at 01:03:23: **4 h 00 min 06 s after the token** (P.3-3: about 4.0 h).

  Under E2's stamp the capture holds the token line, the session line, `declare_evaluation: ok`, `evaluate: 6 shards ended
  cleanly`, `old-cells: 6 shards ended cleanly`, the `left:` line and `SESSION ENDED at the hours bound (4 h)`. E1's part above it
  is unchanged. The author's pane showed `DRIVER EXIT 0`.
- **The 53 cells and the six shards,** written out by hand, equal the code's, as in P.1.
- **Route (i): 53 of 53 cell files.** E1's 33 are unchanged. E2's 20, all hz1x1, were each checked as in P.1:
  - 500 episodes over the 5 × 100 keys, of its own arm, every value a finite float;
  - engine seed 1000, the tier, the cell's seeds and draws, and its declared steps;
  - its mean, recomputed, equal to within 10⁻¹² relative;
  - a write inside E2.

  The 20 record `cuda`, one thread, and main's HEAD at the write, `4237cee` (G.2). hz1x1's `behaviour@random_fresh` has no
  `declared_gradient_steps` key: `method_tier_grid`'s `behaviour` command declares no budget for an algorithmic arm, while the
  grid's `tier_sweep` declares 40,000 for every cell. The hz1x1 aggregates (90 and 40 runs), rewritten during E2, again equal their
  rebuild.
- **Route (ii) is complete:** 13,500 hz1x1 and 13,000 grid episode files, every one declared, and **0 refused**. For every one:
  - its names are its file's;
  - `reproduces_committed` is true, `att_ours == committed_att_ours`, and `committed_att_ours ==` route (i)'s `att_horizon` for the
    same episode.

  E1's 14,405 files are unchanged. E2's 12,095 (hz1x1 11,603, the grid 492) were written inside E2. The 25,500 trained-arm
  episodes name their run's checkpoint, and the two behaviour cells' 1,000 are algorithmic.
- **P3a: 4 of 6 cells.** E1's grid `behaviour@random_fresh`, and E2's three:
  - the grid's `dt_nomix@random_fresh`, written at 22:06:16;
  - hz1x1's `dt@random_fresh`, at 23:26:14;
  - hz1x1's `behaviour@random_fresh`, at 23:39:45, after the restart (Q.2).

  Each has 500 episodes over the 5 × 100 keys and `att_ours ==` route (i) in 500 of 500. Its sequence sha256s are well formed (500
  distinct in each), and so are its 16 or 1 per-intersection sha256s. The two registered references are E3's.
- **The old cells: 13,119 of 18,200 episodes (72 %), 0 refused.** They fall in 32 of the 38 cells: 26 complete, and one cell in
  flight in each shard at the deadline. The other 6 cells are not started. For every episode:
  - it is declared, and its names are its file's;
  - `reproduces_committed` is true, and `att_ours == committed_att_ours`;
  - `committed_att_ours ==` its committed artifact's `att_horizon`, read from the run tree's `docs/data` by the module's
    `old_cell_committed`;
  - `att_engine` is a finite float;
  - it was written in E2, after the old cells began.

  Each shard's files are a prefix of its order (the sorted cells, every sixth; each cell's keys by seed and draw), so no episode was
  skipped. Every episode whose policy is a checkpoint names a file that exists, five distinct ones in each complete seeded cell.
  The rule-based behaviour cells are `algorithmic`, or a `plan` (fixed time).
- **The status, recomputed,** equals the capture's `left:` line and the session's `status.json`:
  - route (i): 0 of 53 cells left;
  - route (ii): 0 grid and 0 hz1x1 episodes left;
  - P3a: 2 of 6 cells left, the references;
  - old cells: 5,081 of 18,200 episodes left.
- **The logs.** Each `evaluate` shard's log holds only:
  - the CLIs' seed lines;
  - route (ii)'s progress lines, with `refused=0` throughout;
  - hz1x1 route (i)'s summary lines, 20 in all. These print means, so they were counted and not read;
  - the module's result: every cell of the shard evaluated, none left, none waiting, not stopped.

  Shard 1's log also holds the driver's restart line (Q.2). Each `old-cells` shard's log holds only progress lines, with `refused=0`,
  and the module's result:
  - `rolled` equals its files on disk, and nothing was skipped;
  - 0 refused, stopped by the bound;
  - `rolled` plus `not_reached` equals its episodes.

  There is no other line: no warning, error or traceback. No STOP, token, `.partial` or `.tmp` file is left.
- **Nothing else changed.**
  - Training: 255 of 255; no checkpoint or record was written after E1 began.
  - Under `output/p9_1*`, against E1's listing: 15,034 of 15,046 paths are unchanged, and none was removed. The other 12 are:
    - 9 directories that gained E2's files;
    - the capture, appended;
    - the two aggregates, rewritten at the same size and equal to their rebuild.

    The 25,268 new paths are all E2's: route (i) 20, route (ii) 12,095, P3a 3, the old cells 13,119, `p3a/hz1x1/`, and the log
    directory with its 29 files.
  - Nothing else under `output/` was written after E2's token, and the corpus listing is unchanged.
  - After E2, outside the sandbox, read-only: no P9.1, P9.3 or pytest process, and the GPU at 699 MiB with no compute process.
- **O.3-7's check is complete.** hz1x1's whole `dt@random_fresh` and `iql_t@mappo1000` cells (500 episodes each) equal the
  coordinator's outputs of 2026-11-04 in `att_horizon`, `episode_reward` and `horizon_vehicle_count` under `==`. With P.1's 750,
  that is **1,750 of 1,750 episodes equal**: the same code, checkpoints and seeds gave the same values alone, six at once, and in
  the campaign.
- **The coordinator's error, in its own check.** Its script first expected 40,000 declared steps on hz1x1's behaviour cell,
  carrying the grid's `tier_sweep` convention over, and flagged the cell. The module's code shows that the key's absence is by
  design. The check was corrected and the script rerun: 0 problems. The error was in the check, not in the run.

## Q.2 The stall in E2, and the stall file the author asked about
- **The author asked** whether `old-cells_shard1.stall0.txt` (empty, 23:47:41) marks a stall. **It does not.** Every worker opens
  its stall file when it starts: `offline/stall_guard.py` truncates it and keeps it open for faulthandler, and the driver reads an
  empty file as "no stall". The six `old-cells` shards opened theirs at 23:47:41, when they started, and all six are empty.
- **One stall did occur, elsewhere:** `evaluate` shard 1, at 23:32:13.
  - `evaluate_shard1.stall0.txt` holds faulthandler's dump (742 bytes, starting with the banner `Timeout (0:02:00)!`).
  - `restarts.log` holds one entry: restart 1 of at most 60, window 120 s, and the dumped stack, equal to the dump.
  - `evaluate_shard1.stall1.txt` (empty, 23:32:14) is the restarted worker's.
- **Where it stalled.** The worker was in P3a's recording of hz1x1's `behaviour@random_fresh`. Its stack ends in
  `record_episode_actions`, at its call into `dt_gate.evaluate_arm`, with no Python frame below. So the worker was in C code
  reached from that line, as when `evaluate_arm` has returned and its frame releases the episode's env, with its CityFlow engine.
  This is consistent with `DEFERRED` 104's destructor race (`docs/notes/P5.3c_HANG_2026-09-30.md`): at the end of an episode,
  after its values exist.
- **B.3's guard worked as designed.**
  - The guard's window is 110–120 s, so the last progress came at about 23:30:13–23:30:23. Then came the dump and exit 1, and
    the driver restarted the worker at 23:32:14, with no new token.
  - The restarted worker resumed from disk. It skipped shard 1's finished cells and wrote one file, the P3a in flight, from
    scratch: 500 episodes in 451 s, each `==` route (i).
  - P3a writes once per cell, so the first attempt's 4 min of work left no file.
  - The stall cost shard 1 about 6 min (369 s). It still ended 8 min before shard 0, so E2 was not lengthened.
- **It is the first stall of P9.1's evaluate stage,** in about 68,000 episodes over E1 and E2. P5.3c's note saw two in about 1,200,
  under heavier load. It changes no value.

## Q.3 E2's hours against P.3-3
E2 ended at 4.0 h, as P.3-3 said, but its parts ran faster than projected:
- The `evaluate` shards ended at 1.70–2.74 h, against the projection's 1.85–2.96 h: 0.88–0.93 of it, shard by shard.
  - With all six running, hz1x1's BC took 1.84–2.04 s per episode, its IQL 1.87–2.17 and its DT 2.69–2.87. E1's six-at-once
    rates were 2.03–2.16, 2.29–2.40 and 2.97–3.03.
  - E2's grid work ended at 1.05 h. E1's hz1x1 units had run mostly beside two grid processes (P.2).
- The old cells did 72 % of their alone-work (4.35 of 6.05 alone-hours), against the projection's 38–50 %.
  - With six at once throughout, they ran 1.34–2.05 times slower than O.2's alone rates: BC-family 1.85, IQL 2.05, DT 1.62, the
    MAPPO-policy cells 1.34, the rule-based 2.01.
  - The projection had taken 2.21, 2.26, 1.64, 2.21 and 1.36 from E1's units.

The projection erred on the slow side. It changes the hours, not the values.

## Q.4 Rulings
1. **E2 is verified.**
2. **E3, the last session, runs at `P91_SESSION_HOURS=2`, with six workers: about 1 h without a break.** From E2's own rates:
   - the `evaluate` pass over the 53 finished cells takes under a minute;
   - the old cells' rest takes 0.62–0.72 h: 5,081 episodes, 333–1,323 per shard, in one to three cells per shard;
   - P3a's two references, in one process, take about 0.35 h;
   - then the status: about 1.0–1.1 h in all. With the references at twice O.2's alone rates, about 1.4 h.

   The bound is a cap, not a target: the stage ends by itself, `COMPLETE`, when the evaluation is complete on disk. If E3 reaches the
   bound, the longest unit that cannot stop is one reference cell, 500 episodes alone (about 8–13 min). So **E3 runs at most about
   2 h 15 min**, and a fourth session, cleared from E3's records, would finish the rest.
3. **Beside E3,** as P.3-5:
   - no training and nothing else on the GPU;
   - nothing memory-heavy;
   - P9.1's own work one pytest process at a time, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`;
   - P9.3 runs nothing.
4. **The restart** is recorded here. P9.1's packet states it where it reports the evaluate sessions, as `restarts.log` has it: one
   restart, `evaluate` shard 1, B.3.
5. **The stall-file question** is answered (Q.2). Nothing follows from it.

## Q.5 ✅ THE TOKEN: evaluate session E3, the last (the author's tmux `p91`)
**E3 runs about 1 hour without a break, and at most about 2 h 15 min** (Q.4-2). It ends by itself.
- **What it runs:**
  - `declare-evaluation`, which finds the three manifests unchanged and writes nothing;
  - the six `evaluate` shards, which find all 53 cells finished and end within a minute;
  - the six `old-cells` shards, resuming from disk, to the end of the old cells (about 0.6–0.7 h);
  - P3a's two registered references, in one process (about 0.35 h);
  - the status, which finds the evaluation complete.
- **Before it:**
  - no other P9.1 stage runs, and nothing runs on the GPU;
  - `output/p9_1_runs/STOP` does not exist;
  - keep the laptop plugged in and awake.
- **The command,** at the pane's prompt, as one line:
  ```
  /home/filip/rltraffic/.venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA: E3 not started')" && touch /home/filip/rltraffic/output/p9_1_runs/TOKEN_evaluate && P91_SESSION_HOURS=2 WORKERS=6 bash /home/filip/rltraffic-p91-run-d/offline/campaigns/p9_1_corrective.sh evaluate c58afb8 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_1_runs/evaluate_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"
  ```
- **The pane** shows:
  - the token line and the session line. Its "about 2 hours" is the bound; the work takes about 1 h;
  - `declare_evaluation: ok`, and within a minute `evaluate: 6 shards ended cleanly`;
  - after about 0.6–0.7 h, `old-cells: 6 shards ended cleanly`;
  - after about 0.35 h more, `p3a_references: ok` and `stage evaluate <stamp>: COMPLETE (status …/status.json)`;
  - then `DRIVER EXIT 0`.
- **Done when** the capture shows those lines under E3's stamp and the pane shows `DRIVER EXIT 0`. A completed stage prints no
  `left:` line.
- **If E3 reaches its bound first,** the pane shows the `left:` line and `SESSION ENDED at the hours bound (2 h)` instead. The rest
  is then cleared from E3's records.
- **STOP, a failure, a stall and Ctrl+C:** as O.4. The guard restarts a stalled worker, as in E2 (Q.2). The restart line goes to
  the shard's log, not the pane. An old-cell or reference episode that does not reproduce its committed value is a finding: the
  worker fails, and nothing is re-run without a ruling.

## Q.6 After E3, and meanwhile
- **After E3:** the coordinator verifies it from disk, as in Q.1, now with the old cells complete, P3a's references, and the
  evaluation's status complete.
- **P9.1's commits `8ab9761` and `11616ea`** (Q18, F.2-4, G.2; Q19 and Q20 offered) are verified by the coordinator in their own
  amendment, before G4, as P.5 said. The branch has no commit since `11616ea`.
- **P9.3** waits for P9.1's merge (`BRIEF_48` J.9, J.11).

---

# ✅ AMENDMENT R — 2026-11-06, evening: evaluate session E3 VERIFIED from disk — P9.1's EVALUATE STAGE COMPLETE (one stall, in P3a's reference recording, restarted by B.3's guard as designed); the verification of P9.1's commits next, before G4

## R.1 E3, verified from disk (read-only; the scripts printed counts, flags, provenance and seconds; no outcome value was read)
- **The session** ran from the run tree at `c58afb8`, which its token line names:
  - token at 16:25:11 CET (stage `20261106T152511Z`), deadline 18:25:11 (the 2-h bound);
  - `declare-evaluation` at 16:25:11–16:25:12, which found the three manifests unchanged and wrote nothing;
  - the six `evaluate` shards at 16:25:12–16:25:14, which found all 53 cells complete and skipped them (1.7 s);
  - the six `old-cells` shards from 16:25:14. They resumed from disk: the skip over E2's episodes and the set-up took 1.6 s. They
    ended after their last episodes, at 0.98–1.35 h (17:23:47–17:46:29), none stopped;
  - P3a's two references in one process from 17:46:30, with one stall and restart (R.2). hz1x1's `behaviour@random` was written at
    17:56:04 and the grid's `random@mappo1000` at 18:06:01;
  - the status, complete. The capture's last line was written at 18:06:06: **1 h 41 min after the token** (Q.4-2: about 1.0–1.1 h;
    R.3).

  Under E3's stamp the capture holds exactly seven lines, in order: the token line, the session line, `declare_evaluation: ok`,
  `evaluate: 6 shards ended cleanly`, `old-cells: 6 shards ended cleanly`, `p3a_references: ok` and `stage evaluate
  20261106T152511Z: COMPLETE (status …/status.json)`. There is no `left:` line. The part above it, E1's and E2's, is unchanged: 1,738
  bytes, as in E2's listing. The author's pane showed `DRIVER EXIT 0`.
- **The 53 cells and the six shards,** written out by hand, equal the code's, as in P.1.
- **Route (i): 53 of 53 cell files, none written in E3.** E1's 33 and E2's 20 were each checked again as in P.1. The 27 hz1x1 files
  record `cuda` and one thread, with main's HEAD at the write: `25410f4` for E1's 7 and `4237cee` for E2's 20 (G.2).
- **Route (ii): complete, none written in E3.** 13,500 hz1x1 and 13,000 grid episode files, every one declared, and **0 refused**.
  Every check of Q.1 holds for every file. E1 wrote 14,405 of them and E2 12,095.
- **P3a: 6 of 6 cells.** E1's and E2's four are unchanged, each 500 of 500 `==` route (i). E3's two are the registered references:
  - **Their committed values were read by the coordinator's script itself,** apart from the module:
    - hz1x1's `behaviour@random` from P4.6's `docs/data/p4_6_grid.json`, whose bytes in main equal the run tree's;
    - the grid's `random@mappo1000` from `output/p5_1/eval_random.json`, at the digest `output/SHA256SUMS_p5_1.txt` lists for it (once),
      last written on 2026-08-18.

    Each is 500 episodes over the 5 × 100 held-out keys, and equals the module's reading (`_registered_reference_values`) in 500 of
    500.
  - **Each P3a file** has 500 episodes over those keys, with `att_ours ==` the committed value in 500 of 500. Every `n_decisions` is
    positive, every sequence sha256 is well formed, and so are the 1 or 16 per-intersection sha256s. Its format, scenario, arm,
    `registered_in` (A27(d)) and digest description are the module's.
  - **Each has 5 distinct sequences, one per seed:** every seed's 100 draws share one 360-decision sequence. This is the registered
    anchor's design, *"the registered random anchor is a five-schedule reference (`method_tier_grid._random_factory`, one generator
    per draw from the training seed)"* (plan §8, 2026-10-15). A27(d) compares with both random references (3.4): this five-schedule
    anchor, and `random_fresh`'s, whose two behaviour cells have 500 distinct sequences each (P.1, Q.1).
- **The old cells: 18,200 of 18,200 episodes, 0 refused.** For every episode:
  - it is declared, and its names are its file's;
  - `reproduces_committed` is true, and `att_ours == committed_att_ours`;
  - `committed_att_ours ==` its committed artifact's `att_horizon`;
  - `att_engine` is a finite float;
  - it was written after its session's old cells began.

  The committed values were read twice and are equal in 38 of 38 cells: once by the module's `old_cell_committed` from the run tree's
  `docs/data`, and once by the coordinator's script from main's. The three artifacts have the same bytes in both trees.

  In each shard, E2 wrote the first 1,897–2,523 episodes of its order and E3 the rest, 333–1,323 (5,081 in all). So no episode was
  skipped or rolled twice, and E2's 13,119 files are unchanged. Every episode whose policy is a checkpoint names a file that exists.
  The behaviour cells without a checkpoint are `algorithmic` (600 episodes) or a `plan` (fixed time, 100).
- **The status, recomputed, is complete,** and equals E3's `status.json`:
  - route (i): 53 of 53 cells;
  - route (ii): 13,500 of 13,500 hz1x1 and 13,000 of 13,000 grid episodes, none refused;
  - P3a: 6 of 6 cells;
  - old cells: 18,200 of 18,200 episodes, none refused;
  - training: 255 of 255.
- **The logs** hold only known lines, and no warning, error or traceback:
  - each `evaluate` shard's log holds only the module's result: every cell of the shard evaluated (9, 9, 9, 9, 9, 8), none left,
    none waiting, not stopped;
  - each `old-cells` shard's log holds only progress lines, with `refused=0`, and the result: `rolled` equal to E3's files,
    `skipped_already_present` equal to E2's, 0 refused, not stopped, 0 not reached;
  - `p3a_references.log` holds the driver's restart line (R.2) and the result: 500 and 500 recorded, not stopped;
  - `declare_evaluation.log` holds the three manifests' counts (13,500, 13,000, 18,200), and `status.err` is empty.

  No STOP, token, `.partial` or `.tmp` file is left.
- **The hz1x1 aggregates** (90 and 40 runs) equal their rebuild. They were last written in E2 (23:29:08) and not in E3, since route
  (i) found every cell on disk.
- **Nothing else changed.**
  - Training: 255 of 255; no checkpoint or record was written after E1 began.
  - Under `output/p9_1*`, against E2's listing: 40,308 of 40,314 paths are unchanged, and none was removed. The other 6 are:
    - `output/p9_1_runs/` itself (the token, created and consumed);
    - `logs/`, `old_cells/hz1x1/`, `p3a/hz1x1/` and `p3a/grid4x4/`, which gained E3's files;
    - the capture, appended.

    The 5,115 new paths are all E3's: the log directory with its 31 files, 5,081 old-cell episodes and 2 P3a files.
  - Nothing else under `output/` was written after E2, and the corpus listing is unchanged.
  - After E3, outside the sandbox, read-only: no P9.1, P9.3 or pytest process, and no compute process on the GPU.
  - The run tree is clean at `c58afb8`. P9.1's branch is at `11616ea` and P9.3's at `12ebfcb`, both unchanged.

## R.2 The stall in E3
- **`p3a_references` stalled at 17:50:20,** in the first reference, hz1x1's `behaviour@random`.
  - `p3a_references.stall0.txt` holds faulthandler's dump: 665 bytes, starting with the banner `Timeout (0:02:00)!`.
  - Its 7 frames run, innermost first, `record_episode_actions` at line 841 (its call into `dt_gate.evaluate_arm`), `record_p3a`,
    `record_registered_references` and `main`, with no Python frame below. That is E2's signature (Q.2), consistent with `DEFERRED`
    104's destructor race.
  - `restarts.log` holds one entry: restart 1 of at most 60, window 120 s, with the dumped stack, equal to the dump.
  - `p3a_references.stall1.txt` (empty, 17:50:20) is the restarted worker's.
  - The driver's restart line is in `p3a_references.log`. The pane shows only `run_logged`'s `p3a_references: ok`.
- **B.3's guard worked as designed.**
  - The worker started at 17:46:30. Its last progress came at about 17:48:20–17:48:30, so its first attempt did about 111–121 s of
    work.
  - The driver restarted it at 17:50:20, with no new token.
  - The restarted worker rolled hz1x1's reference from scratch, in 343 s: P3a writes once per cell, so the first attempt left no
    file. Then it rolled the grid's, in 597 s. Each is 500 of 500 `==` its committed value.
  - The stall cost E3 about 4 min.
- **Both stalls of the evaluate stage came in P3a's recording**, in `record_episode_actions`, which makes one `evaluate_arm` call per
  episode. Both were in hz1x1's random-policy cells: `behaviour@random_fresh` in E2 and `behaviour@random` in E3. That is 2 stalls
  in about 3,450 P3a episodes (the six cells and the two first attempts). Route (i), route (ii) and the old cells had none in about
  71,200 episodes. E3's stall came with one process running, not under load.

  This is an observation, not a diagnosis. It changes no value: every recorded episode reproduces its committed value under `==`, and
  B.3's guard handles it. It goes into `DEFERRED` 104's record at P9.1's merge.

## R.3 E3's hours against Q.4-2
- **E3 took 1 h 41 min,** against Q.4-2's 1.0–1.1 h (about 1.4 h with slow references). It stayed inside its 2-h bound and completed
  by itself.
- **The overrun is one window, from about 16:30 to 17:15.**
  - From 16:32 to 17:14 the six shards together completed 5–32 episodes a minute. Before 16:30 they completed 116–120 a minute,
    and after 17:15, 147–162.
  - Per episode, that is 8–11 times E2's own seconds on the same cells (the median of each 5-minute interval).
  - At 17:15 the rate came back at once, with all six shards still running.
- **Outside the window the projection held.**
  - Before it (16:25–16:30), the cells ran at about 1.3 times E2's rates.
  - The cells started after it ran at 0.75–0.99 of E2's rates, falling to about 0.7 as shards finished.
  - P3a's references ran at 0.69 and 1.19 s per episode, against O.2's alone rates of 1.0 and 1.53.

  The window cost about 40 min. Without it, the old cells took about 0.7 h, as Q.4-2 projected (0.62–0.72 h).
- **The cause is not established from disk.**
  - The slowdown follows the clock, not the cells: it struck all six shards and every family of cell at once, and began and ended
    abruptly. That points to the machine (power, heat or another load), not to the run.
  - The evaluate stage keeps no GPU or CPU trace.
  - If the author knows what the laptop was doing between about 16:30 and 17:15 (Q.5 asked for it plugged in and awake), it is
    recorded. Otherwise it stays unexplained.

  **It changes hours, not values:** every episode in the window reproduces its committed value under `==`.

## R.4 Rulings
1. **E3 is verified, and P9.1's evaluate stage is COMPLETE.** All under the campaign's single engine seed 1000, with 0 refused
   anywhere and every reproduction holding under `==`:
   - route (i): 53 of 53 cells;
   - route (ii): 26,500 of 26,500 episodes;
   - P3a: 6 of 6 cells;
   - the old cells: 18,200 of 18,200 episodes.

   No further evaluate session runs, and nothing more runs from the run tree.
2. **The restarts.** P9.1's packet states both where it reports the evaluate sessions, as `restarts.log` has them:
   - E2: `evaluate` shard 1 (Q.2);
   - E3: `p3a_references` (R.2).

   Both were B.3's, both were in P3a's recording, and each P3a they touched was re-rolled from scratch and reproduces its value
   under `==`.
3. **The hours.** P9.1's packet reports the three sessions' hours as their captures have them: E1 4 h 05 min 35 s (P.1), E2 4 h 00 min
   06 s (Q.1) and E3 1 h 40 min 55 s. It states E3's slow window as R.3 does: unexplained, and changing no value.
4. **`DEFERRED` 104** gains R.2's observation at P9.1's merge.
5. **Next: the verification of P9.1's commits, then G4.** The author runs nothing for P9.1 until G4 is cleared.

## R.5 Next
- **The coordinator** verifies P9.1's commits `8ab9761` and `11616ea` (Q18, F.2-4, G.2; Q19 and Q20 offered) in their own amendment,
  S, from the repository's objects. Their tests run on the CPU only, one pytest process at a time, with `OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1`. G4 is cleared after S.
- **P9.1's branch** stays at `11616ea` until then.
- **`output/`:** nothing is moved or deleted. The stray `.claude` directories stay as recorded (N.2).
- **P9.3** waits for P9.1's merge (`BRIEF_48` J.9, J.11).
