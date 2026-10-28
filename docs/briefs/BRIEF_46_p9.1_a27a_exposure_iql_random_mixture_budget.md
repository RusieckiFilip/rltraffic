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
(outputs /home/filip/d4rl-calib-out/p9_1)`. Five seeds trained from 20:20:29 to 20:47:25, about 5.3 min each.

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
