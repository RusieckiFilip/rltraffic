# FINDINGS -- Review A (P9.4 selection, declaration, training, control; plan sections 2-3, Amendment A1)
Repo: /home/filip/rltraffic-p94, branch task/p9.4-mix50-fresh, HEAD 512c101. Reviewer started 2026-11-19.

## Checklist
1. [x] Q1 select_streams == plan 2.1 rule (mix50 spec, fraction, rng, sort), only random pool replaced -- by experiment vs method_tier_grid.mixture_training_streams
2. [x] Q2 checks (i)-(v) implement plan 2.1 (v per Amendment A1), refuse on failure, write nothing -- by experiment
3. [x] Q3a build_declaration holds every field plan 2.4 lists
4. [x] Q3b refuses per plan 2.2 (source target from expert half) -- by experiment
5. [x] Q3c the 2.3 changes computed from manifests (not hardcoded)
6. [x] Q3d calibrated target by registered route (targets_table/rule_b_target over P7.2b stats)
7. [x] Q4 training_inputs == method_tier_grid.training_inputs step for step except selection call (diff + optional corpus-gated test)
8. [x] Q5 train_run calls dt_gate.train_dt with exactly P9.1's/_run_train's arguments; control torch.equal failure modes
9. [x] Q6a no path trains mix50_fresh before a control record with passed=True
10. [x] Q6b failed control cannot be bypassed by resuming
11. [x] Q6c Amendment A1 re-hash before first gradient step of EVERY session
12. [x] Q7 write barrier: writes only under output/p9_4/ and output/p9_4_runs/, no overwrite
13. [x] Q8 >= 6 mutants in scratch copy, killed/survived
14. [x] Run the two test files (selection, training) and record real results
15. [x] Frozen-file edits in diff main...HEAD (outside offline/, tests/, docs/)
16. [x] Tests that cannot fail / theatre classification
17. [x] Silent assumptions (dtype, sort stability, dict order, NaN)

## Findings (appended in order)

### Item 15 [x] -- frozen-file edits (2026-11-19, first minutes)
- Command: `git diff --stat main...HEAD -- . ':!offline' ':!tests' ':!docs'` -> empty. merge-base == main == 5eedf21 (plan's start commit).
- Result: no file outside offline/, tests/, docs/ touched. NOTE (out of A's scope, pointer for B/C/D): two EXISTING registered
  modules are edited: offline/att_rederivation.py (+9/-1, new keyword `env_settings_builder`, default None keeps registered path)
  and offline/transfer_curve.py (+101). Not frozen by CLAUDE.md §1, but "registered code" -- flag only.
  docs/PROJECT_PLAN.md modified (3 lines); no existing docs/data/ file modified (only new p9_4_declaration.json).

### Item 14 [x] (part 1, ungated) -- the two test files, no RLTRAFFIC_* set
- Command: `cd /home/filip/rltraffic-p94 && /home/filip/rltraffic/.venv/bin/python -m pytest -q -p no:warnings tests/test_p9_4_selection.py tests/test_p9_4_training.py -rs`
- Result (observed): `30 passed, 6 skipped in 1.30s`. Skips: selection.py:226 (x3), :237, training.py:346 (corpus equivalence), :366 (P4.7 refs).
  So in a default run NOTHING about the real corpus / real selection / real control comparison is exercised.

### Item 1 [x] -- Q1 select_streams == plan 2.1 rule, by an independent route
- Code read: offline/p9_4_mix50_fresh.py:273-279 calls grid.mixture_training_streams(grid.TIERS["mix50"], expert, random)
  (method_tier_grid.py:613-639: fraction 0.50 from MIXTURE_EXPERT_FRACTION, rng default_rng(20260813+50), both pools key-sorted,
  expert choice drawn BEFORE random choice from the same rng). Pools: mappo1000 subsample "none" (all 200, key-sorted);
  random_fresh A27 tier subsample "none" (method_tier_grid.py:335-343).
- Experiment: scratchpad/reviews/A/scripts/q1_independent.py -- manifests + numpy only, NO offline.* import: rebuild both pools
  from manifest filenames, default_rng(20260863).choice twice, compare to docs/data/p9_4_declaration.json "streams".
- Observed: pool sizes 200/200; independent selection == declared 200 streams IN ORDER: True; expert half == mix50's expert half
  (p4_7_declaration.json) by path in order: True; random draws == mix50's random draws in order: True (2..196);
  engine-seed transitions {'1001->1000': 61, '1000->1000': 39} == declared; old file names seed1001 x61, seed1000 x39.
- Verdict: Q1 holds on the committed declaration. Draw matching rests on both pools sorting in draw order (zero-padded ep index);
  that is a property of the data, which check (iii) verifies, not of the rule.

### Item 14 [x] (part 2, gated, read-only against the main tree)
- Command: `PYTHONDONTWRITEBYTECODE=1 RLTRAFFIC_CORPUS_V11=/home/filip/rltraffic/datasets_v11 RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output
  python -m pytest -q -p no:warnings -p no:cacheprovider tests/test_p9_4_selection.py tests/test_p9_4_training.py` (from the worktree)
- Observed: `36 passed in 23.76s`, exit 0 (scratchpad/reviews/A/gated_run.txt). The corpus equivalence test
  (tests/test_p9_4_training.py:346, 19.6 s) passed: p94.training_inputs(TIERS["mix50"]) == grid.training_inputs(TIERS["mix50"])
  on selected/kept keys, declared, scale, digest, provenance, group, scenario_id, and torch.equal on every full_batch/top_batch tensor.
  P4.7 reference pins resolved for all 5 seeds (training.py:370).

### Item 2 [x] -- Q2 the five checks
- Read offline/p9_4_mix50_fresh.py:304-362 and build_declaration :553-562 (assert_checks before ANY write; declare() :671-676 writes
  only after build_declaration returns, so a failed check writes nothing -- by construction, the write is strictly after).
- Experiment (scratch copy, direct calls): check_v([{"file_sha256": "a"*64}]) -> passed True (str(None)==str(None): a row with
  NEITHER digest passes the equality); check_v with file_sha256 "z"*64 (not hex) -> True (only len==64 checked); check_iv({}) -> False;
  check_iv one bad ix -> False; check_iii dup in mix50's draws -> False; assert_checks with passed "yes"/1 -> refused (is True).
- Verdict: (i)-(v) implement plan 2.1 / A1 and refuse; check_v's None==None hole is NOT reachable from build_declaration
  (rows always carry str(entry["episode_sha256"]) -- a missing key KeyErrors -- and a computed trajectory digest) -> NON-BLOCKING
  robustness note at offline/p9_4_mix50_fresh.py:342. check_ii's "same_files_as_mix50" (:556) is recorded but not part of passed
  (plan asks only for episode_sha256 equality) -- informational; value in committed declaration: True.

### Item 3 [x] -- Q3a declaration holds plan 2.4's list (committed docs/data/p9_4_declaration.json, inspected)
- Per-stream keys: dataset_dir, engine_seed, episode_file, episode_sha256, file_sha256, flow_draw, half, ix_id, total_return,
  trajectory_digest. 200 rows, 100/100. checks i..v all passed. changes (basis 600->400, scale 40223->41135, seeds 1001->1000 x61,
  1000->1000 x39), normalisation digest be4fa9ee..., target -5959.0, scale 41135.0, prompts b_mean_k100 -7431.0185327454665 and
  naive -5959.0 (both = registered mix50's), recipe, seeds, regime, evaluation {sumo, cityflow}, guards {control,g1,g2},
  git 2872dbd clean. Every plan 2.4 item present.

### Item 6 [x] -- Q3d calibrated target by the registered route
- offline/p9_4_mix50_fresh.py:585-596 -> transfer_calibration.targets_table (offline/transfer_calibration.py:1619-1700) which calls
  rtg_calibration.rule_b_target(best_source_return, probe cityflow/sumo stat) over P7.2b's committed statistics; S=mean,k=100 row is
  "registered_prompt". Only best_source_return (-5959.0) and support range enter; so the calibrated target equals mix50's BY
  CONSTRUCTION (same best_source_return) and the check at :599 can only fail if P7.2b's file moved. Registered route: yes.
- Silent assumption (NON-BLOCKING): :587 `training_set_return_min=-float(scale)` assumes every return is negative (true here:
  training_return_max -5959.0); targets_table does not read that field.

### Item 4 [x] -- Q3b plan 2.2 refusal (source target from the expert half) + independent re-derivation from raw files
- Code: offline/p9_4_mix50_fresh.py:564-570 -- refuses unless max return == -5959.0 AND the set of halves at the max == ["expert"].
- Experiment: scratchpad/reviews/A/scripts/q3_raw.py (numpy+hashlib, no offline.*): for all 200 declared rows, from the npz:
  sum(local_reward) in float64 AND in native float32, my own little-endian trajectory digest, file sha256, seed/draw from the
  file name, and the manifests' (episode_sha256, engine_seed, flow_draw).
- Observed: mismatch counts all 0 (ret64 0, ret32 0, traj 0, file 0, seed 0, draw 0, manifest 0); max return -5959.0 only in the
  expert half; max|return| 41135.0 (= min -41135.0, in the random half); random-half max -36769.0 (far below -5959).
- Verdict: target_rtg, rtg_scale, check (v) and the engine seeds are correct by an independent route; no float32/float64 drift
  (rewards integral). The refusal condition is exactly plan 2.2's.

### Item 5 [x] -- Q3c plan 2.3 changes computed from the manifests
- :602-611 engine-seed transitions read from manifests (old: cf_hz1x1__random manifest by mix50's old file; new: the row's
  manifest engine_seed) -- reproduced independently in q1_independent.py: {'1001->1000': 61, '1000->1000': 39}.
- :648 normalisation basis: mix50_fresh = len(stream_returns(dataset)) = 400, COMPUTED; mix50's 600 is a HARDCODED constant
  (MIX50_SPLIT_STREAMS :108), as is MIX50_RTG_SCALE 40223.0 (:107) -- neither is read from p4_7_declaration.json / a mix50 dataset.
  Values agree with TIERS["mix50"].rtg_scale (method_tier_grid.py:279) and with plan 2.3's text. NON-BLOCKING: the "mix50" side of
  two of the three recorded changes is the plan's number typed in, not a measurement; a typo would not be caught.

### Item 7 [x] -- Q4 training_inputs vs method_tier_grid.training_inputs
- Side-by-side read: offline/p9_4_mix50_fresh.py:778-813 vs offline/method_tier_grid.py:2609-2664. Same order: tier_dataset,
  stack_dataset, group, scenario_id, _component_streams, selection, declaration check, top decile, iql_reward_scale,
  statistics_digest, provenance (same 7 keys), TrainingInputs with the same filter_stacked_to_streams batches. Same callables
  (grid imports stack_dataset from dt_gate and filter_stacked_to_streams/iql_reward_scale from offline_baselines, method_tier_grid.py:70-101).
  Differences: (a) selection = select_streams (mix50's REGISTERED spec, not the passed spec) + an explicit count check;
  (b) components built unconditionally; (c) A27's two checks omitted -- they fire only for spec.tier "mix_m1000_m500"
  (a27_arms.py:260-261) or "random_fresh", never for mix50/mix50_fresh, so the omission is behaviour-neutral.
- Experiment: gated equivalence test (Item 14 part 2) PASSED on the real corpus for mix50, tensor for tensor.
- Note (NON-BLOCKING): select_streams ignores the spec passed to training_inputs (always TIERS["mix50"]); safe for the two
  callers (control, mix50_fresh) since both are mix50-shaped, and :793 re-checks the count; it would silently mis-select for any
  other mixture spec passed in. No such caller exists (grep).

### Item 8 [x] -- Q5 train_run's train_dt arguments
- offline/p9_4_mix50_fresh.py:898-904 vs corrective_round.py:317-323 (P9.1) vs method_tier_grid.py:2783-2800 (_run_train):
  identical keyword set and values: full_batch, state_dim/n_actions from group, seed, declared_gradient_steps (40000),
  raise_to=None, context_length=dt_gate.CONTEXT_LENGTH (20, same object grid imports), batch_size=BATCH_SIZE (64), device,
  checkpoint_path (partial; P9.1 also partial; _run_train the final path), stats=dataset.stats, scenario_id,
  target_rtg/rtg_scale from inputs.spec (mix50: -5959.0/40223.0), provenance, log_every.
- In dt_gate.train_dt (grep, dt_gate.py:747-900) provenance is only STORED (:881-882) and log_every only PRINTS (:852); seeding is
  Utils.seed_everything(seed) + default_rng(seed) (:810, :828); so the differing provenance/checkpoint path cannot move a parameter.
- P4.7's record of mix50 dt seed 101 (docs/data/p4_7_training.json runs[35]): gradient_steps 40000, target_rtg -5959.0 -- consistent.
- Wrong-reason PASS hole (NON-BLOCKING): control_record (:1009) counts compare_models' all_equal only; two EMPTY state dicts give
  all_equal True with n_parameters 0 (context_sweep.py:1379-1388). Not reachable with train_dt checkpoints (gated test shows real
  P4.7 files hold parameters and differ across seeds), but nothing in control_record asserts n_parameters > 0.
- Wrong-reason FAIL candidates checked: regime entered by train_session when enter_regime=True (CLI :1362 uses the default True;
  device None -> cuda :1163); inputs equal to the registered ones (gated test). None found.

### Item 9 [x] -- Q6a can a mix50_fresh run train before a passed control record?
- train_session (offline/p9_4_mix50_fresh.py:1172-1173) calls assert_control_passed before EVERY NEW_GROUP run; it requires the
  file, format p9.4-control/1.0, declared_gradient_steps == 40000, passed is True, n_seeds_equal == 5 (:1041-1052).
  CLI `train` -> train_session only (:1358-1362); `--plan-only` trains nothing.
- The only other caller of train_run is the smoke (offline/p9_4_smoke.py:85), which trains mix50_fresh WITHOUT the gate by design,
  but only under smoke_roots (:1262-1270: scratch must lie outside /home/filip/rltraffic and the repo) and copy_references refuses a
  scratch output equal to the main output (p9_4_smoke.py:36-38). Not a production bypass.
- train_run itself has NO control check (:871-942): the gate lives only in train_session. Any future caller of train_run bypasses
  it -- NON-BLOCKING design note (no such production caller exists; grep of offline/ and the driver).

### Item 10 [x] -- Q6b a failed control cannot be bypassed by resuming
- A failed record is write_once (:1129) and never rewritten; every later session: ensure_control_record returns the existing record
  (:1123-1125) -> ControlFailed at :1159-1160, BEFORE train_plan; train --plan-only -> train_plan raises (:1101-1103).
  Covered by tests/test_p9_4_training.py:211-218 (two sessions after a 4-of-5 control; nothing of the new tier trains).
- Residual: a crash between the 5th control run and the record write leaves no record; the next session recomputes it (fine).

### Item 11 [x] -- Q6c Amendment A1 re-hash in EVERY session + the hours bound
- train_session :1149-1150 load_declaration (pinned sha256 8f1f5c32...) then assert_files_as_declared (all 200 files re-hashed) BEFORE
  the regime, the control record and any run: every session, yes. Re-hash path is the declared absolute dataset_dir; training reads
  roots.corpus_root, but assert_inputs_as_declared compares dataset_dir STRINGS (:820-824), so a different corpus root refuses.
- EXPERIMENT (hours bound): scratch copy, fake wall clock, fake train_run that "takes" 2 h per run, hours=0.5:
  observed `runs trained: 6, elapsed (fake) h: 12.0, ended_by: the plan`. The train session never reads the clock; its bound is
  enforced ONLY through planning estimates (240 s/run, 120 s/inputs). session_stop_reason (:1066-1074) is not used by train.
  Plan section 7 / CLAUDE.md section 5 say a session ends between units AT its hour bound. Same pattern as P9.1's reviewed
  train_session (corrective_round.py:568-596) and disclosed in the driver echo (p9_4_mix50_fresh.sh:218 "at its plan's end").
  Practical risk small (10 runs ~0.73 h planned). Classified NON-BLOCKING (inherited, disclosed); author may want a deadline check.

### Item 12 [x] -- Q7 write barrier
- Experiment (scratch root exp_barrier/): assert_target ALLOWS output/p9_4 itself and runs/training/../../p9_4/ok.json (lands inside);
  REFUSES runs/../p9_4x/a.json and a relative "output/p9_4/a.json" (resolved against CWD -> outside). write_once(run_root) refused at
  its temporary path (PermissionError), nothing left behind. Symlink escape refused (tests/test_p9_4_training.py:43-47).
- Every write in scope goes through assert_target: write_once (:192-195, temp also checked), write_declaration (:209),
  train_run's three paths (:888-891; mkdir only after the check; os.link never overwrites, :909), control record via write_once
  (:1129). train_dt writes only checkpoint_path (= the checked .partial; dt_gate.py:865-899). No delete except own temp/.partial.
- NON-BLOCKING: assert_target compares against the RESOLVED roots, so if output/p9_4 or output/p9_4_runs were themselves a symlink
  (author-made) the writes would follow it wherever it points. Not reachable from the code (it never creates links).
- Verdict: no write path in plan 2-3 code lands outside output/p9_4/ or output/p9_4_runs/, none overwrites.

### Item 13 [x] -- Q8 mutation tests (scratch copy; runner scratchpad/reviews/A/scripts/mutate.py; output mutants.txt)
- Method: 20 single exact-text replacements in copy/offline/p9_4_mix50_fresh.py (each checked to occur exactly once), each run
  against tests/test_p9_4_selection.py + tests/test_p9_4_training.py (ungated, -x), original restored (cmp with HEAD: identical).
  Baseline in the copy: 30 passed, 6 skipped.
- KILLED (14): M1 select_streams uses mix33's spec; M2 pools swapped; M3 check_v == inverted; M4 check_v drops file hash; M6 control
  passes at 4/5; M7 no control gate before new-tier runs; M8 no A1 re-hash in the session; M10 run_status skips digests; M11 inputs
  check skips stream keys; M12 write_once overwrites (os.replace); M14 trajectory digest over first intersection only; M15 tier_spec
  keeps mix50's components; M16 check_i drops distinct; M20 session ignores STOP between runs.
- SURVIVED (6):
  * M5 build_declaration's plan 2.2 refusal drops `at_maximum != ["expert"]` (:566) -- no test calls build_declaration.
  * M9 train_run passes inputs.scale (IQL's reward scale) as rtg_scale (:902) -- no test pins train_run's train_dt keywords.
  * M18 train_run trains seed+1 (:899) -- same gap.
  * M13 train_plan forgets a failed control (:1102-1103) -- redundant for train_session (ControlFailed first); only `--plan-only`
    would print a plan instead of refusing.
  * M17 assert_control_passed ignores declared_gradient_steps (:1047) -- defence in depth, untested.
  * M19 new_tier_inputs drops the statistics-digest check (:851) -- untested; see below why it matters.
- grep: no P9.4 test calls p94.train_run (real), new_tier_inputs, control_inputs or build_declaration (tests/ grep).
- Reading of the survivors: M9/M18 are FAIL-SAFE at runtime (the control's torch.equal 5/5 would fail -> nothing else trains), but a
  tier-conditional argument change would not be caught by anything; P9.1 had "T-train-args" tests comparing the call keyword for
  keyword (corrective_round.py:281-282), P9.4 has none. M19 matters more: A1's re-hash covers only the 200 SELECTED files
  (:709-710), while the normalisation is fitted over the 400-episode split (declaration n_rows 144000, split_streams 400); the
  200 unselected split files are guarded ONLY by the statistics-digest comparison at :851 (and :842 for the control), which no test
  pins. Present and correct in the code today (read), so NON-BLOCKING, but it is the only integrity check on half the
  normalisation basis.

### Item 16 [x] -- load-bearing vs theatre (from the mutation run + reading)
- Load-bearing (each killed >= 1 mutant): selection.py:65 (registered-route comparison; M1, M2), :104 (M16), :139 (M3, M4),
  :170 trajectory digest by independent byte assembly (M14), :199 (M15); training.py:50 (M12), :90 (M10), :148 (M6),
  :221 (M7), :232 (M20), :267 (M8), :308 (M11); gated training.py:346 (registered training_inputs, tensor for tensor -- real corpus).
- Partly theatre: training.py:258 "..._stops_at_the_bound" tests only train_plan's arithmetic; no session is run past a bound, and
  by experiment (Item 11) a session does NOT stop at a measured bound -- the name overclaims. training.py:370 compares a checkpoint
  with ITSELF (tautological half) -- its seed-101-vs-202 half gives it teeth. training.py:327 recipe test re-asserts what
  recipe_record already raises on (redundant, but it does read the real constants; note recipe_record reads
  grid.DECLARED_GRADIENT_STEPS while train_run uses its own GRADIENT_STEPS -- two constants, only one checked).
- Missing (shown by survivors M5, M9, M13, M17, M18, M19): no test of build_declaration's refusals, of train_run's train_dt keywords,
  of new_tier_inputs/control_inputs (statistics digest), of assert_control_passed's step budget.

### Item 17 [x] -- silent assumptions
- dt_gate.train_dt does not override DTConfig's n_layer/n_head/d_model/dropout (dt_gate.py:811-817), so recipe_record reading the
  defaults is valid; warmup = min(1000, total//2) = 1000 at 40000 steps.
- :587 training_set_return_min = -scale assumes all returns negative (true: max -5959.0). :789 `next(iter(dataset.groups))` assumes one
  group (asserted in build_declaration :575-577, not in training_inputs -- same as the registered code). Exact float compares of
  returns (:566, :686-696 registered) are safe because rewards are integral (Item 4: float32 and float64 sums identical on 200/200).
- Hardcoded mix50 comparators 600 / 40223.0 (Item 5) agree with p4_7_declaration.json tiers.mix50 split_streams 600, rtg_scale 40223.0
  (checked by reading the JSON).

### Extra -- other checks done
- No writes during dataset loading (grep of dt_gate.build_training_dataset, offline/dataset.py, offline/offline_baselines.py: no
  save/write/mkdir), so a refused build_declaration leaves nothing behind.
- transfer_curve.checkpoint_identity (used by reference_checkpoint for the control) is NOT among the branch's transfer_curve.py edits
  (diff hunks are in run_cell, _contrast_block, _in_support_block -- pointer for reviews B/C).
- Mutation runs never touched the worktree: copy restored == HEAD (cmp), pytest run with -p no:cacheprovider, PYTHONDONTWRITEBYTECODE=1.
  Untracked 0-byte read-only dotfiles (.bashrc, .mcp.json, ... at 11:27:40) in the worktree root are sandbox mount placeholders,
  not from this review or the code.

## SUMMARY (assembled at the end)
VERDICT for scope A: PASS-WITH-NOTES. No blocking finding. All 17 checklist items done.
