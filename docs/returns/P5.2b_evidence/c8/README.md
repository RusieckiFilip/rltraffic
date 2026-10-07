# P5.2b — C8: `BRIEF_44` Amendment B and B.1 (gate G1, FIX FIRST) — the evidence

Written by the implementer on 2026-10-07, from commands run in this session; the files beside this one are their raw
records. Branch `task/p5.2b-iql-correction`, `main` merged at `b126982` (clean). The design of every item is the plan's
section 14 (`50e3aca`). The harness's `Co-Authored-By` instruction appeared again in this session and was not followed
(`CLAUDE.md` §4b).

## 1. The commits (at most two source files each; tests first)

| commit | what | source files |
|---|---|---|
| `50e3aca` | C8a: the plan's section 14 — where each item lands, what proves it, the two places the amendment and the repo disagree | — |
| `849f472` | C8b: the G1 evidence moved from `docs/notes/p5_2b_g1/` to `docs/returns/P5.2b_evidence/` (B2) | — |
| `5de854f` | C8c: B1.4's identity-swap test and B1.5's cuda-maximum test, green on the code, red against MT2 and the one-device maximum | — |
| `911d974` | C8d: the skeletons (`NotImplementedError`), `Pins.corpus_manifest_sha256`, the fixture's synthetic pin | `offline/iql_correction.py` |
| `364c91d` | C8e: the module's tests, RED: 38 of 117 with the data gates open (`red_c8e_module_tests.txt`) | — |
| `1ad8f79` | C8f: the module, the red tests green; two gated tests of the real pins written WITH the code | `offline/iql_correction.py` |
| `555e28b` | C8g: the driver's tests, RED: 4 of 24 (`red_c8g_driver_tests.txt`) | — |
| `d3d64c8` | C8h: the driver — the late-close branch, the header's restart rule, the pre-flight pinned | `offline/campaigns/p5_2b_correction.sh` |
| `2e5b38d` | C8i: B1.8's test, RED (the report refused on the draws it never reads); RA1's two theatre tests renamed | — |
| `9bda4ca` | C8j: B1.8 — the report verifies every input but the draws | `offline/iql_correction.py` |
| `b4a19f3` | C8k: two of C8's own tests tightened after the mutation run (section 3) | — |

## 2. The red runs

* **C8e** (`364c91d`, gates open): `38 failed, 79 passed in 57.50s`. Every failure is its test's own reason:
  `NotImplementedError` on the five skeletons; `DID NOT RAISE` for each missing refusal — B1.1's `att_engine` case among
  them, which is RA1's MAJOR reproduced: an edited `att_engine` passed every module check; `KeyError` for the missing
  per-seed block, Q1's flag and the seconds' bracket; `('complete', 'partial') != ('complete', 'closing_pending')`;
  the CLI's `manifest` exiting 0 on a partial run. Three tests first failed on the run manifest they had not written
  (a neighbouring refusal); they were fixed BEFORE any implementation existed, so that each fails by `DID NOT RAISE`.
  B1.1's `att_ours` case failed by `Regex pattern did not match`: P8.4b's existing `att_ours` anchor refused first,
  with its own message. The tests of guards that already existed (B1.2, B1.6(c), B.1's M4/M8/M9/M10/M12 guards) were
  green, as designed, and are proven by their mutants (section 3).
* **C8g** (`555e28b`): `4 failed, 20 passed`. The restart with a complete training and its closing canary missing ran
  `status training`, then went straight to (i): no `close-late`, no closing canary. The fresh-run and fully-complete
  cases were green on the old driver, as they must be.
* **C8i** (`2e5b38d`): `FileNotFoundError: 3 held-out draws are not materialised` from the report.

## 3. The mutation run — 47 committed mutant runs, 47 KILLED

Each mutant applied as exact-string edits, COMMITTED in the throwaway detached worktree `/home/filip/rltraffic-p52b-mut`
(created at `9bda4ca`, removed afterwards), the named test file run with the data gates CLOSED (every killing test is
synthetic), `-x` (`run_mutants_c8.py`, the specs `mutants_c8.json`, the records `mutants_c8_results.jsonl`, the console
`mutants_c8_stdout.txt`).

| set | mutants | outcome |
|---|---|---|
| the required re-runs | PM4, MT2, the one-device maximum (and the other device's), RA2's M1, M2, M3, M4, M8, M9, M12 (and M10, a NOTE) | **all KILLED** |
| path (ii)'s three refusals (B1.6(c)) | the resolver pre-check, the refused-cell guard, the final re-check | **all KILLED** |
| one per new refusal, and each new behaviour | the C1 comparison, its identity check, its call; the output root's existence, nesting and digest checks, and `check` without its pins; the hyperparameter comparison and its call; the corpus pin; the corrected level against its cell mean; the original-digest guard; Q1's flag; the corrected sources; `close-late`'s state and foreign-mark refusals; `closing_pending`; the report's mark check; the bracket; the manifest gate and the CLI bypassing it; the per-seed seed check; the reversal; the draws check, both ways | **all KILLED** |
| the driver | the late branch removed; the late branch training again (a second realisation); `close-late` after the canary; the pin removed | **all KILLED** |
| re-run after C8k (`b4a19f3`) | M4, M10 and the draws mutant | **KILLED by `DID NOT RAISE`** |

**Disclosed, from the first run (`9bda4ca`):** M4 and M10 were killed by a `FileNotFoundError` — the report ran past the
disabled guard to the run manifest the test had not written — and the draws mutant by the next refusal down in `check`
(the manifest's existence). Each still died because its guard did not fire, but the neighbouring error masked that. C8k
tightened the two tests (the manifest written after the damage; `check` run before the manifest exists), and the
re-run kills all three by `DID NOT RAISE`. Two kills that pin a refusal's MESSAGE where another check would still refuse
are named as such: `ii-2-refused` (without it, the final re-check refuses with `FileNotFoundError`) and `N-per-seed-seeds`
(without it, `mean_ci95` refuses an empty seed with its own message).

## 4. The checks, the suite, the depth-1 run

* **`check_test_hygiene.sh`** falsified first: a planted `tests/` file with `or True` and a bare `pytest.raises` → TH001,
  TH006, exit 1 (`falsify_hygiene.txt`; the file removed). Then the five task test files: exit 0. With no arguments: exit 1
  on the 16 pre-existing findings in four env/phase test files this branch does not touch (`hygiene_no_arguments.txt`).
* **`check_english.sh`** falsified first: a scratch copy of `offline/iql_correction.py` with a Polish line → exit 1
  (`falsify_english.txt`). Then the twelve task files: exit 0. With no arguments: exit 1 on the four pre-existing hits
  (`.claude/agents/master-coordinator.md:175`, `docs/patches/claude_guard_g1.patch:32`,
  `docs/patches/claude_guard_hygiene.patch:4`, `scripts/claude_guard.sh:47`; `english_no_arguments.txt`).
* **The whole suite** at `b4a19f3`, with G1's exact environment (`OMP`/`MKL` 1; `RLTRAFFIC_OUTPUT_ROOT`,
  `RLTRAFFIC_CORPUS_V11`, `RLTRAFFIC_CORPUS`, `RLTRAFFIC_DRAWS`; `whole_suite_g1_environment.txt`): **`1 failed, 3304
  passed, 56 skipped, 33 warnings in 1099.59s`**. The failure is `DEFERRED` 110's,
  `tests/test_rtg_ablation.py::test_the_spread_table_uses_the_whole_declared_training_set_not_the_probe_subsample`
  (`FileNotFoundError: 'output/p4_dt/dt_seed101.pt'`), untouched by this branch. Against G1 (`8c9f988`: 3,244 passed,
  56 skipped): +60 passed, the 60 tests C8 added, and the same 56 skips. **Disclosed:** my first two runs left
  `RLTRAFFIC_CORPUS` unset (`whole_suite.txt`: 1 failed, 3,297 passed, 63 skipped; `whole_suite_skips.txt` lists
  every skip). The seven extra skips were `tests/test_offline_dataset_corpus.py`'s corpus-backed tests; the G1 command
  read from the session's own record showed the difference, and the third run repeats G1's environment exactly.
* **The depth-1 run** (`BRIEF_42` F.1, `depth1.txt`): `git clone --depth 1 --branch task/p5.2b-iql-correction
  file:///home/filip/rltraffic <scratch>/shallow_c8`; `--is-shallow-repository` `true`; `cat-file -t c507721…` exit 128;
  one commit, HEAD `b4a19f3`, the branch's last code commit; `offline.iql_correction`, `offline.tier_sweep` and
  `tests.p5_2b_fixtures` import from the clone. The four new and changed test files (`tests/test_iql_correction.py
  tests/test_p5_2b_correction_driver.py tests/test_tier_sweep.py tests/test_compute_table.py`), gates open:
  **`405 passed, 1 skipped in 104.08s`** (the skip: `tests/test_tier_sweep.py:1447`, P5.2's campaign output absent).

## 5. The pre-flight stays pinned (B2)

`timed_paths_unchanged.txt` (`timed_paths_unchanged.py`): every function the pre-flight timed —
`training_inputs`, `train_seeds`, the (i) argv and runner, `run_rederivation`, the estimate and the timeouts, and
`tier_sweep`'s `iql_transition_table`, `_run_train_baselines`, `_run_evaluate` and `main` — is byte-identical at
`a71a72c` (the pre-flight's commit) and at the branch tip. `preflight` differs in one line (the barrier's call now passes
the pins), `train_stage` in the hyperparameter check placed between the table's build and the training, and two
recorded fields. **No timed path changed, so the pre-flight was not re-run.** The driver pins it
(`p5_2b_runs/preflight_20261006T195405Z/preflight.json`, sha256 `a2804527…`, re-hashed on disk this session), and
`timeouts` on it still yields `5050.68 2941.69 2891.34`.

## 6. The real record, read-only

`python -P -m offline.iql_correction check` on the real record (the run's out-root, nothing written: `output/p5_2b`
absent before and after) → `ok`, `fresh`: the corpus manifest at the new pin `ebb36187…`; the C1 note's six means equal
under both definitions over P8.4b's 3,000 random-tier cells; the output root at the pinned `SHA256SUMS_p5_2.txt`;
1,152,000 declared rows. With the gates open, the gated tests confirm it, and RA2's measurement by the test's own route:
among the five arms other than IQL, `dt_spatial` is first on seeds 101, 303, 404 and 505 and `dt_nomix` on seed 202,
under both definitions — so if the corrected IQL is no longer first, the AFTER first place reverses on seed 202, and the
artifact will name it.

## 7. For the coordinator (G1.1)

1. **B.1.2's manifest gate is in `manifest_stage`, not in `write_run_manifest`** (plan section 14.3). The CLI's `manifest`,
   the driver's call and the only manual one, runs it. Gating `write_run_manifest` itself breaks seven existing tests:
   one manifests a dummy out-root, and six call it without the synthetic pins a stage check needs. I made none of those
   test changes without a written authorisation. **Question: move the gate, authorising those seven changes?**
2. **B1.3's "the training record marks its seconds":** the training record is written once, before the closing canary
   exists, so the mark is its write-once addendum, `training_random_iql.late_close.json`, written by `close-late` before
   the late canary. The report folds it into `training.seconds` (`bracketed_by`, the training's end, the closing canary's
   time, the mark). P8.2's table quotes that block when it is rebuilt after G2.
3. **B1.6(a):** the output root's two structural checks (it holds `SHA256SUMS_p5_2.txt`; nothing above it does) run on
   every call. The pinned digest is checked where pins are passed: every stage passes them at entry. The barrier's
   helpers keep their signatures, because the existing barrier tests call them on a fake `output/` whose manifest is `x`.
4. **Edited tests, each ordered by the amendment:** T-reproduce (c)'s two halves (B1.7(a)(b)); T-statements' parametrization
   and two case branches (B.1, item 2); RA1's two theatre tests renamed, their bodies unchanged (B1.8). The tests C8 itself
   wrote were changed twice: before the implementation (section 2), and in C8k (section 3).
5. B2 calls `docs/notes/p5_2b_g1/` untracked; it had been committed at `d83d861`. It is now moved, by `git mv`.
