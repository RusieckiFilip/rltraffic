# P5.2b — gate G1 evidence (`BRIEF_44` Amendment A, A4): the pre-flight, the mutation run, the suite, the depth-1 run

Written by the implementer on 2026-10-06 so that the evidence of commits C2–C7 exists in the repository and not only in the
session's scratchpad (`/tmp` is tmpfs on this machine: `PROJECT_PLAN` §7, *an output that exists in one place is not a
record*). Every number below was produced by a command run in this session; the files beside this one are its raw
records. Branch `task/p5.2b-iql-correction`.

**Moved** from `docs/notes/p5_2b_g1/` (committed at `d83d861`) to `docs/returns/P5.2b_evidence/` on 2026-10-06, as
`BRIEF_44` Amendment B, B2 rules (`docs/notes/` is the coordinator's). The files are unchanged but for this paragraph;
C8's evidence is added beside them under `c8/`.

## 1. The G1 pre-flight (Amendment A, A3.1) — RUN, COMPLETE

Command (the driver's documented form), from the committed, clean task worktree at `a71a72c` (`main` merged), with the
machine on mains and the Windows power mode at Best Performance (P8.2's `power-check` exit 0 before the run):

```
bash offline/campaigns/p5_2b_correction.sh --preflight 2>&1 | tee -i -a /home/filip/rltraffic/output/p5_2b_runs/preflight_capture.txt
```

Driver exit 0, 19:54:05Z → 19:55:09Z. The record: `output/p5_2b_runs/preflight_20261006T195405Z/preflight.json`, sha256
`a280452734494479f6d2941b09ff12c16825155080404ee7ce84b95752c4bb4b` (copied here as `preflight_20261006T195405Z.json`,
the same bytes). Every file of the pre-flight directory and the capture are listed in
`output/p5_2b_runs/SHA256SUMS_preflight_20261006T195405Z.txt` (13 lines, all re-verified OK; the list's own sha256
`9a4a17582f4fd02c8e2c1c8a673504d85a1cdfcd14198da4af95407aee24eb6c`; copied here).

| what | value (from the record) |
|---|---|
| status | **COMPLETE**, no reason |
| code | `a71a72cf76b3b1397e7b201889393aafbee56549`, `dirty` false, code root `/home/filip/rltraffic-p52b` |
| regime | CUDA (NVIDIA GeForce RTX 5080 Laptop GPU), torch `2.11.0+cu128`, one torch thread, `OMP`/`MKL` 1, `CUBLAS_WORKSPACE_CONFIG` unset, deterministic algorithms off |
| canaries | open **0.7607 s**, close **0.7665 s**, both *at speed*, reproduced, AC overlay Best Performance |
| the corrected table, built on the real tier | **1,152,000** rows, **3,200** streams, reward scale **0.7429420505200595**, normalisation statistics equal to the five original checkpoints'; built in **18.80 s** |
| rehearsal training (seed 101, CUDA) | **2,000** steps in **16.65 s** (gradient loop) = **8.32 ms/step**; the checkpoint is a rehearsal (`correction.rehearsal`), never a candidate |
| **T-reproduce (a)** — P5.2's evaluate path on the ORIGINAL `grid4x4_random_iql_seed101.pt`, draws 1000–1004 | **reproduced: 5 of 5 episodes, 0 differences** (`att_horizon`, `horizon_vehicle_count`, `episode_reward` under `==`), exit 0, **9.81 s** (1.96 s/episode) |
| **T-reproduce (b)** — P8.4b's `run_campaign` on the same checkpoint and draws | **reproduced: 5 of 5 cells, 0 refused, 0 differences** (every field but the two timings, against P8.4b's committed cells), **9.64 s** (1.93 s/episode) |
| estimate (an estimate, not a bound) | training **1,683.6 s** (table + 5 × 40,000 × 8.32 ms), (i) **980.6 s**, (ii) **963.8 s** — **3,627.9 s ≈ 60.5 min**, plus two canaries, the file list and the report |
| stage timeouts (3 × the estimate, floor 600 s) | train **5,050.7 s**, (i) **2,941.7 s**, (ii) **2,891.3 s** |

**The pin the run needs, for the coordinator to accept at G1** (the driver refuses while it is UNSET):

```
PREFLIGHT_RECORD=p5_2b_runs/preflight_20261006T195405Z/preflight.json
PREFLIGHT_SHA256=a280452734494479f6d2941b09ff12c16825155080404ee7ce84b95752c4bb4b
```

An earlier attempt did not run: at 19:45Z P8.2's `power-check` refused (*"the AC power mode is Better Battery"*); nothing
was written then. The author set Best Performance before the run above.

## 2. The mutation run — 55 committed mutant runs, every mutant dead at `8c9f988`

Each mutant was applied and COMMITTED in a throwaway detached worktree (`/home/filip/rltraffic-p52b-mut`, removed
afterwards), the named tests run, the outcome recorded (`run_mutants.py`, with `mutate.py` and `mutate_json.py`).

| set | base | runs | outcome | records |
|---|---|---|---|---|
| `DEFERRED` 107's tests against P8.2's surviving mutants (M2-2a, M2-2b, M2-3e, M2-3f, M2-6c and its recommitted form, the C2 range of M2-7a's note) | `5c74b3f` | 7 | **7 KILLED** | `mutants_deferred107.jsonl` |
| batch 1: §4.1, the argv, the route, the training arguments, the barrier, the statements, the report, the file list, the stage guards, the pre-flight, the timeouts | `ca4840c` | 31 | **27 KILLED, 4 SURVIVED** | `mutants_batch1.json`, `mutants_results.jsonl` |
| the four survivors, re-run after `8c9f988`'s four new tests | `8c9f988` | 4 | **4 KILLED** | `mutants_rerun.json`, `mutants_results.jsonl` |
| batch 2, gated on the real record: T-protocol on both paths, a held-out draw changed in `dt_gate` (with and without the module's refusal), the mean route, T-rows (the brief's named mutation: filter removed → `assert 2304000 == 1152000`), T-scale | `8c9f988` | 8 | **8 KILLED** | `mutants_batch2.json`, `mutants_results.jsonl` |
| batch 3, the driver: the token before the power check, `-P` dropped, one try for (i), typed timeouts, a pre-flight that needs the token | `8c9f988` | 5 | **5 KILLED** | `mutants_batch3.json`, `mutants_results.jsonl` |

**The four batch-1 survivors were real test gaps** and were closed by new tests, no existing assertion changed:
`M-scale-all-streams` (the synthetic tier's every-stream return span equalled the selected one's — the fixture now widens
it), `M-q2a-ignores-ties` (no case tied `dt_nomix` with an arm after it in `METHODS`), `M-eval-stage-no-model-guard` (the
happy path cannot see the guard), `M-declared-rows-unchecked` (`assert_training_inputs`' refusals untested).

## 3. The suite, the guards and the depth-1 run

- **Whole suite**, gates open, at `8c9f988`: `1 failed, 3244 passed, 56 skipped, 34 warnings in 1200.02s`. The failure is
  `tests/test_rtg_ablation.py::test_the_spread_table_uses_the_whole_declared_training_set_not_the_probe_subsample`
  (`FileNotFoundError: 'output/p4_dt/dt_seed101.pt'`, a path relative to the working directory in a worktree without
  `output/`), untouched by this branch — the coordinator recorded it as `DEFERRED` 110.
- **`check_test_hygiene.sh`**: falsified first (a temporary `tests/` file with `or True` and a bare `pytest.raises` → TH001
  and TH006, exit 1; the file removed), then clean on the five new and changed test files (exit 0); with no arguments, the
  16 pre-existing findings, all in four env/phase test files this branch does not touch.
- **`check_english.sh`**: falsified first (a scratch copy of `offline/iql_correction.py` with a Polish line → exit 1), then
  clean on the nine task files (exit 0); with no arguments, the four pre-existing hits (`.claude/agents/master-coordinator.md`,
  two `docs/patches/*.patch`, `scripts/claude_guard.sh`).
- **Depth-1 run** (`BRIEF_42` F.1): `git clone --depth 1 --branch task/p5.2b-iql-correction file:///home/filip/rltraffic
  <scratch>/shallow`; `--is-shallow-repository` true; `cat-file -t c507721…` exit 128; one commit, HEAD `8c9f988` = the branch
  tip; `offline.iql_correction`, `offline.tier_sweep` and `tests.p5_2b_fixtures` checked to import from the clone; `tests/test_compute_table.py tests/test_tier_sweep.py
  tests/test_iql_correction.py tests/test_p5_2b_correction_driver.py` with the gates open: **`345 passed, 1 skipped in
  99.35s`** (the skip: `tests/test_tier_sweep.py:1447`, P5.2's campaign output absent from the clone).

## 4. Disclosed

- The stage functions' end-to-end tests (`0c1ca93`) and `8c9f988`'s four tests were written AFTER the code they test; their
  strength rests on the mutants of §2 (`M-train-record-no-runs`, `M-train-no-canary`, `M-p8_4b-no-recompute`,
  `M-preflight-always-complete`, `M-timeouts-no-reproduce`, the four re-runs), not on a red run.
- `implementer_log.md` is the session's working log, verbatim.
