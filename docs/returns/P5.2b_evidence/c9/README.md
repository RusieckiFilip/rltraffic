# P5.2b — C9: `BRIEF_44` Amendment C, C2 (the artifact, P8.2's table, the suite) — the evidence

Written by the implementer on 2026-10-07 from commands run in this session; the files beside this one are their records.
Branch `task/p5.2b-iql-correction`, `main` merged at `aa65edc` (clean; "Already up to date" at the session's restart).

| commit | what |
|---|---|
| `c7f0178` | C9a: the correction artifact's T-regress (gated: byte for byte the run's, and regenerated from `output/`) and an ungated cross-check against the coordinator's G3 third route — RED, the file absent |
| `9ccd937` | C9b: `docs/data/p5_2b_iql_correction.json` by hand, byte-identical to the run's (`216b9f24…`) — green |
| `5cf57b2` | C9c: P8.2's table tests — RED (`red_c9c_compute_table_tests.txt`: 4 failed) |
| `7d32328` | C9d: the builder (`offline/compute_table.py`) — the corrected entry; the T-regress, the committed-artifact tests and the Q8(c) test red, as expected |
| `1e26bb7` | C9e: the change Amendment A, A2 Q8(c) authorises, its own commit |
| `3ee4819` | C9f: `docs/data/p8_2_compute.json` rebuilt from the clean tree at `1e26bb7` (`38a7f87d…`) by hand — 113 passed, gates open |
| `30b3b65` | C9g: the synthetic `quotes` / `superseded` test, written after the builder change, proven by its mutants |

* **Mutants** (`mutants_c9.json`, `mutants_c9_results.jsonl`, `mutants_c9_stdout.txt`; runner `../c8/run_mutants_c8.py`; each
  COMMITTED in the throwaway worktree `/home/filip/rltraffic-p52b-mut` at `30b3b65`, removed afterwards): **12 / 12 KILLED**.
* **The whole suite** at `30b3b65`, G1's environment (`whole_suite_g1_environment.txt`): `1 failed, 3314 passed, 56 skipped,
  33 warnings in 1342.37s` — the failure `DEFERRED` 110's (`tests/test_rtg_ablation.py::test_the_spread_table_uses_the_whole_
  declared_training_set_not_the_probe_subsample`); +10 passed against C8's run, the ten tests C9 added.
* **The depth-1 run** (`depth1.txt`): `415 passed, 1 skipped in 116.57s` (the known `tests/test_tier_sweep.py:1447`), the clone
  shallow, `c507721` absent, HEAD `30b3b65`.
