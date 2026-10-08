# P5.2b working log (scratchpad; the packet is the record)

## Session state
- Branch task/p5.2b-iql-correction; plan b40bc3f; merged main afeb231 (Amendment A) as bde7f6a.
- C2 commit 5c74b3f: DEFERRED 107 tests (10 passed, 98 deselected; file baseline before: 98 passed in 21.19s gates open).
- Throwaway worktree /home/filip/rltraffic-p52b-mut (detached), base 5c74b3f.

## DEFERRED 107 mutants (all KILLED)
| mutant | sha | mode | result | reason |
|---|---|---|---|---|
| M2-3f | 0398504 | FULL | 3 failed, 105 passed in 19.79s | assert ('24.6','9.5') == ('9.5','24.6'); ('5.2','30.1') == ('30.1','5.2') |
| M2-3e | 10f06d8 | FULL | 3 failed, 105 passed in 18.83s | assert ('35.9','6.6','12.3') == ('6.6','35.9','12.3'); ('2.2','41.1',..); committed ('35.9','6.6','76.5') |
| M2-2a | f54c4ed | CI | 1 failed, 89 passed, 18 skipped in 4.53s | summary floors != recomputed from groups (test_d107_every_committed_summary...) |
| M2-2b | 5324723 | CI | 1 failed, 89 passed, 18 skipped in 3.52s | devices list != 9 cpu + 9 cuda (test_d107_the_committed_groups...) |
| M2-6c | 193749d | CI | 1 failed, 89 passed, 18 skipped in 3.36s | assert (2, 0.0378...) == (4, 0.0378...) (synthetic) |
| M2-6c recommitted | b7d37d4 | CI | 3 failed, 87 passed, 18 skipped in 3.37s | (7, 0.01303527676708005) == (14, ...) x2 + synthetic |
| C2 (M2-7a note) | 45e302e | FULL | 1 failed, 107 passed in 20.98s | 'for i, actor in enumerate(self.actors)' not in lines 125-145 |

## Commits on task/p5.2b-iql-correction (after Amendment A merge bde7f6a)
- 5c74b3f C2 DEFERRED 107 tests (10 passed)
- d5a770d C3 skeletons + red tests (tier_sweep -k p5_2b: 6 failed, 1 on 'assert 384 == 192' + 5 NotImplementedError; test_iql_correction 50 failed + 2 errors all NotImplementedError; driver 11 failed + 7 errors all FileNotFoundError)
- 2ded253 C4 §4.1 (6 passed; tier_sweep file 151 passed, 1 skipped)
- f78132c C5-C6 module (44 passed 8 skipped gates closed; 203 passed 1 skipped iql+tier_sweep gates open 58.90s). Fixture fix: write_synthetic_run eval-only call.
- 0c1ca93 C6 e2e stage tests (written AFTER stages -> need mutation evidence); 53 passed 8 skipped closed; 61 passed 52.63s open
- ca4840c C7 driver (18 passed 14.37s); echo wording 'attempt'->'try' (no-outcome regex catches 'att')
- Gated real-data run (C5-C6): T-rows setup 20.55s call 3.27s; T-reproduce (a) 11.80s, (b) 10.83s; (c) att_engine 3.76s

## Mutation run (results in mutants_results.jsonl; throwaway worktree /home/filip/rltraffic-p52b-mut)
- Batch 1 at ca4840c: 31 mutants, 27 KILLED, 4 SURVIVED (M-scale-all-streams, M-q2a-ignores-ties, M-eval-stage-no-model-guard, M-declared-rows-unchecked)
- 8c9f988 C7b: four new tests + fixture (second episode of a draw x3 rewards) closing them; re-run @8c9f988: 4/4 KILLED
- Batch 2 (gated, real record) at 8c9f988: 8/8 KILLED (incl. G-rows-filter-and-refusal 'assert 2304000 == 1152000' = brief's named mutation; G-rows-filter-only refusal fires; G-scale-all-streams T-scale; G-heldout-draw both variants; G-mean-route)
- Batch 3 (driver) at 8c9f988: 5/5 KILLED

## Suite / guards at 8c9f988
- whole suite (gates open, worktree): 1 failed, 3244 passed, 56 skipped, 34 warnings in 1200.02s; the failure is tests/test_rtg_ablation.py::test_the_spread_table_uses_the_whole_declared_training_set_not_the_probe_subsample -- FileNotFoundError 'output/p4_dt/dt_seed101.pt' (relative output_root in a worktree without output/), untouched by this branch
- hygiene falsified (TH001+TH006 on a temp file, exit 1) then 5 task test files exit 0; whole suite 16 pre-existing (4 frozen env/phase files)
- english falsified (scratch copy, exit 1) then 9 task files exit 0; all tracked: 4 pre-existing hits (.claude/agents/master-coordinator.md:175, docs/patches/claude_guard_g1.patch:32, docs/patches/claude_guard_hygiene.patch:4, scripts/claude_guard.sh:47)
- depth-1 clone (F.1): shallow true; cat-file c507721 exit 128; 1 commit; HEAD 8c9f988 = branch tip
- depth-1 run (F.1) at 8c9f988: imports from the clone; 4 files gates open: 345 passed, 1 skipped (tests/test_tier_sweep.py:1447 campaign output absent) in 99.35s
- power-check at 19:45Z: REFUSED, AC power mode Better Battery (961cc777...), not Best Performance -> pre-flight not run; asked the author
