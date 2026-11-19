# Review A -- P9.4 selection, declaration, training and the control (plan sections 2-3, Amendment A1)
Read COMMON.md (same directory's parent) first. FINDINGS file: <scratchpad>/reviews/A/FINDINGS.md.
Read: docs/plans/p9.4_mix50_fresh.md (whole, 162 lines); offline/p9_4_mix50_fresh.py lines 1-1190;
tests/test_p9_4_selection.py, tests/test_p9_4_training.py. Registered code you may read by line range:
offline/method_tier_grid.py 520-720 and 2565-2665 (tier_spec, training_streams, mixture_training_streams,
assert_declaration_matches_corpus, _component_streams, training_inputs); offline/corrective_round.py 277-406
(P9.1's train_run, the reference for train_dt's arguments); offline/context_sweep.py 819-875 and 1344-1395
(the regime and compare_models); offline/trajectory_logger.py 905-917 (_episode_digest).
Questions, each to be answered by experiment where possible:
1. Is select_streams exactly plan 2.1's rule (mix50's spec, fraction, rng, sort) with only the random pool replaced?
2. Do checks (i)-(v) implement plan 2.1 (v as Amendment A1 states it) and refuse on failure, writing nothing?
3. Does the declaration (build_declaration) hold everything plan 2.4 lists, refuse per plan 2.2 (source target from
   the expert half), compute the 2.3 changes from the manifests, and compute the calibrated target by the registered
   route (targets_table/rule_b_target over P7.2b's statistics)?
4. Is training_inputs method_tier_grid.training_inputs step for step except the selection call? (the corpus-gated
   equivalence test in tests/test_p9_4_training.py exists; you may run it with RLTRAFFIC_CORPUS_V11 and
   RLTRAFFIC_OUTPUT_ROOT set to the main tree's -- it reads only)
5. Does train_run call dt_gate.train_dt with exactly P9.1's/_run_train's arguments (recipe, batch, context, target,
   scale, stats, scenario_id)? Anything that would make the control's torch.equal fail for a wrong reason?
6. Can any path train a mix50_fresh run before a control record with passed=True exists? Can a failed control be
   bypassed by resuming? Is Amendment A1's re-hash before the first gradient step of EVERY session?
7. Write barrier: can any write here land outside output/p9_4/ and output/p9_4_runs/, or overwrite a file?
8. Mutation tests (in your scratch copy): at least 6 mutants of your choosing on the above; report killed/survived.
