# Review B -- P9.4 evaluation: CityFlow routes, SUMO cells, guards G1 and G2, the registered hooks (plan 4, 4.3)
Read COMMON.md first. FINDINGS file: <scratchpad>/reviews/B/FINDINGS.md.
Read: docs/plans/p9.4_mix50_fresh.md sections 4 and 4.3; offline/p9_4_evaluation.py (whole, 680 lines);
`git -C /home/filip/rltraffic-p94 diff 5eedf21 HEAD -- offline/transfer_curve.py offline/att_rederivation.py` (the
hooks); tests/test_p9_4_evaluation.py, tests/test_p9_4_hooks.py. Registered code by line range:
offline/method_tier_grid.py 2879-2991 (_run_evaluate); offline/corrective_round.py 792-822 and 1231-1251 (P9.1's
routes); offline/att_rederivation.py 1169-1350 (run_campaign); offline/transfer_curve.py: grep for `def run_cell`,
`def validate_cell_payload`, `def chunk_is_reusable` and read those bodies; offline/sumo_anchor_rescore.py 360-482,
581-686, 788-800 (sumo_anchor_episode, validate_anchor_cell, compare_with_registered); offline/corrective_report.py
54-98 (_joined_row, _rerolled).
Questions:
1. Is route (i) method_tier_grid._run_evaluate's body (factory, settings, rollout, completeness, payload)? Is route
   (ii)'s factory the SAME as route (i)'s, so their == agreement means what P9.1's meant?
2. The run_cell hook: can it change ANY registered call's behaviour? Can a caller inject inputs for a registered
   subject, an anchor, grid4x4 or P7.3c? Is everything after the subject's inputs the code every cell runs?
3. new_cell_payload: is the prompt the declared one per arm, are the facts the subject's own (five checkpoints), is
   the checkpoint hashed at consumption and checked against its run record?
4. G1 and G2: do they compare the plan's fields (G1 e_sumo and att_env against p7_3a_zero_shot.json; G2 every field
   p9_2's by_draw lists, plus action_sequence_sha256 where present) under == and stop before the FIRST new cell?
   Can a session start new cells with incomplete or failed guards, or a guard record from other code?
5. Resume: a chunk that exists and is invalid -- is it ever overwritten, moved, or silently reused? (falsify with
   synthetic files in your scratch copy)
6. Worker count capped at 6? Batches end at STOP/the hours bound between units of work?
7. Mutation tests (in your scratch copy): at least 6 mutants; report killed/survived.
