# Review C -- P9.4 report: R1-R7, P7.3a's blocks, the sentences S0-S13 (plan sections 5-6)
Read COMMON.md first. FINDINGS file: <scratchpad>/reviews/C/FINDINGS.md.
Read: docs/plans/p9.4_mix50_fresh.md sections 5 and 6 (the templates are binding text); offline/p9_4_report.py
(whole, 823 lines); tests/test_p9_4_report.py, tests/test_p9_4_report_build.py. Registered code by line range:
offline/transfer_curve.py: grep `def _paired_block`, `def _h3_block`, `def _contrast_block`, `def _in_support_block`,
`def _rho_pair`, `def _seed_means_by_draw` and read those bodies; offline/sumo_anchor_rescore.py 1053-1172
(pair_anchor_rows, rescored_rows, hz1x1_rho_block); offline/offline_baselines.py: grep `def paired_comparison`;
offline/dt_gate.py: grep `def mean_ci95` and `def wilcoxon_signed_rank`.
Questions:
1. Each R-block: the right rows (calibrated and source arms), the right anchors (registered: fixedtime k=4 and
   per-link maxpressure; aligned: fixedtime_kcf and maxpressure_1to1), the right key (e_sumo primary, att_env beside;
   CityFlow att_engine primary, att_ours beside), the right sign (left - right; negative = left faster), the unit
   (per-draw seed means over the 100 held-out draws), the CI (mean_ci95) and the test (two-sided Wilcoxon)?
2. R5 (corrected - original, per arm, readings agreement), R6 (CityFlow, P4.7's cells from p9_1_iql_targets.json),
   R7 (anchor_k200 from p7_3b_anchor.json): correct sources and pairing?
3. Every sentence S0-S13: does the code produce EXACTLY the plan's text for each branch, with the branch chosen as the
   plan defines it (CI below/contains/above 0; rho's six branches; S7's two clauses; S10; S12b only if att_ours
   selects a different branch of S12 than att_engine)? Compare string by string against plan section 6.
4. Formatting: can a printed interval contradict its branch? Is any number rounded so it misleads?
5. load_sources: does every refusal precede every number, is completeness checked against the DECLARATION (1,000
   cells) and not against the files present, is the CityFlow join refused unless the routes agree under ==?
6. Mutation tests (in your scratch copy): at least 8 mutants, including sentence-branch and pairing mutants; report.
