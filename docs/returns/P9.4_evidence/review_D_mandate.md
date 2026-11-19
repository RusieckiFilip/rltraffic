# Review D -- PROJECT_PLAN section 7's PRE-FLIGHT REVIEW of P9.4's driver and its write and resume paths
Read COMMON.md first. FINDINGS file: <scratchpad>/reviews/D/FINDINGS.md.
Scope, by the rule (docs/PROJECT_PLAN.md, grep "PRE-FLIGHT REVIEW"): two questions and nothing else -- CAN THIS RUN
DESTROY OR CORRUPT ANYTHING, and DOES THE RESUME / IDEMPOTENCE LOGIC DO WHAT IT CLAIMS. Statistics are out of scope.
Method: FALSIFICATION -- construct the bad state in your scratch copy (synthetic files under a scratch output root you
pass as --output-root, never the main tree's) and show what the code does.
Read: offline/campaigns/p9_4_mix50_fresh.sh (whole, 240 lines); offline/p9_4_mix50_fresh.py lines 141-250 (barrier,
writes), 679-1190 (declaration pin, runs, control, sessions) and 1191-1402 (checks, CLI); offline/p9_4_evaluation.py
lines 150-330 (route i/ii), 330-560 (SUMO cells, pending, roll); offline/p9_4_smoke.py (whole); tests/test_p9_4_driver.py
and tests/test_p9_4_cli.py.
Checks:
1. Every write and delete path (write_once, write_declaration, train_run's .partial and link, the SUMO chunk writes,
   route (ii)'s files via att_rederivation.run_campaign, the report's writes, the driver's rm of the token, logs):
   can any of them write, delete, move or overwrite outside output/p9_4/ and output/p9_4_runs/, or over an existing
   file? Symlink, '..' and relative-path cases.
2. Resume under truncated / stale / wrong-cell / other-code chunks and records, a .partial checkpoint, a checkpoint
   without its record, a record without its checkpoint, a control record that failed, a guard record that failed or
   came from other code: does each refuse or skip exactly as claimed, and never re-run finished work?
3. The driver: every refusal before the token; the token consumed once per session; STOP never created or removed by
   the driver; WORKERS capped; nothing that must stop the run is piped; the run tree must be clean, detached, at the
   commit; the module must load from the run tree; a second driver refused. Run `bash -n` and the driver's own refusal
   paths with bad arguments (they exit 2 before anything) -- do NOT run a stage.
4. The smoke: can it write anywhere but its scratch root? Can removing its scratch root touch the real output/?
Report per check: what you constructed, what happened, PASS or a finding.
