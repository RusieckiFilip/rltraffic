# Common rules for the four P9.4 review mandates (A, B, C, D)

Repository: /home/filip/rltraffic-p94 (a git worktree), branch task/p9.4-mix50-fresh. Review HEAD of that branch.
The binding plan: docs/plans/p9.4_mix50_fresh.md (sections 1-8 and Amendment A1 at its end). P9.4 is EXPLORATORY and
was added after the results; the plan is tagged p9.4-plan (fdf0e2c) and p9.4-plan-a1 (3bfb007).

Interpreter: /home/filip/rltraffic/.venv/bin/python (run pytest as `/home/filip/rltraffic/.venv/bin/python -m pytest -q
-p no:warnings <files>` from /home/filip/rltraffic-p94). The main tree's corpus is /home/filip/rltraffic/datasets_v11,
its outputs /home/filip/rltraffic/output, its materialised draws /home/filip/rltraffic/scenarios/draws: READ ONLY.

You write NOTHING inside /home/filip/rltraffic-p94 or /home/filip/rltraffic. Mutation tests run in a SCRATCH COPY:
    git -C /home/filip/rltraffic-p94 archive HEAD | tar -x -C <your review dir>/copy
then edit the copy and run pytest FROM the copy (cd <copy>; /home/filip/rltraffic/.venv/bin/python -m pytest ...), so
`offline` imports from the copy (check `python -c "import offline; print(offline.__file__)"` from the copy first).
Never run a SUMO or CityFlow episode and never train: the tests that need them are corpus/output-gated and skip
without RLTRAFFIC_* variables; leave those variables unset.

Findings file discipline (mandatory): your FIRST write, within your first three tool calls, is the numbered checklist
in your FINDINGS.md, every item `[ ]`. After each verified fact or experiment, APPEND item number, the command or
construction, the observed result, `<path>:<line>`, and tick the item -- before the next command. A finding not in the
file does not exist. Read only the line ranges your mandate names (you may grep anywhere).

Severity: BLOCKING = could make a reported number wrong, let a stopping control be bypassed, write/delete/move/overwrite
anything outside output/p9_4/ and output/p9_4_runs/, or contradict the plan; NON-BLOCKING = everything else.
Report in the contract-reviewer's output format, assembled from your FINDINGS.md. Time box: about 15 minutes.
