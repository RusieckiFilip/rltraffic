---
description: Run one brief through Explore -> Plan -> Code -> Commit with human gates
argument-hint: <task id, e.g. P1>
---

Run task **$ARGUMENTS** through the four-phase workflow in `CLAUDE.md` §0.
Stop at every gate and wait for me. Do not run two phases in one message.

## Phase 0 — preflight (silent, one message)
`git status --porcelain` and current branch. If the tree is dirty or we are not on `main`, stop and
tell me — do not "helpfully" stash or switch.

## Phase 1 — EXPLORE (plan mode; read only, write nothing)
1. Read `CLAUDE.md`, `docs/CONTRACTS.md`, and the brief for $ARGUMENTS in `docs/briefs/`.
   If the brief says it supersedes other documents, read **only** it — do not reconcile old versions.
2. Delegate bulk repo reading to the `repo-cartographer` subagent so file contents stay out of this
   context. Ask it precise questions, not "explain the repo".
3. Report, in under 30 lines:
   - what the brief assumes about the repo
   - what the repo actually does (with `path:line` evidence)
   - **every disagreement between the two** — the repo wins, but I decide what that means
   - your assumptions, each with a confidence. Anything under ~95% becomes a question to me now.

**GATE 1 — wait for my go-ahead.**

## Phase 2 — PLAN (still plan mode)
Write `docs/plans/$ARGUMENTS.md` containing:
- the files you will create, and the order you will create them in
- the test list, taken from the brief, one line each
- **what you are explicitly NOT doing** (scope fence, copied from the brief)
- open questions, if any remain
Nothing else changes on disk.

**GATE 2 — I read the plan file. Wait.**

## Phase 3 — CODE
1. `git checkout -b task/<id>-<short-name>`.
2. **Tests first.** Write the test file from the brief's test list. Run it. It must fail for the
   right reason — show me the failure output.
   **GATE 3 — I read the tests before you implement. Wait.**
3. Implement until green. Show the real pytest output. If a test fails, fix the code, never the test.

## Phase 4 — COMMIT
`git add` + commit on the task branch. Show `git diff --stat main...HEAD`.
Write `docs/returns/$ARGUMENTS.md` from `docs/returns/TEMPLATE.md`, filled with real command output.
Then tell me to run `/review $ARGUMENTS`. Do not merge.
