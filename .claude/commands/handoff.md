---
description: Package a finished task for the Master chat and close the session cleanly
argument-hint: <task id>
---

Task $ARGUMENTS is reviewed and merged. Produce the handoff for the Master coordination chat.

1. Print `docs/returns/$ARGUMENTS.md` verbatim.
2. Add a short **Plan delta** section: which checkboxes in `docs/PROJECT_PLAN.md` this task closes,
   and any new task it implies (e.g. a scope fence that turned into a future task id).
3. Add **Assumptions the next phase inherits** — what P<next> will silently rely on because of how
   this was built. This is the phase-boundary check from plan §7; it is the section that catches
   couplings like determinism↔corpus and reward↔RTG before they cost weeks.
4. List anything I still owe someone: mentor questions, pre-registration edits, citations to verify.

Then remind me to `/clear` before the next task.
