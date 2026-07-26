---
description: Independent contract review of the current task branch before merge
argument-hint: <task id>
---

Delegate to the `contract-reviewer` subagent: review task **$ARGUMENTS** on the current branch.

Give the subagent everything it needs in the prompt (it starts with a fresh context): the task id,
the brief path, the branch name, and the instruction to gather the diff itself.

Do not summarize away its findings. Print its report verbatim, then give me your own one-line
recommendation: merge / fix first / escalate to Master chat.
