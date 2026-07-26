---
description: Read-only investigation of a repo question, without polluting the main context
argument-hint: <question>
---

Investigate: **$ARGUMENTS**

Delegate the file reading to the `repo-cartographer` subagent. Do not read large files into this
conversation yourself — I want the answer here, not the source.

Answer with: the finding, the evidence (`path:line` + a few lines of real code), what you could not
determine, and whether it contradicts `docs/CONTRACTS.md`. Write nothing to disk.
