---
description: Print the frozen interface contracts and check the working tree against them
---

1. Print the key points of `docs/CONTRACTS.md` in <=15 lines — especially the non-standard env API.
2. Run `git status --porcelain` and list any modified file that falls under the frozen set defined in
   `CLAUDE.md` rule 1.
3. If any frozen file is modified, tell me exactly how to revert it and why it matters.
