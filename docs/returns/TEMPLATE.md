# RETURN PACKET — <TASK ID> (<short name>)
**Date:** YYYY-MM-DD · **Mode:** Claude Code / chat · **Contracts:** v1

**Status:** DONE / PARTIAL / BLOCKED

**Branch + diff stat**
```
task/<id>-<name>
$ git diff --stat main...HEAD
<paste real output>
```

**Files produced**
- `path` — one line each

**Tests — REAL OUTPUT, not a summary you believe is true**
```
$ pytest tests/<file> -q
<paste the actual tail, including the pass/fail counts>
```
If you did not run them, write `NOT RUN` and say why.

**Self-review checklist** (Y/N — an honest N is worth more than a false Y)
- Contracts in `docs/CONTRACTS.md` honored, including the non-standard env API? …
- All tests actually executed and green? …
- Zero modifications to frozen files (proved by the diff stat above)? …
- Zero new dependencies? …
- Every number in this packet produced by a command run in this session? …

**Deviations from the brief** (each with a justification)
-

**Conflicts found between brief and repo** (repo wins — what did you implement instead?)
-

**Open questions / risks for the Master chat**
-

**What the next task will assume about this one** (phase-boundary check)
-
