# Patches a Claude Code session cannot apply itself

## `claude_guard_hygiene.patch` — wire the test-hygiene check into the guard hook

**Apply with:**
```bash
git apply docs/patches/claude_guard_hygiene.patch
bash -n scripts/claude_guard.sh                      # syntax check
bash scripts/claude_guard.sh --tests-only ; echo "exit=$?"
```
Verified with `git apply --check` on 2026-08-01 against the `scripts/claude_guard.sh` at commit
`9624d73`. If `claude_guard.sh` has changed since, re-derive rather than force.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists
`Edit(scripts/claude_guard.sh)` and `Write(scripts/claude_guard.sh)`, so a session cannot apply it.
That entry is deliberate — CLAUDE.md rule 1: *"a session must not be able to unfreeze itself; the
guard, the hook wiring and the permission deny-list are exactly what stops a wrong assumption from
reaching a frozen file."* The block was **not** circumvented with a Bash heredoc, although that route
exists and the guard explicitly anticipates it: an in-conversation authorisation is a weaker signal
than the configured control, and treating it as stronger is the exact failure the deny-list defends
against. Authorised by the Master chat on 2026-08-01; applying it is a human action by design.

**What it does.**
1. Adds `scripts/check_test_hygiene.sh` to `FROZEN_EXCEPTIONS`, with a dated reason, so the tolerance
   is recorded rather than silent.
2. Runs the hygiene check at the top of the existing `--tests-only` branch.

**Why it hangs off `--tests-only` rather than a new `--hygiene-only` mode.** An earlier draft added a
separate mode. That would have been dead code: `.claude/settings.json` invokes the guard **only** as
`--frozen-only` and `--tests-only`, so a new mode would never fire, and wiring one would additionally
require editing `.claude/settings.json` — also frozen and also denied. Folding it into `--tests-only`
makes the patch self-sufficient, and the check runs on exactly the events that already run pytest.

**Known limitation, accepted knowingly.** `$CHANGED` strips the git status column, so a *renamed* test
file arrives as `old -> new` and escapes the `^tests/` filter. Renames of test files are rare; if that
changes, parse `git status --porcelain -z`.
