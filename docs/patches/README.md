# Patches a Claude Code session cannot apply itself

## `settings_scripts_glob_deny.patch` — glob-deny all of `scripts/`, drop the ten inert `Write(...)` rules

**Apply with:**
```bash
git apply docs/patches/settings_scripts_glob_deny.patch
.venv/bin/python -m json.tool .claude/settings.json > /dev/null && echo "valid JSON"
```
Verified with `git apply --check` on 2026-08-03 against the `.claude/settings.json` at commit `6787f0e`
(deny array 27 entries → 17). If `settings.json` has changed since, re-derive rather than force.
**A restart of any running Claude Code session is required** — permissions are read at session start.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(.claude/**)`, so a
session cannot apply it. Same reasoning as the two entries below; the Bash-heredoc route was again not
taken.

**What it does, in two independent changes.**
1. **Semantic (D3).** `Edit(scripts/claude_guard.sh)` → `Edit(scripts/**)`. The permission layer now
   covers every file in `scripts/`, *including files that do not exist yet*. Deliberately **no**
   permission-level exceptions: `deny` beats `allow` in this system regardless of specificity, so a
   glob and an exception cannot coexist. The two `FROZEN_EXCEPTIONS` (`check_english.sh`,
   `check_test_hygiene.sh`) remain honoured by the **guard**, which is a separate layer — after this
   patch those two files are denied at permission level and permitted at guard level, by design.
2. **Cosmetic.** Removes the ten `Write(...)` deny rules. They are inert: the CLI's own warning is
   *"Write(...) is not matched by file permission checks — only Edit(path) rules are."* Removing them
   is safe **not** because they are inert but because every one of them has an identical-path `Edit(...)`
   twin, and `Edit(path)` provably governs the Write tool. Ten rules that look protective and are not
   are a trap for the next reader.

**Evidence — all three facts established by running the permission system, never by reading about it**
(isolated `/tmp` workspace, `claude -p --settings`, observable = file contents on disk after the
attempt, never the nested agent's self-report):
- `deny` beats `allow`: an explicit `allow: Edit(scripts/check_english.sh)` did **not** survive
  `deny: Edit(scripts/**)`; the file stayed unmodified while a no-rule control in the same run was
  modified, proving the allow array had loaded.
- `Edit(path)` governs the Write tool: a **Write**-tool call succeeded under an `Edit(...)` allow and
  was blocked by an `Edit(...)` deny.
- Pre-flight of this exact deny list: `scripts/brand_new.sh` (a file with no individual rule) DENIED
  and unmodified; `scripts/check_english.sh` DENIED and unmodified; control modified.
- `Read(...)` denies are **not** inert (tested separately) — the inert class is `Write(...)` only.

**The trade, recorded honestly.** D3 buys prospective mechanical cover on every script that does not
exist yet, and pays for it with bounded friction on two temporary files. Measured cost basis: those two
scripts have needed editing about twice in the project's life, at ~2 minutes per patch round-trip, and
the guard's own comments already schedule both exceptions for deletion once they settle. **Revisit
signal:** if either script goes untouched-but-wanting-a-rule *because* of the patch friction, that is
evidence the friction is not as bounded as assumed — reopen the choice rather than absorbing it.

## `runner_thread_pinning.patch` — pin torch to one thread per cell (P0.3-fix)

**Apply with:**
```bash
git apply docs/patches/runner_thread_pinning.patch
.venv/bin/python -c "import experiments.runner as r; print(r.CELL_TORCH_THREADS)"   # -> 1
.venv/bin/pytest tests/test_runner_threading.py -q
```
Verified with `git apply --check` on 2026-08-02 against the `experiments/runner.py` at commit `a0dfc1d`.
If `runner.py` has changed since, re-derive rather than force.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(experiments/*.py)` and
`Write(experiments/*.py)`, so a session cannot apply it. This is the same reasoning recorded for
`claude_guard_hygiene.patch` below, and the Bash-heredoc route was again not taken.

**Authorised** by the Master chat in `docs/briefs/BRIEF_02b_thread_pinning.md`, for
**`experiments/runner.py` only**, on branch `task/p0.3-thread-pinning`, "for the single purpose of
limiting per-worker torch thread counts". The brief's instruction was to use "whatever mechanism the repo
already provides for authorised exceptions ... and if no such mechanism fits, stop and report rather than
weakening the guard". `FROZEN_EXCEPTIONS` does not fit — reaching it would mean editing
`scripts/claude_guard.sh`, which the same authorisation forbids. This directory is the mechanism that
does fit, so `FROZEN_PATTERNS` and `FROZEN_EXCEPTIONS` are left untouched.

**What it does.**
1. Adds `CELL_TORCH_THREADS` and `limit_torch_threads()` to `experiments/runner.py`, with the 2026-07-27
   benchmark table inline as the justification for the value.
2. Calls it in `run_cell`, after the `backend_ready` skip check, so it pins both the pooled path (one
   process per cell) and the sequential path, and a backend-less cell still never imports torch.

**Why `run_cell` and not `ProcessPoolExecutor(initializer=...)` or env vars.** `run_cell` is the unit of
work on both paths, so one call site buys both the parallel win and the `workers=1` win; an
`initializer` reaches only `workers>1`. *(The figures originally quoted here, 5.80x and 1.37x, are
cross-session and were retired 2026-08-03 — the trustworthy measurement is 199.2 s → 50.2 s, ≈3.97x at
workers=6. The argument is unaffected: it turns on **which paths a call site reaches**, not on the size
of the win. More importantly the pin is a **liveness** fix — unpinned parallelism can hang unboundedly,
not merely run slowly — so the justification never rested on speed in the first place.)* Env vars also work today (torch enters lazily via `build_agent`,
so nothing has imported it at fork time) but mutate `os.environ` process-globally and would stop working
silently the day anything imports torch earlier — surfacing as a slow run rather than an error.

**Accepted side effect.** On the sequential path this pins the *calling* process, not a child. That is
what pinning the sequential path asks for (the "1.37x" once quoted here is a retired cross-session
ratio — see the note above), and it is documented in the function's docstring; it does mean
importing `experiments.runner` and running a cell changes the host process's torch thread count.

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
