# rltraffic × Claude Code — setup and working system

Two parts: **A. deployment** (once, ~20 min) and **B. working system** (every day).
Part B matters more. Setup without working discipline is just a faster way to produce code
nobody has read.

---

# PART A — DEPLOYMENT

## A1. Installation (in WSL, not in PowerShell)

```bash
node --version                          # missing? -> Node LTS (nvm or apt)
npm install -g @anthropic-ai/claude-code
cd ~/rltraffic && claude                # first run: browser login
```
In session: `/doctor` (diagnostics), `/status` (model, directory, permissions).
Documentation: https://docs.claude.com/en/docs/claude-code/overview

The repo stays in the Linux FS (`~/rltraffic`), never `/mnt/c/...` — that is in the Decisions Log, and
across a few hundred file operations per session the difference is painful.

## A2. Moving the package from Windows to WSL

The file downloaded from the chat lands in Downloads **on the Windows side**. WSL sees it under
`/mnt/c/`. You do not need to copy it anywhere — unpack it straight into the repo.

```bash
# 1. find your Windows user directory (may differ from your WSL login)
ls /mnt/c/Users/

# 2. substitute the right name and check the files are in place
WINUSER=filip
ls -la /mnt/c/Users/$WINUSER/Downloads/rltraffic_claude_setup_v2.tar.gz
ls -la /mnt/c/Users/$WINUSER/Downloads/project_master_plan_rltraffic*

# 3. look at what is inside BEFORE unpacking into the repo
tar -tzf /mnt/c/Users/$WINUSER/Downloads/rltraffic_claude_setup_v2.tar.gz
#    everything must sit under a single rltraffic_claude_setup/ directory -
#    that is why --strip-components=1 works in the next step

# 4. unpack into the repo
cd ~/rltraffic
git status                       # tree must be clean, you are on main
tar -xzf /mnt/c/Users/$WINUSER/Downloads/rltraffic_claude_setup_v2.tar.gz --strip-components=1
chmod +x scripts/claude_guard.sh
mkdir -p docs/plans

# 5. master plan into the repo - one source of truth instead of a file in Downloads.
#    The browser may have appended (1)/(2)/(3) to the name, so copy the right one EXPLICITLY,
#    not by glob - a glob matching several files will overwrite the target silently.
cp "/mnt/c/Users/$WINUSER/Downloads/<exact-plan-filename>.md" docs/PROJECT_PLAN.md
head -3 docs/PROJECT_PLAN.md     # check this really is the plan, not an old version

# 6. gitignore + commit
cat gitignore_additions.txt >> .gitignore && rm gitignore_additions.txt
git add -A && git commit -m "chore: Claude Code setup, contracts v1.1, brief P1 v2"
```

Unpacking overwrites `CLAUDE.md`, `.claude/`, `docs/CONTRACTS.md`, `docs/briefs/`, `docs/returns/`
and `scripts/claude_guard.sh`. The rest of the repo stays untouched - tar adds, it does not clean
directories. If you unpacked an earlier version of the package, first delete
`docs/briefs/ADDENDUM_A_PATCH.md` and `docs/briefs/BRIEF_01_DELTA.md` - v2 does not contain them, so
they will not disappear on their own, and they contradict Brief #1 v2.

The package is already merged with the master chat's return — `settings.json` and `claude_guard.sh` in
the dual-mode version, `docs/CONTRACTS.md` with C6 v1.1, `BRIEF_01_v2` in `docs/briefs/`, and the
outdated `ADDENDUM_A_PATCH.md` and `BRIEF_01_DELTA.md` removed. You do not need to assemble anything
by hand.

**One change relative to what the master chat sent:** in `settings.json` the hook paths are now
`bash "$CLAUDE_PROJECT_DIR/scripts/claude_guard.sh"` instead of a relative path. Hooks do not always
fire from the repo root — with a relative path the hook silently fails to find the script, and then
you have no protection at all and will not learn about it.

## A3. Verification — mandatory, 3 minutes

Not "I checked that the script works", but "I checked that the hook fires".

```bash
# 1. the script on its own
echo "# test" >> envs/base_traffic_env.py
bash scripts/claude_guard.sh --frozen-only ; echo "exit=$?"   # expected: exit=2 + BLOCKED
git checkout -- envs/base_traffic_env.py
```

```bash
# 2. the hook in a session — the only real test
claude
```
```
/hooks                    # both PostToolUse entries visible?
/permissions              # deny on envs/**, agent/base.py, ...?
!ls .claude/agents/       # contract-reviewer, repo-cartographer, citation-verifier — three .md files?
```
The `/agents` wizard was removed in Claude Code v2.1.220 — subagents are no longer clicked together in
a menu. They are defined by **`.md` files in `.claude/agents/`** (or `~/.claude/agents/` for all
projects). Verification is therefore two-step: `!ls .claude/agents/` confirms the files exist, and
invoking one by name (*"use the repo-cartographer subagent to ..."*) confirms the session actually
registers it. `ls` alone is insufficient — a file with broken frontmatter sits on disk but never loads.

Then, in the session, ask: *"append a `# probe` comment at the end of `envs/base_traffic_env.py`"*.
The correct outcome: permissions block the edit **or** the hook returns BLOCKED. If the file changed
without protest — you have a hole, fix it before going further.

## A4. Terminal or VS Code

- **VS Code (WSL Remote) + the Claude Code extension** — daily work. You install the extension on the
  WSL side. You open the repo with `code .` from `~/rltraffic`. Inline diffs matter here: this is code
  whose results will end up in the paper's tables.
- **A separate terminal + `tmux`** — everything above a few minutes: P2.1 (MAPPO ≥500 episodes),
  P2.2 (corpus campaign). Never inside a Claude Code session. The agent reads logs, it does not hold
  the process.

---

# PART B — WORKING SYSTEM

## B1. Layers — what lives where

Your *Goals → Requirements → Spec → Implementation* chain is already implemented, only split across
two tools. Claude Code does not have to interrogate you about goals, because the goals are in the repo.

| Layer | Artifact | Where it is produced |
|---|---|---|
| Goals | claims C1–C3, headline contribution | Master chat (claude.ai) |
| Requirements | plan §5–§7, contracts §4 | Master chat → `docs/PROJECT_PLAN.md`, `docs/CONTRACTS.md` |
| Spec | task brief | Master chat → `docs/briefs/BRIEF_XX.md` |
| Implementation | code + tests | **Claude Code** |
| QA | contract review | **Claude Code** (`/review`, subagent in a fresh context) |
| Release | merge + Return Packet | Claude Code → `/handoff` → Master chat |
| Iterate | plan update, next brief | Master chat |

**One brief = one sprint = one session = one branch = one Return Packet.** That is the unit of work.
Brief §7 caps it at ≤2 source files + tests — that is not a formality but the boundary below which
review is feasible at all.

## B2. The single-task loop

```
Shift+Tab -> plan mode          (always; visible in the footer)
/task P1
  Phase 1 EXPLORE  -> GATE 1: you read the brief↔repo discrepancies, you accept
  Phase 2 PLAN     -> GATE 2: you read docs/plans/P1.md, you accept
  Phase 3 CODE     -> GATE 3: you read the tests BEFORE the implementation exists
  Phase 4 COMMIT   -> Return Packet in docs/returns/P1.md
/review P1                       -> contract-reviewer, fresh context, read-only
  (fixes -> back to phase 3)
merge into main                  -> only after PASS
/handoff P1                      -> package for the Master chat + what the next phase inherits
/clear
```

### Why gate 3 is the most important one
The tests in Brief #1 encode the alignment convention (`observations T+1, decisions T, r_t from row
t+1`). If the implementation comes first, the tests will be written to confirm it — and an off-by-one
enters the data format, where it costs a corpus regeneration. Test 3 from the brief (exact equality of
`-np.sum(lane_waiting_vehicle_count[t+1])` vs `global_reward[t]`) is the only place where this
convention is genuinely enforced. Read it yourself before the logger exists.

### The 95% rule
It is in `CLAUDE.md`: the agent must list its assumptions with a confidence level and **ask instead of
guessing** below ~95%. Do not delete this as chatter — in this project a bad assumption does not blow
up with an error, it produces a plausible number.

## B3. Context — one principle and four habits

> **The repo is the memory. The context window is scratch space.**
> Whatever must survive lands on disk: the plan in `docs/plans/`, the convention in a docstring, the
> result in `docs/returns/`. Nothing important stays only in the conversation.

1. **`/clear` between tasks. Always.** Context from P1 in a P2 session is not a saving, it is a source
   of silent assumptions nobody stated out loud.
2. **Delegate reading to subagents.** `/explore` and `repo-cartographer` read files in their own
   context and return a conclusion with `path:line`. The main session gets the answer, not the
   sources. This is the most effective context-saving technique you have.
3. **`/compact` only mid-task and always with an instruction**, e.g.:
   `/compact keep: the alignment convention, the decisions taken, the last pytest output; drop: contents of files that are already on disk`.
   An instructionless compact throws out exactly what was hard to establish.
4. **If you are explaining the same thing to the agent a second time — it does not belong in the
   conversation, it belongs in `CLAUDE.md`.**

## B4. Antipatterns (each one would genuinely cost this project)

| Do not | Why |
|---|---|
| "tests should pass" without running them | `CLAUDE.md` §2 forbids it; enforce it in the Return Packet |
| fixing a test so that it passes | default hypothesis: the code is wrong. If the test is wrong — stop, and the decision is yours |
| skipping `/review` "because it is a small change" | the same context that wrote the code is the worst reviewer of that code |
| two tasks in one session | see B3.1 |
| a long simulation inside the agent's session | it eats context and blocks the session; `tmux` |
| always-on skills of the ponytail/caveman kind | YAGNI will delete the state machine, the episode hash and the lane-change guard — that is, exactly what protects the corpus |

## B5. Parallel work (useful at P2)

When P2.0 (randomizer) and P2.1 (MAPPO training) overlap, do not run two sessions in one directory:

```bash
git worktree add ../rltraffic-p20 -b task/p20-randomizer
cd ../rltraffic-p20 && cp -r ~/rltraffic/.claude .   # hooks do not travel to a worktree by themselves
```
Every worktree = its own session, its own branch, zero collisions.

## B6. Daily checklist

- [ ] `git status` clean, I am on `main`
- [ ] I know which brief I am doing and that it is the **only** source of truth for that task
- [ ] plan mode on
- [ ] after the task: `/review` → PASS → merge → `/handoff` → `/clear`
- [ ] Return Packet pasted into the Master chat

---

## Appendix: two known weaknesses of the guard (for a decision, I did not change them)

`claude_guard.sh` is the master chat's version and was tested by it — I left it unchanged. Two things
are worth considering once they start to hurt:

1. **`--tests-only` runs `pytest tests`** — the whole test tree. If `tests/` contains tests requiring
   SUMO/CityFlow, the hook will return exit 2 for reasons unrelated to the change, and the agent will
   start "fixing" someone else's tests. Fix: a `CLAUDE_GUARD_TESTS` variable defaulting to the current
   task's test file.
2. **Parsing `git status --porcelain` with `awk '{$1=""}'`** loses renames (`R  old -> new`) and files
   with spaces. A frozen file renamed instead of edited will pass unnoticed. Fix: `cut -c4-` or
   `git diff --name-only HEAD`.

Both are unlikely today and cheap to fix later — which is why they are here and not in the code.
