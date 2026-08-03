# BRIEF #3 — P0 close-out: guard defect G1, liveness docs, reproducibility, test hardening

**Mode:** Claude Code, in the repo, on a task branch.
**Branch:** `task/p0-closeout`
**Supersedes:** nothing. This is the single source of truth for P0.5 / P0.6 / P0.7 / P0.8.
**Issued:** 2026-08-03 by the Master chat.

> Read this whole brief before planning. It contains **two dated frozen-file authorisations** and
> **two mandatory handoff points** where you stop and wait for a human. Missing either turns a
> correct implementation into a blocked one.

---

## 1. Why this task exists

P0 is closed except for four items. None is on the paper's critical path, and exactly one is a live
hole rather than a documentation or reproducibility gap. That asymmetry sets the order.

| Task | What it is | Severity |
|---|---|---|
| **P0.8** | Guard defect **G1**: new `.py` files in a new subdirectory of `experiments/` escape the frozen-file check | **Live hole** in an enforcement mechanism |
| **P0.5** | `runner.py` says the thread pin is a speed optimisation; it is a **liveness** fix (review finding N1) | Latent — invites a future maintainer to reintroduce a hang |
| **P0.6** | Timing evidence and the §3.1 sanity anchors are not reproducible from the repo (N5 + a new finding) | Reproducibility |
| **P0.7** | `tests/test_runner_threading.py`: one unbounded wait, two non-asserted preconditions (N7, N8), plus retired ratios in its docstring (N3) | Test quality |

**Order is binding: P0.8 → P0.5 → P0.6 → P0.7.** One commit per task. They share no logic, so a
later one failing must not block an earlier one from merging.

**Deviation from the ≤2-source-file rule, declared rather than hidden.** §7 of the plan caps a brief
at ~2 source files plus tests. This brief touches four files. I am bundling them anyway because they
are independent, none changes program logic except a two-line regex, and the total diff is small. If
during planning the diff looks like it will exceed ~200 lines, stop and say so — I will split it.

---

## 2. Frozen-file authorisations (quote the relevant one verbatim in your Return Packet)

> **AUTHORISATION A — 2026-08-03, Master chat, for P0.8.**
> `scripts/claude_guard.sh` may be modified **for the single purpose of fixing defect G1**: making the
> `experiments/` clause of `FROZEN_PATTERNS` robust to `git status --porcelain` collapsing untracked
> directories, and adding the corresponding `experiments/configs/` carve-out to `FROZEN_EXCEPTIONS`.
> Delivered as a patch under `docs/patches/`, applied by a human. **No other change to that file, and
> no change to any other file under `scripts/`, is authorised.** Spent when P0.8 merges.

> **AUTHORISATION B — 2026-08-03, Master chat, for P0.5.**
> `experiments/runner.py` may be modified **for the single purpose of documenting the liveness role of
> `limit_torch_threads()`**. **Comments and docstrings only — zero executable-statement changes**,
> proven mechanically (§5.2, the AST check). Delivered as a patch under `docs/patches/`, applied by a
> human. This does not authorise moving the call site, changing `CELL_TORCH_THREADS`, or any behaviour
> change whatsoever. Spent when P0.5 merges.
>
> *The P0.3-fix authorisation ("limiting per-worker torch thread counts") is spent and does not
> cover this.*

**Both files are also denied at permission level** (`Edit(scripts/**)`, `Edit(experiments/*.py)`), so
you physically cannot apply either patch. That is intended. Write the patch into `docs/patches/`,
verify with `git apply --check`, add a `docs/patches/README.md` entry following the existing ones, and
stop at the handoff point. **Never** reach these paths with a heredoc, `cp`, `sed -i`, `tee` or
`patch` — the guard anticipates that route and taking it is the exact failure the deny-list exists to
prevent.

---

## 3. Frozen context you must not re-derive

Current guard regexes, read from `scripts/claude_guard.sh` on 2026-08-03 (lines 36 and 47):

```bash
FROZEN_PATTERNS='^(envs/|agent/base\.py|agent/utils/utils\.py|agent/MAPPOAgent\.py|algorithms/|rewards\.py|states/|metrics/|utils/|scripts/|\.claude/|CityFlow/|experiments/.*\.py$)'
FROZEN_EXCEPTIONS='^scripts/(check_english|check_test_hygiene)\.sh$'
```

Verified facts you may rely on without re-checking:
- The guard derives everything from `git status --porcelain`, piped through
  `awk '{ $1=""; sub(/^ +/,""); print }'` (line 49), which strips the status column.
- `run_matrix` builds a `ProcessPoolExecutor` **only** under `if workers and workers > 1:`
  (`runner.py:447`). The sequential path never forks.
- `run_matrix` collects with `as_completed` + `future.result()` and **no timeout**
  (`runner.py:463-466`).
- `limit_torch_threads()` is called once, at `runner.py:293`, after the `backend_ready` skip check.
- Experiment outputs resolve to `output/experiments/<name>` (`experiments/config.py:355`) — **nothing
  is written under `experiments/` at runtime**.
- `output/` is gitignored (`.gitignore:219`).

---

## 4. P0.8 — fix guard defect G1  ← **DO THIS FIRST**

### 4.1 The defect, reproduced

`git status --porcelain` collapses a **wholly-untracked directory** into a single entry, so the guard
never sees the file path. Measured on 2026-08-03 (`docs/notes/D3_falsification.md` §4):

```
path created                    what git status REPORTS         guard verdict
experiments/runner.py           [experiments/runner.py]         BLOCKED     correct
experiments/newpkg/foo.py       [experiments/newpkg/]           PERMITTED   <-- DEFECT
envs/newpkg/foo.py              [envs/newpkg/]                  BLOCKED     correct
experiments/configs/new.json    [experiments/configs/]          PERMITTED   correct (intended)
```

`experiments/.*\.py$` is the only frozen pattern that both matches many paths **and** needs the full
path to match. The others are directory prefixes (survive the collapse) or four exact file paths
(`agent/base.py`, `agent/utils/utils.py`, `agent/MAPPOAgent.py`, `rewards.py` — cannot be created
inside a new subdirectory, and sit in tracked directories).

Not reachable via Edit/Write, which the permission layer denies. **Reachable via a Bash heredoc**,
which is the exact route the guard's own header says it exists to cover.

### 4.2 Required change to `scripts/claude_guard.sh`

Convert the `experiments/` clause from extension-anchored to prefix-anchored, and move the deliberate
carve-out into the exceptions list where a reader can see it:

```bash
# ... |experiments/)                     # prefix, like every other directory entry
FROZEN_EXCEPTIONS='^(scripts/(check_english|check_test_hygiene)\.sh$|experiments/configs/)'
```

Mind the grouping: the `$` anchor must continue to apply **only** to the `scripts/` alternative, or
`experiments/configs/` will never match. Add a dated comment above `FROZEN_PATTERNS` stating the
collapse behaviour and why the clause is a prefix — a future reader will otherwise "tidy" it back.

Safe because runtime artifacts go to `output/experiments/`, so a prefix rule cannot misfire on
results, plots or summaries. `experiments/configs/` stays writable, which is the documented intent.

### 4.3 The load-bearing test — `tests/test_claude_guard.py` (NEW FILE)

**The guard currently has zero tests** (checked 2026-08-03: no file under `tests/` mentions
`claude_guard`). The mechanism protecting the entire frozen set is untested, which is why G1 survived
several close readings of its regexes. This test is the most valuable artifact in this brief.

Build a scratch git repo under `tmp_path`, copy the **real** `scripts/claude_guard.sh` into it,
commit a baseline, then assert this truth table by running the real script:

| Path made dirty | Expected |
|---|---|
| `experiments/runner.py` | BLOCKED |
| `experiments/newfile.py` | BLOCKED |
| **`experiments/newpkg/foo.py`** | **BLOCKED** ← the G1 regression |
| `experiments/newpkg/sub/f.py` | BLOCKED |
| `envs/newpkg/foo.py` | BLOCKED |
| `scripts/brand_new.sh` | BLOCKED |
| `.claude/settings.json` | BLOCKED |
| `scripts/check_english.sh` | PERMITTED |
| `scripts/check_test_hygiene.sh` | PERMITTED |
| `experiments/configs/new.json` | PERMITTED |
| `docs/anything.md` | PERMITTED ← **control** |

**Three harness traps I hit personally on 2026-08-03. You will hit them too if you do not guard
against them explicitly:**

1. **Commit the copied guard first.** Left untracked, `scripts/claude_guard.sh` itself dirties
   `git status` and blocks *every* case — including the control. My first run returned BLOCKED for all
   six probes and looked like a triumphant confirmation.
2. **Clean with `git clean -fdq`, not `git checkout -- .`.** The latter does not remove untracked
   files, so one case's leftovers block the next. Three of my rows were contaminated this way and I
   had already read them as confirmations.
3. **Assert the tree is clean before each case** (guard exits 0), and **keep the PERMITTED control
   row**. A control whose expected answer differs from the others is the only thing that
   distinguishes a real confirmation from a broken harness.

Verify the copied guard is byte-identical to the repo's (`sha256`) inside the test, so the test cannot
silently drift onto a stale copy. Skip with an explicit reason if `git` or `bash` is unavailable
(`check_test_hygiene.sh` rejects a reasonless skip).

### 4.4 Required falsification sequence (this is the evidence, not the test passing)

1. Write the test. Run it **against the unpatched guard**. It must **FAIL**, and specifically on the
   `experiments/newpkg/foo.py` row. Paste that failure into the Return Packet — a test that has never
   failed proves nothing.
2. Write the patch. `git apply --check docs/patches/claude_guard_g1.patch`.
3. **🛑 HANDOFF POINT 1 — stop. Ask the human to apply the patch.** You cannot; the path is denied.
4. Re-run. It must **PASS**, with every other row unchanged.

---

## 5. P0.5 — document the liveness role in `experiments/runner.py`

### 5.1 What to write

Nothing on disk says the pin prevents an unbounded hang. The module comment (`runner.py:38-49`) and
the `limit_torch_threads` docstring (`:54-67`) justify it purely on speed. A maintainer reading only
that could drop or relocate the call and silently reintroduce a wedge.

The docstring must state, in this order:

1. **Liveness first.** On the pooled path, a `fork()`ed child that enters an OpenMP parallel region
   with `nthreads > 1` waits forever on team threads `fork()` never duplicated. `run_matrix` collects
   with `as_completed` + `future.result()` and **no timeout**, so this surfaces as a **silent wedge
   with no failure message** — it freezes the suite and the PostToolUse guard, not just the run.
   Reproduced on demand by the P0.3-fix reviewer (`exit=124` at 120 s and 150 s against unpinned code;
   14 passed, 10/10 runs against the committed code).
2. **The ordering constraint that follows.** Anything added to `run_cell` *before* the
   `limit_torch_threads()` call runs in the child's unpinned window. `backend_ready()` already does
   and imports the native backend there — confirmed safe for CityFlow, **unprobed for libsumo and
   moss** (review finding P1; relevant at P7).
3. **Scope, precisely.** The pool exists only under `workers > 1`, so the sequential path never forks
   and the liveness argument does **not** apply to it. There the pin is a performance and determinism
   choice, and it pins the *calling* process as a documented side effect.
4. **Speed demoted to a footnote**, with the retired ratios marked as retired (see §5.3).

⚠️ **I am correcting review finding N1 here; do not "restore" it.** N1 says a maintainer might make
the call "conditional on `workers > 1` … and would silently reintroduce an unbounded hang". That is
imprecise: the fork happens *only* when `workers > 1` (`runner.py:447`), so such a condition would
keep the pin exactly where the hazard is. What genuinely reintroduces the hang is **removing** the
call, **moving it later**, or adding child-side work ahead of it. Write the accurate version.

### 5.2 Mandatory proof that the change is comments-only

Authorisation B permits zero executable changes. Prove it, do not assert it: parse the pre-patch file
(`git show HEAD:experiments/runner.py`) and the patched file, strip every docstring node, and compare
`ast.dump` of both. Comments never enter the AST, so an identical dump proves only comments and
docstrings moved. Paste the result. Run it as a one-off verification script; it does not need to
become a permanent test.

### 5.3 Retired ratios — fix the class, not the sentence

Decisions Log 2026-08-03 banned cross-session ratios: 5.80× and 1.37× divided a baseline measured on a
different day-state. The sweep covered the plan and `docs/patches/README.md` but **missed two files
that use them as live justification**:

- `experiments/runner.py:41-49` — the inline benchmark table (fix here, in P0.5)
- `tests/test_runner_threading.py:5-9, 184` — (fix in P0.7)

Both must present the trustworthy measurement — **199.2 s → 50.2 s, ≈3.97× at workers=6, measured in
one session from a clean shell** — and mark the older table as retired cross-session arithmetic rather
than deleting it (a reader must be able to see the correction happened). Leave
`docs/notes/P0.3_spawn_attempt.md` and `docs/plans/P0.3-fix.md` alone: those are dated session records,
and rewriting history there would be worse than annotating it. Add one dated annotation to each
pointing at the correction.

**🛑 HANDOFF POINT 2** — same as P0.8: write the patch, `git apply --check`, stop, ask the human.

---

## 6. P0.6 — make the timing evidence and the sanity anchors reproducible

Two gaps, same class (a number in the repo whose backing data is not in the repo).

### 6.1 The benchmark config (review finding N5)

The 199.2 s / 50.2 s headline lives in a session scratchpad. Create
`experiments/configs/p0_threading_bench.json` — a **new** config, explicitly permitted by CLAUDE.md
rule 1 and not permission-denied (the deny covers `*.py`). Run it, and record the wall-clock it
actually produces, in one session from a clean shell.

**Scope fence:** record the **pinned** timings only. Do **not** attempt an unpinned re-measurement to
recompute a speedup ratio — unpinned parallelism can hang unboundedly, which is the whole finding, and
you would be deliberately triggering the failure P0.5 is documenting. If the numbers you measure differ
from 199.2 / 50.2, **report yours and do not reconcile them** — different machine state is expected,
and quietly adopting the old numbers would recreate the cross-session error.

### 6.2 The §3.1 sanity anchors (found 2026-08-03)

`docs/p0_baseline_numbers.md` has been cited by plan §3.1 since 2026-07-09 and **has never existed**
(0 commits touching that path in `git log --all`). §7 makes §3.1 the anchor every later phase's
numbers are compared against, and its backing data sits only in `output/experiments/p0_baselines/`,
which is gitignored — a fresh clone cannot reproduce it.

- Commit the raw output under a tracked path (`docs/data/p0_baselines/results.json` + `summary.csv`;
  25 KB total, verified). Do **not** un-ignore `output/`.
- Create `docs/p0_baseline_numbers.md` — the file §3.1 has always pointed at — with the full tables.
- **Load-bearing test:** a test that reads the committed `results.json` and asserts it reproduces the
  six numbers in plan §3.1 (cf_hz1x1: MaxPressure 160.56, Random 307.53, MAPPO 197.91; cf_grid4x4:
  MaxPressure 141.65, Random 207.26, MAPPO 632.95). I verified all six match on 2026-08-03, so this
  test must pass on first run — if it does not, **stop and report**: it means the committed file is not
  the run §3.1 describes, which is a much bigger problem than a missing file.

---

## 7. P0.7 — harden `tests/test_runner_threading.py` (not frozen, ~20 lines)

- **N7 — bound the unbounded wait.** T2 (`test_pool_path_pins_every_worker`, line 149) calls
  `run_matrix`, which has no timeout. If the pin is ever weakened, the suite and the PostToolUse guard
  wedge with no message. Run T2's body under `signal.alarm` or in a `multiprocessing` child with a join
  timeout. T1 already uses `pool.map(..., timeout=60)`; match that generosity — the file runs in ~2.5 s,
  so 60 s cannot flake.
- **N8 — close the asymmetric vacuity guard.** T1 asserts the parent forcing took (line 112). T2
  (line 147) and T3 (line 173) call `torch.set_num_threads(PARENT_THREADS)` and never verify it. One
  torch behaviour change away from two silently vacuous tests. One line each.
- **N3 — correct the over-stated docstring claim** (line 36). "The tests therefore run in a strictly
  harsher configuration than the CLI does" is true **only with respect to torch**: the production
  parent is already 16 OS threads from numpy/OpenBLAS at import, and CPython's fork warning counts
  *Python* threads, so its absence is not evidence of fork safety. Say so.
- **Retired ratios** in the module docstring (lines 5-9) and the comment at line 184 — see §5.3.

Do not weaken any existing assertion to make this easier. Test count must go up, never down.

---

## 8. Scope fence — what NOT to build

- **No behaviour change to `runner.py`.** Comments and docstrings only.
- **No other change to `scripts/claude_guard.sh`** beyond the G1 fix — not the `--tests-only` branch,
  not the rename limitation, not the `FROZEN_EXCEPTIONS` deletion schedule.
- **Do not touch `.claude/settings.json`.** The permission layer is not part of this task.
- **Do not un-ignore `output/`**, and do not commit `plots/`.
- **Do not attempt an unpinned benchmark run** (§6.1).
- **Do not edit `docs/notes/P0.3_spawn_attempt.md` or `docs/plans/P0.3-fix.md`** beyond one dated
  annotation each.
- **Do not rewrite `docs/reviews/P0.3-fix.md`.** N1's imprecision is corrected in the docstring and in
  this brief; the review stays as the record of what the reviewer said.

---

## 9. Definition of Done

- [ ] Four commits on `task/p0-closeout`, in the order P0.8 → P0.5 → P0.6 → P0.7
- [ ] `docs/patches/claude_guard_g1.patch` and `docs/patches/runner_liveness_docs.patch`, both verified
      with `git apply --check`, both with a `docs/patches/README.md` entry matching the existing style
- [ ] Both authorisations quoted verbatim in the Return Packet
- [ ] **G1 test failed before the patch and passed after** — both outputs pasted
- [ ] AST-identity check for `runner.py` pasted, proving comments-only
- [ ] §3.1 reproduction test passes against the committed baseline data
- [ ] `.venv/bin/pytest -q` run in full; real tail pasted; **test count ≥ 249 + your new tests**
      (249 is hearsay from 2026-08-02 — report what you actually observe, and if it is not 249, say so
      rather than reconciling)
- [ ] `bash scripts/check_english.sh <changed paths>` passes
- [ ] `git diff --stat` shows zero modifications to frozen files (the two patches are *files*, not
      modifications — the patched files must show as unmodified in your own diff)
- [ ] Zero new dependencies (stdlib `subprocess`/`ast`/`signal` are fine)
- [ ] Return Packet at `docs/returns/P0-closeout.md` from `docs/returns/TEMPLATE.md`

---

## 10. Return Packet — additions specific to this task

Beyond `docs/returns/TEMPLATE.md`, answer these explicitly:

1. Quote both authorisations. State which files you changed under each, and confirm nothing else.
2. Paste the **failing** G1 test output, then the passing one. Name the human action between them.
3. Paste the AST-identity result for `runner.py`.
4. Report your measured benchmark numbers. State plainly whether they differ from 199.2 / 50.2 and
   **do not reconcile them** if they do.
5. Did the §3.1 reproduction test pass on the first run? If not, stop — do not adjust the test.
6. Which of the three harness traps in §4.3 did you actually hit? Answering "none" is fine and
   informative; answering falsely is not.
7. Open questions for the Master chat.
