# RETURN PACKET — P0-closeout (guard G1, liveness docs, reproducibility, test hardening)
**Date:** 2026-08-04 · **Mode:** Claude Code · **Contracts:** v1.1 · **Brief:** `docs/briefs/BRIEF_03_p0_closeout.md`

**Status:** DONE — four tasks committed in order P0.8 → P0.5 → P0.6 → P0.7, both patches applied by the
human and committed on the branch, full suite green. One DoD checkbox is an honest N (a pre-existing
`check_english.sh` false positive on a comment I did not author — see the checklist and limitations).

**Branch + diff stat**
```
task/p0-closeout
$ git diff --stat main...HEAD
 docs/data/p0_baselines/results.json         | 998 ++++++++++++++++++++++++++++
 docs/data/p0_baselines/summary.csv          |   7 +
 docs/notes/P0.3_spawn_attempt.md            |   7 +
 docs/p0_baseline_numbers.md                 |  49 ++
 docs/patches/README.md                      |  91 +++
 docs/patches/claude_guard_g1.patch          |  53 ++
 docs/patches/runner_liveness_docs.patch     |  96 +++
 docs/plans/P0.3-fix.md                      |   6 +
 docs/plans/p0-closeout.md                   | 146 ++++
 experiments/configs/p0_threading_bench.json |  21 +
 experiments/runner.py                       |  77 ++-
 scripts/claude_guard.sh                     |  30 +-
 tests/test_claude_guard.py                  | 155 +++++
 tests/test_p0_baseline_anchors.py           | 104 +++
 tests/test_runner_threading.py              |  93 ++-
 15 files changed, 1890 insertions(+), 43 deletions(-)
```
Frozen files touched: **exactly two** — `scripts/claude_guard.sh` (Auth A) and `experiments/runner.py`
(Auth B). No other file under the frozen set. `experiments/configs/p0_threading_bench.json` is not frozen
(configs are writable).

**Commits (5: plan + 4 tasks)**
```
e67197b test(threading): harden test_runner_threading -- bound the wait, close vacuity, fix stale docs (P0.7)
e0eabfc data(P0.6): make the §3.1 anchors and the timing benchmark reproducible from the repo
aa46adc docs(runner): document the thread pin as a liveness fix, not a speedup (P0.5)
d9f1b5e fix(guard): close G1 -- new .py in a new experiments/ subdir escaped the frozen check (P0.8)
3624e9b plan(P0-closeout): four-task close-out, order P0.8->P0.5->P0.6->P0.7
```

**Files produced**
- `docs/plans/p0-closeout.md` — the plan (commit 0).
- `tests/test_claude_guard.py` — 13-row truth table over the real guard (P0.8).
- `scripts/claude_guard.sh` — G1 fix, applied by human, committed (P0.8, Auth A).
- `docs/patches/claude_guard_g1.patch` — the G1 patch (audit record).
- `experiments/runner.py` — liveness docstring + retired ratios, applied by human, committed (P0.5, Auth B).
- `docs/patches/runner_liveness_docs.patch` — the P0.5 patch (audit record).
- `docs/patches/README.md` — two new entries.
- `docs/notes/P0.3_spawn_attempt.md`, `docs/plans/P0.3-fix.md` — one dated retirement annotation each.
- `experiments/configs/p0_threading_bench.json` — the timing benchmark config (P0.6).
- `docs/data/p0_baselines/{results.json,summary.csv}` — the 2026-07-09 anchor run, now tracked.
- `docs/p0_baseline_numbers.md` — the §3.1 file that never existed.
- `tests/test_p0_baseline_anchors.py` — anchor reproduction test (P0.6).
- `tests/test_runner_threading.py` — hardened (P0.7).

**Tests — REAL OUTPUT**
```
$ .venv/bin/pytest -q            # full suite
270 passed, 12 warnings in 34.46s
```
270 = 249 (pre-existing, the §9 hearsay confirmed exactly) + 21 new (13 guard + 8 anchors); threading
stayed at 4 functions (hardened, not added to).
```
$ .venv/bin/pytest tests/test_claude_guard.py -q        # after the G1 patch
13 passed in 0.44s
$ .venv/bin/pytest tests/test_p0_baseline_anchors.py -q
8 passed in 0.13s
$ .venv/bin/pytest tests/test_runner_threading.py -q
4 passed, 3 warnings in 2.28s
```

**Self-review checklist**
- Contracts in `docs/CONTRACTS.md` honored, including the non-standard env API? **Y** (used `run.py` /
  `run_matrix` as-is; no env-API code written).
- All tests actually executed and green? **Y** — 270 passed, pasted above.
- Zero modifications to frozen files? **N, by design** — two authorised deltas (`claude_guard.sh` Auth A,
  `runner.py` Auth B), committed here per the coordinator's correction #1 (the brief's "must show
  unmodified in your own diff" was reversed). No other frozen file touched.
- Zero new dependencies? **Y** (stdlib `subprocess`/`ast`/`signal`-not-needed/`multiprocessing`/`queue`/
  `traceback`; numpy already present).
- Every number in this packet produced by a command run this session? **Y**.
- `bash scripts/check_english.sh <changed paths>` passes? **N** — passes on all authored content; the only
  two hits are one identical **pre-existing** `o-acute` false positive (guard line 47, byte-identical to main
  line 39, plus its echo as unchanged context in the patch at line 32). It is the documented false
  positive CLAUDE.md and the flagged line itself describe. I did not author or modify it, and did not edit
  the frozen `check_english.sh`. Every line I added to the guard is pure ASCII (verified).

---

## Brief §10 — task-specific answers

**1. Both authorisations, verbatim, and what changed under each.**

> AUTHORISATION A — 2026-08-03, Master chat, for P0.8. `scripts/claude_guard.sh` may be modified for the
> single purpose of fixing defect G1: making the `experiments/` clause of `FROZEN_PATTERNS` robust to
> `git status --porcelain` collapsing untracked directories, and adding the corresponding
> `experiments/configs/` carve-out to `FROZEN_EXCEPTIONS`. Delivered as a patch under `docs/patches/`,
> applied by a human. No other change to that file, and no change to any other file under `scripts/`, is
> authorised. Spent when P0.8 merges.

Under Auth A I changed **only** `scripts/claude_guard.sh`: the `experiments/` clause of `FROZEN_PATTERNS`
became a prefix; `FROZEN_EXCEPTIONS` gained `experiments/configs/[^/]*\.json$` (tightened from the brief's
bare `experiments/configs/` per the Master-chat review below); plus a dated comment. No other file under
`scripts/`. Confirmed by the diff stat.

> AUTHORISATION B — 2026-08-03, Master chat, for P0.5. `experiments/runner.py` may be modified for the
> single purpose of documenting the liveness role of `limit_torch_threads()`. Comments and docstrings only
> — zero executable-statement changes, proven mechanically (§5.2, the AST check). Delivered as a patch
> under `docs/patches/`, applied by a human. This does not authorise moving the call site, changing
> `CELL_TORCH_THREADS`, or any behaviour change whatsoever. Spent when P0.5 merges. The P0.3-fix
> authorisation ("limiting per-worker torch thread counts") is spent and does not cover this.

Under Auth B I changed **only** `experiments/runner.py`, comments and docstrings only (proven below). No
call-site move, no `CELL_TORCH_THREADS` change, no behaviour change.

**2. G1 test: failing before the patch, passing after; the human action between.**
```
# BEFORE the patch (unpatched real guard):
FAILED tests/test_claude_guard.py::test_frozen_guard_truth_table[new_pkg_py_G1]
        expected BLOCKED (exit 2) for 'experiments/newpkg/foo.py', got exit 0
FAILED tests/test_claude_guard.py::test_frozen_guard_truth_table[new_pkg_sub_py_G1]
        expected BLOCKED (exit 2) for 'experiments/newpkg/sub/f.py', got exit 0
FAILED tests/test_claude_guard.py::test_frozen_guard_truth_table[config_subdir_fail_closed]
        expected BLOCKED (exit 2) for 'experiments/configs/sub/new.json', got exit 0
3 failed, 10 passed in 0.45s
```
**Human action:** applied `docs/patches/claude_guard_g1.patch` (`git apply`; `bash -n` OK) — I cannot; the
path is permission-denied.
```
# AFTER the patch (patched real guard):
13 passed in 0.44s
```

**3. AST-identity result for `runner.py` (against the real, on-disk patched file).**
```
HEAD vs working-tree runner.py, docstrings stripped -> AST IDENTICAL: True (61586 B == 61586 B)
mutation control (int(n_threads) -> int(n_threads)+1): AST equal? False
```
Comments never enter the AST; the only docstring changed is `limit_torch_threads`'s. The mutation control
proves the check detects executable changes. Corroborated: `git apply --stat` touched only `runner.py`, and
a grep for every executable token (`def`/`import torch`/`set_num_threads`/`return int`/`CELL_TORCH_THREADS`)
among the diff's `+/-` lines returned NONE.

**4. Measured benchmark numbers (do not reconcile).**
`experiments/configs/p0_threading_bench.json`, this session, from a clean shell, **both runs pinned**:
- `--workers 1`: **182.9 s** (6/6 ok)
- `--workers 6`: **48.3 s** (6/6 ok)
- ratio ≈ **3.79×**

These **differ** from the recorded 199.2 s → 50.2 s (≈3.97×). Per §6.1 I report mine and **do not
reconcile** — different machine state is expected. No unpinned run was attempted (that is the hang the
finding is about). The docstrings/comments in P0.5/P0.7 still cite the recorded 199.2/50.2 as instructed;
these measured numbers live here only.

**5. Did the §3.1 reproduction test pass on the first run?** **Yes** — `8 passed` first run, no adjustment.
The six anchors reproduce exactly; the double-compute (`np.mean` over per-seed cells) equals the stored
aggregate bitwise for all six, and each rounds to the plan §3.1 value.

**6. Which of the three §4.3 harness traps did I hit?** **None**, and honestly so — I designed around all
three rather than tripping them: fresh scratch repo **per case** (so trap 2, `git checkout` leaving
untracked files, cannot occur); the copied guard is committed in the baseline and sha256-verified, and the
fixture asserts a clean tree + guard-exit-0 before each case (trap 1); the PERMITTED `docs_control` row is
kept (trap 3). I did apply trap 2's `git clean -fdq` discipline in the throwaway `/tmp` end-to-end preview,
which is a different, non-committed harness.

**7. Open questions for the Master chat.** See below.

---

**Deviations from the brief** (each justified)
- **Frozen deltas committed, not left applied-but-uncommitted** — per the coordinator's correction #1
  (the brief's DoD "the patched files must show as unmodified in your own diff" was reversed; precedent is
  P0.3-fix's `runner.py` delta on `main`). Both authorised files are committed with the authorisation
  quoted verbatim in the message.
- **G1 exception tightened to `experiments/configs/[^/]*\.json$`** (JSON-only, top-level), not the brief
  §4.2 bare `experiments/configs/`. The Master-chat review measured that a bare prefix exempts `.py` too,
  reopening G1's exact hole one directory over via a heredoc. Verified safe: configs holds 5 top-level
  `.json`, 0 non-JSON, 0 subdirs, and `p0_threading_bench.json` is top-level JSON.
- **Two extra truth-table rows** (`config_py_not_exempt`, `config_subdir_fail_closed`) added per the same
  review; nested configs fail closed by deliberate choice, documented in the guard and README.
- **Corrected a factual docstring claim in `tests/test_runner_threading.py`** ("four" fork warnings → three)
  as a follow-on to N7, which moved T2's pool forks into a child. This is not editing a test to make it
  pass; it keeps a now-false statement true. Disclosed in full.

**Conflicts found between brief and repo** (repo/coordinator wins)
- Brief DoD vs correction #1 (frozen-file landing): implemented per the correction and precedent.
- Brief §4.2 bare-prefix exception vs the measured `.py` hole: implemented the tightened JSON-only version.
- Brief §4.4 "the two G1 rows remain the ones that flip" vs measurement: **three** rows flip against the
  unpatched guard (the two G1 rows **and** `config_subdir_fail_closed`, which is permitted-by-accident
  today because the old extension anchor never matched the collapsed `experiments/configs/sub/`). The third
  flip is the fail-closed improvement the review designed, so it is correct; flagged for BRIEF_03 §11.

**Limitations of what I shipped**
- `check_english.sh` reports two hits, both the pre-existing `o-acute` false positive on a comment I did not
  author; I did not edit the frozen script. The false positive is documented and scheduled for removal.
- N8's assertions are mirrored from T1's proven-working one; I could not mutation-prove them because
  simulating "torch stops honouring `set_num_threads`" is not feasible in-session. N7's bound **is**
  mutation-proven (stub `run_matrix` to `sleep(1000)`; `_run_matrix_bounded(timeout=3)` failed after 3.0 s,
  not a hang).
- The two `docs/patches/*.patch` files are now redundant with the committed frozen deltas — kept as the
  audit record, matching the `runner_thread_pinning.patch` precedent.

**Open questions / risks for the Master chat**
1. **3-vs-2 flip** (above): fold `config_subdir_fail_closed` into BRIEF_03 §11 as the third flipping row.
2. **`check_english.sh` `o-acute` false positive** now blocks a clean `check_english` on any change to
   `scripts/claude_guard.sh`. Worth scheduling the documented fix (a patch to `check_english.sh`), since
   the guard is otherwise frozen and every future guard edit will re-trip it.
3. The two `FROZEN_EXCEPTIONS` deletion clauses (`check_english`, `check_test_hygiene`) are untouched per
   §8; the new `experiments/configs/[^/]*\.json$` carve-out has no deletion schedule (it is permanent
   intent, not a temporary tolerance) — confirm that reading.

**What the next task will assume about this one** (phase-boundary check)
- The guard blocks a new `.py` anywhere under `experiments/` (including new subdirs) and permits only
  top-level `experiments/configs/*.json`; `tests/test_claude_guard.py` is the regression guard and lives on
  the branch with the patched guard, so both land on `main` together at merge.
- `experiments/runner.py`'s `limit_torch_threads` docstring is the authority on why the pin is a liveness
  fix; `CELL_TORCH_THREADS == 1` unchanged.
- The §3.1 anchors are reproducible from `docs/data/p0_baselines/results.json` via
  `tests/test_p0_baseline_anchors.py`; `docs/p0_baseline_numbers.md` is the file plan §3.1 points at.
- `experiments/configs/p0_threading_bench.json` is the committed timing benchmark (reduced `p0_baselines`).
