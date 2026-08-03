# Plan — P0 close-out (`task/p0-closeout`)

**Brief:** `docs/briefs/BRIEF_03_p0_closeout.md` (single source of truth for P0.5/P0.6/P0.7/P0.8).
**Approved:** 2026-08-03 by the Master chat, with five corrections folded in below.
**Order (binding):** P0.8 → P0.5 → P0.6 → P0.7. One path-scoped commit per task, so a later one turning
messy does not block an earlier one from merging.

---

## Corrections to the brief, applied (Master chat, 2026-08-03)

1. **Frozen-file landing — the brief's DoD was backwards.** The DoD said the patched files "must show as
   unmodified in your own diff." That would ship an applied-but-uncommitted patch: `test_claude_guard.py`
   exercises the *real* guard, so the working tree is green while `main` goes red the instant the branch
   merges, with the fix living on one laptop only. **Corrected against precedent** (P0.3-fix's `runner.py`
   delta is committed on `main` at `64800fb` — verified: `git show HEAD:experiments/runner.py` contains
   `limit_torch_threads`). **Therefore: I `git add` and commit each patched frozen file myself, in the
   same commit as the work depending on it, with the authorisation quoted verbatim in the commit
   message.** An authorised delta that isn't committed is worse than an unauthorised one — it is invisible.
   Operational note: between the human applying a patch and my commit, every Bash call fails with
   `BLOCKED: frozen files were modified`. That is the guard working. Do **not** revert, stash, or weaken
   it — commit and it clears.
2. **Benchmark in-session — yes.** CLAUDE.md §5's tmux rule targets hour-scale work (corpus collection,
   MAPPO training). ~250 s is fine. Both runs pinned, one session, clean shell — that is exactly what
   makes ≈3.97× quotable and 5.80× not.
3. **Line budget — one branch, do not split.** The logic diff is a two-line regex; the bulk is the test
   commissioned by name (`test_claude_guard.py`). One path-scoped commit per task preserves the option to
   merge P0.8/P0.5/P0.6 and leave P0.7 if it turns messy.
4. **N7 via forked `multiprocessing` child + `terminate()` — approved.** `signal.alarm` does not bound it:
   the alarm unwinds through `ProcessPoolExecutor.__exit__` into `shutdown(wait=True)`, which joins the
   wedged workers and re-hangs.
5. **Exact `==` on the anchor double-compute — keep exact.** Both sides are deterministic functions of an
   immutable committed file. A failure there is a real signal, not flake.

---

## Facts verified during exploration (commands run 2026-08-03)

- CityFlow is importable; `backend_ready('cityflow', {})` → `(True, '')`. The benchmark runs, cells do not skip.
- The six §3.1 anchors reproduce exactly from `output/experiments/p0_baselines/results.json`:
  `aggregated[env][pol]['average_travel_time']['mean']` rounds to
  `cf_hz1x1: 160.56 / 307.53 / 197.91`, `cf_grid4x4: 141.65 / 207.26 / 632.95`.
  `np.mean` over the 3 per-seed `cells[*].policies[pol].metrics['average_travel_time']` equals the stored
  aggregated mean **bitwise** for all six → exact `==` double-compute holds.
- Data size: `results.json` 24 731 B + `summary.csv` 820 B ≈ 25 KB.
- Deny-list: `Edit(scripts/**)`, `Edit(experiments/*.py)`, `Edit(experiments/**/*.py)`. A `.json` under
  `experiments/configs/` is not denied → the new bench config is writable.
- Guard regexes match the brief §3 quote verbatim (lines 36/47). Findings confirmed in
  `docs/reviews/P0.3-fix.md`: N1 (98), N3 (117), N5 (126), N7 (136), N8 (138). The trustworthy
  199.2 s / 50.2 s figure is already recorded at `docs/notes/P0.3_spawn_attempt.md` line 121; the retired
  cross-session table is lines 84-86.

---

## Patch-production method (both patches)

Produce each patch in a scratch dir mirroring the path, diffed against `HEAD` content so headers are
`a/<path> b/<path>` and offsets match the clean working tree:

```
mkdir -p /tmp/pX/<dir>; git show HEAD:<path> > /tmp/pX/<path-mirror>
# edit the /tmp copy (not frozen); git -C /tmp/pX diff  ->  docs/patches/<name>.patch
git apply --check docs/patches/<name>.patch     # paste OK
```

Never touch the real frozen file to *build* the patch; never use heredoc/`sed -i`/`tee`/`cp`/`patch`
against a denied path. The human applies the patch; I then `git add` the (now-modified) frozen file.

---

## Execution

### Commit 0 — this plan file.

### P0.8 — guard defect G1 (commit 1) — tests first
1. Write `tests/test_claude_guard.py` (NEW). Scratch git repo in `tmp_path`; baseline mirrors the real
   repo's tracked dirs so `git status` collapse points match D3: tracked `scripts/claude_guard.sh`
   (byte-copy of the real guard, **sha256-verified inside the test**), plus tracked placeholders
   `experiments/runner.py`, `experiments/configs/existing.json`, `envs/pkg.py`, `docs/keep.md`.
   Parametrized 11-row truth table; each row: assert clean → dirty the path → run the **copied** guard
   `--frozen-only` with `cwd=scratch` → assert `returncode` 2 (BLOCKED) / 0 (PERMITTED) →
   `git checkout -- . && git clean -fdq` → re-assert clean. Keep the PERMITTED control (`docs/anything.md`).
   Reasoned skip if `git`/`bash` absent.
2. Run against the unpatched guard → MUST FAIL on `experiments/newpkg/foo.py` (+ `newpkg/sub/f.py`). Paste.
3. Write `docs/patches/claude_guard_g1.patch`: `experiments/.*\.py$` → `experiments/)` prefix;
   `FROZEN_EXCEPTIONS` → `^(scripts/(check_english|check_test_hygiene)\.sh$|experiments/configs/)` (`$` stays
   bound to the `scripts/` alternative only); dated comment above `FROZEN_PATTERNS` explaining the collapse.
   `git apply --check` → paste OK. Add the `docs/patches/README.md` entry.
4. 🛑 HANDOFF 1 — stop; ask the human to apply `claude_guard_g1.patch`.
5. (resume) Re-run the test → MUST PASS, every row. Paste.
6. Commit 1, path-scoped: `scripts/claude_guard.sh` (now patched), `tests/test_claude_guard.py`,
   `docs/patches/claude_guard_g1.patch`, `docs/patches/README.md`. Authorisation A quoted verbatim in the message.

### P0.5 — liveness docs in `runner.py` (commit 2)
1. Write `docs/patches/runner_liveness_docs.patch` (comments/docstrings only). Rewrite the
   `limit_torch_threads` docstring in order: (1) liveness first, (2) ordering constraint, (3) scope,
   (4) speed as footnote — with the **accurate** N1 (removing / moving-later / child-work-ahead reintroduces
   the hang; making it conditional on `workers>1` does not). Retire the cross-session ratio table in the
   comment block (keep it visible, mark retired, add 199.2 s → 50.2 s ≈ 3.97× at workers=6).
2. AST-identity proof (§5.2): parse `git show HEAD:experiments/runner.py` and the patched temp copy, strip
   every docstring node, compare `ast.dump` → "IDENTICAL". One-off python heredoc reading `/tmp` files.
3. Add the README entry. Add one dated annotation each to `docs/notes/P0.3_spawn_attempt.md` and
   `docs/plans/P0.3-fix.md` (do not rewrite their tables).
4. 🛑 HANDOFF 2 — stop; ask the human to apply `runner_liveness_docs.patch`.
5. (resume) Re-run the AST-identity check against the now-patched real `runner.py`; paste.
6. Commit 2, path-scoped: `experiments/runner.py` (now patched), `docs/patches/runner_liveness_docs.patch`,
   `docs/patches/README.md`, `docs/notes/P0.3_spawn_attempt.md`, `docs/plans/P0.3-fix.md`. Authorisation B
   quoted verbatim in the message.

### P0.6 — reproducible timing + §3.1 anchors (commit 3)
1. Create `experiments/configs/p0_threading_bench.json` — reduced `p0_baselines`: 2 envs × 3 seeds = 6
   cells, `train_episodes=10, max_steps=360, delta_time=10`, mappo agent, compare with random+max_pressure.
2. Run pinned twice (no unpinned run): `--workers 1 --no-plot` and `--workers 6 --no-plot`, wall-clock via
   a `time.perf_counter()` subprocess wrapper. Record my numbers; do not reconcile with 199.2/50.2.
3. `cp` existing `output/experiments/p0_baselines/{results.json,summary.csv}` → `docs/data/p0_baselines/`.
   Do not un-ignore `output/`, do not regenerate.
4. Create `docs/p0_baseline_numbers.md` with the full tables from `summary.csv`.
5. Write `tests/test_p0_baseline_anchors.py` (NEW): read committed `docs/data/p0_baselines/results.json`;
   per (env, pol) double-compute `np.mean` over the 3 per-seed cell values, assert exact `==` to the stored
   aggregated mean, assert `round(stored, 2)` equals the §3.1 value. Must pass on first run; if not, STOP.
   Prove strength by mutation (tamper a `/tmp` copy → assert failure).
6. Commit 3, path-scoped: config, `docs/data/p0_baselines/`, `docs/p0_baseline_numbers.md`,
   `tests/test_p0_baseline_anchors.py`.

### P0.7 — harden `tests/test_runner_threading.py` (commit 4)
- N7: wrap T2's `run_matrix` in a forked `multiprocessing` child; `queue.get(timeout=60)` + `join`,
  `terminate()`+`pytest.fail` on timeout. Keep all existing assertions.
- N8: `assert torch.get_num_threads() == PARENT_THREADS, "parent forcing did not take"` after the
  `set_num_threads` in T2 and T3 — one line each.
- N3: correct the "strictly harsher than the CLI" sentence (true only w.r.t. torch).
- Retired ratios: module docstring lines 5-9 and the comment at line 184 → 199.2 s → 50.2 s ≈ 3.97×,
  old ratios marked retired.
- No assertion weakened, no test function removed. Run the file → 4 green; paste.
- Commit 4, path-scoped: `tests/test_runner_threading.py`.

### Finalize
- `.venv/bin/pytest -q` full run; paste tail + count (report actual vs the hearsay 249).
- `bash scripts/check_english.sh <changed paths>`; paste.
- `git diff --stat main...HEAD`.
- Write `docs/returns/P0-closeout.md` with the seven brief-specific answers.

---

## Open questions carried to the Return Packet
The two I raised (frozen-file landing, in-session bench) were answered in the approval; the Return Packet
carries only the brief's seven §10 items plus any new question that surfaces during coding.
