# BRIEF #2b — Thread pinning in `experiments/runner.py` (FROZEN-FILE TASK)
**Task ID:** P0.3-fix · **Contracts:** v1.1 · **Branch:** `task/p0.3-thread-pinning`
**Issued:** 2026-07-27, Master Coordination Chat · **Merge gate:** independent `/review` must pass

> **This task modifies a FROZEN file.** `experiments/` is in the frozen set and `claude_guard.sh` will
> block the edit. That block is correct and must not be disabled. The authorisation to edit exactly
> one file is granted below, by the Master chat, for this task only.

---

## AUTHORISATION (quote this in the Return Packet)

> The Master chat authorises modification of **`experiments/runner.py` only**, on branch
> `task/p0.3-thread-pinning`, for the single purpose of limiting per-worker torch thread counts.
> No other frozen file may be touched. The guard's frozen list itself must **not** be edited — work
> around the block by whatever mechanism the repo already provides for authorised exceptions
> (see `FROZEN_EXCEPTIONS` in `scripts/claude_guard.sh`), and if no such mechanism fits, stop and
> report rather than weakening the guard.

## WHY (measured, not assumed)

Benchmark 2026-07-27, reduced `p0_baselines` (6 cells, serialized runs), documented in
`docs/notes/P0.3_spawn_attempt.md`:

| workers | unpinned | pinned (`OMP=MKL=OPENBLAS=1`) |
|---|---|---|
| 1 | 339.7 s (1.00×) | **247.7 s (1.37×)** |
| 3 | 1165.3 s (**0.29×**) | 76.3 s (4.45×) |
| 6 | ≥1200 s (0.28×) | **58.6 s (5.80×)** |

Two facts drive this task:
1. **Unpinned `workers>1` is a regression** — 3.4× *slower* than sequential. ~48–96 torch OMP threads
   contend on 16 cores. This, not any deadlock, was the July "frozen terminal, 10% CPU" symptom; the
   spawn patch fixed the wrong problem and was correctly reverted.
2. **Pinning helps even at `workers=1` (1.37×)** — torch's default pool hurts these small MLPs. So this
   is a win regardless of whether parallelism is ever used.

Payoff: the P5 online-baseline budget drops from 15–60 h sequential to roughly 3–10 h.

## SCOPE

**In scope:** limit torch's thread count per worker process; a test proving it; a re-measured benchmark
line confirming the speedup holds after the change.

**Out of scope:** the spawn patch (`mp_context`) — refuted, do not add it. Any other change to
`runner.py`. Any other frozen file. `offline/` (untouched — `collect.py` uses no multiprocessing and
parallelises via independent shell processes).

---

## THE CHANGE

In `experiments/runner.py`, ensure each worker process limits its torch threads to 1. `run_cell` is
module-level (verified) and is the function submitted to the pool, so it is the natural place — but
choose deliberately between:

- **(a)** `torch.set_num_threads(1)` at the top of `run_cell`, guarded so it only applies in worker
  processes (or applied unconditionally if the 1.37× sequential gain is wanted everywhere — the
  measurement says it is);
- **(b)** a `ProcessPoolExecutor(initializer=...)` that sets it once per worker;
- **(c)** setting `OMP_NUM_THREADS`/`MKL_NUM_THREADS` env vars before pool creation.

Note that (c) must happen **before torch is imported** in the child to take effect, and torch enters
lazily via `build_agent` inside `run_cell` — so (c) is viable here, but state why you chose what you
chose. Import torch lazily inside the function if needed; **do not add a module-level `import torch`**
to `runner.py`, since its absence at fork time is a verified property of the current design.

Make the thread count a constant or a parameter rather than a bare literal, and document the
measurement that justifies it.

## TESTS

Add to an appropriate existing test file, or create `tests/test_runner_threading.py`:

1. **The property, directly:** run a cell (or a minimal callable) through the pool path and assert
   `torch.get_num_threads() == 1` **inside the worker**. Return it from the worker and assert in the
   parent — do not assert in the parent's own process, which proves nothing about the children.
2. Sequential path (`workers=1`) still produces correct results — reuse the smoke config if a fast
   fixture exists, otherwise assert on a minimal synthetic cell.
3. Whole suite stays green (was 180 tests).

## VERIFICATION (paste raw output in the Return Packet)

1. `pytest tests/ -q` — full suite.
2. **Re-measure**: the same reduced-`p0_baselines` benchmark at `workers=1` and `workers=6`, *without*
   any external `OMP_NUM_THREADS` env var, from a clean shell. The point is proving the fix works
   **from inside the code**, not from the shell environment that produced the original numbers.
   Expected: roughly 247 s and 59 s. Report what you actually get.
3. `bash scripts/claude_guard.sh --frozen-only ; echo "exit=$?"` — and explain the result honestly. If
   the guard flags `experiments/runner.py`, say so and show how the authorised exception was recorded;
   do **not** silence it by editing the frozen list.
4. `git diff --stat main...HEAD` — must show `experiments/runner.py` plus test files and nothing else.

## DEFINITION OF DONE
- [ ] Change applied to `experiments/runner.py` only, with the mechanism choice justified
- [ ] Worker-side thread count asserted by a test that reads it **in the child**
- [ ] Benchmark re-run from a clean shell, raw numbers pasted
- [ ] Guard result explained honestly; frozen list unmodified
- [ ] Committed on `task/p0.3-thread-pinning`; `docs/returns/P0.3-fix.md` written
- [ ] `docs/notes/P0.3_spawn_attempt.md` gets a closing line pointing at the fix commit
