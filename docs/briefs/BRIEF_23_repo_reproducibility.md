# BRIEF 23 — repo reproducibility: packaging, CI, pins

**Mode:** Claude Code · **Branch:** `task/p0.10-reproducibility`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p010 -b task/p0.10-reproducibility main`
**Small task. Runs alongside P5.1 and must not block it.**

⚠️ Absolute paths · pin threads · guards with **no arguments**, counted from **full output** ·
never write "MADT" (C9).

---

## 1. Why now rather than at P10.0

**These three affect whether our numbers are reproducible by anyone else — the claim the paper rests on
hardest.** All three were raised by an external review and **all three were verified against `main` by
the coordinator on 2026-08-16**, so none is hearsay.

**Ordered by severity, and item 1 is worse than it looks.**

## 2. Item 1 — `pip install .` DOES NOT INSTALL `offline/`. Fix this first.

**Measured:** `pyproject.toml`'s `[tool.setuptools.packages.find].include` is
`agent*, algorithms*, envs*, experiments*, metrics*, states*, utils*` — **`offline*` is absent.**

> **So a clean install ships the PLATFORM and not the RESEARCH CODE.** Editable mode works only because
> the directory happens to be on the path. **Anyone reproducing from a clean checkout gets an
> `ImportError` on the half of the repo the paper is about** — `offline/dataset.py`,
> `offline/dt_gate.py`, `offline/offline_baselines.py`, `offline/method_tier_grid.py`,
> `offline/mixture_tiers.py`, `offline/transfer_gate.py`. **Every number in the paper comes from those.**

**Required.** Add `offline*` (and check `offline.policies`, `offline.campaigns` resolve). ⚠️ **Acceptance
is an actual clean-room install, not a diff:** build a wheel or `pip install .` into a **throwaway
venv**, then `python -c "import offline.method_tier_grid"` **from a directory that is not the repo
root** — the cwd is what masks this today. **Show the import FAILING before the fix and passing after.**

## 3. Item 2 — no CI. 923 tests that nothing runs automatically.

**Measured: `.github/workflows` does not exist.** In a project whose §7 rule is *verify by effect, not
by status*, the suite is verified by whoever remembers to run it.

**Required:** one workflow running the **pinned** suite (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` —
`DEFERRED` 41 records the unpinned deadlock) plus the **three guards**, on push and pull request.
⚠️ **`check_test_hygiene.sh` currently exits 1 on 16 pre-existing findings (`DEFERRED` 45) and
`check_english.sh` exits 1 on 5.** **Do not "fix" those to make CI green and do not weaken the guards.**
Record the known baseline and fail only on a **regression against it** — and **prove the regression
check works by introducing a synthetic violation and showing CI go red.** A green badge that cannot go
red is the decoration this project has already caught twice.

⚠️ **`.github/` is not in `FROZEN_PATTERNS`** — verified — so this needs no authorisation. **`scripts/`
is frozen: do not edit a guard.**

## 4. Item 3 — the pins, and the self-contradiction we would have to defend

**Measured:** `requires-python = ">=3.9"` against an actual **3.12.13**; `numpy>=1.24.0`,
`torch>=2.8.0`; no lockfile; no linter or type checker configured.

> **We report differences of 5.68e-14 and treat float reduction order as evidence** — P4.3's 1-ulp
> argument, the 1-vs-16-thread invariance proof, P7.0's float tie-break where `OVL + KS = 1` to twelve
> places — **while the environment producing those numbers is specified by lower bounds only.**
> A referee who notices will ask, and the honest answer today is that we cannot reconstruct it.

**Required:** raise `requires-python` to match reality; pin the runtime deps to the versions actually
used, **measured from the live venv rather than guessed**; and commit a lockfile or a frozen
requirements file **recording the exact environment that produced the merged numbers**. ⚠️ **Do not
upgrade anything.** The goal is to *describe* the environment that produced the results, not to change
it — an upgrade would invalidate every number in `output/`.

## 5. Out of scope

The **licence audit** (that is P2.3, and it is scheduled separately — it can *remove* a contribution
rather than degrade one), the **189 MB of tracked scenarios** (P10.0), any dependency **upgrade**, and
any change under `scripts/`, `.claude/` or any frozen path.

## 6. Definition of Done

- [ ] `offline*` installs; **clean-venv import shown FAILING before and passing after, from outside the
      repo root**
- [ ] CI workflow running the pinned suite + three guards, with the known guard baseline recorded and
      **a synthetic violation shown turning it red**
- [ ] `requires-python`, dependency pins and a lockfile matching the **measured** live environment;
      nothing upgraded
- [ ] Full suite green, tail pasted, pinned state stated; three guards with no arguments, full-output
      counts, each naming its corpus
- [ ] Return Packet at `docs/returns/P0.10.md` with the AI-assistance record
- [ ] A `- [ ] **P0.10**` line added to `PROJECT_PLAN` §6 under Phase P0, left unticked; it is mine, in
      the merge commit
