# BRIEF 25 — P0.10 review fixes (mandated before merge)

**Branch:** `task/p0.10-reproducibility` (continue) · **Review:** `docs/reviews/P0.10.md`,
**PASS-WITH-NOTES**, pinned at `78cd3a9`, **on disk before this round.**
**Read the review in full first.** ⚠️ Absolute paths · pin threads · guards with no arguments, counted
from full output · `git add` **before** running `check_english.sh` — it is vacuous on untracked files.

---

## 0. What this round is and is not

**Your fix is correct and it is proven.** The reviewer's oracle shares nothing with either party — no
`decimal` at all, integer interval arithmetic with directed rounding, Euler's π against your Machin —
and **asserts its enclosure does not straddle a rounding boundary, so the answer is PROVEN correctly
rounded.** 244 arguments, 0 mismatches. **No merged number moves, by three routes**, including a full
regeneration of the P4.3 sidecar (51 leaves, 2 provenance hashes, 21 numeric statistics identical).
**Your reachability claim survived execution: 45/45 on the 43, exactly 322 calls on the 322, landmine
never fired.** And the reviewer ran a check nobody asked for — the pure-Python `_pydecimal` backend —
finding **0 differences**, so the change is robust to the `decimal` backend and not only to libm.

> **This round repairs the PROTECTION, which is weaker than claimed, and one stale measurement. The
> numerics are not in question.**

## 1. R1 — THE LANDMINE IS NAME-BASED AND THE REVIEWER WALKED PAST IT, SUITE GREEN

`monkeypatch.setattr(math, "erfc", …)` rebinds a **module attribute** and **cannot see a reference
bound earlier**. The reviewer reverted the p-value path to platform libm using an early binding —
exactly what `from math import erfc` produces — and got **`997 passed, 3 skipped`, landmine intact.**

**And no value test can help:** on that box **libm reproduces all 322 committed values exactly**, so
the value is blind to the revert **by construction**. Your §15.1 open question is answered **in the
direction you feared: necessary, not sufficient.**

> **REQUIRED: pin the PROPERTY, not a spelling.** The property is *"no platform libm reaches the
> p-value path"*. A **structural** check can enforce it where a runtime patch cannot — e.g. an AST scan
> of `offline/dt_gate.py` for any reference to `math.erfc` / `math.log` / `from math import …` in the
> value path, in the shape of the existing `tests/test_offline_naming_guard.py`. **Acceptance: the
> reviewer's exact early-binding revert must FAIL the suite. Paste that failure.**

## 2. R2 — ONE DELETION REOPENS IT, AND CI'S SECOND libm IS INCIDENTAL

Mutation K fails **exactly one** test; deselecting it leaves **`996 passed, 1 deselected`**. And
`.github/workflows/ci.yml:34` is a bare `runs-on: ubuntu-latest` — **no matrix, no container** — so the
dev-2.43-versus-runner-2.39 coverage **evaporates the day GitHub bumps the image, and nothing fails.**

> **RULED — ADD A SECOND libm TO CI, as a declared matrix.** Two `ubuntu-*` images with **different
> glibc**, or one plus a container, **pinned by version and not by `latest`.** ⚠️ **Without it the
> entire justification for this change is untested going forward**, and the paper's reproducibility
> claim stays a within-machine one for this property too. **Assert the two legs actually differ** —
> print each leg's `platform.libc_ver()` and **fail if they are equal**, or the matrix is decoration.
> **This is the fix that converts a limitation into evidence.**

## 3. Also required

- **R5 — the packet's §1 diffstat is stale and omits the file under review.** It publishes
  **11 files / 2528** measured at `8dd38f8` and claims *"every commit after it touches only this file"*;
  the truth is **13 files / 3101**, with four later commits touching `offline/dt_gate.py` and the test
  file. **Re-measure at HEAD.** ⚠️ **This is the section whose stated purpose is "zero frozen files,
  proved by the diff stat", and it omits the critical-path file the packet exists to disclose** — the
  same family as the measurement-eating errors already in your AI record.
- **R6 — §11's performance figures are warm-memo only.** Cold: **0.319 s** at x=3.52, **0.419 s** at
  x=9, against the reported ~57 µs / ~8 ms. **Report both, labelled**, and note it explains the suite's
  98.8 → 112.9 s.
- **R3 — `_pi_at` does not terminate under a caller's `ROUND_UP` / `ROUND_05UP`** (measured, `SIGALRM`,
  both >8 s against a 0.31 s baseline). **No caller in this repo sets a decimal context, so nothing on
  `main` is affected — fix the exit condition or narrow the test's title, and say which you did.** The
  test claims *"the caller's decimal context cannot change the answer"* while varying only `prec`.
- **Three theatre tests, all in your file:** `:130` (implied by `:129`, cannot fail independently),
  `:225` (`>= 0.0`, near-vacuous), `:224` (`approx(2.0)` at rel 1e-6 where the exact answer is 2.0 to
  the last bit). **Strengthen or delete under §7's conditions.**
- **R7 / R8** — `_pi_at` is a **shared premise** between the code and your "independent" oracle, and
  `DATA_DIR.glob` is **non-recursive** so a future artifact in a `docs/data/` subdirectory escapes the
  `== 322` guard. **Both are cheap.**

## 4. Record, do not fix here

- **R4** — `math.log` remains in the value path (`:309`). The guard-digit argument holds and the
  reviewer accepted it; **record that a libm-free spelling was available and was not taken**, so a
  later tightening knows.
- **R9** — `offline/transfer_gate.py:1318`'s `** 0.5` is committed into `p7_0_gate.json`. **Same class,
  negligible stakes, labelled a reading aid. `DEFERRED` row, not a fix.**
- **F3's exposure list** — five determinism claims a second machine could falsify, led by
  `rtg_calibration.py:799`'s `gate_a_result`, which compares a **live DT + CityFlow re-run** to
  committed records and **has never run on a second machine**. **`DEFERRED` row with the reviewer's
  criterion attached: a test comparing two runs inside one process is immune; one comparing a live
  computation to a committed constant is exposed.**

## 5. Definition of Done

- [ ] R1: the property pinned structurally; **the reviewer's early-binding revert FAILS, failure pasted**
- [ ] R2: CI matrix with two **pinned** images, **asserted to differ by `libc_ver()`**
- [ ] R5 re-measured at HEAD · R6 both figures · R3 fixed or narrowed, said which
- [ ] Three theatre tests strengthened or deleted; R7/R8 closed
- [ ] R4, R9 and F3's list recorded as `DEFERRED` rows
- [ ] **No reported number moves** — prove by regeneration
- [ ] Suite green, tail pasted, pinned; three guards, no arguments, full-output counts, corpus named
- [ ] Packet updated · §6's checkbox unticked; it is mine, in the merge commit
