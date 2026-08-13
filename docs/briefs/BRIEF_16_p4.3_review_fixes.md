# BRIEF 16 — P4.3 review fixes (F1, F2, F5, F6, F7, F9; plus F8, F10, F11, F12 and two tests)

**Mode:** Claude Code implementation session · **Branch:** stay on `task/p4.3-rtg-calibration`
(not merged; these commits extend it) · **Worktree:** `/home/filip/rltraffic-p43`
**Read first, from disk:** `docs/reviews/P4.3.md` — the full review and my disposition — then this file.

⚠️ **Absolute paths in every command.** ⚠️ **Pin threads on every job** (`DEFERRED` 41).

---

## 1. Where you stand

**PASS-WITH-NOTES, no MAJOR affecting a number, and the landscape is going into the paper.** The
reviewer spied `forward` through real rollouts at three grid points — **1,080 tensors, 0 mismatches**
— and independently reproduced P4's review from a session sharing no code with it. Your Gate A
survived four corruption probes including a mean-preserving swap. **The one way this task could have
been silently void was checked hardest and is clean.**

Two things you did are why this round is short: you **reported a failed check as failed** instead of
retro-declaring, and you **volunteered the less flattering correlation** (+0.5425 over nine points)
with its mechanism rather than the +0.9801 that flatters my withdrawn criterion.

## 2. The frozen constraint

> **NO REPORTED NUMBER MAY CHANGE.** Prove it by regenerating `p4_3_rtg.json` and `p4_3_probe.json`
> and comparing recursively against a **pre-declared** expected-difference set (`BRIEF_12` §7.1's
> discipline). F6's fix will legitimately change one provenance field — declare it before you run.

## 3. Blocking

### F1 — the packet says a failed proof passed

§10.2 states *"all three artifacts regenerate with only the declared differences"*, and `f812f12`'s
subject reads *"DEFERRED 39's third proof **PASSES**"*. `BRIEF_15` §15 ruled that **§10c's FAILED
verdict stands as the result** and attempt two is **a supplementary isolation run, never a replacement
verdict**. Your §0 and §8.1 preserve that; §10.2 and the commit subject do not.

⚠️ **This is not held against you: the relay never reached you.** My ruling landed on `main` at
15:08:16, your final commit is 15:28:48, and nobody sent it. **A ruling that exists only in my tree is
a ruling that did not happen** — logged as mine.

**Required:** correct §10.2; add one line noting that `f812f12`'s subject over-reads and that history
is not rewritten to hide it; and **record that attempt two breached `DEFERRED` 41's pin rule** —
stating a breach and arguing it was harmless is honest, omitting it is not.

### F2 — the functions the numbers flow through have no tests (THIRD sighting of this shape)

`evaluate_point` and `run_probe` have **zero coverage**, while the helpers they call are well tested.
Two survivors of all 48 tests:
- recording `current_rtg()` **before** `act()` — an alignment off-by-one in the trajectory **every
  in-support figure is computed from**;
- `run_probe` writing `from_rewards` into `local_return_from_lanes`, which makes your headline
  *"two independent routes agreed on 100/100, 36,000 decisions, 0 disagreements"* **vacuous**.

**This is the same shape as P4.4's F1 (`iql_targets` tested, `train_iql` call site not) and P4.5's N1.
It is now `PROJECT_PLAN` §7: test the function the numbers flow through, not only the one that
computes them.** Add a test with power over each, **each with its mutation executed and pasted**.

### F5 — the leakage guard is asymmetric on the mechanism that matters

`rule_a_target`'s signature is asserted to be exactly `{returns, quantile}`. **`rule_b_target` has no
such assertion, and adding `heldout_att` to it passes 48/48** — while Rule B is the mechanism A8(a)
names and the one P7 will extend. **Assert Rule B's signature the same way, and mutate to prove it.**

### F6 — the provenance field names a commit that does not exist

`p4_3_rtg.json`'s `measurement_git_commits` contains **`8b647a40…`, which is not an ancestor of HEAD
and is on no ref** — an amended-away commit that resolves today only because it has not been gc'd.
**Confirmed independently by me.** `DEFERRED` 39's whole purpose is a provenance a future reader can
check out. **Required:** re-record the field from reachable commits, **and add a check that every
recorded commit is an ancestor of `HEAD` at write time**, so the mechanism cannot record an
unreachable object again. That check is the actual deliverable; the re-record is its consequence.

### F7 — mutations listed, failures not pasted (second occurrence)

`BRIEF_15` §7 and §10 require *"every mutation executed and its failure pasted"*. §7.1 is a verdict
table with no output. **Re-run and paste the tails.** The reviewer re-executed 17 of 18 and all died,
so this is evidence-of-work, not doubt.

### F9 — P7 is handed two contradictory recipes in one sentence

Packet §15 says Rule B in the target domain is `probe_statistic(target) × ratio` **and** that the
association is `best × (target/source)`. Your own docstring says the first form is inexact for a
measurable fraction of statistics. **P7 implementing the first half breaks the exact identity this
task asserted.** Pick the correct one, state it once, and delete the other.

## 4. Fold in — cheap

**F8** (§1's `git diff --stat` is stale: 7 files/4,907 against the real 8/112,665; §8's "~18,000
leaves" for `p4_gate.json` is really **3,616** — 18,208 is `p4_4_baselines.json`) · **F10** (the
docstring's 7.06 % is not reproducible — the reviewer's 200,000-draw replication gives 9.98 %; either
state the sampling distribution or drop the figure and keep the qualitative claim; same for the
`DEFERRED` 43 count claimed 3 and measured 2) · **F11** (the probe's claimed `evaluate_arm`
cross-check on 100/100 has no artifact, log or test on disk — either emit the evidence or withdraw
the claim) · **F12** (the AI-assistance record attributes the packet's creation to `7358deb`; it is
`7ea97fa`).

**And two tests, because one of them is a rule.** `test_the_report_carries_the_withdrawn_criterions_correlation`
asserts the output **against the function under test** — a wiring check named as a correctness check;
rename it or give it a real reference. And `test_pearson_r_matches_an_independent_recomputation`'s
"independent recomputation" **transcribes the same algorithm into the test body** — it catches a typo,
not a shared conceptual error. **Use `np.corrcoef`.** An independent route must not be the same route
retyped; that is now §7.

## 5. Scope fence

- **No new rollouts, no retraining, no grid extension.** The ten points are final.
- **Do not fix F3 or F4** — `DEFERRED` 44 deliberately, so this round stays verifiable.
- **Do not touch** any committed P4/P4.4/P4.5 artifact, `agent/DTAgent.py`, `offline/dataset.py`,
  `agent/OfflineBaselines.py`, `offline/offline_baselines.py`, or any frozen path.
- **No test deleted or weakened.** Count goes up or stays level (707 passed now).

## 6. Definition of Done

- [ ] F1, F2, F5, F6, F7, F9 addressed; F8, F10, F11, F12 and both tests folded in
- [ ] Every mutation in §3 executed and **its failure pasted** — the item you missed last time
- [ ] Both artifacts regenerated and compared against a **pre-declared** difference set; result pasted
- [ ] Full suite green, tail pasted, **stating whether it was pinned**
- [ ] All three guards exit 0; `git diff --stat` shows no frozen path — and **paste the real one**
- [ ] Return Packet appended to `docs/returns/P4.3.md` as `## 16. Review-fix round` — do not overwrite
- [ ] §6's checkbox stays unticked; it is mine, in the merge commit
