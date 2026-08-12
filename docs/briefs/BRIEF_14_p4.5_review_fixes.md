# BRIEF 14 — P4.5 review fixes (N1, N3, N5, N8, N9, N11; plus N4, N6, N7, N10)

**Mode:** Claude Code implementation session · **Branch:** stay on `task/p4.5-bc-seed-selection`
(not merged; these commits extend it) · **Worktree:** `/home/filip/rltraffic-p45`
**Read first, from disk:** `docs/reviews/P4.5.md` — the full review and my disposition — then this file.

---

## 1. Where you stand

**P4.5 is PASS-WITH-NOTES and the result is going into the paper.** The reviewer rebuilt all 20
selections from the raw `.npz` corpus by a route sharing no code with yours, verified **252,000
training rows from the tensors**, retrained **2 of 20 checkpoints** to byte-identical canonical
digests, re-rolled 14 episodes exactly, and compared **18,208 leaves** of a regenerated
`p4_4_baselines.json` — one differing path, the expected `.runtime.git_commit`. **Nothing it found
moves a number.**

Two things you did are the reason this round is short: you **disclosed a surviving mutant instead of
repairing it quietly** (the `match="rounding"` shadowing), and you **caught two false claims in your
own packet before submitting**. The reviewer confirmed both disclosures exact.

## 2. The frozen constraint

> **NO REPORTED NUMBER MAY CHANGE.**

Prove it the way `BRIEF_12` §7.1 established: regenerate `docs/data/p4_5_baselines.json` and
`p4_5_selection.json` and compare **recursively over every path and type** against the committed
copies, with the expected-difference set **declared in a literal before the comparison runs**.
`.runtime.git_commit` is the only expected change. **If anything else differs, stop and report.**

## 3. Blocking items

### N1 — a validator that takes its reference from the data it validates (the one that matters)

`offline/offline_baselines.py:1038`, verified in place by me:

```python
held_out = {int(draw) for draw in payload["held_out_draws"]}
```

The docstring claims invariant 1 is *"no held-out draw enters training"*. **It cannot deliver that**,
because the pool comes from the payload under test. The reviewer emptied the field, planted a real
leak (`flow_draw = 1042`) — **suite green, 81 passed.**

⚠️ **Three instances in one function, not one.** `declared_count` is read from the block being checked
(so "every arm declares 19" passes), and `reference_seeds` from the first arm's own record (so a
uniformly wrong seed set passes). There is also **no `rows == T × len(streams)` check**.

**Required.**
1. **Every reference comes from the DECLARATION, not the argument** — `HELD_OUT_DRAWS`,
   `TRAINING_SEEDS`, `SELECTION_ARMS` — and the payload's corresponding fields are **cross-checked
   against those constants**, so a disagreement raises rather than being adopted.
2. Add the missing `rows == T × len(streams)` invariant.
3. **Tests: three mutations, each executed and pasted** — empty `held_out_draws` **with a planted
   leak** (the reviewer's exact probe), all arms declaring 19, and a uniformly wrong seed set. Each
   must now fail.
4. **Fix the docstring to state what is actually guaranteed.** It is the sentence that made this a
   finding rather than a gap.

### N3 — the CLI runners have no tests, and the RNG property is unprotected

`_run_train_selection`, `_run_gate_selection` and `_run_report_selection` have **zero coverage**.
Mutating the subset RNG to `default_rng(101)` — one subset shared by all five training seeds —
**survives** (`643 passed`). That property is what makes the reported CIs cover subset variance, which
is a claim the packet makes about what the numbers mean.

**Required:** a test that fails when the subset is not drawn from the **training seed's** RNG, with
that mutation executed and pasted. The other two runners need at least one test each with real power.

### N5 — a contradiction on disk

An F6b docstring says the two deltas differ by **4.4e-5**; the true value is **2.4353e-05**, which is
what your own §6.3 says. Correct the docstring.

### N8 — the AI-assistance record is factually wrong, in its first outing

It lists `tests/test_offline_baselines.py` and the pre-existing `offline/offline_baselines.py` as
"AI-edited from a **human** original". **Both were created in `933eb54` by the P4.4 Claude Code
session**, committed under the human's git identity — which is precisely why "who committed it" cannot
be used to infer authorship here.

**Required:** correct it, and **state the method you used to determine authorship** (e.g. `git log
--diff-filter=A` plus the commit's own message), so the next packet does not repeat the inference.
**This section is the one a publisher reads. Being wrong in it is worse than omitting it.**

### N9 — the mutations are listed but their failures are not pasted

`BRIEF_13` §5 and its Definition of Done require *"each shipped with its named mutation executed and
the failure pasted"*. §3.1 gives a KILLED column and no output. **Re-run them and paste the tails.**
If you still hold the original outputs, paste those and say so; if not, re-running is the honest
route. The reviewer re-executed 11 classes and all matched, so this is evidence-of-work, not doubt.

### N11 — §0's table overstates one arm's "clone inherits its source" figure by ~30 %

`bc_any_20` is compared against the **uniform** five-seed mixture (105.5820, gap −0.5363), but the
pooled subset it actually trained on implies a source of **105.4237**, so the honest gap is
**−0.3780**; the rest is sampling skew (`{101:15, 202:28, 303:15, 404:21, 505:21}` against 20 each).
**Report both, and give the transferred fraction at drawn compositions (92.03 %) beside the
equal-weight 92.39 %.** The ordering, the CIs and the conclusion are untouched — this is precision,
not a retraction, and stating it ourselves is worth more than having it found.

## 4. Fold in — cheap, same files

**N4** (`selection_artifact` has no δ-proximity guard; §0.1 reads as though one ran — fix the wording
or add the guard, and say which) · **N6** (a docstring claims a power the test does not have) ·
**N7** (the "no DT comparison" half cannot fail on a BC-only fixture — give it a fixture that can
fail, or delete the claim) · **N10** (`bc_worst2_20` reaches ~0 loss on **all five** seeds with
`behaviour_agreement` exactly 1.0000, not "several" — **and say plainly that perfect imitation of a
worse policy is the mechanism, not a defect**).

## 5. Scope fence

- **No retraining, no re-evaluation, no new arms.** The 20 checkpoints and 2,500 records are final.
- **Do not fix N2** — queued as `DEFERRED` 42 deliberately, so this round stays verifiable.
- **Do not touch** P4.4's merged artifacts, `offline/dt_gate.py`, `offline/dataset.py`,
  `agent/DTAgent.py`, `agent/OfflineBaselines.py`, or any frozen path.
- **No test deleted or weakened.** Count goes up or stays level (645 collected now).

## 6. Definition of Done

- [ ] N1, N3, N5, N8, N9, N11 addressed; N4, N6, N7, N10 folded in
- [ ] Every mutation named in §3 executed and its failure pasted
- [ ] Both artifacts regenerated and compared recursively against a **pre-declared** expected-difference
      set; result pasted
- [ ] Full suite green, real tail pasted, **stating whether it was pinned** (`DEFERRED` 41)
- [ ] All three guards exit 0; `git diff --stat` shows no frozen path
- [ ] Return Packet appended to `docs/returns/P4.5.md` as `## 13. Review-fix round` — do not overwrite
- [ ] §6's checkbox stays unticked; it is mine, in the merge commit
