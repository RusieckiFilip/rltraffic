# BRIEF 12 — P4.4 review fixes (F1–F4, F7–F9)

**Mode:** Claude Code implementation session · **Branch:** stay on `task/p4.4-offline-baselines`
(the task is **not merged**; these commits extend it) · **Worktree:** `/home/filip/rltraffic-p44`
**Supersedes nothing.** `BRIEF_11` still governs P4.4's design; this brief governs only the fix round.
**Read first, from disk:** `docs/reviews/P4.4.md` (the full review and my disposition), then this file.

---

## 1. Where you stand

P4.4 is **PASS-WITH-NOTES**. The independent reviewer retrained 3 of your 15 checkpoints, re-rolled
16 cells, rebuilt the 72,000-row transition table by a route sharing no code with yours, and
regenerated `p4_4_baselines.json` byte-identically. **It could not move a single reported number.**
Your result stands and is going into the paper.

What it did find is a **coverage claim that is false**, and a headline arm with **no test power over
its own learning rule**. Both are in the two places this project cares about most: a docstring that
describes a guarantee the artifact does not provide, and the pre-registered fairness constraint.

**Nothing here is a rebuke of the work.** Your §6.4 disclosure — finding a tautology in your own
`_run_report` by re-reading it — is the behaviour that makes this round small instead of large.

## 2. The frozen constraint on this entire round

> **NO REPORTED NUMBER MAY CHANGE.**

Every ATT, CI, p-value, effect size, verdict, recovered fraction and digest in
`docs/data/p4_4_baselines.json` is now first-hand, independently reproduced, and cited in
`docs/reviews/P4.4.md`. This round adds tests and one artifact field. It does not retrain and it does
not re-evaluate.

**Prove it, do not assert it:** regenerate `p4_4_baselines.json` from the untouched raw outputs in
`output/p4_4/` and diff **every numeric field** against the committed copy. Paste the result. The
only permitted artifact difference anywhere is F3's new block in `p4_4_training.json`.

> ⚠️ **SUPERSEDED 2026-08-12 by §7.1 — read that instead, and the last sentence above is now wrong.**
> The implementer showed that `top_return_filter` is **embedded in `p4_4_baselines.json`**, so F3
> necessarily propagates into both artifacts and the sentence above cannot be satisfied. **§7.1 rules
> that both are regenerated**, in two stages, and that the second comparison is a **full recursive
> JSON comparison over every path and type** — not a numeric-field diff, because verdicts,
> `policy_source` and digests are strings. **The invariant in the box above is unchanged and remains
> the point of the round; only the proof procedure changes.**

If any change you make would alter a number, **stop and report** — that is a finding, not a task.

## 3. The standard for every test in this round

The code is correct. A test you add will pass the moment you write it, and **a passing test proves
nothing here.** Red-first cannot apply.

> **For each test below: write it, then apply the named mutation to the source, run the suite, and
> paste the failure. A test whose mutation you did not execute does not count as delivered.**

Revert every mutation afterwards and confirm the suite is green and the artifact still regenerates
byte-identically. ⚠️ **Byte-identity applies to the stage-1 check only (§7.1); after the F3 patch the
standard is the full recursive comparison.**

---

## 4. Blocking items

### F1 — the false coverage claim (MAJOR)

`offline/offline_baselines.py:554-559`, `iql_targets`'s docstring:

> *"The training loop takes its targets through this function, so the test that pins the behaviour
> guards the path training takes."*

**It does not.** `gamma` and `next_value` are both supplied at the call site in `train_iql`
(~line 852). The identical semantic mutation placed there survives all 58 tests:

```
M2-in-helper    horizon absorbing inside iql_targets  -> 1 failed, 57 passed   KILLED
M2-at-call-site horizon absorbing inside train_iql    -> 58 passed             SURVIVED
```

**Required: make the sentence true, do not delete it.** A test must fail when the *training loop*
stops bootstrapping through the horizon — not only when the helper does. Deleting the claim is the
acceptable fallback only if you can show pinning the call site is infeasible, and you must say why.

### F2 — IQL's learning rule has no behavioural test (MAJOR)

`test_training_iql_records_the_awr_weight_diagnostic` is the only test that runs IQL training and it
asserts key presence and `0 ≤ x ≤ 1`. **The arm that beats our Decision Transformer is protected by
reading.** Four mutations survive 58/58; two, retrained in full, move IQL by +1.08 and +2.40 ATT —
1.7× and 3.8× δ — and both collapse IQL toward BC (agreement 98–99 % against the true 95.0 %).

Add tests with power over each of the four, **each proved by executing the mutation**:

| # | mutation that must fail your test |
|---|---|
| a | `Q_target → Q` in the V (expectile) loss |
| b | the τ-asymmetry reversed (`τ = 0.3` behaviour: the `residual < 0` branch swapped) |
| c | `gamma = 0.0` **at the `train_iql` call site** (this is F1's test; it may be the same test) |
| d | the AWR advantage sign flipped (`v_pred − q_taken`) |

**Do not retrain to test these.** A short run on the synthetic fixture with a known-direction
assertion is the right instrument — e.g. that an expectile loss with τ=0.7 is asymmetric in the
declared direction on hand-built residuals, and that the target of a last transition equals
`r + γ·V(s_T)` when computed by the *training loop*, not by the helper. Cheap, deterministic, and it
fails for the right reason.

### F3 — the mechanism behind %BC is unreported (MAJOR, reporting) — ⚠️ **RESCOPED 2026-08-12, see §7.5: this is a RESULT, not provenance**

The reviewer confirmed and strengthened a finding of mine. The top decile is not a demand selection —
it is a **behaviour-seed** selection:

```
top-20 by return, per seed block: {202: 10, 101: 9, 505: 1, 303: 0, 404: 0}
P(max block count >= 10 | uniform random top-20) = 0.00720     (20,000 permutations)
per-seed HELD-OUT MAPPO@1000 ATT: 202 103.53  101 103.61  404 106.00  505 106.98  303 107.80
Pearson r(training return, held-out ATT) = -0.991, on DISJOINT draw sets (1-200 vs 1000-1099)
```

**Required:**
1. Record the composition in `docs/data/p4_4_training.json` under `top_return_filter` — the per-seed
   counts, the permutation p-value with its iteration count and RNG seed, and the correlation with
   its sample size. Computed from data already on disk; **no simulator, no training.**
   ⚠️ **SUPERSEDED by §7.2:** report the **exact** multivariate-hypergeometric p-value
   **`0.007225300`** as primary, with its null named in the same sentence; the permutation estimate
   becomes a **cross-check**, not the reported figure. I enumerated the exact value independently.
2. State it in the Return Packet **in the same breath as the %BC number**, in the style of the IQL
   binding sentence, and say which mechanism the result claims. Because `D16` makes seed perfectly
   confounded with demand block, *"%BC"* on this corpus is operationally **"clone the best 2 of 5
   behaviour policies"**, which is a different mechanism from a within-policy return filter.
3. Keep §8.5's honest *"I did not test that"* — it is now partly answered, and the answer is that
   demand is excluded and seed selection is not.

### F4 — the declared evaluation path is unpinned (MINOR, but it moves numbers)

`_baseline_factory` (~line 1473) has no test. Switching the declared greedy path to `explore=True`
survives 58/58 and moves the number (BC −0.184, IQL +0.157 over 20 draws). Pin `explore=False`.

### F7 — the plan claims an assertion that does not exist (MINOR)

`docs/plans/p4.4.md` §5.3 and test plan T10 both say the report **asserts** no CI endpoint lies within
1e-3 of ±δ. Verified in place: the code only **records**
`distance_from_ci_endpoints_to_delta` (one site, no assertion). **Add the assertion** — the plan's
design is the better one, observed distances are 0.157 / 0.916 / 0.593 so nothing is at risk, and it
is one line plus a test. If you instead correct the plan, say so explicitly.

### F8 — a test that does not test what its name says (MINOR)

Deleting `pin_torch_threads(args.torch_threads)` from `main()` survives 58/58. Either give the test
power over `main`, or rename it to what it actually checks. **A test name is a claim.**

### F9 — the disclosure undercounts itself (MINOR)

Packet §6.1 says four tests were added after the implementation; **six** were — the two
`merge_training_runs` tests are also absent at the red commit `933eb54`. The substance is fine (they
shipped in the same commit as their code) but the count is wrong. Correct it. *(Confirmed clean: no
test line was ever deleted; the only deletions are `NotImplementedError` stubs.)*

---

## 5. Scope fence — what NOT to do

- **No retraining. No re-evaluation. No new rollouts.** The 15 checkpoints and 2,600 records are final.
- **No new arms**, no %BC fraction sweep, no RTG work (P4.3 owns it), no CQL.
- **Do not touch** `offline/dt_gate.py`, `offline/dataset.py`, `agent/DTAgent.py`,
  `offline/trajectory_logger.py`, or any frozen path.
- **Do not fix F5, F6, F10, F11, F12** — I have queued them as `DEFERRED` 32–36 deliberately, so this
  round stays small enough to verify. If one is a two-line fix you cannot resist, put it in the packet
  as a proposal, not in the diff.
- **Do not weaken or delete a test** to accommodate a new one. Test count goes up or stays level.

## 6. Definition of Done

- [ ] F1, F2 (a–d), F3, F4, F7, F8, F9 all addressed
- [ ] **Every mutation in §4 executed and its failure pasted** — no exceptions, no "would fail"
- [ ] All mutations reverted; full suite green; real `pytest` tail pasted
- [ ] `p4_4_baselines.json` regenerated and **every numeric field diffed against the committed copy**,
      result pasted; only `p4_4_training.json` differs, and only by F3's block
- [ ] Test count ≥ 599 collected
- [ ] `claude_guard.sh --frozen-only`, `check_english.sh`, `check_test_hygiene.sh` all exit 0
- [ ] `git diff --stat` shows no frozen path and no merged-module edit
- [ ] Return Packet **appended to** `docs/returns/P4.4.md` as a new section `## 12. Review-fix round`
      — do not overwrite the packet; it is the record of what was reviewed
- [ ] The §6 checkbox stays **unticked**: it is mine, in the merge commit

## 7. RULINGS on the plan of 2026-08-12 — **GO**, with one conflict resolved and three sharpenings

The plan is approved. Pre-flight facts G1–G5 are accepted; two of them I re-derived myself rather
than accept, and both hold (below). Start where you proposed, with the F2 fixture.

### 7.1 G5 — the conflict, RULED: regenerate BOTH artifacts, in two stages, in this order

You are right that §2 as written cannot be satisfied: `top_return_filter` is embedded in
`p4_4_baselines.json`, so F3 necessarily propagates there. **You were right to refuse to decide it
silently.** My §2 named the sha256 as the instrument and then treated it as the goal; the goal is that
**no reported quantity moves**, and byte-identity was only ever the cheapest proof of it.

> **Ruling: regenerate both. Do NOT freeze `p4_4_baselines.json` at `e887f298…`.**

**Because inconsistent artifacts are a worse defect than a changed hash.** If the baselines artifact
keeps an old `top_return_filter` and the training artifact carries a new one, then P4.3, P5 and the
paper's own pipeline can read the composition from the file that does not have it and conclude it was
never recorded — a silent trap of exactly the class this project exists to catch. A hash that changed
once, with a documented identity proof beside it, is ordinary.

**The two stages, and the order is the whole point:**

1. **BEFORE the F3 patch:** regenerate `p4_4_baselines.json` and prove it is **byte-identical** to
   `e887f298677f38e7878103c7e2c274e1c759ad2c406e776de480695b81163de6`. This isolates *"my source edits
   move nothing"* from *"my source edits moved something the F3 block then masked"*. **If this fails,
   stop and report — that is a finding, not a step.**
2. **AFTER the F3 patch:** regenerate again and compare **recursively over the full JSON, every path
   and every type — not a numeric-field diff.** Verdicts, `policy_source`, canonical digests and
   format strings are all strings, and a numeric comparison would not see them change. The only
   permitted difference is **added keys** under `training.top_return_filter`; every pre-existing path
   must compare `==` at equal type.

Record the old and new sha256 in the packet. I will annotate `docs/reviews/P4.4.md` where it quotes
the old one, dated, rather than rewriting it.

Your `compose` subcommand's refusal — *"refuses if the recomputed kept streams differ from the ones
training used"* — is a real guard rather than a formality, and it is why patching a committed record
is acceptable here. **Validate fully before the first byte is written** (§7's filesystem-mutation
barrier); a refused `compose` must leave the artifact untouched.

### 7.2 G3 — RULED: report the EXACT p-value as primary; the permutation becomes a cross-check

**This supersedes my §4's instruction to report "the permutation p-value with its iteration count and
RNG seed".** An exact combinatorial p-value is strictly better: it removes an RNG seed from the set of
reported quantities, and it cannot be re-rolled.

**I enumerated it independently and get `0.007225300`** — your 0.007225 to every digit you stated.
Report it as exact, with the **null stated in the same sentence**, because a p-value without its null
is not interpretable: *the multivariate hypergeometric null of drawing 20 of 200 streams uniformly
without replacement from five blocks of 40, P(max block count ≥ 10)*. Keep your 20,000-draw Monte
Carlo (0.007750, seed 20260812, MC se 0.00060, 0.9 σ) as a **cross-check that the enumeration is
right** — that is a genuine second route and it is worth pasting.

### 7.3 F2a — prefer a BEHAVIOURAL instrument over the call-spy, and say so if you cannot

Your observation is correct and honest: at Polyak 0.005 a short run cannot separate `q` from
`q_target` numerically. But *"the target network was consulted"* is a weaker assertion than *"using
the wrong network changes the answer"*, and a spy is the kind of test that passes for the wrong reason.

**Preferred: make them behaviourally distinguishable.** Monkeypatch Polyak to **0** so `q_target`
stays frozen at initialisation, and initialise it to something clearly different from `q`. Then the V
fixed point under `q_target` differs from the one under `q` by a margin you can assert on. **Keep the
spy as a supplement, not as the primary assertion.** If the behavioural form cannot be made to work,
ship the spy and **state in the test's docstring why the stronger instrument was not available** —
that is an acceptable outcome, an undisclosed weaker one is not.

### 7.4 The F2 fixture is approved, and I verified its arithmetic rather than trusting it

The τ-expectile of `{+10, −10}` is `20τ − 10`: **+4 at τ=0.7 and −4 at τ=0.3**, confirmed here both
analytically and by numeric argmin over a 2×10⁶-point grid. **An 8-unit margin on a closed-form target
is an excellent instrument** — deterministic, ~1 s, no corpus, no simulator, and it fails for exactly
one reason. Use `monkeypatch` so the constant edits cannot leak between tests.

### 7.5 F3 is a RESULT, not provenance — rescoped, and this is the most important ruling in the round

I scoped F3 as *"record the block, mention it beside the number"*. That was wrong, and the correction
came from the user's review of this brief. **It is a finding about the mechanism of the arm that beats
our headline method, and it is stronger material than the 1.79 ATT margin itself.** The P4.4 packet's
§8.5 disclosed *"%BC's margin may partly be memorisation of an easier subset — I did not test that"*.
**It is now tested, and the answer is neither of the two options that sentence offers:** the filter is
not selecting easier episodes (demand excluded) and it is not mainly selecting better episodes — **it
is selecting better CHECKPOINTS.**

**State the evidence in this structure, because the strength is in the deduction and not in the p-value:**

1. Per-seed training return ranks the five MAPPO checkpoints `101 −6711.0, 202 −6751.8, 404 −7199.2,
   505 −7293.2, 303 −7570.2`; per-seed **held-out** ATT ranks them `202 103.53, 101 103.61,
   404 106.00, 505 106.98, 303 107.80`. **Pearson r = −0.991, n = 5, on DISJOINT draw sets**
   (1–200 vs 1000–1099), so it cannot be produced by shared draw difficulty.
2. The %BC filter ranks streams **by training return, deterministically**. Given (1), it therefore
   selects the strongest checkpoints — **this is a deduction from the selection rule, not a
   coincidence needing a significance test.**
3. The observed composition `{202:10, 101:9, 505:1, 303:0, 404:0}` is the deduction's visible
   consequence, and D16 (seed ≡ demand block) is what makes "seed" and "block" the same thing here.

**Say precisely what this licenses and what it does not.** It licenses: *on this corpus the top-10 %
return filter performed checkpoint selection, and %BC's advantage over BC is therefore at least partly
an effect of training on the two strongest behaviour policies rather than of filtering episode quality
within a policy.* It does **not** license *"return filtering does checkpoint selection in general"* —
that is a hypothesis, and **n = 5 seeds, one tier, one scenario, one backend.** A correlation of
−0.991 over five points is a strong hint and a weak law; say so in the same sentence you report it.

⚠️ **One precision the p-value does not carry.** `p = 0.007225300` tests **concentration** — that some
block holds ≥ 10 of the top 20 — and **not** that the concentrated blocks are the *best-performing*
ones. The identity claim rests on (1) and (2), not on that number. **Do not let the p-value carry a
claim it does not test**; that is the same error class as the reference-class defect corrected on
2026-08-12.

**Why this sharpens C1 rather than threatening it,** and the packet should say so in one sentence:
C1 is that measured data quality decides the outcome. **This is C1 operating one level finer than the
ladder's tiers — composition WITHIN a tier, at checkpoint granularity — measured rather than
asserted.** It is a better result than the margin it explains.

**Still out of this round's fence:** the decisive experiment is BC trained on seeds 101+202 only,
compared against %BC. **I have registered it as P4.5** (`PROJECT_PLAN` §6, `DEFERRED` 38); it needs a
training run, and §5 forbids one here. Name it in the packet as the open test, and do not run it.

### 7.6 Stage 1's FAILURE was my instrument's fault, not yours — accepted, with the sample named

**All three questions answered: yes, yes, and I am writing the row.** You did the right thing twice —
you stopped when Stage 1 failed instead of reasoning past it, and then you built the instrument the
ruling actually needed rather than declaring the ruling unsatisfiable.

**Q1 — the corrected Stage-1 instrument is ACCEPTED.** *Pre-fix module vs post-fix module, same cwd,
same HEAD, same inputs → byte-identical `5289e3ba…`* isolates exactly what Stage 1 existed to isolate:
**do this round's source edits move any output?** They do not. My §7.1 named an identity check on a
container whose contents include a write-time quantity, which is the **same structural class as
`DEFERRED` 29** — an identity claim resting on a container property rather than on content. You named
that yourself; it is the right diagnosis and the second time this class has cost this project a step.
**Verified independently here:** the committed artifacts record `c13aaa9`, and the test your corrected
F1 docstring names (`…bootstraps_through_the_horizon_at_its_call_site`) exists at
`tests/test_offline_baselines.py:939` — so the docstring's claim is now true of the artifact, which
was the point of F1.

**Q2 — Stage 2 as you propose, with two conditions.**
1. **The expected-difference set is a LITERAL DECLARED LIST, written before the comparison runs**, and
   the comparison **fails on any path not in it**. An expected difference named in advance is a
   specification; the same difference explained afterwards is an excuse, and the two are
   indistinguishable in a packet unless the order is visible. Declare `{.runtime.git_commit}` plus the
   additions under `training.top_return_filter`, and nothing else.
2. **State the sample.** *"Only `git_commit` differs"* is true **on this machine, in this session**.
   `cuda_device_name`, `torch_version`, `numpy_version` and `python_version` sit in the same block and
   would differ on other hardware. Say which you checked and where.

**Q3 — the row is mine and I am writing it, and the artifacts say something sharper than we both
said.** A single `git_commit` field **cannot express a chunked campaign's provenance even in
principle**, and this one already does not: measured here,
`output/p4_4/gate_a.json` carries **`738884b`** while the three `eval_*.json` carry **`c13aaa9`**. So
the committed `c13aaa9` is *"when the report was assembled"*, not *"the commit whose code produced
these measurements"* — there is no such single commit. **The field is not merely awkward to
regenerate; it is under-specified for the campaign that produced it.**

**Forward fix: DEFERRED to P4.3** (`DEFERRED` 39), which writes new artifacts and can adopt a split
cleanly — a frozen *measurement* provenance carried from the inputs, and a separate *written-at*
commit. **Not in this round:** it is a schema change, §5's fence holds, and the disclosure below buys
most of its value for none of its risk.

**Binding on §12, and it is a reporting requirement rather than a schema change:** state, **measured
from the eval JSONs rather than remembered**, which commit produced each input — `738884b` for Gate A,
`c13aaa9` for the three evaluation arms — and state plainly that the regenerated artifact's
`runtime.git_commit` is **the commit at which the file was written, not the commit whose code produced
the numbers.** Without that sentence the next reader takes the regenerated value for measurement
provenance and is wrong.

**One more, from §7's own rule that test count is a signal.** You report **606 passed, 1 skipped** =
**607 collected** against 599 before, i.e. **+8**, while the plan listed **7** new tests. Almost
certainly one test split; reconcile it explicitly in §12 rather than leaving the delta unexplained. A
count that does not add up is exactly what the rule exists to surface, and an innocent explanation
stated is worth more than an unstated one.

### 7.7 Smaller rulings

- **F4 — assert on the CALL, not on the resulting behaviour.** Raised by the user, and correct.
  `explore=False → explore=True` moves the number *today* (the reviewer measured BC −0.184, IQL +0.157
  over 20 draws), but that is a property of the **current weights**, not of the code: BC reproduces the
  logged action on 99.8 % of training positions, so a near-deterministic policy can make sampling and
  argmax agree, and the mutation would then survive as **equivalent while proving nothing about the
  guard**. **Primary assertion: what `_baseline_factory` passes.** Add the behavioural check only as a
  supplement, if a difference is demonstrable.
  ⚠️ **This looks like the opposite of §7.3's ruling and is not — read both together.** For F2a the
  question is *semantic* (**does using the wrong network change the answer?**), so behaviour is the
  right level and a call-spy is a proxy for it. For F4 the question is *contractual* (**is the declared
  evaluation path the one that runs?**), and `explore=False` **is** the declared quantity, so asserting
  it is the direct statement rather than a proxy. **The rule underneath both: assert at the level the
  claim is made at.**
- **G3 — two routes under one null test the arithmetic, not the model.** Also the user's, also correct.
  My enumeration and the implementer's Monte Carlo agree to every stated digit, but **both assume the
  multivariate-hypergeometric null** (20 of 200 drawn uniformly without replacement from five blocks
  of 40). The agreement therefore confirms the **calculation** and leaves the **model** untested.
  **Keep §7.2's instruction to name the null in the same sentence, literally** — that is what carries
  the assumption to the reader. **General form, now in `PROJECT_PLAN` §7's redundancy rules:
  independence of ROUTE is not independence of ASSUMPTION**, and our double-computation rule did not
  distinguish them.

- **The 7 new tests are characterisation tests written against correct code.** That is expected in a
  fix round and is not a defect — but **§12 must disclose it in its own words**, rather than relying
  on §6.1's disclosure, which is about the original round.
- **F9: correct the count in place and date it** (four → six), as you propose. Do not rewrite the
  surrounding disclosure; it is a dated record.

## 8. What happens next, so you can see the shape

I merge, tick §6, and write the Decisions Log rows. **P4.3 is the next task** — RTG calibration —
and it inherits a sharpened, falsifiable hypothesis that your result generated: *a correctly
calibrated return prompt makes the DT select the best behaviour mode, closing the gap to %BC.* Either
outcome is a paper result. Your packet's open question 3 asked whether P4.3 was still the right next
task; it is now the **required** one, and your work is why.
