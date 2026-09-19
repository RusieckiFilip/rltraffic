# Patches a Claude Code session cannot apply itself

## `ci_gate_ceiling_191_p7_3b.patch` — the skip ceiling moves 189 → 191 after P7.3b's merge and its CI fix, and all of it is P7.3b's own gating

**Apply with:**
```bash
git apply docs/patches/ci_gate_ceiling_191_p7_3b.patch && .venv/bin/pytest tests/test_ci_gate.py -q && git add .github/ci/ci_baseline.json tests/test_ci_gate.py && git commit -m "ci(ceiling): 189 -> 191 OBSERVED on run 35459798196 at 708f157 -- P7.3b's anchor-corpus gate and its driver test's interpreter skip" && git push origin main
```
**Measured, not read off a summary.** Run `35459798196` on `main` at `708f157` (the merge of P7.3b's CI fix `a790ad1` on top of P7.3b's merge
`732f378`); its only failing step on both suite jobs was the ceiling gate itself (`FAIL pytest-gate: 191 tests skipped against a declared
ceiling of 189`), which is the registered route working. Both legs downloaded with `gh run download`, every `<skipped>` message extracted
from `junit.xml` with `xml.etree` (`output/ci_runs/skips.py`, falsified first against the 189 run: 2,174 / 189 / identical legs), the
`file:line` prefix stripped, the multisets compared **leg against leg** (identical: 2230 tests, 191 skipped, 0 failures, 0 errors on both)
**and run against run** against the 189 run `35338316324`. **Two message texts are new and nothing was removed:** the anchor-corpus gate of
`tests/test_anchor_training.py` (`corpus_or_checkpoint` +1) and `test_the_anchor_driver_refuses_a_bogus_stage_and_creates_nothing`
skipping where the main tree's interpreter is absent (`main_tree_interpreter` +1, the fix of `BRIEF_38` Amendment C). **Two earlier runs
on the merged tree were NOT usable and are named in the entry:** `35451103010` carried 190 skips AND ONE REAL FAILURE (that same
driver test, assuming this laptop's interpreter path), and the run on `732f378` itself was cancelled by the workflow's `cancel-in-progress`.
Verified end to end in a scratch worktree before this file was written: `tests/test_ci_gate.py` green with the new baseline, and
`.github/ci/ci_gate.py pytest-gate` run against BOTH legs' real `junit.xml` + `pytest.txt` under the new baseline, exit 0 on each;
`git apply --check` clean on `main`. Two files: `.github/ci/ci_baseline.json` (the 189 entry nested as `superseded` with its
`why_it_was_wrong`; `re_measure_required_at` re-pointed at P7.3d's merge) and one literal in `tests/test_ci_gate.py` (`CEILING_CHAIN`, now 13 links).

## `agents_relay_discipline.patch` — the coordinator relays nothing but four things; a brief goes to the implementer once, whole, with its gates in it

**Apply with:**
```bash
git apply docs/patches/agents_relay_discipline.patch    # on main, from the repo root
git add .claude/agents/master-coordinator.md .claude/agents/implementer.md
git commit -m "agents: relay nothing but four things; a brief goes once, whole, with its gates (PROJECT_PLAN section 7, 2026-09-19)"
```
The author's process ruling of 2026-09-19 (`PROJECT_PLAN` §7, same date), after P7.3b cost him a dozen relays. Three
hunks in `.claude/agents/master-coordinator.md`: a **Relay nothing but four things** paragraph under *How you behave*
(the author hears from the coordinator only for a token, a registration-changing ruling, a finding that could make a
number wrong, or a finished task); the brief format gains **the gates, in order** and the *once, whole, never section
by section* rule with the source-file cap restated as per-commit; the workflow diagram's step 4 becomes an amendment on
`main` the implementer reads after `git merge main`, with no relay. Two hunks in `.claude/agents/implementer.md`: the
one merge the implementer DOES run — `git merge --no-edit main` into the branch, every session and before every gate,
`BLOCKED` on conflict — and that a passed gate is not reported to the author. Built against the files as they stand
after `agents_incremental_findings.patch` was applied; verified with `git apply --check` on `main` at creation
(`+34/−8`, two files). Docs-only; no code, no tests.

## `agents_incremental_findings.patch` — subagents persist findings as they go; two 25-minute reviewers died on an API limit and returned nothing

**Apply with:**
```bash
git apply docs/patches/agents_incremental_findings.patch    # on main, from the repo root
```
Adds one section to `.claude/agents/contract-reviewer.md` (checklist first within three tool calls; one append per finding before the next command; read only named line ranges; continue-from-file semantics) and one paragraph to `.claude/agents/master-coordinator.md` (every long spawn names a `FINDINGS.md`, splits into ≤ 15-minute mandates, and on an agent's death files the partial and re-spawns a continuation). The rule itself is in `PROJECT_PLAN` §7 (2026-09-11); this patch makes it load with the agents rather than depend on the coordinator remembering to write it into each mandate. Verified with `git apply --check` on `main` at creation. Docs-only change to two agent definitions; no code, no tests.

## `ci_gate_ceiling_189_p7_3a.patch` — the skip ceiling moves 178 → 189 after P7.3a's merge, and all +11 is P7.3a's own gating

**Apply with:**
```bash
git apply docs/patches/ci_gate_ceiling_189_p7_3a.patch    # on main
.venv/bin/pytest tests/test_ci_gate.py -q                 # -> 34 passed
```
Verified with `git apply --check` against `main` at `d92947e` (clean), and **end to end**:
`ci_gate.py pytest-gate` run against the real `junit.xml` and `pytest.txt` of the run being measured,
with the new baseline, prints
`OK pytest-gate: passed=1985 skipped=189 failed=0 errors=0, skip ceiling 189 (0 to spare)` and exits 0.

**Two files:** `.github/ci/ci_baseline.json` and one literal in `tests/test_ci_gate.py`
(`CEILING_CHAIN`, now 12 links deep).

**Measured, not read off a summary.** Run `35338316324` on `main` at `d92947e`, both legs downloaded
with `gh run download`, every `<skipped>` message extracted from `junit.xml` with `xml.etree`, the
`file:line` prefix stripped, and the multisets compared **leg against leg** (identical: 2,174 tests,
189 skipped, 0 failures, 0 errors on both) **and run against run** — the 178 run `35085215720` was
downloaded and diffed directly. 8 message texts are new; **nothing was removed**.

**The classifier was cross-checked before it was used:** applied to the 178 run's own messages it
reproduces that baseline's declared breakdown exactly (92 / 42 / 14 / 2 / 24 / 4). The new families
are `+7 campaign_output` (draw 5's parity ×3, draw 201's ×2, the CityFlow held-out draws, draw 1000's
parity), `+1 corpus_or_checkpoint` (the two subjects' checkpoints), `+1 sumo_traci` (a compound
predicate, assigned to the first condition it names, this file's convention since 156), and two NEW
families rather than forced fits: `main_tree_interpreter` (1) and `shallow_checkout` (1). Each new
family carries its own entry in `ceiling_is_a_consequence_of`, which the gate's own tests require.

⚠️ **The red this closes was NOT only the ceiling.** The preceding run `35335268248` carried 188 skips
**and one real failure** — `test_j1_code_changed_since_reads_real_git_history`, which assumes full git
history while `actions/checkout@v4` checks out shallow. That was fixed on its own branch and merged
BEFORE this ceiling was measured, so no ceiling here was ever committed over a red suite. Three prose
figures inside the baseline that still said 178 / 76 / 14 were refreshed in the same edit — the
staleness those notes were themselves written about.

## `ci_gate_ceiling_139_p8_4b.patch` — the skip ceiling moves 123 → 139, and all +16 is P8.4b's own tests

**Apply with:**
```bash
git apply docs/patches/ci_gate_ceiling_139_p8_4b.patch    # on main
.venv/bin/pytest tests/test_ci_gate.py -q                 # -> 34 passed
```
Verified with `git apply --check`, then applied, `tests/test_ci_gate.py` run (**34 passed**) and
reverted with `cmp` proving both files byte-identical to their originals. **Additionally verified
end-to-end**, which the previous entry did not do: `ci_gate.py pytest-gate` run against the real
`junit.xml` of the run being measured, with the new baseline, returns
`OK pytest-gate: passed=1445 skipped=139 failed=0 errors=0, skip ceiling 139 (0 to spare)` and exits 0.

**Two files:** `.github/ci/ci_baseline.json` and one literal in `tests/test_ci_gate.py`
(`CEILING_CHAIN`), which is exactly the "moving the ceiling is ONE literal" design that the 2026-08-24
chain-walk introduced — the first move to test that promise, and it held.

**Measured, not read off a summary:** `gh run download 33502850557`, `junit.xml` parsed with
`xml.etree`, skips tallied by message. **Both legs identical** at `tests=1584 skipped=139 failures=0
errors=0`. All **33 distinct** skip messages classified by inspection, not by regex:

| category | 123 | 139 | delta |
|---|---|---|---|
| cityflow | 34 | 41 | **+7** |
| corpus_or_checkpoint | 75 | 75 | 0 |
| sumo_traci | 10 | 10 | 0 |
| matplotlib | 2 | 2 | 0 |
| campaign_output | 2 | 11 | **+9** |

⭐ **The +16 is entirely P8.4b's own new tests** — `test_engine_att_reference.py` needs the CityFlow
engine, `test_att_rederivation.py` needs `output/p8_4b_rederivation` and the other campaign trees —
**which is precisely what the superseded entry's expiry named before the merge that triggered it.**
The `predicted_delta` field declines to predict for the **fifth** time running; the four coordinator
predictions before the protocol were wrong, and it was right again here.

⚠️ **Written by the coordinator at the author's explicit instruction.** `tests/**` and `.github/ci/**`
are normally outside the coordinator's role; the crossing is disclosed here, in the baseline's
`how_it_was_obtained`, and in the Decisions Log entry of 2026-09-01. ⚠️ **Two defects in the
coordinator's first draft were caught by `tests/test_ci_gate.py` and not by the coordinator**: three
dropped fields from `re_measure_required_at` (including `predicted_delta`, whose loss `ci_gate.py`
reads behind a `.get` default and therefore cannot notice) and a superseded link nested as the whole
prior `measured` block instead of the reduced `{value, why_it_was_wrong, superseded}` record.

## `ci_gate_ceiling_123_p8_4b.patch` — the skip ceiling moves 121 → 123, and part of it is a reclassification

**Apply with:**
```bash
git apply docs/patches/ci_gate_ceiling_123_p8_4b.patch    # on main
.venv/bin/pytest tests/test_ci_gate.py -q                 # -> 34 passed
```
Verified with `git apply --check`, then applied, `tests/test_ci_gate.py` run (**34 passed**) and
reverted, so the patch is known to apply AND to satisfy the gate it moves. Two files, neither the
coordinator's to write by role (`DEFERRED` 54): `.github/ci/ci_baseline.json` and
`tests/test_ci_gate.py`.

**This is the registered protocol running, not a repair.** `what_to_do` says: *merge, let it go red,
classify `junit.xml`, commit the observed value with its breakdown, do not pre-bump and do not widen
with slack.* That is what this is.

**Measured, not derived — run `33434191684`, both legs, FOUR independent readings:**
`junit.xml` and `pytest.txt` on each of `ubuntu-24.04` and `ubuntu-22.04`, all four giving
**1364 passed, 123 skipped, 0 failures, 0 errors**. The suite step itself PASSED on both legs; only
the ceiling gate failed, which is the registered red.

**The breakdown was classified BY INSPECTION, not by regex** — the 17 distinct `<skipped>` message
prefixes were listed and assigned by hand, because a regex that guesses is how a category silently
moves:

| category | was | now |
|---|---|---|
| `corpus_or_checkpoint` | 76 | **75** |
| `cityflow` | 32 | **34** |
| `sumo_traci` | 10 | 10 |
| `matplotlib` | 2 | 2 |
| `campaign_output` | 1 | **2** |
| **total** | **121** | **123** |

⚠️ **The +2 is NOT one category growing.** `cityflow` +2 and `campaign_output` +1 against
`corpus_or_checkpoint` −1, so part of this move is a **reclassification** of messages the previous
count filed elsewhere, and it is recorded that way rather than as growth.

⚠️ **`ceiling_if_cityflow_were_built` is UNCHANGED at 89**, and that is arithmetic rather than a
stale value: 123 − 34 = 121 − 32 = 89. It looked like a missed edit, so it is called out here.

🔒 **This ceiling expires on the merge of P8.4b — the very branch that produced this patch.**
P8.4b adds `tests/test_engine_att_reference.py` and `tests/test_att_rederivation.py`, both carrying
corpus-, draws- and CityFlow-gated tests, so it **will** breach 123 and is meant to. The expiry says
so before it happens rather than after. No delta is predicted, for the fourth time deliberately:
three coordinator predictions of this delta have been wrong and the no-prediction protocol has been
right every time it was used.

## `ci_gate_ceiling_121_p5_3a.patch` — the skip ceiling moves 104 → 121, exactly as its own expiry said it would

**Apply with:**
```bash
git apply docs/patches/ci_gate_ceiling_121_p5_3a.patch    # on main
.venv/bin/pytest tests/test_ci_gate.py -q                 # -> 34 passed
```
Verified with `git apply --check` against `main` @ `d312dfd` on 2026-08-26, and applied-and-run in a
throwaway worktree. Two files, neither the coordinator's to write by role: `.github/ci/ci_baseline.json`
and `tests/test_ci_gate.py`.

**This is the registered protocol running, not a repair.** `re_measure_required_at.event` named
*"merge of P5.3"*; P5.3a merged at `f4a73d3`; the gate went red on the ceiling and **on nothing else** —
the suite step passed on both legs, as did all three guards, the wheel gate and `libc-matrix`.
`what_to_do` says: *merge, let it go red, classify `junit.xml`, commit the observed value with its
breakdown, do not pre-bump and do not widen with slack.* That is what this is.

**Measured, not derived — run `33011650275`, both legs:**

| category | now | was | delta |
|---|---|---|---|
| corpus / checkpoint | **76** | 59 | **+17** |
| cityflow | 32 | 32 | 0 |
| SUMO / traci | 10 | 10 | 0 |
| matplotlib | 2 | 2 | 0 |
| campaign output | 1 | 1 | 0 |
| **total** | **121** | 104 | **+17** |

**FOUR independent readings agree**: `junit.xml` **and** `pytest.txt` on **each** of the two legs, all
four giving 121 and the same five-category split. **The entire +17 is one family**, which is what
P5.3a should do — it added corpus- and checkpoint-gated tests and no engine-gated ones. Six of the
+17 carry a new message shape, `checkpoint not present in this tree: <path>`; two more are
`surviving draw 1000 is absent` and `no integrity manifests in this tree`.
`ceiling_if_cityflow_were_built` follows the arithmetic to **89** (121 − 32).

### ⭐ The point of interest: this is the first ceiling move under the walked-chain design, and the "one edit" claim was tested rather than repeated

`ci_gate_ceiling_104_and_chain_walk.patch` (2026-08-24) claimed *"moving the ceiling is ONE edit from
here on, not four."* **Measured today: changing `CEILING_CHAIN` from `(104, 98, 72, 62)` to
`(121, 104, 98, 72, 62)` and NOTHING else in that file gave `34 passed`.** Under the previous
depth-addressed form the same move needed four edits and the fourth was masked by the first
(`DEFERRED` 54). The docstring now records the measurement instead of the promise.

**Falsified, not inspected — 7 mutations, 7 behaved as required**, in a throwaway worktree:

```
KILLED   ceiling 121->122 with measured.value left at 121
KILLED   drop the ROOT link 62 -- NOW AT DEPTH 4
KILLED   corpus_or_checkpoint 76->75 (breaks sum == ceiling)
KILLED   ceiling_if_cityflow_were_built 89->72 (the stale value)
KILLED   delete re_measure_required_at.reason
KILLED   drop the NEW 104 link from the middle of the chain
KILLED   CONTROL: reword prose only -- 34 passed, as required
```

⭐ **The second one is the whole argument for the redesign: the root `62` is now at depth 4 and
deleting it still fails the suite.** At the 98 → 104 move, under the old design, the root silently
fell out of the pinned set — which is precisely what the old docstring said must never happen.

**Not touched:** the `re_measure_required_at` protocol text, and `predicted_delta` remains *NOT
PREDICTED* — deliberately, for the third time. Two coordinator predictions of this delta were wrong;
the third expiry was handled with no prediction at all and cost minutes.

## `ci_gate_ceiling_104_and_chain_walk.patch` — DEFERRED 54, THIRD instance, and the shape that guaranteed it

**Apply with:**
```bash
git apply docs/patches/ci_gate_ceiling_104_and_chain_walk.patch    # on main
.venv/bin/pytest tests/test_ci_gate.py -q                          # -> 34 passed
```
Verified with `git apply --check` against `main` @ `884a6df` on 2026-08-24, and applied-and-run in a
throwaway worktree at the same commit. Two files: `.github/ci/ci_baseline.json` and
`tests/test_ci_gate.py`. **Neither is the coordinator's to write** — `tests/**` by role,
`.github/ci/**` by the `DEFERRED` 53 ruling — so this is the patch route, not a workaround.

**What was red.** `69680fa` moved the ceiling in the JSON to 104 and left the second pin untouched:
`tests/test_ci_gate.py:712` still read `== 98`. The suite failed, and the gate failed downstream of
it. All three guards passed on both runner legs, so the TH002 patch introduced nothing.

**Fixed what the file pins, not what the failure showed.** Assertions after the first failure never
run, so the two visible failures were hiding three more:

| # | Pin | On disk | Visible? |
|---|---|---|---|
| 1 | `expiry["reason"]` | **deleted by `69680fa`** | yes (`KeyError`) |
| 2 | `skip_ceiling == 98` | 104 | yes |
| 3 | `superseded.value == 72` | 98 | **masked by 2** |
| 4 | `its_own_superseded.value == 62` | 72 | **masked by 2** |
| 5 | the root `62` | now at depth 3, **pinned by nothing** | **masked by 2** |

**Finding 1 is the JSON's defect, not the test's, and it inverts the natural reading.**
`.github/ci/ci_gate.py:39` declares `reason` part of this file's format and `:308` prints it into the
job summary through `expiry.get('reason', '')` — so the deletion produced an *empty clause in every
job summary* instead of an error. The test was the only thing that noticed. `reason` is restored, and
the loop now pins all five contracted keys.

**Finding 5 is the durable one: the old shape guaranteed its own recurrence.** Each ceiling move adds
a nesting level, so a depth-addressed test needs a *new* assertion every time — and at the 98→104
move the root `62` fell out of the pinned set entirely, which is the one thing the previous
docstring said must never happen. The chain is now **walked** and pinned as one literal,
`CEILING_CHAIN = (104, 98, 72, 62)`, with the root asserted at whatever depth it lands. **Moving the
ceiling is one edit from here on, not four.**

**Two names for one relation is why a walk was impossible, and the coordinator's first draft proved
it.** The JSON keyed the first link `superseded` and every deeper link `its_own_superseded`; the
first version of the walk read the chain as `(104, 98)` and stopped. Caught by running the test, not
by reading it. The JSON is now uniformly `superseded` at every depth, and a recursive **key** check
forbids the second name returning.

**Three dump-greps removed, each measured rather than argued.**
- `assert "40" in json.dumps(block)` — *the alternative (CityFlow built) must be recorded too*.
  **It expired rather than having always been empty:** with the alternative deleted the expression is
  `False` on the 98-era baseline (`4866d52`) and `True` on this one, because the 104 measurement
  itself introduced the string `"1240 passed"` into `measured.result`. **A substring assertion's
  discriminating power is a function of data it does not name, so it can stop discriminating with
  nobody editing it.** Replaced by arithmetic over a new declared `skip_breakdown`
  (`sum == skip_ceiling`) and `ceiling_if_cityflow_were_built == skip_ceiling − breakdown["cityflow"]`.
- `assert "CityFlow" in json.dumps(block)` — **survived deleting the entire CityFlow entry**, carried
  by `"CityFlow not built"` in `measured.condition`. Replaced by a per-category invariant: one
  recorded reason per skip category.
- The coordinator wrote a **third** one in this very patch (`"its_own_superseded" not in
  json.dumps(block)`) and it was tripped by the baseline note *documenting the rename* — a dump-grep
  cannot tell a key from prose about a key. Replaced by the recursive key walk. Recorded here because
  writing the defect one assertion after removing it is the finding.

**Two stale prose entries corrected, and one category that never existed.**
`ceiling_is_a_consequence_of` still said *"32 of these 72 skips … from 72 to 40"* and *"28 corpus-
gated tests"* — the 2026-08-17 numbers, unedited through **two** ceiling moves, because the only
assertion over them was the expired grep. Now 104/72 and 59, and the `campaign_output` category that
`measured.breakdown` counted from the 104 move but this list never listed is added. The new
`len(consequences) == len(breakdown)` assertion is what would have caught it.

**Falsified, not inspected — 10 mutations, 10 behaved as required**, run against the patched tree in
a throwaway worktree:

```
KILLED   M1 delete re_measure_required_at.reason (= 69680fa's own regression)
KILLED   M2 skip_ceiling 104->105 with measured.value left at 104
KILLED   M3 delete the ROOT link 62 from the chain
KILLED   M4 skip_breakdown.cityflow 32->31
KILLED   M5 delete ceiling_if_cityflow_were_built
KILLED   M6 re-introduce the its_own_superseded key name
KILLED   M8 delete the CityFlow prose alternative
KILLED   M9 add a 6th skip category with no reason recorded for it
KILLED   M10 blank why_it_was_wrong on the root link
KILLED   M7 CONTROL: reword unrelated prose -- 34 passed, as required
```

M8 **survived** the first time and is why the third dump-grep was found; M3 is the defect the old
test could not catch at all after the 104 move.

**Measured on the patched tree:** `tests/test_ci_gate.py` **34 passed**; full suite **1288 passed, 56
skipped** (pinned; 56 not 104 because this machine has the corpus and CityFlow — collected totals
agree with the runner's 1238 + 2 + 104 = 1344). Guards: **English 4**, **hygiene 16**, **0 findings in
`test_ci_gate.py`** — the baseline needs no guard edit.

**The ceiling value itself is not re-derived here.** 104 was counted from the runner's `junit.xml` in
`69680fa` and is taken as given; this patch pins it, explains it and makes the parts sum to it.

## `check_test_hygiene_conditional_constant.patch` — TH002 misses `assert X if False else True`

**Apply with:**
```bash
git apply docs/patches/check_test_hygiene_conditional_constant.patch
bash scripts/check_test_hygiene.sh ; echo "exit=$?"   # -> exit 1, still 16 findings, none new
```
Verified with `git apply --check` on 2026-08-24. One hunk, ten lines, adding a second TH002 clause.

**Why:** P5.2's implementer disclosed writing `assert X if False else True` into a draft test — which
reduces to `assert True`, the exact class this guard exists to reject. **Measured with a positive
control, because "the guard found nothing" and "the guard did not look" are indistinguishable:** on a
scratch tree the guard **flags `assert True` as TH002** and **does not flag** the conditional form. So
it looked, and it has a blind spot. TH002's regex matches only a bare constant operand.

**Falsified rather than asserted:** the same scratch tree, current guard **0 findings**, patched guard
**`tests/test_probe.py:3: [TH002] conditional constant: the taken branch is constant`** — and a
legitimate `assert x == 1 if flag else y == 2` stays clean under both.

**Effect on the recorded baseline: NONE.** The 16 findings are unchanged; this adds a rule, not a
finding. `.github/ci/ci_baseline.json` needs no edit.

**Why a patch and not an edit:** `scripts/**` is denied at permission level as a glob, and the guard
honours `check_test_hygiene.sh` as a `FROZEN_EXCEPTIONS` entry — the patch route is the mechanism
(CLAUDE.md §1). The patch was generated by `diff -u` over two copies in `/tmp` rather than hand-written,
after two hand-written attempts produced corrupt hunks.

## `ci_gate_baseline_pins.patch` — DEFERRED 54, the four baseline pins (blocking CI)

**Apply with:**
```bash
git apply docs/patches/ci_gate_baseline_pins.patch      # on main
.venv/bin/pytest tests/test_ci_gate.py -q               # 34 pass once applied
```
Verified 2026-08-19 with `git apply --check` against `tests/test_ci_gate.py` at blob
`3d3b3559`, which is **byte-identical on `main` and on `task/p5.2-tier-sweep`** (`sha256`
compared directly, and `git diff HEAD main -- tests/test_ci_gate.py` is empty) — so the check run
in the branch worktree is a check against `main`'s bytes. 105 lines, six hunks: four literals and
two docstrings.

**Why it is a patch and not a commit, and this one is unusual — it is not a permission problem.**
`tests/` is not frozen and not deny-listed; a session can edit it. **The reason is that the change
is correct on `main` and WRONG on the branch it would have been committed from.** `96e12f8` moved
`.github/ci/ci_baseline.json` on `main` (English total 5 → 4, skip ceiling 72 → 98, superseded
62 → 72 nesting the old 62) and did not move the second copy of those constants in
`tests/test_ci_gate.py`. `task/p5.2-tier-sweep` does not have `96e12f8`, so it still carries the
old baseline. Measured both ways:

| test file | baseline it reads | result |
|---|---|---|
| patched | `main`'s (4 / 98 / 72 / 62) | **34 passed** |
| patched | the branch's (5 / 72 / 62) | **2 failed, 32 passed** |
| unpatched | the branch's | 34 passed |

Committing it on the branch would have left that branch red for the ~50 h of P5.2's campaign, and
P5.2's Definition of Done requires a green suite. **Coordinator ruling, 2026-08-19: land it on
`main` only.**

⚠️ **The alternative — copying `main`'s `ci_baseline.json` onto the branch to make it green — was
offered and REFUSED, and the refusal is upheld in the record:** the branch's tree measures **6**
English hits, not `main`'s 4, because the branch also lacks `a060fa3`'s `ALLOWED_NAMES` patch. That
would have committed *a measurement that does not describe the tree it sits in* — the exact class of
error this file exists to catch.

**What the four changes are, and why one is a DELETION.**
1. `english["total"]` **5 → 4** — `Mikolaj` joined `ALLOWED_NAMES`.
2. `assert "P5.1" in expiry["event"]` — **DELETED, not updated.** The completeness loop immediately
   above already asserts `event`, `reason` and `mandated_by` are each non-empty, so the load-bearing
   check survives; the two number pins still force an edit to this file at every expiry move, so the
   speed bump survives. **The two numbers are the loosenable quantities; the event is a POINTER and
   cannot be widened.** Pinning it fires at every *correct* retarget — it fired when P5.1's merge
   properly moved the mandate to P5.2, and would fire again at P5.3 — which is the class this repo
   refuses: *a check that condemns correct artifacts teaches the reader to ignore it.*
3. `block["skip_ceiling"]` **72 → 98**.
4. `superseded["value"]` **62 → 72**, plus a new `its_own_superseded["value"] == 62` and its
   `why_it_was_wrong`, so the chain **98 → 72 → 62** stays pinned to its root rather than losing the
   original defect at the next move.

🚨 **FOUR changes and not the three the failures show, and the docstrings now say so, because the
next person to move the ceiling hits this same wall: assertions after the first failure in a test
never run**, so a run reporting three failures can still be hiding a fourth.

**Tested before shipping, not after, and the scratch tree was proved to be the one under test.**
`main`'s `test_ci_gate.py`, `ci_baseline.json`, `ci_gate.py` and `ci.yml` were copied to
`/tmp/mainchk`, the patch applied there with `patch -p1`, and the file run: **34 passed**.
⚠️ The README's own warning applies — a scratch copy can go green against the wrong file — so the
scratch baseline was then perturbed (`skip_ceiling` → 999) and the run went **red on exactly the
ceiling test**, and green again on restore. The branch worktree was untouched throughout
(`git diff --quiet -- tests/test_ci_gate.py`).
## `check_english_mikolaj.patch` — add a thesis co-author's given name to `ALLOWED_NAMES`

**Apply with:**
```bash
git apply docs/patches/check_english_mikolaj.patch
bash scripts/check_english.sh ; echo "exit=$?"   # -> exit 1 with 4 hits, DOWN from 6
```
Verified with `git apply --check` on 2026-08-19 against `scripts/check_english.sh` at blob
`6761831bc773`. One hunk, one line: `ALLOWED_NAMES='Paweł|Woliński|Grudziński'` gains `|Mikołaj`.

**Why a patch and not an edit:** `scripts/**` is denied at permission level as a glob, and the guard
honours `check_english.sh` as a `FROZEN_EXCEPTIONS` entry — so the patch route is the mechanism, not a
workaround (CLAUDE.md §1).

**Why this fix and not a translation:** the script's own failure message says *"If a hit is a proper
noun, add it to ALLOWED_NAMES … instead of removing the diacritics from someone's name"*, and its
line-24 comment anticipates extending the list. **Mikołaj Woliński is one of the four thesis authors
this repository is built on; `Woliński` is already allowed and `Mikołaj` was not.**

**Measured effect — the total goes DOWN, and this is the point.** Current hits are **6**; the recorded
baseline is 5, and the regression is `docs/returns/P8.3.md:141`, where the packet documents the gap by
**writing the name** — *"Mikołaj" (an `ALLOWED_NAMES` gap: only `Woliński` is listed)* — so the
sentence describing the defect became an instance of it. Applying this clears **two** hits, both
carrying the l-with-stroke and n-with-acute of that name: the **repository-root** `README.md:30` (the
author list) and `docs/returns/P8.3.md:141`. **6 → 4.**

⚠️ **What it does NOT clear, checked rather than assumed:** `.claude/agents/master-coordinator.md:158`
carries an n-with-acute from a Polish **word** in an illustrative quotation, not from a name, so it
stays; and the three o-with-acute hits (`scripts/claude_guard.sh:47`, two committed patches) are
P0.9's open false positive and stay by design.

🚨 **WHY THIS ENTRY NAMES THE CHARACTERS INSTEAD OF SHOWING THEM, so a later editor does not helpfully
restore them.** The first version of these two paragraphs QUOTED the diacritics, and the guard then
flagged `docs/patches/README.md:26` and `:30` — **so the prose explaining that a sentence describing
the defect had become an instance of it became an instance of it.** Third occurrence of that shape in
two days, this one inside the fix for the second.
> **The discriminating principle, because there IS a standing exemption and this is not it: the three
> tolerated hits carry their character because they QUOTE the guard's own class, and de-diacriticising
> them would FALSIFY a record of what the guard contains. These two lines carried them by choice, in
> prose written today, describing them. Nothing is falsified by naming them.**
> **An exemption is for text the fix would FALSIFY — not for text that finds the fix inconvenient.**
> ⭐ And naming them is the better prose anyway: **a reader cannot tell l-with-stroke from l, or
> n-with-acute from n, at a glance in a monospace diff — which is exactly the failure being
> documented.** It is also what the guard's own error message does. **`.github/ci/ci_baseline.json` records the post-apply total as 4, so CI reports a
mismatch until this patch is applied — deliberately, so the pending action cannot be forgotten.**

## `mappo_metric_keys_guard.patch` — make contract C8 mechanical (AUTHORISATION C)

**Apply with:**
```bash
git apply docs/patches/mappo_metric_keys_guard.patch
.venv/bin/python -c "import agent.MAPPOAgent as m; print(hasattr(m,'env_global_metric_keys'))"  # -> True
.venv/bin/pytest tests/test_mappo_c8_metric_keys_guard.py -q          # 15 pass once applied
.venv/bin/pytest tests/test_migrate_mappo_checkpoints.py -q           # 15 pass once applied
```
Verified with `git apply --check` on 2026-08-07 against `agent/MAPPOAgent.py` at blob `4a6a06b`
(branch base `main` `4430cce`). If `MAPPOAgent.py` has changed since, re-derive rather than force.
161 lines, three hunks: one stdlib import, two module-level functions, and `save`/`load`.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(agent/MAPPOAgent.py)`
and `scripts/claude_guard.sh` lists it in `FROZEN_PATTERNS` with no exception, so a session cannot
apply it — which is the point. The Bash-heredoc / `cp` / `sed -i` route was **not** taken; the patch
body was generated with `difflib` and written only to `docs/patches/`, and `git status agent/` is
empty. AUTHORISATION C is quoted below and was confirmed by the user on 2026-08-06.

**Tested before shipping, not after.** `agent/` was copied to `/tmp/p26_scratch`, the patch applied
there, and both test files run against it: **15 + 15 pass**.

⚠️ **A scratch copy is easy to get wrong, and the failure is silent** — you get a green run against the
*unpatched* file. Measured on 2026-08-07, first from the repo root:

```
PYTHONPATH=/tmp/p26_scratch .venv/bin/python -c "import agent.MAPPOAgent as m; print(m.__file__)"
  -> /home/filip/rltraffic/agent/MAPPOAgent.py     # the UNPATCHED file
```

**The cause is plain `sys.path` ordering, nothing exotic.** Under `python -c`, `sys.path[0]` is the
process cwd — measured `sys.path[:3] == ['', '/tmp/p26_scratch', ...]` — so running from the repo root
puts the repo's own `agent/` *ahead* of the scratch directory. From any other cwd the same command
works:

```
cd /tmp && PYTHONPATH=/tmp/p26_scratch:/home/filip/rltraffic .venv/bin/python -c ...
  -> agent   /tmp/p26_scratch/agent/MAPPOAgent.py   # patched, as intended
  -> offline /home/filip/rltraffic/offline/...      # repo, as intended
```

*(Corrected 2026-08-07. This entry first blamed the editable install's `sys.meta_path` finder,
"which outranks every `sys.path` entry". **That is false and was never measured.** The finder sits at
`sys.meta_path[4]`, **after** `PathFinder`, so `PathFinder` resolves `agent` from `sys.path` first and
the editable finder is never consulted. Its `MAPPING` covers `agent`, `algorithms`, `envs`,
`experiments`, `metrics`, `rewards`, `states`, `utils` — and **not** `offline`. Worse, the
`conftest.py` line written to "remove that finder" filtered on `type(f).__module__ ==
'__editable___zpp_traffic_control_0_1_0_finder'`, but those objects report `__module__` as `builtins`,
so **it removed 0 of 5 entries and was a no-op**; what actually worked was the
`sys.path.insert(0, scratch)` beside it. The conclusion below is unchanged and the 15 + 15 result
stands — every run asserted `__file__` — but the mechanism is corrected here because a wrong mechanism
in a README is acted on later by someone who cannot re-derive it.)*

**The rule that matters, and it is unchanged: assert the path, do not reason about it.** Any
re-verification of this patch must assert `agent.MAPPOAgent.__file__` starts with the scratch prefix
and abort otherwise. Every run reported here did.

**The guard is proven to bite, by mutation.** With the set comparison neutralised (`if True: return`)
the acceptance test fails; with it regressed to a **width** comparison — literally the pre-patch
semantics — it also fails. So the test catches the same-count case specifically, not merely "an error
happened".

> **AUTHORISATION C — 2026-08-06, Master chat.** `agent/MAPPOAgent.py` may be modified for the single
> purpose of making contract C8 mechanical: (a) persist `self._global_metric_keys` in `save()`;
> (b) in `load()`, **assert set-equality** against the env's metric keys when the field is present, and
> emit a **loud warning** when it is absent (pre-migration checkpoints). Nothing else: no behaviour
> change, no new state, no touching `_build_global_features`. Ships as a patch under `docs/patches/`.
> **ADDITION:** the error message must print the checkpoint's key set, the env's key set and the
> **symmetric difference** — a message that names the difference ends the investigation instead of
> starting it, months from now, for someone without today's context.
> Spent when BRIEF_08 merges. **Confirmed by the user 2026-08-06.**

**What it does, in exactly the five behaviour rows the authorisation allows.**
`save()` gains one key, `global_metric_keys`. `load()` gains a check that runs **before** any state
is adopted, so a rejected checkpoint leaves the agent as it was:

| checkpoint / env | behaviour |
|---|---|
| key absent | loud `RuntimeWarning` (pre-migration checkpoint) |
| key present, `None` | loud `RuntimeWarning` (saved before features were built) |
| key present, env exposes no metrics | loud `RuntimeWarning` (cannot check) |
| key present, sets equal | silent |
| key present, sets differ | `ValueError` printing both sets **and the symmetric difference** |

Presence is tested with `in`, never `payload.get(...) is not None` — those collapse the first two
rows, and only the first means "predates the migration".

**`_global_metric_keys` is deliberately NOT assigned from the checkpoint.** It looks like a free
no-op once the sets are equal. It is not: `_build_global_features` reads `metrics.get(key, 0.0)`, so
keys adopted from a checkpoint against a *later* differing env would substitute a silent `0.0` at an
unchanged width — invisible to the width guard **and** to this check, which has already returned by
then. Leaving the field `None` makes the agent freeze from the env's own `info`, which is where the
check can still fire.

**One derivation, shared.** `env_global_metric_keys(env)` is a module-level function in the patch, and
`offline/migrate_mappo_checkpoints.py` **imports** it rather than reimplementing it — two copies of a
key-set derivation are exactly where a guard and the data it guards drift apart with nothing failing.
It mirrors `metrics/base.py::compute_all`'s filter, so it equals `sorted(info["metrics"])` by
construction; it is pure and makes no engine call, so it is safe on an env constructed but never
reset, which is the state `offline/collect.py` loads a checkpoint in.

**Why it is needed.** The existing guard compares only `global_feature_dim`, and
`_global_metric_keys` is not stored in the checkpoint, so a same-width metric *swap* is silent: the
critic reads different semantics under the same indices with no error. Both consumers converge on
`agent.load()` — `experiments/runner.py:241` (the `--from-checkpoint` reporting path) and
`offline/collect.py:155` — so one assertion there covers collection *and* reporting, which an
assertion in `collect.py` alone would not.

## `master_coordinator_decision_line.patch` — define the closing decision line

**Apply with:**
```bash
git apply docs/patches/master_coordinator_decision_line.patch
grep -c "DECISION NEEDED: <what>" .claude/agents/master-coordinator.md   # -> 1
```
Verified with `git apply --check` on 2026-08-07. **Restart the Master session** to take effect.

**Why it is a patch.** `Edit(.claude/**)` is denied — a session cannot edit its own definition.

**What it does.** Defines the two-form closing decision line that every Master turn must end with, and
scopes `DECISION NEEDED` to irreversible or claim-touching choices. **Found because the format existed
only in the conversation:** the already-committed rule references "the closing decision line" while the
line itself had never been recorded, so a restarted session would emit the `THINGS YOU NEED TO DO:`
block and then no line. The project's signature error — a convention believed to be on disk that was
not — applied to its own operating protocol, and caught by the user asking "is it in the agent?".

## `master_coordinator_grep_plan.patch` — add the "grep the plan before you escalate" rule

**Apply with:**
```bash
git apply docs/patches/master_coordinator_grep_plan.patch
grep -c "Grep the plan before you escalate" .claude/agents/master-coordinator.md   # -> 1
```
Verified with `git apply --check` on 2026-08-06 against `.claude/agents/master-coordinator.md` at blob
`356c1e0`. Tracks the file's `100755` mode, so applying does not drop the executable bit.
**A restart of the Master session is required** for an agent-definition change to take effect.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(.claude/**)`, so a
session cannot edit its own definition — which is the point. The heredoc route was not taken.

**What it does.** Adds TWO rules after "One question at a time" (folded into one patch on 2026-08-06 so it is applied once):
1. before marking anything
`DECISION NEEDED`, grep `docs/PROJECT_PLAN.md` for an existing answer, and if escalating anyway, state
where you looked. Requested by the user on 2026-08-06 after the coordinator escalated the MAPPO
training-demand question as a single either/or when §1's pre-registered 2×2 had already settled half of
it — MAPPO-nominal was a required *cell*, never an alternative. The same rule is already live in
PROJECT_PLAN §7, which every session reads; this patch makes it permanent in the agent definition.
2. **Surface every human action** in a fixed `THINGS YOU NEED TO DO:` block immediately before the closing
   decision line, written as `nothing` when empty so its absence is never ambiguous. Added after *this very
   patch* sat unapplied because it was named mid-message rather than in a dedicated block — the user reads
   the end of the turn.

## `claude_guard_g1.patch` — fix guard defect G1 (new `.py` in a new `experiments/` subdir escapes the check)

**Apply with:**
```bash
git apply docs/patches/claude_guard_g1.patch
bash -n scripts/claude_guard.sh                          # syntax check
.venv/bin/pytest tests/test_claude_guard.py -q           # 11 rows, all pass once applied
```
Verified with `git apply --check` on 2026-08-03 against the `scripts/claude_guard.sh` at blob `b6ad313`
(branch base `main` `1ed3a08`). If `claude_guard.sh` has changed since, re-derive rather than force.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(scripts/**)` (D3), so a
session cannot apply it. The Bash-heredoc route the guard's own header anticipates was **not** taken — an
in-conversation authorisation is a weaker signal than the configured control, which is the exact failure
the deny-list defends against.

**Authorised** by the Master chat on 2026-08-03 (BRIEF_03, AUTHORISATION A), for
`scripts/claude_guard.sh` **only**, "for the single purpose of fixing defect G1". No other change to that
file, and no change to any other file under `scripts/`.

**What it does, in two lines of regex plus a dated comment.**
1. `FROZEN_PATTERNS`: the `experiments/` clause becomes a **prefix** (`experiments/`) instead of the
   extension anchor `experiments/.*\.py$`. `git status --porcelain` collapses a wholly-untracked directory
   into one entry (`experiments/newpkg/`), which the extension anchor never matched — so a new `.py` in a
   new `experiments/` subdirectory was PERMITTED (`docs/notes/D3_falsification.md` section 4). A prefix
   survives the collapse, like every other directory entry.
2. `FROZEN_EXCEPTIONS`: gains `experiments/configs/[^/]*\.json$`, so new run configs stay writable now
   that the `experiments/` clause blocks the whole subtree. It is a **narrow** carve-out, not a bare
   prefix, on purpose (Master-chat review, 2026-08-03):
   - `[^/]*\.json$` exempts **only JSON**, so a Bash-heredoc write of a `.py` into `experiments/configs/`
     is still BLOCKED. A bare `experiments/configs/` prefix would have reopened G1's exact hole one
     directory over (measured: `experiments/configs/evil.py` goes BLOCKED -> PERMITTED under a bare
     prefix). The heredoc route is the half the guard exists to cover, which is why this matters.
   - `[^/]*` forbids a slash, so a config in a NEW subdirectory (`experiments/configs/sub/new.json`,
     which git status collapses to `experiments/configs/sub/`) **fails closed** and is BLOCKED.
     Nested config trees are not used today; if they are ever wanted, that should cost a deliberate
     patch, not leak through this carve-out.
   The `$` anchor binds only to the `scripts/` alternative and to the JSON tail.

Verified safe: `experiments/configs/` today holds 5 files, all top-level `.json`, 0 non-JSON, 0
subdirectories, and P0.6's `p0_threading_bench.json` is top-level JSON -- so no false positive.

Safe more broadly because runtime artifacts resolve to `output/experiments/` (`experiments/config.py`),
never under `experiments/` itself, so a prefix cannot misfire on results, plots or summaries. The G1 test
(`tests/test_claude_guard.py`, 13 rows) fails against the pre-patch guard on `new_pkg_py_G1`,
`new_pkg_sub_py_G1` and `config_subdir_fail_closed`, and passes on all 13 once this is applied;
`config_py_not_exempt` passes both before and after, proving the fix costs nothing.

## `runner_liveness_docs.patch` — document that the thread pin is a liveness fix, not a speedup (P0.5)

**Apply with:**
```bash
git apply docs/patches/runner_liveness_docs.patch
.venv/bin/python -c "import ast; ast.parse(open('experiments/runner.py').read()); print('parses OK')"
.venv/bin/pytest tests/test_runner_threading.py -q
```
Verified with `git apply --check` on 2026-08-03 against `experiments/runner.py` at blob `f74feef`
(branch base `main` `1ed3a08`). If `runner.py` has changed since, re-derive rather than force.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(experiments/*.py)` and
`Edit(experiments/**/*.py)`, so a session cannot apply it. Same reasoning as `runner_thread_pinning.patch`;
the Bash-heredoc route was again not taken.

**Authorised** by the Master chat on 2026-08-03 (BRIEF_03, AUTHORISATION B), for `experiments/runner.py`
**only**, "for the single purpose of documenting the liveness role of `limit_torch_threads()`".
**Comments and docstrings only -- zero executable-statement changes.** This does not authorise moving the
call site, changing `CELL_TORCH_THREADS`, or any behaviour change. The P0.3-fix authorisation ("limiting
per-worker torch thread counts") is spent and does not cover this.

**Proof it is comments-only (not asserted -- computed).** Parse `git show HEAD:experiments/runner.py` and
the patched file, strip every docstring node, compare `ast.dump`: **IDENTICAL** (61586 bytes each). A
mutation control (`int(n_threads)` -> `int(n_threads)+1`) breaks the equality, so the check genuinely
detects executable changes. Comments never enter the AST; the only docstring changed is
`limit_torch_threads`'s.

**What it does.**
1. Rewrites the `limit_torch_threads` docstring, liveness first: a forked pooled worker entering an OpenMP
   region with `nthreads>1` waits forever on team threads `fork()` never duplicated, and `run_matrix`'s
   `as_completed`+`future.result()` has no timeout, so it wedges the suite and the guard silently. Then the
   ordering constraint (child-side work *before* the call runs unpinned; `backend_ready()` already does --
   safe for CityFlow, unprobed for libsumo/moss), the scope (pool only under `workers>1`; the sequential
   path never forks and is pinned as a documented side effect), and speed demoted to a footnote.
2. Retires the cross-session ratio table above `CELL_TORCH_THREADS` (kept visible, marked retired) and
   states the trustworthy single-session figure: 199.2 s -> 50.2 s, ~3.97x at workers=6.

**Corrects review finding N1 in passing.** N1 said a maintainer might make the call "conditional on
`workers > 1` ... and would silently reintroduce an unbounded hang". That is imprecise: the fork happens
*only* when `workers > 1`, so such a condition keeps the pin where the hazard is. The docstring states the
accurate version -- removing, moving-later, or adding child-side work ahead of it is what reintroduces the
hang. `docs/reviews/P0.3-fix.md` is left as the record of what the reviewer said.

## `settings_scripts_glob_deny.patch` — glob-deny all of `scripts/`, drop the ten inert `Write(...)` rules

**Apply with:**
```bash
git apply docs/patches/settings_scripts_glob_deny.patch
.venv/bin/python -m json.tool .claude/settings.json > /dev/null && echo "valid JSON"
```
Verified with `git apply --check` on 2026-08-03 against the `.claude/settings.json` at commit `6787f0e`
(deny array 27 entries → 17). If `settings.json` has changed since, re-derive rather than force.
**A restart of any running Claude Code session is required** — permissions are read at session start.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(.claude/**)`, so a
session cannot apply it. Same reasoning as the two entries below; the Bash-heredoc route was again not
taken.

**What it does, in two independent changes.**
1. **Semantic (D3).** `Edit(scripts/claude_guard.sh)` → `Edit(scripts/**)`. The permission layer now
   covers every file in `scripts/`, *including files that do not exist yet*. Deliberately **no**
   permission-level exceptions: `deny` beats `allow` in this system regardless of specificity, so a
   glob and an exception cannot coexist. The two `FROZEN_EXCEPTIONS` (`check_english.sh`,
   `check_test_hygiene.sh`) remain honoured by the **guard**, which is a separate layer — after this
   patch those two files are denied at permission level and permitted at guard level, by design.
2. **Cosmetic.** Removes the ten `Write(...)` deny rules. They are inert: the CLI's own warning is
   *"Write(...) is not matched by file permission checks — only Edit(path) rules are."* Removing them
   is safe **not** because they are inert but because every one of them has an identical-path `Edit(...)`
   twin, and `Edit(path)` provably governs the Write tool. Ten rules that look protective and are not
   are a trap for the next reader.

**Evidence — all three facts established by running the permission system, never by reading about it**
(isolated `/tmp` workspace, `claude -p --settings`, observable = file contents on disk after the
attempt, never the nested agent's self-report):
- `deny` beats `allow`: an explicit `allow: Edit(scripts/check_english.sh)` did **not** survive
  `deny: Edit(scripts/**)`; the file stayed unmodified while a no-rule control in the same run was
  modified, proving the allow array had loaded.
- `Edit(path)` governs the Write tool: a **Write**-tool call succeeded under an `Edit(...)` allow and
  was blocked by an `Edit(...)` deny.
- Pre-flight of this exact deny list: `scripts/brand_new.sh` (a file with no individual rule) DENIED
  and unmodified; `scripts/check_english.sh` DENIED and unmodified; control modified.
- `Read(...)` denies are **not** inert (tested separately) — the inert class is `Write(...)` only.

**The trade, recorded honestly.** D3 buys prospective mechanical cover on every script that does not
exist yet, and pays for it with bounded friction on two temporary files. Measured cost basis: those two
scripts have needed editing about twice in the project's life, at ~2 minutes per patch round-trip, and
the guard's own comments already schedule both exceptions for deletion once they settle. **Revisit
signal:** if either script goes untouched-but-wanting-a-rule *because* of the patch friction, that is
evidence the friction is not as bounded as assumed — reopen the choice rather than absorbing it.

## `runner_thread_pinning.patch` — pin torch to one thread per cell (P0.3-fix)

**Apply with:**
```bash
git apply docs/patches/runner_thread_pinning.patch
.venv/bin/python -c "import experiments.runner as r; print(r.CELL_TORCH_THREADS)"   # -> 1
.venv/bin/pytest tests/test_runner_threading.py -q
```
Verified with `git apply --check` on 2026-08-02 against the `experiments/runner.py` at commit `a0dfc1d`.
If `runner.py` has changed since, re-derive rather than force.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists `Edit(experiments/*.py)` and
`Write(experiments/*.py)`, so a session cannot apply it. This is the same reasoning recorded for
`claude_guard_hygiene.patch` below, and the Bash-heredoc route was again not taken.

**Authorised** by the Master chat in `docs/briefs/BRIEF_02b_thread_pinning.md`, for
**`experiments/runner.py` only**, on branch `task/p0.3-thread-pinning`, "for the single purpose of
limiting per-worker torch thread counts". The brief's instruction was to use "whatever mechanism the repo
already provides for authorised exceptions ... and if no such mechanism fits, stop and report rather than
weakening the guard". `FROZEN_EXCEPTIONS` does not fit — reaching it would mean editing
`scripts/claude_guard.sh`, which the same authorisation forbids. This directory is the mechanism that
does fit, so `FROZEN_PATTERNS` and `FROZEN_EXCEPTIONS` are left untouched.

**What it does.**
1. Adds `CELL_TORCH_THREADS` and `limit_torch_threads()` to `experiments/runner.py`, with the 2026-07-27
   benchmark table inline as the justification for the value.
2. Calls it in `run_cell`, after the `backend_ready` skip check, so it pins both the pooled path (one
   process per cell) and the sequential path, and a backend-less cell still never imports torch.

**Why `run_cell` and not `ProcessPoolExecutor(initializer=...)` or env vars.** `run_cell` is the unit of
work on both paths, so one call site buys both the parallel win and the `workers=1` win; an
`initializer` reaches only `workers>1`. *(The figures originally quoted here, 5.80x and 1.37x, are
cross-session and were retired 2026-08-03 — the trustworthy measurement is 199.2 s → 50.2 s, ≈3.97x at
workers=6. The argument is unaffected: it turns on **which paths a call site reaches**, not on the size
of the win. More importantly the pin is a **liveness** fix — unpinned parallelism can hang unboundedly,
not merely run slowly — so the justification never rested on speed in the first place.)* Env vars also work today (torch enters lazily via `build_agent`,
so nothing has imported it at fork time) but mutate `os.environ` process-globally and would stop working
silently the day anything imports torch earlier — surfacing as a slow run rather than an error.

**Accepted side effect.** On the sequential path this pins the *calling* process, not a child. That is
what pinning the sequential path asks for (the "1.37x" once quoted here is a retired cross-session
ratio — see the note above), and it is documented in the function's docstring; it does mean
importing `experiments.runner` and running a cell changes the host process's torch thread count.

## `claude_guard_hygiene.patch` — wire the test-hygiene check into the guard hook

**Apply with:**
```bash
git apply docs/patches/claude_guard_hygiene.patch
bash -n scripts/claude_guard.sh                      # syntax check
bash scripts/claude_guard.sh --tests-only ; echo "exit=$?"
```
Verified with `git apply --check` on 2026-08-01 against the `scripts/claude_guard.sh` at commit
`9624d73`. If `claude_guard.sh` has changed since, re-derive rather than force.

**Why it is a patch and not a commit.** `.claude/settings.json` deny-lists
`Edit(scripts/claude_guard.sh)` and `Write(scripts/claude_guard.sh)`, so a session cannot apply it.
That entry is deliberate — CLAUDE.md rule 1: *"a session must not be able to unfreeze itself; the
guard, the hook wiring and the permission deny-list are exactly what stops a wrong assumption from
reaching a frozen file."* The block was **not** circumvented with a Bash heredoc, although that route
exists and the guard explicitly anticipates it: an in-conversation authorisation is a weaker signal
than the configured control, and treating it as stronger is the exact failure the deny-list defends
against. Authorised by the Master chat on 2026-08-01; applying it is a human action by design.

**What it does.**
1. Adds `scripts/check_test_hygiene.sh` to `FROZEN_EXCEPTIONS`, with a dated reason, so the tolerance
   is recorded rather than silent.
2. Runs the hygiene check at the top of the existing `--tests-only` branch.

**Why it hangs off `--tests-only` rather than a new `--hygiene-only` mode.** An earlier draft added a
separate mode. That would have been dead code: `.claude/settings.json` invokes the guard **only** as
`--frozen-only` and `--tests-only`, so a new mode would never fire, and wiring one would additionally
require editing `.claude/settings.json` — also frozen and also denied. Folding it into `--tests-only`
makes the patch self-sufficient, and the check runs on exactly the events that already run pytest.

**Known limitation, accepted knowingly.** `$CHANGED` strips the git status column, so a *renamed* test
file arrives as `old -> new` and escapes the `^tests/` filter. Renames of test files are rare; if that
changes, parse `git status --porcelain -z`.

---

## `check_english_explicit_paths.patch` (2026-09-15)

**Apply with:** `git apply docs/patches/check_english_explicit_paths.patch` from the repository root.
Derived against `scripts/check_english.sh` as of `646320f`. If that file has changed since, re-derive
rather than force. **Verified before hand-off:** `git apply --check` exit 0, `bash -n` exit 0 on the
patched candidate, and the nine behavioural cases below executed against both the current and the
patched script.

**Why it is a patch and not a commit.** Same reason as the entries above: `scripts/**` is denied at
permission level, and CLAUDE.md rule 1 records why a session must not be able to unfreeze itself.
⚠️ **This patch was produced after the permission layer refused a `cp` of the script into a scratch
directory — correctly, because that is one of the routes CLAUDE.md names (`cp`, heredoc, `sed -i`,
`tee`, `patch`). The refusal was not worked around.** The candidate was built by reading the file and
writing the modified copy outside the repository, which is a read of the frozen path and a write to
the scratchpad.

**The defect it fixes — a control that reported clean on files it never opened.** Measured 2026-09-15
while reviewing P7.2b. `check_english.sh <paths>` filtered its arguments through `git ls-files`, which
returns **nothing** for a path git does not already track — *every new file, before `git add`* — and
also for a tracked path under a hidden directory such as `.claude/`. The result then met
`[ "${#FILES[@]}" -eq 0 ] && exit 0`, so **nothing to check** and **everything is clean** shared an exit
code. The explicit-path mode is the one the PostToolUse hook and every session use, so the English rule
has been effectively unenforced on new files. It went unnoticed because the *working* no-argument sweep
exits 1 on four benign self-referential hits, which pushed everyone onto the broken mode.
**Consequence for the record: a per-file `exit 0` quoted in any past Return Packet is not evidence.**

**What it does.**
1. Explicit paths are taken **as given** and filtered by shell globs (`EXCLUDE_GLOBS`, kept adjacent to
   the git-pathspec `EXCLUDES` so the two are edited together) instead of by `git ls-files`.
2. A requested path that does not exist is **BLOCKED, exit 1** — never a silent pass.
3. If every given path is deliberately excluded, it exits 0 but prints a loud `NOTE`. *Argued exception
   to PROJECT_PLAN section 7's new rule:* a deliberate exclusion is a decision recorded in the script,
   unlike an untracked path, which is an accident — and the failure being fixed was the **silence**, not
   the exit code. Overrule it to exit 1 if you prefer the stricter reading.
4. Being outside a repository no longer `exit 0`s: absolute paths still work, relative ones are reported
   as missing.
5. The **no-argument sweep is untouched** — verified byte-identical output and exit 1.

**Falsified, not asserted** (current versus patched, nine cases):

| case | current | patched |
|---|---|---|
| tracked file under `.claude/`, contains Polish | 0 (wrong) | **1** |
| untracked new file, contains Polish | 0 (wrong) | **1** |
| tracked clean file | 0 | 0 |
| path that does not exist | 0 (wrong) | **1** |
| absolute path to a violating file | 0 (wrong) | **1** |
| only excluded paths given | 0 silently | 0 with a loud NOTE |
| several clean tracked files | 0 | 0 |
| no-argument sweep | 1 | 1, output byte-identical |
| absolute path from another directory | 0 after a git fatal | **1** |

**Not fixed here, deliberately.** The four benign hits the sweep reports (three are the same comment
about the o-acute false positive, one is illustrative quoted Polish in a frozen agent definition) still
make the repo-wide sweep exit 1. That is the script's own TODO and a separate decision: suppressing them
means either widening `ALLOWED_NAMES` beyond proper nouns or excluding files, and both risk masking a
real hit. Until it is settled, **the sweep's exit code must be read together with its known lines.** ⚠️ **CORRECTED 2026-09-16, after this patch file put CI red:** the first version of this file was a 3-line-context diff and so carried the o-acute comment as CONTEXT, exactly as the two `claude_guard` patches above do — and the sentence that stood here said *expect the sweep to list five paths, not four*, i.e. the coordinator PREDICTED the new finding and did not connect it to CI's English guard, whose baseline enumerates findings per file and fails the job on any NEW file (`findings=5 baseline=4`, run `35089415843`). The file was regenerated with ONE line of context (`diff -U1`), which drops the comment and carries no Polish character; it is the same change, proved both ways — it reverse-applies cleanly against `main` where the fix is committed (`9edfc33`), and forward-applied in a throwaway worktree at `878f72d` it yields a `scripts/check_english.sh` byte-identical to `main`'s. The sweep lists four paths again and the guard gate, run locally against the baseline before pushing, reports no regression.

---

## `ci_gate_ceiling_178_p7_2b.patch` — the skip ceiling moves 158 → 178, and all +20 is P7.2b's own gated tests

**Apply with**, from the repository root, then commit (the author's commit by role, as for every ceiling move):
```
git apply docs/patches/ci_gate_ceiling_178_p7_2b.patch
.venv/bin/pytest tests/test_ci_gate.py -q        # 34 passed
git add .github/ci/ci_baseline.json tests/test_ci_gate.py
git commit -F - <<'MSG'
ci(ceiling): 158 -> 178 observed after P7.2b's merge and Amendment G -- +16 checkpoint-gated, +4 on P7.2a's parity configuration; classified run 35085215720 at 164b519, the merge's own run cancelled by the plan push but its one uploaded leg identical element-by-element
MSG
```
Derived against `.github/ci/ci_baseline.json` and `tests/test_ci_gate.py` as of `164b519`. If either has
changed since, re-derive with the generator rather than force (it is deterministic:
`output/p7_2b_runs/ci_ceiling_178_gen.py`, copied from the coordinator's scratch).

**Registered protocol (`re_measure_required_at.what_to_do`):** merge, let the gate go red, classify
`junit.xml`'s skip messages on both legs, commit the observed value with its breakdown. No pre-bump, no
slack. **This move waited one extra round on purpose:** the first red after the merge (`b2ee947`, run
`35023303912`, 177 skipped) carried a **real failure** — one P7.2b test with no `skipif` — not only the
ceiling breach, and gating it converts that failure into a skip. Committing 177 would have put CI red
again for a number we produced. Amendment G landed first (`8c83fd8`); this is the value measured after it.

**Observed, both legs identical (run `35085215720`, `main` at `164b519`):** 1876 passed, **178 skipped**,
0 failures, 0 errors, 2054 collected; 42 distinct skip messages, identical element-by-element across the
two legs. **Substitution, recorded rather than glossed:** Amendment G's merge commit is `8c83fd8` and its
own run `35085007974` was **cancelled** about seven minutes in by the coordinator's plan-commit push — the
third time this shape has occurred — but its ubuntu-22.04 leg had uploaded a complete `junit.xml`
(2054 / 178 / 0) and its multiset is identical to the classified run's; the 24.04 leg uploaded no
artifact, checked against the API. `git diff --stat 8c83fd8 164b519` is 5 lines in two docs files.

**Delta against the 158 run (`34782398552`), diffed DIRECTLY from both runs' artifacts:** 3 new
messages carrying all 20 new skips, 0 removed, 0 count changes.

| message | count | family |
|---|---|---|
| `P4 checkpoints are not in this tree` | 16 | `corpus_or_checkpoint` 76 → 92 |
| `P7.2a's parity configuration for draw 5 is not present` | 3 | `campaign_output` 20 → 24 |
| `P7.2a's parity configuration for probe-band draw 201 is not present` | 1 | `campaign_output` |

`cityflow` 42, `sumo_traci` 14, `matplotlib` 2, `grid4x4_resco_candidates` 4 unchanged;
92 + 42 + 14 + 2 + 24 + 4 = 178. The delta closes against P7.2b's own count: collected +66 = the 65 tests
of the merge plus Amendment G's one CI-runnable test; passed +46, skipped +20 — to the number what
Amendment G's route-A reproduction produced locally with the two hardcoded roots hidden.

**Verified before hand-off, not asserted:** `git apply --check` exit 0; in a throwaway worktree with the
patch applied, `tests/test_ci_gate.py` **34 passed**; the gate script on **both** completed legs against
the new baseline prints `OK pytest-gate: passed=1876 skipped=178 failed=0 errors=0, skip ceiling 178
(0 to spare)`; and as a control the same leg against the *current* 158 baseline prints `FAIL … 178 tests
skipped against a declared ceiling of 158`, exit 1.

**What else the patch does:** demotes the 158 `measured` block into the `superseded` chain with
`why_it_was_wrong: "Not wrong -- EXPIRED AS DECLARED …"`; refreshes `ceiling_if_cityflow_were_built`
to 136 and the prose *"42 of these 178 … 178 to 136"*; renames `re_measure_required_at.event` to P7.3's
merge; and records, under `predicted_delta`, that the coordinator wrote 178 into the plan beforehand as
arithmetic on an observed 177 — not a prediction from a brief, which is the thing that has been wrong
five times — so the distinction survives.
