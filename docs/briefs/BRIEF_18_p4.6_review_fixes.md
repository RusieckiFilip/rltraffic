# BRIEF 18 — P4.6 review fixes (mandated round before merge)

**Mode:** Claude Code · **Branch:** `task/p4.6-method-tier-grid` (continue on it) · **Worktree:** `/home/filip/rltraffic-p46`
**Review:** `docs/reviews/P4.6.md`, **PASS-WITH-NOTES**, pinned at `0e1b6eb`
**Read first, from disk:** the review, then `docs/briefs/BRIEF_17_p4.6_method_tier_grid.md` **§11 and §12**.

⚠️ **Absolute paths in every command.** ⚠️ **Pin threads on every job.** ⚠️ **Never write "MADT"** (`CONTRACTS` C9).

---

## 0. The verdict, and what it does NOT mean

**No blockers. Every reported number reproduced under independent recomputation** — 25/25 cell means to
**5.68e-14** by a route that did not use `$.episodes`, all **90** paired comparisons to **7.1e-15**,
A5 satisfied on every reported comparison, zero frozen files touched, provenance complete and
reachable, 125/125 chain of custody.

**So no result changes and no number moves.** ⚠️ **If any fix below moves a reported number, STOP and
report it — that is a finding, not a fix.**

What the round repairs: **one wrong table that is materialised on disk**, **a narrative claim that did
not survive checking**, **a diagnostic that is almost entirely circular**, and **four quantities the
paper will quote that no test protects.**

---

## 1. F1 — THE HIGHEST-VALUE FINDING, AND IT REMOVES A READING THAT FLATTERED US

The packet argues P1 failed for a reason that changes its meaning: *the hypothesis was right and the
operationalisation was wrong*, evidenced by the declared continuous companion moving as P1's prose
said (**τ_b = +0.80**).

**That reading does not survive, and it was checked precisely because it was the favourable one.**
The reviewer confirmed the companion **was declared before the run** (`git show 622cf2e:docs/plans/p4.6.md`
§4.1, 18:47; first gradient step ~19:59) — **so this is NOT post-hoc selection, and the packet is not
accused of one.** But:

1. **It is one of four natural continuous readings of "BC's rank worsens", and they disagree in SIGN:**
   `bc − min(other)` absolute **+0.80** (declared) · normalised by tier behaviour ATT **+0.40** ·
   `bc − max(other)` absolute **−0.60** · BC's position within the tier's method spread **−0.53**.
2. **The mechanism is wrong.** The gap grows because **IQL pulls away**, not because BC deteriorates.
   ⚠️ **CORRECTED 2026-08-14 — the bound in this line was FALSE and the error was the coordinator's**,
   relayed from the review without measuring it. It read *"BC tracks its own behaviour policy within
   1.5 ATT on every tier"*. **Measured by the coordinator from `p4_6_grid.json`: it fails on TWO of the
   five tiers, not one.** `mappo1000` **−0.4186** · `mappo500` **−0.8136** · `random` **−0.9296** ·
   `maxpressure` **−1.5019** (over by 0.0019) · `fixedtime` **+3.7616**. **The implementer caught this
   and named `fixedtime`; the marginal `maxpressure` breach is a second instance it did not name, and
   both are recorded here rather than the more convenient one.** **Correct statement: BC lands within
   1.51 ATT of its behaviour policy on four tiers, and +3.7616 on `fixedtime`.** **The mechanism claim
   survives — BC does not deteriorate; IQL pulls away — but the BOUND does not, and a bound is exactly
   the kind of sentence that ends up in a figure caption.**
3. The rank rule and the companion can disagree **by construction**: the rank counts position among
   *all* others, the gap measures distance to the *best* other.

> **REQUIRED.** Rewrite the P1 section to say: **P1 FAILED.** Full stop, no rescue. Then report
> separately, and explicitly labelled **UNREGISTERED AND OBSERVATIONAL**, the finding that actually
> holds: **"IQL's advantage over BC grows as data quality falls."** State that it is a different claim
> from P1, that it was not registered, and that the four continuous readings disagree in sign — **that
> disagreement is itself the reportable fact**, because it shows the continuous companion could not
> have adjudicated P1 in either direction.

**This is the round's most important change and it makes the paper weaker and truer.**

## 2. F2 — a wrong table, materialised in the secured evidence

**`offline/method_tier_grid.py:2589`** prints the report summary with a `behaviour` column read from
`BEHAVIOUR_ATT` (**training draws 1–200**) while all four method columns are held-out (**1000–1099**).
Reading across that row is the **A5-void** comparison. It is on disk at
`output/p4_6/logs/report.log:1-5` and `logs/phase1.log:28-32`, **both inside `SHA256SUMS_p4_6.txt`**.

Measured by the coordinator, independently: **max substitution error 6.3652 ATT**, and **1 of 20
statements flips sign** — `bc@random` reads **+5.4356 (worse)** against the truth **−0.9296 (better)**,
i.e. it reverses whether BC beat the policy that generated its data.

> **REQUIRED.** (a) Fix the print to use `behaviour_cells`, and **label the column with its draw
> range**. (b) **Regenerate `report.log` and `phase1.log`** and tell the coordinator, who re-secures
> `output/p4_6/` and rewrites `SHA256SUMS_p4_6.txt` — **do not edit the secured copy in the main
> tree yourself.** (c) Rename `behaviour_att` → **`tier_label_att_training_draws`** everywhere,
> including `p4_6_declaration.json`. (d) **Give the packet's §0.0 table its draw ranges in the column
> headers** — its row labels are training-draw values on a table of held-out means.

⚠️ **The rename is necessary and NOT sufficient**, which the review proves: the offending site is a
*use*, not a name, and the packet's table is a *layout*. **Enumerate every consumption site after the
fix and classify each**, as the review did (30 sites: 2 WRONG, 7 AMBIGUOUS, 21 CORRECT).

## 3. F3 — "partly circular" is "almost entirely circular", and the artifact says nothing

On the `maxpressure` tier, ρ(stream return, own-tier `att_horizon`) = **−0.9967**, against
**−0.13 / −0.23 / −0.03 / −0.01** on the other four. **19/20 is what the circularity alone produces**,
and the hypergeometric null (expected 2.0) is **inapplicable** because the two sets are not
independent. `grep -c circular` on all three artifacts returns **0, 0, 0** — a consumer sees
`p = 2.23e-24`, the smallest p-value in the task, unflagged.

> **REQUIRED.** **Withdraw check B on `maxpressure`** (or re-reference it to a difficulty probe that is
> not that tier's own behaviour policy) and **carry the reason as a FIELD in `p4_6_grid.json`**, not
> only in prose. Report the measured ρ table above beside it. Correct "partly" to "almost entirely".

## 4. F4 — four quantities the paper will quote, none protected. FIX THE CLASS HERE.

The packet's "22 mutations, 22 killed" is accurate for the 22 it chose. The reviewer ran four more and
**all four survived with `763 passed, 3 skipped`:**

| mutation | site | effect when applied |
|---|---|---|
| `min` → `max` in the P1 gap | `method_tier_grid.py:1461` | **headline τ_b inverts +0.80 → −0.60** |
| `behaviour_cells[tier]` → `BEHAVIOUR_ATT[tier]` | `:1711` | artifact reports 422.5188 for `random`, internally inconsistent |
| `ddof=1` → `ddof=0` | `:884-886` | no P3 decision flips today; `random` moves to **0.44 vehicles** from flipping |
| disable the incomplete-tier refusal | `:1715-1724` | plan T14's second clause is unenforced |

⚠️ **This is `DEFERRED` 33/42/44's family recurring INSIDE the module commissioned to close it.**
`DEFERRED` 44 already ruled: *"if the guard family appears a FOURTH time, stop queueing it and fix the
class."* **It has. Fix all four here and queue none.** Each fix is a test that **fails when the
protection is removed** — paste every failure.

## 5. Smaller, all required

- **`git diff --stat`** in the packet is **9 files / 15,574 insertions**; the real one at `0e1b6eb` is
  **13 files / 133,725** — the paste omits `p4_6_grid.json` (112,859 lines), *the file holding every
  reported number*. CLAUDE.md §7 mandates the real stat. **The conclusion (zero frozen) is correct and
  independently confirmed; only the paste is wrong.**
- **`dt@fixedtime`'s `rank_biserial: -1.0`** with `ties: 98, n_used: 2` needs a **caveat field in the
  JSON**. 45 other pairs carry |r| = 1.0 as genuine effects, so a figure script sorting by |r| cannot
  tell them apart. Prose in §3.7 does not reach a figure script.
- **TH006 count**: the packet says **17**; the reviewer counts **16** and the packet's own per-file
  enumeration sums to 16. Fix the headline. *(The coordinator copied 17 into `DEFERRED` 45 without
  counting — corrected there too.)*
- **"Margins an order of magnitude larger"** is **5.6×** at its weakest (`fixedtime` 1.73). Restate.
- **§3.4 is stale** and contradicts §0.0/§3.6 in the same document (retained from `92811a4`). Remove
  or date-stamp it.
- **The k=6 validation** (§3.1) has almost no discriminating power — the held-out cell's own CI is
  **±0.7442**, so a 0.018 agreement discriminates nothing. **The plan-sha256 assertion is the real
  evidence; say so and demote the ATT line.**

## 6. What the reviewer could NOT verify — disclose, do not re-run

Record these in the packet as **unverified by the independent reviewer**, plainly: the **A2 regression
gate** (the "18,201 leaves, 0 changed" claim — only byte-identity of the committed file was checked);
**Gate G's 100 re-rolled cells** (only the 4 re-used means were confirmed against P4.4); **that 40,000
steps were actually taken** (only recorded provenance, 4 spot-checked); and the **`random` behaviour
arm's seeding equivalence to collection**. **Do not re-run them to close the gap — disclosure is the
deliverable here.**

## 7. Definition of Done

- [ ] F1 rewritten: P1 FAILED, no rescue; the real finding reported as unregistered and observational
- [ ] F2 fixed, artifacts regenerated, every consumption site re-enumerated and classified
- [ ] F3 withdrawn or re-referenced, with the reason **in the artifact**
- [ ] F4: four tests, each proven to fail when its protection is removed, **every failure pasted**
- [ ] §5's six corrections applied · §6's four non-verifications disclosed
- [ ] **No reported number moved** — prove it by regenerating and diffing the artifacts
- [ ] Full suite green, tail pasted, pinned state stated; all three guards reported with their real
      exit codes, measured **without a pipeline** (`$?` after a pipe reads the last command's status)
- [ ] Packet updated · §6's checkbox still unticked; it is the coordinator's, in the merge commit
