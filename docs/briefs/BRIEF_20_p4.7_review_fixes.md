# BRIEF 20 — P4.7 review fixes (mandated round before merge)

**Mode:** Claude Code · **Branch:** `task/p4.7-mixture-tiers` (continue on it) · **Worktree:** `/home/filip/rltraffic-p47`
**Review:** `docs/reviews/P4.7.md`, **PASS-WITH-NOTES**, pinned at `2021cc7` — **on disk before this round, unlike P4.6's.**
**Read the review first, in full.** ⚠️ Absolute paths · pin threads · count from **full output, never a tail**.

---

## 0. What this round is and is not

**No blockers. No number is wrong.** The reviewer recomputed all 12 cell means (worst **2.842e-14**),
Q2's three advantages (identical to 2.8e-14), the constructed reference on five sub-checks, and
verified Gate P1's *actual* result by a **wider** route than the gate itself (1,896 leaves, 0
mismatches). **A8(a) holds with timestamps: the rule and its scorer both predate every mixture number.**

> ⚠️ **If any fix moves a reported number, STOP and report it as a finding.**

What this round repairs: **one overstated strength claim**, **one registered disclosure that was not
delivered while an artifact claimed it was**, **two report-path mutations that survive the whole suite**,
and a set of prose and artifact-labelling defects.

## 1. F1 — Q2 IS THE TASK'S ONLY LOAD-BEARING EVIDENCE AND "DECISIVELY" IS NOT SUPPORTED

Q2 HELD is a **conjunction** and its two halves have very different strength:

| conjunct | evidence | status |
|---|---|---|
| all three advantages **positive** | 15/15 per-seed positive, min **+9.63** | **robust** |
| `mix50 > mix67` | **0/5** seeds reversed, t = **4.70** | **robust** |
| `mix33 > mix50` | **2/5 seeds REVERSED**, gap 27.41, SE 24.06, **t = 1.14** | **does not resolve** |

**And the CIs the packet prints cannot speak to this**: they are draw-level with the 5 seeds averaged
into each unit (`offline/offline_baselines.py:2204`). The convention is inherited and legitimate — **its
blind spot is an order of magnitude larger here than anywhere in phase 1**: between-seed sd of the `bc`
cell is **18.1 / 57.8 / 29.8** on the mixtures against **≤ 6.0** everywhere in phase 1.

> **REQUIRED.** (a) **Report the per-seed advantage table** in the packet and in the artifact.
> (b) **Delete "decisively."** Q2's verdict stands — the declared rule was frozen before the data and it
> HELD — but the packet must say **which conjunct is robust and which does not resolve at this power**.
> (c) State plainly that the companion intervals **do not address the seed dimension**, and give the
> between-seed sd beside them. (d) **Do not change the rule or the verdict.** The fix is to the
> strength claim, not the scoring.

⚠️ **Also required, from N2:** say what Q2's evidence is **about**. %BC spans **1.10 ATT** across the
three mixtures against BC's **115.13**, so the ordering is ~99 % a measurement of **BC's dilution
response**, not of the filter. Q2 stays falsifiable and confirmed; it is not evidence about the filter.

## 2. F2 — THE REGISTERED DISCLOSURE THAT WAS NOT DELIVERED, AND AN ARTIFACT THAT SAYS IT WAS

`docs/plans/p4.7.md:295-298` registered: *the per-tier count of draws appearing in **both** components
is reported, and if it is large the volume check is reported as **weakened** rather than as null.*

**Measured by the reviewer: 39 / 48 / 44** — and mix50's 200 streams collapse to **152 distinct draw
ids**, with `volume` running on `n_kept=20, n_other=132`. **The count is in no artifact, "weakened"
appears nowhere, and `docs/returns/P4.7.md:225` calls the demand half "a real null".**
`offline/mixture_tiers.py:764` asserts the count *is* reported — **false as written**. Its only test
(`tests/test_p4_7_predictions.py:391`) asserts a string contains `"set"` and **cannot fail**.

> **REQUIRED.** Emit the per-tier count in the artifact; apply the registered **weakened** label per
> the rule as declared; correct "a real null" wherever it appears; fix the false docstring; and
> **replace the tautological test with one that fails when the count is absent.**

⚠️ **This is a pre-registered disclosure, so it is not discretionary** — and it is exactly the shape
of A5's mandatory co-report, which this project has already had to fix once.

## 3. F3 — TWO REPORT-PATH MUTATIONS SURVIVE THE FULL SUITE. SIXTH SIGHTING; FIX THE CLASS.

`mixture_grid_artifact`, `_p4_6_prediction_sidecar`, `_load_report_inputs`, `_comparison_objects` and
the `report` subcommand are called by **no test**.

| mutation | effect | suite |
|---|---|---|
| `assert_phase1_reproduces(payload, payload)` | **Gate P1 compares the artifact with itself and can never fail** | 844 passed |
| sidecar reads `payload["predictions"]` not `committed[...]` | P4.6's `NOT SCORABLE` silently becomes `FAILED` — **the exact rescue RULING 2 forbids** | 844 passed |

**`DEFERRED` 33/42/44 said stop queueing this family at the fourth sighting. This is the sixth.**
> **REQUIRED: test the report-assembly WIRING, not only its parts.** Each fix is a test that **fails
> when the protection is removed — paste every failure.**

## 4. Smaller, all required

- **N1** — Q2's tie clause checks **adjacent** pairs; the plan and the artifact's `rule` string say
  *"any two"*. No outcome moves and the direction is conservative, but **the artifact mis-describes its
  own implementation** — align one to the other and say which you changed.
- **N5** — packet headline "within **4.9 / 3.5 / 2.0** ATT of the leader" is **3.7856 / 3.4673 /
  2.0015**; **4.9601** is the DT's gap to the best cell *anywhere*. **Two definitions mixed in one
  triple** — mix33 overstated by 31 %.
- **N6** — the self-review says *"it has not been launched yet"*, stale from the PARTIAL packet and
  contradicted by §1 and `phase2.log`. **A mandatory checklist line that is false on its face.**
- **N7 / N8** — give `Q1.outcome` a **verdict-level** qualifier (Q3 already has the better pattern in
  the same file), and qualify `Q3.falsifies_r2: false` at its own level — unqualified, it reads as
  *R2 survived a test*, when §1b's binding is that **R2 remains OPEN**.
- **N4** — the promised `docs/data/p4_7_prediction_sidecar.json` does not exist (content is correct,
  inside `p4_7_grid.json`). **List the substitution in §12's deviations.**
- **N9** — `17 → 16` `match=` tokens across `6 → 5` pre-existing files; diff stat `619/210329 →
  622/210332`. ⚠️ **The commit that fixed the stat did not re-measure after its own edit** — same family
  as the truncations. Also move `BRIEF_19` §5.2's seed/draw-block report **into the packet**.
- **N10** — **"MADT" appears once on the branch: `docs/plans/p4.7.md:9`, inside the sentence declaring
  it appears nowhere.** Self-negating, and C9 rule 2 forbids the string in prose.
- **N12** — paste the failures for the remaining mutations, or say plainly that 23 are named by their
  killing test rather than pasted.
- **Theatre** — `tests/test_p4_7_predictions.py:118`'s `assert "threshold" not in ... or "no threshold"
  in ...` is a tautology. Fix or delete with the §7 conditions.

## 5. Carry into the packet as disclosures — do NOT re-run to close them

The reviewer could not verify: `DEFERRED` 43's measured totals; the "29 mutations, 29 killed" claim as
a whole (it ran 12 of its own instead, 10 killed, 2 survived); **Gate D and Gate G as executions**;
and **whether the 12 cells' training is reproducible** (~4 h of GPU — artifacts, manifest, 60 distinct
digests and log/artifact mtime correspondence were verified instead). **Disclosure is the deliverable.**

⚠️ **One reviewer NOTE resolved in the design's favour, and worth keeping:** N13 — the paired
comparisons average 5 seeds into each per-draw unit while the constructed reference draws an
independent composition per seed. Measured: the forbidden expectation would still report CIs
**29.2× / 41.0× / 56.8×** narrower on the `bc_top10` rows. **`BRIEF_19` §3's realisation ruling is real
and material where it matters — record the measurement.**

## 6. Definition of Done

- [ ] F1: per-seed table reported, "decisively" gone, the CI blind spot stated, verdict unchanged
- [ ] F1/N2: what Q2's evidence is *about* stated (BC's dilution response, not the filter)
- [ ] F2: the registered count emitted, "weakened" applied, "a real null" corrected, docstring fixed,
      tautological test replaced by one that fails when the count is absent
- [ ] F3: the report-assembly wiring tested; both surviving mutations killed; **failures pasted**
- [ ] §4's nine items · §5's disclosures
- [ ] **No reported number moved** — prove it by regenerating and diffing the artifacts
- [ ] Suite green, tail pasted, pinned state stated; **all guards read with no arguments, counted from
      full output, each naming its corpus**
- [ ] Packet updated · §6's checkbox still unticked; it is mine, in the merge commit
