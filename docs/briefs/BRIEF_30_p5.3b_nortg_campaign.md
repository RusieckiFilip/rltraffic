# BRIEF_30 — P5.3b: does removing the return prompt cost anything, and is the cost data-dependent?

**Task id:** `P5.3b` · **Branch:** `task/p5.3b-nortg-campaign` · **Issued:** 2026-08-26
**Mode:** Explore → Plan → Code → Commit, with human gates. **Start in plan mode.**
**Compute:** GPU training, **15 new cells only**. The `dt` column is reused, never retrained (except
once, as a control — §4.1).

> **Read `BRIEF_28` + amendments A and B, `docs/plans/p5.3a.md`, `docs/reviews/P5.3a.md` and
> `docs/returns/P5.3a.md` first.** Where this brief disagrees with any of them it wins; where it
> disagrees with the **repo**, the repo wins and you flag it (CLAUDE.md §2).

---

## 1. Why this task exists, and how its question changed

P5.3a answered the question P5.3 was created to ask, and the answer moved the goalposts:

- **The prompt is a strong lever on the POLICY.** Paired over P4.3's own per-episode data,
  n = 500: **499 of 500 episodes move** between target `0` and target `−13000`, mean absolute
  **2.1281 ATT**, max **11.58**.
- **And a weak lever on MEAN QUALITY:** the mean of those movements is **+0.9026**, in the registered
  direction.
- **So `A9` has an object** — the model did learn to condition on the token. ⚠️ **What is unproven is
  that the conditioning is worth anything.**

**That is this task's question, and it is sharper than the one in §6:** *not "is there a knob?" — there
is — but **"does removing the knob cost anything, and does the cost track the data's return spread?"***

§1b's **C4** supplies the hypothesis: conditioning is identifiable only where the corpus carries
between-episode return spread. **P5.3a measured that spread for the first time (row B) and it varies
57× across our eight tiers**, so the hypothesis is finally testable.

---

## 2. The tier set — fixed by a rule registered BEFORE the numbers existed

`BRIEF_28` §9 registered: *"(i) `mappo1000`, the headline tier; (ii) the tier with the largest measured
between-episode scaled-RTG sd; (iii) the tier with the smallest."* **Row B resolves it:**

| tier | row B (pooled) | `dt` ATT | role |
|---|---|---|---|
| **`mix50`** | **0.2725** | 107.7026 | **widest spread** — where the prompt should matter most |
| `mappo1000` | 0.0391 | 104.9558 | the declared headline tier; A6's δ and P4.3's sweep live here |
| **`random`** | **0.0048** | 420.3764 | **narrowest** — and P5.3a measured its DT at **0 of 7200 flips** |

**3 tiers × 5 seeds = 15 new `dt_nortg` cells.** Nothing else trains.

⚠️ **The ATT scales differ by 4×** (105 · 108 · 420), which is half the reason §5 forbids a single
equivalence threshold.

---

## 3. Registered predictions — written before any P5.3b number exists

| # | Prediction | Falsifiable how |
|---|---|---|
| **Q1** | **ORDERING, by endpoints:** the paired `\|dt − dt_nortg\|` is **largest on `mix50`** and **smallest on `random`**. | Either endpoint out of place falsifies it. **Endpoints, never a trend** — §1b's R3 was falsified on exactly a monotonicity claim, and the standing instruction is to register endpoints. |
| **Q2** | ⭐ **`random` is a NULL CONTROL, predicted from an independent instrument.** P5.3a measured `random`'s conditioned DT at `flip_rate = 0.000000`, 0 of 7200, on every intervention. If the token carries nothing there, training without it should cost nothing there: **`dt − dt_nortg` on `random` has a CI containing 0.** | A large, CI-excluding difference on `random` **indicts the wiring before it indicts the science** — same direction of inference as A8's `fixedtime` prediction in P5.3a, and registered now so it cannot be reversed later. |
| **Q3** | **Arm validity, mechanical:** every `dt_nortg` checkpoint shows `flip_rate` **exactly 0.0** under P5.3a's probe on all 12 interventions, and carries `rtg_mode == "zero"` in its checkpoint config. | Anything non-zero means `rtg_mode` did not reach the training path. **This is a gate, not a result.** |

⚠️ **Q2 and Q3 are different claims and must not be conflated.** Q3 says the *trained* model ignores a
token it was never shown. Q2 says training without the token cost nothing *on that tier*.

---

## 4. What to build

### 4.1 ⭐ THE LOAD-BEARING TEST — threading `rtg_mode` through `train_dt` must not move the conditioned path

`offline/dt_gate.py:749-754` constructs `DTConfig` with four arguments and **has no `rtg_mode`
parameter** (MJ-5). Adding one is this task's first commit — and it touches the function that trained
**every DT number in the paper**.

> **Retrain ONE committed `dt` cell through the modified `train_dt` and assert its canonical
> `state_dict` digest equals the committed value** in `p4_6_training.json`
> (`canonical_digest_of`, `offline/method_tier_grid.py:1180`). `mappo500` or `maxpressure` seed 101 —
> **not `random`** (§4.4). ~3 minutes at 40,000 steps.
>
> **Then mutate, both directions, and paste both failures:**
> 1. force `rtg_mode="zero"` inside `train_dt` → **must fail**;
> 2. make the new parameter default to `"zero"` → **must fail**.
>
> **If either leaves the suite green, the task is `BLOCKED`.** Do not weaken the test.

🚨 **Why this test and not P5.3a's:** `docs/reviews/P5.3a.md` **MJ-3** measured that the ATT-identity
test is **structurally blind to every config field that only matters at training time** — forcing
`dropout=0.9` passed it, because dropout is inert under `eval()`. **P5.3b trains, so that blind spot
is now live, and a digest over weights is the instrument that sees it.**

### 4.2 The campaign

Reuse P4.6/P4.7's protocol **read from the committed declarations, not restated**: seeds
`[101,202,303,404,505]`, `declared_gradient_steps = 40000`, held-out draws **1000–1099**, the same
`env_settings`, the same corpus dirs, the same `target_rtg`/`rtg_scale` per tier.
**Gate 1 before consuming anything**: canonical digest **and** a `SHA256SUMS_p4_6.txt` /
`SHA256SUMS_p4_7.txt` check **at consumption** (`BRIEF_27` B3(a)).

⚠️ **`mix50`'s `NormalizationStats` is fitted on the UNION of all three mixtures' directories** —
identical for `mix33`/`mix50`/`mix67`, count 216000, std 13155.3172 (P5.3a §8.4). **That is how P4.7
trained it, so reuse is consistent — but the artifact must say so**, or a reader will take the summary
for a property of `mix50`.

### 4.3 Statistics — and the P5.2 lesson is binding

Paired per-draw over **shared draw ids** (A5), per tier: mean difference, 95 % CI, **Wilcoxon**,
**rank-biserial**, wins/losses/ties, `n_shared_draws`, and the **per-seed reversal count**.

🚨 **`docs/reviews/P5.2.md` MJ-4: that packet's docstring claimed the protocol was reused from
`dt_gate._paired`, `wilcoxon_signed_rank` and `offline_baselines.paired_comparison` — none of which
was imported or called.** **Import them and call them.** A test must fail if the qualifier is removed.

### 4.4 Arm validity via P5.3a's probe

Run `offline/rtg_ablation.py probe` over the 15 new checkpoints — **141 s for 40 cells in P5.3a, so
this is about a minute.** Q3 above is the acceptance criterion. ⚠️ **`random` may not be the §4.1
control cell**, because its conditioned DT is already RTG-inert, so a control there would pass whether
or not `rtg_mode` reached the trainer.

---

## 5. ⛔ NO EQUIVALENCE THRESHOLD. Report the measurement and the ordering

`PREREGISTRATION` A7 withdrew the per-tier δ rule on 2026-08-25 because it spanned **eleven orders of
magnitude** and could not return one of its answers on part of its domain. **Do not reinvent it.**

> **Registered: P5.3b issues NO equivalence verdict.** It reports the paired difference, its CI, the
> per-seed reversals, and the tier's own `dt` ATT beside it so the reader can scale it. **Q1's ordering
> and Q2's null control are threshold-free and are the registered claims.**
> ⚠️ **A CI containing 0 is a failure to reject, never a demonstration of equivalence** — A6's own
> words. Write it that way.

---

## 6. Scope fence

1. ⛔ **No tier beyond the three.** The rule chose them before the numbers existed; adding a fourth
   after seeing results is what P5.3a correctly refused.
2. ⛔ **No retraining of the `dt` column** except §4.1's single control cell.
3. ⛔ **No second ablation mode.** `"zero"` only; `rtg_shuffled` remains unregistered.
4. ⛔ **No context-length K** — P5.3c.
5. ⛔ **No editing `offline/method_tier_grid.py`'s `METHODS`** (`:1701` — comparison enumeration).
6. ⛔ **Never write into or delete from** any existing `output/<campaign>/` directory. `output/p5_3b/`
   is yours. **Write `output/SHA256SUMS_p5_3b.txt` at the end** — every campaign has one, and
   `DEFERRED` 56 is what happens when one does not.
7. ⛔ **Not fixing** P5.3a's carried items (cell 2's non-discrimination, MJ-1's undemonstrated chunked
   case, M2/M5/M6) or `DEFERRED` 56.
8. ⚠️ **`docs/data/p4_dt_config.json`'s `architecture` block is literally `payload["config"]`
   (`dt_gate.py:1007`), so your checkpoints write a 9-key config where P4's has 8.** Expected; say so.

---

## 7. Gates, in order

| gate | what it proves | stops the task? |
|---|---|---|
| **0** | **A one-seed timing probe before any campaign commitment** (Amendment R2 — no armchair costing; the coordinator's own "≈52 h" is the cautionary case) | no, but no commitment without it |
| **1** | Reused `dt` checkpoints verified by canonical digest **and** manifest, at consumption | **yes** |
| **2** | §4.1: one committed `dt` cell retrains to its committed digest through the modified `train_dt`, **and both mutations fail** | **yes** |
| **3** | Q3: every `dt_nortg` cell probes to `flip_rate` exactly 0 and carries `rtg_mode == "zero"` | **yes** |

---

## 8. Definition of Done

- [ ] 15 cells trained, evaluated on draws 1000–1099, artifact `docs/data/p5_3b_nortg.json`
- [ ] **Gate 2 passed and BOTH mutations failed, with both failures pasted in full**
- [ ] Gate 3 mechanical on all 15
- [ ] Q1 and Q2 scored against §3 as registered; **no equivalence verdict anywhere**
- [ ] Paired statistics actually imported and called, with a test that fails if the qualifier is removed
- [ ] `output/SHA256SUMS_p5_3b.txt` written; every reused manifest re-verified after the run
- [ ] Whole suite green, pinned, real tail pasted; guards **16** / English / frozen
- [ ] Committed on the task branch; Return Packet at `docs/returns/P5.3b.md`
- [ ] **§6's `P5.3b` checkbox ticked in the merge commit**; **AI-assistance record** written as you go
- [ ] Timing measured and reported per cell — P5.3c needs it

**Review:** critical-path. `contract-reviewer` before merge, mutation evidence not reading.

---

## 9. What the next reader must not be allowed to believe

State these in the packet, plainly:

1. **A null on a tier is not "the prompt is useless".** It is *"removing the prompt did not change mean
   held-out ATT on this corpus, at this budget, at 5 seeds"*.
2. **The three tiers differ in more than return spread** — composition, state coverage and data quality
   move together. **Row B is an axis we can measure, not one we can isolate**, and Q1's ordering is
   therefore consistent-with rather than evidence-for C4.
3. **`random`'s DT is 4× worse in ATT than the other two** (420.38 against ~105). A difference measured
   there is not comparable in magnitude to one measured on `mappo1000`.
