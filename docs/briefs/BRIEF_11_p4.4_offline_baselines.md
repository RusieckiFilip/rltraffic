# BRIEF #11 — P4.4: BC, %BC and IQL on the same corpus

**Mode:** Claude Code, on a task branch. **Branch:** `task/p4.4-offline-baselines`
**Issued:** 2026-08-11 by the Master chat. **Base:** `main` (P2.6, P3 and P4 all merged).
**Filter:** this decides what P4's result *means*. Until it lands, the DT's margin is uninterpretable.

---

## 1. Why this before P4.3

P4 passed its gate: `ATT_MADT = 104.9558` against MaxPressure 176.8912 and MAPPO@1000 105.5820,
paired on all 100 held-out draws, winning 100/100 against MaxPressure and MAPPO@500.

⚠️ **ANNOTATED 2026-08-12, after P4.4 was delivered — this section is a dated record and is annotated, not rewritten.** Both figures above are **seed-averages**. MAPPO@1000's five held-out per-seed means are `103.6087 / 103.5286 / 107.7980 / 105.9976 / 106.9773`; **2 of the 5 checkpoints beat the DT**, and `100/100 vs MAPPO@500` is a win over that tier's seed-average, not over its seed 101 (104.6655). The gate passes under either reading (strict threshold `1.05 × 103.5286 = 108.7050`) and **δ below is unchanged**, since its reference is the pooled mixture the DT actually trained on. See `PROJECT_PLAN.md` §8, row 2026-08-12.

**The DT beat the policy whose data it trained on, by 0.6 %.** That is C1's central question — *does
the model exceed its data* — and **we cannot yet say whether sequence modelling earned it.** A
behaviour-cloning model on the same corpus might do the same, in which case the DT's architecture
contributed nothing and the honest paper says so and pivots weight to the ladder and shift findings.
§1 has said since 2026-07-10 that this is non-negotiable: *"if BC matches MADT, sequence modeling
adds nothing — must be tested."*

**P4.3 (RTG calibration) is deliberately queued behind this.** Calibrating a prompt for a mechanism
whose contribution is unmeasured optimises something we cannot attribute. P4.3 also now has a
measured motive waiting for it — the DT sits **outside its training RTG support for 20.8 % of every
episode**, and the declared target is not the best one (106.46 declared vs 102.05 at target 0).

---

## 2. Frozen protocol — identical to P4, and that is the point

**Any deviation makes the comparison unusable.** Everything below is already settled; do not
re-derive it.

- **Evaluation:** all 100 registered held-out draws `1000–1099`, **paired**, every arm on the
  identical draw set (`PREREGISTRATION.md` §8, amendment A5). A comparison not over shared draws is
  **void**.
- **Training data:** the `cf_hz1x1` **`mappo1000` tier** of `datasets_v11/`, the same tier P4 used.
  Draws `1–999` only; the loader raises on held-out draws — do not route around it.
- **Statistics:** paired **Wilcoxon** over the shared draws, plus mean ± 95 % CI. **Effect sizes are
  mandatory beside every p-value** (§8). Report `att_horizon` **and** `vehicle_count` at the horizon
  **unconditionally**, with the draw ids (A5).
- **Seeds:** ≥5 training seeds, the same five P4 used, evaluated on the same 100 draws so **seed and
  draw are crossed, not nested.** (The corpus's own MAPPO tiers are confounded — review D16 — which
  is exactly why reported comparisons use this design.)
- **Leakage (§6):** every reported model is the checkpoint at a **fixed, pre-declared step count**;
  hyperparameters tuned on this scenario then frozen; and **baselines get the same tuning budget as
  the DT did** — that is this task's whole point, so an under-tuned BC is a straw man and must be
  reported as untuned if it is one.
- **Normalisation statistics fitted on the training split only**, recording the draw ids.

## 3. Deliverables — scope fence

**In scope: BC, %BC, IQL.** Out of scope: RTG calibration (P4.3), any spatial layer (P5.1), any new
scenario. The ≤2-source-file limit stands; `agent/` is writable for *new* files only —
`agent/base.py`, `agent/utils/utils.py` and `agent/MAPPOAgent.py` are frozen (verified: there is no
`agent/**` glob).

### 3.1 BC and %BC
Behaviour cloning on the same windows the DT saw. **%BC filters to the top-10 % of trajectories by
return** (§1). Use `offline/dataset.py` — it already yields per-intersection windows with masks; do
not write a second loader.

### 3.2 IQL
Independent per-intersection. ⚠️ **Registered constraint, and getting it wrong hands the DT an
unearned win:** `terminated` is hardcoded `False` and every episode ends by **time-limit
truncation**, so IQL **must bootstrap through the boundary** and must never treat the horizon as
absorbing (Decisions Log 2026-07-26). Treating a timeout as terminal causes systematic value
underestimation near episode end. **A test must pin this.**

### 3.3 The canonical checkpoint digest — `DEFERRED` 29, introduced here
A checkpoint's file hash depends on **both** its filename (`torch.save` names the zip root after the
output file) **and** its provenance block. **No claim of the form "the model reproduces
byte-identically" is testable at file level.** Ship a small helper computing **sha256 over the
`state_dict` tensor bytes in sorted key order** — filename- and provenance-independent — record it
in every checkpoint artifact you emit, and use it for every determinism claim. Keep the file sha256
for what it does prove: **transport integrity**. ~10 lines plus a test that two differently-named
saves of identical weights produce the **same** canonical digest and **different** file hashes.

## 4. Tests — the load-bearing ones

- **The inference path is exercised with real statistics.** P4's review found its entire online path
  unprotected because no test ever constructed the agent with `stats=`; three mutations survived all
  58 tests and one cost **+3.8 ATT**, most of P4's margin. **Assert that what `act()` feeds the model
  equals what training fed it, exactly, for every step of a real episode.** This is not optional.
- **IQL bootstraps through truncation** — mutate to treat the horizon as terminal and show the test
  fails.
- **%BC's filter selects what it claims** — on a fixture where the top-10 % is known by construction.
- **Padded positions cannot contribute to any loss** (`PAD_ACTION = -1`; `ignore_index=-1`).
- **Determinism by canonical digest**, not by file hash (§3.3).

## 5. Definition of Done
- [ ] `docs/plans/p4.4.md` first, with the **declared step count** for each method
- [ ] BC, %BC, IQL + tests; red-first; mutation proofs pasted
- [ ] All arms evaluated on the same 100 held-out draws, paired, with Wilcoxon + effect sizes
- [ ] **The attribution stated plainly**: does BC match the DT, or not, with the paired test
- [ ] Canonical checkpoint digest shipped and used
- [ ] Full `pytest -q` against the 541 baseline; zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P4.4.md`
- [ ] **§6 checkbox ticked in the merge commit**
- [ ] **Independent review before merge** — critical path

## 6. Return Packet — task-specific
1. The paired DT-vs-BC comparison, with effect size, stated as a plain sentence.
2. Whether each baseline was tuned, and with what budget. **An untuned baseline is reported as
   untuned** (§6.3) — as MaxPressure now is, because it has no parameters to tune.
3. Anything in §2 that disagreed with the repo. **The repo wins; say so loudly.**
4. **If BC matches the DT: say so first, before any interpretation.** That is a registered outcome
   (`PREREGISTRATION.md` §10), not a failure, and the paper already knows what it publishes under it.

---

## 7. Rulings on the P4.4 pre-flight (Master chat, 2026-08-11)

**All nine approved as proposed, with three additions. None changes your design; each closes a way
the result could be over-read.**

**Q1 — `load_episode` + the cross-check test. APPROVED, your reading is right.** The brief's "do not
write a second loader" meant *no second reader of the `.npz` format* — the P3 C-3 ruling — and
`load_episode` **is** that single reader, which `dataset.py` itself calls. So there is no second
reader. **I prefer your option to the accessor**, for a reason beyond leaving reviewed code alone:
the cross-check is a genuine **double computation**, and agreement verified is worth more than
agreement assumed by construction. Keep split enforcement and normalisation in
`TrajectoryWindowDataset`, exactly as you propose — the leakage guard must stay in one place.
Show T3 red under the off-by-one.

**Q2 — BC = the DT's stack minus attention. APPROVED, and it is the right control.** A plainer MLP
would confound architecture family with sequence modelling. **ADDITION (1), and it is the most
important thing in this ruling:** DT − BC is a **combined** difference — attention/context **plus**
RTG conditioning **plus** timestep embedding. **State it that way in the packet and never as
"sequence modelling adds X".** Decomposing it is P4.3's (RTG) and P5.3's (`no-RTG`, context-length
K) registered work. Getting this wrong would put an unsupported attribution in the abstract.

**Q3/Q4 — published IQL package, unswept, batch 1280. APPROVED.** §6.3 permits an untuned baseline
**provided it is reported as untuned**, and the DT is equally untuned (published DT-Gym config,
scaled). Report the **provenance of the values (D4RL locomotion) and that the domain differs** —
continuous control vs discrete-phase TSC — so a weak IQL is not read as evidence about IQL. Batch
1280 is well reasoned; note in the packet that IQL's 1280 transitions are **independent** while the
DT's 1280 positions are **correlated within 64 episodes**, so this equalisation slightly favours
IQL — which is the conservative direction for our claim, and worth saying so.

**Q5 — 40,000 steps, no raise. CONFIRMED.** That is the DT's reported budget, which is what §6.3
matches. The DT has already spent its one pre-declared raise; a different budget for baselines would
break the comparison.

**Q6 — re-run the full DT arm. APPROVED, emphatically, and it is Gate A.** This is the same pattern
that earned trust for P8.0 and for the draw-cycling trainer: **prove the new path reproduces the old
one exactly before using it for anything.** If any of the 500 cells differs, stop and report — a
difference means P4.4's numbers would be measured on a different instrument than P4's, and no
comparison between them would mean anything.

**Q7 — paired-difference CI contains 0. APPROVED as the instantiation of §9's registered wording**
(*"if BC on the expert slice matches MADT within CIs"*). **ADDITION (2): report the CI's WIDTH beside
the verdict.** "Contains 0" is a failure to reject, not a demonstration of equivalence. Reporting the
width converts *"no difference found"* into *"no difference larger than ±X found"*, which is the
statement that is actually true and is what a reviewer will ask for. Cheap, and it makes a null
result publishable rather than merely honest.

**Q8 — background jobs from this session. APPROVED**, on the P4 precedent at the same scale.
Conditions from §7: the campaign aborts on first failure **and** ends with an assertion that
completed runs equal runs requested. If any single job exceeds ~30 minutes, hand that one to the
user's `tmux` rather than absorbing it.

**Q9 — I commission the review, not you.** §7 makes it my step in the lifecycle, and P2.6 merged
without one precisely because it depended on my remembering. Self-review with `contract-reviewer`
if you find it useful; **the mandated pre-merge review is mine and I will run it.**

**ADDITION (3) — register your prediction before training.** Your F11 is a genuine forecast: final
DT cross-entropy of 0.013–0.024 means it nearly memorises the behaviour policy, so **you expect BC to
be close.** Write that into `docs/plans/p4.4.md` **before the first gradient step**, with the
reasoning. This project has repeatedly found that a prediction registered in advance is worth more
than the measurement that follows it, and a confirmed prior here is itself a finding: it would mean
the DT's margin comes from something other than better imitation.

**One thing you flagged and scoped correctly:** P4 reported no effect sizes though §8 makes them
mandatory. Back-filling P4's artifact is outside your fence — **it is now `DEFERRED` row 31 and mine.**

---

## 8. Second ruling — four corrections to §7 (Master chat, 2026-08-11)

**Three of my five additions landed; two did not, and one guard I approved has no discriminating
power at the row it exists for. None of this changes your design.**

### 8.1 δ IS NOW DECLARED — `PREREGISTRATION.md` amendment **A6**, before any baseline exists

Reporting the CI's width was half the fix. Without a declared margin, *"is this small enough to call
them equivalent"* is judged **after** the number is visible — at abstract-writing time, when the
pivot condition is read off. That is the degree of freedom D5 and D7 exist to remove.

> **δ = 0.6263 ATT** — the DT's own paired margin over MAPPO@1000 on the 100 held-out draws
> (105.5820 − 104.9558), from the committed `p4_gate.json`.
>
> **BC matches the DT** iff the 95 % CI of the paired per-draw difference (DT − BC) lies **entirely
> within [−δ, +δ]**. **The DT is genuinely better** iff that CI lies entirely below −δ. Anything
> straddling is **inconclusive at this power**, reported with its width.

**δ is derived, not chosen.** The question is whether sequence modelling earned the DT's margin over
its own data, so that margin *is* the scale at which "matches" must be judged: if BC comes within it,
whatever the DT gained BC gained too. It is strict — 0.597 % of the DT's ATT, 1/115th of its margin
over MaxPressure. Report all three verdict branches explicitly; **do not collapse them to
pass/fail.**

### 8.2 Your F11 forecast must be FALSIFIABLE — and δ is the same quantity

*"BC should be close"* is confirmed by every outcome. That is the tautology rule applied to a
forecast — the same shape as BRIEF_08's monotonicity criterion and BRIEF_09's all-True mask, both
caught in pre-flight. **Register it as: "BC lands within δ = 0.6263 ATT of the DT on the paired
held-out mean."** It can fail, which is the point.

### 8.3 Q3/Q4 — I approved IQL's package without engaging its substance. Doing that now

**IQL is the one arm where "untuned" is not neutral.** BC has essentially no hyperparameters beyond
the shared trunk; IQL has four that materially change behaviour (τ, β, γ, reward scale), and they are
**D4RL locomotion values from continuous-control MuJoCo, transplanted onto a discrete 8-phase action
space**, with a normalisation convention applied outside the domain it was defined for.

**(a) BINDING, regardless of what you do next.** A **losing untuned IQL cannot support any claim of
MADT superiority**, and the packet must say so **in the same sentence that reports the number** —
not in a caveats section. Write it as: *"MADT outperforms an untuned IQL configured from published
D4RL-locomotion values; this is not evidence that MADT outperforms IQL."*

**(b) OPTIONAL, and authorised if you want it.** A bounded sweep over **τ and β only**, ≤6 configs,
**selected on training-split expectile/TD loss — never on evaluation return**, which keeps it inside
D5. Declare the grid and the selection criterion **before** running it. If you sweep, report the grid
and criterion; if you do not, (a) binds alone. Either is defensible; leaving it implicit is the only
option that is not.

### 8.4 Q1 — I approved the route but the guard misses the row at risk

Your cross-check pins `(s_t, a_t)` for `t ∈ [0, T−1]` against the loader — **and the loader by
construction never yields `s_T`, which is the entire reason the deviation exists.** So it verifies
everything except the row at risk: **discriminating power near zero at exactly the point that
matters.** That is my own §7 rule failing on a test I signed off.

**Required, in addition to T3:** an independent assertion that the **last transition's next-state
equals row `T` of the raw `.npz` observation array**, read by a **different path** — raw `np.load`,
not `load_episode` — **with the mutation `row T → row T−1` proved to kill it.** Paste the mutation.

**None of the above changes the plan.** Write `docs/plans/p4.4.md` with δ and the forecast in it, and
go red-first.

---

## 9. Third ruling — two more, both before the first gradient step (2026-08-11)

### 9.1 Report the RECOVERED FRACTION unconditionally, in every branch

δ equals the effect under study, which creates one misleading branch: **BC landing 0.5 ATT worse than
the DT sits inside ±δ and returns "matches", while having recovered only 20.2 % of the DT's margin
over its behaviour policy.** A6's registered reading — *"whatever the DT gained, BC gained too"* —
is not true of that outcome.

**Report `recovered_fraction = (ATT_MAPPO@1000 − ATT_BC) / (ATT_MAPPO@1000 − ATT_DT)` beside the
verdict, always** — not gated on `δ/2`, because a gate would introduce a second chosen threshold to
defend, and one number is not clutter. Same for %BC and IQL.

**And a correction to my own A6:** I wrote *"δ is derived, not chosen"*. The **reference quantity** is
derived; the **multiplier of 1.0 is a choice**. Half the margin, or the margin's CI width, are equally
derivable. A6 is annotated accordingly. The decision rule is unchanged.

### 9.2 The IQL sweep decision must be made BEFORE the first gradient step

As §8.3 stood, the sweep was authorised but optional and **you** chose. If that choice is made after
seeing untuned IQL lose, the **decision** is outcome-driven even though the **selection inside** the
sweep is clean — the same distinction as the budget raise, one level up.

**Write the choice into `docs/plans/p4.4.md` before training anything, with its reason.** Sweep or do
not sweep; either is defensible. Deciding in advance is the whole of it.

**Nothing further from me.** The remaining work is yours.
