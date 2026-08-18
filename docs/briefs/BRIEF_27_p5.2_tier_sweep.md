# BRIEF 27 — P5.2: the spatial DT across the ladder, and the head-count confound

**Mode:** Claude Code · **Branch:** `task/p5.2-tier-sweep`, from `main`
**Worktree:** `git worktree add /home/filip/rltraffic-p52 -b task/p5.2-tier-sweep main`
**Read first:** `PROJECT_PLAN` **§1, §1b (ALL of it — R2/R6's scope condition, R7, C1–C4)**, then
`docs/reviews/P5.1.md` and `docs/returns/P5.1-gate2.md`.

⚠️ Absolute paths · pin threads · `git add` **before** `check_english.sh` · guards with **no
arguments**, counted from **full output** · `tmux` for the campaign (`BRIEF_17` §12, six conditions).

---

## 0. ⚠️ SCOPE CORRECTION BEFORE ANYTHING ELSE — §6's P5.2 NAMES A SCENARIO WE DO NOT HAVE

§6 reads *"Train + evaluate on grid4x4, **hangzhou_4x4** per ladder tier"*. **Measured today:
`datasets_v11/` holds exactly three scenarios — `cf_hz1x1`, `cf_grid4x4`, `cf_cologne3`. There is no
hangzhou_4x4 corpus**, and collecting one is a campaign, not a task.

> **RULED: P5.2 is `cf_grid4x4` ACROSS THE LADDER. hangzhou_4x4 is struck from the item.** If a third
> topology is wanted, **`cf_cologne3` exists (3 intersections, real network) and is the cheap option** —
> but it is **NOT in this brief** and may not be added without a ruling.

## 1. What P5.2 is for

P5.1 measured spatial mixing on **one** tier and found it **decisively harmful, unstably so**
(treatment sd **30.36** against control **0.10**, vehicles at horizon `16.15 · 133.94 · 123.15 · 75.66
· 49.36` against a control flat at ~15). **P5.2 asks whether that survives the data-quality axis** —
the interaction §10's cut 2 called the expendable half, now the whole task.

## 2. 🚨 THE FIRST REGISTERED QUESTION IS THE HEAD COUNT, NOT THE LADDER

§1b's **C3** registered, before P5.1's campaign: `agent/SpatialDTAgent.py:136` declares **`n_head = 1`**,
inherited from `DTAgent.py:107`, and **`:233-235` shows the temporal and spatial sublayers SHARE it —
so the GAT-equivalent is also single-head.** Both in-domain papers use **4 heads with 2 GAT layers**.

> **P5.1's negative result is about the SINGLE-HEAD configuration and says nothing else. Vary
> `n_head ∈ {1, 4}` on ONE tier BEFORE the ladder sweep**, and report it as the first result.
> ⚠️ **If `n_head = 4` reverses P5.1's sign, the ladder sweep is measuring the wrong architecture and
> you STOP and report.** That is a stop rule, not a contingency — **declare it in the plan.**

## 3. 🔒 THE OUT-OF-SAMPLE REGISTRATION — IN THIS BRIEF'S WINDOW, BEFORE ANY P5.2 NUMBER

**The blocker is cleared: P8.3 externally anchored the IQL column** — `bc_top10 > iql > bc` reproducing
the published direction through a shared adapter, no defect visible. **So a prediction registered now
rests on a column that has an anchor rather than one nobody had checked.**

> **Register in `docs/plans/p5.2.md`, before the first gradient step, a falsifiable out-of-sample
> prediction for every arm × tier cell**, with the rule that produces it stated as a function of
> quantities already on disk. **It is scored mechanically and reported whichever way it comes out.**
> ⚠️ **Five of six registered predictions have failed across P4.6/P4.7/P5.1. Write it to be falsifiable
> and build no rescue** — `BRIEF_20` §1 exists because a rescue was attempted once and did not survive.

## 4. Inherited constraints — all binding, none optional

1. **`DEFERRED` 37** is live on 16 intersections. **Execute the mutation, paste the failure.**
2. **`DEFERRED` 21**: masks never bind in this corpus — **the paper may not credit masked-action
   modelling**, and a no-mask ablation measures nothing.
3. **§1b R2/R6's SCOPE CONDITION**: a return-quantile filter is a quality filter **only when returns are
   comparable across the units being filtered.** On a network they are dominated by **load** — P5.1's
   `bc_top10` kept 302/320 streams from the 8 quietest of 16 nodes and collapsed to 749.58.
   **Per-intersection %BC is a NEW ARM, declared and reported BESIDE the global filter, never a
   re-specification.** ⚠️ **The un-re-specified collapse IS a result and must still be reported.**
4. **§1b R7 is a DESCRIPTIVE COUNT.** No family-wise correction exists; **no sentence derived from a
   count of orderings may carry inferential weight.**
5. **Every reported ordering carries its per-seed agreement COUNT and its per-seed RANGE.** ⚠️ **The
   enforcement is in the GENERATOR** — P5.1's J1 was a datum the artifact already held and the prose
   did not read. **If `reverses_on_n_seeds` is non-zero, the qualifier is emitted, not remembered.**
6. **ATT is bounded below and right-skewed.** The registered paired protocol does not change, but
   **state why the mean is the primary and report medians or a distributional comparison beside it for
   every arm whose distribution is visibly not unimodal.** `bc_top10@grid4x4` at 749.58 with 505 of
   ~1300 vehicles in network is the standing example.
7. **The single-Q deviation** (`offline_baselines.py:1960-1963`, no clipped double Q, in every merged
   P4.x number) **stays registered and unrepaired — repairing it invalidates P4.4–P4.7 — and must be
   disclosed wherever an IQL number appears.**
8. **P8.3's numbers are fenced**: they license nothing and may not enter the paper without a review.
9. **Any cancellation argument names WHICH component is common** and verifies the rest or declares them
   uncontrolled (§7, 2026-08-19).

## 5. Cost and the cut that governs it

grid4x4 trains at **2.81 s/episode** against hz1x1's 1.04. **P5.1's single tier was ~17 h.** A full
seven-tier sweep is not affordable in the September window.
> **RULED: three tiers — `mappo1000`, `maxpressure`, `random` — spanning the ladder's measured ATT
> range, not its tier names.** ⚠️ **`mappo060` is WORSE than `fixedtime` (281.89 vs 262.09), so order
> every figure by measured ATT.** **If it overruns, drop to `mappo1000` + `random`** — the endpoints
> carry the interaction; the middle does not.

## 6. Definition of Done

- [ ] `docs/plans/p5.2.md` committed **before any training**, carrying §3's out-of-sample registration,
      §2's stop rule, and the declared tier set
- [ ] `n_head ∈ {1, 4}` on one tier, **reported first**; stop rule honoured if the sign reverses
- [ ] Three tiers × arms, 5 seeds × 100 held-out draws, 40,000 steps; per-intersection %BC as a **new
      arm** beside the global filter
- [ ] Every ordering with its per-seed count **and** range, **emitted by the generator**
- [ ] `DEFERRED` 37's mutation executed, failure pasted; every mutation's failure pasted
- [ ] Campaign in a **user-launched `tmux`**; **no `until`-poll**; `mkdir -p` before `tee`, and the
      script must not clear the log directory after `tee` opens its target (P4.7 and P5.1 both lost a log)
- [ ] Suite green, tail pasted, pinned; three guards, no arguments, full-output counts, corpus named
- [ ] Return Packet at `docs/returns/P5.2.md` with the AI-assistance record
- [ ] §6's checkbox unticked; it is mine, in the merge commit — **and §6's P5.2 text must be corrected
      to strike hangzhou_4x4 in that same commit**
