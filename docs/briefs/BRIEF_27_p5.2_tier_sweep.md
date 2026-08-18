# BRIEF 27 — P5.2: the spatial DT across the ladder, and the head-count confound

**Mode:** Claude Code · **Branch:** `task/p5.2-tier-sweep`, from `main`
**Worktree:** `git worktree add /home/filip/rltraffic-p52 -b task/p5.2-tier-sweep main`
**Read first:** `PROJECT_PLAN` **§1, §1b (ALL of it — R2/R6's scope condition, R7, C1–C4)**, then
`docs/reviews/P5.1.md` and `docs/returns/P5.1-gate2.md`.

⚠️ Absolute paths · pin threads · `git add` **before** `check_english.sh` · guards with **no
arguments**, counted from **full output** · `tmux` for the campaign (`BRIEF_17` §12, six conditions).

---

## ⭐ AMENDMENT A — 2026-08-18, issued while the implementer is in PLAN MODE and before any plan file exists

**Five corrections, all mine, all found by auditing this brief against the artifacts rather than
against its own prose. A1 and A2 change what the task measures; A3 changes what it runs; A4 replaces
an estimate with a measurement; A5 fixes a checkbox this task will otherwise tick falsely.**

### A1 🚨 THE RESULT THIS BRIEF OMITTED, AND IT CHANGES WHAT P5.2 IS FOR

**`dt_nomix` — the identity-graph CONTROL — ranks 1 of 5 method arms on `grid4x4@mappo1000` and beats
the MAPPO policy that collected the corpus.** §1 below leads with the treatment's harm and never says
this. Verified by me on 2026-08-18 from `docs/data/p5_1_grid.json` directly, **not** from the packet's
table:

| paired contrast | mean difference | CI95 | seed reversal |
|---|---|---|---|
| `dt_nomix` − `behaviour` (MAPPO@1000) | **−2.4303** | [−2.5796, −2.2810] | **none** (rb −0.9604, p 7.7e−17) |
| `dt_nomix` − `bc` | **−11.1329** | [−13.7932, −8.4726] | **none** |
| `dt_nomix` − `iql` | **−117.9877** | [−125.0825, −110.8928] | **none** |
| `dt_nomix` − `bc_top10` | **−591.7319** | [−598.6621, −584.8016] | **none** |

**This is the first tier anywhere in this study where a DT arm ranks FIRST among the method arms.**
`PROJECT_PLAN` says *"the DT leads 0 of 8 tiers"* in three places and uses it as P5.2's motivation —
that sentence is about **eight hz1x1 tiers**, and the ninth tier, the first with 16 intersections, is
**led by a DT**. ⚠️ **The fence P5.1's packet already put on it stands and must travel with every
sentence: `dt_nomix` is the identity-graph model, NOT the merged `agent/DTAgent.py`** — one
parameter-shared model over 16 nodes with no cross-node information flow.

> **RULED: §3's out-of-sample registration must cover `dt_nomix`'s RANK and its paired difference
> against `behaviour` AND against the best non-DT arm, PER TIER — not only the `dt_spatial` −
> `dt_nomix` contrast.** The same cells answer two questions and **the second is the more publishable
> one**: (i) does spatial mixing's harm survive the ladder, (ii) **does the DT's lead survive it.**
> ⚠️ `mappo1000` is **already seen** and is therefore NOT out-of-sample for (ii); only `maxpressure`
> and `random` are. **Say so in the registration rather than letting a seen cell be scored as a hit.**

### A2 🔒 THE HEAD-COUNT ARM IS A 2×2, NOT ONE CELL — §2 AS WRITTEN IS CONFOUNDED

§2's stop rule fires on *"`n_head = 4` reverses P5.1's sign"*. **That sign is (`dt_spatial` −
`dt_nomix`), and it is only defined when BOTH arms carry the same head count.** Running
`dt_spatial@4H` against `dt_nomix@1H` varies mixing and head count together and answers neither
question. **The ambiguity is mine and it is the kind that gets frozen into a campaign.**

> **RULED: the head-count arm is `{dt_spatial, dt_nomix} × n_head ∈ {1, 4}` at `mappo1000`.** The two
> 1-head cells are P5.1's and are **reused verbatim** (A3), so the new training is **two cells × 5
> seeds**. **The reported quantity is the INTERACTION** — (spatial − nomix)@4H **minus**
> (spatial − nomix)@1H — with the two simple effects beside it. **The stop rule fires on the sign of
> (spatial − nomix)@4H and on nothing else.**
> ⚠️ **State this in the plan, because the paper may not call the 4-head arm a GAT-head ablation:**
> `SpatialDTAgent.py:233-235` gives the temporal and spatial sublayers the **same** `n_head`, so
> `n_head=4` also raises the TEMPORAL head count, in **both** arms. Verified in source by me today.
> That is a property of the architecture under test, not a defect — but it must be named.

### A3 🔒 `mappo1000` IS NOT RE-RUN — P5.1's CELLS ARE REUSED VERBATIM

**The reason is scientific before it is economic.** Re-training that tier would produce a **second,
slightly different value** for `dt_nomix@grid4x4_mappo1000` (`DEFERRED` 51: GPU/BLAS nondeterminism is
an open exposure, not a closed one), and **a merged, independently reviewed number appearing twice
with two values is worse than any saving it buys.**

> **RULED:** P5.2 consumes `output/p5_1/eval_*.json` for the seven `mappo1000` cells and records their
> provenance in the artifact — source path plus the sha256 from `output/SHA256SUMS_p5_1.txt`, which I
> re-verified **48/48** on 2026-08-18. **Only NEW arms are run at that tier** (per-intersection %BC,
> and the two 4-head cells). **Any `mappo1000` cell P5.2 re-runs anyway must be declared in the plan
> with its reason** — reuse is the default and the exception is what needs an argument.

### A4 📏 MEASURED COST — replacing §5's extrapolation, measured by me tonight from P5.1's preserved mtimes

`output/p5_1/` mtimes are originals (§6 records *"48/48, mtimes preserved"*). Campaign span
**2026-08-17 ~13:45 → 2026-08-18 03:11 ≈ 13 h 26**, not the ~17 h estimated, and the shape matters
more than the total:

| stage | measured |
|---|---|
| DT training | **~59 min per (arm, seed)** → ~4 h 55 per 5-seed DT arm; **9 h 44 for both arms** |
| BC / %BC / IQL training, 3 arms × 5 seeds | **42 min total** |
| evaluation, 7 arms × 5 seeds × 100 draws | **3 h 00** (`behaviour` alone is 56 min) |

**DT training is 73 % of the campaign and scales with the number of DT cells — nothing else does.**
Projected for this brief **with A3's reuse**: ~13.5 h per NEW tier × 2 new tiers + ~11 h for the
4-head pair + ~1.5 h for the new %BC arm ≈ **52 h**. Without reuse, ≈ 65 h.
> §5's cut is unchanged and is now costed: **dropping `maxpressure` saves ~13.5 h.** ⚠️ **If the
> 4-head pair overruns, it does NOT get cut — A2 makes it the task's first result and §2 makes it a
> stop rule.** The tier breadth is the expendable half; the architecture question is not.

### A5 §6's P5.2 LINE ALSO DEMANDS TWO ONLINE BASELINES WE DO NOT HAVE

§0 strikes `hangzhou_4x4`. The same line reads *"compare vs online **MAPPO/IPPO/DQN**"*. `behaviour`
(MAPPO@1000) is the online-MAPPO comparison and exists; **IPPO and DQN have never been trained on
grid4x4** — `DEFERRED` 6 still records `dqn` as an unstarted nice-to-have.
> **RULED: strike `IPPO/DQN` from §6's P5.2 text in the same commit that strikes `hangzhou_4x4`**, or
> the ticked box claims work nobody did. **An unticked box is information; a falsely ticked one is a
> corrupted record.**

---

## 0. ⚠️ SCOPE CORRECTION BEFORE ANYTHING ELSE — §6's P5.2 NAMES A SCENARIO WE DO NOT HAVE

§6 reads *"Train + evaluate on grid4x4, **hangzhou_4x4** per ladder tier"*. **Measured today:
`datasets_v11/` holds exactly three scenarios — `cf_hz1x1`, `cf_grid4x4`, `cf_cologne3`. There is no
hangzhou_4x4 corpus**, and collecting one is a campaign, not a task.

> **RULED: P5.2 is `cf_grid4x4` ACROSS THE LADDER. hangzhou_4x4 is struck from the item.**
> 🚨 **AND `cf_cologne3` IS STRUCK TOO — CORRECTED 2026-08-18, BY MEASUREMENT, AFTER THIS PARAGRAPH
> OFFERED IT AS *"the cheap option"*.** I derived its graph rather than assuming one:
> `adjacency_from_roadnet_file('scenarios/cologne3/cologne3_roadnet_red.json', ix_ids)` over the
> corpus's own node order returns a **3×3 mask with ZERO off-diagonal edges** — cologne3's three
> intersections share no lane, so `A.outgoing_lanes ∩ B.incoming_lanes = ∅` for every pair.
> **Control, same command: grid4x4 returns 48 off-diagonal edges over 16 nodes, exactly a 4×4 grid** —
> so the zero is a property of the network, not a broken tool.
> ⚠️ **CONSEQUENCE: on cologne3 the graph IS the identity, so `dt_spatial` and `dt_nomix` are THE SAME
> MODEL. A spatial-mixing arm there is a tautology by construction** — the failure `PROJECT_PLAN` §7
> (2026-08-07) and `DEFERRED` 46 both exist to prevent, and it would have been discovered after the
> compute was spent. **No third topology is available for the spatial question without a new corpus.**

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
   uncontrolled (§7, 2026-08-18).

## 5. Cost and the cut that governs it

grid4x4 trains at **2.81 s/episode** against hz1x1's 1.04. **P5.1's single tier was ~17 h.** A full
seven-tier sweep is not affordable in the September window.
> **RULED: three tiers — `mappo1000`, `maxpressure`, `random` — spanning the ladder's measured ATT
> range, not its tier names.** ⚠️ **`mappo060` is WORSE than `fixedtime` (281.89 vs 262.09), so order
> every figure by measured ATT.** **If it overruns, drop to `mappo1000` + `random`** — the endpoints
> carry the interaction; the middle does not.

## 6. Definition of Done

- [ ] `docs/plans/p5.2.md` committed **before any training**, carrying §3's out-of-sample registration
      **extended per A1 to `dt_nomix`'s rank and its paired differences**, §2's stop rule as **A2's
      interaction**, the declared tier set, and **A3's reuse declaration with the cells it covers**
- [ ] **A2's 2×2** — `{dt_spatial, dt_nomix} × n_head ∈ {1, 4}` at `mappo1000`, the 1-head pair reused
      from P5.1 — **reported first**, as the interaction with both simple effects beside it; stop rule
      honoured if (spatial − nomix)@4H changes sign
- [ ] Three tiers × arms, 5 seeds × 100 held-out draws, 40,000 steps; per-intersection %BC as a **new
      arm** beside the global filter. **`mappo1000` is reused per A3, with provenance and sha256 in the
      artifact; only new arms run there**
- [ ] Every ordering with its per-seed count **and** range, **emitted by the generator**
- [ ] `DEFERRED` 37's mutation executed, failure pasted; every mutation's failure pasted
- [ ] Campaign in a **user-launched `tmux`**; **no `until`-poll**; `mkdir -p` before `tee`, and the
      script must not clear the log directory after `tee` opens its target (P4.7 and P5.1 both lost a log)
- [ ] Suite green, tail pasted, pinned; three guards, no arguments, full-output counts, corpus named
- [ ] Return Packet at `docs/returns/P5.2.md` with the AI-assistance record
- [ ] §6's checkbox unticked; it is mine, in the merge commit — **and §6's P5.2 text must be corrected
      to strike hangzhou_4x4 AND `IPPO/DQN` (A5) in that same commit**
