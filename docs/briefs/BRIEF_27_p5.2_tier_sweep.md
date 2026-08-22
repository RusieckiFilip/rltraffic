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
🚨 **PROJECTION CORRECTED 2026-08-18 — BOTH TOTALS BELOW WERE WRONG, AND THE FIRST CONTRADICTED ITS
OWN FORMULA IN THE SAME SENTENCE.** It read *"~13.5 × 2 new tiers + ~11 + ~1.5 ≈ **52 h**. Without
reuse, ≈ 65 h."* **13.5 × 2 + 11 + 1.5 = 39.5, not 52** — the 52 was the pre-A3 three-new-tier figure,
left standing when A3 removed a tier from the count, and **65 was never derivable from any version.**
⚠️ **A document that carries 52 in A4 and 38 in B6 has no cost estimate at all.** Recomputed from the
same measured units, with the arithmetic shown so it can be checked rather than trusted:

| what | h | how |
|---|---|---|
| one FULL tier | **13.9** | 9.8 DT training (2 arms × 5 seeds × 59 min) + 0.7 baselines + 3.4 eval (8 arms) |
| `mappo1000` — **reused**, new arms only | **0.5** | per-intersection %BC: train + evaluate |
| 2 new tiers (`maxpressure`, `random`) | **27.8** | 2 × 13.9 |
| 4-head pair at `mappo1000` | **10.8** | 9.8 training + ~1.0 eval |
| **TOTAL, as ruled** | **≈ 39 h** | agrees with the implementer's independent **~38 h** |
| *+ optional `fixedtime` 4th tier (B1)* | *≈ 53* | *+13.9* |
| *without A3's reuse* | *≈ 52.6* | *+13.5 to re-run `mappo1000`* |

⚠️ **The 1.4× over-estimate prior from B6 does NOT apply on top of this: the 59 min/seed above is the
MEASURED per-seed cost from P5.1's checkpoint times, not the 85 min/seed pre-run bench that the prior
was computed against.** Discounting it again would double-count the correction.
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

## ⭐ AMENDMENT B — 2026-08-18, the plan-mode gate: four rulings, and one of the conflicts is MINE

**The implementer returned three conflicts between this brief and the repo. I verified all three from
the artifacts before ruling. All three stand, and C-1 is my error.**

### B0 🚨 C-1 CONFIRMED — §5's ATT FIGURES ARE `cf_hz1x1`'s, QUOTED AS `cf_grid4x4`'s. MINE.

§5 warns *"`mappo060` is WORSE than `fixedtime` (281.89 vs 262.09)"*. **Read from
`docs/data/att_ladder_v11.json` today: `281.89` and `262.09` are `cf_hz1x1`'s cells.** The scenario
this task runs on reads:

| grid4x4 tier | ATT at horizon | grid4x4 tier | ATT at horizon |
|---|---|---|---|
| `mappo1000` | **160.33** | `random` | **257.73** |
| `maxpressure` | **167.49** | `mappo200` | **568.40** |
| `mappo500` | **192.70** | `mappo060` | **1370.22** |
| `fixedtime` | **206.93** | | |

**The instruction survives — `mappo060` is worse than `fixedtime` on grid4x4 too, by 6.6× — but the
RATIONALE does not, and the rationale is what chose the tiers.** ⚠️ **This is the project's signature
error, mine, in a brief: a number true of one scenario stated about another.** **Every figure in this
task orders by `cf_grid4x4`'s column above, and no hz1x1 ATT number may be quoted as grid4x4's.**

### B1 ✅ TIER SET — **KEEP `{mappo1000, maxpressure, random}`**, for a reason the brief never gave, with the weakness DISCLOSED and a fourth tier PRE-DECLARED

The implementer recommended keeping and was right, but the ladder-spacing objection it raised is real
and must travel with the result rather than be dismissed.
> **RULED, and the deciding reason is the question A1 created, not the one §5 argued from: `maxpressure`
> is the tier where the DT ranked 4 of 4 — LAST — on hz1x1** (§1's claim constraint). **P5.2's headline
> question is now *does `dt_nomix`'s lead survive*, and the most falsifying rung available is the one
> where the DT was worst elsewhere.** `fixedtime`, where the DT ranked 2/4, is a softer test.
> ⚠️ **DISCLOSE, in the plan and in the packet, in these terms: on grid4x4 `mappo1000` and
> `maxpressure` are 7.16 ATT apart — 4.5 % — so the three tiers do NOT evenly span the corpus's
> measured range (160.33 → 1370.22), and no sentence may say they do.** **The tiers separate on the
> RETURN axis** (implementer's measurement from raw `.npz`: `mappo1000` −165.6 · `maxpressure` −251.5 ·
> `random` −918.6, a **5.5×** spread) **and C1 is registered in measured return, so the design is sound
> — say that, rather than claiming an ATT span we do not have.**
> ⭐ **`fixedtime` (206.93, return −492.6) IS PRE-DECLARED NOW AS THE FOURTH TIER, to be run only if
> phase B finishes with budget left** (~13.4 h). **Declared before any P5.2 number exists, so adding it
> later is not post-hoc; it is the best-spaced middle rung and it is the tier that broke §1b R3's
> monotonicity and flipped R2's sign on hz1x1, so reproducing those is free information.** **It may not
> REPLACE `maxpressure`.**

### B2 ✅ GATE STRUCTURE — CONFIRMED, exactly P5.1's shape

Gate 1 = plan, tests, code, declaration, campaign script, **committed before any training**. The user
launches the `tmux` campaign. Gate 2 scores it in a later session. **The declaration and the plan are
committed BEFORE the launch, or the registration is worthless.**

### B3 ✅ REUSE GATE — APPROVED, with three conditions, two of them from §7's new rule

The random-anchor re-roll is the right gate: a CPU-deterministic policy, so exact equality is the
correct bar, and it tests draw materialisation **and** the evaluation harness at once.
> **RULED, all three binding.** **(a)** Verify the reused files against `output/SHA256SUMS_p5_1.txt` **at
> consumption time** and record source path **and** digest per reused cell in the artifact — I
> re-verified 48/48 on 2026-08-18, and a digest checked once is not a digest checked when used.
> **(b) The equality check needs a POSITIVE CONTROL and a non-empty assertion** (`PROJECT_PLAN` §7,
> 2026-08-18): assert the comparison set is **exactly 500 episodes** before comparing — so *"found no
> differences"* can never be *"compared nothing"* — and prove the check fires by perturbing one episode
> deliberately and pasting the failure. **(c) On mismatch the campaign REFUSES AND STOPS. It does not
> silently fall back to re-running**, because a fallback hides exactly the drift the gate exists to
> detect, and the mismatch is then a reportable finding about our own reproducibility.

### B4 ✅ SIZE-MATCH THE `random` TIER — YES, AND THE PRECEDENT IS VERIFIED RATHER THAN ASSUMED

C-3 confirmed and it is **wider than reported**: `random` holds **400** episodes on **all three
scenarios** (`cf_hz1x1`, `cf_grid4x4`, `cf_cologne3`); every other tier holds 200.
> **RULED: mirror P4.6 exactly — `subsample="one_per_draw"`, 200 streams, declared RNG, selection
> recorded in the artifact.** ✅ **I verified P4.6 did this rather than taking it on report:
> `p4_6_grid.json` carries `"subsample": "one_per_draw"` with `training_streams: 200` on the
> `target_rtg −38369` cell — the `random` tier — against `"subsample": "none"` elsewhere. So no merged
> P4.6/P4.7 number is confounded with corpus size, and this ruling follows precedent instead of
> inventing one.** ⚠️ **Unmatched, tier would be confounded with training-set size — the one confound
> §7 (2026-08-12) records as already having misled us, where *"plateaued"* tracked training-set size
> and not convergence.**

### B5 📏 THE REGISTERED PREDICTION — THREE ADDITIONS BEFORE IT IS COMMITTED

The proposed rule is falsifiable, uses only on-disk quantities and builds no rescue. **Three things it
must add, all cheap, all before the first gradient step:**
1. **SCORE RANK, NOT ONLY LEVEL.** The paper's sentence is *"the DT leads / does not lead"*. **A ±30 %
   ATT band can hold on every cell while every ordering flips.** Register an explicit **per-tier rank
   prediction** — which arm is 1st — and score it **separately** from the band.
2. **EXCLUDE SEEN CELLS FROM THE DENOMINATOR.** `mappo1000` is reused and already seen; those cells are
   free hits. **Enumerate the out-of-sample cell set explicitly and score `k of N` over that set only.**
3. **FIX THE BAND AND THE THRESHOLD IN THE COMMIT.** ±30 % against a calibrated median error of 23.9 %
   is close to a coin flip per cell, which makes it a real test — **and it may not be widened after any
   result is visible.** State the calibration (median 23.9 %, max 378.5 %) beside it.

### B6 ✅ COST FIGURE CORRECTED — the implementer's number is better than mine

A4 said *"≈ 13 h 26, measured from preserved mtimes"*. **The log is the artifact and it is more precise:
`output/p5_1/logs/campaign.log` reads `[2026-08-17 13:46:36]` start and `03:11:01` finish = 13 h 24 m
25 s.** Mine was inferred by extrapolating backwards from the first checkpoint; theirs was read.
⚠️ **Also worth carrying: the same log shows the pre-run estimate `~85 min/seed measured for
dt_spatial` against ~59 min/seed actual — a 1.4× over-estimate, which is the prior to apply to the
~38 h projection rather than treating it as tight.**

---

## ⭐ AMENDMENT C — 2026-08-18, the plan gate: `docs/plans/p5.2.md` @ `f02c917` is APPROVED, with four requirements

**I audited the plan against the artifacts, not against its summary. It is the strongest planning
document this project has produced.** ⭐ **Its registered level table reproduces EXACTLY: I
re-implemented rule R′ from the plan's own formula against `p5_1_grid.json`, `p4_6_grid.json` and
`att_ladder_v11.json`, and all 13 predicted cells agree to < 5e−5** — and the calibration's anchors
resolve to the known merged values (`104.9558` for the DT, `103.16` for %BC), so the rule is wired to
the artifacts it claims. **`bc_top10_perix` is defined as the controlled contrast it needs to be:**
`ceil(0.10 · n)` **within each node's own stream set** — 20 × 16 = **320**, exactly the global filter's
320 — so the two filter arms differ in **which** streams they keep and not how many.
**Coding may begin once the four below are in the plan file.**

### C1 🔒 PROOF OBLIGATION 6 NEEDS A SAME-DEVICE CONTROL, OR A FAILURE IS UNINTERPRETABLE

Byte-identity between the **old** and **new** trainer is evidence only if the **old** trainer is
byte-identical **to itself** on that device first.
> **REQUIRED: run `spatial_mixing.train_spatial_dt` TWICE at `n_head=1`, same seed, same budget, same
> device, and report whether it reproduces itself — BEFORE comparing it to the new trainer.** If it
> does, `==` is the right bar and the obligation stands as written. **If it does not, byte-identity is
> unavailable on that device and the honest bar is CPU equality plus a REPORTED variance envelope from
> the control** — never a silent relaxation to `allclose`. This is §7's discriminating-power rule: a
> check must report the distance between the right answer and a wrong one, not only that it passed.
> ⚠️ **And say which obligation licenses what, because obligation 6 alone does not license reuse:**
> **6 licenses the CODE PATH** (new trainer ≡ old at 1 head); **B3's digest + random-anchor gate
> licenses the ARTIFACTS** (P5.1's CUDA-trained cells are comparable with P5.2's). Both are needed and
> the plan has both — the risk is a packet sentence that credits one with the other's work.

### C2 🔒 THE EQUIVALENCE RUN EXERCISES A SCHEDULE THE CAMPAIGN NEVER USES

Verified in source today: `offline/spatial_mixing.py:400` computes
`warmup = min(WARMUP_STEPS, max(1, total // 2))` with `WARMUP_STEPS = 1000`. **So warmup is a FUNCTION
OF THE BUDGET** — `total=100 → 50`, `total=500 → 250`, `total=40000 → 1000`. A short-budget run
therefore crosses *its own* boundary automatically (good) **but proves nothing about the schedule the
campaign runs.**
> **REQUIRED, and it costs no training: assert old and new trainers compute the SAME `warmup` and the
> SAME LR multiplier at `total = 40,000`, evaluated at steps `{0, 999, 1000, 1001, 40000}`.** A
> divergence there would change every reported cell while a 100-step equality test stayed green.
> *(`DEFERRED` 36 already records `warmup_steps` as an under-recorded knob; this is the same knob
> becoming load-bearing.)*

### C3 🔒 BRIEF §4.8's P8.3 FENCE IS BINDING AND APPEARS **ZERO** TIMES IN THE PLAN

Measured: `grep -c "P8.3" docs/plans/p5.2.md` → **0**. The single-Q disclosure is present and correct;
the fence is not.
> **REQUIRED, one line in the plan: no P5.2 sentence — plan, artifact, packet or paper — may cite
> P8.3's D4RL numbers as validation of the IQL arm.** P8.3 is an unreviewed external calibration whose
> own §6 fences it. ⚠️ **The failure mode is specific and attractive: justifying the IQL baseline with
> *"externally validated against D4RL"*.** The arm's honest disclosure is the single-Q deviation, which
> the plan already carries.

### C4 📏 Q2b's CONCORDANCE IS INFLATED BY TWO OBVIOUS PAIRS — REPORT THE HARD SUBSET BESIDE IT

The predicted orderings put `bc_top10` at **751 / 1243** and `iql` at **264 / 415**, far from the
other four arms. **Those two arms supply 9 of the 15 pairs per tier, and getting them right is nearly
free — so ≥ 12 of 15 can be met while every ordering that matters is wrong.**
> **REQUIRED, as an ADDITIONAL report and NOT a change to the registered threshold** (the primary is
> registered and stays): **report concordance restricted to the 6 pairs among
> `{dt_nomix, bc_top10_perix, bc, dt_spatial}`** — the four arms predicted within 147 ATT of each
> other at `maxpressure` — **beside the 15-pair count.** The paper's ordering sentence lives in that
> subset. **Adding a secondary before any data exists is free; discovering afterwards that the primary
> was carried by two easy pairs is not.**

### C5 ✅ WHAT I CHECKED AND FOUND SOUND — recorded so a later reader knows the audit's extent

Level table (13/13 to <5e−5) · calibration errors (23.9 / −1.0 / 5.8 / 378.5 / 75.5, median 23.9, 3 of
5 in band) · both predicted orderings sort correctly from the table · Q3b's margins (−0.9655,
−10.1702) follow from it · the size match is 200 episodes / 3,200 streams on every tier, with the unit
deviation correctly reasoned from `build_joint_index`'s refusal · the stop rule is in the phase order
and writes `STOPPED_BY_RULE` · Gate 1 refuses and stops · the completeness assertion reads the
declaration rather than the files being checked · rehearsal knobs execute the control flow and the
40,000-step assertion stops a rehearsal masquerading as a campaign · A5, the ATT-skew requirement,
`DEFERRED` 21, `DEFERRED` 37 and the per-seed emission are all present.
⭐ **§2.4 is a NEW measured finding and it generalises `PROJECT_PLAN` §1b's R2/R6 scope condition: the
global top-decile filter concentrates on 6–11 of 16 nodes on EVERY grid4x4 tier, not only
`mappo1000`.** It is correctly labelled a corpus measurement rather than a prediction.

---

## ⭐ AMENDMENT D — 2026-08-19. **The LAST requirement round on the plan.** One load-bearing gap, two corrections of mine, one durable rule

**The plan amendment at `2baf2cd` is accepted.** C1–C4 all landed, and **C4 landed better than I asked
for**: no threshold was registered on the hard subset, because a second threshold would be a second
registered claim rather than the disclosure C4 wanted. That is the right call. **Phase 3 is OPEN once
D1 is in the plan.** ⚠️ **This is the stopping point: D1 is a data-destruction risk, D2 is one number,
D3 and D4 are my errors. A further round would return style, and §7 says stop then.**

### D1 🚨 THE FILESYSTEM-MUTATION BARRIER IS ABSENT FROM THE PLAN, AND THIS CAMPAIGN IS ITS WORST-CASE SHAPE

Measured in `docs/plans/p5.2.md`: **`mutation barrier` 0 · `delete` 0 · `overwrite` 0 · `out_dir` 0**,
against **`resum` 2**. §7's barrier (*every write AND delete happens after all validation*) has
**already fired twice in this repo** — P1's NB2, where `--overwrite` destroyed a corpus **before**
constructor validation, and P2.0, where draw materialisation ran before the populated-`out_dir` check.
**A resumable campaign is precisely the shape that reintroduces it**: resume logic decides *"is this
cell already done?"* and then either skips or re-runs, and a re-run that truncates before its
replacement validates converts a crash into data loss.

> **REQUIRED, three parts, and the first is the one that matters most.**
> **(a) `output/p5_1/` IS READ-ONLY TO THIS TASK, ENFORCED IN CODE, NOT BY INTENTION.** Those seven
> cells and 25 checkpoints are the **only copy** of the evidence behind a merged, independently
> reviewed result — `output/` is gitignored, there is no backup, and §10 carries that as an open
> exposure for P10.0. **`tier_sweep.py` and `p5_2.sh` must REFUSE any write or delete whose resolved
> path lies under `output/p5_1/`**, with the refusal tested and its failure pasted. **The reuse gate
> reads those files; nothing in this task may open them for writing.**
> **(b) Every write and delete in `tier_sweep.py` happens after all validation.** A refused run
> constructs nothing and destroys nothing.
> **(c) The resume path never truncates or deletes an existing cell before its replacement has
> validated** — write to a temporary name, validate, then move. **Test the crash case: a half-written
> cell must not be mistaken for a complete one**, which is the failure resume logic exists to survive
> and the one that silently poisons a campaign.

### D2 📏 C2's STEP SET HAS ONE BLIND SPOT — ADD STEP 500

The pinned expectations are right: at `total = 40,000`, `warmup = 1000` and
`λ = min(1, (step+1)/1000)` gives `0.001 · 1.0 · 1.0 · 1.0 · 1.0` at `{0, 999, 1000, 1001, 40000}`,
and **step 999 is correctly identified as the discriminating point** — an off-by-one reads 0.999 there.
⚠️ **But four of the five expected values are identical, so the set's whole power sits at 0 and 999,
and a schedule that agreed at both endpoints while differing in SHAPE — a cosine or quadratic ramp —
would pass.**
> **REQUIRED: add step 500, expected `0.501`.** One number, it costs nothing, and it turns a
> two-point check into one that pins the ramp's shape. §7: a check must report its discriminating
> power, and a check whose points nearly all return the same value has little.

### D3 ⚠️ TWO ERRORS OF MINE THAT THE IMPLEMENTER ABSORBED WITHOUT COMPLAINT — CORRECTED HERE

**(i) `C4`'s *"within 147 ATT of each other at `maxpressure`"* is WRONG, and the implementer recovered
the real provenance rather than repeating it.** The four hard arms span **42.17** at `maxpressure` and
**62.71** at `random`; **146.81** is the **pooled two-tier** span (166.09 → 312.90). **I attributed a
pooled figure to one tier — the same scoping error as B0's hz1x1 numbers wearing grid4x4's label, one
week and three instances running.**
**(ii) `C1` described a conflation that was ALREADY IN THE FILE I HAD JUST APPROVED, and I did not say
so.** The approved obligation 6 read *"This is what licenses P5.1's cells as comparators for cells
produced by new code."* **That is the error, sitting in the text, while my requirement discussed it as
a future risk in the packet.** ⚠️ **I named the class and did not check the instance in front of me,
which is the inverse of *fix the class, not the sentence* and is the failure my own §7 rule was
written to prevent.** The implementer caught it and corrected the file.

### D4 🔒 AN APPROVAL MUST PIN A BLOB, NOT A COMMIT — MY OWN CITATION IS ALREADY DEAD

Amendment C approved *"`docs/plans/p5.2.md` @ `f02c917`"*. **The implementer then rebased onto that
amendment, and `f02c917` is no longer reachable from `HEAD`** (checked with `git merge-base
--is-ancestor`). **Measured: the blob is `d8a5da62…` at both `f02c917` and its rebased twin `6c7a049`
— the content I approved is intact and the container I named is gone.**
> ⚠️ **Our workflow GUARANTEES this collision: the coordinator amends the brief on `main`, the
> implementer rebases to see the amendment, and every commit hash the coordinator just cited dies in
> the same act.** **Rule, now in `PROJECT_PLAN` §7: any approval or ruling that pins a document pins
> `git rev-parse <ref>:<path>` — the blob — and names the commit only as a convenience.**
> **The approved plan is blob `d8a5da62`; the amendment is `2baf2cd`.**

### D6 🔒 WHAT A NEGATIVE C1 CONTROL ACTUALLY THREATENS — and the fallback, fixed NOW rather than after

⚠️ **I declared D the last round two paragraphs ago and this is one addition after that declaration.
The reason it cannot wait: the alternative is choosing a fallback AFTER seeing the control's outcome,
which is the loosening §7 forbids. Registering it now can only constrain us.**

The implementer will *"come back if the C1 control comes out negative, since that changes what the
packet can claim about the reused cells."* **The instinct is right and the target is slightly off, and
the difference decides what the fallback has to be.** A negative control does **not** weaken the
reused cells — those are sha256-verified outputs and their validity does not depend on whether the
trainer reproduces. **What it threatens is phase A's headline quantity.**

```
d1 = ATT(dt_spatial@1H) − ATT(dt_nomix@1H)     ← P5.1's cells, produced by the OLD trainer
d4 = ATT(dt_spatial@4H) − ATT(dt_nomix@4H)     ← produced by the NEW trainer
I  = d4 − d1                                    ← conflates HEAD COUNT with TRAINER CHANGE
                                                  unless old ≡ new
```

> **REGISTERED FALLBACK, before the control runs: if the same-device control shows `train_spatial_dt`
> does not reproduce itself, and the old-vs-new difference is not inside that measured envelope, then
> `d1` is RE-PRODUCED WITH THE NEW TRAINER — `dt_spatial@1H` and `dt_nomix@1H` retrained on the new
> code path, ~9.8 h — so that `I` is a within-code-path quantity.** In that branch P5.1's cells stay
> in the report as the **published comparator** and are reported **beside** the re-produced pair, with
> both values shown; they are not replaced and not discarded. ⚠️ **If the two differ materially, THAT
> is a finding about our own reproducibility and it is reported, not reconciled.**
> **Q0's stop rule and every §4 prediction are unchanged in either branch.**

### D2-CORRECTED 🚨 STEP 500 IS ONE STEP FROM THE **WORST** POINT IN THE RAMP — ADD STEP 249. **My number, my error.**

**Not a new review round: I am correcting a value I supplied.** The implementer measured D2's
discrimination honestly and reported it — quadratic `0.251`, square-root `0.7078`, cosine `0.501571`,
so a **5.7e−4** margin against a cosine — and drew the mild conclusion that this reinforces the `==`
bar. **It does, and the sharper reading is that I picked a bad point.** Measured by me over the whole
ramp today:

| step | linear | quadratic | square-root | cosine | \|cos − lin\| |
|---|---|---|---|---|---|
| **249** | 0.250000 | 0.062500 | 0.500000 | 0.146447 | **1.04e−01** |
| 332 | 0.333000 | 0.110889 | 0.577062 | 0.249547 | 8.35e−02 |
| **499** | 0.500000 | 0.250000 | 0.707107 | 0.500000 | **5.55e−17** ← the exact crossing |
| **500** | 0.501000 | 0.251001 | 0.707814 | 0.501571 | **5.71e−04** |

> 🚨 **The midpoint of a monotone ramp is where alternative shapes with the SAME ENDPOINTS CROSS — so
> it is the LEAST informative interior point, not the most.** Step 499 is blind to a cosine at
> **5.55e−17**; my step 500 escapes that only by the `+1` in `(step+1)/warmup`. **Step 249 is 181×
> more discriminating for free.**
> **REQUIRED: add step 249, expected `0.250000`. Keep step 500** — it costs nothing and it is a clean
> linear check. **Nothing else about obligation 6b changes.**
> ⚠️ **The transferable lesson, and it generalises past this test: when choosing probe points to
> separate two curves that agree at their endpoints, pick where they DIVERGE MOST, and MEASURE that
> rather than reaching for the midpoint.** §7 already requires a check to report its discriminating
> power; this is the same rule applied to *where you sample*, not just *whether you sampled*.

### D5 ✅ Verified in the amendment, first-hand

`2baf2cd` touches **only** the two documents — `git diff --stat main...HEAD -- offline/ tests/ agent/`
is **empty**, so no code exists and the registration genuinely predates it. C4's arithmetic checks:
`iql` and `bc_top10` touch **9 of 15** pairs (5 + 5 − 1), leaving exactly **6**. C2's λ values are
correct at every pinned step. §0.1 records the four changes and their direction, so the one edit the
registered set has had is visible and dated rather than silent.

---

## ⭐ AMENDMENT E — 2026-08-19. The C1 control came back NEGATIVE on CUDA. What that licenses, and what it does not

**The control is accepted as measured and it is a good measurement.** CUDA default: 63/66 and 61/66
tensors differ, losses diverge from step 8 and step 4. CPU: 0/66. CUDA + `use_deterministic_algorithms(True)`:
0/66. Cost of determinism **+10.3 %**, and the fixture is representative — **58.0 min/seed against
P5.1's measured 59, 1.7 % apart**, which I verified independently from P5.1's checkpoint mtimes.

### E0 🚨 THE FRAMING NEEDS ONE CORRECTION, AND IT CHANGES WHAT THE DECISION IS ABOUT

The three options are framed as a choice about **`I`'s cleanliness**. That is the smaller half.
> 🚨 **DETERMINISM DOES NOT MAKE THE METHOD STABLE. IT MAKES OUR NUMBERS REPRODUCIBLE.** If
> `dt_spatial`'s outcome is genuinely sensitive to bit-level perturbation, **that is a property of the
> method and a FINDING**, and switching on determinism does not remove it — it picks one arbitrary
> realisation and hides it.
> 🚨 **AND THERE IS A DEBT AGAINST A MERGED RESULT.** P5.1's headline, already merged, reads *"spatial
> mixing sometimes loses control, and how badly is UNSTABLE ACROSS SEEDS"* — treatment sd **30.36**
> against control **0.10**. **That sentence attributes the instability to SEEDS. A competing
> explanation now has a demonstrated mechanism and has never been measured.** P5.1's review already
> listed GPU determinism as could-not-verify; the control has now answered it in the direction that
> makes the question live.

### E1 🔒 THE ENVELOPE MEASUREMENT IS MANDATORY, INDEPENDENT OF THE REGIME CHOICE — and its design is not the one proposed

It is **not** a P5.2 cost. It is owed by P5.1's merged claim, and it is owed whichever regime P5.2 runs in.
> ⚠️ **The proposed design measures the wrong arm.** `dt_nomix` has per-seed sd **0.10** — replicating
> the stable arm gives the noise FLOOR and cannot discriminate anything. **The question lives in
> `dt_spatial`, sd 30.36.**
> **REQUIRED DESIGN: replicate BOTH arms at seed 202 — the seed with the largest per-seed effect
> (`d1 = +72.07`, against `{101: +1.03, 303: +68.50, 404: +32.75, 505: +23.47}`) — at the full 40,000
> steps under DEFAULT CUDA, evaluate both, and compare the replicated `d1(202)` against P5.1's
> +72.07.** ≈ 3 h. **That measures noise directly in the headline quantity's own constituent**, and
> replicating the *most* unstable seed is the highest-information single replicate: a small discrepancy
> there is strong evidence the floor is low; a large one is decisive the other way.
> **It is reported either way and it goes in the paper. No TSC paper I know of reports this envelope.**

### E2 ⛔ OPTION (b) IS REJECTED, and the reason generalises

Deterministic for P5.2's new runs only puts `d4` and `d1` in **different numerical regimes**, which
converts run-to-run noise into a **systematic offset** — worse than the noise it removes, because noise
is disclosable and a systematic is a confound. The implementer's own reading, and it is right.

### E3 ⚠️ A SEAM IN OPTION (c) THAT THE THREE OPTIONS DO NOT MENTION — and how it resolves

Under (c) the whole 2×2 is deterministic, **but the ladder's `mappo1000` column is REUSED from P5.1 and
is default-regime**, while `maxpressure` and `random` would be new and deterministic. **So (c) fixes the
regime across the head axis and opens one across the TIER axis.**
> **RULED, and it resolves cleanly because the two axes ask different questions:** **Q3 and every
> within-tier ranking use one regime per tier** — all arms at a tier are compared only with each other,
> so the seam does not enter them. **`mappo1000`'s rankings keep using P5.1's DEFAULT-regime cells,
> including its `dt_nomix`.** **The deterministic 1-head pair serves the INTERACTION `I` and nothing
> else.** ⚠️ **Q1's LEVEL predictions do cross tiers**, so they carry the regime difference as a small
> systematic — **bounded by E1's envelope and disclosed with the Q1 result.**
> ⚠️ **Note what this means for A3: under (c) the 1-head `mappo1000` DT cells exist TWICE, on purpose.**
> That is D6 operating as registered — both values reported, P5.1's as the published comparator — and it
> is a deliberate exception to A3, not a breach of it.

### E4 ⏸️ THE REGIME CHOICE IS THE AUTHOR'S — my recommendation is (c), and E1 runs first regardless

**(a)** 38.8 h, `I` carries unmeasured variance · **(b)** rejected · **(c)** ≈52.6 h, one regime and
reproducible.
> **Recommended: (c), with E1's 3 h first**, total ≈ 55.6 h. **The justification is a division of
> labour, not a preference: E1's measurement is the SCIENCE — it quantifies a sensitivity that is a
> property of the method — and determinism is the BOOKKEEPING that makes the reported cells
> reproducible.** Doing only (c) would hide the sensitivity; doing only E1 leaves the artifact
> irreproducible. ⚠️ **`DEFERRED` 51 lists five determinism claims a second machine could falsify and
> P10.0 owes a reproducibility section — (c) is the first result in this project whose cells would
> reproduce bit-exactly under a declared flag, and that is a paper asset we do not currently have.**

---

## ⭐ AMENDMENT F — 2026-08-19. The author defers the regime and changes the SEQUENCING. Accepted, with two technical caveats

### F1 ⏸️ RULING (the author's): the regime is NOT chosen now. E1 runs first and the ruling follows from a number

**Accepted, and it is the better call.** E1 measures precisely the quantity that separates (a) from
(c) — replicated `d1(202)` against **+72.07**. **Close → default-CUDA noise in the headline quantity is
measured and small, (a)'s *"unmeasured variance"* becomes a bounded and disclosed one, and (c) buys
13.8 h of nothing. Far → (a) is untenable and (c) is the only way phase A resolves at five seeds.**
**Deferring costs nothing:** `train_tier_dt` does not exist, so the regime is a launch parameter and
not an architectural fork. **Build it with `deterministic: bool`.**

### F2 ⚠️ MY E4 RECOMMENDATION CONTRADICTED MY OWN E0, AND THE AUTHOR CAUGHT IT

E0 says *"switching determinism on does not remove the sensitivity — it picks one arbitrary
realisation and hides it."* **E4 then recommended switching it on.** The author's sharpening stands and
goes into the post-E1 recommendation: **a MEASURED nondeterminism envelope is a stronger
reproducibility contribution than a suppressed one — nobody in this literature reports the envelope and
everybody reports the flag.** ⚠️ **The counter-argument is real and must be weighed beside it: a large
envelope may make the head-count question unanswerable at five seeds without (c).** **Both go in the
recommendation I bring after E1; neither is assumed now.**

### F3 ⛔ "RUN THE 2×2 UNDER BOTH REGIMES" IS REJECTED NOW, so it does not return later

**The author's reason, and it is correct:** default-vs-deterministic conflates the **regime effect**
with **a single draw of the noise**, and `n = 1` cannot separate them — ~11 h for a number nobody can
interpret. **What adds information is more replicates WITHIN default**, and that is a call to make
after E1's first one.

### F4 🔀 CONTINGENT REALLOCATION, declared before the measurement that triggers it

> **If E1 comes back SMALL, the 13.8 h that (c) would have cost goes to the pre-declared `fixedtime`
> fourth rung instead** (B1 already registered it, so this is not post-hoc). **The author's reason, and
> I am adopting it rather than merely accepting it: three matched tiers strengthen C1, which is the
> question the paper asks; numerical-regime robustness strengthens our bookkeeping, which no referee at
> this venue is asking about.** ⭐ It also fixes the ladder's spacing weakness disclosed in B1 —
> `160.33 · 167.49 · 206.93 · 257.73` is a graded ladder where `160 · 167 · 258` is two rungs and a gap.

### F5 🔒 BINDING ON E1's WRITE-UP — it is owed by MERGED work, so it lands with the correction

**Whatever E1 returns, P5.1's headline sentence is corrected or corroborated IN THE SAME COMMIT.** It
reads *"how badly is UNSTABLE ACROSS SEEDS"*, treatment sd **30.36** against control **0.10** — an
attribution to seeds, while a second mechanism now has demonstrated existence and no measurement.
> ⭐ **AND ONE NUMBER THE AUTHOR SUPPLIED THAT I VERIFIED AND FOUND SHARPER THAN STATED. The published
> CI `[+36.05, +43.08]` is over DRAWS. Over SEEDS the same five values give `[+1.88, +77.25]`**
> (t, df = 4; the normal form gives `[+12.96, +66.17]`) — **10.7× wider, with a lower bound of +1.88
> that barely clears zero.** **The DIRECTION is untouched — 5/5 seeds agree in sign and 0/100 draws are
> won — but the MAGNITUDE's confidence over the seed dimension is far weaker than the published
> interval suggests.** ⚠️ **This is P4.7's M1 recurring on a headline result** (*our per-draw unit
> averages the seeds away*), it is `P8.1`'s registered work, and **it must be reported beside the
> per-seed spread `{101: +1.03 … 202: +72.07}`, 70×.**

### F7 🔒 E1 REPORTS A PAIRED CI, NOT TWO NUMBERS — added 2026-08-19, after obligation 6 discharged

E1 is currently specified as *"compare the replicated `d1(202)` against +72.07"*. **Two point estimates
cannot be judged: `72.07` against `68.5` and `72.07` against `20.1` look the same on the page until the
per-draw scatter is known.**
> **REQUIRED: report the PAIRED PER-DRAW difference between the replicate and P5.1's own seed-202
> cells, over the shared draw ids, with `mean_ci95` — for `dt_spatial`, for `dt_nomix`, and for `d1`
> — and state for each whether the interval excludes zero.** ✅ **The data exists and I verified it
> today: `output/p5_1/eval_dt_spatial.json` and `eval_dt_nomix.json` carry 100 seed-202 episodes each,
> keyed by `draw_id` and `seed`, so this is arithmetic over committed data and costs no compute.**
> 🚨 **If the arm-level interval excludes zero, two independent training runs of THE SAME CODE AT THE
> SAME SEED produced measurably different policies. That is the finding, and it is reported as one.**

⭐ **AND THE ATTRIBUTION IS NOW CLEAN, WHICH IS OBLIGATION 6'S RETURN ON ITS OWN COST.** Because the new
trainer reproduces `train_spatial_dt` **byte-exactly at one head on CPU** — identical loss sequence, all
66 tensors — **any difference E1 measures on CUDA is attributable to DEVICE NONDETERMINISM ALONE and
not to the trainer change.** Without obligation 6 the two explanations would be inseparable and E1
would be uninterpretable. **State this in E1's write-up: it is what makes the number mean anything.**

### F6 ⚠️ TWO TECHNICAL CAVEATS ON `deterministic: bool`, so a late failure does not kill a campaign

**(a) It is NOT a per-call parameter underneath.** `torch.use_deterministic_algorithms(True)` is
**process-global**, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be exported **before the CUDA context is
created** — an environment variable set at process entry, not a mid-run toggle. **`deterministic: bool`
is the right INTERFACE; its implementation must act at process entry, and `p5_2.sh` must export the
variable.** A flag flipped after CUDA is initialised can silently fail to take effect.
**(b) Determinism is verified for the two DT arms ONLY.** `use_deterministic_algorithms(True)` **RAISES**
on any op lacking a deterministic implementation. **If (c) is chosen, it must be proved for `bc`,
`bc_top10`, `bc_top10_perix`, `iql` and the EVALUATION path before the campaign starts** — in Gate 0's
precondition block, beside D1(a)'s path check, where it costs seconds instead of failing ten hours in.

---

## 🛑 AMENDMENT G — 2026-08-19. **E1's LAUNCH IS HELD.** `CUBLAS_WORKSPACE_CONFIG` puts E1 in a regime P5.1 never ran in

**The objection is correct on every factual point and I verified all three first-hand:**
`offline/campaigns/p5_1.sh` exports **`OMP_NUM_THREADS` and `MKL_NUM_THREADS` only** (lines 90–91);
**`CUBLAS` appears nowhere else on `main`**; and P5.1's captured launch line reads
`export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and nothing more. **So `d1(202) = +72.07` — the single
quantity E1 exists to compare against — was produced WITHOUT that variable.**

### G1 🚨 THE DEFECT, AND IT IS OPTION (b) ARRIVING THROUGH A LINE LABELLED "harmless"

`p5_2.sh` sets `export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}` unconditionally,
commented *"harmless in the default regime"*. **It is not neutral: it is part of the determinism
recipe** — it fixes the cuBLAS workspace and thereby constrains GEMM kernel selection between runs.
> 🚨 **E1 exists to measure the run-to-run envelope OF THE REGIME P5.1 RAN IN. A replicate carrying a
> variable P5.1 lacked conflates the noise we want with a systematic effect of the cuBLAS
> configuration — which is EXACTLY the pathology `E2` rejected as option (b), reached through a
> comment rather than a decision.**
> ⚠️ **And the direction of the error is the damaging one: the variable SUPPRESSES variance, so E1
> would report an envelope SMALLER than the real one, the regime ruling would go to (a) on a flattering
> number, and P5.1's published figures would still carry the larger, unmeasured envelope.**

### G2 ✅ THE FIX IS FREE AND DOES NOT DEPEND ON SETTLING THE MECHANISM

> **RULED: E1 runs in P5.1's exact environment — `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
> `CUBLAS_WORKSPACE_CONFIG` UNSET.** `p5_2.sh` sets it **only when `--deterministic` is requested**,
> never unconditionally, and the launch block drops the export. **F6(a) is unchanged and still correct
> for the deterministic regime; what was wrong was applying it to a measurement of the default one.**
> **Removing it costs nothing and removes the question entirely, whatever the mechanism turns out to
> be** — which is why this is ruled rather than researched first.

### G3 🔬 THE MECHANISM IS STILL OWED, AND MY OWN ATTEMPT TO SETTLE IT HAD NO DISCRIMINATING POWER

The open question — *did the C1 control's "CUDA default" arm carry the variable?* — is not answerable
from the reports. **I tried to settle it directly and failed, and the failure is worth recording rather
than hiding:** I ran a 60-step GEMM-heavy CUDA model twice per condition and got
**`0/10` tensors differing WITH the variable and `0/10` WITHOUT it.** ⚠️ **That is not evidence the
variable does nothing. It is a NULL INSTRUMENT: my probe never exhibited nondeterminism at all, so it
cannot distinguish the two conditions**, and reporting its agreement as a result would be §7's
discriminating-power rule broken by the person who enforces it. **The real model shows 63/66; mine
shows 0/10; only an instrument that reproduces the phenomenon can test what suppresses it.**
> **REQUIRED, on the harness that already exhibits it — minutes, not hours: re-run the C1 control's
> DEFAULT arm twice, once with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and once with it unset, and report
> both tensor-difference counts.** **If unset differs and set does not, the variable suppresses
> variance here and G1 was a live defect caught before it cost anything. If both differ alike, it
> suppresses nothing on this model and G2 is a cleanliness measure — say which, in the packet.**
> ⚠️ **Also state, from the C1 control's own environment, whether its "default" arm carried the
> variable.** If it did, the C1 numbers themselves are numbers about a regime P5.1 never ran in, and
> that is a disclosure the packet owes independently of E1.

---

## ✅ AMENDMENT H — 2026-08-19. **REGIME RULED: (a) DEFAULT CUDA, FOUR TIERS.** E1 is closed; phase A may start

### H1 ✅ THE RULING (the author's), recorded with its reasoning rather than reconstructed later

> **Phases A/B/C run under DEFAULT CUDA. The tier set is FOUR: `mappo1000` (reused), `maxpressure`,
> `random`, `fixedtime`.** The pre-declared contingency (B1, F4) fires as written.
> **Reasoning on the record: (c) had exactly two stated advantages and E1 removed both.** `I` carries
> **zero** run-to-run variance — measured, not bounded — and the reproducibility benefit is **void**,
> because the reported cells already reproduce bit-exactly without determinism; determinism would
> reproduce the **weights**, and **no number in the paper depends on the weights.** *"Spending 13.8 h to
> make a quantity reproducible that is already reproducible is not a trade I would defend to a
> referee."* **Cost, from A4's corrected arithmetic: ≈53 h.**

### H2 🚨 REGISTERED SCOPE LIMIT — THE ZERO ENVELOPE IS A `mappo1000` MEASUREMENT, AND MARGINS ARE TIER-DEPENDENT

**The author's, and it closes a gap in my own reasoning: I stated the mechanism — a 1e−04 perturbation
stayed below every decision margin — and did not draw out that MARGINS ARE A PROPERTY OF HOW
CONFIDENTLY THE LEARNED POLICY SEPARATES PHASES.** A model trained on the `random` tier's corpus has
**no guarantee of equally wide margins**, and phase B goes there.
> **REGISTERED: the envelope is exactly 0.0000 for `grid4x4@mappo1000`, seed 202, ONE replicate, and it
> is NOT ESTABLISHED FOR THE DEGRADED TIERS.** Every use of the number carries that limit.
> **And the qualification already written travels with it: *envelope = 0* is a statement about the
> METRIC as much as about the training — sub-margin perturbations are invisible, and a larger
> perturbation or a draw nearer a margin would flip DISCONTINUOUSLY. It does NOT mean the optimisation
> is stable.**

### H3 📋 WORK REQUIRED BEFORE PHASE B, because a fourth tier changes a REGISTERED DENOMINATOR

Adding `fixedtime` is not just another cell set — **it changes `Q1`'s aggregate criterion, and moving a
threshold after data exists is loosening.** All of this lands **before phase B runs**:
1. **`fixedtime`'s predicted cells**, computed by the **unchanged rule R′** from the same on-disk
   quantities (`ladder(grid4x4, fixedtime) = 206.9318`), added to §4.0.2's table.
2. **The out-of-sample set N grows from 13 to 19** (six more method cells). **Restate `Q1`'s threshold
   by a STATED PRINCIPLE, not a fresh judgement** — 9/13 is 69.2 %, so name the rule that carries it to
   N = 19 and apply it. **Fix it in the commit; it may not move afterwards.**
3. **`Q2`'s rank prediction for `fixedtime`**, and the **hard-subset** count restated for the new tier.
4. **`Q3`/`Q4`'s per-tier statements** extended to four tiers.
⚠️ **All of it is a function of quantities already on disk, so it is registrable now and post-hoc after
phase B. Do it in the same commit that adds the tier.**

### H4 📌 DISCLOSURES CARRIED FORWARD — neither affects the ruling, both would look like concealment if found later

1. **The provenance `tier` label changed** from `grid4x4_mappo1000` (P5.1) to `mappo1000` (P5.2) — same
   tier, different label, in checkpoints that sit side by side.
2. **The `+10.3 %` determinism timing was measured with `CUBLAS_WORKSPACE_CONFIG` SET IN BOTH ARMS**, so
   it is the **flag's cost GIVEN the variable**, not the flag's cost alone.
3. **The E1 verification chain belongs in the packet as its own paragraph** — the author's instruction:
   the zero was treated as a suspected defect *because it contradicted our own C1 control*; the
   checkpoints were checked; **a file hash was found to carry provenance rather than weights
   (`DEFERRED` 29)**; a canonical `state_dict` digest was computed instead; **every recorded field was
   checked rather than ATT alone**; and **a positive control proved the detector resolves 1e−12.**
   ⚠️ **Without that last step the result would have been indistinguishable from a self-comparison, and
   F7 did not require it.**

---

## ✅ AMENDMENT I — 2026-08-19. The random-tier envelope is UNCONDITIONAL, and a non-zero result is registered as a FINDING

### I1 ✅ RULED (the author's): the replicate becomes unconditional, in phase B

> **One arm-seed is replicated on the `random` tier as part of phase B — not triggered by anyone
> judging that a result *"turns on a small margin"*.** The author's own reason for overturning his
> earlier version, recorded because it is the better half of the exchange: **a trigger that depends on
> someone judging a result they already hold is worse than no safeguard, because it reads as one in the
> plan while firing on attention.** ≈1.1 h, **8 % of what the regime ruling freed**, and it converts an
> extrapolation a referee will press into **a measurement at both ends of the data-quality axis**.
> **Arm: `dt_spatial`**, matching E1's logic — the arm that exhibits the phenomenon, where `dt_nomix`'s
> sd of 0.10 measures only the floor. **Same F7 machinery, same positive control proving 1e−12
> resolution**, because that control is what separated E1's zero from a self-comparison.

### I2 🔒 REGISTERED BEFORE THE NUMBER EXISTS — A NON-ZERO ENVELOPE IS A FINDING, NOT A FAILURE

> **Registered now so that neither outcome can be re-read afterwards.** A non-zero envelope on the
> `random` tier would mean **run-to-run nondeterminism propagates to the metric when the learned policy
> separates phases less confidently** — the decision-margin mechanism, **measured rather than argued,
> and a dependence nobody in this literature reports.**
> ⚠️ **So: a non-zero result is NOT a defect in the campaign, and a zero result is NOT the only
> acceptable outcome. Both are publishable and this sentence says so in advance** — which is exactly
> what `PREREGISTRATION` §10 asks of every outcome branch.

### I3 🔢 THE Q1 THRESHOLD RULE, NAMED AND APPLIED MECHANICALLY — with one figure corrected

> **THE RULE, fixed in the commit and never re-derived: `k = ceil(9/13 × N)`** — the smallest integer
> count whose proportion is at least the registered rate.
> ✅ **It SELF-CHECKS, which is why it is the right rule rather than a plausible one: at `N = 13` it
> returns `k = 9`, reproducing the registered threshold exactly.** So it is a faithful carry-across and
> not a new criterion wearing the old one's clothes. **At `N = 19` it returns `k = 14`** (14/19 =
> 73.68 %; 13/19 = **68.42 %**, which is below the registered 69.23 % and therefore excluded).
> ⚠️ **One figure in the ruling as issued is a slip and is corrected here so it is not copied forward:
> it read *"73.7 % at 13/19 is too low"*. **13/19 is 68.42 %**, not 73.7 %; 73.68 % is 14/19. **The
> conclusion — 14 of 19 — is unaffected and correct.**

---

## ✅ AMENDMENT J — 2026-08-19. **BOTH ARMS, SEED 202, PRE-DECLARED.** The last open item; phase A may start

### J1 ✅ RULED: both arms on the `random` tier, and the SEED is declared now — the catch was the author's

> **Both arms, so the random tier yields `d1` — the SAME QUANTITY E1 reported at `mappo1000`.** The
> author's reason for overturning his own single-arm ruling: *"a single-arm envelope at one end and
> `d1` at the other under one sentence is two things wearing one claim"*, and the fallback it would
> have needed — replicate `dt_nomix` too if `dt_spatial` comes back non-zero — **reinstates the
> conditional trigger removed an hour earlier.** ≈2.2 h, 16 % of what the regime ruling freed.

🚨 **AND THE THING NOBODY HAD NAMED, WHICH IS THE AUTHOR'S AND IS A REAL HOLE: WHICH SEED.**
**E1 chose 202 because it carried the largest per-seed `d1`. That principle CANNOT be applied on the
`random` tier, because the per-seed spread there does not exist until phase B has run.** So any rule of
the form *"replicate the seed with the largest effect"* is **post-hoc selection on a quantity related
to the one being measured** — it would let the envelope be measured on the seed most, or least, likely
to show one, chosen after seeing which is which.
> **DECLARED NOW: seed 202, both arms, `random` tier.** **It is arbitrary in the sense that 202 has no
> intrinsic property on a tier it has not run — and that is exactly why it is the right choice.** It is
> pre-declared, and it makes the two envelope measurements **the same seed at both ends of the ladder**,
> which is the strongest form of the paired comparison the sentence claims.
> ⚠️ **If phase B later shows 202 is unrepresentative of the `random` tier's per-seed spread, that is a
> DISCLOSURE, not a reason to re-select.**

### J2 🔒 THE REPLICATE MUST BE PROVED INDEPENDENT BEFORE ITS ZERO MEANS ANYTHING

**Derived from what actually happened today: E1's `+0.0000` was only interpretable because I checked BY
HAND that the two checkpoints differed — 66/66 tensors, worst 1.22e−04. F7 did not require that, and
without it the result was indistinguishable from a self-comparison.** The machinery must now do what I
did manually.
> **REQUIRED, three parts.** **(a)** The replicate is an **independent training run from scratch** at
> the same seed — never a re-evaluation of phase B's checkpoint, which returns zero **by construction**.
> **(b)** Its artifacts are written under a **distinct key** so no cache, resume branch or report path
> can serve phase B's cell in their place. **(c) Before any envelope is reported, the machinery asserts
> the two runs' CANONICAL `state_dict` DIGESTS DIFFER** — sorted-key tensor bytes, **not** the file
> sha256, which today was shown to differ on provenance alone (`git_commit`, `deterministic`, `n_head`,
> a changed `tier` label) while saying nothing about weights.
> 🚨 **EQUAL digests are a REFUSAL, not a zero.** They mean either two identical models were compared —
> a defect — or training reproduced exactly, which would contradict the C1 control and invalidate the
> measurement's premise. **Either way it forces a look instead of silently reporting the answer the
> question was asked to avoid.** **Report the digests beside the envelope.**

---

## ✅ AMENDMENT K — 2026-08-19. Phases A/C/B approved to launch, with ONE mechanical anchor on the deferred replicate

### K1 ✅ VERIFIED BEFORE APPROVING ~50 h, in the artifacts

**All six `fixedtime` cells reproduce exactly** from my own implementation of the unchanged rule R′
(`T = 206.93176498413087 / 160.33195198059082 = 1.290646`): `dt_nomix` **204.94** < `bc` **222.11** <
`bc_top10_perix` **230.60** < `dt_spatial` **256.31** < `iql` **360.85** < `bc_top10` **1042.75**.
**Q4b now has four graded rungs — `+39.56*` → `+41.74` → `+51.37` → `+62.71` — a real monotonicity
test rather than a two-point line**, which is what the fourth tier was reallocated to buy.
✅ **`k = ceil(9/13 × N)` → 14 of 19, with the `N = 13 → k = 9` self-check recorded.** ✅ **D2's probe
landed: `LR_PROBE_STEPS = (0, 249, 500, 999, 1000, 1001, 40000)` with step 249 named as the
discriminating point.**
⭐ **AND A CONCERN I RAISED AND WITHDREW, recorded because the answer is the interesting part.** The
shell's resume conditions are bare existence tests — `[ -f cell.json ]` and a checkpoint count of 5 —
which would be D1(c)'s exact failure **if the writer were not atomic. It is:** `os.replace` behind the
barrier, training to `<name>.partial` then `replace_guarded`. **A crash therefore leaves a `.partial`,
never a truncated artifact at a final name, so an existence test is sufficient BY CONSTRUCTION.**
**D1(c) is satisfied at the WRITER, which is the right place — a reader-side check would have had to
re-validate every artifact on every resume.**

### K2 🔒 THE DEFERRED REPLICATE NEEDS A MECHANISM, NOT A PROMISE — and it costs nothing

The implementer's reason for leaving I1/J1 out of this run is **correct and I am accepting it**: the
replicate compares against phase B's own `random`-tier cell, and **inventing that wiring while the
ladder runs is precisely how a replicate ends up re-evaluating the cell it is supposed to be
independent of — the zero-by-construction J2 exists to refuse.** Flagging it visibly rather than
burying it is the right call.
> ⚠️ **But *"I will wire it before the ladder finishes"* is a promise, and this project's whole method
> is that a step depending on someone remembering is the weak kind.**
> **REQUIRED, and it is zero new code: enumerate the two replicate cells in
> `docs/data/p5_2_declaration.json` NOW.** The campaign's final completeness assertion already
> **derives its expected cells from the declaration rather than from the files being checked**, so
> listing them makes a campaign that lacks them **refuse to report itself complete, automatically.**
> ✅ **Legitimate as a registration act rather than a change: I1/J1 were RULED into the design today,
> before any phase-B number exists, so this completes the registration rather than editing it.**

---

## ⭐ AMENDMENT O — 2026-08-20, 13:20, mid-campaign. The cost question is ANSWERED and `fixedtime` STAYS; and the ladder's RUN ORDER makes three registered cut rules unexecutable

**Written while `maxpressure`/`dt_spatial` seed 101 is ~16 min into training, deliberately before it
finishes: O3 is cheapest to act on at this exact moment and gets more expensive every minute.**
**O1 and O2 answer the open question I owed the author. O3 is a defect I found by reading the campaign
script rather than its header, and it is not the question he asked — it is the one underneath it.**

### O1 📏 THE MEASURED PER-TIER COST — the number the cut decision was waiting on

Measured by me today from **checkpoint cadence, not log deltas** (`NOTE M3`: the ≈2 h 10 m forward
clock jump after 07:59:54 inflates exactly one interval, and I subtracted **130 min** from the single
interval that spans it — `dt_nomix_h4` seed 505, `07:59 → 11:25`, 206 → 76 min):

| stage | measured | n |
|---|---|---|
| 🚨 DT arm-seed, 40,000 steps — **4-HEAD ONLY, mislabelled here as the population (P4)** | `66 · 59 · 68 · 70 · 80` then `83 · 81 · 82 · 81 · 76` — ~~**mean 74.6 min**~~, range **59–83**. **The ladder is 1-HEAD at ≈57 min** | 10, all `_h4` |
| baselines, 4 arms × 5 seeds | **53 min** (`11:53 → 12:46`) | 1 tier |
| evaluate one DT cell, 5 seeds × 100 draws | **28 min** (twice, `02:04→02:32` and `11:25→11:53`) | 2 |
| evaluate one baseline cell | **18 min** (`12:46 → 13:04`) | 1 |
| evaluate `behaviour` | **56.5 min** (P5.1's log, unchanged) | — |

🚨 **EVERY FIGURE IN THE BLOCK BELOW IS WITHDRAWN — SEE AMENDMENT P4. Corrected: a ladder DT arm-seed
is ≈57 min (1-head), A FULL TIER IS ≈13.5 h, ≈40–42 h remain, and cutting `fixedtime` saves ≈13.5 h =
0.56 days = 2.1 %.** The block stands as the record of what was claimed.

> ~~**A FULL NEW TIER = 16.4 h** — 12.4 DT training (10 arm-seeds × 74.6 min) + 0.9 baselines + 3.1
> evaluation (7 cells). Against the plan §2.5 projection of **13.9 h**: **+18 %**.
> **Remaining as scheduled: `maxpressure` 16.4 + `fixedtime` 16.4 + `random` 15.5 (no behaviour anchor,
> it is the shared gate-1 cell) + the I1/J1 replicate 2.7 = ≈ 51 h**, i.e. ending **≈ 2026-08-22 16:00**.
> **Cutting `fixedtime` saves 16.4 h = 0.68 days.**~~

🚨 **THE 74.6 min FIGURE AND ITS CAVEAT ARE BOTH WITHDRAWN, 2026-08-20, BY THE AUTHOR. SEE AMENDMENT
P4. ALL TEN OBSERVATIONS ARE 4-HEAD CELLS AND THE ENTIRE LADDER IS 1-HEAD** — a number true of a
sample, stated as the population's, **inside the amendment that answers a costing question**. The
withdrawn caveat follows, and it is worse than no caveat because it named the wrong confound and
therefore gave false assurance that the confound had been considered.

⚠️ ~~**What this measurement does NOT settle, stated so it is not read as tighter than it is: 74.6 min is
an ARM-LEVEL MEAN spanning a cool start and a throttled steady state (`NOTE M4`).** The GPU reads
**56 °C** as I write this against the overnight throttled regime, so `maxpressure`'s own seeds may land
nearer 59 than 83. **The mean of a full 10-seed phase is the right planning unit precisely because a
fresh tier reproduces that same cool-start-to-throttled shape**~~ — but the first three `maxpressure`
checkpoints will replace this estimate with a same-tier measurement, and the packet quotes those.
**That last clause is the only part that survived contact, and it is what produced the correction.**

### O2 ✅ RULED: `fixedtime` IS NOT CUT. The saving is ~~2.6~~ **2.1 %** of the budget (corrected in P4) and the rung is the most falsifying one left

**This is the question the author asked me to bring back once phase B gave a real per-tier time. It has
an answer and the answer is no** — and it is a ruling rather than an escalation because **keeping four
tiers is the STATUS QUO under `H1`; cutting is the change.**

1. **The saving is negligible against the constraint it would be protecting.** ~~0.68~~ **0.56 days
   (P4)** against
   **41 days to the end-of-September target** (`PROJECT_PLAN` §10), of which ~2 weeks are the writing
   reserve, leaving ≈27 experimental days. **~~16.4 h is 2.6 %~~ 13.5 h is 2.1 % of that (P4).** The cut contingency was written
   when the tier cost was an extrapolation; the measurement has made the risk it hedged against small.
2. **`fixedtime` is the rung most likely to falsify us, and that is why `B1`/`F4` chose it.** It is
   *"the tier that broke §1b R3's monotonicity and flipped R2's sign on hz1x1"*. **Cutting the rung
   selected for its power to break our own pattern, to buy 2.6 % of the budget, is the trade this
   project exists not to make.**
3. **It is the difference between a monotonicity test and a line.** `K1` measured the four predicted
   rungs at **+39.56\* → +41.74 → +51.37 → +62.71**. Three points test monotonicity; **four are the
   first that can show a KINK** — which is `N2`'s own lesson about the head axis (*two points determine
   a line by construction*) applied one axis over.
4. **Its six cells are already registered** — `H3`/`I3` grew `N` from 13 to 19 and fixed `k = 14` by the
   `ceil(9/13 × N)` rule. Dropping them is mechanically clean under that rule, so **the objection to
   cutting is not procedural**; it is that we would be paying a real scientific price for a saving we
   do not need.

> ⚠️ **The trap this ruling is checked against, and I checked it explicitly:** §10's sequencing ruling
> warns that *"whichever task is running will look like the one that must finish"*. **The test is
> whether I would ADD this tier today at ~~16.4~~ **13.5** h if it were not already scheduled. I would — for
> reasons 2 and 3, which are about what the tier measures and not about it being underway.** That is
> what separates this from sunk cost.

### O3 🚨 THE LADDER RUNS `maxpressure → fixedtime → random`, AND FOUR REGISTERED STATEMENTS SAY IT MUST NOT

**Found by reading `offline/campaigns/p5_2.sh` rather than its header comment.** Verified in the
artifacts, both sides:

| where | what it says |
|---|---|
| `offline/campaigns/p5_2.sh:345` | `for LADDER_TIER in maxpressure fixedtime random; do` |
| `docs/plans/p5.2.md:861` (**the plan's registered phase order**) | `PHASE B maxpressure, then random: 6 method arms + the tier's behaviour anchor` |

**The plan's §5 phase-order block was never updated when `H1` added the fourth tier** — `911796e`
updated §2.5's cost table and `H3`'s registration and left §5 standing. **So the executing order is
not the registered order, and it is not registered anywhere else in the plan** (checked: §5 line 861 is
the only run-order statement; line 661's *"in measured-ATT tier order"* governs how `Q4b` is
**reported**, not what runs when).

🚨 **The consequence is not cosmetic. Four registered statements make `fixedtime` the expendable rung
and `random` the load-bearing one, and the order makes every one of them unexecutable:**

- **`B1`:** *"`fixedtime` … IS PRE-DECLARED NOW AS THE FOURTH TIER, **to be run only if phase B
  finishes with budget left**"* — it is scheduled **inside** phase B, so that condition can never be
  evaluated.
- **brief §5:** *"If it overruns, **drop to `mappo1000` + `random`** — the endpoints carry the
  interaction; the middle does not."* — unreachable: at every moment before hour ~33 we hold the two
  middles and not the endpoint.
- **`NOTE M5` (written yesterday, AFTER `H1`):** *"`fixedtime` is the declared fourth rung and
  therefore **the first thing to drop if the calendar binds**."*
- **`H1` itself** lists the tier set as *"`mappo1000` (reused), `maxpressure`, `random`, `fixedtime`"*.

> 🚨 **AND THE ASYMMETRY IS THE REAL DAMAGE, not the inconsistency.** Under this order the only cut
> ever available is `random`, and `random` carries **(i)** the endpoint of the data-quality axis,
> **(ii)** §1b's R2/R6 load-sorter scope condition at its most extreme, and **(iii)** the **entire
> I1/J1 replicate**, which `K2` wired into the completeness assertion as an **unconditional, mandatory**
> deliverable. **A truncation under the current order costs a REGISTERED MANDATORY item; under
> `maxpressure → random → fixedtime` it costs a PRE-DECLARED OPTIONAL one.** That is the whole
> argument, and it is worth ≤30 minutes.

**What it costs to fix, measured rather than guessed:**

- **One token: `maxpressure fixedtime random` → `maxpressure random fixedtime`.** I verified the change
  is a pure scheduling change: per-tier declarations are generated inside the loop, `random`'s
  `behaviour` skip resolves to the **gate-1** anchor which already exists, and the campaign declaration
  and completeness assertion enumerate cells and are order-independent (`tier_sweep.py:2122` resolves
  each declared cell to the reuse root or the work dir by its own `source` field).
- **Resume is verified in production ON THIS CAMPAIGN, not assumed:** `campaign.log` records **two**
  launches — `16:02:06` and `20:21:53` on 2026-08-19 — and the second correctly logged
  `SKIP E1 training dt_spatial seed 202: checkpoint already on disk` for four completed E1 cells.
- ⭐ **NOW IS THE CHEAPEST MOMENT AND IT IS CHEAPEST BY CONSTRUCTION.** `maxpressure` has **zero**
  checkpoints on disk, so there is nothing to skip and no partial training record. **A kill costs only
  the in-flight seed** (~30 min as I write). **Ten minutes into seed 3 it would cost that plus O4's
  defect firing.**
- ⚠️ **The script must NOT be edited while `bash` is executing it** — `bash` re-reads by byte offset and
  a shifted offset executes garbage. **Kill first, then edit, then relaunch.** A relaunch re-runs the
  gates in **~15 s** (the random anchor re-roll is skipped on its existing artifact).

> ⏸️ **THE CALL IS THE AUTHOR'S, because it costs him a kill and a relaunch of a 51-hour job.** My
> recommendation is to reorder. **What is NOT in question either way: `docs/plans/p5.2.md:861` is stale
> and must be corrected to the four-tier order with its reason, in the implementer's next commit,
> whichever order we run.** A plan that names a two-tier phase B while a four-tier one executes is the
> artifact-versus-description error in our own registration.

### O4 🔎 A LATENT DEFECT THAT A MID-ARM RESUME MAKES LIVE — for the implementer, not for tonight

`tier_sweep.py:1739-1747` writes `training_{tier}_{method}.json` from **`records`, the seeds trained in
THIS invocation**, while `:1712-1714` skips seeds whose checkpoint already exists. **So a training run
resumed mid-arm silently overwrites the arm's training record with a SUBSET** — the skipped seeds'
`final_loss` and `seconds` are lost, and **nothing reads the file** (grepped), so no completeness
assertion or guard can notice.

✅ **Not live today, verified rather than assumed:** every training artifact on disk is complete —
`dt_spatial_h4` and `dt_nomix_h4` carry **5 runs each** with seeds `[101, 202, 303, 404, 505]`,
`baselines` carries **20**, and E1's two single-seed files carry **1** each, as designed.
⚠️ **It fires on the first mid-arm resume, and the campaign advertises resumability.** `NOTE L(a)` and
`M3` both quote per-seed `final_loss` and durations, so it is provenance the packet uses.
> **REQUIRED, in the implementer's next commit, not by interrupting the campaign: merge by
> `(tier, method, seed)` into any existing record rather than overwriting it, and test the resume case
> — write 3 seeds, resume, assert the file holds 5.** ⚠️ **`_run_train`'s per-seed skip branch has been
> READ by me and has never been OBSERVED to execute** (the two production resumes both skipped at whole
> -arm granularity). **State that in the packet rather than crediting it as exercised.**

### O5 📌 DISCLOSURE OWED BY PHASE C — `A3` says only new arms run at `mappo1000`; four did

`p5_2.sh:336` calls `train-baselines` at `mappo1000` with no arm filter, so **`bc`, `bc_top10`,
`bc_top10_perix` and `iql` were all trained there — 20 checkpoints, ~53 min** — while `:338-340`
evaluates **only `bc_top10_perix`.**
✅ **`A3`'s SCIENTIFIC content is intact and I checked the mechanism, not the intention: no second
value for a merged number can be produced, because the report resolves every reused cell through the
declaration's own `"source": "reuse_root"` field** (`tier_sweep.py:348, 2122`), never by globbing
checkpoints. **The 15 unused checkpoints are inert for reporting.**
> **Disclose it in the packet in these terms: `A3`'s wording is *"only NEW arms are run at that tier"*
> and three old arms were TRAINED there and not evaluated.** ⚠️ **The reason to write it down is not
> the 53 minutes: 15 checkpoints now sit in `output/p5_2/checkpoints/` named
> `grid4x4_mappo1000_{bc,bc_top10,iql}_seed*.pt`, i.e. exactly what a future glob-based reader would
> find and mistake for P5.2 cells at a reused tier.** A hazard that is documented is a hazard; an
> undocumented one is a trap for whoever writes the release package.

---

## ⛔ AMENDMENT P — 2026-08-20, 15:05. **THE AUTHOR REFUSES THE INTERRUPTION.** Three of my numbers are wrong, one of my claims is false, and the ruling is better than the escalation that prompted it

**Amendment O asked for a kill and a reorder. The answer is no, and on checking, the escalation should
not have been made in the form I made it.** ⚠️ **What follows corrects me in four places. O2's ruling
— `fixedtime` stays — is the only part that survives intact, and it survives cheaper than I priced it.**

### P1 ⛔ THE RULING (the author's), recorded in his words rather than paraphrased

> **"I am not interrupting the campaign. The reorder does not happen; phase B runs
> `maxpressure → fixedtime → random` as the script stands. The reasoning is the risk, not the compute.
> A resume now is a MID-ARM resume, which is exactly the `O4` case you found today and which is not yet
> fixed. Killing a healthy nineteen-hour run to buy a scheduling property is a trade I am not making
> while the fix for the failure mode it would trigger is still pending."**

⭐ **The argument is stronger than mine and it uses my own finding against my own recommendation.** I
reported `O4` — a mid-arm resume silently truncates the arm's training record — **and then, in the same
message, recommended an action whose entire cost model assumed a resume.** I priced the kill at *"the
in-flight seed"* and never added the defect I had just documented. **`O4` is not a cost I forgot to
carry; it is a cost I had written down four paragraphs earlier.**

### P2 🔒 REGISTERED IN THE AUTHOR'S NAME, at his instruction, so it is never read later as an oversight

> **"With this order, the only tier the cut rule can ever reach is `random` — which carries the
> mandatory I1/J1 replicate that `K2` wired into the completeness assertion. So the ladder is now
> ALL-OR-NOTHING BY CONSTRUCTION: four tiers, or a campaign that mechanically refuses to report itself
> complete."**

✅ **Verified mechanically rather than accepted as reasoning, because it is being registered as a
fact.** `docs/data/p5_2_declaration.json` declares **35 cells** — `mappo1000` 12, `maxpressure` 7,
`fixedtime` 7, `random` 8, plus one campaign-level entry — and carries
`envelope_replicate = {tier: "random", arms: [dt_spatial, dt_nomix], seed: 202,
seed_is_pre_declared: true}`. `assert-complete` derives its expected set from **that file, never from
the files being checked** (`tier_sweep.py:2105-2122`), so **dropping any tier leaves
`CAMPAIGN_COMPLETE` unwritten and `PHASES_COMPLETE` written in its place.** The consequence is real and
it is now a registered property of the design, not a discovered one.

### P3 🚨 `NOTE M4` IS FALSIFIED — the thermal mechanism was mine, asserted from a shape, and it is dead

**The author's falsification, and it is the right instrument:** measured at 15:00:39 after **nineteen
hours of continuous GPU load** — maximum heat soak — `maxpressure/dt_spatial` runs at **58.4 and
55.8 min/seed**. **M4's mechanism predicts the machine is in a throttled steady state NOW and therefore
slow. It is the fastest it has been.** A mechanism falsified by its own prediction is falsified.

⭐ **And there is a MATCHED PAIR that closes it without relying on the head count, which I found while
checking his numbers and which is cleaner than either of our arguments:** the two **1-head** E1 cells
ran at **64.1 and 64.9 min** ten minutes into the session; the two **1-head** `maxpressure` cells ran
at **58.4 and 55.8** nineteen hours in. **Head count held constant, heat soak maximal, and it got
≈13 % FASTER.** That is the controlled version of the test, and it points the same way.

⚠️ **WHAT I MUST NOT DO IS SUBSTITUTE A NEW MECHANISM ASSERTED FROM A DIFFERENT SHAPE — that is
precisely how M4 was written.** Head count separates the two populations and does **not** explain
everything:
- **4-head, exact from the artifacts' own `seconds`:** `dt_spatial_h4` **64.5 · 59.5 · 67.4 · 69.9 ·
  80.7**, `dt_nomix_h4` **81.5 · 81.3 · 81.3 · 81.7 · [205.4 → ~75.4]**. Mean of the 9 clean cells
  **74.2 min**.
- **1-head:** P5.1's arm means **59.1 / 57.5**, E1 **64.1 / 64.9**, `maxpressure` **58.4 / 55.8** —
  range **55.8–64.9**.
- 🚨 **The populations separate in DIRECTION and the estimate is not clean: `dt_spatial_h4` spans
  59.5 → 80.7, a 36 % ramp AT CONSTANT HEAD COUNT, which is larger than the between-head-count gap.**
  **So "4 heads are slower" is supported; a percentage is not, and the packet may not quote one.**
- **Two candidate explanations for the ramp, both killed from data rather than left vague:** thermal
  (P3, above); and per-process allocator growth across the five seeds, which `_run_train` runs in **one
  process** — **falsified because `dt_nomix_h4` opened its own process at 81.5 instead of resetting to
  ~60.** **The within-arm ramp is UNEXPLAINED and is reported as unexplained**, exactly as P5.1's seed
  101 was.

### P4 📏 O1's COST NUMBER IS WITHDRAWN — n = 10 and all ten are 4-head cells, while the whole ladder is 1-head

🚨 **This is the project's signature error, committed by me, in the amendment that exists to answer a
costing question: a quantity true of a SAMPLE reported as the POPULATION's.** The ladder arms are
`dt_spatial` and `dt_nomix`; the ten observations behind **74.6 min** are `dt_spatial_h4` and
`dt_nomix_h4`. ⚠️ **And the caveat I attached made it worse rather than better — it named the wrong
confound (cool start versus throttling) and so certified that the confound had been considered.**

| | withdrawn (O1) | corrected |
|---|---|---|
| DT arm-seed on the ladder | 74.6 min *(4-head)* | **≈57 min** *(1-head; 55.8 and 58.4 measured on this very tier today)* |
| a full tier | 16.4 h | **≈13.5 h**, bounded **13.5–14.8** across the 1-head range 55.8–64.9 |
| remaining | ≈51 h | **≈40–42 h**, ending **the morning of 2026-08-22** |
| cutting `fixedtime` saves | 16.4 h | **≈13.5 h = 0.56 days = 2.1 %** of the remaining experimental budget |

> ✅ **`O2`'s ruling is UNCHANGED and now rests on a smaller number: `fixedtime` stays.** The saving
> shrank from 2.6 % to **2.1 %** while the rung's value did not move, so the trade got worse for
> cutting. **A ruling whose margin widens under correction is one I am happy to leave standing.**
> ⚠️ **Also corrected here: plan §2.5's synthetic benchmark reports `n_head=1` at 163.99 ms/step against
> `n_head=4` at 126.85 and concludes *"no cost penalty"* — i.e. it says 4 heads are 23 % FASTER. The
> campaign says the opposite. The microbenchmark does not reproduce the real workload and its
> conclusion may not be quoted;** `NOTE L(b)`'s *"≈8 % slower"* was computed against the wrong 1-head
> baseline and is superseded by the two populations above.

### P5 🚨 O3's CENTRAL FACTUAL CLAIM IS FALSE — the tier order IS registered, and I checked the plan instead of the artifact

**O3 said: *"the executing order is registered nowhere"*, with the parenthetical *"checked: §5 line 861
is the only run-order statement"*.** ⛔ **Measured today:
`docs/data/p5_2_declaration.json` carries `tier_order: ["mappo1000", "maxpressure", "fixedtime",
"random"]` and `tier_order_basis: "cf_grid4x4's own measured att_horizon (BRIEF_27 B0)"`** — written at
Gate 0, **before phase A, before any P5.2 result existed**, in the machine-readable artifact that drives
the completeness assertion.

> 🚨 **So the order is a PRE-REGISTERED design decision with a stated basis, following `B0`'s own
> ruling, and my escalation described it as an unregistered accident.** ⚠️ **My sentence was true of the
> PLAN and I wrote it as true of the REGISTRATION** — the same quantifier error as P4, twice in one
> message, and the second time inside a paragraph criticising someone else's stale document. **I read
> the plan and the script and did not open the declaration: verify the artifact, not the description,
> failed by the person whose §7 rule that is.**
> ⭐ **It also inverts the merits: reordering would have been an UNREGISTERED DEVIATION FROM A
> REGISTERED DESIGN, decided after phase A's result was visible. The author's refusal protects the
> registration; my recommendation would have breached it.**
> ✅ **What survives, and it is now the whole of O3: `docs/plans/p5.2.md:861` says *"PHASE B maxpressure,
> then random"* and contradicts both the declaration and the script.** `911796e` updated §2.5 and `H3`
> and left that block standing. **The plan is stale; the design is not. Correct the plan line in the
> implementer's next commit** — and correct it **to the declaration**, which is the artifact.

### P6 🔎 THE CLOCK JUMP IS INSIDE A COMMITTED ARTIFACT, not only in the log

`training_mappo1000_dt_nomix_h4.json` records seed 505 at **205.4 min** against its siblings' 81.3–81.7.
**`result.seconds` is wall-clock, so it inherited `NOTE M2`'s ≈130-min forward jump** — corrected value
**≈75.4 min**. `M3` ruled that *durations* be read from checkpoint cadence rather than log deltas; **this
is the same defect one layer in, in a file the packet will quote.**
> **REQUIRED in the packet: quote that cell as `205.4 (recorded) / ≈75.4 (corrected)` with the jump
> named, and do not silently substitute the corrected value.** ⭐ **The durable fix is `M2`'s own
> instrument: a training timer should be `time.monotonic()`, which is immune to a wall-clock jump by
> construction.** Recorded for the implementer; it changes no cell and is not urgent.

### P7 ⚠️ MY PROCEDURE CARRIED A PRECONDITION THAT DECAYS, AND THE AUTHOR CAUGHT IT EXPIRING

**His report:** *"`maxpressure` had two completed checkpoints at 15:00:39 — seed 101 at 14:03:00 and
seed 202 at 14:58:49 — so the 'zero checkpoints' premise in your procedure had already expired when I
read it. Nothing was run against it."*
> ⚠️ **The defect is not that the fact changed; it is that I wrote the fact as an EXPECTATION rather
> than as a GATE.** Step (b) read *"expect: nothing, or only `*.partial`"* and said **nothing about what
> to do if that expectation failed** — so a reader following it at 15:00 would have found two
> checkpoints and no instruction. **A procedure whose correctness depends on a decaying fact must carry
> the CHECK THAT STOPS on it, with the stop condition written out, never the fact.**
> **Rule, and it generalises past this instance: any `THINGS YOU NEED TO DO` step resting on a
> time-sensitive precondition states the precondition as a refusal — *"if this returns anything, STOP
> and come back"* — because the block is read minutes to hours after it is written.**

---

## ⛔ NOTE T — 2026-08-22. **IT WAS NEVER A KILL.** A deterministic code defect, and the traceback was on disk throughout

### T1 ⛔ THE CAUSE, and it is three lines away from where both of us were looking

`output/p5_2/logs/eval_fixedtime_behaviour.log`:
```
behaviour@fixedtime seed 101 over 100 draws
ValueError: no action factory is declared for 'behaviour' at tier 'fixedtime'
```
**Immediate, deterministic, raised at seed 101 before a single episode ran.** Not a SIGKILL, not the
host, not memory, not the GPU passthrough layer.
> 🚨 **We both reasoned from the ABSENCE of a traceback in `campaign.log` and the tmux pane — where it
> was never going to appear, because the script redirects every evaluation to its own per-arm log
> (`> "$LOGS/eval_${TIER}_${ARM}.log" 2>&1`).** The reasoning *"stderr goes through `2>&1 | tee`, so a
> Python exception would be in the log"* is true of the campaign's own output and false of the step's.
> ⚠️ **And *"it died ~17 minutes into a ~56-minute evaluation"* is not a measurement — it failed in
> about a second; 17 minutes is when the pane was next looked at.** The dmesg findings are real and
> unrelated: the `dxg` ioctl errors span ~14.7 h including successful runs, so they are background.

### T2 ⛔ MY ERROR, AND IT IS IN A COMMITTED NOTE

**Note S records *"an unexplained silent SIGKILL"* and *"three kills, three different causes"*, and
S3 makes that a packet sentence.** I took the characterisation from the report and never opened the
per-arm log — **a directory I had listed myself earlier in this same session.** That is *verify the
artifact, not the description*, failed by me, one hour after writing the check that found the cells
intact.
> **CORRECTIONS, both binding on the packet: there were TWO kills, not three** — the screen lock and
> the Windows Update restart. **S3's sentence must read two, with the third interruption described as
> what it was: a deterministic defect that the atomic writer never had to survive.**
> ⭐ **What S1's structural check established is untouched** — 25/25 cells intact, 122/122 checkpoints
> loadable, zero stray temporaries. **That evidence stands; only the story attached to it was wrong.**

### T3 🔧 THE DEFECT AND ITS OWNER — one branch, and the policy already exists

`offline/tier_sweep.py:1900-1914`, the `BEHAVIOUR_METHOD` branch of `_arm_factory`, handles
`tier == "random"`, `"maxpressure"` and `"mappo1000"`. **`fixedtime` is absent**, so it falls through to
the `raise`. **`offline/policies/fixed_time.py:379` already provides `make_fixedtime(env, args, rng)`.**
> 🚨 **ROOT CAUSE, and it is a coordination failure rather than a coding one: `fixedtime` was added
> late (F4/H1), and H3 listed the REGISTRATION work meticulously — R′'s cells, `Q1`'s denominator
> restated by rule, `Q2`'s rank prediction and hard subset, `Q3`/`Q4` extended — and NOBODY LISTED THE
> IMPLEMENTATION.** **The declaration declares 35 cells; the factory can build 34.** ⚠️ **That is the
> 2026-08-19 rule — *a plan says WILL, code says DOES* — recurring at the level of a tier addition.**

### T4 🔒 THE MECHANICAL PREVENTION, required with the fix

> **REQUIRED at Gate 0: assert that EVERY cell the declaration names has a CONSTRUCTIBLE factory —
> loop the declared `(tier, method)` matrix, call `_arm_factory`, and refuse on any that raises.** No
> training, no evaluation, seconds to run. **It would have caught this before three days of compute,
> and it closes the general gap rather than the instance: any future tier or arm added to a
> declaration without an implementation fails at launch instead of at the last cell.**
> ⚠️ **Note where the existing guards were blind: `assert-complete` compares declared cells to cells ON
> DISK, so it can only fire AFTER the work; K2's refusal held correctly and told us a cell was missing,
> not that it was unbuildable.**

### T5 ✅ THE PROPOSED EXPERIMENT IS UNNECESSARY — and it was good design for the hypothesis it addressed

The reduced-draw / same-wall-clock-versus-same-draw-index test is a **correct discriminator** for
time-versus-data, and proposing it rather than *"rerun and hope"* was right — *"a third attempt tests
nothing"* is exactly the standard this project asks for. **It is simply not needed: the failure is
deterministic and the log names it.** ⭐ **The cheaper instrument that beat it, and the one to reach
for first next time: READ THE STEP'S OWN LOG. A campaign that redirects each step to its own file has
put the evidence somewhere the campaign log cannot show you.**

---

## ✅ NOTE S — 2026-08-22. The third kill: 32 cells verified INTACT, and the restart is a plain resume

### S1 ✅ THE CELLS ARE INTACT, NOT MERELY PRESENT — and the author was right to demand the difference

**The objection that earned this check is the correct one: the resume's existence tests are safe ONLY
because the writer is atomic, that atomicity was argued FROM SOURCE, and the process that just died
mid-write is exactly the case the argument was made against.** So it was checked structurally rather
than by presence:

| check | result |
|---|---|
| every `eval_*.json` parses, episode count = seeds × draws | **25 / 25** |
| draw ids exactly `1000–1099`, all five seeds (one for the seed-202 cells) | **25 / 25** |
| non-finite values in `att_horizon`, `episode_reward`, `horizon_vehicle_count` | **0** |
| duplicate `(seed, draw)` pairs | **0** |
| required field set present | **25 / 25** |
| checkpoints load with `model` / `config` / `provenance`, model non-empty | **122 / 122** |
| stray or temporary files under `output/p5_2` | **0** |

⭐ **And completeness was taken from the REGISTERED artifact rather than from my own enumeration** —
`assert-complete` reports **32 of 35**, missing exactly `eval_fixedtime_behaviour` and the two
`envelope_replicate_I1_J1` cells. **No fourth surprise.**

### S2 ✅ RULED: PLAIN RESUME. Nothing needs declaring, and here is what I checked rather than assumed

The author's reading is right, and because three kills is the point to stop trusting a reading, the
candidates were enumerated rather than waved past: **the missing cell is a DETERMINISTIC fixed-time
controller on FIXED held-out draws in a bit-reproducible engine**, so its value cannot depend on when
it runs; **no cell depends on another** — each trains and evaluates independently, which is the same
property that made Amendment Q's order deviation bookkeeping; **Gate 0's draws exist and are
byte-verified on entry**, and **Gate 1's random anchor is already on disk so the re-roll skips**; and
**the tier-order deviation is already declared (Q3)**. **Nothing new is registrable and nothing is
withheld.**

### S3 ⭐ THE PACKET SENTENCE, STRENGTHENED BY WHAT THE CHECK MEASURED

The author proposed: *"D1(c) was validated on three unannounced kills rather than on unit tests."*
**That is right and it can be said harder, because the check above supplies the evidence:**
> **Three kills, three different causes — a screen lock, a Windows Update restart, and an unexplained
> silent SIGKILL — at points nobody chose, one mid-TRAINING and one mid-EVALUATION. The result is
> 25/25 evaluation cells structurally intact, 122/122 checkpoints loadable, zero stray temporaries, and
> exactly the three known cells absent. Each kill cost only the unit in flight.**
> ⭐ **No unit test could have produced that evidence, because the kill POINTS were unchosen — which is
> precisely what makes it stronger than the source argument it replaces.** ⚠️ **State it as validation
> of the WRITER, not of the resume logic: the existence tests are still only as good as the atomicity
> beneath them, and what changed today is that the atomicity is now measured rather than argued.**

⚠️ **Not prescribed, recorded: the cause of the third kill is unknown and I am not speculating.** The
remaining exposure is ~66 min plus the ≈2.2 h replicate, and the resume bounds any further loss to one
unit — **so no protocol change is warranted on a hypothesis.**

---

## ✅ AMENDMENT R — 2026-08-22, ruled BEFORE the ladder's numbers were read

### R1 ✅ THE I1/J1 REPLICATE RUNS TONIGHT, AS ITS OWN PHASE. Approved as framed

**The state is expected, not a defect, and K2 is the reason it is visible:** the campaign will close at
33 of 35 cells with `PHASES_COMPLETE` written and `CAMPAIGN_COMPLETE` withheld, which is exactly the
refusal K2 was built to produce. **The implementer's reason for deferring the wiring stands and was
right: a replicate wired mid-campaign is how it ends up re-evaluating the cell it is meant to be
independent of.**
> **RULED: wire it now and run it tonight — `{dt_spatial, dt_nomix}` at the PRE-DECLARED seed **202** on
> the **`random`** tier (J1), under J2's three conditions: independent training from scratch, a distinct
> artifact key, and the canonical `state_dict` digests asserted DIFFERENT before any envelope is
> reported, **equal digests being a refusal and not a zero**. ≈2.2 h.**
> ⭐ **Wire it BEFORE the packet is drafted, and the author's reason is the right one — but note WHY it
> is safe to wire after the ladder's numbers are visible: the replicate's design is ALREADY FULLY
> REGISTERED by J1 and J2 — arm set, seed, tier, conditions, and the reading of a non-zero result
> (I2). Nothing about it is left to choose, so seeing the numbers cannot influence it.** **That is what
> pre-registration buys, and it is why the deferral was affordable in the first place.**

### R2 💰 COSTS, and the one I refuse to guess

**Proposal 1 — the tier-matched scale contrast. VERIFIED: all four tiers exist on BOTH scenarios**
(`cf_hz1x1` and `cf_grid4x4`, `mappo1000 · maxpressure · fixedtime · random`). **The arithmetic is ZERO
GPU** — every cell is committed.
> ⚠️ **The confound-closing run I will NOT cost from an armchair.** Anchor: P4.6 measured ≈1 h per
> hz1x1 tier for a **four-method** grid, and a 1-node DT should be far cheaper than grid4x4's 16-node
> ≈60 min/seed because the joint forward scales with nodes — but *should be* is exactly the word this
> project's estimates keep dying on, mine at "≈52 h" included. **REQUIRED: a ONE-SEED timing probe
> (≈10 min) before any commitment.** The ~5 h figure is the right order and is unmeasured.
**Proposal 2 — the return/load analysis. ZERO GPU**, corpus arithmetic, minutes to an hour of CPU.
**Proposal 3 — the P7.0 repair. Hours for the re-score, and MORE than hours for the repair itself**,
because `DEFERRED` 23 and §7's rule do not ask for a corrected index — they ask that **any cross-system
comparison PROVE its pairing key from structure alone before using it.** **Producing a proved lane
pairing is the work; re-scoring against it is the cheap part, and costing the second as the whole is
how this becomes a two-day job that was quoted as an afternoon.**

### R3 🔧 REGISTRABILITY — one is registrable, two need tightening, and both defects are real

**PREDICTION 1 — TIGHTEN, because the two sides do not have the same arm set.** hz1x1's grid carries
**4** method arms (`madt`, `bc`, `bc_top10`, `iql`); grid4x4 carries **6** (adding `dt_spatial` and
`bc_top10_perix`). **A rank out of 6 is not comparable with a rank out of 4, so "ranks higher on
grid4x4" is not a well-defined quantity as written.**
> **Register instead: (i) the DT's rank within the COMMON arm set `{DT, bc, bc_top10, iql}` — four on
> both sides; and (ii) the paired difference `DT − bc` as the continuous quantity, which is
> within-scenario and therefore comparable across scenarios without a rank at all.**
> ⚠️ **And "the gap widens as the tier degrades" is a MONOTONICITY claim over four points.** §1b's R3
> was falsified on exactly that shape — `fixedtime` broke the ladder's monotonicity on hz1x1 — and its
> standing instruction is to **report by endpoints, never as a trend.** Register the endpoints.
**PREDICTION 2 — REFORMULATE, because the two sides measure different things.** The load-sorter
mechanism operates **BETWEEN NODES within an episode**; on `cf_hz1x1` there is **one** node, so a
between-node correlation does not exist and the only computable quantity there is across DRAWS. **The
proposed hz1x1-versus-grid4x4 correlation compares a between-node effect with a between-draw one.**
> **Register instead a VARIANCE DECOMPOSITION on grid4x4 alone: of the variance in per-stream return,
> what fraction is BETWEEN nodes versus WITHIN a node across draws? If the between-node component
> dominates, a global return quantile is a node selector — which IS the mechanism, measured directly.**
> **`cf_hz1x1` then enters as the degenerate case where the between-node component is zero BY
> CONSTRUCTION — a statement, not a measurement — and that is a stronger argument than a correlation
> comparison because it cannot be confounded.**
> ✅ **The second half — *"`bc_top10_perix` does not collapse where the global filter does"* — is
> ALREADY REGISTERED as `Q5` and needs no new registration.** ⭐ **And the falsification clause is
> accepted as written: if `perix` collapses too, the explanation is reported as falsified, in P8.3's
> manner.**
**PREDICTION 3 — the measurement is fine; the NOVELTY claim needs a caution.** *"The literature search
found nobody who says it"* rests on a search with three declared degradations — Google Scholar
unreachable, OpenReview term search proven non-functional, 32 of 171 records screened on metadata only.
> ⚠️ **And we hold adjacent evidence in our own tree: `CityFlow/tools/converter/converter_v2.py:464`
> carries a TODO marking direction *"falsely defined"*, so the tooling authors knew direction handling
> was wrong without stating our specific finding.** **Register the claim as *"not stated in the
> literature we searched, and flagged as a known problem in CityFlow's own converter source"* — which
> is both true and more interesting than *"nobody has noticed."***

### R4 ✅ SEQUENCING — endorsed, with one free improvement

P5.3 first is right for the author's own reason: **the no-RTG ablation decides whether `A9` has an
object at all.** ⭐ **But Proposals 1 and 2 are ZERO-GPU analysis, so they run DURING P5.3's training
rather than after it.** Then Proposal 3 as the entry to P7.

---

## ⛔ AMENDMENT Q — 2026-08-21. **O5's CLAIM IS WITHDRAWN — IT IS YESTERDAY'S CORRECTED ERROR, REPEATED.** The reorder proceeds as a DECLARED DEVIATION

### Q1 ⛔ WITHDRAWN: *"order is not a registered quantity"*

**O5 asserted it. It is false, and it is the same sentence `584630b` corrected yesterday.** That commit's
§P5 reads: *"O3's CENTRAL FACTUAL CLAIM IS FALSE — the tier order IS registered, and I checked the plan
instead of the artifact"*, against O3's *"the executing order is registered nowhere"*. **Measured, then
and again now: `docs/data/p5_2_declaration.json` carries
`tier_order: ["mappo1000", "maxpressure", "fixedtime", "random"]` and
`tier_order_basis: "cf_grid4x4's own measured att_horizon (BRIEF_27 B0)"`, written at Gate 0 before any
P5.2 result existed.** **It is a pre-registered design decision with a stated basis.**
> 🚨 **The failure is worse than yesterday's, not equal to it. Yesterday I checked the WRONG artifact —
> `docs/plans/p5.2.md:861` instead of the declaration. Today I checked NOTHING and asserted it.**
> 🚨 **And `584630b` was in my own cold-start log. I read its subject line — *"three of my numbers plus
> one claim do not survive the check"* — and never opened it.** **That is the P0.3 failure, the
> canonical example in my own operating instructions, committed against a commit whose message
> announced that my claims had failed.**
> **RULE, because this is the second time in two days on the SAME FIELD: before asserting that anything
> is or is not registered, grep `docs/data/*_declaration*.json`. THE DECLARATIONS ARE THE REGISTRATION —
> not the plan, not the brief, not the script.** *(The author's instruction, generalised: "Check the
> declaration, not the plan and not the script — that was the exact failure last time.")*

### Q2 ✅ THE REORDER STILL HAPPENS — because the REFUSAL'S OWN REASON HAS EXPIRED, not because I re-litigated it

⚠️ **This is not the settled question being reopened.** `584630b` refused on a stated ground: *"a resume
now is a mid-arm resume, which is the O4 defect I had reported four paragraphs before recommending an
action whose cost model assumed a resume."*
> **That ground is gone, measured: there is no running campaign, and `fixedtime` has ZERO checkpoints.
> The kill costs nothing because there is nothing to kill.** The restart also converted the risk from
> hypothetical to observed, and `random` carries **J1's mandatory replicate** and **A1's headline
> question** while `fixedtime` is B1's optional fourth rung.

### Q3 🔒 IT GOES IN AS A DECLARED DEVIATION — the author's framing, and the reason it is right

> **Executed order: `maxpressure → random → fixedtime`. Registered order:
> `mappo1000 → maxpressure → fixedtime → random`. THE DEVIATION IS DECLARED, WITH ITS REASON, IN THE
> PACKET AND IN THE CAMPAIGN LOG. The declaration is NOT edited** — it is a Gate 0 registration and
> rewriting it to match the execution would destroy the very record that makes this visible.
> ⭐ **Why it is bookkeeping and not science, stated with evidence rather than as an assumption: each
> tier trains on its own corpus and seeds and evaluates on the same fixed 100 held-out draws, with no
> cross-tier state — and E1 MEASURED the run-to-run envelope on this metric at exactly 0.0000. So a
> tier's NUMBERS do not depend on when it ran. Its DURATION does, through thermal state (Note O), and
> durations are already governed by M3's cadence rule.**
> ✅ **Nothing mechanically breaks: `TIER_ORDER` the constant is untouched, so
> `test_the_tier_order_is_grid4x4s_own_measured_att_order` still passes, and B0's rule — *order every
> FIGURE by measured ATT* — is satisfied by the report, which reads the constant and not the run order.**
> ⛔ **What may NOT be written is O5's sentence, *"this changes nothing about the design"*. It changes
> the executed order away from a registered one, and the honest form is that the deviation is declared
> and its effect is confined to duration.**

---

## 🛑 NOTE O — 2026-08-21. THE MACHINE RESTARTED. State recovered, Note M corrected, and one ordering change

### O1 ✅ NOTHING IS CORRUPTED, AND EVERYTHING THROUGH `maxpressure` SURVIVED

Verified from the artifacts, not from the log: **`E1_COMPLETE`, `stop_rule_mappo1000.json`, all four
`mappo1000` DT cells, phase C's `bc_top10_perix`, and the ENTIRE `maxpressure` tier — 30 checkpoints
across 6 arms and 7 evaluation cells including the behaviour anchor.** All four declarations parse.
**Zero `.partial` files.**
> **Lost: exactly one thing — `fixedtime/dt_spatial` seed 101, in training since 02:20:28 and killed
> before its first save.** ⭐ **This is D1(c) and the atomic writer working as designed: the campaign
> died mid-training and left NO half-written artifact at a final name**, so the resume path's existence
> tests remain sound and nothing needs re-validating by hand.

### O2 ⚠️ NOTE M's MECHANISM IS CORRECTED — it was half right, and the half it missed is the head count

**Note M attributed the slowdown to heat soak and reported *"not the architecture"*. The `maxpressure`
tier is the control that experiment needed, and it falsifies the simple version.** Measured gaps:

| arm | heads | gaps (min) |
|---|---|---|
| `mappo1000_dt_spatial_h4` | 4 | **59 · 67 · 70 · 81** — ramp |
| `mappo1000_dt_nomix_h4` | 4 | **81 · 81 · 82** — plateau |
| `maxpressure_dt_spatial` | 1 | **56 · 57 · 64 · 57** — flat |
| `maxpressure_dt_nomix` | 1 | **65 · 64 · 66 · 60** — flat |

> **The 1-head arms ran AFTER phase A's sustained load with about an hour of light work between, and
> they sit at 58–64, not 81. So the machine was not permanently heat-soaked, and pure thermal
> accumulation does not survive.** ⭐ **The reading that fits everything: COLD-START speed is the same
> for both (59 against 56), and SUSTAINED speed differs by head count (≈81 against ≈60). The ramp is
> warm-up to an arm's own steady state, and the head count sets WHICH steady state.**
> ⚠️ **So *"the parameter count is identical, therefore it is not the architecture"* does not hold:
> multi-head attention at fixed `d_model` is the same FLOPs with a different memory access pattern
> (head dim 32 against 128), and wall clock is not parameter count.** **I asserted a mechanism from one
> arm's data; the control arrived a day later and narrowed it.**

### O3 ✅ THE CLOCK JUMP IS CONFIRMED IN EXACTLY THE PREDICTED FORM

Note M said a jump inflates **exactly one interval** and used the absence of such a gap to clear P5.1.
**`mappo1000_dt_nomix_h4`'s gaps are `81 · 81 · 82 · 205`.** The 205 sits among 81s — **an excess of
about 123 min, matching the reported jump.** ⭐ **The instrument that cleared P5.1 has now been
validated on a case where a jump DID occur, which is the discriminating power §7 asks for.**

### O4 📉 THE REMAINING COST IS BETTER THAN NOTE M PROJECTED, because the remaining tiers are 1-HEAD

Note M carried 1.38× into the projection from phase A's 4-head plateau. **The ladder arms are
`n_head = 1` and run at ≈61 min/seed.** Measured from `maxpressure`: training 14:03 → 00:05 ≈ **10 h**,
evaluation 00:05 → 02:20 ≈ **2 h 15**, so **one full ladder tier ≈ 12.3 h.**
> **Two tiers remaining plus the I1/J1 replicate ≈ 27 h, not the ~40 h Note M's 1.38× implied.**

### O5 🚨 ORDERING — THE LOOP RUNS THE OPTIONAL TIER BEFORE A REQUIRED ONE

`p5_2.sh:345` reads `for LADDER_TIER in maxpressure fixedtime random`. **`maxpressure` is done, so the
next tier is `fixedtime` — the tier B1 pre-declared as the OPTIONAL fourth rung — and `random` runs
last.**
> 🚨 **`random` is load-bearing three times over and `fixedtime` is not: it is one of B1's three
> ORIGINAL tiers; it is where **Q3a** predicts the DT LEADS, which is A1's headline question; and it is
> the tier **J1** puts the envelope replicate on. `fixedtime` is the rung added from budget freed by
> the regime ruling.**
> ⚠️ **After one unexplained restart, running the optional tier first means a second restart costs the
> required one.** **RECOMMENDED: swap the loop to `maxpressure random fixedtime`.** It changes no
> registration — the tier set, the predictions, the threshold and the arms are all fixed and order is
> not a registered quantity — and it front-loads the tier the paper needs. ⚠️ **The restart's cause is
> unknown and I am not speculating; the mitigation is the same whatever it was.**

---

## ⭐⭐ NOTE N — 2026-08-20. PHASE A's RESULT, THE INTERACTION COMPUTED, AND TWO REGISTRATIONS BEFORE PHASE B

### N1 ✅ THE INTERACTION, COMPUTED AS A2 REGISTERED IT — paired per-draw, not two intervals

The implementer was right to refuse its own eyeball and ask for the registered quantity. **I computed
it independently from the four cells, pairing on the 100 shared draws** (`I_i = d4_i − d1_i`, each a
5-seed per-draw mean):

| quantity | mean | CI95 |
|---|---|---|
| `d1` — 1 head (P5.1) | **+39.5649** | [+36.0510, +43.0787] |
| `d4` — 4 heads (P5.2) | **+27.1004** | [+23.8150, +30.3858] |
| **`I = d4 − d1`** | **−12.4645** | **[−16.3575, −8.5714]** |

**`d1` and `d4` reproduce the reported values exactly.** ⭐ **The paired interval is STRONGER than the
non-overlap eyeball, not weaker** — pairing removes between-draw variance, so the registered test
resolves more decisively than the conservative one. ⚠️ **And it carries something the eyeball cannot:
`I < 0` on 70 of 100 draws, not on all of them. The 31.5 % reduction is a mean over a split
population and must be reported with that count, per §5's per-seed/per-draw rule.**
> ⭐ **E1 is what makes `I` interpretable at all: `d1` and `d4` come from different training runs, and
> the run-to-run envelope on this metric was measured at exactly 0.0000. Without it, `I` would carry
> unmeasured noise from two independent runs.** ⚠️ **Scope, per H2: that zero was measured on the
> 1-HEAD arms at `mappo1000`. It is not established for the 4-head models, whose decision margins
> could differ — state that beside `I` rather than inheriting it silently.**

### N2 🔒 REGISTERED: THE EXTRAPOLATION IS REFUSED, AND THE REFUSAL IS SHARPER THAN THE TREND

The implementer's arithmetic is right — 1→4 heads is **two** doublings, so **6.23 ATT per doubling**,
putting 8 at ≈20.9, 16 at ≈14.6, 64 at ≈2.2 — and its instinct to register the refusal rather than the
extrapolation is right. **The refusal must be stronger than it framed it.**
> 🚨 **TWO POINTS DETERMINE A LINE BY CONSTRUCTION, SO THEY CARRY NO INFORMATION ABOUT FUNCTIONAL
> FORM.** The data cannot distinguish decay toward zero from a plateau from a reversal. **So even
> *"shrinking but not vanishing"* is unlicensed — it is the linear reading asserted as a shape.**
> **BINDING: the measured claim covers `n_head ∈ {1, 4}` and nothing else. No sentence may state,
> imply or invite a value at 8, 16 or 64 heads, including qualitatively.** The per-doubling figure is
> recorded here **as the thing being refused**, because a referee will compute it in ten seconds and
> silence would look like we had not noticed.

### N3 🔒 REGISTERED: THE LAYER AXIS — and *"we tested the literature's configuration"* IS NOT AVAILABLE

The implementer flags that layers were never varied. **Verified in source, and the limitation is
sharper than "we held them at the inherited value":** `SpatialDTConfig.n_layer = 3` and every
`_SpatialBlock` carries **one temporal and one spatial sublayer**, so our spatial mixing is
**interleaved three times**. The in-domain papers use **4 heads with a separate 2-layer GAT stack**
(2603.22315: *"d=128, L=3, NH=4, 2 GAT layers with 4 heads"*).
> ⚠️ **So we do not have FEWER spatial layers than they do — we have a DIFFERENT ARRANGEMENT at a
> different depth, and we matched only the head count.** **BINDING: the sentence *"we tested the
> literature's configuration"* may not be written. What is true and sufficient is *"the harm persists
> at the head count both in-domain papers use"*.** **The head-count × layer-count interaction is
> unmeasured, and that travels in the same sentence as the result — it is exactly the overclaim the
> result invites.**

### N4 ⭐ WHAT PHASE A ACTUALLY BOUGHT

**P5.1's negative is not a single-head artifact, and that removes the rebuttal we could not answer.**
§1b's **C3** registered the confound before P5.1's campaign landed — *"a collapse or a null is
CONFOUNDED with head count against both architectures ours is closest to"* — and it is now measured
rather than qualified away. ⚠️ **And the harm is head-count DEPENDENT: 31.5 % of it is attributable to
the single-head configuration.** **Both halves are the result; reporting the first without the second
would be the mirror of the overclaim N3 forbids.**

---

## 🚨 NOTE M — 2026-08-20. A CLOCK JUMP IN `campaign.log`, and the slowdown has a mechanism

### M1 ✅ P5.1 IS UNAFFECTED, and I checked it the only way that could have shown otherwise

**The report matters because A4 and B6 both measured P5.1's cost from `campaign.log`, and B6 REPLACED
my mtime-derived figure with the log-derived one — so if that file could straddle clocks, a merged
number was measured by an instrument that changed units.** ⚠️ **Comparing mtimes against log
timestamps would NOT have settled it: both read the same system clock and a jump shifts both alike.**
> **The discriminating check is the INTERNAL GAP PATTERN, because a jump inflates exactly ONE interval.**
> **Measured across P5.1's eight checkpoint intervals: `60.8 · 59.3 · 63.9 · 53.8 | 58.1 · 58.5 · 55.1
> · 58.6` — a band of 53.8–63.9 min with no outlier.** A 2 h 11 m jump would have added **~131 min** to
> one of them. **There is none. P5.1's campaign was single-clock and its 13 h 24 m 25 s stands, as does
> B6's correction.**

### M2 🔒 THE DEFECT IS REAL, AND HERE IS THE PROOF A REVIEWER CAN RE-RUN

The `btime` argument is right that the check has no power, but the positive proof is cleaner and does
not depend on knowing how WSL resyncs:
> **The training process's MONOTONIC age (`etime` 05:59:32 at 10:43:39) places its start at 04:44:07.
> The first checkpoint it wrote is stamped 03:55. A process cannot write a file 49 minutes before it
> starts.** ⭐ **Monotonic elapsed time is immune to a wall-clock jump; that is why the contradiction
> appears at all, and it is the instrument to reach for whenever a duration looks wrong.**
> **Magnitude ≈ 2 h 10 m, forward, after 07:59:54.**

### M3 📋 BINDING ON THE PACKET — both requirements accepted as the implementer framed them

1. **Report phase durations from the CHECKPOINT CADENCE, not from log deltas.** Cadence intervals lie
   within one clock and are therefore valid; only totals spanning the jump are inflated.
   **Measured, all pre-jump: `dt_spatial_h4` 59.5 · 67.4 · 69.9 · 80.7 · then `dt_nomix_h4` 81.3 · 81.3
   · 81.7.**
2. **Disclose the jump with the observation that no CELL is affected** — every cell's numbers come from
   the corpus and the model, not from the clock. **Only durations are, and only those spanning it.**

### M4 ⭐ THE SLOWDOWN IS THERMAL, AND THE CADENCE SHAPE SAYS SO

🚨 **WITHDRAWN 2026-08-20 BY THE AUTHOR — THE MECHANISM ASSERTED BELOW IS FALSIFIED BY ITS OWN
PREDICTION. SEE AMENDMENT P3.** It said a *"cool start rising to a throttled steady state"*; nineteen
hours of continuous load later, 1-head arm-seeds run at **55.8–58.4 min**, i.e. **faster** than the
same configuration ran ten minutes into the session. **The paragraph stands unedited below as the
record of what was claimed, and its opening clause — *"it is NOT the architecture"* — is the part that
was wrong.**

The implementer established it is **not the architecture** — parameter count is identical at 4 heads —
and left the mechanism open. **The shape supplies one, and it is testable rather than asserted:**
> **`dt_spatial_h4` ramps MONOTONICALLY — 59.5 → 67.4 → 69.9 → 80.7 — and `dt_nomix_h4` then sits FLAT
> at 81.3 · 81.3 · 81.7.** **A cool start rising to a throttled steady state is what heat soak looks
> like on a laptop GPU; a per-arm cost difference would step at the arm boundary and stay level within
> each arm, which is the opposite of what the first arm does.**
> ⭐ **It also dissolves the counterintuitive part: the CONTROL is not intrinsically slower than the
> TREATMENT. `dt_nomix_h4` ran entirely in the throttled regime while `dt_spatial_h4`'s average
> includes its cool start.** P5.1's own arms were 58.6 and 59.0 — indistinguishable.

### M5 ⚠️ THE PROJECTION MOVES, AND IT IS A SCHEDULE FACT NOT A DEFECT

**Steady state is ≈81.4 min per DT arm-seed against the 59 every projection used — 1.38×.** A4's
~53 h assumed 9.8 h of DT training per tier; at 81.4 that is **13.6 h per tier**. With three tiers
still to run plus the I1/J1 replicate, **the remaining campaign is on the order of a day and a half
longer than the figure the regime ruling was priced against.** ⚠️ **Nothing about the ruling changes —
it turned on a measured zero envelope, not on cost — but §10's cut order exists for exactly this, and
`fixedtime` is the declared fourth rung and therefore the first thing to drop if the calendar binds.**

---

## 📌 NOTE L — 2026-08-20, written MID-CAMPAIGN so it cannot be a post-hoc reading of Q0

**Observed while checking phase A's progress, from `output/p5_2/logs/train_mappo1000_dt_spatial_h4.log`
and the checkpoint times. Neither is a result; both are recorded now precisely because saying them
AFTER Q0 resolves would be rationalisation.**

**(a) The 4-head arm's training loss is markedly LOWER, and that predicts NEITHER Q0 outcome.**
`dt_spatial_h4` seed 303 finishes at **0.14703**, against P5.1's 1-head `dt_spatial` finals of
**0.2058 · 0.2227 · 0.2103 · 0.1947 · 0.2034** (mean ≈ 0.207) — roughly **29 % lower**.
> ⚠️ **BINDING ON THE PACKET: a lower imitation loss is a DIAGNOSTIC, not evidence about control.**
> **P5.1 already measured exactly this dissociation** — its §4.2 records that *the mixing arm fit its
> data better and controlled worse*, 0.207 against `dt_nomix`'s 0.437 while losing by 39.57 ATT — and
> §7 (2026-08-12) records that on this corpus loss-based criteria track **training-set size**, not
> competence. **So 0.147 is consistent with Q0a HELD and with Q0a FAILED, and may not be offered as
> support for either.** ⭐ **If Q0a fails, the honest sentence is *"4 heads reversed the contrast"*, NOT
> *"the lower loss explained it"*; if Q0a holds, this becomes a THIRD data point in the fit-better /
> control-worse pattern, which is a stronger version of P5.1's finding.**

**(b) `n_head = 4` is free in PARAMETERS and NOT free in WALL CLOCK — the earlier claim was right about
what it measured and is being read too broadly.** Measured per-seed: **65 · 59 · 68 min** against the
1-head **59 min/seed** from P5.1 — mean ≈ **64 min, about 8 % slower.** The `853,128 params either way`
figure is correct and is about **parameters**; multi-head attention at fixed `d_model` reshapes rather
than adds. **Carry the ≈8 % into the remaining projection rather than treating phase A as free.**

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
> **RULED: three tiers — `mappo1000`, `maxpressure`, `random`.** 🚨 **THE JUSTIFICATION IN THIS
> PARAGRAPH IS WITHDRAWN — SEE AMENDMENT B0/B1. `281.89` and `262.09` are `cf_hz1x1`'s cells, quoted
> here as grid4x4's; the tiers do NOT span grid4x4's measured ATT range (160.33 → 1370.22) and no
> sentence may say they do.** The set is kept for a different and better reason (B1: `maxpressure` is
> where the DT ranked LAST on hz1x1, so it is the most falsifying rung for the lead question), it
> separates on the **return** axis by 5.5×, and **`fixedtime` is pre-declared as an optional fourth
> tier**. **Order every figure by grid4x4's own measured ATT.** **If it overruns, drop to `mappo1000` +
> `random`** — the endpoints carry the interaction; the middle does not.

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
- [ ] **B1** — the tier set is `{mappo1000, maxpressure, random}`; **the disclosure is IN the plan and
      the packet**: the top two rungs are 7.16 ATT (4.5 %) apart, the tiers do NOT span grid4x4's
      160.33→1370.22 range, separation is on the RETURN axis (5.5×). **`fixedtime` recorded as the
      pre-declared optional fourth tier, never as a replacement**
- [ ] **B3** — reuse gate: digests re-checked **at consumption**, the equality check asserts **exactly
      500 episodes** before comparing, its **positive control is executed and the failure pasted**, and
      a mismatch **refuses and stops** rather than re-running
- [ ] **B4** — `random` tier size-matched to **200 streams, `one_per_draw`**, declared RNG, selection
      recorded in the artifact
- [ ] **B5** — the registered prediction scores **rank separately from level**, **excludes already-seen
      `mappo1000` cells from the denominator with the out-of-sample set enumerated**, and **fixes band
      and threshold in the commit** with the calibration stated beside them
- [ ] **C1** — the trainer-equivalence obligation runs a **same-device old-vs-old control first**, and
      reports it; `==` only if the control reproduces, otherwise CPU equality plus a reported envelope
- [ ] **C2** — old and new trainers assert the same `warmup` and LR multiplier at **`total = 40,000`**,
      steps `{0, 999, 1000, 1001, 40000}`, computed rather than trained
- [ ] **C3** — the plan carries the **P8.3 fence**: no sentence cites its D4RL numbers as IQL validation
- [ ] **C4** — Q2b reports the **6-pair hard subset** `{dt_nomix, bc_top10_perix, bc, dt_spatial}` beside
      the 15-pair concordance; the registered threshold is unchanged
- [ ] **D1** — `output/p5_1/` **read-only, enforced in code**, refusal tested and pasted; every write and
      delete after all validation; the resume path writes-validates-moves and cannot mistake a
      half-written cell for a complete one
- [ ] **D2** — C2's step set includes **step 249 (expected `0.250000`) AND step 500 (expected `0.501`)**; 249 is the discriminating point, 500 is the linear check
- [ ] **D6** — the registered fallback for a negative C1 control is in the plan **before the control
      runs**: `d1` re-produced on the new code path so `I` is within-code-path, P5.1's cells reported
      beside it rather than replaced
- [ ] **E1** — the envelope: both arms replicated at **seed 202**, 40,000 steps, **default CUDA**,
      evaluated, `d1(202)` compared against P5.1's **+72.07**; reported whichever way it comes out
- [ ] **E3** — `mappo1000`'s within-tier rankings use P5.1's default-regime cells; the deterministic
      1-head pair serves `I` alone; Q1's cross-tier seam is disclosed with the Q1 result
- [ ] **F5** — E1's write-up corrects or corroborates P5.1's *"unstable across seeds"* sentence **in the
      same commit**, and reports the seed-level CI `[+1.88, +77.25]` beside the published draw-level
      `[+36.05, +43.08]` and the 70× per-seed spread
- [ ] **F6** — `deterministic: bool` acts at **process entry** with `CUBLAS_WORKSPACE_CONFIG` exported by
      the script; if (c) is chosen, determinism is proved for **every arm and the evaluation path** in
      Gate 0, not just the two DT arms
- [ ] **F7** — E1 reports the **paired per-draw CI** of (replicate − P5.1) for `dt_spatial`, `dt_nomix`
      and `d1`, each with whether it excludes zero, and states that obligation 6's CPU byte-identity is
      what attributes any difference to the device rather than to the trainer
- [ ] **G2** — E1 runs with `CUBLAS_WORKSPACE_CONFIG` **UNSET**, matching P5.1 exactly; `p5_2.sh` sets it
      only under `--deterministic`
- [ ] **G3** — the with/without comparison run on the C1 harness, both counts reported, plus whether the
      C1 control's own default arm carried the variable
- [ ] **H1** — phases A/B/C under **default CUDA**, four tiers: `mappo1000` (reused), `maxpressure`,
      `random`, `fixedtime`
- [ ] **H2** — the zero envelope is registered as a `mappo1000`-only measurement wherever it is used
- [ ] **H3** — `fixedtime`'s R′ cells, the restated `Q1` denominator and threshold by a stated
      principle, `Q2`'s rank prediction and hard subset, and `Q3`/`Q4` extended — **all before phase B**
- [ ] **H4** — the three disclosures carried into the packet, the E1 verification chain as its own
      paragraph
- [ ] **I1/J1** — **BOTH arms at SEED 202** replicated on the **`random`** tier inside phase B,
      unconditionally, yielding `d1`'s envelope at both ends of the ladder; F7's machinery and the
      1e−12 positive control; the seed is **pre-declared** and is not re-selected after phase B
- [ ] **K2** — the two replicate cells are enumerated in the declaration **before phases A/C/B run**, so
      the completeness assertion refuses a campaign that lacks them
- [ ] **J2** — the replicate is an independent training run under a **distinct artifact key**, and the
      machinery **asserts the two canonical `state_dict` digests DIFFER** before reporting any envelope;
      equal digests are a **refusal**, and the digests are reported beside the number
- [ ] **I2** — the registered reading is in the plan **before the number exists**: a non-zero envelope
      is a finding, a zero envelope is not the only acceptable outcome, both are publishable
- [ ] **I3** — `Q1`'s threshold is `k = ceil(9/13 × N)`, stated in the commit, giving **14 of 19**;
      the rule reproduces `k = 9` at `N = 13` and that self-check is recorded
- [ ] Every ordering with its per-seed count **and** range, **emitted by the generator**
- [ ] `DEFERRED` 37's mutation executed, failure pasted; every mutation's failure pasted
- [ ] Campaign in a **user-launched `tmux`**; **no `until`-poll**; `mkdir -p` before `tee`, and the
      script must not clear the log directory after `tee` opens its target (P4.7 and P5.1 both lost a log)
- [ ] Suite green, tail pasted, pinned; three guards, no arguments, full-output counts, corpus named
- [ ] Return Packet at `docs/returns/P5.2.md` with the AI-assistance record
- [ ] §6's checkbox unticked; it is mine, in the merge commit — **and §6's P5.2 text must be corrected
      to strike hangzhou_4x4 AND `IPPO/DQN` (A5) in that same commit**
