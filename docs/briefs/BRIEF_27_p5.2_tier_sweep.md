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
- [ ] Every ordering with its per-seed count **and** range, **emitted by the generator**
- [ ] `DEFERRED` 37's mutation executed, failure pasted; every mutation's failure pasted
- [ ] Campaign in a **user-launched `tmux`**; **no `until`-poll**; `mkdir -p` before `tee`, and the
      script must not clear the log directory after `tee` opens its target (P4.7 and P5.1 both lost a log)
- [ ] Suite green, tail pasted, pinned; three guards, no arguments, full-output counts, corpus named
- [ ] Return Packet at `docs/returns/P5.2.md` with the AI-assistance record
- [ ] §6's checkbox unticked; it is mine, in the merge commit — **and §6's P5.2 text must be corrected
      to strike hangzhou_4x4 AND `IPPO/DQN` (A5) in that same commit**
