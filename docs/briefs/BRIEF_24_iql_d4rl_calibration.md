# BRIEF 24 — external calibration of our IQL against D4RL

**Mode:** Claude Code · **Branch:** `task/p8.2-iql-calibration`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p82 -b task/p8.2-iql-calibration main`
⚠️ **SCHEDULING CORRECTED 2026-08-17 — this does NOT run concurrently with P5.1's campaign.**
**It runs in the window between P5.1's Gate 1 and Gate 2.** P5.1's campaign is **~13 h on the same
GPU**, and running both at once makes **both sets of timings uninterpretable in a project that reports
training seconds as artifacts** (`p*_training.json` carries them; P4.6's brief budgeted from measured
`s/episode`). ⚠️ **The real deadline is the start of P5.2, not P5.1** — P5.1 is a DT-architecture
question and does not consume IQL, whereas **P5.2 inherits it as a baseline.**
**Read first:** `PROJECT_PLAN` **§1b (R3, R5)**, `docs/reviews/P4.6.md`, `docs/reviews/P4.7.md`.

⚠️ Absolute paths · pin threads · no traffic simulator is needed anywhere in this task.

---

## 1. What this tests, and why it is not a side quest

**It tests our INSTRUMENT, not our reading.** Our IQL currently wins the entire weak half of the C1
ladder while running **published D4RL-locomotion hyperparameters, never tuned for this problem** —
`tau = 0.7`, `beta = 3.0`, `gamma = 0.99`, Polyak `0.005`, weight clip `100`, reward scale
`1000/(max−min)` (verified in `offline/offline_baselines.py:197-200, 1594-1603, 1935-1936`).

**That fact is load-bearing.** §1b's **R5** uses it to retire the *"your IQL was a straw man"*
objection: *it lost every configuration argument in advance and won anyway*. **R5 is only true if the
implementation is CORRECT**, and its correctness currently rests **entirely on internal verification** —
four independent reviews, none of which compared it to anything outside this repo.

🚨 **The anomaly that forces the issue.** Our result contradicts two neighbouring literatures. On
D4RL `halfcheetah-medium-expert-v2` the canonical figures are **BC 45.8, 10%BC 86.4, TD3+BC 92.2,
IQL 89.9, CQL 86.3**, and on hopper/walker2d medium-expert IQL reaches ~110 and matches or beats
10%BC. Offline MARL runs the same way — **OMIGA (2307.11620)** reports behaviour-modelling struggling
and implicit value learning gaining on mixed suboptimal data; **OMSD (2505.05968)** attributes
value-decomposition's advantage specifically to *"the severe multimodal distribution of joint behaviour
policies on mixed-quality datasets"*. **Our P4.7 has %BC leading all three mixtures while IQL drops to
3/4 — the opposite.**

**Two explanations produce exactly that pattern and no dataset statistic can separate them:**
**(A)** our IQL is doing something wrong that four reviews did not catch, or **(B)** our IQL is correct
and the domain differs. **This measurement separates them, and it is the only one of the three
candidate measurements that can.**

> **If it lands near ~90: the implementation is sound, the difference is domain-side, the anomaly
> becomes a RESULT, and every IQL number in the paper gains an external anchor.**
> **If it lands far below: the defect is in IQL and not in the data — and we learn it in an afternoon
> rather than from a referee**, before P5.2 consumes the same IQL as a baseline.

## 1b. ⭐ A SECOND, SHARPER INSTANCE ARRIVED 2026-08-18 — AND IT IS A SCOPE QUESTION, NOT INSTABILITY

**P5.1 ran IQL on grid4x4 and it LOSES TO `random`: 275.8354 against 260.3602, winning only 36 of 100 draws** — independently confirmed by P5.1's reviewer (per-seed `−1.47 / +20.58 / −28.22 / +34.83 / +51.65`, pooled `+15.4753`). **The same untuned IQL won the ENTIRE weak half of the hz1x1 ladder** (1/4 on `maxpressure`, `fixedtime` and `random`).
> **Frame it as SCOPE, not instability.** An ordering reversing on 2 of 5 seeds is instability; **an arm losing to RANDOM on a 16-intersection network while winning the weak half on ONE intersection is a question about IQL itself** — and that is precisely what an external calibration answers. **One measurement now settles both this and the D4RL/offline-MARL anomaly in §1**, which is the argument for running it before P5.2 inherits IQL as a baseline.
⚠️ **Report both instances in the packet.** The hz1x1 anomaly (%BC leading the mixtures while OMIGA and OMSD report value-based methods gaining on multimodal data) and the grid4x4 one (IQL below random) are **two faces of the same question**, and a calibration that lands near ~90 licenses a domain-side reading for both.

## 2. What to run

**Our own `train_iql`, UNCHANGED and UNTUNED, on `halfcheetah-medium-expert-v2`.** Same code path, same
config, published comparator. ⚠️ **Change nothing to make the number look better. If the code path needs
an adapter for a continuous action space, the adapter is the deliverable and its every choice is
declared** — a continuous-action IQL is not the discrete-action one, and any difference in the update
must be stated, not smoothed over.

**Report the D4RL normalised score** against the published table, with seeds and CIs.

## 3. 🔒 DEPENDENCY RULING — the only new dependency this project has authorised

`CLAUDE.md` rule 3 permits **numpy / stdlib / torch / pytest** only. **`d4rl` (and its `mujoco` / `gym`
chain) is authorised for THIS TASK under three binding conditions**, recorded in the Decisions Log:

1. **A SEPARATE VENV.** `d4rl`, `mujoco` and `gym` **never** enter the project's dependency set —
   not `pyproject.toml`, not any requirements file the project installs from. **Create it outside the
   repo and name its path in the packet.**
2. **CALIBRATION-ONLY.** **No module that any paper number flows through may import it.** The
   calibration harness lives in its own file and is imported by nothing in `offline/`. **Assert this
   mechanically: a test that greps `offline/**` for `d4rl`/`mujoco`/`gym` imports and fails if one
   appears.**
3. **It may not change the project's install.** `BRIEF_23` is fixing packaging in parallel; **these two
   tasks must not collide** — if `pyproject.toml` needs touching, it is `BRIEF_23`'s file, not yours.

⚠️ **If the install fights back, STOP and report rather than spending a day on it.** The value is the
measurement, not the wrestling. **A clean "could not install, here is how far I got" is an acceptable
outcome and is worth more than a broken environment.**

## 4. Declare the structural differences BEFORE the number exists

**If (B) holds, these are the candidate explanations, and registering them now stops them being
selected afterwards to fit the result.** Put them in `docs/plans/p8.2.md` before running:

1. **Single intersection.** OMIGA and OMSD are multi-agent SMAC, where *multimodal* means heterogeneous
   **joint** policies across agents. Ours is **one agent under two policies**. Same word, different
   quantity.
2. **Horizon.** D4RL locomotion is ~1000 dense-reward steps; we have **360 decisions**. Kumar et al.
   show the offline-RL advantage **grows with H**, so a shorter horizon predicts less stitching benefit.
3. **Tuning.** Ours is untuned; the D4RL tables are post-tuning. **This cuts against us and must be
   stated in that direction.**
4. **Mode geometry.** Our modes are **extremely disjoint — a 28,378-unit return gap with zero overlap**
   (`DEFERRED` 46) — while medium-expert's are adjacent. **They may not be the same situation despite
   sharing the label "bimodal".**

## 5. What this task may NOT conclude

⚠️ **It may not settle the mode-separability direction.** That needs the two dataset statistics (TQ and
SACo across the eleven tiers; a modality statistic on ours **and** on medium-expert) which are **NOT in
this brief**. **This task answers exactly one question: is our IQL implementation sound.**
⚠️ **And it may not be written up as agreement with the literature if it lands near 90** — it would show
our implementation matches, which is a statement about the code, not about the domain.

## 6. Definition of Done

- [ ] `docs/plans/p8.2.md` committed **before any run**, carrying §4's four differences verbatim
- [ ] Our unchanged `train_iql` run on `halfcheetah-medium-expert-v2`, normalised score with seeds and
      CIs, against the published **89.9**
- [ ] Any continuous-action adapter **declared choice by choice**
- [ ] The separate venv named; **the grep test proving no `offline/**` module imports `d4rl`**
- [ ] Suite green, tail pasted, pinned; three guards, no arguments, full-output counts
- [ ] Return Packet at `docs/returns/P8.2.md` with the AI-assistance record
- [ ] A `- [ ] **P8.2**` line added to `PROJECT_PLAN` §6 under Phase P8, left unticked; it is mine, in
      the merge commit
