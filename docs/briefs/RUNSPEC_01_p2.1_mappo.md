# RUN-SPEC #1 — P2.1: MAPPO training runs

**Not a brief.** No code is written for this. It is a specification for runs the **user** executes in
`tmux`, plus the decisions those runs freeze. **Issued:** 2026-08-06 by the Master chat.

> Everything below that looks like a free parameter is a **registered quantity**. After collection
> nobody can distinguish "5 draws was chosen" from "5 draws was settled on", so the number, the ids and
> the reasoning are written here *before* the runs.

---

## 1. MAPPO has two roles, and both are built (user ruling 2026-08-06)

| id | training demand | serves |
|---|---|---|
| **`mappo_nominal`** | draw **0** only (nominal) | **a required cell of C2's pre-registered 2×2** (§1: {nominal, shift-augmented} × {MADT, MAPPO}) |
| **`mappo_dr`** | **5 draws** from the 1–999 pool | the other 2×2 cell, **and** the behaviour policy for ladder tiers 3–4 |

`mappo_nominal` is **not** an inferior alternative to `mappo_dr`. Deleting it would delete a cell from
a design pre-registered three days earlier. `mappo_dr` exists because a single-demand expert yields a
corpus with narrow state coverage, and MADT cannot learn a regime its data never visited.

---

## 2. Registered quantities

**Draw set for `mappo_dr`: ids `1, 2, 3, 4, 5`, cycled per episode (`draw = ids[ep mod 5]`).**
Reasoning, recorded because the number is not self-evident:
- **Five, not more:** MAPPO is on-policy and sample-hungry — the plan already records it unconverged at
  200 episodes on hangzhou_4x4. At the 500-episode budget, 5 draws give 100 episodes per draw, the same
  per-draw budget as a 100-episode single-demand run. More draws would thin that below what MAPPO needs
  to learn anything.
- **Five, not fewer:** it matches the ≥5-replicate convention already registered in
  `PREREGISTRATION.md` §8, so it introduces no new arbitrary constant.
- **Ids 1–5, not a random subset:** transparent, reproducible, and immune to any suspicion that the
  draw set was selected. All sit inside the 1–999 training pool, hence **disjoint from the held-out
  1000–1099** (D4). Overlap with the corpus collection range is intentional and harmless — the
  behaviour policy is data generation, not a model evaluated on those draws.

**Seeds: 5** (`PREREGISTRATION.md` §8 requires ≥5 training seeds for a reported baseline). The same 5
seeds serve both roles; the ladder's behaviour-policy checkpoints fall out of the Role B runs for free.

**Scenarios: the headline three** — hangzhou_1x1 #1 bc-tyc, grid4x4, cologne3 (P2.0c).

**Reported metric: `att_horizon`** (A1, via the P8.0 reader). Never the running mean.

---

## 3. Checkpoint cadence — three runs, and why that is legitimate

Checkpoints are needed at **60 / 200 / converged**. The harness saves only a *final* checkpoint per
(env, agent, seed) — `_save_agent` is called once after training (`runner.py:357-361`), and
`checkpoint_path` has no episode component. So the cadence comes from **three runs at the same seed**
with `train_episodes` = 60, 200, B and separate `checkpoint_dir`s.

**This is scientifically identical to snapshotting one run, and that was measured, not assumed.**
Trained cf_hz1x1 at seed 0 for 60 and for 120 episodes: the first 60 per-episode training returns are
**exactly equal**. `MAPPOAgent` carries no budget-coupled schedule (its only `epsilon` is a 1e-4
numerical-stability constant; the `epsilon_decay_to_budget` path is DQN's). So a 60-episode run *is*
the long run's state at episode 60.

Cost: 760 episodes per (variant, scenario, seed) instead of 500. Accepted — it buys the cadence with
zero code and zero frozen-file changes.

> If a future agent type **does** couple to `train_episodes`, this reasoning collapses for that agent
> and the three checkpoints become three unrelated policies, confounding every C1 data-quality claim
> with run-to-run variance. Re-run the prefix-equivalence check before reusing this cadence for DQN.

---

## 4. The declared budget and the convergence criterion (D5)

**Declared in advance: B = 500 episodes.** The reported online MAPPO baseline is the checkpoint at B.
**No checkpoint is ever selected by its evaluation score** — that is D5, and it is what separates an
offline-RL paper from a rejected one.

**Convergence is judged on the TRAINING-return curve, never on evaluation ATT.** Judging it on
`att_horizon` would be online model selection wearing a different hat. `_train_agent` already returns
per-episode training returns; use them.

Criterion, with W = 50: let `m1` = mean return over episodes [B−50, B) and `m2` over [B−100, B−50).
Converged iff `|m1 − m2| / max(1, |m2|)` < 0.05 for the final window **and** the preceding one.

If B = 500 fails the criterion on any scenario, **raise B once to 1000 for that scenario and record it
as a `PREREGISTRATION.md` amendment** stating that the raise was driven by the training curve, not by
any evaluation number. Do not raise it twice without coming back.

---

## 5. Sequencing — `mappo_nominal` can start tonight; `mappo_dr` cannot

- **`mappo_nominal` needs no plumbing.** It trains on the shipped nominal demand, so it is configs +
  runs today.
- **`mappo_dr` needs draw materialisation first**: the randomiser must render draws 1–5 to flow files
  and matching scenario configs, and the training run must cycle them per episode. That is a small,
  well-defined implementer task and does not exist yet.

**So start `mappo_nominal` immediately and brief the draw plumbing in parallel.** The longest pole
begins tonight for half the matrix rather than waiting for the whole thing.

---

## 6. Compute estimate (from measured s/episode, plan §P2.0b)

Training: hz1x1 1.04 s/ep, grid4x4 2.81 s/ep, cologne3 ≈ 2.5 s/ep (interpolated from its 0.97 s/ep
rollout — **estimate, not measured**).

Per variant: 760 episodes × 5 seeds × (1.04 + 2.81 + 2.5) ≈ **6.7 h sequential**, ≈ **1.7 h** at the
measured ≈3.97× with `--workers 6`. Both variants ≈ 3.4 h pinned. Overnight-feasible.

⚠️ **Never run unpinned.** Unpinned parallelism does not merely run slowly — it can wedge indefinitely
(`limit_torch_threads`, P0.3-fix). The pin is inside `run_cell`, so any path through `experiments/run.py`
is safe.

---

## 7. What the runs must produce

For each (variant, scenario, seed, budget): a checkpoint, the training-return curve, and evaluation in
**`att_horizon`** with entered/completed counts. Plus, per scenario, the convergence verdict from §4
with the two window means shown.

**Sanity anchors** (`att_horizon`, post-pin, from `docs/data/p0_baselines_horizon/`): cf_hz1x1
MaxPressure 247.75 / Random 413.53; cf_grid4x4 169.05 / 265.75. MAPPO@60 on grid4x4 is expected to be
**far worse than Random** — the post-pin figure is 1397.95, and that is the paper's
"online MARL below random at low interaction budget" motivation, not a bug.
