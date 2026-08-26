# BRIEF_28 — P5.3a: does the return prompt do anything at all?

**Task id:** `P5.3a` · **Branch:** `task/p5.3a-rtg-probe` · **Issued:** 2026-08-24
**Mode:** Explore → Plan → Code → Commit, with human gates. **Start in plan mode.**
**Compute:** **ZERO GPU TRAINING.** Forward passes on checkpoints that already exist, plus one short
live rollout. If you find yourself about to call `train_dt`, you have left the fence.

> **This document supersedes every earlier discussion of P5.3.** Where it disagrees with
> `PROJECT_PLAN` §6, §10, `docs/plans/p4.4.md:295` or any chat, this document wins — **except** where
> it disagrees with the **repo**, in which case the repo wins and you flag it in the Return Packet
> (CLAUDE.md §2).

---

## 1. Why this task exists

`PREREGISTRATION` **A9** names exactly one contribution as our own: **probe-calibrated return
prompting**. Everything else in the paper is a study; that is the mechanism.

Its situation today, stated plainly:

- **P4.3 measured the prompt as a weak lever in-domain.** Sweeping the return target across a
  **13,000-unit** grid moved held-out ATT by **0.9026** (`docs/data/p4_3_rtg.json`,
  `in_support_vs_att.att_range = [104.55635786186343, 105.45900353182209]` — verified by the
  coordinator on 2026-08-24 by opening the artifact).
- **P7.0's cross-domain gate DID NOT RESOLVE** (Branch C, 2026-08-16). So the cross-domain axis, on
  which §1 rests the component's entire case, has no verdict either.
- **Therefore nothing currently establishes that the prompt is a lever at all.**

§1b's **C4** states the trap precisely, and it is the reason this task is not just "one more
ablation". An inference-time sweep **cannot distinguish**:

1. the model is **robust** to the target — a flat but real response surface; from
2. the model learned to **IGNORE** the RTG token, because our corpus lacks the return spread that
   would make the conditioning identifiable.

**Under (2), inertness is not an excluded explanation — it IS the mechanism.** And a Decision
Transformer whose RTG token carries nothing is **BC with extra parameters**, which is exactly what
§1b's **R1** looks like from outside (BC within 1.51 ATT of the DT).

**P7.2 — the paper's only named contribution — is a calibration protocol for that token.** If the
token is inert, P7.2 is a protocol for calibrating a knob that is not connected to anything, and we
would find that out after building it. **This task is the cheap test that runs first.**

### What this task can settle, and what it deliberately cannot

| Question | Settled here? |
|---|---|
| Do the trained DT's **actions** change when the return prompt changes? | **YES — decisively, and with no training.** |
| Is the 0.9026 ATT figure a flat *policy* or a flat *outcome* of a moving policy? | **YES.** These are different findings and the paper must not conflate them. |
| Would a DT trained **without** the prompt perform the same? | **NO — that is P5.3b.** Inference-time invariance does not prove training-time irrelevance: a prompt can shape the weights and then sit unused. |
| Is inertness a property of the **data** (C4's hypothesis)? | **NO — P5.3b.** This task produces the spread table that lets P5.3b *register* that question properly. |

⚠️ **This is not a stop rule.** Even a flip rate of exactly zero does not cancel P5.3b. Report the
finding; do not draw the conclusion the finding merely suggests.

---

## 2. Frozen interface contracts — read these from disk before you plan

**Every path and signature below was read from disk by the coordinator on 2026-08-24. Read them
again yourself. If any disagrees with what is written here, the repo wins — say so in the packet.**

### 2.1 The model and where the RTG enters

`agent/DTAgent.py` — **our own P4.1 deliverable, NOT in the frozen set**, but every merged DT number
flows through it (P4.2, P4.3, P4.6, P4.7).

```
:96    TOKENS_PER_STEP = 3                    # (RTG, state, action) per decision step
:99    @dataclass(frozen=True) class DTConfig:
           state_dim, n_actions,
           context_length=20, n_layer=3, n_head=1, d_model=128,
           dropout=0.1, max_ep_len=360
:126   DTConfig.to_json_obj()   :139 DTConfig.from_json_obj()   # checkpoint round-trip
:279   self.embed_rtg = nn.Linear(1, config.d_model)
:294   DecisionTransformer.forward(rtg, state, action, timestep,
                                   attention_mask=None, avail_mask=None) -> (B, K, n_actions)
:317   rtg must be exactly (B, K, 1); already scaled by the caller
:341-348  tokens = stack([embed_rtg(rtg)+t, embed_state(state)+t, embed_action(a)+t])
                     .reshape(B, 3K, d_model)
:364   the STATE token (index 1 of each step) predicts that step's action
:406-407  DTAgent.__init__(..., target_rtg=None, rtg_scale=None, ...)
:450-451  rtg_scale == 0.0 is refused; it divides the RTG input
:486   _build_model: self.model = DecisionTransformer(self._config)
:522   current_rtg() -> {ix_id: self._target_rtg - context.reward_sum}
:596   rtg_out[start+offset, 0] = np.float32(value / self._rtg_scale)
:662   rtg = self._target_rtg - reward_sum          # the inference-time decrement
:741   save(): writes format_version, config, model, target_rtg, rtg_scale, stats, provenance
:776   load(): REFUSES any format_version != CHECKPOINT_FORMAT_VERSION ("dt-checkpoint/1.0")
:801   load(): self.model = DecisionTransformer(config)   # <-- reconstructs the BASE class
```

🚨 **Line 801 is why this task's mechanism is a config field and not a subclass. Read §4.1.**

### 2.2 What already exists and must be reused, not rebuilt

| Path | What it gives you |
|---|---|
| `offline/rtg_calibration.py:158` | `DECLARED_GRID` — P4.3's **nine** frozen targets: `0, −1000, −2000, −3000, −4000, −5762, −7500, −9991, −13000` |
| `offline/rtg_calibration.py:597` | `agent_with_target(gym_env, checkpoint_path, *, declared_gradient_steps, target_rtg, device)` — **loads THEN applies the target, in that order.** Its docstring records why: `load` overwrites `_target_rtg` from the payload, so a target passed to the constructor is silently discarded and every grid point runs P4's original target. **Use this. Do not hand-roll it.** |
| `offline/rtg_calibration.py:528` | `training_rtg_range(checkpoint_path) -> (min, max)` |
| `offline/dataset.py:235` | `RtgSummary` — `count, min, max, mean, std, quantiles`, **already computed per intersection and stored inside every committed `NormalizationStats`** |
| `offline/dataset.py:247` | `NormalizationStats`; `:896 save_stats`, `:925 load_stats` |
| `offline/dataset.py:390` | `_returns_to_go(rewards)` — the RTG definition itself |
| `offline/dataset.py:444` | `TrajectoryWindowDataset` |
| `offline/horizon_metric.py:66` | `horizon_rollout(...)` — the live-rollout path |
| `offline/method_tier_grid.py:1180` | `canonical_digest_of(path)` → `agent/OfflineBaselines.canonical_state_dict_digest` |
| `offline/method_tier_grid.py:1188,1246` | `assert_reused_checkpoint_identity`, `assert_reused_cells_reproduce` |
| `docs/data/p4_6_grid.json`, `p4_7_grid.json` | the committed `dt` column: **8 hz1x1 tiers**, 5 seeds, 40,000 steps, held-out draws **1000–1099** |
| `docs/data/p4_6_declaration.json` | `seeds = [101,202,303,404,505]`, `declared_gradient_steps = 40000`, `env_settings`, `held_out_draws` |

### 2.3 Non-negotiable platform contracts

```python
info = env.reset(seed=42)                                # returns info ONLY
reward, terminated, truncated, info = env.step(action)   # reward FIRST
```
C6 alignment, the ID-keying rule, and `docs/CONTRACTS.md` apply unchanged.

---

## 3. Scope fence — what NOT to build

**Out of scope. Building any of these is a scope breach, not initiative:**

1. ⛔ **Any training of any model.** No `train_dt`, no new checkpoints. P5.3b trains.
2. ⛔ **`rtg_shuffled`, `rtg_noise`, or any second ablation mode.** One mode: `"zero"`. A shuffled
   variant is the natural follow-up **if** the primary is ambiguous, and it is P5.3b's to register.
3. ⛔ **Touching `TOKENS_PER_STEP` or the 3-token interleaving.** A 2-token variant changes the
   sequence length, the attention pattern and the state-token index at `:364` — the alignment
   convention every merged DT number depends on. **`rtg_mode="zero"` gives an information ablation
   with the architecture held exactly fixed, which is the stronger comparison anyway.**
4. ⛔ **Editing `offline/method_tier_grid.py`.** In particular do **not** add to its `METHODS` tuple:
   `:1701` records that `grid_comparisons` emits pairs in `METHODS` order, so a new entry would
   change the comparison enumeration of two merged artifacts.
5. ⛔ **Context-length K.** That is P5.3c and a separate brief.
6. ⛔ **Re-running or regenerating any P4.6 / P4.7 artifact.** You read them. You do not write them.
7. ⛔ **Any SUMO or cross-backend work.**
8. ⛔ **Fixing `DEFERRED` 43's `match=` sites, or P5.2's open minors MN-1/3/4/5.** Not this task.

**In scope, and nothing else:** the `rtg_mode` mechanism; the sensitivity probe; the two Gate-0
tables; the artifact; the tests.

---

## 4. Per-file requirements

### 4.1 `agent/DTAgent.py` — the mechanism (RULED; implement as specified or come back)

**Add an `rtg_mode` field to `DTConfig`, defaulting to today's behaviour.**

```python
rtg_mode: str = "conditioned"        # "conditioned" | "zero"
```

- `__post_init__` **refuses** any value outside those two, naming both.
- `to_json_obj()` emits it. **`from_json_obj()` defaults it to `"conditioned"` when the key is
  absent**, so every one of the 225 existing checkpoints loads unchanged.
- `DecisionTransformer.forward` zeroes the RTG when the mode says so, **before** the existing shape
  validation is bypassed — i.e. validate first, then substitute, so an ill-shaped `rtg` still raises.
- `DTAgent.__init__` gains an `rtg_mode: str = "conditioned"` keyword that reaches
  `_config_template`.

**Why a config field and not a subclass — ruled 2026-08-24, do not re-litigate:**

- **The decisive reason is `agent/DTAgent.py:801`.** `DTAgent.load` reconstructs
  `DecisionTransformer(config)` — the **base class**. A `NoRTGDecisionTransformer` subclass would
  write a checkpoint with identical `state_dict` keys and shapes, and the ordinary loader would
  rebuild it as a **conditioned** model and evaluate it happily. **A plausible number from the wrong
  model is the exact failure this repo exists to prevent.** With a config field, the mode travels
  *inside the checkpointed config* and the right model is rebuilt automatically.
- **It is the project's own precedent.** `SpatialDTConfig` carries `spatial_mixing: bool`, and
  `dt_nomix` — P5.1's and P5.2's control arm — is that flag set `False`
  (`tests/test_spatial_dt_agent.py:110-112`).
- A subclass would additionally have to override `_build_model` **and** duplicate `load`'s
  validation logic, which then drifts from the original.

🔒 **THE PRICE OF THIS ROUTE IS ONE OBLIGATION, AND IT IS §6's LOAD-BEARING TEST: prove the default
path did not move.** See §6.1.

### 4.2 `offline/rtg_ablation.py` — NEW FILE, the probe and the tables

A CLI module in the shape of the existing campaign modules. Subcommands:

#### `probe` — the instrument

**Teacher-forced RTG sensitivity.** For a given committed `dt` checkpoint and a set of logged
episodes from the corpus the checkpoint was trained on:

- Replay the episode's **logged states and logged actions** through the model. The states are
  therefore **identical across interventions by construction**, so any difference in the output is
  attributable to the RTG alone. This is the whole reason for teacher forcing.
- Build the baseline RTG sequence exactly as `DTAgent` builds it at inference — `target − cumulative
  reward`, then divided by `rtg_scale` (`agent/DTAgent.py:596,662`) — using the checkpoint's own
  recorded `target_rtg`.
- Interventions, **fixed here, before any number is computed**:

  | key | intervention | what it measures |
  |---|---|---|
  | `grid_g0 … grid_g8` | the target replaced by each of `DECLARED_GRID`'s **nine** values | **the direct explanation of P4.3's 0.9026** — same grid, action-level |
  | `zero` | RTG identically 0 in scaled units | exactly what a `rtg_mode="zero"` model would see |
  | `frozen` | RTG held at the target, **never decremented** | quantifies a hazard `agent/DTAgent.py:560,628` documents in prose and nobody has measured |

- **Reported per intervention, per (tier, seed):**
  - **`flip_rate`** — the fraction of decision steps whose `argmax` action differs from baseline.
    **This is the headline.**
  - **`tvd`** — mean total-variation distance between the softmax distributions.
    ⚠️ **Required, and not decoration: a flip rate of 0 with a large TVD means the logits moved a
    great deal and never crossed a decision boundary — a different finding from an inert token, and
    the paper must be able to tell them apart.**
  - **`mean_abs_logit_delta`**, and the count of steps compared.
- **Never report a bare pooled mean.** Report **per seed** and the spread across seeds. P5.1's
  lesson, in the plan's own words: *a mean over those five is a summary that hides its own subject.*
- Masked actions: apply `avail_mask` exactly as the model does, from the logged `avail` stream.
  (`DEFERRED` 21: every mask in this corpus is all-True — so this must not *change* anything, and a
  test should say so rather than the code assuming it.)

#### `crosscheck` — the same quantity by a second route, REQUIRED

The probe measures sensitivity on **behaviour-policy states**. That is the distribution the
conditioning was learned on, and it is *not* the distribution the DT visits when it drives.
**State that limitation in the artifact**, then bound it:

- **One** tier, **one** seed, **one** held-out draw, **two** live `horizon_rollout` runs under two
  different targets (`DECLARED_GRID`'s endpoints), comparing the full action sequences.
- This is minutes of CPU and it converts *"on behaviour states"* into *"and also on the model's own
  states, for one cell"*. It is the CLAUDE.md §2 rule — **critical quantities get computed twice, by
  a different route** — applied to the one number this task exists to produce.

#### `tables` — Gate 0's two tables, the inputs P5.3b's registration needs

1. **Return / RTG spread, per hz1x1 tier**, for all **8** tiers present in P4.6 + P4.7
   (`fixedtime, mappo1000, mappo500, maxpressure, mix33, mix50, mix67, random`):
   - per-episode return: `mean, sd, IQR, min, max`;
   - ⭐ **the between-episode sd of the RTG at fixed timesteps `t ∈ {0, 90, 180, 270}`, and pooled.**
     **This, not the episode-return spread, is the quantity C4's hypothesis is about** — the model
     sees an RTG at *every* step, so identifiability is about how much that value varies across
     episodes at a given point in the episode;
   - the same in **scaled** units (divided by the tier's `rtg_scale`), because that is what the
     network actually receives.
   - **Read `RtgSummary` out of the committed `NormalizationStats` AND recompute it independently
     from the raw episode arrays; assert they agree.** Two routes, per CLAUDE.md §2.
2. **δ per tier** — the DT's own paired margin over its behaviour reference, read from
   `behaviour_comparisons` in the committed `p4_6_grid.json` / `p4_7_grid.json`. On `mappo1000` this
   must reproduce **A6's registered δ = 0.6263**; assert that, because it is a free check that you
   are reading the field A6 meant.

#### `report` — one artifact

`docs/data/p5_3a_rtg_probe.json`. It carries the probe results, both tables, the crosscheck, the
limitation sentence, and **a `runtime` block with the measurement/written-at commit split** that
`DEFERRED` 39 established and P4.6 proved in use.

⛔ **The artifact emits NO VERDICT.** No "the token is inert", no "A9 survives". It emits measured
quantities; the reading is the coordinator's and goes in the packet's discussion. `assert_no_verdicts`
(`offline/method_tier_grid.py:1838`) exists — use it or an equivalent, and test it.

---

## 5. The filesystem-mutation barrier

Every write and every delete happens **after** all validation. This bug has appeared twice in this
project, and **P5.2's BL-1 destroyed six irrecoverable training records in an un-backed-up tree**
three days ago. `output/` is gitignored and has no backup.

- Write to a scratch `out_dir`. **Never** write into `output/p4_6/`, `output/p4_7/`, `output/p4_dt/`
  or `output/p5_*`.
- **Never** delete anything under `output/`.
- Re-verify `SHA256SUMS_p4_3.txt`, `SHA256SUMS_p4_6.txt` and `SHA256SUMS_p4_7.txt` **after** your run
  and paste the counts. They were `10/10`, `125/125` and `112/112` on 2026-08-24.

---

## 6. Tests

Write them **first**, run them, confirm they fail **for the right reason**, then implement.

### 6.1 ⭐ THE LOAD-BEARING TEST — the default path did not move

**This is the test the whole task rests on, because it is the one that protects five merged tasks.**

> **With `rtg_mode` present and defaulted, re-evaluate a committed `dt` cell end to end and assert
> every per-episode ATT reproduces the committed value bit-exactly.**

- Take one `(tier, seed)` from `docs/data/p4_6_grid.json`, its checkpoint from `output/p4_6/`, and
  its committed per-episode ATT array.
- Verify the checkpoint's **canonical digest** against the committed record first
  (`canonical_digest_of`) — that proves you are testing the weights the artifact describes, not a
  file with the right name (`DEFERRED` 29).
- Re-roll it on its held-out draws through the changed code. Assert **bit-exact** equality on every
  episode, not `np.allclose`.
- **Then mutate, both directions, and paste both failures:**
  1. force `rtg_mode` to `"zero"` unconditionally → this test **must fail**;
  2. make `from_json_obj` default to `"zero"` instead of `"conditioned"` → this test **must fail**.

  If either mutation leaves the suite green, the test does not protect what it claims to and the
  task is **BLOCKED** — say so, do not weaken the test.

⚠️ If a full re-roll needs CityFlow and is slow, it may be gated on the corpus env var like the
existing corpus tests — **but it must actually run on this machine and its real output must be in the
packet.** A test that only ever skips is not evidence.

### 6.2 The arm-validity test — and it is the probe pointing at itself

- A model with `rtg_mode="zero"` produces **bit-identical logits** under two very different
  `target_rtg` values. **Must hold.**
- A model with `rtg_mode="conditioned"` and **randomly initialised** weights produces **different**
  logits under the same two targets. **Must hold** — this proves the instrument can see a difference
  at all, so a zero flip rate on a *trained* model is a finding rather than a broken probe.

  🚨 **Without this positive control, `flip_rate = 0` is uninterpretable.** This project has twice
  shipped a harness that returned the same answer for every input (the 2026-08-03 harness that
  returned `BLOCKED` for every path including its control). **This control is not optional.**

### 6.3 Probe correctness

- RTG sequence construction reproduces `DTAgent`'s own, computed by an **independent route** —
  `np.cumsum` over the logged rewards, not by calling the function under test.
- `flip_rate` on an intervention identical to baseline is **exactly 0.0**; on a fixture where a known
  number of steps differ, it is exactly that number over the count. Fixture counts must **not** be
  round multiples that make `ceil`/`floor` or `n/2` agree by accident (P5.2's **MN-5**).
- TVD is in `[0, 1]`, is 0 for identical distributions, and is 1 for disjoint point masses.
- `avail_mask` is applied: a fixture where masking changes the argmax must change the flip rate.
- The `frozen` intervention differs from `baseline` on a fixture where rewards are non-zero, and
  **equals** it when every reward is 0 — an equivalence that must be asserted, since it is the exact
  condition under which the documented hazard is harmless.

### 6.4 Tables

- δ on `mappo1000` equals **0.6263** (A6's registered value).
- The two independent routes to `RtgSummary` agree.
- A tier absent from both grids raises, naming the tier — it does not silently produce an empty row.

### 6.5 Hygiene

- `scripts/check_test_hygiene.sh` (**no arguments** — with arguments it scans a subset) must show
  **16** findings and **none in a file you created**. The 16 are `DEFERRED` 45's inherited TH006s.
- **Every mutation you run gets its failure output pasted**, not summarised as a count. P5.2's
  **MN-6** was exactly this: 3 of 22 pasted.
- Report whether the suite run was thread-pinned (`DEFERRED` 41).

---

## 7. Definition of Done

- [ ] Code complete, no placeholders
- [ ] Tests written **and executed**; real `pytest` tail pasted
- [ ] **§6.1 passed, and BOTH of its mutations failed, with both failures pasted**
- [ ] **§6.2's positive control passed**
- [ ] `git diff --stat` shows **zero** frozen-file modifications
- [ ] Zero new dependencies
- [ ] `docs/data/p5_3a_rtg_probe.json` written, carrying no verdict
- [ ] The three `output/` manifests re-verified after the run, counts pasted
- [ ] Committed on `task/p5.3a-rtg-probe`
- [ ] Return Packet at `docs/returns/P5.3a.md` from `docs/returns/TEMPLATE.md`
- [ ] **`PROJECT_PLAN` §6's `P5.3a` checkbox ticked in the merge commit** (CLAUDE.md §6)
- [ ] **AI-assistance record** complete — four lines, written as you go, never reconstructed
      (CLAUDE.md §8)

**Review:** this is critical-path code. `contract-reviewer` runs on the diff before any merge.
Mutation evidence, not reading.

---

## 8. Return Packet — what it must answer beyond the template

1. **The flip-rate table**, per tier, per seed, per intervention. Never a bare pooled mean.
2. **Does the trained DT's policy respond to the return prompt at all?** Answer with the number, and
   **state which of the two readings it supports** — flat policy, or moving policy with flat outcome.
   ⚠️ **Do not write "the token is inert" unless the flip rate is 0 across the whole grid AND the TVD
   is negligible. If TVD is large and flips are 0, say exactly that; it is a different result.**
3. **The crosscheck**: did the live rollout agree with the teacher-forced probe?
4. **Both Gate-0 tables**, plus the tier ordering the spread table implies.
5. **Timing**, measured: how long the probe took per (tier, seed). P5.3b needs it and
   Amendment R2 forbids armchair estimates.
6. **Anything you found that this brief got wrong.** The coordinator read these signatures on
   2026-08-24; if `train_dt`, `agent_with_target` or `RtgSummary` is not as described, that is a
   finding and it goes in the packet.

---

## 9. Registered before any number exists

**So that a referee can check the goalposts did not move.** Written 2026-08-24, before any P5.3a
number of any kind exists.

- **The nine grid points are P4.3's, unchanged.** We do not choose a range that flatters a result.
- **The headline statistic is `flip_rate`.** Chosen because it is the direct analogue of P5.1's
  graph-ablation measurement (*"48.83 % of actions flip without the graph"*), so it is comparable
  to something we already report.
- **The tier-selection rule for P5.3b is declared NOW, before the spread table is computed:** the
  three tiers are (i) **`mappo1000`**, the headline tier and the one A6's δ and P4.3's sweep were
  measured on; (ii) the tier with the **largest** measured between-episode scaled-RTG sd; (iii) the
  tier with the **smallest**. If (ii) or (iii) is `mappo1000`, take the next one in that direction.
  **A rule, not a choice.**
- **Both outcomes are publishable and neither is a failure.** A responsive prompt gives A9 an object
  and P7.2 a subject. An inert prompt is a **negative result about a mechanism the field is currently
  building on**, and it explains R1 — it would say our DT is BC with extra parameters *on this
  corpus*, which is a finding about the corpus as much as about the model.
  ⚠️ **The scope fence on that sentence, and it is binding: our corpora, our scenario, our budget.
  Not "return conditioning does not work".**

---

## 10. Practicalities

- Interpreter is always `.venv/bin/...`. Shell state does not persist between tool calls.
- Long runs go to a **user-started `tmux`** session, not into your own session. This task should not
  need one; if you think it does, say so before starting it.
- The coordinator commits only from the main tree. You work on your branch.
- ⚠️ **`main` is currently RED in CI**, awaiting
  `docs/patches/ci_gate_ceiling_104_and_chain_walk.patch`. That is `tests/test_ci_gate.py` and
  `.github/ci/ci_baseline.json` only. **It is not yours to fix and it does not touch your files.** If
  your branch shows those two tests failing, that is the inherited state — say so in the packet and
  do not edit either file.

---

# ✅ AMENDMENT A — 2026-08-25, ruled at GATE 1

**All five questions answered, all twelve findings accepted or corrected, and one ruling is a
withdrawal of something this brief registered six days ago and got wrong.** The plan file is
`docs/plans/p5.3a.md` (finding 9 is right; repo convention wins over anything said in chat).

## A0 — what the coordinator re-verified first-hand, and what he took on report

**Re-verified by running the command, 2026-08-25:** finding **1** (`scenarios/draws/cityflow1x1/`
holds exactly **11** dirs — `draw_0000…0005` and `draw_1000…1004`; `.gitignore:227` ignores
`scenarios/draws/`; `git ls-files` returns **0**), finding **12** (`output/p4_6/checkpoints/` carries
only `fixedtime, mappo500, maxpressure, random`; `mappo1000`'s DT column is `output/p4_dt/dt_seed*.pt`),
finding **3** (δ is stored signed; `dt@mappo1000 = −0.6262756469347375`, `dt@fixedtime =
−5.684341886080802e-16`), finding **5** (only committed `NormalizationStats` copy is inside the
checkpoint payload — `find` over `output/`, `datasets_v11/`, `docs/data/` returns nothing), and
`method_tier_grid.py:2035,2104,2114` (`--draws-root` defaults to `scenarios/draws` and resolves each
draw through `draw_config_path`, so **the evaluation path really does need the draws on disk**).

**Taken on your report, and named as such:** findings 4, 6, 7, 8, 10, 11 and the 17 signature checks.

⭐ **Finding 1 is not just a blocker, it is a LOSS, and it belongs in the record:** P4.6 and P4.7
materialised draws 1000–1099 inside their own worktrees, `scenarios/draws/` is gitignored and therefore
**per-worktree**, and retiring those worktrees deleted them. `PROJECT_PLAN` §10 already carries the rule
*"before retiring any worktree, run `git status --porcelain --ignored` and ask whether a merged artifact
references anything ignored there"* — **the rule existed and this still slipped through**, because the
merged artifacts reference the draws by **id** and an id looks like data rather than like a file.

## A1 — Q1: MATERIALISE 1000–1099 into `scenarios/draws/`, with a bit-identity control first

**Ruled: the shared root, not a scratch root.** It is gitignored, idempotent, the resolver already
defaults to it, and **P5.3b needs the same 100 draws** — so this restores a lost shared resource
rather than creating a private one. Restricting §6.1 to the 5 surviving draws is **refused**: the
committed column has 100 and comparing 5 while the artifact says 100 is the sample-versus-population
error this project exists to catch.

> 🔒 **REQUIRED FIRST, AND IT IS FREE: regenerate `draw_1000…1004` into a scratch root and assert
> BIT-IDENTITY against the five that already exist on disk.** If they reproduce, the generator is the
> same function that produced the originals and the other 95 are trustworthy. **If they do NOT
> reproduce, STOP and report `BLOCKED`** — every downstream number would then be measured on draws
> that are not the draws P4.6 evaluated on, and it would be undetectable afterwards.
> ⚠️ **Do not overwrite the surviving five.** Generate to a scratch root, compare, and only then fill
> in the 95 that are missing. Filesystem-mutation barrier: validate, then write.

## A2 — Q2: `output/p5_3a/` IS the right name; my fence was overbroad and I withdraw it

§5 said *"never write into `output/p5_*`"*. I meant **never write into an EXISTING campaign
directory**. As written it forbids the one name that follows the project's own convention
(`output/p4_6`, `output/p5_1`, `output/p5_2`).
> **Corrected fence: create and write `output/p5_3a/`. Never write into, and never delete from,
> `output/p4_3`, `output/p4_dt`, `output/p4_probe`, `output/p4_4`, `output/p4_5`, `output/p4_6`,
> `output/p4_7`, `output/p5_1`, `output/p5_2`, `output/p7_0`, `output/p8_3` or
> `output/checkpoints*`.** Naming them is better than a glob, which is what went wrong here.

## A3 — Q3: YES, add it — one line, and the packet must be honest about the test

Add `rtg_mode=config.rtg_mode` to `from_checkpoint`'s constructor call. **Reason, and it is not the
one offered:** `from_checkpoint` builds the model **twice** — once via `__init__` → `_build_model`
(`:465-466, :486`), once in `load` (`:801`) — and the first build would otherwise construct a
*conditioned* model for a *zero* checkpoint. It is inert today because the second build replaces it
inside the same function. **Add it so the code says what is true**, rather than leaving a comment
explaining why a wrong value is harmless.
> ⚠️ **Required in the packet:** the round-trip test for this **passes with and without the line**,
> so it is a **regression guard, not a discriminating test**. Say that. A test whose docstring claims
> power it does not have is what `docs/reviews/P5.2.md` filed under *"theatre"*.

## A4 — Q4: TWO cells, and the second is chosen by a rule rather than by preference

1. **`mappo1000`, seed 101, from `output/p4_dt/dt_seed101.pt`** — the headline tier, the one A6's δ
   and P4.3's whole 13,000-unit sweep were measured on. Its committed column is the reused one; verify
   it through `assert_reused_checkpoint_identity` exactly as P5.2's Gate 1 did.
2. **The tier with the LARGEST committed `RtgSummary.std` among those with checkpoints under
   `output/p4_6/checkpoints/`, at seed 101.** ⭐ **The rule is not decoration: the identity test is
   strongest where the RTG carries the most variance, because that is where forcing `rtg_mode="zero"`
   does the most damage — and mutation 1 of §6.1 must actually fail.**

> 🚨 **And this is why one cell was not enough: `dt@fixedtime`'s δ is −5.68e-16 and P5.2's reviewer
> showed the grid4x4 DT reproduces the fixed-time controller entry for entry, 0 of 5760 actions
> differing. Had §6.1 landed on `fixedtime`, forcing `rtg_mode="zero"` might have changed nothing,
> mutation 1 would have "passed" by surviving, and the load-bearing test would have certified
> nothing.** Both cells must reproduce bit-exactly; **both mutations must fail on cell 2**, and the
> packet reports whether they also failed on cell 1.

## A5 — Q5: yes, the checkpoint-embedded `stats` payload is the committed `NormalizationStats`

Confirmed by search: there is no other copy. Say so in the artifact, so the next reader does not go
looking for a file.

## A6 — Finding 4 is RIGHT and it corrects THIS BRIEF: §4.2 asked for two routes to two different numbers

§4.2 said *"read `RtgSummary` out of the committed `NormalizationStats` AND recompute it independently
from the raw episode arrays; assert they agree."* **You have shown those are not the same quantity.**
`RtgSummary` is over the **concatenated per-step RTG of every stream in the tier's whole training
split**, `ddof=0`, fitted over `TierSpec.dirs` — not over per-episode returns, and not over the
200-stream subsample.
> **Corrected: the independent recomputation must target THE SAME POPULATION AND THE SAME ESTIMATOR**
> (concatenated per-step RTG over the same dirs, `ddof=0`). **The per-episode-return statistics are a
> SEPARATE ROW of the table and are not a cross-check of anything.** ⚠️ **Asserting agreement between
> them would have failed for a correct implementation on `random` and the mixtures — a test that
> condemns a correct artifact, which is the class this repo refused on 2026-08-19.**
> **Both rows are still wanted.** The between-episode RTG spread at fixed timesteps (§4.2) remains the
> quantity C4's hypothesis is about and is neither of the two above.

## A7 — 🚨 THE δ RULE IS WITHDRAWN. It was mine, it is six days old, and it is broken in BOTH directions

§4.2 and §6.4 registered *"δ per tier = the DT's own paired margin over its behaviour reference"*.
**Measured across all eight tiers from the committed artifacts, 2026-08-25:**

| tier | δ (signed) | CI contains 0? | as an equivalence margin |
|---|---|---|---|
| `mix33` | −214.1190 | no | **vacuously permissive** |
| `mix50` | −159.5624 | no | vacuously permissive |
| `mix67` | −107.0298 | no | vacuously permissive |
| `random` | −8.5076 | no | usable |
| `mappo500` | −1.4360 | no | usable |
| `maxpressure` | **+0.6886** | **YES** | **a DEFICIT, not a margin** |
| `mappo1000` | −0.6263 | no | usable (= A6's 0.6263) |
| `fixedtime` | **−5.68e-16** | **YES** | **impossible by construction** |

**δ spans ELEVEN ORDERS OF MAGNITUDE.** On `fixedtime` a ±5.7e-16 band means **no measured difference
could ever be declared equivalent**; on `mix33` a ±214 band on a scenario whose ATT is ~105–300 means
**every result would be declared equivalent**. On `maxpressure` the "margin" is the DT *losing* to its
behaviour policy. **A rule that cannot return one of its answers on part of its domain is exactly
`DEFERRED` 46's defect — Q1's unfalsifiable-by-construction prediction, which was also mine — and I
reproduced it in a brief written six days after logging that lesson.**

> ⛔ **WITHDRAWN as a decision rule. It survives as a MEASURED TABLE and nothing more.**
> **P5.3a emits, per tier:** signed δ, `ci95_low`, `ci95_high`, `ci95_half_width`, `wilcoxon`,
> `rank_biserial`, `wins/losses/ties`, `n_shared_draws`, and a derived boolean
> **`behaviour_margin_degenerate` = (`ci95_low ≤ 0 ≤ ci95_high`)** — **2 of 8 tiers are `true`**, so
> the flag is not hypothetical.
> **The decision rule for P5.3b is NOT registered here and will not be this one.** That is the
> deferral working as designed: the rule is being chosen *because* the table exists, instead of the
> table being made to fit a rule chosen in advance.
> ✅ **A6's δ = 0.6263 is untouched.** It was registered for `mappo1000` and remains correct there;
> the error was mine in generalising its *form* to seven other tiers.

## A8 — ⭐ NEW REGISTERED PREDICTION, written before any probe number exists

**`fixedtime` will show `flip_rate = 0.0` on every intervention, including `zero` and the two grid
endpoints.** Basis, stated so this is a prediction and not a hedge: `dt@fixedtime`'s δ is −5.68e-16,
i.e. the DT's ATT is bit-equal to the behaviour policy's, and P5.2's independent reviewer showed the
grid4x4 DT reproducing the fixed-time controller **entry for entry, 0 of 5760 actions differing**. A
deterministic cycle has no room for the prompt to act.
> **This is a control on the PROBE, not a result about the prompt.** If `fixedtime` shows non-zero
> flips, **the probe is suspect before the finding is** — that is the direction of inference, and it is
> registered now so it cannot be reversed later. ⚠️ **The hz1x1 evidence is the ATT identity; the
> action-level identity was measured on grid4x4. So this is a prediction, not a restatement.**
> **Add `fixedtime` to the probe's cell set for this reason.**

## A9 — Finding 5: the divergence from `SpatialDTConfig` is DELIBERATE, and the docstring must say why

`SpatialDTConfig.from_json_obj` hard-raises on an absent `spatial_mixing` (`:174-176`); `rtg_mode`
must **default**. You are right that these are opposite, and the difference is not an inconsistency:
**`SpatialDTConfig` was born with its flag and has no checkpoints predating it. `DTConfig` has 225
checkpoints in the wild** (`PROJECT_PLAN` §10), and a hard raise would make every one unloadable.
> **Required: `from_json_obj`'s docstring states the asymmetry, names the 225, and names
> `SpatialDTAgent.py:174-176` as the contrasting case**, so the next reader sees a decision instead of
> a discrepancy. **`to_json_obj` always emits the key**, so only pre-existing checkpoints ever take
> the default. **A test must cover both:** an 8-key payload loads as `"conditioned"`, and a 9-key
> payload round-trips its value.

## A10 — Findings 6, 7, 8, 10, 11: accepted as stated

**6** — `p4_dt_config.json["architecture"]` gaining a 9th key affects nothing here, because **P5.3a
writes no checkpoints**; note it forward for P5.3b, which will. **10** — 16 TH006, matching §6.5;
thank you for refusing to report the truncated 8, which is `DEFERRED` 45's exact mechanism.
**11** — measure the timing; Amendment R2 forbids armchair costing and my own 52 h estimate is the
cautionary case.

## A11 — What does NOT change

The scope fence (§3) except as corrected in **A2**. No training. One mode, `"zero"`. No touching
`TOKENS_PER_STEP`, `METHODS`, or any P4.6/P4.7 artifact. §6.1 stays the load-bearing test and its two
mutations stay mandatory. §6.2's positive control stays mandatory. The artifact still emits **no
verdict**.

## A12 — Gate 1 verdict

✅ **PROCEED TO PLAN.** Write `docs/plans/p5.3a.md` incorporating A1–A11 and stop at Gate 2. **The
plan must state, in its own words, what §6.1 would fail to detect** — every phase boundary in this
project that went wrong went wrong on an assumption nobody wrote down.

---

# ✅ AMENDMENT B — 2026-08-25, ruled at GATE 2

**Verdict: APPROVED TO CODE**, with B0–B5 folded into the plan first. **Two of the three questions
found errors of mine**, and a measurement made while checking the third **reverses my own provisional
reading of it** — that reversal is B2 and it is the most important paragraph here.

## B0 ⛔ FIRST, AND IT IS NOT COSMETIC: `docs/plans/p5.3a.md` ends with two stray tool-call tags

Lines 372–373 are `</content>` and `</invoke>`. **Delete them and re-read the file's tail before
committing.** The plan is a registration document; a referee-facing artifact that ends in the
scaffolding of the tool that wrote it undermines it for free. ⚠️ **This is `PROJECT_PLAN` §7's *verify
your own edits* rule, which the coordinator has broken twice on markdown tables in two days — so it is
raised as a shared failure mode, not as a criticism.**

## B1 ✅ Q1 — **INCLUDE the three mixture tiers in the probe.** 8 tiers × 5 seeds

R3 registered 5 tiers and excluded the mixtures. **Overruled, and B2 is the reason.** You were right
that Gate 2 is the last moment this is free, and right to force the choice now.
> **Registered: the probe covers the SAME EIGHT TIERS as the tables** — `fixedtime, mappo1000,
> mappo500, maxpressure, random, mix33, mix50, mix67` × 5 seeds = **40 cells**. The checkpoints exist
> (`output/p4_7/checkpoints/mix*_dt_seed*.pt`, and they ARE manifest-covered, unlike cell 1 — see B4).
> **Reason, and it is not symmetry for its own sake: `flip_rate` is only interpretable against row B,
> and a correlation over five points where the mixtures are the only structurally different case is
> not a measurement, it is a gesture.** If the probe returns ~0 everywhere on five single-policy
> tiers, the first question asked will be *"and where the corpus is bimodal?"* — and P5.3a would have
> to answer *"we did not look."*

## B2 ⭐⭐ THE MEASUREMENT THAT REVERSED MY OWN READING — `RtgSummary.std` is mostly the within-episode RAMP, and two tiers can share it for OPPOSITE reasons

I checked your cell-2 rule and computed the **scaled** spread (`std / rtg_scale` — what the network
actually receives, since `agent/DTAgent.py:596` divides). **Measured 2026-08-25 from the
checkpoint-embedded stats at seed 101:**

| tier | raw `std` | `rtg_scale` | **scaled** | pure-ramp prediction: `abs(target) / (2*sqrt(3)) / rtg_scale` | measured / ramp |
|---|---|---|---|---|---|
| `random` | 12149.7093 | 40294 | **0.3015** | 0.2749 | **1.10** |
| `fixedtime` | 9303.5481 | 33225 | **0.2800** | 0.2581 | **1.08** |
| `maxpressure` | 5837.9472 | 24115 | **0.2421** | 0.1570 | **1.54** |
| `mappo1000` | 2188.9709 | 9991 | **0.2191** | 0.1665 | **1.32** |
| `mappo500` | 2297.8980 | 11043 | **0.2081** | 0.1663 | **1.25** |
| `mix50` | 13155.3172 | 40223 | **0.3271** | **0.0428** | **7.65** |

**My provisional read was "the mixtures barely differ — scaled spread is 0.208–0.328 everywhere, a
factor of 1.58, so the spread axis is weak." That read is WRONG and the last column is why.**

- On the five single-policy tiers the marginal std is **1.08–1.54× a pure within-episode ramp**, so
  **65–93 % of it is the deterministic decay of RTG from target to zero across 360 steps** — the same
  shape in every tier, carrying no information about *which* episode this is.
- On `mix50` the ratio is **7.65**, because its `target_rtg` is **−5959** (an expert-like quantile)
  while its `rtg_scale` is **40223** (set by the random component). Its spread is **between-episode
  and bimodal**, which is exactly the informative kind.
- **So `random` at 0.3015 and `mix50` at 0.3271 are nearly equal marginally and are not remotely the
  same quantity.** One is a ramp; the other is a corpus that genuinely contains two populations.

> 🔒 **Consequences, all three binding:**
> **(i)** **My A4 cell-2 rule was stated on the wrong quantity** — "largest `RtgSummary.std`" selects
> partly on ramp amplitude. ✅ **The answer is unchanged: `random` ranks 1st raw AND 1st scaled, so
> cell 2 stands.** The rule's *basis* is corrected to the scaled figure; the selection is not re-opened.
> **(ii)** ⭐ **Row B is promoted from a table row to the task's second real result.** It is the only
> quantity here that separates ramp from information, and **nothing in this project has ever measured
> it.** Report it raw and scaled, and report it beside the ramp prediction above so a reader can see
> the decomposition.
> **(iii)** ⚠️ **P5.3b is warned now: a "narrow vs wide spread" axis chosen on the MARGINAL statistic
> would have picked `random` as its wide endpoint and measured a ramp.** The axis must be built on
> row B. **This is the single most useful thing P5.3a can hand forward, and it was not in the brief.**

## B3 ⚠️ Q2 — YOU ARE RIGHT AND MY SENTENCE WAS WRONG. P5.2's Gate 1 did not use that function

A4 said *"exactly as P5.2's Gate 1 did"*. **Measured: `assert_reused_checkpoint_identity` has exactly
one call site in the repo, `offline/method_tier_grid.py:2269` — P4.6/P4.7's gate. `offline/tier_sweep.py`
has none.** P5.2 used its own `assert_reused_digest` (`tier_sweep.py`), which re-verifies a **file
sha256 against `output/SHA256SUMS_p5_1.txt` AT CONSUMPTION** — a manifest route, not this one.
**I described a route from memory instead of opening the file. That is the project's signature error,
and refusing to assume you knew what I meant was the correct response.**
> **RULED — use both, they check different things and both are cheap:**
> **(a)** cell 1 → `assert_reused_checkpoint_identity(p4_4_training, p4_gate, …)` called directly, as
> you proposed. That is the P4 route and it is what A4 should have said.
> **(b)** cell 2 → `canonical_digest_of` against `p4_6_training.json`'s committed value, **plus** a
> `SHA256SUMS_p4_6.txt` check **at consumption**, in `assert_reused_digest`'s spirit: *a digest
> checked once is not a digest checked when used* (`BRIEF_27` B3(a)).
> **(c)** ⚠️ **Cell 1 cannot have (b) — see B4.**

## B4 🚨 NEW FINDING — `output/p4_dt/` IS IN NO INTEGRITY MANIFEST, AND IT HOLDS THE P4 GATE'S FIVE MODELS

Measured across every `output/SHA256SUMS_*.txt`: **0 lines mention `p4_dt`.** Coverage is
`p5_2` 221 · `p4_6` 125 · `p4_7` 112 · `p5_1` 48 · `p7_0` 36 · `p4_5` 25 · `p4_4` 19 · `p8_3` 18 ·
`p4_3` 10 — and `SHA256SUMS_p4_3.txt`'s ten entries are all `p4_3/eval_dt_g*.json`, **not** checkpoints.
**`output/p4_dt/` and `output/p4_probe/` are uncovered.**
> **`output/p4_dt/dt_seed*.pt` are the five models behind the pre-registered P4 gate, the models P4.3
> swept across 13,000 RTG units, the reused `mappo1000` DT column in P4.6 and P4.7 — and §6.1's cell
> 1.** They are gitignored, unbacked-up **and unmanifested**: the one class of evidence in this project
> with no integrity record at all. ⚠️ **Not P5.3a's to fix** — writing a manifest for someone else's
> merged output is out of fence. **`DEFERRED` 56 records it.** State the gap in the packet: cell 1's
> identity rests on a **file sha256** (`DEFERRED` 29: filename-dependent) with **no manifest behind it**,
> and say so rather than letting the two cells read as equally protected.

## B5 ✅ Q3 — GATE 0 IS APPROVED AND IS NOW MANDATORY. It is yours, and it should have been mine

**You are right that §6.1 is unfalsifiable without it**, and the sharpest form of the argument is the
one your plan does not quite make: **if Gate 0 FAILS, §6.1 cannot be run at all** and the task needs a
different instrument — which is something to discover in four minutes, before the edit, rather than
after. It also converts a red §6.1 from ambiguous into attributable.
> **Registered as a gate with the power to stop the task**, exactly as you wrote it. ⭐ **And it is the
> reason the two mutations mean anything: without a pre-edit reproduction, "the mutation made it fail"
> and "this environment cannot reproduce the column" are the same observation.**

## B6 ✅ Accepted as stated, no change required

Your **self-correction** on cell 1's committed identity (a **file sha256**, not a canonical digest —
`method_tier_grid.py:1222`, with `:1233-1235` explaining that `p4_training.json` never carried one).
Confirmed independently: `p4_training.json` contains **neither** `file_sha256` **nor**
`canonical_digest`; `p4_4_training.json` contains both. **Writing the asymmetry into the plan instead
of smoothing it over is exactly right.**
**R6** — `mappo1000`'s `grid_g5` target is `−5762.0`, which **is** that checkpoint's own `target_rtg`
(verified: `torch.load(...)["target_rtg"] == -5762.0`), so its `flip_rate` must be **exactly 0.0**.
⭐ **A live null control inside the campaign, at zero cost, that neither the brief nor Amendment A
thought of.** **R4**'s stratified stream indices, **§2**'s seven-item blindness list, and test **26**
(recomputing row C on the *wrong* population must DISAGREE) are all stronger than what was asked for.

## B7 🧹 `docs/plans/p5.3a.md` §11 item 12 is already stale

It says *"`main` is red pending `ci_gate_ceiling_104_and_chain_walk.patch`"*. **The patch was applied
at `7ee606a` and CI is green — verified from `gh run list`, not from the commit message.** Rewrite it:
the ceiling is **expected** to breach at P5.3a's merge, and the protocol is
`re_measure_required_at.what_to_do` in the baseline — merge, read the observed count from `junit.xml`,
commit it with its breakdown. **Do not pre-bump it.**

## B8 📋 Fold into the plan, then code

B0 (delete the stray tags) · B1 (8 tiers × 5 seeds; update R3, §6, §11 item 13) · B2 (correct A4's
basis; promote row B and report it against the ramp prediction) · B3 (both identity routes) · B4 (state
the manifest gap in the packet) · B5 (Gate 0 registered) · B7 (fix the CI sentence).
**Nothing else in the plan changes.** Tests first, run them red, then implement. Stop before merge for
`contract-reviewer`.
