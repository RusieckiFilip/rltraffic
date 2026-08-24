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
