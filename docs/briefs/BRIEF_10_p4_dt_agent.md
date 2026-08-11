# BRIEF #10 — P4: the single-intersection Decision Transformer and its pre-registered gate

**Mode:** Claude Code, on a task branch. **Branch:** `task/p4-dt-agent`
**Issued:** 2026-08-07 by the Master chat. **Base:** `main` (P2.6 and P3 both merged).
**Filter:** this is the critical path. Everything since 2026-07-08 exists to make this runnable.

---

## 1. What this is, and why it is different from every task before it

Every previous task could only be *wrong*. **This one can be right and still fail** — and a failure
is a registered outcome, not a setback.

`PREREGISTRATION.md` §9 fixes the gate, verbatim, and it is not renegotiable by you or by me:

> **P4.2 gate (single intersection).** With ATT lower-is-better, on hangzhou_1x1 #1, over ≥5 seeds
> with CIs, both must hold: `ATT_MADT ≤ ATT_MaxPressure` **and** `ATT_MADT ≤ 1.05 × ATT_best_online`.
> Being within 5% of a weak MAPPO while losing to a 1970s heuristic is not a pass. **Failing this
> gate triggers diagnosis before any multi-agent scaling — it does not trigger a redefinition of
> the gate.**

**The numbers you must beat already exist**, measured in the registered metric on draws 1–200
(`docs/data/att_ladder_v11.json`, merged):

| cf_hz1x1, `att_horizon` | value |
|---|---|
| MaxPressure | **176.50** ± 1.76 |
| best online = MAPPO@1000 | **105.46** ± 0.43 |
| **⇒ gate threshold** | **ATT_MADT ≤ 110.73** (and ≤ 176.50, which is implied) |
| random | 422.52 · fixedtime 262.09 · mappo060 281.89 · mappo200 125.03 · mappo500 107.50 |

Write `110.73` into the Return Packet as the number you were trying to beat, computed before you
trained anything. **If you find yourself wanting to move it, that is the finding, and you stop and
report instead.**

⚠️ **Prior evidence says the hard case is not here.** DTLight's own pure-offline DT collapsed on
Grid 4×4 (446.8 from weak data against a 48.39 behaviour policy). P4 is 1×1, where DT demonstrably
works. **Expect the fight at P5.2.** If P4 fails, something is wrong with our pipeline, not with
sequence modelling — diagnose in that order.

---

## 2. Frozen facts you must not re-derive

- **`agent/DTAgent.py` is a permitted path.** Verified at both layers on 2026-08-07: the guard's
  `FROZEN_PATTERNS` names only `agent/base.py`, `agent/utils/utils.py`, `agent/MAPPOAgent.py`, and
  `.claude/settings.json` denies only those three. **There is no `agent/**` glob.** A *new* file
  there is fine; do not touch the three named ones.
- **The loader is done and reviewed — use it, do not re-implement it.** `offline/dataset.py`
  provides per-intersection windows: `rtg (K,1) · state (K,D) · action (K,) · avail_mask (K,A) ·
  timestep (K,) · attention_mask (K,)`, left-padded, plus `dataset.item_meta(i)`.
- **`PAD_ACTION = -1`,** exported for exactly this purpose. Use `ignore_index=-1` in the loss. A
  loss that forgets `attention_mask` then **crashes** instead of training on fabricated action-0
  targets. `attention_mask` remains the mechanism; `-1` is the tripwire behind it.
- **`att_per_step` is float32; `att_horizon` computes in float64.** Comparing the stored value
  against a plain Python float returns `True` (NEP 50 weak-scalar promotion) but against
  `float()`, `np.float64` or an array returns `False`. The safe case is the one you will try first.
  The matrix is in `offline/dataset.py`'s docstring — read it, do not re-derive it.
- **Action masking never binds in this corpus.** All 32000 streams are all-`True`, because
  `acyclic` exposes every green phase. Keep masking in the model (the env raises on illegal
  actions, and it binds under cyclic modes and P6 perturbations) but **you cannot and must not
  claim it as a learned capability, and a no-mask ablation here would measure nothing.**
- **Episodes end by truncation, never termination.** `terminated` is hardcoded `False`. DT/RTG is
  unaffected — no bootstrapping, one shared horizon.

## 3. The leakage rules — this is what an offline-RL venue rejects papers on

`PREREGISTRATION.md` §6, all three binding:

1. **No online model selection.** The reported model is the checkpoint at a **fixed, pre-declared
   number of gradient steps**. Declare that number in `docs/plans/p4.md` **before** training.
   Training curves may be shown; they may not choose the model.
2. **Hyperparameters are tuned on hangzhou_1x1 #1 only, then frozen** for every later scenario,
   tier, perturbation and backend. **P4 is that scenario**, so your configuration becomes the
   project's configuration. Record it as a committed artifact, not as prose.
3. **Baselines get the same tuning budget**, and an untuned baseline is reported as untuned.

**D4 split, binding:** train on draws **1–999** (the corpus holds 1–200); **draws 1000–1099 are
held-out and must never enter training, for any method including baselines.** The loader raises if
you ask; do not route around it. Normalisation statistics are fitted on the **training split only**.

---

## 4. Deliverables — SCOPE FENCE, read this before planning

**This brief is P4.1 and P4.2 ONLY.** P4.3 (RTG-conditioning sweep, probe-calibrated prompting) and
P4.4 (BC, %BC, IQL) are **separate briefs** and are **out of scope**. Do not build them. Do not
build a spatial mixing layer — that is P5.1. The ≤2-source-file limit stands.

### 4.1 `agent/DTAgent.py`
Causal GPT-style decision transformer over `(RTG, state, action)` tokens, `BaseAgent`-compatible
`act()`, action masking from `avail_actions`. Mirror `agent/MAPPOAgent.py`'s constructor shape
(`gym_env, ..., device=None, seed=None`) — read it, do not infer it. No new dependencies: torch,
numpy, stdlib.

### 4.2 The gate run
Train on the **cf_hz1x1 v1.1 corpus**, evaluate over **≥5 seeds with CIs**, report `att_horizon`
**and** `vehicle_count` at the horizon (A5 makes the co-report unconditional), **and the draw ids**
(A5: a comparison is valid only over shared draws, void otherwise).

**Which tier to train on is a decision for your plan file, not an afterthought.** State it and
justify it before training. Be aware of the trap: training on `mappo1000` and beating MaxPressure
proves little, because BC would likely do the same — that is exactly what P4.4 exists to test.

---

## 5. Tests — the load-bearing ones

- **Shape and alignment through the model:** a batch from the real loader flows to a loss without
  a shape error, and `(action == -1) == ~attention_mask` holds on every item.
- **Padded positions cannot contribute to the loss.** Mutate the loss to drop `ignore_index` and
  show it **crashes** — that is the `-1` tripwire earning its place.
- **Masking is applied:** logits at illegal actions are `-inf` before the softmax. Build a
  **synthetic** mask with genuine `False` entries — the corpus cannot test this (§2).
- **Determinism:** same seed ⇒ **tensor-identical** parameters after N steps, compared over the `state_dict` — **not** by file hash, which is structurally incapable of matching across two filenames (2026-08-11).
- **The gate is computed by an independent route.** Do not compare against a number the training
  loop printed; re-evaluate the saved checkpoint through `offline/horizon_metric.py`.

⚠️ **`offline/horizon_metric.py` has an open retro-review** (`docs/reviews/P8.0.md`): its extraction
is verified correct by three routes, but **`rederive_anchors.py` can only retrain, never load a
checkpoint** (finding B1). If you need to evaluate a *saved* model through it, that gap is yours to
close, and it must carry `policy_source` = `"checkpoint"` vs `"retrained"` per queue item 0b.

## 6. Definition of Done
- [ ] `docs/plans/p4.md` first, with the **declared gradient-step count** and the **tier choice**
- [ ] `agent/DTAgent.py` + tests; red-first; mutation proofs pasted
- [ ] Gate evaluated ≥5 seeds with CIs, ATT + `vehicle_count` + draw ids
- [ ] **The gate verdict stated plainly as PASS or FAIL against 110.73**, before any interpretation
- [ ] Full `pytest -q`; count against the 478 baseline
- [ ] `git diff --stat` shows zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P4.md`
- [ ] **§6 checkbox ticked in the merge commit** (CLAUDE.md §6)
- [ ] **Independent review before merge** — critical path

## 7. Return Packet — task-specific
1. The declared step count, and confirmation the reported checkpoint is that one and not a better one.
2. The tier you trained on and why.
3. The gate verdict, with both inequalities evaluated explicitly.
4. Anything in §2 that disagreed with the data. **The repo wins; say so loudly.**
5. **If the gate failed: your diagnosis, and no proposal to change the gate.** A failed gate is a
   registered result and §10 of the pre-registration already says what we publish under it.

---

## 8. Ruling on the P4 pre-flight (Master chat, 2026-08-07)

**You found a defect in this brief and you were right to stop. `110.73` is measured on draws 1–200
— the TRAINING pool — and §4.2 told you to evaluate against it.**

**Protocol: C, and the held-out verdict IS the gate. Not because I prefer it — because A was never
available.** `PREREGISTRATION.md` §8 registers the unit of replication: *"For evaluation, the unit is
the **held-out flow draw** … Comparisons across methods use the **same** held-out draws (paired
design)"*, ≥20 per cell. That was registered on 2026-08-03 and closes the question; I failed to check
it when I computed the threshold from the ladder. **Binding:**

1. **The gate is evaluated on the registered held-out pool, all 100 draws `1000–1099`** — the whole
   pool, not a slice, so no selection is possible. §8's minimum is 20; using the pool as registered
   removes a choice rather than making one.
2. **DT, MaxPressure and MAPPO@1000 are all measured live on those same draws**, paired. The gate
   threshold is `1.05 × ATT_MAPPO@1000_heldout`, computed there.
3. **Both thresholds are measured and committed BEFORE the first gradient step** — your safeguard,
   adopted verbatim. The commit that records them must precede the training commit in history.
4. **`110.73` goes in the Return Packet verbatim, labelled as the draws-1–200 secondary reading, and
   is NOT the verdict.** Report it beside the gate; it is the comparison against the committed
   ladder and it is worth having.
5. Report CIs as §9 requires, and the paired **Wilcoxon** over shared draws as the registered
   primary test (§8) beside the inequalities. The gate itself remains the two inequalities.

**Tier `mappo1000`: CONFIRMED**, and your reasoning is the right one — a failure on a weaker tier
would be uninterpretable between pipeline and data. State plainly in the packet that passing here
demonstrates **pipeline capability, not that sequence modelling beats imitation**; that question is
P4.4's and must not be implied. One caveat to carry: expert-only data has the narrowest state
coverage we measured (`mappo1000` was the *lowest* on grid4x4), so a pass here says nothing about
off-distribution behaviour.

**20,000 steps: CONFIRMED, with the response pre-declared now so it cannot be chosen later.** If the
training loss has not plateaued at 20k, you may raise **once** to 40k, **decided on the training
curve alone** — no evaluation number may enter that decision, exactly as amendment A3 handled the
MAPPO budget. A second raise requires coming back. The reported checkpoint is the declared step.

**CUDA: ACCEPTABLE, conditional.** *(⚠️ 2026-08-11: the proof is a TENSOR comparison; see `DEFERRED` 29.)* The double-train byte-identity proof must pass **on the actual
training device and configuration**; if it fails, either make it deterministic or fall back to
CPU-in-tmux — do not report a number from a path you could not reproduce. **Record device, torch and
CUDA versions in the artifact:** P8.0's provenance omitted exactly the state that determines
reduction order, in the file whose MAPPO rows were attributed to it (finding N8).

**Your three traps: all three resolutions confirmed.** On the third — the *target* keeps `-1` so the
`ignore_index` tripwire still fires; the *embedding input* gets its own pad index. That distinction
between input and target was missing from my §2 and you caught it. Paste the tripwire crash.

**One correction that saves you work: the all-True mask claim is already population-verified.** The
P3 review scanned **32000/32000 streams, 0 excluding the taken action** (`docs/reviews/P3.md`). Cite
it; do not re-measure it. Your instinct to state the sample was right — the sample here is the
population, and it is someone else's measurement, so cite it as theirs.
