# BRIEF_38 — P7.3b: the FULL-RETRAIN ANCHOR of the C3 curve (A18(a)), and the five seams P7.3a left unpinned

**Task id:** `P7.3b` · **Branch:** `task/p7.3b-anchor`, cut from `main` at the commit carrying this brief or later · **Worktree:** `/home/filip/rltraffic-p73b` (`git -C /home/filip/rltraffic worktree add /home/filip/rltraffic-p73b -b task/p7.3b-anchor main`; the coordinator never commits from it).
**Mode:** Claude Code, plan mode first, `docs/plans/p7.3b.md` before any code.
**Registered as:** `PROJECT_PLAN` §6 **P7.3b**, split by the author on 2026-09-18: **the anchor ALONE.** The few-shot curve is **P7.3c, deferred not cancelled**; grid4x4 is **P7.3d**.
**Design, all of it already registered — this brief chooses NOTHING scientific:** `PREREGISTRATION` §2 H3, §3.4 ρ, A15 (the ρ pair, the teleport-free regime), A16 (`align_info` the only door), A17(b) (the probe band), **A18(a) (the anchor, quoted in §1 below), A18(b) (why there is no online anchor), A18(c) (the held-out seed rule)**. `BRIEF_37`'s amendments apply where named; its rulings after Amendment J are in the Decisions Log (see `BRIEF_37`'s closing pointer).
**Compute:** ≈ 1.2–1.5 h total — 100 unobserved SUMO episodes (≈ 19 min), five DT trainings (≈ 17 min), 500 observed evaluation cells (≈ 15 min), 200 re-rolled ρ anchors (≈ 7 min), pilot and pre-flight (≈ 15 min). **Every figure is from P7.3a's measured rates and carries A11's label** (stage 2's 19.12 s/cell was measured on a machine that had refused a start at canary 6.42 s minutes earlier).
**Cost:** 4–5 implementer days, of which ≈ 0.5 is §2's pins.

> **Read order:** this brief → `PREREGISTRATION` §12 row **A18** whole, then A17 and A15 → `docs/reviews/P7.3a.md` (both merge reviews: what was found unpinned and what was recomputed) → `docs/returns/P7.3a-3.5b-3.6.md` §19 (the campaign as it ran, and §19.14 *what P7.3b will assume*) → `BRIEF_37` §3.5, §7 and Amendments A2, B1/B2, C2, E4, G1–G4, J1.

---

## 0. What the coordinator verified before writing this (2026-09-18, by running commands)

1. **`train_dt` is generic** (`offline/dt_gate.py:747`): it takes `stacked` tensors, `state_dim`, `n_actions`, `seed`, `declared_gradient_steps`, `context_length`, `batch_size`, `stats`, `scenario_id`, `target_rtg`, `rtg_scale`, `provenance`. **Nothing in it is CityFlow-specific**, so the anchor's work is the corpus loader, the naive in-domain rule, and a declaration artifact — not a trainer.
2. **`assert_probe_draws_disjoint` already exists** (`offline/rtg_calibration.py:284`); A18(a)'s disjointness requirement is a call, not a build.
3. **The 201–300 half of the corpus EXISTS and is gated**: `datasets_sumo_v11/hz1x1_sumo_maxpressure/`, 100 episodes, `ix0_state` (361, 25), `format_version 1.1`, every per-draw record `engine_seed_requested 1000` / `"-1"` / `["cf_parity"]`, and **A17(f) passed 100/100 on it** (`output/p7_3a/logs/a17f.log`, re-run independently by merge reviewer A).
4. **The 301–400 half DOES NOT EXIST**: `scenarios/draws/cityflow1x1/` holds draws 0–5, 201–300 and 1000–1099 only. Both the CityFlow parents and their parity configurations must be rendered by P7.2a's tool.
5. **The corpus directory carries NO `SHA256SUMS`**, which `BRIEF_37` §3.1 required; `trajectory_logger` never writes one and no corpus in this repository has one. The coordinator recorded the 101 digests at `output/p7_3a_runs/coordinator_stage1_verify/` (`e81cf687…`) before anything else touched the tree.
6. **J1(c) bites this task on its first commit:** `code_changed_since` refuses reuse and regeneration when anything outside `docs/` differs between a chunk's commit and HEAD. P7.3a's 4,700 chunks were rolled at `8bf7eae`; **the moment this task's first code commit lands they stop being reusable, and `docs/data/p7_3a_zero_shot.json` regenerates only at its recording commit.** That is by design and it is why §3.4's anchor cells are re-rolled rather than reused.
7. **CI's ceiling is 189** (`main` green at `696ab10`) and `re_measure_required_at` names **this task's merge**.

---

## 1. A18(a), quoted, because this brief implements it and changes nothing in it

> *a DT trained from scratch, on SUMO, on 200 MaxPressure probe episodes — the few-shot band 201–300 (A17(b)) plus draws 301–400, one episode per draw on a fresh env with `reset(seed=1000)` under the parity contract and the teleport-free regime, rendered by P7.2a's tool and logged through the trajectory logger in the canonical frame (A16's `align_info` on every step, C6 v1.1 episodes) — training-pool draws disjoint from both subjects' training draws (1–200) and from the held-out pool (1000–1099), asserted by `assert_probe_draws_disjoint` before the first episode. **Training recipe: P4's, unchanged** (`train_dt`: context 20, 3 layers, 1 head, `d_model` 128, dropout 0.1, batch 64, AdamW lr 0.0001 weight decay 0.0001, warm-up 1000 steps, grad clip 0.25), **40,000 gradient steps**, five seeds 101/202/303/404/505, `target_rtg` = the maximum episode return of the training set and `rtg_scale` = the largest absolute RTG of the training set (the naive in-domain rule), state normalisation fitted on the anchor's own 200 episodes. Evaluated on the held-out pool under (c) with the naive prompt, scored in ρ_sumo under A15's pair with the admitted pair beside it. **It is the curve's k = 200 endpoint and the paper says exactly what it is: what target-domain probe data alone buys the same architecture — not an upper bound on achievable SUMO performance, and not a target-domain online policy.***

**What this task therefore answers, and what it does not.** It answers whether P7.3a's ρ ≈ +1.89 needs CityFlow pre-training at all: if a SUMO-trained DT on 200 MaxPressure episodes lands near ρ ≈ 1, the zero-shot advantage is attributable to the source data rather than to anything target-domain data supplies. **It does NOT settle whether MaxPressure is a weak baseline** — A18(b) registers why there is no online SUMO anchor and requires its absence to be named as a limitation in the paper's C3 section, not in a footnote. Nothing in this task may be written as if the anchor were an upper bound.

---

## 2. THE FIVE PINS COME FIRST, BEFORE ANY ANCHOR CODE

Merge reviewer A found five behaviours that are correct today and that no test would notice if they changed (`docs/reviews/P7.3a.md`). **P7.3b runs a different checkpoint per cell, so the first three bite immediately.** Each gets a test that fails when the behaviour changes, and each named mutation is executed and pasted:

| # | seam | the mutation that must die |
|---|---|---|
| 1 | the engine seed actually handed to `horizon_rollout` (`transfer_curve.py:1104`) — the chunk records the constant, not the call | `ENGINE_SEED → ENGINE_SEED + 1` |
| 2 | `validate_cell_payload`'s `rtg_first == target_rtg` refusal (`:925`) — the only guard that the prompt took effect | the refusal disabled |
| 3 | `_paired_block`'s ATT key (`:1746`) — the registered H3 test row IS this comparison | the key forced to `att_env` |
| 4 | the in-support diagnostic's bounds (`:1128-1130`) | `rtg_min`/`rtg_max` swapped |
| 5 | `run_stage`'s resume-by-content (`:2067`) | replaced by `path.exists()` — the `p5_3b.sh` defect its own docstring cites |

**Also in this first commit:** Finding 4's fix — the all-excluded env-ATT path raises `TypeError` in `_h3_block` and `_contrast_block` instead of carrying on; fix it **and** correct the comment at `transfer_curve.py:1526`, which still asserts otherwise (the packet's §17.1 already states the measured behaviour).

---

## 3. Per-file requirements

### 3.1 `offline/materialise_draws.py` / the draws tree — the 301–400 band (no new module)
Render the CityFlow parents for draws **301–400** and their parity configurations with the existing CLI, into the main tree's `scenarios/draws/cityflow1x1/`. **The tool is P7.2a's and is not modified**; if it cannot render the band without a change, STOP and report it as an open question rather than editing it. Record the 100 `provenance.json` digests in the packet. `assert_probe_draws_disjoint` is called on {201…300} ∪ {301…400} against the subjects' training draws (1–200) and the held-out pool (1000–1099) **before the first episode**.

### 3.2 `offline/collect.py` — the second corpus half, through the same entry
One MaxPressure episode per draw 301–400, `--base-seed 1000 --episodes 1`, the same argv P7.3a's driver used, into `datasets_sumo_v11/hz1x1_sumo_maxpressure_301_400/` (a SEPARATE directory: the logger refuses a populated out-dir, and the 201–300 corpus is A17(f)-gated evidence that must not be touched). **No A17(f) equivalent exists for 301–400** — there is no probe artifact for that band — so the integrity check is a **bit-for-bit re-collection of five declared draws** (301, 325, 350, 375, 400) into a scratch directory, compared under `==` on `ix0_local_reward` sums and `engine_seed_drawn`; SUMO is deterministic at a fixed seed and P7.1 proved it in float64. A mismatch stops the task.
**Write a `SHA256SUMS` for BOTH corpus halves** (§0.5: the brief required one and no corpus has one), and record both digest files in the packet.

### 3.3 NEW `offline/anchor_training.py` — the 200-episode corpus, the naive rule, five seeds
`load_anchor_corpus(dirs) → stacked, stats, facts`: the two v1.1 directories read through the existing dataset loader, 200 episodes asserted, `state_dim` asserted **25** (A16's canonical width — a 32 here means the corpus is not canonical), normalisation fitted on these 200 episodes ONLY. `naive_in_domain_prompt(stacked) → (target_rtg, rtg_scale)`: `target_rtg` = the maximum episode return of the training set, `rtg_scale` = the largest absolute RTG of the training set, **both computed here and recomputed by a different route in the test** (`np.cumsum` on the stored rewards). Then `train_dt(...)` per A18(a)'s recipe verbatim, 40,000 steps, seeds 101/202/303/404/505, checkpoints to `output/p7_3b_anchor/checkpoints/anchor_dt_seed{seed}.pt`.
**Artifact `docs/data/p7_3b_anchor_training.json`:** per seed the checkpoint's sha256, the final loss, the gradient steps, the seconds; the corpus's two directory digests and its 200 episode ids; `target_rtg`, `rtg_scale` and the normalisation stats; `declared_gradient_steps`; provenance. **This artifact is the committed digest record the evaluation pins against** — the anchor's checkpoints have no `SHA256SUMS_p4_*` and G1 requires a committed record.

### 3.4 `offline/transfer_curve.py` — the anchor as a subject whose prompt is its own
A new cell kind or subject entry `anchor_k200` whose checkpoint comes from §3.3's artifact (sha-pinned, G1's shape) and whose prompt is the **naive in-domain** target from the same artifact — **not** from P7.2b's calibration artifact, which registers CityFlow-trained subjects only. Everything else is unchanged and must be: the aligned observed env (A2 — it consumes the canonical frame because it was trained on it), `agent.act(info, explore=False, update_memory=True)` (BRIEF_36 E4), `horizon_rollout(..., 1, ENGINE_SEED)`, the recorder, every refusal.
**The cells:** 5 seeds × 100 held-out draws = 500, plus the `fixedtime` and `maxpressure` anchors on the same draws **re-rolled** (200) because §0.6 makes P7.3a's chunks non-reusable. `report` gains the anchor's rows and its ρ under both definitions; H3's clauses are **unchanged** (they are about `b_mean_k100`), and the anchor is reported beside them as the curve's k = 200 endpoint with A18(a)'s sentence about what it is not, **verbatim, in the artifact**.
**Artifact `docs/data/p7_3b_anchor.json`**, same block layout as P7.3a's, with `denominator_diagnostic`, the env-ATT caveat, `what_this_does_not_say` naming A18(b)'s absent online anchor.

### 3.5 `offline/campaigns/p7_3b_anchor.sh` — the driver, P7.3a's shape
Token, lock, group-leader check, the SigIgn refusal, the dirty-tree refusal, the canary both halves before the token, `record-canary` after, trap **INT TERM HUP** before the token, `tee -a`, artifacts to `$WORK/artifacts` outside the worktree, manifest last and atomic. Stages: **collect 301–400 → the five-draw re-collection check → train (five seeds) → cells → report**. The header's schedule is this task's own pilot, with A11's label and the date.

---

## 4. Tests — written first, red for their own reasons, every named mutation executed and pasted
§2's five pins and Finding 4's fix, each with its mutation · the disjointness assertion refusing an overlapping band · the corpus loader refusing a 32-wide corpus and refusing ≠ 200 episodes · the naive rule recomputed by an independent route · the training artifact's digests matching the files on disk · the anchor's prompt coming from §3.3's artifact and **never** from `p7_2b_calibration.json` (mutation: point it there → dies) · the anchor cell's env being aligned and observed · `report` refusing an anchor cell whose checkpoint sha is not the declared one · the driver's text (T9's shape, all three trapped signals) · **one real SUMO episode at most beyond P7.3a's** (§4's cap). Gated tests name their artifacts (G2). `check_test_hygiene.sh` and the repo-wide `check_english.sh` on everything touched.

## 5. Gates, in order
1. **Plan gate** (`docs/plans/p7.3b.md`): the corpus loader's route from two directories, the naive rule's two quantities and how the test recomputes them, the disjointness call, the anchor's pin source, and **one measured DT-training rate on this machine** (one seed, or a 1,000-step slice, with its canary and date). The coordinator rules.
2. **§2's pins merged first** — they are a separate commit and the coordinator re-runs every mutation before §3 begins.
3. **Pre-flight** (a reviewer, findings file): the driver's refusals executed, the pool's destruction and resume paths, both canary halves.
4. **The token**, written by the author; the campaign attended in tmux through `tee -a`; **the coordinator reads the driver CAPTURE, not only the logs** (2026-09-18's rule: a refused start leaves no entry in `canary.log`).
5. **Packet** `docs/returns/P7.3b.md`; **two sequential merge reviewers** (code + mutations; the artifact recomputed from the raw cells); ruling; merge with §6's box ticked; both refs pushed; the CI ceiling re-measured by the registered route (`re_measure_required_at` names this merge).

## 6. Definition of Done
- [ ] §2's pins and Finding 4's fix, merged first, every mutation executed and pasted.
- [ ] Draws 301–400 rendered; disjointness asserted; both corpus halves carrying `SHA256SUMS`; the five-draw re-collection check bit-for-bit.
- [ ] Five checkpoints trained to A18(a)'s recipe verbatim; `docs/data/p7_3b_anchor_training.json` committed with digests that match the files.
- [ ] 500 anchor cells + 200 re-rolled ρ anchors, every cell clean and at one commit; `docs/data/p7_3b_anchor.json` committed and regenerating byte-identically at its recording commit.
- [ ] The packet reports ρ under both definitions with CIs, the anchor beside H3's unchanged clauses, A18(a)'s "what it is not" sentence and A18(b)'s absent online anchor, both clocks per stage with each stage's canary, the AI-assistance record, and *what P7.3d will assume*.
- [ ] **Nothing interpreted beyond §10's outcome rows.** The anchor is a measurement, not a verdict on P7.3a's number.

## 7. Scope fence — what NOT to build
**No few-shot fine-tuning, no continuation entry point, no k ∈ {5, 20, 100} curve** — that is P7.3c, deferred by the author. **No grid4x4** — P7.3d. **No online SUMO anchor** — A18(b), deferred to P11 and named as a limitation. **No change to P7.3a's committed artifacts**, whose regenerability ends with this task's first code commit (§0.6) — that is expected and is recorded in the packet, not worked around.

## 8. Return Packet
`docs/returns/TEMPLATE.md`, plus: the plan-gate training rate with its date; the disjointness assertion's output; the five-draw re-collection comparison; both corpus digest files; the training artifact's per-seed digests; ρ per definition with CIs and the paired-CI helper named; the anchor's position beside H3's clauses, reported not interpreted; the driver capture of every run including any refused start; the amendments and rulings written against, by letter and by Decisions-Log date; the AI-assistance record's four lines; *what P7.3d will assume*.
