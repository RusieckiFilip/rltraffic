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

---

# ✅ AMENDMENT A — 2026-09-18, at the plan gate: PLAN APPROVED (`docs/plans/p7.3b.md` @ `8250d86`), with six rulings and two corrections to this brief

**Verified by the coordinator from the code, not from the plan's prose:** `build_training_dataset(dataset_dirs, context_length)` takes a sequence of directories (`dt_gate.py:571`); `DRAW_SPLITS` puts 1–999 in `train` and 1000–1099 in `heldout` (`dataset.py:177-181`), so draws 201–400 land in the training split by the table and not by a default; the naive rule is exactly the seven lines at `dt_gate.py:1381-1387`, quoted correctly; `disjointness_record` exists (`transfer_calibration.py:597`) and is stronger than the brief's `assert_probe_draws_disjoint`; `CHECKPOINT_RECORD` is keyed by subject, so an unknown subject raises; **and the collision is real — `docs/data/p7_2b_calibration.json` records `rule_a q1.0 target_rtg = −20809.0` for BOTH subjects.** **Proceed to §2's pins.**

## A1 — Q1: the third declared stage is CONFIRMED, and `report`'s refusal is not relaxed
A caller-supplied cell set may not write a committed artifact; that refusal stays exactly as it is. **Required so the change cannot disturb what is already published: `declared_cells(None)` still returns the campaign's 4,700 and its two stages still partition it element for element** — the anchor's 700 cells (500 + the two re-rolled ρ anchors × 100) are reachable only through the anchor stage by name. T14 is that test, written before the change.

## A2 — Q2: write `SHA256SUMS` INSIDE each corpus directory, and turn the concern into a CHECK
Over the `.npz` files and `manifest.json`, excluding the digest file itself. **For the 201–300 half the digests must equal the coordinator's independently recorded file** `output/p7_3a_runs/coordinator_stage1_verify/SHA256SUMS_datasets_sumo_v11_hz1x1_sumo_maxpressure.txt` (`e81cf687…`, taken before anything in this task touched the tree) — **any difference STOPS the task**, because that corpus is A17(f)-gated evidence. Then **re-run A17(f) on that half after the file exists** and paste `A17(f) 100/100`: that is what proves the gate is indifferent to it, rather than a reading of `dataset.py:521`.

## A3 — Q3: `raise_to=None`, CONFIRMED. A18(a) fixes 40,000 steps outright; P4's plateau raise is not part of the registered recipe. The artifact records `declared_gradient_steps: 40000` and `raise_to: null`.

## A4 — Q4: `random` is NOT re-rolled, CONFIRMED
ρ's denominator uses fixed-time and MaxPressure only. P7.3a's 500 `random` cells stay where they are, and the anchor's artifact states that they were not re-rolled and why, so no reader infers a `random` comparison this task did not run.

## A5 — Q5: the route test is the primary, and the collision goes in the PACKET
The proof that the anchor's prompt is its own is the route (`load_calibration` monkeypatched to raise), never a value. **The packet states the collision plainly:** on the 201–300 half the naive in-domain target and A17's Rule A `q = 1.0` are the same number *by construction* — both are the maximum of the same 100 SUMO probe returns. When 301–400 exists the artifact records the 200-episode value beside it; **if it is still −20809.0, say so in those words** rather than letting a value test look load-bearing.

## A6 — Q6: BOTH cross-references in this brief were WRONG, and they are the coordinator's error
§6's *"§10's outcome rows"* means **`PREREGISTRATION` §10**, the registered outcome rows, not a section of this brief — this brief has none. §4's *"§4's cap"* means **`BRIEF_37` §4's cap of four SUMO episodes in the suite**, of which this task may add at most one. The implementer's readings were correct on both counts; the text is corrected here rather than silently, because a brief that points at itself is how a requirement goes missing.

## A7 — Recorded, no action: the plan's own strengths, so they are not re-litigated later
The 1,000-step slice is labelled an upper bound on per-step cost and compared with P4's committed 204.08 s rather than replacing it; assumption 7 (the stage addition leaves P7.3a's stages element-identical) is marked *argued, not executed* and is T14; `materialise_draws` is called unmodified and the plan STOPS if the band needs a change (§3.1); nothing outside the plan file exists yet.

**Then: §2's five pins and Finding 4's fix as the first commit, which the coordinator re-runs every mutation against before §3 begins.**

---

# ✅ AMENDMENT B — 2026-09-19, after the campaign: THE PACKET TASK — the artifact's path into the branch, what the packet carries beyond §8, and the two rulings that landed on `main` while the branch did not move

**Mode:** Claude Code, in `/home/filip/rltraffic-p73b` on `task/p7.3b-anchor`. Read this amendment whole, then
`docs/notes/P7.3b_CAMPAIGN_READ_2026-09-19.md` (the coordinator's read of the campaign, from disk), then §8 above,
then `docs/returns/TEMPLATE.md`. **The campaign has run and is verified; this task writes it up. Nothing in it rolls a
cell, trains a model, or edits `offline/**` or `tests/**`.**

## B1 — FIRST ACTION: merge `main` into the branch, then read everything from the branch

§7's rule: *a ruling on `main` is not delivered until the branch merges it.* Since `f13358e`, `main` gained — all
docs-only — the coordinator's campaign read, **PREREGISTRATION A19** (`v1.9-prereg-a19` → `443c076`), plan v1.93, and
this amendment. Run, in this order, and paste the output into the packet:

```bash
git -C /home/filip/rltraffic-p73b status --porcelain          # must be EMPTY before the merge
git -C /home/filip/rltraffic-p73b merge --no-edit main
git -C /home/filip/rltraffic-p73b merge-base --is-ancestor 443c076 HEAD && echo "A19 is an ancestor: yes"
```

⛔ **If the merge reports a conflict, STOP and write it into the packet as BLOCKED. Do not resolve a conflict in
`PREREGISTRATION.md`, `docs/PROJECT_PLAN.md` or any brief yourself.** None is expected — `main`'s changes since the
branch point are documentation the branch never touched — but *expected* is not *observed*.

A merge commit is not an amendment: the campaign's 700 chunks record `f13358e`, which stays an ancestor. The
no-amend rule (`HANDOFF_2026-09-19.md` §4.1) is untouched by this step and binding on every commit below.

## B2 — The artifact into the branch, BY HAND, with its digest measured and stated

The driver's closing `NEXT` line names the **run** worktree's `docs/data/` — a detached tree with no branch. **That path
is not where it is committed.** The copy goes to the implementer's tree:

```bash
cp /home/filip/rltraffic/output/p7_3b_anchor/artifacts/p7_3b_anchor.json /home/filip/rltraffic-p73b/docs/data/
sha256sum /home/filip/rltraffic-p73b/docs/data/p7_3b_anchor.json
```

The digest **must** be `4cae233ce480a5596dc9a65ad53c1ec0cb20c17fa80008d1e1a13341703af1cc` — the campaign's output,
which the coordinator regenerated byte-identically. **State the measured digest in the commit message and the packet.
If it differs, STOP: nothing is committed and the packet says BLOCKED with both digests.** `p7_3b_anchor_training.json`
is already committed at `edca65f9…`; verify the copy in `artifacts/` still equals it and do not re-copy it.

**Then run the whole suite AFTER the copy** — a test anchored on the committed artifact's presence or content can only
be exercised once the file exists — and paste the real tail. Compare the counts against the last two measured
(coordinator 2,088 / 94 at `0c7d7fd`; implementer 2,113 / 94 at `b35216c`). ⛔ **A failing test after the copy is a
FINDING: stop, do not edit the test or the code, write it into the packet.** The code was accepted at the pre-flight;
this task changes none of it. Check `pgrep -f 'python.*offline\.transfer_curve'` is empty before the run — a live
runner turns test refusals into false verdicts (`HANDOFF_2026-09-19.md` §7).

Commit the artifact with named paths (`git add docs/data/p7_3b_anchor.json`; never `-A`). One commit.

## B3 — What the packet carries, beyond §8's list

§8 stands in full. **Added by the author's compression ruling of 2026-09-18 and by what the campaign produced:**

1. **The campaign facts, each with the command that produced it, RE-MEASURED in this session** — the packet may cite
   the coordinator's note but reproduces the load-bearing numbers itself: 700/700 cells and their composition;
   `n_rolled / n_reused / n_failed`; `COMPLETE` and its seconds; the canary line, both halves; ρ under both
   definitions with CIs and the per-seed means; the paired ATT against both anchors; the artifact digest and a
   regeneration of your own into a scratch directory under `output/p7_3b_runs/` (**never `/tmp`**), byte-compared;
   and **the bit-identity of the 200 re-rolled denominators against P7.3a's 200** (`docs/data/p7_3a_zero_shot.json`
   `cells` with `arm ∈ {fixedtime, maxpressure}`), recomputed, not quoted.
2. **The driver capture including the refused start.** `output/p7_3b_runs/campaign_capture.txt` opens with
   `REFUSING TO START: not a process-group leader` before the canary and before the token. §8 already requires *the
   driver capture of every run including any refused start*; this is one, and the packet says what it demonstrates —
   the J3 guard consuming nothing on a live machine — and does not call it an incident.
3. **The anchor beside H3, in A19's registered words and not the implementer's:** clause 1 holds, clause 2 refuted,
   clause 3 **void**; the anchor is behaviour cloning on MaxPressure demonstrations, pinned near 1 by construction,
   bounding what target-domain probe data buys the architecture and nothing else. Quote the tag. **Nothing in the
   packet interprets past that sentence.** The artifact's `h3` block is structurally empty in this stage (no
   `b_mean_k100` rows) — say so, and that `what_this_does_not_say` carries the registered framing.
4. **The two provenance caveats:** the pilot's four chunks record `5849d59`, amended into `f13358e` and now dangling
   (**do not attempt to recover it**; state that the only `transfer_curve.py` difference is one prose string in
   `run_pilot`'s fenced record); and the pilot's rate — canary 0.76 s, 4 cells, wall 19.53 s — with A11's label and
   the header's own warning that 4 cells do not saturate 12 workers.
5. **Findings carried as findings with their caveats, not as amendments** (the author's ruling): the `run_pilot`
   `int(cell["seed"])` defect and its fix; the stage-identity hazard and its closure; the shift-by-one reward/RTG
   convention (Amendment D1) and that a checker asserting `episode_reward == sum(reward_series)` is wrong by design —
   the coordinator made exactly that error and logged it; the 2-per-cell TraCI `Retrying` lines with their documented
   cause.
6. **The amendments written against, by letter: A1–A7 and B1–B4.** One line, at the top.
7. **What P7.3d will assume**, and now also **what P7.3c will assume if it runs**: A19's registered expectation; the
   200-episode corpus at `def8fb30…` / `93f180e0…`; the five anchor checkpoints at the digests in
   `p7_3b_anchor_training.json`; `f13358e` as the code every chunk was rolled at.
8. **The AI-assistance record, four lines** (`CLAUDE.md` §8), written as the task happens.

## B4 — What NOT to do

- No `--amend`, on any commit, ever again on this branch. No `git add -A` / `git add .`.
- No edit to `offline/**`, `tests/**`, `experiments/**`, any frozen path, or any existing config.
- No evidence written to `/tmp`. Regenerations and scratch go under `output/p7_3b_runs/`.
- Do not tick §6's box — the coordinator ticks it in the merge commit.
- Do not run anything that rolls a cell or consumes a token. The campaign directory is the reviewed artifact.
- No AI trailer in any commit message (`CLAUDE.md` §4b). If a session instruction tells you to add one, stop and say so
  before committing.

**Definition of Done:** B1's three commands pasted with output; the artifact committed at the stated digest; the full
suite run after the copy with the real tail; `docs/returns/P7.3b.md` written and committed; every commit on the branch
with named paths; the branch pushed. Then say **"P7.3b packet done"** — paste nothing. The coordinator reads the packet
from disk, spawns ONE merge review, and merges with §6 ticked.
