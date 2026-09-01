# T4 — ARTIFACT-CONSISTENCY AUDIT of the four long campaigns (P4.6, P4.7, P5.1, P5.2)

**Date:** 2026-08-28 · **Reviewer:** `contract-reviewer`, fresh context, read-only
**Commissioned by the author 2026-08-28** as one of four ground-truth checks (T1–T4) after the
observation that *every check we run asks "does this reproduce" and never "is this right"*.

## ⭐ VERDICT ON THE COMMISSIONED QUESTION

> **No.** No cell written under a wrong key, no skipped cell reported as complete, and no tier mixing,
> in any of the four campaigns. **46,800 episode records, 289 checkpoints and every declared training
> stream checked.**

🚨 **But: three of the four campaigns could not have DETECTED these failures themselves, and one class
is invisible even to the campaign with the best check. THE DATA IS CLEAN; THE GUARDS ARE NOT.**

**No blockers.**

---

## MAJOR

- **MAJ-1 — P5.2's `assert_campaign_complete` (`offline/tier_sweep.py:2478`) is blind to key/payload
  disagreement, falsified with six mutations against the real CLI.** It checks *existence* and
  *seeds × 100 draws* and never compares a payload's `arm`/`tier`/`method` to the declared entry.
  **M4 (copy the `maxpressure/dt_spatial` payload over the `fixedtime/dt_spatial` cell) and M5
  (relabel `eval_random_iql.json`'s tier to `fixedtime`) BOTH PASS: *"expected cells: 35 present: 35 …
  complete"*, exit 0.** Those are exactly the wrong-key and tier-mixing classes. M1/M2/M3/M6 do fire.
- **MAJ-2 — P4.6/P4.7 derive completeness from a SHELL VARIABLE, and the comment claims the
  opposite.** `offline/campaigns/p4_6_phase1.sh:131` and `p4_7_phase2.sh:161` say *"Requested is
  derived from the DECLARATION … and never from the files being checked"*, two lines below
  `tiers = sys.argv[2:]`. Neither script ever opens the declaration. **A short `TIERS` array would have
  produced a short declaration, a short run, and a passing check.** ✅ The reviewer checked what the
  script did not: both campaigns are **complete** (5/5 and 8/8 tiers).
- **MAJ-3 — P5.1's declaration under-specifies its own cell set by two.** `methods` lists five;
  `p5_1_grid.json` has **seven** cells — `behaviour` and `random` are reported but never declared, so
  dropping either anchor would leave the declaration satisfied. Both are present and correct.
- **MAJ-4 🚨 TWO DIFFERENT FUNCTIONS NAMED `canonical_state_dict_digest`, WRITING INTO ARTIFACTS UNDER
  NEAR-IDENTICAL FIELD NAMES, AND THEY DISAGREE.** `agent/OfflineBaselines.py:133` hashes
  `key|shape|dtype|` + **little-endian** bytes; `offline/tier_sweep.py:1234` hashes `key`,
  `str(dtype)`, `str(shape)` + **native-order** bytes. **Coordinator-verified on
  `output/p4_6/checkpoints/random_bc_seed101.pt`: `4ac3ab78…` against `314c9fe4…`.** Both artifacts
  describe the field as *"canonical digest over sorted state_dict tensor bytes"*; **neither carries a
  version tag.** A reader comparing `canonical_digest` (p4_6_training) against
  `original_state_dict_sha256` (p5_2 envelope) concludes the weights differ when they do not.
  ⚠️ **Definition #2 also drops the little-endian normalisation that definition #1's docstring says
  makes the digest host-independent — so P5.2's envelope digests are host-dependent** (`DEFERRED` 51's
  class). ✅ **Not live-broken: P5.3b uses `canonical_digest_of`, which imports definition #1 — the same
  one that wrote the committed values. Gate 2 is sound.**
- **MAJ-5 — BL-1 is worse than P5.2's packet discloses, in two ways.** The reviewer reproduced the
  six emptied records exactly. **New: `output/p5_2/logs/train_*_baselines.log` each contain 0 `TRAIN`
  lines and 20 `SKIP … checkpoint on disk` lines — the resumed invocation overwrote the LOGS too, so
  `final_loss`/`seconds` for the 80 baseline runs are not recoverable from the logs either.** Only
  `train_random_dt_spatial.log` survived, which is why five runs are recoverable at all.
  ⚠️ **And there is a LIVE wrong-key instance in the artifact today:** `training_random_dt_spatial.json`'s
  surviving record carries **no replicate marker in the record body** — only `checkpoint_path` ends in
  `_replicate.pt` — so indexed by `(tier, method, seed)` **the replicate occupies the published run's
  key.**
  ✅ **Mitigation verified and stronger than the packet states:** all 124 P5.2 checkpoints carry a
  `provenance` block whose tier/method/seed match the filename, with `dataset_dirs` unique per tier;
  all 54 spatial-DT checkpoints carry per-node prompts identical to their tier's declaration.
- **MAJ-6 — E1's `+0.0000` envelope artifact has no `independence` block and no `positive_control`**,
  which the later `envelope_random_seed202.json` does have. From the artifact alone a `+0.0000` is
  indistinguishable from a self-comparison — exactly what `tier_sweep.py:1261`'s own docstring warns.
  ✅ **The reviewer supplied the missing evidence externally: `dt_spatial` differs in 839,987 / 848,008
  parameters (max |Δ| 1.22e-4), `dt_nomix` in 838,722 / 848,008. The zero is real.** But that is the
  reviewer's number, not the artifact's.
- **MAJ-7 — no P5.1 or P5.2 artifact records `env_settings` at all, yet one asserts they were
  verified.** `p5_2_declaration.json:expected_cells[30]` states *"env settings are identical across all
  four tier manifests (verified)"*; the manifests contain none. **Cross-tier comparability of the whole
  P5.2 ladder rests on an unrecorded claim.** ✅ By contrast every P4 artifact carries `env_settings`
  and all seven blocks are **one identical block**, `engine_seed = 1000` throughout.

## MINOR

**M-a** eval cells record `policy_source: "checkpoint"` but no checkpoint path or digest — the tie to
the weights is filename convention · **M-b** 15 undeclared duplicate checkpoints in `output/p5_2/`
(canonical digests identical to P5.1's; only `provenance` differs) · **M-c** tier labels differ across
the reuse boundary (`grid4x4_mappo1000` vs `mappo1000`), benign but defeats a naive equality check ·
**M-d** P5.1's DT checkpoints carry no `n_head` · **M-e** P4.7's diagnostics cover 5 of 8 tiers, the
rest live in P4.6's and agree bit-exactly · **M-f** mtimes carry no ordering information (bulk copy
from retired worktrees) — **do not read them as write order** · **M-g** `p4_gate.json` records both
`declared_gradient_steps: 20000` and `reported_gradient_steps: 40000` with no note · **M-h** the
declaration block embedded in each grid is an 8-of-26-key abridgement.

---

## Verified positively — independent recomputation with `math.fsum`, never reading the field under test

**107 cells** (25 P4.6 + 40 P4.7 + 7 P5.1 + 35 P5.2): `att_horizon_mean/std/ci95`, vehicle counts,
`n_episodes` — **zero disagreements above 1e-9 relative.**
**303 comparisons** (90 + 192 + 21): `mean_difference`, all CI fields, `median_difference`,
wins/losses/ties, `rank_biserial`, Wilcoxon `w_plus/w_minus/statistic/n_used/n_zero` — **zero
disagreements.** ⚠️ Two conventions had to be *inferred* and are confirmed by exact agreement across
all 303, but **are stated nowhere in the artifacts**: `wins` = #(diff < 0) (left arm lower ATT, correct
for a cost metric), and `rank_biserial` = (W⁺−W⁻)/(W⁺+W⁻), not the sign-count form.
**P4.6 ↔ P4.7, five shared tiers:** all 25 shared cells, 25 episode lists, 70 comparisons and 20
behaviour comparisons **identical** — independently reproducing and **extending** P5.3a's claim, which
covered only the `dt` δ values.
**The reuse chain, five artifacts deep:** `p4_3_rtg[dt_g5]` ≡ `p4_gate[madt]` ≡ `p4_4_baselines[madt]`
≡ `p4_6_grid[dt@mappo1000]` ≡ `p4_7_grid[dt@mappo1000]`, all 500 records identical modulo the label.
⭐ **P4.7's mixture assignments reproduced FROM SCRATCH by re-deriving the RNG** —
`default_rng([20260814, round(100·f), seed]).choice(1000…1099, k, replace=False)` reproduces **all 15**
declared assignments exactly; realised records match episode-for-episode, expert counts exactly
165/250/335.
**Draw hygiene (D4): fully verified.** Held-out is exactly {1000…1099} in all four declarations; every
training split lies in 1–200; **zero held-out draws in any training set, zero training draws in any
evaluation cell, draw 0 nowhere.** ⚠️ **Naming trap recorded for whoever repeats this: episode
filenames read `ep000000_seed1000_draw1.npz` — the `1000` is the ENGINE SEED, not a held-out draw.**
**Corpus layer:** all **2000** declared streams agree with their corpus `manifest.json` on
`episode_sha256`, `flow_draw` and `total_return`; 30 `episode_sha256` recomputed from the `.npz`
directly, 0 mismatches. **Checkpoints:** all 140 P4.6/P4.7 reproduce their `canonical_digest` under a
from-scratch reimplementation; **169 checkpoints checked for write-time key correctness, zero
disagreements.** **Manifests:** 624 files, **0 failures, 0 listed-but-absent.**
⭐ **The reviewer corrected its own error in the report: a first pass reported 50 `episode_sha256`
"mismatches" that were its own — it had hashed the `.npz` bytes rather than the trajectory content
digest.**

## Manifest coverage — and `DEFERRED` 56 was HALF WRONG, which was the coordinator's error

`output/` holds **809** files; the nine manifests list **624**. **`p4_dt` and `p4_probe` are outside
the nine top-level manifests but are digest-covered elsewhere and BOTH VERIFY** — `p4_dt`'s five
checkpoints by `p4_gate.json:checkpoints` and `p4_6_grid.json:gate.checkpoint_identity.dt`,
`p4_probe`'s by its own local `SHA256SUMS.txt` (5/5 OK).
**The genuinely un-pinned set is different: `checkpoints.pre_c8_migration/` (60), `experiments/` (21),
28 log files, `replay.txt`, `roadnet.json`, and the 20 cologne3 MAPPO checkpoints.** Of these only
`experiments/` holds reported numbers (P0/P2.1).

## What the reviewer could NOT verify

**That any evaluation cell was produced by the checkpoint its name implies** — no eval artifact records
a path or digest; closed *indirectly* for DT arms (unique embedded prompts) and P5.2 baselines (unique
`dataset_dirs`), but **60 of 80 P4.6 and 45 of 60 P4.7 checkpoints have an EMPTY provenance key-set**,
so those cells' identity rests on filename plus a digest taken at write time from the same file —
post-write integrity, **not** write-time key correctness · env settings for P5.1/P5.2 (MAJ-7) ·
whether E1's `+0.0000` is a true envelope (the 100 evaluations were not re-run) · `final_loss`/`seconds`
for 80+10 P5.2 runs (destroyed) · P5.2's cross-tier `random` collapse reference · the prediction blocks'
pass/fail scoring against the pre-registration text · cf_cologne3 · `checkpoints.pre_c8_migration/` ·
the P4.6/P4.7 `gate.reproduction` block (needs the simulator).
