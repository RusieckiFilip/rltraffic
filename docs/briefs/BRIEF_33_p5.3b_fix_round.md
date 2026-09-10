# BRIEF_33 — P5.3b fix round: BL-2(b), BL-2(c), MJ-3

**Task id:** `P5.3b-fix` · **Branch:** `task/p5.3b-fix-round`, cut from `main` @ `0852ba6` · **Issued:** 2026-09-10
**Mode:** Claude Code, plan mode first, worktree `/home/filip/rltraffic-p53b`
**Review:** `docs/reviews/P5.3b.md` (recorded at `9ca670e`) — PASS-WITH-NOTES, 2 blocking, 10 major
**Scope, decided by the author on 2026-09-10 and NOT re-opened here:** BL-2(b), BL-2(c), MJ-3. BL-1 is
`DEFERRED` 72. Every other review item is `DEFERRED` 73.
**Compute:** no training. **One observer re-roll of 3,000 episodes** (§3.2), user-started in `tmux`,
pre-flight-reviewed first. Everything else is arithmetic over committed bytes.
**Contracts:** `docs/CONTRACTS.md` v1.1 — read it; C1 (env API), C5 (standards) and the C6 alignment
convention bind. The `offline/` output fence (`nortg_campaign.assert_writable`, default-deny) is a
contract of this task's own campaign and is inherited.

> ⚠️ **Numbering note.** `BRIEF_33` was announced in `PROJECT_PLAN` §10 as P8.4b's re-derivation
> brief and **cancelled** by `BRIEF_32` Amendment C (folded into P8.4b). The number was never used
> and is used here. §10's stale sentence is corrected in the commit that issues this brief.

> **Read order:** this brief → `docs/reviews/P5.3b.md` → `docs/returns/P5.3b.md` → `BRIEF_30`
> Amendments D and E → `PREREGISTRATION.md` A11(b), A13 → `offline/engine_att_reference.py`'s module
> docstring. Where this brief disagrees with `BRIEF_30`, this wins. Where it disagrees with the repo,
> **the repo wins and you flag it** (`CLAUDE.md` §2).

---

## 0. What the coordinator re-verified first-hand before writing this, and what it changed

Every number below was produced by a command run on 2026-09-10 against `main` @ `0852ba6` and the
gitignored trees in `/home/filip/rltraffic/output/`. Re-run them; do not inherit them.

**MJ-3 — confirmed exactly as the review states.** `offline/nortg_campaign.py:1543` sets
`"holds": bool(largest == "mix50" and smallest == "random")` with no reference to discriminability,
while `score_q2` (`:1661`) sets `None if not distinct else bool(contains)`. The committed artifact carries
`predictions.Q1.holds: true` beside `smallest_limb.is_evidence: false`. Two tests pin the `true`:
`tests/test_p5_3b_artifact.py:258` and `tests/test_nortg_campaign.py:1212`.

**BL-2(c) — confirmed, and sharpened in three ways the review did not state.** Identity over the eight
fields `att_engine, att_ours, entered, created, never_entered, horizon_vehicle_count, episode_reward,
completed_at_horizon`, per held-out draw:

| comparison | 101 | 202 | 303 | 404 | 505 |
|---|---|---|---|---|---|
| `dt_nortg@mix50` seed *s* vs `dt@random` seed *s* (the review's pairing) | 0/100 | 88/100 | 0/100 | 99/100 | 97/100 |
| `dt_nortg@mix50` seed *s* vs `dt@random` seed **101** | 0/100 | 88/100 | 0/100 | 99/100 | 97/100 |
| `dt_nortg@mix50` seed *s* vs **`dt_nortg@random` seed 101** (committed bytes only) | 0/100 | 88/100 | 0/100 | 99/100 | 97/100 |

1. **`dt@random` is seed-invariant**: seeds 202/303/404/505 are identical to seed 101 on 100/100 draws
   each, and `dt_nortg@random` equals `dt@random` on 100/100 per seed. So the pairing by seed is
   immaterial — there is ONE degenerate policy, and it is the same one for both arms of `random`.
2. **The finding is checkable from a clone.** The `dt@random` rows live only in gitignored
   `output/p8_4b_rederivation/`, but `dt_nortg@random` is in `docs/data/p5_3b_nortg.json` and the
   committed `discriminability.random` record already states it is identical to `dt@random` on 500/500.
   The third row above is the chain a clone can verify. **Report it through that chain.**
3. **Seed 202's 12 non-identical draws are a mixture, not noise.** Draws 1024, 1025, 1033 are
   UN-collapsed (`att_engine` 102.2 / 103.3 / 98.1, `never_entered` 0–1); draw 1055 is collapsed
   differently (795.7 against `dt@random`'s 754.9). So the collapse is **per draw**, i.e. state-dependent,
   not a fixed constant-action policy independent of input. "Same attractor" is therefore an
   **outcome-identity** statement until action identity is measured — which this task does (§3.3).

**BL-2(b) — the review's REQUIREMENT stands; the review's ATTRIBUTION is wrong, and the handoff repeated
it.** A13(b) registers `att_engine − att_ours = POPULATION (E − P) + CLOCK ORIGIN (P − W_running) +
CADENCE (W_running − C)` as a required reported quantity for every report of the two definitions'
difference. P5.3b reports the difference in three places and the decomposition nowhere — **that breach is
real and this task closes it.** But:

- **The three terms cannot be computed from the five A11(b) quantities.** `P` (admitted population, pool
  clock) and `W_running` (admitted population, admission clock, per-second cadence) exist only as
  per-second reconstructions built by `offline/engine_att_reference.py`'s `EngineObservationRecorder`.
  The campaign's episodes carry `att_ours`, `att_engine`, `entered`, `created`, `never_entered` — the
  endpoints, not the intermediates. **So the decomposition is a MEASUREMENT, not arithmetic.**
- **The review's sentence *"the engine metric's extra 215 ATT is entirely vehicles the ablated policy
  never admitted"* is unmeasured, and Gate 0's own committed artifact says it is almost certainly
  false.** `−409.1450 − (−194.4757) = −214.669 = (−2.112) − 212.557` is the identity
  `Δ_engine − Δ_ours ≡ Δ(engine − ours)` — a tautology, not a decomposition. The nearest measured
  analogue, `docs/data/p8_4b_g0_reference.json` on hz1x1 (`att_ours − att_engine` orientation):

  | episode | `never_entered` | population | clock_origin | cadence |
  |---|---|---|---|---|
  | `behaviour@random` draw 1000 | 615 | −1.282 | **−358.082** | +3.747 |
  | `behaviour@random` draw 1001 | 624 | −3.842 | **−375.635** | +3.717 |
  | `bc_top10@random` draw 1009 | 688 | −3.093 | **−397.123** | +3.741 |

  On a policy with the same never-entered profile as the collapsed arm (598 per episode), **≈99 % of the
  gap is CLOCK ORIGIN — the insertion-buffer wait of vehicles that DID enter, which `metrics/cityflow.py`
  starts the clock after and the engine counts** — and the population term is single digits. That is
  exactly the class A13 named twice (population conflated with cadence in A11(3); with clock origin in
  A12(3a)), now a third time, in the review, in the 2026-09-10 Decisions Log row and in
  `HANDOFF_2026-09-10.md` §2. **No paper sentence about what the 215 "is" until it is measured on the
  actual arms. This task measures it.**

---

## 1. Why this task exists

Three things, in the order they should be done:

1. **MJ-3** — complete a ruling already made. Q1 must not be reportable as a whole prediction when one
   of its limbs rests on a tier that cannot discriminate.
2. **BL-2(c)** — put the mechanism the review found into the artifact as an **exploratory** finding, by
   the committed-bytes chain, and add the one measurement that turns "same outcome" into "same policy":
   action-sequence identity.
3. **BL-2(b)** — discharge A13(b) by measuring the three-component decomposition on all 30 P5.3b cells
   under the Gate 0 observer, and embed it where the difference is reported.

---

## 2. MJ-3 — `predictions.Q1.holds`

### 2.1 The rule (this is the ruling; implement it, do not reinterpret it)

Each limb gains two fields: `registered_tier` (`"mix50"` for `largest`, `"random"` for `smallest`) and
`as_registered: bool` (`tier == registered_tier`). Then:

```
holds = False  if any limb has is_evidence == True and as_registered == False   (measured falsification)
holds = True   if every limb has is_evidence == True and as_registered == True
holds = None   otherwise  (some limb rests on a non-discriminating tier, and no evidence-bearing
                           limb is falsified)
```

Add `"holds_rule"` carrying that text verbatim, so the artifact explains its own `None`. Existing fields
(`largest`, `smallest`, the limb records, `limbs_are_scored_separately`) stay. `score_q2`'s convention
(`None` = cannot score) is the precedent; this makes Q1 parallel to it without hiding a real
falsification behind a `None`.

### 2.2 Authorised test edits — quote this paragraph in the packet

*Written authorisation, 2026-09-10: the implementer changes `tests/test_p5_3b_artifact.py:258` from
`assert q1["holds"] == (...)` to an assertion that `holds is None` under the committed discriminability
record (`random.distinct == False`) and that `holds_rule` is present; and changes
`tests/test_nortg_campaign.py:1212` from `assert scored["holds"] is True` to `is None`, keeping every
other assertion in that test.* These are spec changes ruled by the coordinator, not tests weakened to
pass. The existing True/False tests at `tests/test_nortg_campaign.py:662–731` use all-discriminating
records and **must still pass unchanged**.

### 2.3 New tests

- one limb non-discriminating, ordering as registered → `None`;
- one limb non-discriminating AND the evidence-bearing limb displaced → `False` (this is the case the
  simple `None if not all distinct` rule would get wrong — it must be a test);
- both limbs carry `registered_tier` and `as_registered`.
- **Mutation, executed and reported:** restore the old expression at `:1543` → the `None` test fails.

---

## 3. BL-2(b) and BL-2(c) — one instrument, one run, two blocks

### 3.1 What to build: `offline/nortg_decomposition.py` (new)

A thin harness over the **existing, reviewed** Gate 0 instrument. It must not re-implement anything:

- The env: `engine_att_reference.make_observer_env` (`:743`) via `gate_episode` (`:1251`), which
  already takes an arbitrary `choose_action_factory`, reads the five admission quantities between the
  rollout and `env.close()`, calls `env._eng.get_average_travel_time()`, and returns all four
  reconstructions plus `admission_latency`. `GateCell` (`:780`) accepts any `method` string and any
  `role` string; `_scenario_spec("hz1x1")` resolves. **`offline/engine_att_reference.py` is READ-ONLY
  for this task** — its artifact is registered and Gate 0's verdict rests on it. If you find you need
  to change it, stop and put it in the packet.
- The per-cell preamble: exactly `nortg_campaign.evaluate_cell`'s (`:574–632`) — `tier_spec`,
  `env_settings_for_tiers`, `_dt_factory(str(checkpoint), DECLARED_GRADIENT_STEPS, spec.target_rtg,
  device)` from `offline/method_tier_grid.py:2402`, `draw_config_path`, `created_from_flow`,
  `HELD_OUT_DRAWS`, `ENGINE_SEED`. Import them; do not copy them.
- The `dt` checkpoints: resolved as `_run_gate1` (`:1863–1875`) resolves them —
  `TIER_CHECKPOINT_TEMPLATE`, after `assert_reused_dt_identity` has passed in this process. The
  `dt_nortg` checkpoints: `output/p5_3b/checkpoints/<tier>_dt_nortg_seed<seed>.pt` in the MAIN tree,
  digest-checked against `output/SHA256SUMS_p5_3b.txt` before use.
- **Action recording, for §3.3:** wrap the factory so the returned `choose_action(env, info)` records a
  copy of every action it returns (`offline/horizon_metric.py:41`, `ChooseAction`) and returns the
  **same object** untouched. Per episode record `n_decisions`, `action_counts` (per intersection, per
  action index) and `action_sequence_sha256` — sha256 over the `int64` bytes of the `(T, n_ix)` array.
  ⚠️ Name it `action_sequence_sha256`, not `episode_sha256` (C6's is over actions **and** rewards —
  `DEFERRED` 57/70/71's class is one name meaning two things).

**Cells:** all 15 `dt_nortg` cells and all 15 `dt` cells × 100 held-out draws = **3,000 episodes**.
Not a subset: the contrast's decomposition needs both arms on the shared draws, and a subset would
have to argue representativeness across a bimodal arm (3 collapsed seeds, 2 not). Per-cell chunk files
(`decomp_<arm>_<tier>_seed<seed>.json`), idempotent and restartable exactly as `p5_3b.sh` is.

**Per-episode record:** everything `GateEpisode` carries (the four reconstructions, `att_engine_call`,
`att_ours`, the three `term_*`, `decomposition_residual`, `deviation_c1`, `deviation_c3c`, latency
fields, the five admission quantities, timings), plus `committed_att_engine`, `committed_att_ours`,
`reproduces_committed`, the action fields, `checkpoint` and `checkpoint_sha256`.

**Committed reference rows:** for `dt_nortg` — `docs/data/p5_3b_nortg.json:episodes`; for `dt` —
`output/p8_4b_rederivation/cell_hz1x1_dt_at_<tier>_seed<seed>_draw<draw>.json` (via
`nortg_campaign.rederived_dt_episodes`). Both `att_engine` and `att_ours` are compared under `==`.

### 3.2 🔒 THE LOAD-BEARING CHECK, and the filesystem-mutation barrier

`report` refuses to write the artifact — and writes **nothing** — unless, on **every one of the 3,000
episodes**: `att_engine_call == committed_att_engine`, `att_ours == committed_att_ours`,
`deviation_c1 == 0.0`, `deviation_c3c == 0.0`, `decomposition_residual == 0.0`. A single mismatch means
the observer perturbed the trajectory or the harness differs from the campaign's, and a decomposition of
a *different* trajectory is not the required quantity. Gate 0 established `c1 = 0.0` and `c3c = 0.0` on
46 episodes of other arms; **whether the observed DT trajectory equals the campaign's is a new fact and
this check is what establishes it.** All validation precedes every write (`CLAUDE.md` §5 barrier — this
bug has shipped twice).

**Cost, from measurement, so the author gets a number:** P5.3b's evaluation ran at **2.90 s/episode**
(mean of the 15 `timings_seconds.evaluate_seconds` cells ÷ 100). Gate 0's hz1x1 observer episodes
averaged **1.475 s** (23 episodes, max 4.42) against P8.4b's committed **1.29 s** rate
(`offline/att_rederivation.py:89`), i.e. observer overhead ≈ 0.2 s mean. Estimate **3.1–3.5 s/episode →
2.6–2.9 h serial, ≈35 min at 5 workers** (one process per seed, one torch thread each, as `p5_3b.sh`
does). ⚠️ The DT under the observer has not been timed; the first smoke (§5, G2) measures it and the
packet reports the measured rate before the run is scheduled.

### 3.3 BL-2(c) — the `mechanism` block (exploratory, labelled as such)

Under a top-level `mechanism` key in `docs/data/p5_3b_nortg.json`, with
`"registered": false, "exploratory": true, "found_by": "post-merge review, docs/reviews/P5.3b.md, 2026-09-10"`
as the first three fields (`PREREGISTRATION` §2 — confirmatory versus exploratory is fixed, and this is
the latter):

1. **Outcome identity, from committed bytes** (computed in `nortg_campaign report` from its own
   `episodes`): per `mix50` seed, the count of held-out draws on which `dt_nortg@mix50` equals
   `dt_nortg@random` seed 101 on the eight fields listed in §0; plus `dt_nortg@random`'s seed-invariance
   (each seed against seed 101, expect 100/100) and the pointer to `discriminability.random` as the link
   to `dt@random`. List the non-identical draw ids per seed.
2. **Action identity, from the decomposition run** (read from `docs/data/p5_3b_decomposition.json`):
   the same counts using `action_sequence_sha256` instead of outcomes, for `dt_nortg@mix50` vs
   `dt_nortg@random`, for `dt_nortg@random` across seeds, and for `dt@random` vs `dt_nortg@random`.
   Plus, for the collapsed cells, the `action_counts` summary — reported, **not interpreted**: if the
   policy holds one phase, say the histogram; do not write "constant-action" unless every one of its
   decisions on every identical draw is the same action, and say on how many draws that was checked.
3. **A reading string that stays inside the measurement.** Allowed shape: *"on N of 5 seeds, the
   ablated mix50 DT produces on X–Y of 100 held-out draws the same action sequence and the same episode
   outcome as the random-corpus DT, which is itself identical across all 5 seeds and both arms; on the
   other seeds it does not collapse (att_engine A, B against dt's C)."* Not allowed: any sentence about
   *why* it collapses.

### 3.4 BL-2(b) — the decomposition, where the difference is reported

`docs/data/p5_3b_decomposition.json`, format `p5.3b-decomposition/1.0`, with `episodes` (3,000 rows as
in §3.1), `summary.per_arm` (six arms, each in `_decomposition_summary`'s shape — import it; do not
re-implement — plus per-seed means), `summary.contrast` per tier (`dt − dt_nortg` for each term, paired
per draw exactly as `paired_stats` pairs: per-draw mean over seeds, then over the 100 draws; also the
plain 500-episode mean, with the difference between the two averaging orders reported as
`per_seed_differences` does), `provenance`, `runtime`, `what_this_does_not_say`.

**Consistency requirement, asserted in `report`:** for every tier,
`Δ_ours − Δ_engine` from `docs/data/p5_3b_nortg.json:comparisons.<tier>.by_definition` equals
`Δpopulation + Δclock_origin + Δcadence` within **1e-9** (summation order differs, so `==` is not the
right test here — say so in the docstring). Per episode the identity is exact and is checked exactly.

Then `nortg_campaign report` embeds, under every `comparisons.<tier>`, a
`definition_difference_decomposition` block: `source` (path + sha256 of the decomposition artifact),
`n_episodes_per_arm`, `per_arm.{dt,dt_nortg}.{population,clock_origin,cadence,total}` means,
`contrast.{population,clock_origin,cadence,total}`, `residual_max`, and the identity string. The report
**refuses** if the decomposition artifact is absent — A13(b) makes it required, so a report without it is
not a valid report.

`ARTIFACT_FORMAT_VERSION` → `"p5.3b-nortg/1.2"` (new keys = layout change, C6's rule), with a one-line
migration note in the module docstring: 1.1 → 1.2 adds `mechanism`, `comparisons.*.definition_difference_decomposition`,
`predictions.Q1.holds_rule`, `*_limb.registered_tier`, `*_limb.as_registered`; nothing is renamed or removed.

### 3.5 Regeneration, and the proof that no number moved

`docs/data/p5_3b_nortg.json` is **regenerated by `nortg_campaign report`**, never patched by hand
(P8.4b MJ-3's lesson: an artifact nothing in the repo writes). Inputs: the MAIN tree's `output/p5_3b/`
(secured 2026-09-10, 39/39), `output/p8_4b_rederivation/`, `docs/data/p4_6_grid.json`,
`p4_7_grid.json`, `p5_3a_rtg_probe.json`, and the new decomposition artifact.

**Run `report` from a CLEAN tree**: commit the code, run, commit the artifact. The regenerated
`runtime.git_dirty` must be `false` (the merged one carries `true`; MJ-7 is deferred, but there is no
reason to repeat it).

**One-off check, output pasted in the packet:** load `git show bd36a0a:docs/data/p5_3b_nortg.json` and
the regenerated file; walk every JSON path; assert the set of paths that differ is a **subset** of this
enumerated set, and print it:

```
format_version
predictions.Q1.holds  predictions.Q1.holds_rule
predictions.Q1.largest_limb.{registered_tier,as_registered}
predictions.Q1.smallest_limb.{registered_tier,as_registered}
comparisons.<tier>.definition_difference_decomposition   (new)
mechanism                                                 (new)
runtime.*
```

Anything else that differs is a **finding**, not something to add to the list. Every `att_*`,
`mean_difference`, CI, `per_seed`, `discriminability`, `gates`, `cells`, `episodes` and
`reference_dt_cells` value must be `==`.

### 3.6 Driver: `offline/campaigns/p5_3b_decomp.sh` (new)

Same skeleton as `p5_3b.sh`: inputs from the MAIN tree, `set -euo pipefail`, per-stage skip-if-chunk-
exists, five seeds in parallel each pinned to one torch thread, logs per cell, **the one-shot
`AUTHORISED_TO_RUN` token consumed on start** (2026-09-10 ruling — it is a mechanism, not a courtesy),
and the manifest `output/SHA256SUMS_p5_3b_decomp.txt` written last from a `LC_ALL=C sort`.

**Where it writes — a ruling:** `/home/filip/rltraffic/output/p5_3b_decomp/` in the **MAIN** tree, not
the worktree. MJ-9 nearly lost `output/p5_3b/` because it existed only in a worktree; a new directory
in the main tree needs no securing step. The fence makes this safe: a function named
`assert_decomposition_writable` (NOT a second `assert_writable` — `DEFERRED` 57's class) that allows
exactly `output/p5_3b_decomp` and `output/SHA256SUMS_p5_3b_decomp.txt` and refuses every other component
under `output/`, including ones that do not exist yet. `output/p5_3b_decomp/` must not exist at first
start; later starts skip complete chunks.

---

## 4. Scope fence — what NOT to build

- ⛔ **BL-1** (`DEFERRED` 72): no `SHA256SUMS_p8_4b_rederivation.txt`, no
  `test_the_primary_definition_recomputes_from_the_reference_column`. ⭐ Note in the packet that the
  decomposition artifact carries `committed_att_engine` for all 1,500 `dt` rows and a re-rolled
  `att_engine_call` equal to it, so BL-1's guarding test becomes **writable from a clone** after this
  merge. Writing it is BL-1's task, not this one.
- ⛔ MJ-1, MJ-2, MJ-4, MJ-5, MJ-6, MJ-7, MJ-8, MJ-10 and the twelve minors (`DEFERRED` 73). If a fix
  here brushes one of them incidentally (e.g. `git_dirty: false` by process), disclose it in the
  packet's classified section; do not go looking.
- ⛔ No retraining. No new campaign cells. No change to `offline/engine_att_reference.py`,
  `offline/admission_probe.py`, `offline/rtg_ablation.py`, `offline/method_tier_grid.py`,
  `offline/dt_gate.py`. No edit to any committed artifact other than the regenerated
  `p5_3b_nortg.json` and the new `p5_3b_decomposition.json`.
- ⛔ No interpretation of *why* the mix50 ablation collapses. The paper sentence is written by the
  coordinator after the numbers exist.
- ⛔ No new dependency. No frozen file. `experiments/configs/` is not needed.
- ⛔ Do not "fix" `cells[*].att_horizon_mean` (MJ-2) while you are in `report_artifact` — it is a
  rename of a committed key and belongs to `DEFERRED` 73.

---

## 5. Gates, in order

| gate | what | proof |
|---|---|---|
| **G0** | plan file `docs/plans/p5.3b-fix.md`, plan mode, user approves | the plan lists every assumption with confidence; anything load-bearing under ~95 % is a question, not an assumption |
| **G1** | MJ-3 — code + tests + the executed mutation | pytest tail; the mutation's failing output |
| **G2** | decomposition module + driver + tests; **smoke: 1 `dt_nortg` cell and 1 `dt` cell × draws 1000–1001** through the real path, `reproduces_committed` true on all 4 episodes, and the measured s/episode | the 4 records, the timing, `output/p5_3b_decomp/` containing only the smoke chunks |
| **G3** | **pre-flight review** (`PROJECT_PLAN` §7, mandatory — the run exceeds one hour serial AND writes under `output/` while consuming reused merged columns) — coordinator spawns it on the branch; scope: can it destroy anything, does resume do what it claims | `docs/reviews/P5.3b-fix-preflight.md` CLEAR |
| **G4** | author writes the token; run in `tmux`; `report` for the decomposition; `nortg_campaign report` from a clean tree; the §3.5 diff; manifests | packet |
| **G5** | merge review — **blocking**, per the proportionate-review rule: inputs are gitignored and unbacked | `docs/reviews/P5.3b-fix.md` PASS |

Implementer autonomy (§7, 2026-08-31) applies: wrong paths, CLI shape, crashes with obvious causes —
fix and disclose. **A fix that creates a measurement, a change to a registered criterion, or two
explanations you cannot tell apart — stop.**

---

## 6. Tests — including the load-bearing ones

`tests/test_nortg_decomposition.py` (new):

1. 🔒 **Refusal on a mismatched committed row** — synthetic chunk with one episode whose
   `att_engine_call` differs from `committed_att_engine` by `1e-9`: `report` raises and the output path
   does not exist afterwards. **Mutation:** delete the refusal → this test fails. Repeat for
   `att_ours`, `deviation_c1`, `deviation_c3c`, `decomposition_residual` (parametrised).
2. 🔒 **The identity is checked, not assumed** — synthetic episode whose three terms do not sum to
   `att_ours − att_engine_call` → refused. Mutation: drop the check.
3. **Action recording is inert** — a stub factory emitting a known sequence; the wrapped callable
   returns arrays equal to the stub's (`np.array_equal`, same dtype), the record equals them, and the
   sha256 is recomputed in the test by a second route (`hashlib` over `np.asarray(seq, dtype=np.int64).tobytes()`).
4. **The fence** — refuses `output/p5_3b/x`, `output/p8_4b_rederivation/x`, `output/p4_dt/x`,
   `output/never_seen/x`; accepts `output/p5_3b_decomp/x.json` and `output/SHA256SUMS_p5_3b_decomp.txt`.
   Positive control: an accepted path really is accepted.
5. **Contrast consistency** — synthetic per-arm rows where `Δ_ours − Δ_engine ≠ Σ Δterms` beyond 1e-9 →
   refused; within 1e-9 → accepted.
6. **Cell completeness** — 30 chunks × 100 draws required; 29 → refused ("compared nothing" must never
   read as "found no differences").

`tests/test_p5_3b_artifact.py` (edit + add):

7. `predictions.Q1.holds is None` and `holds_rule` present (authorised edit of `:258`).
8. Every `comparisons.<tier>.definition_difference_decomposition` has `n_episodes_per_arm == 500`,
   `residual_max == 0.0`, `population + clock_origin + cadence == total` within 1e-9 for both arms and
   the contrast, and `source.sha256` equals the sha256 of `docs/data/p5_3b_decomposition.json` on disk.
9. `mechanism.outcome_identity` recomputes from the artifact's own `episodes` **by an independent route
   written in the test** (the eight-field equality, per seed) — and equals the shipped counts.
10. `mechanism` carries `registered: false` and `exploratory: true` as literal values.

`tests/test_nortg_campaign.py` (edit + add): §2.2's authorised edit at `:1212`; §2.3's new cases.

Hygiene: `bash scripts/check_test_hygiene.sh` and `bash scripts/check_english.sh` on every new file.
Test count must go **up**. ⚠️ Tests that need `output/` skip on CI; the skip ceiling (144) will move
at merge — the registered protocol is *merge, let it go red, classify `junit.xml`, commit the observed
value as its own commit*. **Do not pre-bump.**

---

## 7. Definition of Done

- [ ] `score_q1` implements §2.1; `holds_rule` shipped; the mutation in §2.3 executed and its output in the packet
- [ ] `offline/nortg_decomposition.py` + `offline/campaigns/p5_3b_decomp.sh`, reusing the surfaces named in §3.1, with no change to the read-only modules in §4
- [ ] 3,000/3,000 episodes with `reproduces_committed == true` on both definitions, `deviation_c1 == deviation_c3c == decomposition_residual == 0.0` — stated as a count, and any refusal reported verbatim
- [ ] `docs/data/p5_3b_decomposition.json` committed; `output/SHA256SUMS_p5_3b_decomp.txt` written; the ten existing manifests still verify (`735/735`) — run `sha256sum -c`, do not assume
- [ ] `docs/data/p5_3b_nortg.json` regenerated by `report` from a clean tree, `runtime.git_dirty == false`, format `p5.3b-nortg/1.2`, and the §3.5 path diff pasted in the packet
- [ ] Tests 1–10 written first, executed, real tail pasted; hygiene and English checks run
- [ ] Zero frozen files touched (`git diff --stat main...HEAD` proves it); zero new deps
- [ ] Pre-flight review CLEAR before the run; merge review PASS before merge
- [ ] `docs/returns/P5.3b-fix.md` from `docs/returns/TEMPLATE.md`, stating **which amendments and which brief it was written against** (§7 rule), with the AI-assistance record, and a **classified** disclosure section (not a running log)
- [ ] `PROJECT_PLAN` §6's `P5.3b-fix` box ticked in the merge commit

---

## 8. What the next reader must not be allowed to believe

- That the −409 vs −194 gap "is the never-admitted population". It is `Δ(engine − ours)`; what that
  gap is made of is what §3.4 measures.
- That `random`'s `+0.0000` is a null. It is non-discrimination, and after this task `Q1.holds` says
  `None` for that reason.
- That "same attractor" means "same policy" before §3.3's action identity is in the artifact.
- That the decomposition artifact closes BL-1. It makes BL-1's test *writable*; the test is not written.

## 9. Return Packet

`docs/returns/P5.3b-fix.md`, from `docs/returns/TEMPLATE.md`. Mandatory extras: the §2.3 mutation
output; the G2 smoke's four records and measured s/episode; the §3.5 path diff; the 3,000-count line;
`sha256sum -c` output for all eleven manifests; the classified disclosure section; the AI-assistance
record; the line *"written against BRIEF_33 (no amendments)"* or the amendment letters if any are issued.

---

# ✅ AMENDMENT A — 2026-09-10, ruled at the plan gate on `docs/plans/p5.3b-fix.md`

**Every claim in the plan's §2 was re-verified in the artifact by the coordinator before ruling:**
`tests/test_nortg_campaign.py:932` carries the literal; `engine_att_reference.py:853–892` implements
`att_ours − att_engine`; `ls output/SHA256SUMS_*.txt | wc -l` → 11; the 2.55× is
`docs/returns/P5.3b.md:178`. **The plan is approved. Commit it as the first commit on the branch, then
start G1.** Written against: this brief + Amendment A — say so in the packet.

## A1 — Q1: AUTHORISED, and the omission was the brief's

*Written authorisation, 2026-09-10: the implementer changes `tests/test_nortg_campaign.py:932` from
`"p5.3b-nortg/1.1"` to `"p5.3b-nortg/1.2"` and extends the adjacent comment with the 1.1 → 1.2
migration line. Same class as §2.2's two edits: a spec change ruled by the brief.* The brief ordered a
version bump without grepping for the literal it would break — *fix the class, not the sentence*,
applied to my own document. Refusing to touch the line unprompted was exactly right.

## A2 — Q2: CONFIRMED as proposed, with one addition against `DEFERRED` 63's class

`report_artifact(decomposition=None)` defaulted; the refusal lives in `_run_report`
(`FileNotFoundError` naming A13(b)) plus `assert_decomposition_embedded(payload)` called after assembly
and covered by a mutation. **Addition:** a guard whose *call* can be deleted with zero test failures is
`DEFERRED` 63 (P8.4b MJ-1/MN-2), so pin the call site too — a test that reads `_run_report`'s source
(`inspect.getsource`) and asserts the guard's name appears in it is cheap and is enough; the mutation
*delete the call* must fail it. No existing test is edited for Q2.

## A3 — Q3: CONFIRMED — the repo's orientation, the negation stated everywhere

Report in the module's `att_ours − att_engine` orientation. The artifact carries `orientation`,
`orientation_note` (that it is the negation of A13(b)'s writing) and the identity string; every
docstring says the same; `term_clock_origin` is expected **negative** and the artifact says so.
Re-signing would be a second implementation of `_decomposition_summary`, which §3.1 forbids — the
registered quantity is the decomposition, not its sign convention.

## A4 — Q4: all six accepted as disclosed

The smoke under `output/p5_3b_decomp/smoke/` (the fence holds; the root's *must not exist* rule is
read as *must hold no run chunks*) · `<arm>` in the chunk filename · the Python-side skip predicate
(complete AND zero-mismatch — `p5_3b.sh`'s `[ -f ]` is the shape `assert_probe_cell_is_ablated`'s
docstring records failing) · `runtime.concurrent_load` · `assert_recordable_tree` on `report` ·
importing the private `_decomposition_summary` (disclosed; no change to the module) · §2.7's reading
that `residual_max` corroborates criterion 1 and nothing further — correct, and the artifact says so.
⭐ **Re-reading the committed rows in `report` rather than trusting the chunk's `reproduces_committed`
flag is the right call and is the load-bearing design decision of this task.**

## A5 — Q5: both corrections to the brief are accepted and logged as the coordinator's

Eleven manifests today, twelve at delivery; paste real `sha256sum -c` counts for all. **The schedule
handed to the author is corrected: ≈61–69 min wall clock at 5 workers**, not ≈35 — the brief used an
assumed 5× where the packet had a *measured* 2.55×. G2's measured observed-DT rate replaces both
estimates before the run is scheduled. Branch point `78c6383` (the brief-issuing commit) supersedes
the header's `0852ba6`; the brief was written before its own commit existed.
