# BRIEF_37 — P7.3a: the zero-shot point of the C3 transfer curve (H3's confirmatory test), with the SUMO draw wiring and A17(f)'s gate

**Task id:** `P7.3a` · **Branch:** `task/p7.3a-zero-shot`, cut from `main` at the commit that carries this brief or later · **Worktree:** `/home/filip/rltraffic-p73a` (`git -C /home/filip/rltraffic worktree add /home/filip/rltraffic-p73a -b task/p7.3a-zero-shot main`; the main tree stays on `main`; the coordinator never commits from this tree).
**Mode:** Claude Code, plan mode first, `docs/plans/p7.3a.md` before any code.
**Registered as:** `PROJECT_PLAN` §6 **P7.3a** — the first half of P7.3, split on 2026-09-16 (Decisions Log) exactly as P7.2 was split into P7.2a/P7.2b. **P7.3b (`BRIEF_38`) is the few-shot curve and the anchor.**
**Design, all of it already registered — this brief chooses NOTHING scientific:** `PREREGISTRATION` §2 H3, §3.4 ρ, A15 (the ρ pair and the teleport-free regime), A16 (`align_info` the only door), A17 (Rule B `S = mean` at k = 100 the registered prompt, the two subjects, the declared exploratory arms, the A17(f) consistency gate), A18(c) (the held-out seed rule). Amendment letters and blocks of `BRIEF_36` apply where named.
**Compute:** one collection stage (100 logged SUMO episodes, unobserved, ≈ 20 min) and one evaluation campaign of **4,700 OBSERVED SUMO episodes** (4 arms × 2 subjects × 5 seeds × 100 draws + fixed-time 100 + MaxPressure 100 + random 5 × 100). ⚠️ **Costed at the OBSERVED rate, which `docs/notes/P7.3_COSTING_2026-09-14.md` did not use for this arm:** P7.1 measured 17.3–19.3 s per observed SUMO episode against 10.4–11.8 s unobserved (`docs/data/p7_1_metric_freeze.json:timing`), and `E_sumo` — the quantity ρ is registered on — exists only through the observer. ≈ 23 h sequential, ≈ 4.8 h at the measured ×4.9 on 8 workers. **The schedule is written from the rate measured at the plan gate (§5), not from this paragraph.**
**Contracts:** `docs/CONTRACTS.md` v1.1 — C1, C2, C6, C8, C9 — plus `BRIEF_36` E4: **every DT decision on SUMO is `agent.act(info, explore=False, update_memory=True)`**, pinned by a kwargs spy over every call. `envs/`, `agent/`, `algorithms/`, `experiments/**/*.py`, `scripts/`, `.claude/` frozen as always.
**Cost:** 3–4 implementer days including pre-flight and two merge reviews. The costing note's P7.3 total was 4–5 days for both halves; this half carries the wiring, so it is the larger.

> **Read order:** this brief → `PREREGISTRATION.md` §2 (H3, its test row), §3.4 (ρ), §12 rows **A18, A17, A16, A15** whole → `docs/returns/P7.2b.md` §17 (*what P7.3 will assume*) → `docs/reviews/P7.2b.md` (what the reviewers found weak, so it is not repeated) → `BRIEF_36` Amendments E and G (the call-shape contract; the gate-names-its-artifact rule) → `docs/notes/P7.3_COSTING_2026-09-14.md`.

---

## 0. What the coordinator verified before writing this (2026-09-16, by running commands, not by reading docstrings)

1. **`offline/collect.py::_require_cityflow_for_draws` (`:403-419`) refuses every `--flow-draw*` flag off CityFlow** and names P7.3 as the owner in its own docstring: a SUMO draw needs the generated `.sumocfg` (P7.2a made it: `draw_NNNN/parity/noteleport.sumocfg`) and a way to name the source. `--flow-source-json` does not exist anywhere in the module.
2. **`offline/dt_gate.py::evaluate_arm` (`:978`) builds `make_env(EnvSpec(...))` from `config_for_draw` and rolls through `horizon_rollout(env, choose_action_factory(env), episodes=1, seed=engine_seed)`.** It has no env-factory parameter. `horizon_rollout` (`offline/horizon_metric.py:66`) is env-agnostic and seeds `reset(seed + ep)`; at `episodes=1` that is `reset(seed=1000)` — A18(c)'s rule, already the probe's.
3. **`offline/aligned_env.py::aligned_sumo_env_for_draw` (`:151`) builds a PLAIN `SumoEnv`** through `collect_style_args → offline.collect._build_env_spec → experiments.envs.make_env` and wraps it. `AlignedEnv` admits any `SumoEnv` subclass by `isinstance`.
4. **`E_sumo` exists only through the observer.** `offline/sumo_att_reference.py::make_observer_sumo_env(sumocfg_path, settings, *, halting_check=True)` mirrors `make_env`'s SUMO branch and returns a subclass with `.recorder`; `reconstruct_sumo_episode(env.recorder)` yields `e_sumo`, the decomposition and the counts A15(b) requires on every cell. There is no post-hoc route from SUMO's own outputs in this repository (a `tripinfo`-based reconstruction would be a second instrument and is `DEFERRED` 83, not this brief's).
5. **The held-out pool's parity configurations exist in the MAIN tree only:** 100 directories `scenarios/draws/cityflow1x1/draw_1000…1099/parity/{noteleport.sumocfg, routes.rou.xml, provenance.json}`, P7.2a's output, gitignored.
6. **`docs/data/p7_2b_calibration.json`** carries eight targets per subject with `role` `registered_prompt` / `ablation`, and per probe draw `local_return` (float64) and `engine_seed_drawn` — the two fields A17(f) compares. Its sha256 is `92b1592de637cee187c56b988ce320611d89706c8f9f02c34fe7ebbe81658d86`.
7. **`docs/data/p4_heldout_thresholds.json`** (`DEFERRED` 80's reference): 1,100 rows over draws 1000–1099, arms `mappo1000` / `mappo500` / `maxpressure`, `engine_seed 1000`, fields `arm, att_horizon, draw_id, episode_reward, horizon_vehicle_count, seed`, with `env_settings`. `offline/materialise_draws.py::verify_p4_3_probe` (`:1554`) is the shape of the gate to generalise.
8. **`train_dt` has no warm-start and no continuation entry point exists** (`grep` over `offline/`): that is P7.3b's, and nothing here touches training.
9. **`tests/test_transfer_calibration.py:42-43` hardcodes absolute roots** into this machine (`DEFERRED` 82), which is why every local run of P7.2b's gated tests had zero skips and CI is the only place those gates ever fired. **New code in this task takes every root as a parameter or an environment variable with today's path as the default** (§3.6); it adds no new absolute path.

---

## 1. Why this task exists, and the split

C3's transfer curve has no point yet. P7.2b delivered the **prompt** (the registered Rule B target, −7185.354721543778 for `mappo1000` and −7431.0185327454665 for `mix50`) and proved the mechanics on one fenced draw; it published no evaluation number, by design. **P7.3a produces the zero-shot point — H3's confirmatory test — and the three anchors ρ needs on the same held-out pool.** It also builds the two pieces of infrastructure P7.3b cannot start without: the SUMO draw wiring through the trajectory logger (the few-shot corpora) and the env-factory evaluation path (every later SUMO evaluation).

**H3, as registered (§2):** *Zero-shot transfer is positive but incomplete — better than fixed-time, worse than within-backend MaxPressure — and closes substantially by k = 100.* Its test row: *MADT zero-shot in SUMO vs the within-backend fixed-time anchor, per paired scenario; unit: paired evaluation draw.* The first two clauses are P7.3a's; the third is P7.3b's.

**Why two briefs.** The whole of P7.3 is five pieces of engineering in five files, 4–5 implementer days and ≈ 25 h of compute; the brief format caps a task at ~2 source files plus tests, and P7.2b's implementer session reached 92 % context on a smaller task. P7.2 was split for the same reasons, and the split is what caught three smoke defects before any number was quoted.

---

## 2. Scope fence — what NOT to build

- **No fine-tuning, no continuation entry point, no anchor corpus, no draws 301–400.** P7.3b.
- **No target but hz1x1 bc-tyc.** grid4x4 enters only under A15(g)'s condition (`DEFERRED` 75); cologne is out (`DEFERRED` 74). Single intersection, so `DEFERRED` 78 (per-intersection targets) does not arise.
- **No MAPPO on SUMO** (A18(b), `DEFERRED` 75(e)). **No interface control** (P7.4).
- **The zero-shot arm set is DECLARED here and closed** (A17(d) permits reduction only by declaration before running): per subject, `b_mean_k100` (registered, H3), `b_max_k100`, `a_q1.0`, `naive` — the four the costing note priced. **Rule B's k = 5 and k = 20 targets are NOT evaluated zero-shot**: they are the few-shot prompts A18(d) attaches to the fine-tuned models, and evaluating them here would add 4,000 episodes to answer a question nobody registered. Recorded as an exclusion, not a choice made after a number.
- **No new instrument for `E_sumo`.** The observer is the registered route; `DEFERRED` 83 parks the cheaper alternative.
- **No re-rooting of existing test modules** (`DEFERRED` 82) — but see §0.9 for new code.
- **No change to `report`'s fence in `transfer_calibration.py`.** P7.2b's smoke stays fenced; it is a mechanics episode on draw 5, not an evaluation. **This brief lifts the fence for the arms it declares in §3.4 and for nothing else** — that is what "until P7.3's brief is written" (A3) meant.

---

## 3. Per-file requirements

### 3.0 `offline/materialise_draws.py` — `DEFERRED` 80's gate for the held-out band (CityFlow, ≈ 2 min at reference speed)
Generalise `verify_p4_3_probe` into `verify_against_artifact(artifact_path, *, arm, out_root)`: the same `run_probe` loop, the artifact's own `env_settings` and `engine_seed`, `==` on every recorded field per draw (`att_horizon`, `episode_reward`, `horizon_vehicle_count`), writes nothing under the tree; a CLI flag `--verify-heldout-thresholds` running it against `docs/data/p4_heldout_thresholds.json`'s `maxpressure` rows on draws 1000–1099. **100/100 or the task stops**: the held-out SUMO draws are derived from these parents, and their pedigree is proved here rather than inherited from P5.3a's five survivors. The existing `--verify-p4-3-probe` becomes a call to the same function and must stay byte-identical in behaviour (test).

### 3.1 `offline/collect.py` — the SUMO draw wiring, in the canonical frame
For `--backend sumo` with any `--flow-draw*` flag: resolve the draw's config through `offline.materialise_draws.parity_sumocfg_path(scenario_key, draw_id, out_root=…)` where `scenario_key` is `transfer_calibration.SCENARIO_KEY`'s mapping from the CityFlow scenario id (state the mapping; do not invent a flag if the id determines it); build the env through the A17(b) route the repository already exercises (`collect_style_args → _build_env_spec → make_env`) **and wrap it in `AlignedEnv`** so the logged per-intersection `state` is the canonical 25-wide frame (A16; A18(a) says *logged through the trajectory logger in the canonical frame*), rewards and every global scalar untouched (F2's pinned contract). `_require_cityflow_for_draws` becomes a check that the backend is wired **and** the draw's parity configuration exists, refusing as loudly as today otherwise. The manifest records: `backend`, the parity `.sumocfg` path and sha256, the alignment's provenance (the permutation and phase-map artifact `align_info` records), `engine_seed_requested` and `engine_seed_drawn`, `time_to_teleport_option` read from the engine, the vehicle-type set seen. **Seeding: one MaxPressure episode per draw on a fresh env with `reset(seed=1000)`** — A17(b)'s and A18(c)'s call. ⚠️ The collector's own seeding path (`--base-seed`, per-episode offsets) must be read, not assumed, and the plan states how the logged episode's `reset` seed is made equal to the probe's; A17(f) cannot hold otherwise.
Output: a v1.1 corpus directory (C6 alignment convention, `FORMAT_VERSION = "1.1"`) under a gitignored `datasets_sumo_v11/` root (parameter with a default; `datasets*/` is already ignored), with `SHA256SUMS`.

### 3.2 `offline/transfer_calibration.py` — A17(f), the consistency gate between P7.2b and P7.3
`assert_logged_corpus_matches_probe(corpus_dir, artifact_path)`: for every draw 201–300, the logged episode's return — the sum of its per-intersection local rewards under C6, which on hz1x1 is the single intersection's `local_return` — equals `probe[draw].local_return` **bit-for-bit under `==`**, and `engine_seed_drawn` equals. **100/100 or P7.3 stops**, and a mismatch is a finding reported with the draw, both values and the difference — never smoothed, never tolerated. The plan must show, from both code paths, that the two quantities are the same definition before the gate is trusted to mean anything (the "same number by two routes" argument of P7.2b applies: agreement confirms the arithmetic, the definitional identity is the claim).

### 3.3 `offline/dt_gate.py::evaluate_arm` — an env factory, with the CityFlow path byte-identical
Add `env_for_draw: Callable[[int], Any] | None = None`. `None` reproduces today's construction **exactly** (same `EnvSpec`, same fields, same order of operations), so P4's committed artifacts regenerate byte-identically through the default — the test in §4 T3 proves it by regenerating a committed slice under `==`. Nothing else in the function changes. The SUMO path passes `env_for_draw=lambda d: aligned_observer_env_for_draw(...)`.

### 3.4 `offline/aligned_env.py` — the observed, aligned env, and the recorder through the wrapper
`aligned_observer_env_for_draw(scenario_key, draw_id, *, out_root, settings)`: `make_observer_sumo_env(parity_sumocfg_path(...), settings)` wrapped in `AlignedEnv`. `AlignedEnv` gains an explicit `recorder` passthrough (a property that returns the wrapped env's `recorder` and raises with a named message if the wrapped env has none) so `reconstruct_sumo_episode` never reaches through `_env`. The forwarding contract (reward, flags, `step`, `average_travel_time`, every global scalar byte-identical — F2) is re-asserted over the observer subclass.

### 3.5 NEW `offline/transfer_curve.py` — the zero-shot evaluation and its artifact (P7.3b extends this file; design it so a few-shot cell is the same cell with a different checkpoint)
- **The cell** `(subject, arm, seed, draw)` = one observed, aligned SUMO episode on held-out draw `d ∈ 1000…1099`: env from §3.4; for a DT arm, `agent_with_target(env, checkpoint_path, declared_steps, target)` (`offline/rtg_calibration.py:597`) and `choose = lambda _env, info: agent.act(info, explore=False, update_memory=True)`; `horizon_rollout(env, choose, episodes=1, seed=1000)`; then `reconstruct_sumo_episode(env.recorder)`. Recorded per cell: `e_sumo`, `att_env` (the admitted pair), A13(b)'s decomposition and the counts (created, entered, never entered, pending, **teleports — asserted 0**, vanished), `vehicle_types_seen` (asserted `{"cf_parity"}`), `time_to_teleport_option` (asserted `"-1"`), `engine_seed_requested` / `_drawn`, `episode_reward` and `att_horizon` (**published — the fence is lifted for these declared arms**), the RTG diagnostics: `rtg_first == target`, `n_decisions_in_support` against both bounds, **and the per-decision reward series and RTG series, so `rtg_advanced_every_decision` is re-derivable by a reviewer (`DEFERRED` 81 closes here)**; actions in range; the checkpoint's sha256; the config's sha256; `canary_seconds`; provenance.
- **The arms**, targets READ from `docs/data/p7_2b_calibration.json` (sha-pinned) by `(rule, statistic, k)` and `role`, never recomputed: `b_mean_k100` (`role == registered_prompt`), `b_max_k100`, `a_q1.0`, `naive`; subjects `mappo1000` (`output/p4_dt/dt_seed{101,202,303,404,505}.pt`) and `mix50` (`output/p4_7/checkpoints/mix50_dt_seed*.pt`), checkpoints sha-pinned against `SHA256SUMS_p4_6` / `SHA256SUMS_p4_7`. **The anchors on the same pool:** `fixedtime` and `maxpressure`, one episode per draw; `random`, five policy seeds 1000–1004 per draw at the same engine seed (A18(c)). ⚠️ *Corrected 2026-09-16 by Amendment G1: `SHA256SUMS_p4_6` lists no `p4_dt/` path and every `SHA256SUMS_*` file is gitignored; the pins are the committed artifacts `p4_gate.json` (`mappo1000`) and `p4_7_training.json` (`mix50`).*
- **Chunks** per cell, atomic, resumable **by content** (the two regime checks, the type set, the checkpoint sha and the config sha re-derived from the chunk — a stored verdict is not evidence; `chunk_is_reusable`'s pattern), moved aside to `failed/` when unusable, never overwritten.
- **`report` → `docs/data/p7_3a_zero_shot.json`:** every cell row; per `(subject, arm, seed)` the 100 paired draws' `ρ_sumo` under **both** definitions — `ρ = (ATT_fixedtime − ATT_arm) / (ATT_fixedtime − ATT_maxpressure)` per draw with the anchors of the same draw (§3.4; fixed-time 0, MaxPressure 1 by construction; values outside [0, 1] are expected and never clipped); per `(subject, arm)` the seed-averaged effect with a CI **from the project's existing paired-CI helper** (the one P5.3b's decomposition used — the implementer names it and its resampling seed at the plan gate; **no new estimator**); **H3's two clauses stated as the registered inequalities and reported, not interpreted** (`ρ_sumo(b_mean_k100) > 0` against fixed-time; `< 1` against MaxPressure; the third clause is P7.3b's); the calibrated-vs-naive contrast per subject with the registered direction beside it (calibrated ≥ naive, larger on `mix50`), **exploratory, never promoted**; the in-support diagnostic per arm (a diagnostic; nothing selects on it); `what_this_does_not_say`. Refusals precede every write, including the last one (the F1 lesson: a test must reach the final refusal). The canary block as E1.4 shaped it (`canary.json` written after the token; `report` re-checks it and refuses without it).

### 3.6 NEW `offline/campaigns/p7_3a_zero_shot.sh` — the driver, P7.2b's shape plus a pool
Token consumed on start; lock; group-leader check; the canary **with its correctness half** before the token, `record-canary` after it, `tee -a`; trap before the token; **stage 1: collection** (§3.1, 100 episodes, unobserved) then **A17(f)** (§3.2) — the driver stops on a mismatch; **stage 2: the evaluation pool** — a `spawn` process pool of 8 workers, one env per process, cells chunked and resumable, the canary re-run and recorded; `report`; manifest last and atomic. Logs appended, never truncated. Every root the driver passes comes from variables with today's paths as defaults (§0.9). **The schedule printed in the driver's header is the rate measured at the plan gate**, with its date and n.

---

## 4. Tests — write them first, red for their own reasons, each named mutation executed and pasted
SUMO- and checkpoint-gated tests carry `skipif` predicates that **name the artifact the test consumes** (Amendment G2: a draw id, a checkpoint path — never a stand-in); at most four SUMO episodes in the suite beyond P7.2b's. `scripts/check_test_hygiene.sh` on every test file; the **repo-wide** `check_english.sh` sweep read together with its known benign lines.

- **T0 — `DEFERRED` 80's gate.** Synthetic artifact + monkeypatched `run_probe`: every field compared under `==`; a single perturbed field refuses, naming draw and field. *Mutation:* compare only `att_horizon` → the `episode_reward` perturbation survives → the test must die.
- **T1 (load-bearing, SUMO) — A17(f) on one draw.** Draw 201 logged through §3.1 reproduces `docs/data/p7_2b_calibration.json:probe["201"].local_return` and `engine_seed_drawn` under `==`. *Mutation:* seed 1001 → refused with both values printed.
- **T2 (SUMO) — the logged frame is canonical.** The corpus's per-intersection `state` width is 25 and the manifest carries the parity sha, the alignment provenance, `"-1"` and `{"cf_parity"}`. *Mutation:* skip the `AlignedEnv` wrap → width 32 → refused.
- **T3 (CityFlow, load-bearing) — the default factory is byte-identical.** Regenerate a committed P4 slice (name it: e.g. `docs/data/p4_heldout_thresholds.json`'s `maxpressure` rows on draws 1000–1001) through `evaluate_arm(env_for_draw=None)` and compare under `==`. *Mutation:* alter one `EnvSpec` field in the default path → the test dies.
- **T4 (SUMO) — the observed, aligned env.** `aligned_observer_env_for_draw` on draw 5 returns an `AlignedEnv` over an observer subclass; `recorder` is reachable through the wrapper; `reconstruct_sumo_episode` yields `e_sumo`; the forwarding contract holds under a spy. *Mutation:* return the plain env → `recorder` raises the named message.
- **T5 (SUMO + checkpoint) — the cell's mechanics.** 20 decisions of `dt_seed101` (`mappo1000`) on draw 5: actions in range, `rtg_first == target`, the per-decision reward series logged and `rtg[t] − rtg[t+1] == reward[t]` re-derived from it under the shift-by-one rule, teleports 0, `{"cf_parity"}`. *Mutation:* `explore=True` → the kwargs spy dies (E4).
- **T6 — the arm set and the targets.** Exactly the four declared arms per subject, targets read by `(rule, statistic, k)` with the registered one carrying `role == registered_prompt`, sha-pinned. *Mutation:* read `b_max` where `b_mean` is declared → the registered-prompt assertion dies; alter the artifact's sha → refused.
- **T7 — ρ arithmetic.** On synthetic cells: fixed-time → 0.0 and MaxPressure → 1.0 exactly under both definitions; pairing is per draw (a cell whose anchors come from another draw is refused). *Mutation:* swap the numerator's anchors → dies.
- **T8 — `report`.** Refuses a missing cell, teleports ≠ 0, a wrong type set, a wrong config sha, a cell from a checkpoint whose sha is not the declared one, a work directory without `canary.json`; **a test reaches the LAST refusal before `_write_json` and asserts the out-dir is empty**; regeneration is byte-identical at the recording commit. *Mutations:* `_write_json` above the last refusal → dies; the fence: an undeclared arm's `episode_reward` reaching the artifact → refused.
- **T9 — the driver.** Text assertions over **comment-free** text: token after canary, `record-canary` after the token, `set -euo pipefail`, `tee -a`, `mkdir -p "$LOGS"` after the token, the pool's worker count from a variable, the collection stage and A17(f) before any evaluation stage. *Mutation:* each moved or deleted → dies.

---

## 5. Gates, in order
1. **Plan gate** (`docs/plans/p7.3a.md`): the collector's seeding path answered from the code; the A17(f) identity argued from both routes; **one observed, aligned, DT-driven episode on draw 5 timed on mains at reference canary** — the campaign's rate basis, with its date; the paired-CI helper named; the arm set restated as declared. The coordinator rules on the plan.
2. **Pre-flight** (≤ 10 min, a reviewer, findings file): destruction and resume paths of the pool, the no-token driver, the canary's two halves, A17(f)'s stop.
3. **Stage-1 checkpoint, read by the coordinator before stage 2 starts:** the 100 logged episodes and **A17(f) 100/100**. This is cheap (≈ 20 min) and it is where a wiring defect surfaces; the evaluation pool does not start until the coordinator has read the gate's output from disk.
4. **The token**, written by the author, and the campaign in an **attended** tmux pane through a pipe and `tee -a`, on mains, canary first.
5. **Packet** `docs/returns/P7.3a.md`, committed with the artifact; then **two sequential merge reviewers** (code + mutations; the artifact recomputed from the raw cells — ρ, the CIs, the H3 inequalities, the fence for undeclared arms); ruling; merge with §6's P7.3a box ticked; the CI ceiling by the registered route (this task adds SUMO-, checkpoint- and campaign-output-gated tests; `re_measure_required_at` already names P7.3).

## 6. Definition of Done
- [ ] §3.0–3.6 delivered; no frozen file touched; no new dependency; no new hardcoded absolute path (§0.9).
- [ ] T0–T9 red first, then green; every named mutation executed and pasted; gated tests name their artifacts; hygiene and the repo-wide English sweep run.
- [ ] Stage 1: 100/100 on A17(f), read by the coordinator from disk before stage 2.
- [ ] The campaign complete under one token with the canary's both halves on disk, manifested; run 1's artifact, cells and logs copied to `output/p7_3a_runs/` with shas **before any re-roll** (§7's rule).
- [ ] `docs/data/p7_3a_zero_shot.json` committed with the packet; regenerates byte-identically at its recording commit.
- [ ] The packet reports H3's two clauses as registered inequalities with effect sizes and CIs, both ρ definitions, the exploratory contrast with its registered direction, the in-support diagnostic, both clocks per stage with the canary, the AI-assistance record — **and interprets nothing beyond §10's outcome rows.**

## 7. THE GATE TO P7.3b — mechanical and outcome-blind, in the author's words
**P7.3b starts when, and only when: A17(f) has passed 100/100; P4's committed artifacts regenerate byte-identically through the default env factory (T3); the zero-shot artifact's provenance is clean (`git_dirty: false`), its canary correctness half is on disk, and its fence for undeclared arms holds; and both merge reviews have ruled.** Every one of those is a property of the *pipeline*. **None of them is a property of the zero-shot number. P7.3b runs whatever the number is** — if `ρ_sumo(b_mean_k100)` is 0.9, P7.3b runs; if it is −4, P7.3b runs; if the DT collapses on SUMO as DTLight's did on Grid 4×4, the few-shot curve is exactly what the paper then needs, and P7.3b runs. This gate must never be read, quoted or applied as a stopping rule on the zero-shot result, and a session that finds itself reasoning *"given how the zero-shot came out, should P7.3b…"* stops and re-reads this paragraph. The pre-registration fixed the curve before any point existed; the gate protects 23 hours of compute from a broken logger, not from an unwelcome number.

## 8. Return Packet
`docs/returns/TEMPLATE.md`, plus: the plan-gate rate with its date; A17(f)'s 100/100 line with the two fields compared; the four-way cell counts (arms × subjects × seeds × draws, and the anchors) against the manifest; ρ under both definitions per `(subject, arm)` with CIs and the paired-CI helper named; H3's inequalities as reported; the contrast and its registered direction; the in-support diagnostic; the canary of every run with its file and sha; where the driver ran; the amendments and blocks written against, by letter; the AI-assistance record's four lines; *what P7.3b will assume*.

---

# ✅ AMENDMENT A — 2026-09-16, at the plan gate: PLAN APPROVED (`docs/plans/p7.3a.md` @ `baf1ca4`), with ten rulings

The five gate items are answered from the code and the transcript, and the coordinator verified each from disk: the transcript exists at the stated sha (`0133d4c8…`) with the stated numbers; the seeding line is `offline/collect.py:722`; the float32 hazard is real (`trajectory_logger.py:118`) and the integrality argument is sound; `mean_ci95` is analytic; `fixedtime` is in `collect.POLICIES` (`:191`); the anchors-through-the-door `KeyError` is in the transcript. **Proceed to code.**

## A1 — Q4: CONFIRMED. "Both definitions" means ρ on `E_sumo` and ρ on `att_env`
One formula (§3.4), two ATT definitions per A15(a)/(b), the pool-clock pair primary and the admitted pair beside it. Not two ρ formulas.

## A2 — ANCHORS RUN ON THE OBSERVED, UNWRAPPED ENV. The brief's §3.5 was wrong to say "env from §3.4" for every cell
Measured by the implementer, not reasoned: `align_info` drops outgoing lanes and re-keys to CityFlow ids, and MaxPressure's pressure is a difference over the env's own SUMO lane ids — wrapping it raises `KeyError: 'road_1_1_2_0'`. **DT arms: observed + aligned. Fixed-time, MaxPressure, random: observed, unwrapped.** T7b (an anchor cell built through `AlignedEnv` is refused) is accepted and required. A16 is unaffected: the door is the only route into a *CityFlow-trained model's* frame, and anchors have no frame to enter.

## A3 — Q3: CONFIRMED. The paired CI is analytic, there is no resampling seed, and the brief presumed one
`paired_stats → paired_comparison → dt_gate.mean_ci95`, `1.96·s/√n`, with the Wilcoxon beside it as P5.3b reported. The packet says *analytic, no resampling* in those words. The brief's phrase *and its resampling seed* was the coordinator presuming a bootstrap the repository does not have; repo wins, and the implementer was right to raise rather than invent.

## A4 — Q5: RESOLVED by the coordinator. `fixedtime` is `collect.POLICIES["fixedtime"]` (`offline/collect.py:191`, `make_fixedtime`), and P7.1's `sumo__fixedtime` cells ran it on SUMO.

## A5 — Q1, the observer's ×3: two candidates FALSIFIED by the coordinator, one left standing, none changes the schedule
- **Not the scenario.** Draw 5 carries **1,836** vehicles against the nominal file's **2,021** that P7.1 timed on (held-out draws 1,787–1,821); draw 5 is *lighter*, and its departure profile is flatter. Density cannot produce a larger observer ratio.
- **Not `libsumo`.** Default `False` on both occasions; P7.1's artifact records no such flag; the harness took `collect_style_args`' defaults.
- **Left standing: the observer's code moved after P7.1's timing.** `offline/sumo_att_reference.py` changed at `7efafa7` (P7.1's fix round, 113 diff lines) after the freeze campaign `7cebd4f` whose timing block is the 17.3–19.3 s. So P7.1 timed an earlier observer. **Ruling A9 turns this from a cost question into a value question.**
- **The machine is also 1.3–1.5× slower than yesterday** on an idle box on mains: canary 0.68 → 0.79 → 0.89 → 0.97 → **1.02 s** (coordinator, 14:05, load 0.65, AC online, 16 cores). Under the 2.0 s threshold, but a monotone drift over 24 h; flagged to the author (§7's second clause: power plan and thermals are invisible from inside the guest).
- **The schedule stands on the measurement, as §5 required.** ≈ 49 h sequential, ≈ 10 h at the measured ×4.9. The residual is recorded, not explained away.

## A6 — Q2: YES, measure fixed-time and random (n = 2 each, draw 5, fenced), and ADD a pool pilot at the pre-flight
The ×4.9 was measured on **unobserved** episodes; an observed cell is ≈ 3× the Python work per step and may scale differently across 16 cores. **Pre-flight adds: 16 observed DT cells on draw 5 (fenced — no outcome printed) through the real pool at 8 workers; the effective per-cell rate from that pilot is the driver header's schedule.** If it is materially worse than ×4.9, try 12 workers in the same pilot and record both; the campaign runs at whichever measured better.

## A7 — T2b (integrality before equality) and the float32 finding: ACCEPTED and carried to the packet
The gate asserts integrality of every stored reward first; a non-integral reward is a finding with the right diagnosis, never a tolerance.

## A8 — Assumption 5 (draw 5 representative, 85 %): ACCEPTED as stated
Held-out draws carry 1,787–1,821 vehicles against draw 5's 1,836; the campaign records per-cell `seconds` and the packet reports its own rate.

## A9 — NEW, REQUIRED before the campaign: the INSTRUMENT regenerates P7.1's frozen value
Because the observer changed after P7.1's freeze (A5), **one observed MaxPressure episode on the NOMINAL teleport-free parity config, under the seed and settings P7.1's `reproduction` block records, must reproduce P7.1's committed `e_sumo` and `att_env` for that cell under `==`** (`docs/data/p7_1_metric_freeze.json:cells.sumo_noteleport__maxpressure`, per-episode values). ≈ 45 s. If it does not reproduce, the observer's change altered the registered instrument's VALUE and P7.3a stops until that is understood — a cost change is harmless, a value change is not. Add as T4b, SUMO-gated, naming the nominal config.

## A10 — Recorded, no action: the traci start retry
`envs/sumo_env.py:196` starts traci with `numRetries=_TRACI_START_NUM_RETRIES`; two `Retrying in 1 seconds` lines appear per episode (≈ 2 s of the 37 s). It is a frozen file and a fixed cost — ≈ 2.6 h of the 49 h sequential, ≈ 0.5 h at 8 workers. Named so nobody tries to optimise it inside this task.

**Then: pre-flight (with A6's pilot and A9's regeneration), the stage-1 checkpoint read by the coordinator, the token, the campaign attended in tmux.**

## A11 — added 2026-09-16 ~15:00, on the author's measurement: the rate was taken on a THERMALLY CONSTRAINED machine
The canary read 1.02 s at 14:05 and **0.81 s** shortly after, with nothing changed but time since the previous SUMO runs; the laptop's underside intake is blocked (glass table). The author will fit a cooling pad before the campaign and measure the canary on it. **Until that measurement exists, the 37.06 s cell rate, the ≈ 3× observer ratio and the ≈ 10 h schedule are measurements under thermal constraint, and every quotation of them says so.** The driver's canary at each stage start is the rate basis of that stage (§7's rule); the pre-flight pilot (A6) is re-run on the pad if the pad's canary differs from the pilot's by more than 10 %.

---

# ✅ AMENDMENT B — 2026-09-16, on the author's ruling: the campaign runs in TWO DECLARED STAGES, so the confirmatory number exists early and there is slack if something breaks — NOT to rescue a deadline; the schedule is not tight

The compute is one night and the author presents in ten days. The coordinator's first framing of this ordering as a deadline rescue was wrong on the coordinator's own arithmetic and is withdrawn; the staging is for **early existence and slack**, which is a good reason on any calendar.

## B1 — The order, declared before any cell runs (A17(d))
- **Stage 1 — confirmatory:** `b_mean_k100` × `mappo1000`, `mix50` × seeds 101/202/303/404/505 × held-out draws 1000–1099 (1,000 DT cells), plus `fixedtime` and `maxpressure` on the same draws (200) — the two anchors §3.4's ρ uses. ≈ 1,200 observed cells.
- **Stage 2 — the rest:** `b_max_k100`, `a_q1.0`, `naive` for both subjects and all seeds (3,000) and `random` × 5 policy seeds (500), on the same draws, resumable in the same work directory under a second token; the driver re-runs the canary at its start.
- **Stage 2 is UNCONDITIONAL.** It runs whatever stage 1 shows, exactly as P7.3b runs whatever the zero-shot number is (§7). Nothing registered moves: five seeds, 100 draws, both subjects, every declared arm on the same draws and seeds. This is a sequence, not a cut, and the packet and the paper say so.

## B2 — `report --stage confirmatory` (an addition to §3.5)
A stage flag whose declared cell set is exactly B1's stage 1; `report` refuses if any stage-1 cell is missing or any undeclared arm is present, and writes `docs/data/p7_3a_zero_shot_stage1.json` with the same block layout as the final artifact (ρ under both definitions for `b_mean_k100` only, the H3 inequalities, the in-support diagnostic, canary, provenance). The final `report` writes `docs/data/p7_3a_zero_shot.json` over all cells and **cites the stage-1 artifact's sha256**; both are committed. The stage-1 cells are byte-identical between the two artifacts (a test: every stage-1 row in the final artifact `==` its row in the stage-1 artifact).

## B3 — What goes on a slide, and what does not
**Anything aimed outside the project states what the number is and what produced it — and carries no label from this project's internal process.** *Review* here means a second Claude session checking the first; *pre-review* on a slide would read as a qualification of the result, and it is not one. The honest description is the registered one: *ρ on the pool-clock ATT (E_sumo) within SUMO; MADT trained on CityFlow (`mappo1000` / `mix50`), zero-shot on SUMO through the frozen alignment, prompted by the Rule B mean k = 100 target from a 100-episode MaxPressure probe; 100 held-out demand draws × 5 seeds; fixed-time and MaxPressure anchors on the same draws; design registered as A15–A18 before any target-domain number existed.* The internal verification steps that precede a number's use (A17(f), both canary halves, the coordinator's read from disk) are conditions on the number being used at all, not qualifiers to print beside it.

---

# ✅ AMENDMENT C — 2026-09-16, on the A9 / A6 / A2 measurements (`40c641b`): the instrument REPRODUCES, the ×3 was the halting cross-check, and nine rulings

**Verified by the coordinator from disk, not from the report:** the tip's diff is one test file and one docs note, no production code; the transcript `a9_a6_a2.txt` at `ef551b98…`; P7.1's frozen ep0 row read by the coordinator directly from `docs/data/p7_1_metric_freeze.json` — `att_env 367.5816210045662`, `att_reference_created_population 475.4507669470559`, counts 2021 / 1752 / 0 — equals the transcript's fresh values under float64 `==`; `run_sumo_arm`'s docstring (`offline/sumo_att_reference.py:1195-1202`) and `:1292` (`env.halting_check = index < halting_episodes`) confirm the halting cross-check runs on the first episode only and costs 3.49×; the three `sumo_noteleport__*` cells carry `all_equal: null` in the freeze's reproduction block; T7b passes and its M1 (the door bypassed) dies under the coordinator's own mutation.

## C1 — A9: PASS. The observer's change at `7efafa7` moved cost, not value; P7.3a proceeds
Bit-for-bit in float64 on the nominal teleport-free MaxPressure episode at seed 1000 — stricter than P7.1's own float32 convention.

## C2 — Q6, the halting cross-check: run it on a DECLARED SUBSET, as the registered instrument itself does — never on every cell, never on none
The check is a verification of the *recorder* (its halting classification against SUMO's own, `halting_n_disagreeing_lane_seconds`), not part of the measurement; it is value-neutral by construction (reads only) and measured so (A9b, identical values, 55.96 → 16.01 s). P7.1's registered convention is `halting_episodes = 1` — the first episode of each arm. **P7.3a's convention: the check is ON for every cell on draw 1000** (every subject × arm × seed, and every anchor including all five `random` seeds — 47 cells of 4,700) **and OFF elsewhere; every chunk records `halting_checked` and, where checked, the three halting fields; `report` REFUSES any checked cell with `halting_n_disagreeing_lane_seconds ≠ 0`** — a recorder disagreement is a finding that stops the campaign, exactly as a teleport does. Declared here, before any cell runs.

## C3 — Q7: 12 workers, CONFIRMED (measured 1.27× better than 8 on 16 cores).

## C4 — Q8: the pool pilot is RE-RUN ON THE COOLING PAD before the token, as A11 requires (0.90 → ≤ 0.81 s is > 10 %); the driver header's schedule is the pad pilot's rate.

## C5 — Q9: P7.1's teleport-free cells were never reproduced by P7.1 — recorded as a CORRECTION to the freeze artifact's standing, and closed for all three anchors here
`p7_1_metric_freeze.json:reproduction` covers 9 of 12 cells (`n_verified: 45`); the three `sumo_noteleport__*` cells — the regime every P7.3 number uses — have `n = 0`. The artifact is not edited (it says so truthfully); the Decisions Log records it. **Required: A9 extended to `fixedtime` and `random` on the nominal teleport-free config, ep0, seed 1000, halting off (≈ 16 s each), reproducing their frozen rows under `==`** — so every anchor the campaign uses has been reproduced by today's instrument before it runs 4,700 cells. Reported in the packet as the first reproduction those three cells ever had.

## C6 — `e_sumo` versus `att_reference_created_population`: two names for one quantity, recorded as a naming trap
A15's text is correct — `reconstruct_sumo_episode` names the quantity `e_sumo` (`:352`, `:764`) — and the artifact key it is written to is `att_reference_created_population` (`:1252`). The ρ code reads the artifact key, documents the alias in one sentence, and the packet states it. A15 is not amended.

## C7 — The schedule, on the measurements: stage 1 ≈ 1 h, the full campaign ≈ 3–3.5 h at 12 workers under C2's convention, on the thermally constrained machine; the pad pilot (C4) sets the header. Three figures were wrong in three directions — the costing note's, the brief's and the plan's — and each was corrected by a measurement, which is what the gate is for.

## C8 — A2 / T7b: ACCEPTED. M1 killed (coordinator's run); M2 (the control deleted) is a non-mutation, correctly so recorded; the coordinator's M3 (the key-provenance assertion neutralised) survives because it weakens a diagnostic, not the claim — noted, not counted. The second half — the cell builder's explicit refusal — lands with `offline/transfer_curve.py`.

## C9 — Process: `git merge main` at the start of every implementer session. Amendments A11 and B were on `main` for an hour before they were read; nothing was harmed because they were read before anything was written, and that is the rule.

**Then: code (§3.0–3.6 with B2's `--stage confirmatory` and C2's halting convention), pre-flight with the pad pilot, the stage-1 checkpoint read by the coordinator, the token.**

---

# ⛔ AMENDMENT D — 2026-09-16, on §3.1/§3.2 (`e380ecb`): the wiring is RIGHT and A17(f) holds under the coordinator's own recomputation — but T1 and T2 were done BY HAND, not as tests, and the logger-side alignment is UNPINNED. Fix before §3.5 is built on it

**Verified from disk:** the collected `output/p7_3a_runs/collect_smoke/ep000000_seed1000_draw201.npz` — `ix0_local_reward` float32, 360 values, all integral; `sum` in float64 = **−23938.0 == `probe[201].local_return`**; `ix0_state` shape **(361, 25)**, canonical; `format_version 1.1`; manifest `backend: sumo` carrying `alignment_provenance`. The seam is as the report describes — `align_for_log` applied at `on_reset`, `on_action` and `on_step_result`, the env unwrapped for MaxPressure — and the gate checks integrality before `==` on both the float64 sum and the seed. Three coordinator mutants on the gate died (integrality dropped; seed equality dropped; and, by the implementer, the seed off by one).

## D1 — The brief's §3.1 said "wrap it in `AlignedEnv`"; the implementer's correction is ACCEPTED and is now the design
A16's door sits at the **logger boundary** for collection: the policy drives and reads the unwrapped SUMO env (MaxPressure's pressure is over SUMO lane ids — A2's fact, the same `KeyError`), and the info is aligned exactly where the logger is fed, so the corpus is in the canonical frame (A18(a)) with rewards untouched (F2). Found by running, not by reading; the `KeyError` is in the record.

## D2 — REQUIRED NOW: T1 and T2 as TESTS, because the coordinator's mutant SURVIVED
Bypassing `align_for_log` — logging the raw 32-wide info — left every test green: `grep` finds no test referencing `align_for_log`, `ix0_state`, a 25-wide assertion, `alignment_provenance`, `parity_sumocfg_sha256`, or a SUMO draw-sweep call. The three new `a17f` tests are gate tests over synthetic corpora (five tests in 1.5 s); the end-to-end run was manual. **§4 requires T1 (load-bearing, SUMO) and T2 (SUMO) as tests**, and they are required before §3.5 is built on this seam:
- **One module-scoped fixture performs ONE real collection** — draw 201, `--backend sumo --flow-draw 201 --base-seed 1000 --episodes 1`, through `collect.main` or the same entry the driver will use — into `tmp_path`, so T1 and T2 share one SUMO episode (the suite's four-episode cap, §4).
- **T1:** `assert_logged_corpus_matches_probe` over that corpus against `docs/data/p7_2b_calibration.json` passes, and the test ALSO asserts the two values directly — `sum(ix0_local_reward)` in float64 `== −23938.0` and `engine_seed_drawn == 437485271` read from the manifest — so the test does not merely trust the gate it exists to exercise. *Mutation:* `--base-seed 1001` → the gate refuses naming both values.
- **T2:** `ix0_state.shape[1] == 25`; the manifest carries `backend == "sumo"`, `parity_sumocfg` and its sha256 equal to the file's, `alignment_provenance` with `scenario` and the per-intersection permutation, `engine_seed_requested == 1000`, `engine_seed_drawn`, `time_to_teleport_option == "-1"`, `vehicle_types_seen == ["cf_parity"]`. *Mutation:* `logged = info` (the alignment bypassed) → the width assertion dies. **This is the mutant that survived; it must die.**
- Both gated with a predicate naming draw 201 (Amendment G2).

## D3 — The changed contract test is ACCEPTED
`test_collect_refuses_flow_draw_on_non_cityflow` encoded a contract §3.1 changes by design; its replacement asserts what remains refusable (an unwired backend by name; a SUMO sweep whose parity configuration is absent) and says why in its docstring. That is how a contract test is retired.

## D4 — T1-M2 (`==` loosened to `abs(d) < 1.0`) survived and is EQUIVALENT GIVEN THE GUARD, accepted as recorded
Two integral sums differ by at least 1.0, so the tolerance and `==` coincide on this domain; the only inputs that separate them are non-integral rewards, which the integrality guard refuses first — and that guard's own mutant dies. Equivalent given the guard, and only given it; the packet says so.

**Then §3.5 and §3.6 as planned.**

---

# ✅ AMENDMENT E — 2026-09-16, evening, on `0c0d5d1` (D2's T1/T2 as tests) and `124bbd1` (§3.5a, the declarative core): D2 is SATISFIED under the coordinator's own mutant runs, §3.5a is ACCEPTED with one test assertion required, and §3.5b's obligations are listed so the split loses nothing

**Verified from disk, by the coordinator's commands, in a throwaway detached worktree at `124bbd1` — never the implementer's tree.** The branch tip had moved twice past the `94228fd` the handoff recorded as unread: `94228fd` is a clean merge of `main` (`git show --cc` → 0 conflict hunks; three docs files), `0c0d5d1` adds 138 lines to `tests/test_transfer_calibration.py`, `124bbd1` adds `offline/transfer_curve.py` (266) and `tests/test_transfer_curve.py` (151). Worktree clean. Frozen guard exit 0; `check_english.sh` exit 0 on the three files; `check_test_hygiene.sh` exit 0 on both test files; branch footprint against `main` 13 files, +2,207 / −57, none frozen. **T1/T2 baseline: 2 passed in 14.06 s — a real SUMO episode, not a skip** (SUMO on `PATH`, `traci` importable, draw 201's parity config on disk, `noteleport.sumocfg` sha `c177e962…`). `transfer_curve` baseline: 8 passed in 1.32 s. Whole suite at `124bbd1`: **1983 passed, 94 skipped in 187.13 s, exit 0** — the figure the implementer reported, now measured by the coordinator (the throwaway tree has no `output/` or `datasets_v11/`, so the artifact-gated tests skip there exactly as they do in CI).

## E1 — D2 SATISFIED: the survivor is dead, and the seam is pinned. §3.5 may stand on it
Each mutant applied by `sed`, its diff printed, the tree restored to 0 dirty paths before the next:
- **D2-M1, the survivor** (`logged = info` at both sites, `collect.py:857` and `:869`): **T2 FAILS at `assert 32 == 25`** with its named message; **T1 PASSES.** KILLED. The T1-green / T2-red shape is the survival's anatomy: alignment touches the state and not the rewards (F2), so only a width assertion can see it — which is why the by-hand T2 left the suite green.
- **D2-M1b / D2-M1c (coordinator's): bypass at the `on_reset` site only, then at the `on_step_result` site only.** Both ERROR at fixture setup in the **logger's own guard** — `LoggerStateError: lane_vehicle_count changed its lane set mid-episode` (16 SUMO lanes against 8 canonical). Killed one layer down, by P1's logger, not by T2. Recorded: T2's width assertion is the only *test-level* pin on the seam; the single-site variants are refused by an older guard. Adequate; nothing further required.
- **D2-M2** (`--base-seed 1001` in the fixture): T1 `assert -23997.0 == -23938.0`, T2 `assert 1001 == 1000`. KILLED by both, exactly as §4 predicted.
- **D2-M3** (`alignment_provenance` dropped): T2 `KeyError: 'alignment_provenance'`. KILLED.
The ordering D2 required was honoured in commit order — `0c0d5d1` precedes `124bbd1` — and the coordinator's read came after both; that is acceptable, because D2 asked for the tests to exist before the build, not for a second gate.

## E2 — REQUIRED in §3.5b's commit: `assert tcv.HALTING_CHECK_DRAW == 1000`, citing C2 by name
**The coordinator's mutant `HALTING_CHECK_DRAW = 1001` SURVIVES — 8 passed.** `test_the_halting_cross_check_subset_is_declared_and_small` pins the subset's *size* (47) and not its *identity*; C2 declares draw 1000 by name. Not a scientific defect — A9b measured the check value-neutral — but a declaration that can move silently is this project's signature error in miniature, and the fix is one line. The other eight mutants died: T6-M1 (`ValueError … the artifact is the registration`), T6-M2 (`DID NOT RAISE`), T7-M1 (by two tests), T7-M2 (`0.0 == -1.0`), T7-M3 (coordinator: denominator guard removed → `ZeroDivisionError`, not the required `ValueError`), B1-M1 (by two: 4,300 ≠ 4,700 and 47's decomposition), B1-M2 (2,200 ≠ 1,200), B1-M3 (coordinator: `range(1000, 1099)` → 4,653 ≠ 4,700 and the pool set).

## E3 — The §3.5a / §3.5b split is ACCEPTED, and these are §3.5b's obligations — enumerated so the split drops none of them
`ArmSpec`'s `(rule, statistic, k)` keys were checked by the coordinator against the artifact's own rows, not through the test that reads them: each subject carries exactly one row per declared key; `(rule_b, mean, 100)` is the only `registered_prompt`; `(rule_a, q1.0, 100)` and `(naive, none, None)` are `ablation`; sha `92b1592d…` as registered. §3.5b must deliver:
- **(a) the artifact LOADER with the sha pin.** T6's second mutation — *alter the artifact's sha → refused* — is **untestable at §3.5a** because `targets_for_subject` takes a mapping and reads no file. It becomes testable with the loader and must be executed and pasted then.
- **(b)** checkpoint sha pins against `SHA256SUMS_p4_6` / `SHA256SUMS_p4_7`, and the config sha, both re-derived from the chunk on resume. ⚠️ *Corrected 2026-09-16 by Amendment G1: `SHA256SUMS_p4_6` lists no `p4_dt/` path and every `SHA256SUMS_*` file is gitignored; the pins are the committed artifacts `p4_gate.json` (`mappo1000`) and `p4_7_training.json` (`mix50`).*
- **(c) the cell runner:** DT cells observed + aligned through `agent_with_target`, `BRIEF_36` E4's kwargs spy on every call (`explore=False, update_memory=True`); anchors observed, **unwrapped** (A2), with the explicit refusal of an anchor cell built through `AlignedEnv` (C8's second half); `halting_check` ON iff `draw_id == HALTING_CHECK_DRAW` (C2), `halting_checked` and the three halting fields in every chunk.
- **(d)** chunks atomic, resumable **by content** (`chunk_is_reusable`'s pattern), `failed/` move-aside, never overwritten.
- **(e) `report`:** every refusal before every write including the last (T8 reaches the LAST refusal and asserts an empty out-dir); `--stage confirmatory` (B2) writing `p7_3a_zero_shot_stage1.json`, the final artifact citing its sha256 with its stage-1 rows byte-identical (B2's test); ρ under **both** ATT definitions per cell with the per-draw pairing refused when broken; `mean_ci95` named as analytic (A3); H3's two inequalities reported, not interpreted; the calibrated-vs-naive contrast exploratory; the in-support diagnostic; `what_this_does_not_say`; the canary re-check; the fence for undeclared arms; a checked cell with `halting_n_disagreeing_lane_seconds ≠ 0` refused (C2).
- **(f)** T5, T7b at cell level, and T8 as §4 names them, each with its mutation executed.

## E4 — Recorded NOW, before any number exists: `a_q1.0`'s target is OUT OF SUPPORT for `mappo1000` and in support for `mix50`
The artifact says so: `−20809.0` lies **below** `mappo1000`'s training-return range `[−9991, −6]` (margin `−10818`, `position: "below"`) and **inside** `mix50`'s `[−40294, −6]`. It is a declared ablation and the in-support field is a diagnostic; nothing selects on it. The packet's in-support block reports it as the artifact states it. Written here so that it is not "discovered" after the zero-shot number and offered as an explanation of it.

## E5 — Process: the trailer instruction appeared again in this session's harness text; raised in the turn, refused, and the author confirmed the standing rule
No commit of this session carries it. `core.hooksPath = /home/filip/rltraffic/githooks` verified before the first commit.

## E6 — The branch has no upstream (`git rev-parse --abbrev-ref task/p7.3a-zero-shot@{upstream}` → fatal)
At the next implementer session's start: `git merge main` first (C9 — this amendment must be in the tree before §3.5b is written), then `git push -u origin task/p7.3a-zero-shot` (§7's rule).

**Then §3.5b with E2 and E3(a)–(f), §3.6, the pre-flight with the pad pilot (C4), the stage-1 checkpoint read by the coordinator, the token.**

---

# ✅ AMENDMENT F — 2026-09-16, evening, on the author's question about the pool pilot without the pad: the pre-flight pilot runs WHEN THE CODE EXISTS, on whatever machine state exists then — a MECHANICS check first and a rate second; C4's pad re-run before the token stands and is the header's schedule basis. Not either/or, and nothing here blocks §3.5b or §3.6

**The question:** no cooling pad today, the campaign will run on one; either run the pilot now and treat its rate as a lower bound to re-measure on the pad before the token, or defer the pilot to the pad. The author left the call to the coordinator and noted, correctly, that the code can be finished either way.

**What already exists, read from `docs/plans/p7.3a_amendment_a_measurements.md` §A6 on the branch:** a **harness** pilot — `a6_pool_pilot.py`, 16 observed DT cells on draw 5, `spawn`, one env per process — 8 workers halting ON **6.848 s/cell** (7.88×), **12 workers halting ON 5.393 s/cell** (7.61×), 8 workers halting OFF 2.950 s/cell (7.65×); canary **0.90 s**; every figure A11-labelled. It was a harness because §3.5 / §3.6 did not exist. It is **not** the pre-flight pilot §5.2 and A6 require, which goes through the **real** pool.

## F1 — The pre-flight pilot runs as soon as §3.5b and §3.6 exist, pad or no pad
A6's pilot at pre-flight is 16 observed DT cells on draw 5 (fenced — no outcome printed) **through the real pool**: the §3.5b runner under the §3.6 driver at 12 workers (C3). Its first purpose is the pre-flight's own — the destruction and resume paths, the no-token driver, one env per process, chunks resumable by content, both canary halves, A17(f)'s stop. Those are the findings that force rework; P7.2b's pre-flight found three; rework wants slack, and deferring the pilot to the pad would move that check to the last moment before the token, which is the worst place for it. The rate is a by-product, recorded with its own canary and labelled *measured under thermal constraint* (A11). Cost ≈ 2–4 minutes; the pad run costs the same again.

## F2 — "Lower bound" is not the right word; "pessimistic expectation, labelled" is
The canary moved 0.68 → 1.02 → 0.81 s in a day with nothing changed but time. A rate taken today is *expected* to be slower than the pad's, but nothing bounds the campaign-day machine state — a warmer room is enough. So the pre-pad rate is recorded as a labelled expectation, not as a bound, and **the only rate that governs a stage is the canary the driver runs at that stage's start** (A11's last sentence; §7's rule); the packet reports each stage's own.

## F3 — C4 stands, under A11's 10 % clause: the pad re-run before the token is the header's schedule basis
On the pad, before the token: the same 16 cells through the same pool, its canary recorded. **If the pad's canary differs from the pre-flight pilot's by more than 10 %, the re-run is required and the driver header's schedule is its rate; if within 10 %, A11 does not require it and the pre-flight rate stands, with its label.** The header names the pilot it comes from — canary, date, n — so a header never quotes a rate for a machine state the run did not have, which is the author's own point and the reason this is written down.

**Nothing changes for the implementer's next commit: §3.5b with E2 and E3(a)–(f), then §3.6, then the pre-flight including F1's pilot.**

---

# ✅ AMENDMENT G — 2026-09-16, late: §3.5b PLAN APPROVED (`docs/plans/p7.3a_section_3_5b.md`, read from the worktree while still uncommitted), with eleven rulings — and a COORDINATOR ERROR in §3.5 / E3(b) corrected: the manifest the brief named does not list the files, and no `SHA256SUMS_*` file is committed

**Verified by the coordinator from the artifacts, not from the plan's prose:** `SHA256SUMS_p4_6.txt` lists `p4_6/checkpoints/*` (125 lines) and no `p4_dt/` path; `grep -l p4_dt output/SHA256SUMS_*.txt` is empty; `.gitignore:228` ignores `output/`, so **every** `SHA256SUMS_*` manifest is a local file; `docs/data/p4_gate.json` `checkpoints[seed]` carries `path` and `sha256` for the five `mappo1000` checkpoints and **all five equal the files on disk under the coordinator's own recomputation**; `docs/data/p4_7_training.json` `runs[35..39]` carry `tier == "mix50"`, `seed`, `file_sha256` (the checkpoint path in those rows is an old worktree's, so a match is by `(tier, seed)`, never by path or index); P7.2b's artifact names checkpoints by **path only** — no digest anywhere in it; `_roll_episodes` passes the final step's `info` to `on_episode` and `_record` reads `info["average_travel_time"]`, while `horizon_rollout` appends `info.get("average_travel_time", 0.0)` after every step and takes `samples[-1]`; `AlignedEnv` defines `__getattr__` and no `__setattr__`; `aligned_observer_env_for_draw` takes `halting_check` (default `False`, `aligned_env.py:213-218`); `build_policy` seeds `default_rng(args.base_seed)` (`sumo_att_reference.py:1146`); `.claude/settings.json` lists the four `git` patterns under **`ask`**, the `deny` list closing at line 22.

## G1 — COORDINATOR ERROR, caught by the implementer: §3.5 (line 71) and E3(b) named `SHA256SUMS_p4_6` as the pin for `mappo1000`. It lists none of those files, the repository has said so since `BRIEF_27` (`nortg_campaign.TIER_MANIFEST["mappo1000"] = None`, `DEFERRED` 56), and the manifests are not committed anyway
Two errors in one clause: a manifest name written from memory (the class in §2 of the coordinator's own rules — a claim about a file's content, stated without opening the file), repeated once in Amendment E without checking. Both lines now carry a dated correction pointing here. **RULING — symmetric, and stronger than the implementer's asymmetric proposal:** each subject's checkpoints are pinned against a **committed** artifact, because a gitignored manifest was never the strongest record available. `mappo1000` → `docs/data/p4_gate.json` `checkpoints[str(seed)].sha256`. `mix50` → `docs/data/p4_7_training.json`, the run row with `tier == "mix50"` and `seed == str(seed)`, its `file_sha256`. Where a local `SHA256SUMS_*` manifest also lists the file (`p4_7` does), it is checked too. Every chunk records `checkpoint_sha256` recomputed from the file and `sha256_checked_against` naming every record compared; a file absent from its committed record is a refusal. `DEFERRED` 56 is named in the packet, and so is this: **P7.3a's artifact is the first in the project to pin the subjects' checkpoints by digest; P7.2b's names them by path only.**

## G2 — The demand identity is the `.sumocfg` AND the routes file it names
The chunk records `config_sha256` (`noteleport.sumocfg`) **and** `routes_sha256` (`routes.rou.xml`), both checked against the draw's `provenance.json` `files` block at run time and re-derived on resume (`chunk_is_reusable`). A cfg digest alone does not pin the demand across a resume — the cfg names the routes file, and a regenerated routes file leaves the cfg's digest unchanged. ρ pairs an arm with the anchors of the same draw; all three cells must provably have run the same demand.

## G3 — ONE construction path for the observed env, not a mirror
The plan's *"six lines mirroring `aligned_env.py:247-260`"* is a second copy of the construction, and two copies diverge silently the day a setting is added to one. Refactor: `observer_env_for_draw(scenario_key, draw_id, *, arm, out_root, sentinel_out_dir, halting_check)` returns the **unwrapped** observer; `aligned_observer_env_for_draw(...)` becomes `AlignedEnv(observer_env_for_draw(...), declared_alignment())`. T4's contract over the aligned one is unchanged. `assert_env_matches_cell` (C8) stays as planned, called by `run_cell` on every cell.

## G4 — `att_env ≡ att_horizon`: CONFIRMED from both sources, with one guard
Same key, same final `info`, same loop shape (`max_steps`, break on terminate/truncate). One guard, because of `horizon_rollout`'s `.get(…, 0.0)` default: the cell refuses a final `info` without `average_travel_time` — a silently defaulted `att_horizon` is not a measurement. Record one value, name it both ways, as the plan says.

## G5 — `halting_check` at construction: CONFIRMED. C2 as `halting_check=(draw_id == HALTING_CHECK_DRAW)` at build time.

## G6 — `random`'s policy seed in `args.base_seed`: CONFIRMED; the engine seed stays `horizon_rollout`'s 1000. The chunk records `policy_seed` and `engine_seed_requested` separately.

## G7 — Q3 (the Wilcoxon beside the analytic CI): ALREADY RULED by A3 — *"with the Wilcoxon beside it as P5.3b reported."* Both blocks; the analytic ρ CI is the registered primary; `paired_comparison` on ATT against each anchor beside it.

## G8 — Q2 (two commits; `PARTIAL` over an unverified driver): YES. The plan file is committed with the first §3.5b commit, as `p7.3a.md` was.

## G9 — The permission claim is a misreading, and the merge and push are the IMPLEMENTER'S to run
`Bash(git merge:*)`, `Bash(git push:*)`, `Bash(git checkout main:*)` — and `Bash(git commit:*)`, through which the session commits every day — are under **`ask`** in `.claude/settings.json`; the `deny` list ends at `git reset --hard`. Nothing in `CLAUDE.md` forbids merging `main` **into** a task branch or pushing that branch; rule 4 forbids committing **to** `main`. So: the implementer runs `git merge main` and `git push -u origin task/p7.3a-zero-shot` from the worktree; the author approves the prompt in that terminal. C9 stands as written. The instinct — refusing to route around what it believed was a deny — was the right one; only the reading was wrong.

## G10 — The trailer instruction: raised again, refused; the author confirmed the standing rule this evening (E5). Nothing to rule.

## G11 — Everything else in the plan stands as written: the loader digest-first (E3a), `chunk_is_reusable` re-deriving identity from disk (E3d), `report`'s nine-step order with `_write_json` last and T8 reaching the refusal above it (E3e), the reward and RTG series stored (`DEFERRED` 81), one new real-simulator episode (T5), G2's named artifacts in every predicate.

**Then: merge and push (G9), red tests, §3.5b, commit; §3.6, commit; the packet — then the pre-flight with F1's pilot.**

---

# ⛔ AMENDMENT H — 2026-09-16, night, on §3.5b + §3.6 at `55498ab` (packet `docs/returns/P7.3a-3.5b-3.6.md`): FIX-FIRST, one round, BEFORE the pre-flight. The code is largely right and the packet is honest; but A17(f) AS WIRED DOES NOT ENFORCE 100/100, the confirmatory stage cannot be restarted, and eight refusals or orderings the packet describes have no test that would notice their removal

**Verified by the coordinator from disk and by running, in two throwaway worktrees:** tip `55498ab`, tree clean, Amendment G an ancestor of `1641198`, no trailer in any of the four new commits; **`origin/task/p7.3a-zero-shot` is at `f6f57ec` — the three commits after the merge are NOT pushed**, whatever the packet's §1 implies. Whole suite **2037 passed / 94 skipped, exit 0** (coordinator's run; the reviewer's independent run agrees). Frozen guard, English, hygiene 0/0/0. All 14 `output/SHA256SUMS_*.txt` verify today, so the driver's last step is not a latent failure. `p4_7_training.json` carries 20 `mix50` rows, four methods × five `int` seeds (coordinator's count). **The contract reviewer ran 22 mutants in its own worktree and returned PASS-WITH-FINDINGS (0 blocking / 3 major / 8 minor) with a re-derivation of the RTG identity by raw arithmetic on a real 20-decision episode;** the coordinator ran six more, of which two survived, and one direct falsification the reviewer had listed as *not verified*.

## H0 — COORDINATOR ERROR, the second in Amendment G1, caught by the implementer: the `mix50` lookup key
G1 wrote *"the run row with `tier == "mix50"` and `seed == str(seed)`"*. The artifact carries **four** rows per `(tier, seed)` — `bc`, `bc_top10`, `iql`, `dt` — and stores `seed` as an **int**. The coordinator had printed two rows through `str(v)` and generalised from them: a sample of two, and the quotation marks were the coordinator's own. The implementer's `(tier, method == "dt", seed: int)` with exactly one match **is the ruling**; G1's prose is corrected by this paragraph, and the packet's open question 1 is answered CONFIRMED.

## H1 — BLOCKING: A17(f) as wired does not enforce "100/100 or stop"
`assert_logged_corpus_matches_probe` defaults `draw_ids=None` to *the draws present in the corpus* (`transfer_calibration.py`, `wanted = sorted(episodes)`); the `a17f` CLI passes no draw set. **Coordinator's run of the CLI on the one-draw smoke corpus in `output/p7_3a_runs/collect_smoke/`: `n_checked: 1, all_match: true, exit 0`.** A 99-draw corpus passes the driver's gate, and P7.3b's few-shot source would be short a draw with nothing saying so. The reviewer verified only that a mismatch raises and propagates, and listed the 100-draw behaviour as not verified — correctly. **Required:** the `a17f` CLI passes `draw_ids=range(PROBE_DRAW_START_DEFAULT, PROBE_DRAW_END_DEFAULT)` (201–300), asserts `n_checked == n_matching == 100`, and prints one line the coordinator reads at the stage-1 checkpoint: `A17(f) 100/100`. It also loads the calibration through `load_calibration` so the gate stage refuses a moved artifact (reviewer MIN-7). *Test:* a synthetic 99-draw corpus is refused naming the missing draw; a 100-draw synthetic corpus with one mismatch is refused naming draw, both values and the difference. *Mutation:* the draw set dropped → the 99-draw test dies.

## H2 — BLOCKING for the pre-flight: the confirmatory stage cannot be restarted
`offline.collect` refuses a populated `--out-dir` (`trajectory_logger.py:450`; `--overwrite` is the only override, and the driver must NEVER pass it). After any failure past the collection — the pool, the report, the manifest — a restart dies at `collect`. **Required, in the driver:** stage 1 becomes *if `$CORPUS` exists → `a17f` over the full band → skip collection; else → collect → `a17f`*. A partial corpus is refused by H1's gate with its missing draws named and is moved aside **by hand**, never deleted by the driver. *T9:* `--overwrite` absent from the driver text; `a17f` present in both branches and before any `cells` call.

## H3 — MAJOR: `set -euo pipefail` is not asserted (reviewer MAJ-1, measured)
Both `set -eu` and the deleted line survive T9. The reviewer reproduced the consequence with the driver's own canary line shape: without `pipefail`, a failed correctness half yields `CANARY_LINE=''`, the script proceeds, **`rm -f "$TOKEN"` runs**, and only `record-canary` fails — a failed canary consumes the one-shot token, the defect class P7.2b's pre-flight found. One assertion.

## H4 — MAJOR: the per-draw pairing guard is pinned only by accident (reviewer MAJ-2)
`test_t8_report_refuses_a_draw_whose_anchors_were_never_declared` passes with the guard removed because its fixture has zero shared draws and the refusal it matches comes from `paired_comparison`. The reviewer's probe — DT cells on draws 1000 and 1001, anchors on 1001 only — **writes an artifact under the mutant, with draw 1000's ρ normalised by draw 1001's anchors**. *Required:* that one-shared-draw fixture as a test; the guard's own message matched.

## H5 — MAJOR: T5 delegates the RTG identity to the helper it should be independent of (reviewer MAJ-3)
`rtg_advanced_every_decision` is change-detection (`rtg[i] != rtg[i-1]` iff `reward[i-1] != 0`); it never compares magnitudes, so a double-counted reward passes. §4 T5 says *re-derived*. The reviewer did the re-derivation by raw arithmetic on a real episode and it holds: `rtg[t-1] − rtg[t] == reward_series[t-1]` exactly for t ≥ 2, `rtg_first == target`, step 0 forced to 0 as `agent/DTAgent.py:706-750` does. *Required:* T5 asserts that arithmetic on the stored series directly — `np.cumsum`-style, no helper — and the docstring states the convention. `DEFERRED` 81 then closes on a test, not on a sentence.

## H6 — MAJOR: two refusals with no test (coordinator's survivors MU5, MU6)
(a) `validate_cell_payload`'s check that `halting_checked` agrees with C2's declaration — `if False:` leaves 62 tests green. A resumed chunk from a run with the flag off on draw 1000 would pass into the artifact. *Tests:* draw 1000 with `halting_checked: False` refused; draw 1001 with `True` refused. (b) `run_cell`'s `att_env != rollout.att_horizon` refusal — `if False:` leaves 62 green; G4's *"two routes compared under `==`"* is a docstring. *Test:* a fake tap whose final info differs from the rollout's last sample → refused.

## H7 — MAJOR: `_h3_block.holds` reads as a verdict
A boolean on the point estimate named `holds` will be read as *confirmed* by anyone who opens the artifact, and the registered test row is the paired comparison. *Required:* rename to `point_estimate_satisfies` and add `ci95_entirely_satisfies` (the whole analytic interval on the claimed side), both mechanical, no verdict word anywhere. Nothing is recorded yet, so the format version stays `1.0`.

## H8 — Same round, cheap (reviewer MIN-1, MIN-3, MIN-5, MIN-6)
- **MIN-1:** `assert_env_matches_cell` before the simulator does any work — pin the ORDER with a spy env recording calls (the reviewer's mutant moving it after the rollout survived T5).
- **MIN-3:** `actions_in_range` hardcodes `< 8`; take the count from the env (`Utils.infer_action_counts`, `CLAUDE.md` rule 5) — a silent dimension assumption for P7.3b otherwise.
- **MIN-5:** `report --stage confirmatory` refuses once stage-2 chunks share the work directory, so the stage-1 artifact is regenerable only from a stage-1-only tree. **Ruling:** the completeness and `extra` checks run against `declared_cells(stage)` by name; every chunk on disk is still validated (an undeclared ARM anywhere refuses, B2's rule), and declared cells of the other stage are counted in the artifact as `n_chunks_outside_stage`, never read. The regeneration test then means what it says after stage 2.
- **MIN-6:** the `canary` subcommand checks before it prints; P7.2b printed first so the line survives a failed check. Match P7.2b.

## H9 — Recorded, no change: reviewer MIN-2 (the step-9 arm fence is unreachable under every tested path — defence in depth, keep it), MIN-8 (driver roots as literal assignments, P7.2b's shape); coordinator: `vehicle_types_seen` is read from the vehicles still present at the horizon — the recorder keeps no types, the collector and P7.2b's probe read it the same way, and the strong guarantee is P7.2a's provenance (`n_bound == n_vehicles`); a tripwire, stated as one in the packet. The reviewer's *theatre or misnamed* list (`"rest" in code` matches *restart*; the anchors-from-another-draw test is caught by completeness) is handed to the implementer for the same round.

## H10 — The packet's open questions
**Q1** CONFIRMED (H0). **Q2, one canary per stage:** CONFIRMED — A11's own sentence, *the driver re-runs the canary at every stage start, so each stage carries its own*; the departure from `BRIEF_36` E1.4's one-stage rule is recorded here. **Q3:** CONFIRMED — B2 verbatim, the final `report` is over all cells and cites the stage-1 sha. **Q4, who runs F1's pilot:** the implementer, in the session after this round, through a `pilot` route into `run_stage` (the `cells=` parameter exists for exactly this): **16 DT cells on draw 5, both subjects, fenced — no ATT, no ρ, no outcome printed — at 12 workers, work dir `output/p7_3a_runs/preflight_pilot/`, transcript with the canary line, date and n at `output/p7_3a_runs/preflight_pilot.txt`;** `report` is never called on it and nothing from that directory reaches `docs/data/`. Then the reviewer's pre-flight pass (§5.2: destruction and resume paths, the no-token driver, both canary halves, H1's stop) with a findings file. F3's pad question when the pad exists.

## H11 — Verified as specified, so it is not re-argued next round
E2/C2, E3(a), G1 as implemented (real identities: `mix50` checked against `['p4_7_training.json', 'SHA256SUMS_p4_7.txt']`, `mappo1000` against `['p4_gate.json']`, `deferred_56: true`), G2, G3 (no second construction path in `transfer_curve.py`), A2/C8 both directions, E4 (one `.act(` call, kwargs spy, AST test), G4, every `run_cell` refusal before the return and `_worker` writing only after, E3(d), `report`'s order with `_write_json` last and the out-dir-empty test, B2, the `cells=`/`docs/data` fence, ρ verbatim §3.4 under both keys never clipped, the per-draw unit (coordinator's MU1: the unit replaced by cells dies on `n`; MU2: mixed definitions die on the anchor identity), `mean_ci95` analytic with `resampling_seed: null`, the driver's order by two independent greps, `--flow-draws-range 201 301` half-open. Mutants: reviewer 22 (18 killed, 1 equivalent, 3 survivors → H3, H4, H8-MIN-1), coordinator 6 (4 killed, 2 survivors → H6).

## H12 — Not executed by anyone yet, and the pre-flight is where it happens
The `spawn` pool end to end; a full real episode through `run_cell`; A17(f) on a real 100-draw corpus; stage-1 versus final byte-identity on real data; the signal handler and the group-leader refusal under a real tmux pane. Until F1's pilot and the stage-1 checkpoint, every one of these is unproven and the packet says so.

**Then, in order: push the three commits (the implementer, the author approving the prompt) · H1–H8 with every named test and mutation, one commit, a packet addendum · the coordinator re-runs H1's falsification and the four survivors · F1's pilot · the reviewer's pre-flight · the stage-1 token.**
