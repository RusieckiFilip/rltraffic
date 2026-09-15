# BRIEF_36 — P7.2b: the target-domain probe and the cross-domain return-prompt calibration (A17, executed)

**Task id:** `P7.2b` · **Branch:** `task/p7.2b-calibration`, cut from `main` after `git fetch` (the commit that carries this brief or later; a header cannot know its own sha) · **Issued:** 2026-09-13
**Mode:** Claude Code, plan mode first, worktree `/home/filip/rltraffic-p53b` (`git -C /home/filip/rltraffic-p53b checkout -b task/p7.2b-calibration main`; the worktree is on the merged `task/p7.2a-sumo-draws`)
**Registered as:** `PROJECT_PLAN` §6 **P7.2b** — the executable half of P7.2, the paper's named method contribution (`PREREGISTRATION` §6.4, §10). **The protocol is FIXED by A17 (`v1.7-prereg-a17`, `cc620cc`, 2026-09-13) and this brief implements it; nothing here chooses anything A17 fixed.** Read A17 in full before the plan; where this brief and A17 disagree, A17 wins and you flag it. A15 (ρ on `att_engine` / `E_sumo`, the teleport-free regime) and A16 (`align_info` is the only route by which a SUMO observation enters a CityFlow-trained model's frame) bind.
**Compute:** one campaign of **100 SUMO MaxPressure episodes** (draws 201–300, one each) plus **two fenced smoke episodes** — at reference speed ≈ 25 min (P7.1: bare hz1x1 SUMO 10.4–11.8 s/episode, `n = 5` per arm). **Run the §7 machine-health canary first and write it beside every timing** (mains power, performance plan, `run_probe` on draw 0 ≈ 0.9 s). Writes under `output/p7_2b/` (new) only; reads `scenarios/draws/cityflow1x1/*/parity/` read-only. `tmux` foreground, driver with token, lock, resume, manifest — `BRIEF_33`/`BRIEF_34`'s shape.
**Contracts:** `docs/CONTRACTS.md` v1.1 — C1, C2 (the `info` dict; `reward` per intersection), C6 (alignment), C8, C9 (`dt` is the arm key). `envs/`, `agent/base.py`, `agent/utils/utils.py`, `utils/`, `metrics/`, `scripts/`, `.claude/` FROZEN. `agent/DTAgent.py` is NOT frozen but is **not touched by this task** (`DEFERRED` 78 owns its multi-intersection change; hz1x1 has one intersection).
**Cost:** 2–3 days. `HANDOFF_2026-09-01` put P7.2 at 3–5 days; P7.2a took one.

> **Read order:** this brief → `PREREGISTRATION.md` §12 rows **A17, A16, A15** (the whole rows) → `docs/plans/p4.3.md` §2 (Rule B's form and the two probe statistics) → `offline/rtg_calibration.py` module docstring, `run_probe` (`:357-441`), `rule_a_target` / `rule_b_target` / `source_domain_ratio` (`:461-527`), `agent_with_target` (`:597-645`), `in_support_counts` (`:559-596`), `training_rtg_range` (`:528-558`) → `offline/backend_alignment.py` module docstring and `align_info` (`:340-478`) → `offline/sumo_att_reference.py::collect_style_args` and its `make_observer_sumo_env` (the engine-read counters: teleports, vehicle type) → `offline/materialise_draws.py::parity_sumocfg_path` → `offline/horizon_metric.py::horizon_rollout` → `docs/data/p4_3_probe.json` (`budgets`, `episodes`, `env_settings`, `engine_seed`) → `docs/returns/P7.2a.md` §12. Where this brief disagrees with the repo, **the repo wins and you flag it**.

---

## 0. What the coordinator verified before writing this (2026-09-13, by running commands)

1. **A live `SumoEnv` on `hangzhou_1x1_bc-tyc`'s teleport-free parity config, built through `collect_style_args("sumo", "maxpressure", cfg) → offline.collect._build_env_spec → experiments.envs.make_env`** exposes: one intersection `intersection_1_1`, `num_phases 16`, 8 incoming lanes, `action_space Discrete(8)`, `Utils.infer_action_counts → [8]`, `max_steps 360`; a raw per-intersection `state` of width **32**; `avail_actions [0..7]`; payload keys `action_applied, avail_actions, current_phase, metrics, reward, state, time_in_phase`; global lane dicts of 16 entries each. **Through `align_info` the same payload has `state` width 25, `current_phase` mapped by A16's `2k → k+1`, `avail_actions` unchanged, and the `reward` key passed through; `Utils.state_from_info` returns `(25,)` and `Utils.extract_valid_actions(payload, 8)` returns 0–7.** After one step the aligned `reward` equals `−Σ lane_waiting` over the intersection's incoming lanes (SUMO ids). So a CityFlow-trained DT with `state_dim 25` can be driven by an aligned SUMO env with no change to the agent.
2. **Both registered subjects' checkpoints** — `output/p4_dt/dt_seed{101..505}.pt` (`target_rtg −5762.0`, `rtg_scale 9991.0`, training RTG range `[−9991, −6]`) and `output/p4_7/checkpoints/mix50_dt_seed{101..505}.pt` (`−5959.0`, `40223.0`, `[−40294, −6]`) — record **`provenance.gradient_steps = 40000`**, `context_length 20`, `state_dim 25`, and are digest-covered (`output/SHA256SUMS_p4_6/p4_7/p5_3b.txt`, `SHA256SUMS_p4_7.txt`). **`load_gate_checkpoint(env, path, declared_gradient_steps=40000)` is the loader for both** (`offline/dt_gate.py:948-977`; a mismatch refuses).
3. **`horizon_rollout` (`offline/horizon_metric.py`) and `agent_with_target` (`offline/rtg_calibration.py:597`) are env-agnostic**: the first needs `reset(seed)`, `step`, `max_steps`; the second needs `.intersections` and `.action_space` (through `DTAgent.__init__`, `agent/DTAgent.py:462-478`). `dt_gate.evaluate_arm` is **not** — it constructs `EnvSpec(backend="cityflow", …)` (`:1007-1013`). This task therefore needs a **wrapper env**, not a change to `dt_gate`; P7.3 will generalise `evaluate_arm` with an env factory later (§7).
4. **`run_probe` (`offline/rtg_calibration.py:357`) is CityFlow-hardcoded and single-intersection** (`backend="cityflow"` at `:389`; refuses >1 intersection at `:396-401`). Its loop is the twin this task builds for SUMO: fresh env per draw, `env.reset(seed=int(engine_seed))`, `agent.act(info)` → `env.step`, per-step `info` kept, `episode_return_two_routes(post_step, ix_id, incoming_lanes)` (`:310-356`) under `==`. **A17(b) declares exactly this loop with `reset(seed=1000)`; the number SUMO receives on `--seed` is the env RNG's first draw under it (`envs/base_traffic_env.py:629-631`), deterministic — record both.**
5. **The 206 parity configs exist in the MAIN tree only** (`/home/filip/rltraffic/scenarios/draws/cityflow1x1/draw_NNNN/parity/noteleport.sumocfg`; `parity_sumocfg_path(scenario_key, draw_id, out_root=…)` is a pure path function). Every one binds `cf_parity` to every vehicle and carries `time-to-teleport −1`, verified from the running engine on draw 1 and from all 206 files (`docs/reviews/P7.2a-halfB.md`). **Draws 201–300 reproduce P4.3's recorded probe returns 100/100 in two independent runs** (`docs/returns/P7.2a.md` §9.4; the coordinator's `coord_band_gate.json`), so Rule B's two halves will be measured on the same demand.
6. **P4.3's source-domain statistics, READ from `docs/data/p4_3_probe.json:budgets[k].statistics`:** mean **−18692.4 / −18186.65 / −18600.59** and max **−15495.0 / −12532.0 / −12532.0** at k = 5 / 20 / 100; `rule_b_target(best_source_return, probe_source_stat, probe_target_stat)` forms the ratio first (`:480-505`, pinned by a test); `source_domain_ratio` is `best / S(source)` (`:506-527`). **Both subjects' training draws for the disjointness assertion:** P4's `docs/data/p4_training.json:training_draw_ids` (1–200); `mix50`'s `docs/data/p4_7_declaration.json:tiers.mix50.training_draws` (152 ids, all ≤ 200). Held-out pool 1000–1099. Draws 201–300 are disjoint from both — assert it, do not assume it.
7. **What a SUMO episode costs at reference speed:** 10.4–11.8 s bare (`docs/data/p7_1_metric_freeze.json:timing`, `n = 5` per arm), 17–19 s under P7.1's full observer. **On a battery-throttled laptop the same episode took 93.5 s** (2026-09-13) — hence the canary.

---

## 1. Why this task exists

`PREREGISTRATION` §10 registers the paper's one method contribution as *RTG calibrated in the target domain from a probe policy run there*. A17 fixed every degree of freedom a referee could suspect — the rule (Rule B, `S = mean`), the probe (MaxPressure, SUMO, draws 201–300, seed, settings, two-route return), the subjects (`mappo1000`, `mix50`), the H3 arm (Rule B mean, k = 100), the exploratory status of calibrated-vs-naive — **before any target-domain probe return existed**. P7.2a produced the demand. This task produces the numbers A17 is a function of: the SUMO probe returns per draw, the four targets per subject (Rule B mean — the registered prompt; Rule B max, Rule A `q = 1.0`, naive — the ablations), the in-support diagnostic of each target against each checkpoint's training range, and a fenced two-episode proof that a CityFlow-trained DT runs on SUMO through `align_info` with the RTG advancing. **It evaluates nothing.** The zero-shot point, the calibrated-vs-naive contrast and every ρ are P7.3's, on the held-out pool.

---

## 2. Scope fence — what NOT to build

- **No evaluation on held-out draws 1000–1099, no ρ, no ATT of any DT reported anywhere.** The two smoke episodes run on **draw 5** (training pool, not in the probe band, not the nominal control), one per subject, seed-101 checkpoint, and the packet and the artifact record their **mechanics only** (decisions, RTG trajectory endpoints, actions in range, engine-read type and regime, exceptions) — **never their ATT or return**. A smoke that prints an ATT is a zero-shot number seen before P7.3's brief is written.
- **No change to `agent/DTAgent.py`, `offline/dt_gate.py`, `offline/rtg_calibration.py`, `offline/backend_alignment.py`, `offline/sumo_att_reference.py`.** Import and reuse; if one of them lacks what you need, that is a plan-gate question, not a patch.
- **No `S` other than `mean` as the primary; `max` is computed and reported. No new quantiles, no grid, no target chosen by any outcome.** Rule A's secondary quantiles {0.5, 0.75, 0.9} are computed as values only, as P4.3 did.
- **No multi-intersection anything** (`DEFERRED` 78), no other scenario, no `collect.py` wiring (P7.3's brief), no few-shot data logging (the probe records returns, not trajectories; A17(f) makes P7.3 reproduce these returns bit-for-bit when it logs).
- **No writes under `scenarios/draws/`**, and nothing is copied out of it.

---

## 3. Per-file requirements

### 3.1 `offline/aligned_env.py` (new) — the only door a SUMO observation walks through

`AlignedEnv(env, alignment: ScenarioAlignment)`: a thin wrapper that delegates `reset`, `step`, `close`, `max_steps`, `intersections`, `action_space` (and any attribute read through `__getattr__` — state which ones you forward explicitly and why) to the wrapped SUMO env and returns `align_info(info, alignment)` from `reset` and from `step` — **and nothing else changes**: the reward tuple, `terminated`/`truncated`, the step counter are the env's. It refuses at construction a wrapped env that is not a SUMO env (backend recorded on the env or the config path's suffix — say which) and refuses to wrap an `AlignedEnv` (double alignment is `align_info`'s own refusal; surface it at construction rather than at the first `reset`). `aligned_sumo_env_for_draw(scenario_key, draw_id, *, out_root, arm_args)`: builds the env through `collect_style_args("sumo", arm, parity_sumocfg_path(...), sentinel_out_dir=…)` → `_build_env_spec` → `make_env`, then wraps it with `alignment_for_scenario("hangzhou_1x1_bc-tyc", cityflow_roadnet=…, sumo_net=…)`. Docstring: format none (no on-disk artifact); the alignment convention is A16's and is cited, not restated.

### 3.2 `offline/transfer_calibration.py` (new) — A17(a)–(b) and (d)'s inputs, executed

**Artifact format version `p7.2b-calibration/1.0`**, one file `docs/data/p7_2b_calibration.json` written by `report`, from per-draw chunks under `output/p7_2b/`.

1. **The SUMO probe** — `run_sumo_probe(draw_ids, *, out_root, engine_seed=1000)`: for each draw a fresh **unwrapped** `SumoEnv` (the probe is MaxPressure; it needs no alignment, and the env's own `info` is what `episode_return_two_routes` reads), built exactly as §0.1, `env.reset(seed=engine_seed)`; `MaxPressureAgent(env)` from `algorithms.max_pressure` (the class `run_probe` uses); 360 decisions; per-step `info` kept; the return by both routes under `==`; **per episode, from the running engine:** every vehicle present at the horizon has type `cf_parity` (`getTypeID`), `simulation.getOption("time-to-teleport") == "-1"`, `n_teleports == 0` counted per step from `getStartingTeleportIDList`, and the engine seed SUMO received (`env._engine_seed`). A single failure of any of these refuses the draw and the run. Record per draw: `draw_id`, `local_return` (float64), `local_return_from_lanes`, `att_horizon`, `horizon_vehicle_count`, `decisions`, `engine_seed_drawn`, `n_teleports`, `vehicle_types_seen`, `seconds` and the canary value the driver measured before the run. **`assert_probe_draws_disjoint` (reused) runs against BOTH subjects' training-draw sets and the held-out pool before the first episode.**
2. **The statistics and the targets** — for `k ∈ (5, 20, 100)` (reuse `PROBE_K_VALUES`, `probe_draw_ids`), `S ∈ {mean, max}` (reuse `probe_statistic`): `S(R_probe_sumo, k)` from this run; `S(R_probe_cityflow, k)` READ from `docs/data/p4_3_probe.json:budgets[k].statistics[S]` (never recomputed, never retyped); for each subject `R_best_source` and `rtg_scale` READ from the seed-101 checkpoint payload and asserted equal across the five seeds; `rule_b_target(best_source_return=R_best, probe_source_stat=S_cf, probe_target_stat=S_sumo)` (reused) for both `S`; `rule_a_target(R_probe_sumo, q)` for `q ∈ {1.0, 0.5, 0.75, 0.9}`; the naive target `= R_best`. **The registered prompt per subject is Rule B, `S = mean`, k = 100** — the artifact names it as `registered_prompt` and every other target as `ablation`. The in-domain identity is asserted as a test, not a claim: with `probe_target_stat == probe_source_stat`, `rule_b_target` returns `R_best` bit-for-bit.
3. **The in-support diagnostic** — for each subject and each target: the target's position relative to the checkpoint's training RTG range (`training_rtg_range`, reused) — below / inside / above, and the margin — **reported, never selecting** (A8; `BRIEF_15` §12.1). No rollout is needed for this: it is arithmetic on the target and the range.
4. **The smoke** — `smoke(subject, draw_id=5)`: `aligned_sumo_env_for_draw` → `agent_with_target(aligned_env, seed-101 checkpoint, declared_gradient_steps=40000, target_rtg=<Rule B mean, k=100>)` → `horizon_rollout(aligned_env, choose, episodes=1, seed=1000)` where `choose` records `agent.current_rtg()` per decision as P4.3's `evaluate_point` does. **Recorded: `decisions == 360`, `rtg_first == target`, `rtg_last`, `n_decisions_in_support`, every action within `[0, 8)`, the engine-read type and regime, `seconds`. NOT recorded, NOT printed, NOT in the packet: `att_horizon`, `episode_reward`, anything ρ-shaped.** The chunk stores the rollout's ATT under a key named `fenced_do_not_report` so that P7.3 can later check the smoke against its own first episode, and `report` refuses to copy that key into `docs/data/`.
5. **`report`** — writes `docs/data/p7_2b_calibration.json` from the chunks: `format_version`, `registered_in: "PREREGISTRATION A17"`, the probe table (100 rows), the statistics per k and S (SUMO measured, CityFlow read, with the artifact path and sha256 it was read from), the targets per subject × rule × S × k with `registered_prompt` marked, the in-support diagnostic, the disjointness record, the smoke mechanics, `what_this_does_not_say` (it evaluates nothing; both pairs will be reported by P7.3; the contrast is exploratory per §2). Byte-identical on re-run; refuses a missing or duplicated draw, a probe episode whose two routes disagree, a non-zero teleport count, a chunk from a different `format_version`, and any statistic whose read-from artifact sha differs from the committed one. **Filesystem-mutation barrier:** every refusal precedes every write.

### 3.3 `offline/campaigns/p7_2b_calibration.sh` (new) — the driver

`BRIEF_34` Amendment D4's shape: token consumed on start (`output/p7_2b/AUTHORISED_TO_RUN`, written by the author), start lock on `pgrep -f 'python.*offline\.transfer_calibration'`, `trap` killing the process group, **the canary first** (`run_probe` on CityFlow draw 0 through the committed settings; refuses to start if > 2.0 s and prints the value), then stages: probe (resumable per draw: a complete chunk is skipped only if its two routes agree, its teleports are 0 and its `format_version` matches — a stored verdict is not evidence), smoke ×2, `report`, `output/SHA256SUMS_p7_2b.txt` last and atomic. Logs appended, never truncated. Nothing under `scenarios/draws/` is written.

---

## 4. Tests — write them first, red for their own reasons, each named mutation executed and pasted

`tests/test_aligned_env.py`, `tests/test_transfer_calibration.py` (new). SUMO-gated tests carry the same skip predicate as `tests/test_materialise_parity.py`; checkpoint-gated tests skip when `output/p4_dt/dt_seed101.pt` is absent (CI has no `output/`). **At most four SUMO episodes in the suite.**

- **T1 (load-bearing, SUMO) — the wrapper is A16's door and nothing else.** On draw 5's parity config: raw `reset` info has `state` width 32; the wrapped one 25; `current_phase` maps by `2k → k+1`; `avail_actions` unchanged; `reward`, `step`, `average_travel_time` and every global scalar byte-identical between raw and wrapped; after 20 steps the wrapped per-step `reward` equals `−Σ lane_waiting` over the intersection's incoming SUMO lanes. Wrapping an `AlignedEnv` raises at construction. *Mutation:* make the wrapper return the raw info from `step` → the width assertion fails.
- **T2 (load-bearing, SUMO + checkpoint) — the DT runs on SUMO only through the wrapper, and the target took effect.** `agent_with_target(aligned_env, dt_seed101.pt, 40000, target)` succeeds; 20 decisions through `horizon_rollout`-shaped stepping; every action in `[0, 8)`; `agent.current_rtg()` after the first decision equals `target − reward_0`; the same construction on the **unwrapped** env raises on the first `act` (state width 32 against the checkpoint's 25). *Mutation:* apply the target before `load()` → the resting-RTG assertion in `agent_with_target` fires (this is P4.3's own guard; confirm it, do not re-prove it).
- **T3 (load-bearing, SUMO) — the probe's two routes agree and the engine reads what the file requested.** One probe episode on draw 5: `local_return == local_return_from_lanes` under `==`; `n_teleports == 0`; type set `{"cf_parity"}`; `getOption("time-to-teleport") == "-1"`; 360 decisions; `engine_seed_drawn` is an int. *Mutation:* point the config at the parent's unbound `routes.rou.xml` → the type assertion fails.
- **T4 — Rule B is reused, not reimplemented, and the arithmetic is right.** `rule_b_target` with equal statistics returns `R_best` bit-for-bit for both subjects (read from the checkpoint payloads); with `probe_target_stat = 2 × probe_source_stat` returns `2 × R_best`; the artifact's `source statistics` equal `p4_3_probe.json:budgets[k].statistics[S]` for all six (k, S) under `==`; the registered prompt is the `mean`/k=100 Rule B target and is labelled so. *Mutation:* swap the two statistics in the call → the k=100 mean target changes and the test fails.
- **T5 — disjointness against both subjects.** The assertion passes for 201–300 against P4's 1–200, `mix50`'s 152 draws and 1000–1099; planting draw 150 (in both training sets) or 1000 (held-out) refuses, naming the band.
- **T6 — the k prefixes are nested and the SUMO statistics are prefix statistics.** `probe_draw_ids(5) ⊂ probe_draw_ids(20) ⊂ probe_draw_ids(100)`; the k=5 mean equals the mean of the first five recorded returns (recomputed from the rows by `sum/len`, not `numpy.mean`).
- **T7 — the in-support diagnostic is arithmetic and reports both subjects' ranges.** Targets below/inside/above a synthetic range classify correctly; the real ranges are read from the payloads' `stats`.
- **T8 — `report` regenerates byte-identically; refuses a missing draw, a two-route disagreement, a non-zero teleport count, a wrong `format_version`, a source-statistics sha mismatch; and the smoke's `fenced_do_not_report` key is NOT in the committed artifact** (assert on the written JSON; mutation: copy it through → the test fails).
- **T9 — the driver's canary gate** (shell-level, cheap): a fake canary time of 3.0 s makes the driver exit non-zero before the token is consumed; a real run's canary value lands in every chunk.

`scripts/check_test_hygiene.sh` on both files. Test count goes up; record the totals.

---

## 5. Gates, in order

1. **Plan gate** — `docs/plans/p7.2b.md`: the wrapper's forwarded attributes and why; how the probe env is built (§0.1's route, quoted); the chunk and artifact schemas; the resume rule; the exact list of files read under `scenarios/draws/` and the statement that none is written; the question list. **Quote A17(b) and confirm each of its clauses maps to a line of your design.**
2. **Tests red → green** in the worktree (SUMO-gated ones run — the machine is at speed; paste the canary value first).
3. **Pre-flight, ≤ 10 min, destruction paths and resume only** (the driver, the chunk-reuse rule, the `report` refusals; can anything under `scenarios/draws/` or a merged `output/` column be touched?). Coordinator spawns it on the green commit. CLEAR logged before the token.
4. **The run**, `tmux` foreground, from the worktree (`bash offline/campaigns/p7_2b_calibration.sh`), token written by the author after CLEAR; **canary printed first**; ≈ 25 min at reference speed.
5. **Packet** `docs/returns/P7.2b.md` — the probe table's summary (means and maxes per k, the six source statistics read), the eight targets per subject with the registered one marked, the in-support positions, the smoke's mechanics (**no ATT**), the canary, both clocks per stage, the amendments it was written against.

---

## 6. Definition of Done

- [ ] `offline/aligned_env.py`, `offline/transfer_calibration.py`, `offline/campaigns/p7_2b_calibration.sh`; T1–T9 red-first then green; every mutation pasted; zero frozen files; zero new dependencies.
- [ ] `docs/data/p7_2b_calibration.json` committed, regenerating byte-identically; `output/SHA256SUMS_p7_2b.txt` written last; the smoke's fenced key absent from the committed artifact.
- [ ] 100/100 probe episodes with two-route equality, 0 teleports, `cf_parity` from the engine, disjointness asserted against both subjects.
- [ ] The eight targets per subject computed from READ inputs (checkpoint payloads; `p4_3_probe.json` statistics with its sha recorded), the registered prompt marked; the in-support diagnostic reported for both subjects.
- [ ] The canary value beside every timing; no rate quoted from a run whose canary exceeded 2.0 s.
- [ ] Packet with the AI-assistance record and *what P7.3 will assume* (§7), written against this brief + its amendments.

---

## 7. What P7.3 will assume about this task — and what it must build itself

- `docs/data/p7_2b_calibration.json:targets[subject].registered_prompt` is the number the H3 zero-shot arm conditions on, per subject, and `ablation` carries the other three (Rule B max, Rule A q1.0, naive). P7.3 reads them; it does not recompute them.
- `probe[draw_id].local_return` (float64) is what P7.3's logged k-shot episode on the same draw must reproduce bit-for-bit (A17(f)); `engine_seed_drawn` is recorded so the logger can assert the same seed.
- `AlignedEnv` is the only way a DT sees SUMO; P7.3 wraps every evaluation env with it and generalises `dt_gate.evaluate_arm` with an env factory (a `BRIEF_37` item, not this one), keeping the CityFlow path byte-identical (P4's artifacts must regenerate).
- The held-out SUMO evaluation pool is `scenarios/draws/cityflow1x1/draw_1000–1099/parity/noteleport.sumocfg`, untouched by this task; its regeneration gate is `DEFERRED` 80.

---

## 8. Return Packet

`docs/returns/TEMPLATE.md`, plus: the canary and both clocks per stage; the per-k probe means and maxes beside the six source statistics; the eight targets per subject with the registered one marked and each target's in-support position; the smoke's mechanics with an explicit line *ATT deliberately not reported*; the amendments the packet was written against, by letter.

---

# ✅ AMENDMENT A — 2026-09-14, at the plan gate: PLAN APPROVED (`docs/plans/p7.2b.md` @ `d4a3a3a`), with seven rulings

Read from the worktree. The canary (0.92 s, mains power, `−32648.0 / 247.75089149261333`, two routes agree) and the 10.85 s/episode measurement on the fenced draw 5 are accepted as the session's rate basis. The A17(b) clause table (§2) is complete. **Both conflicts that touch the registration were re-verified by the coordinator from the checkpoint payloads and the declaration** (§9.1: `mix50` `stats.rtg` `[−40294, −6]` over 216,000 rows against `rtg_scale 40223 = −training_return_min` over 72,000; `mappo1000` coincides at 9991; §9.2: both checkpoints' `provenance.training_draw_ids` are `1..200`, the declaration's `mix50` list is 152 ids in `[2, 199]`; 201–300 is disjoint from the union). The read-order error (§9.3) is the coordinator's and is logged.

## A1 — Q1: report the REGISTERED range under its true name, with the training-set bound beside it; A17 is not amended
`training_rtg_range` returns the checkpoint's `stats.rtg` range — the **split** range, `[−40294, −6]` over 216,000 rows for `mix50` — and that is the number A17(c) registered, so it is the range the in-support diagnostic is computed against. Name the field `support_range_over_the_split` with `n_rows` beside it, and add `training_set_return_min = −rtg_scale` (= the declaration's `training_return_min`, −40223 for `mix50`, −9991 for `mappo1000`, where the two coincide) as a disclosed second bound, with one sentence: *the two differ by 71 on a 40,000 scale for `mix50`; the diagnostic never selects and no claim rests on which bound is used.* No second diagnostic. A17's row is not edited (registration rows never are); the packet and the artifact carry the disclosure and the Decisions Log records it.

## A2 — Q2: CONFIRMED, the plain env. The read order's parenthetical was wrong and it was the coordinator's
`make_observer_sumo_env` counts teleports and has no vehicle-type read; the probe takes both from the engine directly per §3.2.1 (`getStartingTeleportIDList` per step, `getTypeID` for every present vehicle, `getOption("time-to-teleport")` once). ~13 min saved across the band is a consequence, not the reason.

## A3 — Q3: YES — fence `episode_reward`, AND `rtg_last`, AND the per-decision RTG series
`rtg_last = target − Σ rewards` **is** the episode return in disguise; recording it would leak the number the fence exists to hold. The smoke records, and the packet reports, **mechanics only**: `decisions == 360`, `rtg_first == target`, `rtg_advanced_every_decision` (bool: strictly changed on every decision whose reward was non-zero, unchanged otherwise), `n_decisions_in_support` (a count against the registered range), `actions_in_range` (bool), the engine-read type set and regime, `seconds`. Under `fenced_do_not_report`: `att_horizon`, `episode_reward`, `rtg_last`, the RTG series. `report` refuses to emit the key; a test asserts the committed artifact carries none of the four.

## A4 — Q4: CONFIRMED — the smoke runs after the probe, its prompt from the same function `report` uses
The brief's "probe, smoke ×2, report" was an enumeration, not an order; §3.3's order is right.

## A5 — Q5: BOTH — a module constant AND the chunks' record
`P4_3_PROBE_SHA256` is pinned in `offline/transfer_calibration.py` as the committed artifact's digest (compute it once, paste it, and a test asserts the file on disk matches); every chunk records the sha it read; `report` refuses if the file, the constant or any chunk disagree. The constant is the declaration — *these statistics came from THAT artifact* — and the chunk is the evidence. P4.3's artifact regenerates byte-identically (`docs/returns/P4.3.md` §16.0), so the constant moves only in a commit that also changes the artifact, which is a reviewed act.

## A6 — Q6: CONFIRMED — `isinstance(env, envs.sumo_env.SumoEnv)`, imported lazily
The class determines the info's shape; a suffix would accept the one case that matters and reject nothing. Refusing a wrapped `AlignedEnv` at construction is also accepted.

## A7 — Two rulings the plan did not ask for
1. **Disjointness against the UNION.** `assert_probe_draws_disjoint` runs against `provenance.training_draw_ids` of BOTH checkpoints ∪ the declaration's 152 ∪ the held-out pool. The superset is the safe direction; the artifact records all three sources and their sizes, and §9.2's trap in one sentence.
2. **"The §7 canary" means `PROJECT_PLAN` §7's rule** (*THE MACHINE-HEALTH CANARY*, added 2026-09-13); the operative recipe is `BRIEF_36` §3.3. Both briefs are corrected by this amendment rather than edited in place: read every "§7 canary" as *the `PROJECT_PLAN` §7 rule, recipe in `BRIEF_36` §3.3*.

**Proceed to gate 2.** T1's non-zero clause and the fence-first ordering in §11 are accepted as written. The pre-flight (gate 3, ≤ 10 min, destruction and resume paths) is spawned on the green commit. The packet is written against `BRIEF_36` + Amendment A.

---

# ⛔ AMENDMENT B — 2026-09-14, on the pre-flight: CLEAR WITH CONDITIONS. Two fixes and one control, then the coordinator re-runs the no-token driver, then the token

`docs/reviews/P7.2b-preflight.md` — **0 blocking, 0 major, 3 minor.** Everything that can destroy or misread state was exercised and held: six planted corruptions plus a lying verdict all rejected by `chunk_is_reusable`; the planted chunk moved to `failed/` byte-identically with a pre-existing sentinel untouched; a fresh chunk rolled with the same drawn seed and the same return; `report` refuses a missing, duplicated, mismatched or corrupt draw and writes nothing; the fence holds against a fenced block and a stray key. **The two items the reviewer's permission layer stopped it from running, the coordinator ran:** no token → canary 0.77 s → refusal, exit 2, no `output/p7_2b/` in either tree; the canary writes nothing under `scenarios/draws/`.

## B1 — REQUIRED (minor 2): install the trap BEFORE the token is consumed
`p7_2b_calibration.sh` deletes the token at `:115` and installs `trap on_signal INT TERM` at `:137`. A signal in that window consumes the one-shot authorisation and leaves neither `FAILED` nor `COMPLETE`. Move the `trap` (and the `fail`/`on_signal` definitions it needs) above the token block, so that from the first destructive line onward a signal writes `FAILED`. Verify by reading (line numbers in the packet); no test is required for a shell reorder, but state the new order in the driver's header.

## B2 — REQUIRED (minor 1): a syntactically valid non-object chunk is moved aside like any other unclean chunk
`run_sumo_probe` `:287-295` catches `JSONDecodeError` and `chunk_is_reusable` catches `(KeyError, TypeError, ValueError)`; a chunk that parses to a list or a scalar raises `AttributeError` on `payload.get` and the probe dies with a traceback, leaving the file in place. Check `isinstance(payload, Mapping)` first (returning *not reusable*), so the existing move-aside path handles it. **Test:** plant `[]` as a chunk → moved to `failed/`, re-rolled; the test needs no simulator if the roll is monkeypatched, or costs one SUMO episode if not — your choice, stated.

## B3 — REQUIRED (minor 3, as a control rather than a comment): refuse to start unless the script leads its process group
`kill -- -$$` is a no-op unless the driver is its own group leader, which `tmux` foreground gives and a stray `bash script.sh &` does not. Add, before the token: `[ "$(ps -o pgid= -p $$ | tr -d ' ')" = "$$" ] || { echo "REFUSING TO START: not a process-group leader; run in a tmux foreground pane" >&2; exit 2; }`. Fail-closed, one line, and it turns the operator condition of `BRIEF_34` E2 into something the driver checks.

## B4 — Recorded, no action
- The canary runs a CityFlow episode before the token; it writes nothing (`saveReplay: false` in every draw config; the draws tree is 1,442 files with none newer after four canaries and a 100-episode band gate).
- `run_smoke` was not exercised by the pre-flight (its third SUMO episode was not spent); its fence is exercised through `report`'s whitelist, which is where the fence is enforced.

## B5 — Sequence to CLEAR and the token
1. `git -C /home/filip/rltraffic-p53b merge --no-edit main`; B1–B3; the B2 test; the three test files and hygiene; commit; report the sha. **No token.**
2. The coordinator reads the diff, runs the no-token driver once more on the fixed commit (it must now refuse at the group-leader check when run under `timeout`, and at the token when run in a foreground pane — record both), and logs CLEAR.
3. Then the author writes the token: `mkdir -p /home/filip/rltraffic/output/p7_2b && date -Is > /home/filip/rltraffic/output/p7_2b/AUTHORISED_TO_RUN`, and the implementer launches `bash offline/campaigns/p7_2b_calibration.sh` from the worktree in a **tmux foreground pane**, from a committed tree, on mains power. ≈ 20 min. Then the packet, written against `BRIEF_36` + Amendments A–B.

---

# ✅ AMENDMENT C — 2026-09-15: CLEAR on `d284e47`. The token may be written; the run may start

**Verified by the coordinator by execution, not from the relay.** The driver's diff read (the group-leader check before the inputs, `fail()`/`on_signal()` and the `trap` above the token, both `FAILED` writers guarded on `$WORK` existing so a signal before the token still creates nothing; `chunk_is_reusable` returns *not reusable* for any non-Mapping payload and the existing move-aside path handles it; the two new tests and the parametrised sibling). Then the no-token driver run twice on the fixed commit: **under `timeout` (not a group leader) — exit 2 at `REFUSING TO START: not a process-group leader`, nothing else run; under `setsid` (a group leader) — canary 0.80 s, then `REFUSING TO START: no run authorisation token`, exit; no detached process; `output/p7_2b/` absent in both trees; the draws tree at 1,442 files.** The two P7.2b test files: **32 passed**; hygiene, English and `bash -n` clean. One thing seen and explained rather than ignored: `output/replay.txt` in the worktree is a CityFlow test artifact written by the implementer's whole-suite run at 22:11 on 2026-09-14 (an `aigen_1x1` replay; the main tree's copy dates from P7.1) — not the canary's, which writes nothing (`saveReplay: false`; no file under either `output/` is newer than the fix commit).

## C1 — The token, written by the author, once
```bash
mkdir -p /home/filip/rltraffic/output/p7_2b && date -Is > /home/filip/rltraffic/output/p7_2b/AUTHORISED_TO_RUN
```
The driver deletes it on start; it authorises exactly one run.

## C2 — The launch, by the implementer
From a **tmux foreground pane**, on mains power, from the committed worktree (`git -C /home/filip/rltraffic-p53b status --porcelain` empty — every chunk records `git_dirty`):
```bash
cd /home/filip/rltraffic-p53b && bash offline/campaigns/p7_2b_calibration.sh
```
Expected: canary ≈ 0.9 s printed first, then the probe (100 draws, ≈ 18 min), the two smoke episodes, `report`, `output/SHA256SUMS_p7_2b.txt`, `COMPLETE`. **If the canary reads above 2.0 s the driver refuses and the token is untouched** — check the power source and try again. If anything refuses mid-run, `FAILED` names the stage; report it, do not restart until the coordinator has read the log.

## C3 — Then the packet, written against `BRIEF_36` + Amendments A–C
With: the canary and both clocks per stage; the per-k probe means and maxes beside the six source statistics; the eight targets per subject with the registered one marked and each target's in-support position (both bounds, A1); the smoke's mechanics with the explicit line *ATT deliberately not reported*; the disjointness record's three sources; `git_commit` uniform across chunks. Then the merge review.

---

# ⛔ AMENDMENT D — 2026-09-15, on the completed campaign, BEFORE the packet: two fixes and a smoke re-roll under a fresh token

The campaign ran as projected (probe 12:22:57 → 100 chunks by 12:41, smokes 12:41:38 / 12:41:49, `report` 12:42:02, `COMPLETE` 12:42:07; canary 0.89 s in every chunk; 100/100 two-route agreement, 0 teleports, `cf_parity` everywhere, one drawn engine seed; `att_horizon` occurs only in the 100 probe rows; the P4.3 sha matches). **The first target-domain numbers exist and are consistent to the digit with the registered rule:** `S = mean` SUMO/CityFlow at k = 100 is 23195.39 / 18600.59 = 1.2470, so the registered prompts are **−7185.35** (`mappo1000`) and **−7431.02** (`mix50`), both inside their support ranges; Rule A `q = 1.0` (−20809.0) falls **below** `mappo1000`'s support by 10,818, exactly as P4.3 predicted for the probe-quantile rule. The coordinator read all of this from the artifact and the chunks; two things in the smoke block are wrong and must be fixed before a packet is written around them.

## D1 — REQUIRED: the smoke's `rtg_advanced_every_decision` is off by one in the DIAGNOSTIC, not in the agent
`run_smoke` records `agent.current_rtg()` **before** `agent.act(info_t)`, and `DTAgent.act` updates `reward_sum` **inside** the call — so `rtg[t] − rtg[t−1] == −r(info_{t−1})`, the reward carried by the *previous* info. The check at `transfer_calibration.py:~851` compares the change with `rewards_in_info[index]` (the current info's reward) and therefore returns `False` as soon as consecutive rewards differ. **Verified by the coordinator on a 40-decision replay of the `mappo1000` smoke on draw 5: the agent's delta equals `−r(info_{t−1})` exactly on 39/39 decisions; the implementer's alignment is consistent on 38/39.** Fix: compare with `rewards_in_info[index − 1]` (and say in the docstring why the shift exists); add a unit test on a synthetic series — rewards `[0, 0, −2, −4]` → the flag is `True` only under the shifted alignment. Both smoke chunks on disk carry the wrong `False` and must be **re-rolled** (D3).

## D2 — REQUIRED: `report` lifts the canary into the artifact
§3.2.5 asked for `canary (seconds, threshold, verdict)` in `docs/data/p7_2b_calibration.json`; every chunk carries `canary_seconds 0.89` and the artifact carries nothing. `report` records the set of distinct `canary_seconds` across the chunks (refusing if more than one value — one campaign, one canary), the 2.0 s threshold, and the verdict.

## D3 — The re-roll, under a fresh token, through the driver
1. Merge `main`; implement D1 and D2 with their tests; run the two P7.2b test files and hygiene; commit. **Nothing under `output/p7_2b/` is edited by hand.**
2. The coordinator reads the diff and confirms the D1 test kills the unshifted comparison (a one-line mutation).
3. The author writes a fresh token (`mkdir -p /home/filip/rltraffic/output/p7_2b && date -Is > /home/filip/rltraffic/output/p7_2b/AUTHORISED_TO_RUN`); the implementer removes ONLY `output/p7_2b/smoke_mappo1000.json`, `smoke_mix50.json` and `COMPLETE` (state the three names in the packet; the probe chunks stay and are reused by content — the driver's own rule), then restarts the driver in a tmux foreground pane. Expected: canary; 100 × `reused`; two smokes (≈ 20 s); `report` rewritten byte-identically **except** the smoke block and the new canary field — paste the diff of the two artifact versions; the manifest rewritten.
4. The packet, written against `BRIEF_36` + Amendments A–D, with the smoke's mechanics now reading `rtg_advanced_every_decision: True` (or, if it does not, the packet says so and stops — that would be a real finding).

Nothing in the probe table, the statistics or the targets changes; they are already consistent with A17 to the digit, and the coordinator will say so in the merge review's mandate rather than have them recomputed twice.
