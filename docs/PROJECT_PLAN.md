# PROJECT MASTER PLAN — Offline Multi-Agent Decision Transformer for Traffic Signal Control

**Version:** 0.5 · **Last updated:** 2026-07-27 · **Maintained in:** Master Coordination Chat
**Mentor:** Paweł Gora (Quantum AI Foundation) · **Target:** arXiv → IEEE ITSC / IEEE T-ITS / TRB (Q2/Q1)

---

## 1. Headline Contribution (FROZEN — do not renegotiate in sub-chats)

> **"Offline Multi-Agent Decision Transformer (MADT) for traffic signal control, trained purely from logged trajectories, evaluated on a rigorous multi-simulator platform — with cross-backend transfer (CityFlow → SUMO) and rare-event robustness as the headline evaluation axes."**

Three claims, each independently defensible (hardened 2026-07-10 after external critique review):

| # | Claim | Why it survives review |
|---|-------|------------------------|
| C1 | **Data-quality study:** MADT performance as a function of a D4RL-style dataset ladder (random / fixed-time / max-pressure / medium-MAPPO / expert-MAPPO / mixed), trained with **no online exploration** from logged operational data. Ladder released publicly. | Fixes the circularity trap (expert data → trivial imitation; weak data → probably fails). Positioned vs DTLight's DTRL (2 policies, SUMO-only, independent agents): ours has 6 tiers, multi-agent local rewards, **dual-backend paired**. "No online exploration" replaces the rhetorically weak "zero simulator interaction." **Load-bearing.** |
| C2 | **Pre-registered RQ:** does offline sequence modeling degrade more gracefully under scenario shift (demand shocks, incidents, sensor dropout)? Answered via a **2×2 design** ({nominal, shift-augmented} training data × {MADT, MAPPO}) + mechanism ablations (context length, data diversity). | The 2×2 with domain-randomized MAPPO removes the data/architecture confound (the single most likely rejection reason). Known DT risk under shift (ESPER/infeasible-RTG) is addressed by calibrated prompting — and plausibly explains DataLight's negative result. Negative/mixed answer still publishes. **Load-bearing.** |
| C3 | **Dynamics-shift study:** transfer *curve* of CityFlow-trained MADT on SUMO — zero-shot → few-shot (k ∈ {5, 20, 100} episodes, collected by a MaxPressure probe, spanning scenario variants) → full retrain — with within-backend-normalized metrics (vs fixed-time AND max-pressure anchors) and calibrated RTG prompting. Interface documented precisely; one alternative-state-encoding robustness ablation. | No systematic cross-simulator dynamics-shift study for TSC exists (cross-city ≠ cross-backend; X-Light took cross-city). Every curve outcome is publishable. k=1 dropped (statistically meaningless; CityFlow determinism makes same-scenario episodes near-duplicates). Paper stands on C1+C2 if C3 lands negative. |

**Named method component:** *probe-calibrated return prompting* — target RTG expressed relative to a cheap in-domain probe policy (MaxPressure) return distribution (e.g., quantile-space). One mechanism serving both C2 (infeasible-RTG under scenario shift) and C3 (reward-scale shift across backends), with its own ablation. Lifts the paper from "study" toward "method + study."

**Controversy framing (the paper's hook):** the literature openly contradicts itself on whether return-conditioned sequence modeling works for TSC — DataLight (2303.10828) concludes DT *cannot* be applied to TSC; DTLight (2312.07795) and the spatiotemporal MADT (2602.02903) report it working. Our ladder + shift studies + calibration adjudicate *when* it works and *why* it fails.

**Non-negotiable baselines (offline-RL venue standard):** BC and %BC on the expert slice (if BC matches MADT, sequence modeling adds nothing — must be tested), IQL (independent per-intersection), CQL optional; plus online MAPPO/IPPO/DQN, MaxPressure, fixed-time, random; plus **domain-randomized MAPPO** for C2.

**Unified narrative:** *Offline MADT for TSC under two axes of distribution shift — scenario shift (C2) and dynamics shift (C3) — with the data-quality ladder (C1) as the foundation.* Enabled by the repo's ≥7 paired scenarios (4× hangzhou_1x1, cologne1, cologne3, grid4x4 in both CityFlow and SUMO formats; verified 2026-07-09).

**Closest prior art (cite prominently or die):** arXiv:2602.02903 (spatiotemporal MADT, Feb 2026 — architecture overlap, no transfer/ladder); DTLight 2312.07795 + DTRL datasets; DataLight 2303.10828 (negative DT result — our hook); X-Light (cross-city transfer); OffLight; TransformerLight; arXiv:2603.22315 (offline DT for emergency-vehicle corridors, Mar 2026 — overlaps C2's EV perturbation; ours is a robustness study, theirs an EV method); iLLM-TSC; CoLLMLight; LATS. Space is moving monthly — speed on P1–P5 matters.

---

## 2. Hardware & Compute Budget

- **GPU:** NVIDIA RTX 5080 (ample VRAM for DT training; 7–8B local LLM in 4-bit for the optional teacher extension)
- **RAM:** 64 GB — sized for large offline trajectory corpora held in memory during training
- **CPU:** high-end multi-core — the actual bottleneck is simulation; data collection parallelizes across CPU workers (one env per process)
- **Strategy:** simulators are CPU-bound → generate the offline corpus in parallel overnight; DT training is pure supervised learning → GPU-efficient, fast iteration. This asymmetry is *the reason* the offline approach fits our hardware.

---

## 3. Baseline Platform (rltraffic) — verified facts

Repo: `github.com/RusieckiFilip/rltraffic` — the June 2026 UW bachelor's thesis of Bibrowski, Bublik, Pisula & Woliński (sup. G. Grudziński), lineage RESCO → RESCO TensorCell → this framework. Verified by direct code inspection 2026-07-08; thesis PDF reviewed 2026-07-09. Thesis-documented facts: ~3× training speedup vs TensorCell baseline (libsumo in-process + metric caching; ~200k simulator calls/episode on cologne3, only ~2% advance the clock); backend qualitative consistency SUMO↔CityFlow (smoothed-reward corr. 0.93) but **absolute metrics not interchangeable** (CityFlow jumpier, occasional gridlock episodes); IDQN(CNN) most sample-efficient on grid scenarios (spatial inductive bias, Fig. 5.9 ablation); IPPO/MAPPO unconverged at 200 eps on hangzhou_4x4 (Fig. 5.4); thesis experiments use the **PressLight** reward. Our paper cites this as the platform paper; ours is the method paper.

- **Three simulator backends, one API:** CityFlow (vendored, patched for Python 3.12+), SUMO (traci/libsumo), MOSS (GPU). Agents/rewards/metrics/states are backend-neutral.
- **Agents present:** DQN, DQNAgentPFRL (RESCO parity), IPPO, MAPPO (centralized critic + running normalization). Baselines: MaxPressure, random; fixed-time via phase-control modes.
- **Safety semantics:** 4 phase-control modes (acyclic, bounded, cyclic, RESCO-cyclic); enforced yellow/all-red clearance, min/max green; env raises on illegal actions.
- **Scenarios available:** hangzhou_1x1 (×4 variants), hangzhou_4x4_gudang, hangzhou_4x4_hetero, grid4x4, manhattan_28x7, cologne1, cologne3, aigen_1x1, bb5b.
- **Experiment harness:** JSON-config `env × agent × seed` matrix; paired-seed eval; outputs `results.json`, `summary.csv`, plots.
- **Gap we fill:** no offline-RL data path exists — no trajectory logger, no offline dataset format, no sequence-model agent. That is our contribution surface.

### 3.1 P0.2 Baseline anchors (2026-07-09, 3 seeds, 60 train episodes, avg travel time in s)

| Scenario | MaxPressure | Random | MAPPO@60ep |
|---|---|---|---|
| cf_hz1x1 | **160.56 ± 0.00** | 307.53 ± 0.60 | 197.91 ± 1.78 |
| cf_grid4x4 | **141.65 ± 0.00** | 207.26 ± 1.93 | 632.95 ± 51.63 (gridlock, 1040 veh stuck) |

Full tables in `docs/p0_baseline_numbers.md`. Two standing observations:
1. **CityFlow demand is deterministic** (MaxPressure σ=0 across engine seeds) → seed-CIs capture policy stochasticity only; environment stochasticity must come from P6 perturbations / flow randomization. Affects P8 statistics design.
2. **MAPPO@60ep < random on grid4x4** → keep as motivating evidence (online MARL sample-inefficiency at 16 intersections); corroborated independently by thesis Fig. 5.4 (IPPO/MAPPO unconverged at 200 eps on hangzhou_4x4). Reproduce as a "performance vs. interaction budget" motivation figure. P2.1 convergence budget: ≥500 episodes on grid4x4, with explicit convergence verification.

---

## 4. Interface Contracts (FROZEN — paste into every sub-chat brief)

### 4.1 Environment API (non-standard Gym — do not "fix" it)
```python
info = env.reset(seed=42)                                (# returns info ONLY, no obs)
reward, terminated, truncated, info = env.step(action)   # reward FIRST
```
`action` is an `np.ndarray`, one action per intersection, **ordered by `[ix.id for ix in env.intersections]`**.

### 4.2 The `info` dict contract
```python
info = {
    "sim_time": float, "vehicle_count": int, "step": int,
    "average_travel_time": float,
    "lane_vehicle_count": {lane_id: int}, "lane_waiting_vehicle_count": {lane_id: int},
    "metrics": {metric_name: float},                # requested global metrics
    "intersections": {
        ix_id: {
            "state": [...],                          # per-intersection obs vector
            "avail_actions": [0, 2],                 # legal actions NOW (masking!)
            "current_phase": int, "time_in_phase": int,
            "action_applied": bool,
            "metrics": {name: float},
            "reward": float,                         # ONLY if local_reward_fn set
        }, ...
    },
}
```

### 4.3 Agent API (mirror `agent/MAPPOAgent.py`)
```python
class MyAgent(BaseAgent):
    def __init__(self, gym_env, ..., device=None, seed=None)   # reads env.intersections
    def act(self, info, explore=True, update_memory=True) -> np.ndarray
    def observe(self, next_info, reward, terminated, truncated=False) -> dict
    def save(self, path); def load(self, path)
```
Helper functions in `agent/utils/utils.py` (USE THESE, do not reimplement):
`Utils.infer_action_counts(env.action_space, intersections)`, `Utils.extract_per_intersection_info(info, ids)`, `Utils.state_from_info(ix_payload)`, `Utils.extract_valid_actions(ix_payload, n_actions)`, `Utils.scalar_reward(reward)`, `Utils.reward_for_intersection(...)`, `Utils.resolve_device(device)`, `Utils.seed_everything(seed)`.

### 4.4 Coding standards
Python ≥3.12, PyTorch, type hints (`from __future__ import annotations`), numpy `float32` for stored arrays, docstrings in the repo's style, no new heavy deps without a Decisions-Log entry, `pytest` tests for everything, no modifications to `envs/base_traffic_env.py` unless a task explicitly says so.

---

## 5. Offline Data Design (agreed v0.1 — refine in P1/P3)

- **Format:** one `.npz` (compressed) per episode + one `manifest.json` per collection run.
- **Per step, per intersection:** `state (float32)`, `action (int64)`, `avail_actions mask (bool, width = n_actions)`, `local_reward (float32, NaN if unset)`, `current_phase`, `time_in_phase`.
- **Per step, global:** `global_reward`, `vehicle_count`, `sim_time`, selected `metrics`.
- **Episode metadata:** scenario id, backend, engine seed, `delta_time`, `max_steps`, `state_features` list, reward fn names, phase-control mode, behavior-policy id + checkpoint hash, repo git hash, timestamp.
- **Dataset ladder (C1's experimental axis; released publicly as a contribution, positioned vs DTRL):** tiers are defined by **measured normalized return** (random = 0, MaxPressure = 100, computed per scenario post-collection), NOT by policy names — our own P0.2 data shows MaxPressure > MAPPO@60ep, so policy names imply a false quality ordering (critique 2026-07-11b). Source policies (= dataset provenance metadata): random, fixed-time, MaxPressure, MAPPO checkpoints (60/200/converged), ε-noised MAPPO (ε ∈ {0.1, 0.3}), DQN(CNN), mixed. The paper's ladder table reports the measured return distribution of each dataset. If no learned policy beats MaxPressure after convergence, the top slice IS MaxPressure — which sharpens, not weakens, the "does MADT exceed its data" question.
- **Demand randomization is mandatory for every collected episode (P2.0):** with deterministic CityFlow flows, N episodes of a deterministic policy = 1 trajectory × N; effective corpus size of deterministic tiers ≈ 0 (critique 2026-07-11a). Each episode = (scenario, flow-draw seed); nominal flow = draw 0. Randomizer operates at the vehicle-list level (verified: flow.json = per-vehicle insertions with routes) so draws render to BOTH backends (flow.json ↔ .rou.xml) — preserves C3's paired scenarios and fixes P8's environment-stochasticity problem at the source. Logger convention: flow-draw id goes into run_metadata (no logger code change needed).

---

## 6. Task Checklist

### Phase P0 — Platform verification  ✅ (P0.3 patch pending)
- [x] Clone + inspect repo structure, docs, agent/env/reward/state contracts *(Master chat, 2026-07-08)*
- [x] **P0.0** Dev environment: WSL2 Ubuntu 24.04 + torch cu128 (capability (12,0) confirmed) + CityFlow build + SUMO + VS Code WSL Remote *(Return Packet 2026-07-09)*
- [x] **P0.1** Smoke test: PASS *(2026-07-09)*
- [x] **P0.2** Baseline numbers recorded (see §3.1) → `docs/p0_baseline_numbers.md` *(2026-07-09; run sequentially due to fork deadlock)*
- [x] **P0.3** Patch `experiments/runner.py`: `ProcessPoolExecutor(..., mp_context=mp.get_context("spawn"))` — fixes fork deadlock with torch/CityFlow C++ locks. Verify: smoke.json with seeds [7,8,9], `--workers 3`
- [ ] **P0.4** Real pre-registration (within 24 h of the reward decision, so the registered protocol is stable): commit `PREREGISTRATION.md` (RQs, 2×2 design, gates, ladder protocol, transfer-curve design) → git tag `v0.1-prereg` with doc hash; optionally mirror to OSF. ~20 min; without a timestamp "pre-registered" reads as rhetoric

### Phase P1 — Trajectory Logger  ✅ MERGED to main 2026-07-27 (`00c9d42`)
- [x] **P1.1** `offline/trajectory_logger.py`: episode recorder wrapping the act/step/observe loop (wrapper/callback pattern, zero env modifications)
- [x] **P1.2** `offline/collect.py`: CLI — behavior policy × scenario × episodes × seeds → dataset dir
- [x] **P1.3** `tests/test_trajectory_logger.py`: shape/roundtrip/determinism tests
- [x] **P1.4** Return Packet reviewed in Master chat; plan updated
- [ ] **P1.4b** Fold into the P3 brief: key by intersection ID (never index); `local_reward` is composite; episode ends are truncations not terminals
- [x] **P1.5** Independent review sub-chat (critical path): diff + brief + contracts → discrepancy report; merge `task/p1-logger` only after pass

### Phase P2 — Offline corpus generation
- [ ] **P2.0** ⚡ `offline/flow_randomizer.py` (BEFORE any collection): seeded demand draws — departure-time jitter, Bernoulli thinning, volume scaling — at the vehicle-list level, rendering to both flow.json and .rou.xml; nominal = draw 0; unit tests incl. dual-backend render equivalence *(Sub-chat Brief #2a)*
- [ ] **P2.0b** ⚡ Compute-budget sheet from MEASURED per-cell wall-clock (P0.2 run logs already contain this per Brief #0 §7): full matrix in sim-hours → freeze headline scope (proposal: hz1x1 + grid4x4 headline; cologne3 appendix; Manhattan out; online IPPO/DQN baselines headline-only) → go/no-go before P2.1
- [ ] **P2.1** Train MAPPO to convergence on headline scenarios (≥500 eps grid4x4, convergence verified); save checkpoints at 60/200/converged; verify whether converged MAPPO actually exceeds MaxPressure (determines the top ladder slice)
- [ ] **P2.2** Collection campaign: source policies × headline scenarios × episodes-with-flow-draws × 5 seeds (parallel CPU workers, spawn ctx)
- [ ] **P2.5** `offline/policies/fixed_time.py`: fixed-time controller (NOT in repo — verified 2026-07-25, zero matches for `fixed.time|FixedTime`). Needs its own design decisions: cycle length, phase split (equal vs scenario-tuned), interaction with phase-control clearance semantics. Registers into `collect.py`'s policy registry. Ladder Tier 1 depends on it
- [ ] **P2.4** `offline/corpus_lint.py`: post-collection validator — manifests, shapes, NaNs, logged-vs-recomputed reward equality, return distributions, episode-hash duplicate detector *(critical path → gets review sub-chat)*
- [ ] **P2.3** Dataset stats notebook: measured normalized-return distribution per dataset (defines tier labels), state coverage, corpus size; **scenario license audit** (Cologne CC terms, Hangzhou/CityFlow provenance) — redistribution rights must be confirmed before the public-release claim stands; **reward-metric correlation latch** (Spearman of queue- vs pressure-returns against avg travel time; confirms or reverses the queue_length decision before P4)

### Phase P3 — Offline dataset loader
- [ ] **P3.1** `offline/dataset.py`: PyTorch `Dataset` — episode files → per-intersection sequences with **returns-to-go**, context-window slicing (K steps), padding + attention masks
- [ ] **P3.2** Normalization stats computed from corpus (frozen at train time)
- [ ] **P3.3** Loader unit tests incl. RTG correctness verified by an INDEPENDENT computation path (`np.cumsum` on raw arrays) against the loader's own — double-computation rule
- [ ] **P3.4** Independent review sub-chat (critical path) before merge

### Phase P4 — Single-intersection Decision Transformer (validation gate)
- [ ] **P4.1** `agent/DTAgent.py`: causal GPT-style DT over (RTG, state, action) tokens; action masking from `avail_actions`; `BaseAgent`-compatible `act()`
- [ ] **P4.2** Train on hangzhou_1x1 corpus; **GATE (re-anchored, critique 2026-07-11c):** DT-offline ≥ MaxPressure **AND** within 5% of the best available online policy on avg-travel-time (5 seeds, CIs) — being within 5% of a weak MAPPO while losing to a 1970s heuristic is not a pass → if failed, diagnose before proceeding
- [ ] **P4.3** RTG-conditioning sweep + **probe-calibrated return prompting** v1 (MaxPressure-relative / quantile-space targets) — the named method component; ablated here first on 1×1
- [ ] **P4.4** Offline baselines on the same corpus: **BC, %BC (top-10% return filter), IQL** (independent per intersection; CQL optional) — non-negotiable comparators; if BC-on-expert matches MADT, report honestly and pivot the story to the ladder/shift findings

### Phase P5 — Multi-Agent DT (the paper's core model)
- [ ] **P5.1** Spatial mixing layer across intersections (graph attention over road-network adjacency from `RoadnetInfo`) interleaved with temporal causal attention
- [ ] **P5.2** Train + evaluate on grid4x4, hangzhou_4x4 **per ladder tier**; compare vs online MAPPO/IPPO/DQN, MaxPressure, fixed-time, random, **and offline BC/%BC/IQL**
- [ ] **P5.3** Ablations: no-spatial-mixing, no-RTG, context-length K

### Phase P6 — OOD / robustness suite (claim C2)
- [ ] **P6.1** Scenario perturbation tools: demand surges, lane/approach closures, sensor dropout (state masking), emergency-vehicle flows (cite/differentiate arXiv:2603.22315)
- [ ] **P6.2** Train the **domain-randomized MAPPO** baseline (same shift-augmented scenario distribution the MADT sees) — removes the data/architecture confound
- [ ] **P6.3** The **2×2 benchmark**: {nominal, shift-augmented} training data × {MADT, MAPPO} × perturbations, 5 seeds, significance tests; mechanism ablations: context length (in-context adaptation) and dataset diversity; calibrated vs naive RTG prompting under shift

### Phase P7 — Dynamics-shift study (claim C3, reframed 2026-07-09)
- [ ] **P7.0** ⚡ EARLY GATE (~1 day, runs right after P1, parallel to P2): log MaxPressure trajectories on one paired hangzhou_1x1 in BOTH backends; compare state-feature distributions (per-feature KS/overlap) and MaxPressure-normalized returns. Pathological shift → descope C3 to a limitations study, minimal sunk cost
- [ ] **P7.1** Feature-space alignment: freeze the backend-neutral feature set (no backend-specific structured states) **+ freeze the normalized transfer metric** (improvement over within-backend MaxPressure; Decisions Log 2026-07-09)
- [ ] **P7.2** RTG calibration protocol for cross-domain prompting (target return relative to in-backend MaxPressure return, or quantile-space prompting) — novel contribution candidate
- [ ] **P7.3** Transfer curve: CityFlow-trained MADT on SUMO at zero-shot → few-shot (**k ∈ {5, 20, 100}**, collected by the MaxPressure probe, spanning scenario variants — same-scenario episodes are near-duplicates under CityFlow determinism) → full retrain anchor; many seeds + dataset draws; small SUMO corpora only (main corpus stays CityFlow)
- [ ] **P7.4** Alternative-state-encoding robustness ablation (waiting-based vs count-based features) — guards the \"transfer gap = interface mismatch\" objection

### Phase P8 — Statistics & reporting hardening
- [ ] **P8.1** ≥5 seeds everywhere, mean ± 95% CI, paired tests vs strongest baseline; extend `experiments/report.py`
- [ ] **P8.2** Compute/latency reporting: train-time, inference ms/decision, param counts (reviewers ask)

### Phase P9 — *(moved)*
LLM-teacher extension relocated to **P11.4** (stretch goals) on 2026-07-13 — it was already optional, and grouping all "only if time remains" items in one place prevents it from silently competing with load-bearing work.

### Phase P10 — Paper & release
- [ ] **P10.0** Dataset release package: datasheet (Gebru et al. template), license (per P2.3 audit), Zenodo DOI, versioned archive + loading code — reviewers check this when a dataset is claimed as a contribution
- [ ] **P10.1** Outline + related-work matrix (vs the 2024–26 papers in §1)
- [ ] **P10.2** Full draft → mentor review → arXiv → venue submission

### Phase P11 — Stretch goals (ONLY if time remains after P10; never blocks the paper)
Candidates parked here deliberately: each is scientifically attractive but competes with the project's top risk (time exhaustion). Revisit only once P10.2 is submitted.
- [ ] **P11.1** **Transfer triangle** (CityFlow ↔ SUMO ↔ MOSS): promote the C3 transfer *curve* into a three-domain study. Prereqs: convert headline scenarios to MOSS format (backend `envs/moss_env.py` exists, but **zero MOSS configs/scenarios in repo** — verified 2026-07-13) + validate conversion fidelity (conversion errors would contaminate the measured dynamics gap) + third evaluation sweep. Payoff: dynamics shift measured across three independent engines instead of two — substantially strengthens the sim-to-real proxy argument
- [ ] **P11.2** **Reverse-direction transfer ablation** (SUMO → CityFlow): is the transfer gap asymmetric? Would indicate which engine's dynamics are "richer." Cost: a second full MADT training run
- [ ] **P11.3** **City-scale demo** (Manhattan 28×7) on MOSS: MOSS's GPU parallelism only pays off at scale (thesis measured ~3× on hangzhou 4×4, far from the paper's 100× claim, because the network is small). Caveat: MOSS competes with DT training for the same RTX 5080 — schedule sim and training runs separately, never concurrently
- [ ] **P11.4** **LLM-teacher extension** (was P9): local Mistral-7B/Llama-3.1-8B (4-bit) for OOD trajectory labeling / perturbation curricula — training-time only, never in the control loop

---

## 7. Sub-chat Protocol (hardened 2026-07-12 — designed for AI-written code)

Threat model: the dominant failure mode of AI-heavy development is *confident code with a silent semantic bug that nobody independently checked*. Every rule below exists to catch that.

**Execution modes.** Coding tasks preferably run in **Claude Code** on the user's terminal inside the repo (reads real files, runs real tests); chat sub-chats remain the fallback. Briefs carry a mode header: in Claude Code the "attach files" list becomes "read these repo paths", and these guardrails apply: work ONLY on a task branch (`task/<id>-<name>`), NO edits to existing repo files unless the brief says so, commit before finishing, Return Packet includes the actual `pytest` output and `git diff --stat`.

**Lifecycle per task:**
1. Master chat issues the **Brief** (self-contained; embeds §4 contracts; Definition of Done; ≤ ~2 source files + tests — larger tasks get split).
2. Implementation sub-chat (Claude Code preferred) writes code + tests on the task branch, ends with the **Return Packet**.
3. **Critical-path tasks get an independent review sub-chat** (fresh context; receives brief + contracts + diff; finds discrepancies ONLY, writes no features; may run in Claude Code read-only). Critical path = anything the paper's data flows through: trajectory logger, flow randomizer, dataset/RTG loader, DT agent, corpus linter, statistics harness. Merge to main only after review passes.
4. User pastes Return Packet(s) back → Master chat updates checkboxes, issues next brief.
5. **Phase-boundary review:** before freezing any phase, Master chat writes down what that phase assumes about later phases (this caught determinism↔corpus and reward↔RTG couplings late; do it early now).

**Stopping rule for review rounds (added 2026-07-25).** A review round is worth running while it can
still find *load-bearing* defects — things that would force rework (a wrong data format, a missing
baseline, a confounded experiment). Once a round returns only style and preference items, stop and ship:
further rounds trade real progress for churn. Planning documents get **at most one** external review
round, then they freeze. From P1 onward the review function moves into the pipeline (P1.5-style
independent review of an actual code diff against a frozen spec), which is bounded by construction —
reviewing code against a fixed contract terminates; reviewing prose against opinions does not.

**Redundancy rules (non-negotiable):**
- **Double computation of critical quantities:** RTG in the loader verified by an independent `np.cumsum` test path; logged rewards vs rewards recomputed from per-lane counts — equality asserted on real episodes; randomizer dual-backend renders checked for equivalence.
- **Corpus linter (P2.4) after every collection run:** manifests, shapes, NaNs, reward-recompute equality, return distributions, **episode-hash duplicate detector** (residual guard on the determinism failure mode).
- **Sanity anchors:** any phase producing numbers compares against §3.1 (P0.2) before results are accepted.

```markdown
## RETURN PACKET — [Task ID]
**Status:** DONE / PARTIAL / BLOCKED
**Branch + diff stat:** (task branch name; `git diff --stat` vs main)
**Files produced:** (paths + one-line description each)
**Tests:** (actual pytest output summary; run locally/in Claude Code)
**Self-review checklist:** contracts honored? / tests green? / zero edits to existing files? / zero new deps? (Y/N each)
**Deviations from brief:** (any, with justification)
**Open questions / risks for Master chat:** (bullets)
```

**Sub-chats may NOT:** change the headline contribution, alter §4 contracts, add heavy dependencies, or modify `envs/base_traffic_env.py` — such needs go in the Return Packet as an open question.

---

## 8. Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-08 | Headline = Offline MADT + cross-backend transfer (Track B / H3-core) | Least saturated axis; hardware-optimal (sim on CPU, supervised DT on GPU); differentiates vs arXiv:2602.02903 via C3 |
| 2026-07-08 | LLM demoted to optional training-time teacher (P9) | Avoids latency & trustworthiness critiques dominating 2025–26 reviews |
| 2026-07-08 | One `.npz` per episode + run manifest | Simple, appendable, parallel-writer-safe, no new deps (HDF5 rejected for now) |
| 2026-07-08 | Wrapper/callback logger; zero env modifications | Keeps platform contribution separable from method contribution in the paper |
| 2026-07-08 | Validation gate at P4 (1×1 offline ≈ online) before multi-agent scaling | Cheap early falsification of the core hypothesis |
| 2026-07-08 | Dev stack = WSL2 Ubuntu 24.04 + VS Code WSL Remote; repo lives in Linux FS (never /mnt/c) | CityFlow/SUMO need Linux toolchain; cross-OS I/O is 10–50× slower |
| 2026-07-08 | PyTorch from cu128 wheel index; **no** CUDA toolkit, **no** Linux NVIDIA driver in WSL | RTX 5080 = Blackwell sm_120 (needs cu128 kernels); Windows driver projects libcuda into WSL; wheels bundle CUDA runtime |
| 2026-07-09 | Worker pool uses `spawn` start method (P0.3 patch) | Linux `fork` clones torch/CityFlow C++ lock state → deadlock (observed: frozen terminal, 10% CPU). Confirmed at runner.py:405 (no mp_context) |
| 2026-07-09 | grid4x4 MAPPO@60ep result retained as motivation-figure material | Online MARL below-random at low interaction budget = the paper's sample-inefficiency argument, empirically on our own platform |
| 2026-07-09 | C3 transfer reported as **within-backend-normalized** improvement (policy vs MaxPressure evaluated in the same backend), not raw travel times | Thesis §5.2: SUMO/CityFlow absolute metrics not interchangeable (corr. 0.93, CityFlow 8.4× jumpier); raw cross-backend numbers would be rejected on the thesis's own evidence |
| 2026-07-09→12 | **RESOLVED (Master chat decision, mentor free hand): primary reward = queue_length; PressLight = mandatory robustness appendix; empirical safety latch in P2.3** (Spearman corr. of queue- vs pressure-returns with avg travel time across the corpus — if pressure clearly wins, reverse before P4 at zero corpus cost) | RTG must rank trajectories consistently with the headline metric (Σqueue ≈ total delay ≈ travel time); queue is backend-neutral (waiting counts only) while pressure imports turn-topology differences into exactly the C3 interface confound; cross-paper comparability lives in metrics, not rewards; Addendum A makes the corpus reward-agnostic anyway |
| 2026-07-09 | OPTION: MOSS backend for corpus acceleration (thesis: ~3× vs SUMO on 4090, agent-bound) | Only if scenarios exist in MOSS format; not a dependency |
| 2026-07-09 | ACTION: clarify thesis-credit/co-authorship conventions with mentor (thesis is fresh, June 2026) | Platform paper (theirs) vs method paper (ours) split must be agreed early |
| 2026-07-09 | **C3 reframed:** binary "transfer works" → measured transfer curve (zero-shot → few-shot → retrain) under unified "distribution shift" narrative with C2; P7.0 early gate added; RTG calibration (P7.2) promoted to contribution candidate | Thesis §5.2 kills raw-metric comparison, not transfer itself; action semantics backend-neutral by repo design; MaxPressure transfers by construction (existence proof); Δt=10s decisions aggregate over microscopic dynamics differences; ≥7 paired scenarios verified; every curve outcome is publishable — C3 can no longer waste more than ~1 day (the P7.0 gate) |
| 2026-07-10 | **All three claims hardened after external critique** (verified sound on nearly all points): C1 → dataset-ladder study, "no online exploration" phrasing; C2 → pre-registered RQ + 2×2 with domain-randomized MAPPO + mechanism ablations; C3 → k ∈ {5,20,100} with MaxPressure probe; BC/%BC/IQL baselines added as non-negotiable; probe-calibrated return prompting named as the method component serving both C2 and C3 | Circularity, missing-BC, and data/architecture confound were genuine rejection-level flaws; verified prior art: DTLight+DTRL (2312.07795), DataLight (2303.10828), X-Light, arXiv:2603.22315 (EV-DT, Mar 2026) |
| 2026-07-11 | **P2.0 FlowRandomizer mandatory before corpus; episodes keyed (scenario, flow-draw)** — critique (a) accepted; deterministic demand kills deterministic-tier corpus content, not just CIs; randomizer must render dual-backend (vehicle-list level, verified against flow.json structure) | Regenerating the corpus later would cost weeks; fix costs one small tool |
| 2026-07-11 | **Ladder tiers = measured normalized return** (random=0, MaxPressure=100), policy names demoted to provenance — critique (b) accepted; own P0.2 data contradicts the name-implied ordering | Reviewer would catch it before us |
| 2026-07-11 | **P4.2 gate re-anchored:** ≥ MaxPressure AND within 5% of best online — critique (c) accepted | Old gate passable while losing to a 1970s heuristic |
| 2026-07-11 | **Compute budget sheet (P2.0b) gates P2; time exhaustion promoted to top risk** — critique (d) accepted | Hundreds of sim-hours on one workstation; headline/appendix split must precede spending |
| 2026-07-11 | **Pre-registration made real (P0.4):** git tag + doc hash within 24 h of reward decision — critique (e) accepted with one scheduling correction (register AFTER reward is fixed, not today) | Registering a protocol with an OPEN reward metric would undermine the registration |
| 2026-07-11 | **Reward decision de-blocked technically:** Brief #1 Addendum A adds per-lane count logging → any reward (queue/PressLight/pressure) recomputable offline from the corpus; mentor's answer now selects the headline metric only — critique (f) partially superseded by a better fix; mentor email still goes out this week (co-authorship blocks arXiv regardless) | Kilobytes per episode buy immunity to the reward decision |
| 2026-07-13 | **Transfer direction (CityFlow → SUMO) and MOSS exclusion have scientific rationale, not just convenience — write it into the paper** | Direction: (i) corpus economics (20× faster source engine is what makes the project fit one workstation), (ii) mirrors sim-to-real (train cheap/approximate, deploy expensive/accurate — the reverse has no real-world analogue), (iii) argumentative asymmetry (surviving a *fidelity upgrade* is a strong result; downgrade is not). MOSS excluded: GPU contention with DT training (our compute strategy is sim-on-CPU + training-on-GPU), zero MOSS scenarios in repo (conversion errors would contaminate the measured dynamics gap), weaker literature comparability, and its scale advantage doesn't materialize on headline networks. Reviewers will ask "why this direction?" — answer belongs in the paper, not the rebuttal |
| 2026-07-27 | **DETERMINISM EMPIRICALLY CONFIRMED on real CityFlow.** First live collection run: engine seeds 1000 and 1001 produced **byte-identical trajectories**, identical `episode_sha256`, identical return (−502.000). Predicted 2026-07-11 from P0.2's σ=0; now observed. **P2.0 FlowRandomizer is therefore blocking, not a refinement** — without it N episodes of a deterministic policy is one trajectory × N, and the manifest truthfully reports N. Also makes out-dir collision traceless (an overwritten file is identical) | The `episode_sha256` alarm specified in Brief #1 §8 fired on its first real run, exactly as designed |
| 2026-07-27 | **P2 collection config decided:** collect with `--local-reward-fn queue_length --global-reward-weight 0.0`. At the default weight 1.0 every intersection's `local_reward` silently carries the FULL global reward (`base_traffic_env.py:214`) — the composite double-counting trap. `global_reward` is unaffected by the weight, so both clean signals get logged | Surfaced by the first real collection run: `local_reward` came back all-NaN because `local_reward_fn` defaults to None. MADT needs per-intersection RTG |
| 2026-07-27 | **Mutation testing adopted as the review standard for critical-path code** (§7) | The P1 re-review ran 13 code mutations against an isolated copy: mutating `ix_ids` to `sorted()` killed 5 tests (previously 0), and reverting the float64 accumulator made a previously-passing test fail by 3.31. Falsification proved test strength in a way reading never could |
| 2026-07-27 | **SUMO lane-array concern resolved:** `SumoEnv._create_metrics()` always returns a `SumoMetrics` object (never `None`), same for CityFlow, so the empty-lane branch is unreachable there. `_require_lane_arrays` retained as defence-in-depth (MOSS may differ). **P7.0 SUMO collection needs no special metrics handling** | Verified in source 2026-07-27; reviewer's reading was right, Master chat's earlier reading was half-right |
| 2026-07-26 | **`terminated` is hardcoded `False`** (`base_traffic_env.py:604`); every episode ends by time-limit truncation at `max_steps`. Consequence for P4.4: value-based offline baselines (IQL/CQL) **must bootstrap through the boundary**, never treat it as absorbing — treating a timeout as terminal causes systematic value underestimation near episode end and would hand the MADT an unearned win over its own baselines. DT/RTG is unaffected (no bootstrapping, and all episodes share one horizon, so RTG is comparable across the corpus) | Verified 2026-07-26 from the P1 plan's open question; a fair-baseline defect here would be a rejection-level flaw in C1 |
| 2026-07-26 | **`local_reward` in `info` is composite, not purely local:** `_get_local_rewards()` returns `global_reward_weight * global_reward + local_reward_fn(local_metrics[j])`. P3/P4 must not condition on both a per-intersection RTG and a separate global RTG without accounting for the overlap, or the global signal is double-counted | Found in Master-chat verification of the P1 plan; not flagged by the sub-chat |
| 2026-07-26 | **`ix_ids` stored in `env.intersections` order, NOT lexicographic** (Brief #1 v2 said sorted — that was wrong and is hereby corrected). C1 orders the action vector by `env.intersections`, so native order gives `ix{i}_action ≡ action[i]` with no remap table. **P3 loader must key by intersection ID, never by positional index** — CityFlow and SUMO may enumerate in different orders, and positional pairing would silently mismatch intersections in C3 | Eliminating a remap eliminates a whole off-by-one class; `lane_ids` stays lexicographic (frozen at reset) |
| 2026-07-26 | **`collect.py` hard-fails when `L == 0`** (lane arrays empty). SUMO/MOSS return `{}` lane dicts when the metrics pipeline is off (`sumo_env.py:301`, `moss_env.py:653`); CityFlow falls back to a direct engine query and is always populated. The logger records `L=0` honestly (correct layer); the collector refuses to build a corpus that silently violates C6 reward-agnosticism | P7.0 needs SUMO collection specifically; discovering an unusable corpus at P3 costs an overnight run |
| 2026-07-26 | **`utils/` discovered as a frozen dependency of `envs/` (imported by all four backend env files) and added to the frozen set, along with `scripts/**` and `.claude/**` for the same reason — a guard must not be able to unfreeze itself.** One dated exception: `scripts/check_english.sh` stays writable while being tuned, encoded as `FROZEN_EXCEPTIONS` in the guard rather than left as a silent gap | Not enumerated in the original brief; found via `/init`'s repo scan and a grep of the import graph before P1 started |
| 2026-07-25 | **Brief #1 consolidated into a single v2 document**; Addendum A, ADDENDUM_A_PATCH and BRIEF_01_DELTA all superseded and retired | Four documents that disagree in places are how an off-by-one gets frozen into a data format; one source of truth per task is now the standard for every brief |
| 2026-07-25 | **Format alignment fixed and promoted into contract C6 (v1.1):** observations T+1 rows, decisions/outcomes T rows; `local_reward` read at `on_step_result` (post-step), never bundled with the pre-decision observation; uniform "every info recorded once on arrival" write rule replaces any terminal-row special case | Verified in `envs/base_traffic_env.py::step()`: simulation advances BEFORE reward and info are built, so `r_t` is recoverable only from row `t+1` and `info[...]["reward"]` from `step t` is `r_t`. External critique caught the missing final row; the `local_reward` off-by-one (worse — it shifts *every* per-intersection RTG, not just the last) was found in Master-chat verification |
| 2026-07-25 | **Fixed-time split out as P2.5** rather than added to Brief #1 (external critique flagged it as a BLOCKER for Brief #1 — accepted as a gap, rejected as scope) | No fixed-time controller exists in the repo, so this is a policy implementation with real design decisions, not a CLI flag; adding it would break §7's ≤2-file brief limit. `--policy` becomes a registry now so P2.5 needs no CLI change |
| 2026-07-25 | **Claude Code guard hook split into `--frozen-only` (fires after Bash too) and `--tests-only` (file edits only)** | Original matcher covered only the file-edit tools, so a bash heredoc write to a frozen file would go undetected until the next file edit; adding Bash to the single hook would have run pytest after every `ls`, since the test gate cannot early-exit while a task has changed files |
| 2026-07-13 | **P11 stretch-goals phase created** (transfer triangle w/ MOSS, reverse-direction ablation, city-scale demo, LLM-teacher); P9 folded into it | Attractive ideas kept visible but firewalled from the critical path — top risk is time exhaustion, so "nice to have" must never silently compete with load-bearing work |
| 2026-07-12 | Mentor gave free hand on positioning; controversy framing retained as *scientific* motivation (a genuine open question in the literature), not marketing; credit/citation deferred to P10 by mentor (non-blocking) | "Good science first" is compatible with adjudicating a real contradiction |
| 2026-07-12 | **Protocol hardened for AI-written code:** task branches, two-Claude rule (independent review sub-chats on critical path), double computation of critical quantities, corpus linter with duplicate detection, self-review checklist, ≤2-file brief limit, Claude Code as preferred coding mode | Dominant risk of AI-heavy development = silent semantic bug in confident code; redundancy is cheaper than a retracted result |
| 2026-07-11 | **P10.0 release package added** (datasheet, license, Zenodo DOI) + license audit pulled into P2.3 — critique (g) accepted, extended with source-scenario redistribution audit | A promised public dataset without redistribution rights is worse than no promise |
| 2026-07-10 | **Paper hook = adjudicating the literature controversy:** DataLight says DT cannot work for TSC; DTLight and 2602.02903 say it does. Our ladder + shift + calibration explain when and why. | A published negative result on our core method is a framing gift, not a threat — provided we engage it head-on |

---

## 9. Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DT underperforms online MAPPO even on 1×1 | Med | P4 gate; remedies: richer behavior mix, longer context K, %BC-style filtering of top-return trajectories |
| CityFlow/SUMO state features misaligned → C3 collapses | Med | P7.0 early gate (1 day, right after P1) measures the shift before any transfer investment; feature-set discipline in P7.1; C3 reframed as transfer curve → no binary failure mode |
| RTG prompting miscalibrated across backends (reward-scale shift) | Med | P7.2 calibration protocol (MaxPressure-relative or quantile-space prompting); if unsolvable, becomes the characterized-limitation finding |
| ESPER/infeasible-RTG failure under scenario shift (return-conditioning erratic in stochastic regimes) — plausibly why DataLight's DT failed | Med-High | Probe-calibrated prompting (P4.3) tested under shift in P6.3; naive-vs-calibrated ablation makes even the failure mode a finding |
| BC-on-expert matches MADT → sequence modeling adds nothing | Med | Tested early at P4.4 on 1×1; if true, pivot headline weight to ladder + shift + calibration findings (which BC cannot deliver) rather than architecture |
| **Time exhaustion at ~40% of experiments (single workstation vs post-hardening matrix) — TOP RISK, above scoop** | High | P2.0b measured sim-hours sheet + frozen headline/appendix split BEFORE P2; online baselines headline-only; per-tier eval on headline scenarios only; monthly budget re-check in Master chat |
| Scoop cadence (DTLight→DataLight→2602.02903→2603.22315: one relevant paper roughly every 1–3 months) | High | Speed on P1–P5; arXiv early; our unique assets (paired dual-backend scenarios, ladder, calibration) are the hardest to replicate |
| Scooped by follow-ups to arXiv:2602.02903 | Med | Speed on P1–P5; C2+C3 remain differentiators even if core overlaps |
| Manhattan 28×7 too heavy for full sweeps | Low | Headline results on grid4x4 + hangzhou_4x4; Manhattan as scale demo only |
| RESCO/state parity issues across backends | Low | Repo already reproduces RESCO `drq_norm`; reuse it |
| Deterministic CityFlow demand | ~~Med~~ RESOLVED 2026-07-11 | Was mis-scheduled as a P8 statistics issue; actually kills deterministic-tier corpus content. Fixed at the source: P2.0 flow randomization mandatory for every episode; also supplies environment stochasticity for P8 CIs |

---

## 10. Current Status

**Active task:** P2.0 FlowRandomizer (Brief #2a) — **blocking**: the corpus carries no information without it (determinism confirmed 2026-07-27).
**P1 delivered:** `offline/trajectory_logger.py` (780 L), `offline/collect.py` (370 L), `tests/test_trajectory_logger.py` (896 L, 30 tests); whole suite 180 green. Master-chat spot-check on merged code 2026-07-27: `ix_ids` native order ✓, `lane_ids` sorted ✓, `local_reward` from `next_info` ✓, all three mandated docstring statements ✓, NB2 validate-before-mutate ordering ✓. **P0.4 pre-registration UNBLOCKED** (reward decided 2026-07-12) — tag `v0.1-prereg` now. Mentor email no longer needed (free hand given; credit deferred to P10).
**Next in queue:** Brief #2a (P2.0 FlowRandomizer) + P2.0b compute-budget sheet — both BEFORE any corpus collection; then P2.5 (fixed-time), P2.1+. Carried P1 notes for P3: key by intersection ID never index; `local_reward` composite; episode ends are truncations; format v1.0 stores global metrics only. Carried for P2: `--overwrite` deletes rather than appends, so use one out_dir per (policy, scenario, seed-block).
