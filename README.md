# rltraffic — offline reinforcement learning for traffic signal control

A pre-registered study of offline reinforcement learning for traffic signal control. Its main
subject is the Decision Transformer (DT), evaluated against behaviour cloning, return-filtered
behaviour cloning and IQL trained on the same data. The DT is an existing method class, not a
model proposed here: the multi-agent Decision Transformer was proposed by Meng et al.
([arXiv:2112.02845](https://arxiv.org/abs/2112.02845)) and applied to traffic signal control by
Su, Sun & Deng ([arXiv:2602.02903](https://arxiv.org/abs/2602.02903)). The contribution of this
repository is the measurements. It asks three questions:

- **C1 — data.** How does the DT's performance depend on the quality of the data it learns from,
  and does it beat simpler offline methods trained on the same data?
- **C3 — transfer.** How does a model trained in CityFlow perform in SUMO, from zero-shot
  through few-shot fine-tuning to retraining on the target simulator?
- **H4 — context length.** Is DataLight's published negative result for DTs in traffic
  control explained by the short context they used?

Every policy is trained from logged trajectories only. The simulator collects the data and
evaluates the policies; it never trains them. Hypotheses, metrics, decision rules and
equivalence margins were registered and tagged in advance. Every later amendment is dated and
tagged and states which results had already been seen when it was written (`PREREGISTRATION.md`
§12); several, among them A13, A15 and A19, were written after related results existed and are
reported as such.

> **Status (October 2026):** the experiments for the first paper — C1, C3, H4 and the
> compute-and-latency table — are merged. One C1 cell (IQL at the 4×4 grid's random tier) was
> found to have trained on twice its declared data and was re-trained on the declared data: it is
> about 7.5 s slower and every registered verdict stands (P5.2b,
> `docs/data/p5_2b_iql_correction.json`). One registered hypothesis, H2 (robustness under scenario
> shift), was **not tested** and is reported as such (`PREREGISTRATION.md`, amendment A25). The
> paper is in preparation. Nothing here is a settled finding until the paper states it.

---

## Headline results

Short summaries of what the committed artifacts show. The paper gives the full registered
statements, their scope and their caveats; each line points to the artifact or task behind it.

### C1 — performance along a data-quality ladder

Travel times below are the registered primary metric on both scenarios, `att_engine`
(amendment A11's Rule R), which counts every vehicle the demand created; every statement also
holds under the earlier definition unless it says otherwise.

- **Single intersection (CityFlow, Hangzhou 1×1).** The DT leads none of the eight
  single-intersection data tiers we measured — a descriptive count, not an inferential claim.
  On the best tier (data from a converged MAPPO teacher), plain behaviour cloning matches the DT
  within the registered equivalence margin (δ = 0.6263 s), while return-filtered BC and IQL beat
  it beyond that margin (their whole 95 % CI lies past δ). About three quarters of the
  filtered-BC advantage is checkpoint selection — keeping the episodes of the best teacher
  checkpoints — and the rest is return ranking within them; performance falls monotonically as
  the source checkpoints get worse, a dose-response (P4.4, P4.5).
- **Sixteen intersections (CityFlow, synthetic 4×4 grid).** No method leads everywhere. By mean
  travel time the non-spatial DT leads on the MAPPO-teacher tier, BC on the MaxPressure tier (by
  0.9 s, a lead whose paired 95 % CI includes zero under the primary metric but not under the
  earlier one), and IQL on the fixed-time and random tiers. The random-tier IQL cell was first
  trained on twice its declared data and was re-trained on the declared 200 episodes (P5.2b): it is
  7.5 s slower under P5.2's metric and 7.6 s under the primary one, still leads on every training
  seed, and no registered verdict changed. The DT arms are the only arms that never
  collapse below the random-policy anchor (P5.2).
- **Cross-intersection attention.** Adding the cross-intersection attention path destabilises
  training: on the best-data tier the spatial model's travel time varies across training seeds
  about twenty times more than the non-spatial model's, and the trained spatial models barely
  use their neighbours (median r = 0.064 against the registered threshold of 0.10; amendment
  A22, P5.4). The spatial model's penalty is large only on that tier (+40 s); it is +0.7 s on
  MaxPressure data, exactly zero on fixed-time data, and reversed on random data (−0.7 s)
  (P5.1, P5.2).
- **The return prompt.** Removing the return token raises travel time by 409 s on a corpus that
  mixes expert and random episodes (95 % CI 394–424 s; 194 s under the earlier definition, the
  difference being mostly time that vehicles spent waiting to enter the network); on expert data no difference is
  detected (0.12 s, a 95 % CI that includes zero — a failure to reject, not a demonstrated
  equivalence) (P5.3b).

### C3 — CityFlow → SUMO transfer

Transfer is scored within SUMO as ρ = (ATT_fixed-time − ATT_policy) / (ATT_fixed-time − ATT_MaxPressure),
so fixed-time = 0 and SUMO's own MaxPressure = 1. Raw travel times are never compared across
simulators.

- **Zero-shot, no SUMO data.** On the single intersection the CityFlow-trained DT (MAPPO-teacher
  data) reaches ρ = +1.89 [+1.86, +1.92], better than SUMO's MaxPressure; trained on mixed data it
  reaches +1.73 [+1.69, +1.77]. On the 4×4 grid the non-spatial DT reaches
  ρ = +0.89 [+0.87, +0.90] (P7.3a, P7.3d).
- **Full-retrain anchor (single intersection).** A DT trained from scratch on 200 SUMO
  MaxPressure episodes reproduces its demonstrator, ρ = +0.98 [+0.97, +1.00], below the zero-shot
  +1.89. Behaviour cloning on MaxPressure demonstrations is pinned near ρ = 1 by construction, so
  the gap reflects demonstration quality, not harm from target-domain data (P7.3b, amendment
  A19(c)).
- **Few-shot, 4×4 grid (exploratory).** Fine-tuning on 100 SUMO MaxPressure episodes closes
  61.9 % [50.3 %, 73.5 %] of the zero-shot gap to MaxPressure. Fine-tuning on five episodes makes
  the model worse: ρ = +0.77 [+0.76, +0.79] against the zero-shot +0.89 (P7.3c).
- **Scope.** The gap is measured under one frozen feature alignment between the simulators
  (amendment A16). An interface-mismatch component has not been separated from it, so the paper
  does not call it a dynamics gap without that qualifier (A25).

### H4 — context length (confirmatory)

On the single-intersection scenario (MAPPO-teacher data, 100 held-out demand draws, five
training seeds per arm), an improvement with context length K ∈ {1, 2, 5, 10, 20} is **not
detected**. The registered trend contrast is +1.2762 [+0.6927, +1.8597]; a positive value means
travel time rises with K. Descriptively, in no training seed is K = 1 or K = 2 worse than K = 20
by the registered margin δ = 0.63 s, and pooled over seeds both are slightly better than K = 20.
The data point toward longer context being slightly worse, but that direction is not established
across seeds. This sweep therefore gives no support, on this corpus, to context length as the
explanation of DataLight's negative result (P5.3c, outcome (iii)).

### Compute and latency

Median time per decision, measured inside real episodes:

| | one CPU thread | GPU |
|---|---|---|
| DT, K = 20 (1×1, the H4 sweep's model) | 1.43 ms | 1.80 ms |
| DT, K = 1 (1×1) | 0.76 ms | 1.63 ms |
| BC / IQL (1×1) | 0.33 / 0.47 ms | 0.64 / 0.77 ms |
| MAPPO (1×1) | 0.28 ms | 1.09 ms |
| DT, all 16 intersections (4×4; spatial and non-spatial variants) | 18–21 ms | 3.1–3.7 ms |
| MaxPressure (1×1 / 4×4) | 0.01 / 0.25 ms | — |

Every controller decides far faster than the 10-second action interval used throughout. The
numbers come from one laptop (Intel Core Ultra 9 275HX, RTX 5080 Laptop GPU, Linux under WSL2,
mains power, Windows "Best performance" mode); P4's own K = 20 DT times at 1.40 / 1.78 ms. `docs/data/p8_2_compute.json` holds the full table with training
cost and parameter counts, and it carries the caveats the numbers need: p95 sensitivity, the
noise floor between identical computations, and MAPPO's per-intersection GPU transfers (P8.2).

### Stated limitations

- Every multi-intersection result is measured on **one synthetic network** (the 4×4 grid).
- **H2** (robustness under scenario shift) was registered with a confirmatory 2×2 design and not
  tested. No number bearing on it exists in this repository.
- The transfer gap has not been separated from interface mismatch (the state-encoding control
  P7.4 was not run).

---

## Attribution

This repository has **two distinct parts, by different authors**.

### The simulation platform — bachelor's thesis project

Everything that makes traffic simulation, agents and experiments work was built as a bachelor's
thesis at the Faculty of Mathematics, Informatics and Mechanics, University of Warsaw (June 2026):

> **Environment for controlling traffic lights with reinforcement learning**
> Beniamin Bibrowski, Piotr Bublik, Karol Pisula, Mikołaj Woliński
> Supervisor: mgr Grzegorz Grudziński

Their contribution is the simulator-agnostic framework itself. It abstracts the simulator layer so
that agents, rewards, metrics and experiments run unchanged against CityFlow, SUMO or MOSS, and,
according to the thesis, it trains about three times faster than the RESCO TensorCell environment it
replaces (in-process libsumo and metric caching).

| Path | What it is |
|---|---|
| `envs/` | The three simulator backends behind one Gymnasium-style API |
| `agent/` (except the three files listed under the research part) | IDQN (including a RESCO-parity PFRL variant), IPPO and MAPPO agents |
| `algorithms/`, `states/`, `metrics/`, `rewards.py` | The MaxPressure baseline, observation features, the metrics pipeline, reward functions |
| `experiments/` | The config-driven `agents × environments × seeds` framework |
| `CityFlow/` | Vendored CityFlow, patched to build on Python 3.12+ |
| `docs/architecture.md` and the other platform docs | Documentation of all of the above |

Experiments compare against random and MaxPressure baselines. That work is described in
[`docs/README.md`](docs/README.md) and summarised under [Simulation platform](#simulation-platform)
below.

It builds on work by others: the [RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO)
(Ault & Sharon, NeurIPS Datasets & Benchmarks 2021), whose configurable state and reward
formulations, phase-transition semantics and several scenarios it retains; the
TensorCell research group's RESCO fork; and the
[CityFlow](https://github.com/cityflow-project/CityFlow), [SUMO](https://eclipse.dev/sumo/) and
[MOSS](https://github.com/tsinghua-fib-lab/moss) simulators.

### The offline RL research — this project

Built on top of that platform by **Filip Rusiecki**, supervised by Paweł Gora:

| Path | What it is |
|---|---|
| `offline/` | Trajectory logging, the corpus loader, offline training and evaluation, the fixed-time controller (`offline/policies/fixed_time.py`), statistical gates, CityFlow → SUMO alignment and transfer, the context-length sweep, the compute-and-latency harness |
| `offline/campaigns/` | Drivers for the long runs; the later ones refuse to start without the author's run token |
| `agent/DTAgent.py`, `agent/SpatialDTAgent.py`, `agent/OfflineBaselines.py` | This project's implementations of the Decision Transformer, its spatial (cross-intersection attention) variant, and the BC, filtered-BC and IQL baselines |
| `calibration/` | External calibration of the IQL implementation on D4RL (P8.3) |
| `PREREGISTRATION.md` | Registered hypotheses, metrics, decision rules and dated amendments |
| `docs/data/` | The committed result artifacts (JSON) behind the reported numbers |
| `docs/PROJECT_PLAN.md` | Claims, phase checklist, working protocol and decisions log |
| `docs/CONTRACTS.md` | Frozen data-format and semantic contracts |
| `docs/briefs/`, `docs/plans/`, `docs/returns/`, `docs/reviews/`, `docs/notes/` | The task record: what was specified, planned, delivered and independently reviewed, plus campaign reads |
| `scenarios/`, `configs/` | Scenarios, simulator configurations and the corpus-collection configuration |
| `.github/ci/`, `scripts/`, `githooks/` | The CI gate with its pinned skip ceiling, and the English, test-hygiene and guard checks |
| `requirements-frozen.txt` | The exact environment that produced the committed numbers: a record, not an install target |

The two parts are kept separate on purpose. The platform is treated as a frozen dependency, and the
research changes it only where the change is documented as a contract.

---

## How the work is run

Every number in the repository depends on this protocol.

- **Pre-registration before measurement.** Decision thresholds, primary metrics and equivalence
  margins are committed and git-tagged (`v*-prereg-*`) before the run that tests them. The
  registration as amended through A9 (tag `v1.0-prereg-a9`) is deposited on Zenodo (record DOI
  [10.5281/zenodo.21968773](https://zenodo.org/records/21968773)). Later amendments are dated,
  tagged and state whether results had been seen; registered wording is never altered (two
  registered lines carry dated in-place qualifications from 2026-08-07).
- **Held-out evaluation.** A registered split reserves demand draws 1000–1099, which no training run
  may see; the loader enforces it.
- **Paired comparisons.** Every arm is evaluated on the identical held-out draw set, and arms are
  compared paired over draws, with 95 % confidence intervals. The primary metric (`att_engine`,
  amendment A11's Rule R) is average travel time counted over every vehicle the demand created,
  including vehicles still waiting to enter, so a policy cannot score well by keeping vehicles
  out.
- **Independent review before merge.** Critical-path tasks are reviewed before merge by a session
  that did not write them, using mutation testing: a test that survives the mutation it claims to
  catch counts as no coverage. Two merges (P7.0, P8.3) carried no independent review, and their
  numbers are kept out of the paper until one exists. Since October 2026, new and changed test
  files are also run once in a depth-1 clone at every merge review, the way CI checks the
  repository out.
- **Verify the artifact, not its description.** Claims are checked against the committed data,
  recomputed by an independent route, never against the report that summarises them.
- **Guarded runs.** The later long runs start only with the author's token, after canary and
  machine-state checks; the latency run also required mains power and a fixed Windows power mode.

The working rules live in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §7 and
[`CLAUDE.md`](CLAUDE.md).

---

## Simulation platform

*Authored as the bachelor's thesis project described under [Attribution](#attribution).*

Reinforcement learning for traffic signal control, with one agent API across three
interchangeable microscopic traffic simulators.

- **Three simulator backends, one API**: CityFlow (fast C++, vendored), SUMO (TraCI / libsumo) and
  MOSS (GPU-accelerated). Agents, rewards, metrics and experiments are backend-agnostic.
- **Multi-agent RL built in**: IDQN, IPPO and MAPPO (centralised critic), all with action masking
  and per-intersection rewards.
- **Composable observations and rewards**: named state features (including RESCO's `drq_norm`) and
  reward functions (`queue_length`, `presslight`, RESCO's `wait_norm`, …); required metrics are
  enabled automatically.
- **Safe signal semantics**: four phase-control modes (acyclic, bounded, cyclic, RESCO-cyclic) with
  enforced yellow/all-red clearances and min/max green times.
- **Config-driven experiments**: one JSON describes the matrix; the runner trains, evaluates on
  paired seeds, adds baselines and writes `results.json`, `summary.csv` and comparison plots.

Full platform documentation: [`docs/README.md`](docs/README.md).

---

## Installation

Python 3.12 or newer.

```bash
# 1. Install CityFlow (vendored; see https://cityflow.readthedocs.io/en/latest/install.html)
pip install ./CityFlow

# 2. Install this package (editable mode for development)
pip install -e ".[dev]"

# 3. (Optional) extras: plotting for experiment reports
pip install -e ".[viz]"
```

The [`CityFlow/`](CityFlow/) directory is a vendored copy of the upstream
[CityFlow simulator](https://github.com/cityflow-project/CityFlow), patched so it builds and runs
on Python 3.12+ (upstream does not). Install it from this repository, not from PyPI or upstream.

The SUMO and MOSS engines are optional; install them only for those backends (`eclipse-sumo` /
`python-moss`; MOSS needs Linux and CUDA). SUMO's Python client (`traci`) is installed with the
package. The C3 transfer results need SUMO.

## Quick start

Drive an environment directly:

```python
from envs.cityflow_env import CityFlowEnv
from agent.DQNAgent import DQNAgent

env = CityFlowEnv("configs/sim/cityflow1x1.json", max_steps=360, delta_time=10)
agent = DQNAgent(env)

info = env.reset(seed=42)
for _ in range(env.max_steps):
    action = agent.act(info, explore=True)
    reward, terminated, truncated, info = env.step(action)
    agent.observe(info, reward, terminated, truncated)
env.close()
```

Or run a full experiment matrix:

```bash
python experiments/run.py experiments/configs/smoke.json
```

## Tests

```bash
pytest
```

Backend-specific tests skip when an engine is not installed. Tests that need local data skip
unless their environment variables point at it: the output tree (`RLTRAFFIC_OUTPUT_ROOT`), the
corpora (`RLTRAFFIC_CORPUS_V11`, `RLTRAFFIC_CORPUS`, `RLTRAFFIC_SUMO_CORPORA`), P4's checkpoints
(`RLTRAFFIC_P4_DT_CHECKPOINTS`), the scenario draws (`RLTRAFFIC_DRAWS`) and the RESCO grid
(`RLTRAFFIC_GRID4X4_RESCO`). With the data present, the gated regression tests
regenerate committed artifacts from the raw outputs and compare them byte for byte. CI runs the
rest, and `.github/ci/ci_baseline.json` pins how many tests may skip there.

## Citing

The paper is in preparation. Until it appears, please cite the pre-registration:
DOI [10.5281/zenodo.21968773](https://zenodo.org/records/21968773).

## Licence

The code is MIT-licensed; see [`LICENSE`](LICENSE). The redistribution rights of the third-party
network and demand files under `scenarios/` have not yet been audited (`docs/PROJECT_PLAN.md`,
P2.3), so the licence should not be read as covering them.
