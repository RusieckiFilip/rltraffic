# rltraffic — offline reinforcement learning for traffic signal control

A research project studying **what decides the outcome in offline RL for traffic
signal control**: the learning architecture, or the composition of the data it
learns from.

Policies are trained entirely from logged trajectories, with **no online
exploration** — the simulator is used to collect the corpus and to evaluate, never
to train. The work is pre-registered: metrics, decision thresholds and analysis
rules are committed and tagged before the measurements that test them.

> **Status:** research in progress, targeting a publication. The single-intersection
> offline results are complete and independently reviewed; multi-intersection and
> cross-backend work is ongoing. Nothing here should be read as a settled finding
> until the paper states it.

---

## Attribution

This repository has **two distinct parts, by different authors**.

### The simulation platform — bachelor's thesis project

Everything that makes traffic simulation, agents and experiments work was built as
a bachelor's thesis at the Faculty of Mathematics, Informatics and Mechanics,
University of Warsaw (June 2026):

> **Environment for controlling traffic lights with reinforcement learning**
> Beniamin Bibrowski, Piotr Bublik, Karol Pisula, Mikołaj Woliński
> Supervisor: mgr Grzegorz Grudziński

Their contribution is the simulator-agnostic framework itself — abstracting the
simulator layer so that agents, rewards, metrics and experiments run unchanged
against CityFlow, SUMO or MOSS, and removing the performance bottlenecks of the
earlier SUMO-only RESCO TensorCell environment. Concretely:

| Path | What it is |
|---|---|
| `envs/` | The three simulator backends behind one Gymnasium-style API |
| `agent/` | IDQN, IPPO and MAPPO agents, MaxPressure and fixed-time baselines |
| `algorithms/`, `states/`, `metrics/`, `rewards.py` | Learning algorithms, observation features, the metrics pipeline, reward functions |
| `experiments/` | The config-driven `agents × environments × seeds` framework |
| `CityFlow/` | Vendored CityFlow, patched to build on Python 3.12+ |
| `docs/architecture.md` and the other platform docs | Documentation of all of the above |

That work is described in [`docs/README.md`](docs/README.md) and summarised under
[Simulation platform](#simulation-platform) below.

It in turn builds on work by others: the
[RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO) (Ault & Sharon, NeurIPS
Datasets & Benchmarks 2021), whose evaluation protocol, configurable state and
reward formulations, phase-transition semantics and several scenarios it retains;
the TensorCell research group's RESCO fork; and the
[CityFlow](https://github.com/cityflow-project/CityFlow),
[SUMO](https://eclipse.dev/sumo/) and [MOSS](https://github.com/tsinghua-fib-lab/moss)
simulators.

### The offline RL research — this project

Built on top of that platform, by **Filip Rusiecki**, supervised by Paweł Gora:

| Path | What it is |
|---|---|
| `offline/` | Trajectory logging, corpus loader, offline agents, evaluation harness and statistical gates |
| `PREREGISTRATION.md` | Pre-registered hypotheses, metrics, decision rules and dated amendments |
| `docs/PROJECT_PLAN.md` | Claims, phase checklist, working protocol and decisions log |
| `docs/CONTRACTS.md` | Frozen data-format and semantic contracts |
| `docs/briefs/`, `docs/plans/`, `docs/returns/`, `docs/reviews/` | The task record: what was specified, planned, delivered and independently reviewed |
| `scenarios/`, `configs/` | Research scenarios and the corpus-collection configuration |

The two parts are kept separate on purpose: the platform is treated as a frozen
dependency, and the research does not modify it except where a change is
documented as a contract.

---

## The research question

Offline RL learns a policy from a fixed dataset. Its appeal for traffic control is
obvious — no risky exploration on a live network — but it raises a question the
field has largely left unmeasured: **when an offline method performs well, is that
the method, or the data?**

The project answers this by holding everything else fixed and varying one thing at
a time:

- **A dataset quality ladder.** The same offline method trained on corpora
  collected from behaviour policies of measured quality, from random through
  rule-based to converged MAPPO.
- **A complete baseline set.** Behaviour cloning, filtered behaviour cloning and
  IQL against a Decision Transformer — same corpus, same held-out demand draws,
  same seeds, same training budget, same evaluation function.
- **Mechanism, not just ranking.** When a method wins, the project measures *what
  its mechanism actually selected for*, rather than inferring it from the score.

Two further axes are planned: robustness under scenario shift, and transfer of a
CityFlow-trained model to a different simulation engine.

---

## How the work is run

The project's protocol is unusual enough to be worth stating, because every number
in the repository depends on it.

- **Pre-registration before measurement.** Decision thresholds, primary metrics and
  equivalence margins are committed and git-tagged before the run that tests them.
  Amendments are dated and annotated in place; registered text is never edited.
- **Held-out evaluation.** A registered split reserves demand draws that no
  training run may see, enforced at the loader.
- **Paired statistics.** Every arm is evaluated on the identical draw set, compared
  with paired non-parametric tests, and reported with confidence intervals and
  effect sizes.
- **Independent review before merge.** Each task is reviewed by a session that did
  not write it, using mutation testing: a test that survives the mutation it claims
  to catch is reported as providing no coverage.
- **Verify the artifact, not its description.** Claims are checked against the
  committed data, not against the report that summarises it.

The working rules live in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §7 and
[`CLAUDE.md`](CLAUDE.md).

---

## Simulation platform

*Authored as the bachelor's thesis project described under [Attribution](#attribution).*

Reinforcement learning for traffic signal control, with one agent API across three
interchangeable microscopic traffic simulators.

- **Three simulator backends, one API** — CityFlow (fast C++, vendored), SUMO
  (TraCI / libsumo) and MOSS (GPU-accelerated). Agents, rewards, metrics and
  experiments are backend-agnostic.
- **Multi-agent RL built in** — IDQN, IPPO and MAPPO (centralised critic), all with
  action masking and per-intersection rewards.
- **Composable observations and rewards** — named state features (including
  RESCO's `drq_norm`) and reward functions (`queue_length`, `presslight`, RESCO's
  `wait_norm`, …); required metrics are enabled automatically.
- **Safe signal semantics** — four phase-control modes (acyclic, bounded, cyclic,
  RESCO-cyclic) with enforced yellow/all-red clearances and min/max green times.
- **Config-driven experiments** — one JSON describes the matrix; the runner trains,
  evaluates on paired seeds, adds baselines and writes `results.json`,
  `summary.csv` and comparison plots.

Full platform documentation: [`docs/README.md`](docs/README.md).

---

## Installation

```bash
# 1. Install CityFlow (vendored; see https://cityflow.readthedocs.io/en/latest/install.html)
pip install ./CityFlow

# 2. Install this package (editable mode for development)
pip install -e ".[dev]"

# 3. (Optional) extras: plotting for experiment reports
pip install -e ".[viz]"
```

The [`CityFlow/`](CityFlow/) directory is a vendored copy of the upstream
[CityFlow simulator](https://github.com/cityflow-project/CityFlow), patched so it
builds and runs on Python 3.12+ (upstream does not) — install it from this repo,
not from PyPI/upstream.

SUMO and MOSS are optional; install them only for those backends
(`eclipse-sumo` / `python-moss`).

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

Backend-specific tests skip automatically when an engine is not installed. Tests
that read the offline corpus skip unless the corpus environment variables are set.

## Licence

MIT — see [`LICENSE`](LICENSE).
