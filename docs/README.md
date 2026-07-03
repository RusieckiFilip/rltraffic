# zpp-traffic-control — Documentation

Reinforcement-learning environments and agents for **traffic signal control**,
built on top of three interchangeable microscopic traffic simulators:

| Backend | Engine | Notes |
|---|---|---|
| CityFlow | C++ multi-thread simulator (vendored in `CityFlow/`) | fast CPU simulation |
| SUMO | Eclipse SUMO via TraCI or libsumo | reference simulator, GUI support |
| MOSS | GPU-accelerated simulator from Tsinghua FibLab | large-scale scenarios, CUDA |

All three are wrapped behind one Gymnasium-style base class
(`envs.base_traffic_env.BaseTrafficEnv`), so agents, rewards, metrics and the
experiment framework are **backend-agnostic**: the same agent code runs
unchanged against CityFlow, SUMO or MOSS.

## Contents

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | Package layout, data flow, main design decisions |
| [environments.md](environments.md) | The env API, the three simulator backends, seeding |
| [phase-control.md](phase-control.md) | Action semantics: acyclic / cyclic / RESCO-cyclic control |
| [states.md](states.md) | Observation (state) features, including RESCO's `drq_norm` |
| [rewards.md](rewards.md) | Global and per-intersection reward functions |
| [metrics.md](metrics.md) | The metrics pipeline and the metric catalog |
| [agents.md](agents.md) | DQN / IPPO / MAPPO agents and the baselines |
| [experiments.md](experiments.md) | The config-driven `agents × environments × seeds` experiment framework |

## Installation

```bash
# 1. Install CityFlow (vendored; see https://cityflow.readthedocs.io/en/latest/install.html)
pip install ./CityFlow

# 2. Install this package (editable mode for development)
pip install -e ".[dev]"

# 3. (Optional) extras
pip install -e ".[viz]"   # matplotlib (experiment plots)
```

Backend engines are optional and imported lazily:

- **SUMO** — install Eclipse SUMO (`traci` is a core dependency; the
  `eclipse-sumo` pip wheel or a system install both work; `libsumo` is
  optional for in-process speed).
- **MOSS** — `pip install python-moss` (requires CUDA).

An env class fails only when you actually construct it without its engine;
everything else (config validation, tests, dry runs) works without any
simulator installed.

## Quick start

```python
from envs.cityflow_env import CityFlowEnv
from agent.DQNAgent import DQNAgent

env = CityFlowEnv(
    cityflow_config_path="configs/sim/cityflow1x1.json",
    max_steps=360,          # agent decisions per episode
    delta_time=10,          # simulation seconds per decision
    global_reward_fn="queue_length",
)

agent = DQNAgent(env)

info = env.reset(seed=42)
for _ in range(env.max_steps):
    action = agent.act(info, explore=True)
    reward, terminated, truncated, next_info = env.step(action)
    agent.observe(next_info, reward, terminated, truncated)
    info = next_info
    if terminated or truncated:
        break
env.close()
```

Note the loop shape: `reset()` returns only `info`, and `step()` returns
`(reward, terminated, truncated, info)`. There is no separate observation —
agents read per-intersection state vectors directly from
`info["intersections"][id]["state"]` (see [environments.md](environments.md)).

To run a full comparison matrix instead of a hand-written loop:

```bash
python experiments/run.py experiments/configs/smoke.json
python experiments/run.py experiments/configs/example_cologne.json --workers 4
```

See [experiments.md](experiments.md) for the config format.

## Repository layout

```
envs/           Gymnasium-style environments (base + CityFlow/SUMO/MOSS backends)
agent/          Trainable RL agents: DQN, IPPO, MAPPO (+ RESCO-parity PFRL DQN)
algorithms/     Non-learned controllers (MaxPressure)
rewards.py      Reward function registry (global + per-intersection)
metrics/        Lazy, opt-in metrics pipeline per backend
states/         Observation feature composition (+ structured drq_norm)
experiments/    Config-driven experiment runner, registry, reporting
utils/          Roadnet parsing and shared topology models
configs/sim/    CityFlow simulator config files
scenarios/      Road networks and traffic demand (CityFlow, SUMO, RESCO BB5B, ...)
tests/          Pytest suite (runs without any simulator installed)
benchmark_sumo_moss.py   SUMO vs MOSS wall-time benchmark script
CityFlow/       Vendored CityFlow simulator source, patched for Python 3.12+
```

## Running the tests

```bash
pytest
```

Backend-specific tests skip automatically when the corresponding engine is
not installed.
