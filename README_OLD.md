# zpp-traffic-control

Reinforcement learning for **traffic signal control**, with one agent API
across three interchangeable microscopic traffic simulators.

*Developed as our bachelor's thesis project.*

Train a DQN / IPPO / MAPPO agent to control traffic lights, compare it
against MaxPressure and random baselines, and run the whole comparison as a
reproducible `environments × agents × seeds` matrix from a single JSON
config.

## Highlights

- **Three simulator backends, one API** — CityFlow (fast C++, vendored),
  SUMO (TraCI / libsumo), and MOSS (GPU-accelerated). Agents, rewards,
  metrics and experiments are backend-agnostic.
- **Multi-agent RL built in** — independent DQN, independent PPO, and MAPPO
  (centralized critic), all with action masking and per-intersection rewards.
- **Composable observations & rewards** — pick named state features
  (including RESCO's `drq_norm`) and reward functions (`queue_length`,
  `presslight`, RESCO's `wait_norm`, ...); required metrics are enabled
  automatically.
- **Safe signal semantics** — four phase-control modes (acyclic, bounded,
  cyclic, RESCO-cyclic) with enforced yellow/all-red clearances and min/max
  green times.
- **Config-driven experiments** — one JSON describes the matrix; the runner
  trains, evaluates on paired seeds, adds baselines, and writes
  `results.json`, `summary.csv` and comparison plots.

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
[CityFlow simulator](https://github.com/cityflow-project/CityFlow), patched
so it builds and runs on Python 3.12+ (upstream does not) — install it from
this repo, not from PyPI/upstream.

SUMO and MOSS are optional; install them only for those backends
(`eclipse-sumo` / `python-moss`).

## Quick start

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

## Documentation

Full docs live in [`docs/`](docs/README.md): architecture, environments,
phase control, states, rewards, metrics, agents, and the experiment
framework.

## Tests

```bash
pytest
```

Backend-specific tests skip automatically when an engine is not installed.
