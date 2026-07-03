# Agents

Trainable agents live in `agent/`; non-learned controllers in `algorithms/`.
All agents share the same interaction contract — they consume the env's
`info` dict directly:

```python
action = agent.act(info, explore=True)                  # one action per intersection
reward, terminated, truncated, next_info = env.step(action)
agent.observe(next_info, reward, terminated, truncated)
```

Common conventions:

- **Per-intersection state** is read from
  `info["intersections"][id]["state"]`; network input sizes are inferred
  lazily from the first state seen.
- **Action masking**: only actions in `avail_actions` are ever selected, and
  the masks are stored with transitions so bootstrapping/updates never credit
  illegal actions.
- **Rewards**: when `info["intersections"][id]["reward"]` is present (a
  `local_reward_fn` is configured), each learner trains on its own local
  reward; otherwise the global scalar reward is shared.
- **Truncation vs termination**: only true terminations zero the bootstrap;
  truncated episodes (time limit) bootstrap normally.
- **Evaluation**: `act(info, explore=False, update_memory=False)` selects
  greedily/deterministically and stores nothing, so eval rollouts cannot
  corrupt the replay/rollout buffers.
- `save(path)` / `load(path)` persist checkpoints (`.pt`).
- `BaseAgent.train(episodes)` provides a simple built-in training loop.

Constructor pattern: `Agent(env, ..., device=None, seed=None)`. `device`
accepts `"cpu"` / `"cuda"` / `"mps"` (default: auto-detect CUDA).

## DQNAgent (`agent/DQNAgent.py`, type `dqn`)

Independent DQN: one Q-network + replay buffer **per intersection**
(2×128 ReLU MLP by default). Epsilon-greedy exploration annealed linearly
from `epsilon_start` to `epsilon_end` over `epsilon_decay_steps`; target
network synced every `target_update_interval` updates; next-state action
masks stored so the bootstrap max never picks an illegal action.

Key hyperparameters (defaults): `lr=1e-3`, `gamma=0.99`, `batch_size=64`,
`replay_size=100_000`, `min_replay_size=1_000`,
`target_update_interval=200`, `hidden_dim=128`, `epsilon_start=1.0`,
`epsilon_end=0.05`, `epsilon_decay_steps=50_000`.

## IPPOAgent (`agent/IPPOagent.py`, type `ippo`)

Independent PPO: one actor-critic (shared tanh trunk, policy + value heads)
per intersection, trained on its own rollout with GAE, clipped surrogate
objective, entropy bonus and gradient clipping.

Key hyperparameters (defaults): `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`,
`clip_ratio=0.2`, `entropy_coef=0.01`, `value_coef=0.5`, `update_epochs=4`,
`minibatch_size=128`, `rollout_size=1024`, `hidden_dim=128`,
`max_grad_norm=0.5`.

## MAPPOAgent (`agent/MAPPOAgent.py`, type `mappo`)

Multi-agent PPO with **centralized training, decentralized execution**: one
actor per intersection, plus a central critic that sees the concatenated
global state and outputs one value per agent. Critic inputs and value
targets are normalised with running Welford statistics (frozen during
evaluation); advantages are normalised across agents. Same PPO
hyperparameter set as IPPO.

## DQNAgentPFRL (`agent/DQNAgentPFRL.py`)

RESCO-parity independent DQN built on the `pfrl` library — the exact learner
RESCO uses for its IDQN: one `pfrl.agents.DQN` per intersection with RESCO's
hyperparameters and network (`Conv2d(1, 64, 2×2)` over the `drq_norm`
observation, two 64-unit FC layers). Constraints:

- requires `state_features=["drq_norm"]` (SUMO or MOSS) and a phase control
  whose actions are always all available (`resco_cyclic`);
- pfrl couples exploration and learning: `act(explore=True)` must be
  followed by `observe()`; `act(explore=False)` is greedy and stores nothing.

Not part of the experiment registry; used for faithful RESCO comparisons.

## Baselines

### MaxPressureAgent (`algorithms/max_pressure.py`)

Greedy max-pressure controller: for every intersection, picks the phase
whose active road links have the highest total pressure
(Σ incoming-lane counts − Σ outgoing-lane counts). Under cyclic control
modes it emits keep/switch actions toward that phase instead of absolute
indices. Stateless, no training.

### Random

A uniform-random choice among the currently available actions (implemented
inside the experiment runner with its own seeded RNG, independent of the env
seed so all policies are evaluated on identical traffic).

Both baselines are available in experiment configs via
`"compare_with": ["random", "max_pressure"]`.
