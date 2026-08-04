"""Run the experiment matrix: train + evaluate every agent on every
environment, for every seed, and aggregate the results.

The per-cell loop is deliberately tiny because the environments compute
their own rewards: ``act(info) -> step(action) -> observe(next_info, reward)``
with the env's ``info`` dict passed to the agent unchanged (agents read
``info["intersections"][id]`` directly).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.config import AgentSpec, EnvSpec, ExperimentConfig
from experiments.envs import backend_ready, make_env
from experiments.registry import BASELINE_LABELS, build_agent

# (metric name, higher_is_better) — the columns reported everywhere.
METRICS_SPEC: tuple[tuple[str, bool], ...] = (
    ("episode_reward", True),
    ("average_travel_time", False),
    ("final_vehicle_count", False),
    ("average_waiting_queue", False),
)

ChooseAction = Callable[[Any, dict[str, Any]], np.ndarray]


# Torch threads per cell. ``run_cell`` is the unit of work on BOTH paths -- one
# process per cell under ``ProcessPoolExecutor``, and the main process when
# ``workers=1`` -- so pinning here covers both. The value is 1 primarily for LIVENESS,
# not speed: see limit_torch_threads() below for why an unpinned pooled worker can wedge
# the whole suite indefinitely. The small MLPs trained here also do not pay for a thread
# pool, so pinning is a free speedup on top.
#
# Trustworthy timing (single session, clean shell, both runs pinned), recorded in
# docs/notes/P0.3_spawn_attempt.md section 5:
#     199.2 s at workers=1  ->  50.2 s at workers=6   (~3.97x)
# Only same-session numbers are quotable as a ratio.
#
# RETIRED 2026-08-03 -- cross-session arithmetic, kept visible so the correction is
# auditable but NOT to be cited. The table below divided a pinned run by an *unpinned*
# baseline measured on a different day-state, which the Decisions Log banned; the ratios
# (1.37x / 4.45x / 5.80x) are therefore untrustworthy even though the speedup shape holds.
#     Measured 2026-07-27, 16 cores, reduced ``p0_baselines`` (6 cells,
#     train_episodes=10, max_steps=360):
#       workers   unpinned    pinned (OMP=MKL=OPENBLAS=1)
#             1    339.7 s     247.7 s   (1.37x  retired)
#             3   1165.3 s      76.3 s   (4.45x  retired)
#             6   >=1200 s      58.6 s   (5.80x  retired)
# The mechanism is unchanged: torch's default pool takes one thread per core, so N workers
# ask for 16N threads on a 16-core box, and unpinned parallelism both slows down and, worse,
# can hang.
CELL_TORCH_THREADS: int = 1


def limit_torch_threads(n_threads: int = CELL_TORCH_THREADS) -> int:
    """Pin torch's intra-op pool for the calling process; return the count in effect.

    LIVENESS, NOT SPEED, is the primary reason this call exists. On the pooled path a
    ``fork()``ed child that enters an OpenMP parallel region with ``nthreads > 1`` waits
    forever on team threads the ``fork()`` never duplicated. ``run_matrix`` collects with
    ``as_completed`` + ``future.result()`` and NO timeout, so that deadlock surfaces as a
    SILENT WEDGE with no failure message -- it freezes the whole test suite and the
    PostToolUse guard, not just the run. The P0.3-fix reviewer reproduced it on demand
    (``exit=124`` at the 120 s and 150 s kill caps against unpinned code; 14 passed and
    10/10 clean runs against the committed, pinned code).

    ORDERING CONSTRAINT that follows: anything added to ``run_cell`` *before* this call
    runs in the child's still-unpinned window. ``backend_ready()`` already runs there and
    imports the native backend -- confirmed safe for CityFlow, UNPROBED for libsumo and
    moss (relevant at P7). What reintroduces the hang is REMOVING this call, MOVING it
    later, or adding child-side work ahead of it -- NOT making it conditional on
    ``workers > 1``: the pool, and therefore the ``fork()``, exists only under
    ``workers > 1`` (see ``run_matrix``), so such a condition would keep the pin exactly
    where the hazard is.

    SCOPE, precisely: the ``ProcessPoolExecutor`` is built only under ``workers > 1``; the
    sequential (``workers=1``) path never forks, so the liveness argument does not apply to
    it. There the pin is a performance and determinism choice, and -- deliberate side
    effect -- it pins the *calling* process, which is what running a cell in-process asks
    for.

    Speed is a secondary benefit (these small MLPs do not want a 16-thread pool); the
    trustworthy single-session figure is 199.2 s -> 50.2 s (~3.97x at workers=6), and the
    once-quoted 1.37x / 5.80x are retired cross-session ratios (see the note above
    ``CELL_TORCH_THREADS``). Return ``torch.get_num_threads()`` read back after the set, not
    the value requested, so a caller -- or a test running this inside a worker -- observes
    what torch actually did. torch is imported here rather than at module scope on purpose:
    ``experiments`` has no module-level torch import, which keeps ``--dry-run`` instant and
    turns a missing backend into a ``skipped`` cell instead of a crash. A forked worker
    inherits the parent's thread count and has to lower it itself.
    """
    import torch

    torch.set_num_threads(int(n_threads))
    return int(torch.get_num_threads())


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_component(name: str) -> str:
    raw = str(name)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
    if not safe:
        safe = "unnamed"
    # Sanitization can map distinct ids to the same string ("env.a" and
    # "env_a"); a short hash of the raw id keeps checkpoint paths unique.
    if safe != raw:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return safe


def checkpoint_path(root: str | Path, env_id: str, agent_id: str, seed: int) -> Path:
    """Stable checkpoint path for one trained policy in one (env, seed) cell."""
    filename = (
        f"{_safe_component(env_id)}__{_safe_component(agent_id)}__seed{int(seed)}.pt"
    )
    return Path(root) / filename


def _episode_queue(env: Any, info: dict[str, Any]) -> float:
    """Total vehicles waiting on incoming lanes across all intersections."""
    lane_waiting = info.get("lane_waiting_vehicle_count", {}) or {}
    total = 0.0
    for ix in env.intersections:
        total += sum(float(lane_waiting.get(lid, 0.0)) for lid in ix.incoming_lanes)
    return total


def evaluate_policy(
    env: Any,
    choose_action: ChooseAction,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Roll out *choose_action* for *episodes* and average the metrics."""
    max_steps = int(env.max_steps)
    rewards: list[float] = []
    travel: list[float] = []
    queue: list[float] = []
    vehicles: list[float] = []

    for ep in range(int(episodes)):
        info = env.reset(seed=seed + ep)
        reward_sum = 0.0
        travel_samples: list[float] = []
        queue_samples: list[float] = []
        last_vehicle_count = 0.0

        for _ in range(max_steps):
            action = choose_action(env, info)
            reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            travel_samples.append(float(info.get("average_travel_time", 0.0)))
            queue_samples.append(_episode_queue(env, info))
            last_vehicle_count = float(info.get("vehicle_count", 0.0))
            if terminated or truncated:
                break

        rewards.append(reward_sum)
        travel.append(_mean(travel_samples))
        queue.append(_mean(queue_samples))
        vehicles.append(last_vehicle_count)

    return {
        "episode_reward": _mean(rewards),
        "average_travel_time": _mean(travel),
        "final_vehicle_count": _mean(vehicles),
        "average_waiting_queue": _mean(queue),
    }


def _build_agent_for_env(agent_spec: AgentSpec, env_spec: EnvSpec, seed: int) -> Any:
    settings = env_spec.settings
    env = make_env(env_spec)
    try:
        return build_agent(
            agent_spec.type,
            env,
            agent_spec.params,
            device=settings["device"],
            seed=seed,
            train_episodes=settings["train_episodes"],
            max_steps=settings["max_steps"],
        )
    finally:
        env.close()


def _train_agent(agent_spec: AgentSpec, env_spec: EnvSpec, seed: int) -> tuple[Any, list[float]]:
    """Build and train one agent on a fresh env instance, then return it."""
    settings = env_spec.settings
    env = make_env(env_spec)
    try:
        agent = build_agent(
            agent_spec.type,
            env,
            agent_spec.params,
            device=settings["device"],
            seed=seed,
            train_episodes=settings["train_episodes"],
            max_steps=settings["max_steps"],
        )
        train_returns: list[float] = []
        max_steps = int(env.max_steps)
        for ep in range(int(settings["train_episodes"])):
            info = env.reset(seed=seed + ep)
            reward_sum = 0.0
            for _ in range(max_steps):
                action = agent.act(info, explore=True)
                reward, terminated, truncated, next_info = env.step(action)
                reward_sum += float(reward)
                agent.observe(next_info, reward, bool(terminated), bool(truncated))
                info = next_info
                if terminated or truncated:
                    break
            train_returns.append(reward_sum)
        return agent, train_returns
    finally:
        env.close()


def _load_agent(agent_spec: AgentSpec, env_spec: EnvSpec, seed: int, path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    agent = _build_agent_for_env(agent_spec, env_spec, seed)
    agent.load(str(path))
    return agent


def _save_agent(agent: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))


def _agent_chooser(agent: Any) -> ChooseAction:
    def choose(_env: Any, info: dict[str, Any]) -> np.ndarray:
        # update_memory=False: an eval rollout must not store transitions that a
        # later observe() would mispair (all three agents honour this kwarg).
        return agent.act(info, explore=False, update_memory=False)

    return choose


def _baseline_chooser(name: str, env: Any, action_seed: int) -> ChooseAction:
    """Build an eval action selector for a baseline.

    ``action_seed`` seeds only the policy's own randomness (the random
    baseline's action RNG); it is independent of the env-reset seed so every
    policy in a cell can be evaluated on the *same* traffic realization.
    """
    if name == "random":
        rng = np.random.default_rng(action_seed)

        def choose(env: Any, info: dict[str, Any]) -> np.ndarray:
            per_ix = info.get("intersections", {}) or {}
            actions: list[int] = []
            for ix in env.intersections:
                avail = (per_ix.get(ix.id, {}) or {}).get("avail_actions")
                if not avail:
                    avail = list(range(ix.num_phases))
                actions.append(int(rng.choice(avail)))
            return np.asarray(actions, dtype=np.int64)

        return choose

    if name == "max_pressure":
        from algorithms.max_pressure import MaxPressureAgent

        agent = MaxPressureAgent(env)

        def choose(_env: Any, info: dict[str, Any]) -> np.ndarray:
            return np.asarray(agent.act(info), dtype=np.int64).reshape(-1)

        return choose

    raise ValueError(f"unsupported baseline: {name}")  # pragma: no cover


def run_cell(
    env_spec: EnvSpec,
    agents: tuple[AgentSpec, ...],
    seed: int,
    verbose: bool = True,
    checkpoint_dir: str | Path | None = None,
    from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train + evaluate every agent (and the baselines) on one (env, seed).

    Returns a result dict with ``status`` in ``{"ok", "partial", "skipped",
    "error"}``; a missing backend yields ``skipped`` rather than raising.
    Each policy (agent or baseline) is isolated: one failing policy is
    recorded under ``failures`` and the rest of the cell still counts
    (``partial``); ``error`` means no policy succeeded.
    """
    tag = f"{env_spec.id}@seed{seed}"
    base = {"env_id": env_spec.id, "backend": env_spec.backend, "seed": int(seed)}

    settings = env_spec.settings
    ready, reason = backend_ready(
        env_spec.backend, env_spec.paths, libsumo=settings["libsumo"]
    )
    if not ready:
        _log(verbose, f"[skip ] {tag}: {reason}")
        return {**base, "status": "skipped", "reason": reason, "policies": {}, "timings": {}}

    # Pin before ``build_agent`` imports torch -- it enters lazily, inside this
    # call. After the skip check on purpose: a cell with no backend must not pay
    # for a torch import it will never use.
    limit_torch_threads()

    # Seed convention (offsets keep streams disjoint for typical episode counts):
    # train uses seed+ep; EVERY policy is evaluated on the same env-reset seed
    # (eval_seed) so the comparison is paired; a baseline's own action RNG gets a
    # separate offset that does not perturb the env episode sequence.
    eval_seed = seed + 10_000
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    load_root = Path(from_checkpoint) if from_checkpoint is not None else None
    policies: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    failures: dict[str, str] = {}

    for agent_spec in agents:
        try:
            train_returns: list[float] = []
            train_sec: float | None = None
            load_sec: float | None = None
            checkpoint_file: Path | None = None
            loaded_from_checkpoint = load_root is not None

            if load_root is not None:
                checkpoint_file = checkpoint_path(load_root, env_spec.id, agent_spec.id, seed)
                _log(verbose, f"[load ] {tag} :: {agent_spec.id} <- {checkpoint_file}")
                started = time.perf_counter()
                agent = _load_agent(agent_spec, env_spec, seed, checkpoint_file)
                load_sec = time.perf_counter() - started
            else:
                _log(verbose, f"[train] {tag} :: {agent_spec.id} ({agent_spec.type})")
                started = time.perf_counter()
                agent, train_returns = _train_agent(agent_spec, env_spec, seed)
                train_sec = time.perf_counter() - started

                if checkpoint_root is not None:
                    checkpoint_file = checkpoint_path(
                        checkpoint_root, env_spec.id, agent_spec.id, seed
                    )
                    _save_agent(agent, checkpoint_file)

            eval_env = make_env(env_spec)
            try:
                started = time.perf_counter()
                metrics = evaluate_policy(
                    eval_env, _agent_chooser(agent), settings["eval_episodes"], eval_seed
                )
                eval_sec = time.perf_counter() - started
            finally:
                eval_env.close()

            policy_payload = {
                "kind": "agent",
                "type": agent_spec.type,
                "params": dict(agent_spec.params),
                "train_returns": train_returns,
                "metrics": metrics,
            }
            if checkpoint_file is not None:
                policy_payload["checkpoint_path"] = str(checkpoint_file)
                policy_payload["checkpoint_loaded"] = loaded_from_checkpoint
            policies[agent_spec.id] = policy_payload

            timing_payload: dict[str, Any] = {"train_sec": train_sec, "eval_sec": eval_sec}
            if load_sec is not None:
                timing_payload["load_sec"] = load_sec
            timings[agent_spec.id] = timing_payload
        except Exception as exc:
            _log(
                verbose,
                f"[error] {tag} :: {agent_spec.id}: {type(exc).__name__}: {exc}",
            )
            failures[agent_spec.id] = f"{type(exc).__name__}: {exc}"

    for index, baseline in enumerate(settings["compare_with"]):
        label = BASELINE_LABELS[baseline]
        try:
            _log(verbose, f"[eval ] {tag} :: {label}")
            action_seed = seed + 50_000 + index
            eval_env = make_env(env_spec)
            try:
                chooser = _baseline_chooser(baseline, eval_env, action_seed)
                started = time.perf_counter()
                metrics = evaluate_policy(
                    eval_env, chooser, settings["eval_episodes"], eval_seed
                )
                eval_sec = time.perf_counter() - started
            finally:
                eval_env.close()

            policies[label] = {"kind": "baseline", "baseline": baseline, "metrics": metrics}
            timings[label] = {"train_sec": None, "eval_sec": eval_sec}
        except Exception as exc:
            _log(verbose, f"[error] {tag} :: {label}: {type(exc).__name__}: {exc}")
            failures[label] = f"{type(exc).__name__}: {exc}"

    result: dict[str, Any] = {**base, "policies": policies, "timings": timings}
    if failures:
        result["failures"] = failures
        result["reason"] = "; ".join(
            f"{label}: {message}" for label, message in failures.items()
        )
        result["status"] = "partial" if policies else "error"
    else:
        result["status"] = "ok"
    return result


def aggregate(cell_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Group successful policies by (env_id, policy) and reduce over seeds.

    ``partial`` cells contribute their successful policies; ``skipped`` and
    ``error`` cells contribute nothing.
    """
    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for cell in cell_results:
        if cell["status"] not in ("ok", "partial"):
            continue
        env_acc = acc.setdefault(cell["env_id"], {})
        for label, payload in cell["policies"].items():
            label_acc = env_acc.setdefault(label, {})
            for metric, value in payload["metrics"].items():
                label_acc.setdefault(metric, []).append(float(value))

    out: dict[str, Any] = {}
    for env_id, labels in acc.items():
        out[env_id] = {}
        for label, metrics in labels.items():
            out[env_id][label] = {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
                for metric, values in metrics.items()
            }
    return out


def run_matrix(
    config: ExperimentConfig,
    *,
    workers: int = 1,
    verbose: bool = True,
    from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the whole matrix and return a serializable report dict."""
    load_checkpoint_dir = Path(from_checkpoint).resolve() if from_checkpoint is not None else None
    save_checkpoint_dir = None if load_checkpoint_dir is not None else config.checkpoint_dir
    cells = [
        (env_spec, config.agents, seed)
        for env_spec in config.environments
        for seed in config.seeds
    ]

    results: list[dict[str, Any]] = []
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {
                pool.submit(
                    run_cell,
                    env_spec,
                    agents,
                    seed,
                    False,
                    save_checkpoint_dir,
                    load_checkpoint_dir,
                ): (env_spec, seed)
                for env_spec, agents, seed in cells
            }
            for future in as_completed(futures):
                env_spec, seed = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    # run_cell catches Python errors itself; this only fires on a
                    # native crash (segfault -> BrokenProcessPool). Degrade to an
                    # error cell so one bad worker doesn't discard the whole matrix.
                    results.append(
                        {
                            "env_id": env_spec.id,
                            "backend": env_spec.backend,
                            "seed": int(seed),
                            "status": "error",
                            "reason": f"worker crashed: {type(exc).__name__}: {exc}",
                            "policies": {},
                            "timings": {},
                        }
                    )
    else:
        for env_spec, agents, seed in cells:
            results.append(
                run_cell(
                    env_spec,
                    agents,
                    seed,
                    verbose,
                    save_checkpoint_dir,
                    load_checkpoint_dir,
                )
            )

    env_order = {env.id: i for i, env in enumerate(config.environments)}
    results.sort(key=lambda cell: (env_order.get(cell["env_id"], 0), cell["seed"]))

    return {
        "name": config.name,
        "seeds": list(config.seeds),
        "source": str(config.source_path),
        "checkpoint_dir": str(config.checkpoint_dir) if config.checkpoint_dir is not None else None,
        "from_checkpoint": str(load_checkpoint_dir) if load_checkpoint_dir is not None else None,
        "environments": [
            {"id": e.id, "backend": e.backend, "paths": e.paths, "settings": e.settings}
            for e in config.environments
        ],
        "agents": [{"id": a.id, "type": a.type, "params": a.params} for a in config.agents],
        "cells": results,
        "aggregated": aggregate(results),
    }
