"""CLI for one offline-corpus collection run.

Drives an existing behaviour policy over an existing env and hands every callback to
:class:`offline.trajectory_logger.TrajectoryLogger`.  The env is built by
``experiments.envs.make_env`` -- reused, never reimplemented -- so the backend kwarg
table lives in exactly one place.

    python -m offline.collect \\
      --env-config configs/sim/cityflow1x1.json --backend cityflow \\
      --policy maxpressure \\
      --episodes 20 --base-seed 1000 \\
      --max-steps 360 --delta-time 10 \\
      --out-dir datasets/hz1x1_maxpressure

Behaviour policies live in the :data:`POLICIES` registry so the ladder can grow
without touching the CLI.  P1 ships four entries; ``fixedtime`` and ``dqn`` belong to
P2.5 and are deliberately absent -- there is no fixed-time controller in this repo,
and writing one means cycle-length and phase-split decisions that need their own task.

Episode ``i`` uses ``engine_seed = base_seed + i``.  ``flow_draw`` is always ``None``
here: the flow randomiser is P2, and this module only plumbs the field through.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agent.utils.utils import Utils
from experiments.config import BACKENDS, CONTROL_MODES, SETTING_DEFAULTS, EnvSpec
from offline.trajectory_logger import TrajectoryLogger

PolicyFn = Callable[[dict[str, Any]], np.ndarray]
PolicyFactory = Callable[..., PolicyFn]

__all__ = ["POLICIES", "build_parser", "main"]


def _require_lane_arrays(lane_ids: Sequence[str], backend: str) -> None:
    """Abort a collection run whose env reports no lanes.

    SUMO and MOSS return empty ``lane_vehicle_count`` dicts when the metrics pipeline
    is off (``envs/sumo_env.py:301``, ``envs/moss_env.py:653``).  The logger records
    ``L = 0`` honestly -- refusing to log is a collection-policy decision, not a
    format decision -- so the guard belongs here.  A corpus without lane arrays
    silently violates the C6 reward-agnosticism guarantee and would only be discovered
    at P3, after the simulation time has already been spent.
    """
    if len(lane_ids) == 0:
        raise ValueError(
            f"backend {backend!r} reported an empty lane set, so the corpus would "
            "carry no lane_vehicle_count / lane_waiting_vehicle_count arrays and no "
            "reward could be recomputed offline (contract C6). Enable the metrics "
            "pipeline with --metrics naming a registered metric (for example "
            "--metrics average_travel_time; queue_length is a REWARD function, not a "
            "metric) and collect again."
        )


def _intersection_action_counts(env: Any) -> tuple[list[str], list[int]]:
    """Intersection ids in ``env.intersections`` order plus their action counts."""
    intersections = list(env.intersections)
    ix_ids = [str(ix.id) for ix in intersections]
    counts = Utils.infer_action_counts(
        getattr(env, "action_space", None), intersections
    )
    return ix_ids, [int(c) for c in counts]


def _random_legal_action(
    payload: dict[str, Any], n_actions: int, rng: np.random.Generator
) -> int:
    valid = Utils.extract_valid_actions(payload, n_actions)
    return int(valid[int(rng.integers(len(valid)))])


def _make_maxpressure(env: Any, args: argparse.Namespace, rng: np.random.Generator) -> PolicyFn:
    """MaxPressure baseline; ``act(info)`` only, and not a ``BaseAgent`` subclass."""
    from algorithms.max_pressure import MaxPressureAgent

    agent = MaxPressureAgent(env)
    return agent.act


def _make_random(env: Any, args: argparse.Namespace, rng: np.random.Generator) -> PolicyFn:
    """Uniform over each intersection's currently legal actions, seeded."""
    ix_ids, n_actions = _intersection_action_counts(env)

    def act(info: dict[str, Any]) -> np.ndarray:
        payloads = Utils.extract_per_intersection_info(info, ix_ids)
        return np.asarray(
            [
                _random_legal_action(payloads[ix_id], n_actions[i], rng)
                for i, ix_id in enumerate(ix_ids)
            ],
            dtype=np.int64,
        )

    return act


def _make_mappo(env: Any, args: argparse.Namespace, rng: np.random.Generator) -> PolicyFn:
    """Greedy MAPPO: ``act(info, explore=False, update_memory=False)``."""
    from agent.MAPPOAgent import MAPPOAgent

    agent = MAPPOAgent(env, device=args.device, seed=args.base_seed)
    if args.checkpoint:
        agent.load(args.checkpoint)
    else:
        print(
            "WARNING: --policy mappo without --checkpoint collects from an "
            "untrained network; the corpus will be near-random.",
            flush=True,
        )

    def act(info: dict[str, Any]) -> np.ndarray:
        return agent.act(info, explore=False, update_memory=False)

    return act


def _make_mappo_eps(env: Any, args: argparse.Namespace, rng: np.random.Generator) -> PolicyFn:
    """MAPPO with per-intersection epsilon-substitution from a seeded RNG."""
    greedy = _make_mappo(env, args, rng)
    ix_ids, n_actions = _intersection_action_counts(env)
    epsilon = float(args.epsilon)

    def act(info: dict[str, Any]) -> np.ndarray:
        actions = np.asarray(greedy(info), dtype=np.int64).reshape(-1).copy()
        payloads = Utils.extract_per_intersection_info(info, ix_ids)
        for i, ix_id in enumerate(ix_ids):
            if rng.random() < epsilon:
                actions[i] = _random_legal_action(payloads[ix_id], n_actions[i], rng)
        return actions

    return act


POLICIES: dict[str, PolicyFactory] = {
    "maxpressure": _make_maxpressure,
    "random": _make_random,
    "mappo": _make_mappo,
    "mappo_eps": _make_mappo_eps,
}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m offline.collect`` argument parser.

    Defaults for the cell settings come from ``experiments.config.SETTING_DEFAULTS``
    so the collector and the experiment harness cannot drift apart.
    """
    parser = argparse.ArgumentParser(
        prog="python -m offline.collect",
        description="Collect an offline trajectory corpus from an existing policy.",
    )

    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument(
        "--env-config",
        help="simulator config path (cityflow .json / sumo .sumocfg)",
    )
    parser.add_argument("--map-file", help="MOSS map protobuf")
    parser.add_argument("--person-file", help="MOSS person protobuf")

    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--checkpoint", help="checkpoint for the mappo policies")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="per-intersection substitution probability for --policy mappo_eps",
    )
    parser.add_argument("--device", default=SETTING_DEFAULTS["device"])

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard a previous run in --out-dir; without it a populated out-dir is "
        "refused, because restarting there would drop earlier episodes from the "
        "manifest and orphan their .npz files",
    )

    parser.add_argument(
        "--max-steps", type=int, default=SETTING_DEFAULTS["max_steps"]
    )
    parser.add_argument(
        "--delta-time", type=int, default=SETTING_DEFAULTS["delta_time"]
    )
    parser.add_argument(
        "--control-mode",
        choices=sorted(CONTROL_MODES),
        default=SETTING_DEFAULTS["control_mode"],
    )
    parser.add_argument(
        "--global-reward-fn", default=SETTING_DEFAULTS["global_reward_fn"]
    )
    parser.add_argument(
        "--local-reward-fn", default=SETTING_DEFAULTS["local_reward_fn"]
    )
    parser.add_argument(
        "--global-reward-weight",
        type=float,
        default=SETTING_DEFAULTS["global_reward_weight"],
    )
    parser.add_argument(
        "--state-features",
        nargs="+",
        default=list(SETTING_DEFAULTS["state_features"]),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=SETTING_DEFAULTS["metrics"],
        help="explicit metric list; required on sumo/moss for the lane arrays",
    )
    parser.add_argument(
        "--thread-num", type=int, default=SETTING_DEFAULTS["thread_num"]
    )
    parser.add_argument("--libsumo", action="store_true", default=SETTING_DEFAULTS["libsumo"])

    return parser


def _build_env_spec(args: argparse.Namespace) -> EnvSpec:
    """Assemble the ``EnvSpec`` that ``experiments.envs.make_env`` consumes."""
    settings: dict[str, Any] = dict(SETTING_DEFAULTS)
    settings.update(
        {
            "max_steps": int(args.max_steps),
            "delta_time": int(args.delta_time),
            "control_mode": args.control_mode,
            "global_reward_fn": args.global_reward_fn,
            "local_reward_fn": args.local_reward_fn,
            "global_reward_weight": float(args.global_reward_weight),
            "state_features": list(args.state_features),
            "metrics": list(args.metrics) if args.metrics else None,
            "thread_num": int(args.thread_num),
            "gui": False,
            "libsumo": bool(args.libsumo),
        }
    )

    paths: dict[str, str] = {}
    if args.backend == "moss":
        if not (args.map_file and args.person_file):
            raise SystemExit("--backend moss requires --map-file and --person-file")
        paths["map_file"] = str(Path(args.map_file).resolve())
        paths["person_file"] = str(Path(args.person_file).resolve())
        scenario_id = Path(args.map_file).stem
    else:
        if not args.env_config:
            raise SystemExit(f"--backend {args.backend} requires --env-config")
        paths["config"] = str(Path(args.env_config).resolve())
        scenario_id = Path(args.env_config).stem

    return EnvSpec(
        id=scenario_id, backend=args.backend, paths=paths, settings=settings
    )


def _run_metadata(args: argparse.Namespace, spec: EnvSpec) -> dict[str, Any]:
    return {
        "scenario_id": spec.id,
        "backend": spec.backend,
        "env_paths": dict(spec.paths),
        "behavior_policy": args.policy,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": (
            _file_sha256(args.checkpoint) if args.checkpoint else None
        ),
        "epsilon": float(args.epsilon) if args.policy == "mappo_eps" else None,
        "episodes": int(args.episodes),
        "base_seed": int(args.base_seed),
        "delta_time": spec.settings["delta_time"],
        "max_steps": spec.settings["max_steps"],
        "control_mode": spec.settings["control_mode"],
        "state_features": list(spec.settings["state_features"]),
        "global_reward_fn": spec.settings["global_reward_fn"],
        "local_reward_fn": spec.settings["local_reward_fn"],
        "global_reward_weight": spec.settings["global_reward_weight"],
        "metrics": spec.settings["metrics"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one collection run; returns a process exit code."""
    args = build_parser().parse_args(argv)

    Utils.seed_everything(int(args.base_seed))
    spec = _build_env_spec(args)

    from experiments.envs import make_env

    env = make_env(spec)
    try:
        rng = np.random.default_rng(int(args.base_seed))
        policy = POLICIES[args.policy](env, args, rng)
        logger = TrajectoryLogger(
            env,
            args.out_dir,
            run_metadata=_run_metadata(args, spec),
            overwrite=bool(args.overwrite),
        )

        total_steps = 0
        returns: list[float] = []

        for index in range(int(args.episodes)):
            engine_seed = int(args.base_seed) + index
            info = env.reset(seed=engine_seed)
            logger.on_reset(info, engine_seed=engine_seed, flow_draw=None)
            if index == 0:
                _require_lane_arrays(logger.lane_ids, spec.backend)

            episode_return = 0.0
            steps = 0
            for _ in range(env.max_steps):
                action = policy(info)
                logger.on_action(info, action)
                reward, terminated, truncated, info = env.step(action)
                logger.on_step_result(reward, terminated, truncated, info)
                episode_return += float(Utils.scalar_reward(reward))
                steps += 1
                if terminated or truncated:
                    break

            path = logger.finalize_episode()
            total_steps += steps
            returns.append(episode_return)
            print(
                f"episode {index + 1}/{args.episodes}  seed={engine_seed}  "
                f"steps={steps}  return={episode_return:.3f}  -> {path.name}",
                flush=True,
            )

        mean_return = float(np.mean(returns)) if returns else 0.0
        print(
            f"done: {len(returns)} episodes, {total_steps} steps, "
            f"mean return {mean_return:.3f}, manifest {logger.manifest_path}",
            flush=True,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
