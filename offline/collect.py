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
import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agent.utils.utils import Utils
from experiments.config import BACKENDS, CONTROL_MODES, SETTING_DEFAULTS, EnvSpec
from offline.flow_randomizer import (
    DEFAULT_JITTER_SIGMA_S,
    DEFAULT_THIN_P,
    DEFAULT_VOLUME_SCALE,
    FlowRandomizer,
)
from offline.trajectory_logger import TrajectoryLogger

PolicyFn = Callable[[dict[str, Any]], np.ndarray]
PolicyFactory = Callable[..., PolicyFn]

#: Subdirectory of ``--out-dir`` holding the materialised demand for each draw.
FLOW_SUBDIR = "flows"

#: Exactly the two filename shapes this module writes into ``FLOW_SUBDIR``.
#:
#: Deliberately narrow, and never a directory wipe: P1's NB3 found that ``ep*.npz`` also
#: matched an unrelated ``epoch_stats.npz``.  A user file that happens to live in
#: ``flows/`` must survive ``--overwrite``.
_DRAW_FILE_RE = re.compile(r"(?:flow|cityflow)_draw\d+\.json")

__all__ = ["POLICIES", "build_parser", "main"]


def _require_lane_arrays(lane_ids: Sequence[str], backend: str) -> None:
    """Abort a collection run whose env reports no lanes.

    Defence in depth, and expected never to fire on CityFlow or SUMO: both
    ``_create_metrics`` implementations return a metrics object unconditionally
    (``envs/sumo_env.py:286-294``), so the ``else {}`` branch that yields empty lane
    dicts (``envs/sumo_env.py:301``, ``envs/moss_env.py:653``) is unreachable there.
    MOSS is not verified. The check costs nothing and the failure it catches is
    expensive: the logger records ``L = 0`` honestly -- refusing to log is a
    collection-policy decision, not a format decision -- and a corpus without lane
    arrays silently violates the C6 reward-agnosticism guarantee, which would only
    surface at P3 after the simulation time has been spent.
    """
    if len(lane_ids) == 0:
        raise ValueError(
            f"backend {backend!r} reported an empty lane set, so the corpus would "
            "carry no lane_vehicle_count / lane_waiting_vehicle_count arrays and no "
            "reward could be recomputed offline (contract C6). This is not expected "
            "on cityflow or sumo, where the metrics pipeline is always on; treat it "
            "as a backend bug or an unsupported scenario rather than a flag you are "
            "missing, and do not collect from this env."
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

    draws = parser.add_mutually_exclusive_group()
    draws.add_argument(
        "--flow-draw",
        type=int,
        default=None,
        help="collect one demand draw; 0 is the nominal source flow (identity). "
        "Omitting every --flow-draw* flag keeps the original behaviour exactly: the "
        "scenario's own flow file, unmodified.",
    )
    draws.add_argument(
        "--flow-draws",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="sweep these explicit draw ids, e.g. --flow-draws 0 1 2",
    )
    draws.add_argument(
        "--flow-draws-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="sweep draw ids over the half-open interval [START, END), matching "
        "Python's range(): '--flow-draws-range 0 100' is 100 draws, ids 0..99",
    )

    parser.add_argument(
        "--flow-jitter-sigma",
        type=float,
        default=DEFAULT_JITTER_SIGMA_S,
        help="departure jitter standard deviation, seconds",
    )
    parser.add_argument(
        "--flow-thin-p",
        type=float,
        default=DEFAULT_THIN_P,
        help="probability of independently dropping each vehicle",
    )
    parser.add_argument(
        "--flow-volume-scale",
        type=float,
        default=DEFAULT_VOLUME_SCALE,
        help="multiplier on the surviving vehicle count",
    )

    return parser


def _resolve_draw_ids(args: argparse.Namespace) -> list[int | None]:
    """Resolve the mutually exclusive draw flags to a list of draw ids.

    Returns ``[None]`` when no flag was given, which is the signal for "use the
    scenario's own flow file and record ``flow_draw`` as absent".
    """
    if args.flow_draw is not None:
        return [int(args.flow_draw)]
    if args.flow_draws:
        return [int(draw_id) for draw_id in args.flow_draws]
    if args.flow_draws_range:
        start, end = args.flow_draws_range
        return list(range(int(start), int(end)))
    return [None]


def _require_cityflow_for_draws(
    backend: str, draw_ids: Sequence[int | None]
) -> None:
    """Refuse a draw sweep on a backend this task did not wire.

    ``render_sumo`` is real and tested, but pointing a SUMO run at a drawn demand also
    needs a generated ``.sumocfg`` (its ``route-files`` and ``begin`` must agree with the
    rendered ``.rou.xml``) and a flag naming the CityFlow-format source, since a
    ``.sumocfg`` does not reference one.  That is P7.3's job; failing loudly here beats a
    flag that silently does nothing.
    """
    if backend == "cityflow" or all(draw_id is None for draw_id in draw_ids):
        return
    raise SystemExit(
        f"--flow-draw/--flow-draws/--flow-draws-range is wired for backend 'cityflow' "
        f"only, got {backend!r}. The randomiser can render SUMO route files "
        "(FlowRandomizer.render_sumo), but collecting from them additionally needs a "
        "generated .sumocfg and a --flow-source-json flag, which this task does not "
        "provide."
    )


def _clear_stale_draw_files(flows_dir: str | Path) -> list[Path]:
    """Delete only the drawn-flow files this module writes; return what was removed.

    Narrow by construction and non-recursive, per P1's NB3: an unrelated file that
    happens to live in ``flows/`` must survive.  ``--overwrite`` clears the corpus via
    ``TrajectoryLogger``, whose glob deliberately does not reach subdirectories, so the
    drawn flows are cleaned here instead.
    """
    directory = Path(flows_dir)
    if not directory.is_dir():
        return []
    removed: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and _DRAW_FILE_RE.fullmatch(path.name):
            path.unlink()
            removed.append(path)
    return removed


def _cityflow_flow_source(config_path: str | Path) -> Path:
    """Absolute path of the flow file a CityFlow sim config points at.

    Mirrors ``envs/cityflow_env.py:82-88``: a relative ``dir`` is resolved against the
    current working directory, and ``flowFile`` hangs off it.
    """
    path = Path(config_path)
    cfg = json.loads(path.read_bytes())
    cfg_dir = cfg.get("dir", "")
    if not os.path.isabs(cfg_dir):
        cfg_dir = str(Path.cwd() / cfg_dir)
    return Path(os.path.normpath(cfg_dir)) / cfg["flowFile"]


def _write_draw_config(
    source_config: str | Path, drawn_flow: str | Path, out_config: str | Path
) -> Path:
    """Write a CityFlow sim config identical to *source_config* but using *drawn_flow*.

    ``dir`` is left pointing at the scenario (so ``roadnetFile`` still resolves) and
    ``flowFile`` becomes a relative path to the drawn file.  It has to be relative:
    CityFlow resolves the flow by plain string concatenation, ``loadFlow(dir + flowFile)``
    (``CityFlow/src/engine/engine.cpp:65``), so an absolute ``flowFile`` would produce
    ``/scenario//abs/path``.  This mirrors the trick ``envs/cityflow_env.py:97-102``
    already uses for the replay-log paths, and it is verified against a real engine --
    see the pre-flight in ``docs/plans/P2.0.md``.
    """
    source = Path(source_config)
    cfg = json.loads(source.read_bytes())
    cfg_dir = cfg.get("dir", "")
    if not os.path.isabs(cfg_dir):
        cfg_dir = str(Path.cwd() / cfg_dir)
    abs_dir = os.path.normpath(cfg_dir)

    cfg["dir"] = abs_dir + "/"
    cfg["flowFile"] = os.path.relpath(str(Path(drawn_flow).resolve()), abs_dir)

    out = Path(out_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return out


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


def _run_metadata(
    args: argparse.Namespace,
    spec: EnvSpec,
    *,
    draw_ids: Sequence[int | None] = (None,),
    flow_source: str | Path | None = None,
    flow_source_sha256: str | None = None,
) -> dict[str, Any]:
    randomised = any(draw_id is not None for draw_id in draw_ids)
    return {
        "flow_draw_ids": [
            None if draw_id is None else int(draw_id) for draw_id in draw_ids
        ],
        "flow_source_path": None if flow_source is None else str(flow_source),
        "flow_source_sha256": flow_source_sha256,
        # The materialised demand for every draw lives here, relative to out_dir, so a
        # corpus can be traced to the exact vehicle list that produced it.
        "flow_dir": FLOW_SUBDIR if randomised else None,
        "flow_randomizer_params": (
            {
                "base_seed": int(args.base_seed),
                "jitter_sigma_s": float(args.flow_jitter_sigma),
                "thin_p": float(args.flow_thin_p),
                "volume_scale": float(args.flow_volume_scale),
            }
            if randomised
            else None
        ),
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

    draw_ids = _resolve_draw_ids(args)
    _require_cityflow_for_draws(spec.backend, draw_ids)
    randomised = any(draw_id is not None for draw_id in draw_ids)

    randomizer: FlowRandomizer | None = None
    flow_source: Path | None = None
    flows_dir = Path(args.out_dir) / FLOW_SUBDIR
    if randomised:
        flow_source = _cityflow_flow_source(spec.paths["config"])
        randomizer = FlowRandomizer(
            flow_source,
            base_seed=int(args.base_seed),
            jitter_sigma_s=float(args.flow_jitter_sigma),
            thin_p=float(args.flow_thin_p),
            volume_scale=float(args.flow_volume_scale),
        )
        # Resolved eagerly and printed, so the half-open [START, END) convention of
        # --flow-draws-range cannot survive to runtime as an ambiguity.
        print(
            f"flow draws: {len(draw_ids)} -> {draw_ids}\n"
            f"  source {flow_source} ({randomizer.n_source_vehicles} vehicles, "
            f"sha256 {randomizer.source_sha256[:12]})",
            flush=True,
        )
        for removed in _clear_stale_draw_files(flows_dir):
            print(f"  removed stale {removed.name}", flush=True)

    from experiments.envs import make_env

    metadata = _run_metadata(
        args,
        spec,
        draw_ids=draw_ids,
        flow_source=flow_source,
        flow_source_sha256=None if randomizer is None else randomizer.source_sha256,
    )

    logger: TrajectoryLogger | None = None
    env: Any = None
    total_steps = 0
    returns: list[float] = []
    episode_index = 0

    try:
        for draw_id in draw_ids:
            draw_spec = spec
            if draw_id is not None:
                assert randomizer is not None
                entries, provenance = randomizer.draw(draw_id)
                drawn_flow = randomizer.render_cityflow(
                    entries, flows_dir / f"flow_draw{draw_id}.json"
                )
                draw_config = _write_draw_config(
                    spec.paths["config"],
                    drawn_flow,
                    flows_dir / f"cityflow_draw{draw_id}.json",
                )
                draw_spec = dataclasses.replace(
                    spec, paths={**spec.paths, "config": str(draw_config)}
                )
                print(
                    f"draw {draw_id}: {provenance.n_vehicles} vehicles "
                    f"(source {randomizer.n_source_vehicles}) -> {drawn_flow.name}",
                    flush=True,
                )

            # A fresh env per draw is mandatory, not a precaution: CityFlow reads its
            # flow file once in the engine constructor and Engine::reset() never
            # re-reads it (CityFlow/src/engine/engine.cpp:65,754).
            env = make_env(draw_spec)
            rng = np.random.default_rng(int(args.base_seed))
            policy = POLICIES[args.policy](env, args, rng)

            if logger is None:
                logger = TrajectoryLogger(
                    env,
                    args.out_dir,
                    run_metadata=metadata,
                    overwrite=bool(args.overwrite),
                )
            else:
                logger.rebind_env(env)

            for index in range(int(args.episodes)):
                # Seeds restart per draw so the draw is the only variable across draws.
                engine_seed = int(args.base_seed) + index
                info = env.reset(seed=engine_seed)
                logger.on_reset(info, engine_seed=engine_seed, flow_draw=draw_id)
                if episode_index == 0:
                    _require_lane_arrays(logger.lane_ids, draw_spec.backend)

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
                episode_index += 1
                returns.append(episode_return)
                draw_label = "" if draw_id is None else f"  draw={draw_id}"
                print(
                    f"episode {index + 1}/{args.episodes}{draw_label}  "
                    f"seed={engine_seed}  steps={steps}  "
                    f"return={episode_return:.3f}  -> {path.name}",
                    flush=True,
                )

            close = getattr(env, "close", None)
            if callable(close):
                close()
            env = None

        mean_return = float(np.mean(returns)) if returns else 0.0
        assert logger is not None
        print(
            f"done: {len(returns)} episodes over {len(draw_ids)} draw(s) {draw_ids}, "
            f"{total_steps} steps, mean return {mean_return:.3f}, "
            f"manifest {logger.manifest_path}",
            flush=True,
        )
    finally:
        close = getattr(env, "close", None) if env is not None else None
        if callable(close):
            close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
