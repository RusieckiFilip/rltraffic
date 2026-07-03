"""CLI entry point for the experiment framework.

    python experiments/run.py <config.json> [--workers N] [--dry-run] [--no-plot]
                              [--from-checkpoint DIR]

``--dry-run`` validates the config and prints the planned matrix without
importing any simulator backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.config import ExperimentConfig, load_config  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agents x environments traffic-signal experiment matrix.",
    )
    parser.add_argument("config", type=Path, help="Path to the experiment JSON config.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel processes over (env, seed) cells (default 1 = sequential).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the planned matrix without running.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing comparison plots.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        metavar="DIR",
        help="Load per-cell agent checkpoints from DIR and skip training.",
    )
    return parser.parse_args(argv)


def _print_matrix(config: ExperimentConfig) -> None:
    n_cells = len(config.environments) * len(config.seeds)
    n_train = n_cells * len(config.agents)
    print(f"Experiment: {config.name}")
    print(f"Seeds:      {', '.join(map(str, config.seeds))}")
    print(f"Output:     {config.output_dir}")
    if config.checkpoint_dir is not None:
        print(f"Checkpoints: {config.checkpoint_dir}")
    print(
        f"\nMatrix: {len(config.environments)} env x {len(config.agents)} agent "
        f"x {len(config.seeds)} seed = {n_cells} cell(s), {n_train} training run(s)"
    )

    print("\nEnvironments:")
    for env in config.environments:
        files = ", ".join(f"{k}={v}" for k, v in env.paths.items())
        missing = [v for v in env.paths.values() if not Path(v).exists()]
        flag = "   [MISSING FILE]" if missing else ""
        settings = env.settings
        print(f"  - {env.id} [{env.backend}] {files}{flag}")
        print(
            f"      control={settings['control_mode']} reward={settings['global_reward_fn']}"
            f" train_ep={settings['train_episodes']} eval_ep={settings['eval_episodes']}"
            f" max_steps={settings['max_steps']} baselines={settings['compare_with']}"
        )

    print("\nAgents:")
    for agent in config.agents:
        print(f"  - {agent.id} [{agent.type}] {agent.params}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        _print_matrix(config)
        return

    # Imported here so --dry-run never pulls in numpy / backends.
    from experiments.report import write_all
    from experiments.runner import run_matrix

    report = run_matrix(
        config,
        workers=max(1, args.workers),
        verbose=True,
        from_checkpoint=args.from_checkpoint,
    )
    write_all(report, config.output_dir, plots=not args.no_plot)

    cells = report["cells"]
    ok = sum(1 for cell in cells if cell["status"] == "ok")
    print(f"\nCompleted {ok}/{len(cells)} cell(s).")
    if ok == 0:
        raise SystemExit("No cell completed successfully.")


if __name__ == "__main__":
    main()
