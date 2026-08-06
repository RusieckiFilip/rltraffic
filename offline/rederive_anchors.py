"""Re-derive the PROJECT_PLAN §3.1 sanity anchors under prereg A1 (``att_horizon``).

The committed 2026-07-09 data (``docs/data/p0_baselines/``) stores only the running-mean
aggregate and therefore *cannot* yield the horizon value -- a re-run of
``experiments/configs/p0_baselines.json`` is required. This harness reproduces the exact anchor
pipeline and records **both** ``att_horizon`` (A1 primary) and ``att_running_mean`` (legacy) per
cell, so no future comparison can silently mix them.

Faithfulness to the anchor pipeline is why this imports frozen, underscore-private symbols from
``experiments.runner`` -- see the import-site comment in ``_rollout_cell``. ``att_running_mean``
reproducing the committed baseline anchors *exactly* is the load-bearing check that the pipeline
matches.

Output (Master Ruling 2): a NEW directory, never the historical one. ``results.json`` +
``summary.csv`` + ``PROVENANCE.md``, all written only after every cell is computed in memory
(filesystem-mutation barrier). ``--policies`` selects a subset: the P8.0 split runs the two
baselines in-session and the (long, torch-training) MAPPO cells in the user's tmux.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["main", "assert_safe_out_dir", "HISTORICAL_DIR", "DEFAULT_OUT_DIR"]

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = REPO_ROOT / "docs" / "data" / "p0_baselines"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "data" / "p0_baselines_horizon"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "p0_baselines.json"

FORMAT_VERSION = "p0-baselines-horizon-v1.0"
QUANTITIES = ("att_horizon", "att_running_mean")


def assert_safe_out_dir(out_dir: Path) -> None:
    """Refuse to write into the frozen historical anchor directory.

    The 2026-07-09 running-mean data is the validation target; overwriting it would destroy the
    reference the whole re-derivation is checked against. Raises ``ValueError`` (message contains
    ``"historical"``) when *out_dir* resolves to :data:`HISTORICAL_DIR`.
    """
    if Path(out_dir).resolve() == HISTORICAL_DIR.resolve():
        raise ValueError(
            f"refusing to write into the historical anchor directory {HISTORICAL_DIR}: it is the "
            "immutable 2026-07-09 validation target this re-run is checked against. Choose a new "
            "--out-dir (default docs/data/p0_baselines_horizon/)."
        )


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - provenance degrades gracefully
        return "unknown"


def _rollout_cell(env_spec: Any, agents: tuple[Any, ...], compare_with: list[str],
                  seed: int, requested: set[str], verbose: bool) -> dict[str, Any]:
    """One (env, seed) cell: train+eval requested agents, eval requested baselines.

    Reproduces ``experiments.runner.run_cell``'s pipeline exactly by reusing its own building
    blocks, then reads both A1 aggregations from each rollout via ``horizon_rollout``.
    """
    # DELIBERATE private import from the FROZEN experiments.runner. This is the only way to stay
    # byte-faithful to the pipeline that produced the committed anchors; reimplementing training or
    # the choosers would be a different policy and would destroy the exact-reproduction check that
    # is the whole point of re-running (Master Ruling 3, P8.0). Import-only -- never edit runner.
    # Note: importing runner and calling limit_torch_threads pins torch to ONE thread
    # process-globally; that is expected and documented, not a surprise.
    from experiments.envs import make_env
    from experiments.registry import BASELINE_LABELS
    from experiments.runner import (
        _agent_chooser,
        _baseline_chooser,
        _train_agent,
        limit_torch_threads,
    )

    from offline.horizon_metric import horizon_rollout

    settings = env_spec.settings
    eval_seed = seed + 10_000  # runner.run_cell's eval seed convention
    eval_episodes = int(settings["eval_episodes"])
    policies: dict[str, dict[str, float]] = {}

    train_agents = [a for a in agents if a.id in requested]
    if train_agents:
        limit_torch_threads()  # pin before build_agent imports torch (matches run_cell)
    for agent_spec in train_agents:
        if verbose:
            print(f"[train] {env_spec.id}@seed{seed} :: {agent_spec.id}", flush=True)
        agent, _ = _train_agent(agent_spec, env_spec, seed)
        eval_env = make_env(env_spec)
        try:
            roll = horizon_rollout(eval_env, _agent_chooser(agent), eval_episodes, eval_seed)
        finally:
            eval_env.close()
        policies[agent_spec.id] = _cell_payload(roll, kind="agent")

    for index, baseline in enumerate(compare_with):
        if baseline not in requested:
            continue
        label = BASELINE_LABELS[baseline]
        if verbose:
            print(f"[eval ] {env_spec.id}@seed{seed} :: {label}", flush=True)
        action_seed = seed + 50_000 + index  # runner.run_cell's baseline action-seed convention
        eval_env = make_env(env_spec)
        try:
            chooser = _baseline_chooser(baseline, eval_env, action_seed)
            roll = horizon_rollout(eval_env, chooser, eval_episodes, eval_seed)
        finally:
            eval_env.close()
        policies[label] = _cell_payload(roll, kind="baseline")

    return {"env_id": env_spec.id, "seed": int(seed), "status": "ok", "policies": policies}


def _cell_payload(roll: Any, *, kind: str) -> dict[str, float]:
    return {
        "kind": kind,
        "att_horizon": roll.att_horizon,
        "att_running_mean": roll.att_running_mean,
        "episode_reward": roll.episode_reward,
        "final_vehicle_count": roll.final_vehicle_count,
    }


def _aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by (env_id, policy) and reduce over seeds, in cell order (mirrors runner.aggregate).

    Cell order is preserved so the mean/std sum in the same order the anchor test recomputes,
    keeping the double-compute exact.
    """
    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for cell in cells:
        env_acc = acc.setdefault(cell["env_id"], {})
        for label, payload in cell["policies"].items():
            label_acc = env_acc.setdefault(label, {})
            for quantity in QUANTITIES:
                label_acc.setdefault(quantity, []).append(float(payload[quantity]))
    out: dict[str, Any] = {}
    for env_id, labels in acc.items():
        out[env_id] = {}
        for label, quantities in labels.items():
            out[env_id][label] = {
                quantity: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
                for quantity, values in quantities.items()
            }
    return out


def _summary_csv(aggregated: dict[str, Any], cells: list[dict[str, Any]]) -> str:
    header = (
        "env_id,policy,kind,n_seeds,"
        "att_horizon_mean,att_horizon_std,att_running_mean_mean,att_running_mean_std"
    )
    # kind per (env,policy) from the first cell that carries it.
    kind_of: dict[tuple[str, str], str] = {}
    for cell in cells:
        for label, payload in cell["policies"].items():
            kind_of.setdefault((cell["env_id"], label), str(payload.get("kind", "")))
    lines = [header]
    for env_id in sorted(aggregated):
        for policy in sorted(aggregated[env_id]):
            h = aggregated[env_id][policy]["att_horizon"]
            r = aggregated[env_id][policy]["att_running_mean"]
            lines.append(
                f"{env_id},{policy},{kind_of.get((env_id, policy), '')},{h['n']},"
                f"{h['mean']:.4f},{h['std']:.4f},{r['mean']:.4f},{r['std']:.4f}"
            )
    return "\n".join(lines) + "\n"


def _provenance_md(results: dict[str, Any]) -> str:
    present = ", ".join(results["policies_present"]) or "(none)"
    pending = ", ".join(results["pending_policies"]) or "(none)"
    return (
        "# PROVENANCE — docs/data/p0_baselines_horizon\n\n"
        f"- **Generated (UTC):** {results['generated_utc']}\n"
        f"- **Git hash:** `{results['git_hash']}`\n"
        f"- **Source config:** `{results['source_config']}`\n"
        f"- **Format version:** `{results['format_version']}`\n"
        f"- **Quantities held:** `att_horizon` (prereg A1 primary metric) and `att_running_mean` "
        "(legacy runner.py mean-of-samples). Neither is called \"average travel time\" as a bare "
        "name.\n"
        f"- **Policies present:** {present}\n"
        f"- **Policies pending:** {pending}\n\n"
        "## Relationship to `docs/data/p0_baselines/`\n\n"
        "This directory **re-derives** (never replaces) the 2026-07-09 running-mean-only data in "
        "`docs/data/p0_baselines/`. That data stores only the aggregate and cannot yield the "
        "horizon value, so a re-run of `experiments/configs/p0_baselines.json` was required "
        "(prereg amendment A1). The re-run's `att_running_mean` reproduces the committed anchors "
        "exactly for the torch-free baselines (MaxPressure, Random) — that reproduction is the "
        "load-bearing check that the horizon values from the same run are trustworthy. MAPPO "
        "training runs in the user's tmux session (CLAUDE.md:203) and may cross the N2 "
        "float-reduction boundary; when its cells land they are recorded, never adjusted to match.\n\n"
        "Reproduce with:\n\n"
        "```\n"
        ".venv/bin/python -m offline.rederive_anchors --policies random,max_pressure   # in-session\n"
        ".venv/bin/python -m offline.rederive_anchors                                  # full (tmux)\n"
        "```\n"
    )


def _run(config: Any, requested: set[str], verbose: bool) -> dict[str, Any]:
    from experiments.registry import BASELINE_LABELS

    cells: list[dict[str, Any]] = []
    for env_spec in config.environments:
        compare_with = list(env_spec.settings["compare_with"])
        for seed in config.seeds:
            cells.append(
                _rollout_cell(env_spec, config.agents, compare_with, seed, requested, verbose)
            )

    aggregated = _aggregate(cells)

    # Full label universe of the matrix, to name what is present vs pending (never a silent gap).
    all_labels: list[str] = [a.id for a in config.agents]
    seen = set(all_labels)
    for env_spec in config.environments:
        for baseline in env_spec.settings["compare_with"]:
            label = BASELINE_LABELS[baseline]
            if label not in seen:
                all_labels.append(label)
                seen.add(label)
    present = [lab for lab in all_labels if any(lab in c["policies"] for c in cells)]
    pending = [lab for lab in all_labels if lab not in present]

    return {
        "format_version": FORMAT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "source_config": "experiments/configs/p0_baselines.json",
        "quantities": list(QUANTITIES),
        "seeds": [int(s) for s in config.seeds],
        "policies_present": present,
        "pending_policies": pending,
        "cells": cells,
        "aggregated": aggregated,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-derive §3.1 anchors under prereg A1 (att_horizon).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--policies",
        default="mappo,random,max_pressure",
        help="comma-separated subset of {mappo,random,max_pressure} to run",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 == success)."""
    from experiments.config import load_config

    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    assert_safe_out_dir(out_dir)  # barrier: reject the historical dir BEFORE any work

    config = load_config(args.config)
    requested = {tok.strip() for tok in args.policies.split(",") if tok.strip()}
    known = {a.id for a in config.agents}
    for env_spec in config.environments:
        known.update(env_spec.settings["compare_with"])
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"--policies has unknown token(s) {unknown}; known: {sorted(known)}")

    # Compute EVERYTHING before writing (filesystem-mutation barrier).
    results = _run(config, requested, args.verbose)
    results_json = json.dumps(results, indent=2, sort_keys=False)
    summary = _summary_csv(results["aggregated"], results["cells"])
    provenance = _provenance_md(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(results_json + "\n", encoding="utf-8")
    (out_dir / "summary.csv").write_text(summary, encoding="utf-8")
    (out_dir / "PROVENANCE.md").write_text(provenance, encoding="utf-8")

    if args.verbose:
        print(f"[done ] wrote {out_dir} :: present={results['policies_present']} "
              f"pending={results['pending_policies']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
