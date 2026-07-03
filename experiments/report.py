"""Persist and present experiment results.

Writes a full ``results.json`` and a flat ``summary.csv`` (one row per
environment x policy), prints a per-environment leaderboard, and — when
matplotlib is available — saves comparison bar charts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.runner import METRICS_SPEC

_METRIC_NAMES = [name for name, _ in METRICS_SPEC]
_HIGHER_IS_BETTER = dict(METRICS_SPEC)
# Policies are ranked by this primary metric in the leaderboard. Deliberately
# not episode_reward: reward scales differ across global_reward_fn, while travel
# time is comparable across configs.
_PRIMARY_METRIC = "average_travel_time"


def write_results_json(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_summary_csv(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["env_id", "backend", "policy", "kind", "n_seeds"]
    for metric in _METRIC_NAMES:
        fieldnames += [f"{metric}_mean", f"{metric}_std"]

    backend_by_env = {e["id"]: e["backend"] for e in report.get("environments", [])}
    kind_by_label = _policy_kinds(report)

    rows: list[dict[str, Any]] = []
    for env_id, policies in report.get("aggregated", {}).items():
        for policy, metrics in policies.items():
            row: dict[str, Any] = {
                "env_id": env_id,
                "backend": backend_by_env.get(env_id, ""),
                "policy": policy,
                "kind": kind_by_label.get((env_id, policy), ""),
                "n_seeds": _n_seeds(metrics),
            }
            for metric in _METRIC_NAMES:
                stats = metrics.get(metric, {})
                row[f"{metric}_mean"] = _round(stats.get("mean"))
                row[f"{metric}_std"] = _round(stats.get("std"))
            rows.append(row)

    rows.sort(key=lambda r: (r["env_id"], r["policy"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_leaderboard(report: dict[str, Any]) -> None:
    aggregated = report.get("aggregated", {})
    if not aggregated:
        print("No successful runs to report.")
        _print_non_ok(report)
        return

    for env_id, policies in aggregated.items():
        print(f"\n=== {env_id} ===")
        headers = ["Policy"] + [_short(metric) for metric in _METRIC_NAMES]
        rows: list[list[str]] = []
        ranked = sorted(
            policies.items(),
            key=lambda item: item[1].get(_PRIMARY_METRIC, {}).get("mean", float("inf")),
        )
        for policy, metrics in ranked:
            cells = [policy]
            for metric in _METRIC_NAMES:
                stats = metrics.get(metric)
                cells.append(
                    f"{stats['mean']:.2f}±{stats['std']:.2f}" if stats else "-"
                )
            rows.append(cells)
        _print_table(headers, rows)

    _print_non_ok(report)


def save_plots(report: dict[str, Any], out_dir: Path) -> list[Path]:
    """One grouped bar chart per environment (policies x metrics).

    Returns the written paths; empty if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []

    aggregated = report.get("aggregated", {})
    if not aggregated:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for env_id, policies in aggregated.items():
        labels = list(policies)
        fig, axes = plt.subplots(1, len(_METRIC_NAMES), figsize=(4 * len(_METRIC_NAMES), 4))
        if len(_METRIC_NAMES) == 1:
            axes = [axes]
        for ax, metric in zip(axes, _METRIC_NAMES):
            means = [policies[p].get(metric, {}).get("mean", 0.0) for p in labels]
            errs = [policies[p].get(metric, {}).get("std", 0.0) for p in labels]
            ax.bar(range(len(labels)), means, yerr=errs, capsize=3)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            arrow = "↑" if _HIGHER_IS_BETTER.get(metric) else "↓"
            ax.set_title(f"{metric} ({arrow})", fontsize=9)
        fig.suptitle(f"{report.get('name', 'experiment')} — {env_id}")
        fig.tight_layout()
        path = out_dir / f"{_safe(env_id)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    written.extend(_save_cross_env_comparison(report, out_dir, plt, np))
    written.extend(_save_learning_curve_plots(report, out_dir, plt, np))
    return written


def _save_cross_env_comparison(
    report: dict[str, Any],
    out_dir: Path,
    plt: Any,
    np: Any,
) -> list[Path]:
    """One grouped bar chart per metric, bars grouped by environment.

    This is the cross-backend view (e.g. SUMO vs CityFlow on the same map):
    each policy sits on the x-axis with one bar per environment. Only emitted
    when at least two environments produced results.
    """
    aggregated = report.get("aggregated", {})
    env_ids = list(aggregated)
    if len(env_ids) < 2:
        return []

    policies: list[str] = []
    for env_id in env_ids:
        for label in aggregated[env_id]:
            if label not in policies:
                policies.append(label)
    if not policies:
        return []

    n_metrics = len(_METRIC_NAMES)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4.2))
    if n_metrics == 1:
        axes = [axes]
    x = np.arange(len(policies))
    width = 0.8 / len(env_ids)
    for ax, metric in zip(axes, _METRIC_NAMES):
        for i, env_id in enumerate(env_ids):
            means = [aggregated[env_id].get(p, {}).get(metric, {}).get("mean", np.nan) for p in policies]
            errs = [aggregated[env_id].get(p, {}).get(metric, {}).get("std", 0.0) for p in policies]
            offset = (i - (len(env_ids) - 1) / 2) * width
            ax.bar(x + offset, means, width=width, yerr=errs, capsize=3, label=env_id)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
        arrow = "↑" if _HIGHER_IS_BETTER.get(metric) else "↓"
        ax.set_title(f"{metric} ({arrow})", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(env_ids), fontsize=8,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"{report.get('name', 'experiment')} — cross-environment comparison", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = out_dir / "cross_env_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]


def write_all(report: dict[str, Any], out_dir: Path, *, plots: bool = True) -> dict[str, Any]:
    """Write json + csv (+ plots), print the leaderboard, return path info."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_results_json(report, out_dir / "results.json")
    csv_path = write_summary_csv(report, out_dir / "summary.csv")
    print_leaderboard(report)
    print(f"\nResults: {json_path}")
    print(f"Summary: {csv_path}")

    plot_paths: list[Path] = []
    if plots:
        plot_paths = save_plots(report, out_dir / "plots")
        if plot_paths:
            print(f"Plots:   {out_dir / 'plots'} ({len(plot_paths)} file(s))")
    return {"json": json_path, "csv": csv_path, "plots": plot_paths}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _policy_kinds(report: dict[str, Any]) -> dict[tuple[str, str], str]:
    kinds: dict[tuple[str, str], str] = {}
    for cell in report.get("cells", []):
        for label, payload in cell.get("policies", {}).items():
            kinds[(cell["env_id"], label)] = payload.get("kind", "")
    return kinds


def _save_learning_curve_plots(
    report: dict[str, Any],
    out_dir: Path,
    plt: Any,
    np: Any,
) -> list[Path]:
    curves: dict[str, dict[str, list[list[float]]]] = {}
    for cell in report.get("cells", []):
        if cell.get("status") != "ok":
            continue
        env_id = str(cell.get("env_id", ""))
        for policy, payload in cell.get("policies", {}).items():
            if payload.get("kind") != "agent":
                continue
            values = payload.get("train_returns")
            if not isinstance(values, list) or not values:
                continue
            curves.setdefault(env_id, {}).setdefault(policy, []).append(
                [float(value) for value in values]
            )

    written: list[Path] = []
    for env_id, policies in curves.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, runs in policies.items():
            max_len = max(len(run) for run in runs)
            matrix = np.full((len(runs), max_len), np.nan, dtype=float)
            for row, run in enumerate(runs):
                matrix[row, : len(run)] = run
            xs = np.arange(1, max_len + 1)
            means = np.nanmean(matrix, axis=0)
            stds = np.nanstd(matrix, axis=0)
            ax.plot(xs, means, marker="o", linewidth=1.5, label=policy)
            if len(runs) > 1:
                ax.fill_between(xs, means - stds, means + stds, alpha=0.15)

        ax.set_title(f"{report.get('name', 'experiment')} — {env_id} learning curve")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode return")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{_safe(env_id)}_learning_curve.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _n_seeds(metrics: dict[str, Any]) -> int:
    for stats in metrics.values():
        return int(stats.get("n", 0))
    return 0


def _print_non_ok(report: dict[str, Any]) -> None:
    notes = [
        f"  [{cell.get('status')}] {cell.get('env_id')}@seed{cell.get('seed')}: "
        f"{cell.get('reason', '')}"
        for cell in report.get("cells", [])
        if cell.get("status") != "ok"
    ]
    if notes:
        print("\nSkipped / failed cells:")
        print("\n".join(notes))


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _short(metric: str) -> str:
    return {
        "episode_reward": "EpReward",
        "average_travel_time": "TravelTime",
        "final_vehicle_count": "FinalVeh",
        "average_waiting_queue": "Queue",
    }.get(metric, metric)


def _round(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, (int, float)) else ""


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
