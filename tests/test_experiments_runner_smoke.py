"""End-to-end orchestration test on a lightweight fake env.

CityFlow/SUMO/MOSS are not required: ``make_env`` and ``backend_ready`` are
monkeypatched so ``run_matrix`` drives a tiny in-process ``FakeEnv`` that
honours the ``BaseTrafficEnv`` contract. torch (for the real DQN agent) is a
hard dependency and is available, so this exercises the full
train -> eval -> aggregate -> report path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

import experiments.runner as runner
from experiments.config import load_config


class _FakeIx:
    def __init__(self, idx: int, num_phases: int) -> None:
        self.id = f"ix_{idx}"
        self.num_phases = num_phases
        self.incoming_lanes = [f"{self.id}_in_{j}" for j in range(2)]
        self.outgoing_lanes = [f"{self.id}_out_{j}" for j in range(2)]
        # Two roadlinks (in_j -> out_j); enough for MaxPressureAgent, which reads
        # roadlink_lanes and phase_roadlink_mapping.
        self.roadlink_lanes = [
            ([self.incoming_lanes[0]], [self.outgoing_lanes[0]]),
            ([self.incoming_lanes[1]], [self.outgoing_lanes[1]]),
        ]
        _patterns = [[0], [1], [0, 1]]
        self.phase_roadlink_mapping = [_patterns[p % len(_patterns)] for p in range(num_phases)]


class FakeEnv:
    """Minimal env honouring the contract the runner/agents rely on."""

    def __init__(self, n_ix: int = 2, num_phases: int = 3, max_steps: int = 4) -> None:
        self.intersections = [_FakeIx(i, num_phases) for i in range(n_ix)]
        self.num_phases = num_phases
        self.max_steps = int(max_steps)
        self.control_mode = "acyclic"
        if n_ix == 1:
            self.action_space = spaces.Discrete(num_phases)
        else:
            self.action_space = spaces.MultiDiscrete([num_phases] * n_ix)
        self._rng = np.random.default_rng(0)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        return self._info()

    def step(self, action):
        # Real envs reject a wrong-shaped action; mirror that so a runner bug
        # producing a bad action would be caught here.
        action = np.asarray(action).reshape(-1)
        assert action.shape[0] == len(self.intersections)
        self._step += 1
        reward = float(-self._rng.random())
        truncated = self._step >= self.max_steps
        return reward, False, truncated, self._info()

    def close(self) -> None:
        pass

    def _info(self) -> dict:
        lane_vehicle: dict[str, int] = {}
        lane_waiting: dict[str, int] = {}
        intersections: dict[str, dict] = {}
        for ix in self.intersections:
            for lid in (*ix.incoming_lanes, *ix.outgoing_lanes):
                lane_vehicle[lid] = int(self._rng.integers(0, 5))
            for lid in ix.incoming_lanes:
                lane_waiting[lid] = int(self._rng.integers(0, 4))
            intersections[ix.id] = {
                "state": [float(lane_vehicle[l]) for l in ix.incoming_lanes]
                + [0.0] * self.num_phases,
                "avail_actions": list(range(ix.num_phases)),
                "current_phase": 0,
                "time_in_phase": self._step,
            }
        return {
            "intersections": intersections,
            "lane_vehicle_count": lane_vehicle,
            "lane_waiting_vehicle_count": lane_waiting,
            "average_travel_time": float(self._rng.random() * 10),
            "vehicle_count": int(self._rng.integers(0, 20)),
            "step": self._step,
            "metrics": {},
        }


def _smoke_config(
    tmp_path: Path,
    *,
    seeds: list[int] | None = None,
    compare_with: list[str] | None = None,
    checkpoint_dir: Path | None = None,
) -> Path:
    cfg = {
        "name": "fake_smoke",
        "seeds": seeds or [7],
        "defaults": {
            "train_episodes": 1,
            "eval_episodes": 1,
            "max_steps": 4,
            "compare_with": ["random"] if compare_with is None else compare_with,
        },
        "environments": [{"id": "fake", "backend": "cityflow", "config": "x.json"}],
        "agents": [{"id": "dqn", "type": "dqn", "params": {"lr": 0.001, "epsilon_decay_steps": 10}}],
    }
    if checkpoint_dir is not None:
        cfg["checkpoint_dir"] = str(checkpoint_dir)
    path = tmp_path / "exp.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner, "backend_ready", lambda backend, paths, libsumo=False: (True, "")
    )
    monkeypatch.setattr(
        runner, "make_env", lambda env_spec: FakeEnv(max_steps=env_spec.settings["max_steps"])
    )


def test_run_matrix_produces_aggregated_results(tmp_path: Path, fake_backend: None) -> None:
    config = load_config(_smoke_config(tmp_path))
    report = runner.run_matrix(config, workers=1, verbose=False)

    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["status"] == "ok", cell.get("reason")
    assert set(cell["policies"]) == {"dqn", "Random"}

    agg = report["aggregated"]["fake"]
    for policy in ("dqn", "Random"):
        metrics = agg[policy]
        assert set(metrics) == {
            "episode_reward",
            "average_travel_time",
            "final_vehicle_count",
            "average_waiting_queue",
        }
        assert metrics["episode_reward"]["n"] == 1


def test_report_files_written(tmp_path: Path, fake_backend: None) -> None:
    import csv

    from experiments.report import write_all

    config = load_config(_smoke_config(tmp_path))
    report = runner.run_matrix(config, workers=1, verbose=False)

    out_dir = tmp_path / "out"
    paths = write_all(report, out_dir, plots=False)

    # JSON: aggregated shape env -> policy -> metric -> {mean, std, n}
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["name"] == "fake_smoke"
    stats = loaded["aggregated"]["fake"]["dqn"]["episode_reward"]
    assert set(stats) == {"mean", "std", "n"}

    # CSV: real header + a parseable numeric value, one row per policy
    with paths["csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_policy = {r["policy"]: r for r in rows}
    assert {"dqn", "Random"} <= set(by_policy)
    assert by_policy["dqn"]["backend"] == "cityflow"
    assert by_policy["dqn"]["n_seeds"] == "1"
    float(by_policy["dqn"]["episode_reward_mean"])  # must parse as a number


def test_learning_curve_plot_written(tmp_path: Path, fake_backend: None) -> None:
    pytest.importorskip("matplotlib")

    from experiments.report import write_all

    config = load_config(_smoke_config(tmp_path))
    report = runner.run_matrix(config, workers=1, verbose=False)

    paths = write_all(report, tmp_path / "out", plots=True)
    curve = tmp_path / "out" / "plots" / "fake_learning_curve.png"
    assert curve in paths["plots"]
    assert curve.exists()


def test_cross_env_comparison_plot_written(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    from experiments.report import save_plots

    def _stats(value: float) -> dict:
        return {m: {"mean": value, "std": 0.1, "n": 2} for m, _ in runner.METRICS_SPEC}

    report = {
        "name": "cf_vs_sumo",
        "aggregated": {
            "cityflow_map": {"dqn": _stats(1.0), "Random": _stats(2.0)},
            "sumo_map": {"dqn": _stats(1.5), "Random": _stats(2.5)},
        },
        "cells": [],
    }
    written = save_plots(report, tmp_path / "plots")
    cross = tmp_path / "plots" / "cross_env_comparison.png"
    assert cross in written
    assert cross.exists()


def test_missing_backend_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner, "backend_ready", lambda backend, paths, libsumo=False: (False, "engine missing")
    )
    config = load_config(_smoke_config(tmp_path))
    report = runner.run_matrix(config, workers=1, verbose=False)

    cell = report["cells"][0]
    assert cell["status"] == "skipped"
    assert cell["reason"] == "engine missing"
    assert report["aggregated"] == {}


def test_max_pressure_baseline_runs(tmp_path: Path, fake_backend: None) -> None:
    config = load_config(_smoke_config(tmp_path, compare_with=["random", "max_pressure"]))
    report = runner.run_matrix(config, workers=1, verbose=False)

    cell = report["cells"][0]
    assert cell["status"] == "ok", cell.get("reason")
    assert {"dqn", "Random", "MaxPressure"} == set(cell["policies"])
    assert cell["policies"]["MaxPressure"]["kind"] == "baseline"


def test_multi_seed_aggregation(tmp_path: Path, fake_backend: None) -> None:
    config = load_config(_smoke_config(tmp_path, seeds=[7, 8]))
    report = runner.run_matrix(config, workers=1, verbose=False)

    assert [c["seed"] for c in report["cells"]] == [7, 8]
    stats = report["aggregated"]["fake"]["dqn"]["episode_reward"]
    assert stats["n"] == 2  # mean/std now reduce over two seeds


def test_checkpoint_round_trip_load_matches_eval(tmp_path: Path, fake_backend: None) -> None:
    config = load_config(
        _smoke_config(tmp_path, compare_with=[], checkpoint_dir=tmp_path / "ckpts")
    )
    trained = runner.run_matrix(config, workers=1, verbose=False)

    trained_policy = trained["cells"][0]["policies"]["dqn"]
    checkpoint = runner.checkpoint_path(config.checkpoint_dir, "fake", "dqn", 7)
    assert Path(trained_policy["checkpoint_path"]) == checkpoint
    assert trained_policy["checkpoint_loaded"] is False
    assert trained_policy["train_returns"] and len(trained_policy["train_returns"]) == 1
    assert checkpoint.exists()

    loaded = runner.run_matrix(
        config,
        workers=1,
        verbose=False,
        from_checkpoint=config.checkpoint_dir,
    )

    loaded_policy = loaded["cells"][0]["policies"]["dqn"]
    assert loaded["from_checkpoint"] == str(config.checkpoint_dir)
    assert loaded_policy["checkpoint_loaded"] is True
    assert loaded_policy["train_returns"] == []
    assert loaded["cells"][0]["timings"]["dqn"]["train_sec"] is None
    assert loaded["cells"][0]["timings"]["dqn"]["load_sec"] >= 0.0
    assert loaded_policy["metrics"] == trained_policy["metrics"]


def test_make_env_failure_yields_error_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "backend_ready", lambda backend, paths, libsumo=False: (True, "")
    )

    def _boom(_env_spec):
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(runner, "make_env", _boom)
    config = load_config(_smoke_config(tmp_path))
    report = runner.run_matrix(config, workers=1, verbose=False)

    cell = report["cells"][0]
    assert cell["status"] == "error"
    assert "engine blew up" in cell["reason"]
    assert report["aggregated"] == {}


def test_build_agent_resolves_epsilon_decay_to_budget() -> None:
    from experiments.registry import build_agent, validate_agent_params

    env = FakeEnv(max_steps=4)
    params = validate_agent_params("dqn", {})  # epsilon_decay_steps defaults to None
    agent = build_agent(
        "dqn", env, params, device="cpu", seed=1, train_episodes=3, max_steps=4
    )
    assert agent.epsilon_decay_steps == 12  # 3 episodes * 4 steps
