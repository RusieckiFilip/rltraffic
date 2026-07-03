"""Regression tests for Gymnasium-style seeding in ``CityFlowEnv``.

CityFlow's ``Engine.reset()`` re-seeds its RNG with the seed from the
config file, so unless the env re-seeds the engine explicitly, every
episode replays identically and ``reset(seed=...)`` is silently ignored.

The env follows the Gymnasium convention: ``reset(seed=X)`` re-seeds the
env RNG, ``reset()`` continues it, and every reset draws a fresh engine
seed from that RNG and applies it via ``set_random_seed``.

Uses a fake ``cityflow`` module so the tests run without CityFlow
installed.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from utils.cityflow_utils import parse_roadnet

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "sim" / "config_1x1.json"
ROADNET_PATH = ROOT_DIR / "scenarios" / "aigen_1x1" / "roadnet_1x1.json"

LANE_IDS = parse_roadnet(ROADNET_PATH).lane_ids


class FakeEngine:
    """Minimal stand-in for ``cityflow.Engine`` recording seed calls."""

    def __init__(self, config_path: str, thread_num: int = 1) -> None:
        self.config_path = config_path
        self.thread_num = thread_num
        self.seed_calls: list[int] = []
        self.reset_calls = 0

    # --- control APIs -------------------------------------------------
    def reset(self) -> None:
        self.reset_calls += 1

    def set_random_seed(self, seed: int) -> None:
        self.seed_calls.append(int(seed))

    def next_step(self) -> None:
        pass

    def set_tl_phase(self, ix_id: str, phase: int) -> None:
        pass

    # --- read APIs used by CityFlowMetrics / _get_info -----------------
    def get_lane_vehicles(self) -> dict[str, list[str]]:
        return {lid: [] for lid in LANE_IDS}

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        return {lid: 0 for lid in LANE_IDS}

    def get_vehicle_speed(self) -> dict[str, float]:
        return {}

    def get_vehicles(self, include_waiting: bool = False) -> list[str]:
        return []

    def get_current_time(self) -> float:
        return 0.0

    def get_vehicle_count(self) -> int:
        return 0

    def get_average_travel_time(self) -> float:
        return 0.0


@pytest.fixture()
def cityflow_env_module(monkeypatch):
    fake_module = types.ModuleType("cityflow")
    fake_module.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "cityflow", fake_module)

    already_loaded = "envs.cityflow_env" in sys.modules
    module = importlib.import_module("envs.cityflow_env")
    monkeypatch.setattr(module, "cityflow", fake_module)

    yield module

    if not already_loaded:
        # Drop the module bound against the fake so later imports
        # (e.g. tests using a real cityflow install) re-import cleanly.
        sys.modules.pop("envs.cityflow_env", None)


def _make_env(module):
    return module.CityFlowEnv(
        cityflow_config_path=str(CONFIG_PATH),
        delta_time=10,
    )


def test_construction_keeps_config_seed(cityflow_env_module, monkeypatch):
    monkeypatch.chdir(ROOT_DIR)
    env = _make_env(cityflow_env_module)
    try:
        assert isinstance(env._eng, FakeEngine)
        # Before the first reset the engine must keep its config seed.
        assert env._eng.seed_calls == []
    finally:
        env.close()


def test_same_seed_gives_same_engine_seed(cityflow_env_module, monkeypatch):
    monkeypatch.chdir(ROOT_DIR)
    env = _make_env(cityflow_env_module)
    try:
        env.reset(seed=123)
        env.reset(seed=123)
        assert len(env._eng.seed_calls) == 2
        # Re-seeding with the same seed must replay the same episode.
        assert env._eng.seed_calls[0] == env._eng.seed_calls[1]
    finally:
        env.close()


def test_seedless_reset_continues_rng_stream(cityflow_env_module, monkeypatch):
    monkeypatch.chdir(ROOT_DIR)
    env_a = _make_env(cityflow_env_module)
    env_b = _make_env(cityflow_env_module)
    try:
        # Seed once, then continue: episodes differ...
        env_a.reset(seed=123)
        env_a.reset()
        env_a.reset()
        seeds_a = list(env_a._eng.seed_calls)
        assert len(seeds_a) == 3
        assert len(set(seeds_a)) == 3

        # ...but the whole sequence is reproducible from the one seed.
        env_b.reset(seed=123)
        env_b.reset()
        env_b.reset()
        assert env_b._eng.seed_calls == seeds_a
    finally:
        env_a.close()
        env_b.close()


def test_unseeded_reset_still_seeds_engine(cityflow_env_module, monkeypatch):
    monkeypatch.chdir(ROOT_DIR)
    env = _make_env(cityflow_env_module)
    try:
        # Never-seeded env: RNG comes from OS entropy, but the engine
        # must still get a per-episode seed (otherwise CityFlow replays
        # the config seed identically every episode).
        env.reset()
        env.reset()
        assert len(env._eng.seed_calls) == 2
    finally:
        env.close()
