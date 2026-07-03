"""Ensure SumoEnv allows enough TraCI connect retries for large SUMO scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_sumo_env_passes_extended_traci_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    import traci

    captured: dict[str, int] = {}

    def fake_start(cmd, port=None, numRetries=0, **kwargs):
        captured["numRetries"] = int(numRetries)
        raise RuntimeError("abort after capturing traci.start args")

    monkeypatch.setattr(traci, "start", fake_start)

    from envs.sumo_env import SumoEnv

    cfg = ROOT_DIR / "scenarios" / "cologne1" / "cologne1.sumocfg"
    with pytest.raises(RuntimeError, match="abort after capturing"):
        SumoEnv(sumocfg_path=str(cfg), max_steps=1, delta_time=1)

    assert captured["numRetries"] == 120
