"""Tests for the agent/baseline registry (no torch construction here)."""

from __future__ import annotations

import pytest

from experiments.registry import (
    AGENT_TYPES,
    BASELINE_LABELS,
    BASELINE_NAMES,
    get_agent_type,
    validate_agent_params,
)


def test_known_agent_types_present() -> None:
    assert set(AGENT_TYPES) == {"dqn", "ippo", "mappo"}
    assert BASELINE_NAMES == ("random", "max_pressure")
    assert BASELINE_LABELS["random"] == "Random"


def test_get_agent_type_is_case_insensitive() -> None:
    assert get_agent_type("DQN").type == "dqn"


def test_get_agent_type_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unsupported agent type"):
        get_agent_type("ddpg")


def test_validate_agent_params_merges_defaults() -> None:
    merged = validate_agent_params("dqn", {"lr": 0.5})
    assert merged["lr"] == 0.5
    assert merged["gamma"] == 0.99  # default preserved
    assert "epsilon_decay_steps" in merged


def test_validate_agent_params_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unsupported parameter 'foo'"):
        validate_agent_params("ippo", {"foo": 1})


def test_validate_agent_params_none_is_ok() -> None:
    assert validate_agent_params("mappo", None)["lr"] == 3e-4
