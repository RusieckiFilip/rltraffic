"""Validation tests for experiment config loading (no backends required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.config import load_config


def _base_config() -> dict:
    return {
        "name": "unit",
        "seeds": [7, 8],
        "defaults": {
            "train_episodes": 3,
            "eval_episodes": 1,
            "max_steps": 60,
            "global_reward_fn": "queue_length",
            "compare_with": ["random"],
        },
        "environments": [
            {"id": "cf", "backend": "cityflow", "config": "scenario.json"},
        ],
        "agents": [
            {"id": "dqn", "type": "dqn", "params": {"lr": 0.001}},
        ],
    }


def _write(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "exp.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_valid_config_parses(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _base_config()))

    assert config.name == "unit"
    assert config.seeds == (7, 8)
    assert [e.id for e in config.environments] == ["cf"]
    assert [a.id for a in config.agents] == ["dqn"]
    assert config.checkpoint_dir is None

    env = config.environments[0]
    # defaults flow into per-env settings; unspecified ones get framework defaults
    assert env.settings["train_episodes"] == 3
    assert env.settings["control_mode"] == "acyclic"
    assert env.settings["compare_with"] == ["random"]
    # agent params are merged onto the type defaults
    assert env.backend == "cityflow"
    assert config.agents[0].params["lr"] == 0.001
    assert config.agents[0].params["gamma"] == 0.99


def test_config_path_resolved_relative_to_config_file(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _base_config()))
    resolved = Path(config.environments[0].paths["config"])
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "scenario.json").resolve()


def test_checkpoint_dir_resolved_as_named_run_dir(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["checkpoint_dir"] = str(tmp_path / "ckpts")
    config = load_config(_write(tmp_path, cfg))
    assert config.checkpoint_dir == (tmp_path / "ckpts" / "unit").resolve()


def test_env_override_beats_default(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["environments"][0]["overrides"] = {"max_steps": 999, "control_mode": "cyclic"}
    config = load_config(_write(tmp_path, cfg))
    settings = config.environments[0].settings
    assert settings["max_steps"] == 999
    assert settings["control_mode"] == "cyclic"
    # untouched defaults remain
    assert settings["train_episodes"] == 3


def test_moss_requires_map_and_person(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["environments"] = [{"id": "m", "backend": "moss", "map_file": "m.pb"}]
    with pytest.raises(ValueError, match="person_file"):
        load_config(_write(tmp_path, cfg))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.update({"bogus": 1}), "unknown key"),
        (lambda c: c.update({"checkpoint_dir": ""}), "checkpoint_dir"),
        (lambda c: c.pop("agents"), "missing required key"),
        (lambda c: c["environments"][0].update({"backend": "matsim"}), "backend"),
        (lambda c: c["defaults"].update({"global_reward_fn": "nope"}), "Unknown reward_fn"),
        (lambda c: c["defaults"].update({"local_reward_fn": "nope"}), "Unknown local_reward_fn"),
        (lambda c: c["defaults"].update({"control_mode": "weird"}), "control_mode"),
        (lambda c: c["defaults"].update({"compare_with": ["greedy"]}), "baseline"),
        (lambda c: c["defaults"].update({"unknown_setting": 1}), "unknown key"),
        (lambda c: c["agents"][0].update({"type": "sac"}), "unsupported agent type"),
        (lambda c: c["agents"][0]["params"].update({"momentum": 0.9}), "unsupported parameter"),
        (lambda c: c["environments"].append(c["environments"][0]), "duplicate environment id"),
        (lambda c: c["agents"].append({"id": "Random", "type": "dqn"}), "reserved"),
        (lambda c: c.update({"seeds": []}), "non-empty list"),
        (lambda c: c.update({"seeds": [7, 7]}), "duplicates"),
        (lambda c: c.update({"name": "../escape"}), "path separators"),
        (lambda c: c["environments"][0].update({"map_file": "m.pb"}), "not used by backend"),
        (lambda c: c["defaults"].update({"gui": True, "libsumo": True}), "cannot both be true"),
        (lambda c: c["defaults"].update({"control_mode": "acyclic", "delta_time": 5}), "delta_time > 5"),
    ],
)
def test_invalid_config_raises(tmp_path: Path, mutate, match) -> None:
    cfg = _base_config()
    mutate(cfg)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("mode", ["acyclic_bounded", "cyclic", "resco_cyclic"])
def test_other_control_modes_accepted(tmp_path: Path, mode: str) -> None:
    cfg = _base_config()
    cfg["defaults"]["control_mode"] = mode  # delta_time defaults to 10 (>5)
    config = load_config(_write(tmp_path, cfg))
    assert config.environments[0].settings["control_mode"] == mode


def test_empty_compare_with_is_allowed(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["defaults"]["compare_with"] = []
    config = load_config(_write(tmp_path, cfg))
    assert config.environments[0].settings["compare_with"] == []
    assert config.baselines_for(config.environments[0]) == ()


def test_local_reward_fn_accepted(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["defaults"]["local_reward_fn"] = "queue_length"
    config = load_config(_write(tmp_path, cfg))
    assert config.environments[0].settings["local_reward_fn"] == "queue_length"


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config not found"):
        load_config(tmp_path / "does_not_exist.json")
