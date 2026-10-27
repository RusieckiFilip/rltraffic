"""A27(d) in ``offline/collect.py`` (``BRIEF_46`` 3.3): ``--random-rng {per-draw,per-episode}``.

Format version: C6 v1.1 episodes, the logger's own.  Alignment convention (``docs/CONTRACTS.md`` C6): observation rows
``T + 1``, decision and outcome rows ``T``; an action array ``ix<i>_action`` holds the ``T`` decisions in order.

* ``per-draw`` -- the default -- is the registered collector: ONE generator ``default_rng(base_seed)`` per draw, which is
  why the registered random tiers hold two schedules (A27(d)).  The load-bearing test re-collects one registered episode
  (``cf_hz1x1__random/ep000000_seed1000_draw1.npz``: draw 1, engine seed 1000) into ``tmp_path`` and compares every array
  with the corpus file's bytes.  CityFlow is deterministic on hz1x1, so any difference is a regression.
* ``per-episode`` builds ``default_rng([base_seed, draw_id, episode index within the draw])`` for every episode
  (``docs/plans/p9.1.md`` Q1 states which index).  Two episodes of one draw differ; two runs of the same (draw, episode)
  are identical; the schedule is recomputed here from that generator alone, on an env whose masks are all legal.

Tests that need CityFlow or the corpus skip with their reason; the scheme's own logic is tested on the simulator-free
``FakeTrafficEnv``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.collect as collect
from tests.test_flow_randomizer import _collect_argv, _patch_make_env

REPO = Path(__file__).resolve().parents[1]


def _episodes(directory: Path) -> list[Path]:
    return sorted(directory.glob("ep*.npz"))


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


# ----------------------------------------------------------------------
# The scheme, on the simulator-free env
# ----------------------------------------------------------------------


def test_the_default_is_per_draw_and_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_make_env(monkeypatch)
    out = tmp_path / "corpus"
    assert collect.main(_collect_argv(out, "--flow-draws", "1", "2", "--base-seed", "1000")) == 0
    meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_metadata"]
    assert meta["random_rng_scheme"] == "per-draw"


def test_per_draw_replays_one_schedule_in_every_draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The registered defect, reproduced: the generator restarts at ``default_rng(base_seed)`` in every draw."""
    _patch_make_env(monkeypatch)
    out = tmp_path / "corpus"
    assert collect.main(_collect_argv(out, "--flow-draws", "1", "2", "--base-seed", "1000")) == 0
    first, second = (_arrays(path) for path in _episodes(out))
    assert np.array_equal(first["ix0_action"], second["ix0_action"])


def test_per_episode_gives_every_episode_its_own_generator_and_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_make_env(monkeypatch)
    runs = []
    for name in ("a", "b"):
        out = tmp_path / name
        argv = _collect_argv(out, "--flow-draws", "1", "2", "--base-seed", "1000", "--random-rng", "per-episode")
        argv[argv.index("--episodes") + 1] = "2"
        assert collect.main(argv) == 0
        runs.append(out)
    files_a, files_b = _episodes(runs[0]), _episodes(runs[1])
    assert [p.name for p in files_a] == [p.name for p in files_b] and len(files_a) == 4
    for left, right in zip(files_a, files_b):
        a, b = _arrays(left), _arrays(right)
        assert sorted(a) == sorted(b)
        assert all(np.array_equal(a[key], b[key]) for key in a), left.name
    def joint_actions(path: Path) -> tuple[tuple[int, ...], ...]:
        arrays = _arrays(path)
        return tuple(tuple(arrays[f"ix{i}_action"].tolist()) for i in range(len(arrays["ix_ids"])))

    actions = [joint_actions(p) for p in files_a]
    assert len(set(actions)) == 4, "two episodes of one draw, or one episode of two draws, replayed a schedule"
    meta = json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))["run_metadata"]
    assert meta["random_rng_scheme"] == "per-episode"


def test_per_episode_is_default_rng_of_base_seed_draw_and_episode_within_the_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent route: the generator the collector must have used, rebuilt from the scheme's definition alone and
    driven through ``collect._random_legal_action`` -- the same draw per intersection per step the policy makes."""
    import copy

    _patch_make_env(monkeypatch)
    states: list[dict[str, Any]] = []
    real = collect._make_random

    def spy(env: Any, args: Any, rng: np.random.Generator) -> Any:
        states.append(copy.deepcopy(rng.bit_generator.state))  # before the policy draws anything
        return real(env, args, rng)

    monkeypatch.setitem(collect.POLICIES, "random", spy)
    out = tmp_path / "corpus"
    argv = _collect_argv(out, "--flow-draws", "3", "7", "--base-seed", "1000", "--random-rng", "per-episode")
    argv[argv.index("--episodes") + 1] = "2"
    assert collect.main(argv) == 0
    expected = [(1000, draw, index) for draw in (3, 7) for index in (0, 1)]
    assert states == [np.random.default_rng(list(key)).bit_generator.state for key in expected]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--policy", "maxpressure", "--flow-draws", "1"), "random policy"),
        ((), "draw"),
    ],
)
def test_per_episode_is_refused_where_it_has_no_meaning_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...], message: str
) -> None:
    _patch_make_env(monkeypatch)
    out = tmp_path / "corpus"
    argv = _collect_argv(out, "--random-rng", "per-episode")
    if "--policy" in extra:
        argv[argv.index("--policy") + 1] = extra[extra.index("--policy") + 1]
        extra = extra[2:]
    with pytest.raises(ValueError, match=message):
        collect.main([*argv, *extra])
    assert not out.exists()


def test_a_non_random_policy_records_no_scheme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_make_env(monkeypatch)
    out = tmp_path / "corpus"
    argv = _collect_argv(out, "--flow-draws", "1")
    argv[argv.index("--policy") + 1] = "maxpressure"
    monkeypatch.setitem(collect.POLICIES, "maxpressure", lambda env, args, rng: (lambda info: np.zeros(2, dtype=np.int64)))
    assert collect.main(argv) == 0
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_metadata"]["random_rng_scheme"] is None


# ----------------------------------------------------------------------
# The registered episode, regenerated byte for byte (CityFlow + corpus)
# ----------------------------------------------------------------------


def _registered_random_episode() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    root = Path(env_value) if env_value else REPO / "datasets_v11"
    path = root / "cf_hz1x1__random" / "ep000000_seed1000_draw1.npz"
    if not path.is_file():
        pytest.skip(f"{path} not found: set RLTRAFFIC_CORPUS_V11 to the collected datasets_v11/ to regenerate it")
    return path


def _cityflow() -> None:
    pytest.importorskip("cityflow", reason="CityFlow is not installed, so no episode can be collected here")


def _registered_argv(manifest: dict[str, Any], out: Path, *extra: str) -> list[str]:
    meta = manifest["run_metadata"]
    params = meta["flow_randomizer_params"]
    return [
        "--backend", meta["backend"], "--env-config", str(REPO / "configs" / "sim" / "cityflow1x1.json"),
        "--policy", meta["behavior_policy"], "--episodes", "1", "--base-seed", str(meta["base_seed"]),
        "--max-steps", str(meta["max_steps"]), "--delta-time", str(meta["delta_time"]),
        "--control-mode", meta["control_mode"], "--global-reward-fn", meta["global_reward_fn"],
        "--local-reward-fn", meta["local_reward_fn"], "--global-reward-weight", str(meta["global_reward_weight"]),
        "--state-features", *meta["state_features"],
        "--flow-jitter-sigma", str(params["jitter_sigma_s"]), "--flow-thin-p", str(params["thin_p"]),
        "--flow-volume-scale", str(params["volume_scale"]),
        "--flow-draws", "1", "--out-dir", str(out), *extra,
    ]


def test_the_default_regenerates_a_registered_random_episode_byte_for_byte(tmp_path: Path) -> None:
    """⭐ A27(a): the default reproduces the registered behaviour, shown on a registered episode."""
    _cityflow()
    registered = _registered_random_episode()
    manifest = json.loads((registered.parent / "manifest.json").read_text(encoding="utf-8"))
    out = tmp_path / "recollected"
    assert collect.main(_registered_argv(manifest, out)) == 0
    regenerated = out / "ep000000_seed1000_draw1.npz"
    new_meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_metadata"]
    assert new_meta["flow_source_sha256"] == manifest["run_metadata"]["flow_source_sha256"]
    assert new_meta["flow_draw_sha256"]["1"] == manifest["run_metadata"]["flow_draw_sha256"]["1"]
    mine, theirs = _arrays(regenerated), _arrays(registered)
    assert sorted(mine) == sorted(theirs)
    for key in sorted(theirs):
        assert mine[key].dtype == theirs[key].dtype and mine[key].shape == theirs[key].shape, key
        assert mine[key].tobytes() == theirs[key].tobytes(), key


def test_per_episode_on_cityflow_differs_from_the_registered_schedule_and_follows_its_generator(tmp_path: Path) -> None:
    _cityflow()
    registered = _registered_random_episode()
    manifest = json.loads((registered.parent / "manifest.json").read_text(encoding="utf-8"))
    out = tmp_path / "fresh"
    argv = _registered_argv(manifest, out, "--random-rng", "per-episode")
    argv[argv.index("--max-steps") + 1] = "40"
    assert collect.main(argv) == 0
    fresh = _arrays(out / "ep000000_seed1000_draw1.npz")
    assert bool(fresh["ix0_avail_mask"].all()), "an all-legal mask is what makes the schedule the generator's alone"
    rng = np.random.default_rng([1000, 1, 0])
    expected = [int(rng.integers(fresh["ix0_avail_mask"].shape[1])) for _ in range(fresh["ix0_action"].shape[0])]
    assert fresh["ix0_action"].tolist() == expected
    assert fresh["ix0_action"].tolist() != _arrays(registered)["ix0_action"][:40].tolist()
