"""Tests for ``offline.trajectory_logger`` -- the offline corpus format, version "1.0".

No simulator is required: ``FakeTrafficEnv`` honours contracts C1/C2 from
``docs/CONTRACTS.md`` with scripted deterministic dynamics, two intersections of
*different* state widths (4 and 6) and *different* action counts (2 and 3), and three
lanes whose insertion order is deliberately not sorted.

The load-bearing test is :func:`test_global_reward_recomputed_exactly`: the fake env
is scripted so that ``r_t = -sum(waiting counts in the post-step state)``, and the
test recomputes that by a different route -- a plain ``np.sum`` over the stored
``lane_waiting_vehicle_count[t+1]`` -- asserting exact integer equality.  A loose
tolerance there would let an off-by-one into the corpus format.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline.trajectory_logger import (
    FORMAT_VERSION,
    Episode,
    LoggerStateError,
    TrajectoryLogger,
    load_episode,
)
import offline.trajectory_logger as trajectory_logger

# ----------------------------------------------------------------------
# Fake environment (C1 / C2)
# ----------------------------------------------------------------------

# (intersection id, state_dim, n_actions) -- widths differ on purpose: the format
# must never pad across intersections.
#
# The ids are chosen so that env order and lexicographic order DISAGREE: the format
# stores ix_ids in env.intersections order (C1), so that ix{i}_action is exactly
# action[i].  Ids that happened to be already sorted would let a later "fix" to the
# brief's word "sorted" pass the whole suite while breaking that identity on any real
# roadnet (intersection_10_1 sorts before intersection_2_1).
IX_SPECS: tuple[tuple[str, int, int], ...] = (
    ("ix_zulu", 4, 2),
    ("ix_alpha", 6, 3),
)
IX_IDS: tuple[str, ...] = tuple(spec[0] for spec in IX_SPECS)
assert IX_IDS != tuple(sorted(IX_IDS)), "fixture must distinguish env order from sorted"

# Insertion order is NOT sorted, so a logger that forgets to sort is caught.
RAW_LANES: tuple[str, ...] = ("lane_c", "lane_a", "lane_b")
SORTED_LANES: tuple[str, ...] = tuple(sorted(RAW_LANES))

T = 5


class FakeIntersection:
    """Stands in for ``utils.common_utils.IntersectionInfo``."""

    def __init__(self, ix_id: str, num_phases: int) -> None:
        self.id = ix_id
        self.num_phases = num_phases


class FakeTrafficEnv:
    """Deterministic, simulator-free env honouring C1 and C2.

    Deliberately exposes no ``action_space``, so ``Utils.infer_action_counts`` takes
    its ``ix.num_phases`` fallback path.

    Every reward the env emits is also appended to :attr:`emitted_global_rewards` and
    :attr:`emitted_local_rewards`, and every lane-waiting dict to
    :attr:`emitted_lane_waiting`, so a test can compare the logged arrays against what
    the env actually produced rather than against a formula the logger might share.
    Re-deriving those values instead is a trap: ``_waiting`` depends on
    ``_last_action``, so replaying it with a fresh env silently computes a demand the
    run never saw.
    """

    def __init__(
        self,
        *,
        max_steps: int = T,
        delta_time: int = 10,
        demand_scale: int = 1,
        with_local_reward: bool = True,
    ) -> None:
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.intersections = [
            FakeIntersection(ix_id, n_actions) for ix_id, _dim, n_actions in IX_SPECS
        ]
        self.lane_names: list[str] = list(RAW_LANES)
        self._demand_scale = demand_scale
        self._with_local_reward = with_local_reward
        self._step_count = 0
        self._last_action = np.zeros(len(IX_SPECS), dtype=np.int64)
        self.emitted_global_rewards: list[float] = []
        self.emitted_local_rewards: list[dict[str, float]] = []
        self.emitted_lane_waiting: list[dict[str, int]] = []
        self.emitted_lane_vehicle_count: list[dict[str, int]] = []
        self.emitted_metrics: list[dict[str, float]] = []
        self.emitted_intersections: list[dict[str, dict[str, Any]]] = []

    # -- scripted dynamics -------------------------------------------------

    def _waiting(self, lane_idx: int) -> int:
        raw = self._step_count * 3 + lane_idx * 7 + int(self._last_action.sum())
        return (raw % 5) * self._demand_scale

    def _vehicles(self, lane_idx: int) -> int:
        return self._waiting(lane_idx) + lane_idx + 1

    def _state(self, ix_idx: int) -> list[float]:
        dim = IX_SPECS[ix_idx][1]
        base = self._step_count * 10 + ix_idx * 100
        return [float(base + k) for k in range(dim)]

    def _avail_actions(self, ix_idx: int) -> list[int]:
        n_actions = IX_SPECS[ix_idx][2]
        if self._step_count % 2 == 0:
            return list(range(n_actions))
        # Drop exactly one action so the mask is not trivially all-True.
        dropped = self._step_count % n_actions
        return [a for a in range(n_actions) if a != dropped]

    def _local_reward(self, ix_idx: int) -> float:
        return float(100 * (ix_idx + 1) + self._step_count)

    # -- C2 info dict ------------------------------------------------------

    def _build_info(self) -> dict[str, Any]:
        lane_vehicle_count = {
            lane: self._vehicles(i) for i, lane in enumerate(self.lane_names)
        }
        lane_waiting = {
            lane: self._waiting(i) for i, lane in enumerate(self.lane_names)
        }

        intersections: dict[str, dict[str, Any]] = {}
        for i, (ix_id, _dim, _n_actions) in enumerate(IX_SPECS):
            entry: dict[str, Any] = {
                "state": self._state(i),
                "avail_actions": self._avail_actions(i),
                "current_phase": int(self._step_count % IX_SPECS[i][2]),
                "time_in_phase": int((self._step_count * self.delta_time) % 30),
                "action_applied": True,
                "metrics": {"local_queue": float(i + self._step_count)},
            }
            if self._with_local_reward:
                entry["reward"] = self._local_reward(i)
            intersections[ix_id] = entry

        return {
            "sim_time": float(self._step_count * self.delta_time),
            "vehicle_count": int(sum(lane_vehicle_count.values())),
            "step": self._step_count,
            "average_travel_time": 12.5,
            "lane_vehicle_count": lane_vehicle_count,
            "lane_waiting_vehicle_count": lane_waiting,
            # Insertion order is not sorted, so a logger that trusts it is caught.
            "metrics": {
                "queue_length": float(sum(lane_waiting.values())),
                "average_travel_time": 12.5,
            },
            "intersections": intersections,
        }

    # -- C1 API ------------------------------------------------------------

    def _record_emitted(self, info: dict[str, Any]) -> None:
        """Snapshot every observable the logger is expected to store.

        Tests compare against these snapshots rather than re-deriving them: sharing a
        formula with the logger would hide a bug present in both.
        """
        self.emitted_lane_waiting.append(dict(info["lane_waiting_vehicle_count"]))
        self.emitted_lane_vehicle_count.append(dict(info["lane_vehicle_count"]))
        self.emitted_metrics.append(dict(info["metrics"]))
        self.emitted_intersections.append(
            {ix_id: dict(entry) for ix_id, entry in info["intersections"].items()}
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._step_count = 0
        self._last_action = np.zeros(len(IX_SPECS), dtype=np.int64)
        self.emitted_global_rewards = []
        self.emitted_local_rewards = []
        self.emitted_lane_waiting = []
        self.emitted_lane_vehicle_count = []
        self.emitted_metrics = []
        self.emitted_intersections = []
        info = self._build_info()
        self._record_emitted(info)
        return info

    def step(self, action: Any):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        # Order mirrors envs/base_traffic_env.py::step(): advance, then measure.
        self._last_action = action
        self._step_count += 1
        reward = float(
            -sum(self._waiting(i) for i in range(len(self.lane_names)))
        )
        info = self._build_info()
        self._record_emitted(info)
        self.emitted_global_rewards.append(reward)
        self.emitted_local_rewards.append(
            {
                ix_id: info["intersections"][ix_id]["reward"]
                for ix_id in IX_IDS
                if "reward" in info["intersections"][ix_id]
            }
        )
        terminated = False
        truncated = self._step_count >= self.max_steps
        return reward, terminated, truncated, info


# ----------------------------------------------------------------------
# Driving helpers -- the brief's exact loop shape
# ----------------------------------------------------------------------


def _policy(info: dict[str, Any]) -> np.ndarray:
    """Deterministic and always legal: rotate through the currently legal actions."""
    step = int(info["step"])
    actions = []
    for ix_id in IX_IDS:
        avail = info["intersections"][ix_id]["avail_actions"]
        actions.append(avail[step % len(avail)])
    return np.asarray(actions, dtype=np.int64)


def _run_episode(
    env: FakeTrafficEnv,
    logger: TrajectoryLogger,
    *,
    engine_seed: int = 0,
    flow_draw: int | None = None,
) -> Path:
    info = env.reset(seed=engine_seed)
    logger.on_reset(info, engine_seed=engine_seed, flow_draw=flow_draw)
    for _ in range(env.max_steps):
        action = _policy(info)
        logger.on_action(info, action)
        reward, terminated, truncated, info = env.step(action)
        logger.on_step_result(reward, terminated, truncated, info)
        if terminated or truncated:
            break
    return logger.finalize_episode()


def _collect_one(tmp_path: Path, **env_kwargs: Any) -> tuple[FakeTrafficEnv, Episode, Path]:
    env = FakeTrafficEnv(**env_kwargs)
    logger = TrajectoryLogger(env, tmp_path, run_metadata={"scenario": "fake1x2"})
    path = _run_episode(env, logger, engine_seed=1000)
    return env, load_episode(path), path


# ----------------------------------------------------------------------
# 1-3. Shapes, dtypes, no padding
# ----------------------------------------------------------------------


def test_observation_arrays_have_T_plus_1_rows(tmp_path: Path) -> None:
    _env, ep, _path = _collect_one(tmp_path)

    assert ep.episode_length == T
    for arr in (ep.vehicle_count, ep.sim_time, ep.step):
        assert arr.shape == (T + 1,)
    assert ep.metrics.shape == (T + 1, len(ep.metric_keys))
    assert ep.lane_vehicle_count.shape == (T + 1, len(SORTED_LANES))
    assert ep.lane_waiting_vehicle_count.shape == (T + 1, len(SORTED_LANES))
    assert ep.global_reward.shape == (T,)

    for i, ix_id in enumerate(ep.ix_ids):
        ix = ep.intersections[ix_id]
        assert ix.state.shape == (T + 1, IX_SPECS[i][1])
        assert ix.avail_mask.shape == (T + 1, IX_SPECS[i][2])
        assert ix.current_phase.shape == (T + 1,)
        assert ix.time_in_phase.shape == (T + 1,)
        assert ix.action.shape == (T,)
        assert ix.local_reward.shape == (T,)


def test_dtypes_match_contract(tmp_path: Path) -> None:
    _env, ep, _path = _collect_one(tmp_path)

    assert ep.vehicle_count.dtype == np.int64
    assert ep.sim_time.dtype == np.float32
    assert ep.step.dtype == np.int64
    assert ep.metrics.dtype == np.float32
    assert ep.lane_vehicle_count.dtype == np.int32
    assert ep.lane_waiting_vehicle_count.dtype == np.int32
    assert ep.global_reward.dtype == np.float32

    for ix_id in ep.ix_ids:
        ix = ep.intersections[ix_id]
        assert ix.state.dtype == np.float32
        assert ix.avail_mask.dtype == np.bool_
        assert ix.current_phase.dtype == np.int64
        assert ix.time_in_phase.dtype == np.float32
        assert ix.action.dtype == np.int64
        assert ix.local_reward.dtype == np.float32


def test_no_padding_across_intersections(tmp_path: Path) -> None:
    _env, ep, _path = _collect_one(tmp_path)

    narrow = ep.intersections["ix_zulu"]
    wide = ep.intersections["ix_alpha"]
    assert narrow.state.shape[1] == 4
    assert wide.state.shape[1] == 6
    assert narrow.avail_mask.shape[1] == 2
    assert wide.avail_mask.shape[1] == 3


def test_avail_mask_true_is_legal(tmp_path: Path) -> None:
    env, ep, _path = _collect_one(tmp_path)

    # Replay the scripted availability independently of the logger.
    replay = FakeTrafficEnv()
    for t in range(T + 1):
        replay._step_count = t
        for i, ix_id in enumerate(ep.ix_ids):
            expected_idx = replay._avail_actions(i)
            mask = ep.intersections[ix_id].avail_mask[t]
            assert mask.shape == (IX_SPECS[i][2],)
            assert sorted(np.flatnonzero(mask).tolist()) == expected_idx


# ----------------------------------------------------------------------
# 5-6. Roundtrip
# ----------------------------------------------------------------------


def test_roundtrip_load_episode(tmp_path: Path) -> None:
    env, ep, _path = _collect_one(tmp_path)

    assert ep.format_version == FORMAT_VERSION
    assert ep.ix_ids == IX_IDS
    assert set(ep.intersections) == set(IX_IDS)
    assert ep.engine_seed == 1000
    assert ep.flow_draw == -1
    assert ep.terminated is False
    assert ep.truncated is True

    replay = FakeTrafficEnv()
    for t in range(T + 1):
        replay._step_count = t
        for i, ix_id in enumerate(ep.ix_ids):
            assert ep.intersections[ix_id].state[t].tolist() == replay._state(i)

    # Actions land in env.intersections order: ix_ids[i] owns action[i].
    replay_env = FakeTrafficEnv()
    info = replay_env.reset(seed=1000)
    for t in range(T):
        action = _policy(info)
        for i, ix_id in enumerate(ep.ix_ids):
            assert ep.intersections[ix_id].action[t] == action[i]
        _r, _term, _trunc, info = replay_env.step(action)


def test_ix_ids_follow_env_order_not_sorted(tmp_path: Path) -> None:
    """The one deliberate deviation from the brief, pinned.

    The brief said ``ix_ids`` was "sorted"; C1 says the action vector is ordered by
    ``env.intersections``, and C1 outranks the brief.  This test fails if anyone
    "fixes" the code back to sorting, which would silently break the identity
    ``ix{i}_action == action[i]`` on any roadnet whose ids are not already sorted.
    """
    env, ep, _path = _collect_one(tmp_path)

    assert ep.ix_ids == IX_IDS
    assert ep.ix_ids == tuple(str(ix.id) for ix in env.intersections)
    assert ep.ix_ids != tuple(sorted(ep.ix_ids))

    # And the identity that ordering exists to protect: column i of the action vector
    # is stored under ix_ids[i], not under the i-th sorted id.
    replay = FakeTrafficEnv()
    info = replay.reset(seed=1000)
    for t in range(T):
        action = _policy(info)
        for i, ix_id in enumerate(ep.ix_ids):
            assert ep.intersections[ix_id].action[t] == action[i]
        _r, _term, _trunc, info = replay.step(action)

    # The two intersections must not have identical action traces, or the assertion
    # above would hold under a permutation too.
    traces = [ep.intersections[ix_id].action.tolist() for ix_id in ep.ix_ids]
    assert traces[0] != traces[1]


def test_lane_id_order_preserved(tmp_path: Path) -> None:
    env, ep, _path = _collect_one(tmp_path)

    assert ep.lane_ids == SORTED_LANES
    assert ep.lane_ids != RAW_LANES  # the source dict order is not the stored order

    # Compare against the dicts the env actually emitted, not against a re-derived
    # formula: sharing a formula with the logger would hide a bug in both.
    assert len(env.emitted_lane_waiting) == T + 1
    for t in range(T + 1):
        expected = [env.emitted_lane_waiting[t][lane] for lane in ep.lane_ids]
        assert ep.lane_waiting_vehicle_count[t].tolist() == expected


def test_lane_vehicle_count_content(tmp_path: Path) -> None:
    """The *other* lane array, checked for content and not just shape.

    C6 rests reward-agnosticism on both lane arrays -- pressure and PressLight need
    vehicle counts, not waiting counts -- so a bug writing the waiting dict into both
    would leave every other test green.
    """
    env, ep, _path = _collect_one(tmp_path)

    assert len(env.emitted_lane_vehicle_count) == T + 1
    for t in range(T + 1):
        expected = [env.emitted_lane_vehicle_count[t][lane] for lane in ep.lane_ids]
        assert ep.lane_vehicle_count[t].tolist() == expected

    # The two arrays must not be copies of each other.
    assert not np.array_equal(ep.lane_vehicle_count, ep.lane_waiting_vehicle_count)


def test_global_observation_content(tmp_path: Path) -> None:
    """vehicle_count, sim_time, step and the metrics matrix, by content."""
    env, ep, _path = _collect_one(tmp_path)

    assert ep.step.tolist() == list(range(T + 1))
    assert ep.sim_time.tolist() == [float(t * env.delta_time) for t in range(T + 1)]
    for t in range(T + 1):
        assert ep.vehicle_count[t] == sum(env.emitted_lane_vehicle_count[t].values())

    # metric_keys is the sorted freeze, which is NOT the fake env's insertion order.
    assert ep.metric_keys == ("average_travel_time", "queue_length")
    assert ep.metric_keys != tuple(env.emitted_metrics[0])
    assert ep.metrics.shape == (T + 1, 2)
    for t in range(T + 1):
        expected = [env.emitted_metrics[t][key] for key in ep.metric_keys]
        assert ep.metrics[t].tolist() == expected


def test_per_intersection_phase_content(tmp_path: Path) -> None:
    """current_phase and time_in_phase, by content, including the int -> float32 cast."""
    env, ep, _path = _collect_one(tmp_path)

    for t in range(T + 1):
        for i, ix_id in enumerate(ep.ix_ids):
            emitted = env.emitted_intersections[t][ix_id]
            assert ep.intersections[ix_id].current_phase[t] == emitted["current_phase"]
            assert ep.intersections[ix_id].time_in_phase[t] == float(
                emitted["time_in_phase"]
            )


# ----------------------------------------------------------------------
# 7. The load-bearing test: reward recomputation, exact equality
# ----------------------------------------------------------------------


def test_global_reward_recomputed_exactly(tmp_path: Path) -> None:
    env, ep, _path = _collect_one(tmp_path)

    assert ep.global_reward.shape == (T,)
    for t in range(T):
        # Independent route: plain np.sum over the stored post-step lane counts.
        recomputed = -np.sum(ep.lane_waiting_vehicle_count[t + 1])
        assert ep.global_reward[t] == recomputed, f"mismatch at t={t}"

    # And the same values the env actually emitted.
    assert ep.global_reward.tolist() == env.emitted_global_rewards

    # A shifted alignment must NOT accidentally satisfy the assertion above. Assert the
    # mismatch COUNT, not just inequality: "differs somewhere" would still hold on
    # near-degenerate data, and this test's whole value is that the data discriminates.
    shifted = [-int(np.sum(ep.lane_waiting_vehicle_count[t])) for t in range(T)]
    differing = sum(1 for t in range(T) if ep.global_reward[t] != shifted[t])
    assert differing >= T - 1, (
        f"only {differing}/{T} rows distinguish the correct offset from a one-step "
        "shift; the scripted demand has become too flat to pin the convention"
    )


# ----------------------------------------------------------------------
# 8-9. local_reward timing and NaN fill
# ----------------------------------------------------------------------


def test_local_reward_is_outcome_of_step_t(tmp_path: Path) -> None:
    env, ep, _path = _collect_one(tmp_path)

    for ix_id in ep.ix_ids:
        logged = ep.intersections[ix_id].local_reward
        for t in range(T):
            assert logged[t] == env.emitted_local_rewards[t][ix_id]
            if t >= 1:
                # An off-by-one would put step t-1's value here.
                assert logged[t] != env.emitted_local_rewards[t - 1][ix_id]

    # Values differ across intersections too, so a swap would be caught.
    assert (
        ep.intersections["ix_zulu"].local_reward.tolist()
        != ep.intersections["ix_alpha"].local_reward.tolist()
    )


def test_local_reward_nan_when_key_absent(tmp_path: Path) -> None:
    _env, ep, _path = _collect_one(tmp_path, with_local_reward=False)

    for ix_id in ep.ix_ids:
        logged = ep.intersections[ix_id].local_reward
        assert logged.shape == (T,)
        assert np.isnan(logged).all()


# ----------------------------------------------------------------------
# 10-13. State-machine misuse
# ----------------------------------------------------------------------


def test_step_result_before_action_raises(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    info = env.reset(seed=1)
    logger.on_reset(info, engine_seed=1)

    reward, terminated, truncated, next_info = env.step(np.zeros(2, dtype=np.int64))
    with pytest.raises(LoggerStateError):
        logger.on_step_result(reward, terminated, truncated, next_info)


def test_action_twice_in_a_row_raises(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    info = env.reset(seed=1)
    logger.on_reset(info, engine_seed=1)

    action = _policy(info)
    logger.on_action(info, action)
    with pytest.raises(LoggerStateError):
        logger.on_action(info, action)


def test_lane_id_set_change_mid_episode_raises(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    info = env.reset(seed=1)
    logger.on_reset(info, engine_seed=1)

    action = _policy(info)
    logger.on_action(info, action)
    env.lane_names[0] = "lane_zzz"  # silent topology change
    reward, terminated, truncated, next_info = env.step(action)
    with pytest.raises(LoggerStateError):
        logger.on_step_result(reward, terminated, truncated, next_info)


def test_stale_info_to_on_action_raises(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    info = env.reset(seed=1)
    logger.on_reset(info, engine_seed=1)

    stale = dict(info)
    stale["step"] = 99
    with pytest.raises(LoggerStateError):
        logger.on_action(stale, _policy(info))


def test_finalize_with_dangling_action_raises(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    info = env.reset(seed=1)
    logger.on_reset(info, engine_seed=1)
    logger.on_action(info, _policy(info))

    with pytest.raises(LoggerStateError):
        logger.finalize_episode()


# ----------------------------------------------------------------------
# 14-15. Manifest and filename
# ----------------------------------------------------------------------


def _episode_sha256(ep: Episode) -> str:
    """Recompute the episode digest from the reloaded arrays."""
    digest = hashlib.sha256()
    for ix_id in ep.ix_ids:
        digest.update(ep.intersections[ix_id].action.astype("<i8", copy=False).tobytes())
    digest.update(ep.global_reward.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def test_manifest_contents(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(
        env, tmp_path, run_metadata={"scenario": "fake1x2", "backend": "fake"}
    )
    path_a = _run_episode(env, logger, engine_seed=1000)
    path_b = _run_episode(env, logger, engine_seed=1001)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["lane_count"] == len(SORTED_LANES)
    assert manifest["lane_ids_sha256"] == hashlib.sha256(
        "\n".join(SORTED_LANES).encode("utf-8")
    ).hexdigest()
    assert manifest["run_metadata"]["scenario"] == "fake1x2"
    assert "git_hash" in manifest

    assert len(manifest["episodes"]) == 2
    for entry, path in zip(manifest["episodes"], (path_a, path_b)):
        assert entry["filename"] == path.name
        assert entry["episode_length"] == T
        assert entry["flow_draw"] == -1
        ep = load_episode(path)
        assert entry["episode_sha256"] == _episode_sha256(ep)
        assert entry["total_global_reward"] == pytest.approx(
            float(np.sum(ep.global_reward))
        )
    assert manifest["episodes"][0]["engine_seed"] == 1000
    assert manifest["episodes"][1]["engine_seed"] == 1001


def test_reusing_a_populated_out_dir_is_refused(tmp_path: Path) -> None:
    """A restart into a used out_dir would truncate the manifest and orphan .npz files.

    With deterministic demand the overwritten episode file is byte-identical to the one
    it replaces, so the corruption would leave no trace: a manifest reporting N
    episodes for a corpus of 2N.
    """
    _env, ep_first, path_first = _collect_one(tmp_path)
    assert path_first.exists()

    with pytest.raises(FileExistsError) as excinfo:
        TrajectoryLogger(FakeTrafficEnv(), tmp_path)
    assert "--overwrite" in str(excinfo.value)

    # The refusal must not have touched the earlier run.
    assert path_first.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["episodes"]) == 1

    # The explicit opt-in clears the previous run and starts the counter over.
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path, overwrite=True)
    assert not path_first.exists()
    path_second = _run_episode(env, logger, engine_seed=2000)
    assert path_second.name == "ep000000_seed2000.npz"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["episodes"]) == 1
    assert manifest["episodes"][0]["engine_seed"] == 2000

    # A fresh directory is always fine.
    TrajectoryLogger(FakeTrafficEnv(), tmp_path / "fresh")


def test_unserialisable_run_metadata_fails_before_any_episode(tmp_path: Path) -> None:
    """Bad metadata must fail at construction, not after simulator time is spent."""
    with pytest.raises(ValueError) as excinfo:
        TrajectoryLogger(
            FakeTrafficEnv(), tmp_path, run_metadata={"checkpoint": Path("/x.pt")}
        )
    assert "JSON" in str(excinfo.value)

    # Nothing was written.
    assert list(tmp_path.iterdir()) == []


def test_total_global_reward_is_summed_in_float64(tmp_path: Path) -> None:
    """float32 accumulation costs ~1e-2 over a full episode; this number may be quoted."""
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    path = _run_episode(env, logger, engine_seed=1000)
    ep = load_episode(path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    reported = manifest["episodes"][0]["total_global_reward"]
    assert reported == float(np.sum(ep.global_reward, dtype=np.float64))
    # Also equals the env's own emitted rewards, summed independently in python.
    assert reported == float(sum(env.emitted_global_rewards))


def test_flow_draw_in_filename_and_scalar(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path)
    path = _run_episode(env, logger, engine_seed=1000, flow_draw=7)

    assert path.name == "ep000000_seed1000_draw7.npz"
    ep = load_episode(path)
    assert ep.flow_draw == 7

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["episodes"][0]["flow_draw"] == 7


# ----------------------------------------------------------------------
# 16-17. Determinism and duplicate detection
# ----------------------------------------------------------------------


def _array_bytes(ep: Episode) -> bytes:
    chunks = [
        ep.lane_vehicle_count.tobytes(),
        ep.lane_waiting_vehicle_count.tobytes(),
        ep.metrics.tobytes(),
        ep.vehicle_count.tobytes(),
        ep.sim_time.tobytes(),
        ep.step.tobytes(),
        ep.global_reward.tobytes(),
    ]
    for ix_id in ep.ix_ids:
        ix = ep.intersections[ix_id]
        chunks += [
            ix.state.tobytes(),
            ix.avail_mask.tobytes(),
            ix.current_phase.tobytes(),
            ix.time_in_phase.tobytes(),
            ix.action.tobytes(),
            ix.local_reward.tobytes(),
        ]
    return b"".join(chunks)


def _sha_from_manifest(out_dir: Path, index: int = 0) -> str:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["episodes"][index]["episode_sha256"]


def test_two_identical_runs_are_byte_identical(tmp_path: Path) -> None:
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    _env_a, ep_a, _pa = _collect_one(dir_a)
    _env_b, ep_b, _pb = _collect_one(dir_b)

    assert _array_bytes(ep_a) == _array_bytes(ep_b)
    assert _sha_from_manifest(dir_a) == _sha_from_manifest(dir_b)


def test_duplicate_and_divergent_demand_hashes(tmp_path: Path) -> None:
    dir_same = tmp_path / "same"
    dir_dup = tmp_path / "dup"
    dir_other = tmp_path / "other"

    _collect_one(dir_same)
    _collect_one(dir_dup)
    _collect_one(dir_other, demand_scale=2)

    assert _sha_from_manifest(dir_same) == _sha_from_manifest(dir_dup)
    assert _sha_from_manifest(dir_same) != _sha_from_manifest(dir_other)


# ----------------------------------------------------------------------
# 18-19. Documented contract and the collector guard
# ----------------------------------------------------------------------

ALIGNMENT_BLOCK = (
    "row t   = observation before decision t (aligned with a_t)\n"
    "row t+1 = the post-step state that r_t was computed from\n"
    "observations: T+1 rows · decisions: T rows · outcomes: T rows"
)


def _dedent_lines(text: str) -> str:
    """Strip per-line indentation so a docstring can be matched verbatim."""
    return "\n".join(line.strip() for line in text.splitlines())


def test_alignment_block_present_in_docstrings() -> None:
    module_doc = _dedent_lines(trajectory_logger.__doc__ or "")
    episode_doc = _dedent_lines(Episode.__doc__ or "")
    block = _dedent_lines(ALIGNMENT_BLOCK)

    assert block in module_doc
    assert block in episode_doc
    assert '"1.0"' in module_doc
    assert '"1.0"' in episode_doc

    # The three statements the Master chat mandated at GATE 2.
    assert "NOT sorted" in module_doc
    assert "time-limit truncations" in module_doc
    assert "base_traffic_env.py:604" in module_doc
    assert "COMPOSITE" in module_doc
    assert "base_traffic_env.py:204-219" in module_doc


def test_collect_rejects_empty_lane_set() -> None:
    from offline.collect import _require_lane_arrays

    with pytest.raises(ValueError) as excinfo:
        _require_lane_arrays([], "sumo")
    assert "--metrics" in str(excinfo.value)

    # A populated lane set is accepted.
    _require_lane_arrays(list(SORTED_LANES), "cityflow")
