"""Trajectory logger for offline RL dataset collection -- on-disk format version "1.1".

Records one compressed ``.npz`` per episode plus one ``manifest.json`` per collection
run, while an existing policy drives an existing env.  The logger *observes*: it never
calls ``env.step`` itself, and it reads exclusively from the ``info`` dicts it is
handed -- zero additional engine or metric calls, because state readout is ~98% of
simulator call volume on this platform.

Alignment convention (format version "1.1")
-------------------------------------------
row t   = observation before decision t (aligned with a_t)
row t+1 = the post-step state that r_t was computed from
observations: T+1 rows · decisions: T rows · outcomes: T rows

This holds because of the order inside ``envs/base_traffic_env.py::step()``: apply
phases -> advance the simulation -> refresh metrics -> compute the reward -> build
``info``.  Both the reward and the ``info`` returned by step ``t`` therefore describe
the state *after* step ``t``, including ``info["intersections"][j]["reward"]``, which
is ``r_t^j`` and is recorded as an outcome of ``t``.

Every ``info`` handed to the logger is recorded exactly once, at the moment it
arrives.  Termination is not a special case: row ``T`` is simply the last ``info``,
written by the last ``on_step_result``.

    callback                                        writes
    ------------------------------------------------------------------
    on_reset(info_0, ...)                           observation row 0
    on_action(info_t, a_t)                          decision row t
    on_step_result(r_t, term, trunc, info_{t+1})    outcome row t
                                                    observation row t+1

``att_per_step`` (added in v1.1)
--------------------------------
``att_per_step`` is an OBSERVATION, not an outcome.  It carries the per-step
registry metric ``average_travel_time``, read from the top-level
``info["average_travel_time"]`` key by the same callback that records
``vehicle_count`` and ``sim_time``, and it occupies a T+1 row like them -- never a
T row like ``action``, ``global_reward`` or ``local_reward``.  Row ``t`` is the
network state on arrival at decision step ``t``; row ``T`` is ``att_horizon``, the
registered primary metric (A1).  ``att_per_step.mean()`` is ``att_running_mean``,
the legacy quantity A1 exists to retire -- it is never the reported metric.  It is
NOT ``average_time_of_journey``.

The name deliberately departs from the ``info`` key it is read from -- the only field
in this format that does.  ``ep.average_travel_time`` would be a ``(T+1,)`` series
whose most likely misuse, ``.mean()``, reproduces ``att_running_mean`` exactly while
wearing the registered metric's name; that is the P8.0 defect, and freezing it into a
public on-disk format is worse than having it in a helper.  Enforced mechanically by
``tests/test_offline_naming_guard.py``.

Two properties measured on real data, both of which a consumer will otherwise assume
wrongly:

- **It is NOT monotone.**  ``metrics/cityflow.py::_average_travel_time`` averages over
  completed *plus currently-active* vehicles, so a vehicle entering with ~0 elapsed
  time pulls the mean down.  Measured 2026-08-06 on 407 real corpus episodes: 407 of
  407 contain at least one decrease, the earliest at row 4 of 361.  Any cumulative or
  returns-to-go style treatment of this array is therefore wrong.
- **It is independent of the requested metric set.**  ``info["average_travel_time"]``
  is top-level and is read through the metrics object regardless of whether
  ``average_travel_time`` was requested (``envs/cityflow_env.py:269-272`` falls back to
  the raw engine only when the whole pipeline is off).  Measured bit-identical between
  a 1-metric and a 3-metric env across 3 scenarios x 2 policies, max abs diff 0.0.
  **Measured on CityFlow only** -- re-measure before any SUMO or MOSS collection.

**It is stored float32, and ``att_horizon`` computed live is float64.**  C5 mandates
float32 for stored float arrays, so ``att_per_step[-1]`` is the float32 *image* of the
metric, not the metric.  Verified against ``offline/horizon_metric.py`` on a real
360-step CityFlow episode: ``np.float32(horizon_rollout(...).att_horizon)`` and the
stored value have **identical bit patterns**, while the raw float64 comparison differs
by 5.03e-06 -- under one float32 ulp at that magnitude (1.53e-05).  A consumer checking
the corpus against a live rollout must cast to float32 first; comparing the float64
directly with ``==`` will fail for a correct corpus.

``att_per_step`` is deliberately NOT part of ``episode_sha256``.  The digest stays over
``action`` + ``global_reward``, so it identifies a *trajectory*: it therefore compares
across the v1.0 and v1.1 corpora, and P2.4's duplicate detector keeps working unchanged.
ATT is a function of the trajectory, not part of its identity.

Three further facts a consumer of this format must know
-------------------------------------------------------
1. ``ix_ids`` is stored in ``env.intersections`` order, NOT sorted.  Contract C1 says
   the action vector is ordered by ``[ix.id for ix in env.intersections]``, so
   ``ix{i}_action`` is exactly ``action[i]`` with no remapping table.  ``lane_ids``,
   by contrast, IS sorted lexicographically and frozen at the first write.

2. Episode ends are ALWAYS time-limit truncations.  ``terminated`` is hardcoded
   ``False`` at ``envs/base_traffic_env.py:604``; only
   ``truncated = self._step_count >= self.max_steps`` (``:605``) ends an episode.
   There is no true terminal state in this platform, so downstream returns-to-go code
   cannot use ``terminated`` to distinguish a genuine absorbing state from a time
   limit -- every episode boundary is the latter.

3. ``local_reward`` is COMPOSITE, not a purely local signal.  Verified at
   ``envs/base_traffic_env.py:204-219``, intersection *j* receives
   ``global_reward_weight * global_reward + local_reward_fn(local_metrics[j])``
   (global term at ``:214``, local term at ``:217``).  Downstream returns-to-go code
   must NOT double-count the global part by summing ``global_reward`` and
   ``local_reward``.

Layout of one episode file
--------------------------
Global observations (``T+1`` rows): ``vehicle_count`` int64, ``sim_time`` float32,
``step`` int64, ``att_per_step`` float32, ``metrics`` (T+1, M) float32 with
``metric_keys`` (M,).
Per-lane observations (``T+1`` rows): ``lane_vehicle_count`` (T+1, L) int32,
``lane_waiting_vehicle_count`` (T+1, L) int32, ``lane_ids`` (L,).  The ``info`` key
names are used unchanged.  These lane arrays are what make the corpus
reward-agnostic: any standard TSC reward (queue, PressLight, pressure) stays
recomputable offline, so the primary-reward decision never forces recollection.
Reward-agnosticism covers rewards computable from the stored lane counts plus the
scenario topology the manifest references (queue-family and PressLight-family both
qualify).  A reward that reads per-intersection metrics from
``info["intersections"][j]["metrics"]`` (available at ``base_traffic_env.py:478`` but
NOT stored) would require format v1.1.
Per intersection ``i``: ``ix{i}_state`` (T+1, state_dim) float32, ``ix{i}_avail_mask``
(T+1, n_actions) bool, ``ix{i}_current_phase`` (T+1,) int64, ``ix{i}_time_in_phase``
(T+1,) float32, ``ix{i}_action`` (T,) int64, ``ix{i}_local_reward`` (T,) float32.
Global outcomes (``T`` rows): ``global_reward`` float32.
Episode scalars: ``episode_length`` (= T), ``terminated``, ``truncated``,
``engine_seed``, ``flow_draw`` (``-1`` when absent), ``format_version``.

Intersections may have different ``state_dim`` and ``n_actions``; arrays are never
padded across intersections.  Only numeric and unicode arrays are stored, so
``np.load`` works at its safe default ``allow_pickle=False``.

Any change to this layout requires bumping ``FORMAT_VERSION`` and writing a migration
note; the P2.4 linter hard-fails on an unknown format version.

Migration note v1.0 -> v1.1 (2026-08-07)
----------------------------------------
v1.1 is v1.0 plus exactly one array, ``att_per_step``.  No existing array, name or
dtype changed, and the manifest is untouched.  ``load_episode`` reads both versions;
on a v1.0 file ``Episode.att_per_step`` is ``None``, which is the flag a consumer
should branch on rather than probing shapes.  A v1.0 file cannot be upgraded in place:
the per-step values were never recorded, and the metric is not recomputable from the
stored lane counts (it needs per-vehicle departure times).  Upgrading means
re-collecting.  A corpus must not mix the two versions -- see
``offline/corpus_format_check.py``.

Two caveats for P3, both unreachable with the backends as they stand today
------------------------------------------------------------------------
- ``avail_mask`` is built from ``Utils.extract_valid_actions``, which is mandated by
  CLAUDE.md rule 5 and whose contract is "empty result means every action is legal"
  (``agent/utils/utils.py:102``); it also silently drops out-of-range indices
  (``:99``).  An env reporting ``avail_actions == []`` would therefore be logged as an
  all-True mask, not an all-False one.  No concrete ``available_actions()`` in
  ``envs/phase_control.py`` can return empty today, so no stored mask is affected --
  but P3 must not read an all-True mask as positive evidence of an unconstrained
  intersection.
- ``state`` rows are whatever ``Utils.state_from_info`` returns, and its
  ``np.asarray(..., dtype=np.float32)`` does not copy an input that is already a
  float32 ndarray.  All three backends build ``"state"`` with ``.tolist()``
  (``envs/base_traffic_env.py:471``), so a fresh list is converted every step and no
  aliasing occurs; a future backend returning a live buffer would retroactively
  rewrite rows already logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent.utils.utils import Utils

FORMAT_VERSION = "1.1"

#: Versions :func:`load_episode` can read.  The 4800-episode v1.0 corpus predates the
#: ``att_per_step`` field and must keep loading, so the reader accepts a set while the
#: writer emits exactly :data:`FORMAT_VERSION`.  An unknown version is still a hard
#: failure -- guessing an alignment is the one thing this format may never do.
SUPPORTED_FORMAT_VERSIONS: tuple[str, ...] = ("1.0", "1.1")

#: Arrays absent from a v1.0 file; :class:`Episode` carries ``None`` for each.
_V11_ONLY_ARRAYS: tuple[str, ...] = ("att_per_step",)

MANIFEST_NAME = "manifest.json"

# Callback-order states.  The name describes what the logger is waiting for.
_IDLE = "idle"
_AWAIT_ACTION = "await_action"
_AWAIT_STEP = "await_step"

__all__ = [
    "FORMAT_VERSION",
    "SUPPORTED_FORMAT_VERSIONS",
    "Episode",
    "IntersectionEpisode",
    "LoggerStateError",
    "TrajectoryLogger",
    "load_episode",
]


class LoggerStateError(RuntimeError):
    """Raised when the callback order or the frozen episode schema is violated.

    Both failure classes are grouped deliberately: a caller that drives the logger
    out of order and a caller whose env silently changes its lane set mid-episode
    corrupt the corpus in the same way, and both must abort the run rather than be
    reindexed around.
    """


@dataclass(frozen=True)
class IntersectionEpisode:
    """Per-intersection arrays for one episode.

    Observations carry ``T+1`` rows, decisions and outcomes carry ``T``; see the
    module docstring for the alignment convention (format version "1.1").
    """

    state: np.ndarray
    avail_mask: np.ndarray
    current_phase: np.ndarray
    time_in_phase: np.ndarray
    action: np.ndarray
    local_reward: np.ndarray


@dataclass(frozen=True)
class Episode:
    """One episode reloaded from disk -- the reload contract the P3 dataset builds on.

    Alignment convention (format version "1.1")
    -------------------------------------------
    row t   = observation before decision t (aligned with a_t)
    row t+1 = the post-step state that r_t was computed from
    observations: T+1 rows · decisions: T rows · outcomes: T rows

    ``att_per_step`` is an OBSERVATION, not an outcome.  It carries the per-step
    registry metric ``average_travel_time``, read from the top-level
    ``info["average_travel_time"]`` key by the same callback that records
    ``vehicle_count`` and ``sim_time``, and it occupies a T+1 row like them -- never a
    T row like ``action``, ``global_reward`` or ``local_reward``.  Row ``t`` is the
    network state on arrival at decision step ``t``; row ``T`` is ``att_horizon``, the
    registered primary metric (A1).  ``att_per_step.mean()`` is ``att_running_mean``,
    the legacy quantity A1 exists to retire -- it is never the reported metric.  It is
    NOT ``average_time_of_journey``.

    ``att_per_step`` is ``None`` when this episode was written at format version "1.0",
    which predates the field.  Branch on that rather than on the array's shape.

    ``intersections`` is keyed by real intersection id; ``ix_ids`` preserves
    ``env.intersections`` order, so ``ix_ids[i]`` owns ``action[i]`` of the vector
    passed to ``env.step``.  ``lane_ids`` is sorted lexicographically and indexes the
    columns of both lane arrays; ``metric_keys`` indexes the columns of ``metrics``.

    ``flow_draw`` keeps the on-disk ``-1``-means-absent sentinel rather than mapping
    it to ``None``, so the format has exactly one convention for it.  ``terminated``
    is always ``False`` on this platform -- see fact 2 in the module docstring.
    """

    format_version: str
    ix_ids: tuple[str, ...]
    intersections: dict[str, IntersectionEpisode]
    lane_ids: tuple[str, ...]
    lane_vehicle_count: np.ndarray
    lane_waiting_vehicle_count: np.ndarray
    metric_keys: tuple[str, ...]
    metrics: np.ndarray
    vehicle_count: np.ndarray
    sim_time: np.ndarray
    step: np.ndarray
    att_per_step: np.ndarray | None
    global_reward: np.ndarray
    episode_length: int
    terminated: bool
    truncated: bool
    engine_seed: int
    flow_draw: int


# ----------------------------------------------------------------------
# Small array helpers
# ----------------------------------------------------------------------


def _str_array(values: Sequence[str]) -> np.ndarray:
    """Unicode array that survives an empty input without collapsing to float64."""
    if not values:
        return np.empty(0, dtype="<U1")
    return np.asarray([str(v) for v in values], dtype=np.str_)


def _rows_2d(rows: Sequence[Sequence[Any]], dtype: Any, width: int) -> np.ndarray:
    """Stack per-step rows into ``(len(rows), width)``, correct for zero-size cases."""
    return np.asarray(rows, dtype=dtype).reshape(len(rows), width)


def _rows_1d(values: Sequence[Any], dtype: Any) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(len(values))


def _repo_git_hash() -> str:
    """``git rev-parse HEAD`` for the repo holding this module, ``"unknown"`` if absent.

    Resolved against this file's directory rather than the process cwd: the manifest
    records which revision of *this code* produced the corpus, which is not
    necessarily the tree the collector was launched from.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


# ----------------------------------------------------------------------
# Reload
# ----------------------------------------------------------------------


def load_episode(path: str | Path) -> Episode:
    """Load one episode ``.npz`` written by :class:`TrajectoryLogger`.

    Reads every version in :data:`SUPPORTED_FORMAT_VERSIONS`; fields a version predates
    arrive as ``None`` (v1.0 has no ``att_per_step``).  Hard-fails on an unknown
    ``format_version`` rather than guessing an alignment.
    """
    path = Path(path)
    with np.load(path) as data:
        version = str(data["format_version"].item())
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"{path}: format version {version!r} is not readable by this build "
                f"(supported: {list(SUPPORTED_FORMAT_VERSIONS)}); see the migration "
                "note for that version"
            )

        ix_ids = tuple(str(v) for v in data["ix_ids"].tolist())
        intersections = {
            ix_id: IntersectionEpisode(
                state=data[f"ix{i}_state"],
                avail_mask=data[f"ix{i}_avail_mask"],
                current_phase=data[f"ix{i}_current_phase"],
                time_in_phase=data[f"ix{i}_time_in_phase"],
                action=data[f"ix{i}_action"],
                local_reward=data[f"ix{i}_local_reward"],
            )
            for i, ix_id in enumerate(ix_ids)
        }

        return Episode(
            format_version=version,
            ix_ids=ix_ids,
            intersections=intersections,
            lane_ids=tuple(str(v) for v in data["lane_ids"].tolist()),
            lane_vehicle_count=data["lane_vehicle_count"],
            lane_waiting_vehicle_count=data["lane_waiting_vehicle_count"],
            metric_keys=tuple(str(v) for v in data["metric_keys"].tolist()),
            metrics=data["metrics"],
            vehicle_count=data["vehicle_count"],
            sim_time=data["sim_time"],
            step=data["step"],
            # Keyed off what the file actually contains, not off the version string:
            # the two agree by construction, and a file whose version claims v1.1 while
            # missing the array should surface as the linter's problem, not as a
            # KeyError from inside the loader.
            att_per_step=(
                data["att_per_step"] if "att_per_step" in data.files else None
            ),
            global_reward=data["global_reward"],
            episode_length=int(data["episode_length"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            engine_seed=int(data["engine_seed"]),
            flow_draw=int(data["flow_draw"]),
        )


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------


class TrajectoryLogger:
    """Records trajectories to disk while an existing policy runs.

    See the module docstring for the alignment convention and the on-disk layout.
    One ``out_dir`` belongs to one process: the manifest is rewritten in full on every
    ``finalize_episode`` and two concurrent writers would lose each other's episodes.
    """

    def __init__(
        self,
        env: Any,
        out_dir: str | Path,
        run_metadata: Mapping[str, Any] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        self._env = env
        self._out_dir = Path(out_dir)

        # ---- validate everything BEFORE touching the filesystem ----
        # Ordering is load-bearing, not stylistic. Deleting a previous run and only
        # then discovering that this run cannot start would destroy a corpus on behalf
        # of a run that never produces one -- inverting the very property the metadata
        # check exists to provide. A failed construction leaves the previous corpus
        # intact and creates no directories.
        ix_ids: tuple[str, ...] = tuple(str(ix.id) for ix in env.intersections)
        if not ix_ids:
            raise ValueError("env exposes no intersections; nothing to log")
        if len(set(ix_ids)) != len(ix_ids):
            raise ValueError(f"env exposes duplicate intersection ids: {ix_ids}")

        # Captured here, not at the first on_reset, so rebind_env can compare against
        # the env the logger was *constructed* with even on its first call -- which is
        # the call a flow-draw sweep makes, before any episode has run. Pure and cheap:
        # env.action_space exists by now (CityFlowEnv.__init__ runs _setup_spaces before
        # returning), and infer_action_counts falls back to ix.num_phases when it does
        # not, so this cannot fail where the old deferred call would have succeeded.
        ctor_n_actions = [
            int(count)
            for count in Utils.infer_action_counts(
                getattr(env, "action_space", None), list(env.intersections)
            )
        ]

        metadata: dict[str, Any] = dict(run_metadata or {})
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"run_metadata is not JSON-serialisable ({exc}); the manifest could "
                "not be written. Convert Path/numpy values to str/int/float first."
            ) from exc

        stale = self._existing_run_files()
        if stale and not overwrite:
            raise FileExistsError(
                f"{self._out_dir} already contains a collection run "
                f"({len(stale)} file(s), e.g. {stale[0].name}). Writing here would "
                "drop the earlier episodes from the manifest and orphan their .npz "
                "files. Choose a fresh --out-dir, or pass --overwrite to discard the "
                "previous run."
            )

        # ---- only now mutate the filesystem ----
        self._out_dir.mkdir(parents=True, exist_ok=True)
        for path in stale:
            path.unlink()

        self._ix_ids = ix_ids
        self._ctor_n_actions = ctor_n_actions
        self._run_metadata = metadata
        self._state = _IDLE
        self._episode_counter = 0
        self._git_hash = _repo_git_hash()
        self._manifest_episodes: list[dict[str, Any]] = []

        # Run-level topology, frozen at the first on_reset.
        self._run_lane_count: int | None = None
        self._run_lane_ids_sha256: str | None = None

        self._clear_episode()

    def _existing_run_files(self) -> list[Path]:
        """Files in ``out_dir`` that a previous collection run left behind. Read-only.

        Why a populated directory is refused at all: the manifest is rewritten in full
        on every ``finalize_episode`` from an in-memory list that starts empty, and the
        episode counter restarts at zero. Reusing a populated directory would therefore
        drop every earlier episode from the manifest while its ``.npz`` files stay on
        disk orphaned -- a manifest truthfully reporting 20 episodes for a corpus of
        40. With deterministic demand the overwritten file is byte-identical to the one
        it replaces, so the corruption leaves no trace at all.

        The glob matches only the exact filename shape this class writes
        (``ep`` + six digits + ``_seed``), so an unrelated ``epoch_stats.npz`` the user
        happens to keep here survives ``overwrite=True``. It is deliberately not
        recursive: subdirectories are never touched.
        """
        stale = sorted(self._out_dir.glob("ep[0-9][0-9][0-9][0-9][0-9][0-9]_seed*.npz"))
        manifest = self._out_dir / MANIFEST_NAME
        if manifest.exists():
            stale.append(manifest)
        return stale

    # -- introspection -----------------------------------------------------

    @property
    def lane_ids(self) -> tuple[str, ...]:
        """Lane ids frozen at ``on_reset``; empty before the first reset."""
        return self._lane_ids

    @property
    def metric_keys(self) -> tuple[str, ...]:
        """Global metric keys frozen at ``on_reset``; empty before the first reset."""
        return self._metric_keys

    @property
    def manifest_path(self) -> Path:
        """Path of the run manifest."""
        return self._out_dir / MANIFEST_NAME

    def rebind_env(self, env: Any) -> None:
        """Point this logger at a new env object with identical topology.

        A flow-draw sweep must build a **fresh env per draw**: CityFlow reads its flow
        file exactly once, in the engine constructor
        (``CityFlow/src/engine/engine.cpp:65``), and ``Engine::reset()`` (``:744-760``)
        only calls ``flow.reset()`` on the already-parsed in-memory vector -- it never
        re-reads the file.  One logger must nonetheless serve the whole run, because a
        fresh logger per draw would rewrite ``manifest.json`` from an empty list and
        leave the earlier draws' ``.npz`` files orphaned.

        Legal **only between episodes**.  The new env must be topologically identical to
        the one this logger was constructed with: the same intersection ids in the same
        order, and the same per-intersection action counts.  ``lane_ids`` are already
        guarded run-level via ``lane_ids_sha256``, but ``ix_ids`` are not -- a redraw
        silently rebinding onto a different topology would corrupt the corpus in exactly
        the way this format exists to prevent.

        Both halves of the check are unconditional.  ``n_actions`` is compared against
        the counts captured in ``__init__``, **not** against the last episode's: a
        flow-draw sweep rebinds for its first draw before any ``on_reset`` has run, so
        deriving the expectation from a previous episode would skip the guard precisely
        where it is first used.
        """
        if self._state != _IDLE:
            raise LoggerStateError(
                f"rebind_env called in state {self._state!r}: an episode is still open. "
                "Rebinding is legal only between episodes -- call finalize_episode() "
                "first, so the corpus cannot contain an episode split across two envs."
            )

        ix_ids = tuple(str(ix.id) for ix in env.intersections)
        if ix_ids != self._ix_ids:
            raise LoggerStateError(
                f"rebind_env got an env whose ix_ids disagree with this run's "
                f"({ix_ids} vs {self._ix_ids}); one out_dir holds one topology"
            )

        counts = [
            int(count)
            for count in Utils.infer_action_counts(
                getattr(env, "action_space", None), list(env.intersections)
            )
        ]
        if counts != self._ctor_n_actions:
            raise LoggerStateError(
                f"rebind_env got an env whose per-intersection n_actions disagree "
                f"with this run's ({counts} vs {self._ctor_n_actions}); one out_dir "
                "holds one topology"
            )

        self._env = env

    # -- episode buffers ---------------------------------------------------

    def _clear_episode(self) -> None:
        n_ix = len(self._ix_ids)
        self._lane_ids: tuple[str, ...] = ()
        self._metric_keys: tuple[str, ...] = ()
        self._n_actions: list[int] = []
        self._state_dims: list[int] = []

        self._obs_state: list[list[np.ndarray]] = [[] for _ in range(n_ix)]
        self._obs_avail_mask: list[list[np.ndarray]] = [[] for _ in range(n_ix)]
        self._obs_current_phase: list[list[int]] = [[] for _ in range(n_ix)]
        self._obs_time_in_phase: list[list[float]] = [[] for _ in range(n_ix)]
        self._obs_vehicle_count: list[int] = []
        self._obs_sim_time: list[float] = []
        self._obs_step: list[int] = []
        self._obs_att_per_step: list[float] = []
        self._obs_metrics: list[list[float]] = []
        self._obs_lane_vehicle_count: list[list[int]] = []
        self._obs_lane_waiting: list[list[int]] = []

        self._act_action: list[list[int]] = [[] for _ in range(n_ix)]
        self._out_global_reward: list[float] = []
        self._out_local_reward: list[list[float]] = [[] for _ in range(n_ix)]

        self._has_local_reward: bool | None = None
        self._last_step: int = -1
        self._engine_seed: int = -1
        self._flow_draw: int | None = None
        self._terminated: bool = False
        self._truncated: bool = False

    # -- schema validation -------------------------------------------------

    def _check_schema(self, info: dict[str, Any]) -> None:
        """Reject an ``info`` whose lane set or metric set left the frozen episode schema."""
        # v1.1. Read strictly rather than defaulting: C2 mandates this key and all three
        # backends emit it unconditionally (cityflow_env.py:269, sumo_env.py:316,
        # moss_env.py:681). A .get(..., nan) default would turn a schema violation into a
        # silent column of NaNs, discovered at P4 rather than at collection time.
        if "average_travel_time" not in info:
            raise LoggerStateError(
                "info is missing the top-level 'average_travel_time' key, which "
                "contract C2 mandates and which format v1.1 stores as 'att_per_step'. "
                "All three backends emit it unconditionally, so this is an env or "
                "adapter defect; refusing rather than logging a NaN column."
            )
        lane_counts = info["lane_vehicle_count"]
        lane_waiting = info["lane_waiting_vehicle_count"]
        frozen = set(self._lane_ids)
        if set(map(str, lane_counts)) != frozen:
            raise LoggerStateError(
                "lane_vehicle_count changed its lane set mid-episode: frozen "
                f"{sorted(frozen)} vs now {sorted(map(str, lane_counts))}. Silent "
                "reindexing would corrupt offline reward recomputation."
            )
        if set(map(str, lane_waiting)) != frozen:
            raise LoggerStateError(
                "lane_waiting_vehicle_count disagrees with the frozen lane set: "
                f"{sorted(frozen)} vs {sorted(map(str, lane_waiting))}"
            )
        metrics = info.get("metrics") or {}
        if tuple(sorted(map(str, metrics))) != self._metric_keys:
            raise LoggerStateError(
                "global metric keys changed mid-episode: frozen "
                f"{list(self._metric_keys)} vs now {sorted(map(str, metrics))}"
            )

    # -- the single observation write path ---------------------------------

    def _record_observation(self, info: dict[str, Any]) -> None:
        """Append one observation row.  Used by ``on_reset`` and ``on_step_result``."""
        lane_counts = {str(k): v for k, v in info["lane_vehicle_count"].items()}
        lane_waiting = {str(k): v for k, v in info["lane_waiting_vehicle_count"].items()}
        metrics = {str(k): v for k, v in (info.get("metrics") or {}).items()}

        payloads = Utils.extract_per_intersection_info(info, list(self._ix_ids))
        for i, ix_id in enumerate(self._ix_ids):
            payload = payloads[ix_id]

            state = Utils.state_from_info(payload)
            if int(state.shape[0]) != self._state_dims[i]:
                raise LoggerStateError(
                    f"intersection {ix_id!r} changed state width mid-episode: frozen "
                    f"{self._state_dims[i]} vs now {int(state.shape[0])}"
                )
            self._obs_state[i].append(state.astype(np.float32, copy=False))

            mask = np.zeros(self._n_actions[i], dtype=np.bool_)
            mask[Utils.extract_valid_actions(payload, self._n_actions[i])] = True
            self._obs_avail_mask[i].append(mask)

            self._obs_current_phase[i].append(int(payload["current_phase"]))
            self._obs_time_in_phase[i].append(float(payload["time_in_phase"]))

        self._obs_vehicle_count.append(int(info["vehicle_count"]))
        self._obs_sim_time.append(float(info["sim_time"]))
        self._obs_step.append(int(info["step"]))
        # v1.1. The TOP-LEVEL key, not info["metrics"]["average_travel_time"]: the
        # top-level one is present regardless of the requested metric set (verified
        # bit-identical between a 1-metric and a 3-metric env, 3 scenarios x 2 policies),
        # which is exactly what lets ATT enter the corpus without perturbing MAPPO's
        # frozen metric set under contract C8.
        self._obs_att_per_step.append(float(info["average_travel_time"]))
        self._obs_metrics.append([float(metrics[key]) for key in self._metric_keys])
        self._obs_lane_vehicle_count.append(
            [int(lane_counts[lane]) for lane in self._lane_ids]
        )
        self._obs_lane_waiting.append(
            [int(lane_waiting[lane]) for lane in self._lane_ids]
        )

        self._last_step = int(info["step"])

    # -- callbacks ---------------------------------------------------------

    def on_reset(
        self,
        info: dict[str, Any],
        *,
        engine_seed: int,
        flow_draw: int | None = None,
    ) -> None:
        """Start an episode and write observation row 0 from ``info``."""
        if self._state != _IDLE:
            raise LoggerStateError(
                f"on_reset called while an episode is open (state={self._state!r}); "
                "call finalize_episode() first"
            )

        self._clear_episode()
        self._engine_seed = int(engine_seed)
        self._flow_draw = None if flow_draw is None else int(flow_draw)

        # Freeze the episode schema.  lane_ids is sorted; ix order is env order.
        self._lane_ids = tuple(sorted(str(k) for k in info["lane_vehicle_count"]))
        self._metric_keys = tuple(
            sorted(str(k) for k in (info.get("metrics") or {}))
        )
        self._freeze_run_topology()

        counts = Utils.infer_action_counts(
            getattr(self._env, "action_space", None), list(self._env.intersections)
        )
        if len(counts) != len(self._ix_ids):
            raise LoggerStateError(
                f"infer_action_counts returned {len(counts)} entries for "
                f"{len(self._ix_ids)} intersections"
            )
        self._n_actions = [int(c) for c in counts]

        payloads = Utils.extract_per_intersection_info(info, list(self._ix_ids))
        self._state_dims = [
            int(Utils.state_from_info(payloads[ix_id]).shape[0])
            for ix_id in self._ix_ids
        ]

        self._check_schema(info)
        self._record_observation(info)
        self._state = _AWAIT_ACTION

    def on_action(self, info: dict[str, Any], action: Any) -> None:
        """Write decision row ``t``; ``info`` is asserted to be the one already recorded.

        Nothing is read out of ``info`` here -- that dict was recorded by the previous
        callback.  The assertion is free and catches loop misuse immediately.
        """
        if self._state != _AWAIT_ACTION:
            raise LoggerStateError(
                f"on_action called in state {self._state!r}; expected {_AWAIT_ACTION!r}. "
                "Legal order is on_reset -> (on_action -> on_step_result)* -> finalize_episode"
            )

        step = int(info["step"])
        if step != self._last_step:
            raise LoggerStateError(
                f"on_action received info at step {step}, but the last recorded "
                f"observation was step {self._last_step}; the loop is feeding the "
                "logger a different info than the one the action was chosen from"
            )

        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.shape[0] != len(self._ix_ids):
            raise LoggerStateError(
                f"action has {arr.shape[0]} entries for {len(self._ix_ids)} "
                "intersections; C1 requires one int per intersection in env.intersections order"
            )
        for i in range(len(self._ix_ids)):
            self._act_action[i].append(int(arr[i]))

        self._state = _AWAIT_STEP

    def on_step_result(
        self,
        reward: Any,
        terminated: bool,
        truncated: bool,
        next_info: dict[str, Any],
    ) -> None:
        """Write outcome row ``t`` and observation row ``t+1``."""
        if self._state != _AWAIT_STEP:
            raise LoggerStateError(
                f"on_step_result called in state {self._state!r}; expected "
                f"{_AWAIT_STEP!r}. Every step result must be preceded by on_action"
            )

        # Validate before writing anything, so a rejected info leaves no partial row.
        self._check_schema(next_info)
        payloads = Utils.extract_per_intersection_info(next_info, list(self._ix_ids))

        # Utils.reward_for_intersection is deliberately NOT used here: its fallback
        # chain (agent/utils/utils.py:69-75) substitutes the *global* scalar when
        # "reward" is absent, which is exactly the case the format requires to be NaN.
        present = ["reward" in payloads[ix_id] for ix_id in self._ix_ids]
        if self._has_local_reward is None:
            self._has_local_reward = bool(present[0])
        if any(flag is not self._has_local_reward for flag in present):
            raise LoggerStateError(
                "the 'reward' key appears for some intersections and not others "
                f"(episode frozen as has_local_reward={self._has_local_reward}); "
                "local_reward_fn cannot change mid-episode"
            )

        self._out_global_reward.append(float(Utils.scalar_reward(reward)))
        for i, ix_id in enumerate(self._ix_ids):
            payload = payloads[ix_id]
            value = float(payload["reward"]) if self._has_local_reward else float("nan")
            self._out_local_reward[i].append(value)

        self._terminated = bool(terminated)
        self._truncated = bool(truncated)

        self._record_observation(next_info)
        self._state = _AWAIT_ACTION

    # -- writing -----------------------------------------------------------

    def finalize_episode(self) -> Path:
        """Write the episode ``.npz``, update the manifest atomically, return the path."""
        if self._state != _AWAIT_ACTION:
            if self._state == _AWAIT_STEP:
                raise LoggerStateError(
                    "finalize_episode called after on_action with no matching "
                    "on_step_result; the last decision has no outcome"
                )
            raise LoggerStateError(
                f"finalize_episode called in state {self._state!r}; no episode is open"
            )

        arrays = self._build_arrays()
        episode_sha256 = self._episode_digest(arrays)

        filename = f"ep{self._episode_counter:06d}_seed{self._engine_seed}"
        if self._flow_draw is not None:
            filename += f"_draw{self._flow_draw}"
        filename += ".npz"
        path = self._out_dir / filename
        with open(path, "wb") as handle:
            np.savez_compressed(handle, **arrays)

        self._manifest_episodes.append(
            {
                "filename": filename,
                "episode_length": int(arrays["episode_length"]),
                # Summed in float64: the stored rewards are float32 by contract, but
                # accumulating 360 of them in float32 costs ~1e-2 absolute at typical
                # magnitudes, and this number may end up quoted in the paper.
                "total_global_reward": float(
                    np.sum(arrays["global_reward"], dtype=np.float64)
                ),
                "engine_seed": self._engine_seed,
                "flow_draw": -1 if self._flow_draw is None else self._flow_draw,
                "episode_sha256": episode_sha256,
            }
        )
        self._write_manifest()

        self._episode_counter += 1
        self._state = _IDLE
        return path

    def _build_arrays(self) -> dict[str, np.ndarray]:
        n_steps = len(self._out_global_reward)
        n_obs = len(self._obs_step)
        if n_obs != n_steps + 1:  # pragma: no cover - guarded by the state machine
            raise LoggerStateError(
                f"internal alignment violation: {n_obs} observation rows for "
                f"{n_steps} decisions (expected {n_steps + 1})"
            )

        arrays: dict[str, np.ndarray] = {
            "format_version": np.asarray(FORMAT_VERSION),
            "ix_ids": _str_array(self._ix_ids),
            "lane_ids": _str_array(self._lane_ids),
            "metric_keys": _str_array(self._metric_keys),
            "vehicle_count": _rows_1d(self._obs_vehicle_count, np.int64),
            "sim_time": _rows_1d(self._obs_sim_time, np.float32),
            "step": _rows_1d(self._obs_step, np.int64),
            "att_per_step": _rows_1d(self._obs_att_per_step, np.float32),
            "metrics": _rows_2d(
                self._obs_metrics, np.float32, len(self._metric_keys)
            ),
            "lane_vehicle_count": _rows_2d(
                self._obs_lane_vehicle_count, np.int32, len(self._lane_ids)
            ),
            "lane_waiting_vehicle_count": _rows_2d(
                self._obs_lane_waiting, np.int32, len(self._lane_ids)
            ),
            "global_reward": _rows_1d(self._out_global_reward, np.float32),
            "episode_length": np.asarray(n_steps, dtype=np.int64),
            "terminated": np.asarray(self._terminated, dtype=np.bool_),
            "truncated": np.asarray(self._truncated, dtype=np.bool_),
            "engine_seed": np.asarray(self._engine_seed, dtype=np.int64),
            "flow_draw": np.asarray(
                -1 if self._flow_draw is None else self._flow_draw, dtype=np.int64
            ),
        }

        for i in range(len(self._ix_ids)):
            arrays[f"ix{i}_state"] = _rows_2d(
                self._obs_state[i], np.float32, self._state_dims[i]
            )
            arrays[f"ix{i}_avail_mask"] = _rows_2d(
                self._obs_avail_mask[i], np.bool_, self._n_actions[i]
            )
            arrays[f"ix{i}_current_phase"] = _rows_1d(
                self._obs_current_phase[i], np.int64
            )
            arrays[f"ix{i}_time_in_phase"] = _rows_1d(
                self._obs_time_in_phase[i], np.float32
            )
            arrays[f"ix{i}_action"] = _rows_1d(self._act_action[i], np.int64)
            arrays[f"ix{i}_local_reward"] = _rows_1d(
                self._out_local_reward[i], np.float32
            )

        return arrays

    def _episode_digest(self, arrays: Mapping[str, np.ndarray]) -> str:
        """sha256 over the action arrays (env order) then ``global_reward``.

        Byte order is pinned to little-endian rather than left native, so the digest
        the P2.4 duplicate detector compares does not change when the corpus moves to
        another machine.
        """
        digest = hashlib.sha256()
        for i in range(len(self._ix_ids)):
            digest.update(arrays[f"ix{i}_action"].astype("<i8", copy=False).tobytes())
        digest.update(arrays["global_reward"].astype("<f4", copy=False).tobytes())
        return digest.hexdigest()

    def _freeze_run_topology(self) -> None:
        """Freeze lane topology for the whole run, or reject an episode that disagrees.

        One ``out_dir`` is one scenario.  A run whose episodes disagree on the lane
        set is already corrupt; failing here costs one episode, failing at P2.4 costs
        the whole collection run.
        """
        lane_sha = hashlib.sha256(
            "\n".join(self._lane_ids).encode("utf-8")
        ).hexdigest()
        if self._run_lane_ids_sha256 is None:
            self._run_lane_ids_sha256 = lane_sha
            self._run_lane_count = len(self._lane_ids)
        elif lane_sha != self._run_lane_ids_sha256:
            raise LoggerStateError(
                f"episode {self._episode_counter} reports a lane set that disagrees "
                f"with the rest of this run (lane_ids_sha256 {lane_sha} vs "
                f"{self._run_lane_ids_sha256}); one out_dir holds one topology"
            )

    def _write_manifest(self) -> None:
        """Rewrite the manifest via temp file + ``os.replace`` so it is never half-written."""
        manifest = {
            "format_version": FORMAT_VERSION,
            "git_hash": self._git_hash,
            "lane_count": self._run_lane_count,
            "lane_ids_sha256": self._run_lane_ids_sha256,
            "run_metadata": self._run_metadata,
            "episodes": self._manifest_episodes,
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

        handle, tmp_name = tempfile.mkstemp(
            dir=str(self._out_dir), prefix=".manifest-", suffix=".json.tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(tmp_name, self.manifest_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
