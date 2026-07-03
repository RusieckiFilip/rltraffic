from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Iterable, SupportsFloat

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from envs.phase_control import (
    DEFAULT_TRANSITION_RECIPE,
    AcyclicPhases,
    PhaseControl,
    PhasePlan,
    PhaseSegment,
)
from states.base import StateRepresentation, StructuredFeature
from rewards import (
    RewardFn,
    required_metrics_for_reward,
    resolve_local_reward_fn,
    resolve_reward_fn,
)

if TYPE_CHECKING:
    from metrics.base import BaseMetrics
    from utils.common_utils import RoadnetInfo


YELLOW_DURATION = 5
YELLOW_DECISION_THRESHOLD = 5
GREEN_MAX_DURATION = 60
GREEN_MIN_DURATION = 5


class BaseTrafficEnv(abc.ABC):
    def __init__(
        self,
        max_steps: int = 3600,
        delta_time: int = 10,
        phase_control_cls: type[PhaseControl] = AcyclicPhases,
        phase_bounds: np.ndarray | None = None,
        metrics: Iterable[str] | None = None,
        state_features: Iterable[str] | None = None,
        obs_norm: dict[str, float] | None = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.delta_time = int(delta_time)
        self._step_count = 0

        # Gymnasium-style env RNG. Seeded via reset(seed=...); used to
        # draw a fresh backend engine seed on every reset.
        self._np_random: np.random.Generator | None = None
        self._np_random_seed: int | None = None
        self._engine_seed: int | None = None

        # Shared traffic-light control state (populated/overridden by subclasses).
        self._roadnet_info: RoadnetInfo | None = None
        self._intersections: list[Any] = []
        self._num_intersections = 0
        self._phase_control_cls = phase_control_cls
        _probe = phase_control_cls()
        self._phase_control_mode = _probe.name
        self._uses_synthetic_transition = _probe.uses_synthetic_transition
        self._phase_controls: list[PhaseControl] = []
        self._phase_bounds: np.ndarray | None = None
        self._initial_phase_bounds = phase_bounds
        self._last_action_applied: np.ndarray = np.array([], dtype=np.bool_)

        # User-selected metric names (None / [] disables the metrics pipeline).
        self._metric_names: list[str] = list(metrics) if metrics else []

        # Ordered list of state features that compose the observation.
        self._state = StateRepresentation(
            list(state_features) if state_features else [],
            supported_structured=self._supported_structured_features(),
            obs_norm=obs_norm,
        )
        # Metric features (and metrics required by any selected structured
        # feature) must be tracked by the metrics pipeline.
        required_metrics = set(self._state.metric_features())
        registry = self._structured_features()
        for name in self._state.features:
            feature = registry.get(name)
            if feature is not None:
                required_metrics.update(feature.required_metrics)
        if required_metrics:
            self._metric_names = sorted(set(self._metric_names) | required_metrics)

        self._metrics: BaseMetrics | None = None
        self._reward_fn: Any = lambda _metrics: 0.0
        # Per-intersection reward shaping (see _configure_rewards()).
        self._local_reward_fn: RewardFn | None = None
        self._global_reward_weight: float = 1.0
        self._local_required_metrics: set[str] = set()

        # To be populated by subclass in _setup_spaces()
        self.observation_space: spaces.Space
        self.action_space: spaces.Space

    # ------------------------------------------------------------------
    # Abstract interface – every simulator backend must implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _init_simulator(self) -> None:
        """Initialise / reset the underlying simulator engine."""

    @abc.abstractmethod
    def _simulate(self, num_steps: int) -> None:
        """Advance the simulator by *num_steps* simulation steps."""

    @abc.abstractmethod
    def _set_phase(
        self,
        ix_idx: int,
        phase: int,
        duration: int | None = None,
    ) -> None:
        """Apply a validated phase change to the backend simulator."""

    def _configure_rewards(
        self,
        global_reward_fn: str | RewardFn | None,
        *,
        local_reward_fn: str | RewardFn | None = None,
        global_reward_weight: float = 1.0,
    ) -> None:
        """Resolve reward functions and merge the metrics they require.

        ``global_reward_fn`` drives the simulation-wide scalar returned
        by :meth:`step` (default ``"queue_length"``).

        When ``local_reward_fn`` is set, every intersection *j* receives

            ``global_reward_weight * global_reward
            + local_reward_fn(local_metrics[j])``

        in ``info["intersections"][j]["reward"]``.
        """
        effective_global = (
            global_reward_fn if global_reward_fn is not None else "queue_length"
        )

        self._reward_fn = resolve_reward_fn(effective_global)
        self._local_reward_fn = (
            resolve_local_reward_fn(local_reward_fn)
            if local_reward_fn is not None
            else None
        )
        self._global_reward_weight = float(global_reward_weight)

        has_callable = callable(effective_global) or callable(local_reward_fn)
        if has_callable and not self._metric_names:
            raise ValueError(
                "Callable reward functions require an explicit "
                "metrics=[...] list so the metrics pipeline can be "
                "enabled."
            )

        required = required_metrics_for_reward(effective_global)
        if local_reward_fn is not None:
            self._local_required_metrics = required_metrics_for_reward(
                local_reward_fn
            )
            required = required | self._local_required_metrics
        self._metric_names = sorted(set(self._metric_names) | required)

    def _init_metrics(self) -> None:
        """(Re-)build the metrics object once the simulator is alive."""
        self._metrics = self._create_metrics()
        if self._metrics is not None:
            self._metrics.reset()
        self._validate_local_reward_metrics()
        self._state.validate_against_metrics(
            type(self._metrics).REGISTERED if self._metrics is not None else None
        )

    def _validate_local_reward_metrics(self) -> None:
        if self._local_reward_fn is None:
            return
        if self._metrics is None:
            raise ValueError(
                "local_reward_fn requires the metrics pipeline, but this "
                "backend did not create a metrics object."
            )
        missing = sorted(
            n
            for n in self._local_required_metrics
            if not self._metrics.has_local(n)
        )
        if missing:
            raise ValueError(
                "local_reward_fn needs per-intersection implementations "
                f"of these metrics, which {type(self._metrics).__name__} "
                f"does not provide: {', '.join(missing)}."
            )

    def _get_reward(self) -> float:
        """Compute and return the scalar reward for the current state."""
        metrics = self._metrics.compute_all() if self._metrics is not None else {}
        return self._reward_fn(metrics)

    def _get_local_rewards(self) -> dict[str, float] | None:
        """Per-intersection rewards, or ``None`` when not configured.

        For intersection *j*:
        ``global_reward_weight * global_reward
        + local_reward_fn(local_metrics[j])``.
        """
        if self._local_reward_fn is None or self._metrics is None:
            return None
        local_metrics = self._metrics.compute_all_local()
        global_part = self._global_reward_weight * float(self._get_reward())
        return {
            ix.id: global_part
            + float(self._local_reward_fn(local_metrics.get(ix.id, {})))
            for ix in self._intersections
        }

    @abc.abstractmethod
    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary diagnostic information."""

    def _create_metrics(self) -> "BaseMetrics | None":
        """Build and return the metrics object for this backend.

        Override in concrete subclasses.  Returning ``None`` (the
        default) disables the metrics pipeline regardless of any names
        the user passed.
        """
        return None

    # ------------------------------------------------------------------
    # State representation backend hooks
    # ------------------------------------------------------------------

    def _structured_features(self) -> dict[str, StructuredFeature]:
        """Registry of structured (multi-dimensional) state features this
        backend can build, keyed by name.  Default: none.  Backends override
        this to add features (e.g. SUMO's ``drq_norm``); the generic
        dispatch below needs no per-name branching."""
        return {}

    def _supported_structured_features(self) -> set[str]:
        return set(self._structured_features())

    def _build_structured_state(self, name: str, ix_idx: int) -> np.ndarray:
        """Build a structured observation for one intersection."""
        feature = self._structured_features()[name]
        return feature.build(
            self._metrics,
            self._intersections[ix_idx],
            self._phase_controls[ix_idx].current_phase(),
        )

    def _structured_feature_shape(self, name: str) -> tuple[int, ...]:
        """Representative observation shape for a structured feature, used to
        declare ``observation_space``."""
        return self._structured_features()[name].shape(self._intersections)

    @property
    def roadnet_info(self) -> "RoadnetInfo":
        """Parsed road network topology."""
        if self._roadnet_info is None:
            raise AttributeError(
                f"{type(self).__name__} does not expose roadnet_info."
            )
        return self._roadnet_info

    @property
    def intersections(self) -> list[Any]:
        """Controllable (non-virtual) intersections."""
        return self._intersections

    def _setup_spaces(self) -> None:
        """Define ``self.observation_space`` and ``self.action_space``."""
        self._setup_phase_controls()
        # Apply constructor bounds only on the first setup; later resets
        # must preserve any post-construction set_phase_durations() call.
        if self._initial_phase_bounds is not None and self._phase_bounds is None:
            self.set_phase_durations(self._initial_phase_bounds)
        elif (
            (
                self._phase_control_mode in {"cyclic", "resco_cyclic"}
                or self._uses_synthetic_transition
            )
            and self._phase_bounds is None
        ):
            self.set_phase_durations(self._build_default_cyclic_phase_bounds())

        structured = self._state.structured_feature()
        if structured is not None:
            shape = self._structured_feature_shape(structured)
        else:
            shape = (self._state.flat_observation_dim(self._intersections),)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )

        phase_counts = [
            self._phase_controls[i].action_count(ix.num_phases)
            for i, ix in enumerate(self._intersections)
        ]

        if self._num_intersections == 1:
            self.action_space = spaces.Discrete(phase_counts[0])
        else:
            self.action_space = spaces.MultiDiscrete(phase_counts)

    def _create_phase_control(self) -> PhaseControl:
        return self._phase_control_cls()

    def _transition_recipe(self) -> tuple[tuple[str, int], ...]:
        """Synthetic transition segments played when switching greens.

        Used by the acyclic-transition controls; backends that can render
        yellow lights (SUMO) override this.
        """
        return DEFAULT_TRANSITION_RECIPE

    def _setup_phase_controls(self) -> None:
        self._phase_controls = []
        for i, ix_info in enumerate(self._intersections):
            control = self._create_phase_control()
            control.initialize(
                ix_info.num_phases,
                phase_roadlink_mapping=getattr(
                    ix_info, "phase_roadlink_mapping", None
                ),
                phase_durations=getattr(ix_info, "phase_durations", None),
                phase_states=getattr(ix_info, "phase_states", None),
                delta_time=self.delta_time,
                transition_recipe=self._transition_recipe(),
            )

            if self._phase_bounds is None:
                control.set_phase_bounds(None)
            else:
                control.set_phase_bounds(self._phase_bounds[i, : ix_info.num_phases, :])
            self._phase_controls.append(control)

        self._last_action_applied = np.zeros(self._num_intersections, dtype=np.bool_)

    def set_phase_durations(self, bounds: np.ndarray) -> None:
        """Configure min/max phase duration bounds.

        ``bounds`` must have shape ``(num_intersections, max_phases, 2)``
        with ``[..., 0]`` = min and ``[..., 1]`` = max.
        """
        arr = np.asarray(bounds, dtype=np.int64)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(
                "Phase bounds must have shape "
                "(num_intersections, max_phases, 2)."
            )
        if arr.shape[0] != self._num_intersections:
            raise ValueError(
                "Phase bounds first dimension must match number of "
                "intersections."
            )
        if np.any(arr < 0):
            raise ValueError("Phase bounds must be non-negative.")
        if np.any(arr[..., 0] > arr[..., 1]):
            raise ValueError("Phase min duration cannot exceed max duration.")

        for i, ix_info in enumerate(self._intersections):
            if arr.shape[1] < ix_info.num_phases:
                raise ValueError(
                    "Phase bounds second dimension is smaller than required "
                    "phase count."
                )
        self._phase_bounds = arr

        for i, ix_info in enumerate(self._intersections):
            if i >= len(self._phase_controls):
                continue
            self._phase_controls[i].set_phase_bounds(
                arr[i, : ix_info.num_phases, :]
            )

    def _build_default_cyclic_phase_bounds(self) -> np.ndarray:
        """Build green/yellow duration bounds from simulator phase metadata."""
        max_phases = max(ix.num_phases for ix in self._intersections)
        bounds = np.zeros(
            (self._num_intersections, max_phases, 2),
            dtype=np.int64,
        )

        for i, ix_info in enumerate(self._intersections):
            if len(ix_info.phase_roadlink_mapping) < ix_info.num_phases:
                raise ValueError(
                    f"Intersection {ix_info.id} is missing phase roadlink metadata."
                )
            if len(ix_info.phase_durations) < ix_info.num_phases:
                raise ValueError(
                    f"Intersection {ix_info.id} is missing phase duration metadata."
                )

            for phase_idx in range(ix_info.num_phases):
                is_green = (
                    bool(ix_info.phase_roadlink_mapping[phase_idx])
                    and float(ix_info.phase_durations[phase_idx])
                    > YELLOW_DECISION_THRESHOLD
                )
                if is_green:
                    bounds[i, phase_idx] = [
                        GREEN_MIN_DURATION,
                        GREEN_MAX_DURATION,
                    ]
                else:
                    bounds[i, phase_idx] = [
                        YELLOW_DURATION,
                        YELLOW_DURATION,
                    ]

        return bounds

    def _available_intersection_actions(self, ix_idx: int) -> list[int]:
        return self._phase_controls[ix_idx].available_actions()

    def _build_local_state(
        self,
        ix_idx: int,
        lane_vehicle_count: dict[str, int],
        lane_waiting: dict[str, int],
        global_metrics: dict[str, float] | None = None,
        local_metrics: dict[str, float] | None = None,
    ) -> np.ndarray:
        return self._state.build_local_state(
            ix_info=self._intersections[ix_idx],
            ix_idx=ix_idx,
            current_phase=self._phase_controls[ix_idx].current_phase(),
            lane_vehicle_count=lane_vehicle_count,
            lane_waiting=lane_waiting,
            global_metrics=global_metrics,
            local_metrics=local_metrics,
            structured_builder=self._build_structured_state,
        )

    def _build_intersections_info(
        self,
        lane_vehicle_count: dict[str, int],
        lane_waiting: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        local_metrics = (
            self._metrics.compute_all_local()
            if self._metrics is not None
            else None
        )
        global_metrics = (
            self._metrics.compute_all() if self._metrics is not None else {}
        )
        local_rewards = self._get_local_rewards()

        intersections_info: dict[str, dict[str, Any]] = {}
        for i, ix_info in enumerate(self._intersections):
            avail_actions = self._available_intersection_actions(i)
            entry: dict[str, Any] = {
                "state": self._build_local_state(
                    i,
                    lane_vehicle_count,
                    lane_waiting,
                    global_metrics=global_metrics,
                    local_metrics=(
                        (local_metrics or {}).get(ix_info.id, {})
                    ),
                ).tolist(),
                "avail_actions": avail_actions,
                "current_phase": self._phase_controls[i].current_phase(),
                "time_in_phase": self._phase_controls[i].time_in_phase(),
                "action_applied": bool(self._last_action_applied[i]),
            }
            if local_metrics is not None:
                entry["metrics"] = local_metrics.get(ix_info.id, {})
            if local_rewards is not None:
                entry["reward"] = local_rewards[ix_info.id]
            intersections_info[ix_info.id] = entry
        return intersections_info

    def _build_phase_plans(self, action: np.ndarray) -> list[PhasePlan]:
        self._last_action_applied.fill(False)
        if action.shape[0] != self._num_intersections:
            raise ValueError(
                "Action length does not match number of controlled "
                "intersections."
            )

        plans: list[PhasePlan] = []
        for i, ix_info in enumerate(self._intersections):
            control = self._phase_controls[i]
            desired_action = int(action[i])
            avail_actions = self._available_intersection_actions(i)
            if desired_action not in avail_actions:
                raise ValueError(
                    f"Action {desired_action} is not available for "
                    f"intersection {ix_info.id}; available: {avail_actions}"
                )

            plan = control.phase_plan(desired_action, self.delta_time)
            self._validate_phase_plan(i, plan)
            plans.append(plan)

        return plans

    def _validate_phase_plan(self, ix_idx: int, plan: PhasePlan) -> None:
        if not plan.segments:
            raise ValueError("Phase plan must contain at least one segment.")

        total_duration = 0
        ix_info = self._intersections[ix_idx]
        for segment in plan.segments:
            if segment.duration <= 0:
                raise ValueError("Phase segment duration must be positive.")
            # Synthetic transition segments (yellow / all_red) are rendered
            # by the backend and need not reference a controllable phase.
            if (
                segment.kind == "phase"
                and not 0 <= int(segment.phase) < ix_info.num_phases
            ):
                raise ValueError(
                    f"Phase out of range for intersection {ix_info.id}: "
                    f"{segment.phase}"
                )
            total_duration += int(segment.duration)
        if total_duration != self.delta_time:
            raise ValueError(
                "Phase plan duration must equal delta_time; "
                f"got {total_duration}, expected {self.delta_time}."
            )

    def _apply_segment(self, ix_idx: int, segment: PhaseSegment) -> None:
        """Render one plan segment on the backend simulator.

        The default implementation only understands real phase indices;
        backends supporting synthetic transition segments override it.
        """
        if segment.kind != "phase":
            raise NotImplementedError(
                f"{type(self).__name__} does not support "
                f"'{segment.kind}' transition segments."
            )
        self._set_phase(ix_idx, segment.phase, segment.duration)

    def _execute_phase_plans(self, plans: list[PhasePlan]) -> None:
        segment_indices = [0 for _ in plans]
        remaining = [int(plan.segments[0].duration) for plan in plans]

        for i, plan in enumerate(plans):
            self._apply_segment(i, plan.segments[0])

        elapsed = 0
        while elapsed < self.delta_time:
            step_duration = min(value for value in remaining if value > 0)
            self._simulate(step_duration)
            elapsed += step_duration

            for i, plan in enumerate(plans):
                if remaining[i] <= 0:
                    continue
                remaining[i] -= step_duration
                if remaining[i] != 0:
                    continue
                segment_indices[i] += 1
                if segment_indices[i] >= len(plan.segments):
                    continue
                next_segment = plan.segments[segment_indices[i]]
                remaining[i] = int(next_segment.duration)
                self._apply_segment(i, next_segment)

        for i, plan in enumerate(plans):
            self._phase_controls[i].apply_plan(plan)
            self._last_action_applied[i] = bool(plan.action_applied)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(
        self, action: int | np.ndarray
    ) -> tuple[SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64).reshape(-1)

        # Snapshot pre-action state for delta-of-delays metric.
        if self._metrics is not None:
            self._metrics.pre_step()

        # 1. Apply the chosen traffic-light phases and advance the simulation.
        plans = self._build_phase_plans(action)
        self._execute_phase_plans(plans)

        self._step_count += 1

        # 2. Refresh per-step metric cache + episode accumulators.
        if self._metrics is not None:
            self._metrics.update()

        # 3. Collect outputs.
        reward = self._get_reward()
        info = self._get_info()
        terminated = False
        truncated = self._step_count >= self.max_steps

        return reward, terminated, truncated, info

    @property
    def np_random(self) -> np.random.Generator:
        """Env RNG, lazily created (Gymnasium convention)."""
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random(None)
        return self._np_random

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._step_count = 0
        # Gymnasium seeding convention: an explicit seed re-seeds the env
        # RNG; otherwise the existing RNG stream continues. Either way a
        # fresh engine seed is drawn for this episode; backends that
        # support per-episode engine seeding (CityFlow, SUMO) apply it in
        # _init_simulator(). Same seed -> identical episode; seed once ->
        # varied but reproducible episode sequence.
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(int(seed))
        self._engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._init_simulator()
        self._setup_spaces()

        # (Re-)build metrics object now that the simulator is alive.
        self._init_metrics()

        # Seed per-step observation caches from the t=0 sim state so the
        # first action of the episode sees real lane state (RESCo seeds
        # its observation via signal.observe() before the first step).
        # Without this, observations like drq_norm read empty caches and
        # the first state is all zeros except the current-phase channel.
        if self._metrics is not None:
            self._metrics.warmup()

        info = self._get_info()
        return info

    @property
    def metrics(self) -> "BaseMetrics | None":
        """The active metrics object, or ``None`` if metrics are disabled."""
        return self._metrics

    def close(self) -> None:
        """Clean up simulator resources.  Override if needed."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds (approximate)."""
        return self._step_count * self.delta_time

    @property
    def control_mode(self) -> str:
        return self._phase_control_mode
