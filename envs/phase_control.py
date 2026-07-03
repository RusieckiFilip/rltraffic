from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


PhaseControlName = Literal[
    "acyclic", "acyclic_bounded", "cyclic", "resco_cyclic"
]

SegmentKind = Literal["phase", "yellow", "all_red"]


RESCO_DEFAULT_MIN_GREEN = 5

# Phases whose configured file duration is at most this many seconds are
# treated as transition (yellow / clearance) phases, not selectable greens.
TRANSITION_PHASE_MAX_DURATION = 5

# Synthetic transition played between greens when the backend cannot show
# yellow lights (CityFlow, MOSS): a single all-red clearance.
DEFAULT_TRANSITION_RECIPE: tuple[tuple[str, int], ...] = (("all_red", 5),)


@dataclass(frozen=True)
class PhaseSegment:
    phase: int
    duration: int
    kind: SegmentKind = "phase"


@dataclass(frozen=True)
class PhasePlan:
    segments: tuple[PhaseSegment, ...]
    action_applied: bool = False

    @property
    def final_phase(self) -> int:
        if not self.segments:
            raise ValueError("Phase plan must contain at least one segment.")
        return self.segments[-1].phase

    @property
    def final_duration(self) -> int:
        if not self.segments:
            raise ValueError("Phase plan must contain at least one segment.")
        return self.segments[-1].duration


class PhaseControl(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> PhaseControlName:
        pass

    @property
    def uses_synthetic_transition(self) -> bool:
        """Whether switching greens plays a synthetic clearance segment.

        Backends use this to decide whether to inject an all-red phase
        (CityFlow / MOSS) or render yellow lights (SUMO). Defaults to
        ``False``; the acyclic-transition controls override it.
        """
        return False

    @abc.abstractmethod
    def action_count(self, num_phases: int) -> int:
        pass

    @abc.abstractmethod
    def initialize(
        self,
        num_phases: int,
        *,
        phase_roadlink_mapping: Sequence[Sequence[int]] | None = None,
        phase_durations: Sequence[float] | None = None,
        phase_states: Sequence[str] | None = None,
        delta_time: int | None = None,
        transition_recipe: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def set_phase_bounds(self, bounds: np.ndarray | None) -> None:
        pass

    @abc.abstractmethod
    def reset_state(self) -> None:
        pass

    @abc.abstractmethod
    def current_phase(self) -> int:
        pass

    @abc.abstractmethod
    def time_in_phase(self) -> int:
        pass

    @abc.abstractmethod
    def tick(self, delta_seconds: int) -> None:
        pass

    @abc.abstractmethod
    def apply_phase(self, phase: int) -> None:
        pass

    @abc.abstractmethod
    def available_actions(self) -> list[int]:
        pass

    @abc.abstractmethod
    def target_phase(self, action: int) -> int:
        pass

    def phase_plan(self, action: int, delta_seconds: int) -> PhasePlan:
        target = self.target_phase(action)
        return PhasePlan(
            segments=(PhaseSegment(target, int(delta_seconds)),),
            action_applied=(target != self.current_phase()),
        )

    def apply_plan(self, plan: PhasePlan) -> None:
        if plan.action_applied:
            self.apply_phase(plan.final_phase)
            self.tick(plan.final_duration)
        else:
            self.tick(plan.final_duration)


class _StatefulPhaseControl(PhaseControl):
    def __init__(self) -> None:
        self._num_phases = 0
        self._current_phase = 0
        self._time_in_phase = 0
        self._phase_bounds: np.ndarray | None = None

    def initialize(
        self,
        num_phases: int,
        *,
        phase_roadlink_mapping: Sequence[Sequence[int]] | None = None,
        phase_durations: Sequence[float] | None = None,
        phase_states: Sequence[str] | None = None,
        delta_time: int | None = None,
        transition_recipe: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        _ = phase_roadlink_mapping, phase_durations, phase_states, delta_time
        _ = transition_recipe
        if num_phases <= 0:
            raise ValueError("num_phases must be positive.")
        self._num_phases = int(num_phases)
        self.reset_state()

    def set_phase_bounds(self, bounds: np.ndarray | None) -> None:
        if bounds is None:
            self._phase_bounds = None
            return

        arr = np.asarray(bounds, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Phase bounds must have shape (num_phases, 2).")
        if arr.shape[0] < self._num_phases:
            raise ValueError("Phase bounds must include all controlled phases.")
        self._phase_bounds = arr

    def reset_state(self) -> None:
        self._current_phase = 0
        self._time_in_phase = 0

    def current_phase(self) -> int:
        return int(self._current_phase)

    def time_in_phase(self) -> int:
        return int(self._time_in_phase)

    def tick(self, delta_seconds: int) -> None:
        self._time_in_phase += int(delta_seconds)

    def apply_phase(self, phase: int) -> None:
        if not 0 <= int(phase) < self._num_phases:
            raise ValueError(f"Phase out of range: {phase}")
        self._current_phase = int(phase)
        self._time_in_phase = 0

    def _min_duration_reached(self) -> bool:
        if self._phase_bounds is None:
            return True
        min_duration = int(self._phase_bounds[self._current_phase, 0])
        return bool(self._time_in_phase >= min_duration)

    def _max_duration_reached(self) -> bool:
        if self._phase_bounds is None:
            return False
        max_duration = int(self._phase_bounds[self._current_phase, 1])
        return bool(self._time_in_phase >= max_duration)


class CyclicPhases(_StatefulPhaseControl):
    @property
    def name(self) -> PhaseControlName:
        return "cyclic"

    def action_count(self, num_phases: int) -> int:
        _ = num_phases
        return 2

    def available_actions(self) -> list[int]:
        if self._max_duration_reached():
            return [1]
        if not self._min_duration_reached():
            return [0]
        return [0, 1]

    def target_phase(self, action: int) -> int:
        if action == 0:
            return self._current_phase
        if action == 1:
            return (self._current_phase + 1) % self._num_phases
        raise ValueError(f"Unsupported cyclic action: {action}")


class RescoCyclicPhases(_StatefulPhaseControl):
    @property
    def name(self) -> PhaseControlName:
        return "resco_cyclic"

    def __init__(self) -> None:
        super().__init__()
        self._delta_time = 0
        self._green_phases: list[int] = []
        self._next_green_by_phase: dict[int, int] = {}
        self._transition_by_phase: dict[int, tuple[PhaseSegment, ...]] = {}

    def initialize(
        self,
        num_phases: int,
        *,
        phase_roadlink_mapping: Sequence[Sequence[int]] | None = None,
        phase_durations: Sequence[float] | None = None,
        phase_states: Sequence[str] | None = None,
        delta_time: int | None = None,
        transition_recipe: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        _ = transition_recipe
        if phase_roadlink_mapping is None:
            raise ValueError("RescoCyclicPhases requires phase roadlink metadata.")
        if phase_durations is None:
            raise ValueError("RescoCyclicPhases requires phase duration metadata.")
        if delta_time is None:
            raise ValueError("RescoCyclicPhases requires delta_time.")

        super().initialize(num_phases)
        self._delta_time = int(delta_time)
        self._configure_transitions(
            phase_roadlink_mapping,
            phase_durations,
            phase_states,
        )
        self.reset_state()

    def reset_state(self) -> None:
        self._current_phase = self._green_phases[0] if self._green_phases else 0
        self._time_in_phase = 0

    def action_count(self, num_phases: int) -> int:
        _ = num_phases
        return 2

    def available_actions(self) -> list[int]:
        return [0, 1]

    def target_phase(self, action: int) -> int:
        if action == 0:
            return self._current_phase
        if action == 1:
            return self._next_green_by_phase[self._current_phase]
        raise ValueError(f"Unsupported resco cyclic action: {action}")

    def phase_plan(self, action: int, delta_seconds: int) -> PhasePlan:
        delta = int(delta_seconds)
        if delta != self._delta_time:
            raise ValueError(
                "RescoCyclicPhases requires a fixed delta_time matching "
                f"initialization; got {delta}, expected {self._delta_time}."
            )
        if action == 0:
            return self._keep_plan(delta)
        if action != 1:
            raise ValueError(f"Unsupported resco cyclic action: {action}")

        transition = self._transition_by_phase[self._current_phase]
        transition_duration = sum(segment.duration for segment in transition)
        if self._time_in_phase < self._min_green_duration() + transition_duration:
            return self._keep_plan(delta)

        green_duration = delta - transition_duration
        target_green = self._next_green_by_phase[self._current_phase]
        return PhasePlan(
            segments=(
                *transition,
                PhaseSegment(target_green, green_duration),
            ),
            action_applied=True,
        )

    def apply_plan(self, plan: PhasePlan) -> None:
        if not plan.action_applied:
            self.tick(plan.final_duration)
            return
        final_phase = plan.final_phase
        if not 0 <= final_phase < self._num_phases:
            raise ValueError(f"Phase out of range: {final_phase}")
        self._current_phase = int(final_phase)
        self._time_in_phase = int(plan.final_duration)

    def _keep_plan(self, delta: int) -> PhasePlan:
        return PhasePlan(
            segments=(PhaseSegment(self._current_phase, delta),),
            action_applied=False,
        )

    def _min_green_duration(self) -> int:
        if self._phase_bounds is None:
            return RESCO_DEFAULT_MIN_GREEN
        return int(self._phase_bounds[self._current_phase, 0])

    def _configure_transitions(
        self,
        phase_roadlink_mapping: Sequence[Sequence[int]],
        phase_durations: Sequence[float],
        phase_states: Sequence[str] | None,
    ) -> None:
        if len(phase_roadlink_mapping) < self._num_phases:
            raise ValueError(
                "RescoCyclicPhases requires roadlink metadata for every phase."
            )
        if len(phase_durations) < self._num_phases:
            raise ValueError(
                "RescoCyclicPhases requires duration metadata for every phase."
            )

        self._green_phases = [
            phase_idx
            for phase_idx in range(self._num_phases)
            if self._is_green_phase(
                phase_idx,
                phase_roadlink_mapping,
                phase_durations,
                phase_states,
            )
        ]
        if len(self._green_phases) < 2:
            raise ValueError(
                "RescoCyclicPhases requires at least two green phases."
            )

        self._next_green_by_phase = {}
        self._transition_by_phase = {}
        for idx, green_phase in enumerate(self._green_phases):
            next_green = self._green_phases[(idx + 1) % len(self._green_phases)]
            transition_indices = self._phase_range(green_phase, next_green)
            if not transition_indices:
                raise ValueError(
                    "RescoCyclicPhases requires transition phases between "
                    f"green phase {green_phase} and {next_green}."
                )

            transition = tuple(
                PhaseSegment(phase, int(phase_durations[phase]))
                for phase in transition_indices
            )
            if any(segment.duration <= 0 for segment in transition):
                raise ValueError(
                    "RescoCyclicPhases transition durations must be positive."
                )
            transition_duration = sum(segment.duration for segment in transition)
            if transition_duration >= self._delta_time:
                raise ValueError(
                    "RescoCyclicPhases transition duration must be shorter "
                    f"than delta_time; got {transition_duration} >= "
                    f"{self._delta_time}."
                )

            self._next_green_by_phase[green_phase] = next_green
            self._transition_by_phase[green_phase] = transition

    def _phase_range(self, start_phase: int, end_phase: int) -> list[int]:
        if start_phase < end_phase:
            return list(range(start_phase + 1, end_phase))
        return list(range(start_phase + 1, self._num_phases)) + list(
            range(0, end_phase)
        )

    def _is_green_phase(
        self,
        phase_idx: int,
        phase_roadlink_mapping: Sequence[Sequence[int]],
        phase_durations: Sequence[float],
        phase_states: Sequence[str] | None,
    ) -> bool:
        if not bool(phase_roadlink_mapping[phase_idx]):
            return False
        if float(phase_durations[phase_idx]) <= RESCO_DEFAULT_MIN_GREEN:
            return False
        if phase_states is None or len(phase_states) <= phase_idx:
            return True
        state = str(phase_states[phase_idx]).lower()
        return "y" not in state


class _AcyclicTransitionBase(_StatefulPhaseControl):
    """Acyclic green selection with a forced clearance after every green.

    Actions pick among *green* phases only -- phases whose configured file
    duration exceeds ``TRANSITION_PHASE_MAX_DURATION``; the file's own
    short transition phases are never selectable.  Whenever the agent
    switches to a different green, the synthetic transition described by
    ``transition_recipe`` is played first (e.g. yellow then all-red for
    SUMO, a single all-red for CityFlow / MOSS), followed by the target
    green for the remainder of ``delta_time``.

    Subclasses differ only in whether ``available_actions`` honours the
    configured min/max phase-time bounds.
    """

    @property
    def uses_synthetic_transition(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self._delta_time = 0
        self._green_phases: list[int] = []
        self._transition_recipe: tuple[tuple[SegmentKind, int], ...] = ()

    def initialize(
        self,
        num_phases: int,
        *,
        phase_roadlink_mapping: Sequence[Sequence[int]] | None = None,
        phase_durations: Sequence[float] | None = None,
        phase_states: Sequence[str] | None = None,
        delta_time: int | None = None,
        transition_recipe: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        _ = phase_roadlink_mapping, phase_states
        if phase_durations is None:
            raise ValueError(
                "Acyclic transition control requires phase duration metadata."
            )
        if delta_time is None:
            raise ValueError("Acyclic transition control requires delta_time.")

        super().initialize(num_phases)
        if len(phase_durations) < self._num_phases:
            raise ValueError(
                "Acyclic transition control requires duration metadata for "
                "every phase."
            )
        self._delta_time = int(delta_time)
        self._transition_recipe = self._validated_recipe(
            DEFAULT_TRANSITION_RECIPE if transition_recipe is None
            else transition_recipe
        )

        self._green_phases = [
            phase_idx
            for phase_idx in range(self._num_phases)
            if float(phase_durations[phase_idx]) > TRANSITION_PHASE_MAX_DURATION
        ]
        if not self._green_phases:
            raise ValueError(
                "Acyclic transition control requires at least one green phase "
                f"(file duration > {TRANSITION_PHASE_MAX_DURATION})."
            )
        transition_duration = self._transition_duration()
        if len(self._green_phases) > 1 and transition_duration >= self._delta_time:
            raise ValueError(
                "Acyclic transition control transition duration must be "
                f"shorter than delta_time; got {transition_duration} >= "
                f"{self._delta_time}."
            )
        self.reset_state()

    @staticmethod
    def _validated_recipe(
        recipe: Sequence[tuple[str, int]],
    ) -> tuple[tuple[SegmentKind, int], ...]:
        if not recipe:
            raise ValueError(
                "Acyclic transition control requires a non-empty transition "
                "recipe."
            )
        validated: list[tuple[SegmentKind, int]] = []
        for kind, duration in recipe:
            if kind not in ("yellow", "all_red"):
                raise ValueError(
                    f"Unsupported transition segment kind: {kind}"
                )
            if int(duration) <= 0:
                raise ValueError(
                    "Transition segment durations must be positive."
                )
            validated.append((kind, int(duration)))
        return tuple(validated)

    def reset_state(self) -> None:
        self._current_phase = self._green_phases[0] if self._green_phases else 0
        self._time_in_phase = 0

    @property
    def green_phases(self) -> list[int]:
        return list(self._green_phases)

    def action_count(self, num_phases: int) -> int:
        _ = num_phases
        return len(self._green_phases)

    def target_phase(self, action: int) -> int:
        if 0 <= action < len(self._green_phases):
            return self._green_phases[action]
        raise ValueError(
            f"Unsupported acyclic transition action: {action}"
        )

    def phase_plan(self, action: int, delta_seconds: int) -> PhasePlan:
        delta = int(delta_seconds)
        if delta != self._delta_time:
            raise ValueError(
                "Acyclic transition control requires a fixed delta_time "
                f"matching initialization; got {delta}, expected "
                f"{self._delta_time}."
            )
        target = self.target_phase(action)
        if target == self._current_phase:
            return PhasePlan(
                segments=(PhaseSegment(self._current_phase, delta),),
                action_applied=False,
            )

        # Transition segments carry the index of the green being left so
        # backends can derive e.g. the matching yellow state from it.
        transition = tuple(
            PhaseSegment(self._current_phase, duration, kind)
            for kind, duration in self._transition_recipe
        )
        green_duration = delta - self._transition_duration()
        return PhasePlan(
            segments=(*transition, PhaseSegment(target, green_duration)),
            action_applied=True,
        )

    def apply_plan(self, plan: PhasePlan) -> None:
        if not plan.action_applied:
            self.tick(plan.final_duration)
            return
        final_phase = int(plan.final_phase)
        if final_phase not in self._green_phases:
            raise ValueError(
                f"Final phase must be a green phase: {final_phase}"
            )
        self._current_phase = final_phase
        self._time_in_phase = int(plan.final_duration)

    def _transition_duration(self) -> int:
        return sum(duration for _, duration in self._transition_recipe)


class AcyclicPhases(_AcyclicTransitionBase):
    """Acyclic transition control that ignores min/max phase-time bounds.

    The forced clearance after every switch already enforces safe timing,
    so any green is selectable at any tick.
    """

    @property
    def name(self) -> PhaseControlName:
        return "acyclic"

    def available_actions(self) -> list[int]:
        return list(range(len(self._green_phases)))


class AcyclicBoundedPhases(_AcyclicTransitionBase):
    """Acyclic transition control that honours min/max phase-time bounds.

    Behaves like :class:`AcyclicPhases` but the configured min/max green
    durations gate which actions are available: greens are held for at
    least their min duration and forced to switch once max is reached.
    """

    @property
    def name(self) -> PhaseControlName:
        return "acyclic_bounded"

    def available_actions(self) -> list[int]:
        current_action = self._green_phases.index(self._current_phase)
        all_actions = list(range(len(self._green_phases)))
        if self._max_duration_reached():
            available = [a for a in all_actions if a != current_action]
            return available if available else [current_action]
        if not self._min_duration_reached():
            return [current_action]
        return all_actions
