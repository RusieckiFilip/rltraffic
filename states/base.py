"""Composable state (observation) representation for traffic envs.

The per-intersection observation is built from an ordered list of named
**state features**.  A feature is one of:

* a built-in flat block --
  ``lane_vehicle_count`` / ``lane_waiting`` (one value per incoming lane)
  or ``phase_onehot`` (current-phase one-hot);
* any registered **metric** name -- appended as a single scalar, using the
  per-intersection value when the metric has a local implementation and
  otherwise the global value broadcast to every intersection;
* a **structured** feature (e.g. ``drq_norm``) that produces the entire
  observation as a multi-dimensional array; because it cannot be
  concatenated with flat blocks it must be the *only* feature in the list.

This module owns the ordering/concatenation/validation logic
(:class:`StateRepresentation`) and the descriptor + per-backend catalog for
structured features (:class:`StructuredFeature` / :class:`BaseStateFeatures`).
Backend-specific structured features live in the sibling modules
(``states/sumo.py`` etc.), mirroring the ``metrics`` package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_STATE_FEATURES: tuple[str, ...] = (
    "lane_vehicle_count",
    "lane_waiting",
    "phase_onehot",
)
BUILTIN_FLAT_FEATURES = frozenset(DEFAULT_STATE_FEATURES)

# Callback the env passes into :meth:`StateRepresentation.build_local_state`
# to materialise a structured observation: ``(feature_name, ix_idx) -> ndarray``.
StructuredBuilder = Callable[[str, int], np.ndarray]


@dataclass(frozen=True)
class StructuredFeature:
    """A state feature that produces the entire (multi-dimensional)
    observation for one intersection.

    ``build`` and ``shape`` receive only what they need -- the metrics
    object, the intersection, and the current phase -- so providers are
    plain functions independent of the owning env.
    """

    name: str
    build: Callable[[Any, Any, int], np.ndarray]
    """``build(metrics, ix_info, current_phase) -> ndarray`` -- the
    observation for one intersection."""
    shape: Callable[[list[Any]], "tuple[int, ...]"]
    """``shape(intersections) -> tuple`` -- representative shape for
    ``observation_space``."""
    required_metrics: tuple[str, ...] = field(default=())
    """Metrics that must be enabled for this feature to be computable."""


class BaseStateFeatures:
    """Per-backend catalog of structured state features.

    Subclasses declare features as :class:`StructuredFeature` class
    attributes; they are collected into :attr:`STRUCTURED` by
    :meth:`__init_subclass__` (mirroring ``BaseMetrics.REGISTERED``).
    """

    STRUCTURED: dict[str, StructuredFeature] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        registered: dict[str, StructuredFeature] = dict(
            getattr(cls, "STRUCTURED", {})
        )
        for value in vars(cls).values():
            if isinstance(value, StructuredFeature):
                registered[value.name] = value
        cls.STRUCTURED = registered

    @classmethod
    def structured_features(cls) -> dict[str, StructuredFeature]:
        """The structured features this backend can build, keyed by name."""
        return dict(cls.STRUCTURED)


def feature_kind(name: str, structured_names: Iterable[str] = ()) -> str:
    """Classify a feature as ``"builtin"``, ``"structured"`` or ``"metric"``.

    Structured-ness is backend-defined: ``structured_names`` is the set of
    structured features the active backend can build.
    """
    if name in BUILTIN_FLAT_FEATURES:
        return "builtin"
    if name in set(structured_names):
        return "structured"
    return "metric"


class StateRepresentation:
    """Resolves an ordered ``state_features`` list into observations.

    Static structure (non-empty, structured-must-be-sole) is validated at
    construction; metric-existence checks need a live metrics object and are
    deferred to :meth:`validate_against_metrics`.

    ``supported_structured`` is the set of structured-feature names the
    active backend can build; it determines which names classify as
    ``structured`` rather than ``metric``.
    """

    def __init__(
        self,
        features: Iterable[str],
        *,
        supported_structured: Iterable[str] = (),
        obs_norm: dict[str, float] | None = None,
    ) -> None:
        self.features: list[str] = list(features) or list(DEFAULT_STATE_FEATURES)
        self.supported_structured: set[str] = set(supported_structured)
        # Per-feature static divisors applied to the flat observation. A
        # feature absent from the map is left unscaled (divisor 1.0).
        self.obs_norm: dict[str, float] = {
            k: float(v) for k, v in (obs_norm or {}).items()
        }
        self._validate_structure()
        self._validate_obs_norm()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _kind(self, name: str) -> str:
        return feature_kind(name, self.supported_structured)

    def metric_features(self) -> list[str]:
        return [f for f in self.features if self._kind(f) == "metric"]

    def structured_feature(self) -> str | None:
        structured = [f for f in self.features if self._kind(f) == "structured"]
        return structured[0] if structured else None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:
        if not self.features:
            raise ValueError("state_features must contain at least one entry.")
        structured = [f for f in self.features if self._kind(f) == "structured"]
        if structured and len(self.features) > 1:
            raise ValueError(
                f"Structured state feature '{structured[0]}' produces the "
                "entire observation and cannot be combined with other "
                f"features; got {self.features}."
            )

    def _validate_obs_norm(self) -> None:
        for name, scale in self.obs_norm.items():
            if name not in self.features:
                raise ValueError(
                    f"obs_norm references '{name}', which is not in "
                    f"state_features {self.features}."
                )
            if self._kind(name) == "structured":
                raise ValueError(
                    f"obs_norm cannot scale the structured feature '{name}'; "
                    "it already produces a fully-formed observation."
                )
            if not math.isfinite(scale) or scale == 0.0:
                raise ValueError(
                    f"obs_norm['{name}'] must be a finite non-zero number; "
                    f"got {scale}."
                )

    def validate_against_metrics(self, registered: dict[str, Any] | None) -> None:
        """Check that every metric feature exists and is computable.

        ``registered`` is the metrics class's ``REGISTERED`` dict, or
        ``None`` when the metrics pipeline is disabled.
        """
        metric_features = self.metric_features()
        if (metric_features or self.structured_feature()) and registered is None:
            raise ValueError(
                "state_features reference metrics or structured features but "
                "this backend did not create a metrics object; pass an "
                "explicit metrics=[...] list."
            )
        for name in metric_features:
            spec = registered.get(name) if registered else None
            if spec is None:
                available = sorted(registered) if registered else []
                raise ValueError(
                    f"Unknown state feature '{name}'. Built-ins: "
                    f"{sorted(BUILTIN_FLAT_FEATURES)}; structured: "
                    f"{sorted(self.supported_structured)}; metrics: {available}."
                )
            if spec.compute is None and spec.compute_local is None:
                raise ValueError(
                    f"Metric state feature '{name}' has no computable "
                    "implementation."
                )

    # ------------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_feature_width(name: str, ix_info: Any) -> int:
        if name in ("lane_vehicle_count", "lane_waiting"):
            return len(ix_info.incoming_lanes)
        if name == "phase_onehot":
            return int(ix_info.num_phases)
        # metric feature -> one scalar per intersection
        return 1

    def flat_observation_dim(self, intersections: list[Any]) -> int:
        """Total length of the concatenated flat observation across all
        intersections (only valid when there is no structured feature)."""
        return sum(
            self._flat_feature_width(name, ix)
            for ix in intersections
            for name in self.features
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_local_state(
        self,
        *,
        ix_info: Any,
        ix_idx: int,
        current_phase: int,
        lane_vehicle_count: dict[str, int],
        lane_waiting: dict[str, int],
        global_metrics: dict[str, float] | None = None,
        local_metrics: dict[str, float] | None = None,
        structured_builder: StructuredBuilder | None = None,
    ) -> np.ndarray:
        """Assemble one intersection's state vector (or structured array)."""
        structured = self.structured_feature()
        if structured is not None:
            if structured_builder is None:
                raise ValueError(
                    f"structured feature '{structured}' requires a "
                    "structured_builder."
                )
            return structured_builder(structured, ix_idx)

        global_metrics = global_metrics or {}
        local_metrics = local_metrics or {}
        parts: list[np.ndarray] = []
        for name in self.features:
            if name == "lane_vehicle_count":
                part = np.array(
                    [lane_vehicle_count[lid] for lid in ix_info.incoming_lanes],
                    dtype=np.float32,
                )
            elif name == "lane_waiting":
                part = np.array(
                    [lane_waiting[lid] for lid in ix_info.incoming_lanes],
                    dtype=np.float32,
                )
            elif name == "phase_onehot":
                part = np.zeros(int(ix_info.num_phases), dtype=np.float32)
                part[int(current_phase)] = 1.0
            else:
                # Metric feature: prefer the per-intersection value, fall
                # back to the (broadcast) global scalar.
                if name in local_metrics:
                    value = local_metrics[name]
                else:
                    value = global_metrics.get(name, 0.0)
                part = np.array([value], dtype=np.float32)

            scale = self.obs_norm.get(name)
            if scale is not None:
                part = part / np.float32(scale)
            parts.append(part)
        return np.concatenate(parts)
