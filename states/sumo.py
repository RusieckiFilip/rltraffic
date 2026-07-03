"""SUMO-specific structured state features."""

from __future__ import annotations

from typing import Any

import numpy as np

from metrics.sumo import METRIC_STICKY_WAIT
from states.base import BaseStateFeatures, StructuredFeature


def _build_drq_norm(
    metrics: Any, ix_info: Any, current_phase: int
) -> np.ndarray:
    """RESCo's ``states.drq_norm`` observation for one intersection.

    The per-lane feature computation lives in
    :meth:`metrics.sumo.SumoMetrics.drq_norm_observation`; this adapter just
    supplies the intersection and the agent-selected current phase.
    """
    return np.asarray(
        metrics.drq_norm_observation(ix_info, current_phase),
        dtype=np.float32,
    )


def _drq_norm_shape(intersections: list[Any]) -> tuple[int, ...]:
    # drq_norm yields (1, n_lanes, 5) per intersection; report the first
    # intersection's shape as a representative for observation_space.
    # Per-intersection learners build their networks from the actual state
    # arrays in info, so a representative shape is sufficient.
    n_lanes = len(intersections[0].incoming_lanes)
    return (1, n_lanes, 5)


class SumoStateFeatures(BaseStateFeatures):
    """Structured state features SUMO can build."""

    DRQ_NORM = StructuredFeature(
        name="drq_norm",
        build=_build_drq_norm,
        shape=_drq_norm_shape,
        # drq_norm is derived from the sticky-wait tracker, so it must be
        # kept updated every step (enabled generically by the base env).
        required_metrics=(METRIC_STICKY_WAIT,),
    )
