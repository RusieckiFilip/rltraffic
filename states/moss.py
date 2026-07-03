"""MOSS-specific structured state features."""

from __future__ import annotations

from typing import Any

import numpy as np

from states.base import BaseStateFeatures, StructuredFeature


def _build_drq_norm(
    metrics: Any, ix_info: Any, current_phase: int
) -> np.ndarray:
    """RESCo's ``states.drq_norm`` observation for one intersection.

    The per-lane feature computation lives in
    :meth:`metrics.moss.MossMetrics.drq_norm_observation`; this adapter just
    supplies the intersection and the agent-selected current phase.
    """
    return np.asarray(
        metrics.drq_norm_observation(ix_info, current_phase),
        dtype=np.float32,
    )


def _drq_norm_shape(intersections: list[Any]) -> tuple[int, ...]:
    # drq_norm yields (1, n_lanes, 5) per intersection; report the first
    # intersection's shape as a representative for observation_space.
    n_lanes = len(intersections[0].incoming_lanes)
    return (1, n_lanes, 5)


class MossStateFeatures(BaseStateFeatures):
    """Structured state features MOSS can build."""

    DRQ_NORM = StructuredFeature(
        name="drq_norm",
        build=_build_drq_norm,
        shape=_drq_norm_shape,
        # Per-vehicle waiting time is tracked unconditionally by
        # MossMetrics._run_step_hooks, so no extra metric must be enabled.
        required_metrics=(),
    )
