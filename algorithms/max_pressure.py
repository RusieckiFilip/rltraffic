from __future__ import annotations

from typing import Any

import numpy as np

from envs.phase_control import TRANSITION_PHASE_MAX_DURATION
from utils.common_utils import IntersectionInfo


class MaxPressureAgent:
    """Greedy MaxPressure controller for traffic signal control.

    Parameters
    ----------
    env : BaseTrafficEnv
        A traffic environment whose ``intersections`` property returns a
        list of :class:`IntersectionInfo` objects.  ``env.control_mode``
        decides the action vocabulary ``act()`` uses: green-phase action
        indices under the acyclic modes, binary keep/switch actions under
        ``"cyclic"`` / ``"resco_cyclic"``.
    """

    def __init__(self, env: Any) -> None:
        self._intersections: list[IntersectionInfo] = env.intersections
        self._control_mode: str = str(getattr(env, "control_mode", "acyclic"))

    def act(self, info: dict[str, Any]) -> np.ndarray:
        lane_counts: dict[str, int] = info["lane_vehicle_count"]
        intersections_info = info.get("intersections", {}) or {}
        actions: list[int] = []

        for ix in self._intersections:
            ix_payload = intersections_info.get(ix.id, {}) or {}
            if self._control_mode in {"cyclic", "resco_cyclic"}:
                actions.append(
                    self._cyclic_action(ix, ix_payload, lane_counts)
                )
            else:
                actions.append(
                    self._acyclic_action(ix, ix_payload, lane_counts)
                )

        return np.array(actions, dtype=np.int64)

    # TODO: under cyclic control modes the max-pressure phase choice is
    # mapped onto binary keep/switch actions, which only approximates true
    # max-pressure control; revisit if exact phase selection is needed.
    def _cyclic_action(
        self,
        ix: IntersectionInfo,
        ix_payload: dict[str, Any],
        lane_counts: dict[str, int],
    ) -> int:
        candidates = [
            p for p, links in enumerate(ix.phase_roadlink_mapping) if links
        ] or list(range(len(ix.phase_roadlink_mapping)))
        best_phase = max(
            candidates,
            key=lambda p: self._phase_pressure(
                ix, ix.phase_roadlink_mapping[p], lane_counts
            ),
        )

        curr_phase = int(ix_payload.get("current_phase", 0))
        desired_action = 0 if best_phase == curr_phase else 1
        raw_available = ix_payload.get("avail_actions", [0, 1])
        available = [int(action) for action in raw_available]
        if not available:
            raise ValueError(
                f"Intersection {ix.id} has no available cyclic actions."
            )
        if desired_action not in available:
            desired_action = available[0]
        return desired_action

    def _acyclic_action(
        self,
        ix: IntersectionInfo,
        ix_payload: dict[str, Any],
        lane_counts: dict[str, int],
    ) -> int:
        """Highest-pressure *green action* for the acyclic controls.

        Acyclic actions index the control's green-phase list, not the raw
        file phases, so the winning green is translated back to its action
        index.  ``avail_actions`` (when provided) restricts the candidates,
        which honours ``acyclic_bounded`` min/max phase-time gating.
        """
        greens = self._green_phases(ix)
        raw_available = ix_payload.get("avail_actions") or []
        candidates = [
            int(a) for a in raw_available if 0 <= int(a) < len(greens)
        ] or list(range(len(greens)))
        return max(
            candidates,
            key=lambda a: self._phase_pressure(
                ix, ix.phase_roadlink_mapping[greens[a]], lane_counts
            ),
        )

    @staticmethod
    def _green_phases(ix: IntersectionInfo) -> list[int]:
        """Selectable green phases, mirroring the acyclic controls' rule
        (file duration above the transition threshold).

        Falls back to phases with active roadlinks when duration metadata
        is unavailable (e.g. lightweight stub envs in tests).
        """
        durations = list(getattr(ix, "phase_durations", None) or [])
        if len(durations) >= ix.num_phases:
            greens = [
                p
                for p in range(ix.num_phases)
                if float(durations[p]) > TRANSITION_PHASE_MAX_DURATION
            ]
            if greens:
                return greens
        greens = [
            p for p, links in enumerate(ix.phase_roadlink_mapping) if links
        ]
        return greens or list(range(ix.num_phases))

    @staticmethod
    def _phase_pressure(
        ix: IntersectionInfo,
        active_roadlinks: list[int],
        lane_counts: dict[str, int],
    ) -> float:
        """Compute total pressure for a set of active roadlinks.

        For each active roadlink the pressure is:

            sum(incoming lane counts) − sum(outgoing lane counts)

        The phase pressure is the sum over all its roadlinks.
        """
        total = 0.0
        for rl_idx in active_roadlinks:
            in_lanes, out_lanes = ix.roadlink_lanes[rl_idx]
            incoming = sum(lane_counts[lid] for lid in in_lanes)
            outgoing = sum(lane_counts[lid] for lid in out_lanes)
            total += incoming - outgoing
        return total
