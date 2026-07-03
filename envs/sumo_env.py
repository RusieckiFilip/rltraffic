

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np


from envs.base_traffic_env import BaseTrafficEnv
from envs.phase_control import AcyclicPhases, PhaseControl, PhaseSegment
from states.base import StructuredFeature
from states.sumo import SumoStateFeatures

from utils.sumo_utils import (
    extract_sumo_roadnet,
    validate_sumo_inputs_exist,
)
from utils.common_utils import IntersectionInfo, RoadnetInfo
from metrics import SumoMetrics
from metrics.base import BaseMetrics


# TraCI default (60) is often too few while SUMO loads larger networks.
_TRACI_START_NUM_RETRIES = 120

# Synthetic transition between greens used by the acyclic-transition
# controls: the departing green turns yellow, then everything goes red.
TRANSITION_YELLOW_DURATION = 3
TRANSITION_ALL_RED_DURATION = 2


class SumoEnv(BaseTrafficEnv):
    """Gymnasium environment wrapping the SUMO traffic simulator.

    Parameters
    ----------
    sumocfg_path : str | Path
        Path to the SUMO ``.sumocfg`` configuration file.
    max_steps : int
        Maximum number of *agent* decision steps per episode.
    delta_time : int
        Simulation seconds between consecutive agent actions.
    global_reward_fn : str | callable
        Name of a registered reward function (see
        :mod:`rewards`) or a custom callable that accepts
        a metrics dict and returns a float.  Drives the scalar reward
        returned by ``step()`` (default ``"queue_length"``).
    local_reward_fn : str | callable
        Optional per-intersection reward.  When set, every intersection
        ``j`` receives ``global_reward_weight * global_reward +
        local_reward_fn(local_metrics[j])`` in
        ``info["intersections"][j]["reward"]``.
    global_reward_weight : float
        Weight of the global reward inside the per-intersection reward
        (default ``1.0``).
    gui : bool
        If True, run ``sumo-gui`` instead of headless ``sumo``.
    phase_control_cls : type[PhaseControl]
        Strategy class used to translate agent actions into phases.
    phase_bounds : np.ndarray | None
        Optional ``(num_intersections, max_phases, 2)`` min/max phase
        duration bounds.
    metrics : Iterable[str] | None
        Names of metrics to enable.
    state_features : Iterable[str] | None
        Ordered list of features composing each intersection's observation.
        Defaults to ``["lane_vehicle_count", "lane_waiting", "phase_onehot"]``.
        Any registered metric name may be added (appended as a scalar:
        per-intersection if the metric has a local implementation, else the
        global value broadcast to every intersection).  The structured
        feature ``"drq_norm"`` yields RESCO's ``(1, n_lanes, 5)`` observation
        and, because it defines the whole observation, must be the only entry.
    libsumo : bool
        If True, use the in-process ``libsumo`` API instead of TraCI.

    Seeding follows the Gymnasium convention: ``reset(seed=X)`` re-seeds
    the env RNG, ``reset()`` continues it; every reset draws a fresh
    engine seed from that RNG and passes it to SUMO via ``--seed``.
    Seed once for a reproducible sequence of varied episodes; pass the
    same seed every reset for identical episodes.
    """

    def __init__(
        self,
        sumocfg_path: str | Path,
        max_steps: int = 3600,
        delta_time: int = 10,
        global_reward_fn: str | Any | None = None,
        gui: bool = False,
        phase_control_cls: type[PhaseControl] = AcyclicPhases,
        phase_bounds: np.ndarray | None = None,
        metrics: Iterable[str] | None = None,
        libsumo: bool = False,
        local_reward_fn: str | Any | None = None,
        global_reward_weight: float = 1.0,
        state_features: Iterable[str] | None = None,
        obs_norm: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            delta_time=delta_time,
            phase_control_cls=phase_control_cls,
            phase_bounds=phase_bounds,
            metrics=metrics,
            state_features=state_features,
            obs_norm=obs_norm,
        )
        self._config_path = str(Path(sumocfg_path).resolve())
        self._configure_rewards(
            global_reward_fn,
            local_reward_fn=local_reward_fn,
            global_reward_weight=global_reward_weight,
        )
        # _get_info() reports average_travel_time unconditionally, and the
        # completed-trip bookkeeping behind it is opt-in for SUMO; request it
        # always so the reported value is real (parity with CityFlow/MOSS,
        # which track completions unconditionally).
        self._metric_names = sorted(
            set(self._metric_names) | {"average_travel_time"}
        )
        self._use_gui = gui
        self._use_libsumo = bool(libsumo)
        if self._use_gui and self._use_libsumo:
            raise ValueError("SUMO GUI is not supported with libsumo.")
        self._sumo = self._import_sumo_api()

        if not os.path.exists(self._config_path):
            raise FileNotFoundError(f"SUMO config missing: {self._config_path}")
        validate_sumo_inputs_exist(self._config_path)

        # Placeholders populated by _init_simulator() querying SUMO.
        self._roadnet_info: RoadnetInfo | None = None
        self._intersections: list[IntersectionInfo] = []
        self._num_intersections = 0
        self._libsumo_started = False

        self._phase_bounds: np.ndarray | None = None   # Shape (N_intersections, max_phases, 2)
        self._last_action_applied: np.ndarray = np.array([])
        self._tmp_config_dir: str | None = None

        # Boot SUMO, extract the map topology dynamically, and reset
        self._init_simulator()

        # A failure past this point (e.g. an unknown metric name or a phase
        # control rejecting this scenario) must not orphan the SUMO process
        # / TraCI connection that _init_simulator() just started.
        try:
            self._setup_spaces()

            # Build the metrics object now that the engine is alive (will be
            # rebuilt in ``reset()`` for fresh episodes).
            self._init_metrics()
        except Exception:
            self.close()
            raise

    def _import_sumo_api(self) -> Any:
        module_name = "libsumo" if self._use_libsumo else "traci"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if self._use_libsumo:
                raise ModuleNotFoundError(
                    "libsumo was requested, but the Python module is not "
                    "installed for this interpreter."
                ) from exc
            raise

    def _init_simulator(self) -> None:
        sumo_binary = "sumo-gui" if self._use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self._config_path, "--no-step-log", "true", "--no-warnings", "true"]
        # SUMO's accumulated waiting time only covers the last
        # --waiting-time-memory seconds (default 100); raise it past any
        # realistic episode so waiting-time metrics are comparable with
        # CityFlow/MOSS, whose accumulators are unbounded.
        sumo_cmd += ["--waiting-time-memory", "100000"]
        if self._engine_seed is not None:
            sumo_cmd += ["--seed", str(int(self._engine_seed))]

        if self._libsumo_started:
            try:
                self._sumo.close()
            except Exception:
                pass
            self._libsumo_started = False

        try:
            if self._use_libsumo:
                self._sumo.start(sumo_cmd)
            else:
                self._sumo.start(sumo_cmd, numRetries=_TRACI_START_NUM_RETRIES)
        except Exception as exc:
            if not self._use_libsumo:
                from traci.exceptions import FatalTraCIError

                if isinstance(exc, FatalTraCIError):
                    raise RuntimeError(
                        "TraCI could not connect to SUMO. Check that SUMO inputs "
                        "exist relative to the .sumocfg file, that SUMO_HOME is "
                        "set when your installation requires it, and that "
                        "`sumo --version` runs. "
                        f"sumocfg={self._config_path}"
                    ) from exc
            raise
        self._libsumo_started = True

        if self._roadnet_info is None:
            self._roadnet_info = extract_sumo_roadnet(self._sumo)
            self._intersections = self._roadnet_info.intersections
            self._num_intersections = len(self._intersections)
            
        self._last_action_applied = np.zeros(
            self._num_intersections, dtype=np.bool_
        )

    def _simulate(self, num_steps: int) -> None:
        # Advance simulation, letting metrics see every individual step
        # (SUMO only exposes arrivals for the last step taken).
        metrics = self._metrics
        if metrics is None:
            for _ in range(num_steps):
                self._sumo.simulationStep()
            return
        for _ in range(num_steps):
            self._sumo.simulationStep()
            metrics.on_sim_step()

    def _set_phase(
        self,
        ix_idx: int,
        phase: int,
        duration: int | None = None,
    ) -> None:
        ix_info = self._intersections[ix_idx]
        self._sumo.trafficlight.setPhase(ix_info.id, int(phase))
        self._sumo.trafficlight.setPhaseDuration(
            ix_info.id,
            int(duration if duration is not None else self.delta_time) + 1,
        )

    def _transition_recipe(self) -> tuple[tuple[str, int], ...]:
        return (
            ("yellow", TRANSITION_YELLOW_DURATION),
            ("all_red", TRANSITION_ALL_RED_DURATION),
        )

    def _apply_segment(self, ix_idx: int, segment: PhaseSegment) -> None:
        # setRedYellowGreenState pins the TLS to an "online" program, after
        # which setPhase is unreliable, so in transition mode every segment
        # (greens included) is rendered as an explicit state string.
        if not self._uses_synthetic_transition:
            super()._apply_segment(ix_idx, segment)
            return
        ix_info = self._intersections[ix_idx]
        self._sumo.trafficlight.setRedYellowGreenState(
            ix_info.id,
            self._segment_light_state(ix_info, segment),
        )

    @staticmethod
    def _segment_light_state(
        ix_info: IntersectionInfo,
        segment: PhaseSegment,
    ) -> str:
        if len(ix_info.phase_states) <= int(segment.phase):
            raise ValueError(
                f"Intersection {ix_info.id} is missing phase state "
                "metadata required for transition segments."
            )
        base_state = str(ix_info.phase_states[int(segment.phase)])
        if segment.kind == "phase":
            return base_state
        if segment.kind == "yellow":
            return base_state.replace("G", "y").replace("g", "y")
        if segment.kind == "all_red":
            return "r" * len(base_state)
        raise ValueError(
            f"Unsupported transition segment kind: {segment.kind}"
        )

    def _create_metrics(self) -> BaseMetrics | None:
        assert self._roadnet_info is not None
        return SumoMetrics(
            engine=self._sumo,
            intersections=self._intersections,
            metric_names=self._metric_names,
            delta_time=self.delta_time,
            roadnet=self._roadnet_info,
        )

    def _structured_features(self) -> dict[str, StructuredFeature]:
        return SumoStateFeatures.structured_features()


    def _get_info(self) -> dict[str, Any]:
        lane_waiting = (
            self._metrics.lane_waiting_vehicle_count() if self._metrics is not None else {}
        )
        lane_vehicle_count = (
            self._metrics.lane_vehicle_count() if self._metrics is not None else {}
        )

        intersections_info = self._build_intersections_info(
            lane_vehicle_count,
            lane_waiting,
        )

        info: dict[str, Any] = {
            "sim_time": self._sumo.simulation.getTime(),
            "vehicle_count": cast(int, self._sumo.vehicle.getIDCount()),
            "average_travel_time": (
                self._metrics.get("average_travel_time")
                if self._metrics is not None
                else 0.0
            ),
            "lane_vehicle_count": lane_vehicle_count,
            "lane_waiting_vehicle_count": lane_waiting,
            "step": self._step_count,
            "intersections": intersections_info,
        }
        if self._metrics is not None:
            info["metrics"] = self._metrics.compute_all()
        return info

    def close(self) -> None:
        """Clean up the SUMO instance."""
        if self._libsumo_started:
            try:
                self._sumo.close()
            except Exception:
                pass
            self._libsumo_started = False

        if self._tmp_config_dir is not None and os.path.isdir(
            self._tmp_config_dir
        ):
            import shutil
            shutil.rmtree(self._tmp_config_dir, ignore_errors=True)
