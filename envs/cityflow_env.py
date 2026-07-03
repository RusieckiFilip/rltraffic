from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cityflow
import numpy as np

from envs.base_traffic_env import BaseTrafficEnv
from envs.phase_control import AcyclicPhases, PhaseControl, PhaseSegment
from utils.cityflow_utils import parse_roadnet
from utils.common_utils import IntersectionInfo, RoadnetInfo
from metrics import CityFlowMetrics
from metrics.base import BaseMetrics
from states.base import StructuredFeature
from states.cityflow import CityFlowStateFeatures



class CityFlowEnv(BaseTrafficEnv):
    """Gymnasium environment wrapping the CityFlow traffic simulator.

    Parameters
    ----------
    cityflow_config_path : str | Path
        Path to the CityFlow engine configuration JSON file.
    max_steps : int
        Maximum number of *agent* decision steps per episode.
    delta_time : int
        Simulation seconds between consecutive agent actions.
    thread_num : int
        Number of threads for the CityFlow engine.

    Seeding follows the Gymnasium convention: ``reset(seed=X)`` re-seeds
    the env RNG, ``reset()`` continues it; every reset draws a fresh
    engine seed from that RNG and applies it via ``set_random_seed``.
    Seed once for a reproducible sequence of varied episodes; pass the
    same seed every reset for identical episodes.
    """

    def __init__(
        self,
        cityflow_config_path: str | Path = "",
        max_steps: int = 3600,
        delta_time: int = 10,
        thread_num: int = 1,
        global_reward_fn: str | Callable[[dict[str, Any]], float] | None = None,
        phase_control_cls: type[PhaseControl] = AcyclicPhases,
        phase_bounds: np.ndarray | None = None,
        metrics: Iterable[str] | None = None,
        local_reward_fn: str | Callable[[dict[str, Any]], float] | None = None,
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
        self._config_path = str(Path(cityflow_config_path).resolve())
        self._thread_num = thread_num
        self._configure_rewards(
            global_reward_fn,
            local_reward_fn=local_reward_fn,
            global_reward_weight=global_reward_weight,
        )

        with open(self._config_path, "r") as f:
            cfg = json.load(f)

        # Resolve the dir to an absolute path so CityFlow's C++ engine
        # can always find roadnet / flow files regardless of CWD.
        cfg_dir = cfg.get("dir", "")
        if not os.path.isabs(cfg_dir):
            cfg_dir = str(Path.cwd() / cfg_dir)
        self._abs_cfg_dir = os.path.normpath(cfg_dir)

        roadnet_file = cfg.get("roadnetFile", "")
        self._roadnet_path = str(Path(self._abs_cfg_dir) / roadnet_file)

        # Write a temp config with the absolute dir so CityFlow can
        # locate the roadnet/flow files from any working directory.
        cfg["dir"] = self._abs_cfg_dir + "/"

        # Resolve replay log paths relative to CWD (project root).
        # CityFlow resolves these relative to ``dir``, so we compute
        # the relative path from abs_cfg_dir back to the target.
        for key in ("roadnetLogFile", "replayLogFile"):
            log_path = cfg.get(key, "")
            if log_path and not os.path.isabs(log_path):
                abs_log = Path.cwd() / log_path
                abs_log.parent.mkdir(parents=True, exist_ok=True)
                cfg[key] = os.path.relpath(str(abs_log), self._abs_cfg_dir)

        self._tmp_config_dir = tempfile.mkdtemp(prefix="cityflow_")
        self._eng: cityflow.Engine | None = None

        # A failure past this point must not leak the temp config dir (or a
        # started engine), so clean up before re-raising.
        try:
            # CityFlow can only switch phases by index, so the all-red
            # clearance used by the acyclic-transition controls must exist as a
            # real phase: append one to every traffic light in an engine-only
            # copy of the roadnet (index == original phase count per intersection).
            if self._uses_synthetic_transition:
                injected_roadnet = os.path.join(
                    self._tmp_config_dir, "roadnet_all_red.json"
                )
                self._write_all_red_roadnet(self._roadnet_path, injected_roadnet)
                cfg["roadnetFile"] = os.path.relpath(
                    injected_roadnet, self._abs_cfg_dir
                )

            self._engine_config_path = os.path.join(
                self._tmp_config_dir, "config.json"
            )
            with open(self._engine_config_path, "w") as f:
                json.dump(cfg, f)

            # Parse the road network topology
            self._roadnet_info: RoadnetInfo = parse_roadnet(self._roadnet_path)
            self._intersections: list[IntersectionInfo] = (
                self._roadnet_info.intersections
            )
            self._num_intersections = len(self._intersections)

            if self._num_intersections == 0:
                raise ValueError(
                    "No controllable (non-virtual) intersections found in "
                    f"roadnet file: {self._roadnet_path}"
                )

            self._phase_bounds: np.ndarray | None = None
            self._last_action_applied: np.ndarray = np.zeros(
                self._num_intersections, dtype=np.bool_
            )

            # Build spaces and do first reset
            self._init_simulator()
            self._setup_spaces()

            # Build the metrics object now that the engine is alive (will be
            # rebuilt in ``reset()`` for fresh episodes).
            self._init_metrics()
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _init_simulator(self) -> None:
        if self._eng is not None:
            # Reuse the existing engine – CityFlow's reset clears all
            # vehicles and resets simulation time to zero.
            self._eng.reset()
        else:
            self._eng = cityflow.Engine(
                self._engine_config_path, self._thread_num
            )
        if self._engine_seed is not None:
            # CityFlow's Engine.reset() re-seeds its RNG with the seed
            # from the config file, so without this call every episode
            # replays identically and reset(seed=...) is silently
            # ignored. Apply the per-episode seed drawn by the base env
            # (mirrors SumoEnv's --seed handling).
            self._eng.set_random_seed(int(self._engine_seed))
        self._last_action_applied = np.zeros(
            self._num_intersections, dtype=np.bool_
        )

    def _simulate(self, num_steps: int) -> None:
        assert self._eng is not None
        for _ in range(num_steps):
            self._eng.next_step()

    def _set_phase(
        self,
        ix_idx: int,
        phase: int,
        duration: int | None = None,
    ) -> None:
        _ = duration
        assert self._eng is not None
        ix_info = self._intersections[ix_idx]
        self._eng.set_tl_phase(ix_info.id, int(phase))

    @staticmethod
    def _write_all_red_roadnet(src_path: str, dst_path: str) -> None:
        """Copy the roadnet, appending an all-red phase to every TL.

        The injected phase (no available road links) sits at index
        ``len(lightphases)`` of the original file.  Topology parsing keeps
        using the original roadnet, so observation / action spaces never
        see it.
        """
        with open(src_path, "r") as f:
            roadnet = json.load(f)
        for intersection in roadnet.get("intersections", []):
            traffic_light = intersection.get("trafficLight") or {}
            lightphases = traffic_light.get("lightphases")
            if not lightphases:
                continue
            lightphases.append({"time": 5, "availableRoadLinks": []})
        with open(dst_path, "w") as f:
            json.dump(roadnet, f)

    def _apply_segment(self, ix_idx: int, segment: PhaseSegment) -> None:
        if segment.kind == "phase":
            self._set_phase(ix_idx, segment.phase, segment.duration)
            return
        if segment.kind == "all_red":
            assert self._eng is not None
            ix_info = self._intersections[ix_idx]
            self._eng.set_tl_phase(ix_info.id, int(ix_info.num_phases))
            return
        raise ValueError(
            f"Unsupported transition segment kind for CityFlow: "
            f"{segment.kind}"
        )

    def _create_metrics(self) -> BaseMetrics | None:
        if self._eng is None:
            return None
        return CityFlowMetrics(
            engine=self._eng,
            intersections=self._intersections,
            metric_names=self._metric_names,
            delta_time=self.delta_time,
            roadnet=self._roadnet_info,
        )

    def _structured_features(self) -> dict[str, StructuredFeature]:
        return CityFlowStateFeatures.structured_features()


    def _get_info(self) -> dict[str, Any]:
        assert self._eng is not None

        lane_vehicle_count = (
            self._metrics.lane_vehicle_count()
            if self._metrics is not None
            else self._eng.get_lane_vehicle_count()
        )
        lane_waiting = (
            self._metrics.lane_waiting_vehicle_count()
            if self._metrics is not None
            else self._eng.get_lane_waiting_vehicle_count()
        )

        intersections_info = self._build_intersections_info(
            lane_vehicle_count,
            lane_waiting,
        )

        info: dict[str, Any] = {
            "sim_time": self._eng.get_current_time(),
            "vehicle_count": self._eng.get_vehicle_count(),
            "average_travel_time": (
                self._metrics.get("average_travel_time")
                if self._metrics is not None
                else self._eng.get_average_travel_time()
            ),
            "lane_vehicle_count": lane_vehicle_count,
            "lane_waiting_vehicle_count": lane_waiting,
            "step": self._step_count,
            "intersections": intersections_info,
        }
        if self._metrics is not None:
            info["metrics"] = self._metrics.compute_all()
        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up the temporary config file."""
        import shutil

        self._eng = None

        if hasattr(self, "_tmp_config_dir") and os.path.isdir(
            self._tmp_config_dir
        ):
            shutil.rmtree(self._tmp_config_dir, ignore_errors=True)
