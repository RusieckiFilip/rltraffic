"""CityFlow-backed implementation of :class:`BaseMetrics`."""

from __future__ import annotations

from typing import Any, Iterable

from metrics.base import BaseMetrics, register, register_local


# Vehicles that move slower than this are considered "halted" / "waiting".
HALT_SPEED_THRESHOLD = 0.1


class CityFlowMetrics(BaseMetrics):
    """Compute traffic-signal-control metrics from a ``cityflow.Engine``.

    Parameters
    ----------
    engine
        Active ``cityflow.Engine`` instance owned by the env.
    intersections
        ``IntersectionInfo`` list whose ``incoming_lanes`` we monitor for
        the *incoming-lane* metrics.
    metric_names
        Names of metrics the user wants computed.
    delta_time
        Number of simulation seconds between consecutive ``update()``
        calls (i.e. ``env.delta_time``).
    roadnet
        Parsed ``RoadnetInfo`` providing per-road lengths and speed
        limits used to compute *ideal* travel times.
    """

    def __init__(
        self,
        engine: Any,
        intersections: list[Any],
        metric_names: Iterable[str] | None = None,
        *,
        delta_time: float,
        roadnet: Any,
    ) -> None:
        self._roadnet = roadnet
        self._incoming_lane_set: set[str] = {
            lid for ix in intersections for lid in ix.incoming_lanes
        }
        super().__init__(
            engine=engine,
            intersections=intersections,
            metric_names=metric_names,
            delta_time=delta_time,
        )

    # ------------------------------------------------------------------
    # Cached per-step primitives
    # ------------------------------------------------------------------

    def _vehicles(self) -> list[str]:
        return self._cached(
            "vehicles", lambda: list(self._engine.get_vehicles(include_waiting=False))
        )

    def _vehicle_speeds(self) -> dict[str, float]:
        def _fn() -> dict[str, float]:
            raw = self._engine.get_vehicle_speed()
            return {k: float(v) for k, v in raw.items()}

        return self._cached("vehicle_speeds", _fn)

    def _lane_waiting_count(self) -> dict[str, int]:
        return self._cached(
            "lane_waiting", lambda: dict(self._engine.get_lane_waiting_vehicle_count())
        )

    def _lane_vehicles(self) -> dict[str, list[str]]:
        return self._cached(
            "lane_vehicles", lambda: dict(self._engine.get_lane_vehicles())
        )

    def _current_time(self) -> float:
        return self._cached(
            "current_time", lambda: float(self._engine.get_current_time())
        )

    def lane_waiting_vehicle_count(self) -> dict[str, int]:
        waiting = self._lane_waiting_count()
        return {lid: int(count) for lid, count in waiting.items()}

    def lane_vehicle_count(self) -> dict[str, int]:
        lane_vehicles = self._lane_vehicles()
        return {lid: len(vehicles) for lid, vehicles in lane_vehicles.items()}

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        # Per-vehicle accumulators (populated as vehicles appear)
        self._episode["depart_time"] = {}     # vid -> first-seen time
        self._episode["dist_acc"] = {}        # vid -> cumulative distance (m)
        self._episode["wait_acc"] = {}        # vid -> cumulative waiting time (s), mirrors SUMO getAccumulatedWaitingTime with long memory
        self._episode["ideal_tt"] = {}        # vid -> ideal travel time (s)

        # Records for vehicles that have completed their journey.
        self._episode["completed"] = []       # list[dict]
        self._episode["completed_wait_total"] = 0.0

        # Set of vehicles seen on the previous step (used for diffing).
        self._episode["prev_active"] = set()

        # Episode-cumulative integrals (for "*_in_episode" metrics).
        self._episode["wait_total_episode"] = 0.0
        self._episode["wait_incoming_episode"] = 0.0

        # Pre-action snapshot for the action-delta metric.
        self._episode["pre_action"] = {}
        self._episode["throughput_delta"] = 0.0

    def _ideal_route_time(self, route_str: str) -> float:
        """Sum of ``road_length / road_max_speed`` over the route."""
        roads = [r for r in route_str.strip().split() if r]
        total = 0.0
        for rid in roads:
            length = float(self._roadnet.road_lengths.get(rid, 0.0))
            speed = float(self._roadnet.road_max_speeds.get(rid, 0.0))
            if speed > 0.0:
                total += length / speed
        return total

    def _vehicle_route(self, vid: str) -> str:
        try:
            info = self._engine.get_vehicle_info(vid)
        except Exception:
            return ""
        return str(info.get("route", ""))

    def _current_max_speed(self, vid: str) -> float:
        """Speed limit of the road this vehicle is currently on."""
        try:
            info = self._engine.get_vehicle_info(vid)
        except Exception:
            return 0.0
        road = info.get("road", "")
        return float(self._roadnet.road_max_speeds.get(road, 0.0))

    def _snapshot_pre_action(self) -> None:
        """Record ``(t, distance, v_max, depart)`` for every running vehicle.

        ``depart`` is recorded so the delta metric can normalise by each
        vehicle's *travel time since departure* (``t - depart``) rather than
        the absolute simulation clock.  Distance is stored raw (no floor):
        vehicles that have not moved yet (``x <= 0``) are skipped downstream
        instead of being turned into huge ``t / x`` outliers.
        """
        snapshot: dict[str, tuple[float, float, float, float]] = {}
        t = float(self._engine.get_current_time())
        depart_time = self._episode["depart_time"]
        dist_acc = self._episode["dist_acc"]
        for vid in self._engine.get_vehicles(include_waiting=False):
            x = float(dist_acc.get(vid, 0.0))
            v_max = self._current_max_speed(vid)
            depart = float(depart_time.get(vid, t))
            snapshot[vid] = (t, x, v_max, depart)
        self._episode["pre_action"] = snapshot

    def _run_step_hooks(self) -> None:
        """Update episode accumulators based on the latest step."""
        speeds = self._vehicle_speeds()
        active_vehicles = self._vehicles()
        active_set = set(active_vehicles)
        t = self._current_time()
        delta_time = self._delta_time

        depart_time = self._episode["depart_time"]
        dist_acc = self._episode["dist_acc"]
        wait_acc = self._episode["wait_acc"]
        ideal_tt = self._episode["ideal_tt"]

        # 1. New vehicles: seed bookkeeping; cache route ideal-time.
        # The vehicle appeared somewhere inside (t - delta_time, t]; use
        # the window midpoint as an unbiased estimator of departure time.
        depart_estimate = t - delta_time / 2.0
        for vid in active_set - self._episode["prev_active"]:
            depart_time[vid] = depart_estimate
            dist_acc.setdefault(vid, 0.0)
            wait_acc.setdefault(vid, 0.0)
            ideal_tt.setdefault(vid, self._ideal_route_time(self._vehicle_route(vid)))

        # 2. Update distance + waiting accumulators for active vehicles.
        for vid in active_set:
            speed = float(speeds.get(vid, 0.0))
            dist_acc[vid] = dist_acc.get(vid, 0.0) + speed * delta_time
            if speed < HALT_SPEED_THRESHOLD:
                wait_acc[vid] = wait_acc.get(vid, 0.0) + delta_time

        # 3. Detect completed vehicles (left the network this step).
        completed_vids = self._episode["prev_active"] - active_set
        self._episode["throughput_delta"] = float(len(completed_vids))
        for vid in completed_vids:
            real_tt = max(0.0, t - depart_time.get(vid, t))
            ideal = float(ideal_tt.get(vid, 0.0))
            wait = float(wait_acc.get(vid, 0.0))
            delay_ratio = real_tt / ideal if ideal > 0.0 else 0.0
            self._episode["completed"].append(
                {
                    "vid": vid,
                    "real_tt": real_tt,
                    "ideal_tt": ideal,
                    "delay": delay_ratio,
                    "wait_acc": wait,
                }
            )
            self._episode["completed_wait_total"] += wait

        # 4. Episode-cumulative waiting-time integrals.
        if self._requires_any(
            "total_waiting_time_all_vehicles_in_simulation_in_episode",
            "total_waiting_time_on_the_incoming_lanes_in_episode",
        ):
            lane_waiting = self._lane_waiting_count()
            n_wait_total = sum(int(c) for c in lane_waiting.values())
            n_wait_incoming = sum(
                int(lane_waiting.get(lid, 0)) for lid in self._incoming_lane_set
            )
            self._episode["wait_total_episode"] += n_wait_total * delta_time
            self._episode["wait_incoming_episode"] += n_wait_incoming * delta_time

        self._episode["prev_active"] = active_set

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    @register("average_travel_time")
    def _average_travel_time(self) -> float:
        """Average over *all* vehicles that entered the network: completed
        trips count their full travel time, vehicles still en route count
        the time elapsed since departure (no survivorship bias)."""
        completed = self._episode["completed"]
        total = sum(c["real_tt"] for c in completed)
        count = len(completed)
        depart_time = self._episode["depart_time"]
        if depart_time:
            t = self._current_time()
            # depart_time keeps completed vehicles too, so iterate the
            # currently-active set instead of the dict.
            for vid in self._vehicles():
                depart = depart_time.get(vid)
                if depart is not None:
                    total += max(0.0, t - depart)
                    count += 1
        if count == 0:
            return 0.0
        return float(total / count)

    @register("throughput_delta")
    def _throughput_delta(self) -> float:
        return float(self._episode.get("throughput_delta", 0.0))

    @register("count_of_vehicles_completing_journey")
    def _count_completed(self) -> float:
        return float(len(self._episode["completed"]))

    @register("total_time_of_journey")
    def _total_time_of_journey(self) -> float:
        return float(sum(c["real_tt"] for c in self._episode["completed"]))

    @register("average_time_of_journey")
    def _average_time_of_journey(self) -> float:
        completed = self._episode["completed"]
        if not completed:
            return 0.0
        return float(sum(c["real_tt"] for c in completed) / len(completed))

    @register("total_sum_delays_of_all_vehicles_from_all_routes")
    def _total_sum_delays(self) -> float:
        return float(sum(c["delay"] for c in self._episode["completed"]))

    @register("total_average_delays_of_all_vehicles_from_all_routes")
    def _total_average_delays(self) -> float:
        completed = self._episode["completed"]
        if not completed:
            return 0.0
        return float(sum(c["delay"] for c in completed) / len(completed))

    @register("total_average_delays_real_times_by_ideal_times")
    def _total_average_delays_ratio(self) -> float:
        completed = self._episode["completed"]
        ideal_sum = sum(c["ideal_tt"] for c in completed)
        real_sum = sum(c["real_tt"] for c in completed)
        if ideal_sum <= 0.0:
            return 0.0
        return float(real_sum / ideal_sum)

    @register("waiting_time_all_vehicles_for_the_last_time_step_in_simulation")
    def _wait_time_now(self) -> float:
        wait_acc = self._episode["wait_acc"]
        return float(sum(wait_acc.get(vid, 0.0) for vid in self._vehicles()))

    @register_local(
        "waiting_time_all_vehicles_for_the_last_time_step_in_simulation"
    )
    def _wait_time_now_local(self) -> dict[str, float]:
        """Accumulated waiting time of the vehicles currently on each
        intersection's incoming lanes (RESCo's per-junction wait)."""
        wait_acc = self._episode["wait_acc"]
        lane_vehicles = self._lane_vehicles()
        totals: dict[str, float] = {}
        for ix in self._intersections:
            total = 0.0
            for lid in ix.incoming_lanes:
                for vid in lane_vehicles.get(lid, ()):
                    total += float(wait_acc.get(vid, 0.0))
            totals[str(ix.id)] = total
        return totals

    @register(
        "total_waiting_time_all_vehicles_in_simulation_in_episode", episode=True
    )
    def _total_wait_episode(self) -> float:
        return float(self._episode["wait_total_episode"])

    @register(
        "total_waiting_time_on_the_incoming_lanes_in_episode", episode=True
    )
    def _total_wait_incoming_episode(self) -> float:
        return float(self._episode["wait_incoming_episode"])

    @register(
        "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
    )
    def _halting_vehicles(self) -> float:
        speeds = self._vehicle_speeds()
        return float(sum(1 for s in speeds.values() if s < HALT_SPEED_THRESHOLD))

    @register("calculate_average_delta_of_delays_after_action")
    def _avg_delta_delay(self) -> float:
        pre = self._episode.get("pre_action", {})
        if not pre:
            return 0.0
        t_post = self._current_time()
        dist_acc = self._episode["dist_acc"]
        active_set = set(self._vehicles())

        deltas: list[float] = []
        for vid, (t_pre, x_pre, v_max, depart) in pre.items():
            if vid not in active_set:
                continue
            x_post = float(dist_acc.get(vid, 0.0))
            if x_pre <= 0.0 or x_post <= 0.0 or v_max <= 0.0:
                continue
            tau_pre = max(0.0, t_pre - depart)
            tau_post = max(0.0, t_post - depart)
            deltas.append(v_max * (tau_pre / x_pre - tau_post / x_post))
        if not deltas:
            return 0.0
        return float(sum(deltas) / len(deltas))
