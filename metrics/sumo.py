"""SUMO/libsumo-backed implementation of :class:`BaseMetrics`."""

from __future__ import annotations

from typing import Any, Iterable

from metrics.base import BaseMetrics, register, register_local


HALT_SPEED_THRESHOLD = 0.1

METRIC_STICKY_WAIT = "resco_sticky_waiting_time_on_the_incoming_lanes"

# RESCo's Signal.observe only "detects" vehicles whose next traffic light
# is at most this far away (agent_config max_distance for IDQN/IPPO).
_STICKY_MAX_DISTANCE = 200.0

_COMPLETED_TRIP_METRICS = {
    "average_travel_time",
    "count_of_vehicles_completing_journey",
    "total_time_of_journey",
    "average_time_of_journey",
    "total_sum_delays_of_all_vehicles_from_all_routes",
    "total_average_delays_of_all_vehicles_from_all_routes",
    "total_average_delays_real_times_by_ideal_times",
}


class SumoMetrics(BaseMetrics):
    """Compute traffic-signal-control metrics via the ``traci``/``libsumo``
    Python API.

    Parameters
    ----------
    engine
        The imported ``traci`` (or ``libsumo``) module that's already
        connected to a running simulation.
    intersections
        ``IntersectionInfo`` list whose ``incoming_lanes`` we monitor.
    metric_names
        Names of metrics the user wants computed.
    delta_time
        Number of simulation seconds between consecutive ``update()``
        calls.
    roadnet
        Static SUMO road network metadata used to compute ideal route times.
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
        self._monitored_lane_set: set[str] = {
            lid
            for ix in intersections
            for lid in (list(ix.incoming_lanes) + list(ix.outgoing_lanes))
        }
        try:
            self._all_lane_ids: set[str] = set(engine.lane.getIDList())
        except Exception:
            self._all_lane_ids = set(self._monitored_lane_set)
        super().__init__(
            engine=engine,
            intersections=intersections,
            metric_names=metric_names,
            delta_time=delta_time,
        )
        self._track_completed = self._requires_any(*_COMPLETED_TRIP_METRICS)
        self._collect_arrivals = self._track_completed or self.requires(
            "throughput_delta"
        )

    # ------------------------------------------------------------------
    # Cached per-step primitives
    # ------------------------------------------------------------------

    @property
    def _traci(self) -> Any:
        return self._engine

    def _vehicle_ids(self) -> list[str]:
        return self._cached(
            "vehicle_ids", lambda: list(self._traci.vehicle.getIDList())
        )

    def _current_time(self) -> float:
        return self._cached(
            "current_time", lambda: float(self._traci.simulation.getTime())
        )

    def _vehicle_speeds(self) -> dict[str, float]:
        def _fn() -> dict[str, float]:
            v = self._traci.vehicle
            return {vid: float(v.getSpeed(vid)) for vid in self._vehicle_ids()}

        return self._cached("vehicle_speeds", _fn)

    def _vehicle_accumulated_waiting(self) -> dict[str, float]:
        def _fn() -> dict[str, float]:
            v = self._traci.vehicle
            return {
                vid: float(v.getAccumulatedWaitingTime(vid))
                for vid in self._vehicle_ids()
            }

        return self._cached("vehicle_acc_wait", _fn)

    def _vehicle_distances(self) -> dict[str, float]:
        def _fn() -> dict[str, float]:
            v = self._traci.vehicle
            return {vid: float(v.getDistance(vid)) for vid in self._vehicle_ids()}

        return self._cached("vehicle_distances", _fn)

    def _lane_halting_count(self) -> dict[str, int]:
        """Per-lane halting count for *every monitored lane*."""

        def _fn() -> dict[str, int]:
            lane = self._traci.lane
            counts: dict[str, int] = {}
            for lid in self._monitored_lane_set:
                counts[lid] = int(lane.getLastStepHaltingNumber(lid))
            return counts

        return self._cached("lane_halting", _fn)

    def _all_lane_halting_count(self) -> dict[str, int]:
        def _fn() -> dict[str, int]:
            lane = self._traci.lane
            counts: dict[str, int] = {}
            for lid in self._all_lane_ids:
                counts[lid] = int(lane.getLastStepHaltingNumber(lid))
            return counts

        return self._cached("all_lane_halting", _fn)

    def _lane_vehicle_count(self) -> dict[str, int]:
        def _fn() -> dict[str, int]:
            lane = self._traci.lane
            counts: dict[str, int] = {}
            for lid in self._monitored_lane_set:
                counts[lid] = int(lane.getLastStepVehicleNumber(lid))
            return counts

        return self._cached("lane_vehicle_count", _fn)

    def _lane_vehicle_ids(self) -> dict[str, list[str]]:
        """Vehicle ids currently on each *incoming* lane."""

        def _fn() -> dict[str, list[str]]:
            lane = self._traci.lane
            return {
                lid: list(lane.getLastStepVehicleIDs(lid))
                for lid in self._incoming_lane_set
            }

        return self._cached("lane_vehicle_ids", _fn)

    def lane_waiting_vehicle_count(self) -> dict[str, int]:
        return dict(self._lane_halting_count())

    def lane_vehicle_count(self) -> dict[str, int]:
        return dict(self._lane_vehicle_count())

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        self._episode["depart_time"] = {}      # vid -> first-seen time
        self._episode["ideal_tt"] = {}         # vid -> ideal travel time

        # Snapshot of last-known state per active vehicle so we can
        # finalise stats once SUMO drops it from getIDList().
        self._episode["last_seen"] = {}        # vid -> dict
        self._episode["completed"] = []        # finalised journey records

        # (vid, arrival_time) pairs collected by on_sim_step() since the
        # last update().  SUMO's getArrivedIDList only reports the *last*
        # step's arrivals, so with delta_time > 1 they must be gathered
        # after every individual simulation step.
        self._episode["arrived_buffer"] = []
        # Becomes True once the env drives on_sim_step(); until then
        # update() falls back to polling getArrivedIDList itself.
        self._hook_driven = False

        # Episode-cumulative waiting integrals (s).
        self._episode["wait_total_episode"] = 0.0
        self._episode["wait_incoming_episode"] = 0.0

        # Cache: ideal time for each route id (avoid recomputing).
        self._episode["route_ideal_tt"] = {}

        # Pre-action snapshot for the action-delta metric.
        self._episode["pre_action"] = {}
        self._episode["throughput_delta"] = 0.0

        # RESCo-style sticky wait tracker: per intersection, vid -> wait
        # seconds, plus the per-lane "detectable" vehicle lists from the
        # latest step (see _update_sticky_wait).
        self._episode["sticky_wait"] = {
            str(ix.id): {} for ix in self._intersections
        }
        self._episode["sticky_detectable"] = {}

    def _route_ideal_time(self, route_edges: list[str]) -> float:
        cache: dict[tuple, float] = self._episode["route_ideal_tt"]
        key = tuple(route_edges)
        if key in cache:
            return cache[key]
        total = 0.0
        for edge in route_edges:
            length = float(self._roadnet.road_lengths.get(edge, 0.0))
            speed = float(self._roadnet.road_max_speeds.get(edge, 0.0))
            if speed > 0.0:
                total += length / speed
        cache[key] = total
        return total

    def _vehicle_route(self, vid: str) -> list[str]:
        try:
            return list(self._traci.vehicle.getRoute(vid))
        except Exception:
            return []

    def _vehicle_depart_time(self, vid: str, fallback: float) -> float:
        """Actual network-entry time of *vid* (exact, unlike estimating
        from the update window)."""
        try:
            depart = float(self._traci.vehicle.getDeparture(vid))
        except Exception:
            return fallback
        return depart if depart >= 0.0 else fallback

    def on_sim_step(self) -> None:
        if not self._collect_arrivals:
            return
        self._hook_driven = True
        arrived = self._traci.simulation.getArrivedIDList()
        if arrived:
            t = float(self._traci.simulation.getTime())
            buf = self._episode["arrived_buffer"]
            for vid in arrived:
                buf.append((vid, t))

    def _drain_arrivals(self) -> list[tuple[str, float]]:
        """All ``(vid, arrival_time)`` pairs since the previous update."""
        buf: list[tuple[str, float]] = self._episode["arrived_buffer"]
        if not self._hook_driven:
            # Caller advances the simulation without driving
            # on_sim_step(); only the final step's arrivals are visible.
            t = self._current_time()
            buf.extend(
                (vid, t) for vid in self._traci.simulation.getArrivedIDList()
            )
        drained = list(buf)
        buf.clear()
        return drained

    def _snapshot_pre_action(self) -> None:
        """Snapshot ``(t, distance, v_max, depart)`` for every running vehicle.

        ``depart`` is recorded so the delta metric can normalise by each
        vehicle's *travel time since departure* (``t - depart``) rather than
        the absolute simulation clock.  Distance is stored raw (no floor):
        vehicles that have not moved yet (``x <= 0``) are skipped downstream
        instead of being turned into huge ``t / x`` outliers.
        """
        v = self._traci.vehicle
        snapshot: dict[str, tuple[float, float, float, float]] = {}
        t = float(self._traci.simulation.getTime())
        for vid in v.getIDList():
            try:
                x = float(v.getDistance(vid))
                v_max = float(v.getAllowedSpeed(vid))
            except Exception:
                continue
            depart = self._vehicle_depart_time(vid, t)
            snapshot[vid] = (t, x, v_max, depart)
        self._episode["pre_action"] = snapshot

    def warmup(self) -> None:
        """Seed the observation caches from the ``t=0`` sim state.

        Only ``drq_norm`` reads per-step state that ``update()`` would
        otherwise produce: the sticky-wait detection set
        (``sticky_detectable``).  We populate it here without running the
        episode accumulators in ``_run_step_hooks``.  ``getWaitingTime``-
        seeded entries are correct at ``t=0``; no ``delta_time`` is
        accrued because the trackers start empty.
        """
        self._step_cache.clear()
        if self.requires(METRIC_STICKY_WAIT):
            self._update_sticky_wait()

    def _update_sticky_wait(self) -> None:
        """Replicates RESCo's ``Signal.observe`` waiting-time bookkeeping.

        Per junction, only vehicles on incoming lanes whose next traffic
        light is within ``_STICKY_MAX_DISTANCE`` are "detectable".  A
        detectable vehicle enters the tracker once it has stopped
        (``getWaitingTime() > 0``, seeded with that value) and from then
        on accrues ``delta_time`` every step — even while crawling —
        until it leaves the junction's detection set, at which point it
        is forgotten.
        """
        delta_time = self._delta_time
        vehicle = self._traci.vehicle
        lane_vehicles = self._lane_vehicle_ids()
        sticky_all = self._episode["sticky_wait"]
        detectable_by_lane: dict[str, list[str]] = {}
        for ix in self._intersections:
            sticky = sticky_all[str(ix.id)]
            seen: set[str] = set()
            for lid in ix.incoming_lanes:
                detectable = detectable_by_lane.setdefault(lid, [])
                for vid in lane_vehicles.get(lid, ()):
                    try:
                        next_tls = vehicle.getNextTLS(vid)
                    except Exception:
                        continue
                    if not next_tls or float(next_tls[0][2]) > _STICKY_MAX_DISTANCE:
                        continue
                    detectable.append(vid)
                    seen.add(vid)
                    if vid in sticky:
                        sticky[vid] += delta_time
                    else:
                        wait = float(vehicle.getWaitingTime(vid))
                        if wait > 0.0:
                            sticky[vid] = wait
            for vid in list(sticky):
                if vid not in seen:
                    del sticky[vid]
        self._episode["sticky_detectable"] = detectable_by_lane

    def _run_step_hooks(self) -> None:
        delta_time = self._delta_time
        track_completed = self._track_completed
        if self._collect_arrivals:
            arrived = self._drain_arrivals()
            self._episode["throughput_delta"] = float(len(arrived))
        else:
            arrived = []

        if self.requires(METRIC_STICKY_WAIT):
            self._update_sticky_wait()

        if not track_completed:
            if self.requires(
                "total_waiting_time_all_vehicles_in_simulation_in_episode"
            ):
                speeds = self._vehicle_speeds()
                n_wait_total = sum(
                    1 for s in speeds.values() if s < HALT_SPEED_THRESHOLD
                )
                self._episode["wait_total_episode"] += n_wait_total * delta_time

            if self.requires(
                "total_waiting_time_on_the_incoming_lanes_in_episode"
            ):
                halting = self._lane_halting_count()
                n_wait_incoming = sum(
                    int(halting.get(lid, 0)) for lid in self._incoming_lane_set
                )
                self._episode["wait_incoming_episode"] += (
                    n_wait_incoming * delta_time
                )
            return

        t = self._current_time()
        depart_time = self._episode["depart_time"]
        ideal_tt = self._episode["ideal_tt"]
        last_seen = self._episode["last_seen"]

        # 1. Finalise vehicles that arrived since the last update (using
        #    the last snapshot we still have for them).
        for vid, t_arr in arrived:
            seen = last_seen.pop(vid, None)
            if seen is None:
                # Vehicle arrived without ever being snapshotted (e.g. a
                # zero-step vehicle).  Use whatever fallback we have.
                real_tt = max(0.0, t_arr - depart_time.pop(vid, t_arr))
                ideal = ideal_tt.pop(vid, 0.0)
                wait = 0.0
            else:
                real_tt = max(
                    0.0, t_arr - depart_time.pop(vid, seen["depart_time"])
                )
                ideal = ideal_tt.pop(vid, seen["ideal_tt"])
                wait = float(seen["acc_wait"])
            # ReSCo defines "delay" as real_tt / ideal_tt (slowdown ratio).
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

        # 2. Update bookkeeping for currently-active vehicles.
        active = self._vehicle_ids()
        speeds = self._vehicle_speeds()
        distances = self._vehicle_distances()
        acc_wait = self._vehicle_accumulated_waiting()

        # New vehicle: SUMO reports its exact network-entry time; fall
        # back to the window midpoint if the query fails.
        depart_estimate = t - delta_time / 2.0
        for vid in active:
            if vid not in depart_time:
                depart_time[vid] = self._vehicle_depart_time(
                    vid, depart_estimate
                )
                route = self._vehicle_route(vid)
                ideal_tt[vid] = self._route_ideal_time(route)
            last_seen[vid] = {
                "depart_time": depart_time[vid],
                "ideal_tt": ideal_tt[vid],
                "distance": float(distances.get(vid, 0.0)),
                "speed": float(speeds.get(vid, 0.0)),
                "acc_wait": float(acc_wait.get(vid, 0.0)),
            }

        # 3. Episode-cumulative integrals (only if requested).
        if self._requires_any(
            "total_waiting_time_all_vehicles_in_simulation_in_episode",
        ):
            n_wait_total = sum(
                1 for s in speeds.values() if s < HALT_SPEED_THRESHOLD
            )
            self._episode["wait_total_episode"] += n_wait_total * delta_time

        if self._requires_any(
            "total_waiting_time_on_the_incoming_lanes_in_episode",
        ):
            halting = self._lane_halting_count()
            n_wait_incoming = sum(
                int(halting.get(lid, 0)) for lid in self._incoming_lane_set
            )
            self._episode["wait_incoming_episode"] += n_wait_incoming * delta_time

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
            for vid in self._vehicle_ids():
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
        return float(sum(self._vehicle_accumulated_waiting().values()))

    @register_local(
        "waiting_time_all_vehicles_for_the_last_time_step_in_simulation"
    )
    def _wait_time_now_local(self) -> dict[str, float]:
        """Accumulated waiting time of the vehicles currently on each
        intersection's incoming lanes (RESCo's per-junction wait)."""
        acc_wait = self._vehicle_accumulated_waiting()
        lane_vehicles = self._lane_vehicle_ids()
        totals: dict[str, float] = {}
        for ix in self._intersections:
            total = 0.0
            for lid in ix.incoming_lanes:
                for vid in lane_vehicles.get(lid, ()):
                    total += float(acc_wait.get(vid, 0.0))
            totals[str(ix.id)] = total
        return totals

    @register_local(METRIC_STICKY_WAIT)
    def _sticky_wait_local(self) -> dict[str, float]:
        """RESCo's exact per-junction waiting time (see
        ``_update_sticky_wait`` for the bookkeeping rules)."""
        sticky_all = self._episode["sticky_wait"]
        return {
            str(ix.id): float(sum(sticky_all[str(ix.id)].values()))
            for ix in self._intersections
        }

    def drq_norm_observation(
        self, ix: Any, phase_index: int
    ) -> list[list[list[float]]]:
        """RESCo's ``states.drq_norm`` for one intersection.

        Per incoming lane *i*: ``[1 if i == phase_index else 0,
        approach/28, total_wait/28, queue/28, sum(speed/20/28)]`` over
        the lane's detectable vehicles (sticky-wait bookkeeping), with a
        leading channel dim -> shape ``(1, n_lanes, 5)``.
        """
        sticky = self._episode["sticky_wait"][str(ix.id)]
        detectable_by_lane = self._episode["sticky_detectable"]
        speeds = self._vehicle_speeds()
        rows: list[list[float]] = []
        for i, lid in enumerate(ix.incoming_lanes):
            approach = queue = total_wait = speed_sum = 0.0
            for vid in detectable_by_lane.get(lid, ()):
                wait = float(sticky.get(vid, 0.0))
                if wait > 0.0:
                    queue += 1.0
                    total_wait += wait
                else:
                    approach += 1.0
                speed_sum += float(speeds.get(vid, 0.0)) / 20.0 / 28.0
            rows.append([
                1.0 if i == int(phase_index) else 0.0,
                approach / 28.0,
                total_wait / 28.0,
                queue / 28.0,
                speed_sum,
            ])
        return [rows]

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
        return float(sum(int(c) for c in self._all_lane_halting_count().values()))

    @register("calculate_average_delta_of_delays_after_action")
    def _avg_delta_delay(self) -> float:
        pre = self._episode.get("pre_action", {})
        if not pre:
            return 0.0
        t_post = self._current_time()
        distances = self._vehicle_distances()
        active_set = set(self._vehicle_ids())

        deltas: list[float] = []
        for vid, (t_pre, x_pre, v_max, depart) in pre.items():
            if vid not in active_set:
                continue
            x_post = float(distances.get(vid, 0.0))
            if x_pre <= 0.0 or x_post <= 0.0 or v_max <= 0.0:
                continue
            tau_pre = max(0.0, t_pre - depart)
            tau_post = max(0.0, t_post - depart)
            deltas.append(v_max * (tau_pre / x_pre - tau_post / x_post))
        if not deltas:
            return 0.0
        return float(sum(deltas) / len(deltas))
