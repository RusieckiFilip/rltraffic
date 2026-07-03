"""MOSS-backed implementation of :class:`BaseMetrics`.

MOSS (https://github.com/tsinghua-fib-lab/moss) is a GPU-accelerated
microscopic traffic simulator.  Per-vehicle state lives on the GPU and
is fetched in bulk through ``engine.fetch_persons``.  We treat that bulk
fetch as the canonical per-step primitive and derive every metric from
the resulting numpy arrays, mirroring the layout of
:class:`metrics.cityflow.CityFlowMetrics` and
:class:`metrics.sumo.SumoMetrics` so that downstream code can stay
backend-agnostic.

Notes on a few engine specifics:

- Person ``status`` is an ``uint8`` enum: ``0`` SLEEP, ``1`` WALKING,
  ``2`` DRIVING, ``3`` FINISHED.  Only ``DRIVING`` persons are "active"
  for TSC purposes.
- Lane IDs on persons are MOSS lane *ids* (large int), not lane
  indices.  Our :class:`envs.moss_env.MossEnv` exposes
  intersection ``incoming_lanes`` as ``str(lane_id)`` for parity with
  the other backends, so we convert when filtering counts.
- The MOSS engine does not provide per-vehicle ideal travel time, so we
  approximate it as ``total_distance / global_max_speed`` (a
  best-case lower bound using the fastest lane in the network as the
  reference speed).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from metrics.base import BaseMetrics, register, register_local


# Vehicles slower than this are considered halted / waiting.
HALT_SPEED_THRESHOLD = 0.1

# MOSS person status codes (see moss.engine).
_STATUS_SLEEP = 0
_STATUS_DRIVING = 2
_STATUS_FINISHED = 3


class MossMetrics(BaseMetrics):
    """Compute traffic-signal-control metrics from a ``moss.Engine``.

    Parameters
    ----------
    engine
        Active ``moss.Engine`` instance owned by the env.
    intersections
        ``IntersectionInfo`` list whose ``incoming_lanes`` / ``outgoing_lanes``
        we monitor.
    metric_names
        Names of metrics the user wants computed.
    delta_time
        Number of simulation seconds between consecutive ``update()``
        calls (i.e. ``env.delta_time``).
    global_max_speed
        Largest ``max_speed`` (m/s) across all lanes in the loaded map.
        Used as the conservative reference speed when computing per-trip
        ideal travel times.  ``0.0`` disables delay-vs-ideal metrics.
    """

    def __init__(
        self,
        engine: Any,
        intersections: list[Any],
        metric_names: Iterable[str] | None = None,
        *,
        delta_time: float,
        global_max_speed: float = 0.0,
    ) -> None:
        self._incoming_lane_set: set[str] = {
            lid for ix in intersections for lid in ix.incoming_lanes
        }
        self._monitored_lane_set: set[str] = {
            lid
            for ix in intersections
            for lid in (list(ix.incoming_lanes) + list(ix.outgoing_lanes))
        }
        # MOSS lane ids are integers under the hood; pre-cast for fast
        # membership checks against the numpy arrays returned by
        # ``fetch_persons``.
        self._monitored_lane_ids_int: set[int] = {
            int(lid) for lid in self._monitored_lane_set
        }
        self._incoming_lane_ids_int: set[int] = {
            int(lid) for lid in self._incoming_lane_set
        }
        # int lane id -> ids of intersections it feeds into (normally one).
        self._incoming_lane_ix: dict[int, list[str]] = {}
        for ix in intersections:
            for lid in ix.incoming_lanes:
                self._incoming_lane_ix.setdefault(int(lid), []).append(
                    str(ix.id)
                )
        self._global_max_speed: float = (
            float(global_max_speed) if float(global_max_speed) > 0.0 else 0.0
        )
        super().__init__(
            engine=engine,
            intersections=intersections,
            metric_names=metric_names,
            delta_time=delta_time,
        )

    # ------------------------------------------------------------------
    # Cached per-step primitives
    # ------------------------------------------------------------------

    @property
    def _e(self) -> Any:
        return self._engine

    _PERSON_FIELDS = [
        "id",
        "enable",
        "status",
        "lane_id",
        "v",
        "traveling_time",
        "total_distance",
        "departure_time",
    ]

    def _persons(self) -> dict[str, np.ndarray]:
        return self._cached(
            "persons",
            lambda: self._e.fetch_persons(list(self._PERSON_FIELDS)),
        )

    def _current_time(self) -> float:
        return self._cached(
            "current_time", lambda: float(self._e.get_current_time())
        )

    def _active_mask(self) -> np.ndarray:
        def _fn() -> np.ndarray:
            p = self._persons()
            return (p["enable"] == 1) & (p["status"] == _STATUS_DRIVING)

        return self._cached("active_mask", _fn)

    def _finished_mask(self) -> np.ndarray:
        def _fn() -> np.ndarray:
            p = self._persons()
            return (p["enable"] == 1) & (p["status"] == _STATUS_FINISHED)

        return self._cached("finished_mask", _fn)

    def _lane_counts_for(
        self, mask: np.ndarray, lane_set: set[int]
    ) -> dict[str, int]:
        """Aggregate per-lane counts for the persons selected by *mask*,
        restricted to *lane_set*.  Lanes in *lane_set* with no vehicles
        appear in the result with a zero count."""
        result: dict[str, int] = {str(int(lid)): 0 for lid in lane_set}
        if not lane_set:
            return result
        p = self._persons()
        lane_ids = p["lane_id"][mask]
        if lane_ids.size == 0:
            return result
        unique, counts = np.unique(lane_ids, return_counts=True)
        for u, c in zip(unique, counts):
            ui = int(u)
            if ui in lane_set:
                result[str(ui)] = int(c)
        return result

    def _lane_vehicle_count_cached(self) -> dict[str, int]:
        return self._cached(
            "lane_vehicle_count",
            lambda: self._lane_counts_for(
                self._active_mask(), self._monitored_lane_ids_int
            ),
        )

    def _lane_waiting_count_cached(self) -> dict[str, int]:
        def _fn() -> dict[str, int]:
            p = self._persons()
            mask = self._active_mask() & (p["v"] < HALT_SPEED_THRESHOLD)
            return self._lane_counts_for(mask, self._monitored_lane_ids_int)

        return self._cached("lane_waiting", _fn)

    def lane_vehicle_count(self) -> dict[str, int]:
        return dict(self._lane_vehicle_count_cached())

    def lane_waiting_vehicle_count(self) -> dict[str, int]:
        return dict(self._lane_waiting_count_cached())

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        # vid -> {"depart_time", "wait_acc", "real_tt_last", "dist_last"}
        self._episode["seen"] = {}
        # Finalized journey records (one dict per finished vehicle).
        self._episode["completed"] = []
        # Set of vids we have already finalized this episode.
        self._episode["finalized"] = set()

        # Episode-cumulative waiting integrals (s).
        self._episode["wait_total_episode"] = 0.0
        self._episode["wait_incoming_episode"] = 0.0

        # Pre-action snapshot for the action-delta metric.
        self._episode["pre_action"] = {}

        # Number of vehicles whose journey was finalized in the last step.
        self._episode["throughput_delta"] = 0.0

    def _snapshot_pre_action(self) -> None:
        """Snapshot ``(traveling_time, distance, v_max)`` for every running
        vehicle.

        MOSS exposes per-person ``traveling_time`` (elapsed since departure),
        so the delta metric normalises by that rather than the absolute
        simulation clock.  Distance is stored raw (no floor): vehicles that
        have not moved yet (``x <= 0``) are skipped downstream instead of
        being turned into huge ``t / x`` outliers.
        """
        p = self._persons()
        active = self._active_mask()
        ids = p["id"][active]
        if ids.size == 0:
            self._episode["pre_action"] = {}
            return
        dist = p["total_distance"][active]
        tt = p["traveling_time"][active]
        v_now = p["v"][active]
        v_max_ref = self._global_max_speed
        snapshot: dict[int, tuple[float, float, float]] = {}
        for i in range(ids.size):
            vid = int(ids[i])
            x = float(dist[i])
            if v_max_ref > 0.0:
                v_max = v_max_ref
            else:
                # Fallback when no map-level max_speed is known.
                v_max = max(float(v_now[i]), 1e-3)
            snapshot[vid] = (float(tt[i]), x, v_max)
        self._episode["pre_action"] = snapshot

    def _run_step_hooks(self) -> None:
        p = self._persons()
        t = self._current_time()
        delta_time = self._delta_time

        ids = p["id"]
        statuses = p["status"]
        enable = p["enable"]
        traveling_times = p["traveling_time"]
        total_distances = p["total_distance"]
        velocities = p["v"]
        lane_ids = p["lane_id"]

        seen: dict[int, dict[str, float]] = self._episode["seen"]
        finalized: set = self._episode["finalized"]

        active_mask = (enable == 1) & (statuses == _STATUS_DRIVING)
        finished_mask = (enable == 1) & (statuses == _STATUS_FINISHED)

        # 1. Update bookkeeping for currently-active vehicles.
        active_ids = ids[active_mask]
        active_tt = traveling_times[active_mask]
        active_d = total_distances[active_mask]
        active_v = velocities[active_mask]
        for i in range(active_ids.size):
            vid = int(active_ids[i])
            rec = seen.get(vid)
            if rec is None:
                rec = {
                    "depart_time": t - delta_time / 2.0,
                    "wait_acc": 0.0,
                    "real_tt_last": float(active_tt[i]),
                    "dist_last": float(active_d[i]),
                }
                seen[vid] = rec
            else:
                rec["real_tt_last"] = float(active_tt[i])
                rec["dist_last"] = float(active_d[i])
            if float(active_v[i]) < HALT_SPEED_THRESHOLD:
                rec["wait_acc"] += delta_time

        # 2. Finalize newly-finished vehicles.
        finished_ids = ids[finished_mask]
        finished_tt = traveling_times[finished_mask]
        finished_d = total_distances[finished_mask]
        v_max_ref = self._global_max_speed
        new_finished = 0
        for i in range(finished_ids.size):
            vid = int(finished_ids[i])
            if vid in finalized:
                continue
            real_tt = float(finished_tt[i])
            distance = float(finished_d[i])
            ideal_tt = (
                distance / v_max_ref if (v_max_ref > 0.0 and distance > 0.0) else 0.0
            )
            wait_acc = float(seen.get(vid, {}).get("wait_acc", 0.0))
            # "delay" is the real/ideal slowdown ratio, matching the
            # CityFlow and SUMO backends so the delay metrics stay
            # comparable across simulators.
            self._episode["completed"].append(
                {
                    "vid": vid,
                    "real_tt": real_tt,
                    "ideal_tt": ideal_tt,
                    "delay": real_tt / ideal_tt if ideal_tt > 0.0 else 0.0,
                    "wait_acc": wait_acc,
                }
            )
            finalized.add(vid)
            new_finished += 1
        self._episode["throughput_delta"] = float(new_finished)

        # 3. Episode-cumulative waiting-time integrals.
        if self._requires_any(
            "total_waiting_time_all_vehicles_in_simulation_in_episode",
        ):
            halting_all = active_mask & (velocities < HALT_SPEED_THRESHOLD)
            n_wait_total = int(np.sum(halting_all))
            self._episode["wait_total_episode"] += n_wait_total * delta_time

        if self._requires_any(
            "total_waiting_time_on_the_incoming_lanes_in_episode",
        ):
            halting_mask = active_mask & (velocities < HALT_SPEED_THRESHOLD)
            halting_lane_ids = lane_ids[halting_mask]
            incoming_int = self._incoming_lane_ids_int
            if halting_lane_ids.size > 0 and incoming_int:
                in_arr = np.fromiter(incoming_int, dtype=np.int64)
                hits = np.isin(halting_lane_ids, in_arr)
                n_wait_incoming = int(np.sum(hits))
                self._episode["wait_incoming_episode"] += (
                    n_wait_incoming * delta_time
                )

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    @register("average_travel_time")
    def _average_travel_time(self) -> float:
        """Average over *all* vehicles that entered the network: completed
        trips count their full travel time, vehicles still en route count
        their traveling time so far (no survivorship bias)."""
        completed = self._episode["completed"]
        total = float(sum(c["real_tt"] for c in completed))
        count = len(completed)
        p = self._persons()
        active_tt = p["traveling_time"][self._active_mask()]
        total += float(np.sum(active_tt))
        count += int(active_tt.size)
        if count == 0:
            return 0.0
        return float(total / count)

    @register("throughput_delta")
    def _throughput_delta(self) -> float:
        return float(self._episode.get("throughput_delta", 0.0))

    @register("scheduled_due_vehicle_count")
    def _scheduled_due_vehicle_count(self) -> float:
        p = self._persons()
        started = (p["enable"] == 1) & (p["status"] != _STATUS_SLEEP)
        pending_due = (
            (p["enable"] == 1)
            & (p["status"] == _STATUS_SLEEP)
            & (p["departure_time"] <= self._current_time())
        )
        return float(int(np.sum(started | pending_due)))

    @register("started_vehicle_count")
    def _started_vehicle_count(self) -> float:
        p = self._persons()
        started = (p["enable"] == 1) & (p["status"] != _STATUS_SLEEP)
        return float(int(np.sum(started)))

    @register("running_vehicle_count")
    def _running_vehicle_count(self) -> float:
        return float(int(np.sum(self._active_mask())))

    @register("pending_departure_vehicle_count")
    def _pending_departure_vehicle_count(self) -> float:
        p = self._persons()
        pending = (
            (p["enable"] == 1)
            & (p["status"] == _STATUS_SLEEP)
            & (p["departure_time"] <= self._current_time())
        )
        return float(int(np.sum(pending)))

    @register("not_started_vehicle_count")
    def _not_started_vehicle_count(self) -> float:
        p = self._persons()
        sleeping = (p["enable"] == 1) & (p["status"] == _STATUS_SLEEP)
        return float(int(np.sum(sleeping)))

    @register("teleport_count")
    def _teleport_count(self) -> float:
        return 0.0

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
        # Sum of per-vehicle accumulated waiting time for currently
        # active persons (matches the SUMO/CityFlow notion of "current
        # total accumulated wait time across the live fleet").
        seen = self._episode["seen"]
        p = self._persons()
        ids = p["id"][self._active_mask()]
        total = 0.0
        for vid in ids:
            rec = seen.get(int(vid))
            if rec is not None:
                total += float(rec["wait_acc"])
        return float(total)

    @register_local(
        "waiting_time_all_vehicles_for_the_last_time_step_in_simulation"
    )
    def _wait_time_now_local(self) -> dict[str, float]:
        """Accumulated waiting time of the vehicles currently on each
        intersection's incoming lanes (RESCo's per-junction wait)."""
        seen = self._episode["seen"]
        p = self._persons()
        active = self._active_mask()
        ids = p["id"][active]
        lane_ids = p["lane_id"][active]
        totals: dict[str, float] = {
            str(ix.id): 0.0 for ix in self._intersections
        }
        for i in range(ids.size):
            ix_ids = self._incoming_lane_ix.get(int(lane_ids[i]))
            if not ix_ids:
                continue
            rec = seen.get(int(ids[i]))
            if rec is None:
                continue
            wait = float(rec["wait_acc"])
            for ix_id in ix_ids:
                totals[ix_id] += wait
        return totals

    def drq_norm_observation(
        self, ix: Any, phase_index: int
    ) -> list[list[list[float]]]:
        """MOSS analogue of RESCo's ``states.drq_norm`` for one intersection.

        Mirrors :meth:`metrics.sumo.SumoMetrics.drq_norm_observation` so the
        observation layout stays identical across backends. Per incoming lane
        *i*: ``[1 if i == phase_index else 0, approach/28, total_wait/28,
        queue/28, sum(speed/20/28)]`` over the active vehicles currently on
        that lane, with a leading channel dim -> shape ``(1, n_lanes, 5)``.

        MOSS has no per-vehicle "next traffic light within 200 m" query, so
        every active vehicle on an incoming lane is treated as detectable
        (incoming lanes are the junction's approaches). Per-vehicle waiting
        time is the accumulated halt time tracked in ``_run_step_hooks``;
        a vehicle counts toward the queue once it has waited (>0 s), else it
        counts toward the approach.
        """
        p = self._persons()
        active = self._active_mask()
        ids = p["id"][active]
        lane_ids = p["lane_id"][active]
        speeds = p["v"][active]
        seen = self._episode["seen"]

        # Group active vehicles by the lane they occupy.
        by_lane: dict[int, list[tuple[int, float]]] = {}
        for i in range(ids.size):
            by_lane.setdefault(int(lane_ids[i]), []).append(
                (int(ids[i]), float(speeds[i]))
            )

        rows: list[list[float]] = []
        for i, lid in enumerate(ix.incoming_lanes):
            approach = queue = total_wait = speed_sum = 0.0
            for vid, speed in by_lane.get(int(lid), ()):
                wait = float(seen.get(vid, {}).get("wait_acc", 0.0))
                if wait > 0.0:
                    queue += 1.0
                    total_wait += wait
                else:
                    approach += 1.0
                speed_sum += speed / 20.0 / 28.0
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
        p = self._persons()
        mask = self._active_mask() & (p["v"] < HALT_SPEED_THRESHOLD)
        return float(int(np.sum(mask)))

    @register("calculate_average_delta_of_delays_after_action")
    def _avg_delta_delay(self) -> float:
        pre = self._episode.get("pre_action", {})
        if not pre:
            return 0.0
        p = self._persons()
        active = self._active_mask()
        ids_post = p["id"][active]
        dist_post = p["total_distance"][active]
        tt_post = p["traveling_time"][active]
        dist_lookup: dict[int, float] = {
            int(i): float(d) for i, d in zip(ids_post, dist_post)
        }
        tt_lookup: dict[int, float] = {
            int(i): float(tt) for i, tt in zip(ids_post, tt_post)
        }
        active_set = set(dist_lookup.keys())

        deltas: list[float] = []
        for vid, (tau_pre, x_pre, v_max) in pre.items():
            if vid not in active_set:
                continue
            x_post = dist_lookup.get(vid, 0.0)
            if x_pre <= 0.0 or x_post <= 0.0 or v_max <= 0.0:
                continue
            tau_post = tt_lookup.get(vid, 0.0)
            deltas.append(v_max * (tau_pre / x_pre - tau_post / x_post))
        if not deltas:
            return 0.0
        return float(sum(deltas) / len(deltas))
