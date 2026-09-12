"""P7.1 half A: a per-second SUMO reference for average travel time, and the metric freeze.

Artifact format version: ``p7.1-metric-freeze/1.0``.
Written against ``BRIEF_34`` + Amendment A, and ``docs/plans/p7.1.md``.

WHAT THIS MODULE DECIDES, AND WHAT IT DOES NOT
----------------------------------------------
``PREREGISTRATION`` A11's ``Rule R`` is a per-scenario gate on CityFlow and has no SUMO counterpart.
This module **measures**: it builds SUMO's twin of every CityFlow reconstruction
``offline/engine_att_reference.py`` builds, so the two backends can be read under one definition with
the other beside it.  ⛔ **It issues no verdict on which definition is primary on SUMO**
(``BRIEF_34`` section 2.3); ``docs/notes/P7.1_FREEZE.md`` proposes and the coordinator amends.

WHAT SUMO'S OWN ``average_travel_time`` IS, READ FROM SOURCE
------------------------------------------------------------
``metrics/sumo.py:459-477`` averages over completed trips plus the vehicles present at the horizon::

    att = (SUM_completed real_tt + SUM_present (t - depart)) / (n_completed + n_present)

with ``depart`` from ``traci.vehicle.getDeparture`` (``:235-242``) -- the **actual** insertion time,
not the route file's intended one -- and arrivals timestamped per simulated second by ``on_sim_step``
(``:244-253``).  **Never-inserted vehicles are invisible to it.**  So it is the twin of CityFlow's
``W_running`` (admitted population, admission clock), and neither of ``att_engine`` nor ``att_ours``.

⚠️ **AND IT HAS A CADENCE OF ITS OWN, WHICH THE BRIEF ASSUMED AWAY AND THIS MODULE MEASURES.**
``depart_time[vid]`` is filled inside ``update()`` (``metrics/sumo.py:422-426``), which runs once per
**decision step**, not once per second.  A vehicle that departs *and* arrives strictly inside one
``delta_time`` window is never snapshotted, takes the ``seen is None`` branch (``:389-394``) and is
credited ``real_tt = 0.0`` **while still incrementing the denominator**.  A vehicle removed without
arriving (a teleport that cannot be reinserted) leaves the average entirely.  ``BRIEF_34`` Amendment
A5 rules: **measure it with two counters; a non-zero count is SUMO's third term and a finding, never
a fix** -- ``metrics/`` is frozen.

THE ALIGNMENT CONVENTION -- THE ONE THING TO READ BEFORE TRUSTING A NUMBER HERE
------------------------------------------------------------------------------
A snapshot is taken **immediately after each** ``traci.simulationStep()`` and is labelled with the
engine's own ``simulation.getTime()``.  Every per-step list traci exposes -- ``getDepartedIDList``,
``getArrivedIDList``, ``getStartingTeleportIDList`` -- describes **the step just executed**, so a
vehicle whose id appears in the snapshot labelled ``t`` departed (or arrived) during ``(t - dt, t]``
and is credited at ``t``.

🔒 **THE ONE-STEP OFFSET, MEASURED AND NOT ASSUMED -- the SUMO twin of Gate 0's
``first_seen(v) = enterTime(v) + interval``.**  A vehicle is inserted during the step that *ends* at
``intended_depart + dt``, so::

    observed_departure_second(v) == getDeparture(v) + dt                            (S)

Measured on the parity scenario over 600 simulated seconds and 307 departures: ``observed -
getDeparture`` took **exactly one distinct value, 1.0**, and the pool identity below held at every
one of those 600 seconds under ``intended + dt <= t`` while failing at ``t = 3`` under the naive
``intended <= t``.  ⭐ **The identity found that off-by-one on the first real episode rather than
letting it shift every ``E_sumo`` contribution by one second**, which is what the check exists for.
``dt`` is read from ``simulation.getDeltaT()`` -- the simulator's own value -- and cross-checked
against the observed grid, never taken from a config.

Three clocks are recorded for every departure and (S) is enforced as an invariant:

* ``departed_at(v)``  -- the second at which ``v`` first appeared in ``getDepartedIDList``;
* ``actual_depart(v)`` -- ``traci.vehicle.getDeparture(v)``, queried at that same second;
* ``intended_depart(v)`` -- the route file's ``<vehicle depart=...>``, parsed with stdlib XML.

⚠️ **The reconstructions mix a start-of-step clock with an end-of-step one, deliberately, because
the frozen metric does.**  ``metrics/sumo.py`` credits ``t_arrival - getDeparture(v)`` with the
arrival timestamped by ``on_sim_step`` (end of step) and the departure from ``getDeparture`` (start
of step).  ``W_sumo`` therefore uses exactly that pair, which is what makes it comparable to
``att_sumo_env`` at all; ``E_sumo`` and ``P_sumo`` put the route file's ``depart`` in the same
start-of-step slot.  Recomputing either on a single convention would produce a tidier number that
no longer twins the quantity under test.

FOUR RECONSTRUCTIONS, NAMED TO READ ACROSS TO GATE 0's
-------------------------------------------------------
With ``T`` the horizon and ``arrival(v)`` the second ``v`` appeared in ``getArrivedIDList``:

===================  ==========================================  =================  ==================================
name                 population                                  clock origin       contribution
===================  ==========================================  =================  ==================================
``E_sumo``           every ``v`` with ``intended(v) <= T``        intended           ``(arrival(v) or T) - intended(v)``
``P_sumo``           every ``v`` departed by ``T``                intended           ``(arrival(v) or T) - intended(v)``
``W_sumo``           every ``v`` departed by ``T``                **actual**         ``(arrival(v) or T) - actual(v)``
``att_sumo_env``     the env's own metric at the horizon          --                 read, never rebuilt
===================  ==========================================  =================  ==================================

``E_sumo`` is the twin of ``att_engine``: a pool clock over every vehicle the demand file created,
censored at the horizon exactly as CityFlow's pool term censors a vehicle still in ``vehiclePool``.
⚠️ **A vehicle that departed and then vanished without arriving is censored at ``T`` here and is
ABSENT from the env's average.** That is a definitional difference, not an error; it is counted as
``n_vanished_without_arrival`` so a reader can bound it.

The three terms, mirroring ``GateEpisode``'s on the CityFlow side::

    term_population   = P_sumo - E_sumo         (the never-inserted population)
    term_clock_origin = W_sumo - P_sumo         (minus the mean insertion delay)
    term_cadence      = att_sumo_env - W_sumo   (A5: measured, not assumed zero)
    att_sumo_env - E_sumo = population + clock_origin + cadence         (exact, by algebra)

**The second route, required by ``CLAUDE.md`` section 2:** ``term_clock_origin`` must equal
``-mean(getDepartDelay(v))`` over the departed population, reached by a different summation order
(per-vehicle delays averaged, against a difference of two averages of contributions).  **This is the
check the artifact GATES on**, to :data:`CLOCK_ORIGIN_TOLERANCE`.

🚨 **WHY THE DECOMPOSITION RESIDUAL IS REPORTED AND NEVER GATED ON, MEASURED RATHER THAN ARGUED.**
``(P-E) + (W-P) + (att-W) - (att-E)`` cancels ``P`` and ``W`` identically, so **it is an algebraic
tautology over the four numbers and cannot detect a wrong reconstruction at all.**  Measured in
float64 while writing this module: moving ``P`` by ``1e-9`` leaves the residual at exactly ``0.0``,
moving ``P`` by **100** leaves it at ``4.44e-15`` -- *non-zero for a purely numerical reason while
blind to a semantic error of any size*.  Gating on ``== 0.0`` would therefore have bought no
discriminating power and would have produced spurious refusals mid-campaign.  The residual stays in
every record because A13(b) requires the decomposition to be reported and because it documents the
arithmetic; the **power** lives in the second route above and in the reproduction against
``output/p7_0/``.  ⚠️ This is `P5.3b-fix` section 2.7's class recurring (*"decomposition_residual is
algebraically deviation_c1"*), and on SUMO it degenerates further because there is no second engine
reference to anchor an endpoint.

THE POOL IDENTITY
-----------------
``BRIEF_34`` section 2.1 asks for ``departed | pending | not-yet-due == the file's id set`` every
second.  Taken with an instantaneous ``departed`` that is false the moment a vehicle arrives, so the
sound form -- implemented and tested here, and strictly stronger -- is::

    cumulative_departed(<= t)  |  pending(t)  ==  {v : intended(v) + dt <= t}   (disjointly)

which implies the brief's three-way union.  It is checked at **every** simulated second and raises
naming the offending ids, because a silent hole in it would mean ``E_sumo``'s denominator is not the
demand file's population.

THE IMPORT FENCE
----------------
:data:`RECONSTRUCTION_SURFACE` (Layer A) imports nothing from ``metrics/``, mirroring A11's
independence clause: the halting cross-check and ``att_sumo_env`` exist to be compared against the
frozen metric, and a reconstruction that imported it would be checking the metric against itself.
:data:`HALT_SPEED_THRESHOLD` is therefore **re-declared** here rather than imported, and
``tests/test_sumo_att_reference.py`` asserts it equals ``metrics.sumo.HALT_SPEED_THRESHOLD`` -- an
independent declaration plus an executed equality check, which is stronger than an import.
⚠️ **Disclosed rather than hidden:** the env subclass inherits ``SumoEnv._create_metrics`` and its
``_simulate`` **must** keep driving ``metrics.on_sim_step()``; without it ``_drain_arrivals``
(``metrics/sumo.py:255-267``) sees only the last step's arrivals and ``att_sumo_env`` silently
becomes a different quantity.

THE FILESYSTEM-MUTATION BARRIER AND THE FENCE
---------------------------------------------
``output/`` in the main tree is gitignored, is the only copy, and holds twelve manifests.
:func:`assert_metric_freeze_writable` is **default-deny on whole path components**: only ``p7_1`` and
``SHA256SUMS_p7_1.txt`` are writable, and **``p7_0`` in particular is READ by this task and must never
be written by it**.  Every destination is validated before the first directory is created, and
:func:`freeze_artifact` validates all thirty-one episodes before a byte is written.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ALLOWED_P7_1_OUTPUT_ENTRIES",
    "ANCHOR_ARMS",
    "BASE_SEED",
    "DEFAULT_WORK_DIRNAME",
    "EPISODES_PER_ARM",
    "EXPECTED_CAMPAIGN_LABELS",
    "HALT_SPEED_THRESHOLD",
    "P7_0_CELL_DIRS",
    "RECONSTRUCTION_SURFACE",
    "SUMO_REGIMES",
    "TIMING_ROLE",
    "FreezeEpisode",
    "HaltingAgreement",
    "IntendedDepartures",
    "SumoAtt",
    "SumoEpisodeReconstruction",
    "SumoObservationRecorder",
    "assert_campaign_complete",
    "assert_metric_freeze_writable",
    "build_parser",
    "build_policy",
    "chunk_is_reusable",
    "chunk_name",
    "collect_style_args",
    "failed_chunk_destination",
    "freeze_artifact",
    "main",
    "make_observer_sumo_env",
    "p7_0_horizon_att",
    "read_intended_departures",
    "reconstruct_sumo_episode",
    "reproduction_report",
    "reusable_chunk_at",
    "rho_table",
    "route_files_of",
    "row_reproduces_p7_0",
    "run_cityflow_arm",
    "run_sumo_arm",
    "sumo_observer_env_class",
]

#: The on-disk format of every chunk and of ``docs/data/p7_1_metric_freeze.json``.
ARTIFACT_FORMAT_VERSION = "p7.1-metric-freeze/1.0"

#: DEFAULT-DENY.  The **only** two entries under ``output/`` this task may write.  ⚠️ ``p7_0`` is
#: deliberately absent: this campaign reads P7.0's episodes and must never write to them.
ALLOWED_P7_1_OUTPUT_ENTRIES: tuple[str, ...] = ("p7_1", "SHA256SUMS_p7_1.txt")

DEFAULT_WORK_DIRNAME = "p7_1"

#: A4's chunk carries this instead of being told apart by its identity fields: a timing episode and
#: an A1 `random` episode share backend, arm, observer and regime, and differ only in the scenario.
TIMING_ROLE = "timing_only"

#: The tolerance of the clock-origin second route (``BRIEF_34`` section 2.1's ``1e-9``).  This is
#: the decomposition's ONE gating check: unlike the residual it compares two independently recorded
#: quantities -- a difference of averaged contributions against traci's own per-vehicle
#: ``getDepartDelay`` readings -- so it can actually fail.
CLOCK_ORIGIN_TOLERANCE = 1e-9

#: SUMO's halting definition, re-declared rather than imported from the frozen ``metrics/sumo.py``
#: so the cross-check is independent (see the module docstring's import-fence section).  A test
#: asserts it equals ``metrics.sumo.HALT_SPEED_THRESHOLD``.
HALT_SPEED_THRESHOLD = 0.1

#: P7.0's three anchors, in the order ``transfer_gate.CELLS`` declares them per backend.
ANCHOR_ARMS: tuple[str, ...] = ("fixedtime", "maxpressure", "random")

#: The twelve cells ``report`` requires, derived from :data:`ANCHOR_ARMS` so the list cannot drift
#: from the arms it is built out of: 3 arms x {parity observed, parity unobserved, noteleport} on
#: SUMO, plus the 3 CityFlow arms.  ⚠️ A4's ``timing_hz4x4_gudang.json`` is deliberately absent --
#: it is not a ``freeze_*.json`` cell, its ATT is not a result, and it is not part of any table.
EXPECTED_CAMPAIGN_LABELS: tuple[str, ...] = tuple(
    sorted(
        [f"sumo__{arm}" for arm in ANCHOR_ARMS]
        + [f"sumo__{arm}__unobserved" for arm in ANCHOR_ARMS]
        + [f"sumo_noteleport__{arm}" for arm in ANCHOR_ARMS]
        + [f"cityflow__{arm}" for arm in ANCHOR_ARMS]
    )
)

#: The two SUMO regimes (``BRIEF_34`` Amendment D2).  ``parity`` is P7.0's own configuration and is
#: the one the reproduction check is defined against -- it runs with SUMO's default
#: ``--time-to-teleport 300`` and is labelled *teleports enabled*.  ``noteleport`` is the candidate
#: frozen regime: the same network and the same parity route file with teleporting disabled.
#: ⚠️ **Selecting a regime selects its assertion too** (:func:`run_sumo_arm`), so there is no way to
#: run the teleport-free regime without checking that it was in fact teleport-free.
SUMO_REGIMES: tuple[str, ...] = ("parity", "noteleport")

#: P7.0's collection seed and episode count (``transfer_gate.main``'s defaults, and what
#: ``output/p7_0/*/manifest.json`` records).  Episode ``i`` used ``reset(seed=BASE_SEED + i)``.
BASE_SEED = 1000
EPISODES_PER_ARM = 5

#: Where P7.0's episodes live under ``output/``, per backend and arm.
P7_0_CELL_DIRS: Mapping[str, str] = {
    "cityflow__fixedtime": "p7_0/cityflow__fixedtime",
    "cityflow__maxpressure": "p7_0/cityflow__maxpressure",
    "cityflow__random": "p7_0/cityflow__random",
    "sumo__fixedtime": "p7_0/sumo__fixedtime",
    "sumo__maxpressure": "p7_0/sumo__maxpressure",
    "sumo__random": "p7_0/sumo__random",
}

#: Memoised by :func:`sumo_observer_env_class`, which builds the subclass on first use so that
#: importing this module never requires traci or a SUMO installation.
_OBSERVER_ENV_CLASS: type[Any] | None = None

#: LAYER A -- the reconstruction.  These names may import stdlib, numpy, ``envs`` and
#: ``experiments`` only; never ``metrics/``.  The allowlist is data so a test can walk it.
RECONSTRUCTION_SURFACE: tuple[str, ...] = (
    "IntendedDepartures",
    "SumoAtt",
    "HaltingAgreement",
    "SumoEpisodeReconstruction",
    "SumoObservationRecorder",
    "read_intended_departures",
    "route_files_of",
    "reconstruct_sumo_episode",
)


# ----------------------------------------------------------------------
# LAYER A -- the reconstruction
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IntendedDepartures:
    """The demand file's own view of the episode: every vehicle id and when it was due.

    ``by_id`` is the SUMO twin of ``admission_probe.created_from_flow``'s population -- the vehicles
    the demand created, whether or not the simulator ever inserted them.
    """

    by_id: Mapping[str, float]
    source: str

    @property
    def n(self) -> int:
        """How many vehicles the demand file declares."""
        return len(self.by_id)

    def due_by(self, horizon: float, *, step_length: float = 0.0) -> frozenset[str]:
        """Every id the simulator could have inserted at or before *horizon*.

        With *step_length* the measured one-step offset (S) applies: a vehicle declared at
        ``depart = h`` cannot be inserted before ``h + dt``, so it is not in the population a
        horizon of ``h`` averages over.  The default of ``0.0`` gives the literal file reading.
        """
        limit = float(horizon) - float(step_length)
        return frozenset(vid for vid, due in self.by_id.items() if float(due) <= limit)


@dataclass(frozen=True)
class SumoAtt:
    """One reconstruction's value, with the sum and count it came from.

    ``total`` and ``n_ids`` are carried so a caller can re-derive ``value`` by a second route
    instead of trusting it, which is what the tests do.
    """

    value: float
    total: float
    n_ids: int


@dataclass(frozen=True)
class HaltingAgreement:
    """Whether SUMO's own halting count matches counting ``speed < 0.1`` ourselves.

    P7.0 left this "documented, not measured" (``docs/returns/P7.0.md`` section 12).  ``n_missing``
    counts lane-seconds where a listed vehicle had no speed available (it arrived during the step),
    which are excluded from the comparison rather than silently counted as moving.
    """

    n_seconds: int
    n_lane_seconds: int
    max_abs_difference: int
    n_disagreeing_lane_seconds: int
    n_missing: int


@dataclass(frozen=True)
class SumoEpisodeReconstruction:
    """Every reconstruction of one SUMO episode, with the counters the freeze reports."""

    e_sumo: SumoAtt
    p_sumo: SumoAtt
    w_sumo: SumoAtt
    mean_depart_delay: float
    horizon: float
    n_observations: int
    n_intended: int
    n_departed: int
    n_arrived: int
    n_never_inserted: int
    n_pending_at_horizon: int
    n_teleports: int
    n_vanished_without_arrival: int
    n_arrived_never_observed_at_a_boundary: int
    max_abs_depart_clock_deviation: float
    halting: HaltingAgreement


def read_intended_departures(route_file: str | Path) -> IntendedDepartures:
    """Parse ``<vehicle id= depart=>`` out of a SUMO route file with stdlib XML.

    The SUMO twin of ``created_from_flow``: this is the population ``E_sumo``'s denominator is, so
    it is read from the demand file and never from the simulator.  Refuses a duplicate id, a
    non-numeric ``depart`` (``"triggered"``), and the presence of ``<flow>`` or ``<trip>``, whose
    vehicle ids are generated by the simulator and cannot be enumerated from the file.
    """
    path = Path(route_file)
    root = ET.parse(path).getroot()
    generated = [element.tag for element in root.iter() if element.tag in {"flow", "trip"}]
    if generated:
        raise ValueError(
            f"{path} declares {len(generated)} <{generated[0]}> element(s); their vehicle ids are "
            "minted by the simulator at run time, so this file cannot enumerate the population "
            "E_sumo averages over. Expand them to <vehicle> elements or measure a different "
            "scenario -- do not guess the ids"
        )
    by_id: dict[str, float] = {}
    for element in root.iter("vehicle"):
        vid = element.get("id")
        if vid is None:
            raise ValueError(f"{path} holds a <vehicle> with no id; it cannot be counted")
        if vid in by_id:
            raise ValueError(
                f"{path} declares vehicle id {vid!r} twice; a duplicate id would silently shrink "
                "E_sumo's denominator by one and shorten one vehicle's clock"
            )
        raw = element.get("depart")
        try:
            depart = float(str(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} gives vehicle {vid!r} depart={raw!r}, which is not numeric; a vehicle "
                "with no intended departure time (for example 'triggered') has no pool clock, so "
                "E_sumo is undefined for it"
            ) from exc
        by_id[vid] = depart
    if not by_id:
        raise ValueError(f"{path} declares no <vehicle> elements, so it carries no demand")
    return IntendedDepartures(by_id=by_id, source=str(path))


def route_files_of(sumocfg_path: str | Path) -> tuple[Path, ...]:
    """The route files a ``.sumocfg`` declares, resolved against its own directory.

    Refuses a config that declares none: an episode whose demand cannot be read has no ``E_sumo``.
    """
    path = Path(sumocfg_path)
    root = ET.parse(path).getroot()
    declared: list[Path] = []
    for element in root.iter():
        if element.tag not in {"route-files", "route-file"}:
            continue
        value = str(element.get("value") or "")
        for name in (part for chunk in value.split(",") for part in chunk.split()):
            declared.append((path.parent / name).resolve())
    if not declared:
        raise ValueError(
            f"{path} declares no route file, so the demand this episode was asked to run cannot "
            "be read and E_sumo has no population"
        )
    return tuple(declared)


class SumoObservationRecorder:
    """The per-second observation state, kept OUTSIDE the env so it is testable without SUMO.

    ⚠️ **The grain is the whole point.**  ``delta_time`` is 10 s, and the env metric's own
    population bookkeeping runs on that grid; this recorder runs on the simulator's second, which is
    what makes the cadence term measurable instead of assumed.

    Times are the engine's own ``simulation.getTime()`` values, passed in by the caller, so a
    disagreement between our model of the clock and the simulator's surfaces rather than being
    assumed away.
    """

    def __init__(
        self,
        *,
        delta_time: float,
        intended_departures: IntendedDepartures,
        step_length: float = 1.0,
    ) -> None:
        window = float(delta_time)
        if not window > 0.0:
            raise ValueError(
                f"a recorder needs a positive window, got delta_time={delta_time!r}; it selects "
                "the decision boundaries the env metric's own bookkeeping runs on"
            )
        step = float(step_length)
        if not step > 0.0:
            raise ValueError(
                f"a recorder needs a positive simulation step, got step_length={step_length!r}; "
                "it is the measured offset (S) between a vehicle's intended departure and the "
                "second the simulator reports it"
            )
        self.delta_time = window
        self.step_length = step
        self.intended = intended_departures
        # Sorted once so the due set advances in O(n) over the whole episode rather than being
        # rebuilt per second: 3,600 seconds x 2,021 vehicles is not a per-second scan worth paying.
        self._due_order: list[tuple[float, str]] = sorted(
            (float(due), vid) for vid, due in intended_departures.by_id.items()
        )
        self.clear()

    def clear(self) -> None:
        """Drop every observation, so one recorder can serve consecutive episodes."""
        self._times: list[float] = []
        self._due_index = 0
        # The pool identity's left-hand side, maintained incrementally: due, not yet departed.
        self._due_not_departed: set[str] = set()
        self._departed_at: dict[str, float] = {}
        self._actual_depart: dict[str, float] = {}
        self._depart_delay: dict[str, float] = {}
        self._arrived_at: dict[str, float] = {}
        self._present_last: frozenset[str] = frozenset()
        self._pending_last: frozenset[str] = frozenset()
        self._seen_at_boundary: set[str] = set()
        self._arrived_never_at_boundary: set[str] = set()
        self._teleport_ids: set[str] = set()
        self._n_teleport_events = 0
        self._max_depart_clock_deviation = 0.0
        self._halting_seconds = 0
        self._halting_lane_seconds = 0
        self._halting_max_abs = 0
        self._halting_disagreeing = 0
        self._halting_missing = 0

    def observe(
        self,
        *,
        sim_time: float,
        present_ids: Sequence[str],
        departed_ids: Sequence[str],
        arrived_ids: Sequence[str],
        pending_ids: Sequence[str],
        teleport_start_ids: Sequence[str] = (),
        departure_facts: Mapping[str, tuple[float, float]] | None = None,
        halting: Mapping[str, tuple[int, int, int]] | None = None,
    ) -> None:
        """Record one snapshot, taken immediately after a ``simulationStep()``.

        *departure_facts* carries ``{vid: (getDeparture(vid), getDepartDelay(vid))}`` for the ids in
        *departed_ids*; the recorder never calls traci itself.  *halting* carries
        ``{lane: (sumo_halting_number, our_count, n_missing_speed)}``.

        Refuses, each because it would make a plausible number out of a broken stream: a
        non-increasing *sim_time*; a departed id absent from the route file; a second departure of
        the same id; an arrival of an id that never departed; a second arrival; and any second at
        which the pool identity does not hold.
        """
        now = float(sim_time)
        if self._times and now <= self._times[-1]:
            raise ValueError(
                f"simulated time did not advance: this observation is at {now!r} and the previous "
                f"one was at {self._times[-1]!r}. Every snapshot follows a simulationStep(), so a "
                "repeated or receding time means the observer and the simulator are out of step"
            )
        if self._times:
            gap = now - self._times[-1]
            if abs(gap - self.step_length) > 1e-9:
                raise ValueError(
                    f"the observation grid is not uniform: this snapshot is {gap!r} after the "
                    f"previous one but the simulation step is {self.step_length!r}. The one-step "
                    "offset (S) between an intended departure and its reported second is derived "
                    "from that step, so a ragged grid would shift the population by a second"
                )
        self._times.append(now)

        # 1. Everything the simulator could have inserted by `now` joins the pool identity's
        #    left-hand side: a vehicle declared at `d` is inserted during the step ENDING at
        #    `d + dt` (measured -- see the module docstring's (S)).  The pointer never rewinds, so
        #    this is O(n) over the episode.
        due_limit = now - self.step_length + 1e-9
        while self._due_index < len(self._due_order) and self._due_order[self._due_index][0] <= due_limit:
            self._due_not_departed.add(self._due_order[self._due_index][1])
            self._due_index += 1

        # 2. Departures.
        facts = dict(departure_facts or {})
        for vid in departed_ids:
            if vid not in self.intended.by_id:
                raise ValueError(
                    f"{vid!r} departed at {now!r} but is not declared in the route file "
                    f"({self.intended.source}); a vehicle with no intended departure has no pool "
                    "clock, so E_sumo cannot account for it"
                )
            if vid in self._departed_at:
                raise ValueError(
                    f"{vid!r} already departed at {self._departed_at[vid]!r} and departed again at "
                    f"{now!r}; a second departure would overwrite its clock and shorten its "
                    "travel time"
                )
            self._departed_at[vid] = now
            self._due_not_departed.discard(vid)
            if vid in facts:
                actual, delay = facts[vid]
                self._actual_depart[vid] = float(actual)
                self._depart_delay[vid] = float(delay)
                # Invariant (S): the reported second is exactly one simulation step after the
                # simulator's own departure time.  Recorded as a deviation from that measured
                # relation, so it is 0.0 while the convention holds and non-zero the moment it
                # does not -- rather than a constant 1.0 that says nothing.
                self._max_depart_clock_deviation = max(
                    self._max_depart_clock_deviation,
                    abs((float(actual) + self.step_length) - now),
                )

        # 3. Arrivals.
        for vid in arrived_ids:
            if vid not in self._departed_at:
                raise ValueError(
                    f"{vid!r} arrived at {now!r} without ever being reported as departed; an "
                    "arrival is credited against a departure clock and there is none"
                )
            if vid in self._arrived_at:
                raise ValueError(
                    f"{vid!r} already arrived at {self._arrived_at[vid]!r} and arrived again at "
                    f"{now!r}; the second credit would be counted twice"
                )
            self._arrived_at[vid] = now
            if vid not in self._seen_at_boundary:
                self._arrived_never_at_boundary.add(vid)

        # 4. The pool identity, checked EVERY second.  `due_not_departed == pending` is equivalent
        #    to BRIEF_34 section 2.1's three-way union and is strictly stronger: it also asserts the
        #    two sets are disjoint and that nothing pending is undue.
        pending = frozenset(pending_ids)
        if pending != self._due_not_departed:
            unaccounted = sorted(self._due_not_departed - pending)
            unexpected = sorted(pending - self._due_not_departed)
            raise ValueError(
                f"the pool identity fails at t={now!r}: {len(unaccounted)} vehicle(s) are due and "
                f"have not departed yet are not pending ({unaccounted[:5]}), and {len(unexpected)} "
                f"pending vehicle(s) are not due or already departed ({unexpected[:5]}). "
                "E_sumo's denominator is the demand file's population, so a vehicle that falls out "
                "of this accounting is a vehicle the average silently drops"
            )
        self._pending_last = pending

        # 5. Presence, the decision boundaries and the teleport counter.
        self._present_last = frozenset(present_ids)
        if now % self.delta_time == 0.0:
            self._seen_at_boundary.update(self._present_last)
        for vid in teleport_start_ids:
            self._teleport_ids.add(vid)
            self._n_teleport_events += 1

        # 6. The halting cross-check, when the caller sampled it this second.
        if halting is not None:
            self._halting_seconds += 1
            for sumo_count, our_count, missing in halting.values():
                self._halting_lane_seconds += 1
                difference = abs(int(sumo_count) - int(our_count))
                self._halting_max_abs = max(self._halting_max_abs, difference)
                self._halting_disagreeing += 1 if difference else 0
                self._halting_missing += int(missing)

    @property
    def observation_times(self) -> tuple[float, ...]:
        """Every simulated second at which a snapshot was taken, in order."""
        return tuple(self._times)

    @property
    def departed_at(self) -> Mapping[str, float]:
        """``vid -> the second the simulator reported it departed``."""
        return dict(self._departed_at)

    @property
    def actual_depart(self) -> Mapping[str, float]:
        """``vid -> traci.vehicle.getDeparture(vid)``, read at the second it departed."""
        return dict(self._actual_depart)

    @property
    def depart_delay(self) -> Mapping[str, float]:
        """``vid -> traci.vehicle.getDepartDelay(vid)``, read at the second it departed."""
        return dict(self._depart_delay)

    @property
    def arrived_at(self) -> Mapping[str, float]:
        """``vid -> the second the simulator reported it arrived``."""
        return dict(self._arrived_at)

    @property
    def pending_at_horizon(self) -> frozenset[str]:
        """The ids still waiting for insertion at the last observation."""
        return self._pending_last

    @property
    def present_at_horizon(self) -> frozenset[str]:
        """The ids in the network at the last observation."""
        return self._present_last

    @property
    def teleport_ids(self) -> frozenset[str]:
        """Every id that began a teleport during the episode."""
        return frozenset(self._teleport_ids)

    @property
    def n_teleport_events(self) -> int:
        """How many teleports began, counting a vehicle that teleports twice twice."""
        return self._n_teleport_events

    @property
    def max_abs_depart_clock_deviation(self) -> float:
        """``max |(getDeparture(v) + dt) - the second v was reported departed|``.

        The deviation from the MEASURED invariant (S), not the raw gap between the two clocks: the
        gap is a constant ``dt`` by construction and would say nothing, while a non-zero deviation
        means the simulator reported a departure on a schedule this instrument does not model.
        """
        return self._max_depart_clock_deviation

    @property
    def halting_agreement(self) -> HaltingAgreement:
        """The halting-threshold cross-check accumulated so far."""
        return HaltingAgreement(
            n_seconds=self._halting_seconds,
            n_lane_seconds=self._halting_lane_seconds,
            max_abs_difference=self._halting_max_abs,
            n_disagreeing_lane_seconds=self._halting_disagreeing,
            n_missing=self._halting_missing,
        )

    @property
    def n_arrived_never_observed_at_a_boundary(self) -> int:
        """Arrivals of vehicles never present at a decision boundary -- A5's first counter.

        These are exactly the vehicles ``metrics/sumo.py`` credits ``real_tt = 0.0`` while counting
        them in its denominator, so this is the size of the population that can make
        ``att_sumo_env`` differ from ``W_sumo``.
        """
        return len(self._arrived_never_at_boundary)

    @property
    def n_vanished_without_arrival(self) -> int:
        """Departed, absent at the horizon, never reported arrived -- A5's second counter."""
        return len(
            set(self._departed_at) - set(self._arrived_at) - set(self._present_last)
        )


def reconstruct_sumo_episode(
    recorder: SumoObservationRecorder, *, horizon: float | None = None
) -> SumoEpisodeReconstruction:
    """All three reconstructions of the episode *recorder* has just observed.

    *horizon* defaults to the last observation time.  Contributions are censored there: a vehicle
    that has not arrived contributes ``horizon - its clock origin``, which is what CityFlow's pool
    term does for a vehicle still in ``vehiclePool``.
    """
    times = recorder.observation_times
    if not times:
        raise ValueError(
            "nothing was observed, so there is no episode to reconstruct; an empty stream must "
            "refuse rather than report a vacuous 0.0"
        )
    end = float(times[-1]) if horizon is None else float(horizon)

    intended = recorder.intended.by_id
    due = sorted(recorder.intended.due_by(end, step_length=recorder.step_length))
    departed_at = recorder.departed_at
    actual_depart = recorder.actual_depart
    delays = recorder.depart_delay
    arrived_at = recorder.arrived_at
    departed_by_horizon = sorted(vid for vid, when in departed_at.items() if when <= end)

    def _leaves(vid: str) -> float:
        """When the vehicle's clock stops: its arrival, or the horizon if it never arrived."""
        arrival = arrived_at.get(vid)
        return end if arrival is None or arrival > end else float(arrival)

    def _att(population: Sequence[str], origin: Mapping[str, float], label: str) -> SumoAtt:
        total = 0.0
        for vid in population:
            if vid not in origin:
                raise ValueError(
                    f"{label}: {vid!r} has no clock origin recorded, so its contribution cannot be "
                    "computed; a missing origin must refuse rather than default to zero"
                )
            span = _leaves(vid) - float(origin[vid])
            if span < 0.0:
                raise ValueError(
                    f"{label}: {vid!r} left at {_leaves(vid)!r} before its clock started at "
                    f"{origin[vid]!r}; time is monotonic, so a sound stream cannot produce this"
                )
            total += span
        count = len(population)
        if count == 0:
            return SumoAtt(value=0.0, total=0.0, n_ids=0)
        return SumoAtt(value=total / count, total=total, n_ids=count)

    e_sumo = _att(due, intended, "E_sumo")
    p_sumo = _att(departed_by_horizon, intended, "P_sumo")
    w_sumo = _att(departed_by_horizon, actual_depart, "W_sumo")

    missing_delay = [vid for vid in departed_by_horizon if vid not in delays]
    if missing_delay:
        raise ValueError(
            f"{len(missing_delay)} departed vehicle(s) carry no getDepartDelay reading "
            f"(first: {missing_delay[:3]}); it is the second route the clock-origin term is "
            "checked by, and without it that check would be silently skipped"
        )
    mean_delay = (
        sum(float(delays[vid]) for vid in departed_by_horizon) / len(departed_by_horizon)
        if departed_by_horizon
        else 0.0
    )

    return SumoEpisodeReconstruction(
        e_sumo=e_sumo,
        p_sumo=p_sumo,
        w_sumo=w_sumo,
        mean_depart_delay=mean_delay,
        horizon=end,
        n_observations=len(times),
        n_intended=recorder.intended.n,
        n_departed=len(departed_by_horizon),
        n_arrived=sum(1 for vid in arrived_at if arrived_at[vid] <= end),
        n_never_inserted=len(set(due) - set(departed_by_horizon)),
        n_pending_at_horizon=len(recorder.pending_at_horizon),
        n_teleports=recorder.n_teleport_events,
        n_vanished_without_arrival=recorder.n_vanished_without_arrival,
        n_arrived_never_observed_at_a_boundary=recorder.n_arrived_never_observed_at_a_boundary,
        max_abs_depart_clock_deviation=recorder.max_abs_depart_clock_deviation,
        halting=recorder.halting_agreement,
    )


def sumo_observer_env_class() -> type[Any]:
    """Build ``PerSecondSumoObserver``, a ``SumoEnv`` subclass, on first use.

    ⚠️ **The class is created lazily and this is deliberate**, mirroring
    ``engine_att_reference.observer_env_class``: ``envs/sumo_env.py`` imports traci at construction
    time, so binding the subclass at module scope would make importing this module impossible
    wherever SUMO is absent -- including CI, where the SUMO-gated tests skip cleanly.

    The subclass overrides exactly two methods:

    * ``_simulate`` -- its own loop over ``simulationStep()``, **keeping the
      ``metrics.on_sim_step()`` call in the position the frozen env puts it** (see the module
      docstring), then one :meth:`SumoObservationRecorder.observe` per simulated second.
    * ``reset`` -- clears the recorder, then defers to ``super()``.  Returns ``info`` ONLY (C1).
    """
    global _OBSERVER_ENV_CLASS
    if _OBSERVER_ENV_CLASS is not None:
        return _OBSERVER_ENV_CLASS

    from envs.sumo_env import SumoEnv

    class PerSecondSumoObserver(SumoEnv):
        """A ``SumoEnv`` that records the simulator's vehicle-id sets once per simulated second.

        Format version: ``p7.1-metric-freeze/1.0``.  Alignment convention: a snapshot is taken
        IMMEDIATELY AFTER each ``simulationStep()`` and labelled with ``simulation.getTime()``, so
        every per-step list traci returns describes the step just executed.  See the module
        docstring.
        """

        def __init__(self, *args: Any, halting_check: bool = True, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            route_files = route_files_of(self._config_path)
            if len(route_files) != 1:
                raise ValueError(
                    f"{self._config_path} declares {len(route_files)} route files; E_sumo's "
                    "population is read from one demand file and merging several would need a "
                    "declared id-collision policy this task does not have"
                )
            self.intended_departures = read_intended_departures(route_files[0])
            # dt from the simulator itself, never from a config or a default: it is the offset (S)
            # the population rule is built on.
            self.recorder = SumoObservationRecorder(
                delta_time=float(self.delta_time),
                intended_departures=self.intended_departures,
                step_length=float(self._sumo.simulation.getDeltaT()),
            )
            self.halting_check = bool(halting_check)
            self._monitored_incoming_lanes: tuple[str, ...] = tuple(
                sorted({lid for ix in self._intersections for lid in ix.incoming_lanes})
            )

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            self.recorder.clear()
            return super().reset(seed=seed, options=options)

        def _halting_snapshot(self) -> dict[str, tuple[int, int, int]]:
            """SUMO's own halting count against counting ``speed < 0.1`` over the same lane.

            One speed map per second rather than one traci call per lane per vehicle.  A vehicle
            listed on the lane but absent from the speed map arrived during this step; it is
            counted as *missing* rather than silently treated as moving.
            """
            lane = self._sumo.lane
            vehicle = self._sumo.vehicle
            speeds = {vid: float(vehicle.getSpeed(vid)) for vid in vehicle.getIDList()}
            snapshot: dict[str, tuple[int, int, int]] = {}
            for lane_id in self._monitored_incoming_lanes:
                ours = 0
                missing = 0
                for vid in lane.getLastStepVehicleIDs(lane_id):
                    speed = speeds.get(vid)
                    if speed is None:
                        missing += 1
                        continue
                    if speed < HALT_SPEED_THRESHOLD:
                        ours += 1
                snapshot[lane_id] = (
                    int(lane.getLastStepHaltingNumber(lane_id)),
                    ours,
                    missing,
                )
            return snapshot

        def _simulate(self, num_steps: int) -> None:
            sumo = self._sumo
            metrics = self._metrics
            simulation = sumo.simulation
            vehicle = sumo.vehicle
            for _ in range(int(num_steps)):
                sumo.simulationStep()
                # ⚠️ FROZEN BEHAVIOUR, NOT AN OPTIONAL LINE.  envs/sumo_env.py:229-231 drives this
                # hook after every step; without it metrics/sumo.py:255-267 falls back to polling
                # getArrivedIDList once per decision step and att_sumo_env silently becomes a
                # different quantity.  It is called BEFORE the observer's reads, in the frozen
                # env's own order.
                if metrics is not None:
                    metrics.on_sim_step()
                departed = tuple(simulation.getDepartedIDList())
                self.recorder.observe(
                    sim_time=float(simulation.getTime()),
                    present_ids=tuple(vehicle.getIDList()),
                    departed_ids=departed,
                    arrived_ids=tuple(simulation.getArrivedIDList()),
                    pending_ids=tuple(simulation.getPendingVehicles()),
                    teleport_start_ids=tuple(simulation.getStartingTeleportIDList()),
                    departure_facts={
                        vid: (
                            float(vehicle.getDeparture(vid)),
                            float(vehicle.getDepartDelay(vid)),
                        )
                        for vid in departed
                    },
                    halting=self._halting_snapshot() if self.halting_check else None,
                )

    _OBSERVER_ENV_CLASS = PerSecondSumoObserver
    return _OBSERVER_ENV_CLASS


def make_observer_sumo_env(
    sumocfg_path: str | Path,
    settings: Mapping[str, Any],
    *,
    halting_check: bool = True,
) -> Any:
    """Build the observer env exactly as ``experiments.envs.make_env`` builds a ``SumoEnv``.

    ``make_env`` returns the frozen class and cannot return a subclass, so its SUMO branch
    (``experiments/envs.py:114-122``) is mirrored here: the same seven ``common`` keys, the same
    optional ``metrics`` / ``obs_norm`` handling, the same ``gui`` and ``libsumo`` flags, and
    ``experiments.envs.phase_control_cls`` imported rather than a local mapping.
    """
    from experiments.envs import phase_control_cls

    common: dict[str, Any] = {
        "max_steps": settings["max_steps"],
        "delta_time": settings["delta_time"],
        "global_reward_fn": settings["global_reward_fn"],
        "local_reward_fn": settings["local_reward_fn"],
        "global_reward_weight": settings["global_reward_weight"],
        "phase_control_cls": phase_control_cls(settings["control_mode"]),
        "state_features": settings["state_features"],
    }
    if settings["metrics"] is not None:
        common["metrics"] = settings["metrics"]
    if settings["obs_norm"] is not None:
        common["obs_norm"] = settings["obs_norm"]
    return sumo_observer_env_class()(
        sumocfg_path=str(sumocfg_path),
        gui=settings["gui"],
        libsumo=settings["libsumo"],
        halting_check=halting_check,
        **common,
    )


# ----------------------------------------------------------------------
# LAYER B -- the campaign harness
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeEpisode:
    """One episode of one arm: every reconstruction, the env's own metric, and the counters.

    ``att_reference_*`` are this module's reconstructions on SUMO and
    ``engine_att_reference``'s on CityFlow, so one record shape carries both backends and the
    decomposition reads across.  ``att_env`` is ``att_ours`` on CityFlow and ``att_sumo_env`` on
    SUMO -- in both cases the value ``info["average_travel_time"]`` carried at the horizon.
    """

    backend: str
    arm: str
    episode: int
    engine_seed: int
    observer: bool
    att_reference_created_population: float
    att_reference_entered_population: float
    att_reference_entered_running: float
    att_env: float
    att_p7_0_stored: float
    n_created: int
    n_entered: int
    n_never_entered: int
    n_pending_at_horizon: int
    n_teleports: int
    n_vanished_without_arrival: int
    n_arrived_never_observed_at_a_boundary: int
    mean_depart_delay: float
    max_abs_depart_clock_deviation: float
    halting_max_abs_difference: int
    halting_n_lane_seconds: int
    halting_n_disagreeing_lane_seconds: int
    n_observations: int
    seconds: float
    #: Which SUMO configuration produced this episode (:data:`SUMO_REGIMES`).  CityFlow rows carry
    #: the default: there is one CityFlow configuration and no teleporting to disable.
    regime: str = "parity"

    @property
    def term_population(self) -> float:
        """``P - E``: restrict the population, holding the pool clock fixed."""
        return float(self.att_reference_entered_population) - float(
            self.att_reference_created_population
        )

    @property
    def term_clock_origin(self) -> float:
        """``W - P``: move the clock origin, holding the population fixed."""
        return float(self.att_reference_entered_running) - float(
            self.att_reference_entered_population
        )

    @property
    def term_cadence(self) -> float:
        """``att_env - W``: the metric's own update cadence (A5 -- measured, not assumed)."""
        return float(self.att_env) - float(self.att_reference_entered_running)

    @property
    def decomposition_residual(self) -> float:
        """``|(population + clock_origin + cadence) - (att_env - E)|``.

        ⚠️ **Reported, never gated on: it is an algebraic tautology** -- ``P`` and ``W`` cancel, so
        it is zero for any four numbers and detects no wrong reconstruction.  See the module
        docstring for the measurement that establishes this.  :attr:`clock_origin_second_route_error`
        is the check with power.
        """
        total = float(self.att_env) - float(self.att_reference_created_population)
        return abs((self.term_population + self.term_clock_origin + self.term_cadence) - total)

    @property
    def clock_origin_second_route_error(self) -> float:
        """``|term_clock_origin + mean_depart_delay|`` -- the same quantity by two routes.

        ``W - P`` is a difference of two averages of per-vehicle contributions; ``mean_depart_delay``
        averages the per-vehicle delays the simulator reported (``getDepartDelay`` on SUMO, the
        admission latency on CityFlow).  They are equal identically, so a disagreement means one of
        the two clocks was recorded wrong -- which is exactly what the residual cannot see.
        """
        return abs(self.term_clock_origin + float(self.mean_depart_delay))

    @property
    def reproduces_p7_0(self) -> bool:
        """``np.float32(att_env) == np.float32(att_p7_0_stored)``.

        ⚠️ **This is a substitution of the exact form the artifact permits, not a loosening**
        (``BRIEF_34`` Amendment A2): ``att_per_step`` is stored ``float32`` (C6) while the env
        produces ``float64``, so a raw float64 comparison would fail on the logger's rounding and
        not on anything the simulator did.  No tolerance is introduced; the float64 value is kept
        in the record beside it so a full-precision comparison remains possible later.

        ⚠️ **Detection floor, stated rather than left to be discovered:** one ``float32`` ulp, about
        ``3e-5`` at an ATT of 355.  Drift below that is invisible to this check by construction.
        """
        stored = np.float32(self.att_p7_0_stored)
        if np.isnan(stored):
            raise ValueError(
                f"{self.backend}/{self.arm} episode {self.episode} has no P7.0 counterpart, so "
                "there is nothing to reproduce; callers must check has_p7_0_reference first "
                "rather than reading a verdict that does not exist"
            )
        return bool(np.float32(self.att_env) == stored)

    @property
    def has_p7_0_reference(self) -> bool:
        """Whether this episode has a stored P7.0 cell to be compared against at all.

        False for the hz4x4 timing episode, which P7.0 never ran; those rows are reported as
        ``n_unverified`` and are excluded from every reproduction count rather than being counted
        as passes.
        """
        return not bool(np.isnan(np.float32(self.att_p7_0_stored)))

    def as_record(self) -> dict[str, Any]:
        """The JSON row: every constructor field, then the derived quantities beside them."""
        record = {field.name: getattr(self, field.name) for field in fields(self)}
        record.update(
            {
                "term_population": self.term_population,
                "term_clock_origin": self.term_clock_origin,
                "term_cadence": self.term_cadence,
                "decomposition_residual": self.decomposition_residual,
                "clock_origin_second_route_error": self.clock_origin_second_route_error,
                "has_p7_0_reference": self.has_p7_0_reference,
                "reproduces_p7_0": self.reproduces_p7_0 if self.has_p7_0_reference else None,
            }
        )
        return record


def collect_style_args(
    backend: str,
    arm: str,
    config_path: str | Path,
    *,
    episodes: int = EPISODES_PER_ARM,
    base_seed: int = BASE_SEED,
    sentinel_out_dir: str | Path,
) -> argparse.Namespace:
    """P7.0's own collection arguments, parsed by ``offline.collect``'s real parser.

    ``BRIEF_34`` Amendment A3: P7.0's cells were produced by ``collect.main(argv)``
    (``transfer_gate.py:1068``), so the arms are ``collect.POLICIES`` and the settings are
    ``transfer_gate.COLLECT_SETTINGS``.  Building the namespace through ``collect.build_parser()``
    rather than a hand-made ``SimpleNamespace`` is what makes "the same settings" checkable instead
    of retyped.

    ⚠️ *sentinel_out_dir* satisfies the parser's required ``--out-dir`` and **is never opened**:
    this module logs no corpus.  A test asserts it is not created.
    """
    from offline import collect
    from offline.transfer_gate import COLLECT_SETTINGS

    argv = [
        "--backend",
        str(backend),
        "--env-config",
        str(config_path),
        "--policy",
        str(arm),
        "--episodes",
        str(int(episodes)),
        "--base-seed",
        str(int(base_seed)),
        "--out-dir",
        str(sentinel_out_dir),
        *COLLECT_SETTINGS,
    ]
    return collect.build_parser().parse_args(argv)


def build_policy(env: Any, args: argparse.Namespace) -> Callable[[dict[str, Any]], np.ndarray]:
    """``collect.POLICIES[args.policy](env, args, rng)`` with ``rng`` seeded as collection seeded it.

    ⚠️ **One policy object serves all five episodes of an arm** (``collect.py:717-718`` builds it
    once per draw), so the ``random`` arm's generator state carries across episode boundaries.
    Rebuilding it per episode would change the action stream and the reproduction check would fail
    for a reason this harness invented.
    """
    from offline.collect import POLICIES

    rng = np.random.default_rng(int(args.base_seed))
    return POLICIES[args.policy](env, args, rng)


def _roll_episodes(
    env: Any,
    policy: Callable[[dict[str, Any]], np.ndarray],
    *,
    episodes: int,
    base_seed: int,
    on_episode: Callable[[int, int, dict[str, Any], float], None],
    before_episode: Callable[[int], None] | None = None,
) -> None:
    """``collect.main``'s episode loop, line for line, with a hook instead of a logger.

    ``offline/collect.py:720-738``: the seed is ``base_seed + index``, the bound is
    ``env.max_steps``, and the loop breaks on terminate/truncate.  *on_episode* receives
    ``(index, engine_seed, final_info, wall_seconds)`` after each episode and before the next
    reset, which is the only moment the simulator still holds that episode's state.
    """
    for index in range(int(episodes)):
        engine_seed = int(base_seed) + index
        if before_episode is not None:
            before_episode(index)
        started = time.perf_counter()
        info = env.reset(seed=engine_seed)
        for _ in range(int(env.max_steps)):
            action = policy(info)
            reward, terminated, truncated, info = env.step(action)
            del reward
            if terminated or truncated:
                break
        on_episode(index, engine_seed, info, time.perf_counter() - started)


def run_sumo_arm(
    arm: str,
    *,
    config_path: str | Path,
    output_root: str | Path,
    episodes: int = EPISODES_PER_ARM,
    base_seed: int = BASE_SEED,
    observer: bool = True,
    halting_episodes: int = 1,
    regime: str = "parity",
) -> dict[str, Any]:
    """Roll one SUMO anchor arm, mirroring ``collect.main``'s loop, and reconstruct every episode.

    With *observer* false the plain frozen ``SumoEnv`` is used: that arm is at once the
    observer-interference control and the timing basis P7.3 will schedule from (Amendment A6).

    ⚠️ **The halting cross-check runs on the FIRST *halting_episodes* episodes only, and the count
    it actually covered is in every row.**  Measured at G1 on one hz1x1 MaxPressure episode: the
    observer alone costs 14.63 s against the frozen env's 12.26 s (+19 %), while the halting check
    takes it to 42.83 s (3.49x) -- it queries every vehicle's speed every second.  The question it
    answers is whether one threshold constant matches SUMO's own, which does not need every episode:
    one episode already contributes 28,800 lane-seconds.  Set it to 0 to skip, or to *episodes* to
    cover the arm.
    """
    from agent.utils.utils import Utils
    from offline.collect import _build_env_spec

    if regime not in SUMO_REGIMES:
        raise ValueError(f"{regime!r} is not one of {list(SUMO_REGIMES)}")
    # ⚠️ Amendment D2: the regime selects its own assertion. A teleport-free regime that was not in
    # fact teleport-free would otherwise be reported as one.
    require_no_teleports = regime == "noteleport"
    cell = f"sumo__{arm}"
    stored = _stored_reference(cell, output_root, episodes, regime=regime)

    # collect.main:596 seeds the global RNGs before the env is built; the order is part of the
    # protocol, not decoration.
    Utils.seed_everything(int(base_seed))
    args = collect_style_args(
        "sumo",
        arm,
        config_path,
        episodes=episodes,
        base_seed=base_seed,
        sentinel_out_dir=Path(output_root) / "p7_1" / "NEVER_WRITTEN",
    )
    spec = _build_env_spec(args)

    rows: list[FreezeEpisode] = []
    started = time.perf_counter()
    if observer:
        env = make_observer_sumo_env(
            config_path, spec.settings, halting_check=int(halting_episodes) > 0
        )
    else:
        from experiments.envs import make_env

        env = make_env(spec)
    try:
        policy = build_policy(env, args)

        def _record(index: int, engine_seed: int, info: dict[str, Any], seconds: float) -> None:
            att_env = float(info["average_travel_time"])
            if observer:
                built = reconstruct_sumo_episode(env.recorder)
                rows.append(
                    FreezeEpisode(
                        backend="sumo",
                        arm=arm,
                        episode=index,
                        engine_seed=engine_seed,
                        observer=True,
                        att_reference_created_population=built.e_sumo.value,
                        att_reference_entered_population=built.p_sumo.value,
                        att_reference_entered_running=built.w_sumo.value,
                        att_env=att_env,
                        att_p7_0_stored=stored[index],
                        n_created=built.e_sumo.n_ids,
                        n_entered=built.n_departed,
                        n_never_entered=built.n_never_inserted,
                        n_pending_at_horizon=built.n_pending_at_horizon,
                        n_teleports=built.n_teleports,
                        n_vanished_without_arrival=built.n_vanished_without_arrival,
                        n_arrived_never_observed_at_a_boundary=(
                            built.n_arrived_never_observed_at_a_boundary
                        ),
                        mean_depart_delay=built.mean_depart_delay,
                        max_abs_depart_clock_deviation=built.max_abs_depart_clock_deviation,
                        halting_max_abs_difference=built.halting.max_abs_difference,
                        halting_n_lane_seconds=built.halting.n_lane_seconds,
                        halting_n_disagreeing_lane_seconds=(
                            built.halting.n_disagreeing_lane_seconds
                        ),
                        n_observations=built.n_observations,
                        seconds=seconds,
                        regime=regime,
                    )
                )
            else:
                rows.append(
                    _unobserved_row(
                        "sumo", arm, index, engine_seed, att_env, stored, seconds, regime=regime
                    )
                )
            print(
                f"  sumo/{arm} ep{index} seed={engine_seed} att={att_env!r} "
                f"{seconds:.2f}s",
                flush=True,
            )

        def _select_halting(index: int) -> None:
            if observer:
                env.halting_check = index < int(halting_episodes)

        _roll_episodes(
            env,
            policy,
            episodes=episodes,
            base_seed=base_seed,
            on_episode=_record,
            before_episode=_select_halting,
        )
    finally:
        env.close()

    if require_no_teleports:
        offenders = [
            f"ep{row.episode} ({row.n_teleports} teleports, "
            f"{row.n_vanished_without_arrival} vanished)"
            for row in rows
            if row.observer and (row.n_teleports != 0 or row.n_vanished_without_arrival != 0)
        ]
        if offenders:
            raise ValueError(
                f"the {regime!r} regime is defined by teleporting being OFF, but {len(offenders)} "
                f"episode(s) teleported or lost a vehicle: {offenders[:5]}. Either the .sumocfg "
                "does not carry <time-to-teleport value='-1'/> or SUMO removed a vehicle for "
                "another reason; reporting this arm as teleport-free would be false"
            )

    return _chunk_payload(
        "sumo", arm, rows, observer=observer, episodes=episodes, base_seed=base_seed,
        config_path=config_path, seconds=time.perf_counter() - started,
        halting_episodes=int(halting_episodes) if observer else 0, regime=regime,
    )


def _unobserved_row(
    backend: str,
    arm: str,
    index: int,
    engine_seed: int,
    att_env: float,
    stored: Sequence[float],
    seconds: float,
    *,
    regime: str = "parity",
) -> FreezeEpisode:
    """A control episode: the env metric and the wall clock, and NaN where nothing was observed.

    The reconstructions are NaN rather than 0.0 so that an unobserved row can never be averaged
    into a reported reconstruction by accident; :func:`freeze_artifact` checks the decomposition
    only on observed rows and says so.
    """
    nan = float("nan")
    return FreezeEpisode(
        backend=backend,
        arm=arm,
        episode=index,
        engine_seed=engine_seed,
        observer=False,
        att_reference_created_population=nan,
        att_reference_entered_population=nan,
        att_reference_entered_running=nan,
        att_env=att_env,
        att_p7_0_stored=stored[index],
        n_created=-1,
        n_entered=-1,
        n_never_entered=-1,
        n_pending_at_horizon=-1,
        n_teleports=-1,
        n_vanished_without_arrival=-1,
        n_arrived_never_observed_at_a_boundary=-1,
        mean_depart_delay=nan,
        max_abs_depart_clock_deviation=nan,
        halting_max_abs_difference=-1,
        halting_n_lane_seconds=0,
        halting_n_disagreeing_lane_seconds=0,
        n_observations=0,
        seconds=seconds,
        regime=regime,
    )


def _stored_reference(
    cell: str, output_root: str | Path, episodes: int, *, regime: str = "parity"
) -> tuple[float, ...]:
    """P7.0's per-episode cells for *cell*, or NaNs when this cell has no P7.0 counterpart.

    ⚠️ **Only the ``parity`` regime has one.**  A1b runs a different ``.sumocfg``, so comparing it
    against P7.0's teleport-enabled cells would assert that two different configurations produce the
    same number -- the opposite of what A1b is measuring. Its rows are reported as unverified.
    """
    if regime != "parity" or cell not in P7_0_CELL_DIRS:
        return tuple(float("nan") for _ in range(int(episodes)))
    stored = p7_0_horizon_att(cell, output_root)
    if len(stored) < int(episodes):
        raise ValueError(
            f"{cell} has {len(stored)} stored episodes in output/p7_0 but {episodes} were "
            "requested; the reproduction check is per episode and cannot be padded"
        )
    return stored[: int(episodes)]


def _chunk_payload(
    backend: str,
    arm: str,
    rows: Sequence[FreezeEpisode],
    *,
    observer: bool,
    episodes: int,
    base_seed: int,
    config_path: str | Path,
    seconds: float,
    halting_episodes: int,
    regime: str = "parity",
) -> dict[str, Any]:
    """One cell's chunk: the header the resume predicate reads, and every row."""
    records = [row.as_record() for row in rows]
    verified = [row for row in rows if row.has_p7_0_reference]
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "backend": backend,
        "arm": arm,
        "observer": bool(observer),
        "regime": str(regime),
        "halting_episodes": int(halting_episodes),
        "config": str(config_path),
        "episodes": int(episodes),
        "base_seed": int(base_seed),
        "is_complete": len(records) == int(episodes),
        "reproduction": reproduction_report(
            [row.att_env for row in verified], [row.att_p7_0_stored for row in verified]
        ),
        "rows": records,
        "seconds": float(seconds),
        "seconds_per_episode": float(seconds) / max(1, len(records)),
    }


def run_cityflow_arm(
    arm: str,
    *,
    config_path: str | Path,
    output_root: str | Path,
    episodes: int = EPISODES_PER_ARM,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Roll one CityFlow anchor arm through ``engine_att_reference``'s Layer A.

    ``BRIEF_34`` Amendment A4: ``gate_episode`` builds one env per call and would break the RNG
    continuity above, so its *instrument* is reused -- ``make_observer_env``,
    ``EngineObservationRecorder``, ``reconstruct_episode`` -- inside a ``collect``-shaped loop.
    ``offline/engine_att_reference.py`` is imported and never modified.
    """
    from agent.utils.utils import Utils
    from offline.admission_probe import created_from_flow
    from offline.collect import _build_env_spec, _cityflow_flow_source
    from offline.engine_att_reference import make_observer_env, reconstruct_episode

    cell = f"cityflow__{arm}"
    stored = _stored_reference(cell, output_root, episodes)

    Utils.seed_everything(int(base_seed))
    args = collect_style_args(
        "cityflow",
        arm,
        config_path,
        episodes=episodes,
        base_seed=base_seed,
        sentinel_out_dir=Path(output_root) / "p7_1" / "NEVER_WRITTEN",
    )
    spec = _build_env_spec(args)
    horizon = int(args.max_steps) * int(args.delta_time)
    created = created_from_flow(_cityflow_flow_source(config_path), horizon_seconds=horizon)

    rows: list[FreezeEpisode] = []
    started = time.perf_counter()
    env = make_observer_env(config_path, spec.settings)
    try:
        policy = build_policy(env, args)

        def _record(index: int, engine_seed: int, info: dict[str, Any], seconds: float) -> None:
            att_env = float(info["average_travel_time"])
            built = reconstruct_episode(env.recorder)
            rows.append(
                FreezeEpisode(
                    backend="cityflow",
                    arm=arm,
                    episode=index,
                    engine_seed=engine_seed,
                    observer=True,
                    att_reference_created_population=built.engine_population.value,
                    att_reference_entered_population=built.entered_population.value,
                    att_reference_entered_running=built.entered_running.value,
                    att_env=att_env,
                    att_p7_0_stored=stored[index],
                    n_created=built.engine_population.n_ids,
                    n_entered=built.entered_running.n_ids,
                    n_never_entered=built.engine_population.n_ids - built.entered_running.n_ids,
                    # CityFlow's insertion buffer has no "pending at the horizon" readout that is
                    # independent of the pool; never_entered above is the same population, so this
                    # field is -1 rather than a number that would look like a second measurement.
                    n_pending_at_horizon=-1,
                    n_teleports=0,
                    n_vanished_without_arrival=0,
                    n_arrived_never_observed_at_a_boundary=-1,
                    mean_depart_delay=built.latency.mean,
                    max_abs_depart_clock_deviation=0.0,
                    halting_max_abs_difference=-1,
                    halting_n_lane_seconds=0,
                    halting_n_disagreeing_lane_seconds=0,
                    n_observations=built.n_observations,
                    seconds=seconds,
                )
            )
            print(
                f"  cityflow/{arm} ep{index} seed={engine_seed} att={att_env!r} "
                f"{seconds:.2f}s",
                flush=True,
            )

        _roll_episodes(env, policy, episodes=episodes, base_seed=base_seed, on_episode=_record)
    finally:
        env.close()

    payload = _chunk_payload(
        "cityflow",
        arm,
        rows,
        observer=True,
        episodes=episodes,
        base_seed=base_seed,
        config_path=config_path,
        seconds=time.perf_counter() - started,
        halting_episodes=0,
    )
    # Reported, never gating: P8.4b's criterion 2 on this scenario, recomputed here by a second
    # route (the flow file) against the observed pool.
    payload["created_from_flow"] = int(created)
    payload["created_equals_observed_on_every_episode"] = all(
        row.n_created == int(created) for row in rows
    )
    return payload


def p7_0_horizon_att(cell: str, output_root: str | Path) -> tuple[float, ...]:
    """P7.0's stored ``att_per_step[-1]`` per episode, in manifest order, for one cell.

    Read from the ``.npz`` and never from ``docs/data/p7_0_gate.json``: the artifact records the
    per-cell mean, and the reproduction check is per episode.
    """
    from offline.trajectory_logger import MANIFEST_NAME, load_episode

    if cell not in P7_0_CELL_DIRS:
        raise KeyError(
            f"{cell!r} is not a P7.0 cell; the known ones are {sorted(P7_0_CELL_DIRS)}. An unknown "
            "cell must refuse rather than return an empty tuple that would make a reproduction "
            "check pass vacuously"
        )
    directory = Path(output_root) / P7_0_CELL_DIRS[cell]
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    values: list[float] = []
    for entry in manifest["episodes"]:
        episode = load_episode(directory / entry["filename"])
        if episode.att_per_step is None:
            raise ValueError(
                f"{entry['filename']} carries no att_per_step; it predates format v1.1 and the "
                "registered primary metric cannot be read from it"
            )
        values.append(float(episode.att_per_step[-1]))
    return tuple(values)


def reproduction_report(fresh: Sequence[float], stored: Sequence[float]) -> dict[str, Any]:
    """``n_equal / n`` under ``==`` in float32, with the max absolute difference in both precisions.

    Reports counts and never the word "matches" (``BRIEF_34`` section 7).
    """
    fresh_values = [float(v) for v in fresh]
    stored_values = [float(v) for v in stored]
    if len(fresh_values) != len(stored_values):
        raise ValueError(
            f"the reproduction check was handed {len(fresh_values)} fresh values against "
            f"{len(stored_values)} stored ones; the count must be equal or the comparison pairs "
            "episodes with the wrong episodes"
        )
    if not fresh_values:
        return {
            "n": 0,
            "n_equal": 0,
            "all_equal": None,
            "max_abs_difference_float32": None,
            "max_abs_difference_float64": None,
            "per_episode_equal": [],
            "comparison": "np.float32(fresh) == np.float32(stored)",
        }
    fresh32 = np.asarray(fresh_values, dtype=np.float32)
    stored32 = np.asarray(stored_values, dtype=np.float32)
    equal = fresh32 == stored32
    return {
        "n": len(fresh_values),
        "n_equal": int(np.count_nonzero(equal)),
        "all_equal": bool(np.all(equal)),
        "max_abs_difference_float32": float(
            np.max(np.abs(fresh32.astype(np.float64) - stored32.astype(np.float64)))
        ),
        "max_abs_difference_float64": float(
            np.max(
                np.abs(
                    np.asarray(fresh_values, dtype=np.float64)
                    - np.asarray(stored_values, dtype=np.float64)
                )
            )
        ),
        "per_episode_equal": [bool(flag) for flag in equal],
        "comparison": "np.float32(fresh) == np.float32(stored)",
    }


def rho_table(att_by_cell: Mapping[str, float]) -> dict[str, Any]:
    """``PREREGISTRATION`` section 3.4's ρ, per backend, under each ATT definition.

    Delegates the arithmetic to ``transfer_gate.rho`` so the formula has one implementation; the
    test recomputes it from the raw cells by the written formula, which is the second route.
    """
    from offline.transfer_gate import rho

    groups: dict[tuple[str, str], dict[str, float]] = {}
    for key, value in att_by_cell.items():
        backend, arm, definition = str(key).split("__", 2)
        groups.setdefault((backend, definition), {})[arm] = float(value)

    table: dict[str, dict[str, dict[str, float]]] = {}
    for (backend, definition), arms in sorted(groups.items()):
        missing = [anchor for anchor in ("fixedtime", "maxpressure") if anchor not in arms]
        if missing:
            raise ValueError(
                f"{backend}/{definition} is missing the {missing} anchor(s); PREREGISTRATION "
                "section 3.4 normalises between BOTH anchors and a one-anchor ratio is a "
                "different quantity"
            )
        cell: dict[str, float] = {
            "delta": arms["fixedtime"] - arms["maxpressure"],
        }
        for arm, value in sorted(arms.items()):
            cell[arm] = rho(arms["fixedtime"], value, arms["maxpressure"])
        table.setdefault(backend, {})[definition] = cell
    return table


def assert_metric_freeze_writable(path: str | Path) -> Path:
    """DEFAULT-DENY: refuse anything under an ``output/`` that is not this task's own.

    A path is allowed when the component immediately after ``output`` is in
    :data:`ALLOWED_P7_1_OUTPUT_ENTRIES`; **everything else under ``output/`` is refused**, including
    directories that do not exist yet and including ``output/p7_0``, which this campaign READS and
    must never write.

    Matching is on whole **path components** of the resolved path, never on a string prefix: a
    prefix test makes ``output/p7_1`` and ``output/p7_10`` indistinguishable.  Returns the path
    unresolved, so it composes.
    """
    target = Path(path)
    parts = target.resolve().parts
    for index, part in enumerate(parts[:-1]):
        head = parts[index + 1]
        if part == "output" and head not in ALLOWED_P7_1_OUTPUT_ENTRIES:
            raise ValueError(
                f"{target}: output/{head} belongs to another campaign and is read-only here; "
                f"P7.1 writes only {list(ALLOWED_P7_1_OUTPUT_ENTRIES)} (BRIEF_34 section 5 G2). "
                "This fence is default-deny, so a directory added to output/ after this code was "
                "written is protected without being named -- and output/p7_0 in particular is READ "
                "by this task and must never be written by it"
            )
    return target


def chunk_name(
    backend: str, arm: str, *, observer: bool = True, regime: str = "parity"
) -> str:
    """``freeze_<backend>_<arm>[_noteleport][_unobserved].json`` -- one file per cell.

    The regime is in the NAME as well as in the header: A1 and A1b are two measurements of the same
    arm and must not overwrite each other.
    """
    if regime not in SUMO_REGIMES:
        raise ValueError(f"{regime!r} is not one of {list(SUMO_REGIMES)}")
    return (
        f"freeze_{backend}_{arm}"
        f"{'' if regime == 'parity' else '_' + regime}"
        f"{'' if observer else '_unobserved'}.json"
    )


def chunk_is_reusable(
    payload: Mapping[str, Any],
    *,
    backend: str,
    arm: str,
    observer: bool,
    episodes: int,
    regime: str = "parity",
) -> bool:
    """Whether an existing chunk may be skipped rather than re-rolled.

    Only when it is this exact cell, at this format version, complete, and clean -- never on the
    filename alone.  A chunk that does not parse is not reusable and is re-run.

    **"Clean" is re-derived here, never read** (Amendment E1, pre-flight PART 2 MAJOR 1 and 3).
    Three properties beyond identity, each of which a stored field could assert falsely:

    * every row that has a P7.0 reference reproduces it, recomputed from ``att_env`` and
      ``att_p7_0_stored`` rather than from the row's own ``reproduces_p7_0`` flag;
    * on a ``noteleport`` chunk, every observed row really has ``n_teleports == 0`` and
      ``n_vanished_without_arrival == 0`` -- D2's regime assertion lives in the runner, and a file
      placed in the work dir would otherwise bypass it;
    * the row count, the identity of every row, and the header all agree.

    ⚠️ A chunk that fails the reproduction check is **evidence, not garbage**: the caller moves it
    aside (see :func:`failed_chunk_destination`) instead of overwriting it.
    """
    try:
        if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
            return False
        if str(payload.get("backend")) != str(backend):
            return False
        if str(payload.get("arm")) != str(arm):
            return False
        if bool(payload.get("observer")) is not bool(observer):
            return False
        if str(payload.get("regime", "parity")) != str(regime):
            return False
        if not bool(payload.get("is_complete")):
            return False
        if int(payload.get("episodes", -1)) != int(episodes):
            return False
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != int(episodes):
            return False
        # ⭐ The header is not the cell: a chunk whose rows belong to another arm would otherwise
        # be inherited under the right filename.
        for row in rows:
            if str(row["backend"]) != str(backend) or str(row["arm"]) != str(arm):
                return False
            if bool(row["observer"]) is not bool(observer):
                return False
            if str(row.get("regime", "parity")) != str(regime):
                return False
            # MAJOR 3: a cell that failed its reproduction is complete but NOT clean. Recomputed,
            # because the row's own verdict is exactly what a hand-made chunk would lie about.
            if not row_reproduces_p7_0(row):
                return False
            # MAJOR 1: D2's assertion, re-derived on the chunk rather than trusted.
            if str(regime) == "noteleport" and bool(row["observer"]):
                if int(row["n_teleports"]) != 0:
                    return False
                if int(row["n_vanished_without_arrival"]) != 0:
                    return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


def row_reproduces_p7_0(row: Mapping[str, Any]) -> bool:
    """Whether one stored row reproduces its P7.0 cell, recomputed from the two ATT values.

    A row with no P7.0 counterpart (``att_p7_0_stored`` NaN -- every ``noteleport`` row, the hz4x4
    timing episode) has nothing to reproduce and is not counted as a failure.  Comparison is
    Amendment A2's exact form: ``np.float32(fresh) == np.float32(stored)``.
    """
    stored = np.float32(row["att_p7_0_stored"])
    if bool(np.isnan(stored)):
        return True
    return bool(np.float32(row["att_env"]) == stored)


def failed_chunk_destination(destination: str | Path) -> Path:
    """Where a chunk that is complete but not clean is moved so it survives the re-roll.

    ``<work>/failed/<name>``, with a numeric suffix if that slot is taken.

    ⚠️ **Deviation from Amendment E1 item 3, disclosed rather than silently reinterpreted:** the
    amendment says ``<name>.failed.json``, but ``freeze_sumo_maxpressure.json.failed.json`` still
    matches ``report``'s ``freeze_*.json`` glob, so the evidence would be read back in as data --
    the opposite of the intent. A subdirectory is outside a non-recursive glob by construction and
    needs no name filter, which is the ``p5_3a``/``p5_3b`` prefix-trap shape this repo has been
    bitten by. The manifest's ``find`` still lists it, so it stays part of the record.
    """
    target = Path(destination)
    directory = target.parent / "failed"
    candidate = directory / target.name
    index = 1
    while candidate.exists():
        candidate = directory / f"{target.stem}.{index}{target.suffix}"
        index += 1
    return candidate


def reusable_chunk_at(path: str | Path, **kwargs: Any) -> bool:
    """:func:`chunk_is_reusable` over a file, returning ``False`` for anything unreadable.

    A truncated or half-written chunk costs one re-roll, never the campaign.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    return chunk_is_reusable(payload, **kwargs)


def assert_campaign_complete(
    chunks: Sequence[Mapping[str, Any]], *, episodes: int, sources: Sequence[str] | None = None
) -> None:
    """Refuse unless exactly the campaign's twelve cells are present, once each, at *episodes*.

    Amendment E1 item 2, from pre-flight PART 2 MAJOR 2: ``report`` globbed whatever it found, so a
    deleted chunk produced a smaller artifact with rho silently missing a backend, and two files
    whose headers carried the same label silently overwrote one another while the log announced
    twelve cells. **The artifact's shape is a contract, not whatever is on disk.**

    Checked here rather than inside :func:`freeze_artifact` because the cell SET is the driver's
    contract while the cell CONTENT is the artifact's; a later task that legitimately assembles a
    different set calls the second and not the first.  ``_run_report`` calls both, in this order.
    """
    labels: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        try:
            label = _chunk_label(chunk)
        except (AttributeError, KeyError, TypeError) as exc:
            source = (sources[index] if sources else None) or f"chunk {index}"
            raise ValueError(
                f"{source} carries no readable cell identity ({exc}); a chunk that cannot be "
                "labelled cannot be checked against the campaign's cell set"
            ) from exc
        labels[label] = labels.get(label, 0) + 1
        if int(chunk.get("episodes", -1)) != int(episodes):
            raise ValueError(
                f"{label}: the chunk records {chunk.get('episodes')!r} episodes but this campaign "
                f"runs {int(episodes)}; a cell rolled at a different episode count is a different "
                "measurement and must not be averaged into the same table"
            )

    duplicated = sorted(label for label, count in labels.items() if count > 1)
    if duplicated:
        raise ValueError(
            f"{len(duplicated)} cell label(s) appear more than once: {duplicated}. Two files whose "
            "headers claim the same cell would overwrite each other in the artifact while the log "
            "still counted both -- check whether a chunk was copied under another filename"
        )

    expected = set(EXPECTED_CAMPAIGN_LABELS)
    present = set(labels)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing or unexpected:
        raise ValueError(
            f"the campaign is not the declared one: {len(missing)} cell(s) missing {missing} and "
            f"{len(unexpected)} unexpected {unexpected}. The artifact reports rho per backend and "
            "per regime, so a missing anchor removes a whole normalisation rather than shrinking a "
            "sample; run the missing cells or say in the packet why the set changed"
        )


def freeze_artifact(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate everything, then assemble ``docs/data/p7_1_metric_freeze.json``.

    **Raises rather than returning a partial artifact.**  Order is the barrier: completeness, then
    the per-episode reproduction against ``output/p7_0/``, then the exactness of the decomposition
    residual, then the summaries.  The caller writes only what this returns, so a refusal anywhere
    leaves the destination untouched.
    """
    if not chunks:
        raise ValueError(
            "no chunks were supplied, so there is nothing to assemble; an empty campaign must "
            "refuse rather than write a vacuous artifact"
        )

    # 1. Completeness.
    for chunk in chunks:
        label = f"{chunk.get('backend')}/{chunk.get('arm')}"
        if chunk.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"{label}: format version {chunk.get('format_version')!r} is not "
                f"{ARTIFACT_FORMAT_VERSION!r}; a chunk written under another version may not be "
                "read under this one's conventions"
            )
        rows = chunk.get("rows")
        if not bool(chunk.get("is_complete")) or not isinstance(rows, list):
            raise ValueError(
                f"{label}: the chunk is not complete, so its episodes may not be averaged into a "
                "reported number"
            )
        if len(rows) != int(chunk.get("episodes", -1)):
            raise ValueError(
                f"{label}: {len(rows)} rows against {chunk.get('episodes')} episodes requested; "
                "a partial cell is a refusal, not a smaller sample"
            )

    every_row = [row for chunk in chunks for row in chunk["rows"]]

    # 1b. The row schema. Amendment E1 item 4 (mn-6): a row carrying `null` where a count belongs
    #     used to reach the summary and raise a bare TypeError traceback. Every field the checks
    #     and the summaries below read is validated here, once, so a malformed row is a REFUSED
    #     with a name in it rather than a stack trace the driver cannot classify.
    for chunk in chunks:
        regime = str(chunk.get("regime", "parity"))
        for row in chunk["rows"]:
            _assert_row_is_well_formed(row, regime=regime)

    # 2. The reproduction against output/p7_0, RECOMPUTED here rather than trusting the flag the
    #    runner stored: a stored verdict is not evidence.
    failures: list[str] = []
    n_verified = 0
    for row in every_row:
        if not bool(np.isnan(np.float32(row["att_p7_0_stored"]))):
            n_verified += 1
        if not row_reproduces_p7_0(row):
            failures.append(
                f"{row['backend']}/{row['arm']} ep{row['episode']} "
                f"(fresh {row['att_env']!r} against stored {row['att_p7_0_stored']!r})"
            )
    if failures:
        raise ValueError(
            f"{len(failures)} of {n_verified} episodes do not reproduce their P7.0 cell: "
            f"{failures[:5]}. That is a finding about SUMO's determinism and must be read, not "
            "averaged into a table"
        )

    # 2b. D2's regime assertion, RE-DERIVED on the stored rows (Amendment E1 item 1, PART 2
    #     MAJOR 1). The runner raises before it writes such a chunk, so this can only fire on a
    #     file that arrived some other way -- which is exactly the bypass the pre-flight found.
    offenders = [
        f"{row['backend']}/{row['arm']} ep{row['episode']} "
        f"({row['n_teleports']} teleports, {row['n_vanished_without_arrival']} vanished)"
        for chunk in chunks
        if str(chunk.get("regime", "parity")) == "noteleport"
        for row in chunk["rows"]
        if bool(row["observer"])
        and (int(row["n_teleports"]) != 0 or int(row["n_vanished_without_arrival"]) != 0)
    ]
    if offenders:
        raise ValueError(
            f"{len(offenders)} episode(s) in a teleport-free chunk did teleport or lose a vehicle: "
            f"{offenders[:5]}. The noteleport regime IS the claim that this did not happen, so the "
            "artifact may not report it as a teleport-free arm"
        )

    # 3. The clock-origin term by its second route -- the decomposition's ONE gating check.
    #    ⚠️ The residual is deliberately NOT gated on: it is an algebraic tautology (module
    #    docstring). Gating on it would look like a check and be none.
    for row in every_row:
        if not bool(row["observer"]):
            continue
        error = float(row["clock_origin_second_route_error"])
        if not error <= CLOCK_ORIGIN_TOLERANCE:
            raise ValueError(
                f"{row['backend']}/{row['arm']} ep{row['episode']}: the clock-origin term is "
                f"{row['term_clock_origin']!r} but the mean insertion delay is "
                f"{row['mean_depart_delay']!r}, a disagreement of {error!r} against a tolerance of "
                f"{CLOCK_ORIGIN_TOLERANCE!r}. The two are the same quantity by two routes, so one "
                "of the departure clocks was recorded wrong"
            )

    # 4. Summaries.  Only observed chunks carry reconstructions.
    observed = [row for row in every_row if bool(row["observer"])]
    cells: dict[str, Any] = {}
    att_by_cell: dict[str, float] = {}
    for chunk in chunks:
        if not bool(chunk["observer"]):
            continue
        backend, arm = str(chunk["backend"]), str(chunk["arm"])
        regime = str(chunk.get("regime", "parity"))
        # ⚠️ The regime is part of the label, not a footnote: rho is normalised WITHIN one backend
        # and one configuration, and pooling two regimes' anchors would compare a teleporting
        # fixed-time against a teleport-free MaxPressure.
        label = backend if regime == "parity" else f"{backend}_{regime}"
        rows = chunk["rows"]
        summary = {
            "n_episodes": len(rows),
            "att_created_population_mean": _mean(rows, "att_reference_created_population"),
            "att_entered_population_mean": _mean(rows, "att_reference_entered_population"),
            "att_entered_running_mean": _mean(rows, "att_reference_entered_running"),
            "att_env_mean": _mean(rows, "att_env"),
            "att_env_per_episode": [float(row["att_env"]) for row in rows],
            "term_population_mean": _mean(rows, "term_population"),
            "term_clock_origin_mean": _mean(rows, "term_clock_origin"),
            "term_cadence_mean": _mean(rows, "term_cadence"),
            "n_created": [int(row["n_created"]) for row in rows],
            "n_never_entered": [int(row["n_never_entered"]) for row in rows],
            "n_teleports": [int(row["n_teleports"]) for row in rows],
            "n_pending_at_horizon": [int(row["n_pending_at_horizon"]) for row in rows],
            "reproduction": chunk["reproduction"],
            "seconds_per_episode": float(chunk["seconds_per_episode"]),
        }
        summary["regime"] = regime
        cells[f"{label}__{arm}"] = summary
        att_by_cell[f"{label}__{arm}__created_population"] = summary[
            "att_created_population_mean"
        ]
        att_by_cell[f"{label}__{arm}__env"] = summary["att_env_mean"]

    complete_backends = {
        backend
        for backend in {key.split("__", 1)[0] for key in att_by_cell}
        if all(f"{backend}__{arm}__env" in att_by_cell for arm in ANCHOR_ARMS)
    }
    rho_input = {
        key: value
        for key, value in att_by_cell.items()
        if key.split("__", 1)[0] in complete_backends
    }

    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "task": "P7.1",
        "role": (
            "measures both ATT definitions on both backends, each under one per-second instrument; "
            "proposes the freeze and rules nothing"
        ),
        "what_this_does_not_say": [
            "It does not decide whether E_sumo or the env metric is primary on SUMO: Rule R is a "
            "CityFlow gate with no SUMO counterpart and the amendment is the coordinator's.",
            "It does not compare raw travel times across backends; every cross-backend statement "
            "is about rho (PREREGISTRATION section 3.4).",
            "A reproduction difference below one float32 ulp is invisible to the reproduction "
            "check, because output/p7_0 stores att_per_step as float32.",
        ],
        "cells": cells,
        "rho": rho_table(rho_input) if rho_input else {},
        "reproduction": {
            "n_verified": n_verified,
            "n_equal": n_verified - len(failures),
            "comparison": "np.float32(fresh) == np.float32(stored)",
            "detection_floor": "one float32 ulp of the stored value",
            "per_cell": {
                _chunk_label(chunk): chunk["reproduction"]
                for chunk in chunks
            },
        },
        "decomposition": {
            "gated_on": "clock_origin_second_route_error",
            "tolerance": CLOCK_ORIGIN_TOLERANCE,
            "max_clock_origin_second_route_error": max(
                (float(row["clock_origin_second_route_error"]) for row in observed), default=0.0
            ),
            "max_decomposition_residual": max(
                (float(row["decomposition_residual"]) for row in observed), default=0.0
            ),
            "residual_is_reported_not_gated": (
                "(P-E)+(W-P)+(att-W)-(att-E) cancels P and W identically, so the residual is an "
                "algebraic tautology: measured in float64, moving P by 1e-9 leaves it at 0.0 and "
                "moving P by 100 leaves it at 4.44e-15. The power is in the second route above."
            ),
        },
        "cadence": {
            "registered_in": "BRIEF_34 Amendment A5",
            "n_observed_episodes": len(observed),
            "n_with_zero_cadence_term": sum(
                1 for row in observed if float(row["term_cadence"]) == 0.0
            ),
            "max_abs_term_cadence": max(
                (abs(float(row["term_cadence"])) for row in observed), default=0.0
            ),
            "n_arrived_never_observed_at_a_boundary": [
                int(row["n_arrived_never_observed_at_a_boundary"]) for row in observed
            ],
            "n_vanished_without_arrival": [
                int(row["n_vanished_without_arrival"]) for row in observed
            ],
        },
        "halting_threshold": {
            "declared": HALT_SPEED_THRESHOLD,
            "max_abs_difference": max(
                (int(row["halting_max_abs_difference"]) for row in observed), default=-1
            ),
            "n_lane_seconds": sum(int(row["halting_n_lane_seconds"]) for row in observed),
            "n_disagreeing_lane_seconds": sum(
                int(row["halting_n_disagreeing_lane_seconds"]) for row in observed
            ),
        },
        "timing": {
            _chunk_label(chunk): {
                "seconds": float(chunk["seconds"]),
                "seconds_per_episode": float(chunk["seconds_per_episode"]),
                "n": len(chunk["rows"]),
                "observer": bool(chunk["observer"]),
                "regime": str(chunk.get("regime", "parity")),
                "halting_episodes": int(chunk.get("halting_episodes", 0)),
            }
            for chunk in chunks
        },
        "episodes": every_row,
    }


#: Every row field the artifact's checks or summaries read, and the type each must carry.  A row is
#: validated against this once, before anything reads it (Amendment E1 item 4, mn-6).
_ROW_NUMERIC_FIELDS: tuple[str, ...] = (
    "att_reference_created_population",
    "att_reference_entered_population",
    "att_reference_entered_running",
    "att_env",
    "att_p7_0_stored",
    "term_population",
    "term_clock_origin",
    "term_cadence",
    "decomposition_residual",
    "clock_origin_second_route_error",
    "mean_depart_delay",
)
_ROW_INTEGER_FIELDS: tuple[str, ...] = (
    "episode",
    "n_created",
    "n_entered",
    "n_never_entered",
    "n_pending_at_horizon",
    "n_teleports",
    "n_vanished_without_arrival",
    "n_arrived_never_observed_at_a_boundary",
    "halting_max_abs_difference",
    "halting_n_lane_seconds",
    "halting_n_disagreeing_lane_seconds",
    "n_observations",
)


def _assert_row_is_well_formed(row: Mapping[str, Any], *, regime: str) -> None:
    """Refuse a row whose fields are missing, ``null`` or the wrong type, naming the field.

    Unobserved rows carry ``NaN`` and ``-1`` where nothing was reconstructed; those are values, not
    absences, and pass.  ``None`` never does: it is what a hand-edited chunk carries, and reading it
    used to produce a ``TypeError`` traceback instead of a refusal the driver can classify.
    """
    where = (
        f"{row.get('backend', '?')}/{row.get('arm', '?')} ep{row.get('episode', '?')}"
        f"{'' if regime == 'parity' else ' [' + regime + ']'}"
    )
    for name in ("backend", "arm", "observer"):
        if row.get(name) is None:
            raise ValueError(f"{where}: the row has no {name!r}, so it cannot be placed in a cell")
    if not isinstance(row["observer"], bool):
        raise ValueError(
            f"{where}: observer is {row['observer']!r} ({type(row['observer']).__name__}), not a "
            "bool; a JSON string 'false' is truthy and would file a control row as an observed one"
        )
    for name in _ROW_NUMERIC_FIELDS + _ROW_INTEGER_FIELDS:
        if name not in row:
            raise ValueError(f"{where}: the row is missing {name!r}, which the artifact reads")
        value = row[name]
        if value is None or isinstance(value, bool):
            raise ValueError(
                f"{where}: {name} is {value!r}; a count or a travel time cannot be null or a bool, "
                "and reading it would raise inside the summary instead of refusing here"
            )
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{where}: {name} is {value!r} ({type(value).__name__}), not a number"
            )
    for name in _ROW_INTEGER_FIELDS:
        if float(row[name]) != int(row[name]):
            raise ValueError(f"{where}: {name} is {row[name]!r}, which is not a whole count")


def _chunk_label(chunk: Mapping[str, Any]) -> str:
    """``<backend>[_<regime>]__<arm>[__unobserved]`` -- unique per measurement, not per arm."""
    regime = str(chunk.get("regime", "parity"))
    backend = str(chunk["backend"]) if regime == "parity" else f"{chunk['backend']}_{regime}"
    return f"{backend}__{chunk['arm']}" + ("" if chunk["observer"] else "__unobserved")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    """The mean of one field over a cell's rows, refusing an empty cell."""
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"cannot average {key} over an empty cell")
    return float(np.mean(values))


def build_parser() -> argparse.ArgumentParser:
    """``python -m offline.sumo_att_reference`` -- the campaign CLI.

    ``allow_abbrev=False`` on every parser: with argparse's default a flag copied from a sibling
    module would be silently accepted as an abbreviation and mean something the operator did not
    type.
    """
    parser = argparse.ArgumentParser(
        prog="python -m offline.sumo_att_reference",
        description=(
            "P7.1 half A: measure both ATT definitions on both backends. Reports; does not rule."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", default="output", help="the tree that holds p7_0/ and p7_1/")
    parser.add_argument("--work-dir", default=None, help="default <output-root>/p7_1")
    parser.add_argument("--out-dir", default="docs/data")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_ARM)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)

    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, helptext in (
        ("run-sumo", "one SUMO anchor arm on the parity scenario"),
        ("run-cityflow", "one CityFlow anchor arm on the nominal scenario"),
    ):
        cell = subparsers.add_parser(name, help=helptext, allow_abbrev=False)
        cell.add_argument("--arm", choices=sorted(ANCHOR_ARMS), required=True)
        cell.add_argument("--config", default=None, help="override the declared scenario config")
        if name == "run-sumo":
            cell.add_argument(
                "--regime",
                choices=sorted(SUMO_REGIMES),
                default="parity",
                help=(
                    "parity = P7.0's own .sumocfg, teleports ENABLED (the reproduction is defined "
                    "against it); noteleport = A1b's new file with time-to-teleport -1, which also "
                    "ASSERTS that no episode teleported"
                ),
            )
            cell.add_argument(
                "--observer",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="--no-observer rolls the frozen env: the interference control and the timing",
            )
            cell.add_argument(
                "--halting-episodes",
                type=int,
                default=1,
                help=(
                    "run the halting-threshold cross-check on the first N episodes (G1 measured "
                    "it at 2.9x the episode cost on its own; one episode is 28,800 lane-seconds)"
                ),
            )

    timing = subparsers.add_parser(
        "timing-hz4x4",
        help="one hz4x4 gudang episode through the observer -- TIMING ONLY, its ATT is not a result",
        allow_abbrev=False,
    )
    timing.add_argument("--episodes", type=int, default=1)

    subparsers.add_parser("report", help="assemble docs/data/p7_1_metric_freeze.json", allow_abbrev=False)
    return parser


def _work_dir(args: argparse.Namespace) -> Path:
    """The campaign's own directory, fenced before anything is created."""
    return assert_metric_freeze_writable(
        args.work_dir or (Path(args.output_root) / DEFAULT_WORK_DIRNAME)
    )


def _declared_config(command: str, override: str | None, regime: str = "parity") -> Path:
    """The scenario a subcommand runs, defaulting to the one P7.0 used for that backend."""
    if override is not None:
        return Path(override)
    from offline import parity
    from offline.transfer_gate import backend_config

    if command == "run-sumo":
        if regime == "noteleport":
            # Amendment D2's NEW file. The parity .sumocfg is never edited: recorded runs used it.
            return Path(str(parity.DECLARED_PARITY_SUMOCFG).replace(".sumocfg", "_noteleport.sumocfg"))
        return Path(parity.DECLARED_PARITY_SUMOCFG)
    if command == "run-cityflow":
        return Path(backend_config("cityflow"))
    return (
        parity.REPO_ROOT
        / "scenarios"
        / "hangzhou_4x4_gudang_18041610_1h"
        / "hangzhou_4x4_gudang_18041610_1h.sumocfg"
    )


def _run_cell(args: argparse.Namespace) -> int:
    """Roll one cell, skipping only a chunk that is complete, clean and for this exact cell."""
    from offline.dt_gate import write_json_atomic

    work = _work_dir(args)
    backend = "sumo" if args.command == "run-sumo" else "cityflow"
    observer = bool(getattr(args, "observer", True))
    regime = str(getattr(args, "regime", "parity"))
    destination = assert_metric_freeze_writable(
        work / chunk_name(backend, args.arm, observer=observer, regime=regime)
    )
    identity = {
        "backend": backend,
        "arm": args.arm,
        "observer": observer,
        "episodes": int(args.episodes),
        "regime": regime,
    }
    if reusable_chunk_at(destination, **identity):
        # ⚠️ The message says WHAT was checked (Amendment E1 item 3): the previous wording,
        # "complete and clean", was true of a chunk whose reproduction had failed, because the
        # predicate did not look at it. It does now, and the message names the checks.
        print(
            f"{destination.name}: {int(args.episodes)} episodes, this cell "
            f"({backend}/{args.arm}, observer={observer}, regime={regime}), every P7.0 reference "
            "reproduced"
            + (", no teleports" if regime == "noteleport" else "")
            + " -- skipping",
            flush=True,
        )
        return 0
    if destination.exists():
        # A complete chunk that merely failed its checks is EVIDENCE and is moved aside rather than
        # overwritten; anything unreadable or for another cell is simply re-rolled over.
        keep = failed_chunk_destination(destination)
        assert_metric_freeze_writable(keep)
        keep.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, keep)
        print(
            f"{destination.name}: not reusable (incomplete, for another cell, unreadable, "
            f"non-reproducing or teleporting) -- moved to {keep.parent.name}/{keep.name} and "
            "re-running",
            flush=True,
        )

    config = _declared_config(args.command, args.config, regime)
    if backend == "sumo":
        payload = run_sumo_arm(
            args.arm,
            config_path=config,
            output_root=args.output_root,
            episodes=int(args.episodes),
            base_seed=int(args.base_seed),
            observer=observer,
            halting_episodes=int(getattr(args, "halting_episodes", 1)),
            regime=regime,
        )
    else:
        payload = run_cityflow_arm(
            args.arm,
            config_path=config,
            output_root=args.output_root,
            episodes=int(args.episodes),
            base_seed=int(args.base_seed),
        )

    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload, destination)
    reproduction = payload["reproduction"]
    print(
        f"{destination.name}: {reproduction['n_equal']}/{reproduction['n']} episodes reproduce "
        f"their P7.0 cell, {payload['seconds_per_episode']:.2f} s/episode",
        flush=True,
    )
    # Non-zero when a cell does not reproduce, so the driver stops on the first bad cell rather
    # than an hour later; the chunk is written first, so the evidence survives.
    return 0 if reproduction["n"] == 0 or reproduction["all_equal"] else 1


def _run_timing(args: argparse.Namespace) -> int:
    """A4: one hz4x4 gudang episode for the clock alone."""
    from offline.dt_gate import write_json_atomic

    work = _work_dir(args)
    destination = assert_metric_freeze_writable(work / "timing_hz4x4_gudang.json")
    # Amendment E1 item 4 (mn-1): A4 was the one stage a restart always re-rolled, and it is the
    # most expensive one -- 438.69 s under the observer, measured by the pre-flight. The `role`
    # key is what stops an A1 `random` chunk being mistaken for it: the two share every identity
    # field and differ only in the scenario they ran.
    if destination.is_file() and _timing_chunk_is_reusable(destination, episodes=int(args.episodes)):
        print(
            f"{destination.name}: 1 timing episode on hz4x4 gudang, complete -- skipping "
            "(re-rolling it costs 7.3 min and measures the same clock)",
            flush=True,
        )
        return 0
    payload = run_sumo_arm(
        "random",
        config_path=_declared_config("timing-hz4x4", None, "parity"),
        output_root=args.output_root,
        episodes=int(args.episodes),
        base_seed=int(args.base_seed),
        observer=True,
        halting_episodes=0,
    )
    payload["role"] = TIMING_ROLE
    payload["att_is_not_a_result"] = (
        "the shipped gudang route file binds no parity vType (0 of 2983 vehicles, no tau), so "
        "this episode measures the clock and nothing else"
    )
    work.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload, destination)
    print(f"{destination.name}: {payload['seconds_per_episode']:.2f} s/episode", flush=True)
    return 0


def _timing_chunk_is_reusable(path: str | Path, *, episodes: int) -> bool:
    """Whether A4's chunk is complete, for A4, and at this episode count."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping) or payload.get("role") != TIMING_ROLE:
        return False
    return chunk_is_reusable(
        payload, backend="sumo", arm="random", observer=True, episodes=episodes, regime="parity"
    )


def _run_report(args: argparse.Namespace) -> int:
    """Assemble the artifact -- every destination fenced, everything validated, then one write."""
    from offline.dt_gate import write_json_atomic

    destination = assert_metric_freeze_writable(Path(args.out_dir) / "p7_1_metric_freeze.json")
    work = _work_dir(args)
    # ⚠️ Non-recursive and anchored: `smoke/`, `failed/` and any `.p4-….json.tmp` are invisible
    # here by construction rather than by a name filter.
    paths = sorted(work.glob("freeze_*.json"))
    chunks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not chunks:
        raise FileNotFoundError(
            f"{work} holds no freeze_*.json chunk, so there is nothing to report; run the cells "
            "first"
        )
    # The campaign's SHAPE before its content (Amendment E1 item 2): a deleted or duplicated cell
    # is refused here, so the artifact cannot be quietly smaller than the campaign.
    assert_campaign_complete(
        chunks, episodes=int(args.episodes), sources=[str(path) for path in paths]
    )
    payload = freeze_artifact(chunks)
    write_json_atomic(payload, destination)
    print(
        f"{destination}: {len(chunks)} cells, "
        f"{payload['reproduction']['n_equal']}/{payload['reproduction']['n_verified']} episodes "
        "reproduce their P7.0 cell",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; returns a process exit code (non-zero when a check fails)."""
    args = build_parser().parse_args(argv)
    handlers: Mapping[str, Callable[[argparse.Namespace], int]] = {
        "run-sumo": _run_cell,
        "run-cityflow": _run_cell,
        "timing-hz4x4": _run_timing,
        "report": _run_report,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        # A refusal is the designed outcome of every guard here, so it exits non-zero with its own
        # message rather than a traceback the driver would have to parse.
        print(f"REFUSED: {exc}", flush=True)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
