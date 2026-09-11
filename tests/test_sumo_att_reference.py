"""Tests for ``offline/sumo_att_reference.py`` -- P7.1 half A, the metric freeze.

Three layers, and the split matters because only the first runs everywhere:

* **pure** -- the route-file reader, the recorder's invariants, the arithmetic of the three
  reconstructions and the decomposition, the fence, the resume predicate.  No simulator.
* **SUMO-backed** -- opt-in, skipping with a reason naming ``sumo``/``traci``.
* **``output/``-backed** -- opt-in, skipping with a reason naming ``RLTRAFFIC_OUTPUT_ROOT``,
  because ``output/p7_0/`` is gitignored and lives in the main tree.

⚠️ **The synthetic stream below is hand-computed and is the specification.**  Four vehicles, a
horizon of 10 s, one never-inserted vehicle and one that departs late, chosen so that every term of
the decomposition is a different non-zero rational.

🔒 **It obeys the MEASURED one-step offset (S)**: a vehicle declared at ``depart = d`` is reported
departed at second ``getDeparture + 1``, and ``getDeparture >= d``.  Measured on the parity scenario
over 600 s and 307 departures, ``observed - getDeparture`` had exactly one value, ``1.0``.

===  ========  ============  ========  =======  ==========================================
id   intended  getDeparture  reported  arrives  role
===  ========  ============  ========  =======  ==========================================
v0   0         0             1         5        no insertion delay, completes
v1   1         3             4         9        insertion delay 2 s, completes
v2   2         2             3         never    still running at the horizon
v4   4         --            never     --       never inserted: in E's population alone
v3   100       --            --        --       never due inside the horizon: no population
===  ========  ============  ========  =======  ==========================================

    E = (5 + 8 + 8 + 6) / 4 = 27/4 = 6.75          population {v0, v1, v2, v4}, clock intended
    P = (5 + 8 + 8)     / 3 = 21/3 = 7.0           population {v0, v1, v2},     clock intended
    W = (5 + 6 + 8)     / 3 = 19/3                 population {v0, v1, v2},     clock getDeparture

    term_population   = P - E = +0.25
    term_clock_origin = W - P = -2/3 = -mean(depart_delay) over {0, 2, 0}
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline import sumo_att_reference as sar

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------


def _sumo_available() -> bool:
    """Both halves: the Python bindings and the binary the env actually launches."""
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def _traci_available() -> bool:
    """The bindings alone -- enough to build the env subclass and read its source."""
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def output_root() -> Path:
    """``RLTRAFFIC_OUTPUT_ROOT``, else ``<repo>/output``; skip unless P7.0's cells are there."""
    env_value = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")
    candidate = Path(env_value) if env_value else REPO / "output"
    if not (candidate / "p7_0" / "sumo__maxpressure").is_dir():
        pytest.skip(
            f"P7.0's episodes are not present at {candidate}/p7_0: set RLTRAFFIC_OUTPUT_ROOT to "
            "the tree that holds them to run the reproduction tests"
        )
    return candidate


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


SYNTHETIC_INTENDED = {"v0": 0.0, "v1": 1.0, "v2": 2.0, "v4": 4.0, "v3": 100.0}
SYNTHETIC_HORIZON = 10.0


def _route_file(tmp_path: Path, body: str, name: str = "demand.rou.xml") -> Path:
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<routes>\n' + body + "\n</routes>\n",
        encoding="utf-8",
    )
    return path


def _intended() -> sar.IntendedDepartures:
    return sar.IntendedDepartures(by_id=dict(SYNTHETIC_INTENDED), source="synthetic")


SYNTHETIC_STEP = 1.0


def _drive_synthetic(
    recorder: sar.SumoObservationRecorder,
    *,
    drop_pending_at: float | None = None,
    last_second: float = SYNTHETIC_HORIZON,
) -> sar.SumoObservationRecorder:
    """Replay the docstring's four vehicles second by second, as the simulator would report them.

    Observations run ``1 .. last_second``: the first ``simulationStep()`` ends at ``t = dt``, never
    at ``t = 0``.  *drop_pending_at* omits the never-inserted vehicle from ``pending_ids`` at that
    second, which is the positive control for the pool identity: a vehicle that could have been
    inserted, has not departed and is not pending has fallen out of the accounting.
    """
    departs = {1.0: ["v0"], 3.0: ["v2"], 4.0: ["v1"]}
    arrives = {5.0: ["v0"], 9.0: ["v1"]}
    facts = {"v0": (0.0, 0.0), "v1": (3.0, 2.0), "v2": (2.0, 0.0)}
    present: set[str] = set()
    for tick in range(1, int(last_second) + 1):
        now = float(tick)
        departed = departs.get(now, [])
        arrived = arrives.get(now, [])
        present.update(departed)
        present.difference_update(arrived)
        # The measured rule: a vehicle is insertable from `intended + dt` onwards, and anything
        # insertable that has not departed must be pending.
        pending = [
            vid
            for vid, due in SYNTHETIC_INTENDED.items()
            if due + SYNTHETIC_STEP <= now
            and vid not in recorder.departed_at
            and vid not in departed
        ]
        if drop_pending_at is not None and now == drop_pending_at:
            pending = []
        recorder.observe(
            sim_time=now,
            present_ids=sorted(present),
            departed_ids=departed,
            arrived_ids=arrived,
            pending_ids=pending,
            departure_facts={vid: facts[vid] for vid in departed},
        )
    return recorder


def _recorder() -> sar.SumoObservationRecorder:
    return sar.SumoObservationRecorder(
        delta_time=5.0, intended_departures=_intended(), step_length=SYNTHETIC_STEP
    )


def _episode(**overrides: Any) -> sar.FreezeEpisode:
    """A record whose ATTs are the synthetic stream's, so the terms are the docstring's."""
    base: dict[str, Any] = {
        "backend": "sumo",
        "arm": "maxpressure",
        "episode": 0,
        "engine_seed": 1000,
        "observer": True,
        "att_reference_created_population": 6.75,
        "att_reference_entered_population": 7.0,
        "att_reference_entered_running": 19.0 / 3.0,
        "att_env": 19.0 / 3.0,
        "att_p7_0_stored": float(np.float32(19.0 / 3.0)),
        "n_created": 4,
        "n_entered": 3,
        "n_never_entered": 1,
        "n_pending_at_horizon": 1,
        "n_teleports": 0,
        "n_vanished_without_arrival": 0,
        "n_arrived_never_observed_at_a_boundary": 0,
        "mean_depart_delay": 2.0 / 3.0,
        "max_abs_depart_clock_deviation": 0.0,
        "halting_max_abs_difference": 0,
        "halting_n_lane_seconds": 80,
        "halting_n_disagreeing_lane_seconds": 0,
        "n_observations": 10,
        "seconds": 1.0,
    }
    base.update(overrides)
    return sar.FreezeEpisode(**base)


def _chunk(**overrides: Any) -> dict[str, Any]:
    rows = [_episode(episode=i, engine_seed=1000 + i).as_record() for i in range(5)]
    payload: dict[str, Any] = {
        "format_version": sar.ARTIFACT_FORMAT_VERSION,
        "backend": "sumo",
        "arm": "maxpressure",
        "observer": True,
        "episodes": 5,
        "base_seed": 1000,
        "is_complete": True,
        "halting_episodes": 1,
        "config": "scenarios/synthetic.sumocfg",
        "rows": rows,
        "seconds": 60.0,
        "seconds_per_episode": 12.0,
    }
    payload.update(overrides)
    # The header the artifact summarises from, built from whatever rows this chunk ended up with,
    # exactly as `_chunk_payload` builds it at run time.
    verified = [
        row
        for row in payload["rows"]
        if not np.isnan(np.float32(row["att_p7_0_stored"]))
    ]
    payload.setdefault(
        "reproduction",
        sar.reproduction_report(
            [row["att_env"] for row in verified], [row["att_p7_0_stored"] for row in verified]
        ),
    )
    return payload


# ----------------------------------------------------------------------
# 1. The route file: E_sumo's denominator comes from the demand, not the simulator
# ----------------------------------------------------------------------


def test_the_route_reader_reads_every_vehicle_of_the_parity_scenario() -> None:
    """2021 vehicles, departures spanning the episode -- the population ``E_sumo`` averages over."""
    from offline import parity

    intended = sar.read_intended_departures(parity.DECLARED_PARITY_ROU)

    assert intended.n == 2021
    assert len(set(intended.by_id)) == 2021
    assert min(intended.by_id.values()) == 0.0
    assert max(intended.by_id.values()) == 3599.0
    assert intended.due_by(3600.0) == frozenset(intended.by_id)


def test_the_route_reader_refuses_a_duplicate_id(tmp_path: Path) -> None:
    """Two vehicles under one id would silently shrink the denominator by one."""
    path = _route_file(
        tmp_path,
        '<vehicle depart="0" id="a"/>\n<vehicle depart="5" id="a"/>',
    )
    with pytest.raises(ValueError, match="duplicate|appears twice"):
        sar.read_intended_departures(path)


def test_the_route_reader_refuses_a_generated_demand_element(tmp_path: Path) -> None:
    """``<flow>`` ids are minted by the simulator, so the file cannot enumerate the population."""
    path = _route_file(tmp_path, '<flow id="f" begin="0" end="10" number="5"/>')
    with pytest.raises(ValueError, match="flow|trip|enumerat"):
        sar.read_intended_departures(path)


def test_the_route_reader_refuses_a_non_numeric_departure(tmp_path: Path) -> None:
    """``depart="triggered"`` has no intended time, so no pool clock exists for that vehicle."""
    path = _route_file(tmp_path, '<vehicle depart="triggered" id="a"/>')
    with pytest.raises(ValueError, match="triggered|numeric"):
        sar.read_intended_departures(path)


def test_route_files_of_resolves_against_the_config_directory() -> None:
    """The parity ``.sumocfg`` names its route file relatively; a wrong base silently misses it."""
    from offline import parity

    files = sar.route_files_of(parity.DECLARED_PARITY_SUMOCFG)

    assert [p.name for p in files] == [Path(parity.DECLARED_PARITY_ROU).name]
    assert all(p.is_file() for p in files)


# ----------------------------------------------------------------------
# 2. The pool identity, and the control that proves it can fail
# ----------------------------------------------------------------------


def test_the_pool_identity_holds_at_every_second_of_the_synthetic_stream() -> None:
    """``cumulative_departed | pending == {v : intended <= t}``, disjointly, every second."""
    recorder = _drive_synthetic(_recorder())

    assert recorder.observation_times == tuple(float(t) for t in range(1, 11))
    assert set(recorder.departed_at) == {"v0", "v1", "v2"}
    assert recorder.pending_at_horizon == frozenset({"v4"})


def test_the_pool_identity_catches_a_vehicle_that_falls_out_of_the_accounting() -> None:
    """⭐ The positive control. Without it, an identity that never fires proves nothing.

    ``v4`` is due at second 4 and never departs, so it must be pending; dropping it from the
    pending set is exactly the hole that would make ``E_sumo``'s denominator wrong.
    """
    with pytest.raises(ValueError, match="v4"):
        _drive_synthetic(_recorder(), drop_pending_at=5.0)


def test_observe_refuses_a_repeated_or_receding_second() -> None:
    """Every snapshot follows a ``simulationStep()``, so a repeated time means we lost sync."""
    recorder = _recorder()
    recorder.observe(
        sim_time=0.0,
        present_ids=["v0"],
        departed_ids=["v0"],
        arrived_ids=[],
        pending_ids=[],
        departure_facts={"v0": (0.0, 0.0)},
    )
    with pytest.raises(ValueError, match="did not advance"):
        recorder.observe(
            sim_time=0.0,
            present_ids=["v0"],
            departed_ids=[],
            arrived_ids=[],
            pending_ids=[],
        )


def test_observe_refuses_a_departure_the_demand_file_never_declared() -> None:
    """A vehicle the route file does not know cannot have an intended departure."""
    recorder = _recorder()
    with pytest.raises(ValueError, match="ghost|route file|not declared"):
        recorder.observe(
            sim_time=0.0,
            present_ids=["ghost"],
            departed_ids=["ghost"],
            arrived_ids=[],
            pending_ids=[],
            departure_facts={"ghost": (0.0, 0.0)},
        )


def test_observe_refuses_an_arrival_of_a_vehicle_that_never_departed() -> None:
    """Arrivals are credited against a departure clock; without one there is no contribution."""
    recorder = _recorder()
    with pytest.raises(ValueError, match="v1"):
        recorder.observe(
            sim_time=0.0,
            present_ids=[],
            departed_ids=[],
            arrived_ids=["v1"],
            pending_ids=["v1"],
        )


def test_observe_refuses_a_second_departure_of_one_vehicle() -> None:
    """A re-inserted id would overwrite its own departure clock and shorten its travel time."""
    recorder = _recorder()
    recorder.observe(
        sim_time=0.0,
        present_ids=["v0"],
        departed_ids=["v0"],
        arrived_ids=[],
        pending_ids=[],
        departure_facts={"v0": (0.0, 0.0)},
    )
    with pytest.raises(ValueError, match="already departed|twice"):
        recorder.observe(
            sim_time=1.0,
            present_ids=["v0"],
            departed_ids=["v0"],
            arrived_ids=[],
            pending_ids=[],
            departure_facts={"v0": (1.0, 1.0)},
        )


# ----------------------------------------------------------------------
# 3. The three reconstructions, hand-computed
# ----------------------------------------------------------------------


def test_the_three_reconstructions_equal_the_hand_computed_values() -> None:
    """The specification in this file's docstring, asserted as exact rationals."""
    built = sar.reconstruct_sumo_episode(_drive_synthetic(_recorder()), horizon=SYNTHETIC_HORIZON)

    assert built.e_sumo.n_ids == 4
    assert built.p_sumo.n_ids == 3
    assert built.w_sumo.n_ids == 3
    assert built.e_sumo.total == 27.0
    assert built.p_sumo.total == 21.0
    assert built.w_sumo.total == 19.0
    assert built.e_sumo.value == 27.0 / 4.0
    assert built.p_sumo.value == 21.0 / 3.0
    assert built.w_sumo.value == 19.0 / 3.0


def test_the_clock_origin_term_equals_minus_the_mean_departure_delay() -> None:
    """⭐ The second route: two summation orders that must agree.

    ``W - P`` is a difference of two averages of per-vehicle contributions; ``mean(getDepartDelay)``
    averages per-vehicle delays the simulator reported. They are equal identically, so a
    disagreement means one of the two clocks was recorded wrong.
    """
    built = sar.reconstruct_sumo_episode(_drive_synthetic(_recorder()), horizon=SYNTHETIC_HORIZON)

    difference = built.w_sumo.value - built.p_sumo.value
    assert built.mean_depart_delay == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert difference == pytest.approx(-built.mean_depart_delay, abs=1e-9)


def test_the_reconstruction_counts_the_populations_it_averages_over() -> None:
    """``never_entered`` is the gap between the demand file and the simulator, and is reported."""
    built = sar.reconstruct_sumo_episode(_drive_synthetic(_recorder()), horizon=SYNTHETIC_HORIZON)

    assert built.n_intended == 5
    assert built.n_departed == 3
    assert built.n_arrived == 2
    assert built.n_never_inserted == 1
    assert built.n_pending_at_horizon == 1
    assert built.n_vanished_without_arrival == 0
    assert built.max_abs_depart_clock_deviation == 0.0
    assert built.n_observations == 10
    # v0 arrives at t=5 having been present only at 1..4, so it was never in getIDList() at a
    # decision boundary (5 and 10 here) -- exactly the population metrics/sumo.py credits 0.0.
    assert built.n_arrived_never_observed_at_a_boundary == 1


def test_a_vehicle_that_vanishes_without_arriving_is_counted_not_ignored() -> None:
    """A teleport SUMO cannot resolve removes a vehicle; the env metric loses it and we count it."""
    recorder = _recorder()
    recorder.observe(
        sim_time=1.0,
        present_ids=["v0"],
        departed_ids=["v0"],
        arrived_ids=[],
        pending_ids=[],
        departure_facts={"v0": (0.0, 0.0)},
        teleport_start_ids=["v0"],
    )
    recorder.observe(
        sim_time=2.0, present_ids=[], departed_ids=[], arrived_ids=[], pending_ids=["v1"]
    )

    built = sar.reconstruct_sumo_episode(recorder, horizon=2.0)
    assert built.n_vanished_without_arrival == 1
    assert built.n_teleports == 1
    assert recorder.teleport_ids == frozenset({"v0"})


def test_the_halting_cross_check_accumulates_disagreements() -> None:
    """P7.0 left the halting threshold "documented, not measured"; this is the instrument."""
    recorder = _recorder()
    recorder.observe(
        sim_time=1.0,
        present_ids=[],
        departed_ids=[],
        arrived_ids=[],
        # v0 becomes insertable at second 0 + dt = 1 and has not departed, so the pool identity
        # requires it here. The identity refused this stream twice while it was being written --
        # once for an empty pending list and once for the pre-measurement timing -- which is the
        # invariant doing its job on the test's own fixture.
        pending_ids=["v0"],
        halting={"lane_a": (3, 3, 0), "lane_b": (4, 2, 1)},
    )

    agreement = recorder.halting_agreement
    assert agreement.n_lane_seconds == 2
    assert agreement.max_abs_difference == 2
    assert agreement.n_disagreeing_lane_seconds == 1
    assert agreement.n_missing == 1


def test_the_declared_halting_threshold_equals_the_frozen_metric_s() -> None:
    """⭐ Independence with an executed equality check, which is stronger than an import.

    The reconstruction may not import ``metrics/`` (it exists to be compared against it), so the
    threshold is re-declared -- and this test is what stops the two drifting apart silently.
    """
    from metrics.sumo import HALT_SPEED_THRESHOLD as frozen_threshold

    assert sar.HALT_SPEED_THRESHOLD == frozen_threshold


def _imports_metrics(source: str) -> bool:
    """Whether *source* imports anything from ``metrics``, by an AST walk and not by a substring."""
    import ast
    import textwrap

    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Import) and any(a.name.startswith("metrics") for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and str(node.module or "").startswith("metrics"):
            return True
    return False


def test_the_reconstruction_surface_imports_nothing_from_metrics() -> None:
    """A11's independence clause, applied to the SUMO twin and enforced over the declared surface."""
    import inspect

    assert sar.RECONSTRUCTION_SURFACE, "an empty surface would make this check vacuous"
    for name in sar.RECONSTRUCTION_SURFACE:
        assert not _imports_metrics(inspect.getsource(getattr(sar, name))), name


def test_the_import_fence_catches_a_planted_violation() -> None:
    """⭐ The positive control: a fence that has never refused anything proves nothing."""
    planted = "def offender():\n    from metrics.sumo import HALT_SPEED_THRESHOLD\n    return 1\n"
    innocent = "def fine():\n    import numpy as np\n    return np\n"

    assert _imports_metrics(planted)
    assert not _imports_metrics(innocent)


# ----------------------------------------------------------------------
# 4. The decomposition, and the reproduction check's exact form
# ----------------------------------------------------------------------


def test_the_decomposition_is_exact_on_a_record() -> None:
    """``population + clock_origin + cadence == att_env - E``, at ``0.0``, not within a tolerance."""
    episode = _episode()

    assert episode.term_population == 7.0 - 6.75
    assert episode.term_clock_origin == 19.0 / 3.0 - 7.0
    assert episode.term_cadence == 0.0
    assert episode.decomposition_residual == 0.0


def test_the_decomposition_residual_is_a_tautology_and_cannot_be_a_check() -> None:
    """🚨 MEASURED, and it is why the artifact does not gate on this quantity.

    This test was written expecting a 1e-9 perturbation of ``P`` to be caught. It is not: ``P`` and
    ``W`` cancel identically, so the residual is zero for **any** four numbers. Worse, a *large*
    perturbation makes it non-zero only through float rounding -- so gating on ``== 0.0`` would be
    blind to semantic error and would still refuse correct data. The check with power is
    :attr:`clock_origin_second_route_error`, tested below.
    """
    # A semantic error small enough to matter is invisible: exactly zero, in both middle terms
    # and in the endpoint.
    assert _episode(att_reference_entered_population=7.0 + 1e-9).decomposition_residual == 0.0
    assert _episode(att_reference_entered_running=19.0 / 3.0 + 1e-9).decomposition_residual == 0.0
    assert _episode(att_reference_created_population=6.75 + 1e-9).decomposition_residual == 0.0

    # A large one shows up only at float-rounding scale, never in proportion to the error: a
    # 100-unit error and a 50-unit error both land near 1e-15.
    for wrong, size in ((107.0, 100.0), (7.0 + 50.0, 50.0)):
        residual = _episode(att_reference_entered_population=wrong).decomposition_residual
        assert 0.0 < residual < 1e-12
        assert residual < size * 1e-15


def test_the_clock_origin_second_route_catches_a_perturbed_term() -> None:
    """⭐ The check that does have power: two independently recorded routes to one quantity."""
    assert _episode().clock_origin_second_route_error == pytest.approx(0.0, abs=1e-15)
    assert _episode(mean_depart_delay=2.0 / 3.0 + 1e-6).clock_origin_second_route_error > 1e-9
    assert (
        _episode(att_reference_entered_population=7.0 + 1e-6).clock_origin_second_route_error
        > 1e-9
    )


def test_a_non_zero_cadence_term_is_carried_rather_than_swallowed() -> None:
    """Amendment A5: when the env metric and ``W_sumo`` differ, the difference IS the third term."""
    episode = _episode(att_env=19.0 / 3.0 + 0.5, att_p7_0_stored=float(np.float32(19.0 / 3.0 + 0.5)))

    assert episode.term_cadence == pytest.approx(0.5, abs=1e-12)
    assert episode.decomposition_residual == 0.0


def test_the_reproduction_check_is_exact_after_the_storage_rounding() -> None:
    """Amendment A2's form, on P7.0's real ``sumo__maxpressure`` ep0 pair.

    The fresh value is the float64 the env produced when I re-rolled the episode; the stored one is
    what the float32 corpus holds. They are the same number after the logger's rounding.
    """
    episode = _episode(att_env=355.7984322508399, att_p7_0_stored=355.7984313964844)

    assert episode.reproduces_p7_0 is True


def test_the_reproduction_check_refuses_a_one_ulp_perturbation() -> None:
    """⭐ One float32 ulp is the smallest difference the stored precision can carry.

    The mutation has to be at the resolution of the record, not below it -- see the next test,
    which states the cost of that honestly instead of hiding it.
    """
    stored = np.float32(355.7984313964844)
    nudged = float(np.nextafter(stored, np.float32(np.inf)))
    episode = _episode(att_env=nudged, att_p7_0_stored=float(stored))

    assert episode.reproduces_p7_0 is False


def test_the_detection_floor_of_the_reproduction_check_is_one_float32_ulp() -> None:
    """⚠️ DISCLOSED, not hidden: what Amendment A2's exact form cannot see.

    ``att_per_step`` is stored ``float32``, so any drift below one ulp -- about 3e-5 at an ATT of
    355 -- is invisible to this check by construction. The brief's "refuses a 1e-9 perturbation"
    is therefore unsatisfiable against this corpus, and pretending otherwise would be a test that
    passes for the wrong reason.
    """
    stored = np.float32(355.7984313964844)
    ulp = float(np.nextafter(stored, np.float32(np.inf))) - float(stored)

    assert _episode(att_env=355.7984313964844 + 1e-9, att_p7_0_stored=float(stored)).reproduces_p7_0
    assert 1e-9 < ulp < 1e-4


def test_reproduction_report_counts_and_never_claims_a_match() -> None:
    """``n_equal / n`` with the max difference in both precisions (``BRIEF_34`` section 7)."""
    stored = [355.7984313964844, 379.9250793457031]
    fresh = [355.7984322508399, 379.9250793457031 + 1.0]

    report = sar.reproduction_report(fresh, stored)

    assert report["n"] == 2
    assert report["n_equal"] == 1
    assert report["max_abs_difference_float32"] == pytest.approx(1.0, abs=1e-4)
    assert report["max_abs_difference_float64"] == pytest.approx(1.0, abs=1e-4)


def test_reproduction_report_refuses_mismatched_lengths() -> None:
    """A shorter fresh list would silently compare episode 0 against episode 0 and stop."""
    with pytest.raises(ValueError, match="length|count"):
        sar.reproduction_report([1.0], [1.0, 2.0])


# ----------------------------------------------------------------------
# 5. rho, recomputed by a second route inside the test
# ----------------------------------------------------------------------


def test_rho_is_computed_within_each_backend_under_each_definition() -> None:
    """⭐ Recomputed here from the raw cells by the registered formula, not by calling the module."""
    cells = {
        "sumo__fixedtime__created_population": 400.0,
        "sumo__maxpressure__created_population": 300.0,
        "sumo__random__created_population": 700.0,
        "sumo__fixedtime__env": 371.0,
        "sumo__maxpressure__env": 360.0,
        "sumo__random__env": 483.0,
    }

    table = sar.rho_table(cells)

    for definition in ("created_population", "env"):
        ft = cells[f"sumo__fixedtime__{definition}"]
        mp = cells[f"sumo__maxpressure__{definition}"]
        rnd = cells[f"sumo__random__{definition}"]
        assert table["sumo"][definition]["fixedtime"] == 0.0
        assert table["sumo"][definition]["maxpressure"] == 1.0
        assert table["sumo"][definition]["random"] == (ft - rnd) / (ft - mp)
        assert table["sumo"][definition]["delta"] == ft - mp


def test_rho_refuses_a_zero_anchor_span() -> None:
    """Section 3.4's normalisation is undefined when the two anchors coincide."""
    cells = {
        "sumo__fixedtime__env": 300.0,
        "sumo__maxpressure__env": 300.0,
        "sumo__random__env": 700.0,
    }
    with pytest.raises(ValueError, match="span|undefined"):
        sar.rho_table(cells)


# ----------------------------------------------------------------------
# 6. The fence
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "p7_0/sumo__maxpressure/ep000000_seed1000.npz",
        "p7_0/manifest.json",
        "p5_3b/x.json",
        "p5_3b_decomp/decomp.json",
        "p4_dt/dt_seed101.pt",
        "never_seen_before/x.json",
        "SHA256SUMS_p7_0.txt",
        "SHA256SUMS_p5_3b.txt",
    ],
)
def test_the_fence_refuses_everything_under_output_that_is_not_this_task(
    tmp_path: Path, relative: str
) -> None:
    """Default-deny, including the predecessor this campaign READS."""
    with pytest.raises(ValueError, match="another campaign|read-only|belongs"):
        sar.assert_metric_freeze_writable(tmp_path / "output" / relative)


@pytest.mark.parametrize(
    "relative",
    ["p7_1/freeze_sumo_maxpressure.json", "p7_1/smoke/x.json", "SHA256SUMS_p7_1.txt"],
)
def test_the_fence_allows_exactly_this_tasks_own_outputs(tmp_path: Path, relative: str) -> None:
    """Positive control: a fence that refuses everything is not a fence, it is a wall."""
    target = tmp_path / "output" / relative

    assert sar.assert_metric_freeze_writable(target) == target


def test_the_fence_is_not_defeated_by_a_traversal(tmp_path: Path) -> None:
    """``p7_1/../p7_0`` resolves into the predecessor, and a string prefix test would miss it."""
    with pytest.raises(ValueError, match="another campaign|read-only|belongs"):
        sar.assert_metric_freeze_writable(tmp_path / "output" / "p7_1" / ".." / "p7_0" / "x.json")


def test_the_fence_does_not_protect_a_sibling_that_merely_shares_a_prefix(tmp_path: Path) -> None:
    """``p7_10`` is not ``p7_1``: whole-component matching, in both directions."""
    with pytest.raises(ValueError, match="another campaign|read-only|belongs"):
        sar.assert_metric_freeze_writable(tmp_path / "output" / "p7_10" / "x.json")


# ----------------------------------------------------------------------
# 7. Resume
# ----------------------------------------------------------------------


def test_a_complete_and_clean_chunk_is_reusable() -> None:
    """The positive control for the resume predicate."""
    assert sar.chunk_is_reusable(
        _chunk(), backend="sumo", arm="maxpressure", observer=True, episodes=5
    )


@pytest.mark.parametrize(
    "override",
    [
        {"is_complete": False},
        {"format_version": "p7.1-metric-freeze/0.9"},
        {"rows": []},
        {"arm": "random"},
        {"backend": "cityflow"},
        {"observer": False},
        {"rows_arm": "random"},
    ],
    ids=[
        "incomplete",
        "wrong-format-version",
        "no-rows",
        "wrong-arm",
        "wrong-backend",
        "wrong-observer-arm",
        "header-right-rows-wrong",
    ],
)
def test_a_chunk_that_is_not_this_exact_cell_is_re_run(override: dict[str, Any]) -> None:
    """⭐ The filename is not the cell. A chunk under the right name for the wrong cell is re-run.

    The parameters are *descriptions* rather than built payloads on purpose: building them in the
    decorator would call the module at collection time and turn every test in this file into one
    shared error while the module is still a skeleton.
    """
    fields = dict(override)
    rows_arm = fields.pop("rows_arm", None)
    if rows_arm is not None:
        fields["rows"] = [_episode(arm=rows_arm, episode=i).as_record() for i in range(5)]

    assert not sar.chunk_is_reusable(
        _chunk(**fields), backend="sumo", arm="maxpressure", observer=True, episodes=5
    )


@pytest.mark.parametrize("text", ["", "   ", "{", "[]", '"a string"', "null"])
def test_an_unreadable_chunk_is_re_run_rather_than_crashing(tmp_path: Path, text: str) -> None:
    """A truncated write must cost one re-roll, never the campaign."""
    path = tmp_path / "freeze_sumo_maxpressure.json"
    path.write_text(text, encoding="utf-8")

    assert not sar.reusable_chunk_at(
        path, backend="sumo", arm="maxpressure", observer=True, episodes=5
    )


def test_a_missing_chunk_is_not_reusable(tmp_path: Path) -> None:
    """The absent case, stated rather than left to the caller's ``is_file()``."""
    assert not sar.reusable_chunk_at(
        tmp_path / "nothing.json", backend="sumo", arm="maxpressure", observer=True, episodes=5
    )


def test_the_halting_subset_is_a_declared_number_and_not_a_hidden_default() -> None:
    """G1 measured the halting check at 2.9x the episode cost, so it runs on a declared subset.

    The knob is an episode COUNT rather than a boolean, so the artifact can say how much was
    covered instead of implying the whole campaign was.
    """
    parser = sar.build_parser()

    assert parser.parse_args(["run-sumo", "--arm", "maxpressure"]).halting_episodes == 1
    assert (
        parser.parse_args(["run-sumo", "--arm", "random", "--halting-episodes", "5"]).halting_episodes
        == 5
    )
    assert parser.parse_args(["run-sumo", "--arm", "random", "--halting-episodes", "0"]).halting_episodes == 0
    # The unobserved control has no reconstruction to check a threshold against, so it has no knob.
    assert not hasattr(parser.parse_args(["run-cityflow", "--arm", "random"]), "halting_episodes")


def test_an_abbreviated_flag_is_refused_rather_than_guessed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``allow_abbrev=False``: a flag copied from a sibling module must not silently half-match.

    The exit code and the message are both asserted: a bare ``raises(SystemExit)`` would be
    satisfied by argparse exiting for any reason at all, including a typo in this test.
    """
    # `match=r"^2$"` pins argparse's exit CODE through the match mechanism itself -- the idiom
    # tests/test_engine_att_reference.py:933 already uses, and stronger than a bare raises().
    with pytest.raises(SystemExit, match=r"^2$"):
        sar.build_parser().parse_args(["run-sumo", "--arm", "maxpressure", "--halting", "1"])

    assert "--halting" in capsys.readouterr().err


def test_chunk_names_separate_the_observed_and_unobserved_arms() -> None:
    """The two arms of A1 are different measurements and must not overwrite each other."""
    assert sar.chunk_name("sumo", "maxpressure", observer=True) != sar.chunk_name(
        "sumo", "maxpressure", observer=False
    )


# ----------------------------------------------------------------------
# 8. The barrier: validate everything, then write
# ----------------------------------------------------------------------


def test_the_artifact_refuses_an_episode_that_does_not_reproduce_p7_0() -> None:
    """One bad episode refuses the whole artifact; a partial one is never returned."""
    stored = np.float32(355.7984313964844)
    nudged = float(np.nextafter(stored, np.float32(np.inf)))
    rows = [_episode(episode=i).as_record() for i in range(5)]
    rows[3]["att_env"] = nudged
    rows[3]["att_p7_0_stored"] = float(stored)

    with pytest.raises(ValueError, match="reproduc"):
        sar.freeze_artifact([_chunk(rows=rows)])


def test_the_artifact_refuses_a_clock_origin_that_fails_its_second_route() -> None:
    """The one gating check of the decomposition, at the tolerance the brief registered."""
    rows = [_episode(episode=i).as_record() for i in range(5)]
    rows[0] = _episode(episode=0, mean_depart_delay=2.0 / 3.0 + 1e-6).as_record()

    with pytest.raises(ValueError, match="clock-origin|insertion delay|two routes"):
        sar.freeze_artifact([_chunk(rows=rows)])


def test_the_artifact_does_not_gate_on_the_tautological_residual() -> None:
    """⚠️ Stated as a test so the next reader cannot mistake a reported field for a check.

    A row whose ``P`` is wrong by 100 has a non-zero residual and passes, because the residual is
    not evidence; the same row is caught by the second route, which is.
    """
    wrong = _episode(episode=0, att_reference_entered_population=107.0)
    assert wrong.decomposition_residual != 0.0
    assert wrong.clock_origin_second_route_error > sar.CLOCK_ORIGIN_TOLERANCE

    rows = [_episode(episode=i).as_record() for i in range(5)]
    rows[0] = wrong.as_record()
    rows[0]["clock_origin_second_route_error"] = 0.0  # the second route silenced, residual left

    artifact = sar.freeze_artifact([_chunk(rows=rows)])
    assert artifact["decomposition"]["max_decomposition_residual"] > 0.0
    assert artifact["decomposition"]["gated_on"] == "clock_origin_second_route_error"


def test_the_artifact_refuses_an_incomplete_chunk() -> None:
    """A cell that did not finish must not be averaged into a reported number."""
    with pytest.raises(ValueError, match="complete|episodes"):
        sar.freeze_artifact([_chunk(is_complete=False)])


def test_a_refused_report_writes_nothing_and_creates_no_directory(tmp_path: Path) -> None:
    """⭐ The filesystem-mutation barrier: a refused run leaves the tree exactly as it was."""
    work = tmp_path / "output" / "p7_1"
    work.mkdir(parents=True)
    rows = [_episode(episode=i).as_record() for i in range(5)]
    rows[0] = _episode(episode=0, mean_depart_delay=2.0 / 3.0 + 1e-6).as_record()
    (work / sar.chunk_name("sumo", "maxpressure")).write_text(
        json.dumps(_chunk(rows=rows)), encoding="utf-8"
    )
    out_dir = tmp_path / "docs" / "data"
    out_dir.mkdir(parents=True)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    code = sar.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--work-dir",
            str(work),
            "--out-dir",
            str(out_dir),
            "report",
        ]
    )

    assert code != 0
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_the_report_destination_is_fenced_too(tmp_path: Path) -> None:
    """⚠️ ``--out-dir`` is a write path: misdirecting it into ``output/p7_0`` must be refused.

    The chunk is VALID and the destination directory EXISTS, so nothing but the fence can stop this
    write: remove the fence and the artifact lands inside P7.0's tree, which is what the assertion
    on the file's absence detects.
    """
    work = tmp_path / "output" / "p7_1"
    work.mkdir(parents=True)
    (work / sar.chunk_name("sumo", "maxpressure")).write_text(
        json.dumps(_chunk()), encoding="utf-8"
    )
    victim = tmp_path / "output" / "p7_0"
    victim.mkdir(parents=True)

    code = sar.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--work-dir",
            str(work),
            "--out-dir",
            str(victim),
            "report",
        ]
    )

    assert code != 0
    assert sorted(victim.iterdir()) == []


# ----------------------------------------------------------------------
# 9. The collection protocol: the same arms P7.0 ran
# ----------------------------------------------------------------------


def test_collect_style_args_carry_p7_0s_settings(tmp_path: Path) -> None:
    """Amendment A3: the settings come from ``transfer_gate.COLLECT_SETTINGS``, not from prose."""
    from offline import parity

    args = sar.collect_style_args(
        "sumo",
        "maxpressure",
        parity.DECLARED_PARITY_SUMOCFG,
        sentinel_out_dir=tmp_path / "never-written",
    )

    assert args.max_steps == 360
    assert args.delta_time == 10
    assert args.control_mode == "acyclic"
    assert args.global_reward_fn == "queue_length"
    assert args.local_reward_fn == "queue_length"
    assert args.global_reward_weight == 0.0
    assert list(args.state_features) == ["lane_vehicle_count", "lane_waiting", "phase_onehot"]
    assert args.fixed_time_k == 4
    assert args.thread_num == 1
    assert args.base_seed == 1000
    assert args.episodes == 5
    assert args.policy == "maxpressure"


def test_collect_style_args_never_create_the_sentinel_out_dir(tmp_path: Path) -> None:
    """The parser requires ``--out-dir``; this module logs no corpus and must not make one."""
    from offline import parity

    sentinel = tmp_path / "never-written"
    sar.collect_style_args(
        "sumo", "random", parity.DECLARED_PARITY_SUMOCFG, sentinel_out_dir=sentinel
    )

    assert not sentinel.exists()
    assert sorted(tmp_path.iterdir()) == []


def test_p7_0_horizon_att_reads_five_episodes_in_manifest_order(output_root: Path) -> None:
    """The reproduction targets, read from the npz and not from the gate artifact."""
    values = sar.p7_0_horizon_att("sumo__maxpressure", output_root)

    assert len(values) == 5
    assert values[0] == 355.7984313964844
    assert values[-1] == 378.1499938964844


def test_p7_0_horizon_att_refuses_an_unknown_cell(output_root: Path) -> None:
    """A typo in a cell name must not silently return an empty tuple."""
    with pytest.raises(KeyError, match="cell|unknown"):
        sar.p7_0_horizon_att("sumo__nonesuch", output_root)


# ----------------------------------------------------------------------
# 10. The env subclass -- SUMO-backed
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _traci_available(), reason="traci not importable")
def test_the_observer_keeps_driving_the_frozen_metrics_hook() -> None:
    """⭐ ``SumoEnv._simulate`` calls ``metrics.on_sim_step()`` after every step.

    Without it ``_drain_arrivals`` sees only the final step's arrivals and ``att_sumo_env`` becomes
    a different quantity -- silently. The live equality in the smoke test is the real check; this
    pins the call site so its removal cannot pass review unnoticed.
    """
    import inspect

    source = inspect.getsource(sar.sumo_observer_env_class()._simulate)

    assert "on_sim_step" in source
    assert "simulationStep" in source


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
def test_the_observer_env_is_built_like_the_frozen_make_env() -> None:
    """A settings key dropped here would change the measurement, not raise.

    ⚠️ The two envs are built SEQUENTIALLY and never at once: traci's default connection is a
    process-wide singleton, so holding both open raises "Connection 'default' is already active".
    """
    from experiments.envs import make_env
    from offline import parity
    from offline.collect import _build_env_spec

    args = sar.collect_style_args(
        "sumo", "maxpressure", parity.DECLARED_PARITY_SUMOCFG, sentinel_out_dir="/nonexistent"
    )
    spec = _build_env_spec(args)

    def _fingerprint(env: Any) -> dict[str, Any]:
        return {
            "max_steps": env.max_steps,
            "delta_time": env.delta_time,
            "phase_control_cls": env._phase_control_cls,
            "control_mode": env.control_mode,
            "intersections": [ix.id for ix in env.intersections],
            "incoming_lanes": [tuple(ix.incoming_lanes) for ix in env.intersections],
            "metric_names": list(env._metric_names),
            "action_space": repr(env.action_space),
            "state_widths": [len(ix.incoming_lanes) * 2 + ix.num_phases for ix in env.intersections],
        }

    reference = make_env(spec)
    try:
        expected = _fingerprint(reference)
        reference_type = type(reference)
    finally:
        reference.close()

    observed_env = sar.make_observer_sumo_env(
        parity.DECLARED_PARITY_SUMOCFG, spec.settings, halting_check=False
    )
    try:
        assert _fingerprint(observed_env) == expected
        assert isinstance(observed_env, reference_type)
    finally:
        observed_env.close()
