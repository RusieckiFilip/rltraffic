"""P7.3d B.8 (PREREGISTRATION A23): the collision record, and the rule that refuses ONLY an unexplained teleport.

Written against ``BRIEF_39`` Amendments B.8 and B.8.1 and the plan's stage 6 (``docs/plans/p7.3d.md``
§27-§34).  A15(c)'s ``time-to-teleport -1`` disables SUMO's waiting-time ("jam") teleport and NOT its
collision teleport (``collision.action = teleport`` by default): attempt 1 refused two cells for one
collision-teleport each.  A23 keeps such a cell, RECORDS every collision, and refuses only a teleport
that no collision of the SAME simulation step explains.

Four layers, each where a mutant of it dies cheapest:

* **the recorder on a stubbed simulation** -- steps fed to ``SumoObservationRecorder.observe``; the
  same-step match and the collider's fate, read from the arrived and ending-teleport lists;
* **the seam** -- ``PerSecondSumoObserver._simulate`` on a stub TraCI connection, built without a SUMO
  process: the step's ``getCollisions()`` and ``getEndingTeleportIDList()`` reach the SAME ``observe``
  call as its teleport list (the e2e cannot show this: draw 5 has no collision);
* **the payload** -- ``validate_cell_payload`` on ``p7.3d-grid4x4/1.1`` chunks (A23(c)(i), B.8-2(3));
* **D8 (B.8.1-2)** -- a FORCED collision on a TOY network through the REAL ``_simulate`` and real SUMO:
  everything in ``tmp_path``, a fixed seed, ``time-to-teleport -1`` and NO ``collision.*`` option (the
  registered regime), no scenario file of this project touched.  Measured first by
  ``output/p7_3d_runs/b8/d8_toy_measurement_prototype.txt``.

GATES: ``traci`` for the seam and D8; the ``sumo`` and ``netconvert`` binaries for D8, each named.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

import offline.sumo_att_reference as sar
import offline.transfer_curve as tcv
from tests.test_p7_3d_campaign_path import _grid_cell, _grid_payload

ARRIVED = "arrived_at_collision_step"
PUT_BACK = "put_back"
IN_TRANSIT = "in_transit_at_horizon"
UNREGISTERED = "unregistered"


def _traci_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return True


needs_traci = pytest.mark.skipif(not _traci_available(), reason="traci is not importable")


# ==================================================================================
# The recorder on a stubbed simulation -- the same-step match and the fate read
# ==================================================================================
_IDS = ("a", "b", "c", "d")


def _recorder(ids: tuple[str, ...] = _IDS) -> sar.SumoObservationRecorder:
    intended = sar.IntendedDepartures(by_id={vid: 0.0 for vid in ids}, source="synthetic")
    return sar.SumoObservationRecorder(delta_time=10.0, intended_departures=intended, step_length=1.0)


def _collision(collider: str, victim: str, *, lane: str = "L_0", pos: float = 12.5) -> dict[str, Any]:
    """One collision as ``collision_facts`` records it: TraCI's own attribute names."""
    return {
        "collider": collider, "victim": victim, "colliderType": "cf_parity", "victimType": "cf_parity",
        "colliderSpeed": 6.25, "victimSpeed": 0.0, "type": "collision", "lane": lane, "pos": pos,
    }


def _feed(recorder: sar.SumoObservationRecorder, steps: dict[float, dict[str, Any]], *, last: float) -> sar.SumoObservationRecorder:
    """Every vehicle departs at ``t = 1``; each later snapshot as *steps* names it.

    ``absent`` removes a vehicle from the snapshot's presence (a TELEPORTING vehicle is absent from
    ``vehicle.getIDList()`` -- measured on SUMO 1.27.1, D8) and ``back`` restores it.
    """
    present: set[str] = set()
    for tick in range(1, int(last) + 1):
        now = float(tick)
        spec = steps.get(now, {})
        departed = list(recorder.intended.by_id) if tick == 1 else []
        present.update(departed)
        arrived = list(spec.get("arrived", ()))
        present.difference_update(arrived)
        present.difference_update(spec.get("absent", ()))
        present.update(spec.get("back", ()))
        recorder.observe(
            sim_time=now,
            present_ids=sorted(present),
            departed_ids=departed,
            arrived_ids=arrived,
            pending_ids=[],
            departure_facts={vid: (0.0, 0.0) for vid in departed},
            teleport_start_ids=list(spec.get("teleports", ())),
            collisions=list(spec.get("collisions", ())),
            ending_teleport_ids=list(spec.get("ending", ())),
        )
    return recorder


def _event(time: float, collision: dict[str, Any], fate: str, fate_time: float | None = None) -> dict[str, Any]:
    return {"time": time, **collision, "collider_fate": fate, "collider_fate_time": fate_time}


def test_a_collision_and_its_colliders_teleport_in_one_step_are_explained_and_the_arrival_is_the_fate() -> None:
    """A23(c)(i): explained iff the teleporting vehicle is the collider or the victim of a collision
    reported in the SAME step.  Both attempt-1 events: the collider on its last edge, arrived at once."""
    record = _feed(
        _recorder(), {2.0: {"collisions": [_collision("a", "b")], "teleports": ["a"], "arrived": ["a"]}}, last=3.0
    ).collision_record()
    assert record == {
        "collisions": [_event(2.0, _collision("a", "b"), ARRIVED)],
        "teleports": [{"time": 2.0, "vehicle": "a"}],
        "n_collisions": 1,
        "n_explained_teleports": 1,
        "n_unexplained_teleports": 0,
        "vanished_ids": [],
    }


def test_a_victims_teleport_is_explained_and_a_collider_neither_teleported_nor_arrived_is_unregistered() -> None:
    """The VICTIM is the other party the match names (A23(c)(i)).  The collider's fate is then none of
    A23(c)(iii)'s three: recorded as unregistered, never dropped (B.8.1-2, D4)."""
    record = _feed(
        _recorder(),
        {
            2.0: {"collisions": [_collision("a", "b")], "teleports": ["b"], "absent": ["b"]},
            3.0: {"ending": ["b"], "back": ["b"]},
        },
        last=4.0,
    ).collision_record()
    assert record["n_explained_teleports"] == 1 and record["n_unexplained_teleports"] == 0
    assert record["teleports"] == [{"time": 2.0, "vehicle": "b"}]
    assert record["collisions"] == [_event(2.0, _collision("a", "b"), UNREGISTERED)]


def test_a_teleport_one_step_after_its_collision_is_unexplained() -> None:
    """SAME step, not any step: a teleport at t + 1 for a collision at t is unexplained."""
    record = _feed(
        _recorder(),
        {
            2.0: {"collisions": [_collision("a", "b")]},
            3.0: {"teleports": ["a"], "absent": ["a"]},
            4.0: {"ending": ["a"], "back": ["a"]},
        },
        last=5.0,
    ).collision_record()
    assert record["n_explained_teleports"] == 0 and record["n_unexplained_teleports"] == 1
    assert record["teleports"] == [{"time": 3.0, "vehicle": "a"}]
    assert record["collisions"][0]["collider_fate"] == UNREGISTERED


def test_a_teleport_with_no_collision_at_all_is_unexplained() -> None:
    record = _feed(
        _recorder(), {2.0: {"teleports": ["c"], "absent": ["c"]}, 3.0: {"ending": ["c"], "back": ["c"]}}, last=4.0
    ).collision_record()
    assert record["collisions"] == [] and record["n_collisions"] == 0
    assert record["n_explained_teleports"] == 0 and record["n_unexplained_teleports"] == 1


def test_a_collider_put_back_later_or_in_the_same_step_records_when() -> None:
    """A23(c)(iii): a collider on an earlier edge is put back on the first later edge with room -- read
    from ``getEndingTeleportIDList``, at the first step t' >= t (D8 measured both t' == t and t' > t)."""
    later = _feed(
        _recorder(),
        {
            2.0: {"collisions": [_collision("a", "b")], "teleports": ["a"], "absent": ["a"]},
            5.0: {"ending": ["a"], "back": ["a"]},
            7.0: {"arrived": ["a"]},
        },
        last=8.0,
    ).collision_record()
    assert later["collisions"] == [_event(2.0, _collision("a", "b"), PUT_BACK, 5.0)]
    assert later["n_explained_teleports"] == 1 and later["vanished_ids"] == []

    same = _feed(
        _recorder(),
        {2.0: {"collisions": [_collision("a", "b")], "teleports": ["a"], "ending": ["a"]}},
        last=3.0,
    ).collision_record()
    assert same["collisions"] == [_event(2.0, _collision("a", "b"), PUT_BACK, 2.0)]


def test_a_collider_still_teleporting_at_the_horizon_is_in_transit_and_vanished() -> None:
    """Neither arrived nor put back by the last snapshot: in transit, and -- absent from the snapshot,
    as a teleporting vehicle is -- counted by ``n_vanished_without_arrival``, whose ids are recorded."""
    recorder = _feed(
        _recorder(), {2.0: {"collisions": [_collision("a", "b")], "teleports": ["a"], "absent": ["a"]}}, last=6.0
    )
    record = recorder.collision_record()
    assert record["collisions"] == [_event(2.0, _collision("a", "b"), IN_TRANSIT)]
    assert record["vanished_ids"] == ["a"]
    assert recorder.vanished_ids == ("a",) and recorder.n_vanished_without_arrival == 1


def test_a_teleported_collider_that_arrives_later_without_being_put_back_is_unregistered() -> None:
    record = _feed(
        _recorder(),
        {2.0: {"collisions": [_collision("a", "b")], "teleports": ["a"], "absent": ["a"]}, 4.0: {"arrived": ["a"]}},
        last=5.0,
    ).collision_record()
    assert record["collisions"] == [_event(2.0, _collision("a", "b"), UNREGISTERED)]


def test_two_collisions_in_one_step_are_matched_each_in_that_step_and_clear_resets_the_record() -> None:
    recorder = _feed(
        _recorder(),
        {
            2.0: {
                "collisions": [_collision("a", "b"), _collision("c", "d", lane="M_0", pos=40.0)],
                "teleports": ["a", "c"],
                "arrived": ["a", "c"],
            }
        },
        last=3.0,
    )
    record = recorder.collision_record()
    assert record["collisions"] == [
        _event(2.0, _collision("a", "b"), ARRIVED),
        _event(2.0, _collision("c", "d", lane="M_0", pos=40.0), ARRIVED),
    ]
    assert record["n_explained_teleports"] == 2 and record["n_collisions"] == 2
    recorder.clear()
    assert recorder.collision_record() == {
        "collisions": [], "teleports": [], "n_collisions": 0, "n_explained_teleports": 0,
        "n_unexplained_teleports": 0, "vanished_ids": [],
    }


def test_an_observation_without_the_new_reads_keeps_its_old_meaning() -> None:
    """Every existing caller omits the two new keywords: its counters are unchanged, its record empty
    of collisions -- and a teleport it reports is, having no collision beside it, unexplained."""
    recorder = _recorder(("a",))
    recorder.observe(sim_time=1.0, present_ids=["a"], departed_ids=["a"], arrived_ids=[], pending_ids=[],
                     departure_facts={"a": (0.0, 0.0)}, teleport_start_ids=["a"])
    assert recorder.n_teleport_events == 1
    record = recorder.collision_record()
    assert record["collisions"] == [] and record["n_unexplained_teleports"] == 1


@needs_traci
def test_collision_facts_reads_tracis_own_attribute_names() -> None:
    """D3: TraCI's ``Collision`` recorded under exactly its own names, in ``COLLISION_FACT_KEYS`` order."""
    from traci._simulation import Collision

    collision = Collision("628", "969", "cf_parity", "cf_parity", 6.25, 0.0, "collision", "D0right0_0", 12.5)
    facts = sar.collision_facts(collision)
    assert list(facts) == list(sar.COLLISION_FACT_KEYS)
    assert facts == _collision("628", "969", lane="D0right0_0", pos=12.5) | {"colliderSpeed": 6.25}
    assert sar.COLLISION_FACT_KEYS == (
        "collider", "victim", "colliderType", "victimType", "colliderSpeed", "victimSpeed", "type", "lane", "pos",
    )
    assert sar.COLLIDER_FATES == (ARRIVED, PUT_BACK, IN_TRANSIT) and sar.UNREGISTERED_FATE == UNREGISTERED


# ==================================================================================
# The seam -- PerSecondSumoObserver._simulate on a stub TraCI connection
# ==================================================================================
class _StubSimulation:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.index = -1
        self.calls: dict[str, int] = {}

    def _now(self) -> dict[str, Any]:
        return self.script[self.index]

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def getTime(self) -> float:
        return float(self._now()["t"])

    def getDepartedIDList(self) -> tuple[str, ...]:
        return tuple(self._now().get("departed", ()))

    def getArrivedIDList(self) -> tuple[str, ...]:
        return tuple(self._now().get("arrived", ()))

    def getPendingVehicles(self) -> tuple[str, ...]:
        return ()

    def getStartingTeleportIDList(self) -> tuple[str, ...]:
        self._count("getStartingTeleportIDList")
        return tuple(self._now().get("starts", ()))

    def getEndingTeleportIDList(self) -> tuple[str, ...]:
        self._count("getEndingTeleportIDList")
        return tuple(self._now().get("ends", ()))

    def getCollisions(self) -> tuple[Any, ...]:
        self._count("getCollisions")
        return tuple(self._now().get("collisions", ()))


class _StubVehicle:
    def __init__(self, simulation: _StubSimulation) -> None:
        self.simulation = simulation

    def getIDList(self) -> tuple[str, ...]:
        return tuple(self.simulation._now().get("present", ()))

    def getDeparture(self, vid: str) -> float:
        return float(self.simulation._now()["t"]) - 1.0

    def getDepartDelay(self, vid: str) -> float:
        return 0.0


class _StubConnection:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.simulation = _StubSimulation(script)
        self.vehicle = _StubVehicle(self.simulation)

    def simulationStep(self) -> None:
        self.simulation.index += 1


@needs_traci
def test_simulate_hands_the_steps_collisions_and_ending_teleports_to_the_same_observation() -> None:
    """Seam 1 (plan §27): the ONE per-second loop both env paths run.  Each step's ``getCollisions()``
    and ``getEndingTeleportIDList()`` are read once, from the same snapshot as its teleport list, and
    reach the recorder in that snapshot's ``observe`` call.  *Mutations:* ``getCollisions`` ignored;
    ``getEndingTeleportIDList`` ignored -> this dies."""
    from traci._simulation import Collision

    first = Collision("a", "b", "cf_parity", "cf_parity", 6.25, 0.0, "collision", "L_0", 12.5)
    second = Collision("c", "b", "cf_parity", "cf_parity", 5.0, 0.0, "collision", "M_0", 30.0)
    script = [
        {"t": 1.0, "departed": ["a", "b", "c"], "present": ["a", "b", "c"]},
        {"t": 2.0, "collisions": [first], "starts": ["a"], "arrived": ["a"], "present": ["b", "c"]},
        {"t": 3.0, "collisions": [second], "starts": ["c"], "ends": ["c"], "present": ["b", "c"]},
    ]
    stub = _StubConnection(script)
    env = object.__new__(sar.sumo_observer_env_class())
    env._sumo = stub
    env._metrics = None
    env.halting_check = False
    env.recorder = sar.SumoObservationRecorder(
        delta_time=10.0,
        intended_departures=sar.IntendedDepartures(by_id={"a": 0.0, "b": 0.0, "c": 0.0}, source="stub"),
        step_length=1.0,
    )
    env._simulate(3)
    record = env.recorder.collision_record()
    assert record["collisions"] == [
        _event(2.0, sar.collision_facts(first), ARRIVED),
        _event(3.0, sar.collision_facts(second), PUT_BACK, 3.0),
    ]
    assert record["teleports"] == [{"time": 2.0, "vehicle": "a"}, {"time": 3.0, "vehicle": "c"}]
    assert record["n_explained_teleports"] == 2 and record["n_unexplained_teleports"] == 0
    for name in ("getCollisions", "getEndingTeleportIDList", "getStartingTeleportIDList"):
        assert stub.simulation.calls.get(name) == 3, f"{name} must be read once per simulated second"


# ==================================================================================
# The payload -- validate_cell_payload on p7.3d-grid4x4/1.1 chunks (A23(c)(i), B.8-2(3))
# ==================================================================================
def _chunk_event(
    collider: str = "628", victim: str = "969", *, time: float = 2299.0, fate: str = ARRIVED,
    fate_time: float | None = None,
) -> dict[str, Any]:
    return _event(time, _collision(collider, victim, lane="D0right0_0"), fate, fate_time)


def _payload(**overrides: Any) -> dict[str, Any]:
    return _grid_payload(_grid_cell("anchor", "fixedtime", 1020), config_sha="c" * 64, routes_sha="r" * 64,
                         **overrides)


def _one_event(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "n_teleports": 1,
        "teleports": [{"time": 2299.0, "vehicle": "628"}],
        "collisions": [_chunk_event()],
        "n_collisions": 1,
        "n_explained_teleports": 1,
        "n_unexplained_teleports": 0,
    }
    fields.update(overrides)
    return _payload(**fields)


def test_a_teleport_explained_by_its_collider_or_its_victim_is_kept() -> None:
    """A23(c)(ii): the collision is RECORDED and the cell KEPT."""
    tcv.validate_cell_payload(_one_event())
    tcv.validate_cell_payload(_one_event(teleports=[{"time": 2299.0, "vehicle": "969"}]))
    assert tcv.GRID4X4_ARTIFACT_FORMAT_VERSION == "p7.3d-grid4x4/1.1"
    assert _payload()["format_version"] == "p7.3d-grid4x4/1.1"
    tcv.validate_cell_payload(_payload())  # the empty record: no collision, no teleport


def test_an_unexplained_teleport_refuses_the_cell() -> None:
    """A23(c)(i): a teleport no same-step collision explains, of any kind, still REFUSES.
    *Mutation:* the unexplained-teleport check removed -> this dies."""
    with pytest.raises(ValueError, match="unexplained"):
        tcv.validate_cell_payload(
            _payload(n_teleports=1, teleports=[{"time": 100.0, "vehicle": "5"}], n_unexplained_teleports=1)
        )


def test_a_teleport_whose_collision_is_in_another_step_refuses_even_when_the_counts_claim_otherwise() -> None:
    """The chunk's own two lists re-derive the match (D2): counts claiming "explained" over a teleport
    one second after its collision are refused.  *Mutation:* the match loosened to any step -> dies."""
    with pytest.raises(ValueError, match="same"):
        tcv.validate_cell_payload(_one_event(teleports=[{"time": 2300.0, "vehicle": "628"}]))


def test_the_teleport_counts_must_add_up_and_match_the_lists() -> None:
    with pytest.raises(ValueError, match="n_explained_teleports"):
        tcv.validate_cell_payload(_one_event(n_unexplained_teleports=1))
    with pytest.raises(ValueError, match="teleports"):
        tcv.validate_cell_payload(_one_event(n_teleports=2))
    with pytest.raises(ValueError, match="n_collisions"):
        tcv.validate_cell_payload(_one_event(n_collisions=2))


def test_a_vanished_vehicle_no_collision_explains_refuses_and_one_party_to_a_collision_is_kept() -> None:
    """A23(c)(i): departed, never arrived, absent at the horizon WITHOUT being party to a recorded
    collision REFUSES -- a check NEW in 1.1 (finding 1).  *Mutation:* the vanished check removed -> dies."""
    with pytest.raises(ValueError, match="vanished"):
        tcv.validate_cell_payload(_payload(n_vanished_without_arrival=1, vanished_ids=["77"]))
    with pytest.raises(ValueError, match="vanished"):
        tcv.validate_cell_payload(_payload(n_vanished_without_arrival=1, vanished_ids=[]))
    tcv.validate_cell_payload(
        _one_event(collisions=[_chunk_event(fate=IN_TRANSIT)], n_vanished_without_arrival=1, vanished_ids=["628"])
    )


def test_an_unregistered_fate_stops_the_campaign_for_a_ruling() -> None:
    """B.8.1-2 (D4): A STOP, NEVER AN EXCLUSION -- the message names A23 and says the campaign stops."""
    with pytest.raises(ValueError, match=r"A23.*stops"):
        tcv.validate_cell_payload(_one_event(collisions=[_chunk_event(fate=UNREGISTERED)]))


def test_a_grid4x4_chunk_without_the_record_or_at_1_0_is_refused() -> None:
    """B.8-2(3): the record is required, and ``report`` accepts ONLY 1.1 for grid4x4 -- so no attempt-1
    chunk can reach an artifact."""
    missing = _payload()
    missing.pop("collisions")
    with pytest.raises(ValueError, match="collisions"):
        tcv.validate_cell_payload(missing)
    with pytest.raises(ValueError, match="format_version"):
        tcv.validate_cell_payload(_payload(format_version="p7.3d-grid4x4/1.0"))


# ==================================================================================
# D8 (B.8.1-2) -- a FORCED collision on a TOY network: real SUMO, the REAL _simulate
# ==================================================================================
_SUMO = shutil.which("sumo")
_NETCONVERT = shutil.which("netconvert")
needs_sumo = pytest.mark.skipif(_SUMO is None, reason="the sumo binary is not on PATH")
needs_netconvert = pytest.mark.skipif(_NETCONVERT is None, reason="the netconvert binary is not on PATH")

_TOY_CONFIG = """<configuration>
  <input><net-file value="toy.net.xml"/><route-files value="toy.rou.xml"/></input>
  <time><begin value="0"/><end value="600"/><step-length value="1"/></time>
  <processing><time-to-teleport value="-1"/></processing>
</configuration>
"""


def _toy(tmp_path: Path, *, middle_edges_short: bool, route: str, blockers: bool) -> Path:
    """A straight one-lane road ``e0 (200 m) -> e1 -> e2 [-> e3]``, built by netconvert in *tmp_path*.

    With ``blockers`` every later edge is 10 m long and held by a stopped vehicle, so a collider on
    ``e0`` finds no room and stays teleporting across snapshots (D8's case (d)).
    """
    xs = [0, 200, 210, 220, 230] if middle_edges_short else [0, 200, 400, 600]
    nodes = "".join(f'  <node id="n{i}" x="{x}" y="0" type="priority"/>\n' for i, x in enumerate(xs))
    edges = "".join(
        f'  <edge id="e{i}" from="n{i}" to="n{i + 1}" numLanes="1" speed="13.89"/>\n' for i in range(len(xs) - 1)
    )
    (tmp_path / "toy.nod.xml").write_text(f"<nodes>\n{nodes}</nodes>\n", encoding="utf-8")
    (tmp_path / "toy.edg.xml").write_text(f"<edges>\n{edges}</edges>\n", encoding="utf-8")
    subprocess.run(
        [str(_NETCONVERT), "--node-files", "toy.nod.xml", "--edge-files", "toy.edg.xml",
         "--output-file", "toy.net.xml", "--no-turnarounds", "true"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    blocking = ""
    if blockers:
        tail = route.split()[1:]
        for index, edge in enumerate(tail, start=1):
            rest = " ".join(tail[index - 1:])
            blocking += (
                f'  <vehicle id="B{index}" type="car" depart="0" departPos="4"><route edges="{rest}"/>'
                f'<stop lane="{edge}_0" endPos="9" duration="10000"/></vehicle>\n'
            )
    (tmp_path / "toy.rou.xml").write_text(
        "<routes>\n"
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="13.89"/>\n'
        f'  <route id="r" edges="{route}"/>\n'
        f"{blocking}"
        '  <vehicle id="L" type="car" route="r" depart="0"/>\n'
        '  <vehicle id="F" type="car" route="r" depart="4"/>\n'
        "</routes>\n",
        encoding="utf-8",
    )
    (tmp_path / "toy.sumocfg").write_text(_TOY_CONFIG, encoding="utf-8")
    return tmp_path / "toy.sumocfg"


def _roll_toy(config: Path, *, stop_after_collision: float | None = None) -> tuple[sar.SumoObservationRecorder, dict[str, Any]]:
    """The REAL ``_simulate`` one step at a time on a real SUMO connection; F is made to ram L.

    Returns the recorder and what the test reads beside it: the collision step, whether F was in
    ``vehicle.getIDList()`` at each snapshot while it teleported, and the configuration SUMO reports.
    """
    import traci

    label = f"b8_d8_{uuid.uuid4().hex[:8]}"
    traci.start([str(_SUMO), "-c", str(config), "--seed", "42", "--no-step-log", "true"], label=label)
    conn = traci.getConnection(label)
    facts: dict[str, Any] = {"presence_while_teleporting": [], "collision_time": None}
    try:
        facts["collision_action"] = conn.simulation.getOption("collision.action")
        facts["time_to_teleport"] = conn.simulation.getOption("time-to-teleport")
        env = object.__new__(sar.sumo_observer_env_class())
        env._sumo = conn
        env._metrics = None
        env.halting_check = False
        env.recorder = sar.SumoObservationRecorder(
            delta_time=10.0,
            intended_departures=sar.read_intended_departures(config.parent / "toy.rou.xml"),
            step_length=float(conn.simulation.getDeltaT()),
        )
        stopped = False
        teleporting = False
        for _ in range(200):
            env._simulate(1)
            now = float(conn.simulation.getTime())
            present = set(conn.vehicle.getIDList())
            if "F" in conn.simulation.getDepartedIDList():
                conn.vehicle.setSpeedMode("F", 0)
                conn.vehicle.setSpeed("F", 13.89)
            if "L" in present and not stopped and conn.vehicle.getLanePosition("L") > 100.0:
                conn.vehicle.setSpeed("L", 0.0)
                stopped = True
            if facts["collision_time"] is None and conn.simulation.getCollisions():
                facts["collision_time"] = now
            if "F" in conn.simulation.getStartingTeleportIDList():
                teleporting = True
            if "F" in conn.simulation.getEndingTeleportIDList():
                teleporting = False
            if teleporting:
                facts["presence_while_teleporting"].append("F" in present)
            if "F" in conn.simulation.getArrivedIDList():
                break
            if (
                stop_after_collision is not None
                and facts["collision_time"] is not None
                and now >= facts["collision_time"] + stop_after_collision
            ):
                break
        return env.recorder, facts
    finally:
        conn.close()


@needs_sumo
@needs_netconvert
@needs_traci
def test_d8_a_collider_on_its_last_edge_collides_teleports_and_arrives_in_one_snapshot(tmp_path: Path) -> None:
    """A1 and A2 (plan §32) on real SUMO: the collision, the collider's teleport start and its arrival
    are in ONE snapshot -- attempt 1's shape.  Explained; fate ``arrived_at_collision_step``."""
    recorder, facts = _roll_toy(_toy(tmp_path, middle_edges_short=False, route="e0", blockers=False))
    assert facts["collision_action"] == "teleport" and facts["time_to_teleport"] == "-1"
    record = recorder.collision_record()
    assert [(c["collider"], c["victim"], c["time"], c["collider_fate"]) for c in record["collisions"]] == [
        ("F", "L", facts["collision_time"], ARRIVED)
    ]
    assert record["teleports"] == [{"time": facts["collision_time"], "vehicle": "F"}]
    assert record["n_explained_teleports"] == 1 and record["n_unexplained_teleports"] == 0
    assert record["vanished_ids"] == []


@needs_sumo
@needs_netconvert
@needs_traci
def test_d8_a_collider_on_an_earlier_edge_with_room_is_put_back_in_the_collision_step(tmp_path: Path) -> None:
    recorder, facts = _roll_toy(_toy(tmp_path, middle_edges_short=False, route="e0 e1 e2", blockers=False))
    record = recorder.collision_record()
    assert [(c["collider"], c["collider_fate"], c["collider_fate_time"]) for c in record["collisions"]] == [
        ("F", PUT_BACK, facts["collision_time"])
    ]
    assert record["n_explained_teleports"] == 1 and record["n_unexplained_teleports"] == 0
    assert facts["presence_while_teleporting"] == [], "put back within the step: no snapshot sees it teleport"


@needs_sumo
@needs_netconvert
@needs_traci
def test_d8_a_collider_with_no_room_teleports_across_snapshots_absent_from_getidlist(tmp_path: Path) -> None:
    """B.8.1-2's measurement, pinned: while a collider teleports it is NOT in ``vehicle.getIDList()``
    (every snapshot until it is put back), so a collider still teleporting at the horizon IS counted by
    ``n_vanished_without_arrival`` -- A23(c)(iii)'s sentence, confirmed on SUMO 1.27.1 (finding 2)."""
    (tmp_path / "full").mkdir()
    recorder, facts = _roll_toy(_toy(tmp_path / "full", middle_edges_short=True, route="e0 e1 e2 e3", blockers=True))
    record = recorder.collision_record()
    [event] = record["collisions"]
    assert event["collider"] == "F" and event["collider_fate"] == PUT_BACK
    assert event["collider_fate_time"] > event["time"], "the teleport spanned snapshots"
    assert len(facts["presence_while_teleporting"]) >= 2
    assert not any(facts["presence_while_teleporting"]), "a teleporting vehicle is absent from getIDList"

    (tmp_path / "cut").mkdir()
    cut, cut_facts = _roll_toy(
        _toy(tmp_path / "cut", middle_edges_short=True, route="e0 e1 e2 e3", blockers=True), stop_after_collision=3.0
    )
    cut_record = cut.collision_record()
    assert [c["collider_fate"] for c in cut_record["collisions"]] == [IN_TRANSIT]
    assert cut_record["vanished_ids"] == ["F"] and cut.n_vanished_without_arrival == 1
