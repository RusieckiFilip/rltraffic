"""Tests for ``offline.flow_randomizer`` and the ``--flow-draw`` wiring in ``offline.collect``.

No simulator is required: every test operates on a small synthetic vehicle list or reads
a real ``scenarios/*/flow.json`` **read-only**.

The load-bearing tests are :func:`test_two_draws_hash_differently` and
:func:`test_draw_zero_render_is_byte_identical`.  The first is the acceptance bar for the
whole task -- distinct draws must produce distinct demand, because the absence of that
property is what made a 20-episode corpus one trajectory written 20 times.  The second
guards the opposite direction: draw 0 is the nominal control condition every experiment is
compared against, so it must not perturb the demand even cosmetically.

Critical quantities are recomputed by an independent route wherever possible (CLAUDE.md
§2): byte-identity is checked against the raw file bytes, digests are recomputed directly
with ``hashlib``, and :func:`test_render_sumo_reproduces_shipped_hangzhou_routes` checks
the CityFlow->SUMO edge mapping against a ``.rou.xml`` that ships in the repo and that
this code never writes.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.collect as collect
from offline.flow_randomizer import (
    FlowDraw,
    FlowRandomizer,
    sumo_begin_from_sumocfg,
)
from offline.trajectory_logger import LoggerStateError, TrajectoryLogger

# P1's simulator-free fixtures, reused unchanged so the rebind tests exercise the same
# env shape the logger was built against.
from tests.test_trajectory_logger import (
    IX_SPECS,
    FakeIntersection,
    FakeTrafficEnv,
    _run_episode,
)

REPO = Path(__file__).resolve().parents[1]

HANGZHOU = REPO / "scenarios/hangzhou_1x1_bc-tyc_18041610_1h"
HANGZHOU_FLOW = HANGZHOU / "flow.json"
HANGZHOU_ROU = HANGZHOU / "hangzhou_1x1_bc-tyc_18041610_1h.rou.xml"
HANGZHOU_SUMOCFG = HANGZHOU / "hangzhou_1x1_bc-tyc_18041610_1h.sumocfg"

COLOGNE3_FLOW = REPO / "scenarios/cologne3/cologne3_flow.json"
COLOGNE3_ROU = REPO / "scenarios/cologne3/cologne3.rou.xml"
COLOGNE3_SUMOCFG = REPO / "scenarios/cologne3/cologne3.sumocfg"
COLOGNE1_SUMOCFG = REPO / "scenarios/cologne1/cologne1.sumocfg"
GRID4X4_SUMOCFG = REPO / "scenarios/grid4x4/grid4x4.sumocfg"

# Every flow file in the repo, with the indent its bytes are actually written with.
# The pair is the point: a single hardcoded indent cannot reproduce both families, which
# is why the randomiser sniffs rather than assumes.
ALL_FLOWS: tuple[tuple[str, Path, int], ...] = (
    ("hangzhou", HANGZHOU_FLOW, 2),
    ("cologne1", REPO / "scenarios/cologne1/cologne1_flow.json", 4),
    ("cologne3", COLOGNE3_FLOW, 4),
    ("grid4x4", REPO / "scenarios/grid4x4/grid4x4_flow.json", 4),
)


# ----------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------

_VEHICLE = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.11,
    "headwayTime": 2.0,
}


def _entry(i: int, start: int) -> dict[str, Any]:
    """One insertion whose route is unique, so draws can be paired by identity.

    Pairing by list index would be wrong the moment thinning or the final sort moves an
    entry, and a test that silently compares the wrong pairs measures nothing.
    """
    return {
        "vehicle": dict(_VEHICLE),
        "route": [f"road_{i}_a", f"road_{i}_b"],
        "interval": 5,
        "startTime": start,
        "endTime": start,
    }


def _write_synthetic(
    path: Path, n: int, *, first: int = 300, spacing: int = 7, indent: int = 2
) -> list[dict[str, Any]]:
    """Write a synthetic flow file and return the entries it contains.

    ``first=300`` keeps every departure at least 10 sigma clear of zero for the default
    jitter, so the ``>= 0`` clip never fires and cannot bias a sigma measurement.
    """
    entries = [_entry(i, first + i * spacing) for i in range(n)]
    path.write_text(json.dumps(entries, indent=indent), encoding="utf-8")
    return entries


def _starts_by_route(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Map each vehicle's first edge to its departure -- **synthetic fixtures only**.

    Guarded rather than merely documented: real scenarios reuse routes heavily (hangzhou
    has 4 distinct first edges and 8 distinct full routes across 2021 vehicles), so on
    real data this mapping silently collapses thousands of vehicles into a handful of
    keys and any assertion built on it measures nothing.  Only ``_write_synthetic``
    guarantees the uniqueness this needs.
    """
    by_route = {entry["route"][0]: entry["startTime"] for entry in entries}
    assert len(by_route) == len(entries), (
        f"_starts_by_route needs unique routes but {len(entries)} entries collapsed to "
        f"{len(by_route)} keys; use a synthetic fixture, not a real scenario"
    )
    return by_route


def _digest(entries: list[dict[str, Any]]) -> str:
    """sha256 over a canonical rendering of the vehicle list, computed independently."""
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ----------------------------------------------------------------------
# 1-3. draw(0) is the identity, and the renderer proves it byte-for-byte
# ----------------------------------------------------------------------


def test_draw_zero_is_identity(tmp_path: Path) -> None:
    source = tmp_path / "flow.json"
    entries = _write_synthetic(source, 200)

    randomizer = FlowRandomizer(source)
    drawn, provenance = randomizer.draw(0)

    assert drawn == entries
    assert provenance.draw_id == 0
    assert provenance.n_vehicles == len(entries)
    # draw 0 applies the identity, so that is what must be recorded -- not the
    # constructor's settings, which would misdescribe the corpus.
    assert provenance.params == {
        "jitter_sigma_s": 0.0,
        "thin_p": 0.0,
        "volume_scale": 1.0,
    }


def test_draw_zero_returns_a_copy_not_the_internal_list(tmp_path: Path) -> None:
    source = tmp_path / "flow.json"
    _write_synthetic(source, 50)
    randomizer = FlowRandomizer(source)

    first, _ = randomizer.draw(0)
    first[0]["startTime"] = 999999
    first[0]["vehicle"]["length"] = 42.0
    second, _ = randomizer.draw(0)

    assert second[0]["startTime"] != 999999
    assert second[0]["vehicle"]["length"] != 42.0


@pytest.mark.parametrize("name,flow_path,indent", ALL_FLOWS, ids=[f[0] for f in ALL_FLOWS])
def test_draw_zero_render_is_byte_identical(
    name: str, flow_path: Path, indent: int, tmp_path: Path
) -> None:
    """Brief req 4, on all four real scenarios.

    Parametrised deliberately: hangzhou is written with ``indent=2`` and the other three
    with ``indent=4``, so a hardcoded indent would pass on one family and silently
    rewrite the other.
    """
    source_bytes = flow_path.read_bytes()

    randomizer = FlowRandomizer(flow_path)
    entries, _ = randomizer.draw(0)
    out = randomizer.render_cityflow(entries, tmp_path / "rendered.json")

    assert out.read_bytes() == source_bytes
    # And the indent really is the one we claim, so this test cannot pass by accident
    # on a file that happens to round-trip under the wrong setting.
    assert json.dumps(json.loads(source_bytes), indent=indent).encode() == source_bytes


def test_unreproducible_source_is_rejected_at_construction(tmp_path: Path) -> None:
    """A source whose formatting cannot be reproduced must refuse to construct.

    Otherwise draw 0 would silently stop being byte-identical and the nominal control
    condition would drift away from the file it claims to be.
    """
    entries = [_entry(0, 10), _entry(1, 20)]
    odd = tmp_path / "tabs.json"
    odd.write_text(json.dumps(entries, indent="\t"), encoding="utf-8")

    with pytest.raises(ValueError, match="reproduce"):
        FlowRandomizer(odd)


# ----------------------------------------------------------------------
# 4-5. Determinism, and distinctness that means something
# ----------------------------------------------------------------------


def test_aggregate_interval_flow_is_rejected(tmp_path: Path) -> None:
    """An aggregate flow must raise, not be silently collapsed to one vehicle.

    CityFlow expands ``startTime=0, endTime=3600, interval=5`` into 721 vehicles. Every
    transform here treats an entry as a single insertion and sets ``endTime`` to
    ``startTime``, so accepting such a file would cut demand by ~700x while the console
    truthfully reported the entry count and nothing raised -- the silent-wrong-number
    failure this repo exists to prevent.
    """
    aggregate = [_entry(i, 0) for i in range(3)]
    for entry in aggregate:
        entry["endTime"] = 3600

    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate"):
        FlowRandomizer(path)


def test_same_draw_id_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "flow.json"
    _write_synthetic(source, 500)

    a_entries, a_draw = FlowRandomizer(source).draw(3)
    b_entries, b_draw = FlowRandomizer(source).draw(3)

    assert a_entries == b_entries
    assert a_draw == b_draw


def test_different_draws_differ_substantially(tmp_path: Path) -> None:
    """Brief test 3: a *count* of differing vehicles, not a bare ``!=``.

    A single differing element would satisfy ``!=`` and prove nothing, so this asserts
    both that the surviving sets differ by many vehicles and that most of the vehicles
    present in both draws depart at different times.
    """
    from collections import Counter

    randomizer = FlowRandomizer(HANGZHOU_FLOW)
    one, _ = randomizer.draw(1)
    two, _ = randomizer.draw(2)

    # Multisets of (route, departure), because real scenarios reuse routes heavily --
    # hangzhou has only 8 distinct routes across 2021 vehicles, so any per-route keying
    # would collapse the population and measure nothing.
    a = Counter((tuple(e["route"]), e["startTime"]) for e in one)
    b = Counter((tuple(e["route"]), e["startTime"]) for e in two)
    differing = sum(((a - b) + (b - a)).values())

    assert differing > 500, (
        f"only {differing} of ~{len(one)} vehicles differ between draws 1 and 2; "
        "a bare != would pass on a single differing element and prove nothing"
    )


# ----------------------------------------------------------------------
# 6-9. The three transforms
# ----------------------------------------------------------------------


def test_jitter_is_nonnegative_and_sorted(tmp_path: Path) -> None:
    source = tmp_path / "flow.json"
    # Departures crowded against zero, so clipping is actually exercised.
    _write_synthetic(source, 400, first=0, spacing=1)

    entries, _ = FlowRandomizer(source, jitter_sigma_s=60.0).draw(5)
    starts = [entry["startTime"] for entry in entries]

    assert all(s >= 0 for s in starts)
    assert starts == sorted(starts)
    assert all(entry["endTime"] == entry["startTime"] for entry in entries)


def test_jitter_matches_sigma(tmp_path: Path) -> None:
    """The shift really is N(0, sigma), measured by pairing vehicles on their routes."""
    source = tmp_path / "flow.json"
    entries = _write_synthetic(source, 2000)

    drawn, _ = FlowRandomizer(
        source, jitter_sigma_s=30.0, thin_p=0.0, volume_scale=1.0
    ).draw(7)

    before = _starts_by_route(entries)
    after = _starts_by_route(drawn)
    assert set(before) == set(after), "no vehicle should be lost with thin_p=0"

    shift = np.array([after[r] - before[r] for r in before], dtype=np.float64)
    assert abs(float(shift.mean())) < 3.0
    assert 25.5 < float(shift.std()) < 34.5


def test_thinning_binomial_band(tmp_path: Path) -> None:
    source = tmp_path / "flow.json"
    _write_synthetic(source, 1000)

    entries, provenance = FlowRandomizer(
        source, thin_p=0.2, jitter_sigma_s=0.0, volume_scale=1.0
    ).draw(1)

    # thin_p is the DROP probability, so ~800 survive. Deliberately not 0.5, where
    # "drop with probability p" and "keep with probability p" are indistinguishable and
    # an inverted comparison would pass. The band is ~4.4 sigma (sigma = 12.65); the
    # inverted implementation would yield ~200 and fail loudly.
    assert 744 <= len(entries) <= 856
    assert provenance.n_vehicles == len(entries)


@pytest.mark.parametrize("scale,expected", [(1.5, 1500), (0.5, 500), (1.0, 1000)])
def test_volume_scale_changes_count(
    scale: float, expected: int, tmp_path: Path
) -> None:
    source = tmp_path / "flow.json"
    _write_synthetic(source, 1000)

    entries, provenance = FlowRandomizer(
        source, thin_p=0.0, jitter_sigma_s=5.0, volume_scale=scale
    ).draw(2)

    # With thinning off the count is exact, not a band.
    assert len(entries) == expected
    assert provenance.params["volume_scale"] == scale


def test_volume_scale_up_produces_distinct_departures(tmp_path: Path) -> None:
    """Duplicated vehicles must get fresh jitter, not be exact copies."""
    source = tmp_path / "flow.json"
    _write_synthetic(source, 200)

    entries, _ = FlowRandomizer(
        source, thin_p=0.0, jitter_sigma_s=30.0, volume_scale=2.0
    ).draw(4)

    assert len(entries) == 400
    from collections import Counter

    per_route = Counter(entry["route"][0] for entry in entries)
    duplicated = [route for route, count in per_route.items() if count > 1]
    assert duplicated, "volume_scale=2.0 must duplicate vehicles"

    identical = 0
    for route in duplicated:
        starts = [e["startTime"] for e in entries if e["route"][0] == route]
        if len(set(starts)) == 1:
            identical += 1
    assert identical < 0.25 * len(duplicated), (
        "duplicated vehicles kept identical departure times; jitter is not independent"
    )


def test_thinning_removes_different_vehicles_per_draw(tmp_path: Path) -> None:
    """Thinning must be seeded from ``draw_id``, not merely from ``base_seed``.

    Jitter is switched **off** so this cannot be satisfied by departure noise. Without
    it, every distinctness test in this file is dominated by jitter, and an
    implementation whose thinning ignored ``draw_id`` -- removing the *same* vehicles
    every draw -- would pass all of them. That is the whole point of the randomiser: the
    corpus needs different vehicles present, not just the same ones arriving later.
    """
    source = tmp_path / "flow.json"
    entries = _write_synthetic(source, 1000)
    all_routes = {entry["route"][0] for entry in entries}

    randomizer = FlowRandomizer(source, jitter_sigma_s=0.0, thin_p=0.2, volume_scale=1.0)
    removed = {}
    for draw_id in (1, 2, 3):
        drawn, _ = randomizer.draw(draw_id)
        removed[draw_id] = all_routes - {entry["route"][0] for entry in drawn}
        assert removed[draw_id], f"draw {draw_id} removed nothing at thin_p=0.2"

    for a, b in ((1, 2), (1, 3), (2, 3)):
        symmetric = removed[a] ^ removed[b]
        assert len(symmetric) > 100, (
            f"draws {a} and {b} removed near-identical vehicle sets "
            f"(symmetric difference {len(symmetric)}); thinning is not seeded by draw_id"
        )


def test_source_file_never_modified(tmp_path: Path) -> None:
    """Brief test 7, against a real repo scenario."""
    before = hashlib.sha256(HANGZHOU_FLOW.read_bytes()).hexdigest()

    randomizer = FlowRandomizer(HANGZHOU_FLOW)
    for draw_id in (0, 1, 2):
        entries, _ = randomizer.draw(draw_id)
        randomizer.render_cityflow(entries, tmp_path / f"d{draw_id}.json")

    after = hashlib.sha256(HANGZHOU_FLOW.read_bytes()).hexdigest()
    assert after == before


# ----------------------------------------------------------------------
# 10-11. Rendered CityFlow output
# ----------------------------------------------------------------------


def test_render_cityflow_schema_matches_source(tmp_path: Path) -> None:
    source_entries = json.loads(HANGZHOU_FLOW.read_bytes())
    randomizer = FlowRandomizer(HANGZHOU_FLOW)
    entries, _ = randomizer.draw(1)
    out = randomizer.render_cityflow(entries, tmp_path / "drawn.json")

    reloaded = json.loads(out.read_bytes())
    assert isinstance(reloaded, list) and reloaded

    source_keys = set(source_entries[0])
    source_vehicle_keys = set(source_entries[0]["vehicle"])
    for entry in reloaded:
        assert set(entry) == source_keys
        assert set(entry["vehicle"]) == source_vehicle_keys
        assert entry["endTime"] == entry["startTime"]
        # Types are preserved: hangzhou stores startTime and interval as ints, and a
        # corpus is traced back to its demand by file hash, so 5 must not become 5.0.
        assert isinstance(entry["startTime"], int)
        assert isinstance(entry["interval"], type(source_entries[0]["interval"]))


# ----------------------------------------------------------------------
# 12-14. SUMO rendering
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "sumocfg,expected",
    [
        (HANGZHOU_SUMOCFG, 0.0),
        (GRID4X4_SUMOCFG, 0.0),
        (COLOGNE1_SUMOCFG, 25200.0),
        (COLOGNE3_SUMOCFG, 25200.0),
    ],
    ids=["hangzhou", "grid4x4", "cologne1", "cologne3"],
)
def test_sumo_begin_from_sumocfg(sumocfg: Path, expected: float) -> None:
    assert sumo_begin_from_sumocfg(sumocfg) == expected


def test_sumo_begin_raises_when_absent(tmp_path: Path) -> None:
    """A missing <begin> must raise, never default to 0.0.

    Defaulting would render cologne's vehicles at t=0 while the simulation starts at
    25200, producing an empty run that still exits successfully.
    """
    cfg = tmp_path / "no_begin.sumocfg"
    cfg.write_text(
        '<configuration><input><net-file value="x.net.xml"/></input></configuration>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="begin"):
        sumo_begin_from_sumocfg(cfg)


def test_render_sumo_structure(tmp_path: Path) -> None:
    randomizer = FlowRandomizer(COLOGNE3_FLOW)
    entries, _ = randomizer.draw(1)
    offset = sumo_begin_from_sumocfg(COLOGNE3_SUMOCFG)

    out = randomizer.render_sumo(
        entries,
        tmp_path / "drawn.rou.xml",
        template_rou_path=COLOGNE3_ROU,
        depart_offset=offset,
    )

    root = ET.parse(out).getroot()
    assert root.tag == "routes"

    template_vtypes = [dict(v.attrib) for v in ET.parse(COLOGNE3_ROU).getroot().findall("vType")]
    rendered_vtypes = [dict(v.attrib) for v in root.findall("vType")]
    assert rendered_vtypes == template_vtypes, "the scenario's vType must be copied verbatim"

    vehicles = root.findall("vehicle")
    assert len(vehicles) == len(entries)

    departs = [float(v.get("depart")) for v in vehicles]
    assert departs == sorted(departs), "SUMO requires route files sorted by depart"
    assert departs == [float(e["startTime"]) + offset for e in entries]
    assert len({v.get("id") for v in vehicles}) == len(vehicles), "vehicle ids must be unique"

    for vehicle, entry in zip(vehicles, entries):
        assert vehicle.find("route").get("edges").split() == entry["route"]


def test_render_sumo_reproduces_shipped_hangzhou_routes(tmp_path: Path) -> None:
    """The CityFlow->SUMO edge mapping is the identity -- checked against a repo file.

    hangzhou's ``flow.json`` and its ``.rou.xml`` are 1:1 (2021 entries each, routes
    identical by index, departure offset 0), so rendering draw 0 must reproduce the
    shipped file's ``(depart, edges)`` sequence exactly.  This is the evidence claim C3
    rests on, and it is checked against a file this code never writes.
    """
    randomizer = FlowRandomizer(HANGZHOU_FLOW)
    entries, _ = randomizer.draw(0)
    out = randomizer.render_sumo(
        entries,
        tmp_path / "hz.rou.xml",
        template_rou_path=HANGZHOU_ROU,
        depart_offset=sumo_begin_from_sumocfg(HANGZHOU_SUMOCFG),
    )

    def pairs(path: Path) -> list[tuple[float, str]]:
        return [
            (float(v.get("depart")), v.find("route").get("edges"))
            for v in ET.parse(path).getroot().findall("vehicle")
        ]

    rendered = pairs(out)
    shipped = pairs(HANGZHOU_ROU)
    assert len(rendered) == len(shipped) == 2021
    assert rendered == shipped


# ----------------------------------------------------------------------
# 15-16. The acceptance bar, and provenance
# ----------------------------------------------------------------------


def test_two_draws_hash_differently() -> None:
    """THE ACCEPTANCE TEST.

    Engine seeds 1000 and 1001 produced byte-identical trajectories and identical
    ``episode_sha256`` because CityFlow reads its flow file once, in the constructor.
    This is the property whose absence made the corpus information-free.
    """
    randomizer = FlowRandomizer(HANGZHOU_FLOW)
    digests = {}
    for draw_id in (0, 1, 2, 3):
        entries, _ = randomizer.draw(draw_id)
        digests[draw_id] = _digest(entries)

    assert len(set(digests.values())) == 4, f"draws collided: {digests}"


def test_flow_draw_provenance(tmp_path: Path) -> None:
    source = tmp_path / "flow.json"
    entries = _write_synthetic(source, 300)
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    randomizer = FlowRandomizer(source, base_seed=4242)
    drawn, provenance = randomizer.draw(9)

    assert isinstance(provenance, FlowDraw)
    assert provenance.source_sha256 == expected_sha
    assert randomizer.source_sha256 == expected_sha
    assert provenance.seed == 4242 + 9
    assert provenance.n_vehicles == len(drawn)
    assert randomizer.n_source_vehicles == len(entries)


# ----------------------------------------------------------------------
# 17-18. TrajectoryLogger.rebind_env  (GATE-2 Q1)
# ----------------------------------------------------------------------


def _other_topology_env(*, ids: bool = False, phases: bool = False) -> FakeTrafficEnv:
    """A fake env whose topology deliberately disagrees with the frozen one."""
    env = FakeTrafficEnv()
    env.intersections = [
        FakeIntersection(
            f"{ix_id}_renamed" if ids else ix_id,
            n_actions + 1 if phases else n_actions,
        )
        for ix_id, _dim, n_actions in IX_SPECS
    ]
    return env


def test_rebind_env_keeps_one_manifest_across_draws(tmp_path: Path) -> None:
    """One logger, one manifest, across a fresh env per draw.

    A fresh logger per draw would rewrite manifest.json from an empty list and orphan
    the earlier draws' .npz files, so the sweep depends on this working.

    Named for what it proves. It deliberately does **not** assert that the two episodes
    differ: `FakeTrafficEnv`'s dynamics do not depend on the flow draw, and on real
    CityFlow two episodes of the same draw are genuinely identical (the engine seed
    changes nothing observable once demand is fixed). Asserting otherwise here would be
    asserting something false.
    """
    env_a = FakeTrafficEnv()
    logger = TrajectoryLogger(env_a, tmp_path, run_metadata={"scenario": "fake1x2"})
    _run_episode(env_a, logger, engine_seed=1000, flow_draw=0)

    env_b = FakeTrafficEnv()
    logger.rebind_env(env_b)
    # The rebind must actually take effect. With topologies necessarily identical (a
    # mismatch is rejected), nothing the logger writes can distinguish the two envs, so
    # the binding is asserted directly -- otherwise a rebind_env that validated and then
    # forgot to assign would pass this file untouched.
    assert logger._env is env_b

    _run_episode(env_b, logger, engine_seed=1000, flow_draw=1)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 2
    assert [e["flow_draw"] for e in manifest["episodes"]] == [0, 1]


def test_rebind_env_rejects_changed_ix_ids(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path, run_metadata={})
    _run_episode(env, logger, engine_seed=1000)

    with pytest.raises(LoggerStateError, match="ix_ids|intersection"):
        logger.rebind_env(_other_topology_env(ids=True))


def test_rebind_env_rejects_changed_action_counts(tmp_path: Path) -> None:
    """lane_ids are guarded run-level by lane_ids_sha256; n_actions were the open gap."""
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path, run_metadata={})
    _run_episode(env, logger, engine_seed=1000)

    with pytest.raises(LoggerStateError, match="n_actions|action"):
        logger.rebind_env(_other_topology_env(phases=True))


def test_rebind_env_rejects_changed_action_counts_before_first_episode(
    tmp_path: Path,
) -> None:
    """The n_actions guard must hold on the FIRST rebind, not only after an episode.

    This is exactly the call the collector makes: draw 0 rebinds before any ``on_reset``
    has run. An earlier version derived the expected counts from the previous episode, so
    the check was skipped precisely where it is used first.
    """
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path, run_metadata={})

    with pytest.raises(LoggerStateError, match="n_actions|action"):
        logger.rebind_env(_other_topology_env(phases=True))

    # ...and a matching env is still accepted at that same point.
    logger.rebind_env(FakeTrafficEnv())


def test_rebind_env_rejected_mid_episode(tmp_path: Path) -> None:
    env = FakeTrafficEnv()
    logger = TrajectoryLogger(env, tmp_path, run_metadata={})
    info = env.reset(seed=1000)
    logger.on_reset(info, engine_seed=1000, flow_draw=0)

    with pytest.raises(LoggerStateError, match="episode is still open"):
        logger.rebind_env(FakeTrafficEnv())


# ----------------------------------------------------------------------
# 19-21. collect.py wiring
# ----------------------------------------------------------------------


def test_flow_draw_flags_default_to_none() -> None:
    args = collect.build_parser().parse_args(
        ["--backend", "cityflow", "--env-config", "x.json",
         "--policy", "random", "--out-dir", "out"]
    )
    assert args.flow_draw is None
    assert args.flow_draws is None
    assert args.flow_draws_range is None


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], [None]),
        (["--flow-draw", "3"], [3]),
        (["--flow-draws", "1", "2", "5"], [1, 2, 5]),
        # Half-open [START, END), matching range(): 0 4 is four draws, not five.
        (["--flow-draws-range", "0", "4"], [0, 1, 2, 3]),
    ],
    ids=["none", "single", "explicit", "range"],
)
def test_resolve_draw_ids(flags: list[str], expected: list[int | None]) -> None:
    args = collect.build_parser().parse_args(
        ["--backend", "cityflow", "--env-config", "x.json",
         "--policy", "random", "--out-dir", "out", *flags]
    )
    assert collect._resolve_draw_ids(args) == expected


def test_flow_draw_flags_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # argparse raises SystemExit(2), whose str() is the exit code, so match= has nothing
    # to bind to; the message is pinned via capsys below instead.
    # hygiene: allow TH006 - SystemExit(2) from argparse carries no matchable message
    with pytest.raises(SystemExit):
        collect.build_parser().parse_args(
            ["--backend", "cityflow", "--env-config", "x.json",
             "--policy", "random", "--out-dir", "out",
             "--flow-draw", "1", "--flow-draws", "2", "3"]
        )
    # argparse raises SystemExit for *any* CLI error, so without pinning the message
    # this test would pass on a typo'd flag name and prove nothing -- which is exactly
    # how it passed vacuously during the red phase, before these flags existed.
    stderr = capsys.readouterr().err
    assert "not allowed with argument" in stderr
    assert "--flow-draw" in stderr


def test_repeated_draw_ids_are_refused() -> None:
    """Collecting one draw twice yields byte-identical episodes under a manifest
    claiming several draws -- the duplicate-corpus defect this task removes."""
    args = collect.build_parser().parse_args(
        ["--backend", "cityflow", "--env-config", "x.json",
         "--policy", "random", "--out-dir", "out", "--flow-draws", "0", "1", "0"]
    )
    with pytest.raises(SystemExit, match="repeated draw id"):
        collect._resolve_draw_ids(args)


@pytest.mark.parametrize("flags", [["--flow-draw", "-1"], ["--flow-draws-range", "-2", "3"]],
                         ids=["single", "range"])
def test_negative_draw_ids_are_refused(flags: list[str]) -> None:
    args = collect.build_parser().parse_args(
        ["--backend", "cityflow", "--env-config", "x.json",
         "--policy", "random", "--out-dir", "out", *flags]
    )
    with pytest.raises(SystemExit, match="must be >= 0"):
        collect._resolve_draw_ids(args)


@pytest.mark.parametrize("start,end", [("5", "5"), ("3", "1")], ids=["empty", "inverted"])
def test_degenerate_draw_range_is_refused(start: str, end: str) -> None:
    """A typo'd bound must not report success for a corpus that does not exist.

    Resolved to ``[]`` the sweep skips its loop, writes no manifest, and exits 0.
    """
    args = collect.build_parser().parse_args(
        ["--backend", "cityflow", "--env-config", "x.json",
         "--policy", "random", "--out-dir", "out",
         "--flow-draws-range", start, end]
    )
    with pytest.raises(SystemExit, match="START must be < END"):
        collect._resolve_draw_ids(args)


def test_flow_draws_range_is_half_open_in_help() -> None:
    """The [START, END) convention must be visible at the CLI, not just in a test."""
    help_text = collect.build_parser().format_help()
    assert "--flow-draws-range" in help_text
    assert "half-open" in help_text.lower() or "[start, end)" in help_text.lower()


def test_collect_refuses_flow_draw_on_non_cityflow() -> None:
    """The scope fence is visible in the suite rather than silent.

    SUMO collection additionally needs a generated .sumocfg and a --flow-source-json
    flag; that is P7.3, not this task.
    """
    with pytest.raises(SystemExit, match="cityflow"):
        collect._require_cityflow_for_draws("sumo", [0, 1])
    collect._require_cityflow_for_draws("cityflow", [0, 1])
    collect._require_cityflow_for_draws("sumo", [None])


def _patch_make_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``collect.main`` with no simulator; ``main`` imports make_env at call time."""
    import experiments.envs

    monkeypatch.setattr(experiments.envs, "make_env", lambda spec: FakeTrafficEnv())


def _collect_argv(out: Path, *extra: str) -> list[str]:
    return [
        "--backend", "cityflow",
        "--env-config", str(REPO / "configs/sim/cityflow1x1.json"),
        "--policy", "random",
        "--out-dir", str(out),
        "--episodes", "1",
        *extra,
    ]


def test_accepted_run_materialises_draw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control for the test below, which would otherwise pass vacuously."""
    _patch_make_env(monkeypatch)
    out = tmp_path / "corpus"

    assert collect.main(_collect_argv(out, "--flow-draws-range", "0", "2")) == 0

    flows = out / "flows"
    assert (flows / "flow_draw0.json").exists()
    assert (flows / "flow_draw1.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert [e["flow_draw"] for e in manifest["episodes"]] == [0, 1]

    meta = manifest["run_metadata"]
    assert meta["flow_dir"] == "flows"
    assert meta["numpy_version"] == np.__version__

    # The recorded digest must match the bytes actually on disk, recomputed here by an
    # independent route. Without it, "P2.4 can re-derive any episode's demand" holds only
    # while flows/ is intact -- and NEP 19 gives Generator streams no cross-version
    # guarantee, so re-running the RNG is not a substitute.
    for draw_id in ("0", "1"):
        on_disk = (flows / f"flow_draw{draw_id}.json").read_bytes()
        assert meta["flow_draw_sha256"][draw_id] == hashlib.sha256(on_disk).hexdigest()


def test_refused_run_materialises_no_draw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1's NB2 property: a refused run must create nothing.

    The nominal env is built from the source flow and the logger constructed with it
    *before* any draw is materialised, so the populated-out_dir check is what rejects the
    run -- and it rejects it before a single demand file has been written.  The check
    stays the logger's, rather than being duplicated here where it could drift.
    """
    _patch_make_env(monkeypatch)
    out = tmp_path / "corpus"
    out.mkdir()
    # Look exactly like a previous run, so the logger refuses without --overwrite.
    (out / "ep000000_seed1000.npz").write_bytes(b"")
    (out / "manifest.json").write_text("{}")

    with pytest.raises(FileExistsError, match="already contains a collection run"):
        collect.main(_collect_argv(out, "--flow-draws-range", "0", "2"))

    flows = out / "flows"
    assert not flows.exists() or not list(flows.iterdir()), (
        "a refused run materialised demand files; the populated-out_dir check must run "
        "before the first draw is written"
    )


def test_stale_drawn_flow_cleanup_is_narrow(tmp_path: Path) -> None:
    """P1's NB3 lesson, verbatim: never a directory wipe.

    ``ep*.npz`` once matched an unrelated ``epoch_stats.npz``.  The drawn-flow cleanup
    must match only the exact names the collector writes, so a user file that happens to
    live in flows/ survives.
    """
    flows = tmp_path / "flows"
    flows.mkdir()
    (flows / "flow_draw0.json").write_text("[]")
    (flows / "flow_draw12.json").write_text("[]")
    (flows / "cityflow_draw0.json").write_text("{}")
    keep = [
        flows / "flow_draw_notes.json",
        flows / "my_flow_draw1.json",
        flows / "README.md",
    ]
    for path in keep:
        path.write_text("keep me")

    collect._clear_stale_draw_files(flows)

    assert not (flows / "flow_draw0.json").exists()
    assert not (flows / "flow_draw12.json").exists()
    assert not (flows / "cityflow_draw0.json").exists()
    for path in keep:
        assert path.exists(), f"{path.name} must survive the cleanup"
