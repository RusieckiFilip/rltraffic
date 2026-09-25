"""P7.3c C1 (``BRIEF_41``): the grid4x4 SUMO corpus door and A17(f)'s gate, per intersection, keyed by id.

What is pinned, and by what route (``BRIEF_41`` §4: critical quantities are re-derived by a route that does
not call the code under test -- the raw ``.npz`` arrays, JSON read here, ``np.cumsum`` on stored rewards):

* **T-regress (a)** -- the door's alignment is chosen by scenario and hz1x1's is today's
  ``declared_alignment()`` (dataclass equality); one hz1x1 draw (201) collected through the CHANGED door has
  every array ``==`` the committed corpus episode, key by key, dtype and shape included.
* **T-17f** -- one grid4x4 draw (201) through the door: its sixteen returns equal the calibration artifact's,
  re-derived here from the raw rewards and found BY ID through the episode's own ``ix_ids``; zero teleports and
  zero collisions counted every simulated second.  A corpus logged in REVERSED id order passes the gate, and
  one return off by one is refused naming its draw and its intersection.  A collision on an odd second of a
  decision is counted and refuses the episode BEFORE the write.
* The gate's other refusals, the pre-token record, the argv, and the gate's subcommand.

GATES -- each ``skipif`` names what it consumes (B.7.3-5's standard line, plus Amendment A's Q10 variable):
the ``sumo`` binary and ``traci``; ``RLTRAFFIC_DRAWS`` (the parity draw tree); ``RLTRAFFIC_GRID4X4_RESCO``
(grid4x4's SUMO network); ``RLTRAFFIC_OUTPUT_ROOT`` (A20(a)'s five checkpoints); ``RLTRAFFIC_SUMO_CORPORA``
(the committed hz1x1 SUMO corpus).  Two SUMO episodes in this file: one grid4x4 (``BRIEF_41`` §4 caps the suite
at four) and one hz1x1.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import offline.sumo_att_reference as sar
import offline.transfer_calibration as tc
from offline import collect
from offline.trajectory_logger import TrajectoryLogger
from tests.p7_3c_fixtures import (
    DATA,
    DECISIONS,
    GRID,
    ScriptedCorpusEnv,
    calibration,
    calibration_ids,
    grid_run_metadata,
    log_episode,
    probe_return,
    probe_rewards,
    registered_engine_seed_drawn,
    write_scripted_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HZ1X1 = "cityflow1x1"
DRAW = 201
IDS = calibration_ids()


# ----------------------------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------------------------


def _needs_sumo() -> None:
    if shutil.which("sumo") is None:
        pytest.skip("the sumo binary is not on PATH")
    if importlib.util.find_spec("traci") is None:
        pytest.skip("traci is not importable")


def _draws_root(scenario: str, draw: int) -> Path:
    """``RLTRAFFIC_DRAWS``, else this tree's ``scenarios/draws``; skip unless *draw*'s parity config is there."""
    root = Path(os.environ.get("RLTRAFFIC_DRAWS") or REPO_ROOT / "scenarios" / "draws")
    config = root / scenario / f"draw_{int(draw):04d}" / "parity" / "noteleport.sumocfg"
    if not config.is_file():
        pytest.skip(
            f"{config} is absent: set RLTRAFFIC_DRAWS to the materialised parity draw tree (it is gitignored)"
        )
    return root


def _needs_resco() -> None:
    value = os.environ.get("RLTRAFFIC_GRID4X4_RESCO")
    if not value:
        pytest.skip("RLTRAFFIC_GRID4X4_RESCO is unset: grid4x4's alignment reads RESCO's SUMO network")
    net = Path(value) / "resco" / "resco_benchmark" / "environments" / "grid4x4" / "grid4x4.net.xml"
    if not net.is_file():
        pytest.skip(f"RLTRAFFIC_GRID4X4_RESCO is set but RESCO's grid4x4 network is not at {net}")


def _sumo_corpora() -> Path:
    value = os.environ.get("RLTRAFFIC_SUMO_CORPORA")
    if not value:
        pytest.skip(
            "RLTRAFFIC_SUMO_CORPORA is unset: T-regress (a) compares with the committed hz1x1 SUMO corpus "
            "(datasets_sumo_v11/, gitignored)"
        )
    return Path(value)


def _output_root_with_subject() -> Path:
    """``RLTRAFFIC_OUTPUT_ROOT``, else this tree's ``output/``; skip unless A20(a)'s five checkpoints are there."""
    root = Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT") or REPO_ROOT / "output")
    for seed in tc.TRAINING_SEEDS:
        path = root / tc.GRID4X4_CHECKPOINT_SUBDIR / f"{tc.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        if not path.is_file():
            pytest.skip(f"{path} is absent: set RLTRAFFIC_OUTPUT_ROOT to the output tree holding A20(a)'s subject")
    return root


# ----------------------------------------------------------------------------------------------
# T-regress (a): the alignment chosen by scenario, and hz1x1 byte for byte through the changed door
# ----------------------------------------------------------------------------------------------


def test_the_door_resolves_hz1x1_to_todays_declared_alignment() -> None:
    """``BRIEF_41`` C1(i): hz1x1 resolves to exactly the alignment ``declared_alignment()`` builds."""
    from offline.aligned_env import DECLARED_SCENARIO, declared_alignment

    got = collect._alignment_for_door(HZ1X1)
    assert got == declared_alignment()
    assert got.scenario == DECLARED_SCENARIO


def test_the_door_resolves_grid4x4_to_its_own_identity_alignment() -> None:
    """The lookup is BY SCENARIO: grid4x4 gets its own 16-id identity alignment, never hangzhou's.

    *Mutation:* the lookup returning hz1x1's alignment for every key -> this and T-17f die.
    """
    _needs_resco()
    from offline.aligned_env import alignment_for_scenario_key, declared_alignment

    got = collect._alignment_for_door(GRID)
    assert got == alignment_for_scenario_key(GRID)
    assert got != declared_alignment()
    assert got.phase_map_is_hangzhou_shaped is False
    assert sorted(got.intersections) == sorted(IDS)


def test_one_hz1x1_draw_through_the_changed_door_reproduces_the_committed_corpus_episode(tmp_path: Path) -> None:
    """T-regress (a): draw 201 through the door as C1 leaves it, every array ``==`` the committed episode.

    The committed episode was logged by P7.3a's door (the PLAIN env, ``declared_alignment()``); this one by the
    OBSERVED env and the scenario lookup.  Equal arrays are the measurement that neither changed the episode.
    """
    _needs_sumo()
    draws = _draws_root(HZ1X1, DRAW)
    reference = _sumo_corpora() / "hz1x1_sumo_maxpressure" / f"ep000000_seed1000_draw{DRAW}.npz"
    if not reference.is_file():
        pytest.skip(f"{reference} is absent: RLTRAFFIC_SUMO_CORPORA does not hold the hz1x1 corpus")

    out = tmp_path / "hz1x1"
    tc.collect_logged_probe_corpus(HZ1X1, [DRAW], out_dir=out, draws_root=draws)
    produced = out / reference.name
    with np.load(reference) as committed, np.load(produced) as logged:
        assert sorted(committed.files) == sorted(logged.files)
        for key in committed.files:
            want, got = committed[key], logged[key]
            assert want.dtype == got.dtype, key
            assert want.shape == got.shape, key
            assert np.array_equal(want, got), key
    record = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_metadata"]["sumo_draws"][0]
    assert record["n_teleports"] == 0 and record["n_collisions"] == 0


# ----------------------------------------------------------------------------------------------
# T-17f: one grid4x4 draw through the door, the gate by id, the dense counter
# ----------------------------------------------------------------------------------------------


def test_one_grid4x4_draw_through_the_door_reproduces_the_probe_on_all_sixteen_by_id(tmp_path: Path) -> None:
    """T-17f: draw 201 collected through the door; the gate and THIS test's own route both find 16 of 16.

    The test's route: the raw ``.npz``, each intersection located BY ID in the episode's own ``ix_ids``, its
    stored rewards summed by ``np.cumsum`` in float64, compared with the JSON value read here.
    """
    _needs_sumo()
    _needs_resco()
    draws = _draws_root(GRID, DRAW)

    out = tmp_path / "grid"
    tc.collect_logged_probe_corpus(GRID, [DRAW], out_dir=out, draws_root=draws)
    record = tc.assert_logged_corpus_matches_probe_per_intersection(out, draw_ids=[DRAW])
    assert record["n_checked"] == 16 and record["n_matching"] == 16 and record["all_match"] is True

    with np.load(out / f"ep000000_seed1000_draw{DRAW}.npz") as episode:
        ids = [str(value) for value in episode["ix_ids"].tolist()]
        assert sorted(ids) == sorted(IDS)
        assert int(episode["episode_length"]) == DECISIONS
        for ix in IDS:
            rewards = np.asarray(episode[f"ix{ids.index(ix)}_local_reward"], dtype=np.float64)
            assert float(np.cumsum(rewards)[-1]) == probe_return(DRAW, ix), ix

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_metadata"]["scenario_id"] == GRID
    door = manifest["run_metadata"]["sumo_draws"][0]
    assert door["n_teleports"] == 0 and door["n_collisions"] == 0
    assert door["engine_events_counted"] == collect.ENGINE_EVENT_GRAIN
    assert door["engine_seed_drawn"] == registered_engine_seed_drawn()
    assert door["vehicle_types_seen"] == ["cf_parity"] and door["time_to_teleport_option"] == "-1"


def test_the_gate_keys_every_intersection_by_id_on_a_corpus_logged_in_reversed_order(tmp_path: Path) -> None:
    """A corpus whose ``ix_ids`` run D3 .. A0 carries each id's probe return under ITS id -- the gate passes.

    *Mutation:* the gate keyed by position -> it compares A0's rewards with D3's return -> this dies.
    """
    corpus = write_scripted_corpus(
        tmp_path / "reversed", ids=list(reversed(IDS)), draws=[DRAW], rewards_for=probe_rewards
    )
    with np.load(corpus / f"ep000000_seed1000_draw{DRAW}.npz") as episode:
        assert [str(value) for value in episode["ix_ids"].tolist()] == list(reversed(IDS))

    record = tc.assert_logged_corpus_matches_probe_per_intersection(corpus, draw_ids=[DRAW])
    assert record["n_checked"] == 16 and record["n_matching"] == 16 and record["all_match"] is True


def test_the_gate_refuses_one_return_off_by_one_and_names_the_draw_and_the_intersection(tmp_path: Path) -> None:
    """One intersection's first reward moved by 1.0 -- the gate refuses, naming draw 201 and that id."""
    victim = IDS[5]

    def rewards_for(draw: int, ix: str, decisions: int) -> list[float]:
        values = probe_rewards(draw, ix, decisions)
        if ix == victim:
            values[0] -= 1.0
        return values

    corpus = write_scripted_corpus(tmp_path / "off", ids=list(reversed(IDS)), draws=[DRAW], rewards_for=rewards_for)
    with pytest.raises(ValueError, match=re.escape(f"the first is draw {DRAW}, intersection {victim!r}")):
        tc.assert_logged_corpus_matches_probe_per_intersection(corpus, draw_ids=[DRAW])


# ---- the dense counter, on the REAL per-second loop driven by a scripted connection ------------------

_Collision = collections.namedtuple("_Collision", sar.COLLISION_FACT_KEYS)


def _collision(collider: str, victim: str) -> Any:
    return _Collision(collider, victim, "cf_parity", "cf_parity", 6.25, 0.0, "collision", "L_0", 12.5)


class _StubSimulation:
    """TraCI's ``simulation`` domain, scripted one entry per simulated second."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.index = -1

    def _now(self) -> dict[str, Any]:
        return self.script[self.index]

    def getTime(self) -> float:
        return float(self._now()["t"])

    def getDepartedIDList(self) -> tuple[str, ...]:
        return tuple(self._now().get("departed", ()))

    def getArrivedIDList(self) -> tuple[str, ...]:
        return tuple(self._now().get("arrived", ()))

    def getPendingVehicles(self) -> tuple[str, ...]:
        return ()

    def getStartingTeleportIDList(self) -> tuple[str, ...]:
        return tuple(self._now().get("starts", ()))

    def getEndingTeleportIDList(self) -> tuple[str, ...]:
        return tuple(self._now().get("ends", ()))

    def getCollisions(self) -> tuple[Any, ...]:
        return tuple(self._now().get("collisions", ()))

    def getOption(self, name: str) -> str:
        assert name == "time-to-teleport"
        return "-1"


class _StubVehicle:
    def __init__(self, simulation: _StubSimulation) -> None:
        self.simulation = simulation

    def getIDList(self) -> tuple[str, ...]:
        return tuple(self.simulation._now().get("present", ()))

    def getDeparture(self, vid: str) -> float:
        return float(self.simulation._now()["t"]) - 1.0

    def getDepartDelay(self, vid: str) -> float:
        return 0.0

    def getTypeID(self, vid: str) -> str:
        return "cf_parity"


class _StubConnection:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.simulation = _StubSimulation(script)
        self.vehicle = _StubVehicle(self.simulation)

    def simulationStep(self) -> None:
        self.simulation.index += 1


def _one_decision(*, collide: bool) -> list[dict[str, Any]]:
    """Ten simulated seconds -- ONE decision -- with vehicle ``a`` arriving on second 7, not a boundary.

    With ``collide`` it arrives by colliding into ``b``: SUMO teleports the collider and, on its last edge,
    removes it as arrived in the same step (A23(a)'s two events had exactly this shape).
    """
    script: list[dict[str, Any]] = []
    for second in range(1, 11):
        entry: dict[str, Any] = {"t": float(second), "present": ["a", "b"] if second < 7 else ["b"]}
        if second == 1:
            entry["departed"] = ["a", "b"]
        if second == 7:
            entry["arrived"] = ["a"]
            if collide:
                entry["collisions"] = [_collision("a", "b")]
                entry["starts"] = ["a"]
        script.append(entry)
    return script


def _observed_env(script: list[dict[str, Any]]) -> Any:
    """``PerSecondSumoObserver`` without a simulator: the REAL class and its REAL ``_simulate``."""
    env = object.__new__(sar.sumo_observer_env_class())
    env._sumo = _StubConnection(script)
    env._metrics = None
    env.halting_check = False
    env._engine_seed = registered_engine_seed_drawn()
    env.recorder = sar.SumoObservationRecorder(
        delta_time=10.0,
        intended_departures=sar.IntendedDepartures(by_id={"a": 0.0, "b": 0.0}, source="stub"),
        step_length=1.0,
    )
    return env


def _open_logger(out: Path, record: dict[str, Any]) -> tuple[TrajectoryLogger, ScriptedCorpusEnv]:
    """A logger with ONE buffered, unfinalized episode; *record* is shared with its run metadata by reference,
    as ``offline.collect`` shares ``sumo_draws``."""
    env = ScriptedCorpusEnv(["A0", "A1"])
    logger = TrajectoryLogger(env, out, run_metadata={"scenario_id": GRID, "sumo_draws": [record]})
    log_episode(logger, env, draw=DRAW, decisions=3, rewards_for=lambda draw, ix, n: [0.0] * n)
    return logger, env


def test_a_collision_on_an_odd_second_is_counted_and_refuses_the_episode_before_any_write(tmp_path: Path) -> None:
    """``DEFERRED`` 93's fix: the counts come from EVERY simulated second, and an event refuses BEFORE the write.

    *Mutation:* the counter read once per decision (one ``getCollisions()`` at the boundary, second 10) ->
    second 7's collision is missed -> the episode is written -> this dies.
    """
    env = _observed_env(_one_decision(collide=True))
    env._simulate(10)
    events = collect.dense_engine_events(env)
    assert events["n_collisions"] == 1 and events["n_teleports"] == 1

    out = tmp_path / "corpus"
    record: dict[str, Any] = {"draw_id": DRAW}
    logger, _env = _open_logger(out, record)
    with pytest.raises(ValueError, match=re.escape(f"draw {DRAW}: 1 teleport(s) and 1 collision(s)")):
        collect.finish_sumo_draw_episode(env, logger, record=record, draw_id=DRAW, engine_seed=1000)
    assert sorted(path.name for path in out.iterdir()) == [], "a refused episode wrote something"


def test_a_clean_episode_is_written_with_its_dense_counts_in_the_manifest(tmp_path: Path) -> None:
    """The control for the refusal above: the same decision without the collision is written, and its draw's
    record carries the counts, their grain and the engine reads."""
    env = _observed_env(_one_decision(collide=False))
    env._simulate(10)

    out = tmp_path / "corpus"
    record: dict[str, Any] = {"draw_id": DRAW}
    logger, _env = _open_logger(out, record)
    path = collect.finish_sumo_draw_episode(env, logger, record=record, draw_id=DRAW, engine_seed=1000)
    assert path == out / f"ep000000_seed1000_draw{DRAW}.npz" and path.is_file()

    written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_metadata"]["sumo_draws"][0]
    assert written == {
        "draw_id": DRAW,
        "engine_seed_requested": 1000,
        "engine_seed_drawn": registered_engine_seed_drawn(),
        "time_to_teleport_option": "-1",
        "vehicle_types_seen": ["cf_parity"],
        "n_teleports": 0,
        "n_collisions": 0,
        "engine_events_counted": collect.ENGINE_EVENT_GRAIN,
    }


# ----------------------------------------------------------------------------------------------
# The gate's refusals -- each BEFORE anything is compared or returned
# ----------------------------------------------------------------------------------------------


def _corpus(tmp_path: Path, **kwargs: Any) -> Path:
    """A good one-draw corpus unless *kwargs* say otherwise."""
    options: dict[str, Any] = {"ids": list(IDS), "draws": [DRAW], "rewards_for": probe_rewards}
    options.update(kwargs)
    return write_scripted_corpus(tmp_path / "corpus", **options)


def _data_copy(tmp_path: Path, *, tamper: str | None = None) -> Path:
    """The two committed artifacts copied; *tamper* names one whose bytes gain a trailing newline."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("p7_3d_calibration.json", "p7_3d_grid4x4.json"):
        body = (DATA / name).read_bytes()
        (data / name).write_bytes(body + b"\n" if name == tamper else body)
    return data


def _zeros(draw: int, ix: str, decisions: int) -> list[float]:
    return [0.0] * decisions


def _with_record(draw: int, **overrides: Any) -> dict[str, Any]:
    metadata = grid_run_metadata([draw])
    metadata["sumo_draws"][0].update(overrides)
    return metadata


def _without_counts(draw: int) -> dict[str, Any]:
    metadata = grid_run_metadata([draw])
    for key in ("n_teleports", "n_collisions", "engine_events_counted"):
        del metadata["sumo_draws"][0][key]
    return metadata


def _known_or_zero(draw: int, ix: str, decisions: int) -> list[float]:
    """The probe's rewards for an id the artifact knows; zeros for one it does not (the gate must refuse first)."""
    return probe_rewards(draw, ix, decisions) if ix in IDS else _zeros(draw, ix, decisions)


def _half_reward(draw: int, ix: str, decisions: int) -> list[float]:
    values = probe_rewards(draw, ix, decisions)
    if ix == IDS[0]:
        values[0] -= 0.5
        values[1] += 0.5
    return values


Case = Callable[[Path], tuple[Path, Path, list[int]]]

#: case -> (the corpus, the data dir and the requested draws; the refusal it must raise, as a regex)
REFUSALS: dict[str, tuple[Case, str]] = {
    "the calibration artifact at another digest": (
        lambda t: (_corpus(t), _data_copy(t, tamper="p7_3d_calibration.json"), [DRAW]),
        r"p7_3d_calibration\.json: sha256 [0-9a-f]{64} is not the pinned",
    ),
    "the zero-shot artifact at another digest": (
        lambda t: (_corpus(t), _data_copy(t, tamper="p7_3d_grid4x4.json"), [DRAW]),
        r"p7_3d_grid4x4\.json: sha256 [0-9a-f]{64} is not the pinned",
    ),
    "a requested draw absent": (
        lambda t: (_corpus(t), DATA, [DRAW, DRAW + 1]),
        re.escape(f"no logged episode for draw(s) [{DRAW + 1}]"),
    ),
    "a draw not requested": (
        lambda t: (_corpus(t, draws=[DRAW, DRAW + 1], run_metadata=grid_run_metadata([DRAW, DRAW + 1])), DATA, [DRAW]),
        re.escape(f"holds draw(s) [{DRAW + 1}] that were not requested"),
    ),
    "no per-second count": (
        lambda t: (_corpus(t, run_metadata=_without_counts(DRAW)), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: the corpus records no engine-event count"),
    ),
    "a collision": (
        lambda t: (_corpus(t, run_metadata=_with_record(DRAW, n_collisions=1)), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: 0 teleport(s) and 1 collision(s)"),
    ),
    "a teleport": (
        lambda t: (_corpus(t, run_metadata=_with_record(DRAW, n_teleports=1)), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: 1 teleport(s) and 0 collision(s)"),
    ),
    "another engine seed": (
        lambda t: (_corpus(t, run_metadata=_with_record(DRAW, engine_seed_drawn=12345)), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: engine_seed_drawn 12345, not 437485271"),
    ),
    "another vehicle type": (
        lambda t: (_corpus(t, run_metadata=_with_record(DRAW, vehicle_types_seen=["DEFAULT_VEHTYPE"])), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: vehicle types ['DEFAULT_VEHTYPE'], not ['cf_parity']"),
    ),
    "another teleport option": (
        lambda t: (_corpus(t, run_metadata=_with_record(DRAW, time_to_teleport_option="300")), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: time-to-teleport '300', not '-1'"),
    ),
    "a non-integral reward": (
        lambda t: (_corpus(t, rewards_for=_half_reward), DATA, [DRAW]),
        re.escape(f"draw {DRAW}, intersection {IDS[0]!r}: 2 of {DECISIONS} stored per-step rewards are NOT integral"),
    ),
    "an id the artifact does not know": (
        lambda t: (_corpus(t, ids=["Z9", *IDS[1:]], rewards_for=_known_or_zero), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: the episode's intersections differ from the artifact's (missing [{IDS[0]!r}], unknown ['Z9'])"),
    ),
    "another scenario key": (
        lambda t: (_corpus(t, run_metadata=grid_run_metadata([DRAW], scenario_id=HZ1X1)), DATA, [DRAW]),
        re.escape(f"scenario_id {HZ1X1!r}, not {GRID!r}"),
    ),
    "a training draw": (
        lambda t: (_corpus(t, draws=[200], rewards_for=_zeros, run_metadata=grid_run_metadata([200])), DATA, [200]),
        r"the probe draw set is not disjoint",
    ),
    "a short episode": (
        lambda t: (_corpus(t, decisions=DECISIONS - 1), DATA, [DRAW]),
        re.escape(f"draw {DRAW}: {DECISIONS - 1} decision(s), not {DECISIONS}"),
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_the_gate_refuses(case: str, tmp_path: Path) -> None:
    """Every refusal names what it found; none of these corpora may pass A17(f)'s gate."""
    build, pattern = REFUSALS[case]
    corpus, data, draw_ids = build(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        tc.assert_logged_corpus_matches_probe_per_intersection(corpus, data_dir=data, draw_ids=draw_ids)


def test_the_gate_reads_the_engine_seed_from_the_committed_zero_shot_artifact(tmp_path: Path) -> None:
    """Amendment A (Q14): 437485271, read from ``p7_3d_grid4x4.json`` at its pinned digest -- never elsewhere."""
    assert tc.registered_engine_seed_drawn() == registered_engine_seed_drawn() == 437485271
    with pytest.raises(ValueError, match="sha256"):
        tc.registered_engine_seed_drawn(_data_copy(tmp_path, tamper="p7_3d_grid4x4.json"))


def test_the_corpus_band_is_the_calibration_artifacts_probe_band() -> None:
    low, high = calibration()["disjointness"]["probe_draw_ids"]
    assert tc.GRID4X4_CORPUS_DRAWS == tuple(range(int(low), int(high) + 1))


# ----------------------------------------------------------------------------------------------
# The pre-token record, the argv and the gate's subcommand
# ----------------------------------------------------------------------------------------------


def test_the_corpus_preflight_checks_the_inputs_and_the_band_against_the_five_checkpoints(tmp_path: Path) -> None:
    """C1(v) and Q21: the band against the draws the CHECKPOINTS trained on and the held-out pool, before any
    episode; the two committed artifacts at their pins; RESCO's network resolved and digest-checked."""
    root = _output_root_with_subject()
    _needs_resco()
    record = tc.corpus_preflight_record(tc.GRID4X4_CORPUS_DRAWS, output_root=root)
    assert record["disjointness"]["disjoint"] is True
    assert record["disjointness"]["n_probe_draws"] == 100
    assert record["disjointness"]["training_draw_ids"] == [1, 200]
    assert record["disjointness"]["n_training_draws"] == 200
    for key, name in (("calibration_sha256", "p7_3d_calibration.json"), ("zero_shot_artifact_sha256", "p7_3d_grid4x4.json")):
        assert record["inputs"][key] == hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    assert record["engine_seed_drawn"] == registered_engine_seed_drawn()

    for band in ([200, 201], [1000]):
        with pytest.raises(ValueError, match="not disjoint"):
            tc.corpus_preflight_record(band, output_root=root)
    with pytest.raises(ValueError, match="sha256"):
        tc.corpus_preflight_record(
            tc.GRID4X4_CORPUS_DRAWS, output_root=root, data_dir=_data_copy(tmp_path, tamper="p7_3d_calibration.json")
        )


def test_the_corpus_argv_is_the_probes_own_settings_and_never_overwrites(tmp_path: Path) -> None:
    """The corpus is the PROBE through the logger (A17(f)): the env settings the argv parses to equal the ones
    the probe parsed (``collect_style_args``), setting by setting; one episode per draw at seed 1000; the
    CityFlow sim config names the scenario; ``--overwrite`` is never passed."""
    from offline.sumo_att_reference import collect_style_args
    from offline.transfer_gate import COLLECT_SETTINGS

    out, draws = tmp_path / "corpus", tmp_path / "draws"
    argv = tc.logged_corpus_argv(GRID, tc.GRID4X4_CORPUS_DRAWS, out_dir=out, draws_root=draws)
    assert argv[-len(COLLECT_SETTINGS):] == list(COLLECT_SETTINGS)
    assert "--overwrite" not in argv
    parsed = collect.build_parser().parse_args(argv)
    assert parsed.backend == "sumo" and parsed.policy == "maxpressure"
    assert parsed.episodes == 1 and parsed.base_seed == 1000
    assert parsed.flow_draws == list(range(201, 301))
    assert Path(parsed.env_config) == REPO_ROOT / "configs" / "sim" / f"{GRID}.json"
    assert Path(parsed.out_dir) == out and Path(parsed.draws_root) == draws

    probe = collect_style_args("sumo", "maxpressure", tmp_path / "x.sumocfg", sentinel_out_dir=tmp_path / "sentinel")
    for key in (
        "max_steps", "delta_time", "control_mode", "global_reward_fn", "local_reward_fn",
        "global_reward_weight", "state_features", "metrics", "thread_num", "libsumo", "fixed_time_k",
    ):
        assert getattr(parsed, key) == getattr(probe, key), key


def test_the_gate_subcommand_writes_its_record_only_after_the_gate_passes(tmp_path: Path) -> None:
    """The gate record is a WRITE, so it follows every check: a refused corpus leaves no record and no directory."""
    good = write_scripted_corpus(tmp_path / "good", ids=list(reversed(IDS)), draws=[DRAW], rewards_for=probe_rewards)
    record = tmp_path / "run" / "a17f_gate.json"
    common = ["--work-dir", str(tmp_path / "run"), "corpus-gate", "--draws-range", str(DRAW), str(DRAW + 1)]
    assert tc.main([*common, "--corpus-dir", str(good), "--record", str(record)]) == 0
    written = json.loads(record.read_text(encoding="utf-8"))
    assert written["format_version"] == tc.CORPUS_GATE_FORMAT_VERSION
    assert written["n_checked"] == 16 and written["n_matching"] == 16 and written["all_match"] is True

    bad = write_scripted_corpus(tmp_path / "bad", ids=list(IDS), draws=[DRAW], rewards_for=_half_reward)
    refused = tmp_path / "refused" / "a17f_gate.json"
    assert tc.main([*common, "--corpus-dir", str(bad), "--record", str(refused)]) == 1
    assert not refused.parent.exists(), "a refused gate created its record's directory"
