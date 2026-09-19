"""P7.3d C3a (second half): the return-prompt calibration PER INTERSECTION (``BRIEF_39`` §3 C3a,
Amendments A4 and B.1; ``PREREGISTRATION`` A17(e), A20(b), A21).

A17(e): *on a pair with more than one intersection the rule applies per intersection i -- S over
that intersection's probe returns, R_best_source,i from the checkpoint, no pooling.*  These tests
are written BEFORE the code, and the quantity they guard is the registered prompt: a target that is
silently pooled, inverted or read from the wrong field still produces a plausible number.

This file grows with the task.  Its first part is the CityFlow half of Rule B's ratio (Amendment
A4: the source-domain denominator does not exist for grid4x4 and must be measured).

GATES, each naming the artifact it consumes
-------------------------------------------
* CityFlow importable, and ``RLTRAFFIC_DRAWS`` (default: this tree's ``scenarios/draws``) holding
  the grid4x4 CityFlow parent of **draw 201** -- the first probe draw, rendered by P7.3d C1.
* ``RLTRAFFIC_CORPUS_V11`` (default: this tree's ``datasets_v11``) holding the five
  ``cf_grid4x4__mappo1000__seed*`` manifests, for the settings-equality test only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

import offline.rtg_calibration as rtg

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID4X4_KEY = "cityflow_grid4x4"
FIRST_PROBE_DRAW = 201
P4_3_PROBE = REPO_ROOT / "docs/data/p4_3_probe.json"
N_INTERSECTIONS = 16
N_DECISIONS = 360


def _draws_root() -> Path:
    value = os.environ.get("RLTRAFFIC_DRAWS")
    return Path(value) if value else REPO_ROOT / "scenarios/draws"


def _corpus_root() -> Path:
    value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    return Path(value) if value else REPO_ROOT / "datasets_v11"


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except Exception:
        return False
    return True


def _p4_3_settings() -> dict[str, Any]:
    return dict(json.loads(P4_3_PROBE.read_text(encoding="utf-8"))["env_settings"])


# ----------------------------------------------------------------------------------
# A synthetic two-intersection episode: integers, so every sum is exact
# ----------------------------------------------------------------------------------
IDS_AND_LANES = [("A0", ["a_0", "a_1"]), ("B0", ["b_0"])]


def _infos(n_steps: int = 4, *, break_lane_route_of: str | None = None) -> list[dict[str, Any]]:
    """Post-step infos whose per-intersection reward IS minus its incoming lanes' waiting count.

    A0 waits ``t + 1`` on each of two lanes (reward ``-2(t+1)``); B0 waits ``3`` on one lane.
    ``break_lane_route_of`` makes that one intersection's reward stream disagree with its lanes.
    """
    infos: list[dict[str, Any]] = []
    for t in range(n_steps):
        waiting = {"a_0": t + 1, "a_1": t + 1, "b_0": 3, "outgoing_9": 50}
        rewards = {"A0": -2.0 * (t + 1), "B0": -3.0}
        if break_lane_route_of is not None and t == 2:
            rewards[break_lane_route_of] -= 1.0
        infos.append(
            {
                "lane_waiting_vehicle_count": waiting,
                "average_travel_time": 10.0 * (t + 1),
                "vehicle_count": 7.0,
                "intersections": {ix: {"reward": rewards[ix]} for ix in ("A0", "B0")},
            }
        )
    return infos


def _fake_roll(infos: list[dict[str, Any]]) -> Any:
    def roll(config_path: Path, **_: Any) -> Any:
        samples = [float(info["average_travel_time"]) for info in infos]
        return IDS_AND_LANES, infos, samples, float(infos[-1]["vehicle_count"])

    return roll


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, infos: list[dict[str, Any]]) -> Any:
    config = tmp_path / "cityflow.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rtg, "_roll_cityflow_probe_episode", _fake_roll(infos), raising=True)
    return rtg.run_probe_per_intersection(
        draw_ids=[201],
        config_for_draw=lambda _draw: config,
        env_settings={"max_steps": len(infos)},
        scenario_id=GRID4X4_KEY,
        engine_seed=1000,
    )


def test_each_intersection_gets_its_own_return_by_two_routes_and_nothing_is_pooled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The returns are recomputed HERE by raw arithmetic over the synthetic infos, per id.

    *Mutation this is built against:* the per-intersection sums pooled (every id given the
    scenario total) -> ``A0`` would read ``-32`` instead of ``-20`` -> this dies.
    """
    infos = _infos()
    (episode,) = _run(monkeypatch, tmp_path, infos)

    expected = {
        "A0": math.fsum(-2.0 * (t + 1) for t in range(4)),  # -20.0
        "B0": math.fsum(-3.0 for _ in range(4)),  # -12.0
    }
    assert expected == {"A0": -20.0, "B0": -12.0}
    assert episode.local_return == expected
    assert episode.local_return_from_lanes == expected
    assert list(episode.local_return) == ["A0", "B0"], "the env's order (C1), never sorted"
    assert episode.draw_id == 201 and episode.decisions == 4
    assert episode.att_horizon == 40.0 and episode.horizon_vehicle_count == 7.0
    assert all(type(v) is float for v in episode.local_return.values())


def test_a_disagreement_between_the_two_routes_is_refused_naming_the_draw_and_the_intersection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A4: *per-intersection returns by two routes under ==*.  One intersection disagreeing refuses
    the episode; the other fifteen agreeing must not be able to hide it."""
    with pytest.raises(ValueError, match=r"draw 201.*'B0'.*-13\.0.*-12\.0"):
        _run(monkeypatch, tmp_path, _infos(break_lane_route_of="B0"))


def test_a_single_intersection_scenario_is_refused_and_sent_to_run_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """hangzhou's probe is :func:`run_probe`, whose scalar records P4.3's artifact is made of; a
    per-intersection record for it would be a second shape of one registered quantity."""
    infos = _infos()
    config = tmp_path / "cityflow.json"
    config.write_text("{}", encoding="utf-8")

    def roll(config_path: Path, **_: Any) -> Any:
        return IDS_AND_LANES[:1], infos, [1.0] * len(infos), 7.0

    monkeypatch.setattr(rtg, "_roll_cityflow_probe_episode", roll, raising=True)
    with pytest.raises(ValueError, match="run_probe"):
        rtg.run_probe_per_intersection(
            draw_ids=[201], config_for_draw=lambda _d: config, env_settings={"max_steps": 4},
            scenario_id="cityflow1x1", engine_seed=1000,
        )


def test_a_short_episode_and_a_missing_config_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rtg, "_roll_cityflow_probe_episode", _fake_roll(_infos(3)), raising=True)
    config = tmp_path / "cityflow.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"draw 201: 3 decisions, not 4"):
        rtg.run_probe_per_intersection(
            draw_ids=[201], config_for_draw=lambda _d: config, env_settings={"max_steps": 4},
            scenario_id=GRID4X4_KEY, engine_seed=1000,
        )
    with pytest.raises(FileNotFoundError, match="probe draw 202"):
        rtg.run_probe_per_intersection(
            draw_ids=[202], config_for_draw=lambda _d: tmp_path / "absent.json",
            env_settings={"max_steps": 4}, scenario_id=GRID4X4_KEY, engine_seed=1000,
        )


def test_run_probe_still_refuses_a_multi_intersection_scenario_with_its_own_message() -> None:
    """hangzhou's entry point is untouched: its refusal is part of P4.3's contract (T-regress)."""
    import inspect

    source = inspect.getsource(rtg.run_probe)
    assert "the probe records one intersection's return and this scenario has" in source
    assert "_roll_maxpressure_episode" in source, "both probes share ONE rollout loop"


# ----------------------------------------------------------------------------------
# The settings: P4.3's, which ARE the grid4x4 corpus's (plan assumption 6, measured)
# ----------------------------------------------------------------------------------
def test_p4_3s_probe_settings_are_the_grid4x4_corpus_settings_field_by_field() -> None:
    """The probe's reward stream is the RTG stream only if the env is the corpus's env.

    ``compare_with`` is a harness key (which baselines a training run prints beside itself), not an
    env setting; P4.3's artifact strips it before recording and it is the ONLY key that may differ.
    """
    from offline.dt_gate import env_settings_from_manifest

    manifests = sorted(_corpus_root().glob("cf_grid4x4__mappo1000__seed*/manifest.json"))
    if len(manifests) != 5:
        pytest.skip(
            f"{_corpus_root()} does not hold the five cf_grid4x4__mappo1000__seed* manifests: set "
            "RLTRAFFIC_CORPUS_V11 to the datasets_v11 tree (it is gitignored)"
        )
    committed = _p4_3_settings()
    for manifest in manifests:
        settings = dict(env_settings_from_manifest(manifest))
        assert settings.pop("compare_with", None) is not None
        assert settings == committed, manifest
    assert committed["local_reward_fn"] == "queue_length" and committed["max_steps"] == N_DECISIONS
    assert committed["control_mode"] == "acyclic" and committed["delta_time"] == 10


# ----------------------------------------------------------------------------------
# One real grid4x4 CityFlow episode
# ----------------------------------------------------------------------------------
def test_one_real_cityflow_episode_on_draw_201_yields_sixteen_agreeing_integral_returns() -> None:
    from offline.materialise_draws import draw_config_path

    config = draw_config_path(GRID4X4_KEY, FIRST_PROBE_DRAW, out_root=_draws_root())
    if not _cityflow_available():
        pytest.skip("the CityFlow engine is not installed, so no probe episode can be rolled")
    if not config.is_file():
        pytest.skip(
            f"{config} is absent: set RLTRAFFIC_DRAWS to the scenarios/draws tree holding P7.3d "
            f"C1's grid4x4 parent of draw {FIRST_PROBE_DRAW} (it is gitignored)"
        )
    from utils.cityflow_utils import parse_roadnet

    (episode,) = rtg.run_probe_per_intersection(
        draw_ids=[FIRST_PROBE_DRAW],
        config_for_draw=lambda d: draw_config_path(GRID4X4_KEY, d, out_root=_draws_root()),
        env_settings=_p4_3_settings(),
        scenario_id=GRID4X4_KEY,
        engine_seed=1000,
    )

    corpus_order = [
        ix.id for ix in parse_roadnet(REPO_ROOT / "scenarios/grid4x4/grid4x4_roadnet_red.json").intersections
    ]
    assert list(episode.local_return) == corpus_order and len(corpus_order) == N_INTERSECTIONS
    assert episode.local_return == episode.local_return_from_lanes, "two routes, per id, under =="
    assert episode.decisions == N_DECISIONS
    for ix_id, value in episode.local_return.items():
        assert value <= 0.0 and float(value).is_integer(), (ix_id, value)
    assert len(set(episode.local_return.values())) > 1, "sixteen intersections, not one value"
