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


# ==================================================================================
# The subject's constants PER INTERSECTION (Amendments A1, A4)
# ==================================================================================
import hashlib  # noqa: E402
from fractions import Fraction  # noqa: E402

import offline.transfer_calibration as calibration  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)
FAKE_IDS = ["A0", "B0"]


def _output_root() -> Path:
    return Path(os.environ.get("RLTRAFFIC_OUTPUT_ROOT", str(REPO_ROOT / "output")))


def _fake_payload(seed: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": "spatial-dt-checkpoint/1.0",
        "config": {"state_dim": 40, "n_actions": 8, "n_nodes": 2, "context_length": 20,
                   "n_head": 4, "spatial_mixing": False},
        "target_rtg": {"A0": -132.0, "B0": -114.0},
        "rtg_scale": {"A0": 259.0, "B0": 232.0},
        "scenario_id": "cityflow_grid4x4",
        "intersection_ids": list(FAKE_IDS),
        "spatial_mask": [[True, False], [False, True]],
        "stats": {
            "draw_ids": list(range(1, 201)),
            "rtg": {"cityflow_grid4x4": {
                "A0": {"count": 72000, "min": -259.0, "max": 0.0},
                "B0": {"count": 72000, "min": -232.0, "max": 0.0},
            }},
        },
        "provenance": {"gradient_steps": 40000, "seed": seed},
    }
    payload.update(overrides)
    return payload


def _write_fake_checkpoints(root: Path, payloads: dict[int, dict[str, Any]]) -> dict[int, str]:
    import torch

    directory = root / calibration.GRID4X4_CHECKPOINT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    digests: dict[int, str] = {}
    for seed, payload in payloads.items():
        path = directory / f"{calibration.GRID4X4_CHECKPOINT_STEM}{seed}.pt"
        torch.save(payload, path)
        digests[seed] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_the_registered_pins_are_a20s_five_digests() -> None:
    """Typed here, not imported: A20(a)'s prefixes, so a moved pin fails against the registration."""
    prefixes = {101: "329fb6b8", 202: "f5413585", 303: "48076dab", 404: "4b61bc06", 505: "09bd310d"}
    assert {seed: digest[:8] for seed, digest in calibration.GRID4X4_CHECKPOINT_SHA256.items()} == prefixes
    assert all(len(digest) == 64 for digest in calibration.GRID4X4_CHECKPOINT_SHA256.values())
    assert calibration.REGISTERED_K == 100 and calibration.REGISTERED_STATISTIC == "mean"


def test_the_subjects_constants_are_read_per_intersection_from_the_field_a4_names(
    tmp_path: Path,
) -> None:
    """``R_best_source,i`` is ``payload["target_rtg"][i]``; ``stats.rtg`` bounds the SUPPORT only.

    *Mutation this is built against:* ``R_best`` read from ``stats["rtg"][..]["max"]`` -- 0.0 on
    every intersection, because that block is per-window -- -> this dies, and so does every target.
    """
    digests = _write_fake_checkpoints(tmp_path, {seed: _fake_payload(seed) for seed in SEEDS})
    facts = calibration.subject_facts_per_intersection(output_root=tmp_path, expected_sha256=digests)

    assert facts.subject == "mappo1000_dt_nomix_h4" and facts.scenario_id == "cityflow_grid4x4"
    assert facts.intersection_ids == ("A0", "B0")
    assert facts.best_source_return == {"A0": -132.0, "B0": -114.0}
    assert facts.rtg_scale == {"A0": 259.0, "B0": 232.0}
    assert facts.support_range == {"A0": (-259.0, 0.0), "B0": (-232.0, 0.0)}
    assert facts.n_rows == {"A0": 72000, "B0": 72000}
    assert facts.training_draw_ids == tuple(range(1, 201))
    assert facts.checkpoint_sha256 == digests and len(facts.checkpoints) == 5
    assert (facts.state_dim, facts.context_length, facts.n_head, facts.gradient_steps) == (40, 20, 4, 40000)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_rtg": {"A0": -132.0, "B0": -113.0}}, r"seed 303.*'B0'.*target_rtg"),
        ({"rtg_scale": {"A0": 259.0, "B0": 231.0}}, r"seed 303.*'B0'.*rtg_scale"),
        ({"format_version": "dt-checkpoint/1.0"}, r"dt-checkpoint/1\.0"),
        ({"spatial_mask": [[True, True], [False, True]]}, "identity"),
        ({"provenance": {"gradient_steps": 39999, "seed": 303}}, "39999"),
        ({"target_rtg": {"A0": -132.0}}, r"'B0'"),
    ],
)
def test_a_seed_that_disagrees_or_is_not_the_registered_shape_is_refused_by_name(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    payloads = {seed: _fake_payload(seed) for seed in SEEDS}
    payloads[303] = _fake_payload(303, **overrides)
    digests = _write_fake_checkpoints(tmp_path, payloads)
    with pytest.raises(ValueError, match=message):
        calibration.subject_facts_per_intersection(output_root=tmp_path, expected_sha256=digests)


def test_a_checkpoint_at_another_digest_or_absent_is_refused(tmp_path: Path) -> None:
    digests = _write_fake_checkpoints(tmp_path, {seed: _fake_payload(seed) for seed in SEEDS})
    wrong = dict(digests)
    wrong[404] = "0" * 64
    with pytest.raises(ValueError, match=r"seed 404.*" + digests[404][:12]):
        calibration.subject_facts_per_intersection(output_root=tmp_path, expected_sha256=wrong)
    # The default pins are A20's: five synthetic files cannot satisfy them.
    with pytest.raises(ValueError, match="329fb6b8"):
        calibration.subject_facts_per_intersection(output_root=tmp_path)
    (tmp_path / calibration.GRID4X4_CHECKPOINT_SUBDIR / f"{calibration.GRID4X4_CHECKPOINT_STEM}505.pt").unlink()
    with pytest.raises(FileNotFoundError, match="seed505"):
        calibration.subject_facts_per_intersection(output_root=tmp_path, expected_sha256=digests)


def test_the_real_subject_declares_sixteen_prompts_at_a20s_digests() -> None:
    directory = _output_root() / calibration.GRID4X4_CHECKPOINT_SUBDIR
    if not (directory / f"{calibration.GRID4X4_CHECKPOINT_STEM}101.pt").is_file():
        pytest.skip(
            f"{directory} does not hold grid4x4_mappo1000_dt_nomix_h4_seed101.pt: set "
            "RLTRAFFIC_OUTPUT_ROOT to the output tree carrying p5_2/checkpoints (gitignored)"
        )
    facts = calibration.subject_facts_per_intersection(output_root=_output_root())

    assert len(facts.intersection_ids) == N_INTERSECTIONS
    assert facts.checkpoint_sha256 == calibration.GRID4X4_CHECKPOINT_SHA256
    # Literals from the plan gate's own read of the payload (docs/plans/p7.3d.md F1).
    assert facts.best_source_return["A0"] == -132.0 and facts.best_source_return["C2"] == -64.0
    assert facts.rtg_scale["C1"] == 836.0 and facts.rtg_scale["B3"] == 160.0
    assert facts.support_range["A0"] == (-259.0, 0.0)
    assert facts.training_draw_ids == tuple(range(1, 201))
    assert (facts.state_dim, facts.n_head, facts.gradient_steps) == (40, 4, 40000)
    for ix_id in facts.intersection_ids:
        low, high = facts.support_range[ix_id]
        assert low == -facts.rtg_scale[ix_id] and high == 0.0, ix_id
        assert low <= facts.best_source_return[ix_id] <= high, ix_id


# ==================================================================================
# Rule B PER INTERSECTION -- the registered prompt, recomputed here by exact arithmetic
# ==================================================================================
def _returns(scale_a0: int, scale_b0: int, n: int = 100) -> dict[int, dict[str, float]]:
    """Integral returns for draws 201..200+n, different per intersection and per draw."""
    return {
        201 + j: {"A0": float(-(scale_a0 + 3 * j)), "B0": float(-(scale_b0 + 7 * (j % 11)))}
        for j in range(n)
    }


def _facts(tmp_path: Path) -> Any:
    digests = _write_fake_checkpoints(tmp_path, {seed: _fake_payload(seed) for seed in SEEDS})
    return calibration.subject_facts_per_intersection(output_root=tmp_path, expected_sha256=digests)


def _exact_mean(returns: dict[int, dict[str, float]], ix_id: str, k: int) -> float:
    """The mean of the first k draws by RATIONAL arithmetic -- no float summation at all."""
    draws = sorted(returns)[:k]
    total = sum(Fraction(returns[d][ix_id]) for d in draws)
    return float(total / k)


def test_rule_b_is_applied_per_intersection_and_equals_exact_arithmetic(tmp_path: Path) -> None:
    """``target_i(k) = R_i x (mean_sumo_i(k) / mean_cityflow_i(k))``, the ratio formed FIRST.

    Recomputed here with ``fractions.Fraction`` means -- a route that shares nothing with
    ``probe_statistic``'s ``fsum`` -- and compared under ``==``.  *Mutations this is built against:*
    the statistics pooled across intersections; the ratio inverted; k = 20 given the registered role.
    """
    facts = _facts(tmp_path)
    sumo, cityflow = _returns(900, 400), _returns(300, 200)
    stats = calibration.per_intersection_statistics(sumo, cityflow, intersection_ids=FAKE_IDS)
    targets = calibration.per_intersection_targets(facts, stats)

    assert sorted(stats) == ["k100", "k20", "k5"] and sorted(targets) == ["A0", "B0"]
    for k in (5, 20, 100):
        assert stats[f"k{k}"]["draw_ids"] == [201, 200 + k], "nested prefixes 201 .. 200+k"
        for ix_id in FAKE_IDS:
            mean_sumo, mean_cityflow = _exact_mean(sumo, ix_id, k), _exact_mean(cityflow, ix_id, k)
            cell = stats[f"k{k}"]["per_intersection"][ix_id]
            assert cell == {"sumo": mean_sumo, "cityflow": mean_cityflow}
            expected = facts.best_source_return[ix_id] * (mean_sumo / mean_cityflow)
            entry = targets[ix_id][f"k{k}"]
            assert entry["target"] == expected, (ix_id, k)
            assert entry["role"] == ("registered_prompt" if k == 100 else "recorded_not_evaluated")
            assert entry["inputs"] == {
                "best_source_return": facts.best_source_return[ix_id],
                "probe_target_stat": mean_sumo,
                "probe_source_stat": mean_cityflow,
            }
    # No pooling: the two intersections' targets are different numbers, from different inputs.
    assert targets["A0"]["k100"]["target"] != targets["B0"]["k100"]["target"]
    assert [ix for ix in targets if targets[ix]["k100"]["role"] == "registered_prompt"] == FAKE_IDS
    # The in-support diagnostic, per intersection, against THAT intersection's own range.
    # (`in_support_position`'s own fields, read from its body: position / margin / range.)
    for ix_id in FAKE_IDS:
        position = targets[ix_id]["k100"]["in_support"]
        low, high = facts.support_range[ix_id]
        target = targets[ix_id]["k100"]["target"]
        expected_position = "below" if target < low else ("above" if target > high else "inside")
        assert position["position"] == expected_position, ix_id
        assert position["range"] == [low, high] and position["target_rtg"] == target, ix_id
    # With these inputs A0's target leaves its own range and B0's does not -- so the diagnostic is
    # exercised on both sides, and per intersection.
    assert targets["A0"]["k100"]["in_support"]["position"] == "below"
    assert targets["B0"]["k100"]["in_support"]["position"] == "inside"


def test_equal_probe_statistics_return_each_intersections_own_best_return_bit_for_bit(
    tmp_path: Path,
) -> None:
    """P4.3's in-domain identity, per intersection: ratio exactly 1.0, target exactly ``R_i``."""
    facts = _facts(tmp_path)
    same = _returns(417, 263)
    stats = calibration.per_intersection_statistics(same, same, intersection_ids=FAKE_IDS)
    targets = calibration.per_intersection_targets(facts, stats)
    for ix_id in FAKE_IDS:
        for k in (5, 20, 100):
            assert targets[ix_id][f"k{k}"]["target"] == facts.best_source_return[ix_id]


def test_the_two_probes_must_cover_the_same_draws_and_every_intersection() -> None:
    sumo, cityflow = _returns(900, 400), _returns(300, 200)
    shorter = {d: v for d, v in cityflow.items() if d != 250}
    with pytest.raises(ValueError, match="250"):
        calibration.per_intersection_statistics(sumo, shorter, intersection_ids=FAKE_IDS)
    with pytest.raises(ValueError, match=r"k=100 needs 100"):
        calibration.per_intersection_statistics(
            _returns(900, 400, 99), _returns(300, 200, 99), intersection_ids=FAKE_IDS
        )
    holed = {d: dict(v) for d, v in sumo.items()}
    del holed[233]["B0"]
    with pytest.raises(ValueError, match=r"draw 233.*'B0'"):
        calibration.per_intersection_statistics(holed, cityflow, intersection_ids=FAKE_IDS)
    with pytest.raises(ValueError, match="'C0'"):
        calibration.per_intersection_statistics(sumo, cityflow, intersection_ids=["A0", "C0"])
