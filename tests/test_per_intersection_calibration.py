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


# ==================================================================================
# The probe CHUNKS: resumable by content, one domain per prefix (Amendment A4)
# ==================================================================================
def _chunk(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": calibration.PER_INTERSECTION_FORMAT_VERSION,
        "domain": "cityflow",
        "draw_id": 201,
        "scenario_key": GRID4X4_KEY,
        "intersection_ids": list(FAKE_IDS),
        "local_return": {"A0": -20.0, "B0": -12.0},
        "local_return_from_lanes": {"A0": -20.0, "B0": -12.0},
        "two_routes_agree": True,
        "decisions": 4,
        "engine_seed_requested": 1000,
        "att_horizon": 40.0,
        "horizon_vehicle_count": 7.0,
        "config_sha256": "0" * 64,
        "seconds": 1.0,
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    payload.update(overrides)
    return payload


def _reusable(payload: dict[str, Any], **kwargs: Any) -> bool:
    defaults: dict[str, Any] = {
        "domain": "cityflow", "draw_id": 201, "scenario_key": GRID4X4_KEY,
        "intersection_ids": FAKE_IDS, "expected_decisions": 4,
    }
    defaults.update(kwargs)
    return calibration.per_intersection_chunk_is_reusable(payload, **defaults)


def test_a_chunk_is_reusable_only_when_its_own_content_still_says_so() -> None:
    """The stored ``two_routes_agree`` is exactly what a half-written chunk would lie about, so
    the two returns are compared AGAIN here, per intersection."""
    assert _reusable(_chunk()) is True
    assert _reusable(_chunk(two_routes_agree=True, local_return_from_lanes={"A0": -20.0, "B0": -11.0})) is False
    assert _reusable(_chunk(format_version="p7.3d-calibration/0.9")) is False
    assert _reusable(_chunk(draw_id=202)) is False
    assert _reusable(_chunk(domain="sumo")) is False
    assert _reusable(_chunk(scenario_key="cityflow1x1")) is False
    assert _reusable(_chunk(decisions=3)) is False
    assert _reusable(_chunk(local_return={"A0": -20.0})) is False, "an id missing from the returns"
    assert _reusable(_chunk(intersection_ids=["B0", "A0"])) is False, "the env's order is part of it"
    assert _reusable([]) is False and _reusable("text") is False  # type: ignore[arg-type]
    assert _reusable({}) is False


def test_the_two_domains_write_distinct_chunk_names(tmp_path: Path) -> None:
    names = {
        domain: calibration.per_intersection_chunk_path(domain, 201, work_dir=tmp_path).name
        for domain in calibration.PROBE_DOMAINS
    }
    assert names == {"cityflow": "probe_cityflow_draw_0201.json", "sumo": "probe_sumo_draw_0201.json"}
    with pytest.raises(ValueError, match="moss"):
        calibration.per_intersection_chunk_path("moss", 201, work_dir=tmp_path)


def test_the_cityflow_probe_writes_one_chunk_per_draw_and_resumes_from_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The run is resumable by CONTENT: a good chunk is reused without an engine, a corrupt one is
    moved aside and re-rolled, and nothing is overwritten in place."""
    work = tmp_path / "work"
    rolled: list[int] = []

    def fake_probe(*, draw_ids: Any, config_for_draw: Any, env_settings: Any, scenario_id: str, engine_seed: int) -> Any:
        out = []
        for draw_id in draw_ids:
            rolled.append(int(draw_id))
            returns = {"A0": -20.0 - draw_id, "B0": -12.0 - draw_id}
            out.append(rtg.ProbeEpisodePerIntersection(
                draw_id=int(draw_id), local_return=dict(returns),
                local_return_from_lanes=dict(returns), att_horizon=40.0,
                horizon_vehicle_count=7.0, decisions=360))
        return out

    monkeypatch.setattr(rtg, "run_probe_per_intersection", fake_probe, raising=True)
    monkeypatch.setattr(
        calibration, "_cityflow_config_for_draw",
        lambda scenario_key, out_root: (lambda d: tmp_path / "config.json"), raising=True,
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    first = calibration.run_cityflow_probe_per_intersection(
        [201, 202], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=work)
    assert rolled == [201, 202]
    assert first == {201: {"A0": -221.0, "B0": -213.0}, 202: {"A0": -222.0, "B0": -214.0}}
    chunk = calibration.per_intersection_chunk_path("cityflow", 201, work_dir=work)
    assert chunk.is_file() and json.loads(chunk.read_text(encoding="utf-8"))["domain"] == "cityflow"

    # Second run: both chunks reused, the engine never reached.
    again = calibration.run_cityflow_probe_per_intersection(
        [201, 202], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=work)
    assert rolled == [201, 202] and again == first

    # A corrupt chunk is moved aside, not overwritten, and its draw is re-rolled.
    chunk.write_text(json.dumps(_chunk(two_routes_agree=True, decisions=1)), encoding="utf-8")
    third = calibration.run_cityflow_probe_per_intersection(
        [201, 202], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=work)
    assert rolled == [201, 202, 201]
    assert third == first
    assert (work / "failed" / chunk.name).is_file(), "the bad chunk is evidence, not overwritten"

    read_back = calibration.per_intersection_returns_from_chunks("cityflow", work_dir=work)
    assert read_back == first
    assert calibration.per_intersection_returns_from_chunks("sumo", work_dir=work) == {}


# ==================================================================================
# C5: the SUMO probe per intersection -- A17(b)'s five refusals, per id where they apply
# ==================================================================================
SUMO_IDS_AND_LANES = [("A0", ["a_0", "a_1"]), ("B0", ["b_0"])]


def _engine_reads(**overrides: Any) -> dict[str, Any]:
    reads = {
        "n_teleports": 0,
        "vehicle_types_seen": ["cf_parity"],
        "time_to_teleport_option": "-1",
        "engine_seed_drawn": 437485271,
    }
    reads.update(overrides)
    return reads


def _fake_sumo_roll(
    infos: list[dict[str, Any]],
    *,
    ids_and_lanes: Any = None,
    **read_overrides: Any,
) -> Any:
    def roll(config_path: Path, *, engine_seed: int) -> Any:
        samples = [float(info["average_travel_time"]) for info in infos]
        return (
            SUMO_IDS_AND_LANES if ids_and_lanes is None else ids_and_lanes,
            infos,
            samples,
            float(infos[-1]["vehicle_count"]),
            _engine_reads(**read_overrides),
        )

    return roll


def _run_sumo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    infos: list[dict[str, Any]],
    *,
    draws: Any = (201,),
    **kwargs: Any,
) -> Any:
    monkeypatch.setattr(
        calibration, "_roll_sumo_probe_episode_per_intersection",
        _fake_sumo_roll(infos, **kwargs), raising=True,
    )
    monkeypatch.setattr(
        calibration, "_sumo_config_for_draw",
        lambda scenario_key, out_root: (lambda d: tmp_path / "noteleport.sumocfg"), raising=True,
    )
    monkeypatch.setattr(calibration, "EXPECTED_DECISIONS", len(infos), raising=True)
    (tmp_path / "noteleport.sumocfg").write_text("<configuration/>", encoding="utf-8")
    return calibration.run_sumo_probe_per_intersection(
        list(draws), scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=tmp_path / "work",
    )


def test_the_sumo_probe_records_each_intersections_return_by_two_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    returns = _run_sumo(monkeypatch, tmp_path, _infos())
    assert returns == {201: {"A0": -20.0, "B0": -12.0}}

    chunk = json.loads(
        (calibration.per_intersection_chunk_path("sumo", 201, work_dir=tmp_path / "work")
         ).read_text(encoding="utf-8")
    )
    assert chunk["domain"] == "sumo" and chunk["scenario_key"] == GRID4X4_KEY
    assert chunk["local_return"] == chunk["local_return_from_lanes"] == returns[201]
    assert chunk["intersection_ids"] == ["A0", "B0"], "the env's order (C1)"
    assert chunk["n_teleports"] == 0
    assert chunk["vehicle_types_seen"] == ["cf_parity"]
    assert chunk["time_to_teleport_option"] == "-1"
    assert chunk["engine_seed_requested"] == 1000 and chunk["engine_seed_drawn"] == 437485271
    assert chunk["format_version"] == calibration.PER_INTERSECTION_FORMAT_VERSION


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_teleports": 1}, r"draw 201: 1 teleport"),
        ({"vehicle_types_seen": ["DEFAULT_VEHTYPE"]}, r"draw 201.*DEFAULT_VEHTYPE"),
        ({"vehicle_types_seen": ["cf_parity", "pkw"]}, r"draw 201.*pkw"),
        ({"time_to_teleport_option": "300"}, r"draw 201.*'300'"),
    ],
)
def test_each_of_a17bs_engine_refusals_stops_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    """A17(b): every probe episode must show 0 teleports and the effective type ``cf_parity``,
    both READ FROM THE RUNNING ENGINE; A15(c) adds the teleport-free option.  One failure refuses
    the draw AND the run -- a probe that silently ran DEFAULT_VEHTYPE would calibrate the prompt
    against the +49 % confound the whole parity contract exists to remove."""
    with pytest.raises(ValueError, match=message):
        _run_sumo(monkeypatch, tmp_path, _infos(), **overrides)
    assert not (tmp_path / "work").exists() or not list(
        (tmp_path / "work").glob("probe_sumo_*.json")
    ), "a refused draw writes no chunk"


def test_a_two_route_disagreement_names_the_draw_and_the_intersection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=r"draw 201.*'B0'"):
        _run_sumo(monkeypatch, tmp_path, _infos(break_lane_route_of="B0"))


def test_a_single_intersection_scenario_is_refused_and_sent_to_the_scalar_entry_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="run_sumo_probe"):
        _run_sumo(monkeypatch, tmp_path, _infos(), ids_and_lanes=SUMO_IDS_AND_LANES[:1])


def test_a_short_sumo_episode_is_refused_with_both_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The decision count is the registered horizon; a truncated episode is not a shorter probe.

    ⚠️ Its own fixture: ``_run_sumo`` patches ``EXPECTED_DECISIONS`` to the episode's own length,
    so the count must be overridden AFTER it, and the intersection-count refusal (which precedes
    this one in the code) must not be the one that fires -- an earlier version of this test
    asserted the wrong message for exactly that reason.
    """
    monkeypatch.setattr(
        calibration, "_roll_sumo_probe_episode_per_intersection",
        _fake_sumo_roll(_infos()), raising=True,
    )
    monkeypatch.setattr(
        calibration, "_sumo_config_for_draw",
        lambda scenario_key, out_root: (lambda d: tmp_path / "noteleport.sumocfg"), raising=True,
    )
    (tmp_path / "noteleport.sumocfg").write_text("<configuration/>", encoding="utf-8")
    monkeypatch.setattr(calibration, "EXPECTED_DECISIONS", 99, raising=True)
    with pytest.raises(ValueError, match=r"draw 201: 4 decisions, not 99"):
        calibration.run_sumo_probe_per_intersection(
            [201], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=tmp_path / "work",
        )


def test_the_sumo_probe_resumes_by_content_and_moves_a_bad_chunk_aside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rolled: list[int] = []
    infos = _infos()

    def counting_roll(config_path: Path, *, engine_seed: int) -> Any:
        rolled.append(engine_seed)
        return _fake_sumo_roll(infos)(config_path, engine_seed=engine_seed)

    monkeypatch.setattr(
        calibration, "_sumo_config_for_draw",
        lambda scenario_key, out_root: (lambda d: tmp_path / "noteleport.sumocfg"), raising=True,
    )
    monkeypatch.setattr(calibration, "EXPECTED_DECISIONS", len(infos), raising=True)
    monkeypatch.setattr(
        calibration, "_roll_sumo_probe_episode_per_intersection", counting_roll, raising=True
    )
    (tmp_path / "noteleport.sumocfg").write_text("<configuration/>", encoding="utf-8")
    work = tmp_path / "work"

    first = calibration.run_sumo_probe_per_intersection(
        [201], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=work)
    assert len(rolled) == 1
    again = calibration.run_sumo_probe_per_intersection(
        [201], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=work)
    assert len(rolled) == 1 and again == first, "a good chunk is reused without an engine"

    chunk = calibration.per_intersection_chunk_path("sumo", 201, work_dir=work)
    payload = json.loads(chunk.read_text(encoding="utf-8"))
    payload["n_teleports"] = 2          # the engine check the resume path must re-apply
    chunk.write_text(json.dumps(payload), encoding="utf-8")
    third = calibration.run_sumo_probe_per_intersection(
        [201], scenario_key=GRID4X4_KEY, out_root=tmp_path, work_dir=work)
    assert len(rolled) == 2, "a chunk recording a teleport must NOT be reused"
    assert third == first
    assert (work / "failed" / chunk.name).is_file()


def test_a_sumo_chunk_is_not_reusable_when_its_engine_reads_are_wrong() -> None:
    def reusable(**overrides: Any) -> bool:
        payload = _chunk(domain="sumo", **{
            "n_teleports": 0, "vehicle_types_seen": ["cf_parity"],
            "time_to_teleport_option": "-1", **overrides,
        })
        return _reusable(payload, domain="sumo")

    assert reusable() is True
    assert reusable(n_teleports=1) is False
    assert reusable(vehicle_types_seen=["DEFAULT_VEHTYPE"]) is False
    assert reusable(time_to_teleport_option="300") is False
    # A CityFlow chunk carries none of those keys and must still be reusable AS CityFlow.
    assert _reusable(_chunk()) is True


# ==================================================================================
# The SHARED SUMO loop -- pinned against a committed number, and on its early type read
# ==================================================================================
class _FakeSumoEnv:
    """The smallest object ``_roll_sumo_maxpressure_episode`` reads, with a CHANGING vehicle type.

    Under the parity contract the effective type cannot change mid-episode; this env makes it
    change precisely so the early read has something to catch.  A loop that sampled the type only
    at the horizon would report ``['cf_parity']`` and the wrong early type would be invisible --
    which is the read's whole purpose.
    """

    max_steps = 3

    def __init__(self) -> None:
        self._step = 0
        self._engine_seed = 437485271
        self.reset_seed: int | None = None
        env = self

        class _Vehicle:
            @staticmethod
            def getIDList() -> list[str]:
                return ["v0"]

            @staticmethod
            def getTypeID(_vid: str) -> str:
                return "DEFAULT_VEHTYPE" if env._step == 0 else "cf_parity"

        class _Simulation:
            @staticmethod
            def getOption(_name: str) -> str:
                return "-1"

            @staticmethod
            def getStartingTeleportIDList() -> list[str]:
                # One INSIDE the loop (the last iteration reads at _step == 2) and one AFTER it
                # (the post-loop read happens at _step == 3): both reads must count, and an
                # earlier version of this fixture emitted only the first, so the assertion of 2
                # was wrong about the fixture rather than about the code.
                return ["v9"] if env._step in (2, 3) else []

        class _Sumo:
            vehicle = _Vehicle()
            simulation = _Simulation()

        self._sumo = _Sumo()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.reset_seed = seed
        self._step = 0
        return {"step": 0, "intersections": {}}

    def step(self, _action: Any) -> tuple[float, bool, bool, dict[str, Any]]:
        self._step += 1
        return 0.0, False, False, {
            "step": self._step,
            "average_travel_time": 10.0 * self._step,
            "vehicle_count": 5.0,
            "intersections": {},
        }


class _FakePolicy:
    def __init__(self) -> None:
        self.seen_steps: list[int] = []

    def act(self, info: Mapping[str, Any]) -> int:
        self.seen_steps.append(int(info["step"]))
        return 0


def test_the_shared_sumo_loop_reads_the_type_early_and_counts_teleports_across_the_horizon() -> None:
    """*Mutations this is built against:* the type read only at the horizon -> the early
    ``DEFAULT_VEHTYPE`` disappears; the teleport counter dropped after the loop -> the count falls.
    """
    env, policy = _FakeSumoEnv(), _FakePolicy()

    post_step, samples, last_count, reads = calibration._roll_sumo_maxpressure_episode(
        env, policy, engine_seed=1000
    )

    assert env.reset_seed == 1000, "the episode must reset with the requested seed"
    assert policy.seen_steps == [0, 1, 2], "the policy sees the PRE-step info at every decision"
    assert [info["step"] for info in post_step] == [1, 2, 3], "post-step infos, C6's convention"
    assert samples == [10.0, 20.0, 30.0] and last_count == 5.0
    assert reads["vehicle_types_seen"] == ["DEFAULT_VEHTYPE", "cf_parity"], (
        "the early type was not sampled, so a wrong type at the start would be invisible"
    )
    assert reads["n_teleports"] == 2, "one inside the loop at step 2, one after it"
    assert reads["time_to_teleport_option"] == "-1"
    assert reads["engine_seed_drawn"] == 437485271


def test_the_shared_sumo_loop_reproduces_p7_2bs_committed_draw_201_episode() -> None:
    """The strongest available pin on the shared loop: hz1x1's OWN registered probe episode.

    ``_roll_sumo_maxpressure_episode`` was extracted so the two probe entry points share one loop;
    an extraction is only safe if the committed numbers survive it.  This rolls draw 201 through
    the shared loop and compares every recorded field with ``docs/data/p7_2b_calibration.json``
    under ``==`` -- the SUMO counterpart of P4.3's CityFlow draw-201 reproduction, which is what
    caught the same defect on that side.

    *Mutation this is built against:* the loop records the PRE-step info -> the return moves.
    """
    import shutil

    from offline.materialise_draws import parity_sumocfg_path

    try:
        import traci  # noqa: F401
    except ImportError:
        pytest.skip("traci is not importable, so no SUMO episode can be rolled")
    if shutil.which("sumo") is None:
        pytest.skip("the sumo binary is not on PATH")
    config = parity_sumocfg_path("cityflow1x1", FIRST_PROBE_DRAW, out_root=_draws_root())
    if not config.is_file():
        pytest.skip(
            f"{config} is absent: set RLTRAFFIC_DRAWS to the scenarios/draws tree holding P7.2a's "
            f"hz1x1 parity configuration for draw {FIRST_PROBE_DRAW}"
        )

    committed = next(
        row
        for row in json.loads(
            (REPO_ROOT / "docs/data/p7_2b_calibration.json").read_text(encoding="utf-8")
        )["probe"]
        if int(row["draw_id"]) == FIRST_PROBE_DRAW
    )
    record = calibration._roll_one_episode(FIRST_PROBE_DRAW, config, engine_seed=1000)

    assert record.local_return == committed["local_return"] == -23938.0
    assert record.local_return_from_lanes == committed["local_return_from_lanes"]
    assert record.local_return == record.local_return_from_lanes, "two routes, under =="
    assert record.att_horizon == committed["att_horizon"]
    assert record.horizon_vehicle_count == committed["horizon_vehicle_count"]
    assert record.decisions == committed["decisions"] == 360
    assert record.n_teleports == committed["n_teleports"] == 0
    assert list(record.vehicle_types_seen) == committed["vehicle_types_seen"] == ["cf_parity"]
    assert record.engine_seed_drawn == committed["engine_seed_drawn"] == 437485271


# ==================================================================================
# The artifact: docs/data/p7_3d_calibration.json, assembled from BOTH halves' chunks
# ==================================================================================
def _chunk_for(domain: str, draw: int, returns: dict[str, float], **extra: Any) -> dict[str, Any]:
    payload = {
        "format_version": calibration.PER_INTERSECTION_FORMAT_VERSION,
        "domain": domain,
        "draw_id": draw,
        "scenario_key": GRID4X4_KEY,
        "intersection_ids": list(returns),
        "local_return": dict(returns),
        "local_return_from_lanes": dict(returns),
        "two_routes_agree": True,
        "decisions": 360,
        "engine_seed_requested": 1000,
        "att_horizon": 300.0,
        "horizon_vehicle_count": 9.0,
        "config_sha256": "0" * 64,
        "seconds": 1.0,
        "git_commit": "b" * 40,
        "git_dirty": False,
    }
    if domain == "sumo":
        payload.update({
            "engine_seed_drawn": 437485271, "n_teleports": 0,
            "vehicle_types_seen": ["cf_parity"], "time_to_teleport_option": "-1",
        })
    payload.update(extra)
    return payload


def _populate(work: Path, *, draws: range = range(201, 301), skip: tuple[str, int] | None = None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    sumo, cityflow = _returns(900, 400, len(draws)), _returns(300, 200, len(draws))
    for domain, source in (("cityflow", cityflow), ("sumo", sumo)):
        for offset, draw in enumerate(draws):
            if skip == (domain, draw):
                continue
            payload = _chunk_for(domain, draw, source[201 + offset])
            path = calibration.per_intersection_chunk_path(domain, draw, work_dir=work)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(tmp_path: Path, **kwargs: Any) -> Any:
    work = tmp_path / "calibration"
    _populate(work, **kwargs.pop("populate", {}))
    digests = _write_fake_checkpoints(tmp_path, {seed: _fake_payload(seed) for seed in SEEDS})
    return calibration.report_per_intersection_calibration(
        work_dir=work, out_path=tmp_path / "p7_3d_calibration.json", output_root=tmp_path,
        expected_sha256=digests, **kwargs,
    )


def test_the_artifact_carries_every_intersections_target_and_equals_exact_arithmetic(
    tmp_path: Path,
) -> None:
    """The registered prompt, recomputed here by ``Fraction`` means over the CHUNKS ON DISK.

    The artifact must be built from what the chunks say, re-read, so a resumed run's numbers come
    from where a fresh one's do.  *Mutations:* the two halves swapped in the ratio; the artifact
    built from one half's draws only.
    """
    artifact = _report(tmp_path)

    assert artifact["format_version"] == calibration.PER_INTERSECTION_FORMAT_VERSION
    assert artifact["scenario_key"] == GRID4X4_KEY
    assert artifact["registered_statistic"] == "mean" and artifact["registered_k"] == 100
    assert artifact["n_draws"] == 100 and artifact["draw_ids"] == [201, 300]

    sumo, cityflow = _returns(900, 400), _returns(300, 200)
    for ix in FAKE_IDS:
        entry = artifact["per_intersection"][ix]
        assert entry["best_source_return"] == {"A0": -132.0, "B0": -114.0}[ix]
        assert entry["rtg_scale"] == {"A0": 259.0, "B0": 232.0}[ix]
        assert entry["support_range"] == [{"A0": -259.0, "B0": -232.0}[ix], 0.0]
        for k in (5, 20, 100):
            mean_sumo, mean_cityflow = _exact_mean(sumo, ix, k), _exact_mean(cityflow, ix, k)
            cell = entry["budgets"][f"k{k}"]
            assert cell["probe_target_stat"] == mean_sumo, (ix, k)
            assert cell["probe_source_stat"] == mean_cityflow, (ix, k)
            assert cell["target"] == entry["best_source_return"] * (mean_sumo / mean_cityflow)
            assert cell["role"] == ("registered_prompt" if k == 100 else "recorded_not_evaluated")
    registered = [
        ix for ix in artifact["per_intersection"]
        if artifact["per_intersection"][ix]["budgets"]["k100"]["role"] == "registered_prompt"
    ]
    assert registered == FAKE_IDS, "every intersection carries exactly one registered prompt"


def test_the_artifact_records_where_each_constant_was_read_from(tmp_path: Path) -> None:
    """Amendment A4: the artifact states the FIELD each constant came from and why.

    A17(e)'s wording pointed at ``stats["rtg"]`` for ``R_best_source``; the payload says otherwise
    (that block is per-window and its ``max`` is 0.0 on every intersection), and an artifact that
    did not say which field it read would leave a reader to re-derive the answer.
    """
    artifact = _report(tmp_path)
    provenance = artifact["fields_read"]
    assert provenance["best_source_return"].startswith('payload["target_rtg"]')
    assert provenance["rtg_scale"].startswith('payload["rtg_scale"]')
    assert "per-window" in provenance["support_range"]
    assert "A4" in json.dumps(provenance)
    assert artifact["subject"] == "mappo1000_dt_nomix_h4"
    assert sorted(artifact["checkpoint_sha256"]) == ["101", "202", "303", "404", "505"]
    assert artifact["disjointness"]["training_draw_ids"] == [1, 200]
    assert artifact["disjointness"]["probe_draw_ids"] == [201, 300]
    assert artifact["disjointness"]["held_out_draw_ids"] == [1000, 1099]
    assert artifact["what_this_does_not_say"], "the artifact must say what it is not"


def test_every_refusal_precedes_the_write(tmp_path: Path) -> None:
    """A refused report creates NO file -- the filesystem-mutation barrier, at the last write."""
    out = tmp_path / "p7_3d_calibration.json"
    with pytest.raises(ValueError, match=r"100 cityflow chunk\(s\) against 99 sumo chunk\(s\)"):
        _report(tmp_path, populate={"skip": ("sumo", 300)})
    assert not out.exists(), "a refused report must not leave a partial artifact"

    with pytest.raises(ValueError, match=r"k=100 needs 100"):
        _report(tmp_path / "short", populate={"draws": range(201, 251)})
    assert not (tmp_path / "short" / "p7_3d_calibration.json").exists()


def test_a_probe_band_overlapping_the_training_draws_is_refused(tmp_path: Path) -> None:
    """A17(b)'s disjointness, against the draws the CHECKPOINT says it trained on."""
    work = tmp_path / "calibration"
    _populate(work, draws=range(151, 251))
    digests = _write_fake_checkpoints(tmp_path, {seed: _fake_payload(seed) for seed in SEEDS})
    with pytest.raises(ValueError, match="not disjoint"):
        calibration.report_per_intersection_calibration(
            work_dir=work, out_path=tmp_path / "p7_3d_calibration.json",
            output_root=tmp_path, expected_sha256=digests,
        )


def test_the_artifact_regenerates_byte_identically(tmp_path: Path) -> None:
    out = tmp_path / "p7_3d_calibration.json"
    _report(tmp_path)
    first = out.read_bytes()
    _report(tmp_path)
    assert out.read_bytes() == first
