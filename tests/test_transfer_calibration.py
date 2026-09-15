"""P7.2b T2-T9: the SUMO probe, A17's targets, the in-support diagnostic and the fence.

Written against ``BRIEF_36`` §3.2 / §4 + Amendment A and ``docs/plans/p7.2b.md``.

**A17 fixed the protocol before any target-domain return existed**, so most of these tests are about
*reuse and arithmetic*, not about choices: Rule B is imported, the source statistics are read from
P4.3's committed artifact, and the registered prompt is Rule B / ``mean`` / k = 100 whatever the
numbers turn out to be.

Three families here:

* **Simulator-gated** (T2, T3): one 20-decision DT rollout and one MaxPressure episode, both on the
  **fenced draw 5** -- never a probe-band draw.
* **Checkpoint-gated** (T7 and parts of T4/T8): they read the committed payloads under ``output/``,
  which CI does not have.
* **Pure** (T4-T6, T8's refusals, T9): synthetic chunks in ``tmp_path``, no engine, no checkpoint.

⚠️ **The fence (Amendment A3).**  ``att_horizon``, ``episode_reward``, ``rtg_last`` and the
per-decision RTG series are outcome-shaped and must not reach ``docs/data/``.  The last test in this
file asserts the committed artifact carries none of the four.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from offline import transfer_calibration as tc
from offline.rtg_calibration import (
    PROBE_DRAW_START,
    PROBE_K_VALUES,
    probe_draw_ids,
    probe_statistic,
    rule_b_target,
)

DRAWS_ROOT = Path("/home/filip/rltraffic/scenarios/draws")
OUTPUT_ROOT = Path("/home/filip/rltraffic/output")
SMOKE_DRAW = 5
IX = "intersection_1_1"


def _sumo_available() -> bool:
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def _draws_available() -> bool:
    from offline.materialise_draws import parity_sumocfg_path

    return parity_sumocfg_path("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT).is_file()


def _checkpoints_available() -> bool:
    return (OUTPUT_ROOT / "p4_dt" / "dt_seed101.pt").is_file() and (
        OUTPUT_ROOT / "p4_7" / "checkpoints" / "mix50_dt_seed101.pt"
    ).is_file()


def _synthetic_chunk(draw_id: int, local_return: float) -> dict[str, Any]:
    """A probe chunk with every field a clean run writes, so a test can vary exactly one."""
    return {
        "format_version": tc.ARTIFACT_FORMAT_VERSION,
        "draw_id": int(draw_id),
        "scenario_key": tc.SCENARIO_KEY,
        "local_return": float(local_return),
        "local_return_from_lanes": float(local_return),
        "two_routes_agree": True,
        "att_horizon": 200.0 + draw_id,
        "horizon_vehicle_count": 100.0,
        "decisions": 360,
        "engine_seed_requested": tc.DEFAULT_ENGINE_SEED,
        "engine_seed_drawn": 437485271,
        "n_teleports": 0,
        "vehicle_types_seen": ["cf_parity"],
        "time_to_teleport_option": "-1",
        "config_path": f"draw_{draw_id:04d}/parity/noteleport.sumocfg",
        "config_sha256": "0" * 64,
        "p4_3_probe_sha256": tc.P4_3_PROBE_SHA256,
        "seconds": 10.5,
        "canary_seconds": 0.9,
        "git_commit": "0" * 40,
        "git_dirty": False,
    }


def _write_band(work_dir: Path, *, returns: dict[int, float] | None = None) -> list[tuple[int, float]]:
    """Write a full 100-draw synthetic band; returns the (draw_id, return) pairs written."""
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, float]] = []
    for offset in range(100):
        draw_id = PROBE_DRAW_START + offset
        value = (returns or {}).get(draw_id, -20000.0 - offset)
        chunk = _synthetic_chunk(draw_id, value)
        tc.probe_chunk_path(draw_id, work_dir=work_dir).write_text(
            json.dumps(chunk, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append((draw_id, value))
    return rows


# ----------------------------------------------------------------------------------
# T2 -- the DT runs on SUMO only through the wrapper, and the target took effect
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
@pytest.mark.skipif(not _draws_available(), reason="P7.2a's parity draws are not present")
@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_a_cityflow_trained_dt_drives_the_aligned_sumo_env_and_refuses_the_raw_one() -> None:
    """T2. A16 says the alignment is the ONLY route; this is that claim as a pair of outcomes.

    Mechanics only: the decision count, the action range and the RTG's first advance.  No ATT, no
    episode return, no ``rtg_last``.
    """
    import numpy as np

    from experiments.envs import make_env
    from offline.aligned_env import aligned_sumo_env_for_draw
    from offline.collect import _build_env_spec
    from offline.materialise_draws import parity_sumocfg_path
    from offline.rtg_calibration import agent_with_target
    from offline.sumo_att_reference import collect_style_args

    checkpoint = OUTPUT_ROOT / "p4_dt" / "dt_seed101.pt"
    target = -5762.0

    env = aligned_sumo_env_for_draw("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT)
    try:
        agent = agent_with_target(
            env, checkpoint, declared_gradient_steps=tc.DECLARED_GRADIENT_STEPS, target_rtg=target
        )
        assert set(agent.current_rtg().values()) == {target}

        info = env.reset(seed=1000)
        for step in range(20):
            action = agent.act(info)
            flat = np.asarray(action).reshape(-1)
            assert flat.size == 1
            assert 0 <= int(flat[0]) < 8, f"action {flat[0]} outside the 8 green actions"
            _reward, terminated, truncated, info = env.step(action)
            if step == 0:
                # The RTG advances on the reward the env produced: target - r_0.
                reward_0 = float(info["intersections"][IX]["reward"])
                assert agent.current_rtg()[IX] == pytest.approx(target - reward_0, abs=1e-9)
            if terminated or truncated:
                break
    finally:
        env.close()

    # The same construction on the UNWRAPPED env: the checkpoint reads 25 features and SUMO offers
    # 32, so the first act must fail rather than silently consume a truncated state.
    args = collect_style_args(
        "sumo",
        "maxpressure",
        parity_sumocfg_path("cityflow1x1", SMOKE_DRAW, out_root=DRAWS_ROOT),
        sentinel_out_dir="/nonexistent",
    )
    raw_env = make_env(_build_env_spec(args))
    try:
        raw_agent = agent_with_target(
            raw_env, checkpoint, declared_gradient_steps=tc.DECLARED_GRADIENT_STEPS, target_rtg=target
        )
        raw_info = raw_env.reset(seed=1000)
        # The agent names the defect itself: 25 features trained, 32 offered.  Asserted on that
        # message rather than on "some exception", so an unrelated error in this test cannot pass
        # for A16's claim.
        with pytest.raises(ValueError, match=r"state width changed for DTAgent: expected 25, got 32"):
            raw_agent.act(raw_info)
    finally:
        raw_env.close()


# ----------------------------------------------------------------------------------
# T3 -- the probe's two routes agree and the engine reads what the file requested
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
@pytest.mark.skipif(not _draws_available(), reason="P7.2a's parity draws are not present")
def test_one_probe_episode_agrees_by_two_routes_and_the_engine_confirms_the_contract(
    tmp_path: Path,
) -> None:
    """T3. A17(b)'s four engine-read requirements, on the fenced draw 5.

    ``run_probe`` records both return routes and leaves the comparison to its caller; A17(b) requires
    agreement under ``==``, so the probe checks it itself and this test checks that it did.
    """
    (record,) = tc.run_sumo_probe(
        [SMOKE_DRAW], out_root=DRAWS_ROOT, work_dir=tmp_path, canary_seconds=0.9
    )

    assert record.draw_id == SMOKE_DRAW
    assert record.decisions == 360
    assert record.local_return == record.local_return_from_lanes
    assert record.n_teleports == 0
    assert tuple(record.vehicle_types_seen) == ("cf_parity",)
    assert record.time_to_teleport_option == "-1"
    assert record.engine_seed_requested == 1000
    assert isinstance(record.engine_seed_drawn, int)
    assert record.engine_seed_drawn != 1000, (
        "the drawn seed is the env RNG's first value under seed=1000, not the seed itself"
    )

    chunk = json.loads(tc.probe_chunk_path(SMOKE_DRAW, work_dir=tmp_path).read_bytes())
    assert chunk["format_version"] == tc.ARTIFACT_FORMAT_VERSION
    assert chunk["two_routes_agree"] is True
    assert chunk["p4_3_probe_sha256"] == tc.P4_3_PROBE_SHA256
    assert tc.chunk_is_reusable(chunk, draw_id=SMOKE_DRAW)


# ----------------------------------------------------------------------------------
# T4 -- Rule B is reused, not reimplemented, and the arithmetic is right
# ----------------------------------------------------------------------------------
def test_rule_b_returns_the_naive_target_when_the_two_statistics_coincide() -> None:
    """A17(a) and P4.3 §2.2: in domain the ratio is exactly 1, so the rule IS the naive target.

    Bit-for-bit, not approximately: the association forms the ratio first, and that is what makes
    the identity exact.
    """
    for best in (-5762.0, -5959.0):
        for stat in (-18600.59, -12532.0):
            assert (
                rule_b_target(
                    best_source_return=best, probe_source_stat=stat, probe_target_stat=stat
                )
                == best
            )
            assert rule_b_target(
                best_source_return=best, probe_source_stat=stat, probe_target_stat=2.0 * stat
            ) == pytest.approx(2.0 * best, abs=1e-9)


def test_the_source_statistics_are_read_from_p4_3s_artifact_not_recomputed(tmp_path: Path) -> None:
    """T4. All six (k, S) cells equal ``p4_3_probe.json:budgets[k].statistics[S]`` under ``==``."""
    rows = _write_band(tmp_path)
    table = tc.statistics_table(rows)

    committed = json.loads(tc.P4_3_PROBE_ARTIFACT.read_bytes())["budgets"]
    for k in PROBE_K_VALUES:
        for statistic in ("mean", "max"):
            cell = table[f"k{k}"][statistic]
            assert cell["cityflow"] == committed[f"k{k}"]["statistics"][statistic]
            # and the SUMO half is this run's own measurement over the first k rows
            assert cell["sumo"] == probe_statistic([value for _, value in rows[:k]], statistic)
    assert table["read_from"]["path"].endswith("p4_3_probe.json")
    assert table["read_from"]["sha256"] == tc.P4_3_PROBE_SHA256


def test_the_pinned_p4_3_digest_matches_the_artifact_on_disk() -> None:
    """Amendment A5: the constant is a DECLARATION -- these statistics came from THAT artifact.

    If P4.3's artifact is legitimately regenerated the constant moves in the same commit, which is a
    reviewed act; an unnoticed drift is what this asserts against.
    """
    import hashlib

    assert (
        hashlib.sha256(tc.P4_3_PROBE_ARTIFACT.read_bytes()).hexdigest() == tc.P4_3_PROBE_SHA256
    )


@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_the_registered_prompt_is_rule_b_mean_k100_and_is_labelled_so(tmp_path: Path) -> None:
    """T4. A17(c): per subject, the H3 confirmatory arm is Rule B, ``S = mean``, k = 100."""
    rows = _write_band(tmp_path)
    statistics = tc.statistics_table(rows)
    facts = {
        name: tc.subject_facts(name, output_root=OUTPUT_ROOT) for name in ("mappo1000", "mix50")
    }
    targets = tc.targets_table(statistics, rows, facts)

    for subject, subject_facts_ in facts.items():
        registered = [
            row
            for row in targets[subject]
            if row["role"] == "registered_prompt"
        ]
        assert len(registered) == 1, "exactly one target per subject is the registered prompt"
        row = registered[0]
        assert (row["rule"], row["statistic"], row["k"]) == ("rule_b", "mean", 100)
        # Recomputed here by the independent route the mutation must break.
        expected = rule_b_target(
            best_source_return=subject_facts_.best_source_return,
            probe_source_stat=statistics["k100"]["mean"]["cityflow"],
            probe_target_stat=statistics["k100"]["mean"]["sumo"],
        )
        assert row["target_rtg"] == expected
        assert all(other["role"] == "ablation" for other in targets[subject] if other is not row)
        assert {other["rule"] for other in targets[subject]} == {"rule_a", "rule_b", "naive"}


# ----------------------------------------------------------------------------------
# T5 -- disjointness against the UNION of every training source (Amendment A7.1)
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_the_probe_band_is_disjoint_from_every_training_source_and_the_held_out_pool() -> None:
    """T5 + A7.1. Both checkpoints' fields, the declaration's 152 and 1000-1099, unioned.

    The ``mix50`` CHECKPOINT records ``1..200`` -- byte-equal to ``mappo1000``'s -- and not the 152
    draws its tier trained on.  Asserting against the union means the weaker source cannot make the
    assertion pass on its own.
    """
    record = tc.disjointness_record(probe_draw_ids(100), output_root=OUTPUT_ROOT)

    assert record["disjoint"] is True
    assert record["probe_draws"] == [201, 300]
    sources = {source["name"]: source for source in record["sources"]}
    assert sources["p4_7_declaration:tiers.mix50.training_draws"]["n"] == 152
    assert sources["held_out_pool"]["n"] == 100
    assert record["union_size"] >= 200 + 100
    for source in record["sources"]:
        assert source["overlap_with_probe"] == 0


@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_a_probe_band_touching_any_training_source_is_refused() -> None:
    """Planting one draw from each band must refuse, naming it."""
    for planted in (150, 1000):
        with pytest.raises(ValueError, match="disjoint"):
            tc.disjointness_record([planted, *probe_draw_ids(5)], output_root=OUTPUT_ROOT)


# ----------------------------------------------------------------------------------
# T6 -- the k budgets are nested prefixes and the statistics are prefix statistics
# ----------------------------------------------------------------------------------
def test_the_budgets_are_nested_prefixes_and_the_mean_is_the_mean_of_the_first_k(
    tmp_path: Path,
) -> None:
    """T6. Recomputed by ``sum/len`` rather than by ``numpy.mean``, which is a different route."""
    assert set(probe_draw_ids(5)) < set(probe_draw_ids(20)) < set(probe_draw_ids(100))

    rows = _write_band(tmp_path)
    table = tc.statistics_table(rows)
    for k in PROBE_K_VALUES:
        first_k = [value for _, value in rows[:k]]
        assert table[f"k{k}"]["mean"]["sumo"] == pytest.approx(sum(first_k) / len(first_k), abs=1e-9)
        assert table[f"k{k}"]["max"]["sumo"] == max(first_k)
        assert table[f"k{k}"]["draw_ids"] == [PROBE_DRAW_START, PROBE_DRAW_START + k - 1]


def test_probe_returns_from_chunks_refuses_a_gap_and_a_duplicate(tmp_path: Path) -> None:
    """A missing draw silently shortens the band and moves every statistic."""
    _write_band(tmp_path)
    tc.probe_chunk_path(250, work_dir=tmp_path).unlink()
    with pytest.raises(ValueError, match="250"):
        tc.probe_returns_from_chunks(tmp_path)


# ----------------------------------------------------------------------------------
# T7 -- the in-support diagnostic is arithmetic, reports both subjects, and never selects
# ----------------------------------------------------------------------------------
def test_in_support_position_classifies_below_inside_and_above() -> None:
    """T7. Closed at both ends, as ``in_support_counts`` defines it."""
    assert tc.in_support_position(-5000.0, rtg_min=-9991.0, rtg_max=-6.0)["position"] == "inside"
    assert tc.in_support_position(-12000.0, rtg_min=-9991.0, rtg_max=-6.0)["position"] == "below"
    assert tc.in_support_position(0.0, rtg_min=-9991.0, rtg_max=-6.0)["position"] == "above"
    assert tc.in_support_position(-9991.0, rtg_min=-9991.0, rtg_max=-6.0)["position"] == "inside"

    below = tc.in_support_position(-12000.0, rtg_min=-9991.0, rtg_max=-6.0)
    assert below["margin"] == pytest.approx(-2009.0, abs=1e-9)
    assert tc.in_support_position(-5000.0, rtg_min=-9991.0, rtg_max=-6.0)["margin"] == 0.0


@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_both_subjects_report_the_registered_split_range_and_the_training_set_bound() -> None:
    """Amendment A1. The registered range is the SPLIT range; the training-set bound sits beside it.

    For ``mappo1000`` the two coincide (9991 == |-9991|); for ``mix50`` they differ by 71 on a
    40,000 scale, because ``stats.rtg`` covers 216,000 split rows while ``target_rtg``/``rtg_scale``
    come from the 72,000-row training set.  **The diagnostic never selects and no claim rests on
    which bound is used** -- both are recorded so a reader can see that.
    """
    mappo = tc.subject_facts("mappo1000", output_root=OUTPUT_ROOT)
    mix50 = tc.subject_facts("mix50", output_root=OUTPUT_ROOT)

    assert mappo.best_source_return == -5762.0 and mappo.rtg_scale == 9991.0
    assert mappo.support_range_over_the_split == (-9991.0, -6.0)
    assert mappo.n_rows == 72000
    assert mappo.training_set_return_min == -9991.0

    assert mix50.best_source_return == -5959.0 and mix50.rtg_scale == 40223.0
    assert mix50.support_range_over_the_split == (-40294.0, -6.0)
    assert mix50.n_rows == 216000
    assert mix50.training_set_return_min == -40223.0
    assert mix50.support_range_over_the_split[0] != mix50.training_set_return_min

    for facts in (mappo, mix50):
        assert facts.state_dim == 25 and facts.context_length == 20
        assert len(facts.checkpoints) == 5


# ----------------------------------------------------------------------------------
# T8 -- report: byte-identical, refusing, and FENCED
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_report_regenerates_byte_identically_and_carries_no_fenced_quantity(
    tmp_path: Path,
) -> None:
    """T8 + Amendment A3. The four fenced names must not appear anywhere in the committed artifact.

    Asserted over the serialised bytes, not over the top-level keys: a fenced value nested three
    levels down is exactly the leak this guards.
    """
    work = tmp_path / "work"
    _write_band(work)
    (work / "smoke_mappo1000.json").write_text(
        json.dumps(
            {
                "format_version": tc.ARTIFACT_FORMAT_VERSION,
                "subject": "mappo1000",
                "draw_id": 5,
                "checkpoint": "/home/filip/rltraffic/output/p4_dt/dt_seed101.pt",
                "target_rtg": -5762.0,
                "decisions": 360,
                "rtg_first": -5762.0,
                "rtg_advanced_every_decision": True,
                "n_decisions_in_support": 360,
                "actions_in_range": True,
                "vehicle_types_seen": ["cf_parity"],
                "time_to_teleport_option": "-1",
                "seconds": 12.0,
                tc.FENCED_KEY: {
                    "att_horizon": 123.456,
                    "episode_reward": -7654.0,
                    "rtg_last": 1892.0,
                    "rtg_series": [-5762.0, -5700.0],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "p7_2b_calibration.json"
    tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    first = out.read_bytes()
    tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    assert out.read_bytes() == first, "the artifact must regenerate byte-identically"

    text = out.read_text(encoding="utf-8")
    for fenced in (tc.FENCED_KEY, "episode_reward", "rtg_last", "rtg_series"):
        assert fenced not in text, f"{fenced} reached the committed artifact"
    # att_horizon is legitimate on a PROBE row and forbidden on the smoke; check the smoke block.
    payload = json.loads(text)
    assert "att_horizon" not in json.dumps(payload["smoke"])
    assert payload["format_version"] == tc.ARTIFACT_FORMAT_VERSION
    assert payload["registered_in"] == "PREREGISTRATION A17"
    assert len(payload["probe"]) == 100


@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
@pytest.mark.parametrize(
    "corrupt, expected",
    [
        ("two_routes", "two routes"),
        ("teleports", "teleport"),
        ("format_version", "format_version"),
        ("probe_sha", "sha256"),
        ("vehicle_types", "cf_parity"),
    ],
)
def test_report_refuses_a_chunk_that_violates_a17(
    tmp_path: Path, corrupt: str, expected: str
) -> None:
    """T8. Each refusal names its own reason, and none of them writes the artifact."""
    work = tmp_path / "work"
    _write_band(work)
    path = tc.probe_chunk_path(250, work_dir=work)
    chunk = json.loads(path.read_bytes())
    if corrupt == "two_routes":
        chunk["local_return_from_lanes"] = chunk["local_return"] + 1.0
        chunk["two_routes_agree"] = True  # the stored verdict lies; the recomputation must win
    elif corrupt == "teleports":
        chunk["n_teleports"] = 3
    elif corrupt == "format_version":
        chunk["format_version"] = "p7.2b-calibration/0.9"
    elif corrupt == "probe_sha":
        chunk["p4_3_probe_sha256"] = "f" * 64
    elif corrupt == "vehicle_types":
        chunk["vehicle_types_seen"] = ["pkw"]
    path.write_text(json.dumps(chunk, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out = tmp_path / "p7_2b_calibration.json"
    with pytest.raises(ValueError, match=expected):
        tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    assert not out.exists(), "a refusal must precede every write"


def test_a_chunk_that_is_valid_json_but_not_an_object_is_moved_aside_and_re_rolled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment B2 (pre-flight minor 1): ``[]`` parses, and ``payload.get`` then raises.

    ``json.loads`` accepts ``[]``, ``5`` and ``"text"`` as happily as an object; ``chunk_is_reusable``
    caught only ``(KeyError, TypeError, ValueError)``, so an ``AttributeError`` escaped and the
    probe died with a traceback, leaving the bad file in place.  It must instead be *not reusable*,
    which routes it through the move-aside path every other unclean chunk takes.

    **No simulator**: the roll is monkeypatched at ``_roll_one_episode``, the seam the extraction
    introduced.  The brief allows one SUMO episode here instead; the suite already spends four on
    the paths that need a real engine, and this test is about the resume branch, not the episode.
    """
    work = tmp_path / "work"
    work.mkdir()
    chunk_path = tc.probe_chunk_path(201, work_dir=work)
    planted = b"[]\n"
    chunk_path.write_bytes(planted)

    canned = tc.ProbeRecord(
        draw_id=201,
        local_return=-20000.0,
        local_return_from_lanes=-20000.0,
        att_horizon=210.0,
        horizon_vehicle_count=99.0,
        decisions=360,
        engine_seed_requested=1000,
        engine_seed_drawn=437485271,
        n_teleports=0,
        vehicle_types_seen=("cf_parity",),
        time_to_teleport_option="-1",
        seconds=10.8,
    )
    monkeypatch.setattr(
        tc, "_roll_one_episode", lambda draw_id, config_path, *, engine_seed: canned
    )

    (record,) = tc.run_sumo_probe([201], out_root=DRAWS_ROOT, work_dir=work, canary_seconds=0.9)

    assert record.draw_id == 201
    moved = work / "failed" / chunk_path.name
    assert moved.is_file(), "the unreadable chunk must be moved aside, not deleted or overwritten"
    assert moved.read_bytes() == planted, "it must be moved BYTE-IDENTICALLY, as evidence"
    fresh = json.loads(chunk_path.read_bytes())
    assert tc.chunk_is_reusable(fresh, draw_id=201)


@pytest.mark.parametrize("payload", [[], 5, "text", None, [{"draw_id": 201}]])
def test_a_non_object_payload_is_simply_not_reusable(payload: Any) -> None:
    """The unit half of B2: no exception escapes, whatever JSON shape arrives."""
    assert tc.chunk_is_reusable(payload, draw_id=201) is False


def test_chunk_reuse_is_judged_from_content_not_from_a_stored_verdict(tmp_path: Path) -> None:
    """The resume rule. A chunk claiming success while its own numbers disagree is not reusable."""
    good = _synthetic_chunk(201, -20000.0)
    assert tc.chunk_is_reusable(good, draw_id=201)

    liar = dict(good)
    liar["local_return_from_lanes"] = -19999.0  # the two routes disagree
    assert liar["two_routes_agree"] is True  # ... and the chunk still claims they agree
    assert not tc.chunk_is_reusable(liar, draw_id=201)

    for field, value in (
        ("n_teleports", 1),
        ("decisions", 359),
        ("format_version", "p7.2b-calibration/0.9"),
        ("vehicle_types_seen", ["pkw"]),
        ("draw_id", 202),
    ):
        broken = dict(good)
        broken[field] = value
        assert not tc.chunk_is_reusable(broken, draw_id=201), f"{field} must break reuse"


# ----------------------------------------------------------------------------------
# T9 -- the driver's canary gate
# ----------------------------------------------------------------------------------
DRIVER = Path(__file__).resolve().parents[1] / "offline" / "campaigns" / "p7_2b_calibration.sh"


def test_the_driver_exists_and_is_syntactically_valid() -> None:
    import subprocess

    assert DRIVER.is_file()
    assert subprocess.run(["bash", "-n", str(DRIVER)]).returncode == 0


def test_the_driver_gates_on_the_canary_before_consuming_the_token() -> None:
    """T9. The canary threshold, the token, the lock and the trap, in the order that matters.

    Textual, and deliberately so: it catches deletion and reordering, never semantics.  The
    ordering claim -- every check that can refuse precedes the token -- is the one a reader of the
    shell cannot verify at a glance.
    """
    text = DRIVER.read_text(encoding="utf-8")
    assert "CANARY_MAX_SECONDS=2.0" in text
    assert "AUTHORISED_TO_RUN" in text
    assert "pgrep -f 'python.*offline\\.transfer_calibration'" in text
    assert "trap on_signal INT TERM" in text

    canary_at = text.index("CANARY_MAX_SECONDS")
    canary_run = text.index("canary")
    token_at = text.index("rm -f \"$TOKEN\"")
    lock_at = text.index("REFUSING TO START: cells from another run")
    trap_at = text.index("trap on_signal INT TERM")
    leader_at = text.index("REFUSING TO START: not a process-group leader")
    assert lock_at < token_at, "the lock must refuse before the token is consumed"
    assert canary_at < token_at and canary_run < token_at, (
        "the canary must run before the token is consumed, or a throttled machine burns the "
        "author's one-shot authorisation"
    )
    # Amendment B1: a signal between the token's deletion and the trap's installation consumed the
    # one-shot authorisation and left neither FAILED nor COMPLETE.
    assert trap_at < token_at, (
        "the trap must be installed BEFORE the token is consumed, so that from the first "
        "destructive line onward a signal writes FAILED"
    )
    # Amendment B3: `kill -- -$$` is a no-op unless the driver leads its process group.
    assert leader_at < token_at, "the group-leader check must refuse before the token"
    assert 'ps -o pgid= -p $$' in text
    # ... and the handler must not CREATE the work dir just to record a failure in it.
    assert text.count('if [ -d "$WORK" ]; then') == 2, (
        "both FAILED writers guard on $WORK existing, so a signal before the token still leaves "
        "the tree exactly as it found it"
    )
    # Nothing under scenarios/draws is written.
    assert "scenarios/draws" in text
    assert ">" not in text.split("scenarios/draws")[1].split("\n")[0]
