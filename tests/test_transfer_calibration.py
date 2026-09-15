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
import math
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
REPO_DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
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
        "canary_seconds": CHUNK_CANARY_SECONDS,
        "git_commit": "0" * 40,
        "git_dirty": False,
    }


#: Amendment E1.4. The chunks' ``canary_seconds`` is 0.9 and this run's is 0.5, DELIBERATELY
#: different: run 3's artifact reported the chunks' 0.89 as if it were the reporting run's, and a
#: fixture in which the two coincide cannot tell that defect from its fix.
CHUNK_CANARY_SECONDS = 0.9
THIS_RUN_CANARY_SECONDS = 0.5


def _write_canary_record(
    work_dir: Path, *, seconds: float = THIS_RUN_CANARY_SECONDS, **fact_overrides: Any
) -> Path:
    """Write the ``canary.json`` the driver parks in the work directory after the token.

    ``report`` refuses a work directory without one (Amendment E1.4 item 2), so every fixture that
    calls ``report`` needs it; ``_write_band`` therefore writes one and the tests that exercise the
    refusals remove or corrupt it afterwards.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    facts = _canary_facts(**fact_overrides)
    path = work_dir / tc.CANARY_RECORD_NAME
    # The line is built here rather than by ``tc.format_canary_line`` so that every test using
    # ``_write_band`` stays independent of the formatter; the formatter's exact bytes are pinned
    # against a hand-written literal in ``test_the_canary_line_round_trips_through_record_canary``.
    line = f"canary {seconds:.2f} s {json.dumps(facts, sort_keys=True)}"
    path.write_text(
        json.dumps(
            {
                "format_version": tc.ARTIFACT_FORMAT_VERSION,
                "line": line,
                "seconds": float(seconds),
                "facts": facts,
                "threshold_seconds": tc.CANARY_MAX_SECONDS,
                "git_commit": "1" * 40,
                "git_dirty": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_band(work_dir: Path, *, returns: dict[int, float] | None = None) -> list[tuple[int, float]]:
    """Write a full 100-draw synthetic band; returns the (draw_id, return) pairs written.

    It also writes the work directory's ``canary.json``: a directory holding chunks but no canary
    record is, from Amendment E1.4 on, not a run, and ``report`` refuses it.
    """
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
    _write_canary_record(work_dir)
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
# Amendment D1 -- the RTG-advance check, whose alignment is shifted by one on purpose
# ----------------------------------------------------------------------------------
def test_the_rtg_advance_check_uses_the_PREVIOUS_infos_reward() -> None:
    """Amendment D1, on the campaign's own false negative.

    ``run_smoke`` reads ``current_rtg()`` **before** ``act(info_t)`` and the agent updates
    ``reward_sum`` inside that call, so the change at ``t`` is driven by ``r(info_{t-1})``.  The
    first version compared it with ``r(info_t)`` and the completed campaign recorded ``False`` for
    both subjects while the agent was working correctly.

    The synthetic series below is the discriminator Amendment D1 names: rewards ``[0, 0, -2, -4]``
    give ``rtg = [T, T, T, T+2]`` -- the flag is ``True`` under the shifted alignment and ``False``
    under the unshifted one, so a regression cannot pass this test.
    """
    target = -7185.0
    rewards: list[float | None] = [0.0, 0.0, -2.0, -4.0]
    series = [target, target, target, target + 2.0]

    assert tc.rtg_advanced_every_decision(series, rewards) is True

    # What the unshifted comparison would have concluded, recomputed here rather than asserted
    # from memory: at index 2 it looks at r(info_2) = -2 and expects a change that belongs to
    # index 3.
    unshifted = all(
        (series[i] != series[i - 1]) == (rewards[i] != 0.0) for i in range(1, len(series))
    )
    assert unshifted is False, "the synthetic series must discriminate the two alignments"


def test_index_one_is_forced_unchanged_by_the_agents_step_zero_rule() -> None:
    """``DTAgent.act`` adds ``0.0 if step == 0`` whatever the info carries, so the first
    transition is structurally zero and is not evidence either way."""
    target = -100.0
    # A non-zero reward on info_0 that the agent ignores: the RTG must NOT move into index 1.
    assert tc.rtg_advanced_every_decision([target, target, target - 5.0], [-9.0, 5.0, 0.0]) is True
    # ... and a series that DOES move there contradicts the agent's rule.
    assert tc.rtg_advanced_every_decision([target, target + 1.0, target + 1.0], [0.0, 0.0, 0.0]) is False


def test_the_rtg_advance_check_refuses_a_missing_reward_and_a_length_mismatch() -> None:
    """An absent reward cannot be shown to be zero, so it cannot support a True."""
    assert tc.rtg_advanced_every_decision([1.0, 1.0, 1.0], [0.0, None, 0.0]) is False
    with pytest.raises(ValueError, match="per decision"):
        tc.rtg_advanced_every_decision([1.0, 1.0], [0.0])


# ----------------------------------------------------------------------------------
# Amendment E1 -- the smoke drives the DT on the DECLARED EVALUATION PATH
# ----------------------------------------------------------------------------------
def test_every_act_call_in_a_smoke_passes_explore_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment E1. The first smoke took ``DTAgent.act``'s default, which SAMPLES.

    ``explore=True`` draws from the masked softmax through an unseeded ``torch.multinomial``, so
    the same seed, checkpoint and draw gave ``n_decisions_in_support`` 271 then 231.  Fifteen
    evaluation call sites in this repository pass ``explore=False`` -- the argmax the agent's own
    docstring calls *the declared evaluation path* -- and the smoke must match them.

    The spy records the kwargs of **every** call, not just the first: a loop that got it right once
    and then fell back to a default would pass a first-call-only assertion.  No simulator is needed
    -- the env, the agent and the rollout are substituted, which leaves ``run_smoke``'s own
    ``choose`` (where the call shape lives) as the only real code under test.
    """
    import numpy as np

    from offline.horizon_metric import HorizonRollout

    calls: list[dict[str, Any]] = []
    ix_id = "intersection_1_1"

    class _Agent:
        def current_rtg(self) -> dict[str, float]:
            return {ix_id: -7185.0 - len(calls)}

        def act(self, info: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
            calls.append(dict(kwargs))
            return np.zeros(1, dtype=np.int64)

    class _Sumo:
        class simulation:  # noqa: N801 - mirrors traci's module shape
            @staticmethod
            def getOption(name: str) -> str:
                return "-1"

        class vehicle:  # noqa: N801
            @staticmethod
            def getIDList() -> list[str]:
                return ["v0"]

            @staticmethod
            def getTypeID(vehicle_id: str) -> str:
                return "cf_parity"

    class _Env:
        max_steps = 3
        intersections = [type("Ix", (), {"id": ix_id})()]
        _sumo = _Sumo()

        def close(self) -> None:
            pass

    def fake_rollout(env: Any, choose: Any, episodes: int, seed: int) -> HorizonRollout:
        info = {"intersections": {ix_id: {"reward": -2.0}}}
        for _ in range(3):
            choose(env, info)
        return HorizonRollout(
            att_horizon=1.0,
            att_running_mean=1.0,
            episode_reward=-1.0,
            final_vehicle_count=1.0,
            final_completed=float("nan"),
            per_episode_horizon=(1.0,),
            per_episode_running_mean=(1.0,),
            episodes=1,
            seed=seed,
        )

    facts = tc.SubjectFacts(
        subject="mappo1000",
        best_source_return=-5762.0,
        rtg_scale=9991.0,
        support_range_over_the_split=(-9991.0, -6.0),
        n_rows=72000,
        training_set_return_min=-9991.0,
        checkpoints=("/nonexistent/dt_seed101.pt",),
        state_dim=25,
        context_length=20,
    )
    monkeypatch.setattr(tc, "registered_prompt_for", lambda *a, **k: (-7185.0, facts))
    monkeypatch.setattr(tc, "aligned_sumo_env_for_draw", lambda *a, **k: _Env(), raising=False)
    monkeypatch.setattr(
        "offline.aligned_env.aligned_sumo_env_for_draw", lambda *a, **k: _Env()
    )
    monkeypatch.setattr("offline.rtg_calibration.agent_with_target", lambda *a, **k: _Agent())
    monkeypatch.setattr("offline.horizon_metric.horizon_rollout", fake_rollout)

    tc.run_smoke("mappo1000", out_root=tmp_path, work_dir=tmp_path, output_root=tmp_path)

    assert calls, "the spy saw no act() call at all, so it asserted nothing"
    assert all(call == {"explore": False, "update_memory": True} for call in calls), calls


def test_the_module_contains_no_bare_dt_act_call() -> None:
    """Amendment E1's source-level half, by AST rather than by grep.

    The defect was a DEFAULT left in place, so what must be asserted is the absence of a call that
    omits the keywords -- and a text search would trip over the docstrings that quote
    ``agent.act(info)`` while explaining exactly this.  The MaxPressure probe's call is on a
    receiver named ``policy`` because ``MaxPressureAgent.act`` takes no ``explore`` keyword.
    """
    import ast

    source = Path(tc.__file__).read_text(encoding="utf-8")
    offenders: list[int] = []
    max_pressure_calls = 0
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "act" or not isinstance(node.func.value, ast.Name):
            continue
        receiver = node.func.value.id
        keywords = {kw.arg for kw in node.keywords}
        if receiver == "policy":
            max_pressure_calls += 1
            assert not keywords, "MaxPressureAgent.act takes no keywords"
        elif keywords != {"explore", "update_memory"}:
            offenders.append(node.lineno)

    assert offenders == [], f"bare or partial agent.act(...) at line(s) {offenders}"
    assert max_pressure_calls == 1, "the probe's single MaxPressure call must remain, and be one"


# ----------------------------------------------------------------------------------
# Amendment E1.2 -- the canary's CORRECTNESS half
# ----------------------------------------------------------------------------------
def _canary_facts(**overrides: Any) -> dict[str, Any]:
    facts = {
        "decisions": tc.CANARY_REFERENCE_DECISIONS,
        "local_return": tc.CANARY_REFERENCE_LOCAL_RETURN,
        "att_horizon": tc.CANARY_REFERENCE_ATT_HORIZON,
        "two_routes_agree": True,
    }
    facts.update(overrides)
    return facts


def test_check_canary_passes_on_the_recorded_control_episode() -> None:
    """The reference facts are what draw 0 produces; the check must accept them."""
    assert tc.check_canary(_canary_facts()) is None


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"local_return": -32647.0}, "local_return"),
        ({"att_horizon": math.nextafter(tc.CANARY_REFERENCE_ATT_HORIZON, math.inf)}, "att_horizon"),
        ({"two_routes_agree": False}, "two_routes_agree"),
        ({"decisions": 359}, "decisions"),
    ],
)
def test_check_canary_refuses_each_comparison_by_itself(
    overrides: dict[str, Any], expected: str
) -> None:
    """One test per comparison, so a mutation can kill exactly one.

    The ``att_horizon`` case perturbs by exactly ONE ULP via ``math.nextafter``.  A hand-written
    ``247.75089149261334`` does NOT work -- it parses to the same double as the reference, so the
    first version of this test asserted a raise that could never happen.  One ULP is the smallest
    perturbation that is a perturbation at all, which is the strictest honest form of an ``==``
    check: every measurement of draw 0 on record agrees to the last bit, so a near-miss is a
    finding about the engine rather than a tolerance to widen.
    """
    with pytest.raises(ValueError, match=expected):
        tc.check_canary(_canary_facts(**overrides))


def test_the_canary_stage_exits_non_zero_on_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through ``main``, asserting an EXIT CODE -- the driver sees a code, not an exception.

    The observed values must be printed BEFORE the refusal, or a mismatch tells the operator only
    that something was wrong (Amendment E1.2 item 3).

    ⚠️ **Strengthened by Amendment E1.3 item 1, after the coordinator's mutant MC8 -- moving
    ``check_canary`` above the print -- SURVIVED the first version.**  That version asserted
    ``"canary" in out and "-1.0" in out``, and ``main``'s refusal handler prints
    ``transfer_calibration: canary local_return -1.0 != …``, which contains both substrings.  The
    assertion was satisfied by the prose of the error message it was meant to precede.  So this
    version asserts the facts LINE and its POSITION: line 0 is the facts line, it parses, and the
    refusal comes after it.  The refusal message names one comparison and carries no JSON object,
    so it cannot stand in for the facts line.
    """
    from offline.rtg_calibration import ProbeEpisode

    wrong = ProbeEpisode(
        draw_id=0,
        local_return=-1.0,
        local_return_from_lanes=-1.0,
        att_horizon=1.0,
        horizon_vehicle_count=1.0,
        decisions=360,
    )
    monkeypatch.setattr("offline.rtg_calibration.run_probe", lambda **kwargs: [wrong])

    code = tc.main(["--work-dir", "/tmp/p72b_unused", "canary"])
    out = capsys.readouterr().out
    assert code != 0, "a mismatching canary must return a non-zero exit code"

    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[0].startswith("canary "), (
        f"the first line must be the facts line, not {lines[0]!r}; printing after the check means "
        "a mismatch tells the operator only that something was wrong"
    )
    head, separator, payload = lines[0].partition(" s ")
    assert separator == " s ", f"the canary line has no ' s ' separator: {lines[0]!r}"
    assert float(head.split()[1]) >= 0.0, "the seconds field must parse as a number"
    observed = json.loads(payload)
    assert sorted(observed) == sorted(tc.CANARY_FACT_NAMES), (
        f"the facts line must carry all four facts as JSON; got {sorted(observed)}"
    )
    assert observed["local_return"] == -1.0
    assert observed["two_routes_agree"] is True

    refusals = [index for index, line in enumerate(lines) if line.startswith("transfer_calibration:")]
    assert refusals == [1], (
        f"the refusal must follow the facts line, and it is at {refusals}; if it is at 0 the check "
        "ran before the print and the observed values never reached the pane or canary.log"
    )
    assert "two_routes_agree" not in lines[1], (
        "the refusal names one comparison; it must not be mistakable for the facts line"
    )


def test_the_canary_references_are_tied_to_committed_artifacts() -> None:
    """The constants are not free numbers: two committed artifacts carry them.

    ⚠️ **The two anchors are not the same quantity, and the docstring says so rather than imply
    identity.**  ``p7_1_metric_freeze.json``'s ``att_env_mean`` IS the canary's ``att_horizon`` --
    the env's ``average_travel_time`` at the horizon, same scenario, same policy.  The P0.2
    baseline's ``episode_reward``, however, is the **global** queue-length reward summed over the
    episode (that run used ``global_reward_weight 1.0`` and ``local_reward_fn None``), while the
    canary's ``local_return`` is the **per-intersection local** return under
    ``global_reward_weight 0.0``.  They are equal because hz1x1 has exactly ONE intersection, so
    the global and local queue-length rewards coincide -- an independent route to the same number,
    which is why it is worth tying to, and a coincidence of the topology, which is why it must be
    stated.
    """
    freeze = json.loads((REPO_DATA / "p7_1_metric_freeze.json").read_bytes())
    assert (
        freeze["cells"]["cityflow__maxpressure"]["att_env_mean"]
        == tc.CANARY_REFERENCE_ATT_HORIZON
    )

    p0 = json.loads((REPO_DATA / "p0_baselines" / "results.json").read_bytes())
    assert (
        p0["cells"][0]["policies"]["MaxPressure"]["metrics"]["episode_reward"]
        == tc.CANARY_REFERENCE_LOCAL_RETURN
    )
    assert (
        p0["aggregated"]["cf_hz1x1"]["MaxPressure"]["episode_reward"]["mean"]
        == tc.CANARY_REFERENCE_LOCAL_RETURN
    )
    assert tc.CANARY_REFERENCE_DECISIONS == tc.EXPECTED_DECISIONS == 360


# ----------------------------------------------------------------------------------
# Amendment E1.4 -- THIS run's canary reaches the artifact from a manifested canary.json
# ----------------------------------------------------------------------------------
def test_the_canary_line_round_trips_through_record_canary(tmp_path: Path) -> None:
    """``format_canary_line`` -> ``parse_canary_line`` -> ``record-canary`` -> ``canary.json``.

    The expected line is written out by hand rather than by calling the formatter twice: a
    round-trip through one function's own output would agree with itself whatever shape that was,
    and the driver passes this string across a shell boundary, so its exact bytes are the contract.
    """
    facts = _canary_facts()
    line = tc.format_canary_line(0.79, facts)
    assert line == (
        'canary 0.79 s {"att_horizon": 247.75089149261333, "decisions": 360, '
        '"local_return": -32648.0, "two_routes_agree": true}'
    ), "JSON with sorted keys, not a Python repr: the driver's line is parsed back by machine"

    seconds, parsed = tc.parse_canary_line(line)
    assert seconds == 0.79
    assert parsed == facts

    work = tmp_path / "work"
    code = tc.main(["--work-dir", str(work), "record-canary", "--line", line])
    assert code == 0
    record = json.loads((work / tc.CANARY_RECORD_NAME).read_bytes())
    assert record["line"] == line, "the record keeps the exact line the driver captured"
    assert record["seconds"] == 0.79
    assert record["facts"] == facts
    assert record["format_version"] == tc.ARTIFACT_FORMAT_VERSION
    assert record["threshold_seconds"] == tc.CANARY_MAX_SECONDS

    # Provenance, recomputed by an independent route: `git rev-parse HEAD` in this worktree, not
    # the module's own helper. Without it the record cannot say WHICH code measured the canary.
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(DRIVER.parents[2]),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert record["git_commit"] == head
    assert record["git_dirty"] in (True, False)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "canary 0.79 s",
        "0.79 s {}",
        'canary abc s {"att_horizon": 1.0}',
        "canary 0.79 s [1, 2]",
        "canary 0.79 s {not json}",
        'canary 0.79 s {"decisions": 360}',
        'canary 0.79 s {"att_horizon": 1.0, "decisions": 360, "local_return": -1.0}',
    ],
)
def test_record_canary_refuses_a_malformed_line(tmp_path: Path, line: str) -> None:
    """A line that is not exactly the canary's shape is refused, and NOTHING is created.

    The second assertion is the filesystem-mutation barrier: parse first, write second.  A
    ``canary.json`` half-written from a line nobody could parse would be worse than none, because
    ``report`` treats the file's presence as the evidence that a run happened.
    """
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="canary line"):
        tc.record_canary(line, work_dir=work)
    assert not work.exists(), (
        "a refused record must not even create the work directory; validation precedes every write"
    )


def test_report_refuses_a_work_directory_without_a_canary_record(tmp_path: Path) -> None:
    """Amendment E1.4 item 2: from this commit on, a work directory without one is not a run.

    ⚠️ The ``match`` is the EXPLANATION, not the file name, and that is the whole strength of this
    test.  The first version matched on ``canary.json``, and the implementer's mutant ME2 --
    deleting the explicit refusal -- **survived it**: ``Path.read_bytes()`` then raises the
    operating system's own ``FileNotFoundError``, whose message is
    ``[Errno 2] No such file or directory: '…/canary.json'`` and therefore contains the file name.
    The assertion was satisfied by ``ENOENT`` rather than by the refusal it was written to pin, and
    a reader of that traceback would have learned nothing about why the file is required.
    """
    work = tmp_path / "work"
    _write_band(work)
    (work / tc.CANARY_RECORD_NAME).unlink()

    out = tmp_path / "p7_2b_calibration.json"
    with pytest.raises(FileNotFoundError, match="is not a run") as excinfo:
        tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    assert tc.CANARY_RECORD_NAME in str(excinfo.value)
    assert not out.exists()


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"local_return": -32647.0}, "local_return"),
        ({"att_horizon": math.nextafter(tc.CANARY_REFERENCE_ATT_HORIZON, math.inf)}, "att_horizon"),
        ({"two_routes_agree": False}, "two_routes_agree"),
        ({"decisions": 359}, "decisions"),
    ],
)
def test_report_refuses_a_canary_record_whose_facts_fail_the_check(
    tmp_path: Path, overrides: dict[str, Any], expected: str
) -> None:
    """Amendment E1.4 item 2: ``report`` RE-RUNS ``check_canary``; a self-claim is not evidence.

    The driver cannot reach ``record-canary`` with a failing canary -- the stage exits non-zero and
    ``set -euo pipefail`` aborts before the token -- so this refusal guards the case where the file
    was produced by something other than that path, which is exactly when a check is worth having.
    """
    work = tmp_path / "work"
    _write_band(work)
    _write_canary_record(work, **overrides)

    out = tmp_path / "p7_2b_calibration.json"
    with pytest.raises(ValueError, match=expected):
        tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    assert not out.exists()


def test_report_refuses_probe_chunks_from_more_than_one_commit(tmp_path: Path) -> None:
    """Amendment E1.4 item 3: the chunks' commit is reported, so the chunks must agree on one."""
    work = tmp_path / "work"
    _write_band(work)
    path = tc.probe_chunk_path(250, work_dir=work)
    chunk = json.loads(path.read_bytes())
    chunk["git_commit"] = "9" * 40
    path.write_text(json.dumps(chunk, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out = tmp_path / "p7_2b_calibration.json"
    with pytest.raises(ValueError, match="more than one commit"):
        tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    assert not out.exists()


@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_the_artifact_canary_block_separates_this_run_from_the_probe_chunks(
    tmp_path: Path,
) -> None:
    """Amendment E1.4 items 3 and 4, on the two numbers run 3's artifact conflated.

    The chunks are rolled once and REUSED by every re-roll, so their ``canary_seconds`` is the
    canary of whichever run first rolled them -- 0.89 s for run 1 -- while the run that writes the
    artifact measures its own.  Run 3 published run 1's 0.89 beside ``checked_against``, implying a
    check that ``check_canary`` did not exist to perform when run 1 ran.  Here the two fixture
    values differ (0.5 against 0.9) so the assertion can tell them apart.
    """
    work = tmp_path / "work"
    _write_band(work)
    _write_canary_record(work, seconds=THIS_RUN_CANARY_SECONDS)

    out = tmp_path / "p7_2b_calibration.json"
    artifact = tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    block = artifact["canary"]

    # The run that WROTE the artifact, with the facts it was checked against beside it.
    assert block["seconds"] == THIS_RUN_CANARY_SECONDS
    assert block["verdict"] == "at speed"
    assert block["observed"] == _canary_facts()
    assert block["checked_against"]["local_return"] == tc.CANARY_REFERENCE_LOCAL_RETURN
    assert block["checked_against"]["att_horizon"] == tc.CANARY_REFERENCE_ATT_HORIZON
    assert block["checked_against"]["decisions"] == tc.CANARY_REFERENCE_DECISIONS
    assert block["checked_against"]["two_routes_agree"] is True
    assert tc.CANARY_RECORD_NAME in block["source"]
    assert block["git_commit"] == "1" * 40, "the commit that MEASURED the canary, from the record"

    # ... and the chunks', named as the chunks' and claiming nothing about history.
    chunks_block = block["probe_chunks_canary"]
    assert chunks_block["seconds"] == CHUNK_CANARY_SECONDS
    assert chunks_block["git_commit"] == "0" * 40
    assert chunks_block["correctness_half"] == "not recorded in these chunks"
    assert "checked_against" not in chunks_block, (
        "the chunks carry a duration and no facts; a checked_against beside them would assert a "
        "check that nothing performed"
    )

    # Item 4: exactly once in the whole artifact, beside the facts it was checked against.
    assert json.dumps(artifact, sort_keys=True).count('"checked_against"') == 1


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

    # Amendment D2: the run's rate basis is IN the artifact, not only in the chunks.
    # ⚠️ Amendment E1.4 CHANGED WHICH NUMBER THIS IS. It used to read the chunks' 0.9, which is the
    # canary of whichever run first ROLLED them -- reused unchanged by every re-roll afterwards.
    # The block's own seconds is now the reporting run's, from canary.json (0.5 in this fixture),
    # and the chunks' is reported separately under its own name.
    assert payload["canary"]["seconds"] == THIS_RUN_CANARY_SECONDS
    assert payload["canary"]["probe_chunks_canary"]["seconds"] == CHUNK_CANARY_SECONDS
    assert payload["canary"]["threshold_seconds"] == tc.CANARY_MAX_SECONDS == 2.0
    assert payload["canary"]["verdict"] == "at speed"
    assert "PROJECT_PLAN" in payload["canary"]["recipe"]


@pytest.mark.skipif(not _checkpoints_available(), reason="P4 checkpoints are not in this tree")
def test_report_refuses_chunks_from_two_runs_with_different_canaries(tmp_path: Path) -> None:
    """Amendment D2: one campaign, one canary.

    Two distinct values mean two runs' chunks were mixed, and every ``seconds`` in the probe table
    would then be incomparable -- the class of defect the canary exists to prevent, arriving by the
    back door.
    """
    work = tmp_path / "work"
    _write_band(work)
    path = tc.probe_chunk_path(250, work_dir=work)
    chunk = json.loads(path.read_bytes())
    chunk["canary_seconds"] = 6.88  # the throttled machine's value, from BRIEF_35 D3.1
    path.write_text(json.dumps(chunk, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out = tmp_path / "p7_2b_calibration.json"
    with pytest.raises(ValueError, match="one campaign has one canary"):
        tc.report(work_dir=work, out_path=out, output_root=OUTPUT_ROOT)
    assert not out.exists()


def test_the_driver_and_the_module_agree_on_the_canary_threshold() -> None:
    """Two literals, one number. The driver cannot import Python constants, so a test ties them."""
    text = DRIVER.read_text(encoding="utf-8")
    assert f"CANARY_MAX_SECONDS={tc.CANARY_MAX_SECONDS}" in text


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


def _driver_code_text() -> str:
    """The driver with every whole-line comment removed.

    ⚠️ **Amendment E1.3 item 2, after the coordinator's mutant MD2 SURVIVED.**  The test asserted
    ``"tee /dev/stderr" in text``; deleting ``tee`` from the CODE line left the assertion satisfied
    by the COMMENT that explains what ``tee`` is there for.  A text assertion over a file that
    documents itself will be satisfied by its own documentation -- the same class of defect as
    Amendment E1's, one commit later and in the other language.  Assertions about what the driver
    DOES are made over this text; assertions about what it SAYS may use the whole file.
    """
    text = DRIVER.read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


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
    # Amendment E1.2 item 4: the canary's OBSERVED values must reach a manifested file, and the
    # write must sit AFTER the token so Amendment B1's "a refused start creates nothing" holds.
    # EVERY write, not just the first: one line moved above the token would create the log there.
    code = _driver_code_text()
    code_token_at = code.index('rm -f "$TOKEN"')
    log_writes = [
        index
        for index in range(len(code))
        if code.startswith('>> "$LOGS/canary.log"', index)
    ]
    assert log_writes, "the canary line must be appended to canary.log, never truncated into it"
    assert min(log_writes) > code_token_at, (
        "canary.log is written before the token is consumed, so a refused start would create it"
    )

    # ... and the line must be VISIBLE even when the stage exits non-zero, or a mismatching canary
    # aborts with the observed values swallowed by the command substitution. Asserted on the CODE
    # line with its closing parenthesis, over comment-free text: the previous form was satisfied by
    # the comment that documents it (Amendment E1.3 item 2, mutant MD2).
    assert "canary | tee -a /dev/stderr)" in code, (
        "the canary stage's output must be tee'd to stderr, APPENDING: plain `tee /dev/stderr` "
        "re-opens the target with O_TRUNC and destroys a combined log written with `>> log 2>&1`"
    )
    assert code.index("canary | tee -a /dev/stderr)") < code_token_at

    # Amendment E1.4 item 1: this run's canary is parked in a manifested canary.json -- and, like
    # canary.log, only AFTER the token, so a refused start still creates nothing (Amendment B1).
    assert code_token_at < code.index("record-canary"), (
        "record-canary runs before the token is consumed, so a refused start would create "
        "canary.json in the work directory"
    )

    # ... and the handler must not CREATE the work dir just to record a failure in it.
    assert text.count('if [ -d "$WORK" ]; then') == 2, (
        "both FAILED writers guard on $WORK existing, so a signal before the token still leaves "
        "the tree exactly as it found it"
    )
    # Nothing under scenarios/draws is written.
    assert "scenarios/draws" in text
    assert ">" not in text.split("scenarios/draws")[1].split("\n")[0]
