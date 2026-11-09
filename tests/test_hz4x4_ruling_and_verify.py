"""Tests for ``BRIEF_48`` Amendment A, A.2's two required findings -- written before the code they test.

* **Finding 1 -- the fixed-time ruling file.** No code path could collect ``fixedtime`` after a ruling, because the k was
  read only from a sweep artifact whose status is ``CHOSEN`` and that artifact is never edited.  A.4 approves a SEPARATE
  committed ruling file that names the sweep artifact by sha256, the k, and the amendment carrying the author's ruling;
  the collector refuses a ruling whose sweep digest differs from the committed artifact's.
* **Finding 2 -- ``verify_run``'s missing fields.** ``flow_randomizer_params``, ``flow_source_path`` /
  ``flow_source_sha256``, ``metrics`` and ``epsilon`` are compared with their expected values before ``collect verify``
  runs on the six runs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import offline.hz4x4_ladder as hz

REPO = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO / "scenarios" / "hangzhou_4x4_gudang_18041610_1h"
SOURCE_FLOW_NAME = "hangzhou_4x4_gudang_18041610_1h.json"

PROTOCOL: dict[str, Any] = {
    "backend": "cityflow", "base_seed": 1000, "max_steps": 360, "delta_time": 10, "control_mode": "acyclic",
    "global_reward_fn": "queue_length", "local_reward_fn": "queue_length", "global_reward_weight": 0.0,
    "state_features": ["lane_vehicle_count", "lane_waiting", "phase_onehot"],
    "flow_randomizer_params": {"jitter_sigma_s": 30.0, "thin_p": 0.1, "volume_scale": 1.0},
}


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except ImportError:
        return False
    return True


needs_cityflow = pytest.mark.skipif(
    not _cityflow_available(), reason="the native cityflow engine is not importable here"
)


# ======================================================================
# Finding 1: the fixed-time ruling file
# ======================================================================


def _sweep(tmp_path: Path, status: str = "DEFINITION-DEPENDENT", chosen: int | None = None) -> Path:
    path = tmp_path / "p9_3_fixed_time_sweep.json"
    payload = {"format_version": hz.ARTIFACT_FORMAT_SWEEP,
               "decision": {"status": status, "chosen_k": chosen, "k_att_engine": 3, "k_att_ours": 2}}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _brief(tmp_path: Path, text: str = "# AMENDMENT B -- the author's ruling: the fixed-time k is 3\n") -> Path:
    path = tmp_path / "BRIEF_48.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_ruling_names_the_sweep_by_digest_the_k_and_the_amendment(tmp_path: Path) -> None:
    sweep, brief = _sweep(tmp_path), _brief(tmp_path)
    record = hz.fixed_time_ruling_record(sweep_path=sweep, k=3, amendment="AMENDMENT B", brief_path=brief)
    assert record["format_version"] == hz.ARTIFACT_FORMAT_RULING
    assert record["sweep_sha256"] == hashlib.sha256(sweep.read_bytes()).hexdigest()
    assert record["chosen_k"] == 3
    assert record["amendment"] == "AMENDMENT B"
    assert record["brief_sha256"] == hashlib.sha256(brief.read_bytes()).hexdigest()
    assert record["sweep_status"] == "DEFINITION-DEPENDENT"
    ruling = tmp_path / "ruling.json"
    ruling.write_text(json.dumps(record), encoding="utf-8")
    assert hz.fixed_time_k_from_ruling(ruling, sweep) == 3
    assert hz.resolve_fixed_time_k(sweep, ruling) == 3


def test_a_ruling_is_refused_unless_its_brief_carries_the_amendment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AMENDMENT C"):
        hz.fixed_time_ruling_record(sweep_path=_sweep(tmp_path), k=3, amendment="AMENDMENT C",
                                    brief_path=_brief(tmp_path))
    with pytest.raises(ValueError, match="amendment"):
        hz.fixed_time_ruling_record(sweep_path=_sweep(tmp_path), k=3, amendment="", brief_path=_brief(tmp_path))
    with pytest.raises(ValueError, match="k = 5"):
        hz.fixed_time_ruling_record(sweep_path=_sweep(tmp_path), k=5, amendment="AMENDMENT B",
                                    brief_path=_brief(tmp_path))
    chosen = _sweep(tmp_path, status="CHOSEN", chosen=3)
    with pytest.raises(ValueError, match="CHOSEN"):
        hz.fixed_time_ruling_record(sweep_path=chosen, k=3, amendment="AMENDMENT B", brief_path=_brief(tmp_path))


def test_the_collector_refuses_a_ruling_whose_sweep_digest_differs(tmp_path: Path) -> None:
    sweep, brief = _sweep(tmp_path), _brief(tmp_path)
    ruling = tmp_path / "ruling.json"
    ruling.write_text(json.dumps(hz.fixed_time_ruling_record(sweep_path=sweep, k=3, amendment="AMENDMENT B",
                                                             brief_path=brief)), encoding="utf-8")
    sweep.write_text(sweep.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        hz.fixed_time_k_from_ruling(ruling, sweep)
    with pytest.raises(ValueError, match="sha256"):
        hz.resolve_fixed_time_k(sweep, ruling)


def test_a_ruling_file_of_another_format_or_k_is_refused(tmp_path: Path) -> None:
    sweep, brief = _sweep(tmp_path), _brief(tmp_path)
    record = hz.fixed_time_ruling_record(sweep_path=sweep, k=3, amendment="AMENDMENT B", brief_path=brief)
    ruling = tmp_path / "ruling.json"
    ruling.write_text(json.dumps(dict(record, format_version="other/1.0")), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        hz.fixed_time_k_from_ruling(ruling, sweep)
    ruling.write_text(json.dumps(dict(record, chosen_k=5)), encoding="utf-8")
    with pytest.raises(ValueError, match="k = 5"):
        hz.fixed_time_k_from_ruling(ruling, sweep)
    ruling.write_text(json.dumps(dict(record, amendment="")), encoding="utf-8")
    with pytest.raises(ValueError, match="amendment"):
        hz.fixed_time_k_from_ruling(ruling, sweep)


def test_resolve_fixed_time_k_uses_a_chosen_sweep_and_refuses_an_unruled_one(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    chosen = _sweep(tmp_path / "a", status="CHOSEN", chosen=4)
    assert hz.resolve_fixed_time_k(chosen, None) == 4
    unruled = _sweep(tmp_path)
    with pytest.raises(ValueError, match="DEFINITION-DEPENDENT"):
        hz.resolve_fixed_time_k(unruled, None)
    ruling = tmp_path / "ruling.json"
    ruling.write_text(json.dumps({"format_version": hz.ARTIFACT_FORMAT_RULING}), encoding="utf-8")
    with pytest.raises(ValueError, match="chose k itself"):
        hz.resolve_fixed_time_k(chosen, ruling)


def test_the_committed_hz4x4_sweep_needs_a_ruling_before_fixedtime_is_collected() -> None:
    sweep = REPO / "docs" / "data" / "p9_3_fixed_time_sweep.json"
    with pytest.raises(ValueError, match="DEFINITION-DEPENDENT"):
        hz.resolve_fixed_time_k(sweep, None)


def test_the_collect_cli_takes_a_ruling_file() -> None:
    namespace = hz.build_parser().parse_args(["collect", "plan", "--sweep", "s.json", "--ruling", "r.json"])
    assert namespace.ruling == "r.json"
    namespace = hz.build_parser().parse_args(["record-ruling", "--sweep", "s.json", "--k", "3",
                                              "--amendment", "AMENDMENT B", "--out", "r.json"])
    assert (namespace.command, namespace.k, namespace.amendment) == ("record-ruling", 3, "AMENDMENT B")


# ======================================================================
# Finding 2: verify_run compares every run-metadata field it can derive
# ======================================================================


@pytest.fixture
def collected(tmp_path: Path) -> tuple[Any, Path]:
    """One real hz4x4 MaxPressure episode on draw 1 of a scenario copy, collected by ``offline.collect``."""
    from offline.collect import main as collect_main

    scenario = tmp_path / "scenario"
    scenario.mkdir()
    for name in ("roadnet_4X4.json", SOURCE_FLOW_NAME):
        shutil.copyfile(SCENARIO_DIR / name, scenario / name)
    cfg = json.loads((REPO / "configs" / "sim" / "cityflow4x4.json").read_text(encoding="utf-8"))
    cfg["dir"] = str(scenario) + "/"
    config = tmp_path / "cityflow4x4.json"
    config.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    draws_root = tmp_path / "draws"
    hz.materialise_hz4x4_draws([1], source_config=config, out_root=draws_root)
    out_dir = tmp_path / "cf_hz4x4__maxpressure"
    argv = hz.protocol_argv(PROTOCOL, env_config=config, out_dir=out_dir, policy="maxpressure", episodes=1,
                            draw_ids=[1])
    assert collect_main(list(argv)) == 0
    run = hz.CollectionRun(name=out_dir.name, tier="maxpressure", seed=None, out_dir=out_dir, argv=argv,
                           expected_episodes=1, draw_ids=(1,))
    return run, draws_root


@needs_cityflow
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flow_randomizer_params", {"base_seed": 1000, "jitter_sigma_s": 30.0, "thin_p": 0.2, "volume_scale": 1.0}),
        ("flow_source_path", "/elsewhere/flow.json"),
        ("flow_source_sha256", "0" * 64),
        ("metrics", ["number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"]),
        ("epsilon", 0.1),
    ],
)
def test_verify_run_refuses_each_drifted_field(collected: tuple[Any, Path], field: str, value: Any) -> None:
    run, draws_root = collected
    record = hz.verify_run(run, draws_root=draws_root)
    assert record["episodes"] == 1
    manifest_path = run.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert field in manifest["run_metadata"]
    manifest["run_metadata"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        hz.verify_run(run, draws_root=draws_root)


# ======================================================================
# BRIEF_48 Amendment C: the author's ruling, recorded (C.2.1)
# ======================================================================

COMMITTED_SWEEP = REPO / "docs" / "data" / "p9_3_fixed_time_sweep.json"
COMMITTED_RULING = REPO / "docs" / "data" / "p9_3_fixed_time_ruling.json"
BRIEF_48 = REPO / "docs" / "briefs" / "BRIEF_48_p9.3_a27c_hz4x4.md"


def test_the_committed_ruling_gives_k_2_for_the_committed_sweep_only(tmp_path: Path) -> None:
    ruling = json.loads(COMMITTED_RULING.read_bytes())
    assert ruling["chosen_k"] == 2
    assert ruling["amendment"] == "AMENDMENT C — 2026-10-16"
    assert ruling["amendment"] in BRIEF_48.read_text(encoding="utf-8")
    assert ruling["sweep_sha256"] == hashlib.sha256(COMMITTED_SWEEP.read_bytes()).hexdigest()
    assert (ruling["sweep_status"], ruling["sweep_k_att_engine"], ruling["sweep_k_att_ours"]) == (
        "DEFINITION-DEPENDENT", 3, 2)
    assert hz.fixed_time_k_from_ruling(COMMITTED_RULING, COMMITTED_SWEEP) == 2
    assert hz.resolve_fixed_time_k(COMMITTED_SWEEP, COMMITTED_RULING) == 2
    altered = tmp_path / "p9_3_fixed_time_sweep.json"
    altered.write_bytes(COMMITTED_SWEEP.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="sha256"):
        hz.fixed_time_k_from_ruling(COMMITTED_RULING, altered)


# ======================================================================
# BRIEF_48 Amendment F / E.3.5: random_fresh's gate cell and verify_run's registered distinct-schedule check
# ======================================================================


class _StubIx:
    def __init__(self, index: int) -> None:
        self.id = f"ix{index}"
        self.num_phases = 8


class _StubEnv:
    def __init__(self, config_path: Path) -> None:
        self._config_path = str(Path(config_path).resolve())
        self.intersections = [_StubIx(i) for i in range(3)]
        self.action_space = None


def _actions(choose: Any, env: Any, steps: int = 12) -> list[list[int]]:
    info = {"intersections": {ix.id: {"avail_actions": list(range(8))} for ix in env.intersections}}
    return [[int(a) for a in choose(env, info)] for _ in range(steps)]


def _random_fresh_manifest(corpus_root: Path) -> None:
    directory = corpus_root / "cf_hz4x4__random_fresh"
    directory.mkdir(parents=True)
    meta = dict(PROTOCOL, behavior_policy="random", episodes=1, random_rng_scheme="per-episode")
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": meta, "episodes": []}), encoding="utf-8")


def test_the_random_fresh_gate_cell_is_a27ds_fresh_reference(tmp_path: Path, tmp_scenario_config: Path) -> None:
    from types import SimpleNamespace

    import numpy as np

    from offline.collect import _make_random

    draws_root = tmp_path / "draws"
    hz.materialise_hz4x4_draws([1000, 1001], source_config=tmp_scenario_config, out_root=draws_root)
    corpus_root = tmp_path / "corpus"
    _random_fresh_manifest(corpus_root)
    cells = [c for c in hz.gate_cells_for(["random_fresh"]) if c.draw_id in (1000, 1001)]
    assert [c.seed for c in cells] == [101, 101]
    for cell in cells:
        _settings, config, _created, factory = hz._gate_cell_inputs(
            cell, corpus_root=corpus_root, draws_root=draws_root, device="cpu")
        env = _StubEnv(config)
        reference = _make_random(env, SimpleNamespace(), np.random.default_rng([101, int(cell.draw_id)]))
        assert _actions(factory(env), env) == _actions(lambda _e, info: reference(info), env)
    from offline.materialise_draws import draw_config_path

    first = _StubEnv(draw_config_path(hz.SCENARIO_KEY, 1000, out_root=draws_root))
    other = _make_random(first, SimpleNamespace(), np.random.default_rng([101, 1001]))
    _s, _c, _n, factory = hz._gate_cell_inputs(cells[0], corpus_root=corpus_root, draws_root=draws_root, device="cpu")
    assert _actions(factory(first), first) != _actions(lambda _e, info: other(info), first)


@pytest.fixture
def tmp_scenario_config(tmp_path: Path) -> Path:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    for name in ("roadnet_4X4.json", SOURCE_FLOW_NAME):
        shutil.copyfile(SCENARIO_DIR / name, scenario / name)
    cfg = json.loads((REPO / "configs" / "sim" / "cityflow4x4.json").read_text(encoding="utf-8"))
    cfg["dir"] = str(scenario) + "/"
    config = tmp_path / "cityflow4x4.json"
    config.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return config


def _random_run(config: Path, out_dir: Path, scheme: str) -> Any:
    argv = hz.protocol_argv(PROTOCOL, env_config=config, out_dir=out_dir, policy="random", episodes=1, draw_ids=[1, 2],
                            extra=("--random-rng", scheme))
    return hz.CollectionRun(name=out_dir.name, tier="random_fresh", seed=None, out_dir=out_dir, argv=argv,
                            expected_episodes=2, draw_ids=(1, 2))


def _collect_random(tmp_path: Path, config: Path, draws_root: Path, scheme: str) -> Any:
    from offline.collect import main as collect_main

    run = _random_run(config, tmp_path / f"cf_hz4x4__random_fresh_{scheme}", scheme)
    assert collect_main(list(run.argv)) == 0
    return run


def _relabel_manifest(run: Any, scheme: str) -> None:
    manifest_path = run.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["run_metadata"]["random_rng_scheme"] = scheme
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@needs_cityflow
def test_verify_run_applies_a27ds_registered_distinct_schedule_check(tmp_path: Path, tmp_scenario_config: Path) -> None:
    draws_root = tmp_path / "draws"
    hz.materialise_hz4x4_draws([1, 2], source_config=tmp_scenario_config, out_root=draws_root)
    fresh = _collect_random(tmp_path, tmp_scenario_config, draws_root, "per-episode")
    record = hz.verify_run(fresh, draws_root=draws_root)
    assert record["random_rng_scheme"] == "per-episode"
    assert record["distinct_schedules"]["episodes"] == 2
    assert set(record["distinct_schedules"]["per_intersection_distinct"].values()) == {2}
    assert set(record["distinct_sequences_per_intersection"].values()) == {2}
    stale = _collect_random(tmp_path, tmp_scenario_config, draws_root, "per-draw")
    # The scheme check's own wording: the run directory's name carries the scheme too, so a bare "per-draw" would match
    # whichever check raised (mutant V4, the scheme check removed, survived that pattern).
    with pytest.raises(ValueError, match="collected with random_rng_scheme 'per-draw'; A27 \\(d\\) requires"):
        hz.verify_run(stale, draws_root=draws_root)
    # The per-draw arrays under a manifest AND a command that both say per-episode: only the distinct check refuses them.
    _relabel_manifest(stale, "per-episode")
    relabelled = _random_run(tmp_scenario_config, stale.out_dir, "per-episode")
    with pytest.raises(ValueError, match="do not hold 2 distinct"):
        hz.verify_run(relabelled, draws_root=draws_root)
    # The two routes read different lists -- P9.1's the manifest's episodes, this module's every ep*.npz on disk -- so an
    # episode file the manifest does not list (a partial run's leftover) makes them disagree, and verify refuses it.
    shutil.copyfile(sorted(stale.out_dir.glob("ep*.npz"))[0], fresh.out_dir / "ep000099_seed1000_draw1.npz")
    with pytest.raises(ValueError, match="two distinct-sequence counts disagree"):
        hz.verify_run(fresh, draws_root=draws_root)


@needs_cityflow
def test_verify_run_compares_the_generator_scheme_with_the_command(tmp_path: Path, tmp_scenario_config: Path) -> None:
    # M1's --random-rng is a setting the command determines, so the manifest's random_rng_scheme is compared with it
    # like every other setting (A.2 finding 2's rule): a manifest naming another scheme than its command is refused by
    # the comparison itself, whatever the per-tier checks would say.
    draws_root = tmp_path / "draws"
    hz.materialise_hz4x4_draws([1, 2], source_config=tmp_scenario_config, out_root=draws_root)
    stale = _collect_random(tmp_path, tmp_scenario_config, draws_root, "per-draw")
    _relabel_manifest(stale, "per-episode")
    with pytest.raises(ValueError, match="differs from the derived command.*random_rng_scheme"):
        hz.verify_run(stale, draws_root=draws_root)
