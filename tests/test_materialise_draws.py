"""Tests for ``offline.materialise_draws`` (P2.2-draws).

The load-bearing test is :func:`test_two_draws_diverge_under_a_deterministic_policy`:
a draw that renders but does not change simulated behaviour is the failure this whole
subsystem exists to prevent, and only a real engine can refute it.  It runs on
**hangzhou 1x1**, never cologne3: CityFlow determinism is scenario-dependent (measured
over 5 seeds, 2026-08-06: hz1x1 sigma=0.0000, grid4x4 sigma=0.0000, cologne3
sigma=3.3239), so on cologne3 two runs of the *same* draw already differ and a
divergence assertion would pass for the wrong reason.

Every test materialises into ``tmp_path``; none of them touches the real
``scenarios/draws/`` tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offline.flow_randomizer import (
    DEFAULT_BASE_SEED,
    DEFAULT_JITTER_SIGMA_S,
    DEFAULT_THIN_P,
    DEFAULT_VOLUME_SCALE,
    FlowRandomizer,
)
from offline.materialise_draws import (
    FORMAT_VERSION,
    HELD_OUT_POOL,
    TRAINING_POOL,
    draw_config_path,
    draw_dir,
    load_provenance,
    main,
    materialise,
    scenario_key_for_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HZ1X1_CONFIG = REPO_ROOT / "configs/sim/cityflow1x1.json"
GRID4X4_CONFIG = REPO_ROOT / "configs/sim/cityflow_grid4x4.json"
COLOGNE3_CONFIG = REPO_ROOT / "configs/sim/cityflow_cologne3.json"

HZ1X1_KEY = "cityflow1x1"
GRID4X4_KEY = "cityflow_grid4x4"

#: Decision steps in the load-bearing rollout.  At delta_time=10 this is 600 s of the
#: 3600 s hangzhou hour -- long enough for a ~10 % demand change to reach the signal,
#: short enough to keep the suite quick.
ROLLOUT_STEPS = 60


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except Exception:
        return False
    return True


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_snapshot(root: str | Path) -> dict[str, str]:
    """Map every file under *root* to its sha256, relative-path keyed.

    Used to prove that a failed materialisation changed nothing: comparing whole
    snapshots catches a deleted file, which comparing only the files we expect would
    not.
    """
    base = Path(root)
    if not base.exists():
        return {}
    return {
        str(path.relative_to(base)): _sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _cityflow_resolved_flow(config_path: str | Path) -> Path:
    """Resolve ``dir + flowFile`` exactly as CityFlow's C++ engine does.

    ``Engine`` concatenates the two strings (``CityFlow/src/engine/engine.cpp:65``);
    reproducing that here rather than calling the code under test is the independent
    route the alignment claim needs.
    """
    cfg = json.loads(Path(config_path).read_bytes())
    return Path(os.path.normpath(cfg["dir"] + cfg["flowFile"]))


def _cityflow_resolved_roadnet(config_path: str | Path) -> Path:
    cfg = json.loads(Path(config_path).read_bytes())
    return Path(os.path.normpath(cfg["dir"] + cfg["roadnetFile"]))


def _source_flow_path(config_path: str | Path) -> Path:
    """The flow file a source sim config points at, resolved against the repo root."""
    cfg = json.loads(Path(config_path).read_bytes())
    cfg_dir = cfg["dir"]
    if not os.path.isabs(cfg_dir):
        cfg_dir = str(REPO_ROOT / cfg_dir)
    return Path(os.path.normpath(cfg_dir)) / cfg["flowFile"]


def _rollout_digest(config_path: str | Path, steps: int) -> tuple[str, int]:
    """Run MaxPressure on a drawn config; return ``(episode digest, steps taken)``.

    The digest follows the C6 ``episode_sha256`` recipe
    (``offline/trajectory_logger.py:793-803``): every intersection's action array as
    little-endian int64, then the global rewards as little-endian float32.
    """
    from agent.utils.utils import Utils
    from algorithms.max_pressure import MaxPressureAgent
    from experiments.config import SETTING_DEFAULTS, EnvSpec
    from experiments.envs import make_env

    settings: dict[str, Any] = dict(SETTING_DEFAULTS)
    settings.update({"max_steps": steps, "delta_time": 10, "thread_num": 1})
    spec = EnvSpec(
        id="materialise-draws-test",
        backend="cityflow",
        paths={"config": str(Path(config_path).resolve())},
        settings=settings,
    )

    env = make_env(spec)
    try:
        agent = MaxPressureAgent(env)
        info = env.reset(seed=0)
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        for _ in range(steps):
            action = agent.act(info)
            actions.append(np.asarray(action, dtype=np.int64))
            reward, terminated, truncated, info = env.step(action)
            rewards.append(float(Utils.scalar_reward(reward)))
            if terminated or truncated:
                break
    finally:
        env.close()

    stacked = np.stack(actions) if actions else np.zeros((0, 0), dtype=np.int64)
    digest = hashlib.sha256()
    for column in range(stacked.shape[1]):
        digest.update(stacked[:, column].astype("<i8", copy=False).tobytes())
    digest.update(np.asarray(rewards, dtype=np.float32).astype("<f4").tobytes())
    return digest.hexdigest(), len(rewards)


# --------------------------------------------------------------------------
# T1 -- load-bearing: the drawn scenario runs, and different draws differ
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_two_draws_diverge_under_a_deterministic_policy(tmp_path: Path) -> None:
    """Draws 1 and 2 both run, replay identically, and differ from each other.

    The repeat of draw 1 is what makes the inequality attributable to the demand: on
    hz1x1 the engine is deterministic (sigma=0.0000 over 5 seeds), so equal-vs-itself
    and different-vs-draw-2 together rule out "the engine is just noisy".
    """
    materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)

    config_1 = draw_config_path(HZ1X1_KEY, 1, out_root=tmp_path)
    config_2 = draw_config_path(HZ1X1_KEY, 2, out_root=tmp_path)

    digest_1, steps_1 = _rollout_digest(config_1, ROLLOUT_STEPS)
    digest_1_again, _ = _rollout_digest(config_1, ROLLOUT_STEPS)
    digest_2, steps_2 = _rollout_digest(config_2, ROLLOUT_STEPS)

    assert steps_1 == ROLLOUT_STEPS
    assert steps_2 == ROLLOUT_STEPS
    assert digest_1 == digest_1_again
    assert digest_1 != digest_2


# --------------------------------------------------------------------------
# T2 -- draw 0 is the nominal control and must round-trip byte-identically
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_config", [HZ1X1_CONFIG, GRID4X4_CONFIG, COLOGNE3_CONFIG]
)
def test_draw_zero_reproduces_the_source_flow_byte_for_byte(
    source_config: Path, tmp_path: Path
) -> None:
    """P2.0's invariant, re-checked on the materialised artifact itself."""
    [record] = materialise(source_config, [0], out_root=tmp_path)

    assert record.flow_path.read_bytes() == _source_flow_path(source_config).read_bytes()

    provenance = load_provenance(
        scenario_key_for_config(source_config), 0, out_root=tmp_path
    )
    assert provenance["pool"] == "nominal"
    assert provenance["is_nominal_control"] is True


# --------------------------------------------------------------------------
# T3 -- training and held-out ids stay disjoint, judged on what was written
# --------------------------------------------------------------------------


def test_written_draw_ids_respect_the_registered_pools(tmp_path: Path) -> None:
    """Assert on the ids found on disk, never on the ids the caller asked for."""
    requested = [0, 1, 5, 1000, 1004]
    materialise(HZ1X1_CONFIG, requested, out_root=tmp_path)

    found: dict[int, str] = {}
    for directory in sorted((tmp_path / HZ1X1_KEY).iterdir()):
        id_from_path = int(directory.name.removeprefix("draw_"))
        provenance = json.loads((directory / "provenance.json").read_bytes())
        # The path is the index a caller greps by; the record is what a reader trusts.
        # If they ever disagree, a config has been silently confused with another draw.
        assert provenance["draw_id"] == id_from_path
        found[id_from_path] = provenance["pool"]

    assert sorted(found) == sorted(requested)

    training = {i for i, pool in found.items() if pool == "training"}
    held_out = {i for i, pool in found.items() if pool == "held_out"}
    assert training == {1, 5}
    assert held_out == {1000, 1004}
    assert training.isdisjoint(held_out)
    assert all(i in TRAINING_POOL and i not in HELD_OUT_POOL for i in training)
    assert all(i in HELD_OUT_POOL and i not in TRAINING_POOL for i in held_out)


# --------------------------------------------------------------------------
# T4 -- filesystem-mutation barrier, proven by injected failures
# --------------------------------------------------------------------------


def test_failure_while_building_leaves_previous_draws_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draw that explodes mid-build must not cost the draws already on disk."""
    materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)
    before = _tree_snapshot(tmp_path)
    assert before  # the barrier is only meaningful with prior data to protect

    real_draw = FlowRandomizer.draw

    def exploding_draw(self: FlowRandomizer, draw_id: int):  # type: ignore[no-untyped-def]
        if int(draw_id) == 4:
            raise RuntimeError("injected failure while drawing 4")
        return real_draw(self, draw_id)

    monkeypatch.setattr(FlowRandomizer, "draw", exploding_draw)

    with pytest.raises(RuntimeError, match="injected failure while drawing 4"):
        materialise(HZ1X1_CONFIG, [3, 4], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == before
    assert not (tmp_path / HZ1X1_KEY / "draw_0003").exists()
    assert not (tmp_path / HZ1X1_KEY / "draw_0004").exists()
    assert [p.name for p in tmp_path.rglob(".staging*")] == []


def test_failure_while_committing_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure between two renames must not leave half a materialisation behind."""
    import offline.materialise_draws as module

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    before = _tree_snapshot(tmp_path)

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected failure while committing the second draw")
        real_replace(src, dst)

    monkeypatch.setattr(module.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="injected failure while committing"):
        materialise(HZ1X1_CONFIG, [2, 3], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == before
    assert [p.name for p in tmp_path.rglob(".staging*")] == []


# --------------------------------------------------------------------------
# T5 -- provenance completeness, every field recomputed by an independent route
# --------------------------------------------------------------------------


def test_provenance_records_source_digest_draw_id_and_parameters(
    tmp_path: Path,
) -> None:
    records = materialise(HZ1X1_CONFIG, [1, 1000], out_root=tmp_path)
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_source_sha = _sha256_file(_source_flow_path(HZ1X1_CONFIG))

    for record in records:
        provenance = load_provenance(HZ1X1_KEY, record.draw_id, out_root=tmp_path)

        assert provenance["format_version"] == FORMAT_VERSION
        assert provenance["draw_id"] == record.draw_id
        assert provenance["source_flow_sha256"] == expected_source_sha
        assert provenance["draw"]["source_sha256"] == expected_source_sha
        assert provenance["git_commit"] == expected_commit
        assert provenance["randomizer"] == {
            "base_seed": DEFAULT_BASE_SEED,
            "jitter_sigma_s": DEFAULT_JITTER_SIGMA_S,
            "thin_p": DEFAULT_THIN_P,
            "volume_scale": DEFAULT_VOLUME_SCALE,
        }
        assert provenance["numpy_version"] == np.__version__

        # Independent recount: the vehicle list on disk, not the number the writer
        # reported.
        entries = json.loads(record.flow_path.read_bytes())
        assert provenance["draw"]["n_vehicles"] == len(entries)
        assert record.n_vehicles == len(entries)

        for filename, digest in provenance["files"].items():
            assert _sha256_file(record.directory / filename) == digest


# --------------------------------------------------------------------------
# T6 -- the config points at its own draw, and leaves the topology alone
# --------------------------------------------------------------------------


def test_written_config_resolves_to_its_own_draw_and_keeps_the_roadnet(
    tmp_path: Path,
) -> None:
    [record] = materialise(HZ1X1_CONFIG, [3], out_root=tmp_path)
    config_path = draw_config_path(HZ1X1_KEY, 3, out_root=tmp_path)

    assert config_path == record.config_path
    assert _cityflow_resolved_flow(config_path) == record.flow_path.resolve()

    source_cfg = json.loads(HZ1X1_CONFIG.read_bytes())
    drawn_cfg = json.loads(config_path.read_bytes())
    assert sorted(drawn_cfg) == sorted(source_cfg)
    for key in source_cfg:
        if key not in {"dir", "flowFile"}:
            assert drawn_cfg[key] == source_cfg[key]

    # A draw perturbs demand, never topology.
    roadnet = _cityflow_resolved_roadnet(config_path)
    source_roadnet = _source_flow_path(HZ1X1_CONFIG).parent / source_cfg["roadnetFile"]
    assert _sha256_file(roadnet) == _sha256_file(source_roadnet)


# --------------------------------------------------------------------------
# T7 -- re-running is a no-op; a conflict is refused; --force replaces cleanly
# --------------------------------------------------------------------------


def test_rematerialising_the_same_draws_is_a_byte_level_no_op(tmp_path: Path) -> None:
    first = materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)
    snapshot = _tree_snapshot(tmp_path)
    assert [record.action for record in first] == ["written", "written"]

    second = materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)

    assert [record.action for record in second] == ["kept", "kept"]
    assert _tree_snapshot(tmp_path) == snapshot


def test_a_new_git_commit_does_not_make_an_existing_draw_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity is the drawn demand, not the metadata of the run that wrote it."""
    import offline.materialise_draws as module

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    snapshot = _tree_snapshot(tmp_path)

    monkeypatch.setattr(module, "_git_commit", lambda: ("0" * 40, True))
    [record] = materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)

    assert record.action == "kept"
    assert _tree_snapshot(tmp_path) == snapshot


def test_a_conflicting_draw_is_refused_and_only_force_replaces_it(
    tmp_path: Path,
) -> None:
    materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)
    good_snapshot = _tree_snapshot(tmp_path)
    good_flow = (draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "flow.json").read_bytes()

    tampered = draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "flow.json"
    tampered.write_bytes(good_flow + b"\n")
    tampered_snapshot = _tree_snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="differs from the draw that would"):
        materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)

    # Refusing must not repair, delete or partially rewrite anything.
    assert _tree_snapshot(tmp_path) == tampered_snapshot

    records = materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path, force=True)

    assert [record.action for record in records] == ["replaced", "kept"]
    assert tampered.read_bytes() == good_flow
    assert _tree_snapshot(tmp_path) == good_snapshot


# --------------------------------------------------------------------------
# T8 -- SUMO rendering where the scenario is paired, and an honest skip where not
# --------------------------------------------------------------------------


def test_sumo_routes_are_rendered_for_a_paired_scenario(tmp_path: Path) -> None:
    [record] = materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    assert record.sumo_path is not None

    entries = json.loads(record.flow_path.read_bytes())
    root = ET.parse(record.sumo_path).getroot()
    vehicles = root.findall("vehicle")

    # Independent route: the departure column recomputed from the flow file plus the
    # <begin> read straight out of the .sumocfg by this test.
    sumocfg = (
        _source_flow_path(HZ1X1_CONFIG).parent
        / "hangzhou_1x1_bc-tyc_18041610_1h.sumocfg"
    )
    begin = float(ET.parse(sumocfg).getroot().find("./time/begin").get("value"))
    expected = [f"{float(entry['startTime']) + begin:.2f}" for entry in entries]

    assert len(vehicles) == len(entries)
    assert [vehicle.get("depart") for vehicle in vehicles] == expected

    provenance = load_provenance(HZ1X1_KEY, 1, out_root=tmp_path)
    assert provenance["sumo"]["depart_offset"] == begin
    # hangzhou declares <vType id="pkw"> and never references it; the rendering mirrors
    # that faithfully, so the file must not feed a transfer measurement yet.
    assert provenance["sumo"]["vtype_bound"] is False
    assert "P7.0" in provenance["sumo"]["caveat"]
    assert provenance["sumo_skipped_reason"] is None


def test_sumo_rendering_is_skipped_with_a_reason_when_unpaired(tmp_path: Path) -> None:
    """grid4x4's .sumocfg names a route file that does not exist in the repo."""
    [record] = materialise(GRID4X4_CONFIG, [1], out_root=tmp_path)

    assert record.sumo_path is None
    assert not (record.directory / "routes.rou.xml").exists()

    provenance = load_provenance(GRID4X4_KEY, 1, out_root=tmp_path)
    assert provenance["sumo"] is None
    assert "grid4x4.rou.xml" in provenance["sumo_skipped_reason"]


# --------------------------------------------------------------------------
# T9 -- the tool cannot write outside out_root; CLI surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_key", ["..", ".", "a/b", "", "../escape"])
def test_a_scenario_key_that_could_escape_out_root_is_refused(
    bad_key: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="scenario key"):
        draw_dir(bad_key, 1, out_root=tmp_path)


def test_negative_and_duplicate_draw_ids_are_refused_before_any_write(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="draw ids must be >= 0"):
        materialise(HZ1X1_CONFIG, [1, -1], out_root=tmp_path)
    with pytest.raises(ValueError, match="repeated draw id"):
        materialise(HZ1X1_CONFIG, [1, 1], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == {}
    assert not (tmp_path / HZ1X1_KEY).exists()


def test_cli_writes_the_requested_draws_and_dry_run_writes_nothing(
    tmp_path: Path,
) -> None:
    dry_root = tmp_path / "dry"
    assert (
        main(
            [
                "--env-config",
                str(HZ1X1_CONFIG),
                "--draws",
                "1",
                "--draws-range",
                "1000",
                "1002",
                "--out-root",
                str(dry_root),
                "--dry-run",
            ]
        )
        == 0
    )
    assert _tree_snapshot(dry_root) == {}
    assert not dry_root.exists()

    wet_root = tmp_path / "wet"
    assert (
        main(
            [
                "--env-config",
                str(HZ1X1_CONFIG),
                "--draws",
                "1",
                "--draws-range",
                "1000",
                "1002",
                "--out-root",
                str(wet_root),
            ]
        )
        == 0
    )
    written = sorted(path.name for path in (wet_root / HZ1X1_KEY).iterdir())
    assert written == ["draw_0001", "draw_1000", "draw_1001"]
    for draw_id in (1, 1000, 1001):
        assert draw_config_path(HZ1X1_KEY, draw_id, out_root=wet_root).is_file()
