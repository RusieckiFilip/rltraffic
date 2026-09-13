"""P7.2a: the additive ``draw_NNNN/parity/`` phase of :mod:`offline.materialise_draws`.

Written against ``docs/briefs/BRIEF_35_p7.2a_sumo_parity_draws.md`` + Amendment A and
``docs/plans/p7.2a.md``.

Every test materialises into ``tmp_path``; none of them touches the real ``scenarios/draws/``
tree, and none of them writes anywhere under ``output/``.  Two tests need a simulator and
self-skip: T2 needs SUMO (the binding is asserted *in the running engine*, which is the one
claim a file-level check cannot make) and T4 needs CityFlow (P4.3's band gate).

The load-bearing claim of the file is T1: **the parent draw is read, never written.**  Draws
1000-1099 are what every merged held-out number since P4.6 resolves through (``DEFERRED`` 55),
and this phase reaches inside those very directories.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from offline import parity
from offline.materialise_draws import (
    PARITY_DIRNAME,
    PARITY_FORMAT_VERSION,
    PARITY_ROUTES_FILENAME,
    PARITY_SUMOCFG_FILENAME,
    ParityResult,
    draw_dir,
    load_parity_provenance,
    load_provenance,
    main,
    materialise,
    materialise_parity,
    parity_dir,
    parity_sumocfg_path,
    verify_p4_3_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HZ1X1_CONFIG = REPO_ROOT / "configs/sim/cityflow1x1.json"
GRID4X4_CONFIG = REPO_ROOT / "configs/sim/cityflow_grid4x4.json"
HZ1X1_KEY = "cityflow1x1"
P4_3_PROBE = REPO_ROOT / "docs/data/p4_3_probe.json"

#: P4.3's probe band starts here (``offline/rtg_calibration.py:189``); draw 201 is its
#: ``episodes[0]`` and the only band draw any test rolls.
FIRST_PROBE_DRAW = 201


def _cityflow_available() -> bool:
    try:
        import cityflow  # noqa: F401
    except Exception:
        return False
    return True


def _sumo_available() -> bool:
    """Both halves: the Python bindings and the binary the env actually launches."""
    try:
        import traci  # noqa: F401
    except ImportError:
        return False
    return shutil.which("sumo") is not None


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_snapshot(root: str | Path) -> dict[str, str]:
    """Map every file under *root* to its sha256, relative-path keyed.

    Whole-tree rather than expected-files-only: comparing whole snapshots catches a DELETED
    file, which comparing only the files we expect would not.
    """
    base = Path(root)
    if not base.exists():
        return {}
    return {
        str(path.relative_to(base)): _sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _parent_files(out_root: Path, draw_id: int) -> dict[str, str]:
    """The four parent files of one draw, sha256 by name."""
    target = draw_dir(HZ1X1_KEY, draw_id, out_root=out_root)
    return {path.name: _sha256_file(path) for path in sorted(target.iterdir()) if path.is_file()}


def _vehicle_types(rou_path: Path) -> Counter[str]:
    root = ET.parse(rou_path).getroot()
    return Counter(str(vehicle.get("type")) for vehicle in root.findall("vehicle"))


# ----------------------------------------------------------------------------------
# T1 -- the parent is read-only under every path, and the phase is idempotent
# ----------------------------------------------------------------------------------
def test_the_parity_phase_adds_a_subdirectory_and_touches_no_parent_byte(tmp_path: Path) -> None:
    """The claim the whole task rests on, checked by whole-tree snapshot.

    A parity phase that wrote into the parent would also poison ``materialise()``'s own no-op
    check for every future run: ``_existing_conflict`` refuses a draw directory holding an
    unexpected FILE (``offline/materialise_draws.py:606-609``).
    """
    materialise(HZ1X1_CONFIG, [1, 2, 3], out_root=tmp_path)
    before = _tree_snapshot(tmp_path)
    assert before, "the parents must exist before the parity phase runs"

    results = materialise_parity(HZ1X1_CONFIG, [1, 2, 3], out_root=tmp_path)

    after = _tree_snapshot(tmp_path)
    # Every pre-existing path still present AND still identical.  A deleted file makes the
    # left-hand side smaller than `before` and fails here.
    assert {name: digest for name, digest in after.items() if name in before} == before
    assert len(after) == len(before) + 9, "three parity dirs, three files each"

    assert [record.action for record in results] == ["written", "written", "written"]
    assert [record.parent_action for record in results] == ["kept", "kept", "kept"]
    for draw_id in (1, 2, 3):
        target = draw_dir(HZ1X1_KEY, draw_id, out_root=tmp_path) / PARITY_DIRNAME
        assert target.is_dir()
        assert sorted(path.name for path in target.iterdir()) == [
            "noteleport.sumocfg",
            "provenance.json",
            "routes.rou.xml",
        ]

    # Second run: everything kept, not one byte moved.
    snapshot = _tree_snapshot(tmp_path)
    again = materialise_parity(HZ1X1_CONFIG, [1, 2, 3], out_root=tmp_path)
    assert [record.action for record in again] == ["kept", "kept", "kept"]
    assert _tree_snapshot(tmp_path) == snapshot
    assert [path.name for path in tmp_path.rglob(".staging*")] == []


def test_every_vehicle_of_every_drawn_rendering_is_bound_to_the_parity_type(
    tmp_path: Path,
) -> None:
    """0 of n bound is exactly the defect P7.0 found in the shipped file; n of n is the fix.

    Checked against the PARENT's own vehicle count, so a rendering that lost or gained a
    vehicle fails here even if every vehicle it kept was bound.
    """
    materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)

    for draw_id in (1, 2):
        parent = draw_dir(HZ1X1_KEY, draw_id, out_root=tmp_path)
        bound = parity_dir(HZ1X1_KEY, draw_id, out_root=tmp_path) / PARITY_ROUTES_FILENAME
        expected = load_provenance(HZ1X1_KEY, draw_id, out_root=tmp_path)["draw"]["n_vehicles"]

        assert _vehicle_types(parent / "routes.rou.xml") == Counter({"None": expected})
        assert _vehicle_types(bound) == Counter({parity.PARITY_VTYPE_ID: expected})
        report = parity.vtype_binding_report(bound)
        assert parity.binding_is_complete(report)
        assert report.vtype_attributes == parity.parity_vtype_attributes()


def test_a_foreign_subdirectory_inside_a_draw_refuses_and_names_it(tmp_path: Path) -> None:
    """Amendment A3: fail closed on a directory we did not put there.

    Costs nothing today (no draw has a subdirectory) and stops a stray tree from riding along
    inside an artifact directory unnoticed.
    """
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    (draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "leftovers").mkdir()
    snapshot = _tree_snapshot(tmp_path)

    with pytest.raises(ValueError, match="leftovers"):
        materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == snapshot
    assert not (draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / PARITY_DIRNAME).exists()


# ----------------------------------------------------------------------------------
# T2 -- the binding is real IN THE RUNNING ENGINE, not just in the file
# ----------------------------------------------------------------------------------
def test_the_generated_config_builds_the_env_spec_p4_3_recorded(tmp_path: Path) -> None:
    """Amendment A1: one code path, already exercised on SUMO, checkable under ``==``.

    ``COLLECT_SETTINGS`` is an argv tuple, not a settings dict (``transfer_gate.py:1028``), so
    the settings reach ``make_env`` through ``collect``'s own parser.  This asserts the result
    equals what P4.3 recorded rather than asserting a dict someone retyped.
    """
    from offline.collect import _build_env_spec
    from offline.sumo_att_reference import collect_style_args

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    sumocfg = parity_sumocfg_path(HZ1X1_KEY, 1, out_root=tmp_path)

    args = collect_style_args(
        "sumo", "maxpressure", sumocfg, sentinel_out_dir=tmp_path / "never-created"
    )
    spec = _build_env_spec(args)

    recorded = json.loads(P4_3_PROBE.read_bytes())["env_settings"]
    common = sorted(set(spec.settings) & set(recorded))
    assert len(common) == 15
    assert {key: spec.settings[key] for key in common} == {key: recorded[key] for key in common}
    assert spec.backend == "sumo"
    assert Path(spec.paths["config"]) == sumocfg.resolve()
    assert not (tmp_path / "never-created").exists()


@pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not available")
def test_the_parity_type_and_the_teleport_regime_are_what_the_engine_actually_runs(
    tmp_path: Path,
) -> None:
    """The positive control P7.0's binding report demanded and no file check can give.

    A route file can name ``cf_parity`` on every vehicle and still be run by an engine that
    resolved a different type, or with teleporting on.  These three values are read back out of
    the running simulator: the effective type, its maxSpeed, and the option SUMO itself reports.
    """
    from experiments.envs import make_env
    from offline.collect import POLICIES, _build_env_spec
    from offline.sumo_att_reference import collect_style_args

    import numpy as np

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)

    args = collect_style_args(
        "sumo",
        "maxpressure",
        parity_sumocfg_path(HZ1X1_KEY, 1, out_root=tmp_path),
        sentinel_out_dir=tmp_path / "never-created",
    )
    env = make_env(_build_env_spec(args))
    try:
        policy = POLICIES["maxpressure"](env, args, np.random.default_rng(1000))
        info = env.reset(seed=1000)
        # reset(seed=1000) seeds the ENV RNG, which draws SUMO's --seed (C6 note in
        # envs/base_traffic_env.py:629-631); the drawn value is what actually reached SUMO.
        assert isinstance(env._engine_seed, int)

        assert env._sumo.simulation.getOption("time-to-teleport") == "-1"

        seen = 0
        for _ in range(20):
            present = list(env._sumo.vehicle.getIDList())
            for vehicle_id in present:
                assert env._sumo.vehicle.getTypeID(vehicle_id) == parity.PARITY_VTYPE_ID
                assert float(env._sumo.vehicle.getMaxSpeed(vehicle_id)) == 11.11
            seen += len(present)
            if seen and len(present):
                break
            _reward, terminated, truncated, info = env.step(policy(info))
            if terminated or truncated:
                break
        assert seen > 0, "no vehicle entered within 20 decision steps; nothing was asserted"
        assert env._sumo.simulation.getOption("time-to-teleport") == "-1"
    finally:
        env.close()


# ----------------------------------------------------------------------------------
# T3 -- CAP(E) per draw, recorded in the provenance, recomputed here by a second route
# ----------------------------------------------------------------------------------
def test_cap_e_holds_on_every_draw_and_the_record_says_so(tmp_path: Path) -> None:
    """A14(E): demand exact per vehicle. Recomputed here from the files, not read back.

    The provenance field and this test reach the same numbers by different code: the phase uses
    ``conversion_audit``'s extractors, and the assertion below rebuilds the multiset from the
    parsed XML and the parsed JSON directly.
    """
    draws = [1, 2, 3, 4, 5]
    materialise(HZ1X1_CONFIG, draws, out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, draws, out_root=tmp_path)

    for draw_id in draws:
        record = load_parity_provenance(HZ1X1_KEY, draw_id, out_root=tmp_path)
        audit = record["demand_audit"]

        flow = json.loads((draw_dir(HZ1X1_KEY, draw_id, out_root=tmp_path) / "flow.json").read_bytes())
        cityflow = Counter(
            (float(entry["startTime"]), tuple(entry["route"])) for entry in flow
        )
        root = ET.parse(
            parity_dir(HZ1X1_KEY, draw_id, out_root=tmp_path) / PARITY_ROUTES_FILENAME
        ).getroot()
        offset = float(audit["depart_offset"])
        sumo = Counter(
            (
                float(str(vehicle.get("depart"))) - offset,
                tuple(str(vehicle.find("route").get("edges")).split()),  # type: ignore[union-attr]
            )
            for vehicle in root.findall("vehicle")
        )

        assert cityflow == sumo, f"draw {draw_id}: the demand multisets differ"
        assert audit["multiset_equal"] is True
        assert audit["n_cityflow"] == audit["n_sumo"] == sum(cityflow.values())
        assert audit["n_index_aligned_equal"] == audit["n_sumo"]
        assert audit["order_matches"] is True
        assert record["n_bound"] == record["n_vehicles"] == audit["n_sumo"]


def test_a_lost_vehicle_is_refused_in_validation_and_no_draw_gets_a_parity_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The barrier: one bad draw out of five means nothing is written for ANY of the five.

    The fault is planted in the binding step of the LAST requested draw, so a phase that wrote
    as it validated would already have three or four parity directories on disk when it hit it.
    """
    import offline.materialise_draws as md

    draws = [1, 2, 3, 4, 5]
    materialise(HZ1X1_CONFIG, draws, out_root=tmp_path)
    snapshot = _tree_snapshot(tmp_path)

    real = md._render_bound_routes

    def lossy(text: str, *, draw_id: int) -> str:
        rendered = real(text, draw_id=draw_id)
        if draw_id != draws[-1]:
            return rendered
        start = rendered.index("<vehicle")
        end = rendered.index("</vehicle>", start) + len("</vehicle>")
        return rendered[:start] + rendered[end:]

    monkeypatch.setattr(md, "_render_bound_routes", lossy)

    with pytest.raises(ValueError, match="demand"):
        materialise_parity(HZ1X1_CONFIG, draws, out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == snapshot
    for draw_id in draws:
        assert not (draw_dir(HZ1X1_KEY, draw_id, out_root=tmp_path) / PARITY_DIRNAME).exists()
    assert [path.name for path in tmp_path.rglob(".staging*")] == []


# ----------------------------------------------------------------------------------
# T4 -- the P4.3 identity gate: these draws ARE P4.3's, or they are not
# ----------------------------------------------------------------------------------
@pytest.mark.skipif(not _cityflow_available(), reason="cityflow not installed")
def test_regenerated_draw_201_reproduces_p4_3s_recorded_episode(tmp_path: Path) -> None:
    """``DEFERRED`` 55: an id is a POINTER into a gitignored tree, not evidence.

    What licenses regenerating a band a retired worktree took with it is that the regenerated
    draw reproduces the recorded numbers exactly -- all four, under ``==``, read from the
    artifact rather than retyped.  This calls the same function ``--verify-p4-3-probe`` calls
    (Amendment A2 (iv)).
    """
    materialise(HZ1X1_CONFIG, [FIRST_PROBE_DRAW], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [FIRST_PROBE_DRAW], out_root=tmp_path)

    checks = verify_p4_3_probe(HZ1X1_CONFIG, [FIRST_PROBE_DRAW], out_root=tmp_path)

    recorded = json.loads(P4_3_PROBE.read_bytes())
    expected = next(
        episode for episode in recorded["episodes"] if episode["draw_id"] == FIRST_PROBE_DRAW
    )
    assert len(checks) == 1
    check = checks[0]
    assert check.draw_id == FIRST_PROBE_DRAW
    assert check.differing == ()
    assert check.matches is True
    for key in (
        "local_return",
        "local_return_from_lanes",
        "att_horizon",
        "horizon_vehicle_count",
        "decisions",
    ):
        assert check.observed[key] == expected[key], key
    # The two independent return routes of the probe agree, which is what makes the number a
    # measurement rather than a reading of one accumulator.
    assert check.observed["local_return"] == check.observed["local_return_from_lanes"]


# ----------------------------------------------------------------------------------
# T7 -- the os.replace target guard (brief section 0.4)
# ----------------------------------------------------------------------------------
def test_a_target_that_is_not_a_parity_directory_is_refused_before_any_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_commit``'s ``os.replace`` is one wrong target away from replacing a DRAW.

    Draws 1000-1099 are what every merged held-out number since P4.6 resolves through, so the
    target of every rename is checked against three properties before the first one runs.
    """
    import offline.materialise_draws as md

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    snapshot = _tree_snapshot(tmp_path)

    monkeypatch.setattr(
        md, "parity_dir", lambda key, draw_id, *, out_root: draw_dir(key, draw_id, out_root=out_root)
    )

    with pytest.raises(ValueError, match="parity"):
        materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == snapshot


def test_the_commit_guard_rejects_every_target_shape_that_is_not_a_parity_dir(
    tmp_path: Path,
) -> None:
    """The guard itself, called directly: suffix, parent name and containment."""
    import offline.materialise_draws as md

    out_root = tmp_path / "draws"
    good = out_root / HZ1X1_KEY / "draw_0001" / PARITY_DIRNAME
    assert md._checked_parity_target(good, out_root) == good

    # Each shape must be refused for its OWN reason, not merely refused: a guard that rejects
    # everything with one message cannot tell a near-miss name from an escape.
    for bad, reason in (
        (out_root / HZ1X1_KEY / "draw_0001", "must be named"),  # the draw itself
        (out_root / HZ1X1_KEY / PARITY_DIRNAME, "not a .*draw directory"),  # wrong parent
        (out_root / HZ1X1_KEY / "draw_1" / PARITY_DIRNAME, "not a .*draw directory"),  # unpadded
        (out_root / HZ1X1_KEY / "draw_0001" / "parity2", "must be named"),  # near-miss name
        (tmp_path / "elsewhere" / "draw_0001" / PARITY_DIRNAME, "outside out_root"),
    ):
        with pytest.raises(ValueError, match=reason):
            md._checked_parity_target(bad, out_root)


# ----------------------------------------------------------------------------------
# T8 -- the refusals
# ----------------------------------------------------------------------------------
def test_a_parent_whose_digest_disagrees_with_its_provenance_is_refused(tmp_path: Path) -> None:
    """The parent is the INPUT of this derivation; a parent that does not match its own record
    is not a draw whose demand we can claim to have bound."""
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    record_path = draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "provenance.json"
    record = json.loads(record_path.read_bytes())
    record["files"]["flow.json"] = "0" * 64
    record_path.write_bytes(json.dumps(record, indent=2, sort_keys=True).encode() + b"\n")
    snapshot = _tree_snapshot(tmp_path)

    with pytest.raises(ValueError, match="flow.json"):
        materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == snapshot


def test_a_parent_already_marked_bound_is_refused(tmp_path: Path) -> None:
    """Binding twice is the failure ``render_parity_rou_text`` already refuses; catching it at
    the parent's record turns a confusing regex failure into a statement about the input."""
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    record_path = draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "provenance.json"
    record = json.loads(record_path.read_bytes())
    record["sumo"]["vtype_bound"] = True
    record_path.write_bytes(json.dumps(record, indent=2, sort_keys=True).encode() + b"\n")

    with pytest.raises(ValueError, match="vtype_bound"):
        materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)


def test_a_differing_parity_dir_is_refused_and_only_force_replaces_it(tmp_path: Path) -> None:
    """Refused, never silently rewritten -- and ``force`` reaches the subdirectory only."""
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)

    parent_before = _parent_files(tmp_path, 1)
    target = parity_dir(HZ1X1_KEY, 1, out_root=tmp_path)
    (target / PARITY_SUMOCFG_FILENAME).write_text("<configuration/>\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match=PARITY_SUMOCFG_FILENAME):
        materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    assert _parent_files(tmp_path, 1) == parent_before

    replaced = materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path, force=True)
    assert [record.action for record in replaced] == ["replaced"]
    assert _parent_files(tmp_path, 1) == parent_before
    assert "<configuration/>" not in (target / PARITY_SUMOCFG_FILENAME).read_text(encoding="utf-8")
    assert [path.name for path in tmp_path.rglob(".staging*")] == []


def test_force_is_never_passed_to_the_parent_materialisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment A4: a flag that CANNOT reach the parent path is a control; a convention that
    it does not is not.  The spy records the kwarg the parent call actually received."""
    import offline.materialise_draws as md

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    (parity_dir(HZ1X1_KEY, 1, out_root=tmp_path) / PARITY_SUMOCFG_FILENAME).write_text(
        "<configuration/>\n", encoding="utf-8"
    )

    seen: list[dict[str, Any]] = []
    real = md.materialise

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(md, "materialise", spy)
    md.materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path, force=True)

    assert len(seen) == 1
    assert seen[0]["force"] is False, "--force must not reach the parent materialisation"


def test_an_out_root_inside_a_linked_worktree_is_refused_without_the_flag(
    tmp_path: Path,
) -> None:
    """``DEFERRED`` 55's rule as a control: materialise into the MAIN tree, never a worktree.

    The signal is the repository marker itself -- ``.git`` is a DIRECTORY in a main tree and a
    FILE in a linked worktree.
    """
    fake_worktree = tmp_path / "worktree"
    (fake_worktree / "scenarios").mkdir(parents=True)
    (fake_worktree / ".git").write_text("gitdir: /somewhere/.git/worktrees/x\n", encoding="utf-8")
    out_root = fake_worktree / "scenarios" / "draws"

    with pytest.raises(ValueError, match="worktree"):
        materialise_parity(HZ1X1_CONFIG, [1], out_root=out_root)
    assert not out_root.exists(), "a refused run must not create its own out_root"

    # The escape hatch exists and is explicit.
    materialise(HZ1X1_CONFIG, [1], out_root=out_root)
    records = materialise_parity(HZ1X1_CONFIG, [1], out_root=out_root, allow_worktree=True)
    assert [record.action for record in records] == ["written"]


def test_a_scenario_without_a_sumo_pairing_is_refused_with_its_reason(tmp_path: Path) -> None:
    """grid4x4's ``.sumocfg`` names a route file the repo does not contain, so there is nothing
    to bind.  Skipping silently is how a transfer measurement ends up running DEFAULT_VEHTYPE."""
    materialise(GRID4X4_CONFIG, [1], out_root=tmp_path)
    with pytest.raises(ValueError, match="grid4x4.rou.xml"):
        materialise_parity(GRID4X4_CONFIG, [1], out_root=tmp_path)


# ----------------------------------------------------------------------------------
# Amendment A6 -- the reorder: an existing parent's refusal writes nothing at all
# ----------------------------------------------------------------------------------
def test_an_existing_parents_refusal_prevents_a_new_parent_from_being_materialised(
    tmp_path: Path,
) -> None:
    """Amendment A6. Phase 2 runs on every EXISTING parent before phase 1 creates any new one.

    Without the reorder, requesting [1, 201] with draw 1 broken would first materialise 201 --
    a hundred new directories in the campaign's case -- and only then refuse.
    """
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    record_path = draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "provenance.json"
    record = json.loads(record_path.read_bytes())
    record["files"]["flow.json"] = "0" * 64
    record_path.write_bytes(json.dumps(record, indent=2, sort_keys=True).encode() + b"\n")
    snapshot = _tree_snapshot(tmp_path)

    with pytest.raises(ValueError, match="flow.json"):
        materialise_parity(HZ1X1_CONFIG, [1, FIRST_PROBE_DRAW], out_root=tmp_path)

    assert _tree_snapshot(tmp_path) == snapshot
    assert not draw_dir(HZ1X1_KEY, FIRST_PROBE_DRAW, out_root=tmp_path).exists(), (
        "the new parent was materialised before the existing one was validated"
    )


# ----------------------------------------------------------------------------------
# Amendment A7 -- the record's own invariants
# ----------------------------------------------------------------------------------
def test_the_parity_record_carries_no_path_field_without_a_digest_twin(tmp_path: Path) -> None:
    """The parent's rule, extended to the new record rather than assumed for it.

    A path says where a file was; only a digest says which file it was.  ``net.reference`` and
    ``net.resolved`` are paths and are twinned by ``net.sha256``; every ``parent.*`` entry is
    itself a digest.
    """
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    record = load_parity_provenance(HZ1X1_KEY, 1, out_root=tmp_path)

    assert record["format_version"] == PARITY_FORMAT_VERSION
    assert record["parity_contract_version"] == parity.PARITY_CONTRACT_VERSION
    assert record["vtype_id"] == parity.PARITY_VTYPE_ID
    assert record["vtype_attributes"] == parity.parity_vtype_attributes()
    assert record["sumocfg"] == {"begin": 0, "end": parity.SUMO_END_SECONDS, "time_to_teleport": -1}

    parent = draw_dir(HZ1X1_KEY, 1, out_root=tmp_path)
    assert record["parent"] == {
        "flow_sha256": _sha256_file(parent / "flow.json"),
        "routes_sha256": _sha256_file(parent / "routes.rou.xml"),
        "provenance_sha256": _sha256_file(parent / "provenance.json"),
    }
    assert record["net"]["sha256"] == _sha256_file(
        parity.DECLARED_SOURCE_NET
    ), "the SHIPPED network is the one referenced; a copy would be a drift source"

    target = parity_dir(HZ1X1_KEY, 1, out_root=tmp_path)
    assert record["files"] == {
        PARITY_ROUTES_FILENAME: _sha256_file(target / PARITY_ROUTES_FILENAME),
        PARITY_SUMOCFG_FILENAME: _sha256_file(target / PARITY_SUMOCFG_FILENAME),
    }
    # Every path-shaped value is twinned by a digest in the same block.
    assert set(record["net"]) == {"reference", "resolved", "sha256"}
    assert (target / record["net"]["reference"]).resolve() == parity.DECLARED_SOURCE_NET.resolve()


def test_the_parity_paths_are_pure_arithmetic_and_touch_no_filesystem(tmp_path: Path) -> None:
    """P7.2b builds its rotation from these before anything exists, so they must not stat."""
    expected = tmp_path / HZ1X1_KEY / "draw_0201" / PARITY_DIRNAME
    assert parity_dir(HZ1X1_KEY, 201, out_root=tmp_path) == expected
    assert parity_sumocfg_path(HZ1X1_KEY, 201, out_root=tmp_path) == (
        expected / PARITY_SUMOCFG_FILENAME
    )
    assert not expected.exists()
    assert _tree_snapshot(tmp_path) == {}


def test_the_generated_config_names_the_shipped_net_and_the_bound_routes(tmp_path: Path) -> None:
    """A generated config that referenced the PARENT's routes would run DEFAULT_VEHTYPE while
    every file-level check still passed."""
    from utils.sumo_utils import iter_sumo_input_paths, validate_sumo_inputs_exist

    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    sumocfg = parity_sumocfg_path(HZ1X1_KEY, 1, out_root=tmp_path)

    validate_sumo_inputs_exist(sumocfg)
    inputs = iter_sumo_input_paths(sumocfg)
    assert parity.DECLARED_SOURCE_NET.resolve() in inputs
    assert (parity_dir(HZ1X1_KEY, 1, out_root=tmp_path) / PARITY_ROUTES_FILENAME).resolve() in inputs
    assert (
        draw_dir(HZ1X1_KEY, 1, out_root=tmp_path) / "routes.rou.xml"
    ).resolve() not in inputs

    root = ET.parse(sumocfg).getroot()
    teleport = root.find(".//time-to-teleport")
    assert teleport is not None and teleport.get("value") == "-1", "A15(c) binds every generated config"
    assert int(str(root.find(".//end").get("value"))) == parity.SUMO_END_SECONDS  # type: ignore[union-attr]


# ----------------------------------------------------------------------------------
# T5 -- semantic equivalence with the config P7.1 committed by hand
# ----------------------------------------------------------------------------------
def test_draw_zero_is_equivalent_to_the_committed_teleport_free_scenario(tmp_path: Path) -> None:
    """Draw 0 is the nominal control: its demand IS the source demand.

    So the generated scenario must be the committed one in every respect that a simulator can
    observe.  **Bytes are deliberately not compared**: the committed file carries P7.1's
    hand-written header and names its route file by the scenario stem, while a generated one
    names ``routes.rou.xml`` next to it -- differences no engine can see.  What is compared is
    what the engine reads: the resolved network, the three time values, and the demand itself.
    """
    materialise(HZ1X1_CONFIG, [0], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [0], out_root=tmp_path)

    generated = parity_sumocfg_path(HZ1X1_KEY, 0, out_root=tmp_path)
    committed = parity.NOTELEPORT_SUMOCFG

    def _scenario(path: Path) -> dict[str, Any]:
        root = ET.parse(path).getroot()
        return {
            "net": (path.parent / str(root.find(".//net-file").get("value"))).resolve(),  # type: ignore[union-attr]
            "begin": str(root.find(".//begin").get("value")),  # type: ignore[union-attr]
            "end": str(root.find(".//end").get("value")),  # type: ignore[union-attr]
            "teleport": str(root.find(".//time-to-teleport").get("value")),  # type: ignore[union-attr]
        }

    assert _scenario(generated) == _scenario(committed)

    # The depart is a TIME, so it is compared as one, through the same extractor CAP(E) and
    # A15(g)'s 2021/2021 verdict used.  Comparing the raw attribute strings would compare
    # formatting instead: the committed file inherits the shipped .rou.xml's "0" while a drawn
    # rendering carries render_sumo's two-decimal "0.00" -- asserted below, so that the
    # distinction is stated rather than silently sidestepped.
    from offline.conversion_audit import demand_from_route_file

    drawn = parity_dir(HZ1X1_KEY, 0, out_root=tmp_path) / PARITY_ROUTES_FILENAME
    drawn_demand = demand_from_route_file(drawn)
    committed_demand = demand_from_route_file(parity.DECLARED_PARITY_ROU)
    assert Counter(drawn_demand) == Counter(committed_demand)
    assert drawn_demand == committed_demand, "same demand AND the same order"
    assert len(drawn_demand) == 2021

    def _depart_strings(path: Path) -> list[str]:
        root = ET.parse(path).getroot()
        return [str(vehicle.get("depart")) for vehicle in root.findall("vehicle")]

    assert _depart_strings(drawn)[:2] == ["0.00", "3.00"]
    assert _depart_strings(parity.DECLARED_PARITY_ROU)[:2] == ["0", "3"]
    assert _vehicle_types(drawn) == Counter({parity.PARITY_VTYPE_ID: 2021})
    assert generated.read_bytes() != committed.read_bytes(), (
        "if these ever became byte-equal this test would be asserting nothing about equivalence"
    )


# ----------------------------------------------------------------------------------
# T9 -- the dry run and the CLI
# ----------------------------------------------------------------------------------
def test_the_dry_run_writes_nothing_and_reports_each_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three states in one run: a parity dir that exists, one that does not, and a parent that
    does not exist either."""
    materialise(HZ1X1_CONFIG, [1, 2], out_root=tmp_path)
    materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)
    snapshot = _tree_snapshot(tmp_path)

    planned = materialise_parity(
        HZ1X1_CONFIG, [1, 2, FIRST_PROBE_DRAW], out_root=tmp_path, dry_run=True
    )

    assert _tree_snapshot(tmp_path) == snapshot
    assert not draw_dir(HZ1X1_KEY, FIRST_PROBE_DRAW, out_root=tmp_path).exists()
    assert [record.action for record in planned] == ["planned", "planned", "planned"]
    assert [record.parent_action for record in planned] == ["kept", "kept", "planned"]

    assert main(
        [
            "--parity",
            "--env-config",
            str(HZ1X1_CONFIG),
            "--draws",
            "1",
            "2",
            "--out-root",
            str(tmp_path),
            "--dry-run",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "parity kept" in out
    assert "would add parity" in out
    assert _tree_snapshot(tmp_path) == snapshot


def test_the_cli_writes_the_requested_parity_dirs_and_refuses_with_a_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0 on success, 1 on refusal with the reason printed -- the shape ``main`` already has."""
    assert main(
        [
            "--parity",
            "--env-config",
            str(HZ1X1_CONFIG),
            "--draws-range",
            "1",
            "3",
            "--out-root",
            str(tmp_path),
        ]
    ) == 0
    capsys.readouterr()
    for draw_id in (1, 2):
        assert (parity_dir(HZ1X1_KEY, draw_id, out_root=tmp_path) / PARITY_SUMOCFG_FILENAME).is_file()

    (parity_dir(HZ1X1_KEY, 1, out_root=tmp_path) / PARITY_ROUTES_FILENAME).write_text(
        "<routes/>\n", encoding="utf-8"
    )
    snapshot = _tree_snapshot(tmp_path)
    assert main(
        [
            "--parity",
            "--env-config",
            str(HZ1X1_CONFIG),
            "--draws",
            "1",
            "--out-root",
            str(tmp_path),
        ]
    ) == 1
    out = capsys.readouterr().out
    assert "materialise_draws:" in out
    assert PARITY_ROUTES_FILENAME in out
    assert _tree_snapshot(tmp_path) == snapshot


def test_the_result_dataclass_reports_the_paths_it_wrote(tmp_path: Path) -> None:
    """The record a caller acts on must name the files, not just the directory."""
    materialise(HZ1X1_CONFIG, [1], out_root=tmp_path)
    (record,) = materialise_parity(HZ1X1_CONFIG, [1], out_root=tmp_path)

    assert isinstance(record, ParityResult)
    assert record.scenario_key == HZ1X1_KEY
    assert record.draw_id == 1
    assert record.pool == "training"
    assert record.directory == parity_dir(HZ1X1_KEY, 1, out_root=tmp_path)
    assert record.routes_path == record.directory / PARITY_ROUTES_FILENAME
    assert record.sumocfg_path == parity_sumocfg_path(HZ1X1_KEY, 1, out_root=tmp_path)
    assert record.provenance_path == record.directory / "provenance.json"
    assert record.n_bound == record.n_vehicles > 0
    for path in (record.routes_path, record.sumocfg_path, record.provenance_path):
        assert path.is_file()
