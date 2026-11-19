"""P9.4's smoke (``offline/p9_4_smoke.py``): it writes only into its scratch root, copies (never links) the registered
references it reads, returns PASS/FAIL and seconds per path and no other number, and a G1 or G2 cell that differs under
``==`` is a FAIL naming the fields.  The real paths are exercised by running the smoke itself (outside CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import offline.p9_4_evaluation as ev
import offline.p9_4_mix50_fresh as p94
import offline.p9_4_smoke as smoke

REPO_ROOT = Path(__file__).resolve().parent.parent


def _roots(tmp_path: Path) -> p94.Roots:
    return p94.smoke_roots(tmp_path / "scratch", repo_root=REPO_ROOT)


def test_the_references_are_copied_into_the_scratch_root_and_never_linked(tmp_path: Path) -> None:
    source = tmp_path / "main_output"
    (source / "p4_7" / "checkpoints").mkdir(parents=True)
    for seed in p94.SEEDS:
        (source / "p4_7" / "checkpoints" / f"mix50_dt_seed{seed}.pt").write_bytes(f"weights {seed}".encode())
    (source / "SHA256SUMS_p4_7.txt").write_text("sums\n", encoding="utf-8")
    roots = _roots(tmp_path)
    copied = smoke.copy_references(roots, source_output=source)
    assert len(copied) == 6
    for name in copied:
        target = Path(roots.output_root) / name
        assert target.is_file() and not target.is_symlink()
        assert target.read_bytes() == (source / name).read_bytes()
    with pytest.raises(ValueError, match="main tree's"):
        smoke.copy_references(p94.Roots(repo_root=REPO_ROOT, output_root=source, corpus_root=tmp_path,
                                        draws_root=tmp_path), source_output=source)


@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Every heavy call replaced: the smoke's orchestration and its result shape are what is tested."""
    calls: list[str] = []
    monkeypatch.setattr(smoke, "copy_references", lambda roots, **kw: calls.append("copy") or [])
    monkeypatch.setattr(p94, "load_declaration", lambda roots: {"rtg_scale": 41000.0})

    def session(roots: Any, **kwargs: Any) -> Any:
        calls.append(f"train_session steps={kwargs['steps']}")
        raise p94.ControlFailed("CONTROL: 0 of 5 seeds reproduce")

    monkeypatch.setattr(p94, "train_session", session)
    monkeypatch.setattr(p94, "read_control", lambda roots: {"seeds": [{}] * 5, "passed": False})
    monkeypatch.setattr(p94, "new_tier_inputs", lambda roots, declaration: object())
    monkeypatch.setattr(p94, "run_status", lambda roots, run, **kw: "done")
    monkeypatch.setattr(ev, "route_i", lambda roots, declaration, **kw: calls.append(f"route_i {kw}") or {})
    monkeypatch.setattr(ev, "declare_route_ii", lambda roots, **kw: calls.append("declare_route_ii") or {})
    monkeypatch.setattr(ev, "run_route_ii", lambda roots, declaration, **kw: {"rolled": 1, "skipped_already_present": 0})
    monkeypatch.setattr(ev, "roll", lambda roots, tasks, declaration, **kw: calls.extend(f"roll {k}" for k, _ in tasks)
                        or {"rolled": len(tasks), "left": 0, "stopped_by": None, "seconds": [1.0]})
    monkeypatch.setattr(ev, "pending", lambda roots, kind, cells, declaration, **kw: [])
    outcome = {"g1": [], "g2": []}
    monkeypatch.setattr(ev, "g1_comparison", lambda roots, cells: {"mismatches": outcome["g1"]})
    monkeypatch.setattr(ev, "g2_comparison", lambda roots, cells: {"mismatches": outcome["g2"]})
    return {"calls": calls, "outcome": outcome, "roots": _roots(tmp_path)}


def test_the_smoke_runs_every_distinct_path_once_and_reports_only_pass_and_seconds(stubbed: dict[str, Any]) -> None:
    result = smoke.run_smoke(stubbed["roots"], steps=20)
    assert result["all_passed"] is True and len(result["paths"]) == 7
    for path in result["paths"]:
        assert set(path) == {"path", "result", "seconds", "error"} and path["result"] == "PASS"
    calls = stubbed["calls"]
    assert calls[:2] == ["copy", "train_session steps=20"]
    assert "route_i {'seeds': [101], 'draws': [1000], 'steps': 20}" in calls
    assert [c for c in calls if c.startswith("roll")] == ["roll new", "roll new", "roll g1", "roll g2"]


def test_a_guard_cell_that_differs_is_a_fail_naming_its_fields(stubbed: dict[str, Any]) -> None:
    stubbed["outcome"]["g1"] = [{"cell": "cell_mix50_b_mean_k100_seed101_draw1000.json", "differing": ["e_sumo"]}]
    result = smoke.run_smoke(stubbed["roots"], steps=20)
    (g1,) = [p for p in result["paths"] if p["path"].startswith("G1")]
    assert g1["result"] == "FAIL" and "e_sumo" in g1["error"] and result["all_passed"] is False


def test_a_control_path_without_its_five_seed_record_fails(stubbed: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p94, "read_control", lambda roots: None)
    result = smoke.run_smoke(stubbed["roots"], steps=20)
    assert result["paths"][0]["result"] == "FAIL" and "control record" in result["paths"][0]["error"]
    assert [p["path"] for p in result["paths"]][-2:] == ["G1 cell (mix50, b_mean_k100, draw 1000)",
                                                          "G2 cell (fixedtime_kcf, draw 1000)"]
