"""``att_rederivation.run_campaign``'s ``factory_builder`` keyword (P9.1, ``BRIEF_46`` Amendment B, B.1(3); M1's Q5).

No format changes: P8.4b's per-episode cell files (``p8.4b-rederivation/1.0``) are written exactly as before, with the
builder's ``policy_source``.  The default, ``None``, is the module's own ``build_cell_factory`` resolved at CALL time,
so the registered route -- and every test that replaces that module name -- is unchanged; a given builder replaces it.
Alignment: one cell file per ``(scenario, arm, seed, draw)``, as P8.4b names it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offline.att_rederivation as ar
from offline.att_rederivation import CellKey


def _stub_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, att: float = 42.0) -> Any:
    """Everything around the factory replaced: one episode, whose ``att_ours`` is *att*."""
    def fake_probe(**kwargs: Any) -> Any:
        class Episode:
            att_ours = att

            def as_record(self) -> dict[str, Any]:
                return {"arm": kwargs["arm"], "seed": kwargs["seed"], "draw_id": kwargs["draw_id"], "att_ours": att,
                        "att_engine": att - 1.0}

        return Episode()

    monkeypatch.setattr(ar, "rederivation_env_settings", lambda *a, **k: {"max_steps": 1, "delta_time": 1})
    monkeypatch.setattr("offline.admission_probe.probe_episode", fake_probe)
    monkeypatch.setattr("offline.admission_probe.created_from_flow", lambda *a, **k: 1)
    monkeypatch.setattr("offline.materialise_draws.draw_config_path", lambda *a, **k: tmp_path / "c.json")
    return type("R", (), {"work_dir": tmp_path, "draws_root": tmp_path, "corpus_root": tmp_path,
                          "output_root": tmp_path, "repo_root": tmp_path})()


CELLS = [CellKey("hz1x1", "bc@x", 101, 1000)]
COMMITTED = {("hz1x1", "bc@x", 101, 1000): 42.0}
CELL_FILE = "cell_hz1x1_bc_at_x_seed101_draw1000.json"


def test_the_default_is_the_modules_own_builder_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    """Default identity: without the keyword, the module-level ``build_cell_factory`` -- the one a monkeypatch put in
    place -- builds the cell's factory, exactly as before the keyword existed."""
    roots = _stub_campaign(monkeypatch, tmp_path)
    module_calls: list[str] = []
    monkeypatch.setattr(ar, "build_cell_factory",
                        lambda s, arm, seed, r, **k: module_calls.append(arm) or (None, {"kind": "module"}))
    outcome = ar.run_campaign(CELLS, roots=roots, engine_seed=1, device=None, committed=COMMITTED, protected=())
    assert module_calls == ["bc@x"] and outcome["rolled"] == 1
    record = json.loads((tmp_path / CELL_FILE).read_text())
    assert record["policy_source"] == {"kind": "module"} and record["reproduces_committed"] is True


def test_a_given_builder_replaces_the_modules_and_the_module_builder_is_not_called(monkeypatch: pytest.MonkeyPatch,
                                                                                    tmp_path: Path) -> None:
    roots = _stub_campaign(monkeypatch, tmp_path)
    module_calls: list[str] = []
    given: list[str] = []
    monkeypatch.setattr(ar, "build_cell_factory", lambda s, arm, seed, r, **k: module_calls.append(arm) or (None, {}))
    ar.run_campaign(CELLS, roots=roots, engine_seed=1, device=None, committed=COMMITTED, protected=(),
                    factory_builder=lambda s, arm, seed, r, **k: given.append(arm) or (None, {"kind": "given"}))
    assert given == ["bc@x"] and module_calls == []
    assert json.loads((tmp_path / CELL_FILE).read_text())["policy_source"] == {"kind": "given"}


def test_the_keyword_defaults_to_none() -> None:
    import inspect

    parameter = inspect.signature(ar.run_campaign).parameters["factory_builder"]
    assert parameter.default is None and parameter.kind is inspect.Parameter.KEYWORD_ONLY
