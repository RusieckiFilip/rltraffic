"""``att_rederivation.run_campaign``'s ``should_stop`` keyword (P9.1, ``BRIEF_46`` Amendment E, E.4-2b: an evaluation
session ends between episodes).

No format changes: P8.4b's per-episode cell files (``p8.4b-rederivation/1.0``) are written exactly as before.  The default,
``None``, rolls every cell and returns the registered result's keys; given, it is asked before every episode -- never
inside one -- and a ``True`` ends the campaign there, the next call resuming from disk.  Alignment: one cell file per
``(scenario, arm, seed, draw)``, as P8.4b names it.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

import offline.att_rederivation as ar
from offline.att_rederivation import CellKey
from tests.test_att_rederivation_factory_builder import _stub_campaign

CELLS = [CellKey("hz1x1", "bc@x", 101, draw) for draw in (1000, 1001, 1002, 1003)]
COMMITTED = {(c.scenario, c.arm, c.seed, c.draw_id): 42.0 for c in CELLS}


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    roots = _stub_campaign(monkeypatch, tmp_path)
    monkeypatch.setattr(ar, "build_cell_factory", lambda s, arm, seed, r, **k: (None, {"kind": "module"}))
    return ar.run_campaign(CELLS, roots=roots, engine_seed=1, device=None, committed=COMMITTED, protected=(), **kwargs)


def _files(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.glob("cell_*.json"))


def test_without_it_every_episode_is_rolled_and_the_result_is_the_registered_one(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _run(tmp_path, monkeypatch)
    assert set(outcome) == {"rolled", "skipped_already_present", "refused", "n_refused", "seconds"}
    assert outcome["rolled"] == 4 and len(_files(tmp_path)) == 4


def test_a_true_ends_the_campaign_between_episodes_and_the_next_call_resumes_from_disk(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter([False, False, True])
    outcome = _run(tmp_path, monkeypatch, should_stop=lambda: next(answers))
    assert (outcome["rolled"], outcome["stopped"], outcome["not_reached"]) == (2, True, 2)
    assert _files(tmp_path) == ["cell_hz1x1_bc_at_x_seed101_draw1000.json", "cell_hz1x1_bc_at_x_seed101_draw1001.json"]
    again = _run(tmp_path, monkeypatch)
    assert (again["rolled"], again["skipped_already_present"]) == (2, 2) and len(_files(tmp_path)) == 4


def test_it_is_asked_before_every_episode_and_a_false_changes_nothing(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int] = []
    outcome = _run(tmp_path, monkeypatch, should_stop=lambda: asked.append(1) is None and False)
    assert len(asked) == 4 and (outcome["rolled"], outcome["stopped"], outcome["not_reached"]) == (4, False, 0)


def test_the_keyword_defaults_to_none() -> None:
    parameter = inspect.signature(ar.run_campaign).parameters["should_stop"]
    assert parameter.default is None and parameter.kind is inspect.Parameter.KEYWORD_ONLY
