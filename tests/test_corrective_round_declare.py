"""P9.1's declarations (``offline/corrective_round.py``; ``BRIEF_46`` 3.4 ``declare``), written BEFORE any training:

* ``mix_m1000_m500`` -- P4.6's declaration format for the tier and its two components, plus A27(e)'s design facts
  reproduced from the corpus (a refusal otherwise) and the kept decile's composition by source and behaviour seed;
* ``random_fresh`` on each scenario -- the collection checked (A27(d)'s manifest and 200 distinct schedules), the
  declared rule's values computed from the tier's own 200 episodes (hz1x1: ``target_rtg`` / ``rtg_scale``, the values
  plan Q2 types; grid: ``ladder_att`` / ``mean_training_return``, recorded and never typed), and the OLD tier's basis
  quoted from its committed declaration, so the before/after comparison names both of its changes (A27(d)).

Format: ``p9.1-declaration/1.0``.  Fixtures: ``tests/test_method_tier_grid.py``'s one-intersection tier and
``tests/p5_2b_fixtures.py``'s 16-intersection tier (contract C6 v1.1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.a27_arms as arms
import offline.corrective_round as cr
import offline.method_tier_grid as grid
import offline.tier_sweep as ts
from tests import p5_2b_fixtures as fx
from tests.test_method_tier_grid import CONTEXT, fixture_spec, write_tier_dir

REPO = Path(__file__).resolve().parents[1]


def _roots(tmp_path: Path) -> cr.Roots:
    output = tmp_path / "output"
    output.mkdir()
    return cr.Roots(repo_root=REPO, output_root=output, corpus_root=tmp_path / "corpus", draws_root=tmp_path / "draws")


def _bypass_collection_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The two collection checks are tested on their own (test_corrective_round_data, test_a27_arms); here they are
    observed being CALLED, and passed, so the declaration's arithmetic can be checked on a small fixture."""
    calls: list[str] = []
    monkeypatch.setattr(cr, "assert_random_fresh_manifest", lambda manifest: calls.append("manifest"))
    monkeypatch.setattr(arms, "assert_distinct_schedules",
                        lambda directory, expected: calls.append(f"distinct:{expected}") or {"episodes": expected})
    return calls


def test_the_hz1x1_random_fresh_declaration_computes_the_rule_on_its_own_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    directory = write_tier_dir(Path(roots.corpus_root), "cf_hz1x1__random_fresh",
                               episodes=[(1, -1.5, 100.0), (2, -3.0, 130.0), (3, -2.0, 110.0)])
    monkeypatch.setitem(grid.A27_TIERS, "random_fresh", fixture_spec("random_fresh", [directory.name], stream_count=3))
    monkeypatch.setattr(grid, "CONTEXT_LENGTH", CONTEXT)
    calls = _bypass_collection_checks(monkeypatch)
    payload = cr.declare_random_fresh(roots, "hz1x1")
    assert calls == ["manifest", "distinct:3"]
    returns = []
    for name in sorted(p.name for p in directory.glob("ep*.npz")):
        with np.load(directory / name) as arrays:
            returns.append(float(np.sum(arrays["ix0_local_reward"], dtype=np.float64)))
    values = payload["values_to_type"]
    assert (values["target_rtg"], values["rtg_scale"]) == (max(returns), max(abs(r) for r in returns))
    assert payload["training_streams"] == payload["split_streams"] == 3
    assert payload["format_version"] == cr.DECLARATION_FORMAT_VERSION and payload["registered_in"] == "A27(d)"
    assert payload["label"] == cr.LABELS["A27(d)"]
    old = payload["old_tier_basis"]
    assert (old["split_streams"], old["training_streams"], old["rtg_scale"]) == (400, 200, 40294.0)
    assert old["source"] == "docs/data/p4_6_declaration.json"
    assert "400" in payload["what_changes_besides_the_generator"]
    assert (roots.run_root / "declaration_random_fresh_hz1x1.json").is_file()


def test_the_grid_random_fresh_declaration_records_its_ladder_values_computed_not_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    fx.write_synthetic_tier(Path(roots.corpus_root), episodes_per_draw=1)
    monkeypatch.setattr(ts, "A27_TIERS", {"random_fresh": ts.TierSpec(
        tier="random_fresh", dirs=(fx.RANDOM_DIR,), subsample="none", episode_count=3, ladder_att=math.nan,
        mean_training_return=math.nan)})
    monkeypatch.setattr(ts, "adjacency_for_tier", lambda *args, **kwargs: None)
    monkeypatch.setattr(ts, "declaration_artifact", lambda parts, root: {"format_version": ts.DECLARATION_FORMAT_VERSION,
                                                                          "ladder_att": math.nan,
                                                                          "mean_training_return": math.nan})
    calls = _bypass_collection_checks(monkeypatch)
    payload = cr.declare_random_fresh(roots, "grid4x4")
    assert calls == ["manifest", "distinct:3"]
    directory = Path(roots.corpus_root) / fx.RANDOM_DIR
    finals, streams = [], []
    for name in sorted(p.name for p in directory.glob("ep*.npz")):
        with np.load(directory / name) as arrays:
            finals.append(float(arrays["att_per_step"][-1]))
            streams += [float(np.sum(arrays[f"ix{i}_local_reward"], dtype=np.float64))
                        for i in range(len(arrays["ix_ids"]))]
    p52 = payload["p5_2_format_declaration"]
    assert p52["ladder_att"] == sum(finals) / len(finals)
    assert p52["mean_training_return"] == sum(streams) / len(streams)
    assert payload["computed"]["ladder_att_definition"].startswith("mean over the tier's episodes of the final")
    old = payload["old_tier_basis"]
    assert (old["episodes_available"], old["episodes_selected"], old["source"]) == (
        400, 200, "docs/data/p5_2_declaration_random.json")
    assert (roots.run_root / "declaration_random_fresh_grid4x4.json").is_file()


def test_the_mixture_declaration_carries_p4_6s_format_and_a27es_facts(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    seen: dict[str, Any] = {}

    def fake_declaration(corpus_root: Any, tiers: Any) -> dict[str, Any]:
        seen["tiers"] = list(tiers)
        return {"format_version": grid.DECLARATION_FORMAT_VERSION, "tiers": {"mix_m1000_m500": {"target_rtg": -5959.0}}}

    monkeypatch.setattr(grid, "declaration_artifact", fake_declaration)
    monkeypatch.setattr(cr, "_mixture_selection", lambda r: ("selected", "kept"))
    monkeypatch.setattr(arms, "assert_mixture_design_facts", lambda selected, kept: {
        "pool": {"mappo1000": 100, "mappo500": 100}, "kept_decile": arms.MIXTURE_DESIGN_FACTS["kept_decile"],
        "decile_returns": [-6627.0, -5959.0], "weaker_source_kept": 6, "p4a_band_1_to_19": True})
    payload = cr.declare_mixture(roots)
    assert seen["tiers"] == ["mappo1000", "mappo500", "mix_m1000_m500"]
    assert payload["design_facts"] == json.loads(json.dumps(arms.MIXTURE_DESIGN_FACTS))
    assert payload["measured"]["weaker_source_kept"] == 6 and payload["design_facts_reproduced"] is True
    assert payload["registered_in"] == "A27(e)" and payload["label"] == cr.LABELS["A27(e)"]
    assert payload["p4_6_format_declaration"]["format_version"] == grid.DECLARATION_FORMAT_VERSION
    assert (roots.run_root / "declaration_mix_m1000_m500.json").is_file()


def test_the_mixture_declaration_refuses_when_the_facts_do_not_reproduce(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    monkeypatch.setattr(grid, "declaration_artifact", lambda corpus_root, tiers: {"tiers": {}})
    monkeypatch.setattr(cr, "_mixture_selection", lambda r: ("selected", "kept"))

    def refuse(selected: Any, kept: Any) -> Any:
        raise ValueError("the overlap mixture does not reproduce A27(e)'s design facts")

    monkeypatch.setattr(arms, "assert_mixture_design_facts", refuse)
    with pytest.raises(ValueError, match="design facts"):
        cr.declare_mixture(roots)
    assert not (roots.run_root / "declaration_mix_m1000_m500.json").exists()


@pytest.fixture(scope="module")
def corpus_v11_root() -> Path:
    import os

    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO / "datasets_v11"
    if not candidate.is_dir():
        pytest.skip(f"format v1.1 corpus not found at {candidate}: set RLTRAFFIC_CORPUS_V11 to declare the mixture")
    return candidate


def test_the_real_mixture_declares_with_a27es_facts(corpus_v11_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    roots = cr.Roots(repo_root=REPO, output_root=output, corpus_root=corpus_v11_root, draws_root=tmp_path)
    payload = cr.declare_mixture(roots)
    entry = payload["p4_6_format_declaration"]["tiers"]["mix_m1000_m500"]
    assert (entry["target_rtg"], entry["rtg_scale"], entry["training_streams"]) == (-5959.0, 11043.0, 200)
    assert payload["measured"]["kept_decile"] == arms.MIXTURE_DESIGN_FACTS["kept_decile"]


def test_a_declaration_is_written_once_and_never_rewritten(tmp_path: Path) -> None:
    """K21: declaring again is allowed only byte for byte; different content is refused and the file is untouched."""
    roots = _roots(tmp_path)
    path = Path(roots.run_root) / "declaration_probe.json"
    cr._write_declaration(path, {"a": 1}, roots)
    before = path.read_bytes()
    assert cr._write_declaration(path, {"a": 1}, roots) == path.resolve()
    with pytest.raises(FileExistsError, match="not rewritten"):
        cr._write_declaration(path, {"a": 2}, roots)
    assert path.read_bytes() == before
