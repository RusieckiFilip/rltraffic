"""P9.4 plan 2.4 (``docs/plans/p9.4_mix50_fresh.md``; Amendment A1): the committed declaration is the pinned one, its
checks all passed, and -- on the corpus -- its target, return scale and file digests re-derive from the raw ``.npz`` by
a route independent of the code that wrote it (``np.load`` and ``sum(ix<i>_local_reward)`` per stream, ``hashlib`` per
file), never by calling ``build_declaration``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import offline.p9_4_mix50_fresh as p94

REPO_ROOT = Path(__file__).resolve().parent.parent
DECLARATION = REPO_ROOT / "docs" / "data" / "p9_4_declaration.json"


@pytest.fixture(scope="module")
def declaration() -> dict[str, Any]:
    return json.loads(DECLARATION.read_bytes())


def test_the_committed_declaration_is_the_pinned_one_and_loads(declaration: dict[str, Any]) -> None:
    assert hashlib.sha256(DECLARATION.read_bytes()).hexdigest() == p94.DECLARATION_SHA256
    roots = p94.Roots(repo_root=REPO_ROOT, output_root=REPO_ROOT, corpus_root=REPO_ROOT, draws_root=REPO_ROOT)
    assert p94.load_declaration(roots) == declaration


def test_every_check_passed_and_the_plans_fixed_values_hold(declaration: dict[str, Any]) -> None:
    assert {name: check["passed"] for name, check in declaration["checks"].items()} == dict.fromkeys(
        ("i", "ii", "iii", "iv", "v"), True)
    assert (declaration["target_rtg"], declaration["prompts"]["naive"]["target_rtg"],
            declaration["prompts"]["b_mean_k100"]["target_rtg"]) == (-5959.0, -5959.0, -7431.0185327454665)
    assert len(declaration["streams"]) == 200
    assert sorted({row["half"] for row in declaration["streams"]}) == ["expert", "random"]
    assert declaration["changes_besides_the_generator"]["normalisation_basis"]["mix50_split_streams"] == 600
    assert declaration["recipe"]["gradient_steps"] == 40000 and declaration["seeds"] == [101, 202, 303, 404, 505]


@pytest.fixture(scope="module")
def corpus_root() -> Path:
    env_value = os.environ.get("RLTRAFFIC_CORPUS_V11")
    candidate = Path(env_value) if env_value else REPO_ROOT / "datasets_v11"
    if not (candidate / "cf_hz1x1__random_fresh" / "manifest.json").is_file():
        pytest.skip(f"format v1.1 corpus with random_fresh not found at {candidate}: set RLTRAFFIC_CORPUS_V11")
    return candidate


def test_on_the_corpus_the_declared_streams_re_derive_from_the_raw_files(declaration: dict[str, Any],
                                                                       corpus_root: Path) -> None:
    returns = []
    for row in declaration["streams"]:
        path = corpus_root / Path(row["dataset_dir"]).name / row["episode_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"]
        with np.load(path) as arrays:
            index = [str(v) for v in arrays["ix_ids"].tolist()].index(row["ix_id"])
            total = float(np.sum(arrays[f"ix{index}_local_reward"], dtype=np.float64))
        assert total == row["total_return"]
        returns.append(total)
    assert max(returns) == declaration["target_rtg"] == -5959.0
    assert max(abs(min(returns)), abs(max(returns))) == declaration["rtg_scale"]
    random_draws = sorted(row["flow_draw"] for row in declaration["streams"] if row["half"] == "random")
    assert len(random_draws) == len(set(random_draws)) == 100
