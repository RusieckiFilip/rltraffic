"""P5.3c C4 (``BRIEF_42`` §3 C4; Amendment E, E3.1-E3.2): the committed artifact ``docs/data/p5_3c_context_sweep.json``.

* **Its shape (ungated):** the file committed by hand at the digest E3.1 states; the format, 7,000 cells, fourteen arms of
  500; the confirmatory family's outcome ``iii`` with nothing rejected, its sentence equal to ``outcome_sentence`` on the
  committed family, the five arms it was fed in K order and the sweep's own K = 20 arm; the limits the module carries.
* **T-regress (gated on the campaign's chunks):** ``build_context_sweep_artifact`` on the REAL chunks, at HEAD, equals the
  committed file byte for byte through ``json.dumps(..., indent=2, sort_keys=True)``, after exactly the plan's A3
  substitutions (``docs/plans/p5.3c.md`` §11): ``code_changed_since -> []`` -- J1(c), because this HEAD differs from the
  chunks' commit outside ``docs/`` by C4's own test and configuration files, never by ``offline/`` -- and the report's
  ``report_code`` pair for the committed artifact's recorded one.
  *Mutations:* one committed number changed; an estimator detail changed -> this dies.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import offline.context_sweep as cs

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "docs" / "data"
ARTIFACT = DATA / "p5_3c_context_sweep.json"
#: Amendment E, E3.1: the artifact the campaign wrote at ``9c4603c``, committed by hand.
ARTIFACT_SHA256 = "bcdca7eabad82d7a99feed7eed9f2ed9bdc37f3950bbf55811dcf94b76c7abed"
ARTIFACT_BYTES = 239_536
CHUNKS_COMMIT = "9c4603ca960e7338da9b8c5c2a3677a1b0185524"
ARMS = ["mappo1000_k1_b64", "mappo1000_k2_b64", "mappo1000_k5_b64", "mappo1000_k10_b64", "mappo1000_k20_b64"]


def test_the_committed_artifact_is_the_campaigns_at_its_stated_digest_and_has_its_registered_shape() -> None:
    raw = ARTIFACT.read_bytes()
    assert (hashlib.sha256(raw).hexdigest(), len(raw)) == (ARTIFACT_SHA256, ARTIFACT_BYTES)
    artifact = json.loads(raw)
    assert artifact["format_version"] == cs.ARTIFACT_FORMAT_VERSION == "p5.3c-context-sweep/1.0"
    assert artifact["n_cells"] == 7000
    arms = sorted([*cs.registered_arms(), "ref_mappo1000_k20", "ref_mix50_k20"])
    assert artifact["counts"] == {arm: 500 for arm in arms} and len(arms) == 14
    confirmatory = artifact["confirmatory"]
    family = confirmatory["family"]
    assert (confirmatory["subject"], confirmatory["definition"]) == ("mappo1000", "att_engine")
    assert family["holm"]["rejected"] == {"T1": False, "T2": False, "T3": False}
    assert confirmatory["outcome"] == family["outcome"] == cs.outcome_of(family["holm"]["rejected"]) == "iii"
    assert confirmatory["tests_not_rejected"] == ["T1", "T2", "T3"]
    assert confirmatory["sentence"] == cs.outcome_sentence("iii", family=family)
    assert (confirmatory["arms"], confirmatory["k20_arm"]) == (ARMS, "mappo1000_k20_b64")
    assert artifact["what_this_does_not_say"] == list(cs._ARTIFACT_LIMITS)
    provenance = artifact["provenance"]
    assert provenance["chunk_code_commits"] == [CHUNKS_COMMIT]
    assert provenance["report_code"] == {"code_commit": CHUNKS_COMMIT, "code_dirty": False}


_OUTPUT = os.environ.get("RLTRAFFIC_OUTPUT_ROOT")


def _chunks_available() -> str | None:
    if not _OUTPUT:
        return "needs RLTRAFFIC_OUTPUT_ROOT: output/p5_3c/cells/canary.json and the campaign's 7,000 chunks (gitignored)"
    canary = Path(_OUTPUT) / "p5_3c" / "cells" / "canary.json"
    if not canary.is_file():
        return f"{canary} is absent: the campaign's chunks are not on this machine"
    return None


@pytest.mark.skipif(_chunks_available() is not None, reason=str(_chunks_available()))
def test_the_committed_artifact_regenerates_byte_for_byte_from_the_campaigns_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-regress: the REAL report over the REAL work directory, with exactly A3's two substitutions."""
    committed: dict[str, Any] = json.loads(ARTIFACT.read_bytes())
    recorded = dict(committed["provenance"]["report_code"])
    monkeypatch.setattr(cs, "code_changed_since", lambda commit: [], raising=True)
    monkeypatch.setattr(cs, "_code_provenance", lambda: dict(recorded), raising=True)
    output = Path(str(_OUTPUT))
    regenerated = cs.build_context_sweep_artifact(
        output_root=output, corpus_root=output.parent / "datasets_v11", draws_root=output.parent / "scenarios" / "draws",
        data_dir=DATA,
    )
    assert (json.dumps(regenerated, indent=2, sort_keys=True) + "\n").encode("utf-8") == ARTIFACT.read_bytes()
