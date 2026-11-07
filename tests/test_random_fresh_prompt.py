"""A27(d) / plan Q2: hz1x1 ``random_fresh``'s prompt, typed after collection, re-derived from the collected tier's raw
``.npz`` (``offline/method_tier_grid.py`` ``A27_TIERS["random_fresh"]``).

Format: contract C6 v1.1 episodes (``ix0_local_reward``).  Alignment: one stream per collected episode (one
intersection on hz1x1); the tier's training set is all 200 episodes (``subsample = "none"``, A27(d)), so the declared
rule -- ``target_rtg`` = the largest stream return, ``rtg_scale`` = the larger of |smallest| and |largest| return
(``method_tier_grid.recomputed_target_and_scale``, P4's rule) -- is recomputed here from the raw reward arrays, by
numpy sums, never through the module's stream code.  Skips until the tier is collected; then fails until the typed
commit carries the values.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import offline.method_tier_grid as grid

TIER_DIR = Path(os.environ.get("RLTRAFFIC_CORPUS_V11", "/nonexistent")) / "cf_hz1x1__random_fresh"


@pytest.mark.skipif(not (TIER_DIR / "manifest.json").is_file(),
                    reason="the hz1x1 random_fresh tier is collected at G3's first stage (RLTRAFFIC_CORPUS_V11)")
def test_the_typed_prompt_is_the_declared_rule_on_the_collected_tiers_raw_returns() -> None:
    manifest = json.loads((TIER_DIR / "manifest.json").read_text())
    returns = []
    for entry in manifest["episodes"]:
        with np.load(TIER_DIR / entry["filename"]) as arrays:
            assert len(arrays["ix_ids"]) == 1, "hz1x1 has one intersection"
            returns.append(float(np.sum(np.asarray(arrays["ix0_local_reward"], dtype=np.float64))))
    assert len(returns) == 200
    spec = grid.A27_TIERS["random_fresh"]
    assert not (math.isnan(spec.target_rtg) or math.isnan(spec.rtg_scale)), (
        "plan Q2: the tier is collected, so its declaration's values_to_type must now be typed into A27_TIERS")
    assert spec.target_rtg == max(returns)
    assert spec.rtg_scale == max(abs(min(returns)), abs(max(returns)))
