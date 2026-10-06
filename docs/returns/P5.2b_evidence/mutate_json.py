"""JSON-level mutants of docs/data/p8_2_compute.json, rewritten in write_artifact's own format."""
import json
import sys
from pathlib import Path

path, which = Path(sys.argv[1]), sys.argv[2]
before = path.read_text(encoding="utf-8")
artifact = json.loads(before)
assert json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n" == before, "format would drift"
variability = artifact["latency_variability"]
if which == "M2-2a":
    old = variability["summary"]["same_computation_spread_max"]["hz1x1"]
    assert old == 0.06584946650960855, old
    variability["summary"]["same_computation_spread_max"]["hz1x1"] = 0.05584946650960855
    said = artifact["what_this_does_not_say"]
    hits = [i for i, text in enumerate(said) if "6.6% on hz1x1" in text]
    assert len(hits) == 1, hits
    said[hits[0]] = said[hits[0]].replace("6.6% on hz1x1", "5.6% on hz1x1")
elif which == "M2-2b":
    groups = variability["groups"]
    hits = [i for i, g in enumerate(groups)
            if g["device"] == "cuda" and set(g["rows"]) == {"grid4x4.bc", "grid4x4.bc_top10", "grid4x4.bc_top10_perix"}]
    assert len(hits) == 1, hits
    del groups[hits[0]]
else:
    sys.exit(f"unknown mutant {which}")
path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
print(f"{which}: {path} rewritten")
