"""P5.2c G3: comparison_with_the_record.literal_records, second pass.  g3_provenance.py's first pass compared raw strings
and flagged 6 literals as not found (they span a line break, a blockquote marker or markdown bold in the source) and 20
'equal' flags as inconsistent (the record writes U+2212, the artifact '-').  Here both sides are normalised -- markdown
bold removed, blockquote markers stripped at line starts, whitespace runs collapsed, U+2212 read as '-' -- and every
row is re-checked.  Nothing from offline/ is imported."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/home/filip/rltraffic-p52c")
A = json.loads((REPO / "docs/data/p5_2_statements.json").read_text(encoding="utf-8"))
lit = A["comparison_with_the_record"]["literal_records"]


def norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).replace("−", "-").replace("**", "")).strip()


not_found, inconsistent, unequal = [], [], []
for r in lit:
    m = re.match(r"(.+?):(\d+)(?:-(\d+))?$", r["source"])
    lines = (REPO / m.group(1)).read_text(encoding="utf-8").splitlines()
    lo, hi = int(m.group(2)), int(m.group(3) or m.group(2))
    joined = " ".join(re.sub(r"^\s*(>\s*)+", "", line).strip() for line in lines[lo - 1:hi])
    if norm(r["recorded"]) not in norm(joined):
        not_found.append((r["value"], r["source"]))
    if r["equal"] != (norm(r["recorded"]) == norm(r["this_artifact"])):
        inconsistent.append((r["value"], r["recorded"], r["this_artifact"]))
    if not r["equal"]:
        unequal.append((r["value"], r["source"], r["recorded"], r["this_artifact"], r["exception"]))
print(f"{len(lit)} literal rows; recorded literal not found at its cited lines: {len(not_found)} {not_found}")
print(f"'equal' inconsistent with the normalised strings: {len(inconsistent)} {inconsistent}")
print(f"rows not equal: {len(unequal)}; exceptions named: {sorted({u[4] for u in unequal})}")
for u in unequal:
    print("  ", u)
