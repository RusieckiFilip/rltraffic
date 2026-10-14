"""P5.2c G3: the artifact's provenance and its comparison with the record, checked by the coordinator from disk.

1. Every provenance.inputs file's sha256 recomputed (paths under the repo or under output/).
2. provenance.p8_4b_rows and p5_2b_corrected_rows: the digest recomputed by its stated construction (sha256 over
   '<file sha256>  <path relative to output>\\n' in path order) over the files this script itself enumerates: every
   P8.4b row of the 30 grid4x4 cells the artifact reads, and P5.2b's 500 corrected rows.
3. comparison_with_the_record.literal_records: each row's 'recorded' literal found at its cited file:line(s), and
   'equal' consistent with recorded == this_artifact; the exceptions named.
Nothing from offline/ is imported."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path("/home/filip/rltraffic-p52c")
O = Path("/home/filip/rltraffic/output")
A = json.loads((REPO / "docs/data/p5_2_statements.json").read_text(encoding="utf-8"))
P = A["provenance"]
bad = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 1. inputs
for item in P["inputs"]:
    f = item["file"]
    path = O / f[len("output/"):] if f.startswith("output/") else REPO / f
    ok = path.exists() and sha(path) == item["sha256"]
    if not ok:
        bad.append(("input", f))
print(f"inputs: {len(P['inputs'])} files, mismatches: {[b for b in bad if b[0] == 'input']}")

# 2. row digests
cells = sorted(A["cells"])
rows = []
for cid in cells:
    src, rest = cid.split(":")
    method, tier = rest.split("@")
    if src == "p5_2b":
        continue
    prefix = f"cell_grid4x4_{method}_at_{tier}"
    for s in (101, 202, 303, 404, 505):
        for d in range(1000, 1100):
            rows.append(f"p8_4b_rederivation/{prefix}_seed{s}_draw{d}.json")
rows = sorted(set(rows))
text = "".join(f"{sha(O / r)}  {r}\n" for r in rows)
mine = hashlib.sha256(text.encode()).hexdigest()
print(f"p8_4b_rows: n mine {len(rows)} vs {P['p8_4b_rows']['n']}; digest equal: {mine == P['p8_4b_rows']['sha256']}")
if mine != P["p8_4b_rows"]["sha256"] or len(rows) != P["p8_4b_rows"]["n"]:
    bad.append(("p8_4b_rows", mine))
rows2 = sorted(f"p5_2b/rederivation/cell_grid4x4_iql_at_random_seed{s}_draw{d}.json"
               for s in (101, 202, 303, 404, 505) for d in range(1000, 1100))
text2 = "".join(f"{sha(O / r)}  {r}\n" for r in rows2)
mine2 = hashlib.sha256(text2.encode()).hexdigest()
print(f"p5_2b_corrected_rows: n {len(rows2)} vs {P['p5_2b_corrected_rows']['n']}; digest equal: "
      f"{mine2 == P['p5_2b_corrected_rows']['sha256']}")
if mine2 != P["p5_2b_corrected_rows"]["sha256"]:
    bad.append(("p5_2b_corrected_rows", mine2))

# 3. literal records
lit = A["comparison_with_the_record"]["literal_records"]
not_found, inconsistent, exceptions = [], [], []
for r in lit:
    m = re.match(r"(.+?):(\d+)(?:-(\d+))?$", r["source"])
    assert m, r["source"]
    path = REPO / m.group(1)
    lines = path.read_text(encoding="utf-8").splitlines()
    lo, hi = int(m.group(2)), int(m.group(3) or m.group(2))
    window = " ".join(lines[lo - 1:hi])
    if str(r["recorded"]) not in window:
        not_found.append((r["value"], r["source"], r["recorded"]))
    if r["equal"] != (r["recorded"] == r["this_artifact"]):
        inconsistent.append((r["value"], r["recorded"], r["this_artifact"], r["equal"]))
    if r["exception"] is not None or not r["equal"]:
        exceptions.append((r["value"], r["source"], r["recorded"], r["this_artifact"], r["exception"]))
print(f"literal_records: {len(lit)} rows; recorded literal not found at its cited line(s): {len(not_found)}; "
      f"'equal' inconsistent: {len(inconsistent)}")
for x in not_found:
    print("  NOT FOUND", x)
for x in inconsistent:
    print("  INCONSISTENT", x)
print("rows that are not equal or carry an exception:")
for x in exceptions:
    print("  ", x)
print("bad:", bad)
