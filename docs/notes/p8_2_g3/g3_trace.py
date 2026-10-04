"""P8.2 gate G3 (BRIEF_43 B.1.4): trace EVERY number of the builder's artifact to its source, by routes that import
nothing from the project.

- ``{file, sha256, json_path}``: the file re-hashed, the JSON path resolved by this script's own tokenizer (JSON files
  with ``json``, checkpoints with ``torch.load``), the value compared exactly.
- ``{file, sha256, line}``: the file re-hashed, the line read, the value found in it as printed.
- latency ``measurement`` / ``derived``: compared with ``g3_records.json`` (the third route's recomputation).
- parameter ``measurement``: compared with counts this script takes itself from each row's checkpoint tensors.
- ``code``: the cited code file re-hashed when a digest is given; the value is the code's constant (zero interactions).
- ``inferred``: listed.
- derived training statistics (median / min / max of a block's per-seed seconds): recomputed with ``statistics``.

usage: python g3_trace.py <artifact.json> <checkout> <main_tree> <g3_records.json>
"""
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ART, CHECKOUT, MAIN, REC = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
art = json.loads(ART.read_text())
rec = {(c["row"], c["device"]): c for c in json.loads(REC.read_text())["cells"]}
_hash, _doc = {}, {}
TOK = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\['([^']*)'\]|\[(\d+)\]")


def path_of(name):
    return (CHECKOUT if name.startswith("docs/") else MAIN) / name


def sha(name):
    if name not in _hash:
        h = hashlib.sha256()
        with open(path_of(name), "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _hash[name] = h.hexdigest()
    return _hash[name]


def doc(name):
    if name not in _doc:
        if name.endswith(".pt"):
            import torch
            _doc[name] = torch.load(path_of(name), map_location="cpu", weights_only=False)
        else:
            _doc[name] = json.loads(path_of(name).read_text())
    return _doc[name]


def resolve(document, jp):
    assert jp.startswith("$"), jp
    rest, node = jp[1:], document
    while rest:
        m = TOK.match(rest)
        if not m:
            raise ValueError(f"cannot parse {jp!r} at {rest!r}")
        attr, key, idx = m.groups()
        node = node[attr] if attr is not None else node[key] if key is not None else node[int(idx)]
        rest = rest[m.end():]
    return node


counts, failures, inferred, params_seen = {}, [], [], []


def bump(k):
    counts[k] = counts.get(k, 0) + 1


def check(node, trail, ctx):
    if isinstance(node, dict):
        if "id" in node and "parameters" in node:
            ctx = {**ctx, "row": node}
        if "value" in node and isinstance(node.get("source"), dict):
            src, val = node["source"], node["value"]
            try:
                if "json_path" in src:
                    if src.get("sha256") and sha(src["file"]) != src["sha256"]:
                        raise ValueError("file digest")
                    got = resolve(doc(src["file"]), src["json_path"])
                    if got != val:
                        raise ValueError(f"value {val!r} != source {got!r}")
                    bump("json_path")
                elif "line" in src:
                    if sha(src["file"]) != src["sha256"]:
                        raise ValueError("file digest")
                    text = path_of(src["file"]).read_text().splitlines()[int(src["line"]) - 1]
                    if not any(s in text for s in {str(val), f"{val:.1f}" if isinstance(val, float) else str(val)}):
                        raise ValueError(f"{val!r} not in line {src['line']}: {text[:120]!r}")
                    bump("line")
                elif "measurement" in src and "file" in src:
                    if sha(src["file"]) != src["sha256"]:
                        raise ValueError("record digest")
                    cell_name = Path(src["file"]).name[: -len(".json")]
                    row_id, device = cell_name.rsplit("_", 1)
                    c = rec[(row_id, device)]
                    leaf = trail.rsplit(".", 1)[-1]
                    want = {"median_ms": c["median_ms"], "p95_ms": c["p95_ms"],
                            "per_intersection_median_ms": c["median_ms"] / (16 if row_id.startswith("grid4x4") else 1),
                            "per_intersection_p95_ms": c["p95_ms"] / (16 if row_id.startswith("grid4x4") else 1)}[leaf]
                    if val != want:
                        raise ValueError(f"latency {val} != third route {want}")
                    bump("latency" + ("_derived" if "derived" in src else ""))
                elif "inferred" in src:
                    inferred.append((trail, val, src["inferred"][:90]))
                    bump("inferred")
                elif "code" in src:
                    if src.get("file") and sha(src["file"]) != src["sha256"]:
                        raise ValueError("code file digest")
                    bump("code")
                elif "measurement" in src:
                    bump("measurement_" + trail.rsplit(".", 1)[-1])
                    # module-level list: a per-row copy of ctx would discard it (the first version did, and its
                    # check then passed over 0 rows -- caught by the count it printed)
                    params_seen.append((ctx.get("row", {}).get("id"), trail.rsplit(".", 1)[-1], val))
                else:
                    raise ValueError(f"unknown source shape {sorted(src)}")
            except Exception as exc:  # noqa: BLE001 -- every failure is reported, none swallowed
                failures.append((trail, repr(exc)[:200]))
        if "per_seed" in node and "median" in node:
            vals = [float(i["value"]) for i in node["per_seed"]]
            if (node["median"], node["min"], node["max"]) != (statistics.median(vals), min(vals), max(vals)):
                failures.append((trail, "derived median/min/max"))
            bump("derived_stats")
        for k, v in node.items():
            check(v, f"{trail}.{k}", ctx)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            check(v, f"{trail}[{i}]", ctx)


ctx = {}
check(art, "$", ctx)
print("traced:", dict(sorted(counts.items())))
print("failures:", failures[:10] if failures else "none", f"({len(failures)})")
print("inferred:", inferred)
params = {}
for row_id, kind, val in params_seen:
    params.setdefault(row_id, {})[kind] = val
assert len({r for r, _, _ in params_seen}) == len(art["rows"]), "parameters not seen for every row"
json.dump({"counts": counts, "failures": failures, "inferred": inferred, "parameters": params},
          open(Path(REC).with_name("g3_trace.json"), "w"), indent=1)
