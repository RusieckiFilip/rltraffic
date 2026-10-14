"""G5-fix: check the regenerated packet tables (``g4_tables.md``) against the committed artifact by a route that shares no
code with ``g4_tables.py``: (1) every statement entry whose ``numbers`` hold a non-null ``seed_reversal_qualifier`` at any
depth shows that qualifier's seeds in its own cell of the statement table (``BRIEF_45`` Amendment D, D4.2), and no other
cell shows "reverses on"; (2) every entry of every ``leaders`` block's ``leader_minus_runner_up`` is one row of the leaders
table, with its interval and its qualifier (D3.5, D4.2); (3) every ``never_scored`` row whose statement carries a qualifier
shows its seeds.  Prints one line per check and exits non-zero on the first failure.

    python -P docs/returns/P5.2c_evidence/g5fix_tables_check.py docs/data/p5_2_statements.json \
        docs/returns/P5.2c_evidence/g4_tables.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

QUALIFIER = re.compile(r"reverses on (\d+) of 5 training seeds \(([^)]*)\)")


def qualifiers(node: Any) -> set[str]:
    """The seed lists of every non-null qualifier under ``node`` (e.g. {"303, 505"})."""
    if isinstance(node, dict):
        found = {QUALIFIER.search(v).group(2) for k, v in node.items() if k == "seed_reversal_qualifier" and v}
        return found.union(*(qualifiers(v) for k, v in node.items() if k != "seed_reversal_qualifier"))
    if isinstance(node, list):
        return set().union(*(qualifiers(v) for v in node))
    return set()


def section(text: str, name: str) -> list[str]:
    body = text.split(f"<!-- {name} -->\n", 1)[1].split("\n<!-- ", 1)[0]
    return [line for line in body.splitlines() if line.startswith("| ") and not line.startswith("|---")]


def cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split(" | ")]


def main() -> int:
    artifact = json.loads(Path(sys.argv[1]).read_bytes())
    text = Path(sys.argv[2]).read_text(encoding="utf-8")

    # (1) the statement table: row key (id + arm, tier + version label) -> {definition column: cell text}
    rows = section(text, "STATEMENTS")[1:]
    table: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        c = cells(row)
        table[(c[0], c[1])] = {"att_engine": c[3], "att_ours": c[4]}
    expected, shown = 0, 0
    for s in artifact["statements"]:
        label = s["id"] + (f" `{s['arm']}`" if s.get("arm") else "")
        tier = s["tier"] or "—"
        if s["version"] != "both":
            tier += " (corrected IQL)" if s["version"] == "corrected" else " (as committed IQL)"
        cell = table[(label, tier)][s["definition"]]
        seeds = qualifiers(s["numbers"])
        for seed_list in seeds:
            expected += 1
            assert f"({seed_list})" in cell or f"reverses on seed {seed_list}" in cell, (label, tier, s["definition"])
        if not seeds:
            assert "reverses on" not in cell or "reverses on seed none" in cell, (label, tier, s["definition"], cell)
        shown += sum(1 for _ in re.finditer(r"reverses on", cell))
    print(f"(1) statement table: {len(table)} rows; {expected} qualified (entry, seed list) pairs in the artifact, "
          f"each shown in its own cell; {shown} cells say 'reverses on'; no unqualified cell does")
    assert expected == shown, (expected, shown)

    # (2) the leaders table: one row per entry, by (tier, definition, version, leader, runner-up)
    lead_rows = [cells(row) for row in section(text, "LEADERS")[1:]]
    n_entries = 0
    for b in artifact["leaders"]:
        for lead in b["leader_minus_runner_up"]:
            n_entries += 1
            left, right = lead["left_arm"].split("@")[0], lead["right_arm"].split("@")[0]
            match = [r for r in lead_rows if r[0] == b["tier"] and r[1].startswith(f"`{b['definition']}`")
                     and r[2] == b["version"] and (r[3], r[4]) == (left, right)]
            assert len(match) == 1, (b["tier"], b["definition"], b["version"], left, right, len(match))
            mean = lead["mean_difference"]
            assert f"{mean:+.4f}" in match[0][5], (match[0][5], mean)
            seeds = qualifiers(lead)
            assert (match[0][6] == "—") if not seeds else all(f"({s})" in match[0][6] for s in seeds), match[0]
    empty = sum(1 for b in artifact["leaders"] if not b["leader_minus_runner_up"])
    print(f"(2) leaders table: {len(lead_rows)} rows for {n_entries} entries in {len(artifact['leaders'])} blocks "
          f"({empty} without a contrast); every entry is one row with its own mean and qualifier")
    assert len(lead_rows) == n_entries + empty, (len(lead_rows), n_entries, empty)

    # (3) the first-time table: rows whose statement carries a qualifier show its seeds
    first = [cells(row) for row in section(text, "FIRST-TIME SCORES")[1:]]
    n_first, n_qualified = 0, 0
    for block in artifact["comparison_with_the_record"]["never_scored"]:
        for e in block["this_artifact"]:
            n_first += 1
            s = next(s for s in artifact["statements"] if (s["id"], s["tier"], s.get("arm"), s["definition"],
                                                            s["version"]) == (e["id"], e["tier"], e.get("arm"),
                                                                              e["definition"], e["version"]))
            match = [r for r in first if (r[0], r[1], r[2]) == (e["id"], e["tier"] or "—", e["arm"] or "—")
                     and r[3].startswith(f"`{e['definition']}`")]
            assert len(match) == 1, (e["id"], e["tier"], e["definition"], len(match))
            seeds = qualifiers(s["numbers"])
            n_qualified += bool(seeds)
            assert (match[0][6] == "—") if not seeds else all(f"({x})" in match[0][6] for x in seeds), match[0]
    print(f"(3) first-time table: {len(first)} rows for {n_first} never_scored entries; {n_qualified} carry a "
          f"qualifier, each shown")
    assert len(first) == n_first
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
