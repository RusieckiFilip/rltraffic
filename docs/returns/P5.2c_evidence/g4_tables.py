"""Render the Return Packet's tables (``docs/returns/P5.2c.md``) from the committed artifact
``docs/data/p5_2_statements.json`` and nothing else: no ``offline/`` import, no recomputation.  Every number printed is the
artifact's, formatted to four decimals; every verdict and reading is the artifact's own string.

    python -P docs/returns/P5.2c_evidence/g4_tables.py docs/data/p5_2_statements.json > g4_tables.md

Refuses (AssertionError) if the artifact's ``primary`` flags are not exactly ``definition == "att_engine"`` (Rule R), or if a
statement row cannot be rendered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PRIMARY, SECOND = "att_engine", "att_ours"
TIERS = ("mappo1000", "maxpressure", "fixedtime", "random")


def f4(x: float) -> str:
    return "0.0000" if x == 0 else f"{x:+.4f}"


def interval(n: dict[str, Any]) -> str:
    """The mean and its CI; an exact zero is shown by the artifact's own reading of it, which names its source."""
    if n.get("exactly_zero"):
        return n.get("reading") or "exactly zero"
    return f"{f4(n['mean_difference'])} [{f4(n['ci95_low'])}, {f4(n['ci95_high'])}]"


def cell_text(e: dict[str, Any]) -> str:
    """One statement entry as table text: the verdict (bold) or the kind, then its numbers, from the artifact."""
    n, sid, verdict = e["numbers"], e["id"], e["verdict"]
    head = f"**{verdict}**" if verdict is not None else ""
    if sid in ("Q0.d1", "Q4.d"):
        body = interval(n) if n.get("exactly_zero") else f"{interval(n)} ({n['reading']})"
    elif sid == "Q0b":
        reverses = [s for s, v in n["per_seed"]["by_seed"].items()
                    if (v < 0) != (n["mean_difference"] < 0) and v != 0.0]
        body = (f"I = {interval(n)}; I < 0 on {n['draws_negative']} of {n['n_draws']} draws; reverses on seed "
                f"{', '.join(reverses) if reverses else 'none'}")
    elif sid == "Q0.stop_rule":
        body = f"d4 {f4(n['mean_difference'])} [{f4(n['ci95_low'])}, {f4(n['ci95_high'])}]"
    elif sid == "Q1":
        body = f"{n['n_held']} of {n['n_cells']} (k = {n['threshold']})"
    elif sid == "Q1b":
        body = f"{n['measured']:.4f} vs {n['predicted']} ({100 * n['relative_error']:.2f} %)"
    elif sid == "Q1b.both":
        body = "both cells hold"
    elif sid in ("ranking", "Q2b", "Q2b_hard", "D9", "Q4b", "Q4b.three_rung"):
        body = e["reading"]
    elif sid == "Q2a":
        body = f"{e['reading']} (predicted {n['predicted_first']})"
    elif sid == "Q2b.aggregate":
        o = n["overall"]
        body = (f"tiers {', '.join(f'{t} {v}' for t, v in n['tiers'].items())}; overall {o['n_concordant']}/{o['of']} "
                f"(range {o['range'][0]}..{o['range'][1]}), not scored")
    elif sid == "Q3a":
        tied = f", tied with {', '.join(n['tied_with'])}" if n["tied_with"] else ""
        body = f"rank {n['rank']} of {n['of']} (range {n['rank_range'][0]}..{n['rank_range'][1]}{tied})"
    elif sid == "Q3b":
        body = interval(n)
        sub = n.get("sub_prediction")
        if sub is not None:
            body += f"; sub-prediction {'matched' if sub['matched'] else 'NOT matched'}"
    elif sid == "Q3c":
        c = n["contrast"]
        body = f"{interval(c)}, {e['reading']}; best non-DT {n['best_non_dt']}"
    elif sid in ("Q3b.aggregate", "Q4a.aggregate", "Q5a.aggregate", "Q5b.aggregate", "Q5c.aggregate"):
        body = ", ".join(f"{t} {v}" for t, v in n["tiers"].items())
    elif sid in ("Q0a", "Q4a", "Q5a", "Q5b", "Q5c", "Q6"):
        body = interval(n)
    else:
        raise AssertionError(f"no rendering for {sid}")
    return f"{head} {body}".strip() if head else body


def row_label(e: dict[str, Any]) -> tuple[str, str]:
    tier = e["tier"] or "—"
    if e["version"] != "both":
        tier += " (corrected IQL)" if e["version"] == "corrected" else " (as committed IQL)"
    sid = e["id"] + (f" `{e['arm']}`" if e.get("arm") else "")
    return sid, tier


def statement_table(a: dict[str, Any]) -> list[str]:
    statements = a["statements"]
    for e in statements:
        assert e["primary"] is (e["definition"] == PRIMARY), (e["id"], e["tier"], e["definition"], e["primary"])
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for e in statements:
        key = (e["id"], e["tier"], e.get("arm"), e["version"])
        if key not in keyed:
            keyed[key] = {}
            order.append(key)
        assert e["definition"] not in keyed[key], key
        keyed[key][e["definition"]] = e
    out = ["| statement | tier | kind | `att_engine` — **PRIMARY** (Rule R) | `att_ours` | definition-dependent |",
           "|---|---|---|---|---|---|"]
    for key in order:
        pair = keyed[key]
        assert set(pair) == {PRIMARY, SECOND}, key
        p, s = pair[PRIMARY], pair[SECOND]
        sid, tier = row_label(p)
        dep = "**yes**" if p["definition_dependent"] else "no"
        dep += f" (on its {p['definition_dependent_on']})"
        out.append(f"| {sid} | {tier} | {p['kind']} | {cell_text(p)} | {cell_text(s)} | {dep} |")
    out.append("")
    out.append(f"{len(order)} rows, {len(statements)} statement entries (each row is one entry per definition). "
               "Numbers are the artifact's, at four decimals; ``exactly zero`` means every per-draw difference is 0.")
    return out


def first_time(a: dict[str, Any]) -> list[str]:
    """``comparison_with_the_record.never_scored`` as a table.  That block copies ``mean_difference`` and the CI ends from
    each entry's top-level ``numbers``: an aggregate has none (shown as such), and Q3c keeps its mean only in
    ``numbers.contrast``, so for Q3c the mean is read from the statement entry itself and marked with a dagger."""
    out = ["| statement | tier | arm | definition | verdict | mean [95 % CI] | reading |", "|---|---|---|---|---|---|---|"]
    for block in a["comparison_with_the_record"]["never_scored"]:
        for e in block["this_artifact"]:
            tag = f"`{e['definition']}`" + (" **(primary)**" if e["definition"] == PRIMARY else "")
            mean_value = e.get("mean_difference")
            dagger = ""
            if mean_value is None and e["id"] == "Q3c":
                statement = next(s for s in a["statements"] if (s["id"], s["tier"], s["definition"], s["version"])
                                 == (e["id"], e["tier"], e["definition"], e["version"]))
                mean_value, dagger = statement["numbers"]["contrast"]["mean_difference"], " †"
                assert (statement["numbers"]["contrast"]["ci95_low"], statement["numbers"]["contrast"]["ci95_high"]) == (
                    e["ci95_low"], e["ci95_high"]), e
            if mean_value is None:
                mean = "— (an aggregate of the tiers above)"
            elif mean_value == 0 and e.get("ci95_low") == 0 == e.get("ci95_high"):
                mean = "exactly zero"
            else:
                mean = f"{f4(mean_value)}{dagger} [{f4(e['ci95_low'])}, {f4(e['ci95_high'])}]"
            out.append(f"| {e['id']} | {e['tier'] or '—'} | {e['arm'] or '—'} | {tag} | "
                       f"{e['verdict'] or '— (no verdict)'} | {mean} | {e['reading'] or '—'} |")
    out += ["", "† `never_scored` stores `mean_difference: null` for Q3c (the mean sits in the statement's "
                "`numbers.contrast`); the value shown is the statement's, whose CI ends equal `never_scored`'s (asserted)."]
    return out


def record(a: dict[str, Any]) -> list[str]:
    c = a["comparison_with_the_record"]
    rows = c["literal_records"]
    out = [f"`literal_records`: {len(rows)} rows, {sum(r['equal'] for r in rows)} equal, "
           f"{sum(not r['equal'] for r in rows)} unequal, and every unequal row carries a named exception:", "",
           "| source | value | recorded | this artifact | exception |", "|---|---|---|---|---|"]
    for r in rows:
        if not r["equal"]:
            out.append(f"| `{r['source']}` | `{r['value']}` | {r['recorded']} | {r['this_artifact']} | "
                       f"`{r['exception']}` |")
    out += ["", "The two exceptions, in the artifact's words:", ""]
    for name, text in c["exceptions"].items():
        out.append(f"- **`{name}`**: {text}")
    out += ["", "P8.4b's V4 and V5 (`comparison_with_the_record.p8_4b`):", "",
            "| contrast | definition | equal | difference (mean, low, high) | in ulps of the record | the record's route "
            "reproduces it |", "|---|---|---|---|---|---|"]
    for r in c["p8_4b"]:
        d, u = r["difference"], r.get("difference_in_ulps_of_the_record", {})
        out.append(f"| {r['contrast_id']} | `{r['definition']}` | {r['equal']} | "
                   f"{d['mean_difference']!r}, {d['ci95_low']!r}, {d['ci95_high']!r} | "
                   f"{u.get('mean_difference')}, {u.get('ci95_low')}, {u.get('ci95_high')} | "
                   f"{r['record_route_reproduces']} |")
    out += ["", "Q5 (`comparison_with_the_record.q5`, `BRIEF_45` Amendment A, A3), the three records it answers:", ""]
    for r in c["q5"]:
        out.append(f"- `{r['record']['source']}`: *\"{r['record']['text'].splitlines()[0]}\"*")
    q5 = c["q5"][0]
    for r in c["q5"][1:]:
        assert {k: r[k] for k in ("registered_verdicts", "statement", "per_tier")} == {
            k: q5[k] for k in ("registered_verdicts", "statement", "per_tier")}, "the three Q5 rows disagree"
    out += ["", "| | Q5a | Q5b | Q5c |", "|---|---|---|---|"]
    for definition in (PRIMARY, SECOND):
        v = q5["registered_verdicts"][definition]
        tag = f"`{definition}`" + (" (primary)" if definition == PRIMARY else "")
        out.append(f"| registered verdict, {tag} | {v['Q5a']} | {v['Q5b']} | {v['Q5c']} |")
    fx = q5["fixedtime_registered_beside_measured"]
    out.append("| `fixedtime`: registered value | " + " | ".join(f"{fx[s]['registered']:+.2f}" for s in ("Q5a", "Q5b", "Q5c"))
               + " |")
    for definition in (PRIMARY, SECOND):
        cells = []
        for s in ("Q5a", "Q5b", "Q5c"):
            m = fx[s]["measured"][definition]
            cells.append("exactly zero" if m["exactly_zero"] else
                         f"{f4(m['mean_difference'])} [{f4(m['ci95_low'])}, {f4(m['ci95_high'])}] {m['verdict']}")
        out.append(f"| `fixedtime`: measured, `{definition}` | " + " | ".join(cells) + " |")
    out += ["", f"The artifact's statement: *\"{q5['statement']}\"* In detail: {q5['statement_detail']}. "
                f"{q5['whose_error']}."]
    stop = c["p5_2_stop_rule_record"]
    d = stop["difference"]
    out += ["", f"P5.2's stop-rule record: equal {stop['equal']}; difference {d['mean_difference']!r} (mean), "
                f"{d['ci95_low']!r}, {d['ci95_high']!r}; the record's route reproduces it: "
                f"{stop.get('record_route_reproduces')}.",
            f"P5.1's grid (P1 and P3): equal {c['p5_1_grid']['equal']}. "
            f"P5.2b's artifact: equal {c['p5_2b_artifact']['equal']}, "
            f"{sum(x['fields_compared'] for x in c['p5_2b_artifact']['compared'])} fields compared over "
            f"{len(c['p5_2b_artifact']['compared'])} statement sets."]
    return out


def ties(a: dict[str, Any]) -> list[str]:
    """The ``ties`` block: per tier, definition and version, the equal-level groups, the pairs with identical episodes
    (every listed field equal on all 500) and the pairs equal on some episodes only; then every tie-dependent integer
    whose range is not a point, printed once per tier where both definitions give the same list (asserted)."""
    out = ["| tier | definition | version | equal-level groups | identical-episode pairs (fields equal) | pairs equal on "
           "some episodes only |", "|---|---|---|---|---|---|"]
    moving: dict[tuple[str, str], list[str]] = {}
    for t in a["ties"]:
        groups = "; ".join(" = ".join(g) for g in t["equal_level_groups"]) or "none"
        ident = "; ".join(f"{' ≡ '.join(p['pair'])} ({len(p['identical_fields'])} fields, {p['n_episodes']} episodes)"
                          for p in t["identical_episode_pairs"]) or "none"
        partial = "; ".join(f"{' / '.join(p['pair'])}: {p['n_equal']} of {p['n']}"
                            for p in t["pairs_with_equal_episodes"] if p["n_equal"] < p["n"]) or "none"
        out.append(f"| {t['tier']} | `{t['definition']}` | {t['version']} | {groups} | {ident} | {partial} |")
        rows = []
        for i in t["integers"]:
            if "range" in i and i["range"][0] != i["range"][1]:
                rows.append(f"| {i['name']} | {i['scope']} | {i['under_rule']} | {i['range'][0]}..{i['range'][1]} | "
                            f"{i['orders_enumerated']} |")
            elif "possible" in i and len(i["possible"]) > 1:
                rows.append(f"| {i['name']} | {i['scope']} | {i['under_rule']} | any of {', '.join(i['possible'])} | "
                            f"{i['orders_enumerated']} |")
        key = (t["tier"], t["version"])
        if key in moving:
            assert moving[key] == rows, f"the tie-dependent integers differ between definitions at {key}"
        moving[key] = rows
    out += ["", "Tie-dependent integers whose range is not a point (identical under both definitions, asserted):", ""]
    for (tier, version), rows in moving.items():
        if rows:
            out += [f"`{tier}` ({version}):", "", "| integer | scope | under the rule | range | tie-break orders |",
                    "|---|---|---|---|---|", *rows, ""]
    if not any(moving.values()):
        out.append("none")
    out.append(f"Every other tier: none — every integer there is a point. Rule: {a['ties'][0]['rule']}")
    return out


def changes(a: dict[str, Any]) -> list[str]:
    ch = a["changes"]
    moved = [c for c in ch if c["changed_by_rule_r"] or c["changed_by_the_correction"]]
    out = [f"`changes`: {len(ch)} entries; {sum(c['changed_by_the_correction'] for c in ch)} changed by the IQL "
           f"correction, {sum(c['changed_by_rule_r'] for c in ch)} changed by Rule R.", "",
           "| statement | tier | as committed, `att_ours` | corrected, `att_ours` | corrected, `att_engine` (primary) | "
           "changed by |", "|---|---|---|---|---|---|"]

    def show(x: dict[str, Any]) -> str:
        v = x.get("verdict")
        r = x.get("reading")
        num = (f" {f4(x['mean_difference'])} [{f4(x['ci95_low'])}, {f4(x['ci95_high'])}]"
               if x.get("mean_difference") is not None and "ci95_low" in x else "")
        return f"{v or ''} {r or ''}{num}".strip()

    for c in moved:
        by = ", ".join(n for n, f in (("the correction", c["changed_by_the_correction"]),
                                      ("Rule R", c["changed_by_rule_r"])) if f)
        out.append(f"| {c['id']} | {c['tier']} | {show(c['as_committed_att_ours'])} | {show(c['corrected_att_ours'])} | "
                   f"{show(c['corrected_att_engine'])} | {by} |")
    return out


def correction(a: dict[str, Any]) -> list[str]:
    """What the IQL correction changes: every ``corrected`` / ``as_committed`` pair is asserted to share its verdict and
    reading, and the values that read the random tier's IQL cell are shown in both versions."""
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for e in a["statements"]:
        if e["version"] in ("corrected", "as_committed"):
            pairs.setdefault((e["id"], e["tier"], e["definition"]), {})[e["version"]] = e
    for key, v in pairs.items():
        assert (v["corrected"]["verdict"], v["corrected"]["reading"]) == (
            v["as_committed"]["verdict"], v["as_committed"]["reading"]), key
    out = [f"{len(pairs)} statement entries exist in both versions; all {len(pairs)} have the same verdict and reading in "
           "both (asserted). The values that move:", "",
           "| definition | IQL level at `random` | Q1's `iql@random` cell (held; relative error) | Q3c at `random`, "
           "`dt_nomix − iql` |", "|---|---|---|---|"]
    for definition in (PRIMARY, SECOND):
        v = {version: {s: pairs[(s, t, definition)][version] for s, t in (("ranking", "random"), ("Q1", None),
                                                                          ("Q3c", "random"))}
             for version in ("as_committed", "corrected")}
        level = " → ".join(f"{v[x]['ranking']['numbers']['levels']['iql']:.4f}" for x in ("as_committed", "corrected"))
        cells = [next(c for c in v[x]["Q1"]["numbers"]["cells"] if c["cell"] == ["iql", "random"])
                 for x in ("as_committed", "corrected")]
        q1 = " → ".join(f"{c['measured']:.4f} ({'held' if c['held'] else 'not held'}; {100 * c['relative_error']:.2f} %)"
                        for c in cells)
        q3c = " → ".join(interval(v[x]["Q3c"]["numbers"]["contrast"]) for x in ("as_committed", "corrected"))
        tag = f"`{definition}`" + (" **(primary)**" if definition == PRIMARY else "")
        out.append(f"| {tag} | {level} | {q1} | {q3c} |")
    out += ["", "Each cell reads *as committed → corrected*. The IQL level also enters `Q3a` and `D9` at `random` (their "
                "`levels.iql`); Q2a, Q2b and Q2b's aggregate change in no field."]
    return out


def leaders(a: dict[str, Any]) -> list[str]:
    """The descriptive ``leaders`` block (not registered): leader, runner-up, the paired lead and its reading."""
    out = ["| tier | definition | version | leader | runner-up | leader − runner-up [95 % CI] | reading |",
           "|---|---|---|---|---|---|---|"]
    for b in a["leaders"]:
        lead = b["leader_minus_runner_up"][0]
        tag = f"`{b['definition']}`" + (" **(primary)**" if b["primary"] else "")
        out.append(f"| {b['tier']} | {tag} | {b['version']} | {', '.join(b['leaders'])} | {' = '.join(b['runners_up'])} "
                   f"| {interval(lead)} | {b['reading']} |")
    return out


def not_say(a: dict[str, Any]) -> list[str]:
    return [f"{i}. {item}" for i, item in enumerate(a["what_this_does_not_say"], 1)]


def main() -> int:
    a = json.loads(Path(sys.argv[1]).read_bytes())
    sections = (("STATEMENTS", statement_table), ("FIRST-TIME SCORES", first_time), ("RECORD", record),
                ("TIES", ties), ("CHANGES", changes), ("CORRECTION", correction), ("LEADERS", leaders),
                ("DOES NOT SAY", not_say))
    for name, render in sections:
        print(f"<!-- {name} -->")
        print("\n".join(render(a)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
