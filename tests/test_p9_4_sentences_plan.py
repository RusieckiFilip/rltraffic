"""P9.4 plan section 6 (``docs/plans/p9.4_mix50_fresh.md``): every branch of every template S0-S13, as the report writes
it, equals the plan's binding text with its slots filled.

The templates are READ FROM THE PLAN FILE -- the lines between its headings 6 and 7, with the replacements S2, S4, S6 and
S7b state applied as they state them -- so the code cannot drift from the text it is bound to.  Review C (2026-11-19)
found that an S13 branch rewritten into its opposite passed the earlier prefix and suffix checks; every comparison here
is of the WHOLE sentence, under ``==``.  The slots are filled with the report's own formatters, which
``tests/test_p9_4_report_review.py`` pins against typed-out strings.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import offline.p9_4_report as rep

PLAN = Path(__file__).resolve().parent.parent / "docs" / "plans" / "p9.4_mix50_fresh.md"
DIFFERENCE_BRANCHES = {"CI below 0": "below", "CI contains 0": "contains", "CI above 0": "above"}
RHO_BRANCHES = {"CI above 1": "above_1", "CI contains 1 and lies above 0": "contains_1_above_0",
                "CI inside (0, 1)": "inside_0_1", "CI contains 0 and lies below 1": "contains_0_below_1",
                "CI contains 0 and 1": "contains_0_and_1", "CI below 0": "below_0"}


def _section_6() -> list[str]:
    lines = PLAN.read_text(encoding="utf-8").splitlines()
    return lines[lines.index("## 6. Sentences for the paper, one per outcome") + 1:lines.index("## 7. Rules for the runs")]


def _parse() -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, str]]]:
    """Per template: its header line after the name, the quoted strings on it, and its ``- <branch>: "<text>"`` items."""
    headers: dict[str, str] = {}
    quoted: dict[str, list[str]] = {}
    branches: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in _section_6():
        header = re.match(r"^(S\d+b?), (.*)$", line)
        if header is not None:
            current = header.group(1)
            headers[current] = header.group(2)
            quoted[current] = re.findall(r'"([^"]*)"', line)
            branches[current] = {}
            continue
        item = re.match(r'^- ([^:]+): "(.*)"$', line)
        if item is not None and current is not None:
            branches[current][item.group(1)] = item.group(2)
    return headers, quoted, branches


HEADERS, QUOTED, BRANCHES = _parse()


def _derived(base: str, name: str) -> dict[str, str]:
    """S2, S4 and S6: the base template's branches with the one replacement the plan states for *name*."""
    if name == "S4":
        found = re.match(r"^DT − (.*?): S3's three branches, with that name in place of ", HEADERS["S4"])
        assert found is not None, HEADERS["S4"]
        new, (old,) = found.group(1), QUOTED["S4"]
    else:
        new, old = QUOTED[name]
    out = {}
    for branch, text in BRANCHES[base].items():
        assert old in text, (name, branch)
        out[branch] = text.replace(old, new)
    return out


def _fill(template: str, **slots: str) -> str:
    for key, value in slots.items():
        template = template.replace("{" + key + "}", value)
    assert "{" not in template and "}" not in template, template
    return template


def _paired(mean: float, low: float, high: float, wins: int, losses: int, p: float) -> dict[str, Any]:
    return {"mean_difference": mean, "ci95_low": low, "ci95_high": high, "wins": wins, "losses": losses,
            "ties": 100 - wins - losses, "wilcoxon_p": p}


#: One block per branch, with numbers that need more than the default decimals and p-values of every printed form.
DIFFERENCE_CASES = {"CI below 0": _paired(-20.04, -30.3, -9.87, 83, 17, 3.9e-18),
                    "CI contains 0": _paired(-2.0, -5.15, 1.0, 55, 45, 0.0496),
                    "CI above 0": _paired(12.0, 4.0, 20.25, 31, 69, 0.0123)}
RHO_CASES = {"CI above 1": (1.3, 1.1, 1.5), "CI contains 1 and lies above 0": (0.95, 0.85, 1.05),
             "CI inside (0, 1)": (0.5, 0.3, 0.7), "CI contains 0 and lies below 1": (0.05, -0.1, 0.2),
             "CI contains 0 and 1": (0.5, -0.2, 1.2), "CI below 0": (-0.3, -0.5, -0.1)}


def _slots(block: dict[str, Any]) -> dict[str, str]:
    d, lo, hi = rep.fmt_estimate(block["mean_difference"], block["ci95_low"], block["ci95_high"], unit="s")
    return {"d": d, "lo": lo, "hi": hi, "w": str(block["wins"]), "l": str(block["losses"]),
            "p": rep.fmt_p(block["wilcoxon_p"])}


def test_the_plan_parses_into_every_template_and_every_branch() -> None:
    assert set(HEADERS) == {f"S{n}" for n in range(14)} | {"S12b"}  # S7b is stated inside S7's line
    for name in ("S1", "S3", "S8", "S9", "S12", "S13"):
        assert set(BRANCHES[name]) == set(DIFFERENCE_BRANCHES), name
    assert set(BRANCHES["S5"]) == set(RHO_BRANCHES)
    assert set(BRANCHES["S10"]) == {"all the same", "otherwise"}
    for name in ("S0", "S11", "S12b", "S7"):
        assert QUOTED[name], name
    assert "S7b: the same sentence for the aligned anchors." in HEADERS["S7"]


DIFFERENCE_SENTENCES: dict[str, Callable[[dict[str, Any]], dict[str, str]]] = {
    "S1": rep.sentence_s1, "S2": rep.sentence_s2, "S3": rep.sentence_s3, "S4": rep.sentence_s4,
    "S8": rep.sentence_s8, "S9": rep.sentence_s9, "S12": rep.sentence_s12, "S13": rep.sentence_s13,
}
BASES = {"S2": "S1", "S4": "S3"}


@pytest.mark.parametrize("template", sorted(DIFFERENCE_SENTENCES))
@pytest.mark.parametrize("branch", sorted(DIFFERENCE_BRANCHES))
def test_every_branch_of_a_difference_template_is_the_plans_whole_sentence(template: str, branch: str) -> None:
    texts = _derived(BASES[template], template) if template in BASES else BRANCHES[template]
    block = DIFFERENCE_CASES[branch]
    expected = {"template": template, "branch": DIFFERENCE_BRANCHES[branch], "text": _fill(texts[branch], **_slots(block))}
    assert DIFFERENCE_SENTENCES[template](block) == expected


@pytest.mark.parametrize("template", ["S5", "S6"])
@pytest.mark.parametrize("branch", sorted(RHO_BRANCHES))
def test_every_branch_of_s5_and_s6_is_the_plans_whole_sentence(template: str, branch: str) -> None:
    texts = BRANCHES["S5"] if template == "S5" else _derived("S5", "S6")
    mean, low, high = RHO_CASES[branch]
    r, lo, hi = rep.fmt_estimate(mean, low, high, unit="rho")
    produce = rep.sentence_s5 if template == "S5" else rep.sentence_s6
    assert produce({"mean": mean, "ci95_low": low, "ci95_high": high}) == {
        "template": template, "branch": RHO_BRANCHES[branch], "text": _fill(texts[branch], r=r, lo=lo, hi=hi)}


@pytest.mark.parametrize("anchors", ["registered", "aligned"])
@pytest.mark.parametrize("interval", [(0.3, 0.7), (1.1, 1.5), (-0.1, 0.2), (-0.2, 1.2)])
def test_s7_and_s7b_are_the_plans_whole_sentence(anchors: str, interval: tuple[float, float]) -> None:
    low, high = interval
    better, worse = low > 0.0, high < 1.0
    text = (QUOTED["S7"][0].replace("{is | is not} better", ("is" if better else "is not") + " better", 1)
            .replace("{is | is not} worse", ("is" if worse else "is not") + " worse", 1))
    if anchors == "aligned":
        text = text.replace("under the registered anchors", "under the aligned anchors")
    assert "{" not in text
    produced = rep.sentence_s7({"mean": (low + high) / 2.0, "ci95_low": low, "ci95_high": high}, anchors=anchors)
    assert (produced["template"], produced["text"]) == ("S7" if anchors == "registered" else "S7b", text)


def test_s0_s11_and_s12b_are_the_plans_whole_sentences() -> None:
    assert rep.sentence_s0()["text"] == QUOTED["S0"][0]
    m, lo, hi = rep.fmt_estimate(105.33, 103.0, 107.7, unit="s")
    produced = rep.sentence_s11({"mean": 105.33, "ci95_low": 103.0, "ci95_high": 107.7}, never_entered=0.04)
    assert produced["text"] == _fill(QUOTED["S11"][0], m=m, lo=lo, hi=hi, n=rep.fmt_number(0.04, unit="n"))
    engine, ours = _paired(-1.5, -2.5, -0.5, 70, 30, 0.01), _paired(-1.0, -2.0, 0.1, 60, 40, 0.2)
    d, lo, hi = rep.fmt_estimate(-1.0, -2.0, 0.1, unit="s")
    produced_b = rep.sentence_s12b(engine, ours)
    assert produced_b is not None and produced_b["text"] == _fill(QUOTED["S12b"][0], d=d, lo=lo, hi=hi)


def test_s10_is_the_plans_whole_sentence_in_both_branches() -> None:
    same = {"S1": "below", "S3": "above", "S5": "inside_0_1", "S6": "contains_1_above_0"}
    assert rep.sentence_s10(same, dict(same))["text"] == BRANCHES["S10"]["all the same"]
    expected = (BRANCHES["S10"]["otherwise"]
                .replace("{the readings that differ}", "the comparison with the registered MaxPressure")
                .replace("{the original model's branch}", "slower than the registered MaxPressure")
                .replace("{the corrected model's branch}", "not measurably different from the registered MaxPressure"))
    assert rep.sentence_s10({**same, "S3": "contains"}, same)["text"] == expected
