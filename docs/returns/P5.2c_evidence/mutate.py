"""P5.2c's mutation run (``BRIEF_45`` section 5 and Amendment A, A4.4): every named mutant executed, its output logged.

Run on an EXPORTED COPY of the committed tree (``git archive <commit> | tar -x``), never on the working tree: each mutant
is an exact string replacement in the copy, asserted to match the stated number of times, the named tests run against
the copy with ``PYTHONPATH=<copy>``, and the copy's files restored before the next mutant.  An unmutated control run of
every killing test comes first and must pass, so a KILLED verdict means the mutation, not the setup, failed the tests.

    python docs/returns/P5.2c_evidence/mutate.py --repo . --commit HEAD --work <scratch dir> \\
        --output-root /home/filip/rltraffic/output --log <log> --results <json>

Exit 0 iff the control passes and every mutant is KILLED.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

MODULE = "offline/tier_statements.py"
TESTS = "tests/test_tier_statements.py"


@dataclass(frozen=True)
class Mutant:
    """One mutation: ``(file, old, new, count)`` replacements and the tests expected to kill it."""

    mutant_id: str
    description: str
    replacements: tuple[tuple[str, str, str, int], ...]
    tests: tuple[str, ...]
    source: str = field(default="BRIEF_45 section 5")


def t(name: str) -> str:
    return f"{TESTS}::{name}"


MUTANTS: tuple[Mutant, ...] = (
    Mutant("M1", "the tie rule disabled: measured_levels dropped from every concordance call",
           ((MODULE, "measured_levels=measured)", "measured_levels=None)", 2),
            (MODULE, "measured_levels=levels)", "measured_levels=None)", 3)),
           (t("test_t_ties_two_exact_ties_every_integer_and_its_range_by_enumeration"),
            t("test_t_reproduce_record_every_value_p5_2s_record_states_at_its_precision"))),
    Mutant("M2", "episode-level instead of per-draw pairing",
           ((MODULE, "    left_means, right_means = _per_draw_means(a), _per_draw_means(b)\n",
             "    left_means = {(e.seed, e.draw_id): e.att_horizon for e in a}\n"
             "    right_means = {(e.seed, e.draw_id): e.att_horizon for e in b}\n", 1),),
           (t("test_t_contrast_episode_level_pairing_would_change_n_and_the_interval"),
            t("test_t_contrast_equals_statistics_on_dyadic_values_where_every_route_is_exact"))),
    Mutant("M3", "the corrected and original IQL cells swapped",
           ((MODULE, '("iql", "random"): cells[CORRECTED_IQL]},\n            "as_committed": {**common, ("iql", '
                     '"random"): cells[ORIGINAL_IQL]}}',
             '("iql", "random"): cells[ORIGINAL_IQL]},\n            "as_committed": {**common, ("iql", '
             '"random"): cells[CORRECTED_IQL]}}', 1),),
           (t("test_t_versions_the_books_differ_only_at_the_random_tier_iql_cell"),
            t("test_t_versions_the_iql_free_statements_are_stored_once_and_the_iql_ones_twice"),
            t("test_t_reproduce_p5_2b_the_random_tier_on_both_versions_field_by_field"))),
    Mutant("M4", "z = 1.959964 in the interval (dt_gate.mean_ci95, the one place the module's z lives)",
           (("offline/dt_gate.py", "ci95=1.96 * std / math.sqrt(count),", "ci95=1.959964 * std / math.sqrt(count),",
             1),),
           (t("test_t_contrast_equals_statistics_on_dyadic_values_where_every_route_is_exact"),)),
    Mutant("M5", "Q4b scored on the three-rung chain only",
           ((MODULE, '(("Q4b", TIERS), ("Q4b.three_rung"', '(("Q4b", ("mappo1000", "maxpressure", "random")), '
                                                            '("Q4b.three_rung"', 1),),
           (t("test_t_q4b_the_four_rung_chain_and_the_three_rung_bullet_can_disagree"),)),
    Mutant("M6", "Q6 against the behaviour cell instead of the shared random anchor",
           ((MODULE, "    anchor = book[ANCHOR]\n", '    anchor = book[("behaviour", "mappo1000")]\n', 1),),
           (t("test_t_q6_reads_the_shared_anchor_not_the_behaviour_cell"),
            t("test_t_first_time_scores_each_against_this_files_own_recomputation"))),
    Mutant("M7", "Q5b's mappo1000 tier dropped",
           ((MODULE, '("Q5b", ("bc_top10_perix", "bc_top10"), {tier: "negative" for tier in TIERS}),',
             '("Q5b", ("bc_top10_perix", "bc_top10"), {tier: "negative" for tier in OUT_OF_SAMPLE_TIERS}),', 1),),
           (t("test_t_q5b_a_failed_mappo1000_tier_makes_the_aggregate_failed"),)),
    Mutant("M8", "an exact zero folded into HELD",
           ((MODULE, '    if predicted == "negative":\n        return sm._outcome_from_interval(float(low), float(high))\n',
             '    if float(low) == 0.0 == float(high):\n        return "HELD"\n'
             '    if predicted == "negative":\n        return sm._outcome_from_interval(float(low), float(high))\n', 1),),
           (t("test_t_outcome_the_interval_rule_for_both_predicted_signs"),
            t("test_t_contrast_exact_zero_is_ci_zero_not_resolved_sign_zero_and_named"),
            t("test_t_q4a_the_exact_zero_at_fixedtime_is_not_resolved_and_the_aggregate_follows"))),
    Mutant("M9", "definition_dependent always false",
           ((MODULE, 'entry["definition_dependent"] = entry["verdict"] != other["verdict"]',
             'entry["definition_dependent"] = False', 1),
            (MODULE, 'entry["definition_dependent"] = entry["reading"] != other["reading"]',
             'entry["definition_dependent"] = False', 1)),
           (t("test_t_definitions_definition_dependent_is_true_iff_the_two_verdicts_differ"),
            t("test_t_first_time_scores_each_against_this_files_own_recomputation"))),
    Mutant("M10", "the out-path barrier removed",
           ((MODULE, "        out = assert_out_path(args.out, roots, inputs)\n",
             "        out = Path(args.out).resolve()\n", 1),),
           (t("test_t_barrier_refuses_an_out_path_under_the_output_root"),
            t("test_t_barrier_refuses_a_tree_holding_p5_2s_manifest_whatever_the_output_root_says"),
            t("test_t_barrier_refuses_an_input_as_the_out_path"))),
    Mutant("M11", "the per-draw route switched to Python's built-in (compensated) sum (F1)",
           ((MODULE, "    left_means, right_means = _per_draw_means(a), _per_draw_means(b)\n",
             "    left_means = {d: sum(e.att_horizon for e in a if e.draw_id == d) / sum(1 for e in a if e.draw_id == d)"
             " for d in {e.draw_id for e in a}}\n"
             "    right_means = {d: sum(e.att_horizon for e in b if e.draw_id == d) / sum(1 for e in b if e.draw_id == d)"
             " for d in {e.draw_id for e in b}}\n", 1),),
           (t("test_t_contrast_route_pin_is_numpy_per_draw_and_not_the_compensated_sum"),
            t("test_t_reproduce_p5_1_d1_is_p1_and_q6_at_mappo1000_is_p3")), source="plan section 8"),
    Mutant("M12", "the version-identity refusal removed",
           ((MODULE, 'raise ValueError(f"{key} differs between versions', 'print(f"{key} differs between versions', 1),),
           (t("test_t_versions_an_iql_free_statement_that_differs_between_versions_is_refused"),),
           source="plan section 8"),
    Mutant("M13", "Q3a counting a tie as lower",
           ((MODULE, 'sum(measured[method] < measured["dt_nomix"] for method in others)',
             'sum(measured[method] <= measured["dt_nomix"] for method in others)', 1),
            (MODULE, "levels[arm] < levels[subject]),", "levels[arm] <= levels[subject]),", 1)),
           (t("test_t_ties_two_exact_ties_every_integer_and_its_range_by_enumeration"),
            t("test_t_reproduce_record_every_value_p5_2s_record_states_at_its_precision")), source="plan section 8"),
    Mutant("M14", "the random-anchor identity check removed",
           ((MODULE, "    anchor = assert_anchor_identity(payloads[ANCHOR_CELL], payloads[REROLLED_ANCHOR_CELL], "
                     "cells[ANCHOR_CELL].rows,\n                                    cells[REROLLED_ANCHOR_CELL].rows)\n",
             '    anchor = {"removed": True}\n', 1),),
           (t("test_t_inputs_refuses_a_rerolled_anchor_that_differs_from_p5_1s"),), source="plan section 8"),
    Mutant("M15", "iql_correction._p8_4b_rows bypassed",
           ((MODULE, "        rows = ic._p8_4b_rows(output, method, tier, reference)\n",
             "        rows = [json.loads((output / P8_4B_DIR / ar.cell_file_name(ar.CellKey(scenario=SCENARIO, "
             "arm=spec.arm, seed=e.seed, draw_id=e.draw_id))).read_bytes()) for e in reference]\n", 1),),
           (t("test_t_inputs_refuses_a_p8_4b_row_that_does_not_reproduce_its_eval_file"),), source="plan section 8"),
    Mutant("M16", "the README note's means refusal removed on every tier (F13)",
           ((MODULE, '    note_checks = {"random": ic.assert_c1_note_means(output, campaign["cells"], note["payload"])}\n',
             '    note_checks = {"random": {}}\n', 1),
            (MODULE, '        note_checks[tier] = assert_note_means(output, campaign["cells"], note["payload"], tier=tier, '
                     'arms=arms)\n', '        note_checks[tier] = {}\n', 1)),
           (t("test_t_inputs_refuses_an_altered_att_engine_by_the_notes_means"),), source="plan section 8"),
    Mutant("M17", "the dirty-tree refusal removed",
           ((MODULE, '            if git.get("dirty") is not False:\n', "            if False:\n", 1),),
           (t("test_t_cli_build_refuses_a_dirty_tree_and_check_reports_it"),), source="plan section 8"),
    Mutant("M18", "the write-once refusal removed",
           ((MODULE, "    if resolved.exists() or resolved.is_symlink():\n", "    if False:\n", 1),
            (MODULE, "        os.link(temporary, target)\n", "        os.replace(temporary, target)\n", 1)),
           (t("test_t_barrier_an_existing_artifact_is_refused_and_left_byte_identical"),), source="plan section 8"),
    Mutant("M19", "A4.1's module-root check removed",
           ((MODULE, "        code = assert_code_under_root(repo_root)\n", "        code = []\n", 1),
            (MODULE, "        if assert_code_under_root(repo_root) != code:\n", "        if False:\n", 1)),
           (t("test_m19_main_refuses_a_repo_root_the_running_code_is_not_under"),), source="Amendment A, A4.4"),
    Mutant("M20", "the paired_comparison equality refusal removed",
           ((MODULE, "    if (comparison.mean_difference, comparison.ci95_low, comparison.ci95_high) != (stats.mean, low, "
                     "high):\n", "    if False:\n", 1),),
           (t("test_m20_a_one_ulp_perturbation_of_paired_comparisons_mean_is_refused"),), source="Amendment A, A4.4"),
    Mutant("M21", "the collapse_criterion equality refusal removed (F9)",
           ((MODULE, '            if (p5_1["mean_difference"], p5_1["ci95_low"], p5_1["ci95_high"]) != (\n',
             "            if False and (\n", 1),),
           (t("test_t_q6_refuses_when_collapse_criterion_disagrees_with_the_registered_route"),),
           source="added at C5"),
    Mutant("M22", "the record comparison accepting a row neither equal nor explained",
           ((MODULE, "        if not equal and row.exception is None:\n", "        if False:\n", 1),),
           (t("test_a_record_row_neither_equal_nor_explained_refuses_the_build"),), source="added at C5"),
    Mutant("M23", "a declared exception trusted instead of verified",
           ((MODULE, "    if not ok:\n        raise ValueError(f\"{_source(row.file, row.lines)}: the declared exception",
             "    if False:\n        raise ValueError(f\"{_source(row.file, row.lines)}: the declared exception", 1),),
           (t("test_a_declared_exception_is_verified_not_trusted"),), source="added at C5"),
    Mutant("M24", "P5.2b's field-by-field refusal removed",
           ((MODULE, "            if problems:\n", "            if False:\n", 1),),
           (t("test_the_build_refuses_a_p5_2b_statement_it_does_not_reproduce_field_by_field"),),
           source="added at C5"),
    Mutant("M25", "P8.4b's own-route refusal removed",
           ((MODULE, '            if route != tuple(recorded[k] for k in keys) or len(differences) != recorded["n_paired"]:\n',
             "            if False:\n", 1),),
           (t("test_the_build_refuses_a_p8_4b_record_its_own_route_does_not_reproduce"),), source="added at C5"),
    Mutant("M26", "P5.1's P1/P3 equality refusal removed",
           ((MODULE, "    if unequal:\n", "    if False:\n", 1),),
           (t("test_the_build_refuses_a_p5_1_record_it_does_not_reproduce_under_equality"),), source="added at C5"),
    Mutant("M27", "the stop-rule record's own-route refusal removed",
           ((MODULE, "    if tuple(route[k] for k in keys) != tuple(stop_record[k] for k in keys):\n", "    if False:\n",
             1),),
           (t("test_the_build_refuses_a_stop_rule_record_its_own_route_does_not_reproduce"),), source="added at C5"),
    Mutant("M28", "D17's on-disk check removed",
           ((MODULE, "    d17 = _assert_d17(output, sums[P5_2_SUMS_NAME], payloads)\n",
             '    d17 = {"training_records": {}}\n', 1),),
           (t("test_t_inputs_d17_a_deterministic_mappo1000_dt_run_is_refused"),
            t("test_t_inputs_d17_a_complete_one_head_mappo1000_cell_in_p5_2_is_refused")), source="added at C5"),
    Mutant("M29", "an evaluation record's completeness check removed",
           ((MODULE, "    if not ts.cell_is_complete(payload, seeds=seeds, draws=draws, "
                     "declared_steps=ts.DECLARED_GRADIENT_STEPS,\n",
             "    if False and not ts.cell_is_complete(payload, seeds=seeds, draws=draws, "
             "declared_steps=ts.DECLARED_GRADIENT_STEPS,\n", 1),),
           (t("test_t_inputs_refuses_an_incomplete_cell"),), source="added at C5"),
)


def export(repo: Path, commit: str, destination: Path) -> str:
    """``git archive <commit>`` unpacked into *destination*; returns the full commit hash."""
    full = subprocess.run(["git", "-C", str(repo), "rev-parse", commit], capture_output=True, text=True,
                          check=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(repo), "archive", full], capture_output=True, check=True).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")
    return full


def run_tests(tree: Path, tests: tuple[str, ...], output_root: Path) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": str(tree), "RLTRAFFIC_OUTPUT_ROOT": str(output_root),
           "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", "--no-header", *tests]
    completed = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    return completed.returncode, completed.stdout + completed.stderr


def failed_tests(output: str) -> list[str]:
    return sorted({line.split(" ", 1)[1].split(" - ")[0] for line in output.splitlines()
                   if line.startswith(("FAILED ", "ERROR "))})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--work", required=True, help="an empty scratch directory (created)")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    tree = Path(args.work) / "tree"
    commit = export(Path(args.repo), args.commit, tree)
    log = open(args.log, "w", encoding="utf-8")

    def say(text: str) -> None:
        print(text)
        log.write(text + "\n")
        log.flush()

    say(f"# P5.2c mutation run on an exported copy of {commit} at {tree}")
    say(f"# interpreter {sys.executable}; RLTRAFFIC_OUTPUT_ROOT={args.output_root}")
    control_tests = tuple(sorted({test for mutant in MUTANTS for test in mutant.tests}))
    code, output = run_tests(tree, control_tests, Path(args.output_root))
    say(f"\n## control (no mutation): {len(control_tests)} killing tests, pytest exit {code}")
    say(output.strip().splitlines()[-1] if output.strip() else "(no output)")
    results = {"commit": commit, "control": {"tests": list(control_tests), "exit_code": code,
                                             "passed": code == 0}, "mutants": []}
    if code != 0:
        say(output)
        say("CONTROL FAILED: no mutant verdict would mean anything")
        results["all_killed"] = False
        Path(args.results).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    all_killed = True
    for mutant in MUTANTS:
        originals: dict[str, str] = {}
        try:
            for relative, old, new, count in mutant.replacements:
                path = tree / relative
                if relative not in originals:
                    originals[relative] = path.read_text(encoding="utf-8")
                text = path.read_text(encoding="utf-8")
                found = text.count(old)
                if found != count:
                    raise SystemExit(f"{mutant.mutant_id}: {relative} holds {found} occurrence(s) of the replaced "
                                     f"text, not {count}: the mutant would not be the one described")
                path.write_text(text.replace(old, new), encoding="utf-8")
            code, output = run_tests(tree, mutant.tests, Path(args.output_root))
        finally:
            for relative, original in originals.items():
                (tree / relative).write_text(original, encoding="utf-8")
        verdict = "KILLED" if code == 1 else ("SURVIVED" if code == 0 else f"ERROR (pytest exit {code})")
        all_killed = all_killed and verdict == "KILLED"
        failing = failed_tests(output)
        say(f"\n## {mutant.mutant_id} -- {mutant.description} [{mutant.source}]: {verdict}")
        for relative, old, new, count in mutant.replacements:
            say(f"   {relative}: {count} x {old.strip()[:110]!r}\n      -> {new.strip()[:110]!r}")
        say(f"   tests: {', '.join(test.split('::')[1] for test in mutant.tests)}")
        say(f"   failing: {failing}")
        tail = [line for line in output.splitlines() if line.startswith("E ")][:6]
        for line in tail:
            say(f"   {line[:200]}")
        say(f"   {output.strip().splitlines()[-1] if output.strip() else '(no output)'}")
        results["mutants"].append({"id": mutant.mutant_id, "description": mutant.description, "source": mutant.source,
                                   "replacements": [{"file": r, "old": o, "new": n, "count": c}
                                                    for r, o, n, c in mutant.replacements],
                                   "tests": list(mutant.tests), "exit_code": code, "verdict": verdict,
                                   "failing": failing})
    restored = subprocess.run(["git", "-C", str(Path(args.repo)), "archive", commit], capture_output=True,
                              check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(restored)) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name in {r for m in MUTANTS for r, _, _, _ in m.replacements}:
                assert (tree / member.name).read_bytes() == tar.extractfile(member).read(), member.name
    say("\n## every mutated file restored byte for byte to the commit's")
    results["all_killed"] = all_killed
    say(f"\n## {sum(m['verdict'] == 'KILLED' for m in results['mutants'])} of {len(MUTANTS)} KILLED")
    Path(args.results).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.close()
    return 0 if all_killed else 1


if __name__ == "__main__":
    sys.exit(main())
