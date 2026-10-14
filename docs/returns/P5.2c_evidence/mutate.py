"""P5.2c's mutation run (``BRIEF_45`` section 5, Amendment A's A4.4, Amendment B's B1-B3 and Amendment B.1's B.1.3):
every named mutant executed, its output logged.  Amendment B adds the coordinator's seventeen
(``docs/reviews/P5.2c_G1/coord_mutants.py``; CM1 reversed per B2.1, CM15 re-aimed at B2.2's code because the code it
mutated is gone) and a mutant for each B2 ruling.  Amendment B.1 adds the coordinator's four G1.1 survivors (CB0, CB1a,
CB2, CB5c; ``docs/reviews/P5.2c_G1_1/coord_mutants_g11.py``, verbatim) and B.1.2(1)'s two; three more were added at
C6c and are marked so.  Amendment B.2 adds B.2.3's four, on the p-value's definedness and its note (C6d).

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
import hashlib
import io
import json
import os
import subprocess
import sys
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


#: The source of the coordinator's mutants (``BRIEF_45`` Amendment B, B1).
COORD = "Amendment B, B1 (coordinator, docs/reviews/P5.2c_G1/coord_mutants.py)"

#: The source of the coordinator's G1.1 survivors (``BRIEF_45`` Amendment B.1, B.1.1).
COORD_B1 = "Amendment B.1, B.1.1 (coordinator, docs/reviews/P5.2c_G1_1/coord_mutants_g11.py)"


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
           ((MODULE, "    if unresolved.is_symlink() or unresolved.exists():\n", "    if False:\n", 1),
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
    Mutant("M22", "the record comparison accepting a row neither equal nor explained (and carrying it quietly)",
           ((MODULE, "        if not equal and row.exception is None:\n", "        if False:\n", 1),
            (MODULE, '"explanation": None if equal else RECORD_EXCEPTIONS[row.exception],',
             '"explanation": None if equal else RECORD_EXCEPTIONS.get(row.exception),', 1),
            (MODULE, '"verified": None if equal else _verify_exception(row, statements)})',
             '"verified": None if equal or row.exception is None else _verify_exception(row, statements)})', 1)),
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
    Mutant("M30", "M11 and M20 together: the compensated-sum route with the paired_comparison refusal also removed, so "
                  "only the route pin's value assertion stands",
           ((MODULE, "    left_means, right_means = _per_draw_means(a), _per_draw_means(b)\n",
             "    left_means = {d: sum(e.att_horizon for e in a if e.draw_id == d) / sum(1 for e in a if e.draw_id == d)"
             " for d in {e.draw_id for e in a}}\n"
             "    right_means = {d: sum(e.att_horizon for e in b if e.draw_id == d) / sum(1 for e in b if e.draw_id == d)"
             " for d in {e.draw_id for e in b}}\n", 1),
            (MODULE, "    if (comparison.mean_difference, comparison.ci95_low, comparison.ci95_high) != (stats.mean, low, "
                     "high):\n", "    if False:\n", 1)),
           (t("test_t_contrast_route_pin_is_numpy_per_draw_and_not_the_compensated_sum"),), source="added at C5"),
    Mutant("M31", "M2 and M20 together: episode-level pairing with the paired_comparison refusal also removed",
           ((MODULE, "    left_means, right_means = _per_draw_means(a), _per_draw_means(b)\n",
             "    left_means = {(e.seed, e.draw_id): e.att_horizon for e in a}\n"
             "    right_means = {(e.seed, e.draw_id): e.att_horizon for e in b}\n", 1),
            (MODULE, "    if (comparison.mean_difference, comparison.ci95_low, comparison.ci95_high) != (stats.mean, low, "
                     "high):\n", "    if False:\n", 1)),
           (t("test_t_contrast_episode_level_pairing_would_change_n_and_the_interval"),), source="added at C5"),
    # ---- Amendment B, B1: the coordinator's seventeen, verbatim but for CM1 (reversed, B2.1) and CM15 (re-aimed, B2.2).
    Mutant("CM1r", "Q0b: an exactly-zero I scored NOT RESOLVED instead of HELD (CM1 reversed per B2.1)",
           ((MODULE, 'q0b, q0b_reading = "HELD", "exactly zero (identical episodes)',
             'q0b, q0b_reading = "NOT RESOLVED", "exactly zero (identical episodes)', 1),),
           (t("test_b1_cm1_an_exactly_zero_interaction_is_held_by_the_registered_inclusion_rule"),), source=COORD),
    Mutant("CM2", "composite I: CI half-width doubled, mean unchanged",
           ((MODULE, "    stats = mean_ci95(values)\n    low, high = stats.mean - stats.ci95, stats.mean + stats.ci95\n"
                     "    seeds = sorted(",
             "    stats = mean_ci95(values)\n    low, high = stats.mean - 2 * stats.ci95, stats.mean + 2 * stats.ci95\n"
             "    seeds = sorted(", 1),),
           (t("test_t_q0_the_interaction_its_per_seed_values_and_the_three_verdicts"),), source=COORD),
    Mutant("CM3", "Q3b sub-prediction: random no longer predicted to resolve",
           ((MODULE, 'predicted_to_resolve = {"maxpressure": False, "fixedtime": False, "random": True}',
             'predicted_to_resolve = {"maxpressure": False, "fixedtime": False, "random": False}', 1),),
           (t("test_q3b_reads_the_shared_anchor_as_behaviour_at_random_and_reports_the_sub_prediction"),), source=COORD),
    Mutant("CM4", "Q1b scored with LEVEL_BAND (0.30) instead of ANCHOR_BAND (0.10)",
           ((MODULE, "cells=(cell,), band=ts.ANCHOR_BAND, threshold=1)", "cells=(cell,), band=ts.LEVEL_BAND, threshold=1)",
             1),),
           (t("test_b1_cm4_q1b_holds_only_within_ten_percent_of_the_predicted_level"),), source=COORD),
    Mutant("CM5", "Q4a scored on four tiers (mappo1000 added)",
           ((MODULE, "    outcomes = {}\n    for tier in OUT_OF_SAMPLE_TIERS:\n        outcomes[tier] = interval_outcome("
                     "blocks[tier]",
             "    outcomes = {}\n    for tier in TIERS:\n        outcomes[tier] = interval_outcome(blocks[tier]", 1),),
           (t("test_every_registered_statement_is_present_under_both_definitions"),), source=COORD),
    Mutant("CM6", "Q5a scored on four tiers (mappo1000 added)",
           ((MODULE, '("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in OUT_OF_SAMPLE_TIERS}),',
             '("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in TIERS}),', 1),),
           (t("test_t_q5_each_tier_by_its_registered_sign"),), source=COORD),
    Mutant("CM6v", "CM6 with a registered Q5a value supplied at mappo1000, so the KeyError it raises cannot kill it and "
                   "only the tier assertions stand",
           ((MODULE, '("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in OUT_OF_SAMPLE_TIERS}),',
             '("Q5a", ("bc_top10", "bc"), {tier: "positive" for tier in TIERS}),', 1),
            (MODULE, '"Q5a": {"maxpressure": 575.34,', '"Q5a": {"mappo1000": 0.0, "maxpressure": 575.34,', 1)),
           (t("test_t_q5_each_tier_by_its_registered_sign"),
            t("test_every_registered_statement_is_present_under_both_definitions")), source="added at C6"),
    Mutant("CM7", "leaders: per-seed first places read att_ours whatever the definition",
           ((MODULE, "seed_levels = _seed_levels(book, tier, definition, ts.METHODS)",
             'seed_levels = _seed_levels(book, tier, "att_ours", ts.METHODS)', 1),),
           (t("test_b1_cm7_the_leaders_per_seed_first_places_are_read_under_each_definition"),
            t("test_b1_cm7_real_maxpressure_first_places_per_seed_under_each_definition")), source=COORD),
    Mutant("CM8", "collapse audit: the sentence holds if ANY non-DT arm collapses somewhere",
           ((MODULE, '"holds": dt_never and all(collapsing.values())})', '"holds": dt_never and any(collapsing.values())})',
             1),),
           (t("test_b1_cm8_the_sentence_fails_when_one_non_dt_arm_never_collapses"),), source=COORD),
    Mutant("CM9", "contrast: D9 per-seed block always on att_ours",
           ((MODULE, "left_name=left.spec.arm, right_name=right.spec.arm,\n                                 metric=definition)",
             'left_name=left.spec.arm, right_name=right.spec.arm,\n                                 metric="att_ours")', 1),),
           (t("test_b1_cm9_each_definitions_per_seed_block_reads_that_definitions_values"),
            t("test_b1_cm9_real_an_att_engine_per_seed_block_against_the_raw_rows")), source=COORD),
    Mutant("CM10", "Q6: COLLAPSED iff the CI's HIGH end is above zero",
           ((MODULE, 'verdict = "COLLAPSED" if block["ci95_low"] > 0.0 else "NOT COLLAPSED"\n            entries.append(_entry('
                     '"Q6"',
             'verdict = "COLLAPSED" if block["ci95_high"] > 0.0 else "NOT COLLAPSED"\n            entries.append(_entry("Q6"',
             1),),
           (t("test_b1_cm10_a_dt_arm_whose_ci_against_the_anchor_straddles_zero_is_not_collapsed"),), source=COORD),
    Mutant("CM11", "Q5c: random's predicted sign flipped to negative",
           ((MODULE, '"fixedtime": "positive", "random": "positive"}),', '"fixedtime": "positive", "random": "negative"}),',
             1),),
           (t("test_t_q5_each_tier_by_its_registered_sign"),), source=COORD),
    Mutant("CM12", "Q2a: a tie for first counted as the predicted unique first place",
           ((MODULE, '"outcome": "HELD" if firsts == [predicted[0]] else "FAILED",',
             '"outcome": "HELD" if predicted[0] in firsts else "FAILED",', 1),),
           (t("test_b1_cm12_a_predicted_first_place_tied_for_first_is_not_held"),), source=COORD),
    Mutant("CM13", "contrast: exactly_zero if ANY per-draw difference is zero",
           ((MODULE, "    exactly_zero = all(value == 0.0 for value in differences)",
             "    exactly_zero = any(value == 0.0 for value in differences)", 1),),
           (t("test_cm13_a_contrast_zero_on_some_draws_only_is_not_exactly_zero"),
            t("test_t_reproduce_record_every_value_p5_2s_record_states_at_its_precision")), source=COORD),
    Mutant("CM14", "Q4b: the chain scored non-strictly (<=)",
           ((MODULE, "increasing = all(a < b for a, b in zip(values, values[1:]))",
             "increasing = all(a <= b for a, b in zip(values, values[1:]))", 1),),
           (t("test_b1_cm14_a_chain_with_two_equal_rungs_is_not_strictly_increasing"),), source=COORD),
    Mutant("CM15b", "composite I: per-seed values computed as d1 - d4 (CM15 re-aimed at B2.2's code: the composite rows "
                    "passed to the worker swapped)",
           ((MODULE, "worker = ts.per_seed_block(left_rows, right_rows,", "worker = ts.per_seed_block(right_rows, left_rows,",
             1),),
           (t("test_t_q0_the_interaction_its_per_seed_values_and_the_three_verdicts"),), source=COORD),
    Mutant("CM16", "leaders: the lead 'resolves' when the CI's LOW end is below zero",
           ((MODULE, 'resolved = [block["ci95_high"] < 0.0 for block in contrasts]',
             'resolved = [block["ci95_low"] < 0.0 for block in contrasts]', 1),),
           (t("test_b1_cm16_a_lead_whose_ci_straddles_zero_does_not_resolve"),
            t("test_b1_cm16_real_maxpressures_lead_resolves_under_att_ours_and_not_under_att_engine")), source=COORD),
    Mutant("CM17", "Q3b: the aggregate ignores fixedtime (two tiers only)",
           ((MODULE, "                          verdict=conjunction(list(outcomes.values())),\n"
                     '                          numbers={"tiers": outcomes, "rule": "sign negative on all three',
             "                          verdict=conjunction([o for t, o in outcomes.items() if t != \"fixedtime\"]),\n"
             '                          numbers={"tiers": outcomes, "rule": "sign negative on all three', 1),),
           (t("test_b1_cm17_q3b_aggregate_reads_all_three_tiers"),
            t("test_b1_cm17_real_q3b_aggregate_is_not_resolved_under_both_definitions")), source=COORD),
    # ---- Amendment B, B2: a mutant for each ruling's behaviour.
    Mutant("B2.1a", "Q3b's sub-prediction loses its exact-zero flag",
           ((MODULE, '"matched": matched, "exactly_zero": block["exactly_zero"],',
             '"matched": matched, "exactly_zero": False,', 1),),
           (t("test_q3b_reads_the_shared_anchor_as_behaviour_at_random_and_reports_the_sub_prediction"),),
           source="Amendment B, B2.1"),
    Mutant("B2.1b", "Q3b's sub-prediction reading no longer says the match comes from identical episodes",
           ((MODULE, '        if block["exactly_zero"]:\n            source = (', '        if False:\n            source = (', 1),),
           (t("test_q3b_reads_the_shared_anchor_as_behaviour_at_random_and_reports_the_sub_prediction"),),
           source="Amendment B, B2.1"),
    Mutant("B2.2", "I's per-seed block drops the two fields P5.1's schema adds (n_seeds_favouring_left, "
                   "reverses_on_n_seeds)",
           ((MODULE, '        "per_seed": worker["per_seed"],',
             '        "per_seed": {k: v for k, v in worker["per_seed"].items() if k not in ("n_seeds_favouring_left", '
             '"reverses_on_n_seeds")},', 1),),
           (t("test_t_q0_the_interaction_its_per_seed_values_and_the_three_verdicts"),), source="Amendment B, B2.2"),
    Mutant("B2.3a", "measured_first_under_rule names the tie-break winner of a pooled tie",
           ((MODULE, 'statements["q2a"]["measured_first_under_rule"] = order[0] if len(firsts) == 1 else None',
             'statements["q2a"]["measured_first_under_rule"] = order[0]', 1),),
           (t("test_b2_3_a_pooled_tie_for_first_has_no_first_under_the_rule_and_reversals_against_the_pooled_set"),),
           source="Amendment B, B2.3"),
    Mutant("B2.3b", "the reversal list computed against the tie-break winner, not the pooled set",
           ((MODULE, 'int(seed) for seed, block in per_seed.items() if set(block["firsts"]) != set(firsts)]',
             'int(seed) for seed, block in per_seed.items() if block["firsts"] != [order[0]]]', 1),),
           (t("test_b2_3_a_pooled_tie_for_first_has_no_first_under_the_rule_and_reversals_against_the_pooled_set"),),
           source="Amendment B, B2.3"),
    Mutant("B2.4a", "the running-file check removed (B2.4's named mutant)",
           ((MODULE, "    for what, path in running.items():\n        if package not in path.parents:\n",
             "    for what, path in running.items():\n        if False:\n", 1),),
           (t("test_b2_4_the_file_run_by_path_from_another_tree_is_refused"),), source="Amendment B, B2.4"),
    Mutant("B2.4b", "provenance.code hashes the module registered under the name, not the file that runs",
           ((MODULE, 'path = running["this module\'s own file"] if name == "offline.tier_statements" else located(name)[0]',
             "path = located(name)[0]", 1),),
           (t("test_b2_4_provenance_code_hashes_the_file_that_runs"),), source="Amendment B, B2.4"),
    Mutant("B2.5a", "the ulp bound on P8.4b's V4/V5 removed",
           ((MODULE, "            for k in keys:\n                assert_within_ulps(mine[k], recorded[k],",
             "            for k in ():\n                assert_within_ulps(mine[k], recorded[k],", 1),),
           (t("test_b2_5_a_value_five_ulps_from_p8_4bs_record_is_refused_and_four_are_accepted"),),
           source="Amendment B, B2.5"),
    Mutant("B2.5b", "the ulp bound on P5.2's stop-rule record removed",
           ((MODULE, "    for k in keys[:3]:\n        assert_within_ulps(d4[k], stop_record[k],",
             "    for k in ():\n        assert_within_ulps(d4[k], stop_record[k],", 1),),
           (t("test_b2_5_a_stop_rule_record_five_ulps_away_is_refused_and_four_are_accepted"),),
           source="Amendment B, B2.5"),
    Mutant("B2.5c", "the bound loosened to 5 ulps",
           ((MODULE, "ULP_BOUND = 4\n", "ULP_BOUND = 5\n", 1),),
           (t("test_b2_5_a_value_five_ulps_from_p8_4bs_record_is_refused_and_four_are_accepted"),
            t("test_b2_5_a_stop_rule_record_five_ulps_away_is_refused_and_four_are_accepted")), source="Amendment B, B2.5"),
    Mutant("B2.5d", "the bound tightened to 3 ulps (4 must be accepted)",
           ((MODULE, "ULP_BOUND = 4\n", "ULP_BOUND = 3\n", 1),),
           (t("test_b2_5_a_value_five_ulps_from_p8_4bs_record_is_refused_and_four_are_accepted"),
            t("test_b2_5_a_stop_rule_record_five_ulps_away_is_refused_and_four_are_accepted")), source="Amendment B, B2.5"),
    Mutant("B2.6", "GATE 2's Q5 record cited at line 1274 only",
           ((MODULE, "    Quote(_BRIEF_27, (1274, 1281),\n", "    Quote(_BRIEF_27, (1274, 1274),\n", 1),),
           (t("test_b2_6_the_gate_2_record_cites_1274_to_1281_and_quotes_the_sentence_that_is_the_error"),),
           source="Amendment B, B2.6"),
    Mutant("B2.7", "Q6 labelled never scored like the others",
           ((MODULE, '                                   if statement_id == "Q6" else\n',
             "                                   if False else\n", 1),),
           (t("test_b2_7_q6_in_never_scored_names_its_numbers_and_the_recorded_verdict"),), source="Amendment B, B2.7"),
    Mutant("B2.9a", "the out path's symlink refusal removed (R2-B's analogue)",
           ((MODULE, "    if unresolved.is_symlink() or unresolved.exists():\n", "    if unresolved.exists():\n", 1),),
           (t("test_b2_9_a_dangling_symlink_at_the_out_path_is_refused_and_nothing_is_written"),),
           source="Amendment B, B2.9"),
    Mutant("B2.9b", "the exists check back on the resolved path (G1's dead-code form)",
           ((MODULE, "    if unresolved.is_symlink() or unresolved.exists():\n",
             "    if resolved.exists() or resolved.is_symlink():\n", 1),),
           (t("test_b2_9_a_dangling_symlink_at_the_out_path_is_refused_and_nothing_is_written"),),
           source="Amendment B, B2.9"),
    Mutant("B2.9c", "the input-equality guard removed (R2-A)",
           ((MODULE, '    if resolved in inputs_read:\n        raise ValueError(f"{resolved} is an input of this artifact")\n',
             '    if False:\n        raise ValueError(f"{resolved} is an input of this artifact")\n', 1),),
           (t("test_t_barrier_refuses_an_input_as_the_out_path"),), source="Amendment B, B2.9"),
    Mutant("B2.11a", "a line-pinned citation of a living document, extension-less (PROJECT_PLAN:2296), in the module",
           ((MODULE, '"Rule R makes att_engine the primary metric on grid4x4;',
             '"Rule R (PROJECT_PLAN:2296) makes att_engine the primary metric on grid4x4;', 1),),
           (t("test_t_quotes_no_line_pinned_citation_points_into_a_living_document"),), source="Amendment B, B2.11"),
    Mutant("B2.11b", "a line-pinned citation of a living document, backticked (`docs/PROJECT_PLAN.md`:2296)",
           ((MODULE, '"Rule R makes att_engine the primary metric on grid4x4;',
             '"Rule R (`docs/PROJECT_PLAN.md`:2296) makes att_engine the primary metric on grid4x4;', 1),),
           (t("test_t_quotes_no_line_pinned_citation_points_into_a_living_document"),), source="Amendment B, B2.11"),
    Mutant("B2.11c", "a Q5 record quoted from a living document",
           ((MODULE, 'Quote(_PACKET, (30, 30), "CONFIRMED twice', 'Quote("docs/PROJECT_PLAN.md", (30, 30), "CONFIRMED twice',
             1),),
           (t("test_t_quotes_every_record_literal_appears_at_its_cited_lines"),), source="Amendment B, B2.11"),
    Mutant("B2.13", "what_this_does_not_say loses the sentence on what definition_dependent does not compare",
           ((MODULE, "are reported per definition beside the flag, and the flag does not compare them",
             "are reported per definition beside the flag", 1),),
           (t("test_b2_13_what_this_does_not_say_states_what_definition_dependent_does_not_compare"),),
           source="Amendment B, B2.13"),
    Mutant("B2.14", "P8.4b's campaign manifest and marker not listed in provenance.inputs",
           ((MODULE, "    for name in (ar.MANIFEST_NAME, ar.COMPLETE_MARKER):  # BRIEF_45 Amendment B, B2.14",
             "    for name in ():  # BRIEF_45 Amendment B, B2.14", 1),),
           (t("test_b2_14_p8_4bs_campaign_manifest_and_marker_are_listed_with_their_sha256"),
            t("test_b2_14_real_p8_4bs_campaign_manifest_and_marker_are_listed_with_their_sha256")),
           source="Amendment B, B2.14"),
    # ---- Amendment B.1, B.1.1: the coordinator's four G1.1 survivors, his strings verbatim.
    Mutant("CB0", "Q0b's inclusion rule written with strict inequalities (a CI touching zero excluded)",
           ((MODULE, 'elif interaction["ci95_low"] <= 0.0 <= interaction["ci95_high"]:',
             'elif interaction["ci95_low"] < 0.0 < interaction["ci95_high"]:', 1),),
           (t("test_b11_cb0_a_ci_touching_zero_at_one_end_is_included_by_q0bs_rule"),), source=COORD_B1),
    Mutant("CB1a", "Q3b sub-prediction: the exact-zero source always says 'identical episodes'",
           ((MODULE, '"identical episodes" if block["discriminability"]["identical"] else',
             '"identical episodes" if True else', 1),),
           (t("test_b11_cb1a_q3bs_exact_zero_from_episodes_that_differ_is_not_called_identical"),), source=COORD_B1),
    Mutant("CB2", "composite I: the worker's names swapped (d1 named left, d4 right), so the qualifier names the pair "
                  "backwards",
           ((MODULE, 'left_name="d4", right_name="d1", metric=definition)',
             'left_name="d1", right_name="d4", metric=definition)', 1),),
           (t("test_b11_cb2_q0bs_qualifier_names_d4_against_d1_from_this_files_per_seed_loop"),
            t("test_b11_cb2_real_q0bs_per_seed_values_and_qualifier_under_both_definitions")), source=COORD_B1),
    Mutant("CB5c", "B2.5: P8.4b's ulp check on the first key only (the CI ends unchecked)",
           ((MODULE, "            for k in keys:\n                assert_within_ulps(mine[k], recorded[k]",
             "            for k in keys[:1]:\n                assert_within_ulps(mine[k], recorded[k]", 1),),
           (t("test_b2_5_a_value_five_ulps_from_p8_4bs_record_is_refused_and_four_are_accepted"),), source=COORD_B1),
    Mutant("CB5c.stop", "CB5c's twin for P5.2's stop-rule record: the ulp check on the mean only",
           ((MODULE, "    for k in keys[:3]:\n        assert_within_ulps(d4[k], stop_record[k],",
             "    for k in keys[:1]:\n        assert_within_ulps(d4[k], stop_record[k],", 1),),
           (t("test_b2_5_a_stop_rule_record_five_ulps_away_is_refused_and_four_are_accepted"),), source="added at C6c"),
    # ---- Amendment B.1, B.1.2(1): I's identity, which Q0b's exact-zero label reads.
    Mutant("B.1.2(1)a", "I's identity forced True: the label says identical episodes whatever the pairs hold",
           ((MODULE, '    identical = identical_pairs["d4"] and identical_pairs["d1"]\n', "    identical = True\n", 1),),
           (t("test_b11_2_an_i_exactly_zero_because_the_horizon_changes_nothing_is_held_and_not_identical"),
            t("test_b11_2_identical_needs_both_pairs_identical")), source="Amendment B.1, B.1.2(1)"),
    Mutant("B.1.2(1)b", "I's identity forced False: identical episodes are never named",
           ((MODULE, '    identical = identical_pairs["d4"] and identical_pairs["d1"]\n', "    identical = False\n", 1),),
           (t("test_b1_cm1_an_exactly_zero_interaction_is_held_by_the_registered_inclusion_rule"),),
           source="Amendment B.1, B.1.2(1)"),
    Mutant("B.1.2(1)c", "I identical when EITHER pair is (and -> or)",
           ((MODULE, '    identical = identical_pairs["d4"] and identical_pairs["d1"]\n',
             '    identical = identical_pairs["d4"] or identical_pairs["d1"]\n', 1),),
           (t("test_b11_2_identical_needs_both_pairs_identical"),), source="added at C6c"),
    Mutant("B.1.2(1)d", "an exactly-zero I from episodes that are not identical scored NOT RESOLVED (CM1r's twin)",
           ((MODULE, '    elif interaction["exactly_zero"]:\n        q0b, q0b_reading = "HELD", (',
             '    elif interaction["exactly_zero"]:\n        q0b, q0b_reading = "NOT RESOLVED", (', 1),),
           (t("test_b11_2_an_i_exactly_zero_because_the_horizon_changes_nothing_is_held_and_not_identical"),),
           source="added at C6c"),
    # ---- Amendment B.2, B.2.3 (C6d): the p-value's definedness and its note.  Each is listed with the test whose
    # ---- failure the run of record shows; B.2.3 (4)'s gated test kills all four as well, which c6d_gated_probe.txt
    # ---- shows separately (pytest -x stops at the first failure, so one run cannot evidence both).
    Mutant("B.2.3a", "p_value_defined forced True: an empty signed-rank test published as a test",
           ((MODULE, '            "p_value_defined": defined,\n', '            "p_value_defined": True,\n', 1),),
           (t("test_b23_an_exactly_zero_contrasts_p_value_is_labelled_undefined_with_its_convention"),),
           source="Amendment B.2, B.2.3"),
    Mutant("B.2.3b", "p_value_defined forced False: every two-sided test called undefined",
           ((MODULE, '            "p_value_defined": defined,\n', '            "p_value_defined": False,\n', 1),),
           (t("test_b23_a_contrast_zero_on_some_draws_only_has_a_defined_p_value"),
            t("test_b23_a_two_sided_z_comes_from_min_of_the_rank_sums_so_its_sign_is_not_a_direction")),
           source="Amendment B.2, B.2.3"),
    Mutant("B.2.3c", "p_value_defined read off n_zero == 0 instead of n_used > 0: a test with any tied draw called "
                     "undefined",
           ((MODULE, '            "p_value_defined": defined,\n',
             '            "p_value_defined": int(wilcoxon.n_zero) == 0,\n', 1),),
           (t("test_b23_a_contrast_zero_on_some_draws_only_has_a_defined_p_value"),),
           source="Amendment B.2, B.2.3"),
    Mutant("B.2.3d", "the two p-value notes swapped: the empty test gets the two-sided note and the two-sided test "
                     "gets the empty one",
           ((MODULE, '"(BRIEF_45 Amendment B.2, B.2.3)" if defined else',
             '"(BRIEF_45 Amendment B.2, B.2.3)" if not defined else', 1),),
           (t("test_b23_an_exactly_zero_contrasts_p_value_is_labelled_undefined_with_its_convention"),
            t("test_b23_a_contrast_zero_on_some_draws_only_has_a_defined_p_value")),
           source="Amendment B.2, B.2.3"),
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
    say(f"# this script: {Path(__file__).name} at sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
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
