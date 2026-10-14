# G5 / M2a findings -- P5.2c statements artifact (words and summary blocks)
Reviewer: read-only merge reviewer, gate G5, mandate M2a. Started 2026-10-14.
Artifact: /home/filip/rltraffic-p52c/docs/data/p5_2_statements.json
Registered text: /home/filip/rltraffic/docs/plans/p5.2.md:405-708
Rulings: /home/filip/rltraffic/docs/briefs/BRIEF_45_p5.2c_statements.md (s3.3, B.1.2(1), B.2.3, B.2.5, C1, C2)
Scratch scripts: /tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad/g5/m2a/
Known and excluded: comparison_with_the_record.never_scored mean_difference null for Q3c's four rows.

## Checklist
- [x] 1. THE NULL CLASS: walk the artifact; every null leaf, empty list/dict, and key whose same-kind siblings carry a number; classify each as explained "not applicable" vs value read by a name the source does not carry.
- [x] 2. READINGS VS NUMBERS: every statement `reading` vs its own numbers (CI sign, exactly_zero/identical, NOT RESOLVED straddle, Q3c against-DT ci95_low>0, Q4b inverted strictly decreasing); leaders[*].reading; descriptive_collapse_audit rows; Q3b sub_prediction readings.
- [x] 3. NOTES AND LABELS: every *_note, label, role, rule, no_verdict, why_not_scored, predictions_registered_on, explanation, definitions.* text vs numbers and registered text; spot-check >= 10 plan-line citations against docs/plans/p5.2.md.
- [x] 4. what_this_does_not_say, registered_in, role: read in full; anything false or missing given BRIEF_45 C2 and B.2.5 (5-of-500 fixedtime episodes for Q5a/Q5b; empty Wilcoxon tests; Q3c maxpressure under primary definition).
- [x] 5. comparison_with_the_record sub-blocks other than literal_records (p8_4b, p5_1_grid, p5_2_stop_rule_record, p5_2b_artifact, q5, exceptions, never_scored): recorded values vs source files; every explanation sentence vs its numbers (recompute ulps).
- [x] 6. `changes` and primary / definition_dependent: att_engine primary on every pair; changes wording verdict-change vs reading-change.

## Results (appended per item / finding)

### Item 1 -- THE NULL CLASS  [x] done 2026-10-14 (scripts s03_nulls.py, s04_never.py, s08_perseed.py, s10_nulls2.py, s11_changes.py, s13_grep.py)
Walk result: 926 null leaves over 32 generalised paths; 176 empty containers over 14 paths.
Classified as deliberate / explained (no finding): statements.arm (only Q6 carries an arm); statements.tier/verdict/reading for
tier-less, context, count and Q3c rows; Q0b numbers.rank_biserial (explained by rank_biserial_note, A2 Q8); Q2a numbers.reversal
(null exactly where reverses_on_seeds is []; non-null only at maxpressure att_engine, seed 303); ties[*].note (only mappo1000 needs
one); literal_records exception/explanation/verified (null on the 115 equal rows, filled on all 8 unequal rows);
changes.arm/tier (Q6-only arm; tier-less ids). Every seed_reversal_qualifier null was recomputed: s08 recomputes the set of seeds
whose per-seed value has the opposite sign of the pooled mean, for all 76 contrast blocks in statements; the qualifier is non-null
exactly where that set is non-empty (Q0b 101; Q3c maxpressure att_engine 303; Q4.d/Q4a random 202; Q5c random 303,505). 0 mismatches.
Q5b's absence from never_scored is correct (BRIEF_45 s1.3: GATE 2 reported Q5b's values).

F1.1  MAJOR -- another member of the known class, in `changes`: Q3c's rows carry no point value.
  Evidence: changes[34] (Q3c maxpressure), changes[35] (fixedtime), changes[36] (random) carry keys {ci95_low, ci95_high, reading,
  verdict} but NO mean_difference, while every other contrast row in `changes` (Q0a, Q0b, Q3b, Q4.d, Q4a, Q5a-c, Q6) carries
  mean_difference. changes[33] (Q3c mappo1000, context) carries neither the value nor the CI (keys {reading, verdict} only), although
  statements[80]/[81] hold numbers.contrast.mean_difference -11.1329 / -11.4093. Cause is the same name mismatch as the known
  never_scored defect: Q3c stores its value as numbers.mean (and, for the context row, under numbers.contrast.*).
  Consequence: the IQL correction's effect on Q3c at random (att_ours +64.3603 -> +56.8432; att_engine +56.7834, statements[87],[86],[88])
  and Rule R's effect at maxpressure (+1.2993 -> +0.9059, statements[82],[83]) are absent from the block whose job is to show
  "as committed -> corrected att_ours -> corrected att_engine" (BRIEF_45 s3.4).
  Expected: each Q3c changes row carries the value under one consistent name (e.g. mean_difference = numbers.mean), and the
  mappo1000 context row carries numbers.contrast's mean_difference and CI.

F1.2  MINOR -- `changes[*].changed_by_rule_r` / `changed_by_the_correction` are defined nowhere in the artifact.
  Evidence: s13_grep "changed_by" finds the keys only; `definitions` has 15 keys, none about `changes`. From the data the booleans are
  true iff verdict or reading differs (changes[21] D9 maxpressure and changes[34] Q3c maxpressure are the only true values).
  changes[36] (Q3c random) has changed_by_the_correction = false while its CI moves from [62.1046, 66.6160] to [53.9794, 59.7069]
  (a 7.5 s shift of the value). "false" there means "the reading did not change", not "the value did not change".
  Expected: a definitions entry, e.g. "changed_by_*: true iff the verdict or the reading differs; values may change without it".

F1.3  MINOR -- `reverses_on_n_seeds` is asymmetric under sign flip when seeds are exactly zero (Q5b vs Q5a at fixedtime).
  Evidence: statements[112]/[113] Q5a fixedtime by_seed {101:+1.3907, 202:0, 303:+0.0172, 404:0, 505:0}: reverses_on_n_seeds 0.
  statements[122]/[123] Q5b fixedtime is the exact negation (by_seed {101:-1.3907, 202:0, 303:-0.0172, 404:0, 505:0}) yet
  reverses_on_n_seeds = 2 (n_seeds_favouring_left 2; zero seeds counted on the other side). reverses_against_pooled_sign is 0 for
  both and the qualifier is null for both (correct). The artifact's own definition ("size of the seed MINORITY by sign") does not
  say how zero seeds are counted. A paper writer reading reverses_on_n_seeds would write "Q5b at fixedtime reverses on 2 of 5
  seeds", which is false: no seed opposes the pooled sign; 3 seeds are exactly zero.
  Expected: the definition says zero seeds are counted with the non-negative side, or the field is not reported where the minority
  is made of zeros; plus "3 of 5 seeds exactly zero" stated for Q5a/Q5b fixedtime.

F1.4  NOTE -- aggregate statements (Q2b/Q3b/Q4a/Q5a/Q5b/Q5c .aggregate) and Q1b per-tier rows have reading null; never_scored
  mirrors this for Q5a.aggregate / Q5c.aggregate (mean, CI, reading, tier null). Deliberate (an aggregate has no single contrast) but
  not explained in the artifact; a reading such as "NOT RESOLVED: HELD . NOT RESOLVED . HELD" would remove the gap.

### Item 2 -- READINGS VS NUMBERS  [x] done (scripts s06_table.py, s16_readings.py, s17_readings2.py, s18_leaders.py, s20_collapse2.py)
Checked by script, every statement: "resolves ... above zero" => ci95_low > 0; "... below zero" => ci95_high < 0; "does not resolve" =>
straddles and not exactly_zero; "exactly zero (identical episodes)" => exactly_zero AND discriminability.identical AND CI [0,0];
Q0b's "excludes zero; its sign is negative" => ci95_high < 0; Q3c "against" => ci95_low > 0, "for" => ci95_high < 0 (mappo1000 context
rows read numbers.contrast: [-13.79,-8.47] / [-14.09,-8.73]), "NOT RESOLVED" => straddles; Q3c top-level mean/CI == numbers.contrast;
verdict == interval rule from predicted_sign for every interval verdict; Q6 verdict == (ci95_low > 0). 0 problems in 154 statements.
Q4b / Q4b.three_rung: chain 40.0210 > 0.6895 > 0.0 > -0.7098 (and 40.0210 > 0.6895 > -0.7098): strictly decreasing true, the
"inverted" reading is right. Q3b sub_predictions: maxpressure not matched (CI resolves, predicted not to), fixedtime matched via
exact zero (labelled as such), random matched -- all agree with their CIs. Q0.stop_rule CONTINUE with CI [+24.14,+30.77] agrees
with the quoted rule (p5.2.md:526-529). Ranking/Q2a/Q2b/Q3a/D9 readings agree with leaders[*].order and per_seed_first_places.
leaders[0..9].reading: all 10 agree with their leader_minus_runner_up CI (maxpressure att_engine: "does not resolve", CI
[-1.9735, +0.1617], qualifier names seed 303). descriptive_collapse_audit: 34 rows + 16 dt_rows, verdict == (ci95_low > 0) and the
contrast reading agrees with the CI in all 50; sentence_holds: DT arms never collapse; bc, bc_top10, bc_top10_perix, iql each collapse
on >= 1 tier under all four (definition, version) blocks.
No reading contradicts its numbers.

F2.1  MINOR -- brief s3.3's "say so where those blocks appear" (seed label is an evaluation label for behaviour / random-anchor
  cells) is met once, in definitions.per_seed, not in the blocks. Every per_seed.note (76 of 76 in statements, identical text) is the
  generic note; Q3b (right arm behaviour@* / random@grid4x4_mappo1000), Q6 and all 50 collapse-audit contrasts (right arm the random
  anchor) carry no such label. No qualifier string is wrong today (no Q3b/Q6/audit contrast reverses, so no "training seeds" text is
  emitted for them), but a per-seed sentence written from those blocks would call the anchor's labels training seeds.
  Expected: the per_seed.note of every block whose right arm is behaviour@* or random@grid4x4_mappo1000 states it.

### Item 3 -- NOTES AND LABELS  [x] done (scripts s21_quotes.py, s22_cites.py, s23_labels.py, s24_q2q6.py)
Quoted-rule objects ({file, lines, text}): 499 objects, 45 unique; every text appears verbatim (markdown '>', '*', '`', '\' stripped)
inside its cited line range (s21; p5.2.md, p5.1.md and PREREGISTRATION.md). 0 failures. Plan copy in the worktree == main (cmp).
Plain-string citations spot-checked against p5.2.md (16 distinct): 407-412 (att_horizon primary; dt_gate._per_draw_means) OK; 497
(Q1b 167.4920 / 206.9318) OK; 513 (d1 +39.5649 [+36.0510,+43.0787] == statements[0]) OK; 576-578 (Q1b +-10 %) OK; 606-607, 606-609
(tie rule) OK; 611-612 (hard counts 2..2 . 2..4 . 3..3 == statements[46..51]) OK; 612, 613, 614, 615 OK; 645-646 (mappo1000 context)
OK; 655-659 (Q3c: value, no outcome rule) OK; 670-672 (Q4b three-rung) OK; 699-701 (Q6) OK as a location; 674-675 see F3.3.
definitions.* read in full: no false statement found (rule_r.reporting states DD covers "verdicts (or readings)", consistent with
the data; exact_zero, outcomes, interval, paired_unit consistent with the statements).

F3.1  MINOR -- Q2a's quoted rule says "Scored on the 2 out-of-sample tiers" and the artifact scores three, without a word.
  Evidence: statements[26..33].rule[0] quotes p5.2.md:604-605 "...Scored on the 2 out-of-sample tiers and reported per tier."; the
  artifact scores maxpressure, fixedtime and random (C1: "FAILED on all three tiers"). The same staleness in Q2b's "(>= 24 of 30
  overall)" IS explained (statements[42..45].numbers.overall.why_not_scored); Q2a's is not. (fixedtime's predicted first place IS
  registered: p5.2.md:595 and :601, "dt_nomix first".)
  Expected: a Q2a note mirroring Q2b's: "the registered '2 out-of-sample tiers' predates fixedtime; fixedtime's ordering was
  registered with the tier (p5.2.md:595, 601), so Q2a is reported on all three".

F3.2  MINOR -- descriptive_collapse_audit.rule calls a paraphrase "Q6's rule verbatim".
  Evidence: rule = "Q6's rule verbatim (docs/plans/p5.2.md:699-701): COLLAPSED iff the arm's mean is worse than the shared random
  anchor's with the paired CI excluding 0, else NOT COLLAPSED". p5.2.md:699-701 reads "COLLAPSED (for a DT arm on a tier) iff its
  held-out mean att_horizon is worse than the random anchor's with the paired CI excluding 0." The audit applies it to non-DT arms
  (bc, bc_top10, bc_top10_perix, iql) and under att_engine -- exactly where it is NOT verbatim.
  Expected: "Q6's rule (docs/plans/p5.2.md:699-701, registered for a DT arm under att_horizon), applied descriptively to every arm and
  under both definitions".

F3.3  NOTE -- Q4b's four-rung form cites p5.2.md:674-675; the four-rung chain's values and order are on line 673
  ("Predicted, in measured-ATT tier order: +39.5649* -> +41.7402 -> +51.3689 -> +62.7106"); 674-675 only says fixedtime makes four
  rungs. Expected: 673-675.

F3.4  NOTE -- Q6 dt_spatial at random: registered_expectation "COLLAPSED" (p5.2.md:705-706) sits beside verdict "NOT COLLAPSED"
  (statements[150],[151]: CI [-6.2384,-5.2462] / [-6.2516,-5.2586], i.e. dt_spatial BETTER than the anchor) with no field saying the
  registered expectation was not met. Not false; a paper writer must notice it by comparing two fields.

### Item 4 -- what_this_does_not_say, registered_in, role  [x] done (s25_c2.py, inline scripts)
Read in full: 12 what_this_does_not_say sentences, registered_in, role, git. No sentence is false as written. git block
b5b8a98 / dirty false (matches Amendment C0.4). registered_in lists A, B, B.1, B.2 (not C): right for a G2 build; C changes nothing
in the artifact (NOTE only). C2.2 (seed 303 at maxpressure under att_engine) IS carried: statements[53] reading, statements[27]
numbers.reversal, leaders[3].pooled_leaders_lose_first_place_on_seeds ['303']. B.2.5(1)/(2) ARE carried per block
(p_value_defined / p_value_note on every paired_comparison; the defined note says z's sign is not the direction).

F4.1  MAJOR -- comparison_with_the_record.literal_records[81] (value "q3c.all_against", source BRIEF_27:1249-1250) states as
  this_artifact "every paired difference against the best non-DT arm resolves against the DT", equal: true, exception/explanation null.
  Under Rule R's primary definition that sentence is FALSE: statements[83] (Q3c maxpressure att_engine) reads "NOT RESOLVED",
  CI [-0.1617, +1.9735]; changes[34] marks it changed_by_rule_r. BRIEF_45 C2.1: "That holds under att_ours only."
  The row is true only inside comparison_with_the_record.basis ("as_committed att_ours"), and no field on the row, no exception and no
  what_this_does_not_say sentence says so. This is the one literal row whose sentence changes truth value under the primary
  definition; a paper writer reading "this_artifact ... equal: true" keeps the sentence. (Borderline BLOCKER if `basis` is not
  considered sufficient scoping.)
  Expected: the row carries the primary-definition qualifier, e.g. an "under_att_engine" field "does not hold: at maxpressure the
  difference does not resolve (+0.9059 [-0.1617, +1.9735])", or an exception entry naming Rule R; plus a what_this_does_not_say line.

F4.2  MINOR -- literal_records[114]/[115] ("The 31.5 % reduction", "31.5 % of it is attributable", BRIEF_27:1628/1668) are equal under
  att_ours, but the same quantity under the primary definition is 31.4 %: -I/d1 = 12.567618865354106 / 40.020992837476456 = 31.4026 %
  (att_ours 12.464451821819912 / 39.56485820302637 = 31.5038 %), from statements[0],[1],[4],[5]. The artifact carries no att_engine
  value of this ratio, so a paper writer under Rule R who copies the record writes 31.5 % where the primary value is 31.4 %.
  Expected: the ratio under both definitions, or a flag on the two rows.

F4.3  MINOR -- what_this_does_not_say[0] names only "Q1's, Q1b's and Q2's predictions" as att_horizon-only. Q3b's, Q3c's, Q4b's and
  Q5a-c's registered values are equally att_horizon-era predictions and are printed beside att_engine measurements without a label:
  e.g. statements[83] (Q3c maxpressure att_engine) registered_value 0.4326, statements[111] (Q5a att_engine) 575.34, statements[107]
  Q4b chain registered values 39.5649/41.7402/51.3689/62.7106. predictions_registered_on exists on 8 statements only (Q1, Q1b).
  Expected: the sentence covers every registered value (Q1-Q5), or every att_engine statement with a registered_value carries
  predictions_registered_on.

F4.4  MINOR -- C2.3 is not carried. The four Wilcoxon blocks of Q5a/Q5b at fixedtime (statements[112],[113],[122],[123];
  wilcoxon.n_used = 5, n_zero = 95, p = 0.1775) have p_value_defined true and the generic "two-sided signed-rank test (the normal
  approximation ...)" note; nothing says n_used = 5 is below the n >= 20 the normal approximation is stated for (dt_gate.py:399-402);
  grep for "20"/"399-402"/"small sample" finds nothing. The readings ("does not resolve: the CI includes zero") do not say the two
  policies differ on 5 of 500 episodes; that fact lives only in discriminability (n_differing 5, n_paired 500), and per_seed shows
  3 of 5 seeds exactly 0 (202, 404, 505). Expected: the p_value_note of those four blocks names n_used = 5 < 20, and the reading
  (or a note) says "the two policies differ on 5 of 500 episodes".

### Item 5 -- comparison_with_the_record sub-blocks other than literal_records  [x] done (s26_p84b.py, s27_grid.py, inline)
p8_4b (4 rows): every `recorded` value equals docs/data/p8_4b_rederivation.json contrasts[V4|V5].by_definition[def] (mean, CI,
n_paired, outcome). Every `difference` and `difference_in_ulps_of_the_record` recomputed with math.ulp: V4 att_ours mean -1 ulp,
ci95_low 0, ci95_high -2 ulps; V5 att_ours -1/-1/-1 ulps; att_engine rows 0. All equal the stated values; all within 4 ulps.
p5_1_grid: every recorded value found at docs/data/p5_1_grid.json predictions.P1 / predictions.P3.{dt_nomix,dt_spatial}
(wins/losses/ties/rank_biserial/p/outcome checked field by field); this_artifact == recorded for all.
p5_2_stop_rule_record: recorded == /home/filip/rltraffic/output/p5_2/stop_rule_mappo1000.json (27.10040638120646,
[23.815032830215493, 30.38577993219743], CONTINUE); differences recomputed: -1 ulp on each of the three; stated values equal.
p5_2b_artifact: fields_compared 191 / 175 reproduced = scalar leaves + empty containers of docs/data/p5_2b_iql_correction.json
statements.att_ours.{before,after} (192) / att_engine (176) minus the one excluded field. Consistent.
q5 (3 rows): record quotes found at docs/returns/P5.2.md:30, :572-573 and BRIEF_27:1274-1281; fixedtime registered values +820.65,
-812.15, +8.49 == p5.2.md:684, :686, :689; per_tier, primary and registered_verdicts == statements (Q5a NR, Q5b NR, Q5c FAILED,
both definitions). exceptions (2) == the 8 unequal literal rows. never_scored: see F1 (known Q3c class) -- nothing new beyond F1.1/F1.4.

F5.1  BLOCKER -- comparison_with_the_record.q5[0..2].descriptive_reading.{att_ours,att_engine}.reading = "the global return filter
  collapses where the per-intersection one does not", with tiers_where_q5b_resolves = ["mappo1000", "maxpressure", "random"].
  At random this is FALSE by the artifact's own collapse vocabulary and numbers: descriptive_collapse_audit.rows for
  bc_top10_perix@random vs random@grid4x4_mappo1000: COLLAPSED, +40.824 [+36.818, +44.829] (att_ours) and +41.239 [+37.202, +45.277]
  (att_engine) -- the per-intersection filter is 41 s WORSE than the random policy there. The same artifact's
  sentence_holds[*].non_dt_arms_collapse_on.bc_top10_perix = ["random"], and the README sentence "The DT arms are the only arms that
  never collapse" (holds: true) DEPENDS on bc_top10_perix collapsing at random. The two descriptive readings contradict each other at
  random. True on mappo1000 and maxpressure only (bc_top10 COLLAPSED +489.2 / +419.6; bc_top10_perix NOT COLLAPSED -68.1 / -50.1).
  Q5b resolving at a tier shows perix < top10 there, not that perix does not collapse. The record it paraphrases (BRIEF_27:1279-1280)
  made the claim at maxpressure only ("collapses (679.96 on maxpressure) exactly where the per-intersection filter does not
  (210.27)"). Origin: the sentence is the coordinator's own (BRIEF_45 A3, line 302-303), implemented faithfully -- the error is in
  the prescription, but it is in the artifact.
  Expected: the reading restricted to the tiers where it holds, e.g. tiers_where_it_holds = ["mappo1000", "maxpressure"] computed
  as (bc_top10 COLLAPSED and bc_top10_perix NOT COLLAPSED by the audit's rule), with "at random both filters are worse than the random
  anchor (bc_top10 +90.0, bc_top10_perix +40.8); the per-intersection filter is better than the global one there (Q5b HELD) but
  does not escape collapse" -- or the word "collapses" replaced by "is worse than" if Q5b's resolution is all that is meant.

F5.2  MINOR -- comparison_with_the_record.p8_4b labels V5's row "statement": "Q0a" while its outcome is "CONTINUE"; V5 is P5.2's Q0
  STOP RULE (source name "P5.2 Q0 stop rule: d4 ... CI entirely below 0"); Q0a's own verdict is HELD (statements[2],[3]). Likewise V4
  is labelled "statement": "Q0.d1" with outcome "FAILED", which is P5.1 P2a's outcome (source name "P5.1 P2a: ..."); Q0.d1 carries no
  verdict here (statements[0],[1] verdict null). A reader pairs "Q0a" with "CONTINUE" and "Q0.d1" with "FAILED".
  Expected: statement "Q0.stop_rule (d4, the number Q0a also reads)" for V5 and "Q0.d1 (= P5.1 P2a; the outcome is P2a's)" for V4.

F2.1a (addendum to F2.1, from item 6's sweep) MINOR -- F2.1 is concrete, not hypothetical: descriptive_collapse_audit.rows[12],[13],
  [29],[30] (iql@mappo1000 / iql@maxpressure vs random@grid4x4_mappo1000, both definitions) carry qualifiers reading "reverses on 2 of
  5 training seeds (101, 303)" / "1 of 5 training seeds (202)"; the right arm is the random anchor, whose seed label is an evaluation
  label (definitions.per_seed). Also: sentence_holds[*].non_dt_arms_collapse_on.iql = ["mappo1000", "maxpressure"] is stated without
  those reversals (iql is better than the anchor on seeds 101 and 303 at mappo1000 and on 202 at maxpressure, e.g. -28.2185 on 303).
  Expected: "seed labels (iql's training seed; the anchor's evaluation label)" and the reversal named beside the summary.

### Item 6 -- changes, primary, definition_dependent  [x] done (inline script)
primary == (definition == "att_engine") on all 154 statements, all 10 leaders blocks and all 4 sentence_holds blocks: 0 mismatches.
Every statement group (id, tier, arm, version) is paired across the two definitions; definition_dependent == (verdicts differ, or
readings differ where both verdicts are null) on every pair: 0 mismatches. No pair has the same verdict with different readings,
so no reading change hides behind an unchanged verdict. `changes`: changed_by_rule_r is true only on changes[21] (D9 maxpressure,
reading: seed 303 -> dt_nomix first) and changes[34] (Q3c maxpressure, reading "resolves against the DT" -> "NOT RESOLVED"); both
are reading changes and neither row calls them a verdict change (the rows carry verdict null on both sides). changed_by_the_correction
is false everywhere, and no statement's verdict or reading differs between as_committed and corrected. Matches C1 ("No registered
verdict depends on the definition or on the IQL correction. Two readings change under Rule R"). The block has no wording beyond its
booleans; its only defects are F1.1 (Q3c rows carry no value) and F1.2 (the booleans are undefined, and "false" coexists with a 7.5 s
value change at Q3c random).

## Verdict
FAIL -- 1 BLOCKER (F5.1), 2 MAJOR (F1.1, F4.1), 10 MINOR (F1.2, F1.3, F2.1/F2.1a, F3.1, F3.2, F4.2, F4.3, F4.4, F5.2), 3 NOTE
(F1.4, F3.3, F3.4).
Count correction: the MINOR list above has 9 entries, not 10 (F2.1 and F2.1a are one finding).
FINAL VERDICT LINE: FAIL -- BLOCKER 1 (F5.1), MAJOR 2 (F1.1, F4.1), MINOR 9, NOTE 3.
