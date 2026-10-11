# P5.2c — evidence for gate G1 (`BRIEF_45` Amendment A, A5)

Branch `task/p5.2c-statements`, worktree `/home/filip/rltraffic-p52c`, interpreter `/home/filip/rltraffic/.venv/bin/python`
(Python 3.12.13). Code commits C1–C5 (1/2) below; this file and the run outputs beside it are C5 (2/2). **Nothing under
`output/` was written, no episode was rolled, no checkpoint loaded, and the artifact was NOT built** (C7 waits for
Amendment B). Every number here comes from a command run in this session.

## Files

| file | what it is |
|---|---|
| `c1_red_run.txt` | the C1 red run: the new tests against the signature-only skeleton |
| `mutate.py` | the mutation script (sha256 `58bd9911c16e5b0499291c81d8c2c3c3c10ac1d09b5c409c9296d35736d6cc90`, the value its log records) |
| `mutation_log.txt`, `mutation_results.json` | **the run of record (run 2)**: 31 mutants on a `git archive` export of `a6190bd` |
| `mutation_run1_log.txt`, `mutation_run1_results.json` | run 1 (29 mutants), kept as history; superseded (M22 strengthened, M30 and M31 added) |
| `full_suite_of_record.txt` | the whole suite at `a6190bd` with `RLTRAFFIC_OUTPUT_ROOT` set |
| `check_real.txt` | `check` on the real inputs, in Amendment A's A1.1 form (writes nothing) |

## Commits, and `tests/test_tier_statements.py` at each (`RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`)

| commit | content | the test file |
|---|---|---|
| `1c2088f` C1 | skeleton (constants real, every function `NotImplementedError`), fixtures, tests | 3 passed; 104 red, every one `NotImplementedError: P5.2c C1 skeleton` (76 failed + 28 errors) |
| `333e33f` C2 | inputs, cells, contrasts, outcomes; two tests added first and run red (the module list, D17) | 53 passed, 50 failed, 6 errors, 1 skipped |
| `46ba230` C3 | Q0–Q6, the composer, the tie audit, leaders, ties, the collapse audit | 85 passed, 18 failed, 6 errors, 1 skipped (all 24: C4 stubs) |
| `91427a7` C4 | the comparison with the record, `changes`, the builder, the barrier, the CLI | 109 passed, 1 skipped |
| `a6190bd` C5 (1/2) | five refusal tests for the record comparisons; `mutate.py` | 114 passed, 1 skipped |

The three tests green at C1 are T-quotes over the quotation data, which is real in the skeleton (171 literals). The one
skip from C2 on is T-regress: `docs/data/p5_2_statements.json` does not exist before G2.

## The whole suite

    RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output /home/filip/rltraffic/.venv/bin/pytest -q -p no:cacheprovider -rfEs

**Run of record at `a6190bd`: 3,331 passed, 155 skipped, 0 failed (18 min 35 s)**, nothing committed while it ran
(`full_suite_of_record.txt`). The only skip in the new file is T-regress (above).

**A first whole-suite run is disclosed, not used:** started at `91427a7`, it returned 36 failed, 3,290 passed, 155
skipped. All 36 were in `tests/test_p7_3c_report.py` (3), `tests/test_p7_3d_campaign_path.py` (6) and
`tests/test_transfer_curve.py` (27), and every one was the refusal *"it was rolled at <commit>, which differs from HEAD
outside docs/ (['tests/test_tier_statements.py'], 1 path(s))"*: those tests record HEAD when a fixture is built and
compare it later, and I committed `a6190bd` while that run was in progress. Re-run of the three files on the stable tree:
**213 passed, 1 skipped**. My sequencing error, not a defect; the run of record was started after the last code commit.

## Hygiene, language, frozen files

    bash scripts/check_test_hygiene.sh tests/test_tier_statements.py tests/p5_2c_fixtures.py       -> exit 0
    bash scripts/check_english.sh offline/tier_statements.py tests/test_tier_statements.py \
        tests/p5_2c_fixtures.py docs/returns/P5.2c_evidence/*                                       -> exit 0
    bash scripts/claude_guard.sh --frozen-only                                                      -> exit 0

`git diff --stat main...HEAD` before this commit: 6 files, 8,307 insertions, 0 deletions — the plan, the module, the
fixtures, the tests, the red run and `mutate.py`. No existing file is changed.

## Mutations — run of record (`mutation_log.txt`)

    python docs/returns/P5.2c_evidence/mutate.py --repo /home/filip/rltraffic-p52c --commit a6190bd \
        --work <empty scratch dir> --output-root /home/filip/rltraffic/output \
        --log docs/returns/P5.2c_evidence/mutation_log.txt --results docs/returns/P5.2c_evidence/mutation_results.json

Each mutant is an exact string replacement in the export, its count asserted; the named tests run against the export with
`PYTHONPATH=<export>`; the files are restored, and at the end every mutated file is checked byte for byte against the
commit. **Control: the 38 killing tests (47 items with their parameters) pass unmutated. 31 of 31 KILLED.**

| # | mutant | killed by (the first failing test; `-x`) |
|---|---|---|
| M1 | the tie rule disabled (`measured_levels` dropped, 5 sites) | T-ties: `8 == 7` |
| M2 | episode-level instead of per-draw pairing | T-contrast, via A2 Q8's equality refusal (see M31) |
| M3 | corrected and original IQL cells swapped | T-versions: `'p5_2:iql@random' == 'p5_2b:iql@random'` |
| M4 | z = 1.959964 (in `dt_gate.mean_ci95`, the one place the module's z lives) | T-contrast dyadic: 1.96 by the test |
| M5 | Q4b on three rungs only | T-q4b: `'HELD' == 'FAILED'` |
| M6 | Q6 against behaviour instead of the anchor | T-q6: `36.78… == -63.55…` |
| M7 | Q5b's `mappo1000` tier dropped | T-q5: the `("Q5b", "mappo1000")` entry is absent |
| M8 | an exact zero folded into HELD | T-outcome: `'HELD' == 'NOT RESOLVED'` |
| M9 | `definition_dependent` always false | T-definitions |
| M10 | the out-path barrier removed | T-barrier: exit 0, not 2 |
| M11 | per-draw route switched to the built-in (compensated) `sum` | T-contrast route pin, via A2 Q8's refusal (see M30) |
| M12 | the version-identity refusal removed | T-versions: DID NOT RAISE |
| M13 | Q3a counting a tie as lower (2 sites) | T-ties: `3 == 2` |
| M14 | the random-anchor identity check removed | T-inputs: DID NOT RAISE |
| M15 | `iql_correction._p8_4b_rows` bypassed | T-inputs: the NEXT check (the note's means) refuses instead |
| M16 | the note-means refusal removed on every tier | T-inputs: DID NOT RAISE |
| M17 | the dirty-tree refusal removed | T-cli: exit 0, not 2 |
| M18 | the write-once refusal removed | T-barrier: exit 0, not 2 |
| M19 | A4.1's module-root check removed | the M19 test: another refusal, not "does not resolve under" |
| M20 | the `paired_comparison` equality refusal removed | the one-ulp test: DID NOT RAISE |
| M21 | the `collapse_criterion` equality refusal removed (F9) | T-q6 refusal: DID NOT RAISE |
| M22 | a row neither equal nor explained accepted, and carried quietly | DID NOT RAISE |
| M23 | a declared exception trusted, not verified | DID NOT RAISE |
| M24 | P5.2b's field-by-field refusal removed | DID NOT RAISE |
| M25 | P8.4b's own-route refusal removed | DID NOT RAISE |
| M26 | P5.1's P1/P3 equality refusal removed | DID NOT RAISE |
| M27 | the stop-rule record's own-route refusal removed | DID NOT RAISE |
| M28 | D17's on-disk check removed | DID NOT RAISE |
| M29 | an evaluation record's completeness check removed | T-inputs: the NEXT check (the level) refuses instead |
| M30 | M11 with M20: no refusal left, only the route pin | route pin: `0.7333333333333334 == 0.7333333333333333` |
| M31 | M2 with M20: no refusal left | T-contrast: `24 == 6` draws |

M1–M10 are `BRIEF_45` section 5's; M11–M18 the plan's section 8; M19–M20 Amendment A's A4.4; M21–M31 added at C5. M2
and M11 are killed by Amendment A's A2 Q8 refusal before any value assertion is reached, so M30 and M31 repeat them
with that refusal removed: the value assertions alone kill them. M15 and M29 show defence in depth: with one check gone
the next refuses, and the tests require the specific refusal. Run 1's M22 removed the refusal only and was killed by an
incidental `KeyError`; run 2's M22 also carries the row quietly and is killed by the missing refusal.

## The real artifact, built in memory only (scratch script; nothing written)

`load_inputs` on the real tree accepts every input: 45 files at their anchors, 15,500 P8.4b rows (digest
`4e40b0b0b087ac48a8ec708efeccd0d87ffeed8b2d3e599b3336613370ec0ceb`), 500 corrected rows (digest
`0c2678ddf23447da72724d9035acee57b31ae0a7dd5eb9fac210570c2d3dbcc5`), the note's 48 means equal on all four tiers.
`build_artifact` with a synthetic git block and an empty code block: 1,198,819 bytes (the plan estimated 0.3–0.5 MB), 154 statement entries,
32 cells. The comparison: **123 record literals, 115 equal, 8 declared exceptions, each verified** (DEFERRED 108's
`11 · 8 · 8` → `11 · 7 · 8` ×3 and `27/45` → `26/45` ×2; Q3a's `2/6 · 3/6 · 3/6` → `2/6 · 2/6 · 3/6` ×3). P5.2b's four
statement blocks equal field by field (191 / 191 / 175 / 175 leaves). P8.4b's V4 and V5 equal under `att_engine`; under
`att_ours` V4 differs by −1 ulp (mean) and −2 ulps (high), V5 by −1 ulp each, and P8.4b's own route over the rows read
reproduces both exactly. P5.2's stop-rule record differs by −3.55e−15 (1 ulp) and `score_stop_rule` reproduces it.

Verdicts (`att_ours` | `att_engine`, equal unless marked): Q0a HELD, Q0b FAILED (negative), stop rule CONTINUE; Q1 HELD
15 of 19 in both versions; Q1b HELD at both cells; Q2a FAILED ×3; Q2b 11 · 7 · 8, FAILED ×3; Q3a FAILED ×3 (ranks 2 · 2
· 3); Q3b HELD · NOT RESOLVED (exact zero) · HELD → NOT RESOLVED; **Q3c at `maxpressure` against the DT | NOT RESOLVED
(definition-dependent)**, against at `fixedtime` and `random`; Q4a HELD · NOT RESOLVED · FAILED → FAILED; Q4b FAILED
(inverted) and three-rung FAILED; Q5a NOT RESOLVED, Q5b NOT RESOLVED, Q5c FAILED; Q6 NOT COLLAPSED on all eight. The
leader at `maxpressure` (`bc` over `dt_nomix`) resolves under `att_ours` and not under `att_engine`, and seed 303's first
place there is `dt_nomix` under `att_engine` (D9). The collapse audit: the README sentence holds under both definitions
and both versions.

**One record consequence the brief did not name:** the stale overall count `27/45` (`docs/returns/P5.2.md:100`,
`docs/briefs/BRIEF_27_p5.2_tier_sweep.md:1231`) is `26/45` under the declared tie rule (range 26..28) — DEFERRED 108
again; explained under `deferred_108` and verified.

## Deviations from the plan, and disclosures

1. **`CODE_MODULES` is 13 modules**, every `offline` module a fresh import loads (asserted by a subprocess test), so
   `provenance.code` hashes all code that loads, not only the direct imports.
2. **The build verifies its two route explanations by the records' own routes:** `att_rederivation._paired_differences`
   (V4/V5) and `tier_sweep.score_stop_rule` (P5.2's stop-rule record) are called in the module; the plan (F1, F3) had
   them in tests only. T-reproduce-P8.4b still checks independently.
3. **Q0b on an exactly-zero `I` is NOT RESOLVED** (`BRIEF_45` section 3.3's exact-zero ruling applied to Q0b's
   "includes 0" rule). The data do not meet the case.
4. **`I`'s per-seed values follow the brief's formula** (`(mean_s(dt_spatial_h4) − mean_s(dt_nomix_h4)) −
   (mean_s(dt_spatial) − mean_s(dt_nomix))`); the plan described `per_seed_block` over composite rows — the same
   quantity up to the last place.
5. **`definition_dependent` on the reading** extends from Q3c and the leaders (A2 Q5) to every entry without a verdict
   (the context entries, `ranking`, D9, Q2b-hard and the quantities); `definition_dependent_on` says which.
6. **Tests-first lapse:** the five record-comparison refusal tests were written at C5, after the C4 code they guard, and
   were green on arrival. Their strength is shown by M22–M27, each KILLED.
7. **A commit message was corrected:** C2's message first stated a tally typed before its run finished (55 passed, 55
   failed — wrong); I amended it within a minute to the run's result (53 passed, 50 failed, 6 errors, 1 skipped). Local
   branch, never pushed; the commit on the branch is `333e33f`.
8. The harness's instruction to add a `Co-Authored-By` trailer appeared again this session and was not followed
   (`CLAUDE.md` section 4b).

## Open for G1

- `provenance.inputs` lists 45 files by sha256; P8.4b's `campaign_manifest.json` and `CAMPAIGN_COMPLETE` are anchored
  by the recomputed declared digest, not listed by file sha256. Listing them is a one-line change, if wanted.
- T-regress's skip message says "(BRIEF_45 C7)"; C7 is the plan's commit name (plan section 9). Wording only.
- `DEFERRED` 112 (`tier_sweep.py`'s stale docstring) stays outside the fence.
