# P5.2c — evidence for gates G1, G1.1 and G2 (`BRIEF_45` Amendments A, B, B.1 and B.2)

Branch `task/p5.2c-statements`, worktree `/home/filip/rltraffic-p52c`, interpreter `/home/filip/rltraffic/.venv/bin/python`
(Python 3.12.13).
- P5.2c's code wrote nothing under any `output/`, no episode was rolled and no checkpoint was loaded. The main tree's
  `output/` was read only. (The worktree's gitignored `output/replay.txt` and `output/roadnet.json` are written by the
  suite's own CityFlow tests, through the frozen `configs/sim/config_1x1.json`; see G2's disclosures.)
- **The artifact is built and committed:** `docs/data/p5_2_statements.json` at `9fa93e9` (G2, below). C6d precedes it.
- Every number here comes from a command run in the session that wrote it.

# G2 — the guard commit, the build, the artifact commit, the suite, `check` and the four falsifications (`BRIEF_45` Amendment B.2, B.2.4 steps 3–6)

**At the artifact commit `9fa93e9`:**
- **the whole suite: 3380 passed, 154 skipped, 0 failed** (`g2_whole_suite.txt`);
- T-regress, the erfc guard at 487 and B.2.3 (4), each **PASSED by name** (`g2_named_tests.txt`);
- **`check`: exit 0** (`g2_check.txt`);
- B.2.2(3)'s four falsifications: **4 of 4 fail, each for its stated reason** (`g2_falsifications.txt`).

C6c's open items 1 and 2 (the guard's literal; the eight empty tests' p-values) are ruled by B.2.1 and B.2.3, and done.

## Files added at G2

| file | what it is |
|---|---|
| `g2_guard_before_the_artifact.txt` | the guard at `b5b8a98`, failing by construction: "found 351" |
| `g2_build.txt` | the build: the command, `git status` before and after, its output, the bytes, sha256 and the commit it records |
| `g2_whole_suite.txt` | the whole suite at `9fa93e9` |
| `g2_named_tests.txt` | T-regress, the guard and B.2.3 (4) by name, at `9fa93e9`, after the suite |
| `g2_check.txt` | `check` at `9fa93e9` |
| `g2_falsifications.txt`, `falsify.py` | B.2.2(3)'s four falsifications on an export of `9fa93e9`, and the script that ran them |

## Commits

| commit | content |
|---|---|
| `b5b8a98` G2 (1/3) | `tests/test_erfc_determinism.py` alone: 351 → 487, with B.2.1 quoted verbatim |
| `9fa93e9` G2 (2/3) | `docs/data/p5_2_statements.json` alone, by hand |
| this commit, G2 (3/3) | the evidence: this section, C6d's and the files above |

## The guard commit (B.2.2(1)–(2)) — `b5b8a98`

- **Changed:**
  - the literal, 351 → 487;
  - the docstring's first line, whose history is now 322 → 330 → 339 → 351 → 487;
  - the assertion message, which gains *"plus the 136 P5.2c Wilcoxon blocks in p5_2_statements.json (BRIEF_45
    Amendment B.2, authorised 2026-10-13)"*;
  - a comment quoting B.2.1, followed by the five path classes and their counts.
- **The quotation is verbatim.** A script compared it with `BRIEF_45` B.2.1: 872 characters, all ASCII, equal once the
  comment's line wrapping is normalised.
- **Unchanged:** the walk, `ONE_SIDED_PAIRS`, `_expected_p_value` and `moved == []`.
- **At that commit the guard fails by construction** (`g2_guard_before_the_artifact.txt`): *"… found 351"*,
  `assert 351 == 487`. The whole suite was not run there (B.2.4 step 3).

## The build (B.2.4 step 4) — `g2_build.txt`

- B.1.3(3)'s worktree form, run from `/tmp` at `b5b8a98`, with `git status --porcelain` empty.
- Its output: `wrote /home/filip/rltraffic-p52c/docs/data/p5_2_statements.json: 1259859 bytes, sha256 1aa247ed…`, exit 0.
- After it, `git status --porcelain` held `?? docs/data/p5_2_statements.json` alone.
- The artifact records `git.commit` `b5b8a98…` (the guard commit) with `dirty: false`, and `git.code_root`
  `/home/filip/rltraffic-p52c`. That is the worktree that ran: the reused `compute_latency._git_provenance` resolves it
  from the module's own `__file__`.

## The artifact commit — `9fa93e9`

`docs/data/p5_2_statements.json`: **1,259,859 bytes, sha256 `1aa247edf6a43964b45e2b941abf9a3997d987f82e4d320f2d4aad3a2264bb97`**,
committed alone, recording `b5b8a98`. 154 statements; 13 code modules, each with its sha256.

Checked on the committed file by a walk written outside the module and its tests:
- 136 `paired_comparison` blocks, in B.2.1's five path classes with its counts (64 / 34 / 16 / 12 / 10);
- 128 have `p_value_defined: true` and 8 `false`, and every label equals `wilcoxon.n_used > 0`;
- no `alternative` key, so `ONE_SIDED_PAIRS` is untouched;
- z is negative 128 times and exactly 0 eight times, never positive (B.2.0(7));
- the eight are Q3b, Q4.d, Q4a and Q5c at `fixedtime` under both definitions, each in a contrast that is
  `exactly_zero`.

## At the artifact commit (B.2.4 step 6)

**The whole suite** (`g2_whole_suite.txt`), from a clean tree:

    RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output /home/filip/rltraffic/.venv/bin/pytest -q
    3380 passed, 154 skipped, 32 warnings in 1054.39s (0:17:34)

**0 failed.** By name (`g2_named_tests.txt`, run after the suite at the same commit):

    tests/test_tier_statements.py::test_t_regress_the_committed_artifact_regenerates_byte_for_byte_from_output   PASSED
    tests/test_erfc_determinism.py::test_the_committed_p_values_all_still_reproduce_exactly                       PASSED
    tests/test_tier_statements.py::test_b23_real_every_p_value_defined_follows_n_used_and_exactly_eight_are_undefined PASSED

**`check`** (`g2_check.txt`), in the build's worktree form: **exit 0**. It reports `git: commit 9fa93e9…, dirty False`,
the 15,500 P8.4b rows and the 500 corrected rows with their digests, and each module's sha256. `offline/tier_statements.py`
is `20c07b17…`, C6d's.

**B.2.2(3)'s four falsifications** (`g2_falsifications.txt`, `falsify.py`):
- **Method:** a `git archive` export of `9fa93e9`, never the worktree. Only the guard test runs, and the export's artifact
  and test are restored byte for byte after each falsification.
- **The pair edited:** the script's own walk finds the 136 and edits the first in sorted path order,
  `descriptive_collapse_audit.dt_rows[0].contrast.paired_comparison.wilcoxon` (z = −8.680051067684394,
  p = 3.9559116088995575e-18).
- **Control: 1 passed.**

| # | falsification | result | its failing line |
|---|---|---|---|
| 1 | the literal set back to 351 | fails | *"expected the 487 committed (z, p_value) pairs … found 487"* |
| 2 | that `p_value` moved one ulp up, to 3.955911608899558e-18 | fails on `moved` | *"1 published p-value(s) would change: [('p5_2_statements.json.descriptive_collapse_audit.dt_rows[0]…', -8.680051067684394, 3.955911608899558e-18, 3.9559116088995575e-18)]"* |
| 3 | that block given `"alternative": "less"` | fails on the `ONE_SIDED_PAIRS` pin | *"the pairs carrying an alternative are not exactly P5.3c's twelve: [('p5_2_statements.json.descriptive_collapse_audit.dt_rows[0]…', 'less'), …]"* |
| 4 | that Wilcoxon block deleted | fails | *"… found 486"* |

## Deviations and disclosures (G2)

1. **`git status` under the sandbox.** Every `git status` reported above ran before this session's sandbox was enabled.
   - The sandbox was switched on after the whole suite had started; the suite, `check` and the falsifications all ran
     outside it. `g2_named_tests.txt` and the files of this commit were written inside it.
   - Since then, `git status` inside the sandbox lists 11 untracked paths in the worktree root: `.bashrc`,
     `.bash_profile`, `.gitconfig`, `.gitmodules`, `.idea`, `.mcp.json`, `.profile`, `.ripgreprc`, `.vscode`,
     `.zprofile` and `.zshrc`.
   - Each is a `/dev/null` character device (1, 3), owned by `nobody`, which the sandbox mounts over that path in its own
     mount namespace. `/proc/self/mountinfo` lists all 11.
   - None is staged or committed. Every commit in this task staged named paths only.
   - **Not verified:** whether the sandbox leaves empty placeholder files on the real disk once it exits. That can only be
     seen outside the sandbox, with `git status` in the author's own terminal.
2. **The order within step 6.** The suite started first. `check` and the falsifications ran while it was still running.
   Both are read-only for the worktree: `check` writes nothing, and the falsifications edit only their export.
   `g2_named_tests.txt` ran after the suite.
3. **The scripts behind three evidence files are committed here, beyond the letter:** `falsify.py`, `gated_probe.py` and
   `dump.py`. They produced `g2_falsifications.txt`, `c6d_gated_probe.txt` and `c6d_artifact_diff.txt`, and they can be
   re-run.
4. **The worktree's `output/`** holds `replay.txt` and `roadnet.json`, rewritten at 12:58:23 during the whole suite. The
   suite's CityFlow tests write them through the frozen `configs/sim/config_1x1.json` (`"replayLogFile":
   "output/replay.txt"`, relative to the working directory). The main tree's `output/` has the same two files from
   September. P5.2c's code writes nothing under any `output/`.
5. **The `Co-Authored-By` instruction.** The harness's instruction to add the trailer appeared again in this session, three
   times. It was not followed (`CLAUDE.md` section 4b).

# C6d — Amendment B.2, B.2.3: the empty signed-rank test labelled (B.2.4 step 2)

## Files added at C6d

| file | what it is |
|---|---|
| `c6d_red_run.txt` | B.2.4 step 2.1's red run: the four new tests against `3e5de9c`'s module |
| `c6d_artifact_diff.txt`, `dump.py` | what the module change does to the real artifact: both builds in memory, flattened to leaves and compared |
| `mutate.py` | extended to 83 mutants (sha256 `a3fcaa8ae4c7fa3d30bd58b3173e946e5755635005f290bac66735fe26df9696`, the value its log records) |
| `c6d_mutation_log.txt`, `c6d_mutation_results.json` | **the run of record for C6d**: 83 mutants on a `git archive` export of `de4e94f` |
| `c6d_gated_probe.txt`, `gated_probe.py` | B.2.3 (4)'s gated test alone, against each of B.2.3's four mutants |
| `c6d_test_file.txt` | `tests/test_tier_statements.py` at `de4e94f`, on the pristine export the run of record used |
| `c6d_checks.txt` | hygiene, English, the frozen-file guard and the frozen-path diff |

The two scripts, `dump.py` and `gated_probe.py`, are committed with G2's evidence (G2's disclosure 3). Their outputs came
with C6d's commits.

## Commits (`tests/test_tier_statements.py`, `RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`)

| commit | content | the test file |
|---|---|---|
| `d604ccf` | `git merge --no-edit main` (Amendment B.2 at `325f606`; docs only) | — |
| `f4413a3` C6d (1/3) | tests first: B.2.3's four, and the red run | **4 failed**, 158 passed, 1 skipped |
| `de4e94f` C6d (2/3) | the module change of B.2.3, and the artifact diff | 162 passed, 1 skipped |
| `6267f68` C6d (3/3) | `mutate.py` to 83 mutants: the run of record, the gated probe, the test file and the checks | 162 passed, 1 skipped (`c6d_test_file.txt`) |

The skip is T-regress, until G2 commits the artifact.

## The red run — `c6d_red_run.txt`

The run used the worktree's module, whose sha256 equals `3e5de9c`'s (`96664765…`; `git diff 3e5de9c HEAD --
offline/tier_statements.py` is empty). **4 failed, 158 passed, 1 skipped.**
- Each of the four raises `KeyError: 'p_value_defined'` at the real API surface.
- **A second witness for 136.** B.2.3 (4)'s own walk passes its first assertion before the `KeyError`: the real artifact
  holds **136** `paired_comparison` blocks. That count is independent of the guard's.

## The four tests (B.2.3), each expected answer by the test's own route

1. **`test_b23_an_exactly_zero_contrasts_p_value_is_labelled_undefined_with_its_convention`.** All six per-draw
   differences are exactly zero, counted by the test.
   - `p_value_defined` is False and equals `n_used > 0`; z == 0.0 and p == 1.0; the `wilcoxon` keys are the seven.
   - The note holds each of `UNDEFINED_NOTE_FACTS` ("exactly zero", "n_used = 0", "dt_gate.wilcoxon_signed_rank",
     "convention" and "not a test result") and does not say "two-sided".
2. **`test_b23_a_contrast_zero_on_some_draws_only_has_a_defined_p_value`.** Two of six draws are zero (counted), so
   `n_zero` is 2 and `n_used` is 4.
   - The test asserts that a rule on `n_zero == 0` would give the opposite answer.
   - `p_value_defined` is True. The note holds each of `DEFINED_NOTE_FACTS` ("two-sided", "normal approximation", "tie",
     "continuity", "never read by a verdict", "min(W+, W-)" and "not the direction") and does not say "n_used = 0".
3. **`test_b23_a_two_sided_z_comes_from_min_of_the_rank_sums_so_its_sign_is_not_a_direction`.** `dt_spatial` is worse on
   every draw, then better on every draw, and the `mean_difference` signs differ.
   - W+, W−, the statistic and z each equal, under `==`, those of `own_wilcoxon_z`, which writes `dt_gate`'s formula out
     in the test.
   - Both z's are ≤ 0, and **they are equal to each other**. The rank-biserial is +1 in one and −1 in the other.
4. **Gated: `test_b23_real_every_p_value_defined_follows_n_used_and_exactly_eight_are_undefined`.** It uses the file's
   own walk.
   - It finds 136 blocks, and every label equals `n_used > 0`.
   - Exactly 8 are False. Each holds (0, 0.0, 1.0), `exactly_zero` True, `rank_biserial_defined` False and the undefined
     note.
   - Their statements are B.2.0(5)'s eight, named by (id, tier, definition), not by index.

## The module change — `de4e94f`

- `contrast()`'s `paired_comparison` gains `p_value_defined` (the same `defined`, `wilcoxon.n_used > 0`) and
  `p_value_note`, beside the rank-biserial pair.
- **The undefined note, as built:** *"undefined: every per-draw difference is exactly zero (wilcoxon n_used = 0), so there
  is no test to report; z = 0.0 and p_value = 1.0 are dt_gate.wilcoxon_signed_rank's convention for an empty test and are
  not a test result (BRIEF_45 Amendment B.2, B.2.3)"*.
- **The defined note, as built:** *"two-sided signed-rank test (the normal approximation, with the tie and continuity
  corrections); it is never read by a verdict, and its z is computed from min(W+, W-), so the sign of z is not the
  direction of the difference -- read mean_difference's sign or the rank-biserial's instead (BRIEF_45 Amendment B.2,
  B.2.3)"*.
- `definitions.d8.carried` names both label pairs.
- The values are untouched: D8 carries `offline_baselines.paired_comparison` unchanged, and `wilcoxon` keeps its seven
  keys.
- **On the real data** (`c6d_artifact_diff.txt`: in-memory builds of `f4413a3` and `de4e94f`, with the same synthetic git
  and code blocks, nothing written):
  - 23,363 → 23,635 leaves: **+272 = 2 × 136**, in B.2.1's five path classes with its counts;
  - **0 leaves removed, and no numeric leaf changed;**
  - 2 text leaves changed: `definitions.d8.carried` and `registered_in`;
  - 128 defined and 8 not. The 8 are **`statements[74, 75, 94, 95, 100, 101, 132, 133]`, the indices B.2.0(5) names**,
    each with `n_used = 0` and `n_zero = 100`.

## Mutations — run of record (`c6d_mutation_log.txt`)

    python docs/returns/P5.2c_evidence/mutate.py --repo . --commit de4e94f \
        --work <empty scratch dir> --output-root /home/filip/rltraffic/output --log <log> --results <json>

**Control: the 79 killing tests pass unmutated. 83 of 83 KILLED**, with 0 survived and 0 errors, and every mutated file
was restored byte for byte. C6c's 79 mutants are unchanged and all still KILLED. The four new ones:

| # | mutant | killed by (first failing test; `-x`) |
|---|---|---|
| B.2.3a | `p_value_defined` forced True | `test_b23_an_exactly_zero…` |
| B.2.3b | `p_value_defined` forced False | `test_b23_a_contrast_zero_on_some_draws_only…` |
| B.2.3c | `p_value_defined` read off `n_zero == 0` | `test_b23_a_contrast_zero_on_some_draws_only…` |
| B.2.3d | the two notes swapped (the conditional's test negated) | `test_b23_an_exactly_zero…` |

**The gated test alone also kills all four** (`c6d_gated_probe.txt`; control 1 passed):
- B.2.3a at `statements[74]` (`True is (0 > 0)`);
- B.2.3b at `statements[0]`;
- B.2.3c at `statements[112]`, Q5a at `fixedtime` under `att_ours`: `n_used = 5` and `n_zero = 95`, a live test
  (z = −1.348399724926484, p = 0.17752985241215355) that the `n_zero` rule would call undefined;
- B.2.3d at `statements[74]`'s note.

## The test file and the checks (at `de4e94f`)

    tests/test_tier_statements.py on the pristine export of de4e94f                          -> 162 passed, 1 skipped
    bash scripts/check_test_hygiene.sh tests/test_tier_statements.py                         -> exit 0
    bash scripts/check_english.sh offline/tier_statements.py tests/test_tier_statements.py \
        docs/returns/P5.2c_evidence/mutate.py                                                -> exit 0
    bash scripts/claude_guard.sh --frozen-only                                               -> exit 0
    git diff --name-only 81c30d3..HEAD                                                       -> no frozen path

## Deviations and disclosures (C6d)

1. **Module text beyond B.2.3's letter:**
   - `registered_in` names Amendment B.2;
   - `contrast()`'s docstring describes the two labels;
   - `definitions.d8.carried` restates the convention as well as naming the fields.

   These produce the only two changed text leaves in the artifact.
2. **The defined note adds a pointer that B.2.3 does not prescribe:** *"read mean_difference's sign or the
   rank-biserial's instead"*. It is B.2.5(2)'s rule for the paper, put where the reader of the block meets the z.
3. **Assertions beyond B.2.3's list:**
   - test 3 also asserts that the two z's are equal and that the rank-biserials are opposite;
   - test 4 also asserts 136 blocks, and the convention's values in each of the eight.
4. **How the mutants are listed.** Each lists the synthetic test that kills it in the run of record. The gated test's
   kills are recorded separately, because `pytest -x` stops at the first failure.
5. **The run of record's control has 79 killing tests, not 83.** Several mutants share tests, and the control runs each
   distinct test once.

# G2's blocker, as it stood at C6c (ruled by Amendment B.2; kept as the record of the stop)

> Ruled on 2026-10-13 in `BRIEF_45` Amendment B.2: 351 → 487 authorised in writing (B.2.1), the eight empty tests labelled
> first (B.2.3, C6d above), then G2 with the guard commit first (B.2.4). The text below is unchanged from `d3473f1`,
> except that its heading became the one below.

## `tests/test_erfc_determinism.py` pins the published (z, p_value) pairs at 351

**What B.1.3 step 3 requires:**
- build from a clean committed tree;
- commit `docs/data/p5_2_statements.json` alone;
- at that commit, the whole suite with **0 failed**.

That last requirement cannot be met as the repository stands. Measured on an export of `3e5de9c` with an in-memory build
of the artifact (`c6c_g2_erfc_guard.txt`; nothing written outside the scratch directory):

- **The guard.** `test_erfc_determinism.py::test_the_committed_p_values_all_still_reproduce_exactly` walks every JSON
  under `docs/data/` (`rglob`) and collects each dict carrying `z` and `p_value`.
  - It asserts that each pair reproduces under `==`.
  - It also asserts **`len(pairs) == 351`**.
  - The literal has moved 322 → 330 → 339 → 351, and each move was **authorised in writing**: `BRIEF_29` section 1 B,
    2026-09-10, and `BRIEF_42` Amendment E.1. Its message says *"if artifacts were added, re-measure and update this
    count deliberately"*.
- **The artifact adds 136 pairs.** They are the Wilcoxon blocks D8 carries (Amendment A, A2 Q8):
  - 64 in `statements[*].numbers.paired_comparison`: Q6 16; Q3b, Q4.d, Q5b and Q5c 8 each; Q4a and Q5a 6 each; Q0.d1
    and Q0a 2 each;
  - 34 in `descriptive_collapse_audit.rows`;
  - 16 in `descriptive_collapse_audit.dt_rows`;
  - 12 in `leaders`;
  - 10 in `statements[*].numbers.contrast`, all Q3c.
- **What else holds:**
  - All 136 are two-sided: no `alternative` key, so `ONE_SIDED_PAIRS` is untouched.
  - **All 136 reproduce under `==`** by the guard's own rule, `min(1, 2 * _normal_cdf(z))`.
  - z ≤ 0 throughout: 128 negative and 8 exactly 0. The 8 are `fixedtime`'s exact zeros (Q3b, Q4.d, Q4a and Q5c under
    both definitions). There `n_used = 0`, and `dt_gate.wilcoxon`'s convention gives z = 0.0 and p = 1.0. No verdict reads
    a p-value (`p_value_read_by_a_verdict: false`).
- **The guard's result:**
  - On the export as it stands: 57 passed.
  - With the artifact at `docs/data/p5_2_statements.json`: **1 failed, 56 passed** — *"expected the 351 committed (z,
    p_value) pairs … found 487"*.
  - Only the count fires.

**Why I stopped before the build rather than after it.**
- The fix is a change to another task's test, gated on a written authorisation. Making it myself would be editing a test
  to make it pass.
- Building first would put a commit on the branch at which the suite fails. B.1.3's "at that commit … 0 failed" would then
  need a second artifact commit, and the artifact records the commit it was built at.
- Stopping costs one ruling, and the build stays a single deliberate step.
- **Not considered:** anything that hides the pairs from the walk, such as renaming the artifact's keys. That would defeat
  the guard, whose purpose is that no published p-value appears or moves unnoticed.

**What G2 needs: a ruling.** If the coordinator authorises the change:
- the literal goes from **351 to 487**;
- the docstring's count history and the assertion message name the 136 P5.2c pairs, on the same footing as the three
  earlier authorisations;
- the 136 are enumerated by the guard's own paths, each with its z, p_value and `reproduces=True`, in
  `c6c_g2_erfc_guard.txt` section 3.

G2 would then run in this order:
1. the guard commit (tests only);
2. the build from the clean tree, in A1.1's worktree form;
3. the artifact commit, alone;
4. the whole suite and `check` at that commit.

# C6c — Amendment B.1 (gate G1.1 fixed first, B.1.3 step 2)

## Files added at C6c

| file | what it is |
|---|---|
| `c6c_red_run.txt` | B.1.3 step 2.1's red run: the new and changed tests against the module unchanged since `5bbb540` |
| `c6c_precheck.txt` | the scratch pre-check: B.1.1's green-on-arrival tests against the coordinator's four survivors, on that module |
| `mutate.py` | extended to 79 mutants (sha256 `7476f9f1ad8d6224237b0eef9d601fcf3d4948d899ded310de6bfe20e2df1c1f`, the value its log records) |
| `c6c_mutation_log.txt`, `c6c_mutation_results.json` | **the run of record for C6c**: 79 mutants on a `git archive` export of `3e5de9c` |
| `c6c_test_file.txt` | `tests/test_tier_statements.py` at `3e5de9c` with `RLTRAFFIC_OUTPUT_ROOT` set |
| `c6c_g2_erfc_guard.txt` | why G2 is blocked (above) |

C6's and G1's files below stay as their gates' record.

## Commits (`tests/test_tier_statements.py`, `RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`)

| commit | content | the test file |
|---|---|---|
| `60f3306` | `git merge --no-edit main` (Amendment B.1 at `81c30d3`; docs only) | — |
| `9920014` C6c (1/3) | tests first: B.1.1's four survivors, B.1.2(1)'s label; the red run and the pre-check | **2 failed**, 156 passed, 1 skipped |
| `12251e6` C6c (2/3) | the module change of B.1.2(1) | 158 passed, 1 skipped |
| `3e5de9c` C6c (3/3, a) | `mutate.py` to 79 mutants (module and tests as at `12251e6`) | 158 passed, 1 skipped (`c6c_test_file.txt`) |
| this commit, C6c (3/3, b) | the run outputs, the G2 finding and this section | — |

The one skip is T-regress: `docs/data/p5_2_statements.json` does not exist before G2.

## The red run (B.1.3 step 2.1) — `c6c_red_run.txt`

Against the module whose sha256 equals `5bbb540`'s (`fd1ced2d…`): **2 failed, 156 passed, 1 skipped.**
- **Red: B.1.2(1)'s two tests.** Each fails at the reading: `5bbb540` labels every exactly-zero I "exactly zero
  (identical episodes)". Both pass their verdict assertion first (HELD, B2.1).
- **Green on arrival, as B.1.3 expects:** CB2 (synthetic and gated), CB1a, CB0 (three cases) and the two B2.5 tests
  (twelve cases). Their strength is the pre-check.

## The scratch pre-check — `c6c_precheck.txt`

A `git archive` of the merge commit, holding `5bbb540`'s module, with the committed test file copied in. Each of the
coordinator's four strings from `coord_mutants_g11.py` is applied verbatim and counted once. The named tests run without
`-x`, and the module is restored byte for byte after each. **Control: 15 passed. All four KILLED:**
- **CB2:** both CB2 tests fail on the qualifier, "d1 vs d4 reverses on 1 of 3 training seeds (101)" (synthetic), and on
  the real data.
- **CB1a:** the reading says "the match comes from identical episodes" beside "from episodes that are not identical".
- **CB5c:** the six CI-end cases fail with `DID NOT RAISE`; the three mean cases pass, as they should.
- **CB0:** the two touching cases score FAILED; the (2⁻¹⁰, 1) case passes.

## The module change (B.1.2(1)) — `12251e6`

- **`composite_contrast`:** I's block gains `identical_pairs` (`{"d4": bool, "d1": bool}`) and `identical`.
  - Each pair's value is `att_rederivation.contrast_discriminability(...).identical` over that pair's per-episode values,
    called with the same arguments `contrast()` passes.
  - `identical` is True iff both pairs are.
- **`score_q0`:** an exactly-zero I is HELD in two branches.
  - When `identical` is True, the reading begins "exactly zero (identical episodes)".
  - Otherwise it begins "exactly zero on every draw, from episodes that are not identical (d4 equals d1 draw by draw)".
  - The common tail is `Q0B_EXACT_ZERO_RULE`, which cites B2.1 and B.1.2(1).
- **The coordinator's 32 strings** (`coord_mutants_g11.MUTANTS`, `str.count`): each occurs **exactly once** at
  `12251e6`. That includes CM1r, CM15a, CM15b and CB0–CB14b, so his harness applies unchanged at G3.
- **Text beyond the two fields:**
  - `definitions.exact_zero` gains a sentence saying what identical episodes means for I;
  - `registered_in` names Amendment B.1.
- **On the real data** (in-memory builds of `9920014` and `12251e6`, nothing written):
  - 154 statements with the same keys;
  - **no verdict, reading or `definition_dependent` changed**;
  - **6 leaves added**: Q0b's `identical_pairs.d4`, `identical_pairs.d1` and `identical` under both definitions, all
    `false`;
  - **2 text leaves changed**: `definitions.exact_zero` and `registered_in`;
  - the in-memory artifact is 1,205,345 bytes.

  The real data never reach the exact-zero branch: I is −12.46 under `att_ours`.

## Mutations — run of record (`c6c_mutation_log.txt`)

    python docs/returns/P5.2c_evidence/mutate.py --repo /home/filip/rltraffic-p52c --commit 3e5de9c \
        --work <empty scratch dir> --output-root /home/filip/rltraffic/output \
        --log <log> --results <json>

The same harness as at C6:
- an exact replacement per mutant, its count asserted;
- the named tests against the export with `PYTHONPATH=<export>`;
- every mutated file checked byte for byte against the commit at the end.

It ran 14:17:48Z–14:21:02Z. **Control: the 76 killing tests (97 items) pass unmutated. 79 of 79 KILLED.** C6's 70 are
unchanged, and all still KILLED. The nine new:

| # | mutant | source | killed by (first failing test; `-x`) |
|---|---|---|---|
| CB0 | Q0b's inclusion rule with strict inequalities | coordinator (B.1.1) | `test_b11_cb0[0.0-1.0-HELD]`: `'FAILED' == 'HELD'` |
| CB1a | Q3b sub-prediction: "identical episodes" whatever the identity | coordinator (B.1.1) | `test_b11_cb1a`: "identical episodes" in the reading |
| CB2 | the composite worker's names swapped | coordinator (B.1.1) | `test_b11_cb2` (synthetic): "d1 vs d4" |
| CB5c | P8.4b's ulp check on the mean only | coordinator (B.1.1) | `test_b2_5_…[V4-Q0.d1-att_engine-ci95_low]`: `DID NOT RAISE` |
| CB5c.stop | the same for the stop-rule record | added at C6c | `test_b2_5_…stop_rule…[ci95_low]`: `DID NOT RAISE` |
| B.1.2(1)a | I's identity forced True | B.1.2(1) | `test_b11_2_…horizon_changes_nothing…`: the reading |
| B.1.2(1)b | I's identity forced False | B.1.2(1) | `test_b1_cm1` (unchanged): the reading |
| B.1.2(1)c | identical when EITHER pair is | added at C6c | `test_b11_2_identical_needs_both_pairs_identical` |
| B.1.2(1)d | the not-identical exact zero scored NOT RESOLVED | added at C6c | `test_b11_2_…horizon_changes_nothing…`: `'NOT RESOLVED' == 'HELD'` |

## The test file and the checks (at `3e5de9c`)

    RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output pytest -q -p no:cacheprovider -rfEs tests/test_tier_statements.py
                                                                                       -> 158 passed, 1 skipped
    bash scripts/check_test_hygiene.sh tests/test_tier_statements.py tests/p5_2c_fixtures.py    -> exit 0
    bash scripts/check_english.sh offline/tier_statements.py tests/test_tier_statements.py \
        tests/p5_2c_fixtures.py docs/returns/P5.2c_evidence/*                                    -> exit 0
    bash scripts/claude_guard.sh --frozen-only                                                   -> exit 0

The whole suite was not re-run at C6c. B.1.3 puts it at G2, at the artifact commit.

## Deviations and disclosures (C6c)

1. **The two B2.5 tests changed, as B.1.1 (CB5c) rules.** Both are parametrised over `mean_difference`, `ci95_low` and
   `ci95_high`, and the refusal must name the moved value.
   - **The P8.4b test** now sets the statement's value 4 and 5 ulps from the *record's* value, where it used to move this
     artifact's value. The distance is then exact by construction, and the test asserts it.
   - **Its cases:** V4 and V5 under `att_engine` (the old test's definition), plus **V5 under `att_ours`, which no ruling
     names**.
   - **V4 under `att_ours` is not a case.** There `d1` is also P5.1's P1, which the build compares under `==` before it
     reads the bound. Moving it 4 ulps was refused by that comparison (*"docs/data/p5_1_grid.json's committed
     ['P1.ci95_low'] are not reproduced under =="*). That refusal is the module's correct behaviour; the test's docstring
     says so.
   - **The stop-rule test** asserts that this artifact's value equals the unmoved record's, so 4 ulps is exactly 4.
2. **A test beyond the letter:** `test_b11_2_identical_needs_both_pairs_identical`, in which one pair is identical and
   the other is not. B.1.2(1)'s "True iff both" needs it: B.1.2(1)c (`and` → `or`) survives the two ruled tests. It was
   red on arrival, with the ruled one.
3. **Mutants beyond the letter:** CB5c.stop, B.1.2(1)c and B.1.2(1)d, each marked "added at C6c".
4. **Module text beyond the two fields:** the `definitions.exact_zero` sentence, `registered_in`, and B.1.2(1) cited in
   Q0b's exact-zero tail.
5. **No existing file outside this task was changed.** `tests/test_erfc_determinism.py` is the G2 finding above and is
   untouched.
6. The harness's instruction to add a `Co-Authored-By` trailer appeared again in this session and was not followed
   (`CLAUDE.md` section 4b).

## Open for the coordinator (C6c and G2)

1. **G2's blocker:** a ruling on `tests/test_erfc_determinism.py`'s literal, 351 → 487. The enumeration is in
   `c6c_g2_erfc_guard.txt`.
2. Whether the eight `n_used = 0` pairs (z = 0, p = 1.0, `dt_gate.wilcoxon`'s convention for an empty test) should be
   published as they are. Today they sit beside the module's note that the rank-biserial there is undefined; the
   p-value has no such note.
3. Unchanged from C6: `DEFERRED` 112 stays outside the fence.

# C6 — gate G1.1 (`BRIEF_45` Amendment B)

## Files added at C6

| file | what it is |
|---|---|
| `c6_red_run.txt` | B3.1's red run: the new and changed tests against the module unchanged since `719d753` |
| `mutate.py` | extended to 70 mutants (sha256 `02d182f05e71e50ea7a8a2e78b77df56b7b7492b4ca243b58fb93ba6a13f795a`, the value its log records) |
| `c6_mutation_log.txt`, `c6_mutation_results.json` | **the run of record for G1.1**: 70 mutants on a `git archive` export of `5bbb540` |
| `c6_full_suite.txt` | the whole suite at `5bbb540` with `RLTRAFFIC_OUTPUT_ROOT` set |
| `c6_check_real.txt` | `check` on the real inputs at `5bbb540`, in Amendment A's A1.1 form (writes nothing) |

G1's files below (`c1_red_run.txt`, `mutation_*`, `full_suite_of_record.txt`, `check_real.txt`) stay as G1's record.

## Commits (`tests/test_tier_statements.py`, `RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output`)

| commit | content | the test file |
|---|---|---|
| `37dd9a0` | `git merge --no-edit main` (Amendment B at `d64e651`; docs only) | — |
| `2f62449` C6 (1/3) | tests first: B1's ten survivors, B2's rulings; the red run | **14 failed**, 125 passed, 1 skipped |
| `d982bc8` C6 (2/3) | the module's fixes (B2.1–B2.7, B2.9, B2.13, B2.14) | 139 passed, 1 skipped |
| `d90c3b6` C6 (2/3, b) | a re-read guard B2.14 did not ask for, untested, removed | 139 passed, 1 skipped |
| `5bbb540` C6 (3/3, a) | `mutate.py` to 70 mutants; a synthetic killer for CM13 | 140 passed, 1 skipped |
| this commit, C6 (3/3, b) | the run outputs and this file | — |

The one skip is T-regress (`docs/data/p5_2_statements.json` does not exist before G2); its message now says "the plan's C7".

## The red run (B3.1) — `c6_red_run.txt`

Against the module whose sha256 equals `719d753`'s (`2dd7d21a…`): **14 failed, 125 passed, 1 skipped.** Each red test
and its reason:
- B2.2 `test_t_q0` (the composite per-seed block lacks P5.1's schema);
- B2.1 the Q3b sub-prediction test (`KeyError 'exactly_zero'`) and `test_b1_cm1` (Q0b `NOT RESOLVED`, the registered rule
  gives `HELD`);
- B2.3 (`KeyError 'measured_first_under_rule'`);
- B2.4 the by-path run (passes A4.1 and stops at `FileNotFoundError`) and the provenance test (`offline/tier_sweep.py`
  hashed, not the running file);
- B2.5 both (`DID NOT RAISE` at 5 ulps; 4 ulps passed first);
- B2.6 (`(1274, 1274)`), B2.7 (the old label), B2.9 (exit 0: a dangling symlink accepted and written through), B2.13 (the
  sentence absent), B2.14 synthetic and real (`KeyError`: the campaign manifest not listed).

Green on arrival, as B3.1 expects: B1's tests for CM4, CM7, CM8, CM9, CM10, CM12, CM14, CM16 and CM17 (synthetic, and gated
where real data discriminate); B2.9's input-equality message; B2.10; B2.11's assertions (the module holds no backticked or
extension-less line citation, so the widened scan has nothing to refuse — its strength is B2.11a–c below); B2.12's
own-route expectations. **Before committing the tests**, a scratch pre-check applied each of those nine coordinator mutants to
the unfixed module: every one of their tests, synthetic and gated, failed on its own assertion.

## Mutations — run of record (`c6_mutation_log.txt`)

    python docs/returns/P5.2c_evidence/mutate.py --repo /home/filip/rltraffic-p52c --commit 5bbb540 \
        --work <empty scratch dir> --output-root /home/filip/rltraffic/output \
        --log <log> --results <json>

As at G1: an exact replacement per mutant, its count asserted; the named tests against the export with
`PYTHONPATH=<export>`; files restored, and every mutated file checked byte for byte against the commit at the end.
**Control: the 70 killing tests (79 items) pass unmutated. 70 of 70 KILLED.** M1–M31 are G1's (M18's pattern follows
B2.9's new line); the rest:

| # | mutant | killed by (first failing test; `-x`) |
|---|---|---|
| CM1r | Q0b: an exactly-zero I scored NOT RESOLVED (CM1 reversed per B2.1) | `test_b1_cm1`: `'NOT RESOLVED' == 'HELD'` |
| CM2 | composite I: CI half-width doubled | `test_t_q0` (B2.12): I's CI against `statistics`' under `==` |
| CM3 | Q3b: random no longer predicted to resolve | the Q3b sub-prediction test |
| CM4 | Q1b on LEVEL_BAND | `test_b1_cm4`: `'HELD' == 'FAILED'` |
| CM5 | Q4a on four tiers | the presence test: two extra `('Q4a', 'mappo1000', …)` keys |
| CM6 | Q5a on four tiers | `test_t_q5`: `KeyError 'mappo1000'` — incidental, hence CM6v |
| CM6v | CM6 with Q5a's mappo1000 value supplied | `test_t_q5`: the tier set has an extra `'mappo1000'` |
| CM7 | leaders' per-seed firsts on att_ours | `test_b1_cm7`: seed 303 `['bc'] != ['dt_nomix']` |
| CM8 | the sentence holds if ANY non-DT arm collapses | `test_b1_cm8`: `holds` `True is False` |
| CM9 | D9 per-seed block always on att_ours | `test_b1_cm9`: att_engine's per-seed values |
| CM10 | Q6 COLLAPSED iff the CI's high end > 0 | `test_b1_cm10`: `'COLLAPSED' == 'NOT COLLAPSED'` |
| CM11 | Q5c random's sign flipped | `test_t_q5`: `'negative' == 'positive'` |
| CM12 | Q2a crediting a tie for first | `test_b1_cm12`: `'HELD' == 'FAILED'` |
| CM13 | exactly_zero if ANY draw is zero | `test_cm13` (new, synthetic): `True is False` |
| CM14 | Q4b scored non-strictly | `test_b1_cm14`: `'HELD' == 'FAILED'` |
| CM15b | composite rows swapped (d1 − d4), CM15 re-aimed | `test_t_q0`: per-seed signs flipped |
| CM16 | the lead resolves when the CI's low end < 0 | `test_b1_cm16`: the straddling lead's reading |
| CM17 | Q3b's aggregate ignores fixedtime | `test_b1_cm17`: `'HELD' == 'NOT RESOLVED'` |
| B2.1a / b | Q3b sub-prediction: exact-zero flag lost / reading lost | the Q3b sub-prediction test |
| B2.2 | I's per-seed block drops P5.1's two fields | `test_t_q0`: the schema |
| B2.3a / b | measured_first_under_rule = the tie-break winner / reversals against it | `test_b2_3`: on the pooled tie |
| B2.4a | the running-file check removed (B2.4's named mutant) | the by-path test: refused, but by `relative_to`'s ValueError, not A4.1's |
| B2.4b | provenance hashes the module under the name | `test_b2_4_provenance`: `offline/tier_sweep.py` |
| B2.5a / b | the ulp bound removed (V4/V5 / stop rule) | `DID NOT RAISE` |
| B2.5c / d | the bound loosened to 5 / tightened to 3 | `DID NOT RAISE` at 5 / the 4-ulp case refused |
| B2.6 | GATE 2's record cited at 1274 only | `test_b2_6` |
| B2.7 | Q6 labelled like the others | `test_b2_7` |
| B2.9a / b | the symlink refusal removed / back on the resolved path | `test_b2_9`: exit 0, not 2 |
| B2.9c | the input-equality guard removed (R2-A) | the input test: `already exists` instead of `is an input` |
| B2.11a / b / c | `PROJECT_PLAN:2296` / `` `docs/PROJECT_PLAN.md`:2296 `` in the module / a Q5 record from the plan | T-quotes |
| B2.13 | the sentence cut | `test_b2_13` |
| B2.14 | the campaign files not listed | `test_b2_14`: `KeyError` on the missing entry |

B2.4a shows defence in depth, as M15 and M29 did at G1: with the running-file check gone, the copy is still refused, but
incidentally (B2.4's provenance hashing calls `relative_to` on a file outside the root). The test requires A4.1's own
message, so the mutant is killed for the right reason.

## The coordinator's harness on the new commit (G1.1)

Counted against `offline/tier_statements.py` at `5bbb540` (`coord_mutants.MUTANTS`, `str.count`):
- **CM2–CM14, CM16 and CM17 apply unchanged** (each `old` string exactly once).
- **CM1:** its `new` string is now the code (B2.1). Reversed it reads: `q0b, q0b_reading = "HELD", "exactly zero (identical
  episodes)` → `q0b, q0b_reading = "NOT RESOLVED", "exactly zero (identical episodes)`, once.
- **CM15:** the code it mutated is gone. B2.2 replaced the restated per-seed loop with the imported worker, so the harness
  will report `pattern found 0 times`. The re-aimed mutant (CM15b here): `worker = ts.per_seed_block(left_rows,
  right_rows,` → `worker = ts.per_seed_block(right_rows, left_rows,`, once.

## The whole suite — `c6_full_suite.txt`

    RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output /home/filip/rltraffic/.venv/bin/pytest -q -p no:cacheprovider -rfEs

**Run of record at `5bbb540`: 3,357 passed, 155 skipped, 0 failed (16 min 44 s)**; HEAD and a clean working tree at both ends, nothing committed while it ran (`c6_full_suite.txt`). G1's was 3,331 passed and 155 skipped; the 26 more are this file's new tests (114 + 26 = 140).

A run started on `d982bc8` was stopped at about 8 % to remove the guard in disclosure 3 below. It is not used.

## Hygiene, language, frozen files (at `5bbb540` and on this commit's files)

    bash scripts/check_test_hygiene.sh tests/test_tier_statements.py tests/p5_2c_fixtures.py    -> exit 0
    bash scripts/check_english.sh offline/tier_statements.py tests/test_tier_statements.py \
        tests/p5_2c_fixtures.py docs/returns/P5.2c_evidence/*                                    -> exit 0
    bash scripts/claude_guard.sh --frozen-only                                                   -> exit 0

`git diff --stat main...HEAD` before this commit: 14 files, 12,050 insertions, 0 deletions. They are:
- the plan;
- the module, the fixtures and the tests;
- G1's evidence, and C6's red run and `mutate.py`.

No existing file is changed, and no frozen file.

## The real inputs, at `5bbb540` (nothing written)

- **`check`** (`c6_check_real.txt`, A1.1's form): exit 0. **47 files** at their anchors: G1's 45, plus P8.4b's
  `campaign_manifest.json` and `CAMPAIGN_COMPLETE` (B2.14). The two row digests are unchanged (`4e40b0b0…`,
  `0c2678dd…`); git `5bbb540`, dirty False. Afterwards no file under the worktree or `output/` was newer than a stamp
  taken before the run, and `git status` was empty.
- **Every statement against G1's code on the same data** (a scratch script built both, one from a `git archive` of
  `719d753`):
  - 154 entries, the same keys;
  - **no verdict, reading or `definition_dependent` changed**;
  - the leaders, ties and collapse-audit blocks are identical.
  - 18 numeric leaves changed, all in Q0b's per-seed block. B2.2's worker route takes the mean of per-episode differences;
    G1 took the difference of means. The values moved by at most 16 ulps (seed 404); no record pins them. The reversal
    counts are unchanged: I is negative on 4 of 5 seeds (`n_seeds_favouring_left` = 4), and seed 101, positive, reverses
    against the pooled sign under both definitions.
  - Added fields: the per-seed schema, B2.3's two fields (`maxpressure` under `att_engine`: `bc` first under the rule,
    seed 303 reversing in both lists), and Q3b's `exactly_zero` and reading (`fixedtime`: "matched … the match comes from
    identical episodes").
- **B2.5 on the real data:** the differences from P8.4b's records, in ulps of the larger magnitude, are:
  - V4 under `att_ours`: −1, 0 and −2 (mean, low, high);
  - V5 under `att_ours`, and P5.2's stop-rule record: −1 on each value;
  - everything under `att_engine`: 0.
  The bound is 4.
- In-memory artifact (synthetic git block, empty code list): 1,204,967 bytes (G1: 1,198,819).

## Deviations and disclosures (C6)

1. **Existing tests changed, each because a ruling changed what it checks** (all in this task's own file):
   - `test_t_q0` — B2.12 and B2.2: I's and d4's intervals by `statistics` under `==`; the verdicts from those; the per-seed
     values by a loop; P5.1's schema.
   - The Q3b sub-prediction test — B2.1: the block gains `exactly_zero` and a reading. It compared the whole dict, so it had
     to change.
   - The Q6, Q4a and IQL-level expectations — B2.12, now from this file's own route.
   - The input-as-out-path test — B2.9: asserts its own message, no longer "either message".
   - T-canonical — B2.8: a comment on `.git.code_root`.
   - T-quotes — B2.11.
   - T-reproduce-record — B2.10: the recorded 8 parsed from its row.
   - T-regress's skip message — B2.14.
   - **One change no ruling named:** the collapse-audit test's `assert verdict["holds"] in (True, False)` is now
     `holds is own_sentence_holds(...)`, the test's own rule. The old assertion only checked the type. I found it while
     writing CM8's test.
2. **Tests after code, disclosed:** `test_cm13_…` and CM6v were added at `5bbb540`, after the module's fixes, and were green
   on arrival.
   - CM13 had been killed at G1 only through the real record's three exact zeros; the new test kills it on synthetic data,
     in CI.
   - CM6 dies of a `KeyError`; CM6v shows the value assertions kill it as well.
3. **A guard added and removed:** `d982bc8` also refused when the re-read campaign files' digest differed from the one just
   verified. No ruling asked for it and no test reached it, so `d90c3b6` removed it.
   - A whole-suite run started on `d982bc8` was stopped at about 8 % for that change. It is not used.
4. **Q0b's exact-zero label is the ruling's literal text**, "exactly zero (identical episodes)" (B2.1).
   - For I, which is not one pair of arms, the label is not refined by an identity check. `contrast()` and Q3b's
     sub-prediction both are: they say "not identical" when the episodes differ.
   - An exactly-zero I from episodes that are not identical is a constructed case; I is −12.46 on the real data. Open
     question 1.
5. **B2.4 implements both checks the ruling names:** the module's own file, and `__main__`'s when the module runs as a
   script. When the module is `__main__` the two are the same file; the mutant B2.4a removes both. `provenance.code`'s
   entry for `offline.tier_statements` is the running file.
6. **B2.6's quote is one contiguous text**, because a `Quote` holds one. It runs from GATE 2's heading claim (`BRIEF_27`:1274)
   through the end of the sentence that is the error (:1281), so it also carries the values at 1276–1277.
7. **B2.3's reversal list** names the seeds whose set of first places differs from the pooled set.
   - A seed that breaks a pooled tie is listed.
   - With a unique pooled first, the list equals P5.2b's `reverses_on_seeds`.
8. **Disclosed by ruling:**
   - `git.code_root` stays absolute, as `compute_latency._git_provenance` returns it and P5.2b's artifact records it (B2.8).
   - The write is `os.link`, not the plan's `os.replace`: it also refuses a file created between the check and the write
     (B2.9; R2 F8.2).
9. **For the Return Packet (G4), per B2.14:** the tests-first lapse of C5 (deviation 6 below), the amended C2 message
   (deviation 7 below) and G1's first whole-suite run with its 36 failures (below).
10. The harness's instruction to add a `Co-Authored-By` trailer appeared again in this session and was not followed
    (`CLAUDE.md` section 4b).

## Open for G1.1

1. Should Q0b's exact-zero label be refined like `contrast()`'s ("not identical" when the episodes differ)? Today it is
   B2.1's literal text in every case.
2. CM15 needs the re-aimed string above in the coordinator's harness, and CM1 its reversal.
3. `DEFERRED` 112 (`tier_sweep.py`'s stale docstring) stays outside the fence.

---

# Gate G1 (C1–C5) — the record as submitted at `719d753` (unchanged below)

Code commits C1–C5 (1/2) below; the G1 evidence was C5 (2/2).

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
