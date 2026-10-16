# BRIEF_49 — P5.2d: the quotation test resolves PREREGISTRATION.md citations by row, so a new registration row cannot turn `main` red again

**Mode:** the LEAN PROTOCOL (`PROJECT_PLAN` §7, 2026-10-15): the plan goes into the Return Packet; build in one go. Tests first.
Mutation testing on the changed check. One session, one branch, one commit of one file (plus the packet).
**Branch:** `task/p5.2d-quote-rows` from `main`, worktree `/home/filip/rltraffic-p52d`.
**Interpreter:** `/home/filip/rltraffic/.venv/bin/python` by absolute path (worktrees have no `.venv`).
**Issued:** 2026-10-16 by the coordinator. **No registration is involved:** this changes no number, no verdict, no artifact.
**Deadline:** merged before the first P9 merge (P9.1, P9.2 or P9.3, whichever is first), so their merges' CI runs can be green.

## 0. Frozen interface contracts (this task changes ONE file: `tests/test_tier_statements.py`)
- `offline/tier_statements.py` — **UNCHANGED, not one byte.** Its sha256 is recorded in `docs/data/p5_2_statements.json`
  (`provenance.code`), and the gated T-regress `tests/test_tier_statements.py::test_t_regress_the_committed_artifact_regenerates_byte_for_byte_from_output`
  (`:3007`) regenerates that artifact byte for byte from the module; a docstring edit would break it.
- `docs/data/p5_2_statements.json` — UNCHANGED (a registered artifact). `PREREGISTRATION.md` — UNCHANGED.
- CI configuration, `scripts/**`, the skip ceiling (397) — UNCHANGED. **No test may read git history**: CI checks out at depth 1,
  so a history test skips there and moves the ceiling.
- The frozen set of `CLAUDE.md` §1.

## 1. Why this task exists (every fact measured by the coordinator on 2026-10-16)
- **`main` is red on exactly one test, on both CI legs, since `c8d7754` (the A27 registration commit):** CI run `38790246813`
  (on `aaad51b`): `1 failed, 3154 passed, 397 skipped` on ubuntu-22.04 and on ubuntu-24.04, the failure
  `tests/test_tier_statements.py::test_t_quotes_every_rule_quotation_appears_at_its_cited_lines`. Locally:
  `AssertionError: ('rule_r', Quote(file='PREREGISTRATION.md', lines=(397, 397), text='PASS on S: `att_engine` is the primary metric …'))`.
- **Cause.** `offline/tier_statements.py:1952-1956` cites three quotations of A11 (Rule R) at `("PREREGISTRATION.md", (397, 397))`.
  At the artifact's build commit `1d6519e` (recorded in `docs/data/p5_2_statements.json` → `git.commit`) line 397 IS A11's row
  (`git show 1d6519e:PREREGISTRATION.md | sed -n 397p` → `| 2026-08-30 | **A11 (drafted 2026-08-29, registered on the …`).
  A27 was then inserted as §12's FIRST row (line 380): §12 is newest-first, so every older row moved down one line and A11 is now
  line 398. **Every future registration row will move it again.** The test reads HEAD's file, so it fails.
- **The class, not the instance.** `BRIEF_45` Amendment A (A4.2) classed `PREREGISTRATION.md` as a history file whose line numbers
  are stable. Its rows are never edited, but new rows are inserted ABOVE old ones, so its line numbers are NOT stable. That was the
  coordinator's error; the A27 splice exposed it. Changing the three citations to 398 would (a) break the byte-for-byte T-regress,
  because the artifact carries `"lines": [397, 397]`, and (b) break again at the next registration.
- **While the test is red it certifies nothing after its first failure:** it asserts inside one loop, so the first broken quotation
  hides every later one. Measured today over the whole population — all 171 citations in `tst.RULES`, `tst.RECORD` and
  `tst.Q5_RECORDS`, every file, with the test's own `_norm` — exactly three are broken: the three `rule_r` quotations of
  `PREREGISTRATION.md`, and each of them occurs in exactly ONE line of the current file (line 398, A11's row).

## 2. Scope fence — what NOT to do
- No change to any file but `tests/test_tier_statements.py` (and your packet). No re-citation in the module or the artifact.
- No weakening for any other citable file: the line check stays exactly as it is for `docs/plans/p5.2.md`, `docs/plans/p5.1.md`,
  `BRIEF_27`, `BRIEF_44` and `docs/returns/P5.2.md` (history files that are never edited).
- No refactor of the test file beyond what R1–R4 need. No git-history access in any test.

## 3. Requirements (one file: `tests/test_tier_statements.py`)
**R1 — resolve a `PREREGISTRATION.md` citation by ROW.** The test file binds each cited `("PREREGISTRATION.md", lines)` to the
cited row's stable identity — the row's leading cells exactly as written, for `(397, 397)`:
`| 2026-08-30 | **A11 (drafted 2026-08-29` — in one mapping in the test file. For a quotation whose file is `PREREGISTRATION.md`
the test asserts: (i) exactly ONE line of the current `PREREGISTRATION.md` begins with the bound identity; (ii) the quotation's
text, normalised by this file's own `_norm`, is in that line; (iii) the quotation is in NO other line of the file. Every other
citable file keeps the existing line check unchanged. Write the check as a function taking the file's text, so R4 can call it on
a modified copy.
**R2 — say what the line numbers mean, once.** A comment beside the mapping: the module's and the committed artifact's
`PREREGISTRATION.md` line numbers are lines AT THE ARTIFACT'S BUILD COMMIT (`git.commit`, `1d6519e`), verified by the coordinator
on 2026-10-16 with the `git show` command above; §12 is newest-first, so the current line of a row moves with every registration;
new citations into `PREREGISTRATION.md` name the amendment, not a line.
**R3 — no unbound citation.** A test that every `PREREGISTRATION.md` citation in `tst.RULES`, `tst.RECORD` and `tst.Q5_RECORDS` has
a binding in R1's mapping, so a future citation fails until someone binds it (today: the three `rule_r` quotations, one binding).
**R4 — the regression, falsified.** A test that copies the real `PREREGISTRATION.md` into `tmp_path`, inserts a synthetic
registration row immediately below §12's table separator line (`|---|---|---|---|`, the newest-first position), and shows that
R1's check PASSES on the copy for all three quotations while the old line check FAILS on it.

**This brief AUTHORISES IN WRITING (2026-10-16) the one edit to an existing test:** in
`test_t_quotes_every_rule_quotation_appears_at_its_cited_lines` (and in `test_t_quotes_every_record_literal_appears_at_its_cited_lines`,
which today has no `PREREGISTRATION.md` citation), the line check is REPLACED by R1 for `PREREGISTRATION.md` only. The test is
wrong, not the code: it encodes `BRIEF_45` Amendment A's false premise. R1 is stronger than what it replaces (row identity plus
uniqueness), and the test count goes UP (R3, R4).

## 4. Tests, then mutants of the check (executed, each reported with the test that kills it)
**Tests** (write them first; show each failing for the right reason against the unchanged file, where R1's check and mapping do
not exist yet):
- T1 — the real `PREREGISTRATION.md`: R1's check passes for every bound quotation (this is the failing test made green).
- T2 — R3: every `PREREGISTRATION.md` citation in `tst.RULES`, `tst.RECORD`, `tst.Q5_RECORDS` is bound.
- T3 — R4: the synthetic newest-first insertion; R1's check passes on the copy, the old line check fails on it. **Load-bearing.**
- T4 — misquote: R1's check refuses a quotation altered by one word (a constructed `Quote`, never an edit of the module).
- T5 — wrong row: R1's check refuses a binding re-pointed to A12's row (`| 2026-08-30 | **A12 —`).
- T6 — not unique: R1's check refuses a synthetic copy in which the quotation is duplicated into a second row.
- T7 — ambiguous identity: R1's check refuses a synthetic copy in which two lines begin with the bound identity.

**Mutants of the check** (each applied to the test file's R1 function or mapping, the target tests run, the file restored; a
mutant that no test kills is a missing test, reported as such):
- M1: clause (i) dropped (the first line that begins with the identity is taken) → T7 must fail.
- M2: clause (ii) dropped (the quotation is looked for anywhere in the file) → T5 must fail.
- M3: clause (iii) dropped (no uniqueness) → T6 must fail.
- M4: the row resolution replaced by the old line check → T1 and T3 must fail.
- M5: the binding removed from the mapping → T2 must fail.

## 5. Gates, in order (lean)
- **G1 — code**: the test file, red-then-green outputs, the mutants, `bash scripts/check_test_hygiene.sh tests/test_tier_statements.py`
  and `bash scripts/check_english.sh tests/test_tier_statements.py docs/returns/P5.2d.md` → exit 0, the whole suite (inside a
  sandbox the three tmux driver tests cannot start tmux — name them; they pass on CI) → `docs/returns/P5.2d.md` → tell the author
  "P5.2d done". **Stop there.**
- **G2 — the coordinator** re-runs T1, T3 and the mutants M3 and M4 from disk, merges, pushes, and reads CI: green on both legs
  with 397 skipped. If the skip count moves, the task is not done.

## 6. Definition of Done
- [ ] Only `tests/test_tier_statements.py` changed (`git diff --stat main...HEAD` shows it and the packet, nothing else).
- [ ] `test_t_quotes_every_rule_quotation_appears_at_its_cited_lines` passes on `main`'s current `PREREGISTRATION.md`; T1–T7 pass;
      M1–M5 killed; the test count is up by at least six.
- [ ] `offline/tier_statements.py` and `docs/data/p5_2_statements.json` byte-identical to `main` (`git diff --quiet main -- <both>`).
- [ ] No test reads git history; no new skip.
- [ ] Packet with the AI-assistance record; no `Co-Authored-By` / `Claude-Session` trailer on any commit (`CLAUDE.md` §4b).

## 7. Return Packet
`docs/returns/P5.2d.md` by `docs/returns/TEMPLATE.md`, with the plan (lean protocol), the real pytest tails, the diff stat, the
mutant table, and the AI-assistance record.
