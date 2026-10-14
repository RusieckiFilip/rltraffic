# G5 / M1a findings -- P5.2c (task/p5.2c-statements @ 1f17d35)
Reviewer: read-only merge reviewer, gate G5, mandate M1a (fence, inputs, barrier, determinism)
Started: 2026-10-14
Worktree: /home/filip/rltraffic-p52c (read-only). Scratch: .../scratchpad/g5/m1a/

## Checklist
- [x] 1. FENCE: diff --stat main...task/p5.2c-statements; every path allowed; erfc test diff only what B.2.2 allows
- [x] 2. DEPENDENCIES: every import in offline/tier_statements.py is stdlib / numpy / offline/
- [x] 3. INPUT PINS: each input asserted at its digest BEFORE use; mismatch refuses the build
- [x] 4. P8.4b ROW CHECKS: reproduces_committed true AND att_ours == eval att_horizon for (seed, draw); any row read without it?
- [x] 5. THE BARRIER: order of validation vs writes/deletes in build; out under output/ refused; existing out (incl. dangling symlink) refused; failing input leaves no file and existing artifact unchanged; one refusal run in export
- [x] 6. MODULE-ROOT GUARD (A4.1, B2.4): module + every imported offline module under --repo-root; running file == hashed file
- [x] 7. DETERMINISM: sorted keys, allow_nan=False, no timestamp/env field; git block commit+dirty; fields that could differ between builds (git.code_root abs path; T-regress)
- [x] 8. LIVING-DOCUMENT CITATIONS (A4.2): file:line citations into PROJECT_PLAN.md, README.md, DEFERRED.md, CLAUDE.md, BRIEF_45

## Findings (appended per item)

### [x] 1. FENCE -- PASS (no finding)
- Command: `git -C /home/filip/rltraffic diff --stat main...task/p5.2c-statements` (main=7a4a7aa, branch=1f17d35, merge-base=7a4a7aa, worktree HEAD=1f17d35).
  48 files: 47 A, 1 M. Paths: offline/tier_statements.py (A), tests/test_tier_statements.py (A), tests/p5_2c_fixtures.py (A),
  tests/test_erfc_determinism.py (M, +25/-4), docs/data/p5_2_statements.json (A), docs/plans/p5.2c.md (A), docs/returns/P5.2c.md (A),
  docs/returns/P5.2c_evidence/** (41 files A, incl. .py helper scripts dump.py, falsify.py, gated_probe.py, mutate.py, g4_tables.py -- under docs/returns/**, allowed).
- Filter of `--name-only` against the allowed set: zero paths outside it. Zero frozen paths. No deletions/renames.
- erfc diff (2 hunks only, at test_erfc_determinism.py:274 and :343-369): literal 351->487; docstring first line with history
  "322 -> 330 -> 339 -> 351 -> 487"; assertion message gains "plus the 136 P5.2c Wilcoxon blocks in p5_2_statements.json (BRIEF_45 Amendment B.2, authorised 2026-10-13)";
  a comment. Walk, ONE_SIDED_PAIRS, _expected_p_value, `moved == []` untouched (no hunk there).
- Mechanical check (scratch verbatim.py: brief lines 680-687 vs the test comment, whitespace-normalised): VERBATIM EQUAL: True (872 chars).
- Commit separation: b5b8a98 touches tests/test_erfc_determinism.py alone; 9fa93e9 touches docs/data/p5_2_statements.json alone.
- Merge 5a835b4 (main into branch after the artifact) brought docs only (PROJECT_PLAN.md, BRIEF_45, docs/reviews/P5.2c_G3/**): no offline/ or input file changed after the build commit.

### [x] 2. DEPENDENCIES -- PASS (one NOTE)
- `grep -n -E '^\s*(import |from \S+ import )' offline/tier_statements.py`: direct imports are stdlib only
  (__future__, argparse, hashlib, itertools, json, math, os, statistics, sys, dataclasses, pathlib, typing; function-local `importlib` at :378, `re` at :2285),
  numpy (:48), and offline.{att_rederivation, compute_latency, dt_gate, iql_correction, spatial_mixing, tier_sweep, offline_baselines} (:50-58). No new third-party import.
- Transitive closure measured (scratch deps.py: `PYTHONPATH=<worktree> python -B -P deps.py`): importing the module loads non-stdlib
  agent, utils (repo), numpy, torch, torchgen, gymnasium, farama_notifications, typing_extensions, cython_runtime -- all via PRE-EXISTING committed modules
  (offline/dt_gate.py:80-82 `import torch` / `from agent.DTAgent import ...`; offline/dataset.py:144; offline/trajectory_logger.py:172; offline/roadnet_graph.py:75).
  pyproject.toml:25-28 already declares gymnasium==1.3.0, numpy==2.5.1, torch==2.11.0. No new dependency.
- NOTE (no action needed for merge): 13 offline modules are loaded at import (offline, att_rederivation, compute_latency, dataset, dt_gate, iql_correction,
  joint_windows, offline_baselines, roadnet_graph, spatial_mixing, tier_statements, tier_sweep, trajectory_logger), plus the frozen agent/ and utils/ packages.
  Relevant to item 6: agent/ and utils/ are non-offline repo code in the process; see item 6 for whether the guard covers them.

### [x] 3. INPUT PINS -- PASS (one NOTE)
All in offline/tier_statements.py `load_inputs` (:550-696), every check a `raise ValueError` inside load_inputs, i.e. before Inputs is returned
(whether that precedes every write is item 5). Pins at :149-159 (+ iql_correction.py:241-247). Every pin recomputed from disk with `sha256sum` and equal:
- output/SHA256SUMS_p5_1.txt 023607ff...4c61: `read_manifest` (:272-280) single read_bytes -> sha256 -> compare -> parse the SAME bytes. Called :557. PASS.
- output/SHA256SUMS_p5_2.txt fde8309b...c19b (ic.PINS): `ic.p5_2_sums` (iql_correction.py:421-432) hash compared to pin, then parse. Called :558. PASS
  (NOTE: it hashes via _sha256_file and then re-reads with read_text -- a two-read TOCTOU in the brief-mandated pre-existing helper; this module's own
  read_manifest avoids it. Static files, no concurrent writer: no practical effect).
- output/SHA256SUMS_p5_2b.txt f3e1994c...10ed: `read_manifest` :559; additionally cross-checked against the pinned P5.2b artifact's
  inputs.run_manifest.sha256 at :658-660. PASS.
- P8.4b declared_cells_sha256 1f29b469...8f4e (ic.PINS): `ic.assert_p8_4b_campaign` (iql_correction.py:603-627) recomputes the digest from the manifest's own
  cell list, requires recomputed == recorded == CAMPAIGN_COMPLETE marker == pin, and campaign_status complete. Called :590, BEFORE any P8.4b row is read (:612). PASS.
  Disk: manifest 1f29b469..., marker 1f29b469..., 38,500 cells.
- P5.2b re-derivation campaign 09568e28...8c9e: `ic._campaign_anchor` :635 + pin compare :636-638; manifest and marker also each equal their
  SHA256SUMS_p5_2b line (:639-645). ORDER: the 500 corrected rows are read at :631/:633 BEFORE this anchor check, but each row file is first required to
  equal its line in the PINNED p5_2b manifest (`_read_rows_digested(..., listed=...)` :509-511), and the Cell is only constructed at :646, after the check.
  Nothing derived escapes before the check. PASS. Disk: manifest 09568e28..., marker 09568e28..., 500 cells.
- docs/data/p5_2b_iql_correction.json 216b9f24...a1f8: `read_pinned_json` (:299-306) single read, hash before json.loads; :564-566. Equal to main's blob. PASS.
- docs/notes/readme_2026-10-05/c1_rule_r.json 643b73bc...7e19: `ic._verified_note` (iql_correction.py:675-681) single read, hash before json.loads; :568. PASS.
- Extra pins beyond the brief: docs/data/p8_4b_rederivation.json d92321d4... and docs/data/p5_1_grid.json ccb9a315... (read_pinned_json), both equal disk and main.
- Every evaluation file is read by `read_listed_json`/`ic.read_p5_2_json` (hash == manifest line before json.loads) :575-577; the stop-rule record :582,
  D17 training records :527, the P5.2b training record :625 likewise.
- The artifact's provenance records exactly these digests (scratch prov.py on HEAD:docs/data/p5_2_statements.json): p8_4b_campaign 1f29b469.../38500,
  p5_2b_rederivation 09568e28.../500, p8_4b_rows n=15500 sha256 4e40b0b0..., p5_2b_corrected_rows n=500 sha256 0c2678dd..., 47 inputs listed.

### [x] 4. P8.4b ROW CHECKS -- PASS (no finding)
- The check: offline/iql_correction.py:1076-1090 `_p8_4b_rows` raises unless `row.get("reproduces_committed") is True` AND (arm, seed, draw_id) is the
  file's key AND `float(row["att_ours"]) == episode.att_horizon` (exact `!=` test), where `reference` = the eval file's episodes built by
  `ic.episodes_from_rows(payload["episodes"], value_key="att_horizon")` (tier_statements.py:466) after the eval file's manifest hash check.
- Called for every one of the 31 non-P5.2b CELL_SPECS (tier_statements.py:603-614), after :608 requires P8.4b's declared (seed, draw) set of the arm to
  EQUAL the eval file's episode keys. `_read_rows_digested` (:496-512) then re-reads each file and refuses unless its bytes parse to the checked row, so
  the A4.3 digest covers checked content. `_merged_rows` (:478-493) re-checks alignment per (seed, draw) and takes att_ours from the eval file.
  Artifact: p8_4b_rows n = 15,500 = 31 cells x 500 episodes.
- Rows read WITHOUT the reproduces_committed/att_ours check: `assert_note_means` (tier_statements.py:309-343) and `ic.assert_c1_note_means`
  (iql_correction.py:630-672) read P8.4b cell files with an identity check only. Not a gap: (a) they run at :618-621, AFTER the _p8_4b_rows loop;
  (b) their arms (ts.METHODS = dt_spatial, dt_nomix, bc, bc_top10, bc_top10_perix, iql via `_book_arm`, tier_sweep.py:124-131) are all in CELL_SPECS and
  :608 makes their campaign cells exactly the checked keys; (c) they return no value into the artifact (only arms/route/cells_per_arm).
- Corrected IQL cell: `ic._corrected_rows` (iql_correction.py:1253-1274) requires reproduces_committed, arm, key, att_ours == AND committed_att_ours ==
  p5_2b's eval att_horizon, and the rollout checkpoint == the corrected training record's; then each row file == its SHA256SUMS_p5_2b line (:633-634),
  and the cell's att_ours/att_engine/admission summaries must equal P5.2b's committed cells.corrected AND the original cell's cells.original (:648-657). PASS.

### [x] 5. THE BARRIER -- PASS (one MINOR, two NOTEs)
Build path, offline/tier_statements.py `main` :2750-2792, in order:
 (1) parse args :2752; (2) `assert_code_under_root` :2755 [validation]; (3) `git_block()` :2756 [subprocess git rev-parse / git status];
 (4) build: --out required :2759, dirty refused :2761-2763 [validation]; (5) `load_inputs` :2764 [ALL input validation, read-only];
 (6) `assert_out_path` :2783 [validation: .json suffix, tier_sweep.assert_writable vs resolved --output-root, any ancestor holding SHA256SUMS_p5_2.txt,
 equal to an input, parent exists, UNRESOLVED path is_symlink() or exists() -> refuse :2711]; (7) serialize(build_artifact) :2784 [pure, in memory];
 (8) `assert_code_under_root` re-run and compared :2785-2786 [validation]; (9) `write_artifact` :2787 -- the FIRST and only write (dot-temporary "xb",
 fsync, os.link, finally unlink). Every write and delete is after every validation. Exceptions (ValueError, KeyError, OSError) -> "REFUSED", exit 2;
 any other exception type propagates (exit 1, traceback) but still before the write.
Runs (export of HEAD at .../g5/m1a/export; scratch barrier.py calling assert_out_path directly -- `build` never run, per the hard rule):
 E1 under fake --output-root: REFUSED PermissionError. E1b /home/filip/rltraffic/output/p5_2_statements.json: REFUSED PermissionError.
 E1c relative '..' into real output: REFUSED. E1d real output while --output-root names a FAKE root: REFUSED via the ancestor SHA256SUMS_p5_2.txt check.
 E1e via a symlinked dir into real output: REFUSED. E2 existing file: REFUSED FileExistsError, bytes unchanged (sha256 equal before/after).
 E3 dangling symlink -> missing dir: REFUSED (FileNotFoundError on the resolved parent, before the symlink test). E3b dangling symlink -> absent file
 in an existing dir: REFUSED FileExistsError "is a symlink"; target not created. E4 scratch tree holding SHA256SUMS_p5_2.txt: REFUSED. E5 equal to an
 input: REFUSED ValueError. E6 missing parent: REFUSED. E7 .txt: REFUSED. E9 fresh path: accepted, nothing created.
 CLI: `PYTHONPATH=<export_bad> python -B -P -m offline.tier_statements check --repo-root <export_bad> --output-root /home/filip/rltraffic/output --out <scratch>/would_be.json`
 with ONE byte (a space) inserted in export_bad/docs/data/p5_2b_iql_correction.json -> "REFUSED: ValueError: ... is at sha256 13e5813f..., not the pinned 216b9f24..." exit=2;
 no file at --out. Same CLI on the clean export with --out /home/filip/rltraffic/output/p5_2_statements.json -> all 47 inputs verified, p8_4b_rows 15500 digest
 4e40b0b0... and p5_2b_corrected_rows 500 digest 0c2678dd... (both EQUAL to the committed artifact's provenance), then "REFUSED: PermissionError ... read-only" exit=2; no file created.
- MINOR M5.1 -- write_artifact deletes a file it did not create. offline/tier_statements.py:2722-2730: if `.<name>.<pid>.tmp` already exists,
  `open(temporary, "xb")` raises FileExistsError and the `finally: temporary.unlink(missing_ok=True)` then DELETES that pre-existing file.
  Reproduced (scratch tempdel.py): before "foreign temp exists: True"; write_artifact raised FileExistsError; after "foreign temp exists: False".
  After validation, so not a barrier-order violation; needs a PID-named dotfile collision in the out dir, so practically remote. Fix shape (described, not done):
  only unlink the temporary if this call created it.
- NOTE N5.1 -- `git_block()` (:2756 -> compute_latency.py:768-779) runs `git status --porcelain` BEFORE load_inputs; git status may opportunistically refresh
  the worktree's .git index (a write in git metadata, not output/ or docs/). Pre-existing helper, P5.2b precedent; outside the barrier's data scope.
- NOTE N5.2 -- an out path under the CHECKOUT's own gitignored output/ is accepted when that directory holds no SHA256SUMS_p5_2.txt (E8: export/output/x.json ACCEPTED).
  Consistent with plan §6.5's stated rule (refuse under --output-root or any tree holding SHA256SUMS_p5_2.txt); in the main tree <repo>/output IS the protected root.

### [x] 6. MODULE-ROOT GUARD (A4.1, B2.4) -- PASS on the letter and on the real artifact; one MINOR (a residual path), two NOTEs
- Code: offline/tier_statements.py:373-420 `assert_code_under_root`. Checks (a) this module's own `__file__` and, when `__name__ == "__main__"`, `__main__`'s
  `__file__` resolve under <root>/offline (:382-391, B2.4); (b) every loaded `offline*` module and every CODE_MODULES entry resolves under <root>/offline (:403-410);
  (c) `offline.__path__` is exactly [<root>/offline] (:411-414). provenance.code hashes the RUNNING file for offline.tier_statements (:417). Called at :2755
  and again after serialisation, compared, at :2785-2786.
- Ran: code from export_bad on PYTHONPATH, --repo-root naming export -> "REFUSED: ValueError: the running file (this module's own file) is .../export_bad/offline/tier_statements.py,
  which does not resolve under ..." exit=2.
- Real artifact: all 13 provenance.code entries (scratch codehash.py) equal `git show b5b8a98:<path> | sha256` AND HEAD's: "modules recorded: 13 mismatches vs b5b8a98 and HEAD blobs: 0".
  git block = {commit b5b8a98..., dirty false, code_root /home/filip/rltraffic-p52c}; tier_statements.py is 20c07b17... at b5b8a98, HEAD and in the worktree.
- MINOR M6.1 -- --repo-root is never required to be the git TOPLEVEL whose commit is recorded. `git_block` runs `git -C <offline/..>` (compute_latency.py:198, :768-779);
  inside an untracked copy nested in another clean repo, git reports the ENCLOSING repo's HEAD, and `--untracked-files=no` reports it clean.
  Constructed: scratch repo `outer` (one commit f82d0d73 holding only a README), the HEAD export copied untracked to outer/nested;
  `PYTHONPATH=outer/nested python -B -P -m offline.tier_statements check --repo-root outer/nested --output-root /home/filip/rltraffic/output` -> exit=0,
  "git: commit f82d0d738bf160901300e5aed1b9a3e94ce1eb54, dirty False" with no "build would refuse" note -- so build (which refuses only `dirty is not False`, :2761)
  would record a commit that contains none of the code that ran. Exactly A4.1's stated "Why" (a commit the code did not come from), by a route the guard does not cover.
  No number changes and the committed artifact is unaffected (verified above); detectable afterwards only by comparing provenance.code with the commit's blobs.
  Fix shape (described, not done): refuse unless `git -C <root> rev-parse --show-toplevel` resolves to <root>.
- NOTE N6.1 -- the hash is of the file's bytes read at check time (:419), not of the code object loaded at import; a file edited between import and :2755 (or a stale
  timestamp-valid .pyc) would be recorded with bytes that did not run. The re-check at :2785 narrows but cannot close this (both reads are from disk). Remote.
- NOTE N6.2 -- only `offline*` modules are guarded/hashed; the frozen agent/ and utils/ packages are loaded transitively (dt_gate.py:82 `from agent.DTAgent import ...`,
  roadnet_graph.py:75, trajectory_logger.py:172) and are neither required under --repo-root nor hashed. I did not exhaustively prove that no function this module calls
  executes agent/ or utils/ code; the functions it imports by name are statistics/IO helpers. Also: `git status --untracked-files=no` ignores untracked files, so an
  untracked file that shadows a tracked offline module (an untracked package dir or extension module of the same name) would pass dirty=false; its path in provenance.code
  would differ from the committed one. Contrived; recorded for completeness.

### [x] 7. DETERMINISM -- PASS (two NOTEs)
- serialize (offline/tier_statements.py:2683-2685): `json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"`, utf-8. Committed artifact
  (scratch abs.py on HEAD:docs/data/p5_2_statements.json, sha256 1aa247ed...bb97, 1,259,859 bytes): canonical re-serialisation EQUAL, ASCII only, single trailing
  newline, 0 non-finite floats, 0 ISO timestamps; grep of the module for datetime/time./os.environ/getcwd/platform/cpu_count: no source of a time or environment value
  in the artifact (os.getpid is used only for the temporary's name, :2722).
- git block: compute_latency.py:768-779 -> {commit, dirty (tracked files only), code_root}. Artifact: commit b5b8a98..., dirty false, code_root "/home/filip/rltraffic-p52c".
- Only absolute path anywhere in the artifact: `.git.code_root` (walk of every string). Ruled B2.8 (BRIEF_45:431-432). T-canonical exempts exactly that path with a comment
  (tests/test_tier_statements.py:1404-1407). T-regress (:2609-2618) passes the committed payload's own `git` and `provenance.code` into build_artifact, so code_root is handled
  by construction -- which also means T-regress never checks that the code regenerating it is the recorded code (that is T-code-root's job, :1449-1472).
- Independent regeneration (scratch regen.py; the library's load_inputs + build_artifact + serialize IN MEMORY, nothing written, `build` subcommand NOT run), from the
  HEAD export at a DIFFERENT absolute path than the build's checkout, real /home/filip/rltraffic/output, committed git+code blocks injected:
  PYTHONHASHSEED=0 -> regenerated 1aa247ed...bb97 == committed, EQUAL True; PYTHONHASHSEED=987654321 -> 1aa247ed...bb97, EQUAL True;
  and the export's own assert_code_under_root block == the committed provenance.code: True. So no field depends on the checkout path (other than the injected
  code_root), on str-hash iteration order, or on the process.
- Fields that WOULD differ between two builds from the same commit and inputs: only git.code_root (checkout location). A post-merge rebuild on main would also differ in
  git.commit (the merge commit) -- expected; not a determinism defect.
- NOTE N7.1 -- T-canonical (:1385-1390) builds twice in ONE process, so it cannot see hash-seed-dependent ordering; T-regress (fresh process, random hash seed per run)
  is what covers cross-process determinism. My two fixed-seed runs confirm it holds.
- NOTE N7.2 -- `dirty` uses `--untracked-files=no` (see M6.1/N6.2): untracked files never make the tree dirty.

### [x] 8. LIVING-DOCUMENT CITATIONS (A4.2) -- PASS (no finding)
- Module: `grep -o -E '(PROJECT_PLAN|README|DEFERRED|CLAUDE|BRIEF_45)...(:|line|§N:)[0-9]+'` -> none. Every `<file>.md:<N>` in the module points into docs/plans/p5.2.md (14)
  or docs/returns/P5.2.md (1). All 48 `Quote(` file arguments are PREREGISTRATION.md (3), BRIEF_44 (1), docs/plans/p5.1.md (1), docs/plans/p5.2.md (40), _BRIEF_27 (1),
  _PACKET (2); all 123 `RecordRow(` use _BRIEF_27 (63), _PACKET (53), _PLAN (7) (constants :1893-1895). Every one is in CITABLE_FILES (:98-105).
  BRIEF_45 is cited only by section / amendment id (e.g. :390, :410, :516, :591, :716, :763, :1113), never by line.
- Artifact (scratch cites.py on HEAD:docs/data/p5_2_statements.json): living-doc line citations: [] ; file:line citations only into docs/returns/P5.2.md (56),
  BRIEF_27 (64), docs/plans/p5.2.md (46); quote {file, lines} objects only into p5.2.md, PREREGISTRATION.md, BRIEF_44, p5.1.md (plus provenance's manifest
  entries, whose "lines" is a line COUNT, not a citation). Living docs appear only as "PROJECT_PLAN section 8 (dated 2026-10-10)", "PROJECT_PLAN section 10 under its
  dated heading", "DEFERRED 106"/"DEFERRED 108" (item ids) and "README note's ..." -- section/dated-heading form, as A4.2 requires.
- Tests: no read of PROJECT_PLAN / README.md / DEFERRED.md / CLAUDE.md / BRIEF_45 (grep for read_/open/Path); T-quotes asserts quote and record files are in CITABLE_FILES
  (tests/test_tier_statements.py:846, :1520, :1528, :1531, :1548).
- Out of A4.2's stated scope and already acknowledged in A4.2's own "Why": docs/plans/p5.2c.md:63 (V16) and :425 cite "PROJECT_PLAN §7, line 1345" / "§7:1345".

## Not verified (M1a)
- Did not run `build` (hard rule); the build path's write ordering is established by reading main() and by calling assert_out_path / write_artifact / the in-memory
  build_artifact directly. Did not run the pytest suite (other mandates; several tests invoke build on synthetic trees). Did not prove that no executed path reaches
  agent/ or utils/ code (N6.2).

## VERDICT
PASS-WITH-NOTES -- BLOCKER 0, MAJOR 0, MINOR 2 (M5.1 write_artifact deletes a pre-existing temp it did not create; M6.1 git block can name an enclosing repo's commit
when --repo-root is an untracked nested copy), NOTE 8 (item 2: 1; item 3: 1; N5.1, N5.2, N6.1, N6.2, N7.1, N7.2).
