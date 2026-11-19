# Review D -- FINDINGS (P9.4 pre-flight: driver, write/delete paths, resume, smoke)
Reviewer: independent (Claude, Opus 5.5), started 2026-11-19. HEAD under review: 512c101 (to be confirmed).

## Checklist
- [x] 0. Setup: confirm HEAD, scratch copy via git archive, `offline` imports from the copy.
- [x] 1a. write_once: refuses over an existing file? symlink target? '..'/relative path? atomicity (tmp+rename)?
- [x] 1b. write_declaration: where does it write; can it overwrite; pinned digest check.
- [x] 1c. train_run: .partial checkpoint and the final link/rename -- can it overwrite or write outside the roots?
- [x] 1d. SUMO chunk writes (p9_4_evaluation 330-560): path construction, overwrite, outside-root.
- [x] 1e. Route (ii) writes via att_rederivation.run_campaign: where they land (root argument), overwrite.
- [x] 1f. Report writes (p9_4_report): destination, overwrite.
- [x] 1g. Driver's rm of the token, logs, any other rm/mv/cp/ln in the .sh.
- [x] 1h. (partly; driver roots fixed absolute -- see 3) Output-root validation: symlinked output root, '..' in --output-root, relative root, root == main output.
- [x] 2a. Resume: truncated chunk / stale chunk / wrong-cell chunk / other-code chunk -> refuse or skip?
- [x] 2b. Resume: .partial checkpoint present.
- [x] 2c. Resume: checkpoint without record; record without checkpoint.
- [x] 2d. Resume: control record that failed; guard record failed; guard record from other code.
- [x] 2e. Resume: finished work is never re-run (skip logic proven on constructed state).
- [x] 3a. Driver: `bash -n`.
- [x] 3b. Driver: every refusal before the token (bad args exit 2 before anything); run refusal paths.
- [x] 3c. Driver: token consumed once per session; STOP never created/removed by the driver.
- [x] 3d. Driver: WORKERS capped at 6; nothing that must stop the run is piped (pipefail / PIPESTATUS).
- [x] 3e. Driver: run tree clean, detached, at the commit; module loads from the run tree; second driver refused (lock).
- [x] 4a. Smoke: writes only in its scratch root?
- [x] 4b. Smoke: removing its scratch root cannot touch the real output/.
- [x] 5. Run the driver/CLI tests (tests/test_p9_4_driver.py, tests/test_p9_4_cli.py) and record the real result.
- [x] 6. Tests: which are load-bearing vs theatre for the write/resume claims.

## Findings log (append-only)

### [x] 0. Setup (2026-11-19)
- `git rev-parse HEAD` in /home/filip/rltraffic-p94 -> 512c10158b6f8a4b1d1ad17e64c19dd5534f1530 (matches the request).
- `git status --short` lists 11 untracked dotfiles (.bashrc, .gitconfig, .idea, ...): `ls -la` shows each is a char device 1,3
  (/dev/null) owned by nobody -- sandbox bind-mount artifacts, not repository content. Not a finding.
- Scratch copy: `git archive HEAD | tar -x -C D/copy`; from the copy `import offline` resolves to
  D/copy/offline/__init__.py and offline.p9_4_mix50_fresh to D/copy/offline/p9_4_mix50_fresh.py. OK.

### [x] 1a/1b/1h. write_once, write_declaration, assert_target (offline/p9_4_mix50_fresh.py:168-214)
Construction: D/exp/e1a_write_once.py, run from the copy against a synthetic root D/e1a/output (never the main tree).
- a) target exists -> FileExistsError from os.link; existing bytes still "ORIGINAL". PASS (:201).
- b) `output/p9_4/../p7_3a/x.json` -> PermissionError (resolution first). PASS (:173-179).
- c) relative `p9_4/x.json` -> resolved against the CWD (the copy) -> PermissionError. PASS.
- d) symlink inside p9_4 -> outside (dangling, and to an existing file) -> PermissionError; outside untouched. PASS.
- e) **output/p9_4 ITSELF a symlink to a sibling result dir (output2/p9_4 -> output2/p7_3a)**: write_once WROTE
  output2/p7_3a/new.json; overwrite of the existing p7_3a/result.json still refused (FileExistsError).
  Cause: :174 resolves the allowed roots too, so a symlinked root makes its target "allowed". The docstring's claim
  "a symlink is judged by where it LANDS" does not hold for the root itself. Precondition NOT present today:
  `ls -la` of the main output shows output/p9_4 is a real directory (holds declaration.json, Nov 19 11:20) and
  output/p9_4_runs does not exist (the driver's usage line creates it with mkdir -p). -> NON-BLOCKING finding N1
  (add-only, never overwrite; needs a planted symlink).
- f) NaN payload -> ValueError raised inside the `with` at :196-199, BEFORE the try/finally at :200-203, so the empty
  temporary `.nan.json.<pid>.tmp` is LEFT in output/p9_4. Same for any kill (stall guard _exit, SIGKILL) between :196
  and :203. -> NON-BLOCKING N2 (litter only; check whether any scan trips on dot-tmp files: see 2a).
- g) a leftover `.g.json.<pid>.tmp` with the current pid -> FileExistsError, nothing written, leftover untouched
  (liveness only, needs pid reuse). PASS for safety.
- h) path == the root itself -> temporary lands in output/ -> PermissionError before any byte. PASS.
- i) write_declaration over different content -> FileExistsError; identical -> returns without writing. PASS (:207-214).

### [x] 1c. train_run writes (offline/p9_4_mix50_fresh.py:871-942; offline/dt_gate.py:865-870) -- by reading, falsified in 2b/2c
- final, final+".partial", record: each assert_target'ed and checked absent before the first step (:888-891).
  torch.save writes the .partial with "wb" (dt_gate.py:870) -- safe only because :890 proved it absent; os.link(partial,
  final) never overwrites (:909); partial.unlink (:910); record by write_once (:940). No path outside the barrier.
- Crash windows -> on-disk states: during train_dt -> .partial only; between :909 and :910 -> .partial AND final (same
  inode); between :910 and :940 -> final without record. All three are REFUSED by run_status (:954-959) -- see 2b/2c.

### [x] 1d. SUMO chunk writes (offline/p9_4_evaluation.py:481-492) -- by reading
- _sumo_worker writes ONLY via p94.write_once (barrier + exclusive link). transfer_curve.run_cell's docstring and body
  (transfer_curve.py:2843-2990) open nothing for writing. pending() (:494-512) reads, never moves/deletes.

### [x] 1e. Route (ii) writes via att_rederivation.run_campaign -- by reading
- P9.4 passes `protected=()` (p9_4_evaluation.py:322), so tier_sweep.assert_writable checks NOTHING (tier_sweep.py:472-489);
  the only barrier is p94.assert_target(work) on the directory (p9_4_evaluation.py:314). Destinations are
  `work / cell_file_name(cell)` / `work / refusal_file_name(cell)` with work = output/p9_4_runs/eval_ii -> inside.
- write_json_guarded -> dt_gate.write_json_atomic (dt_gate.py write_json_atomic): mkstemp + os.replace, NO fsync,
  and os.replace OVERWRITES an existing destination. So a refusal file of a cell that is refused again in a later
  session is OVERWRITTEN (att_rederivation.py:1159-1160), and the header claim "Nothing under output/ is ... overwritten"
  (driver :30) is not literally true for eval_ii. Inside output/p9_4_runs/, so not BLOCKING by COMMON's rule. -> N4.
- Resume skip is `destination.is_file()` with NO content check (att_rederivation.py run_campaign, ~1236 `if
  destination.is_file(): skipped += 1`). Safe against crashes only because the write is replace-atomic; see 2a for
  what campaign_status does with a bad cell file.

### [x] 1g. Driver rm/mv/cp/ln (offline/campaigns/p9_4_mix50_fresh.sh) -- by reading
- The ONLY rm is :107 `rm -- "$TOKEN"` after `[ -f "$TOKEN" ]` (:106). A symlinked token: -f follows, rm removes the
  link, not its target. No mv/cp/ln/rm -r anywhere. STOP is only tested (-e) at :93, :185, :197 -- never created/removed.
- Log writes with TRUNCATION: :131 `>"$LOGS/$label.log"`, :174 shard logs, :191 `>"$LOGS/status.json"` (written TWICE in
  one session on the complete_or_end -> session_end path, :205 then :198), stall_guard opens the dump with "w"
  (stall_guard.py:67). All under output/p9_4_runs/logs/<UTC second>/; a collision needs two drivers in the same second.
  -> N5 (claim at driver :30 "nothing ... overwritten" is false for the session's own status.json; harmless).
- --stall-dump is opened "w" by stall_guard.install (p9_4_mix50_fresh.py:1347-1350) WITHOUT assert_target: the module's
  CLI will truncate any path given as --stall-dump. The driver only passes $LOGS/<label>.stall<n>.txt. -> N3.

### [x] 2a (hypothesis, see below). Route (ii) refusal persistence -- by reading, to be falsified next
- att_rederivation.run_campaign (offline/att_rederivation.py:1225-1345): `refused` is LOCAL to one call; a refused cell
  writes refused_<cell>.json and NO cell file, so a later call RETRIES it; n_refused counts only the current call.
- P9.4's stop (offline/p9_4_evaluation.py:328-333) raises only when THIS call refused. campaign_status
  (att_rederivation.py:1011-1060) computes complete = "no declared cell without a cell file" and IGNORES refused_*.json
  (it only counts them). cityflow_status (p9_4_evaluation.py:337-350) -> complete = route_i_done and that.
- check(--stage cityflow) (p9_4_mix50_fresh.py:1251-1259) and run_route_ii's pre-checks (p9_4_evaluation.py:303-315)
  never look at refused_*.json.
- Hypothesis H-ref: a disagreement that does not recur on retry is erased (a) in a NEW session, and (b) WITHIN one
  session with no human action: refusal -> later the stall guard _exit(1)s the same shard -> `restarting` (driver
  :142-168) restarts it -> the retry agrees -> the shard exits 0 -> complete_or_end prints COMPLETE.

### [x] 2a. Route (ii) resume under a leftover refusal and a garbage cell file -- FALSIFIED (experiment)
Construction: D/exp/e2a_route_ii.py, synthetic root D/e2a/output: a complete route (i) record (shape accepted by
assert_route_i_complete), declare_route_ii's real manifest, all 500 cell files, PLUS refused_<cell0>.json (an earlier
call's disagreement) and a ZERO-BYTE cell file for cell 1 (crash with no fsync; write_json_atomic does not fsync).
Observed: `cityflow_status complete: True | route_ii present/missing/refused: 500 0 1`.
- So the driver's complete_or_end (driver :203-213, reading status.json["cityflow"]["complete"]) would print
  "COMPLETE" although a disagreement is on disk and one cell holds no episode. run_campaign would also SKIP the
  zero-byte file forever (`destination.is_file()`, att_rederivation.py:1236).
- Backstop holds: the report's gate (offline/p9_4_report.py:518-520 `if not status["complete"] or status["refused"]`)
  evaluates True on this state (refuses), and the report's cell reader corrective_report._rerolled raises
  JSONDecodeError on the zero-byte file. No wrong number can reach the artifacts.
- H-ref confirmed by construction + reading: the stage-level stop "any disagreement stopping the stage" (driver :22,
  p9_4_evaluation.py:296-298) is NOT persistent -- a new session, or a stall-guard restart of the same shard inside
  ONE session (driver :142-168 restarts on exit 1 + banner; the RuntimeError at p9_4_evaluation.py:328-333 is raised
  only after run_campaign returns), retries the refused cell and can end the stage "COMPLETE".
  -> NON-BLOCKING N6 (would be BLOCKING without the report's refused-gate; the gate is the only thing that holds).
  Suggest: run_route_ii / check(--stage cityflow) refuse while any refused_*.json exists, and cityflow_status's
  "complete" require refused == 0.

### [x] 3a. `bash -n offline/campaigns/p9_4_mix50_fresh.sh` -> exit 0.

### [x] 3b (part 1). Driver argument refusals, run from the scratch copy (D/copy, NOT a git repository)
Observed exits: bad stage 2; WORKERS=7/0/abc/' 6' -> 2 ("WORKERS must be 1-6"); P94_SESSION_HOURS=0/-1/1e3 -> 2;
P94_STALL_SECONDS=10 -> 2; P94_MAX_RESTARTS=-1 -> 2; PYTHONDEVMODE='' (set, empty) -> 2. All before any git/python.
- Missing arguments exit 1, not 2: `${1:?usage...}` (driver :64-65) aborts the shell with bash's own status 1.
  Harmless (nothing has run) but the mandate's "they exit 2" does not hold for these two. -> N7 (cosmetic).
- **Vacuous run-tree checks in a NON-git tree**: with a dummy process whose cmdline matches the pgrep pattern
  (`exec -a "/home/filip/rltraffic/.venv/bin/python -P -m offline.p9_4_mix50_fresh ... route-ii --shard 0" sleep 40`),
  `bash D/copy/offline/campaigns/p9_4_mix50_fresh.sh cityflow 512c101` printed git's "fatal: not a git repository"
  and then `REFUSED: another P9.4 stage is running` (exit 2) -- i.e. it PASSED all three run-tree checks (:82 clean,
  :83-84 at commit, :85-87 detached) and was stopped only by pgrep (:88). Cause: :82 tests the stdout of a FAILING
  `git status` (empty => "clean"); :83 compares two EMPTY strings (rev-parse HEAD fails, the right side is `|| true`)
  => equal; :85 symbolic-ref fails => "not on a branch". A tree without .git (cp -r copy, tar extract, a worktree whose
  gitdir moved) runs as if it were the reviewed commit. -> finding F1. Severity judged below (3e).
- pgrep (:88) also works as a second-driver refusal: the dummy, PID 3 in the sandbox's PID namespace, was matched.

### [x] 3e (part 1). Run-tree refusals in a real git tree (scratch clone D/clone, `git clone --no-hardlinks` of the
worktree, detached at 512c10158b6f), with a dummy pgrep-matching process so nothing past :88 runs
- clean, detached, at 512c101 -> passes :82-87, refused at :88 (pgrep). wrong commit 512c101~1 -> REFUSED "is not at";
  garbage `deadbeef` -> REFUSED; untracked file -> REFUSED; modified tracked file -> REFUSED; on a branch at the commit
  -> REFUSED "is on a branch". All exit 2. PASS.
- A branch NAME as <commit> (`task/p9.4-mix50-fresh`) is accepted (:83 rev-parses it); the session line then says
  "at task/p9.4-mix50-fresh" (driver :111). -> part of N8 (see report's commit parsing, not checked further).
- IGNORED files are invisible to :82 (`git status --porcelain` omits them): `offline/__pycache__/junk.pyc` and an
  `offline/p9_4_evaluation.so` (which CPython's FileFinder would import BEFORE the .py) passed the clean check.
  The LOADED check (:91-92) covers only offline/p9_4_mix50_fresh.py. Requires someone to plant a file in a fresh
  `git worktree add` tree. -> N9 (use `--ignored` or `git clean -ndX` to refuse; low likelihood).
- F1 severity: the documented procedure (`git worktree add --detach`) never produces a non-git tree, but the check
  FAILS OPEN on any git error (non-git dir, moved gitdir, "dubious ownership"), and dt_gate.runtime_provenance records
  git_commit "" and continues when git is unavailable, so training proceeds and the session log still claims
  "at <commit>". By COMMON's rule (a stopping control bypassed) -> BLOCKING B1, one-line fix
  (e.g. `porcelain=$(git -C "$WORK_TREE" status --porcelain) || refuse ...` and `head=$(git rev-parse HEAD) || refuse`).
- F1 backstops found (reading): the TRAIN stage is still stopped -- after the token -- by the module's STRICT git check
  (context_sweep.enter_registered_regime -> _code_provenance raises on any git failure, context_sweep.py:294-312), and
  every SUMO cell by transfer_curve._git_provenance (strict, transfer_curve.py:870-903) -> the stage stops at the first
  batch. CityFlow route (i)/(ii) use dt_gate.runtime_provenance, which is LENIENT (git_commit "" on failure), so the
  cityflow stage is the one that would run unverified code. B1 stays BLOCKING by COMMON's rule (fail-open control)
  but is reachable only by an operator error the documented `git worktree add --detach` does not make.

### [x] 3b (part 2) / 3c. Every refusal before the token, with REAL tokens present
Construction: scratch clone D/clone + one scratch commit cabb700 changing ONLY driver :40 MAIN= to D/fakemain
(fakemain: .venv/bin/python = exec wrapper of the real venv python; datasets_v11 and scenarios/draws = symlinks to the
main tree, read-only; output/ = a real scratch dir; P4.7's five mix50 checkpoints + SHA256SUMS_p4_7.txt COPIED in).
- T1 TOKEN_cityflow + STOP -> exit 2 "STOP exists" (:93); afterwards `ls` shows STOP and TOKEN_cityflow both intact.
- T2 TOKEN_cityflow, no control.json -> module check ValueError "control.json is absent" -> exit 2 (:102); token intact.
- T3 control.json {passed:false, n_seeds_equal:4} + TOKEN_train -> check(train) re-hashed the 200 files, read the 5
  references, then ControlFailed (CLI exit 3) -> driver exit 2; token intact. T3b same record + TOKEN_cityflow ->
  "the control did not pass (4 of 5)" -> exit 2; token intact.
- T4 forged control {passed:true, 5 of 5, right format} + TOKEN_sumo, no trained runs -> "mix50_fresh_dt_seed101 is not
  trained" -> exit 2; token intact.
- T5 control seed 101 `.partial` + TOKEN_train -> "an interrupted run; refusing to retrain over it or skip it" -> exit
  2; token intact; the .partial's bytes unchanged ("partial").
- STOP: never created or removed by the driver (reading: only `[ -e ]` at :93/:185/:197) -- T1 confirms it survives.
- Token consumed once: the only `rm` is :107, after the last refusal (:106); a second driver racing for the same token
  dies at `rm` (set -e, no -f). Everything after :107 that can refuse (train_session's own checks, route/sumo
  refusals) costs the token but writes nothing outside the roots. PASS.

### [x] 2a-SUMO. SUMO resume under bad chunks -- FALSIFIED with REAL chunks (experiment)
Construction: D/exp/e2s_sumo_pending.py + e2s_b.py, run from D/clone; synthetic root D/e2s/output with P4.7's five mix50
checkpoints COPIED in; chunks COPIED from the main tree's output/p7_3a (read only), e.g.
cell_mix50_b_mean_k100_seed101_draw1000.json (P7.3a, git_commit 8bf7eae77322, stage "confirmatory").
- (1) P7.3a chunk at G1 cell's own name -> pending() ValueError "may not be reused (invalid: ... the chunk says
  stage='confirmatory' and the cell ... says 'p9_4_g1')"; bytes unchanged (sha256 equal).
- (1b) same payload with stage relabelled to p9_4_g1 -> chunk_problems = ["rolled by other code (['.claude/agents/
  implementer.md', ...])"] -- the J1(c) code check fires; demand and checkpoint digests matched.
- (2) g1[0]'s chunk under g1[1]'s name -> "invalid ..." refused; untouched.
- (3) first 200 bytes only -> "exists and cannot be read (Unterminated string ...)" refused; untouched.
- (4) a registered-mix50 chunk at a mix50_fresh NEW cell's name -> "invalid" refused; untouched.
- (5) a directory at the chunk's name -> "cannot be read ([Errno 21] Is a directory)" refused.
All PASS: each stops the stage, nothing moved/overwritten (offline/p9_4_evaluation.py:494-512, 442-478).
- sumo_status (:672-680) counts `exists()` only (no validation) and does not look at guards.json's "passed"; harmless
  because sumo_session never rolls a new cell without a passing record and pending()/the report re-validate.

### [x] 2b/2c/2d/2e. Training resume states -- FALSIFIED (experiment D/exp/e2t_train_resume.py, root D/e2t/output)
Checkpoint bytes = a COPY of P4.7's mix50_dt_seed101.pt; records written with the real write_once.
- control seed 101 checkpoint + matching record -> run_status "done"; train_plan(hours=100) plans the other 9 in order
  [control 202..505, then hz1x1 101..505], done 1 missing 9 -> finished work is skipped, never re-run. PASS (:1093-1118).
- checkpoint without record -> REFUSED "must both exist"; train_plan refuses too. record without checkpoint -> REFUSED.
- record of another budget (20 steps) -> REFUSED. checkpoint changed after its record (one byte appended) -> REFUSED
  "digests are not the ones its record wrote". final + .partial (crash between :909 and :910) -> REFUSED (.partial).
- train_run over an existing final -> FileExistsError before train_dt is reached (:888-891). PASS.
- control.json {passed:false} -> train_plan refuses; ensure_control_record returns the existing failed record (so
  train_session raises ControlFailed, :1158-1160). PASS.
- control.json from other code ({format "p9.3-control/1.0", passed:true}) -> train_plan plans 9 runs and check(train)
  would pass (it only tests `passed`, :1246-1248), but assert_control_passed refuses it before the first new-tier run
  (:1172-1173). PASS (no new-tier run). Residual: a RIGHT-format control.json carries no code provenance (no
  git_commit/git_dirty, unlike guards.json at p9_4_evaluation.py:622) and its seeds' candidate_file_sha256 are never
  re-checked against the five control checkpoints when it authorises -> N10 (needs a hand-made or stale record).

### [x] 2d-guards. guards.json states (experiment D/exp/e2g_guards.py, run from D/clone, root D/e2g/output)
- passed + git_commit = clone HEAD + git_dirty false -> guards() AUTHORISED, rolled 0 (no G1/G2 re-roll on resume).
- passed:false -> GuardFailed "GUARDS: G1 FAILED" (CLI exit 3 -> driver :132-134 "a stopping control failed").
- passed, git_commit 8bf7eae77322 (other code) -> ValueError "written by other or uncommitted code".
- passed, git_dirty true -> ValueError. wrong format_version -> ValueError. All PASS (p9_4_evaluation.py:614-641).
- Residual (same class as N10): a record that says passed and names this commit is trusted without re-deriving
  g1/g2 from the chunks on disk; only a hand-made record could exploit it.

### [x] 4a/4b. The smoke (offline/p9_4_smoke.py whole; smoke_roots p9_4_mix50_fresh.py:1262-1270)
Construction: D/exp/e4_smoke_roots.py (calls smoke_roots/assert_target/copy_references only; nothing rolled or trained).
- --scratch under the main tree (`/home/filip/rltraffic/output/x`), via `..` (`/home/filip/rltraffic/../rltraffic/output`),
  via a symlink to the main output, or inside the repo -> all REFUSED (ValueError). PASS.
- `/` and `/home/filip` are ALLOWED (output_root /output, /home/filip/output): not the real output, harmless. N11 (cosmetic).
- **Planted symlink**: D/e4/planted/output/p9_4 -> /home/filip/rltraffic/output/p9_4. smoke_roots ALLOWS the scratch
  and assert_target ALLOWS /home/filip/rltraffic/output/p9_4/probe.json (only assert_target was called; `ls -la`
  of the real output/p9_4 afterwards: still only declaration.json). Same root cause as N1: the guard resolves the
  scratch path but never checks that output/, output/p9_4 and output/p9_4_runs beneath it are real directories.
  If p9_4_runs were the planted link, the smoke's deliberately FAILED control record (module docstring :10-12) would be
  written ONCE into the real output/p9_4_runs/control.json and block the real run until moved by hand. Needs a planted
  link -> NON-BLOCKING, folded into N1. The links were removed with `unlink` afterwards.
- Writes: copy_references (:34-47, shutil.copyfile, guarded only by "output_root != the real output" and `exists()`;
  a dangling symlink at a target would be written THROUGH -- unbarriered, needs a planted link; part of N1), then
  train_session/train_run/route_i/route_ii/roll -- all through write_once/assert_target or write_json_atomic on the
  scratch roots. The smoke has NO delete/rmtree/move of its own (grep of p9_4_*.py, att_rederivation, transfer_curve,
  sumo_anchor_rescore, admission_probe, corrective_report: the only link/unlink calls are write_once :201/:203 and
  train_run :909/:910; sumo_anchor_rescore._write_json_atomic and transfer_curve.write_manifest are not on P9.4's paths;
  run_cell and sumo_anchor_episode "write NOTHING").
- 4b: the references are COPIED (new inodes), every hard link the code makes is within one directory of the scratch,
  and no symlink is created anywhere -> `rm -rf <scratch>` cannot reach the real output/. PASS.

### [x] 3e (part 2). The typed <commit> is what the report later reads -- N8 (experiment, from D/copy)
- The driver accepts ANY rev that equals HEAD (:83): `task/p9.4-mix50-fresh` was accepted in 3e part 1 (and `HEAD`
  would be too), and :111 prints the TYPED string, not the resolved sha.
- corrective_report.TOKEN_LINE (corrective_report.py:185-186) needs 7-40 hex: `task/p9.4-mix50-fresh` -> NO MATCH, `HEAD`
  -> NO MATCH (driver_sessions REFUSES such a line, :198-209); `512c101` and the full sha both match, but
  stage_code_commit over a stage whose sessions typed `512c101` once and the full sha once -> ValueError "the sumo
  sessions name the commits ['512c101', '512c10158b6f...']".
- Effect: fail-closed (no wrong number), but a typing inconsistency across sessions is discovered only at REPORT time,
  after the campaign, and the capture files it would take to repair are under output/ (no-edit rule). -> NON-BLOCKING N8.
  Fix: print `$(git -C "$WORK_TREE" rev-parse HEAD)` in the token line, or refuse a <commit> that is not 40 hex.

### [x] 1f. Report writes (offline/p9_4_report.py:804-823) -- by reading
- Not called by the driver. Every refusal (load_sources) precedes every write; each artifact is created with
  open(path, "x") -> never overwrites; an existing differing file -> FileExistsError; identical -> skipped. Destination
  is --out-dir or repo docs/data, with NO barrier (by design: artifacts are committed). Not atomic: a crash mid-write
  leaves a partial file that the next attempt refuses as "other content" (fail-closed). -> N16 (cosmetic).
- CityFlow env (frozen envs/cityflow_env.py:94-125, 293-296): writes its config into its own tempfile.mkdtemp dir and
  rmtree's only that dir; held-out draw configs carry saveReplay false and no log-file keys (draw_1000/cityflow.json
  read), so no replay is written into the draws or the run tree on P9.4's paths. PASS.

### [x] 5. Tests run (real output)
`cd /home/filip/rltraffic-p94 && PYTHONDONTWRITEBYTECODE=1 /home/filip/rltraffic/.venv/bin/python -m pytest -q
 -p no:warnings -p no:cacheprovider tests/test_p9_4_driver.py tests/test_p9_4_cli.py` -> `27 passed in 1.21s`.
No RLTRAFFIC_* variable set (env | grep -c '^RLTRAFFIC_' -> 0). Flags used so the run writes no cache/bytecode in the
worktree (COMMON's write-nothing rule).

### [x] 6. Load-bearing vs theatre (driver + CLI tests; barrier/write_once tests in test_p9_4_training.py read for coverage)
Load-bearing (behavioural, can fail): test_a_bad_stage_worker_count_or_session_hours_is_refused_before_anything_runs
(executes the driver), test_cityflow_and_sumo_refuse_before_the_controls_five_of_five,
test_cityflow_and_sumo_refuse_until_the_new_tiers_five_runs_are_done, test_no_stage_passes_its_check_while_the_stop_file_exists,
test_an_unknown_stage_or_a_bad_worker_count_is_refused_by_the_parser, test_the_status_is_computed_from_disk,
test_the_smoke_root_must_be_outside_the_main_trees_output_and_any_repository, test_no_second_p9_4_stage_runs_beside_a_first
(pattern positives), test_the_driver_never_creates_or_removes_the_stop_file_or_anything_under_output (static but exact).
Weak/static text checks (cannot detect a semantic regression): test_every_refusal_precedes_the_token... (string ORDER only;
it passes with B1's fail-open checks), test_the_token_is_per_stage..., test_the_regime_is_one_thread...,
test_every_cityflow_worker_runs_under_the_stall_restart_loop..., test_the_usage_says_first..., test_the_token_line...
(only a 40-hex example; misses N8). test_check_train_runs_every_refusal_and_writes_nothing mocks the five heavy checks
(wiring only). test_the_train_command_exits_3... mocks train_session (exit-code mapping only).
UNTESTED by any behavioural test: the run-tree git checks (B1), token consumption, `restarting` (banner/exit-code/
MAX_RESTARTS), `shards` failure propagation, `complete_or_end`. Barrier test (test_p9_4_training.py:35-47) covers a
symlink INSIDE p9_4 but not a symlinked p9_4/p9_4_runs (N1); write_once test (:50-58) covers success/exists but not an
exception during the write (N2).

### Other non-blocking notes (reading)
- N12 `shards` (driver :170-182): a failed shard does not stop its siblings; they run to their own end/deadline before
  the stage fails. Compute only; their files are valid route (ii) cells.
- N13 route (i) is one 500-episode unit written once at the end (p9_4_evaluation.py:203-215): a stall-guard restart
  (driver :142-168) re-runs it from episode 1 -- "resumes from the files on disk" (driver :35-36) has nothing to resume
  from for route (i). Deterministic, so no corruption; liveness only.
- N14 SUMO cells run in a multiprocessing Pool with no stall guard (p9_4_evaluation.py:515-548); a worker that dies
  abruptly mid-task would leave imap_unordered waiting and the deadline unchecked. Not verified (no episode run).
- N15 pgrep (:88) is racy across stages: between one driver's :88 and its first matching python (and in the gaps
  route_i -> declare-route-ii, status calls), a driver of ANOTHER stage passes; the token stops only same-stage twins
  (rm without -f). Global worker cap of 6 is per driver.
- Environment note: the sandbox created zero-byte placeholders (.bashrc, .gitconfig, .idea, ... at 11:27:40) in
  /home/filip/rltraffic-p94; if they persist on the host they make this worktree untracked-dirty (the driver would
  refuse it -- fail-safe; the run tree is a separate worktree).

### [x] 3d. WORKERS cap, pipes, and the restart loop -- experiment D/e3d (driver functions extracted verbatim, :54-62
and :127-168 of the copy's driver, sourced with fake workers; MAX_RESTARTS=2)
- WORKERS: driver :67 refuses 7/0/abc/' 6' (3b); module caps `sumo --workers` (_workers, :1290-1296) and roll()
  (p9_4_evaluation.py:523-524) at 6; route-ii `--of` is uncapped in the module but only the driver passes it (=WORKERS).
- Pipes: no stage command is piped; the only pipes are the GPU trace (:116-119) and display-only `tail | sed` AFTER the
  exit code is captured (:133, :136, :178) and `sed -n` over finished logs (:221, :237). PASS.
- restarting/run_logged observed: stall banner twice then success -> 3 invocations, "ok", exit 0; stall banner forever
  -> 3 invocations (1 + MAX_RESTARTS), FAILED exit 1; exit 1 WITHOUT banner (a refusal/traceback) -> 1 invocation, FAILED
  (no restart); exit 3 -> "a stopping control failed" -> exit 1. Each attempt got a fresh dump (stall0..stall2.txt).
  PASS -- matches the header (:32-36).
- shard partition: att_rederivation.shard_cells (att_rederivation.py:960-970) slices the SORTED full key list by index
  mod `of`, independent of disk state -> no overlap within a session; a different WORKERS in a later session still
  covers every cell (present ones are skipped). PASS.

### Additional observation for 3e
- The LOADED check (:91-92) PASSED positively in every clone run T1-T5 (the driver reached :93/:102, which follow it,
  with the module imported from D/clone) and in the non-git copy. Its NEGATIVE case (module importing from elsewhere)
  was not constructed. CityFlow routes have NO strict git check (grep of admission_probe, att_rederivation,
  method_tier_grid: no _code_provenance/_git_provenance/enter_registered_regime/code_changed_since) -- B1's exposure is
  the cityflow stage, and only if training was completed from a valid tree first.

## VERDICT (assembled 2026-11-19): FAIL on one blocking finding (B1, driver :82-87 fail open on a git error, contrary
to the project's own J1(a) rule "could not say must never read as clean"; a few-line fix). Everything else in scope
PASSED its falsification: no write, delete, move or overwrite outside output/p9_4*/ is possible without a planted symlink
(N1), and every resume state refuses or skips as claimed, except that a route (ii) refusal does not persist across a
restart or session (N6), which the report's refused-gate still catches.
- Cleanup: unlinked D/fakemain/datasets_v11 and D/fakemain/scenarios/draws (links into the main tree); no symlink under D/ points outside D/ (3 remain from e1a, all targets inside D/e1a). Main output/p9_4 still holds only declaration.json; output/p9_4_runs still absent.
