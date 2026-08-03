# D3 — on-repo falsification of the permission/guard layers

**Date:** 2026-08-03 · **Run by:** Master chat, in the repo, against the **real** `.claude/settings.json`
after the human applied `docs/patches/settings_scripts_glob_deny.patch` and restarted the session.

This closes the item PROJECT_PLAN §10 recorded as owed: *"Post-application, the Master chat owes one
on-repo falsification run: attempt an edit to a new file under `scripts/` and confirm it is denied
against the real settings, not a simulated rule set."*

The prior evidence for D3 (Decisions Log 2026-08-03) was collected in an isolated `/tmp` workspace via
`claude -p --settings`. That establishes what the permission engine does with a given rule set; it does
**not** establish that this repo is running that rule set. This note closes exactly that gap.

## 0. Pre-state (so the observable is a change, not an assertion)

```
.claude/settings.json  deny array: 17 entries, includes Edit(scripts/**), zero Write(...) rules
scripts/               exactly 3 files
  b72684f3…  check_english.sh
  2fcff16e…  check_test_hygiene.sh
  b65df881…  claude_guard.sh
```

The on-disk `settings.json` was verified to be **exactly** `HEAD + the committed patch` and nothing
else: the patch was applied to a pristine copy of `git show HEAD:.claude/settings.json` in a scratch
directory and the result compared by hash.

```
HEAD+patch : c37aece4c27fa5f9910ea85c7d27249bc08cb50a5f7cdca940be66be678a75cd
worktree   : c37aece4c27fa5f9910ea85c7d27249bc08cb50a5f7cdca940be66be678a75cd   IDENTICAL
```

This matters: without it, "the patch is applied" would rest on the human's report of having applied it.

## 1. Permission layer — probes against the real settings

Observable in every case is **file contents on disk after the attempt**, never the tool's self-report.

| # | Probe | Tool | Prediction | Result | Evidence on disk |
|---|---|---|---|---|---|
| F1 | `scripts/zz_perm_probe.sh` — a **new** file, no individual rule | Write | DENIED | **DENIED** | file absent (`find` across the repo) |
| F2 | `scripts/check_english.sh` — a `FROZEN_EXCEPTIONS` file | Edit | DENIED | **DENIED** | sha256 unchanged; `git status scripts/` clean |
| F3 | `docs/notes/_f3_control.tmp` — no deny rule (**control**) | Write | ALLOWED | **ALLOWED** | file created, then deleted |

F1 is the load-bearing one. It is the only evidence that the **glob** works prospectively — that a
script which did not exist when the rule was written is covered by it. F3 is what makes F1 and F2
interpretable: without a control in the same session and the same permission state, "denied" is
indistinguishable from "the Write tool was broken".

F2 confirms on-repo the layer disagreement that CLAUDE.md rule 1 asserts: the two tunable scripts are
**denied at permission level** and **permitted at guard level**. Both halves are now observed here,
not inherited from the `/tmp` experiment.

## 2. Guard layer — end-to-end, not by re-reading its regexes

Run in a scratch git repo with a byte-identical copy of `scripts/claude_guard.sh`
(sha256 `b65df881…`, compared against the repo's copy in the same command), a real `git status`, and a
clean-tree assertion before every probe.

| Path made dirty | Prediction | Result |
|---|---|---|
| `scripts/check_english.sh` | PERMITTED | **PERMITTED** |
| `scripts/check_test_hygiene.sh` | PERMITTED | **PERMITTED** |
| `scripts/brand_new.sh` | BLOCKED | **BLOCKED** |
| `scripts/claude_guard.sh.bak` | BLOCKED | **BLOCKED** |
| `.claude/settings.json` | BLOCKED | **BLOCKED** |
| `notes/free.txt` (control) | PERMITTED | **PERMITTED** |

**Verdict: D3 holds on-repo, at both layers.** No change to the ruling.

## 3. Two harness errors of my own, and why they are recorded

Both were caught by the control row, which is the argument for always carrying one.

1. **First guard harness: every probe returned BLOCKED, including the control.** Cause: the copied
   `claude_guard.sh` was itself untracked in the scratch repo, so it dirtied `git status` and tripped
   the frozen check on every run regardless of the probe file. Had I not carried a control I would
   have read that table as "the guard blocks everything, excellent" and shipped a false confirmation.
2. **Second harness: cleanup used `git checkout -- .`, which does not remove untracked files.** A
   leftover `experiments/newfile.py` from probe 2 blocked probes 3–5, whose results I initially read
   as confirmations. Fixed with `git clean -fdq` plus an explicit clean-tree assertion before each
   probe — after which probe 3 reversed its verdict and exposed the defect in §4.

A confirmation produced by a broken harness looks exactly like a real one. The only thing that
distinguished them here was a row whose expected answer was the opposite of the others.

## 4. DEFECT FOUND — `G1`: new `.py` files in a new subdirectory of `experiments/` escape the guard

Discovered while probing, not by inspection. `git status --porcelain` **collapses a wholly-untracked
directory into a single entry**, so the guard never sees the file path:

```
path created                    what git status REPORTS         guard verdict
experiments/runner.py           [experiments/runner.py]         BLOCKED     correct
experiments/newfile.py          [experiments/newfile.py]        BLOCKED     correct
experiments/newpkg/foo.py       [experiments/newpkg/]           PERMITTED   <-- DEFECT
experiments/newpkg/sub/f.py     [experiments/newpkg/]           PERMITTED   <-- DEFECT
envs/newpkg/foo.py              [envs/newpkg/]                  BLOCKED     correct
experiments/configs/new.json    [experiments/configs/]          PERMITTED   correct (intended)
```

**Root cause.** `FROZEN_PATTERNS` protects `experiments/` as `experiments/.*\.py$` — the only frozen
pattern anchored on a *file extension*. Every other entry is a directory prefix (`^envs/`, `^scripts/`,
`^\.claude/`, …), so the collapsed form `envs/newpkg/` still matches and still blocks. Only the
`experiments/` clause is defeated by the collapse.

**Reachability, stated honestly.**
- Via the **Edit/Write tools**: not reachable. `Edit(experiments/**/*.py)` is denied at permission
  level, and `Edit(path)` governs the Write tool (Decisions Log 2026-08-03, fact 3).
- Via **Bash** (`cat > experiments/newpkg/foo.py << EOF`, `cp`, `tee`): **reachable**. That route is
  precisely why the guard exists at all — its own header says the frozen check must fire after Bash
  "because a heredoc is not an Edit/Write tool call and would otherwise go unnoticed".
- Not currently exploited: no such directory exists, and nothing in the plan calls for one.

So the exposure is prospective, exactly like the one D3 itself was written to close — and it is in the
half of the defence that the permission layer cannot cover.

**Proposed fix (not applied — `scripts/` is frozen, and this needs a patch + human application).**
Invert the `experiments/` clause from extension-anchored to prefix-anchored, and move the deliberate
carve-out into the exceptions list where it is visible:

```
FROZEN_PATTERNS   … |experiments/)          # prefix, like every other entry
FROZEN_EXCEPTIONS ^(scripts/(check_english|check_test_hygiene)\.sh|experiments/configs/)
```

Checked before proposing: runtime artifacts do **not** land under `experiments/` —
`experiments/config.py:355` resolves `output_dir` to `output/experiments/<name>`, so a prefix rule
cannot misfire on results, plots or summaries. `experiments/configs/` stays writable, which is the
documented intent (new configs are explicitly permitted; editing a config already used for a recorded
run is what remains forbidden, and that is a human rule, not a mechanical one).

**Not fixed in this session, deliberately.** It is off the paper's critical path and unexploited;
P0.4's value decays with time and came first. Scheduled as **P0.8**.
