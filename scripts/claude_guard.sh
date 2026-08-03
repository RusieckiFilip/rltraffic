#!/usr/bin/env bash
# PostToolUse hook for Claude Code.
#
# Three jobs, split into modes so each can run at the right frequency:
#   --frozen-only : fail loudly if any FROZEN file was modified.   (cheap; runs after Bash too)
#   --tests-only  : run the fast test suite if offline/ or tests/ changed. (expensive; file edits only)
#   --lang-only   : enforce the English-only rule on changed files. (cheap; not wired into settings.json)
#   (no flag)     : all three, in that order. Useful for running it by hand.
#
# Why the split: the frozen-file check must also fire after Bash, because a heredoc
# (`cat > envs/x.py << EOF`) is not an Edit/Write tool call and would otherwise go unnoticed until the
# next file edit. But running pytest after every `ls` would make the session unusable — and the test
# gate cannot early-exit on "nothing changed", since during a task there are always changed files.
#
# Detection is derived from `git status`, so it works regardless of which tool made the edit.
#
# Exit 2 = feed the message back to Claude as an error it must react to.
# Exit 0 = silent success.
#
# Install: referenced from .claude/settings.json -> hooks.PostToolUse
# Test it: bash scripts/claude_guard.sh ; echo "exit=$?"

set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 0

MODE="${1:-all}"

# `experiments/` is only partly frozen: the harness code is, but the JSON configs under
# experiments/configs/ are not — new runs need new configs. This must stay in sync with the deny
# list in .claude/settings.json; if the two disagree, the hook wins at the worst moment.
#
# G1 FIX (2026-08-03): the experiments/ clause is a PREFIX, not `experiments/.*\.py$`. `git status
# --porcelain` collapses a wholly-untracked directory into a single entry (e.g. `experiments/newpkg/`),
# so an extension-anchored pattern never saw a new .py in a new subdirectory and let it through
# (docs/notes/D3_falsification.md section 4). A prefix survives the collapse, exactly like every other
# directory entry here. Do NOT "tidy" it back to an extension anchor. Safe because runtime artifacts
# resolve to output/experiments/ (experiments/config.py), never under experiments/ itself; the one
# writable subtree, experiments/configs/, is carved out in FROZEN_EXCEPTIONS below.
#
# `utils/` (top level) is frozen because all four backend env files import it — it is the roadnet
# vocabulary (`RoadnetInfo`, `IntersectionInfo`) the backends agree on, not a scratch helper package.
# `scripts/` and `.claude/` are frozen because this guard, the hook wiring and the permission
# deny-list must not be editable by the session they constrain.
FROZEN_PATTERNS='^(envs/|agent/base\.py|agent/utils/utils\.py|agent/MAPPOAgent\.py|algorithms/|rewards\.py|states/|metrics/|utils/|scripts/|\.claude/|CityFlow/|experiments/)'

# DELIBERATE TEMPORARY EXCEPTION (2026-07-26) — not an oversight, and not a hole to widen.
# check_english.sh is new and still being tuned (see the TODO in its header about `ó` tripping on
# Spanish/Portuguese surnames in the bibliography), so it stays writable while the rest of scripts/
# is frozen. Delete this variable and its use below once the script settles.
#
# SECOND DELIBERATE EXCEPTION (2026-08-01, P2.0 review round 2) -- authorised by the Master chat:
# "scripts/check_test_hygiene.sh belongs in FROZEN_EXCEPTIONS explicitly. An undocumented tolerance
# is the thing the frozen list exists to prevent." It is a new check still gaining rules, so it stays
# writable while it settles. Delete this clause once the rule set is stable.
#
# THIRD EXCEPTION (2026-08-03, G1 fix): top-level JSON under experiments/configs/ is writable — new
# runs need new configs (CLAUDE.md rule 1). Now that the experiments/ clause is a prefix it would
# otherwise block configs, so the carve-out is spelled out here, visible, rather than being implicit in
# an extension anchor. The exception is `experiments/configs/[^/]*\.json$`, NOT a bare prefix, and that
# is deliberate on two counts:
#   * `[^/]*\.json$` exempts only JSON, so a heredoc write of a .py into experiments/configs/ is still
#     BLOCKED — the bare-prefix form would have reopened G1's exact hole one directory over.
#   * `[^/]*` forbids a slash, so a NEW subdirectory under configs/ (which git status collapses to
#     `experiments/configs/sub/`) fails closed and is BLOCKED. Nested config trees are not a thing today;
#     if they are ever wanted, that should cost a deliberate patch, not leak through this carve-out.
# The `$` anchor binds only to the scripts/ alternative and to the JSON tail; do not loosen either.
FROZEN_EXCEPTIONS='^(scripts/(check_english|check_test_hygiene)\.sh$|experiments/configs/[^/]*\.json$)'

CHANGED="$(git status --porcelain 2>/dev/null | awk '{ $1=""; sub(/^ +/,""); print }')"
[ -z "$CHANGED" ] && exit 0

# ---------------------------------------------------------------- frozen files
if [ "$MODE" = "--frozen-only" ] || [ "$MODE" = "all" ]; then
  VIOLATIONS="$(printf '%s\n' "$CHANGED" | grep -E "$FROZEN_PATTERNS" | grep -vE "$FROZEN_EXCEPTIONS" || true)"
  if [ -n "$VIOLATIONS" ]; then
    echo "BLOCKED: frozen files were modified. This project forbids it (see CLAUDE.md rule 1):" >&2
    printf '%s\n' "$VIOLATIONS" >&2
    echo "" >&2
    echo "Do not run \`git checkout\` on this automatically — confirm with the user whether this is" >&2
    echo "their in-progress work or something this session needs to revert." >&2
    echo "Record the need for the change as an open question in the Return Packet." >&2
    exit 2
  fi
fi

# ---------------------------------------------------------------------- tests
if [ "$MODE" = "--tests-only" ] || [ "$MODE" = "all" ]; then
  # Test hygiene first: a sub-second grep, and a test that cannot fail is worth catching before
  # spending 30 s proving the suite green. Added after the P2.0 review found `assert <real check> or
  # True` committed to tests/test_flow_randomizer.py -- an assertion neutralised rather than
  # reported, in the file whose job is to prove the published corpus is not duplicated. CLAUDE.md
  # rule 0 already forbade it and was followed twice in that task and missed once; a rule that
  # depends on remembering is weaker than a grep.
  #
  # Scoped to CHANGED files for the same reason as the language check below: the P0/P1 suite carries
  # 27 pre-existing `pytest.raises` without `match=` (counted 2026-08-01), and failing on all of them
  # would block every tool call until an unrelated cleanup happened.
  #
  # KNOWN LIMITATION: $CHANGED strips the git status column, so a RENAMED test file arrives as
  # "old -> new" and escapes the ^tests/ filter. Rare; parse `--porcelain -z` if that changes.
  if [ -f scripts/check_test_hygiene.sh ]; then
    mapfile -t CHANGED_TESTS < <(printf '%s\n' "$CHANGED" | grep -E '^tests/.*\.py$' || true)
    if [ "${#CHANGED_TESTS[@]}" -gt 0 ] && [ -n "${CHANGED_TESTS[0]}" ]; then
      if ! bash scripts/check_test_hygiene.sh "${CHANGED_TESTS[@]}"; then
        exit 2
      fi
    fi
  fi

  if printf '%s\n' "$CHANGED" | grep -qE '^(offline/|tests/)'; then
    if command -v pytest >/dev/null 2>&1; then
      # `pipefail` is set, so $? after the assignment reflects pytest, not tail.
      OUT="$(pytest tests -q -x --no-header 2>&1 | tail -n 15)"
      STATUS=$?
      if [ $STATUS -ne 0 ]; then
        echo "Tests are failing after this edit:" >&2
        printf '%s\n' "$OUT" >&2
        exit 2
      fi
    fi
  fi
fi

# ------------------------------------------------------------------- language
# Scoped to CHANGED files on purpose. Running it repo-wide would mean one pre-existing violation
# anywhere blocks every subsequent tool call, including calls that touched nothing — the same
# false-positive trap the frozen check falls into when the working tree is already dirty.
if [ "$MODE" = "--lang-only" ] || [ "$MODE" = "all" ]; then
  if [ -x scripts/check_english.sh ] || [ -f scripts/check_english.sh ]; then
    mapfile -t CHANGED_ARR < <(printf '%s\n' "$CHANGED")
    if [ "${#CHANGED_ARR[@]}" -gt 0 ]; then
      if ! bash scripts/check_english.sh "${CHANGED_ARR[@]}" >&2; then
        exit 2
      fi
    fi
  fi
fi

exit 0
