#!/usr/bin/env bash
# Enforce CLAUDE.md §3 "Language": every artifact on disk is written in English.
#
# Detects Polish diacritics in tracked text files. Proper nouns are exempt (see ALLOWED_NAMES) —
# a person's name keeps its spelling, stripping the diacritics would be a corruption, not a fix.
#
# Usage:
#   bash scripts/check_english.sh                 # all tracked files
#   bash scripts/check_english.sh a.md b.py       # only these paths (used by the hook)
#
# Exit 0 = clean.  Exit 1 = at least one violation; the offending file:line pairs go to stderr.
#
# WHY grep -P WITH EXPLICIT CODEPOINTS, and not the obvious `git grep -E '[ąćęłńóśźż]'`:
# git's -E engine treats a multibyte bracket expression as a set of raw BYTES, so any UTF-8 character
# sharing a continuation byte with one of the Polish letters matches. In practice `→` (E2 86 92) is
# reported because of the 86 byte in `Ć` (C4 86); `×`, `–` and `—` match the same way. A sweep of this
# repo with the naive query returned 23 files, of which 22 were pure false positives from arrows and
# en-dashes in prose. `grep -P` with \x{...} codepoints matches characters, not bytes.

set -uo pipefail
# Being outside a repository is NOT "clean": this used to `exit 0`, so an invocation from any other
# directory passed silently. Absolute paths still work with no repository; relative ones are then
# reported as missing below.
TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$TOPLEVEL" ]; then cd "$TOPLEVEL" || exit 1; fi

# TODO known false positive: ó/Ó is not Polish-specific (Spanish/Portuguese names in bibliography
# entries will trip this) — extend ALLOWED_NAMES below when the related-work matrix starts hitting it.
# Ą ą Ć ć Ę ę Ł ł Ń ń Ó ó Ś ś Ź ź Ż ż
PL_CLASS='[\x{0104}\x{0105}\x{0106}\x{0107}\x{0118}\x{0119}\x{0141}\x{0142}\x{0143}\x{0144}\x{00D3}\x{00F3}\x{015A}\x{015B}\x{0179}\x{017A}\x{017B}\x{017C}]'

# Proper nouns permitted to keep their diacritics. Extend this list rather than de-diacriticising a
# name. Kept deliberately narrow: full surnames/given names, not fragments.
ALLOWED_NAMES='Paweł|Woliński|Grudziński|Mikołaj'

# Vendored trees are excluded wholesale: not our prose, not ours to rewrite.
# (`grep -I` handles binaries; this list is for text we still do not own.)
# Plus one self-exclusion, for a different reason than the vendored trees above.
# Shell-glob twins of EXCLUDES, used by the explicit-path branch. Kept adjacent so the two lists are
# edited together: the pathspec form is git-only and cannot filter a path git does not track.
EXCLUDE_GLOBS=(
  'CityFlow/*'
  '*/node_modules/*'
  '*/third_party/*'
  '*/vendor/*'
  'scripts/check_english.sh'
)

EXCLUDES=(
  ':!CityFlow/'
  ':!*/node_modules/*'
  ':!*/third_party/*'
  ':!*/vendor/*'
  # This file documents the Polish character class it detects, so it will always contain the letters
  # it is checking for — without this line a repo-wide sweep can never return 0.
  ':!scripts/check_english.sh'
)

FILES=()
if [ "$#" -gt 0 ]; then
  # Explicit paths are taken AS GIVEN and never filtered through `git ls-files`.
  #
  # WHY, measured 2026-09-15: `git ls-files -z -- <path> "${EXCLUDES[@]}"` returns NOTHING for a path
  # git does not already track -- every new file, before `git add` -- and also for a tracked path under
  # a hidden directory such as `.claude/`. Combined with the unconditional `exit 0` that used to follow,
  # this script reported CLEAN on files it never opened: an untracked file full of Polish exited 0, and
  # so did `.claude/agents/master-coordinator.md`, which the no-argument sweep flags at line 158.
  # New files are exactly what this repository produces, and the explicit-path mode is the one the
  # PostToolUse hook and every session use.
  #
  # PROJECT_PLAN section 7: "nothing to check" and "everything is clean" may never share an exit code.
  MISSING=""
  EXCLUDED=0
  for arg in "$@"; do
    # An absolute path inside the repository becomes repo-relative, so the globs below still apply.
    case "$arg" in
      "$PWD"/*) arg="${arg#"$PWD"/}" ;;
    esac
    if [ ! -e "$arg" ]; then
      MISSING+="  $arg"$'\n'
      continue
    fi
    SKIP=0
    for pattern in "${EXCLUDE_GLOBS[@]}"; do
      # shellcheck disable=SC2254 -- the pattern is a glob on purpose.
      case "$arg" in
        $pattern) SKIP=1; break ;;
      esac
    done
    if [ "$SKIP" -eq 1 ]; then EXCLUDED=$((EXCLUDED + 1)); else FILES+=("$arg"); fi
  done

  if [ -n "$MISSING" ]; then
    echo "BLOCKED: asked to check paths that do not exist, so nothing was checked:" >&2
    printf '%s' "$MISSING" >&2
    exit 1
  fi

  if [ "${#FILES[@]}" -eq 0 ]; then
    # A deliberate exclusion, unlike a missing path, is a decision recorded in EXCLUDE_GLOBS above, so
    # this exits 0 -- but LOUDLY. What this change is about is the silence, not the exit code.
    echo "NOTE: all $EXCLUDED given path(s) are excluded by EXCLUDE_GLOBS; nothing was checked." >&2
    exit 0
  fi
else
  mapfile -d '' FILES < <(git ls-files -z -- . "${EXCLUDES[@]}")
  [ "${#FILES[@]}" -eq 0 ] && exit 0
fi

VIOLATIONS=""
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue                       # deleted-but-tracked
  grep -qI . "$f" 2>/dev/null || continue       # binary or empty -> skip
  # Blank out allowed proper nouns first, then look for anything Polish that is left.
  HIT="$(sed -E "s/(${ALLOWED_NAMES})//g" "$f" 2>/dev/null | grep -nP "$PL_CLASS" || true)"
  if [ -n "$HIT" ]; then
    while IFS= read -r line; do
      VIOLATIONS+="${f}:${line%%:*}"$'\n'
    done <<< "$HIT"
  fi
done

if [ -n "$VIOLATIONS" ]; then
  echo "BLOCKED: non-English text found. CLAUDE.md §3 requires every on-disk artifact in English:" >&2
  printf '%s' "$VIOLATIONS" >&2
  echo "" >&2
  echo "Translate the prose. If a hit is a proper noun, add it to ALLOWED_NAMES in" >&2
  echo "scripts/check_english.sh instead of removing the diacritics from someone's name." >&2
  exit 1
fi

exit 0
