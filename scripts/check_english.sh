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
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 0

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
EXCLUDES=(
  ':!CityFlow/'
  ':!*/node_modules/*'
  ':!*/third_party/*'
  ':!*/vendor/*'
  # This file documents the Polish character class it detects, so it will always contain the letters
  # it is checking for — without this line a repo-wide sweep can never return 0.
  ':!scripts/check_english.sh'
)

if [ "$#" -gt 0 ]; then
  # Explicit paths: keep only the ones git tracks and that survive the exclude filter.
  mapfile -d '' FILES < <(git ls-files -z -- "$@" "${EXCLUDES[@]}")
else
  mapfile -d '' FILES < <(git ls-files -z -- . "${EXCLUDES[@]}")
fi

[ "${#FILES[@]}" -eq 0 ] && exit 0

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
