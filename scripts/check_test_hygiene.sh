#!/usr/bin/env bash
# Mechanical hygiene check over tests/ -- rejects assertions that cannot fail.
#
# WHY THIS EXISTS (P2.0 review, 2026-07-31)
# A review found `assert <real check> or True` committed in tests/test_flow_randomizer.py.
# `X or True` can never fail; the assertion had been neutralised rather than reported,
# in the very file whose job is to prove the published corpus is not duplicated. CLAUDE.md
# §0 already forbids exactly that ("Never edit a test to make it pass ... stop and say
# so"), and the rule worked twice in that task and failed once. A rule that depends on
# remembering is weaker than a grep, so this is the grep.
#
# Usage:
#   bash scripts/check_test_hygiene.sh                 # every tests/**.py
#   bash scripts/check_test_hygiene.sh FILE [FILE...]  # only these (non-tests paths ignored)
#
# Exit 0 = clean. Exit 1 = violations printed to stderr.
#
# WAIVERS. A line may opt out with a comment naming the rule and a reason, either on the
# offending line or on the line directly above it:
#     with pytest.raises(SystemExit):  # hygiene: allow TH006 - argparse exits with a code
# The reason is mandatory (>= 10 characters after the rule id). A waiver you cannot
# justify in a clause is a waiver you should not be taking.
#
# SCOPE. claude_guard.sh runs this over CHANGED files only, deliberately, for the same
# reason the language check is scoped that way: 27 pre-existing `pytest.raises` without
# `match=` live in the P0/P1 suite (counted 2026-07-31), and failing the hook on all of
# them would block every edit in the repo until someone did an unrelated cleanup. Run with no arguments
# to audit the whole suite when you actually want that number.

set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 0

if [ "$#" -gt 0 ]; then
  FILES=()
  REPO="$(pwd)"
  for path in "$@"; do
    # Normalise an absolute path back to repo-relative first. Silently dropping it and
    # exiting 0 would report "clean" on a file that was never opened -- a false clean in
    # a blocking hook is worse than no hook.
    rel="${path#"$REPO"/}"
    case "$rel" in
      tests/*.py) [ -f "$rel" ] && FILES+=("$rel") ;;
      *.py)
        echo "check_test_hygiene: ignoring $path (only tests/**.py is checked)" >&2 ;;
    esac
  done
else
  mapfile -t FILES < <(find tests -name '*.py' -type f 2>/dev/null | sort)
fi
[ "${#FILES[@]}" -eq 0 ] && exit 0

FOUND=0

# Report a violation unless a waiver for this rule, carrying a real reason, sits on the
# offending line or the one immediately above it (long `with pytest.raises(...)` lines
# read better with the justification on its own line).
# $1 rule id, $2 file, $3 line number, $4 line text, $5 explanation, $6 previous line
report() {
  local rule="$1" file="$2" lineno="$3" text="$4" why="$5" prev="${6:-}"
  local waiver="#[[:space:]]*hygiene:[[:space:]]*allow[[:space:]]+$rule[[:space:]]*[-:][[:space:]]*.{10,}"
  if printf '%s' "$text" | grep -qE "$waiver"; then
    return
  fi
  if printf '%s' "$prev" | grep -qE "$waiver"; then
    return
  fi
  echo "$file:$lineno: [$rule] $why" >&2
  echo "    ${text#"${text%%[![:space:]]*}"}" >&2
  FOUND=1
}

# Line-oriented rules. Comment lines are skipped so this file's own examples, and any
# commented-out code, do not trip it.
scan_line_rules() {
  local file="$1"
  local lineno=0 line stripped prev=""
  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    stripped="${line#"${line%%[![:space:]]*}"}"
    case "$stripped" in \#*) prev="$line"; continue ;; esac

    # TH001 -- a truthy disjunct makes the whole assertion unfailable. Matched both on a
    # single-line assert and on a bare continuation line, because the historical offending
    # line was 106 chars and any reformat wraps it into `\n    or True\n)`.
    if printf '%s' "$line" | grep -qE '^[[:space:]]*assert\b.*\bor[[:space:]]+(True|1)\b' \
       || printf '%s' "$line" | grep -qE '^[[:space:]]*or[[:space:]]+(True|1)[[:space:]]*(\)|:|$)'; then
      report TH001 "$file" "$lineno" "$line" \
        "'or True' / 'or 1' makes this assertion impossible to fail" "$prev"
    fi

    # TH002 -- a constant assertion asserts nothing.
    if printf '%s' "$line" | grep -qE '^[[:space:]]*assert[[:space:]]+(True|1|not[[:space:]]+(False|0))[[:space:]]*(,|#|$)'; then
      report TH002 "$file" "$lineno" "$line" \
        "constant assertion: this can never fail" "$prev"
    fi

    # TH002b -- a conditional expression whose taken branch is constant.
    # `assert X if False else True` reduces to `assert True`, and `assert True if C else True`
    # is constant on both branches.  Written after exactly this line reached a delivered test
    # file (P5.2 fix round, self-disclosed by the implementer) and was NOT caught by TH002,
    # which only matches a bare constant operand.
    if printf '%s' "$line" | grep -qE '^[[:space:]]*assert[[:space:]].*[[:space:]]if[[:space:]]+(False|0)[[:space:]]+else[[:space:]]+(True|1)[[:space:]]*(,|#|$)|^[[:space:]]*assert[[:space:]]+(True|1)[[:space:]]+if[[:space:]].*[[:space:]]else[[:space:]]+(True|1)[[:space:]]*(,|#|$)'; then
      report TH002 "$file" "$lineno" "$line" \
        "conditional constant: the taken branch is constant, so this can never fail" "$prev"
    fi

    # TH003 -- dead code disguised as a branch.
    if printf '%s' "$line" | grep -qE '^[[:space:]]*if[[:space:]]+(False|0)[[:space:]]*:'; then
      report TH003 "$file" "$lineno" "$line" \
        "'if False:' -- the body never runs; delete it or fix the condition" "$prev"
    fi

    # TH004 (one-line form) -- `except ValueError: pass` on a single line; the multi-line
    # form is handled by the awk pass below.
    if printf '%s' "$line" | grep -qE '^[[:space:]]*except\b.*:[[:space:]]*pass[[:space:]]*(#|$)'; then
      report TH004 "$file" "$lineno" "$line" \
        "bare 'pass' in an except block swallows the failure" "$prev"
    fi

    # TH005 -- an unexplained skip hides why coverage is missing.
    if printf '%s' "$line" | grep -qE 'pytest\.skip\([[:space:]]*(\)|""[[:space:]]*\)|'"''"'[[:space:]]*\))'; then
      report TH005 "$file" "$lineno" "$line" \
        "pytest.skip() without a reason: state why the test cannot run" "$prev"
    fi

    # TH006 -- pytest.raises with no match= accepts any error of that class, including
    # one raised for an entirely different reason than the test claims to check.
    # A `pytest.raises(` left open at end of line continues onto following lines, where
    # match= may legitimately live; flagging it would be a false positive, and a false
    # positive in a blocking hook gets the whole hook disabled.
    if printf '%s' "$line" | grep -qE 'pytest\.raises\(' \
       && printf '%s' "$line" | grep -qE 'pytest\.raises\([^)]*\)' \
       && ! printf '%s' "$line" | grep -qE 'match[[:space:]]*='; then
      report TH006 "$file" "$lineno" "$line" \
        "pytest.raises without match=: any error of that class satisfies it" "$prev"
    fi

    prev="$line"
  done < "$file"
}

# TH004 -- `except ...: pass` swallows the failure the test was meant to observe.
# Needs two lines of context, so it is a separate awk pass.
scan_except_pass() {
  local file="$1"
  awk -v FNAME="$file" '
    # No \b here: in awk that is a backspace, not a word boundary.
    /^[[:space:]]*except[[:space:]:(]/ { pending = 1; pline = NR; ptext = $0; prev = prevline; prevline = $0; next }
    pending == 1 {
      if ($0 ~ /^[[:space:]]*pass[[:space:]]*(#|$)/ && $0 !~ /hygiene:[[:space:]]*allow[[:space:]]+TH004[[:space:]]*[-:][[:space:]]*.{10,}/ && prev !~ /hygiene:[[:space:]]*allow[[:space:]]+TH004[[:space:]]*[-:][[:space:]]*.{10,}/) {
        printf "%s:%d: [TH004] bare `pass` in an except block swallows the failure\n", FNAME, NR > "/dev/stderr"
        printf "    %s\n", ptext > "/dev/stderr"
        hits++
      }
      pending = 0
    }
    { prevline = $0 }
    # Report every occurrence, not just the first: one exit-on-first-hit pass understates
    # the count and hides later violations until the earlier one is fixed.
    END { if (hits) exit 3 }
  ' "$file"
  [ $? -eq 3 ] && FOUND=1
  return 0
}

for file in "${FILES[@]}"; do
  scan_line_rules "$file"
  scan_except_pass "$file"
done

if [ "$FOUND" -ne 0 ]; then
  echo "" >&2
  echo "Test-hygiene check failed (scripts/check_test_hygiene.sh)." >&2
  echo "These patterns make a test unable to fail, which is worse than no test:" >&2
  echo "a green suite then certifies nothing. Fix the test, or -- if the property" >&2
  echo "genuinely does not hold -- delete the assertion and say so in the Return" >&2
  echo "Packet rather than neutralising it (CLAUDE.md section 0)." >&2
  echo "" >&2
  echo "To waive one line, append: # hygiene: allow TH00N - <reason, 10+ chars>" >&2
  exit 1
fi

exit 0
