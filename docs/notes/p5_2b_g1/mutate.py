"""Exact-string mutation helper: replace OLD with NEW in FILE, refusing unless OLD occurs exactly once."""
import sys
from pathlib import Path

path, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text(encoding="utf-8")
count = text.count(old)
if count != 1:
    sys.exit(f"REFUSED: {old!r} occurs {count} times in {path}, not once")
path.write_text(text.replace(old, new), encoding="utf-8")
print(f"mutated {path}: 1 replacement")
