"""Apply each mutation of a batch file ONE AT A TIME to its source file, run the named tests, restore byte for byte.

Usage: python mutate_batch.py <batch.py> <results.txt>
The batch module defines TARGET (repo-relative path), TESTS (pytest args) and MUTANTS: a list of (name, old, new); each
old text must occur exactly once.  Each result line is appended to the results file as soon as it is known.
"""
from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/filip/rltraffic-p94")
batch = runpy.run_path(sys.argv[1])
results = Path(sys.argv[2])
target = REPO / batch["TARGET"]
original = target.read_bytes()
digest = hashlib.sha256(original).hexdigest()
text = original.decode("utf-8")
for name, old, new in batch["MUTANTS"]:
    count = text.count(old)
    if count != 1:
        line = f"{name}: INVALID (old text occurs {count} times)"
    else:
        try:
            target.write_text(text.replace(old, new), encoding="utf-8")
            run = subprocess.run(
                ["/home/filip/rltraffic/.venv/bin/python", "-m", "pytest", "-q", "-p", "no:warnings", "-p",
                 "no:cacheprovider", "-x", *batch["TESTS"]],
                capture_output=True, text=True, cwd=str(REPO), env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                                                                    "HOME": "/home/filip",
                                                                    "TMPDIR": "/tmp/claude-1000"},
            )
            tail = [l for l in run.stdout.splitlines() if l.strip()][-1:]
            line = f"{name}: {'KILLED' if run.returncode != 0 else 'SURVIVED'} | {' '.join(tail)}"
        finally:
            target.write_bytes(original)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, "RESTORE FAILED"
    with results.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)
