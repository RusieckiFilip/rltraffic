"""C8's mutation runner (BRIEF_44 Amendment B): each mutant applied as exact-string edits, COMMITTED in the throwaway
detached worktree, the named tests run, the outcome recorded as one JSON line; the worktree is reset to BASE after each.

usage: run_mutants_c8.py <worktree> <base-sha> <specs.json> <results.jsonl> <closed-gates-dir>
A spec: {"id", "what", "edits": [[file, old, new], ...], "tests": [pytest args...], "gates": bool}.  With "gates" the
data gates are open (the main tree's output, corpus and draws); without, they point at empty directories.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MUT, BASE, SPECS, OUT, CLOSED = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
PY = "/home/filip/rltraffic/.venv/bin/python"
OPEN = {"RLTRAFFIC_OUTPUT_ROOT": "/home/filip/rltraffic/output", "RLTRAFFIC_CORPUS_V11": "/home/filip/rltraffic/datasets_v11",
        "RLTRAFFIC_DRAWS": "/home/filip/rltraffic/scenarios/draws"}
SHUT = {"RLTRAFFIC_OUTPUT_ROOT": str(CLOSED), "RLTRAFFIC_CORPUS_V11": str(CLOSED), "RLTRAFFIC_DRAWS": str(CLOSED)}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(MUT), *args], capture_output=True, text=True, check=True).stdout.strip()


for spec in json.loads(SPECS.read_text(encoding="utf-8")):
    git("checkout", "-q", "--detach", BASE)
    for file, old, new in spec["edits"]:
        path = MUT / file
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise SystemExit(f"{spec['id']}: the edit's old text occurs {text.count(old)} times in {file}, not once")
        path.write_text(text.replace(old, new), encoding="utf-8")
    git("add", *sorted({file for file, _, _ in spec["edits"]}))
    git("commit", "-q", "-m", f"mutant {spec['id']}: {spec['what']}")
    sha = git("rev-parse", "--short", "HEAD")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **(OPEN if spec.get("gates") else SHUT)}
    started = time.time()
    proc = subprocess.run([PY, "-m", "pytest", *spec["tests"], "-q", "-p", "no:cacheprovider", "-x"], cwd=MUT, env=env,
                          capture_output=True, text=True, timeout=3000)
    lines = proc.stdout.splitlines()
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), lines[-1] if lines else "")
    reason = next((l.strip() for l in lines if l.startswith("E ")), "")
    failed = next((l for l in lines if l.startswith("FAILED ") or l.startswith("ERROR ")), "")
    record = {"id": spec["id"], "sha": sha, "base": BASE[:7], "what": spec["what"],
              "mode": "gates open" if spec.get("gates") else "gates closed",
              "result": "KILLED" if proc.returncode != 0 else "SURVIVED", "summary": summary.strip(),
              "failed": failed[:220], "reason": reason[:300], "seconds": round(time.time() - started, 1)}
    with OUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"{record['id']:28s} {record['sha']} {record['result']:8s} {record['summary'][:48]} | {record['reason'][:120]}",
          flush=True)
git("checkout", "-q", "--detach", BASE)
