"""Run committed mutants in the throwaway worktree: apply, commit, run the named tests, record, reset."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MUT = Path("/home/filip/rltraffic-p52b-mut")
PY = "/home/filip/rltraffic/.venv/bin/python"
S = Path("/tmp/claude-1000/-home-filip-rltraffic-p52b/b8df8e4b-1a97-4037-a120-8bdcf37bc8b4/scratchpad")
BASE = sys.argv[1]
specs = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
GATES = {"RLTRAFFIC_OUTPUT_ROOT": "/home/filip/rltraffic/output", "RLTRAFFIC_CORPUS_V11": "/home/filip/rltraffic/datasets_v11",
         "RLTRAFFIC_DRAWS": "/home/filip/rltraffic/scenarios/draws"}
CLOSED = {"RLTRAFFIC_OUTPUT_ROOT": str(S / "empty_output"), "RLTRAFFIC_CORPUS_V11": str(S / "empty_corpus"),
          "RLTRAFFIC_DRAWS": str(S / "empty_output")}

def git(*args):
    return subprocess.run(["git", "-C", str(MUT), *args], capture_output=True, text=True, check=True).stdout.strip()

for spec in specs:
    git("checkout", "-q", "--detach", BASE)
    for file, old, new in spec["edits"]:
        path = MUT / file
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{spec['id']}: {old!r} occurs {count} times in {file}")
        path.write_text(text.replace(old, new), encoding="utf-8")
    git("add", *sorted({file for file, _, _ in spec["edits"]}))
    git("commit", "-q", "-m", f"mutant {spec['id']}: {spec['what']}")
    sha = git("rev-parse", "--short", "HEAD")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **(GATES if spec.get("gates") else CLOSED)}
    started = time.time()
    proc = subprocess.run([PY, "-m", "pytest", *spec["tests"], "-q", "-p", "no:cacheprovider", "-x"], cwd=MUT, env=env,
                          capture_output=True, text=True, timeout=3000)
    lines = proc.stdout.splitlines()
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), lines[-1] if lines else "")
    first_e = next((l.strip() for l in lines if l.startswith("E ")), "")
    failed = next((l for l in lines if l.startswith("FAILED ") or l.startswith("ERROR ")), "")
    record = {"id": spec["id"], "sha": sha, "what": spec["what"], "mode": "gates open" if spec.get("gates") else "gates closed",
              "result": "KILLED" if proc.returncode != 0 else "SURVIVED", "summary": summary.strip(),
              "failed": failed[:200], "reason": first_e[:260], "seconds": round(time.time() - started, 1)}
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"{record['id']:24s} {record['sha']} {record['result']:8s} {record['summary'][:60]} | {record['reason'][:140]}", flush=True)
git("checkout", "-q", "--detach", BASE)
