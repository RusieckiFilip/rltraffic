"""G1 coordinator mutants (BRIEF_43 gate G1). Each mutant: apply each change exactly once, commit 'mutant <id>' in the
throwaway detached worktree, run the three new test files with the data gates open, record KILLED/SURVIVED with the
summary line and the first failing tests, append to the findings file, reset --hard to the base."""
import datetime, json, os, subprocess, sys
from pathlib import Path
WT = "/home/filip/rltraffic-p82-mut"
BASE = "5c8a33a"
S = Path("/tmp/claude-1000/-home-filip-rltraffic/7546ab2c-e248-4f99-bf16-9f19f47a6a55/scratchpad")
F = S / "G11_COORDINATOR_MUTANTS.md"
PY = "/home/filip/rltraffic/.venv/bin/python"
def sh(cmd, env=None):
    return subprocess.run(cmd, cwd=WT, shell=True, capture_output=True, text=True, env=env)
spec = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
for m in spec:
    if only and m["id"] not in only:
        continue
    assert sh("git rev-parse --short=7 HEAD").stdout.strip() == BASE, "not at base"
    assert sh("git status --porcelain --untracked-files=no").stdout.strip() == "", "dirty before mutant"
    for c in m["changes"]:
        p = os.path.join(WT, c["file"])
        s = open(p, encoding="utf-8").read()
        n = s.count(c["old"])
        assert n == 1, (m["id"], c["file"], n)
        open(p, "w", encoding="utf-8").write(s.replace(c["old"], c["new"]))
    r = sh(f'git commit -qam "mutant {m["id"]}"')
    assert r.returncode == 0, r.stderr
    sha = sh("git rev-parse --short HEAD").stdout.strip()
    bt = S / "g1mut" / f"pt_{m['id']}"
    bt.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=WT,
               RLTRAFFIC_OUTPUT_ROOT="/home/filip/rltraffic/output",
               RLTRAFFIC_CORPUS_V11="/home/filip/rltraffic/datasets_v11",
               RLTRAFFIC_DRAWS="/home/filip/rltraffic/scenarios/draws")
    t = sh(f"{PY} -P -m pytest {m['tests']} -q -p no:cacheprovider --basetemp={bt} -rfEs 2>&1", env=env)
    out = t.stdout.strip().splitlines()
    summary = out[-1] if out else "(no output)"
    failed = [l for l in out if l.startswith("FAILED") or l.startswith("ERROR")]
    skipped = [l for l in out if l.startswith("SKIPPED")]
    verdict = "KILLED" if t.returncode != 0 and failed else ("SURVIVED" if t.returncode == 0 else f"ODD rc={t.returncode}")
    msgs = [l.strip()[:220] for l in out if l.strip().startswith("E ")][:3]
    line = (f"- {m['id']} [{datetime.datetime.now():%H:%M:%S}] mutant commit {sha}: {m['name']} -> **{verdict}**; "
            f"`{summary}`; failing: {[f[:160] for f in failed[:3]]}; skipped: {len(skipped)}; first E-lines: {msgs}")
    print(line, flush=True)
    with open(F, "a", encoding="utf-8") as h:
        h.write(line + "\n")
    sh(f"git reset -q --hard {BASE}")
